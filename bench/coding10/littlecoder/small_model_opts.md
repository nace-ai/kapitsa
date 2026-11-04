# Small-model optimizations, ranked for 4B/9B Qwen on simple Python function tasks

The ranking is expected impact on pass@1 when the model must write one function and return it. Nothing here was
measured in this port (CPU only, no model loading was allowed). The evidence column cites little-coder's own
measurements where they exist. Context: on Aider Polyglot with the same Qwen3.5-9B, little-coder's scaffold scored
45.56% against 19.11% for vanilla Aider (`docs/benchmark-baseline-aider.md`, README), so the scaffold, not the model,
was most of the difference.

| # | Optimization | Setting to use | Source / evidence |
|---|---|---|---|
| 1 | **Test-feedback retry loop** | After the attempt, run the visible tests (or the model's own asserts). On failure, send ONE retry turn: `The tests failed. Output:\n\n```\n{last 4000 chars}\n```\n\nFix the implementation and try again.` Max 1 retry (`max_retries: 1` for 8B+, 2 for <=4B). | `benchmarks/aider_polyglot.py` L169-177. pass_2 is 17.6% of all passes (84.5 first-try + 18.0 retry, `docs/benchmark-reproduction.md`). |
| 2 | **Bounded thinking, with a forced answer when the cap is hit** | Qwen3 4B/9B: thinking on with a 2048-token budget (v0.0.5 `qwen3:8b`/`qwen3.5`; main uses 4096). When it is exceeded, abort, then append `assistant: Here is my analysis so far:\n{last 2000 chars}` + `user: Good analysis. Now produce the final answer directly — no more deliberation needed.` and regenerate with thinking off (`chat_template_kwargs: {enable_thinking: false}` for vLLM/llama.cpp, `think: false` for Ollama). If no budget mechanism exists, turn thinking OFF: an unbounded think that eats `max_tokens` returns no code at all. | v0.0.5 `providers.py` L584-700; `.pi/extensions/thinking-budget/index.ts`; profiles in v0.0.5 `local/config.py`. |
| 3 | **Self-test before answering, in the tool loop** | Give `run_python` and say "write solution.py, then test it with run_python". Cap the loop at 8 turns for simple functions (little-coder: `max_turns` 40 on TB). Score the file on disk if the final text has no code block. | Polyglot prompt (`_build_prompt`) + bash tool; `.pi/extensions/turn-cap/index.ts`. |
| 4 | **Output format constraint + tolerant extraction** | End the task prompt with "reply with the complete final solution in one ```python block". Extract the LAST fenced block (`tools_impl.extract_final_code`); fall back to `solution.py` on disk; strip `<think>…</think>` first. | Harness glue; little-coder's equivalent is "stop calling tools when correct" and grading the file on disk. |
| 5 | **Sampling params** | `temperature 0.3` (every little-coder Qwen profile; 0.2 for TB, 0.4 for GAIA), `top_p 0.95`, `top_k 20`, `min_p 0`, `repetition_penalty 1.0`, `presence_penalty 0` (raise it to 1.0-1.5 only if loops appear). Do not use greedy decoding with thinking on: Qwen's model card warns it causes endless repetition. | `.pi/settings.json` `little_coder.model_profiles`; top_p/top_k/presence are Qwen3 model-card values, not in little-coder. |
| 6 | **Output length caps** | `max_tokens 4096` per turn (all profiles); hard cap `num_predict 8192` against runaway generation; a 10 s wall clock on code runs. | `.pi/settings.json`; `docs/benchmark-baseline-aider.md` §Runaway generation (3 runaway cases before the cap, 0 after). |
| 7 | **Write guard (write_file refuses existing files → edit_file)** | `run_tool(..., write_guard=True)`; the refusal text carries a ready-to-copy edit_file call. This matters less for one short function than for Polyglot, so turn it off (`write_guard=False`) if 4B models loop on refusals. | `.pi/extensions/write-guard/index.ts`. Fired on 57% of Polyglot exercises; 33% of those still passed after redirect (`docs/benchmark-reproduction.md`). |
| 8 | **Tool-call parsing repair** | Parse ```` ```tool ````/```` ```json ```` fences, `<tool_call>` tags and bare `{"name","arguments"}` objects. Repair JSON in this order: literal newlines in strings, then trailing commas, single quotes, unquoted keys, unclosed braces. Accept `input`/`arguments`/`parameters`/`args`, a string-encoded `arguments`, and case-insensitive keys. If the server rejects `tools` (HTTP 400/500), retry once with text-tool instructions. | `.pi/extensions/output-parser/parser.ts`; v0.0.5 `local/output_parser.py`, `providers.py` L628-645. |
| 9 | **Quality monitor corrections** | Per turn, detect an empty reply, an unknown tool, malformed args, or the same call repeated with no edit in between. Send the canned correction (skills_system.md §7) and allow at most 1-2 corrections in a row. | `.pi/extensions/quality-monitor/quality.ts`. |
| 10 | **Few tools, short skill cards, injected last** | <=4 tools. Tool cards with one example call each (300-token budget) go at the END of the prompt, because small models show strong recency bias. Only re-inject a card after that tool errors. | `.pi/extensions/skill-inject/index.ts`, `skills/tools/*.md`; `ALLOWED_TOOLS` in `aider_polyglot.py`. |
| 11 | **Algorithm knowledge cards** | Keyword-scored (phrase 2.0, word 1.0, keep >=2.0), 200-token budget, `## Algorithm Reference`. These help on DP/search/string-rule tasks and do nothing on trivial ones. | `.pi/extensions/knowledge-inject/index.ts`, `skills/knowledge/*.md`. |
| 12 | **Context trimming** | Context 32K for 9B, 8K for <=4B (v0.0.5 `qwen2.5:3b`/`gemma3:4b`). Truncate tool output to first half + last quarter (`tools_impl.MAX_OUTPUT` 8000; little-coder 32000). Keep the last 4000 chars of test output on retry. Compaction is irrelevant for single-function tasks. | v0.0.5 `tool_registry.py` L92-98; `.pi/extensions/context-watchdog/`. On Ollama, set `num_ctx` explicitly or the prompt is silently truncated (CHANGELOG #128). |
| 13 | **Deliberation on failure** | v0.0.5 `deliberation_mode: "on_failure"`, `n_branches: 2`: after a failed attempt, sample 2 branches and keep a passing one. This is a pass@k cost, so use it only if the evaluation budget allows it. | v0.0.5 `local/deliberate.py`, `local/config.py`. |

## Concrete defaults for the harness

```python
SAMPLING = dict(temperature=0.3, top_p=0.95, top_k=20, min_p=0.0, repetition_penalty=1.0,
                presence_penalty=0.0, max_tokens=4096)
THINKING = dict(enable=True, budget_tokens=2048, on_exceed="abort_and_answer_without_thinking")  # or enable=False if no budget hook
LOOP     = dict(max_turns=8, max_corrections=2, test_retries=1, retry_output_chars=4000, tool_timeout_s=10)
CONTEXT  = dict(ctx=32768 if params_b >= 8 else 8192, tool_output_chars=8000)
```
