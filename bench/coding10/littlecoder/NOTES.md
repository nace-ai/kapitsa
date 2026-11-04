# little-coder port notes (task #350)

1. little-coder (github.com/itayinbarr/little-coder) is a coding agent for small local models. Since v0.1.0 it runs on the pi agent (`@mariozechner/pi-coding-agent`) as 35 TypeScript extensions in `.pi/extensions/`. v0.0.x (tag v0.0.5) was a standalone Python CLI.
2. Paper result: Qwen3.5-9B scored 45.56% on Aider Polyglot, against 19.11% for vanilla Aider on the same model. The gain comes from the scaffold (`docs/benchmark-baseline-aider.md`).
3. Base prompt: v0.0.5 `context.py` SYSTEM_PROMPT_TEMPLATE. Main uses pi's own prompt plus extension injections, which go at the conversation tail (`.pi/extensions/_shared/inject.ts`).
4. Skill cards: `skills/tools/*.md` (frontmatter + REQUIRED/RULES/EXAMPLE/RECOVERY), selected per turn by `skill-inject/index.ts` (error > recency > intent keywords, 300-token budget).
5. Knowledge cards: `skills/knowledge/*.md`, keyword-scored by `knowledge-inject/index.ts` (threshold 2.0, 200-token budget).
6. Write guard: `write-guard/index.ts` refuses Write on an existing file and returns an Edit recipe. It fired on 57% of Polyglot exercises.
7. Output parser: `output-parser/parser.ts` (Python original in v0.0.5 `local/output_parser.py`) repairs JSON and pulls tool calls out of prose, fences and `<tool_call>` tags.
8. Quality monitor: `quality-monitor/quality.ts` catches empty, looping, unknown-tool and malformed-args turns and sends a correction.
9. Thinking budget: `thinking-budget/index.ts` (v0.0.5 `providers.py`) caps thinking, then retries with thinking off and the partial reasoning injected.
10. Per-model profiles: `.pi/settings.json` `little_coder.model_profiles`, read by `benchmark-profiles/index.ts` (temp 0.3, max_tokens 4096, thinking 4096). The v0.0.5 `local/config.py` has the <=4B profiles.
11. Benchmark loop: `benchmarks/aider_polyglot.py` runs one attempt, then the tests, then one retry with the failing output.
12. Ported to `skills_system.md`: base prompt (coding subset), text-tool fallback, the 4 tool cards, the knowledge-card format, the task and retry prompts, and the correction messages, each tagged with its source file.
13. Ported to `tools.json` + `tools_impl.py`: write_file (with the guard), edit_file (exact unique oldText, multi-edit), read_file (numbered lines) and run_python (snippet or file, 10 s, temp copy of the workdir). Also the JSON repair and text tool-call parser, plus final-code extraction.
14. Skipped: bash, glob/grep, web, browser, subagents, plan mode, memory and compaction. A single-file function task does not use them, and with fewer tools a small model makes fewer wrong tool calls.
15. Not measured: no model was run (CPU-only task). The ranking in `small_model_opts.md` is expected impact, backed by little-coder's own numbers where cited.
