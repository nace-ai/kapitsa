# What a change must show

Kapitsa sells speed and exact outputs, so a PR is done when it shows it is no slower and that its tokens are the
same, or says on purpose that they are not. What that takes depends on what the diff can reach. `vk checks [BASE]`
(`scripts/required_checks.py`) reads the changed paths and prints the tier, the checks and the models. This file
says why.

The repo rules this builds on are AGENTS.md ("What an agent can and cannot check", "Rules that bite here", "Pull
requests") and CONTRIBUTING.md ("Speed changes", "Output check"). Where they say more, they win. The tiers are this
skill's reading of those rules, not repo policy.

## Tiers

The highest tier among the changed files is the PR's tier. A tier includes every check of the tiers below it.

| Tier | Paths (first match wins; `required_checks.py` has the full list) | Why | Required |
| --- | --- | --- | --- |
| 0 | docs, `.github/`, `tests/*.py`, `tools/*.py`, `bench/` scripts, `src/{api,hub,update,bench,util}/`, the CLI-only parts of `src/server/` (`cluster_cmd`, `update_cmd`, `ui`, `picker`, `bench_cmd`, `chat.html`), cluster discovery and placement | cannot change what runs on the GPU or which token comes out | `vk cpu-tests`, `prek run --all-files` |
| 0.5 | `src/tokenizer/`, `src/engine/grammar.*`, the output parser, tool arguments, logit masks, minja | changes tokens on the CPU: same weights, different text in or out | + `tests/test_structured.py --cli build MODEL_DIR`, `build/parser_test MODEL_DIR` on a model folder's tokenizer (no GPU) |
| 1 | `src/engine/`, `src/loader/`, `src/vision/`, `src/gguf/`, the rest of `src/server/` (serve, beam search, history prefill), `src/cluster/` stages and pipes, `src/systemone/`, any path no rule names | host code on the hot path: scheduling, slots, caches, memory plan | + GPU: `tests/test_server.py` against `vk serve`, the logit gate, one interleaved speed A/B |
| 2 | `src/kernels/`, `src/model/`, `src/metal/`, `src/ane/`, `CMakeLists.txt`, `cmake/` | the computation itself | + changed kernels at small sizes under the Metal validation layer |
| 3 | not a path: the PR *means* to move outputs (a precision, a different computation) | the gate fails on purpose | + say so with the numbers, `bench/quality` (`ni_eval` against bf16), and for a default `bench/reasoning` `math500-quick` |

A bug fix in tier 1 or 2 still runs the gate and the A/B. A fix that slows the hot path or moves tokens is a
different change, and the PR has to say which one it is.

## Which models

Run every model whose code the diff reaches. `required_checks.py` names them and the floor file the gate uses on
this Mac.

- Code every model runs (engine, loader, shared kernels, build flags): **Qwen3.5 4B 4-bit** (dense, Neural Engine
  split) and **Qwen3.6 35B-A3B 4-bit** (MoE). They differ in the ways that matter most: the 4B runs the Neural
  Engine split, and the 35B runs experts batched in other groupings. Its decode floor is 500 to 1300 times the dense
  models', so a change that is bitwise on one can drift on the other.
- Model-specific code adds that model: Flint (`flint*`), Dory (`dory*`, NVFP4), Nemotron-H (Lightning, which has no
  floor file, and Dory), Flash-Next Q2_0 (`qwen4_exp*`, `expert_store*`: experts streamed from the SSD, run with
  `NI_EXPERT_STORE=24` from the internal disk), the 35B for GGUF IQ experts, and Qwen3.8 27B for `qwen35*`.
- `src/kernels/tensor/` engages only on an Apple10 GPU with macOS 26 (M5). Off an M5 the PR cannot claim anything
  about it.
- A model with no folder on this Mac is "Not verified" with its name. It is never covered by argument. AGENTS.md
  records where that was done (the QMT6 entry) as a departure, not as precedent.

## Datasets

| Check | Data | Size |
| --- | --- | --- |
| logit gate | 8 windows of `bench/corpus.txt` (prefixes of 512 to 3200 tokens), 24 teacher-forced rows each, paths `step`, `verify`, `batch`, plus the prefill row | 192 rows a path |
| speed A/B | `kapitsa bench --quick`: the published protocol (512-token coding answers, 4K cold read, follow-up TTFT) | 3 runs a test |
| long context (prefill or attention changes) | `logit_dump --windows/--rows` on longer prefixes, or a 64K prompt's first decode rows; #585 was caught only at 64K | as the change needs |
| quality (tier 3) | `bench/quality` held-out chats (`ids_qwen35`, `ids_qwen38`) with `build/ni_eval --ref` | 4 x 1023 positions |
| reasoning (tier 3 default) | `bench/reasoning` `math500-quick` (50 MATH-500 level-5 problems, thinking on) | 50 problems, up to 32768 tokens each |

`bench/` data is frozen. Never regenerate or reformat it to make a check pass.

## The PR description

- **Title**: the change and its measured effect, e.g. "engine: X (35B decode +3%, outputs bitwise)". If there was no
  GPU run: "... (Not verified on a GPU)".
- **Runs:** Mac model, chip and GPU cores, memory, macOS version and build, `kapitsa --version` of both builds, model
  folders. `vk doctor` prints most of it.
- **Speed**: every run of the A/B in order (A B A B), not the best. Medians alone hide the interleaving.
- **Outputs**: `logit_compare.py`'s report per model and path, verbatim. "Every row bitwise identical" is written
  only when the report says so.
- **Checked how**: the commands, with the evidence directory's file names.
- **Not verified**: every required check that did not run, and why (no GPU permission, model not here, lock held).
- One change per PR, under about 1000 changed lines (the `PR size` check). Never merge your own.
