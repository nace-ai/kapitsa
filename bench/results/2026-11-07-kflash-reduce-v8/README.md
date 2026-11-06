# KFlash direct packed attention output: preliminary validation

Implementation: `0be92def43f00c7c5f15b889daa71309dce4d5a7`; reference: `2fe7c3fa208c098b4e00b1bae9033172dfafaac7`.

The packed o projection now consumes the attention reducer's output directly in V8 fragment order. The existing ordered softmax reduction, half conversion and group sums are unchanged. Debug and skip paths retain X8. On the packed path this removes the duplicate T-layout output and one permutation dispatch per draft layer: 6 dispatches per ordinary 6-layer draft pass. Logical table traffic falls from a 128 KiB X8/T write plus 64 KiB read and 64 KiB V8 write to one 64 KiB V8 write per layer. These are logical buffer operations, not measured DRAM traffic or peak-memory savings; allocations are unchanged.

## Environments and source identity

- Local Apple M4 Pro, 20-core GPU, 48 GiB RAM, macOS 15.7.1 (24G231), Apple Clang 17. Complete baseline and candidate libraries were configured and built separately in Release. Local actual 4B drafter checks select the existing packed-coefficient profile, with 39 packed static draft projections in the logs.
- Remote Apple M3 Ultra, 96 GiB RAM, macOS 26.4.1 (25E253). The exact ARM64 binary compiled on the local Mac with macOS 14 minimum ran remotely; Metal compiled the embedded sources on the M3. This is not a claim of a remote C++ source build. The M3 stock profile does not select packed draft q/kv/o projections, so remote results are isolated reducer checks, not packed-drafter model results.
- Preliminary GPU binaries predate the source commit and had the base version label. `source.json` records their SHA256 and the exact source-file SHA256; those file bytes match the implementation commit. Original reducer source was extracted from the base and appended under `dfl_attn_reduce_original`, retaining the original arithmetic for comparison. Final build/CPU identity is recorded separately when complete.

## Isolated GPU exactness

Both hosts passed all 42 cases with **MTL_SHADER_VALIDATION=1 and MTL_DEBUG_LAYER=1**. Each case checks the original versus refactored X8 output, original+permutation versus direct V8, every group sum, finite output and guard bytes. Cases span8/16 rows,1/2/4/8/16/32/64 splits, CPU-known and GPU-derived plans including the128-key minimum; empty/inactive split inputs contain poison. Static threadgroup allocation is16 bytes for each reducer. The512-entry inverse layout mapping was exhaustively checked on CPU.

Raw results: `validation-m3.log`, `validation-m4.log`. Isolated timing A/B/A/B/A/B is in `timing-*.log`, all batches retained. The first8-split A arms were slower than later A arms, and the local32-split series drifted. These preliminary runs had the exclusive GPU lock but no per-process contention monitor. They are not controlled whole-model performance evidence.

## Actual 4B drafter outputs and graph timings

Model: actual Qwen3.5-4B 4-bit, group64, with the matching actual v1 DFlash2 checkpoint (6 layers, hidden2560). Fixed token-ID inputs of64 and1536 tokens are included; the 1536 input repeats a tokenized passage to exercise the longer context and is not a free-generation quality benchmark. The old/new check runs prefill and a complete drafting pass for3 anchors in normal and debug modes. At each length, **270 output files  / 35,198,376 bytes compare exactly**, including7 full active draft-logit rows for every anchor, hidden states, candidate IDs/values and chain, plus debug intermediates. Output digests/manifests are retained; raw distributions are omitted.

The timing driver repeats the complete `encode_draft`48 times per command buffer with one fixed input/anchor. Each fresh-process arm has12 samples; medians below use the last 10, and all 12 are retained. It includes the draft model's projections/attention/head, but excludes target verify, generation/acceptance scheduling, HTTP and model load. It is not a tokens-per-second benchmark. All model legs hold `/tmp/kapitsa-gpu.lock`, and the original executed orchestration `graph-run.py` records fresh state directories and half-second per-process GPU samples. That original orchestration contains the original model paths; use the portable driver commands below to reproduce with your own model paths.

| Draft graph run | Median GPU µs/pass | Mean / max foreign GPU |
| --- | ---: | ---: |
| graph-64-time-0-base | 2623.278 | 0.14 / 0.96% |
| graph-64-time-1-candidate | 2567.473 | 0.22 / 1.52% |
| graph-64-time-2-base | 2629.237 | 3.82 / 8.85% |
| graph-64-time-3-candidate | 2560.324 | 0.64 / 2.41% |
| graph-1536-time-0-base | 2975.949 | 0.75 / 6.71% |
| graph-1536-time-1-candidate | 2948.906 | 0.00 / 0.00% |
| graph-1536-time-2-base | 2996.851 | 0.28 / 1.47% |
| graph-1536-time-3-candidate | 3025.239 | 4.16 / 11.34% |

The 64-token medians favor the candidate, but one baseline arm includes foreign-GPU spikes. The 1536candidate's second arm has 4.16% mean/11.34% max foreign GPU activity (Lark/Codex); the series does not establish a stable gain there. The monitor's `flagged:false` uses a sustained-load threshold and does not mean every arm was idle. These records do not establish end-to-end inference acceleration or universal non-regression. All samples and unsuccessful/contended observations remain in the evidence.

## Reproduction

Build the implementation checkout with CMake Release (`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`, then `cmake --build build -j --target dflash_attn_test`). Run:

```sh
python3 bench/results/2026-11-07-kflash-reduce-v8/reproduce.py /path/to/implementation /new/output --correctness-only
```

The runner takes the exclusive flock, builds `check.cpp` against that checkout's library and runs both validation layers. Omit `--correctness-only` for the preliminary isolated timing sequence. Use a fresh output directory for every run. For controlled timings also run the repository's contention monitor; this lightweight isolated runner does not claim to detect other GPU apps.

`graph-check.cpp` is the repository's `qmm_draft_check.cpp` plus a timing branch. Compile it against each exact separately-built baseline/candidate library using the same includes and frameworks as `reproduce.py`. Its ordinary invocation is `graph-check MODEL PROMPT_IDS NEW_OUTPUT [--from REFERENCE_OUTPUT]`. It takes the project GPU lock itself. `REDUCE_GRAPH_TIMING=1` enables the repeated draft-graph timing branch; interleave baseline/candidate processes and monitor contention. No timing branch is used for the output comparison.

## Remaining before Ready

The published portable runner was executed against the canonical implementation build with both validation layers:42/42 cases pass (`public-run-validation.log`).

Nonempty sampled continuations/acceptance behavior, HTTP regression, native `kapitsa bench` A/B, a controlled repeat of the 1536graph pair, and another available M4 packed-profile setup remain to be checked. Remote M3 kernel checks cannot substitute for those packed-model checks. No M5/tensor-layout model runs were made. The canonical baseline/candidate full target-model check passed: 584/584 rows bitwise equal, finite and nonzero (prefill 8,step 192,verify 192,batch 192), with the standing 4B floor unchanged. This target-model check complements the separate drafter comparison; it does not run end-to-end speculative generation. Initial invocation from the wrong working directory could not locate bench/corpus.txt and failed before GPU work; that log is retained, and the corrected run names the corpus explicitly. No floor or existing frozen evidence was changed.

The tier-2 scope report (`required-checks.txt`) also calls for model-level fallback coverage on the 27B, 35B-A3B and Flash-Next. Those model checks have not run for this change and remain pending; the shared reducer was only isolated-tested on the two GPUs. The report’s default-path model discovery does not locate the explicit 4B model folder used here.

## Build and CPU checks

Complete Release build, all 21 standard CPU commands (zero failures), supplemental CPU tests, real-tokenizer grammar (308/308) and structured/parser checks, profile/comment gates and `prek run --all-files` pass. Exact commands and logs are included under `cpu/`. The source version is kapitsa 0.2.31 (0be92def). The CPU bench tests use stand-in servers and do not replace native model benchmarks.
