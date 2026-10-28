# M4 validation of the 9B ANE opt-in after updating main

`NI_ANE_H4096=1` enables experimental W8A8 prefill for Qwen3.5-9B. Two 2048-channel inputs cover all 4096 hidden channels; gate/up partial sums are added **before** SwiGLU. Surface dimensions, staging offsets and the memory budget follow that width. The existing 128/512-wide rotations, 2560-channel down-projection segments and 4B/27B input segmentation are retained. A separate `disk_format()` marker isolates saved opt-in prefixes. The option remains **off by default**; `NI_ANE=0` also disables it.

**This changes logits and can change greedy replies.** It is not a lossless optimization. Default-path regression checks and opt-in numerical drift are reported separately below.

## Builds and inputs

**Runs:** Mac mini (2024), Apple M4, 10-core GPU, 24 GiB, macOS 26.5.2 (25F84), SDK 26.2, 2026-10-28. Main was fetched before building, then frozen for this comparison:

- **A:** `kapitsa 0.2.24 (dbf3f3b)`, commit `dbf3f3b330263a93c322ca650914bfd82cdecbea`.
- **B:** `kapitsa 0.2.24 (fe74a3d)`, commit `fe74a3db5db5ef4f07233c795e0b6c00a9a13f2d`; the submitted PR's production code. Subsequent changes only add documentation/evidence.
- [Binary and source manifests](candidate/manifest.json), [baseline manifest](baseline/manifest.json), [source-tree and caller audit](evidence/source-audit.json), [machine](evidence/machine.json), [model-file hashes](evidence/model-inputs.json).

| Weights | Pinned Hugging Face revision |
| --- | --- |
| lmstudio-community/Qwen3.5-4B-MLX-4bit | `c43ee1d65576a5d98de1e8405cac93c371a655c1` |
| lmstudio-community/Qwen3.5-9B-MLX-4bit | `b455506b0f574c74616dbcd56879bde38fafcff3` |

Both are affine 4-bit, group 64. All 29 model, tokenizer and companion files were checked before these runs. The served 9B uses `z-lab/Qwen3.5-9B-DFlash@5fc3b3d474760f18c516db87d84c37edbfd3ede6`; the manifests also pin the 4B drafter and MTP sidecars. This is Kapitsa's native runtime.

## 4K native HTTP timing

This is a custom **loaded-model, uncached-prompt HTTP** protocol using Kapitsa's native benchmark client functions, **not `kapitsa bench --quick`**. Every request has 512 nonempty output tokens, `finish_reason=length`, temperature 0, zero cached prompt tokens, and no native invalid flags. Server flags: `--ctx 32768 --slots 1 --disk-cache 0`; A uses defaults, B adds only `NI_ANE_H4096=1`. DFlash2 stays enabled in both. On this M4, B used 24/24 ANE units, wavefront scheduling and the default 512-row minimum.

The [protocol](evidence/speed-4k-plan.json) was frozen before timing: four pairs in AB / BA / AB / BA order, at most one complete retry per contaminated workload pair, and no retries based on speed. Each leg uses a fresh task home, fixed warmup, completed ANE preparation and the exclusive GPU lock. The display was put to sleep with the user's authorization. No builds, downloads or CPU test suites overlapped timing.

A phase is eligible only with at least two 1-second samples, no gap above 2.5 seconds, other GPU activity **including WindowServer** mean ≤1% and peak ≤2%, nominal thermal state, HID inactivity ≥300 seconds, and no phase swapouts. The preceding cumulative swap sample is included. Both sides must pass in the **same attempt**. The statistic is the paired geometric A/B ratio, with a two-sided Student-t 95% interval on log ratios; confirmation requires at least three eligible pairs and an interval above 1 for each workload.

| Workload | Eligible pairs | Median TTFT A → B (s) | Paired A/B [95% interval] | TTFT reduction |
| --- | ---: | ---: | ---: | ---: |
| Python scheduler | 4 | 17.746 → 6.413 | 2.771 [2.737, 2.806] | 63.92% |
| Chinese engineering explanation | 3 | 17.742 → 6.450 | 2.747 [2.731, 2.764] | 63.60% |

**Both primary workload checks passed.** There were 16 server legs and 30 measured requests: all 15 paired attempts are retained, including eight rejected ones. The Chinese workload's third pair was rejected in both attempts. Intermittent `loginwindow` GPU activity was recorded in rejected phases. All runs remained nominal with no new swapouts or observed user activity.

[Every request in execution order and its eligibility](RUNS.md), [paired selections](evidence/speed-4k-analysis.md), [complete numerical analysis](evidence/speed-4k-analysis.json), [raw requests, replies and server logs](evidence/speed-4k), [raw per-leg telemetry](evidence/legs). The descriptive all-first-attempt ratios, including contaminated data, were 2.742 / 2.740; those are sensitivity results, not additional valid benchmark pairs. Repeated historical workloads estimate timing variability here, not generalization to new workloads or Macs.

The complete HTTP duration is secondary and uses only pairs whose **whole request** also passes the telemetry gate:

| Workload | Eligible complete requests | Median HTTP A → B (s) | Paired A/B [95% interval] | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| Python scheduler | 3 | 27.213 → 15.914 | 1.718 [1.670, 1.768] | 41.80% |
| Chinese engineering explanation | 2 | 36.964 → 25.235 | 1.465 [1.449, 1.481] | 31.73% |

The Chinese complete-request estimate has only two pairs and is exploratory. Python reply text matched in all 4/4 selected TTFT pairs; Chinese reply text differed in all 3/3. Complete-request time can therefore include different continuation work; this is not a general steady-state decode-throughput claim. Both tables exclude model loading, warmup and first ANE compilation. The ANE program was already cached: this run does not measure an empty-cache startup or total system peak memory.

## Output checks

Unmodified `tools/logit_compare.py`, full vocabulary of 248,320 logits, `logit_dump` at 2048-row chunks. The standard eight corpus windows give 8 prefill rows and 192 rows each of step, KFlash tree verify, and batch. Input sequences and record positions match by SHA-256; each dump's build is checked against its frozen manifest.

| Regression against A | Prefill bitwise | Step bitwise | Verify bitwise | Batch bitwise |
| --- | ---: | ---: | ---: | ---: |
| 4B shipped default, ANE active | 8/8 | 192/192 | 192/192 | 192/192 |
| 9B opt-out, `NI_ANE_H4096=0` | 8/8 | 192/192 | 192/192 | 192/192 |

[4B report](evidence/logit-4b-default.txt), [9B opt-out report](evidence/logit-9b-default.txt). These are bitwise results for the measured rows on this Mac, not a claim about every model or prompt.

9B opt-in compared with the 4-bit GPU reference:

| Path | Top-1 flips | Mean KL | p99 KL | Max KL | Mean ΔNLL (nats) |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill | 0/8 | 0.000478475 | 0.00198936 | 0.0020642 | +7.50951e-05 |
| step | 2/192 | 0.000369163 | 0.00721572 | 0.00968994 | +0.00167763 |
| verify | 2/192 | 0.000376744 | 0.00734071 | 0.010125 | +0.00169658 |
| batch | 2/192 | 0.0003639 | 0.00697298 | 0.00952818 | +0.0016312 |

[Verbatim report](evidence/logit-9b-on.txt), [full comparison JSON](evidence/logit-9b-on.json). All opt-in rows differ bitwise; every logit is finite and ANE counters show 576 evaluations with zero reruns. The decode paths score the same positions and are not independent samples. Step flip margins were 0.126926 and 0.045088, so the changes cannot be dismissed as identical outputs at tiny ties.

The 9B has no measured floor file. Explicit zero-drift bounds enforce the unchanged opt-out and record the opt-in's deliberate failure (comparison exit 1); they do not constitute a relaxed floor. No 9B noise-floor pass, `--reorder` exception, or quality-equivalence claim is made. KL is distribution drift in nats, not a percentage quality loss.

A separate fixed 16,384-token prefix was checked with 1 prefill + 24 step + 24 verify rows:

| Path | Top-1 flips | Mean KL | p99 KL | Mean ΔNLL (nats) |
| --- | ---: | ---: | ---: | ---: |
| prefill | 0/1 | 3.77869e-05 | 3.77869e-05 | +0.004189 |
| step | 0/24 | 0.000297822 | 0.00185412 | -0.00477632 |
| verify | 0/24 | 0.000293759 | 0.00181906 | -0.00462989 |

[16K report](evidence/logit-9b-16k.txt). No nonfinite logits or ANE reruns. These are functional/numerical checks, **not 16K timing evidence**. The earlier failed four-workload/16K confirmation is not converted into a pass by this new focused 4K run.

## Integration and review checks

- Release builds of A and B, plus all required CPU targets; `vk cpu-tests` **18/18**, **13 supplemental CPU/tokenizer checks**, `prek run --all-files` and `git diff --check` passed. [Build/CPU commands and logs](evidence/cpu/results.json); frozen binary hashes still matched after the CPU suite.
- Current `ane_segment_check --gpu`: **14 CPU checks, 290 GPU comparisons, 8,995,840 data/guard bytes**, no failures. Both `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1` were enabled. [Stdout](evidence/legs/head-segments-debug/stdout.txt), [validation-layer log](evidence/legs/head-segments-debug/stderr.txt).
- Current unmodified native API tests: **49/49**, structured output `--quick`: **43/43**, separately on 9B opt-in and 4B default, three server slots. [9B logs](evidence/head-9b-native-on), [4B logs](evidence/head-4b-native-default).
- Current `test_ane_split_abort.py`, `NI_ANE_H4096=1`: **17/17**. The injected 250 ms bound abandons the split, reruns the chunk on the GPU, reproduces GPU-only replies, and keeps subsequent prompts on the GPU. [Test output](evidence/legs/head-9b-abort/stdout.txt), [all three server logs](evidence/abort-server-logs).
- Four real cache restarts, B-cold → A-cold → B-warm → A-warm: cached tokens **0 → 0 → 4968 → 4968**; expected JSON and same-mode reply text preserved. [Cache evidence](evidence/ane-cache-cross-modes/summary.json).
- `split_bytes()` has one production caller, `Qwen35::ane_plan`: `rotate_simdgroups(hidden)` rejects an unsupported width before the byte count is computed. Zero is not used as a valid budget. The `Surface tx, y` comment alignment is fixed.

Full-model runs omit the Metal debug layer because unchanged `attn_prefill11g` allocates 20,480 bytes of static threadgroup memory; the layer counts it as 40,960 against the 32,768-byte limit. The changed packing/staging geometry was validated separately at small sizes first. No other session's process was stopped; all experiment servers and GPU locks were released.

## Evidence and reproduction

The pinned commands, environment, model revisions, all input/record hashes, full comparison reports, output text and every timing attempt are included. The corpus is the existing frozen [bench/corpus.txt](../../corpus.txt) in both pinned commits; its hash is retained instead of duplicating the file. [Publication manifest](publication.json) documents path aliases and consistent pseudonyms for unrelated GPU client names; measured counters, timings and ownership remain intact. Raw full-vocabulary arrays, binaries, model weights, private cache bytes and unrelated CPU process inventory are omitted. Workload bytes are unchanged.

Run `python3 verify_archive.py` to check the published hashes, re-evaluate all timing eligibility from the archived telemetry, and reproduce the aggregate analysis in a temporary directory. This does not load a model. To rerun on a Mac, build the two pinned commits in separate worktrees with the recorded CMake targets, use the pinned model packages, and run the commands in `scripts/functional_logits.py`, `scripts/functional_integration.py` and `confirm_4k.py` in a **new output directory**. Expand the documented path aliases and freeze fresh build/protocol manifests before timing; local binary hashes and the machine/time window will naturally differ. Keep the exclusive GPU lock and validation flags. The C++ client/observer sources and their exact link commands are in [helper-manifest.json](evidence/helper-manifest.json).

The [earlier M4 Pro numerical/PPL study](../2026-10-28-m4pro-9b-ane/README.md), pinned to `92e8b88`, remains unchanged. Its 768-row and 8192-position samples are not pooled with this rerun, and its PPL comparison uses 4-bit GPU references rather than bf16.

## Not verified

- M5 or other Macs; the current 27B default, 35B-A3B MoE, other hidden-4096 models, and partial ANE shares were not run here.
- Full `kapitsa bench --quick`, valid 16K performance, and general decode throughput.
- Quality against bf16, broad Chinese/reasoning/task accuracy, vision, and contexts beyond 16K. No default or numerical floor was changed.
- Empty-cache ANE compilation/startup, cancellation while a long background compilation is pending, and memory-pressure behavior or total system peak memory.
