# Qwen3.5-9B ANE opt-in: current-main validation

`NI_ANE_H4096=1` covers the 4096 hidden channels with two 2048-channel ANE inputs. Gate/up partial sums are added before SwiGLU; surface dimensions, staging offsets and memory accounting follow that width. Existing rotations, 2560-channel down-projection segments and 4B/27B input segmentation remain. The option is off by default; `NI_ANE=0` disables it, and saved prefixes have a separate SSD namespace.

**W8A8 prefill changes logits and can change greedy replies. No default or noise floor is changed.**

**Runs:** Mac mini (2024), M4, 10-core GPU, 24 GiB, macOS 26.5.2 (25F84), SDK 26.2, 2026-10-29. **Kapitsa version:** A `kapitsa 0.2.24 (6d5c2aa)` / `6d5c2aad8bb7f158b064b2f41731ed885400140d`; B `kapitsa 0.2.24 (82e9b9e)` / `82e9b9e834e82df15b9cefa1926f3f06aa253451`. Upstream and fork were fetched before this independent rerun; later commits only package the evidence.

Models: `lmstudio-community/Qwen3.5-9B-MLX-4bit@b455506b0f574c74616dbcd56879bde38fafcff3` and `Qwen3.5-4B-MLX-4bit@c43ee1d65576a5d98de1e8405cac93c371a655c1`, affine 4-bit/group-64. 9B DFlash2: `z-lab/Qwen3.5-9B-DFlash@5fc3b3d474760f18c516db87d84c37edbfd3ede6`. [All 29 model/tokenizer/companion hashes](https://github.com/KeyingSong0123/kapitsa/blob/9574eaf247f39b2c000fa798100ac93a5ea4463d/bench/results/2026-10-29-m4-9b-ane-review/evidence/model-inputs.json).

## 4K native HTTP timing

**Flags:** `--ctx 32768 --slots 1 --disk-cache 0`; A defaults, B adds only `NI_ANE_H4096=1`. Both use DFlash2, temperature 0, 512 nonempty output tokens, `finish_reason=length` and zero prompt-cache hits. TTFT is measured with the model loaded after fixed warmup; model loading and first ANE compilation are excluded (compiled programs were cached). This uses the native HTTP client, not `kapitsa bench --quick`.

Four predeclared pairs: AB / BA / AB / BA; at most one complete retry per workload pair, only for telemetry. Both arms must qualify in the same attempt. Exclusive GPU lock, display asleep, HID idle ≥300 s; other GPU activity including WindowServer mean ≤1% and peak ≤2%, nominal thermal state, zero phase swapouts, ≥2 one-second samples with gaps ≤2.5 s. Confidence intervals are Student-t intervals on paired log ratios; confirmation requires ≥3 eligible pairs and a lower bound above 1 for both workloads.

| Workload | Eligible pairs | Median TTFT A → B (s) | Paired A/B [95% CI] | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Python scheduler | 4 | 17.741 → 6.395 | 2.765 [2.734, 2.797] | 63.83% |
| Chinese engineering explanation | 4 | 17.737 → 6.453 | 2.736 [2.693, 2.781] | 63.45% |

Both primary workloads passed the fixed confirmation rule.
All 14 server legs, 22 measured requests and 11 paired attempts are retained, including 3 rejected attempts. [Every request and rejection reason](https://github.com/KeyingSong0123/kapitsa/blob/9574eaf247f39b2c000fa798100ac93a5ea4463d/bench/results/2026-10-29-m4-9b-ane-review/RUNS.md); [full protocol/statistics](https://github.com/KeyingSong0123/kapitsa/blob/9574eaf247f39b2c000fa798100ac93a5ea4463d/bench/results/2026-10-29-m4-9b-ane-review/evidence/speed-4k-analysis.json).

Complete-request duration is secondary and requires the whole request to pass telemetry:

| Workload | Eligible pairs | Median HTTP A → B (s) | Paired A/B [95% CI] | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Python scheduler | 4 | 27.099 → 15.800 | 1.723 [1.688, 1.758] | 41.95% |
| Chinese engineering explanation | 1 | 36.937 → 24.911 | 1.483 — | 32.56% |

Reply text matches in selected TTFT pairs: Python scheduler 4/4; Chinese engineering explanation 0/4. Different continuations can change complete-request time; this is not a general decode-throughput claim. Any secondary estimate with fewer than three pairs is exploratory. All-first-attempt TTFT ratios, including contamination: Python scheduler 2.752; Chinese engineering explanation 2.759; these are sensitivity checks, not extra valid pairs.

## Outputs and integration

Unmodified `logit_dump` / `tools/logit_compare.py`, full 248,320-token vocabulary, standard eight corpus windows, 2048-row chunks. 4B default and 9B explicit opt-out each have **584/584 bitwise-identical rows** against A (8 prefill + 192 each of step, verify and batch). [4B report](https://github.com/KeyingSong0123/kapitsa/blob/9574eaf247f39b2c000fa798100ac93a5ea4463d/bench/results/2026-10-29-m4-9b-ane-review/evidence/logit-4b-default.txt); [9B opt-out report](https://github.com/KeyingSong0123/kapitsa/blob/9574eaf247f39b2c000fa798100ac93a5ea4463d/bench/results/2026-10-29-m4-9b-ane-review/evidence/logit-9b-default.txt).

| 9B opt-in path | Top-1 flips | Mean KL | p99 KL | Mean ΔNLL (nats) |
| --- | ---: | ---: | ---: | ---: |
| prefill | 0/8 | 0.000478475 | 0.00198936 | +7.50951e-05 |
| step | 2/192 | 0.000369163 | 0.00721572 | +0.00167763 |
| verify | 2/192 | 0.000376744 | 0.00734071 | +0.00169658 |
| batch | 2/192 | 0.0003639 | 0.00697298 | +0.0016312 |

Decode paths score the same positions and are not independent samples. All logits are finite; ANE evaluations are nonzero with zero reruns. 9B has no measured floor: explicit zero-drift bounds pass opt-out and deliberately fail opt-in (comparison exit 1). No noise-floor pass, `--reorder` exception or quality equivalence is claimed. [Verbatim opt-in comparison](https://github.com/KeyingSong0123/kapitsa/blob/9574eaf247f39b2c000fa798100ac93a5ea4463d/bench/results/2026-10-29-m4-9b-ane-review/evidence/logit-9b-on.txt).

A separate 16K prefix has 0/49 top-1 flips, with nonzero logit drift. [16K report](https://github.com/KeyingSong0123/kapitsa/blob/9574eaf247f39b2c000fa798100ac93a5ea4463d/bench/results/2026-10-29-m4-9b-ane-review/evidence/logit-9b-16k.txt) is numerical evidence only.

- Release builds; `vk cpu-tests` 18/18, supplemental CPU/tokenizer and CI helper checks, SDK CPU contract/exports, and `prek` passed. [Commands/results](https://github.com/KeyingSong0123/kapitsa/blob/9574eaf247f39b2c000fa798100ac93a5ea4463d/bench/results/2026-10-29-m4-9b-ane-review/evidence/cpu/results.json).
- `ane_segment_check --gpu`: 14 CPU checks, 290 GPU comparisons, 8,995,840 data/guard bytes, zero failures, with both `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`. [Output](https://github.com/KeyingSong0123/kapitsa/blob/9574eaf247f39b2c000fa798100ac93a5ea4463d/bench/results/2026-10-29-m4-9b-ane-review/evidence/legs/head-segments-debug/stdout.txt).
- Native API 49/49 and structured `--quick` 43/43, separately on 9B opt-in and 4B default, three slots. [9B logs](https://github.com/KeyingSong0123/kapitsa/blob/9574eaf247f39b2c000fa798100ac93a5ea4463d/bench/results/2026-10-29-m4-9b-ane-review/evidence/head-9b-native-on); [4B logs](https://github.com/KeyingSong0123/kapitsa/blob/9574eaf247f39b2c000fa798100ac93a5ea4463d/bench/results/2026-10-29-m4-9b-ane-review/evidence/head-4b-native-default).
- Current opt-in abort/fallback check: 17/17; injected 250 ms bound restores GPU-only replies and disables further splitting. [Test and retained server logs](https://github.com/KeyingSong0123/kapitsa/blob/9574eaf247f39b2c000fa798100ac93a5ea4463d/bench/results/2026-10-29-m4-9b-ane-review/evidence/legs/head-9b-abort/stdout.txt).
- Four real restarts of the candidate binary, opt-in cold → opt-out cold → opt-in warm → opt-out warm: cached tokens 0 → 0 → 4968 → 4968 in one shared SSD cache; expected JSON and same-mode replies preserved. [Evidence](https://github.com/KeyingSong0123/kapitsa/blob/9574eaf247f39b2c000fa798100ac93a5ea4463d/bench/results/2026-10-29-m4-9b-ane-review/evidence/ane-cache-cross-modes/summary.json).
- `split_bytes()` has one production caller; `Qwen35::ane_plan` rejects unsupported widths before computing the budget, so zero is not treated as a valid size. The surface comment alignment is fixed. [Source/caller audit](https://github.com/KeyingSong0123/kapitsa/blob/9574eaf247f39b2c000fa798100ac93a5ea4463d/bench/results/2026-10-29-m4-9b-ane-review/evidence/source-audit.json).

Full-model runs omit the debug layer: unchanged `attn_prefill11g` uses 20,480 B static threadgroup memory, counted by the layer as 40,960 B against a 32,768 B limit. Changed packing/staging geometry was checked first at small sizes.

## Evidence and limits

[Full records, commands, hashes and reproduction scripts](https://github.com/KeyingSong0123/kapitsa/blob/9574eaf247f39b2c000fa798100ac93a5ea4463d/bench/results/2026-10-29-m4-9b-ane-review/publication.json) retain every attempt. `python3 verify_archive.py` in the full archive checks file hashes, recomputes telemetry eligibility and reproduces the aggregate without a model. Only the derived GPU mean allows four binary64 ulps for Python-version rounding; decisions and aggregate statistics remain exact. Raw arrays/models/binaries are omitted and local paths normalized as recorded in the manifest. The corpus is the existing `bench/corpus.txt` in both pinned commits. To repeat, build those commits, expand the documented path aliases and run the archived scripts in a fresh output directory with the pinned packages and exclusive lock.

Historical evidence remains separate: [M4 Pro numerical/PPL study at 92e8b88](https://github.com/KeyingSong0123/kapitsa/blob/9f8dd15a81aa67a69b4e1a86a5c4e2c67495194d/bench/results/2026-10-28-m4pro-9b-ane/README.md) and [M4 timing at fe74a3d](https://github.com/KeyingSong0123/kapitsa/blob/9f8dd15a81aa67a69b4e1a86a5c4e2c67495194d/bench/results/2026-10-28-m4-9b-ane-current/README.md). The historical 9B PPL samples include 2.01072 → 2.01052 on 8192 code positions and 6.41118 → 6.41682 (+0.088%) on a separate 4092-position sample with incomplete reference provenance. They are not pooled or evidence of bf16/task-quality equivalence.

**Not verified:** M5/other Macs, current 27B/35B-A3B or other hidden-4096 models, partial ANE shares; `kapitsa bench --quick`, valid 16K timing or general decode throughput; bf16/broad language/reasoning/vision quality; empty-cache compilation, cancellation during long background compilation, memory pressure or system peak memory. The SDK real-model test was built but not run. This CLT installation lacks the offline `metal` compiler; the exercised M4 kernels were compiled by the GPU driver. Upstream CI status is reported in the PR.
