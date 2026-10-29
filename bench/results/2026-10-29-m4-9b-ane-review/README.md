# Qwen3.5-9B ANE opt-in: M4 measurements

`NI_ANE_H4096=1` enables two 2048-channel ANE input segments for hidden size 4096. It is off by default. **The existing W8A8 approximation changes logits and can change greedy replies.**

**Runs:** Mac mini (2024), M4 / 10-core GPU / 24 GiB, macOS 26.5.2 (25F84), SDK 26.2. Measured on 2026-10-29 at A `6d5c2aad8bb7f158b064b2f41731ed885400140d` and B `82e9b9e834e82df15b9cefa1926f3f06aa253451` (Kapitsa 0.2.24). These are historical measurements of those commits, not a rerun after subsequent main merges.

**Model:** Qwen3.5-9B-MLX-4bit, affine group-64, revision `b455506b0f574c74616dbcd56879bde38fafcff3`; DFlash2 revision `5fc3b3d474760f18c516db87d84c37edbfd3ede6`.

**Flags:** `--ctx 32768 --slots 1 --disk-cache 0`; A defaults, B adds `NI_ANE_H4096=1`. Native HTTP, temperature 0, 4099 prompt tokens and 512 nonempty completion tokens, zero cached prompt tokens. Model loading and first ANE compilation are excluded. These are not `kapitsa bench` results.

| Workload / metric | Valid pairs | Median A → B (s) | Paired A/B [95% CI] |
| --- | ---: | ---: | ---: |
| Python scheduler / TTFT | 4 | 17.741 → 6.395 | 2.765 [2.734, 2.797] |
| Chinese explanation / TTFT | 4 | 17.737 → 6.453 | 2.736 [2.693, 2.781] |
| Python scheduler / complete HTTP | 4 | 27.099 → 15.800 | 1.723 [1.688, 1.758] |

Four fixed AB/BA/AB/BA pairs; one retry per workload pair allowed only for telemetry rejection. Exclusive GPU lock, HID idle ≥300 s, other GPU activity including WindowServer mean ≤1% / peak ≤2%, nominal thermals, no phase swapouts. Intervals use paired log ratios. All 22 requests, including rejected attempts, are in the [archived run table][runs]. Chinese complete-request timing has only one valid pair and different replies, so it does not establish an overall speedup. The selected Python replies matched; this is not a general decode-throughput result.

At the measured commits, `logit_dump` / `logit_compare.py` found 584/584 rows bitwise identical for both 4B default and 9B opt-out. For 9B opt-in, step/verify/batch each had 2/192 top-1 flips and mean KL 3.692e-4 / 3.767e-4 / 3.639e-4; the zero-drift gate intentionally failed. No noise floor was changed. The [verbatim comparisons][logs] include a separate 16K numerical check.

Release, CPU and lint passed at the measured candidate. GPU segmentation: 290 comparisons / 8,995,840 data and guard bytes, zero failures, with both Metal validation flags. Full-model checks: native API 49/49 and structured 43/43 on 4B default and 9B opt-in, abort/fallback 17/17, and SSD cache isolation across four restarts. Full-model debug validation is limited by the unchanged prefill kernel's doubled threadgroup-memory accounting.

**Not verified by these measurements:** later merged code, M5/other Macs, current 27B/35B-A3B, other hidden-4096 models, partial ANE shares, broad quality/bf16 equivalence, valid 16K timing, empty-cache compilation and memory pressure. See the PR for checks on its latest head.

[Full protocol, commands, model hashes, limitations and replay verifier][archive] are pinned on the fork's evidence branch. This summary replaces duplicate copies of the run table and logs; their contents remain unchanged in that archive.

[archive]: https://github.com/KeyingSong0123/kapitsa/tree/9574eaf247f39b2c000fa798100ac93a5ea4463d/bench/results/2026-10-29-m4-9b-ane-review
[runs]: https://github.com/KeyingSong0123/kapitsa/blob/9574eaf247f39b2c000fa798100ac93a5ea4463d/bench/results/2026-10-29-m4-9b-ane-review/RUNS.md
[logs]: https://github.com/KeyingSong0123/kapitsa/tree/9574eaf247f39b2c000fa798100ac93a5ea4463d/bench/results/2026-10-29-m4-9b-ane-review/evidence
