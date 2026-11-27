# Dory two-device validation

Dory's split-attention reduction recomputes the same maximum and exponential weights for every output dimension. Share those weights within each SIMD group while keeping each dimension's numerator and denominator additions in their original order. This addresses one component of #296's long-context work.

A separate kernel handles 16–64 splits. The original kernel is unchanged for shorter reductions and larger split counts. No new option, cache layout or split-count policy.

## Default for

Dory decode/verify reductions with 16–64 splits on Apple GPUs, selected by split count for the whole class. Both M3 Ultra and M4 Pro measured this path and passed model-output gates. Other Apple chip classes were not measured; the implementation does not add a chip-specific gate.

## Measured

Runs: M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1 (25E253); M4 Pro, 20 GPU cores, 48 GiB, macOS 15.7.1 (24G231). Candidate source `b3bece8931c78885e682bfb8da771598d721093a`, reference `bbd84e0d53be75866087aa615bdeb6f2411f393f`, Clang 17 Release `-O3`. M4 runs the same arm64 binaries built from those sources on M3; binary hashes and explicit CMake source labels are recorded. The actual Dory NVFP4 checkpoint has the same SHA256 on both Macs.

Synthetic reduction latency medians are **10.54–19.22% lower on M3** and **15.80–28.97% lower on M4**, across the 15 eligible cells (1/3/8 rows, 16/32/33/60/64 splits). Each device retains all 396 timing legs, including unchanged-path controls. This is a reduction-kernel result, not an overall inference speedup.

[Initial kernel evidence](https://github.com/chainhelen/kapitsa/tree/experiment/dory-shared-reduction/bench/results/2026-11-07-dory-shared-weights) and [new complete follow-up](https://github.com/chainhelen/kapitsa/tree/experiment/dory-shared-reduction/bench/results/2026-11-08-dory-two-device-validation) retain raw commands/results with SHA256 manifests. The initial prototype's short-split regression is preserved separately; the final implementation uses the unchanged short kernel.

## Checked

- Current-source Release build, 21 `vk cpu-tests` entries and 19 supplemental CPU checks passed. All-files `prek` passed again; production code has not changed since those builds.
- **Each Mac:** 120 small cases with `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`, then 180 broader GPU cases, zero failures. Finite values match the ordered reference bitwise; empty-row NaN classification, FP64 error and output guards pass, including split boundaries.
- **Each Mac:** Dory per-loop corpus gate, 392/392 rows bitwise; default last-loop corpus gate, 200/200 rows bitwise. At 16K, M3 has 49 per-loop plus 25 last-loop rows bitwise, and M4 has 49 per-loop rows bitwise. All corresponding `logit_compare.py` gates pass.
- Own-build generic controls on M3: Qwen 3.6 35B 4-bit and Qwen 3.5 4B 4-bit with explicit ANE each pass 584/584 finite rows bitwise. The 4B arms actually execute 576 ANE evaluations across 10 chunks with zero reruns and startup verification enabled. Both model tokenizer/parser and structured CLI checks pass.
- M3's two-slot native Dory HTTP suite passes. Earlier one-slot baseline and candidate runs each failed the same three conversation-retention assertions; those results remain failures in the archive.
- GPU legs hold the exclusive lock. New checks report no foreign contention; native conditions and every flag are retained.

## Fixed-input model checks

Fresh-process A/B/A/B with identical inputs, per-loop KV, two model loops, synchronous `Model::encode_step`, no drafter. Every one of these 16 runs produces 256 nonempty tokens with no EOS; all IDs match within each prompt/Mac pair. The 8K/16K inputs repeat the retained corpus prompt and are controlled workloads, not general task-quality measurements.

| Mac / prompt tokens | A1 tok/s | B1 tok/s | A2 tok/s | B2 tok/s |
| --- | ---: | ---: | ---: | ---: |
| M3 / 8192 | 88.017 | 88.184 | 87.216 | 88.000 |
| M3 / 16384 | 80.330 | 81.783 | 81.229 | 81.986 |
| M4 / 3291 | 54.488 | 54.437 | 54.319 | 54.060 |
| M4 / 16384 | 46.333 | 45.657 | 45.505 | 45.586 |

The small model-level differences do not establish a general inference gain. The separate, previously published M3 3291-token runs remain in their original evidence directory.

## Native benchmark

`kapitsa bench --quick` on both Macs, A/B/A/B; actual server configuration is two loops, **last-loop KV, plain decoding**. All eight runs return 0 with empty `invalid` arrays. Their native metadata reports the generic format category as `bf16`; the supplied checkpoint and loader are NVFP4, as recorded in model/config and load logs. The JSON label is retained unchanged.

| Mac / arm | Decode tok/s | Greedy tok/s | 4K prompt tokens | Cold TTFT s | Follow-up median s |
| --- | ---: | ---: | ---: | ---: | ---: |
| M3 / dory-bench-0-reference | 103.68 | 104.44 | 4186 | 3.0122 | 0.1030 |
| M3 / dory-bench-1-dory | 105.29 | 107.40 | 4182 | 3.0104 | 0.0917 |
| M3 / dory-bench-2-reference | 104.00 | 104.36 | 4184 | 3.0123 | 0.1075 |
| M3 / dory-bench-3-dory | 103.99 | 104.97 | 4184 | 3.0117 | 0.0972 |
| M4 / bench-0-reference | 59.67 | 59.84 | 4183 | 7.2676 | 0.1135 |
| M4 / bench-1-dory | 60.11 | 59.91 | 4183 | 7.2689 | 0.1247 |
| M4 / bench-2-reference | 60.01 | 60.33 | 4185 | 7.2656 | 0.1226 |
| M4 / bench-3-dory | 59.53 | 60.23 | 4183 | 7.2681 | 0.1178 |

Repetition flags (all preserved):
- M3 dory-bench-0-reference: none
- M3 dory-bench-1-dory: none
- M3 dory-bench-2-reference: decode: repetition loop (distinct-4-gram 0.4908)
- M3 dory-bench-3-dory: decode: repetition loop (distinct-4-gram 0.3821); decode: repetition loop (distinct-4-gram 0.4872)
- M4 bench-0-reference: decode: repetition loop (distinct-4-gram 0.4971)
- M4 bench-1-dory: none
- M4 bench-2-reference: none
- M4 bench-3-dory: none

Prompt lengths and sampled generation differ between native arms. These runs do not establish a general inference speedup. The kernel-level claim above is scoped to the measured reduction.

## Not verified

Other Apple chip classes, larger model loop counts, and contexts beyond 16K in the new controlled output checks were not exercised. M4's separate HTTP suite and 16K last-loop logit check were not run; M3 covers both. The measurements do not establish a general end-to-end speedup or broader task quality. Existing committed evidence was left unchanged; the new archive includes every new run and retained earlier negative/flagged results.
