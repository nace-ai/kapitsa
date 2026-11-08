# Batched 512-expert router: preliminary evidence

The branch peels the first selection for normalized batches of at least two rows.
The original kernel is unchanged and remains selected for one row and for non-normalized routing.
This is kernel-level investigation; the default remains under review in a Draft.

Production source: `dc82cf0f39b942157e86b0cd5e518e5caed17e61`, based on `f22effb9bdda7fa17be5e4ce614383a35e1cc62a`.
Timing checker: `13fe5470a11fe46d31166ed34fae218bb0da2677`.
The final checker `dec964487d11821c8f4b5737b20ad64b974d0ac0` only corrects the B static-memory diagnostic; its build and
270-case small validation rerun are in `m3/router-reporting/`. Both kernels report 0 static threadgroup bytes.
The two production files have not changed since the production source. Their hashes are in `summary.json`.

Runs: M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1 (25E253), Apple clang 17, Release `-O3`;
M4 Pro, 48 GiB, macOS 15.7.1 (24G231), the same arm64 binary (minimum macOS 14).
No model was loaded for these GPU checks. The M3 CLI and library were explicitly configured at `dc82cf0f`;
`vk build` and `vk doctor` passed. Corrected checker binaries link that unchanged Release library.

## Numerical checks

On each device, `--gpu --small` with `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1` passes 270 cases;
the same small checks with `NI_MATH_SAFE=1` pass 270 cases; normal `--gpu` passes 720 cases.
The total is 1,260 executions per device, including repeated cases and unchanged-path controls.
Rows are 1, 2, 3, 4, 8, 16, 64 and 257; k is 1, 2, 8, 10 or 16, with both normalization modes.
Patterns cover finite random values, ties, large magnitudes, signed zeros, mixed infinities/NaNs,
all negative infinity and all NaN. Index comparisons, finite-weight bit comparisons, NaN classifications,
output guards and the finite fp64 oracle have zero failures. NaN payload bits are not compared.
These are synthetic router checks, not full-model logit or generated-token evidence.

## Timing cohorts and limitations

`timings.csv` contains every leg of the latest same-library, same-buffer cohort, in execution order.
A is the preserved production router. B is the production dispatch choice: the new kernel only for
normalized batches of two or more rows. Both pipelines come from one Metal 3.1 library, and both
timing arms write the same addresses. There are eight A/B pairs per cell, 8,192 dispatches per leg,
with paired warmup. One-row and non-normalized cells are actual A/A controls.

The following values are medians of all eight legs per arm; positive reduction means lower latency.

### M3 Ultra

| Rows | Norm | Comparison | A us | B us | Reduction |
| ---: | ---: | :--- | ---: | ---: | ---: |
| 1 | 0 | A/A | 5.110914 | 5.570770 | -9.00% |
| 1 | 1 | A/A | 5.102783 | 5.440710 | -6.62% |
| 2 | 0 | A/A | 5.362739 | 5.409332 | -0.87% |
| 2 | 1 | A/B | 5.343668 | 5.043645 | +5.61% |
| 3 | 0 | A/A | 5.387927 | 5.396609 | -0.16% |
| 3 | 1 | A/B | 5.250476 | 4.903936 | +6.60% |
| 4 | 0 | A/A | 5.376597 | 5.358159 | +0.34% |
| 4 | 1 | A/B | 5.267690 | 4.948362 | +6.06% |
| 8 | 0 | A/A | 5.283276 | 5.345096 | -1.17% |
| 8 | 1 | A/B | 5.054390 | 5.011495 | +0.85% |
| 16 | 0 | A/A | 5.304764 | 5.290080 | +0.28% |
| 16 | 1 | A/B | 5.107821 | 4.875178 | +4.55% |
| 32 | 0 | A/A | 5.349770 | 5.363516 | -0.26% |
| 32 | 1 | A/B | 5.215793 | 5.057493 | +3.04% |
| 64 | 0 | A/A | 5.206571 | 5.148303 | +1.12% |
| 64 | 1 | A/B | 5.146016 | 5.002358 | +2.79% |
| 256 | 0 | A/A | 4.857712 | 4.861244 | -0.07% |
| 256 | 1 | A/B | 4.722771 | 4.529559 | +4.09% |

### M4 Pro

| Rows | Norm | Comparison | A us | B us | Reduction |
| ---: | ---: | :--- | ---: | ---: | ---: |
| 1 | 0 | A/A | 3.112566 | 3.129435 | -0.54% |
| 1 | 1 | A/A | 2.977991 | 2.987676 | -0.33% |
| 2 | 0 | A/A | 3.088933 | 3.069931 | +0.62% |
| 2 | 1 | A/B | 2.993531 | 2.842989 | +5.03% |
| 3 | 0 | A/A | 3.132708 | 3.140038 | -0.23% |
| 3 | 1 | A/B | 2.938739 | 2.784737 | +5.24% |
| 4 | 0 | A/A | 3.169436 | 3.191223 | -0.69% |
| 4 | 1 | A/B | 3.003370 | 2.878260 | +4.17% |
| 8 | 0 | A/A | 3.263079 | 3.213926 | +1.51% |
| 8 | 1 | A/B | 3.120399 | 2.989245 | +4.20% |
| 16 | 0 | A/A | 3.098152 | 3.109929 | -0.38% |
| 16 | 1 | A/B | 2.975510 | 2.829068 | +4.92% |
| 32 | 0 | A/A | 3.162799 | 3.164897 | -0.07% |
| 32 | 1 | A/B | 3.023625 | 2.898557 | +4.14% |
| 64 | 0 | A/A | 3.219963 | 3.208924 | +0.34% |
| 64 | 1 | A/B | 3.048938 | 2.929977 | +3.90% |
| 256 | 0 | A/A | 5.352882 | 5.340896 | +0.22% |
| 256 | 1 | A/B | 5.051478 | 4.674815 | +7.46% |


M4 eligible-cell medians are 3.90% to 7.46% lower, with A/A controls between -0.69% and +1.51%.
M3 eligible-cell medians are 0.85% to 6.60% lower, but its one-row A/A controls show B 6.62% and
9.00% slower. That unexplained control drift prevents a stable M3 performance conclusion.
These are observations from one cohort per chip, not qualified inference speedups.

Earlier attempts are retained in the archive:

- `m3/router-evidence`, `router-v2-evidence`, `router-more-rows`, `router-warm`,
  `router-final-evidence`, and `m4`: these compared an embedded reference compiled through
  `compile_extra()` (Metal 3.2) with the production library (Metal 3.1). The differing compiler modes
  confound the timing comparison; those measurements cannot qualify the optimization.
- `m3/router-same-library`: same Metal 3.1 library, but timing arms used separate output buffers.
  Its complete measurements are preserved, including unfavorable controls.
- `m4-same-library`: all 1,260 numerical cases passed; timing was aborted when the quiet-GPU
  preflight failed. No timing result is claimed for that attempt.
- `m3/router-same-buffers` and `m4-same-buffers`: the latest cohort summarized above.

No run is removed because it was unfavorable. Earlier superseded CPU jobs were intentionally stopped;
their cancellation records and partial logs are included, not counted as full-suite passes.

## CPU and remaining work

At this immutable snapshot the final production-source job has passed 11 `vk` C++ entries:
`api_test`, `prefix_cache_test`, `fn_store_test`, `fn_dflash_test`, `tokenizer_test`, `iq_layout_test`,
`remote_test`, `cluster_test`, `split_place_test`, `gemm_slices_test`, and `ws_upgrade_test`.
`tests/test_server.py --cli build/kapitsa` is still running; remaining `vk` tests and the supplemental
14 targets are not claimed as complete. Full-file `prek` passed for the final checker.

Not verified: Flash-Next Q2_0 full-model step/verify/prefill logits and temperature-zero continuations,
EXL3 full-model routing, native HTTP regression, interleaved native inference benchmarks, the 4B/35B
control gates listed by `vk checks`, and generated ISA/register usage. No inference speed claim is made.

Follow-up: finish the existing CPU job, investigate timing controls with balanced run order and fresh
processes, provision the affected Flash-Next model, and compare separate candidate/base Release builds
under the GPU lock. Run the required output/HTTP/native benchmark checks with `NI_EXPERT_STORE=24`
from an internal disk and repeat on another applicable setup before qualifying the default.

## Exported evidence

`raw-evidence.tar.gz` holds privacy-filtered derived logs, placeholder command examples, harnesses and snapshot states, not executable binaries.
`manifest.json` is the SHA256 of every uncompressed member. `summary.json` records the archive digest.
Later results must be added under a new evidence path; this snapshot is frozen.

## Privacy export erratum

This archive and its checksum manifest replace the original public export to remove private machine paths, usernames and resource records. The replacement is explicitly a privacy-filtered derived export, not a verbatim raw log. Modified commands contain `$PRIVATE_PATH` or other explicit placeholders and are reproduction examples. Original archive bytes and their SHA256 remain privately preserved for audit. Timing values, output results, pass/fail records and tested source/binary identities are unchanged; no test was rerun for this export. Kernel static allocation measurements remain intact.
