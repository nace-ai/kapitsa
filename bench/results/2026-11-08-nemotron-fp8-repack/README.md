# Nemotron FP8 weight-row packing

Production source `8f3996fea6f1f44537bee4e789b98cd6a531f347`, reference `052104f8f9c99edde129415a7384a1a58dc24ae6`.

The existing FP8 loader rearranges each 16-element group into the byte pairs consumed by its multi-row kernels. The old compiler output used sixteen byte loads and sixteen byte stores per group; a NEON load, byte-table shuffle and store preserve that layout. Other weight formats, dimensions, tensor scales, kernels and model defaults are unchanged. This is the existing default FP8 row packer on Apple silicon, without a new switch or device gate.

Runs: Apple M4 Pro, 48 GiB memory capacity, macOS 15.7.1 (24G231); Apple M3 Ultra, 60 GPU cores, 96 GiB memory capacity, macOS 26.4.1 (25E253). Both used the same SHA256-verified arm64 CPU driver, compiled with Apple clang 17.0.0 and the CMake-equivalent flags `-std=gnu++20 -O3 -arch arm64 -mmacosx-version-min=14.0`, without LTO. The timing driver compiles each source's complete row-packing function separately. A further oracle run linked the candidate to its actual CMake Release library and passed.

## CPU packing measurements

These are initialized, in-memory buffers and four untimed calls per arm, not checkpoint reads or complete model loads. Eight A/B pairs per cell, identical output address, about 256 MiB of input processed per leg, and a complete output-byte gate after every leg. All 128 timing legs passed that gate. [timings.csv](timings.csv) retains every leg. Median microseconds per complete call:

| Rows × width | Input bytes | M4 A → B µs | M4 reduction | M3 A → B µs | M3 reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 × 512 | 131,072 | 15.810 → 2.115 | 86.62% | 139.818 → 14.368 | 89.72% |
| 1,024 × 3,136 | 3,211,264 | 404.558 → 53.922 | 86.67% | 3,447.004 → 384.970 | 88.83% |
| 4,096 × 4,096 | 16,777,216 | 2,079.582 → 274.673 | 86.79% | 18,005.361 → 2,297.695 | 87.24% |
| 8,192 × 4,096 | 33,554,432 | 4,146.820 → 577.497 | 86.07% | 36,015.654 → 4,935.651 | 86.30% |

This is 86.07–89.72% lower CPU packing latency in these cells. It is not a complete model-startup or inference-speed measurement. The two chips' absolute timings differ; no cross-chip throughput claim is made.

## Byte-layout checks

Each chip passed 17,161 cases against an independently indexed oracle. For source element `j` within a group, the oracle assigns its unchanged byte to `4*((j%8)/2)+(j%2)+2*(j/8)`. Tests cover all 256 raw byte codes at all 16 positions, distinct row patterns, unaligned source/output pointers, source immutability, output canaries, source/output tail guard pages, zero/multiple rows, whole-group widths through 4,096 (including the Nano 4B width 3,136), legacy incomplete-group safety controls and both FP4-layout arguments. Other formats are checked against the unchanged reference function. Production FP8 admission requires width divisible by 16; no tail support is added.

The M4 ASan/UBSan correctness build also passed 17,161 cases. The actual candidate CMake Release library passed the same full oracle. A test-only wrong shuffle index failed the independent oracle as expected; production code stayed unchanged. This proves raw-code preservation in the tested layouts, including zero and special codes, rather than changing floating-point conversion or claiming GPU NaN semantics.

## General CPU validation

Own-source CMake Release configure, CLI and CPU-target builds passed. All 21 `vk cpu-tests` entries passed. Fourteen of fifteen supplemental native checks passed, and all five extra CPU scripts were covered after building the omitted `systemone` target and rerunning only the five encoder cases that initially could not execute; their original ten other cases passed.

The remaining supplemental `load_error_test` failure is its cache-residency fixture assertion. Its preceding tensor bounds/read checks passed. One pre-change control and one candidate retry also failed that assertion, and the two executable files have identical SHA256 `9445d9c9e125c3b9e32a0e05a3a82250ffc7926f6d428b362fabd63ab0c2f90a`. The control at `c6fd4ed7` has a byte-identical tree to reference `052104f8`. This test is not reported as passing. Original failures and selective follow-ups remain preserved privately; no test threshold was changed. Cleanup completed and the CPU worker exited.

## Reproduction examples

These relative/placeholder commands reproduce the CPU method; they are not verbatim original execution logs. Run from a Kapitsa checkout containing both recorded commits. `$OUT` is a new output directory, and `$EVIDENCE` points to this results directory:

```bash
python3 "$EVIDENCE/extract.py" --reference 052104f8f9c99edde129415a7384a1a58dc24ae6 \
  --candidate 8f3996fea6f1f44537bee4e789b98cd6a531f347 --out "$OUT"
c++ -std=gnu++20 -O3 -arch arm64 -mmacosx-version-min=14.0 \
  -c "$OUT/reference.cpp" -o "$OUT/reference.o"
c++ -std=gnu++20 -O3 -arch arm64 -mmacosx-version-min=14.0 \
  -c "$OUT/candidate.cpp" -o "$OUT/candidate.o"
c++ -std=gnu++20 -O3 -arch arm64 -mmacosx-version-min=14.0 \
  "$EVIDENCE/cpu_driver.cpp" "$OUT/reference.o" "$OUT/candidate.o" -o "$OUT/driver"
"$OUT/driver"
"$OUT/driver" bench > "$OUT/timings.csv"
```

The driver is CPU only. Source/binary digests, every paired measurement and correctness scope are in `summary.json`. This publication is a privacy-filtered derived summary and copied timing values; original execution commands and machine-local records stay private.

## Not verified

Actual Nemotron Nano 4B FP8 model output, GPU/API checks and complete-model startup or inference timing have not completed for this change. The affected catalog checkpoint is `nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8` at `3fe6dab75665a93884214ad4b1b95cf02717d081`; NVFP4 or BF16 model controls alone do not exercise this FP8 branch. CPU byte equality does not substitute for those model checks.
