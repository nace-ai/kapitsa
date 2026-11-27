# IQ4_XS CPU row decoding

Explicitly redacted, derived evidence. Original execution records are retained privately.
The JSON measurement fields preserve every recorded pass; command examples below are
placeholders, not original execution transcripts. No private machine usage or paths are included.

Production/test source: `d5589c4259b35a3bd5d5cc711b20975705118956`; reference: `b1be7d4f644b3e58d56499404b38b46cf2d2a5ef`.
Both official CMake Release libraries and the common driver use C++20, `-O3`, arm64;
actual target flags do not define `NDEBUG`. Each setup uses its own compiler consistently across arms.

Runs: M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1, Apple clang 17 (`clang-1700.6.4.2`);
M4 Pro, 20 GPU cores, 48 GiB, macOS 15.7.1, Apple clang 17 (`clang-1700.0.13.5`). These are CPU measurements.

The pinned trained GGUF sample is the first 1024 rows of `blk.0.ffn_up.weight`, K=2560.
See [sample identity](MODEL-SAMPLE.json). Each of eight alternating A/B legs uses 16 passes,
1024 independent API calls per pass, and one untimed warmup. All 128 passes per setup pass
finite, unchanged-input and complete 10485760-byte output comparisons against the reference
CPU output. The collector separately compares every complete leg output literally.
The reference CPU output is not an independent numerical oracle; the integrated 327882-case
integer/scalar oracle is a separate check.

| Setup | Pair | Reference median ns/row | Candidate median ns/row | Latency change |
| --- | --- | ---: | ---: | ---: |
| M3 | A1/B1 | 6224.874023 | 1191.243164 | -80.86318% |
| M3 | A2/B2 | 6240.458008 | 1189.778809 | -80.93443% |
| M3 | A3/B3 | 6228.739746 | 1180.195801 | -81.05241% |
| M3 | A4/B4 | 6234.944824 | 1185.729980 | -80.98251% |
| M4 | A1/B1 | 744.283203 | 211.425781 | -71.59337% |
| M4 | A2/B2 | 710.713867 | 206.197266 | -70.98730% |
| M4 | A3/B3 | 680.196777 | 183.919922 | -72.96078% |
| M4 | A4/B4 | 668.151367 | 182.718750 | -72.65309% |

These are warm CPU decoding latencies, not total import/loading or inference acceleration.
Cross-chip absolute timings are reported as measured; their difference has not been explained.
Every pass and source/library/object/driver identity is in [M3](M3-CPU-ROWS.json) and [M4](M4-CPU-ROWS.json).
The published driver is byte-for-byte the private measurement source, with its SHA256 recorded there.

## Checks and limitations

Both own-source integrated IQ4_XS checks pass 327882 oracle/overlap cases and guarded boundary
cases. M4 also passes standalone ASan/UBSan and rejects both deliberate decoder mutants.
M4 broad checks: 22/22 vk CPU entries and 18/19 supplementary native entries pass, along with
six supplementary Python checks, packing/patch checks, prek and diff checks.

`load_error_test` returns 1 at the same raw U8 read-cache threshold in the candidate and one
newly executed pure reference. Their executables differ; test and loader source are unchanged.
The failing path does not invoke IQ4_XS. This unexplained failure is retained, not counted as a pass.
See [checks](CHECKS.json), including pre-execution setup failures and sanitizer scope.

Full trained-model imports, GPU output checks and default interleaved model performance/API
checks are not complete. This evidence does not establish unchanged complete-model outputs,
free MTP generation, universal API success or inference speed. Non-AArch64 execution and M3
sanitizer have not run. The draft remains open for these model qualification steps.

## Reproduction examples

These are derived placeholder examples. Build the reference and candidate official libraries
from the commits above with the same compiler/Release flags, and link `iq4_xs_rows.cpp` to each
library. Extract the exact pinned byte range in MODEL-SAMPLE.json into `$SAMPLE` and verify its digest.

```sh
$REF_DRIVER --input "$SAMPLE" --output reference.f32 --rows 1024 --cols 2560 --make-oracle
$REF_DRIVER --input "$SAMPLE" --oracle reference.f32 --output A1.f32 --rows 1024 --cols 2560 --passes 16
$CAND_DRIVER --input "$SAMPLE" --oracle reference.f32 --output B1.f32 --rows 1024 --cols 2560 --passes 16
```

Continue A2/B2/A3/B3/A4/B4 with new output labels. Every output gate is outside the timed loop.
For pending model qualification, independently import the complete pinned GGUF with each own
build and `--no-drafter`, compare converted files literally, then run the repository logit output
check and default A/B/A/B. Every model/GPU phase must hold the exclusive GPU lock and set the
caller-held markers. Keep all failures and performance warnings; do not adjust any floor.
