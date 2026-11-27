# Router validation follow-up

Runs: Apple M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1 (25E253), Clang 17 Release -O3.
Production source `dc82cf0f39b942157e86b0cd5e518e5caed17e61` is unchanged.
The independently built reference is `f22effb9bdda7fa17be5e4ce614383a35e1cc62a`.
Both model-test binaries were explicitly reconfigured with their eight-character source labels.

## Generic model controls

Qwen 3.5 4B 4-bit and Qwen 3.6 35B-A3B 4-bit each pass all 584 finite rows bitwise: 8 prefill and 192 each of step, verify and batch. Their tokenizer/parser and structured CLI checks also pass. These are controls from this PR's own builds. They do not exercise Flash-Next's changed router. ANE counters are zero.

## Balanced-order kernel diagnostic

Three fresh processes use both production kernels from the same Metal 3.1 library and the same output address. Each process alternates AB and BA pairs. Small/debug and broad numerical coverage from the previously published evidence remains unchanged; this helper changes measurement order only. All three runs return 0 with results retained.

Median latency reduction for normalized multi-row routing, percent (negative is slower):

| Rows | Run 1 | Run 2 | Run 3 |
| ---: | ---: | ---: | ---: |
| 2 | 7.01 | 6.55 | 5.67 |
| 3 | 4.51 | 5.42 | 5.23 |
| 4 | 4.90 | 2.40 | 5.48 |
| 8 | 5.83 | 5.33 | -0.22 |
| 16 | 3.06 | 2.69 | 4.51 |
| 32 | 3.07 | 1.08 | 3.24 |
| 64 | 2.96 | 2.13 | 3.24 |
| 256 | 3.50 | 3.10 | 4.06 |

A/A controls still range approximately -4.09% to +3.47%. The earlier 6.62%/9.00% singleton drift is smaller in this new cohort, but these runs do not prove its cause or turn small noisy differences into an end-to-end speed claim. Earlier Metal 3.2-versus-3.1 cohorts remain confounded and are not qualification evidence.

## Remaining

Full Flash-Next EXL3 and Q2_0 output, continuation, HTTP and native benchmark gates remain pending, as does full-model qualification on a second applicable setup. A complete checkpoint is being prepared; the existing sparse n-gram table is not treated as a complete model. This PR remains a Draft.

`evidence.tar.gz` contains privacy-filtered command examples/results, source and binary identities, comparison reports and the balanced-order helper. `SHA256.json` hashes every archive member. Raw float dumps were hashed and deleted after comparison. Measurements are unchanged; the declared privacy export replaces archived environment records.

## Privacy export erratum

This archive and its checksum manifest replace the original public export to remove private machine paths, usernames and resource records. The replacement is explicitly a privacy-filtered derived export, not a verbatim raw log. Modified commands contain `$PRIVATE_PATH` or other explicit placeholders and are reproduction examples. Original archive bytes and their SHA256 remain privately preserved for audit. Timing values, output results, pass/fail records and tested source/binary identities are unchanged; no test was rerun for this export. Kernel static allocation measurements remain intact.
