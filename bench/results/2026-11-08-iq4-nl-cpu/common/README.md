# IQ4_NL CPU component evidence

These files are explicitly sanitized derived reports, with private originals retained for audit. The included common CPU driver is the exact measured source. The reports preserve every original timed pass, source/binary identity and gate outcome.

Reference production commit: `f7dc2c0c6c5179bb51db51c6c45b3f2b9a3b8d1f`. Candidate production commit: `b9c67e691a5669545eca880f3e75111f9ce87dd0`. Evidence-only commits do not change either production identity. [The published source mapping](../PUBLIC-SOURCE-MAPPING.json) gives the metadata-only public candidate commit with an identical source/test tree; original measurement commit and binary identities remain unchanged.

The existing finite, disjoint arm64 IQ4_NL CPU path now uses NEON table lookup and signed widening with the same exact FP32 products. Nonfinite/overlapping/non-arm64 cases retain scalar behavior. No Metal kernel, format, device profile or environment switch changes.

| Setup | Four paired warm CPU row time changes |
| --- | --- |
| Apple M4 Pro, 20 GPU cores, 48 GiB, macOS 15.7.1, Apple clang 17 (clang-1700.0.13.5) | -58.31%, -73.43%, -72.45%, -73.54% |
| Apple M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1, Apple clang 17 (clang-1700.6.4.2) | -78.65%, -78.56%, -78.70%, -78.56% |

Each setup has eight alternating A/B legs with 16 passes each. Each pass is 1,024 separate official decoder calls on K=2,560 real tensor rows and checks all 10,485,760 output bytes, finite values and unchanged input. Raw outliers and the initial M4 timing trend remain included. Actual flags use arm64, O3, GNU++20 and minimum macOS 14 with no NDEBUG or fast-math. Absolute cross-chip timing differences have no established cause.

The metric is warm per-row CPU dequantization, not total import/loading latency or inference throughput. [CPU-CHECKS.json](CPU-CHECKS.json) covers the focused oracle, sensitivity controls and the synthetic one-tensor runtime gather. Gather coverage is not trained PLE, NgramHash, GPU or full-model gather performance evidence. [CPU-FAILURES.json](CPU-FAILURES.json) retains the candidate and distinct reference load_error_test rc1 failures and the initial compile-only gather correction. The unchanged raw-U8 fixture does not invoke IQ4_NL; its general cause is not diagnosed or repaired. No full-suite PASS is claimed.

Setup model reports contain the completed original import, numerical and default runs with their actual failures. Their numerical byte checks and explicit zero gates are distinct from generic vk recommendations. MLX4-bit Qwen3.5 and Qwen3.6MoE recommendations were not run here, and no matching imported-IQ4_NL floor is asserted or modified. Default native/fixed runtime tests did exercise embedded MTP depth 4; the numerical supplied-sequence tree path is teacher-forced and does not independently qualify freely drafted MTP numerics. Trained PLE and whole import/loading speed remain unverified.
