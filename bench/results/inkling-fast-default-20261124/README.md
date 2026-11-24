# Inkling fast-profile evidence

Runs: M5 Pro, 20-core GPU, 48 GB, macOS 26.5.2, Inkling-2-Mini with the same NVFP4 and MXFP8 checkpoint in both profiles.

## Accepted quality subset

Reference: BF16 compute, same quantized checkpoint. The reference is an archived, rounded log-probability bank with a clipped lower tail. Full-reference KL is unavailable.

The task owner accepted the 62 rows whose per-row top-k unions contain no clipped reference codes and cover at least 99.9% of reference mass. This narrows the earlier all-64-row rule. Zero-based rows 53 and 54 are excluded from conditional KL. They remain in the top-1 count.

| Measure | Precise, each repeat | Fast, each repeat |
|---|---:|---:|
| Mean conditional KL on accepted unions, n=62 | 0.0034085092 | 0.0048458636 |
| Top-1 matches with reference, n=64 | 63 | 64 |

Fast's conditional KL is 1.4216959 times precise's, below the accepted 1.5 threshold. Precise-to-precise conditional KL is zero. The earlier 1.41 ratio used the all-64 retained-support diagnostic, not these certified unions.

The quality run used one binary, SHA-256 `8190dfc928e5` prefix, in precise, fast, precise, fast order. Its recorded build is `0.2.39 (da8f1c7d)` plus the pinned exact-prompt cache repair. The fixture records the full binary and input hashes. The default-selector patch itself has no new GPU run.

The [frozen conditional-quality fixture](<conditional-quality.json>) contains the selected reference log probabilities and native FP32 logits. The excluded rows retain their global top tokens and exclusion reasons. The replay recomputes normalization and conditional KL without loading a model:

```bash
python3 tools/check_inkling_profile_evidence.py bench/results/inkling-fast-default-20261124/conditional-quality.json
python3 tests/test_inkling_profile_evidence.py
```

These are teacher-forced DEV rows from eight contexts. They do not establish generated-answer quality or long-context quality. Fast changes prefill kernels and dense Q8 load packing. It is not a token-identical mode.

## Historical prefill observations

The [individual observations](<prefill-observations.json>) contain all 16 legs, their input and output hashes, and the original source identity. Each fixture ran precise, fast, precise, fast through the same experimental probe binary at commit `da8f1c7d`, not current main.

| Prompt | Precise prefill median, s | Fast prefill median, s | Precise divided by fast |
|---|---:|---:|---:|
| 4K | 32.963 | 5.567 | 5.921 |
| 32K | 333.070 | 48.445 | 6.875 |
| Visible A, 138 tokens | 1.087 | 0.294 | 3.691 |
| Visible B, 138 tokens | 1.086 | 0.295 | 3.684 |

These are descriptive same-binary profile observations. Foreign GPU quiescence was not verified. They are not a benchmark of this PR against current main. All 16 legs completed, but the original runner exited 2 because the exact-token decision did not pass. Every precise repeat matched its precise partner, and every fast repeat matched its fast partner. Precise and fast generated different continuations.

Warm-versus-cold visible first-token time and the equal OFF/ON cache-policy results remain separate open measurements. This default change does not claim a cache improvement.
