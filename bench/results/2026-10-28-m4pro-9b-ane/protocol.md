# ANE error comparison: protocol fixed before GPU measurements

Question: does the optional 4096-hidden ANE path for Qwen3.5-9B show larger deviations from that model's own GPU path than the existing ANE paths of Qwen3.8-27B and Qwen3.5-4B?

## Scope and controls

- One machine: Apple M4 Pro, 48 GiB, macOS 26.4.1 (25E253).
- One unmodified experimental commit for all models: `92e8b882b003971e6569fb58d1147b49e69c5d8a`. Its 4B/27B input segmentation remains the original 2560-channel path. This isolates ANE on/off without unrelated upstream changes.
- Each model compared to its own 4-bit group-64 GPU-only reference. No direct KL between different models.
- Same fixed corpus and token prefixes. Tokenizer hashes must match before sharing token IDs.
- `NI_ANE=0` versus `NI_ANE=1`; `NI_ANE_H4096=1` enables the experimental geometry (has no effect on the older supported geometries).
- `NI_ANE_WAVE=1`, `NI_ANE_ASYNC=0`, `NI_ANE_MIN_ROWS=512`, `NI_ANE_VERIFY=1`, `NI_ANE_KEEP_S=0`. Explicit full FFN shares: 18/18 units (4B), 24/24 (9B), 34/34 (27B), matching this M4 Pro's uncalibrated wavefront defaults and holding the offloaded fraction equal.
- `NI_BAL=0` holds chunk boundaries fixed between GPU and ANE. `NI_KFLASH=1`. Single resident sequence (`--batch 1`, step path only) to bound memory. No drafter is needed: logits come from the target model under teacher forcing.
- A successful ANE logit run must report nonzero ANE chunks/evaluations and zero reruns. Any fallback is invalid, not a zero-error result.
- All model/GPU calls run under the exclusive `/tmp/kapitsa-gpu.lock`. The experiment held the lock exclusively; service-management instructions are omitted from this public copy.
- No default changes, numerical floor edits, or throughput claims.

## Measurements

1. CPU geometry check, then the small `ane_segment_check --gpu` under Metal shader validation and debug layer. This checks new input packing/staging against full-width packing, including guard bytes.
2. Full-vocabulary teacher-forced logits on the repository's eight corpus documents, prefix lengths 3200 down to 512, 96 scored step rows each (768 total), prefill chunk 2048. Two A/B pairs check repeatability. Report prefill's eight rows separately.
3. Full-vocabulary long-context check: two corpus offsets at 4096 and 16384 prompt tokens, 64 scored step rows per window (256 total). One A/B pair. Limited corpus coverage; not evidence for arbitrary long-context workloads.
4. Language-model PPL: the first 1025 tokens of each of the same eight documents; score all 1024 next-token positions, 8192 total, prefill chunk 1024. Use the existing `ni_eval` tool with a GPU top-256 reference. Its PPL uses the actual next-token probability; its accompanying top-256-plus-rest KL is only a lower bound and must be labeled separately from full-vocabulary KL.
5. Use `tools/logit_compare.py --no-gate` for descriptive exact-vocabulary comparisons. Report mean/median/p99/max KL, top-1 flips and reference margins, NLL change, non-finite values and repeat hashes. Preserve per-window metrics. Bootstrap documents/windows, not independent token rows, when estimating cross-model ratios or NLL uncertainty; eight source documents imply limited statistical power.

## Theory and interpretation

For an orthogonal Hadamard rotation R, `(Rw)^T(Rx) = w^T x` in exact arithmetic. With quantization errors ew and ex, the dot-product error is `(Rw)^T ex + ew^T(Rx) + ew^T ex`, plus finite-precision rounding. Absent clipping, each nearest-rounded component's absolute error is bounded by half its quantization step. Those steps depend on the actual rotated row/token maxima.

The 4B, 9B and 27B all use 128-wide input rotations and 512-wide intermediate rotations. 4B has one 2560 input segment; 9B has two 2048 segments; 27B has two 2560 segments. The 9B change does not discard channels or introduce an extra stage of quantization unique to it. Neither hidden width nor segment count establishes a monotonic ordering of end-to-end errors. Weight/activation outliers, nonlinearities, accumulated recurrent/KV changes, depth, and logit margins matter.

Let p be the GPU next-token distribution and q the ANE distribution at the same prefix. `KL(p || q) = sum_v p(v) log(p(v)/q(v))`, in nats. Small logit perturbations have KL approximately half their variance under p; mean and tail KL measure distribution drift, not percent quality loss.

For the observed next tokens y_i, `NLL = -mean_i log p(y_i | prefix_i)` and `PPL = exp(NLL)`. `PPL_ANE / PPL_GPU = exp(NLL_ANE - NLL_GPU)`. Positive delta NLL indicates worse likelihood on this sample, negative indicates better. Both are possible even when KL is positive. Compare relative changes within each model, rather than attributing different models' absolute PPL to implementation error.

These experiments can establish differences on their fixed text sample. They cannot prove universal quality equivalence or attribute every cross-model difference causally to the 2048-channel segmentation.
