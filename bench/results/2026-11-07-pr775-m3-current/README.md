# Qwen3.5 9B segmented ANE: current-head M3 Ultra calibrated share

Runs: Apple M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1 (25E253), SDK 26.2. Kapitsa `9cca6b0d181d020cad9949eb002ba274baeaaf45`, Qwen3.5-9B MLX 4-bit group-64. Model hashes and the eight windows/record IDs match the M4 Pro run. This is fresh evidence at the CI/non-matrix head; it does not relabel the earlier integration runs.

Both arms explicitly set `NI_ANE_UNITS=16`, chosen by the earlier calibration sweep before these quality/bar results. **M3 has no built-in ANE share and no calibration file here: this is not a shipping M3 fast-default measurement.** It represents the non-matrix class with an explicitly calibrated share. M4 built-in-share results are separate.

## Interleaved loaded HTTP latency

The repository `bench/harness/bench.py` HTTP harness, not a community `kapitsa bench` submission: A1/B1/A2/B2, fresh servers/homes, warmup followed by an idle-GPU wait, the same uncached 4135-token request, chunk 2048, context 32768, one slot, no disk cache or drafting. The only timing-arm change is `KAPITSA_PROFILE=precise/fast`; `NI_ANE_ASYNC=0`, `NI_ANE_KEEP_S=0`, `NI_NO_MTP=1` and the calibrated share are common.

| Leg | Profile | TTFT seconds |
|---|---|---:|
| A1 | precise | 3.5389711251482368 |
| B1 | fast | 2.7120331660844386 |
| A2 | precise | 3.5419678329490125 |
| B2 | fast | 2.6959159588441253 |

Means **3.5404694790486246 -> 2.703974562464282 seconds**, **23.6267% lower TTFT**. Every leg had zero cached tokens, empty invalid/flag lists, a successful quiet wait and no flagged foreign-GPU contention. The two-second telemetry had only one sample per leg; it excludes WindowServer and classifies ambient daemons, so this is not strict all-process idle proof or a robust confidence interval. Replies differ. Cold compilation and decode throughput are not this measurement.

## Same-Mac engine bar

MLX 0.32.4 / mlx-lm 0.32.2 was faster: **114.094938 tok/s** versus llama.cpp b12110 **80.998000 tok/s** (medians). It therefore sets this Mac's bar.

**The requested 576 step/verify/batch comparisons pass; the additional eight-row prefill check fails its mean-KL bound.** The failure is retained, not absorbed into a changed threshold or hidden in the aggregate. An unconditional all-path admission claim is not made.

| Path | Rows | Fast flips | MLX flips | Fast mean KL | MLX mean KL | Result |
|---|---:|---:|---:|---:|---:|---|
| step | 192 | 1 | 4 | 0.0002578932421355708 | 0.00039865869471963374 | pass |
| verify | 192 | 1 | 4 | 0.00026662465725817332 | 0.00039953810702752124 | pass |
| batch | 192 | 1 | 4 | 0.0002591454284963073 | 0.00039909276793295738 | pass |
| prefill | 8 | 0 | 0 | 0.00018457479109606169 | 0.00014716473412187535 | FAIL |

The prefill audit verifies all 248,320 logits per row, identical token prefixes across precise/fast/MLX, and that each prefill prefix is exactly one token before the first corresponding step record. Recomputing each row through the same `logit_compare.py` metric reproduces the failed mean. The 3200- and 1280-token windows contribute the largest excesses (`prefill-audit.json`). This is an observed numerical difference, not a discovered row-alignment problem. No rerun was selected to replace it.

Native engine speed uses the same 75-token chat prompt and three greedy 256-token continuations after warmup, as on the M4. Here the temporary GGUF was stored on disk, not a RAM disk; the M3 has more available disk space. Both engines ran under the GPU lock. The downloaded GGUF is removed only after the native runs; its pinned URL, revision and SHA256 are retained. MLX used the same 4-bit files as Kapitsa; llama.cpp used Q4_K_M, a different quantization. Neither result establishes a winner at every context or on another Mac.

Each native engine emitted 192 single-token teacher-forced full-vocabulary rows plus eight prefill rows. Native step rows were reused against the matching Kapitsa batch/verify records: 576 comparisons are correlated, not 576 independent native outputs. The gate commands use the exact measured path-specific bounds, without editing a floor. Zero-drift gating remains a separate expected failure.

## Other checks and limits

Fast logged 576 ANE evaluations and zero reruns; precise logged zero evaluations. The segment checker passed 50 CPU checks and 2098 GPU comparisons with both Metal validation layers enabled. Runtime metadata, geometry logs and model identities are included. The current-head CPU/CI results and the M4 dependency-repaired CPU reruns are recorded separately.

This is the same Qwen3.5-9B model on two non-matrix setups, not coverage of every model/hidden width or ANE share. M5 is excluded from the new fast default. Existing 4B/27B whole-model paths were not re-evaluated here. The [long-prompt study](../2026-11-07-pr775-long-quality/README.md) retains the large prompt-position KL tails and budget-exhausted answers; a bar pass is not a reasoning-equivalence claim.

Large full-vocabulary matrices remain on the measuring machine; their hashes are recorded separately from the committed small logs and metadata. Exact driver snapshots are included for provenance; substitute checkout/model/scratch paths when reproducing them. All model operations must hold `/tmp/kapitsa-gpu.lock`.
