# Qwen3.5 9B segmented ANE: M4 Pro fast-profile evidence

Runs: Apple M4 Pro (20 GPU cores, 48 GiB), macOS 15.7.1, SDK 15.5; Qwen3.5-9B MLX 4-bit. Kapitsa commit `9cca6b0d181d020cad9949eb002ba274baeaaf45` (production change `9be7668ba00e43d60180390da9e229904eade5d8`). Neither arm set `NI_ANE_UNITS`; there was no calibration file. Fast used the built-in 24/24 ANE share. Precise and fast are this same binary, with MTP disabled, disk cache disabled and prefill chunk 2048.

The current-head M4 default-share ABAB and the empirical same-Mac fastest-other-engine bar pass. This does **not** pass zero drift or establish reasoning equivalence. [Long-prompt quality is reported separately](../2026-11-07-pr775-long-quality/README.md), including adverse cases and KL tails. No numerical floor was changed.

## Loaded HTTP ABAB

The repository `bench/harness/bench.py` HTTP harness (not a community `kapitsa bench` submission); one uncached 4135-token request per leg after warmup, interleaved precise/fast/precise/fast:

| Leg | Profile | TTFT seconds |
|---|---|---:|
| A1 | precise | 8.594536082993727 |
| B1 | fast | 3.1626541250152513 |
| A2 | precise | 8.601435332995607 |
| B2 | fast | 3.239195000001928 |

Means: 8.597985707994667 vs 3.200924562508589 seconds, 62.77% lower TTFT. All legs had zero cached tokens, empty invalid/flag lists and a successful idle-GPU wait. The raw contention samples are retained: sampling every two seconds yielded only four samples per precise leg and two per fast leg. The harness excludes WindowServer and treats some daemons as ambient; the saved telemetry lists these classifications, so this is not proof that every process on the Mac was idle. Swap use was approximately 1.135 GB and stable. These numbers describe loaded HTTP TTFT, not cold compilation or decode throughput.

The earlier `performance` attempt was interrupted by the user's pause after A2. Its incomplete legs remain separately recorded and were not mixed into this full ABAB.

## Selecting and measuring the bar

Both competing engines ran natively on this Mac, on the same 75-token chat prompt, one warmup followed by three greedy 256-token continuations with no early EOS. The fastest median single-stream decode selected the bar before the comparisons were gated.

| Engine | Version | Median decode tok/s |
|---|---|---:|
| MLX / mlx-lm | 0.32.4 / 0.32.2 | 50.21212324124796 |
| llama.cpp | b12110, cd959bae9 | 39.0764 |

MLX used the same 4-bit weight files as Kapitsa and its native activation dtype. llama.cpp used unsloth Qwen3.5-9B Q4_K_M, revision `3885219b6810b007914f3a7950a8d1b469d598a5`, SHA256 `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8`. These are practical native-engine packages; GGUF Q4_K_M is not byte-identical to the MLX quantization. A temporary 7 GiB RAM disk held the GGUF for **both** engines' speed measurements, then was detached. Thus memory reserved for the RAM disk was common to both measurements. Native decode timings are a yardstick for bar selection, separate from the HTTP TTFT benchmark.

Both engines teacher-forced the same eight windows and emitted full-vocabulary logits in `logit_dump` layout. Each native engine produced 192 single-token rows, reused for the corresponding Kapitsa step, batch and verify comparisons: 576 comparisons are correlated, not 576 independent native outputs. Eight prefill rows were also compared. The standard repository `logit_compare.py` scored and gated each path against the measured MLX path-specific bar:

| Path | Rows | Fast flips | MLX flips | Fast mean KL | MLX mean KL |
|---|---:|---:|---:|---:|---:|
| step | 192 | 2 | 3 | 0.00036914095033037934 | 0.00037419804915126646 |
| batch | 192 | 2 | 3 | 0.0003638997966277224 | 0.0003731982332792418 |
| verify | 192 | 2 | 3 | 0.0003767443686522171 | 0.0003767599695152809 |
| prefill | 8 | 0 | 0 | 0.00047847517260584294 | 0.0008544665307031944 |

The verify KL margin is only `1.5600863063805048e-8`. A fresh dump of fast produced 584/584 bitwise-identical rows and the repeated MLX dump produced bitwise-identical 192 step rows; the narrow margin is repeatable in this setup, not a robustness claim for other releases, Macs or prompts. Explicit zero-drift gating fails, as expected. Fast recorded 576 ANE evaluations, zero reruns; precise recorded zero evaluations.

## Other checks and scope

The segment checker passed 50 CPU checks and 2098 GPU comparisons with both Metal validation layers enabled. CI for the tested head passed all ten jobs, including segment geometry and profile policy. A local CPU build and suite also ran: 19/21 passed initially; `test_bench` and `test_community` failed because the fresh Python environment lacked `jsonschema`. After installing that dependency, both failed tests passed. Both the initial failure logs and targeted rerun logs are retained in `cpu/`; no code or threshold changed to fix them.

The proposed class is non-matrix Macs with a built-in or calibrated ANE share. This report measures the M4 Pro built-in share. M3 Ultra calibrated-share results are separate; M5 is excluded from the new fast default. Other M4 chips and other hidden widths did not run in this report. Existing 4B/27B paths were not re-evaluated here.

Full-vocabulary matrices remain outside Git because of size. The delivery manifest records their hashes, alongside committed comparison reports, commands, metadata, native probe sources, telemetry and preserved failed/interrupted attempts. The initial competitor attempt failed before inference on a disk-info schema assumption; its log is retained and its temporary device was detached.

The `reproduce/` files are exact snapshots of the measuring drivers, not a new installed harness. For reproduction, substitute the checkout, model and scratch paths recorded in each command; build the named tools from the tested commit, install the pinned MLX versions, and use the b12110 llama.cpp headers/library for the C API probe. Hold `/tmp/kapitsa-gpu.lock` for every model operation. `fast-gates.json` records the exact path-specific comparator commands and exit codes. Compare using the recorded bounds without rounding them. `raw-manifest.json` distinguishes the small included files from the large retained matrices.
