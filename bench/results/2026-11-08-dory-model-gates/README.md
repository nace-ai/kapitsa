# Dory: source-specific model validation

Source: `b3bece8931c78885e682bfb8da771598d721093a`. Reference: `bbd84e0d53be75866087aa615bdeb6f2411f393f`. Runs: Apple M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1 (25E253), Release `-O3`. Each source was configured and built separately. The seven-digit reference version is the unique prefix of the recorded base; candidate versions use eight digits.

The current-source CPU suite now completes with 21 passes and zero failures, plus 19 supplemental passes. The earlier published snapshot remains unchanged.

The Dory NVFP4 checkpoint, using two loops and per-loop KV, produced 392 compared logit rows (8 prefill, 192 step, 192 verify): every row bitwise identical. The usual corpus includes prefixes from 512 to 3200 tokens. On this 60-core GPU, the host selects the new 16–64-split reduction for the longer prefixes.

Four interleaved production-Model continuations of the retained 3291-token prompt each produced 256 tokens, nonempty decoded text and zero EOS tokens (IDs 2 and 11). All token IDs matched. This driver used no drafter and synchronized each decode step. Its A/B/A/B decode rates were 90.869099, 92.648324, 91.061625 and 92.692022 tokens/s, respectively. These are this driver and prompt's timings, not a native benchmark or a general inference speedup.

Not verified in this snapshot: longer-context and last-loop output gates, native HTTP tests and interleaved `kapitsa bench`, and a second applicable device setup. The PR remains Draft while those gates are collected.

`RESULTS.json` retains every command, exit status, contention sample summary, continuation measurement where applicable, and the complete logit report. The driver held `/tmp/kapitsa-gpu.lock`, waited for a quiet GPU and recorded external GPU activity. Raw logit arrays were hashed before removal; metadata and reports remain in `validation.tar.gz`. Its `MANIFEST.json` hashes every archived file. This folder adds evidence without changing the earlier results.
