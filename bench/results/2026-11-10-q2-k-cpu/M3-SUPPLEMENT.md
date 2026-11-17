# Completed M3 CPU supplement

Redacted derived records; originals retained privately. The production decoder and earlier recorded results are unchanged.

Runs: Apple M3 Ultra, macOS 26.4.1, Apple clang 17 (`clang-1700.6.4.2`), C++20 `-O3`. Production `2389fd0459750efbaed1d699bc4c6e62e94de2f6`; reference `4f564ecb87d4ff1375fb72b9d45ae945092cc215`. Both arms use the same driver/header/stubs and verified trained input, replacing only `quant.cpp`.

Both arms pass 958,666 oracle/overlap cases and guarded single/multiple/empty/partial blocks, including the FP exception flags. Trained A/B/A/B medians: **581.706 / 311.258 / 383.851 / 142.151 ns/row**. All 16 samples per leg, binary digests, and source/input digests are preserved in `M3-CPU-VALIDATION.json`. Timing holds the exclusive GPU lock.

Every complete 1024-row decode compares all 4,194,304 output bytes against the separately executed reference, verifies unchanged input, finite values and guards; all four final output files also compare literally equal. The source rows were extracted on M5 and transferred unchanged, not independently extracted or imported on M3. These are CPU decoder timings only. The M5 complete-model evidence and its recorded runtime failures remain as published; no M3 full-model or inference-speed result is claimed.
