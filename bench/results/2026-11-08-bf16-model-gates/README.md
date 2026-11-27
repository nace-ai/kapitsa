# BF16: source-specific model validation

Source: `5e174cb89eb991a705752d033f7f9da8edfeff87`. Reference: `bbd84e0d53be75866087aa615bdeb6f2411f393f`. Runs: Apple M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1 (25E253), Release `-O3`. Each source was configured and built separately. The seven-digit reference version is the unique prefix of the recorded base; candidate versions use eight digits.

The Qwen3.5 4B 4-bit model produced 584 compared logit rows: 8 prefill, 192 step, 192 verify and 192 batch. Every row was bitwise identical between the two source builds. This is GPU model evidence for ordinary finite model weights; the exhaustive CPU NaN regression remains in the earlier frozen evidence folder.

The model metadata reports zero ANE evaluations. This run does not establish malformed-model recovery, ANE coverage or an inference speedup.

Not verified in this snapshot: native HTTP and interleaved `kapitsa bench`, the 35B MoE model and ANE coverage. The PR remains Draft while the remaining checks are completed.

`RESULTS.json` retains every command, exit status, contention sample summary, continuation measurement where applicable, and the complete logit report. The driver held `/tmp/kapitsa-gpu.lock`, waited for a quiet GPU and recorded external GPU activity. Raw logit arrays were hashed before removal; metadata and reports remain in `validation.tar.gz`. Its `MANIFEST.json` hashes every archived file. This folder adds evidence without changing the earlier results.
