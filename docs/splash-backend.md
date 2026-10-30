# KFlash against Splash

How KFlash's interleaved A/B against Inco's Splash releases was run, and what it read. Splash is a separate engine here,
measured the way the [benchmarks](benchmarks.md) measure every engine. The README's Flash-Next 180B row quotes the
2026-10-30 re-measure of this comparison.

<a id="kflash"></a>

| Model (Splash source) | KFlash vs the Splash 1.5.0 release | Run |
|---|---|---|
| `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0` | score 1.051 (95% LB 1.030); decode 1.101x (arm medians 66.3 → 73.6 and 59.2 → 63.6 tok/s on the two prompts); 4K prefill 1.004x (arm medians 1,298 → 1,304 tok/s); 16K prefill 1,249 → 1,260 tok/s. KFlash built with macOS SDK 26.5, the release with SDK 27.0; same tokens as the release on 6 prompts × 22 runs | M5 Pro 48 GB, macOS 26.5.2 (25F84), KFlash `d45f8ea7` against the 1.5.0 release `bf5c552b`, both at `--expert-memory 20G --idle-release off`, 29 Oct 2026 |

At the defaults (no `--expert-memory`), 4 interleaved pairs of the same legs,
30 Oct 2026: every leg had at least 38 GiB free, so Kapitsa passed no `--expert-memory` and each engine's auto plan sized the
cache (27.35 GiB of experts for both engines). Decode 1.142x (standard error 1.0%; arm medians 73.1 → 83.6 and
59.8 → 70.4 tok/s), 4K prefill 1.001x (0.5%; 1,346 → 1,349 tok/s), score 1.069. An earlier set of 4 pairs an hour
before, 0 errors, gave decode 1.199x and 4K prefill 0.998x; it is not counted because its legs skipped the engine status
check that the amended run rule makes every leg run, so both sets were run again. Most of the difference is the stock
arm's own decode moving 69.1 → 73.1 tok/s on the code prompt, so treat the default-size decode ratio as about ±5 pp.
The low-memory setting, which passes `--expert-memory` itself below 38 GiB free, was not measured.

The measurement: Kapitsa and the Splash release, each at `--expert-memory 20G`, 8 legs per arm in the order
S K K S S K K S, each leg 2 decode prompts × 3 runs at temperature 0 (256 tokens), 2 cold 4K prompts and a cold 16K one.
Each ratio is the geometric mean of the per-run K/S ratios within each interleaved pair (12 per decode prompt, 8 at 4K);
the tok/s are each arm's median over its runs, so a ratio is not the quotient of its two medians. The score is the
geometric mean of the decode and 4K prefill ratios, and LB its 95% lower bound (decode's: 1.057; 4K prefill's standard
error 0.1%). The two binaries differ by the patches and by the SDK: Inco built the release with SDK 27.0, this Mac has
26.5, and stock 1.5.0 built from source here (`be376043`) is not the release binary. That stock build was not timed, so
how much of the ratio the SDK accounts for is not measured.

Outputs: under fixed verify rows (`--verify-rows all`), on 6 prompts, 20 runs with the patches on and 2 with H1=3's
gate forced to give up every step each gave the release's temperature-0 output (20 runs of the release per prompt, one
output each), as did 4 runs of stock 1.5.0 built from source. Splash's `teacher_logits` (4 × 257 rows) reads every
logit row bitwise identical to stock 1.5.0's built from source. Under Splash's automatic verify rows the row count
follows measured timing, so neither engine's output is fixed from run to run at near-ties.

The same patches on Splash 1.4.0 (`92a5dfed`, #807's engine), against the 1.4.0 release (`20b36bcc`, which a source
build reproduces byte for byte), same method and day: score 1.064 (LB 1.036), decode 1.118x, 4K prefill 1.014x (at default size: 0.989 (se 0.6%)); the
same tokens as the release on 6 prompts × 22 runs and every logit row bitwise identical.

