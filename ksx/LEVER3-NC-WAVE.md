# Lever 3: noncausal slab wave on the ANE (`NI_ANE_NC_WAVE`). Parked: 8K median 0.989

Task #265, 2026-11-02. Branch `ane-nc-wave`, parked, no PR. Not merged, and not a default anywhere.

**Change.** With `NI_ANE_NONCAUSAL` on (the default since #875, 90e97d4e), a noncausal prefill pass runs each layer's
FFN as ceil(T/2048) slabs on lane 0, as a plain split. The wave puts the slabs on two ANE lanes, so slab k+1's ANE FFN
runs under slab k's GPU work. Run 3's head (ddea5e84, rebased here as 4eb83202 with no change under `src/`) turns the
wave on by default for passes of `NI_ANE_NC_WAVE_MIN_ROWS` (8192) rows or more.

**Setup.** Apple M5 Pro (Mac17,8), macOS 26.5.2 (25F84), AC power, Drex Max q4 (`drex-max-v1-kapitsa-q4`) through
`systemone --once`. Env: `NI_PROFILE=0 NI_NO_MTP=1 NI_PREWARM=0 NI_SYS_CACHE=0`, under `lockf -k /tmp/kapitsa-gpu.lock`.
Both arms ran on the same binary and differ only in env. B/A < 1 means the wave is faster. The rule was fixed before
each run: WIN only if every pair has B/A < 1 and the median is ≤ 0.97 at every gated size.

## Run 2: wave on at every size (781ef83b, `NI_ANE_NC_WAVE=1` vs unset), GPU 22:07:20-22:26:18Z

| Size | Pair 1 (AB) | Pair 2 (BA) | Pair 3 (AB) | Median |
|---|---|---|---|---|
| 4K | 0.967 (9253.6 → 8952.3 ms) | **1.034** (8956.6 → 9262.9) | 0.964 (9059.9 → 8735.3) | 0.967 |
| 16K | 0.973 (48234.5 → 46910.3) | 0.956 (50257.0 → 48056.4) | 0.942 (49404.2 → 46531.5) | 0.956 |

Verdict: NO WIN, because 4K pair 2 is above 1.

Parity: GO. With the wave flag unset, the output is byte-equal to the #875 build's (ddeb027b) at 4K and 16K. B's
output is byte-equal to A's at both sizes. Against the GPU-only output (`NI_ANE_NONCAUSAL=0`): argmax equal, max prob
gap 2.4e-6 at 4K and 6.9e-5 at 16K.

## Run 3: wave from 8192 rows (ddea5e84, default vs `NI_ANE_NC_WAVE=0`), GPU 23:13:05-23:24:23Z (11.3 min)

| Size | Pair 1 (AB) | Pair 2 (BA) | Pair 3 (AB) | Median |
|---|---|---|---|---|
| 8K | 0.989 (19557.3 → 19350.7 ms) | 0.952 (20441.0 → 19462.0) | 0.989 (20070.2 → 19850.0) | **0.989** |
| 16K | 0.980 (47442.5 → 46509.2) | 0.969 (47817.6 → 46352.3) | 0.934 (48986.4 → 45768.5) | 0.969 |

The 16K legs ran one measured pass with no warmup. The 8K legs ran a warmup and one measured pass.

Verdict: **NO WIN, parked.** The threshold is 8192, the smallest size where all three pairs are < 1. The 8K median
of 0.989 is above 0.97. 16K alone would pass (0.969), but it is not counted, because choosing 16K after the run would
mean choosing the bar from the result.

Parity: GO. The default output at 4K, where the wave is off, is byte-equal to run 2's #875 identity dump. The default
output at 16K is byte-equal to run 2's wave dump. At 8K, wave and slabs are byte-equal. Against GPU-only: argmax
equal, max prob gap 4.0e-5 at 8K and 6.9e-5 at 16K. B legs logged `pipelines 4 (8K) / 8 (16K) slabs of 2048 over two
lanes`, and A legs logged no wave line. Every leg logged share 16/34, `chunks 1`, and `reruns 0`. No swap-flagged
leg and no abort guard.

Verdict script note: run 3's first verdict skipped every 16K pair, because a single-run leg prints no `median` line.
It now reads each run's `latency_ms`, and a time it cannot read makes the run VOID. The verdict did not change.

## Result

The wave saves 1-7% from 8K rows up. That is not enough to clear the 0.97 bar with three pairs. No GPU time goes to
this lever again unless a new proposal, with quoted kernel bounds, gets its own GO.
