# Neural Engine prefill share on the M6 (and every later generation) by default

Before this change `ane_builtin_share()` matched the chip name on the substrings "M4" and "M5". An M6 has matrix units
(Apple family above 10), so every GPU path treats it as M5-class, but it had no built-in Neural Engine share: the dense
Qwen models (4B, 27B) read prompts on the GPU alone unless a calibration file or `NI_ANE_UNITS` gave one
(`NI_ANE: no calibration for Apple M6 (run ane_calib) and no default share; GPU only`). The check now reads the
generation number from the name ("Apple M<N>", N >= 4), so the M6 and later get the share `default_units()` already
computes for a matrix-unit GPU (the M5 Pro's costs, scaled to the core count): 15 of 18 units on the 4B and 29 of 34 on
the 27B with 12 GPU cores. That is the wavefront's share, which every causal prefill takes. The plain split's share (no
wavefront: Drex Max's noncausal pass in `kapitsa-systemone`, or `NI_ANE_WAVE=0`) stays at 0 on the M6 and later, as on
M1 and M2, until it is output-checked there (`ane_plain_share()`); a calibration or `NI_ANE_UNITS` still sets one.

Runs: Mac mini (M6, 2026; `Mac18,5`), 12 CPU / 12 GPU cores, 32 GB, macOS 27.2.1 (26B131), on AC power, thermal state
nominal throughout. Reference: `origin/main` 9e3159c4, built in its own worktree; candidate: this branch on the same
commit (the JSONs record 9e3159c4 for both, as `kapitsa --version` reads the commit at configure time). Models:
`4b` (qwen3.5-4b-4bit, catalog) and `27b:kquant` (qwen3.8-27b-kquant, `nace-ai/Qwen3.8-27B-MLX-Q4-KQuant` @ 8e62e9e),
both with their DFlash2 drafters. Each leg is `kapitsa bench --model M --quick --yes` from that build's directory
(bench starts the server from its own binary), one at a time, interleaved.

The 4B pairs ran while Screen Sharing was connected: `ScreensharingAgent` used 4.2-4.5% of the GPU on every leg of both
arms, and bench flagged it. The later 4B legs and all 27B legs ran without it; bench's remaining flags there name the
Claude app (2.1-2.8%, every arm alike).

## Prompt reading, 27B KQuant (bench --quick, 4K prompt, cold)

Three rounds, A B C, A B C, C B A (the last reversed to separate drift from the arms):

| Leg | main (GPU alone) | this PR (29 of 34) | calibrated (31 of 34) |
| ---: | ---: | ---: | ---: |
| 1 | 275.3 tok/s, 15.01 s | 455.9 tok/s, 9.06 s | 466.6 tok/s, 8.85 s |
| 2 | 253.9 tok/s, 16.28 s | 443.6 tok/s, 9.32 s | 455.6 tok/s, 9.07 s |
| 3 | 241.1 tok/s, 17.14 s | 435.5 tok/s, 9.49 s | 463.4 tok/s, 8.92 s |
| mean | 256.8 tok/s, 16.14 s | **445.0 tok/s, 9.29 s** (+73%) | 461.9 tok/s, 8.95 s (+80%) |

The slowest PR leg (435.5) is 58% above the fastest main leg (275.3). Writing (512-token answers, median of 3 per leg):
greedy main 50.98 / 50.34 / 45.86 tok/s, PR 51.70 / 44.17 / 50.27, calibrated 50.88 / 45.20 / 50.79 (means 49.1, 48.7,
49.0); T 0.6 main 42.8 / 45.4 / 46.0, PR 46.5 / 43.6 / 39.3, calibrated 44.0 / 45.1 / 42.4. The ~45 tok/s greedy legs
fall in rounds 2 and 3 across all three arms, so the dip follows the time, not the arm. Peak memory 19.4 GB in every leg.

## Prompt reading, 4B (bench --quick, 4K prompt, cold)

| Leg | main (GPU alone) | this PR (15 of 18) |
| ---: | ---: | ---: |
| 1 | 1,868.9 tok/s, 2.21 s | 2,750.4 tok/s, 1.50 s |
| 2 | 1,894.5 tok/s, 2.18 s | 2,674.6 tok/s, 1.54 s |
| mean | 1,881.7 tok/s, 2.20 s | **2,712.5 tok/s, 1.52 s** (+44%) |

Writing, greedy: main 142.25 / 142.88 tok/s, PR 139.66 / 141.47 (-1.4%, inside the legs' spread); T 0.6 main 138.9 /
133.9, PR 128.9 / 147.2. Peak memory 4.3 GB -> 4.5 GB (the Neural Engine program's 163 MB of Metal buffers).

## Short prompts: 600 and 900 tokens (`short-prompts.txt`)

A prompt of 512 to 1,023 rows runs as one chunk below the wavefront's 1,024 rows, where the split measured slower than
the GPU alone on the M1 Ultra and M2 Pro (so `default_min_rows` gives those 1,024). On the M6 it gives 512. Cold
prefills with `ni_run --pf-reps 5` (five each, state reset in between; the calibration moved aside, since main reads it
too), main c399c549 against this branch, A B A B:

| Prompt | Model | main (GPU alone), median ms | this PR (split), median ms | |
| ---: | --- | ---: | ---: | ---: |
| 600 | 4B | 309.2 / 308.4 | 268.8 / 295.8 | -9 to -13% |
| 900 | 4B | 468.5 / 468.3 | 325.2 / 330.4 | -30% |
| 600 | 27B KQuant | 1,913.0 / 1,908.8 | 1,462.0 / 1,482.0 | -23% |
| 900 | 27B KQuant | 3,256.4 / 3,244.5 | 2,069.4 / 2,089.7 | -36% |

The split is faster in every leg (one 4B rep at 515 ms is in `pr2`'s median of 295.8), so 512 stays the M6's smallest
split chunk; the 1,024 rule of the M1 and M2 does not apply here.

## The share: what a calibration finds on this Mac

`ane_calib --wave` sweeps (`calib-4b.log`, `calib-27b.log`, 2048-row chunks of random tokens):

| Units | 4B ms per chunk (3 passes) | Units | 27B ms per chunk (3 passes) |
| ---: | --- | ---: | --- |
| GPU alone | 1067 / 1079 / 1107 | GPU alone | 8108 / 8761 / 8309 |
| 15 (default) | 682 / 684 / 693 | 29 (default) | 4387 / 4417 / 4201 |
| 16 | 672 / 682 / 701 | 31 | 4188 / 4365 / 4113 |
| 17 | 682 / 683 / 707 | 33 | 4205 / 4345 / 4224 |
| 18 | **653 / 654 / 680** | 34 | 4202 / 4229 / 4229 |

On 12 GPU cores the GPU stays the bottleneck up to (nearly) the whole FFN on the Neural Engine: 18 of 18 is best on the
4B in every pass, and 31-34 tie on the 27B (31 picked twice). The default formula lands 2-4% short of that. Measured
through bench with the calibration written (same binary, calibration file present or moved aside, interleaved):

| | default share | calibrated |
| --- | ---: | ---: |
| 4B (15 -> 18 units), 2 legs each | 2,800.8 / 2,774.4 tok/s | 2,911.1 / 2,912.1 tok/s (+4.4%) |
| 27B (29 -> 31 units), 3 legs each | 445.0 tok/s mean (above) | 461.9 tok/s mean (+3.8%) |

Under 10% on one Mac, so this PR keeps the class formula and does not add an M6 row; the numbers are here for whoever
revisits `default_units()`'s matrix-unit costs with more chips.

## Output check

`output-check.txt`: `build/logit_dump` of main and of this branch on the same prefixes, `tools/logit_compare.py`.

- 4B, main (GPU alone) against the PR's 15 of 18: PASS on `batch`, `step` and `verify` (`qwen3.5-4b-4bit.json`);
  1 flip in 192 rows, a near-tie (reference margin 0.099); prefill KL mean 1.38e-3 against the 1.87e-3 bound.
- 27B KQuant, main with `NI_ANE=0` against the PR's 29 of 34: PASS on all three, 0 flips in 192 rows; prefill KL mean
  1.34e-4. Gated at the plain 4-bit 27B's floor (`qwen3.8-27b-4bit.json`): there is no KQuant floor, and the pack runs
  the same 4-bit group-64 kernels.
- 4B, the calibrated 18 of 18 against `NI_ANE=0`: PASS on all three; 2 near-tie flips in 192 rows (margins 0.25 and
  0.045), the gate's allowance.

Bench's text check flagged a repetition loop (distinct-4-gram 0.46-0.47) in two of the six 4B legs with the split
(`4b-pr-2`, `4b-pr-calibrated-2`), in neither of the two main legs, and in none of the 27B's nine. That is inside the
4B's own rate: 7 of the 14 `qwen3.5-4b` results in `bench/community` on main carry the same flag, on M1 Pro, M1 Max,
M3 Pro, M3 Ultra and M5 Ultra alike.

## Files

- `bench/4b-main-{1,2}.json`, `bench/4b-pr-{1,2}.json`: the 4B A/B, legs in that order (A1 B1 A2 B2).
- `bench/4b-pr-default-share-{1,2}.json`, `bench/4b-pr-calibrated-{1,2}.json`: the 4B calibration A/B (D1 C1 D2 C2).
- `bench/27b-main-{1,2,3}.json`, `bench/27b-pr-{1,2,3}.json`, `bench/27b-pr-calibrated-{1,2,3}.json`: the 27B rounds.
- `calib-4b.log`, `calib-27b.log`: the `ane_calib --wave` sweeps.
- `output-check.txt`: `tools/logit_compare.py` output.
- `short-prompts.txt`: the 600- and 900-token cold prefills (`ni_run --pf-reps 5`).
