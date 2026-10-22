# M5 Pro, 48 GB: the 35B-A3B after #362 and the 27B after #360, 2026-10-23

## Qwen3.6-35B-A3B, main `e6e1f46`

[#362](https://github.com/nace-ai/kapitsa/pull/362) runs the 35B-A3B's single-slot prefill in 4096-row chunks on the M5,
as on the 32-core M1 Max (its serve A/B: cold prefill 1.05x at 4K and 16K, logits bitwise the same). This directory
re-measures the poster's M5 Pro Qwen3.6-35B-A3B cell on main after that merge, with the poster's protocol: the harness,
one config, prefill and decode from the same runs
(`compare_engines.py configs/kapitsa.json --only q36-35b-kapitsa --skip tools,agent`).

Only Kapitsa was re-run, as the poster's rules ask after a merge aimed at one engine. Every other engine's cell stays
its stored run ([2026-10-19](../2026-10-19-m5pro-48gb/README.md), [Splash, 2026-10-22](../2026-10-22-m5-pro-splash/README.md)).

- Mac: MacBook Pro M5 Pro 48 GB (20-core GPU), macOS 26.5.2 (25F84), AC power, display asleep, no other app on the GPU
- Build: main `e6e1f46` (#362's merge), Release, Kapitsa 0.2.9, `kapitsa serve` with the DFlash2 drafter (KFlash
  trees), `--ctx 32768 --slots 1`
- Model: qwen3.6-35b-a3b-4bit
- Prompts: the harness corpus the stored cell used, 4,132 tokens at 4K (4,135 in the last run) and 16,284-16,288 at
  16K, so the comparison with the old cell (4,132 / 16,284) is like for like
- Four launches, each in its own hold of the GPU lock with at least 40% of memory free before it. The first launch of
  the fresh binary is stored but not counted (chosen before the runs; counting all four gives a T = 0.6 median of 212.1
  and greedy 223.1 instead of 214.4 and 222.7, prefill unchanged). The cell is the median of the
  other three, each column its own median.

| Run | Decode T=0.6 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak footprint (MB) | Foreign GPU load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q36-35b-kapitsa-20261023-032830` (first launch, not counted) | 198.63 (196.24-226.31) | 229.89 | 2,728 / 2,192 | 0.06 / 0.07 | 22,268 | 0% max, flagged: no |
| `q36-35b-kapitsa-20261023-032952` | 219.30 (200.02-237.43) | 219.23 | 2,730 / 2,194 | 0.06 / 0.07 | 22,268 | 1.4% max, flagged: no |
| `q36-35b-kapitsa-20261023-033137` | 214.43 (210.88-215.69) | 222.70 | 2,729 / 2,194 | 0.06 / 0.07 | 22,268 | 0% max, flagged: no |
| `q36-35b-kapitsa-20261023-033450` | 209.82 (204.90-230.66) | 223.58 | 2,728 / 2,193 | 0.06 / 0.07 | 22,269 | 0.7% max, flagged: no |
| **Median of the three counted** | **214.43** | **222.70** | **2,729 / 2,194** | 0.06 / 0.07 | | |

Every decode phase has 5 of 5 valid answers, and no run has a contention or invalid flag. The Metal compiler's CPU,
sampled once a second, ran only while each server started (the first and last launches compiled the tensor kernels,
604 and 572 ms; the middle two found them cached) and never above 13% in a measured phase. The runs are in `results/`,
the server logs in `logs/`.

### What changed on the poster

- **Prefill: 2,552 to 2,729 tok/s at 4K (1.07x), 2,056 to 2,194 at 16K (1.07x)**, against the flag-free closing run of
  the Splash session on `4b89f4e`; #362 reported 1.05x with its own serve A/B.
- **Decode: 209 to 214 tok/s** (208.9 to 214.4 sampled, 213.1 to 222.7 greedy). #362 is prefill-only; this is inside
  the runs' spread (200.0-237.4).
- Splash 1.4.0's stored cell (3,514 / 181) still reads the 4K prompt faster (1.29x); Kapitsa decodes 19% faster.

## Qwen3.8-27B, main `b7580af`

[#360](https://github.com/nace-ai/kapitsa/pull/360) runs a cached suffix's chain pass up to 32 rows, not 16. This
re-measures the poster's M5 Pro Qwen3.8-27B cell on main after that merge, with the same protocol
(`compare_engines.py configs/kapitsa.json --only q38-27b-kapitsa --skip tools,agent`). Only Kapitsa was re-run;
mlx-serve's cell stays its two runs from the [2026-10-22 pair on `16f2b6a`](../2026-10-22-m5pro-48gb/README.md).

- Mac: as above (AC power, display asleep, no other app on the GPU)
- Build: main `b7580af` (#360's merge), Release, Kapitsa 0.2.9, `kapitsa serve` with the DFlash2 drafter (KFlash
  trees), `--ctx 32768 --slots 1`
- Model: qwen3.8-27b-4bit-mtp. Every server log shows the default 24/34 Neural Engine units and a cached Neural Engine
  program (ready in 0.1 s).
- Prompts: the harness corpus the stored cell used, 4,131-4,133 tokens at 4K and 16,285-16,288 at 16K (the stored
  cell: 4,131 / 16,287), so the comparison is like for like
- Four launches, each in its own hold of the GPU lock with at least 40% of memory free before it. The first launch of
  the fresh binary compiles the kernels (2,855 ms; 7 ms in the later launches) and is stored but not counted. Another
  session's GPU job ran between the third and fourth launch, in its own hold of the lock, so the fourth started at 04:11.

| Run | Decode T=0.6 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak footprint (MB) | Foreign GPU load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q38-27b-kapitsa-20261023-034805` (first launch, not counted) | 79.15 (73.49-87.86) | 87.81 | 596 / 496 | 0.21 / 0.25 | 20,652 | 1.0% max, flagged: no |
| `q38-27b-kapitsa-20261023-035141` | 78.71 (70.83-84.03) | 88.06 | 658 / 548 | 0.22 / 0.24 | 20,652 | 1.9% max, flagged: no |
| `q38-27b-kapitsa-20261023-035511` | 83.12 (70.66-86.48) | 87.83 | 649 / 545 | 0.22 / 0.25 | 20,652 | 0.8% max, flagged: no |
| `q38-27b-kapitsa-20261023-041131` | 81.37 (68.70-85.68) | 84.37 | 617 / 544 | 0.22 / 0.25 | 20,652 | 1.4% max, flagged: no |
| **Median of the three counted** | **81.37** | **87.83** | **649 / 545** | 0.22 / 0.25 | | |

Every decode phase has 5 of 5 valid answers, and no run has a contention or invalid flag. The Metal compiler's CPU ran
while each server started and stayed at or under 8% in the counted runs' measured phases (21% in the first launch's).

### What changed on the poster

- **Prefill: 618 to 649 tok/s at 4K (1.05x), 541 to 545 at 16K (1.01x)**, against the two runs on `3f7f4c8` after #318.
- **Decode: 79 to 81 tok/s** (79.21 to 81.37 sampled, 85.25 to 87.83 greedy), inside the runs' spread. #360 touches
  only a cached suffix's chain pass, so neither move is #360's.
- **Cached follow-up: no change** (0.22 / 0.25 s against 0.21-0.23 / 0.24-0.26 s on `3f7f4c8`). mlx-serve's stored
  pair answers sooner (0.14 / 0.18 s), and every counted run is slower than its 0.18 s at 16K, so the poster claims no
  follow-up win; the gain to cite stays cached suffix prefill 20-29% faster (probe, #315).
- Against the stored mlx-serve pair (78.16 sampled, 86.23 greedy, 464 prefill) Kapitsa reads the 4K prompt 1.40x as fast
  and leads decode by 4.1% sampled and 1.9% greedy, inside the spread and across days, so the row stays a tie.
- Splash 1.4.0's stored cell (674 / 72) reads the 4K prompt 1.04x as fast; Kapitsa decodes 13% faster.
