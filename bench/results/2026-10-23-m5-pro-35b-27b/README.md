# M5 Pro, 48 GB: the 35B-A3B after #362, 2026-10-23

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
  the fresh binary compiles the tensor kernels (604 ms) and is stored but not counted. The cell is the median of the
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

## What changed on the poster

- **Prefill: 2,552 to 2,729 tok/s at 4K (1.07x), 2,056 to 2,194 at 16K (1.07x)**, against the flag-free closing run of
  the Splash session on `4b89f4e`; #362 reported 1.05x with its own serve A/B.
- **Decode: 209 to 214 tok/s** (208.9 to 214.4 sampled, 213.1 to 222.7 greedy). #362 is prefill-only; this is inside
  the runs' spread (200.0-237.4).
- Splash 1.4.0's stored cell (3,514 / 181) still reads the 4K prompt faster (1.29x); Kapitsa decodes 19% faster.
