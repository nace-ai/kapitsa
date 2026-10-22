# M5 Pro, 48 GB: the 27B after the Neural Engine split moved to the matrix units, 2026-10-22

[#318](https://github.com/nace-ai/kapitsa/pull/318) runs the GPU's share of the 27B's Neural Engine FFN split on the
M5's matrix units and gives the GPU 10 of the 34 units by default (24 stay on the Neural Engine; before, all 34 did).
This directory re-measures the poster's M5 Pro Qwen3.8-27B cell on main after that merge, with the poster's protocol:
the harness, one config under the GPU lock, prefill and decode from the same runs
(`compare_engines.py configs/kapitsa.json --only q38-27b-kapitsa --skip tools,agent`).

Only Kapitsa was re-run, as the poster's rules ask after a merge aimed at one engine. mlx-serve's cell stays its two
runs from the [2026-10-22 pair on `16f2b6a`](../2026-10-22-m5pro-48gb/README.md).

- Mac: MacBook Pro M5 Pro 48 GB (20-core GPU), macOS 26.5.2 (25F84), AC power
- Build: main `3f7f4c8` (#318's merge), Release, `kapitsa serve` with the DFlash2 drafter (KFlash trees, 16-row block),
  `--ctx 32768 --slots 1`
- Model: qwen3.8-27b-4bit-mtp
- Both server logs show `24/34 units ... from the default for this chip` and a cached Neural Engine program (ready in
  0.1-0.2 s), so neither run paid the first-start compile.

| Run | Decode T=0.6 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak footprint (MB) | Foreign GPU load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q38-27b-kapitsa-20261022-195053` | 76.62 (72.83-85.62) | 86.18 | 583 / 536 | 0.21 / 0.24 | 20,562 | 0.97% mean, 3.4% max, flagged: no |
| `q38-27b-kapitsa-20261022-200236` | 81.80 (77.13-91.36) | 84.31 | 653 / 546 | 0.23 / 0.26 | 20,564 | 0.24% mean, 2.8% max, flagged: no |
| **Mean** | **79.21** | **85.25** | **618 / 541** | | | |

Neither run has a contention or invalid flag. The first run waited 244 s for the GPU to go quiet before starting. The
runs are in `results/`, the server logs in `logs/`.

## What changed on the poster

- **Prefill: 459 to 618 tok/s at 4K (1.35x), 446 to 541 at 16K (1.21x)**, against the four-run means on `16f2b6a`.
  #318 reported 494 to 619 and 441 to 516 with its own serve A/B. The old cell's 459 was a mean that included three
  runs flagged for the browser's GPU load, so part of the 1.35x is that load going away, not #318: against the old
  pair's only flag-free run (484 / 453) the gain is 1.28x / 1.19x.
- **Decode: 77 to 79 tok/s** (76.89 to 79.21 sampled, 83.08 to 85.25 greedy). #318 does not touch decode; this is
  inside the runs' spread (72.8-91.4).
- Against the stored mlx-serve pair (78.16 sampled, 86.23 greedy, 464 / 472 prefill), Kapitsa now reads ahead on
  sampled decode by 1.3% (79.21 against 78.16), inside the spread, and mlx-serve stays ahead on greedy by 1.2%. The
  lead crosses two sessions hours apart and the stored mlx-serve runs carry the browser flag where these do not, so the row stays a tie in the README's count of pairs (24 of 26) until mlx-serve is re-run in the same session.
  Kapitsa reads the 4K prompt 1.33x as fast as mlx-serve (618 against 464).
- Against Splash 1.4.0's stored 27B cell (674 / 72, two contention-flagged runs on `4b89f4e`), Splash still reads the
  4K prompt faster, now 1.09x instead of 1.50x; Kapitsa decodes 10% faster at T 0.6 and 20% greedy.

## Not verified

- mlx-serve, Splash and the other engines were not re-run on this build; their cells are the stored runs.
- Not an interleaved A/B with the old build in this session: the before numbers are the stored `16f2b6a` runs.
- Tools and the agent session were skipped (`--skip tools,agent`), as for every poster cell.
