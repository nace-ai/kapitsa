# M5 Pro, 48 GB: the 16-row drafting block against mlx-serve's draft tree, 2026-10-22

[#266](https://github.com/nace-ai/kapitsa/pull/266) made DFlash2 draft **16 rows by default on a one-slot engine**
(8 rows stay the default on two slots, on MoE targets and on M1-M4). The M5 Pro's 27B cell is Kapitsa's closest
decode race on the poster (74 against mlx-serve's 80 tok/s), so this is where the block had to be measured: four
runs on `main` (`16f2b6a`, Kapitsa 0.2.5) and the mlx-serve 26.10.2 build already on this Mac, in **two passes in
opposite order**, so the pass-order effect (the second engine in a pass reads 2-5% faster, see
[2026-10-21](../2026-10-21-m5pro-48gb/README.md)) cancels in the means.

Each engine therefore has two runs, one from each position:

| Pass | Engine | Run | Decode T=0.6 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Peak (GB) | Foreign GPU load |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 (fwd) | Kapitsa | `q38-27b-kapitsa-20261022-095116` | 78.98 (72.57-84.32) | 82.83 | 434 / 439 | 20.28 | 5.81% mean, 31.8% max |
| 1 (fwd) | mlx-serve | `q38-27b-mlxserve-dflash2-20261022-095547` | 74.43 (71.16-88.87) | 84.95 | 462 / 470 | 20.74 | 11.47% mean, 35.2% max |
| 2 (rev) | mlx-serve | `q38-27b-mlxserve-dflash2-20261022-100013` | 81.89 (73.79-87.53) | 87.51 | 467 / 475 | 20.79 | 4.41% mean, 63.8% max |
| 2 (rev) | Kapitsa | `q38-27b-kapitsa-20261022-100545` | 74.80 (70.12-81.46) | 83.32 | 484 / 453 | 20.28 | **0.87% mean, flagged: no** |
| | **Kapitsa mean** | | **76.89** | **83.08** | **459 / 446** | **20.28** | |
| | **mlx-serve mean** | | **78.16** | **86.23** | **464 / 472** | **20.76** | |

The runs are in this directory; the server logs are in `logs/`.

## What the four runs say

**The 16-row block does not flip the 27B cell.** Under the same conditions on the same machine minutes apart,
mlx-serve's draft tree is ahead on decode at both temperatures (78.16 against 76.89 sampled, 1.6%; 86.23 against
83.08 greedy, 3.8%) and ahead on prefill too (4K 464 against 459, 16K 472 against 446). So the poster's ordering holds
because mlx-serve is ahead on decode, and the poster's Kapitsa 4K lead (484 against 428) is a cross-day pairing — this
10-22 run against mlx-serve's 10-21 run — not something these four runs show on the same day. The headline's count of
pairs (16 of 17 on main) does not change.

**The block itself is real.** Every Kapitsa run logged `DFlash2: 16-row drafting block` and drafted at 16.00 rows per
cycle (cf. the 8-row runs in [2026-10-21](../2026-10-21-m5pro-48gb/README.md)), and the flag-free run above reads
484 / 453 tok/s at 4K / 16K against the 2026-10-21 run's 463 / 451, and 74.8 / 83.3 tok/s against 74.1 / 77.4.
The poster's Kapitsa cell is that flag-free run: **75 tok/s decode, 484 tok/s 4K prefill**.

**Why most of the runs are flagged.** The machine's GPU was shared with this session's browser (`Microsoft Edge H`,
the only foreign process in the samples), which showed up as 4-11% mean and up to 64% peak foreign GPU load. The
flags are the harness's own (`CONTENTION: sustained foreign GPU load > 5% in ...`), not hand-set: a run of this pair
cannot be made quiet while the GUI that starts it is open, so the pair was measured with the same contamination on
both engines, and the poster cell comes from the only one of the four with no flagged phase (0.87% mean, no phase
above the flag's 5% sustained mean; the 21.8% single sample is a browser repaint during a 512-token decode, under the
flag's two-sample "sustained" rule).

Two earlier single-engine attempts at the Kapitsa cell (09:35, 77.66 tok/s at 1.4% and 09:45, 82.15 tok/s at 12.6%)
were discarded: the first was flagged on `prefill_16000`/`tools` only, the second everywhere, and a single run cannot
say anything about a pair whose pass order matters. They are not stored here.

## Conditions

macOS 26.5.2 (25F84), `Mac17,8`, M5 Pro 20-core GPU, AC power, High Power mode, `--slots 1`, 32K context, poster
protocol (`compare_engines.py`, `--skip agent`, `--quiet-timeout 180`), GPU lock `/tmp/kapitsa-gpu.lock` held by the
caller for both engines (`lockf -k`, `NACE_GPU_LOCK_HELD=1`); no process was suspended to get the GPU. Swap free was
0.9 GB of 7.0 GB at the start (6.2 GB of that made by other sessions, RAM had 24-36 GB free); the 20.3 GB peak
footprint is unaffected. Kapitsa: `qwen3.8-27b-4bit-mtp` with the `Qwen3.8-27B-DFlash2` drafter, build `16f2b6a`
(0.2.5). mlx-serve: its own 4-bit pack with the DFlash2 draft tree (block 8), 26.10.2.
