# M5 Pro, 48 GB: the 16-row drafting block against mlx-serve's draft tree, 2026-10-22

[#266](https://github.com/nace-ai/kapitsa/pull/266) made DFlash2 draft **16 rows by default on a one-slot engine**
(8 rows stay the default on two slots, on MoE targets and on M1-M4). The M5 Pro's 27B cell is Kapitsa's closest
decode race on the poster (74 against mlx-serve's 80 tok/s), so this is where the block had to be measured: four
runs on `main` (`16f2b6a`, Kapitsa 0.2.5) and the mlx-serve 26.10.2 build already on this Mac, in **two passes in
opposite order**, so the pass-order effect (the second engine in a pass reads 2-5% faster, see
[2026-10-21](../2026-10-21-m5pro-48gb/README.md)) cancels in the means. A second, shorter pair on the same day checks
the other M5 Pro cell that could move, Qwen3.6-35B-A3B, which is a MoE target and so keeps 8 rows (the last section).

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
because mlx-serve is ahead on decode, and the poster now carries these four runs' means for both engines, so its 4K
column has mlx-serve ahead too (464 against 459) instead of the cross-day 484-against-428 pairing it carried before.
The headline's count of pairs (16 of 17 on main) does not change.

**The block itself is real.** Every Kapitsa run logged `DFlash2: 16-row drafting block` and drafted at 16.00 rows per
cycle (cf. the 8-row runs in [2026-10-21](../2026-10-21-m5pro-48gb/README.md)), and the flag-free run above reads
484 / 453 tok/s at 4K / 16K against the 2026-10-21 run's 463 / 451, and 74.8 / 83.3 tok/s against 74.1 / 77.4.
The poster carries the pair, not one run: both its Kapitsa cell and its mlx-serve cell are this section's means,
**77 tok/s decode and 459 tok/s 4K prefill** for Kapitsa against **78 and 464** for mlx-serve, so the two cells were
measured together rather than a day apart. The flag-free run above is where the block's own numbers (484 / 453,
74.8 / 83.3) come from.

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

## 35B-A3B on the build before the tensor-unit expert prefill (10:51-11:14, two passes)

The block is a **dense**-target default: MoE targets keep 8 rows, so the poster's 35B-A3B cell should not move. It was
measured anyway, in the same two-pass shape and under the same lock. **Build: `b07f274`** (the branch whose tree
[#278](https://github.com/nace-ai/kapitsa/pull/278) merged; Kapitsa 0.2.5), which has the 16-row block but **not**
[#279](https://github.com/nace-ai/kapitsa/pull/279): the routed experts' prefill ran the simdgroup path, i.e. the
tensor-unit path was off. This is not current main. mlx-serve is the 26.10.2 build:

| Pass | Engine | Run | Decode T=0.6 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Peak (GB) | Foreign GPU load |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 (fwd) | Kapitsa | `q36-35b-kapitsa-20261022-105127` | 197.48 (194.73-226.47) | 221.79 | 1,814 / 1,461 | 21.91 | **2.36% mean, flagged: no** |
| 1 (fwd) | mlx-serve | `q36-35b-mlxserve-mtp-20261022-105522` | 152.12 (150.05-158.75) | 157.43 | 2,019 / 1,856 | 21.92 | 8.41% mean, 17.3% max |
| 2 (rev) | mlx-serve | `q36-35b-mlxserve-mtp-20261022-110149` | 162.85 (158.38-165.75) | 163.95 | 2,079 / 1,850 | 20.99 | 3.91% mean, 20.7% max |
| 2 (rev) | Kapitsa | `q36-35b-kapitsa-20261022-110818` | 204.38 (178.71-212.37) | 217.68 | 1,752 / 1,413 | 21.91 | 10.87% mean, 21.0% max |
| | **Kapitsa mean** | | **200.93** | **219.74** | **1,783 / 1,437** | **21.91** | |
| | **mlx-serve mean** | | **157.49** | **160.69** | **2,049 / 1,853** | **21.46** | |

**The 16-row block does not move this cell.** Kapitsa's flag-free pass reads 1,814 / 1,461 tok/s at 4K / 16K and
197.48 / 221.79 tok/s decode against the 2026-10-21 run (`a13f8fd`), 1,799 / 1,455 and 195.1 / 223.4: within 1.2% on
every column (decode +1.2%, prefill +0.8% and +0.4%), which is what a MoE target keeping its 8 rows predicts.

**These numbers are not the poster's cell.** The poster's 35B-A3B Kapitsa cell is the 10:25 run on `d53f02e`
([`q36-35b-kapitsa-20261022-102548`](results/q36-35b-kapitsa-20261022-102548.json), tensor-unit expert prefill on,
[#283](https://github.com/nace-ai/kapitsa/pull/283)): 2,459 / 1,838 prefill, 217.0 / 227.0 decode. The four runs
above, taken 30-50 minutes later on the build before that change, read 1,814 / 1,461 and 197.5 / 221.8 (flag-free pass),
so the tensor-unit path is worth about 1.35x at 4K and 1.26x at 16K on this Mac (two builds, different sessions, not an
interleaved A/B; #279's own A/B is the reference). Today's mlx-serve runs, both flagged, read 2,049 / 1,853 at 4K / 16K,
about 18% under its stored 2,494 / 2,283, so they do not refresh its cell. All four runs are stored here as the
before-state; none of them feeds a poster number.

## Conditions

macOS 26.5.2 (25F84), `Mac17,8`, M5 Pro 20-core GPU, AC power, High Power mode, `--slots 1`, 32K context, poster
protocol (`compare_engines.py`, `--skip agent`, `--quiet-timeout 180`), GPU lock `/tmp/kapitsa-gpu.lock` held by the
caller for both engines (`lockf -k`, `NACE_GPU_LOCK_HELD=1`); no process was suspended to get the GPU. Swap free was
0.9 GB of 7.0 GB at the start (6.2 GB of that made by other sessions, RAM had 24-36 GB free); the 20.3 GB peak
footprint is unaffected. Kapitsa: `qwen3.8-27b-4bit-mtp` with the `Qwen3.8-27B-DFlash2` drafter, build `16f2b6a`
(0.2.5). mlx-serve: its own 4-bit pack with the DFlash2 draft tree (block 8), 26.10.2.

## The 35B-A3B cell, after the experts' prefill moved to the tensor units

The poster's M5 Pro Kapitsa cell for Qwen3.6-35B-A3B, re-measured on the build that runs the routed-expert prefill on
the qmt tile by default ([#279](https://github.com/nace-ai/kapitsa/pull/279)); the cell it replaces is the 2026-10-21
run at `a13f8fd` ([../2026-10-21-m5pro-48gb](../2026-10-21-m5pro-48gb/README.md)), 1,799 / 1,455 tok/s of prefill and
195.1 / 223.4 of decode. Written by [bench/harness](../../harness/README.md):
`compare_engines.py configs/kapitsa.json --only 'q36-35b-*'` under the GPU lock.

MacBook Pro (16-inch, 2026), `Mac17,8`, M5 Pro, 48 GB, macOS 26.5.2, on AC power in High Power mode, isolated: the
quiet window passed and no phase is flagged. Kapitsa 0.2.5 at `d53f02e` (#279's branch commit before it was squashed to `2f1978d` on main; `git diff d53f02e 2f1978d -- src` is comment edits, #277's dead-tile removal and update/bench-command changes, so the engine is the same), 4-bit
Qwen3.6-35B-A3B with its DFlash2 drafter.

| Result file | Model | Prefill 4K / 16K (tok/s) | Decode T = 0.6 / greedy (tok/s) | Cached follow-up 4K / 16K (s) | Peak footprint (GB) | Agent session (s) | Flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `q36-35b-kapitsa-20261022-102548.json` | Qwen3.6 35B-A3B, DFlash2 | 2,459 / 1,838 | 217.0 (206.9-237.5) / 227.0 | 0.05 / 0.07 | 21.7 | 35.5 | none |

The expert tile moves each output's K reduction by one half ulp, so the logits are not the simdgroup path's to the bit;
`tools/logit_compare.py --reorder` gates that drift and the cold temperature-0 continuations were identical (the
evidence is in [#279](https://github.com/nace-ai/kapitsa/pull/279)). Decode moves with the prefill because the DFlash2
drafter's verify batches are multi-row passes through the same grouped GEMMs.

## 35B-A3B after #293: the prefill attention on the matrix units (15:00)

[#293](https://github.com/nace-ai/kapitsa/pull/293) made the 35B's prefill attention `qattn_t_g8s4k16` the default on an
M5-class GPU (`NI_QATTN_G8=0` keeps `attn_prefill11r`). One run on the merge commit `4b89f4e` (Kapitsa 0.2.6), same
lock and flags as above, but **on battery power** (the record's `power` reads 'Battery Power', `powermode` 2 = High Power;
the runs above are on AC power), so this is not a like-for-like comparison with them:

| Run | Decode T=0.6 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached 4K / 16K (s) | Peak (GB) | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `q36-35b-kapitsa-20261022-145449` | 200.71 (188.82-211.84) | 219.65 (178.68-222.01) | 2,548 / 2,031 | 0.05 / 0.07 | 21.7 | contention (6.82% mean, 19 of 22 samples) + one decode run looped |

The browser (`Microsoft Edge H`) held 5.8% of the GPU through the decode phases and 6.8% overall, so the harness sat out
its 300 s quiet window and took the record with the flag it always sets. Contention and battery power can only lower the
numbers, so the prefill cells read as lower bounds, but the decode comparison below mixes both effects. Read against
the flag-free run above (`d53f02e`, 2,459 / 1,838 prefill, 217.0 / 227.0 decode): **prefill is up 3.6% at 4K and 10.5% at
16K** even under that load, which is #293's ~11% 16K gain and is the reason the poster's 16K prefill cell is stale; decode
reads 7% and 3% lower, and #293 does not touch the decode path (`qattn_t_g8s4k16` is prefill-only), so those two cells
stay with the flag-free run rather than being rewritten from a contended one. The looped decode run is one of the five
sampled at temperature 0.6 (distinct-4-gram 0.44), the same harness flag the earlier 35B runs carry.
