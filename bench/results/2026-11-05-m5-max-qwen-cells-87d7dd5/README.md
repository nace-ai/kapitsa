# Results, 2026-11-05: Qwen3.8 27B's and Qwen3.6 35B-A3B's M5 Max cells on main 87d7dd5

Kapitsa's Qwen3.8 27B and Qwen3.6 35B-A3B cells on the MacBook Pro M5 Max, re-measured after three prefill changes
since the morning's cells ([2026-11-05-m5-max-qwen-cells](../2026-11-05-m5-max-qwen-cells/README.md), `446165e`):

- **#964:** the 128-row tile's outputs through the cooperative tensors' store, the same bits. 35B-A3B 4K prefill
  -1.8% in `ni_run`; the 27B level.
- **#969:** the single-slot 27B in 4,096-row chunks on the 40-core M5. Each Neural Engine wavefront half is 2,048
  rows. 4K prefill -4.2% in `ni_run`; it moves outputs within the 27B's floor.
- **#970:** the 27B's GEMMs of 17-63 rows on the 64-row tensor tile, e.g. a prompt's last chunk. 4K prefill -2.5% in
  `ni_run`.

Main also gained #934 (sampling argmax, an M3 Ultra change) and #965 (small-row RMSNorm and X8 prep fused, measured on
an M4 Pro), which touch the decode and verify paths of every Qwen.

**The build:** `7afb51d`, main `49304c9` with #970 merged in, the tree of main `87d7dd5`.

Kapitsa only: no other engine ran, and the other engines' cells keep their dates. The labels, model folders and harness
phase are the 2026-11-01 cells': `compare_engines.py configs/kapitsa.json --only <label> --agentic --skip
tools,agent,cache,sustain`, three runs a label, the session idle ([run_cells.sh](run_cells.sh),
[session.log](session.log)).

MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), on AC power in High Power
mode, under the GPU lock.

**A cell** is the median of its runs' medians, with the runs' range: decode a 512-token coding answer, five answers per
run; prefill one cold prompt per size per run.

## The cells, old against new

| | 27B: 2026-11-01, `7f4e709` | 27B: 05 morning, `446165e` | 27B: now, `7afb51d` | 35B-A3B: 05 morning, `446165e` | 35B-A3B: now, `7afb51d` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill 4K | 904 | 1,104 (943-1,111) | **1,126** (915-1,129) | 6,558 (6,554-6,565) | **6,619** (6,615-6,642) |
| Prefill 16K | 880 | 1,032 (943-1,107) | **1,070** (962-1,095) | 5,657 (5,656-5,661) | **5,718** (5,718-5,737) |
| Decode T 0.6 | 143.3 | 142.6 (138.2-154.6) | 143.9 (134.9-147.1) | 352.7 (347.3-365.9) | 349.4 (345.2-358.2) |
| Greedy (T 0) | 152 | 148.3 (146.3-148.5) | 151.5 (144.7-153.0) | 372.0 (358.3-372.8) | 377.3 (369.4-381.7) |
| Cached follow-up 4K / 16K (s) | 0.13 / 0.14 | 0.13 / 0.14 | 0.12 / 0.12 | 0.04 / 0.05 | 0.04 / 0.05 |
| Peak to 16K / with the 32K sweep (GiB) | 20.1 / 21.4 | 20.1 / 21.4 | 20.8 / 22.1 | 22.3 / 22.9 | 22.3 / 22.6 |

**The 27B:** prefill moves 1,104 -> 1,126 at 4K (+2.0%) and 1,032 -> 1,070 at 16K (+3.7%). It now reads prompts
faster than Splash 1.4.0's 2026-10-25 cells at both sizes: 1,126 against 1,039 at 4K, 1,070 against 1,053 at 16K.
Since the 2026-11-01 cell it reads 4K prompts 24.6% faster (904 -> 1,126) and 16K prompts 21.6% faster (880 -> 1,070).

The cell gained less than `ni_run`'s A/Bs, because the cell reads its prompts right after the decode and greedy phases.
The 4K prompt alone, the same build with the harness's other phases skipped ([prefill-alone/](prefill-alone)), read
1,319 and 1,308 tok/s (3.13 and 3.16 s, 4,131 and 4,134 tokens). In the cell the same prompt took 3.67 s after about
40 s of decoding. The server logs show no other work between the phases. The 4,096-row split chunk ran about 20%
slower in the cell, while the Neural Engine was busy 89% of the time (92-93% alone).

The interleaved runs in [after-decode/](after-decode) ([run.sh](after-decode/run.sh)) confirm it. Each arm ran twice, 30 s
idle before each run, the session idle:

| Arm | 4K prompt (ms) | tok/s | The app's GPU load in the prefill phase |
| --- | --- | --- | --- |
| The decode phase, then the prompt (`dec`) | 3,645, 3,674 | 1,131, 1,122 | 8.1%, 9.2% |
| The prompt alone (`alone`) | 3,131, 3,137 | 1,313, 1,310 | 7.6%, 7.4% |
| `dec` with `NI_ANE_KEEP_MS=500` (`keep`) | 3,882, 3,761 | 1,061, 1,096 | 7.9%, 7.6% |

The foreign load is the same, and the Neural Engine and the GPU both slow down: the chip's state after about 20 s of
decoding, not the server's path. A Neural Engine keepalive every 500 ms instead of 2 s makes it worse.

The 27B's memory rose 0.7 GiB from its first phase on (decode 18.76 -> 19.47 GiB, peak to 16K 20.1 -> 20.8). That is
#969's 4,096-row activation buffers, which its memory plan allocates at load.

**The 35B-A3B:** prefill +0.9% at 4K (6,558 -> 6,619) and +1.1% at 16K (5,657 -> 5,718), both past the morning's
ranges, which is #964. The server already read its 4K prompt in one chunk. Decode reads 349 at T 0.6 (345-358) and 377
greedy, inside the morning's ranges (347-366, 358-373).

## Runs

The 27B's three runs ([results/](results), [logs/](logs)):

| Run | Decode T 0.6 | Greedy | Prefill 4K | Prefill 16K | Contention-flagged phases (the desktop app's mean GPU load) |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 143.9 | 144.7 | 915 | 962 | prefill 4K, 16K, sweep (7.2%, 9.1%, 9.2%) |
| 2 | 134.9 | 153.0 | 1,126 | 1,070 | prefill 4K, 16K, sweep (7.8%, 7.8%, 8.6%) |
| 3 | 147.1 | 151.5 | 1,129 | 1,095 | prefill 4K, 16K, sweep (7.8%, 7.6%, 8.4%) |

The 35B-A3B's three runs:

| Run | Decode T 0.6 | Greedy | Prefill 4K | Prefill 16K | Contention-flagged phases (the desktop app's mean GPU load) |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 358.2 | 381.7 | 6,615 | 5,718 | prefill 4K, 16K, sweep (6.7%, 5.7%, 4.6%) |
| 2 | 349.4 | 369.4 | 6,619 | 5,718 | prefill 4K, 16K, sweep (6.5%, 6.1%, 4.9%) |
| 3 | 345.2 | 377.3 | 6,642 | 5,737 | prefill 4K, 16K, sweep (6.4%, 6.1%, 5.1%) |

- **Contention:** the foreign GPU load in every flagged phase is the Claude desktop app's helper, which drove the
  session. No decode or greedy phase is flagged (1.8-3.0%). As in the morning's set, the 27B's first run read its 4K
  prompt slower (915) than the next two (1,126-1,129).
- **Not run:** the other engines, and the 27B's and 35B-A3B's cells on other Macs.
