# Results, 2026-11-05: Qwen3.8 27B's and Qwen3.6 35B-A3B's M5 Max cells on 0.2.29's prefill changes

Kapitsa's Qwen3.8 27B and Qwen3.6 35B-A3B cells on the MacBook Pro M5 Max, re-measured after two prefill changes:

- **#959:** the dense prefill GEMMs on a 128-row tensor tile, bitwise the same output.
- **#961:** the 27B's Neural Engine share retuned for that tile, 16 -> 14 of 34 units. It moves outputs within the
  27B's M5 floor.

**The build:** `446165e`, main `7a03626` with #959 and #961. Main after #961 (`90575b6`) adds #929, a prepacked QMM
for the 4B on an M4 Pro 48 GiB, which this Mac does not run, so the cells are 0.2.29's.

Kapitsa only: no other engine ran, and the other engines' cells keep their dates. The labels, model folders and harness
phase are the 2026-11-01 cells' ([record](../2026-11-01-m5-max-kapitsa/README.md)):
`compare_engines.py configs/kapitsa.json --only <label> --agentic --skip tools,agent,cache,sustain`, three runs a label
([run_cells.sh](run_cells.sh), [session.log](session.log), [session-27b-rerun.log](session-27b-rerun.log)).

MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), on AC power in High Power
mode, under the GPU lock.

**A cell** is the median of its runs' medians, with the runs' range: decode a 512-token coding answer, five answers per
run; prefill one cold prompt per size per run.

## The cells, old against new

| | 27B: 2026-11-01, `7f4e709` | 27B: 2026-11-05, `446165e` | 35B-A3B: 2026-11-04, `3879b44` | 35B-A3B: 2026-11-05, `446165e` |
| --- | ---: | ---: | ---: | ---: |
| Prefill 4K | 904 | **1,104** (943-1,111), +22.1% | 6,137 (6,133-6,151) | **6,558** (6,554-6,565), +6.9% |
| Prefill 16K | 880 | 1,032 (943-1,107), +17.3% | 5,398 (5,375-5,411) | 5,657 (5,656-5,661), +4.8% |
| Decode T 0.6 | 143.3 | 142.6 (138.2-154.6) | 347.1 (341.5-359.9) | 352.7 (347.3-365.9) |
| Greedy (T 0) | 152 | 148.3 (146.3-148.5) | 376.2 (367.1-376.4) | 372.0 (358.3-372.8) |
| Cached follow-up 4K / 16K (s) | 0.13 / 0.14 | 0.13 / 0.14 | 0.04 / 0.05 | 0.04 / 0.05 |
| Peak to 16K / with the 32K sweep (GiB) | 20.1 / 21.4 | 20.1 / 21.4 | 22.0 / 22.7 | 22.3 / 22.9 |

**Decode** did not change: both changes are in the prefill. The decode cells move inside their runs' ranges.

**Prefill** moved past the old cells' ranges on both models. Against Splash 1.4.0's 2026-10-25 cells:
- the 27B now reads 4K prompts faster (1,104 against 1,039), and Splash still reads 16K prompts faster (1,053 against
  1,032);
- the 35B-A3B reads both sizes faster (6,558 against 6,386 at 4K, 5,657 against 5,607 at 16K).

The interleaved A/Bs of #959 and #961 measured the changes alone with `ni_run` (2,048-row chunks):
- **The 27B:** 3,853.7 -> 3,571.5 -> 3,126.9 ms at 4K, -18.9%.
- **The 35B-A3B:** 662.5 -> 627.9 ms, -5.2%.

Through the server, which runs the single-slot 35B-A3B in 4,096-row chunks, the 35B-A3B's 4K time to first token reads
+6.9% tok/s.

## Runs

The 27B's three runs ([results/](results), [logs/](logs)):

| Run | Decode T 0.6 | Greedy | Prefill 4K | Prefill 16K | Contention-flagged phases (the desktop app's mean GPU load) |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 142.6 | 146.3 | 943 | 943 | prefill, sweep (15.0%) |
| 2 | 138.2 | 148.5 | 1,111 | 1,032 | greedy, prefill, sweep (14.9%) |
| 3 | 154.6 | 148.3 | 1,104 | 1,107 | greedy, prefill, sweep (15.1%) |

The 35B-A3B's three runs:

| Run | Decode T 0.6 | Greedy | Prefill 4K | Prefill 16K | Contention-flagged phases (the desktop app's mean GPU load) |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 352.7 | 372.0 | 6,558 | 5,656 | prefill, sweep (4.5%) |
| 2 | 347.3 | 358.3 | 6,565 | 5,657 | prefill, sweep (4.5%) |
| 3 | 365.9 | 372.8 | 6,554 | 5,661 | prefill, sweep (4.6%) |

- **Contention:** the foreign GPU load in every flagged phase is the Claude desktop app's helper, which was driving the
  session. In the 27B's runs it held about 15% of the GPU through the prefill and context sweep, even with the session
  idle. No plain T 0.6 decode phase is flagged. The 27B's first run of each set read its 4K prompt slower (943 here, 907
  in the first set) than the next two (1,104-1,112).
- **The first set:** the 27B's first three runs of this build ([first-27b-runs/](first-27b-runs)) ran while the session
  was busy, with 6-17% foreign load, and in two runs' decode phases too. They read 4K prefill 907, 1,106 and 1,112,
  decode 138.9, 144.3 and 131.8. Their prefill median matches these (1,106). The cell is the rerun's.
- **Memory:** the 35B-A3B's footprint is 0.25 GiB higher from its first phase on (decode 21.31 -> 21.56 GiB, peak 22.0 ->
  22.3 to 16K). Neither #959 nor #961 allocates buffers. The build also has #945, whose load path reads misaligned
  weights through page-aligned 32 MB staging buffers, which likely accounts for it. That was not bisected.
