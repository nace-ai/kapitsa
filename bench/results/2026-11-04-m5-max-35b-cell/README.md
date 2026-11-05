# Results, 2026-11-04: Qwen3.6 35B-A3B's M5 Max cell on main `3879b44`

Kapitsa's Qwen3.6 35B-A3B cell on the MacBook Pro M5 Max, re-measured on main `3879b44`. That is Kapitsa 0.2.28 plus
#951, a results-only PR. Since the 2026-11-01 cell (main `7f4e709`), two changes reached the prefill, both with the
target's outputs bitwise the same:

- **#941:** the prefill on a grouped encoder.
- **#946:** DFlash2's prompt commit on the tensor tile.

Other changes reached the decode, among them #911 (the tree verify attention with its K / V tiles staged).

Kapitsa only: no other engine ran, and the other engines' cells keep their dates. The label, model folder and harness
phase are the 2026-11-01 cell's ([record](../2026-11-01-m5-max-kapitsa/README.md)):
`compare_engines.py configs/kapitsa.json --only q36-35b-kapitsa --agentic --skip tools,agent,cache,sustain`, three runs
in a row from 16:43 to 16:49 local time ([run_cell.sh](run_cell.sh), [session.log](session.log)).

MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), on AC power in High Power
mode, the session idle, under the GPU lock.

**A cell** is the median of its runs' medians, with the runs' range: decode a 512-token coding answer, five answers per
run; prefill one cold prompt per size per run.

## The cell, old against new

| | 2026-11-01, main `7f4e709` | 2026-11-04, main `3879b44` | Change |
| --- | ---: | ---: | ---: |
| Prefill 4K | 6,036 (6,016-6,093) | **6,137** (6,133-6,151) | +1.7% |
| Prefill 16K | 5,298 (5,294-5,309) | 5,398 (5,375-5,411) | +1.9% |
| Decode T 0.6 | 346.0 (341.4-349.2) | **347.1** (341.5-359.9) | +0.3% |
| Greedy (T 0) | 371.1 (360.2-378.4) | 376.2 (367.1-376.4) | +1.4% |
| Cached follow-up 4K / 16K (s) | 0.04 / 0.05 | 0.04 / 0.05 | |
| Peak to 16K / with the 32K sweep (GiB) | 22.0 / 22.5 | 22.0 / 22.7 | |
| Flags | contention: prefill, sweep | one decode loop (below) | |

Runs ([results/](results), [logs/](logs)):

| Run | Decode T 0.6 | Greedy | Prefill 4K | Prefill 16K |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 341.5 | 376.4 | 6,137 | 5,375 |
| 2 | 359.9 | 367.1 | 6,133 | 5,398 |
| 3 | 347.1 | 376.2 | 6,151 | 5,411 |

- **Contention:** no phase of any run is contention-flagged. Other processes averaged 0.27% of the GPU in the worst
  phase.
- **Run 2's loop flag:** one of its T 0.6 answers fell into a repetition loop (distinct 4-grams 0.46), which flatters
  a speculative decoder. The 2026-11-01 method keeps a flagged run in the median and says so. Without run 2, the
  decode median is 344.3.
- **The 4K prefill:** it moved past the old cell's range. That cell is from another session, so the comparison is not
  interleaved.

The interleaved A/Bs of #941 (-2.1% GPU time) and #946 (-1.9% of a 4K prompt, which includes the drafter's commit)
measured the two changes alone. Through the server, the harness's 4K time to first token reads +1.7% tok/s.

## From 1K to 32K context

The context sweep in the same runs: two cold requests per size per run, each answering a 192-token question about the
document; the medians over the runs.

| `q36-35b-kapitsa` | 1K | 2K | 4K | 8K | 16K | 32K (max) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill tok/s, 2026-11-01 | 4,662 | 5,611 | 6,013 | 5,493 | 4,900 | 4,015 |
| Prefill tok/s, 2026-11-04 | 4,651 | 5,773 | 5,906 | 5,548 | 4,963 | 4,084 |
| Decode tok/s, 2026-11-01 | 268.4 | 262.1 | 257.3 | 239.7 | 218.3 | 174.9 |
| Decode tok/s, 2026-11-04 | 269.2 | 269.4 | 268.2 | 257.8 | 236.8 | 219.8 |

The 32K point is 31,550 prompt tokens in both. The long-context decode gain (+26% at 32K) comes from the decode changes
since `7f4e709`, #911 among them, and is not attributed further here. The README's context charts carry the longer
sweep of 2026-11-03.
