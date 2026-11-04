# Results, 2026-11-04: Qwen3.8-Flash-Next's M5 Max cell on main `af7d795`

Kapitsa's Flash-Next 180B-A7B cell on the MacBook Pro M5 Max, re-measured on main `af7d795` after three verify-kernel
changes that each kept the output bitwise identical: #921 (the folded gated residual on wider grids), #922 (the verify
MoE's distinct experts found by a threadgroup) and #923 (the verify MoE reading its rows' inputs through the caches).
Kapitsa only: no other engine ran, and the other engines' Flash-Next cells keep their 2026-10-18 date. The label, model
folder and harness phase are the 2026-11-01 cell's ([record](../2026-11-01-m5-max-kapitsa/README.md)):
`compare_engines.py configs/flash-next.json --only flash-kapitsa --agentic --skip tools,agent,cache,sustain`, MTP head at
depth 2 (the default), three runs in a row from 08:20 to 08:32 local time ([run_cell.sh](run_cell.sh),
[session.log](session.log)). MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246),
on AC power in High Power mode, the Mac otherwise idle, under the GPU lock ([machine.json](machine.json)).

**A cell** is the median of its runs' medians, with the runs' range: decode a 512-token coding answer, five answers per
run; prefill one cold prompt per size per run.

## The cell, old against new

| | 2026-11-01, main `7f4e709` | 2026-11-04, main `af7d795` | Change |
| --- | ---: | ---: | ---: |
| Prefill 4K | 1,701 (1,675-1,721) | **1,726** (1,557-1,738) | +1.5% |
| Prefill 16K | 1,613 (1,580-1,654) | 1,608 (1,516-1,674) | -0.3% |
| Decode T 0.6 | 108.3 (107.7-108.7) | **108.5** (105.7-113.0) | +0.2% |
| Greedy (T 0) | 107.3 (107.1-108.8) | **113.9** (111.6-115.1) | **+6.2%** |
| Cached follow-up 4K / 16K (s) | 0.09 / 0.08 | 0.08 / 0.08 | |
| Peak to 16K / with the 32K sweep (GiB) | 70.6 / 71.2 | 70.7 / 71.2 | |
| Flags | contention: prefill, sweep | none | |

Runs ([results/](results), [logs/](logs)): decode T 0.6 112.99, 105.69, 108.47; greedy 115.08, 113.94, 111.57; prefill
4K 1,557, 1,726, 1,738 and 16K 1,516, 1,674, 1,608. No phase of any run is contention-flagged: other processes' GPU
load averaged 0.04% or less in every phase.

**Greedy is where the three changes show.** Their interleaved A/Bs (`ni_run`, greedy, the same token ids in every run)
measured +2.1% (#921, re-measured idle in #927), +2.5% (#922) and +1.0% (#923), about +5.7% together; the cell's greedy median moved +6.2%
between sessions three days apart. Every one of the three changes was gated as bitwise identical, so the same prompt
writes the same tokens faster.

**At T 0.6 the gain is inside the noise.** Sampled answers take different paths, and the MTP head guesses some paths
better than others: each run's five answers spread over 10-14 tok/s, and the run medians here span 105.7-113.0. Six
earlier runs on #923's head `c90acf6` (below) spread 98.8-114.4 with a median of 110.9. Three runs cannot resolve a 6%
change at that spread, so the bold decode number stays 108, as it read on 2026-11-01 (108.3). It still stands against
mlx-serve's 107.6 from its one 2026-10-18 run, in another session and power mode and not run interleaved, so mlx-serve's
2026-10-18 lead stands as before.

**Prefill did not change**, as expected: none of the three changes touches the prefill path.

## From 1K to 32K context

The context sweep in the same runs: two cold requests per size per run, each answering a 192-token question about the
document; the medians over the runs. Its decode rate is that answer's, not the 512-token coding answer's above.

| `flash-kapitsa` | 1K | 2K | 4K | 8K | 16K | 32K (max) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill tok/s, 2026-11-01 | 1,618 | 1,591 | 1,720 | 1,589 | 1,530 | 1,463 |
| Prefill tok/s, 2026-11-04 | 1,636 | 1,611 | 1,691 | 1,598 | 1,542 | 1,471 |
| Decode tok/s, 2026-11-01 | 89.5 | 87.0 | 82.4 | 80.5 | 79.8 | 78.2 |
| Decode tok/s, 2026-11-04 | 94.0 | 87.1 | 85.1 | 82.5 | 78.6 | 79.0 |

The 32K point is 31,550 prompt tokens in both.

## Earlier the same morning

Two blocks before the cell, kept as evidence for why it was re-run after an idle hour:

- [earlier/cell-c90acf6](earlier/cell-c90acf6): six runs of the same command on #923's head `c90acf6` (main `f9e85aa`
  plus #923), 07:08-07:35. Decode T 0.6 110.9 (98.8-114.4), greedy 111.8
  (111.3-113.9), but prefill 4K 1,594 (1,522-1,743).
- [earlier/prefill-ab](earlier/prefill-ab): the 4K and 16K cold prefills only, the 2026-11-01 cell's build `7f4e709`
  against `c90acf6`, interleaved (old new, new old, old new), 07:37-07:45. 4K: round 0 old 1,813, new 1,835; round 1
  new 1,835, old 982; round 2 old 1,240, new 1,449. Where neither leg was slow the builds were level (1,813 against
  1,835); the drops came in the fourth to sixth legs, on both builds.

No run in either block is contention-flagged, and every run was on AC power in High Power mode. The swing is the
Mac's state, not the build, so the cell waited for an idle hour; its three runs prefill like the 2026-11-01 cell.
