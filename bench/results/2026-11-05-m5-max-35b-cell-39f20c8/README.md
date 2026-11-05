# Results, 2026-11-05: Qwen3.6 35B-A3B's M5 Max cell on main 39f20c8 (the GPU prefill wavefront)

Kapitsa's Qwen3.6 35B-A3B cell on the MacBook Pro M5 Max, re-measured on main `39f20c8`, after #975. On the 40-core M5
Max, #975 runs the single-slot 35B-A3B's prefill in 8,448-row chunks as a two-half wavefront on the GPU alone
(`NI_PF_WAVE`, [its results](../2026-11-05-m5-max-35b-gpu-wave/README.md)).

Kapitsa only: the other engines' cells keep their dates. The label, model folder and harness phase are the earlier
cells': `compare_engines.py configs/kapitsa.json --only q36-35b-kapitsa --agentic --skip tools,agent,cache,sustain`.
Three runs, 120 s idle before each, after 5 minutes idle, the session idle ([run_cells.sh](run_cells.sh),
[session.log](session.log)).

MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), on AC power in High Power
mode, under the GPU lock.

## The cell, old against new

| | 2026-11-05, `7afb51d` ([cells](../2026-11-05-m5-max-qwen-cells-87d7dd5/README.md)) | 2026-11-05, `39f20c8` |
| --- | ---: | ---: |
| Prefill 4K | 6,619 (6,615-6,642) | 6,539 (6,534-6,546) |
| Prefill 16K | 5,718 (5,718-5,737) | **6,046** (6,041-6,052), +5.7% |
| Decode T 0.6 | 349.4 (345.2-358.2) | 351.0 (326.1-353.3) |
| Greedy (T 0) | 377.3 (369.4-381.7) | 368.5 (368.4-375.1) |
| Cached follow-up 4K / 16K (s) | 0.04 / 0.05 | 0.04 / 0.05 |
| Peak to 16K / with the 32K sweep (GiB) | 22.3 / 22.6 | 23.3 / 23.9 |

- **16K:** prefill moves past the earlier cell's range, +5.7%, as the wavefront's interleaved A/B measured.
- **4K:** the prompt is one chunk either way, and the wavefront leaves it level. The old and new builds read
  6,516-6,518 and 6,553-6,557 at 4K when interleaved ([cell-ab](../2026-11-05-m5-max-35b-gpu-wave/cell-ab)). The 1.2%
  here against the morning's cell is the spread between sessions.
- **Decode** runs no prefill kernel. T 0.6 reads 351.0 against 349.4; greedy reads 368.5 against 377.3, inside the overlapping
  run ranges (368.4-375.1 against 369.4-381.7).
- **Memory:** 1.0 GiB more, the 8,448-row chunks' activation buffers.
- **Contention:** no phase is flagged; the desktop app's GPU load was 0.1% or less.

## Runs

| Run | Decode T 0.6 | Greedy | Prefill 4K | Prefill 16K |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 326.1 | 368.5 | 6,539 | 6,041 |
| 2 | 353.3 | 368.4 | 6,546 | 6,052 |
| 3 | 351.0 | 375.1 | 6,534 | 6,046 |

**A first set** of the same build ([hot-runs/](hot-runs)) ran with 60 s gaps after a day of back-to-back GPU work:
- decode 351, 282 and 327;
- 16K 6,021, 5,599 and 5,788;
- no contention flagged.

Its later runs read the machine's state, not the build: decode runs no prefill code. After 5 minutes idle and with
120 s gaps, the set above is steady.
