# Results, 2026-11-02/03: MacBook Pro M5 Max, 128 GB (Dory's verify attention, all rows in one threadgroup)

The A/B and the output check of `dory_attn_partial_rows` (`src/kernels/dory/dory.metal`): Dory's verify attention with
one threadgroup per (chunk, KV head) for all the verify's rows, against `dory_attn_partial`, a threadgroup per row
(`NI_DORY_ATTN_ROWS=0`). Dory1.0 4B (NVFP4) at its default (2 loops, per-loop KV, self-drafts), `--ctx 262144`, the
coding answer after a cold document from the long corpus, one request per size, the two arms interleaved on one build
(`compare_engines.py configs/dory.json sweep-configs/ab8.json --only <label> --agentic --skip
greedy,prefill,tools,agent,cache,sustain --quiet-timeout 60`). MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core
GPU, 128 GB, macOS 26.4 (25E246), AC power, High Power mode, `caffeinate -is`, 23:40-00:00 local time.

## Decode

From the server logs (tok/s; ms per verify cycle):

| Context | Old | New |
| --- | ---: | ---: |
| 33K | 71.1 (31.4), 75.2 (31.0) | 82.2 (29.3), 77.9 (30.1) |
| 133K | 31.5 (74.4), 31.9 (68.5) | 38.1 (63.9), 34.8 (66.4) |

A cycle costs 4% less at 33K and 9% less at 133K on average (the verify's cache read is now one for its rows; the two
self-drafts of each cycle still read the cache a row at a time). Prefill is the same.

## Output check

`logit_dump --paths step,verify --loop-kv per-loop` of main against this branch: every row bitwise identical on every
path ([output-check.txt](output-check.txt)), as the kernel runs each row's per-key update in the same order. The verify
path under `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1` (one window): no errors ([validation.txt](validation.txt)).

## Files

`results/`, `logs/`: the runs (`dory-rows-old`: `NI_DORY_ATTN_ROWS=0`; `dory-rows-new`: the default) and their server
logs. `session/`, `sweep-configs/ab8.json`: the runner and the labels. Local paths are replaced by placeholders.
