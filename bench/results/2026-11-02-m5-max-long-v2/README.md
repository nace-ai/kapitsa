# Results, 2026-11-02 (evening): MacBook Pro M5 Max, 128 GB (the long sweep after the matrix-unit attention)

The context sweep of [2026-11-02-m5-max-long](../2026-11-02-m5-max-long/README.md) again, on main `a2f4210`, for the five
models today's attention changes touch: the verify attention on the M5's matrix units with the context split (Nemotron,
#887; Qwen3.6 35B-A3B and Clover, #892) and Nemotron's prefill attention with its K / V tiles staged (#890). The same
labels with `-v2` ([sweep-configs/resweep.json](sweep-configs/resweep.json)), each model at its default with the server's
context at its limit, two cold requests per size, the decode cell's coding question after a document from the long corpus
(`--sweep-task code`), one run per model. The build: the sources of main `a2f4210` (built before it merged, from main with
#890 and #892; the same files). MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246),
AC power, High Power mode, `caffeinate -is`, 20:49-22:55 local time. The 27B, Flash-Next, Dory and Flint, which those
changes do not run, keep their morning points; README's context charts plot both.

| Prefill / decode, tok/s | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 35B-A3B | 4,686 / 314 | 5,619 / 336 | 5,552 / 302 | 5,218 / 299 | 4,227 / 292 | 3,387 / 263 | 2,559 / 201 | 1,652 / 147 | 987 / 114 | - |
| Clover 35B-A3B | 4,688 / 294 | 5,612 / 289 | 5,816 / 278 | 5,429 / 301 | 4,830 / 221 | 3,933 / 225 | 2,884 / 232 | 1,894 / 181 | - | - |
| Lightning 30B | 5,205 / 241 | 5,865 / 269 | 5,643 / 254 | 5,707 / 277 | 5,520 / 251 | 4,941 / 247 | 3,851 / 212 | 2,981 / 186 | 1,935 / 148 | 1,131 / 116 |
| Super-VL 120B | 956 / 97.4 | 986 / 68.3 | 1,214 / 77.8 | 1,239 / 62.0 | 1,232 / 74.5 | 1,199 / 70.7 | 1,119 / 65.6 | 975 / 64.9 | 755 / 54.1 | 546 / 44.2 |
| Bolt 74B | 916 / 142 | 1,056 / 147 | 1,090 / 145 | 1,087 / 138 | 1,074 / 148 | 1,075 / 126 | 1,017 / 107 | 906 / 109 | 746 / 91.0 | - |

Against the morning's points (decode; prefill, tok/s):

| Before → after | 32K | 128K | 256K | 512K |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 35B-A3B (decode; prefill) | 245 → 263; 3,418 → 3,387 | 128 → 147; 1,756 → 1,652 | 80.2 → 114; 995 → 987 | - |
| Clover 35B-A3B (decode; prefill) | 224 → 225; 3,853 → 3,933 | 130 → 181; 1,828 → 1,894 | - | - |
| Lightning 30B (decode; prefill) | 146 → 247; 4,645 → 4,941 | 61.1 → 186; 2,488 → 2,981 | 37.6 → 148; 1,482 → 1,935 | 21.0 → 116; 823 → 1,131 |
| Super-VL 120B (decode; prefill) | 53.0 → 70.7; 1,141 → 1,199 | 38.6 → 64.9; 859 → 975 | 24.1 → 54.1; 580 → 755 | 16.0 → 44.2; 337 → 546 |
| Bolt 74B (decode; prefill) | 98.5 → 126; 1,040 → 1,075 | 47.5 → 109; 792 → 906 | 31.1 → 91.0; 549 → 746 | - |

A point's decode is the median of the clean answers at that size: an answer that loops (distinct 4-grams below 0.5) or
copies the document (a decode rate above twice the model's 1K rate: Super-VL 953 tok/s at 2K, Bolt 1,119 and 728 at 4K) is
left out. Super-VL's 1K-2K and Bolt's 1K-4K, where no answer was clean (Super-VL at 1K: 0.15 and 0.37; Bolt at 1K: 0.24 and
0.15, at 2K: 0.13 and 0.35), keep the morning's points; Super-VL's 4K, 8K, 32K and 512K and Bolt's 8K and 128K each left
one looping answer out. Prefill is the median of both requests. The short contexts move within run-to-run noise (one run;
the drafted models' tokens per cycle vary with the sampled text). The A/B of each change, interleaved on one build:
`2026-11-02-m5-max-nh-verify-t`, `2026-11-02-m5-max-nh-prefill-ts`, `2026-11-02-m5-max-qwen-verify-t`.

The Claude desktop app's GPU process, driving the session, held 45-57% of the GPU in the 35B's, Clover's and Lightning's
sweeps (the harness's worst-phase mean, contention-flagged): those points are lower than an idle Mac's would be.

## Files

`results/`, `logs/`: the five runs and their server logs. `session/`: the runner's output and script.
`sweep-configs/resweep.json`: the labels. Local paths are replaced by placeholders (`~`, `<run-root>`, `<host>`).
