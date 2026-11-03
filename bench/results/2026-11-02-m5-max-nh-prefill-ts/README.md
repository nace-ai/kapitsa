# Results, 2026-11-02: MacBook Pro M5 Max, 128 GB (Nemotron's prefill attention with staged K / V tiles)

The A/B of `nh_attn_ts_s8` (`src/kernels/tensor/nh_attn_t.metal`): Nemotron-H's prefill attention on the M5's matrix
units with each 32-key K / V tile staged in threadgroup memory for 8 tokens, against `nh_attn_t_s4`, which reads them per
token (`NI_NH_ATTN_TS=0`). Lightning 30B-A3B, Super-VL 120B-A12B and Bolt 74B-A9B, one build, the arms interleaved,
written by [bench/harness](../../harness/README.md): `compare_engines.py configs/<model>.json sweep-configs/ab4.json --only
<label> --agentic --skip greedy,prefill,tools,agent,cache,sustain --quiet-timeout 60`, each label the model's default with
its context at its limit (`--ctx`), a cold document of each size from the long corpus of
[2026-11-02-m5-max-long](../2026-11-02-m5-max-long/README.md) then the coding question, one request per size. MacBook Pro
(16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), AC power, High Power mode, `caffeinate -is`,
18:51-20:06 local time. The build: main `b6345da` (#887) with this change, as the pull request's branch.

## Prefill

| Model | Prompt | Old: s to the first token | New | Speedup | Prefill tok/s, old → new |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lightning 30B-A3B | 133K | 50.6, 49.3 | 41.0, 41.3 | 1.20-1.23x | 2,670 → 3,242 |
| Lightning 30B-A3B | 255K | 164.8, 157.7 | 118.6, 119.9 | 1.32-1.39x | 1,581 → 2,136 |
| Lightning 30B-A3B | 512K | 580.3 | 414.1 | 1.40-1.40x | 882 → 1,236 |
| Super-VL 120B-A12B | 133K | 173.1, 167.8 | 132.0, 132.1 | 1.27-1.31x | 783 → 1,010 |
| Bolt 74B-A9B | 133K | 171.4, 177.3 | 142.8, 142.6 | 1.20-1.24x | 765 → 935 |

The 512K pair ran once each (new, then old). Decode is the same (the change is the prefill's).

## Superseded

`superseded/`: the first A/B (18:07-18:50), with the staged kernel in ~25 ms pieces, as `nh_attn_t_s4` runs: 133K 53.5 /
52.3 s old against 44.6 / 44.7 new, 255K 172.3 / 166.3 against 130.5 / 132.0, but 512K 602.3 against 592.4 (2%): at 512K a
25 ms piece holds 16 row blocks, too few threadgroups to fill the GPU, so the change sizes the staged kernel's pieces at
~200 ms. Its 512K old run overlapped a CPU build (the second A/B's 580.3 s is the cleaner one).

## Output check

`logit_dump` of main `b6345da` against this branch's on Lightning, Super-VL and Bolt (`output-check/`): every row bitwise
identical on every path (prefill, step, verify, batch).

## Contention

The Claude desktop app's GPU process, driving the session, held up to ~10% of the GPU (the harness's worst-phase mean), and
the 60 s quiet wait timed out before most runs; both arms ran under it, interleaved.

## Files

`results/`, `logs/`: the A/B runs (`*-pf-old`: `NI_NH_ATTN_TS=0`; `*-pf-new`: the default) and their server logs.
`superseded/`: the 25 ms-piece A/B. `session/`: the runner's output and script. `sweep-configs/ab4.json`: the labels.
[machine.json](machine.json): the Mac and the build. Local paths are replaced by placeholders: `~`, `<repo>` for this
checkout, `<main>` for main's, `<run-root>` for the harness's output folder, `<host>` for the host name.
