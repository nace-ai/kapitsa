# Results, 2026-11-03: MacBook Pro M5 Max, 128 GB (Dory after its prefill attention moved to simdgroup pairs, four requests a size)

The context sweep of [2026-11-02-m5-max-long](../2026-11-02-m5-max-long/README.md) again for Dory1.0 4B, with its prefill
attention's head dim split over pairs of simdgroups (`dory_attn_td2_g4s8`, #904): the label `-d2`
([sweep-configs/resweep4.json](sweep-configs/resweep4.json)), the model at its default with the server's context at 256K,
the decode cell's coding question after a cold document from the long corpus, four cold requests a size, one run. The
build is #904's (main `32cf851` with that change), the same binary as its A/B
([2026-11-03-m5-max-dory-attn-d2](../2026-11-03-m5-max-dory-attn-d2/README.md); [machine.json](machine.json)).
MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), AC power, High Power mode,
`caffeinate -is`, 05:21-06:04 local time.

A point's decode is the median of the clean answers at that size (a full 512-token answer that neither loops, distinct
4-grams below 0.5, nor copies the document, a decode rate above twice the model's 1K rate): all four at every size here.
Its prefill is the median of all four requests.

| Prefill / decode, tok/s (clean answers of 4) | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K | 256K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dory1.0 4B | 3,307 / 123 (4) | 2,624 / 115 (4) | 2,644 / 106 (4) | 2,651 / 101 (4) | 2,424 / 88.4 (4) | 2,121 / 73.2 (4) | 1,624 / 55.8 (4) | 1,102 / 33.7 (4) | 700 / 22.7 (4) |

| Prefill, tok/s: main `dbf303a` (v3) → now | 32K | 64K | 128K | 256K |
| --- | ---: | ---: | ---: | ---: |
| Dory1.0 4B | 1,625 → 2,121 | 1,151 → 1,624 | 695 → 1,102 | 407 → 700 |

These are the README's context chart lines for Dory. The run is flagged for contention: the Claude desktop app's GPU
process held a mean of 10.6% of the GPU during the sweep (up to 60.1% at a sample) and 2.0% while the decode was timed,
and the 60 s quiet wait timed out (the app at 10%). The v3 run it is compared with ran under more (66.7% during its
sweep), so "v3 → now" compares runs under different loads; #904's interleaved A/B is the measure of the change: the
first token of a 133K-token prompt 1.46-1.48x sooner, of a 255K one 1.57x, decode unchanged. Decode here moves within
the model's run-to-run spread.

## Files

`results/`, `logs/`: the run and its server log. `session/`: the runner's output and script.
`sweep-configs/resweep4.json`: the label. [machine.json](machine.json): the build. Local paths are replaced by
placeholders (`~`, `<repo>`: #904's checkout, `<main>`, `<run-root>`).
