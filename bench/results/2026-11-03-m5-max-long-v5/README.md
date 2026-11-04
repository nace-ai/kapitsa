# Results, 2026-11-03: MacBook Pro M5 Max, 128 GB (the 35B-A3B and the 27B after their verify changes, four requests a size)

The context sweep of [2026-11-02-m5-max-long](../2026-11-02-m5-max-long/README.md) again for Qwen3.6 35B-A3B on main
`6519dbe`, after its tree verify attention staged its K / V tiles (#911), and for Qwen3.8 27B on main `eee2adb`, after
its 8-row tree verify moved to the matrix units (#916) and its drafter took 8-row blocks for prompts of 16K tokens or
more (#917): the labels `-v5` ([sweep-configs/resweep5.json](sweep-configs/resweep5.json)), each model at its default
with the server's context at 256K, the decode cell's coding question after a cold document from the long corpus, four
cold requests a size, one run each, each on a clean Release build of that commit ([machine.json](machine.json)).
MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), AC power, High Power mode,
`caffeinate -is`, 16:17-18:07 local time.

A point's decode is the median of the clean answers at that size (a full 512-token answer that neither loops, distinct
4-grams below 0.5, nor copies the document, a decode rate above twice the model's 1K rate); the count in parentheses.
Its prefill is the median of all four requests.

| Prefill / decode, tok/s (clean answers of 4) | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K | 256K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 35B-A3B | 4,674 / 335 (4) | 5,414 / 331 (4) | 5,311 / 321 (4) | 4,493 / 316 (4) | 4,213 / 291 (4) | 3,649 / 247 (4) | 2,743 / 222 (4) | 1,842 / 197 (3) | 1,114 / 136 (4) |
| Qwen3.8 27B | 743 / 82.4 (1) | 783 / - (0) | 811 / 89.7 (2) | 873 / 74.7 (3) | 843 / 64.0 (2) | 778 / 73.3 (4) | 665 / 65.8 (2) | 511 / 57.8 (2) | 343 / 42.5 (2) |

The 27B often ends its answer early (40-142 tokens, `finish=stop`), more so at the short sizes, and none of its four 2K
answers was clean (three stopped, one looped): its 2K decode point in the README's chart stays its v3 run's (94.2 tok/s,
main `dbf303a`, [2026-11-03-m5-max-long-v3](../2026-11-03-m5-max-long-v3/README.md)). With two clean answers or fewer a
size its points swing (16K below 32K here). The 35B's one unclean answer looped at 128K.

| Decode, the chart's earlier run → now | 32K | 64K | 128K | 256K |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 35B-A3B (v2, main `a2f4210`) | 263 → 247 | 201 → 222 | 147 → 197 | 114 → 136 |
| Qwen3.8 27B (v3, main `dbf303a`) | 68.3 → 73.3 | 55.8 → 65.8 | 40.9 → 57.8 | 32.6 → 42.5 |

These are the README's context chart lines for the two models. Both runs are flagged for contention, and so were the
runs they replace, under more load: the Claude desktop app's GPU process held a mean of 9.9% of the GPU during the
35B's sweep (14.7% at most; 1.9% while the decode was timed; the 60 s quiet wait timed out with the app at 10%) and 3.1%
during the 27B's (53.8% at a sample, flagged for a sustained stretch; 0.0% while the decode was timed; quiet after
19 s), against 57.8% during the 35B's v2 sweep and 22.6% during the 27B's v3 one. So "earlier → now" compares runs
under different loads, as the prefill's moves show (the 27B's 128K: 435 → 511 tok/s, though no change touched its
prefill). The interleaved A/Bs are the measures of the changes:
[#911's](../2026-11-03-m5-max-q35-verify-stage/README.md), the 35B's decode +9.6% at 132K and +11.2% at 254K;
[#916's](../2026-11-03-m5-max-q27-tree-verify-8/README.md), the 27B's 8-row cycle +32.5% at 132K and +44.9% at 254K on
a 2-slot server; [#917's](../2026-11-03-m5-max-q27-narrow-block/README.md), that cycle on a one-slot server, +7.1% at
32K and +25.0% at 131K.

## Files

`results/`, `logs/`: the two runs and their server logs. `session/`: the runner's output (`v5.log`: the 35B,
`v5-27b.log`: the 27B) and script. `sweep-configs/resweep5.json`: the labels. [machine.json](machine.json): the builds.
Local paths are replaced by placeholders (`~`, `<repo>`: the checkout of main the sweeps ran from, `<run-root>`,
`<scratch>`).
