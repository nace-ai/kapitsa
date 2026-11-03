# Results, 2026-11-03: MacBook Pro M5 Max, 128 GB (the 27B and Dory after their verify changes, four requests a size)

The context sweep of [2026-11-02-m5-max-long](../2026-11-02-m5-max-long/README.md) again for Qwen3.8 27B and Dory1.0 4B
on main `dbf303a`, after the 27B's wide tree verify's attention moved to the matrix units (#897) and Dory's verify
attention read the cache once for all its rows (#898): the same labels with `-v3`
([sweep-configs/resweep3.json](sweep-configs/resweep3.json)), each model at its default with the server's context at
256K, the decode cell's coding question after a cold document from the long corpus, **four** cold requests a size (the
earlier sweeps: two), one run each. MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4
(25E246), AC power, High Power mode, `caffeinate -is`, 01:04-03:48 local time.

A point's decode is the median of the clean answers at that size: a full 512-token answer that neither loops (distinct
4-grams below 0.5) nor copies the document (a decode rate above twice the model's 1K rate); the count in parentheses. Its
prefill is the median of all four requests. The 27B's DFlash2 drafter gets 3.5-6.7 tokens a cycle on these answers and
some stop before 512 tokens, so its rates swing from one answer to the next; with four requests a size the long sizes
settle (64K: all four clean).

| Prefill / decode, tok/s (clean answers of 4) | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K | 256K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 27B | 745 / 83.3 (1) | 766 / 94.2 (2) | 807 / 84.2 (2) | 777 / 72.9 (3) | 752 / 69.8 (3) | 679 / 68.3 (3) | 559 / 55.8 (4) | 435 / 40.9 (3) | 303 / 32.6 (1) |
| Dory1.0 4B | 3,104 / 125 (4) | 2,442 / 120 (4) | 2,435 / 108 (4) | 2,288 / 98.3 (4) | 2,008 / 94.4 (4) | 1,625 / 70.3 (4) | 1,151 / 65.8 (4) | 695 / 32.4 (4) | 407 / 21.9 (3) |

| Decode, morning → now | 32K | 64K | 128K | 256K |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8 27B | 74.1 → 68.3 | 49.5 → 55.8 | 34.2 → 40.9 | 27.6 → 32.6 |
| Dory1.0 4B | 70.0 → 70.3 | 51.6 → 65.8 | 35.2 → 32.4 | 20.5 → 21.9 |

These are README's context chart lines for the two models. Dory's moves within its run-to-run spread (its two morning runs
differed by up to 20%); #898's interleaved A/B (`2026-11-02-m5-max-dory-verify-rows`) is the measure of that change: a
verify cycle 9% cheaper at 133K. The 27B's interleaved A/B is `2026-11-02-m5-max-q27-verify-t` (25% at 131K).

Both runs ran under load and are flagged for contention: the Claude desktop app's GPU process held a mean of 66.7% of
the GPU during Dory's sweep (up to 105.7% at a sample) and 22.6% during the 27B's (up to 71.7%), and the 60 s quiet wait
timed out before each (60.1 s and 60.2 s, the app at 11-12%). Their points are therefore likely lower than an idle
Mac's, and the morning runs they are compared with ran under less (sweep means 0.3-16.4% for Dory, 0.2-4.0% for the
27B), so "morning → now" compares runs under different loads: Dory's small moves are not a measured change.

## Files

`results/`, `logs/`: the two runs and their server logs. `session/`: the runner's output and script.
`sweep-configs/resweep3.json`: the labels. [machine.json](machine.json): the build. Local paths are replaced by
placeholders (`~`, `<repo>`, `<run-root>`, `<host>`).
