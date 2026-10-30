# Flash-Next 180B Q2_0 on the M5 Pro: Kapitsa on Splash 1.5.0 with KFlash against plain Splash 1.5.0 (2026-10-30)

A re-measure of the M5 Pro's Flash-Next 180B Q2 row of the README poster, which came from the
[2026-10-27 four-engine run](../2026-10-28-m5-pro-flash-next-engines/README.md) on Kapitsa 0.2.23 and Splash 1.4.0.
The same scripts ran again with only the arms and paths changed ([scripts/](scripts/); the diff against the 2026-10-27
copies is in the PR).

**Result: a tie on every cell.** The rule was written down before the first leg: Kapitsa wins a cell only if its slowest
request beats Splash's fastest. On every cell the two ranges overlap.

| | Kapitsa `--engine splash` (KFlash 1.5.0) | plain Splash 1.5.0 |
| --- | --- | --- |
| Decode, T 0, 256 tokens (tok/s) | 61.3 (53.7-64.7) | 56.5 (51.9-60.2) |
| Decode, T 0.6, 256 tokens (tok/s) | 58.2 (56.7-61.5) | 56.0 (52.7-76.4) |
| Prefill, 1,035-token prompt (tok/s) | 704 (657-718) | 706 (651-711) |
| Prefill, 4,074-token prompt (tok/s) | 1,218 (976-1,269) | 1,178 (1,040-1,262) |
| Prefill, 19,286-token prompt (tok/s) | 1,194 (1,133-1,216) | 1,190 (1,137-1,218) |
| Turn-2 TTFT of a 4K tool-call chat (s) | 0.56 (0.54-0.61) | 0.56 (0.54-0.76) |
| Peak footprint of the serving tree (GiB) | 25.0 | 25.4 |
| Swap growth, memory pressure (max) | 0 GB, level 1 | 0 GB, level 1 |

Medians of 10 requests per engine (5 legs × 2), with min-max in brackets. Prefill is prompt tokens over time to first
token, on a cold prompt (a nonce at the start of each one, 0 cached tokens on every request). Decode is counted from the
first token to the last. The full table is [legs/table.md](legs/table.md), and every request is in
[legs/summary.json](legs/summary.json) and the per-leg JSONs.

## What ran

- **Mac:** MacBook Pro M5 Pro, 48 GB, macOS 26.5.2 (25F84), on AC power, one leg at a time under
  `/tmp/kapitsa-gpu.lock`.
- **K:** `kapitsa serve --engine splash` from main `e67b764f` (kapitsa 0.2.24), at its default `--splash-kflash auto`
  and without `--splash-binary`. Splash 1.5.0 is found through `SPLASH_HOME`, and KFlash comes from `KAPITSA_HOME`'s
  `engines/splash-kflash-1.5.0-18a1058aa46c/`. That is `packaging/splash/build.py --release` on Splash 1.5.0, whose
  `engine/splash` sha256 is `d45f8ea7473e9af1…`, byte-identical to the KFlash 1.5.0 binary of
  [splash-backend.md](../../../docs/splash-backend.md#kflash). Every K log has the line
  `kapitsa: KFlash engine …/splash-kflash-1.5.0-18a1058aa46c/engine/splash`.
- **S:** the plain Splash 1.5.0 release (`engine/splash` sha256 `bf5c552bae97f987…`) through its own Python server. No S
  log mentions KFlash.
- **Both:** `--expert-memory 17179869184 --idle-release off` (16 GiB of experts in memory), every other Splash flag at its
  default (engine `--verify-rows auto` on both), context 32K, the same prompts as the 2026-10-27 run
  ([prompts/](prompts/)).
- **Order:** K S K S, 10 legs from 10:59 to 11:13 UTC. Each leg was one server start, a warm-up, then 2 repetitions. No
  leg had errors or a memory abort, and no leg was dropped.

## What this does and doesn't show

- Kapitsa's decode median is higher at T 0 (61.3 against 56.5), but the ranges overlap, so by the registered rule this
  is not a measured win. Ten requests per engine from five server starts are not enough to separate these engines.
- KFlash's own interleaved A/B (29 Oct 2026, both engines at `--expert-memory 20G`, more runs per arm) is the
  measurement of KFlash against the 1.5.0 release: decode 1.101x and 4K prefill 1.004x, and the same tokens as the release
  on 6 prompts × 22 runs ([splash-backend.md](../../../docs/splash-backend.md#kflash)). This run is the poster row's
  protocol, not that A/B.
- Output text was not compared between K and S in this run.
- The row's decode is a 256-token answer at T 0, and its 4K prompt is 4,074 tokens. The rest of the poster uses a
  512-token answer at T 0.6 on a 4,133-token prompt.
- Kapitsa's native engine and llama.cpp were not re-run. Their cells stay on the 2026-10-27 record.
