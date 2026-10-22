# Results, 2026-10-22: MacBook Pro M5 Pro, 48 GB (Splash 1.4.0 against Kapitsa)

The M5 Pro's panel on the README poster had no Splash cells. These are Splash 1.4.0's runs on the two models it runs,
Qwen3.8 27B and Qwen3.6 35B-A3B (each with its DFlash2 drafter), with Kapitsa first and last around them on each model
(ABBA: Kapitsa, Splash, Splash, Kapitsa), so both engines were measured in the same session under the same desktop load.

Written by [bench/harness](../../harness/README.md): `compare_engines.py configs/kapitsa.json configs/splash.json --only <config> --skip tools,agent`,
one config at a time, each model's four runs in one hold of the GPU lock (`lockf -k /tmp/kapitsa-gpu.lock`), 15:00-15:20 (27B) and
15:20-15:29 (35B-A3B) local time. MacBook Pro (16-inch, 2026), `Mac17,8`, M5 Pro, 20-core GPU, 48 GB, macOS 26.5.2 (25F84), on AC power.
Kapitsa 0.2.6, main `4b89f4e` ([#293](https://github.com/nace-ai/kapitsa/pull/293): the 35B's prefill attention on the tensor units),
Release build, its defaults (KFlash with the DFlash2 drafter, `--ctx 32768 --slots 1`). Splash 1.4.0, the release tarball
`splash-1.4.0-arm64-macos26.tar.gz` (SHA-256 checked against the release's `SHA256SUMS`), unpacked and named with `SPLASH_ROOT`,
its defaults as `configs/splash.json` serves it (DFlash2 drafter, INT8 KV, `--max-context 32K`; the 27B's FFN split between the
GPU and the Neural Engine; the 35B with `--expert-memory off`).

| Result file | Model | Engine | Prefill 4K / 16K (tok/s) | Decode T = 0.6, median (range) / greedy (tok/s) | Peak footprint (GB) | Flags |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `q38-27b-kapitsa-20261022-150053` | Qwen3.8 27B 4-bit | Kapitsa | 429 / 446 | 74.2 (70.7-79.6) / 83.5 | 20.3 | contention in the 16K prefill only (5.1% mean, Microsoft Edge) |
| `q38-27b-splash-20261022-150833` | Qwen3.8 27B 4-bit | Splash 1.4.0 | 670 / 652 | 72.0 (59.1-74.5) / 69.8 | 23.0 | contention (10.1% mean, 21.1% max, Microsoft Edge) |
| `q38-27b-splash-20261022-151346` | Qwen3.8 27B 4-bit | Splash 1.4.0 | 678 / 584 | 72.5 (64.0-80.2) / 71.8 | 22.9 | contention (9.6% mean, 21.7% max, Microsoft Edge) |
| `q38-27b-kapitsa-20261022-151716` | Qwen3.8 27B 4-bit | Kapitsa | 472 / 448 | 76.7 (68.1-83.8) / 83.5 | 20.3 | contention (7.8% mean, 15.4% max, Microsoft Edge) |
| `q36-35b-kapitsa-20261022-152033` | Qwen3.6 35B-A3B 4-bit | Kapitsa | 2,526 / 2,043 | 209.2 (193.2-234.4) / 208.7 | 21.2 | contention (6.9% mean, 14.9% max, Microsoft Edge) |
| `q36-35b-splash-20261022-152205` | Qwen3.6 35B-A3B 4-bit | Splash 1.4.0 | 3,495 / 3,067 | 177.7 (167.6-191.3) / 185.0 | 22.4 | contention (7.4% mean, 14.1% max, Microsoft Edge); repetition loop: decode |
| `q36-35b-splash-20261022-152550` | Qwen3.6 35B-A3B 4-bit | Splash 1.4.0 | 3,532 / 3,114 | 180.5 (166.4-217.0) / 197.2 | 22.5 | contention (6.5% mean, 11.7% max, Microsoft Edge) |
| `q36-35b-kapitsa-20261022-152731` | Qwen3.6 35B-A3B 4-bit | Kapitsa | 2,552 / 2,056 | 208.9 (199.3-216.3) / 213.1 | 21.2 | none |

Means of each engine's two runs:

| Model | Engine | Prefill 4K / 16K (tok/s) | Decode T = 0.6 / greedy (tok/s) |
| --- | --- | ---: | ---: |
| Qwen3.8 27B | Kapitsa | 450 / 447 | 75.5 / 83.5 |
| Qwen3.8 27B | Splash 1.4.0 | 674 / 618 | 72.2 / 70.8 |
| Qwen3.6 35B-A3B | Kapitsa | 2,539 / 2,050 | 209.0 / 210.9 |
| Qwen3.6 35B-A3B | Splash 1.4.0 | 3,514 / 3,091 | 179.1 / 191.1 |

**Splash reads prompts faster; Kapitsa writes faster.** Splash leads prefill on both models: 1.50x Kapitsa's at 4K on
the 27B (674 against 450 tok/s; its long prompts split between the GPU and the Neural Engine) and 1.38x on the 35B-A3B
(3,514 against 2,539). Kapitsa leads decode on both: 5% at T = 0.6 on the 27B (75.5 against 72.2) and 17% on the
35B-A3B (209.0 against 179.1); greedy, 18% and 10%. The M5 Max's 2026-10-22 runs read the same way
([results](../2026-10-22-m5-max/README.md)).

**Contention.** Microsoft Edge, open on the desktop, held 5-10% of the GPU on average through the session (peaks of
15-22%), so the harness's quiet-window wait could not be met. The first run waited out the default 300 s and then ran;
the other seven were started after a 30 s wait (`--quiet-timeout 30`, a local default for this session, not part of the
harness in this repo; the seven later records' `harness.dirty` is true for that one line). Every Splash run is contention-flagged in
every phase. Kapitsa's 35B-A3B closing run carries no flag at all, and its 27B opening run is flagged only in the 16K
prefill. Both engines' runs on a model saw the same desktop, in alternating order, so the comparison holds; read
the flagged numbers as lower bounds.

**Repetition loop.** Splash's first 35B-A3B run has one decode answer in a repetition loop (distinct 4-grams 0.44, still
a valid run); a loop flatters a speculative decoder, so that median reads high, if anything.

**Kapitsa's 35B-A3B cell.** The closing run, flag-free, is the newest Kapitsa 35B-A3B run without a contention or invalid
flag, so under the poster's rule it is the cell now: 2,552 / 209 tok/s, on `4b89f4e`. It replaces the `d53f02e` run
(2,459 / 217, 2026-10-22, before #293), which #301 kept because its own `4b89f4e` run was contention-flagged
(2,548 / 201). Three runs of `4b89f4e` now read 201-209 tok/s decode against `d53f02e`'s 217; one run per build, not an
interleaved A/B, so the gap is not attributed to a change here.

The server logs are in `logs/`. Local paths are replaced by placeholders: `~` for the home folder, `<repo>` for the
Kapitsa checkout, `<run-root>` for the run directory.
