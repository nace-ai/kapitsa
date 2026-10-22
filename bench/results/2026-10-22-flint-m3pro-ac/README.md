# M3 Pro, 36 GB, on AC power: Recursive Flint 40B-A2.5B on Kapitsa and llama.cpp, 2026-10-22

**These numbers supersede the battery run of [2026-10-22-flint-m3pro](../2026-10-22-flint-m3pro/README.md)
([#306](https://github.com/nace-ai/kapitsa/pull/306)).** That run had the Mac on battery throughout, against the
harness README's advice. This is the same protocol and the same context sweep on the same Mac, now on AC power, with
Kapitsa built from main at `cff1b1e`. The result is the same: every decode, prefill and memory number of the clean pass is
within 1% of the battery run's (the follow-up TTFTs differ by one 10 ms step), so on this Mac, with Low Power Mode off, battery power did not slow either engine. The #306 folder
stays as recorded.

Flint 40B-A2.5B is Recursive's coding MoE (256 experts, 6 routed and a shared one a token, 2.5B active). The engines:

- Kapitsa on Recursive's 4-bit MLX export
  ([RecursiveSI/flint-int4](https://huggingface.co/RecursiveSI/flint-int4) @ 3b93f905), built from main at `cff1b1e`
  ([#328](https://github.com/nace-ai/kapitsa/pull/328)). On macOS 14 Kapitsa still runs Flint's three last-threadgroup
  merges as separate dispatches ([#303](https://github.com/nace-ai/kapitsa/pull/303)), because the fused kernels need
  Metal 3.2 (macOS 15). **These numbers are that macOS 14 path.**
- llama.cpp b11804 (`c7fc50941`, the release's macOS arm64 build), on Recursive's Q4_1 GGUF
  ([RecursiveSI/flint-int4-gguf](https://huggingface.co/RecursiveSI/flint-int4-gguf) @ 9c471510), as in #306.
- **rec 0.1.0 does not run here**, as in #306. Its release binary refuses this Mac before loading anything:
  `rec: this model requires 48 GiB of memory; detected 36 GiB` (`logs/flint-rec-20261022-212012.log`, stored as an
  invalid record in `results/`).

There are two measurements, as in #306. **The poster's protocol**
(`bench/harness/compare_engines.py configs/flint.json --skip agent`, the harness of `cff1b1e`) ran Kapitsa and
llama.cpp in alternation, twice in each of two passes. **A context sweep** then measured Kapitsa from 1K to 128K tokens
of prompt, and llama.cpp from 1K to 64K. Every run uses `bench.py --always-reasons`, because Flint reasons before every
answer. Decode samples at the model card's temperature 1.0 (top_p 0.95, top_k 20). Times are this Mac's clock (UTC+6).

## The poster's protocol

**The first pass is not a clean measurement.** On AC power this Mac's display never sleeps (`displaysleep 0`), and its
desktop runs macOS 14's animated aerial wallpaper. While the first pass ran, `ps` showed the wallpaper's video decoder
(`VTDecoderXPCService`, for `WallpaperVideoExtension`) at about 148% CPU, with WindowServer busy too. Both engines came
out 3-9% below the battery run. The harness's contention sampler watches foreign GPU load, so it flagged nothing: this
load was on the CPU and the media engine. `pmset displaysleepnow` stopped it. The second pass slept the display before
each run and logged the decoder's CPU (0.0 every time, `logs/driver-clean-pass.log`). **The second pass is the
measurement.**

| Pass | Engine | Run | Decode T=1.0 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Follow-up TTFT 4K / 16K (s) | Peak (GB) | Foreign GPU load |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 (clean) | Kapitsa | `flint-kapitsa-20261022-214157` | 73.38 (72.86-74.57) | 74.53 | 800 / 680 | 0.16 / 0.19 | 23.04 | 0% |
| 2 (clean) | llama.cpp b11804 | `flint-llamacpp-20261022-214530` | 50.88 (50.75-51.31) | 50.91 | 771 / 645 | 0.36 / 0.46 | 26.56 | 0% |
| 2 (clean) | Kapitsa | `flint-kapitsa-20261022-215002` | 73.83 (72.91-74.28) | 74.53 | 801 / 664 | 0.15 / 0.19 | 23.03 | 0% |
| 2 (clean) | llama.cpp b11804 | `flint-llamacpp-20261022-215331` | 50.85 (50.49-51.05) | 51.08 | 773 / 645 | 0.39 / 0.53 | 26.56 | 0% |
| | **Kapitsa mean** | | **73.61** | **74.53** | **801 / 672** | **0.16 / 0.19** | **23.04** | |
| | **llama.cpp b11804 mean** | | **50.87** | **51.00** | **772 / 645** | **0.37 / 0.50** | **26.56** | |
| 1 (wallpaper) | Kapitsa | `flint-kapitsa-20261022-210349` | 70.72 (68.29-71.67) | 70.47 | 772 / 645 | 0.15 / 0.19 | 23.05 | 0% |
| 1 (wallpaper) | llama.cpp b11804 | `flint-llamacpp-20261022-210728` | 46.60 (46.08-47.05) | 47.45 | 740 / 620 | 0.29 / 0.58 | 26.56 | 0% |
| 1 (wallpaper) | Kapitsa | `flint-kapitsa-20261022-211217` | 68.91 (68.70-70.07) | 70.48 | 773 / 648 | 0.15 / 0.19 | 23.03 | 0% |
| 1 (wallpaper) | llama.cpp b11804 | `flint-llamacpp-20261022-211550` | 47.29 (47.12-47.49) | 47.72 | 741 / 634 | 0.47 / 0.43 | 26.56 | 0% |
| 1 | rec 0.1.0 | `flint-rec-20261022-212012` | × | × | × | × | – | refused: needs 48 GiB |

The runs are in `results/` and their server logs in `logs/`, with each pass's driver log and script
(`driver-first-pass.log` / `run.sh`, `driver-clean-pass.log` / `run2.sh`). Decode is the median of 5 runs of 512
tokens, with the range in brackets. Prefill uses the harness's 4K and 16K prompts (3,874 and 15,066 of Flint's tokens).
Follow-up TTFT is the median time to the first token of a cached second turn. The means are over the clean pass.

- **Against the battery run (#306): decode, prefill and memory the same within 1%.** Kapitsa decodes 73.61 against 73.75 tok/s at T 1.0 and
  74.53 against 74.47 greedy, and reads 801 / 672 against 799 / 674 tok/s. llama.cpp decodes 50.87 against 51.00 and
  51.00 against 51.18, and reads 772 / 645 against 771 / 640. The two runs are not interleaved and Kapitsa's builds
  differ (`cff1b1e` here, `3da3798` there).
- **Prefill.** Kapitsa reads prompts 1.04x as fast as llama.cpp at 4K (801 against 772 tok/s) and at 16K (672 against
  645).
- **Decode.** Kapitsa decodes 1.45x as fast as llama.cpp: 73.6 against 50.9 tok/s at T 1.0, and greedy 74.5 against
  51.0.
- **Follow-up turns.** Kapitsa starts a cached second turn 2.4x sooner than llama.cpp at 4K (0.16 against 0.37 s) and
  2.6x sooner at 16K (0.19 against 0.50 s).
- **Memory.** Kapitsa peaks at 23.0 GB and llama.cpp at 26.6. This Mac's recommended GPU working set is 29.0 GB.
- **Against the M4 Pro** ([its run](../2026-10-22-flint-m4pro/README.md)), as in #306: this Mac reads 4K prompts at
  83% of the M4 Pro's speed (Kapitsa 801 against 968) and decodes at 58% (73.6 against 127.9). llama.cpp reaches 79%
  and 64% (772 against 977, 50.9 against 79.6).

Flags, none of which invalidate a run:

- `follow-up returned no content; stopping the cached series` (three clean-pass runs and one first-pass run). In each,
  the third cached follow-up spent its whole token budget reasoning (`finish=length`) and showed no text. Its first
  token still arrived and was timed, so every cached series has its 3 samples and the medians above use all of them.
- `pages swapped out during the phase` on the first pass's llama.cpp prefills (1,640 pages at 4K in
  `flint-llamacpp-20261022-210728`, 1,220 at 16K in `-211550`). The clean pass has none.

## Context sweep: 1K to 128K tokens

| Prefill (tok/s) | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Kapitsa** | **653** | **751** | **742** | **705** | **630** | **516** | **377** | **245** |
| llama.cpp b11804 | 527 | 786 | 768 | 723 | 633 | 502 | 343 | – |
| <sub>Kapitsa ÷ llama.cpp</sub> | <sub>1.24×</sub> | <sub>0.95×</sub> | <sub>0.97×</sub> | <sub>0.98×</sub> | <sub>1.00×</sub> | <sub>1.03×</sub> | <sub>1.10×</sub> | <sub>–</sub> |

| Decode (tok/s) | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Kapitsa** | **71.6** | **70.0** | **68.1** | **64.5** | **58.1** | **48.9** | **36.9** | **24.8** |
| llama.cpp b11804 | 51.7 | 50.8 | 49.5 | 47.3 | 43.5 | 37.6 | 25.3 | – |
| <sub>Kapitsa ÷ llama.cpp</sub> | <sub>1.38×</sub> | <sub>1.38×</sub> | <sub>1.38×</sub> | <sub>1.36×</sub> | <sub>1.34×</sub> | <sub>1.30×</sub> | <sub>1.46×</sub> | <sub>–</sub> |

- **Prefill.** From 2K to 16K, llama.cpp reads 0-5% faster than Kapitsa. Kapitsa leads by 1.03x at 32K and 1.10x at
  64K. Kapitsa's 1.24x at 1K is a single request against llama.cpp's slowest point, so don't read much into it. Only
  Kapitsa reached 128K, at 245 tok/s: 131,018 tokens read in 8.9 minutes, 33% of its 4K speed.
- **Decode.** Kapitsa decodes 1.30-1.46x as fast as llama.cpp at every length both reached. At 128K it decodes
  24.8 tok/s, 36% of its 4K speed.
- **Against the battery sweep (#306)** the points agree within 2%, except llama.cpp's 64K decode (25.3 here,
  27.6 there). The sizes differ (powers of two here, round thousands there), so each column here is 2.4% longer.
- **Kapitsa's sweep prefill is still lower than its protocol prefill** (742 against 801 tok/s at 4K), while
  llama.cpp's agree (768 against 772), as in #306. The sweep's `--ctx 262144` again made Kapitsa choose 1024-row
  prefill chunks instead of 2048 (`memory: 1024-row prefill chunks (2048 would leave less than one 262144-token
  conversation in the 27.8 GB budget)`, `context-sweep/logs/kapitsa.log`). That is probably part of the gap, but it
  was not measured: no sweep ran at `--ctx 32768`.

How it ran (`context-sweep/`): `ctx_sweep.py` and `sweep_engine.sh` are #306's, unchanged except for this run's paths.
Each server was started with a long context, Kapitsa `--ctx 262144` and llama.cpp `-c 70000` (`-c 140000` does not fit
here, see #306). One engine ran at a time, each in its own hold of the GPU lock, with the display put to sleep first,
and each was warmed by one 2K request. The sizes are powers of two (1,024 to 131,072 tokens), as on the
[M4 Pro](../2026-10-22-flint-m4pro/README.md#context-sweep-1k-to-128k-tokens). The source-file part of the prompt was
read from this Mac's checkout of `cff1b1e`, 1,688,187 tokens for both engines. Each point is a single request with no
repeats, so read a point as about ±3%. The points and each server's command are in `context-sweep/sweep.json`, and the
servers' logs and the driver's are in `context-sweep/logs/`.

**A first sweep was discarded.** It ran 21:20-21:41 while this Mac was downloading models in the background (`kapitsa
pull` under `nice`, 21:19-21:40). The table is the rerun, 21:57-22:18, with nothing else running. The first sweep's
points are kept in `sweep.json` (`first_attempt`). They are within 1.5% of the rerun's except llama.cpp's decode at 1K
(49.4 against 51.7) and 64K (26.2 against 25.3).

## Kapitsa's builds: #306's, this run's and main after it

Main has moved on since `cff1b1e`, and some of the commits touch Flint's code (among them
[#324](https://github.com/nace-ai/kapitsa/pull/324), Flint's GGUF IQ experts). So after the sweep the protocol ran
Kapitsa alone (`--only flint-kapitsa`) with three builds, interleaved A B C A B C, 22:22-22:43, display asleep:
A = `3da3798` (#306's binary, sha256 `0bd51b3833f7…`), B = `cff1b1e` (this run's), C = `a97e596` (main at 22:20,
which reports `kapitsa 0.2.8 (a97e596)`).

| Build | Run | Decode T=1.0 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Follow-up TTFT 4K / 16K (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| A `3da3798` | `flint-kapitsa-20261022-222230` | 74.11 (73.48-74.99) | 74.64 | 801 / 680 | 0.15 / 0.18 |
| B `cff1b1e` | `flint-kapitsa-20261022-222602` | 73.86 (73.20-74.45) | 74.38 | 798 / 670 | 0.16 / 0.19 |
| C `a97e596` | `flint-kapitsa-20261022-222928` | 74.03 (73.61-74.33) | 74.32 | 800 / 669 | 0.15 / 0.19 |
| A `3da3798` | `flint-kapitsa-20261022-223301` | 74.20 (73.21-74.42) | 74.52 | 800 / 680 | 0.16 / 0.19 |
| B `cff1b1e` | `flint-kapitsa-20261022-223626` | 74.06 (73.69-74.42) | 74.40 | 800 / 667 | 0.16 / 0.18 |
| C `a97e596` | `flint-kapitsa-20261022-223949` | 73.76 (72.93-74.26) | 74.62 | 800 / 675 | 0.15 / 0.19 |

Decode, greedy decode, 4K prefill and follow-ups are the same for the three builds within 0.5%. At 16K, A read 680
tok/s in both runs, B 670 and 667, C 669 and 675: B and C about 1-2% below A, a gap no larger than the 680 / 664 that
B's two clean-pass runs above spread over. These are speed runs only; no `logit_dump` was compared. The runs and
logs are in `ab/` (`ab/logs/driver.log`, `ab/logs/run4.sh`).

## Conditions

MacBook Pro (16-inch, Nov 2023), `Mac15,7`, M3 Pro (12-core CPU, 18-core GPU), 36 GB, macOS 14.1 (23B2073). **On AC
power throughout, charged to 100%**, Low Power Mode off, every step under `caffeinate -i`. First pass 21:03-21:20, clean
pass 21:41-21:57, sweep 21:57-22:18, build A/B 22:22-22:43. 0.5-0.6 GB of swap was in use at the first pass's
start and 1.2-1.4 GB at the clean pass's and the A/B's. The GPU lock `/tmp/kapitsa-gpu.lock` was held per engine.

**Contention.** The harness's sampler saw 0% foreign GPU load in every phase of every run. It did not see the first
pass's wallpaper decoder (above), which was found with `ps`, not by the sampler. The context sweep has no contention
sampler; its driver log records that the display was put to sleep before each engine.

**Kapitsa** ran one build in every run and in the sweep: main at `cff1b1e`, its source clean, with `--ctx 32768
--slots 1` and plain decoding in the protocol.

**llama.cpp b11804** ran `llama-server` with the flags of the M4 Pro run and #306: its model card's flags, except
`-b 2048 -ub 2048` instead of the card's `-b 256 -ub 128`, and `-c 32768 -np 1` like Kapitsa.

**Not measured:** rec (it refuses 36 GB Macs), Ollama and LM Studio (neither has Flint here), llama.cpp beyond 64K,
256K context, and macOS 15 on this Mac, where Kapitsa would run its fused kernels.
