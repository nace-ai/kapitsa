# M4 Pro, 48 GB: Recursive Flint 40B-A2.5B on Kapitsa, rec and llama.cpp, 2026-10-22

Flint 40B-A2.5B is Recursive's coding MoE (256 experts, 6 routed and a shared one a token, 2.5B active). Here it runs
on the three engines that have it on this Mac:

- Kapitsa ([#286](https://github.com/nace-ai/kapitsa/pull/286)) on Recursive's 4-bit MLX export
  ([RecursiveSI/flint-int4](https://huggingface.co/RecursiveSI/flint-int4) @ 3b93f905).
- rec 0.1.0, Recursive's own engine ([recursive-oss/rec](https://github.com/recursive-oss/rec): a Go launcher over
  rec-mlx, MLX 0.32 with a resident decode megakernel), on the same files.
- llama.cpp b11804, its first build with Flint, on Recursive's Q4_1 GGUF
  ([RecursiveSI/flint-int4-gguf](https://huggingface.co/RecursiveSI/flint-int4-gguf) @ 9c471510).

There are two measurements. **The poster's protocol**
(`bench/harness/compare_engines.py configs/flint.json --skip agent`) ran Kapitsa and rec in two passes in opposite
order (Kapitsa then rec, rec then Kapitsa), so a pass-order effect cancels in the means, then llama.cpp. **A context
sweep** then measured every engine from 1K to 128K tokens of prompt.

Flint reasons before every answer and has no switch to turn that off. So every run uses `bench.py --always-reasons`:
the reasoning counts as the reply's first part, and the prefill and tool turns get room to finish it. Decode samples at
the model card's temperature 1.0 (top_p 0.95, top_k 20), not the harness's usual 0.6.

## The poster's protocol

| Pass | Engine | Run | Decode T=1.0 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Follow-up TTFT 4K / 16K (s) | Peak (GB) | Foreign GPU load |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (fwd) | Kapitsa | `flint-kapitsa-20261022-120544` | 128.88 (128.23-129.29) | 130.96 | 968 / 816 | 0.12 / 0.14 | 23.27 | 2.23% mean, 23.0% max |
| 1 (fwd) | rec 0.1.0 | `flint-rec-20261022-120842` | 126.42 (126.33-127.36) | 132.30 | 791 / 695 | 0.43 / 0.38 | 27.86 | 12.15% mean, 34.2% max, flagged |
| 2 (rev) | rec 0.1.0 | `flint-rec-20261022-121616` | 126.69 (124.41-126.71) | 132.43 | 801 / 703 | 0.40 / 0.50 | 27.86 | 8.53% mean, 24.3% max, flagged |
| 2 (rev) | Kapitsa | `flint-kapitsa-20261022-122338` | 126.92 (126.04-127.18) | 132.24 | 968 / 819 | 0.13 / 0.15 | 23.27 | 2.17% mean, 10.6% max, flagged |
| 3 | llama.cpp b11804 | `flint-llamacpp-20261022-122927` | 79.57 (78.34-80.35) | 79.93 | 977 / 786 | 0.23 / 0.31 | 26.94 | 3.85% mean, 25.7% max, flagged |
| | **Kapitsa mean** | | **127.90** | **131.60** | **968 / 817** | **0.13 / 0.15** | **23.27** | |
| | **rec 0.1.0 mean** | | **126.56** | **132.37** | **796 / 699** | **0.41 / 0.44** | **27.86** | |
| | **llama.cpp b11804 mean** | | **79.57** | **79.93** | **977 / 786** | **0.23 / 0.31** | **26.94** | |

The runs are in `results/`, their server logs in `logs/`. Decode is the median of 5 runs of 512 tokens (the range in
brackets), prefill the harness's 4K and 16K prompts (3,874 and 15,066 of Flint's tokens), follow-up TTFT the median
time to the first token of a cached second turn.

- **Prefill.** Kapitsa reads prompts 1.22x as fast as rec at 4K (968 against 796 tok/s) and 1.17x at 16K (817
  against 699). llama.cpp reads the 4K prompt as fast as Kapitsa (977, 1% ahead) and the 16K one 4% slower (786).
- **Decode.** Kapitsa and rec are level at this length: 127.9 against 126.6 tok/s at T 1.0 (1% apart), and greedy
  131.6 against 132.4 (rec 0.6% ahead). Both decode 1.6x as fast as llama.cpp (79.6).
- **Follow-up turns.** Kapitsa starts a cached second turn 3x sooner than rec (0.13 against 0.41 s at 4K) and 1.8x
  sooner than llama.cpp (0.23 s).
- **Memory.** Kapitsa peaks at 23.3 GB, llama.cpp at 26.9, rec at 27.9.

**The poster's Flint cells are the means**, as prefill at 4K / decode at T 1.0: Kapitsa 968 / 128, rec 796 / 127,
llama.cpp 977 / 80. Kapitsa's decode lead over rec is inside the runs' own spread. The poster ranks this row as it
ranks every row, by the measured value, so the row counts as Kapitsa's. The same rule gives the M5 Pro's 27B row to
mlx-serve, which leads there by 1.6%.

## Context sweep: 1K to 128K tokens

| Prefill (tok/s) | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Kapitsa** | **858** | **976** | **960** | **907** | **808** | **651** | **464** | **300** |
| rec 0.1.0 | 781 | 800 | 791 | 760 | 699 | 581 | 427 | 262 |
| llama.cpp b11804 | 927 | 1,015 | 978 | 910 | 785 | 602 | 382 | 199 |
| <sub>Kapitsa ÷ rec</sub> | <sub>1.10×</sub> | <sub>1.22×</sub> | <sub>1.21×</sub> | <sub>1.19×</sub> | <sub>1.16×</sub> | <sub>1.12×</sub> | <sub>1.09×</sub> | <sub>1.15×</sub> |
| <sub>Kapitsa ÷ llama.cpp</sub> | <sub>0.93×</sub> | <sub>0.96×</sub> | <sub>0.98×</sub> | <sub>1.00×</sub> | <sub>1.03×</sub> | <sub>1.08×</sub> | <sub>1.21×</sub> | <sub>1.51×</sub> |

| Decode (tok/s) | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Kapitsa** | **124.3** | **121.3** | **119.2** | **113.2** | **106.2** | **85.9** | **66.2** | **45.7** |
| rec 0.1.0 | 126.5 | 123.0 | 116.5 | 107.0 | 95.2 | 79.4 | 56.5 | 35.2 |
| llama.cpp b11804 | 77.5 | 75.1 | 73.4 | 69.7 | 63.7 | 53.8 | 40.5 | 23.8 |
| <sub>Kapitsa ÷ rec</sub> | <sub>0.98×</sub> | <sub>0.99×</sub> | <sub>1.02×</sub> | <sub>1.06×</sub> | <sub>1.12×</sub> | <sub>1.08×</sub> | <sub>1.17×</sub> | <sub>1.30×</sub> |
| <sub>Kapitsa ÷ llama.cpp</sub> | <sub>1.60×</sub> | <sub>1.61×</sub> | <sub>1.62×</sub> | <sub>1.62×</sub> | <sub>1.67×</sub> | <sub>1.60×</sub> | <sub>1.63×</sub> | <sub>1.92×</sub> |

- **Prefill.** Kapitsa reads 1.09-1.22x as fast as rec at every length. llama.cpp reads short prompts faster: 1.08x
  Kapitsa's speed at 1K, falling to 1.02x at 4K. The two are level at 8K (910 against 907 tok/s). From 16K on Kapitsa
  leads, by 1.03x at 16K, 1.08x at 32K, 1.21x at 64K and 1.51x at 128K. Every engine slows as the prompt grows,
  because Flint's 14 full-attention layers read the whole context for each token. At 128K, Kapitsa keeps 31% of its
  4K speed, rec 33% and llama.cpp 20%.
- **Decode.** Kapitsa and rec are level up to 4K (0.98-1.02x). After that Kapitsa pulls ahead: 1.06x at 8K, 1.12x at
  16K, 1.08x at 32K, 1.17x at 64K and 1.30x at 128K. At 128K, Kapitsa keeps 38% of its 4K decode speed, rec 30% and
  llama.cpp 32%. Against llama.cpp, Kapitsa is 1.60-1.67x up to 64K and 1.92x at 128K.

How it ran (`context-sweep/`): `ctx_sweep.py` against each engine's server, started by `sweep_engine.sh` with a long
context (Kapitsa `--ctx 262144`, rec `--context 262144`, llama.cpp `-c 140000`), one engine at a time, each in its own
hold of the GPU lock, and each warmed first by one 2K request.

- **The prompt** at each size is a coding agent's context: `bench/corpus.txt`, then this repository's source files in
  path order, cut to the size in Flint's tokens. A random nonce at its very start keeps any engine from reusing a
  cache. One cold request per size, then a 256-token reply at T 1.0 (top_p 0.95, top_k 20).
- **Prefill** is the prompt's tokens over the time to the first streamed token. Flint's reasoning streams first, so
  this is the prompt read.
- **Decode** is the stream's rate between its 16th and its 4th-from-last event.

`ctx_sweep.py` needs Python's `tokenizers` package, for Flint's tokenizer. The scripts are stored as they ran, with
this session's paths.

llama.cpp's first attempt stopped at its first request: llama-server answers 503 while it loads the model, and the
readiness probe took any answer for ready (`context-sweep/logs/llamacpp-first-attempt-503.log`). The stored
`sweep_engine.sh` is the rerun's, whose probe waits for a 200 (`curl -sf`). Kapitsa and rec ran with the old probe,
and both of their warm-up requests completed (`warmup` in `sweep.json`), so their first measured request found a
loaded server.

The source-file part of the prompt was read from this checkout at each sweep's start. #286's two review commits
landed between the sweeps (comments and docs), so the three sources differ slightly: 1,638,513, 1,639,182 and
1,638,999 tokens for Kapitsa, rec and llama.cpp. Every prompt is cut to the same size, and the first 35K tokens, the
corpus, are the same for all three.

Each point is a single request with no repeats, so read a point as about ±3%. The sweep's 2K and 4K prefill points
bracket the harness's 4K numbers above (Kapitsa 976 and 960 against 968; rec 800 and 791 against 796). 256K was not
run: its prefill alone takes about half an hour per engine. The three engines' points, with each server's command,
are in `context-sweep/sweep.json`, and the servers' logs in `context-sweep/logs/`.

## Conditions

MacBook Pro (16-inch, 2024), `Mac16,7`, M4 Pro (20-core GPU), 48 GB, macOS 27.0.1 (26A434), on AC power in High
Power mode, with 1.0 GB of swap in use at the start and the GPU lock `~/.cache/nace-gpu.lock` held per engine.

**Contention.** This Mac's browser and the Claude app used the GPU during the runs: 2-12% on average, by the harness's
own sampling. Every run but Kapitsa's first is flagged (`CONTENTION: sustained foreign GPU load > 5%` in at least one
phase). The load fell on all three engines and was heaviest in rec's first run. The context sweep has no contention
sampler.

**Kapitsa** ran one binary in every run and in the sweep (sha256 `5bd1f7c349a7…`), built from `4e227c6`: #286's
branch with main `d2cb71d` merged, its source clean. Its second run records `160bfe1`, the harness-only commit on top
(`compare_engines.py` sends the config's key to rec's readiness probe), which the first run had uncommitted. #286's
review fixes after that change neither Flint's kernels nor its outputs: the router kernel's guard on the hidden size
(Flint's is 2048), `kapitsa bench`, and docs. It ran 4-bit Flint with `--ctx 32768 --slots 1` and plain decoding
(Flint has no drafter, and prompt lookup is off by default).

**rec 0.1.0** ran as its release binary, `rec serve --context 32768`, on APFS clones of the same int4 files in the
folder rec keeps its pinned manifest's files in.

**llama.cpp b11804** (`c7fc50941`, the release's macOS arm64 build) ran `llama-server` with its model card's flags
(every layer on the GPU, F16 KV, `-fa on`, `--cache-ram 0`, Flint's sampling), except `-b 2048 -ub 2048` instead of the card's
`-b 256 -ub 128`, and `-c 32768 -np 1` like the other engines. The 2048-token batches are what the other engines' prefill uses. The
card's own batch sizes were not run.

**Not measured:** the other Macs (the M5 Pro row on the poster is dashes), Ollama and LM Studio (neither has Flint
here), and 256K context.
