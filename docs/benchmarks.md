# Benchmarks

How fast Kapitsa runs Qwen3.6-35B-A3B and Qwen3.8-27B on a MacBook Pro (Nemotron 3.5 Bolt and the small Qwen3.5-4B
have their own sections: [Bolt](#nemotron-35-bolt-74b-a9b-2026-10-18), [4B](#qwen35-4b-2026-10-18), and so has
Nemotron 3.5 Super-VL on the [M5 Max](#nemotron-35-super-vl-120b-a12b-m5-max-128-gb-2026-10-18)), compared with the
engines people run these models with on a Mac: llama.cpp, Ollama, LM Studio, mlx-lm, oMLX and MTPLX (and mlx-serve, a
day later: [its own section](#mlx-serve-2026-10-18)). Each engine ran at its latest release with its own best
settings, on the same 4-bit weights wherever it accepts them, through the same requests and the same checks. The
harness and its configs are in [bench/harness](../bench/harness/README.md), and every result file behind these tables
is in [bench/results/2026-10-17](../bench/results/2026-10-17/).

## Summary

On both models Kapitsa was the fastest engine measured on every speed metric: writing (decode, sampled and greedy),
reading a long prompt (prefill at 4K and 16K tokens) and answering a follow-up turn in a long conversation. It also
used the least memory on both models.

| Metric | Qwen3.6-35B-A3B: Kapitsa | Next best | | Qwen3.8-27B: Kapitsa | Next best | |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| Decode, T = 0.6 | 153.7 tok/s | oMLX + DFlash2, 126.5 | 1.22× | 63.0 tok/s | oMLX + DFlash2, 45.3 | 1.39× |
| Decode, greedy | 178.9 tok/s | oMLX + DFlash2, 128.2 | 1.39× | 67.3 tok/s | oMLX + DFlash2, 45.7 | 1.47× |
| Prefill, 4K-token prompt | 1,169 tok/s | oMLX + DFlash2, 1,035 | 1.13× | 389 tok/s | oMLX + DFlash2, 136 | 2.86× |
| Prefill, 16K-token prompt | 986 tok/s | oMLX + DFlash2, 901 | 1.09× | 335 tok/s | oMLX + DFlash2, 128 | 2.60× |
| Cached follow-up, 4K conversation | 0.06 s | MTPLX, 0.10 s | 1.63× | 0.28 s | MTPLX, 0.33 s | 1.17× |
| Cached follow-up, 16K conversation | 0.09 s | MTPLX, 0.15 s | 1.71× | 0.27 s | MTPLX, 0.39 s | 1.46× |
| Peak memory | 21.1 GB | oMLX + DFlash2, 23.4 GB | | 19.4 GB | oMLX (no speculation), 20.5 GB | |

Ratios are computed from the unrounded results. Measured on one MacBook Pro (M4 Pro, 48 GB); on each model the
next fastest engine on every metric ran in the same session as Kapitsa (35B: the morning of 2026-10-17; 27B: one
evening session on the final build). The [conditions](#conditions) section lists what differed between sessions.

For an additional Kapitsa-only run on a different machine, see the [M5 Max / 128 GB report](benchmarks-m5-max.md).
Those runs include background GPU contention and are reported separately from the comparison below.

## Results

Decode figures are the median of 5 runs, with the range in brackets. Prefill is tokens per second of a cold prompt.
A cached follow-up is the time to the first token of the answer to a new message in a conversation that already
holds the 4K or 16K prompt (median of 3 follow-ups). Peak memory is the server's whole process tree, in GB (2^30
bytes). Bold: best in the column. Every run passed the validity checks (thinking off, prompt length within 1% of
the reference, no reused cache on cold prompts) and every engine passed the tool-call checks.

### Qwen3.6-35B-A3B (mixture of experts, 3B active)

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** (DFlash2 drafter) | **153.7** (146.6–165.8) | **178.9** (173.7–184.2) | **1,169** | **986** | **0.06** | **0.09** | **21.1** | pass |
| oMLX + DFlash2 (block 5, adaptive verify) | 126.5 (115.1–135.3)<sup>1</sup> | 128.2 (115.8–135.4) | 1,035 | 901 | 0.29 | 0.40 | 23.4 | pass |
| Ollama, MLX runner + MTP (adaptive depth) | 88.6 (87.1–94.0) | 92.1 (90.9–94.4) | 892 | 776 | 0.21 | 0.32 | 32.2 | pass |
| LM Studio (MLX engine) | 86.0 (85.1–86.8)<sup>2</sup> | 91.1 (90.6–91.9)<sup>2</sup> | 957 | 846 | 0.69 | 0.79 | 24.7 | pass |
| MTPLX (sustained profile, depth 1) | 79.8 (78.7–84.6) | 73.4 (70.6–76.5) | 918 | 804 | 0.10 | 0.15 | 31.7 | pass |
| Ollama, GGUF + MTP (shipped depth 2) | 69.7 (67.9–75.5) | 76.7 (75.0–78.3) | 701 | 541 | 0.24 | 0.28 | 5.6 + 19.9 mapped<sup>6</sup> | pass |
| llama.cpp + MTP (n-max 1) | 68.9 (66.8–69.9) | 65.5 (64.6–69.2) | 663 | 545 | 0.20 | 0.21 | 27.4 | pass |
| llama.cpp + MTP (n-max 3) | 68.1 (65.5–75.0) | 73.5 (70.8–74.2) | 659 | 553 | 0.18 | 0.20 | 27.5 | pass |
| Ollama, GGUF (MTP off) | 63.4 (63.1–65.0) | 63.4 (62.9–63.7) | 768 | 643 | 0.22 | 0.26 | 5.1 + 19.4 mapped<sup>6</sup> | pass |
| llama.cpp | 62.7 (62.5–63.1) | 62.2 (62.1–62.6) | 728 | 630 | 0.18 | 0.20 | 26.2 | pass |
| llama.cpp + MTP (n-max 2) | 60.1 (52.6–68.8) | 56.8 (53.0–59.4) | 584 | 483 | 0.21 | 0.23 | 27.5 | pass |

mlx-lm, and oMLX without speculation, have no 35B row: they were not re-run on the final harness for this model
(their earlier 35B runs used a previous version of the harness and are not comparable).

### Qwen3.8-27B (dense)

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** (DFlash2 drafter)<sup>3</sup> | **63.0** (55.0–66.3) | **67.3** (65.2–67.4) | **389** | **335** | **0.28** | **0.27** | **19.4** | pass |
| oMLX + DFlash2 (block 5, verify dflash) | 45.3 (43.5–46.3) | 45.7 (45.1–45.9) | 136 | 128 | 0.92 | 1.26 | 24.0 | pass |
| MTPLX (turbo profile, depth 3) | 36.2 (35.0–36.5) | 39.3 (38.6–39.4) | 126 | 121 | 0.33 | 0.39 | 33.4 | pass |
| Ollama, MLX runner + MTP (adaptive depth) | 32.2 (31.7–32.8) | 32.2 (32.2–32.5) | 124 | 119 | 0.42 | 0.51 | 32.2 | pass |
| Ollama, GGUF + MTP (shipped depth 4) | 20.4 (19.9–22.1) | 19.1 (18.7–19.7) | 83 | 78 | 0.52 | 0.60 | 13.8 + 15.4 mapped<sup>6</sup> | pass |
| llama.cpp + MTP (n-max 3) | 18.3 (14.9–18.7) | 17.3 (15.2–18.7) | 84 | 77<sup>4</sup> | 0.64 | 0.69 | 30.6 | pass |
| LM Studio (MLX engine) | 15.4 (15.2–15.6)<sup>2</sup> | 15.3 (15.0–15.5)<sup>2</sup> | 127 | 122 | 0.56 | 0.65 | 24.7 | pass |
| oMLX (no speculation) | 13.5 (13.0–13.8) | 13.3 (12.9–14.1) | 109 | 104<sup>4</sup> | 38.52<sup>5</sup> | 157.46<sup>5</sup> | 20.5 | pass |
| llama.cpp + MTP (n-max 1) | 13.3 (12.8–13.9) | 12.7 (12.7–12.8) | 87 | 80 | 0.60 | 0.67 | 31.5 | pass |
| mlx-lm | 13.3 (13.1–13.5) | 13.4 (13.3–13.4) | 100 | 92 | 0.57 | 0.90 | 23.6 | pass |
| llama.cpp + MTP (n-max 2) | 13.0 (10.8–17.1) | 16.3 (15.7–17.0) | 81 | 78 | 0.65 | 0.68 | 31.7 | pass |
| Ollama, GGUF (MTP off) | 10.9 (9.0–12.1) | 9.4 (9.0–11.1) | 89 | 80 | 0.50 | 0.58 | 12.6 + 15.2 mapped<sup>6</sup> | pass |
| llama.cpp | 10.0 (9.8–10.3) | 9.9 (9.9–10.9) | 90 | 82 | 0.63 | 0.64 | 29.9 | pass |

### Notes

1. One of oMLX's five sampled runs, its fastest (135.3 tok/s), fell into a repetition loop (distinct-4-gram ratio
   0.45; the harness flags anything below 0.5), which speculative decoding speeds up. Without that run its median
   is 122.4 tok/s.
2. LM Studio stops at 511 of the 512 requested tokens. Its rates use the 511 tokens it reports.
3. The 27B rows of Kapitsa, oMLX + DFlash2, MTPLX, Ollama's MLX runner and LM Studio come from one session on
   2026-10-17 between 22:08 and 22:58, with Kapitsa's final build (main at e5c6f41). Kapitsa ran before and after
   the others; the table shows the run after. The run before gave 62.2 / 65.0 tok/s decode, 395 / 345 tok/s
   prefill, 0.22 / 0.27 s follow-ups and 19.4 GB ([confirmations](../bench/results/2026-10-17/confirmations/)).
4. The Mac swapped during this 16K prefill (llama.cpp: 96,796 pages, about 1.5 GB; oMLX: 6,611 pages), which may
   have slowed it.
5. oMLX without speculation did not reuse its prompt cache for follow-up turns on this model (it reported 0 reused
   tokens), so every follow-up re-read the whole conversation.
6. Ollama's GGUF runner maps the weights from disk instead of loading them, so they are not in the server's own
   memory: the first figure is the server's memory, the second the resident mapped weights. llama.cpp ran with
   `-lm none`, which loads the weights into its own memory like the other engines.

## What was compared

| Engine | Version measured | Weights | Speculative decoding | Other settings |
| --- | --- | --- | --- | --- |
| Kapitsa | main at b457a52 (35B) and e5c6f41 (27B), 2026-10-17, Release builds | MLX 4-bit | DFlash2 drafter (the default) | defaults; 32K context, 1 slot |
| llama.cpp | 0.7.0, build b11669 (620d2225c), official macOS arm64 release | Ollama's Q4_K_M GGUF | MTP with the head inside the GGUF, `--spec-draft-n-max` 1, 2 and 3; and off | `-c 32768 -np 1 -ngl 999 -fa on -b 2048 -ub 2048 -lm none --jinja`, official Qwen chat template, `--min-p 0` |
| Ollama, GGUF | 0.40.5 (bundled llama.cpp b11669) | its Q4_K_M GGUF tags | built-in MTP at the shipped depth (2 on the 35B, 4 on the 27B); and off | dedicated server: 32K context, 1 slot, flash attention |
| Ollama, MLX runner | 0.40.5 (MLX bebd6422, Metal 4 build) | MLX 4-bit + MTP head, imported without copying | built-in MTP with adaptive depth (no knobs) | same server settings |
| LM Studio | llmster 0.0.25-1 (headless), MLX runtime 1.13.1 (mlx-engine ab52366, mlx 0.32.3) | lmstudio-community MLX 4-bit (the same shards) | none: its MLX engine has none for these models | LM Studio's staff-pick model entry; 32K context, `--parallel 1` |
| mlx-lm | 0.32.2 (mlx 0.32.4) | MLX 4-bit | none for these models | prompt cache (4 sequences, 8 GB), 2048-token prefill steps |
| oMLX | 0.7.1rc1 (mlx 0.32.4, mlx-lm 0.31.4.dev132) | MLX 4-bit | DFlash2 drafter quantized to 4-bit at load, block 5, verify `adaptive` (35B) or `dflash` (27B); and none (27B) | memory guard `aggressive`, 4 GB RAM + 4 GB SSD prompt cache |
| MTPLX | 2.12.2 (mlx 0.32.2, mlx-lm 0.31.3) | MLX 4-bit + MTP head | MTP as the weights' runtime file recommends: profile `sustained`, depth 1 (35B); `turbo`, depth 3 (27B) | inline session post-commit (follow-ups use its session cache) |

"Best settings" means: speculative decoding wherever the engine offers it for these models, at the engine's or the
weights' own recommendation, with each depth measured where it is a knob (llama.cpp), and the fastest variant
found in short smoke tests where there were several (oMLX's verify modes and block sizes). Where an engine has the
settings, context is 32K and there is one conversation slot; every request sends the same sampling parameters. The
configs in [bench/harness/configs](../bench/harness/configs) record each choice and its reason. LM Studio's
llama.cpp runtime on its own Q4_K_M GGUF with MTP drafting was not measured; on the 27B it may decode faster than its
MLX engine (standalone llama.cpp with MTP decodes the 27B at 18.3 tok/s, LM Studio's MLX engine at 15.3).

We also ran three experimental engines through the same harness: AX Engine 7.7.0, Darkbloom 0.9.22 and lithos-metal
0.1.2. None was faster than Kapitsa on any speed metric. They are not in the tables because this repository has no
configs for them.

**Weights.** The MLX engines read the same files, MLX affine 4-bit with group size 64:
Qwen3.6-35B-A3B is mlx-community's 4-bit conversion (router and shared-expert gates in 8 bits) with the model's
MTP head, as published by MTPLX
([Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed](https://huggingface.co/Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed)
@9546051, 19.5 GB including the 0.5 GB head); Qwen3.8-27B is
[lmstudio-community/Qwen3.8-27B-MLX-4bit](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-4bit) @6067b15
(15.0 GB) plus the MTP head of MTPLX's 27B release (0.2 GB). LM Studio's own downloads of these models are the
same shards, and Ollama's MLX runner read them through a zero-copy import, checked tensor by tensor. llama.cpp and
Ollama's GGUF runner read Ollama's Q4_K_M GGUFs (35B: 20.2 GB, blob d372de8e; 27B: 15.7 GB, blob f5f1dd89), which
contain the MTP head; they read slightly more bytes per generated token than the MLX files (their output layer is
Q6_K). Sizes here are in GB of 2^30 bytes, like the memory figures.
Kapitsa and oMLX use the DFlash2 drafters
[incoai/Qwen3.6-35B-A3B-DFlash2](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2) @51ef7b6 and
[incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) @015e795, and both quantize them to
4 bits when loading. Kapitsa's memory includes its drafter (about 0.6 GB on the 35B and 1.6 GB on the 27B).

**Kapitsa's build.** The 35B rows are main at b457a52, run in the same session as the other engines' 35B runs; the
27B rows are the final main, e5c6f41, run in one session with the 27B rows of the engines closest to it. Both were
run from a copy of the build (sha256 in the result files). The final build on the 35B, after the other sessions:
160.2 / 173.2 tok/s decode, 1,127 / 947 tok/s prefill, 0.06 / 0.07 s follow-ups, 20.9 GB, within the spread of the
b457a52 row ([confirmations](../bench/results/2026-10-17/confirmations/)).

## How it was measured

All requests go through each engine's OpenAI-compatible HTTP API, streaming, the way a coding agent talks to it.
The details are in the [harness README](../bench/harness/README.md#what-a-run-measures).

- **Decode:** a 512-token answer to a coding request (an LRU cache module with tests), 5 runs at temperature 0.6
  (top_p 0.95, top_k 20, Qwen's recommendation) and 5 greedy runs. The rate is measured over the middle of the
  stream (the first 16 and last 4 tokens are dropped), so neither time to first token nor engines that send
  several tokens per event distort it. Greedy runs make every engine write nearly the same text.
- **Prefill:** a cold prompt of about 4K and 16K tokens (4,133 and 16,286 with these chat templates) of source code,
  with a random nonce at the start of the system message so no engine can reuse a cached prefix. Rate = prompt
  tokens / time to first token.
- **Cached follow-up:** after each cold prompt, three follow-up questions on the growing conversation, 1 s apart;
  the median time to first token.
- **Tool calls:** streaming and non-streaming requests with two tools, with thinking off and on, and a tool-result
  round trip.
- **Memory:** `phys_footprint_peak` of the server's process tree, read with macOS's `footprint` after every phase.

**What makes a run count.** Thinking is switched off with both `chat_template_kwargs.enable_thinking=false` and
`reasoning_effort: "none"` (engines differ in which they honour), and any reasoning text makes a run invalid. Every
request sets every sampling parameter, including `min_p 0` and neutral penalties, because the engines' defaults
differ. Prompts come from a frozen corpus (excerpts of Hugging Face Transformers source code, sha256 recorded), and
cold prompts must come back within 1% of the reference token count with no cached tokens reported. Decode runs are
checked to stop at exactly 512 tokens and for repetition loops. Before timing, each server is warmed up (a ~2.9K-token
prompt, a tool call, and three full decodes for engines whose speculation adapts across requests).

**One engine at a time, on a quiet GPU.** Each configuration runs alone under an exclusive GPU lock, from server
start until its memory is released, and its timed requests start only after no other app has used more than 5% of
the GPU for 15 s (every run here got there within 21 s). A monitor samples every process's GPU time every 2 s and
flags a run when other apps use more than 5% of the GPU for three samples in a row, or on average. The tables use
one run per configuration: its newest run on the final harness without an invalid measurement or a contention flag.
Every run in the tables passed with no contention flag.

## Conditions

MacBook Pro (Mac16,7), Apple M4 Pro with a 14-core CPU, 20-core GPU, 48 GB of unified memory (273 GB/s); macOS 27.0.1
(26A434); on AC power. mlx-lm, oMLX and MTPLX used the macOS 26 build of the MLX wheels; LM Studio and Ollama ship
their own MLX builds (LM Studio's is the macOS 14 build, Ollama loads its Metal 4 build on macOS 26 and later).

Two conditions differed between the two days, and the result files record both for every run:

- **Power mode.** The runs on 2026-10-16 were in the Automatic energy mode, those on 2026-10-17 in High Power mode.
  Kapitsa, LM Studio, MTPLX and oMLX + DFlash2 on both models, and Ollama's MLX runner on the 27B, ran on
  2026-10-17 in High Power mode.
- **Background GPU load.** Two kinds of desktop GPU load were present at times: the Photos analysis daemons, busy
  for hours after the update to macOS 27, and the desktop app that drove the runs. The harness treats such load as
  ambient: it does not block or flag a run, but it is recorded. It averaged 20–43% of GPU time in the 2026-10-16
  daytime runs (llama.cpp, Ollama GGUF, mlx-lm, oMLX without speculation), 8–33% in the evening runs (Ollama MLX
  runner on the 35B), 0% in the 2026-10-17 morning runs and 1–16% in the 2026-10-17 evening session (the desktop app
  that drove the runs: Kapitsa 1.1%, oMLX + DFlash2 9.1%, MTPLX 8.8%, Ollama's MLX runner 16.4%, LM Studio 0%).

We have not measured how much either changes an engine's speed. On both models the next fastest engine on every
metric ran in the same session and power mode as Kapitsa: on the 35B in the morning of 2026-10-17 with no ambient load,
on the 27B in the evening session above. The 2026-10-16 rows (llama.cpp, Ollama's GGUF runner, mlx-lm, oMLX without
speculation, and Ollama's MLX runner on the 35B) are in Automatic mode with ambient load; none of them is next
fastest on any metric.

## Run index

The tables can be rebuilt from the result files with
`python3 bench/harness/compare_table.py bench/results/2026-10-17 --out /tmp/table`.

| Result file | Started | Power mode | Foreign GPU, mean / max (%) | Ambient GPU, mean (%) | Lock held (s) |
| --- | --- | --- | ---: | ---: | ---: |
| `q36-35b-kapitsa-20261017-084636.json` | 2026-10-17 08:48 | High Power | 0.0 / 0.0 | 0.0 | 107 |
| `q36-35b-omlx-best-20261017-091029.json` | 2026-10-17 09:29 | High Power | 0.0 / 0.0 | 0.0 | 139 |
| `q36-35b-ollama-mlx-q4-20261016-212529.json` | 2026-10-16 21:26 | Automatic | 0.0 / 0.0 | 8.4 | 168 |
| `q36-35b-lmstudio-20261017-085626.json` | 2026-10-17 09:11 | High Power | 0.0 / 0.0 | 0.0 | 160 |
| `q36-35b-mtplx-20261017-085604.json` | 2026-10-17 09:08 | High Power | 0.3 / 14.0 | 0.0 | 205 |
| `q36-35b-ollama-mtp-20261016-145225.json` | 2026-10-16 15:00 | Automatic | 1.4 / 3.1 | 22.6 | 179 |
| `q36-35b-llamacpp-mtp1-20261016-140843.json` | 2026-10-16 14:23 | Automatic | 1.3 / 3.4 | 21.6 | 180 |
| `q36-35b-llamacpp-mtp3-20261016-144042.json` | 2026-10-16 14:50 | Automatic | 1.4 / 3.0 | 23.9 | 172 |
| `q36-35b-ollama-nomtp-20261016-150248.json` | 2026-10-16 15:11 | Automatic | 1.1 / 2.9 | 20.1 | 258 |
| `q36-35b-llamacpp-20261016-135336.json` | 2026-10-16 14:06 | Automatic | 1.4 / 4.3 | 20.1 | 182 |
| `q36-35b-llamacpp-mtp2-20261016-142601.json` | 2026-10-16 14:37 | Automatic | 1.2 / 3.7 | 24.9 | 244 |
| `q38-27b-kapitsa-20261017-225432.json` | 2026-10-17 22:55 | High Power | 0.0 / 0.7 | 1.1 | 225 |
| `q38-27b-ollama-mlx-q4-20261017-221219.json` | 2026-10-17 22:14 | High Power | 0.0 / 1.8 | 16.4 | 504 |
| `q38-27b-mtplx-20261017-222801.json` | 2026-10-17 22:30 | High Power | 0.0 / 0.0 | 8.8 | 569 |
| `q38-27b-omlx-best-20261017-222043.json` | 2026-10-17 22:22 | High Power | 0.0 / 0.8 | 9.1 | 438 |
| `q38-27b-ollama-mtp-20261016-130604.json` | 2026-10-16 13:16 | Automatic | 2.5 / 5.5 | 43.0 | 691 |
| `q38-27b-llamacpp-mtp3-20261016-124604.json` | 2026-10-16 12:55 | Automatic | 2.2 / 5.2 | 40.8 | 730 |
| `q38-27b-lmstudio-20261017-223730.json` | 2026-10-17 22:38 | High Power | 0.0 / 0.0 | 0.0 | 678 |
| `q38-27b-omlx-20261016-161709.json` | 2026-10-16 16:30 | Automatic | 1.4 / 7.6 | 31.6 | 1,219 |
| `q38-27b-llamacpp-mtp1-20261016-115128.json` | 2026-10-16 12:00 | Automatic | 2.2 / 7.0 | 41.9 | 835 |
| `q38-27b-mlxlm-20261016-155425.json` | 2026-10-16 16:04 | Automatic | 1.8 / 6.5 | 30.5 | 792 |
| `q38-27b-llamacpp-mtp2-20261016-121303.json` | 2026-10-16 12:34 | Automatic | 2.2 / 5.0 | 40.9 | 781 |
| `q38-27b-ollama-nomtp-20261016-132653.json` | 2026-10-16 13:38 | Automatic | 1.9 / 8.2 | 36.1 | 959 |
| `q38-27b-llamacpp-20261016-112909.json` | 2026-10-16 11:36 | Automatic | 1.9 / 8.6 | 36.2 | 973 |

Two extra Kapitsa runs from the evening session are in [confirmations](../bench/results/2026-10-17/confirmations/)
(the 27B run before the other engines, `q38-27b-kapitsa-20261017-220825.json`, and the final build on the 35B,
`q36-35b-kapitsa-20261017-225817.json`); they are not in the tables. Times are local (UTC+5); "Started" is when the
benchmark began, after start-up and warm-up. In the result files,
local paths are replaced by placeholders (`~`, `<bench-root>` for the folder the harness ran from before it moved into
this repository, `<kapitsa-checkout>`, `<build-dir>`); nothing else was changed. `engine: "nace-infer"` and
`nace_infer_git` are Kapitsa under its earlier name, and `{root}` in the recorded configs is that earlier folder; the
configs in [bench/harness/configs](../bench/harness/configs) are the same settings in this repository's layout.

## Nemotron 3.5 Bolt 74B-A9B (2026-10-18)

NVIDIA's own 24 GB checkpoint (`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`: NVFP4 Mamba projections, IQ2_XXS experts,
Q8_0 attention, latent projections and LM head) on the engines that load it on a Mac. oMLX and mlx-vlm read it
through the model file of NVIDIA's Bolt bundle for MLX (`txgsync/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-MLX`,
which carries NVIDIA's files byte-identical); Kapitsa reads the safetensors directly. All runs in one session on
2026-10-18 between 06:37 and 07:50, back to back, with [configs/bolt.json](../bench/harness/configs/bolt.json); the
desktop's own GPU work (the app the session ran in: 10-40% of the GPU) was there for every engine alike. Result files:
[bench/results/2026-10-18](../bench/results/2026-10-18/).

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** (LiLiCorr drafter, GPU + Neural Engine prefill) | **63.9** (62.5–66.1)<sup>1</sup> | **67.4** (64.6–69.6) | **436** | **385** | **0.28** | **0.32** | **22.7** + 1.0 mapped<sup>2</sup> | pass |
| oMLX + LiLiCorr drafter | fails<sup>3</sup> | 52.8 (50.3–53.9)<sup>3</sup> | 288 | 297 | 0.75 | 0.84 | 31.5 | pass |
| oMLX (no speculation) | 33.9 (33.5–34.1) | 34.1 (33.6–34.2) | 296 | 300 | 0.72 | 0.72 | 30.3 | pass |
| mlx-vlm (no speculation) | 31.8 (31.1–32.6) | 32.9 (32.4–33.1) | 300 | 296 | 14.52<sup>4</sup> | 57.86<sup>4</sup> | 32.3 | pass |

1. One of the five sampled runs looped (distinct-4-gram 0.43; flagged): 63.6 tok/s, not the fastest. Without it the
   median is 64.2.
2. Kapitsa maps the BF16 embedding from its file: those pages (1.0 GB resident) are clean file pages outside the
   process footprint. Steady footprint after each phase: Kapitsa 22.4–22.7 GB, oMLX 23.9–24.3, oMLX + LiLiCorr
   25.2–25.6, mlx-vlm 23.2–23.5.
3. oMLX 0.7.1rc1 with the drafter fails every request at T > 0 (its sampled DFlash selection), so its sampled decode
   and its prefill phases have no valid result (`bolt-omlx-lilicorr-20261018-063713.json`); the `-greedy` run sends
   temperature 0 in every phase, so it has two greedy series, with medians 52.8 and 45.9: the table shows the higher,
   and its prefill and follow-up numbers.
4. mlx-vlm's server reused no prompt cache for the follow-up turns (0 cached tokens): every follow-up re-read the
   whole conversation.

Kapitsa's earlier run in the same session (before the weight packing that lowered its memory; `confirmations/`) gave
64.7 / 65.6 tok/s decode, 455 / 396 tok/s prefill, 0.25 / 0.33 s follow-ups and 24.9 GB.

## Nemotron 3.5 Super-VL 120B-A12B (M5 Max, 128 GB, 2026-10-18)

NVIDIA's NVFP4 checkpoint (`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`: NVFP4 experts, FP8 Mamba
projections, attention and LM head, BF16 latent projections; 70 GB of weights) needs a 96 GB+ Mac, so this comparison
ran on a MacBook Pro with an M5 Max (40-core GPU, 128 GB, macOS 26.4), with every engine that runs the model on a Mac:

- Kapitsa and mlx-vlm read NVIDIA's checkpoint itself. mlx-vlm's Super-VL support (and its native MTP drafter) is
  newer than its latest release (0.7.8 refuses the model), so it ran at git main 73d366e, an unreleased build.
- oMLX 0.7.1rc1 and mlx-lm 0.32.2 cannot read it (`Model type nemotron_h_omni not supported`). They ran the
  only text-only MLX conversion, `rapid-mlx/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-MLX-4bit` (affine 4-bit from
  NVIDIA's BF16 weights, 69.5 GB). Those weights carry no MTP head, and oMLX's drafter path covers no Nemotron-H
  model, so neither has speculative decoding here.
- llama.cpp 0.7.0 (b11669, the latest release) ran ggml-org's official Q4_0 GGUF (71.8 GB; its NVFP4 GGUF has no
  Metal kernels), with the model's MTP head (ggml-org's `mtp-...-Q4_0.gguf`) at each depth, and with NVIDIA's
  LiLiCorr drafter as ggml-org converted it.

One session on 2026-10-18, every configuration run twice, with
[configs/super-vl.json](../bench/harness/configs/super-vl.json); the table has each configuration's second run. Result
files, weights and revisions: [bench/results/2026-10-18-m5-max](../bench/results/2026-10-18-m5-max/).

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** (LiLiCorr drafter, GPU + Neural Engine prefill) | **84.5** (78.1–93.2) | **87.0** (76.3–90.0) | 627 | 509 | 0.22 | 0.26 | **70.4** + 1.0 mapped<sup>1</sup> | pass |
| Kapitsa (no drafter) | 46.4 (45.0–47.4) | 46.5 (45.3–47.3) | 617 | 501 | 0.20 | 0.25 | 67.7 + 1.0 mapped | pass |
| llama.cpp + MTP (n-max 3) | 65.5 (63.0–69.7) | 66.1 (64.7–67.5) | 833 | 882 | 0.22 | 0.20 | 81.9 | pass |
| llama.cpp + MTP (n-max 2) | 58.8 (56.5–61.8) | 59.4 (58.7–60.0) | 830 | 877 | 0.22 | 0.20 | 81.7 | pass |
| llama.cpp + MTP (n-max 1) | 49.6 (48.6–52.0) | 49.4 (48.9–49.6) | 818 | 874 | 0.23 | 0.21 | 81.6 | pass |
| llama.cpp (no speculation) | 35.9 (35.0–37.1) | 35.4 (35.1–35.6) | 764 | 821 | **0.19** | 0.20 | 78.5 | pass |
| llama.cpp + LiLiCorr drafter (n-max 7) | 26.0 (23.4–31.1) | 31.0 (29.9–31.4) | 681 | 746 | 0.24 | 0.32 | 86.2 | pass |
| oMLX, MLX 4-bit (no speculation) | 51.8 (49.7–52.1) | 50.4 (50.1–51.0) | **989** | **1,049** | 0.33 | 0.43 | 74.8 | pass |
| mlx-lm, MLX 4-bit | 45.5 (44.7–45.8) | 45.4 (44.7–45.6) | 885 | 966 | 0.20 | 0.25 | 75.6 | pass |
| mlx-vlm + MTP (block 4) | 46.3 (43.9–48.5) | 49.1 (48.9–49.6) | 756 | fails<sup>2</sup> | 0.52 | fails<sup>2</sup> | 108.0 | pass |
| mlx-vlm (no speculation) | 30.7 (30.3–30.8) | 30.9 (30.7–31.0) | 710 | 812 | 0.53 | 0.61 | 99.3 | pass |

1. Kapitsa maps the BF16 embedding from its file (1.0 GB of clean file pages outside the process footprint), as on
   Bolt. The others' memory includes their whole weights: mlx-vlm turns the FP8 layers into BF16 at load.
2. mlx-vlm with its MTP drafter runs out of GPU memory on the 16K prompt (`[METAL] Command buffer execution failed:
   Insufficient Memory`, peak 108 GB), in both runs and also with its prefix cache off.

Kapitsa writes fastest, sampled and greedy (1.3× llama.cpp + MTP, the next fastest), and uses the least memory. It
reads prompts slowest: every other engine prefills at 680-1,050 tok/s against Kapitsa's 500-630 (oMLX and mlx-lm on
their 4-bit weights fastest), and its follow-ups (0.22 / 0.26 s) are close to llama.cpp's (0.19 / 0.20 s) and mlx-lm's
rather than ahead.
Speculative settings came from smoke tests: mlx-vlm's MTP block 4 beat 2, 3 (its PR's setting), 5 and 6; llama.cpp's
LiLiCorr drafter is slower than no drafter at all (its NVFP4 GGUF has no Metal kernels), at n-max 7 as at the default 3.

Conditions: desktop apps stayed open. One `com.apple.WebKit` process used 3-5% of the GPU throughout and rose past the
harness's 5% threshold in every engine's prefill phases, so every row but oMLX's is flagged CONTENTION. Kapitsa's
rows show much more of it during its own prefill (31% mean in the 16K phase, which runs on the GPU and the Neural
Engine together; its decode phases saw 4-5% like the others'), in both of its runs; its prefill rates are close to the
ones measured outside the server (672-684 / 514-516 tok/s at 4K / 16K). At T = 0.6 this model repeats itself in some
512-token answers on every engine (flagged, kept). The first run of each configuration is in
[confirmations](../bench/results/2026-10-18-m5-max/confirmations/): Kapitsa 78.7 / 80.3 tok/s decode there, the others
within 3% of the rows above except llama.cpp + LiLiCorr (28.6 tok/s sampled).

## mlx-serve (2026-10-18)

[mlx-serve](https://github.com/ddalcu/mlx-serve) 26.10.2 (its latest release, MLX 0.32.3), a Zig server over MLX with
its own Metal kernels, in both of its speculative setups for these models, MTP and DFlash2, against Kapitsa (main at
c93f27c). One session on 2026-10-18, 12:40-13:38, under one hold of the GPU lock: Kapitsa first and last, mlx-serve in
between, with [configs/mlx-serve.json](../bench/harness/configs/mlx-serve.json). Result files:
[bench/results/2026-10-18-mlx-serve](../bench/results/2026-10-18-mlx-serve/).

Kapitsa was faster than mlx-serve on every metric on both models and used less memory; mlx-serve led in no column. The
closest is writing on the 27B: mlx-serve's DFlash2 draft tree wrote 51.9 tok/s sampled and 55.1 greedy, against
Kapitsa's 58.4 and 61.3 (56.8-61.5 and 60.8-62.5 over its four runs in the session). That is the fastest any engine
but Kapitsa has written this model on this Mac (next: oMLX + DFlash2, 45.3 and 45.7 on 2026-10-17). mlx-serve's tree
gets about as many tokens out of a round as Kapitsa's verify (4.9-5.4 against 4.6-5.3), but its rounds take longer
(about 97 ms against 81-84 ms; among other things it runs the drafter at 8 bits, Kapitsa at 4). On the 35B mlx-serve's
faster setup is MTP, 123.9 / 130.0 tok/s against Kapitsa's 166.8 / 173.1 (138.7 / 161.8 in its run before mlx-serve);
with DFlash2 it switched DFlash off early in every answer. Kapitsa read long prompts 1.14-1.16× faster than
mlx-serve's MTP setup on the 35B and 1.9-2.7× faster on the 27B, and answered follow-ups on the 35B in 0.05-0.07 s
against 0.26-0.38 s (mlx-serve's cache left 63-95 tokens of each follow-up to read there, Kapitsa's 20-27).

**Qwen3.6-35B-A3B**

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** (DFlash2 drafter)<sup>1</sup> | **166.8** (138.2–173.8) | **173.1** (167.1–176.6) | **1,145** | **961** | **0.05** | **0.06** | **20.9** | pass |
| mlx-serve + MTP (depth up to 2) | 123.9 (118.3–130.3) | 130.0 (117.6–135.7) | 986 | 838 | 0.26 | 0.38 | 22.5 | pass |
| mlx-serve + DFlash2 (block 4) | 104.2 (104.1–104.3)<sup>2</sup> | 107.0 (107.0–107.1)<sup>2</sup> | 911 | 783 | 0.26 | 0.38 | 22.9 | pass |

**Qwen3.8-27B**

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** (DFlash2 drafter)<sup>1</sup> | **58.4** (55.3–65.1)<sup>3</sup> | **61.3** (58.5–61.6) | **264**<sup>3</sup> | **323**<sup>3</sup> | **0.23** | **0.26** | **19.4** | pass<sup>3</sup> |
| mlx-serve + DFlash2 (draft tree, block 8) | 51.9 (50.0–54.3) | 55.1 (54.5–56.4) | 126 | 120<sup>3</sup> | 0.27 | 0.40 | 21.5 | pass |
| mlx-serve + MTP (depth up to 4) | 38.0 (34.1–42.3) | 38.9 (37.7–39.6) | 126 | 120<sup>3</sup> | 0.29 | 0.40 | 20.1 | pass |

1. Kapitsa ran before and after mlx-serve; the table shows its newest run. On the 35B that is the run after; the run
   before gave 138.7 (134.9–155.7) / 161.8 (142.0–173.7) tok/s decode, 1,123 / 953 tok/s prefill, 0.05 / 0.07 s
   follow-ups and 20.9 GB. On the 27B it is the second of two re-runs at the end of the session; its four 27B runs
   gave 56.8–61.5 / 60.8–62.5 tok/s decode, 242–296 / 231–323 tok/s prefill, 0.23–0.25 / 0.26 s follow-ups and
   19.4 GB.
2. DFlash ran for the first 8 to 33 rounds of each timed answer, then switched itself off for the rest of it: the
   drafts it accepted per round (1.73–1.78) stayed under mlx-serve's bar for MoE models (1.80). The rest of each
   answer was plain decoding, hence the narrow range.
3. Flagged for contention: Chrome and Slack used more than 5% of the GPU for sustained stretches (up to 90% at peaks)
   in every 27B run of the session, most of all in the long prefills (mlx-serve's 16K prefill takes over two minutes).
   Each flagged 27B setting was re-run (Kapitsa twice; a second re-run of mlx-serve's MTP was stopped in its warm-up
   to keep the session short), every re-run was flagged again, and the table shows the newest run with its flagged
   phases marked. The earlier mlx-serve runs: MTP 36.3 / 38.8 tok/s decode, 127 / 120 tok/s prefill, 0.36 / 0.44 s
   follow-ups, 20.2 GB; DFlash2 52.8 / 55.0, 126 / 117, 0.32 / 0.42 s, 21.5 GB. Ambient load (the camera daemon and
   the app that drove the runs, never flagged) took 10–18% of the GPU in the 27B runs until the last one (3.7%), in
   which Kapitsa's 16K prefill reached 323 tok/s; on a quiet Mac on 2026-10-17 Kapitsa read the 27B's prompts at 389 /
   335 tok/s.

**Weights.** On the 35B mlx-serve read the vendor's own pack,
[ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit](https://huggingface.co/ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit) @6122e2b: its
trunk shards are the ones every engine here reads (sha256-identical), plus its MTP sidecar (mlx-community's 4-bit
head, 0.44 GB). The vendor's 27B pack
([ddalcu/Qwen3.8-27B-MLX-Serve-4bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-4bit) @901aa73) is its own
conversion of the model (bf16 embedding; in the tensors sampled about 0.2% of the 4-bit codes differ from
lmstudio-community's), so on the 27B mlx-serve read the trunk every engine reads with that pack's MTP head (its 29
`mtp.*` tensors byte for byte, 0.29 GB) and its `drafter/`, the same file as the DFlash2 drafter Kapitsa reads. For
DFlash2 on the 35B it read incoai's drafter, also Kapitsa's. `bench/harness/tools/mlx_serve_pack.py` builds both
folders from the shared weights (APFS clones) and the vendor's files that differ.

**Settings.** mlx-serve's own defaults, except where short runs just before the session measured faster (five sampled
decodes each; [tuning](../bench/results/2026-10-18-mlx-serve/README.md#tuning)):

- MTP: the model's head with exact verification (the default; the opt-in lossy `--mtp-typical`/`--mtp-tokenv3`
  verifiers change the output distribution). `--mtp-max-depth 2` on the 35B (127.5 tok/s in the short runs, the
  adaptive default 124.8) and 4 on the 27B (37.8 against 36.3).
- DFlash2: the drafter, which mlx-serve quantizes to 8 bits at load on M1-M4, without the MTP head (`--no-mtp`): with
  both loaded, mlx-serve hands a request to MTP when the previous one is still finishing on the server, and the
  benchmark's back-to-back requests sometimes are (2 of 5 timed answers on the 27B, at half the speed). On the 27B a
  tree of up to 7 drafts a round, block 8 (the defaults; block 6 and linear drafts were slower). On the 35B block 4
  (103.9 tok/s, the default block 5 102.4, block 8 88.2) and no tree: mlx-serve verifies trees only on dense models,
  and on a MoE model it switches DFlash off for the rest of an answer once fewer than 1.8 drafts a round are accepted.
  mlx-serve ships no drafter for the 35B and documents DFlash for dense models, but its loader binds this one.
- PLD (prompt lookup) stays on, its default; mlx-serve runs one speculative mode per request (DFlash, else MTP, else
  PLD), so it never ran. A fresh `HOME` per run, 32K context, one sequence.

## Qwen3.5-4B (2026-10-18)

Qwen3.5-4B, the family's small dense model (32 layers: 24 Gated DeltaNet, 8 full attention), at 4-bit on every engine
of the [main comparison](#results), through the same harness, requests and checks. Each engine ran in its fastest
setup for this model: speculative decoding with the model's MTP head wherever the engine has an MTP path (incoai
publishes no DFlash2 drafter for the 4B; the one third-party DFlash2 drafter, nchapman/Qwen3.5-4B-DFlash2, carries no
license and was not tried), oMLX's and MTPLX's settings picked in short runs before the session, and llama.cpp's MTP
at n-max 1, 2, 3 and 6. Configs: the `q35-4b-*` entries in [bench/harness/configs](../bench/harness/configs); result
files: [bench/results/2026-10-18-4b](../bench/results/2026-10-18-4b/).

Kapitsa was the fastest engine on every metric and used the least memory. It wrote 122 tok/s at T = 0.6, 1.12×
Ollama's MLX runner with MTP (109.0) and 1.17× MTPLX (104.2, against Kapitsa's 122.0 in the same bracket), read long
prompts 1.8–2.2× faster than any other engine, and answered follow-ups in 0.05–0.07 s, against 0.09–0.11 s for the
next fastest (MTPLX). Greedy decoding is the exception to a clear lead: Ollama's MLX runner wrote 123.0 tok/s, 1%
below Kapitsa's 124.5 in the same bracket, and Kapitsa's greedy medians in bracket B were 120.4 and 125.3, so at
greedy the two are within run-to-run variation (note 2).

The runs came in two brackets on 2026-10-18, each opened and closed by Kapitsa, one engine at a time under the GPU
lock, in High Power mode on AC power. Compare each row with Kapitsa's row of the same bracket.

- **A**, 11:04–12:05: every configuration, back to back. Bursts from Slack and Chrome flagged the long-prompt phases
  of three runs (oMLX without speculation, MTPLX, llama.cpp without MTP). From 11:42 Slack, Chrome and a video call
  (`cameracaptured`, macOS's camera daemon) used more than 5% of the GPU for sustained stretches, and every run from
  then on was flagged, Kapitsa's closing run included. The A rows are the unflagged runs between 11:04 and 11:40.
- **B**, 13:39–14:12: Kapitsa, then the configurations whose A runs were flagged (oMLX without speculation, MTPLX,
  llama.cpp without MTP and at n-max 6), then Kapitsa again, each engine started after other apps had used less
  than 3% of the GPU for 60 s. Their flagged A runs, in [superseded](../bench/results/2026-10-18-4b/superseded/),
  decoded within 4% of their B runs: oMLX without speculation 85.0 / 83.2 tok/s, MTPLX 102.7 / 104.7, llama.cpp
  64.0 / 63.9, n-max 6 63.9 / 70.7.

| Engine (setting) | Bracket | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** (MTP head)<sup>1</sup> | A | **122.4** (119.3–123.4) | **124.5** (119.0–127.0) | **1,796** | 1,337 | 0.06 | 0.07 | **4.1** | pass |
| **Kapitsa** (MTP head)<sup>1</sup> | B | 122.0 (118.7–124.2) | 120.4 (94.0–123.6) | 1,788 | **1,427** | **0.05** | **0.06** | **4.1** | pass |
| Ollama, MLX runner + MTP (adaptive depth) | A | 109.0 (99.6–115.6) | 123.0 (116.6–124.9)<sup>2</sup> | 758 | 693 | 0.19 | 0.24 | 12.1 | pass |
| MTPLX (sustained profile, depth 1) | B | 104.2 (99.5–107.0) | 106.3 (105.0–109.0) | 780 | 711 | 0.09<sup>6</sup> | 0.11 | 14.3 | pass |
| oMLX + MTP (Lightning MTP) | A | 102.4 (99.5–108.6) | 104.5 (102.8–109.4) | 803 | 726 | 0.25 | 0.48 | 7.3 | pass |
| oMLX (no speculation) | B | 82.4 (82.0–84.9) | 83.9 (83.0–84.6) | 868 | 784 | 0.26 | 0.36 | 6.1 | pass |
| LM Studio (MLX engine) | A | 81.3 (80.4–81.5)<sup>3</sup> | 85.1 (84.3–85.5)<sup>3</sup> | 801 | 747 | 0.41 | 0.70 | 8.4 | pass |
| mlx-lm | A | 73.1 (71.4–75.0) | 76.4 (74.1–78.6) | 784 | 719 | 0.18 | 0.32 | 7.3 | pass |
| llama.cpp + MTP (n-max 2) | A | 69.6 (68.1–73.0) | 73.4 (70.7–74.3) | 650 | 564 | 0.21 | 0.25 | 11.1 | pass |
| llama.cpp + MTP (n-max 1) | A | 69.3 (68.3–69.8) | 71.8 (68.9–72.0) | 662 | 587 | 0.25 | 0.32 | 11.6 | pass |
| Ollama, GGUF + MTP (shipped depth 1) | A | 68.2 (63.4–70.4) | 72.8 (67.0–73.7) | 712 | 575 | 0.15 | 0.25 | 5.7 + 3.2 mapped<sup>4</sup> | pass |
| llama.cpp + MTP (n-max 3) | A | 65.9 (64.2–71.8) | 77.9 (75.9–78.2) | 687 | 609 | 0.21 | 0.24 | 11.1 | pass |
| llama.cpp | B | 63.4 (62.7–63.5) | 65.3 (64.9–65.5) | 724 | 655 | 0.19 | 0.25 | 9.2 | pass |
| llama.cpp + MTP (n-max 6) | B | 62.3 (56.8–69.2) | 71.9 (70.9–75.2) | 689 | 613<sup>5</sup> | 0.16 | 0.27 | 11.3 | pass |
| Ollama, GGUF (MTP off) | A | 59.9 (46.3–65.3) | 64.5 (64.2–66.0) | 733 | 618 | 0.15 | 0.28 | 5.1 + 2.5 mapped<sup>4</sup> | pass |

1. The Kapitsa rows are each bracket's opening run. Its closing runs were flagged for contention
   ([confirmations](../bench/results/2026-10-18-4b/confirmations/)). A's, at 12:04 during the video call, gave
   115.0 / 111.4 tok/s decode, 1,377 / 1,136 tok/s prefill, 0.06 / 0.06 s follow-ups and 4.1 GB. Its prefill
   dropped the most, probably because the call's camera processing shared the Neural Engine, which runs part of
   Kapitsa's dense prefill: its log shows the Neural Engine busy for 98% and 81% of the 4K and 16K prefills, against
   60% and 45% in the opening run. B's closing run, at 14:10 with Slack busy in its long-prompt phases, gave
   121.4 / 125.3 tok/s, 1,774 / 1,436 tok/s, 0.06 / 0.07 s and 4.1 GB.
2. One of the five greedy runs looped (distinct-4-gram 0.47), which speculative decoding speeds up; without it the
   median is 120.8 tok/s. Ollama's MLX runner drafts up to four tokens a step and adapts how many: in the timed
   answers 2.0 a step on average at greedy and 1.7 at T = 0.6, where Kapitsa drafts one. Deeper drafts pay off when
   they are accepted, which happens more often in greedy decoding, so it gains 13% from T = 0.6 to greedy and Kapitsa
   about 2%; at T = 0.6 Kapitsa is 12% faster (122.4 against 109.0).
3. LM Studio stops at 511 of the 512 requested tokens. Its rates use the 511 tokens it reports.
4. Ollama's GGUF runner maps the weights from disk instead of loading them: the first figure is the server's memory,
   the second the resident mapped weights (the model, and with MTP its draft GGUF).
5. Flagged for contention: Chrome used 9.6% of the GPU on average during this prefill. The only flagged figure in
   the table; A's run of this setting read 603 tok/s here, also flagged.
6. MTPLX's SSD session tier writes only what fits above 10 GiB of free disk. Its B run started with 10.0 GiB free,
   and its log shows that tier skipping its writes ("no disk room"); the benchmark's follow-ups came from its
   session cache in memory, as fast as in its A run with 19 GiB free (0.10 / 0.12 s). LM Studio's MLX engine, which
   keeps follow-up checkpoints on disk only while 10 GiB is free, had 18 GiB.

Repetition loops (distinct-4-gram under 0.5; flagged, not invalid) came up in one of the five sampled runs of
Kapitsa in bracket B (median without it 121.0), oMLX without speculation (82.6), llama.cpp (63.4) and llama.cpp at
n-max 1, 3 and 6 (69.2, 65.3, 62.9), and in two of mlx-lm's (73.4). Peak memory is the whole run's peak. Kapitsa
holds the language model only (2.2 GB; it does not run this model's vision tower), while oMLX, LM Studio, MTPLX and
Ollama's MLX runner also load the file's 0.6 GB vision tower and Ollama's GGUF runner the tag's vision projector.
Engines that keep earlier conversations for follow-up turns grow through the run, MTPLX's session cache most of all
(14.3 GB at its peak). Footprint after the sampled decode runs, before any long prompt: mlx-lm 3.0 GB, Kapitsa 3.4,
oMLX 3.8 (4.6 with MTP), Ollama's MLX runner 4.9, LM Studio 5.2, llama.cpp 6.1 (7.7–8.2 with MTP), MTPLX 6.2, and
Ollama's GGUF runner 2.9 + 2.5 mapped (3.4 + 3.2 with MTP). Of the engines that hold their weights in their own
memory, mlx-lm is the only one below Kapitsa there; it grows to 7.3 GB with the 16K prompt.

**Weights.** The MLX engines read lmstudio-community's MLX 4-bit conversion
([lmstudio-community/Qwen3.5-4B-MLX-4bit](https://huggingface.co/lmstudio-community/Qwen3.5-4B-MLX-4bit) @c43ee1d,
2.8 GB: 2.2 GB of language model, its embedding tied to the output layer, and a 0.6 GB bf16 vision tower), the file
`kapitsa pull 4b` and LM Studio's staff pick download. MLX conversions of the small Qwen3.5 models leave out the
model's MTP head, so each engine with an MTP path got one: Kapitsa its catalog's (Qwen's own head from
Qwen/Qwen3.5-4B, bf16, 0.2 GB, 8-bit at load), MTPLX, oMLX and Ollama's MLX runner the head of MTPLX's 4B release
([Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed](https://huggingface.co/Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed)
@550153b, 4-bit, 0.08 GB, with its runtime file), read beside the same file through APFS clones, an index view
(`bench/harness/tools/omlx_mtp_farm.py`) and Ollama's zero-copy import. llama.cpp and Ollama's GGUF runner read
Ollama's `qwen3.5:4b-mtp-q4_K_M` tag: the Q4_K_M model (2.5 GB, blob a9d0a3e1) and, as a separate draft GGUF, its
MTP layer with a Q8_0 copy of the embedding (0.76 GB, blob 83ba6dc1).

**Settings.** As for the larger models (see [what was compared](#what-was-compared)), except:

| Engine | Version measured | Speculative decoding on the 4B |
| --- | --- | --- |
| Kapitsa | 68b7aee, Release build (b3e8fe9 in the result files: the same source under an earlier commit id) | its default for this model: the MTP head, one draft token a step; dense prefill split between the GPU and the Neural Engine |
| llama.cpp | 0.7.0, build b11669 (620d2225c) | MTP with the tag's draft GGUF (`-md`), `--spec-draft-n-max` 1, 2, 3 and 6 (6 is unsloth's recommendation for this model); and off |
| Ollama, GGUF | 0.40.6 (bundled llama.cpp 620d2225c; the app had updated itself from 0.40.5) | built-in MTP at the tag's shipped depth, 1; and off |
| Ollama, MLX runner | 0.40.6 (MLX bebd642) | built-in MTP, adaptive depth up to 4 |
| LM Studio | llmster 0.0.25-1, MLX runtime 1.13.1 (its staff pick `qwen/qwen3.5-4b`) | none: its MLX engine has none for this model |
| mlx-lm | 0.32.2 (mlx 0.32.4) | none for this model |
| oMLX | 0.7.1rc1 | Lightning MTP, oMLX's own MTP path (adaptive depth, up to 3); and none |
| MTPLX | 2.12.2 | MTP, profile `sustained` (its runtime file's), depth 1 |

oMLX's and MTPLX's settings come from short runs just before the session (three sampled and three greedy 512-token
answers each; [tuning](../bench/results/2026-10-18-4b/README.md#tuning)). oMLX: Lightning MTP 101.4 / 103.8 tok/s,
DFlash with z-lab's Qwen3.5-4B drafter (DFlash v1, quantized to 4 bits at load) at most 89.9 / 98.0 (block 5, verify
`adaptive`; block 9 and verify `dflash` were slower), no speculation 84.9 / 81.8. MTPLX: depth 1 104.5 / 108.6, depth
2 103.8 / 109.2, and depth 3, its runtime file's default, 87.7 / 102.6.

### Full precision (bf16), a partial bracket

The same protocol on the model's bf16 weights,
[mlx-community/Qwen3.5-4B-bf16](https://huggingface.co/mlx-community/Qwen3.5-4B-bf16) @491fdc7 (8.5 GB: 7.8 GB of
language model and the same 0.6 GB vision tower), the files `kapitsa pull 4b:bf16` downloads, with the same Qwen MTP
head as the 4-bit model. Four engine setups ran on 2026-10-18 between 15:30 and 15:55, one at a time under the GPU
lock, each started after other apps had used less than 3% of the GPU for 60 s; the MLX engines read Kapitsa's files
through APFS clones. **The bracket is incomplete:** the disk then ran short, so there is no closing Kapitsa run, and
mlx-lm, LM Studio, llama.cpp and Ollama were not run at bf16 (LM Studio needs 10 GiB of free disk for its follow-up
checkpoints, and llama.cpp and Ollama need Ollama's 9.2 GB BF16 GGUF). Result files:
[bench/results/2026-10-18-4b/bf16](../bench/results/2026-10-18-4b/bf16/).

**Kapitsa was not the fastest writer at bf16.** oMLX with z-lab's DFlash drafter wrote 59.4 tok/s at T = 0.6 and
MTPLX at depth 3 wrote 54.2, against Kapitsa's 44.3: 1.34× and 1.23× faster (greedy: 61.5 and 58.8 against 45.1).
Our best explanation: at bf16 a forward pass reads 7.8 GB of weights, so the rate follows how many tokens a pass
yields. oMLX's DFlash drafts up to four tokens a pass (2.5 on average) and got 2.8 tokens out of each pass, MTPLX
drafts three, and Kapitsa's MTP head one, so a Kapitsa pass yields at most two (depth 1 is its default for this model,
and it has no drafter for the 4B). At 4-bit, where a pass reads about a quarter of the bytes, Kapitsa's depth 1 was
the fastest setup measured. Kapitsa still read long prompts 1.7–1.9× faster than the next fastest engine here,
answered follow-ups fastest and used the least memory.

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| oMLX + DFlash (z-lab drafter, block 5) | **59.4** (54.3–63.0) | **61.5** (59.2–69.1) | 859 | 823 | 0.30 | 0.42 | 12.6 | pass |
| MTPLX (sustained profile, depth 3) | 54.2 (49.7–55.7) | 58.8 (57.4–61.6) | 820 | 751 | 0.11 | 0.15 | 20.3<sup>2</sup> | pass |
| **Kapitsa** (MTP head) | 44.3 (42.6–47.3) | 45.1 (44.0–45.6) | **1,737** | **1,401** | **0.06** | **0.07** | **9.7** | pass |
| oMLX (no speculation) | 27.6 (27.0–27.8) | 26.3 (25.9–27.1)<sup>1</sup> | 907 | 812 | 0.27 | 0.41 | 11.8 | pass |

No run of this bracket was flagged for contention. Kapitsa's run (15:30) came first; there is no closing run to check
it against. One greedy run of Kapitsa, of oMLX + DFlash and of MTPLX looped (distinct-4-gram under 0.5; flagged, not
invalid).

1. Three of oMLX's five greedy runs without speculation looped, and macOS swapped 12,062 pages (about 190 MB) during
   that phase; its sampled median, 27.6 tok/s, is the cleaner figure. Reading 7.8 GB a pass at the M4 Pro's 273 GB/s
   bounds decoding without speculation at about 33 tok/s.
2. MTPLX's peak includes its session cache, which keeps earlier conversations for follow-up turns.

**Settings.** As at 4-bit, except where short runs before the bracket measured faster (three sampled and three greedy
answers each; [tuning](../bench/results/2026-10-18-4b/bf16/tuning/)). oMLX: DFlash with z-lab's Qwen3.5-4B drafter
([z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) @9a1996c, DFlash v1, quantized to 4 bits
at load), block 5, verify `adaptive`, 59.1 / 63.2 tok/s, against block 9 57.0 / 56.8 and Lightning MTP with the bf16
head 49.5 / 48.2 (at 4-bit Lightning MTP had been the faster). MTPLX: depth 3, 54.5 / 59.0, against depth 2 52.4 /
54.9 and depth 1 43.7 / 41.0, with Qwen's bf16 head (Kapitsa's file) and the runtime file of MTPLX's 4B release.
Kapitsa: its defaults, the bf16 weights with the MTP head (`kapitsa pull 4b:bf16`).

## Your Mac: `kapitsa bench`

`kapitsa bench` runs this protocol for Kapitsa alone, from the `kapitsa` binary: no Python, no configs, no weights to
place by hand (the catalog's models, downloaded with their drafters). It sends the same requests as `bench.py`, starts
the server as `compare_engines.py` does, and flags runs alike, so its numbers compare with the tables above; it
records the Mac and its conditions, and with `--submit` adds the result to the
[community results](../bench/community/RESULTS.md), checked as [bench/community](../bench/community/README.md) says.
The [usage guide](usage.md#benchmarking-this-mac) has its options.

## Run benchmarks locally

Everything above can be rerun with the harness in [bench/harness](../bench/harness/README.md) (Python 3, standard
library only):

1. Build Kapitsa and install the engines you want to compare; the harness README says how for each, with the
   settings used here.
2. Download the weights into `~/models` (the commands, with pinned revisions, are in the harness README).
3. Point the harness at them: `cd bench/harness && cp local.env.example local.env`, then edit the paths.
4. Check the plan without starting anything: `python3 compare_engines.py configs/*.json --dry-run`.
5. Run, one engine at a time: `python3 compare_engines.py configs/kapitsa.json configs/llama.cpp.json --only 'q36-35b-*'`.
   Each configuration takes 2 to 20 minutes and holds the GPU lock throughout.
6. Make the tables: `python3 compare_table.py`.

For numbers you can compare across engines, run them all in one session with the same power mode, plugged in, with
browsers and other GPU users closed; the harness flags runs that other apps disturbed. Results depend on the Mac:
other chips, memory sizes and macOS versions will give different numbers.
