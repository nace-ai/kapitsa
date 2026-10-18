# Benchmarks

How fast Kapitsa runs Qwen3.6-35B-A3B and Qwen3.8-27B on a MacBook Pro (Nemotron 3.5 Bolt has
[its own section](#nemotron-35-bolt-74b-a9b-2026-10-18)), compared with the engines people run these models with on a
Mac: llama.cpp, Ollama, LM Studio, mlx-lm, oMLX and MTPLX. Each engine ran at its latest release
with its own best settings, on the same 4-bit weights wherever it accepts them, through the same requests and the
same checks. The harness and its configs are in [bench/harness](../bench/harness/README.md), and every result file
behind these tables is in [bench/results/2026-10-17](../bench/results/2026-10-17/).

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
