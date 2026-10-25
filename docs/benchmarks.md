# Benchmarks

Kapitsa against the engines people run the same models with on a Mac: mlx-serve, oMLX, Splash, Ollama, MTPLX, MLX
(mlx-lm or mlx-vlm), llama.cpp, LM Studio and, for Flint, Recursive's rec. Every engine runs at its latest release in
its fastest setup, on the same weights wherever it reads them, through the same requests and the same checks, one
engine at a time. The README's [poster and per-Mac tables](../README.md#benchmarks) show the results; this page says
where each of their numbers comes from and how it was measured.

- Every run's file, with its server log: [bench/results](../bench/results/). One line per run:
  [bench/README.md](../bench/README.md).
- Your own Mac: `kapitsa bench` ([below](#run-it-yourself)), and the [community results](../bench/community/RESULTS.md).

[At a glance](#at-a-glance) · [How we measure](#how-we-measure) · [M5 Max](#m5-max) · [M5 Pro](#m5-pro) ·
[M4 Pro](#m4-pro) · [M3 Pro](#m3-pro) · [M1 Pro](#m1-pro) · [Deep dives](#deep-dives) · [Run it yourself](#run-it-yourself) ·
[Archive](#archive)

## At a glance

Decode is the writing speed of a 512-token coding answer at T 0.6 (Flint at its card's T 1.0). "Lead" is Kapitsa's
decode over the next fastest engine on that model.

| Mac | Kapitsa's decode lead | Where another engine is ahead |
| --- | --- | --- |
| [M5 Max, 128 GB](#m5-max) | 27B **1.28x** Splash · 35B-A3B **1.21x** Splash · Bolt **2.3x** mlx-vlm · Lightning 1.41x · Super-VL 1.42x · Dory 1.57x | Flash-Next decode: mlx-serve 108 against 106. Prefill: Splash on the 27B (856 against 805) and the 35B-A3B (6,386 against 5,963); mlx-serve, oMLX and MTPLX on Flash-Next |
| [M5 Pro, 48 GB](#m5-pro) | 35B-A3B **1.11x** Splash · Lightning 1.28x · Bolt 1.93x · Dory 1.32x · Flint 1.03x rec | 27B decode: mlx-serve 78 against 77, inside the runs' spread. Prefill: Splash on the 35B-A3B (1.25x) and the 27B (1.11x) |
| [M4 Pro, 48 GB](#m4-pro) | 4B **1.47x** mlx-serve · 27B 1.17x · 35B-A3B 1.19x Splash · Lightning 1.72x · Bolt 1.97x · Dory 1.33x · Flint 1.01x rec | Flint prefill: llama.cpp 977 against 968 |
| [M3 Pro, 36 GB](#m3-pro) | 4B 1.79x · 9B **2.90x** · 27B 1.77x · 35B-A3B 1.84x · Lightning 1.92x · Flint 1.45x · Dory 1.21x | Nano 4B decode: llama.cpp 34.9 against 34.4. Nano 9B v2: mlx-lm 24.9 against 20.9 (both on lighter 4-bit files) |
| [M1 Pro, 16 GB](#m1-pro) | 4B 1.56x · 9B 1.54x · 35B-A3B 2-bit 1.37x · Flint 2-bit 1.42x · Dory 1.30x · Nano 4B 1.32x · Nano 9B v2 1.39x | — |

Kapitsa has the fastest decode in 32 of the 36 model and Mac pairs.

<a id="how-the-readme-poster-was-measured"></a><a id="the-readmes-tables"></a>

## How we measure

The [harness](../bench/harness/README.md) starts an engine's server, warms it up, waits for a quiet GPU and talks to it
over its OpenAI-compatible API, as a coding agent does. `kapitsa bench` runs the same protocol from the binary.

| Phase | What is measured |
| --- | --- |
| Decode, T = 0.6 | A 512-token coding answer (an LRU cache module with tests), 5 runs, the median. The rate is taken over the middle of the stream (the first 16 and last 4 tokens dropped), so the time to first token is not counted. top_p 0.95, top_k 20. |
| Decode, greedy | The same answer at temperature 0, 5 runs: with the same weights every engine writes nearly the same text. |
| Prefill, 4K and 16K | A cold prompt of 4,133 / 16,286 tokens cut from the frozen corpus, a random nonce at its start so nothing is reused. Prompt tokens over time to first token. |
| Cached follow-up | Three follow-up turns on the grown conversation, the median time to first token. |
| Peak memory | `phys_footprint_peak` of the server's whole process tree, in GB of 2^30 bytes (GiB), as every memory and size figure on this page; memory-mapped weights listed apart. |

**What makes a run count.** One engine at a time, under one GPU lock. Every request sets every sampling parameter and
turns thinking off both ways, and any reasoning text makes the run invalid. Cold prompts must come back within 1% of
the reference token count. A decode run must end at exactly 512 tokens, and every answer is saved: a distinct-4-gram
ratio below 0.5 flags a repetition loop, which flatters speculative decoding. Every result records the exact command,
the engines' versions, the weights, macOS, power source and swap.

**Flags.** *Contention*: another process used more than 5% of the GPU during a phase (the app driving a session can
be declared ambient instead: its load is then recorded, not flagged). *Loop*: a decode answer repeated itself.
*Invalid*: a check failed; invalid runs never make a cell.

**How a cell is chosen.**

- Each engine in its fastest setup: speculative decoding wherever it has one (MTP, DFlash2, DSpark, LiLiCorr), each
  depth or block it exposes tried in short runs, the configs' notes say which won. The cell is the fastest decode at
  T 0.6 and the 4K prefill of that same run.
- Kapitsa runs its catalog default. Since [#396](https://github.com/nace-ai/kapitsa/pull/396) every Kapitsa decode
  goes through KFlash, whose source is the model's drafter, its MTP head, prompt lookup or Dory's own drafts.
- Flint decodes at its card's T 1.0, and its reasoning is counted as part of the answer (`--always-reasons`).
- 🟩 marks the fastest decode. A lead inside the runs' spread is still ranked by the measured value, and when Kapitsa
  reads lower the marker goes to the other engine.
- × the engine cannot run that model on that Mac · — not measured.

## Results by Mac

<a id="m5-max"></a>

### MacBook Pro M5 Max, 128 GB

**2026-10-25: the Qwen3.8 27B, Qwen3.6 35B-A3B and Nemotron 3.5 Bolt rows** ([results](../bench/results/2026-10-25-m5-max/README.md)).
MacBook Pro (16-inch, 2026), 40-core GPU, macOS 26.4, AC power. Kapitsa 0.2.17 (main `cd20dcc`) and every engine at its
latest release: Splash 1.4.0, mlx-serve 26.10.3, oMLX 0.7.3.dev2, mlx-vlm 0.7.10, mlx-lm 0.32.2, MTPLX 2.12.2, Ollama
0.40.11 and LM Studio with its MLX runtime 1.13.1, the build it picks on an M5. The MLX engines read the same 4-bit
folders; LM Studio, mlx-serve and Ollama's MLX runner read clones of them.

| Qwen3.6 35B-A3B | Prefill 4K / 16K | Decode T 0.6 / greedy | Cached follow-up 4K / 16K | Peak |
| --- | ---: | ---: | ---: | ---: |
| **Kapitsa** (KFlash, DFlash2 drafter) | 5,963 / 5,270 | **339 / 364** | **0.04 / 0.04 s** | **21.8 GB** |
| Splash (DFlash2) | **6,386 / 5,607** | 281 / 288 | 0.07 / 0.07 s | 22.0 GB |
| mlx-serve (MTP) | 3,101 / 4,847 | 238 / 251 | 0.12 / 0.17 s | 22.7 GB |
| oMLX (DFlash2) | 3,005 / 3,034 | 189 / 238 | 0.17 / 0.25 s | 22.7 GB |
| Ollama (MLX runner, MTP) | 3,077 / 3,539 | 162 / 166 | 0.09 / 0.16 s | 25.3 GB |
| MLX (mlx-vlm, DFlash2) | 4,439 / 4,415 | 138 / 144 | 0.24 / 0.74 s | 23.2 GB |
| LM Studio | 4,672 / 4,171 | 136 / 140 | 0.19 / 0.31 s | 22.9 GB |
| MTPLX (MTP) | 4,291 / 3,845 | 130 / 110 | 0.05 / 0.07 s | 30.1 GB |

| Qwen3.8 27B | Prefill 4K / 16K | Decode T 0.6 / greedy | Cached follow-up 4K / 16K | Peak |
| --- | ---: | ---: | ---: | ---: |
| **Kapitsa** (KFlash, DFlash2 drafter)<sup>1</sup> | 805 / 801 | **147 / 146** | **0.13 / 0.13 s** | **20.1 GB** |
| Splash (DFlash2) | **856 / 830** | 115 / 113 | 0.21 / 0.19 s | 22.9 GB |
| mlx-serve (DFlash2 draft tree) | 643 / 567 | 113 / 121 | 0.14 / 0.13 s | 23.0 GB |
| oMLX (DFlash2) | 715 / 752 | 81 / 85 | 0.31 / 0.49 s | 22.0 GB |
| MTPLX (MTP) | 666 / 615 | 61 / 63 | 0.13 / 0.20 s | 39.5 GB |
| Ollama (MLX runner, MTP) | 676 / 648 | 54 / 58 | 0.18 / 0.20 s | 27.6 GB |
| MLX (mlx-vlm, MTP) | 673 / 657 | 49 / 51 | 0.35 / 0.52 s | 21.0 GB |
| LM Studio | 616 / 614 | 31 / 30 | 0.25 / 0.29 s | 21.9 GB |

1. Its prefill phases carry the contention flag (the session's desktop app at about 5% of the GPU). For most of the
   session that app was declared ambient, and its load reached 7-29% in some other engines' prefill phases (mlx-serve's
   27B, MTPLX's 27B, mlx-vlm's Bolt; 4.3% at most in any decode phase); those runs are being repeated without the
   exemption ([per run](../bench/results/2026-10-25-m5-max/README.md)).

Nemotron 3.5 Bolt 74B-A9B (2/4/8-bit mix): Kapitsa **983 / 158** tok/s (prefill 4K / decode), mlx-vlm 814 / 68, oMLX
749 / 55 (without a drafter: its LiLiCorr path never became ready in 0.7.3.dev2). One of Kapitsa's five decode answers
looped.

- **Kapitsa writes fastest on all three models and answers follow-ups fastest; Splash reads prompts fastest.** Splash
  splits a long prompt's FFN between the GPU and the Neural Engine; on the 27B Kapitsa does too, and the two are 6%
  apart.
- Since the previous session (2026-10-22, main `fd85910`) Kapitsa's own cells moved from 4,702 / 325 to 5,963 / 339 on
  the 35B-A3B and from 459 / 140 to 805 / 147 on the 27B.
- Several engines read lower than on their 2026-10-18 releases with the same configs (oMLX 189 against 232 on the
  35B-A3B, mlx-serve 238 against 264); none of those runs is flagged. Their settings were chosen on the earlier
  releases and were not re-tuned.

**Cells from earlier sessions, not yet re-measured.**

- Qwen3.8-Flash-Next 180B-A7B: Kapitsa 1,165 / 106 on 2026-10-22 (main `fd85910`, MTP head, contention-flagged); the
  other engines on 2026-10-18, each at the release then current
  ([results](../bench/results/2026-10-18-m5-max-flash-next/README.md)). mlx-serve (its vendor's 4.7-bit pack) leads,
  1,720 / 108, then oMLX with MTP (1,260 / 107) and Kapitsa (1,165 / 106). Ollama's MLX runner fails every 4K and 16K prompt there
  (out of GPU memory), LM Studio refuses its 119 GB GGUF on 128 GB, and Kapitsa uses the least memory (70.5 GB;
  mlx-serve 78.8, oMLX 104).
- Lightning, Super-VL and Dory: Kapitsa on 2026-10-22 (main `fd85910`, Dory `039ddea`), every run contention-flagged
  and Super-VL's decode loop-flagged, so read Super-VL's 94 as an upper bound; the other engines on 2026-10-18
  ([Super-VL](../bench/results/2026-10-18-m5-max/README.md), [Lightning](../bench/results/2026-10-18-m5-max-lightning/README.md),
  [the session](../bench/results/2026-10-18-m5-max-session/README.md)). mlx-serve 26.10.2 loads neither Super-VL
  checkpoint; MTPLX runs its text-only conversion without MTP.
- llama.cpp on the 27B and 35B-A3B: b11669 on 2026-10-18 (MTP, n-max 3 and 2), 516 / 51 and 2,881 / 171 (prefill 4K /
  decode); not re-measured yet, so the README header's llama.cpp bar (171) is that earlier session's cell.
- Flint: not measured on this Mac yet.

<a id="m5-pro"></a>

### MacBook Pro M5 Pro, 48 GB

MacBook Pro (16-inch, 2026), 20-core GPU, macOS 26.5.2, AC power. Most engine cells are the 2026-10-19 session, each
engine's fastest setting from its newest run ([results](../bench/results/2026-10-19-m5pro-48gb/README.md)); Kapitsa's
cells are newer.

- **Qwen3.8 27B.** Kapitsa 606 / 77 (16K 618; greedy 83.9): the median of three Kapitsa-only runs on 2026-10-23, main
  `fcd597b`, after [#371](https://github.com/nace-ai/kapitsa/pull/371) and [#381](https://github.com/nace-ai/kapitsa/pull/381)
  ([results](../bench/results/2026-10-23-m5-pro-27b-381/README.md)). mlx-serve's cell is the mean of its two runs in a
  2026-10-22 interleaved pair on `16f2b6a`, 464 / 78: its 2.1% lead (78.2 against 76.6) is inside the spread and crosses
  sessions, and the marker is its. Kapitsa reads the 4K prompt 1.31x as fast; mlx-serve answers cached follow-ups
  sooner (0.14 / 0.18 s against 0.21 / 0.23 s).
- **Qwen3.6 35B-A3B.** Kapitsa 2,802 / 201 (16K 2,479; greedy 222): the median of three flag-free runs on 2026-10-23,
  main `920542c`, after [#365](https://github.com/nace-ai/kapitsa/pull/365) ([results](../bench/results/2026-10-23-m5-pro-35b-365/README.md)).
- <a id="splash-140-against-kapitsa-m5-pro-2026-10-22"></a>**Splash 1.4.0** on 2026-10-22, two runs per model between
  two Kapitsa runs, every Splash run contention-flagged (a browser at 6.5-10% of the GPU): 674 / 72 on the 27B and
  3,514 / 181 on the 35B-A3B ([results](../bench/results/2026-10-22-m5-pro-splash/README.md)). **Splash reads prompts
  faster** (1.25x Kapitsa's 35B-A3B cell at 4K, 1.11x its 27B cell); Kapitsa writes faster (11% and 6%).
- **Lightning:** Kapitsa 3,141 / 159, one run on 2026-10-21 ([results](../bench/results/2026-10-21-m5pro-48gb/README.md)).
  **Dory:** per-loop KV, 70 tok/s ([results](../bench/results/2026-10-22-m5pro-dory/README.md)).
- **Flint:** see [Flint on the M5 Pro](#recursive-flint-40b-a25b-m5-pro-2026-10-22).
- Settings: MLX is mlx-vlm with MTP on the 4B and 27B and with DFlash2 on the 35B-A3B, mlx-lm on Lightning and Dory,
  mlx-vlm on Bolt; oMLX MTP on Lightning and DFlash2 on the 27B and 35B-A3B; mlx-serve MTP except its DFlash2 draft tree
  on the 27B; Ollama its MLX runner with MTP; llama.cpp MTP n-max 3 (2 on the 35B-A3B and Lightning).

<a id="m4-pro"></a>

### MacBook Pro M4 Pro, 48 GB

MacBook Pro (16-inch, 2024), 20-core GPU, macOS 27.0.1, High Power mode, AC power.

- **Kapitsa's cells** (4B 201, 27B 63, 35B-A3B 161, Lightning 112, Bolt 69 tok/s): main with KFlash trees on
  2026-10-21 ([KFlash](#kflash-trees-m4-pro-2026-10-21)). Prefill from the newest unflagged runs (2026-10-19).
- **The other engines:** the 2026-10-17 to 10-19 sessions ([first comparison](../bench/results/2026-10-17/),
  [the table filled in](../bench/results/2026-10-19-m4pro/README.md)); Splash 1.4.0 (27B 53, 35B-A3B 135 tok/s) on
  2026-10-21 in the KFlash session, without a prefill run (its cells show a dash before the slash).
- **Lightning and Bolt:** NVIDIA's NVFP4 Lightning with its DSpark drafter on Kapitsa and mlx-vlm (oMLX and llama.cpp
  not measured); Bolt's 24 GB checkpoint on Kapitsa, oMLX and mlx-vlm.
- **Dory:** 2 loops; Kapitsa per-loop KV, llama.cpp NVIDIA's Dory fork on its Q5_K_M, mlx-lm's Dory pull request on a
  4-bit conversion ([results](../bench/results/2026-10-19-dory/README.md)).
- **Flint:** see [Flint on the M4 Pro](#recursive-flint-40b-a25b-m4-pro-2026-10-22).
- **A real opencode session** on the 35B-A3B, replayed request by request: Kapitsa 49 s, mlx-serve 62 s, oMLX 66 s
  (2026-10-18); with KFlash 51.4 s against Splash's 57.0 s (2026-10-21)
  ([results](../bench/results/2026-10-18-agent-session/README.md)).

<a id="m3-pro"></a>

### MacBook Pro M3 Pro, 36 GB (macOS 14)

MacBook Pro (16-inch, 2023), 18-core GPU, macOS 14.1, AC power, so Kapitsa runs its macOS 14 kernels. One run per cell
of the harness's protocol, the models one after another on 2026-10-22/23
([results](../bench/results/2026-10-22-m3pro-36gb/README.md), [#369](https://github.com/nace-ai/kapitsa/pull/369)):
Kapitsa from main (`e86dea7` for the Qwen models, `c490f81` for the Nemotron models), each in its catalog default; the
other engines in the M4 Pro's setup (Ollama 0.40.8's MLX runner, MTPLX 2.12.2, mlx-lm / mlx-vlm, llama.cpp b11804).

- mlx-serve, oMLX and Splash need a newer macOS (26.2, 15.0 and 26.4): their binaries abort at launch on 14.1. LM Studio
  was not installed.
- **Kapitsa is behind on the two Nano models**, by 1% on the Nano 4B (34.4 against llama.cpp's 34.9) and 16% on the
  Nano 9B v2 (20.9 against mlx-lm's 24.9): Kapitsa runs NVIDIA's own FP8 and NVFP4 checkpoints, which read more bytes
  per token than the others' 4-bit files.
- **Flint:** Kapitsa (main `cff1b1e`) and llama.cpp b11804; rec refuses a 36 GB Mac
  ([Flint on the M3 Pro](#recursive-flint-40b-a25b-m3-pro-36-gb-macos-14-2026-10-22)).
- Several decode phases carry the loop flag; the cells' notes in `tools/readme_images/readme_images.json` say which.

<a id="m1-pro"></a><a id="macbook-pro-m1-pro-16-gb-kapitsa-026-2026-10-22"></a>

### MacBook Pro M1 Pro, 16 GB

MacBook Pro (14-inch, 2021), 14-core GPU, macOS 27.0.1, AC power. Kapitsa's cells are the 0.2.6 release (`76a4970`):
`kapitsa bench`'s full protocol on each catalog model that fits, one after another on 2026-10-22, each in its catalog
default ([community results](../bench/community/results/), `kapitsa-bench-2026-10-22-m1-pro-*`). The other engines'
cells are the 2026-10-19 session ([results](../bench/results/2026-10-19-m1pro-16gb/README.md)).

| Model | Decoding | Decode T 0.6 / greedy | Prefill 4K / 16K | Cached follow-up 4K / 16K | Peak |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B, 4-bit | KFlash (DFlash drafter) | 97.2 / 100.3 | 401 / 354 | 0.12 / 0.15 s | 4.4 GB |
| Qwen3.5-9B, 4-bit | KFlash (DFlash drafter) | 56.8 / 57.8 | 217 / 202 | 0.18 / 0.21 s | 8.0 GB |
| Qwen3.6-35B-A3B, 2-bit experts | plain | 52.2 / 53.4 | 462 / 405 | 0.16 / 0.23 s | 11.2 GB |
| Nemotron Dory1.0 4B, NVFP4 | 2 loops, last-loop KV | 39.2 / 40.2 | 253 / 214 | 0.26 / 0.42 s | 3.6 GB |
| Nemotron 3 Nano 4B, FP8 | prompt lookup | 37.0 / 37.5 | 447 / 401 | 0.14 / 0.19 s | 5.0 GB |
| Nemotron Nano 9B v2, NVFP4 | plain | 24.5 / 24.8 | 189 / 180 | 0.54 / 0.57 s | 7.1 GB |

**By prompt length**, 1K to 32K tokens against the next engine on each model, the same afternoon
([results](../bench/results/2026-10-22-m1pro-16gb/README.md)): Kapitsa reads faster at every size on all six models
(1.13-1.67x) and decodes faster on the three Qwen models at every size. Two gaps remain: Dory's decode from 8K up (at
32K Kapitsa writes 12 tok/s against mlx-lm's 19) and the Nano 9B v2, 1-5% behind mlx-lm on its lighter 4-bit file.

The Qwen3.6-35B-A3B 2-bit row is unsloth's UD-IQ2_XXS GGUF on Kapitsa and llama.cpp b11669; the MLX engines don't read
GGUF. For Flint's 2-bit row see [Flint with 2-bit experts](#recursive-flint-40b-a25b-with-2-bit-experts-m1-pro-16-gb-2026-10-22).

## Deep dives

<a id="kflash-trees-m4-pro-2026-10-21"></a>

### KFlash: a tree of guesses per verify pass (M4 Pro, 2026-10-21)

[#187](https://github.com/nace-ai/kapitsa/pull/187) made KFlash the default: the drafter's candidates are verified as a
best-first tree instead of one chain, in the same pass, sized by a budget measured on each Mac
([how it works](kflash.md)). Main before it (`617f6a2`) against main after it (`8b190f5`), Release builds, on the M4 Pro
([results](../bench/results/2026-10-21-kflash-trees/)):

| Model | Decode T 0.6, before → after | Greedy, before → after |
| --- | ---: | ---: |
| Qwen3.5-4B | 185.7 → **200.8** (+8.1%) | 193.8 → **213.7** (+10.3%) |
| Qwen3.8-27B | 60.4 → **60.5** (+0.2%) | 61.1 → **64.4** (+5.5%) |
| Qwen3.6-35B-A3B | 168.1 → **160.6** (-4.4%) | 177.6 → **177.2** (-0.2%) |
| Nemotron 3.5 Lightning 30B-A3B | 104.3 → **111.7** (+7.1%) | 115.7 → **113.7** (-1.7%) |
| Nemotron 3.5 Bolt 74B-A9B | 64.3 → **68.8** (+6.9%) | 69.4 → **71.2** (+2.6%) |

- One harness run per build: this Mac's runs of one build spread 3-8% at T 0.6. The per-cycle A/B (`ni_run`, five runs
  per prompt, interleaved) reads +2-10% on the 4B and 27B and +0.6% on the 35B-A3B.
- On the dense models the trees' greedy output is identical to plain decoding on every prompt tried.
- Long agent turns gain least: the chain already accepts 2.9-6.4 tokens per cycle there, and a tree cycle costs 5-16%
  more, so those turns land between -5% and +3%. The 16 GB M1 Pro's 4B: +5.6% at T 0.6.

<a id="quality-qwen36-35b-a3b-against-bf16-2026-10-19"></a>

### Output quality: Qwen3.6 35B-A3B against bf16 (2026-10-19)

How far each precision's next-token distributions are from the bf16 model's, teacher-forced over the held-out sequences
of [bench/quality](../bench/quality/README.md) (C++, Python, English prose and a coding-agent chat, 4 x 1023 positions;
`math` reported apart), KL(bf16 || precision) exact over the whole vocabulary
([result file](../bench/results/2026-10-19-quality/qwen3.6-35b-a3b-kl-vs-bf16.json)):

| Precision | KL mean | KL median | KL p95 | Same top token | Perplexity | KL on `math` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | | | | | 5.570 | |
| 8-bit | 0.0168 | 0.0012 | 0.025 | 96.7% | 5.571 | 0.0064 |
| 6-bit | 0.0228 | 0.0033 | 0.056 | 95.9% | 5.602 | 0.0109 |
| 4-bit, the `35b` default | 0.1161 | 0.0272 | 0.341 | 89.1% | 5.783 | 0.0925 |

The 4-bit default is the speed choice; `--model 35b:6bit` is five times closer to bf16, with perplexity within 0.6%.
Nemotron Dory1.0 4B's NVFP4 checkpoint is a KL of 0.020 from its BF16 checkpoint ([precisions](usage.md#precisions)).

### Recursive Flint 40B-A2.5B

Recursive's coding MoE ([#286](https://github.com/nace-ai/kapitsa/pull/286)) on Kapitsa and rec 0.1.0 (Recursive's own
engine) with Recursive's 4-bit MLX export, and llama.cpp b11804, its first build with Flint, on Recursive's Q4_1 GGUF.
Decode at the card's T 1.0; each cell is the mean of that engine's runs.

<a id="recursive-flint-40b-a25b-m5-pro-2026-10-22"></a>

#### M5 Pro, 48 GB (2026-10-22, by prompt length 2026-10-24)

| Engine | Decode T 1.0 / greedy | Prefill 4K / 16K | Follow-up 4K / 16K | Peak |
| --- | ---: | ---: | ---: | ---: |
| **Kapitsa** with [#345](https://github.com/nace-ai/kapitsa/pull/345) (2 runs) | **148.9** / – | **2,691 / 2,457** | **0.10 / 0.12 s** | **23.7 GB** |
| rec 0.1.0 (2 runs) | 144.9 / **151.9** | 2,138 / 1,941 | 0.28 / 0.28 s | 29.1 GB |
| llama.cpp b11804 (1 run) | 92.4 / 92.8 | 2,072 / 1,773 | 0.22 / 0.28 s | 26.9 GB |

Kapitsa's cell is #345's serve A/B with Flint's prefill attention on the M5's matrix units, one run per pass
([results](../bench/results/2026-10-22-flint-m5pro-qattn/README.md)); rec's and llama.cpp's cells are the 2026-10-22
session's ([results](../bench/results/2026-10-22-flint-m5pro/README.md)), rec's runs with the contention flag (a browser at
3-10% of the GPU), llama.cpp's had no foreign load. By prompt length, Kapitsa 0.2.15 (`71d6891`) against llama.cpp
b11840, 3-5 cold requests a point ([results](../bench/results/2026-10-24-flint-m5pro-context/README.md)):

| Median tok/s | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kapitsa prefill | **2,428** | **3,043** | **3,126** | **3,041** | **2,754** | **2,289** | **1,694** | **1,110** |
| llama.cpp prefill | 1,893 | 2,208 | 2,172 | 2,053 | 1,766 | 1,355 | 914 | 549 |
| Kapitsa decode | **145.6** | **142.7** | **137.8** | **131.3** | **116.9** | **97.8** | **74.1** | **49.1** |
| llama.cpp decode | 90.7 | 89.0 | 86.5 | 83.0 | 73.5 | 61.5 | 46.7 | 30.8 |

Kapitsa reads 1.28x as fast as llama.cpp at 1K and 2.02x at 128K, and writes 1.58-1.60x as fast at every length.

<a id="recursive-flint-40b-a25b-m4-pro-2026-10-22"></a>

#### M4 Pro, 48 GB (2026-10-22)

| Engine | Decode T 1.0 / greedy | Prefill 4K / 16K | Follow-up 4K / 16K | Peak |
| --- | ---: | ---: | ---: | ---: |
| **Kapitsa** (2 runs) | **127.9** / 131.6 | 968 / **817** | **0.13 / 0.15 s** | **23.3 GB** |
| rec 0.1.0 (2 runs) | 126.6 / **132.4** | 796 / 699 | 0.41 / 0.44 s | 27.9 GB |
| llama.cpp b11804 (1 run) | 79.6 / 79.9 | **977** / 786 | 0.23 / 0.31 s | 26.9 GB |

By prompt length, 1K to 128K: Kapitsa reads 1.09-1.22x as fast as rec at every length and passes llama.cpp from 16K
(1.51x at 128K); it writes level with rec up to 4K and 1.30x faster at 128K, and 1.6-1.9x llama.cpp's speed throughout
([results](../bench/results/2026-10-22-flint-m4pro/README.md)).

<a id="recursive-flint-40b-a25b-m3-pro-36-gb-macos-14-2026-10-22"></a>

#### M3 Pro, 36 GB, macOS 14 (2026-10-22)

Kapitsa (main `cff1b1e`) runs [#303](https://github.com/nace-ai/kapitsa/pull/303)'s macOS 14 path (Flint's fused
kernels need Metal 3.2); rec 0.1.0 refuses the Mac ("requires 48 GiB"). On AC power, no foreign GPU load
([results](../bench/results/2026-10-22-flint-m3pro-ac/README.md)):

| Engine | Decode T 1.0 / greedy | Prefill 4K / 16K | Follow-up 4K / 16K | Peak |
| --- | ---: | ---: | ---: | ---: |
| **Kapitsa** (2 runs) | **73.6 / 74.5** | **801 / 672** | **0.16 / 0.19 s** | **23.0 GB** |
| llama.cpp b11804 (2 runs) | 50.9 / 51.0 | 772 / 645 | 0.37 / 0.50 s | 26.6 GB |

Kapitsa writes 1.30-1.46x as fast as llama.cpp at every length both reached (1K to 64K) and still 24.8 tok/s at 128K;
llama.cpp reads 0-5% faster from 2K to 16K, Kapitsa from 32K.

<a id="recursive-flint-40b-a25b-with-2-bit-experts-m1-pro-16-gb-2026-10-22"></a>

#### 2-bit experts on a 16 GB M1 Pro (2026-10-22)

The 4-bit export (22.6 GB) does not fit 16 GB; unsloth's `flint-UD-IQ2_XXS.gguf` (11.9 GB) does. Kapitsa's
`flint:16gb` keeps its routed experts in their IQ blocks ([#324](https://github.com/nace-ai/kapitsa/pull/324)) and
requantizes the rest; llama.cpp (master `d772569`) runs the file as it is. Runs alternated L K L K, an 8K context, the 4K
prompt only ([results](../bench/results/2026-10-22-flint-16gb-m1pro/README.md)):

| Engine | Decode T 1.0 | Greedy | Prefill 4K | Follow-up 4K | Peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Kapitsa** (`2b5ab35`) | **60.9 / 61.1** | **61.1 / 61.1** | **414 / 413** | **0.21 / 0.20 s** | **11.6 GB** |
| llama.cpp `d772569` | 42.2 / 42.7 | 42.9 / 42.1 | 355 / 346 | 0.55 / 2.23 s | 11.9 GB |

Kapitsa writes 1.44x as fast on the same file. The requantized tensors cost quality: against llama.cpp on the GGUF as
stored, Kapitsa's next-token distributions are at KL 0.22 (median 0.018), the same top token 89% of the time. The Mac
swapped in some phase of every run (llama.cpp 12K-129K pages a phase, Kapitsa 2K-16K).

<a id="qwen38-flash-next-streamed-from-the-ssd-kapitsa-against-llamacpp-m5-pro-48-gb-2026-10-24"></a>

### Qwen3.8-Flash-Next streamed from the SSD (M5 Pro 48 GB, 2026-10-24)

Flash-Next does not fit in 48 GB, so both engines stream routed experts from the internal disk: Kapitsa main `4b087c8`
(`ni_run`, its Q2_0 pack, the expert store pinned at 24 GB) against llama.cpp `0a9ef00` (`-ngl 99 -fa on`, a Q2_0 GGUF,
mmap), greedy, no speculative decoding, two legs per engine and length in ABBA order
([results](../bench/results/2026-10-24-m5-pro-flash-next-ssd/README.md)):

| Prompt | Kapitsa prefill / decode | llama.cpp prefill / decode | Prefill | Decode (not comparable, re-measure pending) |
| --- | ---: | ---: | --- | --- |
| 85 tokens | 52.7 / 40.5 | **255.0** / 39.0 | Kapitsa 79% slower | not comparable, re-measure pending |
| 1K | 445.0 / 40.8 | **617.5** / 38.3 | Kapitsa 28% slower | not comparable, re-measure pending |
| 4K | 558.2 / 40.0 | 569.4 / 37.1 | level | not comparable, re-measure pending |
| 16K | 555.0 / 34.7 | 549.7 / 36.5 | level | not comparable, re-measure pending |
| 32K | 549.9 / 32.7 | 535.4 / 35.6 | Kapitsa 2.7% faster | not comparable, re-measure pending |

**Decode: not comparable, re-measure pending.** These Kapitsa runs did not stop at the end of the turn (`<|im_end|>`) and the 1K and 32K prompts have no assistant header, so at 1K, 4K and 32K the answer ended after 2, 31 and 63 tokens and the rest of the timed tokens is text generated after it; at 16K Kapitsa stopped at 47 tokens against llama.cpp's 128.

**Not the fastest everywhere.** Kapitsa's auto store size here (28-29.9 GB) stopped four runs at memory pressure, so
the table pins 24 GB; a user's store is larger. The two engines run different quantized weights, so only prefill speed
is compared, and decode is withheld until the re-measure above.

<a id="run-it-yourself"></a>

## Run it yourself

**`kapitsa bench`** runs this protocol for Kapitsa alone, from the binary: no Python, no configs, the catalog's models
downloaded with their drafters. It flags runs as the harness does, and `--submit` adds the result to the
[community results](../bench/community/RESULTS.md); `kapitsa bench compare` runs it against every engine installed
([usage](usage.md#benchmarking-this-mac)).

**The harness** ([bench/harness](../bench/harness/README.md), Python 3, standard library only) reruns everything here:

1. Build Kapitsa and install the engines to compare; the harness README says how, with the settings used here.
2. Download the weights into `~/models` (the commands, with pinned revisions, are in the harness README).
3. `cd bench/harness && cp local.env.example local.env`, then set the paths.
4. Check the plan: `python3 compare_engines.py configs/*.json --dry-run`.
5. Run, one engine at a time: `python3 compare_engines.py configs/kapitsa.json configs/llama.cpp.json --only 'q36-35b-*'`.
6. Tabulate: `python3 compare_table.py`.

Numbers compare across engines when they run in one session, in the same power mode, plugged in, with browsers and
other GPU users closed.

## Archive

Earlier sessions, whose cells have since been re-measured or that measured something the tables don't show. Their
result files and READMEs stay in [bench/results](../bench/results/); each row links them.

| Date | Mac | Session | Results |
| --- | --- | --- | --- |
| 2026-10-16/17 | M4 Pro | <a id="summary"></a><a id="results"></a>The first comparison: Qwen3.6 35B-A3B and Qwen3.8 27B against llama.cpp, Ollama, LM Studio, mlx-lm, oMLX and MTPLX | [2026-10-17](../bench/results/2026-10-17/) |
| 2026-10-17 | M5 Max | <a id="macbook-pro-m5-max-128-gb"></a>Kapitsa alone, contention-flagged ([report](benchmarks-m5-max.md)) | [2026-10-17-m5-max](../bench/results/2026-10-17-m5-max/) |
| 2026-10-18 | M4 Pro | <a id="nemotron-35-bolt-74b-a9b-2026-10-18"></a>Nemotron 3.5 Bolt 74B-A9B | [2026-10-18](../bench/results/2026-10-18/) |
| 2026-10-18 | M4 Pro | <a id="qwen35-4b-2026-10-18"></a><a id="kapitsa-with-z-labs-dflash-drafter"></a><a id="full-precision-bf16-a-partial-bracket"></a>Qwen3.5-4B: every engine, z-lab's DFlash drafter, bf16 | [2026-10-18-4b](../bench/results/2026-10-18-4b/README.md) |
| 2026-10-18 | M4 Pro | <a id="mlx-serve-2026-10-18"></a>mlx-serve: MTP and DFlash2 settings | [2026-10-18-mlx-serve](../bench/results/2026-10-18-mlx-serve/README.md) |
| 2026-10-18 | M4 Pro | <a id="a-real-coding-agent-session-2026-10-18"></a>A real opencode session replayed: Kapitsa 48.6 s against mlx-serve's 61.6 s on the 35B-A3B, 150.2 s against oMLX's 278.1 s on the 27B | [2026-10-18-agent-session](../bench/results/2026-10-18-agent-session/README.md) |
| 2026-10-18 | M5 Max | <a id="qwen38-flash-next-180b-a7b-m5-max-128-gb-2026-10-18"></a>Qwen3.8-Flash-Next against every engine (the current Flash-Next cells) | [2026-10-18-m5-max-flash-next](../bench/results/2026-10-18-m5-max-flash-next/README.md) |
| 2026-10-18 | M5 Max | <a id="nemotron-35-super-vl-120b-a12b-m5-max-128-gb-2026-10-18"></a>Nemotron 3.5 Super-VL against every engine | [2026-10-18-m5-max](../bench/results/2026-10-18-m5-max/README.md) |
| 2026-10-18 | M5 Max | <a id="nemotron-35-lightning-30b-a3b-m5-max-128-gb-2026-10-18"></a>Nemotron 3.5 Lightning against every engine | [2026-10-18-m5-max-lightning](../bench/results/2026-10-18-m5-max-lightning/README.md) |
| 2026-10-18 | M5 Max | Qwen and Nemotron against every engine (the 10-18 M5 Max cells) | [2026-10-18-m5-max-session](../bench/results/2026-10-18-m5-max-session/README.md) |
| 2026-10-19 | M4 Pro | <a id="nemotron-dory10-4b-2026-10-19"></a><a id="nemotron-dory10-4b"></a>Nemotron Dory1.0 4B: loops, KV modes, NVIDIA's llama.cpp fork | [2026-10-19-dory](../bench/results/2026-10-19-dory/README.md) |
| 2026-10-19 | M4 Pro | <a id="the-m4-pro-table-filled-in-2026-10-19"></a>The M4 Pro table filled in | [2026-10-19-m4pro](../bench/results/2026-10-19-m4pro/README.md) |
| 2026-10-19 | M1 Pro | <a id="macbook-pro-m1-pro-16-gb-2026-10-19"></a>The 16 GB Mac against every engine | [2026-10-19-m1pro-16gb](../bench/results/2026-10-19-m1pro-16gb/README.md) |
| 2026-10-19 | M5 Pro | <a id="macbook-pro-m5-pro-48-gb-2026-10-19"></a>The M5 Pro against every engine | [2026-10-19-m5pro-48gb](../bench/results/2026-10-19-m5pro-48gb/README.md) |
| 2026-10-21 | M5 Pro | Prefill re-measured | [2026-10-21-m5pro-48gb](../bench/results/2026-10-21-m5pro-48gb/README.md) |
| 2026-10-22 | M5 Max | Kapitsa and Splash re-measured on main `fd85910` | [2026-10-22-m5-max](../bench/results/2026-10-22-m5-max/README.md) |
| 2026-10-22 | M5 Pro | The 16-row drafting block against mlx-serve's draft tree; the 27B's Neural Engine split on the matrix units | [2026-10-22-m5pro-48gb](../bench/results/2026-10-22-m5pro-48gb/README.md), [27b-ane-split](../bench/results/2026-10-22-m5pro-27b-ane-split/README.md) |
| 2026-10-22 | M5 Pro | Flint on Kapitsa before #345: main `6c13588`, then with #340 (expert GEMMs on the tensor units) | [flint-m5pro](../bench/results/2026-10-22-flint-m5pro/README.md), [flint-m5pro-qtm](../bench/results/2026-10-22-flint-m5pro-qtm/README.md) |
| 2026-10-22 | M3 Pro | Flint on battery (superseded by the AC run) | [2026-10-22-flint-m3pro](../bench/results/2026-10-22-flint-m3pro/README.md) |
| 2026-10-23 | M5 Pro | The 35B-A3B after #362 and the 27B after #360 | [2026-10-23-m5-pro-35b-27b](../bench/results/2026-10-23-m5-pro-35b-27b/README.md) |
