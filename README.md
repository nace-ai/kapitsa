<picture>
  <source media="(max-width: 600px)" srcset="docs/assets/header-narrow.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/header-dark.svg">
  <img alt="Kapitsa: a blazing fast Apple Silicon optimized inference engine built for Qwen, Nemotron and Flint. Writing speed in tokens per second for Qwen3.6 35B-A3B at 4-bit on a MacBook Pro (2026), M5 Max, 128 GB, the seven fastest of the nine engines measured: Kapitsa with KFlash 339, Splash with DFlash2 281, mlx-serve with MTP 235, oMLX with DFlash2 189, llama.cpp with MTP 171, Ollama with MTP 162, MLX with DFlash2 138." src="docs/assets/header-light.svg">
</picture>

Kapitsa is an inference engine for Apple silicon that runs Qwen, Nemotron and Recursive's Flint models locally, built for coding agents. It starts opencode, Claude Code, Codex or pi on a local model, and serves OpenAI and Anthropic APIs.

[Benchmarks](#benchmarks) · [Quickstart](#quickstart) · [CLI](#cli) · [Models](#models) · [How KFlash works](docs/kflash.md) · [Docs](docs/usage.md)

## Benchmarks

Kapitsa has the fastest decode in 32 of the 37 model and Mac pairs measured against nine engines ([how each number was measured](docs/benchmarks.md#how-the-readme-poster-was-measured)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/benchmarks-dark.webp">
  <img alt="Poster of prefill / decode speed for Kapitsa and nine other engines on a MacBook Pro M5 Max, M5 Pro, M4 Pro, M3 Pro and M1 Pro; the numbers are in the poster and in the Models section's tables" src="docs/assets/benchmarks-light.webp">
</picture>

<sub>**Each cell reads prefill / decode in tokens per second.** Prefill is a 4K-token prompt, decode a 512-token answer at T 0.6 (Flint at its card's T 1.0). × cannot run that model · — not measured. How these were measured, run dates and flags: [docs/benchmarks.md](docs/benchmarks.md#how-the-readme-poster-was-measured).</sub>

The poster's numbers, and the models it leaves out, are in each Mac's table under [Models](#models). Every recorded run, with its Mac, model, prefill and decode speed, is in the [benchmark table](bench/README.md). Measure your Mac: `kapitsa bench --submit` adds it to the [community results](bench/community/RESULTS.md). Another engine faster on your Mac? `kapitsa bench compare` ranks every engine you have installed, and `--report` writes a Markdown report.

## 🔥 What's new

- [2026/10] 🪶 **Flint on a 16 GB Mac: 61 tok/s against llama.cpp's 43 on the same 2-bit GGUF** ([numbers](docs/benchmarks.md#recursive-flint-40b-a25b-with-2-bit-experts-m1-pro-16-gb-2026-10-22)).
- [2026/10] 🪨 **Recursive's Flint 40B-A2.5B: decodes faster than llama.cpp, reads prompts faster than rec 0.1.0** ([M4 Pro](docs/benchmarks.md#recursive-flint-40b-a25b-m4-pro-2026-10-22), [M5 Pro](docs/benchmarks.md#recursive-flint-40b-a25b-m5-pro-2026-10-22)).
- [2026/10] 🚀 **KFlash: tree speculative decoding, up to 3.5× llama.cpp and 2.2× oMLX** ([how it works](docs/kflash.md)).
- [2026/10] 🏆 **The fastest decode in 32 of 37 model and Mac pairs** against nine engines, up to 2.9× MLX ([poster](#benchmarks)).
- [2026/10] 🤝 **KFlash budget: three requests share one verify pass**, up to +23% total throughput on the 27B.
- [2026/10] ⚡ **QMT prefill: prompts read up to 1.5× faster on M5** with the Metal 4 matrix units.
- [2026/10] 💾 **SSD prefix cache: a 16K follow-up after a restart starts in 0.16 s instead of 17 s** (`--disk-cache`).
- [2026/10] 🧠 **Prefix reuse: a new question over a shared 4K system prompt answers in 0.06 s instead of 1.9 s.**
- [2026/10] 🧪 **Output check: a faster default must keep the answers** ([how](CONTRIBUTING.md#output-check)).
- [2026/10] 📊 **`kapitsa bench compare`: every installed engine on your Mac, one command.**

## Quickstart

**Requirements:** a Mac with Apple silicon (M1 or newer) and macOS 14 Sonoma or later.

```bash
brew install nace-ai/tap/kapitsa
# or: curl -fsSL https://raw.githubusercontent.com/nace-ai/kapitsa/main/install.sh | bash
```

```bash
kapitsa opencode --model 35b    # also: claude, codex, pi
```

<sub>On a 16-24 GB Mac use `--model 9b`, or `4b` for the quickest replies.</sub>

## CLI

```bash
kapitsa serve                   # chat + OpenAI/Anthropic APIs at http://127.0.0.1:8000
kapitsa claude                  # Claude Code on Kapitsa
kapitsa doctor                  # this Mac's GPU and memory, and which models fit it
kapitsa cluster                 # the Macs on this network that run Kapitsa, and their models
kapitsa bench                   # benchmark this Mac, share it with the community
kapitsa bench compare           # the same test against the other engines installed here
kapitsa pull 27b                # download a model without starting it
kapitsa list                    # downloaded models and their size
kapitsa rm 27b                  # delete a downloaded model
kapitsa import model.gguf       # use a Qwen or Flint GGUF from LM Studio, Ollama or llama.cpp
kapitsa update                  # update now (it also updates itself)
```

<sub>`--model` picks the model and its precision (`35b:6bit`). `kapitsa <command> --help` has the options.
A Mac that cannot hold a model whole uses another Mac on the network that runs it, and two Macs that would each stream
Flash-Next pair to hold it in memory between them ([Several Macs](docs/usage.md#several-macs)); `--private` keeps a
server to its own Mac. [Usage guide](docs/usage.md)</sub>

## Models

The catalog models measured on each Mac, against the engines that run them: one collapsed section per Mac, with a table for each model family. Decode is a 512-token coding answer at T 0.6 (Flint at its card's T 1.0), prefill a 4K-token prompt. Under each model's name are the weights Kapitsa ran; the other engines' weights and how each number was measured are in [docs/benchmarks.md](docs/benchmarks.md#the-readmes-tables). Flint's speed from 1K to 128K tokens of prompt on the M5 Pro, against llama.cpp: [bench/results/2026-10-24-flint-m5pro-context](bench/results/2026-10-24-flint-m5pro-context/README.md).

<sub>Each cell: prefill / decode, tokens/s (a lone number is decode: no prefill run) · 🟩 fastest decode · ≈ a decode tie with Kapitsa · small type: the decoding method (none: no speculative decoding) · × can't run on that Mac · — not measured</sub>

<details>
<summary><b>MacBook Pro (16-inch, 2026) · M5 Max · 128 GB</b> · 8 models</summary>

**Qwen and Nemotron**

| Model | <mark>**Kapitsa**</mark> | <sub>mlx‑serve<br>26.10.2 / 26.10.3</sub> | <sub>oMLX<br>0.7.1rc1 / 0.7.3.dev2</sub> | <sub>Splash<br>1.4.0</sub> | <sub>Ollama<br>0.40.5 / 0.40.11</sub> | <sub>MTPLX<br>2.12.2</sub> | <sub>MLX<br>mlx-vlm 0.7.8 / 0.7.10</sub> | <sub>llama.cpp<br>b11669 / b11871</sub> | <sub>LM&nbsp;Studio<br>MLX runtime 1.13.1 (nax), llama.cpp runtime 2.60.0</sub> |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Qwen3.8-27B** · `27b`<br><sub>dense · 4-bit, 20.2 GB</sub> | 🟩 907 / **133**<br><sub>KFlash · cd20dcc</sub> | 792 / **≈128**<br><sub>DFlash2 · 26.10.3</sub> | 902 / 92<br><sub>DFlash2 · 0.7.3.dev2</sub> | 1039 / 116<br><sub>DFlash2</sub> | 844 / 71<br><sub>MTP · 0.40.11, MLX runner</sub> | 848 / 73<br><sub>MTP</sub> | 856 / 56<br><sub>MTP · mlx-vlm 0.7.10</sub> | 578 / 48<br><sub>MTP · b11871</sub> | 918 / 33<br><sub>MLX runtime 1.13.1 (nax)</sub> |
| **Qwen3.6-35B-A3B** · `35b`<br><sub>MoE, 3B active · 4-bit, 22.1 GB</sub> | 🟩 5963 / **339**<br><sub>KFlash · cd20dcc</sub> | 4622 / 235<br><sub>MTP · 26.10.3</sub> | 3005 / 189<br><sub>DFlash2 · 0.7.3.dev2</sub> | 6386 / 281<br><sub>DFlash2</sub> | 3077 / 162<br><sub>MTP · 0.40.11, MLX runner</sub> | 4291 / 130<br><sub>MTP</sub> | 4439 / 138<br><sub>DFlash2 · mlx-vlm 0.7.10</sub> | 2881 / 171<br><sub>MTP · b11669</sub> | 4672 / 136<br><sub>MLX runtime 1.13.1 (nax)</sub> |
| **Qwen3.8-Flash-Next 180B-A7B** · `flash-next`<br><sub>MoE, 7B active · 4-bit, 105.9 GB</sub> | 1165 / 106<br><sub>KFlash · fd85910</sub> | 🟩 1720 / **108**<br><sub>26.10.2</sub> | 1260 / 107<br><sub>MTP · 0.7.1rc1</sub> | × | 60<br><sub>0.40.6</sub> | 1263 / 88<br><sub>MTP</sub> | 1030 / 77<br><sub>MTP · mlx-vlm 0.7.8</sub> | 842 / 96<br><sub>b11669</sub> | × |
| **Nemotron Dory1.0 4B** · `dory`<br><sub>looped dense · NVFP4, 3.2 GB</sub> | 🟩 **119**<br><sub>KFlash · 039ddea</sub> | × | × | × | × | × | 76<br><sub>mlx-vlm 0.7.8</sub> | 9<br><sub>b11669</sub> | × |
| **Nemotron 3.5 Lightning 30B-A3B** · `lightning`<br><sub>hybrid MoE, 3B active · NVFP4, 22.9 GB</sub> | 🟩 **267**<br><sub>KFlash · fd85910</sub> | 190<br><sub>26.10.2</sub> | 190<br><sub>0.7.1rc1</sub> | × | 153<br><sub>0.40.5</sub> | 131 | 139<br><sub>mlx-vlm 0.7.8</sub> | 159<br><sub>b11669</sub> | 110<br><sub>llama.cpp runtime 2.60.0</sub> |
| **Nemotron 3.5 Bolt 74B-A9B** · `bolt`<br><sub>hybrid MoE, 9B active · 2/4/8-bit mix, 25.1 GB</sub> | 🟩 1038 / **160**<br><sub>KFlash · cd20dcc</sub> | × | 903 / 71<br><sub>0.7.3.dev2</sub> | × | × | × | 936 / 66<br><sub>mlx-vlm 0.7.10</sub> | × | × |
| **Nemotron 3.5 Super-VL 120B-A12B** · `super`<br><sub>hybrid MoE, 12B active · NVFP4, 76.4 GB</sub> | 🟩 **94**<br><sub>KFlash · fd85910</sub> | × | 52<br><sub>0.7.1rc1</sub> | × | 64<br><sub>0.40.5</sub> | 44 | 46<br><sub>mlx-vlm 0.7.8</sub> | 66<br><sub>b11669</sub> | 61<br><sub>llama.cpp runtime 2.60.0</sub> |

**Recursive Flint**

| Model | <mark>**Kapitsa**</mark> | <sub>llama.cpp<br>Q4_1, b11871</sub> | <sub>rec<br>0.1.0</sub> |
| --- | ---: | ---: | ---: |
| **Recursive Flint 40B-A2.5B** · `flint`<br><sub>MoE, 2.5B active · 4-bit, 22.6 GB</sub> | 🟩 5721 / **267**<br><sub>KFlash</sub> | 3827 / 130 | 2863 / 224 |

<sub>The 27B, 35B-A3B and Bolt rows are the 2026-10-25 session (Kapitsa main cd20dcc and each engine's latest release, macOS 26.4); llama.cpp's 35B-A3B cell, the Flash-Next row and the Lightning, Super-VL and Dory rows are the 2026-10-18 to 10-22 runs, not yet re-measured. The Flint row is one block on 2026-10-26 (Kapitsa 0.2.18, [results](bench/results/2026-10-26-m5-max/README.md)). The 27B and Bolt rows are each one block of the row on AC power in High Power mode. The build each cell ran is under its number; details in [bench/results/2026-10-25-m5-max](bench/results/2026-10-25-m5-max/README.md) and [its re-runs](bench/results/2026-10-25-m5-max/2026-10-26-reruns.md).</sub>

</details>

<details>
<summary><b>MacBook Pro (16-inch, 2026) · M5 Pro · 48 GB</b> · 6 models</summary>

**Qwen and Nemotron**

| Model | <mark>**Kapitsa**</mark> | <sub>mlx‑serve<br>26.10.2</sub> | <sub>oMLX<br>0.7.1rc1</sub> | <sub>Splash<br>1.4.0</sub> | <sub>Ollama<br>0.40.5 / 0.40.6</sub> | <sub>MTPLX<br>2.12.2</sub> | <sub>MLX<br>mlx-lm 0.32.2, mlx-vlm 0.7.8</sub> | <sub>llama.cpp<br>b11669 / b11871</sub> | <sub>LM&nbsp;Studio<br>MLX runtime 1.13.1, llama.cpp runtime 2.60.0</sub> |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Qwen3.8-27B** · `27b`<br><sub>dense · 4-bit, 20.2 GB</sub> | 606 / 77<br><sub>KFlash</sub> | 🟩 464 / **78**<br><sub>DFlash2</sub> | 446 / 51<br><sub>DFlash2</sub> | 674 / 72<br><sub>DFlash2</sub> | 470 / 40<br><sub>MTP</sub> | 462 / 41<br><sub>MTP</sub> | 421 / 32<br><sub>MTP</sub> | 322 / 30<br><sub>MTP</sub> | 118 / 17 |
| **Qwen3.6-35B-A3B** · `35b`<br><sub>MoE, 3B active · 4-bit, 22.1 GB</sub> | 🟩 2802 / **201**<br><sub>KFlash</sub> | 2494 / 156<br><sub>MTP</sub> | 1619 / 144<br><sub>DFlash2</sub> | 3514 / 181<br><sub>DFlash2</sub> | 1518 / 102<br><sub>MTP</sub> | 2569 / 89<br><sub>MTP</sub> | 2474 / 98<br><sub>DFlash2</sub> | 1755 / 107<br><sub>MTP</sub> | 997 / 96 |
| **Nemotron Dory1.0 4B** · `dory`<br><sub>looped dense · NVFP4, 3.2 GB</sub> | 🟩 **70**<br><sub>KFlash</sub> | × | × | × | × | × | 53 | 13 | × |
| **Nemotron 3.5 Lightning 30B-A3B** · `lightning`<br><sub>hybrid MoE, 3B active · NVFP4, 22.9 GB</sub> | 🟩 3141 / **159**<br><sub>KFlash</sub> | 1728 / 115<br><sub>MTP</sub> | 1938 / 124<br><sub>MTP</sub> | × | 2369 / 92 | 1743 / 101 | 1881 / 94 | 1915 / 101<br><sub>MTP</sub> | 1479 / 72 |
| **Nemotron 3.5 Bolt 74B-A9B** · `bolt`<br><sub>hybrid MoE, 9B active · 2/4/8-bit mix, 25.1 GB</sub> | 🟩 **79**<br><sub>KFlash</sub> | × | 39 | × | × | × | 41 | × | × |

**Recursive Flint**

| Model | <mark>**Kapitsa**</mark> | <sub>llama.cpp<br>b11804</sub> | <sub>rec<br>0.1.0</sub> |
| --- | ---: | ---: | ---: |
| **Recursive Flint 40B-A2.5B** · `flint`<br><sub>MoE, 2.5B active · 4-bit, 22.6 GB</sub> | 🟩 2691 / **149**<br><sub>KFlash</sub> | 2072 / 92 | 2138 / 145 |

<sub>Qwen3.8-Flash-Next 180B-A7B does not fit in 48 GB, so both engines stream its experts from the SSD. With Q2_0 weights on both and Kapitsa's expert store pinned at 24 GB (the auto size here is about 29.9 GB, but four runs at that size were stopped at memory pressure), Kapitsa's prefill is behind on short prompts, level at 4K and 16K and 2.7% ahead at 32K: not the fastest everywhere ([numbers and caveats](docs/benchmarks.md#qwen38-flash-next-streamed-from-the-ssd-kapitsa-against-llamacpp-m5-pro-48-gb-2026-10-24)).</sub>

</details>

<details>
<summary><b>MacBook Pro (16-inch, 2024) · M4 Pro · 48 GB</b> · 7 models</summary>

**Qwen and Nemotron**

| Model | <mark>**Kapitsa**</mark> | <sub>mlx‑serve<br>26.10.2</sub> | <sub>oMLX<br>0.7.1rc1</sub> | <sub>Splash<br>1.4.0</sub> | <sub>Ollama<br>0.40.5 / 0.40.6</sub> | <sub>MTPLX<br>2.12.2</sub> | <sub>MLX<br>mlx-lm 0.32.2, mlx-vlm 0.7.8</sub> | <sub>llama.cpp<br>b11669 / b11871</sub> | <sub>LM&nbsp;Studio<br>MLX runtime 1.13.1, llama.cpp runtime 2.60.0</sub> |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Qwen3.5-4B** · `4b`<br><sub>dense · 4-bit, 4.6 GB</sub> | 🟩 1930 / **201**<br><sub>KFlash</sub> | 874 / 137<br><sub>MTP</sub> | 803 / 102<br><sub>MTP</sub> | × | 758 / 109<br><sub>MTP</sub> | 780 / 104<br><sub>MTP</sub> | 843 / 88<br><sub>MTP</sub> | 650 / 70<br><sub>MTP</sub> | 801 / 81 |
| **Qwen3.8-27B** · `27b`<br><sub>dense · 4-bit, 20.2 GB</sub> | 🟩 396 / **63**<br><sub>KFlash</sub> | 129 / 54<br><sub>DFlash2</sub> | 136 / 45<br><sub>DFlash2</sub> | 53<br><sub>DFlash2</sub> | 124 / 32<br><sub>MTP</sub> | 126 / 36<br><sub>MTP</sub> | 131 / 28<br><sub>MTP</sub> | 84 / 18<br><sub>MTP</sub> | 127 / 15 |
| **Qwen3.6-35B-A3B** · `35b`<br><sub>MoE, 3B active · 4-bit, 22.1 GB</sub> | 🟩 1168 / **161**<br><sub>KFlash</sub> | 986 / 124<br><sub>MTP</sub> | 1035 / 127<br><sub>DFlash2</sub> | 135<br><sub>DFlash2</sub> | 892 / 89<br><sub>MTP</sub> | 918 / 80<br><sub>MTP</sub> | 985 / 86<br><sub>DFlash2</sub> | 663 / 69<br><sub>MTP</sub> | 957 / 86 |
| **Nemotron Dory1.0 4B** · `dory`<br><sub>looped dense · NVFP4, 3.2 GB</sub> | 🟩 **64**<br><sub>KFlash</sub> | × | × | × | × | × | 48 | 6 | × |
| **Nemotron 3.5 Lightning 30B-A3B** · `lightning`<br><sub>hybrid MoE, 3B active · NVFP4, 22.9 GB</sub> | 🟩 945 / **112**<br><sub>KFlash</sub> | — | — | × | — | — | 721 / 65<br><sub>DSpark</sub> | — | — |
| **Nemotron 3.5 Bolt 74B-A9B** · `bolt`<br><sub>hybrid MoE, 9B active · 2/4/8-bit mix, 25.1 GB</sub> | 🟩 519 / **69**<br><sub>KFlash</sub> | — | 298 / 35 | × | — | — | 300 / 32 | × | — |

**Recursive Flint**

| Model | <mark>**Kapitsa**</mark> | <sub>llama.cpp<br>b11804</sub> | <sub>rec<br>0.1.0</sub> |
| --- | ---: | ---: | ---: |
| **Recursive Flint 40B-A2.5B** · `flint`<br><sub>MoE, 2.5B active · 4-bit, 22.6 GB</sub> | 🟩 968 / **128**<br><sub>KFlash</sub> | 977 / 80 | 796 / 127 |

</details>

<details>
<summary><b>MacBook Pro (16-inch, 2023) · M3 Pro · 36 GB</b> · 10 models</summary>

**Qwen and Nemotron**

| Model | <mark>**Kapitsa**</mark> | <sub>Ollama<br>0.40.8</sub> | <sub>MTPLX<br>2.12.2</sub> | <sub>MLX<br>mlx-lm 0.32.2, mlx-vlm 0.7.8</sub> | <sub>llama.cpp<br>b11669 / b11871</sub> |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Qwen3.5-4B** · `4b`<br><sub>dense · 4-bit, 4.6 GB</sub> | 🟩 703 / **120**<br><sub>KFlash</sub> | 600 / 67<br><sub>MTP</sub> | 612 / 64<br><sub>MTP</sub> | 642 / 56<br><sub>MTP</sub> | 512 / 50<br><sub>MTP · b11804</sub> |
| **Qwen3.5-9B** · `9b`<br><sub>dense · 4-bit, 9.0 GB</sub> | 🟩 381 / **77**<br><sub>KFlash</sub> | — | — | 353 / 27 | 307 / 23<br><sub>b11804</sub> |
| **Qwen3.8-27B** · `27b`<br><sub>dense · 4-bit, 20.2 GB</sub> | 🟩 110 / **40**<br><sub>KFlash</sub> | 99 / 23<br><sub>MTP</sub> | 99 / 22<br><sub>MTP</sub> | 102 / 18<br><sub>MTP</sub> | 86 / 16<br><sub>MTP · b11804</sub> |
| **Qwen3.6-35B-A3B** · `35b`<br><sub>MoE, 3B active · 4-bit, 22.1 GB</sub> | 🟩 855 / **106**<br><sub>KFlash</sub> | 681 / 58<br><sub>MTP</sub> | 671 / 53<br><sub>MTP</sub> | 695 / 53<br><sub>DFlash2</sub> | 578 / 53<br><sub>MTP · b11804</sub> |
| **Nemotron Dory1.0 4B** · `dory`<br><sub>looped dense · NVFP4, 3.2 GB</sub> | 🟩 450 / **36**<br><sub>KFlash</sub> | × | × | 344 / 30 | — |
| **Nemotron 3 Nano 4B** · `nano-4b`<br><sub>hybrid Mamba-2 · FP8, 5.3 GB</sub> | 765 / 34<br><sub>KFlash</sub> | — | — | — | 🟩 413 / **35**<br><sub>Q4_K_M, b11804</sub> |
| **Nemotron Nano 9B v2** · `nano-9b`<br><sub>hybrid Mamba-2 · NVFP4, 7.9 GB</sub> | 330 / 21<br><sub>KFlash</sub> | — | — | 🟩 **25**<br><sub>4-bit</sub> | — |
| **Nemotron 3.5 Lightning 30B-A3B** · `lightning`<br><sub>hybrid MoE, 3B active · NVFP4, 22.9 GB</sub> | 🟩 788 / **75**<br><sub>KFlash</sub> | — | — | 514 / 39<br><sub>DSpark</sub> | — |
| **Nemotron 3.5 Bolt 74B-A9B** · `bolt`<br><sub>hybrid MoE, 9B active · 2/4/8-bit mix, 25.1 GB</sub> | 🟩 256 / **50**<br><sub>KFlash</sub> | — | — | — | — |

**Recursive Flint**

| Model | <mark>**Kapitsa**</mark> | <sub>llama.cpp<br>b11804</sub> |
| --- | ---: | ---: |
| **Recursive Flint 40B-A2.5B** · `flint`<br><sub>MoE, 2.5B active · 4-bit, 22.6 GB</sub> | 🟩 801 / **74**<br><sub>KFlash</sub> | 772 / 51 |

<sub>macOS 14.1, on AC power: Kapitsa's macOS 14 kernels (macOS 15 and later run fused ones). mlx-serve, oMLX and Splash need a newer macOS (×); LM Studio was not installed; rec 0.1.0 refuses this Mac: it requires 48 GiB. Each cell but Flint's is one run of the M4 Pro's protocol (Flint's is the mean of two runs, #341), the other engines in the M4 Pro's setup (not re-tuned here). Nano 4B and 9B v2: Kapitsa runs NVIDIA's published checkpoints, which keep some layers in BF16/FP8, so it reads more per token than the 4-bit files: the 4B's FP8 is ~1 byte per weight against Q4_K_M's ~0.6, the 9B's NVFP4 ~6.11 GB per token against 4.67 GB for mlx-lm's 4-bit. Nano 9B v2 on mlx-lm: no prefill shown, because its template ignores enable_thinking and the timed first token is a thinking token.</sub>

</details>

<details>
<summary><b>MacBook Pro (14-inch, 2021) · M1 Pro · 16 GB</b> · 7 models</summary>

**Qwen and Nemotron**

| Model | <mark>**Kapitsa**</mark> | <sub>mlx‑serve<br>26.10.2</sub> | <sub>oMLX<br>0.7.1rc1</sub> | <sub>Ollama<br>0.40.5 / 0.40.6</sub> | <sub>MTPLX<br>2.12.2</sub> | <sub>MLX<br>mlx-lm 0.32.2, mlx-vlm 0.7.8</sub> | <sub>llama.cpp<br>b11669 / b11871</sub> | <sub>LM&nbsp;Studio<br>MLX runtime 1.13.1, llama.cpp runtime 2.60.0</sub> |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Qwen3.5-4B** · `4b`<br><sub>dense · 4-bit, 4.6 GB</sub> | 🟩 401 / **97**<br><sub>KFlash</sub> | 286 / 62<br><sub>MTP</sub> | 281 / 55<br><sub>MTP</sub> | 298 / 55<br><sub>MTP</sub> | 305 / 42<br><sub>MTP</sub> | 308 / 46 | 315 / 38<br><sub>MTP</sub> | 308 / 47 |
| **Qwen3.5-9B** · `9b`<br><sub>dense · 4-bit, 9.0 GB</sub> | 🟩 217 / **57**<br><sub>KFlash</sub> | 159 / 37<br><sub>MTP</sub> | — | — | — | 170 / 30 | — | — |
| **Qwen3.6-35B-A3B** · `35b:16gb`<br><sub>MoE, 3B active · 2-bit experts (unsloth's GGUF), 10.8 GB</sub> | 🟩 462 / **52**<br><sub>KFlash</sub> | × | × | — | × | × | 38 | — |
| **Nemotron Dory1.0 4B** · `dory`<br><sub>looped dense · NVFP4, 3.2 GB</sub> | 🟩 **39**<br><sub>KFlash</sub> | × | × | × | × | 30 | 4 | × |
| **Nemotron 3 Nano 4B** · `nano-4b`<br><sub>hybrid Mamba-2 · FP8, 5.3 GB</sub> | 🟩 **37**<br><sub>KFlash</sub> | — | — | — | — | 21<br><sub>BF16</sub> | 28<br><sub>Q4_K_M</sub> | — |
| **Nemotron Nano 9B v2** · `nano-9b`<br><sub>hybrid Mamba-2 · NVFP4, 7.9 GB</sub> | 🟩 **25**<br><sub>KFlash</sub> | — | — | — | — | 18 | 14<br><sub>Q4_K_M</sub> | — |

**Recursive Flint**

| Model | <mark>**Kapitsa**</mark> | <sub>llama.cpp<br>d772569</sub> |
| --- | ---: | ---: |
| **Recursive Flint 40B-A2.5B** · `flint:16gb`<br><sub>MoE, 2.5B active · 2-bit experts (unsloth's GGUF), 11.9 GB</sub> | 🟩 414 / **61**<br><sub>KFlash</sub> | 43 |

</details>

<sub>Other precisions: `--model 35b:6bit` is five times closer to bf16 than the 4-bit default ([KL](docs/benchmarks.md#quality-qwen36-35b-a3b-against-bf16-2026-10-19)). All but Lightning, Dory, the two Nemotron Nanos, Flint and the 35B's `16gb` read images. Memory: 16 GB for the 4B and 9B models, 32 GB
for the 27B, 35B, Lightning, Bolt and Flint, 96 GB for Super-VL, 128 GB for Flash-Next. On a 16 or 24 GB Mac, `35b` and `flint` take their `16gb` variants: unsloth's 2-bit GGUFs, downloaded and imported in one step ([how](docs/usage.md#gguf-files)). [Every model's variants and draft models](docs/usage.md#models-and-downloads)</sub>

## How it works

<picture>
  <source media="(max-width: 600px)" srcset="docs/assets/how-it-works-narrow.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/how-it-works-dark.svg">
  <img alt="How Kapitsa works: DFlash2 vs KFlash. DFlash2, the drafter's chain, as Splash, oMLX and mlx-serve run it: the drafter guesses the next 7 tokens, the big model checks them in one pass and keeps them up to the first wrong guess (return a, 2 tokens). KFlash, Kapitsa's: in the same one pass the big model checks a best-first tree of the drafter's top candidates in 8 rows; where the drafter is unsure its runner-ups ride along, and the longest branch the big model agrees with is kept (return a + b, 4 tokens). Same output as plain decoding, tree size measured on each Mac, 8-17% more tokens per pass on short prompts. Below: GPU, hand-written Metal code; Neural Engine, helps read long prompts on the 4B, 27B, Bolt and Super-VL; Memory, keeps conversations ready, up to 4 at once." src="docs/assets/how-it-works-light.svg">
</picture>

Kapitsa decodes with **KFlash**. A draft model such as DFlash2 guesses the next seven tokens, with a few candidates for each one. Engines that run it as published (Splash, oMLX, mlx-serve) check one chain of guesses and stop at the first wrong one. KFlash checks a best-first tree of the drafter's candidates in the same pass, so where the drafter hesitates, its runner-ups are checked too and more tokens survive each pass. The tree's size comes from what a verify row costs on your Mac, measured there, and it changes no output: the big model checks every token. It runs on the published drafters as they are: z-lab's DFlash, DFlash2, and NVIDIA's DSpark and LiLiCorr. [How KFlash works](docs/kflash.md)

## Get release notices

<p align="center"><img src="docs/assets/watch-releases.gif" alt="On the Kapitsa repository page: Watch, then Custom, tick Releases, Apply" width="720"></p>

<sub>Watch → Custom → Releases on [nace-ai/kapitsa](https://github.com/nace-ai/kapitsa) puts every new version in your notifications. A drawn illustration of the steps, not a screen recording; the page source is in [tools/readme_images](tools/readme_images/README.md).</sub>

## Star history

<a href="https://star-history.com/#nace-ai/kapitsa&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=nace-ai/kapitsa&type=Date&theme=dark">
    <img alt="Star history of nace-ai/kapitsa on GitHub" src="https://api.star-history.com/svg?repos=nace-ai/kapitsa&type=Date">
  </picture>
</a>

<sub>[How KFlash works](docs/kflash.md) · [Architecture](docs/architecture.md) · [Contributing](CONTRIBUTING.md) · [Press kit](docs/assets/brand/README.md) ·
Apache 2.0, © 2026 [Nace AI](https://nace.ai) · [Third-party notices](THIRD_PARTY_NOTICES)</sub>
