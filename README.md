<picture>
  <source media="(max-width: 600px)" srcset="docs/assets/header-narrow.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/header-dark.svg">
  <img alt="Kapitsa: the fastest Apple silicon inference engine built for Qwen and Nemotron. Writing speed of Qwen3.6 35B-A3B on a MacBook Pro (2024), M4 Pro, 48 GB, in tokens per second, at 4-bit: Kapitsa 154, oMLX with DFlash2 127, Ollama 89, LM Studio 86, MTPLX 80, llama.cpp with MTP 69." src="docs/assets/header-light.svg">
</picture>

[Install](#install) · [Quick start](#quick-start) · [Models](#models) · [Benchmarks](#benchmarks) · [Connect your apps](#connect-your-apps) · [Architecture](#architecture)

## Install

```bash
brew install nace-ai/tap/kapitsa
```

<details>
<summary>Without Homebrew, or from source</summary>

The same prebuilt binary, checked against the release's checksum:

```bash
curl -fsSL https://raw.githubusercontent.com/nace-ai/kapitsa/main/install.sh | bash
```

From source, with an Apple silicon Mac on macOS 15 or later, the Xcode Command Line Tools and CMake 3.24+:

```bash
git clone https://github.com/nace-ai/kapitsa.git && cd kapitsa && ./install.sh
```

</details>

Kapitsa keeps itself up to date: installed with `install.sh`, a new release installs itself within a day, and with
Homebrew a run in a terminal says when `brew upgrade` has one. `kapitsa update` updates now
([details](docs/usage.md#updating)). Kapitsa 0.1.32 and older don't do this yet: update those once with the installer
or `brew upgrade nace-ai/tap/kapitsa`.

## Quick start

```bash
kapitsa opencode --model 35b
```

This starts Qwen3.6 35B-A3B (a 22.1 GB download the first time) and opens [opencode](https://opencode.ai) on it.
`kapitsa claude` and `kapitsa codex` work the same way. For the chat page and the API at **http://127.0.0.1:8000**,
run `kapitsa serve --model 35b`. The 35B needs a Mac with 32 GB or more; on a 16-24 GB Mac use `--model 4b`.
`kapitsa doctor` shows which models and context sizes fit your Mac.

<!-- Demo slot: record docs/assets/demo.gif from main once the terminal UI (opt/tui) lands, then uncomment.
<p align="center"><img src="docs/assets/demo.gif" alt="kapitsa claude: Claude Code on a local model" width="830"></p>
-->

## Models

| Model | Download | Speculation |
| --- | --- | --- |
| **Qwen3.5-4B** · `4b`<br><sub>dense</sub> | **4-bit · 4.6 GB**<br><sub>also 6-bit 5.6 · 8-bit 6.7 · bf16 10.6 GB</sub> | DFlash / MTP |
| **Qwen3.8-27B** · `27b`<br><sub>dense</sub> | **4-bit · 20.2 GB**<br><sub>also 5-bit 23.5 · 6-bit 26.9 · 8-bit 33.6 · bf16 58.8 GB</sub> | DFlash2 / MTP |
| **Qwen3.6-35B-A3B** · `35b`<br><sub>MoE, 3B active</sub> | **4-bit · 22.1 GB**<br><sub>also 5-bit 26.4 · 6-bit 30.7 · 8-bit 39.4 · bf16 71.9 GB</sub> | DFlash2 / MTP |
| **Nemotron 3.5 Lightning 30B-A3B** · `lightning`<br><sub>hybrid Mamba-2 MoE, 3B active; text only</sub> | **NVIDIA NVFP4 · 22.9 GB**<br><sub>4-bit NVFP4 experts, FP8, BF16</sub> | DSpark |
| **Nemotron 3.5 Bolt 74B-A9B** · `bolt`<br><sub>hybrid Mamba-2 MoE, 9B active</sub> | **NVIDIA 24 GB · 25.1 GB**<br><sub>2-bit experts, NVFP4, 8-bit</sub> | LiLiCorr |
| **Nemotron 3.5 Super-VL 120B-A12B** · `super`<br><sub>hybrid Mamba-2 MoE, 12B active; 96 GB+ Macs</sub> | **NVIDIA NVFP4 · 76.4 GB**<br><sub>4-bit NVFP4 experts, FP8, BF16</sub> | LiLiCorr |

bf16 is full precision; pick another precision with `--model 35b:6bit`, `--model 27b:bf16` and so on.

Downloads include the model's companion files ([precisions](docs/usage.md#precisions)). Bolt is NVIDIA's own `24gb` checkpoint, Super-VL and Lightning NVIDIA's own `nvfp4` ones. Every model above
except Lightning reads [images](docs/usage.md#images). Other Qwen3.5-family models load from a Hugging Face repo, a
folder or a GGUF file ([other models](docs/usage.md#other-models)). Model weights keep their own licenses.

```bash
kapitsa serve --model bolt     # start another model (stop the current server first)
kapitsa pull 27b               # download without starting a server
kapitsa list                   # show downloaded models
kapitsa rm 27b                 # delete one
```

## Benchmarks

Writing speed: tokens per second of a coding answer (decode, temperature 0.6), each engine in the setup that writes
fastest ([all setups](docs/benchmarks.md)). Qwen runs at **4-bit** on every engine: MLX 4-bit (group 64) on Kapitsa
and the MLX engines, Q4_K_M on llama.cpp. Nemotron 3.5 Bolt runs NVIDIA's own 24 GB checkpoint (2-bit experts,
4-bit NVFP4 Mamba layers, 8-bit elsewhere) on every engine. ⚠ marks a number from a run during which other apps used
the GPU (contention flagged): it may be low.

**On your Mac:** `kapitsa bench` measures Kapitsa with the same protocol, and the
[community results](bench/community/RESULTS.md) collect every Mac measured so far: add yours with
`kapitsa bench --submit`.

### MacBook Pro (16-inch, 2024) · M4 Pro · 48 GB

| Engine | Qwen3.5 4B · 4-bit | Qwen3.8 27B · 4-bit | Qwen3.6 35B-A3B · 4-bit | Nemotron 3.5 Bolt · NVIDIA 24 GB |
| --- | ---: | ---: | ---: | ---: |
| llama.cpp + MTP | 70 tok/s | 18 tok/s | 69 tok/s | — |
| Ollama (MLX runner, MTP) | 109 tok/s | 32 tok/s | 89 tok/s | — |
| LM Studio (MLX engine) | 81 tok/s | 15 tok/s | 86 tok/s | — |
| MTPLX | 104 tok/s | 36 tok/s | 80 tok/s | — |
| mlx-vlm | — | — | — | 32 tok/s |
| oMLX | 102 tok/s | 45 tok/s | 127 tok/s | 34 tok/s ⚠ |
| mlx-serve (MTP) | — | 38 tok/s ⚠ | 124 tok/s | — |
| mlx-serve (DFlash2) | — | 52 tok/s ⚠ | 104 tok/s | — |
| **Kapitsa** | **179 tok/s**<sup>4B</sup> | **63 tok/s** | **154 tok/s** | **64 tok/s** |

oMLX runs DFlash2 on the 27B and 35B, its own MTP path on the 4B (its fastest setup there) and Bolt without a drafter:
its LiLiCorr drafter fails every sampled request. Bolt is NVIDIA's 24 GB checkpoint on every engine. mlx-serve ran a day
later on a busier Mac, where Kapitsa wrote 139-167 tok/s on Qwen3.6 35B-A3B · 4-bit and 57-61 tok/s ⚠ on
Qwen3.8 27B · 4-bit in the same session ([details](docs/benchmarks.md#mlx-serve-2026-10-18)); it was not run on the 4B
([details](docs/benchmarks.md#qwen35-4b-2026-10-18)). <sup>4B</sup> With z-lab's DFlash drafter, the 4B's default since
2026-10-18, measured that evening with the previous build run first and last as a control: it wrote 120 tok/s, in line
with its 122 in the bracket above ([details](docs/benchmarks.md#kapitsa-with-z-labs-dflash-drafter)). At bf16 the drafter
takes Kapitsa from 44 to 89 tok/s. Nemotron 3.5 Super-VL does not fit this Mac (70 GB of weights; Kapitsa refuses it
before downloading). Nemotron 3.5 Lightning fits it (18.2 GB of weights) but has not been measured on it yet; its numbers
are from the M5 Max below.

<details>
<summary>Follow-ups and memory, MacBook Pro (16-inch, 2024) · M4 Pro · 48 GB</summary>

A follow-up is the time to the first token of an answer in a conversation that already holds a 4K or 16K prompt.

| MacBook Pro (16-inch, 2024) · M4 Pro · 48 GB | Kapitsa | Next best on this Mac |
| --- | ---: | --- |
| Qwen3.8 27B · 4-bit, follow-up 4K / 16K | 0.28 / 0.27 s | MTPLX, 0.33 / 0.39 s |
| Qwen3.8 27B · 4-bit, peak memory | 19.4 GB | mlx-serve + MTP, 20.1 GB ⚠ |
| Qwen3.6 35B-A3B · 4-bit, follow-up 4K / 16K | 0.06 / 0.09 s | MTPLX, 0.10 / 0.15 s |
| Qwen3.6 35B-A3B · 4-bit, peak memory | 21.1 GB | mlx-serve + MTP, 22.5 GB |
| Nemotron 3.5 Bolt · NVIDIA 24 GB, follow-up 4K / 16K | 0.28 / 0.32 s | oMLX, 0.72 / 0.72 s ⚠ |
| Nemotron 3.5 Bolt · NVIDIA 24 GB, peak memory | 22.7 GB | oMLX, 30.3 GB ⚠ |

</details>

**A real opencode session (19 turns, up to 23K tokens) · Qwen3.6 35B-A3B · 4-bit · MacBook Pro (16-inch, 2024) · M4 Pro · 48 GB**

| Engine | Total time | Writing speed |
| --- | ---: | ---: |
| llama.cpp + MTP | — | — |
| Ollama (MLX runner, MTP) | 73 s | 94 tok/s |
| LM Studio (MLX engine) | 85 s | 68 tok/s |
| MTPLX | 139 s | 72 tok/s |
| oMLX + DFlash2 | 66 s | 107 tok/s |
| mlx-serve (MTP) | 62 s | 110 tok/s |
| mlx-serve (DFlash2) | 72 s | 85 tok/s |
| **Kapitsa** | **49 s** | **141 tok/s** |

opencode fixing a small Python project, recorded on Kapitsa and replayed request by request on every engine: the same
19 requests, from a 5.5K-token system prompt to a 23K-token conversation, greedy, each answer capped at its recorded
length. Total time is the time spent answering them; writing speed is tokens per second while the answers stream.
Kapitsa recorded the session, so the others also re-read about 2 s of recorded answers that differ from their own.
The same session on Qwen3.8 27B · 4-bit, same Mac: Kapitsa 150 s, next oMLX + DFlash2 278 s. llama.cpp + MTP was not
run this time. [Details](docs/benchmarks.md#a-real-coding-agent-session-2026-10-18)

### MacBook Pro (16-inch, 2026) · M5 Max · 128 GB

| Engine | Qwen3.8 27B · 4-bit | Qwen3.6 35B-A3B · 4-bit | Nemotron 3.5 Lightning · NVIDIA NVFP4 | Nemotron 3.5 Super-VL · NVIDIA NVFP4 |
| --- | ---: | ---: | ---: | ---: |
| llama.cpp + MTP | — | — | 159 tok/s | 66 tok/s |
| mlx-lm | — | — | 139 tok/s | 46 tok/s |
| mlx-vlm | — | — | 137 tok/s | 46 tok/s |
| oMLX | — | — | 190 tok/s | 52 tok/s |
| **Kapitsa** | **73 tok/s** ⚠ | **250 tok/s** ⚠ | **239 tok/s** | **85 tok/s** |

On the Nemotron models Kapitsa and mlx-vlm run NVIDIA's NVFP4 checkpoints (mlx-vlm with the DSpark drafter on Lightning
and MTP from its unreleased git main on Super-VL), oMLX and mlx-lm run MLX 4-bit conversions (oMLX with MTP on Lightning;
on Super-VL text only, as they cannot load NVIDIA's checkpoint) and llama.cpp ggml-org's Q4_0 GGUFs. On Lightning Kapitsa
writes fastest but reads prompts slowest (1,900 tok/s at 4K against oMLX's 3,600,
[details](docs/benchmarks.md#nemotron-35-lightning-30b-a3b-m5-max-128-gb-2026-10-18)). The other engines have not been run
on the Qwen models on this Mac yet, nor has Nemotron 3.5 Bolt.

<details>
<summary>Follow-ups and memory, MacBook Pro (16-inch, 2026) · M5 Max · 128 GB</summary>

| MacBook Pro (16-inch, 2026) · M5 Max · 128 GB | Qwen3.8 27B · 4-bit | Qwen3.6 35B-A3B · 4-bit |
| --- | ---: | ---: |
| Follow-up 4K / 16K | 121.6 / 135.3 ms ⚠ | 34.9 / 40.2 ms ⚠ |
| Peak memory | 19.4 GB ⚠ | 21.0 GB ⚠ |

| MacBook Pro · M5 Max · 128 GB | Kapitsa | Best other engine on this Mac |
| --- | ---: | --- |
| Nemotron 3.5 Lightning, follow-up 4K / 16K | 0.07 / 0.09 s | llama.cpp, 0.07 / 0.08 s |
| Nemotron 3.5 Lightning, peak memory | 18.6 GB | oMLX (MLX 4-bit), 20.9 GB |
| Nemotron 3.5 Super-VL, follow-up 4K / 16K | 0.22 / 0.26 s | llama.cpp, 0.19 / 0.20 s |
| Nemotron 3.5 Super-VL, peak memory | 70.4 GB | oMLX (MLX 4-bit), 74.8 GB |

</details>

Same prompts on every engine, thinking off, median of 5 runs, each engine at its latest release. Desktop apps used the
GPU during the Bolt runs, alike for every engine, and during the M5 Max runs, which are flagged (⚠). These measure
speed, not model quality.
[How it was measured](docs/benchmarks.md) · [M5 Max report](docs/benchmarks-m5-max.md) ·
[Run the benchmarks yourself](bench/harness/README.md) · [Community results](bench/community/RESULTS.md)

## Connect your apps

Launch an installed coding agent against Kapitsa:

```bash
kapitsa claude
kapitsa codex
kapitsa opencode
kapitsa pi
```

Each launcher reuses a server on port 8000 or starts one for the session, and stops it when the agent exits. To choose
the model, add `--model bolt` or `--model 35b`. [Agent setup and options](docs/usage.md#coding-agents)

For API clients, use `http://127.0.0.1:8000/v1` for OpenAI Chat Completions and Responses, or
`http://127.0.0.1:8000` for Anthropic Messages. Streaming, reasoning and tool calls are supported. Any API key works
locally unless you start the server with `--api-key`.

<details>
<summary>Example API request</summary>

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.5-4b",
  "messages": [{"role": "user", "content": "Explain speculative decoding."}]
}'
```

</details>

Use `--ctx` to set the context length and `--mem-budget` to limit GPU memory. Requests sent at once (parallel
subagents, several clients) run side by side, taking turns on the GPU, each with the tokens it would get alone;
`--slots` sets how many (default: up to 4, as memory allows), each keeping its conversation cached.
`brew services start kapitsa` keeps a server running in the background.
[All server options and API details](docs/usage.md#server-options)

## Architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-dark.svg">
  <img alt="Kapitsa's architecture: the APIs, the engine with its conversation slots, prefill on the GPU and the Neural Engine, draft-and-verify decoding, the Qwen and Nemotron backends and their Metal kernels." src="docs/assets/architecture-light.svg">
</picture>

* **Reads on two chips.** Prompts are read in chunks on the GPU; for dense models and Nemotron, the Neural Engine
  takes part of each layer at the same time.
* **Writes several tokens per pass.** A draft model proposes a block of tokens (DFlash2 for Qwen, NVIDIA's LiLiCorr for
  Nemotron, the model's MTP head otherwise) and the model checks them all in one GPU pass.
* **Follow-ups read only what's new.** Each conversation slot keeps its KV cache and recurrent-state checkpoints at
  turn boundaries, and the last answer is read in while you read it.
* **Two model families.** Qwen3.5 (Gated DeltaNet and attention, dense or mixture of experts) and Nemotron-H (Mamba-2,
  attention and LatentMoE), with image encoders for both.
* **Hand-written Metal kernels** for each weight format, compiled at runtime, with no MLX or llama.cpp underneath.

[Architecture walkthrough and source map](docs/architecture.md)

## Documentation

* [Usage guide](docs/usage.md): models and downloads, server options, APIs, coding agents, images and memory.
* [Benchmarks](docs/benchmarks.md): every engine, setting and run behind the tables above.
* [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Press kit](docs/assets/brand/README.md)

Kapitsa is licensed under [Apache 2.0](LICENSE), copyright 2026 [Nace AI](https://nace.ai).
[Third-party notices and acknowledgments](THIRD_PARTY_NOTICES) cover included and adapted code.
