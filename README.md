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
| **Qwen3.5-4B** · `4b`<br><sub>dense</sub> | **4-bit · 3.3 GB**<br><sub>also 6-bit 4.3 · 8-bit 5.4 · bf16 9.3 GB</sub> | MTP |
| **Qwen3.6-35B-A3B** · `35b`<br><sub>MoE, 3B active</sub> | **4-bit · 22.1 GB**<br><sub>also 5-bit 26.4 · 6-bit 30.7 · 8-bit 39.4 · bf16 71.9 GB</sub> | DFlash2 / MTP |
| **Qwen3.8-27B** · `27b`<br><sub>dense</sub> | **4-bit · 20.2 GB**<br><sub>also 5-bit 23.5 · 6-bit 26.9 · 8-bit 33.6 · bf16 58.8 GB</sub> | DFlash2 / MTP |
| **Nemotron 3.5 Bolt 74B-A9B** · `bolt`<br><sub>hybrid Mamba-2 MoE, 9B active</sub> | **NVIDIA 24 GB · 25.1 GB**<br><sub>2-bit experts, NVFP4, 8-bit</sub> | LiLiCorr |

bf16 is full precision; pick another precision with `--model 35b:6bit`, `--model 27b:bf16` and so on.

Downloads include the model's companion files ([precisions](docs/usage.md#precisions)). Bolt is NVIDIA's own `24gb` checkpoint. Qwen3.6, Qwen3.8 and Bolt read
[images](docs/usage.md#images); the 4B is text only. Other Qwen3.5-family models load from a Hugging Face repo, a
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

| Engine | Qwen3.6 35B-A3B · 4-bit | Qwen3.8 27B · 4-bit | Qwen3.5 4B · 4-bit |
| --- | ---: | ---: | ---: |
| llama.cpp + MTP | 69 tok/s | 18 tok/s | 70 tok/s |
| Ollama (MLX runner, MTP) | 89 tok/s | 32 tok/s | 109 tok/s |
| LM Studio (MLX engine) | 86 tok/s | 15 tok/s | 81 tok/s |
| MTPLX | 80 tok/s | 36 tok/s | 104 tok/s |
| oMLX + DFlash2 | 127 tok/s | 45 tok/s | 102 tok/s |
| mlx-serve (MTP) | 124 tok/s | 38 tok/s ⚠ | — |
| mlx-serve (DFlash2) | 104 tok/s | 52 tok/s ⚠ | — |
| **Kapitsa** | **154 tok/s** | **63 tok/s** | **122 tok/s** |

| Engine | Nemotron 3.5 Bolt 74B-A9B · NVIDIA 24 GB |
| --- | ---: |
| mlx-vlm | 32 tok/s |
| oMLX | 34 tok/s ⚠ |
| oMLX + LiLiCorr | fails when sampling |
| **Kapitsa** | **64 tok/s** |

Bolt is NVIDIA's 24 GB checkpoint on every engine; oMLX's LiLiCorr drafter fails every sampled request. mlx-serve ran a
day later on a busier Mac, where Kapitsa wrote 139-167 tok/s on Qwen3.6 35B-A3B · 4-bit and 57-61 tok/s ⚠ on
Qwen3.8 27B · 4-bit in the same session ([details](docs/benchmarks.md#mlx-serve-2026-10-18)). On Qwen3.5 4B oMLX
runs its own MTP path, its fastest setup there; mlx-serve was not run on it
([details](docs/benchmarks.md#qwen35-4b-2026-10-18)).

<details>
<summary>Follow-ups and memory, MacBook Pro (16-inch, 2024) · M4 Pro · 48 GB</summary>

A follow-up is the time to the first token of an answer in a conversation that already holds a 4K or 16K prompt.

| MacBook Pro (16-inch, 2024) · M4 Pro · 48 GB | Kapitsa | Next best on this Mac |
| --- | ---: | --- |
| Qwen3.6 35B-A3B · 4-bit, follow-up 4K / 16K | 0.06 / 0.09 s | MTPLX, 0.10 / 0.15 s |
| Qwen3.6 35B-A3B · 4-bit, peak memory | 21.1 GB | mlx-serve + MTP, 22.5 GB |
| Qwen3.8 27B · 4-bit, follow-up 4K / 16K | 0.28 / 0.27 s | MTPLX, 0.33 / 0.39 s |
| Qwen3.8 27B · 4-bit, peak memory | 19.4 GB | mlx-serve + MTP, 20.1 GB ⚠ |
| Nemotron 3.5 Bolt · NVIDIA 24 GB, follow-up 4K / 16K | 0.28 / 0.32 s | oMLX, 0.72 / 0.72 s ⚠ |
| Nemotron 3.5 Bolt · NVIDIA 24 GB, peak memory | 22.7 GB | oMLX, 30.3 GB ⚠ |

</details>

### MacBook Pro (16-inch, 2026) · M5 Max · 128 GB

| Engine | Qwen3.6 35B-A3B · 4-bit | Qwen3.8 27B · 4-bit |
| --- | ---: | ---: |
| **Kapitsa** | **250 tok/s** ⚠ | **73 tok/s** ⚠ |
| Other engines | — | — |

Other engines on this Mac: coming. Nemotron 3.5 Bolt here: not measured yet.

<details>
<summary>Follow-ups and memory, MacBook Pro (16-inch, 2026) · M5 Max · 128 GB</summary>

| MacBook Pro (16-inch, 2026) · M5 Max · 128 GB | Qwen3.6 35B-A3B · 4-bit | Qwen3.8 27B · 4-bit |
| --- | ---: | ---: |
| Follow-up 4K / 16K | 34.9 / 40.2 ms ⚠ | 121.6 / 135.3 ms ⚠ |
| Peak memory | 21.0 GB ⚠ | 19.4 GB ⚠ |

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
