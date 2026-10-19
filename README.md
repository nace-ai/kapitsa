<picture>
  <source media="(max-width: 600px)" srcset="docs/assets/header-narrow.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/header-dark.svg">
  <img alt="Kapitsa: the fastest Apple silicon inference engine built for Qwen and Nemotron. Writing speed of Qwen3.6 35B-A3B on a MacBook Pro (2024), M4 Pro, 48 GB, in tokens per second, at 4-bit: Kapitsa 154, oMLX with DFlash2 127, Ollama 89, LM Studio 86, MTPLX 80, llama.cpp with MTP 69." src="docs/assets/header-light.svg">
</picture>

[Install](#install) · [Benchmarks](#benchmarks) · [Models](#models) · [Docs](docs/usage.md)

## Install

```bash
brew install nace-ai/tap/kapitsa
# or: curl -fsSL https://raw.githubusercontent.com/nace-ai/kapitsa/main/install.sh | bash
```

```bash
kapitsa opencode --model 35b    # also: kapitsa claude, kapitsa codex, kapitsa pi
```

<sub>The 35B needs a Mac with 32 GB or more; use `--model 4b` on 16-24 GB. `kapitsa serve` starts the chat page and the
OpenAI and Anthropic APIs at http://127.0.0.1:8000. Kapitsa updates itself. [Usage guide](docs/usage.md)</sub>

## Benchmarks

<sub>Writing speed of a coding answer, each engine in its fastest setup; the fastest per model in bold; — not measured;
⚠ other apps used the GPU. [How it was measured](docs/benchmarks.md)</sub>

### MacBook Pro (16-inch, 2024) · M4 Pro · 48 GB

| Model | **Kapitsa** | oMLX | mlx‑serve | Ollama | MTPLX | LM Studio | llama.cpp | MLX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ***Qwen*** | | | | | | | | |
| Qwen3.5&nbsp;4B<sup>*</sup><br><sub>4-bit</sub> | **179&nbsp;tok/s** | 102 | — | 109 | 104 | 81 | 70 | 73 |
| Qwen3.8&nbsp;27B<br><sub>4-bit</sub> | **63&nbsp;tok/s** | 45 | 52&nbsp;⚠ | 32 | 36 | 15 | 18 | — |
| Qwen3.6&nbsp;35B<br><sub>4-bit</sub> | **154&nbsp;tok/s** | 127 | 124 | 89 | 80 | 86 | 69 | — |
| Flash‑Next&nbsp;180B<br><sub>4-bit · 128 GB Macs</sub> | — | — | — | — | — | — | — | — |
| ***Nemotron*** | | | | | | | | |
| Dory&nbsp;4B<br><sub>NVFP4</sub> | — | — | — | — | — | — | — | — |
| Lightning&nbsp;30B<br><sub>NVFP4</sub> | — | — | — | — | — | — | — | — |
| Bolt&nbsp;74B<br><sub>NVIDIA 24 GB</sub> | **64&nbsp;tok/s** | 34&nbsp;⚠ | — | —<br><sub>no support</sub> | —<br><sub>no support</sub> | — | —<br><sub>no Metal support</sub> | 32 |
| Super‑VL&nbsp;120B<br><sub>NVFP4 · 96 GB+ Macs</sub> | — | — | — | — | — | — | — | — |

<sub>MLX: Apple's mlx-lm or mlx-vlm, whichever runs the model. oMLX runs DFlash2 on the 27B and 35B, MTP on the 4B, no
drafter on Bolt; mlx-serve the faster of MTP and DFlash2. <sup>*</sup> Measured later with its new drafter; the old build, rerun as a control, matched its 122. A real
opencode session on the 35B: Kapitsa **49 s**, mlx-serve 62 s, oMLX 66 s
([details](docs/benchmarks.md#a-real-coding-agent-session-2026-10-18)).</sub>

### MacBook Pro (16-inch, 2026) · M5 Max · 128 GB

| Model | **Kapitsa** | oMLX | mlx‑serve | Ollama | MTPLX | LM Studio | llama.cpp | MLX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ***Qwen*** | | | | | | | | |
| Qwen3.5&nbsp;4B<br><sub>4-bit</sub> | — | — | — | — | — | — | — | — |
| Qwen3.8&nbsp;27B<br><sub>4-bit</sub> | **73&nbsp;tok/s** | — | — | — | — | — | — | — |
| Qwen3.6&nbsp;35B<br><sub>4-bit</sub> | **250&nbsp;tok/s** | — | — | — | — | — | — | — |
| Flash‑Next&nbsp;180B<br><sub>4-bit</sub> | 102&nbsp;tok/s | **107** | — | — | — | — | 96 | 77&nbsp;⚠ |
| ***Nemotron*** | | | | | | | | |
| Dory&nbsp;4B<br><sub>NVFP4</sub> | — | — | — | — | — | — | — | — |
| Lightning&nbsp;30B<br><sub>NVFP4</sub> | **239&nbsp;tok/s** | 190 | — | — | — | — | 159 | 139 |
| Bolt&nbsp;74B<br><sub>NVIDIA 24 GB</sub> | — | — | — | — | — | — | — | — |
| Super‑VL&nbsp;120B<br><sub>NVFP4</sub> | **85&nbsp;tok/s** | 52 | — | — | — | — | 66 | 46 |

<sub>Other apps used the GPU during Kapitsa's 27B and 35B runs here. Flash-Next: the same 4-bit weights and MTP head on
Kapitsa, oMLX and MLX (mlx-vlm); Kapitsa leads greedy (108
against 106) and on memory (70.5 GB against 104). Nemotron: NVIDIA's NVFP4 on Kapitsa and mlx-vlm, MLX 4-bit on oMLX
and mlx-lm, Q4_0 on llama.cpp; on Lightning oMLX reads prompts faster (3,600 against 1,900 tok/s at 4K).</sub>

<sub>Measure your Mac: `kapitsa bench --submit` adds it to the [community results](bench/community/RESULTS.md).</sub>

## Models

| Model | Download | Speculation |
| --- | --- | --- |
| **Qwen3.5-4B** · `4b`<br><sub>dense</sub> | **4-bit · 4.6 GB**<br><sub>also 6-bit 5.6 · 8-bit 6.7 · bf16 10.6 GB</sub> | DFlash / MTP |
| **Qwen3.8-27B** · `27b`<br><sub>dense</sub> | **4-bit · 20.2 GB**<br><sub>also 5-bit 23.5 · 6-bit 26.9 · 8-bit 33.6 · bf16 58.8 GB</sub> | DFlash2 / MTP |
| **Qwen3.6-35B-A3B** · `35b`<br><sub>MoE, 3B active</sub> | **4-bit · 22.1 GB**<br><sub>also 5-bit 26.4 · 6-bit 30.7 · 8-bit 39.4 · bf16 71.9 GB</sub> | DFlash2 / MTP |
| **Qwen3.8-Flash-Next 180B-A7B** · `flash-next`<br><sub>MoE, 7B active; 128 GB Macs</sub> | **4-bit · 105.9 GB**<br><sub>4/6/8-bit MLX; its 32 GB n-gram table stays on the SSD</sub> | MTP |
| **Nemotron-Labs Dory1.0 4B** · `dory`<br><sub>looped dense, 1-4 loops per token; text only</sub> | **NVIDIA NVFP4 · 3.2 GB**<br><sub>also bf16 8.0 GB</sub> | — |
| **Nemotron 3.5 Lightning 30B-A3B** · `lightning`<br><sub>hybrid Mamba-2 MoE, 3B active; text only</sub> | **NVIDIA NVFP4 · 22.9 GB**<br><sub>4-bit NVFP4 experts, FP8, BF16</sub> | DSpark |
| **Nemotron 3.5 Bolt 74B-A9B** · `bolt`<br><sub>hybrid Mamba-2 MoE, 9B active</sub> | **NVIDIA 24 GB · 25.1 GB**<br><sub>2-bit experts, NVFP4, 8-bit</sub> | LiLiCorr |
| **Nemotron 3.5 Super-VL 120B-A12B** · `super`<br><sub>hybrid Mamba-2 MoE, 12B active; 96 GB+ Macs</sub> | **NVIDIA NVFP4 · 76.4 GB**<br><sub>4-bit NVFP4 experts, FP8, BF16</sub> | LiLiCorr |

<sub>Other precisions: `--model 35b:6bit`, `--model 27b:bf16`. All but Lightning and Dory read images; Dory thinks 1-4
loops per token (`--loops`). [Details](docs/usage.md#models-and-downloads)</sub>

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-dark.svg">
  <img alt="Kapitsa's architecture: the APIs, the engine with its conversation slots, prefill on the GPU and the Neural Engine, draft-and-verify decoding, the Qwen and Nemotron backends and their Metal kernels." src="docs/assets/architecture-light.svg">
</picture>

<sub>[Architecture](docs/architecture.md) · [Contributing](CONTRIBUTING.md) · [Press kit](docs/assets/brand/README.md) ·
Apache 2.0, © 2026 [Nace AI](https://nace.ai) · [Third-party notices](THIRD_PARTY_NOTICES)</sub>
