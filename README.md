<picture>
  <source media="(max-width: 600px)" srcset="docs/assets/header-narrow.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/header-dark.svg">
  <img alt="Kapitsa: the fastest Apple silicon inference engine built for Qwen and Nemotron. Writing speed of Qwen3.6 35B-A3B on a MacBook Pro (2024), M4 Pro, 48 GB, in tokens per second, at 4-bit: Kapitsa 154, oMLX with DFlash2 127, Ollama 89, LM Studio 86, MTPLX 80, llama.cpp with MTP 69." src="docs/assets/header-light.svg">
</picture>

[Install](#install) · [Models](#models) · [Benchmarks](#benchmarks) · [Docs](docs/usage.md)

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

## Benchmarks

<sub>Writing speed of a coding answer, each engine in its fastest setup; ⚠ other apps used the GPU; — not measured.
[How it was measured](docs/benchmarks.md)</sub>

### Qwen · MacBook Pro (16-inch, 2024) · M4 Pro · 48 GB

| Engine | Qwen3.5 4B · 4-bit | Qwen3.8 27B · 4-bit | Qwen3.6 35B-A3B · 4-bit | Qwen3.8-Flash-Next · 4-bit<br><sub>128 GB Macs</sub> |
| --- | ---: | ---: | ---: | ---: |
| llama.cpp + MTP | 70 tok/s | 18 tok/s | 69 tok/s | — |
| Ollama (MLX runner, MTP) | 109 tok/s | 32 tok/s | 89 tok/s | — |
| LM Studio (MLX engine) | 81 tok/s | 15 tok/s | 86 tok/s | — |
| MTPLX | 104 tok/s | 36 tok/s | 80 tok/s | — |
| mlx-lm | 73 tok/s | — | — | — |
| mlx-vlm | — | — | — | — |
| oMLX | 102 tok/s | 45 tok/s | 127 tok/s | — |
| mlx-serve (MTP) | — | 38 tok/s ⚠ | 124 tok/s | — |
| mlx-serve (DFlash2) | — | 52 tok/s ⚠ | 104 tok/s | — |
| **Kapitsa** | **179 tok/s**<sup>4B</sup> | **63 tok/s** | **154 tok/s** | — |

<sub>oMLX runs DFlash2 on the 27B and 35B, MTP on the 4B. <sup>4B</sup> Measured later with its new drafter; the old build,
rerun as a control, matched its 122. A real opencode session on the 35B: Kapitsa **49 s**, mlx-serve 62 s, oMLX 66 s
([details](docs/benchmarks.md#a-real-coding-agent-session-2026-10-18)).</sub>

### Qwen · MacBook Pro (16-inch, 2026) · M5 Max · 128 GB

| Engine | Qwen3.5 4B · 4-bit | Qwen3.8 27B · 4-bit | Qwen3.6 35B-A3B · 4-bit | Qwen3.8-Flash-Next · 4-bit |
| --- | ---: | ---: | ---: | ---: |
| llama.cpp + MTP | — | — | — | 96 tok/s |
| Ollama (MLX runner, MTP) | — | — | — | — |
| LM Studio (MLX engine) | — | — | — | — |
| MTPLX | — | — | — | — |
| mlx-lm | — | — | — | —<br><sub>no support</sub> |
| mlx-vlm | — | — | — | 77 tok/s ⚠ |
| oMLX | — | — | — | **107 tok/s** |
| mlx-serve (MTP) | — | — | — | — |
| mlx-serve (DFlash2) | — | — | — | — |
| **Kapitsa** | — | **73 tok/s** ⚠ | **250 tok/s** ⚠ | 102 tok/s |

<sub>Flash-Next: the same 4-bit weights and MTP head on Kapitsa, oMLX and mlx-vlm; Kapitsa leads greedy (108 against 106)
and on memory (70.5 GB against 104) ([details](docs/benchmarks.md#qwen38-flash-next-180b-a7b-m5-max-128-gb-2026-10-18)).</sub>

### Nemotron · MacBook Pro (16-inch, 2024) · M4 Pro · 48 GB

| Engine | Nemotron-Labs Dory1.0 4B · NVIDIA NVFP4 | Nemotron 3.5 Lightning · NVIDIA NVFP4 | Nemotron 3.5 Bolt · NVIDIA 24 GB | Nemotron 3.5 Super-VL · NVIDIA NVFP4<br><sub>96 GB+ Macs</sub> |
| --- | ---: | ---: | ---: | ---: |
| llama.cpp + MTP | — | — | —<br><sub>no Metal support</sub> | — |
| Ollama | — | — | —<br><sub>no support</sub> | — |
| LM Studio | — | — | — | — |
| mlx-lm | — | — | — | — |
| mlx-vlm | — | — | 32 tok/s | — |
| oMLX | — | — | 34 tok/s ⚠ | — |
| mlx-serve | — | — | — | — |
| **Kapitsa** | — | — | **64 tok/s** | — |

<sub>oMLX runs Bolt without a drafter: its LiLiCorr fails when sampling.</sub>

### Nemotron · MacBook Pro (16-inch, 2026) · M5 Max · 128 GB

| Engine | Nemotron-Labs Dory1.0 4B · NVIDIA NVFP4 | Nemotron 3.5 Lightning · NVIDIA NVFP4 | Nemotron 3.5 Bolt · NVIDIA 24 GB | Nemotron 3.5 Super-VL · NVIDIA NVFP4 |
| --- | ---: | ---: | ---: | ---: |
| llama.cpp + MTP | — | 159 tok/s | — | 66 tok/s |
| Ollama | — | — | — | — |
| LM Studio | — | — | — | — |
| mlx-lm | — | 139 tok/s | — | 46 tok/s |
| mlx-vlm | — | 137 tok/s | — | 46 tok/s |
| oMLX | — | 190 tok/s | — | 52 tok/s |
| mlx-serve | — | — | — | — |
| **Kapitsa** | — | **239 tok/s** | — | **85 tok/s** |

<sub>Kapitsa and mlx-vlm run NVIDIA's NVFP4 checkpoints, oMLX and mlx-lm MLX 4-bit, llama.cpp Q4_0. On Lightning, oMLX
reads prompts faster (3,600 against 1,900 tok/s at 4K).</sub>

<sub>Measure your Mac: `kapitsa bench --submit` adds it to the [community results](bench/community/RESULTS.md).</sub>

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-dark.svg">
  <img alt="Kapitsa's architecture: the APIs, the engine with its conversation slots, prefill on the GPU and the Neural Engine, draft-and-verify decoding, the Qwen and Nemotron backends and their Metal kernels." src="docs/assets/architecture-light.svg">
</picture>

<sub>[Architecture](docs/architecture.md) · [Contributing](CONTRIBUTING.md) · [Press kit](docs/assets/brand/README.md) ·
Apache 2.0, © 2026 [Nace AI](https://nace.ai) · [Third-party notices](THIRD_PARTY_NOTICES)</sub>
