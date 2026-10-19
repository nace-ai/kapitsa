<picture>
  <source media="(max-width: 600px)" srcset="docs/assets/header-narrow.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/header-dark.svg">
  <img alt="Kapitsa: the fastest Apple silicon inference engine built for Qwen and Nemotron. Writing speed of Qwen3.6 35B-A3B on a MacBook Pro (2024), M4 Pro, 48 GB, in tokens per second, at 4-bit: Kapitsa 154, oMLX with DFlash2 127, Ollama 89, LM Studio 86, MTPLX 80, llama.cpp with MTP 69." src="docs/assets/header-light.svg">
</picture>

[Install](#install) · [Benchmarks](#benchmarks) · [Models](#models) · [Docs](docs/usage.md)

## Install

**Requirements:** a Mac with Apple silicon (M1 or newer) and macOS 14 Sonoma or later.

```bash
brew install nace-ai/tap/kapitsa
# or: curl -fsSL https://raw.githubusercontent.com/nace-ai/kapitsa/main/install.sh | bash
```

```bash
kapitsa opencode --model 35b    # also: kapitsa claude, kapitsa codex, kapitsa pi
```

<sub>On a 16-24 GB Mac use `--model 4b`. `kapitsa serve` starts the chat page and the
OpenAI and Anthropic APIs at http://127.0.0.1:8000. Kapitsa updates itself. [Usage guide](docs/usage.md)</sub>

## Benchmarks

<sub>Writing speed of a coding answer, each engine in its fastest setup; Kapitsa highlighted; 🟩 the fastest per model; — not measured;
⚠ other apps used the GPU. [How it was measured](docs/benchmarks.md)</sub>

### MacBook Pro (16-inch, 2024) · M4 Pro · 48 GB

#### Qwen

| Model | <mark>**Kapitsa**</mark> | <sub>oMLX</sub> | <sub>mlx‑serve</sub> | <sub>Ollama</sub> | <sub>MTPLX</sub> | <sub>LM&nbsp;Studio</sub> | <sub>llama.cpp</sub> | <sub>MLX</sub> | <sub>Rapid‑MLX</sub> |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5<br>4B<sup>*</sup><br><sub>4-bit</sub> | 🟩&nbsp;<mark>**179&nbsp;tok/s**</mark> | 102 | 137 | 109 | 104 | 81 | 70 | 88&nbsp;⚠ | 98 |
| Qwen3.8<br>27B<br><sub>4-bit</sub> | 🟩&nbsp;<mark>**63&nbsp;tok/s**</mark> | 45 | 54 | 32 | 36 | 15 | 18 | 28 | — |
| Qwen3.6<br>35B‑A3B<br><sub>4-bit</sub> | 🟩&nbsp;<mark>**154&nbsp;tok/s**</mark> | 127 | 124 | 89 | 80 | 86 | 69 | 86 | — |

#### Nemotron

| Model | <mark>**Kapitsa**</mark> | <sub>oMLX</sub> | <sub>llama.cpp</sub> | <sub>MLX</sub> |
| --- | ---: | ---: | ---: | ---: |
| Nemotron Dory1.0 4B<br><sub>NVFP4</sub> | 🟩&nbsp;<mark>**64&nbsp;tok/s**</mark> | —<br><sub>no support</sub> | 6&nbsp;⚠ | 48 |
| Nemotron 3.5 Lightning 30B‑A3B<br><sub>NVFP4</sub> | 🟩&nbsp;<mark>**106&nbsp;tok/s**</mark> | — | — | 65 |
| Nemotron 3.5 Bolt 74B‑A9B<br><sub>2/4/8-bit mix</sub> | 🟩&nbsp;<mark>**64&nbsp;tok/s**</mark> | 35 | —<br><sub>no Metal support</sub> | 32 |

<sub>Ollama, LM Studio, MTPLX and mlx-serve can't run Dory or Bolt, so this Nemotron table leaves them out.
MLX: Apple's mlx-lm or mlx-vlm, whichever runs the model. oMLX runs DFlash2 on the 27B and 35B, MTP on the 4B, no
drafter on Bolt; mlx-serve the faster of MTP and DFlash2. <sup>*</sup> Measured later with its new drafter; the old build, rerun as a control, matched its 122. A real
opencode session on the 35B: Kapitsa **49 s**, mlx-serve 62 s, oMLX 66 s
([details](docs/benchmarks.md#a-real-coding-agent-session-2026-10-18)).</sub>

<sub>Dory: 2 loops; Kapitsa per-loop KV (`--loop-kv per`), llama.cpp NVIDIA's Dory fork on its Q5_K_M (last-loop KV), MLX mlx-lm's Dory pull request on snsnc's 4-bit (per-loop KV) ([details](docs/benchmarks.md#nemotron-dory10-4b-2026-10-19)).</sub>

<sub>Lightning: NVIDIA's NVFP4 with its DSpark drafter on Kapitsa and MLX (mlx-vlm); oMLX and llama.cpp not measured yet ([details](docs/benchmarks.md#nemotron-35-lightning)).</sub>

### MacBook Pro (16-inch, 2026) · M5 Max · 128 GB

#### Qwen

| Model | <mark>**Kapitsa**</mark> | <sub>oMLX</sub> | <sub>mlx‑serve</sub> | <sub>Ollama</sub> | <sub>MTPLX</sub> | <sub>LM&nbsp;Studio</sub> | <sub>llama.cpp</sub> | <sub>MLX</sub> | <sub>Rapid‑MLX</sub> |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5<br>4B<br><sub>4-bit</sub> | 🟩&nbsp;<mark>**294&nbsp;tok/s**</mark> | — | — | 213 | 175 | 154 | — | — | — |
| Qwen3.8<br>27B<br><sub>4-bit</sub> | 🟩&nbsp;<mark>**115&nbsp;tok/s**</mark> | — | — | 68 | 69 | 32 | — | 30 | — |
| Qwen3.6<br>35B‑A3B<br><sub>4-bit</sub> | 🟩&nbsp;<mark>**272&nbsp;tok/s**</mark> | — | — | 166 | 123 | 127 | — | 112 | — |
| Qwen3.8<br>Flash‑Next<br>180B‑A7B<br><sub>4-bit</sub> | <mark>**102&nbsp;tok/s**</mark> | 107 | 🟩&nbsp;**108** | 60<sup>†</sup> | 88 | —<br><sub>won't load</sub> | 96 | 77 | — |

#### Nemotron

| Model | <mark>**Kapitsa**</mark> | <sub>oMLX</sub> | <sub>mlx‑serve</sub> | <sub>Ollama</sub> | <sub>MTPLX</sub> | <sub>LM&nbsp;Studio</sub> | <sub>llama.cpp</sub> | <sub>MLX</sub> |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nemotron Dory1.0 4B<br><sub>NVFP4</sub> | — | — | —<br><sub>no support</sub> | —<br><sub>no support</sub> | —<br><sub>no support</sub> | —<br><sub>no support</sub> | — | — |
| Nemotron 3.5 Lightning 30B‑A3B<br><sub>NVFP4</sub> | 🟩&nbsp;<mark>**239&nbsp;tok/s**</mark> | 190 | 190 | 153 | 131 | 110 | 159 | 139 |
| Nemotron 3.5 Bolt 74B‑A9B<br><sub>2/4/8-bit mix</sub> | 🟩&nbsp;<mark>**127&nbsp;tok/s**</mark> | — | —<br><sub>no support</sub> | —<br><sub>no support</sub> | —<br><sub>no support</sub> | —<br><sub>no support</sub> | —<br><sub>no Metal support</sub> | — |
| Nemotron 3.5 Super‑VL 120B‑A12B<br><sub>NVFP4</sub> | 🟩&nbsp;<mark>**85&nbsp;tok/s**</mark> | 52 | — | 64 | — | 61 | 66 | 46 |

<sub>Other apps used the GPU during Kapitsa's 27B and 35B runs and the MLX Flash-Next run here. MLX on the 27B and 35B: mlx-lm.
Flash-Next: the same 4-bit weights and MTP head on Kapitsa, oMLX and MLX (mlx-vlm); mlx-serve runs its vendor's 4.7-bit pack, MTPLX its
author's pack, Ollama its NVFP4 tag. <sup>†</sup> Ollama fails every 4K and 16K prompt there (out of GPU memory). LM Studio refuses to load its
119 GB GGUF on 128 GB. Kapitsa uses the least memory (70.5 GB; mlx-serve 78.8, oMLX 104). Nemotron: NVIDIA's NVFP4 on Kapitsa, mlx-vlm and Ollama, MLX 4-bit on oMLX,
mlx-lm, mlx-serve and MTPLX, Q4_0 on llama.cpp, Q4_K_M on LM Studio; on Lightning oMLX reads prompts faster (3,600 against 1,900 tok/s at 4K).</sub>

<sub>Measure your Mac: `kapitsa bench --submit` adds it to the [community results](bench/community/RESULTS.md).</sub>

## Models

| Model | Download | Speculation |
| --- | --- | --- |
| **Qwen3.5-4B** · `4b`<br><sub>dense</sub> | **4-bit · 4.6 GB**<br><sub>also 6-bit 5.6 · 8-bit 6.7 · bf16 10.6 GB</sub> | DFlash / MTP |
| **Qwen3.8-27B** · `27b`<br><sub>dense</sub> | **4-bit · 20.2 GB**<br><sub>also 5-bit 23.5 · 6-bit 26.9 · 8-bit 33.6 · bf16 58.8 GB</sub> | DFlash2 / MTP |
| **Qwen3.6-35B-A3B** · `35b`<br><sub>MoE, 3B active</sub> | **4-bit · 22.1 GB**<br><sub>also 5-bit 26.4 · 6-bit 30.7 · 8-bit 39.4 · bf16 71.9 GB</sub> | DFlash2 / MTP |
| **Qwen3.8-Flash-Next 180B-A7B** · `flash-next`<br><sub>MoE, 7B active; 128 GB Macs</sub> | **4-bit · 105.9 GB**<br><sub>4/6/8-bit MLX; its 32 GB n-gram table stays on the SSD</sub> | MTP |
| **Nemotron Dory1.0 4B** · `dory`<br><sub>looped dense, 1-4 loops per token; text only</sub> | **NVIDIA NVFP4 · 3.2 GB**<br><sub>also bf16 8.0 GB</sub> | — |
| **Nemotron 3.5 Lightning 30B-A3B** · `lightning`<br><sub>hybrid Mamba-2 MoE, 3B active; text only</sub> | **NVIDIA NVFP4 · 22.9 GB**<br><sub>4-bit NVFP4 experts, FP8, BF16</sub> | DSpark |
| **Nemotron 3.5 Bolt 74B-A9B** · `bolt`<br><sub>hybrid Mamba-2 MoE, 9B active</sub> | **NVIDIA 24 GB · 25.1 GB**<br><sub>2-bit experts, NVFP4, 8-bit</sub> | LiLiCorr |
| **Nemotron 3.5 Super-VL 120B-A12B** · `super`<br><sub>hybrid Mamba-2 MoE, 12B active; 96 GB+ Macs</sub> | **NVIDIA NVFP4 · 76.4 GB**<br><sub>4-bit NVFP4 experts, FP8, BF16</sub> | LiLiCorr |

<sub>Other precisions: `--model 35b:6bit`, `--model 27b:bf16`. All but Lightning and Dory read images; Dory thinks 1-4
loops per token (`--loops`). [Details](docs/usage.md#models-and-downloads)
Memory: 16 GB for the 4B models, 32 GB for the 27B, 35B, Lightning and Bolt, 96 GB for Super-VL, 128 GB for
Flash-Next.</sub>

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-dark.svg">
  <img alt="Kapitsa's architecture: the APIs, the engine with its conversation slots, prefill on the GPU and the Neural Engine, draft-and-verify decoding, the Qwen and Nemotron backends and their Metal kernels." src="docs/assets/architecture-light.svg">
</picture>

<sub>[Architecture](docs/architecture.md) · [Contributing](CONTRIBUTING.md) · [Press kit](docs/assets/brand/README.md) ·
Apache 2.0, © 2026 [Nace AI](https://nace.ai) · [Third-party notices](THIRD_PARTY_NOTICES)</sub>
