# Kapitsa

**Fast local Qwen for Apple silicon.** Kapitsa is an inference engine by [Nace.AI](https://nace.ai).

Kapitsa runs Qwen3.6, Qwen3.8 and Qwen3.5 on one Mac behind an OpenAI-compatible server, for coding agents
and chat apps, with a built-in chat page. It is a C++ and Metal engine written for one architecture, the hybrid
Gated DeltaNet + attention design of the Qwen3.5 family. It drafts several tokens per step with DFlash2 and MTP
speculative decoding, prefills dense models on the GPU and the Neural Engine together, and picks a conversation
back up without reading it again. On a 48 GB MacBook Pro it is the fastest of the engines we compared it with
(see [Performance](#performance)). No Python, MLX, llama.cpp or Xcode needed.

## Quick start

An Apple silicon Mac with macOS 15 or later (tested on M4 Pro and M5 Max). The 4B runs on a 16 GB MacBook Air. The
35B-A3B and 27B run on 32 GB Macs with shorter conversations and without the DFlash2 draft model; 48 GB is recommended.
`kapitsa doctor` shows what fits this Mac.

```bash
brew install nace-ai/tap/kapitsa
kapitsa serve --model qwen3.6-35b
```

`brew services start kapitsa` runs `kapitsa serve` in the background instead, now and at every login (log:
`/opt/homebrew/var/log/kapitsa.log`).

No Homebrew? `curl -fsSL https://raw.githubusercontent.com/nace-ai/kapitsa/main/install.sh | bash` installs the same
prebuilt binary.

From source, with the Xcode Command Line Tools and CMake:
`git clone https://github.com/nace-ai/kapitsa && cd kapitsa && ./install.sh`.

The first run of `kapitsa serve` downloads the model and its DFlash2 draft model (22 GB) and prints `Ready`. Later
starts reuse the download, offline too. Without `--model`, Kapitsa picks the best model that fits this Mac.

Once it prints `Ready`, open <http://127.0.0.1:8000> to chat, or point a coding agent or any OpenAI-compatible app
at `http://127.0.0.1:8000/v1`. Press Ctrl+C to stop.

## Use it with Claude Code, Codex, opencode

```bash
kapitsa claude      # Claude Code
kapitsa codex       # Codex
kapitsa opencode    # opencode
```

Each command starts the agent on Kapitsa. It uses the server already running on port 8000, or starts one for the
session with the largest context up to 128K tokens that the Mac's memory holds, and stops it when the agent exits.
Arguments go to the agent, for example `kapitsa claude -p "what does this repo do?"` or `kapitsa codex exec "fix the
failing test"`. `--model 27b` picks the model and `--port` another server. The agent itself must be installed.

## Use the API

OpenAI Chat Completions, with streaming, reasoning (`reasoning_content`) and tool calls:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [{"role": "user", "content": "Explain speculative decoding in one sentence."}]
  }'
```

Images work on Qwen3.6 and Qwen3.8: send OpenAI `image_url` parts, Anthropic image blocks or Responses `input_image`,
as base64 data or an `http(s)` URL.

Coding agents and apps that speak the OpenAI API connect to `http://127.0.0.1:8000/v1`; any API key works unless
the server was started with `--api-key`. Thinking follows the model's default, which is on. `"reasoning_effort": "none"`
or `"chat_template_kwargs": {"enable_thinking": false}` turns it off. Every response reports its `usage`, including
the cached prompt tokens of a follow-up.

The same server speaks the OpenAI Responses API (`POST /v1/responses`) and the Anthropic Messages API
(`POST /v1/messages`, at base URL `http://127.0.0.1:8000`), with streaming, reasoning and tool calls.
[API details](DEVELOPMENT.md#server)

## Models

| Model | `--model` | Variants | Download | Speed-up included |
| --- | --- | --- | ---: | --- |
| Qwen3.6-35B-A3B (mixture of experts, 3B active) | `qwen3.6-35b` or `35b` | `4bit` (default), `5bit`, `6bit`, `8bit`, `bf16` | 22.1 GB | DFlash2 draft model, MTP head |
| Qwen3.8-27B (dense) | `qwen3.8-27b` or `27b` | `4bit` (default), `5bit`, `6bit`, `8bit`, `bf16` | 20.2 GB | DFlash2 draft model, MTP head |
| Qwen3.5-4B (dense) | `qwen3.5-4b` or `4b` | `4bit` (default), `6bit`, `8bit`, `bf16` | 3.3 GB | MTP head |
| Nemotron 3.5 Bolt 74B-A9B (hybrid Mamba-2 mixture of experts, 9B active) | `nemotron-3.5-bolt` or `bolt` | `24gb` (NVIDIA's own 24 GB checkpoint) | 25.1 GB | NVIDIA's LiLiCorr draft model |

A variant is the precision of the weights: `--model 35b` takes the default, `--model 35b:6bit` another. The default
is the fastest; for the best quality take `6bit`: its answers stay closer to the full-precision model (about 96% of
next tokens agree with bf16, against 88% at 4-bit; a fifth of the divergence) and it decodes 25-30% slower. `bf16`
runs unquantized where it fits (the 4B; the 27B and 35B in bf16 need a 96 GB Mac). Every variant uses the model's
draft model and MTP head. The speeds under [Performance](#performance) are the 4-bit variants.
[Quality and speed of every variant](docs/PERF.md#weight-formats-and-default-precisions)

Each variant is pinned to a tested revision. Speculative decoding is lossless: the output follows the model's own
distribution, and greedy output matches decoding without it except at near-ties within rounding noise. Other
Qwen3.5-family models load from a Hugging Face repo (`--model mlx-community/Qwen3.5-9B-6bit`) or a folder (`--model
~/models/my-qwen`) in MLX format, bf16 or MLX's 3- to 8-bit quantizations. The defaults are tuned on the three above.
A GGUF file of a Qwen3.5-family model (LM Studio, Ollama, llama.cpp, Hugging Face GGUF repos) runs too: `kapitsa serve
--model ~/Downloads/Qwen3.5-4B-Q4_K_M.gguf` converts it once into a model folder, and `kapitsa import FILE.gguf` converts
it without starting it.

`kapitsa pull 27b` downloads a model without starting it, `kapitsa list` shows what is downloaded, and
`kapitsa rm 27b` deletes it. Downloads resume after an interruption and are checked against their checksums. Files
already on the Mac are reused instead of downloaded again: the Hugging Face cache, Nace Agent's models, or a folder you
name with `--from`.
[Model storage](DEVELOPMENT.md#models-and-downloads)

## Settings

Add these to `kapitsa serve`:

| Option | Purpose |
| --- | --- |
| `--ctx 32768` | Set the context per conversation. |
| `--slots 2` | Keep this many conversations ready for instant follow-ups. |
| `--dflash none` | Decode with the model's MTP head instead of its DFlash2 draft model. |
| `--spec 0` | Turn speculative decoding off. |
| `--mem-budget 20` | Cap the GPU memory Kapitsa holds, in GB. Default: what macOS recommends for this Mac's GPU, a little less. |
| `--api-key KEY` | Require an API key (a bearer token or `x-api-key`). The chat page asks for it once. |
| `--host 0.0.0.0` | Serve your network. Default: this Mac only (`127.0.0.1:8000`; `--port` changes the port). |

Models are stored in `~/.cache/kapitsa/models`, and `KAPITSA_HOME` moves them. `HF_TOKEN` is used for downloads
when set. `kapitsa serve --help` lists every option, and [DEVELOPMENT.md](DEVELOPMENT.md) has the switches behind each
optimization.

## Performance

Kapitsa on two MacBook Pros, writing speed (decode at temperature 0.6):

| Model | MacBook Pro (2024) · M4 Pro · 48 GB | MacBook Pro · M5 Max · 128 GB |
| --- | ---: | ---: |
| Qwen3.6 35B-A3B | 154 tok/s | 250 tok/s |
| Qwen3.8 27B | 63 tok/s | 73 tok/s |
| Nemotron 3.5 Bolt 74B-A9B | 64 tok/s | not measured yet |

Reading a 4K prompt: 1,169 / 389 / 436 tok/s on the M4 Pro, 2,224 / 430 tok/s on the M5 Max. The M5 Max ran with
desktop apps open (the harness flagged their GPU use), so its numbers are on the low side
([M5 Max report](docs/benchmarks-m5-max.md)). Models that need a 96 GB+ Mac are next on the M5 Max.

**Against the other engines** on the MacBook Pro (2024) · M4 Pro · 48 GB, writing speed with each engine in its fastest
setup:

| Engine | Qwen3.6 35B-A3B | Qwen3.8 27B |
| --- | ---: | ---: |
| llama.cpp (+ MTP) | 69 tok/s | 18 tok/s |
| Ollama (MLX runner, MTP) | 89 tok/s | 32 tok/s |
| LM Studio (MLX engine) | 86 tok/s | 15 tok/s |
| MTPLX | 80 tok/s | 36 tok/s |
| oMLX + DFlash2 | 127 tok/s | 45 tok/s |
| **Kapitsa** | **154 tok/s** | **63 tok/s** |

Kapitsa also reads prompts and answers follow-ups faster, with less memory. On the same Mac, against the next-fastest
engine:

| Metric | Qwen3.6 35B-A3B | Qwen3.8 27B |
| --- | ---: | ---: |
| Decode · coding prompt, temperature 0.6 | 154 tok/s (1.2×) | 63 tok/s (1.4×) |
| Decode · greedy | 179 tok/s (1.4×) | 67 tok/s (1.5×) |
| Prefill · 4K prompt | 1,169 tok/s (1.1×) | 389 tok/s (2.9×) |
| Prefill · 16K prompt | 986 tok/s (1.1×) | 335 tok/s (2.6×) |
| Cached time to first token · 4K / 16K | 0.06 / 0.09 s (1.6× / 1.7×) | 0.28 / 0.27 s (1.2× / 1.5×) |
| Peak memory | 21.1 GB (lowest) | 19.4 GB (lowest) |

**Nemotron 3.5 Bolt 74B-A9B**, NVIDIA's own 24 GB checkpoint, against the engines that load it on a Mac (one
session, back to back; Kapitsa with NVIDIA's LiLiCorr drafter and the GPU and Neural Engine reading prompts together):

| Metric | Kapitsa | Next best |
| --- | ---: | --- |
| Decode · coding prompt, temperature 0.6 | 64 tok/s (1.9×) | oMLX, 34 (its drafter fails when sampling) |
| Decode · greedy | 67 tok/s (1.3×) | oMLX + LiLiCorr, 53 |
| Prefill · 4K / 16K prompt | 436 / 385 tok/s (1.5× / 1.3×) | mlx-vlm 300, oMLX 300 |
| Cached time to first token · 4K / 16K | 0.28 / 0.32 s (2.6× / 2.3×) | oMLX, 0.72 / 0.72 s |
| Peak memory | 22.7 GB (lowest) | oMLX, 30.3 GB |

How it was measured: MacBook Pro (2024), M4 Pro with a 20-core GPU, 48 GB, macOS 27.0.1. The same prompts for every
engine, each at its latest release and its own best settings, the same 4-bit weights wherever the engine accepts them
(GGUF engines on Q4_K_M), thinking off, 32K context, one slot, median of 5 runs, nothing else on the GPU. Ratios are
against the next-fastest engine measured, which ran in the same session as Kapitsa. Every engine passes the same
tool-call checks. For Bolt every engine read NVIDIA's checkpoint (the MLX engines through the MLX bundle's model file),
and the desktop's own GPU work (10-40% of the GPU while the session ran) was there for every engine alike.
[Measurement details](docs/benchmarks.md) ·
[Run benchmarks locally](bench/harness/README.md) ·
[Every optimization and its measurements](docs/PERF.md)


## Design

Each model gets Metal kernels for its exact shapes. Quantized matrix kernels decode weights straight into the GPU's
matrix registers. The DeltaNet scan runs token-parallel, attention is flash-style, and the mixture-of-experts layers
use fused grouped GEMMs. Speculative decoding drafts a block of tokens with the model's DFlash2 draft model, or its
own MTP head, and verifies them in one pass. DeltaNet's recurrent state rolls back through a tape, so a rejected
draft costs nothing. On dense models, prefill splits each feed-forward layer between the GPU and the Neural Engine.
Each conversation keeps its KV cache and a recurrent-state checkpoint at every turn boundary, so a follow-up
processes only the new message. Weights load from the model's safetensors straight into GPU buffers, with no
converted copy on disk.

## More

- [DEVELOPMENT.md](DEVELOPMENT.md): building from source, the server, model downloads, internals, switches,
  benchmarks.
- [docs/PERF.md](docs/PERF.md): every optimization with its measurements.
- [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

Kapitsa is licensed under the [Apache License 2.0](LICENSE), Copyright 2026 Nace AI. It includes cpp-httplib, minja
and nlohmann/json (MIT) and Unicode data ([third-party notices](THIRD_PARTY_NOTICES)). Model weights keep their own
licenses.
