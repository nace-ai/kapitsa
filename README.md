# Kapitsa

**Fast local Qwen for Apple silicon.** Kapitsa by [Nace AI](https://nace.ai).

Kapitsa runs Qwen3.6, Qwen3.8 and Qwen3.5 on one Mac behind an OpenAI-compatible server, for coding agents
and chat apps, with a built-in chat page. It is a C++ and Metal engine written for one architecture, the hybrid
Gated DeltaNet + attention design of the Qwen3.5 family. It drafts several tokens per step with DFlash2 and MTP
speculative decoding, prefills dense models on the GPU and the Neural Engine together, and picks a conversation
back up without reading it again. On a 48 GB MacBook Pro it is the fastest of the engines we compared it with
(see [Performance](#performance)). No Python, MLX, llama.cpp or Xcode needed.

## Quick start

An Apple silicon Mac with macOS 15 or later (tested on an M4 Pro), the Xcode Command Line Tools and CMake. The 4B
runs on a 16 GB MacBook Air. The 35B-A3B and 27B need 32 GB, and 48 GB is recommended. `kapitsa doctor` shows what
fits this Mac.

```bash
git clone https://github.com/nace-ai/kapitsa && cd kapitsa && ./install.sh
kapitsa serve --model qwen3.6-35b
```

`./install.sh` builds Kapitsa in a minute or two and puts `kapitsa` in `/opt/homebrew/bin` or `~/.local/bin`. It
offers to install CMake with Homebrew, and tells you if the folder needs adding to your PATH.

The first run of `kapitsa serve` downloads the model and its DFlash2 draft model (22 GB) and prints `Ready`. Later
starts reuse the download, offline too. Without `--model`, Kapitsa picks the best model that fits this Mac.

Once it prints `Ready`, open <http://127.0.0.1:8000> to chat, or point a coding agent or any OpenAI-compatible app
at `http://127.0.0.1:8000/v1`. Press Ctrl+C to stop.

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

Coding agents and apps that speak the OpenAI API connect to `http://127.0.0.1:8000/v1`; any API key works unless
the server was started with `--api-key`. Thinking follows the model's default, which is on. `"reasoning_effort": "none"`
or `"chat_template_kwargs": {"enable_thinking": false}` turns it off. Every response reports its `usage`, including
the cached prompt tokens of a follow-up.
[API details](DEVELOPMENT.md#server)

## Models

| Model | `--model` | Variants | Download | Speed-up included |
| --- | --- | --- | ---: | --- |
| Qwen3.6-35B-A3B (mixture of experts, 3B active) | `qwen3.6-35b` or `35b` | `4bit` (default) | 22.1 GB | DFlash2 draft model, MTP head |
| Qwen3.8-27B (dense) | `qwen3.8-27b` or `27b` | `4bit` (default) | 20.2 GB | DFlash2 draft model, MTP head |
| Qwen3.5-4B (dense) | `qwen3.5-4b` or `4b` | `4bit` (default) | 3.3 GB | MTP head |

A variant is the precision of the weights: `--model 27b` takes the model's default, and `--model 27b:<variant>`
picks another as more are added (`kapitsa list` shows them all). Each variant is pinned to a tested revision.
Speculative decoding is lossless: the output follows the model's own distribution, and greedy output matches
decoding without it except at near-ties within rounding noise. Other Qwen3.5-family models in MLX 4- or 8-bit
format load from a Hugging Face repo (`--model mlx-community/Qwen3.5-9B-4bit`) or a folder
(`--model ~/models/my-qwen`). The defaults are tuned on the three above.

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
| `--api-key KEY` | Require a bearer token. The chat page asks for it once. |
| `--host 0.0.0.0` | Serve your network. Default: this Mac only (`127.0.0.1:8000`; `--port` changes the port). |

Models are stored in `~/.cache/kapitsa/models`, and `KAPITSA_HOME` moves them. `HF_TOKEN` is used for downloads
when set. `kapitsa serve --help` lists every option, and [DEVELOPMENT.md](DEVELOPMENT.md) has the switches behind each
optimization.

## Performance

Measured on a MacBook Pro (M4 Pro, 20-core GPU, 48 GB) with macOS 27.0.1: the same prompts for every engine, each at
its latest release and its own best settings, the same 4-bit weights wherever the engine accepts them (GGUF engines
on Q4_K_M), thinking off, 32K context, one slot, median of 5 runs, nothing else on the GPU. Ratios are against the
next-fastest engine measured.

<!-- 27B ratios re-measured in the confirmation run -->

| Metric | Qwen3.6-35B-A3B | Qwen3.8-27B |
| --- | ---: | ---: |
| Decode · coding prompt, temperature 0.6 | 154 tok/s (1.2×) | 62 tok/s (2.0×) |
| Decode · greedy | 179 tok/s (1.4×) | 66 tok/s (1.9×) |
| Prefill · 4K prompt | 1,169 tok/s (1.1×) | 395 tok/s (3.1×) |
| Prefill · 16K prompt | 986 tok/s (1.1×) | 330 tok/s (2.7×) |
| Cached time to first token · 4K / 16K | 0.06 / 0.09 s (1.7×) | 0.23 / 0.25 s (1.4× / 1.6×) |
| Peak memory | 21.1 GB (lowest) | 20.5 GB (lowest) |

Against the engines people use, decode at temperature 0.6 in tok/s, each engine in its fastest setup:

| Engine | 35B-A3B | 27B |
| --- | ---: | ---: |
| llama.cpp (+ MTP) | 69 | 18 |
| Ollama (MLX runner, MTP) | 89 | 31 |
| LM Studio (MLX engine) | 86 | 15 |
| MTPLX | 80 | 30 |
| oMLX + DFlash2 | 127 | 30 |
| **Kapitsa** | **154** | **62** |

Prefill is up to 3× faster on the 27B (GPU + Neural Engine), and a follow-up turn starts in 0.06-0.25 s. Every
engine passes the same tool-call checks.
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
