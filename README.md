# Kapitsa

**Fast local Qwen for Apple silicon.**

Kapitsa runs Qwen3.6, Qwen3.8 and Qwen3.5 on one Mac behind an OpenAI-compatible server, for coding agents
and chat apps, with a built-in chat page. It is a C++ and Metal engine written for one architecture, the hybrid
Gated DeltaNet + attention design of the Qwen3.5 family. It drafts several tokens per step with DFlash2 and MTP
speculative decoding, prefills dense models on the GPU and the Neural Engine together, and picks a conversation
back up without reading it again. On a 48 GB MacBook Pro it is the fastest of the engines we compared it with
(see [Performance](#performance)). No Python, MLX, llama.cpp or Xcode needed.

## Quick start

An Apple silicon Mac with macOS 15 or later (tested on an M4 Pro), the Xcode Command Line Tools and CMake. The 4B
runs on 16 GB. The 35B-A3B and 27B need 32 GB, and 48 GB is recommended.

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

Thinking follows the model's default, which is on. `"reasoning_effort": "none"` or
`"chat_template_kwargs": {"enable_thinking": false}` turns it off. Every response reports its `usage`, including the
cached prompt tokens of a follow-up.
[API details](DEVELOPMENT.md#server)

## Models

| Model | `--model` | Download | Speed-up included |
| --- | --- | ---: | --- |
| Qwen3.6-35B-A3B (mixture of experts, 3B active) | `qwen3.6-35b` or `35b` | 22.1 GB | DFlash2 draft model, MTP head |
| Qwen3.8-27B (dense) | `qwen3.8-27b` or `27b` | 20.2 GB | DFlash2 draft model, MTP head |
| Qwen3.5-4B (dense) | `qwen3.5-4b` or `4b` | 3.3 GB | MTP head |

All three are 4-bit MLX weights at pinned, tested revisions. Speculative decoding is lossless: the output follows
the model's own distribution, and greedy output matches decoding without it except at near-ties within rounding
noise. Other Qwen3.5-family models in MLX 4- or 8-bit format load from a Hugging Face repo
(`--model mlx-community/Qwen3.5-9B-4bit`) or a folder (`--model ~/models/my-qwen`). The defaults are tuned on the
three above.

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
| `--api-key KEY` | Require a bearer token. The chat page asks for it once. |
| `--host 0.0.0.0` | Serve your network. Default: this Mac only (`127.0.0.1:8000`; `--port` changes the port). |

Models are stored in `~/.cache/kapitsa/models`, and `KAPITSA_HOME` moves them. `HF_TOKEN` is used for downloads
when set. `kapitsa serve --help` lists every option, and [DEVELOPMENT.md](DEVELOPMENT.md) has the switches behind each
optimization.

## Performance

<!-- TBD cells: the final head-to-head on the M4 Pro fills them (research/inference/scripts in the Nace Agent repo). -->

Measured on a MacBook Pro (M4 Pro, 20-core GPU, 48 GB), with the same coding prompts over HTTP for every engine.
Ratios compare with the next-fastest engine measured.

| Metric | Qwen3.6-35B-A3B | Qwen3.8-27B |
| --- | ---: | ---: |
| Writing speed (decode) | TBD tok/s (TBD×) | TBD tok/s (TBD×) |
| Reading speed (prefill) · 4K-token prompt | TBD tok/s (TBD×) | TBD tok/s (TBD×) |
| Reading speed (prefill) · 16K-token prompt | TBD tok/s (TBD×) | TBD tok/s (TBD×) |
| Cached follow-up · time to first token, 4K / 16K | TBD / TBD s (TBD×) | TBD / TBD s (TBD×) |

The engines people run on Macs, each at its latest release with its own best settings (speculative decoding where
it has it), on the same 4-bit weights where it accepts them. Decode and prefill are in tokens per second, follow-ups
in seconds to the first token:

| Qwen3.6-35B-A3B | Decode | Prefill 4K | Prefill 16K | Follow-up 4K | Follow-up 16K |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Kapitsa** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| llama.cpp | TBD | TBD | TBD | TBD | TBD |
| Ollama | TBD | TBD | TBD | TBD | TBD |
| LM Studio | TBD | TBD | TBD | TBD | TBD |
| mlx-lm | TBD | TBD | TBD | TBD | TBD |
| oMLX | TBD | TBD | TBD | TBD | TBD |
| MTPLX | TBD | TBD | TBD | TBD | TBD |

| Qwen3.8-27B | Decode | Prefill 4K | Prefill 16K | Follow-up 4K | Follow-up 16K |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Kapitsa** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| llama.cpp | TBD | TBD | TBD | TBD | TBD |
| Ollama | TBD | TBD | TBD | TBD | TBD |
| LM Studio | TBD | TBD | TBD | TBD | TBD |
| mlx-lm | TBD | TBD | TBD | TBD | TBD |
| oMLX | TBD | TBD | TBD | TBD | TBD |
| MTPLX | TBD | TBD | TBD | TBD | TBD |

One engine runs at a time on an otherwise idle Mac, with thinking off and the same sampling settings. Each figure is
the median of 5 runs, and cold prompts are cache-busted. A follow-up is the next turn of a conversation that already
holds 4K or 16K tokens.
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
- Third-party code: nlohmann/json, minja and cpp-httplib (MIT). Model weights keep their own licenses.
