# nace-infer

**The fastest way to run Qwen on a Mac.** A from-scratch inference engine for Apple silicon.

nace-infer runs Qwen3.5, 3.6 and 3.8 (dense and mixture-of-experts) on one Mac behind an OpenAI-compatible
server, for coding agents and chat apps. It is written in C++ and Metal for one job: the hybrid Gated
DeltaNet + attention architecture these models use. It decodes several tokens per step with speculative
decoding, prefills dense models on the GPU and the Neural Engine together, and picks a conversation back up
without re-reading what it has already seen. No MLX, no llama.cpp, no Xcode.

<!-- Numbers below are preliminary (2026-10-16) and get replaced by the final fair-harness run. -->

## Quick start

Apple silicon Mac, macOS 15 or later (tested on an M4 Pro with macOS 15.3 and 27.0), CMake and the Xcode
Command Line Tools. The 4-bit 35B and 27B models need 48 GB of unified memory (about 20 and 16 GB of weights).

```bash
git clone https://github.com/nace-ai/nace-infer && cd nace-infer
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
hf download Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed --local-dir ~/models/qwen3.6-35b-a3b-4bit
./build/nace-infer-server --model ~/models/qwen3.6-35b-a3b-4bit --model-id qwen3.6-35b-a3b
```

The server listens on `127.0.0.1:8000`. Kernels compile at load in a few seconds; the first request is
then as fast as every later one.

## Use the API

OpenAI Chat Completions with streaming, reasoning (`reasoning_content`) and tool calls:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [{"role": "user", "content": "Explain speculative decoding in one sentence."}],
    "reasoning_effort": "none"
  }'
```

`"reasoning_effort": "none"` or `"chat_template_kwargs": {"enable_thinking": false}` turns thinking off.
Point any OpenAI-compatible agent at `http://127.0.0.1:8000/v1`.

## Models

| Model | Weights (MLX affine 4-bit, group 64) | Speculative decoding |
| --- | --- | --- |
| Qwen3.6-35B-A3B (MoE, 256 experts) | [`Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed`](https://huggingface.co/Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed) (body = `mlx-community/Qwen3.6-35B-A3B-4bit`, plus its MTP head) | MTP head, depth 1; DFlash2 drafter (in progress) |
| Qwen3.8-27B (dense) | [`lmstudio-community/Qwen3.8-27B-MLX-4bit`](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-4bit) + `mtp.safetensors` from [`Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed`](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed) | MTP head, depth 2; DFlash2 drafter (in progress) |

Other Qwen3.5-family checkpoints in the same MLX format are untested; the defaults are tuned on these two.
Decoding is exact: with speculation on, greedy output is token-for-token the same as without it.

## Settings

| Option | Purpose |
| --- | --- |
| `--ctx 32768` | Context per conversation slot. |
| `--slots 2` | Conversations kept warm for instant follow-ups. |
| `--spec N` | Draft depth (default per model; `0` turns speculation off). |
| `--api-key KEY` | Require a bearer token. |
| `--host`, `--port` | Listen address. |

Everything else, including the switches behind each optimization, is in [DEVELOPMENT.md](DEVELOPMENT.md).

## Performance

Measured on one MacBook Pro (M4 Pro, 20-core GPU, 48 GB, macOS 27.0.1) against every engine we could get
running on the same machine, at its latest release and its own best settings, on the same 4-bit weights
wherever the engine accepts them. In brackets: the fastest other engine measured.

| Metric | Qwen3.6-35B-A3B | Qwen3.8-27B |
| --- | ---: | ---: |
| Decode · coding prompt, greedy | 116-124 tok/s (Splash 125-132) | 29-30 tok/s (Splash 55) |
| Prefill · 4K prompt | 960-1,050 tok/s (Splash 1,011-1,071) | **277-321 tok/s** (Splash 203-232) |
| Prefill · 16K prompt | 832 tok/s | **247-265 tok/s** (Splash 209) |
| Cached time to first token · 4K / 16K prefix | **0.12 / 0.16 s** (llama.cpp 0.18 / 0.20) | **0.25 / 0.28 s** (Ollama 0.50 / 0.58) |

Against the engines most people use, on the same Mac: 35B decode 1.5-1.7x Ollama and llama.cpp (each with
its MTP speculation on); 27B prefill about 3x mlx-lm, Ollama and llama.cpp.

Engines in the comparison: Splash, MTPLX, oMLX, mlx-lm, Ollama (GGUF and MLX runners), llama.cpp (with and without MTP),
TensorFold, mlx-dspark, dflash-mlx, Darkbloom, lithos-metal, Lily, Tatara, Pulsar, AX Engine, Rapid-MLX,
vpipe, mlx-serve, scratchy, PureDrive. Method: the same prompts, a GPU lock so nothing else runs during a
measurement, median of 5 runs, thinking off, identical sampling settings, cold prompts cache-busted.

## Design

Each model gets kernels for its exact shapes: quantized matrix kernels that decode weights straight into the
GPU's matrix registers, a token-parallel DeltaNet scan, flash-style attention, fused mixture-of-experts GEMMs,
and verify kernels that check several drafted tokens for the price of one. Speculative decoding drafts with the
model's own MTP head and verifies the draft in one pass; DeltaNet's recurrent state rolls back through a tape,
so a rejected draft costs nothing. On dense models, prefill splits every feed-forward layer between the GPU
and the Neural Engine, which run side by side (chunks of 512 tokens and more). Conversations keep their KV cache and a recurrent-state
checkpoint at each turn boundary, so a follow-up only processes the new message. Weights load straight from
the model's safetensors into aligned GPU buffers, with no converted copy on disk.

## More

- [DEVELOPMENT.md](DEVELOPMENT.md): build, server options, internals, switches, benchmarks.
- [docs/PERF.md](docs/PERF.md): every optimization with its measurements.
- Third-party code: nlohmann/json, minja and cpp-httplib (MIT). Model weights keep their own licenses.
