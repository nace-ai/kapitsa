# Engine architecture

Kapitsa is a native C++20 inference engine for Apple silicon. Its HTTP APIs share one chat engine; model-specific
backends encode work into Metal command buffers and can offload selected prefill operations to Apple's Neural Engine.
It does not use the MLX or llama.cpp runtime. Compatible checkpoint formats do not imply a runtime dependency.

![Kapitsa request flow: APIs feed a shared chat engine, which selects the Qwen or Nemotron backend. Both use Metal; selected prefill work can use the Neural Engine.](assets/architecture.svg)

Solid arrows show the main execution path. Dashed arrows show optional prefill work on the Neural Engine.
Responses stream back through the shared engine and the API adapter. Model loading and optional image processing
are described below; they are omitted from the overview to keep the request path readable.

## From a request to generated tokens

1. **Normalize the API request.** The server exposes OpenAI Chat Completions, OpenAI Responses and Anthropic
   Messages. Adapters translate these into `ChatRequest`; output adapters return each protocol's streaming events,
   reasoning and tool calls. The built-in chat page uses the same server.
2. **Build the prompt and recover conversation state.** The engine renders the model's Jinja chat template,
   tokenizes the prompt, chooses a conversation slot and reuses a matching prefix. A slot holds attention KV caches,
   recurrent state and checkpoints. Only the uncached suffix needs prefill when a compatible checkpoint exists.
3. **Prefill the target model.** The selected backend processes prompt chunks, using its own attention, recurrent
   and feed-forward kernels. Models with a vision tower can insert image embeddings into those prompt rows.
4. **Draft and verify.** Qwen uses DFlash2 when available, otherwise its MTP head where supported. Bolt uses
   LiLiCorr. The target verifies proposed tokens; accepted tokens advance the sequence, and rejected proposals do
   not become committed recurrent history. Plain generation is available with `--spec 0`.
5. **Stream and retain state.** The output parser separates content, reasoning and complete tool calls. The engine
   retains the slot for follow-ups and can prefill the rendered assistant history while idle.

The engine serializes generation through a mutex. `--slots` keeps multiple conversations warm; it does **not** enable
concurrent model execution or continuous batching. Image fetching and CPU preparation happen before the engine lock.

## Two model backends

| | Qwen3.5 family | Nemotron 3.5 Bolt |
| --- | --- | --- |
| Backend | `Qwen35` | `NemotronH` |
| Recurrent layers | Gated DeltaNet | Mamba-2 |
| Other layers | Attention; dense FFN or MoE | Attention; LatentMoE |
| Default speculation | DFlash2 on 35B/27B; MTP on 4B | LiLiCorr block drafter |
| Checkpoints | MLX-format safetensors; supported GGUF imports | NVIDIA's safetensors checkpoint |
| Weight formats | BF16 and supported affine quantizations | NVFP4, IQ2_XXS, Q8_0 and BF16 by tensor type |
| Neural Engine prefill | Dense FFN portions | Mamba projections and shared-expert work |

Both implement `Model`: memory planning, sequence allocation, KV growth, recurrent checkpoints, prefill and decode.
Speculative paths remain specialized: the engine also holds backend-specific pointers and drafter state. Adding a
new architecture requires more than implementing `Model` if it needs a new drafter, tokenizer behavior or vision tower.

## Where the speed comes from

* **Shape-specific Metal kernels.** Quantized matrix operations, fused projections, attention, recurrent scans and
  grouped expert operations avoid routing all work through general-purpose tensor operators. The Qwen and Nemotron
  backends have separate kernel sets for their shapes and stored weight formats.
* **Several verified tokens per target pass.** DFlash2, MTP and LiLiCorr amortize target-model work when the drafts
  are accepted. Throughput depends on the model, prompt, sampling settings and acceptance rate.
* **Prefix reuse.** KV caches and recurrent-state checkpoints avoid rereading an entire conversation for each turn.
  Image embeddings have their own content-aware cache, so a changed image cannot reuse another image's prompt state.
* **GPU and Neural Engine overlap.** Eligible prefill operations split across both processors. GPU work and Neural
  Engine jobs synchronize through shared events. The split uses additional quantization and is a numerical tradeoff;
  it is distinct from speculative verification. Runtime checks and watchdogs can disable the split and rerun a
  failed chunk on the GPU.

These are implementation mechanisms, not guarantees of a fixed speedup. Measured results and their conditions live
in the [benchmark reports](benchmarks.md), [M5 Max report](benchmarks-m5-max.md) and [optimization log](PERF.md).

## Loading, images and memory

The model catalog pins checkpoint and companion revisions. Downloads resume and validate checksums. The safetensors
loader supplies weights to the backends; Bolt repacks codes and scales into its GPU layout at load time. Supported
Qwen GGUF files are converted once into a reusable model folder.

Vision is optional. `open_encoder` selects `QwenViT` for supported Qwen checkpoints or `RadioViT` for Nemotron
3.5 Bolt's C-RADIO ViT-H/16 tower and projector. The tower loads on the first image request; CPU image preparation
and cached GPU embeddings feed the model's prefill. Qwen uses M-RoPE positions; Bolt wraps image rows in its image
tokens and uses attention without positional embeddings. The current 4B tower shape is unsupported. The text
benchmarks do not measure images.

Memory planning happens before weights load. Per-slot KV caches grow within a budget derived from Metal's recommended
working set or `--mem-budget`. Under pressure, the engine can release cached image data, idle slots' KV caches and
Neural Engine resources. `kapitsa doctor` uses the same planning logic to report what fits.

## Source map

| Component | Entry points |
| --- | --- |
| CLI, HTTP routes and chat page | [`src/server/`](../src/server/) |
| OpenAI Responses and Anthropic adapters | [`src/api/`](../src/api/) |
| Request lifecycle, slots and generation | [`src/engine/engine.cpp`](../src/engine/engine.cpp) |
| Shared model contract and sequence state | [`src/model/model.h`](../src/model/model.h) |
| Qwen backend and drafters | [`qwen35.h`](../src/model/qwen35.h), [`dflash2.h`](../src/model/dflash2.h) |
| Nemotron backend and drafter | [`nemotron_h.h`](../src/model/nemotron_h.h), [`lilicorr.h`](../src/model/lilicorr.h) |
| Metal runtime and kernels | [`src/metal/`](../src/metal/), [`src/kernels/`](../src/kernels/) |
| Neural Engine bridge and splits | [`src/ane/`](../src/ane/), [`nemotron_ane.h`](../src/model/nemotron_ane.h) |
| Image encoders and embedding cache | [`src/vision/`](../src/vision/) |
| Catalog, downloads and weights | [`src/hub/`](../src/hub/), [`src/loader/`](../src/loader/) |

The Neural Engine bridge uses Apple's private framework interfaces. See [third-party notices](../THIRD_PARTY_NOTICES)
for adapted code and design credits, and [DEVELOPMENT.md](../DEVELOPMENT.md) for switches and implementation details.

## Updating the diagram

The SVG is rendered from [architecture.mmd](assets/architecture.mmd), with
[mermaid-config.json](assets/mermaid-config.json). With Mermaid CLI 11.9.0 and its Chrome dependency installed:

```bash
npx --yes @mermaid-js/mermaid-cli@11.9.0 \
  -i docs/assets/architecture.mmd -o docs/assets/architecture.svg \
  -c docs/assets/mermaid-config.json -b white -w 1400
```
