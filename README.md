# nace-infer

A from-scratch inference engine for Apple silicon, built for the Qwen3.5/3.6/3.8 hybrid models (Gated
DeltaNet + gated attention, dense or MoE) that Nace Agent runs locally. C++/Objective-C++ on raw Metal; no MLX,
no llama.cpp. Kernels are embedded as source and compiled at runtime (works with Command Line Tools only).

## Build and run
```
cmake -S . -B build && cmake --build build -j
./build/nace-infer-server --model <MLX model dir> --port 8090   # --spec N overrides the per-model MTP depth
```
OpenAI-compatible: `/v1/chat/completions` (SSE and JSON, `reasoning_content`, tool calls), `/v1/models`, `/health`.
Options: `--ctx` (per-slot context), `--slots` (conversations kept warm), `--spec` (MTP draft depth, 0 = off),
`--api-key`, `--host`.

## What is inside
- `src/kernels/*.metal`: quantized matvec/GEMM (MLX affine 4/8-bit), Gated DeltaNet (decode, verify with a rollback
  tape), attention (split-K decode, flash prefill), MoE (router top-k, fused SwiGLU experts, grouped prefill GEMM),
  on-GPU sampling.
- `src/model/qwen35*.cpp`: the model (decode, batched prefill, MTP speculative decoding).
- `src/engine/`: chat templates (minja), prefix reuse across turns per conversation slot, streaming parser for
  reasoning / content / Qwen XML tool calls.
- `src/tokenizer/`: byte-level BPE identical to HF tokenizers.
- `tools/`: `ni_run` (raw generation, `--spec N`, `NI_PROFILE=1` per-kernel GPU profile), kernel benchmarks.
- `tests/test_server.py`: end-to-end API checks.

Performance notes and history: `docs/PERF.md`. Third-party (MIT): nlohmann/json, minja, cpp-httplib.

## GPU + Neural Engine FFN split (opt-in, `NI_ANE=1`; dense 27B prefill)
`src/ane/`, `src/kernels/ane_ffn.metal`, `src/model/qwen35_ane.cpp`; design in
`research/inference/ane/ANE_DESIGN.md`, measurements in `docs/PERF.md`. For prefill chunks of >= 512 rows each
dense FFN layer is split by intermediate channel: the GPU keeps the leading channels (its 4-bit GEMMs), the
Neural Engine computes the rest as one W8A8 MIL program (Hadamard-rotated int8 activations, per-row int8
weights) through Apple's private `AppleNeuralEngine.framework`, and a GPU join adds its partial down projection
in fp32. Weights are not stored twice: each chunk re-quantizes the ANE's slice one layer ahead from the 4-bit
planes into double-buffered IOSurfaces (~450 MB Metal); only the per-row scales (4 MB) are cached, in
`~/.cache/nace-infer-models/ane-cache/`. GPU and ANE are ordered by shared events with no CPU in the loop; a
watchdog, a startup check and a GPU rerun of a failed chunk keep any failure from being fatal. Ideas from
Splash (incoai/splash, Apache-2.0); our code.
- `NI_ANE_UNITS` (ANE share in 512-channel units; default: the `ane_calib` result, else 25 on an M4 Pro),
  `NI_ANE_MIN_ROWS`, `NI_ANE_FNS`, `NI_ANE_WAVE=1` (two-half wavefront: the ANE's FFN of one half runs under
  the GPU's mixer of the other), `NI_ANE_VERIFY=0`, `NI_ANE_FAULT=<layer>` (exercises the rerun path).
- `tools/ane_calib <model> [--write]` times split layers at several shares and stores the best for this Mac;
  `tools/ane_probe` runs one program on random weights against a CPU reference.
- Not usable under Metal API validation (the ANE needs the shared events' Mach ports). The first start
  compiles the program (~30 s); the ANE daemon caches it, but can evict it.
