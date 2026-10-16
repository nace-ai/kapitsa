# nace-infer

A from-scratch inference engine for Apple silicon, built for the Qwen3.5/3.6/3.8 hybrid models (Gated
DeltaNet + gated attention, dense or MoE) that Nace Agent runs locally. C++/Objective-C++ on raw Metal; no MLX,
no llama.cpp. Kernels are embedded as source and compiled at runtime (works with Command Line Tools only).

## Build and run
```
cmake -S . -B build && cmake --build build -j
./build/nace-infer-server --model <MLX model dir> --port 8090 --spec 1
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
