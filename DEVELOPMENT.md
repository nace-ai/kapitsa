# Development

Building Kapitsa, internals, tuning switches and tools. For what the engine is and how fast it runs, see
[README.md](README.md); measured history lives in [docs/PERF.md](docs/PERF.md).

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

Needs CMake and the Xcode Command Line Tools only. Metal kernels are embedded as source and compiled at
runtime by the OS (`newLibraryWithSource`), so no offline `metal` compiler or Xcode is required. Pipelines are
created at load (`NI_PREWARM=0` skips that and creates them on first use).

## Server

```bash
./build/kapitsa serve --model <model dir> --model-id <id>   # build/nace-infer-server is an alias
```

| Option | Purpose |
| --- | --- |
| `--model <dir>` | MLX-format model directory (affine 4/8-bit; the MTP head may live in `mtp.safetensors`). Required. |
| `--model-id <id>` | Name reported by `/v1/models`. |
| `--host`, `--port` | Listen address (default `127.0.0.1:8000`). |
| `--ctx <tokens>` | Context per conversation slot (default 32768). |
| `--slots <n>` | Conversations kept warm for prefix reuse (default 2). |
| `--spec <n>` | Speculative draft depth; default per model (MoE 1, dense 2); `0` turns it off. |
| `--prefill-chunk <rows>` | Prefill chunk size (default 2048). |
| `--api-key <key>` | Require `Authorization: Bearer <key>` (except `/health`). |

Endpoints: `POST /v1/chat/completions` (SSE or JSON; `reasoning_content`; Qwen XML tool calls returned as
OpenAI `tool_calls`; `usage` in the final chunk with `stream_options.include_usage`), `GET /v1/models`,
`GET /health`. Thinking: `chat_template_kwargs.enable_thinking=false` or `reasoning_effort: "none"`.

## Layout

- `src/kernels/*.metal`: quantized matvec/GEMM (MLX affine 4/8-bit; `qmm3.metal` = qmr prefill GEMM with
  weights dequantized into MMA registers, `qmms.metal` = short-prefill GEMMs, `qmm_moe.metal` = fused MoE
  grouped GEMMs), Gated DeltaNet (decode, token-parallel prefill, verify with a rollback tape), attention
  (split-K decode, prefill v11, split-key prefix attention), MoE routing, on-GPU sampling, speculative
  verify kernels (`verify.metal`).
- `src/model/qwen35*.cpp`: the model (decode, chunked prefill with in-pass checkpoints, MTP speculative
  decoding: draft chain, multi-row verify, tape commit).
- `src/engine/`: chat templates (minja), slots with prefix reuse and recurrent-state checkpoints at turn
  boundaries (a user query's outlives the tool responses after it), streaming parser for reasoning /
  content / tool calls.
- `src/tokenizer/`: byte-level BPE, identical to HF tokenizers.
- `src/loader/`: aligned parallel safetensors loader (HF offsets are unaligned; weights are repacked into
  256-byte-aligned GPU buffers at load).
- `tools/`: `ni_run` (raw generation: `ni_run <model> @<ids file> <n_gen> <temp> [--spec D] [--dump-logits f]`),
  kernel benchmarks (`bench_qmm`, `bench_qmvr`, `bench_attn`, `bench_moe`, `bench_gdn`, `bench_qmms`,
  `bench_mma`), `ni_compile` (compile-check; `NI_ARCHIVE_DIR=<dir>` prints code size, stack and GPRs),
  `ni_kstats --stats` (registers and spills from the pipeline state).
- `bench/prompts/`: token-id prompts for A/B runs (`make_prompts.py` regenerates them);
  `bench/shortpf/`: cached-turn / short-prompt TTFT through the server;
  `bench/ckpt/`: TTFT of a user query after scripted agent tool loops (`tool_loop_ttft.py`, `compare.py`),
  how turn boundaries are classified (`kind_probe.py`).
- `tests/test_server.py`: end-to-end API checks (13).

## Switches (environment, read once at start; the `NI_` prefix predates the name)

Defaults are the measured winners on an M4 Pro; each switch exists to A/B or to fall back.

| Variable | Default | Effect |
| --- | --- | --- |
| `NI_QMM`, `NI_QMM_GU` | on | qmr dense prefill GEMM and fused gate/up/SwiGLU; `0` = original kernels |
| `NI_SHORTPF` | on | short-prefill paths (one weight pass per suffix, short-T GEMM choice); `0` = off |
| `NI_QUERY_CKPT` | on | a user query's turn-boundary checkpoint outlives tool-response ones (tool loops); `0` = oldest first |
| `NI_MOE_FUSED` | 2 | fused MoE prefill GEMMs (1 = shared expert folded in, 0 = original) |
| `NI_GDN_PAR`, `NI_GDN_STEP` | on, `gdn_step_par_l16` | token-parallel DeltaNet prefill; `NI_GDN_PAR=0` = original |
| `NI_ATTN` | `attn_prefill11` | prefill attention kernel name |
| `NI_FULL_DRAFT` | dense on, MoE off | MTP drafts over the full vocabulary (else ids < 32768 + specials) |
| `NI_NO_MTP` | off | do not load the MTP head |
| `NI_PREWARM` | on | create all pipelines at load |
| `NI_PROFILE` | off | per-kernel GPU profile (isolated dispatches; timings of the whole run are not meaningful) |
| `NI_LOG` | 1 | 0 quiet, 1 per-request timing, 2 debug (speculation counters, checkpoint restores and overwrites) |
| `NI_MATH_SAFE` | off | compile kernels without fast math (bitwise A/B checks only; slower) |

## Benchmarks

A/B on the GPU only while holding the shared lock (`lockf -k ~/.cache/nace-gpu.lock <cmd>`), interleave the
runs, and report all of them. Defaults change only after an interleaved A/B plus a greedy/logits check against
the old path (differences allowed only at near-ties). The cross-engine harness lives in the Nace Agent repo
(`research/inference/scripts`: `bench.py`, `compare_engines.py`, `compare_table.py`).

## Third-party

MIT: nlohmann/json, minja (patched: Jinja `undefined` test), cpp-httplib. Ideas credited in code comments
where they come from other engines (Splash, MLX, MTPLX).
