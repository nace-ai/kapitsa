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
| `--spec <n>` | MTP draft depth when no DFlash2 drafter is used; default per model (MoE 1, dense 2); `0` turns speculative decoding off (MTP and DFlash2). |
| `--dflash auto\|none\|<dir>` | DFlash2 block drafter (default `auto`: a `*-DFlash2` drafter matching the model in the model's directory, next to it, in `<parent>/drafters` or in `~/.cache/nace-infer-models/drafters`; MTP when none is found). `none` keeps MTP. `NI_DFLASH` sets the same when the option is absent (`0` = none). |
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
- `src/model/qwen35*.cpp`: the model (decode, chunked prefill with in-pass checkpoints, the GPU + Neural
  Engine FFN split (`qwen35_ane.cpp`), MTP speculative decoding: draft chain, multi-row verify, tape commit;
  the 8-row verify of DFlash2 in `qwen35_v8.cpp` + `verify8.metal`).
- `src/model/dflash2.*`, `src/kernels/spec_dflash.metal`: the DFlash2 block drafter (load-time 4-bit
  quantization, the target-hidden capture hook, the per-slot context ring, the drafting pass, on-GPU acceptance,
  the double-buffered decode cycle). Math: `research/inference/dflash2/SPEC.md` in the Nace Agent repo.
- `src/engine/`: chat templates (minja), slots with prefix reuse and recurrent-state checkpoints at turn
  boundaries (a user query's outlives the tool responses after it), streaming parser for reasoning /
  content / tool calls.
- `src/tokenizer/`: byte-level BPE, identical to HF tokenizers.
- `src/loader/`: aligned parallel safetensors loader (HF offsets are unaligned; weights are repacked into
  256-byte-aligned GPU buffers at load).
- `tools/`: `ni_run` (raw generation: `ni_run <model> @<ids file> <n_gen> <temp> [--spec D | --dflash auto]
  [--ignore-eos] [--ref <plain output>] [--dump-logits f]`; plain decoding by default), DFlash2 checks
  (`dflash_check` drafter parity with the oracle dumps, `dflash_accept_test` S1, `dflash_stat` S2,
  `dflash_seedtest` sampled exactness through the engine's loop, `engine_turns` multi-turn prefix reuse,
  `ni_v8check` / `ni_v8drift` verify-vs-decode numerics, `bench_qmm8` / `bench_v8` GEMMs),
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
| `NI_MOE_FUSED` | 2 | fused MoE prefill GEMMs (1 = shared expert folded in, 0 = original) |
| `NI_MOE_QMR` | 32 | the fused MoE prefill's expert GEMMs on the qmr design (qrm, 32- or 16-row tiles); `0` = qmm_moe |
| `NI_QMM8` | 1 | 8-bit prefill GEMMs on qmr8: 1 = the MTP head, 2 = also the trunk's router / shared gate, 0 = off |
| `NI_SHORTPF_MOE_FMA` | on | short-prefill experts of <= 2 rows as FMA matvecs (qmms_moe2); `0` = 8-row MMA for all |
| `NI_QMMS_TY` | 1 | small qmms dispatches split their token blocks over grid.y (2 = also any > 4 blocks); `0` = off |
| `NI_MOE_BUCKET3`, `NI_TOPK8` | on | parallel-scan MoE bucketing and 8-tokens-per-threadgroup top-k (bitwise equal) |
| `NI_MTP_DEFER` | on | the last prefill chunk's MTP history pass (or, with DFlash2, the drafter's commit of the prompt rows) runs after the first token's command buffer |
| `NI_DFLASH` | auto | server/engine: DFlash2 drafter (`auto`, `0`/`none`, or a directory) when `--dflash` is absent; `ni_run`: off unless set or `--dflash` |
| `NI_DFLASH_VERIFY` | 7 | proposals verified per DFlash2 cycle (1-7) |
| `NI_DFLASH_DYN`, `NI_DFLASH_DYN_T` | MoE 0.5 / 0.3, dense off | dynamic verify depth threshold at T = 0 / T > 0 (`0` = verify all) |
| `NI_DFLASH_EXACT_MOE` | on | sampled DFlash2 cycles run every routed expert of the 8-row verify as an MMA tile (row-independent numerics) |
| `NI_DFLASH_V8` | hidden >= 4096 | the drafter's MLP and LM head on the verify GEMMs |
| `NI_DFLASH_BITS` | 4 | drafter projections quantized at load to 4 or 8 bits |
| `NI_DFLASH_HEAD`, `NI_DFLASH_HEAD_IDS` | 98304, unset | the drafter's top-16 comes from a reduced LM head: the lowest 98304 token ids (the tokenizer's BPE merge order, a frequency proxy) plus every added/special token; `NI_DFLASH_HEAD=n` sets the rows (`0` = full vocabulary), `NI_DFLASH_HEAD_IDS=<file>` takes a ranked uint32 id list instead (65536 rows by default). The verify always uses the full head |
| `NI_DFLASH_CB` | 3 | `ni_run` DFlash2 loop: 3 double-buffered, 2 two command buffers per cycle, 1 one |
| `NI_V8_CM`, `NI_V8_PAIR`, `NI_V8_KPLAN`, `NI_V8_GUSPLIT`, `NI_V8_GDNSPLIT`, `NI_V8_PREPU` | measured winners | 8-row verify kernel knobs (`V8Knobs`, `qwen35.h`); `ni_run` with `NI_AB=key=value,...` runs those settings in odd cycles and reports both halves' GPU time |
| `NI_V8` | off | `1`: every multi-row verify (R >= 2) on the 8-row path (A/B; R <= 4 keeps the MTP-era kernels otherwise) |
| `NI_ANE` | dense on | GPU + Neural Engine split of the dense prefill FFN (two-half wavefront); `0` = GPU only. The drafter capture follows every schedule |
| `NI_SHORTPF_FOLD` | off | short prefills: the shared expert as expert E in the routed expert dispatches |
| `NI_HISTORY_PREFILL` | on | server: after a reply ending with "stop", once idle for `NI_HISTORY_IDLE_MS` (250), prefill the answer as the next turn renders it |
| `NI_QUERY_CKPT` | on | a user query's turn-boundary checkpoint outlives tool-response ones (tool loops); `0` = oldest first |
| `NI_GDN_PAR`, `NI_GDN_STEP` | on, `gdn_step_parr4_l16` | token-parallel DeltaNet prefill (4 value rows per lane); `NI_GDN_PAR=0` = original |
| `NI_ATTN` | `attn_prefill11r` | prefill attention kernel name (11r: v11 skipping unit rescales, bitwise equal) |
| `NI_DUP` | off | `NI_DUP=<kernel>[,...]`: dispatch those kernels twice (in-pipeline cost; idempotent kernels only) |
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
