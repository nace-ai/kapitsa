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
| `--ctx <tokens>` | Context per conversation slot (default 32768). Its KV cache grows as the conversation fills (Memory). |
| `--slots <n>` | Conversations kept warm for prefix reuse (default 2). |
| `--spec <n>` | MTP draft depth when no DFlash2 drafter is used; default per model (35B MoE 1, 27B 2, 4B 1); `0` turns speculative decoding off (MTP and DFlash2). |
| `--dflash auto\|none\|<dir>` | DFlash2 block drafter (default `auto`: a `*-DFlash2` drafter matching the model in the model's directory, next to it, in `<parent>/drafters` or in `~/.cache/nace-infer-models/drafters`; MTP when none is found, or where the drafter's memory would keep one `--ctx` conversation from fitting the memory budget). `none` keeps MTP. `NI_DFLASH` sets the same when the option is absent (`0` = none). |
| `--prefill-chunk <rows>` | Prefill chunk size (default 2048; 1024 when 2048-row buffers would keep one conversation from its full context). |
| `--mem-budget <GB>` | GPU memory Kapitsa may hold (default: Metal's recommended working set minus 4%, at least 512 MB; `NI_MEM_BUDGET`). |
| `--api-key <key>` | Require `Authorization: Bearer <key>` (except `/health`). |

`kapitsa doctor [--model DIR]... [--ctx 65536] [--slots 2] [--dflash auto|none|DIR] [--json]` prints this Mac (chip,
GPU cores, RAM, Metal's recommended working set, the memory budget, the Neural Engine) and, for each model directory
(default: the ones under `~/.cache/nace-infer-models`, `~/.cache/nace-agent-models`, `~/.nace/models`), what the
engine would decide at that context and slot count, from the safetensors headers alone (`plan_load`, shared with the
engine): its memory (with the DFlash2 drafter where the engine would load it), whether it starts, how many tokens of
KV cache fit, the decoding (DFlash2, MTP or plain), the Neural Engine share, rough speeds. `--json` is the same for
programs (the app's fit check: `starts`, `fits_full_context`, `kv_tokens_in_budget`, `decoding`, `dflash_used`).

Endpoints: `POST /v1/chat/completions` (SSE or JSON; `reasoning_content`; Qwen XML tool calls returned as
OpenAI `tool_calls`; `usage` in the final chunk with `stream_options.include_usage`), `GET /v1/models` (also
`context_fit`: the tokens one conversation can hold within the memory budget), `GET /health`. Thinking:
`chat_template_kwargs.enable_thinking=false` or `reasoning_effort: "none"`.

## Memory

Kapitsa holds at most a budget of GPU memory: Metal's recommended working set for this GPU
(`recommendedMaxWorkingSetSize`, which macOS sizes from the RAM: 40.2 GB, 78%, on this 48 GB M4 Pro under macOS 27;
`kapitsa doctor` prints it) minus 4% (at least 512 MB) for the Neural Engine service and other GPU users;
`--mem-budget` / `NI_MEM_BUDGET` set it. What it holds:
- fixed at start: the weights, the MTP head, prefill buffers for one chunk (they scale with `--prefill-chunk`), and per
  slot the recurrent state with its two checkpoints and the verify tape (4B 0.16 GB, 27B 0.49, 35B 0.21);
- the KV cache, per slot, of the positions in use: a slot allocates none until its first conversation and then grows
  in steps (by a quarter, in 1024-position steps, or exactly when the budget is short): new buffers, a GPU copy of the
  old positions, the old ones released in ~256 MB batches (so a growth needs up to 256 MB more for a moment). 4B 36 KB
  per token, 27B 68 KB, 35B 22 KB (MTP head included);
- the Neural Engine split (dense models, where the chip has a share): 0.24 GB (4B) / 0.79 GB (27B) of surfaces and
  scratch, built at the first request after its program compiled;
- the DFlash2 drafter where one is used: its weights (projections quantized at load, codebooks as stored), its
  reduced draft head (rows of the model's LM head), buffers, the first prefill's prompt captures and a 2048-position
  context ring per slot and checkpoint: 27B 2.03 GB, 35B 1.04 GB with 2 slots (the engine logs what loading took
  against this plan). `--dflash auto` leaves it out (MTP instead) where one `--ctx` conversation would not fit the
  budget beside it: context before speed.

Before reading any weights the engine checks that the fixed part plus an 8k-token KV cache fits the budget; if not it
exits with `insufficient memory: ...` (what each part needs and what would help), which the app treats as a memory
error. A request whose prompt does not fit gets an `insufficient memory` error; before that, its slot gives up a
cache the prompt reuses nothing of (the new one starts empty, without a copy), then idle slots give up their KV caches
(least recently used first; their conversations start from a zero state and are prefilled again when they return)
and, last, the Neural Engine split is released (prefill then runs on the GPU alone). A generation that cannot grow its
cache ends with `length`. `kapitsa doctor` shows the numbers for a Mac before anything loads.

## What runs where

Memory at the app's settings (`--ctx 65536 --slots 2`; `kapitsa doctor --json` computes the same on any Mac).
Fixed = weights + MTP head + prefill buffers + both slots' recurrent state (+ the Neural Engine split for dense
models, + the DFlash2 drafter where it is used); the rest of the budget is KV cache, shared by the conversations as
they fill (one conversation needs at most 64k tokens: 4B 2.4 GB, 27B 4.5 GB, 35B 1.4 GB). The budget is Metal's
recommended working set less 4%; macOS sizes it from the RAM and the fraction differs between macOS versions (78% on
this M4 Pro under macOS 27; older releases are commonly reported at about 2/3), so both are shown. Tokens of KV cache
in all, and the decoding the engine picks with the drafters installed:

| Mac (RAM) | budget at 2/3 - 78% | Qwen3.5-4B (3.7 GB fixed) | Qwen3.8-27B (17.6-18.1 GB fixed; drafter +2.03) | Qwen3.6-35B-A3B (20.9-21.2 GB fixed; drafter +1.04) |
| --- | --- | --- | --- | --- |
| MacBook Air 16 GB | 11.0 - 12.9 GB | 198k - 249k tokens, MTP | does not fit | does not fit |
| MacBook Air 24 GB | 16.5 - 19.3 GB | 347k - 424k, MTP | does not fit at 2/3; 23k at 78% (35k once the Neural Engine split is released), MTP | does not fit |
| MacBook Air / Pro 32 GB | 22.0 - 25.7 GB | 497k - 598k, MTP | 62k with MTP (73k without the split) - 80k with DFlash2 | 50k with MTP - 153k with DFlash2 |
| MacBook Pro 48 GB (this M4 Pro) | 33.0 - 38.6 GB | 795k - 947k, MTP | 184k - 265k, DFlash2 | 475k - 724k, DFlash2 |

Where a full 64k conversation would not fit beside the prefill buffers, the engine takes 1024-row prefill chunks
(27B and 35B on 32 GB at 2/3: 0.45 / 0.37 GB less); a conversation that outgrows the budget takes the KV caches of
idle slots, then the Neural Engine split's memory, then is refused with `insufficient memory`.

Speed (tok/s; Qwen3.5-4B / Qwen3.8-27B / Qwen3.6-35B-A3B, 4-bit). Measured on this MacBook Pro (M4 Pro, 20-core
GPU, 273 GB/s); the MacBook Air rows are estimates, not measurements: decode scaled by memory bandwidth (M4 Air
120 GB/s, M5 Air 153 GB/s), GPU prefill by GPU cores (10 of 20), the Neural Engine (the same 16-core unit on the M4
family) taking the dense models' whole FFN. A fanless Air also slows down after a minute or two of sustained work.

| | decode, plain | decode, speculative (default: DFlash2 where the drafter fits, else MTP) | prefill 4k |
| --- | --- | --- | --- |
| M4 Pro 20-core (measured) | 83 / 16 / 105 | 120 (MTP 1) / 61 greedy, 53 at T 0.6 (DFlash2; MTP 2: 29-31) / 170 greedy, 152 at T 0.6 (DFlash2; MTP 1: 116-124) | 2000 (GPU + ANE; GPU alone 890) / 385-393 (GPU + ANE; GPU alone 111) / 1115-1120 |
| M4 Air 10-core (estimate) | ~36 / ~7 / ~46 | ~52 / ~25-30 (DFlash2, 32 GB at 78%; MTP ~13) / ~70 (DFlash2, 32 GB at 78%; MTP ~53) | ~1000 / ~190 / ~550 |
| M5 Air 10-core (estimate) | ~46 / ~9 / ~59 | ~67 / ~30-35 (DFlash2; MTP ~17) / ~90 (DFlash2; MTP ~67) | ~1000 / ~190 / ~550 (Kapitsa does not use the M5's GPU neural accelerators yet) |

The DFlash2 estimates are rougher: its 8-row verify of the 27B is compute bound on 10 cores (2 x 8 x 27e9 flops per
cycle) as much as bandwidth bound, and its drafting pass is compute.

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
  boundaries (a user query's outlives the tool responses after it); `output_parser.h`: streaming parser for
  reasoning / content / tool calls (`tools/parser_test <model dir>` checks it on real token ids, CPU only).
- `src/tokenizer/`: byte-level BPE, identical to HF tokenizers.
- `src/loader/`: aligned parallel safetensors loader (HF offsets are unaligned; weights are repacked into
  256-byte-aligned GPU buffers at load).
- `tools/`: `ni_run` (raw generation: `ni_run <model> @<ids file> <n_gen> <temp> [--spec D | --dflash auto]
  [--ignore-eos] [--ref <plain output>] [--dump-logits f]`; plain decoding by default; `--kv-grow` starts with an
  empty KV cache and grows it before every chunk / step as the server does (not with `--dflash`); `--tbt
  --dump-all-logits f` writes the teacher-forced logits of every prompt position), DFlash2 checks
  (`dflash_check` drafter parity with the oracle dumps, `dflash_accept_test` S1, `dflash_stat` S2,
  `dflash_seedtest` sampled exactness through the engine's loop, `engine_turns` multi-turn prefix reuse,
  `ni_v8check` / `ni_v8drift` verify-vs-decode numerics, `bench_qmm8` / `bench_v8` GEMMs),
  kernel benchmarks (`bench_qmm`, `bench_qmvr`, `bench_qmv` (`SHAPES=kernel:N:K:rows,...` for any shape),
  `bench_attn`, `bench_moe`, `bench_gdn`, `bench_qmms`, `bench_mma`), `ni_compile` (compile-check;
  `NI_ARCHIVE_DIR=<dir>` prints code size, stack and GPRs), `ni_kstats --stats` (registers and spills from the
  pipeline state), `ni_kstats --limits` (every kernel's threadgroup memory against the 32 KB of Apple7-Apple10,
  pipelines below 1024 threads), `mtp_sidecar.py` (an MTP head for an MLX model that dropped it, from the official
  checkpoint by HTTP range requests: `tools/mtp_sidecar.py Qwen/Qwen3.5-4B <model dir>/mtp.safetensors`).
- `bench/prompts/`: token-id prompts for A/B runs (`make_prompts.py` regenerates them);
  `bench/shortpf/`: cached-turn / short-prompt TTFT through the server;
  `bench/ckpt/`: TTFT of a user query after scripted agent tool loops (`tool_loop_ttft.py`, `compare.py`),
  how turn boundaries are classified (`kind_probe.py`).
- `tests/test_server.py`: end-to-end API checks (14).

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
| `NI_ANE`, `NI_ANE_UNITS` | dense models, where the chip has a share | GPU + Neural Engine split of the dense prefill FFN (two-half wavefront); `0` = GPU only; `NI_ANE_UNITS` = the Neural Engine's share in 512-channel units. The drafter capture follows every schedule |
| `NI_SHORTPF_FOLD` | off | short prefills: the shared expert as expert E in the routed expert dispatches |
| `NI_HISTORY_PREFILL` | on | server: after a reply ending with "stop", once idle for `NI_HISTORY_IDLE_MS` (250), prefill the answer as the next turn renders it |
| `NI_QUERY_CKPT` | on | a user query's turn-boundary checkpoint outlives tool-response ones (tool loops); `0` = oldest first |
| `NI_THINK_TOOLCALLS` | on | a tool call the model writes inside its reasoning and ends the turn with is returned as a tool call; `0` = left in the reasoning |
| `NI_GDN_PAR`, `NI_GDN_STEP` | on, `gdn_step_parr4_l16` | token-parallel DeltaNet prefill (4 value rows per lane); `NI_GDN_PAR=0` = original |
| `NI_ATTN` | `attn_prefill11r` | prefill attention kernel name (11r: v11 skipping unit rescales, bitwise equal) |
| `NI_DUP` | off | `NI_DUP=<kernel>[,...]`: dispatch those kernels twice (in-pipeline cost; idempotent kernels only) |
| `NI_FULL_DRAFT` | dense with lm_head < 10% of the weights (27B) | MTP drafts over the full vocabulary (else ids < 32768 + specials) |
| `NI_NO_MTP` | off | do not load the MTP head |
| `NI_MTP_HIDDEN` | the pack's metadata | `pre` / `post`: the trunk hidden the MTP head reads (before / after the final norm) |
| `NI_MEM_BUDGET` | recommended working set - 4% | GPU memory budget in GB (`--mem-budget`) |
| `NI_GPU_CORES` | IORegistry `gpu-core-count` | GPU cores the dispatch heuristics assume (tests of the 8/10-core choices on a bigger GPU) |
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
