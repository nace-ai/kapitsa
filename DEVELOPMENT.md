# Development

Building Kapitsa, internals, tuning switches and tools. For what the engine is and how fast it runs, see
[README.md](README.md); measured history lives in [docs/PERF.md](docs/PERF.md).

## Build from source

```bash
./install.sh                     # checks the Mac and the tools, builds, installs kapitsa on the PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j   # everything, tools included
```

Needs CMake 3.24+ and the Xcode Command Line Tools only. Metal kernels are embedded as source and compiled at
runtime by the OS (`newLibraryWithSource`), so no offline `metal` compiler or Xcode is required. Pipelines are
created at load (`NI_PREWARM=0` skips that and creates them on first use).

`install.sh` checks for Apple silicon (and a terminal not under Rosetta), macOS 14+, the Command Line Tools (opens
their installer) and CMake (offers `brew install cmake`; `--yes` doesn't ask), builds the `kapitsa` target in
Release, and installs it to `/opt/homebrew/bin` when that is writable, else `~/.local/bin` (`--prefix DIR`
overrides), with a note when that folder is not on the PATH. Run it again to update. To uninstall, delete the binary
and `~/.cache/kapitsa`. `packaging/homebrew/kapitsa.rb` is a Homebrew formula for a public release: it needs a tag,
the tarball's sha256 and a tap (see the file); nothing is published.

## Commands

| Command | What it does |
| --- | --- |
| `kapitsa serve [--model M] [options]` | Starts the server (below). `M`: a catalog name or alias, `org/name[@revision]`, or a folder; downloads it first if needed. No `--model`: the first catalog model that fits this Mac's GPU memory budget with room for a 16k-token conversation (Metal's recommended working set less 4%; `NI_MEM_BUDGET` in GB overrides): the 35B-A3B on 48 GB, the 35B-A3B or the 27B on 32-36 GB depending on how much macOS gives the GPU, the 4B below. |
| `kapitsa pull M... [--from DIR]...` | Downloads without serving. `--dry-run [--json]` prints what `M` resolves to (no network). |
| `kapitsa list [--json]` | The models in the store: size on disk, complete or not, MTP head / DFlash2 drafter. |
| `kapitsa rm M...` | Deletes downloaded models (only folders in the store). |
| `kapitsa mtp-sidecar SRC OUT [--revision R] [--hidden post_norm\|pre_norm]` | `tools/mtp_sidecar.py` in C++, same bytes (see below). |
| `kapitsa --help`, `serve --help`, `pull --help`, `--version` | |

Exit codes: 0, 1 for failures (download, port, model), 2 for usage errors (unknown option or model), 130 after
Ctrl+C during a download (the parts are kept). `fatal()` exits with 1 (`NI_ABORT=1` aborts instead, for a backtrace).
Options without a command still mean `serve` (`kapitsa --model DIR`), and `build/nace-infer-server` stays a symlink
to `kapitsa` for scripts from before the rename.

## Server

```bash
kapitsa serve --model 35b                        # or ./build/kapitsa serve --model <folder> --model-id <id>
```

| Option | Purpose |
| --- | --- |
| `--model <name\|org/name[@rev]\|dir>` | The model (above). A folder is an MLX-format model directory (affine 4/8-bit; the MTP head may live in `mtp.safetensors`), used as is. |
| `--model-id <id>` | Name reported by `/v1/models` (default: the catalog name, the repo id, or the folder's name). |
| `--host`, `--port` | Listen address (default `127.0.0.1:8000`). |
| `--ctx <tokens>` | Context per conversation slot (default 32768). |
| `--slots <n>` | Conversations kept warm for prefix reuse (default 2). |
| `--spec <n>` | MTP draft depth when no DFlash2 drafter is used; default per model (MoE 1, dense 2); `0` turns speculative decoding off (MTP and DFlash2). |
| `--dflash auto\|none\|<dir>` | DFlash2 block drafter (default `auto`: a drafter in the model's `dflash2/` or `drafter/` folder, else a `*-DFlash2` drafter matching the model next to it, in `<parent>/drafters` or in `~/.cache/nace-infer-models/drafters`; MTP when none is found). `none` keeps MTP. `NI_DFLASH` sets the same when the option is absent (`0` = none). |
| `--prefill-chunk <rows>` | Prefill chunk size (default 2048). |
| `--api-key <key>` | Require `Authorization: Bearer <key>` (except `GET /`, `/favicon.ico` and `/health`). |

Endpoints: `GET /` (the chat page), `POST /v1/chat/completions` (SSE or JSON; `reasoning_content`; Qwen XML tool
calls returned as OpenAI `tool_calls`; `usage` in the final chunk), `GET /v1/models`, `GET /health`. Thinking:
`chat_template_kwargs.enable_thinking=false` or `reasoning_effort: "none"`.

Startup: the port is checked before anything loads (a busy port is an error: the socket does not use httplib's
default `SO_REUSEPORT`, which would let a second server bind it too), the model is downloaded if needed, the engine
loads, and one line on stdout says `Ready: chat at http://127.0.0.1:8000, API at http://127.0.0.1:8000/v1 ...`.
Logs go to stderr with a `[kapitsa]` prefix.

The chat page is `src/server/chat.html`, embedded at build time (`cmake/embed_file.cmake`). One file, 17 KB: no
libraries, no requests anywhere but this server (the response's Content-Security-Policy enforces it). It streams
`/v1/chat/completions`, shows the reasoning in a collapsible block, has a thinking switch, a stop button and code
blocks with a copy button, shows the model name and each answer's tokens, tok/s (from `usage.completion_tokens`
over the time from the first to the last streamed token), time to first token and cached prompt tokens, and keeps
the conversation in the browser's localStorage. With `--api-key` it asks for the key once (also kept in
localStorage). Markdown is rendered by a small escaping renderer (code fences, inline code, headings, lists, tables,
quotes, bold/italic, http(s) links); model output never reaches the page as HTML.

## Models and downloads

The store is `~/.cache/kapitsa/models/<name>/` (`KAPITSA_HOME` replaces `~/.cache/kapitsa`): a catalog model under
its name, a repo `org/name[@rev]` as `org--name[@rev]`. In a model folder:

```
<file>                          finished, verified files (the drafter in dflash2/, which --dflash auto finds)
.kapitsa/parts/<file>.part      a file being downloaded, written at its byte offsets
.kapitsa/parts/<file>.part.json the byte ranges written so far (resume)
.kapitsa/manifest.json          written last: repo, revision, commit, every file with its size (and sha256), companions
```

A model whose manifest lists every file at its size is complete: `serve` uses it without any network access. A
catalog model is also checked against the revisions this build pins; after an update that pins newer ones, the
next start lists the repos again, checks the files that are there by hash, and fetches what changed (offline, it
keeps using the old copy and says so).

The catalog (`src/hub/catalog.cpp`) pins the same repos and revisions as Nace Agent's
`packages/nace-runtime/src/catalog-data.ts`, so the app and the CLI fetch the same bytes:

| Name (alias) | Weights | Companions |
| --- | --- | --- |
| `qwen3.6-35b-a3b` (`qwen3.6-35b`, `35b`) | `Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed` @ `95460516` (`mlx-community/Qwen3.6-35B-A3B-4bit` plus its MTP head) | DFlash2 drafter `incoai/Qwen3.6-35B-A3B-DFlash2` @ `51ef7b69` (config.json, model.safetensors, LICENSE) in `dflash2/` |
| `qwen3.8-27b` (`27b`) | `lmstudio-community/Qwen3.8-27B-MLX-4bit` @ `6067b15c` | MTP head: `mtp.safetensors` + `mtplx_runtime.json` from `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed` @ `1d5087d2`; DFlash2 drafter `incoai/Qwen3.8-27B-DFlash2` @ `015e7956` in `dflash2/` |
| `qwen3.5-4b` (`4b`) | `lmstudio-community/Qwen3.5-4B-MLX-4bit` @ `c43ee1d6` | MTP head built from `Qwen/Qwen3.5-4B` @ `851bf6e8` (below), sha256 `4f491dd2...` |

Downloads (`src/hub/fetch.cpp`, transport `src/hub/http.mm`):
- The file list comes from the Hub API with sizes and checksums (`/api/models/<repo>/revision/<rev>?blobs=true`:
  sha256 for LFS files, the git blob id for the others). Every file is checked after download: sha256, or the
  blob id (sha1 of `blob <size>\0` + content) for small files. A mismatch deletes the part and says so.
- NSURLSession (part of macOS: system proxies and trust store, no library): 8 connections, one session each so
  they are separate TCP flows (a single flow is capped well below the link on some networks; 8 ranges measured ~4x
  one stream here), 64 MB ranges taken from a queue, small files first. Each range retries 8 times with backoff
  from where it stopped. `Authorization` is not forwarded when the Hub redirects to its CDN.
- Resume: the part's written ranges are saved every 2 s and on Ctrl+C (the first stops cleanly, a second quits at
  once). Running the same command continues; a part whose listing changed starts over.
- Before downloading: the free disk space must cover what is left plus 1 GB, or it stops with what is needed.
  One download per model folder at a time (a lock file). A first attempt that fails before any file arrives leaves
  no folder behind.
- Reuse before download: the same file in the Hugging Face cache (`HF_HUB_CACHE`, `HF_HOME/hub`, else
  `~/.cache/huggingface/hub`, at the pinned commit; the blob's name proves its content), files under `--from`
  folders, and for a catalog model the Nace Agent app's copy (`~/.nace/models/<key>` with its companions); in
  folders, any name up to 3 levels deep, matched by size and then by checksum. They are placed as APFS clones (no
  extra space), or copied across volumes.
- `HF_TOKEN` (or the token `hf auth login` saved) is sent for gated or private repos; `HF_ENDPOINT` points at a
  mirror. Errors say what to do: offline, not found, gated, rate-limited, out of disk.
- A repo outside the catalog takes its top-level files except other formats (`.gguf`, `.bin`, `.pth`, images,
  ...). Its config.json is read first: a model that is not a Qwen3.5-family hybrid (`linear_num_value_heads`) in
  MLX affine 4- or 8-bit is refused before any weights download. Such models get no companions (an MTP head inside
  the repo is used).

The 4B's MTP head: MLX conversions of the small Qwen3.5 models drop it, and the official checkpoint keeps it as 15
bf16 tensors named `mtp.*` (241 MB of 9.3 GB). `kapitsa mtp-sidecar` (and `pull` for the 4B) fetches the index,
each shard's safetensors header and only those tensors' byte ranges, adds 1 to the RMSNorm weights (Hugging Face
stores them zero-centred, MLX conversions as w + 1), keeps the linears bf16 (Kapitsa quantizes them at load) and
writes the header with `base_hidden_variant`: byte for byte what `tools/mtp_sidecar.py` and the Nace Agent app make
(same sha256 for the real head; `tests/test_sidecar.py` compares the two on a checkpoint with every finite bf16
value). It takes 19 s here with parallel ranges; the Python tool, one stream, 94 s.

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
  boundaries, streaming parser for reasoning / content / tool calls.
- `src/hub/`: the model catalog and `--model` resolution (`catalog.cpp`), downloads, reuse, verification, the
  manifest and the MTP sidecar builder (`fetch.cpp`), the NSURLSession transport (`http.mm`).
- `src/server/`: the command line and the HTTP server (`main.cpp`), the chat page (`chat.html`).
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
  `bench/shortpf/`: cached-turn / short-prompt TTFT through the server.
- `tests/test_server.py`: end-to-end API checks against a running server (the chat page, the API, tool calls,
  prefix reuse; `--api-key KEY` adds the key checks), and with `--cli build/kapitsa` offline checks of the command
  line (names and aliases at the app's pinned revisions, repo ids, folders, refusals, a finished download used without
  network, list, rm, the early port check). `tests/test_sidecar.py build/kapitsa`: the sidecar builder against
  `tools/mtp_sidecar.py`.

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
| `NI_GDN_PAR`, `NI_GDN_STEP` | on, `gdn_step_parr4_l16` | token-parallel DeltaNet prefill (4 value rows per lane); `NI_GDN_PAR=0` = original |
| `NI_ATTN` | `attn_prefill11r` | prefill attention kernel name (11r: v11 skipping unit rescales, bitwise equal) |
| `NI_DUP` | off | `NI_DUP=<kernel>[,...]`: dispatch those kernels twice (in-pipeline cost; idempotent kernels only) |
| `NI_FULL_DRAFT` | dense on, MoE off | MTP drafts over the full vocabulary (else ids < 32768 + specials) |
| `NI_NO_MTP` | off | do not load the MTP head |
| `NI_PREWARM` | on | create all pipelines at load |
| `NI_PROFILE` | off | per-kernel GPU profile (isolated dispatches; timings of the whole run are not meaningful) |
| `NI_LOG` | 1 | 0 quiet, 1 per-request timing, 2 debug (speculation counters) |
| `NI_MATH_SAFE` | off | compile kernels without fast math (bitwise A/B checks only; slower) |
| `NI_ABORT` | off | `fatal()` calls abort() (a backtrace in a crash report or debugger) instead of exiting with 1 |
| `NI_MEM_BUDGET` | Metal's recommended working set - 4% | GPU memory budget in GB; `serve` without `--model` picks the model by it |

## Benchmarks

A/B on the GPU only while holding the shared lock (`lockf -k ~/.cache/nace-gpu.lock <cmd>`), interleave the
runs, and report all of them. Defaults change only after an interleaved A/B plus a greedy/logits check against
the old path (differences allowed only at near-ties). The cross-engine harness lives in the Nace Agent repo
(`research/inference/scripts`: `bench.py`, `compare_engines.py`, `compare_table.py`): the same prompts over HTTP for
every engine, a GPU lock so nothing else runs during a measurement, median of 5 runs, thinking off, identical
sampling settings, cold prompts cache-busted, each engine at its latest release and its own best settings. The
README compares with llama.cpp, Ollama, LM Studio, mlx-lm, oMLX and MTPLX; the internal tables cover more engines
(docs/PERF.md, research/inference in the Nace Agent repo).

## Third-party

MIT: nlohmann/json, minja (patched: Jinja `undefined` test), cpp-httplib. Ideas credited in code comments
where they come from other engines (Splash, MLX, MTPLX).
