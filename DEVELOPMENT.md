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
| `kapitsa serve [--model M] [options]` | Starts the server (below). `M`: a catalog name or alias with an optional `:<variant>`, `org/name[@revision]`, or a folder; downloads it first if needed. No `--model`: the first catalog model (best first: 35B-A3B, 27B, 4B) whose memory plan, the engine's own (`conversation_fits`: one conversation of `--ctx` tokens beside the model, its buffers, the slots' state and the Neural Engine split, the test plan_load applies before it keeps a drafter), fits this Mac's budget; a model not downloaded yet is planned from its headers (`hub::plan_dir`). An explicit catalog model the engine would refuse (`start_need` over the budget) is refused before its download, naming the one that fits. |
| `kapitsa pull M... [--from DIR]...` | Downloads without serving. `--dry-run [--json]` prints what `M` resolves to (no network). |
| `kapitsa list [--json]` | The models in the store: size on disk, complete or not, MTP head / DFlash2 drafter. |
| `kapitsa rm M...` | Deletes downloaded models (only folders in the store). |
| `kapitsa doctor [--model M]... [--json]` | This Mac and what fits it (below). |
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
| `--model <name\|org/name[@rev]\|dir>` | The model (above). A folder is an MLX-format model directory (bf16 or MLX affine 3-8-bit, see Weight formats; the MTP head may live in `mtp.safetensors`), used as is. |
| `--model-id <id>` | Name reported by `/v1/models` (default: the catalog name, the repo id, or the folder's name). |
| `--host`, `--port` | Listen address (default `127.0.0.1:8000`). |
| `--ctx <tokens>` | Context per conversation slot (default 32768). Its KV cache grows as the conversation fills (Memory). |
| `--slots <n>` | Conversations kept warm for prefix reuse (default 2). |
| `--spec <n>` | MTP draft depth when no DFlash2 drafter is used; default per model (35B MoE 1, 27B 2, 4B 1); `0` turns speculative decoding off (MTP and DFlash2). |
| `--dflash auto\|none\|<dir>` | DFlash2 block drafter (default `auto`: a drafter in the model's `dflash2/` or `drafter/` folder, else a `*-DFlash2` drafter matching the model next to it, in `<parent>/drafters` or in `~/.cache/nace-infer-models/drafters`; MTP when none is found, or where the drafter's memory would keep one `--ctx` conversation from fitting the memory budget). `none` keeps MTP. `NI_DFLASH` sets the same when the option is absent (`0` = none). |
| `--prefill-chunk <rows>` | Prefill chunk size (default 2048; 1024 when 2048-row buffers would keep one conversation from its full context). |
| `--mem-budget <GB>` | GPU memory Kapitsa may hold (default: Metal's recommended working set minus 4%, at least 512 MB; `NI_MEM_BUDGET`). |
| `--api-key <key>` | Require `Authorization: Bearer <key>` (except `GET /`, `/favicon.ico` and `/health`). |

`kapitsa doctor [--model MODEL]... [--ctx 65536] [--slots 2] [--dflash auto|none|DIR] [--json]` prints this Mac (chip,
GPU cores, RAM, Metal's recommended working set, the memory budget, the Neural Engine) and, for each model, what the
engine would decide at that context and slot count, from the safetensors headers alone (`plan_load`, shared with the
engine): its memory (with the DFlash2 drafter where the engine would load it), whether it starts, how many tokens of
KV cache fit, the decoding (DFlash2, MTP or plain), the Neural Engine share, rough speeds. `--model` takes what
`serve --model` takes; by default it shows the catalog's models (one not downloaded yet is planned from its headers,
fetched once into `~/.cache/kapitsa/plans`), then the models under `~/.cache/kapitsa/models`,
`~/.cache/nace-infer-models`, `~/.cache/nace-agent-models` and `~/.nace/models`. `--json` is the same for programs
(the app's fit check: `starts`, `fits_full_context`, `kv_tokens_in_budget`, `decoding`, `dflash_used`).

Endpoints: `GET /` (the chat page), `POST /v1/chat/completions` (SSE or JSON; `reasoning_content`; Qwen XML tool
calls returned as OpenAI `tool_calls`; `usage` in the final chunk), `GET /v1/models` (also `context_fit`: the
tokens one conversation can hold within the memory budget), `GET /health`. Thinking:
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

The catalog is one table, `src/hub/catalog.cpp`: models, best first, each with variants (the precision of its
weights: `4bit`, `5bit`, `6bit`, `8bit`, `bf16`), a default and (optional) fallbacks. A variant pins its repo to a
revision and lists its companions and its download size; its memory comes from the engine's plan of its headers.
`--model <name>` takes the default variant where that plan leaves room for one `--ctx` conversation with its drafter,
else the first fallback that does (`pick_variant`, src/engine/engine.cpp; `pull`, `serve` without `--model` and
`doctor` decide the same way); `--model <name>:<variant>` takes that one. Its folder is `<name>-<variant>`, and `list`
and `pull --dry-run` show it. The API names a default variant by the model's name (`qwen3.5-4b`) and any other as
`<name>:<variant>`. To add a variant or change a default, edit that table only: the downloader, `pull`, `list` and
`--help` follow it. The default is the highest-quality variant whose decode with speculative decoding stays within
~10% of the 4-bit variant's (today 4-bit everywhere; docs/PERF.md, Weight formats and default precisions); every
precision of a model shares its MTP head and DFlash2 drafter. The 4-bit variants are the Nace Agent app's repos and
revisions, so the app and the CLI fetch the same bytes for them:

| Name (alias) | Variants | Weights | Companions |
| --- | --- | --- | --- |
| `qwen3.6-35b-a3b` (`qwen3.6-35b`, `35b`) | `4bit` (default), `5bit`, `6bit`, `8bit`, `bf16` | 4bit: `Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed` @ `95460516` (`mlx-community/Qwen3.6-35B-A3B-4bit` plus its MTP head); `mlx-community/Qwen3.6-35B-A3B-5bit` @ `c90c1a8b`, `-6bit` @ `cb7e092e` (= lmstudio-community's), `-8bit` @ `e06a74e6`, `-bf16` @ `04bbe07c` | DFlash2 drafter `incoai/Qwen3.6-35B-A3B-DFlash2` @ `51ef7b69` (config.json, model.safetensors, LICENSE) in `dflash2/`; the other variants also take the 4bit repo's MTP head (`mtp.safetensors`, `mtplx_runtime.json`) |
| `qwen3.8-27b` (`27b`) | `4bit` (default), `5bit`, `6bit`, `8bit`, `bf16` | `lmstudio-community/Qwen3.8-27B-MLX-4bit` @ `6067b15c`, `-5bit` @ `b9ad7b92`, `-6bit` @ `05f38c1f`, `-8bit` @ `241ebb5f`; `mlx-community/Qwen3.8-27B-bf16` @ `6f265714` | MTP head: `mtp.safetensors` + `mtplx_runtime.json` from `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed` @ `1d5087d2`; DFlash2 drafter `incoai/Qwen3.8-27B-DFlash2` @ `015e7956` in `dflash2/` |
| `qwen3.5-4b` (`4b`) | `4bit` (default), `6bit`, `8bit`, `bf16` | `lmstudio-community/Qwen3.5-4B-MLX-4bit` @ `c43ee1d6`, `mlx-community/Qwen3.5-4B-6bit` @ `fbb1092a`, `lmstudio-community/Qwen3.5-4B-MLX-8bit` @ `59098e5f`, `mlx-community/Qwen3.5-4B-bf16` @ `491fdc7c` | MTP head built from `Qwen/Qwen3.5-4B` @ `851bf6e8` (below), sha256 `4f491dd2...` |

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
  ...). Its config.json is read first: a model that is not a Qwen3.5-family hybrid (`linear_num_value_heads`) in a
  format Kapitsa runs (bf16, MLX affine 3/4/5/6/8-bit; Weight formats) is refused before any weights download. Such
  models get no companions (an MTP head inside the repo is used).

The 4B's MTP head: MLX conversions of the small Qwen3.5 models drop it, and the official checkpoint keeps it as 15
bf16 tensors named `mtp.*` (241 MB of 9.3 GB). `kapitsa mtp-sidecar` (and `pull` for the 4B) fetches the index,
each shard's safetensors header and only those tensors' byte ranges, adds 1 to the RMSNorm weights (Hugging Face
stores them zero-centred, MLX conversions as w + 1), keeps the linears bf16 (Kapitsa quantizes them at load) and
writes the header with `base_hidden_variant`: byte for byte what `tools/mtp_sidecar.py` and the Nace Agent app make
(same sha256 for the real head; `tests/test_sidecar.py` compares the two on a checkpoint with every finite bf16
value). It takes 19 s here with parallel ranges; the Python tool, one stream, 94 s.

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

These are the 4-bit variants, the defaults (the app's too); `kapitsa doctor` lists every variant and what it needs
(the 35B in 6-bit, 28.2 GB of weights, fits a 48 GB Mac with a full conversation and its drafter; in 8-bit, 36.8 GB,
only with shorter ones).

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

## Weight formats

Kapitsa reads MLX-format checkpoints (and, for dense models, the official Hugging Face layout) in these formats,
each linear in its own (a checkpoint's `quantization` block may give every module another one):

| Format | Where it is found | Kernels |
| --- | --- | --- |
| MLX affine 4-bit, group 64 | the `-4bit` packs | the specialized kernels (qmv2/qmv3, qmr, qmms, qmm8, MoE) |
| MLX affine 8-bit, group 64 | `-8bit` packs, routers and shared-expert gates of every MoE pack, mixed recipes | specialized where they exist (qmv2_b8, qmv3_b8, qmr8, qmms_b8, qmm8_b8), `wfmt.metal` for the rest |
| MLX affine 3, 5, 6-bit, group 64; 4-bit, group 32 | `-3bit`/`-5bit`/`-6bit` packs, mixed recipes, the MTPLX 27B pack (4-bit group 32) | `wfmt.metal` (MoE experts: 3/5/6/8-bit and bf16; 4-bit group 32 dense only, no MoE pack uses it) |
| bf16 | `-bf16` conversions, the official Hugging Face checkpoints (dense) | `wfmt.metal` |

`src/kernels/wfmt.metal` has a format-generic form of every weight-reading kernel: decode matvecs (single, several
projections sharing an input, fused gate/up), the embedding lookup, per-row MoE (decode and MTP verify), the 8-row
verify GEMM and MoE passes (DFlash2), the prefill GEMMs on the qmr design (dense, fused gate/up, MoE experts), the
short-prefill GEMMs, and the Neural Engine split's weight staging. MLX packs b-bit codes as one little-endian bit
stream per row (value j at bits [b*j, b*j + b)), so a lane's run of 16 values is b/2 words, word-aligned for even b
and 16-bit aligned for odd b; codes become exact half operands (1024 + q, or 0.25 + q/4096 in the qmr design) and
the dequantization arithmetic is the 4-bit kernels'. bf16 weights are the MMAs' half operands (exact for |w| in
[6.1e-5, 65504]) and fp32 in the matvecs. These kernels are a second Metal library compiled on first use (~5 s once,
then the system's shader cache), so a 4-bit model never pays for them. Each format's kernels are instantiated in the
lists at the end of `wfmt.metal`: a format missing from a list fails at load with the kernel's name.

The Neural Engine split stages its rotated int8 weights from any of these formats (the same bytes as from 4-bit:
bitwise-equal outputs when only the FFN is re-encoded), and the DFlash2 drafter and the MTP heads run with a trunk
of any format (the drafter makes an 8-bit copy of the rows of an LM head that is neither 4- nor 8-bit). The Hugging
Face layout of the dense models (`model.language_model.*`, zero-centred RMSNorm weights, conv1d as stored) is
converted at load; MoE checkpoints in that layout (fused expert tensors) are refused with a message, as are MLX's
`mxfp4` / `nvfp4` / `mxfp8` modes (`kapitsa pull` refuses such repos before downloading).

A model whose weights alone exceed the memory budget is refused before anything loads (`... does not fit this Mac:
its weights alone (69.3 GB, bf16) exceed ...`), from the same plan of the safetensors headers as the fit check
(`Qwen35::plan`, formats per module).

Testing formats without downloading them: `NI_REQUANT=<bits>[:<spec>]` re-encodes a 4-bit model's linears exactly as
8, 6, 5-bit or 4-bit group 32 (codes q << (bits - 4), scales / 2^(bits - 4)), and `NI_DEQUANT=<spec>` makes them bf16
(w = s q + b rounded to bf16), at load (`spec`: `all`, `embed`, `head`, `attn`, `gdn`, `router`, `experts`, `shared`,
`mlp`, after an optional layer range `a-b:`). The function stays the 4-bit model's, so outputs must match it to
rounding, and the kernels run at the speed of a real model in that format. `tools/ni_eval` scores a model
teacher-forced on token-id sequences (next-token KL against a reference: another model in the same process, exact, or
a reference file of top-K log-probs; top-1 agreement, perplexity), and `bench/quality/` holds the held-out sequences
and the MLX-side tools (`eval_stream.py` evaluates models too big for the Mac against their bf16 checkpoint streamed
from Hugging Face one layer at a time).

## Layout

- `src/kernels/*.metal`: quantized matvec/GEMM (MLX affine 4/8-bit; `qmm3.metal` = qmr prefill GEMM with
  weights dequantized into MMA registers, `qmms.metal` = short-prefill GEMMs, `qmm_moe.metal` = fused MoE
  grouped GEMMs), Gated DeltaNet (decode, token-parallel prefill, verify with a rollback tape), attention
  (split-K decode, prefill v11, split-key prefix attention), MoE routing, on-GPU sampling, speculative
  verify kernels (`verify.metal`); `wfmt.metal`: every weight-reading kernel for the other formats (bf16, 3/5/6/8-bit,
  4-bit group 32; Weight formats above).
- `src/model/qwen35*.cpp`: the model (decode, chunked prefill with in-pass checkpoints, the GPU + Neural
  Engine FFN split (`qwen35_ane.cpp`), MTP speculative decoding: draft chain, multi-row verify, tape commit;
  the 8-row verify of DFlash2 in `qwen35_v8.cpp` + `verify8.metal`).
- `src/model/dflash2.*`, `src/kernels/spec_dflash.metal`: the DFlash2 block drafter (load-time 4-bit
  quantization, the target-hidden capture hook, the per-slot context ring, the drafting pass, on-GPU acceptance,
  the double-buffered decode cycle).
- `src/engine/`: chat templates (minja), slots with prefix reuse and recurrent-state checkpoints at turn
  boundaries (a user query's outlives the tool responses after it); `output_parser.h`: streaming parser for
  reasoning / content / tool calls (`tools/parser_test <model dir>` checks it on real token ids, CPU only).
- `src/hub/`: the model catalog (`catalog.cpp`: the table of models and variants), `--model` resolution
  (`resolve.cpp`), downloads, reuse, verification, the manifest and the MTP sidecar builder (`fetch.cpp`), the
  NSURLSession transport (`http.mm`).
- `src/server/`: the command line and the HTTP server (`main.cpp`), the chat page (`chat.html`).
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
- `tools/ni_eval`: teacher-forced quality of a model against a reference (`ni_eval <model> @ids... --ref-model
  <dir>` exact, `--dump-ref f` / `--ref f` top-K reference files; JSON lines per sequence and in total).
- `bench/quality/`: held-out token ids (C++, Python, English prose, a coding-agent chat with tool calls; 4 x 1024
  tokens per tokenizer) and the quantization measurement tools (Weight formats above; `README.md` there).
- `bench/prompts/`: token-id prompts for A/B runs (`make_prompts.py` regenerates them);
  `bench/shortpf/`: cached-turn / short-prompt TTFT through the server;
  `bench/ckpt/`: TTFT of a user query after scripted agent tool loops (`tool_loop_ttft.py`, `compare.py`),
  how turn boundaries are classified (`kind_probe.py`).
- `tests/test_server.py`: end-to-end API checks against a running server (the chat page, the API, tool calls,
  prefix reuse, thinking + tools; `--api-key KEY` adds the key checks), and with `--cli build/kapitsa` offline checks
  of the command line (names and aliases at the app's pinned revisions, repo ids, folders, refusals, a finished
  download used without network, list, rm, the early port check). `tests/test_sidecar.py build/kapitsa`: the sidecar
  builder against `tools/mtp_sidecar.py`.

## Switches (environment, read once at start; the `NI_` prefix predates the name)

Defaults are the measured winners on an M4 Pro; each switch exists to A/B or to fall back.

| Variable | Default | Effect |
| --- | --- | --- |
| `NI_QMM`, `NI_QMM_GU` | on | qmr dense prefill GEMM and fused gate/up/SwiGLU; `0` = original kernels |
| `NI_SHORTPF` | on | short-prefill paths (one weight pass per suffix, short-T GEMM choice); `0` = off |
| `NI_MOE_FUSED` | 2 | fused MoE prefill GEMMs (1 = shared expert folded in, 0 = original) |
| `NI_MOE_QMR` | 32 | the fused MoE prefill's expert GEMMs on the qmr design (qrm, 32- or 16-row tiles); `0` = qmm_moe |
| `NI_QMM8` | 1 | 8-bit prefill GEMMs on qmr8: 1 = every 8-bit linear but the trunk's MoE router and shared-expert gate (the MTP head, an 8-bit trunk), 2 = also those two, 0 = off |
| `NI_QRF16_MF` | on | bf16 large-prompt prefill GEMMs with the M tiles along grid x (threadgroups in flight share a weight tile through the cache); `0` = N tiles first |
| `NI_REQUANT`, `NI_DEQUANT` | unset | tests: a 4-bit model's linears re-encoded at load as `8`/`6`/`5`/`4g32` (`NI_REQUANT=6:experts`) or dequantized to bf16 (`NI_DEQUANT=0-3:all,head`): the same function at another format's speed (Weight formats) |
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
| `NI_MEM_BUDGET` | recommended working set - 4% | GPU memory budget in GB (`--mem-budget`); `serve` without `--model` picks the model by it |
| `NI_GPU_CORES` | IORegistry `gpu-core-count` | GPU cores the dispatch heuristics assume (tests of the 8/10-core choices on a bigger GPU) |
| `NI_PREWARM` | on | create all pipelines at load |
| `NI_PROFILE` | off | per-kernel GPU profile (isolated dispatches; timings of the whole run are not meaningful) |
| `NI_LOG` | 1 | 0 quiet, 1 per-request timing, 2 debug (speculation counters, checkpoint restores and overwrites) |
| `NI_MATH_SAFE` | off | compile kernels without fast math (bitwise A/B checks only; slower) |
| `NI_ABORT` | off | `fatal()` calls abort() (a backtrace in a crash report or debugger) instead of exiting with 1 |

## Benchmarks

A/B on the GPU only while nothing else runs on it (we hold a shared lock: `lockf -k ~/.cache/nace-gpu.lock <cmd>`),
interleave the runs, and report all of them. Defaults change only after an interleaved A/B plus a greedy/logits check
against the old path (differences allowed only at near-ties). `bench/` has the in-repo pieces: token-id prompts for
`ni_run` A/Bs (`bench/prompts/`), cached-turn and short-prompt TTFT through the server (`bench/shortpf/cached_ttft.py
--corpus FILE`) and tool-loop TTFT (`bench/ckpt/`).

### Run benchmarks locally

[Run benchmarks locally](bench/harness/README.md): the cross-engine harness behind the README's tables (the same
prompts over HTTP for every engine, one engine at a time on an otherwise idle Mac, median of 5 runs, thinking off,
identical sampling settings, cold prompts cache-busted, each engine at its latest release and its own best settings).
Every table, with all columns and engines: [docs/benchmarks.md](docs/benchmarks.md).

## Third-party

MIT: nlohmann/json, minja (patched: Jinja `undefined` test), cpp-httplib; the tokenizer's Unicode ranges come from
the Unicode Character Database. Their notices: [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). Ideas credited in code
comments where they come from other engines; the code is Kapitsa's own.
