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
overrides), with a note when that folder is not on the PATH. It installs `LICENSE` and `THIRD_PARTY_NOTICES`
to `../share/kapitsa` relative to that binary folder, and prints the location. Run it again to
update. To uninstall, delete the binary, that notices folder and `~/.cache/kapitsa`.
`packaging/homebrew/kapitsa.rb` installs the same documents in Homebrew's package share directory. It is a
formula for a public release: it needs a tag, the tarball's sha256 and a tap (see the file); nothing is published.

## Commands

| Command | What it does |
| --- | --- |
| `kapitsa serve [--model M] [options]` | Starts the server (below). `M`: a catalog name or alias with an optional `:<variant>`, `org/name[@revision]`, or a folder; downloads it first if needed. No `--model`: the first catalog model (best first: 35B-A3B, 27B, 4B) whose memory plan, the engine's own (`conversation_fits`: one conversation of `--ctx` tokens beside the model, its buffers, the slots' state and the Neural Engine split, the test plan_load applies before it keeps a drafter), fits this Mac's budget; a model not downloaded yet is planned from its headers (`hub::plan_dir`). An explicit catalog model the engine would refuse (`start_need` over the budget) is refused before its download, naming the one that fits. |
| `kapitsa pull M... [--from DIR]...` | Downloads without serving. `--dry-run [--json]` prints what `M` resolves to (no network). |
| `kapitsa list [--json]` | The models in the store: size on disk, complete or not, MTP head / DFlash2 drafter. |
| `kapitsa rm M...` | Deletes downloaded models (only folders in the store). |
| `kapitsa doctor [--model M]... [--json]` | This Mac and what fits it (below). |
| `kapitsa claude\|codex\|opencode [--model M] [--port P] [--ctx N] [--api-key K] [args]` | Starts that coding agent on Kapitsa (Coding agents, below). Other arguments, and all after `--`, go to the agent. |
| `kapitsa mtp-sidecar SRC OUT [--revision R] [--hidden post_norm\|pre_norm]` | `tools/mtp_sidecar.py` in C++, same bytes (see below). |
| `kapitsa import FILE.gguf [--name N] [--mtp F] [--threads N] [--force]` | Converts a GGUF of a Qwen3.5-family model once into a store folder (Weight formats, GGUF files). `--model FILE.gguf` (serve and the agent launchers) does it on first use. |
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
| `--api-key <key>` | Require `Authorization: Bearer <key>` or `x-api-key: <key>` (except `GET /`, `/favicon.ico` and `/health`). |

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
calls returned as OpenAI `tool_calls`; `usage` in the final chunk), `POST /v1/responses` (OpenAI Responses),
`POST /v1/messages` and `POST /v1/messages/count_tokens` (Anthropic Messages), `GET /v1/models` (OpenAI's list with
Anthropic's fields; also `context_fit`: the tokens one conversation can hold within the memory budget), `GET /health`.
Thinking: `chat_template_kwargs.enable_thinking=false` or `reasoning_effort: "none"`; a `reasoning_effort` (or
Responses `reasoning.effort`, Anthropic `output_config.effort`) of `low` or `medium` goes to the template, and
`high`, `xhigh`, `max` leave its default (Qwen3.8's highest level; its template raises on `high`).

The Anthropic and Responses APIs are translations onto the same chat requests (`src/api/anthropic.cpp`,
`src/api/responses.cpp`; the server only wires the routes):

- **Anthropic Messages.** `system` (text blocks; Claude Code's `x-anthropic-billing-header` block, which changes with
  every request, is dropped) becomes the first system message; a system message inside `messages` stays where it is
  and is rendered as a system turn there (the engine does that for any later system message: the Qwen3.5+ templates
  raise on one). `thinking` blocks become the assistant turn's `reasoning_content`, `tool_use` its tool calls,
  `tool_result` blocks tool messages before the rest of their turn (`is_error`: "Error: " first; `tool_reference`:
  the tool's name), `redacted_thinking` is dropped. Images become `image_url` parts (read by models with a vision
  tower, see Images; other models get a short note), plain-text documents their text, PDFs a short note. Tools need an
  `input_schema` (server tools are left out, with a log line); `tool_choice` `none` drops them; `tool_addition` /
  `tool_removal` blocks change the tool list. Thinking is on for `enabled` and `adaptive`, off for `disabled` and
  when absent, as Anthropic's API does: Claude Code sends `adaptive` with its agent loop and nothing with its short
  side requests (titles, state classifiers), which then answer without thinking first. `context_management` keeping
  every turn's thinking (`clear_thinking_20251015`, `keep: "all"`, which Claude Code sends) sets the template's
  `preserve_thinking`: every assistant turn is rendered with its reasoning, so each request's prompt continues the
  previous one, the generated reply included, and the engine prefills only the new messages. A reply's thinking
  block carries its reasoning in its `signature` (`kapitsa.r1.` + base64), so it comes back exact even with
  `display: "omitted"` (the thinking text is then empty, as requested). Stop reasons: `end_turn`, `max_tokens`,
  `model_context_window_exceeded` (the context ran out), `stop_sequence` (with the string), `tool_use`. `usage`:
  `input_tokens` = prompt minus cached, `cache_read_input_tokens` = cached. The stream's `message_start` goes out once
  the prompt is tokenized and matched against the cache (before the prefill) with those counts. A prompt longer than
  the context is a 400 `invalid_request_error` "prompt is too long: N tokens > M maximum", which Claude Code answers
  by compacting.
- **OpenAI Responses.** `instructions` and the leading system/developer messages become the system message; later
  ones are system turns in place. `function_call`, `custom_tool_call` and `reasoning` items join the assistant turn
  they belong to (a reply's calls stay one turn even when a client sends each call followed by its output: output
  item ids name their response); `*_output` items become tool messages. A reasoning item's text comes from its
  `encrypted_content` when it is ours (`kapitsa.r1.` + base64: Codex, with `store: false`, sends it back), else its
  `reasoning_text` content, else its summary. Tools: `function`; `custom` (freeform, e.g. Codex's `apply_patch`) as a
  function of one string parameter, `input`, whose call goes out as a `custom_tool_call`; `namespace` tools as
  `<namespace>__<name>` (calls go out with their `namespace`); built-in tools are left out with a log line.
  `reasoning.effort` `none`/`minimal` turns thinking off. `previous_response_id` continues one of the last 64 stored
  responses (an unknown id is a 400). Output: reasoning items (`content` `reasoning_text` and `encrypted_content`),
  messages, function or custom tool calls; a max_output_tokens cut (or the context running out) ends with
  `response.incomplete`. A prompt longer than the context is a 400 with code `context_length_exceeded`.
- **Streams.** The engine runs in its own thread and hands events to the connection's thread, so a request that
  fails before its first event still gets a plain HTTP error, and a stream gets a keepalive (Anthropic `ping`;
  Responses `response.in_progress` before any output item, an SSE comment after) after 15 s without an event, while
  the engine waits for another request or prefills a long prompt. After a reply the server prefills the history as
  for chat completions (`NI_HISTORY_PREFILL`).

`tools/api_test` checks the translation on the CPU (request mapping, the exact event sequences, the launchers'
settings and, with a model folder, tool loops through the chat template token for token);
`tests/sdk_check.py build/api_test` has the official `anthropic` and `openai` Python SDKs parse every recorded reply
(a venv with both is enough; no model).

### Coding agents

`kapitsa claude`, `kapitsa codex` and `kapitsa opencode` (`src/server/launch.cpp`; each client's settings in
`src/api/clients.cpp`) check that the agent is installed (else say how to install it), then use the Kapitsa
answering on `127.0.0.1:--port` (its model and context from `/v1/models`; `context_fit` when that is smaller) or
start one: the model is downloaded first if needed (its progress in the terminal), then `kapitsa serve --model M
--port P --ctx N` runs from the same binary in a session of its own, logging to `~/.cache/kapitsa/logs/serve-P.log`,
with `N` the largest context from 128K down in 8K steps for which one conversation fits the memory budget beside
the model (`conversation_fits`, the engine's own plan; Claude Code's system prompt and tools alone take 20-25K
tokens, so the 32K default is too small). It waits for `/health`, showing the server's last log line, then runs the
agent in the terminal's foreground process group (Ctrl+C and Ctrl+Z reach the agent; signals sent to the launcher
are passed on), stops the server it started when the agent exits, and exits with the agent's code. The server
also exits by itself if the launcher dies (`KAPITSA_PARENT_PID`, watched with kqueue). What each agent gets:

- **Claude Code**: `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL` and every `ANTHROPIC_DEFAULT_*_MODEL` /
  `ANTHROPIC_SMALL_FAST_MODEL` set to the model, `CLAUDE_CODE_MAX_CONTEXT_TOKENS` and `CLAUDE_CODE_AUTO_COMPACT_WINDOW`
  set to the context, `CLAUDE_CODE_MAX_OUTPUT_TOKENS` a quarter of a context under 128K,
  `CLAUDE_CODE_USE_BEDROCK/VERTEX/FOUNDRY=0`; `--model M --disallowedTools WebSearch` (Anthropic's servers run its
  search). The key goes through `--settings '{"apiKeyHelper": ...}'` (layered over the user's settings for this run),
  which prints `$KAPITSA_API_KEY` (the server's key, or "kapitsa"): Claude Code then sends that key, and only it, as
  `x-api-key` and as the bearer token. With `ANTHROPIC_AUTH_TOKEN` alone it also sent the user's saved Console key
  (from the keychain) in `x-api-key`; `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` are unset (the first asks
  whether to use a custom key, the second beside a helper adds a "Both ... set" warning). A user with a saved
  Console key sees one notice, "apiKeyHelper overriding saved Console key" (a claude.ai login: "... overriding
  Claude subscription login"); their key and login are untouched.
- **Codex**: `-c` overrides for a `kapitsa` model provider (`wire_api="responses"`, `env_key="KAPITSA_API_KEY"`),
  `model`, `model_context_window`, `model_auto_compact_token_limit` (90% of it), `web_search="disabled"` and
  `show_raw_agent_reasoning=true` (Qwen's reasoning is reasoning text); the user's own `-c` go with them, before
  any subcommand (a subcommand's list would replace the root's).
- **opencode**: `OPENCODE_CONFIG_CONTENT` (merged into any the user has) with a `kapitsa` provider
  (`@ai-sdk/openai-compatible` on `/v1`, reasoning on so `reasoning_content` goes back with each turn, the window,
  input and output limits, effort variants), the model for `model`, `small_model` and the built-in agents; from
  opencode 2, `--standalone` (its background service would not see the environment).

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

## Images

Models with a vision tower take images: the Qwen3.6-35B-A3B and Qwen3.8-27B packs include theirs (bf16, 0.89 /
0.92 GB). Send OpenAI `image_url` content parts (`{"type": "image_url", "image_url": {"url": ...}}`) with a `data:`
URL (base64; PNG, JPEG, WebP, GIF, HEIC, TIFF, BMP: whatever ImageIO reads) or an `http(s)` URL (downloaded with a
32 MB cap and a 30 s limit: `NI_IMAGE_MAX_MB`, `NI_IMAGE_TIMEOUT_S`). The engine takes the images from the chat
messages, so any API that maps its image blocks to these parts works; a part counts as an image exactly when the chat
template renders it as one (an `image` or `image_url` key, or `"type": "image"`, also with an Anthropic-style
`source`); the Anthropic and Responses APIs map their image blocks and `input_image` items to these parts. Images
may sit in user, assistant and tool messages. Video is not supported (a 400 that says so). On a model without a vision
tower an image becomes a short note in the prompt, so agents that attach screenshots keep working. `usage` and
`/v1/messages/count_tokens` count an image as its rows (a 1280x800 screenshot: 1000 tokens).

The shared part (`src/vision/`: `image.mm`, `encoder.*`, `embed.h`) fetches and decodes images, prepares them
through the model's encoder, expands each image's placeholder token to its embedding rows, keys those rows for prefix
reuse, loads the encoder on the first image that needs encoding and keeps it within the memory budget. Per model, an
`ImageEncoder` holds the processor and the tower (`qwen_vit.*` + `src/kernels/vit.metal` for the Qwen3.5 family: a
new model's encoder is one more class and one line in `open_encoder`), and the model applies its own positions
(`Qwen35::set_rope_positions`: Qwen's interleaved M-RoPE, `attn_prep_m`; a model whose attention takes no positions
needs nothing there).

- Processor: decoded as PIL's `convert("RGB")` decodes (no color management, alpha dropped, the EXIF orientation
  applied), resized by smart_resize (multiples of 32 within the model's 65536 to 16.7M pixels, `--image-max-pixels`
  lowers the maximum) with Pillow's bicubic resampling reproduced to the bit, normalized, cut into 16x16 patches over
  2 frames. `tests/vision/check_cpu.py` checks it against Pillow and Hugging Face's processors (bit-exact with the
  reference arithmetic; decoding exact for PNG, WebP, GIF, BMP, TIFF; JPEG goes through Apple's decoder, 0.6 levels
  from libjpeg on average, up to ~20 on edges).
- Tower: the bf16 prefill GEMMs (`qrf_f16_*`) with a fp32 residual stream, LayerNorm, GELU, 2D RoPE and a
  non-causal flash attention over the image's patches (head dim 72) in `vit.metal`; the MLP's 4304 columns padded to
  4352 with zeros at load. Its work goes to the GPU in command buffers of ~25 ms (long ones hold up the display).
  Scratch is allocated per image and freed after it.
- Prefill: the image rows take the encoder's bf16 rows instead of their placeholder's embedding (`Seq::embeds`,
  also for the MTP head's history pass); attention layers rotate image rows by (t, h, w) and the text after an image
  by its position less the image's saving (`Seq::rope_pos`, `rope_delta`; decode, verify, MTP and DFlash2 follow it).
  A text-only conversation runs exactly the kernels it ran before.
- Prefix reuse and checkpoints compare image rows by keys derived from the image's content (SHA-256 of its decoded
  pixels and size) and check every image in a reused prefix against the slot's, so an image never reuses another's
  state; the same image resent (an agent's screenshot every turn) reuses the cache. The embeddings of recent images
  are kept (`NI_IMAGE_CACHE_MB`, default 256) and `data:` URLs already seen are not decoded again.
- Memory: the encoder's weights and an image's scratch are allocated within the budget (other slots' KV caches and
  the Neural Engine split give way first if needed); when the KV cache needs the room the cached embeddings go first
  and the encoder's weights last (they load again with the next image).

Accuracy (`tests/vision/`): the processor's output equals mlx_vlm's bit for bit (and Hugging Face's PIL processor
within 1 ulp; its default torchvision processor resizes differently, by a level on 0.1-0.3% of values); M-RoPE
positions equal `get_rope_index` over 200 random prompts with 1-4 images; the tower's embeddings are within 0.24%
(relative L2) of a float32 reference, where mlx_vlm's bf16 tower is 3.7% off; next-token distributions over answers
about a chart, a terminal screenshot, a photo and two images at once agree with mlx_vlm to a mean KL of 0.002-0.010
(35B) and 0.0005-0.003 (27B), top-1 96-100% (text-only prompts: 0.0015-0.0045 and 0.0002-0.0003; part of the image
cases' divergence is mlx_vlm's own tower error).

Speed (M4 Pro, 35B): a 1280x800 screenshot (1000 tokens) encodes in 0.84 s, and its first answer token comes 1.7 s
after the request (2.3 s for the first image after the server starts: the encoder's 0.9 GB load in ~0.4 s); sent
again it is cached: 0.02 s. 1920x1080 (2040 tokens): 2.4 s to encode, 4.2 s to the first token; 2880x1800 (5040
tokens): 10.9 s and 15.5 s. The tower's attention spans all of an image's patches, so its cost grows with their
square: its GEMMs run at ~7 TFLOP/s and its attention at ~6.7 (59% and 37% of a 1280x800 encode), and
`--image-max-pixels` trades detail for time. Text-only requests are unchanged: the same outputs bit for bit
(teacher-forced log-probs on both models, greedy decodes, a multi-turn server conversation) at the same prefill and
decode speed.

`tools/vit_check` runs the pieces alone (decode, resize, smart_resize, processor output, M-RoPE positions, the
tower on raw pixels, kernel checks, encode timing); `tests/vision/ref_tower.py` is the float32 reference of the tower
(Hugging Face's, on the CPU), `tests/vision/ref_mlx.py` writes mlx_vlm's next-token distributions for
`ni_eval --image`, `tests/vision/check_server.py` checks answers, prefix reuse and errors through a running server.

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

### GGUF files

`kapitsa import` (src/gguf/) converts llama.cpp's `qwen35` / `qwen35moe` GGUFs (LM Studio, Ollama blobs with their
draft layer as the MTP head, Hugging Face GGUF repos; split files by any part) into a normal MLX-format folder, tensor
by tensor in 64 MB pieces: config.json from the metadata, tokenizer.json / tokenizer_config.json / chat_template.jinja
from `tokenizer.ggml.*` (control tokens special, user-defined ones not, padding left out), safetensors, and
mtp.safetensors from the file's MTP block or an `mtp-*.gguf` beside it. llama.cpp's transforms are undone: A_log from
`ssm_a` = -exp(A_log), DeltaNet value heads from tiled back to grouped order (qkv's V rows, z, a, b, conv1d's V
channels, A_log, dt_bias, out_proj's columns), conv1d to [C, 4, 1], fused gate_up experts split; RMSNorm weights stay
1 + w (MLX's convention too). Formats per GGUF type, with the error against the bf16 weights relative to the GGUF
type's own (1.00 = no added error; measured on Qwen3.5-0.8B / 4B / 3.6-35B-A3B tensors against Qwen's checkpoints):

| GGUF type | Kapitsa format | added error |
| --- | --- | --- |
| BF16 / F16 / F32 | bf16 | none (BF16), bf16 rounding (F16 / F32; norms bitwise as from the HF checkpoint) |
| Q4_0, Q4_1, Q4_K | 4-bit group 32, the GGUF's codes, scale and offset rounded to bf16 | 1.005 |
| Q8_0 | 8-bit group 64 | 1.67 of Q8_0's (tiny) error |
| Q6_K | 8-bit group 64 | 1.09 |
| Q5_0, Q5_1, Q5_K | 6-bit group 64 | 1.31 (8-bit: 1.02) |
| IQ4_NL, IQ4_XS, MXFP4, NVFP4 | 5-bit group 64 | IQ4_XS 1.27, NVFP4 1.07 |
| Q3_K, Q2_K (warning) | 4-bit group 32 | 1.20, 1.03 |
| MoE experts of 4-bit types | 5-bit group 64 (no group-32 MoE kernels) | 1.31 (4-bit group 64 would be 2.29) |

Re-quantization takes per group the best of the min-max grid, a symmetric grid and least-squares refits (all after
rounding scale and bias to bf16). A layer's experts share one format (from their least precise tensor), its shared
expert is 8-bit; a dense layer's gate and up share the finer of their formats. The MTP head is written as Kapitsa's
sidecar reads it (8-bit group-64 linears, numbered 4-bit group-32 experts). IQ1 / IQ2 / IQ3 (they need llama.cpp's
codebooks), ternary and Hadamard-rotated files are refused, as are Qwen3-Next and other architectures.

Checked on Qwen3.5-0.8B Q4_K_M (unsloth, with MTP), teacher-forced over bench/quality (4092 positions): Kapitsa on
the converted folder against llama.cpp on the GGUF (CPU) KL 0.0077, top-1 96.7%, perplexity 9.69 against 9.73; the
conversion alone (both weight sets run by llama.cpp in F16) KL 0.0030, top-1 98.0%, less than llama.cpp's own Q4_K
kernels move the same weights (KL 0.0052 against their F16 run). The converted tokenizer encodes bench/corpus.txt
and 300 other texts to the original tokenizer's ids, and the chat template renders as with jinja2.

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
- `src/vision/`: image input (Images above): `image.mm` (image parts, fetching, ImageIO decoding, digests),
  `encoder.*` (the encoder interface, lazy loading, caches, placeholder expansion, prefix keys), `embed.h` (what a
  model's prefill sees), `qwen_vit.*` (the Qwen3.5 processor and ViT); kernels in `vit.metal` (the ViT) and
  `mm.metal` (embedding rows, `attn_prep_m`).
- `src/engine/`: chat templates (minja), slots with prefix reuse and recurrent-state checkpoints at turn
  boundaries (a user query's outlives the tool responses after it; a new conversation takes an empty slot, else a
  one-off's, else the conversation idle longest, and a conversation whose request was still running when it arrived
  counts as continued: opencode sends its title request beside its first request); `output_parser.h`: streaming parser for
  reasoning / content / tool calls (`tools/parser_test <model dir>` checks it on real token ids, CPU only).
- `src/hub/`: the model catalog (`catalog.cpp`: the table of models and variants), `--model` resolution
  (`resolve.cpp`), downloads, reuse, verification, the manifest and the MTP sidecar builder (`fetch.cpp`), the
  NSURLSession transport (`http.mm`).
- `src/server/`: the command line and the HTTP server (`main.cpp`), the coding-agent launchers (`launch.cpp`), the chat
  page (`chat.html`).
- `src/api/`: the Anthropic Messages (`anthropic.cpp`) and OpenAI Responses (`responses.cpp`) translations onto the
  engine's chat requests and their stream writers, and each coding agent's settings (`clients.cpp`).
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
  <dir>` exact, `--dump-ref f` / `--ref f` top-K reference files; JSON lines per sequence and in total; `--image f`
  for sequences with images). `tools/vit_check`: the image pieces alone (Images above).
- `bench/quality/`: held-out token ids (C++, Python, English prose, a coding-agent chat with tool calls; 4 x 1024
  tokens per tokenizer) and the quantization measurement tools (Weight formats above; `README.md` there).
- `bench/prompts/`: token-id prompts for A/B runs (`make_prompts.py` regenerates them);
  `bench/shortpf/`: cached-turn / short-prompt TTFT through the server;
  `bench/ckpt/`: TTFT of a user query after scripted agent tool loops (`tool_loop_ttft.py`, `compare.py`),
  how turn boundaries are classified (`kind_probe.py`).
- `tools/api_test [model_dir]`: the API translations on the CPU (Server above); `tests/sdk_check.py
  build/api_test`: the official Python SDKs parse their replies.
- `tests/test_server.py`: end-to-end API checks against a running server (the chat page, chat completions,
  Anthropic Messages and Responses tool loops with the cache across them, their errors, tool calls, prefix reuse,
  thinking + tools; `--api-key KEY` adds the key checks), and with `--cli build/kapitsa` offline checks
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
| `NI_IMAGE_MAX_MB`, `NI_IMAGE_TIMEOUT_S` | 32, 30 | images: largest encoded image taken (MB), http(s) download limit (s) |
| `NI_IMAGE_CACHE_MB` | 256 | images: GPU memory kept for the embeddings of recent images (an image sent again is not encoded again) |
| `NI_VIT_ATTN` | `vit_attn_d72_s4` | the ViT's attention kernel (`vit_attn_d72_s{4,8}[r2]`: simdgroups per threadgroup, 8-query blocks per simdgroup) |
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
the Unicode Character Database. Two small parts of the Neural Engine code are adapted from Apache-2.0 code; their
notices are in the adapted files and [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). Keep both with source and binary
distributions; model weights retain their own licenses.
