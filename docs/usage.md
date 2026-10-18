# Using Kapitsa

The reference for the `kapitsa` command and its server. To install Kapitsa, see the [README](../README.md); to build it
from source, [CONTRIBUTING.md](../CONTRIBUTING.md); for how the engine works, [architecture.md](architecture.md).

## Commands

| Command | What it does |
| --- | --- |
| `kapitsa serve [--model M] [options]` | Starts the server: the chat page at `http://127.0.0.1:8000`, the APIs under `/v1`. Without `--model`, a terminal asks which model ([Which model](#which-model)); elsewhere it is the best model for this Mac ([Server options](#server-options)). |
| `kapitsa claude`, `kapitsa codex`, `kapitsa opencode`, `kapitsa pi` | Starts that coding agent on Kapitsa ([Coding agents](#coding-agents)). |
| `kapitsa pull M... [--from DIR]...` | Downloads models without starting them. `--dry-run [--json]` shows what `M` resolves to and downloads nothing. |
| `kapitsa list [--json]` | The downloaded models: size on disk, complete or not, MTP head and draft model. |
| `kapitsa rm M...` | Deletes downloaded models. Folders outside the store are left alone. |
| `kapitsa import FILE.gguf [options]` | Converts a GGUF file into a model folder ([GGUF files](#gguf-files)). |
| `kapitsa doctor [--model M]... [--json]` | This Mac and which models fit it ([Memory and `kapitsa doctor`](#memory-and-kapitsa-doctor)). |
| `kapitsa bench [--model M] [--quick] [--submit]` | Benchmarks Kapitsa on this Mac with the published protocol, and adds the result to the community results when asked ([Benchmarking this Mac](#benchmarking-this-mac)). |
| `kapitsa --help`, `kapitsa COMMAND --help`, `kapitsa --version` | Help and version. |

Exit codes: 0 success, 1 failure (download, port, model), 2 usage error (unknown option or model), 130 after Ctrl+C in a
download (the parts are kept) or when the model question is cancelled; the coding-agent commands exit with the agent's
code.

### Which model

Run in a terminal without `--model`, `kapitsa serve` and the coding-agent commands ask which model to start: the models
with their download size, ✓ where downloaded and ★ on the best for this Mac, then the chosen model's precisions with
their sizes and whether each fits this Mac's memory (the engine's own plan of its headers; a model or precision that
does not fit is marked too big and can still be chosen, with a warning). ↑ and ↓ (or j and k) move, Enter chooses, Esc
goes back or cancels, Ctrl+C cancels. A model not downloaded yet says how much it downloads, then downloads as usual.

The first choice is the last one, so Enter twice starts the same model again. It is kept in
`~/.cache/kapitsa/state.json` (`KAPITSA_HOME`), and a `--model` that names a catalog model, typed in a terminal, updates
it too; with no last choice, the best model for this Mac comes first. A coding-agent command that finds a server
already running on its port uses it and says so (stop it to pick another model).

Nothing is asked where nobody could answer: when stdin, stdout or stderr is not a terminal (the Nace Agent app,
scripts, `brew services`), with `TERM=dumb`, `CI` or `KAPITSA_NO_PROMPT=1` set, with `--yes`, or in a background job.
Those start the best model for this Mac, with the same output as before.

## Models and downloads

| Name | Aliases | What it is | Download | Variants |
| --- | --- | --- | ---: | --- |
| `qwen3.6-35b-a3b` | `35b`, `qwen3.6-35b`, `35b-a3b` | Qwen3.6, mixture of experts with 3B parameters active per token: quick to reply and to read | 22.1 GB | `4bit` (default), `5bit`, `6bit`, `8bit`, `bf16` |
| `qwen3.8-27b` | `27b` | Qwen3.8, dense: the strongest coder of the three Qwen models, slower to write | 20.2 GB | `4bit` (default), `5bit`, `6bit`, `8bit`, `bf16` |
| `qwen3.5-4b` | `4b` | Qwen3.5, dense and small: for 16-24 GB Macs and simple tasks | 3.3 GB | `4bit` (default), `6bit`, `8bit`, `bf16` |
| `nemotron-3.5-bolt` | `bolt`, `nemotron-bolt` | NVIDIA Nemotron 3.5 Bolt 74B-A9B: hybrid Mamba-2 mixture of experts, 9B active, 2-bit experts; an agentic model | 25.1 GB | `24gb` |
| `nemotron-3.5-super-vl` | `super-vl`, `super`, `nemotron-super` | NVIDIA Nemotron 3.5 Super-VL 120B-A12B: hybrid Mamba-2 mixture of experts, 12B active, 4-bit (NVFP4) experts; needs a 96 GB+ Mac | 76.4 GB | `nvfp4` |
| `nemotron-3.5-lightning` | `lightning`, `nemotron-lightning` | NVIDIA Nemotron 3.5 Lightning 30B-A3B: hybrid Mamba-2 mixture of experts, 3B active, 4-bit (NVFP4) experts; quick to reply and to read on 32 GB Macs and up | 22.9 GB | `nvfp4` |

A download includes the companion files that speed decoding up: an MTP head and a DFlash2 draft model for the 35B and
27B, an MTP head for the 4B, NVIDIA's LiLiCorr draft model for Bolt and for Super-VL, its DSpark draft model for
Lightning.

`--model` takes a name or alias (`35b`), one with a variant (`35b:6bit`), a Hugging Face repo (`org/name[@revision]`,
see [Other models](#other-models)), a folder (a path starting with `/`, `~`, `./` or `../`), a `.gguf` file
([GGUF files](#gguf-files)) or the name of a folder in the store.

The first start of a model downloads it; later starts use the copy on disk, offline too. A catalog model that would not
start on this Mac is refused before the download, naming one that fits. Without `--model`, `kapitsa serve` takes the
first of `35b`, `27b` and `4b` that holds one conversation of `--ctx` tokens within this Mac's memory budget (a model
not downloaded yet is planned from its file headers, a few hundred KB).

The API calls the model by its name (`qwen3.5-4b`), as `name:variant` for a variant other than the default
(`qwen3.6-35b-a3b:6bit`), or by its repo id or folder name; `--model-id` sets another. The server runs one model, so a
request's `model` field is not checked.

### Precisions

Each Qwen model comes in several precisions, its variants. Download sizes, companions included:

| Variant | Weights | `35b` | `27b` | `4b` |
| --- | --- | ---: | ---: | ---: |
| `4bit` (default) | MLX affine 4-bit, group 64: the fastest | 22.1 GB | 20.2 GB | 3.3 GB |
| `5bit` | MLX affine 5-bit, group 64 | 26.4 GB | 23.5 GB | - |
| `6bit` | MLX affine 6-bit, group 64: the quality choice | 30.7 GB | 26.9 GB | 4.3 GB |
| `8bit` | MLX affine 8-bit, group 64 | 39.4 GB | 33.6 GB | 5.4 GB |
| `bf16` | unquantized, where it fits | 71.9 GB | 58.8 GB | 9.3 GB |

Each step up stays closer to the original weights, decodes slower and needs more memory; `kapitsa doctor` shows which
variants fit this Mac. Every variant keeps the model's MTP head and draft model, and pins its repo to a revision, so a
download is always the same bytes. After a Kapitsa update that pins newer revisions, the next start fetches only the
files that changed (offline, it keeps using the copy it has). Bolt has one variant, `24gb`: NVIDIA's own
mixed-precision checkpoint (IQ2_XXS experts, NVFP4 Mamba layers, Q8_0 attention, latent and head, BF16 embedding).
Super-VL has one variant, `nvfp4`: NVIDIA's own checkpoint as stored (NVFP4 routed and shared experts, FP8 Mamba
projections, attention and head, BF16 latent projections, which load as 8-bit, router and embedding).

### Other models

Any Hugging Face repo of a Qwen3.5-family model (Qwen3.5, Qwen3.6, Qwen3.8) in MLX format runs: bf16, or MLX affine 3,
4, 5, 6 or 8-bit with groups of 64 (4-bit also 32), mixed recipes included. Kapitsa reads the repo's `config.json`
before any weights, then downloads its top-level files except other formats (`.gguf`, `.bin`, images and the like). A
catalog variant's repo (`mlx-community/Qwen3.6-35B-A3B-6bit`) comes with that variant's MTP head and draft model; other
repos use the MTP head they contain, if any.

A folder needs `config.json`, the `*.safetensors` weights and `tokenizer.json`, and may hold an MTP head in
`mtp.safetensors`. The official Hugging Face checkpoints of the dense models run as well.

Refused, with a message: other architectures, other quantizations (MLX's `mxfp4`, `nvfp4` and `mxfp8` modes among
them; NVIDIA's own NVFP4 checkpoints of the catalog's Nemotron models are read as they are), mixture-of-experts
checkpoints in the Hugging Face layout, and weights larger than this Mac's memory budget.

### GGUF files

`kapitsa import FILE.gguf` converts a GGUF file once into a model folder in the store and prints its name for
`kapitsa serve --model`; `kapitsa serve --model FILE.gguf` and the coding-agent commands convert on first use and reuse
the conversion after that. It reads the GGUF architectures `qwen35` and `qwen35moe` (Qwen3.5, 3.6 and 3.8) as LM Studio,
Ollama (its blob files) and Hugging Face GGUF repos ship them; a split file is named by any of its parts.

The MTP head comes from the file, an `mtp-*.gguf` beside it, an Ollama model's draft layer, or `--mtp FILE`. Options:
`--name NAME` (the folder; default: from the file name), `--threads N` (default: the performance cores, at most 8),
`--force` (convert again over an existing folder), `--dry-run` (show the formats, size and folder, write nothing).

| GGUF type | Converted to |
| --- | --- |
| BF16, F16, F32 | bf16 |
| Q4_0, Q4_1, Q4_K | 4-bit group 32, the file's own codes |
| Q8_0, Q6_K | 8-bit group 64 |
| Q5_0, Q5_1, Q5_K | 6-bit group 64 |
| IQ4_NL, IQ4_XS, MXFP4, NVFP4 | 5-bit group 64 |
| IQ3_XXS, IQ3_S, Q3_K | 4-bit group 32 |
| IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M, Q2_K | 3-bit group 64 |

In a mostly 4-bit-or-better file (unsloth's UD mixes, for example), the 1-3-bit tensors take the file's majority format.
The experts of mixture-of-experts layers take 5-bit group 64 from 4-bit types, 4-bit group 64 from 3-bit and 3-bit group
64 from 1-2-bit types. Refused: the ternary TQ1_0 and TQ2_0, Hadamard-rotated weights, Qwen3-Next, other architectures.

### Where models are stored

Models live in `~/.cache/kapitsa/models`; `KAPITSA_HOME` moves `~/.cache/kapitsa`. Folders are named `<name>-<variant>`
for catalog models (`qwen3.6-35b-a3b-4bit`), `org--name` or `org--name@revision` for repos, and by `--name` for GGUF
conversions. Inside, `.kapitsa/` holds the parts of files being downloaded and the manifest, written last.

- Every file is checked after download against its sha256, or its git blob id for small files; a mismatch deletes it
  and says so. Ctrl+C keeps what has arrived (a second one quits at once) and the same command continues. A download
  needs what is left plus 1 GB of free disk; a finished one needs no network access.
- Files already on this Mac are used instead of downloaded, as APFS clones (no extra space; copies across volumes): the
  Hugging Face cache (`HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`), folders given to
  `kapitsa pull M --from DIR` (matched by checksum, up to 3 levels deep) and, for a catalog model, the Nace Agent app's
  copy in `~/.nace/models`.
- A catalog model's folder that Kapitsa did not download (copied in by hand, say) is adopted: its files are checked
  against the pinned checksums, kept where they match, and only the others fetched. Until then `kapitsa list` shows it
  as not checked yet.
- `HF_TOKEN`, or the token `hf auth login` saved, is sent for gated or private repos; `HF_ENDPOINT` points at a mirror.

`kapitsa list` shows the store and the variants not downloaded yet; `kapitsa rm 35b:6bit` deletes a variant
(`kapitsa rm 35b`: the default one, or the only one downloaded).

## Server options

`kapitsa serve` checks the port, downloads the model if needed and loads it, then prints `Ready:` and the chat and API
addresses. In a terminal the engine's log goes to `~/.cache/kapitsa/logs/serve-PORT.log` (`--verbose` shows it on
screen too), otherwise to stderr; Ctrl+C stops it.

| Option | Default | Effect |
| --- | --- | --- |
| `--model MODEL` | the best model for this Mac | A name, `name:variant`, `org/name[@revision]`, a folder or a `.gguf` file ([Models and downloads](#models-and-downloads)). |
| `--host HOST` | `127.0.0.1` | Listen address; `0.0.0.0` serves your network ([API keys and network access](#api-keys-and-network-access)). |
| `--port PORT` | `8000` | Listen port. A port in use is an error. |
| `--api-key KEY` | none | Require `Authorization: Bearer KEY` or `x-api-key: KEY`. |
| `--model-id ID` | the model's name | The model's name in the API. |
| `--ctx TOKENS` | `32768` | Context per conversation. Its KV cache grows as the conversation fills. |
| `--slots N` | `4` where memory allows, else `3` or `2` | Requests that run at once, each with its conversation's cache kept for follow-ups; more wait for a free slot, first come first served. |
| `--dflash auto\|none\|DIR` | `auto` | Draft model: `auto` uses the one downloaded with the model, unless it would keep one `--ctx` conversation from fitting; `none` decodes without it (with the MTP head on Qwen); `DIR` names a drafter folder. |
| `--spec N` | `1` (`2` on the 27B) | MTP draft depth when no draft model is used; `0` turns speculative decoding off. |
| `--mem-budget GB` | macOS's recommended working set, a little less | GPU memory Kapitsa may hold ([Memory and `kapitsa doctor`](#memory-and-kapitsa-doctor)). |
| `--prefill-chunk N` | `2048` (`1024` when memory is short) | Prompt tokens per prefill pass. |
| `--image-max-pixels N` | `1200000` (Bolt and Super-VL `1400000`) | Larger images are resized to about N pixels; `0` = the model's own limit ([Images](#images)). |
| `--verbose` | off | In a terminal, show the engine's log on screen too. |
| `--yes` | off | In a terminal without `--model`, start the best model for this Mac instead of asking ([Which model](#which-model)). |

Requests sent at once (a coding agent's parallel subagents and side requests, several clients) each run on a slot of
their own and take turns on the GPU between prompt chunks and decode steps: a request sent during a long reply starts
after a fraction of a second instead of after that reply, and each request gets the tokens it would get alone. The GPU
is shared, not multiplied: two replies at once write about as many tokens per second in all as one reply alone, each at
about half the speed. Without `--slots` the server takes 4 slots where they fit as well as 2 would (room for one full
`--ctx` conversation, the same draft model and prefill chunk), else 3, else 2; `GET /v1/models` reports them. With
every slot busy a request waits for one (the log says so); a streamed chat completion whose client disconnects stops
waiting.

## APIs

OpenAI clients use the base URL `http://127.0.0.1:8000/v1`, Anthropic clients `http://127.0.0.1:8000`.

| Endpoint | What it is |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions: `reasoning_content`, `tool_calls`, and `usage` (in a stream's last chunk) with `prompt_tokens_details.cached_tokens`. `max_tokens` defaults to 16384. |
| `POST /v1/responses` | OpenAI Responses: reasoning items, function and custom tools, `usage.input_tokens_details.cached_tokens`. `previous_response_id` continues one of the last 64 stored responses (unknown ids are a 400). Built-in tools such as web search are left out. |
| `POST /v1/messages` | Anthropic Messages: thinking blocks, `tool_use` and `tool_result`; `usage.input_tokens` counts the uncached prompt, `cache_read_input_tokens` the cached part. Server tools are left out. |
| `POST /v1/messages/count_tokens` | Anthropic's token count of a request. |
| `GET /v1/models` | The model being served (OpenAI's list with Anthropic's fields); `context_fit` is the tokens one conversation can hold within the memory budget, `slots` the requests that run at once. |
| `GET /health` | `{"status":"ok"}` |
| `GET /` | The chat page: one file that talks only to this server and keeps its conversations in the browser. |

Every API streams (server-sent events) or answers in one JSON reply. A prompt longer than the context is a 400 error
(Messages: "prompt is too long", which Claude Code answers by compacting; Responses: `context_length_exceeded`).

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "qwen3.5-4b",
  "messages": [{"role": "user", "content": "Explain speculative decoding in two sentences."}], "stream": true}'
curl http://127.0.0.1:8000/v1/messages -H 'Content-Type: application/json' -d '{"model": "qwen3.5-4b",
  "max_tokens": 1024, "messages": [{"role": "user", "content": "Write a haiku about unified memory."}]}'
```

### Thinking

The models think before they answer. Chat Completions and Responses think by default; Messages follows Anthropic's API
and thinks only when the request asks for it.

| API | Thinking off | Thinking on | Effort |
| --- | --- | --- | --- |
| Chat Completions | `"reasoning_effort": "none"`, or `"chat_template_kwargs": {"enable_thinking": false}` | by default | `reasoning_effort` |
| Responses | `"reasoning": {"effort": "none"}` | by default | `reasoning.effort` |
| Messages | by default, or `"thinking": {"type": "disabled"}` | `"thinking": {"type": "enabled"}` or `"adaptive"` | `output_config.effort` |

Effort levels: `none` and `minimal` turn thinking off; `low` and `medium` go to the chat template, where Qwen3.8 uses
them; `high`, `xhigh` and `max` keep the template's default, Qwen3.8's highest level. The other models think the same at
every level. In Messages, `"thinking": {..., "display": "omitted"}` returns thinking blocks without their text.

### API keys and network access

The server listens on `127.0.0.1` and needs no key: any key a client sends is accepted. With `--api-key KEY`, every
request must carry `Authorization: Bearer KEY` or `x-api-key: KEY`, or it gets a 401. `GET /health` and the chat page
need no key; the chat page asks for it once and keeps it in the browser. To serve other machines, listen on every
address and always set a key: `kapitsa serve --host 0.0.0.0 --api-key YOUR-KEY` (see [SECURITY.md](../SECURITY.md)).

## Coding agents

```bash
kapitsa claude --model 27b                    # Claude Code on the 27B
kapitsa claude -p "what does this repo do?"   # other arguments go to the agent
kapitsa codex exec "fix the failing test"
kapitsa opencode run "add a --verbose flag"
kapitsa pi -p "list the TODOs in this repo"
```

Each command checks that the agent is installed (else it says how to install it). If Kapitsa is serving on `127.0.0.1:P`
(`--port P`, default 8000), the agent uses that server, its model and its context (a `--model` naming another model is
an error). Otherwise the command downloads the model if needed, starts `kapitsa serve` with the largest context up to
128K tokens that one conversation can hold beside the model in this Mac's memory (`--ctx N` sets it), logging to
`~/.cache/kapitsa/logs/serve-P.log`, and stops it when the agent exits or the command is killed.

Without `--model` in a terminal, the command asks which model to start ([Which model](#which-model)); `--yes` skips
the question. `--api-key K` is the server's key, if it has one. All other arguments go to the agent, and everything after
`--` too (for the agent's own `--model`, `--port` or `--help`). Ctrl+C and Ctrl+Z reach the agent, and the command exits
with its exit code. Each agent is set up for that run only (its configuration files are not changed), and its other
tools and MCP servers work as usual:

- **Claude Code** uses the Anthropic Messages API, with the model for all of its model choices and its context window
  and compaction set to the server's context. Web search is disabled (Claude Code runs it on Anthropic's servers). The
  key goes through an API key helper, so a saved Console key or claude.ai login is never sent to Kapitsa and stays as
  it is (Claude Code notes that the helper overrides it).
- **Codex** uses the Responses API through a `kapitsa` model provider, with the model's context window and compaction
  at 90% of it. Web search is disabled (Codex runs it on OpenAI's servers), the model's reasoning is shown, and your own
  `-c` options still apply.
- **opencode** uses a `kapitsa` provider, OpenAI-compatible on `/v1`, with reasoning on, the context and output limits
  and effort variants, for its main, small and built-in agents, merged into your `OPENCODE_CONFIG_CONTENT`. From
  opencode 2 on, it runs with `--standalone` unless you pick a server.
- **Pi** uses the Anthropic Messages API through a `kapitsa` provider, with the server's model and context window. An
  extension adds the provider for that run (`pi -e ~/.cache/kapitsa/agents/pi-kapitsa.js`); `~/.pi` is not changed.
  Each turn's reasoning goes back as it was generated (in the thinking block's signature, a turn cut short included),
  so every request of a tool loop continues the prompt the server has cached; Pi's thinking levels become the model's
  reasoning efforts. Install it with `npm install -g @earendil-works/pi-coding-agent`.

## Images

Every catalog model except `nemotron-3.5-lightning` (a text-only model) reads images, in every precision:
`qwen3.5-4b`, `qwen3.6-35b-a3b`, `qwen3.8-27b`, `nemotron-3.5-bolt` and `nemotron-3.5-super-vl`. Their vision towers come with the download and load with the first
image: about 0.67 GB of GPU memory for the 4B's, 0.9 GB for the 35B's and the 27B's (bf16 in every precision), 1.7 GB
for Bolt's and for Super-VL's. On a model without one, such as Lightning or a text-only conversion, each image becomes a short note
in the prompt, so agents that attach screenshots keep working.

| API | Image part |
| --- | --- |
| Chat Completions | `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` |
| Messages | `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}`, or a `url` source |
| Responses | `{"type": "input_image", "image_url": "data:image/png;base64,..."}` |

- Sources: `data:` URLs (base64) and `http(s)` URLs, in user, assistant and tool messages; up to 32 MB per image and
  30 s per download (`NI_IMAGE_MAX_MB`, `NI_IMAGE_TIMEOUT_S`). Formats: what macOS's ImageIO decodes (PNG, JPEG, WebP,
  GIF, HEIC, TIFF, BMP). Video is not supported (a 400 error).
- Size: images over 1.2 million pixels (Bolt and Super-VL: 1.4 million) are scaled down to about that; a 2880x1800 screenshot becomes
  1376x864. `--image-max-pixels N` sets the cap, `0` lifts it to the model's own limit.
- Tokens: an image costs prompt tokens by its size (a 1280x800 screenshot: 1000 tokens), in `usage` and `count_tokens`.
- Cache: the same image sent again (an agent's screenshot every turn) reuses the cached prompt, and the embeddings of
  recent images are kept (`NI_IMAGE_CACHE_MB`, default 256).

## Memory and `kapitsa doctor`

Kapitsa holds at most a budget of GPU memory: Metal's recommended working set, which macOS sizes from the RAM, less 4%
(at least 512 MB) for the Neural Engine service and other GPU users; `--mem-budget GB` (or `NI_MEM_BUDGET`) sets it.
Fixed at start are the weights, the MTP head, the prefill buffers (they scale with `--prefill-chunk`), each slot's
recurrent state and, where used, the Neural Engine split's buffers and the draft model. Each conversation's KV cache
grows as it fills: 36 KB per token on the 4B, 68 KB on the 27B, 22 KB on the 35B, 8 KB on Bolt and on Super-VL, 6 KB on Lightning. When memory is short:

- At start, the fixed part and an 8K-token KV cache must fit, else the server exits with `insufficient memory: ...`,
  what each part needs and what would help (fewer `--slots`, `--prefill-chunk 1024`, no draft model, a smaller model,
  `--mem-budget`). `kapitsa serve --model NAME` checks this before downloading.
- Where the draft model or 2048-token prefill chunks would keep one `--ctx` conversation from fitting, the engine
  leaves the draft model out or uses 1024-token chunks.
- A prompt that needs more room takes it from cached image embeddings, then from idle slots, least recently used first
  (their conversations are read again when they return), then from the Neural Engine split (prefill then runs on the
  GPU alone) and last from the image encoder; failing that, the request gets an `insufficient memory` error. With
  other requests in flight, a request that fits only once one of them ends waits for it, and only a prompt that does
  not fit even alone gets the error; a reply in progress may take the cache of a request still waiting (that request
  then reads its prompt again). A reply that cannot grow its cache ends with finish reason `length`.

`kapitsa doctor` shows this Mac (chip, GPU cores, RAM, macOS, the working set, Kapitsa's budget, the Neural Engine) and,
per model, its memory part by part, whether it starts, the tokens of KV cache that fit, its decoding (draft model, MTP
or plain), the image encoder's size and a rough speed estimate; then each catalog variant, whether it fits and which one
`--model NAME` takes. It assumes `--ctx 65536` and, per model, the slots the server would start with (the catalog check:
2; `--ctx`, `--slots`, `--prefill-chunk`, `--dflash` and `--mem-budget` change that); `--model M`, repeatable, checks
other models, downloaded or not; `--json` prints the same for programs (`starts`, `fits_full_context`,
`kv_tokens_in_budget`, `decoding`, ...).

What fits, for the 4-bit variants at `--ctx 65536 --slots 2`, with the tokens of KV cache that fit in all. macOS sets
the working set at about 2/3 to 78% of the RAM, depending on its version, hence the ranges:

| Mac RAM | Budget | `4b` | `27b` | `35b` |
| --- | --- | --- | --- | --- |
| 16 GB | 11.0-12.9 GB | yes (198K-249K) | no | no |
| 24 GB | 16.5-19.3 GB | yes (347K-424K) | only at the larger budget (23K) | no |
| 32 GB | 22.0-25.7 GB | yes (497K-598K) | yes (62K-80K) | yes (50K-153K) |
| 48 GB | 33.0-38.6 GB | yes (795K-947K) | yes (184K-265K) | yes (475K-724K) |

On a 48 GB Mac the 35B also fits in 6-bit with a full conversation and its draft model, in 8-bit only with shorter ones.
For Bolt and the other variants, run `kapitsa doctor`.

Super-VL's weights alone are 70.1 GB, so it needs a 96 GB+ Mac: on 128 GB (a 110.8 GB budget) it runs with its draft
model (3.5 GB at two slots) and room for millions of tokens of KV cache; on 96 GB it starts without the draft model
(about 29K tokens of KV cache in all at a 72 GB budget) or with it from an 80 GB budget; on 48 and 64 GB Macs it is
refused before the download (`... does not fit this Mac: its weights alone (70.1 GB, ...) exceed ...`).

Lightning's weights are 18.2 GB (its MTP head is not loaded): on 16 GB Macs and on 24 GB at the smaller budget it is
refused before the download, on 24 GB at 19.3 GB it starts without its draft model (79K tokens of KV cache in all at
`--ctx 65536`), on 32 GB it runs with the draft model (0.7 GB at two slots) and 370K-970K tokens of KV cache, and from
48 GB two conversations of its full 1M-token context fit.

## Benchmarking this Mac

`kapitsa bench` measures Kapitsa on this Mac the way the [published benchmarks](benchmarks.md) measured it, so its
numbers compare with them and with the [community results](../bench/community/RESULTS.md) from other Macs.

```bash
kapitsa bench                     # in a terminal it asks which model; elsewhere the best model for this Mac
kapitsa bench --model 35b:6bit    # a given model and precision
kapitsa bench --quick             # a shorter run: 3 runs per writing test, the 4K prompt only
kapitsa bench --submit            # then add the result to the community results (it asks first)
```

It runs the protocol of [bench/harness](../bench/harness/README.md) (`bench.py`), request for request, as a client
of the local API: it starts the model's server (`kapitsa serve` with one conversation slot and a 32K context, `NI_*`
variables left out; downloading the model first if needed), warms it up, waits until other apps leave the GPU alone
for 15 seconds (5 minutes at most, 2 for `--quick`), then measures:

| Measurement | How |
| --- | --- |
| Writing | A 512-token coding answer, at temperature 0.6 and greedy, 5 runs each (3 with `--quick`), median; the rate over the middle of the stream |
| Reading | A cold prompt of about 4K and 16K tokens (the 4K one with `--quick`): prompt tokens per second to the first token |
| Follow-up | Three new messages in that conversation, 1 s apart: the median time to the first token |
| Peak memory | The server's `phys_footprint_peak`, what macOS's `footprint` shows (lower in a quick run, which holds no 16K conversation) |
| Tool calls | Streamed and whole replies with two tools, thinking off and on, and a round trip |

While it measures, it samples every process's GPU time, the thermal state and the power source every 2 seconds, and
flags what could skew a number: other apps using more than 5% of the GPU for three samples in a row or on average
(the measurement is marked), the battery, Low Power Mode, a serious thermal state, swapping. macOS's photo and media
analysis and the terminal showing the run count as ambient load: recorded, never flagged.

The result, about 2 KB of JSON ([bench/community/schema.json](../bench/community/schema.json)), goes to
`~/.cache/kapitsa/bench/` (`--out FILE` puts it elsewhere; `--json` also prints it). It records the Mac (its model
identifier with Apple's name and year, chip, CPU and GPU cores, memory, macOS version and build), Kapitsa's version, the
model, its precision and how it decoded, the numbers and the conditions; no serial number, host name, user name or
path. A run takes 3 to 15 minutes and holds the GPU lock that the benchmark harness uses (`~/.cache/nace-gpu.lock`), so
it never runs beside a harness run.

`--submit`, or yes to the question at the end of a run in a terminal, adds the result to the community results. It
shows the exact JSON first and sends it only after a yes typed at the terminal: with [gh](https://cli.github.com)
logged in, as a pull request from your fork (it uses your gh login and nothing else); without gh, by opening the
"Benchmark result" issue form in the browser with the JSON filled in, for you to submit. `kapitsa bench submit [FILE]`
sends a saved result (the newest by default), and `--submit --dry-run` prints the gh commands or the issue's address
without sending anything. Only results of the catalog's models are taken, at any precision. How results are checked:
[bench/community](../bench/community/README.md).

## Running in the background

Homebrew arrives with the first release. `brew services start kapitsa` then runs `kapitsa serve` (the best model for
this Mac, on `127.0.0.1:8000`) in the background and at each login, restarting it after an error;
`brew services stop kapitsa` stops it. Its log, the first download's progress too: `/opt/homebrew/var/log/kapitsa.log`.

## Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `KAPITSA_HOME` | `~/.cache/kapitsa` | Kapitsa's folder: models in `models/`, the coding-agent servers' logs in `logs/`, `kapitsa bench`'s results in `bench/`, the last model chosen in `state.json`. |
| `KAPITSA_NO_PROMPT` | | `1`: never ask which model to start; without `--model`, the best model for this Mac ([Which model](#which-model)). |
| `BENCH_GPU_LOCK` | `~/.cache/nace-gpu.lock` | `kapitsa bench`: the GPU lock it takes, the benchmark harness's; `NACE_GPU_LOCK_HELD=1` when the caller holds it. |
| `BENCH_GPU_AMBIENT` | | `kapitsa bench`: more process names whose GPU use is ambient (recorded, never flagged), comma-separated, as for the harness. |
| `KAPITSA_BENCH_REPO` | `nace-ai/kapitsa` | `kapitsa bench --submit`: the repository results go to. |
| `HF_TOKEN` | the token `hf auth login` saved | Hugging Face token for gated or private repos. |
| `HF_ENDPOINT` | `https://huggingface.co` | A Hugging Face mirror. |
| `HF_HOME`, `HF_HUB_CACHE` | `~/.cache/huggingface`, `$HF_HOME/hub` | The Hugging Face cache, whose files downloads reuse. |
| `NI_LOG` | `1` | `0` quiet, `1` a line per request with its timings, `2` debug. |
| `NI_MEM_BUDGET` | | The GPU memory budget in GB, as `--mem-budget`. |
| `NI_ANE` | on where supported | `0` keeps prefill on the GPU, without the Neural Engine. |
| `NI_DFLASH` | `auto` | As `--dflash` when the option is absent: `auto`, `none` or a folder. |
| `NI_TURN_MS` | `30` | Requests at once: how long a decoding request keeps the GPU before it gives its turn to a waiting one, in ms; `0` = after every decode step. |
| `NI_HISTORY_PREFILL` | on | After a reply, while the server is idle, prepare the conversation's next turn so a follow-up starts sooner; `0` turns it off. |
| `NI_IMAGE_MAX_MB` | `32` | The largest image taken, in MB. |
| `NI_IMAGE_TIMEOUT_S` | `30` | The time limit for downloading an `http(s)` image, in seconds. |
| `NI_IMAGE_CACHE_MB` | `256` | GPU memory kept for the embeddings of recent images. |
| `NI_NH_BF16` | off | Nemotron Super-VL: `1` keeps its BF16 latent projections in BF16 instead of 8-bit (exact; about 4% slower decoding). |
| `NI_NH_ATTN_Q8` | off | Nemotron Lightning: `1` stores its BF16 attention projections in 8-bit (about 3% faster decoding, slightly less accurate). |
| `NI_DSPARK_DYN` | `0.2` | Nemotron Lightning's DSpark draft model: how many of its 7 drafts are verified, chosen per cycle as for `NI_LILI_DYN`; `0` verifies all 7. |
| `NI_DSPARK_K` | `16` | Nemotron Lightning's DSpark draft model: the candidates per draft position its Markov head rescores (1-32). |
| `NI_LILI_DYN` | `0.04` | Nemotron's LiLiCorr draft model: how many of its 7 drafts are verified is chosen per cycle, weighing their expected acceptance against each extra row's cost (this value, relative to a cycle's fixed cost); `0` verifies all 7. |
| `NI_PREWARM` | on | `0` creates the GPU pipelines on first use instead of at load. |

The other `NI_*` switches exist for development and A/B tests.

## Updating and uninstalling

Homebrew and the curl installer work from the first release on. To update:

```bash
brew upgrade kapitsa                                                                  # Homebrew
curl -fsSL https://raw.githubusercontent.com/nace-ai/kapitsa/main/install.sh | bash   # the installer, run again
git pull && ./install.sh                                                              # a source build, in its clone
```

To uninstall: `brew services stop kapitsa` if you started the service, then `brew uninstall kapitsa`; or delete the
binary `install.sh` placed (in `/opt/homebrew/bin`, `~/.local/bin` or your `--prefix`) and `../share/kapitsa` next to
that folder. Deleting `~/.cache/kapitsa` (or `$KAPITSA_HOME`) removes the models and logs.
