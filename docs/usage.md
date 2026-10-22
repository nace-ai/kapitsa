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
| `kapitsa rm M...` | Deletes downloaded models and says how much space that gave back. Folders outside the store are left alone. |
| `kapitsa import FILE.gguf [options]` | Converts a GGUF file into a model folder ([GGUF files](#gguf-files)). |
| `kapitsa doctor [--model M]... [--json]` | This Mac and which models fit it ([Memory and `kapitsa doctor`](#memory-and-kapitsa-doctor)). |
| `kapitsa bench [--model M] [--quick] [--submit]` | Benchmarks Kapitsa on this Mac with the published protocol, and adds the result to the community results when asked ([Benchmarking this Mac](#benchmarking-this-mac)). |
| `kapitsa update [--check] [--version X.Y.Z] [--rollback] [--auto on\|off]` | Updates Kapitsa to its latest release; it also updates itself ([Updating](#updating)). |
| `kapitsa --help`, `kapitsa COMMAND --help`, `kapitsa --version` | Help and version. |

Exit codes: 0 success, 1 failure (download, port, model), 2 usage error (unknown option or model), 130 after Ctrl+C in a
download (the parts are kept) or when the model question is cancelled; the coding-agent commands exit with the agent's
code; `kapitsa update --check` exits 100 when a newer release is out.

### Which model

Run in a terminal without `--model`, `kapitsa serve` and the coding-agent commands ask which model to start: the models
with their download size, ✓ where downloaded and ★ on the best for this Mac, then the chosen model's precisions with
their sizes and whether each fits this Mac's memory (the engine's own plan of its headers; a model or precision that
does not fit is marked too big and can still be chosen, with a warning). Each shows the context it would run with: Tab
(Shift+Tab back) changes it on both screens, between the default ([Context](#context)), 32K, 128K, 256K and max, the
model's own limit, and a model or precision that would run without its draft model at that context says so. ↑ and ↓
(or j and k) move, Enter chooses, Esc goes back or cancels, Ctrl+C cancels. A model not downloaded yet says how much it
downloads, then downloads as usual.

The first choice is the last one, with its context, so Enter twice starts the same model again. It is kept in
`~/.cache/kapitsa/state.json` (`KAPITSA_HOME`), and a `--model` that names a catalog model, typed in a terminal, updates
it too; with no last choice, the best model for this Mac comes first. A coding-agent command that finds a server
already running on its port uses it and says so (stop it to pick another model).

Nothing is asked where nobody could answer: when stdin, stdout or stderr is not a terminal (the Nace Agent app,
scripts, `brew services`), with `TERM=dumb`, `CI` or `KAPITSA_NO_PROMPT=1` set, with `--yes`, or in a background job.
Those start the best model for this Mac, with the same output as before.

## Models and downloads

| Name | Aliases | What it is | Download | Variants |
| --- | --- | --- | ---: | --- |
| `qwen3.5-4b` | `4b` | Qwen3.5, dense and small: for 16-24 GB Macs and simple tasks | 4.6 GB | `4bit` (default), `6bit`, `8bit`, `bf16` |
| `qwen3.5-9b` | `9b` | Qwen3.5, dense: the 4B's bigger sibling, the most capable model for a 16 GB Mac | 9.0 GB | `4bit` (default) |
| `qwen3.8-27b` | `27b` | Qwen3.8, dense: the strongest coder of the three Qwen models, slower to write | 20.2 GB | `4bit` (default), `5bit`, `6bit`, `8bit`, `bf16` |
| `qwen3.6-35b-a3b` | `35b`, `qwen3.6-35b`, `35b-a3b` | Qwen3.6, mixture of experts with 3B parameters active per token: quick to reply and to read | 22.1 GB | `4bit` (default), `5bit`, `6bit`, `8bit`, `bf16`, `16gb` (where `4bit` does not fit: 2-bit experts, 10.8 GB; [GGUF files](#gguf-files)) |
| `qwen3.8-flash-next` | `flash-next`, `flash` | Qwen3.8-Flash-Next 180B-A7B: mixture of 512 experts with 4-stream hyper-connections, an n-gram embedding and sparse attention past 2K tokens; for 128 GB Macs | 105.9 GB | `4bit` |
| `nemotron-labs-dory1.0-4b` | `dory`, `dory-4b`, `dory1.0-4b` | NVIDIA Nemotron Dory1.0 4B: a looped model, its shared block run 1-4 times per token ([thinking depth](#dory-thinking-depth)); text only | 3.2 GB | `nvfp4` (default), `bf16` |
| `nemotron-3-nano-4b` | `nano-4b`, `nemotron-nano-4b` | NVIDIA Nemotron 3 Nano 4B: hybrid Mamba-2, dense, FP8; text only | 5.3 GB | `fp8` (default), `bf16` |
| `nemotron-nano-9b-v2` | `nano-9b`, `nemotron-nano-9b` | NVIDIA Nemotron Nano 9B v2: hybrid Mamba-2, dense, NVFP4; reasoning switched with `/think`; text only | 7.9 GB | `nvfp4` |
| `nemotron-3.5-lightning` | `lightning`, `nemotron-lightning` | NVIDIA Nemotron 3.5 Lightning 30B-A3B: hybrid Mamba-2 mixture of experts, 3B active, 4-bit (NVFP4) experts; quick to reply and to read on 32 GB Macs and up | 22.9 GB | `nvfp4` |
| `nemotron-3.5-bolt` | `bolt`, `nemotron-bolt` | NVIDIA Nemotron 3.5 Bolt 74B-A9B: hybrid Mamba-2 mixture of experts, 9B active, 2-bit experts; an agentic model | 25.1 GB | `24gb` |
| `nemotron-3.5-super-vl` | `super-vl`, `super`, `nemotron-super` | NVIDIA Nemotron 3.5 Super-VL 120B-A12B: hybrid Mamba-2 mixture of experts, 12B active, 4-bit (NVFP4) experts; needs a 96 GB+ Mac | 76.4 GB | `nvfp4` |
| `flint` | `recursive-flint`, `flint-40b` | Recursive Flint 40B-A2.5B: mixture of 256 experts, 2.5B active, sliding-window and full attention; a coding model that reasons before it answers; text only; 32 GB Macs and up | 22.6 GB | `4bit` (default), `16gb` (where `4bit` does not fit: 2-bit experts, 11.9 GB; [GGUF files](#gguf-files)) |

A download includes the companion files that speed decoding up: an MTP head and a DFlash2 draft model for the 35B and
27B, an MTP head and z-lab's DFlash draft model for the 4B, NVIDIA's LiLiCorr draft model for Bolt and for Super-VL,
its DSpark draft model for Lightning; Flash-Next's repo holds its MTP head. Dory and Flint have none: they decode without one, as
does the 35B's `16gb` (its GGUF has no MTP head, and a draft model would not fit beside it).

Flash-Next's `4bit` is Nace AI's quantization of Qwen's bf16 release
([nace-ai/Qwen3.8-Flash-Next-MLX-4bit](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX-4bit), its precision per
tensor chosen by KL against bf16; Qwen publishes no 4-bit). Its recipe keeps the hyper-connections and attention at 6
bits and the router, indexer and n-gram gates at 8 (71.4 GB of weights, 4.1 GB read per token); its 32 GB n-gram table
(`ple.safetensors`) stays on disk and is read through the page cache.

`--model` takes a name or alias (`35b`), one with a variant (`35b:6bit`), a Hugging Face repo (`org/name[@revision]`,
see [Other models](#other-models)), a folder (a path starting with `/`, `~`, `./` or `../`), a `.gguf` file
([GGUF files](#gguf-files)) or the name of a folder in the store.

The first start of a model downloads it; later starts use the copy on disk, offline too. A name without a variant takes
the default where it holds one conversation of 32K tokens (or of the `--context` given) with its draft model, else the
first of the model's smaller variants that does, else the smallest: `35b` and `flint` on a 16 or 24 GB Mac are
`35b:16gb` and `flint:16gb`. A catalog model that would not start on this Mac is refused before the download, naming one
that fits. Without `--model`, `kapitsa serve` takes the first of `35b`, `27b`, `9b` and `4b` that holds one conversation
of 32K tokens (or of the `--context` given) within this Mac's memory budget (a model not downloaded yet is planned from
its file headers, a few hundred KB).

The API calls the model by its name (`qwen3.5-4b`), as `name:variant` for a variant other than the default
(`qwen3.6-35b-a3b:6bit`), or by its repo id or folder name; `--model-id` sets another. The server runs one model, so a
request's `model` field is not checked.

### Precisions

Each Qwen model comes in several precisions, its variants. Download sizes, companions included:

| Variant | Weights | `4b` | `9b` | `27b` | `35b` |
| --- | --- | ---: | ---: | ---: | ---: |
| `4bit` (default) | MLX affine 4-bit, group 64: the fastest | 4.6 GB | 9.0 GB | 20.2 GB | 22.1 GB |
| `5bit` | MLX affine 5-bit, group 64 | - | - | 23.5 GB | 26.4 GB |
| `6bit` | MLX affine 6-bit, group 64: the quality choice | 5.6 GB | - | 26.9 GB | 30.7 GB |
| `8bit` | MLX affine 8-bit, group 64 | 6.7 GB | - | 33.6 GB | 39.4 GB |
| `bf16` | unquantized, where it fits | 10.6 GB | - | 58.8 GB | 71.9 GB |
| `16gb` | 2-bit experts: unsloth's UD-IQ2_XXS GGUF, imported ([GGUF files](#gguf-files)); `35b` takes it where `4bit` does not fit (16 and 24 GB Macs) | - | - | - | 10.8 GB |

Each step up stays closer to the original weights, decodes slower and needs more memory; `kapitsa doctor` shows which
variants fit this Mac. Every variant but `16gb` keeps the model's MTP head and draft model; each pins its repo to a
revision, so a download is always the same bytes. After a Kapitsa update that pins newer revisions, the next start
fetches only the files that changed (offline, it keeps using the copy it has). Bolt has one variant, `24gb`: NVIDIA's
own mixed-precision checkpoint (IQ2_XXS experts, NVFP4 Mamba layers, Q8_0 attention, latent and head, BF16 embedding).
Super-VL has one variant, `nvfp4`: NVIDIA's own checkpoint as stored (NVFP4 routed and shared experts, FP8 Mamba
projections, attention and head, BF16 latent projections, which load as 8-bit, router and embedding). Dory has two,
NVIDIA's own checkpoints as stored: `nvfp4` (the default: NVFP4 linears, the format most of its pre-training ran in,
with BF16 embedding and head; 3.2 GB) and `bf16` (8.0 GB). Along two replies (320 positions), the NVFP4 checkpoint's
next-token distributions are a KL of 0.020 from the BF16 checkpoint's on average (NVIDIA's code, 2 loops), with the same
top token at 94% of positions. With `nvfp4` the LM head loads as 8-bit rows (quantized from its BF16 at load: a
next-token KL of 7e-6 from the BF16 head, the same top token at 319 of 320 positions, about 7% faster writing;
`NI_DORY_HEAD=bf16` keeps it BF16). Flint has two: `4bit` (the default, Recursive's MLX export) and `16gb`
(unsloth's UD-IQ2_XXS GGUF, imported: [GGUF files](#gguf-files)).

### Other models

Any Hugging Face repo of a Qwen3.5-family model (Qwen3.5, Qwen3.6, Qwen3.8) in MLX format runs: bf16, or MLX affine 3,
4, 5, 6 or 8-bit with groups of 64 (4-bit also 32), mixed recipes included. Kapitsa reads the repo's `config.json`
before any weights, then downloads its top-level files except other formats (`.gguf`, `.bin`, images and the like). A
catalog variant's repo (`mlx-community/Qwen3.6-35B-A3B-6bit`) comes with that variant's MTP head and draft model; other
repos use the MTP head they contain, if any.

A folder needs `config.json`, the `*.safetensors` weights and `tokenizer.json`, and may hold an MTP head in
`mtp.safetensors`. The official Hugging Face checkpoints of the dense models run as well. A Qwen3.8-Flash-Next
(`qwen4_exp`) folder also needs its `ple.safetensors` table.

NVIDIA's dense Nemotron-H checkpoints run from a folder too (`kapitsa serve --model DIR`): Mamba-2, attention and dense
MLP layers, no experts, as [Nemotron Nano 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) (BF16) and
[Nemotron Nano 9B v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4) (NVFP4, with its attention, first
and last layers and LM head left BF16; the head loads as 8-bit rows, `NI_NH_HEAD`). They have no draft model. The 4B
decodes with prompt lookup: drafts from earlier in the conversation (up to 4 per cycle), verified in one pass, so the
reply is the one plain decoding gives, on an M1 Pro 4-8% sooner in BF16. In FP8 (with its 8-bit head) it is 1% sooner
on the benchmark's code, 1.4-2.3x on replies that repeat the prompt (a rewrite of code it was given) and 4-5% later on
free prose, where few drafts are accepted. The 9B v2 decodes plainly, as prompt lookup measured
within 2% either way on its NVFP4 weights there; `--spec N` sets the drafts per cycle for either, `--spec 0` turns
prompt lookup off. The 9B v2's template turns thinking on and off with `/think` and `/no_think` in a message: the APIs'
switches (`enable_thinking`, `reasoning_effort`, Anthropic's `thinking`) set that flag, and its
`<TOOLCALL>[...]</TOOLCALL>` calls come back as tool calls. That template also drops a past turn's reasoning and its
`<think>` tags, so a follow-up request reuses the cache up to the previous reply, not through it.

Recursive's Flint (`--model flint`: [RecursiveSI/flint-int4](https://huggingface.co/RecursiveSI/flint-int4), its MLX
4-bit export) has 49 layers: 35 of sliding-window attention (513 positions, a ring per conversation) and 14 of full
attention without positional encoding, each head with a learned sink, then 256 experts routed by sigmoid scores (6 a
token) beside an always-on shared expert. It reasons before every answer (its template has no switch for that), and its
tool calls, in GLM's `<tool_call>` format, come back as tool calls. It has no draft model and decodes plainly: prompt
lookup (`--spec N`) checks 2-4 rows a pass at 1.6-2.8 times a decode step's cost on an M4 Pro (each row brings its own 6
experts), and on the benchmark's code a draft exists for 1 token in 4, so it is off unless asked for. With it on, a reply
can part from plain decoding's beyond a near-tie of the next token: Flint's router rounds its scores to bf16 before it
takes its top 6, and where a score sits on a rounding boundary a verify pass, which sums in another order than a decode
step, can pick another expert, and every layer after it computes something else. The output check met one such row in
192 (layer 16 at a window's position 1684), at verify passes of 2, 4 and 8 rows alike.

Refused, with a message: other architectures, other quantizations (MLX's `mxfp4`, `nvfp4` and `mxfp8` modes among
them; NVIDIA's own NVFP4 checkpoints of the catalog's Nemotron models are read as they are), mixture-of-experts
checkpoints in the Hugging Face layout, and weights larger than this Mac's memory budget.

### GGUF files

`kapitsa import FILE.gguf` converts a GGUF file once into a model folder in the store and prints its name for
`kapitsa serve --model`; `kapitsa serve --model FILE.gguf` and the coding-agent commands convert on first use and reuse
the conversion after that. It reads the GGUF architectures `qwen35` and `qwen35moe` (Qwen3.5, 3.6 and 3.8) as LM Studio,
Ollama (its blob files) and Hugging Face GGUF repos ship them, and `flint` (Recursive's Flint, where its routed experts
are IQ blocks: below); a split file is named by any of its parts.

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
64 from 1-2-bit types, except routed experts in IQ2_XXS, IQ2_S or IQ3_S (and IQ3_XXS or IQ1_S with
`KAPITSA_IMPORT_IQ_NEW=1`): those stay in the file's own blocks, which Kapitsa's kernels read as they are. The folder's experts file is then an APFS clone of the GGUF (no second copy; on
another volume, or with `KAPITSA_IMPORT_COPY=1`, a copy), and the model locks its weights in memory as it loads. Refused:
the ternary TQ1_0 and TQ2_0, Hadamard-rotated weights, Qwen3-Next, other architectures.

That is how Qwen3.6-35B-A3B runs on a 16 GB Mac: the UD-IQ2_XXS file of
[unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) (10.8 GB; IQ2_XXS gate and up experts,
IQ2_S and IQ3_S down experts) imports in seconds into 11.0 GB of weights, inside a 16 GB Mac's 12.2 GB budget with room
for 21K tokens of KV cache. The catalog has it as the 35B's `16gb` variant, which `35b` takes where the 4-bit default
does not fit (16 and 24 GB Macs):

```bash
kapitsa serve 35b          # on a 16 GB Mac: 35b:16gb, downloaded and imported the first time
kapitsa pull 35b:16gb      # the same, without starting it
```

It downloads that one file, pinned to a revision and checked against its sha256 like every catalog file (resumed after
Ctrl+C; taken from the Hugging Face cache or a `--from` folder when it is already on this Mac), then imports it into
`qwen3.6-35b-a3b-16gb` in the store. The folder then holds the conversion, not the file: its experts file is an APFS
clone of the file, 12.7 GB on disk in all with the file's 10.8 GB. The pull needs that much free disk, and 6 GB more
that the import keeps free; a stopped import starts again from the downloaded file. Before the download, its memory
plan is the import's plan of the file's header (16 MB, fetched once). `kapitsa pull --dry-run 35b:16gb` shows the file,
its size and the folder's size after the import. A copy of the file you already have works too: `kapitsa pull
35b:16gb --from DIR` (a clone, no download), or `kapitsa import Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` (a folder outside the
catalog, named after the file).

On a MacBook Pro M1 Pro 16 GB it decodes 52.6 tok/s (llama.cpp b11669 on the same file: 38.0-38.8) and reads a
4K-token prompt in 9.0 s (11.8 s). Against llama.cpp's Metal backend on the same file its next-token distributions are at
KL 0.023 with the same top token 97.5% of the time (llama.cpp's own CPU backend: 0.012 and 97.5%); part of the gap is
the conversion of the file's Q5_K tensors to 6-bit. It has no MTP head, so it decodes plainly, and no vision tower: it
reads text only.

Flint's GGUFs import their own way. Their routed experts must be IQ blocks (IQ1_S, IQ2_XXS, IQ2_S, IQ3_XXS or IQ3_S),
which stay as they are; attention, the dense MLP and the shared expert take 4-bit group 64, the embedding 3-bit group
64, the LM head 6-bit group 64, and the router stays bf16. A Flint GGUF with other experts (Recursive's Q4_1 file) is
refused, pointing at `flint:4bit`. unsloth's `flint-UD-IQ2_XXS.gguf`
([unsloth/flint-GGUF](https://huggingface.co/unsloth/flint-GGUF), 11.9 GB; IQ1_S, IQ2_XXS, IQ2_S and IQ3_XXS experts)
imports in 6 s into 11.73 GB of weights, 13.0 GB on disk in all with the file's 11.9 GB (the experts file is an APFS
clone of it). In a 16 GB Mac's 12.2 GB budget that leaves room for one conversation of 8K tokens: the engine runs it
with one slot, 448-row prefill chunks and an 8K default context ([Memory](#memory-and-kapitsa-doctor)). The catalog has
it as Flint's `16gb` variant, which `flint` takes where the 4-bit default does not fit (16 and 24 GB Macs):

```bash
kapitsa serve flint        # on a 16 GB Mac: flint:16gb, downloaded and imported the first time
```

On a MacBook Pro M1 Pro 16 GB it decodes 61 tok/s (llama.cpp d772569 on the same file, runs alternated: 42.4) and
reads a 4K-token prompt in 9.4 s (10.9-11.2 s), at a peak of 11.6 GB (12.2 GB). The requantized tensors cost quality: against llama.cpp on the
same file its next-token distributions are at KL 0.22 (median 0.018), with the same top token 89% of the time; on the
same weights (its import written back as a GGUF) the two engines agree at KL 0.045 (median 0.001, 97.4%).

### Where models are stored

Models live in `~/.cache/kapitsa/models`; `KAPITSA_HOME` moves `~/.cache/kapitsa`. Folders are named `<name>-<variant>`
for catalog models (`qwen3.6-35b-a3b-4bit`), `org--name` or `org--name@revision` for repos, and by `--name` for GGUF
conversions. Inside, `.kapitsa/` holds the parts of files being downloaded and the manifest, written last (`35b:16gb`'s
folder holds its GGUF file until the import replaces the folder with the conversion).

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
(`kapitsa rm 35b`: the default one, or the only one downloaded). It says how much space it gave back: a model that is an APFS clone of
another copy (Nace Agent's, a `--from` folder, the Hugging Face cache) frees nothing.

When a download does not fit the disk (Kapitsa keeps 1 GB spare, 6 GB for `35b:16gb`'s import), `kapitsa serve`,
`kapitsa pull`, `kapitsa bench` and the coding-agent commands, run in a terminal, list the other downloaded models with
what deleting each would free: space selects, Enter deletes them and the download goes on once they free enough, Esc
cancels. Elsewhere (the app, scripts) the error names the `kapitsa rm` commands that would make room.

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
| `--allow-switch` | off | Let a request naming another model, or `POST /v1/models/switch`, load that model (it may download gigabytes). Off, the `model` field is ignored, as clients send names of their own. On a network `--host` it needs `--api-key`. |
| `--model-id ID` | the model's name | The model's name in the API. |
| `--context N` | `128k`, less where memory is tight | Context per conversation: tokens (`131072`), `k` or `m` of them (`128k`, `1m`), or `max`, the model's own limit ([Context](#context)). `--ctx N` works too. |
| `--slots N` | `4` where memory allows, else `3` or `2` | Requests that run at once, each with its conversation's cache kept for follow-ups; more wait for a free slot, first come first served. |
| `--dflash auto\|none\|DIR` | `auto` | Draft model: `auto` uses the one downloaded with the model, unless it would keep one `--context` conversation from fitting; `none` decodes without it (with the MTP head on Qwen); `DIR` names a drafter folder. |
| `--spec N` | `1` on the 35B, `4` on the 27B and 4B | MTP draft depth when no draft model is used; `0` turns speculative decoding off. Dory with `--loop-kv per-loop`: drafts per cycle (by default 2 at 1-2 loops, 3 at 3-4; [speculative decoding](#dory-thinking-depth)). Dense Nemotron-H (Nemotron Nano): prompt lookup's drafts per cycle (by default 4 with BF16 or FP8 MLP layers, off with NVFP4). Flint: prompt lookup's drafts per cycle (off by default). |
| `--mem-budget GB` | macOS's recommended working set, a little less | GPU memory Kapitsa may hold ([Memory and `kapitsa doctor`](#memory-and-kapitsa-doctor)). |
| `--prefill-chunk N` | `2048` (`1024` when memory is short) | Prompt tokens per prefill pass. On a 32-core M1 Max and on the M5 (Apple10 GPUs), a single slot with the 35B-A3B 4-bit group-64 model uses `4096` when its memory plan fits. An MoE model's pass holds up to 1/32 more rows than the chunk (`2112` for `2048`), and the prompt is split evenly over the passes it needs, so a prompt a few tokens past a multiple of the chunk does not pay a pass of its own for them. `NI_BAL=0` restores the plain chunking. |
| `--disk-cache SIZE` | `auto` | The SSD prefix cache: conversations kept on the SSD, also across restarts ([SSD prefix cache](#ssd-prefix-cache)). `auto`: a quarter of the free space beyond 10 GiB, at most 32 GiB; a size (`16G`, `512M`) caps it within that; `0` turns it off. |
| `--loops N` | `2` | Dory: how many times its shared block runs per token, 1-4 ([thinking depth](#dory-thinking-depth)); requests may choose their own. |
| `--loop-kv MODE` | `last-loop` | Dory: the shared block's KV cache, `last-loop` (one per layer) or `per-loop` (one per loop, more memory). |
| `--image-max-pixels N` | `1200000` (Bolt and Super-VL `1400000`) | Larger images are resized to about N pixels; `0` = the model's own limit ([Images](#images)). |
| `--verbose` | off | In a terminal, show the engine's log on screen too. |
| `--yes` | off | In a terminal without `--model`, start the best model for this Mac instead of asking ([Which model](#which-model)). |

### Context

`--context` sets how long one conversation can get: its prompt, its history and the reply share it. Without it, `kapitsa
serve` and the coding-agent commands take 128K tokens, or the model's own limit where that is less, or less where this
Mac's memory is tight: the most, in 8K steps down to 32K, that keeps the draft model the model has at 32K, with one full
conversation beside it (below 32K only where one slot cannot hold 32K tokens: Flint's `16gb` on a 16 GB Mac takes 8K).
The server says which context it took and why. A context costs no memory until a conversation fills it (the KV cache
grows with it), so the default only sizes what memory is kept for.

`--context max` takes the model's own limit (`max_position_embeddings` in its `config.json`): 256K for the Qwen models,
Dory and Bolt, 1M for Lightning and Super-VL. A longer `--context` runs at that limit, and the server says so. Where a
long context leaves no room for the draft model, the model runs without it (slower); the model picker
([Which model](#which-model)) and `kapitsa doctor` show where. A longer context also lets a prompt grow longer before a
coding agent compacts it, and a long prompt takes a while to read whenever it is not cached.

### Requests at once

Requests sent at once (a coding agent's parallel subagents and side requests, several clients) each run on a slot of
their own. Their prompts take turns on the GPU between chunks, so a request sent during a long reply starts after a
fraction of a second instead of after that reply. While two or more of them decode, one GPU pass serves them all (with
the draft models of Qwen3.8 27B, Qwen3.6 35B-A3B and Nemotron 3.5 Bolt): each weight is read once for every request,
so replies at once write more tokens per second in all than one reply alone, each a little slower than alone. Tokens
per second in all, 512-token replies to code prompts, greedy, on a MacBook Pro with an M4 Pro (48 GB):

| Requests at once | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8 27B, 4-bit | 57 | 66 (58) | 66 (56) | 67 (58) |
| Qwen3.6 35B-A3B, 4-bit | 154 | 196 (160) | 206 (149) | 202 (161) |
| Nemotron 3.5 Bolt, NVIDIA 24 GB | 65 | 77 (63) | 64 (63) | 75 (62) |

In parentheses: the same requests taking turns on the GPU, as with `NI_BATCH=0`. With the Qwen models KFlash chooses
each shared pass's rows (`NI_BATCH_KF`): how many in all, up to 8, and which request gets them, from each request's
drafts and what a row costs on this Mac; three requests now share 8 rows instead of 2 each (or, on the 27B, taking
turns). A request alone gets the tokens it always gets. Requests decoded
together can get other tokens than alone where two candidates nearly tie (they are within the batched pass's rounding of
each other), and sampled requests draw other random numbers, so their replies differ as two samples do. With the Qwen3.5
4B's DFlash drafter, Nemotron 3.5 Lightning and Super-VL (not measured batched yet; `NI_BATCH=1` batches them), without
a draft model (`--dflash none`) and with `NI_BATCH=0`, requests at once take turns on the GPU, each with the tokens it
gets alone, and write about as many tokens per second in all as one alone.

Without `--slots` the server takes 4 slots where they fit as well as 2 would (room for one full `--context` conversation,
the same draft model and prefill chunk), else 3, else 2; `GET /v1/models` reports them. With every slot busy a request
waits for one (the log says so); a streamed chat completion whose client disconnects stops waiting.

### SSD prefix cache

A slot keeps its conversation's state in memory until another conversation needs the slot, and the server keeps
nothing when it stops. The SSD prefix cache keeps conversations on the SSD as well: a request whose prompt starts with
a stored conversation (its next turn, a tool result, the same agent session after `kapitsa claude` or the server
restarted) reads that state back and prefills only what follows it. `usage.prompt_tokens_details.cached_tokens`
(Anthropic: `cache_read_input_tokens`) counts the tokens read back, as for a slot's own cache.

- **What is kept.** The state a follow-up resumes from: the KV cache up to it, the recurrent state, the tokens, and the
  draft model's context (the DFlash drafters' context ring; with the MTP head, its KV rows and the draft rows it
  gathered). A slot's state is kept as it is, byte for byte, so a request resumed from the SSD writes what it would
  from the slot. A conversation's next turns add only their new KV rows and states: the earlier ones are shared.
  Images in a conversation are kept with it by their content: a prompt reuses a stored state only through the same
  images. States nearer the start than 1024 tokens are not kept (quick to prefill again).
- **When it is written.** After a turn, once the server has been idle for 10 s, in the background (a request that
  arrives stops it, and nothing is written while requests run); when a request is about to replace a conversation the
  SSD does not have yet (another conversation takes its slot, or memory for one), what is missing is copied first and
  written in the background; and when the server stops (Ctrl+C, SIGTERM, a `kapitsa claude` session ending), for up to
  8 s. A conversation is read back only when it is deeper than anything a slot holds.
- **Where.** `~/.cache/kapitsa/prefix-cache` (`KAPITSA_HOME`), kept out of Time Machine and readable by your user only
  (it holds the conversations' tokens). One folder per model and draft model (a digest of their files and of the state
  formats), so no model reads another's states; one server uses a folder at a time (another waits up to 30 s for it,
  then runs without the cache). Every file has a CRC-32C and is written to a temporary name, then renamed: a damaged
  or partial file is a miss, never a wrong state.
- **How much.** `--disk-cache auto` (the default) takes at most a quarter of what the volume has free beyond 10 GiB
  (5% of a volume smaller than 200 GiB), its own files counted, and at most 32 GiB: on a disk with 28 GiB free, 4.5
  GiB. It follows the free space as the disk fills or empties, giving up the least recently used conversations first.
  `--disk-cache 16G` caps it at 16 GiB within the same rule; `--disk-cache 0` turns it off. Other models' caches that
  no server uses count against the limit too and go first; one no server has used for 14 days is removed when a server
  starts.
- **Models.** The Qwen3.5-family models (Qwen3.5 4B, Qwen3.6 35B-A3B, Qwen3.8 27B), with their draft models or the
  MTP head, and Dory (a state is read back only for the loop count that computed it). Nemotron-H models and
  Qwen3.8-Flash-Next run without it (the log says so).

The server says at start where the cache is, its limit and how many conversations it took back; each read and write
has a log line (`SSD cache: ...`).

Measured on a MacBook Pro 16-inch (M4 Pro, 20-core GPU, 48 GB), macOS 27.0.1, by `tests/test_disk_cache.py --ttft`
(greedy, thinking off, the DFlash2 drafters, two runs each): the time to the first token of a conversation's second
turn (the first turn's prompt and answer, then a short question) with nothing cached, from the SSD after the server
restarted, and from the slot that answered the first turn, each sent right after a short request (with the GPU
awake). From the SSD the second turn writes the slot's text, token for token.

| Model | Second turn | Nothing cached | From the SSD, after a restart | From the slot | Read | Written: turn 1, turn 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 35B-A3B | 4,359 tokens | 3.81-3.83 s | 0.12 s | 0.06 s | 205 MB in 54 ms | 321 MB, 239 MB |
| Qwen3.6 35B-A3B | 16,322 tokens | 17.2-18.0 s | 0.16-0.18 s | 0.08-0.09 s | 450 MB in 81-82 ms | 566 MB, 242 MB |
| Qwen3.8 27B | 4,365 tokens | 12.2 s | 0.40-0.41 s | 0.27 s | 483 MB in 107-128 ms | 682 MB, 418 MB |
| Qwen3.8 27B | 16,325 tokens | 46.7-48.0 s | 0.65-0.70 s | 0.32-0.33 s | 1,267 MB in 286-291 ms | 1,466 MB, 432 MB |

A save writes the KV rows the SSD lacks and the states of its new restore points (the recurrent state and the draft
model's context: about 110 MB each on the 35B, 200 MB on the 27B), so each turn the server idles after writes a few
hundred MB (the table's turn 2).

## APIs

OpenAI clients use the base URL `http://127.0.0.1:8000/v1`, Anthropic clients `http://127.0.0.1:8000`.

| Endpoint | What it is |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions: `reasoning_content`, `tool_calls`, and `usage` (in a stream's last chunk) with `prompt_tokens_details.cached_tokens`. `max_tokens` defaults to 16384. Automatically switches to the requested `model` on demand if different from the currently active model. |
| `POST /v1/responses` | OpenAI Responses: reasoning items, function and custom tools, `usage.input_tokens_details.cached_tokens`. `previous_response_id` continues one of the last 64 stored responses (unknown ids are a 400). Built-in tools such as web search are left out. Supports on-demand model switching via `model`. |
| `POST /v1/messages` | Anthropic Messages: thinking blocks, `tool_use` and `tool_result`; `usage.input_tokens` counts the uncached prompt, `cache_read_input_tokens` the cached part. Server tools are left out. Supports on-demand model switching via `model`. |
| `POST /v1/messages/count_tokens` | Anthropic's token count of a request. |
| `GET /v1/models` | Available models (active model first, downloaded models, and catalog models); includes `active`, `downloaded`, `context_fit`, and `slots`. |
| `GET /v1/models/:model` | Information about a specific model. |
| `POST /v1/models/switch` | With `--allow-switch`: switches the active model, `{"model": "<name>"}` (403 without the flag). While it loads, other requests get 503 with `Retry-After`. A model that fails to load leaves the previous one serving. |
| `GET /health` | `{"status":"ok"}` |
| `GET /` | The chat page: one file with dynamic client-side model selection dropdown that talks only to this server and keeps its conversations in the browser. |

Every API streams (server-sent events) or answers in one JSON reply. A prompt longer than the context is a 400 error,
streamed or not (Messages: "prompt is too long", which Claude Code answers by compacting; Responses and Chat
Completions: `context_length_exceeded`). A stream with nothing to send for 15 s (a long prefill; for Chat Completions,
whose stream starts before the request has a slot, also the wait for one) gets a keepalive, so clients do not drop
it: a `ping` event (Messages), a `response.in_progress` event (Responses), a chunk with an empty delta (Chat
Completions).

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

### Structured output

A request can hold the reply to a JSON Schema, to any JSON object, or to tool calls. Each token is then sampled under a
mask of the tokens that keep the reply valid, so it cannot leave the format: greedy picks the best allowed token,
sampling draws from the allowed tokens' renormalized distribution (after `top_k` and `top_p`, as without a format).
The reasoning stays free; with thinking on, the format holds from `</think>` on. The prompt is not changed: say in the
messages what the reply should contain.

| API | JSON Schema | Any JSON object | A tool call |
| --- | --- | --- | --- |
| Chat Completions | `"response_format": {"type": "json_schema", "json_schema": {"name": ..., "schema": {...}, "strict": true}}` | `"response_format": {"type": "json_object"}` | `"tool_choice": "required"` or `{"type": "function", "function": {"name": ...}}` |
| Responses | `"text": {"format": {"type": "json_schema", "name": ..., "schema": {...}, "strict": true}}` | `"text": {"format": {"type": "json_object"}}` | `"tool_choice": "required"`, `{"type": "function", "name": ...}` (or `"custom"`), or `{"type": "allowed_tools", "mode": "required", "tools": [...]}` |
| Messages | `"output_format": {"type": "json_schema", "schema": {...}}` (or `output_config.format`) | | `"tool_choice": {"type": "any"}` or `{"type": "tool", "name": ...}` |

- **Tool calls:** a forced call is written in the model's own format (Qwen's `<tool_call>` XML), names one of the
  request's tools (the named one) and has arguments that follow the tool's `parameters` schema: the required ones, each
  at most once, no others where the schema closes the object, values of the declared types (an argument declared
  `"type": "string"` is written as raw text, as the template writes it: one of its `enum` values if it has them, its
  length, `pattern` and `format` not held; others as JSON following their schema). One call or more, or one
  with `"parallel_tool_calls": false` (Messages: `"disable_parallel_tool_use": true`). A tool marked `"strict": true`
  has its arguments held to its schema whenever the model calls it, with `tool_choice` `auto` too (the model may still
  answer in text). In Responses a function without `strict` is not strict (OpenAI's API makes it strict by default:
  coding agents send their tools without it and stay unconstrained).
- **A format beside tools** (`tool_choice` `auto`): the reply is the JSON or tool calls.
- **`strict`:** objects take no keys their schema does not declare, unless it says `additionalProperties`; without
  `strict` they take any (JSON Schema's rule). Messages formats are strict.
- **The JSON Schema subset held:** `type` (one or a list), `properties`, `required`, `additionalProperties` (a boolean or
  a schema), `enum`, `const`, `items` and `prefixItems` with `minItems` and `maxItems`, `anyOf`, `oneOf` (as `anyOf`: a
  value matching two alternatives is not refused), `allOf` (merged), `$ref` within the schema (`#/$defs/...`, recursion
  included), `nullable`, a string's `minLength` and `maxLength` (an escape counts as one character), and an integer's
  `minimum`, `maximum`, `exclusiveMinimum` and `exclusiveMaximum`. Not held (the reply may break them): a string's
  `pattern` and `format`, a number's bounds (but a `minimum` of 0 or more, which rules out a minus sign), `multipleOf`,
  `uniqueItems`, `not` and `if`/`then`; `patternProperties` is read as allowing no other keys. A remote `$ref`, an
  unknown type, a schema no value satisfies or one nested deeper than 64 levels is a 400 naming the field.
- **Layout:** between two JSON tokens at most 48 bytes of whitespace with at most 2 line breaks, nothing after the
  value, at most 20 digits before a number's point (19 for integers), 24 after it and 3 in its exponent: a model that
  starts repeating spaces or digits is held to a short run instead of writing to `max_tokens`. A reply cut by
  `max_tokens` or a stop sequence is cut JSON, as with OpenAI.
- **Speed:** a mask over Qwen's 248K-token vocabulary costs about 3 µs at the median and under 50 µs in 99 cases of
  100 on an M4 Pro (the first time a server meets a kind of text, it builds a cache entry for it: a few ms, once),
  computed while the GPU runs the layers of the pass that samples under it. With a DFlash2 draft model (the Qwen
  models) drafts still verify, each verify row under its own mask (`NI_GRAMMAR_SPEC=0` decodes one token per pass
  instead), and requests at once still share batched passes, each member's rows under its own masks. Qwen3.5 4B on
  the M4 Pro: a reply the format does not change decodes as fast as without it (168 tok/s with KFlash, 90 without);
  five JSON replies under schemas, 177 tok/s with KFlash cycles under masks against 88 tok/s one token per pass, the
  same greedy replies. A format that changes what the model writes changes how many drafts are accepted, and so the
  speed.
  With the other drafters (MTP, LiLiCorr, DSpark, prompt lookup, Dory's own) the formatted part decodes one token per
  pass and takes turns on the GPU.

### Dory: thinking depth

NVIDIA's Nemotron Dory1.0 4B is a looped model: 16 input layers, a shared block of 40 layers, 16 output layers,
and the shared block runs 1 to 4 times per token. Each loop is thinking in latent space: more compute per token, the
same weights, no extra tokens. It is separate from the visible reasoning (the `<think>` block), which stays as the
chat template and the request say (thinking on by default in Chat Completions and Responses).

| Loops | What NVIDIA reports | Speed |
| --- | --- | --- |
| 1 | the cheapest per token; SWE-bench Verified 50.3 | the fastest |
| 2 (default) | most of the gain; SWE-bench Verified 59.5 | |
| 4 | best on agentic tasks; SWE-bench Verified 61.8 | the slowest |

- **Server default:** `kapitsa serve --model dory --loops 4` (or `kapitsa claude --model dory --loops 4`); without it, 2
  loops.
- **Per request:** `"loops": N` in the request body (OpenAI clients: `extra_body={"loops": 4}`) sets it for that request.
  Without it, the agents' own effort controls choose: OpenAI's `reasoning_effort` and Responses' `reasoning.effort`
  (`minimal` and `low` 1 loop, `medium` 2, `high` and `xhigh` 4; `none` only turns the visible thinking off and keeps
  the default), Anthropic's `thinking` (disabled 1 loop, a
  `budget_tokens` under 4096 1, under 16384 2, above 4; adaptive the server's default) and `output_config.effort`.
  Every reply reports the loops it ran in `usage.kapitsa.loops`, and the server log names them.
- **Cache:** a conversation's cached prompt is only valid for the loop count that computed it; a request with another
  count reads its prompt again (on another slot when one is free, so the conversation in progress keeps its cache).
- **KV cache (`--loop-kv`):** `last-loop` (the default, NVIDIA's `recurrent_kv_cache_mode: last_loop`) keeps one cache per
  layer, each loop writing over the previous one's, so a conversation's KV cache is the size of a one-loop model's at
  any loop count: 36 KB per token plus 114 MB of sliding-window rings. `per-loop` keeps a cache per loop, as NVIDIA's vLLM
  and llama.cpp commands run it (56 KB per token at 2 loops, 96 KB at 4). `kapitsa doctor` shows both. The two differ
  little: on the BF16 weights at 2 loops, along four replies (504 positions), per-loop's next-token distributions are a
  KL of 0.0009 from last-loop's on average, with the same top token at 99.2% of positions (about a twentieth of the
  NVFP4 checkpoint's difference from BF16); greedy replies first differ after 10, 88 and 130 tokens, and not within 160
  on the fourth. Long prompts move them further apart (a prompt longer than a prefill chunk, 2,048 tokens, reads its
  later chunks against final-loop keys with last-loop KV): after a 2,866-token prompt, a KL of 0.044 and the same top
  token at 92.5% of 160 positions, the greedy replies differing after 17 tokens. Both modes match NVIDIA's code for
  their mode.
- **Speculative decoding (`per-loop` only):** with `--loop-kv per-loop`, each cycle drafts a few tokens and one pass of
  the full loop count checks them: drafts from earlier in the conversation when its last tokens occurred before
  (prompt lookup), else from the model itself run with one loop (its input layers and first loop are the full model's
  own, so the check reuses them). The reply is the one plain decoding writes (up to rounding where two tokens are
  nearly tied: the checking pass reads its rows at half precision). On the M4 Pro, 128-token replies against plain
  per-loop decoding: 2 loops +17% after a short prompt, about even after 2,866 tokens; 4 loops +57% and +19% (2 drafts
  a cycle at 2 loops, 3 at 4). Last-loop KV cannot check several tokens at once (decoding token by token, each token's
  loops see the earlier tokens' final-loop keys, which a pass over several tokens does not have yet), so it decodes
  one token per pass. `NI_DORY_SPEC=ngram` keeps only prompt lookup, `--spec 0` turns it off.
- **Replies that think to the end:** as NVIDIA's `force_nonempty_content` (on by default here), a reply whose answer is
  empty returns its reasoning as its content too; `"chat_template_kwargs": {"force_nonempty_content": false}` turns it
  off.

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
an error). Otherwise the command downloads the model if needed, starts `kapitsa serve` with serve's default context
([Context](#context): 128K, less where memory is tight; `--context N` sets it), logging to
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

Every catalog model except `nemotron-3.5-lightning`, `nemotron-labs-dory1.0-4b`, `nemotron-3-nano-4b` and
`nemotron-nano-9b-v2` (text-only models) reads images, in every precision: `qwen3.5-4b`, `qwen3.5-9b`,
`qwen3.6-35b-a3b`, `qwen3.8-27b`, `qwen3.8-flash-next`, `nemotron-3.5-bolt` and `nemotron-3.5-super-vl`. Their vision
towers come with the download and load with the first image: about 0.67 GB of GPU memory for the 4B's, 0.9 GB for the
9B's, the 35B's, the 27B's and Flash-Next's (bf16 in every precision), 1.7 GB for Bolt's and for Super-VL's. On a model
without one, such as Lightning, Dory or a text-only conversion, each image becomes a short
note in the prompt, so agents that attach screenshots keep working.

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
grows as it fills: 36 KB per token on the 4B, 68 KB on the 27B, 22 KB on the 35B, 28 KB on Flint (plus 38 MB for its
sliding-window rings), 8 KB on Bolt and on Super-VL, 6 KB on Lightning, 31 KB on Flash-Next. Flash-Next's n-gram table
(32 GB) is not in the budget: it is mapped from its file, and each token reads 16 rows of it on the CPU, so macOS keeps
the parts in use in the page cache and can drop them under pressure. When memory is short:

- At start, the fixed part and an 8K-token KV cache must fit, else the server exits with `insufficient memory: ...`,
  what each part needs and what would help (fewer `--slots`, `--prefill-chunk 1024`, no draft model, a smaller model,
  `--mem-budget`). `kapitsa serve --model NAME` checks this before downloading.
- Where the draft model or 2048-token prefill chunks would keep one `--context` conversation from fitting, the engine
  leaves the draft model out or uses 1024-token chunks. Where even those leave less than the tokens it starts with (the
  context, up to 8K), it takes smaller chunks (a multiple of 64 rows, at least 256), one slot where two would not start,
  and a default context below 32K (in 8K steps, 1K steps below 8K, down to 4K): Flint's `16gb` on a 16 GB Mac runs
  with 448-row chunks, one slot and 8K.
- A prompt that needs more room takes it from cached image embeddings, then from idle slots, least recently used first
  (their conversations are read again when they return), then from the Neural Engine split (prefill then runs on the
  GPU alone) and last from the image encoder; failing that, the request gets an `insufficient memory` error. With
  other requests in flight, a request that fits only once one of them ends waits for it, and only a prompt that does
  not fit even alone gets the error; a reply in progress may take the cache of a request still waiting (that request
  then reads its prompt again). A reply that cannot grow its cache ends with finish reason `length`.

`kapitsa doctor` shows this Mac (chip, GPU cores, RAM, macOS, the working set, Kapitsa's budget, the Neural Engine) and,
per model, its memory part by part, whether it starts, the tokens of KV cache that fit, its decoding (draft model, MTP
or plain), the image encoder's size and a rough speed estimate; then each catalog variant, whether it fits and which one
`--model NAME` takes. It assumes `--context 64k` and, per model, the slots the server would start with (the catalog
check: 2, or 1 where 2 would not start; `--context`, `--slots`, `--prefill-chunk`, `--dflash` and `--mem-budget` change
that), and shows the context `kapitsa serve` takes there by default; `--model M`, repeatable, checks other models,
downloaded or not; `--json` prints the same for programs (`starts`, `fits_full_context`, `kv_tokens_in_budget`,
`default_context`, `context_limit`, `decoding`, ...).

What fits, for the 4-bit variants at `--context 64k --slots 2`, with the tokens of KV cache that fit in all. macOS sets
the working set at about 2/3 to 78% of the RAM, depending on its version, hence the ranges:

| Mac RAM | Budget | `4b` | `9b` | `27b` | `35b` |
| --- | --- | --- | --- | --- | --- |
| 16 GB | 11.0-12.9 GB | yes (174K-225K) | yes (89K-140K) | no | no (`16gb`: from a 12 GB budget, 23K-57K) |
| 24 GB | 16.5-19.3 GB | yes (323K-399K) | yes (238K-314K) | only at the larger budget (23K) | no (`16gb`: yes, 219K-356K) |
| 32 GB | 22.0-25.7 GB | yes (472K-572K) | yes (387K-487K) | yes (62K-80K) | yes (50K-153K) |
| 48 GB | 33.0-38.6 GB | yes (770K-922K) | yes (685K-837K) | yes (184K-265K) | yes (475K-724K) |

On a 48 GB Mac the 35B also fits in 6-bit with a full conversation and its draft model, in 8-bit only with shorter ones.
Flash-Next needs a 128 GB Mac: on 48 and 64 GB its weights alone exceed the budget, and on 96 GB (about 72 GB) its fixed
part with an 8K cache (74.7 GB) does not fit, so it is refused before the download. For Bolt and the other variants,
run `kapitsa doctor`.

A Flash-Next folder whose routed experts are in Q2_0 (2.25 bits a weight; `tools/qwen4_exp_q20.py` builds one from
ISTA-DASLab's GSQ-RCO GGUF and Nace AI's 4-bit pack: 38.3 GB of model files, plus the 32 GB n-gram table) does not
need them all in memory: it keeps as many as the budget leaves after its other weights, its buffers, the slots'
state and one conversation's KV cache of up to 32K tokens, with the process's GPU memory within 40% of the RAM, and
reads the others from the SSD when a token routes them (`src/model/expert_store.h`). Why 40%: such a store waits on
the CPU at every layer, and on a 48 GB M4 Pro every command buffer took 0.5-3.5 s longer once the process's GPU memory
passed about 45% of the RAM (15-24 GB stores), while 4-12 GB stores ran at full speed.
`kapitsa doctor` counts the smallest such store (2.1 GB) in its weights and lists the rest as streamed.
`NI_EXPERT_STORE=<GB>` sets the store's size (any size), `all` keeps every expert, `off` loads them as stacked tensors
(the whole pack must then fit); `NI_EXPERT_READERS` (default 16) sets the SSD reader threads. With `NI_LOG=1` each
request's log says how many of its experts were in memory and how much was read.

On that 48 GB M4 Pro (macOS 27.0.1; a 12 GB store, 35% of the experts, the automatic size there; 17-20 GB peak
memory), with the published protocol's prompts: 512-token code answers at 16.8 tok/s greedy and 17.8 at T = 0.6 with
MTP, 16.1 and 16.4 without; prefill 265-284 tok/s at 4K and 237 at 16K, with `--prefill-chunk 4096` 357 and 407; a
follow-up's first token in 0.7-1.0 s. Decode finds ~90% of its experts in memory and reads 105-200 MB a token; a
prefill chunk reads nearly every expert (~40 GB), so 4096-row chunks halve the reads a token. Any prompt of 64 tokens
or more reads every expert of each layer: an 85-token one waits 6-8 s for its first token. The n-gram table of those
runs held only the rows of their prompts and (mostly) their answers; a full table adds reads of its rows for other
text.

Super-VL's weights alone are 70.1 GB, so it needs a 96 GB+ Mac: on 128 GB (a 110.8 GB budget) it runs with its draft
model (3.5 GB at two slots) and room for millions of tokens of KV cache; on 96 GB it starts without the draft model
(about 29K tokens of KV cache in all at a 72 GB budget) or with it from an 80 GB budget; on 48 and 64 GB Macs it is
refused before the download (`... does not fit this Mac: its weights alone (70.1 GB, ...) exceed ...`).

Lightning's weights are 18.2 GB (its MTP head is not loaded): on 16 GB Macs and on 24 GB at the smaller budget it is
refused before the download, on 24 GB at 19.3 GB it starts without its draft model (79K tokens of KV cache in all at
`--context 64k`), on 32 GB it runs with the draft model (0.7 GB at two slots) and 370K-970K tokens of KV cache, and from
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
| Writing | A 512-token coding answer, at temperature 0.6 (Flint at 1.0, its model card's; the result records which) and greedy, 5 runs each (3 with `--quick`), median; the rate over the middle of the stream |
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
path. A run takes 3 to 15 minutes and holds the GPU lock that the benchmark harness uses (`/tmp/kapitsa-gpu.lock`), so
it never runs beside a harness run.

At the end of a run in a terminal (or with `--submit`) it asks once whether to add the result to the community
results: Enter is yes, `n` keeps it on your Mac only. With [gh](https://cli.github.com) logged in it opens a pull
request from your fork (it uses your gh login and nothing else); without gh nothing is sent and it says how to get gh
(results are added only as pull requests from your fork, never as issues). `kapitsa bench submit [FILE]` sends a saved result (the
newest by default), and `--submit --dry-run` prints exactly what it would send, with the gh commands and the pull request,
without sending anything. Only results of the catalog's models are taken, at any precision. How results are checked:
[bench/community](../bench/community/README.md).

### Comparing engines on this Mac

`kapitsa bench compare` runs the same protocol against the other local engines that are installed here, one at a time,
and ranks them. Nothing is installed or downloaded for you, nothing is sent anywhere, and nothing is shared unless you
paste the report yourself.

```bash
kapitsa bench compare                 # asks which model (4b, 27b, 35b) in a terminal; 4b elsewhere
kapitsa bench compare --list          # which engines it found, which are missing and how to get them; measures nothing
kapitsa bench compare --model 4b --engines kapitsa,omlx,mlx-serve
kapitsa bench compare --quick         # 4K prompt only, 2 decode runs: about 3 minutes per engine
kapitsa bench compare --json          # JSON on stdout (progress goes to stderr)
kapitsa bench compare --report        # a Markdown report to keep, or to quote in a PR
```

It looks for oMLX, mlx-serve, mlx-lm, llama.cpp, Ollama, LM Studio, MTPLX, Rapid-MLX and mlx-vlm on your `PATH` and at
the usual `OMLX_BIN`, `MLX_SERVE_BIN`, `MLX_LM_SERVER`, `LLAMA_SERVER`, `OLLAMA_BIN`, `LMS_BIN` and so on. For each one
it runs the best configuration the [harness](../bench/harness/README.md) has for the model (its speculative decoding on,
when it has one, the way that engine is meant to be run) through `bench/harness/compare_engines.py`, under the GPU lock, with
thinking off and the same sampler, prompts and corpus for everyone. An engine that is not installed, or whose weights
are not on this Mac, is listed under "skipped" with the reason and how to get it. One engine failing does not stop the
others.

The table ranks engines by decode tok/s, with prefill tok/s, time to first token (cold and cached) and peak memory; the best value in each column is green. Below it, one line per measure says where Kapitsa stands (1st or not) and by what factor against the best other engine.
Files go to `~/.cache/kapitsa/bench/compare/<time>/` (`--out` changes that): `compare.json`, `report.md` and logs.

The report has a "Who beats Kapitsa here, and why" part; its reasons come only from the configurations (speculative
decoding, weights folder, versions) and are leads, not findings. **Run it on a quiet Mac, plugged in**: a run waits up
to 5 minutes per engine for other apps to leave the GPU and marks a number it could not get quiet. It refuses to start
while another engine or model server is running (`--ignore-running` overrides). If another engine beats Kapitsa on your
Mac, the report is yours to keep or to quote in a pull request; nothing is sent anywhere. The harness ships with
Kapitsa, in `share/kapitsa/bench/harness` beside the binary (`KAPITSA_HARNESS` names another copy), so an `install.sh`
or Homebrew install needs no clone; it needs macOS Python 3.

## Running in the background

Homebrew arrives with the first release. `brew services start kapitsa` then runs `kapitsa serve` (the best model for
this Mac, on `127.0.0.1:8000`) in the background and at each login, restarting it after an error;
`brew services stop kapitsa` stops it. Its log, the first download's progress too: `/opt/homebrew/var/log/kapitsa.log`.

## Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `KAPITSA_HOME` | `~/.cache/kapitsa` | Kapitsa's folder: models in `models/`, the coding-agent servers' logs in `logs/`, `kapitsa bench`'s results in `bench/`, the last model chosen in `state.json`, the automatic update's state in `update.json`, KFlash's measurements (this Mac's cost per verified row count and the draft model's calibration, per model, draft model and chip) in `kflash/`, saved as the server runs and when it stops, the [SSD prefix cache](#ssd-prefix-cache) in `prefix-cache/`. |
| `KAPITSA_NO_AUTO_UPDATE` | | `1`: no automatic update and no check in the background ([Updating](#updating)); `kapitsa update` still works. |
| `KAPITSA_UPDATE_URL` | `https://github.com/nace-ai/kapitsa/releases` | Where updates come from: a GitHub releases page or a copy of one (https, or http to this Mac). |
| `KAPITSA_UPDATE_TIMEOUT_S` | `300` | The background check's time limit, in seconds. |
| `KAPITSA_NO_PROMPT` | | `1`: never ask which model to start; without `--model`, the best model for this Mac ([Which model](#which-model)). |
| `BENCH_GPU_LOCK` | `/tmp/kapitsa-gpu.lock` | `kapitsa bench`: the GPU lock it takes, the benchmark harness's; `NACE_GPU_LOCK_HELD=1` when the caller holds it. |
| `BENCH_GPU_AMBIENT` | | `kapitsa bench`: more process names whose GPU use is ambient (recorded, never flagged), comma-separated, as for the harness. |
| `KAPITSA_BENCH_REPO` | `nace-ai/kapitsa` | `kapitsa bench --submit`: the repository results go to. |
| `HF_TOKEN` | the token `hf auth login` saved | Hugging Face token for gated or private repos. |
| `HF_ENDPOINT` | `https://huggingface.co` | A Hugging Face mirror. |
| `HF_HOME`, `HF_HUB_CACHE` | `~/.cache/huggingface`, `$HF_HOME/hub` | The Hugging Face cache, whose files downloads reuse. |
| `NI_LOG` | `1` | `0` quiet, `1` a line per request with its timings, `2` debug. |
| `NI_DFLASH_CORE4` | Auto | `1` selects the newer four-row 4-bit drafter core, `0` the earlier core. Auto uses the newer core on Apple10 and the 32-core M1 Max. |
| `NI_MEM_BUDGET` | | The GPU memory budget in GB, as `--mem-budget`. |
| `NI_VPF` | on for a dense target that verifies 16 rows wide and is big enough for the pass to pay (weight scale `hidden*intermediate*layers` >= 3e9, which the 27B meets and the 4B does not; the middle is unmeasured, so it stays off) | A cached turn's short new suffix (one chunk of at most `NI_VPF_MAX` rows over a cached prefix of at least 1K rows, no grammar mask, image rows or MTP speculation) runs as verify passes - one pass per 16 rows, 8 where the target verifies narrow - instead of the prefill path's small-T pass, whose GEMMs stream the weights at 53-115 GB/s where a verify pass reaches 210 GB/s. A cached follow-up's time to first token drops 16-27% on the 27B at a 4K to 16K prefix. `1` also takes the path on any target the default leaves out (a MoE model, whose verify reads the union of its rows' experts; a target whose tables hold 8 rows; a small dense model, where the pass costs more than the narrow prefill it replaces - the 4B pays ~30-40 ms for it), `0` restores the prefill path everywhere. |
| `NI_VPF_MAX` | `128` | The longest cached suffix `NI_VPF` takes, in rows (clamped to 2047, the drafter's prompt-capture window). The longest suffix measured is 116 rows, and at a 16K prefix it costs 757 ms as verify passes against 994 ms on the prefill path (64 rows: 491 against 527), so the pass still wins there; 128 caps the default just above that measurement, and past it the two have not been compared. |
| `NI_VPF_WIDE` | on | `0` keeps `NI_VPF` passes at 8 rows on a target that verifies 16 wide: an experiment knob, not a measured configuration. |
| `NI_ANE` | on where supported | `0` keeps prefill on the GPU, without the Neural Engine. |
| `NI_ANE_MMA` | on for the Qwen 4B and 27B | M5 with macOS 26, 4-bit group-64 Qwen dense models: the Neural Engine split's default share for the GPU's FFN on the matrix units (27B: 24 of 34 units, 4B: 13 of 18). Measured on the 20-core M5 Pro only. Other models (the 9B) keep the older default share until measured. `1` takes it on any such model, `0` keeps the older one. This moves the share only: whenever the GPU keeps channels (any share below all units, a calibration, `NI_ANE_UNITS`), they run on the tensor-ops kernel, so the 9B and `0` are not fully pre-#318. |
| `NI_ANE_KEEP_S` | `300` | Seconds after the last prefill chunk (or model load) during which a background evaluation keeps the Neural Engine awake, so a short prompt after a pause skips its 70–120 ms wake. `0` turns it off. |
| `NI_ANE_KEEP_MS` | `2000` | Period of that keepalive evaluation in milliseconds (floor 100). |
| `NI_DFLASH` | `auto` | As `--dflash` when the option is absent: `auto`, `none` or a folder. |
| `NI_DFLASH_TREE` | on | KFlash on the Qwen draft models: the draft model's guesses are checked as a best-first tree per pass (several continuations at once) instead of one chain; `0` checks a chain (overrides `NI_KFLASH` on these models). Greedy replies are plain decoding's up to near-ties; sampled ones follow the same distribution. The tree's shape follows the draft model's probabilities as calibrated against this Mac's own acceptances, and its size this Mac's measured cost per verified rows: 8 rows per pass (16 or 32 where `NI_KFLASH_ROWS` names them, on the 27B and the 4B) (whichever the drafter's confidence makes pay on this Mac; the costs are measured in the first passes and kept per model and chip), 1 to 8 rows per pass on the 35B (its pass routes experts for those rows alone). |
| `NI_KFLASH_ROWS` | `8` | The tree sizes (verified rows) KFlash may pick on the dense Qwen models (27B, 4B), as a list of 8, 16 and 32. `8` keeps every pass at 8 rows (the cost this release is tuned for); `8,16,32` lets the budget measure 16- and 32-row verifies on this Mac and pick per cycle, where its cost table says they pay; alone, the 27B's measured tok/s gain is within noise (docs/kflash.md, "Eight rows"), and it is the 16-row drafting block that fills the wider rows with something worth verifying. That block is the default on a one-slot engine and sets the verify to 16 rows whatever this says (see `NI_DFLASH_BLOCK`), 32 included: a 16-row block drafts a 15-guess lattice, so a 32-row verify would check nothing the 16 does at twice the rows. The 35B-A3B and the Nemotron drafters always verify 1 to 8 rows. |
| `NI_DFLASH_BLOCK` | `auto` | The DFlash2 drafter's block on the dense Qwen models: `8` (7 guesses, the tile the verify is tuned for), `16` (an anchor and 15 guesses, drafted as two 8-row tiles from one weight read by the pair kernel, verified in 16-row tree passes), or `auto` (16 on a one-slot engine that takes the 16-row paths, 8 on a server with more slots: a 16-row block has no batched cycle, so requests decoding together would take turns instead of sharing a pass). `16` needs KFlash trees and a target with the wide verify, drafts on the matrix units only (M5-class GPU, macOS 26, 4-bit group-64 projections, not the v1 drafter), which is checked when the model loads, and pins the verify at 16 rows. It measured 76.07 tok/s greedy on the 27B's harness corpus with `8` against 87.59 on the default (+15%, ten interleaved runs each), and 71.70 against 79.77 at temperature 0.6 (+11%) (docs/kflash.md, "Sixteen-row drafting blocks"). Read while a drafter loads, once per drafter: a server that switches models decides each one's block again, and the engine's memory plan reserves the rows it will verify. `auto` may be written out; it is the default. |
| `NI_TURN_MS` | `30` | Requests at once: how long a decoding request keeps the GPU before it gives its turn to a waiting one, in ms; `0` = after every decode step. |
| `NI_BATCH` | on where measured | Requests decoding at once share one GPU pass ([Requests at once](#requests-at-once)): the 27B, the 35B and Bolt; `1`: the 4B, Lightning and Super-VL too (not measured yet); `0`: they take turns instead, each with the tokens it gets alone. |
| `NI_BATCH_KF` | on | Requests decoding at once with the Qwen draft models (2 to 4 of them): KFlash chooses each shared pass's rows, how many in all and which request gets them, from the draft model's calibrated probabilities and this Mac's measured cost of a shared pass (cached with the KFlash budget). `0`: a fixed number per request, as measured before (two requests 4 rows each, three or four 2 each; the 27B's three take turns). |
| `NI_BATCH_KF_FLOOR` | `0` | With `NI_BATCH_KF`: `1` gives every request in a shared pass at least one draft row. |
| `NI_HISTORY_PREFILL` | on | After a reply, while the server is idle, prepare the conversation's next turn so a follow-up starts sooner; `0` turns it off. |
| `NI_GRAMMAR_SPEC` | on | [Structured output](#structured-output) with a DFlash2 draft model (the Qwen models): drafts are still verified, each verify row under its own mask; `0` decodes the formatted part one token per pass. |
| `NI_DISK_CACHE` | | As `--disk-cache` when the option is absent; tools that start an engine without `kapitsa serve` keep no SSD prefix cache unless it is set. |
| `NI_DISK_CACHE_IDLE_MS` | `10000` | SSD prefix cache: how long the server is idle after a turn before it writes the conversation, in ms. |
| `NI_DISK_CACHE_MIN` | `1024` | SSD prefix cache: the shallowest state kept, in tokens. |
| `NI_DISK_CACHE_DIR` | `$KAPITSA_HOME/prefix-cache` | SSD prefix cache: the folder of its namespaces. |
| `NI_IMAGE_MAX_MB` | `32` | The largest image taken, in MB. |
| `NI_IMAGE_TIMEOUT_S` | `30` | The time limit for downloading an `http(s)` image, in seconds. |
| `NI_IMAGE_CACHE_MB` | `256` | GPU memory kept for the embeddings of recent images. |
| `NI_MOE_QMR_K` | Auto | Input staging width for native 4-bit group-64 expert prefill: `32` or `64`; `0` means auto. Auto uses paired 32-input staging (`64`) for hidden size 2048 and expert intermediate size 512 on the 32-core M1 Max; other cases retain `32`. |
| `NI_MOE_QTM` | on with the dense QMT path | Qwen3.6 35B-A3B routed expert prefill on Metal 4 tensor ops (Apple10 GPU, macOS 26, 4-bit group-64 experts; `src/kernels/tensor/qmt_moe.metal`): 64-row tiles of the MoE bucket, as the dense QMT prefill; `0` keeps the simdgroup kernels, and a Mac without the tensor units (or with `NI_QMT=0`) runs them anyway. 4K prefill 2,272 -> 1,702 ms and 16K 11,093 -> 8,834 ms on the M5 Pro (macOS 26.0, `d53f02e`, interleaved A/B, same prompt and power mode), 25% and 20% off the time to first token; the switch covers every Apple10 GPU on macOS 26, M5 Max and Ultra included, of which only the M5 Pro was measured. The tile reassociates each output's K reduction, so its logits are not the simdgroup path's: it is gated by `tools/logit_compare.py --reorder` (about 1.1e-3 mean KL, 1.3x the drift the simdgroup kernels already carry against the order before them), not by the model's noise floor. |
| `NI_QATTN_G8` | on, on an M5-class GPU with macOS 26 | Qwen3.6 35B-A3B prefill attention on the same Metal 4 tensor ops (`src/kernels/tensor/qattn_t.metal`, head dim 256, 16 query heads over 2 KV heads): a 16-row tile holds two queries and all 8 query heads of a KV head, 8 queries per 128-thread group; `0` keeps `attn_prefill11r`, and `NI_ATTN` overrides both. On the M5 Pro 48 GB (macOS 26.5.2), through `bench/harness` on `487ea80` with the experts already on the tensor units, three interleaved sessions: 16K prefill 1,801 -> 1,999 tok/s (+11%) and 4K flat (2,424 -> 2,476); on `330b216`, before the experts moved, the same A/B read 4K prefill 1,693 -> 1,636 ms and 16K 8,819 -> 7,929 ms (median of three), so the 16K rate goes from 0.75 of the 4K rate to 0.81. It keeps 11r's products (half x half is exact in fp32) and only reassociates the K reductions, but its decode rows move 5.6e-4 mean KL, which is 1.10x the 35B's noise-floor gate (`tools/logit_compare.py` FAIL, `--reorder` PASS). It is the default anyway, by the maintainer's call recorded in `AGENTS.md`, so on an M5-class GPU it departs from the gate by that much until `tools/logit_floors/qwen3.6-35b-a3b-4bit.json` (an M4 Pro measurement of the kernels before this) is re-measured here in a PR of its own. |
| `NI_NH_BF16` | off | Nemotron Super-VL: `1` keeps its BF16 latent projections in BF16 instead of 8-bit (exact; about 4% slower decoding). |
| `NI_NH_ATTN_Q8` | off | Nemotron Lightning: `1` stores its BF16 attention projections in 8-bit (about 3% faster decoding, slightly less accurate). |
| `NI_NH_HEAD` | `q8` with NVFP4 or FP8 | Dense Nemotron-H models (Nemotron Nano): `bf16` keeps an NVFP4 or FP8 checkpoint's BF16 LM head (the 9B v2's, the 4B FP8's) BF16 instead of 8-bit rows; `q8` also runs a BF16 checkpoint's head as 8-bit rows. |
| `NI_NH_NGRAM` | `4` with BF16 or FP8, `0` with NVFP4 | Dense Nemotron-H models (Nemotron Nano): prompt lookup's drafts per cycle when `--spec` is not given; `0` decodes plainly. |
| `NI_FLINT_NGRAM` | `0` | Flint: prompt lookup's drafts per cycle when `--spec` is not given (up to 7); `0` decodes plainly. |
| `NI_FLINT_MOE_QTM` | on with the dense QMT path | Flint: the routed experts' prefill GEMMs on Metal 4 tensor ops (Apple10 GPU, macOS 26, 4-bit group-64 experts; `src/kernels/tensor/qmt_moe.metal`'s `qmtm_g_m64_n64`): gate, up and down each as a grouped GEMM over the MoE bucket's 64-row expert tiles to fp32 rows, so `flint_swiglu_scaled` keeps its whole-row overflow scaling. `0` keeps `qmm_moe_b4_m16_n64` (16-row simdgroup tiles), which a Mac without the tensor units (or with `NI_QMT=0`) runs anyway. Prefill 1,392 -> 2,262 tok/s at 4,000 tokens and 1,128 -> 1,635 at 16,000 through the harness on an M5 Pro (decode unchanged); the routed experts were 60% of a 4K prefill. The tile reassociates each output's K reduction, and Flint carries that further than the 35B: 1.4e-2 mean KL, p99 0.22 and 3 top-1 flips (all near-ties) in 192 verify rows against `NI_FLINT_MOE_QTM=0` (`logit_dump`, M5 Pro), the size of the drift the shipped dense QMT path already gives Flint (1.7e-2, 1 flip, against `NI_QMT=0`). That passes `tools/logit_compare.py --reorder` (`flint-4bit.m5.json`'s order-change gate) on the prefill, step and verify rows; against the model's M5 noise-floor gate the mean KL and flips pass and the p99 does not (0.212 step, 0.219 verify, over 0.192). |
| `NI_FLINT_QATTN` | on with the dense QMT path | Flint: the prefill attention on the matrix units (`tensor/qattn_t_flint.metal`'s `qattn_flint_s8k32p`, 16 tokens x the KV head's 8 query heads per threadgroup, 32-key tiles, latest blocks first) instead of `flint_attn_prefill`'s 8x8x8 simdgroup MMAs: the same half operands, fp32 scores and statistics, sinks, two key sources and sliding window, only the products' summation order differs. Prefill 2,241 -> 2,691 tok/s at 4K, 1,621 -> 2,457 at 16K and 1,158 -> 2,086 at 32K on an M5 Pro (`compare_engines` ABBA, means of 2 passes; decode unchanged). Against `NI_FLINT_QATTN=0` (`logit_dump`, 192 verify rows): KL mean 2.0e-2, p99 0.48, 4 top-1 flips (all near-ties); that passes `tools/logit_compare.py --reorder` (`flint-4bit.m5.json`'s order-change gate) on the prefill, step and verify rows and fails the model's M5 noise-floor gate on top-1 (0.979) and p99. `0` runs the simdgroup kernel, and a GPU before Apple10 or a macOS before 26 takes it anyway. |
| `NI_NH_FP4` | `l2` on M1 Macs, else `l0` | Dense Nemotron-H models: how NVFP4 codes are laid out for the GPU, as `NI_DORY_FP4`; both give the same outputs to the bit. |
| `NI_DSPARK_DYN` | `0.3` on M5 Macs, else `0.2` | Nemotron Lightning's DSpark draft model with chains (`NI_KFLASH=0`): how many of its 7 drafts are verified, chosen per cycle as for `NI_LILI_DYN`; `0` verifies all 7. |
| `NI_DSPARK_K` | `16` | Nemotron Lightning's DSpark draft model: the candidates per draft position its Markov head rescores (1-32). |
| `NI_LILI_DYN` | `0.04` | Nemotron's LiLiCorr draft model with chains (`NI_KFLASH=0`): how many of its 7 drafts are verified is chosen per cycle, weighing their expected acceptance against each extra row's cost (this value, relative to a cycle's fixed cost); `0` verifies all 7. |
| `NI_KFLASH` | on | KFlash on every draft model (the Qwen 27B, 4B and 35B, Nemotron's LiLiCorr and DSpark models): the draft model's guesses are checked as a best-first tree per pass instead of one chain; the tree's size comes from this Mac's measured cost per verified row count, its shape from the draft model's probabilities as calibrated against its own acceptances. On MoE targets (the Qwen 35B, Nemotron's), where every verified row costs experts, each pass verifies 1 to 8 rows, as many as their expected tokens are worth; the dense Qwen models verify whole 8-row tiles, 16 or 32 rows where `NI_KFLASH_ROWS` names them and the Mac's cost table says they pay, and 16 rows when the 16-row drafting block is on (a one-slot engine by default). `0` checks a chain on every model. Replies are plain decoding's up to near-ties (greedy: 17 of the harness's 20 runs wrote plain decoding's reply, the other three took the corpus's one near-tie branch; sampled: by the same law). |
| `NI_DORY_HEAD` | `q8` with `nvfp4` | Dory: `bf16` keeps the `nvfp4` checkpoint's LM head BF16 (about 7% slower writing; next-token KL 7e-6 from the 8-bit rows it runs as by default); `q8` also runs the `bf16` checkpoint's head as 8-bit rows. |
| `NI_DORY_FP4` | `l2` on M1 Macs, else `l0` | Dory: how the `nvfp4` checkpoint's codes are laid out for the GPU. `l2` decodes 12-16% faster on the M1's GPU; both give the same outputs to the bit. |
| `NI_DORY_TG` | on, on an M5-class GPU with macOS 26 | Dory: the prefill GEMMs over 32 rows on the matrix units (`tensor/nh_gemm_t.metal`'s `nh_gemm_t_fp4_m64`, the Nemotron-H models' own choice above 32 rows) instead of `nh_gemm_fp4_m64`'s simdgroup MMAs: the same 64-row tiles, threadgroup dequantization and epilogue, only the summation order inside the matrix unit differs (output check against `NI_DORY_TG=0`: prefill KL mean 2.3e-6, step 1.3e-6, top-1 agreement 1.0000, well inside this model's floor). Prefill 6,930 -> 2,734 ms at 4,132 tokens and 30,091 -> 14,031 ms at 16,493 on an M5 Pro, the 64-row GEMM's share of the step 86% -> 47%; `0` runs the simdgroup kernels, which a GPU before Apple10 or a macOS before 26 takes anyway. The prefill attention has a switch of its own (`NI_DORY_ATTN_T`), so this one moves only the GEMMs. |
| `NI_DORY_ATTN_T` | on, on an M5-class GPU with macOS 26 | Dory: the prefill attention on the matrix units (`tensor/nh_dory_attn_t.metal`'s `dory_attn_t_g4s8k16`, 32 tokens and 4 KV heads per threadgroup, 8 simdgroups) instead of `dory_attn_prefill`'s 8x8x8 simdgroup MMAs: the same fp32 scores, the same two key sources, sliding window and fp32 store, only the products and their summation order differ (output check against `NI_DORY_ATTN_T=0`: top-1 agreement 1.0000, no flips, prefill KL mean 1.2e-6 and step 1.5e-6 over 2 prefill windows, 2.6e-6 and 5.3e-6 over 6 windows, where the sliding layers' ring wraps; the order-change gate, and 600 times inside this model's floor of 1.4e-3). The attention 717 -> 579 ms at 4,132 tokens and 6,644 -> 5,468 ms at 16,493 on an M5 Pro, prefill 2,636 -> 2,534 and 14,160 -> 13,250 ms (interleaved medians), 1,213 -> 1,293 and 1,030 -> 1,131 tok/s through the harness; `0` runs the simdgroup kernel, `NI_DORY_ATTN_NS=4` the narrower 16-token threadgroup, and a GPU before Apple10 or a macOS before 26 takes the simdgroup kernel anyway. Independent of `NI_DORY_TG`. A prefill of fewer than `NI_DORY_ATTN_T_MIN` rows (default 64) keeps the simdgroup kernel on both settings: that is one 32-row threadgroup per block either way, and the tensor kernel's staged tiles then cost more than they save — the cached follow-up of an agent turn at 16K took 0.219 s against 0.159 s (+60 ms, +38%) and at 4K 0.124 s against 0.105 s in the interleaved A/B above, so those rows now run the kernel they ran before this switch. Cold prefill is thousands of rows at a time and always takes the tensor kernel. |
| `NI_DORY_SPEC` | `both` | Dory with `--loop-kv per-loop`: the drafts of speculative decoding, `both` (prompt lookup, else the model itself at one loop), `ngram` (prompt lookup only) or `self`; `--spec 0` decodes plainly. |
| `NI_DFLASH_V8` | Auto | `1` uses the target eight-row projection kernels in the drafter, `0` the earlier path. Auto includes hidden size 2048 on the 32-core M1 Max with an MoE target and native 4-bit head, in addition to hidden sizes at least 2560. This does not change speculation depth. |
| `NI_PREWARM` | on | `0` creates the GPU pipelines on first use instead of at load. |

The other `NI_*` switches exist for development and A/B tests.

## Updating

`kapitsa update` updates Kapitsa to its latest release, the way it was installed:

| Installed with | `kapitsa update` | Automatically |
| --- | --- | --- |
| `install.sh`, or a copied release binary | Downloads the release and puts it in place (below). | Installed within a day; the next run in a terminal says so. |
| Homebrew (`/opt/homebrew/Cellar/kapitsa`) | Runs `brew upgrade nace-ai/tap/kapitsa` in a terminal (`--fetch-HEAD` for a `--HEAD` install); elsewhere it prints that command. | A run in a terminal says when a release is out. |
| A build from source (a `CMakeCache.txt` beside the binary or above it) | Says how: `git -C SRC pull && cmake --build BUILD`. | A run in a terminal says when a release is out. |
| `./install.sh` in a clone (a binary built from source, copied) | Replaces it with the release build. | A run in a terminal says when a release is out. |
| An app such as Nace Agent (a binary inside `*.app/Contents/`) | Nothing: the app keeps its own Kapitsa. | Nothing. |

An update of an install.sh install downloads the release's tarball and `.sha256` from GitHub into a new folder beside the
binary and checks the tarball against the checksum. It checks the new binary's code signature (valid, signed as
Kapitsa releases are, and by the same Developer ID team as the running binary when that one has one) and that its
`--version` is the release's. Then it renames it over the old binary in one step, after checking that no `kapitsa bench`
is running from it (a benchmark's servers must all run one binary) and that the file there is still the kapitsa that
runs. The old binary is kept as `kapitsa.previous` beside the license files (`../share/kapitsa` from the binary's
folder, where `install.sh` puts them, updated too, with the benchmark harness in `bench/harness` there: your own
`local.env` in it is kept, other edits are replaced, and `--rollback` leaves the newer harness). A release that fails a check changes nothing. A server that is
running keeps the version it started with until it restarts. When the binary's folder is not yours to write (say
`/usr/local/bin`), `kapitsa update` prints the command that works: `sudo /usr/local/bin/kapitsa update`.

- `kapitsa update --check` only says whether a newer release is out: exit 0 when this one is up to date, 100 when a
  newer one is out, 1 when it can't tell. A build newer than the latest release (or a dev build of it) is up to date.
- `kapitsa update --rollback` puts `kapitsa.previous` back, and keeps the binary it replaces in its place (a second
  `--rollback` returns to it). Automatic updates then skip the release rolled back from, until a newer one is out.
- `kapitsa update --version X.Y.Z` installs that release, an older one too (nothing else ever installs an older one).
- `kapitsa update --auto off` (and `on`) turns automatic updates off for this `KAPITSA_HOME`.

Automatic updates: once a day, a `kapitsa` command (not `--help`, `--version`, `kapitsa update` or `kapitsa bench`)
starts a check in the background: a separate process in its own session, at background priority, with a time limit of
5 minutes (`KAPITSA_UPDATE_TIMEOUT_S`). The command itself does not wait for it and prints nothing more. For a
Homebrew install, a build from source, a binary built from source, or one in a folder you can't write, the check only
runs from a command in a terminal and only records a release that is out. The next run in a terminal prints one line: `Updated Kapitsa to
0.1.33 (from 0.1.32); restart running servers to use it`, or `Kapitsa 0.1.33 is available: brew upgrade
nace-ai/tap/kapitsa`, or why a release was not installed. A check that can't reach GitHub stays silent and runs again
the next day. `KAPITSA_NO_AUTO_UPDATE=1`, `kapitsa update --auto off` and `CI` (set, and not `0` or `false`) turn
automatic updates off; they are also off for root.

What each binary's last check found is in `~/.cache/kapitsa/update.json` (`KAPITSA_HOME`): when it ran, the latest
release, the result, and a release to skip after a rollback.

Privacy: a check asks only `github.com/nace-ai/kapitsa/releases/latest` where the latest release is (a HEAD request,
with no GitHub API and so no rate limit). An update downloads two files from the release, through GitHub's file host
(`*.githubusercontent.com`); no other redirect is followed. Requests carry a fixed User-Agent (`kapitsa`) and
language, no cookies and nothing about this Mac. There is no telemetry.

Automatic updates start with the release after 0.1.32. Kapitsa 0.1.32 and older update once by hand: run the installer
again (`curl -fsSL https://raw.githubusercontent.com/nace-ai/kapitsa/main/install.sh | bash`), or
`brew upgrade nace-ai/tap/kapitsa`, or `git pull && ./install.sh` in a clone.

## Uninstalling

`brew services stop kapitsa` if you started the service, then `brew uninstall kapitsa`; or delete the binary
`install.sh` placed (in `/opt/homebrew/bin`, `~/.local/bin` or your `--prefix`) and `../share/kapitsa` next to that
folder (the license files, the benchmark harness, and `kapitsa.previous` after an update). Deleting `~/.cache/kapitsa` (or `$KAPITSA_HOME`)
removes the models, logs, update state and the SSD prefix cache (`prefix-cache/` alone: the cache).
