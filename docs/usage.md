# Using Kapitsa

The reference for the `kapitsa` command and its server. To install Kapitsa, see the [README](../README.md); to build it
from source, [CONTRIBUTING.md](../CONTRIBUTING.md); for how the engine works, [architecture.md](architecture.md).

## Commands

| Command | What it does |
| --- | --- |
| `kapitsa serve [--model M] [options]` | Starts the server: the chat page at `http://127.0.0.1:8000`, the APIs under `/v1`, visible to Kapitsa on the other Macs of the network ([Several Macs](#several-macs)). Without `--model`, a terminal asks which model ([Which model](#which-model)); elsewhere it is the best model for this Mac ([Server options](#server-options)). |
| `kapitsa cluster [--json] [--watch]` | The Macs on this network that run Kapitsa: the model each serves, how, how fast, and the link to it ([Several Macs](#several-macs)). `kapitsa macs` is the same. |
| `kapitsa claude`, `kapitsa codex`, `kapitsa opencode`, `kapitsa pi` | Starts that coding agent on Kapitsa ([Coding agents](#coding-agents)). |
| `kapitsa pull M... [--from DIR]...` | Downloads models without starting them. `--dry-run [--json]` shows what `M` resolves to and downloads nothing. |
| `kapitsa list [--json]` | The downloaded models: size on disk, complete or not, MTP head and draft model. |
| `kapitsa rm M...` | Deletes downloaded models and says how much space that gave back. Folders outside the store are left alone. |
| `kapitsa import FILE.gguf [options]` | Converts a GGUF file into a model folder ([GGUF files](#gguf-files)). |
| `kapitsa doctor [--model M]... [--json]` | This Mac and which models fit it ([Memory and `kapitsa doctor`](#memory-and-kapitsa-doctor)). |
| `kapitsa bench [--model M] [--prefill-chunk N] [--quick] [--submit]` | Benchmarks Kapitsa on this Mac with the published protocol, and adds the result to the community results when asked ([Benchmarking this Mac](#benchmarking-this-mac)). |
| `kapitsa update [--check] [--version X.Y.Z] [--rollback] [--auto on\|off]` | Updates Kapitsa to its latest release; it also updates itself ([Updating](#updating)). |
| `kapitsa stage --model M [--layers A-B]` | Holds a run of Flash-Next's layers for another Mac's `kapitsa serve --split` ([Flash-Next across Macs](#flash-next-across-macs)). |
| `kapitsa --help`, `kapitsa COMMAND --help`, `kapitsa --version` | Help and version. |

Exit codes: 0 success, 1 failure (download, port, model), 2 usage error (unknown option or model), 130 after Ctrl+C in a
download (the parts are kept) or when the model question is cancelled; the coding-agent commands exit with the agent's
code; `kapitsa update --check` exits 100 when a newer release is out.

### Which model

Run in a terminal without `--model`, `kapitsa serve` and the coding-agent commands ask which model to start: the models
with their download size, ✓ where downloaded and ★ on the best for this Mac, then the chosen model's precisions with
their sizes and whether each fits this Mac's memory (the engine's own plan of its headers; a model or precision that
does not fit is marked too big and can still be chosen, with a warning). Each shows the context it would run with: Tab
(Shift+Tab back) changes it on both screens, between the default ([Context](#context): the model's limit where memory holds it), 32K, 128K, 256K and max, the
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
| `nanbeige4.3-33b` | `nanbeige`, `nanbeige4.3`, `nanbeige-33b` | Nanbeige4.3-33B: a dense 33B (Llama architecture) for agents and coding, reasoning switched with `enable_thinking`; text only; 32 GB Macs and up in 4-bit | 19.8 GB | `4bit` (default: AWQ, 8-bit embedding and LM head), `8bit` (35.2 GB), `bf16` (66.3 GB) |

A download includes the companion files that speed decoding up: an MTP head and a DFlash2 draft model for the 35B and
27B, an MTP head and z-lab's DFlash draft model for the 4B, NVIDIA's LiLiCorr draft model for Bolt and for Super-VL,
its DSpark draft model for Lightning; Flash-Next's repo holds its MTP head. Dory, Flint and Nanbeige have none: they decode without one, as
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
4, 5, 6 or 8-bit with groups of 64 (4-bit also 32), mixed recipes included, except ones whose dense MLP gives
`gate_proj` and `up_proj` different formats (as `mlx-community/Qwen3.8-27B-OptiQ-4bit`): those are refused when they
load. Kapitsa reads the repo's `config.json` before any weights, then downloads its top-level files except other formats
(`.gguf`, `.bin`, images and the like). A catalog variant's repo (`mlx-community/Qwen3.6-35B-A3B-6bit`) comes with that
variant's MTP head and draft model; other repos use the MTP head they contain, if any.

A folder needs `config.json`, the `*.safetensors` weights and `tokenizer.json`, and may hold an MTP head in
`mtp.safetensors`. The official Hugging Face checkpoints of the dense models run as well. A Qwen3.8-Flash-Next
(`qwen4_exp`) folder also needs its `ple.safetensors` table.

NVIDIA's dense Nemotron-H checkpoints run from a folder too (`kapitsa serve --model DIR`): Mamba-2, attention and dense
MLP layers, no experts, as [Nemotron Nano 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) (BF16) and
[Nemotron Nano 9B v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4) (NVFP4, with its attention, first
and last layers and LM head left BF16; the head loads as 8-bit rows, `NI_NH_HEAD`). `tools/nh_fp4_requant.py` writes
a copy of that checkpoint with those attention, first and last layers in NVFP4 as well (the head as it is): a token
reads 4.97 GB in place of 6.11 and decodes 1.20x as fast on an M3 Pro, at a mean KL of 0.0169 against the BF16 release
where NVIDIA's checkpoint gives 0.0116 (top-1 93.7% against 95.1%); serve its output folder with `--model DIR`. They
have no draft model. The 4B
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

Nanbeige4.3-33B (`--model nanbeige`: Nace AI's MLX packs of [Nanbeige/Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B),
[4-bit AWQ](https://huggingface.co/kapitsaprime/Nanbeige4.3-33B-MLX-4bit) by default and
[8-bit](https://huggingface.co/kapitsaprime/Nanbeige4.3-33B-MLX-8bit), or `:bf16` as published; KL from bf16 0.163 and
0.015 on 768 rows, top-1 92.3% and 98.6%, the repos' cards have the tables) is a dense Llama: 52 layers of full attention (56 query and 8 KV heads of 128, RoPE base 1.2e8) and a
SwiGLU MLP of 22,016. It runs on Flint's backend as a Flint without sinks, q / k norms, experts or sliding windows
(`model_type` `llama`; an MLX affine 4-bit or 8-bit group-64 conversion of it runs too). Its tokenizer is
SentencePiece-style BPE (spaces as U+2581, characters outside the vocabulary as byte tokens), read from its
`tokenizer.json`; its chat template writes Qwen's markup: `<think>` reasoning, on unless `enable_thinking` is false, and
XML tool calls in `<tool_call>`, which come back as tool calls. It keeps every past turn's reasoning in the prompt
unless `preserve_thinking` is false. No draft model: plain decoding, prompt lookup with `--spec N`.

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
addresses. A model this Mac would not hold whole that another Mac on the network runs whole is neither downloaded nor
loaded: the server sends its requests there ([Several Macs](#several-macs)). In a terminal the engine's log goes to
`~/.cache/kapitsa/logs/serve-PORT.log` (`--verbose` shows it on screen too), otherwise to stderr; Ctrl+C stops it.

| Option | Default | Effect |
| --- | --- | --- |
| `--model MODEL` | the best model for this Mac | A name, `name:variant`, `org/name[@revision]`, a folder or a `.gguf` file ([Models and downloads](#models-and-downloads)). |
| `--host HOST` | `0.0.0.0` | Listen address: every address, this Mac's and its network's ([API keys and network access](#api-keys-and-network-access)). `127.0.0.1` keeps the server to this Mac, as `--private` does. |
| `--port PORT` | `8000` | Listen port. A port in use is an error, on any address. |
| `--private` | off | This Mac only: listen on `127.0.0.1`, stay out of sight of Kapitsa on the other Macs and never send them requests ([Several Macs](#several-macs)). |
| `--local` | off | Run the model on this Mac even when another Mac on the network runs it whole ([Several Macs](#several-macs)). |
| `--api-key KEY` | none | Require `Authorization: Bearer KEY` or `x-api-key: KEY`. |
| `--allow-switch` | off | Let a request naming another model, or `POST /v1/models/switch`, load that model (it may download gigabytes). Off, the `model` field is ignored, as clients send names of their own. Without `--api-key` it keeps the server on this Mac, as `--private` does, and says so; with a network `--host` it needs `--api-key`. |
| `--model-id ID` | the model's name | The model's name in the API. |
| `--context N` | the model's limit (256K, 1M), less where memory is tight | Context per conversation: tokens (`131072`), `k` or `m` of them (`128k`, `1m`), or `max`, the model's own limit ([Context](#context)). `--ctx N` works too. |
| `--slots N` | `4` where memory allows, else `3` or `2` | Requests that run at once, each with its conversation's cache kept for follow-ups; more wait for a free slot, first come first served. |
| `--dflash auto\|none\|DIR` | `auto` | Draft model: `auto` uses the one downloaded with the model, unless it would keep one `--context` conversation from fitting; `none` decodes without it (with the MTP head on Qwen); `DIR` names a drafter folder. |
| `--spec N` | `1` on the 35B, `4` on the 27B and 4B | MTP draft depth when no draft model is used; `0` turns speculative decoding off. Dory with `--loop-kv per-loop`: drafts per cycle (by default 2 at 1-2 loops, 3 at 3-4; [speculative decoding](#dory-thinking-depth)). Dense Nemotron-H (Nemotron Nano): prompt lookup's drafts per cycle (by default 4 with BF16 or FP8 MLP layers, off with NVFP4). Flint and Nanbeige: prompt lookup's drafts per cycle (off by default). |
| `--mem-budget GB` | macOS's recommended working set, a little less | GPU memory Kapitsa may hold ([Memory and `kapitsa doctor`](#memory-and-kapitsa-doctor)). |
| `--prefill-chunk N` | `2048` (`1024` when memory is short) | Prompt tokens per prefill pass. On a 32-core M1 Max and on the M5 (Apple10 GPUs), a single slot with the 35B-A3B 4-bit group-64 model uses `4096` when its memory plan fits. An MoE model's pass holds up to 1/32 more rows than the chunk (`2112` for `2048`), and the prompt is split evenly over the passes it needs, so a prompt a few tokens past a multiple of the chunk does not pay a pass of its own for them. `NI_BAL=0` restores the plain chunking. |
| `--disk-cache SIZE` | `auto` | The SSD prefix cache: conversations kept on the SSD, also across restarts ([SSD prefix cache](#ssd-prefix-cache)). `auto`: a quarter of the free space beyond 10 GiB, at most 32 GiB; a size (`16G`, `512M`) caps it within that; `0` turns it off. |
| `--loops N` | `2` | Dory: how many times its shared block runs per token, 1-4 ([thinking depth](#dory-thinking-depth)); requests may choose their own. |
| `--loop-kv MODE` | `last-loop` | Dory: the shared block's KV cache, `last-loop` (one per layer) or `per-loop` (one per loop, more memory). |
| `--timing WHAT` | `both` | What the request log and the chat page show of the wait before the first token: `ttft` (the time to the first token), `prefill` (prompt tokens read per second) or `both`. The rate counts only the prompt tokens that were not served from the conversation's cache, over the engine's own prefill time: a follow-up that adds 80 tokens to a cached 20K prompt shows its 80 tokens' rate, and the chat page says "uncached". The API reports the time as `usage.kapitsa.prefill_ms` (OpenAI chat completions), so a client's rate is `(prompt_tokens - cached_tokens) / prefill_ms` (`usage.kapitsa.kflash` is `{"source", "cycles"}`: the KFlash proposal source of the model's decode and the cycles the request ran, left out with `NI_KFLASH=0` or `NI_KFLASH_WRAP=0`); `prefill_ms` is left out when the prompt had images (their encoding runs before the clock starts) or the request gave up its turn to another one while prefilling (`--slots` above 1), since the time is then not the time to read those tokens, and the request log shows `–`. |
| `--image-max-pixels N` | `1200000` (Bolt and Super-VL `1400000`) | Larger images are resized to about N pixels; `0` = the model's own limit ([Images](#images)). |
| `--verbose` | off | In a terminal, show the engine's log on screen too. |
| `--yes` | off | In a terminal without `--model`, start the best model for this Mac instead of asking ([Which model](#which-model)). |

### Context

`--context` sets how long one conversation can get: its prompt, its history and the reply share it. Without it, `kapitsa
serve` and the coding-agent commands take the model's own limit (256K for the Qwen models, 1M for Lightning and
Super-VL; 128K for a model that states none), or less where this Mac's memory is tight (the GPU's working set less a 4%
margin, or `--mem-budget`): the most, in 8K steps down to 32K, that keeps the draft model the model has at 32K, with one full
conversation beside it and, without `--slots`, the slots (3 or 4) the old 128K default would have got and, without `--prefill-chunk`, the prefill chunk (so a longer default never costs slots or prefill speed; a context
between 128K and the limit is taken only where it keeps them, and never less than 128K's own fit) (below 32K only where one slot cannot hold 32K tokens: Flint's `16gb` on a 16 GB Mac takes 8K).
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
| `POST /v1/chat/completions` | OpenAI Chat Completions: `reasoning_content`, `tool_calls`, and `usage` (in a stream's last chunk) with `prompt_tokens_details.cached_tokens`. `max_tokens` defaults to 16384 (reasoning, visible and tool-call tokens all count against it, so a client that sends no `max_tokens` is cut at 16384). Automatically switches to the requested `model` on demand if different from the currently active model. |
| `POST /v1/responses` | OpenAI Responses: reasoning items, function and custom tools, `usage.input_tokens_details.cached_tokens`. `previous_response_id` continues one of the last 64 stored responses (unknown ids are a 400). Built-in tools such as web search are left out. Supports on-demand model switching via `model`. |
| `POST /v1/messages` | Anthropic Messages: thinking blocks, `tool_use` and `tool_result`; `usage.input_tokens` counts the uncached prompt, `cache_read_input_tokens` the cached part. Server tools are left out. Supports on-demand model switching via `model`. |
| `POST /v1/messages/count_tokens` | Anthropic's token count of a request. |
| `POST /v1/draft` | Reads the user's draft of a conversation's next message ahead while they type, so the request, once sent, prefills only what was typed since: `{"model", "after", "text"}` ([below](#reading-ahead-while-the-user-types)). |
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

Drex decision models answer a state and a set of questions with a probability per option instead of generated text.
They are served by a separate program, `kapitsa-systemone` (`build/systemone` in a clone; `POST /v1/systemone`), not by
`kapitsa serve`: [docs/systemone.md](systemone.md).

### Reading ahead while the user types

An app can send the draft of a conversation's next message to `POST /v1/draft` at each pause in the typing. The
server then prefills the next request's prompt as far as the draft goes, so that the request, once sent, prefills only
what was typed after the last draft: a long paste is read while the user finishes the question under it. The idea is
MAFDET's ([TheLastOxygen/mafdet](https://github.com/TheLastOxygen/mafdet), on llama.cpp); the Qwen3.5 family's
recurrent layers cannot be cut back to any token, so Kapitsa keeps a checkpoint at the start of the user's turn instead.

```bash
curl http://127.0.0.1:8000/v1/draft -H 'Content-Type: application/json' -d '{"model": "qwen3.5-4b",
  "after": "<the text of the last assistant reply>", "text": "<a 2,000-token paste, and the question typed under it>"}'
{"status":"prefilled","prompt_tokens":5255,"cached_tokens":3186,"prefilled_tokens":2068,"ms":2545.2}
```

- **Which conversation.** The most recent of the last 32 replies the server ended with `stop` (any API) whose text
  equals `after`, both trimmed: the app sends no conversation id, only the reply and the draft.
- **What is read.** The next request as the server will render it (the conversation, that reply as clients send it
  back), then the user's turn holding the draft up to its last whitespace (the word being typed can still grow), less
  its last token. First a checkpoint is kept where the user's turn starts (where a request without drafts starts),
  with the drafter's context: a draft that extends the last one continues it, one edited anywhere or deleted goes back
  to that checkpoint and reads what one draft of its text would. The request continues the drafts' state when its
  prompt extends it (`usage.prompt_tokens_details.cached_tokens` counts what the drafts read), else resumes from the
  checkpoint.
- **Never in the way.** Read only while the server is idle (`busy` with a request in flight); a request that arrives
  stops the reading at the next prefill chunk, keeping what was read. Its KV cache grows within the slot's memory.
- **The same reply?** On a dense model's GPU path, yes: the Qwen3.5 4B writes the same tokens with drafts as without
  (`tests/test_draft.py`). A draft cuts the new message into other prefill passes; since `NI_SHORTPF_EXACT` a short pass
  (at most 128 rows) gives a row the same bits whatever its length, on the Qwen3.6 35B-A3B too. Before it, a MoE
  model's reply could take the other word where two nearly tie, as it did when a follow-up arrived before the history
  prefill (within its 250 ms idle wait, or with `NI_HISTORY_PREFILL=0`: the answer and the message read in one pass),
  and on a MacBook Pro 16-inch (M4 Pro, 48 GB), on a 3,234-token follow-up, all six of the 35B-A3B's short-message
  draft cases took another word, at the token where the one-pass history path did or later (the paste's reply stayed
  the same); over the reply tokens before it, the gap between the two best logits moved 0.007-0.045 on average with
  drafts against 0.038 for the one-pass path (at most 0.29 against 0.14, at one token; with an image 0.081 against
  0.080), and where a reply parts, the no-draft gap was 0.021-0.107: the one-pass path parts at 0.107 too, beyond the
  0.060 the model's noise floor counts as a near tie. The cause was how the last rows were cut into short prefill
  passes, not speculative decoding (plain decoding and MTP did too). A dense model's Neural Engine split (the 4B's
  default for chunks of 512 rows or more) computes the FFN in 8 bits, so after a long draft the request's last rows
  are the GPU's: in one series on the 4B a paste's reply changed, in the table's rounds it did not. `NI_ANE=0` keeps
  the prompt on the GPU.
- **Models.** The Qwen3.5 family with its drafters, the MTP head or none. Nemotron-H models, Dory, Flint and Nanbeige answer
  `unsupported`. Text without spaces (Chinese, Japanese) is read only up to its last whitespace.

The reply is always 200 (400: a body without `after` and `text` as strings; `"think": true`, not supported yet):

| Field | |
| --- | --- |
| `status` | `prefilled` (read), `unchanged` (nothing new), `interrupted` (a request arrived: stopped at a chunk, the part read kept), `busy` (a request in flight: nothing done), `no_conversation` (no recent reply is `after`, or its slot has since gone to another conversation), `unsupported` (this model, or an image in the conversation not yet read), `no_room` (beyond the context or the memory for its KV cache) |
| `prompt_tokens` | The prompt as far as the draft can be read; the slot holds all of it but its last token once read |
| `cached_tokens` | Of those, what the slot held before (the live state, or the checkpoint it went back to) |
| `prefilled_tokens` | Read by this call |
| `ms` | The call's time in the engine |

Measured on a MacBook Pro 16-inch (M4 Pro, 20-core GPU, 48 GB), macOS 27.0.1, by `tests/test_draft.py --measure
--rounds 3` (greedy, thinking off, the DFlash drafters, servers as by default, each case on a fresh server): a 5,315-
token first turn and its answer, then the next message with no draft, right after a draft of all of it (read up to its
last word), or of all but its last word; the time to the request's first token, sent right after a one-token request
of another conversation (the GPU awake). The replies were the same in all three cases, every round:

| Model | Next message | No draft | After a draft of all of it | Of all but its last word | The draft's own time |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.5 4B | ~40 tokens (prompt 5,425) | 0.072-0.073 s (5,381 cached) | 0.028-0.030 s (5,413) | 0.028-0.029 s (5,412) | 50-54 ms |
| Qwen3.5 4B | a ~2,000-token paste (prompt 7,461) | 1.27-1.28 s (5,381 cached) | 0.029 s (7,449) | 0.029-0.031 s (7,448) | 1.32-1.34 s |
| Qwen3.6 35B-A3B | ~40 tokens (prompt 5,407) | 0.076-0.078 s (5,363 cached) | 0.032-0.034 s (5,395) | 0.035-0.036 s (5,394) | 59-91 ms |
| Qwen3.6 35B-A3B | a ~2,000-token paste (prompt 7,443) | 2.02-2.11 s (5,363 cached) | 0.033-0.045 s (7,431) | 0.036 s (7,430) | 2.12-2.18 s |

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

The server listens on every address (`0.0.0.0`) and needs no key: any key a client sends is accepted, from this Mac
and from any machine that reaches it on the network ([Several Macs](#several-macs)). With `--api-key KEY`, every request
must carry `Authorization: Bearer KEY` or `x-api-key: KEY`, or it gets a 401. `GET /health`, `GET /cluster` and the
chat page need no key; the chat page asks for it once and keeps it in the browser. `--private` keeps the server to this
Mac (`127.0.0.1`), as before (see [SECURITY.md](../SECURITY.md)).

## Several Macs

Every `kapitsa serve` is visible to Kapitsa on the other Macs of its network, with no password: it registers itself
over Bonjour (the service `_kapitsa._tcp`) with the model it serves, whether it holds it whole in memory, its recent
decode speed and this Mac's computer name, chip and memory; `GET /cluster` gives the same as JSON. `--private` keeps a
server to this Mac as before: it listens on `127.0.0.1`, is not visible and sends no request elsewhere. A `--host` of
`127.0.0.1` does the same, so scripts that name it are unchanged, and so does `--allow-switch` without `--api-key`
(any client could load models).

A Mac that would not hold the model it is asked for whole sends its requests to a Mac on the network that does, as if
that Mac were its own server. Whole means in memory: the engine's plan of the model fits this Mac's GPU memory budget
(the check `kapitsa serve` makes before a download) and no routed experts would stream from the SSD (a
Qwen3.8-Flash-Next Q2_0 pack whose experts do not all fit). Such a Mac looks for 1.5 s for a server that runs the same
model whole and answers: the same variant first, else the most precise one, then the shortest round trip, then the
fastest. If there is one, nothing downloads or loads here. Every request under `/v1` (Chat Completions, Messages,
count_tokens, Responses, the model list) is forwarded with its headers, the API key included, and the reply relayed as
it comes back, a stream chunk by chunk; the ready card says where the model runs, and each request's line says where it
went. While that Mac does not answer (asleep, stopped, off the network), requests get a 503 that says so, and the server
keeps looking: they go there again when it is back, or to another Mac that runs the model whole. `--local` runs the
model here anyway. `kapitsa claude` and the other coding-agent commands download nothing for such a model either: the
server they start sends its requests on.

Macs that would each stream Qwen3.8-Flash-Next's experts from the SSD pair on their own, and then hold it in memory
between them as [Flash-Next across Macs](#flash-next-across-macs) does with `--split`. A `kapitsa serve` that would
stream the model looks for another server of the same model and variant that runs it alone and streamed, within a
round trip of 15 ms (a pass of the split crosses the network once there and back). If the Macs that run it that way
hold the model in memory between them (two, or more when two do not: three 24 GB Macs, say), each becomes the stage of
the one that started first: it loads the layers that server gives it and sends its own requests there, as a Mac that
routes them does. That server is the head: once its stages and it hold the model in memory, it places the layers,
loads its own share again (requests get a 503 for the few seconds that takes) and runs the model across the Macs. Of
two servers that started alone (at the same time, or before they could see each other), the later one gives way: it
starts again as the other's stage. A head that has decoded alone measures the split: two replies in a row slower than
80% of its rate alone, and it runs alone again and does not pair with those Macs again. When a stage goes away, the
reply in flight ends with an error and the head runs the model alone again, then pairs with the stages left if they
still hold it; when the head goes away, its stages start again on their own after 30 s. A Mac that streams the model
says how much memory a second Mac needs to pair with it. `--local` and `--private` keep a server out of pairing.

A Mac that starts the model later, while a split runs, joins it as a stage. The head works out a token's pass with it
(each Mac's layers over its memory bandwidth, a network hop more): when that is a tenth faster, the head loads its
share alone again and places the layers over every Mac, and a stage given other layers than it holds starts again to
load them (its old buffers are not freed in place), so the Macs run alone for the seconds that takes. Otherwise the
new Mac stays a spare: it sends its requests to the head, and takes layers when a stage leaves. Measured over office
Wi-Fi with `--split` and both Macs streaming (M3 Pro 48 GB head, M4 Pro 24 GB stage): 11.0 tok/s and a 4K-token prompt
in 108 s, against 18.0 tok/s and 14 s on the M3 Pro alone, which is why Macs pair only when they hold the model in
memory. A pair in memory over Wi-Fi is not measured yet; a long prompt's rows (189 MB for a 2048-token chunk) take
seconds to cross, so its prefill can be slower paired than streamed alone.

Models are matched by name: a catalog model by its name and variant (`qwen3.8-flash-next:4bit`), also when it is served
from a folder named as the store names one (`qwen3.8-flash-next-q2_0` is `qwen3.8-flash-next:q2_0`), and any other
folder or repo by its API id. `NI_CLUSTER_ROUTE=1` routes whenever another server runs the model whole, even one this
Mac could run: two servers on one Mac, for a check.

`kapitsa cluster` (or `kapitsa macs`) lists the Macs running Kapitsa on this network after about 2 seconds of looking:
this Mac first, then each server with its chip and memory, the model it serves, how it runs it (`in memory`, `streams
from SSD`, `proxy` to the Mac it sends its requests to, or `stage` of the Mac whose split it holds layers of), its recent decode speed and the link to it (this Mac,
Wi-Fi, Ethernet or Thunderbolt, from the interface that reaches it) with the median of three round trips. A server that
Bonjour lists but that does not answer is shown as `not answering`. `--json` gives the same as JSON, `--watch` keeps it
up to date until Ctrl+C.

Many office and campus Wi-Fi networks do not pass Bonjour between computers: there `kapitsa cluster` shows no other Mac
although the Macs reach each other. Name the other Macs' servers by address instead: `--peer 10.10.50.111` (port 8000
unless given, `HOST:PORT` otherwise) on `kapitsa serve` or `kapitsa cluster`, `KAPITSA_PEERS=host[:port],...` for any
command (`kapitsa claude` too), or one per line in `~/.cache/kapitsa/peers`. Those servers are read as the Macs Bonjour
finds, so routing and pairing work the same; a server found both ways counts once.

macOS asks once whether the app that runs `kapitsa` (the terminal, or the app that starts the server) may find devices
on the local network. Refused, nothing is found or advertised, and `kapitsa cluster` says so: allow it in System
Settings > Privacy & Security > Local Network, on each Mac. With the macOS firewall on, allow `kapitsa`'s incoming
connections too.

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
([Context](#context): the model's limit, less where memory is tight; `--context N` sets it), logging to
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
sliding-window rings), 8 KB on Bolt and on Super-VL, 6 KB on Lightning, 31 KB on Flash-Next, 208 KB on Nanbeige4.3-33B. Flash-Next's n-gram table
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
state and one conversation's KV cache of up to 32K tokens, with the process's GPU memory within 40% of the RAM before the
M5 (no such cap on it) and no more than the system's available memory less max(4 GB, RAM/12), and reads the others from
the SSD when a token routes them (`src/model/expert_store.h`). Why 40% before the M5: such a store waits on
the CPU at every layer, and on a 48 GB M4 Pro every command buffer took 0.5-3.5 s longer once the process's GPU memory
passed about 45% of the RAM (15-24 GB stores), while 4-12 GB stores ran at full speed.
`kapitsa doctor` counts the smallest such store (2.1 GB) in its weights, lists the rest as streamed, and prints the
store `kapitsa serve` would pick now and why, the same line the engine logs at load (`expert store: ...`: all experts
resident, what the budget leaves, capped by the available memory, or capped by the RAM share; `--json`: `expert_store`
with `bytes`, `reason`, `text`, and the `context` and `slots` it is planned at; the terminal view shows a shorter
line). Without `--context` it plans at the context serve takes here without one (`default_context`), not at doctor's
64K, the same way the engine resolves it (`start_plan`). The available memory changes with what else runs and what the
file cache holds, so the printed size is a snapshot.
`NI_EXPERT_STORE=<GB>` sets the store's size (any size), `all` keeps every expert, `off` loads them as stacked tensors
(the whole pack must then fit); `NI_EXPERT_CAP=<share>` (0 to 1; 0 = none, other values warn and are ignored) sets the RAM share the automatic size keeps the process's GPU memory within for a store short of all the experts (default 0.40 before the M5, none on it); `NI_EXPERT_SETTLE_MS` (default 10000; 0 = one sample) caps how long the automatic size waits, when the system's available memory would cut the store, for memory a process that just exited still holds (it stops once 3 s pass without 0.5 GB more); `NI_EXPERT_READERS` (default 16) sets the SSD reader threads; `NI_PRED_ALL_MIN` (default 64) is the pass length from which a prefill pass has the whole next layer read ahead, and `NI_SCAN_MIN` (default 16) the length from which a pass is read as a sequential sweep (its evictions take the layers it has already passed; `NI_SCAN_VICTIM=0` turns that victim rule off and leaves the store's ARC rule, as before it; a `NI_PRED_ALL_MIN` or `NI_SCAN_MIN` that is not a row count of 0 or more warns and is ignored, default kept). All three keep the store's defaults: raise `NI_PRED_ALL_MIN` to have a long prefill read ahead less and wait less per chunk; set any of them to compare a device's numbers. With `NI_LOG=1` each
request's log says how many of its experts were in memory and how much was read.
On an M5, decoding through such a store keeps the GPU busy across those per-layer waits with a tiny kernel on a second
queue (the GPU heater, `Gpu::keep_warm`), so it does not drop its clock or park between layers: on a 48 GB M5 Pro with
a 29.9 GB store, decode 42.6-44.0 tok/s instead of 25.7-30.5 after an 85-token prompt and 42.4-42.7 instead of 35.5-35.6
after a 4,132-token one, the same tokens and logits bit for bit. GPU power rises about 3x during decode (about 5 W to
17 W); CPU+GPU+DRAM energy a token stays about the same. `NI_GPU_HEAT=0` turns it off,
`NI_GPU_HEAT=1` turns it on on other chips (off by default there: not measured). A store that holds every expert, and
every other model, never runs it.

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

### Flash-Next across Macs

Several Macs can run one Flash-Next together, each holding a run of its layers in memory: two 48 GB Macs hold the Q2_0
pack (38.3 GB) with no experts read from the SSD; the 4-bit pack needs three of them, or one 128 GB Mac (two Macs
that both run `kapitsa serve` with it pair on their own: [Several Macs](#several-macs)). The Mac that
takes the requests, the head, holds the first layers, the embedding, the n-gram table, the final mix, the LM head and
the MTP head; every other Mac runs `kapitsa stage` and holds a run of the layers after them:

```
# each other Mac
kapitsa stage --model ~/models/flash-next-q2
# this Mac: the head
kapitsa serve --model ~/models/flash-next-q2 --split other-mac.local
```

The head asks each stage for its GPU memory budget and its memory bandwidth, and gives each Mac the run of layers that
fits its memory with the fastest pass (`src/cluster/place.h`: a Mac's layers cost their decode bytes over its bandwidth,
one Mac after another; a layer's weights count with two conversations' states on a linear-attention layer and two
32K-token conversations' caches on a full-attention one); the stage loads them then. When the Macs' memory does not hold
every routed expert of a Q2_0 pack, each Mac keeps its layers' dense weights and a store of 3 layers' experts and streams
its other experts from its SSD, as one Mac alone does, and the head says so (how fast that runs is not measured); there
the runs follow the Macs' memory and not their bandwidth alone: as if each held the same multiple of its budget, the
least that fits, so each streams about the same share of its experts, and the Mac with the least memory no longer takes
most of the layers (a 24 GB stage beside a larger head keeps 18 of 48, where bandwidth alone gave it 46). A
stage started with `--layers A-B` keeps those instead, and every stage of the split must then be started that way.
Several stages are listed in the order of their layers: `--split a.local,b.local`.

A pass runs the head's layers, goes to the first stage, on through the others and back to the head
(`src/cluster/pipe.h`): one hand-off per Mac whatever its rows, so a verify of the MTP head's two drafts costs the network
what one token does. The rows travel as the fp32 state the layers read and write (51 KB a row for Flash-Next; 40 KB on
a prefill's way back), so the replies are the same bits as the model whole on one Mac: `tools/split_check` compares a
split model with the whole one pass by pass (prefill chunks, a checkpoint inside one, decode steps, the MTP head's
drafts, verifies with pending rows, a restore, a reset, caches growing), and `tests/test_split.py` sends the same
requests to a whole and a split server.

Every Mac needs the model's `config.json` and its trunk's safetensors files, the same ones (the head checks a digest of
them); a stage needs neither the n-gram table (`ple.safetensors`) nor the MTP head (`mtp.safetensors`). A stage serves
one head at a time and waits for the next when it leaves. The head waits for each pass's rows on the CPU, not the GPU:
a command buffer that waits on the GPU for about 5 s is killed with a GPU timeout, and a network can stall that long.
When a stage stops answering (stopped, asleep or off the network; TCP keepalive notices a silent one in about 11-15 s), the request in flight ends with
an error that says so, later ones at once, and the head needs a restart. macOS may ask, on a stage's first start,
whether `kapitsa` may accept incoming connections (the firewall), and on the head whether the terminal may use the
local network.

What the network adds is a round trip per pass and its rows. Between two processes on one Mac (127.0.0.1; a synthetic
pack of `tools/fn_synth.py --hidden 2560`, Flash-Next's own width, so its rows are the real 51 KB; M4 Pro, `split_check
--repeat 15`, medians) the split added 0.46 ms to a decode step (1.47 -> 1.92 ms), 0.37 ms to a verify of 3 rows (2.44
-> 2.80 ms) and 35.5 ms to a prefill chunk of 2048 rows (167 -> 202 ms: 105 MB out, 84 MB back). Between Macs it is
not measured yet. Over Wi-Fi the rows also take their time on the air: a decode step's 102 KB and a 3-row verify's
307 KB add a few milliseconds beside the network's own round trip (3-10 ms on a quiet network, an estimate), and a
2048-row prefill chunk's 189 MB take seconds, so long prompts read far slower there than over Thunderbolt.

## Benchmarking this Mac

`kapitsa bench` measures Kapitsa on this Mac the way the [published benchmarks](benchmarks.md) measured it, so its
numbers compare with them and with the [community results](../bench/community/RESULTS.md) from other Macs.

```bash
kapitsa bench                     # in a terminal it asks which model; elsewhere the best model for this Mac
kapitsa bench --model 35b:6bit    # a given model and precision
kapitsa bench --model 35b --prefill-chunk 4096  # override prompt tokens per prefill pass
kapitsa bench --quick             # a shorter run: 3 runs per writing test, the 4K prompt only
kapitsa bench --submit            # then add the result to the community results (it asks first)
```

`--prefill-chunk N` passes a positive integer to the benchmark server. Omit it to keep automatic memory planning.
The terminal, Markdown and JSON reports record the requested size; the model may balance actual pass sizes.
Custom-chunk runs are labelled `prefill` or `prefill-quick` and cannot be submitted as standard community results.

It runs the protocol of [bench/harness](../bench/harness/README.md) (`bench.py`), request for request, as a client
of the local API: it starts the model's server (`kapitsa serve` with one conversation slot and a 32K context, `NI_*`
variables left out; downloading the model first if needed), warms it up, waits until other apps leave the GPU alone
for 15 seconds (5 minutes at most, 2 for `--quick`), then measures:

| Measurement | How |
| --- | --- |
| Writing | A 512-token coding answer, at temperature 0.6 (Flint at 1.0, its model card's; the result records which) and greedy, 5 runs each (3 with `--quick`), median; the rate over the middle of the stream |
| Reading | A cold prompt of about 4K and 16K tokens (the 4K one with `--quick`): prompt tokens per second to the first token |
| Follow-up | Three new messages in that conversation, 1 s apart: the median time to the first token, and the rate the new (uncached) part of the prompt was read at (`followup_suffix_tps` in the result's JSON, additive: older results lack it) |
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
| `NI_V8_TREE_FUSE` | `0` | `1` fuses Qwen's eight-row KFlash tree-zone tail and the history-partial reduction. Each lane keeps the tail in registers instead of writing it to global scratch and reading it in another dispatch. The history chunks and summation order stay the same. Wide verifies, batched chains and other model backends keep their attention paths. |
| `NI_DFLASH_CORE4` | Auto | `1` selects the newer four-row 4-bit drafter core, `0` the earlier core. Auto uses the newer core on Apple10 and the 32-core M1 Max. |
| `NI_MEM_BUDGET` | | The GPU memory budget in GB, as `--mem-budget`. |
| `NI_CLUSTER_ROUTE` | | `1`: `kapitsa serve` sends its requests to another server that runs the model whole even when this Mac could run it ([Several Macs](#several-macs)); for a check with two servers on one Mac. |
| `NI_CLUSTER_PAIR` | | `1`: `kapitsa serve` pairs for any Flash-Next folder, even one this Mac holds whole, and counts it as streamed until it is split; a stage whose head went away runs alone after 5 s, not 30, and a head pairs again 3 s after a stage was lost, not 60, and gives a stage that joins its split layers whenever the Macs hold the model with it, not only when a token would be a tenth faster ([Several Macs](#several-macs)); for `tests/test_pair.py`, several servers on one Mac. |
| `NI_VPF` | on for a dense target whose drafter drafts a 16-row block and that is big enough for the pass to pay (weight scale `hidden*intermediate*layers` >= 3e9, which the 27B meets and the 4B does not; the middle is unmeasured, so it stays off) | A cached turn's short new suffix (one chunk of at most `NI_VPF_MAX` rows over a cached prefix of at least 1K rows, no grammar mask, image rows or MTP speculation) runs as verify passes - one pass per up to 32 rows where the target's verify holds them (`NI_VPF_ROWS`), 16 or 8 where it does not - instead of the prefill path's small-T pass, whose GEMMs stream the weights at 53-115 GB/s where a verify pass reaches 210 GB/s. A cached follow-up's time to first token drops 16-27% on the 27B at a 4K to 16K prefix. `1` also takes the path on any target the default leaves out (a MoE model, whose verify reads the union of its rows' experts; a target whose tables hold 8 rows; a small dense model, where the pass costs more than the narrow prefill it replaces - the 4B pays ~30-40 ms for it), `0` restores the prefill path everywhere. |
| `NI_VPF_MAX` | `64` | The longest cached suffix `NI_VPF` takes, in rows (clamped to 2047, the drafter's prompt-capture window). A pass costs one weights read whatever the suffix, and the prefill path's is a 64-row tile, so the crossover is where the prefill path stops staging its GEMMs: at a 16K prefix 47 rows run 399 ms as passes against 447 ms on the staged prefill path, 55 rows 378 against 484, while 73 rows prefill in 373 ms against 433 ms as passes and 124 rows in 442 against 665 (PR #366). `64`, the short-prefill boundary, caps the default just below that crossover. |
| `NI_VPF_SPLIT_MIN` | `25` | The shortest suffix that still takes `NI_VPF` passes when the turn boundary falls *inside* it. A boundary inside the suffix needs the recurrence state folded up to that row for the checkpoint; the plain path saves that checkpoint from its chunk's snapshot for free, while the passes pay a fold pass (~70 ms at a 16K prefix) for it, and the plain path's own cost steps up as it gains a small-T pass. Measured at a 16K prefix on the 27B, two interleaved legs a setting: 21 rows 209/209 ms on the prefill path against 169/169 as passes, 24 rows 214/214 against 237/236, 25 rows 266/266 against 237/235, 26 rows 266/268 against 237/238, 38 rows 339/339 against 290/290. The crossover is between 24 and 25 rows, so a shorter suffix holding a boundary takes the prefill path, exactly the request it took before the passes existed. Rows alone do not decide it - the fold splits the suffix into two pass-quantised pieces - and 30 rows is the one measured size above the crossover where the passes lose (280/280 against 269/269). `0` takes the passes for every suffix `NI_VPF_MAX` allows. |
| `NI_VPF_WIDE` | on | `0` keeps `NI_VPF` passes at 8 rows on a target that verifies wide: an experiment knob, not a measured configuration. |
| `NI_VPF_ROWS` | the target's measured pass width: `32` on a dense target at the `NI_VPF` weight floor (above), `8` elsewhere and whenever `NI_VPF=0` | The widest whole-tile chain pass (`8`, `16`, `32`) a cached suffix runs. A pass streams the whole trunk's weights once, so the pass count is what a follow-up's time to first token is made of: a 96-row suffix is three 32-row passes instead of six 16-row ones. The model sizes its verify buffers, its per-sequence verify tape and the plan's reservation for this at load (`Qwen35::vpf_pass_rows`), so the knob is read then, not per request; a remainder under a wide pass still runs narrower (a 24-row tail of 32-row passes runs as one 24-row pass, which the 32-row capacity holds). `8` is the narrow pass every target runs (and what a MoE model gets: its verify reads the union of its rows' experts). |
| `NI_SHORTPF_ATTN_MAX` | `128` | Attention's row limit on the short-prefill path: a chunk up to this many rows runs `attn_prefill11g` (the keys split over threadgroups and folded by `attn11_reduce`), a longer one `attn_prefill11r`. The two agree bit for bit on `bench_attn`'s checksums and error, and 11g is the faster one while the chunk is small (12.7% at 141 rows, 11.3% at 162, equal from 192 rows up), so `192` was measured as a setting: on the 27B it was 2-4% at 129..192 rows, measured before the 27B had a `qattn_t` form (an M5 now runs 141 rows in 4.16 ms with `qattn_t_g6d2s8` against 9.74 ms for 11g, so there the setting would be slower). It is not the default because this branch runs before the `qattn_t` one, so on an M5 a model whose heads fit `qattn_t` (4B, 35B and, since `qattn_t_g6d2s8`, the 27B) would take 11g in that window instead of `qattn_t`, which reassociates differently and was not gated for those models. |
| `NI_SHORTPF_EXACT` | on | A short prefill pass (at most 128 rows: a cached turn's new message, the reply the server prefills while idle, a draft read ahead) gives each row the same bits whatever the pass's length, so reading a prompt's tail in several such passes gives the logits one pass gives (a follow-up after the idle history prefill replies as with `NI_HISTORY_PREFILL=0`). The short path's choices that followed the row count are fixed: each projection group keeps one GEMM family (`qmms` over up to 128 rows in one dispatch, the qmr tiles for the 27B's large groups; `NI_SHORTPF_EXACT_GEMM=qmms` or `tile` forces one), the 8-bit router `qmms`, the routed experts `qmms_moe3` (`qmms_moe2`'s FMA arithmetic for every expert: an MMA that reproduces it from 3 rows), attention's keys split at fixed tiles (`NI_SHORTPF_EXACT_SEG`, 32 tiles of 32 keys) through `attn_prefill11gs` even for one split. Passes above 128 rows run the large-prefill kernels, whose sums are ordered otherwise, so a tail read across that bound can still part at a near-tie. `0` restores the row-count choices. Cost on the 35B-A3B (MacBook Pro M4 Pro): a pass of 1-16 rows after a 16K prefix the same, 32-64 rows +4-7%, 116-128 rows +5.5%; cold prefill and decode unchanged (their passes are not short). |
| `NI_SP_QMT` | `1` | `0` restores the earlier preference. A chunk of 64 to 128 rows whose projections the tensor-op GEMM covers - 4-bit group 64, `K` a multiple of 64, a dense model - runs on the 64-row-tile `qmt` / `qmtg` kernels rather than the short-prefill GEMMs, which are built for chunks of at most 64 rows: one weights read through register tiles, cheaper than the tile kernels only while that read is one tile's. At a 16K cached prefix it makes a 73-row chunk 14% and a 124-row one 33% faster, and its output gate (one 27B and one 4B, `logit_dump` at a 128-row chunk, `NI_SP_QMT=0` against the default) passes both models' floor files on all three decode paths with no top-1 flip and a mean KL of 3.4e-07 (27B) and 6.3e-06 (4B, PR #366). MoE expert GEMMs and every MoE model keep the short-prefill tiles whatever this says. |
| `NI_ANE` | on where supported | `0` keeps prefill on the GPU, without the Neural Engine. |
| `NI_ANE_MMA` | on for the Qwen 4B and 27B | M5 with macOS 26, 4-bit group-64 Qwen dense models: the Neural Engine split's default share for the GPU's FFN on the matrix units (27B: 24 of 34 units, 4B: 13 of 18). Measured on the 20-core M5 Pro only. Other models (the 9B) keep the older default share until measured. `1` takes it on any such model, `0` keeps the older one. This moves the share only: whenever the GPU keeps channels (any share below all units, a calibration, `NI_ANE_UNITS`), they run on the tensor-ops kernel, so the 9B and `0` are not fully pre-#318. |
| `NI_ANE_SPLIT_WAIT_MS` | `60000` | A split chunk whose Neural Engine work has not finished within this many ms gives up the split: the log says the bound ran out and that the chunk reruns on the GPU, and the rest of the process stays on the GPU. A job that started and never reported is retired by its own 2 s watchdog first; this bound is the backstop for a chunk stuck before that. `tests/test_ane_split_abort.py` runs it at 250 ms with an injected fault. |
| `NI_ANE_KEEP_S` | `300` | Seconds after the last prefill chunk (or model load) during which a background evaluation keeps the Neural Engine awake, so a short prompt after a pause skips its 70–120 ms wake. `0` turns it off. |
| `NI_ANE_KEEP_MS` | `2000` | Period of that keepalive evaluation in milliseconds (floor 100). |
| `NI_DFLASH` | `auto` | As `--dflash` when the option is absent: `auto`, `none` or a folder. |
| `NI_DFLASH_TREE` | on | KFlash on the Qwen draft models: the draft model's guesses are checked as a best-first tree per pass (several continuations at once) instead of one chain; `0` checks a chain (overrides `NI_KFLASH` on these models). Greedy replies are plain decoding's up to near-ties; sampled ones follow the same distribution. The tree's shape follows the draft model's probabilities as calibrated against this Mac's own acceptances, and its size this Mac's measured cost per verified rows: 8 rows per pass (16 or 32 where `NI_KFLASH_ROWS` names them, on the 27B and the 4B) (whichever the drafter's confidence makes pay on this Mac; the costs are measured in the first passes and kept per model and chip), 1 to 8 rows per pass on the 35B (its pass routes experts for those rows alone). |
| `NI_KFLASH_ROWS` | `8` | The tree sizes (verified rows) KFlash may pick on the dense Qwen models (27B, 4B), as a list of 8, 16 and 32. `8` keeps every pass at 8 rows (the cost this release is tuned for); `8,16,32` lets the budget measure 16- and 32-row verifies on this Mac and pick per cycle, where its cost table says they pay; alone, the 27B's measured tok/s gain is within noise (docs/kflash.md, "Eight rows"), and it is the 16-row drafting block that fills the wider rows with something worth verifying. That block is the default on a one-slot engine and sets the verify to 16 rows whatever this says (see `NI_DFLASH_BLOCK`), 32 included: a 16-row block drafts a 15-guess lattice, so a 32-row verify would check nothing the 16 does at twice the rows. The 35B-A3B and the Nemotron drafters always verify 1 to 8 rows. |
| `NI_DFLASH_BLOCK` | `auto` | The DFlash2 drafter's block on the dense Qwen models: `8` (7 guesses, the tile the verify is tuned for), `16` (an anchor and 15 guesses, drafted as two 8-row tiles from one weight read by the pair kernel, verified in 16-row tree passes), or `auto` (16 on a one-slot engine that takes the 16-row paths, 8 on a server with more slots: a 16-row block has no batched cycle, so requests decoding together would take turns instead of sharing a pass). `16` needs KFlash trees and a target with the wide verify, drafts on the matrix units only (M5-class GPU, macOS 26, 4-bit group-64 projections, not the v1 drafter), which is checked when the model loads, and pins the verify at 16 rows. It measured 76.07 tok/s greedy on the 27B's harness corpus with `8` against 87.59 on the default (+15%, ten interleaved runs each), and 71.70 against 79.77 at temperature 0.6 (+11%) (docs/kflash.md, "Sixteen-row drafting blocks"). Read while a drafter loads, once per drafter: a server that switches models decides each one's block again, and the engine's memory plan reserves the rows it will verify. `auto` may be written out; it is the default. |
| `NI_DFLASH_CAP` | off | The verify-depth cap on the Qwen DFlash2 drafters (the F74 verify-cost decomposition), opt-in: `n` proposals verified per cycle (1 to block − 1, so 1-7 on the 8-row block), keeping chains instead of the KFlash trees (the trees need the full-depth verify). Only a genuine cap engages: a value at or above the full depth of the block that runs leaves it at full depth with the trees — though any value below 15 keeps the 8-row block (trees at depth 7) and only at or above 15 is it a no-op outright — and below the full depth the cap is the same axis as `NI_DFLASH_VERIFY=<n>` — and decides even where the trees are explicitly asked (`NI_KFLASH=1`, `NI_DFLASH_TREE=1`), because a capped verify cannot tree. The measured basis: two clean same-policy mode-3 chain anchors at depths 2/3, ids-identical on the F74 legs, read cheaper per token than any deeper measured arm; the ship comparison itself — shipping tree-7 against the capped chain-2 — buys 1.46× per token and 2.01× per cycle in the shipping build, measured on one pinned binary (ledger rulings v2.39-197; quiet-window-confirmed, §8w canonical 1.44–1.46×/token within-window pinned, arm-qualified; the earlier 1.56× carrier figure is historical only, it crosses eras at both ends), and the shipping default stays the full-depth trees until the shipping-binary sweep (4K and 32K, ids match, ABBA ≥ 8) lands (docs/kflash.md). The cap also buys reproducibility itself: across the 2026-10-25 quiet-window legs 03:02:34–03:13:37 local and their repeats (M5 Pro 48 GB, macOS 26.5.2, one pinned binary, seven scored legs, sequential single-tenant), the kernel sums replicate to 0.3–0.8% on both paths, but the wall drifts up to 2.9% on the uncapped default only, while capped arms hold to ±0.5–1.3% — a capped wall is roughly 2–5× tighter than the uncapped default's worst drift (two-level state-dependent noise canon, ledger §8y-2). `NI_DFLASH_VERIFY=<n>` sets the depth directly, overrides this, and leaves the tree knobs to themselves. Read while a drafter loads. |
| `NI_DFLASH_GATE` | off | The downward depth gate (an experiment): `NI_DFLASH_GATE=<p>[:<win>]` caps each chain cycle's verify depth at the first proposal position whose rolling marginal acceptance — among the cycles where the shipping tau kept it, over the last `win` cycles (default 64) — fell below the break-even `p`, after a warmup of `win` cycles. Host-only, downward-only (never above the static depth), off by default; without it the behavior is unchanged. It needs the dynamic depth (an MoE target, `NI_DFLASH_DYN > 0`) **and a chain**: under the default KFlash trees it logs as inert — add `NI_KFLASH=0` or `NI_DFLASH_TREE=0` to run it. `NI_DFLASH_GATE_TRACE=1` prints one line per cycle with the observed pair and the cap. |
| `NI_TURN_MS` | `30` | Requests at once: how long a decoding request keeps the GPU before it gives its turn to a waiting one, in ms; `0` = after every decode step. |
| `NI_BATCH` | on where measured | Requests decoding at once share one GPU pass ([Requests at once](#requests-at-once)): the 27B, the 35B and Bolt; `1`: the 4B, Lightning and Super-VL too (not measured yet); `0`: they take turns instead, each with the tokens it gets alone. |
| `NI_BATCH_KF` | on | Requests decoding at once with the Qwen draft models (2 to 4 of them): KFlash chooses each shared pass's rows, how many in all and which request gets them, from the draft model's calibrated probabilities and this Mac's measured cost of a shared pass (cached with the KFlash budget). `0`: a fixed number per request, as measured before (two requests 4 rows each, three or four 2 each; the 27B's three take turns). |
| `NI_BATCH_KF_FLOOR` | `0` | With `NI_BATCH_KF`: `1` gives every request in a shared pass at least one draft row. |
| `NI_HISTORY_PREFILL` | on | After a reply, while the server is idle, prepare the conversation's next turn so a follow-up starts sooner; `0` turns it off. |
| `NI_HISTORY_WAIT_MS` | 5000 | How long a history prefill waits for a quiet moment (no request started, none running) before it gives up; `0` gives up at the first busy moment. |
| `NI_HISTORY_DISK` | on | A history prefill with no checkpoint to start from reads the conversation's stored state from the SSD prefix cache; `0` turns it off. |
| `NI_HISTORY_ANYSLOT` | on | When another request used the conversation's slot meanwhile (`--slots 1`: the other conversation's), the history prefill takes the slot a new conversation would take, saves that slot's conversation to the SSD first (it gives up if that save did not happen), reads its own conversation back and prefills there; `0` restores the earlier behaviour (give up). Not used with Dory, a LiLiCorr drafter or image requests. |
| `NI_HISTORY_YIELD` | off | `1`: a history prefill stops at a chunk boundary when a request arrives. The rows done stay, the rest is not resumed, so the conversation's next question may read them again. Never yields with a DFlash2 or LiLiCorr drafter. |
| `NI_GRAMMAR_SPEC` | on | [Structured output](#structured-output) with a DFlash2 draft model (the Qwen models): drafts are still verified, each verify row under its own mask; `0` decodes the formatted part one token per pass. |
| `NI_DISK_CACHE` | | As `--disk-cache` when the option is absent; tools that start an engine without `kapitsa serve` keep no SSD prefix cache unless it is set. |
| `NI_DISK_CACHE_IDLE_MS` | `10000` | SSD prefix cache: how long the server is idle after a turn before it writes the conversation, in ms. |
| `NI_DISK_CACHE_RETRY_MS` | `60000` | SSD prefix cache: after a save the cache had no room for, how long before a slot is written again, in ms (`0`: after every quiet spell). Until then the slot counts as unsaved. |
| `NI_DISK_CACHE_MIN` | `1024` | SSD prefix cache: the shallowest state kept, in tokens. |
| `NI_DISK_CACHE_DIR` | `$KAPITSA_HOME/prefix-cache` | SSD prefix cache: the folder of its namespaces. |
| `NI_IMAGE_MAX_MB` | `32` | The largest image taken, in MB. |
| `NI_IMAGE_TIMEOUT_S` | `30` | The time limit for downloading an `http(s)` image, in seconds. |
| `NI_IMAGE_CACHE_MB` | `256` | GPU memory kept for the embeddings of recent images. |
| `NI_MOE_QMR_K` | Auto | Input staging width for native 4-bit group-64 expert prefill: `32` or `64`; `0` means auto. Auto uses paired 32-input staging (`64`) for hidden size 2048 and expert intermediate size 512 on the 32-core M1 Max; other cases retain `32`. |
| `NI_MOE_QTM` | on with the dense QMT path | Qwen3.6 35B-A3B routed expert prefill on Metal 4 tensor ops (Apple10 GPU, macOS 26, 4-bit group-64 experts; `src/kernels/tensor/qmt_moe.metal`): 64-row tiles of the MoE bucket, as the dense QMT prefill; `0` keeps the simdgroup kernels, and a Mac without the tensor units (or with `NI_QMT=0`) runs them anyway. 4K prefill 2,272 -> 1,702 ms and 16K 11,093 -> 8,834 ms on the M5 Pro (macOS 26.0, `d53f02e`, interleaved A/B, same prompt and power mode), 25% and 20% off the time to first token; the switch covers every Apple10 GPU on macOS 26, M5 Max and Ultra included, of which only the M5 Pro was measured. The tile reassociates each output's K reduction, so its logits are not the simdgroup path's: it is gated by `tools/logit_compare.py --reorder` (about 1.1e-3 mean KL, 1.3x the drift the simdgroup kernels already carry against the order before them), not by the model's noise floor. |
| `NI_MOE_QTM_Q20` | on with the dense QMT path | Qwen3.8-Flash-Next's Q2_0 routed expert prefill on Metal 4 tensor ops (Apple10 GPU, macOS 26; `src/kernels/tensor/qmt_moe_q20.metal`): `qmt_moe.metal`'s 64-row tile design on the Q2_0 weight rows, gate/up and down each as a grouped GEMM over the MoE bucket's expert tiles, keeping the qrm kernels' per-element arithmetic — the Q2_0 decode is exact in one half FMA a weight on both paths and the products accumulate in fp32 — so only the tiling reassociates each output's K reduction. `0` keeps the simdgroup kernels (`qrm_gu_q20_m32_n64` / `qrm_dn_q20_m32_n128`), and a Mac without the tensor units (or with `NI_QMT=0`) runs them anyway. 4K prefill 7,456 -> 5,640 ms, 554 -> 733 tok/s (interleaved ABBA, same prompt) on the M5 Pro 48 GB (macOS 26.5.2, `3dcc4fb3`), 24% off the time to first token; decode and verify rows run the same kernels as before, and 16K and 32K prefill are unmeasured on this model. Its logits are not the simdgroup path's: against `NI_MOE_QTM_Q20=0` on the same binary (the env switch is the only difference; `logit_dump`), prefill mean KL 2.9e-3, step/verify mean KL 1.1e-3, p99 0.024, 3 top-1 flips in 192 rows — all near-ties by the gate's own threshold, far flips 0, one flip reproducing the calibration source's own hairline margin — and the mean NLL improves 6.6e-2 nats on the prefill rows; greedy continuations are bitwise identical on all three on/off A/B pairs measured (the gate leg, its rerun and the math-safe leg). The model's order-change gate (`tools/logit_floors/flash-next-q20.m5.json`) was calibrated from the drift the shipped defaults already carry against `NI_QMT=0`, whose own prefill drift is 2.9e-3: this change passes it on the step/verify mean KL, the p99, the median, the NLL and the far flips, and fails on the prefill mean and median KL — bounds taken as 2x the step-path calibration numbers, which the calibration source's own prefill drift also exceeds — plus the top-1 by one near-tie flip past its allowance. It is the default anyway, and on an M5-class GPU it departs from the gate by that much until the floor file is re-measured with this path on in a PR of its own. Loading the tensor library adds no measurable residency: the peak RSS is identical across the two arms (29.5 GB) with the 24 GB expert store. The Metal validation layer cannot validate this path in situ: Flash-Next prefill dispatches the attention kernels before any expert tile and those exceed the doubled cap — `attn_prefill11g` is 20,480 B static and the layer counts it twice against the 32,768 B cap — so the run aborts whatever this switch says; the new tiles themselves are layer-readable (16,384 B static), and the variant coverage was exercised bitwise instead (tail tiles at an 85-token prompt, the stage-off `_o` and arena `_slot` twins bitwise-equal to their staged defaults). |
| `NI_QMT6` | on with the dense QMT path | The 6-bit group-64 **dense** weights' large-prefill GEMMs (M >= 64) on Metal 4 tensor ops (Apple10 GPU, macOS 26; `qmt6_m64_n64` in `src/kernels/tensor/qmt6.metal`): `qmt_m64_n64`'s 64x64x32 double-buffered tile on the f6g64 slab, its weight decode a bitwise mirror of `qrf_load_w`'s half-domain affine — the code enters as the half `0x3400|q`, the scale pre-multiplied by 4096, one half fma a weight — so the dequantized halfs and the products over them are bitwise the simdgroup path's and only each output's K reduction is reassociated: one 32-K `matmul2d` over 4 simdgroups a step instead of four 8-K simdgroup MMAs, the same exchange `qmt_m64_n64` made for the 4-bit weights. `0` keeps `wfmt.metal`'s `qrf_f6g64_m32_n128`, and a Mac without the tensor units (or with `NI_QMT=0`) runs that anyway. The switch takes every 6-bit group-64 dense projection — the Qwen3.5/3.6/3.8 packs' 6-bit weights; the drift dumps and the speed legs are one model, Qwen3.8-Flash-Next q20 with a 24 GB expert store, and the other packs inherit the reassociation claim by argument (the kernel reads no other weights and performs the same half-domain decode), not by measurement on each of them. 4K prefill 690.9 -> 872.7 tok/s, +26.3% mean / +16.4% conservative floor (worst A vs best B) on interleaved ABBA, four legs, one window, same binary pair and env delta, same prompt (ni_run `48d4d213` vs `a76ce376` on the M5 Pro 48 GB, macOS 26.5.2 — the legs predate the #495 rebase, so the relative deltas are the gate numbers, not the absolutes), about 21% off the time to first token; decode is neutral across the arms and greedy continuations are bitwise identical (the 4K ids-md5 `2ba67d51…` holds on all four legs, the 85-token decode dump `ef97e361…` on both gate arms). Its logits are not the simdgroup path's: against the isolated parent reference (`c3b574e2` vs parent `2d68a751`, `tools/logit_dump`, `tools/logit_compare.py --reorder`) all three paths show 0 top-1 flips and top-1 agreement 1.0, and every KL bound of the model's order-change gate (`tools/logit_floors/flash-next-q20.m5.json`) passes — prefill mean KL 4.9e-4 against the 1.8e-3 bound, p99 3.1e-3 against 2.8e-2, median 8.0e-5 against 1.0e-4 — but `max_nll_change` is exceeded by 9.1% (prefill mean NLL +3.809e-2 against 3.49e-2, a bound exactly 2x the calibration source's own 8-record prefill drift): landed as escalated item #510, accepted default-ON on the bitwise token-selection evidence. The acceptance authority is user-set governance, not maintainer adjudication: no explicit user acceptance of this departure was obtained — it ran on the standing user ruling to ship the QMT family default-on without re-asking ('default on, dont ask me again') and the pre-registered decision procedure that executed Branch A (ratify with this recorded departure) when the 01:31 local (20:31 UTC, 2026-10-24) decision deadline passed with no user answer; a late explicit user decline supersedes this entry and is executed as a one-line default-off revert (`NI_QMT6=0`) without further discussion (AGENTS.md, acceptance entry). Superseded by any >=192-record prefill NLL measurement should one ever exist. On an M5-class GPU the path departs from the gate by that much until the floor file is re-measured with this path on, in a PR of its own. The Metal validation layer can dispatch the tile — its 16,384 B of static threadgroup memory reads as exactly 32,768 B under the layer's double count, at the cap — and 196 small-shape dispatches covering the M/N tail classes ran layer-clean and bitwise-equal to an exact CPU reference (#514); a full prefill in situ under the layer still aborts at `attn_prefill11m` (20,496 B) whatever this switch says, as with the expert tiles. |
| `NI_QMT_I8` | off | The 4-bit group-64 dense large-prefill GEMMs (M >= 64) as W4A8 on the M5's int8 matrix units (Apple10 GPU, macOS 26; `src/kernels/tensor/qmti8.metal`): `qx8_g64` quantizes X per row per group of 64 (scale absmax / 127, round to nearest), then `qmti8s_m64_n128` multiplies the int8 X by the 4-bit weights with a fresh int32 product per 64-wide K step, rescaled into fp32. `bench_qmm`, M 4096 with the quantize pass timed: 27.9 / 25.2 TFLOP/s against `qmt_m64_n64`'s 23.4 / 19.8 (N 10240 K 2560, N 2560 K 6144). Lossy (activation rounding): on Flash-Next's four 9K–24.6K windows against off, step mean KL 7.0e-3, p99 7.6e-2, 2 top-1 flips in 96 rows, NLL +0.041 nats (measured on a pack with a sparse PLE table; on `flash-next-q20-full` against Kapitsa's reference at chunk 2048 the same rows move to mean KL 7.0e-3, p99 7.8e-2, 4 top-1 flips and |NLL Δ| 0.036 nats, against Splash's 0.72 / 10.0 / 13 / 0.45, and 192 short rows give 1.9e-3 / 2.0e-2 / 3 flips / 0.0047 against Splash's 0.069 / 1.10 / 8 / 0.016). `1` turns it on. Before using it on a GPU other than the M5 Pro, run `coop_layout`: it checks the int32 product layout `qmti8s` reads. Static threadgroup memory 29,184 B (the validation layer cannot dispatch it; its `qmti8s_m32_n64` instantiation, 14,592 B, runs the same code under the layer). |
| `NI_HC_FUSE` | on (`0` off) | The hyper-connection prefill's mix folded into the up projection's tensor-ops GEMM (`qmt6_mix_m64_n64` in `src/kernels/tensor/qmt6.metal`): one dispatch in place of `enc_qmm(up)` + `hc_mix_h`, the tile running `qmt6_m64_n64`'s 64x64x32 k-loop once per stream (weight rows `s*D + c`) and draining each run's fp32 accumulator into a second `matmul2d` destination tensor as `sigmoid(m) * Xn` in `hc_mix_impl`'s ascending-s order, then writing `u = half(sum / S)`. The [T, S*D] fp32 mix buffer is never materialized — on Flash-Next q20 at 16K prefill that is 84 MB written plus 84 MB read saved per hyper-connection per chunk, and one dispatch fewer per hyper-connection per chunk. The weight decode, tile shape, k-loop and `matmul2d` call sequence are `qmt6_m64_n64`'s unchanged and the sum's order, divisor and casts are `hc_mix_h`'s, so the outputs are bitwise identical to the replaced path's: Flash-Next Q2 on the M5 Pro, 25 Oct 2026, `logit_dump` at chunk 2048 and 4096, every row bitwise identical (its 16,384 B of static threadgroup memory is `qmt6_m64_n64`'s exactly, at the layer-doubling cap). Engages only where the up GEMM it replaces would itself run as `qmt6_m64_n64` — `NI_QMM` not 0, the qmt6 tensor-op path live (`NI_QMT6=0`, or a Mac without the dense QMT path, turns the fold off too), at least 64 rows in the pass, and not the short-prefill `enc_proj` detour; against a qrf/qmr baseline the fold stays off, because the fused tile is `qmt6_m64_n64`'s derivative and against those it would be a reassociation rather than the bitwise claim. Also needs the same 6-bit group-64 up/down weights in one format (the Flash-Next q20 pack), `hc_rank % 64 == 0` and `D % 64 == 0`; the decode mix (`enc_gr_rows`) and the MTP short-row mix (`mtp4_fuse`, R ≤ 8) keep their own paths, while MTP long-draft rows (R > 8) go through `enc_gr_prefill` and fold under the same conditions as any prefill pass. Not prewarmed: the pipeline compiles on first use (qwen4_exp.cpp's `pipe()` triggers the tensor-library compile on the first qmt lookup, as qwen35.cpp's does). |
| `NI_MOE_STAGE` | on | The tensor-op routed expert tiles (`NI_MOE_QTM`, `NI_MOE_QTM_Q20`, `NI_FLINT_MOE_QTM`) write their outputs as whole rows staged through threadgroup memory instead of one store per accumulator element, and a simdgroup whose rows are all past its tile's live rows skips its products. Same values, bitwise; `0` runs the `qmtm_*_o` kernels, main's kernels exactly (element stores, no skip). |
| `NI_QATTN_G8` | on, on an M5-class GPU with macOS 26 | Qwen3.6 35B-A3B prefill attention on the same Metal 4 tensor ops (`src/kernels/tensor/qattn_t.metal`, head dim 256, 16 query heads over 2 KV heads): a 16-row tile holds two queries and all 8 query heads of a KV head, 8 queries per 128-thread group (by default now in the head-dim-split form, `NI_QATTN_D2` below); `0` keeps `attn_prefill11r`, and `NI_ATTN` overrides both. On the M5 Pro 48 GB (macOS 26.5.2), through `bench/harness` on `487ea80` with the experts already on the tensor units, three interleaved sessions: 16K prefill 1,801 -> 1,999 tok/s (+11%) and 4K flat (2,424 -> 2,476); on `330b216`, before the experts moved, the same A/B read 4K prefill 1,693 -> 1,636 ms and 16K 8,819 -> 7,929 ms (median of three), so the 16K rate goes from 0.75 of the 4K rate to 0.81. It keeps 11r's products (half x half is exact in fp32) and only reassociates the K reductions, but its decode rows move 5.6e-4 mean KL, which is 1.10x the 35B's noise-floor gate (`tools/logit_compare.py` FAIL, `--reorder` PASS). It is the default anyway, by the maintainer's call recorded in `AGENTS.md`, so on an M5-class GPU it departs from the gate by that much until `tools/logit_floors/qwen3.6-35b-a3b-4bit.json` (an M4 Pro measurement of the kernels before this) is re-measured here in a PR of its own. |
| `NI_QATTN_D2` | on, where `NI_QATTN_G8` is | The 35B-A3B's prefill attention on `qattn_t_g8d2s8`: the head dim split over a pair of simdgroups, each holding half of Q and of the output accumulators, so a lane keeps 64 accumulator floats instead of 128 (`qattn_t_g8s4k16` is bound by its registers). The pair adds its two partial score tiles through threadgroup memory, so both run the same softmax. `bench_attn` on the M5 Pro, a 16K prompt's 2nd and 4th chunks (T 4224 against 8448 keys, T 3328 against 16000): 1.82x and 1.76x, error against the fp64 reference the same. Only the score sum over the head dim is reassociated: `tools/logit_compare.py --reorder` passes; the floor gate does not (mean KL 6.5e-4 against 5.1e-4), and the default departs from it by that much, accepted on the same terms as `NI_QATTN_G8` (AGENTS.md). Its static threadgroup memory is 24,576 B, so `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1` cannot dispatch it (the layer counts it twice against the 32,768 B cap). `0` runs `qattn_t_g8s4k16`, bit-identical to before. |
| `NI_QATTN_G6` | on, on an M5-class GPU with macOS 26 | Qwen3.8 27B prefill attention (head dim 256, 24 query heads over 4 KV heads) on the same tensor ops: `qattn_t_g6d2s8`, the `NI_QATTN_D2` kernel with its 16-row tiles taken over (token, head) rows, so a tile can span parts of three tokens, each row with its own causal limit; 64 rows per threadgroup. `bench_attn` on the M5 Pro: a 141 / 512-row suffix after a 16K prefix 4.16 / 12.76 ms against `attn_prefill11r`'s 10.93 / 30.77; T 2048 against 4096 / 16384 keys 8.85 / 44.23 against 22.21 / 111.64 ms. Passes the 27B's floor gate (`tools/logit_floors/qwen3.8-27b-4bit.json`; batch mean KL 1.4e-4, no top-1 flips). Static threadgroup memory 24,576 B, as `NI_QATTN_D2` (the validation layer cannot dispatch it). `0` keeps `attn_prefill11r`, bit-identical to before. |
| `NI_QATTN_G4D2` | on, on an M5-class GPU with macOS 26 | Qwen3.5 4B and 9B prefill attention (head dim 256, 16 query heads over 4 KV heads) on `qattn_t_g4d2s8`: the `NI_QATTN_G6` kernel with 4 heads per KV head, 64 (token, head) rows per threadgroup, in place of `qattn_t_s8k16`. `bench_attn` on the M5 Max (40 cores, macOS 26.4), the 4B's heads, T 2048 against 4096 / 8192 / 16384 keys: 3.09 / 7.34 / 15.95 ms against 5.15 / 11.48 / 24.21 (1.67x / 1.56x / 1.52x), error against the fp64 reference the same (3.29e-4 against 3.30e-4 at 16K). Through `bench/harness`, three interleaved A/B pairs (prefill phases only): the 4B 4K 5,561 -> 5,749 tok/s (+3.4%), 16K 4,652 -> 5,203 (+11.9%); the 9B 4K 2,727 -> 2,761 (+1.2%), 16K 2,475 -> 2,589 (+4.6%). Only the score sum over the head dim is reassociated: the 4B passes its floor gate (`tools/logit_floors/qwen3.5-4b-4bit.json`; decode paths' mean KL 1.3e-4, no top-1 flips), though that floor is an M4 Pro measurement of kernels before the M5's tensor paths, and the 4B's floor on an M5-class GPU is still to be re-measured with `tools/logit_floor.py` in a PR of its own; the 9B has no floor file: its prefill rows move 1.4e-7 mean KL and its decode paths 2.6-5.8e-7, no top-1 flips (`logit_dump` at 2048-row chunks, so every window's prefill runs this kernel: `NI_PROFILE` on a 2048-row 9B prefill shows `qattn_t_g4d2s8` at 1.44 ms a call in place of `qattn_t_s8k16`'s 2.47; the same binary with the switch off and on gives the same figures). The 4B's larger drift is its Neural Engine split's, whose W8A8 share re-quantizes the attention's output each chunk: with `NI_ANE=0` the same switch moves the 4B's decode paths 2.5-7.9e-7, as the 9B's, which cannot take the split (its hidden size is not whole 2560-channel segments). Static threadgroup memory 24,576 B, as `NI_QATTN_D2` (the validation layer cannot dispatch it). `0` keeps `qattn_t_s8k16`, bit-identical to before. |
| `NI_QSA_FUSE` | off | Qwen3.8-Flash-Next indexer selection as one dispatch instead of four: `qsa_all` runs `qsa_prep` + `qsa_block` + `qsa_score` + `qsa_select` per row in a single 1024-thread threadgroup — the index queries as eight head slices of 128 threads (`tg_sum` is positional, so a slice reduces exactly as its prep threadgroup), the block keys not yet in the block cache re-derived from the raw indexer keys (`float(half(x))`, bitwise the `ik` values the four-dispatch chain reads), the scores and the radix select unchanged. Decode and verify sparse rows only (`qsa_fusable`: R ≤ 16, indexer heads ≤ 8, head dim 128, compress ratio a power of two in [2, 16]); intended to be bitwise identical by construction, and `fn_check kernels` compares it with the four dispatches byte for byte — that comparison has not run. Scoring is one threadgroup per row here, against `qsa_score`'s `ceil(nb / 256)`, so the GPU A/B has to cover a long context (at least 64K) before any default is considered. Static threadgroup memory 14,472 B (the validation layer can dispatch it). Not verified on a GPU. |
| `NI_QSA_ROWFUSE` | off | Qwen3.8-Flash-Next indexer scores four rows per threadgroup (`qsa_score_rows`) instead of one (`qsa_score`): the four rows' query matrices are staged in threadgroup memory and each sub-chunk of 16 key blocks is staged once and shared, so at a verify pass of 8 rows two threadgroups cover them and the block bytes are read 4x fewer times. Each score is computed by the same expression on the same values (per-head dot, same max, same order, same scale), so the outputs are expected bit-identical; `fn_check kernels` compares the two kernels score for score (`test_qsa_rows`, R 3/4/8, stray scores gated at 0) and its `test_qsa_timing` prices the pair at nb 1024 and 8192 over R 1/2/4/8 — neither comparison has run. Sparse decode and verify rows only: R in [2, 8], indexer heads <= 4, HD a multiple of 4 with HD/4 <= 32, nb > 8. Static threadgroup memory 16,384 B, which the validation layer counts twice, so it sits exactly at the 32,768 B cap. Scoring is one threadgroup per four rows against `qsa_score`'s `ceil(nb / 256)` per row, so the GPU A/B has to reach a long context before any default is considered. Not verified on a GPU. |
| `NI_QSA_TATTN` | off | Qwen3.8-Flash-Next long-context prefill (past 8192 keys, the per-row lists path): `1` runs the sparse attention on the matrix units (`qattn_t_sparse_g12d2` in `src/kernels/tensor/qsattn_t.metal`) instead of `attn_sparse_rows`. Off by default since [#585](https://github.com/nace-ai/kapitsa/issues/585): on a 64K Flash-Next prompt it moves the first decode row's top token from ` of` to `<|im_end|>`, and the reply ends at once and loops. One threadgroup per row and KV head: two simdgroups split the head dimension, the 12 query heads are the fragment rows, and 32 listed keys go per step. Static threadgroup memory 4,096 B. Needs 12 query heads per KV head, head dimension 256, an M5-class GPU and macOS 26. Otherwise the run keeps `attn_sparse_rows` and logs that it did. With it on, the Q2_0 pack also takes 4096-row prefill chunks for a single-stream prefill of 8192+ tokens ([#571](https://github.com/nace-ai/kapitsa/pull/571); 32K +0.7 to +13.7% against 2048 with TATTN on, −1.2 to −7.9% with it off, so with it off the pack keeps 2048). 32K prefill 763–785 -> 843–888 tok/s (+10 to +15%) on the M5 Pro 48 GB (macOS 26.5.2, full-table Q2 pack, chunk 2048, 4 interleaved pairs against `0`), decode unchanged. The Metal validation layer has not run it end to end: a layered run stops first on the dense `attn_prefill11r` (40,960 B threadgroup memory against the layer's 32,768 B). Q and P are cast to half for the matrix-unit products and the softmax is updated per 32-key tile, so the logits are not `attn_sparse_rows`'s. On the full-table Q2 pack, four 9K–24.6K windows (96 step rows, `logit_dump --chunk 2048`) against `0`: mean KL 4.4e-3, p99 6.3e-2, 3 top-1 flips, |NLL Δ| 0.013 nats; the 192 short rows (under 8192 keys) are bitwise. That fails `--reorder` but is within Splash's drift from the same reference on every metric (mean KL 0.72, p99 10.0, 13 flips, |NLL Δ| 0.45). |
| `NI_HC_PRECOMB` | on for the Q2_0-expert pack (`0` off); off for the 4-bit pack (`1` on) | Qwen3.8-Flash-Next prefill: the MoE's `moe_combine` sum computed inside the next layer's gated residual instead of as its own dispatch — `hc_pre_comb_h`'s row threadgroup first accumulates the expert sum into threadgroup memory (phase A replicates `moe_combine`'s float4 arithmetic verbatim: ascending routed slots, the shared slot last, the same token-major `y` addressing), then runs `hc_pre_h`'s loop injecting that sum from threadgroup memory in place of reading `yb_`, which stays zeroed from the previous gated residual until the next mixer output overwrites it. Engages only where nothing reads `yb_` in between — no armed DFlash2 capture of the layer, no PLE boundary, debug cut or final layer after it — and only for D % 4 == 0 with D ≤ 3968 (one float4 of the sum per thread of the row's 1024); anywhere else the separate `moe_combine` runs as before. One dispatch fewer per folded prefill layer and one [T, D] fp32 read-modify-write round trip saved; bitwise identical to `moe_combine` + `hc_pre_h` (Flash-Next Q2 on the M5 Pro, 25 Oct 2026: `logit_dump` at chunk 2048 and 4096, every row bitwise identical). Static threadgroup memory 16,384 B (at the validation layer's doubled cap exactly, so the layer can dispatch it). The 4-bit pack keeps it off by default: there the shared expert folds into the routed GEMMs, so the held sum reads the tail rows of the routed output instead of a separate shared buffer, and no dump has covered that addressing yet. |
| `NI_NH_BF16` | off | Nemotron Super-VL: `1` keeps its BF16 latent projections in BF16 instead of 8-bit (exact; about 4% slower decoding). |
| `NI_NH_ATTN_Q8` | off | Nemotron Lightning: `1` stores its BF16 attention projections in 8-bit (about 3% faster decoding, slightly less accurate). |
| `NI_NH_HEAD` | `q8` with NVFP4 or FP8 | Dense Nemotron-H models (Nemotron Nano): `bf16` keeps an NVFP4 or FP8 checkpoint's BF16 LM head (the 9B v2's, the 4B FP8's) BF16 instead of 8-bit rows; `q8` also runs a BF16 checkpoint's head as 8-bit rows. |
| `NI_NH_NGRAM` | `4` with BF16 or FP8, `0` with NVFP4 | Dense Nemotron-H models (Nemotron Nano): prompt lookup's drafts per cycle when `--spec` is not given; `0` decodes plainly. |
| `NI_FLINT_NGRAM` | `0` | Flint: prompt lookup's drafts per cycle when `--spec` is not given (up to 7); `0` decodes plainly. |
| `NI_FLINT_MOE_QTM` | on with the dense QMT path | Flint: the routed experts' prefill GEMMs on Metal 4 tensor ops (Apple10 GPU, macOS 26, 4-bit group-64 experts; `src/kernels/tensor/qmt_moe.metal`'s `qmtm_g_m64_n64`): gate, up and down each as a grouped GEMM over the MoE bucket's 64-row expert tiles to fp32 rows, so `flint_swiglu_scaled` keeps its whole-row overflow scaling. `0` keeps `qmm_moe_b4_m16_n64` (16-row simdgroup tiles), which a Mac without the tensor units (or with `NI_QMT=0`) runs anyway. Prefill 1,392 -> 2,262 tok/s at 4,000 tokens and 1,128 -> 1,635 at 16,000 through the harness on an M5 Pro (decode unchanged); the routed experts were 60% of a 4K prefill. The tile reassociates each output's K reduction, and Flint carries that further than the 35B: 1.4e-2 mean KL, p99 0.22 and 3 top-1 flips (all near-ties) in 192 verify rows against `NI_FLINT_MOE_QTM=0` (`logit_dump`, M5 Pro), the size of the drift the shipped dense QMT path already gives Flint (1.7e-2, 1 flip, against `NI_QMT=0`). That passes `tools/logit_compare.py --reorder` (`flint-4bit.m5.json`'s order-change gate) on the prefill, step and verify rows; against the model's M5 noise-floor gate the mean KL and flips pass and the p99 does not (0.212 step, 0.219 verify, over 0.192). |
| `NI_FLINT_QATTN` | on with the dense QMT path | Flint: the prefill attention on the matrix units (`tensor/qattn_t_flint.metal`'s `qattn_flint_s8k32p`, 16 tokens x the KV head's 8 query heads per threadgroup, 32-key tiles, latest blocks first) instead of `flint_attn_prefill`'s 8x8x8 simdgroup MMAs: the same half operands, fp32 scores and statistics, sinks, two key sources and sliding window, only the products' summation order differs. Prefill 2,241 -> 2,691 tok/s at 4K, 1,621 -> 2,457 at 16K and 1,158 -> 2,086 at 32K on an M5 Pro (`compare_engines` ABBA, means of 2 passes; decode unchanged). Against `NI_FLINT_QATTN=0` (`logit_dump`, 192 verify rows): KL mean 2.0e-2, p99 0.48, 4 top-1 flips (all near-ties); that passes `tools/logit_compare.py --reorder` (`flint-4bit.m5.json`'s order-change gate) on the prefill, step and verify rows and fails the model's M5 noise-floor gate on top-1 (0.979) and p99. `0` runs the simdgroup kernel, and a GPU before Apple10 or a macOS before 26 takes it anyway. Nanbeige4.3 (on Flint's backend, 7 query heads per KV head) takes the kernels' `_g7` forms under the same switch. |
| `NI_FLINT_MOE4_ROWS` | `8` | Flint: the most rows of a verify pass (the pending token plus the KFlash / prompt-lookup drafts, up to 8) whose routed experts take the distinct-expert kernels (`moe4_unique`, `moe4_gate_up`, `moe4_down`: each distinct expert read once for the rows that chose it) instead of the per-row `moe_gate_up3_r2` and `moe_down2_w`, which read every expert once per row. Same products in the same order, so bitwise either way; `4` is main's split (the distinct-expert kernels for 2-4 rows) and a kill switch, `8` (the default) also covers 5-8-row passes. A verify pass of 5 / 6 / 7 / 8 rows 22.3 / 26.0 / 29.9 / 33.7 -> 21.0 / 24.5 / 28.2 / 31.9 ms on an M5 Pro at a 1K context (4.22 -> 3.99 ms a token at 8 rows). |
| `NI_FLINT_SPLIT` | on before Apple9 or on an Ultra; off on Apple9 and later non-Ultra Macs | Flint: the decode kernels' merges (`flint_router_topk`, `flint_router_shared`, `flint_attn_d2`) as dispatches of their own (`flint_router_topk_merge`, `flint_router_shared_merge`, `flint_attn_d2_merge`) instead of in the threadgroup that finishes last, where the merge sits behind device-scope fences that macOS 14's Metal does not have: macOS 14 takes the split merges whatever the export, and so does every Ultra (two dies joined by UltraFusion, garbage past a 32-key tile, #421) and every M1 or M2 export before Apple9, where the last threadgroup read stale data behind those fences (#421's M1 Pro; #478's M1 Max answered garbage on its 4-bit export). Apple9 and later non-Ultra Macs keep the fused merges (`0`). `1` forces the split merges; `0` keeps the fused ones on macOS 15 and later, while macOS 14 always splits. The same products are merged in the same order, at one extra dispatch a merge. |
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

## Benchmarking a chosen context window

```sh
kapitsa bench --model 35b --ctx 64k --out bench-64k.json
kapitsa bench --model 35b --ctx 128k --quick --out bench-128k-quick.json
```

`--ctx` (also `--context`) first runs the existing standard benchmark with its original 32K server settings.
It then restarts the server at the selected window and appends a separate context section. `--quick` shortens
only the standard section (three writing runs per temperature and 4K reading); the context section always uses
three repetitions. Without `--ctx`, the standard protocol is unchanged.

The added section measures:

- Uncached time to first token (TTFT), with a fresh prefix for each repetition.
- Cached TTFT from a follow-up to each of the three cold requests.
- Cold input throughput and new-input throughput, defined as input tokens divided by client TTFT. The cached
  rate excludes reused tokens. These effective rates include request overhead and first-token latency; they are
  not isolated GPU prefill measurements, and cached tokens are not counted as newly processed tokens.
- Cached token count and percentage from API usage; unavailable counts leave the derived rates null.
- Writing speed at the model's benchmark temperature (usually 0.6; Flint 1.0) and greedy with long prompts, actual
  prompt token counts, and peak memory.

TTFT, input rates and cache reuse show median and minimum/maximum across three samples; writing uses the existing
middle-window streaming rate. JSON retains the individual samples, validity warnings and context-stage conditions
under `context`. Standard measurements remain under `results`, with their own original protocol and conditions.
The combined report is labelled `context` or `context-quick` and cannot be submitted as a standard community run. Paired
TTFT speedup compares each cold request with its own follow-up, then reports the median and range of those ratios.

Before the context stage, a standalone standard report is saved beside the combined output as
`<output-name>.standard.json` (replacing the `.json` suffix). It remains available if you interrupt the long-context
test. Reports are written to a temporary sibling file, flushed, and renamed into place to avoid partial JSON files.
The context-stage GPU-idle wait is shown separately. Its decoding mode is shown only when it differs from the standard run (and is always saved in JSON). Missing cache counts are flagged; inconsistent counts are
excluded from summaries rather than producing misleading throughput.

Values are token counts or binary suffixes (`64k` = 65536). The minimum is `4k`; `max` is not accepted. The token-count
endpoint fits repeated text from the frozen corpus near the requested window, reserving 2048 tokens for answers
and follow-up turns (3072 for reasoning-only models such as Flint). Windows above the model's advertised limit or available server capacity are rejected. Large
windows take longer and need more memory. A context-stage failure is recorded while retaining the standard results.
Compare the added measurements only with runs at the same context and model settings.

During context prefill, a token progress bar shows completed prompt tokens / total prompt tokens and a percentage.
The counts come from GPU-completed prefill chunks, sampled by the server about once per second, and include reused
cache tokens. The bar moves in chunks; tokenization or slot waits are labelled separately. This is actual work
completed, not an estimate from elapsed time. Telemetry events do not start TTFT or count toward generated tokens.
Older servers that do not send telemetry retain the elapsed-time spinner and report diagnostics as unavailable.

The context stage shows warm-up and GPU-idle status, then progress through nine measured steps: three cold/cached
reading pairs, three writing runs at T=0.6 and three greedy writing runs. Each step shows an elapsed-time spinner
and leaves a completion line with measured latency or speed. These counters describe completed workloads, not
a token-level percentage or an estimated finish time.

The context summary groups setup/cache, latency/throughput and generation. Token counts use comma-separated
integers, TTFT uses three decimals in seconds, generation rates and cache speedup use one decimal, and memory
uses two decimals. Input throughput medians appear beside TTFT; full precision and ranges remain in JSON.

Context writing also records per-request decode cycles, tokens produced per cycle (excluding the first token),
and accepted draft tokens per cycle for supported speculative modes. Accepted draft tokens count committed output
beyond the one target token per cycle. This is a cycle-yield diagnostic, not a proposed-token acceptance percentage;
a tree drafter does not expose a suitable denominator here. The JSON retains each run and median/range summaries.

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
`local.env` in it is kept, other edits are replaced, and `--rollback` leaves the newer harness). `kapitsa-systemone`
and `kapitsa-drex-pointer` beside the binary are replaced from the release too, the server only when it is signed as
Kapitsa's (`ai.nace.kapitsa-systemone`, by the same team); one that fails that check is left as it was, with a warning,
and `--rollback` leaves both. A release that fails a check changes nothing. A server that is
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

`brew services stop kapitsa` if you started the service, then `brew uninstall kapitsa`; or delete the binaries
`install.sh` placed (`kapitsa`, `kapitsa-systemone` and `kapitsa-drex-pointer`, in `/opt/homebrew/bin`,
`~/.local/bin` or your `--prefix`) and `../share/kapitsa` next to that
folder (the license files, the benchmark harness, and `kapitsa.previous` after an update). Deleting `~/.cache/kapitsa` (or `$KAPITSA_HOME`)
removes the models, logs, update state and the SSD prefix cache (`prefix-cache/` alone: the cache).
