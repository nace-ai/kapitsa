# Contributing

Thanks for helping make Kapitsa faster. A few rules keep it fast and correct.

## Build from source

You need a Mac with Apple silicon and macOS 14 or later (Kapitsa is tested on macOS 15 and later), the Xcode Command
Line Tools (`xcode-select --install`) and CMake 3.24 or later.

```bash
git clone https://github.com/nace-ai/kapitsa.git && cd kapitsa
./install.sh                                                               # check, build, put kapitsa on the PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j   # or: everything, tools included
```

In a clone, `./install.sh` checks for Apple silicon (and a terminal not running under Rosetta), macOS 14 or later, the
Command Line Tools (it opens their installer) and CMake (it offers `brew install cmake`; `--yes` doesn't ask). It builds
the `kapitsa` target in Release and installs it to `/opt/homebrew/bin` when that is writable, else to `~/.local/bin`
(`--prefix DIR` picks another folder), with `LICENSE` and `THIRD_PARTY_NOTICES` in `../share/kapitsa` relative to that
folder. Run it again to update; `./install.sh --help` lists its options. By hand, `cmake --build build --target kapitsa`
builds only the `kapitsa` command, into `build/kapitsa`.

The Metal kernels are embedded in the binary as source and compiled by macOS at runtime, so neither Xcode nor the
offline `metal` compiler is needed. The GPU pipelines are created when a model loads; `NI_PREWARM=0` creates them on
first use instead.

## Tests

These run on the CPU. The tools (`build/api_test`, `build/parser_test`, `build/tokenizer_test`, `build/vit_check`,
`build/bench_test`) come with the full build.

| Command | What it checks | Needs |
| --- | --- | --- |
| `python3 tests/test_server.py --cli build/kapitsa` | The command line, offline: model names and aliases, `pull --dry-run`, `list`, `rm`, refusals, early port errors | nothing |
| `python3 tests/test_sidecar.py build/kapitsa` | `kapitsa mtp-sidecar` writes the same bytes as `tools/mtp_sidecar.py` | nothing |
| `python3 tests/test_bench.py build` | `kapitsa bench` against the harness: the same requests as `bench.py` (both run against a scripted server), its rates on the published runs' data, the result file, submitting (dry runs, a stand-in gh), the GPU lock; `build/bench_test self` checks the Mac table and formats | `pip install jsonschema` for the schema check |
| `python3 tests/test_update.py build` | `kapitsa update` and the automatic update against a release server in the test (`KAPITSA_UPDATE_URL`; nothing from GitHub), on temporary installs made by `install.sh`: real signed releases, the checks that refuse bad ones, the switch, the rollback, Homebrew, app and source installs, the background check (throttle, off switches, time limit, `kapitsa bench`), what the requests carry; `build/update_test` checks version order and kinds of install | nothing (the full build, for `build/update_test`) |
| `python3 tests/test_community.py build` | The community results: `tools/community_check.py`'s checks and labels, `tools/community_table.py`, the pull request check against a stand-in API | `pip install jsonschema` |
| `python3 tests/test_logit_compare.py` | `tools/logit_compare.py` on synthetic dumps: its numbers against a direct computation, records matched in any order, the gate's verdicts and refusals, numpy against the standard library; `tools/logit_floor.py`'s gate | nothing |
| `python3 tests/test_reasoning.py` | `bench/reasoning`: the grader's equivalences and near misses, the frozen problem sets, a run against a scripted server (the requests, grading, resuming, the GPU lock on a test file) and the report; with the problems downloaded, `reasoning.py check` and the sets' draw | nothing (`python3 bench/reasoning/reasoning.py fetch` for the last two) |
| `python3 tests/test_structured.py --cli build [MODEL_DIR]` | Structured output, offline: `build/grammar_test` (JSON Schema and tool-call grammars against accepted, open and rejected texts, the schemas refused; with a tokenizer, every mask along valid replies and random walks against brute force, and the mask's time per token) and how the three APIs read `response_format`, `text.format`, `output_format`, `tool_choice` and strict tools (`build/structured_check`) | nothing (the 4B's tokenizer, when downloaded, for the mask checks) |
| `build/prefix_cache_test` | The SSD prefix cache's store: pieces and entries written and read back byte for byte and taken back after a restart, CRCs, torn writes, the quota and its eviction order, the volume filling up, threads writing and reading at once, other models' caches, the namespace's lock | nothing |
| `build/fn_store_test` | The streamed-expert cache (`src/model/expert_store.h`) on a synthetic pack in a temporary folder, a thread playing the GPU: every routed expert in its slot at each handshake and still there at the next, read-ahead (right and wrong), verify windows, prefill sweeps, a hot set kept through a sweep (ARC), direct and widened 16 KiB reads | nothing |
| `build/fn_dflash_test [--gpu]` | Qwen3.8-Flash-Next's DFlash2 plumbing: the target's streams from `hc_count`, the drafts' `target_hidden_reduction` checks; `--gpu`: the `stream_mean` capture kernel against the CPU (GPU lock) | nothing |
| `python3 tools/qwen4_exp_q20_test.py` | The Q2_0 Flash-Next converter on a tiny synthetic GGUF and pack: planes, dequantization, config, index, sparse table | numpy |
| `build/fn_q2_bench check --small`, `build/fn_q2_bench store` | The Q2_0 expert kernels at small shapes against a CPU reference in double (and through the slot table, bit for bit), and through the expert store mid-pass; run them under `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1` and the GPU lock | the GPU |
| `build/api_test [MODEL_DIR...]` | The Anthropic Messages and OpenAI Responses translations: request mapping, exact event sequences, each coding agent's settings; the Chat Completions stream's bytes, keepalive and errors; with a model folder, tool loops through its chat template | a model folder for the template checks (default: the 35B and 27B, if downloaded) |
| `venv/bin/python tests/sdk_check.py build/api_test` | The official `anthropic` and `openai` Python SDKs parse every recorded reply | a venv: `python3 -m venv venv && venv/bin/pip install anthropic openai` |
| `build/parser_test MODEL_DIR` | The streaming parser (reasoning, content, tool calls) on real token ids; with a Flint folder, its GLM-format tool calls too | a Qwen or Flint model folder |
| `build/tokenizer_test` | Flint's pre-tokenizer split against the pieces Hugging Face tokenizers 0.23.3 makes of the same texts (a synthetic tokenizer.json, one token per piece) | nothing |
| `build/iq_layout_test` | GGUF IQ expert blocks (IQ1_S, IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S) repacked into the layout of `src/kernels/wfmt_iq.metal` and decoded as the kernels build their MMA operands, from the Metal source's own grid tables, against `gguf::dequantize`; the type helpers | nothing |
| `build/bench_moe_iq --validate --fmt GU:DN [--topk K --shared SI]` | The IQ MoE kernels (rows, 8-row verify, prefill) at small shapes against a CPU reference in double; without `--validate`, the real shapes and their time per call. Run `--validate` under `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1` and the GPU lock | the GPU |
| `build/flint_compile` | Compiles Flint's kernels and the base library's pipelines it runs (no GPU work); run Flint's kernels themselves under `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1` (`build/logit_dump` on a Flint folder takes the 16-key tiles there) | a Mac with Metal |
| `python3 tests/vision/check_cpu.py build/vit_check MODEL_DIR [--quick]` | The Qwen image processor against Pillow (and Hugging Face's processors when torch and transformers are installed) | a Qwen3.5-4B, Qwen3.6 or Qwen3.8 folder; Pillow and numpy |
| `python3 tests/vision/check_radio_cpu.py MODEL_DIR build/vit_check IMAGE.png...` | Nemotron 3.5 Bolt's image processor against NVIDIA's own | the Bolt folder; torch |

End to end, against a running server (this one runs the 4B on the GPU and downloads it, 3.3 GB, the first time):

```bash
build/kapitsa serve --model 4b --port 8090 &
python3 tests/test_server.py http://127.0.0.1:8090   # --api-key KEY for a server started with one
kill %1
```

The SSD prefix cache across restarts (starts its own servers, takes the GPU lock; a temporary `KAPITSA_HOME`, a 2 GiB
cache, removed at the end): `python3 tests/test_disk_cache.py build/kapitsa ~/.cache/kapitsa/models/qwen3.5-4b-4bit`
(`--ttft` also times the follow-up cold, from the SSD and from memory).

## Layout

- `src/server/`: the command line, the HTTP server, the coding-agent launchers and the chat page.
- `src/api/`: the Anthropic Messages and OpenAI Responses translations, and each coding agent's settings.
- `src/engine/`: chat templates, conversation slots and prefix reuse, the SSD prefix cache (`prefix_cache.*`: the
  store; `engine_disk.cpp`: what a slot saves and restores), the output parser, `kapitsa doctor`.
- `src/model/`: the Qwen3.5-family and Nemotron-H backends and their drafters; `src/kernels/` and `src/metal/`: the
  Metal kernels and runtime.
- `src/ane/`: the Neural Engine bridge; `src/vision/`: image decoding, processors and vision towers.
- `src/hub/`: the model catalog, `--model` resolution and downloads; `src/gguf/`: `kapitsa import`.
- `src/update/`: Kapitsa updating itself (the kind of install, the latest release, the checks, the switch);
  `src/server/update_cmd.cpp`: `kapitsa update` and the daily background check.
- `src/bench/`: `kapitsa bench`: the benchmark harness's protocol, the Mac and its conditions, the result file and its
  submission; `bench/community/`: the community results and their checks.
- `src/loader/`: the safetensors loader; `src/tokenizer/`: byte-level BPE, identical to Hugging Face's tokenizers.
- `tools/`: check, measurement and benchmark programs, each built as `build/<name>`; `bench/`: benchmark prompts,
  quality data and the cross-engine harness (`bench/batch/`: requests at once, their latency, throughput and exactness,
  batched decoding's tokens against single-stream ones, and a stress test); `tests/`: the checks above.

[docs/architecture.md](docs/architecture.md) walks through a request and maps the sources;
[docs/usage.md](docs/usage.md) documents the commands, options and switches.

## Code style

- Keep new code in the style of the code around it: C++20, in the existing formatting. Comments explain why.
- No new dependencies without a strong reason: Kapitsa builds with the Xcode Command Line Tools and CMake alone.
- Run new or changed Metal kernels at small sizes first, with `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`: a kernel
  that reads or writes out of bounds can fault the GPU and take the desktop down with it. The layer counts static
  threadgroup memory twice on macOS 26.x, so kernels above 16,384 B of it abort under the layer whatever the change is
  (AGENTS.md has the measurements); say so in the PR instead.

## Speed changes

- Measure on the GPU only when nothing else runs on it, and interleave the runs of the old and the new build (A B A B
  ...). Report every run, not the best one, with the Mac, the macOS version and the commit.
- A default changes only after an interleaved A/B and the [output check](#output-check): the logits against the old
  build's, with differences allowed only within this Mac's noise floor (near-ties).
- Put the numbers in the pull request, with the change that produced them.

## Output check

Did a change move the model's outputs beyond noise? Dump the logits over the whole vocabulary after the same fixed
prefixes with the old and the new build, compare the dumps, and gate the difference at twice the model's noise floor on
this Mac:

```bash
build-main/logit_dump MODEL_DIR /tmp/ref              # the old build (origin/main built in build-main); GPU lock
build/logit_dump MODEL_DIR /tmp/new --from /tmp/ref   # the new build, on the reference's prefixes
python3 tools/logit_compare.py /tmp/ref /tmp/new      # exit status 1 past the gate
```

- `logit_dump` (its header has the details) prefills 8 windows cut from `bench/corpus.txt` (prefixes of 512 to 3200
  tokens) as the engine prefills a prompt, then feeds the next 24 tokens of each teacher-forced (every row the text's own
  next token, whatever the model would pick) through each decode path: `step` (plain decoding), `verify` (the
  single-stream verify: KFlash's tree verify on Qwen, as DFlash2's cycle runs it) and `batch` (four requests in one
  pass, as the engine batches them); `prefill` holds each window's last prompt row as the prefill samples it. Per path
  it writes `sequences.txt`, `records.txt` and `logits.f32` (about 190 MB on the Qwen models: delete them after):
  Splash's layout, so another engine's dump of the same prefixes compares too.
- `tools/logit_compare.py` matches records by their prefix tokens and reports, in fp64 over the vocabulary, the rows
  that are bitwise the same, the top-1 agreement and every flip with the reference's margin between the two tokens, the
  mean, median, p99 and largest KL of the candidate from the reference, and the change in the next token's negative
  log-likelihood. It passes when the mean and p99 KL are at most twice the floor's, there are at most twice its flips
  (two in 192 rows at least), and every flip is a near-tie: at a reference margin under `near_tie`, the largest margin
  the floor's own differences moved by half or more (twice them can flip it). When the two dumps' `prefill` rows are
  bitwise the same, the change touched only the passes after the prefill, and the decode paths are held to the decode
  floor; otherwise to the whole floor. The bounds are in `tools/logit_floors/<model>.json` (the dump's `meta.json`
  names the model); `--against SOURCE` takes one source of the floor, a flag overrides its own bound. With numpy it
  takes a second per path, without ~20 s.
- The noise floor is how far the logits of the same prefixes move between the ways the engine legitimately computes
  them: alone or decoding together with three others, plain decoding or the verify, DFlash2's chains or KFlash's trees,
  a prefill cut into chunks of 512 rows instead of 2048 (a conversation's cached turns cut it anywhere), the Neural
  Engine split or the GPU alone (dense models: the split takes chunks of 512 rows or more, and the server prefills on
  the GPU until its program is ready). The worst of them per measure is the floor; the decode floor is the worst of
  those that leave the prefill bitwise the same (the first three). `tools/logit_floor.py MODEL_DIR` measures both
  (minutes on the GPU) and writes the JSON with every source's numbers. On the M4 Pro (MacBook Pro 16-inch 2024,
  48 GB), macOS 27.0.1, 0.1.39 (0553f55), the KL of the 192 verify rows (Dory: of plain decoding), mean / p99:

  | Model | Decode passes (alone vs 4 together) | Chunks of 512 vs 2048 | Neural Engine vs GPU | Gate: KL mean / p99, near-tie | Decode gate |
  | --- | --- | --- | --- | --- | --- |
  | Qwen3.6 35B-A3B 4-bit | 2.6e-4 / 3.0e-3 | 2.0e-4 / 6.8e-3 | (no split) | 5.1e-4 / 1.4e-2, 0.060 | 5.1e-4 / 6.0e-3, 0.060 |
  | Qwen3.8 27B 4-bit | 2.0e-7 / 3.3e-6 | 6.6e-5 / 1.5e-3 (GPU alone: 4.6e-8 / 1.2e-6) | 1.9e-4 / 7.0e-3 | 3.8e-4 / 1.4e-2, 0.40 | 4.1e-7 / 6.6e-6, 0 |
  | Qwen3.5 4B 4-bit | 5.1e-7 / 5.3e-6 | 2.1e-4 / 3.7e-3 (GPU alone: 1.0e-6 / 7.0e-6) | 9.3e-4 / 6.6e-3, 1 flip | 1.9e-3 / 1.3e-2, 1.52 | 1.0e-6 / 1.1e-5, 0 |
  | Nemotron-Labs Dory 4B NVFP4 | (one request at a time) | 1.4e-3 / 1.6e-2, 2 flips | (no split) | 2.8e-3 / 3.1e-2, 0.99 | bitwise only |

  The same build twice is bitwise the same on every path of all four. On the dense models the batched rows are bitwise
  the chain verify's (alone against together is KFlash's tree verify against chains; plain decoding is closer still),
  and chunking moves the prefill only where it moves rows between the Neural Engine and the GPU. The 35B's decode
  floor is 500 to 1300 times theirs: batched, its experts run in other groupings than in the tree verify's
  row-independent tiles, and a few rows move far (one by KL 0.034). A near-tie bound of 0 means no row came within
  twice the floor's reach of a flip: any flip fails. Dory's chunking changes its arithmetic (last-loop KV: a chunk's
  rows see each other's same-loop keys), not only its rounding. For scale, on the 4B: the verify's MLP gate/up on one
  fused kernel (`NI_V8_GUSPLIT=0`) passes the decode gate (KL mean 7.2e-7); its codes as 1024 + q (`NI_V8_CM=0`) fail
  it (3.3e-6, one row at 5.4e-4); the same weights re-encoded as 8-bit (`NI_REQUANT=8`: every kernel other) pass the
  whole gate (7.7e-4, one flip at a margin of 0.03).
- An M5 or later takes `tools/logit_floors/<model>.m<N>.json` when there is one (`logit_compare.floor_name`, from the
  dump's `meta.json` Mac; `--floor` still wins): its tensor_ops paths sum in an order no earlier GPU runs, so its floor
  is its own, and `logit_floor.py` writes it there instead of over the M4 Pro's. `flint-4bit.m5.json` (M5 Pro 48 GB,
  macOS 26.5.2, 0.2.7 `9d88378`) is the first. Flint's repeat and 512-row chunks are bitwise on both chips, so its floor
  is the verify against plain decoding alone, a few rows whose routing flips: on 768 rows (`--rows 96`) the medians are
  1.6e-6 and 4.8e-6 against means of 1.9e-3 with the shipped tensor_ops prefill GEMMs and 1.1e-2 with `NI_QMT=0`, a
  swing of 6x (23x on 192 rows, kept in the file's `rows_192`); no source of Flint's moves the prefill. Both paths ship
  on the M5 (the default and its kill switch), so the gate is twice the worst of the two on 768 rows (`--also
  NI_QMT=0`): mean 2.2e-2, p99 0.19, 6 flips in 768 (M4 Pro: 1.0e-2 / 8.4e-2 on 192). The file also carries Flint's
  own `reorder_gate`, which `--reorder` takes instead of the 35B's (exceeded 7-13x by a pure reorder of Flint): twice
  the qmt against `NI_QMT=0` drift of the decode paths, fast and safe math (mean 2.8e-2, median 1.7e-4, p99 0.39), with
  a median bound (3.3e-4) a larger error on every row would cross first (not yet shown with one). It bounds a reorder's
  drift only (a flip under 5.8 logits is not refused by itself): a change gated with it also shows the per-GEMM check
  and the temperature-0 continuations above. `logit_floor.py --rows 96 --also NI_QMT=0 --reorder` measures all of it;
  `--rescore` derives the floors and both gates again from the file's sources (this Mac's file by default).
- Put the comparison's output in the pull request. A change that is not meant to move outputs shows "every row bitwise
  identical" or passes the gate; one that moves them on purpose (a precision, a quantization) fails it and says so, with
  the numbers. A floor changes only with the conditions it measures (re-measured, in a pull request of its own), never
  to let a change pass.
- Raw dumps of `nh_eval --vcheck`, `ni_v8check` (`NI_V8CHECK_DUMP`) and `ni_run --dump-all-logits` compare with
  `tools/logit_compare.py --vocab V`. Other questions have their own checks: `tools/kflash_invariance.py` and
  `moe_tree_check` (a tree's size or shape must not change a row, bit for bit), `bench/batch/match.py` (batched
  against single-stream greedy replies through the server); `batch_check` and `ni_v8check` time their passes while a
  kernel is developed.

## Benchmarks

The [benchmark harness](bench/harness/README.md) runs Kapitsa and other engines through the same client-side benchmark,
one engine at a time; [docs/benchmarks.md](docs/benchmarks.md) has the published results.

## Releasing

A release is a pull request that sets the version (`CMakeLists.txt`, the `KAPITSA_VERSION` fallbacks in `src/`, the
`install.sh` example and `packaging/homebrew/kapitsa.rb`) and, once it is merged, the tag `vX.Y.Z` on the merge commit.
The tag runs [`.github/workflows/release.yml`](.github/workflows/release.yml): it builds the tarball, publishes the
GitHub release with its `.sha256`, and installs and tests the Homebrew formula locally before it reaches users. Then
write the release notes (`gh release edit vX.Y.Z --notes-file FILE`), one line per user-facing change, crediting
outside authors.

### The `TAP_GITHUB_TOKEN` secret (Homebrew tap)

Without this repository secret the workflow cannot push the formula to
[nace-ai/homebrew-tap](https://github.com/nace-ai/homebrew-tap): its job summary shows the formula and someone commits it
there by hand. To make the tap update automatic, an owner of the `nace-ai` organization sets it up once:

1. Create a [fine-grained personal access token](https://github.com/settings/personal-access-tokens/new) (for a shared
   machine user rather than a person, where the organization has one). Resource owner: `nace-ai`. Repository access:
   only `nace-ai/homebrew-tap`. Permission: Repository permissions, Contents, Read and write; nothing else. Pick an
   expiry (90 days to a year) and note the date. If the organization requires approval for tokens, an owner approves it
   under Organization settings, Personal access tokens.
2. Add it to this repository: Settings, Secrets and variables, Actions, New repository secret, name `TAP_GITHUB_TOKEN`,
   the token as the value. Or: `gh secret set TAP_GITHUB_TOKEN -R nace-ai/kapitsa` and paste it.
3. Check it on the next release: the `Homebrew formula` step should end with a `kapitsa X.Y.Z` commit by
   `github-actions[bot]` in the tap instead of the formula in the job summary. If the push fails with a 403, the token
   lacks Contents write on the tap, has expired, or is waiting for organization approval.

When the token expires, create a new one the same way and replace the secret. Never commit the token or paste it into an
issue or pull request.

## Pull requests

- One change per pull request, with its measurements when it touches speed or memory.
- New code follows the [code style](#code-style) above.
- By contributing you agree that your contribution is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).

Report security issues privately: see [SECURITY.md](SECURITY.md).
