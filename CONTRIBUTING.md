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

These run on the CPU. The tools (`build/api_test`, `build/parser_test`, `build/vit_check`, `build/bench_test`) come with
the full build.

| Command | What it checks | Needs |
| --- | --- | --- |
| `python3 tests/test_server.py --cli build/kapitsa` | The command line, offline: model names and aliases, `pull --dry-run`, `list`, `rm`, refusals, early port errors | nothing |
| `python3 tests/test_sidecar.py build/kapitsa` | `kapitsa mtp-sidecar` writes the same bytes as `tools/mtp_sidecar.py` | nothing |
| `python3 tests/test_bench.py build` | `kapitsa bench` against the harness: the same requests as `bench.py` (both run against a scripted server), its rates on the published runs' data, the result file, submitting (dry runs, a stand-in gh), the GPU lock; `build/bench_test self` checks the Mac table and formats | `pip install jsonschema` for the schema check |
| `python3 tests/test_update.py build` | `kapitsa update` and the automatic update against a release server in the test (`KAPITSA_UPDATE_URL`; nothing from GitHub), on temporary installs made by `install.sh`: real signed releases, the checks that refuse bad ones, the switch, the rollback, Homebrew, app and source installs, the background check (throttle, off switches, time limit, `kapitsa bench`), what the requests carry; `build/update_test` checks version order and kinds of install | nothing (the full build, for `build/update_test`) |
| `python3 tests/test_community.py build` | The community results: `tools/community_check.py`'s checks and labels, `tools/community_table.py`, the pull request check against a stand-in API | `pip install jsonschema` |
| `python3 tests/test_logit_compare.py` | `tools/logit_compare.py` on synthetic dumps: its numbers against a direct computation, records matched in any order, the gate's verdicts and refusals, numpy against the standard library | nothing |
| `python3 tests/test_reasoning.py` | `bench/reasoning`: the grader's equivalences and near misses, the frozen problem sets, a run against a scripted server (the requests, grading, resuming, the GPU lock on a test file) and the report; with the problems downloaded, `reasoning.py check` and the sets' draw | nothing (`python3 bench/reasoning/reasoning.py fetch` for the last two) |
| `python3 tests/test_structured.py --cli build [MODEL_DIR]` | Structured output, offline: `build/grammar_test` (JSON Schema and tool-call grammars against accepted, open and rejected texts, the schemas refused; with a tokenizer, every mask along valid replies and random walks against brute force, and the mask's time per token) and how the three APIs read `response_format`, `text.format`, `output_format`, `tool_choice` and strict tools (`build/structured_check`) | nothing (the 4B's tokenizer, when downloaded, for the mask checks) |
| `build/prefix_cache_test` | The SSD prefix cache's store: pieces and entries written and read back byte for byte and taken back after a restart, CRCs, torn writes, the quota and its eviction order, the volume filling up, threads writing and reading at once, other models' caches, the namespace's lock | nothing |
| `build/api_test [MODEL_DIR...]` | The Anthropic Messages and OpenAI Responses translations: request mapping, exact event sequences, each coding agent's settings; the Chat Completions stream's bytes, keepalive and errors; with a model folder, tool loops through its chat template | a model folder for the template checks (default: the 35B and 27B, if downloaded) |
| `venv/bin/python tests/sdk_check.py build/api_test` | The official `anthropic` and `openai` Python SDKs parse every recorded reply | a venv: `python3 -m venv venv && venv/bin/pip install anthropic openai` |
| `build/parser_test MODEL_DIR` | The streaming parser (reasoning, content, tool calls) on real token ids | a Qwen model folder |
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
  that reads or writes out of bounds can fault the GPU and take the desktop down with it.

## Speed changes

- Measure on the GPU only when nothing else runs on it, and interleave the runs of the old and the new build (A B A B
  ...). Report every run, not the best one, with the Mac, the macOS version and the commit.
- A default changes only after an interleaved A/B and a check that the output stays the same: greedy tokens and
  logits against the old path, with differences allowed only at near-ties.
- Put the numbers in the pull request, with the change that produced them.

## Benchmarks

The [benchmark harness](bench/harness/README.md) runs Kapitsa and other engines through the same client-side benchmark,
one engine at a time; [docs/benchmarks.md](docs/benchmarks.md) has the published results.

## Pull requests

- One change per pull request, with its measurements when it touches speed or memory.
- New code follows the [code style](#code-style) above.
- By contributing you agree that your contribution is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).

Report security issues privately: see [SECURITY.md](SECURITY.md).
