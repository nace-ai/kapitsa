# JSON tool names: Nano 9B tokenizer checks

An empty JSON function name previously became a tool call instead of preserving its raw `<TOOLCALL>` text. The production change rejects empty names and names containing only space, LF, CR or tab, using the existing `trim_ws`. Nonblank names remain unmodified, including surrounding whitespace and non-ASCII text. Other whitespace classes are outside this predicate.

Runs: Apple M3 Ultra, 60 GPU cores, 96 GiB; macOS26.4.1 (25E253), Apple clang17, CMake Release `-O3`. Reference `ef1053e0dc172477e70f1685fe7b39c1863c7437`; candidate `2ebf44a48d680212cbbe89a163fbd7b6a51abc32`. The reference's `nh_chat_check` was rebuilt with the same new test source as the candidate; production headers differ only by the one guard. Both own checkouts were explicitly configured. Candidate CLI reports `kapitsa 0.2.31 (2ebf44a)`. Subsequent commits add evidence only.

## Reproduction

The catalog's NVIDIA-Nemotron-Nano-9B-v2-NVFP4 metadata at revision `8556c9164ddb43fe1f4f4ad730593b3c5e3f7328` was used unchanged, without weights. Tokenizer SHA256 `3277c00fe5fb3963b3cb7c07b7f183722d2af4d775a4aea7cfb3684d7cccbc2f`. Its template is in tokenizer_config.json. Download identities and all three metadata hashes are recorded in `nano-tokenizer-manifest.json`; the metadata itself is not committed.

`build/nh_chat_check /Users/bytedance/kapitsa-json-tools-20261107/nano9b-tokenizer` uses the engine's text-tag detection and streams real token IDs into OutputParser. It confirms `/think` flags, text tags and the six-token generation prompt. It does not load model weights or execute a kernel.

| Build | PASS | FAIL | Exit |
| --- | ---: | ---: | ---: |
| Reference plus new tests | 34 | 24 | 1 |
| Candidate | 58 | 0 | 0 |

The 39 added assertions include 24 changed outcomes: three blank names in an object, a one-item list and both mixed-list orders, each alone and followed by a separate valid block. All malformed blocks now remain raw text atomically, including a valid item before the invalid one; the later block still becomes one valid call with its original arguments. Twelve disabled-parser controls and three nonblank-name controls pass on both versions. The 19 existing chat/template tests also pass on both.

## Other checks

- Full candidate Release CLI/library and CPU targets built through `vk build`.
- All 21 `vk cpu-tests` suites pass. Separate supplemental targets and Python checks, vocabulary grammar, Qwen4B/35B parser and structured CLI controls pass. Individual transcripts, commands and exit codes are in `checks/`.
- The first vocabulary-grammar invocation passed a tokenizer file where the tool requires its parent model folder and exited with a load error. Its original command/error remain; `grammar-4b-correct` uses the folder and passes. This command correction changes no source.
- Local and remote `prek run --all-files` pass. No earlier benchmark record is modified.
- The new tests are in the existing manual tokenizer-based `nh_chat_check` tool. Hosted CI does not run these regressions or provide this tokenizer. The evidence above is the executed coverage.

## Not verified

No GPU, real Nano generation, HTTP server, model logit gate or performance measurement ran for this CPU parsing change. No inference speedup or model-token equality is claimed. Qwen parser/structured controls were run on the candidate only, not on the reference; the paired reference/candidate comparison is the full Nano chat check above.

`vk checks` correctly classifies the production header as tier0.5 but treats the test tool's unclassified path as tier1. The only tools change adds CPU checks, so the actual validation scope follows the parser tier; no host execution or kernel code changed. No feature, inference default, format or device gate is introduced.

`nano-base-build.sh`, `nano-candidate-checks.sh`, `nano-controls.sh` and `nano-grammar-correct.sh` retain exact commands. The compiler, OS, complete source commits, binary hashes and tokenizer hashes are saved beside the outcomes. `manifest.json` hashes every evidence file except itself. The reference's deliberate failing regression is retained; it is not counted as a green run.

Raw `.cmd` records and grammar transcripts retain their original trailing whitespace. The full diff whitespace check reports those records; source and authored prose are clean, and the configured `prek` checks pass.
