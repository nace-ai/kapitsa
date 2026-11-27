# Nameless Qwen tool-call regression

CPU verification on Apple M3 Ultra (Mac15,14), 96 GiB, macOS 26.4.1 (25E253), Apple Clang 17.0.0, Release `-O3`.

Reference headers/library: `4d1d744538063a17bc85e88a08c08c8061511c95`.
Candidate full build/CPU source: `d871a692b8b09d3515ea748e40d3edf444026ee7` (`kapitsa 0.2.31 (d871a69)`).
The subsequent commit `2e13d7db` changes only an API regression assertion to check a block's type before looking
up text on the failing reference. The final API test source was compiled independently against each source's
headers/library. Its SHA256 is recorded in `source-identity.json` and `evidence/final-api-source-sha256.txt`.
Production source is identical after the full build; no model or performance result from another change is used.

## Results

- Release CLI and CPU targets, extra targets: built. `build.exit` is 0.
- `vk cpu-tests`: 21 suites PASS, 0 failed (`evidence/cpu.log`); each log is under `evidence/vk/`.
- Additional AGENTS CPU tests and format/profile checks: PASS (individual logs and `build.sh`).
- Remote `prek run --all-files`: PASS. Final test-only assertion change also passed local prek.
- `parser_test`: 125 assertions PASS with each 4B/35B tokenizer, including 65 new blank-name cases/controls.
- `tests/test_structured.py --cli build MODEL_DIR`: PASS with each 4B/35B tokenizer.
- Final API test, 4B: candidate 344 PASS / 2 FAIL; reference 328 PASS / 18 FAIL. All 20 new nameless checks pass on the
  candidate; 16 of them fail on the reference (the other 4 check the stream terminator).
- Final API test, 35B: 346 PASS / 0 FAIL, including 20 new nameless checks. No 35B reference API run.

The 2 remaining 4B API failures are the existing Anthropic reasoning-preservation round-trip assertions, at tokens
320 and 408, with identical diagnostics on reference and candidate. Both logs are retained. `tokens.exit` is 1
because the aggregate includes these failures; it must not be interpreted as an all-pass API suite. All new API
assertions pass and none of the observed API failures is newly introduced by this fix.

## Reproduction and limits

`build.sh` contains the configure, build, CPU tests, extra checks, lint and cleanup commands.
`tokens-independent.py` contains the tokenizer-only checks and the reference/candidate compilation of the final
`tools/api_test.cpp` source (copied as `api-regression.cpp` at the recorded path). Exact executed argument arrays
and exit codes are in `evidence/token-commands.json`. Model-folder tokenizer/template hashes are in
`evidence/tokenizer-manifest.json`. These checks read tokenizer metadata and supplied generated text, not weights.

No model generation, HTTP server, GPU validation, logits or performance measurement was run for this change.
The production diff is tier 0.5; the whole-diff helper report conservatively classifies the two existing C++ test
files as unclassified tier 1. Those files build test executables, not production GPU paths.

`evidence/cleanup.log` confirms no server was started and verification scratch was removed. Logs here are frozen.
