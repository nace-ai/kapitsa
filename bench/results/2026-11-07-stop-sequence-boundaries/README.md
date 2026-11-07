# Stop-string boundary checks

Reference: `7a5d673e9544b538a54cf7b4d7c5cbcef2e49d5c`.
Implementation: `cf2f25c7f9f38f9babfc51a0def9d55697544b73`.
Runs: Apple M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1 (25E253), Apple clang 17.0.0. Release builds were independently configured from both actual Git revisions. Their version strings use 7-character abbreviated hashes.

The initial reproduction used the older independently configured b171f983 binary. Its engine/output parser/Chat Completions source files are unchanged in 7a5d673e. `repro-base.json` records `stop="one,"` leaking `one` in streamed and nonstreamed replies. The dedicated paired API regression uses the exact current reference and candidate instead.

## CPU evidence

`cpu/cpu.log` records 20 passing suites and one failed suite. `api_test` passed 346 checks, including 76 new stop-filter cases. Tokenizer/parser/structured tests, supplementary targets and final lint are retained. `grammar.log` is the 233-check invocation without a tokenizer; its skipped mask checks must not be counted as executed.

The initial complete `test_update.py` failed once while repeatedly executing a signed binary during its replacement: it observed an empty version. The complete reference control passed, while the complete candidate repeat failed the same assertion. Both failures are retained. Two candidate diagnostic runs of sections 1–3 and one reference diagnostic run passed, but this does not explain or erase the full-run failures. The diagnostic wrapper only adds reporting for an empty version and stops after section 3; it is not another full-suite pass. No update code or version-reporting path changed in this PR. This remains an unresolved validation limitation.

`build-cpu.sh` stopped after the first suite failure. `resume-cpu.sh` built the current reference and ran the related full update tests, then stopped at the candidate repeat failure. `finish-cpu.sh` performed the remaining checks without relabeling the initial suite as passing. The marker is `cpu-checked`, not `cpu-passed`.

## GPU evidence and remaining work

`gpu/4b-http-*-stops.log`: the current reference leaks `one` in all four Chat Completions/Messages streamed/nonstreamed cases; candidate passes every stop and terminal-prefix check. `gpu/4b-http-*.log`: both standard HTTP runs have 47 passes and the same two failures (concurrent completion ordering and queued-stream keepalive timing).

`gpu/4b-compare.json`: 584 finite rows are bitwise identical (8 prefill + 192 each step/verify/batch), with no gate failures. This is this branch's own paired 4B run, not another PR's result.

An additional full vocabulary grammar-mask test was still running when the first native benchmark started. Any overlapping timing must be disclosed and not treated as a clean performance gate. Its pending outcome is not represented by the earlier tokenizer-free grammar log.

Only completed reports included here establish results. The live worker continues paired standard HTTP, logit comparison and native A/B on 4B and 35B in a separate run directory. Pending checks are not passes, and no whole-model speedup or uniform no-slowdown is claimed. Later measurements must be added in a new evidence directory.

Dory fallback behavior with an actual checkpoint and the alternate backend's real runtime remain unverified. No kernel or tokenizer code changed.
