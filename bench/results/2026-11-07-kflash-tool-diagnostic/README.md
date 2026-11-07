# Independent tool-loop diagnosis

Runs: Mac Studio (2025), Apple M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1 (25E253). The complete table SHA256 and own binary identities are in `results.json`; the test source SHA256 is `4ef02e7091c2e863857af4f62b9eed2f8f9535164262cebb62274fa4f8587c25`. The same original HTTP helper functions and fixtures were extracted from that test source.

## Targeted tool-loop diagnosis

An independent reference/candidate/reference/candidate diagnostic used this PR's own `2fe7c3fa` / `0be92def` complete binaries and the same full pinned table on the M3 Ultra. Each arm started a fresh server, then ran four original-default requests, a greedy control (temperature 0, seed 11), and sampled controls (temperature 0.7, seeds 11 and 29). The original two-extra-round assertion was retained. Later requests up to eight extra rounds were diagnostic observations only.

| Arm | Original-default trials passed | Greedy, seed 11 | T=0.7, seed 11 | T=0.7, seed 29 |
| --- | ---: | --- | --- | --- |
| 0 base | 2/4 | PASS | FAIL | PASS |
| 1 candidate | 4/4 | PASS | FAIL | PASS |
| 2 base | 1/4 | PASS | FAIL | PASS |
| 3 candidate | 1/4 | PASS | FAIL | PASS |

For **each of the three fixed controls**, normalized response sequences match across all four arms: content, reasoning, finish reason and tool function names/arguments. Random tool IDs and usage/timing fields are excluded from this comparison. This is a small request-level comparison, not a new full-vocabulary numerical gate. Default trials leave temperature and seed unset; four per arm do not establish comparative failure rates.

The original-limit failure occurs on both versions. The fixture in `tests/test_server.py` returns `3` or `3 /etc/hosts` for every `run_shell` command, even requests for `cat`, `xxd` or another operation. Transcripts show repeated verification calls and, in some failed cases, unterminated tool markup. This exposes a shared fixture limitation; it does not establish that the fixture alone explains every failure. Fixed controls reproducing matching failures support a shared test/model behavior in these cases, rather than evidence of changed numerical outputs. The earlier complete-suite **49 PASS reference / 48 PASS plus 1 FAIL candidate** record remains unchanged and is not relabeled all-pass.

Every request/response transcript, fresh-server command, source/executable identity and original-versus-extended outcome is in [`2026-11-07-kflash-tool-diagnostic/`](https://github.com/chainhelen/kapitsa/tree/perf/kflash-reduce-v8/bench/results/2026-11-07-kflash-tool-diagnostic). Production source, original test assertions and prior evidence were not changed. The RAM-table storage limitation still applies; these tool checks make no timing or memory claim.

To reproduce a case, start its recorded server command while holding the GPU lock with `NI_LOCK_HELD=1`, and replay its transcript's request bodies to `/v1/chat/completions` in order. Tool results are fixture strings; no host shell command in a model response was executed. Preserve each returned tool ID when supplying that tool result. Replaying an unseeded case need not reproduce the original outputs. Each arm ran all seven cases on one fresh server; controls followed the default trials, so the record includes ordinary in-memory prefix reuse. Disk cache was disabled.

Case files retain complete requests and responses; `results.json` retains per-case outcomes and normalized response signatures. Passing after more than the original two additional rounds is not an original-test pass. This bundle contains only this PR's own observations. `SHA256SUMS` covers every bundle file except itself.
