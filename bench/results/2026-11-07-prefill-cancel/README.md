# Completed-prefix recovery after prefill cancellation

This is preliminary functional and CPU evidence for a Draft. It does not qualify a speedup, a full-model output gate, or the whole default class.

## Sources and environment

- Reference: `2c41b56cc6c5f0895d4359d5ff33d1459b22bc39`.
- Engine implementation and initial CPU build: `926bce8b93b1c79bc268dbe92046c4b6f0526576`.
- Socket-preflight correction: `6001dc16e4e8c32860ba778054435d578402f1d4`.
- Final GPU and tokenizer checks: `3ce432031569e364af6731ffb59aec1bc67355ef`. This changes a comment after the harness correction. The engine object SHA256 before and after that edit is identical; the executable version was refreshed by explicit CMake configure/build.
- Apple M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1 (25E253), Apple clang 17, CMake Release `-O3`.
- Existing Qwen3.5 4B 4-bit folder and its automatically loaded DFlash drafter. One slot, context 32768, SSD cache off, `NI_ANE=0`, temporary homes, exclusive GPU lock.

`source-environment.json` retains the initial identities and model metadata hashes. `remote/source-environment-final.json` identifies the final source, test, executable and engine object. The last `remote/evidence/cancel-checkpoint-final/3ce43203/build.log` is the subsequent parser/grammar target build; the CLI build result is in `remote/qualify-final.log`, with explicit configuration in `remote/configure-final.log`.

## Functional regression

The same corrected test runs an uninterrupted control and a twice-cancelled continuation. Each completed response requests 32 output tokens.

| Observation | Reference candidate leg | Final candidate leg |
| --- | ---: | ---: |
| Short continuation cached tokens | 2,051 | 2,051 |
| Retry after two cancellations, cached tokens | 0 | 2,051 |
| Completed retry reused by next request | 15,216 | 15,216 |
| Cold-cancel retry cached tokens | 0 | 0 |
| Checks passed / failed | 7 / 1 | 8 / 0 |

The reference fails only `retry keeps the completed prefix`. Both final candidate cancellations are confirmed by the server at 7,697 / 15,223 prompt tokens. The cold request saves a new system checkpoint at 2,900, cancels at 10,017 / 16,026, then retries cold. The final candidate seed, retry and repeat returned text match the corresponding controls. No token-ID or full-vocabulary logit comparison was performed.

- `reference-regression/`: corrected baseline run; `reference.log` and `reference.rc` record the intentional failing regression.
- `remote/regression-4b-final/`: exact requests, SSE events, returned usage/text, server logs and eight-check summary.
- `initial-harness-failure/`: original reference attempt. It completed the first leg, then rejected the previous server's closing sockets at port preflight. `SO_REUSEADDR` fixes the harness; the original log remains.

These traces contain per-request timings, but they are not idle, interleaved A/B evidence. No inference speedup is claimed.

## Build and CPU checks

The initial Release build passed. The full `vk cpu-tests` invocation passed 20 of 21 suites. Its updater watcher saw an empty version during atomic executable replacement. A single isolated complete updater rerun passed 148 requests with zero failures; the original failure remains unexplained and retained. This is not one clean all-pass CPU sweep.

Fifteen supplementary checks passed before the comment gate rejected the phrase “only after” in the new event-order comment. The comment was reworded without altering a gate, and the four remaining supplementary checks passed. The separate packaging patch-application check passed. Parser and structured CLI checks passed with both the existing 4B and 35B tokenizer folders. All commands and logs are under `remote/`; local `prek` logs are retained separately.

## Pending before Ready

Run the full HTTP suites, old/new model logit gates and idle native A/B/A/B on dense 4B and MoE 35B. Repeat the cancellation regression on 35B and qualify retained allocation size and cancellation/retry latency. Investigate the initial updater failure. Cover ANE, MTP without a block drafter, LiLiCorr/DSpark, Dory rings, split models, images, multiple slots, memory pressure, and SSD restore/write-behind as applicable. None of those checks is established by the short returned-text comparison.

`remote/remote-manifest.json` verifies the downloaded snapshot. `SHA256SUMS` covers this new evidence bundle except itself. Existing published result files were not changed.
