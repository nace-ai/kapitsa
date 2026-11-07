# Stop-string follow-up checks

Reference: `7a5d673e9544b538a54cf7b4d7c5cbcef2e49d5c`.
Implementation: `cf2f25c7f9f38f9babfc51a0def9d55697544b73`.
Runs: Apple M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1 (25E253), Apple clang 17.0.0. The independently configured Release binaries report `kapitsa 0.2.31 (7a5d673)` and `kapitsa 0.2.31 (cf2f25c)`.

This folder adds completed evidence to the earlier boundary and grammar folders without modifying them. No production source changed after cf2f25c7. Commands, full API responses, model metadata and source/binary hashes are retained in the JSON and logs. Every model run held the exclusive GPU lock; its children inherited `NI_LOCK_HELD=1`.

## API and numerical checks

Both Qwen3.5 4B 4-bit and Qwen3.6 35B-A3B 4-bit reproduce the original four split-stop leaks on the reference. Candidate passes all dedicated stop assertions, including Chat Completions/Messages, streaming/nonstreaming, matched metadata, and unmatched suffixes at EOS/token limit. Each model's current-reference/candidate logit comparison reports 584 finite bitwise-identical rows: 8 prefill, 192 step, 192 verify and 192 batch; all gates pass. The raw comparison JSON and verbatim reports are retained. These results do not claim a model path not run.

Standard HTTP: 35B has 49 passes on each build. 4B has 47 passes and the same two failures on each: concurrent completion ordering and queued-stream keepalive timing. Failures remain failures.

## Native A/B measurements

`kapitsa bench --model MODEL --quick --json --out FILE` produced each result, in A/B/A/B order within each series. A is reference, B is candidate. 4B uses its discovered DFlash2 drafter/KFlash; 35B uses plain decoding. No environment-switch A/B was attempted. Each row gives all three sampled and greedy runs, not a selected best run. Complete flags, prompt sizes, thermal/power conditions, cache follow-ups and command times remain in each original JSON. All 12 commands exited 0 with empty `invalid`; that does not erase warning flags.

| Series / arm | Sampled decode tok/s (3 runs) | Greedy tok/s (3 runs) | 4K TTFT s | Other GPU mean/max % | Flags |
| --- | --- | --- | --- | --- | --- |
| gpu-results/35b-bench-0-base | [191.92, 181.68, 186.64] | [190.77, 189.18, 187.67] | 1.4865 | 0.05/0.53 | none |
| gpu-results/35b-bench-1-candidate | [187.24, 186.63, 187.34] | [190.25, 191.62, 184.26] | 1.4852 | 0.0/0.0 | none |
| gpu-results/35b-bench-2-base | [188.91, 182.42, 181.77] | [184.3, 183.55, 183.83] | 1.4847 | 0.05/0.54 | none |
| gpu-results/35b-bench-3-candidate | [190.07, 183.89, 181.33] | [183.31, 184.59, 185.32] | 1.4875 | 0.0/0.0 | none |
| gpu-results/4b-bench-0-base | [335.09, 343.38, 374.65] | [384.93, 330.76, 329.69] | 1.937 | 0.09/0.63 | none |
| gpu-results/4b-bench-1-candidate | [410.76, 371.98, 345.14] | [365.71, 379.71, 384.68] | 1.9373 | 0.0/0.0 | decode: repetition loop (distinct-4-gram 0.3856); decode: repetition loop (distinct-4-gram 0.4575) |
| gpu-results/4b-bench-2-base | [310.1, 339.51, 352.16] | [364.69, 373.76, 372.26] | 1.9368 | 0.0/0.0 | none |
| gpu-results/4b-bench-3-candidate | [380.34, 337.33, 315.12] | [371.36, 384.41, 367.72] | 1.936 | 0.07/0.52 | none |
| m3-4b-quiet/4b-repeat-0-base | [352.56, 341.26, 356.85] | [374.61, 335.58, 349.61] | 1.9351 | 0.1/0.67 | decode: repetition loop (distinct-4-gram 0.4945) |
| m3-4b-quiet/4b-repeat-1-candidate | [380.96, 338.27, 324.77] | [359.59, 375.96, 371.7] | 1.9349 | 0.0/0.0 | decode: repetition loop (distinct-4-gram 0.4759) |
| m3-4b-quiet/4b-repeat-2-base | [332.4, 330.53, 345.28] | [342.39, 374.42, 367.4] | 1.9368 | 0.0/0.0 | decode: repetition loop (distinct-4-gram 0.4593) |
| m3-4b-quiet/4b-repeat-3-candidate | [369.95, 339.76, 340.88] | [384.43, 374.61, 332.83] | 1.9384 | 0.0/0.0 | none |

The original 4B series overlapped the full vocabulary grammar check during at least its first two arms; retain it as confounded timing. A complete second 4B series ran after that check and the main worker had finished. All four replacement preflights returned true; the helper reports foreign processes above 0.5%, so its requested 0.1% quiet threshold is not proof of zero activity. Native in-run GPU measurements are shown above; all runs report nominal thermals, AC power, zero swapouts and no contention entries.

For the replacement 4B series, sampled medians A/B/A/B are 352.56/338.27/332.40/340.88 tok/s and greedy medians are 349.61/371.70/367.40/374.61. The first three arms carry sampled-output repetition warnings. For 35B the sampled medians are 186.64/187.24/182.42/183.89 and greedy medians are 189.18/190.25/183.83/184.59. Prompt lengths and sampled continuations vary in this protocol. These measurements establish completed A/B coverage, not an inference speedup or a uniform no-slowdown claim.

## Remaining limits

The earlier two full candidate updater failures remain recorded. New complete diagnostics (only add exit/stdout/stderr reporting, no truncation) pass for candidate, reference and candidate again. A bounded watcher-only series then captures candidate SIGKILL (return -9, empty stdout/stderr) at its seventh case, after eight passing reference cases. A separate reference-only control captures the same SIGKILL/empty-version failure at its eighth case. All 23 watcher cases are retained: 16 reference cases (15 pass, one fails) and seven candidate cases (six pass, one fails). Watcher cases are not full-suite passes; the runner exits 0 when its diagnostic execution completes, so read WATCH_RESULT/FAIL, not just exit status.

This reproduces the failure on the pre-change reference and establishes that the failure does not require the stop-filter patch. It does not identify who sent SIGKILL or prove an OS/signature root cause. The same updater and version-report source is unchanged between builds; --version never enters the stop filter. The original failures are retained as an existing updater validation limitation, not counted as passes. Diagnostics never edit production sources or the checked-in test. The bounded controls stop after capturing the failure, not after finding a passing run.

Actual Dory checkpoint fallback and the alternate backend runtime have not been exercised. The buffer/filter CPU cases do not prove those execution paths. No new kernel, precision or default is introduced.
