# Complete-table HTTP and native benchmark checks

Runs: Mac Studio (2025), Apple M3 Ultra (60 GPU cores), 96 GiB, macOS 26.4.1 (25E253), AC power, automatic mode, nominal thermal state. Separate complete Release builds: reference `934a6d72f2a393b6987baf0b98ebe5a9e07d69be`, candidate `dd166319cc746fd439beb77119416e610f9acf63`. The executable version labels and SHA256 values are in `results.json`; these are unchanged binaries from the earlier build/CPU evidence.

Model: `beamster/Qwen3.8-Flash-Next-Sushi-2bpw`, revision `f37e7ba46f0f2993b6ca373221530c579f258712`. The complete 32,000,153,976-byte n-gram table passed SHA256 `c8ab74bc343408cf3923d7d64b3698fbeb3e78c07ce7f85a650a8278731251d2` before the runs. All 214 other files were previously source-digest verified. Only the n-gram table resides on a temporary 34 GiB RAM-backed HFS+ volume; all other files remain on the internal SSD through symlinks. The earlier sparse fixture was not changed.

## HTTP

Both original `tests/test_server.py` invocations exit 0: reference **49 PASS, 0 FAIL**, candidate **49 PASS, 0 FAIL**. Each runs against a fresh private server with two slots, context 32768 and disk cache disabled. The exact server commands, test commands, environments and full logs are included. The tests use their original sampling defaults. A suite pass is not an exhaustive arbitrary-request guarantee.

## Native benchmark

The unchanged `kapitsa bench --quick --json` ran in reference/candidate/reference/candidate order while holding the GPU lock. All four commands exit 0, tool checks pass, and `flags` and `invalid` are empty. Each arm contains three sampled and three greedy decode measurements; individual measurements are retained in its JSON.

| Arm | Sampled median tok/s | Greedy median tok/s | 4K TTFT s |
| --- | ---: | ---: | ---: |
| 1019-bench-0-base | 56.65 | 59.10 | 4.9501 |
| 1019-bench-1-candidate | 57.17 | 59.69 | 4.8596 |
| 1019-bench-2-base | 57.81 | 58.86 | 4.8559 |
| 1019-bench-3-candidate | 54.49 | 58.83 | 4.8492 |

All four per-process contention summaries are unflagged; the first reference arm includes two brief busy samples (maximum foreign GPU 14.63%, mean 0.16%, `loginwindow`). Later arm means are 0.04%, 0.03% and 0.03%. Native conditions report no contention and zero swapouts. Raw monitor samples and all conditions remain in `results.json`.

Sampled performance varies and there is no consistent inference gain. This admission fix makes no speedup claim. Fresh benchmark nonces and sampled answers differ, so benchmark output is not exact-token evidence; the independent frozen window and fixed-generation bundles provide that evidence.

## Limits

These measurements do not establish SSD cold-table loading performance. Process peak-memory reports omit the RAM volume's system memory and must not be interpreted as total model memory or a memory reduction. No new long-context or constrained streamed-expert-budget run was performed. The full-table suite does not expand the prior fixed-output checks to all possible requests or other Macs. This bundle contains only this change's results; shared model bytes are not shared validation results.

`results.json` records source identities, table identity, commands, return codes, HTTP outcomes and benchmark conditions. `SHA256SUMS` covers all bundle files except itself. Earlier published evidence is unchanged.
