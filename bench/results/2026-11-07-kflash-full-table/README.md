# Complete-table Flash-Next follow-up

## Complete-table Flash-Next HTTP and native benchmark

Runs: Mac Studio (2025), Apple M3 Ultra (60 GPU cores), 96 GiB, macOS 26.4.1 (25E253), AC power and nominal thermal state. The exact complete ARM64 executables were compiled on the M4 Pro and transferred with SHA256 verification; Metal compiles on the remote GPU. Reference version `2fe7c3fa208c098b4e00b1bae9033172dfafaac7`; candidate implementation/version `0be92def43f00c7c5f15b889daa71309dce4d5a7`. The candidate checkout at build time was documentation head `a263df926a987c9b67e027cb4e5bd9287afddc79`; CMake was explicitly reconfigured with the implementation version, and production source hashes are recorded.

Pinned model: `beamster/Qwen3.8-Flash-Next-Sushi-2bpw`, revision `f37e7ba46f0f2993b6ca373221530c579f258712`. The full 32,000,153,976-byte n-gram table passed SHA256 `c8ab74bc343408cf3923d7d64b3698fbeb3e78c07ce7f85a650a8278731251d2`. Only that table is on a temporary 34 GiB RAM-backed HFS+ volume; all 214 source-verified non-table files remain on the internal SSD via symlinks. Earlier sparse inputs and published evidence are unchanged.

The unchanged HTTP suite reports **49 PASS / 0 FAIL on the reference**, **48 PASS / 1 FAIL on the candidate**. The candidate failure is `tool result: final answer mentions 3`, with empty content after the allowed two extra tool-result rounds. Temperature and seed are unset in that test; the cause is not established. Both fresh private servers use two slots, context 32768 and disk cache disabled. All other assertions pass. The original 4B HTTP series remains **48 PASS / 1 keepalive FAIL on each arm**; this new Flash-Next test does not replace it. An independent targeted tool-loop diagnostic for this PR is pending.

Native `kapitsa bench --quick --json` ran reference/candidate/reference/candidate under the GPU lock:

| Arm | Sampled median tok/s | Greedy median tok/s | 4K TTFT s |
| --- | ---: | ---: | ---: |
| 1023-bench-0-base | 57.43 | 58.77 | 4.8604 |
| 1023-bench-1-candidate | 54.95 | 60.03 | 4.8526 |
| 1023-bench-2-base | 56.20 | 58.62 | 4.8463 |
| 1023-bench-3-candidate | 55.64 | 59.15 | 4.8407 |

Every arm exits 0, tool checks pass, `flags: []`, `invalid: []`. Native conditions are nominal, with zero swapouts and no reported contention. Per-process mean foreign GPU is 0.01%, 0.01%, 0.03%, 0.03%; all four summaries are unflagged with zero busy samples. The JSON files retain all three sampled and three greedy runs per arm, and all monitor samples. Fresh nonces and sampled answers differ; these are not exact-token comparisons. There is no consistent overall inference-speed or memory benefit claim.

This model uses its MTP fallback, not the packed 4B draft projection changed here. These checks therefore do not supply another applicable packed-model setup. RAM-table timings do not establish normal SSD cold-table loading performance; process peak-memory figures omit the RAM volume's system memory. These are this PR's own results; shared input preparation does not reuse another PR's validation.

`results.json` records source/executable identities, table identity, commands, environments, return codes, HTTP results, native conditions and contention samples. Full HTTP/server logs and each native JSON are retained. Earlier evidence remains frozen.
