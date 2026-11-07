# Complete-table affine decoder follow-up

## Complete-table HTTP and native benchmark

Runs: Mac Studio (2025), Apple M3 Ultra (60 GPU cores), 96 GiB, macOS 26.4.1 (25E253), nominal thermal state and AC power. Own complete Release binaries of reference `01b4c6d69fbe69e862c00358f88c94a363129592` and implementation `8c0a7ec3bf72623f6c7f92524ef229f3643a9e67`, with matching configured version labels and SHA256 identities.

Model: `beamster/Qwen3.8-Flash-Next-Sushi-2bpw`, revision `f37e7ba46f0f2993b6ca373221530c579f258712`. The complete 32,000,153,976-byte table passed SHA256 `c8ab74bc343408cf3923d7d64b3698fbeb3e78c07ce7f85a650a8278731251d2` before testing. Only the table resides on a temporary 34 GiB RAM-backed HFS+ volume; 214 source-verified non-table files remain on the internal SSD through symlinks. Earlier sparse fixtures and evidence are unchanged.

The original HTTP suite exits 0 with **49 PASS / 0 FAIL on the reference** and exits 1 with **48 PASS / 1 FAIL on the candidate**. The candidate failure is `tool result: final answer mentions 3`, with empty final content. That test leaves temperature and seed unset and permits two additional tool-result rounds. The cause has not yet been established; the initial failure is retained, and the PR remains Draft. Each server uses two slots, context 32768 and disk cache disabled. All other assertions pass, including the other tool-loop API checks.

Native `kapitsa bench --quick --json`, A/B/A/B under the GPU lock:

| Arm | Sampled median tok/s | Greedy median tok/s | 4K TTFT s |
| --- | ---: | ---: | ---: |
| 1022-bench-0-base | 56.71 | 59.60 | 4.8480 |
| 1022-bench-1-candidate | 58.38 | 59.57 | 4.8517 |
| 1022-bench-2-base | 55.42 | 59.38 | 4.8459 |
| 1022-bench-3-candidate | 57.47 | 58.78 | 4.8466 |

All four arms exit 0 with tool checks passing, `flags: []` and `invalid: []`. All per-process contention summaries are unflagged, with mean foreign GPU 0.04%, 0.03%, 0.04%, 0.03%; each has zero busy samples. Native conditions report nominal thermal state and zero swapouts. Every arm's three sampled and three greedy measurements are retained in JSON. Sampled medians favor the candidate in these pairs, but greedy and TTFT do not show a consistent gain, so no general inference-speed improvement is claimed. Nonces and sampled answers differ; this benchmark is not exact-token evidence.

RAM-backed table timing does not establish SSD cold-table loading performance. Process peak-memory values omit the RAM volume's system memory and do not establish total memory use or a reduction. These are this PR's own outputs; other changes share input bytes only.

`results.json` records executable identities, model/table identity, exact commands and environments, return codes, HTTP outcomes, all benchmark conditions and contention samples. Individual HTTP/server logs and benchmark JSON are retained. No previous evidence was overwritten. A targeted tool-loop diagnostic is pending; later evidence must keep this initial failure visible.
