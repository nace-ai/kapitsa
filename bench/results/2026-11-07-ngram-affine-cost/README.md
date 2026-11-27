# Resident affine Q4 cost decomposition

This follow-up measures the complete library row loop, prefetch calls, and gather separately. It supports an operator-level improvement; it does not establish overall inference acceleration or cold-table loading performance.

Runs: M3 Ultra (Mac15,14), 96 GiB, macOS 26.4.1 (25E253); M4 Pro (Mac16,7), 48 GiB, macOS 15.7.1 (24G231). Reference `01b4c6d69fbe69e862c00358f88c94a363129592`; implementation `8c0a7ec3bf72623f6c7f92524ef229f3643a9e67`. Exact complete-library and probe hashes plus compilation/link arguments are in each host's `identities.json`. The M4 executes the same M3-built arm64 binaries (deployment target macOS 14), not a fresh local compilation. This predates the rebase onto `4d1d7445`; those revisions remain the source of these timings.

Pinned input: `beamster/Qwen3.8-Flash-Next-Sushi-2bpw` at `f37e7ba46f0f2993b6ca373221530c579f258712`, first recorded corpus sequence. T=1/16/512/1536, 16 heads, dimension 160, group 32. Each phase prefaults rows, warms for 250 ms, then uses an adaptive repeat count targeting 100 ms per sample, nine samples per case. Every sample, process user/system CPU time, command, environment, battery/power report and thermal-limit report is retained.

Both hosts use sparse tables with original header, logical size, row IDs and tensor offsets. Only accessed rows contain verified source bytes. M4 reconstructed 16,064 distinct required rows from an already verified 83,302-row payload, and rechecked every byte; all T prefixes are covered. It allocated 768,131,072 bytes for a 32,000,153,976-byte logical file. An earlier broader local reconstruction stopped at its disk-reserve guard; the incomplete generated file was removed before these runs. This was a preparation failure, not a failed numerical test. No complete model was loaded in this diagnostic.

## Single-token components

Microseconds per 16-row operation; each entry is the median of nine samples. A/B/A/B order is preserved.

| Arm | Row loop | Prefetch only | Default gather |
| --- | ---: | ---: | ---: |
| M3 default-0-base | 1.825 | 6.517 | 8.336 |
| M3 default-1-candidate | 0.291 | 6.424 | 6.851 |
| M3 default-2-base | 1.813 | 6.438 | 8.271 |
| M3 default-3-candidate | 0.291 | 6.534 | 7.370 |
| M4 default-0-base | 1.662 | 5.644 | 7.685 |
| M4 default-1-candidate | 0.267 | 5.573 | 5.998 |
| M4 default-2-base | 1.659 | 5.643 | 7.438 |
| M4 default-3-candidate | 0.273 | 5.748 | 6.069 |

Default single-token gather is 10.9–17.8% lower latency in the two M3 pairs and 18.4–22.0% lower in the M4 pairs. The row loop itself is about 6.1–6.3x faster here. Prefetch cost dominates much of the remaining gather time: M3 single-token prefetch spends about 4.15–4.26 us in system CPU time, versus effectively zero in the row loop. These are observations under the recorded background load, not confidence bounds or controlled-clock results.

## Complete gather, every arm

`no-prefetch` runs set the existing `NI_PLE_PREFETCH=0` only for diagnosis. They are not production-default results and do not justify disabling prefetch. All `default` runs leave it unset. The driver directly calls the library, so the variable reaches the measured process.

| Arm | T=1 us | T=16 us | T=512 us | T=1536 us |
| --- | ---: | ---: | ---: | ---: |
| M3 default-0-base | 8.336 | 277.712 | 6454.758 | 19210.750 |
| M3 default-1-candidate | 6.851 | 273.419 | 6394.636 | 19638.900 |
| M3 default-2-base | 8.271 | 278.301 | 6503.172 | 19860.217 |
| M3 default-3-candidate | 7.370 | 235.038 | 5226.893 | 15374.389 |
| M3 no-prefetch-0-base | 1.842 | 68.936 | 236.032 | 582.721 |
| M3 no-prefetch-1-candidate | 0.293 | 70.889 | 118.542 | 233.493 |
| M3 no-prefetch-2-base | 1.901 | 70.583 | 236.117 | 588.284 |
| M3 no-prefetch-3-candidate | 0.292 | 72.806 | 118.212 | 214.593 |
| M4 default-0-base | 7.685 | 217.017 | 6092.469 | 19696.708 |
| M4 default-1-candidate | 5.998 | 214.526 | 5869.003 | 19108.100 |
| M4 default-2-base | 7.438 | 226.234 | 7782.575 | 18135.017 |
| M4 default-3-candidate | 6.069 | 213.753 | 7630.845 | 19218.108 |
| M4 no-prefetch-0-base | 1.659 | 77.725 | 201.260 | 549.236 |
| M4 no-prefetch-1-candidate | 0.272 | 63.047 | 92.861 | 186.900 |
| M4 no-prefetch-2-base | 1.685 | 78.307 | 196.647 | 572.158 |
| M4 no-prefetch-3-candidate | 0.275 | 74.729 | 91.748 | 155.650 |

Larger default gathers are not consistently faster: M3 first T=1536 pair and M4 second T=1536 pair regress. Prefetch, worker start/scheduling and background load remain part of these timings. All observations are retained; no general gather or model-throughput claim is made. The probe verifies finite/nonzero outputs and guards, but its sums are not a bitwise comparator; a separate fresh M4 probe additionally compares all 10,280,960 bytes from 16,064 required trained rows through both complete libraries. They match byte for byte, with finite/nonzero values and intact guards (`m4/row-correctness.json`). Prior mapped-row and M3 model output checks provide broader coverage.

## Measurement conditions and limits

Each host held its exclusive GPU measurement lock to avoid overlap with GPU validation jobs, although this probe is CPU-only. Both report no recorded thermal/performance warnings. Neither CPU clocks nor other applications' CPU use were controlled. Background CPU activity was substantial, particularly on the M4; do not describe these hosts as idle. `ps %cpu` totals can exceed 100% across cores and are not instantaneous hardware utilization.

| Arm | Mean foreign ps CPU % | Maximum foreign ps CPU % |
| --- | ---: | ---: |
| M3 default-0-base | 50.47 | 120.7 |
| M3 default-1-candidate | 28.50 | 77.0 |
| M3 default-2-base | 36.66 | 84.1 |
| M3 default-3-candidate | 51.68 | 218.1 |
| M3 no-prefetch-0-base | 27.74 | 43.1 |
| M3 no-prefetch-1-candidate | 70.54 | 188.7 |
| M3 no-prefetch-2-base | 33.71 | 55.9 |
| M3 no-prefetch-3-candidate | 44.02 | 120.4 |
| M4 default-0-base | 185.01 | 315.4 |
| M4 default-1-candidate | 91.61 | 179.7 |
| M4 default-2-base | 165.96 | 424.4 |
| M4 default-3-candidate | 171.93 | 325.8 |
| M4 no-prefetch-0-base | 80.10 | 132.3 |
| M4 no-prefetch-1-candidate | 165.31 | 358.2 |
| M4 no-prefetch-2-base | 98.47 | 175.4 |
| M4 no-prefetch-3-candidate | 144.31 | 346.6 |

Raw command lines are in `results.json`. Reproduce using `cost-check.cpp` linked to each complete Release library with that build's `ni_run` compile/link flags. Invoke `cost-<arm> MODEL CORPUS default`, and separately `NI_PLE_PREFETCH=0 cost-<arm> MODEL CORPUS no-prefetch`. Use the same source-verified table rows on both arms, preserving their original offsets; validate all accessed rows before accepting results. `run-cost.py ROOT MODEL` records the interleaving and conditions and refuses an existing result directory. Models, binary payloads and executables are not committed.
