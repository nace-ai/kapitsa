# Wider model validation

Runs: Apple M4 Pro, 20 GPU cores, 48 GiB, macOS 15.7.1 (24G231), AC power, automatic power mode. Reference source `2fe7c3fa208c098b4e00b1bae9033172dfafaac7`; implementation `0be92def43f00c7c5f15b889daa71309dce4d5a7`. These are this PR's own complete baseline/candidate builds and fresh outputs, with no other PR's results substituted.

Both builds were explicitly CMake-configured for the preceding follow-up and reused unchanged here. The candidate cache retained its implementation label `0be92def` while that checkout was at documentation/evidence head `a263df92`. Version strings, source/library/binary hashes, model metadata and exact commands are in `identity-and-commands.json`.

The 35B OQ4E target-model output check covers 584 full-vocabulary rows: 8 prefill and 192 each for step, verify and batch. Every row is bitwise identical and finite. No calibrated floor exists for this exact model; the comparator reports exact equality without a floor claim. Its full report, logs and per-path metadata/records/sequences are included. Raw float dumps were removed after comparison. The preceding 27B 584-row output/floor check remains frozen in the separate `2026-11-07-kflash-graph-and-27b` folder.

The native `kapitsa bench --quick --json` series ran A/B/A/B for each model, holding the GPU lock and using a fresh home per arm. Each result contains three sampled and three greedy decode measurements, 4K prefill/cached follow-ups and tool checks. Every arm exited zero, has `flags: []` and `invalid: []`, and passes tool checks. No samples were dropped. The 27B thermal state was fair throughout; 35B was nominal throughout; no swapouts occurred.

| Native arm | Sampled median tok/s | Greedy median tok/s | 4K TTFT s | Peak GB | Foreign GPU mean/max |
| --- | ---: | ---: | ---: | ---: | ---: |
| bench-27b-0-base | 60.83 | 66.21 | 10.4510 | 19.18 | 0.04/0.73% |
| bench-27b-1-candidate | 61.18 | 66.77 | 10.4139 | 19.18 | 0.08/0.88% |
| bench-27b-2-base | 61.03 | 66.88 | 10.4510 | 19.18 | 0.01/0.51% |
| bench-27b-3-candidate | 58.79 | 66.12 | 10.4076 | 19.18 | 0.07/1.00% |
| bench-35b-0-base | 111.14 | 112.85 | 3.6935 | 20.57 | 0.03/0.62% |
| bench-35b-1-candidate | 112.34 | 112.45 | 3.6981 | 20.56 | 0.04/0.71% |
| bench-35b-2-base | 110.80 | 113.41 | 3.6969 | 20.57 | 0.00/0.00% |
| bench-35b-3-candidate | 109.12 | 112.40 | 3.7000 | 20.57 | 0.04/0.76% |

27B is Qwen3.8-27B 4-bit/group64. Native logs show its five-layer DFlash2 drafter with hidden size 5120; this is the existing unpacked path. The 35B model is Qwen3.6-35B-A3B OQ4E with MTP; every server log explicitly reports no matching external drafter. The 35B series therefore exercises fallback behavior, not packed-drafter execution. Neither series constitutes another packed 4B setup.

Fresh native request nonces and sampled continuations vary across invocations, so these are not an exact-input numerical comparison. They show no consistent inference gain and no meaningful memory reduction; no such claim is made. The fixed corpus comparisons provide the separate numerical evidence. Earlier 4B HTTP results remain 48 PASS / 1 matching keepalive-timing FAIL per arm. Flash-Next fallback coverage and the packed path's performance assessment remain incomplete, so the PR remains Draft.
