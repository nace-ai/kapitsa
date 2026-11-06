# Controlled 64-token draft graph repeat

Runs: Apple M4 Pro, 20 GPU cores, 48 GiB, macOS 15.7.1 (24G231). Actual Qwen3.5-4B 4-bit/group-64 target with its matching six-layer v1 DFlash2 drafter (hidden 2560), using the existing packed projection profile.

Reference source: `2fe7c3fa208c098b4e00b1bae9033172dfafaac7`. Candidate implementation: `0be92def43f00c7c5f15b889daa71309dce4d5a7`. Both original complete-graph binaries were reused unchanged and their SHA256 values were checked against the original records before execution. They were configured before the implementation commit and carry the base version label; the original source-file hashes establish the candidate identity. No source was changed or rebuilt for this repeat.

This six-arm A/B/A/B/A/B series follows the earlier 64-token series with observed foreign GPU activity. One process held the exclusive GPU lock. Before every arm, foreign GPU activity had to stay at or below 1% for 10 seconds. All 160 in-run foreign and ambient GPU samples were zero; each timing sample reported nominal thermal state. The raw samples and exact argv/environment are retained in `results.json` and the six logs.

Each process measures 12 command buffers of 48 complete draft passes with fixed inputs and anchor. The table uses the median of the last 10 samples; the first two remain in the evidence.

| Arm | Median GPU microseconds per pass |
| --- | ---: |
| 0-base | 2653.536458 |
| 1-candidate | 2644.997830 |
| 2-base | 2656.056857 |
| 3-candidate | 2645.973958 |
| 4-base | 2652.839843 |
| 5-candidate | 2646.842015 |

Adjacent reference/candidate reductions are 0.322%, 0.380% and 0.226%. This is a small, consistent directional difference in this short graph series, not a demonstrated generation-throughput gain. The clean 1536-token repeat had opposite pair directions and the native benchmark series showed no stable inference gain. These results do not justify a broad speedup claim or settle the remaining default-path performance review. No more unchanged graph repetitions are planned without a new implementation or hypothesis.

This timing driver excludes target verification, acceptance, scheduling, HTTP and load time. Numerical correctness checks were not repeated: the previously published 270-file actual packed-drafter comparisons at each length and canonical target-model output checks remain separate, frozen evidence. This is not another hardware/model setup.

Reproduction uses the graph driver, input IDs, build/source identity records and binary recipes in `../2026-11-07-kflash-reduce-v8/`. `run-original.py` is the exact orchestration snapshot with the original workspace layout; paths must be adapted for another checkout and the output folder must be new. Keep its exclusive GPU lock and per-arm quiet/contention checks. `results.json` preserves the exact executed paths and environment. Published earlier files are unchanged.

Not verified here: Flash-Next fallback, another applicable packed-profile setup, unrestricted HTTP correctness or a full-inference performance improvement. The separate HTTP comparison still has 48 PASS / 1 identical busy-slot keepalive FAIL per arm. No floor or default was changed for this measurement.
