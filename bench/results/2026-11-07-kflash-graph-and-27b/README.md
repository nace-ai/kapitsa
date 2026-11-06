# Longer draft graph and 27B target-model checks

Runs: Apple M4 Pro, 20 GPU cores, 48 GiB, macOS15.7.1 (24G231). Reference source `2fe7c3fa208c098b4e00b1bae9033172dfafaac7`; candidate implementation `0be92def43f00c7c5f15b889daa71309dce4d5a7`. Source is unchanged from the earlier evidence.

The 1536-token graph repeat uses the same original checked graph binaries and exact prompt as the first series. Their binary hashes match the original records. Those preliminary binaries were configured before the implementation commit and carry the base label; the original source-file hashes establish the source identity. No new binary was built for this repeat. The graph driver is retained in the initial evidence folder.

One caller held the exclusive GPU lock. Before each arm, foreign GPU use had to remain at or below1% for10seconds. All sampled foreign and ambient GPU activity during these four arms was0%; samples are retained. Each process ran12timed samples of48complete draft passes per command buffer. The table uses the median of the final10samples, while every sample remains in `graph/results.json` and the raw logs.

| Arm | Median GPU microseconds/pass |
| --- | ---: |
| 0 reference | 2930.126302 |
| 1 candidate | 3022.451823 |
| 2 reference | 3030.211807 |
| 3 candidate | 2926.739583 |

The first adjacent pair favors the reference and the second favors the candidate. This controlled repeat does not establish a consistent1536-token graph benefit. It measures only the draft graph, excluding acceptance, target verification, server scheduling and load time; no inference speedup follows from it. Earlier measurements remain unchanged.

The separate27B output check uses the complete canonical baseline and candidate builds. Both were explicitly CMake-configured during the preceding follow-up; the candidate version label is pinned to implementation0be92def while that checkout's documentation/evidence head was a263df92. Version strings and source/library/binary hashes are recorded in `27b/identity-and-commands.json`. No results from another PR are used.

Model: Qwen3.8-27B4-bit/group64. Eight fixed corpus windows,14,848prefilled tokens:8prefill rows and192rows each for step, verify and batch, **584/584 full-vocabulary rows bitwise identical**, with no non-finite values. The unchanged standing27B floor comparison passes; no floor was edited. Commands, metadata, row manifests, sequences and the complete comparator report are retained. Raw float dumps were removed only after the comparison succeeded.

This is target-model/fallback coverage. These logit-dump runs do not establish a second packed drafter setup or check end-to-end speculative generation. Their load/prefill timings are not controlled performance measurements; the baseline compiled ANE programs and the candidate reused that program cache. Separate27Bnative A/B/A/B benchmarks and35Bchecks are still running and are not included or claimed complete here. Flash-Next fallback and broader packed-drafter validation remain pending. The earlier paired4BHTTP results remain48PASS/1samekeepalive-timingFAIL per arm.
