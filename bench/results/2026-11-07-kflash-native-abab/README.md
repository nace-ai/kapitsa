# Native 4B benchmark follow-up

Runs: Apple M4 Pro, 20 GPU cores, 48 GiB, macOS 15.7.1 (24G231), AC power, automatic power mode. Model: Qwen3.5-4B 4-bit with its matching six-layer v1 DFlash2 drafter. Each server log confirms 39 packed draft projections. These are `kapitsa bench --quick --json` runs, with three sampled and three greedy decode measurements of 512 requested tokens, a 4K prefill and cached follow-ups, and tool checks. Exact commands and individual rates are retained.

Reference implementation: `2fe7c3fa208c098b4e00b1bae9033172dfafaac7`. Candidate implementation: `0be92def43f00c7c5f15b889daa71309dce4d5a7`. Both checkouts were explicitly configured and built for the preceding continuation/HTTP checks. Candidate CMake retained the implementation version label `0be92def`, while the checkout was at documentation/evidence head `a263df92`. `identities.json` records source and binary hashes, version strings, and model metadata. The same binary hashes were checked before this sequence; neither binary was rebuilt between arms.

Each arm used a new scratch home. One caller held `/tmp/kapitsa-gpu.lock` for the A/B/A/B sequence and passed `NI_LOCK_HELD=1`; native benchmark quiet checks and contention monitoring stayed enabled. Other applications may still use the GPU. Every completed arm is included, with all flags preserved.

| Arm | Sampled median tok/s | Greedy median tok/s | 4K TTFT s | Peak GB | Mean / max foreign GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 reference | 197.12 | 203.53 | 2.1592 | 6.90 | 3.59 / 9.65% |
| 1 candidate | 178.29 | 207.17 | 2.2457 | 6.94 | 4.06 / 19.65% |
| 2 reference | 193.32 | 210.19 | 2.1016 | 6.94 | 0.00 / 0.00% |
| 3 candidate | 194.60 | 206.24 | 2.0997 | 6.94 | 0.00 / 0.00% |

All four commands exited zero, report `invalid: []`, and pass tool checks. Flags: arm0 has contention during prefill and tools; arm1 has prefill contention and a sampled repetition flag (distinct-4-gram0.4506); arm2 has a sampled repetition flag0.4612; arm3 has a sampled repetition flag0.4602. No flag was suppressed. These measurements do not establish an inference speedup or memory reduction, and the contended measurements are not accepted as idle-GPU timing evidence.

The native protocol generates fresh session nonces (`src/bench/protocol.cpp::t_prefill`) and uses sampled decoding. The observed prefill counts are4135/4134/4134/4133. These native invocations are not an exact-input numerical comparison. Separate fixed-input logits and fixed256-ID continuation evidence cover equality; no token-equality claim is made from these benchmark files.

The earlier baseline attempt ended after warm-up without a result JSON. Its log and command are retained as `interrupted-base.*`; it is not counted as a completed measurement. The new sequence uses separate labels and files.

Still pending: a controlled repeat of the1536-token draft graph, another available packed-model setup, and required27B/35B/Flash-Next fallback model checks. The paired HTTP evidence in the separate continuation folder remains48PASS/1same keepalive-timingFAIL per arm. This native sequence does not waive those limits or make the PR Ready.
