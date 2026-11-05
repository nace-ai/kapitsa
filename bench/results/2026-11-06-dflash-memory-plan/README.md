# DFlash memory-plan regression

Reference `39f20c8d`, candidate `30320cab`. Later evidence-only commits change no production source. Same 4B DFlash v1 drafter on both Macs; its SHA256 is recorded in `evidence.json` and was checked after transfer.

Build `dflash_plan_test` and run its CPU mode. For the negative control, compile that same test source against the reference library. The reference fails the new low-rank CPU assertions. For real allocations, run `NI_DFLASH_LOWRANK=0/768 build/dflash_plan_test --gpu MODEL` under the GPU lock. The tool loads the target first, then compares the drafter's Metal allocation delta with its planned persistent allocation, removing first-prefill and the target's 4 MiB allowance. It permits 8 MiB for page rounding and existing small scratch estimates. It does not measure transient CPU factorization/load memory.

M3 Ultra / rank 768: actual 580,780,032 bytes, old plan 535,595,008, new plan 579,897,728. The old regression fails; the new remaining 882,304-byte gap is within the existing buffer-rounding allowance. Rank 0 passes on both versions.

M4 Pro / default packed path, rank 0: actual 1,026,555,904 bytes, old plan 938,487,808 (fails), new plan 1,082,441,728 (passes). Rank 768 passes on both versions because the old reservation for the full packed head happened to cover the low-rank replacement. The new bound is conservative: it assumes all raw coefficient slabs and keeps room for factorization failure followed by the full packed-head fallback.

The target logit gate is 584/584 rows bitwise. The real rank-768 drafter intermediate check passes all 270 cases / 35,198,376 bytes. Four rank-768 generation legs produce 1,024 matching nonempty tokens, no EOS and 7.930 tokens/cycle. A separate default `kapitsa bench --quick` ABAB check is retained; the command clears NI_* and therefore does not enable rank 768 on M3. The baseline third reference run flags sampled repetition loops; all runs have no hardware/contention flags. No speedup is claimed.

The full HTTP suite has 47 passes and the same two timing-only failures on reference and candidate (the counting reply/holder end before the test's scheduled side request/second keepalive). A larger context/max-token run reproduces both failures. These are retained, not called passes. Other models, base-M4 default hardware and a forced factorization-failure experiment were not tested.

Complete `vk cpu-tests`, supplemental AGENTS.md CPU tools and `prek run --all-files` passed on the remote candidate. The logs are included in `evidence.json`.
