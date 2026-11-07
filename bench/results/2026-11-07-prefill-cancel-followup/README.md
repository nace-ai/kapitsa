# Prefill cancellation follow-up on M3 Ultra

## What changed and why

A client disconnect during continuation prefill discarded already completed checkpoints. Retrying the recorded 15K-token continuation then read the entire prompt again.

After pending GPU work finishes, restore the deepest surviving checkpoint within the reused prefix and its paired drafter state. Discard checkpoints written by the unfinished prefill, trim token/image/KV-provenance metadata, and keep the empty-slot fallback when no completed checkpoint survives. The regression cancels twice, compares against the same uninterrupted continuation, and checks a cold cancellation after a new system checkpoint.

## Default for

All native engine models with a surviving completed checkpoint in the reused prefix, with no new switch or hardware gate. Live-only prefixes without a checkpoint retain the discard fallback. This preserves grown KV capacity until ordinary eviction or memory pressure; it allocates no extra checkpoint and changes no serialized state layout.

The dense Qwen3.5 4B and MoE Qwen3.6 35B-A3B setups below ran the interleaved A/B and output gates. Other architecture/drafter combinations are listed under Not verified.

## Measured

Runs: Apple M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1 (25E253), Apple clang 17, CMake Release `-O3`. Reference `2c41b56cc6c5f0895d4359d5ff33d1459b22bc39`; candidate engine `3ce432031569e364af6731ffb59aec1bc67355ef`. Both were explicitly configured and built; CLI versions and source/binary hashes are in the evidence. The corrected regression is test-only commit `d1b3c89154ecceb5e6ff1c792d7d4934bfc94c31`, SHA256 `d4c0c7f2c407208cfc670f31f4d1e6b4ff398bf14804dd0f897dccb6f7e53f58`, run from a separate test snapshot against those unchanged binaries.

Models: `qwen3.5-4b-4bit` (dense, folder's DFlash2 drafter) and `Qwen3.6-35B-A3B-4bit` (MoE, plain decoding). Cancellation flags: one slot, context 32768, SSD cache off, `NI_ANE=0`, temperature 0, max_tokens 32, temporary homes and exclusive GPU lock. Each arm includes its own uninterrupted reference request. GPU quiet checks passed before each arm; monitoring flagged none, while preserving brief loginwindow samples in the raw data.

| Cancellation A/B/A/B | Retry cached tokens | Client retry seconds | Server prefill ms | Checks pass/fail |
| --- | ---: | ---: | ---: | ---: |
| 4b-0-base | 0 | 7.769 | 7615.1 | 8/1 |
| 4b-1-candidate | 2051 | 6.877 | 6720.1 | 9/0 |
| 4b-2-base | 0 | 7.768 | 7614.8 | 8/1 |
| 4b-3-candidate | 2051 | 6.868 | 6716.1 | 9/0 |
| 35b-0-base | 0 | 6.560 | 6412.2 | 8/1 |
| 35b-1-candidate | 2051 | 5.835 | 5685.8 | 9/0 |
| 35b-2-base | 0 | 6.566 | 6418.1 | 8/1 |
| 35b-3-candidate | 2051 | 5.830 | 5681.6 | 9/0 |

Every baseline fails only prefix retention; every candidate passes all nine checks. Candidate retry text/reasoning/finish reason matches its uninterrupted reference and subsequent repeat. Actual replies are 32 tokens on 4B and 22 on 35B. Cold-cancel retries reuse zero tokens. Retry time is about 11.5% lower on 4B and 11.1% lower on 35B in these two pairs; this is a cancellation/retry workload result, not a general inference speedup. Server-confirmed cancellation durations and process RSS samples are retained; RSS is not an isolated Metal/KV allocation measurement.

Native baseline checks used `kapitsa bench --model MODEL --quick --json` with its normal default environment handling and fresh homes, interleaved A/B/A/B under the GPU lock:

| Native run | Greedy tok/s | Sampled tok/s | 4K TTFT s | Peak GB | Flags |
| --- | ---: | ---: | ---: | ---: | --- |
| 4b-bench-0-base | 378.62 | 349.19 | 1.9370 | 4.55 | none |
| 4b-bench-1-candidate | 379.47 | 381.59 | 1.9365 | 4.55 | none |
| 4b-bench-2-base | 367.16 | 351.98 | 1.9352 | 4.55 | none |
| 4b-bench-3-candidate | 347.31 | 351.59 | 1.9358 | 4.53 | decode: repetition loop (distinct-4-gram 0.4763) |
| 35b-bench-0-base | 186.39 | 183.89 | 1.4860 | 19.45 | none |
| 35b-bench-1-candidate | 189.78 | 188.80 | 1.4876 | 19.45 | none |
| 35b-bench-2-base | 188.01 | 187.13 | 1.4852 | 19.45 | decode: repetition loop (distinct-4-gram 0.4735) |
| 35b-bench-3-candidate | 189.41 | 189.43 | 1.4886 | 19.45 | none |

All eight native runs exited 0 with `invalid: []` and no recorded contention. Retain the slow 4B candidate B2 and both repetition flags. There is no uniform no-slowdown or overall inference speedup claim from these variable rows. Full conditions and all three per-test measurements are saved in each JSON.

Output gate: each model has 584 finite, bitwise-identical rows: prefill 8, batch 192, step 192, verify 192; total 1,168 rows. No floor was changed. Four interleaved 256-token continuations per model have identical token IDs, nonempty decoded text and zero early EOS. These continuations use MTP (`--dflash none --spec 2`) on 4B and plain `--spec 0` on 35B.

<details>
<summary>Verbatim logit comparison reports</summary>

```text
prefill: 8 records (qwen3.5-4b-4bit)
  reference: 0.2.31 (2c41b56), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (3ce4320), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
batch: 192 records (qwen3.5-4b-4bit)
  reference: 0.2.31 (2c41b56), batch {"members":4,"rows":2}, prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (3ce4320), batch {"members":4,"rows":2}, prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
  PASS (tools/logit_floors/qwen3.5-4b-4bit.json: the decode passes' gate (the prefill rows are bitwise the same))
step: 192 records (qwen3.5-4b-4bit)
  reference: 0.2.31 (2c41b56), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (3ce4320), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
  PASS (tools/logit_floors/qwen3.5-4b-4bit.json: the decode passes' gate (the prefill rows are bitwise the same))
verify: 192 records (qwen3.5-4b-4bit)
  reference: 0.2.31 (2c41b56), verify "KFlash tree (a chain of 8 live rows)", prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (3ce4320), verify "KFlash tree (a chain of 8 live rows)", prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
  PASS (tools/logit_floors/qwen3.5-4b-4bit.json: the decode passes' gate (the prefill rows are bitwise the same))
prefill: 8 records (Qwen3.6-35B-A3B-4bit)
  reference: 0.2.31 (2c41b56), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (3ce4320), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
batch: 192 records (Qwen3.6-35B-A3B-4bit)
  reference: 0.2.31 (2c41b56), batch {"members":4,"rows":2}, prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (3ce4320), batch {"members":4,"rows":2}, prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
  PASS (tools/logit_floors/Qwen3.6-35B-A3B-4bit.json: the decode passes' gate (the prefill rows are bitwise the same))
step: 192 records (Qwen3.6-35B-A3B-4bit)
  reference: 0.2.31 (2c41b56), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (3ce4320), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
  PASS (tools/logit_floors/Qwen3.6-35B-A3B-4bit.json: the decode passes' gate (the prefill rows are bitwise the same))
verify: 192 records (Qwen3.6-35B-A3B-4bit)
  reference: 0.2.31 (2c41b56), verify "KFlash tree (a chain of 8 live rows)", prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (3ce4320), verify "KFlash tree (a chain of 8 live rows)", prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
  PASS (tools/logit_floors/Qwen3.6-35B-A3B-4bit.json: the decode passes' gate (the prefill rows are bitwise the same))
```

</details>

## Checked

- Own Release builds passed. Initial CPU sweep at implementation `926bce8b` passed 20/21 suites; the updater watcher observed an empty version during atomic replacement. The original failure is retained and unexplained. One isolated complete rerun passed 148 requests. This follow-up ran complete instrumented updater suites on both explicit reference/candidate builds, passing 140 reference and 148 candidate requests, without reproducing the diagnostic. This does not turn the initial sweep into a clean pass.
- Nineteen supplementary CPU checks plus the separate packaging patch-application check passed. Both 4B/35B tokenizers passed parser 125, grammar 308 and structured CLI 48 checks each. These previously published source-specific logs remain unchanged.
- Full HTTP: 35B **49 pass/0 fail** on both builds; 4B **47 pass/the same 2 failures** on both (short-versus-long concurrency completion order and busy-slot keepalive/timing). No all-HTTP-pass claim.
- 4B MTP cancellation: actual response usage reports MTP, 11 cycles; baseline retry cache 0 versus candidate 2051, with identical nonempty returned text.
- Two-slot memory pressure, context 16,384, budget 4.9 GB, prefill chunk 1,024: candidate's retained 2,051-token slot is actually dropped to make room for another 13,236-token context; retry is cold and returned text equals baseline. This checks ordinary eviction of retained capacity, not OS memory-pressure notification.
- File-cache restart on a dedicated 14 GiB HFS+ RAM volume: both sources restore 2,051 text tokens or 2,117 tokens including an image; text matches, cached images are not reencoded. Earlier state was already saved by the baseline, so this is restore compatibility, not a candidate-only persistence benefit. It does not measure physical SSD latency/durability.
- 4B image cancellation with file caching disabled: baseline retry cache 0 versus candidate 2,117; identical nonempty returned text, and the candidate reports the image in the cached prefix with zero encoded images.
- Final `prek run --all-files` output is recorded in `prek-final.log`. No Metal kernel changed.

Preserved unsuccessful attempts: the original 35B regression compared the long retry's 2,051-row checkpoint with the short request's 2,081 live rows (7 pass/1 fail). The uninterrupted long request itself reused 2,051; the test now compares that matching continuation and requires a positive control cache. An initial 4.7 GB memory fixture requested 15,223 tokens after the engine advertised 14,214 and never reached cancellation. On the physical SSD, only 3.9 GiB was free, so the unchanged 10 GiB reserve gave a zero cache quota; text/image restart retention assertions failed. A real temporary RAM volume supplied space for the file-cache checks. The earlier port/TIME_WAIT and comment-gate attempts remain in the first bundle.

Evidence: [initial frozen bundle](https://github.com/chainhelen/kapitsa/tree/experiment/prefill-cancel-checkpoint/bench/results/2026-11-07-prefill-cancel), [new follow-up results, every arm, commands, failures, reports and SHA256 manifest](https://github.com/chainhelen/kapitsa/tree/experiment/prefill-cancel-checkpoint/bench/results/2026-11-07-prefill-cancel-followup). Existing 171 data files are unchanged.

## Not verified

- Other chips; ANE split, LiLiCorr/DSpark, Dory rings, distributed split models, alternate backends and contexts beyond the recorded 15–16K cancellation prompts.
- Physical SSD restart under sufficient free space, crash/power-loss recovery, idle write-behind races, simultaneous cancellation across active slots and OS-level memory pressure. The RAM-volume checks exercise actual file serialization/reopen but do not substitute for physical-storage performance or durability evidence.
- Hosted fork CI requires maintainer approval; it has not been self-approved. The updater's original empty-version cause remains unresolved, and the two shared 4B HTTP failures remain failures.
