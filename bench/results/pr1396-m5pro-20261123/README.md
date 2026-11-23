# PR1396: our local M5 Pro run

**Technical HOLD at `8eb6db9b44940f26d693d913c92f09e89d340229`.** Generated text matches at both requested lengths. The recorded timing ratios pass the arithmetic check, but missing isolation observations prevent an accepted speed verdict. These are our measurements, not a reproduction claim for the contributor's other Macs.

## Measured

Runs: Apple M5 Pro, Mac17,8, 48 GiB RAM, macOS 26.5.2 (25F84), 23 November 2026. Native NVFP4 Inkling-2-Mini; identical internally staged weights and metadata for both arms. Both binaries are Release builds with `-O3`.

- A: main `37f6f2fb1cd89541cc04019e63b5790332b87037`.
- B: PR1396 `8eb6db9b44940f26d693d913c92f09e89d340229`.
- Explicit controls: `--context 65536 --slots 1 --profile fast --dflash none --spec 0 --disk-cache 0`. The 65,536 value is capacity, not the prompt length.
- Nonstreaming `/api/chat`, temperature 0, top-k 1, top-p 1, seed 0, 256 generated tokens. Each arm starts its own server and receives the same short warmup before its measured request.
- A B A B separately at actual 4,096 and 32,768 prompt tokens. Both the count endpoint and each generation response confirm the requested prompt count. Server logs report zero cached prompt tokens.
- Every completion has nonempty thinking text and an empty final-content field, with 256 tokens and length termination. The output check hashes canonical UTF-8 JSON containing **both** fields. It does not establish a completed answer or logit identity.

| Actual prompt tokens | A1 tok/s | B1 tok/s | A2 tok/s | B2 tok/s |
|---:|---:|---:|---:|---:|
| 4,096 | 39.2183 | 39.3536 | 39.3849 | 39.6037 |
| 32,768 | 29.5707 | 30.2680 | 31.2826 | 31.2194 |

| Actual prompt tokens | Main mean | PR mean | PR/main | Raw arithmetic |
|---:|---:|---:|---:|---|
| 4,096 | 39.3016 | 39.4786 | 1.004505 | passes 0.98 |
| 32,768 | 30.4267 | 30.7437 | 1.010420 | passes 0.98 |

The counter definition is `(eval_count - 1) * 1e9 / eval_duration_ns`: 255 post-prefill tokens divided by the backend decode duration. These small positive differences are **qualified timing observations**, not an accepted speedup.

All four text hashes match within each length:

```text
4096:  20f35afc847a08a7874d264070e7d5e75c65fb1db9643f857ea58e395d6d0818
32768: bf9a56d068e036d11c317b76f2983303fafa9f5542314b26917a3c06d02aea5b
```

## Why the hold remains

The runner omitted start/end load, foreign-process CPU monitoring, and the DSH host's 15-minute startup guard. The exclusive GPU lock and the marker that paused declared CPU-heavy jobs do not supply these missing interval observations. Similar rates do not repair that gap. This is **missing evidence of isolation**, not evidence that contention occurred.

The independent audit also found weaker encoded memory thresholds and an unsafe, unexercised deadline-stop path that could stop the outer lock holder before cleanup. These are defects in our measurement harness, not findings against PR1396. No observed memory-limit crossing, orphan, or premature lock release is claimed.

The [raw summary](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/ab-run-02/summary.json>)'s `pass: true` describes arithmetic and text checks only. It is not a technical GO. The [admissibility record](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/admissibility.json>) and [independent audit](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/independent-audit.md>) qualify it. A future accepted timing campaign needs corrected guards and a new GPU booking; none is queued under this result.

## Preserved evidence and limits

All eight raw successful responses, launch arguments/environment, resource snapshots, records, fixtures, binary identities and final cleanup remain preserved. The fixed request bodies remain in the local evidence bundle; the public fixture manifest records their SHA-256 values. Two earlier harness launch failures are also retained: an immutable-lock opener error before model code, followed by eight TIME_WAIT port-probe errors before any timed arm started. The socket fix passed both new regression tests; the recorded CPU harness suite passed 13 tests.

The completed campaign returned zero. Its cleanup receipt confirms no private listener; the deadline-stop branch was not exercised. Storage cleanup allowed internal staging without lowering the 100 GiB disk floor. The staged target's SHA-256 matched the original.

Not checked here: logits/noise-floor comparison, precise profile, other model families or chip classes, maximum-context saturation, simultaneous slots, or vision. This fixed local run does not turn the contributor's one-sided M5 Max observations into paired A/B evidence. PR1397 is excluded.

The review lane owns the harness correction and any separately booked admissible timing run. The contributor owns missing broader default evidence. Ordinary exact-head CI, current-main CI, thread resolution and fresh integration gates remain required before any merge.
