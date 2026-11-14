# QMR coefficient cache: M4 / Qwen3.5-9B

Draft evidence: prefill, TTFT and complete-request improvements pass the local
performance targets. The predeclared decode uncertainty gate does **not** pass.
This result does not establish merge readiness or complete acceptance.

Runs: Mac mini M4, 10 GPU cores, 24 GiB RAM, macOS 26.5.2 (25F84),
Xcode 26.6 (17F113), SDK 26.5, AppleClang 21. Release `-O3`, C++20,
arm64, deployment target 14.0. Baseline: `61287d4403904b54f75fe1f15918766ed0b57316`.
The candidate was measured as that baseline plus this three-source-file patch.
The publication checkout preserves those source bytes and compiler flags.

Measured binary SHA256:

- A: `b7d63d63d5dc7d0c7782bd6dfd13edd0a2646975982f8faf9ff684155fa8f962`
- B: `7b01c3c4f7c3dbf464185f95d302b5dea0d254d4c98b3a39e362eb884aa6ec3f`

## Work and sampling

These are real HTTP streaming requests collected by the campaign harness;
`kapitsa bench` did not produce them. Each arm starts a fresh server/home,
loads the same local Qwen3.5-9B-4bit and attached DFlash2 model, runs a full
same-length 512-token warmup, then measures one resident-model request.
Both warm and main requests must have exactly 4096 or 16384 input tokens,
512 effective output tokens, and zero cached tokens. The frozen request JSON,
rendered token IDs and text are in `requests.tar.gz`.

Flags: one slot; `--context 20k --prefill-chunk 2048 --disk-cache 0 --dflash auto
--profile precise --idle-unload 0`; `NI_ANE=0`; temperature 0, top_k 1, top_p 1,
seed 42, thinking disabled. No Instruments, NI_PROFILE, output tracing or Metal
validation layers during acceptance timing. All model work holds the exclusive
GPU flock, passes both caller-held markers and the inherited FD, and cleans up
children before releasing the lock. Builds/downloads/heavy CPU work are excluded.

The original six-pair requirement was explicitly amended to three valid pairs
per scenario. Each pair uses A then B and the first valid complete attempt;
initial attempt plus at most three environmental retries. All failed attempts
remain in the reports and raw archives. Work/output failures are not environment
retries. No latency cutoff is used to discard a run.

Environmental gates include AC/normal thermal and memory pressure, stable power
mode, no new swapouts, foreign compute GPU mean <=1% and no three consecutive
samples >5%, other desktop GPU mean <=5% and no three samples >15%, no three
samples of one foreign CPU process >100% or aggregate >150%, and no foreign
build/download/benchmark process. The amended protocol records WindowServer
without invalidating on it; loginwindow remains in the foreign-compute category.
GPU percentages are accumulated process GPU-time deltas divided by wall time,
not percentages of physical GPU cores.

Formal06 and the separately frozen formal07 replication used identical candidate
bytes/settings. Formal07 was initiated after formal06's failed gate. Report both
separately; the combined six-pair sensitivity analysis is descriptive and is not
an independently preplanned confirmation or an optional-stopping correction.

## Complete requests

Positive reduction means faster. Ratios are paired geometric means. Each interval
is a 95% t interval of paired log(B/A), n=3, df=2. Every underlying sample, including
client whole-decode, is in `formal06.json` and `formal07.json`.

| Batch | Input | Metric | Reduction | 95% B/A |
|---|---:|---|---:|---|
| 06 | 4096 | TTFT | 3.838% | [0.961023, 0.962210] |
| 06 | 4096 | prefill | 3.834% | [0.961141, 0.962171] |
| 06 | 4096 | whole decode, engine | 0.224% | [0.993068, 1.002475] |
| 06 | 4096 | complete request | 2.474% | [0.973645, 0.976885] |
| 06 | 16384 | TTFT | 4.784% | [0.934332, 0.970327] |
| 06 | 16384 | prefill | 4.786% | [0.934312, 0.970319] |
| 06 | 16384 | whole decode, engine | **-1.269%** | [0.950109, 1.079403] |
| 06 | 16384 | complete request | 3.972% | [0.941015, 0.979941] |
| 07 | 4096 | TTFT | 3.474% | [0.950311, 0.980452] |
| 07 | 4096 | prefill | 3.471% | [0.950464, 0.980350] |
| 07 | 4096 | whole decode, engine | -0.177% | [0.989103, 1.014598] |
| 07 | 4096 | complete request | 2.094% | [0.967537, 0.990717] |
| 07 | 16384 | TTFT | 4.625% | [0.950058, 0.957455] |
| 07 | 16384 | prefill | 4.625% | [0.950048, 0.957467] |
| 07 | 16384 | whole decode, engine | -0.092% | [0.995672, 1.006195] |
| 07 | 16384 | complete request | 3.992% | [0.957374, 0.962785] |

TTFT and total require B/A <=0.98 and interval upper bound <1. Decode requires
the interval to contain 1 and its upper bound <=1.011191037 (4K) or <=1.001625944
(16K). These local runtime bands were frozen from three baseline measurements
before screening; they are not a repository-wide rule or a model logit floor.
The small calibration and three-pair acceptance samples limit the conclusion.

Formal06's 16K decode increase is dominated by one valid 11.227 ->11.711 second
pair. Background process activity nearby does not prove causation and did not
meet the frozen invalidation rule. That pair is retained. Formal07 and combined
sensitivity also fail the full decode criterion. A CI containing 1 establishes
neither a significant slowdown nor equivalence within the frozen band.

## Numerical and resource contract

Dense output is `Y[M,N] = X[M,K] * W[N,K]^T`, optionally added to the existing
FP32 Y after the complete K reduction. X is FP16. W is unsigned affine Q4,
group64, with BF16 scale and additive bias. Preserve the original
`half(float(S)*4096)`, `half(float(B))`, half dequantization FMA, FP32 accumulation,
and K32/f0..3 matrix-accumulate order. K is divisible by 64; original M/N tail
clamps, X zero fill, masks and strides remain. GU preserves two FP32 sums and
the `half(silu(g)*u)` epilogue.

The change transposes shared X and loads adjacent M entries as half2, with
2048 B static threadgroup storage instead of 2560 B. Dense uses
BM32/BN128/BK32/WM1/WN4, 128 threads; GU uses BM32/BN64/BK32, 128 threads.
Compiler metadata reports dense GPR 80 ->85 and GU 78 ->73, no spills;
these are not measured occupancy values.

Preconverted half2 coefficients are immutable after load. The 200 projection
caches total 412 MiB. They retain packed weights and the original BF16 tensors.
Cache, fallback scratch and retired scratch share a 512 MiB payload cap. The
constructor waits for conversion before publishing ready; teardown waits before
release. Unknown/sliced projections use ordered scratch packing. Model unload
resets the cache. No persisted KV/state format changes.

The current patch selects this path within the existing Qwen35 QMR defaults for
eligible Q4/group64 large-M dense/GU calls. No new hardware/model gate is added.
Only M4/9B has evidence. Other class members and a second setup have not been
validated; broader default eligibility remains unresolved.

## Real-shape controls and attribution

Observed 4K/16K requests have two/eight 2048-row chunks. The following seven-pair
controls compare the original compute path to the same candidate compute shader
with per-call coefficient packing included. They exclude host encoding and model
initialization; weighted sums are estimates, not additive request measurements.

| Use | M x N x K | Calls 4K /16K | Original weighted ms/4K | Compute candidate ms/4K |
|---|---|---|---:|---:|
| fused gate/up | 2048 x12288 x4096, two projections | 64 /256 | 7198.932 | 6916.925 |
| down | 2048 x4096 x12288 | 64 /256 | 3623.099 | 3474.448 |
| qkv /attention q | 2048 x8192 x4096 | 64 /256 | 2413.891 | 2316.654 |
| output | 2048 x4096 x4096 | 64 /256 | 1210.596 | 1162.054 |
| GDN z | 2048 x4096 x4096 | 48 /192 | 906.424 | 870.173 |
| k/v | 2048 x1024 x4096 | 32 /128 | 155.473 | 149.101 |

The compute control saves about 619 ms/4K; caching alone adds about 14.8 ms/4K
in a separate incremental microbenchmark. This agrees in scale with the real
prefill reduction. Code differences are confined to this prefill GEMM and its
coefficient helpers, and full target outputs match. Request measurements supply
the actual TTFT/total results above.

Two Instruments Metal Performance Limiters captures exercised the target shader
but exposed no usable GPU counter values and had sampling gaps. Cache, arithmetic,
register/occupancy and barrier bottlenecks remain hypotheses. The CPU-only Metal
doctor report and runtime pipeline-limit results are included in the raw controls.

A subsequent diagnostic 16K warm/main census found identical decode kernel/grid
counts and resource binding metadata; no QMR kernel or coefficient binding occurs
there. The 34 observed decode kernels have identical native archive sections and
pipeline limits. Indirect grids, actual residency, scheduling and clocks are not
established by that census. It is not acceptance timing and does not waive the gate.

## Validation and costs

Small dense and GU tests each compare 742,926 complete output elements with
`MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`. Full real-shape checks compare
213,909,504 FP32 dense and 150,994,944 half GU outputs, plus guards, padding,
nontrivial strides, add, both signs of actual coefficient extrema, fresh inputs,
grouped projections, and M=129/1919/2032 tails/suffixes. Short-M selection is preserved.

Strict request checks compare all live target-logit rows across the full 248320
vocabulary: 1137/1273 rows for 4K warm/main and 1321/1273 for 16K. All rows,
512 accepted logits, 512 actual output IDs and tree metadata match bitwise per
request. Standard `logit_dump` adds 584 bitwise-equal rows (8 prefill, 192 each
step/verify/batch). No floor adjustment, `--reorder`, or precision relaxation.
Instrumentation was confined to separate numerical/diagnostic runs.

Formal07 resident physical-footprint increases across all attempts are
411.17–413.50 MiB. Candidate median ready times are 13.199 s (4K), 12.800 s (16K),
versus 12.758 s for each baseline scenario. Candidate median lifecycle physical
peaks are 9647.16/9985.24 MiB, versus 9233.88/9572.71 MiB. Candidate initialization
wall times range 19.175–69.007 ms, median about 21 ms; GPU times 8.858–10.819 ms.
Allocation/conversion/wait are included before ready. All formal06/07 lifecycle
swapout deltas are zero. Timed memory pressure is normal; formal07 memorystatus
is 27–33. Raw records retain startup, warmup, resident, peak, RSS, swap and pressure;
initialization pressure has boundary snapshots rather than continuous sampling.

The publication checkout produces a byte-identical copy of measured binary B.
Its fresh `vk cpu-tests`, 14 supplemental C++ checks, profile/logit-dump harness
and tokenizer checks, and `prek run --all-files` all pass. An earlier full CPU
suite had an APFS last-copy accounting failure also reproduced on baseline;
that history is retained and was not reproduced in this publication check. The
server API suite passes 55 checks at its required two slots; performance and
strict requests use one slot.

## Raw evidence and reproduction

- `formal06.json`, `formal07.json`: every attempted pair, validity, metrics and CIs.
- `formal06-raw.tar.gz`, `formal07-raw.tar.gz`: unmodified run/lock records, server
  logs and complete timestamped warm/main SSE, including invalid attempts.
- `combined-sensitivity.json`: descriptive six-pair analysis with the failed gate.
- `requests.tar.gz`: exact input JSON/text/IDs, fixed before screening.
- `validation-raw.tar.gz`: full-output summaries, logit comparison, request-oracle
  hashes, CPU failure disposition and decode diagnostic comparisons.
- `compute-control-raw.tar.gz`: seven-pair control output, suffix checks, pipeline
  resources, model identity, build flags and CPU-only doctor output.
- `publication-cpu-raw.tar.gz`: fresh pre-push CPU/lint results, exact binary/source
  identity, previous-data/statistics audits and retained APFS baseline reproduction.
- `manifest.json`: archive and member SHA256, with original local evidence paths.

Build A from the baseline commit and B from this PR with the same Release flags.
To replay the frozen workload, extract `requests.tar.gz`, then run each server
under the exclusive GPU lock with the flags above and a fresh `KAPITSA_HOME`.
POST `warm-SIZE.json`, wait for completion and GPU quiet, then POST `main-SIZE.json`
to `/v1/chat/completions`. Timestamp SSE at the client; first content-bearing event
defines TTFT, `[DONE]` defines complete time, and usage/logs supply engine intervals.
The exact server argv/environment and timestamped events are in every raw run.
Retain all attempts and use the fixed three-pair/bounded-retry design above.
Simple curl wall-time alone is not equivalent to these TTFT measurements.

For the separate standard output check, hold the GPU lock, pass the caller-held
markers, and use `NI_ANE=0` with both `logit_dump` builds:

```sh
lockf -k -t 0 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 NI_ANE=0 \
  build-A/logit_dump MODEL_DIR NEW_REF_DIR
lockf -k -t 0 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 NI_ANE=0 \
  build-B/logit_dump MODEL_DIR NEW_CANDIDATE_DIR --from NEW_REF_DIR
python3 tools/logit_compare.py NEW_REF_DIR NEW_CANDIDATE_DIR
```

Model weights, executables, multi-GB logit tensors, personal plans and experiment
notes are not included. Full local evidence remains in the campaign directory
named by the manifest. Other chips, models, alternative sampling settings and
broader default eligibility are not verified.
