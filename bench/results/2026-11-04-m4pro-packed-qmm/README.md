# Packed Q4 QMM on M4 Pro 20-core / 48 GiB

Enable the existing packed-Q4 path for Qwen3.5 4B on this additional device profile. The production change is the
device/default predicate in `Qwen35::v8_packed_coeff_on`; the Metal kernels, coefficient encoding, K partitions and
format checks are unchanged. The M4 10-core / 24 GiB default and the `=0` kill switch remain available.

Runs: MacBook Pro (14-inch, 2024), Mac16,8, Apple M4 Pro (14 CPU cores / 20 GPU cores), 48 GiB, macOS 26.4.1 (25E253),
AC power, automatic power mode. CMake 4.4.4, Apple clang 21.0.0, Release (`-O3`). Both binaries report Kapitsa 0.2.26.

- A: `e87d0f878fe5f6eeeaed36cd85659d63d8f0981a`, the main branch fetched before the change.
- B: `6c0617fc30d5ba79a37a41f42e325a87511fe08d`, the proposed default with the environment variable unset.
- Target: `lmstudio-community/Qwen3.5-4B-MLX-4bit`, revision `c43ee1d65576a5d98de1e8405cac93c371a655c1`, Q4/group64.
- Drafter: `z-lab/Qwen3.5-4B-DFlash`, revision `9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf`, six-layer v1, Q4 at load,
  reduced head 98,304 rows. No MTP sidecar was installed.
- Binary/source hashes: [baseline.json](baseline.json), [candidate.json](candidate.json). Model, tokenizer and drafter
  hashes: [model.json](model.json), rechecked unchanged after performance measurements.

## Controlled inference results

One private local server per arm, one slot, 8K capacity, SSD prefix cache disabled, seed 42, 512 generated tokens,
EOS honored, thinking disabled. Short and 4K prompts contain 82 and 4132 tokens. All sixteen responses finish with
`length`, contain nonempty text and report zero cached tokens. Text is identical across A/B within each
temperature/context. The [prompts](prompts/) were frozen before these comparisons.

Each arm waits at least 45 seconds without our GPU work and at least 15 continuous seconds in
`NSProcessInfoThermalStateNominal`, then warms a separate 2.9K prompt and a separate 512-token coding response.
The existing GPU quiet gate runs before timing. Thermal state is recorded before and after each request; every
controlled request is nominal at both endpoints. Builds and profiling are outside the timing windows.

Whole-decode throughput is `(completion_tokens - 1) / (last content-event time - first content-event time)`, from the
unchanged `bench/harness/bench.py::decode_rates`. The same helper's middle-window estimates, TTFT, complete text,
event times, VM counters and contention samples are in the raw [greedy](greedy/) and [sampled](sampled/) records.

| Temperature | Context | A1 tok/s | B1 tok/s | A2 tok/s | B2 tok/s | Use |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | short | 183.178 | 188.691 | 183.945 | 189.168 | **+2.92%**, all checks pass |
| 0 | 4K | 146.767 | 150.649 | 148.545 | 150.952 | Diagnostic: A1 had 16 swap-in pages |
| 0.6 | short | 169.344 | 173.073 | 169.522 | 173.365 | Diagnostic: distinct-4 = 0.4883 |
| 0.6 | 4K | 167.776 | 174.761 | 168.376 | 171.269 | **+2.94%**, all checks pass |

The default is supported by short greedy decode (183.562 -> 188.930 tok/s, +2.92%) and 4K sampled decode
(168.076 -> 173.015 tok/s, +2.94%). Observed same-build ranges divided by their median are 0.418% / 0.253% for short
greedy A/B, and 0.357% / 2.019% for 4K sampled A/B. Both pairs improve in each accepted case. These are two
observations per binary/case, not a broad confidence interval or a claim about every prompt.

The greedy 4K series includes a 16-page swap-in in A1 and does not establish a 4K greedy speedup. The sampled short
responses trigger the existing harness's distinct-4 repetition threshold (0.4883 < 0.5) in every arm. That same check
was applied offline to every saved response and recorded in [analysis.json](analysis.json) without altering the raw
records. Those two series are retained as diagnostics and do not support the headline.

Packing allocates 2150.9 MiB for 249 target projections and 343.9 MiB for 19 drafter projections: **2494.8 MiB,
about 2.44 GiB additional static buffers**. Original weights stay available for other paths. This allocation and
load-time packing are not included in the already-loaded server's decode throughput.

This establishes compatibility and a measured gain on the 20-core GPU; it does not establish optimal tiling for it.
NF1 reduces the accumulators and reduction scratch per SIMD group but also reduces input reuse across output
channels. Packing changes access order and coefficient representation, not the matrix multiplication's FLOP count.
The benefit can therefore change with memory/compute balance and scheduling. The earlier M4 measurements in #896
used different prompts and macOS, so their larger percentage cannot be attributed to core count alone without a
controlled per-shape comparison and hardware counters.

## Native benchmark series retained

These six runs were produced by `kapitsa bench --quick --json`, A1 B1 A2 B2 A3 B3, before the controlled comparisons.
Each arm has three sampled writing runs and three greedy writing runs. Every individual rate remains in the
unmodified [native](native/) JSON.

| Arm | Sampled median tok/s | Greedy median tok/s | 4K prefill tok/s | Follow-up TTFT s | Peak GB | Thermal | Native flags |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A1 | 191.69 | 192.92 | 1973.1 | 0.0582 | 4.27 | nominal | none |
| B1 | 193.78 | 210.95 | 1935.9 | 0.0574 | 6.84 | nominal | none |
| A2 | 192.83 | 198.82 | 1944.5 | 0.0562 | 4.27 | nominal | sampled repetition: distinct-4 0.4841 |
| B2 | 182.39 | 209.01 | 1959.4 | 0.0561 | 6.83 | nominal | none |
| A3 | 103.85 | 130.74 | 1040.4 | 0.0591 | 4.27 | fair | none |
| B3 | 135.04 | 150.72 | 1409.1 | 0.0552 | 6.83 | fair | none |

A3 and B3 entered `fair` thermal state. The same baseline's greedy median fell from 192.92/198.82 to 130.74 tok/s,
and its 4K prefill nearly halved. The native harness does not flag `fair` as throttling, so its empty flags must not
be read as stable conditions. A2 also has a sampled-response repetition flag. This series alone cannot establish a
default; it motivated the recorded cooldown control and fixed-input/fixed-seed comparisons. No native run was
replaced. The [initial protocol](speed-protocol.json), [sampling follow-up](sampled-followup-protocol.json) and
[thermal control](thermal-followup-protocol.json) preserve those decisions.

## Correctness and API checks

- The existing CPU codec check passes all 267,378 constructed coefficient pairs.
- Small GPU checks with both `MTL_SHADER_VALIDATION=1` and `MTL_DEBUG_LAYER=1`: 1152 codec, 1600 ordinary, 384 segment
  and 80 special-scale cases pass. The packed kernel uses 2048 bytes of static threadgroup memory.
- The full-shape suite checks 14,134,272 GPU outputs, all bitwise equal to the original kernel. Independent FP64
  samples have maximum normalized error 4.87891285e-7, below the unchanged 1e-5 limit. Original/fallback and candidate
  dispatch counts are retained in [checks](checks/).
- Default B and B with `NI_V8_PACKED_COEFF_NF1=0`: each matches all 584 target logit rows against A bitwise
  (8 prefill, 192 step, 192 verify, 192 batch). The existing floor file is unchanged, with no `--reorder`:
  [default gate](checks/logits-default.txt), [disabled gate](checks/logits-disabled.txt).
- Short and 4K draft checks: 270 tensors / 35,198,376 bytes each, zero mismatches, using normal and debug passes on
  three anchors: [short](checks/draft-short.txt), [4K](checks/draft-4k.txt).
- The isolated-dispatch profiles show 142 calls per cycle to `qmm8snb_b4_nf2` in A and `qmm8pc_b4_nf1` in B. These
  profiles are dispatch-coverage diagnostics; their timing fields are not inference performance evidence.
- Supplemental CLI traces at temperatures 0 and 0.6 on both frozen prompts produce 512 nonempty-output token IDs
  per run, identical between A/B. All eight runs contain zero IDs 248044 or 248046, with all 512 tokens before any
  end token. [Token IDs and checks](checks/token-traces.json) and the `tokens-*.txt` cycle logs are retained;
  instrumented CLI timing fields are not used for the speed verdict. [Decoded text checks](checks/token-text-validation.json)
  confirm nonempty output before the A/B ID comparison. These compare builds within the CLI's own prefill schedule;
  they do not assert that the CLI and the server's default Neural Engine prefill produce the same 4K continuation.
- All 19 `vk cpu-tests` driver commands and 18 supplemental CPU commands pass, including the real tokenizer/parser
  checks; `prek run --all-files` passes.
- The unmodified API suite with two slots and 32K context gives **48 passes and one identical keepalive-test failure
  in both A and B**. Unread holder streams stop at about 23 seconds, before the second heartbeat; the HTTP library
  has a five-second write timeout. A supplemental probe continuously drains both holders and retains the original
  heartbeat timing bounds. Both builds pass: heartbeats at about 15.06 and 30.12 seconds, first token after release
  at about 30.3 seconds. The stock failures and the [drained-holder probe](checks/keepalive-drained.json) are retained;
  no server or API test source was changed.

## Reproduce

Use the pinned model and local drafter above on the same device. Build both revisions in separate checkouts with
Release flags and the same compiler. The numerical checks use the targets `kapitsa logit_dump qmm_packed_coeff_check
qmm_draft_check ni_run`.

```sh
# OLD and NEW are the build directories; MODEL is the unchanged model folder.
# DATA is this results directory. All output directories below must be new.
MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 "$NEW/qmm_packed_coeff_check" check-small
"$OLD/logit_dump" "$MODEL" /tmp/qmm-pro-old
"$NEW/logit_dump" "$MODEL" /tmp/qmm-pro-new --from /tmp/qmm-pro-old
python3 tools/logit_compare.py /tmp/qmm-pro-old /tmp/qmm-pro-new
NI_PROFILE=1 "$OLD/qmm_draft_check" "$MODEL" "$DATA/prompts/short.ids" /tmp/qmm-pro-draft-old
NI_PROFILE=1 "$NEW/qmm_draft_check" "$MODEL" "$DATA/prompts/short.ids" /tmp/qmm-pro-draft-new --from /tmp/qmm-pro-draft-old
```

Each checker takes the project GPU lock itself. The controlled-run source snapshots are in [harness](harness/).
See [harness/README.md](harness/README.md) for the original directory layout and locked invocation. The auxiliary
records here are not `bench/harness` schema-2 run records or community submissions and are therefore not rows in the
generated `bench/README.md` table.

## Not verified

Other Macs, GPU core counts and RAM capacities; other macOS versions; other models and Q2/Q3 or other quantization
formats; contexts beyond the measured 4K prompts; a sustained-load speed claim under thermal pressure. The generic
path classifier also names the 27B and 35B; those broader-model GPU runs are outside this 4B-only device change and
were not run. The original M4 default was preserved in source, not remeasured on an M4.
