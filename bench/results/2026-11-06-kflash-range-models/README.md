# Additional range-scan model validation

Runs: MacBook Pro (16-inch, 2024), Apple M4 Pro (14 CPU / 20 GPU cores), 48 GiB,
macOS 15.7.1 (24G231), Apple Clang 17, CMake Release -O3 / deployment target 14.
Reference d65a0afa; candidate implementation 279ce236. Actual binaries remain
0.2.31 (d65a0afa) / 0.2.31 (279ce236). Later branch commits add evidence only.
identity.json records build binary hashes and the tree heads captured before the run.
The previous two evidence folders are unchanged. CPU/edge/actual-drafter validation,
recorded-condition CPU helper timings and 4B HTTP/logit/native results remain there.

## Outputs

Qwen3.5 9B 4-bit and Qwen3.8 27B 4-bit: each compares 584 rows, all byte-identical
and finite (8 prefill, 192 batch, 192 step, 192 verify). The complete standing
logit_compare output, its JSON and the original metadata are retained. Default
floors were used; no --reorder or floor edits. Both candidates used --from the same
model's reference dump, then raw distributions were removed after reports were saved.
commands.json records the actual commands and the deliberate 9B model symlink overlay;
all underlying model files were unchanged. The 27B folder's existing drafter was used.

The fixed 4B control uses continuation.ids (an already published chat-templated
counting prompt), ni_run MODEL @continuation.ids 1024 0 --dflash auto, under the GPU
flock with NI_LOCK_HELD=1. Each arm has a new KAPITSA_HOME. No forced ignore-EOS,
low-rank, rounding, ANE or production-default override was used. All four A/B/A/B
runs produce 1,024 tokens, zero EOS and nonempty decoded text, with complete token-ID
equality. The recorded drafter lines identify v1, 4-bit, six layers/hidden2560 and
the same actual checkpoint already hashed in the previous evidence folder.
ni_run reports 1,023 *timed* decode tokens because its first output came from prefill.
All four have 129 cycles and 7.930 tokens/cycle. Decode rates A429.4/B449.6/A459.6/B453.3
tok/s and drafter load A543/B486/A541/B493 ms are diagnostic observations: thermal/power
were captured but foreign-GPU telemetry was not, so they are not an idle-GPU speed claim.
This fixed counting prompt does not prove every workload is unchanged.

## Native benchmarks

kapitsa bench --quick --json produced all rows below, each with three sampled/greedy
repetitions and per-stage statistics in the native JSON. Default model execution,
isolated KAPITSA_HOME, native GPU flock and native quiet wait. Every run is retained.

| Run | Sampled tok/s | Greedy tok/s | 4K TTFT s | Follow-up s | Peak GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| bench-9b-range-01-base | 127.10 | 135.81 | 8.6599 | 0.0749 | 8.80 |
| bench-9b-range-02-range | 124.24 | 134.94 | 8.6496 | 0.0816 | 8.79 |
| bench-9b-range-03-base | 118.76 | 131.23 | 8.6556 | 0.0762 | 8.80 |
| bench-9b-range-04-range | 129.21 | 134.58 | 8.6437 | 0.0762 | 8.79 |
| bench-27b-range-01-base | 59.93 | 63.52 | 10.6305 | 0.2918 | 19.18 |
| bench-27b-range-02-range | 58.01 | 66.98 | 10.5585 | 0.2235 | 19.18 |
| bench-27b-range-03-base | 67.55 | 66.49 | 10.6242 | 0.2383 | 19.18 |
| bench-27b-range-04-range | 60.79 | 65.72 | 10.5354 | 0.2302 | 19.18 |

bench-limitations.json retains every condition and flag. The 9B first baseline has a
sampled repetition warning. The 27B first baseline has sustained foreign GPU activity
(mean14.44%, max76.88%, Chrome/ChatGPT/Lark), so it cannot establish an idle comparison.
Other runs also record desktop GPU activity. These 9B/27B rows are reported as limited
model diagnostics, not used to advertise inference speed or a universal no-slowdown claim.
The earlier 4B native series is also retained, including its lower candidate greedy rate;
the fixed-input control above did not reproduce that directional slowdown for its prompt.

## Scope and reproduction

The measured benefit remains the 45–49% reduction in the existing CPU quantizer's
full-matrix function time on the actual 4B/9B BF16 matrices, recorded independently
with nominal thermal state/AC power after other test jobs ended. It is not an inference
speedup. This change adds no model/device gate, kernel, strategy or user option.

Use the full Release builds and the exact commands in commands.json and bench-commands.json,
substituting your own model/evidence paths. Build ni_run explicitly; it needs an outer
project GPU lock, unlike logit_dump and kapitsa bench which take it internally. Compile
decode.cpp against the same library to decode generated IDs with the unchanged tokenizer.
CPU regression/build/prek logs are in ../2026-11-06-kflash-range-followup/.

Not verified: an actual Flash-Next checkpoint, a 35B DFlash checkpoint, other Macs, and
stable whole-model throughput improvement. Those checks are not inferred from the helper
timings or another PR's vision results. This submission is ready for review with these
explicit limits; it does not assert that every model/hardware combination was tested.
