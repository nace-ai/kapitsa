# PR #775: proposed fast-profile integration

Candidate `3400e870cfe0cacccf69948fdc6385bc8e55a4f2`, reference main `508720651f35a3bcb6587543b20aa8da250c73a9`.
Kapitsa 0.2.31, Release builds; M3 Ultra, 60 GPU cores, 96 GB, macOS 26.4.1.
This report verifies the configuration change. It does **not** establish the fastest-other-engine
bar required for admission to fast (docs/profiles.md). No floor was changed.

## Behavior and output checks

New 2048-channel inputs (hidden 2048/4096/6144/8192) follow the fast profile;
precise keeps the GPU path. NI_ANE=1/0 overrides the profile. Existing 2560/5120
inputs retain their defaults. Model load, memory planning and disk cache identity
use one policy. Hardware/calibration, format, memory and startup guards remain.

Real model: lmstudio-community/Qwen3.5-9B-MLX-4bit at
b455506b0f574c74616dbcd56879bde38fafcff3, same verified weights as the preceding
M3 report. All arms explicitly set NI_ANE_UNITS=16, selected by the earlier
calibration; this does not add or measure shipping M3 default shares.
Chunk 2048, fixed eight windows, full-vocabulary logits, 8 prefill rows and 192
each of step/verify/batch. NI_NO_MTP=1, ANE async and keepalive disabled.

- Main precise vs candidate precise: all 584 rows bitwise identical.
- Candidate precise vs fast with NI_ANE=0: all 584 rows bitwise identical.
- Candidate fast vs precise with NI_ANE=1: all 584 rows bitwise identical.
- Fast: 576 ANE evaluations, zero reruns; precise and explicit-off: zero evaluations.
- Actual disk_format metadata matches within precise/off and fast/force, and differs
  between those groups. This checks namespace selection, not persisted SSD replay.

Fast vs precise intentionally fails explicit zero bounds: prefill 0/8 top-1 flips,
mean KL 1.846e-4; step 1/192, 2.579e-4; verify 1/192, 2.666e-4; batch 1/192,
2.591e-4. The continuation paths flip at the same sequence/prefix, not independent
examples. The failed comparison is included. W8A8 is not a reassociation exception.

## Loaded HTTP latency, same candidate binary

A=precise, B=fast. Two interleaved pairs, fresh server/home per leg, 4135 prompt
tokens, zero cached tokens, fixed seed 20261106, warmup then quiet wait, no drafter.
Compilation/startup is excluded. Requests, every run and telemetry are included.

| Leg | Profile | Seconds |
| --- | --- | ---: |
| A1 | precise | 3.546557666 |
| B1 | fast | 2.692346709 |
| A2 | precise | 3.546572875 |
| B2 | fast | 2.710552333 |

Mean 3.546565 -> 2.701450 seconds: **23.83% lower latency**.
All request hashes agree across legs. All harness invalid/flag lists are empty;
quiet checks pass and monitored foreign GPU use is zero. Limitation: 2-second
telemetry yields two samples for each precise leg and one per fast leg; the helper
excludes WindowServer and labels ambient daemons. This is not strict all-process
idle proof or a robust confidence interval. A/B replies differ. These are HTTP
harness results, not a community kapitsa bench submission or decode-throughput claim.

## Validation and retained failure

Release builds pass. The 16 fresh-process profile/override combinations pass,
including invalid/empty settings and disabled memory planning. Profile table,
comment-gate and profile-gate tests and lint pass. Metal validation layers:
50 CPU geometry checks, 2098 GPU comparisons, 34,105,984 data/guard bytes, zero
failures. The updated abort harness exercises fast with a calibrated-share wrapper:
17/17 pass, including explicit GPU control and fault replay.

The first CPU suite accidentally inherited NI_NO_MTP=1 from the model tests.
Its split_place_test expected MTP head bytes and failed (8 MB instead of 12 MB).
That run was stopped after the independent GPU/output tests had finished. Its
partial log and failure are retained. A complete rerun in a clean environment
passed all 21 named CPU checks (cpu-clean.log); no assertion or threshold changed.
The full log collection includes both runs.

## Remaining scope

Fast admission still needs the **same-machine, same-row 9B fastest-other-engine**
comparison: top-1 flips and mean full-vocabulary KL must both meet that bar.
Neither the 35B bar, the unsupported-model fallback nor bf16 quality is a substitute.
No such bar is claimed here. The M4 historical results retain their old commits;
this integration was run on M3 Ultra only. Whole-model 4B/27B output checks were
not repeated for this integration (their policy/packing geometry was checked);
other new widths and noncausal combinations lack model-level evidence.

The earlier bf16 and Math500 runs were not repeated or relabeled. In particular,
49/50 math prompts did not trigger ANE, and the sole exposed prompt changed from
correct to exhausting the 32768-token budget without an answer. See
[the preceding report](../2026-11-06-m3ultra-9b-ane/README.md). Moving to fast does
not erase this adverse example or establish reasoning equivalence.

The manifest records source/binary identities and hashes for the complete archived
log collection (archive-relative paths; this directory is a compact subset).
Full raw logit arrays remain on the remote machine; the delivered archive excludes
those arrays and model weights. Subsequent evidence/docs changes do not alter the
tested production source.
