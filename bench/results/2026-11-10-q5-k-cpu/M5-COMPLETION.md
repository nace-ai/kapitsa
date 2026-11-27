# Completed M5 model validation

This additive record completes the pending model work described in the earlier frozen CPU records. These are redacted derived results, not raw logs; the measured source and binary identities in `M5-MODEL-VALIDATION.json` are unchanged.

Runs: M5 Pro (20 GPU cores, 48 GiB), macOS 26.6.2, Apple clang 17.0.0 (clang-1700.0.13.5). Reference `aea6d0973b13fa22a47643cdf24defab8d953afb`; candidate `340eb56dac4c8dc95d064046a05aa070a827cb51`. Both were explicitly configured and built in Release, then independently imported the complete pinned Airy GGUF identified in the JSON. Its Q5_K token embeddings exercise this decoder before repacking. All model phases held the exclusive GPU lock.

Both imports produced eight literally identical files. Each arm produced 584 rows (145,018,880 fp32 values) across prefill, step, teacher-forced verify and batch. Every value is finite; complete outputs and recorded inputs were compared literally and match. Report-only comparison and the separate strict zero-drift gate both pass. No noise floor was changed.

Native `kapitsa bench --quick --json` ran A/B/A/B. All four exit 1; none has a valid decode or greedy median. All reasoning, repetition, early-stop and cached-empty flags and invalid results remain in the JSON. These runs do not qualify inference performance.

The fixed-request A/B/A/B has six matching request/content SHA pairs across all four legs. Completion counts are 512, 4, 227, 512, 512 and 512; only requests 1 and 4 return nonempty content. Every row has invalid reasoning output; early termination, empty content and repetition remain disclosed. Hash agreement is not a literal continuation comparison, and empty content is not meaningful continuation equivalence. Contention flags are retained as booleans without private telemetry; flagged legs are not idle-GPU performance evidence.

HTTP: reference 37 PASS / 17 FAIL; candidate 38 PASS / 16 FAIL. Every candidate failure also occurs on the unmodified reference. The reference-only failure is `thinking: finish=stop + usage`. Failed assertion names are retained; this is not an all-pass claim or a comparison of complete HTTP response bytes.

The M3/M5 CPU decoder measurements remain separate evidence of CPU decoding speed. This supplement makes no whole-import or inference speedup claim. Complete-model execution on M3, other chip variants, other model families, ASan, non-AArch64 execution and free-generating MTP remain unverified. No Metal kernel changed.
