# PR #775: rebase onto 3f3b3cef

Rebased the 19-commit branch at `9102e1d4` onto upstream `3f3b3cef4b2e81138e072c86c8448247abfc21a4`, preserving each original author and subject. The only textual conflict was in `docs/profiles.md`; both upstream's Nemotron floor-failure history and the stricter new 9B fallback were retained. Upstream added `Knob` to `DefaultRow`. The automatically merged ANE initializer failed to build; the independent `50fadecc932863bff4d3f81a7a20d3913be3e15b` fix adds `Knob::kNone`, as for the existing descriptive fast-profile row. It does not change ANE dispatch, input geometry, arithmetic or profile behavior.

The fresh runs below use production code at `50fadecc`. The final source edits clarify descriptive versus runtime `default_on` rows and update displayed evidence percentages/citations; they do not change dispatch or arithmetic. The three previous evidence folders remain unchanged. Long-reasoning/BF16 results remain pinned to `9cca6b0d`; they are not relabeled as new-head measurements.

## Output check

Same 9B 4-bit model, fixed eight windows and full-vocabulary teacher-forced rows as the prior reports. Rehash model files, run the GPU segment checker under both Metal validation layers, then dump precise and fast. Compare each profile to its prior output with explicit zero bounds. Compare new precise to the retained native MLX/llama.cpp dumps and apply the same faster-engine bar separately to every path, including the extra eight prefill rows. Native engine speed selection and dumps are reused, not rerun or called fresh measurements.

| Mac / profile | Rows identical to 9cca6b0d | Current bar |
| --- | ---: | --- |
| M4 Pro precise | 584/584 | reference path |
| M4 Pro fast | 584/584 | all four paths pass |
| M3 Ultra precise | 584/584 | reference path |
| M3 Ultra fast, explicit share 16 | 584/584 | 576 decode comparisons pass; extra eight prefill rows fail |

M3 prefill mean KL remains 0.0001845747910965 versus the MLX bar 0.0001471647341223. Both have zero top-1 flips. The M3 comparisons used Python's scalar implementation this time; rounding differs from the earlier NumPy report by about 1e-15, not a changed dump. M4 uses NumPy as before. Model hashes matched on both Macs, and both GPU segment checks passed 2098 comparisons with validation enabled.

The M3 calibrated run is not the M3 shipping default. Explicit `NI_ANE_UNITS=16` remains required. The M4 run has no units override or calibration file and uses its built-in share. M5/matrix GPUs remain excluded from this new fast default. The previously observed M3 prefill exceedance is not an all-path admission pass.

## Interleaved performance

Fresh precise/fast/precise/fast HTTP runs on the M4 Pro; same deterministic 4135-token uncached request, no draft or disk cache, exclusive GPU lock and contention telemetry. This is the HTTP harness, not a community CLI benchmark submission.

| M4 Pro leg | Profile | TTFT, seconds |
| --- | --- | ---: |
| A1 | precise | 8.598488667 |
| B1 | fast | 3.290835875 |
| A2 | precise | 8.596896166 |
| B2 | fast | 3.253709750 |

Mean precise 8.597692416 -> fast 3.272272813 seconds: **61.94% lower TTFT**. All request hashes match, all four legs are quiet, and invalid/flag lists are empty. Two-second telemetry yielded four samples on each precise leg and two on each fast leg. The fast logs confirm the built-in 24/24 share, no units override.

An earlier complete ABAB is also retained under `performance-io-overlap/`. Evidence hashing/copying ran during that attempt; it is not used for the headline. The full ABAB above was repeated with evidence work idle, without selecting legs.

Fresh M3 ABAB has not completed: another task holds its GPU lock. The prior **23.63%** result remains explicitly pinned to `9cca6b0d` in [the M3 report](../2026-11-07-pr775-m3-current/README.md); it is not a 50fadecc timing. The new M3 output check above is complete. No unrelated lock holder was interrupted.

## Validation and retained failures

The full `vk cpu-tests` run passed 20 of 21 checks. `test_server_cli` failed one assertion: after deleting an APFS clone, deleting the remaining copy reported no space freed instead of MB freed. Its entire log remains included. A first targeted reproduction with fresh files passed both assertions on the candidate; the same targeted reproduction also passed on a separately built clean `3f3b3cef` CLI. The three relevant source/test files are byte-identical to upstream. This does not establish the cause of the intermittent full-suite failure, and it is not a claim that the full suite passed. No test or production deletion behavior was changed. Device-profile and 32-case ANE policy checks, profile/comment guards and logit-dump output-directory tests passed. Final lint/docs checks are recorded separately.

The first post-rebase build failed on the missing `Knob` initializer; its output and the dependent controller failures are retained. The subsequent local source rebuild used the cached CMake version string `9cca6b0d` even though the source was `50fadecc`. Its output was bitwise identical to the old run, but the version discrepancy is explicitly recorded and the full run was repeated after configuring CMake and asserting the new binary version. No old result or metadata was rewritten. All eight raw matrices remain at their original local paths. Each SHA256 matches its immutable pre-rebase matrix, so content-verified hardlinks remove redundant storage. An incomplete remote copy is not used as evidence.
