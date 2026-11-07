# PR #775: CI wiring and non-matrix fast default

This revision addresses review comment [6690600497](https://github.com/nace-ai/kapitsa/pull/775#issuecomment-6690600497), items 3, 4 and 6. It does not establish fast-profile numerical admission.

## Change

- CI builds `ane_segment_check`, runs its CPU geometry checks without `--gpu`, and runs `tests/test_ane_profile.py`.
- New 2048-channel input shapes default to fast only on non-matrix Macs. Matrix/wide devices, including M5, retain the GPU default for these shapes. Explicit `NI_ANE=1` still overrides; `NI_ANE=0` disables either profile. Legacy 2560-channel geometry (4B/27B) is unchanged.
- Load, memory planning and cache namespace continue to share `input_profile_enabled`. Precise and explicit overrides avoid a live device read; the new fast default reads the existing centralized `matrix_units` predicate.
- The device row describes the non-matrix ANE class with runtime share checks: M4 built-in shares or calibration on other non-matrix Macs. M3 results are calibrated, not built-in defaults.
- Usage, profile candidates, numerical-history introduction and contributor test instructions are synchronized. The candidate row links the measured M3 integration instead of using the historical M4 63% headline. Existing results are preserved.

## Source identity and verification scope

Implementation commit: `9be7668ba00e43d60180390da9e229904eade5d8`. CPU-only validation on the authorized remote M3 Ultra. No model, GPU kernel comparison, timing or quality evaluation was run for this revision. These checks do not establish M5 runtime correctness or current-head M4 performance.

The remote checkout started at `4a9cd5082019033b3c3abd41488f2e8b04a5593f` and applied `source.patch`. Its SHA256 is `595fb897239878d6da1d9b53c1ee8d3fb5e455bdd8237f7454039fba8abaf8b1`, independently matched against the local diff. The build's version string retains the parent hash because validation preceded the commit. Source comments were synchronized after the first build; `build-final.log` records the final incremental rebuild and `lint-final.log` the final lint run.

- Release CLI/CPU-target build and extra segment/device-profile targets passed.
- Geometry: 50 checks, 0 failures.
- Policy: 32 fresh processes across injected non-matrix/matrix capability, unset/precise/fast/invalid profile and unset/empty/0/1 NI_ANE. No live GPU detection in injected cases. Device-independent disabled cases also assert zero ANE memory/units in the real memory planner; the injected M5-only exclusion checks policy, not a real M5 memory plan.
- Device profile: injected M3/M4 match with ANE; M5 matrix and wide classes do not; devices without ANE do not match.
- Lint and profile/comment gates passed.
- Complete clean-environment `vk cpu-tests`: 21 named tests passed, 0 failed.
- PR size: 536 counted lines (cap 1000), before this excluded results directory.

## Still outstanding

- Current-head M4 Pro precise/fast ABAB and full-vocabulary output checks, without `NI_ANE_UNITS`.
- A 9B fast bar against the measured faster of MLX/llama.cpp on the same Mac and the same eight windows/576 continuation rows. Both top-1 flips and mean KL must meet that bar; no bar or Members row has been invented.
- ANE-exposed reasoning prompts over 512 tokens against bf16, including per-position KL tails. The earlier 49/50 unexposed prompts and one adverse exposed example remain disclosed.
- Approval and successful completion of the fork CI workflow on this head. Local checks are not a GitHub CI pass.

The earlier model runs remain pinned to their original source commits in [the fast integration report](../2026-11-07-pr775-fast-profile/README.md) and [the bf16/math report](../2026-11-06-m3ultra-9b-ane/README.md).
