# PR #775 rebase validation on main 7381db4

Rebased candidate: `eeb432ec853835ed32c610c58cae3ed23cc754a9`.
Base: `7381db446cbaf6f74f1578dd040fe66d3c613642`.
Previous branch head: `2b80fadc2451bf6bb56718d470b957f55b156c7e`.
Machine: same M3 Ultra 60 GPU cores / 96 GB / macOS 26.4.1 as the preceding report.

Only docs/usage.md conflicted: retain main's updated default-policy wording for
NI_QSA_FUSE/NI_QSA_ROWFUSE while replaying the PR's historical ANE wording edits.
The 13 existing commits and their authorship remain separate. All committers use
chainhelen <chainhelen@gmail.com>. ANE source, segment checker and abort harness
are unchanged from the previous branch head. Main also changed device predicates
and loader code, so old model measurements are not relabeled as current-head runs.

Validation on the rebased candidate:

- Release build and vk cpu-tests: all 21 named tests passed, zero failures.
- device_profile_test, dflash_row_test and affine_pack_test passed.
- New comment-gate check, profile-gate check and lint passed.
- Metal validation layers enabled: 50 CPU segment checks, 2098 GPU comparisons,
  34,105,984 data/guard bytes, zero failures.
- Official ANE abort/fallback harness: 17/17 passed, using the same external
  NI_ANE_UNITS=16 wrapper and model as the preceding run. GPU work held the lock.
- PR size check: 397 counted lines, under the 1000-line limit.

One supplementary test fails on BOTH candidate and a separately built clean
main 7381db4: load_error_test's page-cache assertion. Both report exactly
`page cache: 3 -> 515 of 4420 pages`, then `FAIL after <= before + pages / 10`.
Both preceding dtype and header validation groups pass. The PR does not modify
the loader or this test; both original failure logs are retained here. No test
threshold was adjusted. This is a disclosed baseline failure, not an all-green claim.

No full logit comparison, bf16 quality, Math500 or performance ABAB was repeated
for this rebase. Those results retain the original tested commit identifiers in
../2026-11-06-m3ultra-9b-ane/README.md. In particular, the numerical gate and the
exposed math regression remain unresolved; passing these compatibility checks
does not establish merge readiness.
