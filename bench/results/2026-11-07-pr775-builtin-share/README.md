# PR #775: built-in-share fast default

This implements review [6698446394](https://github.com/nace-ai/kapitsa/pull/775#issuecomment-6698446394), option (a): the new 2048-channel fast default is limited to non-matrix Macs with a built-in ANE share (M4 today). Calibration-only chips, including M1/M2/M3, stay on the GPU for these new widths even if `NI_ANE_UNITS` or a calibration file supplies a share. The measured M3 prefill exceedance is the reason for this exclusion; the adverse measurements remain unchanged in their original folders.

The shared `ane_builtin_share` predicate is also used by `default_units`, preserving the old M4/M5 family test and every share value. Matrix GPUs remain excluded from the new default. `NI_ANE=1/0` still explicitly overrides the profile. Legacy 2560-channel inputs (4B/27B) keep their previous policy. Loading, memory planning and the SSD-cache namespace still use the same input policy.

`docs/profiles.md` now has a final Members row for this class. The fallback paragraph and the Nemotron candidate verdict are restored verbatim from the rebase base; this PR changes neither the cross-model fallback nor another feature's admission. The 9B bar rows and all M3 failure records are retained.

## Verification at 92689bea

- The injected CPU matrix covers M1, M2, M3 Ultra, M4 Pro, M4 Max, M5 Pro and M5 Max: four profiles, four `NI_ANE` states and two share-override states, **224 cases**. M3 stays off under fast even with share 16, while explicit `NI_ANE=1` remains effective. Device-table tests additionally cover base M4 and ANE availability. CPU geometry remains 50 checks.
- M4 Pro, 20 GPU cores, 48 GiB, macOS 15.7.1: fresh fast dump with no units override or calibration file, same model hashes and eight windows as the previous run. All **584/584 rows are bitwise identical** to `50fadecc` (192 step, 192 batch, 192 verify and eight prefill); 576 ANE evaluations, zero reruns. This is a new output check, not a new speed run. The prior M4 same-Mac MLX bar therefore has identical inputs and outcomes.
- The existing isolated M4 loaded-HTTP ABAB at `50fadecc` remains **61.94% lower mean TTFT**, 8.597692 -> 3.272273 s, built-in 24/24 share, MTP off. The reviewer explicitly allowed retaining this timing when M4 share selection remains unchanged. No timing is relabeled as 92689bea.
- The first full CPU run passed 16 checks (including the service/CLI and APFS clone checks), then the machine ran out of disk space during the update test. The five affected checks—update, PR-size harness, logit comparison harness, sampled-token harness and structured CLI—each passed on their first targeted rerun after freeing redundant storage. All 21 required checks are covered, but this was not a single uninterrupted green run. Lint, profile/comment guards and documentation links also passed. Initial failures and the targeted reruns are both retained; no assertion changed.

No M3 model/GPU run was performed for this policy-only narrowing. Its injected facts exercise the same pure predicate the live device read uses. The obsolete queued M3 timing controller was stopped while still waiting for another task's GPU lock; it started no timing leg and no unrelated process was interrupted.

The long-prompt diagnostic remains pinned to `9cca6b0d`: BF16 3/4, precise 1/4, fast 1/4, with the large prompt-position KL tails preserved. It is a disclosure, not an equivalence claim. Zero drift between GPU and W8A8 still fails. The new check above compares old fast against unchanged M4 fast, not fast against precise.

## Disk failure and raw data

The initial update installer reported `No space left on device`; subsequent checks could not create fixtures or logs, and the controller could not write its final status. The retained `run/status.json` therefore still says `cpu`; it is not a live process. `retry/status.json` records completion. Original failure logs and `disk-failure.json` document this.

Twelve existing workspace matrices were byte-verified by SHA256 against immutable previous matrices and replaced with hardlinks to the identical bytes. All original paths and contents remain available; only duplicate allocation was removed. The failed update test's own abandoned temporary install tree was removed after verifying no process used it. No other task's files or processes were changed. Future matrix dumps must use new output labels.
