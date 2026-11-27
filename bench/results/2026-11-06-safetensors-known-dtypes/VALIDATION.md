# Selected dtype rejection: completed validation

Implementation `61fdbafb`, baseline `96d2de68`. Initial Draft records in this folder are preserved.

- `remote-validation.json`: M3 Ultra 60-GPU-core / 96-GiB / macOS 26.4.1, exact source and test hashes, explicitly configured Release builds, controlled probe, regression linked against the baseline, full CPU run and isolated updater retry, 4B logits and HTTP outcomes.
- `local-validation.json`: M4 Pro 48-GiB / macOS 15.7.1, full Release `load_error_test` runs. Candidate `df6f609a` contains the tested implementation plus initial evidence; the loader hash equals the remote implementation's.
- `bench-*.json`: all four interleaved 4B benchmark runs, including flags and condition samples.

The new dtype and filter cases pass; the new regression fails against the baseline. All 584 4B logit rows are bitwise identical. The full candidate loading test passed locally. The remote CPU suite's single updater failure passed on isolated retry; this is not a claim of a clean first full run.

The pre-existing page-cache residency assertion was not consistently passing: remote baseline 3 → 1474 and 3 → 897 resident pages out of 4420; remote candidate 3 → 515. Local baseline also failed at 3 → 897; the local candidate passed with 897 → 897. These figures are not evidence of a cache improvement. The new dtype/filter cases passed before the remote assertion, and the full candidate `load_error_test` passed locally. The changed code does not alter payload I/O. All failure records are retained.

Both builds passed 47 HTTP tests and failed the same two existing timing assertions: the long counting reply ended at 693 tokens before the delayed side request, and the slot holder ended before the second 15-second keepalive. This is not an all-pass HTTP claim.

| Run | Sampled tok/s | Greedy tok/s | 4K TTFT s | Follow-up s | Flags |
|---|---:|---:|---:|---:|---|
| bench-01-base | 338.46 | 371.44 | 1.9380 | 0.0320 | [] |
| bench-02-candidate | 350.61 | 375.67 | 1.9373 | 0.0309 | [] |
| bench-03-base | 322.98 | 362.69 | 1.9377 | 0.0313 | [] |
| bench-04-candidate | 338.53 | 372.53 | 1.9400 | 0.0332 | [] |

All benchmark flag and invalid lists are empty. The sampled benchmark has no fixed seed. No throughput gain is claimed; fixed-input logits provide the numerical comparison.

Other model families and other GPUs were not tested. The existing residency assertion's root cause remains unresolved; this PR changes dtype admission, not payload I/O. No speedup or page-cache improvement is claimed.

To rerun, explicitly configure each source revision (`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`), build `load_error_test`, `kapitsa` and `logit_dump`, and run `build/load_error_test` in its default CPU mode. Run `vk cpu-tests`, the supplemental targets from AGENTS.md and `prek run --all-files`. `build/logit_dump MODEL NEW_DIR`, the candidate with `--from BASE_DIR`, and `tools/logit_compare.py` provide the output check; they hold the GPU lock. Use `vk serve MODEL` for `tests/test_server.py URL`, then `vk cleanup`. Run the four `kapitsa bench --model MODEL --quick --json --out NEW_FILE` arms sequentially; `bench` holds the lock itself. Keep every raw result, including failures and flags, under new filenames.
