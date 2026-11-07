# Affine MTP sidecar: M3 Ultra / 27B follow-up

Runs: Apple M3 Ultra, 60 GPU cores, 96 GiB; macOS 26.4.1 (25E253), Apple clang 17, Release `-O3`; Qwen3.8 27B 4-bit with its real affine MTP sidecar. Reference `b171f98363abd5dfd8ae5d24f786ca77b3801a4a`; production `8e12c3d09ac4118775755000d75f5fdbe6cbca35`. Binaries report `kapitsa 0.2.31 (b171f98)` / `(8e12c3d)`. Both complete builds were explicitly configured per revision; this addition changes no source or earlier evidence.

The same 20-file model used for the M4 comparison was copied to the task's temporary 26 GiB RAM volume. The queue verified every size and SHA256 before running. This is a resident-storage comparison, not an SSD-cold load measurement. `remote-27b-queue.json`, `remote-27b.json` and `ram-model-setup.json` retain the identities. All 74 collected result files were checked against remote SHA256s; the compiled tokenizer helper is omitted here, with its build command retained.

## Output and API checks

- 584 finite bitwise-identical logit rows: 8 prefill and 192 each batch/step/verify. Zero top-1 flips, KL and gate failures. Candidate dumps use reference prefixes. Existing standing floor files are unchanged; they were not remeasured on this M3. Both builds report no ANE execution. This brings this PR's four setup comparisons to 2,336 rows, including the earlier M3/4B, M3/35B and M4/27B results.
- Four fixed-input A/B/A/B continuations force MTP with `--dflash none --spec 2`. Each produces 256 nonempty token IDs, zero EOS tokens and SHA256 `bf9bd80862daa6d72d4dd83f0341dc9f7f0b60de161ee9f28397a35bf0f84b6b`. Every arm loads the real dense MTP head, reports 96 cycles and 2.66 tokens/cycle. Rates are 73.1 / 75.6 / 75.1 / 75.4 tok/s. Recorded foreign and ambient GPU activity are zero in these monitors.
- Paired HTTP tests each report **48 PASS + 1 FAIL**, exit 1. Both fail the existing busy-slot keepalive timing assertion: no keepalives, first token at 0.263523 s / 0.262741 s. These are not passing suites. Full logs and server arguments are retained.

## Native measurements

`kapitsa bench --quick --json`, with separate servers and scratch homes, under `/tmp/kapitsa-gpu.lock`. Flags: no algorithm overrides; `NI_LOCK_HELD=1` communicates the caller's lock; auto-update and prompts disabled. Native arms discover DFlash2/KFlash in the model folder; they do not measure the forced-MTP path above. Sampled prompts have fresh nonces.

Ordered A/B/A/B (base/candidate/base/candidate), all three trials per arm:

| Arm | Sampled trials, tok/s | Greedy trials, tok/s | Median sampled / greedy | 4K TTFT, s | Follow-up median, s | Peak memory, GB |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 27b-bench-0-base | 131.72, 149.17, 128.78 | 146.04, 145.32, 144.71 | 131.72 / 145.32 | 11.9950 | 0.1368 | 19.39 |
| 27b-bench-1-candidate | 126.52, 142.66, 135.29 | 143.18, 144.73, 144.8 | 135.29 / 144.73 | 12.0000 | 0.1531 | 19.39 |
| 27b-bench-2-base | 117.34, 130.08, 143.12 | 141.66, 143.15, 144.71 | 130.08 / 143.15 | 12.0014 | 0.1371 | 19.40 |
| 27b-bench-3-candidate | 124.6, 141.27, 137.22 | 146.13, 144.95, 146.55 | 137.22 / 146.13 | 11.9978 | 0.1378 | 19.39 |

All four commands exit 0 with empty `invalid` and `flags`, AC power, nominal thermals, zero swapouts, zero recorded foreign GPU load and no contention entries. Each corrected quiet preflight checks the boolean tuple field and passes. The helper reports processes above 0.5%, so requesting 0.1% does not prove complete absence of background activity. No inference speedup or uniform no-slowdown is claimed. These clean M3 arms complete the pending 27B comparison; they do not erase earlier M4 contention or repetition flags.

## Evidence and limits

`results/STATE.json` records every command, source/binary hash, model metadata, outcome and condition. `results/27b-compare.log` contains the exact per-path output report. `results/*.command.json`, HTTP logs, continuation IDs/text and native JSONs are unmodified records. Large logit payloads are omitted; their comparison reports and dump metadata remain. `manifest.json` records all files in this addition except itself.

Earlier directories `2026-11-07-mtp-affine-shapes` and `2026-11-07-mtp-affine-gpu` are unchanged. Their CPU cache-residency failures, paired HTTP failures, original 14 native arms and contended M4 repeat remain disclosed. No source change required repeating completed build/CPU checks; final all-files lint was run for this evidence addition. A real stacked-affine expert sidecar, longer-context quality, M5 tensor kernels and other model families remain untested. Synthetic loader fixtures establish their own admission/conversion coverage only. No kernel changed.
