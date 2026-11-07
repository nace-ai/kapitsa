# BF16 conversion follow-up

Runs: M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1 (25E253), Clang 17 Release -O3.
Candidate source: `5e174cb89eb991a705752d033f7f9da8edfeff87`.
Reference: `bbd84e0d53be75866087aa615bdeb6f2411f393f`.
CMake was configured explicitly with those eight-character version labels before the new checks.

## Outputs and CPU controls

- Qwen 3.6 35B-A3B 4-bit: 584/584 finite rows bitwise identical (8 prefill, 192 step, 192 verify, 192 batch).
- Qwen 3.5 4B 4-bit with `NI_ANE=1 NI_ANE_UNITS=1`: 584/584 finite rows bitwise identical. Each arm actually ran 576 ANE evaluations across 10 chunks; zero reruns. The startup check stayed enabled. The M3 has no default calibrated ANE share, so this is an explicit coverage run.
- Both model tokenizer/parser and structured CLI checks passed on the candidate.
- Previously published candidate build, 21 CPU entries, 19 supplemental checks and exhaustive NaN/finite conversion tests remain applicable: production code has not changed. Final all-files prek passed again.

## Fixed-input diagnostics

Each A/B/A/B leg starts a fresh process and scratch home, consumes identical 41-token chat input, and returns 256 nonempty tokens with no end token. All IDs match within each model. These are ni_run diagnostics, not the native benchmark protocol or an inference speedup claim.

| Mode | A1 tok/s | B1 tok/s | A2 tok/s | B2 tok/s |
| --- | ---: | ---: | ---: | ---: |
| 4B plain | 197.8 | 200.8 | 198.1 | 197.6 |
| 4B KFlash/DFlash2 | 405.7 | 396.6 | 394.5 | 403.7 |
| 35B plain | 187.3 | 185.2 | 187.5 | 187.1 |

The 4B speculative arms all take 66 cycles, 3.864 tokens per cycle, the same histogram and acceptance by position. Their candidate difference is smaller than the same-build spread. The earlier approximately 10% candidate deficit was not reproduced by this fixed-input diagnostic; this does not establish performance equivalence on every workload. The earlier native A/B/A/B (355.06 / 334.36 / 372.60 / 328.42 tok/s), varying prompts/acceptance and repetition flags remain in this archive.

## HTTP timing diagnosis

The earlier two-slot default 4B runs each failed the same two timing assertions. The long response completed in approximately 0.9 s, before the delayed side request was sent at 1.0 s. Busy-slot holders completed around 17.2–17.3 s, before the second keepalive due at 30 s. Those timing preconditions were absent in both arms; the original failures remain failures.

With `--slots 2 --ctx 32768 --disk-cache 0 --dflash none --spec 0`, the unmodified HTTP suite passes all 49 assertions on both reference and candidate, including side-request ordering and both 15-second keepalives. Candidate 35B with default drafting also passes all 49 assertions. This separate control does not relabel the original default 4B suite as all-pass.

## Evidence and limits

`evidence.tar.gz` contains unmodified logs, exact commands, version/binary metadata, output comparison reports, IDs/text, timings and the previous native measurements. `SHA256.json` hashes every archive member. The original multi-candidate coordination state is retained as provenance; only the BF16 pair's results support this PR. GPU legs held the exclusive GPU lock and reported no foreign GPU contention. Raw float buffers were hashed and removed after comparison, as the verification workflow directs. Existing committed results were not edited.

No malformed neural model was loaded, and these checks do not claim that a network containing NaNs generates useful output. Other models and chip classes were not exercised by this follow-up. No overall inference speedup is claimed.
