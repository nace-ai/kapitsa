# Small-row RMSNorm + X8 preparation on M4 Pro

Runs: Apple M4 Pro, 20 GPU cores, 48 GiB, macOS 15.7.1 (24G231), AC power. Model: `qwen3.6-35b-a3b-oq4e-mtp`, mixed 4/5/6/8-bit weights, its embedded native MoE MTP head.

Base: `16e2e826b330144f8e8a0f3880c3fe3e33c1024a`. Candidate: the source in this change. Both Release builds report the base commit; `results.json` records the final source and binary SHA256 identities.

## Scope and method

The fallback of `enc_qmvr_multi` previously dispatched `rows_norm_f32`, then prepared X8 operands for its first generic projection. The fused kernel keeps the original 256-thread scalar sum and `tg_sum`, then uses the same X8 half conversion, group sums and table stores. Each 256-column tile repeats the norm reduction. Normalized fp32 rows remain available for later projections; their own preparation and all GEMM kernels stay as before.

Default scope: 35B-A3B geometry on the 20-core M4 Pro, width 2048, one or two rows, with a normalized first projection taking the generic X8 fallback. `NI_NORM_X8=0` restores separate preparation.

Generation uses the stock `ni_run`, not `kapitsa bench`: `ni_run MODEL @prompt.ids 1024 0 --spec 1 --dflash none`. The prompts are code-context copies ending at the assistant header. They contain 1020 and 4143 input tokens; the original benchmark inputs were not edited. The first generated token is excluded: 1023 of 1024 tokens are timed.

Every leg takes the GPU lock, waits for eight quiet seconds below 2% foreign GPU activity, and samples contention every 0.25 s. Any sample above 2% invalidates its entire four-leg group. WindowServer is excluded by the repository monitor. Accepted groups have nonempty decoded text, 1024 output tokens, zero end tokens, identical generated IDs within each workload, and no new swapouts. The accepted 1K group temporarily hid ChatGPT and paused iTerm; both were restored. Later attempts paused additional interfering UI processes; every surviving paused process was resumed.

## Generation

| Workload / group | Order | Decode token/s, every run | Valid for timing |
|---|---|---|---|
| final-bench-1k | ABAB | A: 117.0, B: 118.6, A: 117.0, B: 118.3 | yes |
| final-bench-4k | BABA | B: 106.1, A: 105.0, B: 106.0, A: 104.5 | no |
| final-bench-4k-clean | BABA | B: 99.3, A: 104.4, B: 104.2, A: 103.6 | no |
| final-bench-4k-paused | BABA | B: 104.9, A: 103.5, B: 107.5, A: 108.4 | no |

Accepted group medians:

| Prompt | Original token/s | Fused token/s | Change |
|---|---:|---:|---:|
| 1K | 117.00 | 118.45 | +1.24% |

All three 4K groups are excluded: the first and last had Codex GPU activity, and the middle group had Lark / browser activity. Every recorded leg is retained in `results.json`; no 4K throughput gain is claimed. The 1K workload took 513 cycles (histogram `3 510`, reported acceptance 0.99); 4K took 558 (`93 465`, 0.83). These are two code prompts, not broad workload coverage.

## Isolated preparation

32 AB/BA pairs per shape after warmup; GPU time includes RMSNorm and X8 preparation, not GEMM or generation. The production baseline retains normalized output directly, without an extra copy.

| K | R | Original median us | Fused median us |
|---:|---:|---:|---:|
| 2048 | 1 | 5.855 | 3.657 |
| 2048 | 2 | 5.813 | 4.687 |
| 2560 | 1 | 6.315 | 3.863 |
| 2560 | 2 | 5.685 | 4.757 |
| 4096 | 1 | 6.885 | 4.152 |
| 4096 | 2 | 7.483 | 6.356 |
| 5120 | 1 | 8.491 | 4.965 |
| 5120 | 2 | 8.013 | 6.999 |

All 256 pairs favored fusion. Only width 2048 is enabled by default. Raw pairs are in `results.json`.

## Output checks

- `norm_x8_test small`: 65/65 with `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`; `check`: 225/225. Bytewise tables, sums, normalized rows, low/transposed tables and guard bytes, plus an independent fp64 norm reference.
- Standard `logit_dump` of original and final builds: 35B oQ4e, 4B and 27B, eight windows each. Per model: 8 prefill rows, 192 step, 192 verify and 192 batch rows; every row bitwise identical. Reports are in `results.json`.
- Comparison command: `tools/logit_compare.py REF NEW --min-top1 1 --max-mean-kl 0 --max-p99-kl 0 --near-tie 0 --max-far-flips 0 --max-nll-change 0`. The oQ4e pack has no calibrated floor here; strict zero-drift bounds were used instead of borrowing another pack's floor. No floor was changed.
- `norm_x8_model_check`: R=1 (24 rows) and R=2 (48 rows), fixed prefixes; trunk logits and MTP hidden rows all bitwise identical. Profiling confirms 81 fused calls per pass in each candidate and zero in the disabled arm. These correctness runs are not timing evidence.
- Native-MTP server: `--ctx 8192 --slots 2 --dflash none --spec 1`, full `tests/test_server.py`: ALL PASSED. An earlier one-slot run failed three cache-retention assertions; the rerun with two slots passed those assertions.
- Release builds, the AGENTS.md CPU suite, real-tokenizer structured/parser checks, and all-file `prek` passed. Python checks needing numpy/jsonschema were rerun successfully in an isolated environment after missing-dependency failures.

## Reproduce and limits

`results.json` includes both prompt ID arrays, per-run rates/GPU-cycle times, output hashes/end-token counts, contention and swap counters, binary/source identities, and MTP shard hashes. Write a prompt array as comma-separated IDs, then run the command above against each build in the recorded order. The operator and small-row check commands are documented in CONTRIBUTING.md.

Not measured: official `kapitsa bench` metrics, DFlash2 drafting, native MTP depths above one, long contexts, other Macs, or performance on other models. The 4B and 27B runs are output regressions for the shared X8 preparation body. Prototype screens on an earlier base are retained locally and are not used for this change's headline numbers.
