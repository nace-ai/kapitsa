# Paired-angle RoPE on current main, M4 Pro

Evidence layout superseded for review: the complete original records are preserved unchanged on [`archive/rope-full-evidence-20261031`](https://github.com/chainhelen/kapitsa/tree/743a78dd00b42ae0d752896446209e343531cf29/bench/results), including the older-baseline experiment, all CPU/server logs, every API attempt and dump metadata. This PR retains the current-baseline measurements and correctness reports needed to review and reproduce the change. No recorded measurement was edited.

Runs: MacBook Pro (16-inch, 2024), Apple M4 Pro, 20 GPU cores, 48 GB, macOS 15.7.1 (24G231), AC power, 2026-10-31.

A is upstream `15f8a57fb9a1d8775596cacfc4dc8264b03aee89`; B is `1b89569a0d21619f448ba902f200bcd2d365fe56`, Kapitsa 0.2.25. The original model-output and default-request measurements use those commits; the supplemental greedy check below names its separately rebuilt candidate. At measurement time the additional uncommitted changes only extended the test matrix, updated its reference comment, documented the tool and added its CI build target; [the exact patch](gpu/uncommitted.patch) and [kernel source hashes](manifest.json) are retained. The reference kernels match this upstream commit exactly after renaming.

## Change and operator measurements

`attn_prep` and `attn_prep_m` compute each rotary pair's precise pow/cos/sin once and share the two results. The existing row barrier also publishes the angle arrays. Rotation arithmetic and dispatches stay the same. Static threadgroup storage grows from 2,176 to 4,224 bytes. The default Qwen3.8-27B preparation path on this M4 Pro reaches the change. KFlash tree verification uses the separate, unchanged `attn_prep_tree` kernel.

Shape: D=256, rotary width=64, 24 query heads, 4 KV heads. Each invocation warms both arms four times with 64 dispatches, then measures six interleaved A/B pairs of 64 dispatches per shape. Times below are the median per-dispatch GPU time. Reduction is `1 - B/A`. Two independent invocations, all thermal states nominal:

| Round | Rows | Kernel | A median (µs) | B median (µs) | Time reduction |
|---|---:|---|---:|---:|---:|
| 1 | 1 | attn_prep | 11.690 | 11.694 | -0.03% |
| 1 | 1 | attn_prep_m | 3.245 | 3.243 | 0.09% |
| 1 | 8 | attn_prep | 4.709 | 4.457 | 5.36% |
| 1 | 8 | attn_prep_m | 4.958 | 4.796 | 3.26% |
| 1 | 128 | attn_prep | 39.549 | 36.108 | 8.70% |
| 1 | 128 | attn_prep_m | 40.899 | 36.966 | 9.62% |
| 1 | 512 | attn_prep | 150.232 | 137.906 | 8.20% |
| 1 | 512 | attn_prep_m | 153.964 | 139.438 | 9.43% |
| 2 | 1 | attn_prep | 3.159 | 3.105 | 1.70% |
| 2 | 1 | attn_prep_m | 3.172 | 3.175 | -0.10% |
| 2 | 8 | attn_prep | 4.801 | 4.619 | 3.80% |
| 2 | 8 | attn_prep_m | 4.859 | 4.842 | 0.35% |
| 2 | 128 | attn_prep | 39.423 | 35.780 | 9.24% |
| 2 | 128 | attn_prep_m | 40.996 | 37.139 | 9.41% |
| 2 | 512 | attn_prep | 149.989 | 136.668 | 8.88% |
| 2 | 512 | attn_prep_m | 154.293 | 138.883 | 9.99% |


The 128/512-row cases reduce preparation time by 8.20–9.99% in both invocations. Single-row timings vary markedly between invocations; no single-row benefit is claimed. Eight-row benefits are smaller and variable. Every sample is retained in [round 1](gpu/operator-1.jsonl) and [round 2](gpu/operator-2.jsonl).

## Correctness

- [54 small cases](gpu/small-validation.log) passed with `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1` before the full GPU matrix.
- [264 full cases](gpu/full-correctness.log): D=64/128/256/512, T=1/8/128/512, rotary widths 0/2/32/64/128/256/512 where width <= D, ordinary RoPE and both M-RoPE position modes. Q/K/V output buffers match the original kernels bitwise, including untouched cache rows; output guard bytes remain intact.
- [Standard model output gate](gpu/logit-compare.txt): 584 rows bitwise identical (8 prefill, 192 step, 192 batch, 192 verify). The frozen `qwen3.8-27b-4bit.json` floor was used unchanged; no `--reorder`.
- [64K stress gate](gpu/long-compare.txt): 17 rows bitwise identical (1 prefill, 8 step, 8 verify). The [synthetic teacher-forced input](long-input/README.md) repeats a corpus window to 65,545 token IDs. This is numerical stress coverage; its sequential dump timings are not a speed benchmark.
- [35 CPU test groups and prek](cpu-summary.txt) passed, including the latest main tests and the installed 27B tokenizer/parser checks.

The old build directory needed an explicit CMake reconfigure to discover newly added main targets. `vk doctor` reports a false STALE warning because it expects eight SHA characters while these CLI binaries print seven; [both version strings](https://github.com/chainhelen/kapitsa/tree/743a78dd00b42ae0d752896446209e343531cf29/bench/results/2026-10-31-m4pro-rope-current/gpu/build-versions.txt) match the recorded commits. Both checkouts were explicitly configured and rebuilt before GPU work.

## Full-request A/B

Produced by `kapitsa bench --model MODEL_DIR --quick --json --out FILE`, in A1/B1/A2/B2 order, with a fresh `KAPITSA_HOME` per run. The normal ANE and KFlash/DFlash2 defaults were retained; [performance environment](gpu/performance-env.json). Thermal state was fair throughout these runs; each report has empty contention and invalid lists, no swapouts, and AC power.

| Run | Decode tok/s | Greedy tok/s | 4K prefill tok/s | Follow-up TTFT (s) | Thermal | Invalid |
|---|---:|---:|---:|---:|---|---|
| A1 | 59.42 | 63.58 | 381.9 | 0.2384 | fair | [] |
| B1 | 58.11 | 63.3 | 388.8 | 0.2402 | fair | [] |
| A2 | 57.16 | 62.16 | 389.8 | 0.2402 | fair | [] |
| B2 | 59.7 | 64.64 | 388.3 | 0.2409 | fair | [] |


No measurable end-to-end speedup is claimed: direction varies between pairs, and differences fit within the same-build spread. Every [JSON report](gpu/bench-A1.json), stdout/stderr transcript and [server log](https://github.com/chainhelen/kapitsa/tree/743a78dd00b42ae0d752896446209e343531cf29/bench/results/2026-10-31-m4pro-rope-current/server-logs) is retained. [Decode validity](https://github.com/chainhelen/kapitsa/tree/743a78dd00b42ae0d752896446209e343531cf29/bench/results/2026-10-31-m4pro-rope-current/gpu/decode-validity.json) records all 24 measured samples: 512 tokens and finish=length, six valid samples per run. The server logs retain tokens per cycle and cycle counts; the benchmark JSON also records accepted draft throughput and shares. The drafter is DFlash2 from the model's `dflash2` folder, SHA256 `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.

## Live API control

Both builds were served privately on localhost with context 32k. The first run explicitly used one slot; it produced three conversation-retention failures plus the keepalive failure on both builds. Those three retention checks pass when the server's default four slots are used. The initial failures are preserved, along with both full default-slot runs:

| Configuration | Build | Pass | Fail | Skip |
|---|---|---:|---:|---:|
| 1 slot | A | 44 | 4 | 1 |
| 1 slot | B | 44 | 4 | 1 |
| default (4 slots) | A | 48 | 1 | 0 |
| default (4 slots) | B | 48 | 1 | 0 |

With default slots, both builds fail only the existing "empty chunk every 15 s" test: its long-request holder ends after two tokens, before a 15-second keepalive can be required. This suite is not reported as fully passing. See [baseline default-slot results](gpu/api-default-A-tests.log), [candidate default-slot results](gpu/api-default-B-tests.log) and the corresponding server logs. No test prompt or API/cache implementation was changed in this PR.

## Free-running greedy completion check

A separate HTTP check on 2026-10-31 uses A=`15f8a57`, B=`743a78d`, the same 27B model and this Mac, temperature 0, `--spec 0`, context 8192, one slot, and a fresh scratch home per run. The 2,957-token [prompt](greedy/prompt.txt) and all four generated texts are retained. ANE startup completed before requests. This checks ordinary decode, where the changed preparation kernel runs for each token. Each server held the GPU lock.

All four nonempty completions contain 512 tokens and finish with `length`; the returned UTF-8 text is byte-identical across A1/B1/A2/B2. The HTTP responses report source `none` and 511 cycles. Raw token-ID traces were not collected; the equality claim here is for returned text, alongside the fixed-prefix logit gates above.

| Run | Build | Output tokens | Finish | Decode tok/s | Prefill ms | Thermal before/after | Text |
|---|---|---:|---|---:|---:|---|---|
| [A1](greedy/A1.json) | kapitsa 0.2.25 (15f8a57) | 512 | length | 14.7023 | 7996.1 | 0/0 | [output](greedy/A1-output.txt) |
| [B1](greedy/B1.json) | kapitsa 0.2.25 (743a78d) | 512 | length | 14.7085 | 8054.6 | 0/0 | [output](greedy/B1-output.txt) |
| [A2](greedy/A2.json) | kapitsa 0.2.25 (15f8a57) | 512 | length | 14.8252 | 8073.0 | 0/0 | [output](greedy/A2-output.txt) |
| [B2](greedy/B2.json) | kapitsa 0.2.25 (743a78d) | 512 | length | 14.6872 | 7966.5 | 0/0 | [output](greedy/B2-output.txt) |

Thermal state 0 means nominal. All four were on AC. CPU validation was running concurrently, so these incidental timings are retained for completeness and are not a new performance claim. The default-speculation performance measurements remain the full-request A/B above.

Shared output SHA256: `7407ce83e410ee3713eb5ccfa3ef25040e6bee993958bc81198b51e203278c80`.

The runner is `tools/bench_plain_decode.py`; the local copy only changes the candidate metadata label from `reduce` to `rope`. Its requests, flags and assertions are unchanged. Here `R`/`C` are the baseline/candidate checkout paths and `M` is the model folder. Reproduce with:

```sh
python3 tools/bench_plain_decode.py --reference "$R/build/kapitsa" --candidate "$C/build/kapitsa" \
  --model "$M" --prompt bench/results/2026-10-31-m4pro-rope-current/greedy/prompt.txt \
  --out /path/to/new-greedy-evidence --port 18135
```

[Readiness checks](ready-checks.txt): all 35 CPU/tokenizer groups and prek passed again; documentation links and the generated benchmark index also passed. The production kernels and test reference were checked against the measured source and upstream, respectively.

## Not verified

- Qwen3.5-4B and Qwen3.6-35B-A3B output gates, request A/B and live APIs: neither model is installed here. The tier-2 [coverage checklist](https://github.com/chainhelen/kapitsa/tree/743a78dd00b42ae0d752896446209e343531cf29/bench/results/2026-10-31-m4pro-rope-current/required-checks.txt) names these models; they remain explicitly unverified.
- Other Mac generations or macOS versions.
- Real image/video prompts: M-RoPE coverage here is synthetic.
- No end-to-end speedup, single-row speedup or KFlash tree-preparation speedup is claimed.

## Reproduce

Build `kapitsa` and `logit_dump` with CMake in both comparison checkouts, and `rope_pair_test` in the candidate checkout. The test tool takes the GPU lock itself:

```sh
MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 build/rope_pair_test --gpu --small
build/rope_pair_test --gpu
build/rope_pair_test --bench
build/rope_pair_test --bench
```

For the model checks, use the repository's `verify-kapitsa` output-check procedure: baseline `logit_dump`, candidate `logit_dump --from REF_DUMP`, then `tools/logit_compare.py`. Repeat with the supplied `long-input` and `--paths step,verify`. Run the standard quick request benchmark A/B/A/B with separate scratch homes. Run `tests/test_server.py URL` against `vk serve MODEL_DIR --context 32k` on each build, preserving the default slots.

The [2026-10-29 experiment](https://github.com/chainhelen/kapitsa/tree/743a78dd00b42ae0d752896446209e343531cf29/bench/results/2026-10-29-m4pro-operators/rope/README.md) remains unchanged as historical evidence against its older baseline. This directory contains the new-main revalidation.
