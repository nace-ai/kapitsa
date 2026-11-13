# Profiles: precise and fast

Kapitsa runs one of two profiles.

- **precise** (the default) is today's Kapitsa: every default passed the output check against the model's noise floor
  on this Mac (CONTRIBUTING.md, "Output check").
- **fast** is precise plus every speedup that moves the outputs no more than the fastest other engine does for that
  model on that Mac. `kapitsa serve --profile fast` or `KAPITSA_PROFILE=fast` turns it on.

A fast member is on for the whole class of Macs or models it targets, as every default is (AGENTS.md, "Defaults follow
the concept"); its class is a row in `src/engine/device_profile.cpp`. Outside its class the fast profile is the precise one.

## The bar

For a model and a Mac, the bar is how far the fastest other engine's logits are from Kapitsa's precise path on the same
rows: its top-1 flips and its mean KL, from `tools/logit_compare.py`. A fast member qualifies when, on those rows, its
flips and its mean KL against precise are both at or below the bar, and an interleaved A/B shows it faster.

"Fastest other engine" is measured, not assumed: the engine with the highest single-stream decode on that model and Mac
in [benchmarks.md](benchmarks.md). It runs the rows teacher-forced, full vocabulary, in `logit_dump`'s layout
(`sequences.txt`, `records.txt`, `logits.f32`), so the same `logit_compare.py` reads both engines.

| Model, Mac | Fastest other engine | Rows | Top-1 flips | Mean KL |
|---|---|---|---|---|
| Qwen3.8-Flash-Next Q2_0, M5 Pro | Splash 1.5.0 | 72 (16K, 32K, 64K windows) | 16 | 0.46 |
| Qwen3.6-35B-A3B 4-bit, M5 Pro | Splash 1.5.0 | 192 (8 windows, 512 to 3,200 tokens) | 4 | 2.1e-3 |

On Flash-Next, Splash's rows were scored from its prefix cache; on the 32 rows also scored cold, the two agree (same
top-1 on 31 of 32 rows, mean KL at most 1.5e-2).

**A model no other engine runs** (Nemotron 3.5 Lightning today) takes the strictest bar measured on any model: the
lowest flip rate (flips per row) and the lowest mean KL in the table above, both from the same row. Today that is the
35B-A3B's: 4 flips in 192 rows (2.1%), mean KL 2.1e-3. The member's own output check runs on its own model's rows and
must be at or below both. When another engine starts running that model, its measured bar replaces the fallback.

### Qwen3.5-9B model-specific bar

These #775 measurements apply to Qwen3.5-9B only. They are separate from the fallback table above and do not
change the fallback or another candidate's admission.

| Model, Mac | Fastest other engine | Rows | Top-1 flips | Mean KL |
|---|---|---|---|---|
| Qwen3.5-9B 4-bit, M4 Pro | MLX 0.32.4 / mlx-lm 0.32.2 | 576 comparisons (192 shared positions, step/verify/batch) | 9 | 3.747187506485964e-4 |
| Qwen3.5-9B 4-bit, M3 Ultra | MLX 0.32.4 / mlx-lm 0.32.2 | 576 comparisons (192 shared positions, step/verify/batch) | 12 | 3.9909652322670414e-4 |

The 9B rows select MLX from measured native decode: 50.212 vs llama.cpp b12110 at 39.076 tok/s on the M4 Pro,
114.095 vs 80.998 on the M3 Ultra. Each engine's 192 native single-token rows are reused for Kapitsa's step, verify
and batch paths: 576 correlated comparisons, not 576 independent native outputs. The requested 576-row checks
pass separately per path on both Macs. The M4 also passes eight prefill rows; **the M3 prefill check fails**:
fast mean KL 1.845747910960617e-4 vs MLX 1.4716473412187535e-4, both with zero flips. It is not hidden in the
aggregate bar. [M4 exact bounds and narrow verify margin](../bench/results/2026-11-07-pr775-m4-default-share/README.md);
[M3 bounds, failed prefill and row-alignment audit](../bench/results/2026-11-07-pr775-m3-current/README.md).


## Members

| Member | Model, Mac | Switch | Faster | Drift against precise (rows specified per member) |
|---|---|---|---|---|
| Sparse prefill attention on the matrix units, Q and P as hi + lo halves, at the usual 2048-row chunks | Qwen3.8-Flash-Next on Macs with matrix units; measured: Q2_0, M5 Pro (the 4-bit pack and the M5 Max did not run) | `NI_QSA_TATTN` | prefill 8% faster at 32K, 10% at 64K; decode the same per draft cycle (4 interleaved pairs) | 72 rows (16K, 32K, 64K windows): 4 flips, mean KL 8.8e-3, NLL +0.018 nats |
| Segmented dense ANE prefill (#775) | Dense Qwen 2048-channel inputs through hidden 8192 on non-matrix Macs with a built-in ANE share (M4 today); measured Qwen3.5-9B on M4 Pro | `NI_ANE` | loaded 4135-token HTTP TTFT 61.94% lower at `50fadecc` (MTP off) | 576 correlated comparisons: 6 flips / mean KL 3.699283718701063e-4, against MLX 9 / 3.747187506485964e-4; all eight prefill rows also pass |


The ANE default for these widths excludes matrix GPUs and M1 to M3 (M1 and M2 have a share only for the 2560/5120
inputs; the M3 needs a calibration). The measured M3 Ultra prefill exceeds its
same-Mac MLX bar, so a calibration file or `NI_ANE_UNITS` alone no longer enables these new widths under fast.
An explicit diagnostic `NI_ANE=1` still overrides the profile and requires a usable share. The M3 measurements
are retained as the reason for this exclusion, not as evidence of a shipping M3 default. No bar was loosened.
[Built-in-share policy checks](../bench/results/2026-11-07-pr775-builtin-share/README.md) cover the excluded
chips, explicit overrides and unchanged M4 outputs.
The [long-prompt diagnostic](../bench/results/2026-11-07-pr775-long-quality/README.md) retains the large KL tails:
precise/fast each scored 1/4, BF16 3/4; this does not establish equivalence.

Sparse attention was measured 6 Nov 2026 on an M5 Pro 48 GB, macOS 26.5.2, `logit_dump --paths step` at the profile's chunk: `precise` is
bitwise identical to main on all 72 rows, and `fast` at 4096-row chunks is bitwise identical to `fast` at 2048.

Speed, same Mac, `ni_run` with a 32K and a 64K prompt and 128 tokens of decode, 4 interleaved rounds of precise, fast,
and fast with 4096-row chunks (the chunks `NI_QSA_TATTN=1` takes since #571), expert store pinned at 24 GB; then
precise against fast on the rebased build (4 legs each). Legs and logs: [results](../bench/results/2026-11-06-fast-profile-tattn/).

| | 32K prefill | 64K prefill | Decode, ms per draft cycle |
|---|---|---|---|
| fast (2048-row chunks) | 8% faster | 10% faster | the same at 32K, 4% faster at 64K |
| fast, #986's build 4187b609 (32K: fn_ab5; 64K: fn_ab6, every fast leg under memory pressure, fn_ab5's 64K legs voided by the swap guard) | 10% faster | 7% faster | the same at 32K (1%, inside the legs' spread), 3% faster at 64K |
| fast with 4096-row chunks | 15% faster | 17% faster | 7 to 8% slower; 7 of 8 legs under memory pressure (the clean one 4% slower) |

Decode tokens per second also differ because the greedy text differs (2.05 against 1.95 tokens per cycle at 32K), not
because a step is slower. The 4096-row chunks add 0.5 GB to the prefill group's slots, and on a 48 GB Mac that tips
the pinned run into memory pressure; with the store left to size itself, the run swapped past 8 GB and was stopped. So
`fast` keeps 2048-row chunks; `NI_QSA_TATTN=1` keeps #571's 4096 (`Qwen4Exp::tattn_chunks`).

## Candidates

The historical ranking (6 Nov 2026, 17:30) uses measured speed, then model coverage, then drift. The 9B row was
updated on 7 Nov; rows name their measured Mac. Drift is against Kapitsa's precise path on that Mac.

| # | Speedup | Models, Mac | What moves | Measured speed | Measured drift (vs exact) | Status |
|---|---|---|---|---|---|---|
| 1 | TATTN sparse attention, Q and P as hi + lo halves (#818, `NI_QSA_TATTN`) | Flash-Next Q2_0, M5 Pro | sparse prefill attention on the matrix units, about 22 bits per operand | prefill 8% faster at 32K, 10% at 64K; decode the same per cycle (with 4096-row chunks: 15% and 17%, but decode 7 to 8% slower per cycle under memory pressure) | 4 of 72 flips, mean KL 8.8e-3 | first fast member |
| 2 | W4A8 dense prefill GEMMs (`NI_QMT_I8`) | 4-bit group-64 dense projections on an M5: Flash-Next, 35B-A3B | activations rounded to int8 per 64-group | Flash-Next 32K prefill 3-13% faster (#572, before grouped prefill, which it turns off) | 35B, 192 rows: 5 flips, mean KL 4.0e-3 (bar 4, 2.1e-3); Flash-Next with TATTN: 6 of 72, 2.1e-2 | 35B: over the bar, and 4K prefill 3.5% slower (4 pairs); not a member |
| 3 | TATTN single half (`NI_QSA_TATTN`, before #818) | Flash-Next, M5 | Q, K, P in half | 32K prefill 10-15% faster | 3 of 96 flips, mean KL 4.4e-3; a 64K reply ends at once (#585) | replaced by 1 |
| 4 | Neural Engine prefill for hidden 2048-8192 (#775) | Non-matrix Macs with a built-in ANE share (M4 today); matrix Macs and M1 to M3 excluded | W8A8 FFN on the Neural Engine | M4 Pro built-in share: loaded HTTP TTFT 61.94% less (MTP off) | 6/576 correlated comparisons, mean KL 3.699283718701063e-4; [rebase evidence](../bench/results/2026-11-07-pr775-rebase-3f3b/README.md) | member above; M3 excluded for its prefill bar failure; zero drift fails; long-prompt quality limits remain disclosed |
| 5 | int8 KV cache (`NI_KV_I8`) | Qwen3.5/3.6/3.8, Flint | KV rows int8 with a row scale | 35B on an M4 16 GB: decode 4% faster, 1.9x the context; 35B on the M5 Pro 48 GB, 4K prompt: decode about the same (117.6 against 116.8 tok/s, 4 pairs; the greedy text differs, 43 decoded tokens against 41) | 35B, 192 rows: 3 flips, mean KL 1.1e-3 (bar 4, 2.1e-3) | inside the 35B bar but not faster on the M5 Pro; its win is context on small Macs; not a member here |
| 6 | Verify MoE without row-independent numerics (`NI_DFLASH_EXACT_MOE=0`) | 35B-A3B, Clover | a verify row's logits depend on the other rows' routing | source comment: exact verify about 3% slower | 35B, 192 verify rows: 0 flips, mean KL 1.8e-4, passes the precise floor gate | decode about the same on the M5 Pro (4 pairs); passes precise's gate, so a precise question, not fast |
| 7 | Nemotron Lightning attention projections in 8-bit (`NI_NH_ATTN_Q8`) | Lightning | BF16 to 8-bit weights | M5 Pro 48 GB, 8 interleaved `kapitsa bench --quick` legs: decode about the same (T 0.6 +1%, T 0 -4%, ranges overlap), 4K prefill 8% slower ([results](../bench/results/2026-11-07-lightning-attn-q8/)) | 192 rows: 1 flip (a near-tie), mean KL 6.5e-4; inside the fallback bar (4 of 192, 2.1e-3); fails the floor's gate (mean KL 6.5e-4 against 2.9e-4, p99 1.5e-2 against 6.8e-3), so it can only join `fast`, never `precise` | not a member: not faster |
| 8 | Fused Flash-Next indexer selection (`NI_QSA_FUSE`) | Flash-Next decode | block keys a few ulp off | Flash-Next 32K, decode about 6% slower (40.5 against 42.9 tok/s, 2 legs against 1; 1 leg void for load) | 72 rows: every row bitwise identical to precise | slower and no drift; not a member |
| 9 | Prompt prefix routed to 4 experts instead of 8 (#853) | 35B-A3B | fewer experts per prefix token | not reproduced | 7 flips, mean KL 3.6e-2 | closed |

Already lossy defaults (they passed their gates, so they are in precise): `NI_MOE_QTM`, `NI_QATTN_G8`/`D2` on the
35B-A3B, `NI_QMT6` on Flash-Next, the Neural Engine W8A8 split on the 4B and 27B, `NI_FLINT_SELF`.

## Rules

- **precise never moves.** A member never runs under precise. A change that passes the floor's gate belongs in precise.
- **One switch per member.** The member reads its switch through `profile_switch(env, fast_here, what)`
  (`src/engine/profile.h`): the switch's own variable wins (`0` off, anything else on), and when it is unset the member is
  on under fast where `fast_here` is true: the member's class, read from the profile table or a check the call site already has. The fast profile logs one line per member it turns on.
- **On by default counts.** A member on in the fast profile for its class is on by default there (AGENTS.md,
  "Defaults follow the concept, not the measured cell").
- **Measure at what ships.** The output check runs at the prefill chunk and paths the profile runs; the bar is
  re-measured when the other engine releases a faster version.
- **Members stack.** Before a second member joins for the same model, the stack is checked against the bar as a whole.
