# Profiles: precise and fast

Kapitsa runs one of two profiles.

- **precise** (the default) is today's Kapitsa: every default passed the output check against the model's noise floor
  on this Mac (CONTRIBUTING.md, "Output check").
- **fast** is precise plus every speedup that moves the outputs no more than the fastest other engine does for that
  model on that Mac. `kapitsa serve --profile fast` or `KAPITSA_PROFILE=fast` turns it on.

A fast member is on only for the model and Mac it was measured on. Anywhere else the fast profile is the precise one.

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

Splash's rows were scored from its prefix cache; on the 32 rows also scored cold, the two agree (no flip between them on
24 of 32 rows' top-1, mean KL at most 1.5e-2).

## Members

| Member | Model, Mac | Switch | Faster | Drift against precise (72 rows: 16K, 32K, 64K windows) |
|---|---|---|---|---|
| Sparse prefill attention on the matrix units, Q and P as hi + lo halves; 4096-row prefill chunks with it | Qwen3.8-Flash-Next Q2_0, M5 Pro | `NI_QSA_TATTN` | prefill 12% faster at 32K, 20% at 64K (#818) | 4 flips, mean KL 8.8e-3, NLL +0.018 nats |

Measured 6 Nov 2026 on an M5 Pro 48 GB, macOS 26.5.2, `logit_dump --paths step` at the profile's chunk: `precise` is
bitwise identical to main on all 72 rows, and `fast` at 4096-row chunks is bitwise identical to `fast` at 2048.

## Candidates

Ranked by measured speed, then how many models they cover, then drift. Status as of 6 Nov 2026.

| # | Speedup | Models, Mac | What moves | Measured speed | Measured drift (vs exact) | Status |
|---|---|---|---|---|---|---|
| 1 | TATTN sparse attention, Q and P as hi + lo halves (#818, `NI_QSA_TATTN`) | Flash-Next Q2_0, M5 Pro | sparse prefill attention on the matrix units, about 22 bits per operand; turns on 4096-row chunks | prefill 12% faster at 32K, 20% at 64K; decode about 3% slower | 4 of 72 flips, mean KL 8.8e-3 | first fast member |
| 2 | W4A8 dense prefill GEMMs (`NI_QMT_I8`) | 4-bit group-64 dense projections on an M5: Flash-Next, 35B-A3B | activations rounded to int8 per 64-group | Flash-Next 32K prefill 3-13% faster (#572, before grouped prefill, which it turns off) | Flash-Next: 4 of 96 flips, mean KL 7.0e-3 | measured here, see Members |
| 3 | TATTN single half (`NI_QSA_TATTN`, before #818) | Flash-Next, M5 | Q, K, P in half | 32K prefill 10-15% faster | 3 of 96 flips, mean KL 4.4e-3; a 64K reply ends at once (#585) | replaced by 1 |
| 4 | Neural Engine prefill for hidden 2048-8192 (#775) | Qwen3.5 9B, M4 Pro | W8A8 FFN on the Neural Engine | 4K time to first token 63% less | 2 of 192 flips, mean KL 3.6e-4 | open PR |
| 5 | int8 KV cache (`NI_KV_I8`) | Qwen3.5/3.6/3.8, Flint | KV rows int8 with a row scale | 35B on an M4 16 GB: decode 4% faster, 1.9x the context | to 9.7K: 3 of 200 flips, mean KL 6.4e-2 | opt-in; on an M5 it turns off the 35B's tree-verify attention kernel, likely slower there |
| 6 | Verify MoE without row-independent numerics (`NI_DFLASH_EXACT_MOE=0`) | 35B-A3B, Clover | a verify row's logits depend on the other rows' routing | source comment: exact verify about 3% slower | not measured before | measured here, see Members |
| 7 | Nemotron Lightning attention projections in 8-bit (`NI_NH_ATTN_Q8`) | Lightning | BF16 to 8-bit weights | decode about 3% faster | not measured | opt-in |
| 8 | Fused Flash-Next indexer selection (`NI_QSA_FUSE`) | Flash-Next decode | block keys a few ulp off | not measured | kernel check only | opt-in |
| 9 | Prompt prefix routed to 4 experts instead of 8 (#853) | 35B-A3B | fewer experts per prefix token | not reproduced | 7 flips, mean KL 3.6e-2 | closed |

Already lossy defaults (they passed their gates, so they are in precise): `NI_MOE_QTM`, `NI_QATTN_G8`/`D2` on the
35B-A3B, `NI_QMT6` on Flash-Next, the Neural Engine W8A8 split on the 4B and 27B, `NI_FLINT_SELF`.

## Rules

- **precise never moves.** A member never runs under precise. A change that passes the floor's gate belongs in precise.
- **One switch per member.** The member reads its switch through `profile_switch(env, fast_here, what)`
  (`src/engine/profile.h`): the switch's own variable wins (`0` off, anything else on), and when it is unset the member is
  on under fast where `fast_here` names the measured model and Mac. The fast profile logs one line per member it turns on.
- **On by default counts.** A member on in the fast profile for a model and Mac is on by default there (AGENTS.md,
  "New features ship on by default somewhere").
- **Measure at what ships.** The output check runs at the prefill chunk and paths the profile runs; the bar is
  re-measured when the other engine releases a faster version.
- **Members stack.** Before a second member joins for the same model, the stack is checked against the bar as a whole.
