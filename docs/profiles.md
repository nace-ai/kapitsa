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

## Members

| Member | Model, Mac | Switch | Faster | Drift against precise (72 rows: 16K, 32K, 64K windows) |
|---|---|---|---|---|
| Sparse prefill attention on the matrix units, Q and P as hi + lo halves, at the usual 2048-row chunks | Qwen3.8-Flash-Next on Macs with matrix units; measured: Q2_0, M5 Pro (the 4-bit pack and the M5 Max did not run) | `NI_QSA_TATTN` | prefill 8% faster at 32K, 10% at 64K; decode the same per draft cycle (4 interleaved pairs) | 4 flips, mean KL 8.8e-3, NLL +0.018 nats |

Measured 6 Nov 2026 on an M5 Pro 48 GB, macOS 26.5.2, `logit_dump --paths step` at the profile's chunk: `precise` is
bitwise identical to main on all 72 rows, and `fast` at 4096-row chunks is bitwise identical to `fast` at 2048.

Speed, same Mac, `ni_run` with a 32K and a 64K prompt and 128 tokens of decode, 4 interleaved rounds of precise, fast,
and fast with 4096-row chunks (the chunks `NI_QSA_TATTN=1` takes since #571), expert store pinned at 24 GB:

| | 32K prefill | 64K prefill | Decode, ms per draft cycle |
|---|---|---|---|
| fast (2048-row chunks) | 8% faster | 10% faster | the same at 32K, 4% faster at 64K |
| fast with 4096-row chunks | 15% faster | 17% faster | 7 to 8% slower; every leg under memory pressure |

Decode tokens per second also differ because the greedy text differs (2.05 against 1.95 tokens per cycle at 32K), not
because a step is slower. The 4096-row chunks add 0.5 GB to the prefill group's slots, and on a 48 GB Mac that tips
the pinned run into memory pressure; with the store left to size itself, the run swapped past 8 GB and was stopped. So
`fast` keeps 2048-row chunks; `NI_QSA_TATTN=1` keeps #571's 4096 (`Qwen4Exp::tattn_chunks`).

## Candidates

Ranked by measured speed, then how many models they cover, then drift. Status as of 6 Nov 2026, 17:30. Drift is against Kapitsa's precise path on the M5 Pro.

| # | Speedup | Models, Mac | What moves | Measured speed | Measured drift (vs exact) | Status |
|---|---|---|---|---|---|---|
| 1 | TATTN sparse attention, Q and P as hi + lo halves (#818, `NI_QSA_TATTN`) | Flash-Next Q2_0, M5 Pro | sparse prefill attention on the matrix units, about 22 bits per operand | prefill 8% faster at 32K, 10% at 64K; decode the same per cycle (with 4096-row chunks: 15% and 17%, but decode 7 to 8% slower per cycle under memory pressure) | 4 of 72 flips, mean KL 8.8e-3 | first fast member |
| 2 | W4A8 dense prefill GEMMs (`NI_QMT_I8`) | 4-bit group-64 dense projections on an M5: Flash-Next, 35B-A3B | activations rounded to int8 per 64-group | Flash-Next 32K prefill 3-13% faster (#572, before grouped prefill, which it turns off) | 35B, 192 rows: 5 flips, mean KL 4.0e-3 (bar 4, 2.1e-3); Flash-Next with TATTN: 6 of 72, 2.1e-2 | 35B: over the bar, and 4K prefill 3.5% slower (4 pairs); not a member |
| 3 | TATTN single half (`NI_QSA_TATTN`, before #818) | Flash-Next, M5 | Q, K, P in half | 32K prefill 10-15% faster | 3 of 96 flips, mean KL 4.4e-3; a 64K reply ends at once (#585) | replaced by 1 |
| 4 | Neural Engine prefill for hidden 2048-8192 (#775) | Qwen3.5 9B, M4 Pro | W8A8 FFN on the Neural Engine | 4K time to first token 63% less | 2 of 192 flips, mean KL 3.6e-4 | open PR |
| 5 | int8 KV cache (`NI_KV_I8`) | Qwen3.5/3.6/3.8, Flint | KV rows int8 with a row scale | 35B on an M4 16 GB: decode 4% faster, 1.9x the context; 35B on the M5 Pro 48 GB, 4K prompt: decode about the same (117.0 against 116.8 tok/s, 4 pairs) | 35B, 192 rows: 3 flips, mean KL 1.1e-3 (bar 4, 2.1e-3) | inside the 35B bar but not faster on the M5 Pro; its win is context on small Macs; not a member here |
| 6 | Verify MoE without row-independent numerics (`NI_DFLASH_EXACT_MOE=0`) | 35B-A3B, Clover | a verify row's logits depend on the other rows' routing | source comment: exact verify about 3% slower | 35B, 192 verify rows: 0 flips, mean KL 1.8e-4, passes the precise floor gate | decode about the same on the M5 Pro (4 pairs); passes precise's gate, so a precise question, not fast |
| 7 | Nemotron Lightning attention projections in 8-bit (`NI_NH_ATTN_Q8`) | Lightning | BF16 to 8-bit weights | decode about 3% faster | 192 rows: 1 flip (a near-tie), mean KL 6.5e-4; inside the fallback bar (4 of 192, 2.1e-3) | fast candidate; speed A/B not run: the model sits on the external drive, unmounted on 6 Nov |
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
