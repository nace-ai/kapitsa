# Results, 2026-11-03: MacBook Pro M5 Max, 128 GB (Dory's decode attention with a row's keys 8 at a time)

The A/Bs, the output check and the kernel timings of `dory_attn_partial_k8` and `dory_attn_partial_rows_k8`
(`src/kernels/dory/dory.metal`): Dory's decode and verify attention with each row taking the keys of a tile 8 at a time
(8 scores, one running max over them and the row's history, one rescale), against `dory_attn_partial` and
`dory_attn_partial_rows`, one key at a time (`NI_DORY_ATTN_K8=0`). Dory1.0 4B (NVFP4) at its default (2 loops, per-loop
KV, self-drafts), one build. MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), AC
power, High Power mode, `caffeinate -is`, 2026-11-03 ([machine.json](machine.json): the build).

**This change moves outputs.** Dory's floor file has a decode gate of zero (its decode paths were bitwise repeatable and
no source varied them), and the decode rows here are not bitwise the old ones, so the floor's decode gate fails; see the
output check.

## Decode at temperature 0

Interleaved on one build (old, new, old, new; [t0/](t0)): one server per arm, the same two cold documents from the long
corpus (134,348 and 259,404 tokens with the template) and the same question, 512 tokens at temperature 0. The four arms
wrote the same text at both sizes, so their cycles and drafts match (2.47 and 2.41 tokens a cycle on every arm) and the
server logs' time per cycle compares like with like. The arms shared an SSD prefix cache: the 259K document was prefilled
once, by the first arm, and restored by the others (the 134K one is prefilled by each).

| Context | Old: tok/s (ms per cycle) | New: tok/s (ms per cycle) | Per cycle |
| ---: | ---: | ---: | ---: |
| 134K | 45.0 (54.9), 43.8 (56.4) | 48.4 (51.1), 47.8 (51.7) | -7.6% |
| 259K | 25.2 (95.6), 25.2 (95.8) | 28.2 (85.5), 27.8 (86.7) | -10.0% |

The script took the GPU lock for the whole run; it samples no GPU load, and the Claude desktop app was running.

The harness A/B before it ([harness-ab/](harness-ab), `compare_engines.py --agentic --sweep-task code`, T 0.6, one
request a size) is too noisy to read a change of this size from: the sweep phase ran under 54-56% foreign GPU load (the
desktop app), the first old run's 33K answer under a load spike (39.5 ms a cycle against 30.5 on the second), and the
new arm's 254K answer looped (distinct 4-grams 0.30), which drafts by prompt lookup instead of the model's own 1-loop
drafts and so costs less a cycle for that reason alone (61.2 against 104.4 ms). Its clean pairs agree with the table:
33K 30.5 -> 27.5 / 27.5 ms a cycle, 133K 59.0 -> 54.2.

## Output check

`logit_dump --paths step,verify --loop-kv per-loop` on this build with `NI_DORY_ATTN_K8=0` against the default
([output-check/](output-check)):

| Path | Rows | Bitwise | Top-1 | Flips | KL mean | KL p99 | Floor's decode gate | `--reorder` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| prefill | 8 | 8 | 1.0000 | 0 | 0 | 0 | | |
| step | 192 | 0 | 1.0000 | 0 | 3.26e-8 | 4.46e-7 | FAIL (gate 0) | PASS |
| verify | 192 | 0 | 1.0000 | 0 | 8.42e-7 | 8.33e-6 | FAIL (gate 0) | PASS |

The floor's general gate (2.84e-3 / 3.13e-2, set by its prefill chunking source) is 3,000 times the verify rows' mean KL;
its decode gate is zero because nothing varied Dory's decode paths when the floor was measured
(`tools/logit_floors/nemotron-labs-dory1.0-4b-nvfp4.json`, an M4 Pro). The `--reorder` bounds are for a reassociated
reduction; this change is not only that (each weight is the exponential of a difference to a max taken over 8 keys, then
rescaled, where the old kernel rescaled after every key), so that PASS is not a claim that it is.

## Temperature-0 continuations

The server at its default with `NI_DORY_ATTN_K8=0` and with the default ([greedy/](greedy)): the cold 4,131- and
16,465-token documents of #904's check and the same question, 512 tokens. The texts are identical at both sizes (1,858 and
2,132 characters); with the 134K and 259K ones above, the two kernels wrote the same text on all four documents.

## Kernel

One full-attention layer of Dory's heads, 40 chunks per KV head as the engine splits these lengths, best of 5
([kernel/](kernel): `dd_check.swift` and its output):

| Positions | Rows | Old | New | |
| ---: | ---: | ---: | ---: | ---: |
| 131K | 3 | 2.93 ms | 1.52 ms | 1.93x |
| 262K | 1 | 2.22 ms | 2.07 ms | 1.07x |
| 262K | 3 | 3.77 ms | 3.03 ms | 1.25x |
| 262K | 4 | 5.22 ms | 3.74 ms | 1.40x |

Combined over the chunks as `dory_attn_reduce` does, the new rows are within 1.1e-5 of the old ones. The 32K timings
(`k8.out`, `k8b.out`) moved between repeats by up to 2x for both kernels, so they are not quoted.
Both kernels hold 16 KB of threadgroup memory, as the old ones: `logit_dump --paths step,verify --loop-kv per-loop` under `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`, one window and then all 8 (512-3,200 tokens, where the sliding
layers' ring wraps), ran with no errors ([validation.txt](validation.txt)).
`s8.out`: a variant computing the 8 scores first and then running the old per-key update in order is bitwise the old
kernel but no faster (3-row verify 3.68 -> 4.01 ms at 262K): the serial exponentials, not the score sums, are the bound.

## Files

`t0/`: the temperature-0 A/B, its script and the four servers' logs. `harness-ab/`: the harness runs, their logs and
labels. `output-check/`, `greedy/`, `kernel/`: as above. `session/`: the chain that ran the output check, the
continuations and the harness A/B. Local paths are replaced by placeholders (`~`, `<repo>`, `<main>`, `<run-root>`,
`<scratch>`).
