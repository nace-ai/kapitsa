# Results, 2026-11-03: MacBook Pro M5 Max, 128 GB (Dory's prefill attention with the head dim split over simdgroup pairs)

The A/B, the output check and the kernel checks of `dory_attn_td2_g4s8` (`src/kernels/tensor/nh_dory_attn_t.metal`):
Dory's prefill attention on the M5's matrix units with each simdgroup holding half of Q and of O, a pair of them per 16
rows, against `dory_attn_t_g4s8k16`, which holds all of both (`NI_DORY_ATTN_D2=0`). Dory1.0 4B (NVFP4) at its default
(2 loops, per-loop KV, self-drafts), `--ctx 262144`, the coding answer after a cold document from the long corpus, one
request per size, the two arms interleaved on one build (`compare_engines.py configs/dory.json sweep-configs/ab10.json
--only <label> --agentic --skip greedy,prefill,tools,agent,cache,sustain --quiet-timeout 60`). MacBook Pro (16-inch,
2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), AC power, High Power mode, `caffeinate -is`,
04:31-05:10 local time ([machine.json](machine.json): the build).

## Prefill

Seconds to the first token, each value one cold request:

| Prompt | Old | New | Speedup | Prefill tok/s, old → new |
| ---: | ---: | ---: | ---: | ---: |
| 33K | 20.10, 20.53 | 16.02, 16.32 | 1.23-1.28x | 1,632 → 2,050 |
| 133K | 185.23, 184.67 | 124.74, 126.21 | 1.46-1.48x | 721 → 1,063 |
| 255K | 586.69 | 374.67 | 1.57x | 434 → 680 |

Decode is the same (the change is the prefill's): 72.5 / 71.0 against 68.1 / 71.1 tok/s at 33K, 32.7 / 33.7 against
33.8 / 32.3 at 133K, 19.3 against 20.1 at 255K.

The 33K and 133K runs (old, new, old, new) had the GPU to themselves: the quiet wait ended after 19 s before each, and
the sweep's mean foreign GPU load was 0.0-0.3%. The 255K pair (old, then new) did not: the Claude desktop app's GPU
process held a mean of 26.0% during the old run's sweep and 11.0% during the new one's, which favours the new arm, and
both JSONs are flagged. The old arm's 586.7 s is still faster than the 621-627 s this model's 255K prompt took on main
before the change (the four 255K runs of `2026-11-03-m5-max-long-v3`, #900). The first try of that pair asked
for 255,000 tokens of document, which the chat template took past the context ([failed-255000/](failed-255000): HTTP
400 on both arms, nothing measured); the pair was re-run at the 240,000 the long-context sweep uses.

The 255K saving is consistent with the idle pairs: the attention's work grows with the square of the prompt (x3.65 from
133K to 255K), so the idle 133K pairs' savings of 60.5 s and 58.5 s predict 213-221 s at 255K, and the loaded pair saved
212.0 s; the kernel alone is 1.78x at 254K positions, the prompt 1.57x end to end.

Scope: the switch is on wherever the tensor prefill attention runs (an Apple10 GPU, macOS 26, 4 query heads per KV
head), so on every M5-class Mac, but everything here was measured on the M5 Max; the other M5 Macs take it untested.

## Kernel

One full-attention layer of Dory's heads (16 query heads over 4 KV heads, head dim 256), 2,048 rows after the cached
positions, best of 3 ([kernel/timing.out](kernel/timing.out); `kernel/da_check.swift` compiles the branch's tensor
source as `kernel/mksrc.py` assembles it):

| Positions | `dory_attn_t_g4s8k16` | `dory_attn_td2_g4s8` | |
| ---: | ---: | ---: | ---: |
| 14K | 27.9 ms (18.5 TFLOP/s) | 14.8 ms (34.7) | 1.89x |
| 63K | 117.6 ms (18.4) | 67.1 ms (32.3) | 1.75x |
| 129K | 249.2 ms (17.5) | 130.7 ms (33.4) | 1.91x |
| 254K | 475.0 ms (18.0) | 267.2 ms (32.0) | 1.78x |

- Against a double-precision softmax over each sampled row's keys (16 rows of the chunk, every head), both kernels' error
  is the same at every size: 9.4e-6 / 9.7e-6 at 14K, 1.04e-5 / 1.04e-5 at 63K, 9.68e-6 / 9.68e-6 at 129K, 1.14e-5 /
  1.12e-5 at 254K; the two differ by 1.9e-6 - 2.8e-6. With short prefixes, where a row averages few keys, the same holds
  at 1.1e-4 / 1.1e-4 ([kernel/ref.out](kernel/ref.out)).
- Ragged chunks (37, 64, 1,000, 2,047 and 4,096 rows after 0, 1, 333 and 8,192 positions), a 1,024-key sliding window over
  a ring of 1,024 slots and over a full cache: every output finite, at most 1.0e-4 from the old kernel's
  ([kernel/shapes.out](kernel/shapes.out)).
- Static threadgroup memory 24,576 B (two 16-key half tiles and the pairs' fp32 score exchange). The validation layer
  counts it twice, past its 32,768 B cap, so it cannot dispatch the kernel; the host runs `dory_attn_t_g4s8k16` (16 KB)
  under `MTL_SHADER_VALIDATION`.

## Output check

`logit_dump --paths step,verify --loop-kv per-loop` on this build with `NI_DORY_ATTN_D2=0` against the default
([output-check/](output-check)): top-1 agreement 1.0000 and no flips on every path; KL mean / p99 8.9e-6 / 5.5e-5
(prefill, 8 rows), 1.09e-5 / 2.42e-4 (step, 192 rows), 1.11e-5 / 2.18e-4 (verify, 192 rows), against this model's floor
gate of 2.84e-3 / 3.13e-2 (`tools/logit_floors/nemotron-labs-dory1.0-4b-nvfp4.json`): PASS, and PASS on the order-change
gate (`--reorder`).

## Temperature-0 continuations

The model served at its default (`--loop-kv per`, `--ctx 65536`) with `NI_DORY_ATTN_D2=0` and with the default, the same
cold documents from the long corpus (4,131 and 16,465 tokens with the template) and the same question, 256 tokens at
temperature 0 ([greedy/](greedy)): the two texts are identical at both sizes (1,270 and 1,086 characters). The servers'
logs (`server-old.log`: `NI_DORY_ATTN_D2=0`, `server-new.log`: the default) are cold starts with no quiet wait, so their
prompt times are not a measurement (the 16K prompt took 7.6 s in `server-old.log` and 6.5 s in `server-new.log`; the A/B
above and the kernel timings, the sliding layers' included, are the measurements).

## Files

`results/`, `logs/`: the runs (`dory-d2-old*`: `NI_DORY_ATTN_D2=0`; `dory-d2-new*`: the default) and their server logs.
`session/`, `sweep-configs/ab10.json`: the runner, its output and the labels (the 255K labels' size is the 240,000 they
ran at). `kernel/`: the kernel checks, their scripts and output. `greedy/`: the continuations, the script and both
servers' logs. Local paths are replaced by placeholders (`~`, `<repo>`,
`<main>`, `<run-root>`, `<scratch>`).
