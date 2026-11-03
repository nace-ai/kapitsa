# Results, 2026-11-03: MacBook Pro M5 Max, 128 GB (the 35B's tree verify attention with its K / V tiles staged)

`attn8ts_partial_zone_g8` (`src/kernels/tensor/v8attn_t.metal`, the default, `NI_V8T_STAGE`): the 8-row KFlash tree
verify's history attention of Qwen3.6 35B-A3B and Clover (16 query heads over 2 KV heads, head dim 256) on the M5's
matrix units, with each 32-key K / V tile read from the cache once into threadgroup memory for the threadgroup's four
simdgroup pairs, against `attn8t_partial_zone_g8` (`NI_V8T_STAGE=0`), where each pair reads the tile for itself (every
byte four times). The fragments, products, exchange, softmax and P V are the same and in the same order, so the
partials are the same to the bit. Qwen3.6 35B-A3B 4-bit at its default (DFlash2 with KFlash's tree verify), MacBook Pro
(16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), AC power, High Power mode,
`caffeinate -is`, 2026-11-03 ([machine.json](machine.json)).

## Decode at temperature 0

Interleaved on one build ([ab/run2](ab/run2)): a first arm only to fill the SSD cache, then old, new, old, new; one
server per arm, the same two cold documents from the long corpus (132,037 and 254,212 tokens with the template) and the
same question at temperature 0, thinking off. Every measured arm restored the 254K document's state from the shared
SSD cache and prefilled the 132K one, and the four wrote the same text at both sizes (384 and 500 tokens: both answers
end before the 512-token budget).

| Context | Old: tok/s (ms a cycle) | New: tok/s (ms a cycle) | Decode |
| ---: | ---: | ---: | ---: |
| 132K | 145.9 (22.4), 143.9 (22.7) | 156.7 (20.9), 160.8 (20.4) | +9.6% |
| 254K | 102.2 (30.7), 100.5 (31.2) | 113.2 (27.4), 112.2 (28.0) | +11.2% |

Tokens a cycle 3.27 at 132K on every arm, 3.10-3.14 at 254K (KFlash's row budget adapts between arms, and the cycles'
trees with it; the text does not depend on them). The first run ([ab/run1](ab/run1): old, new, old, new, no warm-up
arm) agrees, 144.5 / 139.3 -> 154.2 / 152.4 tok/s at 132K and 96.0 / 93.8 -> 108.3 / 106.4 at 254K, but its first arm
prefilled the 254K document cold where the others restored it (a different prefill chunking: 512 tokens there against
500 restored), and its script kept only the first arm's text, so run 2 was made. The script took the GPU lock for each
run; it samples no GPU load, and the Claude desktop app was running.

## Output check

`logit_dump --paths step,verify` (the verify: KFlash's tree, a chain of 8 live rows) on this build with
`NI_V8T_STAGE=0` and with the default ([output-check/](output-check)): every prefill, step and verify row bitwise
identical (192 step and 192 verify rows).

## Kernel

One layer of the 35B's heads, 8 rows of a random tree, the engine's zone chunking for a capacity of the positions + 64,
best of 10, interleaved ([kernel/](kernel): `qvs.swift` compiles the branch's tensor source as `mksrc.py` assembles
it; `qvs2.out`):

| Positions | `attn8t_partial_zone_g8` | `attn8ts_partial_zone_g8` | |
| ---: | ---: | ---: | ---: |
| 32K | 0.465 ms (144 GB/s) | 0.353 ms (190 GB/s) | 1.32x |
| 64K | 0.433 ms (310 GB/s) | 0.361 ms (372 GB/s) | 1.20x |
| 128K | 0.841 ms (319 GB/s) | 0.728 ms (369 GB/s) | 1.15x |
| 256K | 1.578 ms (340 GB/s) | 1.400 ms (383 GB/s) | 1.13x |

Every partial (O, max, sum of every row, head and chunk) bitwise the same at 40 to 262,144 positions (`qvs1.out`,
`qvs2.out`). Tried and dropped (`variants.patch`): the next tile prefetched into registers while the current one is
computed (`attn8tp`: 0.85x at 256K, register pressure), K staged alone with V read as before in 16 KB (`attn8tk`,
`qvs3.out`: 0.83x against the staged kernel), and the 27B's wide verify kernel staged the same way
(`attn8ts_partial_tree_g6`, `qvs4.out`, `CFG=g6`: 0.84x at 256K; three pairs share a tile there, and 32 KB against its
12 KB costs more than the reads save). `qvs.swift` gained `CFG=g6` after the zone runs.

The staged kernel holds 32 KB of threadgroup memory (both tiles; the pairs' score exchange lives in the K tile once the
scores are taken). The validation layer counts it twice, past its 32,768 B cap, so it cannot dispatch it: the host keeps
`attn8t_partial_zone_g8` under `MTL_SHADER_VALIDATION`, and the staged kernel ran without the layer.

## Files

`ab/`: the A/B script (`ab_q35.py`, as run 2 ran it: run 1 ran it before it kept every arm's text and took
`AB_WARM`), each run's command, output, results and servers' logs. `output-check/`: the comparison and its command
(`ab/run1/run.sh`, which ran the output check first). `kernel/`: the harness, its outputs and the dropped variants.
Local paths are replaced by placeholders (`~`, `<repo>`, `<run-root>`, `<scratch>`).
