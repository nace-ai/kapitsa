# Results, 2026-11-03: MacBook Pro M5 Max, 128 GB (the 27B's 8-row tree verify on its wide verify's kernel)

Qwen3.8 27B's 8-row KFlash tree verify (the 8-row drafting block: a server of more than one slot, or
`NI_DFLASH_BLOCK=8`) takes the tree path its 16-row wide verify takes, `attn8t_partial_tree_g6` on the M5's matrix
units with one row tile (#897's kernel), instead of the zone path, whose history keys ran `attn8m_partial_zone` on the
shader ALUs (`NI_V8_TREE_G6=0`). No kernel changes: `src/model/qwen35_v8.cpp` picks the path. Qwen3.8 27B 4-bit with
its DFlash2 drafter and KFlash, MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4
(25E246), AC power, High Power mode, `caffeinate -is`, 2026-11-03 ([machine.json](machine.json)).

## Where a cycle went

`ni_run` with `NI_CYCLE_TRACE=1` at 131,404 tokens of context, 256 tokens at temperature 0 ([trace/](trace)); and
`NI_PROFILE=1` over 15 cycles of the zone path (`zone-8row-and-profile.out`): `attn8m_partial_zone` 41.2 ms of an
85.5 ms cycle (48%), the dense projections 37 ms.

| Drafting block, verify | Decode | Tokens a cycle | ms a cycle (verify) |
| --- | ---: | ---: | ---: |
| 8 rows, zone path (before) | 48.6 tok/s | 4.11 | 84.7 (80.6) |
| 16 rows, wide verify (the one-slot default) | 49.3 | 4.55 | 92.4 (86.9) |
| 16 rows, every cycle at 8 rows (`NI_KFLASH_FORCE=8`) | 51.6 | 3.86 | 74.9 (68.7) |
| 8 rows, tree path (this change) | 62.2 | 4.11 | 66.2 (61.9) |

## Decode, a 2-slot server

Interleaved on the final build on an idle GPU ([ab/](ab): a first arm only to fill the SSD cache, then `NI_V8_TREE_G6=0`,
the default, `0`, the default): `kapitsa serve --slots 2` (its default drafting block is 8 rows), the same two cold
documents from the long corpus (132,037 and 254,212 tokens with the template) and the same question, 512 tokens at
temperature 0, thinking off. Every measured arm restored both documents from the shared SSD cache, and the four wrote the
same text at both sizes (512 and 398 tokens). Each request waited for a quiet GPU (no foreign process above 5% for 15 s:
`bench/harness/gpu_hogs.py`) and sampled the foreign GPU load every 2 s while it ran: every measured request waited
18.5-45.3 s and ran at a mean and a maximum of 0.0% foreign load (the warm-up arm, which is not measured, ran beside the
Claude desktop app's helper at 3-4%).

| Context | Zone path: tok/s (ms a cycle) | Tree path: tok/s (ms a cycle) | Decode |
| ---: | ---: | ---: | ---: |
| 132K | 53.2 (76.2), 52.8 (76.8) | 70.3 (57.7), 70.2 (57.7) | +32.5% |
| 254K | 31.1 (110.1), 31.0 (110.4) | 45.0 (76.1), 45.0 (76.0) | +44.9% |

The first run ([ab-first/](ab-first), the same arms and build) sampled no GPU load and ran while the desktop app was in
use: 43.7 / 45.1 -> 57.1 / 57.4 tok/s at 132K and 26.3 / 26.1 -> 34.9 / 35.0 at 254K, the same text. The review asked for
the run on an idle GPU above.

## Output check

`logit_dump --paths step,verify` (the verify: KFlash's tree, a chain of 8 live rows) on this build with
`NI_V8_TREE_G6=0` and with the default, against the 27B's M5 floor (`tools/logit_floors/qwen3.8-27b-4bit-mtp.m5.json`)
([output-check/](output-check)): prefill and step rows bitwise identical; verify rows top-1 1.0000 (no flips), KL mean
1.50e-7, p99 1.72e-6, max 2.46e-6, against the decode gate's 4.88e-7 / 3.7e-6: PASS.

## Not run

The validation layer: `logit_dump` under `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1` aborts in the prefill before any
verify (`qattn_t_g6d2s8`, 24 KB static, read as 48 KB; with `NI_QATTN_G6=0`, `attn_prefill11r`, 20 KB, read as 40 KB:
the layer's doubling AGENTS.md describes), and the verify kernel is #897's, unchanged.

## Dropped

[dropped/](dropped): a zone kernel on the matrix units for 6 query heads per KV head (`attn8t_partial_zone_g6`, P as a
half pair) ran the 8-row cycle at 57.4 tok/s at 131K but failed the floor's decode gate (verify KL p99 4.94e-6 against
3.7e-6), and the tree path was faster and inside it.

## Files

`ab/`: the A/B script (`ab_q.py`), command, output (with each request's quiet wait and foreign GPU load), results with
every arm's text and the five servers' logs; `ab-first/`: the first run, its own script.
`trace/`: the `ni_run` commands and outputs (the zone path and the 16-row block on main 1b0b577's code, the forced 8
rows on the dropped kernel's build, whose wide path is main's; the tree path on this branch). `output-check/`, `dropped/`:
as above. Local paths are replaced by placeholders (`~`, `<repo>`, `<run-root>`, `<scratch>`).
