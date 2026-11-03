# Results, 2026-11-03: MacBook Pro M5 Max, 128 GB (the 27B's drafting block chosen per request)

A DFlash2 drafter loaded with a 16-row block (a one-slot engine's default on the dense Qwen models) now drafts 8-row
blocks for a request whose prompt holds 16,384 tokens or more (`DFlash2::set_block`, `NI_DFLASH_NARROW`,
`NI_DFLASH_NARROW_MIN`), with the budget, rows and depth a drafter loaded with `NI_DFLASH_BLOCK=8` has. Built on #916, so
an 8-row tree verify of the 27B runs `attn8t_partial_tree_g6`. Qwen3.8 27B 4-bit with its DFlash2 drafter and KFlash,
MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), AC power, High Power mode,
`caffeinate -is`, 2026-11-03 ([machine.json](machine.json)).

## Where 8 rows overtake 16

`ni_run` with `NI_DFLASH_BLOCK=16` and `=8` on #916's build, prompts from the long corpus tokenized with the 27B's
tokenizer (`crossover/tok.cpp`), 256 tokens at temperature 0, `NI_CYCLE_TRACE=1` ([crossover/](crossover); the 131K
pair is #916's `trace/block16.out` and `trace/tree-8row.out`):

| Context | 16-row block | 8-row block |
| ---: | ---: | ---: |
| 8,156 | 84.0 tok/s (4.11 tokens a cycle) | 85.0 (3.70) |
| 32,141 | 63.5 (3.92) | 69.5 (3.49) |
| 64,542 | 58.4 (4.25) | 64.9 (3.86) |
| 131,404 | 49.3 (4.55) | 62.2 (4.11) |

A 16-row cycle verifies twice the rows, and its history attention, which grows with the context, costs twice as much;
past ~8K tokens the deeper lattice no longer pays for it. At ~1K the 16-row block keeps its lead (90.6 against 69.3 tok/s,
the measurement behind `KfBudget::pin_wide`), hence a choice per request rather than one block.

## Decode, a one-slot server

Interleaved on this branch's build ([ab/](ab): a first arm only to fill the SSD cache, then `NI_DFLASH_NARROW=0`, the
default, `0`, the default, and last a server loading the drafter with `NI_DFLASH_BLOCK=8`; `run.out` labels that arm
"new", as the script labels any arm without `NI_DFLASH_NARROW=0`): `kapitsa serve --slots 1`, three cold documents from
the long corpus (8,223, 32,373 and 132,037 tokens with the template), the same question, 512 tokens at temperature 0,
thinking off. Every measured arm restored the 131K document from the shared SSD cache and prefilled the other two.

| Context | 16 rows always: tok/s (tokens a cycle) | Default: tok/s (tokens a cycle) | Decode |
| ---: | ---: | ---: | ---: |
| 8K | 94.6, 87.2 (4.53, 4.58) | 87.7, 85.5 (4.58, 4.53) | (the same 16-row path) |
| 32K | 63.9, 64.9 (4.15, 4.22) | 69.9, 69.7 (3.73, 3.73) | +8.4% |
| 131K | 45.4, 46.2 (4.48, 4.56) | 55.8, 56.2 (3.99, 4.02) | +22.3% |

At 8K both arms draft 16 rows (the servers' KFlash lines: 16 rows a cycle), so that row is run-to-run spread, not the
change. The `NI_DFLASH_BLOCK=8` server: 87.5, 67.7 and 55.3 tok/s.

**The text.** Every arm wrote the same text at every size: both "16 rows always" arms, the three default arms (the
warm-up included) and the `NI_DFLASH_BLOCK=8` server (8K: 422 tokens, 32K and 131K: 512). The 8-row block of a drafter
loaded with 16 rows is the drafter loaded with 8 rows, token for token, and at temperature 0 the reply does not depend
on the block.

The script took the GPU lock for the whole run; it samples no GPU load, and the Claude desktop app was running.

## Output check

No kernel and no verify path changes: an 8-row cycle runs the 8-row tree verify (#916's path, whose output check is in
`2026-11-03-m5-max-q27-tree-verify-8`), as a drafter loaded with 8 rows does; the drafts decide which rows are verified,
not what a verified row computes. The texts above are the check.

## Files

`ab/`: the A/B script (`ab_q.py`), command, output, results with every arm's text and the six servers' logs.
`crossover/`: the `ni_run` command and output, and the tokenizer helper. Local paths are replaced by placeholders (`~`,
`<repo>`, `<run-root>`, `<scratch>`).
