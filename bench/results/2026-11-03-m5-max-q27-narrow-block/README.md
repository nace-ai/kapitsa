# Results, 2026-11-03: MacBook Pro M5 Max, 128 GB (the 27B's drafting block chosen per request)

A DFlash2 drafter loaded with a 16-row block (a one-slot engine's default on the dense Qwen models) now drafts 8-row
blocks for a request whose prompt holds 16,384 tokens or more (`DFlash2::set_block`, `NI_DFLASH_NARROW`,
`NI_DFLASH_NARROW_MIN`), with the budget, rows and depth a drafter loaded with `NI_DFLASH_BLOCK=8` has: by default on
the M5 Max, where it was measured (`NI_DFLASH_NARROW=1` on the other Macs a 16-row block runs on). Built on #916, so
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

Interleaved on this branch's final build ([ab/](ab): a first arm only to fill the SSD cache, then `NI_DFLASH_NARROW=0`,
the default, `0`, the default, and last a server loading the drafter with `NI_DFLASH_BLOCK=8`; `run.out` labels that arm
"new", as the script labels any arm without `NI_DFLASH_NARROW=0`): `kapitsa serve --slots 1`, three cold documents from
the long corpus (8,223, 32,373 and 132,037 tokens with the template), the same question, 512 tokens at temperature 0,
thinking off. Every measured arm restored the 131K document from the shared SSD cache and prefilled the other two.

| Context | 16 rows always: tok/s (tokens a cycle) | Default: tok/s (tokens a cycle) | Decode |
| ---: | ---: | ---: | ---: |
| 8K | 99.9, 100.0 (4.58, 4.58) | 100.1, 98.8 (4.58, 4.53) | (the same 16-row path) |
| 32K | 79.7, 80.5 (4.19, 4.22) | 85.9, 85.6 (3.70, 3.70) | +7.1% |
| 131K | 56.2, 56.1 (4.56, 4.56) | 70.3, 70.1 (4.02, 4.02) | +25.0% |

At 8K both arms draft 16 rows (the servers' KFlash lines: 16 rows a cycle). The `NI_DFLASH_BLOCK=8` server: 98.9, 83.7
and 69.8 tok/s.

**GPU load.** Each request waited for a quiet GPU (no foreign process above 5% for 15 s, at most 60 s:
`bench/harness/gpu_hogs.py`) and sampled the foreign GPU load every 2 s while it ran (`run.out`, `ab_q.json`'s "load").
Nothing else ran, but the Claude desktop app's helper process, which this session runs in, held 2.7-4.6% mean (9.2% at
most) through the four measured arms alike, so most quiet waits ran out at 60 s; no request reached the harness's
contention flag (a mean above 5%, or three samples in a row above it), and the `NI_DFLASH_BLOCK=8` arm ran at 0.0%.

**Earlier runs.** [ab-second/](ab-second) (the same arms and load sampling on the branch before its default was scoped
to the M5 Max, the same code on this Mac): 79.8 / 80.4 -> 86.9 / 86.9 tok/s at 32K and 56.2 / 56.3 -> 70.1 / 70.0 at
131K, 2.1-4.5% foreign load, none flagged. [ab-first/](ab-first) (no load sample, the desktop app in use): 63.9 / 64.9 ->
69.9 / 69.7 at 32K and 45.4 / 46.2 -> 55.8 / 56.2 at 131K.

**The text.** In every run every arm wrote the same text at every size: the "16 rows always" arms, the default arms (the
warm-ups included) and the `NI_DFLASH_BLOCK=8` server (8K: 422 tokens, 32K and 131K: 512). The 8-row block of a drafter
loaded with 16 rows is the drafter loaded with 8 rows, token for token, and at temperature 0 the reply does not depend
on the block.

## Output check

No kernel and no verify path changes: an 8-row cycle runs the 8-row tree verify (#916's path, whose output check is in
`2026-11-03-m5-max-q27-tree-verify-8`), as a drafter loaded with 8 rows does; the drafts decide which rows are verified,
not what a verified row computes. The texts above are the check.

## Files

`ab/`, `ab-second/`, `ab-first/`: each run's A/B script (`ab_q.py`; the first run's before it sampled the load),
command, output, results with every arm's text and the six servers' logs.
`crossover/`: the `ni_run` command and output, and the tokenizer helper. Local paths are replaced by placeholders (`~`,
`<repo>`, `<run-root>`, `<scratch>`).
