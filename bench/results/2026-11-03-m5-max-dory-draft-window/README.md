# Results, 2026-11-03: MacBook Pro M5 Max, 128 GB (a 4K window for Dory's self-drafts' output layers)

Dory's self-draft (`Dory::encode_draft`: the model with one loop, then its 16 output layers and the head) with the two
full-attention layers among its output layers reading only the last 4,096 positions (`NI_DORY_DRAFT_WIN`, the default
on an M5-class GPU), against every position (`NI_DORY_DRAFT_WIN=0`, as before). The verify writes those layers' keys at
the drafted positions again before any row reads them, and reads none of the draft's, so the reply is the same; only
the drafts, and so the tokens a cycle, change. Dory1.0 4B (NVFP4) at its default (2 loops, per-loop KV, self-drafts),
MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), AC power, High Power mode,
`caffeinate -is`, 2026-11-03 ([machine.json](machine.json)).

## Decode at temperature 0

Interleaved on one build ([t0/](t0): full, 4,096, 1,024, full, 4,096, 1,024): one server per arm, the same two cold
documents from the long corpus (134,348 and 259,404 tokens with the template) and the same question, 512 tokens at
temperature 0. The arms shared an SSD prefix cache: the 259K document was restored by every arm, the 134K one prefilled
by each. The six arms wrote the same text at both sizes (`run.out`'s last line says "four arms": the script is #906's
with the arms made a list, and it compares all six).

| Context | Every position: tok/s (ms a cycle) | 4,096: tok/s (ms a cycle) | 1,024: tok/s (ms a cycle) |
| ---: | ---: | ---: | ---: |
| 134K | 47.0 (52.6), 45.9 (53.8) | 48.7 (50.4), 47.6 (51.7) | 48.2 (51.0), 47.1 (52.1) |
| 259K | 27.2 (88.5), 26.5 (91.0) | 28.2 (83.5), 27.4 (86.0) | 27.9 (84.3), 27.3 (86.4) |

Tokens a cycle: 2.47 / 2.41 with every position, 2.46 / 2.35 with either window (134K / 259K). The 4,096 window took
both pairs at both sizes: decode 3.7% and 3.5% faster on average, a cycle 2.2 ms and 5.0 ms cheaper, for slightly
fewer accepted drafts at 259K. The script took the GPU lock for the whole run; it samples no GPU load, and the Claude
desktop app was running.

## Output check

The reply cannot change: the verify (`encode_verify_drafted`) writes the output layers' keys and values at every row
it verifies before its attention reads them, so no row of it reads a key the draft wrote, and the drafts only decide
which rows are verified. The texts above are identical across the six arms at both sizes. `logit_dump` runs no
self-draft (its step and verify paths are untouched by this change), so it has nothing to compare here.

## Files

`t0/`: the script (`ab_dw.py`), the command (`run.sh`), its output (`run.out`), the results with the reference text
(`ab_dw.json`) and the six servers' logs. Local paths are replaced by placeholders (`~`, `<run-root>`).
