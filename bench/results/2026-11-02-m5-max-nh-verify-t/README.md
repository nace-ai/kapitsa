# Results, 2026-11-02: MacBook Pro M5 Max, 128 GB (Nemotron's verify attention on the matrix units)

The A/B and the output check of `nh_attn_t_part[_tree]` (`src/kernels/tensor/nh_attn_t.metal`): a Nemotron-H verify's
attention on the M5's matrix units with the context split in chunks, against `nh_attn_partial[_tree]` (`NI_NH_ATTN_TV=0`).
Lightning 30B-A3B, Super-VL 120B-A12B and Bolt 74B-A9B, one build each run, the two arms interleaved (old, new, old, new),
written by [bench/harness](../../harness/README.md): `compare_engines.py configs/<model>.json sweep-configs/ab2.json
--only <label> --agentic --skip greedy,prefill,tools,agent,cache,sustain --quiet-timeout 60`, each label the model's
default with its context at its limit (`--ctx`), the context sweep's coding answer after a cold document
(`--sweep-task code`, the long corpus of [2026-11-02-m5-max-long](../2026-11-02-m5-max-long/README.md)), one request per
size. MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), AC power, High Power
mode, `caffeinate -is`, 16:33-17:58 local time ([machine.json](machine.json): the builds).

## Decode at long context

Tok/s, the middle of a 512-token answer; each value one cold request:

| Model | Context | Old: decode (two runs) | New: decode (two runs) | Speedup | Prefill, old / new |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lightning 30B-A3B | 33K | 143.2, 164.3 | 234.8, 230.2 | 1.40-1.64x | 5,073 / 4,972 |
| Lightning 30B-A3B | 133K | 65.2, 67.8 | 196.6, 206.4 | 2.90-3.16x | 2,392 / 2,450 |
| Lightning 30B-A3B | 255K | 41.9, 39.7 | 159.4, 173.8 | 3.81-4.38x | 1,497 / 1,545 |
| Super-VL 120B-A12B | 33K | 53.1, 56.3 | 71.9, 72.4 | 1.28-1.36x | 1,127 / 1,178 |
| Super-VL 120B-A12B | 133K | 33.7, 34.9 | 73.6, 67.2 | 1.93-2.18x | 771 / 763 |
| Bolt 74B-A9B | 33K | 84.5, 102.5 | 108.6, 121.3 | 1.06-1.44x | 1,068 / 1,071 |
| Bolt 74B-A9B | 133K | 54.0, 50.3 | 107.5, 118.8 | 1.99-2.36x | 745 / 764 |

Prefill is the same (the change runs in the verify only). With the cheaper cycles KFlash's size rule picks smaller trees
at long context (Lightning at 255K: 4.3 rows per cycle against 5.7, `logs/`), so tokens per cycle fall a little while
tokens per second rise.

## Short context

Lightning at its default 32K context, the decode phase's 85-token prompt (`sweep-configs/ab3.json`, three interleaved
pairs, 5 answers each): T 0.6 old 284.8 / 293.2 / 286.9, new 263.3 / 283.5 / 278.0 tok/s; greedy old 278.9 / 280.4 /
289.4, new 279.5 / 270.8 / 291.5. A verify cycle costs the same (the server logs: old 12.10 / 12.31 / 11.96 ms, new
12.00 / 12.17 / 12.19); the T 0.6 difference is the tokens per cycle of different sampled texts (old 3.34-3.48, new
3.23-3.42), and greedy is level.

## Output check

`logit_dump` of main `7ebcf73` against this branch's, `tools/logit_compare.py` at twice each model's floor
(`tools/logit_floors/<model>.m5.json`, measured on main): [output-check/](output-check). Prefill, step and batch: every
row bitwise identical; verify: 0 of 192 rows bitwise, top-1 1.0000, 0 flips, KL mean / p99 4.51e-5 / 8.67e-4 (Lightning;
gate 9.95e-5 / 1.86e-3), 1.85e-5 / 3.66e-4 (Super-VL; 5.41e-5 / 8.13e-4), 2.24e-5 / 4.94e-4 (Bolt; 8.03e-5 / 1.31e-3):
PASS on all three.

## Contention

Every run is contention-flagged: the Claude desktop app's GPU process (`Claude Helper`), driving the session, held 8-47%
of the GPU (the worst phase's mean) while the agent worked, and the 60 s quiet wait timed out before most runs. Both arms
ran under it, interleaved.

## Files

- `results/`, `logs/`: every run (`*-ab-old`: `NI_NH_ATTN_TV=0`, `*-ab-new`: the default; `lightning-short-*`: the
  short-context pairs) and its server log.
- `output-check/`: `logit_compare.py`'s report per model.
- `session/`: the runner's console output and scripts. `sweep-configs/`: the labels, passed after the model's config.

Local paths are replaced by placeholders: `~` for the home folder, `<repo>` for the Kapitsa checkout, `<main>` for main's
checkout, `<run-root>` for the harness's output folder, `<dumps>` for the logit dumps, `<host>` for the host name.
