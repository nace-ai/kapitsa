# Results, 2026-11-02: MacBook Pro M5 Max, 128 GB (the 27B's wide tree verify attention on the matrix units)

The A/B and the checks of `attn8t_partial_tree_g6` (`src/kernels/tensor/v8attn_t.metal`): Qwen3.8 27B's 16-row wide
KFlash tree verify's history attention on the M5's matrix units, against `attn8m_partial_tree` (`NI_V8_ATTN_T=0`).
MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), AC power, High Power mode,
`caffeinate -is`, 23:01-23:33 local time ([machine.json](machine.json): the builds).

## Decode

Interleaved on one build (old, new, old, new; `compare_engines.py configs/kapitsa.json sweep-configs/ab7.json --only
<label> --agentic --skip greedy,prefill,tools,agent,cache,sustain --quiet-timeout 60`, the 27B's default with `--ctx
262144`, the coding answer after a cold document from the long corpus, one request per size). From the server logs:

| Context | Old: tok/s (ms per verify cycle) | New: tok/s (ms per verify cycle) |
| --- | ---: | ---: |
| 33K | 51.4 (71.3), 60.8 (70.1) | 69.9 (63.0), 71.1 (62.5) |
| 131K | 34.6 (132.0), 39.7 (133.0) | 40.3 (100.7), 55.6 (98.8) |

A verify cycle (16 rows) costs 11% less at 33K and 25% less at 131K; tokens per cycle vary with the sampled text (3.7-5.5
on both arms), which is why the tokens per second spread. Prefill is the same.

## Numerics

- One layer of the 27B's heads, 16 rows as two 8-row tiles, random Q / K / V and trees, against a double-precision softmax
  over each row's history and ancestors (every row and head, 3 to 65,536 positions): the new kernel within 1.2e-7 -
  1.5e-6 of `attn8m_partial_tree` and as far from fp64 as it (3.8e-5 - 1.3e-3 both). P is a half pair (hi + lo), so only
  the order of the sums differs from `attn8m`'s fp32 P; with P as half alone the difference was 8e-6 - 4.4e-4.
- Temperature-0 continuations ([greedy/](greedy)): the 27B served with its DFlash2 drafter (16 rows per cycle, the wide
  verify), the same 4K- and 32K-token documents and question, 256 tokens: identical texts with `NI_V8_ATTN_T=0` and the
  default (1,204 and 1,167 characters).
- `logit_dump` ([output-check.txt](output-check.txt), main `a2f4210` against the branch, gated at the 27B's M5 floor, #895):
  every row bitwise identical on every path. Its verify path runs 8-row tree verifies, which on the 27B take the zone
  kernel (`attn8m_partial_zone`), not the wide one: this check does not reach the change. (As `output-check.txt` and
  `machine.json` say, it ran main `b6345da`, not `a2f4210`, against the branch's earlier build `24a747c`, with P as half
  alone.)
- The wide verify's own gate, added on 2026-11-03 after the merge ([output-check-wide/](output-check-wide)): `logit_dump
  --verify-rows 16` and `--verify-rows 32` (a KFlash tree of a chain of 16 or 32 live rows, the wide tree verify the
  27B's DFlash2 block runs; `--rows 32`, 256 records), `NI_V8_ATTN_T=0` against the default on one binary, main
  `32cf851` (the merged kernel) with the `logit_dump` that adds the flag. Verify rows: 0 of 256 bitwise (the candidate
  ran `attn8t_partial_tree_g6`), top-1 1.0000, 0 flips, KL mean / p99 1.73e-7 / 2.21e-6 at 16 rows and 2.25e-7 /
  1.59e-6 at 32, against the decode gate of 4.88e-7 / 3.7e-6 (twice the floor): PASS on both. The prefill rows are
  bitwise identical. `compare8.txt`: main's `logit_dump` against the flag's at the default 8 rows, every row bitwise
  identical.

## Contention

Every run is flagged for contention in its JSON (`contention`). The Claude desktop app's GPU process, driving the
session, held a mean of 38.4-45.7% of the GPU during the sweep (old 45.7% and 41.4%, new 44.7% and 38.4%; up to 62% at a
sample) and 6.2-10.5% while the decode was timed, and the 60 s quiet wait timed out before every run. Both arms ran under
it, interleaved, at similar loads; the per-cycle costs in the table are the server logs' own and vary less with it than
the tokens per second.

## Files

`results/`, `logs/`: the A/B runs (`q27-v8t-old`: `NI_V8_ATTN_T=0`; `q27-v8t-new`: the default) and their server logs.
`greedy/`: the continuations, both servers' logs and the script. `output-check-wide/`: the wide verify's gate (the
comparisons, each dump's `meta.json`, the scripts and their output; the logits themselves are not kept, 254 MB a dump). `session/`, `sweep-configs/ab7.json`: the runner and the
labels. Local paths are replaced by placeholders (`~`, `<repo>`, `<main>`, `<run-root>`, `<dumps>`, `<scratch>`, `<host>`).
