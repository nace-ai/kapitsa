# Results, 2026-11-03: MacBook Pro M5 Max, 128 GB (Flint's verify attention on the matrix units)

The A/B, the output check and the kernel checks of `flint_attn_t_rows` (`src/kernels/tensor/qattn_t_flint.metal`):
Flint's attention for a pass of 2-8 rows (a prompt-lookup verify, a prefill chunk of at most 8 rows) on the M5's
matrix units, against `flint_attn_partial_rows` (`NI_FLINT_ATTN_T=0`). One build, main `e93901d` with the change.
MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), AC power, High Power mode,
`caffeinate -is`, 2026-11-03 ([machine.json](machine.json)).

## Decode with prompt lookup, temperature 0

Flint at its default but for `--spec 7` (prompt lookup, up to 7 drafts a cycle; off by default), the same two cold
documents from the long corpus (122,621 and 235,892 tokens with the template) and the same question, 512 tokens at
temperature 0, one server per arm, interleaved ([t0/](t0)); plain decoding (no `--spec`) first, for scale. Every arm
wrote the same text at both sizes, plain decoding's. The arms shared an SSD prefix cache, so the 236K document was
prefilled once.

| Context | Plain decoding | `--spec 7`, `NI_FLINT_ATTN_T=0` | `--spec 7`, the default | |
| ---: | ---: | ---: | ---: | ---: |
| 123K | 97.0 tok/s | 34.6, 34.6 (1.19 tokens a cycle) | 84.9, 84.4 (1.32) | 2.45x |
| 236K | 60.6 tok/s | 26.7, 26.7 (1.20) | 59.3, 59.2 (1.20) | 2.22x |

Prompt lookup's verify passes were what made it 2.3-2.8 times slower than plain decoding at these lengths; with this
kernel it is within 2% (236K) and 12% (123K) of plain decoding on this reply, which drafts little (1.2-1.3 tokens a
cycle). Plain decoding does not run this kernel. The script samples no GPU load; the desktop app was running.

## Output check

`logit_dump --paths step,verify` on this build with `NI_FLINT_ATTN_T=0` against the default
([output-check/compare.txt](output-check/compare.txt)): the prefill and step rows every one bitwise identical; the verify
rows (8 a pass) top-1 agreement 0.9896 (2 flips, both near-ties: reference margins 0.117 and 0.047), KL mean 5.40e-3,
p99 9.77e-2, against the M5 floor's decode gate of 0.0218 / 0.192 (`tools/logit_floors/flint-4bit.m5.json`): PASS.

## Kernel

One full-attention layer of Flint's heads (32 query heads over 4 KV heads, head dim 128), the engine's 40 chunks per KV
head, best of 5 ([kernel/](kernel): `fd_check.swift`, the sources assembled by `mkflint.py` and `mksrc.py`):

| Positions | Rows | `flint_attn_partial_rows` | `flint_attn_t_rows` | |
| ---: | ---: | ---: | ---: | ---: |
| 131K | 4 | 3.76 ms | 0.48 ms | 7.9x |
| 131K | 8 | 5.53 ms | 0.52 ms | 10.6x |
| 262K | 4 | 5.78 ms | 0.93 ms | 6.2x |
| 262K | 8 | 11.01 ms | 1.00 ms | 11.0x |

In `ft.out` and `fe.out` the shipped kernel is the one named `flint_attn_t_rows_q1` (Q as half); the one named
`flint_attn_t_rows` there took Q as a half pair (hi + lo), twice the score products for an error against an fp64 softmax
of 4.5e-6 instead of 5.7e-6 at 32K keys, and `_p` padded its tile rows to 136 halves, no faster: both were dropped.
`fw.out` ran the shipped kernel under its own name.

- Against a double-precision softmax (every row, heads 0, 9, 18, 31): 5.7e-6 - 7.7e-6 at 32K keys, 1.6e-5 - 2.0e-5 at 1K,
  at most 1.1e-4 at 37 (`fe.out`, `fw.out`); `flint_attn_partial_rows`'s fp32 products: ~1e-8. What is left is P as
  half, as in Flint's tensor prefill kernel.
- Ragged rows (2, 3, 5, 7, 8 after 37 and 5,000 positions), a 513-key sliding window over a ring of 528 slots (after 100
  to 100,000 positions) and over a full cache: every output finite (`fw.out`).
- 16 KB of threadgroup memory (32-key K and V tiles of 128 halves), which the validation layer still dispatches: the
  verify path under `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`, one window and then all 8, ran with no errors
  ([validation.txt](validation.txt)).

## Files

`t0/`: the decode A/B, its script and the five servers' logs. `output-check/`, `kernel/`: as above. Local paths are
replaced by placeholders (`~`, `<repo>`, `<main>`, `<run-root>`, `<scratch>`, `<dumps>`).
