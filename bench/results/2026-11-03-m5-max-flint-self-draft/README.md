# Results, 2026-11-03: MacBook Pro M5 Max, 128 GB (Flint drafting with itself at long contexts)

Flint's self-drafts (MagicDec, arXiv 2408.11049: the model as its own draft, reading a short window of its KV), on by
default on an M5-class GPU from 180K-token prompts: a cycle runs 2 decode steps whose 14 full-attention layers read only
the last 4,096 positions (`Flint::encode_draft`; the sliding-window layers as always), each sampled with the draw its
verify row will make (`sample_stage2` with the same seed and step), then verifies them with the pending token in one
3-row pass over the whole cache (`flint_attn_t_rows`, #908). At 236K tokens a plain step reads ~6.6 GB of KV for one
token; the verify reads it once for up to three. Against `NI_FLINT_SELF=0` (plain decoding, as before). Flint 40B
(MLX 4-bit), MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), AC power, High
Power mode, `caffeinate -is`, 2026-11-03 ([machine.json](machine.json)).

## Decode

Interleaved on the final build ([ab/](ab): off, default, off, default): one server per arm, the same two cold documents
from the long corpus (187,317 and 235,892 tokens with the template) and the same question, 512 tokens at temperature 0
and at the model's sampling (T 1.0, top-p 0.95, top-k 20, seed 7). The arms shared an SSD prefix cache (the 236K
document restored by every arm, the 187K one prefilled by each).

| Context | Sampling | Off: tok/s | Default: tok/s (tokens a cycle) | |
| ---: | --- | ---: | ---: | ---: |
| 187K | T 0 | 70.3, 69.5 | 77.5, 76.3 (2.30) | +10% |
| 187K | T 1.0 | 68.9, 68.6 | 74.4, 74.1 (2.28) | +8% |
| 236K | T 0 | 60.7, 59.6 | 77.0, 76.6 (2.43) | +28% |
| 236K | T 1.0 | 60.0, 59.2 | 64.6, 64.7 (2.07) | +8% |

The script took the GPU lock for the whole run; it samples no GPU load, and the Claude desktop app was running.

**Why 180K and 2 drafts** ([explore/](explore), one server per arm, not interleaved): at 123K tokens 2 drafts were
slower than plain decoding (92.3 / 91.2 -> 88.5 / 84.6 tok/s at T 0 / T 1.0), at 187K faster (68.7 / 68.2 -> 75.7 /
73.8). 3 drafts accepted more (2.95 tokens a cycle at 236K, T 0) but cost a draft step and a verify row each, no faster
than 2 at 236K and slower below; 4 slower still; a 16,384-position window slower than 4,096; prompt lookup in front of
the self-drafts (`NI_FLINT_NGRAM=7`) slower than the self-drafts alone. A cycle costs about one plain step plus ~6.5 ms
a draft at 236K. `run1.out` compared the replies' content only, which was empty (Flint reasons for all 512 tokens), so
its "texts equal" line says nothing; `run2.out` and `ab/` compare the reasoning and the content.

## The replies

Each build writes the same text every time (both "off" arms alike, both "default" arms alike, at every size and
sampling). The default's text leaves plain decoding's at one word in each reply (187K, T 0, character 1,114:
"AudioFlamingo3, etc." against "ALBERT, ALIGN, etc."; 236K, T 0, character 734: "multimodal model" against
"vision-language model"; at T 1.0 at characters 410 and 71) and goes on coherently from there; the margins at those
words were not measured. A decode step and a multi-row verify differ in the order their sums run, which flips a token
where two are nearly tied: Flint's floor holds that as noise (its "step" source, plain decoding against the 8-row
verify), and prompt lookup (`--spec K`) has always run with it.

## Output check

This change moves no kernel: the drafts only decide which tokens a verify checks, and the verify is the one prompt lookup
runs. What it changes is which path writes a long reply's tokens: the 3-row verify instead of the decode step. So the
check is `logit_dump --paths step,verify` of this build with `NI_FLINT_VERIFY_ROWS=3`, the plain step as the reference
and the 3-row verify as the candidate, against Flint's M5 floor (`tools/logit_floors/flint-4bit.m5.json`; measured on
an M5 Pro at 0.2.7), and the same with the 8-row verify, the floor's own "step" source, on this Mac
([output-check/](output-check)):

| Rows | Verify | Top-1 | Flips (near-ties) | Largest flip margin | KL mean | KL p99 | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 192 | 3 rows | 0.9792 | 4 (4) | 0.30 | 7.02e-3 | 9.60e-2 | FAIL: top-1 |
| 768 | 3 rows | 0.9805 | 15 (15) | 1.09 | 7.54e-3 | 1.15e-1 | FAIL: top-1 |
| 768 | 8 rows | 0.9805 | 15 (15) | 1.09 | 7.32e-3 | 1.27e-1 | FAIL: top-1 |

The gate: top-1 at least 0.9844, KL mean / p99 at most 0.0218 / 0.192, no flip at a margin of 6.55 or more. **It
fails on top-1 agreement**: 15 of 768 rows flip, every one a near-tie (the largest at a margin of 1.09); KL passes.
The floor file records the same comparison, plain decoding against the 8-row verify (its "step" source), at top-1
0.9922 (6 flips in 768 rows) on an M5 Pro at 0.2.7, which passes. Re-run here, on the M5 Max at cd20cf5, that
comparison (the 8-row verify `--spec K` runs) gives 0.9805 with 15 flips, as the 3-row verify does. That re-run is a
measurement of this Mac and build, not the floor, which is left as it is (a re-measure is a PR of its own): the gate as
recorded is failed.

## Files

`ab/`: the A/B's script (`ab_self.py`), command (`run.sh`), output (`run.out`), results with every text
(`ab_self.json`) and the four servers' logs. `explore/`: the two exploratory runs' commands and outputs (run1 on a
build that committed the cycle as one command buffer, run2 on one that commits each draft as it is encoded, as the
final build does). `output-check/`: the comparisons. Local paths are replaced by placeholders (`~`, `<repo>`,
`<run-root>`, `<scratch>`).
