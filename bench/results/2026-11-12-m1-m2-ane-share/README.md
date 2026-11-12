# Neural Engine prefill share on M1 and M2 by default

Before this change an M1 or M2 had no built-in Neural Engine share, so the dense Qwen models (27B, 4B) read prompts
on the GPU alone unless a calibration file or `NI_ANE_UNITS` gave one. `default_units()` now gives M1 and M2 a
share from the same GPU/ANE balance the M4 and M5 use, with costs measured on these chips; `default_min_rows()`
keeps chunks under 1024 rows on the GPU where the plain split would lose (the M1 Ultra), and those chips still
build the 512-row functions so a 1024-2047-row chunk runs the wavefront.

Runs: AWS `mac2-m1ultra.metal` (M1 Ultra, 20 CPU / 64 GPU cores, 128 GB) and `mac2-m2pro.metal` (M2 Pro, 12 CPU /
19 GPU cores, 32 GB), macOS 26.7.2, Apple clang 21. Reference: `origin/main` a47df347 (`main`), built on each box;
candidate: this branch. Models: Kapitsa's `27b` (qwen3.8-27b-4bit) and `4b` (qwen3.5-4b-4bit). One slot, context
65536, prefill chunk 2048, `--idle-unload 0`; every serve inside `lockf -k /tmp/kapitsa-gpu.lock`, one at a time,
legs interleaved (ABBA or ABAB). Splash-M1 1.6.1-m1 (`mlx-community/Qwen3.8-27B-4bit`, its own drafter) ran in the
same pass on the same prompts. Prompts: `bench.py` sweep prompts (cold, nonce-busted); decode is greedy, 400 tokens.
Scripts: `legs520.sh` (one leg per server), `dec32.py` (cold prompts at five corpus offsets). Every leg's line is
in `m1ultra/legs.log` and `m2pro/legs.log`, its JSON in `*/results/`.

Builds in the logs: `main` is origin/main; `pr5` is this branch's head; `pr`, `pr3` and `pr4` are its earlier
steps. On the M2 Pro `pr` and `pr4` take the same paths as `pr5` (smallest split chunk 512, functions from 512).
On the M1 Ultra the rows above come from `pr4` for prompts under 1024 tokens and at 16K (paths `pr5` keeps) and
from `pr5` at 1100-1530 tokens. The M1 Ultra's `pr`, `pr3` and `pr4` are the steps the last three bullets under
"How the defaults were set" fixed, in that order.

## Share sweep (ane_calib, 27B, 2048-row chunks, wavefront)

| Units of 34 | M1 Ultra ms per chunk | M2 Pro ms per chunk |
| ---: | ---: | ---: |
| GPU alone | 7136-7138 | 23405-23424 |
| 12 | 6745 | 19707 |
| 17 / 16 | 5922 (17) | 17643 (16) |
| 20 | **5508** (1.30x) | 15569 |
| 24 | 5703 | 13530 |
| 28 | 5973 | 11507 |
| 30 | 6071 | **10777** (2.17x) |
| 31 | 6227 | 11190 |
| 32 | 6230 | 11689 |
| 33, 34 | ANE compile fails | ANE compile fails |

The defaults land on the measured optima: 20 units on the M1 Ultra, 30 on the M2 Pro. Raw: `m1ultra/calib-*.log`,
`m2pro/calib-*.log`.

## Prompt reading, 27B (time to first token, two legs each side)

| Prompt tokens | M1 Ultra main | M1 Ultra PR | Splash-M1 | M2 Pro main | M2 Pro PR | Splash-M1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ~710 | 2.72 s | 2.72 s | | | | |
| ~812 | 3.02 s | not run | 2.62 s | 9.6 s | 8.0 s | 5.9 s |
| ~955 | 3.59-3.60 s | 3.58-3.59 s | | | | |
| ~1210 | 4.36-4.43 s | 4.00-4.09 s | | | | |
| ~1460 | 5.23-5.24 s | 4.51-4.52 s | | | | |
| ~1590 | 5.67 s | 4.56 s | 4.56 s | 18.3-18.6 s | 9.04 s | 10.9 s |
| 16,315 | 61.2 s (266 tok/s) | 48.4-48.7 s (336) | 47.0 s (347) | 204.2 s (80) | 104.2 s (157) | 119.2 s (137) |

Short-prompt decode is unchanged (M1 Ultra 106.5 tok/s both, M2 Pro 40.8 both). Decode after a 16K prompt, M1
Ultra, 10 requests each side (5 offsets x 2 legs): mean 68.4 tok/s main, 67.4 PR; tokens per draft cycle 3.91 and
3.84. The W8A8 prefill changes the greedy text, and per-request rates spread 55-93 tok/s on both sides.

## Prompt reading, 4B (M1 Ultra, 11 of 18 units)

16,315 tokens: 10.25 s main, 9.40 s PR (8% less). 1,400 tokens: 0.83 s both. ~710 tokens: 0.48 s main, 0.47 s PR
(GPU alone on both).

## Output check (logit_dump, origin/main vs PR)

| Model, Mac | Path | Top-1 flips | Mean KL | p99 KL | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| 27B, M1 Ultra | batch / step / verify | 0 / 0 / 0 of 192 | 5.9e-5 / 5.9e-5 / 6.0e-5 | 9.2-9.5e-4 | PASS |
| 27B, M2 Pro | batch / step / verify | 0 / 0 / 0 of 192 | 1.21e-4 each | 1.06-1.13e-3 | PASS |
| 4B, M1 Ultra | batch / step / verify | 0 / 0 / 0 of 192 | 1.13e-4 each | 2.0e-3 | PASS |

Gates: `tools/logit_floors/qwen3.8-27b-4bit.json`, `tools/logit_floors/qwen3.5-4b-4bit.json`. Raw:
`m1ultra/ld5-27b-compare.txt`, `m1ultra/ld5-4b-compare.txt`, `m2pro/ld-compare.txt`. The main build ran the
GPU alone on both Macs (`NI_ANE: no calibration for Apple M1 Ultra (run ane_calib) and no default share`).

## How the defaults were set

- Share: the wavefront balance in `default_units()` with M1 and M2 costs from `ane_calib` (ANE ms per 512-channel
  unit, the GPU part per unit and the mixer, per layer, 2048 rows). The plain split's ANE cost puts both chips a
  little under their measured wavefront optimum, so the wavefront uses the cost that hits it (M1 3.76 ms, M2
  4.58 ms); the plain split (a noncausal pass) keeps the measured one. Capped at 32 units: 33 and 34 fail ANE
  compilation on both chips.
- Smallest split chunk: a chunk under 1024 rows runs the plain split at the wavefront's share. On the M1 Ultra
  that lost (620-812 tokens: 2.39-3.02 s alone, 3.22-3.63 s split), on the M2 Pro it won (812 tokens 9.6 -> 8.0 s).
  `default_min_rows()` returns 1024 where the 512-row ANE cost at that share exceeds the GPU alone; `ane_calib`'s
  own plain-split fit picks the same (M1 Ultra "chunks >= 1024 rows", M2 Pro 512). Under 1024 tokens the M1 Ultra
  PR runs the GPU alone, as main (~710 and ~955 rows above; ~812 sits between them).
- With 1024 as the smallest chunk the 512-row functions are still built: without them a 1024-2047-row chunk's
  halves had no function and it ran the plain split (1306 tokens: 5.36 s, against 4.69 s on the GPU alone).
- Each wavefront half needs half the smallest function, not half the smallest chunk: otherwise a chunk of
  1024-1151 rows still ran the plain split (a 1207-token prompt: 5.08 s, against 4.0 s with the halves).

Not run: M1, M1 Pro, M1 Max, M2, M2 Max, M2 Ultra (their shares come from the same costs scaled by GPU cores); a
fanless MacBook Air; the Drex Max noncausal pass on M1/M2.
