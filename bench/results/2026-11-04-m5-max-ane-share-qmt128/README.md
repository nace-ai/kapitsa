# Results, 2026-11-04: the 27B's Neural Engine share retuned for the 128-row tile, M5 Max

The Qwen3.8 27B splits each prefill FFN between the Neural Engine and the GPU. Without a calibration, the share comes
from `ane::default_units`, from the per-unit costs measured on the M5 Pro with the GPU's share on the matrix units.

The 128-row tile (#959, `NI_QMT128`) runs the GPU's share faster on the M5 Max (`bench_qmm`, the 27B's shapes):
- gate/up 1.195x, down 1.112x, so a unit costs (2/3) / 1.195 + (1/3) / 1.112 = 0.86 of what it did: 1.52 -> 1.30 ms;
- the mixer's projections about 1.15x: 25 -> 22 ms.

With those two constants, the wavefront balance at 40 cores moves from 16 to 14 of the FFN's 34 units, which the sweep
below measured as the best share. The plain split (a noncausal pass, Drex Max) keeps its measured 1.52: it was not
re-measured.

MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), on AC power, under the GPU
lock, the session idle. `ni_run qwen3.8-27b-4bit-mtp @prompt-ids.txt 8 0 --pf-reps 3 --dflash none`: the 4,096 ids of
`bench/quality/ids_qwen38` ([prompt-ids.txt](prompt-ids.txt)), 2,048-row chunks, 45 s idle before each run.

## The sweep

On #959's build, `NI_ANE_UNITS` set per run, two passes in the same order ([sweep/](sweep)). Mean of the six reps
(three a run):

| Units of 34 | Pass 1 reps (ms) | Pass 2 reps (ms) | Mean |
| ---: | --- | --- | ---: |
| 8 | 3,345, 3,463, 3,704 | 3,319, 3,398, 3,650 | 3,480 |
| 12 | 3,086, 3,121, 3,429 | 3,095, 3,137, 3,443 | 3,218 |
| **14** | 3,096, 3,169, 3,411 | 3,087, 3,103, 3,329 | **3,199** |
| 16 (the default before) | 3,571, 3,559, 3,564 | 3,579, 3,631, 3,889 | 3,632 |
| 20 | 4,426, 4,424, 4,552 | 4,411, 4,412, 4,550 | 4,459 |
| GPU alone (`NI_ANE=0`) | 3,928, 4,026, 4,266 | 3,895, 3,954, 4,188 | 4,043 |

## Old against new

[run_ab.sh](run_ab.sh), [run_ab.log](run_ab.log). The two arms ran interleaved, 4 rounds alternating old new / new old,
each from its own Release build:
- **Old:** #959's branch, `b5c43a5`, which takes 16 units.
- **New:** this branch, which takes 14.

| Rep | Old: 16 units (ms, per round) | New: 14 units (ms, per round) | Change |
| --- | --- | --- | ---: |
| 1 | 3,560.6, 3,597.0, 3,587.9, 3,573.9 | 3,112.7, 3,125.3, 3,137.5, 3,132.0 | **-12.7%** (3,579.8 -> 3,126.9) |
| 0 | mean 3,571.8 | mean 3,091.0 | -13.5% |
| 2 | mean 3,706.4 | mean 3,353.2 | -9.5% |

Against main before #959 (#959's own A/B, rep 1 3,853.7 ms), the two together take the 27B's 4K prefill to 3,126.9 ms:
-18.9%.

## Output check

A different share moves which FFN channels run on the Neural Engine's W8A8 program instead of the GPU's 4-bit GEMMs, so
the outputs move. `logit_dump` of the old build, then the new on the reference's prefixes, `tools/logit_compare.py` with
the 27B's M5 floor (`qwen3.8-27b-4bit-mtp.m5.json`, whose gate is set by its Neural Engine source)
([output-check/](output-check)):

| Path | Records | Top-1 | KL mean | KL p99 | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| prefill | 8 | 1.0000, 0 flips | 5.37e-4 | 3.71e-3 | the prefill rows moved, so the floor's gate holds the decode paths |
| step | 192 | 1.0000, 0 flips | 6.76e-5 | 6.75e-4 | PASS (gate 1.18e-4 / 1.93e-3) |
| verify | 192 | 1.0000, 0 flips | 6.86e-5 | 7.12e-4 | PASS |
| batch | 192 | 1.0000, 0 flips | 6.76e-5 | 6.70e-4 | PASS |

The dumps' metadata name the old build `fa0bc00` and the new `b5c43a5`: both are the commits their trees were
configured at. The old tree is #959's (`b5c43a5`, built before its commit), the new is #959 plus this change.

## Not measured

- **Other Macs:** the same constants move the M5 Pro's default from 24 to 22 units and the 20-core M5's from 32 to 31
  (the 4B: 13 to 12, 17 to 16; on the M5 Max it stays at 8). None of those were measured. A calibration
  (`ane_calib --wave --write`) or `NI_ANE_UNITS` overrides the default.
- **The server's time to first token** (`bench/harness`).
