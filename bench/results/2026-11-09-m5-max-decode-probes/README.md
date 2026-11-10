# Results, 2026-11-09: MacBook Pro M5 Max, 128 GB (Qwen decode: where a cycle goes, what was tried)

A day of decode work on the two Qwen cells, Qwen3.8-27B (the catalog's `qwen3.8-27b`, KFlash with the 16-row-block
DFlash2 retrain, nace-ai/Qwen3.8-27B-DFlash2-Kapitsa) and Qwen3.6-35B-A3B (`qwen3.6-35b-a3b`, KFlash with an 8-row
tree), aiming at +15% single-stream decode with the same tokens. Nothing reached it. What paid went into two pull
requests (#1130, #1131); everything else is recorded here so that a retry starts from these numbers.

MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), AC power, High Power mode,
under the GPU lock ([machine.json](machine.json)). Probes are patches on main 888a5679 behind switches that are off by
default ([probes.diff](probes.diff): `NI_V8_PF`, `NI_V8_MOE_R1`, `NI_V8T_STAGE=2`, `NI_DFL_NT`, `V8Knobs::dfl` bit 256, the
knob `gpf`, `BENCH_REPLAY_LAYER`), so each A/B runs one binary. `NI_PROFILE=1` sums isolated dispatches: it ranks kernels, it is
not decode speed. `ni_run` per-cycle A/Bs (`NI_AB=<knobs>`: odd cycles with the knobs) compare two settings under the
same GPU conditions; cell runs are `bench/harness/compare_engines.py configs/kapitsa.json --only <label> --agentic
--skip tools,agent,cache,sustain --quiet-timeout 60`.

## The display: an intermittent measurement condition

The machine was locked with the Aerial wallpaper. Under `caffeinate -dis` (the display held on) the lock screen keeps
decoding and drawing its video on the GPU (`WallpaperAerialsExtension`, `VTDecoderXPCService`), which the harness's
foreign-GPU monitor does not count. Same binary (origin/main 888a5679), 27B cell, interleaved (on, off, on, off; the
display asleep through `caffeinate -is` and `pmset displaysleepnow`):

| Display | decode T 0.6 (median of 5) | greedy | 4K TTFT | sweep decode 1K / 4K / 31K |
| --- | --- | --- | --- | --- |
| on | 150.5, 149.8 | 147.4, 153.1 | 4.12, 3.75 s | 88 / 77 / 65; 99 / 85 / 70 |
| asleep | 155.9, 145.9 | 156.2, 151.7 | 3.70, 3.79 s | 89 / 81 / 77; 112 / 83 / 68 |

One of the two display-on legs ran 5-10% slow in every phase, and the other did not. Two display-on `ni_run` legs that
day ran twice as slow in every phase. The morning's display-on baseline (below: 141-144 greedy) sits under all four
legs on the same code, so the display is not all of that difference: the agent's own activity in the desktop app,
and the KFlash cache's state, also differed. Every A/B here compares arms under one condition. The PRs' runs after
19:00 had the display asleep.

## Baseline (origin/main 888a5679, display on)

| Cell | decode T 0.6, median of 5 (runs) | greedy | sweep decode 1K / 4K / 31K |
| --- | --- | --- | --- |
| 35B (`q36-35b-kapitsa`) | 324.8, 330.6 | 370.5, 405.9 | 259 / 256 / 188; 251 / 248 / 169 |
| 27B (`q38-27b-kapitsa`) | 142.2, 140.2 | 141.3, 143.6 | 84 / 62 / 66; 87 / 79 / 66 |

Single requests at T 0.6 ranged 307-342 (35B) and 110-149 (27B) within a run: cells cannot resolve a change of a few
percent; the per-cycle `ni_run` A/Bs do.

## Where a cycle goes

`ni_run` with `NI_CYCLE_TRACE=1`, `bench/prompts/code_<model>.txt` (85 tokens), 512 tokens greedy, display on. The GPU
never waits for the host: the next cycle's drafting half is committed behind each verify, and the host encodes a
verify in 0.2-0.3 ms while the drafting half runs.

| | drafting pass | verify | tokens a cycle | tok/s |
| --- | --- | --- | --- | --- |
| 35B (8-row trees) | 1.32 ms | 9.91 ms | 3.65 | 324.6 |
| 27B (16-row block) | 4.70 ms | 38.64 ms | 6.81 | 156.9 |

`NI_PROFILE` of the real tree cycles (`ni_run`, 48 tokens), ms a cycle of isolated dispatches:

- 27B, short prompt (43.2): the verify's `qmm8t2p_b4_nt16` 31.0 (the GEMMs, ~465 GB/s, the same at 8 and 16 rows:
  memory-bound), the drafter's `qmm8t2dp_b4_nt32` 3.73 (~1.1 GB at ~300 GB/s), `gdn_step_verify8s_tree` 2.58 (54 us a
  layer at 16 rows, 28.5 at 8), the LM head 1.36, `x8_prep2` 1.13 (145 calls), `gdn_prep_verify8w_tree` 1.11, the tree
  attention 0.51.
- 27B at 32K (the 8-row block it drafts from 16K tokens): `attn8t_partial_tree_g6` 5.82 (364 us a layer, ~370 GB/s of
  KV); the 8-row drafter on the target's MMA kernels (`qmm8snb`) 2.46.
- 35B, short prompt (11.9): `moe8e_gu_r2` 3.01 + `moe8e_down_r2` 1.50, the dense GEMMs 3.8 (`qmm8t` in_proj 39 us for
  14.2 MB, out / o 20 us for 4.7 MB), the DeltaNet step 0.63, `moe8_plan` 0.44, the router 0.38, the zone attention
  0.27 + 0.15.
- 35B at 32K / 128K: `attn8ts_partial_zone_g8` 1.93 / 7.68 (193 / 768 us a layer, ~350 GB/s of KV), the reduce 0.37 /
  0.46.

The verify's time per cycle against the context (35B: 8.2 / 11.2 / 18.0 ms at 4K / 32K / 128K; 27B, 8-row block: 48.1
/ 68.2 ms at 32K / 128K) is the attention's; those `ni_run`s ran with a locked KFlash calibration (below), so their
tokens a cycle are not the drafter's.

## Probes

| Probe | Result | Outputs |
| --- | --- | --- |
| The system-level cache ([slc_probe.cpp](slc_probe.cpp)) | a re-read of 12-48 MB ~900 GB/s, of <= 8 MB ~1.8 TB/s, against ~550 cold; 64 MB and up: DRAM. A prefetch beside a latency-bound chain hides 35-65% of the next read (8 MB: 12 of 18 us; 32 MB: 24 of 62); after >= 48 MB of other reads the prefetched lines are gone | - |
| Weight prefetch in the verify (`NI_V8_PF`): `v8_touch` reads the out / o projection (and the MoE's router and shared expert) beside the DeltaNet step / the attention, on a grouped encoder | lost: the grouped encoder alone +0.18-0.23 ms a cycle (35B and 27B); with the prefetch +0.26 (35B), +0.7 (27B): it slows the step it is paired with more than it saves | bitwise |
| `NI_KFLASH_DFS=1` (depth-first tree rows) on the M5, 27B 16-row | 40.79 vs 40.84 ms a cycle: no change | the same tokens |
| `NI_V8_TREE_FUSE=1` (tree zone + reduce in one kernel) on the M5, 35B | 8.69-8.74 vs 8.62-8.72 ms: slower | bitwise |
| The exact MoE passes' one-row tiles run one row's chains (`NI_V8_MOE_R1`, `moe8e_*_r2o`) | -0.06 ms a verify (`bench_verify8` A/B), replay neutral: not ALU-bound | bitwise |
| The MoE passes alone (`BENCH_REPLAY`), 35B: 34.1 distinct routed experts a layer, 55 two-row tiles | gate/up 4.17 ms per 40 layers (~400 GB/s), 3.42 with one layer's weights warm (-18%); down 2.08, 1.39 warm (-33%): gate/up is mostly not waiting for DRAM | - |
| The 35B zone attention with K staged alone (16 KB, V from the cache; `NI_V8T_STAGE=2`) | 247 vs 230 us a layer at 32K, 963 vs 881 at 128K: slower | bitwise |
| The 27B drafter's GEMMs on 16-row matrix-unit tiles, its 8-row block on its own `qmm8t` GEMMs | drafting pass 4.41 -> 3.96 ms (16-row), 3.42 -> 3.08 (8-row); 27B cycle -0.57 ms (-1.4%) | the same tokens: **#1130** |
| KFlash calibration lock-out | a cache whose position 2 fit to (-8, 0.1) kept the 35B at 223 tok/s (1.92 tokens a cycle) on every later request; unlocked, 345 on the next | the same tokens: **#1131** |
| The tree DeltaNet step requests the next row's inputs ahead (`gdn_step_verify8s_tree_pf`, knob `gpf`) | -0.006 / +0.048 ms a cycle (35B), +0.16 / +0.19 (27B, 16 rows): not waiting on its loads | bitwise |
| `pad=1` (an empty dispatch after every verify GEMM, ~200 a verify) | +0.07 / -0.04 ms (27B), -0.01 / -0.08 (35B): a dependent launch costs ~nothing; a fused pair saves only the second kernel's own latency | - |

The `NI_AB=skip=...` breakdowns are clean on the 27B's drafting pass (gate/up 1.80, down 0.93, the reduced head
0.80, projections 0.87, ring commit 0.58, attention 0.26 ms) and not on the verifies: a skipped part changes the rows,
hence the MoE's routing and the trees' sizes, so the A and B cycles do different work.

## KFlash's calibration can lock itself out (#1131)

The budget fits each tree position's calibration from the nodes its trees verify. A fit pessimistic enough to keep a
position out of every tree gets no record to correct it, and the cache (`~/.cache/kapitsa/kflash/`) carries it into
every later process. Here one prompt (a document the 35B began to repeat, which the drafter could not continue) fit
position 2 to the clamp, (-8.00, 0.10). From then on every request, code included, decoded with 2-row trees:

| From the poisoned cache | first request | second request |
| --- | --- | --- |
| origin/main | 223.0 tok/s, 1.921 tokens a cycle | 225.5, 1.921 |
| #1131 | 307.5, 2.954 (recovering) | 345.3, 3.650 |

With a healthy cache both builds run the same (35B 346.7 / 349.2 against 349.3 / 348.2 tok/s, 27B 171.4 / 171.4).

## Not tried, and why

- Smaller or different weights, drafter retraining, the online drafter's history lane (#912): out of scope.
- Wider 27B trees (32 rows): the verify GEMM is memory-bound at 16 rows, but the DeltaNet step, preps and attention
  scale with the rows (~+5 ms a cycle for 16 more), more than the ~0.5 token a cycle wider trees would add.
- A two-row-tile-free (RT 4) MoE pass, register-pipelined attention, repacked GEMM layouts: measured before
  (`bench/results/2026-11-03-m5-max-qwen-probes`, and the notes in `src/kernels/verify8.metal`).
