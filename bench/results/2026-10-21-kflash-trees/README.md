# Results, 2026-10-21: KFlash trees (#187) against main before them

KFlash ([docs/kflash-map.md](../../../docs/kflash-map.md)) replaces the drafter's chain with a best-first tree of its
candidates, verified exactly in the same 8-row pass and sized by a budget measured on each Mac. These are the runs behind
[KFlash trees](../../../docs/benchmarks.md#kflash-trees-m4-pro-2026-10-21) in docs/benchmarks.md, on a MacBook Pro (M4 Pro,
48 GB, `Mac16,7`, macOS 27.0.1, High Power mode, on AC power): two Release builds side by side, main before the trees at
617f6a2 ("old main") and the `kflash-trees` branch at 72a61e3, whose tree is main after #187 (8b190f5, "new main"). Times
are local (UTC+5). Every run held the GPU lock (`~/.cache/nace-gpu.lock`); no other job used it.

Weights: the `kapitsa pull` folders of Qwen3.5-4B, Qwen3.8-27B and Qwen3.6-35B-A3B at 4-bit, each with the drafter it
installs next to the model (`dflash2/`: z-lab's DFlash for the 4B, DFlash2 for the 27B and the 35B). Nemotron: Nemotron 3.5 Lightning 30B-A3B and Nemotron 3.5 Bolt 74B-A9B, NVIDIA's checkpoints as `kapitsa pull` fetches them, with their drafters (DSpark on Lightning, LiLiCorr on Bolt); KFlash verifies trees of those drafters' lattices.

## Summary

| Model | Decode, T = 0.6: old → new main | Greedy: old → new main | Per-cycle A/B, greedy | Per-cycle A/B, T = 0.6 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | 185.7 → **200.8** (+8.1%) | 193.8 → **213.7** (+10.3%) | +2.1% (6 prompts) | +9.6% (3 prompts) |
| Qwen3.8-27B | 60.4 → **60.5** (+0.2%) | 61.1 → **64.4** (+5.5%) | +6.0% (3 prompts) | +2.8% (2 prompts) |
| Qwen3.6-35B-A3B | 168.1 → **160.6** (-4.4%) | 177.6 → **177.2** (-0.2%) | +0.6% (5 prompts) | — |
| Nemotron 3.5 Lightning 30B-A3B | 104.3 → **111.7** (+7.1%) | 115.7 → **113.7** (-1.7%) | — | — |
| Nemotron 3.5 Bolt 74B-A9B | 64.3 → **68.8** (+6.9%) | 69.4 → **71.2** (+2.6%) | — | — |

The replayed opencode session on the 35B: 51.7 s on old main, 51.4 s on new main (-0.6%; writing rate 138.7 → 140.8 tok/s, +1.5%).

## The published protocol

[bench/harness](../../harness/README.md) through `kapitsa serve`, as in the README's tables, one run per build and model,
old main first: 5 runs of the 512-token coding answer at T = 0.6 and 5 greedy, tool calls, peak memory (`--skip
prefill,agent`: the trees do not touch the prefill), and the replayed opencode session on the 35B (the README's agent
row; new main first). Configs: [kflash-merge.json](kflash-merge.json), the stock `q35-4b-kapitsa`, `q38-27b-kapitsa` and
`q36-35b-kapitsa` on the `kapitsa pull` folders, one per build, with three extra warm-up decodes on both (`adaptive_warm`:
KFlash's budget adapts across requests, as the harness does for every engine whose speculation adapts). Result files:
`<label>-<time>.json`, where `<main-checkout>` and `<kflash-trees-checkout>` stand for the two builds' worktrees,
`<models>` for the weights folder and `<out>` for the run folder. One run per build is a quick reading: the same build's
runs on this Mac have spread 3-8% at T = 0.6 (169-183 tok/s on the 4B, 2026-10-19), so the per-cycle A/B below is the
measurement to trust for small differences.

| Model | Build | Result file | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Tool calls | Peak memory (GB) | Flagged |
| --- | --- | --- | ---: | ---: | :---: | ---: | --- |
| Qwen3.5-4B | old main (617f6a2) | `q35-4b-kapitsa-main-20261021-070239.json` | 185.7 (174.6–191.6) | 193.8 (174.6–203.3) | pass | 4.2 | CONTENTION: sustained foreign GPU load > 5% in ['tools'] (mean 2.41%, top ['Claude Helper']) |
| Qwen3.5-4B | new main (8b190f5) | `q35-4b-kapitsa-trees-20261021-071011.json` | 200.8 (172.7–210.2) | 213.7 (191.5–217.6) | pass | 4.2 | repetition loop in 1 of 5 sampled runs; CONTENTION: sustained foreign GPU load > 5% in ['decode', 'decode_greedy'] (mean 8.18%, top ['Google Chrome He']) |
| Qwen3.8-27B | old main (617f6a2) | `q38-27b-kapitsa-main-20261021-072237.json` | 62.5 (58.4–63.9) | 61.6 (59.8–64.9) | pass | 19.1 | CONTENTION: sustained foreign GPU load > 5% in ['decode', 'decode_greedy'] (mean 11.69%, top ['Claude Helper']) |
| Qwen3.8-27B | old main (617f6a2) | `q38-27b-kapitsa-main-20261021-082042.json` | 58.3 (55.3–66.5) | 60.6 (60.0–62.2) | pass | 19.1 | CONTENTION: sustained foreign GPU load > 5% in ['decode', 'decode_greedy'] (mean 4.9%, top ['Google Chrome He', 'Claude Helper']) |
| Qwen3.8-27B | new main (8b190f5) | `q38-27b-kapitsa-trees-20261021-073025.json` | 57.9 (51.1–60.6) | 63.4 (62.8–64.7) | pass | 19.1 | CONTENTION: sustained foreign GPU load > 5% in ['decode', 'decode_greedy', 'tools'] (mean 20.21%, top ['Claude Helper', 'Google Chrome He']) |
| Qwen3.8-27B | new main (8b190f5) | `q38-27b-kapitsa-trees-20261021-081611.json` | 63.2 (54.8–64.9) | 65.5 (62.5–67.2) | pass | 19.1 | CONTENTION: sustained foreign GPU load > 5% in ['decode', 'decode_greedy', 'tools'] (mean 25.54%, top ['Google Chrome He', 'Claude Helper']) |
| Qwen3.6-35B-A3B | old main (617f6a2) | `q36-35b-kapitsa-main-20261021-071809.json` | 168.1 (152.1–179.8) | 177.6 (164.0–181.0) | pass | 20.5 | — |
| Qwen3.6-35B-A3B | new main (8b190f5) | `q36-35b-kapitsa-trees-20261021-072030.json` | 160.6 (153.5–166.7) | 177.2 (173.2–185.3) | pass | 20.5 | — |
| Nemotron 3.5 Lightning 30B-A3B | old main (617f6a2) | `lightning-kapitsa-main-20261021-084144.json` | 104.3 (101.2–110.2) | 115.7 (108.8–119.0) | pass | 18.4 | CONTENTION: sustained foreign GPU load > 5% in ['decode_greedy', 'tools'] (mean 9.08%, top ['Claude Helper']) |
| Nemotron 3.5 Lightning 30B-A3B | new main (8b190f5) | `lightning-kapitsa-trees-20261021-084539.json` | 111.7 (102.1–117.2) | 113.7 (105.4–120.6) | pass | 18.4 | CONTENTION: sustained foreign GPU load > 5% in ['decode', 'decode_greedy', 'tools'] (mean 6.33%, top ['Claude Helper', 'Google Chrome He']) |
| Nemotron 3.5 Bolt 74B-A9B | old main (617f6a2) | `bolt-kapitsa-main-20261021-084818.json` | 64.3 (61.9–67.4) | 69.4 (66.0–70.5) | pass | 22.4 | repetition loop in 1 of 5 sampled runs; CONTENTION: sustained foreign GPU load > 5% in ['decode_greedy', 'tools'] (mean 6.77%, top ['Claude Helper', 'Google Chrome He']) |
| Nemotron 3.5 Bolt 74B-A9B | new main (8b190f5) | `bolt-kapitsa-trees-20261021-085316.json` | 68.8 (62.9–79.3) | 71.2 (66.2–77.9) | pass | 22.6 | CONTENTION: sustained foreign GPU load > 5% in ['decode', 'decode_greedy', 'tools'] (mean 8.78%, top ['Claude Helper', 'Google Chrome He']) |
| Qwen3.6-35B-A3B | Splash 1.4.0 | `q36-35b-splash-20261021-074742.json` | 134.7 (132.3–144.5) | 140.9 (124.1–144.5) | pass | 21.9 | — |
| Qwen3.8-27B | Splash 1.4.0 | `q38-27b-splash-20261021-075612.json` | 52.3 (50.1–61.5) | 54.2 (49.5–56.4) | pass | 21.6 | CONTENTION: sustained foreign GPU load > 5% in ['decode', 'tools'] (mean 2.61%, top ['Google Chrome He', 'Claude Helper', 'Slack Helper']) |
| Qwen3.8-27B | Splash 1.4.0 | `q38-27b-splash-20261021-081022.json` | 52.9 (47.4–55.1) | 57.6 (55.2–58.0) | pass | 21.6 | CONTENTION: sustained foreign GPU load > 5% in ['decode', 'decode_greedy', 'tools'] (mean 6.7%, top ['Claude Helper', 'Google Chrome He']) |

The replayed opencode session on Qwen3.6-35B-A3B (19 requests, prompts up to 22,779 tokens):

| Build | Result file | Total (s) | Prefill (s) | Decode (s) | Writing rate (tok/s) | Follow-up TTFT, median (s) | Tokens written | Same answers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Splash 1.4.0 | `q36-35b-splash-20261021-080112.json` | 56.9 | 29.4 | 27.4 | 122.1 | 0.539 | 3321 | 10 of 19 |
| Splash 1.4.0 | `q36-35b-splash-20261021-080623.json` | 57.1 | 29.7 | 27.4 | 122.0 | 0.539 | 3321 | 10 of 19 |
| new main (8b190f5) | `q36-35b-kapitsa-trees-20261021-073811.json` | 51.4 | 25.6 | 25.8 | 140.8 | 0.470 | 3503 | 14 of 19 |
| old main (617f6a2) | `q36-35b-kapitsa-main-20261021-074106.json` | 51.7 | 25.2 | 26.5 | 138.7 | 0.304 | 3552 | 15 of 19 |

**MacBook Pro M1 Pro, 16 GB** (`MacBookPro18,3`, macOS 27.0.1, on AC power), Qwen3.5-4B, the same protocol, old, new, new, old:

| Build | Result file | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Tool calls | Flagged |
| --- | --- | ---: | ---: | :---: | --- |
| old main (617f6a2) | `m1-pro/q35-4b-kapitsa-main-20261021-074703.json` | 90.7 (83.6–97.1) | 93.5 (85.4–93.8) | pass | — |
| old main (617f6a2) | `m1-pro/q35-4b-kapitsa-main-20261021-075318.json` | 89.2 (74.8–97.3) | 88.0 (83.9–91.4) | pass | repetition loop in 1 of 5 sampled runs |
| new main (8b190f5) | `m1-pro/q35-4b-kapitsa-trees-20261021-074911.json` | 92.6 (83.5–101.5) | 97.6 (95.6–101.1) | pass | — |
| new main (8b190f5) | `m1-pro/q35-4b-kapitsa-trees-20261021-075115.json` | 97.5 (91.2–98.4) | 91.8 (87.0–98.6) | pass | repetition loop in 2 of 5 sampled runs |

## Per-cycle A/B

`ni_run` on each build, five runs per prompt interleaved old main, trees, chain, trees, old main, where chain is the
kflash-trees build with `NI_KFLASH=0` (its drafter chains, as old main), plus one plain-decoding run (`--dflash none`) as
the greedy reference. Prompts: the 512-token coding answer (`bench/prompts/code_27b.txt`), a 512-token prose answer, and
requests of the recorded opencode session ([bench/harness/agent-traces](../../harness/agent-traces/)) as token ids,
18,523-22,779 prompt tokens, each generating its recorded answer's length plus 16 tokens (r06 548, r07 776, r12 338, r13
281, r18 337). Every run writes exactly its token count (`--ignore-eos`). Greedy, then T = 0.6 with seeds 1-3
(`NI_SEED`) on the 4B and 27B. The 27B ran one agent request (each run prefills ~19K tokens for about a minute there).
KFlash's budget cache was primed by one 256-token run per model, the state a user has after a first request. One line per
run in [ab.csv](ab.csv): tok/s over the decode, cycles, tokens per cycle, GPU ms per cycle, a hash of the generated ids,
the thermal state. Averages are geometric means over the prompts; on the dense models only prompts whose old-main and
trees texts are identical count (all but one).

### Qwen3.5-4B, greedy: trees vs old main +2.1% (geometric mean of 6 prompts; r13 left out: main's text departs from plain decoding there and the trees' does not (all 7: -0.9%))

| prompt | main tok/s | trees tok/s | trees vs old main | chain on the trees build | tok/cycle main → trees | GPU ms/cycle main → trees | same ids as plain decoding (main / trees / chain) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| code | 179.3 | 188.5 | **+5.1%** | 178.7 (-0.3%) | 2.95 → 3.30 | 16.45 → 17.46 | yes / yes / yes |
| prose | 139.1 | 151.7 | **+9.1%** | 138.7 (-0.3%) | 2.31 → 2.63 | 16.61 → 17.34 | yes / yes / yes |
| r06 | 142.3 | 142.9 | **+0.4%** | 140.7 (-1.1%) | 3.06 → 3.27 | 21.45 → 22.82 | yes / yes / yes |
| r07 | 197.5 | 192.2 | **-2.7%** | 197.5 (+0.0%) | 4.26 → 4.45 | 21.52 → 23.14 | yes / yes / yes |
| r12 | 201.0 | 206.7 | **+2.8%** | 201.8 (+0.4%) | 4.43 → 4.81 | 22.00 → 23.23 | yes / yes / yes |
| r13<sup>†</sup> | 117.2 | 97.3 | **-17.0%** | 116.9 (-0.3%) | 2.57 → 2.28 | 21.87 → 23.36 | no / yes / no |
| r18 | 130.2 | 127.8 | **-1.8%** | 130.1 (-0.0%) | 2.92 → 3.07 | 22.42 → 23.95 | yes / yes / yes |

### Qwen3.8-27B, greedy: trees vs old main +6.0% (geometric mean of 3 prompts)

| prompt | main tok/s | trees tok/s | trees vs old main | chain on the trees build | tok/cycle main → trees | GPU ms/cycle main → trees | same ids as plain decoding (main / trees / chain) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| code | 61.0 | 64.7 | **+6.0%** | 60.9 (-0.2%) | 4.78 → 5.16 | 78.17 → 79.67 | yes / yes / yes |
| prose | 35.0 | 40.6 | **+15.8%** | 34.9 (-0.4%) | 2.76 → 3.24 | 78.71 → 79.82 | yes / yes / yes |
| r07 | 68.3 | 66.3 | **-2.9%** | 68.5 (+0.3%) | 6.35 → 6.49 | 92.88 → 97.65 | yes / yes / yes |

### Qwen3.6-35B-A3B, greedy: trees vs old main +0.6% (geometric mean of 5 prompts; on this MoE model main's and the trees' texts part at near-ties by design (tree verifies use row-independent expert numerics), so every prompt counts)

| prompt | main tok/s | trees tok/s | trees vs old main | chain on the trees build | tok/cycle main → trees | GPU ms/cycle main → trees | same ids as plain decoding (main / trees / chain) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| code | 176.1 | 181.7 | **+3.2%** | 173.2 (-1.6%) | 3.43 → 3.96 | 19.44 → 21.77 | yes / no / yes |
| prose | 131.9 | 139.4 | **+5.7%** | 133.1 (+0.9%) | 2.42 → 2.72 | 18.34 → 19.46 | no / no / no |
| r06 | 144.3 | 143.5 | **-0.6%** | 141.6 (-1.9%) | 3.99 → 4.38 | 27.62 → 30.42 | no / no / no |
| r07 | 153.6 | 146.1 | **-4.9%** | 149.1 (-2.9%) | 4.53 → 5.00 | 29.50 → 34.22 | no / no / no |
| r12 | 108.9 | 108.9 | **+0.0%** | 109.8 (+0.8%) | 2.96 → 3.26 | 27.08 → 29.84 | no / no / no |

### Qwen3.5-4B, T 0.6, seeds 1-3: trees vs old main +9.6% (geometric mean of 3 prompts)

| prompt | main tok/s | trees tok/s | trees vs old main | chain on the trees build | tok/cycle main → trees | GPU ms/cycle main → trees | trees text = main text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| code | 182.7 | 194.1 | **+6.2%** | 183.1 (+0.2%) | 3.07 → 3.41 | 16.80 → 17.54 | 0 of 6 |
| prose | 141.1 | 142.9 | **+1.3%** | 141.3 (+0.2%) | 2.39 → 2.51 | 16.90 → 17.54 | 0 of 6 |
| r06 | 125.2 | 153.2 | **+22.4%** | 124.8 (-0.3%) | 2.73 → 3.56 | 21.79 → 23.23 | 0 of 6 |

### Qwen3.8-27B, T 0.6, seeds 1-3: trees vs old main +2.8% (geometric mean of 2 prompts)

| prompt | main tok/s | trees tok/s | trees vs old main | chain on the trees build | tok/cycle main → trees | GPU ms/cycle main → trees | trees text = main text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| code | 62.1 | 64.5 | **+3.8%** | 61.8 (-0.5%) | 4.87 → 5.15 | 78.30 → 79.88 | 0 of 6 |
| prose | 37.3 | 38.0 | **+1.9%** | 36.2 (-2.9%) | 2.93 → 3.09 | 78.55 → 81.39 | 0 of 6 |

## Exactness

- Dense models, greedy: the trees' output is token-identical to plain decoding on every prompt (4B 7 of 7, 27B 3 of 3).
  Old main's chain departs from plain decoding once, on the 4B's r13 at token 127 of 281 (a near-tie), and so does the
  chain of the trees build there.
- The 35B (MoE), greedy: old main, the trees and the chain all depart from one-row plain decoding at near-ties, the
  multi-row verify's numerics (code: old main exact, trees from token 177; prose: both from token 31; r06: both from
  token 103, and there old main and the trees are identical). The chain of the trees build is identical to old main.
- The 35B, the same code, 04:51-04:56: bitwise-equal logits and hidden states across 11 tree schedules
  (`tools/moe_tree_check`, 0 of 55 pairs, two prompts), greedy ids identical under forced tree sizes 1, 0, 3, 5, 8, -1
  and the budget's (`tools/kflash_invariance.py`, 0 of 21 pairs, two prompts), the server suite passes (a request alone
  gives the same reply every time), `dflash_accept_test` passes.

## Disturbed runs

From 05:38 to 06:10 macOS's media analysis (`mediaanalysisd`) took the Neural Engine and the GPU: one run's Neural Engine
evaluations took 3.1 s instead of 11 ms (its 18.5K-token prefill 15 minutes), another waited 2.9 s per cycle for the GPU,
and GPU time per cycle doubled. That block (the 4B at T = 0.6) was stopped and run again; from then on every run waited
until the media analysis daemons were idle. The tables leave out any run whose wall time per cycle exceeds its GPU time
by 15%, whose Neural Engine evaluations exceed 150 ms (normal: 11 ms on the 4B, 35 ms on the 27B), or whose GPU time per
cycle is 15% above the fastest run of its group: none.

## Why long agent turns gain less

On the 18-23K-token agent requests the chain already accepts 2.9-6.4 tokens per cycle, so the tree adds 2-10%, while a
tree cycle costs 5-8% more than a chain cycle there on the dense models (1-6% at short context) and 10-16% more on the
35B (6-12% at short context), so those turns land between −5% and +3%.
On the dense models the part that grows with the context is the tree-zone attention (55b6153): it splits the history
into chunks of a fixed size (128 keys, 160 at the agent requests' ~20K capacity) so that a row's numbers do not depend on
the tree, about 3x the partial results of the chain's verify at 19K tokens (40 chunks per KV head, sized to the context).
That chunk also follows the KV capacity, which `kapitsa serve` grows by 25% in 1,024-position steps, so past 16,256
positions a row verified near a growth step can get a different chunk under a different tree schedule. Partitioning the
zone by position alone (128-key chunks below 4K, 256 below 8K, 512 beyond) would close that gap and bring the partials at
19K from ~122 to ~69 per KV head; not done yet.

## Not covered

- Nemotron: KFlash trees are on by default for Nemotron-H with DSpark or LiLiCorr drafters, but no such model is on this
  Mac (Lightning is 23 GB); its speed is not measured here.
- Other Macs, the 35B at T = 0.6 (left out for time), and the 35B's sampled exactness.
- What each determinism fix costs on its own (`NI_V8_TREE_ATTN=0`, `NI_V8_TREE_PIN=0`) and other zone chunk sizes
  (`NI_V8_TREE_CHUNK`): deferred to the zone's repartition.
