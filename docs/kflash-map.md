# KFlash: where the work is

A map of the KFlash work as of 2026-10-21: which branch holds what, what was measured, what is still open, and how
to resume. **KFlash trees are in `main` since [#187](https://github.com/nace-ai/kapitsa/pull/187) (8b190f5, the
`kflash-trees` branch squashed), on by default on every target; [docs/kflash.md](kflash.md) explains how they work and
[docs/benchmarks.md](benchmarks.md#kflash-trees-m4-pro-2026-10-21) has the numbers.** KFlash is Kapitsa's tree-verify generation of speculative decoding. The target checks a best-first tree of
the drafter's candidates instead of one chain, exactly, sized by a budget that each Mac measures for itself.

## Branches on origin

| Branch | What it holds | State |
| --- | --- | --- |
| `main` | the engine | KFlash trees merged as #187 (8b190f5) |
| `kflash` | everything: trees (Qwen + Nemotron), the shared budget, the candidate decoder (trainer, Metal, results), budget persistence, 16/32-row verifies | integration branch, no PRs, tip `43947bf` |
| `kflash-trees` | the release candidate: trees only, no decoder | merged into `main` as #187 |
| `claude/dflash-3-kapitsa-f294ab` (PR #109, closed) | the copy lane (prompt lookup inside the DFlash loop), merged with main after #187: it runs in chain cycles only (`NI_KFLASH=0`) | closed 2026-10-21; next step: the copy chain as a branch of the KFlash tree (in chain cycles it took a 21K-token agent request on the 4B from 114 to 166 tok/s) |
| `claude/copy-lane-ext` | the copy lane's follow-up work in progress: batched decode, Nemotron, skipping unused drafts | `4c58527`, marked WIP; it builds; `dflash_accept_test` passes on one seed stream but fails one new case on the default stream (unchecked); never run on a model; not KFlash |

### `kflash-trees` commits (no decoder code; authored by the user)

| Commit | What |
| --- | --- |
| `2c597ef` | The trees, squashed from `kflash` `e8c84b0..44e0e72`, before any decoder code: Qwen dense trees (7 nodes in one 8-row tile), Nemotron-H trees (Mamba-2 tree scan), the shared tree descriptor / builder / acceptance, and KfBudget (per-depth calibration, cost per row count, cached per model / drafter / chip). |
| `4afa619` | `kapitsa serve` saves the KfBudget cache on shutdown and every 256 cycles; every tool measures a cycle's whole GPU time (same patch as `kflash` `bc46716`). |
| `2acc61b` | One switch and one rule: trees on dense targets, the chain on MoE targets. `NI_KFLASH=1` forces trees, `NI_KFLASH=0` chains everywhere. |
| `fa3edc2` | `tools/tensor_probe.mm` builds with SDKs before macOS 26 (found on the M1 Pro fleet Mac). |
| `4fc87d0` | Trees on every target. On MoE targets (Qwen3.6-35B, Nemotron-H) the tree's size is dynamic: the budget picks 1..8 live rows per cycle, and the verify routes experts only for those rows. `NI_KFLASH=0` chains everywhere. |
| (this file) | the map |

**Done: dynamic trees on MoE targets (`4fc87d0`).**
- The rule is trees on every target. Dense targets verify whole 8-row tiles.
- On MoE targets, the budget picks 1..8 live rows per cycle and the verify's expert work covers only those
  rows, so a tree of n rows costs about the experts of n rows.
- Not completed: the sampled exactness check on the 35B and more sampled seeds. The user chose to commit
  without them.

### `kflash` commit map

| Commits | Piece | Status |
| --- | --- | --- |
| `e8c84b0` | Nemotron-H trees: DSpark / LiLiCorr lattices, Mamba-2 tree scan, exact acceptance | in `kflash-trees` |
| `a7aaea4`, `7224485`, `de5c4d0`, `18c6576` | Qwen dense trees; online calibration and measured cost; `tree_sim.py`; one shared descriptor / builder / acceptance / budget | in `kflash-trees` |
| `44e0e72` | docs for `NI_DFLASH_TREE` / `NI_KFLASH` | in `kflash-trees` |
| `75260f0`, `8d220ef`, `d3f55b4`, `d9139c4`, `fc2a4a4`, `8f4351f`, `7a9af69`, `e5aad53` | the candidate decoder in Metal, rescoring the tree, plus its pass cap, 8-bit option and budget-cache key | `kflash` only (opt-in) |
| `169c4a0` and the RESULTS commits (`bb0dd71`, `bcadc5c`, `fcb8be9`) | the decoder's data generator, MLX trainer, evaluation, `tools/kflash/RESULTS.md` | `kflash` only |
| `bc46716` | budget persistence | in `kflash-trees` as `4afa619` |
| `43947bf` | 16- and 32-row tree verifies on dense Qwen; the size rule picks 8 / 16 / 32 per cycle | `kflash` only, not yet decided for `kflash-trees` |

**The switches differ by branch.** On `kflash`, `NI_KFLASH=1` on a Qwen model loads the candidate decoder; on
Nemotron it turns trees on. On `kflash-trees`, `NI_KFLASH` is only the tree switch.

## Measured (end to end, output identical to plain decoding, unless noted)

Trees of 7 nodes vs `main`, interleaved, M4 Pro 48 GB:

| | code | prose | agent requests |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B | +9.9% | +18.6% | +5.4% |
| Qwen3.5-4B | +9.8% | +10.8% | +3.7% |

- **M1 Pro 16 GB (Qwen3.5-4B):** +2% to +16% on code across greedy and sampled runs, +6% to +13% on the agent replay.
  The budget picks 8 rows, and on that GPU 8 rows cost 6% more than 1 row.
- **Qwen3.6-35B MoE, interleaved, tree vs mean(main, chain), M4 Pro:**
  - Dynamic trees (`4fc87d0`), greedy: code +1.5%, prose +0.8%, agent requests +4.1% / -4.0%.
  - Dynamic trees, T 0.6: +9% on average over 11 seeded runs, -7% to +27% per seed.
  - A forced 8-row tree measured 0% greedy and -7% sampled.
- **16/32-row verifies on the M4 Pro:**
  - At 4-bit, 16 rows cost 1.6-1.75× an 8-row verify for 1.09-1.14× the tokens, so the rule keeps 8.
  - At 8-bit, it keeps 8.
  - Qwen3.5-4B in bf16: 16 rows cost 1.05× and give +11-16% tok/s (sizes forced; the interleaved bf16 A/B was
    deferred).
- **Replay, bigger trees:** tokens per cycle over the chain on the 27B are +4% at 7 nodes, +16% at 15 and +21% at 31
  on the agent session, and +9% / +23% / +28% on code. That is before the cost of the extra rows.
- **Candidate decoder** (`kflash` only, opt-in):
  - Replay on mixed held-out text, accepted tokens over calibrated trees: +2.7% / +5.0% / +6.5% at 8 / 16 / 32 rows.
  - Engine, 48 unseen agent turns: +12.9% decode speed for a session's mix (+28% on tool-call turns).
  - Code and prose: -2% to -4%.
  - It learned the harness's tool-call syntax, not general text. Retraining is parked.

## Not in git (by design)

- **Private docs:** the KFlash spec, parked ideas (including the online-tuned selector, for Alisher), the fleet Mac
  setup script, and the sessions' notes and result tables. They live outside this public repo, in a private handoff
  bundle.
- **The 4B decoder's weights and its training data:** in the coordinator Mac's model cache
  (`<model>/kflash/`, `~/.cache/kapitsa/kflash-dev/`). Never committed.
- **The fleet Mac's local branches:** `benchmarks/m1pro-*` predate this work and look superseded by #135;
  `m1pro/kflash-budget-save` landed as `bc46716`.

## Open items

**From the merge bench (2026-10-21, [results](../bench/results/2026-10-21-kflash-trees/README.md)):**
- **Long agent turns gain least.** On 18-23K-token requests the chain already accepts 2.9-6.4 tokens per cycle, and a
  tree cycle costs 5-8% more on the dense models (1-6% at short context), so those turns land between −5% and +3%.
  The part that grows with the context is the tree-zone attention: its chunk is fixed (128 keys, 160 at a ~20K
  capacity), about 3x the chain's partial results at 19K tokens. Proposed: partition the zone by position alone
  (128-key chunks below 4K, 256 below 8K, 512 beyond; ~69 partials per KV head at 19K, within the 1,024-entry buffer up
  to a 32K context). Measure first with `NI_V8_TREE_CHUNK=256 / 512` (no rebuild).
- **Determinism gap past 16K tokens in the server.** The zone's chunk follows the KV capacity, which `kapitsa serve`
  grows by 25% in 1,024-position steps; past 16,256 positions a row verified near a growth step can get a different
  chunk under a different tree schedule. The position-only partition closes it. Not covered by any check yet (all run
  at short contexts with a fixed capacity).
- **The 35B's tree cycle costs 6-16% more GPU time than its chain**, which eats most of its 10-15% acceptance gain:
  per-cycle A/B +0.6% greedy; harness −4.4% at T 0.6 (one run each). Two causes, split unmeasured: the trees verify
  more rows per cycle than the chain's dynamic depth did (4.8-7.1 against 3.5-5.7), and every MoE row routes its own
  experts; and the row-independent expert numerics (`0b52781`, about 3% by its code comment, never timed on greedy).
  So the budget's size choice on MoE is worth checking against its measured cost first.

1. Dynamic MoE trees: done (`4fc87d0`; their greedy invariance is item 2). Optional: the 35B's sampled exactness check and more sampled seeds.
2. **Greedy replies on MoE with dynamic trees: fixed in the attention (`NI_V8_TREE_ATTN=2`, tree zone); checked on the M5 Pro.**
   - The bug: on the Qwen3.6-35B, `kapitsa serve` failed `tests/test_server.py` "a request alone gives the same reply
     every time (greedy)" with dynamic trees and passed with `NI_KFLASH=0`. The tree's size and shape follow the
     budget's learned state, and the 35B's router flips its top-8 on a last-bit difference between two rows of
     the same path.
   - Earlier fixes (`fd9b6a9`, `0b52781`: the GDN step's fused update, row-independent expert numerics, the tree
     block as its own attention chunk). They were reported as verified on the M4 Pro (server suite passing, 4B and
     27B identical to plain), but they did not make the 35B invariant: on the M5 Pro the suite failed that test on some
     runs, 3 of 5 prompts gave different replies across 6 cold runs, and `kflash_invariance.py` found 4 of 5
     prompts whose replies depend on the forced tree size. The GDN fused update (`NI_V8_TREE_PIN`) had no effect on
     any output we measured.
   - The root cause, found by hashing the logits and each layer's hidden state of the same teacher-forced
     tokens verified under different tree schedules (`tools/moe_tree_check.cpp`): the first row that differs is
     always in layer 3, the first full-attention layer (GDN layers 0-2 are bitwise equal). The attention split a row's
     keys into history chunks whose size and boundaries followed the cycle's anchor (`Lmax = pos + rows`), plus the
     tree block, so the same row summed the same keys in a different order whenever the anchor or the tree
     shape differed. The router then flips on the last bits. It isn't visible as an argmax difference for many tokens,
     which is why the old checks missed it.
   - The fix (`NI_V8_TREE_ATTN=2`, the default; `1` = the previous attention, `0` = the block inside the MMA tiles):
     a row at position `q` takes the keys below `tree_zone(q) = (q - 7) / 32 * 32` through the MMA tiles, in chunks of
     one fixed size (`NI_V8_TREE_CHUNK`, 128; independent of the anchor and the position), and the keys from there to
     `q` (8 to 39 of them: the cache rows below the anchor, then the row's ancestors) through a serial fp32
     online softmax in position order (`attn8_tree_zone`). A row's bits depend on its position and its path only.
   - Checks, Qwen3.6-35B-A3B 4-bit on the M5 Pro, 2026-10-20:
     - `moe_tree_check` (new; bitwise logits and per-layer hidden state of 11 tree schedules, chains, tails, jumps,
       random and mixed trees, 64 teacher-forced positions of one prompt, 96 of another): 52 of 55 schedule pairs differ before, 0 of 55 after;
     - `kflash_invariance.py` (now every pair, plus the budget's own `auto` trees; 5 prompts, 384 tokens, sizes
       1,0,3,5,8,-1,auto): 4 of 5 prompts differ before (6, 17, 6, 17 of 21 pairs), 0 of 21 pairs differ on all
       5 after;
     - 6 cold-prefill repeats of 5 server prompts: 0 of 5 prompts differ;
     - server suite, twice: 47/47 both times;
     - plain decoding (35B, 5 prompts), the 4B's greedy replies (plain and trees of every size), and the 27B's (one 192-token
       prompt, plain and trees): identical to before.
   - Cost, 35B greedy, interleaved A/B against `NI_V8_TREE_ATTN=1` (6 runs per arm on the two 512-token prompts, 2 per
     arm on 4K and 16K-token prompts): 0.2 to 0.3 ms more GPU time per cycle (16.5 to 16.7 ms -> 16.8 to 16.9 ms, 22.7 -> 23.0 ms at
     16K), which is about 1% in tok/s on the short prompts (code 205.1 -> 203.0, prose 165.8 -> 163.4, 4K 229.7 -> 232.1 (noise: 2 runs per arm),
     16K 189.3 -> 187.3). Tokens per cycle are the same.
   - Also checked on the M4 Pro, 2026-10-21:
     - `moe_tree_check`: 0 of 55 pairs differ, on two prompts (96 and 64 positions);
     - `kflash_invariance.py` (sizes 1,0,3,5,8,-1,auto, 384 tokens): 0 of 21 pairs differ, on two prompts;
     - the 35B server suite: all pass;
     - `dflash_accept_test`: all pass;
     - 4B and 27B greedy: identical to plain decoding.
   - Not covered: the 35B at temperature > 0 (the sampled exactness check), other MoE targets, and Macs other than
     the M5 Pro and M4 Pro. `tools/moe_tree_check` and `kflash_invariance.py` are the tests to run after touching the verify
     attention or the GDN step.
3. **`nh_tree_check` fails its planted-branch acceptance cases since `6cd6a11`** (the merge of `main`: chained DSpark
   drafting alongside tree drafting).
   - On the M4 Pro, greedy: 86 of 115 cycles accepted other than the planted branch; sampled: 83 of 112.
   - Everything else in it passes, including the real loops: chain and tree, greedy and sampled, 0 tokens differ from
     plain decoding.
   - Most likely the test's planted-branch harness disagrees with the new chained DSpark drafting, but it needs
     checking before Nemotron trees are trusted. It passed at `50a41f5`.
4. Nemotron speed: tree vs chain on Lightning. This needs the 23 GB model; the user said not to download it for now.
5. 16/32-row verifies: **in PRs [#245](https://github.com/nace-ai/kapitsa/pull/245) (the kernels) and
   [#246](https://github.com/nace-ai/kapitsa/pull/246) (the budget chooses 8, 16 or 32 rows), opt-in:**
   - `NI_KFLASH_ROWS` defaults to `8`, one 8-row tile; naming `8,16,32` lets the budget measure and pick wider tiles
     per cycle. (The 16-row drafting block later in this file overrides both: on a one-slot engine it runs every cycle
     at 16 rows.) With the tile-pair MLP a 16-row verify cycle costs 1.07× and a 32-row one 1.49× the 8-row cycle on the
     27B's coding answer; on a paired A/B (27B, greedy, n=3) asking for `8,16,32` gave a median +3% tok/s
     (72.9 vs 70.5) at +12% tokens per cycle, ranges overlapping, so not claimed as a gain.
   - A wide verify's attention is the 8-row tree attention (`attn8m_partial_tree`: each row's ancestors by mask), not
     the 8-row path's tree-zone split (`NI_V8_TREE_ATTN=2`), which is a kernel of one tile. The rows' logits therefore
     round differently from the 8-row path's: measured on the 4B, max |logit diff| 0.019 (0.0005 of the row's range),
     every live row's argmax equal to plain decoding, and greedy replies identical. Making the zone split work on more
     than one tile is open work, not a correctness item.
6. Merging to `main` waits for the user. `main` has moved on, so bring the branch up to date first, then rerun the
   checks and a short A/B.
7. Parked:
   - the decoder's measured on/off switch;
   - the online-tuned selector;
   - the 4-bit verify on the M5's matrix units, which needs an M5 in the fleet.

## How to resume

```bash
git fetch origin && git switch kflash-trees && git pull --ff-only
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 12
# exactness (GPU; hold the shared lock, at most 20 minutes per hold)
lockf -k -t 3000 /tmp/kapitsa-gpu.lock build/dflash_accept_test 20000
python3 tools/nh_synth.py /tmp/nh_small --dspark --small && build/nh_tree_check /tmp/nh_small
build/ni_run ~/.cache/kapitsa/models/qwen3.5-4b-4bit @bench/prompts/code_27b.txt 256 0 --dflash none > /tmp/plain.txt
build/ni_run ~/.cache/kapitsa/models/qwen3.5-4b-4bit @bench/prompts/code_27b.txt 256 0 --dflash auto --ref /tmp/plain.txt
# A/B: interleave chain and tree runs (C B B C) on code, prose and agent requests; read tok/s and tok/cycle
NI_KFLASH=0 build/ni_run <model> @bench/prompts/code_27b.txt 512 0 --dflash auto --ignore-eos
build/ni_run <model> @bench/prompts/code_27b.txt 512 0 --dflash auto --ignore-eos
```

The KfBudget cache is at `~/.cache/kapitsa/kflash/<model>-<drafter>-b<bits>-<chip>.json`. Delete it to make the
budget re-measure.
