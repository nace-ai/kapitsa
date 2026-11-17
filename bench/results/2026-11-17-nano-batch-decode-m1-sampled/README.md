# Nemotron Nano batched decode: the M1, sampled requests and 6K contexts (2026-11-17)

More evidence for `../2026-11-16-nano-batch-decode/` (read its README first): the same change at the branch head
e4b69ec6 against main 9bd41ecd, on a third chip class (the M1, the profile's legacy class) and with seeded sampled
requests and ~6K-token contexts decoding together on the M2 and the M4. Every model run held the Mac's GPU lock
(`lockf -k /tmp/kapitsa-gpu.lock`); so did the builds, which ran on each Mac from `git archive` of the two commits.

## Setups

| Folder | Mac | macOS | Model | Builds |
| --- | --- | --- | --- | --- |
| `m1-nano4b/` | Mac mini M1 (Macmini9,1), 8-core GPU, 8 GB | 27.0 (26A428) | `nemotron-3-nano-4b-fp8` | main 9bd41ecd, branch e4b69ec6 |
| `m2-sampled/` | Mac mini M2 (Mac14,3), 10-core GPU, 16 GB | 27.2.1 (26B131) | `nemotron-3-nano-4b-fp8` | main 9bd41ecd, branch e4b69ec6 |
| `m4-sampled/` | Mac mini M4 (Mac16,10), 10-core GPU, 16 GB | 26.6.2 (25G83) | `nemotron-3-nano-4b-fp8` | main 9bd41ecd, branch e4b69ec6 |
| `m4-nano9b-batch2-recheck/` | (CPU only) | | `nemotron-nano-9b-v2-nvfp4` | the 2026-11-16 M4 dumps (9ef19349) |

## The M1 (8 GB): interleaved A/B and the output check

`scripts/chain.sh MODEL m1-nano4b ab`: `scripts/arm.sh` and `scripts/exact.sh` as in the 2026-11-16 folder, 3 pairs
(main first in pairs 1 and 3), each pair under one GPU lock, then `exact.sh` and `compare.sh` under the lock.

The Nano 4B FP8 does not fit 4 slots in 8 GB: with 4 slots' state the server allocates 6.38 GB before any KV cache
(`m1-nano4b/memory-probe/serve-4-*.log`; 3 slots 6.02 GB). So the M1 ran `SLOTS=2 NS="1 2"
SERVE_ARGS="--mem-budget 5.9 --prefill-chunk 512" NI_MEM_BUDGET=5.9`: 2 slots, 1 and 2 requests at once, 16K context,
15,306 tokens of KV room. The first try used 5.7 GB, the budget of the earlier M1 runs of this model: the server
started with 3,099 tokens of KV room and refused every request, main's and the branch's alike ("insufficient
memory ... (0 tokens)", `memory-probe/chain-m1-budget5.7.out`: 4 arms), so that run was discarded and the chain re-run at 5.9 GB
(`memory-probe/req-5.9.log`, `req-6.1.log`: 2 requests of 64 tokens at once, both answered). No arm swapped out;
35 to 835 swap-ins per arm (`env.txt`).

Medians of the 3 pairs (`m1-nano4b/summary.txt`), tok/s in all, prefills included, per request in parentheses:

| | 1 at once | 2 at once | one request, decode |
| --- | ---: | ---: | ---: |
| M1, 4B FP8, main 9bd41ecd | 15.84 | 15.09 (7.65-8.05) | 24.76 |
| M1, 4B FP8, branch e4b69ec6 | 15.80 | **24.52** (12.59-12.62), 1.63x | 24.76 |

One request, decode, per prompt (mean of its 3 runs), the branch against main: +0.03%, +0.12%, +0.13% (main
24.76-24.85, 24.63-24.88, 15.24-15.26; branch 24.76-24.87, 24.75-24.89, 15.25-15.29). The M1's multi-row NVFP4
products are bound by their arithmetic (the comment above the Nemotron-H prompt-lookup default in
`src/engine/engine.cpp`); this model is FP8, and its shared pass is faster here too.

Outputs (`m1-nano4b/exact/`), the same checks as the 2026-11-16 folder:

- `logit_dump`, main against the branch: every path, every row bitwise identical (`cmp-main-pr.txt`; by construction:
  `logit_dump` does not run `Engine::chat`, see the note in the 2026-11-16 README).
- Batch rows (2 x 1 and 4 x 1) against the 8-row verify: bitwise identical (`cmp-verify-batch4.txt`; `sha256.txt`:
  the verify, 4-member and 2-member `logits.f32` are the same bytes). Against one-row steps: max 0.0128 in a logit, KL mean
  2.1e-7, 0 top-1 flips in 192 rows (`cmp-step-batch4.txt`, `cmp-step-batch2.txt`).
- The 8 greedy requests one at a time: main and the branch, 8 of 8 the same texts; the same texts as the M2's alone
  (`../2026-11-16-nano-batch-decode/m2-nano4b-pr2/exact/pr-alone.json`).
- Together (2 slots, pipelined): 8 at once 8 of 8 identical to alone; 2 at once 2 of 2.
- Traced (pipelining off): 7 of 8 identical, 2077 of 2221 tokens before the first difference; the one difference is
  g3 at token 156, single-stream margin 0.0013: the M2's traced difference, the same token and margin.

## Sampled requests and 6K contexts together (M2 and M4)

`scripts/sampled.sh MODEL TAG`, one GPU lock for the whole run: builds in the order main, branch, branch, main, each on
a fresh `kapitsa serve --slots 4 --ctx 16384 --disk-cache 0`. On each, `scripts/sampled.py` sends two plans, first one
request at a time and then both at once:

- `seeded`: the two seeded requests of `bench/batch/concurrency.py`'s exact plan (the plan #21 used for this check):
  `code_t06_seed7_400` (T = 0.6, seed 7, 400 tokens, an 83-token prompt) and `ctx2k_t06_seed3` (T = 0.6, seed 3, up to
  300 tokens, a 2,015-token prompt).
- `ctx6k`: two requests on ~6K-token contexts (6,185 and 6,652 prompt tokens), 300 tokens each, one greedy and one
  seeded (T = 0.6, seed 5).

The together runs follow the alone runs on the same server, so their prompts come from the prefix cache (time to first
token 0.05 to 0.5 s against 4 to 19 s alone): the together spans are decode, the alone spans include the prefills. Compare
main together with the branch together, and each request's decode tok/s. `scripts/sampled_summary.py` writes each
folder's `summary.txt`. No arm swapped out; the M2's first arm saw 20 swap-ins (`*-env.txt`).

Tok/s in all of the two requests together (two runs per build), and each request's decode tok/s:

| | main together | branch together | per request, alone | per request, together: main / branch |
| --- | ---: | ---: | ---: | ---: |
| M2, seeded | 22.31, 22.25 | 30.57, 30.64 | 20.5 / 26.8-27.0 | 13.7-13.9 / 18.6-19.3 |
| M2, ~6K contexts | 21.26, 21.24 | 35.09, 35.09 | 21.1-21.4 | 10.7 / 17.6-17.7 |
| M4, seeded | 25.00, 27.15 | 37.16, 37.02 | 25.0-25.3 / 31.35-32.70 | 15.0-17.0 / 22.6-23.3 |
| M4, ~6K contexts | 25.91, 25.71 | 43.11, 43.13 | 23.7-26.0 | 12.9-13.0 / 21.6-21.7 |

Texts (`summary.txt`, against the branch's first alone run):

- One at a time, main and the branch give the same 4 replies on both Macs, every run: the seeded replies repeat.
- Main together (taking turns): the same as alone, every request.
- Branch together: the greedy and the seeded ~6K-context replies and `ctx2k_t06_seed3` are the same as alone on both
  Macs, both runs. `code_t06_seed7_400` differs from its reply alone at character 433, the same way in both runs, on
  both Macs: the M2's reply together is the M4's reply alone, and the M4's together is the M2's alone (the alone replies
  of the two Macs differ there too; the other three alone replies are the same on both Macs). Each member samples with
  its own seed and RNG position; the batched pass rounds differently from the passes the request runs alone, which
  moves a draw that close to a boundary, as the two Macs' own rounding does.

## The 9B v2's `cmp-step-batch2.txt`

`m4-nano9b-batch2-recheck/`: `tools/logit_compare.py --no-gate` re-run on copies of the M4's finished dumps
(`ld-pr/step`, `ld-pr/batch`, `ld-pr-b2/batch`, with `meta.json`). `cmp-step-batch2.txt` now names the candidate
(`batch {"members":2,"rows":1}`) and gives the same numbers as the original; `cmp-batch4-batch2.txt`: the 2-member and
4-member batch dumps are bitwise identical, every row (`sha256.txt`: the same `logits.f32`). See the note in
`../2026-11-16-nano-batch-decode/README.md`.
