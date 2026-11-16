# Nemotron Nano: requests decoding at once share one pass (2026-11-16)

Before this change, requests sent at once to a Nemotron Nano (`--slots 2` or more) took turns on the GPU: every
decode step served one request, so two or four replies at once wrote about as many tokens per second in all as one
reply alone. With it, while two or more Nano requests decode, one pass over every request's pending token
(`NemotronH::encode_verify_multi`, plain rows, the batch loop LiLiCorr on Bolt already uses) serves them all.

## Setups

| Folder | Mac | macOS | Model | Builds |
| --- | --- | --- | --- | --- |
| `m2-nano4b-pr2/` | Mac mini M2 (Mac14,3), 10-core GPU, 16 GB | 27.2.1 (26B131) | `nemotron-3-nano-4b-fp8` | main 1846c9ce, branch 9ef19349 |
| `m2-nano4b/` | Mac mini M2 | 27.2.1 | `nemotron-3-nano-4b-fp8` | main 1846c9ce, branch ca676e2e (first commit; the earlier run) |
| `m4-nano4b/` | Mac mini M4 (Mac16,10), 10-core GPU, 16 GB | 26.6.2 (25G83) | `nemotron-3-nano-4b-fp8` | main 1846c9ce, branch 9ef19349 |
| `m4-nano9b/` | Mac mini M4 | 26.6.2 | `nemotron-nano-9b-v2-nvfp4` (plain steps, its default) | main 1846c9ce, branch 9ef19349; 256-token replies, `--private` |
| `m2-nano9b-spec4-probe/` | Mac mini M2 | 27.2.1 | `nemotron-nano-9b-v2-nvfp4` with `--spec 4` | branch ca676e2e, batched against `NI_BATCH=0` |

ca676e2e batches the Nano 4B's prompt-lookup loop; 9ef19349 adds the plain-step loop (the 9B v2's default). Between
the two the 4B's path differs only in the flag that says it batches (`nh_batch_`, true for the 4B in both).

Every GPU step held the Mac's GPU lock (`lockf -k /tmp/kapitsa-gpu.lock`), one pair (or the whole block) under one
lock; `env.txt` in each arm has `vm_stat`'s swap counters before and after and the two busiest processes (on the M2
`audiomxd` at ~76% of one core throughout, alike for both builds). No other GPU work ran while the lock was held.
No arm swapped out; in `m2-nano4b-pr2/` three arms (one main, two branch) saw 4 to 44 swap-ins (16 KB pages).

## How

`scripts/arm.sh BIN MODEL OUT` is one arm: `kapitsa serve --slots 4 --ctx 16384 --disk-cache 0`, then
`bench/batch/concurrency.py streams` with 1, 2 and 4 requests at once (512-token greedy replies to the code prompt,
thinking off, cold prompts behind a nonce), then `concurrency.py single` (the same 3 prompts one at a time: decode
tok/s, time to first token excluded). `scripts/rest.sh` runs the pairs interleaved (pair 1 main first, pair 2 the
branch first, pair 3 main first) and then `scripts/exact.sh`:

- `logit_dump` of main and of the branch (`--paths step,verify,batch`, the branch `--from` main's dump), and the
  branch's batch path with 2 members of 1 row (`ld-pr-b2`); `scripts/compare.sh` runs `tools/logit_compare.py
  --no-gate` on them (there is no `tools/logit_floors/` file for the Nano 4B: the comparison reports, it does not gate).
- `scripts/gtexts.py`: the 8 greedy requests of `concurrency.py greedy` (300 tokens), one at a time on main, one at a
  time on the branch, all 8 at once on the branch (4 slots: 4 decode together, the rest as slots free), and the first 2
  at once.
- The same 8 requests with `NI_TRACE_TOKENS` on the branch, alone and together (`bench/batch/match.py` compares the
  traces: tokens and the single-stream margin between the two largest logits where they first differ). The trace reads
  each pass's logits, which turns the batch's pipelining off (`NI_BATCH_PIPE`), so its batched passes are encoded one at
  a time; the texts above are from the default, pipelined path.

## Throughput (tok/s in all, prefills included; per request in parentheses)

Medians of 3 interleaved pairs, at 9ef19349 (M2: `m2-nano4b-pr2/`; the earlier ca676e2e run, `m2-nano4b/`, read
36.55 and 64.78 against 22.47 and 22.01, decode 36.87 against 36.88); every arm is in each folder's `summary.txt` and its arm folders.

| | 1 at once | 2 at once | 4 at once | one request, decode |
| --- | ---: | ---: | ---: | ---: |
| M2, 4B FP8, main | 23.44 | 22.51 (11.4-12.0) | 22.04 (5.6-6.0) | 36.89 |
| M2, 4B FP8, branch | 23.48 | **36.58** (18.8-18.9), 1.63x | **64.87** (16.9-17.3), 2.94x | 36.88 |
| M4, 4B FP8, main | 28.36 | 27.07 (13.7-14.4) | 26.45 (6.7-7.2) | 44.85 |
| M4, 4B FP8, branch | 28.45 | **44.54** (22.8-22.9), 1.65x | **80.21** (20.8-21.4), 3.03x | 44.71 |
| M4, 9B v2 NVFP4, main | 15.07 | 15.02 (7.8) | 15.03 (3.9) | 15.70 |
| M4, 9B v2 NVFP4, branch | 15.07 | **26.87** (14.4-14.5), 1.79x | **45.79** (12.8-13.2), 3.05x | 15.69 |

"One request, decode": `single`'s 3 prompts x 3 pairs, median (the prompts differ only in a nonce; one of them gets
fewer prompt-lookup drafts accepted, ~22.7 on the M2 and ~27.3 on the M4 in both builds). Per prompt (mean of its 3
runs), the branch against main: M2 -0.01%, +0.12%, +0.81% (the third only because one main run read 22.13, the
other eight runs of that prompt 22.67-22.75; medians 22.71 main, 22.70 branch); M4 -0.14%, -0.36%, -0.20%. The M4's second prompt is the
one whose runs do not overlap (main 44.75-44.99, branch 44.66-44.74). The 9B (256-token replies): -0.11%, -0.02%,
-0.02%, every run 15.66-15.72 in both builds. Time to first token is unchanged (prompts
take turns on the GPU as before: the `ttft` lists in each arm's `streams*.json`).

The 9B v2 probe (`m2-nano9b-spec4-probe/`, 256-token replies, 2 runs each, order batched / turns / turns / batched):
with `--spec 4` the 9B decodes with prompt lookup, so ca676e2e batches it. In all, 1 / 2 / 4 at once: batched 13.60
/ 20.98 / 35.80 and 13.61 / 21.01 / 35.90; taking turns 13.60 / 13.13 / 12.89 and 13.59 / 13.14 / 12.90.

## Outputs

**One request alone, main against the branch** (both Macs, both models): `logit_dump`'s every path (`prefill`, `step`, `verify`,
`batch`) is bitwise identical, every row (`*/exact/cmp-main-pr.txt`), and the 8 greedy replies one at a time are the
same texts (`main-alone.json` against `pr-alone.json`: 8 of 8). The alone path's code is not changed: a request takes
the batch only when another one decodes.

**Together against alone.** A batched pass runs the members' rows as a pass of 2 or more rows (GEMM kernels on
half-precision activations), where a request alone with no draft to verify runs one row. On the `logit_dump` windows
(192 rows), 2 members x 1 row and 4 x 1 are both bitwise identical to the 8-row verify the 4B already runs alone when
prompt lookup proposes drafts, and differ from one-row steps by at most 0.0128 in a logit, mean KL 2.1e-7 (M2) and
2.5e-7 (M4), no top-1 flip (`cmp-step-batch4.txt`, `cmp-step-batch2.txt`, `cmp-verify-batch4.txt`). The 9B v2 on the
M4, whose requests alone are one-row steps: batched rows bitwise identical to its 8-row verify, against steps at most
0.0097 in a logit, mean KL 1.2e-7, no top-1 flip.

Greedy replies, 8 requests x up to 300 tokens:

| | M2 (9ef19349) | M2 (ca676e2e) | M4 |
| --- | --- | --- | --- |
| 8 at once (pipelined), texts identical to alone | 7 of 8 (g3 differs at character 674) | 7 of 8 (g3 at 674) | 6 of 8 (g1 at character 764, g3 at 674) |
| 2 at once (pipelined) | 2 of 2 | 2 of 2 | 1 of 2 (g1, as above) |
| 8 at once, traced | 7 of 8; 2077 of 2221 tokens before the first difference | 8 of 8; 2221 of 2221 tokens | 6 of 8; 1971 of 2221 tokens |

The 9B v2 on the M4 (`m4-nano9b/exact/`): 8 at once and 2 at once, every text identical to alone (8 of 8, 2 of 2).
Its plain-step loop alone writes no `NI_TRACE_TOKENS` lines (the trace covers the speculative loops and the batch), so
its trace files hold only the batched side and are not compared.

The traced M4 differences are at single-stream margins of 0.00083 (g1, token 194) and 0.00025 (g3, token 156); the
traced M2 one (9ef19349) is g3 at token 156, margin 0.0013. The M2's pipelined g3 difference is at token 156 too (character 674; decoded from the traced alone run's tokens with the
model's tokenizer), where the alone margin is 0.0013. For scale: 12.8% of all greedy tokens in these replies have a
margin under 0.5 (both Macs). g3 alone also differs between the two Macs (M2 and M4 alone, same binary, same tokens up to there):
its alone text on the M4 is the M2's together text. Every difference is a near-tie of that kind.
