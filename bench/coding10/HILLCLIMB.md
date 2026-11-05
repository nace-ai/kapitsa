# coding10 hillclimb (#350 B3)

Target arm: `lc` (little-coder's prompt, tools and loop). `opt0` (no tools, thinking off) is already 10/10 on
both models in round 0, so it is the time reference, not the arm being climbed.

Each round adds one change to the best arm so far. Every round's arms run interleaved per prompt with `opt0` and
the unchanged `lc` in the same leg, 3 reps, and `log_round.py` appends one row per (model, arm) to `hillclimb.tsv`.

**Stop line.** Stop at the first round where an `lc_*` arm passes 30/30 (10 prompts x 3 reps) on both qwen3.5-4b
and qwen3.5-9b with a median wall time per prompt no more than 2x `opt0`'s median in the same leg, or after
8 rounds, whichever comes first. What ships then: the best arm (all 30/30, lowest summed median over both
models) becomes `arms/lc_best.json`, and the final table goes into the task summary.

| round | change on top of the previous round |
|---|---|
| 0 | `lc` as ported: thinking on with a 2048-token budget, T 0.3, four tools, 8 turns |
| 1 | thinking off |
| 2 | Qwen's non-thinking sampling: T 0.7, top_p 0.8, top_k 20 |
| 3 | only write_file and run_python, at most 4 turns |
| 4 | on round 1 (best 4B, 30/30): reply DONE once tests pass, solution.py is the answer (no code repeat) |
| 5 | round 4 + Qwen's non-thinking sampling (round 2 gave the 9B 30/30) |

Rounds 1-3 (3 reps, 5 Nov 03:30): 4B opt0 30/30 0.4 s; lc 29/30 5.0 s; r1 30/30 2.8 s; r2 29/30 2.7 s; r3 29/30 3.1 s.
9B opt0 30/30 0.7 s; lc 28/30 6.5 s; r1 29/30 6.2 s; r2 30/30 6.7 s; r3 26/30 7.0 s. No arm meets the stop line; r3 dropped.
