# coding10 hillclimb (#350 B3)

Target arm: `lc` (little-coder's prompt, tools and loop). `opt0` (no tools, thinking off) is already 10/10 on
both models in round 0, so it is the time reference, not the arm being climbed.

Each round adds one change to the best arm so far. Every round's arms run interleaved per prompt with `opt0` and
the unchanged `lc` in the same leg, 3 reps, and `log_round.py` appends one row per (model, arm) to `hillclimb.tsv`.

**Stop line.** Stop at the first round where an `lc_*` arm passes 30/30 (10 prompts x 3 reps) on both qwen3.5-4b
and qwen3.5-9b with a median wall time per prompt of at most 1.5 s (4B) and 3 s (9B), or after round 7 (set 5 Nov 10:20,
replacing "2x opt0 or 8 rounds"), whichever comes first. What ships then: the best arm (all 30/30, lowest summed median over both
models) becomes `arms/lc_best.json`, and the final table goes into the task summary.

| round | change on top of the previous round |
|---|---|
| 0 | `lc` as ported: thinking on with a 2048-token budget, T 0.3, four tools, 8 turns |
| 1 | thinking off |
| 2 | Qwen's non-thinking sampling: T 0.7, top_p 0.8, top_k 20 |
| 3 | only write_file and run_python, at most 4 turns |
| 4 | on round 1 (best 4B, 30/30): reply DONE once tests pass, solution.py is the answer (no code repeat) |
| 5 | round 4 + Qwen's non-thinking sampling (round 2 gave the 9B 30/30) |
| 6 | round 5 + the examples run automatically after each write/edit of solution.py (saves the run_python turn) |
| 7 | round 6 + a compact system prompt (644 to 178 words; tool cards cut to one line) |

Rounds 1-3 (3 reps, 5 Nov 03:30): 4B opt0 30/30 0.4 s; lc 29/30 5.0 s; r1 30/30 2.8 s; r2 29/30 2.7 s; r3 29/30 3.1 s.
9B opt0 30/30 0.7 s; lc 28/30 6.5 s; r1 29/30 6.2 s; r2 30/30 6.7 s; r3 26/30 7.0 s. No arm meets the stop line; r3 dropped.

Rounds 4-5 (3 reps, 5 Nov 10:13): 4B opt0 30/30 0.4 s; r1 30/30 3.5 s; r4 30/30 1.7 s; r5 30/30 1.6 s. 9B not run: its weights are on the external drive, which was unplugged (server could not start).

Rounds 6-7, 4B (3 reps, 5 Nov 10:45, all 150 prompts on AC): opt0 30/30 0.3 s; r4 30/30 1.6 s; r5 30/30 1.3 s; r6 30/30 1.5 s; r7 30/30 1.0 s.
r7 meets the 4B line and ships as `arms/lc_best.json` on 4B evidence; the 9B leg (k350-hc3-9b2, `--require-ac`) confirms or replaces it.

9B, rounds 1-7 (3 reps, 5 Nov 11:01-11:38, all 180 prompts on AC): opt0 30/30 0.6 s; r1 27/30 7.3 s; r4 27/30 8.8 s; r5 28/30 9.3 s; r6 28/30 4.5 s; r7 30/30 2.0 s.
Most 9B failures are is_palindrome('') returning False. Stop line met: lc_best (= r7) is 30/30 on both models, 1.0 s (4B) and 2.0 s (9B) per prompt. Hillclimb closed.
