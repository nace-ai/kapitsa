# coding10: ten small coding prompts for local models

Ten one-function Python tasks (FizzBuzz to run-length encoding) with hidden asserts, in
[`bench/prompts/coding10.json`](../prompts/coding10.json). A prompt passes when the returned function runs every
hidden assert clean; the time is wall clock from the first request to the finished answer, tool turns included.
The references pass 10/10 and a wrong stub 0/10.

An arm is a JSON file in `arms/`: system prompt, sampling, thinking on or off, and optionally tools and an agent loop.

- `base`: the prompt as is, Kapitsa's defaults.
- `opt0`: thinking off, Qwen's non-thinking sampling (T 0.7, top_p 0.8, top_k 20), "reply with only the function".
- `lc`: [little-coder](https://github.com/itayinbarr/little-coder)'s prompt, tool cards and tools (write_file,
  edit_file, read_file, run_python) ported to Python, see `littlecoder/NOTES.md`.
- `lc_best`: `lc` after seven tuning rounds (below).

## Run

```bash
REPS=3 bash bench/coding10/run_gpu.sh OUT "4b 9b" \
  bench/coding10/arms/opt0.json bench/coding10/arms/lc_best.json
```

`run_gpu.sh` takes the GPU lock (`$BENCH_GPU_LOCK`), starts `kapitsa serve` per model, runs the arms interleaved per prompt, and writes `OUT/<model>.json`.
The harness waits for AC power before each prompt and reruns one that ends on battery (`--require-ac`).

## Results (M5 Pro 48 GB, Qwen 3.5 4-bit, 5 Nov 2026)

Pass count over 10 prompts x 3 runs, median seconds per prompt. Raw rows: `bench/results/coding10/`. The `lc` and
`lc_best` rows below come from different legs of the same day (03:30 and 10:45-11:38); the same-leg rerun after the
table runs them together.

| arm | 4B | 9B |
|---|---|---|
| base (1 run) | 7/10, 3.0 s | 7/10, 6.3 s |
| opt0 | 30/30, 0.3 s | 30/30, 0.6 s |
| lc as ported | 29/30, 5.0 s | 28/30, 6.5 s |
| lc_best | 30/30, 1.0 s | 30/30, 2.0 s |

Without tools, the small-model settings alone solve every prompt fastest. Tuned, little-coder's tool loop is about 5x
faster on 4B and 3x on 9B than as ported, comparing legs run hours apart; the same arm moved by up to 1 s and by
1-3 passes of 30 between legs. The arms were tuned and scored on the same ten prompts.

Same-leg rerun (7 Nov 2026, same Mac, AC power): `opt0`, `lc` and `lc_best` interleaved per prompt in one leg per
model (`same-4b.json`, `same-9b.json`).

| arm | 4B | 9B |
|---|---|---|
| opt0 | 30/30, 0.4 s | 29/30, 0.7 s |
| lc as ported | 29/30, 4.4 s | 28/30, 7.9 s |
| lc_best | 30/30, 0.9 s | 29/30, 2.2 s |

Run side by side, the tuned tool loop is about 5x faster than as ported on 4B and about 3.5x on 9B, so the earlier
ratios hold. The 9B misses are one run each of `p05_roman` (opt0) and `p06_merge_intervals` (lc_best).

## Tuning rounds

Each round changed one thing on the best arm so far and ran interleaved with `opt0`, 3 runs. Rounds 1-3 also ran
`lc` in the same leg; rounds 4-7 ran with earlier rounds' arms instead (`hc2`, `hc3` in the results). The stop line,
set before the later rounds: 30/30 on both models at a median of at most 1.5 s (4B) and 3 s (9B), or round 7.
`log_round.py` writes one row per model and arm to `bench/results/coding10/hillclimb.tsv`.

| round | change | 4B | 9B |
|---|---|---|---|
| 1 | thinking off | 30/30, 2.8 s | 29/30, 6.2 s |
| 2 | + Qwen's non-thinking sampling | 29/30, 2.7 s | 30/30, 6.7 s |
| 3 | only write_file and run_python, 4 turns (dropped) | 29/30, 3.1 s | 26/30, 7.0 s |
| 4 | round 1 + reply DONE when tests pass, no code repeat | 30/30, 1.7 s | 27/30, 8.8 s |
| 5 | round 4 + Qwen's non-thinking sampling | 30/30, 1.6 s | 28/30, 9.3 s |
| 6 | round 5 + examples run after each write | 30/30, 1.5 s | 28/30, 4.5 s |
| 7 | round 6 + short system prompt (644 to 178 words), ships as `lc_best` | 30/30, 1.0 s | 30/30, 2.0 s |

Rounds 4-7 on 9B ran in one later leg. Most 9B misses are `is_palindrome('')` returning False.
