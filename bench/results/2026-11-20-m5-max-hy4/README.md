# Hy4-flash on the M5 Max, 2026-11-20

Tencent's Hy4-flash (`hy_v4_flash`) from `nace-ai/Hy4-flash-MLX-4bit`: an 8-bit trunk, 512 routed experts per layer
in 4-bit group 64 (83.6 GB), four Engram tables (39.6 GB, read on the CPU). It is a model folder, not a catalog entry.

Mac: fearampgreed's MacBook Pro (16-inch, 2026), M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), on AC power, display asleep, default profile.

## The README cell

`kapitsa-bench-21c5d9ce.json`: `kapitsa bench --model <the pack's folder>` (full protocol), Kapitsa 0.2.37 at
21c5d9ce: #1360's head before its review commit 3be693bd, on top of #1359. Every expert is in memory and thinking off
renders the template's `no_think`. 3be693bd only changed how the expert store is sized (`a4::default_arena`: the
engine's budget must also leave 2 GB beyond the experts). On this Mac the merged code (main 4012c0e6) still takes every
expert, as the server's log at 3be693bd shows, but `kapitsa bench` was not re-run on it.

| decode T 0.6 | greedy | prefill 4K | prefill 16K | follow-up TTFT | tool calls | peak memory |
| --- | --- | --- | --- | --- | --- | --- |
| 36.35 tok/s (5 of 5 valid) | 32.99 | 596 tok/s | 625 tok/s | 0.14 / 0.20 s | pass | 88.3 GB |

The bench flagged two decode runs for repetition loops in the model's own greedy answer (distinct-4-gram 0.45 and
0.47). They count as valid. Its other conditions were clean: thermal state fair, other apps' GPU at most 1.9%, 0
swap-outs, no sleep or battery in `pmset -g log` over the run.

## How it got there

The same Mac and pack on the same day:

| build | decode T 0.6 | prefill 4K | what changed |
| --- | --- | --- | --- |
| 1e2dac03 (main before) | 25.0-27.3 (reasoning text: invalid) | 585 | thinking off ignored; tool calls HTTP 400 |
| 5d13ce79 (#1359's head before its squash) | 24.12, 5 of 5 valid | 624 | thinking off and tool calls fixed |
| 21c5d9ce (#1360's head before its review commit) | 36.35 | 596 | every expert in memory (69% before) |

Each row is one run of `kapitsa bench`; the rows are not an interleaved comparison. #1360's interleaved A B A B is in
its pull request: decode 25.35 / 25.53 -> 36.69 / 36.73 tok/s, with the logits bitwise the same.

## Against mlx-lm

`head-to-head-mlx.json`, 2026-11-20, the same Mac. Both engines got the same prompts: a short coding question (512
tokens out), then the first 4K, 16K and 32K tokens of `bench/corpus.txt` with a request to summarize (256 tokens out).
Thinking was off (`reasoning_effort` `no_think`), temperature 0.6, top-p 0.95. Every prompt was cold, and each engine
ran alone under the GPU lock.

- **Kapitsa:** at b5536ba8 (#1361's head, merged as 179c099f) on the 4-bit pack, timed by its server log (every
  prefill pass, then the decode rate).
- **mlx-lm 0.32.2** (MLX 0.32.4), timed by `stream_generate`'s own rates. On the 4-bit pack it does not run on this
  Mac: it warns the model "requires 126731 MB which is close to the maximum recommended size of 110100 MB", then
  fails with Metal's "Insufficient Memory". So mlx-lm ran the 3.1-bit pack (`nace-ai/Hy4-flash-MLX-3.1bit`, 85.5 GB,
  3.12 effective bits a weight) through that pack's own `run_mlx.load()`.

| prompt | Kapitsa 4-bit, precise: prefill / decode | Kapitsa 4-bit, fast profile | mlx-lm, 3.1-bit pack |
| --- | --- | --- | --- |
| short (~50 tokens) | 209 / 36.4 tok/s | 225 / 35.9 | 125 / 20.8 |
| 4K | 657 / 35.2 | 968 / 34.9 | 228 / 13.2 |
| 16K | 699 / 35.1 | 971 / 34.1 | 300 / 7.0 |
| 32K | 677 / 34.6 | 971 / 34.8 | 304 / 4.4 |

Peak memory: Kapitsa 88 GB with every 4-bit expert in memory (the `kapitsa bench` run at 21c5d9ce above; the
head-to-head did not record Kapitsa's peak), the Engram tables read from their mapped files on the CPU; mlx-lm 64 GB at the short prompt and 100 GB at 32K on the 3.1-bit pack.

These are one run of each, not interleaved. mlx-lm's timers and Kapitsa's server log measure the same two phases.
