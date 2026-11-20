# Hy4-flash on the M5 Max, 2026-11-20

Tencent's Hy4-flash (`hy_v4_flash`) from `nace-ai/Hy4-flash-MLX-4bit`: an 8-bit trunk, 512 routed experts per layer
in 4-bit group 64 (83.6 GB), four Engram tables (39.6 GB, read on the CPU). It is a model folder, not a catalog entry.

Mac: MacBook Pro (16-inch, 2026), M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), on AC power, display asleep.

## The README cell

`kapitsa-bench-21c5d9ce.json`: `kapitsa bench --model <the pack's folder>` (full protocol), Kapitsa 0.2.37 at
21c5d9ce, the head merged as #1360, whose tree is main's 4012c0e6. Every expert is in memory (#1360), and thinking
off renders the template's `no_think` (#1359).

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
| 5d13ce79 (#1359) | 24.12, 5 of 5 valid | 624 | thinking off and tool calls fixed |
| 21c5d9ce (#1360) | 36.35 | 596 | every expert in memory (69% before) |

Each row is one run of `kapitsa bench`; the rows are not an interleaved comparison. #1360's interleaved A B A B is in
its pull request: decode 25.35 / 25.53 -> 36.69 / 36.73 tok/s, with the logits bitwise the same.
