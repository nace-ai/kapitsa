# Results, 2026-10-21: MacBook Pro M5 Pro, 48 GB (prefill re-measure)

Kapitsa's prefill on the M5 Pro after the M5 prefill kernels landed ([#205](https://github.com/nace-ai/kapitsa/pull/205) dense 4-bit
GEMMs, [#215](https://github.com/nace-ai/kapitsa/pull/215) attention, the Nemotron-H kernels of 2026-10-20), written by
[bench/harness](../../harness/README.md): `compare_engines.py configs/kapitsa.json configs/lightning.json --skip agent`, the full
protocol minus the agent session, one config at a time under the GPU lock. MacBook Pro (16-inch, 2026), `Mac17,8`, M5 Pro, 48 GB, macOS 26.5.2, on AC power.
Kapitsa 0.2.2: the first set (19:56-20:12 local, UTC+5) is b50f1fc (main after #239); the second set (22:00-22:30) records `git_head` a13f8fd, a commit on the #241 branch that differs from b50f1fc only in README, docs, poster and stored results (no source or build files), so it is the same engine.
A third set (23:54) is current main at `3a5bbf0`, after [#243](https://github.com/nace-ai/kapitsa/pull/243) (ANE keepalive)
and [#245](https://github.com/nace-ai/kapitsa/pull/245) (KFlash wide tree verify), a single 4B re-run because the 20:00 4B
run above predates both.

The README poster's M5 Pro cells for Kapitsa on the 27B, 35B-A3B and Lightning are the second set of runs below (22:00-22:30), prefill and
decode from the same run; the 4B is not on the poster's M5 Pro panel any more. The first set (19:56-20:12) predates them and is kept. The
mlx-serve cell on the 27B is a re-run the same evening (`q38-27b-mlxserve-dflash2-20261021-222511.json`, mlx-serve 26.10.2, DFlash2); the other
engines were not re-run, so their cells are still the 2026-10-18 and 10-19 runs and are not interleaved with Kapitsa's new ones.
Run with `compare_engines.py configs/kapitsa.json --only <config> --skip agent` (and `configs/mlx-serve.json` for mlx-serve), one config at a time under the GPU lock.

| Result file | Model | Prefill 4K / 16K (tok/s) | Decode T = 0.6 / greedy (tok/s) | Flags |
| --- | --- | ---: | ---: | --- |
| `q35-4b-kapitsa-20261021-200057.json` | Qwen3.5 4B, DFlash | 3,032 / 2,736 | 217.6 / 231.5 | none |
| `q38-27b-kapitsa-20261021-195748.json` | Qwen3.8 27B, DFlash2 | 465 / 451 | 68.0 / 77.2 | none |
| `q36-35b-kapitsa-20261021-195622.json` | Qwen3.6 35B-A3B, DFlash2 | 1,823 / 1,468 | 208.1 / 233.9 | decode: repetition loop in one of five runs (prefill valid) |
| `q36-35b-kapitsa-20261021-200652.json` (second run) | Qwen3.6 35B-A3B, DFlash2 | 1,824 / 1,469 | 198.6 | none |
| `lightning-kapitsa-20261021-200211.json` | Lightning 30B, DSpark | 3,048 / 2,989 | 167.8 / 165.7 | none |

Second set (22:00-22:30, the poster):

| Result file | Model | Prefill 4K / 16K (tok/s) | Decode T = 0.6 / greedy (tok/s) | Flags |
| --- | --- | ---: | ---: | --- |
| `q36-35b-kapitsa-20261021-220021.json` | Qwen3.6 35B-A3B, DFlash2 | 1,799 / 1,455 | 195.1 / 223.4 | none |
| `q38-27b-kapitsa-20261021-221847.json` | Qwen3.8 27B, DFlash2 | 463 / 451 | 74.1 / 77.4 | none |
| `lightning-kapitsa-20261021-221051.json` | Lightning 30B, DSpark | 3,141 / 2,982 | 159.3 / 170.8 | none |
| `q38-27b-mlxserve-dflash2-20261021-222511.json` | Qwen3.8 27B, mlx-serve 26.10.2, DFlash2 | 428 / 454 | 79.9 / 89.4 | none |
| `q38-27b-kapitsa-20261021-220229.json` | Qwen3.8 27B, DFlash2 | 440 / 432 | 67.4 / 70.7 | contention: a browser used 14% of the GPU; not used, re-run as the 27B line above |

Third set (23:54, current main `3a5bbf0`, on AC power in High Power mode, 0% foreign GPU): the 4B again, the same
command and settings as the first set. The 20:00 run above is kept for the record.

| Result file | Model | Prefill 4K / 16K (tok/s) | Decode T = 0.6 / greedy (tok/s) | Flags |
| --- | --- | ---: | ---: | --- |
| `q35-4b-kapitsa-20261021-235433.json` | Qwen3.5 4B, DFlash | 3,269 / 2,735 | 226.7 / 238.8 | none |
