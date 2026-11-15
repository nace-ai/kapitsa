# Mac mini M6, 32 GB: the Qwen3.8 27B KQuant on Kapitsa and mlx-lm, 2026-11-15

Asked for in the review of [#1275](https://github.com/nace-ai/kapitsa/pull/1275): docs/profiles.md names a model's bar
engine by measured single-stream decode on that Mac, and benchmarks.md had no M6 row. This directory adds the M6's
27B cells for mlx-lm, with Kapitsa main beside it in the same session. llama.cpp, mlx-serve, mlx-vlm and the other
engines were not run.

- Mac: Mac mini (2026), Apple M6, 12-core GPU, 32 GB, macOS 27.2.1 (26B131), AC power
- Model: Kapitsa's KQuant 27B folder (`qwen3.8-27b-kquant`: MLX affine 4-bit, group 64; its MTP head and DFlash2
  drafter side-cars are read by Kapitsa and ignored by mlx-lm)
- Kapitsa: main `fe6a17af` (0.2.36), Release, `kapitsa serve --ctx 32768 --slots 1` with its DFlash2 drafter, Neural
  Engine split per this Mac's calibration (31 of 34 units). The harness records the checkout's commit (`2c3c25c`) as
  the version; the binary is main's.
- mlx-lm: 0.32.2 (MLX 0.32.4), `mlx_lm.server` with configs/mlx-lm.json's settings; no speculative decoding
- Harness: `compare_engines.py` with `m6-27b.json` (configs/mlx-lm.json's and configs/kapitsa.json's 27B entries on
  this folder), two interleaved rounds: mlx-lm, Kapitsa, mlx-lm, Kapitsa, each launch in its own hold of the GPU lock.
  Results in `results/` (home and scratch paths anonymized).

| engine | decode T 0.6 tok/s | greedy tok/s | prefill 4K tok/s | prefill 16K tok/s | cached TTFT 4K / 16K s | peak GB |
|---|---|---|---|---|---|---|
| Kapitsa | 42.8 (41.0-47.8) | 52.4 (45.9-55.8) | 473 | 403 | 0.25 / 0.29 | 20.3 |
| mlx-lm | 9.5 (9.3-9.6) | 9.6 (9.5-9.7) | 283 | 251 | 0.67 / 0.80 | 22.6 |

Both Kapitsa runs and the first mlx-lm run are flagged CONTENTION: the desktop app the run was started from (Claude
Helper) drew a mean 2.7-2.8% of the GPU during Kapitsa's prefill phases (max 11%), 1.3% during mlx-lm's (max 5.8%).
Decode phases were not flagged.

What this does not settle: mlx-lm has no speculative decoding for this model, and on the Macs in benchmarks.md the
27B's fastest other engines run MTP or a DFlash2 draft tree (mlx-serve, mlx-vlm, llama.cpp). Those were not measured
here, so this row does not show that mlx-lm is the M6's fastest other engine.
