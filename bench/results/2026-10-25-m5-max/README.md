# Results, 2026-10-25: MacBook Pro M5 Max, 128 GB (every engine at its latest release)

The M5 Max panel of the README poster, re-measured on main at `cd20dcc` and on each engine's latest release. Written by
[bench/harness](../../harness/README.md): `compare_engines.py configs/*.json --only <config> --skip tools,agent`, one
config at a time under the GPU lock, 04:13-07:01 local time. MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core
GPU, 128 GB, macOS 26.4 (25E246), on AC power.

**Builds.** Kapitsa 0.2.17, main `cd20dcc` (Release). Splash 1.4.0 (the release tarball), mlx-serve 26.10.3 (the release
binary, mlx 0.32.3), oMLX 0.7.3.dev2 (the release wheel, mlx 0.32.4), mlx-vlm 0.7.10 and mlx-lm 0.32.2 (mlx 0.32.4),
MTPLX 2.12.2, Ollama 0.40.11 (`ollama-darwin.tgz`), LM Studio llmster 0.0.25-1 with the MLX runtime 1.13.1 it selects
by default on an M5 (`mlx-llm-mac-arm64-apple-metal-nax-advsimd`) and the llama.cpp runtime 2.64.0. Against the
2026-10-18 session, mlx-serve, oMLX, mlx-vlm and Ollama are newer releases; the configs are unchanged apart from their
recorded version strings. The Bolt mlx-vlm run records its version as "unknown" (that config's auto-detection): it ran
mlx-vlm 0.7.10 from the same environment as the Qwen runs.

**Weights.** The folders `kapitsa pull` writes (Qwen3.6-35B-A3B: MTPLX's release of mlx-community's 4-bit; Qwen3.8-27B:
lmstudio-community's 4-bit with MTPLX's MTP head; the DFlash2 drafters), shared by every MLX engine. LM Studio, mlx-serve
and Ollama's MLX runner read APFS clones of the same shards (their hub entries and the vendors' MTP heads are the only
files downloaded for them). Bolt: the `txgsync` MLX bundle of NVIDIA's 24 GB checkpoint for every engine.

**Not run yet.** llama.cpp b11871 and Ollama's GGUF runner on the 27B and 35B-A3B (their GGUF tags were downloading),
Flash-Next, Flint, Lightning and Super-VL. The README keeps those cells' earlier runs, each with its build.

| Model | Engine (setting) | Version | Prefill 4K / 16K (tok/s) | Decode T = 0.6, median (range) | Greedy | Cached TTFT 4K / 16K (s) | Peak (GiB) | App GPU %, prefill / decode | Flags |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.8-27B | Kapitsa (DFlash2 drafter) | cd20dcc | 805 / 801 | 146.7 (108.3-154.6) | 146.0 | 0.13 / 0.13 | 20.1 | 7.5 / 2.4 | contention |
| Qwen3.8-27B | Splash 1.4.0 (DFlash2 drafter) | 1.4.0 | 856 / 830 | 115.0 (101.6-134.1) | 113.3 | 0.21 / 0.19 | 22.9 | 12.8 / 3.4 |  |
| Qwen3.8-27B | mlx-serve + DFlash2 (draft tree, block 8) | 26.10.3 | 643 / 567 | 112.8 (96.2-121.9) | 120.7 | 0.14 / 0.13 | 23.0 | 28.8 / 3.4 |  |
| Qwen3.8-27B | oMLX + DFlash2 (drafter 4-bit, block 5, verify dflash) | omlx 0.7.3.dev2 mlx_lm 0.32.3.dev8+g0d4d3d04f mlx 0.32.4 | 715 / 752 | 80.6 (72.9-83.1) | 84.7 | 0.31 / 0.49 | 22.0 | 4.5 / 1.8 |  |
| Qwen3.8-27B | mlx-serve + MTP (depth up to 4) | 26.10.3 | 720 / 661 | 68.2 (61.9-74.6) | 69.4 | 0.16 / 0.20 | 23.0 | 9.2 / 1.9 | swap |
| Qwen3.8-27B | MTPLX (turbo profile, depth 3) | mtplx 2.12.2 mlx_lm 0.31.3 mlx 0.32.2 | 666 / 615 | 61.1 (47.4-65.3) | 62.5 | 0.13 / 0.20 | 39.5 | 18.8 / 2.4 |  |
| Qwen3.8-27B | Ollama MLX runner (MLX 4-bit + MTP head, adaptive depth) | 0.40.11 | 676 / 648 | 53.8 (52.7-56.5) | 58.0 | 0.18 / 0.20 | 27.6 | 0.5 / 0.1 |  |
| Qwen3.8-27B | mlx-vlm server + MTP | mlx-vlm 0.7.10, mlx 0.32.4 | 673 / 657 | 48.5 (47.1-49.2) | 51.2 | 0.35 / 0.52 | 21.0 | 0.4 / 0.6 |  |
| Qwen3.8-27B | LM Studio (MLX engine) | llmster 0.0.25-1, MLX runtime 1.13.1 (mlx-engine ab52366, mlx 0.32.3, mlx-lm 0.31.4.dev132, mlx-vlm 0.7.3) | 616 / 614 | 31.2 (31.2-31.6) | 30.4 | 0.25 / 0.29 | 21.9 | 4.8 / 1.5 | 511 tokens |
| Qwen3.8-27B | mlx-lm | mlx_lm 0.32.2 mlx 0.32.4 | 574 / 577 | 29.3 (28.6-29.8) | 27.7 | 0.25 / 0.31 | 21.7 | 13.5 / 3.9 |  |
| Qwen3.8-27B | oMLX (no speculation) | omlx 0.7.3.dev2 mlx_lm 0.32.3.dev8+g0d4d3d04f mlx 0.32.4 | 655 / 648 | 28.2 (28.0-28.5) | 26.5 | 0.32 / 0.50 | 20.0 | 16.4 / 3.3 |  |
| Qwen3.6-35B-A3B | Kapitsa (DFlash2 drafter) | cd20dcc | 5,963 / 5,270 | 339.1 (322.5-360.3) | 364.5 | 0.04 / 0.04 | 21.8 | 0.0 / 0.0 |  |
| Qwen3.6-35B-A3B | Splash 1.4.0 (DFlash2 drafter) | 1.4.0 | 6,386 / 5,607 | 281.5 (262.8-318.8) | 287.9 | 0.07 / 0.07 | 22.0 | 0.0 / 0.0 |  |
| Qwen3.6-35B-A3B | mlx-serve + MTP (depth up to 2) | 26.10.3 | 3,101 / 4,847 | 238.1 (230.6-257.7) | 251.1 | 0.12 / 0.17 | 22.7 | 11.0 / 3.6 |  |
| Qwen3.6-35B-A3B | oMLX + DFlash2 (drafter 4-bit, block 5, adaptive verify) | omlx 0.7.3.dev2 mlx_lm 0.32.3.dev8+g0d4d3d04f mlx 0.32.4 | 3,005 / 3,034 | 188.7 (169.4-213.3) | 237.8 | 0.17 / 0.25 | 22.7 | 4.7 / 1.6 |  |
| Qwen3.6-35B-A3B | oMLX (no speculation) | omlx 0.7.3.dev2 mlx_lm 0.32.3.dev8+g0d4d3d04f mlx 0.32.4 | 2,484 / 3,006 | 185.9 (178.5-191.0) | 188.9 | 0.14 / 0.25 | 22.0 | 6.6 / 2.4 |  |
| Qwen3.6-35B-A3B | mlx-serve + DFlash2 (block 4) | 26.10.3 | 3,742 / 4,481 | 177.8 (158.9-182.1) | 193.6 | 0.12 / 0.18 | 22.8 | 7.5 / 1.2 |  |
| Qwen3.6-35B-A3B | Ollama MLX runner (MLX 4-bit + MTP head, adaptive depth) | 0.40.11 | 3,077 / 3,539 | 161.6 (157.6-176.1) | 166.0 | 0.09 / 0.16 | 25.3 | 0.1 / 0.0 |  |
| Qwen3.6-35B-A3B | mlx-vlm server + DFlash2 | mlx-vlm 0.7.10, mlx 0.32.4 | 4,439 / 4,415 | 137.8 (129.8-144.9) | 143.7 | 0.24 / 0.74 | 23.2 | 0.0 / 1.1 |  |
| Qwen3.6-35B-A3B | LM Studio (MLX engine) | llmster 0.0.25-1, MLX runtime 1.13.1 (mlx-engine ab52366, mlx 0.32.3, mlx-lm 0.31.4.dev132, mlx-vlm 0.7.3) | 4,672 / 4,171 | 136.3 (134.9-136.7) | 139.7 | 0.19 / 0.31 | 22.9 | 0.0 / 0.0 | 511 tokens |
| Qwen3.6-35B-A3B | MTPLX (sustained profile, depth 1) | mtplx 2.12.2 mlx_lm 0.31.3 mlx 0.32.2 | 4,291 / 3,845 | 129.8 (125.7-134.5) | 110.5 | 0.05 / 0.07 | 30.1 | 0.5 / 0.1 | loop |
| Qwen3.6-35B-A3B | mlx-lm | mlx_lm 0.32.2 mlx 0.32.4 | 4,379 / 3,871 | 115.5 (110.9-117.5) | 121.7 | 0.11 / 0.15 | 20.8 | 0.0 / 0.0 |  |
| Nemotron-3.5-Bolt-74B-A9B-24GB | Kapitsa (LiLiCorr drafter, GPU + Neural Engine prefill) | cd20dcc | 983 / 1,042 | 157.5 (145.2-160.5) | 154.0 | 0.15 / 0.16 | 22.8 | 7.0 / 2.2 | loop |
| Nemotron-3.5-Bolt-74B-A9B-24GB | mlx-vlm server (the bundle's model file, no drafter) | unknown | 814 / 723 | 68.1 (64.2-69.4) | 63.9 | 4.86 / 22.95 | 25.6 | 21.9 / 4.3 | loop, no cache reuse |
| Nemotron-3.5-Bolt-74B-A9B-24GB | oMLX (the bundle's model file, no drafter) | omlx 0.7.3.dev2 mlx_lm 0.32.3.dev8+g0d4d3d04f mlx 0.32.4 | 749 / 800 | 55.2 (54.1-56.5) | 57.5 | 0.38 / 0.45 | 25.7 | 11.9 / 3.1 | loop |
| Qwen3.5-4B | Kapitsa (DFlash drafter) | cd20dcc | 3,614 / 4,103 | 272.9 (270.7-308.1) | 318.1 | 0.03 / 0.04 | 5.0 | 4.3 / 2.2 | loop |
| Qwen3.5-4B | LM Studio (MLX engine) | llmster 0.0.25-1, MLX runtime 1.13.1 (mlx-engine ab52366, mlx 0.32.3, mlx-lm 0.31.4.dev132, mlx-vlm 0.7.3) | 4,892 / 4,525 | 155.2 (153.0-157.2) | 166.3 | 0.16 / 0.28 | 7.3 | 6.0 / 3.2 | loop, 511 tokens |
| Qwen3.5-4B | mlx-lm | mlx_lm 0.32.2 mlx 0.32.4 | 4,717 / 4,513 | 134.0 (131.2-138.2) | 145.3 | 0.08 / 0.13 | 6.1 | 3.0 / 1.4 | loop |

**Flags and the desktop app's GPU load.** *contention*: Kapitsa's 27B prefill phases ran beside the Claude desktop app's GPU process (4.6% of the GPU on average, 7.5% in its 16K prefill). From the next run until 08:05 that app was declared ambient (`BENCH_GPU_AMBIENT`, the harness's switch for the app driving a benchmark), so its load no longer blocked the quiet wait or flagged a run. It still ran: the *App GPU %* column is the worst phase's mean load from that app (ambient plus foreign) in the prefill and in the decode phases. It stayed at 4.3% or less in every decode phase, but reached 7-29% in some prefill phases (mlx-serve's 27B DFlash2 run 28.8% in its 16K prefill, mlx-vlm's Bolt run 21.8%, MTPLX's 27B 18.8%), so those runs' prefill cells were not held to the standard Kapitsa's 27B run was. Fourteen runs carried more than 5% in a prefill phase: Kapitsa's 27B (flagged) and Bolt runs, mlx-serve's four Qwen runs, MTPLX's, mlx-lm's, oMLX's and Splash's 27B, oMLX's 35B-A3B and Bolt, mlx-vlm's Bolt, and LM Studio's 4B. All but the 4B's (the 4B is off the M5 Max panel) are being run again without the exemption, the app's load blocking the quiet wait and flagging every engine alike. *loop*: one or more decode answers fell into a repetition loop (distinct 4-grams below 0.5), which flatters a
speculative decoder. *511 tokens*: LM Studio reports 511 completion tokens for a 512-token answer, as in its earlier
sessions. *no cache reuse*: mlx-vlm's follow-up turns on Bolt re-read the whole conversation. *swap*: 16 MB swapped out
in one prefill phase.

**Invalid, kept in `invalid/`.** `bolt-omlx-lilicorr`: oMLX 0.7.3.dev2 with the bundle's LiLiCorr drafter never became
ready (900 s), so oMLX's Bolt cell is its run without a drafter.

**Conditions.** Downloads for the rest of the session ran during these runs (at most ~9 MB/s), and the Qwen3.5-4B rows
were measured before that model was dropped from the M5 Max panel; they are stored, not on the poster. Several engines
read below their 2026-10-18 numbers on this Mac with the same configs (oMLX 0.7.3.dev2 most, 189 against 232 tok/s on
the 35B-A3B); none of those runs was flagged, so the configs' settings, chosen in smoke tests on the earlier releases,
may no longer be each engine's best. They were not re-tuned.

The server logs are in `logs/`. Local paths are replaced by placeholders: `~` for the home folder, `<repo>` for the
Kapitsa checkout, `<run-root>` for the run directory, `<host>` for the host name.
