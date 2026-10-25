# Results, 2026-10-25: MacBook Pro M5 Max, 128 GB (every engine at its latest release)

The M5 Max panel of the README poster, re-measured on main at `cd20dcc` and on each engine's latest release. Written by
[bench/harness](../../harness/README.md): `compare_engines.py configs/*.json --only <config> --skip tools,agent`, one
config at a time under the GPU lock. First pass 04:13-07:01 local time; the runs that carried the desktop app's GPU load
in a prefill phase again from 08:05 to 09:09 (see *Flags*); the 27B's Kapitsa and Splash cells and the Bolt row from
blocks run back to back under one lock (see *The 27B row* and *The Bolt row*). MacBook Pro (16-inch, 2026), `Mac17,6`,
M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246).

**Power.** The first pass and the re-runs up to 08:27 ran on AC power in automatic mode. From 08:44 to 09:12 the Mac ran
on battery (High Power mode), and from about 12:55 on AC in High Power mode (`powermode 2`; Low Power Mode was off). The
*Power* column gives each run's state as the harness recorded it at the run's start. No current cell comes from a battery
run; the 27B Kapitsa and Splash cells and the Bolt row are High Power mode runs on AC, each row's engines alike.

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

**The 27B row: one block of three runs each.** Kapitsa's newest flag-free 27B run (129.8 tok/s at T = 0.6) and Splash's
(130.5) were 0.5% apart, inside either engine's run-to-run spread. Decided before running: Kapitsa and Splash in turn
(K S K S K S), each cell the median of its three runs, prefill and decode each their own median, whatever they read. A
first set of three pairs (09:38-13:11) was spread over three hours between another session's GPU jobs, every one of its
prefill phases carried the desktop app's GPU load (5.8-14.4%), and its third pair ran in High Power mode; it is listed
under *Superseded runs*. The block that replaces it ran 16:41-16:56 under one GPU lock, on AC power in High Power mode,
with the app's load at 0.2% or less in every prefill phase and no flag on any run.

| | Kapitsa | Splash |
| --- | ---: | ---: |
| Decode T = 0.6, the three runs | 131.8, 130.0, 128.8 | 92.6, 105.2, 112.4 |
| **Decode T = 0.6, median** | **130.0** | 105.2 |
| Greedy, median | **143.6** | 110.5 |
| Prefill 4K, median | 801 | **948** |
| Prefill 16K, median | 813 | **913** |

Kapitsa writes 24% faster (30% greedy) and Splash reads prompts 18% faster at 4K, 12% at 16K. Splash's decode runs here
(92.6-112.4) read below its earlier ones (115.0-133.6, superseded), and its answers range 89.8-119.3 tok/s; Kapitsa's
range 110.0-153.9.

A first attempt at pair 1 (09:11) was stopped when the other session's GPU work took priority; it is in `aborted/` and
not counted. Its Kapitsa run finished: 900 / 51.0, three of its five answers at 34-51 tok/s with no foreign GPU load in
its decode phase (0.55%; its 16K prefill carries the contention flag); the cause is not known (the harness's contention
check measures GPU load only, not CPU or memory traffic). Splash's run was stopped during warm-up.

**The Bolt row: one block.** The first-pass Bolt runs let the app's load through (7-22% in a prefill phase), and their
repeats (08:51-09:09) ran on battery. The row's three engines ran again back to back, 13:27-13:49, on AC in High Power
mode. All three carry the contention flag in their prefill phases (the app's load 7.4-10.3%), alike; their decode
phases carried 2.6% or less. The block is the row.

**Current runs.** Per config: in a block (the 27B Kapitsa and Splash cells, the Bolt row), the block; otherwise its
newest run on AC power without a contention flag on its 4K prefill or decode phase that is not one of the first-pass
runs repeated (see *Flags*). Three configs have no such run and appear only under *Superseded runs*: mlx-lm's 27B (its
repeat flagged at 10.7%) and oMLX's 27B and 35B-A3B without speculation (their repeats on battery); none is a poster
cell.

| Model | Engine (setting) | Version | Started | Prefill 4K / 16K (tok/s) | Decode T = 0.6, median (range) | Greedy | Cached TTFT 4K / 16K (s) | Peak (GiB) | App GPU %, prefill / decode | Power | Flags |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Qwen3.8-27B | Kapitsa (DFlash2 drafter), block run 1 | cd20dcc | 16:41 | 787 / 813 | 131.8 (117.5-153.9) | 143.6 | 0.12 / 0.13 | 20.1 | 0.2 / 0.0 | AC, high power |  |
| Qwen3.8-27B | Kapitsa (DFlash2 drafter), block run 2 | cd20dcc | 16:45 | 861 / 798 | 130.0 (121.1-137.6) | 145.2 | 0.12 / 0.14 | 20.1 | 0.1 / 0.1 | AC, high power |  |
| Qwen3.8-27B | Kapitsa (DFlash2 drafter), block run 3 | cd20dcc | 16:50 | 801 / 814 | 128.8 (110.0-134.9) | 135.4 | 0.12 / 0.14 | 20.1 | 0.1 / 1.6 | AC, high power |  |
| Qwen3.8-27B | Kapitsa (DFlash2 drafter), **median of the block** | cd20dcc |  | 801 / 813 | 130.0 | 143.6 | 0.12 / 0.14 | 20.1 |  |  |  |
| Qwen3.8-27B | mlx-serve + DFlash2 (draft tree, block 8) | 26.10.3 | 08:22 | 715 / 638 | 120.1 (111.5-144.7) | 131.8 | 0.08 / 0.12 | 23.1 | 2.1 / 0.2 | AC, automatic |  |
| Qwen3.8-27B | Splash 1.4.0 (DFlash2 drafter), block run 1 | 1.4.0 | 16:43 | 948 / 928 | 92.6 (89.8-99.6) | 101.3 | 0.17 / 0.17 | 22.6 | 0.0 / 0.0 | AC, high power |  |
| Qwen3.8-27B | Splash 1.4.0 (DFlash2 drafter), block run 2 | 1.4.0 | 16:47 | 948 / 910 | 105.2 (98.4-108.9) | 110.5 | 0.15 / 0.19 | 22.7 | 0.2 / 0.0 | AC, high power |  |
| Qwen3.8-27B | Splash 1.4.0 (DFlash2 drafter), block run 3 | 1.4.0 | 16:52 | 902 / 913 | 112.4 (97.8-119.3) | 112.7 | 0.17 / 0.18 | 22.7 | 0.2 / 0.0 | AC, high power |  |
| Qwen3.8-27B | Splash 1.4.0 (DFlash2 drafter), **median of the block** | 1.4.0 |  | 948 / 913 | 105.2 | 110.5 | 0.17 / 0.18 | 22.7 |  |  |  |
| Qwen3.8-27B | oMLX + DFlash2 (drafter 4-bit, block 5, verify dflash) | omlx 0.7.3.dev2 mlx_lm 0.32.3.dev8+g0d4d3d04f mlx 0.32.4 | 05:05 | 715 / 752 | 80.6 (72.9-83.1) | 84.7 | 0.31 / 0.49 | 22.0 | 4.5 / 1.8 | AC, automatic |  |
| Qwen3.8-27B | mlx-serve + MTP (depth up to 4) | 26.10.3 | 08:24 | 717 / 659 | 68.4 (64.5-69.2) | 68.7 | 0.12 / 0.16 | 22.9 | 2.3 / 0.1 | AC, automatic |  |
| Qwen3.8-27B | MTPLX (turbo profile, depth 3) | mtplx 2.12.2 mlx_lm 0.31.3 mlx 0.32.2 | 08:17 | 709 / 663 | 61.9 (57.6-68.3) | 68.3 | 0.13 / 0.16 | 38.4 | 1.4 / 0.8 | AC, automatic |  |
| Qwen3.8-27B | Ollama MLX runner (MLX 4-bit + MTP head, adaptive depth) | 0.40.11 | 06:01 | 676 / 648 | 53.8 (52.7-56.5) | 58.0 | 0.18 / 0.20 | 27.6 | 0.5 / 0.1 | AC, automatic |  |
| Qwen3.8-27B | mlx-vlm server + MTP | mlx-vlm 0.7.10, mlx 0.32.4 | 05:08 | 673 / 657 | 48.5 (47.1-49.2) | 51.2 | 0.35 / 0.52 | 21.0 | 0.4 / 0.6 | AC, automatic |  |
| Qwen3.8-27B | LM Studio (MLX engine) | llmster 0.0.25-1, MLX runtime 1.13.1 (mlx-engine ab52366, mlx 0.32.3, mlx-lm 0.31.4.dev132, mlx-vlm 0.7.3) | 05:50 | 616 / 614 | 31.2 (31.2-31.6) | 30.4 | 0.25 / 0.29 | 21.9 | 4.8 / 1.5 | AC, automatic | 511 tokens |
| Qwen3.6-35B-A3B | Kapitsa (DFlash2 drafter) | cd20dcc | 04:13 | 5,963 / 5,270 | 339.1 (322.5-360.3) | 364.5 | 0.04 / 0.04 | 21.8 | 0.0 / 0.0 | AC, automatic |  |
| Qwen3.6-35B-A3B | Splash 1.4.0 (DFlash2 drafter) | 1.4.0 | 04:36 | 6,386 / 5,607 | 281.5 (262.8-318.8) | 287.9 | 0.07 / 0.07 | 22.0 | 0.0 / 0.0 | AC, automatic |  |
| Qwen3.6-35B-A3B | mlx-serve + MTP (depth up to 2) | 26.10.3 | 08:21 | 4,622 / 5,369 | 235.5 (231.7-245.1) | 254.9 | 0.13 / 0.16 | 22.7 | 0.0 / 0.0 | AC, automatic |  |
| Qwen3.6-35B-A3B | mlx-serve + DFlash2 (block 4) | 26.10.3 | 08:11 | 5,798 / 5,300 | 203.4 (200.5-237.0) | 216.5 | 0.12 / 0.17 | 22.9 | 4.3 / 1.4 | AC, automatic |  |
| Qwen3.6-35B-A3B | oMLX + DFlash2 (drafter 4-bit, block 5, adaptive verify) | omlx 0.7.3.dev2 mlx_lm 0.32.3.dev8+g0d4d3d04f mlx 0.32.4 | 04:59 | 3,005 / 3,034 | 188.7 (169.4-213.3) | 237.8 | 0.17 / 0.25 | 22.7 | 4.7 / 1.6 | AC, automatic |  |
| Qwen3.6-35B-A3B | Ollama MLX runner (MLX 4-bit + MTP head, adaptive depth) | 0.40.11 | 06:00 | 3,077 / 3,539 | 161.6 (157.6-176.1) | 166.0 | 0.09 / 0.16 | 25.3 | 0.1 / 0.0 | AC, automatic |  |
| Qwen3.6-35B-A3B | mlx-vlm server + DFlash2 | mlx-vlm 0.7.10, mlx 0.32.4 | 04:32 | 4,439 / 4,415 | 137.8 (129.8-144.9) | 143.7 | 0.24 / 0.74 | 23.2 | 0.0 / 1.1 | AC, automatic |  |
| Qwen3.6-35B-A3B | LM Studio (MLX engine) | llmster 0.0.25-1, MLX runtime 1.13.1 (mlx-engine ab52366, mlx 0.32.3, mlx-lm 0.31.4.dev132, mlx-vlm 0.7.3) | 04:18 | 4,672 / 4,171 | 136.3 (134.9-136.7) | 139.7 | 0.19 / 0.31 | 22.9 | 0.0 / 0.0 | AC, automatic | 511 tokens |
| Qwen3.6-35B-A3B | MTPLX (sustained profile, depth 1) | mtplx 2.12.2 mlx_lm 0.31.3 mlx 0.32.2 | 04:14 | 4,291 / 3,845 | 129.8 (125.7-134.5) | 110.5 | 0.05 / 0.07 | 30.1 | 0.5 / 0.1 | AC, automatic | loop |
| Qwen3.6-35B-A3B | mlx-lm | mlx_lm 0.32.2 mlx 0.32.4 | 04:17 | 4,379 / 3,871 | 115.5 (110.9-117.5) | 121.7 | 0.11 / 0.15 | 20.8 | 0.0 / 0.0 | AC, automatic |  |
| Nemotron-3.5-Bolt-74B-A9B-24GB | Kapitsa (LiLiCorr drafter, GPU + Neural Engine prefill), Bolt block | cd20dcc | 13:27 | 1,014 / 1,041 | 155.8 (143.5-163.9) | 162.0 | 0.14 / 0.16 | 22.8 | 9.8 / 2.2 | AC, high power | contention (4K prefill, 16K prefill), loop |
| Nemotron-3.5-Bolt-74B-A9B-24GB | mlx-vlm server (the bundle's model file, no drafter), Bolt block | unknown | 13:41 | 943 / 1,015 | 70.9 (69.5-71.0) | 69.1 | 4.25 / 16.19 | 25.6 | 10.3 / 2.1 | AC, high power | contention (4K prefill, 16K prefill), no cache reuse |
| Nemotron-3.5-Bolt-74B-A9B-24GB | oMLX (the bundle's model file, no drafter), Bolt block | omlx 0.7.3.dev2 mlx_lm 0.32.3.dev8+g0d4d3d04f mlx 0.32.4 | 13:33 | 878 / 934 | 70.5 (67.6-71.0) | 66.8 | 0.33 / 0.40 | 25.7 | 8.0 / 2.6 | AC, high power | contention (4K prefill, 16K prefill) |
| Qwen3.5-4B | Kapitsa (DFlash drafter) | cd20dcc | 05:55 | 3,614 / 4,103 | 272.9 (270.7-308.1) | 318.1 | 0.03 / 0.04 | 5.0 | 4.3 / 2.2 | AC, automatic | loop |
| Qwen3.5-4B | LM Studio (MLX engine) | llmster 0.0.25-1, MLX runtime 1.13.1 (mlx-engine ab52366, mlx 0.32.3, mlx-lm 0.31.4.dev132, mlx-vlm 0.7.3) | 06:40 | 4,892 / 4,525 | 155.2 (153.0-157.2) | 166.3 | 0.16 / 0.28 | 7.3 | 6.0 / 3.2 | AC, automatic | loop, 511 tokens |
| Qwen3.5-4B | mlx-lm | mlx_lm 0.32.2 mlx 0.32.4 | 05:56 | 4,717 / 4,513 | 134.0 (131.2-138.2) | 145.3 | 0.08 / 0.13 | 6.1 | 3.0 / 1.4 | AC, automatic | loop |

**Flags and the desktop app's GPU load.** *contention*: Kapitsa's 27B prefill phases ran beside the Claude desktop app's GPU process (4.6% of the GPU on average, 7.5% in its 16K prefill). From the next run until 08:05 that app was declared ambient (`BENCH_GPU_AMBIENT`, the harness's switch for the app driving a benchmark), so its load no longer blocked the quiet wait or flagged a run. It still ran: the *App GPU %* column is the worst phase's mean load from that app (ambient plus foreign) in the prefill and in the decode phases. It stayed at 4.3% or less in every decode phase, but reached 7-29% in some prefill phases (mlx-serve's 27B DFlash2 run 28.8% in its 16K prefill, mlx-vlm's Bolt run 21.8%, MTPLX's 27B 18.8%), so those runs' prefill cells were not held to the standard Kapitsa's 27B run was. Fourteen runs carried more than 5% in a prefill phase: Kapitsa's 27B (flagged) and Bolt runs, mlx-serve's four Qwen runs, MTPLX's, mlx-lm's, oMLX's and Splash's 27B, oMLX's 35B-A3B and Bolt, mlx-vlm's Bolt, and LM Studio's 4B. All but the 4B's (the 4B is off the M5 Max panel) were run again from 08:05 to 09:09 without the exemption, the app's load blocking the quiet wait and flagging every engine alike; those re-runs replace the first runs (listed under *Superseded runs*), where they qualify (see *Current runs*). LM Studio's 4B run, never repeated, stays with its 6.0%; the 4B is not on the M5 Max panel. *loop*: one or more decode answers fell into a repetition loop (distinct 4-grams below 0.5), which flatters a
speculative decoder. *511 tokens*: LM Studio reports 511 completion tokens for a 512-token answer, as in its earlier
sessions. *no cache reuse*: mlx-vlm's follow-up turns on Bolt re-read the whole conversation. *swap*: 16 MB swapped out
in one prefill phase.

**Superseded runs.** The runs the re-runs and the blocks replaced, and the runs that do not qualify (battery,
flagged), kept for the record (*App GPU %* includes the load the exemption let through).

| Model | Engine (setting) | Version | Started | Prefill 4K / 16K (tok/s) | Decode T = 0.6, median (range) | Greedy | Cached TTFT 4K / 16K (s) | Peak (GiB) | App GPU %, prefill / decode | Power | Flags |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Qwen3.8-27B | Kapitsa (DFlash2 drafter) | cd20dcc | 04:53 | 805 / 801 | 146.7 (108.3-154.6) | 146.0 | 0.13 / 0.13 | 20.1 | 7.5 / 2.4 | AC, automatic | contention (4K prefill, 16K prefill) |
| Qwen3.8-27B | Kapitsa (DFlash2 drafter) | cd20dcc | 08:05 | 771 / 794 | 138.3 (126.0-150.7) | 141.3 | 0.13 / 0.14 | 20.1 | 6.2 / 3.4 | AC, automatic | contention (4K prefill) |
| Qwen3.8-27B | Kapitsa (DFlash2 drafter) | cd20dcc | 09:09 | 909 / 892 | 129.8 (122.1-148.8) | 144.9 | 0.11 / 0.13 | 20.1 | 0.0 / 0.1 | battery, high power |  |
| Qwen3.8-27B | Kapitsa (DFlash2 drafter) | cd20dcc | 09:38 | 742 / 775 | 142.3 (125.5-156.5) | 143.3 | 0.13 / 0.14 | 20.1 | 14.4 / 4.9 | AC, automatic | contention (4K prefill, 16K prefill) |
| Qwen3.8-27B | Kapitsa (DFlash2 drafter) | cd20dcc | 12:48 | 752 / 727 | 115.2 (103.5-120.9) | 131.2 | 0.13 / 0.14 | 20.1 | 6.8 / 3.0 | AC, automatic | contention (4K prefill, 16K prefill) |
| Qwen3.8-27B | Kapitsa (DFlash2 drafter) | cd20dcc | 12:58 | 776 / 792 | 135.4 (121.7-140.5) | 143.7 | 0.12 / 0.13 | 20.1 | 9.7 / 2.7 | AC, high power | contention (4K prefill, 16K prefill) |
| Qwen3.8-27B | MTPLX (turbo profile, depth 3) | mtplx 2.12.2 mlx_lm 0.31.3 mlx 0.32.2 | 05:11 | 666 / 615 | 61.1 (47.4-65.3) | 62.5 | 0.13 / 0.20 | 39.5 | 18.8 / 2.4 | AC, automatic |  |
| Qwen3.8-27B | Splash 1.4.0 (DFlash2 drafter) | 1.4.0 | 05:01 | 856 / 830 | 115.0 (101.6-134.1) | 113.3 | 0.21 / 0.19 | 22.9 | 12.8 / 3.4 | AC, automatic |  |
| Qwen3.8-27B | Splash 1.4.0 (DFlash2 drafter) | 1.4.0 | 08:13 | 866 / 926 | 130.5 (118.4-133.7) | 122.5 | 0.16 / 0.22 | 22.9 | 0.3 / 0.4 | AC, automatic |  |
| Qwen3.8-27B | Splash 1.4.0 (DFlash2 drafter) | 1.4.0 | 12:40 | 867 / 909 | 126.8 (113.4-137.1) | 124.0 | 0.20 / 0.19 | 22.7 | 9.8 / 2.5 | AC, automatic | contention (4K prefill, 16K prefill) |
| Qwen3.8-27B | Splash 1.4.0 (DFlash2 drafter) | 1.4.0 | 12:50 | 850 / 900 | 133.6 (114.9-136.6) | 123.9 | 0.20 / 0.20 | 22.7 | 12.3 / 6.1 | AC, automatic | contention (4K prefill, 16K prefill, greedy) |
| Qwen3.8-27B | Splash 1.4.0 (DFlash2 drafter) | 1.4.0 | 13:04 | 860 / 915 | 119.3 (109.0-145.0) | 122.0 | 0.18 / 0.20 | 22.7 | 9.6 / 6.1 | AC, high power | contention (4K prefill, 16K prefill, decode, greedy) |
| Qwen3.8-27B | mlx-lm | mlx_lm 0.32.2 mlx 0.32.4 | 05:31 | 574 / 577 | 29.3 (28.6-29.8) | 27.7 | 0.25 / 0.31 | 21.7 | 13.5 / 3.9 | AC, automatic |  |
| Qwen3.8-27B | mlx-lm | mlx_lm 0.32.2 mlx 0.32.4 | 08:27 | 643 / 675 | 31.0 (17.9-31.3) | 29.7 | 0.24 / 0.30 | 20.7 | 10.7 / 2.4 | AC, automatic | contention (4K prefill, 16K prefill) |
| Qwen3.8-27B | mlx-serve + DFlash2 (draft tree, block 8) | 26.10.3 | 05:22 | 643 / 567 | 112.8 (96.2-121.9) | 120.7 | 0.14 / 0.13 | 23.0 | 28.8 / 3.4 | AC, automatic |  |
| Qwen3.8-27B | mlx-serve + MTP (depth up to 4) | 26.10.3 | 05:25 | 720 / 661 | 68.2 (61.9-74.6) | 69.4 | 0.16 / 0.20 | 23.0 | 9.2 / 1.9 | AC, automatic | swap |
| Qwen3.8-27B | oMLX (no speculation) | omlx 0.7.3.dev2 mlx_lm 0.32.3.dev8+g0d4d3d04f mlx 0.32.4 | 05:43 | 655 / 648 | 28.2 (28.0-28.5) | 26.5 | 0.32 / 0.50 | 20.0 | 16.4 / 3.3 | AC, automatic |  |
| Qwen3.8-27B | oMLX (no speculation) | omlx 0.7.3.dev2 mlx_lm 0.32.3.dev8+g0d4d3d04f mlx 0.32.4 | 08:45 | 809 / 834 | 29.3 (28.6-30.6) | 29.7 | 0.32 / 0.45 | 20.0 | 6.0 / 1.2 | battery, high power | contention (16K prefill) |
| Qwen3.6-35B-A3B | mlx-serve + DFlash2 (block 4) | 26.10.3 | 05:19 | 3,742 / 4,481 | 177.8 (158.9-182.1) | 193.6 | 0.12 / 0.18 | 22.8 | 7.5 / 1.2 | AC, automatic |  |
| Qwen3.6-35B-A3B | mlx-serve + MTP (depth up to 2) | 26.10.3 | 05:16 | 3,101 / 4,847 | 238.1 (230.6-257.7) | 251.1 | 0.12 / 0.17 | 22.7 | 11.0 / 3.6 | AC, automatic |  |
| Qwen3.6-35B-A3B | oMLX (no speculation) | omlx 0.7.3.dev2 mlx_lm 0.32.3.dev8+g0d4d3d04f mlx 0.32.4 | 05:38 | 2,484 / 3,006 | 185.9 (178.5-191.0) | 188.9 | 0.14 / 0.25 | 22.0 | 6.6 / 2.4 | AC, automatic |  |
| Qwen3.6-35B-A3B | oMLX (no speculation) | omlx 0.7.3.dev2 mlx_lm 0.32.3.dev8+g0d4d3d04f mlx 0.32.4 | 08:44 | 2,631 / 3,318 | 195.0 (192.4-199.7) | 202.5 | 0.15 / 0.25 | 22.0 | 0.0 / 0.0 | battery, high power |  |
| Nemotron-3.5-Bolt-74B-A9B-24GB | Kapitsa (LiLiCorr drafter, GPU + Neural Engine prefill) | cd20dcc | 06:42 | 983 / 1,042 | 157.5 (145.2-160.5) | 154.0 | 0.15 / 0.16 | 22.8 | 7.0 / 2.2 | AC, automatic | loop |
| Nemotron-3.5-Bolt-74B-A9B-24GB | Kapitsa (LiLiCorr drafter, GPU + Neural Engine prefill) | cd20dcc | 08:51 | 1,032 / 1,058 | 147.7 (142.2-159.2) | 154.8 | 0.15 / 0.16 | 22.8 | 5.8 / 1.5 | battery, high power | contention (16K prefill) |
| Nemotron-3.5-Bolt-74B-A9B-24GB | mlx-vlm server (the bundle's model file, no drafter) | unknown | 06:56 | 814 / 723 | 68.1 (64.2-69.4) | 63.9 | 4.86 / 22.95 | 25.6 | 21.9 / 4.3 | AC, automatic | loop, no cache reuse |
| Nemotron-3.5-Bolt-74B-A9B-24GB | mlx-vlm server (the bundle's model file, no drafter) | unknown | 09:04 | 976 / 1,054 | 70.3 (70.2-70.4) | 69.8 | 4.18 / 15.90 | 25.6 | 0.1 / 0.0 | battery, high power | loop, no cache reuse |
| Nemotron-3.5-Bolt-74B-A9B-24GB | oMLX (the bundle's model file, no drafter) | omlx 0.7.3.dev2 mlx_lm 0.32.3.dev8+g0d4d3d04f mlx 0.32.4 | 06:53 | 749 / 800 | 55.2 (54.1-56.5) | 57.5 | 0.38 / 0.45 | 25.7 | 11.9 / 3.1 | AC, automatic | loop |
| Nemotron-3.5-Bolt-74B-A9B-24GB | oMLX (the bundle's model file, no drafter) | omlx 0.7.3.dev2 mlx_lm 0.32.3.dev8+g0d4d3d04f mlx 0.32.4 | 08:57 | 887 / 1,003 | 69.6 (67.4-71.7) | 67.9 | 0.36 / 0.42 | 25.7 | 5.6 / 1.5 | battery, high power | contention (16K prefill) |

**Invalid, kept in `invalid/`.** `bolt-omlx-lilicorr`: oMLX 0.7.3.dev2 with the bundle's LiLiCorr drafter never became
ready (900 s), so oMLX's Bolt cell is its run without a drafter.

**Conditions.** Downloads for the rest of the session ran during these runs (at most ~9 MB/s), and the Qwen3.5-4B rows
were measured before that model was dropped from the M5 Max panel; they are stored, not on the poster. Several engines
read below their 2026-10-18 numbers on this Mac with the same configs (oMLX 0.7.3.dev2 most: its DFlash2 config
read 189 against 232 tok/s on the 35B-A3B); none of those runs was flagged, so the configs' settings, chosen in smoke tests on the earlier releases,
may no longer be each engine's best. They were not re-tuned.

The server logs are in `logs/` (the aborted attempt's in `aborted/`). Local paths are replaced by placeholders: `~` for the home folder, `<repo>` for the
Kapitsa checkout, `<run-root>` for the run directory, `<host>` for the host name.
