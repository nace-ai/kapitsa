# Results, 2026-10-25/26 overnight: MacBook Pro M5 Max, 128 GB

The runs after the [2026-10-25 session](../2026-10-25-m5-max/README.md) and [its re-runs](../2026-10-25-m5-max/2026-10-26-reruns.md):
Flint 40B for the first time on this Mac, Kapitsa on current main for the models whose code changed, the 27B's
interleaved Kapitsa/mlx-serve pairs, and the 35B-A3B row again. Same Mac (MacBook Pro 16-inch 2026, `Mac17,6`, M5 Max,
40-core GPU, 128 GB, macOS 26.4), same harness (`compare_engines.py configs/*.json --only <config> --skip tools,agent`, one
config at a time under the GPU lock). Every run is on AC power in High Power mode (`powermode 2`, Low Power Mode off).

**The quiet-GPU rule.** A run whose worst phase carries more than about 2% of the desktop app's GPU load is contended
and is not a cell. The Claude desktop app draws GPU whenever one of its sessions is active (5-10% here), so three
attempts at the same blocks (03:18-03:55, 04:04-05:44) are contended and only the last (05:47-06:12, every phase at
0.7% or less) gives cells. All attempts are in the table below.

**Flint 40B-A2.5B, the poster's new M5 Max row** (one block at 05:47-05:50, at the card's T = 1.0): Kapitsa 0.2.18
(`ed2b07d`) **5,721 / 267.0** tok/s (prefill 4K / decode), rec 0.1.0 2,863 / 224.4, llama.cpp b11871 3,827 / 129.6.
Kapitsa writes 19% faster than rec in this block (1.06-1.25x over the night's runs: rec read 214.5-243.6, Kapitsa 258.8-267.0, so the ranking holds in every pairing) and 2.1x llama.cpp, and reads prompts 49% faster than llama.cpp. rec's cached
follow-up returned no content in this run (*empty follow-up*), as two of Kapitsa's earlier 0.2.18 attempts did (03:18,
03:58, both contended); its 05:47 run answered every follow-up.

**The 27B: Kapitsa against mlx-serve's draft tree, three interleaved pairs** (K M K M K M, 05:58-06:11, decided before
running): Kapitsa 127.5, 121.4, 123.2 (median 123.2); mlx-serve DFlash2 119.4, 131.3, 124.5 (median 124.5). Each median
lies inside the other engine's run range, so the row stays a tie, as the poster marks it. Both engines read below their
20:44 block (133.2 and 127.9): the Mac had run benchmarks for most of the night; the pairs compare in one state.

**The 35B-A3B: not on the poster.** The row ran as one block at 02:12-03:02 (every engine but Kapitsa and Splash quiet;
four of the six Kapitsa/Splash runs contended) and Kapitsa/Splash again as a quiet K S K S K S block at 05:51-05:57:
Kapitsa 296.5, 320.2, 289.1 (median 296.5), Splash 284.6, 260.9, 269.1 (median 269.1), prefill medians 4,976 and 5,677.
Both read 10-15% below every earlier run of theirs on this Mac, in prefill and decode alike, so these cells do not
compare with the rest of the row, measured hours earlier. The poster keeps the 35B-A3B row of the 2026-10-25 first
pass (automatic mode, one session), and the header with it; llama.cpp b11871's 35B-A3B runs (n-max 3: 2,722 / 162.3)
are recorded here, not put next to it.

**Flash-Next, Lightning and Super-VL on current main (`61a4c2d`)**, Kapitsa's code for Flash-Next having changed since
`cd20dcc`: Flash-Next 1,534 / 100.4, Lightning 4,884 / 260.4, Super-VL 1,163 / 93.0. The other engines' weights for
these models are not on this Mac (only oMLX's and mlx-vlm's runs ran), so the rows cannot be measured whole; their
poster cells stay the 2026-10-18 to 10-22 runs.

| Started | Model | Engine (setting) | Version | Prefill 4K / 16K (tok/s) | Decode T 0.6 (Flint T 1.0), median | Greedy | App GPU %, worst phase | Power | Flags |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2026-10-25 13:21 | Flint 40B-A2.5B | Kapitsa | 61a4c2d | 5,663 / 5,032 | 261.1 | 263.7 | 3.3 | AC, high power |  |
| 2026-10-25 16:54 | Qwen3.8-Flash-Next | Kapitsa (MTP, default depth) | 61a4c2d | 1,534 / 1,529 | 100.4 | 102.9 | 0.2 | AC, high power |  |
| 2026-10-25 16:56 | Nemotron-3.5-Lightning-30B-A3B-NVFP4 | Kapitsa (DSpark drafter) | 61a4c2d | 4,884 / 6,210 | 260.4 | 286.3 | 0.1 | AC, high power |  |
| 2026-10-25 16:57 | Nemotron-3.5-Super-VL-120B-A12B-NVFP4 | Kapitsa (LiLiCorr drafter, GPU + Neural Engine prefill) | 61a4c2d | 1,163 / 1,234 | 93.0 | 91.1 | 0.1 | AC, high power | loop |
| 2026-10-25 16:59 | Qwen3.6-35B-A3B | Ollama GGUF + MTP (shipped depth 2) | 0.40.11 | 2,702 / 2,614 | 169.5 | 172.7 | 0.5 | AC, high power |  |
| 2026-10-25 22:06 | Qwen3.6-35B-A3B | llama.cpp + MTP (n-max 1) | 0.7.0-dev (b11871, b3fa69955) | 2,791 / 3,039 | 157.4 | 159.5 | 0.1 | AC, high power | loop |
| 2026-10-25 22:07 | Qwen3.6-35B-A3B | llama.cpp + MTP (n-max 2) | 0.7.0-dev (b11871, b3fa69955) | 3,242 / 2,857 | 174.9 | 180.2 | 0.2 | AC, high power |  |
| 2026-10-25 22:09 | Qwen3.6-35B-A3B | llama.cpp + MTP (n-max 3) | 0.7.0-dev (b11871, b3fa69955) | 2,936 / 2,875 | 176.6 | 175.3 | 0.6 | AC, high power |  |
| 2026-10-25 22:10 | Flint 40B-A2.5B | llama.cpp b11804 | 0.7.0-dev (b11871, b3fa69955) | 3,793 / 3,290 | 127.6 | 129.0 | 0.1 | AC, high power |  |
| 2026-10-25 22:12 | Flint 40B-A2.5B | rec 0.1.0 | 0.1.0 | 4,121 / 3,684 | 242.5 | 257.0 | 0.1 | AC, high power |  |
| 2026-10-25 22:13 | Qwen3.8-Flash-Next | oMLX + MTP (Lightning MTP, fixed depth 3) | omlx 0.7.3.dev2 mlx_lm 0.32.3.dev8+g0d4d3d04f mlx 0.32.4 | 1,095 / 1,150 | 106.3 | 94.1 | 0.0 | AC, high power |  |
| 2026-10-25 22:16 | Qwen3.8-Flash-Next | oMLX (no speculation) | omlx 0.7.3.dev2 mlx_lm 0.32.3.dev8+g0d4d3d04f mlx 0.32.4 | 1,155 / 1,179 | 59.6 | 60.7 | 0.1 | AC, high power |  |
| 2026-10-25 22:20 | Qwen3.8-Flash-Next | mlx-vlm server (no speculation) | mlx-vlm 0.7.10, mlx 0.32.4 | 411 / 716 | 52.6 | 55.9 | 0.1 | AC, high power |  |
| 2026-10-25 22:23 | Nemotron-3.5-Lightning-30B-A3B-NVFP4 | mlx-vlm server + DSpark drafter (block 6) | mlx-vlm 0.7.10, mlx 0.32.4 | 2,788 / 2,920 | 130.6 | 140.8 | 0.1 | AC, high power |  |
| 2026-10-25 22:25 | Nemotron-3.5-Lightning-30B-A3B-NVFP4 | mlx-vlm server (git main) + native MTP (block 4) | mlx-vlm git main 73d366e (unreleased; its metadata says 0.7.8), mlx 0.32.4 | 2,972 / 3,098 | 127.2 | 135.8 | 0.0 | AC, high power |  |
| 2026-10-25 22:27 | Nemotron-3.5-Super-VL-120B-A12B-NVFP4 | mlx-vlm server (git main, native MTP, block 4) | mlx-vlm git main 73d366e (unreleased; its metadata says 0.7.8), mlx 0.32.4 | 828 / – | 46.5 | 51.8 | 0.4 | AC, high power | loop |
| 2026-10-26 02:01 | Flint 40B-A2.5B | Kapitsa | 61a4c2d | 5,751 / 5,035 | 263.2 | 265.8 | 2.6 | AC, high power |  |
| 2026-10-26 02:03 | Flint 40B-A2.5B | llama.cpp b11804 | 0.7.0-dev (b11871, b3fa69955) | 3,578 / 3,042 | 129.0 | 124.2 | 9.0 | AC, high power | contention |
| 2026-10-26 02:05 | Flint 40B-A2.5B | rec 0.1.0 | 0.1.0 | 3,530 / 2,975 | 241.4 | 256.5 | 12.3 | AC, high power | contention, empty follow-up |
| 2026-10-26 02:12 | Qwen3.6-35B-A3B | Kapitsa (DFlash2 drafter) | cd20dcc | 6,018 / 5,288 | 343.2 | 373.3 | 0.4 | AC, high power |  |
| 2026-10-26 02:16 | Qwen3.6-35B-A3B | Splash 1.4.0 (DFlash2 drafter) | 1.4.0 | 6,645 / 5,895 | 263.0 | 309.3 | 11.5 | AC, high power | contention |
| 2026-10-26 02:23 | Qwen3.6-35B-A3B | Kapitsa (DFlash2 drafter) | cd20dcc | 6,111 / 5,303 | 340.6 | 374.8 | 6.8 | AC, high power | contention |
| 2026-10-26 02:29 | Qwen3.6-35B-A3B | Splash 1.4.0 (DFlash2 drafter) | 1.4.0 | 6,813 / 6,032 | 271.7 | 299.0 | 5.9 | AC, high power | contention |
| 2026-10-26 02:35 | Qwen3.6-35B-A3B | Kapitsa (DFlash2 drafter) | cd20dcc | 6,009 / 5,199 | 345.1 | 369.0 | 6.4 | AC, high power | contention |
| 2026-10-26 02:40 | Qwen3.6-35B-A3B | Splash 1.4.0 (DFlash2 drafter) | 1.4.0 | 6,625 / 5,977 | 284.4 | 302.2 | 0.0 | AC, high power |  |
| 2026-10-26 02:43 | Qwen3.6-35B-A3B | mlx-serve + MTP (depth up to 2) | 26.10.3 | 5,137 / 5,017 | 263.6 | 275.3 | 0.0 | AC, high power |  |
| 2026-10-26 02:44 | Qwen3.6-35B-A3B | mlx-serve + DFlash2 (block 4) | 26.10.3 | 4,078 / 4,970 | 194.4 | 200.4 | 0.0 | AC, high power | loop |
| 2026-10-26 02:45 | Qwen3.6-35B-A3B | oMLX + DFlash2 (drafter 4-bit, block 5, adaptive verify) | omlx 0.7.3.dev2 mlx_lm 0.32.3.dev8+g0d4d3d04f mlx 0.32.4 | 2,560 / 3,177 | 215.8 | 209.5 | 0.0 | AC, high power |  |
| 2026-10-26 02:47 | Qwen3.6-35B-A3B | oMLX (no speculation) | omlx 0.7.3.dev2 mlx_lm 0.32.3.dev8+g0d4d3d04f mlx 0.32.4 | 2,703 / 3,312 | 192.4 | 203.2 | 0.0 | AC, high power |  |
| 2026-10-26 02:48 | Qwen3.6-35B-A3B | Ollama MLX runner (MLX 4-bit + MTP head, adaptive depth) | 0.40.11 | 3,934 / 4,605 | 174.2 | 165.3 | 0.0 | AC, high power |  |
| 2026-10-26 02:50 | Qwen3.6-35B-A3B | Ollama GGUF + MTP (shipped depth 2) | 0.40.11 | 2,609 / 2,585 | 149.2 | 156.8 | 0.0 | AC, high power |  |
| 2026-10-26 02:51 | Qwen3.6-35B-A3B | MTPLX (sustained profile, depth 1) | mtplx 2.12.2 mlx_lm 0.31.3 mlx 0.32.2 | 4,249 / 4,358 | 131.7 | 120.0 | 0.0 | AC, high power |  |
| 2026-10-26 02:53 | Qwen3.6-35B-A3B | mlx-vlm server + DFlash2 | mlx-vlm 0.7.10, mlx 0.32.4 | 3,942 / 4,620 | 137.9 | 153.2 | 0.1 | AC, high power |  |
| 2026-10-26 02:55 | Qwen3.6-35B-A3B | mlx-lm | mlx_lm 0.32.2 mlx 0.32.4 | 3,835 / 4,366 | 122.0 | 127.8 | 0.0 | AC, high power |  |
| 2026-10-26 02:56 | Qwen3.6-35B-A3B | LM Studio (MLX engine) | llmster 0.0.25-1, MLX runtime 1.13.1 (mlx-engine ab52366, mlx 0.32.3, mlx-lm 0.31.4.dev132, mlx-vlm 0.7.3) | 4,075 / 4,594 | 139.6 | 143.8 | 0.0 | AC, high power | 511 tokens |
| 2026-10-26 02:58 | Qwen3.6-35B-A3B | llama.cpp + MTP (n-max 1) | 0.7.0-dev (b11871, b3fa69955) | 2,712 / 2,900 | 148.5 | 150.2 | 0.0 | AC, high power |  |
| 2026-10-26 02:59 | Qwen3.6-35B-A3B | llama.cpp + MTP (n-max 2) | 0.7.0-dev (b11871, b3fa69955) | 2,702 / 2,876 | 150.1 | 157.2 | 0.0 | AC, high power |  |
| 2026-10-26 03:00 | Qwen3.6-35B-A3B | llama.cpp + MTP (n-max 3) | 0.7.0-dev (b11871, b3fa69955) | 2,722 / 2,894 | 162.3 | 158.8 | 0.3 | AC, high power |  |
| 2026-10-26 03:18 | Flint 40B-A2.5B | Kapitsa | ed2b07d | 5,758 / 4,891 | 263.7 | 264.7 | 6.5 | AC, high power | contention, empty follow-up |
| 2026-10-26 03:20 | Flint 40B-A2.5B | llama.cpp b11804 | 0.7.0-dev (b11871, b3fa69955) | 3,898 / 3,090 | 127.8 | 129.6 | 1.4 | AC, high power |  |
| 2026-10-26 03:24 | Flint 40B-A2.5B | rec 0.1.0 | 0.1.0 | 2,603 / 2,615 | 214.5 | 225.5 | 0.0 | AC, high power |  |
| 2026-10-26 03:36 | Qwen3.6-35B-A3B | Kapitsa (DFlash2 drafter) | cd20dcc | 5,539 / 4,969 | 275.9 | 366.7 | 3.2 | AC, high power |  |
| 2026-10-26 03:37 | Qwen3.6-35B-A3B | Splash 1.4.0 (DFlash2 drafter) | 1.4.0 | 6,754 / 6,033 | 277.9 | 305.9 | 2.9 | AC, high power |  |
| 2026-10-26 03:38 | Qwen3.6-35B-A3B | Kapitsa (DFlash2 drafter) | cd20dcc | 5,823 / 4,965 | 334.1 | 372.1 | 3.3 | AC, high power |  |
| 2026-10-26 03:39 | Qwen3.6-35B-A3B | Splash 1.4.0 (DFlash2 drafter) | 1.4.0 | 6,219 / 6,031 | 294.8 | 306.7 | 2.8 | AC, high power |  |
| 2026-10-26 03:40 | Qwen3.6-35B-A3B | Kapitsa (DFlash2 drafter) | cd20dcc | 5,389 / 5,299 | 329.4 | 352.2 | 3.3 | AC, high power |  |
| 2026-10-26 03:41 | Qwen3.6-35B-A3B | Splash 1.4.0 (DFlash2 drafter) | 1.4.0 | 6,246 / 6,038 | 288.7 | 294.5 | 3.0 | AC, high power |  |
| 2026-10-26 03:43 | Qwen3.8-27B | Kapitsa (DFlash2 drafter) | cd20dcc | 906 / 867 | 136.7 | 149.3 | 4.5 | AC, high power | contention |
| 2026-10-26 03:45 | Qwen3.8-27B | mlx-serve + DFlash2 (draft tree, block 8) | 26.10.3 | 737 / 679 | 118.9 | 133.8 | 5.5 | AC, high power | contention |
| 2026-10-26 03:47 | Qwen3.8-27B | Kapitsa (DFlash2 drafter) | cd20dcc | 915 / 900 | 129.7 | 149.6 | 4.5 | AC, high power | contention |
| 2026-10-26 03:49 | Qwen3.8-27B | mlx-serve + DFlash2 (draft tree, block 8) | 26.10.3 | 813 / 750 | 132.7 | 143.4 | 5.0 | AC, high power | contention |
| 2026-10-26 03:51 | Qwen3.8-27B | Kapitsa (DFlash2 drafter) | cd20dcc | 919 / 912 | 143.4 | 149.4 | 4.5 | AC, high power | contention |
| 2026-10-26 03:52 | Qwen3.8-27B | mlx-serve + DFlash2 (draft tree, block 8) | 26.10.3 | 811 / 740 | 125.5 | 143.8 | 5.5 | AC, high power | contention |
| 2026-10-26 03:58 | Flint 40B-A2.5B | Kapitsa | ed2b07d | 5,691 / 4,906 | 258.8 | 258.7 | 4.5 | AC, high power | empty follow-up |
| 2026-10-26 04:04 | Flint 40B-A2.5B | llama.cpp b11804 | 0.7.0-dev (b11871, b3fa69955) | 3,969 / 3,157 | 128.0 | 127.8 | 5.0 | AC, high power | contention, empty follow-up |
| 2026-10-26 04:11 | Flint 40B-A2.5B | rec 0.1.0 | 0.1.0 | 4,126 / 3,788 | 243.6 | 259.0 | 7.1 | AC, high power | contention |
| 2026-10-26 04:18 | Qwen3.6-35B-A3B | Kapitsa (DFlash2 drafter) | cd20dcc | 6,063 / 5,260 | 323.4 | 369.2 | 9.8 | AC, high power | contention |
| 2026-10-26 04:23 | Qwen3.6-35B-A3B | Splash 1.4.0 (DFlash2 drafter) | 1.4.0 | 6,774 / 6,025 | 294.8 | 312.5 | 8.0 | AC, high power | contention |
| 2026-10-26 04:29 | Qwen3.6-35B-A3B | Kapitsa (DFlash2 drafter) | cd20dcc | 6,070 / 5,269 | 352.5 | 371.0 | 9.0 | AC, high power | contention |
| 2026-10-26 04:35 | Qwen3.6-35B-A3B | Splash 1.4.0 (DFlash2 drafter) | 1.4.0 | 6,783 / 6,028 | 292.0 | 300.0 | 7.3 | AC, high power | contention |
| 2026-10-26 04:41 | Qwen3.6-35B-A3B | Kapitsa (DFlash2 drafter) | cd20dcc | 6,022 / 5,282 | 346.6 | 373.4 | 9.0 | AC, high power | contention |
| 2026-10-26 04:46 | Qwen3.6-35B-A3B | Splash 1.4.0 (DFlash2 drafter) | 1.4.0 | 6,752 / 6,023 | 280.3 | 305.9 | 7.4 | AC, high power | contention |
| 2026-10-26 05:04 | Qwen3.8-27B | Kapitsa (DFlash2 drafter) | cd20dcc | 796 / 797 | 153.7 | 145.9 | 8.2 | AC, high power | contention |
| 2026-10-26 05:10 | Qwen3.8-27B | mlx-serve + DFlash2 (draft tree, block 8) | 26.10.3 | 725 / 668 | 136.8 | 145.3 | 7.6 | AC, high power | contention |
| 2026-10-26 05:17 | Qwen3.8-27B | Kapitsa (DFlash2 drafter) | cd20dcc | 797 / 815 | 128.3 | 143.2 | 6.6 | AC, high power | contention |
| 2026-10-26 05:24 | Qwen3.8-27B | mlx-serve + DFlash2 (draft tree, block 8) | 26.10.3 | 730 / 668 | 133.2 | 144.7 | 8.2 | AC, high power | contention |
| 2026-10-26 05:31 | Qwen3.8-27B | Kapitsa (DFlash2 drafter) | cd20dcc | 804 / 812 | 142.0 | 149.8 | 7.0 | AC, high power | contention |
| 2026-10-26 05:37 | Qwen3.8-27B | mlx-serve + DFlash2 (draft tree, block 8) | 26.10.3 | 730 / 667 | 133.1 | 144.3 | 8.4 | AC, high power | contention |
| 2026-10-26 05:47 | Flint 40B-A2.5B | Kapitsa | ed2b07d | 5,721 / 5,049 | 267.0 | 268.2 | 0.0 | AC, high power |  |
| 2026-10-26 05:48 | Flint 40B-A2.5B | llama.cpp b11804 | 0.7.0-dev (b11871, b3fa69955) | 3,827 / 2,574 | 129.6 | 129.0 | 0.0 | AC, high power |  |
| 2026-10-26 05:50 | Flint 40B-A2.5B | rec 0.1.0 | 0.1.0 | 2,863 / 2,881 | 224.4 | 237.2 | 0.2 | AC, high power | empty follow-up |
| 2026-10-26 05:51 | Qwen3.6-35B-A3B | Kapitsa (DFlash2 drafter) | cd20dcc | 4,535 / 5,059 | 296.5 | 314.8 | 0.0 | AC, high power |  |
| 2026-10-26 05:52 | Qwen3.6-35B-A3B | Splash 1.4.0 (DFlash2 drafter) | 1.4.0 | 6,024 / 5,434 | 284.6 | 299.6 | 0.2 | AC, high power |  |
| 2026-10-26 05:54 | Qwen3.6-35B-A3B | Kapitsa (DFlash2 drafter) | cd20dcc | 5,167 / 5,284 | 320.2 | 351.5 | 0.0 | AC, high power |  |
| 2026-10-26 05:55 | Qwen3.6-35B-A3B | Splash 1.4.0 (DFlash2 drafter) | 1.4.0 | 5,650 / 6,025 | 260.9 | 269.6 | 0.2 | AC, high power |  |
| 2026-10-26 05:56 | Qwen3.6-35B-A3B | Kapitsa (DFlash2 drafter) | cd20dcc | 4,976 / 5,258 | 289.1 | 309.0 | 0.7 | AC, high power |  |
| 2026-10-26 05:57 | Qwen3.6-35B-A3B | Splash 1.4.0 (DFlash2 drafter) | 1.4.0 | 5,677 / 6,007 | 269.1 | 265.4 | 0.0 | AC, high power |  |
| 2026-10-26 05:58 | Qwen3.8-27B | Kapitsa (DFlash2 drafter) | cd20dcc | 832 / 824 | 127.5 | 134.2 | 0.2 | AC, high power |  |
| 2026-10-26 06:00 | Qwen3.8-27B | mlx-serve + DFlash2 (draft tree, block 8) | 26.10.3 | 740 / 681 | 119.4 | 133.8 | 0.2 | AC, high power |  |
| 2026-10-26 06:02 | Qwen3.8-27B | Kapitsa (DFlash2 drafter) | cd20dcc | 838 / 837 | 121.3 | 134.2 | 0.2 | AC, high power |  |
| 2026-10-26 06:04 | Qwen3.8-27B | mlx-serve + DFlash2 (draft tree, block 8) | 26.10.3 | 764 / 689 | 131.3 | 144.3 | 0.0 | AC, high power |  |
| 2026-10-26 06:07 | Qwen3.8-27B | Kapitsa (DFlash2 drafter) | cd20dcc | 844 / 846 | 123.2 | 137.2 | 0.0 | AC, high power |  |
| 2026-10-26 06:09 | Qwen3.8-27B | mlx-serve + DFlash2 (draft tree, block 8) | 26.10.3 | 763 / 710 | 124.5 | 138.0 | 0.1 | AC, high power |  |

Flags: *contention*: more than 5% foreign GPU load sustained in a phase (here the desktop app); *loop*: a decode answer
fell into a repetition loop; *511 tokens*: LM Studio's count; *no cache reuse*: the follow-up re-read the conversation;
*empty follow-up*: a cached follow-up turn returned no content and the harness stopped that series. Raw records in
`results/`, server logs in `logs/`; local paths and the host name are replaced by placeholders.
