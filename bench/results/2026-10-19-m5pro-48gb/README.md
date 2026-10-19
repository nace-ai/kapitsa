# Results, 2026-10-19: MacBook Pro M5 Pro, 48 GB

The result files behind [the M5 Pro section](../../../docs/benchmarks.md#macbook-pro-m5-pro-48-gb-2026-10-19) of
docs/benchmarks.md, written by [bench/harness](../../harness/README.md) on a MacBook Pro (16-inch, 2026), `Mac17,8`:
M5 Pro with an 18-core CPU (6 super, 12 performance) and a 20-core GPU, 48 GB, macOS 26.5.2 (25F84), on AC power in
High Power mode. Times are local (UTC+5).

Every engine ran the harness's full protocol (5 runs of the coding answer at T = 0.6 and 5 greedy, 4K and 16K cold
prompts with three cached follow-ups each, tool calls, peak memory; `--skip agent`), one config at a time under the
GPU lock, 10:09-15:34. Kapitsa is main at 68e964a (0.1.36 and one README commit). The disk held one model family at a
time, so the runs go model by model; the session closed with a control round on an idle machine (14:36-15:00): Kapitsa
and the fastest other engine on each model again; Rapid-MLX's 27B and 35B runs came after it (15:27-15:34), once its
27B pack had downloaded. The README's cells (marked **(cell)**) are each engine's fastest setting, from its newest
run.


## Qwen3.5-4B at 4-bit

| Result file | Started | Run | Decode, T = 0.6 / greedy (tok/s) | Flags and invalid phases |
| --- | --- | --- | ---: | --- |
| `q35-4b-kapitsa-20261019-101245.json` | 10:13 | Kapitsa (DFlash drafter) | 189.3 / 191.93 | — |
| `q35-4b-mtplx-20261019-105752.json` **(cell)** | 10:58 | MTPLX (sustained profile, depth 1) | 113.14 / 117.85 | decode: repetition loop (distinct-4-gram 0.4665) |
| `q35-4b-mlxlm-20261019-110807.json` | 11:08 | mlx-lm | 81.2 / 84.74 | decode: repetition loop (distinct-4-gram 0.3695); decode: repetition loop (distinct-4-gram 0.4857) |
| `q35-4b-mlxserve-mtp-20261019-113254.json` **(cell)** | 11:33 | mlx-serve + MTP | 162.6 / 166.27 | — |
| `q35-4b-mlxserve-dflash-20261019-113429.json` | 11:35 | mlx-serve + DFlash (z-lab drafter) | 142.71 / 149.04 | decode: repetition loop (distinct-4-gram 0.3888) |
| `q35-4b-lmstudio-20261019-114903.json` **(cell)** | 11:49 | LM Studio (MLX engine) | 88.65 / 92.02 | decode: completion_tokens=511 != 512; decode: repetition loop (distinct-4-gram 0.4624); decode: repetition loop (distinct-4-gram 0.4729); decode_greedy: completion_tokens=511 != 512 |
| `q35-4b-mlxvlm-20261019-121338.json` | 12:14 | mlx-vlm server (no speculation) | 88.04 / 92.54 | decode: repetition loop (distinct-4-gram 0.3893) |
| `q35-4b-rapidmlx-20261019-122440.json` | 12:25 | Rapid-MLX (no speculation) | 88.62 / 92.25 | decode: repetition loop (distinct-4-gram 0.4543); prefill_16000: 15468 pages swapped out during the phase |
| `q35-4b-rapidmlx-mtp-20261019-122730.json` **(cell)** | 12:28 | Rapid-MLX + MTP | 112.4 / 120.46 | decode: repetition loop (distinct-4-gram 0.446) |
| `q35-4b-ollama-mtp-20261019-125918.json` | 12:59 | Ollama GGUF + MTP (shipped depth 1) | 86.85 / 90.72 | — |
| `q35-4b-ollama-nomtp-20261019-130119.json` | 13:01 | Ollama GGUF (MTP off) | 73.87 / 77.07 | — |
| `q35-4b-ollama-mlx-q4-20261019-130332.json` **(cell)** | 13:04 | Ollama MLX runner (MLX 4-bit + MTP head, adaptive depth) | 133.98 / 140.49 | decode: repetition loop (distinct-4-gram 0.4652); decode: repetition loop (distinct-4-gram 0.4786) |
| `q35-4b-llamacpp-20261019-133236.json` | 13:33 | llama.cpp | 77.48 / 77.16 | — |
| `q35-4b-llamacpp-mtp2-20261019-133444.json` | 13:35 | llama.cpp + MTP (n-max 2) | 90.37 / 100.16 | — |
| `q35-4b-llamacpp-mtp3-20261019-133638.json` **(cell)** | 13:37 | llama.cpp + MTP (n-max 3) | 93.04 / 103.37 | — |
| `q35-4b-omlx-best-20261019-135853.json` **(cell)** | 14:01 | oMLX + MTP (Lightning MTP, MTPLX's head) | 113.4 / 114.1 | — |
| `q35-4b-mlxvlm-mtp-20261019-141826.json` **(cell)** | 14:19 | mlx-vlm server + MTP | 100.79 / 115.81 | — |
| `q35-4b-kapitsa-20261019-144159.json` **(cell)** | 14:42 | Kapitsa (DFlash drafter) | 178.81 / 201.76 | prefill_16000: 23955 pages swapped out during the phase |

## Qwen3.8-27B at 4-bit

| Result file | Started | Run | Decode, T = 0.6 / greedy (tok/s) | Flags and invalid phases |
| --- | --- | --- | ---: | --- |
| `q38-27b-kapitsa-20261019-100723.json` | 10:09 | Kapitsa (DFlash2 drafter) | 55.59 / 57.14 | CONTENTION: sustained foreign GPU load > 5% in ['tools'] (mean 2.22%, top ['Slack Helper']) |
| `q38-27b-mtplx-20261019-110005.json` | 11:01 | MTPLX (turbo profile, depth 3) | 40.63 / 43.17 | INVALID prefill_16000: usage.prompt_tokens missing; INVALID cached_16000: turn 1 returned no content; not substituting text, cached test skipped |
| `q38-27b-mlxlm-20261019-111015.json` | 11:10 | mlx-lm | 16.33 / 16.92 | — |
| `q38-27b-mlxserve-mtp-20261019-113617.json` | 11:37 | mlx-serve + MTP (depth up to 4) | 48.6 / 49.9 | decode: 4136 pages swapped out during the phase; decode_greedy: 3296 pages swapped out during the phase; prefill_4000: 10700 pages swapped out during the phase; prefill_16000: 2360 pages swapped out during the phase |
| `q38-27b-mlxserve-dflash2-20261019-114040.json` | 11:41 | mlx-serve + DFlash2 (draft tree, block 8) | 79.13 / 89.94 | decode: 2996 pages swapped out during the phase; decode_greedy: 1384 pages swapped out during the phase; prefill_4000: 1636 pages swapped out during the phase; prefill_16000: 1252 pages swapped out during the phase; CONTENTION: sustained foreign GPU load > 5% in ['prefill_4000', 'prefill_16000'] (mean 1.92%, top ['Slack Helper']) |
| `q38-27b-lmstudio-20261019-115135.json` **(cell)** | 11:52 | LM Studio (MLX engine) | 16.81 / 16.88 | decode: completion_tokens=511 != 512; decode_greedy: completion_tokens=511 != 512 |
| `q38-27b-omlx-best-20261019-120435.json` **(cell)** | 12:05 | oMLX + DFlash2 (drafter 4-bit, block 5, verify dflash) | 50.61 / 52.13 | — |
| `q38-27b-mlxvlm-20261019-121545.json` | 12:16 | mlx-vlm server (no speculation) | 16.92 / 16.78 | prefill_4000: 1784 pages swapped out during the phase; prefill_16000: 10988 pages swapped out during the phase |
| `q38-27b-ollama-mtp-20261019-130515.json` | 13:05 | Ollama GGUF + MTP (shipped depth 4) | 29.5 / 33.8 | — |
| `q38-27b-ollama-nomtp-20261019-131035.json` | 13:11 | Ollama GGUF (MTP off) | 15.32 / 15.2 | decode: 22558 pages swapped out during the phase; CONTENTION: sustained foreign GPU load > 5% in ['prefill_16000'] (mean 0.49%, top ['Google Chrome He']) |
| `q38-27b-ollama-mlx-q4-20261019-131924.json` **(cell)** | 13:20 | Ollama MLX runner (MLX 4-bit + MTP head, adaptive depth) | 40.04 / 41.19 | — |
| `q38-27b-llamacpp-20261019-133830.json` | 13:39 | llama.cpp | 15.44 / 15.44 | — |
| `q38-27b-llamacpp-mtp2-20261019-134723.json` | 13:48 | llama.cpp + MTP (n-max 2) | 26.56 / 27.93 | decode: 5593 pages swapped out during the phase; CONTENTION: sustained foreign GPU load > 5% in ['prefill_16000'] (mean 0.54%, top ['Slack Helper']) |
| `q38-27b-llamacpp-mtp3-20261019-135322.json` **(cell)** | 13:53 | llama.cpp + MTP (n-max 3) | 29.57 / 30.73 | CONTENTION: sustained foreign GPU load > 5% in ['decode_greedy', 'prefill_4000'] (mean 2.26%, top ['Slack Helper']) |
| `q38-27b-mlxvlm-mtp-20261019-142033.json` **(cell)** | 14:21 | mlx-vlm server + MTP | 31.97 / 34.22 | CONTENTION: sustained foreign GPU load > 5% in ['prefill_16000'] (mean 1.88%, top ['Slack Helper']) |
| `q38-27b-mtplx-20261019-142626.json` **(cell)** | 14:32 | MTPLX (turbo profile, depth 3) | 40.53 / 44.06 | — |
| `q38-27b-kapitsa-20261019-143820.json` **(cell)** | 14:38 | Kapitsa (DFlash2 drafter) | 60.66 / 64.79 | — |
| `q38-27b-mlxserve-dflash2-20261019-144659.json` **(cell)** | 14:47 | mlx-serve + DFlash2 (draft tree, block 8) | 71.22 / 85.58 | — |
| `q38-27b-rapidmlx-mtp-20261019-152934.json` **(cell)** | 15:30 | Rapid-MLX + MTP (its own 27B pack, MTP sidecar on by default) | 34.92 / 35.02 | — |

## Qwen3.6-35B-A3B at 4-bit

| Result file | Started | Run | Decode, T = 0.6 / greedy (tok/s) | Flags and invalid phases |
| --- | --- | --- | ---: | --- |
| `q36-35b-mtplx-20261019-105442.json` **(cell)** | 10:55 | MTPLX (sustained profile, depth 1) | 88.56 / 83.75 | — |
| `q36-35b-mlxlm-20261019-110554.json` | 11:06 | mlx-lm | 77.8 / 83.75 | — |
| `q36-35b-mlxserve-mtp-20261019-112917.json` | 11:29 | mlx-serve + MTP (depth up to 2) | 155.47 / 170.04 | — |
| `q36-35b-mlxserve-dflash2-20261019-113059.json` | 11:31 | mlx-serve + DFlash2 (block 4) | 126.85 / 131.16 | — |
| `q36-35b-lmstudio-20261019-114601.json` **(cell)** | 11:47 | LM Studio (MLX engine) | 95.62 / 97.85 | decode: completion_tokens=511 != 512; decode_greedy: completion_tokens=511 != 512 |
| `q36-35b-omlx-best-20261019-120215.json` | 12:03 | oMLX + DFlash2 (drafter 4-bit, block 5, adaptive verify) | 153.98 / 150.42 | — |
| `q36-35b-mlxvlm-dflash2-20261019-120912.json` **(cell)** | 12:10 | mlx-vlm server + DFlash2 | 98.17 / 106.51 | — |
| `q36-35b-mlxvlm-20261019-121133.json` | 12:11 | mlx-vlm server (no speculation) | 94.3 / 97.43 | INVALID cached_16000: follow-up: content contains </think> |
| `q36-35b-ollama-mtp-20261019-125232.json` **(cell)** | 12:53 | Ollama GGUF + MTP (shipped depth 2) | 102.4 / 103.5 | — |
| `q36-35b-ollama-nomtp-20261019-125436.json` | 12:55 | Ollama GGUF (MTP off) | 79.48 / 81.24 | — |
| `q36-35b-ollama-mlx-q4-20261019-125649.json` | 12:57 | Ollama MLX runner (MLX 4-bit + MTP head, adaptive depth) | 94.39 / 89.5 | — |
| `q36-35b-llamacpp-20261019-132646.json` | 13:27 | llama.cpp | 81.96 / 82.19 | — |
| `q36-35b-llamacpp-mtp2-20261019-132856.json` **(cell)** | 13:29 | llama.cpp + MTP (n-max 2) | 107.26 / 108.8 | — |
| `q36-35b-llamacpp-mtp3-20261019-133047.json` | 13:31 | llama.cpp + MTP (n-max 3) | 105.9 / 109.11 | — |
| `q36-35b-kapitsa-20261019-143638.json` **(cell)** | 14:37 | Kapitsa (DFlash2 drafter) | 166.84 / 189.53 | — |
| `q36-35b-mlxserve-mtp-20261019-144322.json` **(cell)** | 14:43 | mlx-serve + MTP (depth up to 2) | 155.71 / 159.65 | — |
| `q36-35b-omlx-best-20261019-144505.json` **(cell)** | 14:45 | oMLX + DFlash2 (drafter 4-bit, block 5, adaptive verify) | 144.44 / 151.17 | — |
| `q36-35b-rapidmlx-20261019-152703.json` **(cell)** | 15:27 | Rapid-MLX (no speculation: its catalog disables it for this MoE) | 75.66 / 77.64 | — |

## Nemotron Dory1.0 4B

| Result file | Started | Run | Decode, T = 0.6 / greedy (tok/s) | Flags and invalid phases |
| --- | --- | --- | ---: | --- |
| `dory-kapitsa-perloop-20261019-102509.json` | 10:25 | Kapitsa (NVFP4, 2 loops, per-loop KV, speculative decoding) | 69.82 / 77.49 | — |
| `dory-kapitsa-20261019-102756.json` | 10:28 | Kapitsa (NVFP4, 2 loops, last-loop KV) | 67.78 / 68.86 | decode: repetition loop (distinct-4-gram 0.4628); decode: repetition loop (distinct-4-gram 0.4791) |
| `dory-llamacpp-q5km-20261019-141030.json` **(cell)** | 14:11 | llama.cpp (NVIDIA fork, Q5_K_M, 2 loops, last-loop KV) | 13.18 / 13.22 | — |
| `dory-mlxlm-4bit-20261019-141441.json` | 14:15 | mlx-lm (PR #1988, 4-bit, 2 loops, per-loop KV) | 54.1 / 55.02 | INVALID prefill_4000: content contains </think> |
| `dory-mlxlm-4bit-20261019-145715.json` **(cell)** | 14:57 | mlx-lm (PR #1988, 4-bit, 2 loops, per-loop KV) | 52.63 / 53.8 | INVALID prefill_4000: content contains </think> |
| `dory-kapitsa-perloop-20261019-150011.json` **(cell)** | 15:00 | Kapitsa (NVFP4, 2 loops, per-loop KV, speculative decoding) | 69.62 / 75.77 | — |

## Nemotron 3.5 Lightning 30B-A3B

| Result file | Started | Run | Decode, T = 0.6 / greedy (tok/s) | Flags and invalid phases |
| --- | --- | --- | ---: | --- |
| `lightning-kapitsa-20261019-103049.json` | 10:31 | Kapitsa (DSpark drafter) | 131.77 / 143.25 | — |
| `lightning-mlxlm-20261019-111812.json` **(cell)** | 11:18 | mlx-lm (MLX 4-bit) | 93.73 / 96.76 | — |
| `lightning-mtplx-20261019-112011.json` **(cell)** | 11:21 | MTPLX (MLX 4-bit with MTP head, depth 1) | 100.57 / 104.91 | — |
| `lightning-llamacpp-mtp1-20261019-112259.json` | 11:23 | llama.cpp + MTP (n-max 1) | 90.24 / 88.48 | — |
| `lightning-llamacpp-mtp2-20261019-112503.json` **(cell)** | 11:25 | llama.cpp + MTP (n-max 2) | 101.09 / 100.41 | — |
| `lightning-llamacpp-mtp3-20261019-112655.json` | 11:27 | llama.cpp + MTP (n-max 3) | 94.15 / 98.89 | decode_greedy: repetition loop (distinct-4-gram 0.4843) |
| `lightning-mlxserve-mtp-20261019-114400.json` **(cell)** | 11:44 | mlx-serve + MTP (MLX 4-bit with MTP head) | 114.51 / 117.15 | decode_greedy: completion_tokens=391 != 512; decode_greedy: finish_reason=stop (expected length); decode: 3560 pages swapped out during the phase; decode_greedy: 4572 pages swapped out during the phase; prefill_4000: 4124 pages swapped out during the phase; prefill_16000: 4376 pages swapped out during the phase |
| `lightning-omlx-20261019-124019.json` | 12:40 | oMLX (MLX 4-bit, no speculation) | 101.77 / 104.59 | — |
| `lightning-omlx-mtp-20261019-124212.json` | 12:45 | oMLX + MTP (Lightning MTP, depth 2, MLX 4-bit with MTP head) | 121.09 / 126.32 | — |
| `lightning-mlxvlm-dspark-20261019-124707.json` | 12:47 | mlx-vlm server + DSpark drafter (block 6) | 87.05 / 88.66 | — |
| `lightning-mlxvlm-20261019-124943.json` | 12:50 | mlx-vlm server (no speculation) | 62.52 / 60.37 | — |
| `lightning-ollama-mlx-20261019-132413.json` **(cell)** | 13:25 | Ollama MLX runner (NVIDIA NVFP4 tag + MTP head, adaptive depth) | 91.54 / 88.78 | — |
| `lightning-llamacpp-dspark-20261019-140237.json` | 14:03 | llama.cpp + DSpark drafter (n-max 3) | 64.44 / 64.94 | — |
| `lightning-llamacpp-20261019-140504.json` | 14:05 | llama.cpp (Q4_0 GGUF) | 68.88 / 68.93 | decode_greedy: repetition loop (distinct-4-gram 0.4759) |
| `lightning-lmstudio-20261019-140720.json` **(cell)** | 14:08 | LM Studio (llama.cpp engine, Q4_K_M GGUF) | 72.35 / 72.23 | — |
| `lightning-kapitsa-20261019-145019.json` **(cell)** | 14:50 | Kapitsa (DSpark drafter) | 135.82 / 132.03 | — |
| `lightning-omlx-mtp-20261019-145210.json` **(cell)** | 14:52 | oMLX + MTP (Lightning MTP, depth 2, MLX 4-bit with MTP head) | 123.64 / 124.43 | — |
| `lightning-mlxvlm-mtp-20261019-145424.json` | 14:55 | mlx-vlm server (git main) + native MTP (block 4) | 77.87 / 81.1 | prefill_16000: 105636 pages swapped out during the phase; INVALID prefill_16000: usage.prompt_tokens missing; INVALID cached_16000: turn 1 returned no content; not substituting text, cached test skipped |

## Nemotron 3.5 Bolt 74B-A9B

| Result file | Started | Run | Decode, T = 0.6 / greedy (tok/s) | Flags and invalid phases |
| --- | --- | --- | ---: | --- |
| `bolt-kapitsa-20261019-103248.json` **(cell)** | 10:33 | Kapitsa (LiLiCorr drafter, GPU + Neural Engine prefill) | 79.4 / 75.79 | — |
| `bolt-omlx-20261019-122935.json` **(cell)** | 12:30 | oMLX (the bundle's model file, no drafter) | 39.04 / 39.83 | — |
| `bolt-mlxvlm-20261019-123407.json` **(cell)** | 12:34 | mlx-vlm server (the bundle's model file, no drafter) | 40.8 / 40.39 | cached_4000: follow-up reused only 0 of 4182 prefix tokens; cached_4000: follow-up reused only 0 of 4244 prefix tokens; cached_4000: follow-up reused only 0 of 4287 prefix tokens; cached_16000: follow-up reused only 0 of 16514 prefix tokens; cached_16000: follow-up reused only 0 of 16580 prefix tokens; cached_16000: follow-up reused only 0 of 16623 prefix tokens |
UNUSED [] MISSINGCELLS []
