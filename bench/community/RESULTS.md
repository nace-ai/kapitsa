<!-- Written by tools/community_table.py from bench/community/results/*.json: do not edit by hand. -->

# Community results

Kapitsa measured on people's own Macs with `kapitsa bench`: the protocol of the published benchmarks ([bench/harness](../harness/README.md)), as a client of the local server. Add yours with `kapitsa bench --submit`; [README.md](README.md) says how runs are checked.

Writing: tokens per second of a 512-token coding answer, median of the runs (5, or 3 in a quick run), at temperature 0.6 and greedy. Reading: a cold prompt of about 4K and 16K tokens, prompt tokens per second to its first token. Follow-up: the time to the first token of a new message in that conversation. Memory: the server's peak, GB of 2^30 bytes (a quick run reads no 16K prompt, so its peak is lower: about 0.4 GB on the 4B). ⚠: other apps used the GPU during that measurement (contention flagged), so it may be low. Different Macs, macOS versions and Kapitsa versions give different numbers; these are what each run measured.

| Mac | Runs |
| --- | ---: |
| [MacBook Pro (14-inch, 2026) · M5 Pro 16-core GPU · 48 GB](#macbook-pro-14-inch-2026--m5-pro-16-core-gpu--48-gb) | 1 |
| [Mac mini (2024) · M4 10-core GPU · 32 GB](#mac-mini-2024--m4-10-core-gpu--32-gb) | 1 |
| [Mac mini (2024) · M4 Pro 16-core GPU · 48 GB](#mac-mini-2024--m4-pro-16-core-gpu--48-gb) | 2 |
| [Mac mini (2024) · M4 Pro 20-core GPU · 64 GB](#mac-mini-2024--m4-pro-20-core-gpu--64-gb) | 1 |
| [MacBook Pro (14-inch, 2024) · M4 10-core GPU · 16 GB](#macbook-pro-14-inch-2024--m4-10-core-gpu--16-gb) | 2 |
| [MacBook Pro (14-inch, 2024) · M4 Max 40-core GPU · 128 GB](#macbook-pro-14-inch-2024--m4-max-40-core-gpu--128-gb) | 1 |
| [MacBook Pro (16-inch, 2024) · M4 Max 40-core GPU · 48 GB](#macbook-pro-16-inch-2024--m4-max-40-core-gpu--48-gb) | 1 |
| [MacBook Pro (16-inch, 2024) · M4 Max 40-core GPU · 64 GB](#macbook-pro-16-inch-2024--m4-max-40-core-gpu--64-gb) | 1 |
| [MacBook Pro (16-inch, Nov 2023) · M3 Max 40-core GPU · 48 GB](#macbook-pro-16-inch-nov-2023--m3-max-40-core-gpu--48-gb) | 2 |
| [MacBook Pro (16-inch, Nov 2023) · M3 Pro 18-core GPU · 36 GB](#macbook-pro-16-inch-nov-2023--m3-pro-18-core-gpu--36-gb) | 6 |
| [MacBook Pro (13-inch, M2, 2022) · M2 10-core GPU · 16 GB](#macbook-pro-13-inch-m2-2022--m2-10-core-gpu--16-gb) | 4 |
| [MacBook Pro (14-inch, 2021) · M1 Pro 14-core GPU · 16 GB](#macbook-pro-14-inch-2021--m1-pro-14-core-gpu--16-gb) | 1 |
| [MacBook Pro (16-inch, 2021) · M1 Max 32-core GPU · 64 GB](#macbook-pro-16-inch-2021--m1-max-32-core-gpu--64-gb) | 1 |

## MacBook Pro (14-inch, 2026) · M5 Pro 16-core GPU · 48 GB

### Qwen3.8 27B · 6-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-19](results/kapitsa-bench-2026-10-19-m5-pro-qwen3.8-27b-6bit-f05370.json) | 45.0 tok/s | 42.9 tok/s | 333 tok/s | 300 tok/s | 0.26 s | 0.30 s | 26.8 GB | pass | DFlash2 | 0.1.38 | 26.4 | – |

## Mac mini (2024) · M4 10-core GPU · 32 GB

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m4-qwen3.8-27b-4bit-dfe350.json) | 29.3 tok/s | 29.0 tok/s | 185 tok/s | 160 tok/s ⚠ | 0.47 s | 0.65 s ⚠ | 19.4 GB | pass | unknown | 0.2.0 | 27.0.1 | swap, contention (prefill_16000) |

## Mac mini (2024) · M4 Pro 16-core GPU · 48 GB

### Qwen3.6 35B-A3B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-19](results/kapitsa-bench-2026-10-19-m4-pro-qwen3.6-35b-a3b-4bit-quick-52e3d9.json) quick | 145.8 tok/s ⚠ | 165.1 tok/s ⚠ | 878 tok/s ⚠ | – | 0.07 s ⚠ | – | 20.5 GB | pass ⚠ | DFlash2 | 0.1.38 | 27.0.1 | swap, contention (decode, decode_greedy, prefill_4000, tools) |

### Nemotron 3.5 Bolt 74B-A9B · NVIDIA 24 GB

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-19](results/kapitsa-bench-2026-10-19-m4-pro-nemotron-3.5-bolt-24gb-quick-de0f3f.json) quick | 56.2 tok/s ⚠ | 54.3 tok/s ⚠ | 333 tok/s ⚠ | – | 0.31 s ⚠ | – | 22.6 GB + 1.0 mapped | pass ⚠ | LiLiCorr | 0.1.38 | 27.0.1 | contention (decode, decode_greedy, prefill_4000, tools) |

## Mac mini (2024) · M4 Pro 20-core GPU · 64 GB

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-19](results/kapitsa-bench-2026-10-19-m4-pro-qwen3.8-27b-4bit-807730.json) | 58.2 tok/s | 62.4 tok/s | 396 tok/s | 350 tok/s | 0.25 s | 0.28 s | 19.4 GB | pass | DFlash2 | 0.1.37 | 26.5.2 | – |

## MacBook Pro (14-inch, 2024) · M4 10-core GPU · 16 GB

### Qwen3.5 4B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m4-qwen3.5-4b-4bit-f573f9.json) | 86.2 tok/s | 100.7 tok/s | 956 tok/s | 745 tok/s | 0.08 s | 0.12 s | 4.7 GB | pass | unknown | 0.2.0 | 27.0.1 | – |

### Qwen3.5 9B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m4-qwen3.5-9b-4bit-0242f4.json) | 54.6 tok/s | 61.7 tok/s | 233 tok/s | 218 tok/s | 0.14 s | 0.17 s | 8.0 GB | pass | unknown | 0.2.0 | 27.0.1 | swap |

## MacBook Pro (14-inch, 2024) · M4 Max 40-core GPU · 128 GB

### Qwen3.8 Flash-Next 180B-A7B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m4-max-qwen3.8-flash-next-4bit-66620e.json) | 76.1 tok/s | 73.0 tok/s | 650 tok/s | 641 tok/s | 0.11 s | 0.11 s | 70.5 GB + 5.5 mapped | pass | MTP | 0.2.0 | 26.6.2 | – |

## MacBook Pro (16-inch, 2024) · M4 Max 40-core GPU · 48 GB

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m4-max-qwen3.8-27b-4bit-a695d3.json) | 109.2 tok/s | 107.2 tok/s | 418 tok/s | 394 tok/s | 0.16 s | 0.17 s | 19.5 GB | pass | unknown | 0.2.0 | 27.0 | – |

## MacBook Pro (16-inch, 2024) · M4 Max 40-core GPU · 64 GB

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-20](results/kapitsa-bench-2026-10-20-m4-max-qwen3.8-27b-4bit-2782fe.json) | 118.2 tok/s | 109.5 tok/s | 464 tok/s | 455 tok/s | 0.16 s | 0.18 s | 19.4 GB | pass | DFlash2 | 0.1.39 | 27.0.1 | – |

## MacBook Pro (16-inch, Nov 2023) · M3 Max 40-core GPU · 48 GB

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m3-max-qwen3.8-27b-4bit-7a5f41.json) | 96.4 tok/s | 96.2 tok/s | 224 tok/s | 203 tok/s | 0.17 s | 0.19 s | 19.1 GB | pass | unknown | 0.2.0 | 27.0.1 | on battery |
| [2026-10-20](results/kapitsa-bench-2026-10-20-m3-max-qwen3.8-27b-4bit-d7286c.json) | 90.2 tok/s | 90.9 tok/s | 225 tok/s | 232 tok/s | 0.16 s | 0.18 s | 19.1 GB | pass | DFlash2 | 0.1.39 | 26.6.2 | swap |

## MacBook Pro (16-inch, Nov 2023) · M3 Pro 18-core GPU · 36 GB

### Qwen3.5 4B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-19](results/kapitsa-bench-2026-10-19-qwen3.5-4b.json) | 113.4 tok/s | 124.3 tok/s | 718 tok/s | 643 tok/s | 0.06 s | 0.07 s | 4.5 GB | pass | DFlash2 | 0.1.37 | 26.6.2 | repetition loop |

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-19](results/kapitsa-bench-2026-10-19-qwen3.8-27b.json) | 37.0 tok/s | 39.5 tok/s | 112 tok/s | 106 tok/s | 0.28 s | 0.35 s | 19.1 GB | pass | DFlash2 | 0.1.37 | 26.6.2 | – |

### Qwen3.6 35B-A3B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-19](results/kapitsa-bench-2026-10-19-qwen3.6-35b-a3b.json) | 106.0 tok/s | 116.2 tok/s | 907 tok/s | 770 tok/s | 0.07 s | 0.09 s | 20.8 GB | pass | DFlash2 | 0.1.37 | 26.6.2 | – |

### Nemotron 3.5 Lightning 30B-A3B · nvfp4

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-19](results/kapitsa-bench-2026-10-19-nemotron-3.5-lightning.json) | 73.2 tok/s | 72.3 tok/s | 746 tok/s | 675 tok/s | 0.17 s | 0.20 s | 18.6 GB + 0.7 mapped | pass | DSpark | 0.1.37 | 26.6.2 | – |

### Nemotron 3.5 Bolt 74B-A9B · NVIDIA 24 GB

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-19](results/kapitsa-bench-2026-10-19-nemotron-3.5-bolt.json) | 43.4 tok/s | 48.1 tok/s | 413 tok/s | 407 tok/s | 0.32 s | 0.35 s | 22.8 GB + 1.0 mapped | pass | LiLiCorr | 0.1.37 | 26.6.2 | repetition loop |

### Nemotron Dory1.0 4B · NVFP4

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-19](results/kapitsa-bench-2026-10-19-nemotron-labs-dory1.0-4b.json) | 36.7 tok/s | 36.7 tok/s | 448 tok/s | 396 tok/s | 0.13 s | 0.20 s | 3.6 GB + 0.6 mapped | pass | plain | 0.1.37 | 26.6.2 | repetition loop |

## MacBook Pro (13-inch, M2, 2022) · M2 10-core GPU · 16 GB

### Qwen3.5 4B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-19](results/kapitsa-bench-2026-10-19-m2-qwen3.5-4b-4bit-31545f.json) | 64.5 tok/s ⚠ | 62.7 tok/s ⚠ | 261 tok/s ⚠ | 246 tok/s ⚠ | 0.13 s ⚠ | 0.21 s ⚠ | 4.4 GB | pass | DFlash2 | 0.1.38 | 26.6.2 | swap, contention (decode, decode_greedy, prefill_4000, prefill_16000) |

### Qwen3.5 9B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m2-qwen3.5-9b-4bit-3f740d.json) | 44.7 tok/s | 46.3 tok/s | 159 tok/s | 143 tok/s | 0.23 s | 0.25 s | 8.0 GB | pass | unknown | 0.2.0 | 27.0.1 | swap |

### Nemotron 3 Nano 4B · FP8

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m2-nemotron-3-nano-4b-fp8-8a53b4.json) | 21.7 tok/s | 21.3 tok/s | 355 tok/s | 321 tok/s ⚠ | 0.21 s | 0.37 s ⚠ | 5.0 GB | pass | unknown | 0.2.0 | 27.0.1 | swap, contention (prefill_16000) |

### Nemotron Dory1.0 4B · NVFP4

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m2-nemotron-labs-dory1.0-4b-nvfp4-b19520.json) | 22.6 tok/s | 23.9 tok/s | 203 tok/s | 171 tok/s | 0.37 s | 0.47 s | 3.5 GB | pass | plain | 0.2.0 | 27.0.1 | – |

## MacBook Pro (14-inch, 2021) · M1 Pro 14-core GPU · 16 GB

### Qwen3.5 4B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-19](results/kapitsa-bench-2026-10-19-m1-pro-qwen3.5-4b-4bit-b8f71e.json) | 92.8 tok/s | 89.9 tok/s | 311 tok/s | 272 tok/s | 0.12 s | 0.14 s | 4.4 GB | pass | DFlash2 | 0.1.36 | 27.0.1 | repetition loop |

## MacBook Pro (16-inch, 2021) · M1 Max 32-core GPU · 64 GB

### Qwen3.8 27B · 6-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-19](results/kapitsa-bench-2026-10-19-m1-max-qwen3.8-27b-6bit-8cb92e.json) | 38.5 tok/s | 35.3 tok/s | 93 tok/s | 87 tok/s ⚠ | 0.34 s | 0.37 s ⚠ | 26.0 GB | pass | DFlash2 | 0.1.37 | 26.6.2 | contention (prefill_16000) |
