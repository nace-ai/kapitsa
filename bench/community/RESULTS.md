<!-- Written by tools/community_table.py from bench/community/results/*.json: do not edit by hand. -->

# Community results

Kapitsa measured on people's own Macs with `kapitsa bench`: the protocol of the published benchmarks ([bench/harness](../harness/README.md)), as a client of the local server. Add yours with `kapitsa bench --submit`; [README.md](README.md) says how runs are checked.

Writing: tokens per second of a 512-token coding answer, median of the runs (5, or 3 in a quick run), at temperature 0.6 (Flint at 1.0, its model card's) and greedy. Reading: a cold prompt of about 4K and 16K tokens, prompt tokens per second to its first token. Follow-up: the time to the first token of a new message in that conversation. Memory: the server's peak, GB of 2^30 bytes (a quick run reads no 16K prompt, so its peak is lower: about 0.4 GB on the 4B). ⚠: other apps used the GPU during that measurement (contention flagged), so it may be low. Different Macs, macOS versions and Kapitsa versions give different numbers; these are what each run measured.

| Mac | Runs |
| --- | ---: |
| [Mac Studio (2026) · M5 Max 32-core GPU · 36 GB](#mac-studio-2026--m5-max-32-core-gpu--36-gb) | 5 |
| [Mac Studio (2026) · M5 Ultra 80-core GPU · 256 GB](#mac-studio-2026--m5-ultra-80-core-gpu--256-gb) | 8 |
| [MacBook Air (13-inch, 2026) · M5 10-core GPU · 24 GB](#macbook-air-13-inch-2026--m5-10-core-gpu--24-gb) | 1 |
| [MacBook Air (15-inch, 2026) · M5 10-core GPU · 32 GB](#macbook-air-15-inch-2026--m5-10-core-gpu--32-gb) | 1 |
| [MacBook Pro (14-inch, 2026) · M5 Max 40-core GPU · 64 GB](#macbook-pro-14-inch-2026--m5-max-40-core-gpu--64-gb) | 1 |
| [MacBook Pro (14-inch, 2026) · M5 Pro 16-core GPU · 48 GB](#macbook-pro-14-inch-2026--m5-pro-16-core-gpu--48-gb) | 3 |
| [MacBook Pro (14-inch, 2026) · M5 Pro 20-core GPU · 64 GB](#macbook-pro-14-inch-2026--m5-pro-20-core-gpu--64-gb) | 1 |
| [MacBook Pro (16-inch, 2026) · M5 Pro 20-core GPU · 48 GB](#macbook-pro-16-inch-2026--m5-pro-20-core-gpu--48-gb) | 1 |
| [Mac Studio (2025) · M3 Ultra 80-core GPU · 512 GB](#mac-studio-2025--m3-ultra-80-core-gpu--512-gb) | 4 |
| [Mac Studio (2025) · M4 Max 40-core GPU · 128 GB](#mac-studio-2025--m4-max-40-core-gpu--128-gb) | 2 |
| [MacBook Pro (14-inch, 2025) · M5 10-core GPU · 32 GB](#macbook-pro-14-inch-2025--m5-10-core-gpu--32-gb) | 7 |
| [Mac mini (2024) · M4 10-core GPU · 24 GB](#mac-mini-2024--m4-10-core-gpu--24-gb) | 2 |
| [Mac mini (2024) · M4 10-core GPU · 32 GB](#mac-mini-2024--m4-10-core-gpu--32-gb) | 1 |
| [Mac mini (2024) · M4 Pro 16-core GPU · 48 GB](#mac-mini-2024--m4-pro-16-core-gpu--48-gb) | 2 |
| [Mac mini (2024) · M4 Pro 20-core GPU · 64 GB](#mac-mini-2024--m4-pro-20-core-gpu--64-gb) | 4 |
| [MacBook Pro (14-inch, 2024) · M4 10-core GPU · 16 GB](#macbook-pro-14-inch-2024--m4-10-core-gpu--16-gb) | 2 |
| [MacBook Pro (14-inch, 2024) · M4 Max 40-core GPU · 64 GB](#macbook-pro-14-inch-2024--m4-max-40-core-gpu--64-gb) | 2 |
| [MacBook Pro (14-inch, 2024) · M4 Max 40-core GPU · 128 GB](#macbook-pro-14-inch-2024--m4-max-40-core-gpu--128-gb) | 1 |
| [MacBook Pro (14-inch, 2024) · M4 Pro 16-core GPU · 48 GB](#macbook-pro-14-inch-2024--m4-pro-16-core-gpu--48-gb) | 1 |
| [MacBook Pro (16-inch, 2024) · M4 Max 40-core GPU · 48 GB](#macbook-pro-16-inch-2024--m4-max-40-core-gpu--48-gb) | 1 |
| [MacBook Pro (16-inch, 2024) · M4 Max 40-core GPU · 64 GB](#macbook-pro-16-inch-2024--m4-max-40-core-gpu--64-gb) | 1 |
| [MacBook Pro (16-inch, 2024) · M4 Pro 20-core GPU · 48 GB](#macbook-pro-16-inch-2024--m4-pro-20-core-gpu--48-gb) | 1 |
| [Mac Studio (2023) · M2 Max 38-core GPU · 64 GB](#mac-studio-2023--m2-max-38-core-gpu--64-gb) | 1 |
| [MacBook Pro (14-inch, 2023) · M2 Max 30-core GPU · 64 GB](#macbook-pro-14-inch-2023--m2-max-30-core-gpu--64-gb) | 2 |
| [MacBook Pro (14-inch, Nov 2023) · M3 Max 40-core GPU · 64 GB](#macbook-pro-14-inch-nov-2023--m3-max-40-core-gpu--64-gb) | 2 |
| [MacBook Pro (14-inch, Nov 2023) · M3 Pro 14-core GPU · 36 GB](#macbook-pro-14-inch-nov-2023--m3-pro-14-core-gpu--36-gb) | 1 |
| [MacBook Pro (16-inch, 2023) · M2 Pro 19-core GPU · 32 GB](#macbook-pro-16-inch-2023--m2-pro-19-core-gpu--32-gb) | 1 |
| [MacBook Pro (16-inch, Nov 2023) · M3 Max 40-core GPU · 48 GB](#macbook-pro-16-inch-nov-2023--m3-max-40-core-gpu--48-gb) | 3 |
| [MacBook Pro (16-inch, Nov 2023) · M3 Max 30-core GPU · 96 GB](#macbook-pro-16-inch-nov-2023--m3-max-30-core-gpu--96-gb) | 1 |
| [MacBook Pro (16-inch, Nov 2023) · M3 Pro 18-core GPU · 36 GB](#macbook-pro-16-inch-nov-2023--m3-pro-18-core-gpu--36-gb) | 16 |
| [MacBook Pro (13-inch, M2, 2022) · M2 10-core GPU · 16 GB](#macbook-pro-13-inch-m2-2022--m2-10-core-gpu--16-gb) | 5 |
| [MacBook Pro (14-inch, 2021) · M1 Pro 14-core GPU · 16 GB](#macbook-pro-14-inch-2021--m1-pro-14-core-gpu--16-gb) | 7 |
| [MacBook Pro (14-inch, 2021) · M1 Pro 14-core GPU · 32 GB](#macbook-pro-14-inch-2021--m1-pro-14-core-gpu--32-gb) | 2 |
| [MacBook Pro (16-inch, 2021) · M1 Max 32-core GPU · 32 GB](#macbook-pro-16-inch-2021--m1-max-32-core-gpu--32-gb) | 3 |
| [MacBook Pro (16-inch, 2021) · M1 Max 32-core GPU · 64 GB](#macbook-pro-16-inch-2021--m1-max-32-core-gpu--64-gb) | 4 |
| [Mac mini (M1, 2020) · M1 8-core GPU · 16 GB](#mac-mini-m1-2020--m1-8-core-gpu--16-gb) | 1 |

## Mac Studio (2026) · M5 Max 32-core GPU · 36 GB

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m5-max-qwen3.8-27b-4bit-11a65e.json) | 97.5 tok/s | 102.3 tok/s | 518 tok/s | 510 tok/s | 0.16 s | 0.17 s | 19.6 GB | pass | KFlash | 0.2.4 | 27.0.1 | – |

### Qwen3.6 35B-A3B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m5-max-qwen3.6-35b-a3b-4bit-f36c35.json) | 292.1 tok/s | 303.6 tok/s | 2,871 tok/s | 2,328 tok/s | 0.04 s | 0.05 s | 20.9 GB | pass | KFlash | 0.2.4 | 27.0.1 | – |

### Nemotron 3.5 Bolt 74B-A9B · NVIDIA 24 GB

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m5-max-nemotron-3.5-bolt-24gb-1e1ba2.json) | 129.8 tok/s | 144.5 tok/s | 930 tok/s | 962 tok/s | 0.17 s | 0.17 s | 22.9 GB + 1.0 mapped | pass | LiLiCorr | 0.2.4 | 27.0.1 | – |

### Nemotron 3 Nano 4B · FP8

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m5-max-nemotron-3-nano-4b-fp8-d4efda.json) | 94.7 tok/s | 94.2 tok/s | 4,936 tok/s | 3,964 tok/s | 0.06 s | 0.07 s | 5.1 GB + 0.5 mapped | pass | prompt-lookup | 0.2.4 | 27.0.1 | – |

### Nemotron Nano 9B v2 · NVFP4

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m5-max-nemotron-nano-9b-v2-nvfp4-e5c0af.json) | 60.9 tok/s | 60.8 tok/s | 2,230 tok/s | 2,014 tok/s | 0.09 s | 0.10 s | 7.2 GB + 1.1 mapped | pass | plain | 0.2.4 | 27.0.1 | short answer |

## Mac Studio (2026) · M5 Ultra 80-core GPU · 256 GB

### Qwen3.5 4B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m5-ultra-qwen3.5-4b-4bit-966310.json) | 522.1 tok/s | 592.5 tok/s | 8,024 tok/s | 6,796 tok/s | 0.03 s | 0.03 s | 4.8 GB | pass | KFlash | 0.2.15 | 27.0.1 | – |

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-23](results/kapitsa-bench-2026-10-23-m5-ultra-qwen3.8-27b-4bit-791b6b.json) | 239.9 tok/s | 257.0 tok/s | 1,744 tok/s | 1,695 tok/s | 0.09 s | 0.09 s | 19.5 GB | pass | KFlash | 0.2.14 | 27.0.1 | – |
| [2026-10-23](results/kapitsa-bench-2026-10-23-m5-ultra-qwen3.8-27b-4bit-8ebe35.json) | 231.5 tok/s | 268.6 tok/s | 1,738 tok/s | 1,696 tok/s | 0.09 s | 0.10 s | 19.5 GB | pass | KFlash | 0.2.12 | 27.0.1 | – |

### Qwen3.6 35B-A3B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m5-ultra-qwen3.6-35b-a3b-4bit-328200.json) | 432.9 tok/s | 476.0 tok/s | 10,459 tok/s | 9,167 tok/s | 0.03 s | 0.03 s | 21.7 GB | pass | KFlash | 0.2.15 | 27.0.1 | – |

### Nemotron 3.5 Bolt 74B-A9B · NVIDIA 24 GB

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m5-ultra-nemotron-3.5-bolt-24gb-24d712.json) | 205.7 tok/s | 215.6 tok/s | 1,446 tok/s | 1,549 tok/s | 0.10 s | 0.11 s | 23.0 GB + 1.0 mapped | pass | LiLiCorr | 0.2.15 | 27.0.1 | – |

### Recursive Flint 40B-A2.5B · 4-bit

| Run | Writing, T = 1.0 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m5-ultra-flint-4bit-e814c0.json) | 288.2 tok/s | 288.4 tok/s | 9,296 tok/s | 9,093 tok/s | 0.08 s | 0.08 s | 22.5 GB | pass | plain | 0.2.15 | 27.0.1 | – |

### Qwen3.5 9B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m5-ultra-qwen3.5-9b-4bit-bc813b.json) | 449.3 tok/s | 425.1 tok/s | 4,841 tok/s | 4,664 tok/s | 0.04 s | 0.04 s | 8.0 GB | pass | KFlash | 0.2.15 | 27.0.1 | – |

### Qwen3.8 Flash-Next 180B-A7B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m5-ultra-qwen3.8-flash-next-4bit-7b9318.json) | 142.5 tok/s | 153.0 tok/s | 2,808 tok/s | 2,644 tok/s | 0.06 s | 0.07 s | 70.8 GB + 5.6 mapped | pass | MTP | 0.2.14 | 27.0.1 | repetition loop |

## MacBook Air (13-inch, 2026) · M5 10-core GPU · 24 GB

### Qwen3.5 9B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m5-qwen3.5-9b-4bit-904e74.json) | 63.7 tok/s | 70.9 tok/s | 534 tok/s | 485 tok/s ⚠ | 0.13 s | 0.15 s ⚠ | 8.0 GB | pass | KFlash | 0.2.6 | 26.6 | contention (prefill_16000), on battery |

## MacBook Air (15-inch, 2026) · M5 10-core GPU · 32 GB

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-23](results/kapitsa-bench-2026-10-23-m5-qwen3.8-27b-4bit-ba2545.json) | 30.2 tok/s | 33.3 tok/s | 189 tok/s | 172 tok/s ⚠ | 0.47 s | 0.57 s ⚠ | 19.6 GB | pass | KFlash | 0.2.11 | 27.0.1 | swap, contention (prefill_16000) |

## MacBook Pro (14-inch, 2026) · M5 Max 40-core GPU · 64 GB

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m5-max-qwen3.8-27b-4bit-2f8a42.json) | 116.5 tok/s | 113.8 tok/s | 424 tok/s | 420 tok/s | 0.14 s | 0.15 s | 19.5 GB | pass | KFlash | 0.2.0 | 27.0 | – |

## MacBook Pro (14-inch, 2026) · M5 Pro 16-core GPU · 48 GB

### Qwen3.8 27B · 6-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-19](results/kapitsa-bench-2026-10-19-m5-pro-qwen3.8-27b-6bit-f05370.json) | 45.0 tok/s | 42.9 tok/s | 333 tok/s | 300 tok/s | 0.26 s | 0.30 s | 26.8 GB | pass | DFlash2 | 0.1.38 | 26.4 | – |

### Qwen3.6 35B-A3B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m5-pro-qwen3.6-35b-a3b-4bit-fb9eed.json) | 196.4 tok/s | 216.2 tok/s | 2,599 tok/s | 2,258 tok/s | 0.06 s | 0.08 s | 21.4 GB | pass | KFlash | 0.2.16 | 26.7 | – |

### Qwen3.6 35B-A3B · 8-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m5-pro-qwen3.6-35b-a3b-8bit-77002f.json) | 101.3 tok/s | 100.2 tok/s | 946 tok/s | 911 tok/s | 0.10 s | 0.11 s | 36.0 GB | pass | MTP | 0.2.16 | 26.7 | swap |

## MacBook Pro (14-inch, 2026) · M5 Pro 20-core GPU · 64 GB

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-25](results/kapitsa-bench-2026-10-25-m5-pro-qwen3.8-27b-4bit-79ff3e.json) | 75.0 tok/s | 90.1 tok/s | 645 tok/s | 635 tok/s | 0.21 s | 0.23 s | 19.6 GB | pass | KFlash | 0.2.17 | 27.0.1 | – |

## MacBook Pro (16-inch, 2026) · M5 Pro 20-core GPU · 48 GB

### Recursive Flint 40B-A2.5B · 4-bit

| Run | Writing, T = 1.0 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m5-pro-flint-4bit-3adeb5.json) | 154.1 tok/s | 153.1 tok/s | 1,390 tok/s | 1,123 tok/s | 0.12 s | 0.14 s | 22.2 GB | pass | plain | 0.2.7 | 26.7.1 | swap |

## Mac Studio (2025) · M3 Ultra 80-core GPU · 512 GB

### Qwen3.5 4B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m3-ultra-qwen3.5-4b-4bit-b07938.json) | 386.0 tok/s | 427.2 tok/s | 2,817 tok/s | 2,677 tok/s | 0.02 s | 0.03 s | 4.8 GB | pass | KFlash | 0.2.15 | 27.0.1 | repetition loop |

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m3-ultra-qwen3.8-27b-4bit-2fec89.json) | 161.1 tok/s | 166.0 tok/s | 464 tok/s | 456 tok/s | 0.11 s | 0.11 s | 19.3 GB | pass | KFlash | 0.2.15 | 27.0.1 | – |

### Recursive Flint 40B-A2.5B · 4-bit

| Run | Writing, T = 1.0 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m3-ultra-flint-4bit-ef70f4.json) | 191.0 tok/s | 192.5 tok/s | 2,924 tok/s | 2,599 tok/s | 0.09 s | 0.11 s | 22.6 GB | pass | plain | 0.2.15 | 27.0.1 | – |

### Qwen3.8 Flash-Next 180B-A7B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m3-ultra-qwen3.8-flash-next-4bit-110a29.json) | 96.8 tok/s | 100.3 tok/s | 1,410 tok/s | 1,342 tok/s | 0.07 s | 0.08 s | 70.8 GB + 5.5 mapped | pass | MTP | 0.2.15 | 27.0.1 | – |

## Mac Studio (2025) · M4 Max 40-core GPU · 128 GB

### Qwen3.6 35B-A3B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-26](results/kapitsa-bench-2026-10-26-m4-max-qwen3.6-35b-a3b-4bit-387063.json) | 269.7 tok/s | 297.1 tok/s | 2,262 tok/s | 1,899 tok/s | 0.04 s | 0.04 s | 21.0 GB | pass | KFlash | 0.2.17 | 15.7.7 | – |

### Qwen3.6 35B-A3B · 8-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-26](results/kapitsa-bench-2026-10-26-m4-max-qwen3.6-35b-a3b-8bit-b9ed27.json) | 184.9 tok/s | 202.4 tok/s | 2,214 tok/s | 1,863 tok/s | 0.04 s | 0.07 s | 37.4 GB | pass | KFlash | 0.2.17 | 15.7.7 | – |

## MacBook Pro (14-inch, 2025) · M5 10-core GPU · 32 GB

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m5-qwen3.8-27b-4bit-0778ef.json) | 33.3 tok/s | 37.4 tok/s | 323 tok/s | 306 tok/s | 0.40 s | 0.47 s | 19.6 GB | pass | KFlash | 0.2.17 | 27.0 | – |

### Qwen3.6 35B-A3B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m5-qwen3.6-35b-a3b-4bit-b1956a.json) | 115.7 tok/s | 116.2 tok/s | 1,414 tok/s | 1,250 tok/s | 0.10 s | 0.12 s | 21.4 GB | pass | KFlash | 0.2.17 | 27.0 | – |

### Nemotron 3.5 Lightning 30B-A3B · nvfp4

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-23](results/kapitsa-bench-2026-10-23-m5-nemotron-3.5-lightning-nvfp4-d6e8d0.json) | 70.8 tok/s | 74.6 tok/s | 1,513 tok/s | 1,236 tok/s | 0.16 s | 0.18 s | 18.9 GB | pass | DSpark | 0.2.10 | 27.0 | – |

### Nemotron 3.5 Bolt 74B-A9B · NVIDIA 24 GB

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m5-nemotron-3.5-bolt-24gb-470aee.json) | 46.7 tok/s | 50.2 tok/s | 361 tok/s | 346 tok/s | 0.37 s | 0.40 s | 23.0 GB | pass | LiLiCorr | 0.2.17 | 27.0 | – |

### Recursive Flint 40B-A2.5B · 4-bit

| Run | Writing, T = 1.0 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-23](results/kapitsa-bench-2026-10-23-m5-flint-4bit-e4dd76.json) | 77.0 tok/s | 78.0 tok/s | 1,364 tok/s | 1,189 tok/s | 0.17 s | 0.18 s | 22.2 GB | pass | plain | 0.2.10 | 27.0 | – |

### Qwen3.5 9B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-23](results/kapitsa-bench-2026-10-23-m5-qwen3.5-9b-4bit-26b16d.json) | 65.3 tok/s | 73.2 tok/s | 689 tok/s | 620 tok/s | 0.12 s | 0.14 s | 8.0 GB | pass | KFlash | 0.2.10 | 27.0 | repetition loop |

### Nemotron Dory1.0 4B · NVFP4

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m5-nemotron-labs-dory1.0-4b-nvfp4-65def6.json) | 35.5 tok/s | 35.9 tok/s | 778 tok/s | 630 tok/s | 0.17 s | 0.24 s | 3.7 GB + 0.6 mapped | pass | plain | 0.2.17 | 27.0 | repetition loop |

## Mac mini (2024) · M4 10-core GPU · 24 GB

### Qwen3.6 35B-A3B · 2-bit experts

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-23](results/kapitsa-bench-2026-10-23-m4-qwen3.6-35b-a3b-16gb-236e27.json) | 46.2 tok/s | 46.2 tok/s | 551 tok/s | 456 tok/s | 0.12 s | 0.13 s | 11.5 GB | pass | plain | 0.2.12 | 15.5 | – |

### Qwen3.5 9B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m4-qwen3.5-9b-4bit-b872ef.json) | 53.1 tok/s | 56.8 tok/s | 231 tok/s | 217 tok/s | 0.13 s | 0.15 s | 8.0 GB | pass | KFlash | 0.2.3 | 15.5 | repetition loop |

## Mac mini (2024) · M4 10-core GPU · 32 GB

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m4-qwen3.8-27b-4bit-dfe350.json) | 29.3 tok/s | 29.0 tok/s | 185 tok/s | 160 tok/s ⚠ | 0.47 s | 0.65 s ⚠ | 19.4 GB | pass | KFlash | 0.2.0 | 27.0.1 | swap, contention (prefill_16000) |

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
| [2026-10-23](results/kapitsa-bench-2026-10-23-m4-pro-qwen3.8-27b-4bit-6ca9d5.json) | 64.5 tok/s | 67.8 tok/s | 397 tok/s | 348 tok/s | 0.25 s | 0.27 s | 19.7 GB | pass | KFlash | 0.2.12 | 27.0.1 | – |
| [2026-10-23](results/kapitsa-bench-2026-10-23-m4-pro-qwen3.8-27b-4bit-333903.json) | 55.0 tok/s | 58.2 tok/s | 358 tok/s | 313 tok/s | 0.25 s | 0.27 s | 19.7 GB | pass | KFlash | 0.2.11 | 27.0.1 | – |
| [2026-10-19](results/kapitsa-bench-2026-10-19-m4-pro-qwen3.8-27b-4bit-807730.json) | 58.2 tok/s | 62.4 tok/s | 396 tok/s | 350 tok/s | 0.25 s | 0.28 s | 19.4 GB | pass | DFlash2 | 0.1.37 | 26.5.2 | – |

### Recursive Flint 40B-A2.5B · 4-bit

| Run | Writing, T = 1.0 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-23](results/kapitsa-bench-2026-10-23-m4-pro-flint-4bit-9137d4.json) | 138.3 tok/s | 125.5 tok/s | 966 tok/s | 820 tok/s | 0.14 s | 0.17 s | 22.3 GB | pass | plain | 0.2.12 | 27.0.1 | – |

## MacBook Pro (14-inch, 2024) · M4 10-core GPU · 16 GB

### Qwen3.5 4B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m4-qwen3.5-4b-4bit-f573f9.json) | 86.2 tok/s | 100.7 tok/s | 956 tok/s | 745 tok/s | 0.08 s | 0.12 s | 4.7 GB | pass | KFlash | 0.2.0 | 27.0.1 | – |

### Qwen3.5 9B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m4-qwen3.5-9b-4bit-0242f4.json) | 54.6 tok/s | 61.7 tok/s | 233 tok/s | 218 tok/s | 0.14 s | 0.17 s | 8.0 GB | pass | KFlash | 0.2.0 | 27.0.1 | swap |

## MacBook Pro (14-inch, 2024) · M4 Max 40-core GPU · 64 GB

### Recursive Flint 40B-A2.5B · 4-bit

| Run | Writing, T = 1.0 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m4-max-flint-4bit-f1a3a3.json) | 212.7 tok/s | 213.8 tok/s | 1,508 tok/s | 1,316 tok/s | 0.12 s | 0.14 s | 22.4 GB | pass | plain | 0.2.12 | 27.0.1 | on battery |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m4-max-flint-4bit-8c07e0.json) | 136.1 tok/s | 147.9 tok/s | 1,023 tok/s ⚠ | 798 tok/s ⚠ | 0.11 s ⚠ | 0.14 s ⚠ | 22.4 GB | pass | plain | 0.2.12 | 27.0.1 | contention (prefill_4000, prefill_16000) |

## MacBook Pro (14-inch, 2024) · M4 Max 40-core GPU · 128 GB

### Qwen3.8 Flash-Next 180B-A7B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m4-max-qwen3.8-flash-next-4bit-66620e.json) | 76.1 tok/s | 73.0 tok/s | 650 tok/s | 641 tok/s | 0.11 s | 0.11 s | 70.5 GB + 5.5 mapped | pass | MTP | 0.2.0 | 26.6.2 | – |

## MacBook Pro (14-inch, 2024) · M4 Pro 16-core GPU · 48 GB

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m4-pro-qwen3.8-27b-4bit-c6b34c.json) | 52.3 tok/s | 51.8 tok/s | 271 tok/s | 222 tok/s | 0.28 s | 0.30 s | 19.6 GB | pass | KFlash | 0.2.2 | 27.0.1 | swap, on battery |

## MacBook Pro (16-inch, 2024) · M4 Max 40-core GPU · 48 GB

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m4-max-qwen3.8-27b-4bit-a695d3.json) | 109.2 tok/s | 107.2 tok/s | 418 tok/s | 394 tok/s | 0.16 s | 0.17 s | 19.5 GB | pass | KFlash | 0.2.0 | 27.0 | – |

## MacBook Pro (16-inch, 2024) · M4 Max 40-core GPU · 64 GB

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-20](results/kapitsa-bench-2026-10-20-m4-max-qwen3.8-27b-4bit-2782fe.json) | 118.2 tok/s | 109.5 tok/s | 464 tok/s | 455 tok/s | 0.16 s | 0.18 s | 19.4 GB | pass | DFlash2 | 0.1.39 | 27.0.1 | – |

## MacBook Pro (16-inch, 2024) · M4 Pro 20-core GPU · 48 GB

### Qwen3.6 35B-A3B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m4-pro-qwen3.6-35b-a3b-4bit-94bd4f.json) | 167.6 tok/s | 179.0 tok/s | 1,165 tok/s | 982 tok/s | 0.06 s | 0.08 s | 20.9 GB | pass | KFlash | 0.2.4 | 27.0.1 | on battery |

## Mac Studio (2023) · M2 Max 38-core GPU · 64 GB

### Qwen3.6 35B-A3B · 8-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m2-max-qwen3.6-35b-a3b-8bit-7fd8b5.json) | 150.4 tok/s | 156.0 tok/s | 1,278 tok/s | 1,034 tok/s | 0.06 s | 0.07 s | 37.0 GB | pass | KFlash | 0.2.0 | 27.0.1 | – |

## MacBook Pro (14-inch, 2023) · M2 Max 30-core GPU · 64 GB

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m2-max-qwen3.8-27b-4bit-6978c7.json) | 41.2 tok/s | 41.5 tok/s | 120 tok/s | 116 tok/s | 0.28 s | 0.32 s | 19.1 GB | pass | KFlash | 0.2.7 | 27.0.1 | – |

### Qwen3.8 27B · 8-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m2-max-qwen3.8-27b-8bit-3412ed.json) | 26.6 tok/s | 28.3 tok/s | 117 tok/s | 112 tok/s | 0.29 s | 0.33 s | 31.7 GB | pass | KFlash | 0.2.7 | 27.0.1 | – |

## MacBook Pro (14-inch, Nov 2023) · M3 Max 40-core GPU · 64 GB

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m3-max-qwen3.8-27b-4bit-1f309c.json) | 86.3 tok/s | 88.3 tok/s | 210 tok/s | 201 tok/s | 0.19 s | 0.21 s | 19.1 GB | pass | KFlash | 0.2.17 | 15.8.1 | – |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m3-max-qwen3.8-27b-4bit-9c0686.json) | 84.9 tok/s | 88.6 tok/s | 209 tok/s | 201 tok/s | 0.17 s | 0.20 s | 19.1 GB | pass | KFlash | 0.2.17 | 15.8.1 | – |

## MacBook Pro (14-inch, Nov 2023) · M3 Pro 14-core GPU · 36 GB

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-25](results/kapitsa-bench-2026-10-25-m3-pro-qwen3.8-27b-4bit-785de1.json) | 37.6 tok/s | 40.2 tok/s | 86 tok/s | 81 tok/s ⚠ | 0.36 s | 0.43 s ⚠ | 19.1 GB | pass ⚠ | KFlash | 0.2.17 | 27.0.1 | swap, contention (prefill_16000, tools) |

## MacBook Pro (16-inch, 2023) · M2 Pro 19-core GPU · 32 GB

### Qwen3.6 35B-A3B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-25](results/kapitsa-bench-2026-10-25-m2-pro-qwen3.6-35b-a3b-4bit-aca090.json) | 112.8 tok/s | 115.3 tok/s | 682 tok/s | 536 tok/s | 0.09 s | 0.12 s | 20.8 GB | pass | KFlash | 0.2.17 | 26.5.1 | – |

## MacBook Pro (16-inch, Nov 2023) · M3 Max 40-core GPU · 48 GB

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m3-max-qwen3.8-27b-4bit-7a5f41.json) | 96.4 tok/s | 96.2 tok/s | 224 tok/s | 203 tok/s | 0.17 s | 0.19 s | 19.1 GB | pass | KFlash | 0.2.0 | 27.0.1 | on battery |
| [2026-10-20](results/kapitsa-bench-2026-10-20-m3-max-qwen3.8-27b-4bit-d7286c.json) | 90.2 tok/s | 90.9 tok/s | 225 tok/s | 232 tok/s | 0.16 s | 0.18 s | 19.1 GB | pass | DFlash2 | 0.1.39 | 26.6.2 | swap |

### Recursive Flint 40B-A2.5B · 4-bit

| Run | Writing, T = 1.0 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-25](results/kapitsa-bench-2026-10-25-m3-max-flint-4bit-3a5e24.json) | 169.3 tok/s | 164.4 tok/s | 1,608 tok/s | 1,313 tok/s | 0.16 s | 0.16 s | 22.4 GB | pass | plain | 0.2.17 | 27.0 | – |

## MacBook Pro (16-inch, Nov 2023) · M3 Max 30-core GPU · 96 GB

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-23](results/kapitsa-bench-2026-10-23-m3-max-qwen3.8-27b-4bit-747600.json) | 70.8 tok/s | 71.2 tok/s | 179 tok/s | 150 tok/s | 0.20 s | 0.22 s | 19.1 GB | pass | KFlash | 0.2.12 | 27.0.1 | – |

## MacBook Pro (16-inch, Nov 2023) · M3 Pro 18-core GPU · 36 GB

### Qwen3.5 4B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m3-pro-qwen3.5-4b-4bit-a57268.json) | 127.0 tok/s | 133.3 tok/s | 702 tok/s | 640 tok/s | 0.07 s | 0.09 s | 4.5 GB | pass | KFlash | 0.2.9 | 14.1 | – |
| [2026-10-19](results/kapitsa-bench-2026-10-19-qwen3.5-4b.json) | 113.4 tok/s | 124.3 tok/s | 718 tok/s | 643 tok/s | 0.06 s | 0.07 s | 4.5 GB | pass | DFlash2 | 0.1.37 | 26.6.2 | repetition loop |

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m3-pro-qwen3.8-27b-4bit-6e2c36.json) | 38.1 tok/s | 41.5 tok/s | 110 tok/s | 106 tok/s | 0.35 s | 0.40 s | 19.1 GB | pass | KFlash | 0.2.9 | 14.1 | – |
| [2026-10-19](results/kapitsa-bench-2026-10-19-qwen3.8-27b.json) | 37.0 tok/s | 39.5 tok/s | 112 tok/s | 106 tok/s | 0.28 s | 0.35 s | 19.1 GB | pass | DFlash2 | 0.1.37 | 26.6.2 | – |

### Qwen3.6 35B-A3B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m3-pro-qwen3.6-35b-a3b-4bit-97b0e8.json) | 109.3 tok/s | 113.7 tok/s | 911 tok/s | 771 tok/s | 0.09 s | 0.10 s | 20.9 GB | pass | KFlash | 0.2.9 | 14.1 | – |
| [2026-10-19](results/kapitsa-bench-2026-10-19-qwen3.6-35b-a3b.json) | 106.0 tok/s | 116.2 tok/s | 907 tok/s | 770 tok/s | 0.07 s | 0.09 s | 20.8 GB | pass | DFlash2 | 0.1.37 | 26.6.2 | – |

### Nemotron 3.5 Lightning 30B-A3B · nvfp4

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m3-pro-nemotron-3.5-lightning-nvfp4-8e1858.json) | 73.5 tok/s | 80.3 tok/s | 789 tok/s | 696 tok/s | 0.19 s | 0.21 s | 18.6 GB | pass | DSpark | 0.2.9 | 14.1 | – |
| [2026-10-19](results/kapitsa-bench-2026-10-19-nemotron-3.5-lightning.json) | 73.2 tok/s | 72.3 tok/s | 746 tok/s | 675 tok/s | 0.17 s | 0.20 s | 18.6 GB + 0.7 mapped | pass | DSpark | 0.1.37 | 26.6.2 | – |

### Nemotron 3.5 Bolt 74B-A9B · NVIDIA 24 GB

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m3-pro-nemotron-3.5-bolt-24gb-6d5f2c.json) | 48.0 tok/s | 50.0 tok/s | 261 tok/s | 250 tok/s | 0.33 s | 0.36 s | 22.4 GB | pass | LiLiCorr | 0.2.9 | 14.1 | – |
| [2026-10-19](results/kapitsa-bench-2026-10-19-nemotron-3.5-bolt.json) | 43.4 tok/s | 48.1 tok/s | 413 tok/s | 407 tok/s | 0.32 s | 0.35 s | 22.8 GB + 1.0 mapped | pass | LiLiCorr | 0.1.37 | 26.6.2 | repetition loop |

### Recursive Flint 40B-A2.5B · 4-bit

| Run | Writing, T = 1.0 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m3-pro-flint-4bit-4a6bd5.json) | 73.2 tok/s | 73.5 tok/s | 799 tok/s | 676 tok/s | 0.15 s | 0.19 s | 22.2 GB | pass | plain | 0.2.9 | 14.1 | – |

### Qwen3.5 9B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m3-pro-qwen3.5-9b-4bit-df2a12.json) | 74.1 tok/s | 84.3 tok/s | 381 tok/s | 366 tok/s | 0.10 s | 0.12 s | 8.0 GB | pass | KFlash | 0.2.9 | 14.1 | – |

### Nemotron 3 Nano 4B · FP8

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m3-pro-nemotron-3-nano-4b-fp8-ccbdb5.json) | 33.4 tok/s | 33.0 tok/s | 766 tok/s | 698 tok/s | 0.11 s | 0.14 s | 5.1 GB | pass | prompt-lookup | 0.2.9 | 14.1 | repetition loop |

### Nemotron Dory1.0 4B · NVFP4

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m3-pro-nemotron-labs-dory1.0-4b-nvfp4-057aaa.json) | 36.4 tok/s | 36.5 tok/s | 450 tok/s | 394 tok/s | 0.16 s | 0.23 s | 3.7 GB | pass | plain | 0.2.9 | 14.1 | – |
| [2026-10-19](results/kapitsa-bench-2026-10-19-nemotron-labs-dory1.0-4b.json) | 36.7 tok/s | 36.7 tok/s | 448 tok/s | 396 tok/s | 0.13 s | 0.20 s | 3.6 GB + 0.6 mapped | pass | plain | 0.1.37 | 26.6.2 | repetition loop |

### Nemotron Nano 9B v2 · NVFP4

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m3-pro-nemotron-nano-9b-v2-nvfp4-170e38.json) | 20.8 tok/s | 21.0 tok/s | 330 tok/s | 316 tok/s | 0.33 s | 0.36 s | 7.2 GB | pass | plain | 0.2.9 | 14.1 | – |

## MacBook Pro (13-inch, M2, 2022) · M2 10-core GPU · 16 GB

### Qwen3.5 4B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-19](results/kapitsa-bench-2026-10-19-m2-qwen3.5-4b-4bit-31545f.json) | 64.5 tok/s ⚠ | 62.7 tok/s ⚠ | 261 tok/s ⚠ | 246 tok/s ⚠ | 0.13 s ⚠ | 0.21 s ⚠ | 4.4 GB | pass | DFlash2 | 0.1.38 | 26.6.2 | swap, contention (decode, decode_greedy, prefill_4000, prefill_16000) |

### Qwen3.5 4B · 8-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m2-qwen3.5-4b-8bit-d82c4f.json) | 48.3 tok/s | 52.4 tok/s | 273 tok/s | 235 tok/s | 0.15 s | 0.19 s | 6.5 GB | pass | KFlash | 0.2.0 | 27.0.1 | repetition loop |

### Qwen3.5 9B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m2-qwen3.5-9b-4bit-3f740d.json) | 44.7 tok/s | 46.3 tok/s | 159 tok/s | 143 tok/s | 0.23 s | 0.25 s | 8.0 GB | pass | KFlash | 0.2.0 | 27.0.1 | swap |

### Nemotron 3 Nano 4B · FP8

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m2-nemotron-3-nano-4b-fp8-8a53b4.json) | 21.7 tok/s | 21.3 tok/s | 355 tok/s | 321 tok/s ⚠ | 0.21 s | 0.37 s ⚠ | 5.0 GB | pass | prompt-lookup | 0.2.0 | 27.0.1 | swap, contention (prefill_16000) |

### Nemotron Dory1.0 4B · NVFP4

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-21](results/kapitsa-bench-2026-10-21-m2-nemotron-labs-dory1.0-4b-nvfp4-b19520.json) | 22.6 tok/s | 23.9 tok/s | 203 tok/s | 171 tok/s | 0.37 s | 0.47 s | 3.5 GB | pass | plain | 0.2.0 | 27.0.1 | – |

## MacBook Pro (14-inch, 2021) · M1 Pro 14-core GPU · 16 GB

### Qwen3.5 4B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m1-pro-qwen3.5-4b-4bit-9ce409.json) | 97.2 tok/s | 100.3 tok/s | 401 tok/s | 354 tok/s | 0.12 s | 0.15 s | 4.4 GB | pass | KFlash | 0.2.6 | 27.0.1 | repetition loop |
| [2026-10-19](results/kapitsa-bench-2026-10-19-m1-pro-qwen3.5-4b-4bit-b8f71e.json) | 92.8 tok/s | 89.9 tok/s | 311 tok/s | 272 tok/s | 0.12 s | 0.14 s | 4.4 GB | pass | DFlash2 | 0.1.36 | 27.0.1 | repetition loop |

### Qwen3.6 35B-A3B · 2-bit experts

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m1-pro-qwen3.6-35b-a3b-16gb-2d5e51.json) | 52.2 tok/s | 53.4 tok/s | 462 tok/s | 405 tok/s | 0.16 s | 0.23 s | 11.2 GB | pass | plain | 0.2.6 | 27.0.1 | swap |

### Qwen3.5 9B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m1-pro-qwen3.5-9b-4bit-486428.json) | 56.8 tok/s | 57.8 tok/s | 217 tok/s | 202 tok/s | 0.18 s | 0.21 s | 8.0 GB | pass | KFlash | 0.2.6 | 27.0.1 | repetition loop, swap |

### Nemotron 3 Nano 4B · FP8

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m1-pro-nemotron-3-nano-4b-fp8-eff933.json) | 37.0 tok/s | 37.5 tok/s | 447 tok/s | 401 tok/s | 0.14 s | 0.19 s | 5.0 GB + 0.8 mapped | pass | prompt-lookup | 0.2.6 | 27.0.1 | – |

### Nemotron Dory1.0 4B · NVFP4

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m1-pro-nemotron-labs-dory1.0-4b-nvfp4-5bb4fd.json) | 39.2 tok/s | 40.2 tok/s | 253 tok/s | 214 tok/s | 0.26 s | 0.42 s | 3.6 GB + 0.6 mapped | pass | plain | 0.2.6 | 27.0.1 | repetition loop |

### Nemotron Nano 9B v2 · NVFP4

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m1-pro-nemotron-nano-9b-v2-nvfp4-4a4028.json) | 24.5 tok/s | 24.8 tok/s | 189 tok/s | 180 tok/s | 0.54 s | 0.57 s | 7.1 GB + 0.6 mapped | pass | plain | 0.2.6 | 27.0.1 | – |

## MacBook Pro (14-inch, 2021) · M1 Pro 14-core GPU · 32 GB

### Qwen3.8 27B · 5-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m1-pro-qwen3.8-27b-5bit-37871b.json) | 18.8 tok/s | 20.5 tok/s | 50 tok/s | 49 tok/s | 0.80 s | 0.80 s | 22.3 GB | pass | KFlash | 0.2.15 | 27.2 | swap |

### Nemotron 3.5 Bolt 74B-A9B · NVIDIA 24 GB

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m1-pro-nemotron-3.5-bolt-24gb-a5fa01.json) | 22.0 tok/s | 22.3 tok/s | 174 tok/s | 189 tok/s | 0.82 s | 0.63 s | 22.5 GB + 0.9 mapped | pass | LiLiCorr | 0.2.15 | 27.2 | swap |

## MacBook Pro (16-inch, 2021) · M1 Max 32-core GPU · 32 GB

### Qwen3.5 4B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m1-max-qwen3.5-4b-4bit-243c62.json) | 166.8 tok/s | 186.0 tok/s | 850 tok/s | 770 tok/s | 0.07 s | 0.08 s | 4.4 GB | pass | KFlash | 0.2.15 | 27.0 | – |

### Recursive Flint 40B-A2.5B · 4-bit

| Run | Writing, T = 1.0 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m1-max-flint-4bit-0e5b8d.json) | 98.2 tok/s | 98.5 tok/s | 792 tok/s | 695 tok/s | 0.19 s | 0.26 s | 22.2 GB | pass | plain | 0.2.16 | 27.0 | cached_4000 |

### Nemotron Dory1.0 4B · NVFP4

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-24](results/kapitsa-bench-2026-10-24-m1-max-nemotron-labs-dory1.0-4b-nvfp4-80a594.json) | 63.2 tok/s | 64.3 tok/s | 565 tok/s | 479 tok/s | 0.14 s | 0.24 s | 3.7 GB + 0.6 mapped | pass | plain | 0.2.16 | 27.0 | – |

## MacBook Pro (16-inch, 2021) · M1 Max 32-core GPU · 64 GB

### Qwen3.8 27B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m1-max-qwen3.8-27b-4bit-2ee77b.json) | 58.3 tok/s | 62.3 tok/s | 120 tok/s | 102 tok/s | 0.28 s | 0.31 s | 19.1 GB | pass | KFlash | 0.2.4 | 27.0.1 | – |

### Qwen3.8 27B · 6-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-25](results/kapitsa-bench-2026-10-25-m1-max-qwen3.8-27b-6bit-a58dbf.json) | 36.0 tok/s ⚠ | 38.4 tok/s ⚠ | 104 tok/s | 104 tok/s ⚠ | 0.36 s | 0.36 s ⚠ | 26.3 GB | pass | KFlash | 0.2.16 | 27.0.1 | contention (decode, decode_greedy, prefill_16000) |
| [2026-10-19](results/kapitsa-bench-2026-10-19-m1-max-qwen3.8-27b-6bit-8cb92e.json) | 38.5 tok/s | 35.3 tok/s | 93 tok/s | 87 tok/s ⚠ | 0.34 s | 0.37 s ⚠ | 26.0 GB | pass | DFlash2 | 0.1.37 | 26.6.2 | contention (prefill_16000) |

### Qwen3.6 35B-A3B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m1-max-qwen3.6-35b-a3b-4bit-6d1e2e.json) | 145.1 tok/s | 163.5 tok/s | 1,012 tok/s | 874 tok/s | 0.07 s | 0.08 s | 20.8 GB | pass | KFlash | 0.2.3 | 27.0.1 | – |

## Mac mini (M1, 2020) · M1 8-core GPU · 16 GB

### Qwen3.5 4B · 4-bit

| Run | Writing, T = 0.6 | Writing, greedy | Reading 4K | Reading 16K | Follow-up 4K | Follow-up 16K | Peak memory | Tool calls | Decoding | Kapitsa | macOS | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- | --- | --- |
| [2026-10-22](results/kapitsa-bench-2026-10-22-m1-qwen3.5-4b-4bit-5b5b8e.json) | 48.9 tok/s | 52.8 tok/s | 230 tok/s | 206 tok/s | 0.20 s | 0.23 s | 4.4 GB | pass | KFlash | 0.2.2 | 26.6.2 | swap |
