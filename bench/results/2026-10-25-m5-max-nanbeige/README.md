# M5 Max, 128 GB: Nanbeige4.3-33B on Kapitsa, mlx-lm, oMLX, mlx-vlm, llama.cpp and Ollama, 2026-10-25

Nanbeige4.3-33B is a dense 33B Llama (52 layers, 56 query / 8 KV heads, 512K context), which Kapitsa runs on Flint's
dense kernels since #605, with the 8-bit/bf16 prefill GEMMs and the 7-head tensor attention of #606. Everything here
ran on one MacBook Pro M5 Max with 128 GB, on macOS 26.4, on AC power, with nothing else on the GPU.

Weights:

- **4-bit**: [kapitsaprime/Nanbeige4.3-33B-MLX-4bit](https://huggingface.co/kapitsaprime/Nanbeige4.3-33B-MLX-4bit)
  @ 5901f0d4 (the catalog's default). Every decoder layer is AWQ 4-bit (`mlx_lm.awq`, affine, group 64), and the
  embedding and LM head are 8-bit. This is the folder that Kapitsa, mlx-lm, oMLX and mlx-vlm all read.
- **8-bit**: [kapitsaprime/Nanbeige4.3-33B-MLX-8bit](https://huggingface.co/kapitsaprime/Nanbeige4.3-33B-MLX-8bit)
  @ b3302e04.
- **llama.cpp and Ollama**: Q4_K_M and Q8_0 GGUFs made from Nanbeige/Nanbeige4.3-33B @ c1093c50 with llama.cpp
  b11871's own tools (`configs/nanbeige.json` describes the two tokenizer metadata fixes).

## Real chat prompts: the harness

Command: `bench/harness/compare_engines.py configs/nanbeige.json --skip agent`. It ran each engine in turn on
Nanbeige's chat template, with thinking off and the harness's sampling (T 0.6, top_p 0.95, top_k 20). The 4K and 16K
prompts are 5,088 and 19,954 of Nanbeige's tokens (the config's `--ref-prompt-tokens`).

| Engine | Decode T=0.6 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached TTFT 4K / 16K (s) | Peak memory (GB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Kapitsa** 5bee92c | **28.8** (28.5-29.0) | **28.7** | 532 / 483 | 0.26 / 0.43 | **25.3** |
| oMLX 0.7.3.dev2 | 28.0 (27.3-28.2) | 28.4 | **766 / 634** | 0.50 / 1.56 | 35.2 |
| mlx-vlm server 0.7.10 | 27.6 (27.6-27.7) | 28.0 | 705 / 589 | 0.66 / 1.52 | 34.6 |
| mlx-lm 0.32.2 | 27.0 (26.9-27.0) | 27.4 | 668 / 573 | 0.40 / 0.65 | 40.6 |
| llama.cpp b11871 | 25.9 (25.8-26.1) | 25.4 | 496 / 394 | **0.25 / 0.33** | 33.3 |
| Ollama 0.40.11 | 25.6 (25.5-25.8) | 24.8 | 503 / 394 | 0.25 / 0.39 | 14.5 + 18.4 mmap |

The harness wrote `results/compare_table.md` and the records beside it. The server logs are in `logs/`. Its flags are
in the full table:

- The decode-loop flags (repetition at T 0.6) on the GGUF engines and mlx-vlm do not change the speed numbers.
- The cached-TTFT flags mark engines whose follow-up turn did not reuse the prefix after a `length` finish.

## Context sweep: 1K to 128K tokens

`context-sweep/sweep.sh` measured each engine's own benchmark tool, interleaved per length:

- Kapitsa: `flint_bench --prompt N --steps 64`, built from the #606 branch.
- mlx-lm 0.32.2: `mlx_lm.benchmark -p N -g 64`.
- llama.cpp b11871: `llama-bench -fa 1 -b 2048 -ub 2048`, with decode measured at depth N.

Before every run the script rested 20 s and then waited until no other process had held more than 5% of the GPU for
15 s (`gpu_hogs.py`). `context-sweep/sweep_table.py DIR` prints the tables from the raw outputs in `4bit/` and `8bit/`.

The first Kapitsa run at 1K was cold (631 tok/s). The figure below is the warm repeat's second rep, from
`4bit-warm/kapitsa-1024.txt`; the other two engines warm up inside their own tools.

**4-bit** (the MLX AWQ pack for Kapitsa and mlx-lm, Q4_K_M for llama.cpp)

| Context | Prefill: Kapitsa | mlx-lm | llama.cpp | Decode: Kapitsa | mlx-lm | llama.cpp |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 730 | **823** | 665 | **28.2** | 28.1 | 26.7 |
| 4K | 678 | **773** | 555 | 27.0 | **27.2** | 25.0 |
| 16K | **511** | 501 | 345 | **23.3** | 21.8 | 17.9 |
| 32K | 387 | **455** | 295 | **20.0** | 18.6 | 16.2 |
| 64K | 324 | **410** | 233 | **16.0** | 15.4 | 12.6 |
| 128K | 242 | **277** | 140 | **11.3** | 11.0 | 8.2 |

**8-bit** (the MLX 8-bit pack for Kapitsa and mlx-lm, Q8_0 for llama.cpp)

| Context | Prefill: Kapitsa | mlx-lm | llama.cpp | Decode: Kapitsa | mlx-lm | llama.cpp |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 665 | **725** | 708 | 15.6 | 15.8 | **16.3** |
| 4K | 652 | **699** | 575 | 15.1 | 15.3 | **15.8** |
| 16K | **510** | 461 | 360 | **13.9** | 13.7 | 12.8 |
| 32K | 397 | **433** | 311 | **12.8** | 12.7 | 12.4 |

What the sweep shows:

- **Decode.** Kapitsa ties mlx-lm at 1K and 4K and leads from 16K to 128K by 3-8%. llama.cpp trails both from 4K on.
- **Prefill.** mlx-lm is ahead at most lengths: 13-14% at 1K and 4K, and 14-27% from 32K up. Kapitsa leads only at
  16K. llama.cpp is the slowest from 4K up.

Single runs swing by up to about 20% as the chip heats, so read gaps under about 10% as noise.
