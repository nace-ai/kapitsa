# Results, 2026-10-22/23: MacBook Pro M3 Pro, 36 GB, the catalog against the other engines, prompt length 1K to 32K

Kapitsa on the ten catalog models that fit a 36 GB Mac, against every other engine that runs on macOS 14 for that
model, then 1K to 32K against the fastest of them. The Mac: a MacBook Pro (16-inch, Nov 2023), `Mac15,7`, M3 Pro with a
12-core CPU and an 18-core GPU, 36 GB, macOS 14.1 (23B2073), on AC power (Automatic power mode, Low Power Mode off).
Times are the Mac's local time (UTC+6).

The same Mac's Flint runs are in [2026-10-22-flint-m3pro-ac](../2026-10-22-flint-m3pro-ac/README.md) (#341), whose
Flint × llama.cpp cell and sweep this run reuses.

## How it ran

- One model and one engine at a time, each leg under `lockf -k /tmp/kapitsa-gpu.lock`, with the display put to sleep
  first (`pmset displaysleepnow`: on AC this Mac's display stays on, and its animated wallpaper decodes video at ~150% CPU,
  which cost 4% of decode in #341's first pass), at least 40% of memory free and 30 GB of disk free, no download
  running. The queue scripts and their log are in [queue/](queue/); the servers' logs in [logs/](logs/).
- **The protocol** (`compare_engines.py --skip agent`, the configs on main plus [m3pro.json](m3pro.json)): five
  512-token answers at the model's temperature (T = 0.6; Flint 1.0, with `--always-reasons`), median; five greedy; a
  cold 4K- and 16K-token prompt, prefill = prompt tokens over the time to the first token; tool calls with thinking on
  and off. The Nemotron-H cells are checked against the Nemotron prompt reference (4184 / 16517 tokens, Bolt 4183 /
  16514), the Qwen ones against 4133 / 16286, Flint's against 3874 / 15066.
- **The sweep** (`--agentic --skip decode,greedy,prefill,tools,agent,cache,sustain`, the harness's `t_sweep`): one cold
  request per size, a prompt cut from the frozen corpus to about 1K, 2K, 4K, 8K, 16K and the most a 32K context holds
  with the answer (~31K tokens), then a 192-token answer; prefill as above, decode over the middle of the answer.
- **The rivals** are the M4 Pro's fastest setup per engine (the harness configs on main), not re-tuned on the M3. Which
  ran fresh and which were reused, each one's version and its full server command: [RIVALS.md](RIVALS.md). The user's
  rule: a rival runs once per Mac, only where this Mac has no number for it; Kapitsa runs fresh on every model.
- **`kapitsa bench`** ran the same protocol on all ten models as community results:
  `bench/community/results/kapitsa-bench-2026-10-22-m3-pro-*.json` (c490f81, every one `invalid: []`), logs in
  [logs/kapitsa-bench/](logs/kapitsa-bench/). They are another client of the same server and read close to the harness
  cells (the 35B-A3B's 4K prefill 911 there against 855 here); the tables below are the harness's only.

### Kapitsa's commit

- The Qwen models and Flint: main **e86dea7** (0.2.9, #343), built on this Mac. Flint's sweep: **c490f81** (below).
- The Nemotron-H models (Dory, Nano 4B, Nano 9B v2, Lightning, Bolt): main **c490f81**, the first main with #351. On
  e86dea7 Kapitsa could not run them on macOS 14: their kernels used device-scope fences (Metal 3.2, macOS 15+) and the
  server stopped at start with a Metal compile error. Those five records (`sw-*-kapitsa-20261022-2323*`, "server never
  became ready") are kept.
- e86dea7..c490f81 does not change what the Qwen models or Flint run on an M3: #351 is the Nemotron-H kernels, #348
  Dory's prefill attention, #344 limits an M5-only default (`qmt_on()`, the M5's matrix units) to the 4B and 27B, #347
  the label `kapitsa bench` gives Flint's temperature; the rest are docs.
- The 4B and 27B first ran on 0.2.8 (daf6e95): `sw-q35-4b-kapitsa-20261022-230611` and
  `sw-q38-27b-kapitsa-20261022-230824`, kept, superseded by the e86dea7 runs above them in the table.

## What it says

- **Decode: Kapitsa is fastest on 7 of the 9 models with a rival**: the 4B 1.79x the next engine, the 9B 2.90x, the 27B
  1.77x, the 35B-A3B 1.85x, Flint 1.44x (llama.cpp, #341), Dory 1.21x, Lightning 1.92x. **It is behind on the Nano 4B
  (0.99x, 34.4 against 34.9 tok/s) and the Nano 9B v2 (0.84x, 20.9 against 24.9)**, where the rivals read other, smaller
  files: Kapitsa runs NVIDIA's published checkpoints, which keep some layers in BF16 or FP8 (the 9B v2's NVFP4 reads
  ~6.11 GB per token against 4.67 GB for mlx-community's 4-bit file; the 4B's FP8 is ~1 byte per parameter against
  Q4_K_M's ~0.6).
- **Prefill at 4K: Kapitsa is ahead on all 8 models with a valid rival number**, 1.04x (Flint) to 1.85x (Nano 4B); the least on
  Flint and the 9B (1.08x).
- **The sweep keeps both orders at every size from 1K to 32K**: Kapitsa reads faster at every size on all seven models
  with a rival prefill number, and writes faster at every size on six of them; on the Nano 4B it writes 0.94-0.99x
  llama.cpp, on the Nano 9B v2 0.85-0.86x mlx-lm.
- **Memory: Kapitsa's peak is lower on every model** (the server's `phys_footprint_peak`), e.g. the 27B 19.3 GB
  against 25.6-27.9 GB, Dory 4.1 against 10.0.

## The protocol

Decode and greedy: median tok/s of five 512-token answers. Each model's rows by decode, Kapitsa first; the first rival
is the one swept.

| Model | Engine | Version | Decode tok/s | Greedy tok/s | Prefill 4K / 16K tok/s | TTFT 4K s | Peak GB | Record |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.5 4B | **Kapitsa** | e86dea7 | 119.9 | 131.3 | 703 / 638 | 5.88 | 4.9 | [sw-q35-4b-kapitsa-20261022-232718](results/sw-q35-4b-kapitsa-20261022-232718.json) |
|  | Ollama MLX runner (MLX 4-bit + MTP head, adaptive depth) | 0.40.8 | 66.9 | 71.1 | 600 / 540 | 6.89 | 12.1 | [q35-4b-ollama-mlx-q4-20261023-000904](results/q35-4b-ollama-mlx-q4-20261023-000904.json) |
|  | MTPLX (sustained profile, depth 1) | mtplx 2.12.2 mlx_lm 0.31.3 mlx 0.32.2 | 63.8 | 66.6 | 612 / 557 | 6.75 | 14.8 | [q35-4b-mtplx-20261022-234042](results/q35-4b-mtplx-20261022-234042.json) |
|  | mlx-vlm server + MTP | mlx-vlm 0.7.8, mlx 0.32.4 | 56.4 | 66.6 | 642 / 592 | 6.43 | 9.3 | [q35-4b-mlxvlm-mtp-20261022-234453](results/q35-4b-mlxvlm-mtp-20261022-234453.json) |
|  | llama.cpp + MTP (n-max 2) | 0.7.0-dev (b11804, c7fc50941) | 49.7 | 55.0 | 512 / 475 | 8.07 | 10.3 | [q35-4b-llamacpp-mtp2-20261022-234909](results/q35-4b-llamacpp-mtp2-20261022-234909.json) |
| Qwen3.5 9B | **Kapitsa** | e86dea7 | 77.5 | 84.2 | 381 / 365 | 10.85 | 8.0 | [sw-q35-9b-kapitsa-20261022-231956](results/sw-q35-9b-kapitsa-20261022-231956.json) |
|  | mlx-lm | mlx_lm 0.32.2 mlx 0.32.4 | 26.7 | 27.4 | 353 / 339 | 11.71 | 9.5 | [q35-9b-mlxlm-20261023-024404](results/q35-9b-mlxlm-20261023-024404.json) |
|  | llama.cpp (unsloth Q4_K_M GGUF) | 0.7.0-dev (b11804, c7fc50941) | 23.4 | 23.4 | 307 / 296 | 13.45 | 10.6 | [q35-9b-llamacpp-20261023-050750](results/q35-9b-llamacpp-20261023-050750.json) |
| Qwen3.8 27B | **Kapitsa** | e86dea7 | 39.8 | 41.6 | 110 / 106 | 37.57 | 19.3 | [sw-q38-27b-kapitsa-20261022-232930](results/sw-q38-27b-kapitsa-20261022-232930.json) |
|  | Ollama MLX runner (MLX 4-bit + MTP head, adaptive depth) | 0.40.8 | 22.5 | 23.6 | 99 / 94 | 41.77 | 27.9 | [q38-27b-ollama-mlx-q4-20261023-004222](results/q38-27b-ollama-mlx-q4-20261023-004222.json) |
|  | MTPLX (turbo profile, depth 3) | mtplx 2.12.2 mlx_lm 0.31.3 mlx 0.32.2 | 22.2 | 24.4 | 99 / 95 | 41.57 | 26.4 | [q38-27b-mtplx-20261023-001312](results/q38-27b-mtplx-20261023-001312.json) |
|  | mlx-vlm server + MTP | mlx-vlm 0.7.8, mlx 0.32.4 | 18.1 | 19.4 | 102 / 98 | 40.69 | 25.6 | [q38-27b-mlxvlm-mtp-20261023-002908](results/q38-27b-mlxvlm-mtp-20261023-002908.json) |
|  | llama.cpp + MTP (n-max 3) | 0.7.0-dev (b11804, c7fc50941) | 15.9 | 16.2 | 86 / 84 | 47.86 | 27.7 | [q38-27b-llamacpp-mtp3-20261023-023056](results/q38-27b-llamacpp-mtp3-20261023-023056.json) |
| Qwen3.6 35B-A3B | **Kapitsa** | e86dea7 | 106.1 | 115.7 | 855 / 764 | 4.83 | 21.2 | [sw-q36-35b-kapitsa-20261022-231735](results/sw-q36-35b-kapitsa-20261022-231735.json) |
|  | Ollama MLX runner (MLX 4-bit + MTP head, adaptive depth) | 0.40.8 | 57.5 | 64.3 | 681 / 595 | 6.07 | 26.1 | [q36-35b-ollama-mlx-q4-20261023-010300](results/q36-35b-ollama-mlx-q4-20261023-010300.json) |
|  | MTPLX (sustained profile, depth 1) | mtplx 2.12.2 mlx_lm 0.31.3 mlx 0.32.2 | 53.3 | 49.3 | 671 / 604 | 6.16 | 27.4 | [q36-35b-mtplx-20261023-005319](results/q36-35b-mtplx-20261023-005319.json) |
|  | mlx-vlm server + DFlash2 | mlx-vlm 0.7.8, mlx 0.32.4 | 53.0 | 62.5 | 695 / 652 | 5.95 | 25.6 | [q36-35b-mlxvlm-dflash2-20261023-005830](results/q36-35b-mlxvlm-dflash2-20261023-005830.json) |
|  | llama.cpp + MTP (n-max 1) | 0.7.0-dev (b11804, c7fc50941) | 52.8 | 52.5 | 578 / 527 | 7.15 | 25.8 | [q36-35b-llamacpp-mtp1-20261023-021933](results/q36-35b-llamacpp-mtp1-20261023-021933.json) |
| Flint 40B-A2.5B | **Kapitsa** | e86dea7 | 73.3 | 74.6 | 800 / 663 | 4.84 | 22.5 | [sw-flint-kapitsa-20261022-233719](results/sw-flint-kapitsa-20261022-233719.json) |
| Nemotron-Labs Dory1.0 4B | **Kapitsa** | c490f81 | 36.1 | 36.7 | 450 / 393 | 9.31 | 4.1 | [sw-dory-kapitsa-20261023-032253](results/sw-dory-kapitsa-20261023-032253.json) |
|  | mlx-lm (PR #1988, 4-bit, 2 loops, per-loop KV) | mlx_lm 0.32.2.dev0 mlx 0.32.4 | 30.0 | 30.6 | 344 / 312 | 12.15 | 10.0 | [dory-mlxlm-4bit-20261023-011927](results/dory-mlxlm-4bit-20261023-011927.json) |
| Nemotron 3 Nano 4B | **Kapitsa** | c490f81 | 34.4 | 33.0 | 765 / 698 | 5.47 | 5.1 | [sw-nano-4b-kapitsa-20261023-032747](results/sw-nano-4b-kapitsa-20261023-032747.json) |
|  | llama.cpp (NVIDIA's Q4_K_M GGUF) | 0.7.0-dev (b11804, c7fc50941) | 34.9 | 34.9 | 413 / 398 | 10.13 | 8.7 | [nano-4b-llamacpp-20261023-025004](results/nano-4b-llamacpp-20261023-025004.json) |
| Nemotron Nano 9B v2 | **Kapitsa** | c490f81 | 20.9 | 21.0 | 330 / 316 | 12.68 | 7.2 | [sw-nano-9b-kapitsa-20261023-033155](results/sw-nano-9b-kapitsa-20261023-033155.json) |
|  | mlx-lm (mlx-community 4-bit) | mlx_lm 0.32.2 mlx 0.32.4 | 24.9 | 25.3 | — / — | — | 14.1 | [nano-9b-mlxlm-4bit-20261023-012955](results/nano-9b-mlxlm-4bit-20261023-012955.json) |
| Nemotron 3.5 Lightning 30B-A3B | **Kapitsa** | c490f81 | 75.3 | 80.4 | 788 / 697 | 5.31 | 18.6 | [sw-lightning-kapitsa-20261023-033915](results/sw-lightning-kapitsa-20261023-033915.json) |
|  | mlx-vlm server + DSpark drafter (block 6) | mlx-vlm 0.7.8, mlx 0.32.4 | 39.2 | 40.0 | 514 / 498 | 8.14 | 25.9 | [lightning-mlxvlm-dspark-20261023-013824](results/lightning-mlxvlm-dspark-20261023-013824.json) |
| Nemotron 3.5 Bolt 74B-A9B | **Kapitsa** | c490f81 | 50.2 | 50.9 | 256 / 250 | 16.34 | 22.3 | [sw-bolt-kapitsa-20261023-034205](results/sw-bolt-kapitsa-20261023-034205.json) |

Flint × llama.cpp b11804 (Q4_1) is #341's clean pass, measured 2026-10-22 on this Mac: 50.9 tok/s decode, 772 / 645
tok/s at 4K / 16K ([flint-llamacpp-20261022-214530](../2026-10-22-flint-m3pro-ac/results/flint-llamacpp-20261022-214530.json),
[-215331](../2026-10-22-flint-m3pro-ac/results/flint-llamacpp-20261022-215331.json)). Flint × rec does not load on 36 GB
(#306, #341).

- **Nano 9B v2 × mlx-lm: decode only.** Its 24.9 tok/s (5 of 5 valid) stands; its prefill and follow-ups are INVALID.
  The model's chat template turns thinking off only for a `/no_think` in a message and ignores the
  `enable_thinking: false` the harness sends every engine, so mlx-lm thinks (`</think>` in the answer), and its "prefill"
  is the time to the first token of that thinking answer. This record also predates the Nemotron prompt reference, so
  its prompt length is flagged against the Qwen one. The harness sends every engine the same request, so it was not
  rerun with another one.
- `sw-q35-9b-kapitsa`: `tools_pass` false. With thinking on, the answer after the tool result came back empty; the
  tool calls themselves passed, and so did every check with thinking off. The decode and prefill phases are valid.
- Bolt has no rival: mlx-vlm needs files the catalog bundle does not have.
- Not measured. × needs a newer macOS than 14.1: mlx-serve (26.2+), oMLX (15+), Splash. —: LM Studio (not installed),
  Dory × llama.cpp (needs NVIDIA's fork), Lightning × llama.cpp (its GGUF was not downloaded).

## Kapitsa against the fastest rival, 1K to 32K

One cold request per size; a single point's ratio is within this Mac's run-to-run spread (a few percent; more for the
speculative engines' decode, whose acceptance varies with the prompt: Ollama's MTP head, mlx-vlm's DSpark drafter).

**Qwen3.5 4B**: Kapitsa (e86dea7) against Ollama MLX runner (MLX 4-bit + MTP head, adaptive depth) (0.40.8); records [sw-q35-4b-kapitsa-20261023-014331](results/sw-q35-4b-kapitsa-20261023-014331.json), [q35-4b-ollama-mlx-q4-20261023-025442](results/q35-4b-ollama-mlx-q4-20261023-025442.json)

| Prompt | Prefill, Kapitsa (tok/s) | Prefill, rival | Kapitsa / rival | Decode, Kapitsa (tok/s) | Decode, rival | Kapitsa / rival |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K (1,103 tok) | 699 | 599 | 1.17× | 103.2 | 63.0 | 1.64× |
| 2K (2,130 tok) | 714 | 608 | 1.17× | 102.7 | 62.5 | 1.64× |
| 4K (4,158 tok) | 706 | 602 | 1.17× | 90.9 | 66.4 | 1.37× |
| 8K (8,305 tok) | 686 | 581 | 1.18× | 85.7 | 54.8 | 1.56× |
| 16K (16,310 tok) | 638 | 540 | 1.18× | 73.7 | 53.0 | 1.39× |
| 32K (31,548 tok) | 556 | 475 | 1.17× | 57.9 | 45.3 | 1.28× |

Peak memory: Kapitsa 5.3 GB, rival 16.0 GB.

**Qwen3.5 9B**: Kapitsa (e86dea7) against mlx-lm (mlx_lm 0.32.2 mlx 0.32.4); records [sw-q35-9b-kapitsa-20261023-020153](results/sw-q35-9b-kapitsa-20261023-020153.json), [q35-9b-mlxlm-20261023-031754](results/q35-9b-mlxlm-20261023-031754.json)

| Prompt | Prefill, Kapitsa (tok/s) | Prefill, rival | Kapitsa / rival | Decode, Kapitsa (tok/s) | Decode, rival | Kapitsa / rival |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K (1,103 tok) | 374 | 342 | 1.09× | 57.5 | 25.6 | 2.24× |
| 2K (2,129 tok) | 382 | 348 | 1.10× | 64.5 | 25.9 | 2.48× |
| 4K (4,157 tok) | 377 | 353 | 1.07× | 60.5 | 25.4 | 2.38× |
| 8K (8,304 tok) | 379 | 349 | 1.09× | 57.3 | 25.0 | 2.29× |
| 16K (16,310 tok) | 365 | 338 | 1.08× | 50.4 | 23.2 | 2.17× |
| 32K (31,550 tok) | 336 | 317 | 1.06× | 46.9 | 21.2 | 2.22× |

Peak memory: Kapitsa 8.4 GB, rival 12.4 GB.

**Qwen3.8 27B**: Kapitsa (e86dea7) against Ollama MLX runner (MLX 4-bit + MTP head, adaptive depth) (0.40.8); records [sw-q38-27b-kapitsa-20261023-014616](results/sw-q38-27b-kapitsa-20261023-014616.json), [q38-27b-ollama-mlx-q4-20261023-025815](results/q38-27b-ollama-mlx-q4-20261023-025815.json)

| Prompt | Prefill, Kapitsa (tok/s) | Prefill, rival | Kapitsa / rival | Decode, Kapitsa (tok/s) | Decode, rival | Kapitsa / rival |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K (1,103 tok) | 108 | 96 | 1.12× | 28.6 | 17.4 | 1.65× |
| 2K (2,132 tok) | 111 | 98 | 1.12× | 25.7 | 17.2 | 1.49× |
| 4K (4,158 tok) | 111 | 99 | 1.12× | 25.2 | 17.9 | 1.41× |
| 8K (8,308 tok) | 110 | 97 | 1.13× | 25.4 | 16.2 | 1.57× |
| 16K (16,311 tok) | 106 | 94 | 1.13× | 27.3 | 17.0 | 1.61× |
| 32K (31,550 tok) | 99 | 87 | 1.13× | 20.6 | 13.0 | 1.58× |

Peak memory: Kapitsa 19.7 GB, rival 31.6 GB. The Ollama sweep swapped (the harness flags 28,700 pages swapped out
during it, peak 31.6 GB on this 36 GB Mac); its protocol run above did not.

**Qwen3.6 35B-A3B**: Kapitsa (e86dea7) against Ollama MLX runner (MLX 4-bit + MTP head, adaptive depth) (0.40.8); records [sw-q36-35b-kapitsa-20261023-015915](results/sw-q36-35b-kapitsa-20261023-015915.json), [q36-35b-ollama-mlx-q4-20261023-031417](results/q36-35b-ollama-mlx-q4-20261023-031417.json)

| Prompt | Prefill, Kapitsa (tok/s) | Prefill, rival | Kapitsa / rival | Decode, Kapitsa (tok/s) | Decode, rival | Kapitsa / rival |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K (1,105 tok) | 662 | 623 | 1.06× | 86.6 | 59.2 | 1.46× |
| 2K (2,128 tok) | 886 | 674 | 1.31× | 84.4 | 60.5 | 1.40× |
| 4K (4,157 tok) | 905 | 674 | 1.34× | 86.4 | 58.1 | 1.49× |
| 8K (8,305 tok) | 856 | 656 | 1.31× | 74.7 | 50.9 | 1.47× |
| 16K (16,308 tok) | 768 | 597 | 1.29× | 62.5 | 47.0 | 1.33× |
| 32K (31,552 tok) | 640 | 496 | 1.29× | 57.0 | 41.8 | 1.36× |

Peak memory: Kapitsa 21.4 GB, rival 27.9 GB.

**Flint 40B-A2.5B**: Kapitsa (c490f81); record [sw-flint-kapitsa-20261023-051434](results/sw-flint-kapitsa-20261023-051434.json)

| Prompt | Prefill, Kapitsa (tok/s) | Decode, Kapitsa (tok/s) |
| ---: | ---: | ---: |
| 1K (1,034 tok) | 648 | 71.2 |
| 2K (1,997 tok) | 777 | 70.8 |
| 4K (3,893 tok) | 790 | 68.4 |
| 8K (7,697 tok) | 752 | 64.6 |
| 16K (15,088 tok) | 677 | 58.5 |
| 32K (29,158 tok) | 565 | 49.7 |

Peak memory: Kapitsa 22.7 GB.

**Nemotron-Labs Dory1.0 4B**: Kapitsa (c490f81) against mlx-lm (PR #1988, 4-bit, 2 loops, per-loop KV) (mlx_lm 0.32.2.dev0 mlx 0.32.4); records [sw-dory-kapitsa-20261023-034647](results/sw-dory-kapitsa-20261023-034647.json), [dory-mlxlm-4bit-20261023-035113](results/dory-mlxlm-4bit-20261023-035113.json)

| Prompt | Prefill, Kapitsa (tok/s) | Prefill, rival | Kapitsa / rival | Decode, Kapitsa (tok/s) | Decode, rival | Kapitsa / rival |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K (1,114 tok) | 457 | 367 | 1.24× | 34.1 | 28.6 | 1.19× |
| 2K (2,169 tok) | 452 | 355 | 1.27× | 33.7 | 27.9 | 1.21× |
| 4K (4,208 tok) | 447 | 348 | 1.29× | 33.2 | 27.8 | 1.19× |
| 8K (8,412 tok) | 423 | 334 | 1.27× | 30.9 | 26.1 | 1.19× |
| 16K (16,541 tok) | 394 | 311 | 1.26× | 28.5 | 23.3 | 1.23× |
| 32K (32,005 tok) | 334 | 278 | 1.20× | 24.3 | 20.2 | 1.20× |

Peak memory: Kapitsa 4.4 GB, rival 13.2 GB.

**Nemotron 3 Nano 4B**: Kapitsa (c490f81) against llama.cpp (NVIDIA's Q4_K_M GGUF) (0.7.0-dev (b11804, c7fc50941)); records [sw-nano-4b-kapitsa-20261023-035631](results/sw-nano-4b-kapitsa-20261023-035631.json), [nano-4b-llamacpp-20261023-035932](results/nano-4b-llamacpp-20261023-035932.json)

| Prompt | Prefill, Kapitsa (tok/s) | Prefill, rival | Kapitsa / rival | Decode, Kapitsa (tok/s) | Decode, rival | Kapitsa / rival |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K (1,113 tok) | 716 | 432 | 1.66× | 34.1 | 34.8 | 0.98× |
| 2K (2,167 tok) | 760 | 413 | 1.84× | 32.5 | 34.6 | 0.94× |
| 4K (4,209 tok) | 765 | 412 | 1.85× | 33.2 | 34.2 | 0.97× |
| 8K (8,413 tok) | 741 | 411 | 1.80× | 31.5 | 33.5 | 0.94× |
| 16K (16,543 tok) | 700 | 399 | 1.75× | 31.6 | 32.0 | 0.99× |
| 32K (32,004 tok) | 593 | 377 | 1.57× | 26.9 | 27.6 | 0.97× |

Peak memory: Kapitsa 5.3 GB, rival 7.1 GB.

**Nemotron Nano 9B v2**: Kapitsa (c490f81) against mlx-lm (mlx-community 4-bit) (mlx_lm 0.32.2 mlx 0.32.4); records [sw-nano-9b-kapitsa-20261023-040345](results/sw-nano-9b-kapitsa-20261023-040345.json), [nano-9b-mlxlm-4bit-20261023-040917](results/nano-9b-mlxlm-4bit-20261023-040917.json)

| Prompt | Prefill, Kapitsa (tok/s) | Prefill, rival | Kapitsa / rival | Decode, Kapitsa (tok/s) | Decode, rival | Kapitsa / rival |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K (1,111 tok) | 312 | — | — | 20.6 | 24.0 | 0.86× |
| 2K (2,165 tok) | 329 | — | — | 20.5 | 23.7 | 0.86× |
| 4K (4,207 tok) | 332 | — | — | 20.2 | 23.7 | 0.85× |
| 8K (8,412 tok) | 321 | — | — | 20.2 | 23.3 | 0.86× |
| 16K (16,540 tok) | 316 | — | — | 19.8 | 23.4 | 0.85× |
| 32K (32,007 tok) | 291 | — | — | 19.1 | 22.5 | 0.85× |

Peak memory: Kapitsa 7.4 GB, rival 15.2 GB.

**Nemotron 3.5 Lightning 30B-A3B**: Kapitsa (c490f81) against mlx-vlm server + DSpark drafter (block 6) (mlx-vlm 0.7.8, mlx 0.32.4); records [sw-lightning-kapitsa-20261023-041546](results/sw-lightning-kapitsa-20261023-041546.json), [lightning-mlxvlm-dspark-20261023-042959](results/lightning-mlxvlm-dspark-20261023-042959.json)

| Prompt | Prefill, Kapitsa (tok/s) | Prefill, rival | Kapitsa / rival | Decode, Kapitsa (tok/s) | Decode, rival | Kapitsa / rival |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K (1,112 tok) | 609 | 444 | 1.37× | 56.1 | 33.7 | 1.66× |
| 2K (2,166 tok) | 750 | 503 | 1.49× | 51.6 | 32.4 | 1.59× |
| 4K (4,209 tok) | 792 | 500 | 1.58× | 52.0 | 24.7 | 2.11× |
| 8K (8,413 tok) | 748 | 515 | 1.45× | 48.5 | 23.9 | 2.03× |
| 16K (16,543 tok) | 696 | 497 | 1.40× | 45.4 | 27.4 | 1.66× |
| 32K (32,006 tok) | 589 | 464 | 1.27× | 34.0 | 26.2 | 1.30× |

Peak memory: Kapitsa 18.7 GB, rival 26.0 GB.

**Nemotron 3.5 Bolt 74B-A9B**: Kapitsa (c490f81); record [sw-bolt-kapitsa-20261023-043437](results/sw-bolt-kapitsa-20261023-043437.json)

| Prompt | Prefill, Kapitsa (tok/s) | Decode, Kapitsa (tok/s) |
| ---: | ---: | ---: |
| 1K (1,113 tok) | 230 | 38.6 |
| 2K (2,168 tok) | 256 | 32.2 |
| 4K (4,208 tok) | 262 | 36.8 |
| 8K (8,407 tok) | 256 | 40.1 |
| 16K (16,539 tok) | 251 | 31.2 |
| 32K (32,002 tok) | 228 | 30.6 |

Peak memory: Kapitsa 22.3 GB.

Flint's rival sweep is #341's ([context-sweep](../2026-10-22-flint-m3pro-ac/context-sweep/)): another script
(`ctx_sweep.py`, power-of-two sizes, a 256-token answer), so its points are not this table's. There, Kapitsa (cff1b1e)
against llama.cpp b11804: prefill 653 / 527 at 1K, 742 / 768 at 4K, 516 / 502 at 32K; decode 71.6 / 51.7, 68.1 / 49.5,
48.9 / 37.6.

Nano 9B v2 × mlx-lm: its sweep record passed the harness's checks, but it is the same thinking answer as above, so its
prefill is not shown; its decode is.

## Flags on the cells used

The harness's soft flags (the record stays valid) on the records in the tables above:

- **Repetition loop** in one or more of the five T = 0.6 answers (distinct-4-gram below 0.5), on both sides:
  Kapitsa on the 4B (`sw-q35-4b-kapitsa-20261022-232718`, 0.45) and Dory (`sw-dory-kapitsa-20261023-032253`, 0.45);
  Ollama on the 4B (decode 0.38 and greedy 0.47) and the 35B-A3B (0.47), MTPLX on the 4B (0.49), llama.cpp on the 9B
  (0.45), mlx-lm on Dory (0.50). The Nano 4B's `kapitsa bench` result carries two (0.03 and 0.45).
- **`sw-nano-9b-kapitsa-20261023-033155`**: one of the five T = 0.6 answers stopped at 84 tokens (`finish_reason=stop`,
  not 512, at 21.02 tok/s); the median of the five is 20.85, of the four 512-token answers 20.84.
- **Nano 9B v2 × mlx-lm**: its follow-up turns hit the length limit (`finish_reason=length`), part of the thinking
  answer above; its prefill is not shown.
- **The 27B × Ollama sweep** swapped (28,700 pages), see its table.

## Records set aside

Kept, not used above:

- `results/display-woken/`: the first 9B × llama.cpp and × mlx-lm runs. The display woke during them (01:11:53 to
  01:19:22, the lock screen; the video decoder idle). Both ran again with the display asleep.
- `results/download-overlap/`: the first Lightning × mlx-vlm and Bolt sweeps, which overlapped a model download on
  this Mac. Both ran again in full.
- `results/qwen-ref/`: the first Nano 4B × llama.cpp run, INVALID only for its prompt length against the Qwen reference
  (4187 / 16519 tokens, within 1% of the Nemotron one). It ran again against the Nemotron reference.
- `sw-flint-kapitsa-20261022-232410`: INVALID, Flint's reasoning text in the decode answers (run before
  `--always-reasons` was added to its config).
- `sw-flint-kapitsa-20261023-020611`: the first Flint sweep, INVALID because the harness's sweep ignored
  `--always-reasons` (fixed in #361). Swept again on c490f81 with the fix: `sw-flint-kapitsa-20261023-051434`, above.
- The five e86dea7 Nemotron-H records and the two 0.2.8 records (above).

The queue's rival picker first read the wrong field of the protocol records and picked no rival for the 4B, 27B and
35B-A3B sweeps; it was fixed and those rival sweeps ran after the Kapitsa ones, on the same Mac with the same checks.
