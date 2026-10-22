# Benchmarks

How fast Kapitsa runs Qwen3.6-35B-A3B and Qwen3.8-27B on a MacBook Pro (Nemotron 3.5 Bolt and the small Qwen3.5-4B
have their own sections: [Bolt](#nemotron-35-bolt-74b-a9b-2026-10-18), [4B](#qwen35-4b-2026-10-18), and so have
Qwen3.8-Flash-Next and Nemotron 3.5 Super-VL and Lightning on the M5 Max:
[Flash-Next](#qwen38-flash-next-180b-a7b-m5-max-128-gb-2026-10-18),
[Super-VL](#nemotron-35-super-vl-120b-a12b-m5-max-128-gb-2026-10-18),
[Lightning](#nemotron-35-lightning-30b-a3b-m5-max-128-gb-2026-10-18)), compared with the engines people run these
models with on a Mac: llama.cpp, Ollama, LM Studio, mlx-lm, oMLX and MTPLX (and mlx-serve, a day later: [its own
section](#mlx-serve-2026-10-18); the M4 Pro table's later cells: [their own section](#the-m4-pro-table-filled-in-2026-10-19);
Nemotron Dory1.0 4B: [its own section](#nemotron-dory10-4b-2026-10-19); the 16 GB MacBook Pro M1 Pro: [its own
section](#macbook-pro-m1-pro-16-gb-2026-10-19)). [A real coding-agent session](#a-real-coding-agent-session-2026-10-18), replayed
request by request, puts the engines through what an agent does. Each engine ran at its latest release with its own
best settings, on the same 4-bit weights wherever it accepts them, through the same requests and the same checks. The
harness and its configs are in [bench/harness](../bench/harness/README.md), and every result file behind these tables
is in [bench/results/2026-10-17](../bench/results/2026-10-17/).

## The README's tables

What each cell of the README's poster and its collapsed Nemotron table ran, Mac by Mac (the runs are in the sections below). The poster's prefill / decode cells read the 4K-prompt prefill from `bench/results`: Kapitsa's prefill is from its newest run without a contention or invalid flag, and its decode is the number in the cell's decode column; they can come from different runs (the M5 Pro's Kapitsa cells for the 27B, 35B-A3B and Lightning are single 2026-10-21 runs, prefill and decode together; the other M5 Pro cells are the 2026-10-18 and 10-19 runs). The other engines' prefill is from the stored run closest to the decode value shown, flagged or not.

**MacBook Pro M4 Pro 48 GB**
- Ollama, LM Studio, MTPLX and mlx-serve can't run Dory or Bolt, so the Nemotron table leaves them out.
- MLX: Apple's mlx-lm or mlx-vlm, whichever runs the model. oMLX runs DFlash2 on the 27B and 35B, MTP on the 4B, no
  drafter on Bolt; mlx-serve the faster of MTP and DFlash2.
- Kapitsa's Qwen cells (4B 201, 27B 63, 35B 161 tok/s): main with KFlash trees
  ([#187](https://github.com/nace-ai/kapitsa/pull/187)), measured 2026-10-21 against main before it
  ([KFlash trees](#kflash-trees-m4-pro-2026-10-21)). Its Nemotron cells: KFlash on Lightning 112 and Bolt 69 tok/s, measured the same day.
- Splash 1.4.0 (Inco AI, `bench/harness/configs/splash.json`): the 27B (53) and the 35B (135) with its DFlash2
  drafts, measured 2026-10-21 in the same session as Kapitsa's KFlash runs. It runs only those two models and needs an
  M3 or newer. Rapid-MLX left the README's tables for it; its runs stay in the sections below.
- Other apps used the GPU during MLX's 4B run and llama.cpp's Dory run (contention flagged).
- A real opencode session on the 35B: Kapitsa 49 s, mlx-serve 62 s, oMLX 66 s
  ([details](#a-real-coding-agent-session-2026-10-18)); on 2026-10-21, Kapitsa with KFlash 51.4 s and Splash 1.4.0 57.0 s
  ([details](#kflash-trees-m4-pro-2026-10-21)).
- Dory: 2 loops; Kapitsa per-loop KV (`--loop-kv per`), llama.cpp NVIDIA's Dory fork on its Q5_K_M (last-loop KV),
  MLX mlx-lm's Dory pull request on snsnc's 4-bit (per-loop KV) ([details](#nemotron-dory10-4b-2026-10-19)).
- Lightning: NVIDIA's NVFP4 with its DSpark drafter on Kapitsa and MLX (mlx-vlm); oMLX and llama.cpp not measured yet
  ([details](#nemotron-35-lightning)).

**MacBook Pro M5 Max 128 GB**
- One session, Kapitsa first and last (each cell is the newest run with an unflagged decode phase, else the newest run).
  Results: [bench/results/2026-10-18-m5-max-session](../bench/results/2026-10-18-m5-max-session/).
- MLX: mlx-vlm with MTP on the 4B (Qwen's 4-bit head, block 3; mlx-lm 135) and the 27B (the model's head; mlx-lm 30),
  mlx-vlm with DFlash2 on the 35B (incoai's drafter, block 5; mlx-lm 112). oMLX runs DFlash2 on the 27B and 35B, MTP on the 4B, no drafter on Bolt;
  mlx-serve MTP on the 4B (Qwen's 4-bit head) and 35B (DFlash2 209 tok/s), DFlash2 on the 27B (MTP 80); llama.cpp MTP n-max 2 on the 4B and
  35B, 3 on the 27B.
- Other apps used the GPU during both of Kapitsa's 27B runs (contention flagged: 102 and, first run, 115 tok/s; greedy
  113 and 110) and the MLX Flash-Next run. mlx-serve's DFlash2 draft tree on the 27B (blocks of 16) wrote 130 tok/s
  sampled, 143 greedy, unflagged.
- Dory: 2 loops; Kapitsa per-loop KV (`--loop-kv per`, build 36c2937; its closing run, 117 tok/s, greedy 130; the first, 113, was flagged), llama.cpp
  NVIDIA's Dory fork (98a74c97) on its Q5_K_M, MLX mlx-lm's Dory pull request (208065f) on snsnc's 4-bit.
- Super-VL: mlx-serve 26.10.2 loads neither NVIDIA's checkpoint nor the text-only MLX 4-bit conversion
  (`error: UnsupportedNemotronLatentMoe`). MTPLX 2.12.2 runs that conversion without MTP (it has no MTP head): 44 tok/s;
  its memory guard aborts the 16K-token prompt (the run is INVALID for that; its decode, 4K and tool phases are valid). mlx-serve's DFlash on the 4B: 228 tok/s, below its MTP.
- Rapid-MLX 0.16.3: on the 4B MTP with Qwen's 4-bit head, 3 drafts; on the 27B MTP on its own pack (its mtp/ sidecar,
  3 drafts; served by path it needs the head named, else it decodes plain at 32 tok/s); on the 35B no speculation (its catalog turns it off
  for this MoE).
- Bolt: oMLX's LiLiCorr drafter fails at T = 0.6; run greedy throughout it writes 121-124 tok/s, against Kapitsa's
  greedy 125 (the first run 119).
- Flash-Next: the same 4-bit weights and MTP head on Kapitsa, oMLX and MLX (mlx-vlm); mlx-serve runs its vendor's
  4.7-bit pack, MTPLX its author's pack, Ollama its NVFP4 tag. Ollama fails every 4K and 16K prompt there (out of GPU
  memory). LM Studio refuses to load its 119 GB GGUF on 128 GB; Rapid-MLX 0.16.3 refuses the 4-bit pack (its guard estimates
  a 148 GB working set and warns of a kernel panic). Neither guard was overridden. Kapitsa uses the least memory (70.5 GB; mlx-serve
  78.8, oMLX 104).
- Nemotron: NVIDIA's NVFP4 on Kapitsa, mlx-vlm and Ollama, MLX 4-bit on oMLX, mlx-lm, mlx-serve and MTPLX, Q4_0 on
  llama.cpp, Q4_K_M on LM Studio. On Lightning oMLX reads prompts faster (3,600 against 1,900 tok/s at 4K).

**MacBook Pro M1 Pro 16 GB**
- Qwen: the full protocol in one hold of the GPU lock, Kapitsa 0.1.36 first and last (its runs: 85.5 and 78.5 tok/s at
  T = 0.6). Kapitsa's 4B cell is now main with KFlash, 2026-10-21: the mean of two runs (92.6 and 97.5 tok/s; main before
  it 90.7 and 89.2, [details](#kflash-trees-m4-pro-2026-10-21)). Splash needs an M3 or newer. Each engine in its fastest setting on this Mac, chosen in short runs:
  oMLX MTP, mlx-serve MTP depth 2, Ollama's MLX runner with MTP, MTPLX depth 1, llama.cpp MTP n-max 3, MLX mlx-lm.
  Rapid-MLX (no speculation, its faster setting here) in a bracket of its own, Kapitsa 0.1.36 first and last.
  Kapitsa 0.1.36 reads 16K-token prompts more slowly than mlx-lm, Rapid-MLX and LM Studio here (269 tok/s against 289,
  286 and 277); main since #89 and #92 reads them at 343-349.
- Dory: 2 loops; Kapitsa last-loop KV, its default, measured later with 0.1.37 (Dory's NVFP4 in layout L2 on the M1:
  40.5 tok/s, against 35.2 with 0.1.36 in the bracket; per-loop speculation 36.7), llama.cpp NVIDIA's Dory fork on its
  Q5_K_M, MLX mlx-lm's Dory pull request on snsnc's 4-bit ([details](#macbook-pro-m1-pro-16-gb-2026-10-19)).
- Qwen3.5-9B: Kapitsa 0.1.37 with its DFlash drafter (the better of first and last, 50.7 tok/s), mlx-serve MTP depth 2,
  MLX mlx-lm; the other engines' installs had been removed to make room for it ([details](#qwen35-9b-at-4-bit)).
- Qwen3.6-35B-A3B with 2-bit experts: unsloth's UD-IQ2_XXS GGUF on Kapitsa (#134) and llama.cpp b11669, which swapped
  on this Mac ([details](#qwen36-35b-a3b-with-2-bit-experts-gguf-ud-iq2_xxs)); the MLX engines don't read GGUF files,
  and Ollama and LM Studio were not run on it.
- Nemotron Nano: Kapitsa on NVIDIA's FP8 (the 4B, with #129's 8-bit head and #131's tail kernels) and NVFP4 (the 9B v2);
  llama.cpp on Q4_K_M GGUFs, MLX mlx-lm on NVIDIA's BF16 (4B) and NVFP4 through its `nemotron_h` code (9B v2), from
  #103's runs ([details](#nemotron-3-nano-4b-and-nemotron-nano-9b-v2)).

**MacBook Pro M5 Pro 48 GB**
- One session, model by model, closing with a control round on an idle machine; each cell is the engine's fastest
  setting from its newest run ([details](#macbook-pro-m5-pro-48-gb-2026-10-19)).
- MLX: mlx-vlm with MTP on the 4B and 27B, DFlash2 on the 35B; mlx-lm on Lightning and Dory, mlx-vlm on Bolt. oMLX
  runs MTP on the 4B and Lightning, DFlash2 on the 27B and 35B, no drafter on Bolt; mlx-serve MTP on the 4B, 35B and
  Lightning, its DFlash2 draft tree on the 27B; Ollama its MLX runner with MTP, GGUF with MTP on the 35B; llama.cpp
  MTP n-max 3 on the 4B and 27B, 2 on the 35B and Lightning; Rapid-MLX MTP with 3 drafts on the 4B and 27B (its own
  27B pack), no speculation on the 35B (its catalog turns it off for this model).
- Dory: 2 loops; Kapitsa per-loop KV, llama.cpp NVIDIA's Dory fork on its Q5_K_M, MLX mlx-lm's Dory pull request on
  snsnc's 4-bit.

## How the README poster was measured

Tokens per second, each engine in its fastest setup, its decoding method under each value (no spec: no speculative decoding). Every cell reads prefill / decode: prefill on a 4K-token prompt, decode of a 512-token coding answer. × cannot run that model on that Mac · — not measured (a dash before the slash: no prefill run for that engine). The poster shows the models that differ most per Mac; the [details](#the-readmes-tables) and [bench/README.md](../bench/README.md) have every cell.

Prefill is the reading speed on a 4K-token prompt. Kapitsa's comes from the newest run without a contention or invalid flag: 2026-10-21 on the M5 Pro (Kapitsa 0.2.2, built from a13f8fd for the 22:00-22:30 runs on the poster, b50f1fc for the 19:56-20:12 ones; a13f8fd differs from b50f1fc only in README, docs, poster and stored results, no source or build files, so the engine is the same; macOS 26.5.2, [results](../bench/results/2026-10-21-m5pro-48gb/README.md)), 2026-10-19 on the M4 Pro (4B: 06:20 run) and M1 Pro, 2026-10-18 on the M5 Max, where every stored run is flagged for GPU contention, so read its prefill as a lower bound. Prefill on the M5 Max predates the M5 prefill kernels ([#205](https://github.com/nace-ai/kapitsa/pull/205), [#215](https://github.com/nace-ai/kapitsa/pull/215), Nemotron-H's) and is likely higher now. The other engines' prefill is from their own stored run closest to the decode value shown, in the same session as that decode (2026-10-18 to 10-21), flagged or not: several of those runs carry contention flags, LM Studio's every one, and all of the M5 Max's. Where another engine reads faster, the poster shows it: the headline, the blue outlines and the orange bars rank decode only. Ollama on Flash-Next (M5 Max) and Splash (no prefill run) show a dash. The M5 Pro's Kapitsa cells (27B, 35B-A3B, Lightning) are one run each from 2026-10-21, prefill and decode together, and mlx-serve's 27B cell is a re-run the same evening; the other engines' M5 Pro cells are still the 2026-10-19 runs, so those pairs were not measured side by side (on the 27B mlx-serve's decode went from 71 in the 10-19 run to 80, and it is ahead of Kapitsa's 74). KFlash decode cells were measured on 2026-10-21 ([#187](https://github.com/nace-ai/kapitsa/pull/187), [numbers](#kflash-trees-m4-pro-2026-10-21)); the DFlash, DFlash2, DSpark and LiLiCorr cells come from earlier runs. Every run is listed in [bench/README.md](../bench/README.md).

## Quality: Qwen3.6 35B-A3B against bf16 (2026-10-19)

How far each precision's next-token distributions are from the bf16 model's, teacher-forced over the held-out sequences
of [bench/quality](../bench/quality/README.md): C++, Python, English prose and a coding-agent chat, 4 x 1023 positions,
with the `math` sequence reported apart. The bf16 checkpoint (mlx-community/Qwen3.6-35B-A3B-bf16 at 04bbe07) streamed
from Hugging Face one layer at a time (`eval_stream.py --f32`: fp32 activations on both sides), and KL(bf16 || precision)
is exact over the whole vocabulary.

| Precision | KL mean | KL median | KL p95 | Same top token | Perplexity | KL on `math` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | | | | | 5.570 | |
| 8-bit | 0.0168 | 0.0012 | 0.025 | 96.7% | 5.571 | 0.0064 |
| 6-bit | 0.0228 | 0.0033 | 0.056 | 95.9% | 5.602 | 0.0109 |
| 4-bit, Kapitsa's default for `35b` | 0.1161 | 0.0272 | 0.341 | 89.1% | 5.783 | 0.0925 |

- The 4-bit default is the speed choice; `--model 35b:6bit` is five times closer to bf16, with perplexity within 0.6%.
- 8-bit matches bf16's perplexity (+0.02%); its mean KL is close to 6-bit's because of a few outlying positions (its
  median is a third of 6-bit's).
- The 4-bit row is the pack Kapitsa downloads (Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed at 9546051: 4-bit group
  64, the router and shared-expert gates 8-bit). The 6-bit and 8-bit rows are round-to-nearest from the bf16 weights,
  group 64, as `mlx_lm.convert` makes the mlx-community packs Kapitsa's `6bit` and `8bit` variants download.
- Nemotron Dory1.0 4B's NVFP4 checkpoint is a KL of 0.020 from its BF16 checkpoint ([Precisions](usage.md#precisions)).
- Result file: [bench/results/2026-10-19-quality](../bench/results/2026-10-19-quality/qwen3.6-35b-a3b-kl-vs-bf16.json).

## Summary

On both models Kapitsa was the fastest engine measured on every speed metric: writing (decode, sampled and greedy),
reading a long prompt (prefill at 4K and 16K tokens) and answering a follow-up turn in a long conversation. It also
used the least memory on both models.

| Metric | Qwen3.6-35B-A3B: Kapitsa | Next best | | Qwen3.8-27B: Kapitsa | Next best | |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| Decode, T = 0.6 | 153.7 tok/s | oMLX + DFlash2, 126.5 | 1.22× | 63.0 tok/s | oMLX + DFlash2, 45.3 | 1.39× |
| Decode, greedy | 178.9 tok/s | oMLX + DFlash2, 128.2 | 1.39× | 67.3 tok/s | oMLX + DFlash2, 45.7 | 1.47× |
| Prefill, 4K-token prompt | 1,169 tok/s | oMLX + DFlash2, 1,035 | 1.13× | 389 tok/s | oMLX + DFlash2, 136 | 2.86× |
| Prefill, 16K-token prompt | 986 tok/s | oMLX + DFlash2, 901 | 1.09× | 335 tok/s | oMLX + DFlash2, 128 | 2.60× |
| Cached follow-up, 4K conversation | 0.06 s | MTPLX, 0.10 s | 1.63× | 0.28 s | MTPLX, 0.33 s | 1.17× |
| Cached follow-up, 16K conversation | 0.09 s | MTPLX, 0.15 s | 1.71× | 0.27 s | MTPLX, 0.39 s | 1.46× |
| Peak memory | 21.1 GB | oMLX + DFlash2, 23.4 GB | | 19.4 GB | oMLX (no speculation), 20.5 GB | |

Ratios are computed from the unrounded results. Measured on one MacBook Pro (M4 Pro, 48 GB); on each model the
next fastest engine on every metric ran in the same session as Kapitsa (35B: the morning of 2026-10-17; 27B: one
evening session on the final build). The [conditions](#conditions) section lists what differed between sessions.

In [a real opencode session](#a-real-coding-agent-session-2026-10-18) replayed on every engine (19 requests, up to
23K tokens; MacBook Pro, M4 Pro, 48 GB, 2026-10-18), Kapitsa answered every request in 48.6 s on Qwen3.6-35B-A3B at
4-bit, 1.27× faster than the next fastest (mlx-serve + MTP, 61.6 s), and in 150.2 s on Qwen3.8-27B at 4-bit, 1.85×
faster than the next (oMLX + DFlash2, 278.1 s).

For an additional Kapitsa-only run on a different machine, see the [M5 Max / 128 GB report](benchmarks-m5-max.md).
Those runs include background GPU contention and are reported separately from the comparison below.

## Results

Decode figures are the median of 5 runs, with the range in brackets. Prefill is tokens per second of a cold prompt.
A cached follow-up is the time to the first token of the answer to a new message in a conversation that already
holds the 4K or 16K prompt (median of 3 follow-ups). Peak memory is the server's whole process tree, in GB (2^30
bytes). Bold: best in the column. Every run passed the validity checks (thinking off, prompt length within 1% of
the reference, no reused cache on cold prompts) and every engine passed the tool-call checks.

### Qwen3.6-35B-A3B (mixture of experts, 3B active)

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** (DFlash2 drafter) | **153.7** (146.6–165.8) | **178.9** (173.7–184.2) | **1,169** | **986** | **0.06** | **0.09** | **21.1** | pass |
| oMLX + DFlash2 (block 5, adaptive verify) | 126.5 (115.1–135.3)<sup>1</sup> | 128.2 (115.8–135.4) | 1,035 | 901 | 0.29 | 0.40 | 23.4 | pass |
| Ollama, MLX runner + MTP (adaptive depth) | 88.6 (87.1–94.0) | 92.1 (90.9–94.4) | 892 | 776 | 0.21 | 0.32 | 32.2 | pass |
| LM Studio (MLX engine) | 86.0 (85.1–86.8)<sup>2</sup> | 91.1 (90.6–91.9)<sup>2</sup> | 957 | 846 | 0.69 | 0.79 | 24.7 | pass |
| MTPLX (sustained profile, depth 1) | 79.8 (78.7–84.6) | 73.4 (70.6–76.5) | 918 | 804 | 0.10 | 0.15 | 31.7 | pass |
| Ollama, GGUF + MTP (shipped depth 2) | 69.7 (67.9–75.5) | 76.7 (75.0–78.3) | 701 | 541 | 0.24 | 0.28 | 5.6 + 19.9 mapped<sup>6</sup> | pass |
| llama.cpp + MTP (n-max 1) | 68.9 (66.8–69.9) | 65.5 (64.6–69.2) | 663 | 545 | 0.20 | 0.21 | 27.4 | pass |
| llama.cpp + MTP (n-max 3) | 68.1 (65.5–75.0) | 73.5 (70.8–74.2) | 659 | 553 | 0.18 | 0.20 | 27.5 | pass |
| Ollama, GGUF (MTP off) | 63.4 (63.1–65.0) | 63.4 (62.9–63.7) | 768 | 643 | 0.22 | 0.26 | 5.1 + 19.4 mapped<sup>6</sup> | pass |
| llama.cpp | 62.7 (62.5–63.1) | 62.2 (62.1–62.6) | 728 | 630 | 0.18 | 0.20 | 26.2 | pass |
| llama.cpp + MTP (n-max 2) | 60.1 (52.6–68.8) | 56.8 (53.0–59.4) | 584 | 483 | 0.21 | 0.23 | 27.5 | pass |

mlx-lm, and oMLX without speculation, have no 35B row: they were not re-run on the final harness for this model
(their earlier 35B runs used a previous version of the harness and are not comparable).

### Qwen3.8-27B (dense)

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** (DFlash2 drafter)<sup>3</sup> | **63.0** (55.0–66.3) | **67.3** (65.2–67.4) | **389** | **335** | **0.28** | **0.27** | **19.4** | pass |
| oMLX + DFlash2 (block 5, verify dflash) | 45.3 (43.5–46.3) | 45.7 (45.1–45.9) | 136 | 128 | 0.92 | 1.26 | 24.0 | pass |
| MTPLX (turbo profile, depth 3) | 36.2 (35.0–36.5) | 39.3 (38.6–39.4) | 126 | 121 | 0.33 | 0.39 | 33.4 | pass |
| Ollama, MLX runner + MTP (adaptive depth) | 32.2 (31.7–32.8) | 32.2 (32.2–32.5) | 124 | 119 | 0.42 | 0.51 | 32.2 | pass |
| Ollama, GGUF + MTP (shipped depth 4) | 20.4 (19.9–22.1) | 19.1 (18.7–19.7) | 83 | 78 | 0.52 | 0.60 | 13.8 + 15.4 mapped<sup>6</sup> | pass |
| llama.cpp + MTP (n-max 3) | 18.3 (14.9–18.7) | 17.3 (15.2–18.7) | 84 | 77<sup>4</sup> | 0.64 | 0.69 | 30.6 | pass |
| LM Studio (MLX engine) | 15.4 (15.2–15.6)<sup>2</sup> | 15.3 (15.0–15.5)<sup>2</sup> | 127 | 122 | 0.56 | 0.65 | 24.7 | pass |
| oMLX (no speculation) | 13.5 (13.0–13.8) | 13.3 (12.9–14.1) | 109 | 104<sup>4</sup> | 38.52<sup>5</sup> | 157.46<sup>5</sup> | 20.5 | pass |
| llama.cpp + MTP (n-max 1) | 13.3 (12.8–13.9) | 12.7 (12.7–12.8) | 87 | 80 | 0.60 | 0.67 | 31.5 | pass |
| mlx-lm | 13.3 (13.1–13.5) | 13.4 (13.3–13.4) | 100 | 92 | 0.57 | 0.90 | 23.6 | pass |
| llama.cpp + MTP (n-max 2) | 13.0 (10.8–17.1) | 16.3 (15.7–17.0) | 81 | 78 | 0.65 | 0.68 | 31.7 | pass |
| Ollama, GGUF (MTP off) | 10.9 (9.0–12.1) | 9.4 (9.0–11.1) | 89 | 80 | 0.50 | 0.58 | 12.6 + 15.2 mapped<sup>6</sup> | pass |
| llama.cpp | 10.0 (9.8–10.3) | 9.9 (9.9–10.9) | 90 | 82 | 0.63 | 0.64 | 29.9 | pass |

### Notes

1. One of oMLX's five sampled runs, its fastest (135.3 tok/s), fell into a repetition loop (distinct-4-gram ratio
   0.45; the harness flags anything below 0.5), which speculative decoding speeds up. Without that run its median
   is 122.4 tok/s.
2. LM Studio stops at 511 of the 512 requested tokens. Its rates use the 511 tokens it reports.
3. The 27B rows of Kapitsa, oMLX + DFlash2, MTPLX, Ollama's MLX runner and LM Studio come from one session on
   2026-10-17 between 22:08 and 22:58, with Kapitsa's final build (main at e5c6f41). Kapitsa ran before and after
   the others; the table shows the run after. The run before gave 62.2 / 65.0 tok/s decode, 395 / 345 tok/s
   prefill, 0.22 / 0.27 s follow-ups and 19.4 GB ([confirmations](../bench/results/2026-10-17/confirmations/)).
4. The Mac swapped during this 16K prefill (llama.cpp: 96,796 pages, about 1.5 GB; oMLX: 6,611 pages), which may
   have slowed it.
5. oMLX without speculation did not reuse its prompt cache for follow-up turns on this model (it reported 0 reused
   tokens), so every follow-up re-read the whole conversation.
6. Ollama's GGUF runner maps the weights from disk instead of loading them, so they are not in the server's own
   memory: the first figure is the server's memory, the second the resident mapped weights. llama.cpp ran with
   `-lm none`, which loads the weights into its own memory like the other engines.

## What was compared

| Engine | Version measured | Weights | Speculative decoding | Other settings |
| --- | --- | --- | --- | --- |
| Kapitsa | main at b457a52 (35B) and e5c6f41 (27B), 2026-10-17, Release builds | MLX 4-bit | DFlash2 drafter (the default) | defaults; 32K context, 1 slot |
| llama.cpp | 0.7.0, build b11669 (620d2225c), official macOS arm64 release | Ollama's Q4_K_M GGUF | MTP with the head inside the GGUF, `--spec-draft-n-max` 1, 2 and 3; and off | `-c 32768 -np 1 -ngl 999 -fa on -b 2048 -ub 2048 -lm none --jinja`, official Qwen chat template, `--min-p 0` |
| Ollama, GGUF | 0.40.5 (bundled llama.cpp b11669) | its Q4_K_M GGUF tags | built-in MTP at the shipped depth (2 on the 35B, 4 on the 27B); and off | dedicated server: 32K context, 1 slot, flash attention |
| Ollama, MLX runner | 0.40.5 (MLX bebd6422, Metal 4 build) | MLX 4-bit + MTP head, imported without copying | built-in MTP with adaptive depth (no knobs) | same server settings |
| LM Studio | llmster 0.0.25-1 (headless), MLX runtime 1.13.1 (mlx-engine ab52366, mlx 0.32.3) | lmstudio-community MLX 4-bit (the same shards) | none: its MLX engine has none for these models | LM Studio's staff-pick model entry; 32K context, `--parallel 1` |
| mlx-lm | 0.32.2 (mlx 0.32.4) | MLX 4-bit | none for these models | prompt cache (4 sequences, 8 GB), 2048-token prefill steps |
| oMLX | 0.7.1rc1 (mlx 0.32.4, mlx-lm 0.31.4.dev132) | MLX 4-bit | DFlash2 drafter quantized to 4-bit at load, block 5, verify `adaptive` (35B) or `dflash` (27B); and none (27B) | memory guard `aggressive`, 4 GB RAM + 4 GB SSD prompt cache |
| MTPLX | 2.12.2 (mlx 0.32.2, mlx-lm 0.31.3) | MLX 4-bit + MTP head | MTP as the weights' runtime file recommends: profile `sustained`, depth 1 (35B); `turbo`, depth 3 (27B) | inline session post-commit (follow-ups use its session cache) |

"Best settings" means: speculative decoding wherever the engine offers it for these models, at the engine's or the
weights' own recommendation, with each depth measured where it is a knob (llama.cpp), and the fastest variant
found in short smoke tests where there were several (oMLX's verify modes and block sizes). Where an engine has the
settings, context is 32K and there is one conversation slot; every request sends the same sampling parameters. The
configs in [bench/harness/configs](../bench/harness/configs) record each choice and its reason. LM Studio's
llama.cpp runtime on its own Q4_K_M GGUF with MTP drafting was not measured; on the 27B it may decode faster than its
MLX engine (standalone llama.cpp with MTP decodes the 27B at 18.3 tok/s, LM Studio's MLX engine at 15.3).

We also ran three experimental engines through the same harness: AX Engine 7.7.0, Darkbloom 0.9.22 and lithos-metal
0.1.2. None was faster than Kapitsa on any speed metric. They are not in the tables because this repository has no
configs for them.

**Weights.** The MLX engines read the same files, MLX affine 4-bit with group size 64:
Qwen3.6-35B-A3B is mlx-community's 4-bit conversion (router and shared-expert gates in 8 bits) with the model's
MTP head, as published by MTPLX
([Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed](https://huggingface.co/Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed)
@9546051, 19.5 GB including the 0.5 GB head); Qwen3.8-27B is
[lmstudio-community/Qwen3.8-27B-MLX-4bit](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-4bit) @6067b15
(15.0 GB) plus the MTP head of MTPLX's 27B release (0.2 GB). LM Studio's own downloads of these models are the
same shards, and Ollama's MLX runner read them through a zero-copy import, checked tensor by tensor. llama.cpp and
Ollama's GGUF runner read Ollama's Q4_K_M GGUFs (35B: 20.2 GB, blob d372de8e; 27B: 15.7 GB, blob f5f1dd89), which
contain the MTP head; they read slightly more bytes per generated token than the MLX files (their output layer is
Q6_K). Sizes here are in GB of 2^30 bytes, like the memory figures.
Kapitsa and oMLX use the DFlash2 drafters
[incoai/Qwen3.6-35B-A3B-DFlash2](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2) @51ef7b6 and
[incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) @015e795, and both quantize them to
4 bits when loading. Kapitsa's memory includes its drafter (about 0.6 GB on the 35B and 1.6 GB on the 27B).

**Kapitsa's build.** The 35B rows are main at b457a52, run in the same session as the other engines' 35B runs; the
27B rows are the final main, e5c6f41, run in one session with the 27B rows of the engines closest to it. Both were
run from a copy of the build (sha256 in the result files). The final build on the 35B, after the other sessions:
160.2 / 173.2 tok/s decode, 1,127 / 947 tok/s prefill, 0.06 / 0.07 s follow-ups, 20.9 GB, within the spread of the
b457a52 row ([confirmations](../bench/results/2026-10-17/confirmations/)).

## How it was measured

All requests go through each engine's OpenAI-compatible HTTP API, streaming, the way a coding agent talks to it.
The details are in the [harness README](../bench/harness/README.md#what-a-run-measures).

- **Decode:** a 512-token answer to a coding request (an LRU cache module with tests), 5 runs at temperature 0.6
  (top_p 0.95, top_k 20, Qwen's recommendation) and 5 greedy runs. The rate is measured over the middle of the
  stream (the first 16 and last 4 tokens are dropped), so neither time to first token nor engines that send
  several tokens per event distort it. Greedy runs make every engine write nearly the same text.
- **Prefill:** a cold prompt of about 4K and 16K tokens (4,133 and 16,286 with these chat templates) of source code,
  with a random nonce at the start of the system message so no engine can reuse a cached prefix. Rate = prompt
  tokens / time to first token.
- **Cached follow-up:** after each cold prompt, three follow-up questions on the growing conversation, 1 s apart;
  the median time to first token.
- **Tool calls:** streaming and non-streaming requests with two tools, with thinking off and on, and a tool-result
  round trip.
- **Memory:** `phys_footprint_peak` of the server's process tree, read with macOS's `footprint` after every phase.

**What makes a run count.** Thinking is switched off with both `chat_template_kwargs.enable_thinking=false` and
`reasoning_effort: "none"` (engines differ in which they honour), and any reasoning text makes a run invalid. Every
request sets every sampling parameter, including `min_p 0` and neutral penalties, because the engines' defaults
differ. Prompts come from a frozen corpus (excerpts of Hugging Face Transformers source code, sha256 recorded), and
cold prompts must come back within 1% of the reference token count with no cached tokens reported. Decode runs are
checked to stop at exactly 512 tokens and for repetition loops. Before timing, each server is warmed up (a ~2.9K-token
prompt, a tool call, and three full decodes for engines whose speculation adapts across requests).

**One engine at a time, on a quiet GPU.** Each configuration runs alone under an exclusive GPU lock, from server
start until its memory is released, and its timed requests start only after no other app has used more than 5% of
the GPU for 15 s (every run here got there within 21 s). A monitor samples every process's GPU time every 2 s and
flags a run when other apps use more than 5% of the GPU for three samples in a row, or on average. The tables use
one run per configuration: its newest run on the final harness without an invalid measurement or a contention flag.
Every run in the tables passed with no contention flag.

## Conditions

MacBook Pro (Mac16,7), Apple M4 Pro with a 14-core CPU, 20-core GPU, 48 GB of unified memory (273 GB/s); macOS 27.0.1
(26A434); on AC power. mlx-lm, oMLX and MTPLX used the macOS 26 build of the MLX wheels; LM Studio and Ollama ship
their own MLX builds (LM Studio's is the macOS 14 build, Ollama loads its Metal 4 build on macOS 26 and later).

Two conditions differed between the two days, and the result files record both for every run:

- **Power mode.** The runs on 2026-10-16 were in the Automatic energy mode, those on 2026-10-17 in High Power mode.
  Kapitsa, LM Studio, MTPLX and oMLX + DFlash2 on both models, and Ollama's MLX runner on the 27B, ran on
  2026-10-17 in High Power mode.
- **Background GPU load.** Two kinds of desktop GPU load were present at times: the Photos analysis daemons, busy
  for hours after the update to macOS 27, and the desktop app that drove the runs. The harness treats such load as
  ambient: it does not block or flag a run, but it is recorded. It averaged 20–43% of GPU time in the 2026-10-16
  daytime runs (llama.cpp, Ollama GGUF, mlx-lm, oMLX without speculation), 8–33% in the evening runs (Ollama MLX
  runner on the 35B), 0% in the 2026-10-17 morning runs and 1–16% in the 2026-10-17 evening session (the desktop app
  that drove the runs: Kapitsa 1.1%, oMLX + DFlash2 9.1%, MTPLX 8.8%, Ollama's MLX runner 16.4%, LM Studio 0%).

We have not measured how much either changes an engine's speed. On both models the next fastest engine on every
metric ran in the same session and power mode as Kapitsa: on the 35B in the morning of 2026-10-17 with no ambient load,
on the 27B in the evening session above. The 2026-10-16 rows (llama.cpp, Ollama's GGUF runner, mlx-lm, oMLX without
speculation, and Ollama's MLX runner on the 35B) are in Automatic mode with ambient load; none of them is next
fastest on any metric.

## Run index

The tables can be rebuilt from the result files with
`python3 bench/harness/compare_table.py bench/results/2026-10-17 --out /tmp/table`.

| Result file | Started | Power mode | Foreign GPU, mean / max (%) | Ambient GPU, mean (%) | Lock held (s) |
| --- | --- | --- | ---: | ---: | ---: |
| `q36-35b-kapitsa-20261017-084636.json` | 2026-10-17 08:48 | High Power | 0.0 / 0.0 | 0.0 | 107 |
| `q36-35b-omlx-best-20261017-091029.json` | 2026-10-17 09:29 | High Power | 0.0 / 0.0 | 0.0 | 139 |
| `q36-35b-ollama-mlx-q4-20261016-212529.json` | 2026-10-16 21:26 | Automatic | 0.0 / 0.0 | 8.4 | 168 |
| `q36-35b-lmstudio-20261017-085626.json` | 2026-10-17 09:11 | High Power | 0.0 / 0.0 | 0.0 | 160 |
| `q36-35b-mtplx-20261017-085604.json` | 2026-10-17 09:08 | High Power | 0.3 / 14.0 | 0.0 | 205 |
| `q36-35b-ollama-mtp-20261016-145225.json` | 2026-10-16 15:00 | Automatic | 1.4 / 3.1 | 22.6 | 179 |
| `q36-35b-llamacpp-mtp1-20261016-140843.json` | 2026-10-16 14:23 | Automatic | 1.3 / 3.4 | 21.6 | 180 |
| `q36-35b-llamacpp-mtp3-20261016-144042.json` | 2026-10-16 14:50 | Automatic | 1.4 / 3.0 | 23.9 | 172 |
| `q36-35b-ollama-nomtp-20261016-150248.json` | 2026-10-16 15:11 | Automatic | 1.1 / 2.9 | 20.1 | 258 |
| `q36-35b-llamacpp-20261016-135336.json` | 2026-10-16 14:06 | Automatic | 1.4 / 4.3 | 20.1 | 182 |
| `q36-35b-llamacpp-mtp2-20261016-142601.json` | 2026-10-16 14:37 | Automatic | 1.2 / 3.7 | 24.9 | 244 |
| `q38-27b-kapitsa-20261017-225432.json` | 2026-10-17 22:55 | High Power | 0.0 / 0.7 | 1.1 | 225 |
| `q38-27b-ollama-mlx-q4-20261017-221219.json` | 2026-10-17 22:14 | High Power | 0.0 / 1.8 | 16.4 | 504 |
| `q38-27b-mtplx-20261017-222801.json` | 2026-10-17 22:30 | High Power | 0.0 / 0.0 | 8.8 | 569 |
| `q38-27b-omlx-best-20261017-222043.json` | 2026-10-17 22:22 | High Power | 0.0 / 0.8 | 9.1 | 438 |
| `q38-27b-ollama-mtp-20261016-130604.json` | 2026-10-16 13:16 | Automatic | 2.5 / 5.5 | 43.0 | 691 |
| `q38-27b-llamacpp-mtp3-20261016-124604.json` | 2026-10-16 12:55 | Automatic | 2.2 / 5.2 | 40.8 | 730 |
| `q38-27b-lmstudio-20261017-223730.json` | 2026-10-17 22:38 | High Power | 0.0 / 0.0 | 0.0 | 678 |
| `q38-27b-omlx-20261016-161709.json` | 2026-10-16 16:30 | Automatic | 1.4 / 7.6 | 31.6 | 1,219 |
| `q38-27b-llamacpp-mtp1-20261016-115128.json` | 2026-10-16 12:00 | Automatic | 2.2 / 7.0 | 41.9 | 835 |
| `q38-27b-mlxlm-20261016-155425.json` | 2026-10-16 16:04 | Automatic | 1.8 / 6.5 | 30.5 | 792 |
| `q38-27b-llamacpp-mtp2-20261016-121303.json` | 2026-10-16 12:34 | Automatic | 2.2 / 5.0 | 40.9 | 781 |
| `q38-27b-ollama-nomtp-20261016-132653.json` | 2026-10-16 13:38 | Automatic | 1.9 / 8.2 | 36.1 | 959 |
| `q38-27b-llamacpp-20261016-112909.json` | 2026-10-16 11:36 | Automatic | 1.9 / 8.6 | 36.2 | 973 |

Two extra Kapitsa runs from the evening session are in [confirmations](../bench/results/2026-10-17/confirmations/)
(the 27B run before the other engines, `q38-27b-kapitsa-20261017-220825.json`, and the final build on the 35B,
`q36-35b-kapitsa-20261017-225817.json`); they are not in the tables. Times are local (UTC+5); "Started" is when the
benchmark began, after start-up and warm-up. In the result files,
local paths are replaced by placeholders (`~`, `<bench-root>` for the folder the harness ran from before it moved into
this repository, `<kapitsa-checkout>`, `<build-dir>`); nothing else was changed. `engine: "nace-infer"` and
`nace_infer_git` are Kapitsa under its earlier name, and `{root}` in the recorded configs is that earlier folder; the
configs in [bench/harness/configs](../bench/harness/configs) are the same settings in this repository's layout.

## Qwen3.8-Flash-Next 180B-A7B (M5 Max, 128 GB, 2026-10-18)

Qwen's 180B mixture of experts (`qwen4_exp`: 512 experts, 10 active; Gated DeltaNet and sparse attention; a 51B-parameter
n-gram embedding table), on the M5 Max (40-core GPU, 128 GB, macOS 26.4), against every engine that runs it on a Mac:

- Kapitsa, mlx-vlm 0.7.8 and oMLX 0.7.1rc1 read the same folder, Nace AI's MLX pack
  (`nace-ai/Qwen3.8-Flash-Next-MLX-4bit`: routed experts, Gated DeltaNet, shared experts and LM head 4-bit, attention and
  hyper-connections 6-bit, the rest 8-bit; 71.4 GB plus the 32 GB n-gram table at 4-bit and the 1.5 GB MTP head;
  105.9 GB). Kapitsa and mlx-vlm map the table from the SSD and read its rows on demand; oMLX keeps it in memory. Each
  engine runs the model's own MTP head: Kapitsa at depth 2 (its default), oMLX at a fixed depth 3, mlx-vlm at block 4
  (from a drafter folder holding the pack's own head, which mlx-vlm 0.7.8 cannot read from the pack itself).
- oMLX also runs its author's own quantization, `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` (oQ4e: an importance-matrix mix of
  4-8-bit at group 64 / 128, routed experts 4-bit, 106.3 GB, the most downloaded MLX conversion), with MTP at a fixed
  depth 2.
- llama.cpp 0.7.0 (b11669, the latest release) runs ggml-org's official IQ4_NL GGUF (routed experts IQ4_NL, dense
  layers Q8_0, the table Q4_0; 102.0 GB) with ggml-org's MTP GGUF, n-max 4.
- Ollama 0.40.6 runs the tag it serves on Apple silicon, `qwen3.8-flash-next:125b-mlx` (NVFP4, 104.9 GB), in its MLX
  runner with its adaptive MTP, and its GGUF tag `125b-a6b-q4_K_M` (Q4_K_M, 120.1 GB; no MTP draft in that tag) in its
  GGUF runner.

One session on 2026-10-18, every configuration run twice, with
[configs/flash-next.json](../bench/harness/configs/flash-next.json); the table has each configuration's second valid
run. Kapitsa is build 95fbcd6 of `opt/flash-next` (MTP at depth 2, its default from that build). Result files,
recipes, weights and revisions:
[bench/results/2026-10-18-m5-max-flash-next](../bench/results/2026-10-18-m5-max-flash-next/).

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB)<sup>3</sup> | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** (MTP, depth 2) | 101.8 (100.6–109.0) | **107.6** (103.7–109.0) | 860 | 850 | 0.08 | **0.07** | 70.5 + 5.5 mapped | pass |
| Kapitsa (no speculation) | 84.9 (84.6–85.5) | 84.8 (84.3–85.7) | 862 | 868 | **0.08** | 0.08 | **70.3** + 5.2 mapped | pass |
| oMLX 0.7.1rc1 + MTP (fixed depth 3), Nace pack | **106.5** (103.2–112.1) | 106.4 (105.2–108.3) | 1,247 | 1,316 | 0.26 | 0.42 | 103.9 | pass |
| oMLX, Nace pack (no speculation) | 59.5 (59.3–60.2) | 60.0 (59.7–60.2) | 1,339 | 1,394 | 0.25 | 0.38 | 103.4 | pass |
| oMLX + MTP (fixed depth 2), oQ4e ⚠ | 105.5 (95.9–109.4) | 98.7 (95.9–101.5) | 1,395 | 1,383 | 0.28 | 0.45 | 104.0 | pass |
| oMLX, oQ4e (no speculation) ⚠ | 69.7 (64.6–71.3) | 66.8 (66.4–68.7) | **1,521** | **1,691** | 0.25 | 0.34 | 103.6 | pass |
| llama.cpp 0.7.0 + MTP (n-max 4), IQ4_NL | 96.1 (84.9–98.2) | 94.6 (92.1–99.2) | 842 | 890 | 0.25 | 0.24 | 83.2 + 2.3 mapped | fail<sup>2</sup> |
| llama.cpp, IQ4_NL (no speculation) | 46.4 (45.9–46.6) | 45.1 (42.6–45.7) | 848 | 913 | 0.23 | 0.23 | 79.3 + 2.1 mapped | pass |
| mlx-vlm 0.7.8 + MTP (block 4), Nace pack ⚠ | 76.8 (70.1–77.0) | 77.0 (72.4–77.5) | 1,030 | 1,049 | 0.38 | 0.49 | 77.0 + 5.3 mapped | pass |
| mlx-vlm, Nace pack (no speculation) ⚠ | 53.0 (47.2–53.9) | 52.4 (48.1–53.6) | 948 | 1,031 | 0.43 | 0.55 | 74.3 + 5.0 mapped | pass |
| Ollama 0.40.6 MLX runner + MTP, NVFP4 tag | 60.0 (57.8–64.9) | 62.8 (59.5–68.8) | —<sup>1</sup> | — | — | — | 108.4 | pass |
| Ollama GGUF runner, Q4_K_M tag (no MTP) | 50.1 (49.3–50.2) | 48.3 (48.1–48.8) | 705 | 672 | 0.22 | 0.24 | 88.0 + 2.2 mapped | pass |

⚠: contention flagged in the run's decode phases (another app used the GPU while it wrote; see Conditions). Most other
rows are flagged in their prefill phases only, for a desktop process using 3-6% of the GPU.

1. Ollama's MLX runner fails every 4K and 16K prompt on this Mac: it holds 98-100 GiB with this tag, and the prefill
   exceeds the 107.5 GiB Metal working-set limit ("Insufficient Memory"). Its decode and tool calls work.
2. The second run fails one tool-call check (the model listed the workspace with `run_shell` instead of reading the
   file); the first run passed every check.
3. Memory is the server's peak footprint. "+ mapped": file pages of memory-mapped weights resident at the end (Kapitsa's
   and mlx-vlm's n-gram table rows, llama.cpp's and Ollama's GGUF table); oMLX and Ollama's MLX runner hold the 32 GB
   table in their footprint.

Speculative settings came from smoke tests (two runs each, 4K prompt): oMLX's fixed MTP depth 3 on the pack beat its
adaptive default (up to 3) and depths 1, 2, 4 and 5, and depth 2 was its best on oQ4e; mlx-vlm's MTP block 4 beat the
drafter's default and blocks 2, 3 and 5; llama.cpp's n-max 4 beat 1-3, 5 and 6 at T = 0.6. The smoke-test numbers are in
the results folder's README.

Conditions: desktop apps stayed open. A `com.apple.WebKit` process used 3-6% of the GPU most of the day, so most runs
are flagged in their prefill phases. A Google Chrome helper used much more during mlx-vlm's second pass (7-10% mean)
and both oQ4e passes (14-16% mean, peaks of 60%): those rows are flagged in every phase and probably read low (oQ4e's
MTP smoke test just before gave 110.7 tok/s sampled against the table's 105.5). Downloads did not overlap any run
except the first `flash-kapitsa` run (the end of the Ollama GGUF download; 98.9 / 102.6 tok/s there). The first run of
each configuration is in [confirmations](../bench/results/2026-10-18-m5-max-flash-next/confirmations/): sampled decode
within 5% of the rows above except mlx-vlm + MTP's (81.0 tok/s, before the Chrome load), oQ4e's plain one (64.9) and
llama.cpp + MTP's (89.3). Not run: mlx-lm (no release supports `qwen4_exp`; only an open pull request, which also drops the MTP head),
LM Studio and MTPLX.

Earlier Kapitsa builds measured in the same session are kept in the results folder, outside the table: 8912f1b, whose
server crashed on every 16K prompt (fixed in 62ab7ed: sparse-attention scratch buffers released while still in use),
and 62ab7ed itself (MTP at depth 1, its default then: 104.0 / 102.1 tok/s sampled, plain 83.7 / 85.2; 4K prefill
858-898 tok/s, 16K 869-902; follow-ups 0.08 s; 70.5 GB).

## Nemotron 3.5 Bolt 74B-A9B (2026-10-18)

NVIDIA's own 24 GB checkpoint (`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`: NVFP4 Mamba projections, IQ2_XXS experts,
Q8_0 attention, latent projections and LM head) on the engines that load it on a Mac. oMLX and mlx-vlm read it
through the model file of NVIDIA's Bolt bundle for MLX (`txgsync/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-MLX`,
which carries NVIDIA's files byte-identical); Kapitsa reads the safetensors directly. All runs in one session on
2026-10-18 between 06:37 and 07:50, back to back, with [configs/bolt.json](../bench/harness/configs/bolt.json); the
desktop's own GPU work (the app the session ran in: 10-40% of the GPU) was there for every engine alike. Result files:
[bench/results/2026-10-18](../bench/results/2026-10-18/).

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** (LiLiCorr drafter, GPU + Neural Engine prefill) | **63.9** (62.5–66.1)<sup>1</sup> | **67.4** (64.6–69.6) | **436** | **385** | **0.28** | **0.32** | **22.7** + 1.0 mapped<sup>2</sup> | pass |
| oMLX + LiLiCorr drafter | fails<sup>3</sup> | 52.8 (50.3–53.9)<sup>3</sup> | 288 | 297 | 0.75 | 0.84 | 31.5 | pass |
| oMLX (no speculation) | 33.9 (33.5–34.1) | 34.1 (33.6–34.2) | 296 | 300 | 0.72 | 0.72 | 30.3 | pass |
| mlx-vlm (no speculation) | 31.8 (31.1–32.6) | 32.9 (32.4–33.1) | 300 | 296 | 14.52<sup>4</sup> | 57.86<sup>4</sup> | 32.3 | pass |

1. One of the five sampled runs looped (distinct-4-gram 0.43; flagged): 63.6 tok/s, not the fastest. Without it the
   median is 64.2.
2. Kapitsa maps the BF16 embedding from its file: those pages (1.0 GB resident) are clean file pages outside the
   process footprint. Steady footprint after each phase: Kapitsa 22.4–22.7 GB, oMLX 23.9–24.3, oMLX + LiLiCorr
   25.2–25.6, mlx-vlm 23.2–23.5.
3. oMLX 0.7.1rc1 with the drafter fails every request at T > 0 (its sampled DFlash selection), so its sampled decode
   and its prefill phases have no valid result (`bolt-omlx-lilicorr-20261018-063713.json`); the `-greedy` run sends
   temperature 0 in every phase, so it has two greedy series, with medians 52.8 and 45.9: the table shows the higher,
   and its prefill and follow-up numbers.
4. mlx-vlm's server reused no prompt cache for the follow-up turns (0 cached tokens): every follow-up re-read the
   whole conversation.

Kapitsa's earlier run in the same session (before the weight packing that lowered its memory; `confirmations/`) gave
64.7 / 65.6 tok/s decode, 455 / 396 tok/s prefill, 0.25 / 0.33 s follow-ups and 24.9 GB.

## Nemotron 3.5 Super-VL 120B-A12B (M5 Max, 128 GB, 2026-10-18)

NVIDIA's NVFP4 checkpoint (`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`: NVFP4 experts, FP8 Mamba
projections, attention and LM head, BF16 latent projections; 70 GB of weights) needs a 96 GB+ Mac, so this comparison
ran on a MacBook Pro with an M5 Max (40-core GPU, 128 GB, macOS 26.4), with every engine that runs the model on a Mac:

- Kapitsa and mlx-vlm read NVIDIA's checkpoint itself. mlx-vlm's Super-VL support (and its native MTP drafter) is
  newer than its latest release (0.7.8 refuses the model), so it ran at git main 73d366e, an unreleased build.
- oMLX 0.7.1rc1 and mlx-lm 0.32.2 cannot read it (`Model type nemotron_h_omni not supported`). They ran the
  only text-only MLX conversion, `rapid-mlx/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-MLX-4bit` (affine 4-bit from
  NVIDIA's BF16 weights, 69.5 GB). Those weights carry no MTP head, and oMLX's drafter path covers no Nemotron-H
  model, so neither has speculative decoding here.
- llama.cpp 0.7.0 (b11669, the latest release) ran ggml-org's official Q4_0 GGUF (71.8 GB; its NVFP4 GGUF has no
  Metal kernels), with the model's MTP head (ggml-org's `mtp-...-Q4_0.gguf`) at each depth, and with NVIDIA's
  LiLiCorr drafter as ggml-org converted it.

One session on 2026-10-18, every configuration run twice, with
[configs/super-vl.json](../bench/harness/configs/super-vl.json); the table has each configuration's second run. Result
files, weights and revisions: [bench/results/2026-10-18-m5-max](../bench/results/2026-10-18-m5-max/).

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** (LiLiCorr drafter, GPU + Neural Engine prefill) | **84.5** (78.1–93.2) | **87.0** (76.3–90.0) | 627 | 509 | 0.22 | 0.26 | **70.4** + 1.0 mapped<sup>1</sup> | pass |
| Kapitsa (no drafter) | 46.4 (45.0–47.4) | 46.5 (45.3–47.3) | 617 | 501 | 0.20 | 0.25 | 67.7 + 1.0 mapped | pass |
| llama.cpp + MTP (n-max 3) | 65.5 (63.0–69.7) | 66.1 (64.7–67.5) | 833 | 882 | 0.22 | 0.20 | 81.9 | pass |
| llama.cpp + MTP (n-max 2) | 58.8 (56.5–61.8) | 59.4 (58.7–60.0) | 830 | 877 | 0.22 | 0.20 | 81.7 | pass |
| llama.cpp + MTP (n-max 1) | 49.6 (48.6–52.0) | 49.4 (48.9–49.6) | 818 | 874 | 0.23 | 0.21 | 81.6 | pass |
| llama.cpp (no speculation) | 35.9 (35.0–37.1) | 35.4 (35.1–35.6) | 764 | 821 | **0.19** | 0.20 | 78.5 | pass |
| llama.cpp + LiLiCorr drafter (n-max 7) | 26.0 (23.4–31.1) | 31.0 (29.9–31.4) | 681 | 746 | 0.24 | 0.32 | 86.2 | pass |
| oMLX, MLX 4-bit (no speculation) | 51.8 (49.7–52.1) | 50.4 (50.1–51.0) | **989** | **1,049** | 0.33 | 0.43 | 74.8 | pass |
| mlx-lm, MLX 4-bit | 45.5 (44.7–45.8) | 45.4 (44.7–45.6) | 885 | 966 | 0.20 | 0.25 | 75.6 | pass |
| mlx-vlm + MTP (block 4) | 46.3 (43.9–48.5) | 49.1 (48.9–49.6) | 756 | fails<sup>2</sup> | 0.52 | fails<sup>2</sup> | 108.0 | pass |
| mlx-vlm (no speculation) | 30.7 (30.3–30.8) | 30.9 (30.7–31.0) | 710 | 812 | 0.53 | 0.61 | 99.3 | pass |

1. Kapitsa maps the BF16 embedding from its file (1.0 GB of clean file pages outside the process footprint), as on
   Bolt. The others' memory includes their whole weights: mlx-vlm turns the FP8 layers into BF16 at load.
2. mlx-vlm with its MTP drafter runs out of GPU memory on the 16K prompt (`[METAL] Command buffer execution failed:
   Insufficient Memory`, peak 108 GB), in both runs and also with its prefix cache off.

Kapitsa writes fastest, sampled and greedy (1.3× llama.cpp + MTP, the next fastest), and uses the least memory. It
reads prompts slowest: every other engine prefills at 680-1,050 tok/s against Kapitsa's 500-630 (oMLX and mlx-lm on
their 4-bit weights fastest), and its follow-ups (0.22 / 0.26 s) are close to llama.cpp's (0.19 / 0.20 s) and mlx-lm's
rather than ahead.
Speculative settings came from smoke tests: mlx-vlm's MTP block 4 beat 2, 3 (its PR's setting), 5 and 6; llama.cpp's
LiLiCorr drafter is slower than no drafter at all (its NVFP4 GGUF has no Metal kernels), at n-max 7 as at the default 3.

Conditions: desktop apps stayed open. One `com.apple.WebKit` process used 3-5% of the GPU throughout and rose past the
harness's 5% threshold in every engine's prefill phases, so every row but oMLX's is flagged CONTENTION. Kapitsa's
rows show much more of it during its own prefill (31% mean in the 16K phase, which runs on the GPU and the Neural
Engine together; its decode phases saw 4-5% like the others'), in both of its runs; its prefill rates are close to the
ones measured outside the server (672-684 / 514-516 tok/s at 4K / 16K). At T = 0.6 this model repeats itself in some
512-token answers on every engine (flagged, kept). The first run of each configuration is in
[confirmations](../bench/results/2026-10-18-m5-max/confirmations/): Kapitsa 78.7 / 80.3 tok/s decode there, the others
within 3% of the rows above except llama.cpp + LiLiCorr (28.6 tok/s sampled).

## Nemotron 3.5 Lightning 30B-A3B (M5 Max, 128 GB, 2026-10-18)

NVIDIA's NVFP4 checkpoint (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`: NVFP4 experts, shared expert and LM
head, FP8 Mamba projections, BF16 attention; 18.2 GB of weights) with its DSpark draft model, on the M5 Max (40-core
GPU, 128 GB, macOS 26.4), against every engine that runs the model on a Mac:

- Kapitsa and mlx-vlm 0.7.8 read NVIDIA's checkpoint itself, both with the same DSpark drafter. mlx-vlm's MTP drafter
  for Nemotron-H is newer than that release, so its MTP row ran git main 73d366e, an unreleased build.
- oMLX 0.7.1rc1 and mlx-lm 0.32.2 cannot read it (ModelOpt `MIXED_PRECISION`: NVFP4 experts with FP8 Mamba
  projections; mlx-lm refuses it, oMLX fails to load it). They ran mlx-community's MLX 4-bit conversion
  (`mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit`, affine 4-bit from NVIDIA's BF16 weights, 17.8 GB).
  oMLX's MTP path ran EigenLabs' 4-bit conversion, which keeps the MTP head (mlx-community's drops it); neither oMLX
  nor mlx-lm has a drafter path for Nemotron-H.
- llama.cpp 0.7.0 (b11669, the latest release) ran ggml-org's official Q4_0 GGUF (18.9 GB) with the model's MTP head
  (ggml-org's `mtp-...-Q4_0.gguf`) at each depth, and NVIDIA's DSpark drafter as a community GGUF (NVFP4; ggml-org
  publishes none).

One session on 2026-10-18, every configuration run twice, with
[configs/lightning.json](../bench/harness/configs/lightning.json); the table has each configuration's second valid run.
Result files, weights and revisions: [bench/results/2026-10-18-m5-max-lightning](../bench/results/2026-10-18-m5-max-lightning/).

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** (DSpark drafter) | **238.7** (231.1–243.4) | **218.1** (211.2–258.2)<sup>1</sup> | 1,899 | 1,861 | 0.07 | 0.09 | **18.6** + 0.7 mapped<sup>2</sup> | pass |
| Kapitsa (no drafter) | 160.1 (160.1–160.3) | 160.3 (160.1–160.7) | 1,850 | 1,802 | 0.08 | 0.09 | 17.9 + 0.7 mapped | pass |
| oMLX + MTP (depth 2), MLX 4-bit | 189.9 (183.5–196.2) | 199.3 (188.7–202.4) | 3,521 | 3,613 | 0.12 | 0.15 | 21.6 | pass |
| oMLX, MLX 4-bit (no speculation) | 164.7 (161.8–166.2) | 166.6 (164.8–167.6) | **3,606** | **3,756** | 0.11 | 0.14 | 20.9 | pass |
| llama.cpp + MTP (n-max 3) | 159.1 (147.1–176.1) | 162.0 (150.6–170.0) | 2,835 | 2,812 | 0.08 | 0.09 | 24.2 | pass |
| llama.cpp + MTP (n-max 2) | 156.3 (148.2–167.0) | 163.0 (151.8–168.4) | 2,838 | 2,841 | 0.08 | **0.08** | 24.1 | pass |
| llama.cpp + MTP (n-max 1) | 142.4 (140.5–148.5) | 142.3 (140.7–146.3) | 2,831 | 2,840 | 0.07 | 0.08 | 24.1 | pass |
| llama.cpp (no speculation) | 116.8 (115.5–117.0) | 113.2 (112.8–114.8) | 2,662 | 2,823 | **0.07** | 0.08 | 22.4 | pass |
| llama.cpp + DSpark drafter (n-max 3) | 85.1 (84.9–86.5) | 88.4 (80.0–92.3) | 2,572 | 2,650 | 0.08 | 0.09 | 25.3 | pass |
| mlx-lm, MLX 4-bit | 139.3 (135.5–140.7) | 145.6 (145.0–146.9) | 3,365 | 3,422 | 0.08 | 0.11 | 21.1 | pass |
| mlx-vlm + DSpark drafter (block 6) | 137.4 (131.3–144.4) | 146.3 (120.6–149.5) | 2,400 | 2,722 | 0.28 | 0.31 | 29.3 | pass |
| mlx-vlm + MTP (git main, block 4) | 121.5 (107.1–132.2) | 122.5 (116.1–132.6) | 2,524 | 2,842 | 0.28 | 0.32 | 46.5 | pass |
| mlx-vlm (no speculation) | 108.9 (108.5–109.0) | 111.0 (109.4–111.4) | 2,557 | 2,939 | 0.29 | 0.31 | 27.7 | pass |

1. Greedy decoding with the drafter varies from answer to answer (near-ties in the verified tokens send the five
   greedy answers down slightly different paths with different acceptance): Kapitsa's first run gave a greedy median
   of 241.7 (236.2–254.8) and a sampled one of 229.9.
2. Kapitsa maps the BF16 embedding from its file (0.7 GB of clean file pages outside the process footprint). The
   others' memory includes their whole weights: mlx-vlm turns the FP8 layers into BF16 at load.

Kapitsa writes fastest, sampled (1.26× oMLX + MTP, the next fastest) and greedy (1.09×), and uses the least memory. It
reads prompts slowest: every other engine prefills at 2,400–3,760 tok/s against Kapitsa's 1,860–1,900 (oMLX on its
4-bit weights fastest, 1.9–2.0× Kapitsa; Kapitsa runs Lightning's prefill on the GPU only), and its follow-ups
(0.07 / 0.09 s) tie llama.cpp's rather than lead. Without a drafter, Kapitsa's 160 tok/s is just behind oMLX's plain
165 on its smaller 4-bit weights (Kapitsa reads Lightning's FP8 Mamba and BF16 attention layers as stored).
Speculative settings came from smoke tests: mlx-vlm's DSpark block 6 beat 4 and 8 (NVIDIA's DFlash drafter was slower
still), its MTP block 4 beat 2, 3 and 5, oMLX's fixed MTP depth 2 beat its default depth 1, depths 3 and 4 and its
adaptive mode, and llama.cpp's DSpark GGUF was faster at n-max 3 than 7, and slower than no drafter at either (its
NVFP4 drafter GGUF).

Conditions: desktop apps stayed open. One `com.apple.WebKit` process used 5–7% of the GPU throughout, so every run is
flagged CONTENTION in its prefill phases except oMLX + MTP's and mlx-vlm's plain one (foreign load 3–6% mean, up to
16%; Kapitsa's, in its longer prefill phases, 5.7–6.4% mean). llama.cpp + MTP n-max 1's second run is INVALID (one
sampled answer closed a reasoning block it never opened, `</think>`), so it ran a third time, the run in the table. At
T = 0.6 two of llama.cpp + MTP n-max 2's answers and one greedy answer of plain llama.cpp repeat themselves (flagged,
kept). The first run of each configuration is in
[confirmations](../bench/results/2026-10-18-m5-max-lightning/confirmations/): Kapitsa 229.9 / 241.7 tok/s decode
there (1.18× / 1.18× oMLX + MTP's 195.4 / 204.5 in the same pass), the others' sampled decode within 5% of the rows
above except llama.cpp + MTP n-max 1 (128.5 tok/s) and llama.cpp + DSpark (89.8).

## mlx-serve (2026-10-18)

[mlx-serve](https://github.com/ddalcu/mlx-serve) 26.10.2 (its latest release, MLX 0.32.3), a Zig server over MLX with
its own Metal kernels, in both of its speculative setups for these models, MTP and DFlash2, against Kapitsa (main at
c93f27c). One session on 2026-10-18, 12:40-13:38, under one hold of the GPU lock: Kapitsa first and last, mlx-serve in
between, with [configs/mlx-serve.json](../bench/harness/configs/mlx-serve.json). Result files:
[bench/results/2026-10-18-mlx-serve](../bench/results/2026-10-18-mlx-serve/).

Kapitsa was faster than mlx-serve on every metric on both models and used less memory; mlx-serve led in no column. The
closest is writing on the 27B: mlx-serve's DFlash2 draft tree wrote 51.9 tok/s sampled and 55.1 greedy, against
Kapitsa's 58.4 and 61.3 (56.8-61.5 and 60.8-62.5 over its four runs in the session). That is the fastest any engine
but Kapitsa has written this model on this Mac (next: oMLX + DFlash2, 45.3 and 45.7 on 2026-10-17). mlx-serve's tree
gets about as many tokens out of a round as Kapitsa's verify (4.9-5.4 against 4.6-5.3), but its rounds take longer
(about 97 ms against 81-84 ms; among other things it runs the drafter at 8 bits, Kapitsa at 4). On the 35B mlx-serve's
faster setup is MTP, 123.9 / 130.0 tok/s against Kapitsa's 166.8 / 173.1 (138.7 / 161.8 in its run before mlx-serve);
with DFlash2 it switched DFlash off early in every answer. Kapitsa read long prompts 1.14-1.16× faster than
mlx-serve's MTP setup on the 35B and 1.9-2.7× faster on the 27B, and answered follow-ups on the 35B in 0.05-0.07 s
against 0.26-0.38 s (mlx-serve's cache left 63-95 tokens of each follow-up to read there, Kapitsa's 20-27).

**Qwen3.6-35B-A3B**

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** (DFlash2 drafter)<sup>1</sup> | **166.8** (138.2–173.8) | **173.1** (167.1–176.6) | **1,145** | **961** | **0.05** | **0.06** | **20.9** | pass |
| mlx-serve + MTP (depth up to 2) | 123.9 (118.3–130.3) | 130.0 (117.6–135.7) | 986 | 838 | 0.26 | 0.38 | 22.5 | pass |
| mlx-serve + DFlash2 (block 4) | 104.2 (104.1–104.3)<sup>2</sup> | 107.0 (107.0–107.1)<sup>2</sup> | 911 | 783 | 0.26 | 0.38 | 22.9 | pass |

**Qwen3.8-27B**

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** (DFlash2 drafter)<sup>1</sup> | **58.4** (55.3–65.1)<sup>3</sup> | **61.3** (58.5–61.6) | **264**<sup>3</sup> | **323**<sup>3</sup> | **0.23** | **0.26** | **19.4** | pass<sup>3</sup> |
| mlx-serve + DFlash2 (draft tree, block 8) | 51.9 (50.0–54.3) | 55.1 (54.5–56.4) | 126 | 120<sup>3</sup> | 0.27 | 0.40 | 21.5 | pass |
| mlx-serve + MTP (depth up to 4) | 38.0 (34.1–42.3) | 38.9 (37.7–39.6) | 126 | 120<sup>3</sup> | 0.29 | 0.40 | 20.1 | pass |

1. Kapitsa ran before and after mlx-serve; the table shows its newest run. On the 35B that is the run after; the run
   before gave 138.7 (134.9–155.7) / 161.8 (142.0–173.7) tok/s decode, 1,123 / 953 tok/s prefill, 0.05 / 0.07 s
   follow-ups and 20.9 GB. On the 27B it is the second of two re-runs at the end of the session; its four 27B runs
   gave 56.8–61.5 / 60.8–62.5 tok/s decode, 242–296 / 231–323 tok/s prefill, 0.23–0.25 / 0.26 s follow-ups and
   19.4 GB.
2. DFlash ran for the first 8 to 33 rounds of each timed answer, then switched itself off for the rest of it: the
   drafts it accepted per round (1.73–1.78) stayed under mlx-serve's bar for MoE models (1.80). The rest of each
   answer was plain decoding, hence the narrow range.
3. Flagged for contention: Chrome and Slack used more than 5% of the GPU for sustained stretches (up to 90% at peaks)
   in every 27B run of the session, most of all in the long prefills (mlx-serve's 16K prefill takes over two minutes).
   Each flagged 27B setting was re-run (Kapitsa twice; a second re-run of mlx-serve's MTP was stopped in its warm-up
   to keep the session short), every re-run was flagged again, and the table shows the newest run with its flagged
   phases marked. The earlier mlx-serve runs: MTP 36.3 / 38.8 tok/s decode, 127 / 120 tok/s prefill, 0.36 / 0.44 s
   follow-ups, 20.2 GB; DFlash2 52.8 / 55.0, 126 / 117, 0.32 / 0.42 s, 21.5 GB. Ambient load (the camera daemon and
   the app that drove the runs, never flagged) took 10–18% of the GPU in the 27B runs until the last one (3.7%), in
   which Kapitsa's 16K prefill reached 323 tok/s; on a quiet Mac on 2026-10-17 Kapitsa read the 27B's prompts at 389 /
   335 tok/s.

**Weights.** On the 35B mlx-serve read the vendor's own pack,
[ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit](https://huggingface.co/ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit) @6122e2b: its
trunk shards are the ones every engine here reads (sha256-identical), plus its MTP sidecar (mlx-community's 4-bit
head, 0.44 GB). The vendor's 27B pack
([ddalcu/Qwen3.8-27B-MLX-Serve-4bit](https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-4bit) @901aa73) is its own
conversion of the model (bf16 embedding; in the tensors sampled about 0.2% of the 4-bit codes differ from
lmstudio-community's), so on the 27B mlx-serve read the trunk every engine reads with that pack's MTP head (its 29
`mtp.*` tensors byte for byte, 0.29 GB) and its `drafter/`, the same file as the DFlash2 drafter Kapitsa reads. For
DFlash2 on the 35B it read incoai's drafter, also Kapitsa's. `bench/harness/tools/mlx_serve_pack.py` builds both
folders from the shared weights (APFS clones) and the vendor's files that differ.

**Settings.** mlx-serve's own defaults, except where short runs just before the session measured faster (five sampled
decodes each; [tuning](../bench/results/2026-10-18-mlx-serve/README.md#tuning)):

- MTP: the model's head with exact verification (the default; the opt-in lossy `--mtp-typical`/`--mtp-tokenv3`
  verifiers change the output distribution). `--mtp-max-depth 2` on the 35B (127.5 tok/s in the short runs, the
  adaptive default 124.8) and 4 on the 27B (37.8 against 36.3).
- DFlash2: the drafter, which mlx-serve quantizes to 8 bits at load on M1-M4, without the MTP head (`--no-mtp`): with
  both loaded, mlx-serve hands a request to MTP when the previous one is still finishing on the server, and the
  benchmark's back-to-back requests sometimes are (2 of 5 timed answers on the 27B, at half the speed). On the 27B a
  tree of up to 7 drafts a round, block 8 (the defaults; block 6 and linear drafts were slower). On the 35B block 4
  (103.9 tok/s, the default block 5 102.4, block 8 88.2) and no tree: mlx-serve verifies trees only on dense models,
  and on a MoE model it switches DFlash off for the rest of an answer once fewer than 1.8 drafts a round are accepted.
  mlx-serve ships no drafter for the 35B and documents DFlash for dense models, but its loader binds this one.
- PLD (prompt lookup) stays on, its default; mlx-serve runs one speculative mode per request (DFlash, else MTP, else
  PLD), so it never ran. A fresh `HOME` per run, 32K context, one sequence.

## Qwen3.5-4B (2026-10-18)

Qwen3.5-4B, the family's small dense model (32 layers: 24 Gated DeltaNet, 8 full attention), at 4-bit on every engine
of the [main comparison](#results), through the same harness, requests and checks. Each engine ran in its fastest
setup for this model: speculative decoding with the model's MTP head wherever the engine has an MTP path (incoai
publishes no DFlash2 drafter for the 4B; with the one third-party DFlash2 drafter oMLX was slower than with its MTP
path, [below](#omlx-with-a-third-party-dflash2-drafter)), oMLX's and MTPLX's settings picked in short runs before the
session, and llama.cpp's MTP at n-max 1, 2, 3 and 6. Configs: the `q35-4b-*` entries in
[bench/harness/configs](../bench/harness/configs); result files:
[bench/results/2026-10-18-4b](../bench/results/2026-10-18-4b/).

Kapitsa was the fastest engine on every metric and used the least memory. It wrote 122 tok/s at T = 0.6, 1.12×
Ollama's MLX runner with MTP (109.0) and 1.17× MTPLX (104.2, against Kapitsa's 122.0 in the same bracket), read long
prompts 1.8–2.2× faster than any other engine, and answered follow-ups in 0.05–0.07 s, against 0.09–0.11 s for the
next fastest (MTPLX). Greedy decoding is the exception to a clear lead: Ollama's MLX runner wrote 123.0 tok/s, 1%
below Kapitsa's 124.5 in the same bracket, and Kapitsa's greedy medians in bracket B were 120.4 and 125.3, so at
greedy the two are within run-to-run variation (note 2). Later that day z-lab's DFlash drafter became Kapitsa's default
for the 4B: 178.7 tok/s at T = 0.6 and 197.7 greedy (bracket C, note 7), 1.6× the next engine here; it holds 0.6 GB
more than the MTP head did ([Kapitsa with z-lab's DFlash drafter](#kapitsa-with-z-labs-dflash-drafter)).

The runs came in two brackets on 2026-10-18, each opened and closed by Kapitsa, one engine at a time under the GPU
lock, in High Power mode on AC power. Compare each row with Kapitsa's row of the same bracket.

- **A**, 11:04–12:05: every configuration, back to back. Bursts from Slack and Chrome flagged the long-prompt phases
  of three runs (oMLX without speculation, MTPLX, llama.cpp without MTP). From 11:42 Slack, Chrome and a video call
  (`cameracaptured`, macOS's camera daemon) used more than 5% of the GPU for sustained stretches, and every run from
  then on was flagged, Kapitsa's closing run included. The A rows are the unflagged runs between 11:04 and 11:40.
- **B**, 13:39–14:12: Kapitsa, then the configurations whose A runs were flagged (oMLX without speculation, MTPLX,
  llama.cpp without MTP and at n-max 6), then Kapitsa again, each engine started after other apps had used less
  than 3% of the GPU for 60 s. Their flagged A runs, in [superseded](../bench/results/2026-10-18-4b/superseded/),
  decoded within 4% of their B runs: oMLX without speculation 85.0 / 83.2 tok/s, MTPLX 102.7 / 104.7, llama.cpp
  64.0 / 63.9, n-max 6 63.9 / 70.7.

| Engine (setting) | Bracket | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** (MTP head)<sup>1</sup> | A | 122.4 (119.3–123.4) | 124.5 (119.0–127.0) | 1,796 | 1,337 | 0.06 | 0.07 | **4.1** | pass |
| **Kapitsa** (MTP head)<sup>1</sup> | B | 122.0 (118.7–124.2) | 120.4 (94.0–123.6) | 1,788 | 1,427 | **0.05** | **0.06** | **4.1** | pass |
| **Kapitsa** (z-lab's DFlash drafter, its default since)<sup>7</sup> | C | **178.7** (173.2–194.9) | **197.7** (179.2–204.3) | **1,954** | **1,595** | 0.06 | **0.06** | 4.7 | pass |
| Ollama, MLX runner + MTP (adaptive depth) | A | 109.0 (99.6–115.6) | 123.0 (116.6–124.9)<sup>2</sup> | 758 | 693 | 0.19 | 0.24 | 12.1 | pass |
| MTPLX (sustained profile, depth 1) | B | 104.2 (99.5–107.0) | 106.3 (105.0–109.0) | 780 | 711 | 0.09<sup>6</sup> | 0.11 | 14.3 | pass |
| oMLX + MTP (Lightning MTP) | A | 102.4 (99.5–108.6) | 104.5 (102.8–109.4) | 803 | 726 | 0.25 | 0.48 | 7.3 | pass |
| oMLX (no speculation) | B | 82.4 (82.0–84.9) | 83.9 (83.0–84.6) | 868 | 784 | 0.26 | 0.36 | 6.1 | pass |
| LM Studio (MLX engine) | A | 81.3 (80.4–81.5)<sup>3</sup> | 85.1 (84.3–85.5)<sup>3</sup> | 801 | 747 | 0.41 | 0.70 | 8.4 | pass |
| mlx-lm | A | 73.1 (71.4–75.0) | 76.4 (74.1–78.6) | 784 | 719 | 0.18 | 0.32 | 7.3 | pass |
| llama.cpp + MTP (n-max 2) | A | 69.6 (68.1–73.0) | 73.4 (70.7–74.3) | 650 | 564 | 0.21 | 0.25 | 11.1 | pass |
| llama.cpp + MTP (n-max 1) | A | 69.3 (68.3–69.8) | 71.8 (68.9–72.0) | 662 | 587 | 0.25 | 0.32 | 11.6 | pass |
| Ollama, GGUF + MTP (shipped depth 1) | A | 68.2 (63.4–70.4) | 72.8 (67.0–73.7) | 712 | 575 | 0.15 | 0.25 | 5.7 + 3.2 mapped<sup>4</sup> | pass |
| llama.cpp + MTP (n-max 3) | A | 65.9 (64.2–71.8) | 77.9 (75.9–78.2) | 687 | 609 | 0.21 | 0.24 | 11.1 | pass |
| llama.cpp | B | 63.4 (62.7–63.5) | 65.3 (64.9–65.5) | 724 | 655 | 0.19 | 0.25 | 9.2 | pass |
| llama.cpp + MTP (n-max 6) | B | 62.3 (56.8–69.2) | 71.9 (70.9–75.2) | 689 | 613<sup>5</sup> | 0.16 | 0.27 | 11.3 | pass |
| Ollama, GGUF (MTP off) | A | 59.9 (46.3–65.3) | 64.5 (64.2–66.0) | 733 | 618 | 0.15 | 0.28 | 5.1 + 2.5 mapped<sup>4</sup> | pass |

1. The Kapitsa rows are each bracket's opening run. Its closing runs were flagged for contention
   ([confirmations](../bench/results/2026-10-18-4b/confirmations/)). A's, at 12:04 during the video call, gave
   115.0 / 111.4 tok/s decode, 1,377 / 1,136 tok/s prefill, 0.06 / 0.06 s follow-ups and 4.1 GB. Its prefill
   dropped the most, probably because the call's camera processing shared the Neural Engine, which runs part of
   Kapitsa's dense prefill: its log shows the Neural Engine busy for 98% and 81% of the 4K and 16K prefills, against
   60% and 45% in the opening run. B's closing run, at 14:10 with Slack busy in its long-prompt phases, gave
   121.4 / 125.3 tok/s, 1,774 / 1,436 tok/s, 0.06 / 0.07 s and 4.1 GB.
2. One of the five greedy runs looped (distinct-4-gram 0.47), which speculative decoding speeds up; without it the
   median is 120.8 tok/s. Ollama's MLX runner drafts up to four tokens a step and adapts how many: in the timed
   answers 2.0 a step on average at greedy and 1.7 at T = 0.6, where Kapitsa drafts one. Deeper drafts pay off when
   they are accepted, which happens more often in greedy decoding, so it gains 13% from T = 0.6 to greedy and Kapitsa
   about 2%; at T = 0.6 Kapitsa is 12% faster (122.4 against 109.0).
3. LM Studio stops at 511 of the 512 requested tokens. Its rates use the 511 tokens it reports.
4. Ollama's GGUF runner maps the weights from disk instead of loading them: the first figure is the server's memory,
   the second the resident mapped weights (the model, and with MTP its draft GGUF).
5. Flagged for contention: Chrome used 9.6% of the GPU on average during this prefill. The only flagged figure in
   the table; A's run of this setting read 603 tok/s here, also flagged.
6. MTPLX's SSD session tier writes only what fits above 10 GiB of free disk. Its B run started with 10.0 GiB free,
   and its log shows that tier skipping its writes ("no disk room"); the benchmark's follow-ups came from its
   session cache in memory, as fast as in its A run with 19 GiB free (0.10 / 0.12 s). LM Studio's MLX engine, which
   keeps follow-up checkpoints on disk only while 10 GiB is free, had 18 GiB.
7. Bracket C, 22:49–23:09, after Kapitsa gained the drafter: Kapitsa's previous build (MTP head) opened and closed it
   at 120.5 and 120.2 tok/s, as in A and B, so conditions matched
   ([Kapitsa with z-lab's DFlash drafter](#kapitsa-with-z-labs-dflash-drafter)). One of the five sampled runs looped
   (distinct-4-gram 0.46); without it the median is 177.2.

Repetition loops (distinct-4-gram under 0.5; flagged, not invalid) came up in one of the five sampled runs of
Kapitsa in bracket B (median without it 121.0), oMLX without speculation (82.6), llama.cpp (63.4) and llama.cpp at
n-max 1, 3 and 6 (69.2, 65.3, 62.9), and in two of mlx-lm's (73.4). Peak memory is the whole run's peak. Kapitsa
holds the language model only (2.2 GB; it does not run this model's vision tower), while oMLX, LM Studio, MTPLX and
Ollama's MLX runner also load the file's 0.6 GB vision tower and Ollama's GGUF runner the tag's vision projector.
Engines that keep earlier conversations for follow-up turns grow through the run, MTPLX's session cache most of all
(14.3 GB at its peak). Footprint after the sampled decode runs, before any long prompt: mlx-lm 3.0 GB, Kapitsa 3.4,
oMLX 3.8 (4.6 with MTP), Ollama's MLX runner 4.9, LM Studio 5.2, llama.cpp 6.1 (7.7–8.2 with MTP), MTPLX 6.2, and
Ollama's GGUF runner 2.9 + 2.5 mapped (3.4 + 3.2 with MTP). Of the engines that hold their weights in their own
memory, mlx-lm is the only one below Kapitsa there; it grows to 7.3 GB with the 16K prompt.

**Weights.** The MLX engines read lmstudio-community's MLX 4-bit conversion
([lmstudio-community/Qwen3.5-4B-MLX-4bit](https://huggingface.co/lmstudio-community/Qwen3.5-4B-MLX-4bit) @c43ee1d,
2.8 GB: 2.2 GB of language model, its embedding tied to the output layer, and a 0.6 GB bf16 vision tower), the file
`kapitsa pull 4b` and LM Studio's staff pick download. MLX conversions of the small Qwen3.5 models leave out the
model's MTP head, so each engine with an MTP path got one: Kapitsa its catalog's (Qwen's own head from
Qwen/Qwen3.5-4B, bf16, 0.2 GB, 8-bit at load), MTPLX, oMLX and Ollama's MLX runner the head of MTPLX's 4B release
([Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed](https://huggingface.co/Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed)
@550153b, 4-bit, 0.08 GB, with its runtime file), read beside the same file through APFS clones, an index view
(`bench/harness/tools/omlx_mtp_farm.py`) and Ollama's zero-copy import. llama.cpp and Ollama's GGUF runner read
Ollama's `qwen3.5:4b-mtp-q4_K_M` tag: the Q4_K_M model (2.5 GB, blob a9d0a3e1) and, as a separate draft GGUF, its
MTP layer with a Q8_0 copy of the embedding (0.76 GB, blob 83ba6dc1).

**Settings.** As for the larger models (see [what was compared](#what-was-compared)), except:

| Engine | Version measured | Speculative decoding on the 4B |
| --- | --- | --- |
| Kapitsa | 68b7aee, Release build (b3e8fe9 in the result files: the same source under an earlier commit id) | its default for this model: the MTP head, one draft token a step; dense prefill split between the GPU and the Neural Engine |
| llama.cpp | 0.7.0, build b11669 (620d2225c) | MTP with the tag's draft GGUF (`-md`), `--spec-draft-n-max` 1, 2, 3 and 6 (6 is unsloth's recommendation for this model); and off |
| Ollama, GGUF | 0.40.6 (bundled llama.cpp 620d2225c; the app had updated itself from 0.40.5) | built-in MTP at the tag's shipped depth, 1; and off |
| Ollama, MLX runner | 0.40.6 (MLX bebd642) | built-in MTP, adaptive depth up to 4 |
| LM Studio | llmster 0.0.25-1, MLX runtime 1.13.1 (its staff pick `qwen/qwen3.5-4b`) | none: its MLX engine has none for this model |
| mlx-lm | 0.32.2 (mlx 0.32.4) | none for this model |
| oMLX | 0.7.1rc1 | Lightning MTP, oMLX's own MTP path (adaptive depth, up to 3); and none |
| MTPLX | 2.12.2 | MTP, profile `sustained` (its runtime file's), depth 1 |

oMLX's and MTPLX's settings come from short runs just before the session (three sampled and three greedy 512-token
answers each; [tuning](../bench/results/2026-10-18-4b/README.md#tuning)). oMLX: Lightning MTP 101.4 / 103.8 tok/s,
DFlash with z-lab's Qwen3.5-4B drafter (DFlash v1, quantized to 4 bits at load) at most 89.9 / 98.0 (block 5, verify
`adaptive`; block 9 and verify `dflash` were slower), no speculation 84.9 / 81.8. MTPLX: depth 1 104.5 / 108.6, depth
2 103.8 / 109.2, and depth 3, its runtime file's default, 87.7 / 102.6.

### oMLX with a third-party DFlash2 drafter

[nchapman/Qwen3.5-4B-DFlash2](https://huggingface.co/nchapman/Qwen3.5-4B-DFlash2) @8d8bb62, a DFlash2 drafter trained
with vLLM's speculators library against a 4-bit GPTQ Qwen3.5-4B, ran on oMLX later on 2026-10-18 (21:42–21:56), in one
hold of the GPU lock with Kapitsa first and last. It has no license: it was used locally for this measurement and is
not in this repository. oMLX reads DFlash2 drafters in incoai's layout; this one's tensors carry the same names, so a
local view rewrote only its config. The fastest setting in short runs (block 5, verify `adaptive`, the drafter
quantized to 4 bits at load; blocks 8 and 16 and verify `dflash` were slower) wrote 87.0 (81.0–90.8) tok/s at
T = 0.6 and 94.6 (91.7–100.3) greedy, read prompts at 875 / 791 tok/s, answered follow-ups in 0.23 / 0.40 s and peaked
at 7.6 GB: slower than oMLX's MTP path (102.4 / 104.5 in bracket A), which the table above keeps. In the same bracket
Kapitsa wrote 129.7 and 125.6 tok/s at T = 0.6 (126.8 and 126.7 greedy), read prompts at 1,811 / 1,438 and 1,805 /
1,455 tok/s and peaked at 4.1 GB. oMLX accepted about 60% of the drafter's drafts and got about 2 tokens out of a pass,
against the 3.8 its card reports under vLLM, so oMLX may not run this drafter as it was trained. Result files:
[nchapman-dflash2](../bench/results/2026-10-18-4b/nchapman-dflash2/).

### Full precision (bf16), a partial bracket

The same protocol on the model's bf16 weights,
[mlx-community/Qwen3.5-4B-bf16](https://huggingface.co/mlx-community/Qwen3.5-4B-bf16) @491fdc7 (8.5 GB: 7.8 GB of
language model and the same 0.6 GB vision tower), the files `kapitsa pull 4b:bf16` downloads, with the same Qwen MTP
head as the 4-bit model. Four engine setups ran on 2026-10-18 between 15:30 and 15:55, one at a time under the GPU
lock, each started after other apps had used less than 3% of the GPU for 60 s; the MLX engines read Kapitsa's files
through APFS clones. **The bracket is incomplete:** the disk then ran short, so there is no closing Kapitsa run, and
mlx-lm, LM Studio, llama.cpp and Ollama were not run at bf16 (LM Studio needs 10 GiB of free disk for its follow-up
checkpoints, and llama.cpp and Ollama need Ollama's 9.2 GB BF16 GGUF). Result files:
[bench/results/2026-10-18-4b/bf16](../bench/results/2026-10-18-4b/bf16/).

**Kapitsa was not the fastest writer at bf16 in this bracket** (later that day it was, with the same z-lab drafter:
89.1 tok/s, 1.5× oMLX + DFlash; table, note 3). oMLX with z-lab's DFlash drafter wrote 59.4 tok/s at T = 0.6 and
MTPLX at depth 3 wrote 54.2, against Kapitsa's 44.3: 1.34× and 1.23× faster (greedy: 61.5 and 58.8 against 45.1).
Our best explanation: at bf16 a forward pass reads 7.8 GB of weights, so the rate follows how many tokens a pass
yields. oMLX's DFlash drafts up to four tokens a pass (2.5 on average) and got 2.8 tokens out of each pass, MTPLX
drafts three, and Kapitsa's MTP head one, so a Kapitsa pass yields at most two (depth 1 is its default for this model,
and it has no drafter for the 4B). At 4-bit, where a pass reads about a quarter of the bytes, Kapitsa's depth 1 was
the fastest setup measured. Kapitsa still read long prompts 1.7–1.9× faster than the next fastest engine here,
answered follow-ups fastest and used the least memory.

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** (z-lab's DFlash drafter, its default since)<sup>3</sup> | **89.1** (80.8–97.6) | **87.6** (85.2–88.4) | **1,893** | **1,556** | **0.06** | **0.07** | 11.0 | pass |
| oMLX + DFlash (z-lab drafter, block 5) | 59.4 (54.3–63.0) | 61.5 (59.2–69.1) | 859 | 823 | 0.30 | 0.42 | 12.6 | pass |
| MTPLX (sustained profile, depth 3) | 54.2 (49.7–55.7) | 58.8 (57.4–61.6) | 820 | 751 | 0.11 | 0.15 | 20.3<sup>2</sup> | pass |
| **Kapitsa** (MTP head) | 44.3 (42.6–47.3) | 45.1 (44.0–45.6) | 1,737 | 1,401 | **0.06** | **0.07** | **9.7** | pass |
| oMLX (no speculation) | 27.6 (27.0–27.8) | 26.3 (25.9–27.1)<sup>1</sup> | 907 | 812 | 0.27 | 0.41 | 11.8 | pass |

No run of this bracket was flagged for contention. Kapitsa's run (15:30) came first; there is no closing run to check
it against. One greedy run of Kapitsa, of oMLX + DFlash and of MTPLX looped (distinct-4-gram under 0.5; flagged, not
invalid).

1. Three of oMLX's five greedy runs without speculation looped, and macOS swapped 12,062 pages (about 190 MB) during
   that phase; its sampled median, 27.6 tok/s, is the cleaner figure. Reading 7.8 GB a pass at the M4 Pro's 273 GB/s
   bounds decoding without speculation at about 33 tok/s.
2. MTPLX's peak includes its session cache, which keeps earlier conversations for follow-up turns.
3. Measured later the same day (22:56), in the hold described under
   [Kapitsa with z-lab's DFlash drafter](#kapitsa-with-z-labs-dflash-drafter), where the previous build opened and
   closed at 44.1 and 45.9 tok/s (this bracket's 44.3).

**Settings.** As at 4-bit, except where short runs before the bracket measured faster (three sampled and three greedy
answers each; [tuning](../bench/results/2026-10-18-4b/bf16/tuning/)). oMLX: DFlash with z-lab's Qwen3.5-4B drafter
([z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) @9a1996c, DFlash v1, quantized to 4 bits
at load), block 5, verify `adaptive`, 59.1 / 63.2 tok/s, against block 9 57.0 / 56.8 and Lightning MTP with the bf16
head 49.5 / 48.2 (at 4-bit Lightning MTP had been the faster). MTPLX: depth 3, 54.5 / 59.0, against depth 2 52.4 /
54.9 and depth 1 43.7 / 41.0, with Qwen's bf16 head (Kapitsa's file) and the runtime file of MTPLX's 4B release.
Kapitsa: its defaults, the bf16 weights with the MTP head (`kapitsa pull 4b:bf16`).

### Kapitsa with z-lab's DFlash drafter

[z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) @9a1996c (Apache-2.0), the DFlash v1
drafter oMLX ran above, now runs on Kapitsa's DFlash2 path: six Qwen3 layers over the target's captured features with
the geometry of the DFlash2 drafters, without their block convolutions and candidate selector. Kapitsa quantizes it to
4 bits at load and drafts 7 tokens a pass (the drafter was trained with blocks of 16); `kapitsa pull 4b` fetches it as
a companion of every 4B variant, and without it (`--dflash none`, or a memory budget that leaves it out) the MTP head
drafts 4 tokens a pass. One hold of the GPU lock on 2026-10-18, 22:49–23:09, same Mac, High Power mode, AC power:
Kapitsa's previous build (ecf8394, the MTP head at depth 1) first and last as the control, the build with the drafter
between, its defaults and its MTP head alone; result files in
[bench/results/2026-10-18-4b/dflash](../bench/results/2026-10-18-4b/dflash/).

| Kapitsa (setting) | Weights | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| previous build, MTP head (opening) | 4-bit | 120.5 (117.6–120.9) | 124.2 (122.0–125.1) | 1,783 | 1,429<sup>1</sup> | 0.05 | 0.05 | 4.1 | pass |
| **DFlash drafter** (default) | 4-bit | **178.7** (173.2–194.9)<sup>2</sup> | **197.7** (179.2–204.3) | **1,954** | **1,595** | 0.06 | 0.06 | 4.7 | pass |
| MTP head, depth 4 (`--dflash none`) | 4-bit | 159.0 (152.8–169.0) | 168.6 (163.5–173.1) | 1,800 | 1,461 | 0.06 | 0.06 | 4.0 | pass |
| previous build, MTP head (closing) | 4-bit | 120.2 (119.4–125.0) | 125.5 (123.3–128.1) | 1,805 | 1,452 | 0.05 | 0.07 | 4.1 | pass |
| previous build, MTP head (opening) | bf16 | 44.1 (43.3–45.9)<sup>1</sup> | 45.8 (45.5–46.1) | 1,770 | 1,429 | 0.06 | 0.07 | 9.7 | pass |
| **DFlash drafter** (default) | bf16 | **89.1** (80.8–97.6) | **87.6** (85.2–88.4) | **1,893** | **1,556** | 0.06 | 0.07 | 11.0 | pass |
| MTP head, depth 4 (`--dflash none`) | bf16 | 71.0 (69.0–74.4) | 74.5 (73.5–79.9) | 1,768 | 1,432 | 0.06 | 0.08 | 9.7 | pass |
| previous build, MTP head (closing) | bf16 | 45.9 (45.2–46.6) | 45.4 (45.1–46.1) | 1,765 | 1,432 | 0.06 | 0.07 | 9.7 | pass |

The control's openings and closings agree within 4% and match its runs in brackets A and B (122.4 / 122.0) and in the
bf16 bracket (44.3). With the drafter Kapitsa writes 1.48× its previous rate at 4-bit and 2.0× at bf16, where a pass
reads 7.8 GB of weights and the drafter turns each pass into about 3 tokens. Prompts read faster too (1.07-1.10×):
the drafter's prompt pass covers the last 2047 positions, where the MTP head's ran over every prompt position. The
drafter costs 0.6 GB at 4-bit and 1.3 GB at bf16 (its 4-bit weights, a draft head of 98,304 output-layer rows, 8-bit
at bf16, and its context rings).

1. Flagged for contention: Slack used the GPU during the opening run's 16K prefill (mean 4.0%, 4-bit) and decode
   (2.7%, bf16).
2. One of the five sampled runs looped (distinct-4-gram 0.46); without it the median is 177.2. Loops (flagged, not
   invalid) also came up in two of the opening bf16 run's five sampled answers and one sampled and one greedy answer
   of its closing run.

## A real coding-agent session (2026-10-18)

The tables above time one thing at a time: a 512-token answer, one long prompt, a short follow-up. A coding agent
needs all three at once, request after request, so this section replays a real agent session on every engine.

**The session.** [opencode](https://opencode.ai) 2.0.28 (open source) fixing a small Python project
([bench/harness/agent-task](../bench/harness/agent-task/README.md): a CSV ledger library whose 35 tests fail in 15
places from five bugs in four files), recorded on Kapitsa serving Qwen3.6-35B-A3B at 4-bit, greedy, with thinking off.
opencode needed 19 requests to fix all five bugs: it ran the tests, read the four modules, the four test files and the
sample ledger, made nine edits, ran the tests twice more and summed up. Its system prompt and tool definitions come to
5,485 tokens; every request sends the whole conversation again, which grows to 22,779 tokens with the files and the
test output; the answers total 3,577 tokens, 18 of the 19 ending in tool calls.

**The replay.** Every engine gets the same 19 requests in the recorded order, with the recorded messages: the same
system prompt and tools, the same files and test output, and the recorded answers in the history, never the engine's
own. Each request is streamed, greedy (temperature 0, top_k 1), with thinking off, and capped at the recorded
answer's length. The tool definitions are rendered into the system prompt the way Qwen's chat template renders them,
so every engine reads the same tokens (each request's length is checked against the template's count) and streams
every token it writes. The agent's own pauses between requests (0.7 s in all) are replayed but not counted. Details:
[the harness README](../bench/harness/README.md#agent-session).

One bracket per model on 2026-10-18, each under its own hold of the GPU lock with Kapitsa (main at 113bfcd) first and
last and the six other setups in between, each with the configs and settings of the tables above (32K context, one
conversation): the 35B between 16:19 and 16:37, where no run was flagged for GPU contention, the 27B below. No run
failed a check. Result files: [bench/results/2026-10-18-agent-session](../bench/results/2026-10-18-agent-session/).

**Qwen3.6-35B-A3B**

| Engine (setting) | Total time (s) | Prefill (s) | Decode (s) | Writing (tok/s) | First TTFT (s) | Follow-up TTFT, median / p90 (s) | Tokens written | Answers as recorded | Peak memory (GB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Kapitsa** (DFlash2 drafter)<sup>1</sup> | **48.6** | **22.5** | **26.2** | **141.2** | **5.21** | **0.08** / **4.87** | 3,577 | 19 of 19 | **21.1** |
| mlx-serve + MTP (depth up to 2) | 61.6 | 32.1 | 29.5 | 109.8 | 5.87 | 0.69 / 5.64 | 3,254 | 7 of 19 | 22.3 |
| oMLX + DFlash2 (block 5, adaptive verify) | 66.4 | 34.3 | 32.1 | 107.2 | 5.41 | 0.82 / 5.67 | 3,431 | 10 of 19 | 23.7 |
| mlx-serve + DFlash2 (block 4) | 72.2 | 32.3 | 39.9 | 84.6 | 5.91 | 0.70 / 5.71 | 3,380 | 7 of 19 | 22.4 |
| Ollama, MLX runner + MTP (adaptive depth) | 72.8 | 33.1<sup>2</sup> | 39.7 | 93.8 | 5.97 | 0.71 / 6.06<sup>2</sup> | 3,475 | 6 of 19 | 32.2 |
| LM Studio (MLX engine) | 85.2 | 34.4 | 50.8 | 68.2 | 6.21 | 0.85 / 5.86 | 3,481 | 9 of 19 | 25.0 |
| MTPLX (sustained profile, depth 1) | 138.8 | 38.5<sup>2</sup> | 100.4<sup>3</sup> | 71.5 | 6.11 | 0.70 / 6.12<sup>2</sup> | 3,393 | —<sup>2</sup> | 29.3 |
| llama.cpp + MTP | —<sup>4</sup> | — | — | — | — | — | — | — | — |

Total time is the sum of the 19 requests' times, from sending a request to the end of its stream. Prefill is the sum
of the times to the first token (reading what the engine did not reuse from its cache, plus that token), decode the
rest. Writing is all answers' tokens after their first streamed event over the time from the first to the last event
(to the end of the stream for an answer whose end was never streamed). The first request is a cold 5,485-token
prompt; the follow-up TTFT covers the other 18. Tokens written are out of the
3,577 recorded; an answer counts as recorded when its text and tool calls are the recorded ones. Peak memory is the
server's whole process tree.

1. Kapitsa ran before and after the other engines; the table shows the run after. The run before: 48.7 s, 22.5 / 26.2
   s, 140.9 tok/s, 5.25 s, 0.08 / 4.86 s, 21.1 GB.
2. Ollama returns a tool call only once it is complete, even when the request has no tools (its Qwen parser always
   runs), and MTPLX, asked without tools, replaces a tool call with a notice that it sends at the end. On those turns
   (Ollama 3, MTPLX 2) the first event brings the whole answer, so their time to first token is left out of the
   follow-up figures and their prefill time is estimated from the session's writing rate. MTPLX's answers cannot be
   compared with the recorded ones, since the notice replaces their tool calls.
3. MTPLX keeps each stream open after the last token while it saves the conversation to its session cache (inline
   post-commit, the setting that lets its follow-ups hit that cache): 57.8 s of its 100.4 s.
4. Not run: llama.cpp reads Ollama's Q4_K_M GGUF files (20.2 GB and 15.7 GB), which are no longer on this Mac, and
   its free disk (6 to 13 GB that day) had no room for them.

**What the recording engine gains.** Kapitsa recorded the session, so its greedy answers are the recorded ones and it
finds each in its cache on the next request. The other engines run the same weights with other arithmetic, so their
greedy answers differ wherever two tokens are nearly tied (on 9 to 13 of the 19 turns); the next request then carries
the recorded answer, and they restore their cache to the end of the previous prompt and read it: 1.7K to 2.4K tokens
over the session, about 2 s at their prefill rates (MTPLX 3.3K, 3.6 s). On the requests that followed an answer
matching the recording, where nothing had to be restored, their median time to first token was 0.19 to 0.67 s
(mlx-serve + DFlash2 0.19, mlx-serve + MTP 0.40, LM Studio 0.41, Ollama 0.53, oMLX 0.67) against Kapitsa's 0.08 to
0.18 s on the same requests. Kapitsa's lead in total time, 13 s over mlx-serve + MTP, does not come from this.

**Qwen3.8-27B**

The 27B replays the same session, the 35B's answers included. Recorded the same way on the 27B, opencode never got to
an edit: from its 14th request on it ran the same shell command again and again (greedy decoding can lock a model into
such a loop) until the recording proxy stopped it at 80 requests. With the 35B's answers in the history, every
engine's own 27B answers differ from the recorded ones at almost every turn, Kapitsa's included, so on each request
every engine restored its cache to the end of the previous prompt (within 31 tokens, where an engine reports it) and
read the recorded answer: the same work for every engine, and no engine has the recording's edge. The prompt lengths
are the same with the 27B's tokenizer and template. One session on 2026-10-18 between 20:51 and 21:42 under its own
hold of the GPU lock, Kapitsa first and last.

| Engine (setting) | Total time (s) | Prefill (s) | Decode (s) | Writing (tok/s) | First TTFT (s) | Follow-up TTFT, median / p90 (s) | Tokens written | Peak memory (GB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Kapitsa** (DFlash2 drafter)<sup>1</sup> | **150.2** | **95.2** | **55.0** | **59.7** | **15.05** | **2.74** / **16.10** | 3,180 | **19.9** |
| oMLX + DFlash2 (block 5, verify dflash) | 278.1 | 202.5 | 75.6 | 42.6 | 42.11 | 3.56 / 37.31 | 3,160 | 28.0 |
| mlx-serve + MTP (depth up to 4)<sup>2</sup> | 298.7 | 202.0 | 96.6 | 32.9 | 42.98 | 3.46 / 38.24 | 3,191 | 21.6 |
| Ollama, MLX runner + MTP (adaptive depth)<sup>2, 3</sup> | 302.9 | 198.5 | 104.4 | 32.3 | —<sup>3</sup> | 3.26 / 38.24 | 3,176 | 30.4 |
| mlx-serve + DFlash2 (draft tree, block 8) | 318.7 | 221.1 | 97.6 | 32.3 | 43.24 | 3.50 / 38.35 | 3,168 | 22.8 |
| MTPLX (turbo profile, depth 3)<sup>2, 3</sup> | 340.7 | 201.3 | 139.4<sup>4</sup> | 28.4 | —<sup>3</sup> | 3.17 / 38.06 | 3,221 | 30.8 |
| LM Studio (MLX engine)<sup>5</sup> | 450.0 | 222.5 | 227.5 | 13.9 | 43.00 | 4.91 / 39.91 | 3,191 | 26.9 |
| llama.cpp + MTP | —<sup>6</sup> | — | — | — | — | — | — | — |

The columns are those of the 35B table; the tokens written are out of the 3,577 of the 35B's answers, which cap each
27B answer. Answers that match the recorded ones: 0 to 3 of 19 on every engine.

1. Kapitsa ran before and after the other engines; the table shows the run after. The run before: 149.4 s, 93.9 /
   55.5 s, 59.1 tok/s, 15.21 s, 2.73 / 15.82 s, 19.9 GB (flagged: Slack used the GPU, 4.9% on average). An earlier,
   unflagged run that day (16:43, in a bracket the Mac's sleep cut short) gave 151.5 s
   ([confirmations](../bench/results/2026-10-18-agent-session/confirmations/)).
2. Flagged for GPU contention: Chrome and Slack used more than 5% of the GPU in short bursts (on average 0.9% during
   mlx-serve + MTP, 2.4% during Ollama, 1.3% during MTPLX). Not re-run: an unflagged run of Ollama's that day gave
   304.8 s (confirmations), and Kapitsa's flagged and unflagged runs differ by 1 s.
3. Ollama and MTPLX hold an answer that begins with a tool call (as on the 35B): their first answer did, so it has no
   time to first token, and the follow-up figures leave out 2 such turns each. Ollama also drops a tool call that the
   recorded length cuts short without streaming any of it (5 turns): their writing time runs to the end of the
   stream.
4. Of which 26.9 s after the last token: MTPLX's inline session commit.
5. LM Studio keeps the checkpoints it restores from on disk, only while 10 GiB of disk are free; this run had 11.6
   GiB. In a first attempt that day it had less, kept no checkpoints, and re-read the whole conversation on every
   request (66 to 104 s per request).
6. Not run, as on the 35B.

Run index (local time, UTC+5; "Started" is when the session began, after start-up, warm-up and the quiet-GPU wait):

| Result file | Started | Foreign GPU, mean / max (%) | Ambient GPU, mean (%) | Free disk at start (GiB) | Flagged for contention |
| --- | --- | ---: | ---: | ---: | --- |
| `q36-35b-kapitsa-20261018-161859.json` | 16:19 | 0.3 / 3.5 | 0.0 | 11.4 | — |
| `q36-35b-ollama-mlx-q4-20261018-162027.json` | 16:21 | 0.0 / 0.0 | 5.7 | 11.4 | — |
| `q36-35b-lmstudio-20261018-162321.json` | 16:24 | 0.0 / 0.0 | 3.3 | 11.4 | — |
| `q36-35b-mtplx-20261018-162554.json` | 16:26 | 0.1 / 1.4 | 1.3 | 10.3 | — |
| `q36-35b-omlx-best-20261018-162930.json` | 16:30 | 0.0 / 0.0 | 3.7 | 10.0 | — |
| `q36-35b-mlxserve-mtp-20261018-163132.json` | 16:32 | 0.1 / 1.2 | 0.0 | 9.9 | — |
| `q36-35b-mlxserve-dflash2-20261018-163326.json` | 16:34 | 0.1 / 1.1 | 0.0 | 9.8 | — |
| `q36-35b-kapitsa-20261018-163530.json` | 16:36 | 0.3 / 3.0 | 0.0 | 9.7 | — |
| `q38-27b-kapitsa-20261018-205048.json` | 20:51 | 4.9 / 29.7 | 1.6 | 10.1 | Slack |
| `q38-27b-ollama-mlx-q4-20261018-205419.json` | 20:56 | 2.4 / 65.2 | 0.3 | 13.1 | Slack, Chrome |
| `q38-27b-lmstudio-20261018-210117.json` | 21:02 | 0.1 / 7.0 | 9.9 | 11.6 | — |
| `q38-27b-mtplx-20261018-211005.json` | 21:12 | 1.3 / 30.5 | 7.4 | 11.5 | Chrome |
| `q38-27b-omlx-best-20261018-211808.json` | 21:20 | 0.1 / 3.0 | 13.2 | 10.2 | — |
| `q38-27b-mlxserve-mtp-20261018-212503.json` | 21:26 | 0.9 / 37.1 | 8.3 | 10.9 | Chrome, Slack |
| `q38-27b-mlxserve-dflash2-20261018-213146.json` | 21:33 | 0.4 / 10.7 | 10.5 | 11.9 | — |
| `q38-27b-kapitsa-20261018-213837.json` | 21:39 | 0.3 / 2.0 | 16.1 | 11.8 | — |

## The M4 Pro table filled in (2026-10-19)

Runs for the M4 Pro table's empty and flagged cells, on the same MacBook Pro (M4 Pro, 48 GB, macOS 27.0.1, High Power
mode, AC power) with this harness: one hold of the GPU lock per model, Kapitsa (main at 474b3e6, and 0.1.35 for the
last holds) first and last as the control. Result files: [bench/results/2026-10-19-m4pro](../bench/results/2026-10-19-m4pro/).

### Qwen: mlx-serve on the 4B, the MLX column on the 35B

One hold of the GPU lock (06:16–06:31), each model between two Kapitsa runs. mlx-serve wrote 137.0 tok/s on
Qwen3.5-4B and mlx-vlm 86.0 on Qwen3.6-35B-A3B, both at 4-bit, at T = 0.6; nothing in their runs was flagged.

| Engine (setting) | Model | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Kapitsa (DFlash drafter), before<sup>1</sup> | 4B | 183.4 (167.1–193.3) | 189.9 (186.8–196.8) | 1,937 | 1,595 | 0.05 | 0.06 | 4.7 | pass |
| mlx-serve + MTP (max depth 3) | 4B | 137.0 (132.7–141.0) | 147.5 (136.4–151.5) | 874 | 797 | 0.22 | 0.28 | 8.4 | pass |
| Kapitsa (DFlash drafter), after | 4B | 169.2 (164.2–186.1) | 194.0 (173.3–202.5) | 1,930 | 1,543 | 0.05 | 0.07 | 4.7 | pass |
| Kapitsa (DFlash2 drafter), before | 35B | 159.1 (150.8–165.6) | 176.2 (160.9–181.7) | 1,164 | 986 | 0.06 | 0.08 | 21.2 | pass |
| mlx-vlm + DFlash2 | 35B | 86.0 (81.0–87.2) | 90.7 (89.2–93.8) | 985 | 879 | 0.36 | 2.81<sup>2</sup> | 29.4 | pass |
| Kapitsa (DFlash2 drafter), after | 35B | 159.6 (155.0–180.1) | 176.5 (170.2–181.0) | 1,168 | 987 | 0.06 | 0.07 | 21.2 | pass |

1. Flagged for contention: Chrome used the GPU during its runs (up to 14%). The table keeps Kapitsa's 179 and 154; its
   runs here gave 169–183 and 159–160.
2. mlx-vlm restores this hybrid model's state at 2,048-token checkpoints, so two of the three 16K follow-ups re-read
   about 2,000 tokens.

- **mlx-serve 26.10.2 on the 4B.** The vendor publishes no 4B pack, so mlx-serve read the 4B trunk every engine reads
  with mlx-community/Qwen3.5-4B-MTP-4bit, Qwen's own MTP head at 4 bits, as its MTP sidecar
  (`tools/mlx_serve_pack.py qwen3.5-4b`). MTP with exact verification at max depth 3, the fastest in short runs before
  the hold (three sampled decodes each, T = 0.6): depth 3 136.0 tok/s, 4 134.1, 2 130.5, 1 120.3, the adaptive default
  124.0. DFlash with z-lab's 4B drafter, a DFlash v1 drafter that mlx-serve loads, was slower: 124.5 at block 4,
  118.5 at its default 5, 111.2 at 6.
- **The MLX column on the 35B is mlx-vlm** 0.7.8 with incoai's DFlash2 drafter (the one Kapitsa and oMLX read), its
  fastest setup in short runs: DFlash2 85.2 tok/s, at block 8 81.6, without speculation 81.7. Its MTP path does not
  load the MTP head of MTPLX's 35B release (experts stored one by one). mlx-lm has no speculative path for these
  models. Its runs also replayed [the opencode session](#a-real-coding-agent-session-2026-10-18): 79.3 s, against
  Kapitsa's 48.2 s in the same hold.
- **Not filled this time:** the 27B's MLX cell (in short runs mlx-vlm with MTP at block 3 wrote 27.2 tok/s, DFlash2
  24.6), a clean mlx-serve re-run on the 27B (its ⚠ stays), and the 4B's MLX cell, which keeps mlx-lm's 73 (mlx-vlm with
  MTP wrote 86.5 there in short runs). The short runs are in
  [tuning](../bench/results/2026-10-19-m4pro/tuning/).

### Qwen with Kapitsa 0.1.35: the MLX column on the 4B and 27B, mlx-serve on the 27B

One hold per model (06:40–06:49 the 4B, 06:49–07:14 the 27B), Kapitsa 0.1.35 (main at a47341b) first and last, the
agent session left out. The MLX cells are mlx-vlm's: 88.1 tok/s on Qwen3.5-4B and 27.9 on Qwen3.8-27B at T = 0.6, both
with MTP. mlx-serve's clean 27B run with DFlash2 wrote 54.2, which replaces its flagged 51.9 of 2026-10-18.

| Engine (setting) | Model | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Kapitsa (DFlash drafter), before | 4B | 176.1 (160.1–189.8) | 197.9 (189.9–204.1) | 1,940 | 1,592 | 0.05 | 0.06 | 4.7 | pass |
| mlx-vlm + MTP (block 3)<sup>1</sup> | 4B | 88.1 (86.0–90.3) | 105.4 (101.0–108.8) | 843 | 766 | 0.31 | 3.11 | 10.5 | pass |
| Rapid-MLX + MTP (3 drafts) | 4B | 98.1 (93.0–101.8) | 111.6 (106.8–114.8) | 814 | 767 | 0.25 | 0.33 | 7.9 | pass |
| Kapitsa (DFlash drafter), after | 4B | 177.9 (168.0–179.6) | 179.7 (170.1–186.1) | 1,819 | 1,586 | 0.06 | 0.06 | 4.7 | pass |
| Kapitsa (DFlash2 drafter), before<sup>2</sup> | 27B | 56.2 (51.2–66.2) | 59.4 (59.2–61.9) | 336 | 283 | 0.23 | 0.26 | 19.4 | pass |
| mlx-vlm + MTP (block 3) | 27B | 27.9 (27.6–28.4) | 30.5 (30.5–30.5) | 131 | 125 | 0.71 | 1.18 | 26.6 | pass |
| mlx-serve + DFlash2 (draft tree, block 8) | 27B | 54.2 (48.6–57.2) | 57.0 (55.8–63.8) | 129 | 122 | 0.29 | 0.40 | 21.5 | pass |
| Kapitsa (DFlash2 drafter), after | 27B | 64.8 (62.0–66.6) | 64.8 (63.1–65.7) | 396 | 350 | 0.22 | 0.26 | 19.4 | pass |

1. Flagged for contention in its 16K prefill (Chrome and Slack, up to 17% of the GPU); its decode phases were clean.
   The table marks the cell.
2. Flagged for contention in both prefills (Chrome, up to 55%), which also shows in its prefill rates. Kapitsa's closing
   run, clean, gave 64.8 against the table's 63; on the 4B its runs gave 176.1 and 177.9 against 179.

- **The MLX column.** mlx-vlm 0.7.8 with MTP is faster than mlx-lm on both: mlx-lm has no speculative path for these
  models and wrote 73.1 tok/s on the 4B on 2026-10-18. The 4B's drafter is mlx-community/Qwen3.5-4B-MTP-4bit (Qwen's
  head at 4 bits), the 27B's the head of MTPLX's 27B release in the model folder; block 3 was the fastest of 2 to 5 in
  the short runs above.
- **mlx-serve on the 27B.** DFlash2 as on 2026-10-18 (the vendor's drafter, a tree of up to 7 drafts, `--no-mtp`). Its
  MTP setup, 38.0 tok/s ⚠ on 2026-10-18, was not re-run, to finish sooner.
- **Rapid-MLX** 0.16.3 (`uv pip install rapid-mlx`), on the 4B only: MTP with the head its catalog pairs with the model
  (mlx-community/Qwen3.5-4B-MTP-4bit), 3 drafts, the fastest in short runs (2 drafts 100.7, without speculation 80.4
  tok/s). Its 35B, 27B and Bolt runs were not made. It loads Bolt through the MLX bundle's model file, on its multimodal
  path.

### Nemotron 3.5 Bolt

oMLX again (no drafter), with no run flagged: 34.6 (33.6–34.7) tok/s at T = 0.6 and 34.1 greedy, prefill 298 / 307
tok/s at 4K / 16K, follow-ups 0.73 / 0.83 s, peak 30.4 GB. Its 2026-10-18 run, whose 16K prefill was flagged, gave
33.9 / 34.1, 296 / 300, 0.72 / 0.72 s and 30.3 GB. Kapitsa before and after it: 66.4 and 69.1 tok/s at T = 0.6 (69.9
and 68.4 greedy), 502–519 / 492–513 tok/s prefill, 0.23–0.26 / 0.27–0.29 s follow-ups, 22.7–22.8 GB (the table keeps
its 2026-10-18 figure, 63.9).

oMLX reads Bolt through the layout of NVIDIA's Bolt bundle for MLX, whose config.json names the bundle's model file
(`model_file: nemotron_bolt.py`, model type `nemotron_h_nano_omni`). The folder `kapitsa pull bolt` writes holds
NVIDIA's own config.json (model type `nemotron_h_omni`), which oMLX 0.7.1rc1 refuses (`Model type nemotron_h_omni not
supported`, [failed](../bench/results/2026-10-19-m4pro/failed/)); this run read NVIDIA's files with the bundle's
config.json, the layout of the 2026-10-18 runs and of the [harness README](../bench/harness/README.md).

The engines whose Bolt cell says "no support" (checked 2026-10-19):

- **Ollama** 0.40.6, its latest release (0.40.7-rc0 changes nothing for Bolt): its library page still says Apple
  Silicon / Metal is not supported for `nemotron-3.5-bolt`.
- **LM Studio** (llmster 0.0.25-1): no newer runtime on its stable or beta channel than MLX 1.13.1 and llama.cpp 2.60.0
  (llama.cpp b11698). The MLX runtime refused NVIDIA's checkpoint on 2026-10-18 (`No LM Runtime found for model format
  'torchSafetensors'`); the llama.cpp runtime's Metal library has MXFP4 kernels but none for NVFP4, the format of
  Bolt's Mamba projections in NVIDIA's GGUF, so those layers would run on the CPU, as in llama.cpp. LM Studio's catalog
  has no Bolt entry.
- **mlx-serve** 26.10.2, its latest release: it reads the config, loads the checkpoint's 28,885 tensors and stops at
  `MISSING WEIGHT: model.embed_tokens.weight` (Bolt's weights sit under `language_model.backbone`, in IQ2_XXS, Q8_0 and
  NVFP4 blocks, none of mlx-serve's quantization modes), with NVIDIA's config.json and with the bundle's
  ([logs](../bench/results/2026-10-19-m4pro/support-checks/)).

### Nemotron 3.5 Lightning

NVIDIA's NVFP4 checkpoint with its DSpark drafter, the folder `kapitsa pull lightning` writes, with the setups of the
[M5 Max bracket](#nemotron-35-lightning-30b-a3b-m5-max-128-gb-2026-10-18). One hold, 07:47–07:55, Kapitsa 0.1.35
first and last; mlx-vlm 0.7.8 read the same files with the same DSpark drafter (block 6). Nothing was flagged.

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K (tok/s) | Prefill 16K (tok/s) | Cached follow-up 4K (s) | Cached follow-up 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Kapitsa (DSpark drafter), before | 105.8 (102.4–110.2) | 116.0 (105.4–117.8) | 945 | 864 | 0.15 | 0.17 | 18.6 + 0.7 mapped | pass |
| mlx-vlm + DSpark (block 6) | 65.0 (63.6–67.0) | 67.7 (66.9–71.1) | 721 | 658 | 0.30 | 0.33 | 28.8 | pass |
| Kapitsa (DSpark drafter), after | 106.0 (93.1–117.0) | 115.2 (103.2–126.2) | 944 | 859 | 0.15 | 0.15 | 18.6 + 0.7 mapped | pass |

The MLX cell is mlx-vlm's: mlx-lm cannot read NVIDIA's checkpoint and runs an MLX 4-bit conversion, which was not
measured here. oMLX and llama.cpp were not measured either. This disk holds Kapitsa's copy (21.4 GiB) or one
competitor's MLX 4-bit or Q4_0 files (17–19 GiB), not both, so their runs could not have Kapitsa in the same hold, and
they were stopped to finish sooner.

## Nemotron Dory1.0 4B (2026-10-19)

NVIDIA's looped 4B (its 40-sublayer shared block runs 2 times per token by default) on the M4 Pro (48 GB, macOS 27.0.1).
The README's cells come from one hold of the GPU lock: Kapitsa (main at 18a03e1) first and last, the harness's T = 0.6
coding answer (its decode phase; llama.cpp 2 runs of 256 tokens, the others 5 of 512), with
[configs/dory.json](../bench/harness/configs/dory.json). Result files:
[bench/results/2026-10-19-dory](../bench/results/2026-10-19-dory/).

| Engine (setting) | Weights | T = 0.6 (tok/s) |
| --- | --- | ---: |
| **Kapitsa** (per-loop KV, `--loop-kv per`: speculative decoding), first / last | NVIDIA's NVFP4 (8-bit LM head) | 62.9&nbsp;⚠ / 64.1 |
| Kapitsa (last-loop KV, its default) | NVIDIA's NVFP4 (8-bit LM head) | 59.6 |
| llama.cpp: NVIDIA's Dory fork (`brianchmielNV/llama.cpp` at 98a74c97, the model card's build), last-loop KV, `-fa on` | NVIDIA's Q5_K_M GGUF | 5.7&nbsp;⚠ |
| mlx-lm with its Dory pull request (`ml-explore/mlx-lm#1988` at 208065f), per-loop KV (its default) | snsnc's MLX 4-bit (affine, groups of 64, from NVIDIA's BF16) | 48.3 |

All at 2 loops; ⚠: another app used just over the harness's 5% threshold of the GPU during that run (mean 5.1%).
The README shows Kapitsa's clean last run and llama.cpp's run in this hold. Earlier the same day, in their own holds
and unflagged, the llama.cpp fork wrote 8.3 tok/s at T = 0.6 (6.4 greedy) and, with the same GGUF: prefill
430 / 389 tok/s at 4K / 16K, cached follow-ups 0.60 / 0.56 s, 11.1 GB peak (`-c 32768`), tool calls pass; with
per-loop KV 6.1 tok/s and 15.6 GB, with its n-gram speculation (`--spec-type ngram-mod`) 6.1 tok/s at T = 0.6; at 1 and 4
loops 9.3 and 3.4 tok/s. Its graph runs dozens of small operations per sublayer, so a token's 112 sublayer passes cost
thousands of dispatches, and its decoding stays near 8 tok/s whatever the KV mode or attention setting
(`-fa on` / `off`, `-ub` 512-2048: 5.4-6.1 tok/s after a 4K prompt).

The engines whose Dory cell says "no support" (checked 2026-10-19): Ollama 0.40.6 (its llama.cpp library has no `dory`
architecture), LM Studio (llama.cpp runtime 2.60.0: no `dory` architecture; MLX runtime with mlx-lm 0.31.4: no Dory
model), oMLX, mlx-serve and MTPLX (no Dory model in their code). The MLX column is mlx-lm with its unmerged Dory pull
request; released mlx-lm and mlx-vlm have no Dory model.

## MacBook Pro M1 Pro, 16 GB (2026-10-19)

The 16 GB Mac: a MacBook Pro (14-inch, 2021), `MacBookPro18,3`, M1 Pro with an 8-core CPU and a 14-core GPU, 16 GB,
macOS 27.0.1, on AC power (Automatic power mode; this model has no High Power mode). Of the catalog, Qwen3.5-4B,
Qwen3.5-9B, Nemotron Dory1.0 4B, Nemotron 3 Nano 4B and Nemotron Nano 9B v2 fit it: `kapitsa doctor` gives Kapitsa a
12.2 GB GPU budget (Metal's recommended working set is 12.7 GB), and `kapitsa serve` refuses the 27B, the 35B,
Flash-Next, Lightning, Bolt and Super-VL before downloading anything, each because its weights alone exceed that
budget. Qwen3.6-35B-A3B fits as a GGUF with 2-bit experts instead: unsloth's UD-IQ2_XXS file through `kapitsa import`
([#134](https://github.com/nace-ai/kapitsa/pull/134), [below](#qwen36-35b-a3b-with-2-bit-experts-gguf-ud-iq2_xxs)). Without `--model` it starts the 4B at 4-bit; a 27,476-token
conversation in its default 32K context ran without swapping (the server's footprint: 5.3 GB).

It is also the first M1 that ran Kapitsa. 0.1.35 stopped at the first sampled request with the default DFlash drafter:
the M1's GPU compiles the drafter's `dfl_dist_rows` kernel for 576 threads per threadgroup, and it is dispatched with
1024 ([#55](https://github.com/nace-ai/kapitsa/pull/55): a kernel is now compiled for the size it is dispatched with;
0.1.36 has the fix). main also did not build with the Command Line Tools a Mac on macOS 14.4 or earlier can install
([#54](https://github.com/nace-ai/kapitsa/pull/54)). With 0.1.36 on this Mac the server tests pass on the 4B (44 checks)
and on Dory (52), greedy answers are the same with DFlash, MTP and plain decoding (six prompts, up to 17,919 prompt
tokens), and so are Dory's with per-loop speculative decoding and without, at 2 and 4 loops.

### Qwen3.5-4B at 4-bit: the full protocol

One hold of the GPU lock, 10:09-10:48, Kapitsa 0.1.36 first and last, every engine in the setting the short runs below
chose: the harness's full protocol (5 runs of the coding answer at T = 0.6 and 5 greedy, 4K and 16K cold prompts with
three cached follow-ups each, tool calls, peak memory). Result files:
[bench/results/2026-10-19-m1pro-16gb](../bench/results/2026-10-19-m1pro-16gb/).

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** 0.1.36 (DFlash drafter), first | **85.5** (77.9-88.6) | 88.0 (86.0-90.3) | 306 / 269 | **0.11** / **0.15** | **4.4** | pass |
| **Kapitsa**, last | 78.5 (75.1-88.8) | **90.0** (87.4-91.8) | 306 / 269 | 0.12 / 0.15 | **4.4** | pass |
| mlx-serve 26.10.2 (MTP, max depth 2) | 62.3 (60.2-63.2) | 62.9 (61.5-63.3) | 286 / 229 | 0.62 / 1.08 | 5.9 | pass |
| Ollama 0.40.6 (MLX runner + MTP) | 55.3 (54.8-56.1) | 57.4 (57.0-58.3) | swapped<sup>1</sup> | 0.32 / 0.54 | 11.1 | pass |
| oMLX 0.7.1rc1 (MTP) | 54.6 (54.2-56.3) | 57.4 (56.8-57.6) | 281 / 240 | 0.52 / 0.96 | 7.4 | pass |
| LM Studio (MLX runtime 1.13.1) | 47.3 (46.1-48.3)<sup>2</sup> | 50.7 (50.4-51.1)<sup>2</sup> | **308** / 277 | 0.84 / 1.55 | 8.8 | pass |
| mlx-lm 0.32.2 | 46.4 (45.8-47.3) | 49.8 (49.5-50.1) | **308** / **289** | 0.38 / 0.48 | 7.3 | pass |
| MTPLX 2.12.2 (depth 1) | 41.5 (39.8-42.4) | 43.9 (42.5-45.0) | 305 / —<sup>3</sup> | 0.19 / —<sup>3</sup> | 9.3 | fail |
| llama.cpp b11669 (MTP, n-max 3; Q4_K_M) | 37.8 (37.4-39.0) | 44.0 (43.8-46.7) | swapped<sup>1</sup> | 0.46 / 0.70 | 11.1 | pass |

Kapitsa writes 1.37x the next engine's rate at T = 0.6 (85.5 against mlx-serve's 62.3) and 1.40x greedy (88.0
against 62.9), answers cached follow-ups soonest (0.11 / 0.15 s; next MTPLX at 4K, 0.19 s, and mlx-lm at 16K, 0.48 s),
and uses the least memory. It loses on reading long prompts: on this Mac its prefill runs on
the GPU alone (its Neural Engine split has no share for the M1), and mlx-lm reads a 16K-token prompt 7% faster (289
against 269 tok/s), LM Studio 3% faster (277); at 4K they are even (306-308).

Since 0.1.37, main has three M1-specific changes, measured on this Mac in interleaved A/Bs against main before each.
[#89](https://github.com/nace-ai/kapitsa/pull/89) runs the prefill attention on 16-key tiles.
[#92](https://github.com/nace-ai/kapitsa/pull/92) runs the prefill GEMMs on tiles with 64-deep K steps. Together they
take the 4B's 4K / 16K-token prompts from 301-305 / 264-267 tok/s (main before them) to 388-389 / 343-349, ahead of
mlx-lm's 308 / 289 here.
[#77](https://github.com/nace-ai/kapitsa/pull/77) changes the 8-row verify GEMM's tiling, for 3-5% shorter DFlash
cycles (greedy decode 88.2 / 88.1 to 91.5 / 90.1 tok/s). The table above is 0.1.36; the README's cells move with the
next release's run.

1. More than 16 GB in use during the prompt phases (Ollama's MLX runner peaked at 11.1 GB in its own processes, llama.cpp
   at 11.1 GB): pages were swapped out during the 4K and 16K prefills (Ollama 126,496 pages at 16K, llama.cpp 83,764),
   so those phases do not count; their decode phases did not swap.
2. LM Studio stops at 511 of the 512 requested tokens; its rates use the 511.
3. MTPLX answered the 16K prompt without its token counts (the harness's check fails) and failed the tool-call checks.

Rapid-MLX 0.16.3 ran in a bracket of its own, 12:51-13:02, with the same protocol and Kapitsa 0.1.36 first and last
(its result files: [rapid-mlx](../bench/results/2026-10-19-m1pro-16gb/rapid-mlx/)); its short runs at 12:12-12:16 chose
no speculation (51.1 tok/s at T = 0.6) over MTP with Qwen's head (44.4):

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** 0.1.36 (DFlash drafter), first | 78.4 (71.4-83.0) | **88.2** (86.0-88.9) | **305** / 267 | **0.11** / **0.14** | **4.4** | pass |
| **Kapitsa**, last | **89.3** (74.1-120.6)<sup>1</sup> | 86.8 (82.0-88.7) | 302 / 265 | 0.12 / 0.15 | **4.4** | pass |
| Rapid-MLX 0.16.3 (no speculation) | 51.5 (51.5-51.8) | 55.1 (50.5-55.6) | 284 / **286** | 0.76 / 0.81 | 7.2 | pass |

1. Three of the five answers fell into repetition loops (distinct 4-grams below 0.5), which flatters speculative
   decoding.

### Qwen3.5-4B at 4-bit: short runs of every engine

One hold of the GPU lock, 08:48-09:30, Kapitsa first (LM Studio in another at 09:35): each engine and setting in a short run of the harness's decode
phase, the T = 0.6 coding answer, 3 runs of 512 tokens, median (`--skip greedy,prefill,tools,agent --bench-args
"--decode-runs 3"`), on the same 4-bit weights where the engine accepts them; nothing was flagged for contention.
Kapitsa is the 0.1.36 release. These runs chose each engine's setting for the README's cells; the full protocol
(5 runs, greedy, 4K and 16K prefill, cached follow-ups, memory, tool calls) follows. Result files:
[bench/results/2026-10-19-m1pro-16gb](../bench/results/2026-10-19-m1pro-16gb/).

| Engine (setting) | T = 0.6 (tok/s): median (min-max) |
| --- | ---: |
| **Kapitsa** 0.1.36 (DFlash drafter, its default) | **85.4** (72.4-95.6) |
| mlx-serve 26.10.2 + MTP, max depth 2 | 62.4 (61.8-62.8) |
| mlx-serve + MTP, max depth 3 (its setting on the M4 Pro) | 61.1 (60.1-62.7) |
| mlx-serve + MTP, max depth 4 | 60.8 (58.5-69.3) |
| oMLX 0.7.1rc1 + MTP (Lightning MTP, MTPLX's head) | 56.5 (55.4-57.2) |
| Ollama 0.40.6, MLX runner + MTP (adaptive depth) | 56.0 (55.8-56.8) |
| mlx-serve + DFlash (z-lab's drafter), block 4 | 55.6 (54.2-56.0) |
| mlx-serve + DFlash, block 5 | 52.0 (51.1-55.4) |
| oMLX + DFlash (z-lab's drafter at 4-bit, adaptive verify), block 4 | 50.8 (50.5-51.1) |
| mlx-lm 0.32.2 | 47.4 (46.0-47.5) |
| LM Studio (llmster 0.0.25-1, MLX runtime 1.13.1) | 44.6 (44.5-44.9) |
| oMLX + DFlash, block 5 | 44.1 (43.9-46.5)<sup>1</sup> |
| MTPLX 2.12.2 (sustained profile, depth 1) | 41.6 (41.5-41.7) |
| llama.cpp b11669 + MTP, n-max 3 (Q4_K_M) | 40.8 (38.8-42.7) |
| llama.cpp + MTP, n-max 6 | 39.2 (35.7-42.8) |
| llama.cpp, no MTP | 38.2 (38.1-38.2) |
| llama.cpp + MTP, n-max 2 | 37.7 (37.1-41.0) |
| mlx-vlm 0.7.8 + MTP, block 2 | 37.2 (36.4-37.8)<sup>1</sup> |
| mlx-vlm + MTP, block 3 (its setting on the M4 Pro) | 36.6 (35.5-36.9) |
| MTPLX, depth 2 | 35.4 (34.6-36.0)<sup>2</sup> |
| llama.cpp + MTP, n-max 1 | 34.5 (33.7-34.9) |
| Ollama, GGUF + MTP (shipped depth 1, Q4_K_M) | 33.1 (32.9-34.3) |
| mlx-vlm + MTP, block 4 | 6.9 (6.7-6.9) |
Kapitsa wrote 1.37x the next engine's rate (mlx-serve with MTP). On this GPU speculative decoding pays less: a verify
pass reads the weights once for several tokens but has to compute all of them, and the M1 computes more slowly for its
bandwidth. Kapitsa's 8-row verify of the 4B reads its weights at 78-91 GB/s, against 130-177 GB/s for a one-token step
(`bench_v8`, `bench_qmv`), so a verify costs about twice a step; several engines' MTP is no faster than their plain
decoding here (llama.cpp at n-max 1 and 2, mlx-vlm at any block). Kapitsa's defaults (DFlash, all 7 drafts verified)
held up on this Mac too, in a sweep of 3-run decodes the same morning (T = 0.6 / greedy tok/s): verify depth 7 82.4 /
88.0 and again 80.2 / 84.3, depth 5 84.3 / 83.6, depth 3 72.1 / 74.3; MTP instead of the drafter, depth 4 78.6 / 78.9.

1. One or two of the three runs fell into a repetition loop (distinct 4-grams below 0.5), which flatters speculative
   decoding.
2. The disk had 9.5 GiB free at the start, under the 10 GiB below which MTPLX keeps no prompt cache on disk (no
   effect on this phase).

### Nemotron Dory1.0 4B

The README's cell comes from one hold of the GPU lock, 09:37-09:50, Kapitsa first and last, the decode phase of the
harness (the T = 0.6 coding answer; llama.cpp 2 runs of 256 tokens, the others 5 of 512) with
[configs/dory.json](../bench/harness/configs/dory.json), as [on the M4 Pro](#nemotron-dory10-4b-2026-10-19).

| Engine (setting) | Weights | T = 0.6 (tok/s): median (min-max) |
| --- | --- | ---: |
| **Kapitsa** 0.1.36 (last-loop KV, its default: one token per pass) | NVIDIA's NVFP4 (8-bit LM head) | **35.2** (34.7-35.5) |
| Kapitsa (per-loop KV, `--loop-kv per`: speculative decoding), first / last | NVIDIA's NVFP4 (8-bit LM head) | 30.8 (30.0-31.7) / 31.7 (30.7-33.6) |
| mlx-lm with its Dory pull request (`ml-explore/mlx-lm#1988` at 208065f), per-loop KV | snsnc's MLX 4-bit | 29.9 (29.6-30.1)<sup>1</sup> |
| llama.cpp: NVIDIA's Dory fork (`brianchmielNV/llama.cpp` at 98a74c97), last-loop KV | NVIDIA's Q5_K_M GGUF | 4.4 (4.4-4.5) |

All at 2 loops. Unlike on the M4 Pro (per-loop 64.1 against last-loop 59.6 tok/s), Kapitsa's per-loop speculative
decoding is slower than its default here: a verify pass of 2-8 rows costs the M1 about 1.6 times a one-row step
(`dory_mv_bench`, the q|k|v|g product: 170 against 108 us), more than the accepted drafts make up, so the README shows
the default. Kapitsa 0.1.37, the next release, reads Dory's NVFP4 in layout L2 on the M1 ([#72](https://github.com/nace-ai/kapitsa/pull/72),
the same outputs to the bit): in its own hold at 10:49, 40.5 tok/s (40.0-40.8) with last-loop KV and 36.7 (34.6-37.2)
with per-loop speculation, against 35.2 and 30.8 / 31.7 with 0.1.36 above; the README's cell is the 40.5.

The engines with no Dory model were checked on this Mac: oMLX 0.7.1rc1 (no `mlx_lm.models.dory`), mlx-serve 26.10.2
(`MissingWeight` at load), MTPLX 2.12.2 ("incompatible-architecture"), Ollama 0.40.6 ("unknown model architecture:
'dory'"); LM Studio's runtimes (llama.cpp 2.60.0, MLX 1.13.1) are the versions found without Dory support on the M4 Pro.

1. One of the five answers echoed the prompt and closed a `</think>` block, which makes the harness mark the result
   invalid; the median is over the other four.

### Qwen3.5-9B at 4-bit

In the catalog since [#105](https://github.com/nace-ai/kapitsa/pull/105) (`kapitsa serve 9b`, the default model on
16-24 GB Macs), with these weights, its MTP head as a sidecar and this drafter.


The largest Qwen3.5 that fits this Mac with a drafter: lmstudio-community/Qwen3.5-9B-MLX-4bit (`b455506`, 5.6 GB) and
z-lab/Qwen3.5-9B-DFlash (`5fc3b3d`), served with `kapitsa serve --model <folder> --dflash <drafter folder>`. Kapitsa
0.1.37 holds it in the 12.2 GB budget with the full 32K context: 7.9 GB allocated with its default 4 slots, the drafter
1.0 GB of it at 4 bits. On it `tests/test_server.py` stopped at an HTTP 500: a reply that `max_tokens` cut inside an
emoji was not valid UTF-8 ([#76](https://github.com/nace-ai/kapitsa/pull/76) fixes that; then its 44 checks pass).
Greedy answers with DFlash equal plain decoding's on 5 of the 6 prompts with 0.1.37. The sixth parts late in the
answer where plain decoding's top two logits are 0.0003 apart; it equals plain decoding's too with
[#77](https://github.com/nace-ai/kapitsa/pull/77)'s verify tiling.

The full protocol in one hold of the GPU lock, 12:28-12:51, Kapitsa 0.1.37 first and last, each engine in the setting
its short run chose (below). Result files and the configs:
[bench/results/2026-10-19-m1pro-16gb/qwen3.5-9b](../bench/results/2026-10-19-m1pro-16gb/qwen3.5-9b/).

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** 0.1.37 (DFlash drafter), first | 47.1 (44.1-63.3) | **54.1** (49.8-54.9) | 166 / 154 | **0.18** / **0.20** | 7.8 | pass |
| **Kapitsa**, last | **50.7** (44.4-54.5) | 52.1 (50.0-55.0) | 164 / 153 | **0.18** / 0.24 | 7.8 | pass |
| mlx-serve 26.10.2 (MTP, max depth 2) | 37.4 (35.9-38.5) | 37.9 (37.1-38.6) | 159 / —<sup>1</sup> | 0.79 / —<sup>1</sup> | **7.0** | fail<sup>2</sup> |
| mlx-lm 0.32.2 | 29.7 (29.5-29.8) | 31.2 (30.7-31.3) | **170** / **165** | 0.59 / 0.71 | 9.6 | pass |

Kapitsa decodes 1.26-1.36x mlx-serve's rate and 1.59-1.71x mlx-lm's, and answers cached follow-ups 3-4x sooner.
It loses on reading prompts: mlx-lm reads a 4K-token prompt 2-3% faster and a 16K-token one 7% faster (165 against
153-154 tok/s).

1. mlx-serve refused the 16K-token prompt: "Prompt (16286 tokens) requires ~1636MB GPU memory but only ~958MB is
   available".
2. With thinking on, mlx-serve answered the tool-call request without a call.

Short runs before it (12:01-12:11), 3 decodes of the T = 0.6 coding answer each, as for the 4B:

| Engine (setting) | T = 0.6 (tok/s): median (min-max) |
| --- | ---: |
| **Kapitsa** 0.1.37 (DFlash drafter) | **40.2** (40.0-50.5) |
| mlx-serve + MTP (Qwen's head at 4 bits, mlx-community/Qwen3.5-9B-MTP-4bit `222dfd2`), max depth 2 | 37.1 (35.8-38.1) |
| mlx-serve + MTP, max depth 3 | 35.9 (35.6-36.1) |
| mlx-serve + DFlash (z-lab's drafter), block 4 | 29.6 (26.4-29.6) |
| mlx-lm 0.32.2 | 29.5 (29.4-29.7) |

Not run on the 9B: oMLX, MTPLX, Ollama, LM Studio and llama.cpp. Their installs had been removed to make room for the
9B's 8 GB, and this Mac's disk has no room for them next to it. mlx-serve's MTP head is attached with
`tools/mlx_serve_pack.py qwen3.5-9b` (the trunk cloned, the head's tensors renamed to `mtp.*`).

### Qwen3.6-35B-A3B with 2-bit experts (GGUF UD-IQ2_XXS)

The catalog's 35B starts at 22.1 GB (4-bit). unsloth's UD-IQ2_XXS GGUF of it (`unsloth/Qwen3.6-35B-A3B-GGUF` at
`a483e9e`, 10.76 GB: IQ2_XXS gate and up experts, IQ2_S and IQ3_S down experts) runs with its routed experts in their
own blocks ([#134](https://github.com/nace-ai/kapitsa/pull/134)): `kapitsa import` takes 7 s and writes 1.92 GB (the
experts file is an APFS clone of the GGUF), and `kapitsa doctor` gives 10.99 GB of weights, room for 21,811 tokens of
KV cache in the 12.2 GB budget. It has no MTP head and no drafter fits beside it, so both engines decode plainly.

bench.py through each engine's server, the same file in both:

| Engine | Decode, T = 0.6 / greedy (tok/s) | 4K prompt (s) | 16K prompt (s) | Cached follow-up 4K / 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** | **52.6 / 53.2** | **9.0** | **39.7-41.2**<sup>1</sup> | **0.34 / 0.33** | **11.2** | pass |
| llama.cpp b11669 | 38.0-38.7 / 38.8<sup>2</sup> | 11.8-12.4 | 52.0-53.2 | 0.38-0.40 / 0.41-0.80 | 11.65-16.1<sup>2</sup> | pass |

Teacher-forced over 1,020 positions against llama.cpp's Metal backend: KL 0.023 (median 0.019), top-1 agreement
97.5%, perplexity 2.442 against 2.450; llama.cpp's own CPU backend is at KL 0.012 (median 0.001), top-1 97.45%.
Part of Kapitsa's gap is the existing conversion of the file's Q5_K tensors to 6-bit (an 8-bit import: 0.019); on a
4-layer slice of the file the IQ experts alone match llama.cpp's CPU-against-Metal noise (median 0.0043 against
0.0042). Greedy answers are identical in both engines on 7 of 8 prompts. `tests/test_server.py`: 45 of 47, the two
others counting tests this file itself stops early on (llama.cpp's answers are the same).

1. 39.7 s in a swap-free `ni_run` rep; the server's 41.2 s run swapped 11K pages.
2. llama.cpp swapped in nearly every phase on this Mac: 10.7 GB with the harness's standard settings (16.1 GB peak),
   still 5K-50K pages per phase with `--cache-ram 0 --ctx-checkpoints 4` and its weights locked (11.65 GB peak). Its
   four runs agree within about 2%, but by the swap rule only its greedy 38.8 tok/s counts.

### Nemotron 3 Nano 4B and Nemotron Nano 9B v2

NVIDIA's two small dense Nemotron-H models (Mamba-2, attention and MLP layers, no experts), text only, with no drafter:
Kapitsa decodes the 4B with prompt lookup ([#103](https://github.com/nace-ai/kapitsa/pull/103)). Both are in the
catalog since [#127](https://github.com/nace-ai/kapitsa/pull/127): `nano-4b` downloads NVIDIA's FP8 checkpoint
(ModelOpt W8A16, 5.3 GB; `nano-4b:bf16` the BF16 one, 8.0 GB), `nano-9b` its NVFP4 one (7.9 GB). An FP8 or NVFP4
checkpoint's BF16 LM head loads as 8-bit rows ([#129](https://github.com/nace-ai/kapitsa/pull/129)).

bench.py through each engine's server, one slot, the T = 0.6 and greedy coding answers (512 tokens, median of 3) and a
cold 4K-token prompt. Kapitsa's rows are from the catalog's downloads (2026-10-19 and 10-20); the other engines' from
#103's runs on this Mac (2026-10-19), as are their footnotes.

| Model | Engine | Weights | Decode, T = 0.6 / greedy (tok/s) | Prefill 4K (tok/s) | Peak memory (GB) |
| --- | --- | --- | ---: | ---: | ---: |
| Nano 4B | **Kapitsa** (prompt lookup) | NVIDIA's FP8, 8-bit head | **37.3 / 37.2**<sup>4</sup> | **449** | **4.6** |
| Nano 4B | Kapitsa (prompt lookup) | NVIDIA's BF16 | 24.4 / 23.8 | **449** | 8.0 |
| Nano 4B | llama.cpp b11669 | NVIDIA's Q4_K_M GGUF | 28.2 / 28.4 | 262 | 6.8 |
| Nano 4B | mlx-lm 0.32.2 | NVIDIA's BF16 | 21.0 / 20.5 | 379<sup>1</sup> | 10.7-12.4<sup>3</sup> |
| Nano 9B v2 | **Kapitsa** | NVIDIA's NVFP4, 8-bit head | **25.1 / 25.0** | **187** | **7.3** |
| Nano 9B v2 | llama.cpp b11669 | Q4_K_M GGUF | 14.2 / 14.2 | 118 | 10.8-11.1 |
| Nano 9B v2 | mlx-lm 0.32.2 | NVIDIA's NVFP4<sup>2</sup> | 18.1 / 17.9 | 135<sup>1</sup> | 10.7-12.4<sup>3</sup> |

On the 4B, FP8 decodes 1.5x BF16 in 3.4 GB less memory, and it is the catalog's default. llama.cpp on a 4-bit file reads
fewer bytes per token (2.8 GB against FP8's 4.1 GB), and Kapitsa's FP8 still decodes 1.3x its rate. The 4B's prompt
lookup gains most where a reply repeats its prompt (1.4-2.3x on a code rewrite, [#131](https://github.com/nace-ai/kapitsa/pull/131)),
and it is 4-5% slower on free prose, where few drafts match.

- FP8 against BF16 on the 4B, teacher-forced over 8,172 positions (prose, Python, C++, a chat and a math transcript):
  KL(BF16 || FP8) 0.0014 on the decode and prefill paths, top-1 agreement 98.3%, perplexity 6.595 against 6.599. The
  8-bit head adds 3.9e-5 (top-1 99.6% against FP8 with its BF16 head). NVIDIA's own evaluations of the two checkpoints
  are within a point or two on most benchmarks and lower on Tau2's agent tasks.
- `tests/test_server.py`: 47/47 on the 4B (FP8 and BF16), 43/47 on the 9B v2, whose own template drops past turns'
  reasoning (two cache-reuse checks) and which ends a "count to 2000" reply after 86 tokens, as the reference does.
- bench.py marks the prefill prompts as 1-1.5% longer than its Qwen-calibrated counts (Nemotron's template), so those
  are each engine's own figures rather than the harness's cross-engine ones.

1. mlx-lm's prefill ran out of memory at its default 2048-token step on both models; its figures are at step 512.
2. mlx-lm can't load ModelOpt NVFP4: its `nemotron_h` code ran the NVFP4 weights on MLX's nvfp4 matmul (#103).
3. #103 gives mlx-lm's peak as one range over both models' runs.
4. With [#129](https://github.com/nace-ai/kapitsa/pull/129)'s 8-bit head and #131's FP8 tail kernels, 2026-10-20; two
   rounds: T = 0.6 37.3 and 38.6, greedy 37.15 and 37.13. The prefill and memory columns are the catalog run's
   (2026-10-19, BF16 head).

## MacBook Pro M5 Pro, 48 GB (2026-10-19)

A MacBook Pro (16-inch, 2026), `Mac17,8`: M5 Pro with an 18-core CPU (6 super, 12 performance) and a 20-core GPU,
48 GB, macOS 26.5.2 (25F84), on AC power in High Power mode. Every engine ran the harness's full protocol (5 runs of the
coding answer at T = 0.6 and 5 greedy, 4K and 16K cold prompts with three cached follow-ups each, tool calls, peak
memory) in the settings measured on the M4 Pro and M5 Max, one config at a time under the GPU lock, 10:09-15:34.
Kapitsa is main at 68e964a (0.1.36 and one README commit). The disk held one model family at a time, so the runs went
model by model; the session closed with a control round on an idle machine (14:36-15:00): Kapitsa and the fastest
other engine on each model again. The tables show each engine's fastest setting, from its newest run, as the README
does. The Mac held about 6 GB in swap throughout. Result files, every run listed with its flags:
[bench/results/2026-10-19-m5pro-48gb](../bench/results/2026-10-19-m5pro-48gb/).

Each Kapitsa table below also carries a paragraph with that model's newest flag-free run on 0.2.2, which is after the
M5 prefill kernels and the KFlash-tree default; the other engines' rows are the 2026-10-19 session in every table.
Current main also peaks 0.3-0.8 GB above these rows on all four models (4.7 → 5.4 GB, 19.4 → 20.2, 20.8 → 21.4, 18.6 →
18.9 GB). The growth is the SSD prefix cache ([#212](https://github.com/nace-ai/kapitsa/pull/212)), which is new since
0.1.36 and on by default in `serve`: rebuilt binaries run with the table's own protocol (16K sweep, tool calls, cache
turn, `--slots 1`) peak at 4,798 MB at 0.1.36, 4,799 MB at #212's parent, 5,610 MB at #212 itself and 5,665 MB at
180a650, and the same protocol at 180a650 with `--disk-cache 0` peaks at 4,851 MB (tools 4,358, sweep 4,851, cache
turn 4,851), so the cache is about 0.8 GB of that 5,665 MB and the commits after #212 add about 50 MB. That split is
measured on the 4B; the 27B, 35B and Lightning rows were not re-run with the cache off. The largest thing the cache
holds is the copy it takes of a conversation that is being replaced: when a request replaces a conversation the SSD
lacks, `serve` copies that conversation's KV into RAM before the write-behind thread writes it to the SSD (each copy
up to `capture_max`, RAM/32 capped at 1 GiB — 604 MB for a 16K conversation on the 4B) and frees it once the write
finishes, which is why the peak lands in the phase that replaces the sweep's conversation and why the doctor's plan
total, a steady-state figure, does not move. The stored 4B run above, 5,553 MB, is the same engine at 180a650 in a
louder session, and peak footprint does not track the 7-12% foreign GPU load these runs were flagged with.

### Qwen3.5-4B at 4-bit

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** 0.1.36 (DFlash drafter) | **178.8** (166.7-185.2)<sup>1</sup> | **201.8** (186.2-204.7) | 2,091 / 1,753 | **0.05** / **0.06** | **4.7** | pass |
| mlx-serve 26.10.2 (MTP) | 162.6 (154.7-164.7) | 166.3 (162.3-167.7) | 2,717 / 2,571 | 0.15 / 0.18 | 7.7 | pass |
| Ollama 0.40.6 (MLX runner + MTP) | 134.0 (126.8-139.5) | 140.5 (136.3-149.2) | 2,609 / 2,628 | 0.12 / 0.15 | 9.3 | pass |
| oMLX 0.7.1rc1 (MTP) | 113.4 (110.5-122.0) | 114.1 (110.5-122.6) | **2,842** / 2,650 | 0.17 / 0.28 | 7.5 | pass |
| MTPLX 2.12.2 (depth 1) | 113.1 (112.2-116.6) | 117.8 (117.4-119.3) | 2,697 / 2,496 | 0.06 / 0.08 | 15.4 | pass |
| Rapid-MLX 0.16.3 (MTP, 3 drafts) | 112.4 (105.4-113.0) | 120.5 (117.0-121.9) | 2,736 / 2,480 | 0.14 / 0.24 | 7.9 | pass |
| mlx-vlm 0.7.8 (MTP) | 100.8 (96.5-104.2) | 115.8 (113.9-120.0) | 2,756 / **2,755** | 0.24 / 1.05 | 9.4 | pass |
| llama.cpp b11669 (MTP, n-max 3; Q4_K_M) | 93.0 (90.2-97.1) | 103.4 (101.4-107.4) | 1,799 / 1,642 | 0.19 / 0.22 | 11.1 | pass |
| LM Studio (MLX runtime 1.13.1) | 88.7 (87.8-90.5) | 92.0 (90.8-93.9) | 861 / 777 | 0.40 / 0.67 | 9.0 | pass |

Kapitsa's row above is the 2026-10-19 session, and that build predates the M5 prefill kernels ([#205](https://github.com/nace-ai/kapitsa/pull/205),
[#215](https://github.com/nace-ai/kapitsa/pull/215)). On current main (0.2.2, `3a5bbf0`, after [#243](https://github.com/nace-ai/kapitsa/pull/243)
and [#245](https://github.com/nace-ai/kapitsa/pull/245)) the 4B reads **3,269 / 2,735 tok/s** at 4K / 16K and decodes **226.7 / 238.8 tok/s**
(T = 0.6 / greedy), with cached follow-ups at 0.05 / 0.06 s and a peak footprint of 5.4 GB: flag-free, 0% foreign GPU, on AC power in High
Power mode, [`q35-4b-kapitsa-20261021-235433.json`](../bench/results/2026-10-21-m5pro-48gb/README.md). The other engines' rows are still their
2026-10-19 runs, so the columns are not side-by-side measurements and are read as stored: Kapitsa reads 1.15x oMLX's 2,842 at 4K and 1.03x its
2,650 at 16K, ahead of every other engine's decode, and behind only mlx-vlm's 2,755 at 16K (0.7%).

### Qwen3.8-27B at 4-bit

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| mlx-serve 26.10.2 (DFlash2 draft tree, block 8) | **71.2** (66.5-79.1)<sup>2</sup> | **85.6** (82.4-87.1) | **496** / **476** | **0.14** / **0.17** | 20.7 | pass |
| **Kapitsa** 0.1.36 (DFlash2 drafter) | 60.7 (55.0-66.0)<sup>2</sup> | 64.8 (64.0-65.6) | 403 / 359 | 0.21 / 0.24 | **19.4** | pass |
| oMLX 0.7.1rc1 (DFlash2) | 50.6 (45.6-53.1) | 52.1 (51.9-52.2) | 446 / 442 | 0.50 / 0.73 | 21.8 | pass |
| MTPLX 2.12.2 (depth 3) | 40.5 (38.8-42.0) | 44.1 (43.0-44.5) | 471 / 450 | 0.22 / 0.26 | 29.8 | pass |
| Ollama 0.40.6 (MLX runner + MTP) | 40.0 (36.1-43.4) | 41.2 (40.7-41.7) | 470 / 456 | 0.29 / 0.34 | 29.6 | pass |
| Rapid-MLX 0.16.3 (MTP on its own pack, 3 drafts) | 34.9 (30.8-38.8) | 35.0 (34.6-35.1) | 458 / 457 | 0.49 / 0.57 | 23.9 | pass |
| mlx-vlm 0.7.8 (MTP) | 32.0 (30.2-32.3) | 34.2 (33.7-34.4) | 421 / 405<sup>3</sup> | 0.49 / 0.70 | 26.3 | pass |
| llama.cpp b11669 (MTP, n-max 3; Q4_K_M) | 29.6 (27.5-31.8) | 30.7 (29.2-31.2)<sup>3</sup> | 322<sup>3</sup> / 312 | 0.52 / 0.54 | 31.8 | pass |
| LM Studio (MLX runtime 1.13.1) | 16.8 (16.7-17.4) | 16.9 (16.4-17.4) | 118 / 129 | 0.52 / 0.62 | 23.1 | pass |

Kapitsa's row above is the 2026-10-19 session, and that build predates the M5 prefill kernels
([#205](https://github.com/nace-ai/kapitsa/pull/205), [#215](https://github.com/nace-ai/kapitsa/pull/215)) and the
KFlash-tree decode default ([#187](https://github.com/nace-ai/kapitsa/pull/187)). On 0.2.2 (`a13f8fd`) the 27B reads
**463 / 451 tok/s** at 4K / 16K and decodes **74.1 / 77.4 tok/s** (T = 0.6 / greedy), with cached follow-ups at
0.21 / 0.23 s and a peak footprint of 20.2 GB: flag-free, on AC power in High Power mode, in two runs that agree to
0.4% (465 / 451 at 19:57, 2.4 h before),
[`q38-27b-kapitsa-20261021-221847.json`](../bench/results/2026-10-21-m5pro-48gb/README.md).
Read as stored beside the 10-19 rows: Kapitsa's sampled decode is the table's fastest (74.1 against mlx-serve's 71.2,
1.04x), mlx-serve keeps greedy (85.6, 1.11x) and both prefill cells (496 / 476 against 463 / 451, 1.07x / 1.06x), and
Kapitsa reads the prompt fourth of nine at both sizes.

### Qwen3.6-35B-A3B at 4-bit

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** 0.1.36 (DFlash2 drafter) | **166.8** (136.4-183.8) | **189.5** (182.3-197.2) | 1,258 / 1,046 | **0.05** / **0.06** | **20.8** | pass |
| mlx-serve 26.10.2 (MTP) | 155.7 (144.8-158.2) | 159.7 (150.4-161.9) | 2,494 / 2,283 | 0.20 / 0.26 | 22.1 | pass |
| oMLX 0.7.1rc1 (DFlash2) | 144.4 (131.5-152.6) | 151.2 (144.4-163.0) | 1,619 / 1,658 | 0.22 / 0.35 | 23.1 | pass |
| llama.cpp b11669 (MTP, n-max 2; Q4_K_M) | 107.3 (99.8-108.9) | 108.8 (107.5-110.8) | 1,755 / 1,558 | 0.16 / 0.16 | 27.4 | pass |
| Ollama 0.40.6 (GGUF + MTP) | 102.4 (96.3-104.3) | 103.5 (102.3-107.6) | 1,518 / 1,383 | 0.14 / 0.21 | 5.6 + 19.9 mapped<sup>4</sup> | pass |
| mlx-vlm 0.7.8 (DFlash2) | 98.2 (91.3-100.0) | 106.5 (101.8-111.9) | 2,474 / **2,401** | 0.34 / 1.25 | 27.5 | pass |
| LM Studio (MLX runtime 1.13.1) | 95.6 (95.1-96.1) | 97.8 (95.0-98.8) | 997 / 883 | 0.36 / 0.68 | 24.2 | pass |
| MTPLX 2.12.2 (depth 1) | 88.6 (87.0-91.7) | 83.8 (79.1-85.4) | **2,569** / 2,303 | 0.08 / 0.11 | 28.6 | pass |
| Rapid-MLX 0.16.3 (no speculation)<sup>8</sup> | 75.7 (74.1-76.0) | 77.6 (77.0-77.8) | 1,826 / 1,703 | 0.22 / 0.31 | 23.8 | pass |

Kapitsa's row above is the 2026-10-19 session, before the M5 prefill kernels ([#205](https://github.com/nace-ai/kapitsa/pull/205),
[#215](https://github.com/nace-ai/kapitsa/pull/215)) and the KFlash-tree decode default ([#187](https://github.com/nace-ai/kapitsa/pull/187)).
On 0.2.2 (`a13f8fd`) the 35B reads **1,799 / 1,455 tok/s** at 4K / 16K and decodes **195.1 / 223.4 tok/s** (T = 0.6 / greedy),
1.25x / 1.40x mlx-serve, with cached follow-ups at 0.05 / 0.07 s (the table's fastest) and a peak footprint of 21.4 GB:
flag-free, on AC power in High Power mode, and 1.4% behind the first set that evening (20:06, 1,824 / 1,469),
[`q36-35b-kapitsa-20261021-220021.json`](../bench/results/2026-10-21-m5pro-48gb/README.md). Read as stored it is fifth of
nine on prefill at 4K (MTPLX 2,569, mlx-serve 2,494 and mlx-vlm 2,474 lead) and seventh at 16K (mlx-vlm 2,401,
MTPLX 2,303, mlx-serve 2,283 lead).

On 0.2.2 (`a13f8fd`, `3a5bbf0`) Kapitsa decodes fastest in four of the tables above: 226.7, 74.1,
195.1 and 159.3 tok/s at T = 0.6 on the 4B, the 27B, the 35B and Lightning, 1.39x, 1.04x, 1.25x and 1.29x the next
engine, and 1.44x, 1.40x and 1.37x greedy on the 4B, 35B and Lightning (mlx-serve's block-8 DFlash2 draft tree keeps
the 27B's greedy cell, 85.6 against 77.4, 1.11x). It answers cached follow-ups soonest on the 4B and the 35B
(0.05 / 0.06 s and 0.05 / 0.07 s; next MTPLX, 0.06 / 0.08 s and 0.08 / 0.11 s). It reads prompts fastest on the 4B at
4K (3,269 against oMLX's 2,842) and on Lightning at both sizes, and second on the 4B at 16K (2,735 against mlx-vlm's
2,755, 0.7%). It is fourth of nine on the 27B at both sizes, behind mlx-serve (496 / 476), MTPLX (471) and Ollama (470)
at 4K and mlx-serve, Rapid-MLX (457) and Ollama (456) at 16K; on the 35B it is fifth at 4K behind MTPLX, mlx-serve,
mlx-vlm and Rapid-MLX, and seventh at 16K. The pre-kernel Kapitsa rows in
these tables understate current main by 1.15x to 3.07x on prefill, and the other engines' rows are still the
2026-10-19 session, so every column is read as stored rather than as one side-by-side run.

1. Kapitsa's first 4B run, at 10:13: 189.3 tok/s at T = 0.6 (191.9 greedy). The Mac swapped 23,955 pages during the
   closing run's 16K prefill.
2. Their earlier runs: mlx-serve 79.1 at 11:41 (89.9 greedy; the Mac swapped during every phase), Kapitsa 55.6 at
   10:09 (57.1 greedy; another app used the GPU during its tool calls).
3. Another app (Slack) used the GPU during this phase (contention flagged); the T = 0.6 decode was not flagged.
4. The GGUF runners of Ollama and LM Studio map the weights from disk: the first figure is the server's memory, the
   second the resident mapped weights.
5. One greedy answer stopped at 391 tokens instead of running to 512, and the Mac swapped 3,500-4,600 pages during
   each phase.
6. mlx-vlm reused none of the conversation's prompt for follow-ups on Bolt, so each follow-up re-read all of it.
7. An answer in this phase contained `</think>`, which marks the run invalid; its decode phases are valid.
8. Rapid-MLX's catalog turns speculation off for this mixture-of-experts model. Its 27B and 35B runs came last,
   15:27-15:34, after the control round, once the 27B pack had downloaded.

### Nemotron 3.5 Lightning 30B-A3B

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** 0.1.36 (DSpark drafter) | **135.8** (127.0-141.4) | **132.0** (120.3-136.0) | 1,022 / 953 | 0.12 / 0.14 | **18.6** | pass |
| oMLX 0.7.1rc1 (MTP, depth 2; MLX 4-bit) | 123.6 (119.8-128.8) | 124.4 (120.1-127.0) | 1,938 / 2,021 | 0.20 / 0.29 | 21.6 | pass |
| mlx-serve 26.10.2 (MTP; MLX 4-bit) | 114.5 (113.5-115.5)<sup>5</sup> | 117.2 (115.2-131.0)<sup>5</sup> | 1,728 / 1,847 | **0.09** / **0.10** | 19.8 | pass |
| llama.cpp b11669 (MTP, n-max 2; Q4_0) | 101.1 (94.5-101.8) | 100.4 (92.9-104.2) | 1,915 / 1,813 | 0.13 / 0.14 | 24.1 | pass |
| MTPLX 2.12.2 (depth 1; MLX 4-bit) | 100.6 (94.9-102.8) | 104.9 (99.8-107.2) | 1,743 / 1,779 | **0.09** / **0.10** | 29.8 | pass |
| mlx-lm 0.32.2 (MLX 4-bit) | 93.7 (91.8-94.7) | 96.8 (96.6-98.3) | 1,881 / 1,939 | 0.13 / 0.17 | 21.1 | pass |
| Ollama 0.40.6 (MLX runner, NVFP4 tag + MTP) | 91.5 (82.1-96.3) | 88.8 (87.4-98.4) | **2,369** / **2,310** | 0.17 / 0.21 | 35.8 | pass |
| LM Studio (llama.cpp runtime; Q4_K_M) | 72.3 (72.0-72.5) | 72.2 (71.7-72.5) | 1,479 / 1,443 | 0.18 / 0.16 | 4.7 + 22.9 mapped<sup>4</sup> | pass |

Kapitsa's row above is the 2026-10-19 session, before the M5 prefill kernels ([#205](https://github.com/nace-ai/kapitsa/pull/205),
[#215](https://github.com/nace-ai/kapitsa/pull/215)) and the KFlash-tree decode default ([#187](https://github.com/nace-ai/kapitsa/pull/187)).
On 0.2.2 (`a13f8fd`) Lightning reads **3,141 / 2,982 tok/s** at 4K / 16K, fastest in the table at both sizes (1.33x / 1.29x
Ollama's 2,369 / 2,310), and decodes **159.3 / 170.8 tok/s** (1.29x / 1.37x oMLX's 123.6 / 124.4), with cached follow-ups at
0.09 / 0.10 s (level with mlx-serve and MTPLX) and a peak footprint of 18.9 GB: flag-free, on AC power in High Power mode,
and 3.1% ahead of the first set that evening (20:02, 3,048 / 2,989),
[`lightning-kapitsa-20261021-221051.json`](../bench/results/2026-10-21-m5pro-48gb/README.md).

### Nemotron 3.5 Bolt 74B-A9B

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** 0.1.36 (LiLiCorr drafter) | **79.4** (71.0-81.4) | **75.8** (69.5-80.2) | **588** / **580** | **0.21** / **0.25** | **22.6** | pass |
| mlx-vlm 0.7.8 (no drafter) | 40.8 (40.1-40.8) | 40.4 (40.3-41.0) | 524 / 540 | 8.17 / 30.12<sup>6</sup> | 32.3 | pass |
| oMLX 0.7.1rc1 (no drafter) | 39.0 (38.1-39.8) | 39.8 (39.6-39.9) | 462 / 533 | 0.72 / 0.77 | 30.3 | pass |

oMLX's LiLiCorr drafter was not run: Bolt's files had left the disk before its greedy-only run came up (the drafter
fails at T = 0.6, as on the M5 Max).

Kapitsa's Bolt cells are the 0.1.36 build too, and unlike the four models above they could not be re-measured: Bolt's
files had left the disk before the M5 prefill kernels ([#205](https://github.com/nace-ai/kapitsa/pull/205),
[#215](https://github.com/nace-ai/kapitsa/pull/215)) and the KFlash-tree default ([#187](https://github.com/nace-ai/kapitsa/pull/187))
landed, and they were not downloaded again (this Mac has no Bolt checkpoint). Both its prefill and its decode cells are
therefore lower bounds for current main; on the models that were re-measured those two changes moved 4K prefill by
1.15x to 3.07x (see the four paragraphs above).

### Nemotron Dory1.0 4B

| Engine (setting) | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak memory (GB) | Tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** 0.1.36 (per-loop KV, speculative decoding; NVFP4) | **69.6** (61.1-71.5) | **75.8** (75.1-76.7) | 595 / 528 | **0.10** / **0.15** | **4.1** | pass |
| mlx-lm, Dory pull request (per-loop KV; snsnc 4-bit) | 52.6 (51.9-54.7) | 53.8 (53.6-55.6) | **1,401**<sup>7</sup> / **1,278** | 0.17 / 0.24 | 9.9 | pass |
| llama.cpp, NVIDIA's Dory fork (last-loop KV; Q5_K_M) | 13.2 (13.2-13.2) | 13.2 (13.2-13.3) | 938 / 951 | 0.51 / 0.51 | 11.8 | pass |

All at 2 loops. Kapitsa's last-loop KV, its default, wrote 67.8 tok/s at T = 0.6 (68.9 greedy) at 10:28; the README
shows per-loop KV with speculative decoding. The "no support" cells are those found on the M4 Pro, at the same
versions ([Bolt](#nemotron-35-bolt), [Dory](#nemotron-dory10-4b-2026-10-19)).

Kapitsa's row above is the 2026-10-19 session, before the M5 prefill kernels
([#205](https://github.com/nace-ai/kapitsa/pull/205)). On 2026-10-22 03:26 (main at `180a650`, per-loop KV, the row's
setting) it read **618 / 543 tok/s** at 4K / 16K and decoded **72.4 / 80.5 tok/s** (T = 0.6 / greedy), with cached
follow-ups at 0.10 / 0.15 s and a peak footprint of 5.1 GB plus 0.65 GB of mapped clean pages; its last-loop KV
default wrote 71.7 (one of the five runs looped, flagged) / 72.5 at 4.3 GB,
[`dory-kapitsa-perloop-20261022-032657.json`](../bench/results/2026-10-22-m5pro-dory/README.md); the 5.1 GB peak is
1.0 GB above the row's 4.1, more than the 0.3-0.8 GB rise of the four models above, and is not attributed here. The
greedy cells are 1.06x the row's (79.8-82.3 against 75.1-76.7, disjoint), but that is one session each, not an
interleaved A/B, and at T = 0.6 the two overlap (72.4, 69.0-74.5, against 69.6, 61.1-71.5). Prefill is only 1.04x /
1.03x where the models above gained much more, because the QMT prefill kernels cover the Qwen3.5 backend and Dory's
own kernels are unchanged. Another app used 7% of the GPU through both prefill phases of the run (flagged), so those
two prefill cells are lower bounds; the decode phases ran quiet.

## KFlash trees (M4 Pro, 2026-10-21)

[#187](https://github.com/nace-ai/kapitsa/pull/187) made KFlash the default. Every drafter's candidates are now verified
as a best-first tree instead of one chain, in the same 8-row pass, sized by a budget measured on each Mac
([how KFlash works](kflash.md)). On the same MacBook Pro (M4 Pro, 48 GB, macOS 27.0.1, High Power mode, AC power), main
before it (617f6a2) against main after it (8b190f5), both Release builds, plus Splash 1.4.0 on the two models it runs.
The M1 Pro 16 GB ran the 4B. Result files, the configs and one line per A/B run:
[bench/results/2026-10-21-kflash-trees](../bench/results/2026-10-21-kflash-trees/).

| Model | Decode, T = 0.6: old → new main | Greedy: old → new main | Per-cycle A/B, greedy | Per-cycle A/B, T = 0.6 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | 185.7 → **200.8** (+8.1%) | 193.8 → **213.7** (+10.3%) | +2.1% (6 prompts) | +9.6% (3 prompts) |
| Qwen3.8-27B | 60.4 → **60.5** (+0.2%) | 61.1 → **64.4** (+5.5%) | +6.0% (3 prompts) | +2.8% (2 prompts) |
| Qwen3.6-35B-A3B | 168.1 → **160.6** (-4.4%) | 177.6 → **177.2** (-0.2%) | +0.6% (5 prompts) | — |
| Nemotron 3.5 Lightning 30B-A3B | 104.3 → **111.7** (+7.1%) | 115.7 → **113.7** (-1.7%) | — | — |
| Nemotron 3.5 Bolt 74B-A9B | 64.3 → **68.8** (+6.9%) | 69.4 → **71.2** (+2.6%) | — | — |

The replayed opencode session on the 35B: 51.7 s on old main, 51.4 s on new main (-0.6%; writing rate 138.7 → 140.8 tok/s, +1.5%).

Splash 1.4.0, the same protocol, its own defaults (DFlash2 draft, INT8 KV): the 35B 134.7 tok/s at T = 0.6 (140.9 greedy), the 27B 52.9 (57.6 greedy), and
the agent session on the 35B in 57.0 s (Kapitsa with KFlash 51.4 s). The M1 Pro's 4B, alternating builds (old, new, new,
old): 90.0 → 95.0 tok/s at T = 0.6 (+5.6%), 90.8 → 94.8 greedy (+4.4%).

- **The harness's protocol**, one run per build: 5 sampled and 5 greedy runs of the coding answer, plus tool calls.
  Prefill is left out because the trees don't touch it; the agent session runs on the 35B. One run per build is a
  quick reading: this Mac's runs of one build have spread 3-8% at T = 0.6.
- **The per-cycle A/B** (`ni_run`): five runs per prompt, interleaved old main, trees, chain, trees, old main. The prompts
  are a coding answer, a prose answer and 18-23K-token requests of the recorded agent session. On the dense models the
  trees' greedy output is identical to plain decoding on every prompt; old main's chain departs from it once (the 4B's
  r13, a near-tie). The new build's chain runs at old main's speed (−0.3% to +0.4% on the dense models).
- **Where the trees gain least: long agent turns.** There the chain already accepts 2.9-6.4 tokens per cycle. A tree
  cycle costs 5-8% more on the dense models and 10-16% more on the 35B, so those turns land between −5% and +3%.
  - On the dense models, the part that grows with the context is the tree-zone attention's fixed-size chunks: about 3x
    the chain's partial results at 19K tokens. A repartition by position is proposed in the
    [results' README](../bench/results/2026-10-21-kflash-trees/README.md#why-long-agent-turns-gain-less).
  - On the 35B, the trees verify more rows than the chain did, and each row routes its own experts.

## Your Mac: `kapitsa bench`

`kapitsa bench` runs this protocol for Kapitsa alone, from the `kapitsa` binary: no Python, no configs, no weights to
place by hand (the catalog's models, downloaded with their drafters). It sends the same requests as `bench.py`, starts
the server as `compare_engines.py` does, and flags runs alike, so its numbers compare with the tables above; it
records the Mac and its conditions, and with `--submit` adds the result to the
[community results](../bench/community/RESULTS.md), checked as [bench/community](../bench/community/README.md) says.
The [usage guide](usage.md#benchmarking-this-mac) has its options.

## Run benchmarks locally

Everything above can be rerun with the harness in [bench/harness](../bench/harness/README.md) (Python 3, standard
library only):

1. Build Kapitsa and install the engines you want to compare; the harness README says how for each, with the
   settings used here.
2. Download the weights into `~/models` (the commands, with pinned revisions, are in the harness README).
3. Point the harness at them: `cd bench/harness && cp local.env.example local.env`, then edit the paths.
4. Check the plan without starting anything: `python3 compare_engines.py configs/*.json --dry-run`.
5. Run, one engine at a time: `python3 compare_engines.py configs/kapitsa.json configs/llama.cpp.json --only 'q36-35b-*'`.
   Each configuration takes 2 to 20 minutes and holds the GPU lock throughout.
6. Make the tables: `python3 compare_table.py`.

For numbers you can compare across engines, run them all in one session with the same power mode, plugged in, with
browsers and other GPU users closed; the harness flags runs that other apps disturbed. Results depend on the Mac:
other chips, memory sizes and macOS versions will give different numbers.
