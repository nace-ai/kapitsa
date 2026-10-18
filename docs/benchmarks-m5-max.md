# Kapitsa on M5 Max, 128 GB

Measured on 2026-10-17 (Qwen3.6 and Qwen3.8-27B) and 2026-10-18 (Nemotron 3.5 Super-VL and Lightning, Qwen3.8-Flash-Next) with the stock benchmark harness. **Every run passed correctness checks but was flagged for background GPU contention.** These are measurements with desktop applications open, not an idle-machine comparison or a maximum-performance claim. Super-VL, a 96 GB+ model, Lightning and Flash-Next, a 128 GB model, were also measured against every other engine that runs them on a Mac ([Super-VL](#nemotron-35-super-vl-against-the-other-engines), [Lightning](#nemotron-35-lightning-against-the-other-engines), [Flash-Next](#qwen38-flash-next-against-the-other-engines)).

## Results

| Metric | Qwen3.6-35B-A3B | Qwen3.8-27B | Nemotron 3.5 Super-VL 120B-A12B | Nemotron 3.5 Lightning 30B-A3B | Qwen3.8-Flash-Next 180B-A7B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sampled decode, median (range), tok/s | 249.8 (236.7–300.2) | 72.9 (65.7–83.5) | 84.5 (78.1–93.2) | 238.7 (231.1–243.4) | 101.8 (100.6–109.0) |
| Greedy decode, median (range), tok/s | 297.4 (275.3–300.7) | 83.7 (79.6–92.7) | 87.0 (76.3–90.0) | 218.1 (211.2–258.2) | 107.6 (103.7–109.0) |
| Cold prefill, 4K, tok/s | 2,224 | 430 | 627 | 1,899 | 860 |
| Cold prefill, 16K, tok/s | 2,011 | 400 | 509 | 1,861 | 850 |
| Cached TTFT, 4K, ms | 34.9 | 121.6 | 220 | 73 | 82 |
| Cached TTFT, 16K, ms | 40.2 | 135.3 | 260 | 88 | 75 |
| Peak process-tree memory, GiB | 21.0 | 19.4 | 70.4 (+1.0 mapped) | 18.6 (+0.7 mapped) | 70.5 (+5.5 mapped) |
| Tool-call checks | Pass | Pass | Pass | Pass | Pass |
| Invalid measurements | None | None | None | None | None |
| GPU contention | Flagged | Flagged | Flagged (prefill phases) | Flagged (prefill phases) | Flagged (prefill phases) |

Super-VL is NVIDIA's NVFP4 checkpoint with its LiLiCorr drafter, Kapitsa's default for it (`kapitsa pull super-vl`); two of its ten decode answers repeated themselves (flagged by the harness, kept), and its prompts were 4,186 / 16,515 tokens. Lightning is NVIDIA's NVFP4 checkpoint with its DSpark drafter, Kapitsa's default for it (`kapitsa pull lightning`); none of its ten decode answers repeated itself, and its prompts were 4,184 / 16,516 tokens. Flash-Next is Nace AI's MLX pack (`kapitsa pull flash-next`) with the model's own MTP head at depth 2, Kapitsa's default for it, build 95fbcd6 of `opt/flash-next`; none of its ten decode answers repeated itself, and its prompts were 4,132 / 16,287 tokens. For the Qwen models: each decode median has five valid trials. All 20 decode requests ended at exactly 512 tokens with `finish_reason=length`, with no thinking leak or repetition-loop flag. Cold prompts reported zero cached tokens and stayed within the harness's 1% token-count tolerance. The 35B prompt lengths were 4,135 / 16,287 tokens; the 27B lengths were 4,134 / 16,285. Each prefill cell is one cold request; each cached TTFT cell is the median of three follow-ups. This session contains one full run per model; the ranges above are within-run ranges, not confidence intervals or independent-session variation.

## Background load and runtime checks

| Condition | Qwen3.6-35B-A3B | Qwen3.8-27B | Super-VL | Lightning | Flash-Next |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mean foreign GPU activity | 10.11% | 17.67% | 13.95% | 6.41% | 6.27% |
| Maximum foreign GPU activity | 28.71% | 35.09% | 64.19% | 11.28% | 15.17% |
| Quiet wait before measurement | 187.2 s | 18.8 s | 300 s (not quiet; measured anyway) | 120 s (not quiet; measured anyway) | 120 s (not quiet; measured anyway) |
| Ambient GPU activity, mean | 0% | 0% | 0% | 0% | 0% |

The monitor identified `cmux` and `com.apple.WebKit` as the main foreign GPU users and flagged every measured phase. Both models reached the harness's initial quiet window, but background activity resumed during measurement. The default 5% contention threshold and ambient-process list were unchanged. Keep this qualification attached when quoting these scores.

The machine already had approximately 23.9 GiB of swap in use before the session. There were **zero additional swap-outs** during every measured phase. These swap counters are system-wide; they are not model memory usage. Both servers shut down successfully, left no surviving server processes, and released their memory after the tests.

The 27B run used the default GPU + Neural Engine prefill split. Its background compilation completed in 42.6 seconds, and its built-in startup check reported 0.79% deviation relative to the Neural Engine portion of the GPU reference, below the engine's 10% cutoff. This is a runtime numerical check, not a model-quality evaluation. The split was enabled before the timed 4K and 16K prefill requests. The 35B MoE prefill runs on the GPU.

## Qwen3.8-Flash-Next against the other engines

Measured on 2026-10-18 with [configs/flash-next.json](../bench/harness/configs/flash-next.json): every engine that runs
Flash-Next on a Mac, each at its latest release with its own best settings, every configuration run twice; the table has
the second valid runs. Kapitsa, mlx-vlm and oMLX read the same folder (Nace AI's MLX pack); oMLX also ran its author's
oQ4e quantization, llama.cpp ggml-org's IQ4_NL GGUF, Ollama its NVFP4 MLX tag and its Q4_K_M GGUF tag. Details,
footnotes, recipes and the weights:
[docs/benchmarks.md](benchmarks.md#qwen38-flash-next-180b-a7b-m5-max-128-gb-2026-10-18).

| Engine (setting) | Decode T = 0.6, tok/s | Greedy, tok/s | Prefill 4K / 16K, tok/s | Cached TTFT 4K / 16K, s | Peak memory, GiB | Tools |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** 95fbcd6 (MTP, depth 2) | 101.8 (100.6–109.0) | 107.6 (103.7–109.0) | 860 / 850 | 0.08 / 0.07 | 70.5 + 5.5 mapped | Pass |
| Kapitsa (no speculation) | 84.9 (84.6–85.5) | 84.8 (84.3–85.7) | 862 / 868 | 0.08 / 0.08 | 70.3 + 5.2 mapped | Pass |
| oMLX 0.7.1rc1 + MTP (fixed depth 3), Nace pack | 106.5 (103.2–112.1) | 106.4 (105.2–108.3) | 1,247 / 1,316 | 0.26 / 0.42 | 103.9 | Pass |
| oMLX, Nace pack | 59.5 (59.3–60.2) | 60.0 (59.7–60.2) | 1,339 / 1,394 | 0.25 / 0.38 | 103.4 | Pass |
| oMLX + MTP (fixed depth 2), oQ4e ⚠ | 105.5 (95.9–109.4) | 98.7 (95.9–101.5) | 1,395 / 1,383 | 0.28 / 0.45 | 104.0 | Pass |
| oMLX, oQ4e ⚠ | 69.7 (64.6–71.3) | 66.8 (66.4–68.7) | 1,521 / 1,691 | 0.25 / 0.34 | 103.6 | Pass |
| llama.cpp 0.7.0 + MTP (n-max 4), IQ4_NL | 96.1 (84.9–98.2) | 94.6 (92.1–99.2) | 842 / 890 | 0.25 / 0.24 | 83.2 + 2.3 mapped | Fail |
| llama.cpp, IQ4_NL | 46.4 (45.9–46.6) | 45.1 (42.6–45.7) | 848 / 913 | 0.23 / 0.23 | 79.3 + 2.1 mapped | Pass |
| mlx-vlm 0.7.8 + MTP (block 4), Nace pack ⚠ | 76.8 (70.1–77.0) | 77.0 (72.4–77.5) | 1,030 / 1,049 | 0.38 / 0.49 | 77.0 + 5.3 mapped | Pass |
| mlx-vlm, Nace pack ⚠ | 53.0 (47.2–53.9) | 52.4 (48.1–53.6) | 948 / 1,031 | 0.43 / 0.55 | 74.3 + 5.0 mapped | Pass |
| Ollama 0.40.6 MLX runner + MTP, NVFP4 | 60.0 (57.8–64.9) | 62.8 (59.5–68.8) | fails (out of GPU memory) | — | 108.4 | Pass |
| Ollama GGUF runner, Q4_K_M | 50.1 (49.3–50.2) | 48.3 (48.1–48.8) | 705 / 672 | 0.22 / 0.24 | 88.0 + 2.2 mapped | Pass |

⚠: contention flagged in the decode phases (a Google Chrome helper used 7-16% of the GPU on average during those runs);
the other rows are flagged at most in their prefill phases. Ollama's MLX runner fails every 4K and 16K prompt (out of
Metal working-set memory); llama.cpp + MTP's second run fails one tool-call check (its first passed). oMLX and Ollama's
MLX runner hold the 32 GB n-gram table in memory; Kapitsa, mlx-vlm and llama.cpp map it from the SSD. Not run: mlx-lm
(no release supports `qwen4_exp`).

## Nemotron 3.5 Super-VL against the other engines

Measured on 2026-10-18 with [configs/super-vl.json](../bench/harness/configs/super-vl.json): every engine that runs
Super-VL on a Mac, each at its latest release with its own best settings (mlx-vlm at an unreleased build, git main
73d366e, because 0.7.8 does not load the model), every configuration run twice; the table has the second runs. oMLX and
mlx-lm cannot read NVIDIA's checkpoint and ran the text-only MLX 4-bit conversion; llama.cpp ran ggml-org's Q4_0 GGUF.
Details, footnotes and the weights: [docs/benchmarks.md](benchmarks.md#nemotron-35-super-vl-120b-a12b-m5-max-128-gb-2026-10-18).

| Engine (setting) | Decode T = 0.6, tok/s | Greedy, tok/s | Prefill 4K / 16K, tok/s | Cached TTFT 4K / 16K, s | Peak memory, GiB | Tools |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** (LiLiCorr drafter) | **84.5** (78.1–93.2) | **87.0** (76.3–90.0) | 627 / 509 | 0.22 / 0.26 | **70.4** + 1.0 mapped | Pass |
| Kapitsa (no drafter) | 46.4 (45.0–47.4) | 46.5 (45.3–47.3) | 617 / 501 | 0.20 / 0.25 | 67.7 + 1.0 mapped | Pass |
| llama.cpp 0.7.0 + MTP (n-max 3) | 65.5 (63.0–69.7) | 66.1 (64.7–67.5) | 833 / 882 | 0.22 / 0.20 | 81.9 | Pass |
| llama.cpp + MTP (n-max 2) | 58.8 (56.5–61.8) | 59.4 (58.7–60.0) | 830 / 877 | 0.22 / 0.20 | 81.7 | Pass |
| llama.cpp + MTP (n-max 1) | 49.6 (48.6–52.0) | 49.4 (48.9–49.6) | 818 / 874 | 0.23 / 0.21 | 81.6 | Pass |
| llama.cpp (no speculation) | 35.9 (35.0–37.1) | 35.4 (35.1–35.6) | 764 / 821 | 0.19 / 0.20 | 78.5 | Pass |
| llama.cpp + LiLiCorr (n-max 7) | 26.0 (23.4–31.1) | 31.0 (29.9–31.4) | 681 / 746 | 0.24 / 0.32 | 86.2 | Pass |
| oMLX 0.7.1rc1, MLX 4-bit | 51.8 (49.7–52.1) | 50.4 (50.1–51.0) | 989 / 1,049 | 0.33 / 0.43 | 74.8 | Pass |
| mlx-lm 0.32.2, MLX 4-bit | 45.5 (44.7–45.8) | 45.4 (44.7–45.6) | 885 / 966 | 0.20 / 0.25 | 75.6 | Pass |
| mlx-vlm (git main) + MTP (block 4) | 46.3 (43.9–48.5) | 49.1 (48.9–49.6) | 756 / fails (out of GPU memory) | 0.52 / fails | 108.0 | Pass |
| mlx-vlm (git main) | 30.7 (30.3–30.8) | 30.9 (30.7–31.0) | 710 / 812 | 0.53 / 0.61 | 99.3 | Pass |

Every row except oMLX's is flagged for contention: a desktop `com.apple.WebKit` process used 3-5% of the GPU
throughout and more during the engines' prefill phases (most during Kapitsa's, up to 64%). Not run: mlx-lm and oMLX on
NVIDIA's checkpoint and mlx-vlm 0.7.8 (none loads model type `nemotron_h_omni`), llama.cpp on the NVFP4 GGUF (no Metal
kernels). On the 48 GB M4 Pro, Kapitsa refuses Super-VL before downloading it: its weights alone (70.1 GB) exceed the
38.6 GB of GPU memory there.

## Nemotron 3.5 Lightning against the other engines

Measured on 2026-10-18 with [configs/lightning.json](../bench/harness/configs/lightning.json): every engine that runs
Lightning on a Mac, each at its latest release with its own best settings (mlx-vlm's MTP row at an unreleased build,
git main 73d366e, because its Nemotron-H MTP drafter came after 0.7.8), every configuration run twice; the table has
the second valid runs. oMLX and mlx-lm cannot read NVIDIA's checkpoint and ran MLX 4-bit conversions; llama.cpp ran
ggml-org's Q4_0 GGUF. Details, footnotes and the weights:
[docs/benchmarks.md](benchmarks.md#nemotron-35-lightning-30b-a3b-m5-max-128-gb-2026-10-18).

| Engine (setting) | Decode T = 0.6, tok/s | Greedy, tok/s | Prefill 4K / 16K, tok/s | Cached TTFT 4K / 16K, s | Peak memory, GiB | Tools |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| **Kapitsa** (DSpark drafter) | **238.7** (231.1–243.4) | **218.1** (211.2–258.2) | 1,899 / 1,861 | 0.07 / 0.09 | **18.6** + 0.7 mapped | Pass |
| Kapitsa (no drafter) | 160.1 (160.1–160.3) | 160.3 (160.1–160.7) | 1,850 / 1,802 | 0.08 / 0.09 | 17.9 + 0.7 mapped | Pass |
| oMLX 0.7.1rc1 + MTP (depth 2), MLX 4-bit | 189.9 (183.5–196.2) | 199.3 (188.7–202.4) | 3,521 / 3,613 | 0.12 / 0.15 | 21.6 | Pass |
| oMLX, MLX 4-bit | 164.7 (161.8–166.2) | 166.6 (164.8–167.6) | **3,606 / 3,756** | 0.11 / 0.14 | 20.9 | Pass |
| llama.cpp 0.7.0 + MTP (n-max 3) | 159.1 (147.1–176.1) | 162.0 (150.6–170.0) | 2,835 / 2,812 | 0.08 / 0.09 | 24.2 | Pass |
| llama.cpp + MTP (n-max 2) | 156.3 (148.2–167.0) | 163.0 (151.8–168.4) | 2,838 / 2,841 | 0.08 / **0.08** | 24.1 | Pass |
| llama.cpp + MTP (n-max 1) | 142.4 (140.5–148.5) | 142.3 (140.7–146.3) | 2,831 / 2,840 | 0.07 / 0.08 | 24.1 | Pass |
| llama.cpp (no speculation) | 116.8 (115.5–117.0) | 113.2 (112.8–114.8) | 2,662 / 2,823 | **0.07** / 0.08 | 22.4 | Pass |
| llama.cpp + DSpark (n-max 3) | 85.1 (84.9–86.5) | 88.4 (80.0–92.3) | 2,572 / 2,650 | 0.08 / 0.09 | 25.3 | Pass |
| mlx-lm 0.32.2, MLX 4-bit | 139.3 (135.5–140.7) | 145.6 (145.0–146.9) | 3,365 / 3,422 | 0.08 / 0.11 | 21.1 | Pass |
| mlx-vlm 0.7.8 + DSpark (block 6) | 137.4 (131.3–144.4) | 146.3 (120.6–149.5) | 2,400 / 2,722 | 0.28 / 0.31 | 29.3 | Pass |
| mlx-vlm (git main) + MTP (block 4) | 121.5 (107.1–132.2) | 122.5 (116.1–132.6) | 2,524 / 2,842 | 0.28 / 0.32 | 46.5 | Pass |
| mlx-vlm 0.7.8 | 108.9 (108.5–109.0) | 111.0 (109.4–111.4) | 2,557 / 2,939 | 0.29 / 0.31 | 27.7 | Pass |

Kapitsa writes fastest (1.26× oMLX + MTP sampled, 1.09× greedy) and uses the least memory; it prefills slowest (about
half of oMLX's rate) and its follow-ups tie llama.cpp's. Every row except oMLX + MTP's and mlx-vlm's plain one is
flagged for contention: a desktop `com.apple.WebKit` process used 5–7% of the GPU throughout, more during the prefill
phases. Not run: mlx-lm and oMLX on NVIDIA's checkpoint (ModelOpt `MIXED_PRECISION`), oMLX or mlx-lm with a drafter
(no Nemotron-H drafter path), mlx-vlm 0.7.8 with MTP. Lightning also fits the 48 GB M4 Pro; it has not been measured
there.

## Machine and method

Measured on 2026-10-17 on a MacBook Pro (`Mac17,6`) with an Apple M5 Max, 18 CPU cores, 40 GPU cores and 128 GiB unified memory. macOS 26.4 (25E246), AC power, Automatic power mode. The battery was charging; the internal display and an external 3840 × 1600 display at 144 Hz were attached. Normal desktop applications remained open.

The Release build uses Apple Clang 21.0.0 and CMake 4.3.3, built from `c36bfe3759ac5a2127a7abdc06ecfb88f1028dd8`. The committed machine record includes the executable SHA-256. These measurements use the existing harness and stock Kapitsa configs: 4-bit affine weights (group size 64), DFlash2, 32,768-token context and one conversation slot. Both models and their companion files were downloaded using `kapitsa pull 35b 27b`, which pins the revisions in `src/hub/catalog.cpp`.

Decode is the median of five 512-token coding answers at each temperature, measured over the middle of the streamed response. Sampling uses temperature 0.6, top_p 0.95 and top_k 20; greedy uses temperature 0. Cold prefill uses the frozen corpus with a nonce to prevent prefix reuse. Cached TTFT is the median of three append-only follow-up turns. Text benchmarks disable thinking. Tool checks exercise streaming, non-streaming and tool-result round trips with thinking on and off. Memory is the peak physical footprint of the server process tree, in GiB.

This is an inference performance measurement on one laptop. It does not measure model task accuracy. Only Super-VL, Lightning and Flash-Next were compared with other engines on this laptop. The separate M4 Pro report used a different machine and macOS version; these results do not establish an isolated hardware speedup.

## Recorded artifacts

[Raw results and logs](../bench/results/2026-10-17-m5-max/) include every run in this session:

* [35B result](../bench/results/2026-10-17-m5-max/q36-35b-kapitsa-20261017-163120.json)
* [27B result](../bench/results/2026-10-17-m5-max/q38-27b-kapitsa-20261017-163524.json)
* [Machine and build record](../bench/results/2026-10-17-m5-max/machine.json)
* Super-VL, all engines: [results, logs, machine record and model manifests](../bench/results/2026-10-18-m5-max/) (Kapitsa built from `opt/super-vl`, d6fb99d)
* Lightning, all engines: [results, logs, machine record and model manifests](../bench/results/2026-10-18-m5-max-lightning/) (Kapitsa built from `opt/lightning`, 80e4608)
* Flash-Next, all engines: [results, logs, machine record and model manifests](../bench/results/2026-10-18-m5-max-flash-next/) (Kapitsa built from `opt/flash-next`, 95fbcd6; earlier builds 8912f1b and 62ab7ed in `earlier-builds/`)
* [Pinned model manifests with file sizes and SHA-256 hashes](../bench/results/2026-10-17-m5-max/model-manifests/)
* [Harness transcript](../bench/results/2026-10-17-m5-max/harness.log) and [server logs](../bench/results/2026-10-17-m5-max/logs/)

Local timestamps use America/Los_Angeles (UTC−07:00); the manifest timestamps use UTC. The measurements ran on a clean checkout of the recorded source commit. Published artifacts replace the checkout path with `<repo>`, the run directory with `<run-root>`, and the home directory with `~`. No numerical results, generated answers, validity checks or contention flags were removed. The model files themselves are not included.

To regenerate the table from the committed results:

```bash
python3 bench/harness/compare_table.py bench/results/2026-10-17-m5-max --out /tmp/kapitsa-m5-table
```

## Reproduce

From a checkout at the source revision above, with the Xcode Command Line Tools and CMake installed:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target kapitsa -j 6
caffeinate -i build/kapitsa pull 35b 27b

# Reuse the CLI cache with the stock harness folder layout (no duplicate weights).
export BENCH_MODELS="$HOME/.cache/kapitsa/benchmark-models"
mkdir -p "$BENCH_MODELS/drafters"
ln -s "$HOME/.cache/kapitsa/models/qwen3.6-35b-a3b-4bit" "$BENCH_MODELS/qwen3.6-35b-a3b-4bit"
ln -s "$HOME/.cache/kapitsa/models/qwen3.8-27b-4bit" "$BENCH_MODELS/qwen3.8-27b-4bit-mtp"
ln -s "$HOME/.cache/kapitsa/models/qwen3.6-35b-a3b-4bit/dflash2" "$BENCH_MODELS/drafters/Qwen3.6-35B-A3B-DFlash2"
ln -s "$HOME/.cache/kapitsa/models/qwen3.8-27b-4bit/dflash2" "$BENCH_MODELS/drafters/Qwen3.8-27B-DFlash2"

python3 bench/harness/compare_engines.py bench/harness/configs/kapitsa.json \
  --only 'q36-35b-kapitsa,q38-27b-kapitsa' --out-dir /tmp/kapitsa-m5-run --dry-run
caffeinate -i python3 -u bench/harness/compare_engines.py bench/harness/configs/kapitsa.json \
  --only 'q36-35b-kapitsa,q38-27b-kapitsa' --out-dir /tmp/kapitsa-m5-run
python3 bench/harness/compare_table.py /tmp/kapitsa-m5-run/results --out /tmp/kapitsa-m5-table
```

The symlink commands assume that these four links do not already exist. The harness starts one local server at a time, warms it up, waits for a quiet GPU, measures and stops its own server. Keep other GPU workloads idle while measuring. The download requires approximately 42 GB of disk space for both models and their companion files.
