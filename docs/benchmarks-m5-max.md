# Kapitsa on M5 Max, 128 GB

Measured on 2026-10-17 with the stock benchmark harness. **Both runs passed correctness checks but were flagged for background GPU contention.** These are measurements with desktop applications open, not an idle-machine comparison or a maximum-performance claim.

## Results

| Metric | Qwen3.6-35B-A3B | Qwen3.8-27B |
| --- | ---: | ---: |
| Sampled decode, median (range), tok/s | 249.8 (236.7–300.2) | 72.9 (65.7–83.5) |
| Greedy decode, median (range), tok/s | 297.4 (275.3–300.7) | 83.7 (79.6–92.7) |
| Cold prefill, 4K, tok/s | 2,224 | 430 |
| Cold prefill, 16K, tok/s | 2,011 | 400 |
| Cached TTFT, 4K, ms | 34.9 | 121.6 |
| Cached TTFT, 16K, ms | 40.2 | 135.3 |
| Peak process-tree memory, GiB | 21.0 | 19.4 |
| Tool-call checks | Pass | Pass |
| Invalid measurements | None | None |
| GPU contention | Flagged | Flagged |

Each decode median has five valid trials. All 20 decode requests ended at exactly 512 tokens with `finish_reason=length`, with no thinking leak or repetition-loop flag. Cold prompts reported zero cached tokens and stayed within the harness's 1% token-count tolerance. The 35B prompt lengths were 4,135 / 16,287 tokens; the 27B lengths were 4,134 / 16,285. Each prefill cell is one cold request; each cached TTFT cell is the median of three follow-ups. This session contains one full run per model; the ranges above are within-run ranges, not confidence intervals or independent-session variation.

## Background load and runtime checks

| Condition | Qwen3.6-35B-A3B | Qwen3.8-27B |
| --- | ---: | ---: |
| Mean foreign GPU activity | 10.11% | 17.67% |
| Maximum foreign GPU activity | 28.71% | 35.09% |
| Quiet wait before measurement | 187.2 s | 18.8 s |
| Ambient GPU activity, mean | 0% | 0% |

The monitor identified `cmux` and `com.apple.WebKit` as the main foreign GPU users and flagged every measured phase. Both models reached the harness's initial quiet window, but background activity resumed during measurement. The default 5% contention threshold and ambient-process list were unchanged. Keep this qualification attached when quoting these scores.

The machine already had approximately 23.9 GiB of swap in use before the session. There were **zero additional swap-outs** during every measured phase. These swap counters are system-wide; they are not model memory usage. Both servers shut down successfully, left no surviving server processes, and released their memory after the tests.

The 27B run used the default GPU + Neural Engine prefill split. Its background compilation completed in 42.6 seconds, and its built-in startup check reported 0.79% deviation relative to the Neural Engine portion of the GPU reference, below the engine's 10% cutoff. This is a runtime numerical check, not a model-quality evaluation. The split was enabled before the timed 4K and 16K prefill requests. The 35B MoE prefill runs on the GPU.

## Machine and method

Measured on 2026-10-17 on a MacBook Pro (`Mac17,6`) with an Apple M5 Max, 18 CPU cores, 40 GPU cores and 128 GiB unified memory. macOS 26.4 (25E246), AC power, Automatic power mode. The battery was charging; the internal display and an external 3840 × 1600 display at 144 Hz were attached. Normal desktop applications remained open.

The Release build uses Apple Clang 21.0.0 and CMake 4.3.3, built from `c36bfe3759ac5a2127a7abdc06ecfb88f1028dd8`. The committed machine record includes the executable SHA-256. These measurements use the existing harness and stock Kapitsa configs: 4-bit affine weights (group size 64), DFlash2, 32,768-token context and one conversation slot. Both models and their companion files were downloaded using `kapitsa pull 35b 27b`, which pins the revisions in `src/hub/catalog.cpp`.

Decode is the median of five 512-token coding answers at each temperature, measured over the middle of the streamed response. Sampling uses temperature 0.6, top_p 0.95 and top_k 20; greedy uses temperature 0. Cold prefill uses the frozen corpus with a nonce to prevent prefix reuse. Cached TTFT is the median of three append-only follow-up turns. Text benchmarks disable thinking. Tool checks exercise streaming, non-streaming and tool-result round trips with thinking on and off. Memory is the peak physical footprint of the server process tree, in GiB.

This is an inference performance measurement of Kapitsa on one laptop. It does not measure model task accuracy or compare other engines on this laptop. The separate M4 Pro report used a different machine and macOS version; these results do not establish an isolated hardware speedup.

## Recorded artifacts

[Raw results and logs](../bench/results/2026-10-17-m5-max/) include every run in this session:

* [35B result](../bench/results/2026-10-17-m5-max/q36-35b-kapitsa-20261017-163120.json)
* [27B result](../bench/results/2026-10-17-m5-max/q38-27b-kapitsa-20261017-163524.json)
* [Machine and build record](../bench/results/2026-10-17-m5-max/machine.json)
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
