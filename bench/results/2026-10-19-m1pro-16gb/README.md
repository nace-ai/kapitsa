# Results, 2026-10-19: MacBook Pro M1 Pro, 16 GB

The result files behind [the M1 Pro section](../../../docs/benchmarks.md#macbook-pro-m1-pro-16-gb-2026-10-19) of
docs/benchmarks.md, written by [bench/harness](../../harness/README.md) on a MacBook Pro (14-inch, 2021),
`MacBookPro18,3`: M1 Pro with an 8-core CPU and a 14-core GPU, 16 GB, macOS 27.0.1 (26A434), on AC power (Automatic
power mode). Times are local (UTC+5).

## Qwen3.5-4B at 4-bit, short runs ([tuning](tuning/))

One hold of the GPU lock, 08:48-09:30, Kapitsa first, and LM Studio in another at 09:35. Each config ran its decode
phase only: `--skip greedy,prefill,tools,agent --bench-args "--decode-runs 3"`, the T = 0.6 coding answer, 3 runs of
512 tokens. The configs are the harness's `q35-4b-*` and the settings in [tuning/m1-tuning.json](tuning/m1-tuning.json)
(pass it after `configs/*.json`). These runs chose each engine's setting for the README's cells; the full protocol
follows.

| Result file | Started | Run | Decode, T = 0.6 (tok/s): median (min-max) | Flags |
| --- | --- | --- | ---: | --- |
| `q35-4b-kapitsa-20261019-084834.json` | 08:49 | Kapitsa (DFlash drafter) | 85.4 (72.4-95.6) | — |
| `q35-4b-llamacpp-20261019-084935.json` | 08:50 | llama.cpp | 38.2 (38.1-38.2) | — |
| `q35-4b-llamacpp-mtp1-20261019-085102.json` | 08:51 | llama.cpp + MTP (n-max 1) | 34.5 (33.7-34.9) | — |
| `q35-4b-llamacpp-mtp2-20261019-085230.json` | 08:53 | llama.cpp + MTP (n-max 2) | 37.7 (37.1-41.0) | — |
| `q35-4b-llamacpp-mtp3-20261019-085352.json` | 08:54 | llama.cpp + MTP (n-max 3) | 40.8 (38.8-42.7) | — |
| `q35-4b-llamacpp-mtp6-20261019-085513.json` | 08:55 | llama.cpp + MTP (n-max 6) | 39.2 (35.7-42.8) | — |
| `q35-4b-mlxlm-20261019-085635.json` | 08:57 | mlx-lm | 47.4 (46.0-47.5) | — |
| `q35-4b-mlxserve-mtp-20261019-085756.json` | 08:59 | mlx-serve + MTP | 61.1 (60.1-62.7) | — |
| `q35-4b-mlxserve-dflash-20261019-085937.json` | 09:00 | mlx-serve + DFlash (z-lab drafter) | 55.6 (54.2-56.0) | — |
| `q35-4b-mlxvlm-mtp-20261019-090121.json` | 09:02 | mlx-vlm server + MTP | 36.6 (35.5-36.9) | — |
| `q35-4b-mtplx-20261019-090335.json` | 09:05 | MTPLX (sustained profile, depth 1) | 41.6 (41.5-41.7) | — |
| `q35-4b-ollama-mtp-20261019-090618.json` | 09:06 | Ollama GGUF + MTP (shipped depth 1) | 33.1 (32.9-34.3) | — |
| `q35-4b-ollama-mlx-q4-20261019-090748.json` | 09:08 | Ollama MLX runner (MLX 4-bit + MTP head, adaptive depth) | 56.0 (55.8-56.8) | — |
| `q35-4b-omlx-best-20261019-090931.json` | 09:11 | oMLX + MTP (Lightning MTP, MTPLX's head) | 56.5 (55.4-57.2) | — |
| `q35-4b-omlx-dflash-b4-20261019-091139.json` | 09:12 | oMLX + DFlash (z-lab drafter 4-bit, block 4, adaptive verify) | 50.8 (50.5-51.1) | — |
| `q35-4b-omlx-dflash-b5-20261019-091336.json` | 09:14 | oMLX + DFlash (z-lab drafter 4-bit, block 5, adaptive verify) | 44.1 (43.9-46.5) | decode: repetition loop (distinct-4-gram 0.4571); decode: repetition loop (distinct-4-gram 0.4814) |
| `q35-4b-mlxserve-mtp-d2-20261019-091542.json` | 09:16 | mlx-serve + MTP (max depth 2) | 62.4 (61.8-62.8) | — |
| `q35-4b-mlxserve-mtp-d4-20261019-091715.json` | 09:18 | mlx-serve + MTP (max depth 4) | 60.8 (58.5-69.3) | — |
| `q35-4b-mlxserve-dflash-b5-20261019-091850.json` | 09:19 | mlx-serve + DFlash (z-lab drafter, block 5) | 52.0 (51.1-55.4) | — |
| `q35-4b-mtplx-d2-20261019-092029.json` | 09:22 | MTPLX (sustained profile, depth 2) | 35.4 (34.6-36.0) | disk: 9.5 GiB free at start, under the 10 GiB below which mtplx keeps no prompt cache on disk |
| `q35-4b-mlxvlm-mtp-b2-20261019-092313.json` | 09:24 | mlx-vlm server + MTP (block 2) | 37.2 (36.4-37.8) | decode: repetition loop (distinct-4-gram 0.4711) |
| `q35-4b-mlxvlm-mtp-b4-20261019-092521.json` | 09:29 | mlx-vlm server + MTP (block 4) | 6.9 (6.7-6.9) | — |
| `q35-4b-lmstudio-20261019-093551.json` | 09:37 | LM Studio (MLX engine) | 44.6 (44.5-44.9) | decode: completion_tokens=511 != 512; disk: 9.9 GiB free at start, under the 10 GiB below which lmstudio keeps no prompt cache on disk |

- Kapitsa is the 0.1.36 release binary (954fca0). Its file records the version `cdc1ca5`: the harness takes the version
  of a binary that sits in a git checkout from that checkout, and the release was installed in `/opt/homebrew/bin`,
  which is Homebrew's own checkout. Later runs use a copy outside any checkout.
- The session held the GPU lock around each hold (`lockf -k /tmp/kapitsa-gpu.lock`), so the harness did not take it per
  run (`NACE_GPU_LOCK_HELD=1`). The desktop app that drove the runs counts as ambient (`BENCH_GPU_AMBIENT`); nothing was
  flagged for contention.
- Local paths in the files are replaced by placeholders: `<models>` the weights folder ($BENCH_MODELS), `<out>` the
  harness output folder, `<kapitsa-checkout>` the Kapitsa checkout, `<kapitsa-models>` Kapitsa's model cache, `<home>`
  the home folder, `<scratch>` the session's scratch folder, and the engines' install folders by name
  (`<omlx-venv>`, `<mlx-lm-venv>`, `<mlx-vlm-venv>`, `<mtplx-venv>`, `<llama.cpp>`, `<mlx-serve>`, `<ollama>`, `<lmstudio>`).
