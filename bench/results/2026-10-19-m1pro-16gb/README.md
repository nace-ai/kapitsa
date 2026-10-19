# Results, 2026-10-19: MacBook Pro M1 Pro, 16 GB

The result files behind [the M1 Pro section](../../../docs/benchmarks.md#macbook-pro-m1-pro-16-gb-2026-10-19) of
docs/benchmarks.md, written by [bench/harness](../../harness/README.md) on a MacBook Pro (14-inch, 2021),
`MacBookPro18,3`: M1 Pro with an 8-core CPU and a 14-core GPU, 16 GB, macOS 27.0.1 (26A434), on AC power (Automatic
power mode). Times are local (UTC+5).

## Qwen3.5-4B at 4-bit, the full protocol (the folder itself)

One hold of the GPU lock, 10:09-10:48, Kapitsa 0.1.36 (the release, from a copy outside any git checkout) first and
last, each engine in the setting the short runs chose, the harness's full protocol (`--skip agent` is implied: the 4B
has no recorded agent session). The engine state each run left was deleted after it (the 16 GB Mac's disk).

| Result file | Started | Run | Decode, T = 0.6 / greedy (tok/s) | Flags and invalid phases |
| --- | --- | --- | ---: | --- |
| `q35-4b-kapitsa-20261019-100914.json` | 10:09 | Kapitsa (DFlash drafter) | 85.5 / 88.0 | decode: repetition loop (distinct-4-gram 0.4747) |
| `q35-4b-omlx-best-20261019-101234.json` | 10:13 | oMLX + MTP (Lightning MTP, MTPLX's head) | 54.6 / 57.4 | decode: repetition loop (distinct-4-gram 0.4724); decode: repetition loop (distinct-4-gram 0.4733) |
| `q35-4b-mlxserve-mtp-d2-20261019-101733.json` | 10:18 | mlx-serve + MTP (max depth 2) | 62.3 / 62.9 | — |
| `q35-4b-ollama-mlx-q4-20261019-102204.json` | 10:23 | Ollama MLX runner (MLX 4-bit + MTP head, adaptive depth) | 55.3 / 57.4 | decode: repetition loop (distinct-4-gram 0.4761); prefill_4000: 6964 pages swapped out during the phase; prefill_16000: 126496 pages swapped out during the phase |
| `q35-4b-mtplx-20261019-102635.json` | 10:28 | MTPLX (sustained profile, depth 1) | 41.5 / 43.9 | prefill_16000: usage.prompt_tokens missing; cached_16000: turn 1 returned no content; not substituting text, cached test skipped; decode_greedy: repetition loop (distinct-4-gram 0.4724) |
| `q35-4b-lmstudio-20261019-103118.json` | 10:32 | LM Studio (MLX engine) | 47.3 / 50.7 | decode: completion_tokens=511 != 512; decode: repetition loop (distinct-4-gram 0.4565); decode: repetition loop (distinct-4-gram 0.4642); decode_greedy: completion_tokens=511 != 512 |
| `q35-4b-llamacpp-mtp3-20261019-103601.json` | 10:36 | llama.cpp + MTP (n-max 3) | 37.8 / 44.0 | prefill_4000: 8904 pages swapped out during the phase; prefill_16000: 83764 pages swapped out during the phase; tools: 74060 pages swapped out during the phase |
| `q35-4b-mlxlm-20261019-104040.json` | 10:41 | mlx-lm | 46.4 / 49.8 | decode_greedy: repetition loop (distinct-4-gram 0.4478) |
| `q35-4b-kapitsa-20261019-104505.json` | 10:45 | Kapitsa (DFlash drafter) | 78.5 / 90.0 | — |

Ollama's and llama.cpp's prefill phases swapped (pages swapped out during the phase), so they do not count; MTPLX's
16K prompt came back without token counts and its tool calls failed.

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

## Nemotron Dory1.0 4B ([dory](dory/))

One hold of the GPU lock, 09:37-09:50, Kapitsa (the 0.1.36 release, from a copy outside any git checkout) first and
last: `configs/dory.json`, the decode phase (`--skip greedy,prefill,tools,agent`; llama.cpp 2 runs of 256 tokens, the
others 5 of 512).

| Result file | Started | Run | Decode, T = 0.6 (tok/s): median (min-max) | Flags |
| --- | --- | --- | ---: | --- |
| `dory-kapitsa-perloop-20261019-093758.json` | 09:38 | Kapitsa (NVFP4, 2 loops, per-loop KV, speculative decoding) | 30.8 (30.0-31.7) | — |
| `dory-kapitsa-20261019-094017.json` | 09:40 | Kapitsa (NVFP4, 2 loops, last-loop KV) | 35.2 (34.7-35.5) | — |
| `dory-llamacpp-q5km-20261019-094216.json` | 09:43 | llama.cpp (NVIDIA fork, Q5_K_M, 2 loops, last-loop KV) | 4.4 (4.4-4.5) | decode: repetition loop (distinct-4-gram 0.4955) |
| `dory-mlxlm-4bit-20261019-094528.json` | 09:46 | mlx-lm (PR #1988, 4-bit, 2 loops, per-loop KV) | 29.9 (29.6-30.1) | decode: content contains </think>; decode: repetition loop (distinct-4-gram 0.4786) |
| `dory-kapitsa-perloop-20261019-094750.json` | 09:48 | Kapitsa (NVFP4, 2 loops, per-loop KV, speculative decoding) | 31.7 (30.7-33.6) | — |

The mlx-lm result is marked invalid because one of its five answers contains `</think>`; its median is over the other
four.

## Rapid-MLX on Qwen3.5-4B ([rapid-mlx](rapid-mlx/))

Rapid-MLX 0.16.3 (mlx 0.32.4, mlx-lm 0.31.3, mlx-vlm 0.7.2) was installed after the other engines' runs. Its two
settings in short runs ([rapid-mlx/tuning](rapid-mlx/tuning/)), 12:12-12:16, the decode phase only, as above; then the
faster in a bracket of its own, 12:51-13:02, the full protocol, Kapitsa 0.1.36 first and last.

| Result file | Started | Run | Decode, T = 0.6 / greedy (tok/s) | Flags |
| --- | --- | --- | ---: | --- |
| `tuning/q35-4b-rapidmlx-20261019-121142.json` | 12:13 | Rapid-MLX (no speculation), short run | 51.1 / — | decode: repetition loop (distinct-4-gram 0.4776) |
| `tuning/q35-4b-rapidmlx-mtp-20261019-121356.json` | 12:15 | Rapid-MLX + MTP, short run | 44.4 / — | — |
| `q35-4b-kapitsa-20261019-125108.json` | 12:51 | Kapitsa (DFlash drafter) | 78.4 / 88.2 | decode: repetition loop (distinct-4-gram 0.4566) |
| `q35-4b-rapidmlx-20261019-125436.json` | 12:55 | Rapid-MLX (no speculation) | 51.5 / 55.1 | — |
| `q35-4b-kapitsa-20261019-125911.json` | 12:59 | Kapitsa (DFlash drafter) | 89.3 / 86.8 | decode: repetition loop (3 of 5 answers) |

## Qwen3.5-9B at 4-bit ([qwen3.5-9b](qwen3.5-9b/))

Not in Kapitsa's catalog: lmstudio-community/Qwen3.5-9B-MLX-4bit @b455506 with z-lab/Qwen3.5-9B-DFlash @5fc3b3d.
The configs are in [qwen3.5-9b/q35-9b.json](qwen3.5-9b/q35-9b.json) (pass it after `configs/*.json`); mlx-serve's
folder is `tools/mlx_serve_pack.py qwen3.5-9b`. Kapitsa is the 0.1.37 release (a copy outside any git checkout).

The full protocol, one hold of the GPU lock, 12:28-12:51, Kapitsa first and last:

| Result file | Started | Run | Decode, T = 0.6 / greedy (tok/s) | Flags and invalid phases |
| --- | --- | --- | ---: | --- |
| `q35-9b-kapitsa-20261019-122844.json` | 12:29 | Kapitsa (DFlash drafter) | 47.1 / 54.1 | decode: repetition loop (distinct-4-gram 0.4818) |
| `q35-9b-mlxserve-mtp-d2-20261019-123412.json` | 12:35 | mlx-serve + MTP (max depth 2) | 37.4 / 37.9 | prefill_16000: request failed (HTTP 400, the prompt does not fit mlx-serve's memory check) |
| `q35-9b-mlxlm-20261019-123857.json` | 12:39 | mlx-lm | 29.7 / 31.2 | decode: repetition loop (distinct-4-gram 0.4532) |
| `q35-9b-kapitsa-20261019-124541.json` | 12:46 | Kapitsa (DFlash drafter) | 50.7 / 52.1 | — |

No phase swapped. Short runs before it ([qwen3.5-9b/tuning](qwen3.5-9b/tuning/)), 12:01-12:11, the decode phase only
(`--skip greedy,prefill,tools,agent --bench-args "--decode-runs 3"`):

| Result file | Started | Run | Decode, T = 0.6 (tok/s): median (min-max) |
| --- | --- | --- | ---: |
| `q35-9b-kapitsa-20261019-115732.json` | 12:01 | Kapitsa (DFlash drafter) | 40.2 (40.0-50.5) |
| `q35-9b-mlxlm-20261019-120228.json` | 12:03 | mlx-lm | 29.5 (29.4-29.7) |
| `q35-9b-mlxserve-dflash-20261019-120414.json` | 12:05 | mlx-serve + DFlash (z-lab drafter, block 4) | 29.6 (26.4-29.6) |
| `q35-9b-mlxserve-mtp-d2-20261019-120659.json` | 12:08 | mlx-serve + MTP (max depth 2) | 37.1 (35.8-38.1) |
| `q35-9b-mlxserve-mtp-d3-20261019-120920.json` | 12:10 | mlx-serve + MTP (max depth 3) | 35.9 (35.6-36.1) |

## Notes

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
