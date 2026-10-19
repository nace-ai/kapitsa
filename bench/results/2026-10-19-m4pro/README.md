# Results, 2026-10-19: the M4 Pro table filled in

The result files behind [the M4 Pro table filled in](../../../docs/benchmarks.md#the-m4-pro-table-filled-in-2026-10-19)
of docs/benchmarks.md, written by [bench/harness](../../harness/README.md) on a MacBook Pro (M4 Pro, 48 GB, macOS
27.0.1, High Power mode, on AC power). One hold of the GPU lock per model, Kapitsa (main at 474b3e6, a Release build of
the harness's checkout) first and last. Times are local (UTC+5); "Started" is when the benchmark began, after start-up
and warm-up.

## Qwen (configs/kapitsa.json, mlx-serve.json, mlx-vlm.json)

One hold, 06:16–06:31, each model between two Kapitsa runs.

| Result file | Started | Run | Decode, T = 0.6 / greedy (tok/s) | Foreign GPU, mean / max (%) | Ambient GPU, mean (%) | Flagged |
| --- | --- | --- | --- | ---: | ---: | --- |
| `q35-4b-kapitsa-20261019-061628.json` | 06:16 | Kapitsa, before | 183.4 / 189.9 | 1.1 / 14.0 | 0.9 | contention (Chrome) |
| `q35-4b-mlxserve-mtp-20261019-061757.json` | 06:18 | mlx-serve, MTP | 137.0 / 147.5 | 0.0 / 0.9 | 0.0 | — |
| `q35-4b-kapitsa-20261019-062002.json` | 06:20 | Kapitsa, after | 169.2 / 194.0 | 1.2 / 7.6 | 2.7 | — |
| `q36-35b-kapitsa-20261019-062128.json` | 06:22 | Kapitsa, before | 159.1 / 176.2 | 0.2 / 5.9 | 0.7 | — |
| `q36-35b-mlxvlm-dflash2-20261019-062405.json` | 06:24 | mlx-vlm, DFlash2 | 86.0 / 90.7 | 0.2 / 4.8 | 0.6 | — |
| `q36-35b-kapitsa-20261019-062814.json` | 06:28 | Kapitsa, after | 159.6 / 176.5 | 0.0 / 0.0 | 0.0 | — |

mlx-serve read `tools/mlx_serve_pack.py qwen3.5-4b`'s folder (the shared 4B trunk with mlx-community's 4-bit MTP head
as its sidecar); the head here was quantized locally from Qwen's bf16 head with MLX's affine 4-bit, group 64, and equals
mlx-community/Qwen3.5-4B-MTP-4bit @ab6f59b on every tensor (whole tensors up to 64 KB, the first and last 32 KB of the
larger ones, compared by range requests). The 35B runs also replayed the agent session (bench.py's `agent` phase): mlx-vlm 79.3 s, Kapitsa 48.2 s.

### Tuning

[tuning](tuning/) holds the short runs that chose the settings, 04:36–05:42 (three sampled decodes each,
`--decode-runs 3 --skip greedy,prefill,tools,agent`), on the configs above with the setting in the name.

| Model | Engine | Setting: decode, T = 0.6 (tok/s) |
| --- | --- | --- |
| 4B | mlx-serve | MTP max depth 3: 136.0; 4: 134.1; 2: 130.5; 1: 120.3; adaptive: 124.0. DFlash (z-lab's drafter) block 4: 124.5; 5 (default): 118.5; 6: 111.2 |
| 4B | mlx-vlm | MTP (4-bit head) block 3: 86.5; 2: 83.7; 4: 82.7; MTP (bf16 head) block 3: 88.9; no speculation: 78.8; DFlash (z-lab's drafter) block 16: 73.5; 8: 72.7 |
| 35B | mlx-vlm | DFlash2 (default block): 85.2; block 8: 81.6; no speculation: 81.7; MTP with MTPLX's head: does not load |
| 27B | mlx-vlm | MTP block 3: 27.2; 4: 24.6; 5: 18.1; DFlash2: 24.6 |

## Nemotron 3.5 Bolt (configs/bolt.json)

| Result file | Started | Run | Decode, T = 0.6 / greedy (tok/s) | Foreign GPU, mean / max (%) | Ambient GPU, mean (%) | Flagged |
| --- | --- | --- | --- | ---: | ---: | --- |
| `bolt-kapitsa-20261019-050216.json` | 05:03 | Kapitsa, before | 66.4 / 69.9 | 0.0 / 0.7 | 7.2 | — |
| `bolt-omlx-20261019-050538.json` | 05:06 | oMLX, no drafter | 34.6 / 34.1 | 0.1 / 4.6 | 4.2 | — |
| `bolt-kapitsa-20261019-051156.json` | 05:12 | Kapitsa, after | 69.1 / 68.4 | 0.0 / 0.8 | 6.7 | — |

oMLX read NVIDIA's files with the config.json of NVIDIA's Bolt bundle for MLX
(`txgsync/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-MLX` @98c0cb7, which names the bundle's model file), the
harness README's layout; Kapitsa read the folder `kapitsa pull bolt` writes, with NVIDIA's own config.json.

- [confirmations](confirmations/): Kapitsa in an earlier hold the same morning (04:08–04:15, main at 7bf5c07): 66.4 /
  71.4 and 62.0 / 68.6 tok/s, prefill 531–532 / 523 tok/s, nothing flagged.
- [failed](failed/): oMLX in that earlier hold, on the folder with NVIDIA's own config.json: it refuses the model type
  (`nemotron_h_omni`), so every request failed.
- [support-checks](support-checks/): mlx-serve 26.10.2 loading Bolt, with NVIDIA's config.json and with the bundle's;
  it stops at a missing `model.embed_tokens.weight` both times.

Local paths in the files are replaced by placeholders: `<models>` the weights folder ($BENCH_MODELS), `<out>` the
harness output folder, `<kapitsa-checkout>` the Kapitsa checkout the build came from, `<kapitsa-models>` Kapitsa's model
cache, `<omlx-venv>` and `<mlx-serve-26.10.2>` the engines, `<smoke-out>` the short runs' output folder, `<scratch>` the session's scratch folder. The session held
the GPU lock around each hold (`lockf -k ~/.cache/nace-gpu.lock`), so the harness did not take it per run
(`NACE_GPU_LOCK_HELD=1`). The desktop app that drove the runs and the camera daemon count as ambient
(`BENCH_GPU_AMBIENT`); other apps' GPU use is foreign and flags a run.
