# Results, 2026-10-19: the M4 Pro table filled in

The result files behind [the M4 Pro table filled in](../../../docs/benchmarks.md#the-m4-pro-table-filled-in-2026-10-19)
of docs/benchmarks.md, written by [bench/harness](../../harness/README.md) on a MacBook Pro (M4 Pro, 48 GB, macOS
27.0.1, High Power mode, on AC power). One hold of the GPU lock per model, Kapitsa (main at 474b3e6, a Release build of
the harness's checkout) first and last. Times are local (UTC+5); "Started" is when the benchmark began, after start-up
and warm-up.

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
cache, `<omlx-venv>` and `<mlx-serve-26.10.2>` the engines, `<scratch>` the session's scratch folder. The session held
the GPU lock around each hold (`lockf -k ~/.cache/nace-gpu.lock`), so the harness did not take it per run
(`NACE_GPU_LOCK_HELD=1`). The desktop app that drove the runs and the camera daemon count as ambient
(`BENCH_GPU_AMBIENT`); other apps' GPU use is foreign and flags a run.
