# Results, 2026-10-18: mlx-serve

The result files behind the [mlx-serve section](../../../docs/benchmarks.md#mlx-serve-2026-10-18) of
docs/benchmarks.md: mlx-serve 26.10.2 with MTP and with DFlash2 on Qwen3.6-35B-A3B and Qwen3.8-27B, and Kapitsa (main
at c93f27c) on both models before and after them, back to back under one hold of the GPU lock (2026-10-18,
12:40-13:38), written by [bench/harness](../../harness/README.md) on a MacBook Pro (M4 Pro, 48 GB, macOS 27.0.1, High
Power mode, on AC power) with [configs/mlx-serve.json](../../harness/configs/mlx-serve.json) and
[configs/kapitsa.json](../../harness/configs/kapitsa.json). Rebuild the tables with

```bash
python3 bench/harness/compare_table.py bench/results/2026-10-18-mlx-serve --out /tmp/table
```

It shows each configuration's newest clean run, else its newest run with its flags: here the same runs as the docs
tables (on the 27B, where every run was flagged, the newest).

## Runs

In the order they ran. "Started" is when the benchmark began, after start-up and warm-up; times are local (UTC+5).
A run is flagged when apps outside the benchmark (foreign) used more than 5% of the GPU for a sustained stretch.

| Result file | Started | Run | Foreign GPU, mean / max (%) | Ambient GPU, mean (%) | Lock held (s) | Flagged for contention |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `q36-35b-kapitsa-20261018-124014.json` | 12:41 | Kapitsa, before | 0.8 / 5.3 | 7.9 | 173 | — |
| `q38-27b-kapitsa-20261018-124307.json` | 12:44 | Kapitsa, before | 4.5 / 38.6 | 10.7 | 280 | sampled decode, 4K prefill, 16K prefill (Slack Helper, Google Chrome Helper) |
| `q36-35b-mlxserve-mtp-20261018-124748.json` | 12:48 | mlx-serve MTP | 2.5 / 5.9 | 5.1 | 138 | — |
| `q38-27b-mlxserve-mtp-20261018-125006.json` | 12:51 | mlx-serve MTP | 2.4 / 39.2 | 13.6 | 466 | 16K prefill (Google Chrome Helper, Slack Helper) |
| `q38-27b-mlxserve-dflash2-20261018-125751.json` | 12:59 | mlx-serve DFlash2 | 10.1 / 65.3 | 14.8 | 414 | greedy decode, 16K prefill (Google Chrome Helper, Slack Helper) |
| `q36-35b-mlxserve-dflash2-20261018-130445.json` | 13:05 | mlx-serve DFlash2 | 0.3 / 2.1 | 3.0 | 156 | — |
| `q38-27b-mlxserve-mtp-20261018-130722.json` | 13:08 | mlx-serve MTP, re-run | 2.5 / 55.5 | 13.2 | 460 | 16K prefill (Google Chrome Helper, Slack Helper) |
| `q36-35b-kapitsa-20261018-131530.json` | 13:16 | Kapitsa, after | 0.6 / 3.1 | 9.3 | 109 | — |
| `q38-27b-kapitsa-20261018-131719.json` | 13:18 | Kapitsa, after | 10.3 / 90.5 | 10.3 | 273 | greedy decode, 4K prefill, 16K prefill, tool calls (Google Chrome Helper, Slack Helper) |
| `q38-27b-mlxserve-dflash2-20261018-132151.json` | 13:23 | mlx-serve DFlash2, re-run | 3.0 / 10.6 | 17.5 | 431 | 16K prefill (Google Chrome Helper) |
| `q38-27b-kapitsa-20261018-132903.json` | 13:29 | Kapitsa, re-run | 8.5 / 41.0 | 10.3 | 273 | sampled decode, greedy decode, 4K prefill, 16K prefill (Google Chrome Helper) |
| `q38-27b-kapitsa-20261018-133336.json` | 13:35 | Kapitsa, re-run | 4.9 / 31.1 | 3.7 | 286 | sampled decode, 4K prefill, 16K prefill, tool calls (Google Chrome Helper) |

Every 27B run was flagged: Chrome and Slack kept using more than 5% of the GPU for stretches, most of all in the long
prefills. Each flagged 27B setting was re-run in the same hold of the lock, once or twice, and every re-run was flagged
again; the docs tables show each setting's newest run and mark its flagged phases. The 35B runs are all clean.

The recorded ambient mean of `q38-27b-kapitsa-20261018-133336.json` is -264.0: two of its samples came out negative
when an ambient process closed one of its GPU clients, which takes that client's GPU time out of the process's total.
The table leaves those two samples out; the harness now clamps such drops (`gpu_hogs.py`).

Two attempts are not in the folder: a second re-run of the 27B MTP setting, stopped in its warm-up so the DFlash2
re-run could follow, and the first try of that re-run, which found the stopped server still holding its port and did
not start.

## Tuning

[tuning](tuning/) holds the short runs that chose mlx-serve's settings, just before (12:06-12:39, in a hold of the GPU
lock of their own): the usual warm-up, then five sampled decodes (`--decode-runs 5 --greedy-runs 0 --skip
prefill,tools`). The last column reads the server's `[spec-stats]` line of each timed answer (also in each file's
`server_log`): which speculative mode ran, and how many drafts it accepted per round out of how many it proposed.

| Run | Setting | Decode, T = 0.6 (tok/s) | Timed answers, from the server log |
| --- | --- | ---: | --- |
| `smoke-q36-mtp-auto` | MTP auto depth (cap 6) | 124.8 (121.2–131.2) | MTP in 5: 1.45 of 1.89 drafts accepted per round |
| `smoke-q36-mtp-d2` | MTP max depth 2 | 127.5 (123.7–128.6) | MTP in 5: 1.38 of 1.83 drafts accepted per round |
| `smoke-q36-mtp-d3` | MTP max depth 3 | 117.5 (109.3–122.0) | MTP in 5: 1.44 of 2.09 drafts accepted per round |
| `smoke-q36-mtp-d1` | MTP max depth 1 | 114.7 (112.3–116.9) | MTP in 5: 0.84 of 1.00 drafts accepted per round |
| `smoke-q36-mtp-nopld` | MTP auto depth, --no-pld | 120.8 (113.3–127.8) | MTP in 5: 1.57 of 2.10 drafts accepted per round |
| `smoke-q36-dfl-mtphead` | DFlash2 + MTP head loaded, auto block | 98.7 (97.5–102.9) | DFlash in 5: 1.78 of 4.00 drafts accepted per round, switched off mid-answer in 5 |
| `smoke-q36-dfl-nomtp` | DFlash2 --no-mtp, auto block | 102.4 (98.0–110.5) | DFlash in 5: 1.80 of 4.00 drafts accepted per round, switched off mid-answer in 3 |
| `smoke-q36-dfl-nomtp-b4` | DFlash2 --no-mtp block 4 | 103.9 (100.5–104.7) | DFlash in 5: 1.75 of 3.00 drafts accepted per round, switched off mid-answer in 5 |
| `smoke-q36-dfl-nomtp-b8` | DFlash2 --no-mtp block 8 | 88.2 (80.0–99.5) | DFlash in 5: 2.66 of 7.00 drafts accepted per round |
| `smoke-q38-mtp-auto` | MTP auto depth (cap 6) | 36.3 (34.9–37.4) | MTP in 5: 2.52 of 3.42 drafts accepted per round |
| `smoke-q38-mtp-d3` | MTP max depth 3 | 37.7 (37.1–39.4) | MTP in 5: 2.24 of 3.00 drafts accepted per round |
| `smoke-q38-mtp-d4` | MTP max depth 4 | 37.8 (37.6–39.5) | MTP in 5: 2.53 of 3.46 drafts accepted per round |
| `smoke-q38-dfl-mtphead` | DFlash2 tree + MTP head loaded (pack default) | 50.1 (26.6–55.0) | MTP in 2: 1.61 of 1.98 drafts accepted per round; DFlash in 3: 4.22 of 7.00 drafts accepted per round |
| `smoke-q38-dfl-nomtp` | DFlash2 tree --no-mtp | 51.5 (47.4–54.0) | DFlash in 5: 3.97 of 7.00 drafts accepted per round |
| `smoke-q38-dfl-nomtp-b6` | DFlash2 tree --no-mtp block 6 | 45.4 (43.1–50.1) | DFlash in 5: 3.45 of 5.00 drafts accepted per round |
| `smoke-q38-dfl-nomtp-notree` | DFlash2 no selector/tree --no-mtp | 48.0 (41.7–48.0) | DFlash in 5: 3.97 of 7.00 drafts accepted per round |

Chosen: MTP `--mtp-max-depth 2` on the 35B and `4` on the 27B; DFlash2 with `--no-mtp`, block 8 and the draft tree (the
defaults) on the 27B, block 4 on the 35B. With the MTP head loaded beside the drafter, mlx-serve hands a request to MTP
when the previous one is still finishing on the server, which the benchmark's back-to-back requests sometimes are (2 of
5 timed answers on the 27B), so the DFlash2 configs run without the head. `--no-pld` changes nothing here: mlx-serve
runs PLD only when neither MTP nor a drafter is active.

## Files

The fields are those of [2026-10-17](../2026-10-17/README.md). Local paths are replaced by placeholders: `<models>`
for the weights folder (`BENCH_MODELS`), `<out>` and `<smoke-out>` for the output folders, `<scratch>` for the
session's scratch folder, `<kapitsa-checkout>` for the Kapitsa worktree the session ran from and `<mlx-serve-26.10.2>`
for the unpacked release. The session held the GPU lock (`lockf -k ~/.cache/nace-gpu.lock`) around all the runs of
each part (tuning; then Kapitsa, mlx-serve and Kapitsa again), so the harness's own per-run lock pointed at a private
file (`BENCH_GPU_LOCK`, recorded as `gpu_lock_path`). The Mac was in use during the session: the camera daemon
(`cameracaptured`) counted as ambient load (`BENCH_GPU_AMBIENT`, with the desktop app that drove the runs; together
3-18% of the GPU in the runs), like macOS's photo analysis daemons; browser and chat apps stayed foreign, so their
bursts flag a run.
