# Results, 2026-10-18: Qwen3.5-4B

The result files behind the [Qwen3.5-4B section](../../../docs/benchmarks.md#qwen35-4b-2026-10-18) of
docs/benchmarks.md: every engine at 4-bit on Qwen3.5-4B, in two brackets on 2026-10-18 (A 11:04-12:05, B 13:39-14:12),
each opened and closed by Kapitsa (68b7aee, recorded as b3e8fe9: the same source under an earlier commit id), written
by [bench/harness](../../harness/README.md) with the `q35-4b-*` configs in [bench/harness/configs](../../harness/configs)
on a MacBook Pro (M4 Pro, 48 GB, macOS 27.0.1, High Power mode, on AC power). Rebuild the table with

```bash
python3 bench/harness/compare_table.py bench/results/2026-10-18-4b --out /tmp/table
```

It shows each configuration's newest clean run: the rows of the docs table, with one Kapitsa row (bracket B's opening
run; the docs table also shows bracket A's).

- The folder itself: the runs in the docs table.
- [confirmations](confirmations/): Kapitsa's closing run of each bracket, both flagged for contention.
- [superseded](superseded/): the runs of bracket A that were flagged for contention, whose configurations were run
  again (in A while the Mac was busy, then in B).
- [tuning](tuning/): the short runs that chose oMLX's and MTPLX's settings.
- [logs](logs/): each run's server log, in full.

## Runs

In the order they ran. "Started" is when the benchmark began, after start-up and warm-up; times are local (UTC+5).
A run is flagged when apps outside the benchmark (foreign) used more than 5% of the GPU for a sustained stretch.
"Free disk" is at the start of the run (`provenance.system_start.disk_free_bytes`).

| Result file | Started | Bracket | Run | Foreign GPU, mean / max (%) | Ambient GPU, mean (%) | Lock held (s) | Free disk (GiB) | Flagged for contention |
| --- | --- | :---: | --- | ---: | ---: | ---: | ---: | --- |
| `q35-4b-kapitsa-20261018-110434.json` | 11:05 | A | Kapitsa, opening | 0.0 / 0.5 | 9.1 | 147 | 14.5 | — |
| `q35-4b-omlx-best-20261018-110702.json` | 11:07 | A | oMLX + MTP | 0.1 / 2.8 | 8.0 | 151 | 14.1 | — |
| `superseded/q35-4b-omlx-20261018-110933.json` | 11:10 | A | oMLX, no speculation | 1.7 / 18.2 | 2.5 | 155 | 17.7 | 4K prefill, 16K prefill (Slack Helper) |
| `superseded/q35-4b-mtplx-20261018-111208.json` | 11:13 | A | MTPLX | 1.6 / 23.2 | 1.3 | 179 | 19.1 | 4K prefill (Google Chrome Helper) |
| `q35-4b-ollama-mtp-20261018-111508.json` | 11:15 | A | Ollama GGUF + MTP | 0.2 / 5.1 | 8.0 | 169 | 13.4 | — |
| `q35-4b-ollama-nomtp-20261018-111757.json` | 11:18 | A | Ollama GGUF, MTP off | 1.4 / 26.3 | 14.6 | 181 | 16.2 | — |
| `q35-4b-ollama-mlx-q4-20261018-112059.json` | 11:21 | A | Ollama MLX runner + MTP | 0.8 / 6.3 | 9.0 | 160 | 18.1 | — |
| `q35-4b-lmstudio-20261018-112339.json` | 11:24 | A | LM Studio | 0.8 / 24.1 | 3.7 | 184 | 18.1 | — |
| `q35-4b-mlxlm-20261018-112645.json` | 11:27 | A | mlx-lm | 0.6 / 10.4 | 3.9 | 158 | 18.1 | — |
| `superseded/q35-4b-llamacpp-20261018-112924.json` | 11:29 | A | llama.cpp | 2.0 / 27.9 | 6.1 | 171 | 18.1 | 16K prefill (Google Chrome Helper, Slack Helper) |
| `q35-4b-llamacpp-mtp1-20261018-113215.json` | 11:32 | A | llama.cpp + MTP, n-max 1 | 0.4 / 10.1 | 4.5 | 167 | 17.9 | — |
| `q35-4b-llamacpp-mtp2-20261018-113502.json` | 11:35 | A | llama.cpp + MTP, n-max 2 | 0.0 / 0.0 | 11.8 | 161 | 18.0 | — |
| `q35-4b-llamacpp-mtp3-20261018-113743.json` | 11:38 | A | llama.cpp + MTP, n-max 3 | 0.0 / 0.0 | 8.6 | 157 | 17.8 | — |
| `superseded/q35-4b-llamacpp-mtp6-20261018-114020.json` | 11:40 | A | llama.cpp + MTP, n-max 6 | 6.0 / 66.6 | 2.9 | 167 | 17.8 | 4K prefill, 16K prefill (Slack Helper, Google Chrome Helper) |
| `superseded/q35-4b-omlx-20261018-114308.json` | 11:43 | A | oMLX, no speculation, re-run | 1.6 / 16.4 | 7.0 | 155 | 17.8 | tool calls (Google Chrome Helper, cameracaptured) |
| `superseded/q35-4b-mtplx-20261018-114544.json` | 11:51 | A | MTPLX, re-run | 9.3 / 39.9 | 2.5 | 475 | 15.6 | sampled decode, greedy decode, 4K prefill, 16K prefill (Google Chrome Helper, cameracaptured) |
| `superseded/q35-4b-llamacpp-20261018-115339.json` | 11:54 | A | llama.cpp, re-run | 8.1 / 34.2 | 4.0 | 208 | 11.5 | greedy decode, 4K prefill, 16K prefill, tool calls (Google Chrome Helper, cameracaptured, Slack Helper) |
| `superseded/q35-4b-llamacpp-mtp6-20261018-115707.json` | 11:58 | A | llama.cpp + MTP n-max 6, re-run | 10.4 / 56.1 | 0.8 | 246 | 15.4 | 4K prefill, 16K prefill (cameracaptured, Google Chrome Helper) |
| `confirmations/q35-4b-kapitsa-20261018-120114.json` | 12:04 | A | Kapitsa, closing | 3.7 / 18.0 | 14.3 | 268 | 17.5 | 16K prefill, tool calls (cameracaptured, Google Chrome Helper) |
| `q35-4b-kapitsa-20261018-133930.json` | 13:40 | B | Kapitsa, opening | 0.1 / 1.8 | 3.4 | 114 | 8.9 | — |
| `q35-4b-llamacpp-20261018-134313.json` | 13:43 | B | llama.cpp | 0.0 / 0.8 | 7.7 | 170 | 8.9 | — |
| `q35-4b-llamacpp-mtp6-20261018-134603.json` | 13:46 | B | llama.cpp + MTP, n-max 6 | 2.9 / 45.4 | 1.6 | 164 | 9.2 | 16K prefill (Google Chrome Helper) |
| `q35-4b-omlx-20261018-135225.json` | 13:52 | B | oMLX, no speculation | 0.5 / 6.0 | 3.9 | 152 | 12.2 | — |
| `q35-4b-mtplx-20261018-135712.json` | 13:58 | B | MTPLX | 0.5 / 9.6 | 1.0 | 176 | 10.0 | — |
| `confirmations/q35-4b-kapitsa-20261018-141016.json` | 14:10 | B | Kapitsa, closing | 4.1 / 19.0 | 0.5 | 106 | 12.0 | 4K prefill, 16K prefill (Slack Helper) |

In bracket A, from 11:42 Slack, Chrome and a video call (`cameracaptured`, macOS's camera daemon) kept flagging runs;
the four configurations whose runs had been flagged were run again in A as they came up, flagged again, and run a
third time in B. The llama.cpp re-run at 11:54 also overlapped another job's model load (11:53-11:56) that did not
take the GPU lock. Bracket B started each engine after other apps had used less than 3% of the GPU for 60 s; its
llama.cpp n-max 6 run (Chrome, in the 16K prefill) and Kapitsa's closing run (Slack, in both prefills) were still
flagged. The docs table marks the one flagged figure it uses.

LM Studio's MLX engine keeps the checkpoints that serve follow-up turns on disk only while at least 10 GiB of the disk
is free, and MTPLX's SSD session tier writes only what fits above 10 GiB free; LM Studio's run had 18.1 GiB. MTPLX's B
run had 10.0 GiB, and its log shows the SSD tier skipping its writes ("no disk room"); its follow-ups came from its
in-memory session cache, as fast as in its A run with 19.1 GiB (0.09 / 0.11 s against 0.10 / 0.12 s).

## Tuning

[tuning](tuning/) holds the short runs that chose oMLX's and MTPLX's settings, just before bracket A (10:32-11:04),
under the GPU lock: the usual warm-up, then three sampled and three greedy decodes (`--decode-runs 3 --skip
prefill,tools`). Each file records its full config.

| Run | Setting | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) |
| --- | --- | ---: | ---: |
| `smoke-q35-4b-omlx-mtp` | oMLX, Lightning MTP with MTPLX's head | 101.4 (99.3–102.2) | 103.8 (103.4–104.1) |
| `smoke-q35-4b-omlx-dflash-b5-adaptive` | oMLX, DFlash with z-lab/Qwen3.5-4B-DFlash (4-bit at load), block 5, verify `adaptive` | 89.8 (86.5–102.1) | 98.0 (93.5–99.3) |
| `smoke-q35-4b-omlx-dflash-b5-dflash` | the same, verify `dflash` | 89.7 (88.6–91.6) | 91.5 (85.9–95.9) |
| `smoke-q35-4b-omlx-dflash-b9-adaptive` | block 9, verify `adaptive` | 75.3 (64.2–76.2) | 72.6 (70.2–81.7) |
| `smoke-q35-4b-omlx-dflash-b9-dflash` | block 9, verify `dflash` | 56.1 (50.9–60.5) | 56.5 (55.6–59.1)<sup>1</sup> |
| `smoke-q35-4b-omlx-plain` | oMLX, no speculation | 84.9 (83.9–86.2) | 81.8 (80.8–84.2) |
| `smoke-q35-4b-mtplx-d1` | MTPLX, profile `sustained`, depth 1 | 104.5 (101.7–105.0) | 108.6 (104.7–109.7) |
| `smoke-q35-4b-mtplx-d2` | depth 2 | 103.8 (102.7–106.6) | 109.2 (108.3–110.9)<sup>2</sup> |
| `smoke-q35-4b-mtplx-d3` | depth 3, the runtime file's default | 87.7 (87.6–91.5) | 102.6 (95.8–103.2) |

1. One greedy run looped (distinct-4-gram 0.48).
2. Flagged for contention.

Chosen: Lightning MTP for oMLX, the fastest at both temperatures; for MTPLX depth 1, the fastest sampled and within 1%
of depth 2 at greedy. z-lab's drafter is DFlash v1; incoai publishes no DFlash2 drafter for the 4B.

## Files

The fields are those of [2026-10-17](../2026-10-17/README.md). Local paths are replaced by placeholders: `~` for the
home folder, `<kapitsa-checkout>` for the Kapitsa worktree the runs started from (Kapitsa ran from a copy of the
Release build at `<kapitsa-checkout>/build/kapitsa`, sha256 in the result files), `<models>` for the weights folder
(`BENCH_MODELS`), `<out>` for the output folder (`BENCH_OUT`), `<ollama-mlx-store>` for the Ollama store the MLX
weights were imported into (`OLLAMA_MLX_MODELS`), `<engines-root>` for the folder holding the other engines' installs
and `<scratch>` for the session's scratch folder (the tuning runs' output and config); nothing else was changed. Each
engine's runs held the GPU lock (`lockf -k -t 3000 ~/.cache/nace-gpu.lock`, with a time limit), taken by the script
that queued them and released between engines; the harness, started inside it with `NACE_GPU_LOCK_HELD=1`, did not
take the lock again (`gpu_lock.harness_lock`). Ambient load is the desktop app that drove the runs
(`BENCH_GPU_AMBIENT`); every other process, the camera daemon included, counted as foreign.
