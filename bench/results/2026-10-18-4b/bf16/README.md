# Results, 2026-10-18: Qwen3.5-4B at bf16 (partial)

The result files behind the [bf16 part](../../../../docs/benchmarks.md#full-precision-bf16-a-partial-bracket) of the
Qwen3.5-4B section of docs/benchmarks.md: Kapitsa, oMLX (with z-lab's DFlash drafter, and without speculation) and
MTPLX on mlx-community/Qwen3.5-4B-bf16, one engine at a time under the GPU lock, between 15:30 and 15:55 on a MacBook
Pro (M4 Pro, 48 GB, macOS 27.0.1, High Power mode, on AC power), with the `q35-4b-bf16-*` configs in
[bench/harness/configs](../../../harness/configs). Rebuild the table with

```bash
python3 bench/harness/compare_table.py bench/results/2026-10-18-4b/bf16 --out /tmp/table
```

The bracket is incomplete: the disk ran short after these runs, so Kapitsa has no closing run, and mlx-lm, LM Studio,
llama.cpp and Ollama were not run at bf16.

| Result file | Started | Run | Foreign GPU, mean / max (%) | Ambient GPU, mean (%) | Lock held (s) | Free disk (GiB) | Flagged for contention |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `q35-4b-bf16-kapitsa-20261018-153021.json` | 15:30 | Kapitsa | 0.1 / 2.8 | 2.6 | 198 | 8.8 | — |
| `q35-4b-bf16-omlx-best-20261018-153957.json` | 15:40 | oMLX + DFlash | 0.0 / 0.6 | 0.0 | 217 | 9.6 | — |
| `q35-4b-bf16-omlx-20261018-154334.json` | 15:44 | oMLX, no speculation | 0.0 / 0.8 | 2.6 | 346 | 7.4 | — |
| `q35-4b-bf16-mtplx-20261018-155122.json` | 15:52 | MTPLX | 0.1 / 1.3 | 0.0 | 236 | 11.5 | — |

The two oMLX configs ran in one hold of the GPU lock; other jobs on the Mac had turns between the engines. MTPLX's SSD
session tier writes only what fits above 10 GiB of free disk, so it kept little on disk here; its follow-ups came from
its session cache in memory, as at 4-bit.

## Tuning

[tuning](tuning/) holds the short runs that chose oMLX's and MTPLX's settings, just before (14:16-15:09): the usual
warm-up, then three sampled and three greedy decodes (`--decode-runs 3 --skip prefill,tools`). Each file records its
full config.

| Run | Setting | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) |
| --- | --- | ---: | ---: |
| `smoke-q35-4b-bf16-omlx-dflash-b5` | oMLX, DFlash with z-lab/Qwen3.5-4B-DFlash (4-bit at load), block 5, verify `adaptive` | 59.1 (53.4–60.2) | 63.2 (60.5–67.0) |
| `smoke-q35-4b-bf16-omlx-dflash-b9` | the same, block 9 | 57.0 (48.5–60.8) | 56.8 (52.6–59.2) |
| `smoke-q35-4b-bf16-omlx-mtp` | oMLX, Lightning MTP with the bf16 head | 49.5 (44.8–50.0) | 48.2 (47.5–48.7) |
| `smoke-q35-4b-bf16-mtplx-d1` | MTPLX, profile `sustained`, depth 1 | 43.7 (42.2–44.7) | 41.0 (38.6–44.2) |
| `smoke-q35-4b-bf16-mtplx-d2` | depth 2 | 52.4 (51.7–52.8) | 54.9 (51.7–55.6) |
| `smoke-q35-4b-bf16-mtplx-d3` | depth 3 | 54.5 (52.0–58.0) | 59.0 (57.9–61.9) |

Chosen: DFlash block 5 for oMLX and depth 3 for MTPLX, the fastest at both temperatures. One run of each MTPLX setting
and of oMLX's block-5 setting looped (distinct-4-gram under 0.5), which speculative decoding speeds up; the order holds
without them.

## Files

As in [the 4-bit folder](../README.md#files): local paths are replaced by placeholders (`~`, `<kapitsa-checkout>`,
`<models>`, `<out>`, `<engines-root>`, `<scratch>`); nothing else was changed. `<models>/qwen3.5-4b-bf16-mtp` was a link
to the folder `kapitsa pull 4b:bf16` fills; `<models>/qwen3.5-4b-bf16-mtplx` held APFS clones of its files plus the
runtime file of MTPLX's 4B release.
