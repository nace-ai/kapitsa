# Results, 2026-10-18: Nemotron 3.5 Bolt 74B-A9B

The result files behind the Bolt tables of [docs/benchmarks.md](../../../docs/benchmarks.md): one run per engine
configuration, back to back in one session (2026-10-18, 06:37-07:50), written by [bench/harness](../../harness/README.md) on a
MacBook Pro (M4 Pro, 48 GB, macOS 27.0.1) with [configs/bolt.json](../../harness/configs/bolt.json)'s settings. Every
engine read the same folder: NVIDIA's 24 GB checkpoint and its LiLiCorr drafter, plus the MLX bundle's model file
for the MLX engines. Rebuild the tables with

```bash
python3 bench/harness/compare_table.py bench/results/2026-10-18 --out /tmp/table
```

The fields are those of [2026-10-17](../2026-10-17/README.md). Local paths are replaced by placeholders: `~` for the
home folder, `<bench-root>` for the folder the harness ran from, `<build-dir>` for the frozen Kapitsa build. The
Kapitsa run says `engine: "nace-infer"` (Kapitsa's earlier name). oMLX with the drafter fails every request at
temperature 0.6 (its sampled DFlash selection), so its own file has no valid sampled decode or prefill and its
`-greedy` file sends temperature 0 in every phase. [confirmations](confirmations/) holds Kapitsa's earlier run of the
same session (before the weight packing that lowered its memory), outside the tables.
