# Results, 2026-10-16/17

The result files behind [docs/benchmarks.md](../../../docs/benchmarks.md): one run per engine configuration, written by
[bench/harness](../../harness/README.md) on a MacBook Pro (M4 Pro, 48 GB, macOS 27.0.1). Rebuild the tables with

```bash
python3 bench/harness/compare_table.py bench/results/2026-10-17 --out /tmp/table
```

Each file holds every request's timings and text, the summaries, validity flags, contention samples, memory per
phase, the server command and environment, engine versions, weights, system state and the lines of the server log
that matter. The main fields: `decode_summary` and `decode_greedy_summary` (median, min, max tok/s), `prefill[]`
(`prefill_tps`, `prompt_tokens`, and `cached.median_ttft_s` for the follow-ups), `tools_pass`, `peak_memory_mb`,
`mapped_file_clean_mb`, `invalid`, `flags`, `contention` and `provenance`.

Local paths are replaced by placeholders: `~` for the home folder, `<bench-root>` for the folder the harness ran from
before it moved into this repository, `<kapitsa-checkout>` and `<build-dir>`; nothing else was changed. The Kapitsa
runs say `engine: "nace-infer"` (Kapitsa's earlier name), and `{root}` in the recorded configs is that earlier folder:
[bench/harness/configs](../../harness/configs) has the same settings in this repository's layout.
