# Results, 2026-10-19: Nemotron-Labs Dory1.0 4B

The result files behind [Nemotron-Labs Dory1.0 4B](../../../docs/benchmarks.md#nemotron-dory10-4b-2026-10-19) of
docs/benchmarks.md, written by [bench/harness](../../harness/README.md) with [configs/dory.json](../../harness/configs/dory.json)
on a MacBook Pro (M4 Pro, 48 GB, macOS 27.0.1, on AC power). Times are local (UTC+5).

- The README's cells: one hold of the GPU lock, the decode phase only (the T = 0.6 coding answer): Kapitsa (main at
  18a03e1) with per-loop KV and with its last-loop default first, NVIDIA's llama.cpp fork on NVIDIA's Q5_K_M, mlx-lm
  with its Dory pull request on snsnc's MLX 4-bit, Kapitsa with per-loop KV last. Each competitor's files sat on a RAM
  disk made for its run.
- `dory-llamacpp-q5km-*` from 03:50 to 04:36: the llama.cpp fork's full runs in their own holds (prefill, follow-ups,
  tool calls, memory), per-loop KV, its n-gram speculation, 1 and 4 loops.

Local paths in the files are replaced by placeholders: `<models>` and `<ramdisk>` where the competitors' weights were,
`<out>` the harness output folder, `<kapitsa-checkout>` and `<kapitsa-build>` the Kapitsa checkout and build,
`<kapitsa-models>` Kapitsa's model cache, `<llama.cpp-dory-fork>` and `<mlx-lm-venv>` the engines, `<scratch>` the
session's scratch folder.
