# Results, 2026-10-18, M5 Max: Qwen and Nemotron against every engine

The result files behind the M5 Max cells of the README added on 2026-10-18 evening: written by
[bench/harness](../../harness/README.md) on a MacBook Pro (`Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4, AC
power), one engine at a time under the shared GPU lock, every phase but the agent replay (`--skip agent`), each
configuration once, with a 45-minute limit per run. Kapitsa ran from build 474b3e6.

- [results/](results/): one JSON per run (`compare_table.py` reads them); [logs/](logs/): the servers' own logs;
  [transcripts/](transcripts/): the harness's output per run; `session-qwen.log`, `session-nemotron.log`: start and end
  of each run in the two queues (they shared the lock, so their runs interleave and never overlap).
- New configs: `lightning-ollama-mlx`, `-lmstudio`, `-mtplx`, `-mlxserve-mtp` ([lightning.json](../../harness/configs/lightning.json)),
  `super-ollama-mlx`, `-lmstudio` ([super-vl.json](../../harness/configs/super-vl.json)), `flash-mtplx`,
  `-mlxserve-mtp`, `-lmstudio` ([flash-next.json](../../harness/configs/flash-next.json)); their notes say why each
  setting was chosen.

**Contention.** A `com.apple.WebKit` GPU process stayed open all session (3-8% of the GPU on average), so nearly every
run is flagged in its prefill phases. The decode phase, the README's writing speed, is flagged only in Kapitsa's 27B
and 35B runs (5.4% and 6.1% mean); the session ends with Kapitsa again.

**Kapitsa first and last.** The Qwen queue opened (16:50-17:16) and closed (23:23-23:49) with Kapitsa on the 35B, 27B,
4B and Bolt; both runs of each are here. The README takes the newest run with an unflagged decode phase, else the
newest run.

**Not kept.** The first two attempts at `bolt-omlx-lilicorr` and `-greedy` (22:12, 22:46) never started a server: the
config clears `OMLX_*`, and the launcher found no `omlx` on PATH; they ran again with it on PATH (23:10). The
sampled `bolt-omlx-lilicorr` run is kept INVALID: oMLX 0.7.1rc1's drafter path fails at T > 0.

**Failed.** `flash-lmstudio`: LM Studio's load guardrail refused the 119 GB GGUF ("requires approximately 117.52 GB of
memory ... would likely overload your system"); it was not overridden.

**Dory.** `dory-kapitsa` here is build 474b3e6 with last-loop KV; the README's Dory row uses a newer build with per-loop
KV, so this run is not in the README.

Local paths are replaced by placeholders: `~` for the home folder, `<repo>` for the Kapitsa checkout, `<run-root>` for
the run directory.
