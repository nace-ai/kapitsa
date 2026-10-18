# Results, 2026-10-18: a real coding-agent session

The result files behind the [agent-session section](../../../docs/benchmarks.md#a-real-coding-agent-session-2026-10-18)
of docs/benchmarks.md: a recorded opencode session
([agent-traces/opencode-ledger.json](../../harness/agent-traces/opencode-ledger.json), sha256 `bf4b69fe…` in each
file's `agent.trace`) replayed on every engine, written by [bench/harness](../../harness/README.md) (bench.py's agent
phase, [agent_replay.py](../../harness/agent_replay.py)) on a MacBook Pro (M4 Pro, 48 GB, macOS 27.0.1, High Power
mode, on AC power) with the configs in [bench/harness/configs](../../harness/configs). One bracket per model, each under
one hold of the GPU lock with Kapitsa (main at 113bfcd; the branch adds only bench/) first and last: the 35B on
2026-10-18 between 16:19 and 16:37, the 27B between 20:51 and 21:42. Rebuild the tables with

```bash
python3 bench/harness/compare_table.py bench/results/2026-10-18-agent-session --out /tmp/table
```

The runs replay only the agent session (`--skip decode,greedy,prefill,tools`), after the usual start-up, warm-up and
quiet-GPU wait. Each file's `agent` holds every replayed request (`turns`: the prompt and cached tokens as the engine
reported them, the answer's tokens and text, whether it is the recorded answer, the time to first token, the writing
window, the total time and every streamed event's time) and the session summary (`summary`);
`contention.phases.agent` is the GPU monitor's view of the session and `memory.agent` the server's memory after it.

The summaries were recomputed from the stored per-turn records (the events' times and sizes) by the final
`agent_replay.resummarize`, which counts a turn as held back when its first event carries most of the answer (MTPLX's
notices) and ends the writing window of an answer whose end was never streamed at the end of the stream (Ollama drops a
tool call that the recorded length cut short); each file keeps the summary as run in
`agent.summary_recomputed.summary_as_run`. Nothing else was changed, except that local
paths are replaced by placeholders: `~` for the home folder, `<scratch>` for the session's scratch folder (`<out>` its
output folder, `<models>` its weights folder of links), `<kapitsa-checkout>` for the Kapitsa worktree,
`<engines-root>`, `<venvs>`, `<ollama-mlx-store>`, `<mlx-serve-26.10.2>` and `<llama.cpp-0.7.0>` for the engines'
folders, `<kapitsa-models>` and `<dev-models>` for the weights the links point to. The session held the GPU lock
(`lockf -k ~/.cache/nace-gpu.lock`) around each bracket, so the harness's own per-run lock pointed at a private file
(`gpu_lock_path`). The desktop app that drove the runs and the camera daemon count as ambient GPU load
(`BENCH_GPU_AMBIENT`), like macOS's photo analysis daemons.

## Runs

In the order they ran. "Started" is when the session began, after start-up, warm-up and the quiet-GPU wait; times are
local (UTC+5). LM Studio and MTPLX started with more than the 10 GiB of free disk below which they keep no prompt cache
on disk. The 35B runs were not flagged for GPU contention; four 27B runs were, for short bursts of Chrome and Slack
(the desktop was in use again), and were not re-run (see the docs' notes).

| Result file | Started | Foreign GPU, mean / max (%) | Ambient GPU, mean (%) | Free disk at start (GiB) | Lock held (s) | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `q36-35b-kapitsa-20261018-161859.json` | 16:19 | 0.3 / 3.5 | 0.0 | 11.4 | 88 | Kapitsa, before |
| `q36-35b-ollama-mlx-q4-20261018-162027.json` | 16:21 | 0.0 / 0.0 | 5.7 | 11.4 | 174 | 3 answers held back (tool calls) |
| `q36-35b-lmstudio-20261018-162321.json` | 16:24 | 0.0 / 0.0 | 3.3 | 11.4 | 153 | |
| `q36-35b-mtplx-20261018-162554.json` | 16:26 | 0.1 / 1.4 | 1.3 | 10.3 | 216 | 2 answers held back (notices) |
| `q36-35b-omlx-best-20261018-162930.json` | 16:30 | 0.0 / 0.0 | 3.7 | 10.0 | 122 | |
| `q36-35b-mlxserve-mtp-20261018-163132.json` | 16:32 | 0.1 / 1.2 | 0.0 | 9.9 | 114 | |
| `q36-35b-mlxserve-dflash2-20261018-163326.json` | 16:34 | 0.1 / 1.1 | 0.0 | 9.8 | 124 | |
| `q36-35b-kapitsa-20261018-163530.json` | 16:36 | 0.3 / 3.0 | 0.0 | 9.7 | 90 | Kapitsa, after |
| `q38-27b-kapitsa-20261018-205048.json` | 20:51 | 4.9 / 29.7 | 1.6 | 10.1 | 211 | Kapitsa, before; flagged (Slack) |
| `q38-27b-ollama-mlx-q4-20261018-205419.json` | 20:56 | 2.4 / 65.2 | 0.3 | 13.1 | 418 | flagged (Slack, Chrome); 3 answers held back, 5 ends not streamed |
| `q38-27b-lmstudio-20261018-210117.json` | 21:02 | 0.1 / 7.0 | 9.9 | 11.6 | 525 | |
| `q38-27b-mtplx-20261018-211005.json` | 21:12 | 1.3 / 30.5 | 7.4 | 11.5 | 482 | flagged (Chrome); 3 answers held back (notices) |
| `q38-27b-omlx-best-20261018-211808.json` | 21:20 | 0.1 / 3.0 | 13.2 | 10.2 | 415 | |
| `q38-27b-mlxserve-mtp-20261018-212503.json` | 21:26 | 0.9 / 37.1 | 8.3 | 10.9 | 403 | flagged (Chrome, Slack) |
| `q38-27b-mlxserve-dflash2-20261018-213146.json` | 21:33 | 0.4 / 10.7 | 10.5 | 11.9 | 411 | |
| `q38-27b-kapitsa-20261018-213837.json` | 21:39 | 0.3 / 2.0 | 16.1 | 11.8 | 209 | Kapitsa, after |

[confirmations](confirmations/) holds two unflagged 27B runs from an earlier attempt at the 27B bracket (16:43-16:53,
Kapitsa and Ollama), outside the tables: the Mac went to sleep during that attempt's LM Studio run, so the bracket was
run again from the start. The 27B session is the 35B's: opencode, recorded greedily on the 27B, looped on one command
and never solved the task, so the 27B replays the 35B's requests and answers (see the docs).

llama.cpp + MTP did not run: it reads Ollama's Q4_K_M GGUF files, which are no longer on this Mac, and there was no
room to download them again. Before the brackets, the 35B was imported again for Ollama's MLX runner, as its tag was
gone from the store (`tools/ollama_mlx_import.py`, zero-copy; `tools/ollama_mlx_verify.py` passed, with the 25
elements of one MTP norm it documents).
