# Cross-engine benchmark harness

The harness behind [docs/benchmarks.md](../../docs/benchmarks.md). It runs Kapitsa and other local inference engines
through the same client-side benchmark, one engine at a time on an otherwise idle GPU, and records enough provenance
to audit every number. Any engine with an OpenAI-compatible `/v1/chat/completions` endpoint works.

Requirements: a Mac with Apple silicon, Python 3.9 or later (standard library only; macOS's own `python3` works),
and the engines you want to compare. Memory, GPU load and the GPU lock use macOS tools (`footprint`, `ioreg`,
`vm_stat`, `lockf`), so the harness runs on macOS only.

## Quick start

```bash
cd bench/harness
cp local.env.example local.env          # where your weights and engines are; see "Weights" and "Engines"
python3 compare_engines.py configs/*.json --list
python3 compare_engines.py configs/kapitsa.json configs/llama.cpp.json --dry-run   # resolve everything, start nothing
python3 compare_engines.py configs/kapitsa.json configs/llama.cpp.json --only 'q36-35b-*'
python3 compare_engines.py configs/kapitsa.json --skip decode,greedy,prefill,tools   # only the agent session
python3 compare_table.py                # tables of the newest clean run per config, from runs/results
```

`--dry-run` prints, per config, whether it is ready (or what is missing), the exact server command, environment,
preparation steps, the benchmark command and where the result will go. One config takes 2 to 20 minutes (the slow
engines on the dense 27B take longest). The published runs used a 48 GB Mac; several engines need more than 32 GB
for the 35B and 27B models.

To tabulate the published runs: `python3 compare_table.py ../results/2026-10-17 --out /tmp/table`.

For Kapitsa alone, `kapitsa bench` runs this protocol from the binary itself (src/bench/protocol.cpp ports `bench.py`
request for request, and the server start, warm-up, quiet wait and contention monitor of `compare_engines.py` and
`gpu_hogs.py`; it takes the same GPU lock). Its results compare with this harness's and go to the
[community results](../community/RESULTS.md). `tests/test_bench.py` runs both clients against one scripted server and
checks that they send the same requests. Two differences: it reads memory with `proc_pid_rusage` (the same
`phys_footprint_peak` that `footprint` prints) and does not sample `footprint` and `ps` every 2 s while it measures;
and its client is C++, whose own share of a time to first token is about 1-1.5 ms smaller than `bench.py`'s (measured
against a server that answers at once): nothing next to a cold prompt's seconds, but 2-3% of a 50-60 ms follow-up, so
its follow-ups read slightly faster than the harness's for the same server.

## What a run measures

`compare_engines.py` starts the engine's server, waits until it answers, warms it up, waits for a quiet GPU and runs
`bench.py`, which talks to the server over HTTP like a coding agent does:

| Measurement | How |
| --- | --- |
| Decode, T = 0.6 | A 512-token coding answer (an LRU cache module with tests), streamed; 5 runs, median with min and max. The rate is taken over the middle of the stream (first 16 and last 4 tokens dropped), so time to first token is not counted and engines that send several tokens per event are not flattered. Sampling: temperature 0.6, top_p 0.95, top_k 20 (Qwen's recommendation). |
| Decode, greedy | The same prompt at temperature 0 (plus top_k 1, top_p 1), 5 runs. With the same weights every engine writes nearly the same text, so this removes differences in what was sampled. |
| Prefill, 4K and 16K | A cold prompt of 4,133 / 16,286 tokens (source code from the frozen corpus) with a random nonce right after `<\|im_start\|>system`, so no engine can reuse a cached prefix. Rate = prompt tokens / time to first token. |
| Cached TTFT, 4K and 16K | After that prompt, three follow-up user turns on the growing conversation (append-only, 1 s apart, as an agent loop), median time to first token. Engines that cache prefixes answer from the cache. |
| Tool calls | Streaming and non-streaming requests with two tools, with thinking off and on, plus a tool-result round trip. "pass" = every check passed. |
| Agent session | A real coding-agent session, replayed request by request: opencode fixing a small Python project, recorded on Kapitsa ([Agent session](#agent-session)). Per request the time to first token, the writing rate and the time; per session the total time, prefill and decode time, the mean writing rate and the follow-up TTFT (median, p90). |
| Agent-harness metrics (opt-in, `--agentic`) | Cold start (spawn to first token, and the first request's penalty over later ones), cold TTFT and decode tok/s at 1K to the largest context that fits `--max-ctx`, TTFT of a new question on a shared prefix, TTFT of a follow-up in a grown conversation, and whether a prefix survives another conversation, the agent replay's cache hit rate, sustained decode with a thermal check, idle vs peak memory. Details: [Agent-harness metrics](#agent-harness-metrics-agentic). |
| Peak memory | `phys_footprint_peak` of the server's whole process tree, in bytes, after every phase. Memory-mapped weights (llama.cpp's default mmap, Ollama's GGUF runner) are not part of `phys_footprint`; they are recorded separately as resident mapped files. |

## Agent-harness metrics (`--agentic`)

Off by default, so existing results and their comparability are untouched. `compare_engines.py --agentic` (or
`bench.py --agentic`) adds these, from `agentic.py`, with the same fairness rules as the core phases (explicit sampling,
thinking off, a nonce right after the system start on cold prompts, middle-window decode rates, the GPU lock):

| Metric | How |
| --- | --- |
| Cold start | `compare_engines` measures spawn to ready, then three ~1K-token requests before any warm-up: spawn to first token, and the first request's penalty over the median of the others. The file cache is not purged (that needs root), so this is a warm-file-cache start. |
| Sweep | Cold requests at `--sweep-sizes` (default 1000,2000,4000,8000,16000,max; max = the largest that fits `--max-ctx`), `--sweep-repeats` each: TTFT, prefill tok/s and decode tok/s at that context; decode retention = largest over smallest. |
| Cache | A fixed ~`--cache-tokens` prefix: cold TTFT, a follow-up in the grown conversation (what a chat or tool loop sends), three new questions on the same prefix, then a second conversation and the first again: does the prefix survive? Engines that keep only live state or turn checkpoints reuse the follow-up but not a new question on a shared prefix, so the two are reported apart. Engines that report no cached tokens show `-`, a reported 0 shows 0%. The agent replay's follow-up cache hit rate (cached over prompt tokens) is added too. The last turn depends on how many conversations the server keeps live at once (its `--slots`, which `compare_engines` reads from the config's command and `compare_table` prints next to the result): with one slot the second conversation evicts the first, so the first one re-prefills; that is the configuration, not a lost prefix, and the run carries a flag saying so. Kapitsa's `q35-4b-kapitsa` passes `--slots 1` for parity with llama.cpp, `q35-4b-kapitsa-4slots` measures the engine's own default (4). |
| Sustain | Back-to-back decodes for `--sustain-s` (default 300), last phase; first-third vs last-third median tok/s, flagged above 10% falloff, with `pmset` thermal state before and after. Battery power is flagged; Low Power Mode makes the run INVALID. |
| Memory | Idle footprint of the server before any request next to the peak, so the KV cost is visible. |

`--skip sweep,cache,sustain` drops one; `compare_table.py` writes an extra table and `_agentic.csv`. Smoke runs
(labels starting `smoke`) stay out of the tables.

## What makes a run count

Each engine has different hidden defaults. The harness sends everything explicitly and checks the answers:

- **One engine at a time.** Every run holds an exclusive lock on `/tmp/kapitsa-gpu.lock` (BSD `flock`, the same
  lock `lockf -k` takes; `BENCH_GPU_LOCK` names another file) from server start until its memory is released. Run
  every other GPU job on the Mac under the same lock (`lockf -k /tmp/kapitsa-gpu.lock <command>`), or close it.
  The default lives in `/tmp`, not `~/.cache`, so that an unlocked `compare_engines.py` and a `lockf -k
  /tmp/kapitsa-gpu.lock` direct run in another session or tree take the same file: `flock` and `lockf` on one path do
  block each other (checked on macOS 26: `lockf -k` waits out a Python `fcntl.flock(LOCK_EX)` holder of the same file),
  and two GPU jobs that take different files do not exclude each other at all.
  A `BENCH_GPU_LOCK=~/.cache/nace-gpu.lock` left in a `local.env`, shell profile or launchd job from before keeps that
  caller on the old file: remove it.
  To wait in `lockf` or put a time limit on a run, take the lock outside and run one engine's configs per call, e.g.
  `lockf -k -t 3000 /tmp/kapitsa-gpu.lock env NACE_GPU_LOCK_HELD=1 python3 compare_engines.py <configs> --only <labels>`:
  the harness then does not take the lock itself, and the lock is released between engines.
- **Quiet GPU.** After the warm-up the harness waits (up to 5 minutes) until no other process uses more than 5% of
  the GPU for 15 s and the server itself is idle; if that never happens, it measures anyway and lets the monitor flag
  the run. A contention monitor samples every process's GPU time every 2 s during the benchmark; foreign load above
  5% for three samples in a row, or on average, flags the result `CONTENTION` (per phase, with the culprit).
  WindowServer and *ambient* processes (macOS's photo and media analysis daemons, plus anything in
  `BENCH_GPU_AMBIENT`) never block or flag a run; their load is recorded separately.
- **Thinking off, checked.** Every request sends both `chat_template_kwargs.enable_thinking=false` and
  `reasoning_effort: "none"` (Ollama and LM Studio honour only the second, mlx-lm and MTPLX only the first). Any
  reasoning text, or `<think>` in the answer, makes the run INVALID.
- **Same sampler everywhere.** Every request sets temperature, top_p, top_k, `min_p 0`, presence and frequency
  penalty 0 and `repeat_penalty 1.0`, because llama.cpp defaults `min_p` to 0.05 and Ollama's qwen3.6 tag sets
  presence_penalty 1.5. `repetition_penalty` is never sent (mlx-lm adds a logits processor for any non-zero value).
- **Same prompts.** Long prompts are cut from the frozen corpus (`corpus/`, sha256 recorded in every result). Cold
  prompts must come back within ±1% of the reference token count (catches truncation and chat-template drift) and
  must not report cached tokens.
- **Honest decode numbers.** A decode run must end on `length` at exactly 512 tokens (else flagged); every text is
  saved, and a distinct-4-gram ratio below 0.5 flags a repetition loop, which inflates speculative decoding.
- **Warm-up.** One-time costs are paid before timing: a ~2.9K-token prompt with a 64-token answer and a tool call;
  for engines whose speculation adapts across requests (Ollama's MLX runner, MTPLX, oMLX's drafter), also three full
  decodes.
- **Provenance.** Each result records the exact server command and environment (secrets redacted), detected engine
  versions, weight files and sizes, macOS version, power source and power mode, swap, GPU memory, the harness's git
  revision, every contention sample and the server log lines that matter.

`compare_table.py` takes, per config, the newest run with no INVALID reason and no contention flag (else the newest
valid one, marked), and marks invalid cells with `!`.

## Agent session

The other phases time one thing at a time. A coding agent does them all at once: a long system prompt with tool
definitions, then turn after turn in which the conversation grows by file contents and test output and the model
answers briefly, mostly with a tool call. Every turn sends the whole conversation again, so what decides an agent's
speed is how much of it the engine reuses from its cache, how fast it reads the rest and how fast it writes. The
agent phase (`agent_replay.py`, bench.py's last phase) replays a recorded session of a real agent.

**The session.** [agent-traces/opencode-ledger.json](agent-traces/opencode-ledger.json) is
[opencode](https://opencode.ai) 2.0.28 fixing [agent-task](agent-task/README.md), a small Python library whose test
suite fails from five bugs in four files. It was recorded on Kapitsa serving Qwen3.6-35B-A3B at 4-bit, greedy, with
thinking off: opencode ran with the provider configuration `kapitsa opencode` writes (effort `none`), in an empty home
folder (no user configuration, skills or instruction files), against `tools/agent_record.py proxy`, which logs every
`/v1/chat/completions` request opencode sends and the length of every answer. opencode fixed all five bugs in 19
requests (all 35 tests pass, the tests unchanged): the prompt grows from 5,485 to 22,779 tokens, mostly by the files
it read and the test output, and the answers total 3,577 tokens (18 tool-call turns and a closing summary). The proxy
showed the model the paths the trace keeps (`/work/task` for the project, `/work/tmp`, `/home/user`; `--rewrite`) and
translated the tool calls back for opencode, so the recording and every replay ask for the same answers: replayed
on Kapitsa, all 19 answers come back token for token. `agent_record.py build` refuses to write a trace that still
names the user, the Mac or the home folder.

[agent-traces/index.json](agent-traces/index.json) maps a config's model to the session it replays; a model without
an entry gets no agent phase. Qwen3.8-27B replays the 35B's session too: recorded the same way on the 27B, opencode
repeated one shell command from its 14th request on until the proxy stopped it at 80, the task unsolved (greedy
decoding can lock a model into a loop). With another model's answers in the history, every engine's own answer differs
from the recorded one at almost every turn, so on the next request each engine restores its cache to the end of the
previous prompt and reads the recorded answer; that holds for every engine alike, the one that recorded nothing
included. On the recording model the recording engine has an edge: its answers are the recorded ones, while another
engine's greedy answers differ wherever rounding decides a close call (9 to 13 of 19 turns on the 35B), and it reads the
recorded answer instead of finding its own in its cache (`same_answer` per turn shows where).

**The replay.** The requests go out in the recorded order with the recorded messages, never the engine's own
answers, so every engine reads exactly the same conversation: the same system prompt and tools, the same files and
test output. Each is streamed, greedy (temperature 0, top_k 1, the rest neutral), with thinking off by both switches,
and with `max_tokens` equal to the recorded answer's length: an engine that writes the same answer stops where the
recording stopped, one that writes another (different weights or arithmetic) stops at its own end or at that length,
and its token count is recorded. Between requests the replay waits as long as the agent did while it ran its tools
(at most 5 s each), so engines that use the pause (Kapitsa reads its last answer into the cache) get the time they
would get in real use. One `X-Session-Id` for the whole session.

The tool definitions go into the system message exactly as the Qwen chat template renders them (`--agent-tool-mode
rendered`, the default), and the requests carry no `tools` field: the prompt is token for token the one the template
builds from the request with tools, it is identical on every engine, and the engine streams every token it writes
(engines hold a parsed tool call back until its closing tag, so with `tools` the first token of a tool-call answer
would arrive with the whole call on some engines and not on others). `--agent-tool-mode api` sends the requests as
recorded, tools included. Ollama parses tool calls from Qwen's output even without `tools`; on such turns the
first event is the whole call, and the replay treats them as below.

**Measured.** Per request: the time to the first token (TTFT: reading whatever the engine could not reuse from its
cache, plus the first token), the writing rate (the tokens after the first streamed event over the time to the last
event) and the total time. Per session: the total time (the sum of the requests; with the agent's pauses it is the
wall time), the prefill time (the sum of the TTFTs) and the decode time (the rest), the mean writing rate (all answers'
window tokens over all their window time), the first request's TTFT (a cold prompt of about 5.5K tokens) and the median
and p90 TTFT of the follow-ups (the requests that continue the previous one). A turn whose first event is a held tool
call has no TTFT of its own: it is left out of the TTFT statistics and its prefill time is estimated as its time minus
its tokens at the session's writing rate (`prefill_estimated_turns` in the result). The same goes for a turn whose
first event carries most of the answer (MTPLX, asked without tools, replaces a tool call it will not run with a
notice it sends at the end). An answer whose end was never streamed (fewer characters than tokens, ending on a small
event: Ollama drops a tool call that `max_tokens` cut short) has its writing window run to the end of the stream. Each answer is also compared with the recorded one (its text and tool calls, raw or
parsed; `same_answers`), and the time a stream stays open after its last token is summed (`after_last_token_s`).

**Checks.** Thinking text in an answer, a failed request, or a prompt shorter than 90% of the reference length (the
engine cut the conversation) make the session INVALID; a prompt more than 2% off the reference is flagged. The
reference lengths are stored in the trace, counted with Qwen3.6's tokenizer and chat template
(`tools/agent_trace_tokens.py`, which also checks that the rendered and the API forms of every request are the same
text). The quiet-GPU wait and the contention monitor apply as to every phase (the phase is `agent` in the result). A
session still running after 40 minutes (`--agent-timeout`) stops and is INVALID. The longest request plus its answer
is 23,101 tokens: every config's 32K context holds it. compare_engines.py flags runs of LM Studio and MTPLX that
start with less than 10 GiB of free disk, below which they keep no prompt cache on disk (`provenance.system_start.disk_free_bytes`).

**Running it.** `python3 compare_engines.py <configs> --skip decode,greedy,prefill,tools` runs only the agent
session; `--skip agent` leaves it out of a full run; `--agent-trace FILE` replays another trace. By hand, against a
running server: `python3 agent_replay.py --base-url ... --model ...` (`--max-turns 3` for a quick check).

**Recording another session.** Serve the model (under the GPU lock), put the proxy in front of it, run the agent
against the proxy in a copy of the project, then build the trace and its reference lengths:

```bash
kapitsa serve --model 35b --port 8291 --ctx 65536 &
W=$(mktemp -d) && cp -R agent-task/project $W/task
python3 tools/agent_record.py proxy --listen 127.0.0.1:8299 --upstream http://127.0.0.1:8291 --log rec.jsonl \
    --greedy --no-think --rewrite "$W/task=/work/task" --anonymize &
# opencode in $W/task with an empty HOME and the provider `kapitsa opencode` writes, its baseURL http://127.0.0.1:8299/v1
(cd $W/task && opencode run --standalone --auto -m 'kapitsa/qwen3.6-35b-a3b#none' --title ledger) < agent-task/prompt.txt
python3 tools/agent_record.py build rec.jsonl --workdir $W/task --out agent-traces/my-trace.json
python3 tools/agent_trace_tokens.py agent-traces/my-trace.json --model-dir ~/models/qwen3.6-35b-a3b-4bit --write
```

## Weights

Every MLX engine reads the same 4-bit weights (MLX affine 4-bit, group 64), from one folder per model under
`BENCH_MODELS` (default `~/models`). llama.cpp and Ollama read Ollama's Q4_K_M GGUF of each model.

```bash
M=~/models   # = BENCH_MODELS
# Qwen3.6-35B-A3B: mlx-community/Qwen3.6-35B-A3B-4bit with its MTP head (MTPLX's release of it)
hf download Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed --revision 95460516fbfe34c48f577f67c9ed191f5d61fbaf --local-dir $M/qwen3.6-35b-a3b-4bit
# Qwen3.8-27B: lmstudio-community's 4-bit trunk, plus the MTP head and MTPLX runtime file of MTPLX's release
hf download lmstudio-community/Qwen3.8-27B-MLX-4bit --revision 6067b15cf581666a4aecf6af3afaba4bb5efc20c --local-dir $M/qwen3.8-27b-4bit-mtp
hf download Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed mtp.safetensors mtplx_runtime.json --revision 1d5087d2062c02b279180a53e4016cf9cd7a3d7e --local-dir $M/qwen3.8-27b-4bit-mtp
# DFlash2 drafters (Kapitsa and oMLX)
hf download incoai/Qwen3.6-35B-A3B-DFlash2 --revision 51ef7b6923ad6c14cb1bb41c37a9041446496ab6 --local-dir $M/drafters/Qwen3.6-35B-A3B-DFlash2
hf download incoai/Qwen3.8-27B-DFlash2 --revision 015e795645c74b1a0eeef3b570031fb62e769bc5 --local-dir $M/drafters/Qwen3.8-27B-DFlash2
# Qwen3.5-4B: lmstudio-community's 4-bit trunk plus Qwen's own MTP head (the files `kapitsa pull 4b` puts in
# ~/.cache/kapitsa/models/qwen3.5-4b-4bit; a link to that folder works too)
hf download lmstudio-community/Qwen3.5-4B-MLX-4bit --revision c43ee1d65576a5d98de1e8405cac93c371a655c1 --local-dir $M/qwen3.5-4b-4bit-mtp
python3 ../../tools/mtp_sidecar.py Qwen/Qwen3.5-4B $M/qwen3.5-4b-4bit-mtp/mtp.safetensors
# ... and for MTPLX and Ollama's MLX runner the same trunk with the MTP head and runtime file of MTPLX's 4B release
mkdir $M/qwen3.5-4b-4bit-mtplx && cp -c $M/qwen3.5-4b-4bit-mtp/*.json $M/qwen3.5-4b-4bit-mtp/*.jinja $M/qwen3.5-4b-4bit-mtp/model.safetensors $M/qwen3.5-4b-4bit-mtplx/
hf download Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed mtp.safetensors mtplx_runtime.json --revision 550153beba237bb1f2f6ccba31ccedfe48ff78ec --local-dir $M/qwen3.5-4b-4bit-mtplx
# ... and for oMLX's Lightning MTP a view of that folder whose index lists the head (no data copied)
python3 tools/omlx_mtp_farm.py $M/qwen3.5-4b-4bit-mtplx $M/qwen3.5-4b-4bit-omlx-mtp
# Qwen3.5-4B at bf16 (q35-4b-bf16-*): the files `kapitsa pull 4b:bf16` puts in ~/.cache/kapitsa/models/qwen3.5-4b-bf16
# (mlx-community/Qwen3.5-4B-bf16 @491fdc7 plus the same MTP head); MTPLX reads clones of them plus its 4B runtime file,
# and oMLX's DFlash drafter is z-lab's DFlash v1 drafter for the 4B
ln -s ~/.cache/kapitsa/models/qwen3.5-4b-bf16 $M/qwen3.5-4b-bf16-mtp
mkdir $M/qwen3.5-4b-bf16-mtplx && cp -c $M/qwen3.5-4b-bf16-mtp/* $M/qwen3.5-4b-bf16-mtplx/ && cp -c $M/qwen3.5-4b-4bit-mtplx/mtplx_runtime.json $M/qwen3.5-4b-bf16-mtplx/
hf download z-lab/Qwen3.5-4B-DFlash config.json model.safetensors --revision 9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf --local-dir $M/drafters/Qwen3.5-4B-DFlash
# Qwen's 4B MTP head at 4 bits, which mlx-serve's 4B folder reads (tools/mlx_serve_pack.py qwen3.5-4b, under mlx-serve)
hf download mlx-community/Qwen3.5-4B-MTP-4bit config.json model.safetensors --revision ab6f59bc6627196c611ab8851638651078170485 --local-dir $M/drafters/Qwen3.5-4B-MTP-4bit
# GGUF (llama.cpp and Ollama): see "Ollama"
```

These are the files the published runs read (checked against the Hub at these revisions: every file by size, the
MTP heads and drafters by sha256). Folders may be symlinks.

## Engines

Each engine runs at its latest release at the time of measuring and with its own best settings for these models:
speculative decoding where it has it (each MTP depth measured where it is a knob), 32K context, one conversation
slot, flash attention where it is a switch. The configs' `note` fields say why each setting was chosen. Versions in
brackets are the ones in [docs/benchmarks.md](../../docs/benchmarks.md); `version: auto` records what you have.

### Kapitsa (`configs/kapitsa.json`)

Build this checkout (`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`); the config uses
`build/kapitsa`, else `kapitsa` on PATH, or `KAPITSA_BIN`. Default settings, with the DFlash2 drafter named
explicitly so a missing drafter fails the run instead of falling back to MTP; `--slots 1` for parity with the other
engines' single slot. The version is the checkout's git revision. The SSD prefix cache (on by default) keeps its
states in the harness's state directory, emptied before and after each run as oMLX's SSD KV cache is: the agent
session replays a trace's own messages, so a later run would otherwise read what an earlier one left.

### llama.cpp (`configs/llama.cpp.json`) [0.7.0, build b11669]

The official release (`llama-<build>-bin-macos-arm64.tar.gz` from github.com/ggml-org/llama.cpp/releases) or
`brew install llama.cpp`; set `LLAMA_SERVER` unless `llama-server` is on PATH. Four configs per model: plain and MTP
with `--spec-draft-n-max` 1, 2 and 3 (the MTP head is inside the GGUF; for the 4B it is the Ollama tag's separate
draft GGUF, passed with `-md`, and n-max 6, unsloth's recommendation for this model, is a fifth config). Settings:
`-c 32768 -np 1 -ngl 999 -fa on -b 2048 -ub 2048 -lm none --jinja`, the official Qwen chat template from the MLX
folder (Ollama's GGUFs carry none), `--min-p 0`. `-lm none` loads the weights instead of mapping them, so they count
in the memory column like everyone else's. `flag_check` maps flags a newer llama-server renamed;
`--dry-run --check-flags` shows the mapping.

### Ollama (`configs/ollama.json`) [0.40.5]

The Ollama app (ollama.com or github.com/ollama/ollama/releases); `ollama` on PATH or the app is found, else set
`OLLAMA_BIN`. The harness starts its own `ollama serve` on another port over your model store, with 32K context, one
slot and flash attention, so the app's settings do not leak in.

- **GGUF, MTP on (as shipped):** `ollama pull qwen3.6:35b-a3b-mtp-q4_K_M`, `ollama pull qwen3.8:27b-mtp-q4_K_M` and
  `ollama pull qwen3.5:4b-mtp-q4_K_M` (explicit tags: on Apple silicon the bare tags resolve to the MLX builds).
  Ollama's own MTP depth is used: 2 for the 35B, 4 for the 27B, 1 for the 4B (whose MTP head is a separate draft
  GGUF in the tag, `{ollama_gguf_draft:...}`). These blobs are also llama.cpp's weights.
- **GGUF, MTP off:** a derived tag on the same blob, e.g. for the 35B a Modelfile with
  `FROM qwen3.6:35b-a3b-mtp-q4_K_M` and `PARAMETER draft_num_predict 0`, then
  `ollama create qwen3.6-35b-a3b-nomtp -f Modelfile` (and `qwen3.8-27b-nomtp`, `qwen3.5-4b-nomtp` likewise).
- **MLX runner:** Ollama's MLX runner on the same MLX 4-bit weights as the other engines, imported into a separate
  store without copying by `tools/ollama_mlx_import.py` (APFS clones with Ollama's tensor names; needs numpy and
  the store on the same volume as the weights; `tools/ollama_mlx_verify.py` checks it tensor by tensor). Start a
  dedicated server on that store, then import (the 27B: `--renderer qwen3.8 --parser qwen3.5 --requires 0.32.12`,
  default `--mtp-quant`, `presence_penalty 0`; the 4B: `--src $M/qwen3.5-4b-4bit-mtplx --name
  nace/qwen3.5:4b-mlx-4bit-mtp`, default `--mtp-quant`, otherwise as the 35B):

  ```bash
  S=~/models/ollama-mlx   # = OLLAMA_MLX_MODELS
  OLLAMA_HOST=127.0.0.1:8193 OLLAMA_MODELS=$S OLLAMA_NOPRUNE=1 ollama serve &
  python3 tools/ollama_mlx_import.py --src ~/models/qwen3.6-35b-a3b-4bit --name nace/qwen3.6:35b-a3b-mlx-4bit-mtp \
      --models $S --work /tmp/ollama-mlx-work --mtp-quant 4:32 --renderer qwen3.5 --parser qwen3.5 --requires 0.19.0 \
      --params '{"min_p": 0, "presence_penalty": 1.5, "repeat_penalty": 1, "temperature": 1, "top_k": 20, "top_p": 0.95}'
  ```

  The runner's MTP depth adapts across requests and has no knobs.

### LM Studio (`configs/lmstudio.json`) [llmster 0.0.25-1, MLX runtime 1.13.1]

Headless LM Studio: `curl -fsSL https://lmstudio.ai/install.sh | bash`, then `lms runtime update --all`. Get LM Studio's
staff picks as its users do: `lms get qwen/qwen3.6-35b-a3b`, `lms get qwen/qwen3.8-27b` and `lms get qwen/qwen3.5-4b`,
MLX 4-bit variants (lmstudio-community's MLX repos: the same shards as the MLX folders above, so you can link those
files into `~/.lmstudio/models/lmstudio-community/<repo>/` instead of downloading them again, but keep the hub entry:
only its `model.yaml` lets `reasoning_effort: "none"` switch thinking off). `launchers/lmstudio_serve.sh` starts a fresh
daemon per run, loads the model with `--context-length 32768 --parallel 1`, starts the server and streams LM
Studio's logs. Its MLX engine has no speculative decoding for these models, and nothing else to tune. Its prompt
cache keeps the checkpoints that serve follow-up turns on disk (an unlinked file in /tmp) only while at least 10 GiB
of the disk is free; below that, a follow-up turn of these hybrid models reads the whole conversation again.

### mlx-lm (`configs/mlx-lm.json`) [0.32.2, mlx 0.32.4]

`uv tool install mlx-lm` (or a venv with `pip install mlx-lm`); set `MLX_LM_SERVER` unless `mlx_lm.server` is on PATH.
mlx-lm has no MTP or drafter path for these models. Prompt cache for follow-up turns, 2048-token prefill steps,
`HF_HUB_OFFLINE=1` (the server downloads whatever model a request names otherwise).

### oMLX (`configs/omlx.json`) [0.7.1rc1, mlx 0.32.4]

The release wheel from github.com/jundot/omlx/releases into its own venv (`uv venv && uv pip install <wheel URL>`);
set `OMLX_BIN`. Best measured setting: DFlash2 speculative decoding with the same drafter as Kapitsa, quantized to
4-bit at load, block 5 (the DFlash2 authors' oMLX recipe), verify mode `adaptive` on the 35B and `dflash` on the 27B
(whichever was faster in smoke tests). On the 4B, Lightning MTP (oMLX's built-in MTP path) with the MTP head of
MTPLX's 4B release, through the index view `tools/omlx_mtp_farm.py` makes; it was faster than DFlash with z-lab's
DFlash v1 drafter for this model at any block size and verify mode tried. Plain configs for comparison. The harness
builds oMLX's model folder and settings file in its state directory and empties oMLX's SSD KV cache before each run.

### Nemotron 3.5 Bolt (`configs/bolt.json`) [oMLX 0.7.1rc1, mlx-vlm 0.7.8, mlx 0.32.4]

One folder, `nemotron-3.5-bolt-24gb`, for every engine: NVIDIA's 24 GB checkpoint with its LiLiCorr drafter in
`dflash/`, plus the MLX model file the MLX engines need (`nemotron_bolt.py`, Trust Remote Code). The MLX bundle
carries NVIDIA's files byte-identical, so it serves all three:

```bash
hf download txgsync/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-MLX --revision 98c0cb759b3e6d7ee6f962a7282901a804498e54 --local-dir $M/nemotron-3.5-bolt-24gb
```

(`kapitsa serve --model bolt` downloads NVIDIA's own repos instead: `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`
and its `-LiLiCorr` drafter.) oMLX and mlx-vlm are the engines that load this checkpoint on a Mac; oMLX's drafter
path works only greedy in 0.7.1rc1, so `bolt-omlx-lilicorr-greedy` measures it at temperature 0 throughout. The
oMLX + drafter launcher (`launchers/omlx_bolt_serve.sh`) follows the bundle's README: load, unload, load again with
the drafter. `MLX_VLM_PYTHON` names a Python with mlx-vlm.

### Nemotron 3.5 Super-VL (`configs/super-vl.json`) [mlx-vlm git main 73d366e, oMLX 0.7.1rc1, mlx-lm 0.32.2, llama.cpp 0.7.0]

A 96 GB+ model (70 GB of weights); the published runs used a 128 GB M5 Max. Kapitsa and mlx-vlm read NVIDIA's own
NVFP4 checkpoint, the folder `kapitsa pull super-vl` writes (the LiLiCorr drafter in `dflash/`), linked as
`$M/nemotron-3.5-super-vl-nvfp4`. mlx-vlm's Super-VL support (PR #2528, with its native MTP drafter) is newer than its
0.7.8 release, so `MLX_VLM_MAIN_PYTHON` names a venv with git main
(`uv pip install "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm@73d366ecf923"`). oMLX and mlx-lm cannot read
NVIDIA's checkpoint (model type `nemotron_h_omni`) and run the text-only MLX 4-bit conversion instead; llama.cpp runs
ggml-org's official Q4_0 GGUF, its MTP head and NVIDIA's LiLiCorr drafter converted by ggml-org:

```bash
hf download rapid-mlx/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-MLX-4bit --revision 016789035d80736393d9bfcdb63d6911002e424d --local-dir $M/super-vl-mlx-4bit-rapid
hf download ggml-org/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf \
    mtp-NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf lilicorr-NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4.gguf \
    --revision 3c91ad1be60865386af92e8147dafb32470b097d --local-dir $M/super-vl-gguf   # or set SUPER_VL_GGUF_DIR
```

The configs' notes give each choice (mlx-vlm's MTP block size and llama.cpp's drafter depths come from smoke tests).
The 4K / 16K reference prompt lengths (4183 / 16513) were counted with this model's tokenizer and chat template.

### Nemotron 3.5 Lightning (`configs/lightning.json`) [mlx-vlm 0.7.8 and git main 73d366e, oMLX 0.7.1rc1, mlx-lm 0.32.2, llama.cpp 0.7.0]

Kapitsa and mlx-vlm read NVIDIA's own NVFP4 checkpoint, the folder `kapitsa pull lightning` writes (the DSpark drafter
in `dflash/`), linked as `$M/nemotron-3.5-lightning-nvfp4`; mlx-vlm 0.7.8 runs it plain and with the DSpark drafter,
and its MTP drafter for Nemotron-H is newer than that release, so `lightning-mlxvlm-mtp` uses `MLX_VLM_MAIN_PYTHON`
(git main 73d366e, as for Super-VL). oMLX and mlx-lm cannot read NVIDIA's checkpoint (ModelOpt `MIXED_PRECISION`:
NVFP4 experts with FP8 Mamba projections) and run mlx-community's MLX 4-bit conversion; oMLX's MTP path runs a 4-bit
conversion that keeps the MTP head. llama.cpp runs ggml-org's official Q4_0 GGUF with its MTP head, and NVIDIA's
DSpark drafter as a GGUF:

```bash
hf download mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit --revision 55ac8c89261109b36c04371cd3f479a4594208c8 --local-dir $M/lightning-mlx-4bit
hf download EigenLabs/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit-mtp --revision 153c469494b3f16a0eaa752dae25cfc9452ce5c6 --local-dir $M/lightning-mlx-4bit-mtp
hf download ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf \
    mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf --revision 8a08a1c81dadcc75d35dbb96016cfd344b632e67 --local-dir $M/lightning-gguf
hf download magnitudedev/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark-GGUF --revision ed35fc990759330e325ab9e41bcf80737ab4dafc --local-dir $M/lightning-gguf/dspark
```

(`LIGHTNING_GGUF_DIR` names another GGUF folder.) The configs' notes give each choice; the speculative depths and
block sizes come from smoke tests. The 4K / 16K reference prompt lengths (4184 / 16517) were counted with this
model's tokenizer and chat template.

### Qwen3.8-Flash-Next (`configs/flash-next.json`) [oMLX 0.7.1rc1, mlx-vlm 0.7.8, llama.cpp 0.7.0, Ollama 0.40.6]

A 128 GB model (105.9 GB of weights, a 32 GB n-gram table among them); the published runs used a 128 GB M5 Max.
Kapitsa, mlx-vlm and oMLX read the same folder, Nace AI's MLX pack (the one `kapitsa pull flash-next` downloads),
linked as `$M/qwen3.8-flash-next-4bit`. oMLX also runs its author's own quantization (oQ4e), llama.cpp ggml-org's
official IQ4_NL GGUF with its MTP head, and Ollama the MLX tag it serves on Apple silicon and its GGUF tag:

```bash
hf download nace-ai/Qwen3.8-Flash-Next-MLX-4bit --revision 5d423f9f563c1f1b7ca7945b268e65c0b92b0eae --local-dir $M/qwen3.8-flash-next-4bit
hf download Jundot/Qwen3.8-Flash-Next-oQ4e-mtp --revision c4ddfb54a967a9bc74d7da16425f55f0872f8aac --local-dir $M/qwen3.8-flash-next-oq4e-mtp
hf download ggml-org/Qwen3.8-Flash-Next-GGUF Qwen3.8-Flash-Next-IQ4_NL-00001-of-00002.gguf Qwen3.8-Flash-Next-IQ4_NL-00002-of-00002.gguf \
    mtp-Qwen3.8-Flash-Next-Q4_0.gguf --revision 0a61d2b8d851e0bb9f22b973befd714413d63c6f --local-dir $M/qwen3.8-flash-next-gguf
ollama pull qwen3.8-flash-next:125b-mlx      # into OLLAMA_MODELS; then a tag that only sets the context:
printf 'FROM qwen3.8-flash-next:125b-mlx\nPARAMETER num_ctx 32768\n' > Modelfile
ollama create nace/qwen3.8-flash-next:125b-mlx-32k -f Modelfile
ollama pull qwen3.8-flash-next:125b-a6b-q4_K_M   # Ollama's GGUF runner (no MTP draft in this tag)
```

(`FLASH_NEXT_GGUF_DIR` names another GGUF folder.) The Ollama tag's own parameters set `num_ctx 262144`, which overrides
`OLLAMA_CONTEXT_LENGTH`, hence the derived tag. mlx-vlm 0.7.8 cannot take the pack folder itself as its MTP drafter
(its splitter builds the head's 8-bit and 6-bit layers as 4-bit, because the pack names their precisions with the
`mtp.` prefix the drafter's modules lack), so `flash-mlxvlm-mtp` reads a drafter folder made from the pack's own head
with mlx-vlm's splitter, the same 74 tensors byte for byte:

```bash
$MLX_VLM_PYTHON - $M/qwen3.8-flash-next-4bit $M/qwen3.8-flash-next-4bit-mlxvlm-mtp <<'EOF'
import json, sys
from pathlib import Path
import mlx.core as mx
from mlx_vlm.speculative.drafters.qwen4_exp_mtp.split import Qwen4ExpMTPSplitter
pack, out = Path(sys.argv[1]), Path(sys.argv[2])
config, weights = Qwen4ExpMTPSplitter().prepare(pack, source_config=json.load(open(pack / "config.json")))
q = {k[4:] if k.startswith("mtp.") else k: v for k, v in config["quantization"].items()
     if not isinstance(v, (dict, bool)) or k.startswith("mtp.")}
config["quantization"] = config["quantization_config"] = q
out.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(str(out / "model.safetensors"), {k.removeprefix("mtp."): v for k, v in weights.items()}, metadata={"format": "mlx"})
json.dump(config, open(out / "config.json", "w"), indent=1)
EOF
```

The configs' notes give each choice; the speculative depths and block sizes come from smoke tests. The 4K / 16K
reference prompt lengths (4133 / 16286, the harness's defaults) were counted with this model's tokenizer and chat
template.

### MTPLX (`configs/mtplx.json`) [2.12.2]

`uv tool install mtplx` (or `pip install mtplx`); set `MTPLX_BIN` unless `mtplx` is on PATH. The settings are the
ones the weights' `mtplx_runtime.json` recommends: profile `sustained`, depth 1 for the 35B (whose folder is MTPLX's
own release), and `turbo`, depth 3 for the 27B (the runtime file of MTPLX's 27B release). Inline session post-commit,
so follow-up turns hit its session cache.

### mlx-serve (`configs/mlx-serve.json`) [26.10.2, mlx 0.32.3]

The release binary (`mlx-serve-bin-macos-arm64.tar.gz` from github.com/ddalcu/mlx-serve/releases, unpacked; or
`brew install mlx-serve`); set `MLX_SERVE_BIN` unless `mlx-serve` is on PATH. Its model folders are built from the
weights above (APFS clones) plus the vendor's files that differ, which the script downloads (and checks):

```bash
python3 tools/mlx_serve_pack.py qwen3.6-35b-a3b   # into <models>/mlx-serve/: --models DIR, else $BENCH_MODELS, else ~/models
python3 tools/mlx_serve_pack.py qwen3.8-27b
python3 tools/mlx_serve_pack.py qwen3.5-4b --head-src $M/drafters/Qwen3.5-4B-MTP-4bit/model.safetensors
```

On the 35B that is the vendor's pack itself: the same trunk shards (sha256-identical) with its MTP sidecar
`mtp/weights.safetensors`. On the 27B the vendor's pack is its own conversion (bf16 embedding, some tensors
quantized differently, 18.2 GB), so the folder holds the trunk every engine reads plus the vendor's MTP head (its 29
`mtp.*` tensors, byte for byte, as a sidecar) and the vendor's `drafter/` (the same file as the DFlash2 drafter above).
The vendor publishes no 4B pack: that folder is the 4B trunk every engine reads plus mlx-community's 4-bit MTP head
(above) with its tensors renamed to `mtp.*`, as the vendor's 35B pack does with the 35B head. Its DFlash config reads
z-lab's 4B drafter, a DFlash v1 drafter, which mlx-serve detects from its config.

Two setups per model, at mlx-serve's own defaults except where short runs just before the session measured faster (five
sampled decodes each; the [tuning table](../results/2026-10-18-mlx-serve/README.md#tuning)):

- **MTP:** the model's MTP head with exact verification (the default; the opt-in lossy `--mtp-typical` and
  `--mtp-tokenv3` verifiers change the output distribution) and `--no-drafter`. `--mtp-max-depth 2` on the 35B and
  4 on the 27B: a few percent faster than the adaptive default (cap 6 on an M4 Pro).
- **DFlash2:** the drafter named with `--drafter` (mlx-serve quantizes it to 8 bits at load on M1-M4) and
  `--no-mtp`: with the head loaded, a request that arrives while the previous one is still finishing on the server, as
  the benchmark's back-to-back requests do, is handed to MTP. On the 27B, a tree of up to 7 drafts a round (block 8,
  the defaults). On the 35B, `--draft-block-size 4` and no tree: mlx-serve binds DFlash drafters to MoE trunks, but its
  break-even bar for MoE verifies (1.8 drafts accepted per round) switched DFlash off early in every timed answer.
- PLD (prompt lookup) stays on, its default; mlx-serve runs one speculative mode per request (DFlash, else MTP, else
  PLD), so it never runs in these setups.

A fresh `HOME` per run (in `{state}`): no saved settings apply, and the speculation cost table mlx-serve learns while it
runs (`~/.mlx-serve/round-cost/`) starts empty; the warm-up's three full decodes train it. 32K context, one sequence
(its default `--max-concurrent 1`).

### mlx-vlm (`configs/mlx-vlm.json`) [0.7.8, mlx 0.32.4]

`uv pip install mlx-vlm` into a venv and set `MLX_VLM_PYTHON` to its Python (the published runs used the oMLX venv's
copy of the 0.7.8 release, the build Bolt ran on). The server reads the same MLX 4-bit folders as the other engines,
one sequence (`--max-num-seqs 1`), 2048-token prefill steps, and automatic prefix caching in memory (`APC_ENABLED=1`,
`APC_DISK_ENABLED=0`; off by default), so follow-up turns reuse the conversation. On the 35B its fastest setup is DFlash2
with the drafter above (`--draft-kind dflash`); the config's `note` gives the short runs that chose it.

## Configs

`configs/*.json` are lists of configs; `compare_engines.py` merges the files you pass, in order. Keep labels stable:
tables group runs by label. The full field list is in `compare_engines.py`'s docstring; the main ones:

| Field | Meaning |
| --- | --- |
| `label`, `display`, `engine`, `model_name` | Result file name and table grouping; engine kind (`kapitsa`, `llama.cpp`, `ollama`, `lmstudio`, `mlx-lm`, `omlx`, `mtplx` get version detection; anything else works without). |
| `cmd`, `env`, `env_clear_prefixes` | Server command (a bare program name is looked up on PATH), extra environment, inherited variables to drop. `cmd: null` benchmarks a server that is already running. |
| `base_url`, `model` | OpenAI base URL and the model id requests name. |
| `extends`, `cmd_append` | Inherit an earlier config (also across files) and add server arguments. |
| `requires_paths`, `requires_ollama_tag` | Skip with a message when something is missing. |
| `prepare` | `symlink`, `write_json`, `link_files` (per-file symlinks) and `clean_dir` (only inside `{state}`) steps before start. |
| `stop`, `pid_file` | Extra stop command; pid file of a server that detaches from its launcher (its process tree is measured). |
| `adaptive_warm`, `ready_timeout`, `flag_check`, `gguf_expect` | Three extra warm-up decodes; start-up timeout; llama-server flag mapping; expected GGUF blob digest. |
| `sampling_overrides`, `no_think_style`, `bench_args` | Passed to `bench.py` for an engine that rejects a field (`{"repeat_penalty": null}` drops it). |
| `version`, `version_cmd`, `log_grep`, `note` | Fixed version string or `auto`; a command whose output is recorded; server-log lines to keep; free text. |

Placeholders, usable in `cmd`, `env`, `prepare`, `stop`, `requires_paths`, `pid_file`, `model_path` and `model`:

| Placeholder | Value |
| --- | --- |
| `{models}` | `BENCH_MODELS`, default `~/models` |
| `{repo}`, `{harness}` | This Kapitsa checkout; `bench/harness` |
| `{out}`, `{state}` | The output directory (`--out-dir`, `BENCH_OUT`, default `runs/`); its `state/` subfolder for engine state |
| `{home}` | Your home folder (a leading `~` is expanded too) |
| `{env:NAME}`, `{env:NAME=DEFAULT}` | An environment variable (or `local.env` entry), else `DEFAULT` (which may hold placeholders) |
| `{first:A\|B}` | The first alternative that is an existing file or a command on PATH |
| `{ollama_gguf:TAG\|TAG2}` | The GGUF weights blob of an Ollama tag, read from the store's manifests |
| `{ollama_gguf_draft:TAG\|TAG2}` | The tag's separate draft GGUF (its `image.draft` layer), e.g. the MTP head of `qwen3.5:4b-mtp-q4_K_M` |

Environment variables (or `local.env`): `BENCH_MODELS`, `BENCH_OUT`, `BENCH_GPU_LOCK`, `BENCH_GPU_AMBIENT`, and the
engine locations in `local.env.example`.

## Output

Under `runs/` (or `--out-dir`): `results/<label>-<time>.json` (every request's timings and text, the summaries,
flags, contention samples, memory per phase and the provenance), `logs/<label>-<time>.log` (the server's own log)
and `state/` (engine state the harness owns). `compare_table.py [results dirs]` writes `compare_table.md` (with an
agent-session table per model), `compare_table.csv` and `compare_table_agent.csv`.

## Benchmarking a server by hand

`bench.py` runs against any server that is already up and takes the GPU lock itself:

```bash
python3 bench.py --base-url http://127.0.0.1:8000/v1 --model qwen3.6-35b-a3b --label my-run --pid <server pid>
```

`--pid` measures that process tree's memory; `--decode-runs`, `--sizes`, `--skip decode,greedy,prefill,tools` and
the sampling flags change the run (`--help`). Wrap your own server's whole lifetime in the lock instead, and tell
`bench.py` so: `lockf -k /tmp/kapitsa-gpu.lock sh -c 'start server; NACE_GPU_LOCK_HELD=1 python3 bench.py ...'`.
`python3 gpu_hogs.py` lists what is using the GPU right now.

`agent_replay.py` replays the agent session alone against a running server (it takes the GPU lock the same way):

```bash
python3 agent_replay.py --base-url http://127.0.0.1:8000/v1 --model qwen3.6-35b-a3b --label my-run
```

## Files

| File | What it does |
| --- | --- |
| `bench.py` | The benchmark client: one server, all measurements, one result JSON |
| `compare_engines.py` | Runs configs one at a time: lock, start, warm up, quiet wait, `bench.py`, stop, provenance |
| `compare_table.py` | Markdown and CSV tables from result files |
| `agentic.py` | The `--agentic` phases: cold start, context sweep, cache, sustained decode and thermal state |
| `agent_replay.py` | The agent-session replay (bench.py's `agent` phase, or by hand) |
| `agent-task/` | The coding task the agent solved: the project it got, the prompt, the bugs |
| `agent-traces/` | Recorded agent sessions (`opencode-ledger.json`) |
| `tools/agent_record.py` | The recording proxy and the trace builder (paths and names sanitized) |
| `tools/agent_trace_tokens.py` | Reference prompt lengths of a trace with a model's tokenizer and chat template (needs transformers) |
| `gpu_hogs.py` | GPU load per process (ioreg), the GPU lock, the quiet wait and the contention monitor |
| `configs/` | One file per engine, both models |
| `corpus/` | The frozen prompt corpus and its notice (Apache-2.0 excerpts of Hugging Face Transformers) |
| `launchers/lmstudio_serve.sh` | LM Studio's daemon as a foreground benchmark server |
| `launchers/omlx_bolt_serve.sh` | oMLX with Bolt's LiLiCorr drafter (load, unload, reload with the drafter), then the harness port |
| `tools/mlx_serve_pack.py` | mlx-serve's model folders from the shared weights (clones) and the vendor files that differ |
| `tools/ollama_mlx_import.py`, `tools/ollama_mlx_verify.py` | Zero-copy import of MLX weights for Ollama's MLX runner, and its check |
| `tools/omlx_mtp_farm.py` | Zero-copy view of an MLX folder whose index lists its side-car MTP head, for oMLX's Lightning MTP |
| `local.env.example` | Paths for this Mac, copied to `local.env` |
