# Splash as an engine backend

Kapitsa can serve a model through a Splash engine child process (`splash serve-native`) instead of its own Metal
runtime. Kapitsa keeps the HTTP server, the API translations and the output parser, and renders and tokenizes the chat
(with the template and tokenizer files from Splash's model folder, see "The template"); Splash keeps the weights, the GPU
and the decode loop. This page records the shape and what the first version does not do.

## Constraints

- One backend per engine, fixed when the engine starts. A request is never moved to the other one, and a request the
  backend cannot serve is refused before any work starts (`invalid_request`), not run on the other.
- Kapitsa owns prompt rendering and tokenization. Splash gets token ids, as its own frontend sends them.
- State a backend persists (SSD prefix cache, expert store) is not shared with the other: Splash keeps its own cache in
  its own `--cache-dir`; the native `engine_disk` states are not read or written in this mode.
- Native Kapitsa does not link or depend on Splash or MLX. `src/splash/` has no Metal and no naceinfer dependency.

## Pieces

| Piece | File | State |
|---|---|---|
| Wire codec, protocol v11 (client frames, strict event decoder, incremental parser) | `src/splash/protocol.*` | done, tested against bytes from Splash 1.4.0's `protocol.py` (`tools/splash_protocol_test`) |
| Child process: spawn, READY, submit, cancel, status, draining shutdown, death handling | `src/splash/runtime.*` | done, tested against a fake engine (`tools/splash_runtime_test`) |
| Engine splash mode: tokenizer, template, stop ids, no `model_` (`EngineOptions::splash_model`) | `src/engine/engine_splash.cpp`, `engine.cpp` | done, built; exercised by `tests/test_splash_engine.py` |
| `chat_splash`: render, tokenize, stream TOKENS through `OutputParser`, DONE to `ChatResult`; `native_execution` picks it when the engine is in splash mode | `src/engine/engine_splash.cpp`, `src/server/native_execution.cpp` | done; reply, reasoning split, max_tokens, stop string, streaming, refusals and client abandon checked against a fake engine |
| Server binding: `kapitsa serve --engine splash [--splash-binary P] [--splash-model DIR] [--splash-arg A]...` (skips memory planning, model pick, download, routing and the disk cache) | `src/server/main.cpp` | done |
| Discovery: the binary and the model folder when the flags are not given, with an error that names what is missing | `src/splash/discover.*` | done, tested on a throwaway tree (`tools/splash_discover_test`) |
| A/B against plain Splash: identity (text, and token ids through the tap) and speed | `tools/splash_ab.py`, `tools/splash_tap.py` | done; see Verification |
| CPU test: `tests/test_splash_engine.py build/kapitsa` with `tests/fake_splash.py` (a protocol-v11 stand-in); without a Splash model folder it uses `tests/fixtures/splash_model` | `tests/` | done; run in `ci.yml` by this PR |

## Runtime contract

- `start()` spawns `argv` (built as Splash's `server.py:_native_command` builds it: `serve-native <model dir> <ctx|auto>
  <mem|auto> ...`) with stdin and stdout as pipes and stderr inherited, then waits for READY.
- A request's events reach its `Sink` from the reader thread, in order, ending in exactly one `on_done` or `on_error`.
- The child dying, closing its output, sending a frame that breaks the protocol, or an engine-level ERROR fails every
  held request with `engine_unhealthy` and leaves the Runtime dead. There is no relaunch: the owner starts a new one.
  A protocol failure kills the child, because its stream cannot be trusted afterwards.
- Admission follows READY's `max_concurrent_requests`; a full pool throws `PendingLimitExceeded` (the server answers 503). The id and both deadlines
  are stamped at `submit`, as `runtime.py` does.
- `shutdown()` closes the child's stdin (it drains and exits), then SIGTERM, then SIGKILL, each after `stop_grace_s`.

## What the first version refuses

Structured-output grammar (needs MASK_REQUEST round trips), image inputs, loop counts, and a prompt longer than the
engine's context. Each is a 400 `invalid_request_error` before anything is queued, as is a sampling field outside Splash's
range (the error names the field). A request that finds the engine at its `max_concurrent_requests` gets 503 with
`Retry-After`. Kapitsa's own model folders and Splash's resolved model are different artifacts (a safetensors
`flash-next-q20` against a GGUF Q2_0 target plus an incoai DFlash2 drafter), so the child is pointed at Splash's resolved
model directory (`--splash-model`), never at Kapitsa's weights.

## Finding Splash

Without `--splash-binary`, `serve` looks for `engine/splash` under `$SPLASH_HOME`, under the folder a `splash` on `PATH`
resolves into, then `~/Applications/Splash`, `/Applications/Splash` and `~/Splash`. Without `--splash-model` it uses the
only resolved model under `~/Library/Application Support/Splash/models/.resolved/` that has a tokenizer; with several it
lists them (family and folder name) and exits, and `--splash-model` takes a folder or the first 6 or more characters of
its name. When nothing is found the error says where it looked and which flag to give.

## The template

Kapitsa's `flash-next-q20/chat_template.jinja` and Splash's resolved `tokenizer/chat_template.jinja` are not the same file
(Splash's merges consecutive system messages and maps `high` to `xhigh`), and the two `tokenizer.json` files differ in
vocabulary size (Splash's 248320 against Kapitsa's 248044) and in how merges are written. So splash mode loads the tokenizer and the template
from Splash's model folder, so the prompt is the one Splash would build.

## Verification

Machine: M5 Pro, 48 GB, macOS 26.5.2, Splash 1.4.0 (Qwen3.8 Flash-Next), `--verify-rows all`, temperature 0, 128 output tokens,
one server at a time under the GPU lock.

- CPU: `splash_protocol_test`, `splash_runtime_test`, `splash_discover_test` and `tests/test_splash_engine.py` (the
  server against `tests/fake_splash.py`, with Splash's real tokenizer and template). No model or GPU.
  They run locally (`ctest`, or the binaries in `build/`); CI runs the four C++ tests and `tests/test_splash_engine.py` (`ci.yml`).
- Tokenizer (`tools/splash_tokenizer_check.py`): Kapitsa loading Splash's `tokenizer.json` gives the ids the `tokenizers`
  library gives, on 7 texts (Latin, Cyrillic, CJK, emoji, code with tabs, digits, URL, special-token text, NBSP and CR),
  0 mismatches.
- Identity by text (`tools/splash_ab.py`, 3 chat prompts): prompt and completion token counts are equal to plain Splash's
  (63/72, 69/160, 67/160), and each side repeats itself. Prompt 3's text is byte-identical; prompts 1 and 2 differ only by
  one trailing newline on the reasoning text, which plain Splash keeps and Kapitsa's output parser strips. The HTTP APIs
  return no ids, so `--tap` puts `tools/splash_tap.py` in front of the engine binary on both sides and compares each
  request's prompt ids and output ids at the native pipe. It was run once (27 Oct 2026, M5 Pro 48 GB, macOS 26.5.2, Splash 1.4.0), at T=0 with seed 1, `--verify-rows all` and `--expert-memory 16 GiB` on both sides. That is not Splash's default memory plan; the expert cache size does not change what is computed, and the default plan was not run for this comparison. Three requests (66, 1031 and 4070 prompt tokens, 16, 128 and 95 output tokens): prompt ids and output ids are identical on both sides, no first divergence, ids md5 `47c93518…`, `605fa935…`, `5f762994…`. One run per side, so it is not a statistical claim.
- Speed: measured in PR #751 (`bench/results/2026-10-28-m5-pro-flash-next-engines/`), not here. There Kapitsa `--engine splash`
  (`7e478053` and `ff6151c8`) and plain Splash 1.4.0 are level within noise, n 6 to 10 per cell, with the raw legs and
  commands in that folder. Medians for `ff6151c8` against its paired plain Splash (K2 against S2), all at `--expert-memory 16 GiB --idle-release off`, `--verify-rows` not passed (engine default `auto`), M5 Pro 48 GB, macOS 26.5.2, 27 Oct 2026:

  | | TTFT 1K, s | TTFT 4K, s | prompt tokens / TTFT, 4K, tok/s |
  |---|---|---|---|
  | K2 against S2 | 1.45 against 1.47 | 3.45 against 3.55 | 1181 against 1149 |

  The last column is the prompt's token count divided by TTFT (`prefill_tps_median` in `summary.json`). TTFT also holds the
  request's fixed cost (template, tokenizing, the first decode step, HTTP), so this is not prefill throughput: the same
  engine reads 732 tok/s at 1K and 1286 at 4K below. Compare the TTFT columns; this one is only a ratio of the two engines.

  Decode (T=0 and T=0.6) and turn-2 TTFT for both Splash arms: see `bench/results/2026-10-28-m5-pro-flash-next-engines` (#751),
  which is the record of these runs and of its caveats (other sessions were using the Mac). This shows neither a speed-up nor a
  slowdown from the bridge.
  An earlier single run of this branch before the split (n=6, commit not recorded) is superseded by it and no longer quoted.
- Default memory plan (no `--expert-memory`; 33.0 GiB free and reclaimable at start; n=1 per size, so no comparison):
  engine up in 8.9 s, 1K TTFT 1.41 s (732 tok/s as prompt tokens / TTFT), decode 74.9 tok/s; 4K TTFT 3.17 s (1286 tok/s the same way), decode
  61.8 tok/s; the child's peak RSS 1.9 GiB.

- Second turn on a ~4K chat (`tools/splash_ab.py --turn2`, 3 reps per side, Splash's default memory plan, M5 Pro, 27 Oct 2026).
  Turn 1 is 4072 prompt tokens; turn 2 is the same chat plus the reply (content only, as clients send it) and a short
  question. Plain Splash: turn-2 TTFT 0.26, 0.25, 0.19 s, 4064 cached tokens. An earlier version of `chat_splash` put the
  reasoning back into the history, as the native path does: 0.38, 0.42, 0.32 s with the same 4064 cached tokens and prompts
  of 4183 to 4257 tokens (plain Splash's: 4122 to 4131). Splash caches prompt blocks, not outputs, so the longer prompt only
  cost prefill. `chat_splash` therefore sends the client's history as it is, as plain Splash does: turn-2 TTFT 0.26, 0.25,
  0.20 s, prompts 4124, 4131, 4122 tokens, 4064 cached. One run of 3 reps per row; not a tool-call chat.
  The turn-2 speed belongs to Splash's own prefix cache; Kapitsa's bridge only avoids breaking it. Turn-2 TTFT for both Splash arms
  on a 4K tool-call chat: see `bench/results/2026-10-28-m5-pro-flash-next-engines` (#751). It does not reproduce the 0.2 s of
  these three reps on a plain chat, and the gap was not isolated. Quote #751, not these three reps.

## Options

- `--splash-persistent-cache`: passes `--persistent-cache on` and `--cache-dir ~/Library/Caches/Kapitsa/splash-prefix-cache/<name>-<hash>`
  (one directory per model: the model folder's real path, symlinks and a trailing slash resolved, gives the name and a 64-bit hash of the path, so two folders with the same name elsewhere do not share one) unless `--splash-arg --cache-dir` is given; a `--splash-arg --persistent-cache` (on or off) wins and the flag adds
  nothing. It writes to the user's disk; off by default.
- A request's own `cache_salt` (`/v1/chat/completions`; a string, else a 400 naming the field) is hashed as Splash's frontend does
  (BLAKE2b, 16 bytes, person `splash-salt`, checked against Python's `hashlib` in `splash_plan_test`), so requests share cached
  prompts only with the same salt, as with plain Splash, whatever the flags below. It wins over the conversation salt. The
  other two endpoints do not read the field yet.
- `--splash-conversation-salt`: a cache salt per conversation (hash of the messages up to the first user message), so
  conversations stop sharing cached prompts such as a common system prompt, for requests that bring no `cache_salt`. Off by default, which is Splash's behaviour
  without a `cache_salt`.
- The child gets `--idle-release off` unless given, and, when free and reclaimable memory is between 22 and 38 GiB and no
  `--expert-memory` is given, `--expert-memory` of free GiB minus 14 (at least 8). The formula rests on one measured pass
  (16 GiB at 30.4 GiB free) and Splash's 2 GiB protected for macOS; it is not validated across memory states. At 33.0 GiB free the default plan started and ran without it (Verification), so
  from 33 to 38 GiB this replaces a working default with a smaller expert cache, and no run compares the two. The 38 GiB bound is
  a margin over one refusal seen at 35.2 GiB under a system pressure warning, not a measured need. In that window the forced `--expert-memory` turns Splash's own "not enough free
  memory to start" refusal into an engine that starts and streams experts from the SSD, so it is slower than a full cache; pass
  `--splash-arg --expert-memory <size>` to choose, or free memory first. Kept at 38 GiB on purpose: one default-plan start at
  33.0 GiB and one refusal at 35.2 GiB mean the default plan does not reliably start above 33, and a failed start is worse than
  an unmeasured slowdown. Follow-up: revisit with an A/B of the default plan against the forced cache at 33 to 38 GiB free.

Splash refuses to start when the Mac has too little free and reclaimable memory. Its `not enough free memory to start ...`
is the child's own line on stderr; `serve` then exits 1 with `the Splash engine did not start: engine_exited: the splash engine
closed its output`. Plain Splash stops the same way. The default-plan run above started with 33.0 GiB free and reclaimable, so
on this Mac the threshold is at or below that figure; no other threshold is claimed here. It follows memory pressure, not one
number, and what is free depends on what else is running.
