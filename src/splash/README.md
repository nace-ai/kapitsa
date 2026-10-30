# Developer note: the child-process engine mode

This note is for contributors. Users run Kapitsa (`kapitsa serve`), which decodes with KFlash; `docs/usage.md` is their guide.

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
| Structured output: Kapitsa's grammar answers the engine's `MASK_REQUEST`s (`TOKEN_MASK` constraint) in place of Splash's llguidance constraints | `src/engine/engine_splash.cpp`, `src/splash/protocol.*`, `src/splash/runtime.*` | done; masks, a refused token, `ignore_eos` and narrow masks checked against a fake engine. Not verified on a GPU |
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
`Retry-After` (queueing past the pool, as plain Splash's `--queue-size` does, is #756). Once the engine has exited, every
route (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`) answers 503 `the Splash engine has exited (...); restart
kapitsa serve`, without `Retry-After`, since nothing relaunches it. On all three routes an omitted temperature, top_p or
top_k is Splash's default (1.0, 0.95, 20), and a seed follows Splash's `frontend.py`: absent or -1 is random, any other
integer is used as its 64 bits. Kapitsa's own model folders and Splash's resolved model are different artifacts (a safetensors
`flash-next-q20` against a GGUF Q2_0 target plus an incoai DFlash2 drafter), so the child is pointed at Splash's resolved
model directory (`--splash-model`), never at Kapitsa's weights.

## Finding Splash

Without `--splash-binary`, `serve` looks for `engine/splash` under `$SPLASH_HOME`, under the folder a `splash` on `PATH`
resolves into, then `~/Applications/Splash`, `/Applications/Splash` and `~/Splash`, and last the binary of the last run
(`splash.json`). Without `--splash-model` it uses the only resolved model under
`~/Library/Application Support/Splash/models/.resolved/` that has a tokenizer, else the last run's, else the most recently
resolved one; it says which on stderr and how to pick another, and it does not exit. `--splash-model` takes a folder, the
first 6 or more characters of a folder's name, or a family or source name such as `flash-next` or `27b` (the most recently
resolved folder of that model; two different models matching is an error that lists them). A run that starts writes its
binary and model to `<kapitsa home>/splash.json` (`$KAPITSA_HOME`, else `~/.cache/kapitsa`). `serve --engine splash --check`
prints the binary and model that would run and what is missing, starts nothing and writes nothing; it exits 0 when ready and
1 when not. When nothing is found the error says where it looked and which flag to give.

## KFlash

`packaging/splash/` keeps Kapitsa's changes to Splash as patches against one pinned Splash release, and `build.py`
builds the patched engine into `<kapitsa home>/engines/splash-kflash-<tag>-<id>/` ([its README](../../packaging/splash/README.md)).
The patches are off unless the child's environment turns them on, so the same binary runs as stock Splash without it.
`kapitsa serve --engine splash` picks the engine (`src/splash/discover.h`, `pick_engine`):

- `--splash-kflash auto` (default): the KFlash build, with `SPLASH_KSX_H1=3` and `SPLASH_KSX_KERNELS`, for the models
  below, when it is built, `--splash-binary` is not given and the stock engine it replaces is the release KFlash is
  built on (its `release.json` says 1.5.0). Any other model, no build, or another Splash release (1.4.0, say) runs stock
  Splash, and `kapitsa serve` prints why.
- `on`: the KFlash build for any model. `off`: stock Splash.

What KFlash measured against the Splash releases is in [docs/splash-backend.md](../../docs/splash-backend.md).

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
  request's prompt ids and output ids at the native pipe. It was run once (27 Oct 2026, M5 Pro 48 GB, macOS 26.5.2, Splash 1.4.0, Kapitsa commit not recorded), at T=0 with seed 1, `--verify-rows all` and `--expert-memory 16 GiB` on both sides. That is not Splash's default memory plan; the expert cache size does not change what is computed, and the default plan was not run for this comparison. Three requests (66, 1031 and 4070 prompt tokens, 16, 128 and 95 output tokens): prompt ids and output ids are identical on both sides, no first divergence, ids md5 `47c93518…`, `605fa935…`, `5f762994…`. One run per side, so it is not a statistical claim.
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

## Flag reference and measurements

`kapitsa serve --engine splash` runs [Splash](https://github.com/incoai/splash) as the engine and this server as the
front: Splash keeps the weights, the GPU and the decode loop; Kapitsa keeps the HTTP server and its chat page, the
OpenAI and Anthropic APIs, streaming and the output parser, and renders the prompt with Splash's own tokenizer and
chat template, so the model, the prompt and the tokens are Splash's. Its engine is a child process (`splash
serve-native`) spoken to over its native protocol on a pipe, so Kapitsa links neither the engine nor its library and
Splash's own Apache-2.0 licence and notices apply ([THIRD_PARTY_NOTICES](../../THIRD_PARTY_NOTICES)); the ANE declarations
and the few Metal kernels adapted from Splash keep their own file headers.

```bash
kapitsa serve --engine splash                            # the engine and the model, found where Splash installs them
kapitsa serve --engine splash --check                    # what it would run, and what is missing: nothing is started
kapitsa serve --engine splash --splash-model flash-next  # one of several: a family, a source name or the start of a folder name
```

| Option | Default | Effect |
| --- | --- | --- |
| `--engine native\|splash` | `native` | `splash` runs text generation in a Splash child process; `native` is Kapitsa's own engine. |
| `--splash-binary PATH` | found | The Splash binary that runs `serve-native`: `engine/splash` under `$SPLASH_HOME` (that variable set to a folder without it is an error, not a reason to look elsewhere), the install a `splash` on `PATH` resolves into, then `~/Applications/Splash`, `/Applications/Splash`, `~/Splash`, and last the binary the last run used, so a Splash installed anywhere needs this once. A path given here must be an executable file. |
| `--splash-model NAME` | found | The model to run: a folder, or a name. A name is the first 6 or more characters of one of Splash's resolved folder names, or a part, in any case, of a model's family or of its source (`model` in its `model.json`): `flash-next`, `27b`, `Q2_0`. A name may hold a `/`, as a source such as `org/Model:Q2_0` does. When Splash resolved that model more than once, the newest folder is used; two different models matching is an error that lists them, and so is a name that matches none. Nothing is downloaded. Not given, it is the only folder under `~/Library/Application Support/Splash/models/.resolved/` that has a tokenizer, else the one the last run used, else the most recently resolved, and `serve` says which on stderr and how to pick another. The folder's own tokenizer and chat template are what this server renders with, so the prompt is the one Splash would build. |
| `--check` | off | With `--engine splash`: prints the engine binary, the resolved models (family, folder name and source), the model that would run, and for whatever is missing the error and the flag that fixes it, `Model to run: <folder>`, with several models and none named the note that says which one was picked (`the most recently resolved of N`, or `the one the last run used`) and how to pick another, then `Ready: run the same command without --check`. It refuses what `serve` refuses, such as a `--splash-binary` that is no executable file. Exits 0 when both are found and 1 otherwise, starts nothing and writes nothing, the last-run record included. |
| `--splash-arg ARG` | none | One more argument for `splash serve-native`, repeatable (`--splash-arg --verify-rows --splash-arg 2`). Kapitsa adds `--idle-release off` and sometimes `--expert-memory` itself (below); an argument given here replaces its own. |
| `--splash-persistent-cache` | off | Splash's SSD prefix cache kept across restarts: `--persistent-cache on` and `--cache-dir ~/Library/Caches/Kapitsa/splash-prefix-cache/<folder name>-<hash>`, unless `--splash-arg` gives either. The hash is of the model folder's real path, so two folders with one name elsewhere do not share a cache. Off, the child starts with an empty cache and writes nothing to disk. |
| `--splash-conversation-salt` | off | A cache salt per conversation (a hash of its messages up to the first user message), so conversations stop sharing cached prompts such as a common system prompt; off is one shared cache, as Splash without `cache_salt`. It separates conversations; it is not a speed option. |
| `--splash-kflash auto\|on\|off` | `auto` | KFlash's Splash engine: Splash with Kapitsa's patches, built by `packaging/splash/build.py` into `<kapitsa home>/engines/` ([packaging/splash/README.md](../../packaging/splash/README.md)). `auto` runs it for the models an interleaved A/B measured faster with it ([KFlash](#kflash)) when it is built, `--splash-binary` is not given and the stock Splash it replaces is the release KFlash is built on (1.5.0; another Splash release is kept), and says so; `on` runs it for any model; `off` runs stock Splash. |

All seven Splash options need `--engine splash` and are an error without it. A `serve` that starts writes the binary and
model it used to `splash.json` in Kapitsa's home (`~/.cache/kapitsa`, or `$KAPITSA_HOME`), so the next run needs neither
flag. Splash's model folder is the model:
`--model`, `--split` and `--allow-switch` are usage errors in this mode, and this Mac's memory plan, the model catalog,
downloads and routing to another Mac do not apply.

Splash's own default frees the weights after ten idle minutes and the next request then waits for them to come back,
so the child is given `--idle-release off` unless `--splash-arg` says otherwise. Splash takes `off` or a number of
seconds, not `on`.

When this Mac is short of memory, Kapitsa also sizes Splash's expert cache: with 22 to 38 GiB free and reclaimable, and
no `--expert-memory` in `--splash-arg`, the child gets `--expert-memory` of the free GiB less 14, at least 8. Below
22 GiB, or with your own `--expert-memory`, the plan is left to Splash, whose default plan has no fixed threshold: it started at 33.0 GiB free and refused at 35.2 GiB under memory pressure. The
formula rests on one measured pass (a 16 GiB cache at 30.4 GiB free) and is not validated across memory states. From 33
to 38 GiB free it replaces a default plan that can start with a smaller expert cache, which streams experts from the SSD
and is slower than a full one, and no run compares the two; pass `--splash-arg --expert-memory <size>` to choose. What
the child was given is one log line, `[kapitsa] splash: engine arguments added: ...`.

A request's own `cache_salt` on `/v1/chat/completions` (a non-empty string, else a 400 naming the field) is hashed as
Splash's frontend does, so requests share cached prompts only with the same salt, and it wins over
`--splash-conversation-salt`. The other two endpoints do not read the field.

If Splash exits during its warmup, for example because this Mac has too little free memory, `serve` exits 1 with `the
Splash engine did not start: engine_exited: the splash engine closed its output`; Splash's own reason is the line
above Kapitsa's.

Structured output (a `response_format` grammar, strict tools, `tool_choice`) goes to the engine as a request with the
`TOKEN_MASK` constraint, and Kapitsa's own grammar (`src/engine/grammar.h`) answers the engine's mask requests: row 0
is the mask after the tokens committed so far, row i the mask after the draft's simulation token i too, so drafts still
verify under masks. Splash's Python constraints (llguidance) are not involved. As in Kapitsa's own engine the grammar
starts after `</think>` when the model reasons, and a row before it allows every token; a row after a simulation token
the grammar refuses repeats the last valid row, as Splash's own `TokenConstraint` does. The reader thread only queues
a mask request, in order with the tokens, and the request's worker answers it, so a mask is computed against exactly the
tokens committed before it. A request that also sets `ignore_eos` is refused (Splash takes no constraint with it), and
one whose engine asks for masks narrower than the tokenizer fails with an error saying so. (Not verified on a GPU:
`tests/test_splash_engine.py` checks it against `tests/fake_splash.py`.)

Text chat otherwise: Splash's native protocol carries text and sampling, so a request with an image or a loop count is
refused with the error type `invalid_request_error` and the engine's reason as the message. Nothing is moved to Kapitsa's own engine, and the engine is fixed for the server's life: one
`serve` is one engine, not one per request. Two failures are not refusals: a request that arrives while the engine is
at its concurrent-request limit is a 503 with `Retry-After` and `the Splash engine is at its concurrent-request
limit`, and one that arrives after the engine has exited is a 503 without `Retry-After`, since nothing relaunches it:
`the Splash engine has exited (...); restart kapitsa serve`, on `/v1/chat/completions`, `/v1/messages` and
`/v1/responses` alike.

Splash's sampler is its own frontend's: a request that leaves out temperature, top_p or top_k gets Splash's default —
1.0, 0.95 and 20 — on all three routes, rather than the per-model ones Kapitsa's own engine applies. Sampling fields Kapitsa's own
engine does not use (`presence_penalty`, `frequency_penalty`, `repetition_penalty`, `min_p`, `ignore_eos`) are passed
to it, and a value outside Splash's range (temperature in [0, 2], top_p in (0, 1]) is a 400 naming the field rather
than a clamp. A thinking budget in the request is honoured, as Kapitsa's own engine honours it.

### Measured on an M5 Pro

The same model, Qwen3.8-Flash-Next Q2_0, four ways on one M5 Pro 48 GB: Kapitsa's server in front of Splash's engine,
plain Splash 1.4.0, Kapitsa's own engine with MTP, and llama.cpp; one engine at a time under the GPU lock, arms
interleaved, decode at temperature 0 with seed 1. No arm leads every row, and none is called best here.

| Metric | Prompt | Kapitsa on Splash | plain Splash 1.4.0 | Kapitsa native, MTP | llama.cpp |
| --- | --- | ---: | ---: | ---: | ---: |
| TTFT | 1,036 tokens | 1.45 s | 1.47 s | 1.41 s | 28.12 s |
| TTFT | 4,075 tokens | 3.45 s | 3.55 s | 4.81 s | 110.74 s |
| TTFT | 19,287 tokens | 16.54 s | 17.32 s | 25.48 s | 634.24 s |
| Prompt tokens / TTFT | 1,036 tokens | 715 tok/s | 706 tok/s | 733 tok/s | 37 tok/s |
| Prompt tokens / TTFT | 4,075 tokens | 1181 tok/s | 1149 tok/s | 847 tok/s | 37 tok/s |
| Prompt tokens / TTFT | 19,287 tokens | 1166 tok/s | 1114 tok/s | 757 tok/s | 30 tok/s |
| Decode, T=0 | 256 tokens from a short prompt | 55.9 tok/s | 55.5 tok/s | 53.7 tok/s | 9.0 tok/s |
| Turn-2 TTFT | 4K chat, second turn | 0.56 s | 0.57 s | 0.26 s | 3.34 s |
| System memory at peak | wired GiB / swap growth / max pressure | 31.6 / 0.00 GB / 1 | 31.6 / 0.00 GB / 1 | 36.5 / 4.47 GB / 4 | 8.6 / 0.00 GB / 1 |

<sub>Runs: `Mac17,8` (M5 Pro, 48 GB), macOS 26.5.2, 2026-10-27, in
[bench/results/2026-10-28-m5-pro-flash-next-engines](../../bench/results/2026-10-28-m5-pro-flash-next-engines/README.md) (PR #751) (the
table is `legs/summary.json` and `legs/table.md` there). n is the number of requests per cell: Kapitsa on Splash 10, plain
Splash 10, Kapitsa native 12 (9 for the 19,287-token prompt and turn 2, where three legs were stopped by the memory
guard), llama.cpp 2. Prompt sizes are the tokens the Splash tokenizer counted (the other arms differ by one).
**Kapitsa on Splash is `ff6151c8`, the head of PR #743 when measured (#743 merged as `f4c826f4`; the request path is the same except `cache_salt`, and the memory plan did not run because it was forced)**; plain Splash is 1.4.0, Kapitsa
native is `e840651c`, llama.cpp is `c7474afc`. Every arm ran a forced memory plan, not a default: both Splash arms with
`--splash-arg --expert-memory 17179869184 --splash-arg --idle-release off`, native with `NI_EXPERT_STORE=24`, and
other sessions shared the Mac. "Prompt tokens / TTFT" is uncached prompt tokens over time to first token, so it
includes the fixed per-request cost and is not pure prefill throughput. System memory is system-wide (wired
memory, swap growth, the highest `kern.memorystatus_vm_pressure_level`: 1 normal, 4 critical), the maximum over every
leg of the arm.</sub>

Read across it:

- **Re-measured on 2026-10-30 with KFlash and Splash 1.5.0**, with the same scripts, interleaved K S K S
  ([results](../../bench/results/2026-10-30-m5-pro-flash-next-kflash/README.md)): this server at its default `--splash-kflash auto` read 61.3 tok/s decode at T 0 and
  1,218 tok/s on the 4K prompt against plain Splash 1.5.0's 56.5 and 1,178, n 10 each. The ranges overlap on every
  cell, so it is a tie, not a measured speed-up. The table above is the 2026-10-27 run.
- **This server in front of Splash, and plain Splash, are close.** Decode medians are 55.9 and 55.5 tok/s, with single
  requests spread from 49 to 62 tok/s across the two arms. The prompt tokens / TTFT figure for the server is
  +1.3%, +2.8% and +4.7% against plain Splash at 1K, 4K and 19K, and turn-2 TTFT is 0.56 s against 0.57 s. These are
  medians of 10 requests from a few legs of one run; this is not a measured speed-up.
- **Against Kapitsa's own engine it is a different trade.** On this prompt-tokens-over-TTFT figure, Splash's engine is
  1.39 times native at 4K and 1.54 times at 19K, and level at 1K; decode medians are 53.7 tok/s native and 55.9 here, inside the
  spread; the second turn is native's, 0.26 s against 0.56 s (native cached 4435 of 4476 prompt tokens, the Splash arms 4320 of
  about 4426). Native also reached memory pressure level 4 with up to 4.5 GB of swap growth in three legs with a 24 GB
  expert store, while both Splash arms stayed at level 1 with no swap growth.
- **llama.cpp is in the table only because it ran.** Its 66 GB Q2_0 GGUF is memory-mapped on a 48 GB Mac, and both
  of its two legs started with swap already in use. It ran about four times slower than llama.cpp `0a9ef00` on the same
  file on 2026-10-24 (39 tok/s decode, not this run's `c7474afc` with `--jinja`), so the build, the flags or the memory
  state is likely, and the cause was not isolated. That is not a fair number for llama.cpp, and it is printed so that its
  absence is not read as a claim.

Output text was not compared across the four engines, and no Splash arm passed `--verify-rows`, so the engine default
`auto` applied, under which greedy and seeded outputs on this MoE target can differ between runs.

Kapitsa's own engine against plain Splash, each side with its default drafter and sampling, is measured per Mac in
[benchmarks](../../docs/benchmarks.md#m5-pro); this table is about the mode itself, at the memory plans named above rather than
the defaults.
