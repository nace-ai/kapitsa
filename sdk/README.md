# Kapitsa in-process C SDK

An opt-in C ABI around `ni::Engine` for desktop applications. Create an engine once, reuse
its loaded model across requests, stream deltas and cancel from another thread. Models must
already exist locally; the library does not start HTTP servers or child processes.

## Build and use

Apple Silicon, macOS 14+, Xcode Command Line Tools and CMake 3.24+:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKAPITSA_BUILD_SDK=ON
cmake --build build -j 6 --target kapitsa_sdk sdk_contract_test sdk_model_test load_error_test
ctest --test-dir build -R '^(sdk_contract_test|load_error_test)$' --output-on-failure
```

Include `sdk/c/kapitsa.h` and link `libkapitsa_sdk.dylib`. For a C source file:

```sh
cc client.c -I sdk/c -L build -lkapitsa_sdk -Wl,-rpath,"$PWD/build" -o client
```

The call sequence is `kapitsa_engine_create` → `kapitsa_request_create` → `kapitsa_generate`.
`kapitsa_generate` blocks and invokes the optional callback on the calling thread. Return 0
from the callback to cancel, or call `kapitsa_request_cancel` from another thread. Distinct
requests can generate on separate threads using the engine's native slots. Destroy each request
after generation has returned, then destroy the engine. [The real-model test](c/model_test.cpp)
shows reuse, streaming, concurrent requests and cancellation through these C functions.

## Interface and ownership

The [C header](c/kapitsa.h) defines the eight exported functions, options, request fields,
statuses and ownership. Requests retain an engine after its public handle is released.
One live SDK engine is allowed per process; attempting to create another returns BUSY immediately,
regardless of the lock timeout. Do not mix separate users of Kapitsa's global GPU runtime in one process.

The SDK takes the cooperative GPU file lock. `gpu_lock_timeout_ms` defaults to 0: if another
holder owns it, creation returns BUSY immediately. A positive value permits waiting up to that
many milliseconds, returning BUSY on expiry. Other lock failures return ERROR. This deadline
covers lock waiting only; model loading remains synchronous and can take longer. Run creation
on a worker thread if the UI must stay responsive. Existing CLI callers keep their blocking wait.
Timed waits poll `flock(LOCK_NB)` about every 5 ms without FIFO guarantees; a CLI process already
queued in blocking `flock(LOCK_EX)` may acquire the lock first, so an SDK caller can time out
under contention while queued CLI processes acquire the lock.
The lock remains held while the engine or a retained request exists, including idle periods.
A host already holding it sets `NI_LOCK_HELD=1` or `NACE_GPU_LOCK_HELD=1` to avoid acquiring it twice.

During `kapitsa_engine_create`, synchronous loader and Metal initialization failures return
`KAPITSA_ERROR` with their diagnostic: missing or damaged weights, unsupported model data,
insufficient memory, and device, library or pipeline creation failures. Joined loading workers
return failures to the creating thread. Failed construction closes its files, releases its
model buffers and GPU lock, and allows another creation attempt. Malformed SDK options still
return `KAPITSA_INVALID_ARGUMENT`.

Recovery is limited to construction. Fatal errors during generation or in independent
background services (including streamed-expert I/O) can still terminate the host process.
Applications requiring crash isolation should load the library inside their own worker process.
The CLI and `serve` do not enable SDK recovery and keep their existing fatal-error behavior.

## Request and result contract

Omitted sampling fields use the loaded model's defaults. The SDK rejects unknown top-level
fields; its JSON is a library interface, not an OpenAI HTTP request body.

| Engine option | Meaning |
| --- | --- |
| `model_dir` | Required local model folder containing `config.json` and the model files. |
| `context` | Token capacity per slot; default 32768, subject to the engine's memory plan. |
| `gpu_lock_timeout_ms` | Lock wait only: integer 0–2147483647 ms; default 0 returns BUSY immediately. Positive values wait until acquired or the deadline, then return BUSY. |
| `slots` | Concurrent conversation slots, 1–4; SDK default 1. More slots need more memory. |
| `prefill_chunk` | Rows per prefill pass; 0 asks the engine to choose. |
| `spec_depth` | -1 uses the model default; 0 disables speculative decoding. |
| `mem_budget_gb` | GPU memory budget in GB; 0 uses the native default. |
| `dflash` | Native drafter selection: `auto`, `none`, or a local directory; empty inherits native configuration. |
| `disk_cache` | SSD prefix-cache budget such as `16G`, `auto`, or `0`; empty inherits `NI_DISK_CACHE`, otherwise off. |

Requests accept `messages`, `tools`, `template_kwargs`, `max_tokens`, `temperature`, `top_p`,
`top_k`, `seed` and `stop`. `messages` is a nonempty array of messages with a string `role`.
Image content uses the existing engine's message handling. Structured-output schemas and model
switching are not exposed by ABI version 1.

A streaming callback receives `content`, `reasoning` and complete `tool_calls` deltas.
The final result contains the accumulated text/tool calls, `finish_reason`, `error`,
`prompt_tokens`, `cached_tokens`, `completion_tokens`, `prefill_ms` and `decode_ms`.
Each tool call has `id`, `name` and `arguments`; `arguments` is JSON **text**, not a parsed object.
`finish_reason` is `stop`, `length`, `tool_calls`, `cancelled` or `error`.

C results and errors are separately allocated strings. Free every non-null output, even when
status is an error. A request cancelled before generation starts has no result string; cancellation
after output begins can return a partial result. Invalid UTF-8 bytes are replaced with U+FFFD.
Consequently concatenated deltas need not equal final text when a character is split across deltas;
use the final result as the complete response. No byte-perfect streaming claim is made.

Cancellation is cooperative: it sets a flag and returns immediately. Join/wait for generation
before destroying the request. The same request cannot generate twice. A cancellation racing
with a completed response can leave that response successful. C handles cannot be destroyed
concurrently with another operation on the same handle. The host must enforce these lifetime rules.

## Validation

CI builds the C library and the test executables, runs CPU checks with no model/GPU,
and checks that only the eight C symbols are exported. The CPU tests cover malformed JSON,
integer bounds, split UTF-8, lock contention, timeout expiry, acquisition before the deadline,
caller-held locks, lock I/O errors, joined loading failures and cleanup after failed construction.
`load_error_test --gpu` checks Metal compilation/pipeline errors, retries and buffer rollback.

The real-model test is deliberately excluded from CTest. With explicit GPU authorization and
an idle GPU, run:

```sh
MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 build/sdk_model_test /local/model-directory
```

It uses a temporary `BENCH_GPU_LOCK` file and exercises long prefill, prefix reuse, callback
cancellation, pre-cancellation, one-shot requests, three two-slot rounds with a third thread
cancelling, and requests retaining a released engine and its lock. It checks the C ABI directly
and needs no language wrapper. To check ANE availability separately, run without Metal validation
and report the startup checks or fallback actually observed. These are functional tests, not
performance benchmarks or exhaustive concurrency stress; record the Mac, model and commit in the PR.
