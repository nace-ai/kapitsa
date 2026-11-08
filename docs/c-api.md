# Kapitsa C API and Swift package

Kapitsa ships an in-process C ABI (`sdk/c/kapitsa.h`) that lets Mac apps embed
the engine without the HTTP server.

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKAPITSA_BUILD_SDK=ON
cmake --build build -j --target kapitsa_sdk
# installs libkapitsa_sdk.dylib and include/kapitsa.h
cmake --install build --prefix /usr/local
```

## API surface

| Function | What it does |
|---|---|
| `kapitsa_engine_create(json, &engine, &err)` | Load a model into GPU memory (takes the GPU lock) |
| `kapitsa_request_create(engine, json, &req, &err)` | Build a one-shot chat request |
| `kapitsa_generate(req, cb, ud, &result, &err)` | Stream deltas through a callback; blocking |
| `kapitsa_request_cancel(req)` | Cancel from any thread |
| `kapitsa_request_destroy(req)` / `kapitsa_engine_destroy(engine)` | Release resources |
| `kapitsa_string_free(s)` | Free returned strings |

Options and request bodies are JSON. Deltas arrive as JSON with `content`,
`reasoning`, `tool_calls` fields. `gpu_lock_timeout_ms` defaults to 0, which
returns `BUSY` immediately when the lock is held; positive values wait up to
that many milliseconds. See `sdk/c/kapitsa.h` for the full contract.

## Swift

A `Package.swift` at the repo root exposes `sdk/c/kapitsa.h` as a Swift module
(`CKapitsa`) via a module map in `swift/`. The module map auto-links
`libkapitsa_sdk`; pass `-Xlinker -L/usr/local/lib -Xlinker -rpath -Xlinker /usr/local/lib`
(or wherever CMake installed the dylib) so the linker finds it at build
time (`-L`) and at launch (`-rpath`).

```swift
import CKapitsa

var engine: OpaquePointer?
var error: UnsafeMutablePointer<CChar>?
let status = kapitsa_engine_create(optionsJSON, &engine, &error)
```

## Thread safety

- An engine handle is safe to share across threads for `request_create`.
- A request is single-threaded except `kapitsa_request_cancel`, which is safe
  from any thread.
- One live SDK engine per process. A second `engine_create` returns `BUSY`.
- The GPU lock is held while the engine or any retained request exists.
- Other runtime fatal errors (including live streamed-expert reads) can
  terminate the host process. Applications needing crash isolation should
  use a worker process.

## Tests

`sdk/c/contract_test.cpp` (CPU, in CI) covers the ABI version, option
validation, null guards and error paths. GPU tests (`model_test.cpp`,
`deferred_test.cpp`) need a model directory and the GPU lock.
