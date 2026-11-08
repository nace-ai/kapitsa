# Kapitsa C API

A minimal C interface for embedding Kapitsa inference in C, Objective-C and Swift programs.

## Why

Apps on Apple silicon that want local inference today link llama.cpp or MLX. Kapitsa is faster, but it only
exists as a CLI and an HTTP server. A C API lets an Xcode project or a Swift package call it directly, without
starting a server.

## Header

[`include/kapitsa.h`](../include/kapitsa.h) declares the full interface:

| Function | What it does |
|---|---|
| `kapitsa_init()` | One-time library setup (Metal device, thread pool) |
| `kapitsa_load(opts, &ctx)` | Load a model directory into GPU memory |
| `kapitsa_generate(ctx, opts, cb, ud)` | Stream a completion token by token through a callback |
| `kapitsa_cancel(ctx)` | Stop a running generation from any thread |
| `kapitsa_free(ctx)` | Release the model and GPU memory |
| `kapitsa_version()` | Library version string |
| `kapitsa_last_error(ctx)` | Last error message |

### Types

- **`kapitsa_ctx`**: opaque. Holds the loaded model, KV cache and engine state. One per model; not thread-safe.
- **`kapitsa_status`**: enum: `OK`, `INVALID_ARG`, `IO`, `OOM`, `MODEL`, `CANCELLED`, `INTERNAL`.
- **`kapitsa_load_opts`**: model path, context length, thread count, GPU memory budget.
- **`kapitsa_generate_opts`**: prompt, temperature, max tokens, system prompt.
- **`kapitsa_token_cb`**: `bool cb(token, len, userdata)`. Return `false` to stop.

## Swift package

The repo root has a `Package.swift` that wraps the C header as a Swift module (`CKapitsa`).
A consuming project adds it as a local package dependency:

```swift
// Package.swift
.package(path: "../kapitsa"),
.target(name: "MyApp", dependencies: [
    .product(name: "Kapitsa", package: "kapitsa"),
]),
```

The C library (`libkapitsa.a` or `.dylib`) must be built with CMake and linked by the Xcode project.
A higher-level Swift wrapper with `async/await` streaming is a follow-up.

## Implementation status

`src/lib/kapitsa_lib.cpp` wraps the Engine: `kapitsa_load` creates an `Engine` with the model directory,
`kapitsa_generate` builds a `ChatRequest` and calls `Engine::chat()` with a streaming callback, and
`kapitsa_cancel` sets an atomic flag the engine's cancellation callback reads. Tested on the 4B model:
load, stream 8 tokens, cancel mid-stream (tools/kapitsa_c_test.c).

## Design notes

- **Streaming, not blocking.** `kapitsa_generate` calls back for each token. The caller controls the loop
  (UI update, cancellation check). A blocking `generate_sync` that returns the full text is easy to build on
  top and can be added later.
- **One context per model.** Loading is expensive (seconds); generating is cheap. A long-lived context amortises
  the load and keeps the KV cache warm across calls.
- **No global state besides init.** Multiple contexts (different models) can coexist.
- **C, not C++.** The header is plain C so it works with Swift's C importer, Objective-C, and any FFI.
  The implementation is C++ because the Engine is.
