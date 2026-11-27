# Rust in-process SDK

An owning Rust interface to [Kapitsa's C ABI](../README.md), with no Cargo dependencies.
`Engine` loads a local model, `Request` runs once, and `CancelHandle` cancels cooperatively
from another thread. Options and outputs use the C ABI's JSON strings; applications may use
their preferred JSON library. This is a local path crate, not a published crates.io package.

## Build

Build the native library on Apple silicon with macOS 14+ and Xcode Command Line Tools:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKAPITSA_BUILD_SDK=ON
cmake --build build -j 6 --target kapitsa_sdk
export KAPITSA_LIB_DIR="$PWD/build"
export DYLD_LIBRARY_PATH="$KAPITSA_LIB_DIR"
cargo build --manifest-path sdk/rust/Cargo.toml
```

The crate needs Rust 1.74 or newer. `KAPITSA_LIB_DIR` is a build-time library search directory;
it does not locate models or configure the runtime loader. For a packaged application, embed
`libkapitsa_sdk.dylib` and configure its runtime search path/signing with the application.
The SDK's ABI version is checked before constructing an engine.

Add a local dependency to the application's `Cargo.toml`:

```toml
[dependencies]
kapitsa = { path = "../kapitsa/sdk/rust" }
```

```rust,no_run
use kapitsa::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::new(r#"{"model_dir":"/local/model","slots":2}"#)?;
    let mut request = engine.request(
        r#"{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}"#,
    )?;
    let result = request.generate_with(|delta_json| {
        println!("{delta_json}");
        true // false requests cancellation
    })?;
    println!("{result}");
    Ok(())
}
```

`Engine::new`, generation and final teardown block. Run them on application-owned worker
threads when a UI or async executor must remain responsive. Distinct requests can generate
on different threads up to the native engine's slot count. The crate starts no HTTP server,
subprocess or async runtime and changes no inference defaults.

## Ownership and cancellation

`Request::cancellation()` returns a cloneable handle. Move it to another thread and call
`cancel()` to set the native flag; cancellation does not join generation. A cancellation
handle keeps the request and its engine alive even after the `Request` or `Engine` wrapper
is dropped. Drop every handle to release the engine's GPU lock. There is one live native
engine per process, including engines retained through requests or cancellation handles.

Generation borrows the request mutably, so safe Rust cannot destroy it while generation is
running or start a second generation concurrently. A second call after completion returns
`ErrorKind::Busy`. The callback receives borrowed UTF-8 JSON valid for that call; copy it if
it must outlive the callback. It runs synchronously on the generation thread. Starting another
generation from that callback returns `ErrorKind::Busy` without consuming the other request;
run it after the callback returns. Cancellation from a callback is allowed. With Rust's
unwinding panic mode, a callback panic requests cancellation, waits for native return,
frees native output strings and then resumes unwinding in Rust. `panic=abort` still aborts.

Errors expose `kind`, `message` and `partial_result`. In particular, cancellation after output
begins can return partial JSON in the error, while pre-cancellation has no result. Final
results are authoritative: native UTF-8 repair means concatenated deltas need not equal final
text when a character crosses a delta boundary. Native deferred Metal errors invalidate the
engine: drop all handles before creating another. Other fatal native errors can still terminate
the process; applications requiring crash isolation must provide their own worker process.

## Checks

With the native library built, the CPU checks need Rust, Clippy, rustfmt, C++ and Python 3:

```sh
python3 sdk/rust/test.py build
```

The shared C ABI test double exercises Unicode callbacks, native error strings and partial
results, pre-cancellation, callback cancellation/panic, nested-generation rejection, cancellation from another thread,
concurrent requests, engine retention and final destruction. Its live-object/string counter
must return to zero. A separate run links the real library and rejects invalid inputs before
model loading. The test double is not evidence of GPU or model correctness.

The ignored model check must be explicitly authorized and uses the native cooperative GPU
lock. Supply valid engine options for an already downloaded model (two slots, speculation off):

```sh
export KAPITSA_TEST_OPTIONS='{"model_dir":"/local/model","slots":2,"context":4096,"spec_depth":0,"dflash":"none","disk_cache":"0"}'
cargo test --manifest-path sdk/rust/Cargo.toml --test model -- --ignored
```

It covers actual streaming, one-shot use, cancellation, two requests, retained-engine use and
reload. These are functional checks, not an inference speed or numerical-equivalence benchmark.
