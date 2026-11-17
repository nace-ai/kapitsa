# Python in-process SDK

Python 3.9+ bindings for Kapitsa's [C ABI](../c/kapitsa.h), using only the standard
library at runtime. Loading and generation are synchronous. Calls go directly to
the native library; model files must already exist locally.

Build the library and install the Python package from this checkout:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKAPITSA_BUILD_SDK=ON
cmake --build build -j 6 --target kapitsa_sdk
python3 -m pip install ./sdk/python
export KAPITSA_LIBRARY="$PWD/build/libkapitsa_sdk.dylib"
```

For source-tree use, set `PYTHONPATH=sdk/python` instead of installing. The library
path must be explicit, through `KAPITSA_LIBRARY` or `Engine.load(..., library_path=...)`.
The package checks ABI version 1 and retains the loaded library for its handles.

```python
import os
from kapitsa import Engine

with Engine.load({"model_dir": os.environ["MODEL"], "disk_cache": "0"}) as engine:
    result = engine.generate(
        {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32},
        on_delta=lambda delta: print(delta["content"], end="", flush=True),
    )
    print(result["finish_reason"])
```

Options and requests are dictionaries. Their fields and the returned dictionaries
are the [C SDK contract](../README.md#request-and-result-contract), including
reasoning and tool calls. Tool `arguments` remains JSON text. Unknown fields are
rejected by the native API. Use the final result for complete text: a native delta
can replace an incomplete UTF-8 sequence, so concatenated deltas need not match it.

## Lifetime, threads and cancellation

Use context managers or call `close()` explicitly; garbage collection does not
release native handles. `Engine.generate()` creates and closes a request for you.
For cancellation from another thread, create a request yourself:

```python
from concurrent.futures import ThreadPoolExecutor
from kapitsa import KapitsaError, Status

with Engine.load({"model_dir": os.environ["MODEL"]}) as engine:
    with engine.request({"messages": [{"role": "user", "content": "Hello"}]}) as request:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(request.run)
            request.cancel()  # May precede generation or race with completion.
            try:
                result = future.result()
            except KapitsaError as error:
                if error.status != Status.CANCELLED:
                    raise
                result = error.result  # None before generation; otherwise possibly partial.
```

Native calls release the GIL. Distinct requests can run on separate threads,
subject to the engine's slot count. Each request runs once. `Request.close()`
cancels an active run, waits for its native call and output cleanup, then destroys
the handle. Cancellation is cooperative and does not guarantee a bounded wait.
`Engine.close()` releases the public engine handle; existing requests remain valid
and retain the model and GPU lock until they too are closed. One live native engine
is allowed per process, including engines retained by requests.

A callback receives one delta dictionary on the generating thread. Return `False`
to cancel, or call `request.cancel()`. Other return values continue. Callback
exceptions, including `KeyboardInterrupt`, cancel and are re-raised after C returns;
they never unwind through C. Callbacks must be synchronous, and must not call
`Request.run()` or `Request.close()`, or wait for a thread closing their request.
Use `asyncio.to_thread` for asynchronous hosts; cancelling that Python task alone
does not cancel native generation. Keep the request and call `cancel()`, then wait
for the worker before closing it. Do not reuse native handles across `fork()`.

`KapitsaError` exposes the numeric `status` and any partial `result`. The binding
frees both native result and error strings, including failures. The C SDK's
[error-recovery limits](../README.md#interface-and-ownership) still apply: this
binding does not isolate a host from native fatal errors.

## Tests

```sh
python3 sdk/python/test_sdk.py
```

This compiles the existing native ABI double and checks Unicode, ownership,
one-shot requests, reuse, cancellation, callback exceptions, cross-thread cleanup
and deferred errors with allocated partial results. It checks that no fixture
handles or strings remain after each test. It loads no model and uses no GPU.

After building the real library, its CPU option, lock and failed-load checks are:

```sh
python3 sdk/python/test_native.py --library "$KAPITSA_LIBRARY"
```

With GPU authorization and an existing model, add `--model "$MODEL"` to exercise
generation, streaming, cancellation, retained requests and reload on the real
engine. The native engine takes the shared GPU lock; it fails with BUSY if another
holder owns it. This is functional coverage, not a numerical or speed benchmark.
