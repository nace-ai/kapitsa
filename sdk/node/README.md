# Node.js / TypeScript SDK

ES modules for Node.js 22+, Apple silicon and macOS 14+. This optional package uses Koffi
to call the existing C ABI; it adds no dependency to the CLI or C library build.
Loading, generation and teardown run on native worker threads. No HTTP server, model download
or child process is started. JavaScript and TypeScript use the same interface and declarations.

## Build and use

From the Kapitsa checkout:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKAPITSA_BUILD_SDK=ON
cmake --build build -j 4 --target kapitsa_sdk
npm ci --prefix sdk/node
export KAPITSA_LIBRARY="$PWD/build/libkapitsa_sdk.dylib"
```

Import from the checkout, or install its `sdk/node` directory into your application with
`npm install /path/to/kapitsa/sdk/node` and import from `@kapitsa/sdk`. This package is not
published to npm and does not bundle the dylib; ship the library alongside your application.

```js
import { Engine } from './sdk/node/index.js';

const engine = await Engine.load({ model_dir: '/local/model', slots: 2 });
try {
  const result = await engine.generate({
    messages: [{ role: 'user', content: 'Explain attention briefly.' }], max_tokens: 128,
  }, { onDelta: delta => { process.stdout.write(delta.content); } });
  console.log(result.completion_tokens);
} finally { await engine.close(); }
```

`Engine.load(options, { libraryPath })` overrides `KAPITSA_LIBRARY`. Options and requests use
the [C ABI's JSON fields](../README.md#request-and-result-contract), including snake_case names.
Reuse the engine for successive calls; `Promise.all` can submit distinct requests to its native
slots. No JavaScript request queue is added; native busy/errors are returned to the caller.

## Cancellation and lifetime

Pass `{ signal: controller.signal }` and call `controller.abort()`, or return `false` from
`onDelta`. Callbacks run synchronously on the JS event-loop thread; keep them short and do not
return promises. A thrown callback value cancels generation and is rethrown after native cleanup.
`close()` rejects new work, cancels active requests, waits for them, then releases the engine.
Always await it in `finally`; garbage collection does not close an engine. It is safe to start
closing inside a callback, but await the close promise outside that callback.

Native failures reject with `KapitsaError`: `status` is a member of `Status`, and `result` holds
any partial response (or null). Cancellation uses `Status.CANCELLED`. After the deferred Metal
errors covered by #793, close the failed engine before loading another. The one-engine-per-process
rule, lifetime-held GPU lock and [remaining runtime fatal errors](../README.md#interface-and-ownership)
also apply here. For crash isolation, put the SDK in a long-lived child process owned by your app.
This package does not define an IPC protocol. Electron packaging and Worker termination are not
covered by the Node tests; do not terminate a Worker while it has native calls in flight.

## Tests

`KAPITSA_LIBRARY` set as above: `npm test --prefix sdk/node` checks a CPU-only native fixture
and real C option validation; `npm run typecheck --prefix sdk/node` checks TypeScript usage.
With GPU authorization and an idle GPU, run the opt-in real-model test under the shared lock:

```sh
lockf -k -t 0 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 KAPITSA_MODEL=/local/model npm run test:gpu --prefix sdk/node
```
