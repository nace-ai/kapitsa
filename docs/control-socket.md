# Control socket, protocol 1

Each running `kapitsa serve` instance has a control socket. Clients use it to see the instance's state and to stop
it or switch its model. Clients are the Kapitan app, `kapitsa ps`, `stop` and `logs`, and any other UI. The
inference port serves the OpenAI, Anthropic and Ollama APIs and a page in front of this socket ("The engine view"
below). Everything an administrator does goes through this socket, or through that page with the socket's own
handlers and this run's key.

This page specifies the wire format. The machine-readable form is [schema/control/](../schema/control/), JSON Schema
2020-12, with one file per message. [schema/control/examples/](../schema/control/examples/) holds a valid example of
every message and error code, plus examples that must be refused. `python3 tests/test_control_schema.py` checks
both. The instance configuration's fields belong to `kapitsa serve --config -`
([serve-config.schema.json](serve-config.schema.json)). This protocol refers to them and never copies them.

## Transport

The transport is HTTP/1.1 over a Unix domain socket, with JSON bodies (`Content-Type: application/json`, at most
64 KiB). `GET /events` answers with Server-Sent Events. HTTP gives each request a status code and needs no framing of
its own, and SSE is plain HTTP, so every client can use the HTTP library it already has (URLSession,
`curl --unix-socket`, Python's `http.client`) with no custom protocol.

Paths carry no version (`/status`, not `/v1/status`). A server speaks exactly one protocol, and `GET /status` reports
it.

## Socket, access and discovery

| What | Where |
| --- | --- |
| Socket | `$KAPITSA_HOME/run/<instance_id>.sock`; `KAPITSA_HOME` defaults to `~/.cache/kapitsa` |
| Start lock | `$KAPITSA_HOME/run/<instance_id>.lock` |
| Permissions | `run/` is `0700`, the socket `0600` |

- **Access.** Only the user who started the instance can reach the socket, because the directory is private. The
  server also checks each connection's peer with `getpeereid` and answers `403 forbidden_peer` to any other user
  (root included). There is no token and no TCP port.
- **Instance id.** 1 to 32 of `A-Z a-z 0-9 _ -`, `default` when the configuration gives none. macOS limits a socket
  path to 103 bytes, and 32 keeps a usual home folder's path under it. A server whose socket path is longer anyway,
  or whose socket cannot listen (a volume that takes no sockets), names the path in a warning and serves without a
  socket. (Instances the configuration names, a later change, will refuse to start instead.)
- **Identity fields.** [control.schema.json](../schema/control/control.schema.json) defines `protocol`,
  `instance_id` and `name`. The control socket's implementation adds them to `kapitsa serve --config -` with these
  meanings, and the control schema then refers to them there.
- **Discovery.** A client lists `run/*.sock` and connects to each one. A socket that accepts the connection is a live
  instance. A refused connection (`ECONNREFUSED`) alone proves nothing, because macOS also refuses while a server is
  between `bind` and `listen` or its backlog is full. The client then tries the instance's lock, an exclusive `flock`
  on the `.lock` file, without waiting. If it gets the lock, no server owns the socket and the instance is stale. If
  not, the instance is starting or busy, and the client retries.
- **Stale sockets.** Clients never delete a socket they did not create, except a stale one, while they hold its lock.
  They release the lock at once.
- **Start race.** A starting server takes the lock, waiting at most 1 second so that a client's brief check never makes
  it fail, and holds it until it exits. A server that does not get it exits with "instance `<id>` is already
  running". Until the configuration names instances, every server is `default`: a second one (as `kapitsa bench`
  starts beside a running server) serves without a socket and says so, as does a server that cannot make `run/`. The winner removes any old socket file, emits its first event (`state` `starting`), then binds and
  listens. The lock's file descriptor and the listening socket are opened `O_CLOEXEC`. Otherwise a child such as
  the Splash engine would inherit the lock and keep it after the server is killed, and clients would see the instance
  as starting or busy forever.
- **Restarts.** `run_id` is new each time the process starts. It is in `GET /status` and in every event id, so a
  client sees that an instance restarted even though the socket path is unchanged.

## Endpoints

| Method and path | Body | Success |
| --- | --- | --- |
| `GET /status` | none | `200` [status](../schema/control/status.schema.json) |
| `GET /events` | none; optional `Last-Event-ID` header | `200`, an SSE stream of [events](../schema/control/event.schema.json) |
| `POST /stop` | [stop request](../schema/control/stop-request.schema.json) | `202` [stop response](../schema/control/stop-response.schema.json) |
| `POST /model` | [model switch request](../schema/control/model-switch-request.schema.json) | `200` [model switch response](../schema/control/model-switch-response.schema.json) |

Any other path answers `404 not_found`, and a wrong method answers `405 method_not_allowed`. `POST /stop` and
`POST /model` take an optional `run_id`. When it is not the running run's id, the request fails with
`409 run_id_mismatch`, so a click on an old view can never stop or switch a restarted instance.

### Status

`GET /status` returns the instance's identity, its state, the model, the active requests, and `config`. `name` is the configuration's `name`, or the `instance_id`
when that is empty, so a client always has something to show. `config` is the instance's effective configuration in exactly
the keys, types and ranges of `kapitsa serve --config -`, minus `api_key`. `last_event_id` is the newest event's id.
A client that sends it as `Last-Event-ID` reads every event after this status with no gap and no duplicate.

| `state` | What it means | `model` |
| --- | --- | --- |
| `starting` | The process is up; nothing is loading yet | `null` |
| `loading` | The model loads: at start, or again after a release | `null` at start, else the model that loads |
| `ready` | The model is in memory and serves requests | the model |
| `switching` | `POST /model` or an inference request is loading another model | the model that served before |
| `released` | No model is in memory; the next inference request loads `model` again | the model that comes back |
| `stopping` | The instance is shutting down | the last model, or `null` |

`released` has two causes. With `idle_unload`, the server gives the model's memory back after a quiet spell. The next
inference request then waits while the state reads `loading` and then `ready`. If that load fails, the state returns to
`released` with a `message`, the request fails, and the next request tries again. The other cause is a failed switch
whose previous model could not be loaded back either (below).

`engine` shows what the engine holds, sampled once per tick, so a view can draw memory, slots and drafts. It is
present only while `state` is `ready`, and never in Splash mode, whose engine runs in another process. A switch frees
the old engine before the new one loads, so there is nothing to sample while `switching`.

| Field | What it holds |
| --- | --- |
| `memory.budget_bytes` | The GPU memory the engine may hold (`mem_budget`) |
| `memory.allocated_bytes` | The GPU memory it holds now |
| `memory.weights_bytes` | Of that, what the engine held before its slots: the model's weights, the drafter's and the buffers they run in |
| `memory.kv_bytes` | Of that, the slots' KV caches as allocated (each slot's capacity in positions times the bytes per position). The rest is the slots' recurrent state and buffers allocated later |
| `slots[]` | One item per slot: `index`, `busy`, `tokens` in its live state, and `request_id` while busy |
| `drafts` | `cycles` that verified a draft and the `tokens` they wrote, counted since the model loaded. Absent when the model decodes without drafts. A counter that goes down means the model loaded again: start the baseline over |

Each cycle writes one token of its own plus the draft tokens it accepts. A view gets tokens per cycle as
`tokens / cycles`, and the share of tokens that came from accepted drafts as `(tokens - cycles) / tokens`, both from
the change between two ticks.

### Stop

`POST /stop` answers `202` and then stops the instance. Requests that are still running end with `finish_reason`
`cancelled`, and the `state` event `stopping` follows. The process exits with status 0 and removes its socket.
During a model switch, a stop takes effect as soon as the load can be abandoned, and the pending `POST /model` answers
`503 stopping`.

**Not yet in `kapitsa serve`.** Running requests are cut when the process exits; they do not end with `finish_reason`
`cancelled`. A stop during a switch drops the pending `POST /model`'s connection instead of answering `503 stopping`.
A stop while the instance is `starting` takes effect at once. One while it loads its first model takes effect before
the next piece of weights its loader reads (at most 32 MB, one tensor, or one streamed expert's read, the fill of
a model whose experts stream from disk included); the steps that read nothing (compiling
kernels, the Neural Engine program) finish first.
A model still downloading stops at once, its downloaded parts kept for the next start. Every lifecycle exits with status 0, a terminal's after clearing its status block. A second
`POST /stop` answers `202` and changes nothing, and `POST /model` after a stop answers `503 stopping`.

### Model switch

`POST /model` loads another model in the same instance and answers once the new model is ready. The request takes
`model` (required) and, optionally, `model_id`, `context`, `slots`, `spec`, `dflash`, `prefill_chunk`, `mem_budget`
and `disk_cache`. These are the serve configuration's own fields, with the same names, types and ranges. A field
left out keeps the instance's current value. The switch behaves as `POST /v1/models/switch` does today:

- Requests already running finish first. New inference requests get `503` with `Retry-After` until the switch ends.
- The `state` events read `switching`, then `ready` with the new `model`.
- A model that fails to load leaves the previous one serving. The request answers `500 load_failed` with
  `details.state` `ready`, and the `state` event `ready` carries the previous `model` and a `message` that says why.
- To load the new model, the server first frees the old one. If the new one fails and the old one then fails to
  load back too, no model is in memory. The request answers `500 load_failed` with `details.state` `released`, and
  the `state` event `released` names the previous `model` with a `message` giving both failures. The instance keeps
  running, and the next inference request tries to load the previous model again.
- A client that disconnects does not stop the switch. It reads the outcome from the `state` events or `GET /status`.

The socket needs no `--allow-switch`, because its user already controls the process. `allow_switch` still
governs whether an inference request that names another model may load it. `POST /v1/models/switch` leaves the
inference port.

**Not yet in `kapitsa serve`.** A switch takes only `model`: any other serve field answers `400 invalid_value` with
`details.field` naming it. A Splash instance (`engine` `splash`) refuses every switch with `400 invalid_value`,
because its engine serves its `splash_model` folder only. Before the first model has loaded, `POST /model` answers
`503 busy` with `Retry-After`.

## Lifecycles

`lifecycle` in status says who owns the process. The control messages are the same in all three cases. The only
difference is what happens around them.

| `lifecycle` | Owner | What it means for a client |
| --- | --- | --- |
| `attached` | The app that started it (`parent_pid` names the app) | The instance exits with its parent. `POST /stop` ends it sooner. |
| `keep_running` | launchd, through a LaunchAgent | It outlives the app. After `POST /stop` it stays stopped until the app or the next login starts it again. A crash restarts it with a new `run_id`. |
| `foreground` | A terminal's `kapitsa serve` | `POST /stop` clears its status block and ends it with status 0. |

## Events

`GET /events` streams every event of the instance's event bus (`server::Events`). Each event is one SSE frame:

```text
id: 7f3a9c2e5b1d4e60.912
event: request_progress
data: {"request_id":"req-41","api":"openai","phase":"writing","prompt_tokens":2310,...}
```

The schema describes a frame as the JSON object `{"id", "type", "data"}`. The `id` is `<run_id>.<seq>`, and `seq`
counts from 1 for each run. The server sends a `: keepalive` comment every 15 seconds.

| `type` | When | `data` |
| --- | --- | --- |
| `state` | The instance's state changes | `state`, `model`, an optional `message` |
| `request_started` | A request arrives on the inference port | `request_id`, `api`, `client` |
| `request_progress` | Every tick, once per active request | the same fields as a `requests` item in status |
| `request_finished` | A request ends | token counts, times, rates, `finish_reason`, and `error` when it failed |
| `engine` | Every tick while status has `engine` | the same object as status `engine` |
| `log` | A log line at or above the server's level | `level`, `message` (at most 4096 bytes) |
| `reset` | The cursor cannot be continued | `reason`, `status` |

**Cursor.** The server keeps at least the last 1024 events. With no `Last-Event-ID` header, the stream starts at the
next event. With a header, the server replays every event after it, then continues live. When it cannot replay, it
sends one `reset` event first, whose `status` is a fresh `GET /status`, and then continues live. The reset's `id` is
`status.last_event_id`, so the client's cursor stays valid. That id always exists, because a server's first event
comes before it accepts connections. `reason` is one of:

- `expired`: the cursor is older than the oldest event the server still holds, or the events after it add up to more
  than 512 KiB, half of what a stream may fall behind (below), so replaying them would close the stream again.
- `run_changed`: the cursor's `run_id` is another run's, because the instance restarted.
- `invalid_cursor`: the header cannot be parsed, or names an event that never existed.

**Rates.** The server samples each active request once per tick (`tick_ms` in status). `tokens_per_second` is the
completion tokens written in that tick divided by the tick's length. Status and `request_progress` both report the
last tick's sample, so every observer sees the same rate however often it reads.

**Slow readers.** The bus never waits for a reader. The server holds at most 1 MiB of unsent events per stream and
closes a stream that falls further behind. Its client reconnects with `Last-Event-ID` and gets the replay or a
`reset`. At most 16 streams are open at once; a further `GET /events` answers `503 busy`.

**Logs.** The server drops log lines below its level before they reach the bus, so a quiet server builds no events
for them. The level is `info`, or `debug` when the server runs with `verbose`.

## The engine view

A browser cannot open the socket, so `kapitsa serve` also serves a page of what the instance holds and does at
`http://127.0.0.1:<port>/engine` (`src/server/engine.html`). The page reaches the socket's four endpoints over HTTP on
the API port, with the same bodies and answers: `GET /engine/status`, `GET /engine/events`, `POST /engine/stop` and
`POST /engine/model`. Its Stop button asks once more before it sends; its Switch box takes a model name, folder or GGUF
file, as `POST /model` does.

- **This run's key.** The socket admits its own user only, and the HTTP side keeps that. At start the server makes a
  key, 24 random bytes, and writes it to `$KAPITSA_HOME/run/<instance_id>.token` (`0600`, in the `0700` `run/`),
  which it removes on exit. Every path but the page needs `Authorization: Bearer <key>`; without it they answer
  `403 forbidden_peer`. The key replaces the API key on these paths: a page holding the key does not need the API
  key, and the API key opens none of them. So whoever can stop or switch over HTTP can read a file only the socket's
  user can read, as on the socket. A new run makes a new key.
- **The link.** On a terminal, `kapitsa serve` prints the page's link with the key in its fragment
  (`/engine#t=<key>`); without one it prints the key file's path instead. The key never goes into the log. A browser
  never sends a fragment to a server. The page keeps the key for its tab and takes it out of the address bar. Another
  client, such as Kapitan, can read the key from the file. The page itself holds no data and needs no key.
- **This Mac only.** Any of these gets `403 forbidden_peer`, the page included: a peer that is not `127.0.0.1` or `::1`;
  a `Host` other than `127.0.0.1`, `localhost` or `[::1]` (a page elsewhere that points its own name at `127.0.0.1`);
  an `Origin` other than `http://` plus that `Host`; a `Forwarded`, `X-Forwarded-For`, `X-Forwarded-Host` or `Via`
  header (a proxy on this Mac would otherwise bring other machines in); a body over 64 KiB.
- **Methods.** The page, status and events take `GET` and `HEAD`, stop and model take `POST`; another method gets
  `405 method_not_allowed`, and any other path under `/engine/` gets `404 not_found`. A stop or a switch also needs
  `Content-Type: application/json`, else `400 bad_request`: a form on another site can send only form or text bodies,
  and no key. No CORS header is sent, so a fetch from another site with the key gets no further than its preflight.
  Every answer carries `Cache-Control: no-store`. The page's `Content-Security-Policy` lets it reach only its own
  server and forbids framing it.
- **The log.** A page opened after the server started resumes the stream from the run's second event, so the bus
  replays the log lines it still keeps (the last 1024 events, `kKeep`) and the page shows them, marked `earlier`:
  events carry no time. Once more than 1024 events have passed since the run's second, the bus answers with the
  `expired` reset instead, and the page shows only what the server logs from then on. A button above the log hides every
  line below a warning.
- **The socket's answers.** A stop or a switch runs the socket's own checks (fields, `run_id`) and handlers, and
  answers as `POST /stop` and `POST /model` do. The page sends the `run_id` it shows, so a page left open on a run that
  has since restarted gets `409 run_id_mismatch` instead of stopping the new one. A switch answers once the new model
  is ready, as on the socket.
- **At most 3 views.** Each open `GET /engine/events` holds one of the API server's threads, at least 8 in all and
  shared with requests, so at most 3 are open at once. A fourth gets `503 busy` with `Retry-After: 5`. These count
  toward the bus's 16 streams.
- **No socket.** A server without a control socket answers the four endpoints with `503 internal`. That happens when
  another server holds its instance name or `run/` cannot be made. A server that routes to other Macs has its own
  socket and engine view, which show that server, not the Mac it forwards to.

## Versioning

`protocol` is an integer, 1 here. Clients send nothing to choose it. They read `protocol` from `GET /status` and
refuse to go on when it is higher than the highest they know. Clients ignore response fields and event types they do
not know.

- **No new protocol:** new optional response fields, new event types, new endpoints, new error codes (a client that
  does not know a code acts on the HTTP status).
- **A new protocol:** removing, renaming or retyping a field, a new required request field, a new value in an
  existing enum of a response (other than an error `code` or an event `type`), or any change to a path, a status code, the SSE framing or the cursor format.

## Errors

Every non-2xx response has the same body ([error.schema.json](../schema/control/error.schema.json)):

```json
{"error": {"code": "unknown_field", "message": "unknown field mem_budget_gb", "details": {"field": "mem_budget_gb"}}}
```

`code` is stable, and clients branch on it. `message` is for people and may change. `details` is optional and holds
only the fields listed below.

| Code | HTTP | When | `details` |
| --- | --- | --- | --- |
| `bad_request` | 400 | The body is not a JSON object, or is over 64 KiB | none |
| `unknown_field` | 400 | The body has a field the request does not define | `field` |
| `invalid_value` | 400 | A field has the wrong type or is out of range | `field` |
| `forbidden_peer` | 403 | The peer's user is not the server's | none |
| `not_found` | 404 | No such path | none |
| `model_not_found` | 404 | `POST /model` names no catalog model, folder or file | `model` |
| `method_not_allowed` | 405 | The path exists but not with this method | none |
| `run_id_mismatch` | 409 | The request's `run_id` is not the running run's | `run_id` (the running one) |
| `busy` | 503 | A model switch is already running, or 16 event streams are open; `Retry-After` is set | `retry_after_s` |
| `stopping` | 503 | The instance is stopping | none |
| `load_failed` | 500 | `POST /model` could not load the model | `model`, `state` (`ready` or `released`) |
| `internal` | 500 | Anything else | none |
