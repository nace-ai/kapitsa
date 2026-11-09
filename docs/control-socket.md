# Control socket, protocol 1

Each running `kapitsa serve` instance has a control socket. Clients use it to see the instance's state and to stop
it or switch its model. Clients are the Kapitan app, `kapitsa ps`, `stop` and `logs`, and any other UI. The
inference port serves only the OpenAI, Anthropic and Ollama APIs. Everything an administrator does goes through this
socket.

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
- **Instance id.** 1 to 32 of `A-Z a-z 0-9 _ -`. macOS limits a socket path to 103 bytes. A server whose socket path
  is longer refuses to start, and its error names the path.
- **Discovery.** A client lists `run/*.sock` and connects to each one. A socket that accepts the connection is a live
  instance. A refused connection (`ECONNREFUSED`) means the instance is stale: its process is gone.
- **Stale sockets.** Clients never delete a socket they did not create. A client may remove a stale socket only while
  it holds that instance's lock, which proves no server owns it. The lock is an exclusive `flock` that a client takes
  without waiting.
- **Start race.** A starting server takes the lock with `flock(LOCK_EX | LOCK_NB)` and holds it until it exits. The
  loser exits with "instance `<id>` is already running". The winner removes any old socket file, then binds.
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

`GET /status` returns the instance's identity, its state (`starting`, `loading`, `ready`, `switching`, `stopping`),
the loaded model, the active requests, and `config`. `config` is the instance's effective configuration in exactly
the keys, types and ranges of `kapitsa serve --config -`, minus `api_key`. `last_event_id` is the newest event's id.
A client that sends it as `Last-Event-ID` reads every event after this status with no gap and no duplicate.

### Stop

`POST /stop` answers `202` and then stops the instance. Requests that are still running end with `finish_reason`
`cancelled`, and the `state` event `stopping` follows. The process exits with status 0 and removes its socket.

### Model switch

`POST /model` loads another model in the same instance and answers once the new model is ready. The request takes
`model` (required) and, optionally, `model_id`, `context`, `slots`, `spec`, `dflash`, `prefill_chunk`, `mem_budget`
and `disk_cache`. These are the serve configuration's own fields, with the same names, types and ranges. A field
left out keeps the instance's current value. During the switch the `state` events read `switching` and then
`ready`. The socket needs no `--allow-switch`, because its user already controls the process. `allow_switch` still
governs whether an inference request that names another model may load it. `POST /v1/models/switch` leaves the
inference port.

## Lifecycles

`lifecycle` in status says who owns the process. The control messages are the same in all three cases. The only
difference is what happens around them.

| `lifecycle` | Owner | What it means for a client |
| --- | --- | --- |
| `attached` | The app that started it (`parent_pid` names the app) | The instance exits with its parent. `POST /stop` ends it sooner. |
| `keep_running` | launchd, through a LaunchAgent | It outlives the app. After `POST /stop` it stays stopped until the app or the next login starts it again. A crash restarts it with a new `run_id`. |
| `foreground` | A terminal's `kapitsa serve` | `POST /stop` ends it as Ctrl-C would. |

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
| `log` | A log line at or above the server's level | `level`, `message` (at most 4096 bytes) |
| `reset` | The cursor cannot be continued | `reason`, `status` |

**Cursor.** The server keeps at least the last 1024 events. With no `Last-Event-ID` header, the stream starts at the
next event. With a header, the server replays every event after it, then continues live. When it cannot replay, it
sends one `reset` event first, whose `status` is a fresh `GET /status`, and then continues live. The reset's `id` is
`status.last_event_id`, so the client's cursor stays valid. `reason` is one of:

- `expired`: the cursor is older than the oldest event the server still holds.
- `run_changed`: the cursor's `run_id` is another run's, because the instance restarted.
- `invalid_cursor`: the header cannot be parsed, or names an event that never existed.

**Rates.** The server samples each active request once per tick (`tick_ms` in status). `tokens_per_second` is the
completion tokens written in that tick divided by the tick's length. Status and `request_progress` both report the
last tick's sample, so every observer sees the same rate however often it reads.

**Logs.** The server drops log lines below its level before they reach the bus, so a quiet server builds no events
for them. The level is `info`, or `debug` when the server runs with `verbose`.

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
| `busy` | 503 | A model switch is already running; `Retry-After` is set | `retry_after_s` |
| `stopping` | 503 | The instance is stopping | none |
| `internal` | 500 | Anything else, for example a failed model load | none |
