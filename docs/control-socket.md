# Control socket, protocol 1 (outline)

OUTLINE: names and shapes only; prose, examples and the test follow. Open step-1 names are marked OPEN.

## Transport
- HTTP/1.1 over a Unix domain socket. JSON bodies; `GET /events` is Server-Sent Events.
- Paths are unversioned (`/status`, not `/v1/status`); the version is `protocol` in `GET /status`.

## Auth, discovery, start race
- Socket: `$KAPITSA_HOME/run/<instance_id>.sock`; `run/` is 0700, the socket 0600.
- Peer check: `getpeereid`; a peer whose uid is not the server's gets `403 forbidden_peer`. No token, no TCP port.
- Discovery: every `run/*.sock`. Connected = live; `ECONNREFUSED` = stale (any client may unlink it).
- Start race: `flock(run/<instance_id>.lock)` held for the server's life; the loser exits with "instance already running".
- Restart detection: `run_id` (new per process start) in status and in every event id.

## Endpoints
| Method, path | Request | 2xx response |
| --- | --- | --- |
| `GET /status` | none | `status` |
| `GET /events` | header `Last-Event-ID` (optional) | SSE stream of `event` |
| `POST /stop` | `stop_request` | `202 stop_response` |
| `POST /model` | `model_switch_request` | `200 model_switch_response` |

The inference port serves only the OpenAI, Anthropic and Ollama APIs; `POST /v1/models/switch` moves here.

## Messages and fields
- `status`: protocol, instance_id, run_id, name, pid, version, started_at, lifecycle, parent_pid, state, api_url, model, config, tick_ms, requests[]
  - `lifecycle`: `attached` (child of the app, exits with it) | `keep_running` (launchd LaunchAgent) | `foreground` (a terminal's `kapitsa serve`)
  - `state`: `starting` | `loading` | `ready` | `switching` | `stopping`
  - `model`: id, context_length, slots
  - `config`: the effective instance configuration, exactly step 1's `kapitsa serve --config -` JSON (`$ref` docs/serve-config.schema.json)
  - `requests[]`: request_id, client, phase (`waiting|prefill|writing`), prompt_tokens, cached_tokens, completion_tokens, elapsed_ms, tokens_per_second
- `event` (SSE `id: <run_id>.<seq>`, `event: <type>`, `data: <json>`), types:
  - `state`: state, model
  - `request_started`: request_id, client
  - `request_progress`: one per active request per tick: the `requests[]` item fields
  - `request_finished`: request_id, finish_reason, error, prompt_tokens, cached_tokens, completion_tokens, queue_ms, prefill_ms, decode_ms, tokens_per_second, prefill_tokens_per_second
  - `log`: level (`debug|info|warn|error`), message (level-filtered before the bus)
  - `reset`: reason (`expired|run_changed|invalid_cursor`), status
- `stop_request`: run_id (optional precondition)
- `stop_response`: run_id, state (`stopping`)
- `model_switch_request`: model (required), run_id (optional precondition), and these config fields, same keys and ranges as step 1: context, slots, spec, dflash, prefill_chunk, mem_budget, disk_cache
- `model_switch_response`: run_id, model, previous_model
- `error`: `{"error": {"code", "message", "details"?}}`

## Instance configuration fields (from step 1, worktree svc-config at 7c511224 + service.h)
Serve flags with `_` for `-`: model, model_id, engine, host, port, profile, private, local, peer, context, slots, spec,
dflash, prefill_chunk, api_key, allow_switch, idle_unload, mem_budget, image_max_pixels, timing, disk_cache, loops,
loop_kv, split, verbose, yes, splash_binary, splash_model, splash_arg, splash_persistent_cache,
splash_conversation_salt, splash_kflash. POC (#873) had protocol, instance_id, name, mem_budget_gb.

OPEN (asked through the lane): `mem_budget` vs `mem_budget_gb`; `idle_unload` vs `idle_unload_min`; `peer` vs `peers`;
whether `protocol`, `instance_id` and `name` are config fields; the schema's path and `$id`.

## Versioning
- `protocol` integer, 1, equal to the config's `protocol`. Clients send nothing; they read it from `GET /status`.
- Additive (no bump): new optional response fields, new event types, new endpoints, new error codes (clients fall back on the HTTP status).
- Bump: removing, renaming or retyping a field; a new required request field; changing an endpoint, a status code, the SSE framing or the cursor.

## Errors
| code | HTTP |
| --- | --- |
| bad_request | 400 |
| unknown_field | 400 |
| invalid_value | 400 |
| forbidden_peer | 403 |
| not_found | 404 |
| model_not_found | 404 |
| method_not_allowed | 405 |
| run_id_mismatch | 409 |
| busy | 503 (+ `Retry-After`) |
| stopping | 503 |
| internal | 500 |
