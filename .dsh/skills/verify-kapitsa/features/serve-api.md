# Serve and its APIs

`kapitsa serve` loads one model and answers the OpenAI Chat Completions and Responses APIs and Anthropic's Messages
API on one port, with a chat page at `/`. Clients such as Claude Code, Codex and the OpenAI and Anthropic SDKs talk to
it unchanged. It prints `Ready:` once loaded. A port already in use is an error.

## Sub-features

- SV1: `GET /health` returns `{"status":"ok"}`; `GET /v1/models` lists the model under its API name (`--model-id` renames it)
- SV2: `POST /v1/chat/completions`, streamed and not, with tools
- SV3: `POST /v1/responses`
- SV4: `POST /v1/messages` and `/v1/messages/count_tokens`
- SV5: `GET /`: the chat page (strict CSP, no outside requests)
- SV6: `--api-key KEY`: requests without `Authorization: Bearer KEY` or `x-api-key: KEY` get 401; `/health` stays open
- SV7: `POST /v1/models/switch` is 403 without `--allow-switch`
- SV8: follow-ups reuse the conversation's cache (a fast second turn)

## How to get to it (user POV)

`kapitsa serve --model 4b`, then point a client at `http://127.0.0.1:8000/v1`, or open `http://127.0.0.1:8000` in a
browser. `kapitsa claude` / `codex` / `opencode` / `pi` start the client wired to it.

## Driving it with vk

Preconditions: the task allows loading a model; `$V doctor` shows the GPU lock free; a model folder such as
`$NACE_MODELS/qwen3.5-4b-4bit-mtp` exists. Start it with `$V serve "$NACE_MODELS/qwen3.5-4b-4bit-mtp"`, which prints
`ready: http://127.0.0.1:8090 (pid N)`.

| ID | Command | Observable result |
| --- | --- | --- |
| SV1 | `curl -s 127.0.0.1:8090/health; curl -s 127.0.0.1:8090/v1/models` | `{"status":"ok"}`; a list whose `data[0].id` is the model's name |
| SV2 | `curl -s 127.0.0.1:8090/v1/chat/completions -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":16}'` | 200; `choices[0].message.content` non-empty, `usage` present |
| SV3 | `curl -s 127.0.0.1:8090/v1/responses -H 'content-type: application/json' -d '{"input":"Say hi","max_output_tokens":16}'` | 200; an `output` array with a `message` item |
| SV4 | `curl -s 127.0.0.1:8090/v1/messages -H 'content-type: application/json' -d '{"model":"x","max_tokens":16,"messages":[{"role":"user","content":"Say hi"}]}'` | 200; `content[0].type == "text"` |
| SV5 | `curl -si 127.0.0.1:8090/ \| head -20` | 200, `Content-Security-Policy: default-src 'none'; ...`, HTML |
| SV7 | `curl -s -o /dev/null -w '%{http_code}' -X POST 127.0.0.1:8090/v1/models/switch` | `403` |
| all | `python3 tests/test_server.py http://127.0.0.1:8090` | the repo's end-to-end API checks pass (chat page, three APIs, tool loops, caching, errors) |
| SV6 | `$V stop`, then `$V serve DIR --api-key k1`, then curl SV2 with and without `-H 'authorization: Bearer k1'` | 401 without it, 200 with it; `/health` 200 either way |

Finish with `$V stop`.

## Gotchas

- Pass a model folder. A catalog name downloads into the scratch home (gigabytes), and the scratch home is deleted at
  cleanup.
- The model loads for tens of seconds (the 35B for minutes). `vk serve` waits up to 600 s.
- Without `--model-id`, the API name is the model's name; the `"model"` field of requests is ignored unless
  `--allow-switch`.
- `vk serve` passes `--host 127.0.0.1 --private`, so the test server never shows up in `kapitsa cluster` on other Macs.
- Requests decoded together may differ at near-ties from the same request alone (docs/architecture.md). Compare
  single-stream replies only.
