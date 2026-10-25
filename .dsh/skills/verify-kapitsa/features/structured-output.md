# Structured output

A client can ask for a reply that is valid JSON against a schema, or for a forced tool call. Kapitsa constrains
decoding with a grammar, so the reply parses every time. The three APIs each have their own field:
Chat Completions `response_format` (`json_schema` or `json_object`), Responses `text.format`, Messages
`output_format` or `output_config.format`. A tool call is forced with `tool_choice` `required` or a named tool on the
OpenAI APIs, and with `{"type":"any"}` or a named tool on Messages. Strict tools hold their arguments to the schema.

## Sub-features

- SO1: the grammar and token-mask engine (offline: `build/grammar_test`)
- SO2: how each API reads the fields above (offline: `build/structured_check`)
- SO3: token masks against brute force on a real model's tokenizer (offline, needs a model folder; no GPU)
- SO4: live replies valid against their schema, greedy and sampled, on all three APIs (GPU)
- SO5: forced and strict tool calls (GPU)

## How to get to it (user POV)

Send `"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {...}, "strict": true}}` to
`/v1/chat/completions`, or use an SDK's structured-output helper pointed at `kapitsa serve`.

## Driving it with vk

Preconditions: `$V build` done. SO3 needs a model folder (tokenizer only). SO4 and SO5 need the task's GPU permission
and `$V serve DIR` running.

| ID | Command | Observable result |
| --- | --- | --- |
| SO1, SO2 | `python3 tests/test_structured.py --cli build` | exit 0; grammar_test and structured_check report no failures |
| SO3 | `python3 tests/test_structured.py --cli build "$NACE_MODELS/qwen3.5-4b-4bit-mtp"` | exit 0; masks agree with brute force on that tokenizer |
| SO4, SO5 | `python3 tests/test_structured.py http://127.0.0.1:8090 --quick` | exit 0; every reply validates (needs `jsonschema`) |
| SO4 by hand | `curl -s 127.0.0.1:8090/v1/chat/completions -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"A city and its country"}],"max_tokens":64,"response_format":{"type":"json_schema","json_schema":{"name":"c","strict":true,"schema":{"type":"object","properties":{"city":{"type":"string"},"country":{"type":"string"}},"required":["city","country"],"additionalProperties":false}}}}'` | `choices[0].message.content` parses as JSON with exactly `city` and `country` |

## Gotchas

- SO1–SO3 are tier 0.5 evidence (references/pr-checks.md). They load no weights and do not touch the GPU, so they are
  allowed without GPU permission.
- With reasoning on, the schema applies to the answer after the thinking, not to the thinking.
- A schema the grammar cannot express gets an error reply, not an unconstrained one. That is a correct result, not a
  failure.
