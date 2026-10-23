# System One: decisions instead of tokens

Drex v2.0 and Drex Max are decision models: they read a state and a set of questions and answer with a
probability per option, not with generated text. `build/systemone` serves that protocol on Kapitsa, so a Drex
release GGUF can be run natively on the Mac instead of through llama.cpp's server: the state is encoded once,
each question is scored from it, and the answer comes back as calibrated probabilities.

```bash
# the pointers head is the GGUF's five F32 pointer tensors in a plain binary (no conversion, no quantization)
python3 tools/drex_pointer.py "drex-v2.0-Q8_0.gguf" ~/.cache/kapitsa/models/drex-v2.0-q8/pointer.bin

# serve POST /v1/systemone
build/systemone --model ~/.cache/kapitsa/models/drex-v2.0-q8 --port 8000 --context 16384
curl -s localhost:8000/v1/systemone -d '{"state": "Record a: the authorization flag is enabled.",
  "questions": {"q": {"type": "noul", "instructions": "Is record a authorized?"}}}'
```

## The protocol

The request carries a state (any JSON value; objects and arrays are rendered as indented text) and a map of
questions. A question is one of three types:

| type | criteria | answer |
|---|---|---|
| `noul` | optional `{"true": ..., "false": ...}` text for the two options | `noul`: the probability of yes |
| `choice` | an object of named options, in order, up to 255 | `choice`: the most likely name, `confidence`, and `probabilities` per name |
| `score` | an ordered list of levels, lowest first | `score`: the expected level, `legend`, and `probabilities` per level |

```json
{"state": {"note": "the shipment awaits inspection"},
 "questions": {"route": {"type": "choice", "instructions": "Which desk?",
                         "criteria": {"customs": {"description": "hold and inspect"},
                                      "transit": null}},
               "urgent": {"type": "noul", "instructions": "Is it urgent?"},
               "quality": {"type": "score", "instructions": "How clear is the note?",
                           "criteria": ["unclear", "usable", "clear"]}}}
```

The reply is `{model, answers, usage: {input_tokens, input_tokens_details: {cached_tokens}, output_tokens}, latency_ms, timings}`; probabilities are rounded to
four decimals, `confidence` is the separation of the best option from a flat distribution, and `output_tokens`
counts the tokens of the answer JSON, as in the reference server. A `{"requests": [...]}` list scores up to 1024
states in one call with one model in memory.

## How it runs on Kapitsa

- **Encoding.** The state is one text row: the `fim_prefix` delimiter then the rendered state. Every question is
  its own branch row: `fim_middle`, the instructions, `box_start` + option text + `box_end` per option, and
  `fim_suffix` last. A branch's positions continue from the state's, so it attends to the state and to itself only.
  Text that looks like a delimiter is rewritten to `¦name¦` before tokenizing, so caller data cannot forge an
  option boundary (the reference does the same).
- **The state once.** The state is prefilled from an empty recurrent state, then the recurrent state (GDN conv and
  rec buffers) at that position is checkpointed. Attention rows are positional, so a branch never reads another
  branch's rows, and one request's state leaves nothing behind for the next: a batch element or a request to a
  long-running server is prefilled exactly as it would be alone.
- **A branch per question.** Each question restores the checkpoint, prefills only its own branch, and reads the
  layer-40 residual rows the head needs: the `fim_suffix` row (the pointer query) and the `box_end` rows (the
  keys). The final RMSNorm is applied on the host in f32, then the pointer head's two projections and the dot
  product per option, scaled by `1/sqrt(256)` and by the head's temperature. Softmax gives the probabilities.
- **The head is exact.** The pointer head is 2.1M parameters in f32, loaded from the GGUF's own tensors, so it
  contributes no quantization error; only the 40 transformer layers are quantized.

Quantization matters little for quality here: a decision is a comparison between options read from the same
hidden states, so the two sides of the comparison move together. The Q8_0 release keeps the answer margins
large; the 4-bit file is the one to use when speed matters (`docs/benchmarks.md` has the numbers for generation,
and the same quantized layers run here).

## Cached states

The protocol's usual shape is one state and several rounds of questions on it (a document, then a follow-up), and
every request re-sends the whole state. The server keeps the states it has prefilled in a bounded cache, keyed by
the model's files and the state's tokens: a request whose state it holds restores that state's recurrent checkpoint
and attention rows instead of prefilling it, so what is left is the branch pass. `timings.cached_tokens` reports how
many positions came from the cache (0: the request prefilled its state), under `timings` and as
`usage.input_tokens_details.cached_tokens`, the name an OpenAI-shaped client reads.

- `--cache-mb` (default 1024) is the cache's byte cap; `0`, or `NI_SYS_CACHE=0`, turns it off and serves every
  request cold. Least-recently-used states are evicted for a new one, and a state larger than the whole cap is served
  but not kept.
- A hit's answer is the cold answer: the restored rows and the checkpoint are the same bytes the cold pass wrote, so
  the branch pass runs on identical inputs. `tests/test_systemone.py URL` checks that a re-sent state hits and that
  the hit's answers and probabilities match a cold run, and that a different state never hits.

## Limits and options

| flag | default | what it caps |
|---|---|---|
| `--context` | 16384 | the sequence's positions: a request's state plus its longest question row must fit |
| `--cache-mb` | 1024 | memory for prefilled states (`0` or `NI_SYS_CACHE=0`: every request cold) |
| `SYSTEMONE_MAX_STATE` | 16384 | the state's tokens |
| `SYSTEMONE_MAX_BRANCH` | 16384 | one question's branch tokens |
| `SYSTEMONE_MAX_PACKED` | 16384 | a whole request, all questions packed |

`SYSTEMONE_CONTEXT` sets all three caps at once; a specific variable overrides its own cap. A request that does
not fit is rejected with 400 and a message that says which limit it hit, before any GPU work starts. The context
can be raised to 131072 if the Mac has the memory for the KV cache.

`--once request.json --repeat 5 --warmup 1` scores one request from a file and prints the measured latencies, which
is how the numbers below were taken. `--encode request.json` needs no GPU at all: with only the tokenizer loaded it
prints the encoded layout (state tokens, each branch's tokens, which rows are read, the option keys and legend),
which is the fastest way to see what a request turns into.

## Tests

`tests/test_systemone.py --cli build MODEL_DIR` checks the encoder offline (the layout, the option order, the
shielding, the limits, the 400s) and `tests/test_systemone.py URL` checks a running server: the three question
types, a batch, determinism, that a question's answer does not move when other questions share its state, and the
state cache: a re-sent state reports `cached_tokens` equal to the state's positions, its answers equal a cold run's
to 4 decimals, and another state reports 0.
