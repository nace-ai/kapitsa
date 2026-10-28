# Run Drex Max v1 on Kapitsa

Drex Max v1 uses the [System One HTTP API](systemone.md#the-protocol), including `choice`, `noul`, `score`, and the
`requests` batch wrapper. It produces probabilities without generating tokens. These steps import a release GGUF
and run the native Metal implementation, without a llama.cpp runtime.

This implementation is experimental. CPU checks pass, but native probability parity against the pinned v1 reference
has not been established. The first small-context GPU attempt stopped at the Metal validation assertion below.

## Import the model and readout head

Build the CLI and server:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target kapitsa systemone
```

Import the backbone, then extract its small readout head into the imported model directory. To keep both source
weights and imported weights on an external drive, set `KAPITSA_HOME` before importing:

```bash
export KAPITSA_HOME="/Volumes/Extreme Pro/kapitsa"
GGUF="/Volumes/Extreme Pro/parked/drex-max-v1-Q8_0-release/drex-max-v1-Q8_0.gguf"
build/kapitsa import "$GGUF" --name drex-max-v1
python3 tools/drex_pointer.py "$GGUF" "$KAPITSA_HOME/models/drex-max-v1/pointer.bin"
```

The importer preserves `qwen35.attention.causal=false` as `text_config.causal=false`. The server requires this
noncausal backbone with a readout head, and rejects a causal backbone paired with that head. It also rejects a
projected v2 pointer head paired with a noncausal backbone. Do not switch modes by changing only the head file.

The Q8_0 importer converts the backbone to Kapitsa's affine 8-bit group-64 layout. It does not retain the original
Q8_0 blocks. The released GGUF's readout and temperature tensors are F32 even though its backbone is Q8_0.

## Serve and inspect a request

```bash
build/systemone --model "$KAPITSA_HOME/models/drex-max-v1" --port 8000 --context 2048
curl -s localhost:8000/v1/systemone -d '{"state":"The shipment awaits inspection.","questions":{"route":{"type":"choice","instructions":"Which desk?","criteria":{"customs":"hold and inspect","transit":null}},"urgent":{"type":"noul","instructions":"Is the shipment urgent?"},"quality":{"type":"score","instructions":"How clear is the note?","criteria":["unclear","usable","clear"]}}}'
```

Each question must fit the context as one complete prompt. The server allocates whole-context scratch and KV
buffers at startup. Do not use a 128K or 262K context on a 48 GB Mac: even with MTP disabled, those allocations
exceed 52 GiB and 104 GiB before weights. A 32K context needs about 13.3 GiB before weights and host memory;
its fit still requires a monitored model run. Increasing `--context` is not a bounded-memory prefill mode.

Max does not reuse a shared-state checkpoint or prefix-cache entry; `cached_tokens` stays zero. The reported
input-token count sums complete question prompts, including repeated state.

For tokenizer-only inspection, use `--encode request.json` with the imported model directory. This mode reads the
configuration and tokenizer, without loading backbone weights or using the GPU. Max layouts have an empty shared
state, one complete prompt per branch, and a decide row at each branch's final token. Option indices are readout
output indices, not hidden-state row positions.

## Encoding and readout

Max wraps each question in ChatML with a classification system message, the state, instructions, named option
codes, and an assistant prefix containing an empty think block. String states stay unchanged; other JSON states
use JSON text with spaces after commas and colons. Caller text resembling special tokens is shielded before
wrapping. Nonempty image arrays are rejected.

The server resets recurrent state before each question and evaluates its entire prompt at position zero. Full
attention is bidirectional within that prompt. Gated DeltaNet and convolution remain sequential. Questions never
attend to one another, and full-attention suffix tokens can affect the state's hidden representation, so ordinary
chunked prefill and v2 state caching are not equivalent execution paths.

The final normalized hidden row is multiplied by the first N rows of `pointer.readout.weight`, then divided by
`pointer.temperature.weight`. There is no bias or square-root scaling. Softmax and answer formatting use the
existing System One protocol.

`tools/drex_pointer.py` detects this head from its tensor names. It writes little-endian `pointer.bin` with magic
`KPS1`, version 2, hidden width, output width, F32 readout weights in output-major order, and F32 temperature.
Version 1 projected q/k heads remain supported for Drex v2.0.

## CPU checks

```bash
python3 tests/test_drex_max.py --cli build
```

The synthetic tests check head extraction and tokenizer-only Max layouts without backbone weights or the GPU.
Runtime Metal validation and a probability comparison against a pinned reference are separate checks; a C++ build
does not compile the embedded Metal kernels.

## GPU validation status

The 2026-10-28 small-context check against public Drex Max v1 `f0f69bfa` completed the reference requests, but the
native server aborted before returning probabilities. With `MTL_SHADER_VALIDATION=1` and `MTL_DEBUG_LAYER=1`, Metal
reported `qmms_gu_b8_t8` using 65,536 bytes of threadgroup memory against a 32,768-byte limit. This existing fused
GEMM kernel declares 32,768 bytes of static storage; the validation layer counts that storage twice on this Mac.

The maximum absolute probability-difference gate of 0.02 remains untested. The conditional 32K probe was not run;
128K and 262K are excluded by the allocation estimates above. No performance claim follows from this attempt.
