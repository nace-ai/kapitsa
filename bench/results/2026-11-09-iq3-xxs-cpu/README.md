# IQ3_XXS CPU row decoding

These files are explicitly redacted derived evidence, not raw execution logs.
Original commands and complete artifacts remain private. The copied driver is
exactly the one used for the measurements; it accepts input and output paths as
arguments. This folder qualifies CPU row decoding only.

Candidate production source: `32e7aa73868f7f12a92a649fffa324d7d198b5f0`.
Reference: `7c5112243ba73252dd161d49b6b53207a6851940`.
Each arm uses its own explicitly configured CMake Release library, C++20 -O3,
arm64, with actual target flags that do not contain NDEBUG.

MODEL.json identifies the fully downloaded and SHA256-verified GGUF, its 109
IQ3_XXS tensors, and the first 1,024 rows of blk.0.ffn_up.weight (K=1,024).
The mixed-format file is not described as wholly IQ3_XXS.

One untimed warmup precedes 16 passes in each leg. Every pass decodes all rows
with one call per row, compares all 4,194,304 output bytes against the reference,
checks finite values, input preservation and output canaries. The complete leg
files are also compared literally after the eight alternating A/B legs.
Every per-pass timing is retained, including slower or noisier samples.
These timings do not measure full import or inference acceleration.

## Reproduction example

The commands below use placeholders and are examples, not purported raw logs.
Use the same driver bytes for both own-source builds. `$BUILD` selects the
current arm's CMake Release directory and `$DRIVER` is the copied driver here.
`$SAMPLE` is the 401,408 bytes at offset 193,726,304 of the pinned GGUF, with
SHA256 196c6a58a2837683980887483d180b871574721121a008d4df560c84b6cad0c7.

```sh
clang++ -O3 -std=gnu++20 -arch arm64 -mmacosx-version-min=14.0 \
  -I src -I third_party "$DRIVER" "$BUILD/libnaceinfer.a" -o trained-decoder
./trained-decoder "$SAMPLE" "$REFERENCE_GOLD" "$NEW_LEG_OUTPUT"
```

Run reference A1 first with a new, absent reference-gold file; every later arm
must read that existing file unchanged. Give every leg a new output filename.
The gold is 4,194,304 bytes. Run A1/B1/A2/B2/A3/B3/A4/B4.

## Coverage and remaining checks

The integrated test uses an integer representation oracle for finite scales,
the unchanged scalar expression for exceptional values and FP flags, all half
patterns under four rounding modes, every grid index and scale nibble with
boundary half scales, parity signs, overlapping blocks and inaccessible guard
pages. Deliberate parity and scale corruptions are rejected. Focused UBSan
instruments the harness, quant.cpp and gguf.cpp, not the whole engine.

Full independent model imports, GPU logit/token gates, native/fixed-request
inference A/B, HTTP model tests, ASan and non-AArch64 execution remain unverified
for this source change. This is CPU evidence for an open Draft, not merge or
whole-engine numerical/performance qualification. No Metal source changed.

## Completed M5 trained-row measurements

Runs: M5 Pro (20 GPU cores, 48 GiB), macOS 26.6.2, Apple clang 17 (clang-1700.0.13.5). CPU execution only.

| Setup | Pair | Reference ns/row | Candidate ns/row | Latency change |
|---|---|---:|---:|---:|
| M5 | 1 | 435.322754 | 181.539062 | -58.29782% |
| M5 | 2 | 322.326660 | 112.528809 | -65.08858% |
| M5 | 3 | 251.282227 | 101.888184 | -59.45269% |
| M5 | 4 | 256.571289 | 109.395996 | -57.36234% |

Each row is the median of its 16 passes; M5.json retains every pass. CHECKS.json records 21 passing CPU-suite entries and the one updater failure and the source-digest identity of the pre-commit build.
