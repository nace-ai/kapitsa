# Unsupported selected safetensors dtypes

Initial CPU-only evidence for source `61fdbafb`, baseline `96d2de68`.
This is a loader rejection fix, with no inference-speed claim.

A BOOL tensor declares eight bytes, but the loader maps BOOL to Unknown and gives it a zero-byte
shape/dtype size. The probe uses an eight-byte backing array, so it demonstrates the writes without
an actual allocation overrun: baseline replaces all eight guard bytes; candidate rejects before reading.
Its GPU stubs throw if reached; this check initializes no Metal device.

```sh
tmp=$(mktemp -d)
git show 96d2de68:src/loader/safetensors.cpp > "$tmp/baseline.cpp"
clang++ -O3 -std=c++20 -mmacosx-version-min=14.0 -Isrc -Ithird_party \
  bench/results/2026-11-06-safetensors-known-dtypes/probe.cpp "$tmp/baseline.cpp" -o "$tmp/baseline"
clang++ -O3 -std=c++20 -mmacosx-version-min=14.0 -Isrc -Ithird_party \
  bench/results/2026-11-06-safetensors-known-dtypes/probe.cpp src/loader/safetensors.cpp -o "$tmp/candidate"
"$tmp/baseline"
"$tmp/candidate"
```

The new cases in `tools/load_error_test.cpp` cover six unsupported/unknown strings with empty and
nonempty ranges. Eager, deferred and indexed admission reject them, including file/tensor/dtype in
the error. Tensor filters still skip them; retained U8 indexed/deferred payload reads keep guard bytes;
known selected tensors still work in header-only plans. The initial check compiled those exact new
cases and the existing metadata cases separately against the production loader, with throwing GPU stubs.

`initial-checks.json` records only what had run before Draft publication. Full Release/CPU/prek and
real 4B logit, HTTP and interleaved benchmark verification are queued on the remote M3 Ultra, after
other PRs finish their timings. Later results will be added as separate files. No GPU or whole-model
compatibility claim follows from this initial record. Other model families remain unverified.
