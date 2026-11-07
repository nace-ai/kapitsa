# Mapped n-gram tensor admission

Reference: `934a6d72`. Candidate implementation: `877e7b4d`.
Runs: M4 Pro, 48 GiB, macOS 15.7.1, Apple Clang 17, C++20 `-O3`.

A 78-byte file has a valid header declaring a `[128,160]` BF16 tensor with
40,960 payload bytes, but contains no payload. `probe.cpp` only opens the
table; it never accesses a row in this malformed mapping. Reference admission
succeeds, while the candidate throws `FatalError` naming the tensor and file.

The new regression in `tools/fn_store_test.cpp` covers 26 malformed fixtures
and complete BF16, FP8 and affine fixtures with output guards. The existing
suite also checks nonzero affine rows across unequal shards, GGUF IQ4_NL/Q8_0,
and gather behavior. The initial baseline red check fails at the first missing
payload assertion, without accessing a malformed row (`baseline-red.log`).

The `initial/` record is a CPU-only selected-source build. Both probes compile
their actual `ngram.cpp`; the full test compiles the candidate's actual test,
`ngram.cpp` and test-hook `expert_store.cpp`. They link against the same Release
support library built at `279ce236` (a separate range-scan branch based on
`d65a0afa`). The record includes its SHA256 and every command. This is not a
complete standalone build of `877e7b4d`. Full CMake build/CPU regression is
queued separately. No model or GPU was loaded and no inference performance
is claimed.

Reproduce from this checkout on a Mac, using a new output directory:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j6 --target naceinfer fn_store_test
build/fn_store_test
python3 bench/results/2026-11-06-ngram-bounds/reproduce.py \
  --support-build build --support-revision "$(git rev-parse HEAD)" \
  --out /tmp/ngram-bounds-new-run
```

Full builds, the rest of the CPU suite and `prek`, and Flash-Next valid-model
load/output/HTTP checks remain prerequisites before marking the Draft ready.
GPU/model checks must take the shared GPU lock. The remote Mac is occupied by
another task. The GGUF admission path and truncation occurring *after* successful
mapping are outside this change.
