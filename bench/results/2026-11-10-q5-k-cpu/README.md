# Q5_K CPU decoding evidence

These JSON files are redacted derived evidence, not raw logs. Original records remain private. Candidate `cea2b64784d12ad187f3a584e4847178c9d8c805`; reference `2e308aeac6d6fe95801e7957996034d6bbfa6ac3`. Both arms use the same standalone driver and block-type stubs with their own `src/gguf/quant.cpp`, C++20 `-O3`.

Runs: M5 Pro (20 GPU cores, 48 GiB), macOS 26.6.2, Apple clang 17 (`clang-1700.0.13.5`); M3 Ultra, macOS 26.4.1, Apple clang 17 (`clang-1700.6.4.2`). CPU A/B/A/B holds the exclusive GPU lock. No model or GPU is loaded by these drivers.

| Input / device | A1 reference | B1 candidate | A2 reference | B2 candidate |
|---|---:|---:|---:|---:|
| Synthetic / M5 | 296.001 | 234.576 | 324.178 | 238.172 |
| Trained Airy / M5 | 278.157 | 191.752 | 185.791 | 123.819 |
| Trained Airy / M3 | 363.525 | 347.371 | 229.655 | 145.548 |

Median ns/row within each leg; all 16 samples per leg are retained in the JSON. Leg-to-leg variation is substantial, including the small M3 first-pair difference. This establishes CPU decoder measurements, not a whole-import or inference speedup, nor universal absence of regressions.

Synthetic input: fixed seed, 1024 rows x 2048 values; each sample times 64 complete decodes. Trained input: first 1024 rows x 1024 values of `token_embd.weight`, Q5_K, from complete pinned `Acnoryx/Airy@ba4ce94d5e7bcdd4997f4dbf6275848c99c7fbb5`, `gguf/Airy-0.8b-IQ1_S.gguf`. Full-file SHA256 `8065a5ec831f74f8e38e12b219abb61aa564bf677b97c70cd049747bcda90711` was checked; the 720896-byte sample SHA256 is `53a3f4fdf264d62cf204005d241380188788b798ba8253f8ab52d89839935ccd`. M3 received those exact sample bytes, not an independent import.

A1 creates the gold output using the independently built reference. Each timed sample compares every output byte against it, checks finite values and unchanged input, and checks surrounding guards. All four final files also compare literally equal: 8 MiB synthetic, 4 MiB trained on each chip. Hashes alone are not the numerical gate.

M5 reference, candidate and UBSan candidate each pass 2,187,466 oracle/overlap cases plus guarded buffers; M3 reference and candidate pass the same checks. The finite oracle uses int64 arithmetic, covers all half patterns in both scale positions, 64 scales/minima, all 32 codes, four rounding modes, FP flags and signed zero. Exceptional scales and sequential overlap compare against the unchanged original scalar expression. A deliberate low-nibble-mask mutation is rejected.

## Reproduction

The following are path-placeholder examples, not raw command transcripts. Set `REF_SOURCE` and `CAND_SOURCE` to the two checkouts, `EVIDENCE` to this directory, `SAMPLE` to the verified trained bytes, and `OUT` to a new output directory. Build each arm with the same compiler and flags:

```sh
clang++ -std=c++20 -O3 -I"$REF_SOURCE/src" -I"$REF_SOURCE/third_party"   "$EVIDENCE/trained_driver.cpp" "$EVIDENCE/block_type_stubs.cpp"   "$REF_SOURCE/src/gguf/quant.cpp" -o "$OUT/reference"
clang++ -std=c++20 -O3 -I"$CAND_SOURCE/src" -I"$CAND_SOURCE/third_party"   "$EVIDENCE/trained_driver.cpp" "$EVIDENCE/block_type_stubs.cpp"   "$CAND_SOURCE/src/gguf/quant.cpp" -o "$OUT/candidate"
```

Under one exclusive GPU flock, run reference/candidate/reference/candidate in sequence, passing `SAMPLE`, one initially absent gold-output file and a unique output file per leg. The trained driver takes those three positional arguments. The synthetic driver takes only the gold-output and output paths. Keep every stdout sample and compare the four complete outputs literally. Never reuse a frozen result label for a new run.

## Remaining model validation

These records do not yet establish complete imported-file equivalence, GPU logits, free continuations, native inference performance or HTTP behavior. The applicable follow-up uses this full Airy model, independent imports from the two actual source builds, complete logits/input comparison, native and fixed A/B/A/B, and HTTP results with failures preserved. No Metal kernel changed. Other chips, ASan and non-AArch64 execution have not run.
