# Q4_K CPU decoding, M5 Pro

These are redacted derived records. Raw commands, logs and output binaries are retained privately. The drivers are the exact measured source; example commands below are placeholder-based reproductions, not raw logs.

Production `54b034f0a4607074a11f60472c42472070efb494`; reference `d91f34efd96a93aeb68e8bc993c091b2d90d4c85`. Both standalone arms use the same driver and block-type stubs, replacing only `src/gguf/quant.cpp`, compiled with Apple clang 17 (`clang-1700.0.13.5`), C++20, `-O3`. Integrated Release checks are recorded separately. Hardware: Apple M5 Pro, 20 GPU cores, 48 GiB memory capacity, macOS 26.6.2. Each CPU timing sequence holds the exclusive GPU lock.

The finite, nonoverlapping arm64 path expands nibbles directly into fp32 vectors. Half times 6-bit scale times 4-bit code fits exactly in fp32; the final addition retains scalar rounding. Nonfinite scales and overlapping blocks use the original staged scalar expression. `decode4` and Metal are unchanged.

| Input / dimensions | A1 reference | B1 candidate | A2 reference | B2 candidate |
|---|---:|---:|---:|---:|
| Fixed-seed synthetic, 1024 x 2048 | 411.442 | 160.529 | 414.986 | 159.079 |
| Airy trained Q4_K, 1024 x 1024 | 277.344 | 132.426 | 261.678 | 85.713 |
| Flint trained Q4_K, 1024 x 1024 | 248.983 | 85.632 | 214.538 | 79.712 |

Values are median ns/row within each leg; every individual sample is in the JSON. These are CPU decoder timings, not import or inference throughput. Synthetic legs time 16 samples of 64 complete decodes; trained legs time 16 complete decodes after one warmup. Every measured sample compares the complete output with the independently executed reference outside timing and checks input bytes, finite values and guards. All four final files also compare literally equal: 8 MiB for synthetic and 4 MiB for each trained input.

`airy-trained.json` records the complete pinned Airy-0.8b-Q2_K file identity and the Q4_K rows actually present in that mixed file. Its ordinary dense import preserves these Q4_K codes and bypasses `dq_q4_k`; it is trained-row CPU evidence only.

`flint-trained.json` records an exact pinned range of `unsloth/flint-GGUF`'s `flint-UD-IQ2_XXS.gguf`, revision `20468edfc38ab43d6914cc88b08cbecb05d0fc62`: the first 1024 rows of `blk.32.ffn_down_shexp.weight`. The full-file hash there is the published expected digest, not a claim that a full download was checked. These shared-expert Q4_K tensors convert to 4-bit group 64 in `build_flint`, which reaches the modified decoder. Full-model validation is separate and pending at this submission.

## Correctness and limitations

The main reference, candidate and UBSan candidate each passed 2,187,466 oracle/overlap cases plus guarded single/multiple/empty/partial blocks. Finite expected values come from an independent integer oracle; exceptional scales and overlapping writes use the exact pre-change expression. Coverage includes every half bit pattern in both scale positions, all 64 scales and minima, all 16 codes, four rounding modes, FP exception flags, signed zeros, unaligned input, and sequential overlap. A low-nibble mask fault (15 to 7) was rejected. An initial refactoring of the exceptional-value test reference failed on unchanged main due to different NaN propagation; the corrected reference preserves the original expression. That failed attempt is retained.

The two trained inputs belong to different model families, on one chip. Other chips, full-model output gates and whole-import/inference speed are not yet qualified. No new or modified Metal kernel is present.

## Reproduce a CPU leg

Use fresh output names; a reference leg creates the golden file, and the candidate compares against it. `trained_driver.cpp` expects the 589,824-byte, 1024-row Q4_K slice described by the corresponding JSON; a full GGUF cannot be passed directly.

```sh
clang++ -std=c++20 -O3 -I "$SRC/src" -I "$SRC/third_party" \
  "$RESULTS/trained_driver.cpp" "$RESULTS/block_type_stubs.cpp" \
  "$SRC/src/gguf/quant.cpp" -o "$BIN"
lockf -k /tmp/kapitsa-gpu.lock "$BIN" "$ROWS" "$GOLD" "$OUTPUT"
```

For the synthetic driver, pass only `$GOLD` and `$OUTPUT`. Repeat the two separately built arms in A B A B order under one exclusive lock for comparable timing. The block-type stub only supplies Q4_K metadata; decoding uses the full unmodified arm-specific `quant.cpp`.
