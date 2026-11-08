# Five-bit CPU packing: preliminary evidence

These are privacy-filtered derived summaries, not raw execution logs. Numeric measurement values are preserved. The source files are the exact standalone drivers used, including the copied and renamed baseline pack function; the latter is not represented as an unchanged copy of the entire original source file.

Candidate source: `90e10efce8462ede4c756ca77ccdb1510f4e9faf`. Reference function: `66a5f3bc473df9af8b72ded9090c3a8acc16a43e`.

The change packs each 32-code five-bit tile into five output words, preserving masking and the legacy overlap path. It is used for all nonoverlapping five-bit inputs satisfying the existing word-aligned length contract. Other bit widths, affine fitting/scales and Metal code are unchanged.

## Completed component checks

Both M3 Ultra and M4 Pro passed 19,404 independent-byte, overlap and protected-page cases. M4 also passed the same cases with ASan/UBSan. A wrong cross-word shift and a bypassed overlap fallback were each rejected by the test.

Each setup measured eight alternating A/B legs for synthetic widths 2560, 3136, 4096 and 8192, then a retained pinned model slice of 1,024 rows of width 2560. The baseline function and candidate function receive identical codes. Every leg compares the complete packed output and unchanged codes, scales and biases. JSON records contain every leg and the exact row/pass counts.

On the real model slice, adjacent B/A warm pack latency changes were -64.6977%, -65.0001%, -63.7074%, -64.0019% on M4 and -83.5749%, -83.7054%, -83.8500%, -83.6031% on M3. Quantize-plus-pack changes were -4.1349%, -3.8569%, -3.2747%, -6.2368% and -9.2049%, -8.8525%, -9.0070%, -9.1279%, respectively. These are CPU component measurements, with other host CPU activity not fully isolated; they do not establish complete import/loading or inference gains.

## Reproduction examples

The following are derived relative-path/placeholder examples, not the original recorded commands. Run from the candidate checkout; `$EVIDENCE` names this evidence folder and `$ROWS` the pinned slice described in the JSON. Both standalone arms share candidate dequantization and affine fitting; only A's packing function is replaced with the copied baseline function.

```sh
c++ -O3 -std=c++20 -arch arm64 -Isrc -Ithird_party -Itools src/gguf/quant.cpp src/gguf/gguf.cpp "$EVIDENCE/focused.cpp" -o "$OUT/focused"
"$OUT/focused"
c++ -O3 -std=c++20 -arch arm64 -Isrc -Ithird_party src/gguf/quant.cpp src/gguf/gguf.cpp "$EVIDENCE/screen.cpp" "$EVIDENCE/reference-pack.cpp" -o "$OUT/screen"
"$OUT/screen"
c++ -O3 -std=c++20 -arch arm64 -Isrc -Ithird_party src/gguf/quant.cpp src/gguf/gguf.cpp "$EVIDENCE/real-rows.cpp" "$EVIDENCE/reference-pack.cpp" -o "$OUT/real-rows"
"$OUT/real-rows" "$ROWS"
```

## Own-source M4 model checks

`CPU-CHECKS.json` records the full build, 22 CPU entries, 18 successful supplementary native tests, six supplementary scripts and lint. The additional `load_error_test` failed its unchanged raw-U8 page-cache assertion in the candidate and in one actual separate reference execution; neither run is called a pass.

The fresh imports compared all nine complete files literally. Each source produced 584 rows (145,018,880 float32 values): all finite, complete output and scored input bytes identical. Explicit zero-drift step/verify/batch checks passed; prefill separately passed its finite and literal checks. Verify uses a teacher-forced KFlash tree chain, not free MTP generation. The initial import-retention controller stop and recovery are disclosed in `M4-IMPORT.json`; neither completed build nor import was repeated.

The four native A/B/A/B legs returned rc 0 and empty invalid arrays. B1 retains three repetition flags; other native legs have no flags. All six fixed decode request/content SHA pairs match across all four legs, and all 24 requests produced 512 tokens. Sampled seed 3 has the same repetition flag in every leg, so this is not an all-pass repetition benchmark. Prefill/cached flags are empty, but those texts were not byte-compared, and differing generated histories do not form matched-input cached performance. Both HTTP runs returned rc 0, 55 PASS and 0 FAIL. All per-leg values and flags are in the derived JSON summaries; no general inference speed or universal API conclusion follows.

The fixed-request driver is the unchanged [existing driver](../2026-11-08-q8-0-cpu/fixed_seed_driver.cpp), SHA256 `6b022a4791ac656f74ca84dc7b2090e7bfd34e5ad0e7a1c6bc1197e8adcd8310`. It uses the source tree's benchmark protocol with seeds 1/2/3. Repeat the default native and fixed-request legs in A/B/A/B order on the second setup, retaining every result.

## Outstanding validation and follow-up

This remains a Draft while the second setup's complete import, model-output and default-runtime qualification is unfinished. The pinned model and exact source commits are available for that continuation. It must preserve the import byte gate, the full-vocabulary logit gate, all original A/B legs, flags and failures. Any new evidence will be added under new names.

The following are derived command examples, not original execution logs. Configure each exact source separately. Every model command must run under the exclusive GPU lock; the held markers apply only inside that lock. Use separate import homes and retain fresh converted-file comparisons before any reuse.

```sh
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" --target kapitsa logit_dump ni_run -j 2
lockf -k "$GPU_LOCK" env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 KAPITSA_HOME="$RUN_HOME" "$BUILD/kapitsa" import "$GGUF" --name five-bit-pack-review --threads 4 --no-drafter
lockf -k "$GPU_LOCK" env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 "$REF_BUILD/logit_dump" "$REF_MODEL" "$REF_OUT" --paths step,verify,batch --windows 8 --rows 24 --chunk 2048 --batch 4 --batch-rows 2 --corpus "$REF_SRC/bench/corpus.txt"
lockf -k "$GPU_LOCK" env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 "$CAND_BUILD/logit_dump" "$CAND_MODEL" "$CAND_OUT" --paths step,verify,batch --windows 8 --rows 24 --chunk 2048 --batch 4 --batch-rows 2 --from "$REF_OUT"
lockf -k "$GPU_LOCK" env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 "$BUILD/kapitsa" bench --model "$MODEL" --quick --json --out "$LEG_RESULT"
python3 "$CAND_SRC/tools/logit_compare.py" "$REF_OUT" "$CAND_OUT" --min-top1 1 --max-mean-kl 0 --max-median-kl 0 --max-p99-kl 0 --near-tie 0 --max-far-flips 0 --max-nll-change 0
```
