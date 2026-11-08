# CPU split-GGUF table routing

This is a privacy-filtered derived summary of the actual CPU checks. Original command records and logs remain private. Command examples below are derived portable examples, not original execution logs.

`NgramTable::open_gguf` now maps the file owning the selected tensor. The descriptor retains the tensor’s absolute byte offset within its owning file. The temporary GGUF reader can close before the retained read-only mapping is used. Decoder, hash, prefetch, Metal and profile behavior are unchanged.

Reference source: `dc4b885b2915d3ffccb91ce9f43e707c7d0c8a26`. Candidate source: `315540a26d3d47f198425812acfc24d59e15dc3e`. [summary.json](summary.json) records exact source, driver and binary hashes and all results.

The same focused test source runs 16 cases: IQ4_NL/Q8_0, tensor in either of two parts, caller naming either part, and a same-size zero-filled or shorter valid unrelated part. Headers, alignments and absolute offsets differ. Each genuine payload is nonzero and independently read through the GGUF reader. All seven decoded rows and nine reordered/repeated gathered rows are compared in full against direct block dequantization.

On M4 the pre-fix code returns 1: four wrong-file byte cases and four false truncated-tensor cases fail. The changed reader returns 0 and all 16 cases pass; the existing single-file, bounds, affine and expert-store checks pass too. These focused binaries precede the source commit and are identified by their exact source hashes. Committed-source broader M4 and targeted M3 checks are recorded separately.

Committed validation: M4 Release configure/build/version, all 21 `vk` CPU entries, 15 supplementary native checks, six supplementary CPU scripts, full `prek`, whitespace check and `vk cleanup` pass. The optional reasoning dataset check/rebuild case is skipped; no dataset fetch/rebuild ran. M3 Release configure/build and the complete `fn_store_test`, including all 16 split cases, pass. Both use committed source `315540a2`, actual `-O3` without `NDEBUG`, and unchanged source/test hashes. No GPU or model execution ran.

Runs: Apple M4 Pro / 20 GPU cores / 48 GiB / macOS 15.7.1 / Apple clang 17.0.0 (1700.0.13.5); Apple M3 Ultra / 60 GPU cores / 96 GiB / macOS 26.4.1 / Apple clang 17.0.0 (1700.6.4.2).

The current `load_error_test` execution passes. Its byte-identical executable failed a cache-residency assertion in earlier recorded work; that historical result remains there. No new reference execution or causal repair of that separate assertion is claimed.

Portable examples:

```sh
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j 4 --target fn_store_test
"$BUILD/fn_store_test"
.dsh/skills/verify-kapitsa/scripts/vk cpu-tests
prek run --all-files
```

Not verified: GPU/model execution, actual trained GGUF PLE tables, whole-model output equivalence, performance, automatic split-filename discovery in model-folder loading, untested formats/worker schedules and whole-library sanitizers. This is a CPU file-selection repair; these checks establish the fixture row/gather bytes, without an inference or speed claim.

`vk checks` assigns this source file to its generic tier-2 shared-model rule. Its Qwen3.5 4B and Qwen3.6 35B-A3B model/model-tokenizer/API/logit/decode/interleaved-speed recommendations were not run. Neither model calls this reader; no completion of that generic tier-2 suite is claimed.
