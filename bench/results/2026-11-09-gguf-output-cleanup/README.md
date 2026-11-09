# GGUF output descriptor cleanup

These files are explicitly redacted structured derivations of the recorded executions, not raw logs. Original commands and outputs are retained privately. No earlier published result is replaced.

Production and test source: `1af5144c9b264c15c74bb2a695a000f03dcd8a33`.
Reference importer source: `e21f08b45fde070cb554bb73897e67b547b5c279`.

| Setup | Reference scenarios passing | Candidate scenarios passing | Reference leaked handles | Candidate leaked handles |
|---|---:|---:|---:|---:|
| M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1, Apple clang 17 (clang-1700.6.4.2) | 5/16 | 16/16 | 15 | 0 |
| M4 Pro, 20 GPU cores, 48 GiB, macOS 15.7.1, Apple clang 17 (clang-1700.0.13.5) | 5/16 | 16/16 | 15 | 0 |

## Method

Each setup explicitly configured its own Release build, built the CLI and integrated regression, and ran the integrated regression. The formal comparison compiled the same final driver with the entire reference or candidate `import.cpp`, linking the same own-candidate runtime archive per setup. This is a differential of the importer translation unit, not two whole-tree reference/candidate builds. Compiler flags were C++20, `-O3`, arm64, macOS minimum 14.0; source, executable and archive identities are in the setup JSON files.

The driver calls the complete `import_model` transaction on synthetic dense and dense-plus-MTP GGUF fixtures. It injects failures in truncation, header/data writes, cancellation, second-output creation, and checked close. Wrappers are armed only during that call and match only the test's temporary safetensors outputs. No production fault hook, trained model, GPU or resource exhaustion is involved.

The reference leaks one or two output descriptors in 11 of the 16 scenarios, 15 handles in total. The driver counts these handles before explicitly cleaning up its own leaked descriptors. The candidate passes all 16 scenarios without leaks. Both successful controls and failed force-imports compare the complete set of fixture file contents against the earlier successful import **within the same arm**; this is not a cross-build trained-model output comparison. The diagnostic and temporary-directory checks pass. The close-error control reuses the closed descriptor and verifies it is not closed again during unwinding.

M4 ASan/UBSan passes all 16 scenarios. Instrumentation covers the importer and driver; the common runtime archive is not instrumented. M3 sanitizer was not run. Earlier staging failed before extraction/build/tests on M3; the same source archive was recovered and the final recorded runs completed. Preliminary drivers are not mixed into the final comparison.

## Broad CPU checks

The own-source M4 build passed all 22 `vk` CPU entries, 18/19 supplementary native entries, six supplementary Python checks, packing and patch checks, `prek`, and diff checking. `CPU.json` retains each exit code and original-log digest.

The supplementary `load_error_test` returned 1 at its existing raw-read cache threshold after the preceding read/bounds controls passed. The cause of this execution's threshold failure is not diagnosed. That path does not call the GGUF importer: `import_model` and `OutFile` are absent from the linked executable, and its test/weight-store source is unchanged from the reference. Its executable differs from earlier retained controls; no new reference loader execution or same-executable comparison is claimed. This is not an all-tests-pass result.

## Reproduction examples

The following relative-path and placeholder commands are derived examples, not the original execution log. Run in a checkout of the production/test source. Use distinct output names; retain all failures.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target kapitsa gguf_output_cleanup_test
build/gguf_output_cleanup_test

git show e21f08b45fde070cb554bb73897e67b547b5c279:src/gguf/import.cpp > "$REF_IMPORT"
# Run once with IMPORT_SOURCE="$REF_IMPORT", then with IMPORT_SOURCE=src/gguf/import.cpp.
# OUT is a distinct executable path for each arm.
clang++ -std=gnu++20 -O3 -arch arm64 -mmacosx-version-min=14.0 \
  -Iinclude -Isrc -Ithird_party tools/gguf_output_cleanup_test.cpp \
  "$IMPORT_SOURCE" build/libnaceinfer.a \
  -framework Metal -framework Foundation -framework CoreFoundation \
  -framework IOSurface -framework IOKit -framework ImageIO -framework CoreGraphics \
  -framework Security -framework SystemConfiguration -framework Accelerate -o "$OUT"
"$OUT"
```

The sanitizer example adds `-fsanitize=address,undefined -fno-sanitize-recover=all` to the candidate compile command; that does not instrument the already-built archive.

## Not verified

The APFS-cloned expert branch, trained-model import/output, GPU, and inference performance. Synthetic MTP import is not MTP generation. No speed or universal correctness claim is made.
