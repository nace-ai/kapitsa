# Integration after rebase onto b1be7d4f

This is an explicitly redacted derivation of new execution records, not a raw log. Original records remain private. The four previously published evidence files are unchanged and retain their original source identities.

Tested checkout: `082e46d38a994c2287a87c28cc92c7a04e5a1f56` (production/test commit `2f5c7c90a7574ebc4258b70a53edeb3d6e006df2`), based on `b1be7d4f644b3e58d56499404b38b46cf2d2a5ef`. The rebase preserves main's complete CI target list and row-worker lifetime step, and adds the output-cleanup target and step. The output-cleanup test driver is unchanged.

Runs: M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1, Apple clang 17 (`clang-1700.6.4.2`); M4 Pro, 20 GPU cores, 48 GiB, macOS 15.7.1, Apple clang 17 (`clang-1700.0.13.5`). Both Release builds use `-O3`, C++20 and arm64 without `NDEBUG`. CPU-only synthetic fixtures; no trained model or GPU run.

| Check | M3 | M4 |
|---|---|---|
| Explicit Release configure/build and CLI version | Pass | Pass |
| Rebased integrated output-cleanup regression | 16/16 pass, zero leaked descriptors | Initial run aborted before any scenario; not a pass |
| Main's row-worker lifetime regression | Pass | Pass |
| Broad `vk` CPU entries | Not repeated | 22/22 pass |
| Supplementary native entries | Not repeated | 19/19 pass |
| Six supplementary Python checks, packing/patch checks, `prek`, diff check | Not repeated | Pass |

The M4 integrated attempt stopped after an importer precondition failure, then attempted to enumerate absent output. A private diagnostic-only driver confirmed the precondition failure; it is not counted as a qualifying run. The unchanged production/test source subsequently passed the full integrated regression on M3. No failure or failed execution was replaced.

The first M3 archive validation rejected an internal relative symlink before extraction or any build/test command. After validating that the link stayed inside the source tree, the same verified archive was used for the completed execution; no test was replayed from that failed preflight.

The rebased M4 `load_error_test` returned 0. Its earlier raw-read cache-threshold failure remains recorded in `CPU.json` and is still unexplained; this pass does not establish a fix. The new loader test and safetensors source match the new base, and the linked executable has no `import_model` or `OutFile` symbols. No fresh reference loader run or same-executable comparison is claimed.

[Structured results, scenario rows, identities and original-log digests](REBASE-b1be7d4f.json). Identity keys are relative artifact labels. This is rebase integration evidence, not another two-chip reference/candidate differential or sanitizer run. Original differential and sanitizer results remain tied to the commits named in the original README.

## Reproduction examples

These relative-path and placeholder commands are derived examples, not the original execution log. Use a checkout of the tested source and a suitable existing scratch directory for `TEST_TMP`.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target kapitsa gguf_output_cleanup_test
TMPDIR="$TEST_TMP" build/gguf_output_cleanup_test
python3 tests/test_gguf_parallel_rows.py
bash .dsh/skills/verify-kapitsa/scripts/vk cpu-tests
```

## Not verified

The APFS-cloned expert branch, trained-model import/output, GPU, inference performance, and M3 sanitizer. The synthetic MTP fixture checks import transactions, not MTP generation. No speed or universal test-success claim is made.
