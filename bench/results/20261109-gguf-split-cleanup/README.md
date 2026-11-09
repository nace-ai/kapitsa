# GGUF split-count rejection cleanup

This is a sanitized, derived evidence summary. Original execution commands and logs
are retained privately. The commands below are portable reproduction examples,
not transcripts of the recorded runs. `focus.cpp` is the exact recorded driver;
its SHA256 and the source, header, and binary SHA256 values are in
`focused-results.json`.

## Change and source identity

Production fix: `fe0409a96aef4472ff5795cd7f643f7b2ea07db2`.
Final regression header: `ab796a55cc217373f9be8d0fc72e8b43e1050891`.
Reference GGUF implementation: `47dbe3cbd1516af07be108385236d6f40e492b61`.
The test-only follow-up uses the standard temporary directory configuration.
It does not change the production fix.

The constructor previously checked `split.count` and `split.tensors.count` after
its cleanup handler. If either check threw, the incomplete `File` had no destructor
call and retained the descriptors of all opened parts. Both checks now run inside
the existing `try` block. Their predicates and diagnostics are preserved.
This does not claim to handle every possible allocation failure in the constructor.

## Focused CPU results

Each reference/candidate pair uses the same driver and final regression header,
with only the GGUF implementation replaced for the reference arm. Both use C++20
and `-O3`. The additional M4 sanitizer run uses `-O1 -g` with AddressSanitizer and
UndefinedBehaviorSanitizer. Every listed compilation succeeded.

| Setup and arm | Part-count rejection FD delta | Tensor-count rejection FD delta | Exit |
| --- | ---: | ---: | ---: |
| M3 Ultra, reference | +32 | +32 | 1 (expected regression failure) |
| M3 Ultra, candidate | 0 | 0 | 0 |
| M4 Pro, reference | +32 | +32 | 1 (expected regression failure) |
| M4 Pro, candidate | 0 | 0 | 0 |
| M4 Pro, candidate ASan/UBSan | 0 | 0 | 0 |

Each column has 16 rejected opens of a two-part synthetic GGUF, alternating the
entry filename across eight repetitions. The test checks descriptor counts after
every rejection, using `fcntl(F_GETFD)` on descriptors 0 through 1023 in these
isolated test processes. These are leaked-descriptor counts, not latency results.

The same regression checks malformed and missing second parts, duplicate tensor
names, corrected files reopened from either part, omitted optional split metadata,
the existing unsplit-filename behavior, and eight complete synthetic tensor reads
with byte comparisons and destination guards. A successful focused exit includes
these controls. No trained model is used.

The regression also runs in the default `load_error_test` suite, which is already
registered with CTest and run by CI. The focused switch permits checking this
failure path separately from the suite's other loader and cache checks.

Runs:

- M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1,
  Apple clang 17.0.0 (`clang-1700.6.4.2`).
- M4 Pro, 20 GPU cores, 48 GiB, macOS 15.7.1,
  Apple clang 17.0.0 (`clang-1700.0.13.5`).

## Broader CPU checks

The Release build at `fe0409a9` completed. Of 22 `vk cpu-tests` entries, 21
passed. All 19 supplementary native suites, six supplementary Python checks,
the Qwen 4 packing check, the packaged patch-application check, `prek`, and
whitespace checks passed. The full `load_error_test` passed in this broad run.
The grammar check additionally used an existing tokenizer: 308 checks passed,
including brute-force mask comparisons at 5,471 states over 248,320 tokens.
Its tokenizer digest is recorded in `cpu-results.json`; this is CPU metadata
validation, not model inference.

`test_update.py` exited 1: while replacing a temporary installation, one
`--version` observation was empty rather than either expected version. The
failure is retained. Its cause was not established; that harness does not retain
the subprocess status or stderr of the empty observation. The changed GGUF
constructor is not invoked by the updater or `--version`; those implementations
and the updater test are unchanged. No updater or general stability claim is made.

A separately configured and built reference at `47dbe3cb` actually ran the
updater control once. Its executed checks passed (exit 0); two companion-install
checks were skipped because that control did not build the optional helper.
The reference's full `load_error_test` exited 1 at the unchanged raw-read
page-cache threshold after the GGUF range controls passed. These are distinct
outcomes, not an all-passing baseline or a claimed cache fix.

The broad native executables predate the test-only temporary-directory follow-up
at `ab796a55`. The focused two-chip results above use the final header. Final
integrated rebuild results are recorded separately in `final-build-results.json`.
That explicit Release rebuild, focused regression, `prek`, and whitespace check
passed. Its full `load_error_test` passed all preceding GGUF controls, including
both zero-leak cohorts, then exited 1 at the same unchanged raw-read cache
threshold as the reference. Thus the earlier candidate full-suite pass and the
later cache-threshold failure are both retained. This change does not diagnose
or fix the updater observation or cache-threshold variability.

## Earlier attempts

An early M4 reference run reproduced both leaks but then stopped on an incorrect
test expectation for the malformed-part diagnostic. That expectation was corrected
before the final reference/candidate runs. The original result was retained.

An initial local dependency preflight and an initial remote harness setup failed
before GGUF tests. The first compiled M3 pair then failed to create the fixture
directory, so neither arm reached GGUF validation. After the test-only temporary
directory change, both final M3 arms completed as shown above. Those earlier
failures are not counted as successful checks.

The first reference-control build also stopped before tests because its source
export omitted a required catalog metadata file. The file was restored exactly
from the reference commit before the successful configure/build and single
reference executions described above.

## Reproduce

From the candidate checkout, the integrated regression is CPU-only:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 6 --target kapitsa load_error_test
build/load_error_test --gguf-split-cleanup
```

For the isolated old/new comparison, set `$EVIDENCE` to this evidence directory
and `$REF_SOURCE` to an output filename for the reference GGUF implementation.
Set `TMPDIR` to an existing writable test directory if needed.

```sh
git show 47dbe3cbd1516af07be108385236d6f40e492b61:src/gguf/gguf.cpp > "$REF_SOURCE"
clang++ -std=c++20 -O3 -I src -I third_party -I tools \
  "$EVIDENCE/focus.cpp" "$REF_SOURCE" -o reference-split-cleanup
clang++ -std=c++20 -O3 -I src -I third_party -I tools \
  "$EVIDENCE/focus.cpp" src/gguf/gguf.cpp -o candidate-split-cleanup
./reference-split-cleanup  # Expected exit 1 and two +32 descriptor deltas.
./candidate-split-cleanup # Expected exit 0 and two zero descriptor deltas.
```

## Not verified

No GPU, trained-model import, logits, HTTP model inference, or inference-performance
run was performed for this change. It changes cleanup after rejected metadata;
the synthetic read checks do not establish model-output equivalence or a speedup.
