# GGUF row-worker launch cleanup

These files are explicitly redacted, structured derivations of private execution
records. They are not raw execution logs. Source, test outcomes and failure cases
are preserved; command examples below use repository-relative paths or explicit
placeholders. Private originals and their digests are retained for audit.

## Source and scope

- Reference: `47dbe3cbd1516af07be108385236d6f40e492b61`.
- Production and regression: `49e5b3213d3f0a441f9358dd91d3aa325ad963c9`.
- CI wiring only: `53ee62c44fb85fed413f279b6ad05f1769251552`.
- The compiled CLI reports `kapitsa 0.2.32 (49e5b321)`. Evidence and CI-only
  commits do not change the production source tested here.

The catch around `parallel_rows`' launch loop joins successfully started workers
before unwinding the callback, shared error state, mutex and conversion buffers.
It then rethrows the launch exception. Conversion arithmetic and row partitioning
are not changed. This evidence establishes CPU exception lifetime behavior, not
model output or inference performance.

## Focused method and results

`tests/test_gguf_parallel_rows.py` extracts the actual complete function without
rewriting it, then compiles it with a common C++20 driver at `-O3 -pthread`.
The caller's `operator new` is instrumented to throw `bad_alloc` at every allocation
observed during a four-worker, 64-row launch. Worker allocations are not injected.
Each case runs in a fresh process. A terminate handler exits with sentinel 86
instead of aborting, so a reference failure is a recorded `std::terminate` call.

Each arm first passes 51 controls: row coverage for seven row counts (including
zero) and seven thread counts (including nonpositive values), plus serial and
parallel worker-error propagation. The launch sweep runs with and without a worker
exception: 15 injected allocations per mode, plus no-injection and one-past-last
controls, 34 subprocess cases total.

On both Macs, reference allocation positions 5–15 invoke terminate in both modes
(22 failed cases). Candidate joins started workers and rethrows `bad_alloc` in all
30 injected cases; all controls pass. The same candidate matrix passes ASan/UBSan
on M4. M3 sanitizer execution was not performed. Matrices preserve every case,
exit status, started/finished worker count and terminate diagnostic.

| Setup | macOS | Compiler | Focused evidence |
|---|---|---|---|
| M3 Ultra, 60 GPU cores, 96 GiB | 26.4.1 | Apple clang 17, clang-1700.6.4.2 | [reference/candidate](M3-fault-matrix.json) |
| M4 Pro, 20 GPU cores, 48 GiB | 15.7.1 | Apple clang 17, clang-1700.0.13.5 | [reference/candidate](M4-fault-matrix.json), [ASan/UBSan](M4-sanitizer.json) |

Initial remote staging failed before test execution. The completed remote test
used the same driver and source files; their SHA256 values matched the local files.

## Broad checks and limits

- Explicit Release configure/build and recorded binary identity: pass.
- `vk cpu-tests`: 21/22 entries pass; the original CLI clone-removal reporting assertion fails. One isolated fixture execution per reference/candidate passes both assertions; the original failure remains unexplained.
- Supplementary native tests: 18/19 pass; `load_error_test` fails the existing raw-read cache threshold. Its complete executable bytes equal the retained reference build, whose prior execution also failed that assertion. No reference loader rerun is claimed.
- Six supplementary Python checks, converter packing check, patch-application check, `prek`, diff check and cleanup pass.
- Both-chip focused regression and M4 ASan/UBSan pass. See [broad results](M4-broad-checks.json) for individual stage outcomes.

The original full CLI suite has one failure in its clone-removal reporting check.
A separate extraction of that exact fixture passes both assertions on one fresh
reference execution and one candidate execution. This does not erase or diagnose
the original failure. The `rm` path calls `hub::freeable_bytes`; it does not invoke
the GGUF row helper.

`load_error_test` fails its existing `uncached_reads` threshold. The newly built
candidate executable and the retained reference executable compare equal over
their complete bytes (SHA256
`45ef4dd8cca8d0d01a944c4246585d4c56b3260c05f97cda63b193182e8279ad`).
The prior reference execution also failed that assertion; it was not repeated
here. The importer object is not linked into this test. This is a same-executable
control, not a claim that the loader suite passed.

No GPU, trained-model import, logits, generated text or speed measurements were
run. Fault injection covers allocations in the actual helper, not a full import
transaction or OS-level `pthread_create` exhaustion. Unrelated importer cleanup
is not qualified by this fix.

## Reproduction examples

These are portable examples, not verbatim execution logs:

```sh
python3 tests/test_gguf_parallel_rows.py
git show 47dbe3cbd1516af07be108385236d6f40e492b61:src/gguf/import.cpp > "$REF_SOURCE"
python3 tests/test_gguf_parallel_rows.py --source "$REF_SOURCE"
python3 tests/test_gguf_parallel_rows.py --sanitize
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 6 --target kapitsa load_error_test
```

The reference regression invocation is expected to exit 1 and list the 22
terminate cases on the tested compiler versions; it is not expected to pass.
