# Q8_0 CPU decoder evidence

The JSON files are explicitly sanitized derived reports from the recorded exact-source executions. Original commands, logs and result receipts remain privately verifiable. Machine usage, addresses and private paths are omitted; numerical results and genuine validation limitations are retained. No inference-speed claim follows from the CPU row measurements.

`M3-CPU-ROWS.json` and `M4-CPU-ROWS.json` retain every timing leg. `FOCUSED-CPU.json` distinguishes Release, sanitizer, mutants and exact-executable reuse. `M4-CPU-CHECKS.json` preserves the independent U8 cache assertion failures. `M4-IMPORT-COMPARISON.json` checks every imported file and actual Q8_0 module formats; `M4-MODEL-OUTPUT.json` preserves full-row output gates and strict-zero scope. The copied `fixed_seed_driver.cpp` is the byte-identical common driver for the fixed-request tests; its private execution-condition fields remain private when results are exported. It links against the reference official library and drives each arm's separately started server using the same request bytes and seeds. The final default-request reports retain original failures and warnings through explicitly identified sanitized descriptions.

`M4-Q8-ROW-GATHER.json` records the separate default-prefetch row/gather composition checks, using the copied `q8_ngram_row_gather.cpp` and a synthetic wrapper around real Q8_0 bytes. The wrapper is not a trained PLE model; this check measures no performance. Prefetch-off was not run.

The copied `q8_0_rows.cpp` measurement driver is byte-identical to the source used for these CPU intervals. It links against each source's official library. Build with the same Release ARM64 flags and frameworks as that library; gate/hash/allocation work is outside its measured row loop.

The following portable examples are derived reproductions, not original execution logs. `$REF_BUILD`, `$CAND_BUILD`, `$REF_IMPORT_HOME`, `$CAND_IMPORT_HOME`, `$REF_MODEL`, `$CAND_MODEL`, `$GGUF`, `$FIXTURE`, `$ROWS_A` and `$ROWS_B` identify your independently built/reference/candidate artifacts. Use fresh output labels; frozen published runs must not be overwritten.

```sh
: "${BENCH_GPU_LOCK:?Set the exclusive GPU lock file}"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j --target kapitsa iq_layout_test logit_dump ni_run
build/iq_layout_test

# Each arm uses its own isolated import home and exact-source CLI.
KAPITSA_HOME="$REF_IMPORT_HOME" NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 lockf -k "$BENCH_GPU_LOCK" "$REF_BUILD/kapitsa" import "$GGUF" --name q8-review --threads 4 --no-drafter
KAPITSA_HOME="$CAND_IMPORT_HOME" NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 lockf -k "$BENCH_GPU_LOCK" "$CAND_BUILD/kapitsa" import "$GGUF" --name q8-review --threads 4 --no-drafter

"$ROWS_A" --input "$FIXTURE" --rows 1024 --cols 4096 --make-oracle --output oracle.f32
# Run A1 B1 A2 B2 A3 B3 A4 B4 with both drivers; check input/binary hashes around each leg.
"$ROWS_A" --input "$FIXTURE" --rows 1024 --cols 4096 --passes 16 --oracle oracle.f32 --output A1.f32
"$ROWS_B" --input "$FIXTURE" --rows 1024 --cols 4096 --passes 16 --oracle oracle.f32 --output B1.f32

NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 lockf -k "$BENCH_GPU_LOCK" "$REF_BUILD/logit_dump" "$REF_MODEL" output-A --paths step,verify,batch --windows 8 --rows 24 --chunk 2048 --batch 4 --batch-rows 2 --corpus bench/corpus.txt
NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 lockf -k "$BENCH_GPU_LOCK" "$CAND_BUILD/logit_dump" "$CAND_MODEL" output-B --paths step,verify,batch --windows 8 --rows 24 --chunk 2048 --batch 4 --batch-rows 2 --from output-A
python3 tools/logit_compare.py output-A output-B --no-gate --json report-only.json
python3 tools/logit_compare.py output-A output-B --min-top1 1 --max-mean-kl 0 --max-median-kl 0 --max-p99-kl 0 --near-tie 0 --max-far-flips 0 --max-nll-change 0 --json strict-zero.json
```

For output comparisons, `$REF_MODEL` and `$CAND_MODEL` identify the independently imported directories for their respective arms. The strict-zero tool thresholds apply to step/verify/batch; prefill additionally requires its independent finite/full-byte comparison. Teacher-forced KFlash tree verification does not qualify freely generated MTP numerics. No floor is modified.
