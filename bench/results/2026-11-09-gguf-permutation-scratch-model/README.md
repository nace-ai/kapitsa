# Four-bit permutation: completed model validation

These are explicitly privacy-filtered derived summaries, not raw execution logs.
Original records remain private. No previous result file is modified: the earlier
[CPU-only snapshot](../2026-11-09-gguf-permutation-scratch/README.md) retains its
then-pending model plan, which this additive record completes.

Candidate production `9493a437a823343a2b257767c45eae9a98e7827c` changes only the
direct exact-four-bit `convert_rows` scratch/permutation; reference is
`39446144b7f181eeecaf4bdf5a4ddb34bb5dd104`. No Metal, decoder, rounding, profile,
floor or production preflight guard changed. Both builds explicitly configured
Release with actual `-O3`, arm64 and no `NDEBUG`; BUILDS files preserve source and
binary identities. All model phases held the exclusive GPU lock with caller-held
markers. Both workers and their six explicit servers per setup completed and exited.

Runs: M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1, Apple clang 17.0.0
(clang-1700.6.4.2); M4 Pro, 20 GPU cores, 48 GiB, macOS 15.7.1, Apple clang
17.0.0 (clang-1700.0.13.5). Model on both:
`bartowski/Qwen_Qwen3.5-4B-GGUF@4168f45a16a1290d65a4ec0fa312ae917a4c15d6`,
`Qwen_Qwen3.5-4B-Q4_0.gguf`, 2,776,033,248 bytes,
SHA256 `3e977de32dfb9377ff827f210091cce5d3c05c7b0f0b9b0c8005df7c0677a49b`.

## Imported weights and numerical outputs

Each setup executed independent reference and candidate `--no-drafter` imports.
All nine complete files, including the manifest, compare literally equal before
deduplicating their retained storage. The measured trunk
`language_model.model.layers.4.linear_attn.out_proj` is affine four-bit/group32:
U32 weight `[2560,512]`, BF16 scales/biases `[2560,128]`. Per-module and MTP
differences remain; this is not a claim that every tensor has the same format.

Each arm produced 584 full-vocabulary rows: 8 prefill, 192 step, 192 verify and
192 batch, with vocabulary 248,320. All 145,018,880 float32 values per arm are
finite. Complete 580,075,520-byte output and scored-input byte comparisons pass.
The report-only comparison reports all rows bitwise identical and zero drift.
An independent explicit strict-zero comparison passes step/verify/batch; those
flags do not threshold prefill, whose finite and literal-byte gate passed separately.
No numerical floor was changed. Verify uses teacher-forced trunk KFlash tree
verification with a chain of eight live rows, not free-generating MTP qualification.

## Default runtime checks

Each setup ran native A1/B1/A2/B2 plus a separate fixed-request A1/B1/A2/B2 cohort.
Each arm uses its own imported model and exact-source binary, default precise
profile, context 32768 and disk cache disabled. Native quick benchmarks start
their own servers. Fixed runs use explicit single-slot servers and the identical
[fixed-seed driver](../2026-11-08-q8-0-cpu/fixed_seed_driver.cpp), SHA256
`6b022a4791ac656f74ca84dc7b2090e7bfd34e5ad0e7a1c6bc1197e8adcd8310`, compiled
against the same reference library for both arms on each setup. The driver's
request seeds 1/2/3 are consumed by the actual sampler. Each of six sampled/greedy
decode cases has identical request and complete generated-content SHA256 across
the four legs. All 24 rows per setup generate 512 nonempty tokens and pass length,
rate, repetition, execution and quiet gates. Cold-prefill gates pass too.

| Setup | Matched adjacent decode-rate change range | HTTP reference / candidate |
| --- | --- | --- |
| M3 Ultra | -1.581849% to +1.302739% | each 54 PASS, 1 FAIL, rc 1 |
| M4 Pro | -0.867818% to +2.291569% | each 55 PASS, 0 FAIL, rc 0 |

Every fixed row, native timing leg, output hash and qualification flag is preserved
in the derived JSON files. Native tool semantic checks pass in all four legs on
both setups, with rc 0 and no `invalid` entries. M3 has no native flags. M4 native
B2 has a failed runtime environment gate for `decode_greedy`; that original timing
is unqualified. Its flag remains explicitly redacted as an environment failure,
with private usage detail omitted. It was not silently replaced by a later run.
The separate fixed cohort passed its gates on every leg. None of these timing
variations establish an inference speedup or universal absence of regression.

On M3, both sources fail the unchanged queued chat keepalive fixture: it requires
two empty chunks around 15/30 seconds, but first content arrived at 18.42-18.68
seconds after just one keepalive. The same reference failure is not evidence of
a converter regression. No HTTP all-pass claim is made. HTTP.json retains actual
exit codes, counts and event timings; generated response IDs are omitted.

Prefill and cached-response text were not byte-compared. The six decode content
hashes do not cover them, and different generated follow-up histories are not
matched-input cached performance. Other model families, complete import/loading
latency, free-generating MTP numerical behavior and universal API behavior remain
unverified. Q4_1/Q4_K have CPU differential coverage, not separate full models.

Original orchestration/preflight failures are retained: M3 initially lacked source
hash metadata, M4 initially lacked an explicit CMake executable, and each first
reference import stopped at an unchanged preflight guard before conversion.
Only the failed phases were resumed after fixing the validation environment.
Completed configure/build and later successful model phases were not rerun.

## Derived reproduction examples

These portable commands describe the executed method; they are not original logs.
Set `$REF_SRC`, `$CAND_SRC`, `$REF_BUILD`, `$CAND_BUILD`, `$GGUF`, `$REF_HOME`,
`$CAND_HOME`, `$REF_MODEL` and `$CAND_MODEL` to the corresponding exact sources,
builds and fresh imports. Use a new output directory for each experiment. Run
each model command under the exclusive lock; a wrapper holding it passes both
caller-held markers to children and retains it until any server has stopped.

```sh
cmake -S "$REF_SRC" -B "$REF_BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 -DKAPITSA_GIT=39446144b7f181eeecaf4bdf5a4ddb34bb5dd104
cmake --build "$REF_BUILD" -j 2 --target kapitsa logit_dump ni_run
cmake -S "$CAND_SRC" -B "$CAND_BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 -DKAPITSA_GIT=9493a437a823343a2b257767c45eae9a98e7827c
cmake --build "$CAND_BUILD" -j 2 --target kapitsa logit_dump ni_run
lockf -k /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 KAPITSA_HOME="$REF_HOME" "$REF_BUILD/kapitsa" import "$GGUF" --name qwen35-exact4-permutation-review --threads 4 --no-drafter
lockf -k /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 KAPITSA_HOME="$CAND_HOME" "$CAND_BUILD/kapitsa" import "$GGUF" --name qwen35-exact4-permutation-review --threads 4 --no-drafter
lockf -k /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 KAPITSA_PROFILE=precise "$REF_BUILD/logit_dump" "$REF_MODEL" output/reference --paths step,verify,batch --windows 8 --rows 24 --chunk 2048 --batch 4 --batch-rows 2 --corpus bench/corpus.txt
lockf -k /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 KAPITSA_PROFILE=precise "$CAND_BUILD/logit_dump" "$CAND_MODEL" output/candidate --paths step,verify,batch --windows 8 --rows 24 --chunk 2048 --batch 4 --batch-rows 2 --from output/reference
python3 tools/logit_compare.py output/reference output/candidate --no-gate --json report-only.json
python3 tools/logit_compare.py output/reference output/candidate --min-top1 1 --max-mean-kl 0 --max-median-kl 0 --max-p99-kl 0 --near-tie 0 --max-far-flips 0 --max-nll-change 0 --json strict-zero.json
lockf -k /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 KAPITSA_PROFILE=precise "$REF_BUILD/kapitsa" bench --model "$REF_MODEL" --quick --json --out native_A1.json
```

Repeat native with the respective candidate/reference binaries and models in
B1/A2/B2 order. For fixed runs, start each corresponding server under one retained
exclusive lock with `serve --model "$MODEL" --model-id exact4-permutation-review
--private --host 127.0.0.1 --port "$PORT" --ctx 32768 --slots 1 --disk-cache 0`,
then invoke the common driver with `$PORT`, `exact4-permutation-review` and
`$SERVER_PID`. Stop it before switching arms. HTTP uses each arm's
`tests/test_server.py "http://127.0.0.1:$PORT"` against its own two-slot server.
The lock and server lifecycle are required parts of these examples.
