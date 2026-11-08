# Exact CPU Q8 load quantization

The existing Nemotron-H symmetric Q8 quantizer now uses SIMD for each finite 32-float block. The scalar float scale/reciprocal, stored half scale, nearest-away codes and clamp range remain unchanged. Nonfinite inputs, an overflowed reciprocal and overlapping input/output bytes retain the former scalar loops. No Metal source, precision, profile table or switch changes.

Reference: `95a0a552fab5c359dd91b716e9936918d436d713`. Tested production: `7ece78ea37e6f68a2d35ff8fa5e20753645b84c7`. Later edits correct one test comment and add this evidence; the tested production file is unchanged. [source-metadata.json](source-metadata.json) records exact extracted functions. Only function symbols, a `noinline` attribute and terminal file blank lines change in the exported translation units; both arms use the same compiler and flags without LTO. The repository license is included.

## Setups and measured CPU cost

- Apple M3 Ultra, 60 GPU cores, 96 GB RAM, macOS 26.4.1; Apple clang version 17.0.0 (clang-1700.6.4.2).
- Apple M4 Pro, 20 GPU cores, 48 GB RAM, macOS 15.7.1; Apple clang version 17.0.0 (clang-1700.0.13.5).

Actual Release targets use `-O3 -std=gnu++20 -arch arm64 -mmacosx-version-min=14.0`, without `NDEBUG`. The CMake cache's `-O3 -DNDEBUG` string is not the target flags. Standalone driver translation units use matching `-O3 -std=c++20` flags and no LTO.

The primary measurement makes a separate `K=3136` call for each of 1024 generated BF16-derived float rows. Each leg performs 16 passes, or 16,384 calls. Two warmup passes precede eight interleaved A/B/A/B/A/B/A/B legs. Every leg compares complete code/scale bytes. Timing excludes BF16 expansion, I/O, allocation and complete loading. These additional runs hold the exclusive model/GPU lock after the model jobs exit. All eight timing rows and input/output digests are retained in `m3-cpu-per-row.json` and `m4-cpu-per-row.json`.

| Setup | Reference ns/row | Candidate ns/row | Lower CPU latency |
| --- | ---: | ---: | ---: |
| M3 Ultra | 20722.576 | 5360.774 | 74.13% |
| M4 Pro | 3196.117 | 599.077 | 81.26% |

The original CPU driver separately flattens 1024 rows into **one `K×1024`-element call**, rather than 1024 row calls. It also retains eight legs per width, 16 repetitions per leg and complete byte gates. These warm CPU diagnostics precede the model phases; they are not total loading or inference measurements. They use this exact committed production implementation, rather than an earlier exploratory screen.

| Setup | Width label | A median ms/call | B median ms/call | Lower CPU latency |
| --- | ---: | ---: | ---: | ---: |
| M3 | 1536 | 10.397680 | 2.694371 | 74.09% |
| M3 | 2048 | 13.856747 | 3.582712 | 74.14% |
| M3 | 3072 | 20.759247 | 5.360762 | 74.18% |
| M3 | 3136 | 21.201492 | 5.480042 | 74.15% |
| M3 | 4096 | 27.688635 | 7.165754 | 74.12% |
| M4 | 1536 | 1.595172 | 0.287691 | 81.96% |
| M4 | 2048 | 1.936599 | 0.381854 | 80.28% |
| M4 | 3072 | 2.925505 | 0.573473 | 80.40% |
| M4 | 3136 | 2.968548 | 0.589104 | 80.16% |
| M4 | 4096 | 3.852556 | 0.770124 | 80.01% |

## Actual checkpoint and model outputs

The unchanged checkpoint is `nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8@3fe6dab75665a93884214ad4b1b95cf02717d081`. Its `lm_head.weight` is BF16 `[131072,3136]`, with 822,083,584 payload bytes. This dense FP8 model's existing default converts its BF16 head to Q8 at load, exercising the changed function.

Both setups stream all 131,072 real BF16 rows through the two production functions. Every code and half-scale byte compares equal; nonfinite input count is zero. Input/code/scale SHA256 and full row counts are in `m3-real-head.json`/`m4-real-head.json`. A hash alone does not replace the executed row-by-row byte gate.

Each setup's own Release `logit_dump` produces 584 rows/arm: 8 prefill, 192 step, 192 teacher-forced verify, and 192 batch. Each arm contains 76,546,048 finite float32 values. The executed audit compares every logits byte and unchanged recorded inputs, not just selected tokens or hashes. Both arms are fully byte-identical. Report-only metrics and explicit strict-zero decoding gates pass with no floor modification. Prefill has its own complete-byte/finite audit; its report is kept separate from the decoding gate. See `*-output.json`, `*-report-only.json` and `*-strict-zero.json`.

On each setup, four default precise servers run A/B/A/B against six predeclared seed1/2/3 requests per leg. All 24 outputs per setup have 512 tokens, nonempty content, length termination, passing rate/repetition gates and matching request/full-content SHA across all four legs. Warmup, quiet, execution and cold-prefill gates pass. The rates below are the harness's middle-window estimates, which can use a character-based token estimate.

- M3: adjacent matched-case middle-window rate deltas -0.260% to +2.526%; cold ~4K TTFT 1.862403–1.879872 s.
- M4: adjacent matched-case middle-window rate deltas -0.738% to +0.695%; cold ~4K TTFT 4.327482–4.331045 s.

No inference acceleration or universal absence of regression is claimed. The six decode content hashes do not cover generated prefill/cache texts. Prefill/cache texts were not byte-compared. Later cached turns contain generated histories and therefore are not matched-input cached-performance pairs.

## CPU checks and original limitations

The committed-source build, 22 `vk cpu-tests` entries (including the new registered oracle), six supplemental CPU scripts, `prek` and diff check complete. The oracle passes 278,058 code/scale/canary cases, including floating flags and boundary/overlap coverage; M3 runs its own registered binary too. Fourteen of fifteen supplemental native tests pass. `load_error_test` fails its existing cache-residency assertion. A separately rebuilt reference95a0 executable is byte-identical to the failed candidate executable, so no duplicate execution is required; this is an identical-executable control, not a separately run reference PASS. Original failing details stay privately verifiable; `cpu-integration.json` retains the failure and binary identities.

The initial changed-source Release oracle, before the production commit, also passes 278,058 cases; its source bodies match the committed files, but its old version label is explicitly recorded. The exact-function extraction/shim sanitizer run passes 273,907 finite cases, deliberately skipping 4,151 exceptional legacy cases; it is not whole-library sanitizer coverage. Truncating rounding and removing the overlap guard are both rejected by the independent oracle. The initial raw safe-text outcomes and source identities are retained.

M4 default HTTP runs each pass 49/49 checks. M3 reference HTTP has 39 PASS/10 FAIL; candidate has 36 PASS/12 FAIL, plus the int-argument replay assertion is not executed because the returned tool arguments do not contain `max_lines`. Failure sets differ and the API cause remains unlocated. `*-http.json` lists every executed failure by test name; original detailed responses remain private.

All native `kapitsa bench --quick` legs return zero with `invalid=[]`. M3 nevertheless fails all six tool-semantic cases in every leg, has cached first-turn length warnings in B1/A2, and a repetition flag in B2. M4 tool semantics pass, but reference A2 has a repetition flag. These real flags are retained in `*-native.json` and are not replaced by fixed-request results. M3 fixed cached first-turn length warnings appear in all four legs; M4 fixed cached flags are empty. The exact quantizer byte and model-output evidence does not support attributing those API limitations to this CPU change, and does not establish universal API correctness.

Other Nemotron-H checkpoints, M5/ANE/split/vision environments, whole-library sanitizers, total cold-loading acceleration and every possible generated dialogue were not verified. No Metal kernel changed, so the small-kernel shader-validation procedure is inapplicable here.

## Portable reproduction examples

These are derived relative-path examples, **not raw execution logs**. Original commands and private records are retained separately. Resolve `$REPO` (the source checkout), `$MODEL`, `$MODELFILE` and `$HEAD_OFFSET` from the unchanged checkpoint. Use the authorized setup's exclusive lock; do not nest a lock under an already held lock.

```sh
cd production-driver
clang++ -std=c++20 -O3 -c reference-q8.cpp -o reference-q8.o
clang++ -std=c++20 -O3 -c candidate-q8.cpp -o candidate-q8.o
clang++ -std=c++20 -O3 -Wno-deprecated-declarations cpu-evidence.cpp reference-q8.o candidate-q8.o -o cpu-evidence
clang++ -std=c++20 -O3 -Wno-deprecated-declarations cpu-per-row-evidence.cpp reference-q8.o candidate-q8.o -o cpu-per-row-evidence
./cpu-evidence
NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 lockf -k -t 0 "$BENCH_GPU_LOCK" ./cpu-per-row-evidence
NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 lockf -k -t 0 "$BENCH_GPU_LOCK" ./cpu-evidence --bf16 "$MODELFILE" "$HEAD_OFFSET" 131072 3136
cd "$REPO"
NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 lockf -k -t 0 "$BENCH_GPU_LOCK" "$REF_BUILD/logit_dump" "$MODEL" output/reference --paths step,verify,batch --windows 8 --rows 24 --chunk 2048 --corpus bench/corpus.txt
NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 lockf -k -t 0 "$BENCH_GPU_LOCK" "$CAND_BUILD/logit_dump" "$MODEL" output/candidate --paths step,verify,batch --windows 8 --rows 24 --chunk 2048 --from output/reference
python3 tools/logit_compare.py output/reference output/candidate --json report-only.json
python3 tools/logit_compare.py output/reference output/candidate --min-top1 1 --max-mean-kl 0 --max-median-kl 0 --max-p99-kl 0 --near-tie 0 --max-far-flips 0 --max-nll-change 0 --json strict-zero.json
```

## Export identity

All JSON summaries are explicitly privacy-filtered derivatives. Allowed configuration, source/binary/model identities, methods, measured numbers and pass/fail/flags remain intact. Raw archives and private environment records are not uploaded. Driver functions retain their exact collected bodies; the exported reference/candidate files omit terminal blank lines, with original and exported hashes identified separately. Safe initial outcome lines retain their collected bytes. The manifest hashes this public export, not the private archive. No existing frozen evidence is changed.
