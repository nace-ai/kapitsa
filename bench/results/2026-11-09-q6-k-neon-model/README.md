# Q6_K decoder: full-model validation

These are explicitly redacted derived exports, not original execution logs. Private original records remain archived.
Numeric results and source/binary identities are unchanged. The earlier CPU results in
[the CPU evidence folder](../2026-11-09-q6-k-neon-cpu/README.md) remain frozen.

Production candidate: `4f4475338a523de2d2c32a9c8e66c1fbdc623b23`.
Reference: `4b9ce88708282f754b76cbd7214e188786680c89`.
Both arms use fresh explicit Release configuration and their own binaries; actual target flags are
`-O3 -arch arm64`, with no `NDEBUG`. Source and binary hashes are in each IDENTITY file.

Runs: M3 Ultra / 60 GPU cores / 96 GiB / macOS 26.4.1 / Apple clang 17 (clang-1700.6.4.2);
M4 Pro / 20 GPU cores / 48 GiB / macOS 15.7.1 / Apple clang 17 (clang-1700.0.13.5).
The model is `bartowski/Qwen_Qwen3.5-4B-GGUF@4168f45a16a1290d65a4ec0fa312ae917a4c15d6`,
`Qwen_Qwen3.5-4B-Q4_0.gguf`, SHA256 `3e977de32dfb9377ff827f210091cce5d3c05c7b0f0b9b0c8005df7c0677a49b`.

## Conversion and numerical checks

Each chip performed fresh `--no-drafter` imports with both binaries. All nine imported files, including the
manifest, were compared in full and are byte-identical within each chip. The five source Q6_K tensors
(embedding and four attention-query weights) import as affine eight-bit/group64, checked against actual
configuration and tensor metadata. This does not assert that every module or the MTP weights use that format.

Early import attempts stopped at the prerequisite guard before conversion. Those failures remain in the
IDENTITY files. Later labeled attempts completed with the same source and binary identities. Only after
a complete literal comparison of the newly converted reference did the worker reuse identical retained bytes.
A previous conversion is not presented as a new run.

On each chip, each arm produced 584 rows: 8 prefill, 192 step, 192 verify and 192 batch rows; vocabulary 248,320.
All 145,018,880 float32 values per arm are finite. The complete 580,075,520 output bytes and recorded input bytes
match within each chip. Report-only comparisons show every row bitwise identical. Explicit zero-drift gates
pass for step/verify/batch; prefill has its separate finite/full-byte check because the comparator does not apply
those thresholds to prefill. No noise floor was modified.

The verify metadata says `KFlash tree (a chain of 8 live rows)`: this is teacher-forced trunk verification,
not a free-generating MTP output qualification. The CPU decoder changes no Metal source.

## Default precise serving

All model phases held the exclusive GPU lock with the caller-held markers. Each chip ran native A/B/A/B,
then the same fixed-request driver against default precise A/B/A/B servers.
Each fixed leg has three sampled and three greedy cases, using predeclared seeds 1/2/3.
All 24 decode records per chip completed 512 tokens with nonempty output and length finish. The six request
and complete generated-content SHA256 pairs match across all four legs on that chip. These are hash comparisons,
separate from the literal complete-byte numerical checks above.

The fixed driver is the existing
[fixed_seed_driver.cpp](../2026-11-08-q8-0-cpu/fixed_seed_driver.cpp), used unchanged
(SHA256 `6b022a4791ac656f74ca84dc7b2090e7bfd34e5ad0e7a1c6bc1197e8adcd8310`).
It was compiled once per chip against that chip's reference library; the benchmark helpers match in the
two sources. The seed is present in each actual request. Each IDENTITY file retains its executed driver hash.

| Chip | Matched adjacent decode-rate changes | Cold 4K TTFT, all four legs | HTTP reference / candidate |
| --- | --- | --- | --- |
| M3 Ultra | -1.71427% to +1.49398% | 2.06275–2.07230 s | each 54 PASS / 1 FAIL, rc 1 |
| M4 Pro | -1.98065% to +2.18487% | 2.27540–2.28028 s | each 55 PASS / 0 FAIL, rc 0 |

The fixed decode length/nonempty/rate/repetition, quiet, warmup, cold-prefill and execution gates pass
on both chips. All cached checks have empty flags/invalid lists, but their third follow-up ends at the
length limit in every leg; this is not a claim that all cached replies stop naturally.

Original native legs all return rc 0, with empty invalid lists and tool-call checks passing.
M3 A1 has a repetition warning plus failed runtime-environment qualification; M3 B1 has failed
runtime-environment qualification. M4 A1 also has failed runtime-environment qualification. Other native
legs have empty flags. Native results therefore do not all qualify for performance claims.

The M3 HTTP failure is the same queued-stream keepalive assertion in both arms. It requires two keepalives
before the first content token. Reference observed one at 15.12166 s, then content at 18.56775 s; candidate
observed one at 15.11207 s, then content at 23.44081 s. The failure is preserved, not called a pass.
Both fresh imported models are byte-identical, and this serving test does not invoke Q6_K import decoding;
these observations do not establish a new decoder regression. The holder/slot timing cause is not proven.


All per-leg results, original return codes, flags and invalid entries are retained in FIXED and API-NATIVE.
For a private machine-state flag, the derived export explicitly records failed runtime-environment qualification
without disclosing machine usage. Prefill/cache text was not byte-compared. Their later generated histories
are not asserted to match, so cached timings do not establish a matched-input speed change.
No whole-import or inference speedup, universal API success, or universal absence of regressions is claimed.

## Reproduction examples

The following are derived examples with placeholders and relative paths, not original command transcripts.
Use isolated checkouts and import homes; configure each source explicitly with its actual commit.
Take the exclusive GPU lock for import, model output and serving, and set both caller-held markers inside it.
Do not enter a second lock around a tool that already owns the lock.

```sh
cmake -S "$REF_SRC" -B "$REF_BUILD" -DCMAKE_BUILD_TYPE=Release -DKAPITSA_GIT="$REF_COMMIT"
cmake --build "$REF_BUILD" -j --target kapitsa logit_dump ni_run
cmake -S "$CAND_SRC" -B "$CAND_BUILD" -DCMAKE_BUILD_TYPE=Release -DKAPITSA_GIT="$CAND_COMMIT"
cmake --build "$CAND_BUILD" -j --target kapitsa logit_dump ni_run
# In the exclusive lock, with NI_LOCK_HELD=1 and NACE_GPU_LOCK_HELD=1:
"$REF_BUILD/logit_dump" "$REF_MODEL" output/reference --paths step,verify,batch --windows 8 --rows 24 --chunk 2048 --batch 4 --batch-rows 2 --corpus "$REF_SRC/bench/corpus.txt"
"$CAND_BUILD/logit_dump" "$CAND_MODEL" output/candidate --paths step,verify,batch --windows 8 --rows 24 --chunk 2048 --batch 4 --batch-rows 2 --from output/reference
python3 "$CAND_SRC/tools/logit_compare.py" output/reference output/candidate --no-gate --json report-only.json
python3 "$CAND_SRC/tools/logit_compare.py" output/reference output/candidate --min-top1 1 --max-mean-kl 0 --max-median-kl 0 --max-p99-kl 0 --near-tie 0 --max-far-flips 0 --max-nll-change 0 --json strict-zero.json
# Each A/B/A/B leg uses its corresponding binary and imported model.
KAPITSA_PROFILE=precise "$BUILD/kapitsa" bench --model "$MODEL" --quick --json --out native-new-label.json
KAPITSA_PROFILE=precise "$BUILD/kapitsa" serve --model "$MODEL" --model-id q6-k-neon-review --private --host "$BIND_HOST" --port "$PORT" --ctx 32768 --slots 1 --disk-cache 0
./fixed-seed-driver "$PORT" q6-k-neon-review "$SERVER_PID"
# HTTP used separate two-slot servers with the otherwise same settings.
python3 "$SRC/tests/test_server.py" "$SERVER_URL"
```

Whole-import timing, other neural model families and non-arm64 execution remain unverified.
