# Four-bit CPU packing: full-model validation

These are explicitly redacted derived exports, not original execution logs. Private original records remain archived.
Numeric results and source/binary identities are unchanged. The earlier
[CPU evidence](../four-bit-pack-neon-20261109/README.md) is frozen.

Production candidate: `7d1b0bbcdcfb36a495036eb0d6d00405fd5955ca`.
Reference: `4b9ce88708282f754b76cbd7214e188786680c89`.
Each source was explicitly configured in Release and built independently; actual target flags are
`-O3 -arch arm64`, without `NDEBUG`. IDENTITY records the source and binary hashes.
The published branch also removes a trailing blank test-header line, with identical rebuilt object and executable;
that formatting-only change is not a different production algorithm or a repeated model run.

Runs: M3 Ultra / 60 GPU cores / 96 GiB / macOS 26.4.1 / Apple clang 17 (clang-1700.6.4.2);
M4 Pro / 20 GPU cores / 48 GiB / macOS 15.7.1 / Apple clang 17 (clang-1700.0.13.5).
Model: `bartowski/Qwen_Qwen3.5-4B-GGUF@4168f45a16a1290d65a4ec0fa312ae917a4c15d6`,
`Qwen_Qwen3.5-4B-Q4_0.gguf`, SHA256 `3e977de32dfb9377ff827f210091cce5d3c05c7b0f0b9b0c8005df7c0677a49b`.

## Conversion and numerical checks

Each chip ran one successful fresh `--no-drafter` reference conversion and one candidate conversion.
All nine files, including the manifest, were compared literally and match within each chip.
The Q4_0 source modules `layers.0.linear_attn.in_proj_z` and `in_proj_qkv` import as affine four-bit/group32:
U32 packed weights and BF16 scales/biases, checked against actual tensor metadata. Other module/MTP formats
are retained; this does not assert every tensor is four-bit.

Earlier prerequisite guards stopped before conversion (two M3 attempts, one M4 attempt). Their rc 1 records remain
in IDENTITY. Later labeled attempts use unchanged source and binary identities. Only after a full literal
comparison of each freshly converted reference did the worker reuse identical retained bytes. Existing conversion
results are not presented as newly executed conversions.

Each arm on each chip produced 584 rows: 8 prefill, 192 step, 192 verify and 192 batch, vocabulary 248,320.
All 145,018,880 float32 values per arm are finite; all 580,075,520 output bytes and scored input bytes match
within each chip. The report-only comparison marks every row bitwise identical. Explicit zero-drift gates
pass for step/verify/batch. Prefill has a separate finite/full-byte check because these comparator thresholds
do not apply to prefill. No noise floor was modified.

The actual verify metadata is `KFlash tree (a chain of 8 live rows)`: teacher-forced trunk verification,
not free-generating MTP qualification. No Metal source changed.

## Default precise serving

All model phases held the exclusive GPU lock and caller-held markers. Each chip ran native A/B/A/B and then
fixed-request default precise A/B/A/B, using predeclared seeds 1/2/3 for three sampled and three greedy cases.
Every fixed leg completed six nonempty 512-token records with length finish and positive rate. All six request
and full generated-content SHA256 pairs match across four legs on each chip. Hash comparisons apply only to
these six decode cases; they are separate from the literal numerical byte comparisons above.

The unchanged [fixed driver](../2026-11-08-q8-0-cpu/fixed_seed_driver.cpp) has SHA256
`6b022a4791ac656f74ca84dc7b2090e7bfd34e5ad0e7a1c6bc1197e8adcd8310`; it was compiled once per chip against
that chip's reference library. Shared benchmark helpers match between source versions. The seed appears in
each actual request. FIXED contains every leg and matched adjacent decode-rate pair.

| Chip | Matched adjacent decode-rate changes | Cold 4K TTFT (A1, B1, A2, B2) | HTTP reference / candidate |
| --- | --- | --- | --- |
| M3 | -1.19608% to +1.45453% | 2.06908, 2.06599, 2.06485, 2.06844 s | 54 PASS, 1 FAIL (rc 1) / 54 PASS, 1 FAIL (rc 1) |
| M4 | -0.62045% to +0.79031% | 2.27647, 2.27692, 2.27329, 2.27733 s | 55 PASS, 0 FAIL (rc 0) / 55 PASS, 0 FAIL (rc 0) |

| Native leg | rc | Tool checks | Flags | Invalid |
| --- | ---: | --- | --- | --- |
| M3 A1 | 0 | pass | none | none |
| M3 B1 | 0 | pass | none | none |
| M3 A2 | 0 | pass | none | none |
| M3 B2 | 0 | pass | none | none |
| M4 A1 | 0 | pass | none | none |
| M4 B1 | 0 | pass | none | none |
| M4 A2 | 0 | pass | none | none |
| M4 B2 | 0 | pass | none | none |

- M3 reference: the queued-stream test failed its requirement for two keepalives before the first content token. The complete original assertion and measured event times remain in API-NATIVE.
- M3 candidate: the queued-stream test failed its requirement for two keepalives before the first content token. The complete original assertion and measured event times remain in API-NATIVE.

The separately converted model files are byte-identical. This serving test does not call CPU four-bit import packing; its slot/keepalive timing cause is not established, and this result is not evidence of a new packer regression. The failed checks are not counted as passes.

Fixed decode length/nonempty/rate/repetition, quiet, warmup, cold-prefill and execution gates pass on both chips.
Every fixed cached third follow-up finishes at the length limit, even where flags and invalid are empty;
this is not a claim that all cached replies stop naturally. Prefill/cache text was not byte-compared.
Later generated histories are not asserted to match, so cached timings do not establish matched-input performance.
Native generated histories are likewise not asserted to match. All original return codes, flags and failed checks
are retained in the derived records, with any private environment detail explicitly omitted.
The API fixture labels that mention a file are predefined tool-call prompts, not private file contents.

These checks qualify the exact CPU packing change on the two measured setups. The CPU component improvement
does not establish whole-import or inference acceleration. No universal API success or universal absence of
performance regressions is claimed.

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
KAPITSA_PROFILE=precise "$BUILD/kapitsa" serve --model "$MODEL" --model-id pack4-neon-review --private --host "$BIND_HOST" --port "$PORT" --ctx 32768 --slots 1 --disk-cache 0
./fixed-seed-driver "$PORT" pack4-neon-review "$SERVER_PID"
# HTTP used separate two-slot servers with the otherwise same settings.
python3 "$SRC/tests/test_server.py" "$SERVER_URL"
```

Whole-import timing, other neural model families and non-arm64 execution remain unverified.
