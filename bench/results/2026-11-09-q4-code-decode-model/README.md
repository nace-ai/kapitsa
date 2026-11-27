# Q4 CPU decode: full-model validation

Explicitly redacted derived evidence. Public files contain permitted hardware/compiler configuration,
source/model identities, methods and actual outcomes; private raw records are not uploaded.

Production: `5bcbe7e2069f55485473ba45755ab4c65a3c44c5`.
Reference: `87c105238186beadb375805e47a89791a499319a`.
Model: `bartowski/Qwen_Qwen3.5-4B-GGUF@4168f45a16a1290d65a4ec0fa312ae917a4c15d6`,
`Qwen_Qwen3.5-4B-Q4_0.gguf`; the IDENTITY files preserve its complete SHA256.
The import plan includes 180 Q4_0 and four Q4_1 tensors, with the other original formats retained.
The CPU timing evidence measures Q4_0 real rows only; this mixed model is not a separate Q4_1 timing setup.

Each chip uses fresh explicitly configured Release reference/candidate builds and independently
converts the complete model with `--no-drafter`. Nine complete model/config/tokenizer/manifest files
are compared literally. Source and binary identities and original step return codes are in IDENTITY.
M3 reuses previously retained tensor bytes only after its fresh reference conversion and full literal
comparison; its fresh versioned manifest is preserved. M4 retains both fresh imports through comparison.

Each arm has 584 rows: eight prefill plus 192 step, 192 verify and 192 batch rows, vocabulary 248320,
145018880 float32 values. All values are finite; complete outputs and recorded inputs are byte-equal.
Report-only and strict-zero checks pass; prefill has a separate finite/literal check. No floor changes.
The actual verify path is teacher-forced KFlash trunk with an eight-live-row chain, not free-generating MTP.

Default precise validation uses native A/B/A/B, then fixed seeds 1/2/3 with 24 x 512-token decode records
per chip and six identical request/content SHA pairs across all four legs, then both-arm HTTP checks.
GPU stages use the exclusive lock and caller-held markers. No Metal kernel was changed.

| Chip | Matched adjacent decode-rate changes | Cold 4K TTFT (A1, B1, A2, B2) | HTTP reference; candidate |
|---|---:|---:|---|
| M3 | -1.32458% to +0.97881% | 2.09785, 2.09662, 2.09666, 2.09415 s | 54 PASS / 1 FAIL (rc 1); 54 PASS / 1 FAIL (rc 1) |
| M4 | -0.67788% to +2.19251% | 2.34057, 2.31477, 2.28689, 2.29464 s | 55 PASS / 0 FAIL (rc 0); 55 PASS / 0 FAIL (rc 0) |

HTTP failures, if any, are retained below and in API-NATIVE; none is counted as a pass:

- M3 reference: `FAIL chat stream: while every slot is busy the stream starts at once, gets an empty chunk every 15 s, and its first token comes once a slot frees  first chunk (0.0030698776245117188, {'id': 'chatcmpl-179419400691745', 'object': 'chat.completion.chunk', 'created': 1794194006, 'model': 'q4-codes-neon-review', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]}), keepalives at [15.133121967315674], first token at 18.618426084518433`
- M3 candidate: `FAIL chat stream: while every slot is busy the stream starts at once, gets an empty chunk every 15 s, and its first token comes once a slot frees  first chunk (0.0027022361755371094, {'id': 'chatcmpl-179419416591745', 'object': 'chat.completion.chunk', 'created': 1794194165, 'model': 'q4-codes-neon-review', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]}), keepalives at [15.115432024002075], first token at 23.313252210617065`

The converted model files are byte-identical. Serving these converted files does not call CPU Q4
import expansion. API/cache limitations are disclosed without asserting their cause or adding an
unrelated serving fix. Every native rc/flag/invalid, tool-call outcome, and fixed cached follow-up
is retained in the JSON, including any warnings. Six decode content hashes do not cover prefill/cache
text; later generated histories are not a matched-input cache-performance comparison. All original
cached third turns ended at `length`, and those outcomes remain in the evidence.

Recovery history: earlier import prerequisite attempts stopped before conversion. M3 resumes its
original builds. A retained-model manifest version transition from 0.2.31 to 0.2.32 required a private
harness adjustment; new manifest bytes are preserved, and current A/B comparison is unmodified.
The earlier M4 temporary validation environment was lost after an observed successful nine-file
import comparison; complete later output/default evidence was unavailable and is not counted.
M4 therefore rebuilt/reimported the identical source commits in a persistent task directory.
The M3 fixed driver initially failed its temporary-file prerequisite before compilation; a new labeled
compile uses the same source/flags and continues only the previously unexecuted fixed/HTTP phases.
A private recovery harness initially selected the old reference model location; that server exited
before requests. It was corrected to the model reference recorded by the successful native run, with
model/source bytes unchanged. The failed server attempt remains in IDENTITY; no numerical/native replay.
This table describes that recovery run only. Completed CPU oracle/timing checks were not repeated.

M4's original native B1 and A2 include environment qualification warnings. The original fixed A1
also has a warning on its last greedy decode case and prefill. Original rates above are retained
observations, and the warning-bearing measurements do not qualify an inference performance claim.
`M4-FIXED-SUPPLEMENT.json` preserves a separately labeled, predeclared additional complete A/B/A/B
cohort with the exact same binaries, imported models and driver. Its actual warnings and qualification
status remain explicit; it does not replace the original cohort or silently convert a failed condition
into a pass. The supplement is the only extra cohort; every leg is retained.

The directly measured speed change is CPU import expansion. Its two-chip CPU interleaved measurements
and complete code/scale/offset byte gates are in the frozen CPU evidence. The only production caller
of `decode4` is in `src/gguf/import.cpp`; serving the already imported model does not call this function.
The independent full-file import and GPU numerical comparisons check that this CPU transformation
preserves the resulting model. The serving measurements supplement that evidence without establishing
a whole-inference benefit or proving the cause of any measurement warning.

These results do not establish whole-import or inference acceleration, universal API success,
universal absence of performance regression, free-generating MTP numerical qualification, prefill/cache text equality,
other neural model families, non-arm64 execution, or a separate Q4_1 timing setup.

Reproduction commands must use the pinned model and designated source builds. Any `$MODEL`,
`$REF_BUILD` or `$CAND_BUILD` commands supplied with these results are derived reproduction examples,
not original execution transcripts. The earlier CPU evidence remains frozen.

## Derived reproduction examples

Use clean source trees at the two recorded commits and explicitly configure each Release build
with `-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` before building
`kapitsa logit_dump ni_run`. Keep both imports separate through full file comparison. `$GGUF`
is the pinned input, and `$REF_MODEL`/`$CAND_MODEL` are the resulting separate model folders.
All commands that load a model run under an exclusive `$BENCH_GPU_LOCK`; the lock owner sets
`NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1` only inside that lock. Use `KAPITSA_PROFILE=precise`,
disable automatic updates, and isolate `KAPITSA_HOME` for each execution. Do not run arms concurrently.

```sh
"$REF_BUILD/logit_dump" "$REF_MODEL" output/reference \
  --paths step,verify,batch --windows 8 --rows 24 --chunk 2048 --batch 4 --batch-rows 2 \
  --corpus "$REF_SRC/bench/corpus.txt"
"$CAND_BUILD/logit_dump" "$CAND_MODEL" output/candidate \
  --paths step,verify,batch --windows 8 --rows 24 --chunk 2048 --batch 4 --batch-rows 2 \
  --from output/reference
python3 "$CAND_SRC/tools/logit_compare.py" output/reference output/candidate \
  --no-gate --json report-only.json
python3 "$CAND_SRC/tools/logit_compare.py" output/reference output/candidate \
  --min-top1 1 --max-mean-kl 0 --max-median-kl 0 --max-p99-kl 0 \
  --near-tie 0 --max-far-flips 0 --max-nll-change 0 --json strict-zero.json
```

Independently count finite values and compare complete logits, record and sequence bytes, including
prefill. The strict comparison tool does not apply decoding thresholds to prefill. Run native
`kapitsa bench --model "$MODEL" --quick --json --out "$NEW_LABEL.json"` in A/B/A/B order.
The fixed request driver is the existing
[`fixed_seed_driver.cpp`](../2026-11-08-q8-0-cpu/fixed_seed_driver.cpp), SHA256
`6b022a4791ac656f74ca84dc7b2090e7bfd34e5ad0e7a1c6bc1197e8adcd8310`.
It is compiled at `-std=c++20 -O3 -arch arm64` against the reference library and the unchanged
benchmark headers/helpers, then used for all four arms. Each arm has its own default precise server
with `--ctx 32768 --slots 1 --disk-cache 0`; invoke the driver as
`"$FIXED_DRIVER" "$PORT" "$MODEL_ID" "$OWN_SERVER_PID"`. The HTTP checks use separate servers
with the same settings except `--slots 2` and run
`python3 "$SRC/tests/test_server.py" "$SERVER_URL"`. These are methods for reproduction, not
verbatim execution logs; all outcomes, including environment qualification failures, must be retained.
