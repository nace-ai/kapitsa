# IQ4_XS full-model validation

These JSON files are explicitly redacted, derived summaries, not original execution logs.
Original commands, logs and archives remain private. The fixed-request C++ driver is an exact
copy of the executed source: SHA256 `6b022a4791ac656f74ca84dc7b2090e7bfd34e5ad0e7a1c6bc1197e8adcd8310`.
Its raw output includes diagnostic fields that are omitted from the public summaries.

Production/test source: `d5589c4259b35a3bd5d5cc711b20975705118956`.
Reference: `b1be7d4f644b3e58d56499404b38b46cf2d2a5ef`.
This evidence addition does not change production or test code. Prior CPU evidence remains
unchanged in `../2026-11-09-iq4-xs-cpu/`.

## Setups and model

- M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1; Apple clang 17 (`clang-1700.6.4.2`).
- M5 Pro, 20 GPU cores, 48 GiB, macOS 26.6.2; Apple clang 17 (`clang-1700.0.13.5`).

The `LOCAL-*` files are the subsequent M5 execution, verified against system chip/OS queries
and both arms' runtime metadata. Earlier M4 Pro CPU measurements are a separate setup.
Own CMake Release targets use `-O3`, arm64; actual target flags have no `NDEBUG`.
Binary hashes are recorded per setup and arm. All model stages held the exclusive GPU lock.

Model: `bartowski/Qwen_Qwen3.5-4B-GGUF@4168f45a16a1290d65a4ec0fa312ae917a4c15d6`,
`Qwen_Qwen3.5-4B-IQ4_XS.gguf`, SHA256
`628124c61a806897d3f28dcd416be6daff94f62f731eb900d4381875eac9e412`.

M3 independently imported both arms with `--no-drafter`: all nine output files, including
manifest, were compared literally and match. The measured trunk uses affine 5-bit/group-64
after import; other modules' format differences are retained. The M5 arms share one transferred
M3 reference import, with file sizes and SHA256 checked. No independent M5 import was executed.

## Numerical and request checks

On each setup, each arm produced 584 rows: 8 prefill and 192 each of step, verify and batch,
vocabulary 248320. All 145018880 float32 values per arm are finite. Complete output files and
recorded inputs match literally. Report-only and strict-zero comparisons exit 0. The tool's
strict thresholds apply to decoding paths; prefill has a separate finite/literal-byte check.
Verify is a teacher-forced KFlash tree, a chain of eight live rows, not free MTP generation.
No noise floor was changed.

The native A/B/A/B files retain every run, return code, flag and invalid result. M3 candidate
B1/B2 have repetition flags; M5 candidate B2 has one. M5 B1's sampled and greedy medians are
lower than A1. Native requests have different nonces/histories, so these are not matched-input
performance comparisons and do not establish a cause.

The subsequent fixed driver uses identical decode requests, seeds 1/2/3, sampled and greedy,
512 tokens each, in A/B/A/B order: 24 requests per setup. Each of the six cases has the same
request SHA256 and complete-content SHA256 across its four legs. Length/nonempty/rate/repetition,
quiet and warmup gates pass. Adjacent-pair middle-window rate changes are -0.00744..+1.45270%
on M3 and -0.87333..+1.93136% on M5. These are not overall inference-speed claims.
The full per-run measurements and rate method are in the fixed JSON files.

Prefill/cache text was not byte-compared. The third cached turn ends at its length limit in
every fixed leg; different generated histories are not matched-input cache evidence.
The decode content-hash gate does not cover these prefill/cache responses.

HTTP: M3 reference and candidate each return 1 with 54 PASS/1 FAIL, the same queued-stream
keepalive assertion (one keepalive before the first token, instead of two required). Its
cause remains unresolved. M5 reference and candidate each return 0 with 55 PASS/0 FAIL.
These results do not establish universal API/cache behavior. Earlier raw-U8 load-error CPU
failures are retained in the prior evidence; that test path does not call IQ4_XS decoding.

## Derived command examples

These are relative/placeholder examples derived from the executed commands, not raw logs.
`$REF_BUILD` and `$CAND_BUILD` refer to the corresponding recorded builds; `$MODEL` is the
verified import. The caller holds the exclusive lock and sets both held markers for model
tools. Run the arms sequentially, never concurrently.

```sh
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1
"$REF_BUILD/logit_dump" "$MODEL" output/reference --paths step,verify,batch \
  --windows 8 --rows 24 --chunk 2048 --batch 4 --batch-rows 2 --corpus bench/corpus.txt
"$CAND_BUILD/logit_dump" "$MODEL" output/candidate --paths step,verify,batch \
  --windows 8 --rows 24 --chunk 2048 --batch 4 --batch-rows 2 --from output/reference
python3 tools/logit_compare.py output/reference output/candidate --min-top1 1 \
  --max-mean-kl 0 --max-median-kl 0 --max-p99-kl 0 --near-tie 0 \
  --max-far-flips 0 --max-nll-change 0 --json strict-zero.json
```

The supplied fixed driver is linked to the reference Release library with C++20, `-O3`, arm64,
macOS14 minimum and the engine's framework dependencies. Each arm uses its own server binary
with `serve --model "$MODEL" --model-id iq4-xs-review --private --host "$LOOPBACK_HOST" --port "$PORT"
--ctx 32768 --slots 1 --disk-cache 0`. Invoke `fixed-seed-driver "$PORT" iq4-xs-review "$SERVER_PID"`.
The fixed driver exercises the existing bench request builders and preserves the actual seed
in each request. HTTP checks use the arm's `tests/test_server.py` against a two-slot server.

M3 sanitizer, non-AArch64 execution, independent M5 import and universal API/cache behavior
were not verified. The first M3 import stopped at a pre-conversion staging guard; a subsequent
fixed-driver compilation failed creating temporary files before any fixed requests. Those
setup failures remain recorded and were recovered without repeating completed model phases.
