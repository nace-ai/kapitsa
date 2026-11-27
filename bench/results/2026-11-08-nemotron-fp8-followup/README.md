# FP8 model follow-up checks

These are explicitly derived privacy-filtered exports, not raw execution logs or raw native benchmark JSON. Original evidence remains privately verifiable. Reference source is `052104f8f9c99edde129415a7384a1a58dc24ae6`; candidate production source is `8f3996fea6f1f44537bee4e789b98cd6a531f347`.

The affected checkpoint is `nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8@3fe6dab75665a93884214ad4b1b95cf02717d081`.

## M3 full-model output

M3 Ultra, 60 GPU cores, 96 GiB memory capacity, macOS 26.4.1, Apple clang 17. Each arm has 584 rows (8 prefill and 192 each step, teacher-forced verify and batch records), or 76,546,048 float32 elements. Every element is finite, complete bytes and inputs match, and report-only, explicit zero-drift and literal full-file checks passed. No floor was modified. The original source bundle's two missing embedded benchmark assets were recovered from the exact commits before explicit Release builds; each original archived source file and each added asset was verified. Downloads and model outputs were not repeated. Source/binary identities and model digests are in `m3-output-summary.json`.

## M4 fixed-request comparison

The native M4 A1/B1/A2/B2 results remain separately published, including B1's real 13-token sampled stop and exit code 1. This follow-up addresses randomized native requests using predetermined nonzero seeds 1/2/3 and fixed request nonces. It reuses the benchmark request, SSE, rate and output checks with each source's own default server, under the exclusive GPU lock. It is a custom method and does not replace the native cohort.

All 24 decode records (six matched cases across four legs) generated 512 tokens, finished by length and passed output/rate/repetition gates. Request SHA256 and complete generated-content SHA256 match across all four legs for every case: 12,288 tokens total. Prefill/cached rate checks also passed; their textual answers were not byte-compared. The separate model-output checks provide the logit-byte evidence.

Paired decode changes range from -1.052% to +0.366%; cold 4K TTFT is approximately 4.331–4.333 seconds. These observations establish no inference speedup or general absence of regression. Every recorded row is preserved in `m4-fixed-request-summary.json`.

M3 default HTTP completed with reference 40 PASS/8 FAIL and candidate 37 PASS/11 FAIL, both rc1, with differing assertions and outputs. `m3-http-summary.json` preserves filtered failure labels and original private-log hashes. The cause is unresolved, so this is not a shared-failure or all-pass claim.

M3 native A1/B1/A2/B2 returned rc0/empty invalid lists, but every leg failed all six tool-call semantic checks; B1/A2/B2 also have cached first-turn length warnings. `m3-native-summary.json` retains every leg, including those failures. Process rc0 does not qualify the API or performance gates. Controlled diagnosis follows separately and does not overwrite these runs. CPU packing latency measurements remain separate evidence and measure the packing function rather than full checkpoint loading or inference.

## Compiler metadata and reproduction

`compile-metadata-clarification.json` distinguishes cached Release defaults from actual generated target flags. Earlier M4 summary `cxx_flags` values were the CMake cache default `-O3 -DNDEBUG`; actual target builds use `-O3` without `NDEBUG`. Binary identities and measurements are unchanged; the earlier snapshots remain available.

`portable-reproduction/` contains a clearly labeled source adaptation of the fixed-request client. It keeps request/SSE/rate helpers, adds mandatory caller-held GPU-lock guards, and omits machine probes. Its compile/link and no-lock rejection checks passed without model/GPU repeats. It does not itself establish a new performance qualification.
