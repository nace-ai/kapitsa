# M3 fixed-request and controlled API follow-up

These are derived privacy-filtered summaries, not raw execution logs or native `kapitsa bench` JSON. Original requests, complete responses, traces and source/binary records remain privately verifiable. Reference source is `052104f8f9c99edde129415a7384a1a58dc24ae6`; candidate production source is `8f3996fea6f1f44537bee4e789b98cd6a531f347`. Runs used M3 Ultra, 60 GPU cores, 96 GiB memory capacity, macOS 26.4.1, Apple clang 17, and `nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8@3fe6dab75665a93884214ad4b1b95cf02717d081`.

## Fixed default A/B/A/B

The predeclared six decode cases use seeds 1/2/3 with fixed request nonces: three sampled and three greedy 512-token requests. Each arm uses its own precise server, 32768-token context, one slot and disk cache disabled, under the exclusive GPU lock. The unchanged measured API client uses Kapitsa's request/SSE/rate/repetition helpers. It is a custom cohort and does not replace the native measurements.

All 24 decode records are nonempty, generate 512 tokens, finish by length and pass rate/repetition gates with empty row flags and invalid lists. Each of the six matched cases has identical request and complete generated-content SHA256 across all four legs. All four processes return rc0; quiet and cold-prefill gates pass. Paired decode changes range from -1.6855% to +0.2288%; cold 4K TTFT is 1.8699–1.8758 s. This establishes no inference speedup or universal absence of regression. Every row is in `fixed-request-summary.json`.

Cached follow-ups have empty invalid lists and actual cache reuse, but B1/A2/B2 retain the first-turn length warning, including the unchanged reference A2. Cached flags are not all clear. Different generated histories are not equal-input cached performance evidence. The content hashes cover the six decode cases per leg; prefill/cached textual answers were not byte-compared. The separate 584-row numerical audit checks full logit bytes.

The portable source adaptation and reproduction instructions are in `../2026-11-08-nemotron-fp8-followup/portable-reproduction/`. They omit machine probes and require the caller to hold the GPU lock and propagate its markers. No GPU run of that portable adaptation is claimed.

## Controlled API diagnosis

The original default HTTP suites remain reference 40 PASS/8 FAIL and candidate 37 PASS/11 FAIL, both rc1. All four original native legs retain failed tool semantics; their rc0 does not waive those failures. No original run is overwritten.

The source-derived failed fixtures add declared Chat seed 1; Anthropic/Responses use source greedy settings without assuming seed support. Synthetic tool-result histories and new-prefix probes are explicit adaptations. Each arm runs the default source GET sequence, an independent first-thinking server and a shared-system-prefix-checkpoint-disabled diagnostic. Proposed tool calls are captured as model output and never executed.

All 12 diagnostic legs retain rc1. The full user text is retained by the pinned template, and nonzero seeds reach the actual sampler. Across 19 sequential generation fixtures, unchanged reference A1/A2 match generated semantics in 4/19 cases and unchanged candidate B1/B2 in 3/19. Independent first-thinking also differs within both unchanged sources. The full 26-record comparisons additionally contain two GETs and five concurrency/disconnect probes; the abandoned stream is intentionally partial. `controlled-api-summary.json` preserves every diagnostic record and its failures with explicit denominators.

This reproduces baseline API instability without establishing a row-packer regression. Its runtime/path/logit cause remains unresolved. Disabling the shared prefix checkpoint is diagnosis only: it neither disables all reuse nor qualifies the default. No API, kernel or default code changed in this follow-up.
