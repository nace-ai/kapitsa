# Portable fixed-request and seed reproduction

This is an adaptation of the measured API client, not the original worker or a raw execution log. The original measured client source SHA256 is `326eeb19c2454735723fd30437a319181f1ba1febe3f0af28cf21bf247146571`. The portable client omits worker lifecycle and machine condition probes. The published measured results retain those checks internally; a new performance qualification must also check an idle GPU and the repository's phase gates.

From the matching Kapitsa checkout, build its Release CLI/library and create `results/` and compile the client using relative repository paths (`$REPRO_DIR` is this evidence directory’s `portable-reproduction/` folder):

```sh
c++ -std=c++20 -O3 -mmacosx-version-min=14.0 -Isrc -Ithird_party "$REPRO_DIR/portable_fixed_seed.cpp" build/libnaceinfer.a -framework Metal -framework Foundation -framework CoreFoundation -framework IOSurface -framework IOKit -framework ImageIO -framework CoreGraphics -framework Security -framework SystemConfiguration -framework Accelerate -o results/portable-fixed-seed
```

Start each arm's own `kapitsa serve` using the pinned FP8 model, model ID `fp8-review`, precise profile, 32768-token context, one slot and disk cache disabled. Hold the repository's exclusive GPU lock across the server and client, and propagate `NI_LOCK_HELD=1` and `NACE_GPU_LOCK_HELD=1` to both. The portable client and wrapper reject a missing caller-held marker; the caller must actually hold the lock. Use a fresh server and scratch home for each arm. Create a new results directory first. After server readiness, run:

```sh
python3 "$REPRO_DIR/run_fixed_seed.py" --server-url "$SERVER_URL" --model-id fp8-review --driver results/portable-fixed-seed --out results/fixed-A1.json
```

Repeat in A1/B1/A2/B2 order with the recorded reference/candidate source versions. All labels must be new; the wrapper rejects an existing output. The client uses the repository's exact request, SSE, middle-window rate and repetition helpers, with seeds 1/2/3 and fixed nonces declared before testing, for three sampled and three greedy 512-token requests. It also runs a fixed-nonce greedy 4K prefill and three cached follow-ups.

Compare each matched case's request SHA256, generated-content SHA256, nonempty content, 512 completion tokens, length finish, distinct-4 >= 0.5, flags and invalid entries across all four arms. Check prefill and cached flags too. These text digests apply only to the six decode cases; the separate logit-dump evidence checks complete numerical bytes. Retain failures and the original unseeded native cohort, including its short B1 output. Do not choose new seeds until a test passes or claim that this custom cohort was produced by `kapitsa bench`.
