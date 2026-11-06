# KFlash packed-output follow-up: generation and HTTP

Runs: Apple M4 Pro, 20 GPU cores, 48 GiB, macOS 15.7.1. Actual Qwen3.5-4B 4-bit/group64 with its matching six-layer v1 DFlash2 drafter. Reference `2fe7c3fa`; implementation `0be92def`; checkout head at the follow-up `a263df92` only adds evidence/documentation after the implementation. Each checkout was explicitly reconfigured and rebuilt. The candidate CMake cache explicitly pins `KAPITSA_GIT=0be92def`, which matches the unchanged implementation sources; `identity.json` records the checkout heads, actual binary version strings, binary/library-related source hashes and the drafter/tokenizer digests. No version label is evidence for an unrecorded source change.

## Fixed nonempty speculative continuations

The same 63-token chat-templated compiler-tutorial prompt, temperature 0 and 256 output tokens was run in A/B/A/B order with fresh per-arm homes and the project GPU lock. All four output-ID arrays are identical, every decoded output is nonempty, and every arm has zero configured EOS/end tokens in its 256 output IDs. Logs show 39 packed static draft projections and the actual DFlash2 v1 drafter load. All four runs used 87 cycles with the same acceptance histogram `[15, 19, 28, 15, 6, 0, 2, 2]`. The decode line reports 255 post-prefill tokens; the saved ID array includes the initial prefill sample, for 256 total. The histogram records accepted proposals, including any tail truncated by the requested output limit; it is not a claim about unrestricted acceptance throughput.

All four transcripts, decoded text files, output IDs, commands and half-second GPU contention samples are retained. First A and B were flagged for foreign GPU work (mean/max 5.65/12.52% and 10.99/54.02%); later arms measured 1.30/1.79% and 3.40/6.64%. These runs establish the stated output/acceptance checks, not an inference speedup.

## Paired HTTP regression

Fresh reference and candidate servers used two slots, 32768 context and disk cache disabled, under the same GPU lock. Each ran its repository `tests/test_server.py`: **48 PASS and 1 FAIL**, exit 1 for both. The sole shared failing test is the busy-slot streaming keepalive timing assertion. It requires exactly two keepalives 15 seconds apart; these runs observed one near 15.06 seconds and a first token at 22.64/22.81 seconds. Full details are in `http.json` and the raw logs. This is matching observed baseline behavior, **not an all-pass HTTP result**. No assertion or production behavior was changed to hide it.

## Reproduction and scope

`run.py` and `tokenizer.cpp` are the exact executed orchestration and CPU tokenizer helper. The script uses the original local paths and also contains the subsequent native benchmark stage; adapt its tree/model/output paths when reproducing elsewhere. Per-command JSON files preserve the actual commands. Its original first native benchmark attempt was interrupted after warm-up with no result JSON and is not represented as a completed measurement here; native A/B/A/B is being resumed under new labels. These files are the completed continuation and HTTP stages only.

Previously published reducer, draft-buffer, target-logit and CPU checks remain in the separate initial evidence folder. No tests from another PR count here. Full-model fallback checks on 27B/35B-A3B/Flash-Next, another packed-profile setup, controlled long-input graph timings and completed native benchmarks remain outstanding; the PR stays Draft.
