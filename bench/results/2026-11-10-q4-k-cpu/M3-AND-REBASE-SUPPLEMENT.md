# Later M3 validation and rebase

This additive supplement updates the initial README's pending chip/full-file coverage without changing any recorded run.

Apple M3 Ultra, macOS 26.4.1, Apple clang 17 (`clang-1700.6.4.2`), C++20 `-O3`: reference and candidate each pass 2,187,466 oracle/overlap cases plus guarded blocks. Trained Airy Q4_K A/B/A/B medians are **530.335 / 242.635 / 325.216 / 104.411 ns/row**. All samples, source and binary identities are in `M3-CPU-VALIDATION.json`. Every complete decode and all four final output files compare literally equal over 4,194,304 bytes; input, finite-value and guard checks also pass. The sample was extracted on M5 and transferred unchanged; this is CPU-only M3 evidence.

After #1134 and #1139 merged, the branch was rebased onto `2e308aeac6d6fe95801e7957996034d6bbfa6ac3`. The Q4_K decoder function, oracle header and earlier result files remain byte-identical. Both upstream test calls and the new Q4_K call are retained. `REBASE-CHECKS.json` records the fresh configure/build, IQ-layout, API, prefix-cache and lint results. Earlier benchmark source IDs remain their actual measured commits.

The complete pinned Flint GGUF now passes its published SHA256; see `FLINT-SOURCE-VERIFIED.json`. Complete-model import/output/runtime checks remain separate and pending. No inference-speed claim.
