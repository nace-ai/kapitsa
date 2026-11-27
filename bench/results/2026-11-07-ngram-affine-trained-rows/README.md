# Affine Q4 trained-table CPU row checks

Runs: remote Apple M3 Ultra, 96 GiB, macOS 26.4.1. Reference `01b4c6d69fbe69e862c00358f88c94a363129592`; implementation `8c0a7ec3bf72623f6c7f92524ef229f3643a9e67`. The existing independently configured/built complete libraries from this PR were reused; their SHA256 values and the newly compiled probe identities are in `result.json`.

Actual table: `beamster/Qwen3.8-Flash-Next-Sushi-2bpw` pinned revision `f37e7ba46f0f2993b6ca373221530c579f258712`, affine Q4/group32, row dimension160. This is a 32,000,153,976-byte logical sparse table, not a complete checkpoint download. Required row payloads were fetched from the pinned source with exact Content-Range checks and row-byte rechecks. Shared preparation/coverage metadata is included; the decoding outputs below were freshly produced by this PR's own reference and candidate libraries. No other PR's numerical output is counted.

| Input row set | Rows | FP32 output bytes per variant | Result |
| --- | ---: | ---: | --- |
| Fixed corpus windows | 83,302 | 53,313,280 | Complete byte equality; finite; destination guards intact |
| Fixed chat prefix | 640 | 409,600 | Complete byte equality; finite; destination guards intact |

The row sets can overlap; they are reported separately rather than claimed as disjoint coverage. The corpus output contains12,427,213 nonzero values; the chat output contains95,362. Each output has dimension160. SHA256 values for both outputs and both source libraries are recorded separately in `result.json`; raw arrays were removed after equality and digest checks. All per-arm logs and executed commands remain.

`run.py` and `probe.cpp` preserve the executed harness, using its original remote paths. The runner compiles the same probe against each existing full library using that build's recorded compiler/link flags, calls `NgramTable::row` for the covered row IDs, checks finite/nonzero values and output guards, and compares all output bytes. To reproduce elsewhere, supply the pinned table with verified coverage, the two source builds and matching input row sets, then adapt the paths. The source-level NEON change is not modified here.

This is CPU correctness coverage on trained table values, not trained-model GPU coverage, a row/gather timing result, or an inference speedup. Concurrent model downloads were active; no timing is claimed. Full trained-model logits/continuations, controlled row/gather timing, paired HTTP and native benchmarks remain pending. Existing synthetic GPU and arithmetic evidence is frozen separately. The PR remains Draft.
