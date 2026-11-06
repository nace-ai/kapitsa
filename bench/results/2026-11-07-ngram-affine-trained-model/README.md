# Trained affine rows, gather and model outputs

Runs: Apple M3 Ultra, 96 GiB, macOS 26.4.1 (25E253). Own complete Release reference `01b4c6d69fbe69e862c00358f88c94a363129592` and candidate `8c0a7ec3bf72623f6c7f92524ef229f3643a9e67`. Both were explicitly configured with matching version labels. Exact library/binary identities, compile/link commands and run commands are retained in `results.json`; the full-library gather probe and original orchestration are included. Source is unchanged.

Pinned model: `beamster/Qwen3.8-Flash-Next-Sushi-2bpw` revision `f37e7ba46f0f2993b6ca373221530c579f258712`. Shared preparation supplies actual model bytes and explicitly covered row inputs only. This PR's model outputs, row/gather outputs and reference comparisons were all computed independently by its own binaries. The exact-length 32,000,153,976-byte table remains sparse and is not an arbitrary-request checkpoint.

Each probe arm measures the complete `row()` loop and threaded `gather()` over the fixed covered inputs: 16 heads, dimension 160, token counts 1/16/512/1536. There are 12 samples for each case; the table is the median of the final 10. All individual samples remain in the raw logs. Timing excludes initial model/table setup but includes row or gather dispatch. Both modes write all results to guarded output buffers; all four 42,291,200-byte collected outputs are identical, finite and nonzero, with guards intact.

| Mode | Tokens | A0 µs | B1 µs | A2 µs | B3 µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| row-loop | 1 | 8.205312 | 1.020728 | 1.809375 | 0.304375 |
| gather | 1 | 12.822707 | 14.404480 | 8.365625 | 6.848855 |
| row-loop | 16 | 34.141662 | 7.170312 | 30.596875 | 4.661463 |
| gather | 16 | 289.270838 | 230.774488 | 231.858338 | 232.430725 |
| row-loop | 512 | 1104.833250 | 279.302250 | 1112.999750 | 279.239750 |
| gather | 512 | 6691.718750 | 4996.947750 | 5149.572750 | 4992.864500 |
| row-loop | 1536 | 3529.583500 | 968.562500 | 3520.875000 | 1172.125000 |
| gather | 1536 | 19905.125000 | 14894.479500 | 15200.375000 | 15377.937500 |

The row loop improves in both adjacent pairs. Complete gather does not show a uniform win: the first A arm has visible warm-up/drift, 16-token gather is essentially unchanged in the second pair, and 1536-token gather is slightly slower there. CPU frequency/thermal state and per-process contention were not recorded. The exclusive GPU lock prevents another conforming GPU benchmark from overlapping but does not establish idle CPUs. This series does not establish a stable complete-gather or inference-throughput speedup.

The independent trained-model comparison passes for 392/392 full-vocabulary rows: 8 prefill, 192 step and 192 verify. Every row is byte identical, finite and nonzero. The comparator exits zero; it reports no calibrated floor for this exact model. Before accepting the outputs, both runs' recorded sequences were compared to the prepared corpus and all 83,302 required table rows were rechecked.

Four further unmodified reference/candidate/reference/candidate `ni_run` arms each generate 256 IDs, no EOS and nonempty decoded text, using the same fixed prompt and `--dflash none`. Every ID and each 254,279,680-byte complete logit dump match within this PR's own comparison. Each run's inputs are checked against the 3,655 prepared row set. Full output hashes, IDs, decoded text and logs are retained; raw float dumps were removed only after successful comparisons. This is fixed greedy generation, not unrestricted MTP/drafter generation or language-quality evaluation.

Not verified: a complete checkpoint, arbitrary HTTP requests, native interleaved `kapitsa bench` and a second trained-model/chip setup for the targeted class. The remaining table allocation exceeds current remote free disk space. No default or noise floor changed for this verification. Earlier published evidence is frozen; the PR remains Draft.
