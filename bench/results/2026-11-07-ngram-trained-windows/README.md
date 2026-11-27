# Remote trained-weight window validation

Runs: Apple M3 Ultra,96GiB, macOS26.4.1 (25E253). Complete separately configured Release checkouts: reference `934a6d72f2a393b6987baf0b98ebe5a9e07d69be`, candidate `dd166319cc746fd439beb77119416e610f9acf63`. Configure logs, runtime version strings, Git identities and exact commands are included. These are this PR's own builds and outputs.

Model source: `beamster/Qwen3.8-Flash-Next-Sushi-2bpw`, pinned revision `f37e7ba46f0f2993b6ca373221530c579f258712`. All214non-table files totaling37,575,753,844bytes were downloaded and checked against their pinned source digests. The affine Q4/group32 n-gram table is an exact-length sparse file:32,000,153,976logical bytes. Only the required real rows were fetched. This is not a complete checkpoint or coverage for arbitrary requests.

The eight exact corpus windows require83,302unique n-gram rows. Each HTTP range was checked for its exact Content-Range and total table length; every required row's bytes were rechecked before the two unmodified model runs. `corpus-base.json` and `corpus-candidate.json` contain the input tokens and hash indices; the driver checked that both model runs used exactly those sequences. The full table mapping's bounds are valid, and every row consumed by these fixed windows is populated. The source and row preparation helpers are included.

| Output path | Full-vocabulary rows | Bytes per arm | Result |
| --- | ---: | ---: | --- |
| Prefill | 8 | 7,946,240 | Byte equal, finite, nonzero |
| Step | 192 | 190,709,760 | Byte equal, finite, nonzero |
| Verify | 192 | 190,709,760 | Byte equal, finite, nonzero |

All392rows are byte equal. The unchanged comparator exits0; there is no calibrated floor for this exact model, so this is an exact-byte comparison, not a calibrated-floor claim. Full output digests, nonzero counts and sequence/record manifests are retained. GPU work held the shared exclusive lock and began after the other holder released it. No timing or inference-speed claim is made from these runs.

The same two complete libraries also independently decoded the83,302corpus rows on CPU:53,313,280bytes per arm, byte equal, finite, with intact destination guards. That earlier completed CPU evidence is included; it was not rerun for this publication. Corpus rows may overlap other fixtures and must not be counted as disjoint coverage.

Still running at publication: a separate driver fills each new greedy-input row from the same pinned source before allowing it to be consumed. Four subsequent unmodified256-token A/B/A/B runs have not completed and are not claimed here. Their full coverage and decoded-output checks must be reviewed separately. The published window checks do not validate arbitrary HTTP requests, native benchmark requests, MTP/drafter-generated inputs, long-context/streamed-expert configurations or a complete n-gram table. The PR remains Draft pending the remaining validation.
