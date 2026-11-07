# Remote CPU and synthetic Flash-Next integration

Runs: M3 Ultra, 96 GiB, macOS 26.4.1 (25E253), Apple Clang 17, Release `-O3`.
Complete reference checkout `01b4c6d69fbe69e862c00358f88c94a363129592` versus complete implementation checkout `8c0a7ec3bf72623f6c7f92524ef229f3643a9e67`. Both were freshly configured with their actual source commit as `KAPITSA_GIT`; version strings and library/binary hashes are in `evidence/synthetic/identity.json`. The later PR commits contain evidence only.

## Results

Both complete builds and `fn_store_test` pass. Candidate's 21 `vk cpu-tests` entries pass, as do the supplemental tests and lint in `build-cpu.sh`. The additional device-profile executable and profile/comment gate checks pass in `evidence/followup-1730/`; the same folder records scratch-state cleanup. These results belong to this PR's own builds.

A fresh `fn_synth.py --seed 1` pack uses random weights, six layers, hidden width512, vocabulary2048 and an8214x64 Q4 group32 n-gram table. The entire synthetic model is populated. `make_table_variants.py` retains identical affine payloads in sharded and `.bin` containers; a third `.bin` table contains different deterministic BF16 values. Manifests and SHA256s are retained. The conversion recipe was also used for #1019, but all executions, builds and result files here are newly produced for this PR.

Under one exclusive `/tmp/kapitsa-gpu.lock`, each table form ran prefill+step and token-by-token (`--tbt`) in A/B/A/B order: **24 runs**, each producing **64 identical generated token IDs** within its comparison group. The actual dump sizes are **16 prompt rows** for prefill+step and **64 rows** for `--tbt`; the former is not a complete dump of its generated-token logits. Every retained dump hash matches its comparison arms; checked floats are finite and nonzero. Random weights generate repetitive synthetic IDs; this does not establish natural-language quality or useful throughput.

Separately, `logit_dump --paths step,verify --from synthetic-fixed-windows` compared **3 prefill,72 step and72 verify rows per layout** (441 total full-vocabulary rows), all byte equal and finite/nonzero. It used three48-token sequences, with step/verify record positions17–40. The affine layouts agree with each other. The changed BF16 payload changes the logits, confirming table values affect the graph. There is no calibrated model floor for this synthetic graph; this reports exact bytes without inventing one.

`evidence/synthetic/commands.json` records every command, start time, exit status and elapsed wall time;32 commands exited0. Raw logs include engine-reported rates but they are **not controlled performance measurements**. Actual-checkpoint downloads ran concurrently. Raw logits were removed only after hashes, descriptors, metadata and equality checks were recorded. The GPU lock was released after the final check.

## Reproduce

The committed `synthetic-gpu.py` is the exact executed harness (its SHA256 is recorded), and `build-cpu.sh` records the exact remote build/test recipe, including its original absolute paths. The scripts refuse to overwrite existing result logs. Use a new writable run folder; do not execute in this frozen results directory.

1. In that folder, prepare complete clean Git checkouts called `base` at `01b4c6d69fbe69e862c00358f88c94a363129592` and `candidate` at `8c0a7ec3bf72623f6c7f92524ef229f3643a9e67`.
2. Follow `build-cpu.sh`, changing only its run-root and installed tool/Python PATHs. Explicitly reconfigure each build with its matching `KAPITSA_GIT`; build the CLI, ni_run, logit_dump, all CPU targets and systemone. Run the full candidate CPU/supplemental/lint checks. That recipe creates `cpu-passed` only after success. Also run the three follow-up CPU checks listed above.
3. Copy `synthetic-gpu.py` and `make_table_variants.py` to the run root and run `python3 -u synthetic-gpu.py` with NumPy installed and model/GPU permission. It generates its own model and fixed windows, checks exact checkout identities and waits for the shared GPU lock. All32 commands and byte/ID checks run before `synthetic-gpu-passed` is written.
4. Inspect `generation-results.json`, `teacher-forced-results.json`, `controls.json`, `identity.json` and all exit codes. Clean the candidate's `vk` scratch state after completion.

## Not verified

Real trained-model row/gather timing, trained GPU outputs or natural-language continuations, HTTP and native benchmarks remain pending. No inference speedup is established. Synthetic GPU coverage is on the M3 Ultra only; earlier arithmetic-only checks also ran on the M4 Pro. The PR remains Draft. The actual-weight download and covered-row preparation for subsequent independent tests are recorded separately; these tests did not use the sparse trained table.
