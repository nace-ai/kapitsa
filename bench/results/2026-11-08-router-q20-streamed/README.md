# Flash-Next Q2_0 streamed-router output check

This adds completed non-EXL3 model evidence for the regular and next-layer prediction router call sites in PR #1056. Production source remains `dc82cf0f39b942157e86b0cd5e518e5caed17e61`; reference is `f22effb9bdda7fa17be5e4ce614383a35e1cc62a`. Each source was explicitly configured and built in CMake Release (`-O3`), and the actual binary SHA256s match the previously recorded builds.

Runs: Apple M3 Ultra, 60 GPU cores, 96 GiB memory capacity, macOS 26.4.1 (25E253), Apple clang 17. Both model runs held the exclusive GPU lock, with the caller-held markers set. Both returned 0.

The model combines the pinned Nace dense/MTP pack and ISTA Q2_0 routed experts listed in `summary.json`, with the complete compatible Sushi trained n-gram table. The full table and tokenizer digests are recorded. The table is not asserted byte-equivalent to the original Nace table. Sparse converter staging was removed before loading. Conversion completed all 51 trunk/MTP payloads. A separate source-plane check passed 288 seeded expert draws with replacement across 48 trunk layers; it did not check every expert or MTP expert planes.

`NI_EXPERT_STORE=4 NI_ROWS_OLD=0` deliberately selected an incomplete expert store: the model load confirmed 2,872 of 24,576 experts available in that store. This exercises both the non-EXL3 regular and next-layer prediction branches during three-row trunk verification. Store sizes in the summary describe the test settings, not the machine's memory usage.

| Path | Rows per arm | Finite rows per arm | Bitwise-identical rows | Top-1 flips | Mean KL | Max absolute difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill | 8 | 8 | 8 | 0 | 0 | 0 |
| step | 192 | 192 | 192 | 0 | 0 | 0 |
| three-row trunk verify | 192 | 192 | 192 | 0 | 0 | 0 |

Both arms used the same 8 windows and 14,848 prefilled tokens, with chunk 2,048 and 24 recorded rows per window. The candidate loaded the reference inputs via `--from`; sequence and record bytes match. All 97,341,440 float32 elements in each arm are finite. Full raw-file byte comparisons and SHA256s agree, not only a diagnostic hash.

The original `logit_compare.py` report says every row is bitwise identical, but is report-only because no model noise floor matches this hybrid identity. It remains preserved. An additional comparison of those same files passed explicit zero-drift flags for step and verify (top-1 1.0, all KL/NLL/flip limits 0); no floor was created or changed. The prefill result is independently checked for complete finite/bitwise equality; its tool metadata has no separate prefill bounds. This is an explicit zero-drift check, not a measured noise-floor gate.

The three-row verify path uses recorded input tokens and invokes trunk verification. Loading the MTP head and using `--spec 2` here do not run free-generating MTP drafting, sampling or acceptance. Q2_0 MTP continuations, acceptance rates, interleaved inference performance and an M4 full-model repetition remain unverified. The elapsed times of these sequential output dumps are not speed evidence.

## Reproduction examples

These are path-placeholder examples of the recorded method, not verbatim original execution logs. `$MODEL` identifies the pinned hybrid model above; `$REF_BUILD` and `$CAND_BUILD` are each source's Release build. Run the two model commands sequentially. Each `logit_dump` takes the exclusive GPU lock itself; these examples do not set caller-held markers:

```bash
NI_EXPERT_STORE=4 NI_ROWS_OLD=0 NI_LOG=1 \
  "$REF_BUILD/logit_dump" "$MODEL" output/reference --paths step,verify --spec 2 \
  --windows 8 --rows 24 --chunk 2048 --corpus bench/corpus.txt
NI_EXPERT_STORE=4 NI_ROWS_OLD=0 NI_LOG=1 \
  "$CAND_BUILD/logit_dump" "$MODEL" output/candidate --paths step,verify --spec 2 \
  --windows 8 --rows 24 --chunk 2048 --from output/reference
python3 tools/logit_compare.py output/reference output/candidate --json report-only.json
python3 tools/logit_compare.py output/reference output/candidate \
  --min-top1 1 --max-mean-kl 0 --max-median-kl 0 --max-p99-kl 0 \
  --near-tie 0 --max-far-flips 0 --max-nll-change 0 --json strict-zero.json
```

This publication is a privacy-filtered derived summary and report, not a raw execution archive. Original commands, load logs, raw metadata and output files remain privately verifiable. `summary.json` preserves source/binary identities, model digests, input/output hashes, command exit codes and complete-row finite checks. Existing published result files are unchanged.
