# Rebase validation after the n-gram admission fix

Reference: `4d1d744538063a17bc85e88a08c08c8061511c95`; implementation: `6bd1dd1c725f72385f717b467860dab43d9ea0cb`. The rebase retains the merged admission tests and this change's arithmetic regression. `rebase-source-equivalence.json` verifies that prefetch, row and gather source text, and the arithmetic regression, are unchanged from implementation `8c0a7ec3`. This is a source comparison, not a claim of identical binaries.

Runs: remote Mac Studio, Apple M3 Ultra (60 GPU cores), 96 GiB, macOS 26.4.1 (25E253), Apple Clang 17, Release `-O3`. Both fresh worktrees were explicitly configured before building. Version output records `kapitsa 0.2.31 (4d1d7445)` and `kapitsa 0.2.31 (6bd1dd1c)`; source trees, model executables and libraries have independent recorded identities.

## Build and CPU

Both complete Release CLI/library builds and `fn_store_test` pass. The baseline and candidate each reject 26 invalid mapped-table fixtures and accept three valid formats. The candidate additionally checks 1,449,984 bitwise arithmetic values with output guards. The candidate's complete 21-command CPU suite reports `0 failed`; supplemental execution/session, HTTP stream, drafter attach, layout, config/JSON, affine pack, System One, protocol/runtime, device-profile, converter, profile/comment gates and lint checks pass. Both worktrees' verify scratch states were cleaned. Exact build/test commands are in `rebase-build.sh`, with all logs under `cpu/`.

## Trained model outputs

Pinned model: `beamster/Qwen3.8-Flash-Next-Sushi-2bpw` revision `f37e7ba46f0f2993b6ca373221530c579f258712`. Existing verified non-table weights and the sparse table fixture are reused as inputs. Every one of the 83,302 corpus rows and 3,655 fixed-generation rows is checked against source bytes before accepting these runs. No full-table download or RAM-table reconstruction was repeated. Unpopulated table rows do not establish coverage for arbitrary prompts.

Both rebased complete libraries produce fresh outputs under the exclusive GPU lock. The 392 full-vocabulary rows (8 prefill, 192 step, 192 verify) are finite/nonzero and byte-identical. Comparator output:

```text
prefill: 8 records (model-sushi)
  reference: 0.2.31 (4d1d7445), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (6bd1dd1c), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
step: 192 records (model-sushi)
  reference: 0.2.31 (4d1d7445), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (6bd1dd1c), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
verify: 192 records (model-sushi)
  reference: 0.2.31 (4d1d7445), verify "MTP cycle's chain of 3 rows", prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (6bd1dd1c), verify "MTP cycle's chain of 3 rows", prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
no noise floor for this model and no bounds given: reported only (tools/logit_floor.py measures one)

```

No calibrated noise floor exists for this exact model; the numerical evidence here is zero byte difference on every recorded row, not a calibrated tolerance claim.

Four fresh unmodified reference/candidate/reference/candidate greedy runs each produce 256 IDs without EOS and nonempty decoded text. Every token ID and each 254,279,680-byte logit dump match within this comparison. Actual generated row IDs are checked against the prepared set after every run. The pre-existing tokenizer-only helper verifies nonempty decoded text; the model outputs come from the new builds. All commands, executable hashes, checked input rows, per-path metadata, IDs and output digests are under `model/`; large raw float dumps were removed after their byte comparisons. The exact executed driver is `rebase-gpu.py`.

## Limits

This is new numerical and integration evidence for the rebase, not a new timing run. The previous M3/M4 cost probe and full-table native benchmark retain their original `01b4c6d6` / `8c0a7ec3` identities and all observations. Complete-table HTTP was not rerun after this rebase; its historical 49 PASS reference versus 48 PASS and one FAIL candidate, and subsequent matching fixed-control failures, remain disclosed in the prior evidence. A second trained-model/chip setup, constrained streamed-expert/long-context configurations and SSD cold-table behavior remain unverified. The M4 cost and row checks are CPU-only. No Metal source, default selection, prefetch policy or calibrated floor changed.
