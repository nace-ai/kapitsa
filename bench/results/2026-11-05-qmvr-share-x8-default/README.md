# Shared X8: final-source checks on M4 Pro

The final source passes the output and functional checks. Repeated speed confirmation against the new base is still pending. These results make the proposed default reviewable; they do not establish a final-source speedup.

Runs: Apple M4 Pro, 20 GPU cores, 48 GiB; macOS 15.7.1 (24G231), AC power. Model: `scottlowry/Qwen3.6-35B-A3B-oQ4e-mtp`, revision `b5b701ea50a5baae1fe0cf88f5ea8ca7e935ba9b`.

- Baseline: `8a1e67a6cda9ca962775bc923b215f935d5ad6c1`, including the RMSNorm/X8 fusion from #965.
- Tested production source: `c27f9e26ce8af72cfeb58a5b0e0cc16541907444`. The commit adding this directory changes only recorded evidence.
- Both final A/B builds leave `NI_QMVR_SHARE_X8` and `NI_NORM_X8` unset. Generation tests use native MTP depth 1 and disable DFlash.
- `evidence.json` contains binary/source hashes, comparison results, every GPU attempt's audit, raw HTTP results, and individual CLI results. Machine-local paths are replaced with named placeholders; measurement values are unchanged.
- `logit-compare.txt` is the final comparison's exact output. `checks.log` retains CPU, HTTP regression and lint output, including the two initial missing-dependency failures and successful reruns.

## Output and functional checks

The full eight-window check compares the final default against unmodified upstream:

| Path | Rows | Bitwise identical | Non-finite rows |
| --- | ---: | ---: | ---: |
| Prefill | 8 | 8 | 0 |
| Step | 192 | 192 | 0 |
| Tree verify | 192 | 192 | 0 |
| Batch | 192 | 192 | 0 |

All 584 rows pass. The exact oQ4e pack has no published floor in the tested checkout. Explicit zero KL/NLL drift and full top-1 agreement bounds, plus independent all-row bitwise/finite assertions, were fixed before the run. No floor was borrowed, measured or edited.

The candidate logit-dump process recorded 16 system swap-in pages, zero pageouts/swapouts and no sustained GPU contention. This is correctness evidence, not a timing measurement.

Additional checks:

- R=1..8 at a 512-token prefix: all 36 trunk-logit and 36 MTP-hidden rows match for default, explicit sharing on/off, normalization fusion off, and both off.
- `NI_NO_QMVR_MULTI=1`: candidate and upstream with the same flag match all 36 trunk and 36 MTP rows. An initial comparison against the normal default was an invalid cross-setting expectation: this flag also changes pre-existing specialized kernels. The same-setting control resolved it without a production edit.
- Small real-model check at prefix 1, R=1..8 with `MTL_SHADER_VALIDATION=1` and `MTL_DEBUG_LAYER=1`: passed. No Metal kernel changes in this PR.
- All 20 `vk cpu-tests` groups passed. The initial bench/community failures were missing `jsonschema`; both passed after installing it in a task-local dependency directory. All 16 additional CPU groups also passed.
- Full two-slot `test_server.py` HTTP regression passed with native MTP depth 1, 32K context and disk cache disabled.
- All 24 fixed-input HTTP generations were nonempty, 512 tokens, finished by length, and matched the corresponding output text across A/B/A/B.
- All ten short CLI runs (two warmups and four measured pairs) emitted the same 512 tokens with no EOS, 278 MTP cycles, and acceptance histogram `45 233`.
- Release build, full-repository `prek`, whitespace and PR-size checks passed.

CPU, server, switch and Metal checks ran on identical production source before the version-stamp rebuild for `c27f9e26`. The `mixed_rows_check` and `ni_run` hashes did not change. The final standard output gate used the `c27f9e26`-stamped `logit_dump` binary. Hashes and the reference driver's build command are in `evidence.json`.

## Timing status

The predeclared rule requires at least two complete valid interleaved pairs. Each pair is excluded if either leg fails its environment checks, independent of its speed. HTTP timings require zero system swap-ins/swapouts after warmup. Short CLI timings require zero lifecycle pageouts/swap-ins/swap-outs. These zero-swap filters are this experiment's additional criteria, not a claim that the reviewer mandated these exact thresholds.

| Comparison | Complete pairs | Eligible pairs | Result |
| --- | ---: | ---: | --- |
| Standard project HTTP harness | 3 | 0 | All pairs excluded for system swap-ins; native harness flags/invalid arrays were empty |
| Fixed-input HTTP control | 2 | 0 | Both candidate legs clean; both baseline legs had swap-ins |
| Short fixed-prompt CLI | 4 | 1 | Only pair 3 clean; final confirmation pair excluded |

The single clean CLI pair measured 110.4 to 112.1 tok/s on an 881-token prompt. It emitted 512 tokens, with 511 timed decode tokens; loading and prefill are excluded. This is one observation, **not a replicated speed claim or an HTTP throughput result**.

The short sequence was added after the standard HTTP pairs were excluded: two untimed warmups, then three predeclared A/B pairs. Because only pair 3 passed the environment checks, exactly one additional confirmation pair was declared and run; it also failed the environment criterion. No further pairs were added. Every attempt is retained in `evidence.json`.

The previously reviewed HTTP result, 112.32 to 114.41 tok/s greedy (about +1.9%), used upstream `a0f76af6`. Sampled generation was unchanged (110.57 to 110.48 tok/s). That historical comparison is in the [earlier review attachment](https://github.com/nace-ai/kapitsa/pull/837#issuecomment-6657421062); it is not relabeled as a measurement against `8a1e67a6`.

## Reproduction

Build the two pinned commits as Release builds in separate checkouts. Use the same model revision on the same Mac. Leave `NI_QMVR_SHARE_X8` and `NI_NORM_X8` unset; do not inherit other tuning overrides.

On a quiet GPU, the standard output commands are:

```sh
env -u NI_QMVR_SHARE_X8 -u NI_NORM_X8 NI_DFLASH=0 \
  "$BASE_BUILD/logit_dump" "$MODEL_DIR" "$CHECK_DIR/base"
env -u NI_QMVR_SHARE_X8 -u NI_NORM_X8 NI_DFLASH=0 \
  "$CANDIDATE_BUILD/logit_dump" "$MODEL_DIR" "$CHECK_DIR/candidate" \
  --from "$CHECK_DIR/base"
python3 tools/logit_compare.py "$CHECK_DIR/base" "$CHECK_DIR/candidate" \
  --min-top1 1 --max-mean-kl 0 --max-median-kl 0 --max-p99-kl 0 \
  --near-tie 0 --max-far-flips 0 --max-nll-change 0
```

`logit_dump` takes the GPU lock itself. Verify all 584 rows are bitwise identical and finite, in addition to the compare command's exit status. Raw `.f32` files were hashed and removed after comparison; sequences, metadata and reports were retained.

The standard speed sequence uses the unchanged `bench/harness/compare_engines.py` / `bench.py`, three sampled and three greedy 512-token generations per leg, 4K prefill and cached follow-ups, and tool checks. Order: A1 B1 A2 B2 A3 B3. Server flags: `--ctx 32768 --slots 1 --disk-cache 0 --spec 1 --dflash none`. Exact configurations and commands are in each raw HTTP result's `config` and `provenance` fields.

The fixed-input HTTP control uses the same project request/timing helpers, input nonce and sampling seeds 837001/837002/837003, temperatures 0.6 and 0, in A1 B1 A2 B2 order. Short CLI legs use `ni_run MODEL @prompt-881.ids 512 0 --spec 1 --dflash none`; the exact 881-token prompt is included in this directory. The original run command called this same file `<evidence>/prompt-1k.ids`; it differs from the older October 31 1024-token prompt. Run each model process through `vk gpu run`, without a nested lock, and record power, GPU activity and VM counters. The short CLI parser also checks nonempty decoded text, EOS counts, token equality and MTP cycle/histogram equality.

## Not verified

- Repeated final-source speed confirmation: fewer than two complete pairs passed the declared environment criteria.
- Other Macs, model packs, and performance without MTP.
