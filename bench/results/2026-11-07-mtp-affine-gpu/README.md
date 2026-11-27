# Affine MTP full-model verification

Reference: `b171f98363abd5dfd8ae5d24f786ca77b3801a4a`.
Candidate production source: `8e12c3d09ac4118775755000d75f5fdbe6cbca35`.
M3 builds identify as `b171f98` / `8e12c3d`; M4 builds identify as `b171f983` / `02e53017`.
`02e53017f19aaac71d4f64795ec55d0fb4c12d0e` adds the original evidence only; the production source hashes match the M3 candidate.
Every complete build was explicitly CMake-configured for its revision, Release `-O3`, Apple Clang 17.
These runs compare complete reference and candidate binaries. They do not use the earlier loader-only relink as a full-model baseline.

Runs: Mac Studio M3 Ultra (60 GPU cores), 96 GB, macOS 26.4.1 (25E253), with Qwen3.5 4B 4-bit and Qwen3.6 35B-A3B 4-bit;
MacBook Pro M4 Pro (20 GPU cores), 48 GB, macOS 15.7.1 (24G231), with Qwen3.8 27B 4-bit and an additional Qwen3.6 35B-A3B OQ4E MTP continuation.
Exact model paths, config/tokenizer digests, available sidecar digests, binary hashes and command environments are in `execution.json` / `results.json` and the command records.
Existing models were used unchanged. No model was downloaded for these runs.

## Output checks

Each primary model compares 8 prefill rows and 192 rows each for batch, step and verify: **584 per model, 1,752 total**.
Every row is finite and bitwise identical. All standing floor checks pass; the stored floor files were not changed or remeasured.
The candidate uses `--from` the reference prefixes. The M3 comparisons use the repository's standing floors rather than newly measured M3 floors; exact equality is stronger than the numeric threshold.
The 4B and 35B M3 dump metadata report ANE chunks/evals 0; the 27B M4 reports 10 chunks and 1,152 evals in each arm.
Do not interpret the M3 runs as ANE coverage. The reports below retain the actual path metadata.

The fixed chat prompt requests at least 2,000 words. Each continuation produces 256 IDs, nonempty decoded text, no end tokens (248044/248046), and all 256 IDs equal across the old/new arms for that model.
The printed decode rate covers 255 tokens after the initial prefill token.
`--dflash none --spec 2` forces the existing MTP head for 4B, 27B and the extra local 35B. The remote 35B has no MTP head or matching drafter and uses `--spec 0`.
The 4B head is a BF16 sidecar; the 27B head is the real affine sidecar reaching `own_qlin`; the extra 35B uses in-shards MoE MTP, which is compatibility coverage outside the changed sidecar path.

MTP statistics are the same within each model: 4B 123 cycles, histogram `38 37 48`, 2.07 tokens/cycle, accepted/drafted 0.54;
27B 96 cycles, histogram `7 19 70`, 2.66 tokens/cycle, accepted/drafted 0.83;
extra 35B 102 cycles, histogram `16 18 68`, 2.50 tokens/cycle, accepted/drafted 0.75.
The acceptance histograms imply 1.08, 1.66 and 1.51 accepted draft tokens/cycle, respectively (sum of accepted drafts divided by cycles).
The raw load lines name the loaded head and vocabulary range. The remote 35B logs explicitly say that speculative decoding is off.

Contended continuation timings are retained for transparency and excluded from performance conclusions. Their actual IDs/text remain recorded.

| Model / arm | IDs | EOS | Decode tok/s (255 tokens) | Contention flagged |
| --- | ---: | ---: | ---: | --- |
| 4b-continuation-0-base | 256 | 0 | 252.7 | False |
| 4b-continuation-1-candidate | 256 | 0 | 251.8 | False |
| 4b-continuation-2-base | 256 | 0 | 255.3 | False |
| 4b-continuation-3-candidate | 256 | 0 | 252.8 | False |
| 35b-continuation-0-base | 256 | 0 | 185.8 | False |
| 35b-continuation-1-candidate | 256 | 0 | 185.1 | False |
| 35b-continuation-2-base | 256 | 0 | 182.4 | False |
| 35b-continuation-3-candidate | 256 | 0 | 182.4 | False |
| 27b-continuation-0-base | 256 | 0 | 32.5 | False |
| 27b-continuation-1-candidate | 256 | 0 | 32.5 | True |
| 27b-continuation-2-base | 256 | 0 | 32.7 | False |
| 27b-continuation-3-candidate | 256 | 0 | 32.8 | False |
| 35b-mtp-0-base | 256 | 0 | 90.2 | True |
| 35b-mtp-1-candidate | 256 | 0 | 106.7 | True |
| 35b-mtp-2-base | 256 | 0 | 125.6 | True |
| 35b-mtp-3-candidate | 256 | 0 | 126.0 | False |

<details>
<summary>4b on M3 Ultra: exact comparison report</summary>

```text
prefill: 8 records (qwen3.5-4b-4bit)
  reference: 0.2.31 (b171f98), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (8e12c3d), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
batch: 192 records (qwen3.5-4b-4bit)
  reference: 0.2.31 (b171f98), batch {"members":4,"rows":2}, prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (8e12c3d), batch {"members":4,"rows":2}, prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
  PASS (tools/logit_floors/qwen3.5-4b-4bit.json: the decode passes' gate (the prefill rows are bitwise the same))
step: 192 records (qwen3.5-4b-4bit)
  reference: 0.2.31 (b171f98), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (8e12c3d), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
  PASS (tools/logit_floors/qwen3.5-4b-4bit.json: the decode passes' gate (the prefill rows are bitwise the same))
verify: 192 records (qwen3.5-4b-4bit)
  reference: 0.2.31 (b171f98), verify "KFlash tree (a chain of 8 live rows)", prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (8e12c3d), verify "KFlash tree (a chain of 8 live rows)", prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
  PASS (tools/logit_floors/qwen3.5-4b-4bit.json: the decode passes' gate (the prefill rows are bitwise the same))
```
</details>

<details>
<summary>35b on M3 Ultra: exact comparison report</summary>

```text
prefill: 8 records (Qwen3.6-35B-A3B-4bit)
  reference: 0.2.31 (b171f98), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (8e12c3d), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
batch: 192 records (Qwen3.6-35B-A3B-4bit)
  reference: 0.2.31 (b171f98), batch {"members":4,"rows":2}, prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (8e12c3d), batch {"members":4,"rows":2}, prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
  PASS (tools/logit_floors/Qwen3.6-35B-A3B-4bit.json: the decode passes' gate (the prefill rows are bitwise the same))
step: 192 records (Qwen3.6-35B-A3B-4bit)
  reference: 0.2.31 (b171f98), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (8e12c3d), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
  PASS (tools/logit_floors/Qwen3.6-35B-A3B-4bit.json: the decode passes' gate (the prefill rows are bitwise the same))
verify: 192 records (Qwen3.6-35B-A3B-4bit)
  reference: 0.2.31 (b171f98), verify "KFlash tree (a chain of 8 live rows)", prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (8e12c3d), verify "KFlash tree (a chain of 8 live rows)", prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
  PASS (tools/logit_floors/Qwen3.6-35B-A3B-4bit.json: the decode passes' gate (the prefill rows are bitwise the same))
```
</details>

<details>
<summary>27b on M4 Pro: exact comparison report</summary>

```text
prefill: 8 records (qwen3.8-27b-4bit)
  reference: 0.2.31 (b171f983), prefill_chunk 2048, ane {"chunks":10,"evals":1152,"reruns":0}
  candidate: 0.2.31 (02e53017), prefill_chunk 2048, ane {"chunks":10,"evals":1152,"reruns":0}
  every row bitwise identical
batch: 192 records (qwen3.8-27b-4bit)
  reference: 0.2.31 (b171f983), batch {"members":4,"rows":2}, prefill_chunk 2048, ane {"chunks":10,"evals":1152,"reruns":0}
  candidate: 0.2.31 (02e53017), batch {"members":4,"rows":2}, prefill_chunk 2048, ane {"chunks":10,"evals":1152,"reruns":0}
  every row bitwise identical
  PASS (tools/logit_floors/qwen3.8-27b-4bit.json: the decode passes' gate (the prefill rows are bitwise the same))
step: 192 records (qwen3.8-27b-4bit)
  reference: 0.2.31 (b171f983), prefill_chunk 2048, ane {"chunks":10,"evals":1152,"reruns":0}
  candidate: 0.2.31 (02e53017), prefill_chunk 2048, ane {"chunks":10,"evals":1152,"reruns":0}
  every row bitwise identical
  PASS (tools/logit_floors/qwen3.8-27b-4bit.json: the decode passes' gate (the prefill rows are bitwise the same))
verify: 192 records (qwen3.8-27b-4bit)
  reference: 0.2.31 (b171f983), verify "KFlash tree (a chain of 8 live rows)", prefill_chunk 2048, ane {"chunks":10,"evals":1152,"reruns":0}
  candidate: 0.2.31 (02e53017), verify "KFlash tree (a chain of 8 live rows)", prefill_chunk 2048, ane {"chunks":10,"evals":1152,"reruns":0}
  every row bitwise identical
  PASS (tools/logit_floors/qwen3.8-27b-4bit.json: the decode passes' gate (the prefill rows are bitwise the same))
```
</details>

## Native benchmark method and interpretation

| Model / arm | Sampled tok/s | Greedy tok/s | 4K TTFT (s) | Follow-up (s) | Flags |
| --- | ---: | ---: | ---: | ---: | --- |
| 4b-bench-0-base | 373.60 | 369.27 | 1.9377 | 0.0309 | none |
| 4b-bench-1-candidate | 338.40 | 368.95 | 1.9368 | 0.0305 | decode: repetition loop (distinct-4-gram 0.3758); decode: repetition loop (distinct-4-gram 0.4686) |
| 4b-bench-2-base | 337.31 | 369.11 | 1.9375 | 0.0355 | none |
| 4b-bench-3-candidate | 340.33 | 360.19 | 1.9364 | 0.0348 | none |
| 35b-bench-0-base | 183.94 | 185.82 | 1.4840 | 0.0381 | none |
| 35b-bench-1-candidate | 181.28 | 182.77 | 1.4875 | 0.0431 | decode: repetition loop (distinct-4-gram 0.4438) |
| 35b-bench-2-base | 188.55 | 186.11 | 1.4856 | 0.0370 | none |
| 35b-bench-3-candidate | 182.33 | 183.74 | 1.4878 | 0.0425 | none |
| 27b-bench-0-base | 64.62 | 66.58 | 10.5246 | 0.2487 | none |
| 27b-bench-1-candidate | 58.08 | 65.99 | 10.5301 | 0.2524 | none |
| 27b-bench-2-base | 59.52 | 66.15 | 10.6841 | 0.2514 | CONTENTION: sustained foreign GPU load > 5% in ['prefill_4000'] (mean 0.85%, top ['Lark']) |
| 27b-bench-3-candidate | 51.17 | 66.21 | 10.7854 | 0.2764 | none |
| 27b-repeat-base | 60.63 | 65.30 | 10.5261 | 0.2510 | none |
| 27b-repeat-candidate | 47.97 | 51.65 | 12.4507 | 0.2675 | CONTENTION: sustained foreign GPU load > 5% in ['decode', 'decode_greedy', 'prefill_4000', 'tools'] (mean 77.32%, top ['Google Chrome']) |

| Arm | Three sampled runs (tok/s) | Three greedy runs (tok/s) |
| --- | --- | --- |
| 4b-bench-0-base | [398.46, 373.6, 343.53] | [369.27, 376.43, 360.7] |
| 4b-bench-1-candidate | [338.4, 334.02, 375.27] | [372.44, 368.95, 362.34] |
| 4b-bench-2-base | [337.31, 382.11, 287.2] | [371.31, 367.5, 369.11] |
| 4b-bench-3-candidate | [340.33, 336.16, 383.57] | [360.19, 341.44, 371.0] |
| 35b-bench-0-base | [185.1, 183.94, 180.85] | [185.82, 183.31, 186.16] |
| 35b-bench-1-candidate | [181.24, 181.28, 181.32] | [181.87, 183.18, 182.77] |
| 35b-bench-2-base | [190.16, 188.55, 186.4] | [188.66, 186.11, 185.86] |
| 35b-bench-3-candidate | [181.49, 182.33, 188.27] | [183.19, 183.74, 184.34] |
| 27b-bench-0-base | [60.4, 64.92, 64.62] | [66.1, 66.58, 67.4] |
| 27b-bench-1-candidate | [55.38, 58.08, 64.27] | [65.99, 65.15, 66.69] |
| 27b-bench-2-base | [59.52, 64.16, 55.26] | [65.91, 66.48, 66.15] |
| 27b-bench-3-candidate | [51.17, 50.14, 52.4] | [62.96, 66.21, 66.68] |
| 27b-repeat-base | [61.19, 60.63, 55.58] | [63.97, 65.85, 65.3] |
| 27b-repeat-candidate | [67.7, 45.63, 47.97] | [51.65, 50.99, 54.82] |

Each arm runs `kapitsa bench --model <existing-folder> --quick --json --out <new-file>` with the complete binary for that arm.
The caller holds `/tmp/kapitsa-gpu.lock` and sets `NI_LOCK_HELD=1`. Each arm uses a separate scratch `KAPITSA_HOME`; native bench starts its own server.
No environment-switch A/A experiment is used. The recorded native path is DFlash2/KFlash for 4B and 27B, and plain decoding for the remote 35B.
Native bench inserts fresh nonces into prompts, so its sampled generations are not exact-output evidence; the fixed-input continuations and logits provide that evidence separately.
Every completed native arm exited 0 with `invalid: []`. Individual three-run samples and the saved conditions/flags are retained.

No inference speedup or memory benefit is claimed. The remote 35B candidate greedy medians are 1.1–1.6% below their adjacent reference arms;
its fixed-input plain continuation pairs are 185.8→185.1 and 182.4→182.4 tok/s. This is not a uniform no-slowdown claim or proof that the difference is noise.
The code change validates tensor metadata during loading and does not change a kernel or decode strategy.
Peak native memory was 4.55 GB (4B), 19.45 GB (35B) and 19.18 GB (27B) in both source revisions; the later 27B repeat reports 19.17 GB in both arms.

The original 27B A2 arm reports sustained Lark activity during prefill. Its whole A/B pair remains above and is repeated under new labels.
The original 27B first candidate MTP continuation and the first three extra 35B MTP arms report foreign GPU activity.
Repetition flags in the first 4B and 35B candidate native arms remain visible. No flagged measurement is silently removed.
Short foreign activity below the native sustained threshold is recorded in each JSON's conditions, even when the flag list is empty.

The original standalone wrapper tested the truthiness of `gpu_hogs.wait_quiet()`'s tuple, so its preflight did not enforce the returned success field.
This limitation is retained with the original scripts. Native bench's own contention records, and the continuation monitors, determine which results were affected.
The first strict native-repeat wrapper was stopped while waiting for its predecessor, before launching any measurement.
The corrected repeat scripts check `quiet[0]` and record the complete tuple; the helper reports foreign processes above 0.5%, even when configured with a 0.1% limit.

## HTTP and CPU outcomes

`tests/test_server.py URL` ran against each complete build's server, slots 2, context 32,768, disk cache 0, under the GPU lock.

| Setup | Reference | Candidate | Failures |
| --- | --- | --- | --- |
| M3 / 4B | 47 PASS, 2 FAIL | 47 PASS, 2 FAIL | In-flight ordering; busy-slot stream keepalive |
| M3 / 35B | 49 PASS, 0 FAIL | 49 PASS, 0 FAIL | None |
| M4 / 27B | 48 PASS, 1 FAIL | 48 PASS, 1 FAIL | Busy-slot stream keepalive |

For the 4B ordering assertion the long answer ended in 0.9 s before the side request ended in 2.4 s, in both arms.
The exact keepalive observations and exit-1 results remain in the HTTP logs. These are matched baseline failures, not an all-pass HTTP result.

The original frozen bundle records the Release build, all 21 `vk cpu-tests` suites, 308 grammar assertions, tokenizer-only parser/structured checks,
20 passing supplemental commands and final lint. Its CPU `load_error_test` failed the unchanged residency assertion in both arms (3→962 and 3→1,474 of 4,420 pages).
The complete remote reference CPU run also fails (3→897); the remote candidate passes (3→3). This does not establish a fix for that OS-cache-sensitive assertion.
`evidence-gpu-1041/cpu-extra-results.json` records their actual exit codes. No all-CPU-green claim is made.

On the M3, `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 build/load_error_test --mtp-gpu` rejects 13 malformed loader cases and accepts 30 valid cases.
The same cases passed locally in the original bundle. These use Metal buffers without dispatching a kernel.
Both reference and candidate affine CPU helpers pass on M3. No Metal kernel changed, so there is no changed-kernel dispatch to validate.

## Remaining clean performance check

The corrected 27B native repeat has a clean reference arm, but its candidate arm reports sustained Chrome activity in decode, greedy decode, prefill and tool calls (mean 77.32%, maximum 98.54%). It is not a clean performance comparison.
The additional 35B quiet continuation repeat was stopped while waiting for an idle local GPU, before launching any model; its original four completed continuation records remain above.
This evidence does not complete the idle 27B performance gate. That gate remains pending on the M3 using an identical copy of the 27B model; file hashes are to be verified before loading it. No results from that pending job are claimed here.

## Scope and reproduction

Real stacked-affine expert sidecars were not available: that exact layout is covered by synthetic loader fixtures, not a full model.
In-shards MTP and numbered-expert stacking admission are outside this fix. Complete architecture-dimension validation, other model families,
M5 tensor kernels, longer-context quality runs and other Macs were not verified by this change.

Copy the recorded scripts/configuration files to a fresh scratch directory and change their absolute paths for the machine being used.
Build both revisions with an explicit `cmake -S ... -B ... -DCMAKE_BUILD_TYPE=Release`, then the named binaries.
`run_checks.py` documents the commands actually executed, including the original preflight limitation above; use the corrected tuple check when rerunning it.
Never write a new run into this frozen directory. The large raw `.f32` arrays were deleted after comparison; reports, metadata, record indices and source hashes remain.
The previous `2026-11-07-mtp-affine-shapes` bundle is unchanged; its historical pending-inference note describes its own measurement stage.
