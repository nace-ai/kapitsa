# Flash-Next index-cache growth

Flash-Next index-cache growth retains every old index buffer until all replacements have been allocated and copied. Attention-cache growth already releases old buffers in batches. Apply that same policy to index caches: allocate and copy a batch, wait for its command buffer, then release that batch's old buffers. A batch targets 256 MiB of destination buffers and includes at least one buffer; a single larger buffer can exceed the target. Layout and copied contents are unchanged.

The included `index_growth_check` loads the actual model class under the GPU lock, fills each index buffer with distinct deterministic bytes, grows the cache and compares every retained byte. Its allocation sampling is diagnostic and may miss a transient maximum; resource-usage diagnostics are omitted from public evidence.

## Default for

All Flash-Next index-cache growth, including MTP index caches. No opt-in switch or hardware-specific gate. Complete-model validation ran on both M3 Ultra and M5 Pro.

## Measured

**Redacted derived evidence, not original execution logs:** the JSON files in this directory. Original records remain private and unchanged. Command placeholders and relative paths are illustrative.

Reference production source: `903cdfcebafb3d138d50c9c83e7787912ee26b81`.
Candidate production source: `cbbf2dcf614761d834e1e4129e6a621ace3a3d6e`; subsequent evidence additions do not change production code. Both use CMake Release. Binary and source-file SHA256 values are in the evidence.

Runs:
- Apple M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1, Apple clang 17.0.0 (`clang-1700.6.4.2`).
- Apple M5 Pro, 20 GPU cores, 48 GiB, macOS 26.6.2, Apple clang 17.0.0 (`clang-1700.0.13.5`).

Both setups used the same synthetic fixture and complete Qwen3.8-Flash-Next Sushi pack, with EXL3 2-bit experts and mixed 8-bit/bf16 weights. All 215 model files on M5 were SHA256-verified against the M3 identities; no conversion was performed. The restored n-gram table matches `c8ab74bc343408cf3923d7d64b3698fbeb3e78c07ce7f85a650a8278731251d2`.

### Cache preservation

| Setup | Capacity change | Buffers | Retained bytes compared per arm | A1 / B1 / A2 / B2 growth, ms |
|---|---:|---:|---:|---|
| M5 synthetic | 262,144 → 327,680 | 8 | 402,655,232 | 117.440 / 108.121 / 109.697 / 104.720 |
| M3 synthetic | 262,144 → 327,680 | 8 | 402,655,232 | 363.576 / 361.724 / 359.671 / 369.846 |
| M3 complete model | 65,536 → 81,920 | 26 | 327,162,368 | 4586.472 / 522.706 / 498.249 / 458.210 |
| M5 complete model | 65,536 → 81,920 | 26 | 327,162,368 | 1826.589 / 627.620 / 217.380 / 132.738 |

Every arm compared all retained bytes with its input patterns: **zero mismatches**. These byte counts describe compared test data. Growth timings are not inference throughput; the first complete-model reference samples are retained in full, with no warm-up sample excluded.

Both chips completed the small synthetic 64 → 128 check with `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`: eight buffers, all 100,352 retained bytes equal. The complete-model M5 small check also passed for 26 buffers and 326,144 retained bytes, without those validation flags. No Metal kernel changes are included.

M5 synthetic measurements used the preliminary checker. The final checker adds strict capacity parsing and scoped sampler-thread cleanup; it was used in both M3 arms and both M5 complete-model arms.

### Complete-model outputs

Separate reference and candidate `logit_dump` executions on each chip used all eight corpus windows with `--paths step,verify`; prefill is emitted automatically. **Each arm on each chip has 392 finite rows: eight prefill, 192 step and 192 verify, all bitwise identical between A and B.** Independent audits compared all bytes of the complete `logits.f32`, `records.txt` and `sequences.txt` files for every path.

No floor file exists for this pack, so the comparator is reports-only. This is an independently verified exact zero-difference result, not an exit-code floor verdict. Verify rows are teacher-forced coverage, not free-running MTP equality. Flash-Next has no batch path.

The archived M3 builds report version 0.2.33 without an embedded commit. Their source archive's 292 compiler-input files match the reference commit; the candidate overlay and checker were checked separately. Source audits and binary hashes establish identity. M5 builds report their respective commits.

### Native model benchmarks

Unmodified `kapitsa bench --quick --json`, interleaved A/B/A/B on each setup, with the exclusive GPU lock:

| Setup / arm | Sampled decode, tok/s | Greedy decode, tok/s | 4K prefill, tok/s |
|---|---:|---:|---:|
| M3 A1 | 54.16 | 57.49 | 828.3 |
| M3 B1 | 57.28 | 57.49 | 823.3 |
| M3 A2 | 53.65 | 57.77 | 828.3 |
| M3 B2 | 56.46 | 57.93 | 828.2 |
| M5 A1 | 44.64 | 46.90 | 319.6 |
| M5 B1 | 44.15 | 47.18 | 320.0 |
| M5 A2 | 45.43 | 46.62 | 318.4 |
| M5 B2 | 41.61 | 47.08 | 317.4 |

All eight records have empty `invalid` lists and pass tool-call checks. Warning categories are retained below; resource-usage details are redacted:

- M3 A1: decode: repetition loop (distinct-4-gram 0.4017).
- M5 B1: decode_greedy: resource-pressure warning (resource-usage details redacted).
- M5 B1: prefill_4000: resource-pressure warning (resource-usage details redacted).

All per-run samples and cached follow-up timings are retained in the derived JSON files, including slower samples. M3 B2's cached follow-up was slower than A2 (0.2133 s versus 0.1701 s). M5 resource-pressure warnings limit interpretation of its throughput. Native requests vary and are not the fixed-input comparison above. **No overall inference speedup or universal absence of regressions is claimed.**

## Checked

- Release CLI, logit-dump and growth-checker builds on both setups; explicit CMake configuration per source tree.
- All 23 `vk cpu-tests` commands and `prek run --all-files` passed.
- Malformed, overflowing and decreasing capacities are rejected before GPU use.
- Cache retained-byte checks, complete-model output checks and native interleaved runs described above.
- Initial failed attempts remain recorded: a missing working-directory corpus, unsupported batch coverage and an invalid explicit `prefill` path argument. The completed run uses the source working directory and supported paths.

## Not verified

- The long-context failure has not been reproduced as an allocation failure. Evidence establishes bounded retention in the implementation and successful cache copies, not out-of-memory recovery.
- Other chips, model packs and long-context inference workloads.
- Free-running MTP output equivalence is not established by the teacher-forced verify check.
- The existing loader reports that this model's requested YaRN RoPE is not applied in both arms. This change does not alter that behavior.

Evidence files:
- `cache-growth.json`: every cache-growth run and small validation configuration.
- `m3-output-check.json` and `m5-output-check.json`: complete-byte comparisons, finite checks, comparator metrics and binary identities.
- `m3-native-abab.json` and `m5-native-abab.json`: every native benchmark sample, warning category and invalid list.
- `model-identity.json`: all model file digests and synthetic-fixture identities.
- `validation.json`: CPU checks, source audit and retained failed attempts.

Each derived JSON records private source-record names and SHA256 provenance.
