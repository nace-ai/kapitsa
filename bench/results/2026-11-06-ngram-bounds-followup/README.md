# Follow-up validation

Runs: MacBook Pro (16-inch, 2024), Mac16,7, M4 Pro (14 CPU / 20 GPU cores), 48 GiB,
macOS 15.7.1 (24G231), Apple Clang 17, CMake Release (-O3), macOS deployment target 14.
Reference `934a6d72`, implementation `877e7b4d`. `identity.json` records the tested tree head,
actual build version and library SHA256. Evidence-only commits do not change the implementation.
Both A/B checkouts were explicitly CMake-configured before building; the reference reports d65a0afa
for the range/vision comparisons. Original published records remain unchanged in their original folder.

`cpu/` contains the full 21-command `vk cpu-tests` result and raw per-test logs, supplemental tests,
real 4B tokenizer/parser/structured checks, build/configure, prek and cleanup logs. Additional checks:
qwen4_config_test, test_drex_max.py, qwen4_exp_pack_test.py, and systemone_codes_driver build.
`dflash_accept_test` was invoked with `--cpu`. No GPU validation is implied by these CPU results.
`required-checks.txt` is the path classifier's checklist, not a statement that every listed model ran;
its model-folder auto-detection misses existing folders here. The explicit coverage below is authoritative.

Rebuild with `cmake -S TREE -B TREE/build -DCMAKE_BUILD_TYPE=Release`, then build the named
CPU targets listed in AGENTS.md and run `.dsh/skills/verify-kapitsa/scripts/vk cpu-tests` plus
the supplemental commands in AGENTS.md. Use `prek run --all-files` after adding new material.

This is now a complete standalone build at 399dae9b, replacing the *coverage limitation* of the
initial selected-source probe (the old record is retained). `cpu/vk/fn_store_test.log` reports
26 malformed fixtures rejected, 3 valid formats, and all existing unequal-shard affine, GGUF and
gather tests passing. The original 78-byte missing-payload reproducer and reference failure remain
in ../2026-11-06-ngram-bounds/. No new numerical or performance result is inferred from them.

Not verified: an actual Flash-Next checkpoint is unavailable locally. Valid-model loading,
applicable model output, HTTP and interleaved A/B validation remain pending. Unrelated dense-Qwen
GPU checks from the other PRs are deliberately not evidence for this fix. Draft remains appropriate.
