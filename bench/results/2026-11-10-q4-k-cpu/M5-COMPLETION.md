# Q4_K complete-model supplement

This additive supplement completes the pending model items in the earlier records. JSON and text files are redacted derived evidence, not raw logs; raw records and measured binaries remain private. No earlier result file changed.

Runs: M5 Pro, 20 GPU cores, 48 GiB, macOS 26.6.2, Apple clang 17 (clang-1700.0.13.5). Reference `2e308aeac6d6fe95801e7957996034d6bbfa6ac3`, candidate `0c34e2397f8555e3c9a07774c0d3d2d1c994f8ba`, explicit CMake Release builds. Full pinned Flint model and binary/source digests are in M5-MODEL-VALIDATION.json.

- Two independent complete imports: all 9 files literally equal. The three Q4_K shared-expert tensors use the changed dequantization path before packing to group 64.
- Each arm: 392 rows (8 prefill, 192 step, 192 teacher-forced verify), 58,856,448 fp32 values, all finite. Complete output and recorded input bytes compare literally equal. Report-only and strict zero-drift checks pass; no floor was changed. Flint has no batch path. Teacher-forced verification is not free-running MTP.
- Native `kapitsa bench --quick --json` A/B/A/B exits 0 in all four legs. Decode medians: 149.77 / 145.86 / 148.51 / 148.27 tok/s; greedy medians: 148.69 / 149.46 / 148.65 / 148.70. Every sample is retained. A1/B1/B2 have a cached-series flag for a follow-up without content. B1's lower throughput remains; no inference-speedup or universal non-regression claim.
- Fixed A/B/A/B: all six requests generate 512 tokens but empty content in every leg, with reasoning-only invalid results. All six request/content SHA256 pairs agree across legs; empty hashes do not establish meaningful continuation equivalence. Prefill/template and cached-series invalid results remain recorded.
- HTTP: 53 PASS / 2 FAIL in each arm, with the same failed assertion names: conversation retention during one-offs, and stream keepalive while slots are busy. These also fail in the unmodified main reference. Counts do not imply identical response content; this is not an all-tests-pass claim.

The CPU default is supported by its completed M3/M5 decoder oracles and trained interleaved CPU measurements, with this independent M5 import/output check. Other chips, M3 full-model runtime, ASan, non-AArch64 execution and free-generating MTP were not run. No Metal kernel changed.
