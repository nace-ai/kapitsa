# IQ3_S M3 post-rebase evidence

Explicitly redacted derived summaries, not raw execution logs. Original records
and commands remain private. This adds M3 results; all earlier evidence is frozen.

Runs: M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1,
Apple clang 17 (clang-1700.6.4.2), own CMake Release builds, C++20 -O3.
Candidate: `60d312d1ee278cc0b07f2f6cc1ae3162c6265a36`.
Reference: `1ea325c87b0302db86fea7c5d0bddfc2c72257ad`.

The pinned Qwen3.5-4B IQ3_M model and nine imported files are the same verified
original M5 reference import documented in the preceding evidence folders.
There were zero independent M3 imports.

Each arm has 584 rows, all finite, with complete output and scored input bytes
literally equal. Strict-zero step/verify/batch checks pass; prefill uses the
separate finite/literal check. The eight-live-row KFlash chain is teacher-forced
trunk coverage, not free-generating MTP. No floor changed.

Native A/B/A/B returns 0 with empty invalid lists. A1's repetition flag and
all lower candidate rates remain in NATIVE.json. Fixed A/B/A/B has 24 requests
of 512 tokens; six complete request/content hash pairs agree across all legs.
Quiet/warmup gates pass and decode/prefill flags are empty. Adjacent-pair middle
window changes span -0.69873% to +1.83160%. No inference acceleration or universal
no-regression claim. Prefill/cache text was not compared; the third cached turn
in each leg ends at the length limit.

Both HTTP arms have 54 PASS / 1 FAIL on the same keepalive assertion. The cause
remains unresolved. HTTP.json retains the observed times. Other model families,
non-AArch64 execution, M3 sanitizer coverage and free MTP remain unqualified.
