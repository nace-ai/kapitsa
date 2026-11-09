# IQ3_S post-rebase validation

These are explicitly redacted derived summaries, not raw execution logs. Original
commands, complete results and artifacts are retained privately. The original
measurements in `../2026-11-09-iq3-s-cpu/` remain unchanged.

Candidate production source: `60d312d1ee278cc0b07f2f6cc1ae3162c6265a36`.
Reference: `1ea325c87b0302db86fea7c5d0bddfc2c72257ad`.
Later evidence-only commits do not change the binaries measured here.

Runs: M5 Pro, 20 GPU cores, 48 GiB, macOS 26.6.2,
Apple clang 17 (clang-1700.0.13.5), own explicit CMake Release builds,
C++20, arm64, `-O3`, actual target flags without `NDEBUG`.

## Model and numerical check

The pinned IQ3_M GGUF and its IQ3_S tensors are identified in M5-NUMERICAL.json
and the original evidence folder. Both arms share the original M5 reference
import, whose nine files were verified by digest. There are zero new independent
imports on this source pair. The original independent imports at 5229f9c5/ee8ce8c8
were literally equal; they are not relabeled as new imports.

Each arm produced 584 rows (8 prefill, 192 each of step/verify/batch), vocabulary
248,320: all 145,018,880 float32 values finite, complete output and scored input
bytes equal. Report-only and strict-zero checks returned 0; prefill uses the
separate finite/literal gate, since the strict thresholds cover step/verify/batch.
No floor changed. Verify is the KFlash tree with eight live chain rows, a
teacher-forced trunk check, not free MTP generation.

## Native and fixed requests

Native A1/B1/A2/B2 completed with rc 0 and empty invalid lists. A1 has two
repetition flags. M5-NATIVE.json retains every run's rates and all flags.

The original fixed sequence was A1/B1/A2/B2. B1 failed its initial quiet gate
with rc 1 before any decode request. After the original HTTP checks, A3/B3 ran
with the same binaries, driver, model and settings. A2/B2/A3/B3 provide 24 decode
requests of 512 tokens (sampled/greedy, seeds 1/2/3), with all six complete
request/content hash pairs equal across those four legs. The extra original A1
and failed B1 are also retained in M5-FIXED.json.

B2 sampled seed 2 has a contention flag, so its timing is not idle-GPU evidence.
A3 prefill separately has a contention flag. All rates remain in the file;
adjacent-pair middle-window changes span -2.92130% to +4.37656%, including the
flagged timing. No inference acceleration or universal no-regression is claimed.
Prefill/cache text was not compared and different histories do not establish
matched cache performance. Cache finish reasons are retained.

The fixed driver is unchanged from the source in the original evidence folder:
SHA256 `6b022a4791ac656f74ca84dc7b2090e7bfd34e5ad0e7a1c6bc1197e8adcd8310`.
All model phases held the exclusive GPU lock with caller-held markers.

## HTTP and remaining coverage

Reference: 54 PASS / 1 FAIL, rc 1. Candidate: 55 PASS / 0 FAIL, rc 0.
The reference's `next turn: answer` assertion received another `run_shell` tool
call instead of textual content. Its cause remains unresolved. This baseline
limitation is retained, not an all-checks-pass result or a universal API claim.

The M5 source-specific CPU checks passed before the model runs. On M3 Ultra
(60 GPU cores, 96 GiB, macOS 26.4.1, Apple clang 17 clang-1700.6.4.2), both
source-specific Release builds and integrated CPU tests passed; the candidate
ran 614,602 oracle cases. M3 full-model results are pending and are not claimed
in this folder. No additional sanitizer run was made; the original ASan
initialization failure remains disclosed. Other model families, other AArch64
chips, non-AArch64 execution and free-generating MTP are not qualified.
