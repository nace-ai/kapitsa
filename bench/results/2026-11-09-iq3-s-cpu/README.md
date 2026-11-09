# IQ3_S CPU decoder validation

These files are explicitly redacted derived summaries, not raw execution logs.
Original commands, logs, and full comparison artifacts are retained privately.
The two C++ drivers are exact copies of the executed sources; their SHA256 values
are recorded in CHECKS.json. Commands below are reproducible placeholder examples,
not verbatim execution records.

## Source and scope

The measurements in this folder used candidate
`5229f9c59dbf76d76f335175f4982ba77c0afb77` and reference
`ee8ce8c8f6e3a5a1e4461015bdc2f2653f6d7b11`.
The later rebase to `1ea325c8` retains the identical `dq_iq3_s` function and the
IQ3_S oracle. These results retain their original source identities; they are
not measurements of the rebased binaries.

The default applies to finite, non-overlapping IQ3_S CPU blocks on AArch64.
Non-finite half scales and input/output overlap retain the original scalar path.
No Metal kernel, quantization policy, or inference default changed.

Runs:

- M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1,
  Apple clang 17 (clang-1700.6.4.2).
- M5 Pro, 20 GPU cores, 48 GiB, macOS 26.6.2,
  Apple clang 17 (clang-1700.0.13.5).

The CPU row driver was linked against each source's own CMake Release library
(`-O3`, arm64, C++20). Each setup ran A1 B1 A2 B2 A3 B3 A4 B4, with one warmup
and 16 measured passes per leg. Each pass decodes the first 1,024 rows of the
trained `blk.0.ffn_up.weight` tensor (width 2,560), using one library call per row.
All 128 passes per setup compare all 10,485,760 output bytes, require finite
values, and check input bytes and output canaries. Complete leg files also match.
All per-pass timings appear in M3-CPU.json and M5-CPU.json.

| Setup | Pair | Reference ns/row | Candidate ns/row | Latency change |
|---|---|---:|---:|---:|
| M3 | 1 | 5156.0874025 | 2251.3427735 | -56.33622% |
| M3 | 2 | 5127.4008785 | 2250.8339845 | -56.10185% |
| M3 | 3 | 5154.6225585 | 2257.7719725 | -56.19908% |
| M3 | 4 | 5111.125 | 2250.0810545 | -55.97679% |
| M5 | 1 | 771.321289 | 329.793457 | -57.24305% |
| M5 | 2 | 570.577148 | 267.272949 | -53.15744% |
| M5 | 3 | 564.677246 | 269.815918 | -52.21767% |
| M5 | 4 | 566.9560545 | 257.141113 | -54.64532% |

This is CPU row decoding evidence, not a measurement of complete import or
inference acceleration. The absolute difference between chips is not explained
by these runs. Other AArch64 chips and other model families were not measured.

## Model identity and output checks

Model: `bartowski/Qwen_Qwen3.5-4B-GGUF` at
`4168f45a16a1290d65a4ec0fa312ae917a4c15d6`, file
`Qwen_Qwen3.5-4B-IQ3_M.gguf`, 2,483,333,088 bytes, SHA256
`5ad76abae6b7a27d1463d90e6d86886850769e34906a88cfa4347cae72c91502`.
The mixed-format model contains 140 IQ3_S tensors. The measured tensor has
GGUF type 21 (IQ3_S); its first 1,024 rows occupy 1,126,400 bytes, SHA256
`79b2b8381df6c247724d7a6d7127032075a862167bb3afe82a4dc3637ad549ae`.

M5 independently imported both arms with `--no-drafter`; all nine resulting
files, including the manifest, match literally. The global configuration and
measured trunk FFN-up module use affine 4-bit/group-32 packing. This does not
assert the same format for every module.

Both M5 arms produced 584 rows: 8 prefill, 192 step, 192 verify, and 192 batch,
with vocabulary 248,320. All 145,018,880 float32 values per arm are finite;
complete output and scored input bytes match. Report-only and strict-zero
comparison return 0. Strict thresholds apply to step/verify/batch; prefill is
covered by the separate finite and literal-byte check. No floor was changed.
Verify uses the KFlash tree with eight live chain rows, a teacher-forced trunk
check, not free-generating MTP validation.

Example commands, with the appropriate per-arm imported model and own builds:

```sh
"$REF_BUILD/logit_dump" "$REF_MODEL" output/reference --paths step,verify,batch --windows 8 --rows 24 --chunk 2048 --batch 4 --batch-rows 2 --corpus bench/corpus.txt
"$CAND_BUILD/logit_dump" "$CAND_MODEL" output/candidate --paths step,verify,batch --windows 8 --rows 24 --chunk 2048 --batch 4 --batch-rows 2 --from output/reference
python3 tools/logit_compare.py output/reference output/candidate --no-gate --json report-only.json
python3 tools/logit_compare.py output/reference output/candidate --min-top1 1 --max-mean-kl 0 --max-median-kl 0 --max-p99-kl 0 --near-tie 0 --max-far-flips 0 --max-nll-change 0 --json strict-zero.json
```

These model commands require an exclusive GPU lock. A caller holding that lock
sets both `NI_LOCK_HELD=1` and `NACE_GPU_LOCK_HELD=1` for wrapped tools.

## Default behavior and retained limitations

M5 native A/B/A/B returned 0 with empty invalid lists. A1 reported contention
in prefill/tools; its affected timings are not idle-GPU evidence. Private monitor
details are removed from the derived flag. Other legs have empty flags. Every
native sampled/greedy timing is retained in M5-NATIVE.json.

The fixed driver ran 24 decode requests of 512 tokens: sampled and greedy,
seeds 1/2/3, on each A/B/A/B leg. All six complete request/content hash pairs
match across the four legs. All decode rows have empty flags/invalid lists and
pass the recorded quiet/warmup checks. Adjacent-pair middle-window rate changes
range from -1.92969% to +0.20870%; this is not an inference speedup claim.
B1 prefill separately reports contention. The third cached turn ends in `length`
in every leg. Prefill/cache text was not compared, and different histories do
not establish matched cache performance. These limitations remain in M5-FIXED.json.

HTTP: 55 PASS / 0 FAIL on each M5 arm. This covers the executed cases on this
model and setup, not universal API behavior. M3 full-model validation was not
part of the original measurements in this folder.

CHECKS.json records the 614,602-case oracle, two rejected mutants, CPU suite,
and standalone UBSan coverage. ASan compiled but stalled during runtime
initialization before main: no ASan check executed. Initial failed special-value
harness controls and the corrected scalar reference semantics remain disclosed.
