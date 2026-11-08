# Nemotron-H FP8 model checks on M4 Pro

These are derived, privacy-filtered exports of completed output and default HTTP checks, not raw execution logs.
Original execution records remain privately verifiable. No timing or result has been regenerated.

Reference source: `052104f8f9c99edde129415a7384a1a58dc24ae6`.
Candidate production source: `8f3996fea6f1f44537bee4e789b98cd6a531f347`.
Runs: M4 Pro, 20 GPU cores, 48 GiB memory capacity, macOS 15.7.1, Apple clang 17.
Model: `nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8@3fe6dab75665a93884214ad4b1b95cf02717d081`.

The actual loader reports FP8 Mamba and MLP matrices, exercising the changed row packer. Same-source Release binary identities and public checkpoint digests are in `output-summary.json`.

All 584 rows per arm (8 prefill plus 192 each step, teacher-forced verify and batch records) are finite and have identical complete bytes and inputs. Each arm contains 76,546,048 float32 elements. The report-only comparison and separate explicit zero-drift check both completed; no floor was modified. These are correctness checks, not an inference speed measurement.

The own-source default HTTP suites each completed with 49 passes, zero failures and exit code 0. Outcomes and log digests match; see `http-summary.json`.

Portable commands in the JSON use placeholders and are examples, not rewritten original logs. Run model commands under an exclusive GPU lock, setting both `NI_LOCK_HELD=1` and `NACE_GPU_LOCK_HELD=1` only while the caller actually holds it.

M3 full-model validation and native interleaved A/B/A/B checks remain separate, unfinished work. This folder establishes no complete model-load or inference speedup.
