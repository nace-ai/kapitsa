# PR #775: M3 Ultra / Qwen3.5-9B supplementary evidence

W8A8 changes outputs. Loaded request latency improved, but the only ANE-exposed
Math500 question regressed. This does not establish quality equivalence or pass a
formal 9B numerical floor. No floor was added or relaxed.

Runs: M3 Ultra, 28 CPU / 60 GPU cores, 96 GB, macOS 26.4.1 (25E253), SDK 26.2.
Kapitsa 0.2.31, Release; reference `2129b4e89c9cea1240bfbee81c9bcaab5156db96`,
candidate `30325c9bd09ddd9f46d4fb070b754483cfb5de24`.
Subsequent evidence/docs commits do not change tested production sources.

## Configuration

Model: `lmstudio-community/Qwen3.5-9B-MLX-4bit` at
`b455506b0f574c74616dbcd56879bde38fafcff3` (4-bit group 64).
Independent reference: `Qwen/Qwen3.5-9B` at
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, bf16 weights with MLX `--f32` activations.
Weight shards/tokenizers match upstream hashes; identity JSONs include package versions.

A is main with `NI_ANE=0`; B is candidate with `NI_ANE_UNITS=16`. Chunk 2048,
no drafter or disk prefix cache. Exact settings are recorded in logs/requests.
M3 has no built-in default ANE share. A prior 4/8/12/16/20/24-unit sweep selected
16 before quality evaluation. This explicitly configured run does not measure
shipping M3 defaults. No global calibration was installed. All GPU/model runs
held `/tmp/kapitsa-gpu.lock` exclusively.

The candidate extends eligible hidden widths to multiples of 2048 up to 8192,
retaining precedence of the existing 2560 geometry. Existing chip, format,
memory, chunk and startup checks still apply. This run covers only width 4096.

## Quality and coverage

Candidate opt-out is bitwise identical to main on all 584 full-vocabulary logit
rows. ANE recorded 576 evaluations and zero reruns, with nonzero drift:

| Path | Top-1 flips | Mean KL |
| --- | ---: | ---: |
| Prefill | 0/8 | 1.846e-4 |
| Step | 1/192 | 2.579e-4 |
| Verify | 1/192 | 2.666e-4 |
| Batch | 1/192 | 2.591e-4 |

The continuation flips share sequence 0 / prefix 3210 and are not independent.
Explicit zero-bound checks fail. W8A8 changes precision, so `--reorder` does not apply.

Math500 quick: 50 questions, thinking on, full 32768-token response budget,
context 40960, fixed sampling/seeds, one slot. GPU 37/50 (74%), ANE 36/50 (72%).
Paired difference -2 percentage points, bootstrap 95% interval [-6, 0]; no
claim of equivalence. GPU/ANE had 12/13 truncated or no-answer responses and
both had 32 looping responses under the harness metric.

**49/50 prompts were below the 512-token ANE threshold; their full responses were
identical. The sole exposed prompt (857 tokens), `test/prealgebra/1044.json`,
changed from correct `5.4` (25944 tokens, stop) to no answer (32768 tokens, length).**
Server evidence confirms one split chunk and 64 ANE evaluations. This adverse
example must not be obscured by the overall score; exposed reasoning coverage is inadequate.

Independent bf16 quality used unmodified `bench/quality/eval_mlx.py` and `ni_eval`:

| Corpus / positions | bf16 PPL | GPU PPL | ANE PPL | GPU mean KL | ANE mean KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| C++, Python, prose, chat / 4092 | 6.05750 | 6.41118 | 6.42738 | 0.114717 | 0.115258 |
| Held-out math / 1023 | 3.00402 | 3.11591 | 3.10860 | 0.0999068 | 0.100894 |

KL uses top-256 plus remaining mass: a lower bound, not exact full-vocabulary KL.
Primary top-1 agreement with bf16 is 0.88123 / 0.88074 (GPU/ANE); primary PPL
increases about 0.253%. Lower math PPL does not demonstrate preserved reasoning.
Per-domain logs are included.

## Loaded request latency: A B A B

Repository HTTP benchmark harness, fixed 4135-token prompt, zero cached tokens,
fresh server/home each leg, warmup before quiet wait, seed 20261106.

| Leg | Seconds |
| --- | ---: |
| A1 | 3.537277250 |
| B1 | 2.678795792 |
| A2 | 3.550766583 |
| B2 | 2.692343084 |

Mean 3.544022 -> 2.685569 seconds: 24.22% reduction. Compilation/startup excluded.
This is supplementary HTTP evidence, not a community `kapitsa bench` submission
or a decode throughput claim. A/B replies differ. All three request hashes match
across legs; timed request SHA256:
`4f788880c578359e9eadec51adcb22785591c0e06b82ae1a0ca3e73961357538`.

Harness invalid/flag lists were empty; quiet checks passed and monitored foreign
GPU use was zero. Limit: 2-second telemetry has two samples for A1 and one for
each other leg; helper excludes WindowServer and labels three ambient daemons.
This is not proof of strict all-process idle. Two pairs do not give a robust CI.

## Validation and remaining requirements

- Both Release builds succeeded; CPU suite 21 named tests passed, zero failures.
- `prek run --all-files` passed for tested candidate.
- `ane_segment_check --gpu` with Metal validation: 50 CPU checks, 2098 GPU
  comparisons, 34,105,984 data/guard bytes, zero failures.
- Official abort/fallback harness: 17/17 passed. An external binary wrapper sets
  units16 because the harness filters NI_* variables; harness source was unchanged.

The PR is not demonstrated merge-ready under current output/default policy:
the numerical gate is unresolved, the exposed reasoning example regressed,
and the targeted class requires a second setup. Historical measurements are not
a waiver. Further long-prompt reasoning evidence and a maintainer precision-policy
decision are needed. An opt-in flag or freshly relaxed floor is not a substitute.

## Evidence retention

This directory contains compact raw logs, identities and performance requests.
`source-manifest.json` records source/binary identities and hashes of the complete
149-file collection, including controllers, all math replies and server logs.
The contributor retains and delivers `review-evidence.tar.gz` with the reply draft;
weights, binaries and large raw arrays remain remote and are excluded from it.
Manifest paths are archive-relative; not all entries are in this compact directory.
All original results, including failed comparisons, are retained unchanged.
