# CPU draft-row affine decoding

Reference source: `f92f4215`. Implementation: `77b8464b`.

The scale and bias are converted once per group rather than once per output element. The bit extraction and
floating-point expression are unchanged. This folder measures CPU row decoding, not model inference.

## Reproduce the CPU comparison

```sh
python3 bench/results/2026-11-06-draft-row-dequant/reproduce.py \
  --reference f92f4215 --candidate 77b8464b \
  --model /path/to/qwen3.5-4b-4bit/model.safetensors --out /tmp/row-comparison-new
```

The output directory must be new. The script extracts `dfl_row_bf16` from the two exact Git revisions and compiles
separate functions with Apple Clang/C++20/`-O3`. The driver uses CPU-only Buffer views and supplies the ordinary
MLX affine row-byte calculation; IQ and EXL3 tensors are not accepted. It compares every decoded BF16 value in
three complete real matrices, saves SHA256 digests, and records four reference/candidate timing pairs. File reads,
comparisons and hashes are outside timing. Source revisions, source hashes and the executable hash are recorded.

Timing takes the project GPU lock to avoid competing engine jobs. `--check-only` skips timing and the lock;
neither mode initializes Metal. The source checkout, macOS Command Line Tools and Python's standard library suffice.
The committed `dflash_row_test` separately calls the linked production implementation on independently packed
2–8-bit rows, dense BF16 copies, Buffer offsets and guards; CI runs that test.

## Exercise a caller that actually converts the draft head

The ordinary 4-bit head does not need conversion. To exercise the affected caller without claiming a native
6-bit checkpoint, the following script widens the real head's Q4 codes to Q6, preserving every code value, scale,
bias and other tensor. The tokenizer and drafter link to the original model. This optional fixture-generation
step uses NumPy, as the repository's existing conversion tools do; it is not a runtime dependency.

```sh
python3 bench/results/2026-11-06-draft-row-dequant/make_head_fixture.py \
  --source /path/to/qwen3.5-4b-4bit --out /tmp/4b-expanded-q6-head-new
```

The fixture folder must be new. Its `fixture.json` explicitly identifies it as a derived test fixture. The script
decodes every widened code to check equality before writing it. Build both revisions with an explicit CMake
configure, then run `qmm_draft_check MODEL PROMPT_IDS NEW_OUTPUT` for the reference and use `--from REFERENCE_OUTPUT`
for the candidate. Use the frozen `prompt.ids` in this folder. The tools acquire the GPU lock themselves. Confirm
both logs say the 98,304-row draft head is an **8-bit copy of the 6-bit** LM head, then inspect the comparison of
all 270 saved buffers. A separate pair of `logit_dump` runs and `logit_compare.py` checks model outputs.

The standard 4B logit/HTTP/benchmark runs are additional regression checks. They alone do not exercise the mixed
head conversion. A derived Q6 fixture is not evidence for an independently quantized native 6/8-bit model pack.

## Recorded results

All real CPU outputs matched. The actual derived-Q6 conversion produced 270 identical buffers, and its 584 logit rows were bitwise identical. Full CPU, supplemental tests, parser/tokenizer and prek passed. Stock 4B logits were 584/584 rows bitwise identical.

| Fixture / matrix | N x K | Reference ms (4 runs) | Candidate ms (4 runs) | Median reduction |
|---|---|---|---|---:|
| stock: language_model.model.embed_tokens | 248320 x 2560 | 697.852, 699.071, 705.635, 701.055 | 576.966, 586.155, 578.033, 575.803 | 17.5% |
| stock: language_model.model.layers.0.mlp.down_proj | 2560 x 9216 | 25.455, 26.082, 25.827, 25.860 | 21.226, 21.287, 21.375, 21.326 | 17.6% |
| stock: language_model.model.layers.0.mlp.gate_proj | 9216 x 2560 | 25.609, 25.976, 25.866, 25.928 | 21.300, 21.314, 21.304, 21.306 | 17.7% |
| expanded Q6 head: language_model.model.embed_tokens | 248320 x 2560 | 707.629, 709.200, 723.817, 698.587 | 589.106, 590.036, 588.068, 593.382 | 16.8% |
| expanded Q6 head: language_model.model.layers.0.mlp.down_proj | 2560 x 9216 | 25.380, 25.867, 25.853, 25.860 | 20.921, 21.199, 21.138, 21.100 | 18.3% |
| expanded Q6 head: language_model.model.layers.0.mlp.gate_proj | 9216 x 2560 | 25.260, 25.869, 25.893, 25.900 | 20.878, 21.144, 21.048, 21.000 | 18.8% |

| Run | Sampled tok/s | Greedy tok/s | 4K TTFT s | Follow-up s | Flags |
|---|---:|---:|---:|---:|---|
| bench-05-base | 316.69 | 365.07 | 1.9390 | 0.0314 | [] |
| bench-06-row | 331.61 | 375.85 | 1.9365 | 0.0302 | [] |
| bench-07-base | 341.32 | 356.48 | 1.9373 | 0.0325 | ['decode: repetition loop (distinct-4-gram 0.4284)'] |
| bench-08-row | 336.95 | 380.10 | 1.9361 | 0.0337 | [] |
| bench-09-base-expanded-q6 | 328.32 | 358.32 | 1.9350 | 0.0351 | [] |
| bench-10-row-expanded-q6 | 324.48 | 329.62 | 1.9370 | 0.0357 | [] |
| bench-11-base-expanded-q6 | 309.40 | 360.17 | 1.9360 | 0.0406 | [] |
| bench-12-row-expanded-q6 | 352.49 | 356.81 | 1.9361 | 0.0365 | [] |

The sampled benchmark has no fixed seed. Repetition flags occurred in bench-07-base, and every flagged run is retained. The fixed-input logit checks establish output equality; these sampled benchmarks do not establish a decode-throughput gain.

Both baseline and candidate passed 47 HTTP tests and failed the same two existing timing checks: the long counting reply ended at 693 tokens before the delayed side request, and the slot holder ended before a second 15-second keepalive. The full failure records are retained; this is not an all-pass HTTP claim.

`evidence.json` contains exact revisions, source hashes, build/version logs, CPU measurements and numerical reports. The separate benchmark JSONs retain all flags and invalid lists. CPU helper improvements are not full-model inference gains. The coverage limits above still apply.
