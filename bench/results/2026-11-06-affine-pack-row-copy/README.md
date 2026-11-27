# CPU Q4 packed-image construction

Reference source: `f92f4215`. Implementation: `f34582bc`.

This change copies each source row in two contiguous four-word spans into the existing SIMD execution layout.
The coefficient codec and the kernels do not change. This folder's CPU measurements cover construction of the
packed image, including its allocation and coefficient preparation; they are not inference measurements.

## Reproduce the CPU comparison

On the PR checkout, with the original 4B `model.safetensors` already downloaded:

```sh
python3 bench/results/2026-11-06-affine-pack-row-copy/reproduce.py \
  --reference f92f4215 --candidate f34582bc \
  --model /path/to/qwen3.5-4b-4bit/model.safetensors --out /tmp/pack-comparison-new
```

The output directory must be new. The script reads the exact helper from each Git revision, compiles separate
wrappers with Apple Clang/C++20/`-O3`, compares every packed word and header on three complete real matrices,
then records four reference/candidate pairs after warming both implementations. File reads, comparisons and
SHA256 calculations are outside the timing. The candidate helper is renamed to prevent weak inline definitions
from making both wrappers call one implementation. The script records source and executable hashes.

Timing takes the project's GPU lock to avoid competing engine jobs. `--check-only` skips timing and the lock;
neither mode initializes Metal. Both require only the source checkout, macOS Command Line Tools and Python's
standard library. No model is downloaded.

## GPU and model checks

Build the normal Release tools at each recorded source revision, explicitly rerunning CMake configure for each
revision. Run `qmm_packed_coeff_check check-small` and `check-codec` with both `MTL_SHADER_VALIDATION=1` and
`MTL_DEBUG_LAYER=1`. These tools acquire the project GPU lock themselves. CPU codec coverage is
`qmm_packed_coeff_check check-cpu-codec`; it initializes no GPU.

The M3 Ultra runs the changed packed images in these direct checks. Its default model path does not enable the
M4-gated packed layout, so the stock 4B logit/HTTP/benchmark runs are regression checks. They do not establish
an M4 full-model loading or inference gain. The same applies to the CPU function timing above.

## Recorded results

All three real packed images matched; direct packed-kernel checks passed with both Metal validation layers. Full CPU, supplemental tests, parser/tokenizer and prek passed. Stock 4B logits were 584/584 rows bitwise identical.

| Fixture / matrix | N x K | Reference ms (4 runs) | Candidate ms (4 runs) | Median reduction |
|---|---|---|---|---:|
| stock: language_model.model.embed_tokens | 248320 x 2560 | 73.199, 72.963, 72.465, 71.814 | 45.188, 44.583, 43.812, 44.067 | 39.0% |
| stock: language_model.model.layers.0.mlp.down_proj | 2560 x 9216 | 1.924, 1.917, 1.990, 2.049 | 1.071, 1.059, 1.149, 1.101 | 44.5% |
| stock: language_model.model.layers.0.mlp.gate_proj | 9216 x 2560 | 2.127, 2.071, 2.018, 1.997 | 1.091, 1.058, 1.146, 1.130 | 45.7% |

| Run | Sampled tok/s | Greedy tok/s | 4K TTFT s | Follow-up s | Flags |
|---|---:|---:|---:|---:|---|
| bench-01-base | 354.70 | 372.67 | 1.9398 | 0.0316 | [] |
| bench-02-pack | 326.08 | 359.24 | 1.9393 | 0.0386 | ['decode: repetition loop (distinct-4-gram 0.452)'] |
| bench-03-base | 342.57 | 370.30 | 1.9397 | 0.0305 | [] |
| bench-04-pack | 328.62 | 367.69 | 1.9380 | 0.0314 | [] |

The sampled benchmark has no fixed seed. Repetition flags occurred in bench-02-pack, and every flagged run is retained. The fixed-input logit checks establish output equality; these sampled benchmarks do not establish a decode-throughput gain.

Both baseline and candidate passed 47 HTTP tests and failed the same two existing timing checks: the long counting reply ended at 693 tokens before the delayed side request, and the slot holder ended before a second 15-second keepalive. The full failure records are retained; this is not an all-pass HTTP claim.

`evidence.json` contains exact revisions, source hashes, build/version logs, CPU measurements and numerical reports. The separate benchmark JSONs retain all flags and invalid lists. CPU helper improvements are not full-model inference gains. The coverage limits above still apply.
