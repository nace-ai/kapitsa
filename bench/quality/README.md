# Quantization quality and speed

Tools to measure how far a quantized model's next-token distributions are from the bf16 model, and how fast each
precision runs: the measurements behind the default precisions (4-bit by default, 6-bit as the quality choice).

- `ids_qwen35/`, `ids_qwen38/`: the held-out token sequences (1024 tokens each: C++, Python, English technical prose, a
  coding-agent chat with tool calls rendered with the model's chat template, and `math`: ten level-5 MATH-500 problems
  with their reference solutions as a chat; `ids_qwen35` for Qwen3.5 / 3.6, `ids_qwen38` for Qwen3.8, whose template
  differs). The published tables average the first four (4 x 1023 positions); `math` is reported on its own
  (`eval_stream.py --aside math`). `make_heldout.py <model dir> [out]` rebuilds them from `src/` (`src/math.json`
  names the problems: MATH, Hendrycks et al. 2021, MIT license).
- `../../build/ni_eval <model> [--ref-model M | --ref F | --dump-ref F] @ids...`: Kapitsa's teacher-forced
  next-token distributions; KL(reference || model), top-1 agreement, perplexity.
- `eval_mlx.py`: the same with MLX (mlx-lm) for a local model, the same reference file format; `--f32` runs MLX in
  fp32 (no bf16 activation rounding), `--greedy N` prints greedy ids with each step's top-2 margin.
- `eval_stream.py`: models whose bf16 checkpoint fits neither memory nor disk: streams it from Hugging Face one
  decoder layer at a time and runs every candidate quantization beside it (round-to-nearest recipes made from the
  bf16 layer, or published packs streamed from their repos); exact KL over the vocabulary at the head. `--f32`
  (used for every published number) computes both sides with fp32 activations: MLX's bf16 activations add rounding
  noise of the size of an 8-bit quantization's error. `--local DIR` takes the bf16 weights from a folder instead.
  Recipes can set any layer range or part (`rtn:4,L56-63=8`, `rtn:bf16,L5.mix=4`: the docstring has the grammar), and
  candidates that agree on the layers so far share one hidden state, so per-part sensitivity forks (the BF16 model with
  one part 4-bit) cost one pass from their layer on: one streaming run scores ~150 variants of the 27B. Quantized heads
  are applied as fp32 quantized matmuls (exact); before 2026-10-19 they were dequantized with their bf16 scales, which
  rounds each dequantized head weight to bf16. `--positions` saves per-position KL, top-1 and NLL for every
  candidate.
- `eval_qwen4_exp.py`: the same for Qwen3.8-Flash-Next (qwen4_exp, 360 GB bf16) with mlx-vlm's modules, one sequence
  at a time (mlx-vlm's batched QSA prefill is wrong for B > 1), fp32 with `MLX_ENABLE_TF32=0` (MLX otherwise runs fp32
  GEMMs as TF32 on the M5, 8e-4 relative error per GEMM; this applies to `eval_stream.py --f32` too); also the
  streaming pass of `tools/convert_qwen4_exp.py` (experts staged 4-bit, the rest bf16) and `local` scoring of recipes
  and of the finished pack (reference files for `ni_eval --ref`, per-layer hidden states).
- `check_rtn.py`: whether a published MLX pack is exactly round-to-nearest of the bf16 weights (one layer; `--gpu`
  quantizes on the GPU as `mlx_lm.convert` does, the CPU rounds ~1% of the codes differently).
- `fetch_text.py`: downloads a repo's text-model tensors only (parallel range requests).
- `mixed_pack.py`: a mixed-precision pack for measurements: APFS clones of a local pack plus the chosen modules of
  another precision's published pack (range requests) in one more shard, with MLX per-module quantization entries.
- `speed_variants.py`: ni_run decode (DFlash2 at T = 0.6 and greedy, plain) and 4k prefill of several variants,
  interleaved: the protocol of the default-precision rule.
- `speed_mlx.py`: MLX prefill and decode speed of a local model (the yardstick for Kapitsa's numbers).
- `table.py`: markdown tables from the results.

GPU work takes `~/.cache/nace-gpu.lock` (as the other benchmarks); `NI_LOCK_HELD=1` when the caller holds it.
