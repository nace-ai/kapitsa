# Quantization quality and speed

Tools to measure how far a quantized model's next-token distributions are from the bf16 model, and how fast each
precision runs: the measurements behind the default precisions (4-bit by default, 6-bit as the quality choice).

- `ids_qwen35/`, `ids_qwen38/`: the held-out token sequences (4 x 1024 tokens: C++, Python, English technical prose, a
  coding-agent chat with tool calls rendered with the model's chat template; `ids_qwen35` for Qwen3.5 / 3.6,
  `ids_qwen38` for Qwen3.8, whose template differs). `make_heldout.py <model dir> [out]` rebuilds them from `src/`.
- `../../build/ni_eval <model> [--ref-model M | --ref F | --dump-ref F] @ids...`: Kapitsa's teacher-forced
  next-token distributions; KL(reference || model), top-1 agreement, perplexity.
- `eval_mlx.py`: the same with MLX (mlx-lm) for a local model, the same reference file format; `--f32` runs MLX in
  fp32 (no bf16 activation rounding), `--greedy N` prints greedy ids with each step's top-2 margin.
- `eval_stream.py`: models whose bf16 checkpoint fits neither memory nor disk: streams it from Hugging Face one
  decoder layer at a time and runs every candidate quantization beside it (round-to-nearest recipes made from the
  bf16 layer, or published packs streamed from their repos); exact KL over the vocabulary at the head. `--f32`
  (used for every published number) computes both sides with fp32 activations: MLX's bf16 activations add rounding
  noise of the size of an 8-bit quantization's error. `--local DIR` takes the bf16 weights from a folder instead.
- `check_rtn.py`: whether a published MLX pack is exactly round-to-nearest of the bf16 weights (one layer; `--gpu`
  quantizes on the GPU as `mlx_lm.convert` does, the CPU rounds ~1% of the codes differently).
- `fetch_text.py`: downloads a repo's text-model tensors only (parallel range requests).
- `speed_mlx.py`: MLX prefill and decode speed of a local model (the yardstick for Kapitsa's numbers).
- `table.py`: markdown tables from the results.

GPU work takes `~/.cache/nace-gpu.lock` (as the other benchmarks); `NI_LOCK_HELD=1` when the caller holds it.
