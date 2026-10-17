# Acknowledgments and provenance

Kapitsa is developed and maintained by Nace AI. It incorporates third-party libraries, adapted interface and
kernel code, and techniques informed by other inference engines. Those components retain their respective
licenses. This document records credit; it does not imply endorsement by the projects or their authors.

## Splash (Inco), Apache-2.0

[Splash](https://github.com/incoai/splash) is a reference for the GPU/Neural Engine split and several
speculative-decoding optimizations. The mapping below uses the upstream revision examined for this attribution
review, [`be3b083c5f413673d44e44881042e875e674e333`](https://github.com/incoai/splash/tree/be3b083c5f413673d44e44881042e875e674e333).
It is a reference revision, not a claim about the exact revision originally consulted during development.

| Kapitsa component | Upstream reference and contribution |
| --- | --- |
| `src/ane/ane.mm` | `runtime/ane/Program.mm`: adapted private AppleNeuralEngine interface declarations and method-signature table. Kapitsa uses its own protocol names, API resolution and service-queue/watchdog integration. |
| `src/ane/ane.h`, `src/ane/ane.mm` | `runtime/ane/Program.mm`, `runtime/ane/Handoff.mm`: interface validation, IOSurface-backed programs, GPU/ANE synchronization and watchdog design. |
| `src/ane/ffn_split.{h,cpp}`, `src/model/qwen35_ane.cpp` | `runtime/ops/AneFfn.cpp`: feed-forward channel split, partial down-projection join, Hadamard rotations, scale layout and staging weights ahead of execution. |
| `src/kernels/ane_ffn.metal` | `runtime/metal/kernels/prefill/ane_ffn.metal`: adapted rotation helper (`ane_ffn_rotate_block` upstream, `ane_rotate_block` here) and GPU-side packing/staging design. Changes to the helper are identified in its file header. |
| `src/kernels/verify8.metal` | `runtime/metal/kernels/decode/linear_gguf_sgmatrix.metal`: exact integer-code matrix operands with a per-group scale/bias epilogue. Kapitsa applies the technique to MLX-layout weights and half activations. |
| `src/model/dflash2.cpp`, `src/kernels/spec_dflash.metal` | Drafter quantization at load, per-row draft-attention window, exact-code matrix operands and separate block K/V scratch, credited in the source. |

The two closely corresponding interface/helper portions are credited as adaptations. The other entries identify
referenced techniques; they do not assert that every implementation in those files was copied from Splash.
Splash's Apache-2.0 license is the same license text supplied in Kapitsa's `LICENSE`. See
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES) and retain the per-file attribution when distributing modified sources.

## MLX (Apple), MIT

[MLX](https://github.com/ml-explore/mlx) supplies reference designs for the attention tile layout and scheduling
in `src/kernels/attn.metal`, matrix-fragment layouts used by the Metal kernels, and the affine quantization layout
read by Kapitsa. Its quantization behavior is also a reference for `src/model/dflash2.cpp`. The relevant
comparisons are recorded in [docs/PERF.md](docs/PERF.md), and the source comments identify these references.

Kapitsa does not link to the MLX runtime. Reading MLX-format checkpoints is separate from using MLX as an
execution engine. We acknowledge the reference work and include MLX's MIT license notice in
`THIRD_PARTY_NOTICES`. The review reference is
[`b6642e604843776705b57b1eaecb6f57e519b793`](https://github.com/ml-explore/mlx/tree/b6642e604843776705b57b1eaecb6f57e519b793),
not an assertion of the original development revision.

## Models and draft models

The target models are from [Qwen](https://huggingface.co/Qwen). The catalog uses conversions and MTP packs from
[mlx-community](https://huggingface.co/mlx-community), [LM Studio](https://huggingface.co/lmstudio-community) and
[MTPLX contributors](https://huggingface.co/Youssofal). The default DFlash2 draft models are published by
[Inco](https://huggingface.co/incoai).

Model weights are downloaded separately and keep their own licenses. The repository IDs and pinned revisions
are recorded in `src/hub/catalog.cpp` and [DEVELOPMENT.md](DEVELOPMENT.md#models-and-downloads). Nace AI's license
for the engine does not replace the licenses for any target model, conversion, MTP head or draft model.

## Libraries and data

- cpp-httplib, minja and nlohmann/json are vendored under MIT licenses, with notices retained in `third_party/`.
- The tokenizer's Unicode tables use the Unicode Character Database under Unicode License V3.
- The benchmark corpus includes Hugging Face Transformers excerpts under Apache-2.0, with individual headers
  retained and details in `bench/harness/corpus/NOTICE.md`.

The notices for these components are collected in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).

## Distribution

Include `LICENSE`, `THIRD_PARTY_NOTICES` and this document with released binaries. The source installer places
them in `../share/kapitsa` relative to its binary destination; the Homebrew formula uses its package share
directory. Source distributions must also retain the attribution and modification notices in individual files.
When adding or adapting third-party code, record its upstream source, license and changes here and in the file.
