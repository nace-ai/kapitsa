// The 8-row verify kernels (src/kernels/verify8.metal) on their own, for a model that is not Qwen35: the X8 table of
// up to 8 rows (x8_prep2, an RMSNorm folded in) and the qmm8 GEMM over it, with the tiling and the inner loop
// Qwen35's 8-row verify picks (v8_plan, V8Knobs). Flint's dense Llama (Nanbeige4.3) runs its 3-8 row verify passes on them.
#pragma once
#include "metal/gpu.h"

#include <cstdint>
#include <vector>

namespace ni {

struct V8Seg {
  const Buffer *w, *s, *b;  // MLX affine codes, scales, biases (group 64)
  uint32_t N;
  Buffer y;
  uint32_t y_ld = 0;  // 0: N
  bool add = false;   // y += X W^T
};

// The X8 table and group sums of rows x [R][K] (x_ld apart; 0: K), each scaled by its RMSNorm with weight `norm` when
// it is given. K a multiple of 256, R <= 8.
void v8_enc_x8(Encoder& e, const Buffer& x, int R, int K, const Buffer* norm, float eps, const Buffer& table,
               const Buffer& sums, const Buffer* table_t = nullptr);
// Whether this Mac runs the 4-bit GEMMs on the matrix units (qmm8t, src/kernels/tensor/verify8t.metal: an Apple10 GPU on
// macOS 26, V8Knobs::tw): then v8_enc_x8 also writes table_t (K * 16 bytes) and v8_enc_qmm8 reads it for 4-bit matrices
bool v8_tensor_ok();
// y_i[R, N_i] = X W_i^T (+ y_i) for up to 4 matrices of one format (4 or 8 bits, group 64) sharing K, from the table
void v8_enc_qmm8(Encoder& e, const std::vector<V8Seg>& segs, int R, uint32_t K, int bits, const Buffer& table,
                 const Buffer& sums, const Buffer* table_t = nullptr);

}  // namespace ni
