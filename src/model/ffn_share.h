// Which dense FFN layers the Neural Engine split's GPU share runs on the matrix units (enc_mlp_gpu_part_qmt).
// plan() decides it from the safetensors shapes and init_ane() from the loaded QLinears; both call layer_mma, so the
// planned and the loaded default share agree with what the GPU part then runs.
#pragma once

#include <cstdint>

namespace ni::ffn_share {

// an MLX affine projection's format: bits per weight and group size (0 / 0: not an affine U32 pack)
struct Fmt {
  int bits = 0, gs = 0;
  bool operator==(const Fmt& o) const { return bits == o.bits && gs == o.gs; }
};

// The format from an affine pack's shapes: K inputs, w_cols U32 words and s_cols scale groups per output row.
inline Fmt from_shape(int64_t K, int64_t w_cols, int64_t s_cols) {
  if (K <= 0 || w_cols <= 0 || s_cols <= 0 || (w_cols * 32) % K || K % s_cols) return {};
  return {int(w_cols * 32 / K), int(K / s_cols)};
}

// NI_QMTF's formats this process runs on the tensor tile (qmtf_on's answers)
struct Qmtf {
  bool f3g64 = false, f8g64 = false, f4g32 = false;
  bool has(Fmt f) const {
    return (f == Fmt{3, 64} && f3g64) || (f == Fmt{8, 64} && f8g64) || (f == Fmt{4, 32} && f4g32);
  }
};

// One layer's gate, up and down: the matrix-unit GPU part runs it only when all three share one format, and that
// format is 4-bit group 64 (qmt) or one of NI_QMTF's. A mix (say 3-bit gate/up with a 4-bit g64 down, as mlx-lm's
// mixed recipes make) runs the generic GPU part, so it keeps the older default share.
inline bool layer_mma(Fmt gate, Fmt up, Fmt down, Qmtf q) {
  if (!(gate == up) || !(gate == down)) return false;
  return gate == Fmt{4, 64} || q.has(gate);
}

}  // namespace ni::ffn_share
