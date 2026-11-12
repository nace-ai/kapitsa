// Which kernels the Neural Engine split's GPU part runs a dense FFN layer on, from its gate, up and down formats. The one
// rule for three callers: plan() (from the safetensors shapes) and init_ane() pick the matrix-unit default share
// (ffn_mma) for a model whose every layer has a matrix-unit path, and enc_mlp_gpu_part_qmt dispatches by it, so the
// planned share, the loaded share and the kernels that run agree.
#pragma once

#include <cstdint>
#include <vector>

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

// NI_QMTF's formats this process runs on the tensor tile (Qwen35::qmtf_formats)
struct Qmtf {
  bool f3g64 = false, f8g64 = false, f4g32 = false, f5g64 = false;
  bool has(Fmt f) const {
    return (f == Fmt{3, 64} && f3g64) || (f == Fmt{8, 64} && f8g64) || (f == Fmt{4, 32} && f4g32) ||
           (f == Fmt{5, 64} && f5g64);
  }
};

enum class Path {
  generic,  // the simdgroup kernels: the layer keeps the older default share
  qmt,      // gate, up and down all 4-bit group 64: qmtg / qmt_ld (src/kernels/tensor/qmt.metal)
  qmtf,     // gate and up in one NI_QMTF format, down in any: qmtfg / qmtf_ld (qmtf.metal)
};

// One layer's GPU part. A GGUF Q4_K_M import is qmtf (Q4_K gate/up -> 4-bit g32, Q6_K down -> 8-bit g64); a mix
// with a 4-bit g64 or 6-bit projection (mlx-lm's mixed recipes: 3-bit gate/up, 4-bit or 6-bit down) is generic.
inline Path gpu_part(Fmt gate, Fmt up, Fmt down, Qmtf q) {
  if (gate == Fmt{4, 64} && up == gate && down == gate) return Path::qmt;
  if (q.has(gate) && up == gate && q.has(down)) return Path::qmtf;
  return Path::generic;
}

// a dense FFN layer's gate, up and down formats
struct LayerFmt {
  Fmt gate, up, down;
};

// The split's default share for the model (plan()'s and init_ane()'s ffn_mma): the matrix-unit share when every layer's
// GPU part runs on the matrix units. One layer on the generic kernels keeps the older share for the whole model: the
// share is one row split for every layer, and the generic kernels' layer would take the matrix-unit share's GPU rows
// at the simdgroup kernels' speed.
inline bool matrix_unit_share(const std::vector<LayerFmt>& layers, Qmtf q) {
  if (layers.empty()) return false;
  for (const LayerFmt& l : layers)
    if (gpu_part(l.gate, l.up, l.down, q) == Path::generic) return false;
  return true;
}

}  // namespace ni::ffn_share
