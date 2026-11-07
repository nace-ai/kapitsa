#pragma once
// Pure helpers of the MLX affine loader (no GPU): the bit width of a packed tensor from its shapes, and the float
// parameter -> bf16 conversion with the count of values that are not exactly representable. Checked on the CPU by
// tools/affine_pack_test.cpp.
#include "util/bf16.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace ni {

// Group-128 packs can reuse these group-64 kernels by duplicating each scale and bias; the packed codes stay put.
inline int affine_kernel_group_size(int bits, int gs) {
  return gs == 128 && (bits == 5 || bits == 8) ? 64 : gs;
}

// Bits per weight of an MLX affine tensor that is `words` uint32 per row, `groups` scales per row, group size gs: the
// bits b with words * 32 == groups * gs * b, or 0 when the shapes fit no whole width (a truncated or mislabelled tensor).
inline int affine_bits(int64_t words, int64_t groups, int64_t gs) {
  const int64_t limit = std::numeric_limits<int64_t>::max();
  if (words <= 0 || groups <= 0 || gs <= 0 || groups > limit / gs || words > limit / 32) return 0;
  const int64_t K = groups * gs;
  if ((words * 32) % K) return 0;
  const int64_t b = words * 32 / K;
  return b >= 2 && b <= 8 ? int(b) : 0;
}

// A sidecar's affine matrix [N, words] or stacked experts [E, N, words]. Parameters have the same leading
// dimensions and one scale and bias per group. Validate before indexing shapes, casting dimensions or copying data.
inline bool affine_matrix_shape(std::span<const int64_t> w, std::span<const int64_t> s,
                                std::span<const int64_t> b, int64_t gs) {
  if ((w.size() != 2 && w.size() != 3) || s.size() != w.size() || b.size() != w.size() || gs <= 0) return false;
  const int64_t limit = std::numeric_limits<uint32_t>::max();
  for (size_t i = 0; i < w.size(); ++i) {
    if (w[i] <= 0 || s[i] <= 0 || b[i] != s[i]) return false;
    if (i + 1 < w.size() && (w[i] != s[i] || w[i] > limit)) return false;
  }
  return s.back() <= limit / gs && affine_bits(w.back(), s.back(), gs) != 0;
}

// float32 / float16 values -> bf16 bits (round to nearest even); returns how many values changed. Every float16 value,
// subnormals included, is a normal bf16 number (bf16 has float32's exponent range), so only precision can be lost:
// float16 keeps 11 significant bits, bf16 8.
inline size_t to_bf16(const float* src, uint16_t* dst, size_t n) {
  size_t lossy = 0;
  for (size_t i = 0; i < n; ++i) {
    dst[i] = f32_to_bf16(src[i]);
    if (bf16_to_f32(dst[i]) != src[i]) ++lossy;
  }
  return lossy;
}
inline size_t to_bf16(const _Float16* src, uint16_t* dst, size_t n) {
  size_t lossy = 0;
  for (size_t i = 0; i < n; ++i) {
    const float v = float(src[i]);
    dst[i] = f32_to_bf16(v);
    if (bf16_to_f32(dst[i]) != v) ++lossy;
  }
  return lossy;
}

}  // namespace ni
