#pragma once
// Pure helpers of the MLX affine loader (no GPU): the bit width of a packed tensor from its shapes, and the float
// parameter -> bf16 conversion with the count of values that are not exactly representable. Checked on the CPU by
// tools/affine_pack_test.cpp.
#include "util/bf16.h"

#include <cstddef>
#include <cstdint>

namespace ni {

// Bits per weight of an MLX affine tensor that is `words` uint32 per row, `groups` scales per row, group size gs: the
// bits b with words * 32 == groups * gs * b, or 0 when the shapes fit no whole width (a truncated or mislabelled tensor).
inline int affine_bits(int64_t words, int64_t groups, int64_t gs) {
  if (words <= 0 || groups <= 0 || gs <= 0) return 0;
  const int64_t K = groups * gs;
  if ((words * 32) % K) return 0;
  const int64_t b = words * 32 / K;
  return b >= 2 && b <= 8 ? int(b) : 0;
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
