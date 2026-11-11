#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace gguf_test {
// Exceptional scales may produce NaNs with different signs or payloads. Everything else,
// including signed zeros and infinities, must match byte for byte. Integer inspection keeps
// the decoder's floating-point exception flags intact, even for signaling NaNs.
inline bool same_f32_or_nan(const void* a, const void* b, size_t count) {
  const auto* x = static_cast<const uint8_t*>(a);
  const auto* y = static_cast<const uint8_t*>(b);
  for (size_t i = 0; i < count; ++i) {
    uint32_t u, v;
    memcpy(&u, x + i * sizeof(u), sizeof(u));
    memcpy(&v, y + i * sizeof(v), sizeof(v));
    if (u != v && !((u & 0x7fffffff) > 0x7f800000 && (v & 0x7fffffff) > 0x7f800000)) return false;
  }
  return true;
}
}  // namespace gguf_test
