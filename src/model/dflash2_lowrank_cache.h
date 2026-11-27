#pragma once

#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace ni {

// The factors depend on the dequantized head, including its Q4 codes. Scales and biases alone do not identify
// those weights: two packs can use the same quantization intervals with different codes inside them.
inline std::string dfl_lowrank_cache_key(const uint32_t* w, const uint16_t* sc, const uint16_t* bi,
                                       uint32_t N, uint32_t K, uint32_t r) {
  CC_SHA256_CTX ctx;
  CC_SHA256_Init(&ctx);
  auto mix = [&](const void* data, size_t n) {
    auto p = static_cast<const unsigned char*>(data);
    while (n) {
      const size_t take = std::min(n, size_t(std::numeric_limits<CC_LONG>::max()));
      CC_SHA256_Update(&ctx, p, CC_LONG(take));
      p += take;
      n -= take;
    }
  };
  const uint32_t shape[4] = {N, K, r, 1u};  // the factor algorithm's version
  mix(shape, sizeof shape);
  mix(w, size_t(N) * (K / 8) * sizeof(*w));
  mix(sc, size_t(N) * (K / 64) * sizeof(*sc));
  mix(bi, size_t(N) * (K / 64) * sizeof(*bi));
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256_Final(digest, &ctx);
  constexpr char hex[] = "0123456789abcdef";
  std::string key;
  for (unsigned char b : digest) { key += hex[b >> 4]; key += hex[b & 15]; }
  return key;
}

}  // namespace ni
