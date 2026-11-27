#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
void reference_pack(const uint8_t* codes, size_t n, int bits, uint32_t* words) {
  if (bits == 8) {
    memcpy(words, codes, n);  // byte j of the little-endian stream is code j
    return;
  }
  if (bits == 4) {
    for (size_t w = 0; w < n / 8; ++w) {
      uint32_t v = 0;
      for (int k = 0; k < 8; ++k) v |= uint32_t(codes[8 * w + k] & 0xF) << (4 * k);
      words[w] = v;
    }
    return;
  }
  const size_t nw = n * size_t(bits) / 32;
  std::fill(words, words + nw, 0u);
  const uint32_t mask = (1u << bits) - 1u;
  for (size_t j = 0; j < n; ++j) {
    const uint64_t bit = uint64_t(j) * uint64_t(bits);
    const uint32_t c = codes[j] & mask, s = uint32_t(bit % 32);
    words[bit / 32] |= c << s;
    if (s + uint32_t(bits) > 32) words[bit / 32 + 1] |= c >> (32 - s);
  }
}
