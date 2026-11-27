// Packed-code bytes checked independently, including masking, overlap order and page boundaries.
#pragma once

#include "gguf/quant.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

namespace pack_codes_test {
inline void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}
inline std::vector<uint8_t> oracle(const uint8_t* q, size_t n, int bits) {
  std::vector<uint8_t> out(n * bits / 8);
  for (size_t j = 0; j < n; ++j)
    for (int b = 0; b < bits; ++b)
      if (q[j] & (1u << b)) out[(j * bits + b) / 8] |= 1u << ((j * bits + b) % 8);
  return out;
}
inline void check(const std::vector<uint8_t>& input, size_t offset, size_t n, int bits) {
  const auto copy = input;
  const auto expected = oracle(input.data() + offset, n, bits);
  std::vector<uint32_t> out(n * bits / 32 + 2, 0xabababab);
  ni::gguf::pack_codes(input.data() + offset, n, bits, out.data() + 1);
  require(out.front() == 0xabababab && out.back() == 0xabababab, "packed output guards");
  require(expected.empty() || !memcmp(out.data() + 1, expected.data(), expected.size()), "packed byte oracle");
  require(copy == input, "packed source unchanged");
}
inline void overlap_reference(uint8_t* source, size_t n, uint32_t* words) {
  // Preserve the existing zero-before-read and left-to-right semantics when the buffers overlap.
  std::fill(words, words + n * 5 / 32, 0u);
  for (size_t j = 0; j < n; ++j) {
    const uint8_t code = source[j] & 31;
    for (size_t b = 0; b < 5; ++b)
      if (code & (1u << b)) words[(j * 5 + b) / 32] |= uint32_t(1) << ((j * 5 + b) % 32);
  }
}
inline int run() {
  try {
    std::mt19937 rng(34797);
    size_t cases = 0;
    std::vector<uint8_t> q(32);
    for (size_t lane = 0; lane < 32; ++lane) for (int v = 0; v < 256; ++v) {
      std::fill(q.begin(), q.end(), 0); q[lane] = uint8_t(v);
      check(q, 0, 32, 5); ++cases;
    }
    for (int bits : {2, 3, 4, 5, 6, 8}) for (size_t n : {0, 32, 64, 96, 256, 512, 2560, 3136, 4096, 8192})
      for (size_t offset = 0; offset < 16; ++offset) {
        q.resize(n + offset + 16); for (auto& x : q) x = uint8_t(rng());
        check(q, offset, n, bits); ++cases;
      }
    for (size_t n : {32, 64, 96, 256, 512}) for (size_t src = 32; src < 96; ++src)
      for (size_t dst = 0; dst < 128; dst += 4) {
        std::vector<uint32_t> a(512), b; for (auto& x : a) x = rng(); b = a;
        overlap_reference(reinterpret_cast<uint8_t*>(a.data()) + src, n, a.data() + dst / 4);
        ni::gguf::pack_codes(reinterpret_cast<uint8_t*>(b.data()) + src, n, 5, b.data() + dst / 4);
        require(a == b, "overlap full-buffer oracle"); ++cases;
      }
    const size_t page = size_t(sysconf(_SC_PAGESIZE));
    struct Pages {
      uint8_t* p;
      size_t page;
      ~Pages() { if (p != MAP_FAILED) munmap(p, 3 * page); }
    } input{static_cast<uint8_t*>(mmap(nullptr, 3 * page, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0)), page},
      output{static_cast<uint8_t*>(mmap(nullptr, 3 * page, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0)), page};
    require(input.p != MAP_FAILED && output.p != MAP_FAILED, "guard pages allocate");
    require(!mprotect(input.p + page, page, PROT_READ | PROT_WRITE) &&
            !mprotect(output.p + page, page, PROT_READ | PROT_WRITE), "guard pages protect");
    for (size_t n : {0, 32, 64, 96, 256, 512}) for (bool end : {false, true}) {
      const size_t bytes = n * 5 / 8;
      uint8_t* src = input.p + page + (end ? page - n : 0);
      uint8_t* dst = output.p + page + (end ? page - bytes : 0);
      for (size_t j = 0; j < n; ++j) src[j] = uint8_t(rng());
      const auto expected = oracle(src, n, 5);
      memset(output.p + page, 0xab, page);
      ni::gguf::pack_codes(src, n, 5, reinterpret_cast<uint32_t*>(dst));
      require(!bytes || !memcmp(dst, expected.data(), bytes), "guarded packed bytes");
      for (size_t j = 0; j < page; ++j)
        if (output.p + page + j < dst || output.p + page + j >= dst + bytes)
          require(output.p[page + j] == 0xab, "guarded output outside range");
      ++cases;
    }
    printf("  pack_codes: %zu independent-byte, overlap and page-boundary cases PASS\n", cases);
    return 0;
  } catch (const std::exception& e) {
    printf("  pack_codes FAILED: %s\n", e.what());
    return 1;
  }
}
}  // namespace pack_codes_test
