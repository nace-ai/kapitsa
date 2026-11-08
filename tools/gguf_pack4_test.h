// Four-bit packed bytes: independent bit oracle, scalar overlap order and bounded reads/writes.
#pragma once
#include "gguf/quant.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

namespace pack4_test {
inline void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}
inline std::vector<uint8_t> oracle(const uint8_t* q, size_t n) {
  const size_t used = n / 8 * 8;
  std::vector<uint8_t> out(used / 2);
  for (size_t i = 0; i < used; ++i)
    for (unsigned bit = 0; bit < 4; ++bit)
      if (q[i] & (1u << bit)) out[(i * 4 + bit) / 8] |= uint8_t(1u << ((i * 4 + bit) % 8));
  return out;
}
inline void check(const std::vector<uint8_t>& q, size_t offset, size_t n) {
  const auto copy = q, expected = oracle(q.data() + offset, n);
  std::vector<uint32_t> out(n / 8 + 2, 0xa5a5a5a5);
  ni::gguf::pack_codes(q.data() + offset, n, 4, out.data() + 1);
  require(out.front() == 0xa5a5a5a5 && out.back() == 0xa5a5a5a5, "pack4 output canary");
  require(expected.empty() || !memcmp(out.data() + 1, expected.data(), expected.size()), "pack4 independent bytes");
  require(q == copy, "pack4 input unchanged");
}
#if defined(__clang__)
__attribute__((noinline, optnone))
#endif
inline void scalar(const uint8_t* q, size_t n, uint32_t* out) {
  for (size_t w = 0; w < n / 8; ++w) {
    uint32_t word = 0;
    for (int j = 0; j < 8; ++j) word |= uint32_t(q[8 * w + j] & 15) << (4 * j);
    out[w] = word;
  }
}
inline int run() {
  try {
    size_t cases = 0;
    std::vector<uint8_t> q(32);
    for (unsigned a = 0; a < 256; ++a) for (unsigned b = 0; b < 256; ++b) {
      for (size_t i = 0; i < 32; ++i) q[i] = uint8_t(i % 2 ? b : a);
      check(q, 0, 32); ++cases;
    }
    for (size_t lane = 0; lane < 32; ++lane) for (unsigned value = 0; value < 256; ++value) {
      std::fill(q.begin(), q.end(), 0); q[lane] = uint8_t(value);
      check(q, 0, 32); ++cases;
    }
    std::mt19937 rng(44817);
    for (size_t n = 0; n <= 129; ++n) for (size_t offset = 0; offset < 32; ++offset) {
      q.resize(n + offset + 32); for (auto& x : q) x = uint8_t(rng());
      check(q, offset, n); ++cases;
    }
    for (size_t n : {255, 256, 257, 511, 512, 1023, 1024, 2560, 3136, 4096, 8192}) {
      q.resize(n + 31); for (auto& x : q) x = uint8_t(rng());
      for (size_t offset = 0; offset < 32; ++offset) { check(q, offset, n); ++cases; }
    }
    for (size_t n : {0, 7, 8, 24, 31, 32, 33, 40, 63, 64, 65, 96, 127, 128, 511, 512})
      for (size_t src = 256; src < 288; ++src) for (size_t dst = 0; dst <= 768; dst += 4) {
        std::array<uint32_t, 1024> a, b;
        for (auto& x : a) x = rng(); b = a;
        scalar(reinterpret_cast<uint8_t*>(a.data()) + src, n, a.data() + dst / 4);
        ni::gguf::pack_codes(reinterpret_cast<uint8_t*>(b.data()) + src, n, 4, b.data() + dst / 4);
        require(a == b, "pack4 overlap full-buffer bytes"); ++cases;
      }
    const size_t page = size_t(sysconf(_SC_PAGESIZE));
    struct Pages {
      uint8_t* p; size_t page;
      ~Pages() { if (p != MAP_FAILED) munmap(p, page * 3); }
    } input{static_cast<uint8_t*>(mmap(nullptr, page * 3, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0)), page},
      output{static_cast<uint8_t*>(mmap(nullptr, page * 3, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0)), page};
    require(input.p != MAP_FAILED && output.p != MAP_FAILED, "pack4 pages");
    require(!mprotect(input.p + page, page, PROT_READ | PROT_WRITE) &&
            !mprotect(output.p + page, page, PROT_READ | PROT_WRITE), "pack4 protect");
    for (size_t n : {0, 1, 7, 8, 15, 24, 31, 32, 33, 40, 63, 64, 65, 511, 512})
      for (bool end : {false, true}) {
        const size_t used = n / 8 * 8, bytes = used / 2;
        uint8_t* src = input.p + page + (end ? page - used : 0);
        uint8_t* dst = output.p + page + (end ? page - bytes : 0);
        for (size_t i = 0; i < used; ++i) src[i] = uint8_t(rng());
        const auto expected = oracle(src, n);
        memset(output.p + page, 0xa5, page);
        ni::gguf::pack_codes(src, n, 4, reinterpret_cast<uint32_t*>(dst));
        require(!bytes || !memcmp(dst, expected.data(), bytes), "pack4 guarded bytes");
        for (size_t i = 0; i < page; ++i)
          if (output.p + page + i < dst || output.p + page + i >= dst + bytes)
            require(output.p[page + i] == 0xa5, "pack4 guarded write range");
        ++cases;
      }
    printf("  pack4: %zu byte-oracle, overlap, tail and protected-page cases PASS\n", cases);
    return 0;
  } catch (const std::exception& e) {
    printf("  pack4 FAILED: %s\n", e.what()); return 1;
  }
}
}  // namespace pack4_test

