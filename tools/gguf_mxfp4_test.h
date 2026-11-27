// MXFP4 CPU decoding: byte layout, scale range, rounding, overlap order and page bounds.
#pragma once
#include "gguf/gguf.h"
#include "gguf/quant.h"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

namespace mxfp4_test {
inline void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}
inline void scalar(const uint8_t* src, float* dst, size_t n) {
  constexpr int values[] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  for (size_t block = 0; block < n / 32; ++block) {
    const uint8_t* b = src + 17 * block;
    const float scale = std::ldexp(1.0f, int(b[0]) - 128);
    for (int j = 0; j < 16; ++j) {
      dst[32 * block + j] = scale * float(values[b[1 + j] & 15]);
      dst[32 * block + j + 16] = scale * float(values[b[1 + j] >> 4]);
    }
  }
}
inline int run() {
  const int old_round = std::fegetround();
  try {
    size_t cases = 0;
    for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
      require(std::fesetround(mode) == 0, "MXFP4 set rounding mode");
      for (int e = 0; e < 256; ++e) for (int q = 0; q < 256; ++q) for (int align : {0, 1, 7, 15}) {
        std::array<uint8_t, 33> input{};
        input[align] = uint8_t(e);
        std::memset(input.data() + align + 1, q, 16);
        const auto saved = input;
        std::array<float, 40> got, expected;
        got.fill(123); expected.fill(123);
        std::feclearexcept(FE_ALL_EXCEPT);
        scalar(input.data() + align, expected.data(), 39);
        const int flags = std::fetestexcept(FE_ALL_EXCEPT);
        std::feclearexcept(FE_ALL_EXCEPT);
        ni::gguf::dequantize(ni::gguf::kMXFP4, input.data() + align, got.data(), 39);
        require(std::memcmp(got.data(), expected.data(), sizeof(got)) == 0, "MXFP4 exact oracle and tail");
        require(std::fetestexcept(FE_ALL_EXCEPT) == flags, "MXFP4 floating exceptions");
        require(input == saved, "MXFP4 input unchanged");
        ++cases;
      }
    }
    std::fesetround(FE_TONEAREST);
    for (int seed = 0; seed < 256; ++seed) {
      std::array<uint8_t, 68> input{};
      for (size_t i = 0; i < input.size(); ++i) input[i] = uint8_t(i * 37 + seed);
      std::array<float, 135> got{}, expected{};
      scalar(input.data(), expected.data(), 135);
      ni::gguf::dequantize(ni::gguf::kMXFP4, input.data(), got.data(), 135);
      require(std::memcmp(got.data(), expected.data(), sizeof(got)) == 0, "MXFP4 mixed lanes and blocks");
    }
    for (size_t source = 0; source < 128; ++source) for (size_t dest = 0; dest < 32; ++dest) {
      std::array<float, 128> got{}, expected{};
      auto* bytes = reinterpret_cast<uint8_t*>(got.data());
      for (size_t i = 0; i < sizeof(got); ++i) bytes[i] = uint8_t(i * 37 + 71);
      expected = got;
      scalar(reinterpret_cast<const uint8_t*>(expected.data()) + source, expected.data() + dest, 64);
      ni::gguf::dequantize(ni::gguf::kMXFP4, bytes + source, got.data() + dest, 64);
      require(std::memcmp(got.data(), expected.data(), sizeof(got)) == 0, "MXFP4 overlapping scalar order");
    }
    const size_t page = size_t(sysconf(_SC_PAGESIZE));
    for (bool end : {false, true}) {
      void* a = mmap(nullptr, page * 3, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
      void* b = mmap(nullptr, page * 3, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
      require(a != MAP_FAILED && b != MAP_FAILED, "MXFP4 guard mappings");
      require(mprotect(static_cast<uint8_t*>(a) + page, page, PROT_READ | PROT_WRITE) == 0, "MXFP4 input page");
      require(mprotect(static_cast<uint8_t*>(b) + page, page, PROT_READ | PROT_WRITE) == 0, "MXFP4 output page");
      auto* in = static_cast<uint8_t*>(a) + page + (end ? page - 17 : 0);
      auto* out = reinterpret_cast<float*>(static_cast<uint8_t*>(b) + page + (end ? page - 128 : 0));
      in[0] = 127; for (int j = 0; j < 16; ++j) in[j + 1] = uint8_t(j * 37);
      std::array<float, 32> expected{}; scalar(in, expected.data(), 32);
      mprotect(static_cast<uint8_t*>(a) + page, page, PROT_READ);
      ni::gguf::dequantize(ni::gguf::kMXFP4, in, out, 32);
      require(std::memcmp(out, expected.data(), 128) == 0, "MXFP4 guarded exact output");
      munmap(a, page * 3); munmap(b, page * 3);
    }
    std::fesetround(old_round);
    std::printf("MXFP4: %zu oracle cases, 4096 overlap cases, page guards passed\n", cases);
    return 0;
  } catch (const std::exception& e) {
    std::fesetround(old_round);
    std::printf("MXFP4 FAILED: %s\n", e.what());
    return 1;
  }
}
}  // namespace mxfp4_test
