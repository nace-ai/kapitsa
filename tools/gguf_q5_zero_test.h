#pragma once

#include "gguf/gguf.h"
#include "gguf/quant.h"

#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

namespace q5_zero_test {

inline void scalar(const uint8_t* b, float* out) {
  uint16_t h;
  uint32_t high;
  memcpy(&h, b, 2);
  memcpy(&high, b + 2, 4);
  const float d = ni::gguf::from_f16(h);
  for (int j = 0; j < 16; ++j) {
    const int lo = int(b[6 + j] & 15) | int(((high >> j) & 1u) << 4);
    const int hi = int(b[6 + j] >> 4) | int(((high >> (j + 16)) & 1u) << 4);
    out[j] = d * float(lo - 16);
    out[j + 16] = d * float(hi - 16);
  }
}

// Finite half times a signed five-bit integer is exact in fp32. Form the value
// independently from its integer significand and exponent, including signed zero.
inline float finite_oracle(uint16_t h, int code) {
  const int exponent = (h >> 10) & 31;
  const int mantissa = (h & 1023) | (exponent ? 1024 : 0);
  int64_t product = int64_t(mantissa) * code;
  if (h & 0x8000) product = -product;
  if (!product) return std::copysign(0.0f, bool(h & 0x8000) != (code < 0) ? -1.0f : 1.0f);
  return float(std::ldexp(double(product), exponent ? exponent - 25 : -24));
}

inline void put_code(uint8_t* b, int index, unsigned value) {
  const int j = index % 16, shift = index < 16 ? 0 : 4;
  b[6 + j] = uint8_t((b[6 + j] & ~(15u << shift)) | ((value & 15) << shift));
  b[2 + index / 8] = uint8_t((b[2 + index / 8] & ~(1u << (index % 8))) |
                            ((value >> 4) << (index % 8)));
}

inline int run() {
  const int saved_round = std::fegetround();
  size_t cases = 0;
  auto fail = [&](const char* label) {
    std::fesetround(saved_round);
    std::printf("FAIL Q5_0 %s after %zu cases\n", label, cases);
    return 1;
  };
  alignas(16) std::array<uint8_t, 64> block{};
  std::array<float, 34> out{}, expected{};
  auto check = [&](const uint8_t* b) {
    out.fill(12345.0f);
    expected.fill(12345.0f);
    uint16_t h;
    uint32_t high;
    memcpy(&h, b, 2);
    memcpy(&high, b + 2, 4);
    std::feclearexcept(FE_ALL_EXCEPT);
    if ((h & 0x7c00u) == 0x7c00u) scalar(b, expected.data() + 1);
    else for (int i = 0; i < 32; ++i) {
      const int code = ((b[6 + i % 16] >> (i < 16 ? 0 : 4)) & 15) | int(((high >> i) & 1u) << 4);
      expected[1 + i] = finite_oracle(h, code - 16);
    }
    const int flags = std::fetestexcept(FE_ALL_EXCEPT);
    std::array<uint8_t, 22> before;
    memcpy(before.data(), b, before.size());
    std::feclearexcept(FE_ALL_EXCEPT);
    ni::gguf::dequantize(ni::gguf::kQ5_0, b, out.data() + 1, 32);
    ++cases;
    return !memcmp(out.data(), expected.data(), sizeof(out)) &&
           !memcmp(before.data(), b, before.size()) && std::fetestexcept(FE_ALL_EXCEPT) == flags;
  };
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    if (std::fesetround(mode)) return fail("rounding mode");
    for (uint32_t h = 0; h < 65536; ++h) {
      uint8_t* b = block.data() + h % 16;
      memset(b, 0, 22);
      const auto half = uint16_t(h);
      memcpy(b, &half, 2);
      for (int i = 0; i < 32; ++i) put_code(b, i, unsigned(i));
      if (!check(b)) return fail("half/code sweep");
    }
    for (unsigned bit = 0; bit < 32; ++bit) for (unsigned low = 0; low < 16; ++low) {
      uint8_t* b = block.data() + low;
      memset(b, 0, 22);
      const uint16_t h = 0xb555;
      memcpy(b, &h, 2);
      for (int i = 0; i < 32; ++i) put_code(b, i, low | (unsigned(i) == bit ? 16 : 0));
      if (!check(b)) return fail("high-bit lane");
    }
    // Compare the complete arena to the original sequential expression when a
    // write can change a later packed input byte; also cover the exact guard edges.
    for (int offset = 0; offset < 4; ++offset) for (int dst = 28; dst <= 188; dst += 4) {
      alignas(16) std::array<uint8_t, 512> actual{}, wanted{};
      for (size_t i = 0; i < actual.size(); ++i) actual[i] = uint8_t(i * 71 + offset);
      const uint16_t h = 0x3555;
      memcpy(actual.data() + 160 + offset, &h, 2);
      wanted = actual;
      std::feclearexcept(FE_ALL_EXCEPT);
      scalar(wanted.data() + 160 + offset, reinterpret_cast<float*>(wanted.data() + dst));
      const int flags = std::fetestexcept(FE_ALL_EXCEPT);
      std::feclearexcept(FE_ALL_EXCEPT);
      ni::gguf::dequantize(ni::gguf::kQ5_0, actual.data() + 160 + offset,
                           reinterpret_cast<float*>(actual.data() + dst), 32);
      ++cases;
      if (actual != wanted || flags != std::fetestexcept(FE_ALL_EXCEPT)) return fail("overlap");
    }
  }
  std::fesetround(saved_round);
  const long page = sysconf(_SC_PAGESIZE);
  if (page < 256) return fail("page size");
  auto* in = static_cast<uint8_t*>(mmap(nullptr, size_t(page) * 2, PROT_READ | PROT_WRITE,
                                       MAP_PRIVATE | MAP_ANON, -1, 0));
  auto* dest = static_cast<uint8_t*>(mmap(nullptr, size_t(page) * 2, PROT_READ | PROT_WRITE,
                                         MAP_PRIVATE | MAP_ANON, -1, 0));
  if (in == MAP_FAILED || dest == MAP_FAILED) {
    if (in != MAP_FAILED) munmap(in, size_t(page) * 2);
    if (dest != MAP_FAILED) munmap(dest, size_t(page) * 2);
    return fail("mapping");
  }
  const bool protected_ok = !mprotect(in + page, size_t(page), PROT_NONE) &&
                            !mprotect(dest + page, size_t(page), PROT_NONE);
  if (protected_ok) {
    uint8_t* b = in + page - 22;
    memset(b, 0x55, 22);
    float* y = reinterpret_cast<float*>(dest + page - 128);
    ni::gguf::dequantize(ni::gguf::kQ5_0, b, y, 32);
    std::array<float, 32> gold;
    scalar(b, gold.data());
    if (memcmp(y, gold.data(), sizeof(gold))) {
      munmap(in, size_t(page) * 2); munmap(dest, size_t(page) * 2);
      return fail("guarded output");
    }
    ni::gguf::dequantize(ni::gguf::kQ5_0, in + page, reinterpret_cast<float*>(dest + page), 0);
    ni::gguf::dequantize(ni::gguf::kQ5_0, in + page, reinterpret_cast<float*>(dest + page), 31);
  }
  munmap(in, size_t(page) * 2); munmap(dest, size_t(page) * 2);
  if (!protected_ok) return fail("page protection");
  std::printf("PASS Q5_0: %zu oracle/overlap cases and guarded boundaries\n", cases);
  return 0;
}

}  // namespace q5_zero_test
