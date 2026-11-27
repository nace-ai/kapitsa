#pragma once

#include "gguf/gguf.h"
#include "gguf/quant.h"

#include <array>
#include <cfenv>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

namespace q4_codes_test {
namespace gg = ni::gguf;

inline float half_at(const uint8_t* p) {
  uint16_t h;
  memcpy(&h, p, sizeof(h));
  return gg::from_f16(h);
}

// Read the format's two planes independently, retaining each scalar read/store for alias cases.
inline void oracle(uint32_t type, const uint8_t* src, uint8_t* codes, float* scale, float* offset, size_t n) {
  const size_t header = type == gg::kQ4_0 ? 2 : 4;
  for (size_t block = 0; block < n / 32; ++block) {
    const uint8_t* p = src + block * (header + 16);
    const float d = half_at(p);
    scale[block] = d;
    offset[block] = type == gg::kQ4_0 ? d * -8.0f : half_at(p + 2);
    for (size_t lane = 0; lane < 16; ++lane) {
      codes[block * 32 + lane] = p[header + lane] % 16;
      codes[block * 32 + 16 + lane] = p[header + lane] / 16;
    }
  }
}

inline int run() {
  size_t cases = 0;
  int failed = 0;
  std::mt19937 rng(57019);
  auto check = [&](bool ok) {
    ++cases;
    if (!ok && !failed++) printf("FAILED: Q4 codes independent bytes/flags, case %zu\n", cases);
  };
  std::vector<float> arena(4096);
  auto fill = [&] {
    auto* bytes = reinterpret_cast<uint8_t*>(arena.data());
    for (size_t i = 0; i < arena.size() * sizeof(float); ++i) bytes[i] = uint8_t(rng());
  };
  auto compare = [&](uint32_t type, size_t n, size_t input, size_t output, size_t scales, size_t offsets) {
    auto expected = arena, actual = arena;
    auto* a = reinterpret_cast<uint8_t*>(expected.data());
    auto* b = reinterpret_cast<uint8_t*>(actual.data());
    feclearexcept(FE_ALL_EXCEPT);
    oracle(type, a + input, a + output, expected.data() + scales / 4, expected.data() + offsets / 4, n);
    const int flags = fetestexcept(FE_ALL_EXCEPT);
    feclearexcept(FE_ALL_EXCEPT);
    gg::decode4(type, b + input, b + output, actual.data() + scales / 4, actual.data() + offsets / 4, n);
    const int got_flags = fetestexcept(FE_ALL_EXCEPT);
    check(flags == got_flags && memcmp(a, b, arena.size() * sizeof(float)) == 0);
  };

  for (const uint32_t type : {gg::kQ4_0, gg::kQ4_1}) {
    fill();
    auto* bytes = reinterpret_cast<uint8_t*>(arena.data());
    // Every half encoding, including signed zeros, subnormals, infinities and NaN payloads.
    for (unsigned h = 0; h < 65536; ++h) {
      const uint16_t value = uint16_t(h), other = uint16_t(h ^ 0x935a);
      memcpy(bytes + 128, &value, 2);
      if (type == gg::kQ4_1) memcpy(bytes + 130, &other, 2);
      compare(type, 32, 128, 4096, 8192, 12288);
    }
    // Every packed-byte value in every lane, with distinct surrounding lanes.
    for (size_t lane = 0; lane < 16; ++lane) {
      for (unsigned value = 0; value < 256; ++value) {
        bytes[128 + (type == gg::kQ4_0 ? 2 : 4) + lane] = uint8_t(value);
        compare(type, 32, 128, 4096, 8192, 12288);
      }
    }
    for (size_t n = 0; n <= 321; ++n) {
      for (size_t alignment = 0; alignment < 16; ++alignment) {
        compare(type, n, 128 + alignment, 4096 + (15 - alignment), 8192, 12288);
      }
    }
    for (const size_t n : {512, 2559, 2560, 2561, 4096}) compare(type, n, 128, 4096, 8192, 12288);
    // Both overlap directions, exact alias, adjacent ranges, cross-block writes and tails.
    for (size_t n = 0; n <= 129; ++n) {
      for (int delta = -160; delta <= 160; ++delta) {
        compare(type, n, 256, size_t(256 + delta), 8192, 12288);
      }
    }
    for (int i = 0; i < 10000; ++i) {
      const size_t n = rng() % 130;
      compare(type, n, 128 + rng() % 128, 128 + rng() % 256, 128 + 4 * (rng() % 64),
              128 + 4 * (rng() % 64));
    }
    for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
      const int previous = fegetround();
      if (fesetround(mode) == 0) {
        for (int i = 0; i < 64; ++i) compare(type, 257, 128 + i % 16, 4096 + i % 16, 8192, 12288);
      }
      fesetround(previous);
    }
  }

  const size_t page = size_t(sysconf(_SC_PAGESIZE));
  auto* input = static_cast<uint8_t*>(mmap(nullptr, 3 * page, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0));
  auto* output = static_cast<uint8_t*>(mmap(nullptr, 3 * page, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0));
  if (input == MAP_FAILED || output == MAP_FAILED) {
    if (input != MAP_FAILED) munmap(input, 3 * page);
    if (output != MAP_FAILED) munmap(output, 3 * page);
    printf("FAILED: Q4 codes guarded allocation\n");
    return 1;
  }
  if (mprotect(input + page, page, PROT_READ | PROT_WRITE) ||
      mprotect(output + page, page, PROT_READ | PROT_WRITE)) {
    munmap(input, 3 * page);
    munmap(output, 3 * page);
    printf("FAILED: Q4 codes guarded protection\n");
    return 1;
  }
  for (const uint32_t type : {gg::kQ4_0, gg::kQ4_1}) {
    for (size_t n = 0; n <= 160; ++n) {
      for (int end = 0; end < 2; ++end) {
        const size_t src_bytes = n / 32 * (type == gg::kQ4_0 ? 18 : 20), dst_bytes = n / 32 * 32;
        auto* src = input + (end ? 2 * page - src_bytes : page);
        auto* dst = output + (end ? 2 * page - dst_bytes : page);
        memset(output + page, 0xa7, page);
        for (size_t j = 0; j < src_bytes; ++j) src[j] = uint8_t(j * 71 + 3);
        std::array<uint8_t, 160> want{};
        std::array<float, 8> as{}, ao{}, bs{}, bo{};
        oracle(type, src, want.data(), as.data(), ao.data(), n);
        gg::decode4(type, src, dst, bs.data(), bo.data(), n);
        bool intact = true;
        for (size_t j = 0; j < page; ++j) {
          const uint8_t* p = output + page + j;
          if (p < dst || p >= dst + dst_bytes) intact &= *p == 0xa7;
        }
        check(intact && memcmp(want.data(), dst, dst_bytes) == 0 && memcmp(as.data(), bs.data(), sizeof(as)) == 0 &&
              memcmp(ao.data(), bo.data(), sizeof(ao)) == 0);
      }
    }
  }
  munmap(input, 3 * page);
  munmap(output, 3 * page);
  feclearexcept(FE_ALL_EXCEPT);
  printf("  Q4 codes: %zu byte, half-scale, FP-flag, alias, tail and guarded cases %s\n", cases,
         failed ? "FAIL" : "PASS");
  return failed != 0;
}
}  // namespace q4_codes_test
