// CPU Q8_0 decoding: independent dyadic-bit oracle, scalar exceptional/overlap semantics, flags and bounds.
#pragma once

#include "gguf/gguf.h"
#include "gguf/quant.h"

#include <array>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

#pragma STDC FENV_ACCESS ON

namespace q8_test {
namespace gg = ni::gguf;

inline void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}

// A finite half is an integer significand times a power of two. Multiplication by any signed byte has at most
// 18 significant bits, so its fp32 bits can be constructed using integer operations, without the decoder's math.
inline uint32_t finite_bits(uint16_t h, uint8_t code) {
  const int q = code < 128 ? int(code) : int(code) - 256;
  const unsigned e = (h >> 10) & 31, m = h & 1023;
  const uint32_t sign = uint32_t(((h >> 15) != 0) != (q < 0)) << 31;
  const uint32_t product = (e ? 1024 + m : m) * unsigned(q < 0 ? -q : q);
  if (!product) return sign;
  const int top = 31 - __builtin_clz(product), power = e ? int(e) - 25 : -24;
  return sign | (uint32_t(top + power + 127) << 23) | ((product << (23 - top)) & 0x7fffffu);
}

#if defined(__clang__)
__attribute__((noinline, optnone))
#elif defined(__GNUC__)
__attribute__((noinline, optimize("O0")))
#endif
inline void scalar(const uint8_t* src, float* dst, size_t n) {
  for (size_t block = 0; block < n / 32; ++block) {
    uint16_t h;
    memcpy(&h, src + block * 34, 2);
    _Float16 half;
    memcpy(&half, &h, 2);
    const volatile float d = float(half);
    for (int j = 0; j < 32; ++j) {
      const volatile int8_t q = *reinterpret_cast<const volatile int8_t*>(src + block * 34 + 2 + j);
      dst[block * 32 + j] = d * float(q);
    }
  }
}

inline void put_block(uint8_t* b, uint16_t h, unsigned pattern) {
  memcpy(b, &h, 2);
  for (unsigned j = 0; j < 32; ++j) b[j + 2] = uint8_t(j * 73 + pattern * 29);
}

struct Environment {
  fenv_t saved;
  Environment() { require(fegetenv(&saved) == 0, "save floating environment"); }
  ~Environment() { fesetenv(&saved); }
};

inline size_t exhaustive() {
  alignas(16) uint8_t block[34];
  alignas(16) float got[32], want[32];
  uint32_t bits[32];
  size_t cases = 0;
  const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
  for (int mode : modes) {
    require(fesetround(mode) == 0, "set rounding mode");
    for (unsigned scale = 0; scale < 65536; ++scale) {
      const uint16_t h = uint16_t(scale);
      memcpy(block, &h, 2);
      const bool finite = (h & 0x7c00u) != 0x7c00u;
      feclearexcept(FE_ALL_EXCEPT);
      for (unsigned q = 0; q < 256; ++q) {
        memset(block + 2, int(q), 32);
        int flags = 0;
        if (finite) {
          const uint32_t v = finite_bits(h, uint8_t(q));
          for (auto& lane : bits) lane = v;
          memcpy(want, bits, sizeof(bits));
        } else {
          feclearexcept(FE_ALL_EXCEPT);
          scalar(block, want, 32);
          flags = fetestexcept(FE_ALL_EXCEPT);
          feclearexcept(FE_ALL_EXCEPT);
        }
        gg::dequantize(gg::kQ8_0, block, got, 32);
        if (memcmp(got, want, sizeof(got)) || (!finite && fetestexcept(FE_ALL_EXCEPT) != flags)) {
          printf("  Q8_0 mismatch: mode=%d half=%04x code=%02x flags=%d/%d\n", mode, scale, q,
                 fetestexcept(FE_ALL_EXCEPT), flags);
          throw std::runtime_error("Q8_0 exhaustive bytes/flags");
        }
        ++cases;
      }
      // Every finite conversion/product is exact and normal or zero in fp32; the full code group must raise no flags.
      if (finite) require(fetestexcept(FE_ALL_EXCEPT) == 0, "Q8_0 finite group exception flags");
    }
  }
  printf("  Q8_0 exhaustive: %zu blocks, %zu lane products, all half/code/lane/rounding bytes and flags exact\n", cases,
         cases * 32);
  return cases;
}

inline size_t mixed_alignment() {
  const uint16_t scales[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3555, 0x3c00, 0xbc00, 0x7bff,
                             0xfbff, 0x7c00, 0xfc00, 0x7c01, 0x7e55, 0xfc01};
  alignas(16) std::array<uint8_t, 34 + 31 + 16> raw;
  alignas(16) std::array<uint8_t, 128 + 64> output;
  float want[32];
  size_t cases = 0;
  const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
  for (int mode : modes) {
    require(fesetround(mode) == 0, "set mixed rounding mode");
    for (uint16_t h : scales)
      for (unsigned pattern = 0; pattern < 8; ++pattern)
        for (unsigned src_align = 0; src_align < 32; ++src_align)
          for (unsigned dst_align = 0; dst_align < 16; dst_align += 4) {
            raw.fill(0x5a);
            put_block(raw.data() + src_align, h, pattern);
            const auto unchanged = raw;
            output.fill(0xa5);
            float* dst = reinterpret_cast<float*>(output.data() + 16 + dst_align);
            feclearexcept(FE_ALL_EXCEPT);
            scalar(raw.data() + src_align, want, 32);
            const int flags = fetestexcept(FE_ALL_EXCEPT);
            feclearexcept(FE_ALL_EXCEPT);
            gg::dequantize(gg::kQ8_0, raw.data() + src_align, dst, 32);
            require(fetestexcept(FE_ALL_EXCEPT) == flags, "Q8_0 mixed exception flags");
            require(memcmp(dst, want, 128) == 0, "Q8_0 mixed/alignment bytes");
            require(raw == unchanged, "Q8_0 disjoint input changed");
            for (size_t j = 0; j < output.size(); ++j)
              if (j < 16 + dst_align || j >= 144 + dst_align)
                require(output[j] == 0xa5, "Q8_0 output canary");
            ++cases;
          }
  }
  // A decoder must retain flags already set by its caller, including on its finite vector path.
  uint16_t h = 0x3c00;
  put_block(raw.data(), h, 0);
  feclearexcept(FE_ALL_EXCEPT);
  require(feraiseexcept(FE_DIVBYZERO | FE_INEXACT) == 0, "raise prior flags");
  gg::dequantize(gg::kQ8_0, raw.data(), want, 32);
  require(fetestexcept(FE_ALL_EXCEPT) == (FE_DIVBYZERO | FE_INEXACT), "Q8_0 prior flags retained");
  printf("  Q8_0 mixed lanes/source alignments/destination alignments: %zu blocks and intact canaries\n", cases);
  return cases;
}

inline size_t overlap() {
  alignas(16) std::array<uint8_t, 1024> got, want;
  const uint16_t scales[] = {0x3c00, 0xbc00, 1, 0x7bff, 0x7c00, 0x7c01};
  size_t cases = 0;
  const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
  for (int mode : modes) {
    require(fesetround(mode) == 0, "set overlap rounding mode");
    for (uint16_t h : scales)
      for (size_t blocks = 1; blocks <= 3; ++blocks)
        for (size_t source = 256; source < 260; ++source)
          for (size_t dest = 128; dest <= 384; dest += 4) {
            got.fill(0x5a);
            for (size_t j = 0; j < blocks; ++j) put_block(got.data() + source + j * 34, h, unsigned(j));
            want = got;
            feclearexcept(FE_ALL_EXCEPT);
            scalar(want.data() + source, reinterpret_cast<float*>(want.data() + dest), blocks * 32);
            const int flags = fetestexcept(FE_ALL_EXCEPT);
            feclearexcept(FE_ALL_EXCEPT);
            gg::dequantize(gg::kQ8_0, got.data() + source, reinterpret_cast<float*>(got.data() + dest), blocks * 32);
            require(fetestexcept(FE_ALL_EXCEPT) == flags, "Q8_0 overlap exception flags");
            require(got == want, "Q8_0 sequential overlap/whole-buffer bytes");
            ++cases;
          }
  }
  printf("  Q8_0 single/multiple-block overlap and adjacent ranges: %zu whole-buffer/flag comparisons\n", cases);
  return cases;
}

struct GuardPages {
  size_t page = size_t(sysconf(_SC_PAGESIZE));
  uint8_t* base = nullptr;
  GuardPages() {
    require(page >= 256, "page size");
    void* p = mmap(nullptr, 3 * page, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
    require(p != MAP_FAILED, "mmap guard pages");
    base = static_cast<uint8_t*>(p);
    if (mprotect(base + page, page, PROT_READ | PROT_WRITE) != 0) {
      munmap(base, 3 * page);
      throw std::runtime_error("mprotect guard page");
    }
  }
  ~GuardPages() { if (base) munmap(base, 3 * page); }
  GuardPages(const GuardPages&) = delete;
  GuardPages& operator=(const GuardPages&) = delete;
};

inline size_t bounds_contract() {
  GuardPages src, dst;
  const uint16_t scales[] = {1, 0x3c00, 0xbc00, 0x7bff, 0x7c00, 0x7c01};
  float want[64];
  size_t cases = 0;
  const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
  for (int mode : modes) {
    require(fesetround(mode) == 0, "set bounds rounding mode");
    for (uint16_t h : scales)
      for (size_t blocks = 1; blocks <= 2; ++blocks)
        for (bool at_end : {false, true}) {
          uint8_t* input = src.base + src.page + (at_end ? src.page - blocks * 34 : 0);
          float* output = reinterpret_cast<float*>(dst.base + dst.page + (at_end ? dst.page - blocks * 128 : 0));
          for (size_t j = 0; j < blocks; ++j) put_block(input + j * 34, h, unsigned(j));
          feclearexcept(FE_ALL_EXCEPT);
          scalar(input, want, blocks * 32);
          const int flags = fetestexcept(FE_ALL_EXCEPT);
          feclearexcept(FE_ALL_EXCEPT);
          gg::dequantize(gg::kQ8_0, input, output, blocks * 32);
          require(fetestexcept(FE_ALL_EXCEPT) == flags, "Q8_0 guard page flags");
          require(memcmp(output, want, blocks * 128) == 0, "Q8_0 guard page bytes");
          ++cases;
        }
  }
  // Whole blocks are the API contract. Preserve its n/32 truncation, including no accesses below one block.
  for (size_t n : {size_t(0), size_t(1), size_t(31)})
    gg::dequantize(gg::kQ8_0, src.base, reinterpret_cast<float*>(dst.base), n);
  uint8_t* input = src.base + 2 * src.page - 34;
  float* output = reinterpret_cast<float*>(dst.base + 2 * dst.page - 128);
  put_block(input, 0x3c00, 3);
  scalar(input, want, 32);
  gg::dequantize(gg::kQ8_0, input, output, 63);
  require(memcmp(output, want, 128) == 0, "Q8_0 partial trailing block truncation");
  alignas(16) std::array<uint8_t, 256> tail;
  tail.fill(0xa5);
  gg::dequantize(gg::kQ8_0, input, reinterpret_cast<float*>(tail.data()), 33);
  require(memcmp(tail.data(), want, 128) == 0, "Q8_0 complete block before partial tail");
  for (size_t j = 128; j < tail.size(); ++j) require(tail[j] == 0xa5, "Q8_0 untouched partial output tail");
  bool threw = false;
  try {
    gg::dequantize(0xffffffffu, src.base, reinterpret_cast<float*>(dst.base), 0);
  } catch (const std::exception&) {
    threw = true;
  }
  require(threw, "unsupported decoder throws before input/output access");
  printf("  Q8_0 guard-page boundaries: %zu guarded cases; zero/subblock/tail and unsupported-type contracts intact\n", cases);
  return cases;
}

inline int run() {
  try {
    Environment env;
    mixed_alignment();
    overlap();
    bounds_contract();
    exhaustive();
    return 0;
  } catch (const std::exception& e) {
    printf("  FAILED: %s\n", e.what());
    return 1;
  }
}

}  // namespace q8_test
