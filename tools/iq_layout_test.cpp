#include "gguf_q3_k_test.h"
#include "gguf_iq3_xxs_test.h"
#include "gguf_iq2_xxs_test.h"
#include "gguf_q8_0_test.h"
#include "gguf_q6_k_test.h"
#include "gguf_q2_k_test.h"
#include "gguf_iq4_xs_test.h"
#include "gguf_iq3_s_test.h"
#include "gguf_iq2_s_test.h"
#include "gguf_iq2_xs_test.h"
#include "gguf_iq1_s_test.h"
#include "gguf_q4_codes_test.h"
#include "gguf_pack_codes_test.h"
#include "gguf_pack4_test.h"
#include "gguf_mxfp4_test.h"

// The GPU layout of llama.cpp's IQ blocks (src/kernels/wfmt_iq.metal, gguf::iq_repack) checked on the CPU: random blocks
// of every type the IQ MoE kernels read (IQ1_S, IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S) are repacked in two row ranges (as the
// loader's threads do), then every value is decoded from the repacked sections as the kernels build their MMA operands
// (iq_a16 / iq_a8: the integer A = magnitude x (2s + 1) with its sign, from the grid tables of the embedded Metal source
// itself) and compared with gguf::dequantize of the original blocks: dequantize's value / (d x the format's constant)
// must round to A exactly, and A must lie in the exact-half range the MMA forms rely on (|A| < 2048). IQ1_S's decode
// path (iq1s_dot32: the grid's nibbles n = g + 1, the shift as delta - 1) is checked the same way. Then the type helpers
// (block bytes, kernel suffixes, config modes) and the layout's size (the GGUF's bytes).
//   iq_layout_test        (no model, no GPU)
#include "gguf/gguf.h"
#include "gguf/quant.h"

#include <array>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

namespace ni {
extern const char* const kKernelSource;
}

namespace gg = ni::gguf;

namespace {

int g_fail = 0;

void check(bool ok, const char* what, const std::string& detail = "") {
  if (!ok) {
    ++g_fail;
    if (g_fail <= 20) printf("  FAILED: %s %s\n", what, detail.c_str());
  }
}

// a `constant <type> name[n] = { ... };` table of the Metal source
std::vector<uint64_t> metal_table(const char* decl, size_t n) {
  const std::string src = ni::kKernelSource;
  const size_t a = src.find(decl);
  if (a == std::string::npos) {
    printf("FAILED: %s not in the kernel source\n", decl);
    exit(1);
  }
  const size_t b = src.find("};", a);
  std::vector<uint64_t> v;
  for (size_t p = src.find("0x", a); p != std::string::npos && p < b; p = src.find("0x", p + 2))
    v.push_back(std::stoull(src.substr(p + 2, 16), nullptr, 16));
  if (v.size() != n) {
    printf("FAILED: %s has %zu entries, %zu expected\n", decl, v.size(), n);
    exit(1);
  }
  return v;
}

uint32_t rd32(const uint8_t* p) {
  uint32_t v;
  memcpy(&v, p, 4);
  return v;
}
uint16_t rd16(const uint8_t* p) {
  uint16_t v;
  memcpy(&v, p, 2);
  return v;
}
float half_at(const uint8_t* p) { return gg::from_f16(rd16(p)); }

// byte e (value e of a grid point of 8) of a permuted grid entry: words (m0, m1, m4, m5), (m2, m3, m6, m7)
uint32_t perm_byte(uint64_t P, int e) {
  static const int at[8] = {0, 1, 4, 5, 2, 3, 6, 7};
  return uint32_t(P >> (8 * at[e])) & 0xFF;
}

uint32_t s8(uint32_t s7) { return s7 | (uint32_t(__builtin_popcount(s7) & 1) << 7); }

struct Tables {
  std::vector<uint64_t> iq2xxs, iq2s, iq3s, iq3xxs, iq1s;
};

// value k of row r from the repacked matrix: the MMA operand A and the factor f with value = A d f
struct Dec {
  int A = 0;
  double f = 0, d = 0;
  double alt = 0;  // IQ1_S: the rows path's value, d (2s + 1) (n + delta - 1)
};

Dec decode(uint32_t t, const Tables& T, const uint8_t* buf, const gg::IqLayout& L, size_t K, size_t r, size_t k) {
  const size_t nsb = K / 32, j = k / 32, i = r * nsb + j, v = k % 32, l = v / 8, e = v % 8;
  Dec o;
  o.d = half_at(buf + L.d + (r * (K / 256) + k / 256) * 2);
  if (t == gg::kIQ2_XXS) {
    const uint32_t qx = rd32(buf + i * 8), qy = rd32(buf + i * 8 + 4);
    const uint32_t m = perm_byte(T.iq2xxs[(qx >> (8 * l)) & 0xFF], int(e));
    const bool neg = (s8((qy >> (7 * l)) & 127) >> e) & 1;
    o.A = int(m * ((qy >> 27) | 1u)) * (neg ? -1 : 1);
    o.f = 1.0 / 8;
  } else if (t == gg::kIQ2_S) {
    const uint32_t qx = rd32(buf + i * 8), qy = rd32(buf + i * 8 + 4), x1 = rd16(buf + L.x1 + i * 2);
    const uint32_t idx = ((qx >> (8 * l)) & 0xFF) | (((x1 >> (2 * l)) & 3) << 8);
    const uint32_t m = perm_byte(T.iq2s[idx], int(e));
    const bool neg = (qy >> (8 * l + e)) & 1;
    const uint32_t s = (x1 >> (8 + 4 * (v / 16))) & 15;
    o.A = int(m * (2 * s + 1)) * (neg ? -1 : 1);
    o.f = 1.0 / 8;
  } else if (t == gg::kIQ3_S) {
    const size_t p = v / 4;  // grid point of 4 values
    const uint32_t qb = buf[i * 8 + p], qh = buf[L.x2 + i], sg = rd32(buf + L.x1 + i * 4);
    const uint32_t m = (uint32_t(T.iq3s[qb | (((qh >> p) & 1) << 8)]) >> (8 * (v % 4))) & 0xFF;
    const uint32_t s = (buf[L.x3 + i / 2] >> (4 * (i & 1))) & 15;
    o.A = int(m * (2 * s + 1)) * (((sg >> v) & 1) ? -1 : 1);
    o.f = 1.0;
  } else if (t == gg::kIQ3_XXS) {
    const size_t p = v / 4;
    const uint32_t qb = buf[i * 8 + p], aux = rd32(buf + L.x1 + i * 4);
    const uint32_t m = (uint32_t(T.iq3xxs[qb]) >> (8 * (v % 4))) & 0xFF;
    const bool neg = (s8((aux >> (7 * l)) & 127) >> e) & 1;
    o.A = int(m * ((aux >> 27) | 1u)) * (neg ? -1 : 1);
    o.f = 1.0 / 4;
  } else {  // IQ1_S
    const uint32_t q = rd32(buf + i * 4), qh = rd16(buf + L.x1 + i * 2);
    const uint32_t G = uint32_t(T.iq1s[((q >> (8 * l)) & 0xFF) | (((qh >> (3 * l)) & 7) << 8)]);
    const int n = int((G >> (8 * (e % 4) + 4 * (e / 4))) & 15);
    const int c = int(2 * ((qh >> 12) & 7) + 1), dl = (qh & 0x8000) ? -1 : 1;
    o.A = (8 * (n - 1) + dl) * c;
    o.f = 1.0 / 8;
    o.alt = o.d * c * (n + ((qh & 0x8000) ? -1.125 : -0.875));
  }
  return o;
}

void test_type(uint32_t t, const Tables& T, std::mt19937& rng) {
  const size_t E = 3, N = 5, K = 512, rows = E * N, bb = gg::iq_block_bytes(t), nb = K / 256;
  std::vector<uint8_t> raw(rows * nb * bb);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_real_distribution<float> ud(1e-4f, 3e-2f);
  for (size_t b = 0; b < rows * nb; ++b) {
    uint8_t* p = raw.data() + b * bb;
    for (size_t x = 2; x < bb; ++x) p[x] = uint8_t(byte(rng));
    _Float16 d = _Float16(ud(rng) * (b % 3 == 0 ? -1.0f : 1.0f));
    memcpy(p, &d, 2);
  }
  const gg::IqLayout L = gg::iq_layout(t, rows, K);
  check(L.bytes == rows * nb * bb, "layout size", gg::type_name(t) + ": " + std::to_string(L.bytes));
  std::vector<uint8_t> buf(L.bytes, 0xA5);
  const size_t split = 7;  // two row ranges, as the loader's threads repack them
  gg::iq_repack(t, raw.data(), rows, K, 0, split, buf.data());
  gg::iq_repack(t, raw.data() + split * nb * bb, rows, K, split, rows, buf.data());
  std::vector<float> ref(rows * K);
  gg::dequantize(t, raw.data(), ref.data(), ref.size());
  int maxA = 0;
  size_t bad = 0, bad_alt = 0;
  for (size_t r = 0; r < rows; ++r)
    for (size_t k = 0; k < K; ++k) {
      const Dec o = decode(t, T, buf.data(), L, K, r, k);
      const double want = double(ref[r * K + k]) / (o.d * o.f);
      maxA = std::max(maxA, std::abs(o.A));
      if (std::fabs(want - o.A) > 1e-3) {
        if (++bad <= 3)
          check(false, "value", gg::type_name(t) + " row " + std::to_string(r) + " col " + std::to_string(k) + ": A " +
                                    std::to_string(o.A) + ", dequantize / (d f) " + std::to_string(want));
      }
      if (t == gg::kIQ1_S && std::fabs(o.alt - ref[r * K + k]) > 1e-6 * std::fabs(o.d) * 16 && ++bad_alt <= 3)
        check(false, "IQ1_S rows-path value", std::to_string(o.alt) + " vs " + std::to_string(ref[r * K + k]));
    }
  check(maxA < 2048, "operand range", gg::type_name(t) + ": " + std::to_string(maxA));
  printf("  %-8s %3zu-byte blocks: %zu values, |A| <= %4d, %s\n", gg::type_name(t).c_str(), bb, rows * K, maxA,
         bad || bad_alt ? "MISMATCH" : "every operand exact");
}


// The scalar format oracle keeps the original read/store order, including overlapping buffers.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void iq4_nl_reference(const uint8_t* b, float* y) {
  static constexpr int8_t levels[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
  _Float16 h;
  memcpy(&h, b, 2);
  const float d = float(h);
  const uint8_t* q = b + 2;
  for (int j = 0; j < 16; ++j) {
    y[j] = d * float(levels[q[j] & 15]);
    y[j + 16] = d * float(levels[q[j] >> 4]);
  }
}

void iq4_nl_block(uint8_t* b, uint16_t h, unsigned pattern) {
  memcpy(b, &h, 2);
  for (unsigned j = 0; j < 16; ++j)
    b[2 + j] = uint8_t(((5 * j + pattern) & 15) | (((9 * j + pattern / 16) & 15) << 4));
}

void test_iq4_nl() {
  fenv_t saved;
  check(std::fegetenv(&saved) == 0, "IQ4_NL save FP environment");
  size_t cases = 0, bad = 0;
  const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
  auto compare = [&](const uint8_t* src, float* got, float* want, const char* what) {
    std::feclearexcept(FE_ALL_EXCEPT);
    iq4_nl_reference(src, want);
    const int ref_flags = std::fetestexcept(FE_ALL_EXCEPT);
    std::feclearexcept(FE_ALL_EXCEPT);
    gg::dequantize(gg::kIQ4_NL, src, got, 32);
    const int flags = std::fetestexcept(FE_ALL_EXCEPT);
    ++cases;
    if (memcmp(got, want, 32 * sizeof(float)) || flags != ref_flags) {
      if (++bad <= 3) check(false, what, "case " + std::to_string(cases) + " flags " + std::to_string(flags) +
                                              " vs " + std::to_string(ref_flags));
    }
  };
  for (int mode : modes) {
    check(std::fesetround(mode) == 0, "IQ4_NL set rounding mode");
    // Every half bit pattern times every signed table level, in both nibble/output halves.
    std::array<uint8_t, 18> raw;
    std::array<float, 34> got, want;
    for (uint32_t h = 0; h <= 0xffff; ++h) {
      iq4_nl_block(raw.data(), uint16_t(h), h);
      got.fill(123.5f);
      want.fill(123.5f);
      compare(raw.data(), got.data() + 1, want.data() + 1, "IQ4_NL half bits/nibbles/flags");
      check(got.front() == 123.5f && got.back() == 123.5f, "IQ4_NL guarded output");
    }
    for (size_t position = 0; position < 16; ++position)
      for (unsigned packed = 0; packed < 256; ++packed) {
        iq4_nl_block(raw.data(), 0x3555, unsigned(position * 19));
        raw[2 + position] = uint8_t(packed);
        compare(raw.data(), got.data() + 1, want.data() + 1, "IQ4_NL packed byte/position");
      }
    for (size_t offset = 0; offset < 32; ++offset) {
      std::array<uint8_t, 18 + 31> bytes;
      bytes.fill(0xa7);
      iq4_nl_block(bytes.data() + offset, 0x3555, unsigned(offset * 17));
      const auto original = bytes;
      for (size_t out_offset = 0; out_offset < 4; ++out_offset) {
        std::array<float, 38> result, reference;
        result.fill(123.5f); reference.fill(123.5f);
        compare(bytes.data() + offset, result.data() + 1 + out_offset, reference.data() + 1 + out_offset,
                "IQ4_NL unaligned input/output");
        check(result == reference && bytes == original, "IQ4_NL unaligned guards and input preserved");
      }
    }
    // All byte source alignments and nearby float-aligned output positions, including touching ranges.
    for (size_t alignment = 0; alignment < 16; ++alignment)
      for (size_t out_offset = 32; out_offset <= 356; out_offset += 4)
        for (uint16_t h : {uint16_t(0), uint16_t(0x8000), uint16_t(0x3c00), uint16_t(0x7bff),
                           uint16_t(0x7c00), uint16_t(0x7d55), uint16_t(0xfd55)}) {
          alignas(16) std::array<uint8_t, 512> result, reference;
          result.fill(0xa7);
          const size_t src_offset = 160 + alignment;
          iq4_nl_block(result.data() + src_offset, h, unsigned(out_offset + alignment));
          reference = result;
          std::feclearexcept(FE_ALL_EXCEPT);
          iq4_nl_reference(reference.data() + src_offset, reinterpret_cast<float*>(reference.data() + out_offset));
          const int ref_flags = std::fetestexcept(FE_ALL_EXCEPT);
          std::feclearexcept(FE_ALL_EXCEPT);
          gg::dequantize(gg::kIQ4_NL, result.data() + src_offset, reinterpret_cast<float*>(result.data() + out_offset), 32);
          const int flags = std::fetestexcept(FE_ALL_EXCEPT);
          ++cases;
          if (result != reference || flags != ref_flags) {
            if (++bad <= 3) check(false, "IQ4_NL overlap bytes/flags", "source " + std::to_string(src_offset) +
                                                                        " output " + std::to_string(out_offset));
          }
        }
  }
  // Earlier blocks can overwrite later input headers/codes: compare the complete sequential backing bytes.
  for (size_t blocks : {size_t(2), size_t(3), size_t(5)})
    for (size_t alignment = 0; alignment < 16; ++alignment)
      for (size_t out_offset = 192; out_offset <= 640; out_offset += 4) {
        alignas(16) std::array<uint8_t, 2048> result, reference;
        result.fill(0xa7);
        const size_t src_offset = 512 + alignment;
        for (size_t i = 0; i < blocks; ++i)
          iq4_nl_block(result.data() + src_offset + 18 * i, i % 2 ? 0x3555 : 0x3c00, unsigned(i * 37));
        reference = result;
        std::feclearexcept(FE_ALL_EXCEPT);
        for (size_t i = 0; i < blocks; ++i)
          iq4_nl_reference(reference.data() + src_offset + 18 * i,
                           reinterpret_cast<float*>(reference.data() + out_offset) + 32 * i);
        const int ref_flags = std::fetestexcept(FE_ALL_EXCEPT);
        std::feclearexcept(FE_ALL_EXCEPT);
        gg::dequantize(gg::kIQ4_NL, result.data() + src_offset,
                       reinterpret_cast<float*>(result.data() + out_offset), 32 * blocks);
        const int flags = std::fetestexcept(FE_ALL_EXCEPT);
        ++cases;
        if (result != reference || flags != ref_flags) {
          if (++bad <= 3) check(false, "IQ4_NL sequential overlap", std::to_string(blocks) + " blocks");
        }
      }
  check(bad == 0, "IQ4_NL scalar oracle", std::to_string(bad) + " mismatches in " + std::to_string(cases) + " cases");
  check(std::fesetenv(&saved) == 0, "IQ4_NL restore FP environment");
  printf("  IQ4_NL: %zu half/nibble/rounding/unaligned/overlap byte and FP-flag cases\n", cases);
}

void test_iq4_nl_pages() {
  const long page_size = sysconf(_SC_PAGESIZE);
  check(page_size >= 4096, "IQ4_NL page size");
  if (page_size < 4096) return;
  const size_t page = size_t(page_size);
  void* input = mmap(nullptr, 3 * page, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
  void* output = mmap(nullptr, 3 * page, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
  check(input != MAP_FAILED && output != MAP_FAILED, "IQ4_NL guard page maps");
  if (input == MAP_FAILED || output == MAP_FAILED) {
    if (input != MAP_FAILED) munmap(input, 3 * page);
    if (output != MAP_FAILED) munmap(output, 3 * page);
    return;
  }
  auto* ib = static_cast<uint8_t*>(input) + page;
  auto* ob = static_cast<uint8_t*>(output) + page;
  const bool writable = mprotect(ib, page, PROT_READ | PROT_WRITE) == 0 &&
                        mprotect(ob, page, PROT_READ | PROT_WRITE) == 0;
  check(writable, "IQ4_NL writable middle pages");
  if (writable) {
    for (size_t n : {size_t(0), size_t(1), size_t(31), size_t(32), size_t(33), size_t(63),
                     size_t(64), size_t(65), size_t(96), size_t(160)}) {
      const size_t blocks = n / 32;
      uint8_t* src = ib + page - 18 * blocks;
      float* dst = reinterpret_cast<float*>(ob + page - n * sizeof(float));
      memset(ob, 0xa7, page);
      std::vector<float> reference(n);
      if (n) memset(reference.data(), 0xa7, n * sizeof(float));
      for (size_t i = 0; i < blocks; ++i) {
        iq4_nl_block(src + 18 * i, i % 2 ? 0x8000 : 0x3c00, unsigned(i * 19));
        iq4_nl_reference(src + 18 * i, reference.data() + 32 * i);
      }
      // The generic decoder visits only whole blocks, leaving a partial-block suffix untouched.
      gg::dequantize(gg::kIQ4_NL, src, dst, n);
      check(!n || !memcmp(dst, reference.data(), n * sizeof(float)), "IQ4_NL page-end blocks/tail", std::to_string(n));
      check(ob[page - n * sizeof(float) - 1] == 0xa7, "IQ4_NL output prefix guard");
    }
    bool caught = false;
    try { gg::dequantize(UINT32_MAX, ib + page, reinterpret_cast<float*>(ob + page), 32); }
    catch (const gg::Error&) { caught = true; }
    check(caught, "IQ4_NL unsupported type fails before input/output access");
    std::array<float, 32> reference;
    iq4_nl_block(ib, 0x3c00, 5);
    iq4_nl_reference(ib, reference.data());
    gg::dequantize(gg::kIQ4_NL, ib, reinterpret_cast<float*>(ob), 32);
    check(!memcmp(ob, reference.data(), sizeof(reference)) && ob[sizeof(reference)] == 0xa7,
          "IQ4_NL page-start read and recovery after error");
  }
  munmap(input, 3 * page);
  munmap(output, 3 * page);
  printf("  IQ4_NL: protected-page block bounds, tails and error recovery\n");
}

}  // namespace

int main() {
  if (q3_k_test::run()) return 1;
  if (iq3_s_test::run()) return 1;
  if (iq2_s_test::run()) return 1;
  if (iq2_xs_test::run()) return 1;
  if (iq1_s_test::run()) return 1;
  g_fail += q6_test::run();
  g_fail += iq2_xxs_test::run();
  g_fail += q2_k_test::run();
  g_fail += iq3_xxs_test::run();
  g_fail += iq4_xs_test::run();
  g_fail += pack_codes_test::run();
  test_iq4_nl();
  test_iq4_nl_pages();
  g_fail += q4_codes_test::run();
  Tables T;
  T.iq2xxs = metal_table("constant ulong iq2xxs_gridp[256]", 256);
  T.iq2s = metal_table("constant ulong iq2s_gridp[1024]", 1024);
  T.iq3s = metal_table("constant uint iq3s_grid[512]", 512);
  T.iq3xxs = metal_table("constant uint iq3xxs_grid[256]", 256);
  T.iq1s = metal_table("constant uint iq1s_grid_gpu[2048]", 2048);
  std::mt19937 rng(11);
  printf("IQ blocks in wfmt_iq.metal's layout against gguf::dequantize:\n");
  const uint32_t types[] = {gg::kIQ1_S, gg::kIQ2_XXS, gg::kIQ2_S, gg::kIQ3_XXS, gg::kIQ3_S};
  const char* modes[] = {"iq1_s", "iq2_xxs", "iq2_s", "iq3_xxs", "iq3_s"};
  const char* sfx[] = {"iq1s", "iq2xxs", "iq2s", "iq3xxs", "iq3s"};
  const size_t bytes[] = {50, 66, 82, 98, 110};
  for (int i = 0; i < 5; ++i) {
    const uint32_t t = types[i];
    check(gg::iq_gpu_type(t), "iq_gpu_type", gg::type_name(t));
    check(gg::iq_block_bytes(t) == bytes[i], "iq_block_bytes", gg::type_name(t));
    check(gg::iq_suffix(t) == sfx[i], "iq_suffix", gg::type_name(t) + " -> " + gg::iq_suffix(t));
    check(gg::iq_mode_type(modes[i]) == t, "iq_mode_type", modes[i]);
    test_type(t, T, rng);
  }
  check(!gg::iq_gpu_type(gg::kIQ2_XS) && !gg::iq_gpu_type(gg::kIQ1_M) && !gg::iq_gpu_type(gg::kQ4_K), "iq_gpu_type",
        "takes a type without a layout");
  check(gg::iq_mode_type("affine") == 0 && gg::iq_mode_type("iq2_xs") == 0, "iq_mode_type", "names another mode");
  g_fail += q8_test::run();
  g_fail += pack4_test::run();
  g_fail += mxfp4_test::run();
  if (g_fail) {
    printf("%d check(s) FAILED\n", g_fail);
    return 1;
  }
  printf("all passed\n");
  return 0;
}
