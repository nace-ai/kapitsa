#!/usr/bin/env python3
"""CPU-only checks of the actual QMT metadata helper and weight-load schedule.

This compiles extracted Metal source with C++ qualifier/type stubs, not Metal.
It does not establish GPU compilation, synchronization, register use or speed.
"""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/kernels/tensor/qmt.metal"


def block(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


# The candidate must leave the shipped fp32 FMA -> half conversion untouched.
DEC = """NI_INLINE void qmt_dec(const QmtRaw r, threadgroup half* dst) {
  half w[16];
  for (int j = 0; j < 8; ++j) {
    w[j] = half(fma(float((r.w.x >> (4 * j)) & 15u), r.s, r.b));
    w[8 + j] = half(fma(float((r.w.y >> (4 * j)) & 15u), r.s, r.b));
  }
  *(threadgroup half4*)(dst) = half4(w[0], w[1], w[2], w[3]);
  *(threadgroup half4*)(dst + 4) = half4(w[4], w[5], w[6], w[7]);
  *(threadgroup half4*)(dst + 8) = half4(w[8], w[9], w[10], w[11]);
  *(threadgroup half4*)(dst + 12) = half4(w[12], w[13], w[14], w[15]);
}"""

PREAMBLE = r"""
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#define device
#define thread
#define threadgroup
#define NI_INLINE inline
using uint = uint32_t;
using std::min;
using std::fma;
using half = _Float16;
struct half4 {
  half x, y, z, w;
  half4(half a, half b, half c, half d) : x(a), y(b), z(c), w(d) {}
};
struct alignas(8) uint2 { uint x, y; };
static std::vector<const void*> metadata_reads;
struct bfloat {
  float value;
  operator float() const { metadata_reads.push_back(this); return value; }
};
struct QmtParams { uint M, N, K, x_ld, y_ld, add, gs, w_es, s_es; };
static uint64_t loads = 0, cases = 0, decoded = 0;
"""

HARNESS = r"""
template<int NP, int BN, int BK>
void check(uint N, uint K, uint M, bool leading_view) {
  // The ld variant reads a leading-K view with larger physical row strides.
  const uint physical_k = K + (leading_view ? 128 : 0);
  const uint wld = physical_k / 8, sld = physical_k / 64;
  std::array<std::vector<uint>, NP> weights;
  std::array<std::vector<bfloat>, NP> scales, biases;
  for (int q = 0; q < NP; ++q) {
    weights[q].resize(size_t(N) * wld);
    scales[q].resize(size_t(N) * sld);
    biases[q].resize(size_t(N) * sld);
    for (size_t i = 0; i < weights[q].size(); ++i)
      weights[q][i] = uint(i * 0x9e3779b9u + uint(q) * 0x1234567u);
    for (size_t i = 0; i < scales[q].size(); ++i) {
      // Exactly representable bf16 values, including signs, zero, tiny and large.
      constexpr uint sb[] = {0x0000, 0x8000, 0x3a80, 0xba80, 0x3f80, 0xbf80, 0x4001, 0x0080};
      constexpr uint bb[] = {0x0000, 0x8000, 0x3e81, 0xbe81, 0x4080, 0xc080, 0x3f01, 0x8080};
      scales[q][i].value = std::bit_cast<float>(sb[(i + q) % 8] << 16);
      biases[q][i].value = std::bit_cast<float>(bb[(i * 3 + q) % 8] << 16);
    }
  }
  const uint tile_n = NP == 2 ? BN / 2 : BN;
  for (uint n0 = 0; n0 < N; n0 += tile_n) {
    for (uint tid = 0; tid < 128; ++tid) {
      QmtParams p{M, N, K, K, N, 0, 64, 0, 0};
      const uint* W[NP];
      const bfloat* S[NP];
      const bfloat* B[NP];
      for (int q = 0; q < NP; ++q) W[q] = weights[q].data();
      @MAPPING@
      QmtRaw wv[WP];
      uint current_k = 0, next_store = 0, products = 0, barriers = 0;
      auto load = [&](uint k0) {
        current_k = k0;
        for (int q = 0; q < NP; ++q) {
          // Reuse must not even access metadata: poison both pointers.
          const bool reuse = BK == 32 && (k0 & 63u) != 0;
          S[q] = reuse ? nullptr : scales[q].data();
          B[q] = reuse ? nullptr : biases[q].data();
        }
        metadata_reads.clear();
        @LOAD@
        const bool refresh = BK != 32 || (k0 & 63u) == 0;
        assert(metadata_reads.size() == (refresh ? 2 * WP : 0));
        size_t read = 0;
        for (int i = 0; i < WP; ++i) {
          const int q = NP == 2 && wr[i] >= BN / 2 ? 1 : 0;
          const uint n = NP == 2 ? min(n0 + wr[i] % (BN / 2), N - 1) : min(n0 + wr[i], N - 1);
          const uint k = k0 + wk[i];
          const size_t g = size_t(n) * sld + k / 64;
          const size_t w = size_t(n) * wld + k / 8;
          assert(wn[i] == n);
          assert(wv[i].w.x == weights[q][w] && wv[i].w.y == weights[q][w + 1]);
          assert(std::bit_cast<uint>(wv[i].s) == std::bit_cast<uint>(scales[q][g].value));
          assert(std::bit_cast<uint>(wv[i].b) == std::bit_cast<uint>(biases[q][g].value));
          if (refresh) {
            assert(metadata_reads[read++] == &scales[q][g]);
            assert(metadata_reads[read++] == &biases[q][g]);
          }
          alignas(16) half actual[16];
          qmt_dec(wv[i], actual);
          for (uint j = 0; j < 16; ++j) {
            const uint packed = weights[q][w + j / 8];
            const float nibble = float((packed >> (4 * (j % 8))) & 15u);
            const half expected = half(std::fma(nibble, scales[q][g].value, biases[q][g].value));
            assert(std::bit_cast<uint16_t>(actual[j]) == std::bit_cast<uint16_t>(expected));
            ++decoded;
          }
          ++loads;
        }
      };
      auto store = [&](uint b) {
        assert(current_k == next_store * BK && b == (next_store & 1));
        ++next_store;
      };
      struct Op {
        uint& products;
        void run(int buffer, int other, int&) {
          assert(buffer == int(products & 1) && other == buffer);
          ++products;
        }
      } op{products};
      const int tX0 = 0, tW0 = 0, tX1 = 1, tW1 = 1;
      int acc = 0;
      const int mem_flags_mem_threadgroup = 0;
      auto barrier = [&](int) { ++barriers; };
      @SCHEDULE@
      assert(next_store == K / BK && products == K / BK && barriers == K / BK + 1);
    }
  }
  ++cases;
}
int main() {
  for (uint N : {1u, 31u, 32u, 33u, 63u, 64u, 65u})
    for (uint K : {64u, 128u, 192u, 256u})
      for (uint M : {1u, 63u, 64u, 65u})
        for (bool view : {false, true}) {
          check<1, 64, 32>(N, K, M, view);
          check<2, 64, 32>(N, K, M, view);
          check<1, 64, 64>(N, K, M, view);
          check<2, 64, 64>(N, K, M, view);
        }
  // Exercise long group sequences at the measured models' projection K sizes.
  for (uint K : {5120u, 17408u})
    for (uint N : {1u, 33u}) {
      check<1, 64, 32>(N, K, 4096, false);
      check<2, 64, 32>(N, K, 2048, false);
      check<1, 64, 32>(N, K, 4096, true);
      check<2, 64, 64>(N, K, 2048, true);
    }
  std::printf("PASS: %llu cases, %llu packed loads, %llu decoded weights; BK32 metadata reads halved, BK64 refreshes every load\n",
              (unsigned long long)cases, (unsigned long long)loads, (unsigned long long)decoded);
}
"""


class QmtMetadataCacheTest(unittest.TestCase):
    def test_extracted_source(self):
        source = SOURCE.read_text()
        raw = block(source, "NI_INLINE void qmt_raw(")
        dec = block(source, "NI_INLINE void qmt_dec(")
        self.assertEqual(dec, DEC)
        mapping = source[source.index("  constexpr int WP ="):source.index("  uint4 xv[XP];")]
        load = block(source[source.index("  auto load ="):], "for (int i = 0; i < WP; ++i)")
        start = source.index("  load(0);")
        schedule = source[start:source.index("  if (NP == 2) {", start)]
        # Only the Metal barrier spelling needs translating in this schedule.
        schedule = schedule.replace("threadgroup_barrier(mem_flags::mem_threadgroup)",
                                    "barrier(mem_flags_mem_threadgroup)")
        body = HARNESS.replace("@MAPPING@", mapping).replace("@LOAD@", load).replace("@SCHEDULE@", schedule)
        struct = block(source, "struct QmtRaw") + ";\n"
        with tempfile.TemporaryDirectory(prefix="qmt-metadata-cpu-") as directory:
            cpp = Path(directory) / "check.cpp"
            binary = Path(directory) / "check"
            cpp.write_text(PREAMBLE + struct + raw + "\n" + dec + body)
            subprocess.run([os.environ.get("CXX", "clang++"), "-std=c++20", "-O1", "-g",
                            "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                            str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
