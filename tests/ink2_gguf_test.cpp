#include "gguf/gguf.h"
#include "model/inkling2_gguf.h"
#include "gguf/quant.h"
#include "kernels/inkling2/gguf_blocks.h"

#include <bit>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
  std::mt19937 rng(754);
  size_t checked = 0;
  for (unsigned type : {6u, 8u, 12u, 13u, 14u, 20u, 23u}) {
    const unsigned n = ink2_gguf::block_values(type), bytes = ink2_gguf::block_bytes(type);
    for (int trial = 0; trial < 1000; ++trial) {
      std::vector<unsigned char> b(bytes * 3);
      for (auto& v : b) v = static_cast<unsigned char>(rng());
      // Finite, nonzero half scales; exercise arbitrary bitplanes, signs, subscales and adjacent blocks.
      for (unsigned block = 0; block < 3; ++block) {
        const unsigned off = block * bytes + (type == 14 ? 208 : 0);
        b[off + 1] = static_cast<unsigned char>(0x08 + rng() % 0x68);
        if (type == 12 || type == 13) b[off + 3] = static_cast<unsigned char>(0x08 + rng() % 0x68);
      }
      std::vector<float> ref(n * 3);
      ni::gguf::dequantize(type, b.data(), ref.data(), ref.size());
      for (unsigned k = 0; k < ref.size(); ++k) {
        const float got = ink2_gguf::value(b.data(), type, k);
        if (got != ref[k]) {
          std::fprintf(stderr, "type %u trial %d k %u: %.9g != %.9g\n", type, trial, k, got, ref[k]);
          return 1;
        }
        ++checked;
      }
    }
  }
  for (unsigned K : {64u, 1152u, 2304u}) {
    for (int trial = 0; trial < 100; ++trial) {
      std::vector<uint8_t> raw(K / 64 * 36), q(K / 2), scales(K / 16);
      for (auto& b : raw) b = uint8_t(rng());
      for (size_t block = 0; block < K / 64; ++block)
        for (size_t s = 0; s < 4; ++s) raw[block * 36 + s] = uint8_t(size_t(trial) * 4 + block * 4 + s);
      ni::ink2::unpack_nvfp4_row(raw.data(), K, q.data(), scales.data());
      std::vector<float> ref(K);
      ni::gguf::dequantize(ni::gguf::kNVFP4, raw.data(), ref.data(), K);
      constexpr float levels[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, 0, -.5f, -1, -1.5f, -2, -3, -4, -6};
      for (unsigned k = 0; k < K; ++k) {
        const uint8_t s = scales[k / 16];
        const unsigned ex = (s >> 3) & 15, mant = s & 7;
        const float scale = ex ? std::ldexp(1.0f + float(mant) / 8, int(ex) - 7) : std::ldexp(float(mant), -9);
        const float got = scale * levels[(q[k / 2] >> ((k % 2) * 4)) & 15];
        if (got != ref[k]) {
          std::fprintf(stderr, "NVFP4 repack K=%u k=%u: %.9g != %.9g\n", K, k, got, ref[k]);
          return 1;
        }
        ++checked;
      }
      std::vector<uint8_t> q8(K / 32 * 34), codes(K);
      std::vector<uint16_t> half_scales(K / 32);
      for (auto& b : q8) b = uint8_t(rng());
      ni::ink2::unpack_q8_row(q8.data(), K, codes.data(), half_scales.data());
      for (unsigned block = 0; block < K / 32; ++block)
        if (std::memcmp(q8.data() + block * 34, &half_scales[block], 2) ||
            std::memcmp(q8.data() + block * 34 + 2, codes.data() + block * 32, 32)) return 1;
    }
  }
  for (size_t row = 0; row < 2304; ++row) {
    const size_t native = ni::ink2::native_gu_row(row, 1152);
    if (native / 2 + (native % 2) * 1152 != row) return 1;
  }
  std::printf("ink2 GGUF: %zu scalar reads/repacked NVFP4 values match CPU dequantization; Q8 bytes and 2304 gate/up rows match\n", checked);
}
