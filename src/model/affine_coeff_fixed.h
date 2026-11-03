#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ni {

inline constexpr uint32_t kAffineCoeffFixedBiasBase = 355;

// Lossless auxiliary image of BF16 affine coefficients. Each 16-output x
// 2-group block is 17 uints: a header, then one pair of ushort codes per output.
// Compressed headers duplicate S_min in each halfword; bias-code deltas use
// a fixed base. High-bit headers point to a raw slab relative to that header.
struct AffineCoeffFixed {
  std::vector<uint32_t> words;
  size_t compressed_blocks = 0;
  size_t blocks = 0;
};

inline AffineCoeffFixed pack_affine_coeff_fixed(const uint16_t* scales, const uint16_t* biases,
                                               uint32_t outputs, uint32_t groups) {
  if (!outputs || !groups) throw std::invalid_argument("empty affine coefficient image");
  const size_t nb = (size_t(outputs) + 15) / 16, gp = (size_t(groups) + 1) / 2;
  if (gp > std::numeric_limits<size_t>::max() / nb / 17)
    throw std::length_error("affine coefficient image too large");
  AffineCoeffFixed out{std::vector<uint32_t>(nb * gp * 17), 0, nb * gp};
  for (size_t n = 0; n < nb; ++n) for (size_t p = 0; p < gp; ++p) {
    int slo = 65535, shi = 0, dlo = 65535, dhi = -65535;
    bool valid = true;
    for (size_t row = n * 16; row < std::min((n + 1) * 16, size_t(outputs)); ++row)
      for (size_t g = p * 2; g < std::min(p * 2 + 2, size_t(groups)); ++g) {
        const uint16_t s = scales[row * groups + g], b = biases[row * groups + g];
        const int sm = s & 0x7fff, bm = b & 0x7fff, d = bm - sm;
        valid &= ((s ^ b) & 0x8000) != 0 && sm < 0x7f80 && bm < 0x7f80;
        slo = std::min(slo, sm); shi = std::max(shi, sm);
        dlo = std::min(dlo, d); dhi = std::max(dhi, d);
      }
    uint32_t* block = out.words.data() + (n * gp + p) * 17;
    if (!valid || shi - slo > 255 || dlo < int(kAffineCoeffFixedBiasBase) || dhi > int(kAffineCoeffFixedBiasBase + 127)) {
      block[0] = 0xffffffffu;
      continue;
    }
    block[0] = uint32_t(slo) * 0x10001u;
    ++out.compressed_blocks;
    for (size_t row = n * 16; row < std::min((n + 1) * 16, size_t(outputs)); ++row)
      for (size_t g = p * 2; g < std::min(p * 2 + 2, size_t(groups)); ++g) {
        const uint16_t s = scales[row * groups + g], b = biases[row * groups + g];
        const int sm = s & 0x7fff, bm = b & 0x7fff;
        const uint32_t code = uint32_t(sm - slo) | (uint32_t(bm - sm - int(kAffineCoeffFixedBiasBase)) << 8) | (s & 0x8000u);
        block[1 + row % 16] |= code << (16 * (g % 2));
      }
  }
  const size_t raw_blocks = out.blocks - out.compressed_blocks;
  if (raw_blocks) {
    // One alignment pad per image, then 128-byte raw slabs. Keeping both raw
    // groups of a row together permits one aligned uint2 load on the GPU.
    size_t next = (out.words.size() + 31) & ~size_t(31);
    if (next >= 0x80000000ull || raw_blocks > (0x80000000ull - next) / 32)
      throw std::length_error("affine raw coefficient offset too large");
    out.words.resize(next + raw_blocks * 32);
    for (size_t n = 0; n < nb; ++n) for (size_t p = 0; p < gp; ++p) {
      const size_t pos = (n * gp + p) * 17;
      if (out.words[pos] != 0xffffffffu) continue;
      out.words[pos] = 0x80000000u | uint32_t(next - pos);
      for (size_t row = n * 16; row < std::min((n + 1) * 16, size_t(outputs)); ++row)
        for (size_t g = p * 2; g < std::min(p * 2 + 2, size_t(groups)); ++g) {
          out.words[next + (row % 16) * 2] |= uint32_t(scales[row * groups + g]) << (16 * (g % 2));
          out.words[next + (row % 16) * 2 + 1] |= uint32_t(biases[row * groups + g]) << (16 * (g % 2));
        }
      next += 32;
    }
  }
  return out;
}

// Q4 payload in the SIMD's execution order, with the same lossless coefficient
// codes. Headers are separate so every 1088-byte body starts on a 64-byte edge.
struct AffinePackedCoeff {
  std::vector<uint32_t> words, headers;
  size_t compressed_blocks = 0, blocks = 0;
};
inline AffinePackedCoeff pack_affine_packed_coeff(const uint32_t* weights, const uint16_t* scales,
                                                 const uint16_t* biases, uint32_t outputs, uint32_t groups) {
  const auto coefficients = pack_affine_coeff_fixed(scales, biases, outputs, groups);
  if (coefficients.blocks > (0x80000000ull - 32) / 304)
    throw std::length_error("packed affine payload offset too large");
  const size_t raw_count = coefficients.blocks - coefficients.compressed_blocks;
  const size_t body_end = coefficients.blocks * 272;
  size_t raw = (body_end + 31) & ~size_t(31);
  AffinePackedCoeff out{std::vector<uint32_t>(raw_count ? raw + raw_count * 32 : body_end),
                        std::vector<uint32_t>(coefficients.blocks), coefficients.compressed_blocks, coefficients.blocks};
  const size_t pairs = (size_t(groups) + 1) / 2;
  for (size_t block = 0; block < out.blocks; ++block) {
    const size_t first_row = (block / pairs) * 16, first_group = (block % pairs) * 2;
    const uint32_t h = coefficients.words[block * 17];
    out.headers[block] = h;
    std::copy_n(coefficients.words.data() + block * 17 + 1, 16, out.words.data() + block * 272 + 256);
    if (h & 0x80000000u) {
      const size_t original_raw = block * 17 + (h & 0x7fffffffu);
      out.headers[block] = 0x80000000u | uint32_t(raw);
      std::copy_n(coefficients.words.data() + original_raw, 32, out.words.data() + raw);
      raw += 32;
    }
    for (uint32_t half = 0; half < 2; ++half) for (uint32_t f = 0; f < 2; ++f)
      for (uint32_t lane = 0; lane < 32; ++lane) {
        const uint32_t q = lane >> 2, fm = (q & 4) | ((lane >> 1) & 3);
        const uint32_t fn = ((q & 2) << 1) | ((lane & 1) << 1);
        const size_t n = std::min(first_row + f * 8 + fm, size_t(outputs) - 1);
        const size_t g = first_group + half;
        if (g < groups) std::copy_n(weights + (n * groups + g) * 8 + fn, 2,
                                    out.words.data() + block * 272 + (half * 2 + f) * 64 + lane * 2);
      }
  }
  return out;
}

}  // namespace ni
