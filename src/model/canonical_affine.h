#pragma once

#include "model/affine_coeff_fixed.h"
#include <array>
#include <string>
#include <string_view>

namespace ni {
inline constexpr size_t kCanonicalScratchBytes = size_t(64) << 20;
inline constexpr size_t kCanonicalXpBytes = size_t(48) << 20;
inline constexpr size_t kCanonicalExtraLimit = size_t(512) << 20;
enum class CanonicalRole { Gate, Up, Down, GdnQkv, GdnZ, GdnA, GdnB, GdnOut, Q, K, V, O };
struct CanonicalKey {
  int layer = -1;
  CanonicalRole role = CanonicalRole::Gate;
  int part = -1;  // 0 W, 1 S, 2 B
  explicit operator bool() const { return layer >= 0 && part >= 0; }
};
inline CanonicalKey canonical_key(std::string_view name) {
  constexpr std::array<std::string_view, 3> prefixes = {
      "language_model.model.layers.", "model.layers.", "model.language_model.layers."};
  bool found = false;
  for (auto prefix : prefixes) if (name.starts_with(prefix)) { name.remove_prefix(prefix.size()); found = true; break; }
  if (!found || name.empty()) return {};
  int layer = 0;
  size_t n = 0;
  while (n < name.size() && name[n] >= '0' && name[n] <= '9') {
    layer = layer * 10 + int(name[n++] - '0');
    if (layer >= 32) return {};
  }
  if (!n || n == name.size() || name[n] != '.') return {};
  name.remove_prefix(n + 1);
  constexpr std::array<std::string_view, 3> suffixes = {".weight", ".scales", ".biases"};
  int part = -1;
  for (int p = 0; p < 3; ++p) if (name.ends_with(suffixes[p])) {
    part = p; name.remove_suffix(suffixes[p].size()); break;
  }
  if (part < 0) return {};
  constexpr std::array<std::string_view, 12> projections = {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
      "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a", "linear_attn.in_proj_b",
      "linear_attn.out_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"};
  for (size_t i = 0; i < projections.size(); ++i)
    if (name == projections[i]) return {layer, CanonicalRole(i), part};
  return {};
}
inline size_t canonical_align(size_t n, size_t align) {
  if (n > std::numeric_limits<size_t>::max() - (align - 1)) throw std::length_error("canonical alignment overflow");
  return (n + align - 1) / align * align;
}
struct CanonicalImageSize { size_t payload, headers, pages; };
inline CanonicalImageSize canonical_image_size(size_t blocks, size_t compressed) {
  if (!blocks || compressed > blocks || blocks > (0x80000000ull - 32) / 304)
    throw std::length_error("canonical image size overflow");
  const size_t raw = blocks - compressed, body = blocks * 272;
  const size_t payload = (raw ? ((body + 31) & ~size_t(31)) + raw * 32 : body) * 4;
  const size_t headers = blocks * 4;
  return {payload, headers, canonical_align(payload, 16384) + canonical_align(headers, 16384)};
}
// Same bytes as pack_affine_packed_coeff, written directly into final shared buffers.
inline void canonical_pack_into(uint32_t* payload, uint32_t* headers, const uint32_t* weights,
                                const AffineCoeffFixed& c, uint32_t outputs, uint32_t groups) {
  const auto size = canonical_image_size(c.blocks, c.compressed_blocks);
  if (!payload || !headers || !weights || !outputs || !groups ||
      c.blocks != ((size_t(outputs) + 15) / 16) * ((size_t(groups) + 1) / 2))
    throw std::invalid_argument("canonical pack arguments");
  std::fill_n(payload, size.payload / 4, 0u);
  const size_t pairs = (size_t(groups) + 1) / 2;
  size_t raw = (c.blocks * 272 + 31) & ~size_t(31);
  for (size_t block = 0; block < c.blocks; ++block) {
    const size_t first_row = (block / pairs) * 16, first_group = (block % pairs) * 2;
    const uint32_t h = c.words[block * 17];
    headers[block] = h;
    std::copy_n(c.words.data() + block * 17 + 1, 16, payload + block * 272 + 256);
    if (h & 0x80000000u) {
      headers[block] = 0x80000000u | uint32_t(raw);
      std::copy_n(c.words.data() + block * 17 + (h & 0x7fffffffu), 32, payload + raw);
      raw += 32;
    }
    for (uint32_t half = 0; half < 2; ++half) for (uint32_t f = 0; f < 2; ++f)
      for (uint32_t fm = 0; fm < 8; ++fm) {
        const size_t n = std::min(first_row + f * 8 + fm, size_t(outputs) - 1), g = first_group + half;
        if (g >= groups) continue;
        const uint32_t* src = weights + (n * groups + g) * 8;
        uint32_t* dst = payload + block * 272 + (half * 2 + f) * 64 + ((fm & 4) << 3) + ((fm & 3) << 2);
        std::copy_n(src, 4, dst); std::copy_n(src + 4, 4, dst + 16);
      }
  }
}
struct CanonicalSlice { size_t w, s, b, end; };
inline CanonicalSlice canonical_slice(size_t begin, uint32_t N, uint32_t K) {
  if (!N || !K || K % 64) throw std::invalid_argument("canonical scratch shape");
  const size_t wb = size_t(N) * K / 2, sb = size_t(N) * (K / 64) * 2;
  if (begin > kCanonicalScratchBytes || wb > kCanonicalScratchBytes || sb > kCanonicalScratchBytes)
    throw std::length_error("canonical consumer exceeds 64 MiB scratch");
  const size_t w = canonical_align(begin, 256), s = canonical_align(w + wb, 256),
               b = canonical_align(s + sb, 256), end = canonical_align(b + sb, 256);
  if (end > kCanonicalScratchBytes) throw std::length_error("canonical consumer exceeds 64 MiB scratch");
  return {w, s, b, end};
}
// XP representation has one fixed arena; overflow/unsupported shapes are rejected before encoding.
struct CanonicalXpSlice { size_t w = 0, sb = 0, end = 0; };
inline bool canonical_xp_slice(size_t begin, uint32_t N, uint32_t K, CanonicalXpSlice& out) {
  if (!N || !K || K % 64 || begin > kCanonicalScratchBytes) return false;
  const uint64_t wb = uint64_t(N) * K / 2, sb = uint64_t(N) * (K / 64) * 4;
  if (wb > kCanonicalScratchBytes || sb > kCanonicalScratchBytes) return false;
  const size_t w = canonical_align(begin, 256), coefficients = canonical_align(w + size_t(wb), 256);
  const size_t end = canonical_align(coefficients + size_t(sb), 256);
  if (end > kCanonicalScratchBytes) return false;
  out = {w, coefficients, end};
  return true;
}
inline bool canonical_xp_size(int M, uint32_t K, size_t& bytes) {
  if (M <= 0 || !K || K % 64) return false;
  const uint64_t mp = (uint64_t(M) + 31) / 32 * 32, elements = mp * K;
  if (elements > kCanonicalXpBytes / 2 || elements > UINT32_MAX) return false;
  bytes = size_t(elements) * 2;
  return true;
}
}  // namespace ni
