#pragma once
#include "model/sampler.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ni::ink2draft {
constexpr uint32_t H = 2304, HQ = 24, HKV = 8, D = 128, I = 9216, R = 256, WIDTH = 8, CHUNK = 512;
constexpr uint32_t window(size_t layer) { return layer == 1 ? 0 : 1024; }
constexpr size_t last_bytes = H * sizeof(float);
inline size_t add(size_t a, size_t b) {
  if (a > std::numeric_limits<size_t>::max() - b) throw std::overflow_error("ink2 draft: byte sum overflow");
  return a + b;
}
inline size_t mul(size_t a, size_t b) {
  if (b && a > std::numeric_limits<size_t>::max() / b) throw std::overflow_error("ink2 draft: byte product overflow");
  return a * b;
}
inline void fits(size_t bytes, size_t limit) {
  if (bytes > limit) throw std::length_error("ink2 draft: allocation exceeds maxBufferLength");
}
inline size_t kv_bytes(uint32_t cap) { return mul(cap, HKV * D * 2); }
inline size_t empty_seq_bytes() {
  size_t bytes = last_bytes;
  for (size_t i = 0; i < 3; ++i) bytes = add(bytes, mul(2, kv_bytes(window(i))));
  return bytes;
}
inline uint32_t append_end(uint32_t pos, int rows) {
  if (rows < 1 || uint64_t(pos) + uint32_t(rows) > uint64_t(std::numeric_limits<int>::max()))
    throw std::overflow_error("ink2 draft: invalid append or context overflow");
  return pos + uint32_t(rows);
}
struct Growth { uint32_t cap; size_t one_buffer, allocation_bytes; };
inline Growth growth(uint32_t cap, uint32_t need, size_t limit) {
  if (need <= cap) return {cap, 0, 0};
  const uint64_t next = std::max({uint64_t(4096), uint64_t(need), uint64_t(cap) * 2});
  if (next > std::numeric_limits<uint32_t>::max()) throw std::overflow_error("ink2 draft: capacity overflow");
  const size_t one = kv_bytes(uint32_t(next));
  fits(one, limit);
  return {uint32_t(next), one, mul(2, one)};
}
enum class Scratch {
  blend, rnn_condition, rnn_recurrent, mixed, hidden, residual, summed, normed, q, k, v, rotated, attended,
  projection, gate, up, logits, scores, condition_input, condition, recurrent_input, recurrent, state, signal,
  picks, cand_v, cand_i, probability, fold_transposed, fold_tile, folded, count
};
struct SharedLayout {
  std::array<size_t, size_t(Scratch::count)> bytes;
  explicit SharedLayout(uint32_t vocab) {
    const size_t V = vocab;
    const uint32_t lists = topk_lists(vocab, 1);
    bytes = {16 * 4, 3 * R * 2 * H * 4, 3 * R * 2 * R * 4, CHUNK * H * 4,
      WIDTH * H * 4, WIDTH * H * 4, WIDTH * H * 4, WIDTH * H * 4, WIDTH * HQ * D * 4,
      CHUNK * HKV * D * 4, CHUNK * HKV * D * 4, CHUNK * HKV * D * 4, WIDTH * HQ * D * 4,
      WIDTH * H * 4, WIDTH * I * 4, WIDTH * I * 4, mul(7 * 4, V), mul(4, V),
      7 * 2 * H * 4, 7 * 3 * R * 4, 2 * R * 4, 3 * R * 4, R * 4, R * 4,
      WIDTH * sizeof(int), lists * 4, lists * 4, 4, (R + 1) * H * 4, (R + 1) * 4096 * 4,
      mul((R + 1) * 2, V)};
  }
  size_t operator[](Scratch id) const { return bytes[size_t(id)]; }
  size_t total() const {
    size_t n = 0;
    for (size_t b : bytes) n = add(n, b);
    return n;
  }
  void check_limit(size_t limit) const { for (size_t b : bytes) fits(b, limit); }
};
}  // namespace ni::ink2draft
