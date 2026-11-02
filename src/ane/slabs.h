// Row slabs of a noncausal prefill pass longer than the Neural Engine split's largest function
// (NI_ANE_NONCAUSAL, Qwen35::enc_mlp_prefill_slabs). A noncausal pass runs whole, so its FFN is
// split slab by slab: ceil(T / max_rows) slabs of balanced size, each rounded up to 64 rows (the
// GPU part's GEMM tile) and at most max_rows; the last slab takes the remainder.
#pragma once

namespace ni::ane {

constexpr int slab_count(int T, int max_rows) { return T <= 0 ? 0 : (T + max_rows - 1) / max_rows; }

// rows of every slab but the last
constexpr int slab_rows(int T, int max_rows) {
  const int n = slab_count(T, max_rows);
  if (n <= 1) return T;
  const int r = ((T + n - 1) / n + 63) / 64 * 64;
  return r < max_rows ? r : max_rows;
}

// rows of the last slab
constexpr int last_slab_rows(int T, int max_rows) {
  const int n = slab_count(T, max_rows);
  return n <= 1 ? T : T - (n - 1) * slab_rows(T, max_rows);
}

static_assert(slab_count(2048, 2048) == 1 && slab_rows(2048, 2048) == 2048);
static_assert(slab_count(4096, 2048) == 2 && slab_rows(4096, 2048) == 2048 && last_slab_rows(4096, 2048) == 2048);
static_assert(slab_count(16384, 2048) == 8 && slab_rows(16384, 2048) == 2048 && last_slab_rows(16384, 2048) == 2048);
static_assert(slab_rows(2049, 2048) == 1088 && last_slab_rows(2049, 2048) == 961);
static_assert(slab_rows(5000, 2048) == 1728 && last_slab_rows(5000, 2048) == 1544);
static_assert(last_slab_rows(32769, 2048) > 0 && last_slab_rows(32769, 2048) <= 2048);

}  // namespace ni::ane
