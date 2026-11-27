// Cache identity for DFlash2's low-rank factors: no model, no GPU.
#include "model/dflash2_lowrank_cache.h"
#include <cstdio>
#include <vector>

namespace ni {
bool dfl_lowrank_factors(const uint32_t*, const uint16_t*, const uint16_t*, uint32_t, uint32_t, uint32_t,
                         std::vector<float>&, std::vector<float>&);
}

int main() {
  using namespace ni;
  constexpr uint32_t N = 128, K = 64, r = 16;
  std::vector<uint32_t> w(size_t(N) * K / 8);
  uint32_t rng = 42;
  for (auto& q : w) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; q = rng; }
  std::vector<uint16_t> sc(N, 0x3d80), bi(N, 0xbf00);
  auto key = [&] { return dfl_lowrank_cache_key(w.data(), sc.data(), bi.data(), N, K, r); };
  const std::string original = key();
  int fails = 0;
  auto check = [&](bool ok, const char* why) { if (!ok) { ++fails; printf("FAIL %s\n", why); } };
  check(original.size() == 64 && key() == original, "same complete head has the same SHA256 key");
  std::vector<float> A, P, changed_A, changed_P;
  check(dfl_lowrank_factors(w.data(), sc.data(), bi.data(), N, K, r, A, P), "reference factors");
  // Coefficients and shape stay identical. The former key therefore collided for these two different heads.
  w.back() ^= 0xffffffffu;
  check(key() != original, "Q4 codes, including the final word, invalidate the cache");
  check(dfl_lowrank_factors(w.data(), sc.data(), bi.data(), N, K, r, changed_A, changed_P), "changed factors");
  check(A != changed_A || P != changed_P, "different codes really change the cached factors");
  w.back() ^= 0xffffffffu;
  sc.back() ^= 1;
  check(key() != original, "scales invalidate the cache");
  sc.back() ^= 1;
  bi.back() ^= 1;
  check(key() != original, "biases invalidate the cache");
  bi.back() ^= 1;
  check(dfl_lowrank_cache_key(w.data(), sc.data(), bi.data(), N, K, r / 2) != original, "rank invalidates the cache");
  check(dfl_lowrank_cache_key(w.data(), sc.data(), bi.data(), N / 2, K, r) != original, "shape invalidates the cache");
  check(key() == original, "restoring the head restores its key");
  printf("dflash_lowrank_cache_test: %s\n", fails ? "FAILED" : "ALL PASSED");
  return fails ? 1 : 0;
}
