#include <sys/resource.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "util/bf16.h"
namespace ni {
bool dfl_lowrank_factors(const uint32_t*, const uint16_t*, const uint16_t*, uint32_t, uint32_t, uint32_t,
                         std::vector<float>&, std::vector<float>&);
}
int main(int argc, char** argv) {
  if (argc != 5) return 2;
  unsigned N = atoi(argv[1]), K = atoi(argv[2]), r = atoi(argv[3]);
  std::vector<uint32_t> w(size_t(N) * K / 8);
  uint32_t rng = 42;
  for (auto& q : w) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    q = rng;
  }
  std::vector<uint16_t> s(size_t(N) * K / 64, ni::f32_to_bf16(0.0625f)), b(s.size(), ni::f32_to_bf16(-0.5f));
  std::vector<float> A, P;
  auto start = std::chrono::steady_clock::now();
  bool ok = ni::dfl_lowrank_factors(w.data(), s.data(), b.data(), N, K, r, A, P);
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  if (!ok) return 1;
  std::ofstream f(argv[4], std::ios::binary);
  f.write((char*)A.data(), A.size() * 4);
  f.write((char*)P.data(), P.size() * 4);
  struct rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  printf("N%u K%u r%u %.3f ms maxrss %ld bytes outputs %zu bytes\n", N, K, r, ms, ru.ru_maxrss,
         (A.size() + P.size()) * 4);
}
