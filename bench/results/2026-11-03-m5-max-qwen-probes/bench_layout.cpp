// Load-only bandwidth of three weight layouts in qmm8t's access pattern (16 weight rows a simdgroup, one 64-group a
// step): split (codes row-major + bf16 scale and bias arrays, 2-byte loads), sbpair (one uint per row-group: scale |
// bias << 16), tiled (per 16-row block and group: 512 B of codes then 64 B of scale/bias pairs, contiguous).
#include "metal/gpu.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace ni;
static const char* kSrc = R"MSL(
struct LP { uint N, K, ks; };
[[kernel]] void ld_split(const device uint* W [[buffer(0)]], const device ushort* S [[buffer(1)]],
                         const device ushort* B [[buffer(2)]], device uint* out [[buffer(3)]], constant LP& p [[buffer(4)]],
                         uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint gpr = p.K / 64, wpr = p.K / 8, n0 = tg * 16;
  const uint gper = gpr / p.ks, g0 = sg * gper, g1 = g0 + gper;
  const uint c = (lane & 1) | (((lane >> 3) & 1) << 1), r = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
  uint a = 0;
  for (uint g = g0; g < g1; ++g)
    for (int j = 0; j < 2; ++j) {
      const uint row = n0 + r + 8 * j;
      const uint2 w = *(const device uint2*)(W + row * wpr + 2 * c + g * 8);
      a ^= w.x ^ w.y ^ uint(S[row * gpr + g]) ^ (uint(B[row * gpr + g]) << 16);
    }
  if (a == 0x9e3779b9u) out[0] = 1;
}
[[kernel]] void ld_sbpair(const device uint* W [[buffer(0)]], const device uint* SB [[buffer(1)]],
                          const device ushort* B [[buffer(2)]], device uint* out [[buffer(3)]], constant LP& p [[buffer(4)]],
                          uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint gpr = p.K / 64, wpr = p.K / 8, n0 = tg * 16;
  const uint gper = gpr / p.ks, g0 = sg * gper, g1 = g0 + gper;
  const uint c = (lane & 1) | (((lane >> 3) & 1) << 1), r = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
  uint a = 0;
  for (uint g = g0; g < g1; ++g)
    for (int j = 0; j < 2; ++j) {
      const uint row = n0 + r + 8 * j;
      const uint2 w = *(const device uint2*)(W + row * wpr + 2 * c + g * 8);
      a ^= w.x ^ w.y ^ SB[row * gpr + g];
    }
  if (a == 0x9e3779b9u) out[0] = 1;
}
[[kernel]] void ld_tiled(const device uint* W [[buffer(0)]], const device ushort* S [[buffer(1)]],
                         const device ushort* B [[buffer(2)]], device uint* out [[buffer(3)]], constant LP& p [[buffer(4)]],
                         uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint gpr = p.K / 64;
  const uint gper = gpr / p.ks, g0 = sg * gper, g1 = g0 + gper;
  const device uint* blk = W + size_t(tg) * gpr * 144;  // 576 B = 144 words a group
  uint a = 0;
  for (uint g = g0; g < g1; ++g) {
    const device uint* t = blk + g * 144;
    const uint4 w = *(const device uint4*)(t + 4 * lane);
    a ^= w.x ^ w.y ^ w.z ^ w.w;
    if (lane < 16) a ^= t[128 + lane];
  }
  if (a == 0x9e3779b9u) out[0] = 1;
}
)MSL";
int main(int argc, char** argv) {
  const uint32_t N = argc > 1 ? atoi(argv[1]) : 12352, K = argc > 2 ? atoi(argv[2]) : 2048;
  Gpu& g = Gpu::get();
  g.compile(kSrc);
  Buffer out = g.alloc(64);
  const size_t wb = size_t(N) * K / 2, sb = size_t(N) * (K / 64) * 2;
  const size_t bytes = wb + 2 * sb;
  const int copies = std::clamp(int(1.5e9 / double(bytes)), 2, 200);
  std::vector<Buffer> W, S, B;
  for (int c = 0; c < copies; ++c) {
    W.push_back(g.alloc(wb + 2 * sb, nullptr, false)); memset(W.back().ptr(), c + 1, wb + 2 * sb);
    S.push_back(g.alloc(2 * sb, nullptr, false)); memset(S.back().ptr(), c + 2, 2 * sb);
    B.push_back(g.alloc(sb, nullptr, false)); memset(B.back().ptr(), c + 3, sb);
  }
  printf("N %u K %u: %.2f MB a matrix (codes %.2f + scales/biases %.2f), %d copies\n", N, K, bytes / 1e6, wb / 1e6, 2 * sb / 1e6, copies);
  for (uint32_t ks : {1u, 2u, 4u}) {
    for (int rep = 0; rep < 2; ++rep)
      for (const char* kn : {"ld_split", "ld_sbpair", "ld_tiled"}) {
        Pipeline p = g.pipeline(kn);
        std::vector<double> t;
        for (int it = 0; it < 7; ++it) {
          auto e = g.begin();
          e->use(p);
          struct { uint32_t N, K, ks; } lp{N, K, ks};
          for (int c = 0; c < copies; ++c) {
            e->buf(0, W[c]); e->buf(1, S[c]); e->buf(2, B[c]); e->buf(3, out); e->val(4, lp);
            e->dispatch(N / 16, 1, 1, 32 * ks);
          }
          e->commit_and_wait();
          t.push_back(e->gpu_ms() / copies);
        }
        std::sort(t.begin(), t.end());
        if (rep) printf("ks %u %-10s %8.2f us  %6.1f GB/s (best %6.1f)\n", ks, kn, t[3] * 1000, bytes / (t[3] * 1e-3) / 1e9, bytes / (t[0] * 1e-3) / 1e9);
      }
  }
  return 0;
}
