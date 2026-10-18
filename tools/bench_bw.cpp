// DRAM read-bandwidth ceiling for the decode matvec sizes: streaming read kernels (no math) over rotating buffers.
//   bench_bw [MB per buffer ...]   (default 6 12 27 357)
// Kernels (appended to the library at compile time): bw16 (grid-stride uint4 loads), bw16x4 (4 independent uint4
// loads per iteration), bwrow (the matvec access pattern: a simdgroup reads 4 rows of 1280 bytes, 8 bytes per lane
// per row per step).
#include "metal/gpu.h"
#include "util/log.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

using namespace ni;

static const char* kSrc = R"MSL(
[[kernel]] void bw16(const device uint4* p [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& n [[buffer(2)]],
                     uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
  uint4 a = 0;
  for (uint i = gid; i < n; i += gsz) a ^= p[i];
  if (all(a == uint4(0x9e3779b9u))) out[0] = 1;
}
[[kernel]] void bw16x4(const device uint4* p [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& n [[buffer(2)]],
                       uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
  uint4 a = 0, b = 0, c = 0, d = 0;
  uint i = gid;
  for (; i + 3 * gsz < n; i += 4 * gsz) { a ^= p[i]; b ^= p[i + gsz]; c ^= p[i + 2 * gsz]; d ^= p[i + 3 * gsz]; }
  for (; i < n; i += gsz) a ^= p[i];
  a ^= b ^ c ^ d;
  if (all(a == uint4(0x9e3779b9u))) out[0] = 1;
}
// rows of 1280 bytes (K = 2560 at 4 bits): simdgroup = 4 rows, lane reads 8 bytes per row per step (5 steps)
[[kernel]] void bwrow(const device uint2* p [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& n [[buffer(2)]],
                      uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint row0 = (tg * nsg + sg) * 4;
  const uint wpr = 160;  // uint2 per row
  uint2 a = 0;
  for (uint s = 0; s < 5; ++s)
    for (uint r = 0; r < 4; ++r) {
      const uint i = (row0 + r) * wpr + s * 32 + lane;
      if (i < n) a ^= p[i];
    }
  if (all(a == uint2(0x9e3779b9u))) out[0] = 1;
}
)MSL";

int main(int argc, char** argv) {
  std::vector<double> mbs;
  for (int i = 1; i < argc; ++i) mbs.push_back(atof(argv[i]));
  if (mbs.empty()) mbs = {6, 12, 27, 357};
  Gpu& g = Gpu::get();
  g.compile(kSrc);
  Buffer out = g.alloc(64);
  for (double mb : mbs) {
    const size_t bytes = size_t(mb * 1e6) / 1280 * 1280;
    const int copies = std::clamp(int(1.2e9 / double(bytes)), 2, 200);
    std::vector<Buffer> bufs;
    for (int c = 0; c < copies; ++c) {
      bufs.push_back(g.alloc(bytes, nullptr, false));
      memset(bufs.back().ptr(), c + 1, bytes);
    }
    for (const char* kn : {"bw16", "bw16x4", "bwrow"}) {
      Pipeline p = g.pipeline(kn);
      std::vector<double> t;
      for (int rep = 0; rep < 5; ++rep) {
        auto e = g.begin();
        e->use(p);
        for (auto& b : bufs) {
          e->buf(0, b);
          e->buf(1, out);
          if (!strcmp(kn, "bwrow")) {
            const uint32_t n = uint32_t(bytes / 8);
            e->val(2, n);
            const uint32_t rows = uint32_t(bytes / 1280);
            e->dispatch((rows + 7) / 8, 1, 1, 64);
          } else {
            const uint32_t n = uint32_t(bytes / 16);
            e->val(2, n);
            e->dispatch(20 * 8, 1, 1, 256);
          }
        }
        e->commit_and_wait();
        t.push_back(e->gpu_ms() / copies);
      }
      std::sort(t.begin(), t.end());
      printf("%-7s %7.1f MB  %8.2f us  %6.1f GB/s (best %6.1f)\n", kn, bytes / 1e6, t[2] * 1000, bytes / (t[2] * 1e-3) / 1e9,
             bytes / (t[0] * 1e-3) / 1e9);
    }
    for (auto& b : bufs) g.release(b);
  }
  return 0;
}
