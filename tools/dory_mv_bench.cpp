// Dory's NVFP4 matrix products in isolation, on its four shapes (q|k|v|g 2560 -> 10240, o 4096 -> 2560, gate|up
// 2560 -> 14336, down 7168 -> 2560), random codes: one row on nh_x4_mv_l0 (decode), 2 / 4 / 8 rows on the 8-row MMA
// kernel nh_x4_v8_l0 (half rows: what a verify pass runs) and on the generic nh_mv_fp4_r* (fp32 rows).
//   dory_mv_bench [iters]
// Each case: `iters` dispatches back to back in one command buffer (dependent, as decoding runs them), best of 5
// buffers; us per product and GB/s of weight bytes.
#include "metal/gpu.h"
#include "util/log.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace ni {
extern const char* const kNhKernelSource;
}  // namespace ni
using namespace ni;

namespace {
struct MvP {
  uint32_t N, K, x_stride, y_stride;
  float gs;
  uint32_t R;
};
double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
  Gpu& g = Gpu::get();
  g.compile_extra("Nemotron-H", kNhKernelSource);
  struct Shape {
    const char* name;
    uint32_t N, K;
  };
  const Shape shapes[] = {{"qkvg 2560->10240", 10240, 2560}, {"o 4096->2560", 2560, 4096},
                          {"gate|up 2560->14336", 14336, 2560}, {"down 7168->2560", 2560, 7168}};
  struct Kern {
    const char* name;
    uint32_t rows_per_sg, sgs;
  };
  const Kern kerns[] = {{"nh_x4_mv_l0", 4, 2}};
  std::mt19937 rng(7);
  for (const Shape& sh : shapes) {
    const size_t qb = size_t(sh.N) * sh.K / 2, sb = size_t(sh.N) * sh.K / 16;
    // several weight copies so successive products do not hit the cache (> 48 MB in all)
    const int copies = int(std::max<size_t>(4, (size_t(96) << 20) / (qb + sb) + 1));
    const size_t ncopy = size_t(copies);
    std::vector<Buffer> Q(ncopy), S(ncopy);
    for (int c = 0; c < copies; ++c) {
      Q[size_t(c)] = g.alloc(qb, "bench_q");
      S[size_t(c)] = g.alloc(sb, "bench_s");
      uint8_t* q = Q[size_t(c)].as<uint8_t>();
      uint8_t* s = S[size_t(c)].as<uint8_t>();
      for (size_t i = 0; i < qb; ++i) q[i] = uint8_t(rng());
      for (size_t i = 0; i < sb; ++i) s[i] = uint8_t(0x30 + rng() % 16);  // UE4M3 around 1
    }
    Buffer x = g.alloc(size_t(sh.K) * 4, "bench_x"), y = g.alloc(size_t(sh.N) * 4, "bench_y");
    std::normal_distribution<float> nd(0.0f, 0.02f);
    for (uint32_t i = 0; i < sh.K; ++i) x.as<float>()[i] = nd(rng);
    const MvP p{sh.N, sh.K, sh.K, sh.N, 0.01f, 1};
    printf("%s (%.1f MB of weights):\n", sh.name, double(qb + sb) / 1e6);
    for (const Kern& k : kerns) {
      const Pipeline pl = g.pipeline(k.name);
      const uint32_t per_tg = k.rows_per_sg * k.sgs;
      auto enc = [&](Encoder& e, int c, const Buffer& out) {
        e.use(pl);
        e.buf(0, Q[size_t(c)]);
        e.buf(1, S[size_t(c)]);
        e.buf(2, x);
        e.buf(3, out);
        e.val(4, p);
        e.dispatch((sh.N + per_tg - 1) / per_tg, 1, 1, 32 * k.sgs);
      };
      double best = 1e30;
      for (int rep = 0; rep < 5; ++rep) {
        auto e = g.begin();
        for (int i = 0; i < iters; ++i) enc(*e, i % copies, y);
        const double t0 = now_s();
        e->commit_and_wait();
        best = std::min(best, now_s() - t0);
      }
      const double us = best / iters * 1e6;
      printf("  R=1 %-18s %7.1f us  %6.1f GB/s\n", k.name, us, double(qb + sb) / (us * 1e3));
    }
    // R activation rows: Nemotron-H's generic nh_mv_fp4_r<R> (fp32 rows), the 8-row MMA kernel nh_x4_v8_l0 (half rows)
    for (uint32_t R : {2u, 4u, 8u}) {
      Buffer xr = g.alloc(size_t(R) * sh.K * 4, "bench_xr"), xh = g.alloc(size_t(R) * sh.K * 2, "bench_xh"),
             yr = g.alloc(size_t(R) * sh.N * 4, "bench_yr");
      for (uint32_t r = 0; r < R; ++r)
        for (uint32_t i = 0; i < sh.K; ++i) {
          const float v = r == 0 ? x.as<float>()[i] : nd(rng);
          xr.as<float>()[size_t(r) * sh.K + i] = v;
          xh.as<__fp16>()[size_t(r) * sh.K + i] = __fp16(v);
        }
      const MvP pr{sh.N, sh.K, sh.K, sh.N, 0.01f, R};
      struct V8P {
        uint32_t N, K, x_ld, y_ld, R;
        float gs;
        uint32_t epi;
      } pv{sh.N, sh.K, sh.K, sh.N, R, 0.01f, 0};
      const std::string gen = "nh_mv_fp4_r" + std::to_string(R);
      for (int kind = 1; kind < 3; ++kind) {
        const Pipeline pl = g.pipeline(kind == 1 ? gen : std::string("nh_x4_v8_l0"));
        auto enc = [&](Encoder& e, int c) {
          e.use(pl);
          if (kind == 2) {
            e.buf(0, xh);
            e.buf(1, Q[size_t(c)]);
            e.buf(2, S[size_t(c)]);
            e.buf(3, yr);
            e.val(4, pv);
            e.dispatch((sh.N + 15) / 16, 1, 1, 256);
            return;
          }
          e.buf(0, Q[size_t(c)]);
          e.buf(1, S[size_t(c)]);
          e.buf(2, xr);
          e.buf(3, yr);
          e.val(4, pr);
          const uint32_t nr = R <= 2 ? 4 : R == 4 ? 2 : 1;
          e.dispatch((sh.N + nr * 4 - 1) / (nr * 4), 1, 1, 128);
        };
        double best = 1e30;
        for (int rep = 0; rep < 5; ++rep) {
          auto e = g.begin();
          for (int i = 0; i < iters; ++i) enc(*e, i % copies);
          const double t0 = now_s();
          e->commit_and_wait();
          best = std::min(best, now_s() - t0);
        }
        const double us = best / iters * 1e6;
        printf("  R=%u %-18s %7.1f us  %6.1f GB/s\n", R, kind == 1 ? gen.c_str() : "nh_x4_v8_l0", us,
               double(qb + sb) / (us * 1e3));
      }
    }
  }
  return 0;
}
