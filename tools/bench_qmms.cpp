// Small-T quantized GEMM benchmark + correctness: qmms_b{4,8}_t{TB} (src/kernels/qmms.metal) vs the
// prefill GEMM the engine picks at that T (qmm_b4_m32_n32 below 64 rows, m64_n64 from 64).
//   bench_qmms            35B + 27B projection shapes, T = 1..64
// env: TS="8 16 32" token counts; KS="1 2 4" K splits; NSG="4 8" simdgroups per threadgroup;
//      SHAPES=35|27; BITS=8 (8-bit weights); GU=1 (gate/up+SwiGLU kernel, two matrices).
// Reports us, weight GB/s, TFLOP/s and the max relative error of 64 sampled outputs vs a CPU
// reference that dequantizes like qmm (half(s*q + b)) and multiplies in double.
#include "metal/gpu.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"

#include <cmath>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;

struct QmmsP {
  uint32_t K, T, x_ld, nseg, ks;
  uint32_t N[4], y_ld[4], add[4], tg_begin[5];
};

static uint16_t f2h(float f) {
  __fp16 h = (__fp16)f;
  uint16_t u;
  memcpy(&u, &h, 2);
  return u;
}
static float h2f(uint16_t u) {
  __fp16 h;
  memcpy(&h, &u, 2);
  return float(h);
}
static std::vector<int> ints(const char* env, std::vector<int> def) {
  const char* v = getenv(env);
  if (!v || !*v) return def;
  std::vector<int> r;
  std::stringstream ss(v);
  int x;
  while (ss >> x) r.push_back(x);
  return r;
}

int main() {
  Gpu& g = Gpu::get();
  std::mt19937 rng(7);
  const int bits = getenv("BITS") ? atoi(getenv("BITS")) : 4;
  const bool gu = env_on("GU");
  struct Shape {
    uint32_t N, K;
    const char* what;
  };
  std::vector<Shape> shapes;
  const char* sh = getenv("SHAPES");
  if (!sh || !strcmp(sh, "35"))
    shapes.insert(shapes.end(), {{8192, 2048, "35B qkv"}, {4096, 2048, "35B z"}, {2048, 4096, "35B out"},
                                 {512, 2048, "35B shared g/u"}, {2048, 512, "35B shared down"}, {256, 2048, "35B router"}});
  if (!sh || !strcmp(sh, "27"))
    shapes.insert(shapes.end(), {{10240, 5120, "27B qkv"}, {6144, 5120, "27B z"}, {5120, 6144, "27B out"},
                                 {17408, 5120, "27B gate/up"}, {5120, 17408, "27B down"}});
  const auto Ts = ints("TS", {1, 2, 4, 8, 16, 24, 32, 48, 64});
  const auto KSs = ints("KS", {1, 2, 4});
  const auto NSGs = ints("NSG", {4, 8});
  const int nmat = gu ? 2 : 1;
  for (auto shp : shapes) {
    const uint32_t N = shp.N, K = shp.K, Tmax = 64;
    const size_t wwords = size_t(N) * K * bits / 32, groups = size_t(N) * K / 64;
    Buffer W[2], S[2], B[2];
    for (int m = 0; m < nmat; ++m) {
      W[m] = g.alloc(wwords * 4);
      S[m] = g.alloc(groups * 2);
      B[m] = g.alloc(groups * 2);
      for (size_t i = 0; i < wwords; ++i) W[m].as<uint32_t>()[i] = rng();
      std::uniform_real_distribution<float> d(-1, 1);
      for (size_t i = 0; i < groups; ++i) {
        S[m].as<uint16_t>()[i] = f32_to_bf16((bits == 4 ? 0.01f : 0.0008f) * (1.0f + d(rng)));
        B[m].as<uint16_t>()[i] = f32_to_bf16(0.05f * d(rng));
      }
    }
    Buffer X = g.alloc(size_t(Tmax) * K * 2), Y = g.alloc(size_t(Tmax) * N * 4), H = g.alloc(size_t(Tmax) * N * 2);
    {
      std::uniform_real_distribution<float> d(-1, 1);
      for (size_t i = 0; i < size_t(Tmax) * K; ++i) X.as<uint16_t>()[i] = f2h(d(rng));
    }
    auto wval = [&](int m, uint32_t n, uint32_t k) {
      uint32_t q;
      if (bits == 4) q = (W[m].as<uint32_t>()[(size_t(n) * K + k) / 8] >> (4 * (k % 8))) & 15;
      else q = (W[m].as<uint32_t>()[(size_t(n) * K + k) / 4] >> (8 * (k % 4))) & 255;
      float wv = bf16_to_f32(S[m].as<uint16_t>()[(size_t(n) * K + k) / 64]) * q +
                 bf16_to_f32(B[m].as<uint16_t>()[(size_t(n) * K + k) / 64]);
      return double(float(__fp16(wv)));
    };
    const double gbytes = nmat * (double(N) * K * bits / 8 + double(groups) * 4) / 1e9;
    printf("---- %s: N=%u K=%u %d-bit%s (%.1f MB) ----\n", shp.what, N, K, bits, gu ? " gate/up" : "",
           gbytes * 1e3);
    for (int T : Ts) {
      auto bench = [&](const std::string& label, auto&& encode) {
        auto run = [&](int reps) {
          auto e = g.begin();
          for (int r = 0; r < reps; ++r) encode(*e);
          e->commit_and_wait();
          return e->gpu_ms();
        };
        run(2);
        int reps = std::max(5, int(3e-3 / std::max(gbytes / 200, 1e-6)));
        double ms = run(reps) / reps;
        double maxrel = 0;
        for (int s = 0; s < 64; ++s) {
          uint32_t t = rng() % T, n = rng() % N;
          double ref[2] = {0, 0};
          for (int m = 0; m < nmat; ++m)
            for (uint32_t k = 0; k < K; ++k) ref[m] += wval(m, n, k) * h2f(X.as<uint16_t>()[size_t(t) * K + k]);
          double want, got;
          if (gu) {
            want = ref[0] / (1 + std::exp(-ref[0])) * ref[1];
            got = label.rfind("qmm_", 0) == 0 ? want : h2f(H.as<uint16_t>()[size_t(t) * N + n]);
            maxrel = std::max(maxrel, std::fabs(got - want) / (std::fabs(want) + 1.0));
          } else {
            want = ref[0];
            got = Y.as<float>()[size_t(t) * N + n];
            maxrel = std::max(maxrel, std::fabs(got - want) / (std::fabs(want) + 1.0));
          }
        }
        printf("T=%2d %-24s %8.1f us  %6.1f GB/s  %5.2f TFLOP/s  err=%.1e\n", T, label.c_str(), ms * 1000,
               gbytes / (ms * 1e-3), 2.0 * nmat * T * N * K / (ms * 1e-3) / 1e12, maxrel);
      };
      // baseline: the engine's qmm choice (gate/up: two qmm calls, no SwiGLU)
      if (bits == 4 || T < 64) {
        const bool big = T >= 64 && bits == 4;
        const int BM = big ? 64 : 32, BN = big ? 64 : 32;
        std::string qn = "qmm_b" + std::to_string(bits) + (big ? "_m64_n64" : "_m32_n32");
        Pipeline qp = g.pipeline(qn);
        kp::Qmm prm{uint32_t(T), N, K, K, N, 0, 64, 0, 0};
        bench(qn, [&](Encoder& e) {
          for (int m = 0; m < nmat; ++m) {
            e.use(qp);
            e.buf(0, X); e.buf(1, W[m]); e.buf(2, S[m]); e.buf(3, B[m]); e.buf(4, Y);
            e.val(5, prm);
            e.dispatch((N + BN - 1) / BN, (T + BM - 1) / BM, 1, 128);
          }
        });
      }
      const int TB = (T + 7) / 8;
      Pipeline sp = g.pipeline(std::string(gu ? "qmms_gu_b" : "qmms_b") + std::to_string(bits) + "_t" + std::to_string(TB));
      for (int nsg : NSGs)
        for (int ks : KSs) {
          if (nsg % ks) continue;
          QmmsP p{};
          p.K = K; p.T = uint32_t(T); p.x_ld = K; p.nseg = 1; p.ks = uint32_t(ks);
          p.N[0] = N; p.y_ld[0] = N; p.add[0] = 0;
          const uint32_t rbs = uint32_t(nsg / ks);
          p.tg_begin[0] = 0;
          p.tg_begin[1] = (N / 8 + rbs - 1) / rbs;
          std::string label = std::string(sp.name) + " nsg" + std::to_string(nsg) + " ks" + std::to_string(ks);
          bench(label, [&](Encoder& e) {
            e.use(sp);
            e.buf(0, W[0]); e.buf(1, S[0]); e.buf(2, B[0]); e.buf(3, gu ? H : Y);
            e.buf(4, W[gu ? 1 : 0]); e.buf(5, S[gu ? 1 : 0]); e.buf(6, B[gu ? 1 : 0]); e.buf(7, Y);
            for (int i = 8; i < 16; ++i) e.buf(i, (i % 4 == 3) ? Y : (i % 4 == 0 ? W[0] : S[0]));
            e.buf(16, X);
            e.val(17, p);
            e.dispatch(p.tg_begin[1], 1, 1, 32 * nsg);
          });
        }
    }
    for (int m = 0; m < nmat; ++m) {
      g.release(W[m]);
      g.release(S[m]);
      g.release(B[m]);
    }
    g.release(X);
    g.release(Y);
    g.release(H);
  }
}
