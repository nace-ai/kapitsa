// Flint's routed-expert prefill GEMMs, the tensor_ops tile against the simdgroup kernel, on synthetic weights of Flint's
// shapes (hidden 2048, 256 experts of 512, top-6, 4-bit g64): gate / up (token rows -> fp32 pair rows, K 2048) and down
// (pair rows -> fp32 pair rows, K 512), each as flint.cpp's enc_moe_prefill dispatches it --
//   simd   qmm_moe_b4_m16_n64 over 16-row expert tiles (NI_FLINT_MOE_QTM=0)
//   qtm    qmtm_g_m64_n64 over 64-row expert tiles (tensor/qmt_moe.metal, the M5 default)
// Both read the same dequantized halfs (half(s * q + b)) and accumulate in fp32, so they differ by the order of the K
// sum only. Printed per GEMM: each kernel's error against a double-precision CPU reference on sampled outputs (max
// |got - ref| / (|ref| + 1), as bench_moe), and qtm against simd over every output in units of one half step at the
// tensor's largest |value| (the max, and how many are over 1), plus the GPU time (median of trials, 10 runs each).
//   flint_moe_check [T=2048] [trials=5]
// Takes no GPU lock: run it under one (lockf -k /tmp/kapitsa-gpu.lock ...).
#include "metal/gpu.h"
#include "util/bf16.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace ni { extern const char* const kTensorKernelSource; }
using namespace ni;

namespace {

struct QmmParams {
  uint32_t M, N, K, x_ld, y_ld, add, gs, w_expert_stride, s_expert_stride;
};
struct Qmm2Params {
  uint32_t N, K, x_ld, y_ld, gs, w_expert_stride, s_expert_stride, shared_e;
};

std::mt19937_64 rng(7);
float rndf() { return std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng); }

struct QMat {
  Buffer w, s, b;
  uint32_t E, N, K;
};

QMat make_qmat(Gpu& g, uint32_t E, uint32_t N, uint32_t K) {
  const size_t n = size_t(E) * N * K;
  QMat m{g.alloc(n / 2), g.alloc(n / 64 * 2), g.alloc(n / 64 * 2), E, N, K};
  for (size_t i = 0; i < n / 8; ++i) m.w.as<uint32_t>()[i] = uint32_t(rng());
  for (size_t i = 0; i < n / 64; ++i) {
    m.s.as<uint16_t>()[i] = f32_to_bf16(0.004f * (1.0f + 0.5f * rndf()));
    m.b.as<uint16_t>()[i] = f32_to_bf16(-0.03f + 0.01f * rndf());
  }
  return m;
}

double wq(const QMat& m, uint32_t e, uint32_t n, uint32_t k) {  // half(s * q + b), the kernels' dequantization
  const size_t i = (size_t(e) * m.N + n) * m.K + k;
  const uint32_t q = (m.w.as<uint32_t>()[i / 8] >> (4 * (i % 8))) & 15;
  const float v = bf16_to_f32(m.s.as<uint16_t>()[i / 64]) * float(q) + bf16_to_f32(m.b.as<uint16_t>()[i / 64]);
  return double(float(__fp16(v)));
}

double h2d(const Buffer& b, size_t i) {
  __fp16 h;
  memcpy(&h, b.as<uint16_t>() + i, 2);
  return double(float(h));
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  const uint32_t T = argc > 1 ? uint32_t(atoi(argv[1])) : 2048;
  const int trials = argc > 2 ? atoi(argv[2]) : 5;
  constexpr uint32_t D = 2048, I = 512, E = 256, k = 6, gs = 64;
  const uint32_t P = T * k;
  Gpu& g = Gpu::get();
  g.compile_extra("tensor", kTensorKernelSource, true);

  const QMat eg = make_qmat(g, E, I, D), ed = make_qmat(g, E, D, I);
  Buffer xt = g.alloc(size_t(T) * D * 2), xp = g.alloc(size_t(P) * I * 2);  // token rows; pair rows (down's input)
  for (size_t i = 0; i < size_t(T) * D; ++i) {
    const __fp16 h = __fp16(2.0f * rndf());
    memcpy(xt.as<uint16_t>() + i, &h, 2);
  }
  for (size_t i = 0; i < size_t(P) * I; ++i) {
    const __fp16 h = __fp16(8.0f * rndf());
    memcpy(xp.as<uint16_t>() + i, &h, 2);
  }
  // routing: k distinct experts per token, Zipf-like popularity over a fixed permutation (as bench_moe's skew 1)
  std::vector<uint32_t> ex(P);
  {
    std::vector<uint32_t> perm(E);
    std::iota(perm.begin(), perm.end(), 0u);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<double> cdf(E);
    double acc = 0;
    for (uint32_t r = 0; r < E; ++r) cdf[r] = acc += 1.0 / (r + 8.0);
    for (uint32_t t = 0; t < T; ++t)
      for (uint32_t got = 0; got < k;) {
        const double u = std::uniform_real_distribution<double>(0, acc)(rng);
        const uint32_t e = perm[size_t(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin())];
        bool dup = false;
        for (uint32_t j = 0; j < got; ++j) dup |= ex[t * k + j] == e;
        if (!dup) ex[t * k + got++] = e;
      }
  }
  // sorted pairs (by expert, then pair id): yrows = pair id, xrows = token (gate / up) or pair id (down)
  std::vector<uint32_t> order(P);
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return ex[a] < ex[b]; });
  Buffer yrows = g.alloc(size_t(P) * 4), xrows_t = g.alloc(size_t(P) * 4);
  for (uint32_t r = 0; r < P; ++r) {
    yrows.as<uint32_t>()[r] = order[r];
    xrows_t.as<uint32_t>()[r] = order[r] / k;
  }
  struct Tiles {
    Buffer t, n;
    uint32_t max;
  };
  auto make_tiles = [&](uint32_t bm) {  // moe_bucket's tiles: [expert, begin, end) runs of at most bm sorted rows
    std::vector<uint32_t> v;
    for (uint32_t r = 0; r < P;) {
      uint32_t end = r;
      while (end < P && ex[order[end]] == ex[order[r]] && end - r < bm) ++end;
      v.insert(v.end(), {ex[order[r]], r, end});
      r = end;
    }
    Tiles t{g.alloc(v.size() * 4), g.alloc(16), (P + bm - 1) / bm + E};
    memcpy(t.t.as<uint32_t>(), v.data(), v.size() * 4);
    t.n.as<uint32_t>()[0] = uint32_t(v.size() / 3);
    return t;
  };
  const Tiles t16 = make_tiles(16), t64 = make_tiles(64);

  struct Gemm {
    const char* name;
    const QMat& w;
    const Buffer& x;
    const Buffer& xrows;
  };
  const Gemm gemms[2] = {{"gate/up (K 2048)", eg, xt, xrows_t}, {"down (K 512)", ed, xp, yrows}};
  printf("T=%u, %u pairs, %u / %u tiles of 16 / 64 rows\n", T, P, t16.n.as<uint32_t>()[0], t64.n.as<uint32_t>()[0]);
  for (const Gemm& G : gemms) {
    const QMat& W = G.w;
    const uint32_t w_es = W.N * (W.K / 8), s_es = W.N * (W.K / gs);
    Buffer y[2] = {g.alloc(size_t(P) * W.N * 4), g.alloc(size_t(P) * W.N * 4)};
    auto enc = [&](Encoder& e, int path) {
      if (path == 0) {
        const QmmParams p{P, W.N, W.K, W.K, W.N, 0, gs, w_es, s_es};
        e.use(g.pipeline("qmm_moe_b4_m16_n64"));
        e.val(5, p);
      } else {
        const Qmm2Params p{W.N, W.K, W.K, W.N, gs, w_es, s_es, 0xffffffffu};
        e.use(g.pipeline("qmtm_g_m64_n64"));
        e.val(5, p);
      }
      const Tiles& t = path == 0 ? t16 : t64;
      e.buf(0, G.x);
      e.buf(1, W.w);
      e.buf(2, W.s);
      e.buf(3, W.b);
      e.buf(4, y[path]);
      e.buf(6, t.t);
      e.buf(7, t.n);
      e.buf(8, G.xrows);
      e.buf(9, yrows);
      e.dispatch(W.N / 64, t.max, 1, 128);
    };
    std::vector<double> ms[2];
    for (int tr = 0; tr < trials; ++tr)
      for (int path = 0; path < 2; ++path) {
        auto e = g.begin();
        for (int r = 0; r < 10; ++r) enc(*e, path);
        e->commit_and_wait();
        ms[path].push_back(e->gpu_ms() / 10);
      }
    // against the reference: sampled (sorted row, column) outputs
    double err[2] = {0, 0};
    std::mt19937 srng(11);
    for (int i = 0; i < 4096; ++i) {
      const uint32_t r = srng() % P, n = srng() % W.N, e = ex[order[r]], xr = G.xrows.as<uint32_t>()[r];
      double ref = 0;
      for (uint32_t kk = 0; kk < W.K; ++kk) ref += h2d(G.x, size_t(xr) * W.K + kk) * wq(W, e, n, kk);
      for (int path = 0; path < 2; ++path) {
        const double got = y[path].as<float>()[size_t(order[r]) * W.N + n];
        err[path] = std::max(err[path], std::fabs(got - ref) / (std::fabs(ref) + 1));
      }
    }
    // qtm against simd, every output, in half steps at the tensor's largest |value|
    const size_t n_out = size_t(P) * W.N;
    double amax = 0, dmax = 0;
    for (size_t i = 0; i < n_out; ++i) amax = std::max(amax, double(std::fabs(y[0].as<float>()[i])));
    const double step = std::ldexp(1.0, int(std::floor(std::log2(amax))) - 10);  // one half ulp at amax
    size_t over = 0, same = 0;
    for (size_t i = 0; i < n_out; ++i) {
      const float a = y[0].as<float>()[i], b = y[1].as<float>()[i];
      const double d = std::fabs(double(a) - double(b)) / step;
      dmax = std::max(dmax, d);
      over += d > 1.0;
      same += a == b;
    }
    printf("%-17s simd %7.3f ms  qtm %7.3f ms (%.2fx) | vs reference: simd %.1e, qtm %.1e | qtm vs simd: max %.3f "
           "half steps at |y| max %.1f, %zu of %zu over 1, %zu bitwise equal\n",
           G.name, median(ms[0]), median(ms[1]), median(ms[0]) / median(ms[1]), err[0], err[1], dmax, amax, over,
           n_out, same);
  }
  return 0;
}
