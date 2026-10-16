// MoE prefill GEMM benchmark + correctness on one synthetic Qwen3.6-35B-A3B layer (D=2048, I=512,
// 256 experts, top-8, shared expert I=512, 4-bit g64): the original path (moe_bucket,
// qmm_moe_b4_m16_n64 gate/up -> fp32, swiglu_half, down; shared expert as 3 dense GEMMs + swiglu)
// against the NI_MOE_FUSED path (moe_bucket2, qmm_moe_gu, qmm_moe_dn; shared expert folded in) for
// 16/32/64-row tiles. Outputs must match the original bit for bit (same dequantized halfs, same
// MMA order); the tool prints mismatch counts and max differences.
//   bench_moe [T=2048] [skew=1]   (skew=0: uniform routing; 1: Zipf-like expert popularity)
#include "metal/gpu.h"
#include "util/bf16.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ni;

struct QmmParams {
  uint32_t M, N, K, x_ld, y_ld, add, gs, w_expert_stride, s_expert_stride;
};
struct Qmm2Params {
  uint32_t N, K, x_ld, y_ld, gs, w_expert_stride, s_expert_stride, shared_e;
};
struct BucketParams {
  uint32_t P, k, E, BM;
};
struct Bucket2Params {
  uint32_t P, k, E, BM, S;
};

// 4-bit g64 affine matrix stack [n_mat, N, K]
struct QMat {
  Buffer w, s, b;
  uint32_t N, K;
};

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static inline uint32_t rnd() {  // xorshift64*
  g_rng ^= g_rng >> 12;
  g_rng ^= g_rng << 25;
  g_rng ^= g_rng >> 27;
  return uint32_t((g_rng * 2685821657736338717ull) >> 32);
}
static inline float rndf() { return (rnd() >> 8) * (2.0f / 16777216.0f) - 1.0f; }  // [-1, 1)

static QMat make_qmat(Gpu& g, uint32_t n_mat, uint32_t N, uint32_t K) {
  QMat m{g.alloc(size_t(n_mat) * N * K / 2), g.alloc(size_t(n_mat) * N * K / 64 * 2),
         g.alloc(size_t(n_mat) * N * K / 64 * 2), N, K};
  for (size_t i = 0; i < size_t(n_mat) * N * K / 8; ++i) m.w.as<uint32_t>()[i] = rnd();
  for (size_t i = 0; i < size_t(n_mat) * N * K / 64; ++i) {
    m.s.as<uint16_t>()[i] = f32_to_bf16(0.004f * (1.0f + 0.5f * rndf()));
    m.b.as<uint16_t>()[i] = f32_to_bf16(-0.03f + 0.01f * rndf());
  }
  return m;
}

static float h2f(uint16_t u) {
  __fp16 h;
  memcpy(&h, &u, 2);
  return float(h);
}

int main(int argc, char** argv) {
  const uint32_t T = argc > 1 ? atoi(argv[1]) : 2048;
  const bool skew = argc > 2 ? atoi(argv[2]) != 0 : true;
  const uint32_t D = 2048, I = 512, E = 256, k = 8, gs = 64, P = T * k;
  Gpu& g = Gpu::get();

  QMat eg = make_qmat(g, E, I, D), eu = make_qmat(g, E, I, D), ed = make_qmat(g, E, D, I);
  QMat sg = make_qmat(g, 1, I, D), su = make_qmat(g, 1, I, D), sd = make_qmat(g, 1, D, I);
  Buffer xh = g.alloc(size_t(T) * D * 2);
  for (size_t i = 0; i < size_t(T) * D; ++i) {
    __fp16 h = rndf();
    memcpy(xh.as<uint16_t>() + i, &h, 2);
  }
  // routing: k distinct experts per token; with skew, popularity ~ 1/(rank+8) over a fixed permutation
  Buffer idx = g.alloc(size_t(P) * 4);
  {
    std::vector<double> cdf(E);
    std::vector<uint32_t> perm(E);
    std::iota(perm.begin(), perm.end(), 0u);
    std::mt19937 prng(3);
    std::shuffle(perm.begin(), perm.end(), prng);
    double acc = 0;
    for (uint32_t r = 0; r < E; ++r) cdf[r] = acc += skew ? 1.0 / (r + 8.0) : 1.0;
    std::vector<uint32_t> cnt(E, 0);
    for (uint32_t t = 0; t < T; ++t) {
      uint32_t got = 0;
      while (got < k) {
        double u = (rnd() / 4294967296.0) * acc;
        uint32_t e = perm[std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin()];
        bool dup = false;
        for (uint32_t j = 0; j < got; ++j) dup |= idx.as<uint32_t>()[t * k + j] == e;
        if (dup) continue;
        idx.as<uint32_t>()[t * k + got++] = e;
        cnt[e]++;
      }
    }
    std::sort(cnt.begin(), cnt.end());
    printf("T=%u P=%u, rows per expert: min %u median %u max %u (%s routing)\n", T, P, cnt[0], cnt[E / 2], cnt[E - 1],
           skew ? "skewed" : "uniform");
  }

  // ---- original path buffers
  Buffer spair = g.alloc(size_t(P) * 4), stok = g.alloc(size_t(P) * 4);
  Buffer tiles = g.alloc((size_t(P) + E + 1) * 3 * 4), ntiles = g.alloc(16);
  Buffer hg = g.alloc(size_t(P) * I * 4), hu = g.alloc(size_t(P) * I * 4), hhalf = g.alloc(size_t(P) * I * 2);
  Buffer yd = g.alloc(size_t(P) * D * 4);
  Buffer sgb = g.alloc(size_t(T) * I * 4), sub = g.alloc(size_t(T) * I * 4), hh = g.alloc(size_t(T) * I * 2);
  Buffer ys = g.alloc(size_t(T) * D * 4);
  // ---- fused path buffers (P pair rows + T shared rows)
  const size_t R2 = size_t(P) + T;
  Buffer yrows2 = g.alloc(R2 * 4), xrows2 = g.alloc(R2 * 4);
  Buffer tiles2 = g.alloc((R2 + E + 1) * 3 * 4), ntiles2 = g.alloc(16);
  Buffer h2 = g.alloc(R2 * I * 2), yd2 = g.alloc(R2 * D * 4);

  const uint32_t wst_gu = I * (D / 8), sst_gu = I * (D / gs), wst_dn = D * (I / 8), sst_dn = D * (I / gs);
  // stage encoders (each appends its dispatches to e)
  auto old_bucket = [&](Encoder& e) {
    BucketParams bp{P, k, E, 16};
    e.use(g.pipeline("moe_bucket"));
    e.buf(0, idx); e.buf(1, spair); e.buf(2, stok); e.buf(3, tiles); e.buf(4, ntiles);
    e.val(5, bp);
    e.dispatch(1, 1, 1, 1024);
  };
  auto old_qmm_moe = [&](Encoder& e, const QMat& m, const Buffer& x, const Buffer& y, const Buffer& xr, const Buffer& yr,
                         uint32_t wst, uint32_t sst) {
    QmmParams p{P, m.N, m.K, m.K, m.N, 0, gs, wst, sst};
    e.use(g.pipeline("qmm_moe_b4_m16_n64"));
    e.buf(0, x); e.buf(1, m.w); e.buf(2, m.s); e.buf(3, m.b); e.buf(4, y);
    e.val(5, p);
    e.buf(6, tiles); e.buf(7, ntiles); e.buf(8, xr); e.buf(9, yr);
    e.dispatch(m.N / 64, (P + 15) / 16 + E, 1, 128);
  };
  auto swiglu = [&](Encoder& e, const Buffer& a, const Buffer& b, const Buffer& h, uint32_t n) {
    e.use(g.pipeline("swiglu_half"));
    e.buf(0, a); e.buf(1, b); e.buf(2, h); e.val(3, n);
    e.dispatch((n + 255) / 256, 1, 1, 256);
  };
  auto dense = [&](Encoder& e, const QMat& m, const Buffer& x, const Buffer& y) {
    QmmParams p{T, m.N, m.K, m.K, m.N, 0, gs, 0, 0};
    e.use(g.pipeline("qmm_b4_m64_n64"));
    e.buf(0, x); e.buf(1, m.w); e.buf(2, m.s); e.buf(3, m.b); e.buf(4, y);
    e.val(5, p);
    e.dispatch(m.N / 64, (T + 63) / 64, 1, 128);
  };
  auto old_gate_up = [&](Encoder& e) {
    old_qmm_moe(e, eg, xh, hg, stok, spair, wst_gu, sst_gu);
    old_qmm_moe(e, eu, xh, hu, stok, spair, wst_gu, sst_gu);
  };
  auto old_swiglu = [&](Encoder& e) { swiglu(e, hg, hu, hhalf, P * I); };
  auto old_down = [&](Encoder& e) { old_qmm_moe(e, ed, hhalf, yd, spair, spair, wst_dn, sst_dn); };
  auto old_shared = [&](Encoder& e) {
    dense(e, sg, xh, sgb);
    dense(e, su, xh, sub);
    swiglu(e, sgb, sub, hh, T * I);
    dense(e, sd, hh, ys);
  };

  int bm = 64;
  bool fold = true;
  auto max_tiles2 = [&]() { return (P + bm - 1) / bm + E + (fold ? (T + bm - 1) / bm : 0); };
  auto new_bucket = [&](Encoder& e) {
    Bucket2Params bp{P, k, E, uint32_t(bm), fold ? T : 0};
    e.use(g.pipeline("moe_bucket2"));
    e.buf(0, idx); e.buf(1, yrows2); e.buf(2, xrows2); e.buf(3, tiles2); e.buf(4, ntiles2);
    e.val(5, bp);
    e.dispatch(1, 1, 1, 1024);
  };
  auto new_gate_up = [&](Encoder& e) {
    Qmm2Params p{I, D, D, I, gs, wst_gu, sst_gu, E};
    e.use(g.pipeline("qmm_moe_gu_b4_m" + std::to_string(bm) + "_n32"));
    e.buf(0, xh);
    e.buf(1, eg.w); e.buf(2, eg.s); e.buf(3, eg.b);
    e.buf(4, eu.w); e.buf(5, eu.s); e.buf(6, eu.b);
    e.buf(7, h2);
    e.val(8, p);
    e.buf(9, tiles2); e.buf(10, ntiles2); e.buf(11, xrows2);
    e.buf(12, sg.w); e.buf(13, sg.s); e.buf(14, sg.b);
    e.buf(15, su.w); e.buf(16, su.s); e.buf(17, su.b);
    e.dispatch(I / 32, max_tiles2(), 1, 128);
  };
  auto new_down = [&](Encoder& e) {
    Qmm2Params p{D, I, I, D, gs, wst_dn, sst_dn, E};
    e.use(g.pipeline("qmm_moe_dn_b4_m" + std::to_string(bm) + "_n64"));
    e.buf(0, h2);
    e.buf(1, ed.w); e.buf(2, ed.s); e.buf(3, ed.b);
    e.buf(4, yd2);
    e.val(5, p);
    e.buf(6, tiles2); e.buf(7, ntiles2); e.buf(8, yrows2);
    e.buf(9, sd.w); e.buf(10, sd.s); e.buf(11, sd.b);
    e.dispatch(D / 64, max_tiles2(), 1, 128);
  };

  // GPU ms per run of `f`, averaged over reps in one command buffer (after one warm-up run)
  auto gpu_time = [&](auto&& f, int reps) {
    { auto e = g.begin(); f(*e); e->commit_and_wait(); }
    auto e = g.begin();
    for (int r = 0; r < reps; ++r) f(*e);
    e->commit_and_wait();
    return e->gpu_ms() / reps;
  };
  const double f_gu = 2.0 * 2 * P * I * D, f_dn = 2.0 * P * D * I, f_sh = 3 * 2.0 * T * I * D;
  auto tf = [](double flop, double ms) { return flop / (ms * 1e-3) / 1e12; };
  const int reps = 10;

  // ---- original path: run once for reference outputs, then time each stage
  {
    auto e = g.begin();
    old_bucket(*e); old_gate_up(*e); old_swiglu(*e); old_down(*e); old_shared(*e);
    e->commit_and_wait();
  }
  double o_b = gpu_time(old_bucket, reps), o_gu = gpu_time(old_gate_up, reps), o_sw = gpu_time(old_swiglu, reps);
  double o_dn = gpu_time(old_down, reps), o_sh = gpu_time(old_shared, reps);
  double o_all = gpu_time([&](Encoder& e) { old_bucket(e); old_gate_up(e); old_swiglu(e); old_down(e); old_shared(e); }, reps);
  printf("original : bucket %6.1f us | gate+up %7.1f us (%4.2f TFLOP/s) | swiglu %6.1f us | down %7.1f us (%4.2f) | "
         "shared %6.1f us (%4.2f) | total %7.1f us\n",
         o_b * 1e3, o_gu * 1e3, tf(f_gu, o_gu), o_sw * 1e3, o_dn * 1e3, tf(f_dn, o_dn), o_sh * 1e3, tf(f_sh, o_sh),
         o_all * 1e3);

  for (int b : {16, 32, 64}) {
    bm = b;
    for (bool fl : {true, false}) {
      fold = fl;
      memset(h2.ptr(), 0, h2.size);
      memset(yd2.ptr(), 0, yd2.size);
      {
        auto e = g.begin();
        new_bucket(*e); new_gate_up(*e); new_down(*e);
        e->commit_and_wait();
      }
      // correctness against the original path
      size_t h_bad = 0, y_bad = 0, s_bad = 0;
      double h_max = 0, y_max = 0, s_max = 0;
      const uint32_t* yr = yrows2.as<uint32_t>();
      for (size_t m = 0; m < P; ++m) {
        const uint16_t* a = hhalf.as<uint16_t>() + size_t(yr[m]) * I;
        const uint16_t* c = h2.as<uint16_t>() + m * I;
        for (uint32_t n = 0; n < I; ++n)
          if (a[n] != c[n]) { ++h_bad; h_max = std::max(h_max, double(std::fabs(h2f(a[n]) - h2f(c[n])))); }
      }
      for (size_t i = 0; i < size_t(P) * D; ++i) {
        float a = yd.as<float>()[i], c = yd2.as<float>()[i];
        if (a != c) { ++y_bad; y_max = std::max(y_max, double(std::fabs(a - c))); }
      }
      if (fold)
        for (size_t i = 0; i < size_t(T) * D; ++i) {
          float a = ys.as<float>()[i], c = yd2.as<float>()[size_t(P) * D + i];
          if (a != c) { ++s_bad; s_max = std::max(s_max, double(std::fabs(a - c))); }
        }
      double n_b = gpu_time(new_bucket, reps), n_gu = gpu_time(new_gate_up, reps), n_dn = gpu_time(new_down, reps);
      double n_all = fold ? gpu_time([&](Encoder& e) { new_bucket(e); new_gate_up(e); new_down(e); }, reps)
                          : gpu_time([&](Encoder& e) { new_bucket(e); new_gate_up(e); new_down(e); old_shared(e); }, reps);
      // with the shared expert folded in, its FLOPs are inside gate+up / down
      double fgu = f_gu + (fold ? 2 * 2.0 * T * I * D : 0), fdn = f_dn + (fold ? 2.0 * T * D * I : 0);
      printf("fused m%-2d%s: bucket %6.1f us | gate+up %7.1f us (%4.2f TFLOP/s) | down %7.1f us (%4.2f) | total %7.1f us "
             "(%.2fx) | mismatches h %zu (max %.2g) y %zu (max %.2g) shared %zu (max %.2g)\n",
             bm, fold ? " +shared" : " dense-sh", n_b * 1e3, n_gu * 1e3, tf(fgu, n_gu), n_dn * 1e3, tf(fdn, n_dn),
             n_all * 1e3, o_all / n_all, h_bad, h_max, y_bad, y_max, s_bad, s_max);
    }
  }
  return 0;
}
