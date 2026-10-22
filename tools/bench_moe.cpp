// MoE prefill GEMM benchmark + correctness on one synthetic Qwen3.6-35B-A3B layer (D=2048, I=512,
// 256 experts, top-8, 4-bit g64), routed experts only (the default NI_MOE_FUSED=2 keeps the shared
// expert on the dense path):
//   orig    moe_bucket (16-row tiles), qmm_moe_b4_m16_n64 gate and up -> fp32, swiglu_half, down
//   fused   moe_bucket2 (64-row tiles), qmm_moe_gu_b4_m64_n32 (gate/up/SwiGLU), qmm_moe_dn_b4_m64_n64
//           (the default NI_MOE_FUSED path; must equal orig bit for bit: same dequantized halfs and
//           MMA order)
//   qrm     moe_bucket2 (BM-row tiles), qrm_gu_* / qrm_dn_* (qmm_moe3.metal: weights dequantized
//           straight into MMA registers; agrees with orig to rounding)
//   qtm     moe_bucket2 (64-row tiles), qmtm_gu_m64_n64 / qmtm_dn_m64_n64 (tensor/qmt_moe.metal: the
//           qmt tile on the routed experts' grouped GEMMs; same 4-bit g64 weights, fp32 accumulate)
// Every path is checked against a double-precision CPU reference on sampled outputs (gate/up: on
// the dequantized halfs of orig's rule half(s*q + b); down: on the path's own half h rows), as
// max |got - ref| / (|ref| + 1) like bench_qmm. Timing: trials interleaved over the paths, each the
// mean of `reps` back-to-back runs in one command buffer; medians printed.
//   bench_moe [T=2048] [skew=1] [trials=5]   (skew=0: uniform random routing; 1: Zipf-like popularity;
//                                             2: balanced, every expert gets P/E rows)
//   BENCH_QRM="gu:dn,gu:dn" picks the qrm kernel pairs (tile rows/width parsed from _m<BM>_n<BN>).
#include "metal/gpu.h"
#include "util/bf16.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace ni { extern const char* const kTensorKernelSource; }
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

// dequantized weight (orig's rule: float s*q + b rounded to half) of matrix e, row n, column k
static double wq(const QMat& m, uint32_t e, uint32_t n, uint32_t k) {
  const size_t row = size_t(e) * m.N + n, i = row * m.K + k;
  const uint32_t q = (m.w.as<uint32_t>()[i / 8] >> (4 * (i % 8))) & 15;
  const float v = bf16_to_f32(m.s.as<uint16_t>()[i / 64]) * float(q) + bf16_to_f32(m.b.as<uint16_t>()[i / 64]);
  return double(float(__fp16(v)));
}

static int name_int(const std::string& n, const char* key) {
  size_t p = n.rfind(key);
  return p == std::string::npos ? 0 : atoi(n.c_str() + p + strlen(key));
}

int main(int argc, char** argv) {
  const uint32_t T = argc > 1 ? atoi(argv[1]) : 2048;
  const int skew = argc > 2 ? atoi(argv[2]) : 1;  // 0 uniform random, 1 Zipf-like, 2 balanced (P/E rows each)
  const int trials = argc > 3 ? atoi(argv[3]) : 5;
  const uint32_t D = 2048, I = 512, E = 256, k = 8, gs = 64, P = T * k;
  Gpu& g = Gpu::get();
  g.compile_extra("tensor", kTensorKernelSource, true);  // qmtm_* (the routed experts on tensor_ops)

  QMat eg = make_qmat(g, E, I, D), eu = make_qmat(g, E, I, D), ed = make_qmat(g, E, D, I);
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
      if (skew == 2)
        for (; got < k; ++got) {
          const uint32_t e = (t * k + got) % E;
          idx.as<uint32_t>()[t * k + got] = e;
          cnt[e]++;
        }
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
           skew == 2 ? "balanced" : skew ? "skewed" : "uniform");
  }
  const uint32_t wst_gu = I * (D / 8), sst_gu = I * (D / gs), wst_dn = D * (I / 8), sst_dn = D * (I / gs);

  // ---- paths
  struct Path {
    std::string name, gu, dn;
    int kind;  // 0 orig, 1 fused (qmm_moe), 2 qrm, 3 short-prefill 8-row tiles (qmms_moe*)
    uint32_t bm, bn_gu, bn_dn;
    Buffer yrows, xrows, tiles, ntiles, hg, hu, h, yd;
    std::vector<double> t_gu, t_dn, t_all;
    double err_h = 0, err_y = 0;
    size_t bad_h = 0, bad_y = 0;  // bitwise mismatches vs orig (fused only)
  };
  std::vector<Path> paths;
  paths.push_back({"orig", "qmm_moe_b4_m16_n64", "qmm_moe_b4_m16_n64", 0, 16, 64, 64});
  paths.push_back({"fused64", "qmm_moe_gu_b4_m64_n32", "qmm_moe_dn_b4_m64_n64", 1, 64, 32, 64});
  {
    const char* env = getenv("BENCH_QRM");
    std::string spec = env && *env ? env : "qrm_gu_k32_m32_n64:qrm_dn_k32_m32_n128";
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
      size_t c = item.find(':');
      if (c == std::string::npos) fatal("BENCH_QRM item %s: expected gu:dn", item.c_str());
      std::string gu = item.substr(0, c), dn = item.substr(c + 1);
      uint32_t bm = uint32_t(name_int(gu, "_m"));
      if (bm != uint32_t(name_int(dn, "_m"))) fatal("BENCH_QRM %s: gate/up and down tile rows differ", item.c_str());
      const bool qtm = gu.rfind("qmtm_gu_") == 0;
      const size_t strip = gu.rfind("qrm_gu_") == 0 ? 7 : 0;
      const size_t strip_dn = dn.rfind("qrm_dn_") == 0 ? 7 : strip;
      paths.push_back({gu.substr(strip) + "+" + dn.substr(strip_dn), gu, dn, 2, bm,
                       uint32_t(name_int(gu, "_n")) / (qtm ? 2u : 1u), uint32_t(name_int(dn, "_n"))});
    }
  }
  // the routed experts on tensor_ops (tensor/qmt_moe.metal), the NI_MOE_QTM path; BENCH_QTM=0 skips it.
  // Its gate/up threadgroup covers BN / 2 output columns (NP = 2: gate and up of the same columns), so the gu
  // grid is N / (BN / 2) -- the `bn_gu` here is the grid divisor, 32 for qmtm_gu_m64_n64.
  if (!getenv("BENCH_QTM") || atoi(getenv("BENCH_QTM")))
    paths.push_back({"qmtm_m64_n64", "qmtm_gu_m64_n64", "qmtm_dn_m64_n64", 2, 64, 32, 64});
  // BENCH_SHORT=1: the short-prefill expert paths (8-row tiles): qmms_moe (8-row MMA) and the hybrid
  // qmms_moe2 (tiles of <= 2 rows by FMAs)
  if (getenv("BENCH_SHORT") && atoi(getenv("BENCH_SHORT"))) {
    paths.push_back({"qmms_moe (8-row MMA)", "qmms_moe_gu_b4", "qmms_moe_dn_b4", 3, 8, 32, 32});
    paths.push_back({"qmms_moe2 (FMA <= 2 rows)", "qmms_moe2_gu_b4", "qmms_moe2_dn_b4", 3, 8, 32, 32});
  }
  for (auto& p : paths) {
    const size_t R = P;
    p.yrows = g.alloc(R * 4);
    p.xrows = g.alloc(R * 4);
    p.tiles = g.alloc((R + E + 1) * 3 * 4);
    p.ntiles = g.alloc(16);
    if (p.kind == 0) {
      p.hg = g.alloc(R * I * 4);
      p.hu = g.alloc(R * I * 4);
    }
    p.h = g.alloc(R * I * 2);
    p.yd = g.alloc(R * size_t(D) * 4);
  }

  auto bucket = [&](Encoder& e, Path& p) {
    if (p.kind == 0) {
      BucketParams bp{P, k, E, p.bm};
      e.use(g.pipeline("moe_bucket"));
      e.buf(0, idx); e.buf(1, p.yrows); e.buf(2, p.xrows); e.buf(3, p.tiles); e.buf(4, p.ntiles);
      e.val(5, bp);
    } else {
      Bucket2Params bp{P, k, E, p.bm, 0};
      e.use(g.pipeline("moe_bucket2"));
      e.buf(0, idx); e.buf(1, p.yrows); e.buf(2, p.xrows); e.buf(3, p.tiles); e.buf(4, p.ntiles);
      e.val(5, bp);
    }
    e.dispatch(1, 1, 1, 1024);
  };
  auto max_tiles = [&](const Path& p) { return (P + p.bm - 1) / p.bm + E; };
  auto old_qmm_moe = [&](Encoder& e, Path& p, const QMat& m, const Buffer& x, const Buffer& y, const Buffer& xr,
                         const Buffer& yr, uint32_t wst, uint32_t sst) {
    QmmParams q{P, m.N, m.K, m.K, m.N, 0, gs, wst, sst};
    e.use(g.pipeline(p.gu));
    e.buf(0, x); e.buf(1, m.w); e.buf(2, m.s); e.buf(3, m.b); e.buf(4, y);
    e.val(5, q);
    e.buf(6, p.tiles); e.buf(7, p.ntiles); e.buf(8, xr); e.buf(9, yr);
    e.dispatch(m.N / 64, max_tiles(p), 1, 128);
  };
  auto gate_up = [&](Encoder& e, Path& p) {
    if (p.kind == 0) {
      old_qmm_moe(e, p, eg, xh, p.hg, p.xrows, p.yrows, wst_gu, sst_gu);
      old_qmm_moe(e, p, eu, xh, p.hu, p.xrows, p.yrows, wst_gu, sst_gu);
      uint32_t n = P * I;
      e.use(g.pipeline("swiglu_half"));
      e.buf(0, p.hg); e.buf(1, p.hu); e.buf(2, p.h); e.val(3, n);
      e.dispatch((n + 255) / 256, 1, 1, 256);
      return;
    }
    Qmm2Params q{I, D, D, I, gs, wst_gu, sst_gu, E};
    {
      e.use(g.pipeline(p.gu));
      e.buf(0, xh);
      e.buf(1, eg.w); e.buf(2, eg.s); e.buf(3, eg.b);
      e.buf(4, eu.w); e.buf(5, eu.s); e.buf(6, eu.b);
      e.buf(7, p.h);
      e.val(8, q);
      e.buf(9, p.tiles); e.buf(10, p.ntiles); e.buf(11, p.xrows);
      e.buf(12, eg.w); e.buf(13, eg.s); e.buf(14, eg.b);
      e.buf(15, eu.w); e.buf(16, eu.s); e.buf(17, eu.b);
      e.dispatch(I / p.bn_gu, max_tiles(p), 1, 128);
    }
  };
  auto down = [&](Encoder& e, Path& p) {
    if (p.kind == 0) {
      QmmParams q{P, D, I, I, D, 0, gs, wst_dn, sst_dn};
      e.use(g.pipeline(p.dn));
      e.buf(0, p.h); e.buf(1, ed.w); e.buf(2, ed.s); e.buf(3, ed.b); e.buf(4, p.yd);
      e.val(5, q);
      e.buf(6, p.tiles); e.buf(7, p.ntiles); e.buf(8, p.yrows); e.buf(9, p.yrows);
      e.dispatch(D / 64, max_tiles(p), 1, 128);
      return;
    }
    Qmm2Params q{D, I, I, D, gs, wst_dn, sst_dn, E};
    {
      e.use(g.pipeline(p.dn));
      e.buf(0, p.h);
      e.buf(1, ed.w); e.buf(2, ed.s); e.buf(3, ed.b);
      e.buf(4, p.yd);
      e.val(5, q);
      e.buf(6, p.tiles); e.buf(7, p.ntiles); e.buf(8, p.yrows);
      e.buf(9, ed.w); e.buf(10, ed.s); e.buf(11, ed.b);
      e.dispatch(D / p.bn_dn, max_tiles(p), 1, 128);
    }
  };

  // GPU ms per run of `f`, averaged over reps in one command buffer
  auto gpu_time = [&](auto&& f, int reps) {
    auto e = g.begin();
    for (int r = 0; r < reps; ++r) f(*e);
    e->commit_and_wait();
    return e->gpu_ms() / reps;
  };
  const double f_gu = 2.0 * 2 * P * I * D, f_dn = 2.0 * P * D * I;
  auto tf = [](double flop, double ms) { return flop / (ms * 1e-3) / 1e12; };
  const int reps = 10;

  // ---- moe_bucket3 (parallel scans) against moe_bucket2: identical tiles, same rows per tile
  for (uint32_t bm : {8u, 32u, 64u}) {
    Path a = paths.back(), b = paths.back();
    a.yrows = g.alloc(size_t(P) * 4); a.xrows = g.alloc(size_t(P) * 4);
    a.tiles = g.alloc((size_t(P) + E + 1) * 3 * 4); a.ntiles = g.alloc(16);
    b.yrows = g.alloc(size_t(P) * 4); b.xrows = g.alloc(size_t(P) * 4);
    b.tiles = g.alloc((size_t(P) + E + 1) * 3 * 4); b.ntiles = g.alloc(16);
    auto run = [&](const char* kn, Path& q, int reps) {
      Bucket2Params bp{P, k, E, bm, 0};
      auto e = g.begin();
      for (int r = 0; r < reps; ++r) {
        e->use(g.pipeline(kn));
        e->buf(0, idx); e->buf(1, q.yrows); e->buf(2, q.xrows); e->buf(3, q.tiles); e->buf(4, q.ntiles);
        e->val(5, bp);
        e->dispatch(1, 1, 1, 1024);
      }
      e->commit_and_wait();
      return e->gpu_ms() / reps;
    };
    run("moe_bucket2", a, 1);
    run("moe_bucket3", b, 1);
    // tiles must be identical; the rows of an expert (the atomic scatter orders them arbitrarily in
    // both kernels) must be the same set over the expert's whole row range
    const uint32_t na = a.ntiles.as<uint32_t>()[0], nb = b.ntiles.as<uint32_t>()[0];
    bool same = na == nb && !memcmp(a.tiles.ptr(), b.tiles.ptr(), size_t(na) * 12);
    for (uint32_t t = 0; same && t < na;) {
      const uint32_t ex = a.tiles.as<uint32_t>()[t * 3 + 0], r0 = a.tiles.as<uint32_t>()[t * 3 + 1];
      uint32_t r1 = a.tiles.as<uint32_t>()[t * 3 + 2];
      for (++t; t < na && a.tiles.as<uint32_t>()[t * 3 + 0] == ex; ++t) r1 = a.tiles.as<uint32_t>()[t * 3 + 2];
      std::vector<uint64_t> va, vb;
      for (uint32_t r = r0; r < r1; ++r) {
        va.push_back(uint64_t(a.yrows.as<uint32_t>()[r]) << 32 | a.xrows.as<uint32_t>()[r]);
        vb.push_back(uint64_t(b.yrows.as<uint32_t>()[r]) << 32 | b.xrows.as<uint32_t>()[r]);
      }
      std::sort(va.begin(), va.end());
      std::sort(vb.begin(), vb.end());
      same = va == vb;
    }
    double t2 = 1e9, t3 = 1e9;
    for (int i = 0; i < 3; ++i) {
      t2 = std::min(t2, run("moe_bucket2", a, 20));
      t3 = std::min(t3, run("moe_bucket3", b, 20));
    }
    printf("bucket BM=%2u: %u tiles; moe_bucket2 %.1f us, moe_bucket3 %.1f us; %s\n", bm, na, t2 * 1e3, t3 * 1e3,
           same ? "identical tiles and rows" : "MISMATCH");
  }

  // ---- one run of every path, then correctness
  for (auto& p : paths) {
    memset(p.h.ptr(), 0, p.h.size);
    memset(p.yd.ptr(), 0, p.yd.size);
    auto e = g.begin();
    bucket(*e, p); gate_up(*e, p); down(*e, p);
    e->commit_and_wait();
  }
  {
    // h row of pair `pair` for path p (orig: by pair; others: sorted rows -> find through yrows)
    std::vector<std::vector<uint32_t>> sorted_of(paths.size(), std::vector<uint32_t>(P));
    for (size_t pi = 0; pi < paths.size(); ++pi)
      for (uint32_t m = 0; m < P; ++m) sorted_of[pi][paths[pi].yrows.as<uint32_t>()[m]] = m;
    auto hrow = [&](size_t pi, uint32_t pair) -> const uint16_t* {
      const Path& p = paths[pi];
      return p.h.as<uint16_t>() + size_t(p.kind == 0 ? pair : sorted_of[pi][pair]) * I;
    };
    std::mt19937 r2(11);
    const int samples = 192;
    for (int t = 0; t < samples; ++t) {
      const uint32_t pair = t == 0 ? P - 1 : r2() % P, tok = pair / k, e = idx.as<uint32_t>()[pair];
      const uint32_t n = t == 0 ? I - 1 : r2() % I, nd = t == 0 ? D - 1 : r2() % D;
      double gv = 0, uv = 0;
      for (uint32_t kk = 0; kk < D; ++kk) {
        const double x = h2f(xh.as<uint16_t>()[size_t(tok) * D + kk]);
        gv += wq(eg, e, n, kk) * x;
        uv += wq(eu, e, n, kk) * x;
      }
      const double href = gv / (1.0 + std::exp(-gv)) * uv;
      for (size_t pi = 0; pi < paths.size(); ++pi) {
        Path& p = paths[pi];
        const uint16_t* hr = hrow(pi, pair);
        p.err_h = std::max(p.err_h, std::fabs(h2f(hr[n]) - href) / (std::fabs(href) + 1.0));
        double yref = 0;
        for (uint32_t j = 0; j < I; ++j) yref += wq(ed, e, nd, j) * h2f(hr[j]);
        const double got = p.yd.as<float>()[size_t(pair) * D + nd];
        p.err_y = std::max(p.err_y, std::fabs(got - yref) / (std::fabs(yref) + 1.0));
      }
    }
    // bitwise: fused vs orig (same dequantized halfs, same MMA order)
    const Path& o = paths[0];
    for (size_t pi = 1; pi < paths.size(); ++pi) {
      Path& p = paths[pi];
      for (uint32_t pair = 0; pair < P; ++pair) {
        const uint16_t *a = hrow(0, pair), *c = hrow(pi, pair);
        for (uint32_t n = 0; n < I; ++n) p.bad_h += a[n] != c[n];
      }
      for (size_t i = 0; i < size_t(P) * D; ++i) p.bad_y += o.yd.as<float>()[i] != p.yd.as<float>()[i];
    }
  }
  // ---- timing, trials interleaved over the paths
  for (auto& p : paths) gpu_time([&](Encoder& e) { bucket(e, p); gate_up(e, p); down(e, p); }, 1);
  for (int t = 0; t < trials; ++t)
    for (auto& p : paths) {
      p.t_gu.push_back(gpu_time([&](Encoder& e) { gate_up(e, p); }, reps));
      p.t_dn.push_back(gpu_time([&](Encoder& e) { down(e, p); }, reps));
      p.t_all.push_back(gpu_time([&](Encoder& e) { bucket(e, p); gate_up(e, p); down(e, p); }, reps));
    }
  auto med = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  const double o_all = med(paths[0].t_all);
  for (auto& p : paths) {
    const double gu = med(p.t_gu), dn = med(p.t_dn), all = med(p.t_all);
    printf("%-34s bm %2u | gate+up %7.1f us (%4.2f TFLOP/s) | down %7.1f us (%4.2f) | total %7.1f us (%.3fx) | "
           "err h %.1e y %.1e%s",
           p.name.c_str(), p.bm, gu * 1e3, tf(f_gu, gu), dn * 1e3, tf(f_dn, dn), all * 1e3, o_all / all, p.err_h,
           p.err_y, p.err_h > 4e-3 || p.err_y > 1e-3 ? " FAIL" : "");
    if (p.kind != 0) printf(" | vs orig: h %zu / y %zu values differ", p.bad_h, p.bad_y);
    printf("\n");
  }
  return 0;
}
