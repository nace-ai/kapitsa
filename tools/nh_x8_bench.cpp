// Nemotron 3.5 Super-VL's weight kernels on synthetic weights of the real shapes, against fp64 products of the
// dequantized weights: FP8 E4M3 (src/kernels/nemotron/nh_x_fp8.metal; nh_mv / nh_gemm FP8 fallbacks) on the Mamba
// projections, attention and LM head, NVFP4 routed experts with per-expert tensor scales (nh_x4_moe_up / _dn per
// pair, nh_gemm_fp4_moe_* on the bucketed pairs) and the BF16 prefill GEMM. rel.err = max |y - ref| / max |ref|;
// timing as nh_x4_bench (10 calls per command buffer, median of --reps, rotating over weight copies of ~300 MB so
// calls read DRAM).
//   nh_x8_bench [--reps n] [--small] [--limit s] [filter ...]
// --small: tiny odd shapes, one copy, two rounds: run it under MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 first.
#include "metal/gpu.h"
#include "model/nemotron_h.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace ni {
extern const char* const kNhKernelSource;
}
using namespace ni;

namespace {

uint64_t g_rng = 0x9E3779B97F4A7C15ull;
uint64_t rnd() {
  uint64_t z = (g_rng += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
float nrand() {
  float s = 0;
  for (int i = 0; i < 4; ++i) s += float(rnd() & 0xFFFF) / 65535.0f;
  return (s - 2.0f) * 1.7320508f;
}
double e4m3(uint8_t x) {  // E4M3FN, sign included (0x7F / 0xFF NaN are never generated)
  const int e = (x >> 3) & 0xF, m = x & 7;
  const double v = e == 0 ? std::ldexp(double(m), -9) : std::ldexp(1.0 + double(m) / 8.0, e - 7);
  return (x & 0x80) ? -v : v;
}
const double kE2M1[16] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6};

void parallel_for(size_t n, const std::function<void(size_t, size_t)>& f) {
  const size_t nt = std::max(1u, std::min(12u, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  for (size_t t = 0; t < nt; ++t) th.emplace_back([&, t] { f(n * t / nt, n * (t + 1) / nt); });
  for (auto& x : th) x.join();
}

// a checkpoint-format matrix: raw bytes as stored, its dequantized weights (fp32, exact), the loader's repack
struct Mat {
  NhFmt fmt = NhFmt::FP8;
  uint32_t N = 0, K = 0;
  float gs = 1.0f;               // the true tensor scale
  std::vector<uint8_t> q, s;     // repacked codes, scales
  std::vector<float> w;          // dequantized x gs
};

Mat make(NhFmt fmt, uint32_t N, uint32_t K, float gs) {
  Mat m;
  m.fmt = fmt;
  m.N = N;
  m.K = K;
  m.gs = gs;
  m.w.resize(size_t(N) * K);
  if (fmt == NhFmt::FP8) {
    // realistic E4M3 weights: |e| mostly 3..12, some subnormals, both signs
    std::vector<uint8_t> raw(size_t(N) * K);
    for (size_t i = 0; i < raw.size(); ++i) {
      const uint64_t r = rnd();
      uint8_t b = (r & 31) == 0 ? uint8_t((r >> 8) & 7) : uint8_t(0x18 + ((r >> 8) % 0x48));
      if ((b & 0x7F) == 0x7F) b = 0x7E;
      b |= uint8_t((r >> 20) & 0x80);
      raw[i] = b;
      m.w[i] = float(e4m3(b) * gs);
    }
    m.q.resize(raw.size());
    nh_repack_rows(NhFmt::FP8, K, raw.data(), N, m.q.data(), nullptr);
  } else if (fmt == NhFmt::FP4) {
    std::vector<uint8_t> raw(size_t(N) * K / 2), code(size_t(N) * K);
    m.s.resize(size_t(N) * K / 16);
    for (size_t i = 0; i < code.size(); ++i) code[i] = uint8_t(rnd() & 15);
    for (size_t i = 0; i < raw.size(); ++i) raw[i] = uint8_t(code[2 * i] | (code[2 * i + 1] << 4));  // ModelOpt
    for (auto& b : m.s) {
      const uint64_t r = rnd();
      b = (r & 63) == 0 ? uint8_t((r >> 8) & 7) : uint8_t(0x20 + ((r >> 8) % 0x50));
      if (b == 0x7F) b = 0x7E;
    }
    for (size_t i = 0; i < m.w.size(); ++i) m.w[i] = float(kE2M1[code[i]] * e4m3(m.s[i / 16]) * gs);
    m.q.resize(raw.size());
    nh_repack_rows(NhFmt::FP4, K, raw.data(), N, m.q.data(), nullptr);
  } else {  // BF16
    m.q.resize(size_t(N) * K * 2);
    for (size_t i = 0; i < m.w.size(); ++i) {
      const float v = nrand() * 0.05f;
      uint32_t u;
      memcpy(&u, &v, 4);
      const uint16_t h = uint16_t(u >> 16);
      memcpy(m.q.data() + 2 * i, &h, 2);
      u = uint32_t(h) << 16;
      memcpy(&m.w[i], &u, 4);
    }
  }
  return m;
}

double max_abs(const std::vector<double>& v) {
  double m = 0;
  for (double x : v) m = std::max(m, std::fabs(x));
  return std::max(m, 1e-30);
}

struct Timer {
  std::vector<double> t;
  double med() {
    std::sort(t.begin(), t.end());
    return t.empty() ? 0 : t[t.size() / 2];
  }
};

}  // namespace

int main(int argc, char** argv) {
  int reps = 9, limit = 240;
  bool small = false;
  std::vector<std::string> only;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
    else if (a == "--limit" && i + 1 < argc) limit = atoi(argv[++i]);
    else if (a == "--small") small = true;
    else only.push_back(a);
  }
  auto want = [&](const std::string& n) {
    if (only.empty()) return true;
    for (auto& o : only)
      if (n.find(o) != std::string::npos) return true;
    return false;
  };
  alarm(unsigned(limit));  // a hard wall-clock limit (SIGALRM ends the process)
  if (small) reps = std::min(reps, 2);
  Gpu& g = Gpu::get();
  g.compile_extra("Nemotron-H", kNhKernelSource);
  Buffer dummy = g.alloc(4096, "dummy");
  Buffer part = g.alloc(size_t(96) << 20, "x4part"), cnt = g.alloc(1 << 16, "x4cnt");
  memset(cnt.ptr(), 0, cnt.size);

  // ---------------- FP8 dense ----------------
  struct Shape {
    const char* name;
    uint32_t N, K;
    float gs;
  };
  const Shape big[] = {{"in", 18560, 4096, 0.0021f}, {"out", 4096, 8192, 0.0013f}, {"q", 4096, 4096, 0.0009f},
                       {"kv", 256, 4096, 0.0011f},   {"head", 131072, 4096, 0.0007f}};
  const Shape tiny[] = {{"in", 200, 1024, 0.5f}, {"out", 72, 2048, 0.37f}, {"odd", 77, 1152, 1.0f}};
  struct Run {
    std::string name;
    int kind;  // 0 mv (fp32 x), 1 v8 / mm (half X, NhV8P), 2 gemm (NhGemmP)
    uint32_t rows_tg, threads, R, sk;
    int epi;
  };
  for (const Shape& sh : small ? std::vector<Shape>(std::begin(tiny), std::end(tiny)) : std::vector<Shape>(std::begin(big), std::end(big))) {
    std::vector<Run> runs = {{"nh_x8_mv", 0, 8, 64, 1, 1, 0},       {"nh_x8_mv_add", 0, 8, 64, 1, 1, 1},
                             {"nh_x8_mv_relu2", 0, 8, 64, 1, 1, 2}, {"nh_x8_mv_n2", 0, 4, 64, 1, 1, 0},
                             {"nh_x8_mv_n8", 0, 16, 64, 1, 1, 0},   {"nh_x8_mv", 0, 16, 128, 1, 1, 0},
                             {"nh_mv_fp8_r1", 0, 16, 128, 1, 1, 0}, {"nh_x8_v8", 1, 16, 256, 8, 1, 0},
                             {"nh_x8_v8", 1, 16, 256, 5, 1, 1},     {"nh_x8_v8", 1, 16, 256, 3, 1, 3},
                             {"nh_gemm_fp8_m16", 2, 16, 128, 13, 1, 0}, {"nh_gemm_fp8_m64", 2, 64, 128, 100, 1, 0},
                             {"nh_gemm_fp8_m64", 2, 64, 128, 512, 1, 0}};
    for (uint32_t sk : {1u, 2u, 4u, 8u}) {
      runs.push_back({"nh_x8_mm16", 1, 64, 128, 13, sk, 0});
      runs.push_back({"nh_x8_mm24", 1, 64, 128, 24, sk, 1});
      runs.push_back({"nh_x8_mm32", 1, 64, 128, 29, sk, 0});
    }
    std::vector<Run> sel;
    for (auto& r : runs) {
      char lab[128];
      snprintf(lab, sizeof lab, "fp8 %s %s R=%u sk=%u", sh.name, r.name.c_str(), r.R, r.sk);
      if (want(lab)) sel.push_back(r);
    }
    if (sel.empty()) continue;
    Mat m = make(NhFmt::FP8, sh.N, sh.K, sh.gs);
    const float gsk = sh.gs * 256.0f;  // NhMat::gs of FP8
    const size_t wb = size_t(sh.N) * sh.K;
    const int copies = small ? 1 : std::max(2, int(std::ceil(300e6 / double(wb))));
    Buffer W = g.alloc(wb * copies, "w8", false);
    for (int c = 0; c < copies; ++c) memcpy(W.as<uint8_t>() + c * wb, m.q.data(), wb);
    const uint32_t MX = 512;
    std::vector<float> x(size_t(MX) * sh.K);
    for (auto& v : x) v = float(__fp16(nrand()));
    // reference for all MX rows
    std::vector<double> ref(size_t(MX) * sh.N);
    parallel_for(sh.N, [&](size_t n0, size_t n1) {
      for (size_t n = n0; n < n1; ++n)
        for (uint32_t t = 0; t < MX; ++t) {
          double a = 0;
          for (uint32_t k = 0; k < sh.K; ++k) a += double(m.w[n * sh.K + k]) * double(x[size_t(t) * sh.K + k]);
          ref[size_t(t) * sh.N + n] = a;
        }
    });
    Buffer xf = g.alloc(size_t(MX) * sh.K * 4, "xf"), xh = g.alloc(size_t(MX) * sh.K * 2, "xh");
    memcpy(xf.ptr(), x.data(), x.size() * 4);
    for (size_t i = 0; i < x.size(); ++i) xh.as<__fp16>()[i] = __fp16(x[i]);
    Buffer y = g.alloc(size_t(MX) * sh.N * 4, "y");
    printf("== fp8 %s %ux%u (%d copies)\n", sh.name, sh.N, sh.K, copies);
    auto enc = [&](Encoder& e, const Run& r, int copy, int epi) {
      e.use(NemotronH::pipeline(r.name));
      e.buf(0, r.kind == 0 ? W : xh, r.kind == 0 ? size_t(copy) * wb : 0);
      if (r.kind == 0) {
        struct { uint32_t N, K, xs, ys; float gs; uint32_t R; } p{sh.N, sh.K, sh.K, sh.N, gsk, 1};
        e.buf(1, W, size_t(copy) * wb);
        e.buf(2, xf);
        e.buf(3, y);
        e.bytes(4, &p, sizeof p);
        e.dispatch((sh.N + r.rows_tg - 1) / r.rows_tg, 1, 1, r.threads);
      } else if (r.kind == 1) {
        struct { uint32_t N, K, x_ld, y_ld, R; float gs; uint32_t epi; } p{sh.N, sh.K, sh.K, sh.N, r.R, gsk, uint32_t(epi)};
        e.buf(1, W, size_t(copy) * wb);
        e.buf(2, W, size_t(copy) * wb);
        e.buf(3, y);
        e.bytes(4, &p, sizeof p);
        if (r.name == "nh_x8_v8") {
          e.dispatch((sh.N + 15) / 16, 1, 1, 256);
        } else {
          e.buf(5, part);
          e.buf(6, cnt);
          e.dispatch(r.sk, (sh.N + 63) / 64, 1, 128);
        }
      } else {
        struct { uint32_t M, N, K, x_ld, y_ld; float gs; uint32_t epi, qe, se, k; } p{r.R, sh.N, sh.K, sh.K, sh.N, gsk, uint32_t(epi), 0, 0, 1};
        e.buf(1, W, size_t(copy) * wb);
        e.buf(2, W, size_t(copy) * wb);
        e.buf(3, y);
        e.bytes(4, &p, sizeof p);
        e.buf(5, dummy);
        e.buf(6, dummy);
        e.buf(7, dummy);
        e.buf(8, dummy);
        e.dispatch((sh.N + 63) / 64, (r.R + r.rows_tg - 1) / r.rows_tg, 1, 128);
      }
    };
    for (const Run& r : sel) {
      if (r.kind == 1 && r.name != "nh_x8_v8" && size_t(r.sk) * ((sh.N + 63) / 64) * 2048 * 4 > part.size) continue;
      // check: y prefilled (add epilogue: random base; others: a sentinel that rows >= R must keep)
      std::vector<float> y0(size_t(MX) * sh.N);
      for (auto& v : y0) v = r.epi == 1 ? nrand() : -777.0f;
      memcpy(y.ptr(), y0.data(), y0.size() * 4);
      {
        auto e = g.begin();
        enc(*e, r, 0, r.epi);
        e->commit_and_wait();
      }
      double me = 0, mr = 0;
      bool pad = false;
      for (uint32_t t = 0; t < std::min(MX, std::max(r.R, 8u)); ++t)
        for (uint32_t n = 0; n < sh.N; ++n) {
          const size_t o = size_t(t) * sh.N + n;
          double v;
          if (r.epi == 3) v = t < r.R ? double(float(y.as<__fp16>()[o])) : 0;
          else v = y.as<float>()[o];
          if (t >= r.R) {
            if (r.epi != 3 && y.as<float>()[o] != y0[o]) pad = true;
            continue;
          }
          double rf = ref[o];
          if (r.epi == 1) v -= y0[o];
          if (r.epi == 2) rf = rf > 0 ? rf * rf : 0;
          me = std::max(me, std::isfinite(v) ? std::fabs(rf - v) : 1e30);
          mr = std::max(mr, std::fabs(rf));
        }
      for (size_t k = 0; k < cnt.size / 4; ++k)
        if (cnt.as<uint32_t>()[k]) {
          printf("  split-K counter %zu left at %u\n", k, cnt.as<uint32_t>()[k]);
          memset(cnt.ptr(), 0, cnt.size);
          break;
        }
      Timer tm;
      int cyc = 0;
      for (int rd = 0; rd < reps + 2; ++rd) {
        auto e = g.begin();
        for (int c = 0; c < 10; ++c) enc(*e, r, (cyc = (cyc + 1) % copies), 0);
        e->commit_and_wait();
        if (rd >= 2) tm.t.push_back(e->gpu_ms() / 10);
      }
      const double ms = tm.med();
      printf("fp8 %-4s %-16s R=%-3u sk=%u epi %d %8.4f ms %6.1f GB/s", sh.name, r.name.c_str(), r.R, r.sk, r.epi, ms,
             double(wb) / ms / 1e6);
      if (r.R > 1) printf(" %5.2f TF", 2.0 * double(wb) * r.R / ms / 1e9);
      printf("  err %.1e%s\n", me / std::max(mr, 1e-30), pad ? " PAD-WRITTEN" : "");
      fflush(stdout);
    }
    for (Buffer* b : {&W, &xf, &xh, &y}) g.release(*b);
  }

  // ---------------- BF16 prefill GEMM ----------------
  if (want("bf16")) {
    const uint32_t N = small ? 72 : 1024, K = small ? 256 : 4096, M = small ? 37 : 512;
    Mat m = make(NhFmt::BF16, N, K, 1.0f);
    Buffer W = g.alloc(m.q.size(), "wbf", false);
    memcpy(W.ptr(), m.q.data(), m.q.size());
    std::vector<float> x(size_t(M) * K);
    for (auto& v : x) v = float(__fp16(nrand()));
    Buffer xh = g.alloc(x.size() * 2, "xh"), y = g.alloc(size_t(M) * N * 4, "y");
    for (size_t i = 0; i < x.size(); ++i) xh.as<__fp16>()[i] = __fp16(x[i]);
    for (const char* name : {"nh_gemm_bf16w_m16", "nh_gemm_bf16w_m64"}) {
      const uint32_t bm = strstr(name, "m16") ? 16 : 64;
      struct { uint32_t M, N, K, x_ld, y_ld; float gs; uint32_t epi, qe, se, k; } p{M, N, K, K, N, 1.0f, 0, 0, 0, 1};
      Timer tm;
      for (int rd = 0; rd < reps + 2; ++rd) {
        auto e = g.begin();
        for (int c = 0; c < (rd ? 10 : 1); ++c) {
          e->use(NemotronH::pipeline(name));
          e->buf(0, xh);
          e->buf(1, W);
          e->buf(2, W);
          e->buf(3, y);
          e->bytes(4, &p, sizeof p);
          for (int b = 5; b <= 8; ++b) e->buf(b, dummy);
          e->dispatch((N + 63) / 64, (M + bm - 1) / bm, 1, 128);
        }
        e->commit_and_wait();
        if (rd >= 2) tm.t.push_back(e->gpu_ms() / 10);
      }
      double me = 0, mr = 0;
      for (uint32_t t = 0; t < M; ++t)
        for (uint32_t n = 0; n < N; ++n) {
          double a = 0;
          for (uint32_t k = 0; k < K; ++k) a += double(m.w[size_t(n) * K + k]) * x[size_t(t) * K + k];
          me = std::max(me, std::fabs(a - y.as<float>()[size_t(t) * N + n]));
          mr = std::max(mr, std::fabs(a));
        }
      printf("bf16 %s M=%u %ux%u %8.4f ms %5.2f TF  err %.1e\n", name, M, N, K, tm.med(),
             2.0 * N * K * M / tm.med() / 1e9, me / mr);
    }
    for (Buffer* b : {&W, &xh, &y}) g.release(*b);
  }

  // ---------------- NVFP4 routed experts ----------------
  if (want("moe")) {
    const uint32_t E = small ? 8 : 512, I = small ? 384 : 2688, D = small ? 256 : 1024, k = small ? 3 : 22;
    // (E x 3.1 MB exceeds the caches: the pairs pick experts spread over all of it, as a model's do)
    printf("== moe NVFP4 E %u, I %u, latent %u, top-%u\n", E, I, D, k);
    std::vector<Mat> up(E), dn(E);
    std::vector<float> gsu(E), gsd(E);
    for (uint32_t e = 0; e < E; ++e) {
      gsu[e] = 0.0004f * (1.0f + 0.5f * float(e % 7));
      gsd[e] = 0.0003f * (1.0f + 0.3f * float(e % 5));
      up[e] = make(NhFmt::FP4, I, D, gsu[e]);
      dn[e] = make(NhFmt::FP4, D, I, gsd[e]);
    }
    const size_t uq = size_t(I) * D / 2, us = size_t(I) * D / 16, dq = size_t(D) * I / 2, ds = size_t(D) * I / 16;
    Buffer Uq = g.alloc(uq * E, "upq", false), Us = g.alloc(us * E, "ups", false), Dq = g.alloc(dq * E, "dnq", false),
           Ds = g.alloc(ds * E, "dns", false), Gu = g.alloc(E * 4, "gsu"), Gd = g.alloc(E * 4, "gsd");
    for (uint32_t e = 0; e < E; ++e) {
      memcpy(Uq.as<uint8_t>() + e * uq, up[e].q.data(), uq);
      memcpy(Us.as<uint8_t>() + e * us, up[e].s.data(), us);
      memcpy(Dq.as<uint8_t>() + e * dq, dn[e].q.data(), dq);
      memcpy(Ds.as<uint8_t>() + e * ds, dn[e].s.data(), ds);
    }
    memcpy(Gu.ptr(), gsu.data(), E * 4);
    memcpy(Gd.ptr(), gsd.data(), E * 4);
    for (uint32_t R : {1u, 8u, small ? 37u : 512u}) {
      const uint32_t P = R * k;
      std::vector<int> eid(P);
      std::vector<float> ew(P), lat(size_t(R) * D);
      for (uint32_t r = 0; r < R; ++r) {
        std::vector<int> used;
        for (uint32_t j = 0; j < k; ++j) {
          int e;
          do e = int(rnd() % E);
          while (std::find(used.begin(), used.end(), e) != used.end());
          used.push_back(e);
          eid[r * k + j] = e;
          ew[r * k + j] = 0.05f + float(rnd() % 100) / 400.0f;
        }
      }
      for (auto& v : lat) v = float(__fp16(nrand()));
      // reference: h = relu(up . lat)^2, out[r] = sum_j ew * dn . h
      std::vector<double> href(size_t(P) * I), oref(size_t(R) * D, 0.0);
      parallel_for(P, [&](size_t p0, size_t p1) {
        for (size_t pr = p0; pr < p1; ++pr) {
          const Mat& u = up[size_t(eid[pr])];
          const float* xr = lat.data() + (pr / k) * D;
          for (uint32_t n = 0; n < I; ++n) {
            double a = 0;
            for (uint32_t kk = 0; kk < D; ++kk) a += double(u.w[size_t(n) * D + kk]) * xr[kk];
            href[pr * I + n] = a > 0 ? a * a : 0;
          }
        }
      });
      // the down projection reads h as the kernels hold it: fp32 (per pair kernels) or half (GEMMs)
      auto down_ref = [&](bool half_h) {
        std::vector<double> o(size_t(R) * D, 0.0);
        parallel_for(R, [&](size_t r0, size_t r1) {
          for (size_t r = r0; r < r1; ++r)
            for (uint32_t j = 0; j < k; ++j) {
              const size_t pr = r * k + j;
              const Mat& d = dn[size_t(eid[pr])];
              for (uint32_t n = 0; n < D; ++n) {
                double a = 0;
                for (uint32_t kk = 0; kk < I; ++kk) {
                  const double h = half_h ? double(float(__fp16(float(href[pr * I + kk])))) : double(float(href[pr * I + kk]));
                  a += double(d.w[size_t(n) * I + kk]) * h;
                }
                o[r * D + n] += double(ew[pr]) * a;
              }
            }
        });
        return o;
      };
      Buffer eidb = g.alloc(P * 4, "eid"), ewb = g.alloc(P * 4, "ew"), latf = g.alloc(lat.size() * 4, "latf"),
             lath = g.alloc(lat.size() * 2, "lath"), hb = g.alloc(size_t(P) * I * 4, "h"),
             pb = g.alloc(size_t(P) * D * 4, "part"), ob = g.alloc(size_t(R) * D * 4, "out"), live = g.alloc(16, "live");
      memcpy(eidb.ptr(), eid.data(), P * 4);
      memcpy(ewb.ptr(), ew.data(), P * 4);
      memcpy(latf.ptr(), lat.data(), lat.size() * 4);
      for (size_t i = 0; i < lat.size(); ++i) lath.as<__fp16>()[i] = __fp16(lat[i]);
      live.as<uint32_t>()[0] = 0xffffffffu;
      struct { uint32_t I, D, k, uqs, uss, dqs, dss, hs; } mp{I, D, k, uint32_t(uq), uint32_t(us), uint32_t(dq), uint32_t(ds), I};
      if (R <= 8 && want("moe pair")) {
        auto enc = [&](Encoder& e) {
          e.use(NemotronH::pipeline("nh_x4_moe_up"));
          e.buf(0, Uq);
          e.buf(1, Us);
          e.buf(2, latf);
          e.buf(3, hb);
          e.buf(4, eidb);
          e.bytes(5, &mp, sizeof mp);
          e.buf(7, live);
          e.buf(9, Gu);
          e.dispatch((I + 7) / 8, P, 1, 64);
          e.use(NemotronH::pipeline("nh_x4_moe_dn"));
          e.buf(0, Dq);
          e.buf(1, Ds);
          e.buf(2, hb);
          e.buf(3, pb);
          e.buf(4, eidb);
          e.buf(5, ewb);
          e.bytes(6, &mp, sizeof mp);
          e.buf(8, live);
          e.buf(9, Gd);
          e.dispatch((D + 7) / 8, P, 1, 64);
          e.use(NemotronH::pipeline("nh_moe2_sum"));
          e.buf(0, pb);
          e.buf(1, ob);
          e.bytes(2, &mp, sizeof mp);
          e.buf(3, live);
          e.dispatch(R, 1, 1, 256);
        };
        {
          auto e = g.begin();
          enc(*e);
          e->commit_and_wait();
        }
        double he = 0, hm = 0;
        for (size_t i = 0; i < href.size(); ++i) {
          he = std::max(he, std::fabs(href[i] - hb.as<float>()[i]));
          hm = std::max(hm, std::fabs(href[i]));
        }
        const std::vector<double> o = down_ref(false);
        double oe = 0;
        for (size_t i = 0; i < o.size(); ++i) oe = std::max(oe, std::fabs(o[i] - ob.as<float>()[i]));
        Timer tm;
        for (int rd = 0; rd < reps + 2; ++rd) {
          auto e = g.begin();
          for (int c = 0; c < 10; ++c) enc(*e);
          e->commit_and_wait();
          if (rd >= 2) tm.t.push_back(e->gpu_ms() / 10);
        }
        const double bytes = double(P) * (uq + us + dq + ds);
        printf("moe pair R=%u: up + down + sum %8.4f ms %6.1f GB/s  err h %.1e out %.1e\n", R, tm.med(),
               bytes / tm.med() / 1e6, he / hm, oe / max_abs(o));
      }
      if (R > 8 && want("moe gemm")) {
        // bucket the pairs, then the prefill GEMMs as the engine runs them (rows per tile from the pairs per expert)
        const uint32_t per = P / E, bm = per >= 40 ? 64 : per >= 12 ? 32 : 16;
        const std::string ms = "_m" + std::to_string(bm);
        Buffer sorted = g.alloc(P * 4, "sorted"), rank = g.alloc(P * 4, "rank"),
               tiles = g.alloc((P / 16 + E + 2) * 16, "tiles"), ntl = g.alloc(16, "ntiles"),
               hh = g.alloc(size_t(P) * I * 2, "hh"), ows = g.alloc(size_t(R) * D * 2, "lo");
        const uint32_t max_tiles = P / bm + E + 1;
        auto enc = [&](Encoder& e) {
          struct { uint32_t P, E, BM, k; } bp{P, E, bm, 0};
          e.use(NemotronH::pipeline("nh_moe_bucket"));
          e.buf(0, eidb);
          e.buf(1, sorted);
          e.buf(2, tiles);
          e.buf(3, ntl);
          e.buf(4, rank);
          e.bytes(5, &bp, sizeof bp);
          e.buf(6, live);
          e.dispatch(1, 1, 1, 1024);
          struct { uint32_t M, N, K, x_ld, y_ld; float gs; uint32_t epi, qe, se, k; } gu{P, I, D, D, I, 1.0f, 2,
                                                                                          uint32_t(uq), uint32_t(us), k};
          e.use(NemotronH::pipeline("nh_gemm_fp4_moe_up" + ms));
          e.buf(0, lath);
          e.buf(1, Uq);
          e.buf(2, Us);
          e.buf(3, hh);
          e.bytes(4, &gu, sizeof gu);
          e.buf(5, tiles);
          e.buf(6, ntl);
          e.buf(7, sorted);
          e.buf(8, dummy);
          e.buf(9, Gu);
          e.dispatch((I + 63) / 64, max_tiles, 1, 128);
          struct { uint32_t M, N, K, x_ld, y_ld; float gs; uint32_t epi, qe, se, k; } gd{P, D, I, I, D, 1.0f, 0,
                                                                                          uint32_t(dq), uint32_t(ds), k};
          e.use(NemotronH::pipeline("nh_gemm_fp4_moe_dn" + ms));
          e.buf(0, hh);
          e.buf(1, Dq);
          e.buf(2, Ds);
          e.buf(3, pb);
          e.bytes(4, &gd, sizeof gd);
          e.buf(5, tiles);
          e.buf(6, ntl);
          e.buf(7, sorted);
          e.buf(8, dummy);
          e.buf(9, Gd);
          e.dispatch((D + 63) / 64, max_tiles, 1, 128);
          struct { uint32_t D, k; } cp{D, k};
          e.use(NemotronH::pipeline("nh_moe_combine"));
          e.buf(0, pb);
          e.buf(1, ewb);
          e.buf(2, ows);
          e.bytes(3, &cp, sizeof cp);
          e.dispatch(R, 1, 1, 256);
        };
        {
          auto e = g.begin();
          enc(*e);
          e->commit_and_wait();
        }
        double he = 0, hm = 0;
        for (uint32_t j = 0; j < P; ++j) {  // hh is in sorted order: row j holds pair sorted[j]
          const uint32_t pr = sorted.as<uint32_t>()[j];
          for (uint32_t n = 0; n < I; ++n) {
            he = std::max(he, std::fabs(href[size_t(pr) * I + n] - double(float(hh.as<__fp16>()[size_t(j) * I + n]))));
            hm = std::max(hm, std::fabs(href[size_t(pr) * I + n]));
          }
        }
        const std::vector<double> o = down_ref(true);
        double oe = 0;
        for (size_t i = 0; i < o.size(); ++i) oe = std::max(oe, std::fabs(o[i] - double(float(ows.as<__fp16>()[i]))));
        Timer tm;
        for (int rd = 0; rd < reps + 2; ++rd) {
          auto e = g.begin();
          enc(*e);
          e->commit_and_wait();
          if (rd >= 2) tm.t.push_back(e->gpu_ms());
        }
        printf("moe gemm R=%u (tiles of %u): bucket + up + down + combine %8.3f ms %5.2f TF  err h %.1e out %.1e\n", R,
               bm, tm.med(), 4.0 * P * I * D / tm.med() / 1e9, he / hm, oe / max_abs(o));
        for (Buffer* b : {&sorted, &rank, &tiles, &ntl, &hh, &ows}) g.release(*b);
      }
      for (Buffer* b : {&eidb, &ewb, &latf, &lath, &hb, &pb, &ob, &live}) g.release(*b);
    }
  }
  return 0;
}
