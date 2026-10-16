// Gated DeltaNet prefill microbenchmark + equivalence check on random data: the default path
// (gdn_prep -> gdn_step -> rows_to_half) vs the NI_GDN_PAR path (gdn_prep_par -> gdn_step_par_l*
// -> gdn_gnorm), both from the same conv/recurrent state. Per-kernel GPU time is the best of 5
// isolated dispatches; outputs, the rolled conv window and the recurrent state are compared bitwise.
//   bench_gdn [T=512] [Hv=32] [Hk=16] [step kernels...]   (35B: Hv 32 Hk 16; 27B: Hv 48 Hk 16)
// NI_GDN_SGS sets the simdgroups per threadgroup of the step kernels (default 2, as the engine).
// NI_MATH_SAFE=1 compiles without fast math: then every output must be bit-identical (with fast
// math, expect rounding-level differences since each kernel's sums get reassociated differently).
#include "metal/gpu.h"
#include "model/kparams.h"
#include "util/log.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>
using namespace ni;

static uint16_t to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7fff + ((u >> 16) & 1);  // round to nearest even
  return uint16_t(u >> 16);
}

int main(int argc, char** argv) {
  const uint32_t T = argc > 1 ? atoi(argv[1]) : 512, Hv = argc > 2 ? atoi(argv[2]) : 32;
  const uint32_t Hk = argc > 3 ? atoi(argv[3]) : 16;
  std::vector<std::string> steps;
  for (int i = 4; i < argc; ++i) steps.push_back(argv[i]);
  if (steps.empty()) steps = {"gdn_step_par_l4", "gdn_step_par_l8", "gdn_step_par_l16", "gdn_step_par_l32"};
  const uint32_t sgs = getenv("NI_GDN_SGS") ? uint32_t(atoi(getenv("NI_GDN_SGS"))) : 2u;
  constexpr uint32_t D = 128;
  const uint32_t C = 2 * Hk * D + Hv * D, VD = Hv * D, QK = Hk * D;
  const float eps = 1e-6f;
  Gpu& g = Gpu::get();

  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0, 1);
  auto fill = [&](const Buffer& b, size_t n, float sd, float mean) {
    for (size_t i = 0; i < n; ++i) b.as<float>()[i] = mean + sd * nd(rng);
  };
  auto fill_bf = [&](const Buffer& b, size_t n, float sd, float mean) {
    for (size_t i = 0; i < n; ++i) b.as<uint16_t>()[i] = to_bf16(mean + sd * nd(rng));
  };
  Buffer qkv = g.alloc(size_t(T) * C * 4), a = g.alloc(size_t(T) * Hv * 4), bb = g.alloc(size_t(T) * Hv * 4);
  Buffer z = g.alloc(size_t(T) * VD * 4);
  Buffer conv_w = g.alloc(size_t(C) * 4 * 2), A_log = g.alloc(Hv * 2), dt_bias = g.alloc(Hv * 2);
  Buffer norm_w = g.alloc(D * 2);
  Buffer conv0 = g.alloc(size_t(3) * C * 4), rec0 = g.alloc(size_t(Hv) * D * D * 4);
  fill(qkv, size_t(T) * C, 1.0f, 0.0f);
  fill(a, size_t(T) * Hv, 1.0f, 0.0f);
  fill(bb, size_t(T) * Hv, 1.0f, 0.0f);
  fill(z, size_t(T) * VD, 1.0f, 0.0f);
  fill_bf(conv_w, size_t(C) * 4, 0.3f, 0.0f);
  fill_bf(A_log, Hv, 0.5f, 0.0f);
  fill_bf(dt_bias, Hv, 0.5f, 0.0f);
  fill_bf(norm_w, D, 0.1f, 1.0f);
  fill(conv0, size_t(3) * C, 1.0f, 0.0f);
  fill(rec0, size_t(Hv) * D * D, 0.05f, 0.0f);

  struct Path {
    Buffer conv, rec, q, k, v, gg, beta, out, y;
  };
  auto make_path = [&]() {
    Path p;
    p.conv = g.alloc(size_t(3) * C * 4);
    p.rec = g.alloc(size_t(Hv) * D * D * 4);
    p.q = g.alloc(size_t(T) * QK * 4);
    p.k = g.alloc(size_t(T) * QK * 4);
    p.v = g.alloc(size_t(T) * VD * 4);
    p.gg = g.alloc(size_t(T) * Hv * 4);
    p.beta = g.alloc(size_t(T) * Hv * 4);
    p.out = g.alloc(size_t(T) * VD * 4);
    p.y = g.alloc(size_t(T) * VD * 2);
    return p;
  };
  Path ref = make_path(), par = make_path();
  auto reset = [&](Path& p) {
    memcpy(p.conv.ptr(), conv0.ptr(), conv0.size);
    memcpy(p.rec.ptr(), rec0.ptr(), rec0.size);
  };

  const kp::GdnPrep pp{T, Hk, Hv, C};
  const kp::GdnStep sp{T, Hk, Hv, eps};
  const kp::Rows rp{VD, 0, eps};
  auto enc_prep = [&](Encoder& e, Path& P, bool parallel) {
    e.use(g.pipeline(parallel ? "gdn_prep_par" : "gdn_prep"));
    e.buf(0, qkv); e.buf(1, a); e.buf(2, bb); e.buf(3, P.conv);
    e.buf(4, conv_w); e.buf(5, A_log); e.buf(6, dt_bias);
    e.buf(7, P.q); e.buf(8, P.k); e.buf(9, P.v); e.buf(10, P.gg); e.buf(11, P.beta);
    e.val(12, pp);
    if (parallel) e.dispatch((T + 31) / 32, C / 128 + 1, 1, 128);
    else e.dispatch(C / 128, 1, 1, 128);
  };
  auto enc_step = [&](Encoder& e, Path& P) {
    e.use(g.pipeline("gdn_step"));
    e.buf(0, P.q); e.buf(1, P.k); e.buf(2, P.v); e.buf(3, P.gg); e.buf(4, P.beta);
    e.buf(5, P.rec); e.buf(6, z); e.buf(7, norm_w); e.buf(8, P.out);
    e.val(9, sp);
    e.dispatch(Hv, 1, 1, 1024);
  };
  auto enc_half = [&](Encoder& e, Path& P) {
    e.use(g.pipeline("rows_to_half"));
    e.buf(0, P.out); e.buf(1, P.out); e.buf(2, P.y);
    e.val(3, rp);
    e.dispatch(T, 1, 1, 256);
  };
  auto enc_step_par = [&](Encoder& e, Path& P, const std::string& name) {
    uint32_t lanes = uint32_t(atoi(name.c_str() + name.rfind("_l") + 2));
    uint32_t rows_tg = lanes && lanes <= 32 ? sgs * (32 / lanes) : 0;
    if (!rows_tg || 128 % rows_tg) fatal("%s with %u simdgroups: bad layout", name.c_str(), sgs);
    e.use(g.pipeline(name));
    e.buf(0, P.q); e.buf(1, P.k); e.buf(2, P.v); e.buf(3, P.gg); e.buf(4, P.beta);
    e.buf(5, P.rec); e.buf(6, P.out);
    e.val(7, sp);
    e.dispatch(Hv * (128 / rows_tg), 1, 1, 32 * sgs);
  };
  auto enc_gnorm = [&](Encoder& e, Path& P) {
    e.use(g.pipeline("gdn_gnorm"));
    e.buf(0, P.out); e.buf(1, z); e.buf(2, norm_w); e.buf(3, P.y);
    e.val(4, sp);
    e.dispatch((T * Hv + 7) / 8, 1, 1, 256);
  };
  // best of 5 after a warmup; each run is its own command buffer (state keeps evolving: timing only)
  auto timed = [&](const std::function<void(Encoder&)>& f) {
    double best = 1e9;
    for (int i = 0; i < 6; ++i) {
      auto e = g.begin();
      f(*e);
      e->commit_and_wait();
      if (i) best = std::min(best, e->gpu_ms());
    }
    return best;
  };
  auto run = [&](const std::function<void(Encoder&)>& f) {
    auto e = g.begin();
    f(*e);
    e->commit_and_wait();
  };
  // bitwise comparison of float (or half) arrays; max |diff| is also given relative to max |ref|.
  // NaN anywhere counts as a mismatch with an infinite diff.
  double worst_rel = 0;
  auto cmp = [&worst_rel](const char* what, const Buffer& x, const Buffer& y, size_t n, bool half) {
    size_t bad = 0;
    double mx = 0, ref_mx = 0;
    for (size_t i = 0; i < n; ++i) {
      double dx, dy;
      bool same;
      if (half) {
        same = x.as<uint16_t>()[i] == y.as<uint16_t>()[i];
        dx = double(x.as<__fp16>()[i]);
        dy = double(y.as<__fp16>()[i]);
      } else {
        same = x.as<uint32_t>()[i] == y.as<uint32_t>()[i];
        dx = x.as<float>()[i];
        dy = y.as<float>()[i];
      }
      ref_mx = std::max(ref_mx, std::fabs(dx));
      if (!same) {
        ++bad;
        mx = std::max(mx, std::isnan(dx) || std::isnan(dy) ? INFINITY : std::fabs(dx - dy));
      }
    }
    double rel = mx / std::max(ref_mx, 1e-30);
    worst_rel = std::max(worst_rel, rel);
    printf("    %-5s %9zu / %-9zu differ  max |diff| %.3g (%.2g of max |ref|)\n", what, bad, n, mx, rel);
    return bad;
  };

  printf("T=%u Hv=%u Hk=%u (C=%u), step threadgroups of %u simdgroups\n", T, Hv, Hk, C, sgs);
  double t_prep = timed([&](Encoder& e) { enc_prep(e, ref, false); });
  double t_step = timed([&](Encoder& e) { enc_step(e, ref); });
  double t_half = timed([&](Encoder& e) { enc_half(e, ref); });
  printf("  default: gdn_prep %.3f ms + gdn_step %.3f ms + rows_to_half %.3f ms = %.3f ms\n", t_prep, t_step,
         t_half, t_prep + t_step + t_half);
  double t_prep_par = timed([&](Encoder& e) { enc_prep(e, par, true); });
  double t_gnorm = timed([&](Encoder& e) { enc_gnorm(e, par); });

  reset(ref);
  run([&](Encoder& e) {
    enc_prep(e, ref, false);
    enc_step(e, ref);
    enc_half(e, ref);
  });
  size_t bad_total = 0;
  for (const std::string& name : steps) {
    double t_sp = timed([&](Encoder& e) { enc_step_par(e, par, name); });
    printf("  NI_GDN_PAR %s: gdn_prep_par %.3f ms + step %.3f ms + gdn_gnorm %.3f ms = %.3f ms (%.2fx)\n",
           name.c_str(), t_prep_par, t_sp, t_gnorm, t_prep_par + t_sp + t_gnorm,
           (t_prep + t_step + t_half) / (t_prep_par + t_sp + t_gnorm));
    reset(par);
    run([&](Encoder& e) {
      enc_prep(e, par, true);
      enc_step_par(e, par, name);
      enc_gnorm(e, par);
    });
    bad_total += cmp("q", ref.q, par.q, size_t(T) * QK, false);
    bad_total += cmp("k", ref.k, par.k, size_t(T) * QK, false);
    bad_total += cmp("v", ref.v, par.v, size_t(T) * VD, false);
    bad_total += cmp("g", ref.gg, par.gg, size_t(T) * Hv, false);
    bad_total += cmp("beta", ref.beta, par.beta, size_t(T) * Hv, false);
    bad_total += cmp("conv", ref.conv, par.conv, size_t(3) * C, false);
    bad_total += cmp("state", ref.rec, par.rec, size_t(Hv) * D * D, false);
    bad_total += cmp("y", ref.y, par.y, size_t(T) * VD, true);
  }
  // Under the engine's fast math each kernel's sums are reassociated differently, so expect small
  // differences (fp32 rounding, a few 1-ulp half outputs); NI_MATH_SAFE=1 must give bit-identical.
  if (!bad_total) printf("all outputs bit-identical\n");
  else printf("outputs differ, worst max |diff| %.2g of max |ref| (%s)\n", worst_rel,
              worst_rel < 1e-3 ? "rounding-level; NI_MATH_SAFE=1 should be bit-identical" : "TOO LARGE: likely a bug");
  return 0;
}
