// Hy4-flash GPU blocks (src/model/hy4_ops.h) against fp64 CPU versions written from the reference
// modeling_hy_v4_flash.py (transformers c2ec8505), on random inputs. No model.
//   hy4_ops_check --small   small sizes (run first, under MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1)
//   hy4_ops_check           Hy4 sizes (D 3072, E 512 k 16, H 32/64, T 1/7/300/700, window 512, ring caches)
//   NI_KV_I8=1 hy4_ops_check   the int8 K/V rows (looser attention tolerance)
//   hy4_ops_check --bench   timings: decode H 64 over 2048 keys, prefill T 512
// Takes no GPU lock itself: run it under `lockf -k -t 600 /tmp/kapitsa-gpu.lock`.
#include "loader/safetensors.h"
#include "metal/gpu.h"
#include "model/hy4_ops.h"
#include "util/log.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace ni;
namespace {

std::mt19937_64 rng(1234);
double normal() { return std::normal_distribution<double>(0.0, 1.0)(rng); }
double uni(double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); }

float bf16_round(float x) {  // round to nearest even
  uint32_t b;
  memcpy(&b, &x, 4);
  b += 0x7fffu + ((b >> 16) & 1u);
  b &= 0xffff0000u;
  float r;
  memcpy(&r, &b, 4);
  return r;
}
uint16_t bf16_bits(float x) {
  x = bf16_round(x);
  uint32_t b;
  memcpy(&b, &x, 4);
  return uint16_t(b >> 16);
}
double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double half_val(uint16_t h) {
  const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
  double v = e == 0 ? std::ldexp(double(m), -24) : std::ldexp(double(m | 1024), e - 25);
  return s ? -v : v;
}

Buffer fbuf(size_t n, const char* label = "check") {
  Buffer b = Gpu::get().alloc(std::max<size_t>(n, 1) * 4, label);
  memset(b.ptr(), 0, std::max<size_t>(n, 1) * 4);
  return b;
}
void put(const Buffer& b, const std::vector<double>& v) {
  for (size_t i = 0; i < v.size(); ++i) b.as<float>()[i] = float(v[i]);
}
std::vector<double> get(const Buffer& b, size_t n, bool half = false) {
  std::vector<double> r(n);
  for (size_t i = 0; i < n; ++i) r[i] = half ? half_val(b.as<uint16_t>()[i]) : double(b.as<float>()[i]);
  return r;
}

// A weight tensor of n values in dtype dt; vals is rounded to what was stored (the CPU reference reads those)
Tensor tensor(std::vector<double>& vals, DType dt) {
  Tensor t;
  t.dtype = dt;
  t.shape = {int64_t(vals.size())};
  t.buf = Gpu::get().alloc(vals.size() * (dt == DType::F32 ? 4 : 2), "check_w");
  for (size_t i = 0; i < vals.size(); ++i) {
    if (dt == DType::F32) {
      t.buf.as<float>()[i] = float(vals[i]);
      vals[i] = float(vals[i]);
    } else if (dt == DType::F16) {
      const _Float16 h = _Float16(float(vals[i]));
      memcpy(&t.buf.as<uint16_t>()[i], &h, 2);
      vals[i] = float(h);
    } else {
      const float r = bf16_round(float(vals[i]));
      t.buf.as<uint16_t>()[i] = bf16_bits(float(vals[i]));
      vals[i] = r;
    }
  }
  return t;
}
const char* dtn(DType t) { return t == DType::BF16 ? "bf16" : t == DType::F16 ? "f16" : "f32"; }
std::vector<double> randv(size_t n, double scale = 1.0, double mean = 0.0) {
  std::vector<double> v(n);
  for (auto& x : v) x = mean + scale * normal();
  return v;
}

struct Err {
  double max_abs = 0, max_ref = 0;
  size_t n = 0, bad = 0;
  void add(double g, double r) {
    const double d = std::fabs(g - r);
    if (!(d <= max_abs)) max_abs = std::isnan(d) ? INFINITY : std::max(max_abs, d);
    max_ref = std::max(max_ref, std::fabs(r));
    ++n;
  }
  double rel() const { return max_ref > 0 ? max_abs / max_ref : max_abs; }
};
int failures = 0;
void report(const std::string& name, const Err& e, double tol) {
  const bool ok = e.rel() <= tol;
  if (!ok) ++failures;
  printf("%-58s n %9zu  max_abs %.3e  rel %.3e  (tol %.0e) %s\n", name.c_str(), e.n, e.max_abs, e.rel(), tol,
         ok ? "ok" : "FAIL");
  fflush(stdout);
}

void run(const std::function<void(Encoder&)>& f) {
  auto e = Gpu::get().begin();
  f(*e);
  e->commit_and_wait();
}

// ---- hyper-connections ----
void check_hc(uint32_t D, int R, uint32_t nout, bool with_y, DType fdt, bool u_half) {
  const uint32_t S = 4, SD = S * D;
  auto X = randv(size_t(R) * SD, 1.0);
  for (size_t i = 0; i < X.size(); ++i) X[i] *= 1.0 + (i / D) % 4;  // streams of different sizes
  auto y = randv(size_t(R) * D, 0.5);
  std::vector<double> pp(size_t(R) * S);
  for (auto& v : pp) v = uni(0.1, 2.0);
  auto fn = randv(size_t(nout) * SD, 0.02);
  auto base = randv(nout, 0.5);
  std::vector<double> scale = nout == 8 ? std::vector<double>{0.7, 0.4} : std::vector<double>{0.6};
  Tensor tfn = tensor(fn, fdt), tb = tensor(base, fdt), ts = tensor(scale, DType::F32);
  hy4::Hc w{&tfn, &tb, &ts, nout};
  hy4::HcConst c;
  c.D = D;
  Buffer bX = fbuf(X.size()), by = fbuf(y.size()), bpp = fbuf(pp.size()), bu = fbuf(size_t(R) * D),
         bpo = fbuf(size_t(R) * S);
  put(bX, X);
  put(by, y);
  put(bpp, pp);
  run([&](Encoder& e) {
    hy4::enc_hc_pre(e, w, c, R, bX, with_y ? &by : nullptr, with_y ? &bpp : nullptr, bu, u_half,
                    nout == 8 ? &bpo : nullptr);
  });
  Err eu, ep, ex;
  auto gu = get(bu, size_t(R) * D, u_half), gpo = get(bpo, size_t(R) * S), gX = get(bX, X.size());
  for (int r = 0; r < R; ++r) {
    std::vector<double> x(X.begin() + size_t(r) * SD, X.begin() + size_t(r + 1) * SD);
    if (with_y)  // residual + post * y (HYV4FlashDecoderLayer), flushed into X first
      for (uint32_t s = 0; s < S; ++s)
        for (uint32_t i = 0; i < D; ++i) x[s * D + i] += pp[r * S + s] * y[size_t(r) * D + i];
    for (size_t i = 0; i < SD; ++i) ex.add(gX[size_t(r) * SD + i], x[i]);
    double ss = 0;
    for (double v : x) ss += v * v;
    const double inv = 1.0 / std::sqrt(ss / SD + c.rms_eps);  // HYV4FlashUnweightedRMSNorm
    std::vector<double> mix(nout);
    for (uint32_t j = 0; j < nout; ++j) {
      double d = 0;
      for (size_t i = 0; i < SD; ++i) d += fn[j * SD + i] * x[i];
      mix[j] = d * inv;  // F.linear(flat, fn) * input_norm(flat)
    }
    double pre[4];
    for (uint32_t s = 0; s < S; ++s) pre[s] = sigm(mix[s] * scale[0] + base[s]) + c.hc_eps;
    if (nout == 8)
      for (uint32_t s = 0; s < S; ++s)
        ep.add(gpo[r * S + s], c.mag * sigm(mix[S + s] * scale[1] + base[S + s]) + c.hc_eps);
    for (uint32_t i = 0; i < D; ++i) {
      double u = 0;
      for (uint32_t s = 0; s < S; ++s) u += pre[s] * x[s * D + i];
      eu.add(gu[size_t(r) * D + i], u);
    }
  }
  char nm[128];
  snprintf(nm, sizeof nm, "hc_pre D %u R %d nout %u%s fn %s%s", D, R, nout, with_y ? " +y" : "",
           dtn(fdt), u_half ? " u half" : "");
  report(std::string(nm) + ": u", eu, u_half ? 2e-3 : 2e-5);
  if (nout == 8) report(std::string(nm) + ": post", ep, 2e-5);
  if (with_y) report(std::string(nm) + ": X", ex, 1e-6);
}

void check_inject_expand(uint32_t D, int R) {
  auto X = randv(size_t(R) * 4 * D), y = randv(size_t(R) * D), post = randv(size_t(R) * 4), emb = randv(size_t(R) * D);
  Buffer bX = fbuf(X.size()), by = fbuf(y.size()), bp = fbuf(post.size()), be = fbuf(emb.size()), bX2 = fbuf(X.size());
  put(bX, X);
  put(by, y);
  put(bp, post);
  put(be, emb);
  run([&](Encoder& e) {
    hy4::enc_hc_inject(e, D, R, bX, by, bp);
    hy4::enc_hc_expand(e, D, R, be, bX2);
  });
  Err ei, ee;
  auto gX = get(bX, X.size()), gX2 = get(bX2, X.size());
  for (int r = 0; r < R; ++r)
    for (uint32_t s = 0; s < 4; ++s)
      for (uint32_t i = 0; i < D; ++i) {
        const size_t k = (size_t(r) * 4 + s) * D + i;
        ei.add(gX[k], X[k] + post[r * 4 + s] * y[size_t(r) * D + i]);
        ee.add(gX2[k], float(emb[size_t(r) * D + i]));
      }
  report("hc_inject D " + std::to_string(D) + " R " + std::to_string(R), ei, 1e-6);
  report("hc_expand D " + std::to_string(D) + " R " + std::to_string(R), ee, 0);
}

// ---- RMSNorm, add, SwiGLU ----
void check_rmsnorm(uint32_t D, int R, DType wdt, bool y_half, bool alias) {
  auto x = randv(size_t(R) * D, 3.0);
  auto w = randv(D, 0.3, 1.0);
  Tensor tw = tensor(w, wdt);
  Buffer bx = fbuf(x.size()), by = fbuf(x.size());
  put(bx, x);
  const Buffer& out = alias ? bx : by;
  run([&](Encoder& e) { hy4::enc_rmsnorm(e, D, R, bx, tw, 1e-5f, out, y_half); });
  auto g = get(out, x.size(), y_half);
  Err er;
  for (int r = 0; r < R; ++r) {
    double ss = 0;
    for (uint32_t i = 0; i < D; ++i) ss += x[size_t(r) * D + i] * x[size_t(r) * D + i];
    const double inv = 1.0 / std::sqrt(ss / D + 1e-5);
    for (uint32_t i = 0; i < D; ++i) er.add(g[size_t(r) * D + i], w[i] * (x[size_t(r) * D + i] * inv));
  }
  char nm[128];
  snprintf(nm, sizeof nm, "rmsnorm D %u R %d w %s%s%s", D, R, dtn(wdt),
           y_half ? " y half" : "", alias ? " in place" : "");
  report(nm, er, y_half ? 2e-3 : 2e-6);
}

void check_add_swiglu(uint32_t n, int R) {
  auto x = randv(size_t(R) * n), y = randv(size_t(R) * n);
  Buffer bx = fbuf(x.size()), by = fbuf(y.size());
  put(bx, x);
  put(by, y);
  run([&](Encoder& e) { hy4::enc_add(e, n, R, bx, by); });
  Err ea;
  auto g = get(by, y.size());
  for (size_t i = 0; i < y.size(); ++i) ea.add(g[i], float(x[i]) + float(y[i]));
  report("add n " + std::to_string(n) + " R " + std::to_string(R), ea, 1e-7);
  for (float limit : {10.0f, 0.0f})
    for (bool hh : {false, true}) {
      auto gv = randv(size_t(R) * n, 8.0), uv = randv(size_t(R) * n, 8.0);
      Buffer bg = fbuf(gv.size()), bu = fbuf(uv.size()), bh = fbuf(gv.size());
      put(bg, gv);
      put(bu, uv);
      run([&](Encoder& e) { hy4::enc_swiglu(e, n, R, bg, bu, bh, limit, hh); });
      auto gh = get(bh, gv.size(), hh);
      Err es;
      for (size_t i = 0; i < gv.size(); ++i) {
        double gg = float(gv[i]), uu = float(uv[i]);
        if (limit > 0) {  // HYV4FlashExperts._apply_gate
          gg = std::min(gg, double(limit));
          uu = std::clamp(uu, -double(limit), double(limit));
        }
        es.add(gh[i], gg * sigm(gg) * uu);
      }
      char nm[96];
      snprintf(nm, sizeof nm, "swiglu n %u R %d limit %g%s", n, R, limit, hh ? " h half" : "");
      report(nm, es, hh ? 2e-3 : 2e-6);
    }
}

// ---- router ----
void check_router(uint32_t E, uint32_t k, int R) {
  auto lg = randv(size_t(R) * E, 2.0);
  std::vector<double> bias(E);
  for (auto& b : bias) b = uni(-0.1, 0.1);
  Tensor tb = tensor(bias, DType::F32);
  Buffer bl = fbuf(lg.size()), bi = fbuf(size_t(R) * k), bw = fbuf(size_t(R) * k);
  put(bl, lg);
  const float scale = 3.998f;
  run([&](Encoder& e) { hy4::enc_router(e, E, k, scale, R, bl, tb, bi, bw); });
  Err ew;
  size_t idx_bad = 0, near_ties = 0, order_bad = 0;
  for (int r = 0; r < R; ++r) {
    std::vector<double> s(E), ch(E);
    for (uint32_t i = 0; i < E; ++i) {
      s[i] = sigm(float(lg[size_t(r) * E + i]));
      ch[i] = s[i] + bias[i];  // scores_for_choice
    }
    std::vector<uint32_t> ord(E);
    std::iota(ord.begin(), ord.end(), 0u);
    std::stable_sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return ch[a] > ch[b]; });
    if (k < E && ch[ord[k - 1]] - ch[ord[k]] < 1e-6) ++near_ties;
    double den = 1e-20;
    for (uint32_t j = 0; j < k; ++j) den += s[ord[j]];  // weights from the unbiased scores
    std::vector<std::pair<uint32_t, double>> want, got;
    for (uint32_t j = 0; j < k; ++j) {
      want.push_back({ord[j], s[ord[j]] / den * scale});
      got.push_back({bi.as<uint32_t>()[r * k + j], double(bw.as<float>()[r * k + j])});
    }
    // the picks in descending order of s + bias, the lower index first on a tie (beyond fp32's reach: a near tie)
    for (uint32_t j = 1; j < k; ++j) {
      const uint32_t a = std::min(got[j - 1].first, E - 1), b = std::min(got[j].first, E - 1);
      if (ch[b] - ch[a] > 1e-6 || (ch[a] == ch[b] && a > b)) ++order_bad;
    }
    std::sort(want.begin(), want.end());
    std::sort(got.begin(), got.end());
    for (uint32_t j = 0; j < k; ++j) {
      if (want[j].first != got[j].first) ++idx_bad;
      ew.add(got[j].second, want[j].second);
    }
  }
  char nm[96];
  snprintf(nm, sizeof nm, "router E %u k %u R %d: weights (idx mismatches %zu, near ties %zu, out of order %zu)", E,
           k, R, idx_bad, near_ties, order_bad);
  if (idx_bad > near_ties || order_bad) ++failures;
  report(nm, ew, 2e-6);
}

// ---- Engram ----
void check_engram(uint32_t D, int R, DType wdt) {
  const float eps = 1.1920929e-07f;
  auto X = randv(size_t(R) * 4 * D, 1.5), vraw = randv(size_t(R) * D, 2.0), kraw = randv(size_t(4) * R * D, 2.0);
  auto vb = randv(D, 0.2), pg = randv(D, 0.3, 1.0);
  std::vector<double> kb[4], nk[4], nq[4], hs[4];
  Tensor tkb[4], tnk[4], tnq[4], ths[4];
  hy4::EngramW w;
  Tensor tvb = tensor(vb, wdt), tpg = tensor(pg, wdt);
  w.vb = &tvb;
  w.pre_gamma = &tpg;
  for (int s = 0; s < 4; ++s) {
    kb[s] = randv(D, 0.2);
    nk[s] = randv(D, 0.3, 1.0);
    nq[s] = randv(D, 0.3, 1.0);
    hs[s] = randv(D, 0.2);
    tkb[s] = tensor(kb[s], wdt);
    tnk[s] = tensor(nk[s], wdt);
    tnq[s] = tensor(nq[s], wdt);
    ths[s] = tensor(hs[s], wdt);
    w.kb[s] = &tkb[s];
    w.nk[s] = &tnk[s];
    w.nq[s] = &tnq[s];
    w.hc_scale[s] = &ths[s];
  }
  // make one row's gates land near 0 and both signs: correlated keys and queries
  for (uint32_t i = 0; i < D; ++i) kraw[size_t(0) * R * D + i] = X[i] * 3.0 - kb[0][i];
  Buffer bX = fbuf(X.size()), bv = fbuf(vraw.size()), bk = fbuf(kraw.size());
  put(bX, X);
  put(bv, vraw);
  put(bk, kraw);
  run([&](Encoder& e) { hy4::enc_engram(e, w, D, eps, R, bv, bk, bX); });
  auto g = get(bX, X.size());
  Err ex;
  auto rms_inv = [&](const std::vector<double>& v) {
    double ss = 0;
    for (double x : v) ss += x * x;
    return 1.0 / std::sqrt(ss / v.size() + eps);
  };
  for (int r = 0; r < R; ++r) {
    std::vector<double> v(D);
    for (uint32_t i = 0; i < D; ++i) v[i] = float(vraw[size_t(r) * D + i]) + vb[i];
    const double iv = rms_inv(v);
    std::vector<double> main(D);
    for (uint32_t i = 0; i < D; ++i) main[i] = pg[i] * (v[i] * iv);  // unified_pre_gamma * unified_rmsnorm(v)
    for (int s = 0; s < 4; ++s) {
      std::vector<double> k(D), q(D);
      for (uint32_t i = 0; i < D; ++i) {
        k[i] = float(kraw[(size_t(s) * R + r) * D + i]) + kb[s][i];
        q[i] = float(X[(size_t(r) * 4 + s) * D + i]);
      }
      const double ik = rms_inv(k), iq = rms_inv(q);
      double gate = 0;
      for (uint32_t i = 0; i < D; ++i) gate += (nk[s][i] * (k[i] * ik)) * (nq[s][i] * (q[i] * iq));
      gate *= 1.0 / std::sqrt(double(D));
      gate = std::sqrt(std::max(std::fabs(gate), 1e-6)) * (gate > 0 ? 1.0 : gate < 0 ? -1.0 : 0.0);
      gate = sigm(gate);
      for (uint32_t i = 0; i < D; ++i) ex.add(g[(size_t(r) * 4 + s) * D + i], q[i] + gate * main[i] + hs[s][i] * main[i]);
    }
  }
  char nm[96];
  snprintf(nm, sizeof nm, "engram D %u R %d w %s", D, R, dtn(wdt));
  report(nm, ex, 2e-6);
  // the same pass with every third row masked (a pad, image or video token): those rows keep X as it was, the others
  // equal the unmasked pass's bit for bit
  std::vector<uint32_t> m(static_cast<size_t>(R));
  for (int r = 0; r < R; ++r) m[size_t(r)] = r % 3 != 1;
  Buffer bX2 = fbuf(X.size()), bm = fbuf(size_t(R), "check_mask");
  put(bX2, X);
  memcpy(bm.ptr(), m.data(), size_t(R) * 4);
  run([&](Encoder& e) { hy4::enc_engram(e, w, D, eps, R, bv, bk, bX2, &bm); });
  const auto g2 = get(bX2, X.size());
  size_t bad = 0, nmask = 0;
  for (int r = 0; r < R; ++r) {
    nmask += !m[size_t(r)];
    for (size_t i = 0; i < size_t(4) * D; ++i) {
      const size_t j = size_t(r) * 4 * D + i;
      if (g2[j] != (m[size_t(r)] ? g[j] : double(float(X[j])))) ++bad;
    }
  }
  if (bad) ++failures;
  printf("%-58s %zu masked rows, %zu values differ  %s\n", (std::string(nm) + " masked").c_str(), nmask, bad,
         bad ? "FAIL" : "ok");
  fflush(stdout);
}

// ---- attention ----
struct AttnCase {
  uint32_t H, window;
  bool ring;
  uint32_t cap;
  std::vector<std::pair<int, int>> steps;  // (pos0, T)
  bool sinks_bf16 = false;
};

void check_attn(const AttnCase& ac) {
  const uint32_t H = ac.H, HD = 256, KR_LD = 576 + 8, KR_OFF = 512;
  int N = 0;
  for (auto& s : ac.steps) N = std::max(N, s.first + s.second);
  const float theta = ac.window ? 1e4f : 1e7f;
  // inputs for every position
  auto q = randv(size_t(N) * H * HD, 2.0), kvb = randv(size_t(N) * H * 448, 1.0), kr = randv(size_t(N) * KR_LD, 1.0),
       gate = randv(size_t(N) * H * HD, 1.0);
  std::vector<double> sk(H);
  for (auto& v : sk) v = uni(-1.0, 8.0);
  Tensor tsk = tensor(sk, ac.sinks_bf16 ? DType::BF16 : DType::F32);
  hy4::AttnLayer L;
  L.H = H;
  L.theta = theta;
  L.window = ac.window;
  L.sinks = &tsk;
  // CPU: roped q, K, V (fp64; cos/sin as the reference: fp32 angle, bf16 tables)
  std::vector<double> cs(size_t(N) * 32), sn(size_t(N) * 32);
  for (int p = 0; p < N; ++p)
    for (int i = 0; i < 32; ++i) {
      const float invf = 1.0f / std::pow(theta, float(2 * i) / 64.0f);  // 1 / base ** (arange(0, 64, 2) / 64)
      const float ang = float(p) * invf;                                  // position_ids.float() * inv_freq
      cs[size_t(p) * 32 + i] = bf16_round(float(std::cos(double(ang))));  // cos.to(bf16)
      sn[size_t(p) * 32 + i] = bf16_round(float(std::sin(double(ang))));
    }
  auto rope = [&](double* x, int p) {  // apply_rotary_pos_emb: cos[..., :32].repeat_interleave(2), rotate_half
    for (int i = 0; i < 32; ++i) {
      const double c = cs[size_t(p) * 32 + i], s = sn[size_t(p) * 32 + i];
      const double x0 = x[2 * i], x1 = x[2 * i + 1];
      x[2 * i] = x0 * c - x1 * s;
      x[2 * i + 1] = x1 * c + x0 * s;
    }
  };
  std::vector<double> Q(size_t(N) * H * HD), K(size_t(N) * H * HD), V(size_t(N) * H * HD);
  for (int p = 0; p < N; ++p) {
    double kro[64];
    for (int i = 0; i < 64; ++i) kro[i] = float(kr[size_t(p) * KR_LD + KR_OFF + i]);
    rope(kro, p);
    for (uint32_t h = 0; h < H; ++h) {
      double* qr = &Q[(size_t(p) * H + h) * HD];
      double* kk = &K[(size_t(p) * H + h) * HD];
      double* vv = &V[(size_t(p) * H + h) * HD];
      for (uint32_t d = 0; d < HD; ++d) qr[d] = float(q[(size_t(p) * H + h) * HD + d]);
      rope(qr + 192, p);
      for (uint32_t d = 0; d < 192; ++d) kk[d] = float(kvb[(size_t(p) * H + h) * 448 + d]);
      for (int d = 0; d < 64; ++d) kk[192 + d] = kro[d];
      for (uint32_t d = 0; d < HD; ++d) vv[d] = float(kvb[(size_t(p) * H + h) * 448 + 192 + d]);
    }
  }
  Buffer bq = fbuf(q.size()), bkvb = fbuf(kvb.size()), bkr = fbuf(kr.size()), bg = fbuf(gate.size()),
         bo = fbuf(gate.size());
  put(bq, q);
  put(bkvb, kvb);
  put(bkr, kr);
  put(bg, gate);
  int Tmax = 1;
  for (auto& s : ac.steps) Tmax = std::max(Tmax, s.second);
  Buffer scratch = Gpu::get().alloc(hy4::attn_scratch_bytes(H, Tmax), "check_scratch");
  hy4::KvCache kv = hy4::alloc_kv(H, ac.cap, "check_kv", ac.ring);
  int stepi = 0;
  for (auto [pos0, T] : ac.steps) {
    const bool oh = stepi++ == 1;  // one step writes half rows
    const size_t qoff = size_t(pos0) * H * HD * 4;
    run([&](Encoder& e) {
      hy4::enc_attn(e, L, kv, pos0, T, bq.view(qoff, size_t(T) * H * HD * 4),
                    bkvb.view(size_t(pos0) * H * 448 * 4, size_t(T) * H * 448 * 4),
                    bkr.view(size_t(pos0) * KR_LD * 4, size_t(T) * KR_LD * 4), KR_LD, KR_OFF,
                    bg.view(qoff, size_t(T) * H * HD * 4), bo, oh, scratch);
    });
    auto go = get(bo, size_t(T) * H * HD, oh);
    std::vector<Err> errs(H);
    std::vector<std::array<double, 3>> dist(H, {0.0, 0.0, 0.0});  // window - 1, window + 1, no sink
    std::vector<std::thread> th;
    const unsigned nth = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    for (unsigned w = 0; w < nth; ++w)
      th.emplace_back([&, w] {
        std::vector<double> sc, o(HD);
        for (uint32_t h = w; h < H; h += nth)
          for (int t = 0; t < T; ++t) {
            const int p = pos0 + t;
            const int lo = ac.window && p + 1 > int(ac.window) ? p + 1 - int(ac.window) : 0;
            const double* qr = &Q[(size_t(p) * H + h) * HD];
            sc.assign(p + 1 - lo, 0.0);
            double mx = sk[h];  // the sink joins the softmax (eager_attention_forward)
            for (int j = lo; j <= p; ++j) {
              const double* kk = &K[(size_t(j) * H + h) * HD];
              double d = 0;
              for (uint32_t i = 0; i < HD; ++i) d += qr[i] * kk[i];
              sc[j - lo] = d / 16.0;  // scaling = qk_head_dim ** -0.5
              mx = std::max(mx, sc[j - lo]);
            }
            const double esink = std::exp(sk[h] - mx);
            double den = esink;
            std::fill(o.begin(), o.end(), 0.0);
            for (int j = lo; j <= p; ++j) {
              const double e = std::exp(sc[j - lo] - mx);
              den += e;
              const double* vv = &V[(size_t(j) * H + h) * HD];
              for (uint32_t i = 0; i < HD; ++i) o[i] += e * vv[i];
            }
            // the same row with the window one key shorter / longer and without the sink: the GPU's error must
            // stay well below these distances, so an off-by-one window or a dropped sink cannot hide in the tolerance
            const bool shorter = ac.window && p > lo, longer = ac.window && lo > 0;
            double e_lo = 0, e_ext = 0;
            const double* v_lo = &V[(size_t(lo) * H + h) * HD];
            const double* v_ext = longer ? &V[(size_t(lo - 1) * H + h) * HD] : v_lo;
            if (shorter) e_lo = std::exp(sc[0] - mx);
            if (longer) {
              const double* kk = &K[(size_t(lo - 1) * H + h) * HD];
              double d = 0;
              for (uint32_t i = 0; i < HD; ++i) d += qr[i] * kk[i];
              e_ext = std::exp(d / 16.0 - mx);
            }
            for (uint32_t i = 0; i < HD; ++i) {
              const size_t k = (size_t(t) * H + h) * HD + i;
              const double gt = sigm(float(gate[size_t(pos0) * H * HD + k]));
              const double ref = o[i] / den * gt;
              errs[h].add(go[k], ref);
              dist[h][2] = std::max(dist[h][2], std::fabs(o[i] / (den - esink) * gt - ref));
              if (shorter) dist[h][0] = std::max(dist[h][0], std::fabs((o[i] - e_lo * v_lo[i]) / (den - e_lo) * gt - ref));
              if (longer) dist[h][1] = std::max(dist[h][1], std::fabs((o[i] + e_ext * v_ext[i]) / (den + e_ext) * gt - ref));
            }
          }
      });
    for (auto& t : th) t.join();
    Err all;
    double dmax[3] = {0, 0, 0};
    for (uint32_t h = 0; h < H; ++h) {
      all.max_abs = std::max(all.max_abs, errs[h].max_abs);
      all.max_ref = std::max(all.max_ref, errs[h].max_ref);
      all.n += errs[h].n;
      for (int i = 0; i < 3; ++i) dmax[i] = std::max(dmax[i], dist[h][i]);
    }
    char nm[160];
    snprintf(nm, sizeof nm, "attn H %u win %u %s cap %u: pos0 %d T %d%s%s", H, ac.window, ac.ring ? "ring" : "lin",
             ac.cap, pos0, T, oh ? " out half" : "", ac.sinks_bf16 ? " sinks bf16" : "");
    // fp16 K/V rows: ~6e-4 measured; NI_KV_I8's int8 rows: ~1.5% (their own precision, not checked against the alternatives)
    report(nm, all, kv_i8() ? 3e-2 : 4e-3);
    if (kv_i8()) continue;
    // distinguishability: max abs error under a quarter of each alternative's max abs distance (0: not exercised)
    bool dok = true;
    for (int i = 0; i < 3; ++i)
      if (dmax[i] > 0 && all.max_abs > 0.25 * dmax[i]) dok = false;
    if (!dok) ++failures;
    printf("    vs window-1 %.2e  window+1 %.2e  no sink %.2e  %s\n", dmax[0], dmax[1], dmax[2],
           dok ? "ok" : "FAIL (error not well below an alternative's distance)");
  }
  hy4::free_kv(kv);
}


// ---- latent MLA: enc_attn_latent vs fp64 (latent semantics: bf16 latents and roped k_rope) and vs enc_attn ----
void check_attn_latent(const AttnCase& ac) {
  const uint32_t H = ac.H, HD = 256, KR_LD = 576 + 8, KR_OFF = 512, LAT_LD = 576 + 8;
  int N = 0, Tmax = 1;
  for (auto& s : ac.steps) {
    N = std::max(N, s.first + s.second);
    Tmax = std::max(Tmax, s.second);
  }
  const float theta = ac.window ? 1e4f : 1e7f;
  auto q = randv(size_t(N) * H * HD, 2.0), lat = randv(size_t(N) * LAT_LD, 1.0), kr = randv(size_t(N) * KR_LD, 1.0),
       gate = randv(size_t(N) * H * HD, 1.0), wkv = randv(size_t(H) * 448 * 512, 0.05);
  std::vector<double> sk(H);
  for (auto& v : sk) v = uni(-1.0, 8.0);
  Tensor tsk = tensor(sk, ac.sinks_bf16 ? DType::BF16 : DType::F32);
  Tensor tkv = tensor(wkv, DType::BF16);
  tkv.shape = {int64_t(H) * 448, 512};
  hy4::AttnLayer L;
  L.H = H;
  L.theta = theta;
  L.window = ac.window;
  L.sinks = &tsk;
  std::vector<double> cs(size_t(N) * 32), sn(size_t(N) * 32);
  for (int p = 0; p < N; ++p)
    for (int i = 0; i < 32; ++i) {
      const float invf = 1.0f / std::pow(theta, float(2 * i) / 64.0f);
      const float ang = float(p) * invf;
      cs[size_t(p) * 32 + i] = bf16_round(float(std::cos(double(ang))));
      sn[size_t(p) * 32 + i] = bf16_round(float(std::sin(double(ang))));
    }
  auto rope = [&](double* x, int p) {
    for (int i = 0; i < 32; ++i) {
      const double c = cs[size_t(p) * 32 + i], s = sn[size_t(p) * 32 + i];
      const double x0 = x[2 * i], x1 = x[2 * i + 1];
      x[2 * i] = x0 * c - x1 * s;
      x[2 * i + 1] = x1 * c + x0 * s;
    }
  };
  // CPU K, V from the bf16 latents; kvb (fp32 [N][H][448]) feeds enc_attn the same K-nope and V
  std::vector<double> Q(size_t(N) * H * HD), K(size_t(N) * H * HD), V(size_t(N) * H * HD), kvb(size_t(N) * H * 448);
  {
    std::vector<std::thread> th;
    const unsigned nth = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    for (unsigned w = 0; w < nth; ++w)
      th.emplace_back([&, w] {
        for (int p = w; p < N; p += nth) {
          double c[512], kro[64];
          for (int i = 0; i < 512; ++i) c[i] = bf16_round(float(lat[size_t(p) * LAT_LD + i]));
          for (int i = 0; i < 64; ++i) kro[i] = float(kr[size_t(p) * KR_LD + KR_OFF + i]);
          rope(kro, p);
          for (int i = 0; i < 64; ++i) kro[i] = bf16_round(float(kro[i]));
          for (uint32_t h = 0; h < H; ++h) {
            double* qr = &Q[(size_t(p) * H + h) * HD];
            for (uint32_t d = 0; d < HD; ++d) qr[d] = float(q[(size_t(p) * H + h) * HD + d]);
            rope(qr + 192, p);
            for (uint32_t n = 0; n < 448; ++n) {
              const double* wr = &wkv[(size_t(h) * 448 + n) * 512];
              double d = 0;
              for (int i = 0; i < 512; ++i) d += wr[i] * c[i];
              kvb[(size_t(p) * H + h) * 448 + n] = d;
              if (n < 192) K[(size_t(p) * H + h) * HD + n] = d;
              else V[(size_t(p) * H + h) * HD + n - 192] = d;
            }
            for (int i = 0; i < 64; ++i) K[(size_t(p) * H + h) * HD + 192 + i] = kro[i];
          }
        }
      });
    for (auto& t : th) t.join();
  }
  Buffer bq = fbuf(q.size()), blat = fbuf(lat.size()), bkr = fbuf(kr.size()), bg = fbuf(gate.size()),
         bo = fbuf(gate.size()), bo2 = fbuf(gate.size()), bkvb = fbuf(kvb.size());
  put(bq, q);
  put(blat, lat);
  put(bkr, kr);
  put(bg, gate);
  put(bkvb, kvb);
  const uint32_t max_keys = ac.window ? uint32_t(Tmax) + ac.window - 1 : uint32_t(N);
  Buffer scratch = Gpu::get().alloc(hy4::latent_scratch_bytes(H, Tmax, max_keys), "check_scratch");
  Buffer scratch2 = Gpu::get().alloc(hy4::attn_scratch_bytes(H, Tmax), "check_scratch2");
  hy4::LatentCache lc = hy4::alloc_latent(ac.cap, "check_latent", ac.ring);
  hy4::KvCache kv = hy4::alloc_kv(H, ac.ring ? ac.cap : uint32_t(N), "check_kv", ac.ring);
  int stepi = 0;
  for (auto [pos0, T] : ac.steps) {
    const bool oh = stepi++ == 1;
    const size_t qoff = size_t(pos0) * H * HD * 4;
    run([&](Encoder& e) {
      hy4::enc_attn_latent(e, L, lc, pos0, T, bq.view(qoff, size_t(T) * H * HD * 4), H * HD,
                           blat.view(size_t(pos0) * LAT_LD * 4, size_t(T) * LAT_LD * 4), LAT_LD,
                           bkr.view(size_t(pos0) * KR_LD * 4, size_t(T) * KR_LD * 4), KR_LD, KR_OFF, tkv,
                           bg.view(qoff, size_t(T) * H * HD * 4), bo, oh, scratch);
      hy4::enc_attn(e, L, kv, pos0, T, bq.view(qoff, size_t(T) * H * HD * 4),
                    bkvb.view(size_t(pos0) * H * 448 * 4, size_t(T) * H * 448 * 4),
                    bkr.view(size_t(pos0) * KR_LD * 4, size_t(T) * KR_LD * 4), KR_LD, KR_OFF,
                    bg.view(qoff, size_t(T) * H * HD * 4), bo2, oh, scratch2);
    });
    auto go = get(bo, size_t(T) * H * HD, oh), go2 = get(bo2, size_t(T) * H * HD, oh);
    std::vector<Err> errs(H), errs2(H);
    std::vector<std::array<double, 3>> dist(H, {0.0, 0.0, 0.0});  // window - 1, window + 1, no sink
    std::vector<std::thread> th;
    const unsigned nth = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    for (unsigned w = 0; w < nth; ++w)
      th.emplace_back([&, w] {
        std::vector<double> sc, o(HD);
        for (uint32_t h = w; h < H; h += nth)
          for (int t = 0; t < T; ++t) {
            const int p = pos0 + t;
            const int lo = ac.window && p + 1 > int(ac.window) ? p + 1 - int(ac.window) : 0;
            const double* qr = &Q[(size_t(p) * H + h) * HD];
            sc.assign(p + 1 - lo, 0.0);
            double mx = sk[h];
            for (int j = lo; j <= p; ++j) {
              const double* kk = &K[(size_t(j) * H + h) * HD];
              double d = 0;
              for (uint32_t i = 0; i < HD; ++i) d += qr[i] * kk[i];
              sc[j - lo] = d / 16.0;
              mx = std::max(mx, sc[j - lo]);
            }
            const double esink = std::exp(sk[h] - mx);
            double den = esink;
            std::fill(o.begin(), o.end(), 0.0);
            for (int j = lo; j <= p; ++j) {
              const double e = std::exp(sc[j - lo] - mx);
              den += e;
              const double* vv = &V[(size_t(j) * H + h) * HD];
              for (uint32_t i = 0; i < HD; ++i) o[i] += e * vv[i];
            }
            const bool shorter = ac.window && p > lo, longer = ac.window && lo > 0;
            double e_lo = 0, e_ext = 0;
            const double* v_lo = &V[(size_t(lo) * H + h) * HD];
            const double* v_ext = longer ? &V[(size_t(lo - 1) * H + h) * HD] : v_lo;
            if (shorter) e_lo = std::exp(sc[0] - mx);
            if (longer) {
              const double* kk = &K[(size_t(lo - 1) * H + h) * HD];
              double d = 0;
              for (uint32_t i = 0; i < HD; ++i) d += qr[i] * kk[i];
              e_ext = std::exp(d / 16.0 - mx);
            }
            for (uint32_t i = 0; i < HD; ++i) {
              const size_t k = (size_t(t) * H + h) * HD + i;
              const double gt = sigm(float(gate[size_t(pos0) * H * HD + k]));
              const double ref = o[i] / den * gt;
              errs[h].add(go[k], ref);
              errs2[h].add(go[k], go2[k]);
              dist[h][2] = std::max(dist[h][2], std::fabs(o[i] / (den - esink) * gt - ref));
              if (shorter) dist[h][0] = std::max(dist[h][0], std::fabs((o[i] - e_lo * v_lo[i]) / (den - e_lo) * gt - ref));
              if (longer) dist[h][1] = std::max(dist[h][1], std::fabs((o[i] + e_ext * v_ext[i]) / (den + e_ext) * gt - ref));
            }
          }
      });
    for (auto& t : th) t.join();
    Err all, vs;
    double dmax[3] = {0, 0, 0};
    for (uint32_t h = 0; h < H; ++h) {
      for (int i = 0; i < 3; ++i) dmax[i] = std::max(dmax[i], dist[h][i]);
      all.max_abs = std::max(all.max_abs, errs[h].max_abs);
      all.max_ref = std::max(all.max_ref, errs[h].max_ref);
      all.n += errs[h].n;
      vs.max_abs = std::max(vs.max_abs, errs2[h].max_abs);
      vs.max_ref = std::max(vs.max_ref, errs2[h].max_ref);
      vs.n += errs2[h].n;
    }
    char nm[160];
    snprintf(nm, sizeof nm, "latent H %u win %u %s cap %u: pos0 %d T %d%s%s", H, ac.window, ac.ring ? "ring" : "lin",
             ac.cap, pos0, T, oh ? " out half" : "", ac.sinks_bf16 ? " sinks bf16" : "");
    report(nm, all, kv_i8() && T > 16 ? 3e-2 : 4e-3);
    printf("    vs enc_attn (expanded, fp16 K/V): max_abs %.3e  rel %.3e\n", vs.max_abs, vs.rel());
    if (kv_i8() && T > 16) continue;
    bool dok = true;
    for (int i = 0; i < 3; ++i)
      if (dmax[i] > 0 && all.max_abs > 0.25 * dmax[i]) dok = false;
    if (!dok) ++failures;
    printf("    vs window-1 %.2e  window+1 %.2e  no sink %.2e  %s\n", dmax[0], dmax[1], dmax[2],
           dok ? "ok" : "FAIL (error not well below an alternative's distance)");
  }
  hy4::free_latent(lc);
  hy4::free_kv(kv);
}


// ---- compressed indexer: caches, selection and selected attention vs fp64 (modeling_hy_v4_flash.py semantics) ----
template <typename F>
void par_for(int n, F f) {
  std::vector<std::thread> th;
  const int nth = int(std::max(1u, std::min(16u, std::thread::hardware_concurrency())));
  for (int w = 0; w < nth; ++w)
    th.emplace_back([&, w] {
      for (int i = w; i < n; i += nth) f(i);
    });
  for (auto& t : th) t.join();
}
// top-511 of lg[0, G) (ties: lower g first) plus G, ascending
std::vector<uint32_t> ix_select_ref(const double* lg, uint32_t G) {
  std::vector<uint32_t> r;
  if (G <= 511) {
    for (uint32_t g = 0; g <= G; ++g) r.push_back(g);
    return r;
  }
  std::vector<uint32_t> idx(G);
  for (uint32_t g = 0; g < G; ++g) idx[g] = g;
  std::partial_sort(idx.begin(), idx.begin() + 511, idx.end(),
                    [&](uint32_t a, uint32_t b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
  r.assign(idx.begin(), idx.begin() + 511);
  std::sort(r.begin(), r.end());
  r.push_back(G);
  return r;
}

void check_indexer(uint32_t H, std::vector<std::pair<int, int>> steps, int row_stride) {
  const uint32_t HD = 256, LAT_LD = 576 + 8, KR_OFF = 512, TK_LD = 128 + 16, QI_LD = 2048 + 8, HW_LD = 16 + 4;
  int N = 0, Tmax = 1;
  for (auto& st : steps) {
    N = std::max(N, st.first + st.second);
    Tmax = std::max(Tmax, st.second);
  }
  const float theta = 1e7f;
  // inputs (raw projection outputs) and weights
  auto q = randv(size_t(N) * H * HD, 2.0), lat = randv(size_t(N) * LAT_LD, 1.0), gate = randv(size_t(N) * H * HD, 1.0),
       wkv = randv(size_t(H) * 448 * 512, 0.05), tk = randv(size_t(N) * TK_LD, 1.0), qi = randv(size_t(N) * QI_LD, 1.0),
       hw = randv(size_t(N) * HW_LD, 1.0), tkn = randv(128, 0.2, 1.0), wv = randv(128 * 128, 0.09),
       wg = randv(128 * 128, 0.09), ape = randv(4 * 128, 0.5), kn = randv(128, 0.2, 1.0);
  // a low-dimensional component shared by keys and queries, so some groups score high for every query (as in a model)
  {
    auto dir = randv(128, 1.0);
    for (int p2 = 0; p2 < N; ++p2) {
      const double a = uni(-1.0, 3.0);
      for (int d = 0; d < 128; ++d) tk[size_t(p2) * TK_LD + d] += a * dir[d];
      for (int h = 0; h < 16; ++h)
        for (int d = 0; d < 64; ++d) qi[size_t(p2) * QI_LD + h * 128 + d] += 0.5 * dir[d];
    }
  }
  std::vector<double> sk(H);
  for (auto& v : sk) v = uni(-1.0, 8.0);
  Tensor tsk = tensor(sk, DType::F32), tkv = tensor(wkv, DType::BF16), ttkn = tensor(tkn, DType::BF16),
         twv = tensor(wv, DType::F32), twg = tensor(wg, DType::F32), tape = tensor(ape, DType::F32),
         tkn2 = tensor(kn, DType::F32);
  tkv.shape = {int64_t(H) * 448, 512};
  hy4::AttnLayer L;
  L.H = H;
  L.theta = theta;
  L.sinks = &tsk;
  hy4::IndexerWeights W;
  W.tk_norm = &ttkn;
  W.wv = &twv;
  W.wg = &twg;
  W.ape = &tape;
  W.key_norm = &tkn2;
  auto bf = [](double x) { return double(bf16_round(float(x))); };
  std::vector<double> cs(size_t(N) * 32), sn(size_t(N) * 32);
  for (int p2 = 0; p2 < N; ++p2)
    for (int i = 0; i < 32; ++i) {
      const float invf = 1.0f / std::pow(theta, float(2 * i) / 64.0f);
      const float ang = float(p2) * invf;
      cs[size_t(p2) * 32 + i] = bf16_round(std::cos(ang));
      sn[size_t(p2) * 32 + i] = bf16_round(std::sin(ang));
    }
  auto rope = [&](double* x, int p2) {  // 64 dims, bf16 cos/sin
    for (int i = 0; i < 32; ++i) {
      const double c = cs[size_t(p2) * 32 + i], s2 = sn[size_t(p2) * 32 + i];
      const double x0 = x[2 * i], x1 = x[2 * i + 1];
      x[2 * i] = x0 * c - x1 * s2;
      x[2 * i + 1] = x1 * c + x0 * s2;
    }
  };
  // CPU token keys, group keys, queries, head weights
  const int NG = N / 4;
  std::vector<double> tok(size_t(N) * 128), grp(size_t(NG) * 128), qx(size_t(N) * 16 * 128), hwx(size_t(N) * 16);
  par_for(N, [&](int p2) {
    double x[128], ss = 0;
    for (int d = 0; d < 128; ++d) {
      x[d] = bf(tk[size_t(p2) * TK_LD + d]);
      ss += x[d] * x[d];
    }
    const double r = 1.0 / std::sqrt(ss / 128 + 1e-5);
    for (int d = 0; d < 128; ++d) tok[size_t(p2) * 128 + d] = bf(bf(tkn[d]) * bf(x[d] * r));
    for (int h = 0; h < 16; ++h) {
      double* qr = &qx[(size_t(p2) * 16 + h) * 128];
      for (int d = 0; d < 128; ++d) qr[d] = bf(qi[size_t(p2) * QI_LD + h * 128 + d]);
      rope(qr + 64, p2);
      for (int d = 64; d < 128; ++d) qr[d] = bf(qr[d]);
      hwx[size_t(p2) * 16 + h] = bf(hw[size_t(p2) * HW_LD + h]) * double(0.02209708691207961f);
    }
  });
  par_for(NG, [&](int g) {
    double pv[128], ss = 0;
    for (int d = 0; d < 128; ++d) {
      double v[4], a[4], m = -1e300;
      for (int i = 0; i < 4; ++i) {
        const double* x = &tok[size_t(g * 4 + i) * 128];
        double sv = 0, sa = 0;
        for (int k = 0; k < 128; ++k) {
          sv += double(float(wv[d * 128 + k])) * x[k];
          sa += double(float(wg[d * 128 + k])) * x[k];
        }
        v[i] = sv;
        a[i] = sa + double(float(ape[i * 128 + d]));
        m = std::max(m, a[i]);
      }
      double es = 0, o = 0;
      for (int i = 0; i < 4; ++i) es += (a[i] = std::exp(a[i] - m));
      for (int i = 0; i < 4; ++i) o += v[i] * a[i] / es;
      pv[d] = o;
      ss += o * o;
    }
    const double r = 1.0 / std::sqrt(ss / 128 + 1e-5);
    double* k = &grp[size_t(g) * 128];
    for (int d = 0; d < 128; ++d) k[d] = bf(double(float(kn[d])) * pv[d] * r);
    rope(k + 64, g * 4);
    for (int d = 64; d < 128; ++d) k[d] = bf(k[d]);
  });
  // logits of query row qrow (of qs / hs) at position p2 against group keys G_keys
  auto logits_of = [&](const std::vector<double>& G_keys, const double* qs, const double* hs, int p2,
                       std::vector<double>& lg) {
    const uint32_t G = uint32_t(p2) / 4;
    lg.assign(G, 0.0);
    for (uint32_t g = 0; g < G; ++g) {
      double l = 0;
      for (int h = 0; h < 16; ++h) {
        const double* qr = qs + size_t(h) * 128;
        const double* k = &G_keys[size_t(g) * 128];
        double d = 0;
        for (int i = 0; i < 128; ++i) d += qr[i] * k[i];
        l += hs[h] * std::max(d, 0.0);
      }
      lg[g] = l;
    }
  };
  // CPU Q (roped), K, V per head from the bf16 latents
  std::vector<double> Q(size_t(N) * H * HD), K(size_t(N) * H * HD), V(size_t(N) * H * HD);
  par_for(N, [&](int p2) {
    double c[512], kro[64];
    for (int i = 0; i < 512; ++i) c[i] = bf(lat[size_t(p2) * LAT_LD + i]);
    for (int i = 0; i < 64; ++i) kro[i] = float(lat[size_t(p2) * LAT_LD + KR_OFF + i]);
    rope(kro, p2);
    for (int i = 0; i < 64; ++i) kro[i] = bf(kro[i]);
    for (uint32_t h = 0; h < H; ++h) {
      double* qr = &Q[(size_t(p2) * H + h) * HD];
      for (uint32_t d = 0; d < HD; ++d) qr[d] = float(q[(size_t(p2) * H + h) * HD + d]);
      rope(qr + 192, p2);
      for (uint32_t n = 0; n < 448; ++n) {
        const double* wr = &wkv[(size_t(h) * 448 + n) * 512];
        double d = 0;
        for (int i = 0; i < 512; ++i) d += wr[i] * c[i];
        if (n < 192) K[(size_t(p2) * H + h) * HD + n] = d;
        else V[(size_t(p2) * H + h) * HD + n - 192] = d;
      }
      for (int i = 0; i < 64; ++i) K[(size_t(p2) * H + h) * HD + 192 + i] = kro[i];
    }
  });
  Buffer bq = fbuf(q.size()), blat = fbuf(lat.size()), bg = fbuf(gate.size()), bo = fbuf(gate.size()),
         btk = fbuf(tk.size()), bqi = fbuf(qi.size()), bhw = fbuf(hw.size());
  put(bq, q);
  put(blat, lat);
  put(bg, gate);
  put(btk, tk);
  put(bqi, qi);
  put(bhw, hw);
  Buffer bjunk = fbuf(size_t(16) * TK_LD);
  put(bjunk, randv(size_t(16) * TK_LD, 3.0));
  Gpu& gpu = Gpu::get();
  Buffer scratch = gpu.alloc(hy4::latent_scratch_bytes(H, Tmax, uint32_t(N)), "check_scratch");
  Buffer iscr = gpu.alloc(hy4::index_scratch_bytes(Tmax, uint32_t(N)), "check_iscratch");
  Buffer bsel = gpu.alloc(size_t(Tmax) * hy4::kIndexGroups * 4, "check_sel");
  hy4::LatentCache lc = hy4::alloc_latent(uint32_t(N), "check_latent");
  hy4::IndexerCache ic = hy4::alloc_indexer(16, "check_indexer");  // grown on the way, as the model would
  int stepi = 0;
  for (auto [pos0, T] : steps) {
    const bool oh = stepi++ == 2;
    const bool use_sel = pos0 + T > int(hy4::kIndexDense);
    if (uint32_t(pos0 + T) > ic.cap) hy4::grow_indexer(ic, uint32_t(pos0 + T) + 100, uint32_t(pos0));
    const size_t qoff = size_t(pos0) * H * HD * 4;
    if (T <= 16)  // a rejected draft block first left other token keys (and pooled groups) at these positions
      run([&](Encoder& e) { hy4::enc_index_write(e, L, W, ic, pos0, T, bjunk, TK_LD, 0); });
    run([&](Encoder& e) {
      hy4::enc_index_write(e, L, W, ic, pos0, T, btk.view(size_t(pos0) * TK_LD * 4, size_t(T) * TK_LD * 4), TK_LD, 0);
      if (use_sel)
        hy4::enc_index_select(e, L, ic, pos0, T, bqi.view(size_t(pos0) * QI_LD * 4, size_t(T) * QI_LD * 4), QI_LD, 0,
                              bhw.view(size_t(pos0) * HW_LD * 4, size_t(T) * HW_LD * 4), HW_LD, 0, bsel, iscr);
      hy4::enc_attn_latent(e, L, lc, pos0, T, bq.view(qoff, size_t(T) * H * HD * 4), H * HD,
                           blat.view(size_t(pos0) * LAT_LD * 4, size_t(T) * LAT_LD * 4), LAT_LD,
                           blat.view(size_t(pos0) * LAT_LD * 4, size_t(T) * LAT_LD * 4), LAT_LD, KR_OFF, tkv,
                           bg.view(qoff, size_t(T) * H * HD * 4), bo, oh, scratch, use_sel ? &bsel : nullptr);
    });
    char nm[160];
    // caches: every position written so far (GPU values are bf16 values held in half)
    {
      const int n_tok = pos0 + T, n_grp = (pos0 + T) / 4;
      auto gt = get(ic.tok, size_t(n_tok) * 128, true), gg = get(ic.grp, size_t(n_grp) * 128, true);
      Err et, eg;
      size_t dt_ = 0, dg_ = 0;
      for (size_t i = 0; i < size_t(n_tok) * 128; ++i) {
        et.add(gt[i], tok[i]);
        dt_ += gt[i] != tok[i];
      }
      for (size_t i = 0; i < size_t(n_grp) * 128; ++i) {
        eg.add(gg[i], grp[i]);
        dg_ += gg[i] != grp[i];
      }
      snprintf(nm, sizeof nm, "indexer token keys (%d positions, %zu values not the same bf16)", n_tok, dt_);
      report(nm, et, 2e-2);
      snprintf(nm, sizeof nm, "indexer group keys (%d groups, %zu values not the same bf16)", n_grp, dg_);
      report(nm, eg, 2e-2);
    }
    std::vector<uint32_t> gsel;
    std::vector<std::vector<uint32_t>> rsel(T);  // the selection attention is checked with (GPU's, or dense)
    if (use_sel) {
      gsel.resize(size_t(T) * 512);
      memcpy(gsel.data(), bsel.ptr(), gsel.size() * 4);
      auto gg = get(ic.grp, size_t((pos0 + T) / 4) * 128, true);  // the groups written so far
      // the GPU's queries and head weights (index scratch: qx half [T][16][128] | hwx fp32 [T][16])
      auto gq = get(iscr, size_t(T) * 16 * 128, true);
      std::vector<double> ghw(size_t(T) * 16);
      {
        const size_t o_hw = (size_t(T) * 16 * 128 * 2 + 255) / 256 * 256;
        const float* f = (const float*)((const char*)iscr.ptr() + o_hw);
        for (size_t i = 0; i < ghw.size(); ++i) ghw[i] = f[i];
      }
      // (a) GPU selection vs fp64 selection from the GPU's own group keys: only near-ties may differ
      // (b) vs fp64 selection from the CPU's keys: informational (bf16 flips in the keys move logits)
      std::vector<int> bad_a(T, 0), miss_a(T, 0), miss_b(T, 0), sel_rows(T, 0);
      std::vector<double> gap_a(T, 0.0), gap_b(T, 0.0);
      par_for(T, [&](int t) {
        const int p2 = pos0 + t;
        const uint32_t G = uint32_t(p2) / 4, n = std::min(G, 511u) + 1;
        rsel[t].assign(gsel.begin() + size_t(t) * 512, gsel.begin() + size_t(t) * 512 + n);
        if (G <= 511) {
          for (uint32_t i = 0; i < n; ++i) bad_a[t] += rsel[t][i] != i;
          return;
        }
        sel_rows[t] = 1;
        for (int which = 0; which < 2; ++which) {
          std::vector<double> lg;
          if (which) logits_of(grp, &qx[size_t(p2) * 16 * 128], &hwx[size_t(p2) * 16], p2, lg);
          else logits_of(gg, &gq[size_t(t) * 16 * 128], &ghw[size_t(t) * 16], p2, lg);
          auto ref = ix_select_ref(lg.data(), G);
          double scale = 0;
          for (double v : lg) scale = std::max(scale, std::fabs(v));
          std::vector<double> sorted(lg);
          std::nth_element(sorted.begin(), sorted.begin() + 510, sorted.end(), std::greater<double>());
          const double thr = sorted[510];
          std::vector<uint32_t> diff;
          std::set_symmetric_difference(ref.begin(), ref.end(), rsel[t].begin(), rsel[t].end(), std::back_inserter(diff));
          for (uint32_t g : diff) {
            const double gap = g < G ? std::fabs(lg[g] - thr) / scale : 1.0;
            (which ? gap_b : gap_a)[t] = std::max((which ? gap_b : gap_a)[t], gap);
          }
          (which ? miss_b : miss_a)[t] = int(diff.size()) / 2;
          if (!std::is_sorted(rsel[t].begin(), rsel[t].end()) || rsel[t].back() != G) bad_a[t]++;
        }
      });
      int rows = 0, bad = 0, ma = 0, mb = 0;
      double ga = 0, gb = 0;
      for (int t = 0; t < T; ++t) {
        rows += sel_rows[t];
        bad += bad_a[t];
        ma += miss_a[t];
        mb += miss_b[t];
        ga = std::max(ga, gap_a[t]);
        gb = std::max(gb, gap_b[t]);
      }
      // (a) is the kernels' own error (fp32 sums in another order): near-ties only. (b) adds the bf16 roundings
      // the GPU and the fp64 reference can land on either side of (a key or query value one bf16 step apart).
      const bool ok = bad == 0 && ga < 1e-5 && gb < 2e-3;
      if (!ok) ++failures;
      printf("indexer select pos0 %d T %d: %d rows past 2048; vs fp64 on the GPU's keys and queries: %d groups swapped (max gap "
             "%.1e of the row's scale)%s; vs fp64 on fp64 keys and queries: %d swapped (max gap %.1e)  %s\n",
             pos0, T, rows, ma, ga, bad ? ", ORDER/DENSE ROWS WRONG" : "", mb, gb, ok ? "ok" : "FAIL");
    } else {
      for (int t = 0; t < T; ++t) {
        const uint32_t G = uint32_t(pos0 + t) / 4;
        for (uint32_t g = 0; g <= G; ++g) rsel[t].push_back(g);
      }
    }
    // attention vs fp64 over rsel; distance to dense attention shows the selection is applied
    auto go = get(bo, size_t(T) * H * HD, oh);
    std::vector<Err> errs(T);
    std::vector<double> dden(T, 0.0);
    std::vector<int> rows;
    for (int t = 0; t < T; t += (T > 64 ? row_stride : 1)) rows.push_back(t);
    if (rows.back() != T - 1) rows.push_back(T - 1);
    par_for(int(rows.size()), [&](int ri) {
      const int t = rows[ri], p2 = pos0 + t;
      std::vector<int> keys;
      for (uint32_t g : rsel[t])
        for (int i = 0; i < 4; ++i)
          if (int(g) * 4 + i <= p2) keys.push_back(int(g) * 4 + i);
      std::vector<double> sc, o(HD), od(HD);
      for (uint32_t h = 0; h < H; ++h) {
        const double* qr = &Q[(size_t(p2) * H + h) * HD];
        auto attend = [&](const std::vector<int>& ks, std::vector<double>& out) {
          sc.resize(ks.size());
          double mx = sk[h];
          for (size_t j = 0; j < ks.size(); ++j) {
            const double* kk = &K[(size_t(ks[j]) * H + h) * HD];
            double d = 0;
            for (uint32_t i = 0; i < HD; ++i) d += qr[i] * kk[i];
            sc[j] = d / 16.0;
            mx = std::max(mx, sc[j]);
          }
          double den = std::exp(sk[h] - mx);
          std::fill(out.begin(), out.end(), 0.0);
          for (size_t j = 0; j < ks.size(); ++j) {
            const double e2 = std::exp(sc[j] - mx);
            den += e2;
            const double* vv = &V[(size_t(ks[j]) * H + h) * HD];
            for (uint32_t i = 0; i < HD; ++i) out[i] += e2 * vv[i];
          }
          for (auto& x : out) x /= den;
        };
        attend(keys, o);
        const bool sparse = int(keys.size()) < p2 + 1;
        if (sparse) {
          std::vector<int> all(p2 + 1);
          for (int j = 0; j <= p2; ++j) all[j] = j;
          attend(all, od);
        }
        for (uint32_t i = 0; i < HD; ++i) {
          const size_t k = (size_t(t) * H + h) * HD + i;
          const double gt = sigm(float(gate[size_t(pos0) * H * HD + k]));
          errs[t].add(go[k], o[i] * gt);
          if (sparse) dden[t] = std::max(dden[t], std::fabs(od[i] - o[i]) * gt);
        }
      }
    });
    Err all;
    double dd = 0;
    for (int t : rows) {
      all.max_abs = std::max(all.max_abs, errs[t].max_abs);
      all.max_ref = std::max(all.max_ref, errs[t].max_ref);
      all.n += errs[t].n;
      dd = std::max(dd, dden[t]);
    }
    snprintf(nm, sizeof nm, "indexed attn H %u: pos0 %d T %d%s%s", H, pos0, T, use_sel ? " sel" : " dense",
             oh ? " out half" : "");
    report(nm, all, kv_i8() && T > 16 && !use_sel ? 3e-2 : 4e-3);  // int8 K/V rows on the expanded prefill
    if (use_sel) {
      const bool dok = all.max_abs < 0.25 * dd;
      if (!dok) ++failures;
      printf("    vs dense attention %.2e  %s\n", dd, dok ? "ok" : "FAIL (error not well below the dense distance)");
    }
    if (use_sel && T == 6) {  // a drafter block after this verify: 3 drafts accepted, source row 3, a = pos0 + 4
      const int acc = 3, a = pos0 + acc + 1, TD = 6;
      Buffer bidx = gpu.alloc(4, "check_idx"), brow = gpu.alloc(hy4::kIndexGroups * 4, "check_row");
      *(int*)bidx.ptr() = acc;
      const size_t qa = size_t(a) * H * HD * 4;
      run([&](Encoder& e) {
        hy4::enc_index_pick(e, bsel, T, &bidx, 0, brow);
        hy4::enc_attn_latent(e, L, lc, a, TD, bq.view(qa, size_t(TD) * H * HD * 4), H * HD,
                             blat.view(size_t(a) * LAT_LD * 4, size_t(TD) * LAT_LD * 4), LAT_LD,
                             blat.view(size_t(a) * LAT_LD * 4, size_t(TD) * LAT_LD * 4), LAT_LD, KR_OFF, tkv,
                             bg.view(qa, size_t(TD) * H * HD * 4), bo, false, scratch, &brow, true);
      });
      const auto& srow = rsel[acc];
      bool pick_ok = memcmp(brow.ptr(), srow.data(), srow.size() * 4) == 0;
      auto gd = get(bo, size_t(TD) * H * HD);
      Err ed;
      double dd2 = 0;
      for (int t = 0; t < TD; ++t) {
        std::vector<int> keys;  // the spec's key set
        for (uint32_t g : srow)
          for (int i = 0; i < 4; ++i)
            if (int(g) * 4 + i < a) keys.push_back(int(g) * 4 + i);
        for (int j = a; j <= a + t; ++j) keys.push_back(j);
        std::vector<int> all(a + t + 1);
        for (int j = 0; j <= a + t; ++j) all[j] = j;
        for (uint32_t h = 0; h < H; ++h) {
          const double* qr = &Q[(size_t(a + t) * H + h) * HD];
          auto attend = [&](const std::vector<int>& ks, std::vector<double>& out) {
            std::vector<double> sc(ks.size());
            double mx = sk[h];
            for (size_t j = 0; j < ks.size(); ++j) {
              const double* kk = &K[(size_t(ks[j]) * H + h) * HD];
              double d = 0;
              for (uint32_t i = 0; i < HD; ++i) d += qr[i] * kk[i];
              mx = std::max(mx, sc[j] = d / 16.0);
            }
            double den = std::exp(sk[h] - mx);
            out.assign(HD, 0.0);
            for (size_t j = 0; j < ks.size(); ++j) {
              const double e2 = std::exp(sc[j] - mx);
              den += e2;
              for (uint32_t i = 0; i < HD; ++i) out[i] += e2 * V[(size_t(ks[j]) * H + h) * HD + i];
            }
            for (auto& x : out) x /= den;
          };
          std::vector<double> o, od;
          attend(keys, o);
          attend(all, od);
          for (uint32_t i = 0; i < HD; ++i) {
            const size_t k = (size_t(t) * H + h) * HD + i;
            const double gt = sigm(float(gate[size_t(a) * H * HD + k]));
            ed.add(gd[k], o[i] * gt);
            dd2 = std::max(dd2, std::fabs(od[i] - o[i]) * gt);
          }
        }
      }
      snprintf(nm, sizeof nm, "drafter block H %u: a %d T %d, source row %d's selection%s", H, a, TD, acc,
               pick_ok ? "" : " (PICKED ROW WRONG)");
      if (!pick_ok) ++failures;
      report(nm, ed, 4e-3);
      const bool dok = ed.max_abs < 0.25 * dd2;
      if (!dok) ++failures;
      printf("    vs dense attention %.2e  %s\n", dd2, dok ? "ok" : "FAIL (error not well below the dense distance)");
    }
  }
  hy4::free_latent(lc);
  hy4::free_indexer(ic);
}

// ---- timings ----
void bench() {
  Gpu& g = Gpu::get();
  for (uint32_t H : {64u, 32u}) {  // decode over 2048 keys (window 0: all of them)
    const uint32_t cap = 2048;
    hy4::KvCache kv = hy4::alloc_kv(H, cap, "bench_kv");
    std::vector<double> sk(H, 0.5);
    Tensor tsk = tensor(sk, DType::F32);
    hy4::AttnLayer L;
    L.H = H;
    L.sinks = &tsk;
    Buffer q = fbuf(size_t(H) * 256), kvb = fbuf(size_t(H) * 448), kr = fbuf(576), gate = fbuf(size_t(H) * 256),
           out = fbuf(size_t(H) * 256);
    Buffer scratch = g.alloc(hy4::attn_scratch_bytes(H, 1), "bench_scratch");
    const int iters = 200;
    for (int rep = 0; rep < 3; ++rep) {
      auto e = g.begin();
      for (int i = 0; i < iters; ++i) hy4::enc_attn(*e, L, kv, cap - 1, 1, q, kvb, kr, 576, 512, gate, out, false, scratch);
      e->commit_and_wait();
      const double us = e->gpu_ms() * 1000 / iters;
      const double gb = 2.0 * H * cap * kv_row_bytes(256) / (us * 1e-6) / 1e9;
      printf("decode H %u over %u keys: %.1f us per call (prep + split-K + reduce), %.0f GB/s of K/V\n", H, cap, us, gb);
    }
    for (int T : {6, 16}) {  // verify-sized: enc_attn's prefill kernel
      Buffer q2 = fbuf(size_t(T) * H * 256), kvb2 = fbuf(size_t(T) * H * 448), kr2 = fbuf(size_t(T) * 576),
             g2 = fbuf(size_t(T) * H * 256), o2 = fbuf(size_t(T) * H * 256);
      Buffer s2 = g.alloc(hy4::attn_scratch_bytes(H, T), "bench_scratch");
      auto e = g.begin();
      for (int i = 0; i < iters; ++i)
        hy4::enc_attn(*e, L, kv, cap - T, T, q2, kvb2, kr2, 576, 512, g2, o2, false, s2);
      e->commit_and_wait();
      printf("decode H %u T %d over %u keys: %.1f us per call (enc_attn, expanded)\n", H, T, cap,
             e->gpu_ms() * 1000 / iters);
    }
    hy4::free_kv(kv);
  }
  for (uint32_t ctx : {8192u, 32768u, 131072u}) {  // the compressed indexer and selected attention, H 32
    const uint32_t H = 32;
    hy4::IndexerCache ic = hy4::alloc_indexer(ctx + 512, "bench_indexer");
    {  // random group keys (bf16 values in half) so the select sees a spread of logits
      __fp16* gp = (__fp16*)ic.grp.ptr();
      for (size_t i = 0; i < size_t(ic.cap / 4) * 128; ++i) gp[i] = __fp16(bf16_round(float(uni(-1.0, 1.0))));
    }
    hy4::LatentCache lc = hy4::alloc_latent(ctx + 512, "bench_latent");
    std::vector<double> sk(H, 0.5), wkv = randv(size_t(H) * 448 * 512, 0.05), tkn = randv(128, 0.1, 1.0),
                        w128 = randv(128 * 128, 0.09), ape = randv(512, 0.5), kn = randv(128, 0.1, 1.0);
    Tensor tsk = tensor(sk, DType::F32), tkv = tensor(wkv, DType::BF16), ttkn = tensor(tkn, DType::BF16),
           twv = tensor(w128, DType::F32), twg = tensor(w128, DType::F32), tape = tensor(ape, DType::F32),
           tkn2 = tensor(kn, DType::F32);
    hy4::AttnLayer L;
    L.H = H;
    L.sinks = &tsk;
    hy4::IndexerWeights W{&ttkn, &twv, &twg, &tape, &tkn2};
    for (int T : {1, 6, 16, 512}) {
      const int pos0 = int(ctx) - T;
      Buffer qi = fbuf(size_t(T) * 2048), hw = fbuf(size_t(T) * 16), tk = fbuf(size_t(T) * 128);
      put(qi, randv(size_t(T) * 2048, 1.0));
      put(hw, randv(size_t(T) * 16, 1.0));
      put(tk, randv(size_t(T) * 128, 1.0));
      Buffer q = fbuf(size_t(T) * H * 256), lat = fbuf(size_t(T) * 512), kr = fbuf(size_t(T) * 64),
             gate = fbuf(size_t(T) * H * 256), out = fbuf(size_t(T) * H * 256);
      Buffer sel = g.alloc(size_t(T) * hy4::kIndexGroups * 4, "bench_sel");
      Buffer iscr = g.alloc(hy4::index_scratch_bytes(T, ctx), "bench_iscratch");
      Buffer scr = g.alloc(hy4::latent_scratch_bytes(H, T, 0), "bench_scratch");
      const int iters = T >= 512 ? 5 : 50;
      double ms[3] = {0, 0, 0};
      for (int rep = 0; rep < 2; ++rep) {
        auto e = g.begin();
        for (int i = 0; i < iters; ++i) hy4::enc_index_write(*e, L, W, ic, pos0, T, tk, 128, 0);
        e->commit_and_wait();
        ms[0] = e->gpu_ms() / iters;
        e = g.begin();
        for (int i = 0; i < iters; ++i) hy4::enc_index_select(*e, L, ic, pos0, T, qi, 2048, 0, hw, 16, 0, sel, iscr);
        e->commit_and_wait();
        ms[1] = e->gpu_ms() / iters;
        e = g.begin();
        for (int i = 0; i < iters; ++i)
          hy4::enc_attn_latent(*e, L, lc, pos0, T, q, H * 256, lat, 512, kr, 64, 0, tkv, gate, out, false, scr, &sel);
        e->commit_and_wait();
        ms[2] = e->gpu_ms() / iters;
      }
      printf("indexer ctx %u T %d: write %.1f us, select %.1f us, selected attn (incl. kv_b products) %.1f us\n", ctx, T,
             ms[0] * 1000, ms[1] * 1000, ms[2] * 1000);
    }
    hy4::free_indexer(ic);
    hy4::free_latent(lc);
  }
  for (uint32_t H : {64u, 32u})  // latent decode over 2048 keys: T 1, 6 and 16
    for (int T : {1, 6, 16}) {
      const uint32_t cap = 2048 + 16;
      hy4::LatentCache lc = hy4::alloc_latent(cap, "bench_latent");
      std::vector<double> sk(H, 0.5), wkv = randv(size_t(H) * 448 * 512, 0.05);
      Tensor tsk = tensor(sk, DType::F32);
      // 4 copies of kv_b (and of the cache) taken in turn: a working set past the system-level cache, as 42 layers
      Tensor tkvs[4];
      hy4::LatentCache lcs[4];
      for (int c = 0; c < 4; ++c) {
        tkvs[c] = tensor(wkv, DType::BF16);
        lcs[c] = c ? hy4::alloc_latent(cap, "bench_latent") : lc;
      }
      hy4::AttnLayer L;
      L.H = H;
      L.sinks = &tsk;
      Buffer q = fbuf(size_t(T) * H * 256), lat = fbuf(size_t(T) * 512), kr = fbuf(size_t(T) * 64),
             gate = fbuf(size_t(T) * H * 256), out = fbuf(size_t(T) * H * 256);
      Buffer scratch = g.alloc(hy4::latent_scratch_bytes(H, T, 0), "bench_scratch");
      const int iters = 200;
      for (int rep = 0; rep < 3; ++rep) {
        auto e = g.begin();
        for (int i = 0; i < iters; ++i)
          hy4::enc_attn_latent(*e, L, lcs[i % 4], 2048 - T, T, q, H * 256, lat, 512, kr, 64, 0, tkvs[i % 4], gate, out,
                               false, scratch);
        e->commit_and_wait();
        printf("latent decode H %u T %d over 2048 keys: %.1f us per call (incl. W_UK, W_UV: %.1f MB of kv_b)\n", H, T,
               e->gpu_ms() * 1000 / iters, H * 448 * 512 * 2 / 1e6);
      }
      for (auto& c : lcs) hy4::free_latent(c);
    }
  for (auto [H, win, pos0] : {std::tuple<uint32_t, uint32_t, int>{64, 512, 0}, {32, 0, 0}, {64, 512, 1536}, {32, 0, 1536}}) {
    const int T = 512;
    hy4::LatentCache lc = hy4::alloc_latent(win ? win + T - 1 : pos0 + T, "bench_latent", win != 0);
    std::vector<double> sk(H, 0.5), wkv = randv(size_t(H) * 448 * 512, 0.05);
    Tensor tsk = tensor(sk, DType::F32), tkv = tensor(wkv, DType::BF16);
    hy4::AttnLayer L;
    L.H = H;
    L.window = win;
    L.sinks = &tsk;
    Buffer q = fbuf(size_t(T) * H * 256), lat = fbuf(size_t(T) * 512), kr = fbuf(size_t(T) * 64),
           gate = fbuf(size_t(T) * H * 256), out = fbuf(size_t(T) * H * 256);
    Buffer scratch = g.alloc(hy4::latent_scratch_bytes(H, T, win ? T + win - 1 : pos0 + T), "bench_scratch");
    const int iters = 20;
    for (int rep = 0; rep < 3; ++rep) {
      auto e = g.begin();
      for (int i = 0; i < iters; ++i)
        hy4::enc_attn_latent(*e, L, lc, pos0, T, q, H * 256, lat, 512, kr, 64, 0, tkv, gate, out, false, scratch);
      e->commit_and_wait();
      printf("latent prefill H %u window %u T %d at pos0 %d: %.3f ms per call (%s)\n", H, win, T, pos0,
             e->gpu_ms() / iters, getenv("NI_HY4_MLA_PREFILL") ? "absorbed" : "expand from cache");
    }
    hy4::free_latent(lc);
  }
  for (auto [H, win, pos0] : {std::tuple<uint32_t, uint32_t, int>{64, 512, 0}, {32, 0, 0}, {64, 512, 1536}, {32, 0, 1536}}) {
    const int T = 512;
    const uint32_t cap = pos0 + T;
    hy4::KvCache kv = hy4::alloc_kv(H, cap, "bench_kv");
    std::vector<double> sk(H, 0.5);
    Tensor tsk = tensor(sk, DType::F32);
    hy4::AttnLayer L;
    L.H = H;
    L.window = win;
    L.sinks = &tsk;
    Buffer q = fbuf(size_t(T) * H * 256), kvb = fbuf(size_t(T) * H * 448), kr = fbuf(size_t(T) * 576),
           gate = fbuf(size_t(T) * H * 256), out = fbuf(size_t(T) * H * 256);
    Buffer scratch = g.alloc(hy4::attn_scratch_bytes(H, T), "bench_scratch");
    const int iters = 20;
    for (int rep = 0; rep < 3; ++rep) {
      auto e = g.begin();
      for (int i = 0; i < iters; ++i) hy4::enc_attn(*e, L, kv, pos0, T, q, kvb, kr, 576, 512, gate, out, false, scratch);
      e->commit_and_wait();
      printf("prefill H %u window %u T %d at pos0 %d: %.3f ms per call (%s)\n", H, win, T, pos0,
             e->gpu_ms() / iters, getenv("NI_HY4_ATTN_K16") ? "k16" : "k32");
    }
    hy4::free_kv(kv);
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool small = false, do_bench = false;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--small")) small = true;
    else if (!strcmp(argv[i], "--bench")) do_bench = true;
    else {
      fprintf(stderr, "usage: hy4_ops_check [--small] [--bench]\n");
      return 2;
    }
  }
  printf("GPU %s\n", Gpu::get().name().c_str());
  if (do_bench) {
    bench();
    return 0;
  }
  const uint32_t D = small ? 64 : 3072;
  const int R = small ? 3 : 5;
  for (uint32_t nout : {8u, 4u})
    for (bool y : {false, true}) {
      if (nout == 4 && y) continue;
      check_hc(D, 1, nout, y, DType::BF16, false);
      check_hc(D, R, nout, y, DType::F32, false);
    }
  check_hc(D, R, 8, true, DType::BF16, true);
  check_inject_expand(D, R);
  check_rmsnorm(D, R, DType::BF16, false, false);
  check_rmsnorm(D, 1, DType::F32, false, true);
  check_rmsnorm(D, R, DType::BF16, true, false);
  check_add_swiglu(small ? 100 : 1536, R);
  check_router(small ? 64 : 512, small ? 4 : 16, small ? 5 : 64);
  check_router(small ? 40 : 512, small ? 8 : 16, 1);
  check_engram(D, R, DType::BF16);
  check_engram(D, 1, DType::F32);
  if (small) {
    check_attn({2, 16, false, 200, {{0, 70}, {70, 1}, {71, 7}, {78, 40}, {118, 1}}});
    check_attn({2, 0, false, 120, {{0, 40}, {40, 1}, {41, 7}, {48, 70}}, true});
    check_attn({2, 16, true, 16 + 40 - 1, {{0, 40}, {40, 1}, {41, 7}, {48, 40}, {88, 1}, {89, 1}}});
    check_attn_latent({16, 0, false, 200, {{0, 40}, {40, 1}, {41, 6}, {47, 16}, {63, 70}, {133, 1}}});
    check_attn_latent({16, 16, true, 16 + 40 - 1, {{0, 40}, {40, 1}, {41, 6}, {47, 16}, {63, 40}, {103, 1}}, true});
    check_indexer(16, {{0, 1000}, {1000, 1030}, {2030, 1}, {2031, 30}, {2061, 1}, {2062, 6}, {2068, 16}, {2084, 3}}, 1);
  } else {
    check_attn({32, 0, false, 1100, {{0, 700}, {700, 1}, {701, 7}, {708, 300}, {1008, 1}}});
    check_attn({64, 512, false, 1100, {{0, 700}, {700, 300}, {1000, 1}, {1001, 7}}, true});
    check_attn({64, 512, true, 512 + 700 - 1, {{0, 300}, {300, 700}, {1000, 700}, {1700, 1}, {1701, 7}, {1708, 1}}});
    check_attn_latent({32, 0, false, 1100, {{0, 700}, {700, 1}, {701, 6}, {707, 16}, {723, 300}, {1023, 1}}});
    check_attn_latent(
        {64, 512, true, 512 + 700 - 1, {{0, 300}, {300, 700}, {1000, 1}, {1001, 6}, {1007, 16}, {1023, 40}}, true});
    check_indexer(32, {{0, 1500}, {1500, 1000}, {2500, 1}, {2501, 6}, {2507, 16}, {2523, 450}, {2973, 1}, {2974, 3},
                       {2977, 23}},
                  7);
  }
  // fp16 weights (after the rest: the random draws of the checks above stay what they were)
  check_hc(D, R, 8, false, DType::F16, false);
  check_rmsnorm(D, R, DType::F16, false, false);
  check_engram(D, R, DType::F16);
  printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
