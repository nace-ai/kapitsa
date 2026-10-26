// Qwen3.8-Flash-Next checks.
//   fn_check ngram @ids             the PLE n-gram ids of every position ("t: id0 .. id15"), CPU only: compare with a
//                                   port of the reference's hash (transformers' Qwen4ExpTextNGramEmbedding)
//   fn_check selftest               the hash on a built-in sequence against ids computed by that port (CPU only)
//   fn_check kernels                the Flash-Next kernels (src/kernels/qwen4.metal) on small random inputs against a CPU
//                                   reference in double; one dispatch per command buffer, so it suits
//                                   MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1
//   fn_check qsa-time               the qsa_score / qsa_score_rows A/B only, 40 dispatches behind one commit: a timing run,
//                                   so it wants the layer OFF and an idle GPU, interleaved against the old build (AGENTS.md)
//   fn_check compile-only           compiles the embedded kernel library and exits: the zero-dispatch gate for a .metal
//                                   edit (no model, no buffer, no command buffer, ~1 s). A successful C++ build says
//                                   nothing about the sources (they are embedded as text and compiled at run time), so
//                                   this is the cheapest run that can catch a source fault. Device-touching: fn_check
//                                   takes no lock itself, so run it under `lockf -k /tmp/kapitsa-gpu.lock` like any other
//                                   GPU run, with MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1
#include "metal/gpu.h"
#include "model/ngram.h"
#include "util/bf16.h"
#include "util/log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;

namespace {

std::vector<int> read_ids(const std::string& arg0) {
  std::string arg = arg0;
  if (!arg.empty() && arg[0] == '@') {
    std::ifstream f(arg.substr(1));
    if (!f) fatal("cannot read %s", arg.c_str() + 1);
    arg.assign(std::istreambuf_iterator<char>(f), {});
  }
  std::vector<int> v;
  std::stringstream ss(arg);
  std::string tok;
  while (std::getline(ss, tok, ','))
    if (tok.find_first_not_of(" \n\r\t") != std::string::npos) v.push_back(std::stoi(tok));
  return v;
}

// Qwen3.8-Flash-Next's constants (config.json: vocab 248320, seed 1234, base 20000000, 8 heads per n-gram, n = 3,
// eos 248044)
NgramHash flash_hash() { return NgramHash::make(248320, 1234, 0, 20000000, 8, 3, 248044); }

int selftest() {
  const NgramHash h = flash_hash();
  // the reference's constants (SPEC: ngram_consts.json)
  const int64_t mult[3] = {23703573157769ll, 20109073645365ll, 8052911324071ll};
  int bad = 0;
  for (int i = 0; i < 3; ++i) bad += h.mult[i] != mult[i];
  bad += h.mod[0] != 20000003 || h.mod[15] != 20000171 || h.off[15] != 300001275 || h.rows != 320001446;
  // tokens with eos (248044) at the start, in the middle and twice in a row; ids from the port (ngram_ref.py)
  const std::vector<int> tok = {248044, 17, 248044, 5, 6, 7, 248044, 248044, 9, 10};
  const int64_t want[10][2] = {  // heads 0 and 15 of each position
      {9663979, 300986548},  {7961118, 308564037},  {5560156, 305423709},  {15389869, 315720898},
      {15567690, 307414745}, {9319853, 313224935},  {10204458, 310772445}, {9663979, 300986548},
      {18043673, 307951937}, {9058818, 309980676}};
  std::vector<int64_t> ids(16);
  int wrong = 0;
  for (size_t t = 0; t < tok.size(); ++t) {
    h.ids(tok.data(), int(t), ids.data());
    wrong += ids[0] != want[t][0] || ids[15] != want[t][1];
  }
  printf("n-gram hash: constants %s, ids %s\n", bad ? "MISMATCH" : "ok", wrong ? "MISMATCH" : "ok");
  return bad || wrong ? 1 : 0;
}

// ---- GPU kernels against a CPU reference ----
std::mt19937 rng(7);
float urand(float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); }

struct Buf {
  Buffer b;
  size_t n = 0;
  Buf(size_t count, size_t elem = 4) : n(count) { b = Gpu::get().alloc(std::max<size_t>(count * elem, 16)); }
  float* f() { return b.as<float>(); }
  uint16_t* u16() { return b.as<uint16_t>(); }
  void fill(float a, float c) { for (size_t i = 0; i < n; ++i) f()[i] = urand(a, c); }
  void fill_bf16(float a, float c) { for (size_t i = 0; i < n; ++i) u16()[i] = f32_to_bf16(urand(a, c)); }
};
float bf(const Buf& b, size_t i) { return bf16_to_f32(b.b.as<uint16_t>()[i]); }
float hf(const Buf& b, size_t i) { return float(b.b.as<_Float16>()[i]); }

void run(const char* kernel, const std::function<void(Encoder&)>& bind, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t tx) {
  auto e = Gpu::get().begin();
  e->use(Gpu::get().pipeline(kernel));
  bind(*e);
  e->dispatch(gx, gy, gz, tx);
  e->commit_and_wait();
}

int g_fail = 0;
void report(const char* what, double err, double tol) {
  const bool bad = !(err <= tol);
  g_fail += bad;
  printf("%-34s max rel err %.2e%s\n", what, err, bad ? "  FAILED" : "");
}
// max |a - b| / max(|b|, floor) over n values
template <class A, class B>
double cmp(size_t n, A a, B b, double floor = 1e-3) {
  double e = 0;
  for (size_t i = 0; i < n; ++i) e = std::max(e, std::fabs(double(a(i)) - double(b(i))) / std::max(std::fabs(double(b(i))), floor));
  return e;
}
double sigm(double x) { return 1 / (1 + std::exp(-x)); }
double silu(double x) { return x * sigm(x); }

struct HcP {
  uint32_t R, D, S, has_y, has_inj;
  float eps;
};
struct PleP {
  uint32_t R, D, S, pos0;
  float eps;
  uint32_t ring, dil;
};
struct QsaP {
  uint32_t R, NH, HD, rot, pos0, ratio, ld;
  float eps, theta;
};

void test_hc() {
  const uint32_t R = 3, D = 200, S = 4, SD = S * D;
  for (int half_out = 0; half_out < 2; ++half_out)
    for (int mode = 0; mode < 2; ++mode) {
      Buf X(R * SD), y(R * D), injp(R * S), w(SD), winj(S * SD, 2), xn(R * SD), inj(R * S);
      X.fill(-2, 2), y.fill(-1, 1), injp.fill(0, 2), w.fill(0.5f, 1.5f), winj.fill_bf16(-0.05f, 0.05f);
      std::vector<float> X0(X.f(), X.f() + R * SD), y0(y.f(), y.f() + R * D);
      HcP p{R, D, S, uint32_t(mode), uint32_t(mode), 1e-6f};
      run(half_out ? "hc_pre_h" : "hc_pre", [&](Encoder& e) {
        e.buf(0, X.b); e.buf(1, y.b); e.buf(2, injp.b); e.buf(3, w.b); e.buf(4, winj.b); e.buf(5, xn.b); e.buf(6, inj.b);
        e.val(7, p);
      }, R, 1, 1, 256);
      std::vector<double> rx(R * SD), rxn(R * SD), rinj(R * S);
      for (uint32_t r = 0; r < R; ++r) {
        for (uint32_t i = 0; i < SD; ++i) rx[r * SD + i] = X0[r * SD + i] + (mode ? injp.f()[r * S + i / D] * y0[r * D + i % D] : 0.0);
        for (uint32_t s = 0; s < S; ++s) {
          double ss = 0;
          for (uint32_t c = 0; c < D; ++c) ss += rx[r * SD + s * D + c] * rx[r * SD + s * D + c];
          const double inv = 1 / std::sqrt(ss / D + 1e-6);
          for (uint32_t c = 0; c < D; ++c) rxn[r * SD + s * D + c] = rx[r * SD + s * D + c] * inv * w.f()[s * D + c];
        }
        for (uint32_t q = 0; q < S; ++q) {
          double d = 0;
          for (uint32_t i = 0; i < SD; ++i) d += bf(winj, q * SD + i) * rxn[r * SD + i];
          rinj[r * S + q] = 2 * sigm(d / S);
        }
      }
      char name[64];
      snprintf(name, sizeof name, "%s %s: X", half_out ? "hc_pre_h" : "hc_pre", mode ? "inject" : "plain");
      report(name, cmp(R * SD, [&](size_t i) { return X.f()[i]; }, [&](size_t i) { return rx[i]; }), 1e-4);
      snprintf(name, sizeof name, "%s %s: Xn", half_out ? "hc_pre_h" : "hc_pre", mode ? "inject" : "plain");
      report(name, cmp(R * SD, [&](size_t i) { return half_out ? hf(xn, i) : xn.f()[i]; }, [&](size_t i) { return rxn[i]; }, 1e-2),
             half_out ? 2e-3 : 1e-5);
      if (mode) {
        report("  inj", cmp(R * S, [&](size_t i) { return inj.f()[i]; }, [&](size_t i) { return rinj[i]; }), 1e-4);
        report("  y zeroed", cmp(R * D, [&](size_t i) { return y.f()[i]; }, [](size_t) { return 0.0; }, 1.0), 0);
      }
    }
  // hc_act, hc_mix
  {
    Buf h(R * 320), m(R * SD), xn(R * SD), u(R * D);
    h.fill(-8, 8), m.fill(-4, 4), xn.fill(-2, 2);
    std::vector<float> h0(h.f(), h.f() + h.n);
    const uint32_t n = uint32_t(h.n);
    const float inv_s = 0.25f;
    run("hc_act", [&](Encoder& e) { e.buf(0, h.b); e.val(1, n); e.val(2, inv_s); }, (n + 255) / 256, 1, 1, 256);
    report("hc_act", cmp(h.n, [&](size_t i) { return h.f()[i]; }, [&](size_t i) { return silu(h0[i] / 4); }), 1e-5);
    HcP p{R, D, S, 0, 0, 1e-6f};
    const uint32_t ld = D;
    run("hc_mix", [&](Encoder& e) { e.buf(0, m.b); e.buf(1, xn.b); e.buf(2, u.b); e.val(3, p); e.val(4, ld); }, (D + 255) / 256, R, 1, 256);
    report("hc_mix", cmp(R * D, [&](size_t i) { return u.f()[i]; }, [&](size_t i) {
      const size_t r = i / D, c = i % D;
      double a = 0;
      for (uint32_t s = 0; s < S; ++s) a += sigm(m.f()[r * SD + s * D + c]) * xn.f()[r * SD + s * D + c];
      return a / S;
    }), 1e-4);
  }
}

// the one-row gated residual (hc_down + hc_up) against fp64, BITS-bit group-64 weights
void test_hc_row(int bits) {
  const uint32_t D = 512, S = 4, SD = S * D, RK = 64, G = 64;
  Buf X(SD), y(D), injp(S), w(SD), winj(S * SD, 2), X2(SD), h(RK), inv(4), inj(4), u(D);
  Buf Wd(size_t(RK) * SD * bits / 32), Sd(RK * SD / G, 2), Bd(RK * SD / G, 2);
  Buf Wu(size_t(SD) * RK * bits / 32), Su(SD * RK / G, 2), Bu(SD * RK / G, 2);
  X.fill(-2, 2), y.fill(-1, 1), injp.fill(0, 2), w.fill(0.5f, 1.5f), winj.fill_bf16(-0.05f, 0.05f);
  for (size_t i = 0; i < Wd.n; ++i) Wd.b.as<uint32_t>()[i] = uint32_t(rng());
  for (size_t i = 0; i < Wu.n; ++i) Wu.b.as<uint32_t>()[i] = uint32_t(rng());
  const float sc = bits == 4 ? 0.004f : 0.0003f;
  Sd.fill_bf16(sc / 2, sc), Bd.fill_bf16(-sc * (1 << bits) / 2, 0), Su.fill_bf16(sc, 2 * sc), Bu.fill_bf16(-sc * (1 << bits), 0);
  std::vector<float> X0(X.f(), X.f() + SD), y0(y.f(), y.f() + D);
  struct { uint32_t D, S, rank, has_y, has_inj; float eps; } p{D, S, RK, 1, 1, 1e-6f};
  const std::string sfx = "_b" + std::to_string(bits);
  run(("hc_down" + sfx).c_str(), [&](Encoder& e) {
    e.buf(0, X.b); e.buf(1, y.b); e.buf(2, injp.b); e.buf(3, w.b); e.buf(4, Wd.b); e.buf(5, Sd.b); e.buf(6, Bd.b);
    e.buf(7, winj.b); e.buf(8, X2.b); e.buf(9, h.b); e.buf(10, inv.b); e.buf(11, inj.b); e.val(12, p);
  }, RK / 8 + 1, 1, 1, 256);
  run(("hc_up" + sfx).c_str(), [&](Encoder& e) {
    e.buf(0, h.b); e.buf(1, X2.b); e.buf(2, inv.b); e.buf(3, w.b); e.buf(4, Wu.b); e.buf(5, Su.b); e.buf(6, Bu.b);
    e.buf(7, u.b); e.buf(8, X.b); e.buf(9, y.b); e.val(10, p);
  }, D / 8, 1, 1, 256);
  auto deq = [&](const Buf& W, const Buf& Sc, const Buf& Bi, uint32_t n, uint32_t k, uint32_t K) {
    const uint32_t per = 32 / bits;
    const uint32_t q = (W.b.as<uint32_t>()[(size_t(n) * K + k) / per] >> (bits * (k % per))) & ((1u << bits) - 1);
    return double(bf(Sc, n * (K / G) + k / G)) * q + bf(Bi, n * (K / G) + k / G);
  };
  std::vector<double> x(SD), xn(SD), hh(RK), uu(D), ij(S);
  for (uint32_t i = 0; i < SD; ++i) x[i] = X0[i] + injp.f()[i / D] * y0[i % D];
  for (uint32_t s = 0; s < S; ++s) {
    double ss = 0;
    for (uint32_t c = 0; c < D; ++c) ss += x[s * D + c] * x[s * D + c];
    for (uint32_t c = 0; c < D; ++c) xn[s * D + c] = x[s * D + c] / std::sqrt(ss / D + 1e-6) * w.f()[s * D + c];
  }
  for (uint32_t r = 0; r < RK; ++r) {
    double a = 0;
    for (uint32_t k = 0; k < SD; ++k) a += deq(Wd, Sd, Bd, r, k, SD) * xn[k];
    hh[r] = silu(a / S);
  }
  for (uint32_t q = 0; q < S; ++q) {
    double a = 0;
    for (uint32_t i = 0; i < SD; ++i) a += bf(winj, q * SD + i) * xn[i];
    ij[q] = 2 * sigm(a / S);
  }
  for (uint32_t c = 0; c < D; ++c) {
    double a = 0;
    for (uint32_t s = 0; s < S; ++s) {
      double m = 0;
      for (uint32_t k = 0; k < RK; ++k) m += deq(Wu, Su, Bu, s * D + c, k, RK) * hh[k];
      a += sigm(m) * xn[s * D + c];
    }
    uu[c] = a / S;
  }
  const std::string n = "hc_down/up" + sfx;
  report((n + ": h").c_str(), cmp(RK, [&](size_t i) { return h.f()[i]; }, [&](size_t i) { return hh[i]; }, 1e-2), 1e-4);
  report((n + ": inj").c_str(), cmp(S, [&](size_t i) { return inj.f()[i]; }, [&](size_t i) { return ij[i]; }), 1e-5);
  report((n + ": u").c_str(), cmp(D, [&](size_t i) { return u.f()[i]; }, [&](size_t i) { return uu[i]; }, 1e-2), 1e-4);
  report((n + ": X").c_str(), cmp(SD, [&](size_t i) { return X.f()[i]; }, [&](size_t i) { return x[i]; }), 1e-5);
  report((n + ": y zeroed").c_str(), cmp(D, [&](size_t i) { return y.f()[i]; }, [](size_t) { return 0.0; }, 1.0), 0);
}

// the R-row low-rank mix after hc_pre (hc_down_sk + hc_up_mix) against fp64
void test_hc_mix_sk(int bits) {
  const uint32_t D = 512, S = 4, SD = S * D, RK = 128, G = 64, R = 3, KC = SD / 1024;
  Buf xn(R * SD), part(KC * R * RK), u(R * D);
  Buf Wd(size_t(RK) * SD * bits / 32), Sd(RK * SD / G, 2), Bd(RK * SD / G, 2);
  Buf Wu(size_t(SD) * RK * bits / 32), Su(SD * RK / G, 2), Bu(SD * RK / G, 2);
  xn.fill(-2, 2);
  for (size_t i = 0; i < Wd.n; ++i) Wd.b.as<uint32_t>()[i] = uint32_t(rng());
  for (size_t i = 0; i < Wu.n; ++i) Wu.b.as<uint32_t>()[i] = uint32_t(rng());
  const float sc = bits == 4 ? 0.004f : 0.0003f;
  Sd.fill_bf16(sc / 2, sc), Bd.fill_bf16(-sc * (1 << bits) / 2, 0), Su.fill_bf16(sc, 2 * sc), Bu.fill_bf16(-sc * (1 << bits), 0);
  struct { uint32_t R, D, S, rank, kc, u_ld; } p{R, D, S, RK, KC, D};
  const std::string sfx = "_b" + std::to_string(bits);
  run(("hc_down_sk" + sfx).c_str(), [&](Encoder& e) {
    e.buf(0, xn.b); e.buf(1, Wd.b); e.buf(2, Sd.b); e.buf(3, Bd.b); e.buf(4, part.b); e.val(5, p);
  }, RK / 8, KC, R, 256);
  run(("hc_up_mix" + sfx).c_str(), [&](Encoder& e) {
    e.buf(0, part.b); e.buf(1, xn.b); e.buf(2, Wu.b); e.buf(3, Su.b); e.buf(4, Bu.b); e.buf(5, u.b); e.val(6, p);
  }, D / 16, R, 1, 256);
  auto deq = [&](const Buf& W, const Buf& Sc, const Buf& Bi, uint32_t n, uint32_t k, uint32_t K) {
    // value k of row n: bits [k * bits, k * bits + bits) of the row's little-endian bit stream
    const uint64_t bit = (uint64_t(n) * K + k) * uint64_t(bits);
    const uint32_t* w = W.b.as<uint32_t>();
    uint64_t v = w[bit / 32];
    if (bit % 32 + bits > 32) v |= uint64_t(w[bit / 32 + 1]) << 32;
    const uint32_t q = uint32_t(v >> (bit % 32)) & ((1u << bits) - 1);
    return double(bf(Sc, n * (K / G) + k / G)) * q + bf(Bi, n * (K / G) + k / G);
  };
  double err = 0;
  for (uint32_t r = 0; r < R; ++r) {
    std::vector<double> hh(RK);
    for (uint32_t n = 0; n < RK; ++n) {
      double a = 0;
      for (uint32_t k = 0; k < SD; ++k) a += deq(Wd, Sd, Bd, n, k, SD) * xn.f()[r * SD + k];
      hh[n] = silu(a / S);
    }
    for (uint32_t c = 0; c < D; ++c) {
      double a = 0;
      for (uint32_t s2 = 0; s2 < S; ++s2) {
        double m = 0;
        for (uint32_t k = 0; k < RK; ++k) m += deq(Wu, Su, Bu, s2 * D + c, k, RK) * hh[k];
        a += sigm(m) * xn.f()[r * SD + s2 * D + c];
      }
      const double ref = a / S;
      err = std::max(err, std::fabs(ref - u.f()[r * D + c]) / std::max(std::fabs(ref), 1e-2));
    }
  }
  report(("hc_down_sk + hc_up_mix" + sfx + ": u").c_str(), err, 1e-4);
}

// the folded gated residual (hc_down_sk2 + hc_up_mix2: injection, norms, mix, injection weights) against fp64;
// D = 768 so 1024-input chunks straddle streams mid-chunk as at D = 2560
void test_hc_fold(int bits) {
  const uint32_t D = 768, S = 4, SD = S * D, RK = 128, G = 64, R = 3, KC = SD / 1024;
  for (int mode = 0; mode < 2; ++mode) {
    Buf X(R * SD), y(R * D), injp(R * S), w(SD), winj(S * SD, 2), inj(R * S), u(R * D);
    Buf part(KC * R * RK * 2), X2(R * SD), stat(R * KC * 10);
    Buf Wd(size_t(RK) * SD * bits / 32), Sd(RK * SD / G, 2), Bd(RK * SD / G, 2);
    Buf Wu(size_t(SD) * RK * bits / 32), Su(SD * RK / G, 2), Bu(SD * RK / G, 2);
    X.fill(-2, 2), y.fill(-1, 1), injp.fill(0, 2), w.fill(0.5f, 1.5f), winj.fill_bf16(-0.05f, 0.05f);
    for (size_t i = 0; i < Wd.n; ++i) Wd.b.as<uint32_t>()[i] = uint32_t(rng());
    for (size_t i = 0; i < Wu.n; ++i) Wu.b.as<uint32_t>()[i] = uint32_t(rng());
    const float sc = bits == 4 ? 0.004f : 0.0003f;
    Sd.fill_bf16(sc / 2, sc), Bd.fill_bf16(-sc * (1 << bits) / 2, 0), Su.fill_bf16(sc, 2 * sc), Bu.fill_bf16(-sc * (1 << bits), 0);
    std::vector<float> X0(X.f(), X.f() + R * SD), y0(y.f(), y.f() + R * D);
    struct { uint32_t R, D, S, rank, kc, u_ld, has_y, has_inj; float eps; } p{R, D, S, RK, KC, D, uint32_t(mode), uint32_t(mode), 1e-6f};
    const std::string sfx = "_b" + std::to_string(bits);
    run(("hc_down_sk2" + sfx).c_str(), [&](Encoder& e) {
      e.buf(0, X.b); e.buf(1, y.b); e.buf(2, injp.b); e.buf(3, w.b); e.buf(4, winj.b); e.buf(5, Wd.b); e.buf(6, Sd.b);
      e.buf(7, Bd.b); e.buf(8, part.b); e.buf(9, X2.b); e.buf(10, stat.b); e.val(11, p);
    }, RK / 8, KC, R, 256);
    run(("hc_up_mix2" + sfx).c_str(), [&](Encoder& e) {
      e.buf(0, part.b); e.buf(1, X2.b); e.buf(2, stat.b); e.buf(3, w.b); e.buf(4, Wu.b); e.buf(5, Su.b); e.buf(6, Bu.b);
      e.buf(7, u.b); e.buf(8, X.b); e.buf(9, y.b); e.buf(10, inj.b); e.val(11, p);
    }, D / 16, R, 1, 256);
    auto deq = [&](const Buf& W, const Buf& Sc, const Buf& Bi, uint32_t n, uint32_t k, uint32_t K) {
      const uint64_t bit = (uint64_t(n) * K + k) * uint64_t(bits);
      const uint32_t* wq = W.b.as<uint32_t>();
      uint64_t v = wq[bit / 32];
      if (bit % 32 + bits > 32) v |= uint64_t(wq[bit / 32 + 1]) << 32;
      const uint32_t q = uint32_t(v >> (bit % 32)) & ((1u << bits) - 1);
      return double(bf(Sc, n * (K / G) + k / G)) * q + bf(Bi, n * (K / G) + k / G);
    };
    double eu = 0, ex = 0, ei = 0;
    for (uint32_t r = 0; r < R; ++r) {
      std::vector<double> rx(SD), rxn(SD), hh(RK);
      for (uint32_t i = 0; i < SD; ++i) rx[i] = X0[r * SD + i] + (mode ? injp.f()[r * S + i / D] * y0[r * D + i % D] : 0.0);
      for (uint32_t s2 = 0; s2 < S; ++s2) {
        double ss = 0;
        for (uint32_t c = 0; c < D; ++c) ss += rx[s2 * D + c] * rx[s2 * D + c];
        const double iv = 1 / std::sqrt(ss / D + 1e-6);
        for (uint32_t c = 0; c < D; ++c) rxn[s2 * D + c] = rx[s2 * D + c] * iv * w.f()[s2 * D + c];
      }
      for (uint32_t i = 0; i < SD; ++i) ex = std::max(ex, std::fabs(X.f()[r * SD + i] - rx[i]) / std::max(std::fabs(rx[i]), 1e-3));
      if (mode)
        for (uint32_t q = 0; q < S; ++q) {
          double d = 0;
          for (uint32_t i = 0; i < SD; ++i) d += bf(winj, q * SD + i) * rxn[i];
          const double ref = 2 * sigm(d / S);
          ei = std::max(ei, std::fabs(inj.f()[r * S + q] - ref) / ref);
        }
      for (uint32_t n = 0; n < RK; ++n) {
        double a = 0;
        for (uint32_t k = 0; k < SD; ++k) a += deq(Wd, Sd, Bd, n, k, SD) * rxn[k];
        hh[n] = silu(a / S);
      }
      for (uint32_t c = 0; c < D; ++c) {
        double a = 0;
        for (uint32_t s2 = 0; s2 < S; ++s2) {
          double m = 0;
          for (uint32_t k = 0; k < RK; ++k) m += deq(Wu, Su, Bu, s2 * D + c, k, RK) * hh[k];
          a += sigm(m) * rxn[s2 * D + c];
        }
        const double ref = a / S;
        eu = std::max(eu, std::fabs(ref - u.f()[r * D + c]) / std::max(std::fabs(ref), 1e-2));
      }
    }
    const std::string nm = "hc_down_sk2 + hc_up_mix2" + sfx + (mode ? " inject" : " plain");
    report((nm + ": u").c_str(), eu, 1e-4);
    report((nm + ": X").c_str(), ex, 1e-4);  // (x + a y: fma vs fp64, relative to values near 1e-3)
    if (mode) {
      report((nm + ": inj").c_str(), ei, 1e-4);
      report((nm + ": y zeroed").c_str(), cmp(R * D, [&](size_t i) { return y.f()[i]; }, [](size_t) { return 0.0; }, 1.0), 0);
    }
  }
}

// CPU reference of the PLE block over rows [0, R) at positions pos0.. with gvn history `hist` (positions < pos0)
void ple_ref(uint32_t R, uint32_t D, uint32_t S, uint32_t pos0, const float* key, const float* val, std::vector<double>& X,
             const float* wk, const float* wq, const float* wc, const Buf& cw, std::vector<std::vector<double>>& hist) {
  const uint32_t SD = S * D;
  for (uint32_t r = 0; r < R; ++r) {
    std::vector<double> gvn(SD), gv(SD);
    for (uint32_t s = 0; s < S; ++s) {
      double kss = 0, xss = 0;
      for (uint32_t c = 0; c < D; ++c) {
        kss += double(key[r * SD + s * D + c]) * key[r * SD + s * D + c];
        xss += X[r * SD + s * D + c] * X[r * SD + s * D + c];
      }
      const double ki = 1 / std::sqrt(kss / D + 1e-6), xi = 1 / std::sqrt(xss / D + 1e-6);
      double g = 0;
      for (uint32_t c = 0; c < D; ++c)
        g += key[r * SD + s * D + c] * ki * wk[s * D + c] * X[r * SD + s * D + c] * xi * wq[s * D + c];
      g /= std::sqrt(double(D));
      g = (g > 0 ? 1 : g < 0 ? -1 : 0) * std::sqrt(std::max(std::fabs(g), 1e-6));
      double gss = 0;
      for (uint32_t c = 0; c < D; ++c) {
        gv[s * D + c] = sigm(g) * val[r * D + c];
        gss += gv[s * D + c] * gv[s * D + c];
      }
      for (uint32_t c = 0; c < D; ++c) gvn[s * D + c] = gv[s * D + c] / std::sqrt(gss / D + 1e-6) * wc[s * D + c];
    }
    hist.push_back(gvn);  // hist[p] = gvn of position p
    const uint32_t pos = pos0 + r;
    for (uint32_t c = 0; c < SD; ++c) {
      double acc = 0;
      for (uint32_t j = 0; j < 4; ++j) {
        const uint32_t back = 3 * (3 - j);
        if (pos >= back) acc += bf(cw, c * 4 + j) * hist[pos - back][c];
      }
      X[r * SD + c] += gv[c] + silu(acc);
    }
  }
}

void test_ple() {
  const uint32_t D = 128, S = 4, SD = S * D, RING = 32;
  for (int prefill = 0; prefill < 2; ++prefill) {
    const uint32_t R = prefill ? 13 : 5, pos0 = prefill ? 7 : 4;
    Buf key(R * SD), val(R * D), X(R * SD), wk(SD), wq(SD), wc(SD), cw(SD * 4, 2), ring(RING * SD);
    key.fill(-1, 1), val.fill(-1, 1), X.fill(-2, 2), wk.fill(0.5f, 1.5f), wq.fill(0.5f, 1.5f), wc.fill(0.5f, 1.5f);
    cw.fill_bf16(-0.5f, 0.5f), ring.fill(-3, 3);  // ring slots of positions >= pos0: garbage the kernels must not read
    std::vector<std::vector<double>> hist;
    for (uint32_t q = 0; q < pos0; ++q) {
      for (uint32_t c = 0; c < SD; ++c) ring.f()[(q % RING) * SD + c] = urand(-1, 1);
      hist.push_back(std::vector<double>(ring.f() + (q % RING) * SD, ring.f() + (q % RING + 1) * SD));
    }
    std::vector<double> rx(X.f(), X.f() + X.n);
    ple_ref(R, D, S, pos0, key.f(), val.f(), rx, wk.f(), wq.f(), wc.f(), cw, hist);
    PleP p{R, D, S, pos0, 1e-6f, RING, 3};
    if (!prefill) {
      run("ple_rows", [&](Encoder& e) {
        e.buf(0, key.b); e.buf(1, val.b); e.buf(2, X.b); e.buf(3, wk.b); e.buf(4, wq.b); e.buf(5, wc.b); e.buf(6, cw.b);
        e.buf(7, ring.b); e.val(8, p);
      }, S, 1, 1, D / 4);
      report("ple_rows: X", cmp(X.n, [&](size_t i) { return X.f()[i]; }, [&](size_t i) { return rx[i]; }), 1e-4);
      if (getenv("FN_DEBUG"))
        for (uint32_t r = 0; r < R; ++r)
          printf("row %u: %f %f | %f %f\n", r, X.f()[r * SD], rx[r * SD], X.f()[r * SD + D + 5], rx[r * SD + D + 5]);
    } else {
      Buf gvb(R * SD), gvnb(R * SD), snap(RING * SD);
      run("ple_gate_rows", [&](Encoder& e) {
        e.buf(0, key.b); e.buf(1, val.b); e.buf(2, X.b); e.buf(3, wk.b); e.buf(4, wq.b); e.buf(5, wc.b); e.buf(6, gvb.b);
        e.buf(7, gvnb.b); e.val(8, p);
      }, R, S, 1, D / 4);
      run("ple_conv_rows", [&](Encoder& e) {
        e.buf(0, gvb.b); e.buf(1, gvnb.b); e.buf(2, X.b); e.buf(3, cw.b); e.buf(4, ring.b); e.val(5, p);
      }, (SD + 255) / 256, R, 1, 256);
      report("ple_gate/conv_rows: X", cmp(X.n, [&](size_t i) { return X.f()[i]; }, [&](size_t i) { return rx[i]; }), 1e-4);
      const uint32_t cpos = pos0 + 5, end = pos0 + R;  // a snapshot inside the pass, then the live ring
      run("ple_ring_rows", [&](Encoder& e) {
        e.buf(0, gvnb.b); e.buf(1, ring.b); e.buf(2, snap.b); e.val(3, p); e.val(4, cpos);
      }, (SD + 255) / 256, RING, 1, 256);
      run("ple_ring_rows", [&](Encoder& e) {
        e.buf(0, gvnb.b); e.buf(1, ring.b); e.buf(2, ring.b); e.val(3, p); e.val(4, end);
      }, (SD + 255) / 256, RING, 1, 256);
      double es = 0, el = 0;
      for (uint32_t q = (cpos > RING ? cpos - RING : 0); q < cpos; ++q)
        for (uint32_t c = 0; c < SD; ++c) es = std::max(es, std::fabs(snap.f()[(q % RING) * SD + c] - hist[q][c]));
      for (uint32_t q = (end > RING ? end - RING : 0); q < end; ++q)
        for (uint32_t c = 0; c < SD; ++c) el = std::max(el, std::fabs(ring.f()[(q % RING) * SD + c] - hist[q][c]));
      report("ple_ring_rows: snapshot", es, 1e-4);
      report("ple_ring_rows: live", el, 1e-4);
    }
  }
}

void test_qsa() {
  const uint32_t R = 9, NH = 4, HD = 128, LD = (NH + 1) * HD, pos0 = 3, ROT = 64;
  const float theta = 1e7f;
  Buf iqk(R * LD), wq(HD), wk(HD), iq(R * NH * HD), ik((pos0 + R) * HD, 2), bk(((pos0 + R) / 4 + 1) * HD);
  iqk.fill(-3, 3), wq.fill(0.5f, 1.5f), wk.fill(0.5f, 1.5f);
  for (uint32_t i = 0; i < pos0 * HD; ++i) ik.b.as<_Float16>()[i] = _Float16(urand(-3, 3));
  QsaP p{R, NH, HD, ROT, pos0, 4, LD, 1e-6f, theta};
  struct { uint32_t mn; int32_t delta; uint32_t sec1, sec2; } mr{0, 0, 33, 30};  // text rows: positions as they are
  run("qsa_prep", [&](Encoder& e) {
    e.buf(0, iqk.b); e.buf(1, wq.b); e.buf(2, iq.b); e.buf(3, ik.b); e.val(4, p); e.buf(5, iq.b); e.val(6, mr);
  }, NH + 1, R, 1, HD);
  run("qsa_block", [&](Encoder& e) {
    e.buf(0, ik.b); e.buf(1, wk.b); e.buf(2, bk.b); e.val(3, p); e.buf(4, ik.b); e.val(5, mr);
  }, R, 1, 1, HD);
  auto rope = [&](std::vector<double>& v, uint32_t pos) {
    std::vector<double> o = v;
    for (uint32_t d = 0; d < ROT; ++d) {
      const uint32_t i = d % (ROT / 2);
      const double ang = pos * std::pow(double(theta), -2.0 * i / ROT);
      o[d] = d < ROT / 2 ? v[d] * std::cos(ang) - v[d + ROT / 2] * std::sin(ang) : v[d] * std::cos(ang) + v[d - ROT / 2] * std::sin(ang);
    }
    v = o;
  };
  auto norm = [&](std::vector<double>& v, const float* w) {
    double ss = 0;
    for (double x : v) ss += x * x;
    for (uint32_t d = 0; d < HD; ++d) v[d] = v[d] / std::sqrt(ss / HD + 1e-6) * w[d];
  };
  std::vector<double> riq(R * NH * HD);
  for (uint32_t r = 0; r < R; ++r)
    for (uint32_t h = 0; h < NH; ++h) {
      std::vector<double> v(iqk.f() + r * LD + h * HD, iqk.f() + r * LD + (h + 1) * HD);
      norm(v, wq.f());
      rope(v, pos0 + r);
      std::copy(v.begin(), v.end(), riq.begin() + (r * NH + h) * HD);
    }
  report("qsa_prep: iq", cmp(riq.size(), [&](size_t i) { return iq.f()[i]; }, [&](size_t i) { return riq[i]; }, 1e-2), 1e-4);
  report("qsa_prep: ik", cmp(R * HD, [&](size_t i) { return hf(ik, pos0 * HD + i); },
                             [&](size_t i) { return iqk.f()[(i / HD) * LD + NH * HD + i % HD]; }, 1e-2), 1e-3);
  double eb = 0;
  for (uint32_t j = 0; j * 4 + 3 < pos0 + R; ++j) {
    if (j * 4 + 3 < pos0) continue;
    std::vector<double> m(HD, 0.0);
    for (uint32_t t = 0; t < 4; ++t)
      for (uint32_t d = 0; d < HD; ++d) m[d] += hf(ik, (j * 4 + t) * HD + d) / 4;
    norm(m, wk.f());
    rope(m, j * 4);
    for (uint32_t d = 0; d < HD; ++d) eb = std::max(eb, std::fabs(bk.f()[j * HD + d] - m[d]) / std::max(std::fabs(m[d]), 1e-2));
  }
  report("qsa_block: block keys", eb, 1e-4);
}

// A/B of two score kernels in one command buffer, reps dispatches behind one commit: the model-level ms/C of the
// indexer scan is per layer and too small to see through a full token, so it is measured here in isolation. A
// command of its own (`qsa-time`), not part of `kernels`: the numbers mean nothing under MTL_SHADER_VALIDATION, and
// 320 dispatches would make the correctness pass slower for it. The R
// sweep separates the two explanations for the scan's cost: time growing with R at fixed nb means every row pays the
// key bytes (a byte-bound kernel, which fusing fixes), flat time means the rows already share the lines and the cost
// is the work itself.
void timeit(const char* kernel, const std::function<void(Encoder&)>& bind, uint32_t gx, uint32_t gy, uint32_t gz,
            uint32_t tx, uint32_t reps, double* ms) {
  auto warm = Gpu::get().begin();
  warm->use(Gpu::get().pipeline(kernel));
  bind(*warm);
  warm->dispatch(gx, gy, gz, tx);
  warm->commit_and_wait();
  auto e = Gpu::get().begin();
  for (uint32_t i = 0; i < reps; ++i) {
    e->use(Gpu::get().pipeline(kernel));
    bind(*e);
    e->dispatch(gx, gy, gz, tx);
  }
  const auto t0 = std::chrono::steady_clock::now();
  e->commit_and_wait();
  *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
}

void test_qsa_timing() {
  const uint32_t NH = 4, HD = 128, RATIO = 4, TOPB = 512, REPS = 40;  // TOPB = 2048 / 4, as Flash-Next's budget
  printf("  qsa_score A/B, ms per dispatch (12 sparse layers per cycle => x12 for ms/C)\n");
  for (uint32_t nb : {1024u, 8192u}) {  // 4K and 32K of context: nb = ctx / ratio
    const uint32_t pos0 = nb * RATIO - 8;
    Buf bk(nb * HD);
    bk.fill(-1, 1);
    for (uint32_t R : {1u, 2u, 4u, 8u}) {
      Buf iq(R * NH * HD), s(R * nb);
      iq.fill(-1, 1);
      const uint32_t nbmax = (pos0 + R) / RATIO;
      struct { uint32_t R, NH, HD, pos0, ratio, topb, s_ld, i_ld; } p{R, NH, HD, pos0, RATIO, TOPB, nbmax, nbmax};
      double m1 = 0;
      timeit("qsa_score", [&](Encoder& e) { e.buf(0, iq.b); e.buf(1, bk.b); e.buf(2, s.b); e.val(3, p); },
             (nbmax + 255) / 256, R, 1, 256, REPS, &m1);
      char fused[48] = " -";
      if (R >= 2) {
        double m2 = 0;
        timeit("qsa_score_rows", [&](Encoder& e) { e.buf(0, iq.b); e.buf(1, bk.b); e.buf(2, s.b); e.val(3, p); },
               (nbmax + 127) / 128, (R + 3) / 4, 1, 64, REPS, &m2);
        if (m2 > 0) snprintf(fused, sizeof fused, "%7.4f (%4.2fx faster)", m2, m1 / m2);
      }
      printf("    nb=%5u R=%u  per-row %7.4f ms  fused %s\n", nb, R, m1, fused);
    }
  }
}

// qsa_score_rows (NI_QSA_ROWFUSE) must agree with qsa_score bit for bit: the fused kernel computes each score by
// the same expression on the same values and only shares the staged key blocks across rows, so a difference here
// means the compiler reassociated something and the change is reorder-class, not bitwise (the logit md5 gate then
// escalates instead of accepting it).
void test_qsa_rows() {
  const uint32_t NH = 4, HD = 128, RATIO = 4, TOPB = 8, pos0 = 4095;  // sparse regime, nb >> 128
  for (uint32_t R : {3u, 4u, 8u}) {
    const uint32_t nbmax = (pos0 + R) / RATIO, ILD = TOPB * RATIO + RATIO;
    Buf iq(R * NH * HD), bk(nbmax * HD), s0(R * nbmax), s1(R * nbmax);
    iq.fill(-1, 1), bk.fill(-1, 1);
    // a tie and a shared block, so a mis-staged sub-chunk shows up as well
    for (uint32_t i = 0; i < HD; ++i) bk.f()[17 * HD + i] = bk.f()[33 * HD + i] = bk.f()[0 * HD + i];
    struct { uint32_t R, NH, HD, pos0, ratio, topb, s_ld, i_ld; } p{R, NH, HD, pos0, RATIO, TOPB, nbmax, ILD};
    run("qsa_score", [&](Encoder& e) { e.buf(0, iq.b); e.buf(1, bk.b); e.buf(2, s0.b); e.val(3, p); },
        (nbmax + 255) / 256, R, 1, 256);
    run("qsa_score_rows", [&](Encoder& e) { e.buf(0, iq.b); e.buf(1, bk.b); e.buf(2, s1.b); e.val(3, p); },
        (nbmax + 127) / 128, (R + 3) / 4, 1, 64);
    double e = 0;
    uint32_t bits = 0, first = ~0u;
    for (uint32_t i = 0; i < R * nbmax; ++i) {
      const float a = s0.f()[i], b = s1.f()[i];
      if (a != b) {
        ++bits;
        if (first == ~0u) first = i;
        e = std::max(e, double(std::fabs(a - b)));
      }
    }
    const std::string at = first == ~0u ? "" : " (first at " + std::to_string(first) + ")";
    printf("  %s: R=%u stray scores %u/%u%s\n", "qsa_score_rows", R, bits, R * nbmax, at.c_str());
    report("qsa_score_rows: bitwise", e, 0);
  }
}

// block scores, the top-k selection (ties: lowest indices) with the tail, attention over the lists
void test_qsa_sparse() {
  const uint32_t R = 3, NH = 4, HD = 128, RATIO = 4, TOPB = 8, pos0 = 97, Hq = 24, Hkv = 2, D = 256, CAP = 128;
  const uint32_t nbmax = (pos0 + R) / RATIO, ILD = TOPB * RATIO + RATIO;
  Buf iq(R * NH * HD), bk(nbmax * HD), score(R * nbmax), idx(R * ILD), cnt(R);
  iq.fill(-1, 1), bk.fill(-1, 1);
  for (uint32_t i = 0; i < HD; ++i) bk.f()[5 * HD + i] = bk.f()[9 * HD + i];  // a tie between blocks 5 and 9
  struct { uint32_t R, NH, HD, pos0, ratio, topb, s_ld, i_ld; } p{R, NH, HD, pos0, RATIO, TOPB, nbmax, ILD};
  run("qsa_score", [&](Encoder& e) { e.buf(0, iq.b); e.buf(1, bk.b); e.buf(2, score.b); e.val(3, p); }, (nbmax + 255) / 256, R, 1, 256);
  run("qsa_select", [&](Encoder& e) { e.buf(0, score.b); e.buf(1, idx.b); e.buf(2, cnt.b); e.val(3, p); }, R, 1, 1, 1024);
  double es = 0;
  int bad_sel = 0;
  std::vector<std::vector<uint32_t>> lists(R);
  for (uint32_t r = 0; r < R; ++r) {
    const uint32_t n = pos0 + r + 1, nb = n / RATIO;
    std::vector<std::pair<double, uint32_t>> sc;
    for (uint32_t j = 0; j < nb; ++j) {
      double a = 0;
      for (uint32_t h = 0; h < NH; ++h) {
        double d = 0;
        for (uint32_t i = 0; i < HD; ++i) d += double(iq.f()[(r * NH + h) * HD + i]) * bk.f()[j * HD + i];
        a += std::max(d, 0.0);
      }
      a /= std::sqrt(double(HD));
      es = std::max(es, std::fabs(a - score.f()[r * nbmax + j]) / std::max(a, 1e-2));
      sc.push_back({double(score.f()[r * nbmax + j]), j});  // select on the GPU's scores (ties as computed)
    }
    std::stable_sort(sc.begin(), sc.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::vector<uint32_t> blocks;
    for (uint32_t i = 0; i < TOPB; ++i) blocks.push_back(sc[i].second);
    std::sort(blocks.begin(), blocks.end());
    std::vector<uint32_t> want;
    for (uint32_t j : blocks)
      for (uint32_t t = 0; t < RATIO; ++t) want.push_back(j * RATIO + t);
    for (uint32_t t = nb * RATIO; t < n; ++t) want.push_back(t);
    std::vector<uint32_t> got(idx.b.as<uint32_t>() + r * ILD, idx.b.as<uint32_t>() + r * ILD + cnt.b.as<uint32_t>()[r]);
    bad_sel += got != want;
    lists[r] = got;
  }
  report("qsa_score", es, 1e-5);
  report("qsa_select: lists (mismatching rows)", bad_sel, 0);
  // attention over the lists, both forms
  Buf q(R * Hq * D), qg(R * Hq * 2 * D), kc(Hkv * CAP * D, 2), vc(Hkv * CAP * D, 2), out(R * Hq * D), po(R * Hq * 64 * D), pml(R * Hq * 64 * 2);
  q.fill(-1, 1), qg.fill(-2, 2);
  for (size_t i = 0; i < kc.n; ++i) kc.b.as<_Float16>()[i] = _Float16(urand(-1, 1)), vc.b.as<_Float16>()[i] = _Float16(urand(-1, 1));
  std::vector<double> ref(R * Hq * D);
  for (uint32_t r = 0; r < R; ++r)
    for (uint32_t h = 0; h < Hq; ++h) {
      const uint32_t kv = h / (Hq / Hkv);
      std::vector<double> s;
      double mx = -1e30;
      for (uint32_t t : lists[r]) {
        double d = 0;
        for (uint32_t i = 0; i < D; ++i) d += double(q.f()[(r * Hq + h) * D + i]) * float(kc.b.as<_Float16>()[(kv * CAP + t) * D + i]);
        s.push_back(d / 16);
        mx = std::max(mx, d / 16);
      }
      double den = 0;
      for (double& x : s) den += (x = std::exp(x - mx));
      for (uint32_t i = 0; i < D; ++i) {
        double o = 0;
        for (size_t k = 0; k < s.size(); ++k) o += s[k] * float(vc.b.as<_Float16>()[(kv * CAP + lists[r][k]) * D + i]);
        ref[(r * Hq + h) * D + i] = o / den * sigm(qg.f()[(r * Hq + h) * 2 * D + D + i]);
      }
    }
  struct { uint32_t Hq, Hkv, chunk, max_seq; float scale; uint32_t nchunks, i_ld; } ap{Hq, Hkv, ILD, CAP, 1.0f / 16, 1, ILD};
  run("attn_sparse_rows", [&](Encoder& e) {
    e.buf(0, q.b); e.buf(1, kc.b); e.buf(2, vc.b); e.buf(3, qg.b); e.buf(4, out.b); e.val(5, ap); e.buf(6, idx.b); e.buf(7, cnt.b);
  }, Hkv, R, 1, 32 * (Hq / Hkv));
  report("attn_sparse_rows", cmp(out.n, [&](size_t i) { return out.f()[i]; }, [&](size_t i) { return ref[i]; }, 1e-2), 2e-3);
  ap.chunk = 16, ap.nchunks = (ILD + 15) / 16;
  run("attn_rows_sparse", [&](Encoder& e) {
    e.buf(0, q.b); e.buf(1, kc.b); e.buf(2, vc.b); e.buf(3, po.b); e.buf(4, pml.b); e.val(5, ap); e.buf(6, idx.b); e.buf(7, cnt.b);
  }, ap.nchunks, Hkv, R, 32 * (Hq / Hkv));
  struct { uint32_t Hq, Hkv, pos0, chunk, max_seq; float scale; uint32_t nchunks; } rp{Hq, Hkv, 0, 16, CAP, 1.0f / 16, ap.nchunks};
  Buf out2(R * Hq * D);
  run("attn_rows_reduce", [&](Encoder& e) { e.buf(0, po.b); e.buf(1, pml.b); e.buf(2, qg.b); e.buf(3, out2.b); e.val(4, rp); }, Hq, R, 1, D);
  report("attn_rows_sparse + reduce", cmp(out2.n, [&](size_t i) { return out2.f()[i]; }, [&](size_t i) { return ref[i]; }, 1e-2), 2e-3);
}

// The fused chain (NI_QSA_FUSE) against the four dispatches it replaces, run on the same state (the chain is the
// reference; no CPU model): ik, idx and cnt byte for byte, bk and score to the last bits (see rel below)
void test_qsa_all() {
  const uint32_t NH = 4, HD = 128, LD = (NH + 1) * HD, ROT = 64, RATIO = 4, TOPB = 8;
  const float theta = 1e7f;
  struct { uint32_t mn; int32_t delta; uint32_t sec1, sec2; } mr{0, 0, 33, 30};  // text rows: positions as they are
  // R9: mid-block start, two re-derived blocks, two completing rows; R1-open / R1-close: the decode shape with a row
  // that re-derives nothing / completes a block; R9-dense: every block re-derived (jold = 0), the dense select path
  const uint32_t cfg[4][2] = {{9, 97}, {1, 4093}, {1, 4095}, {9, 3}};
  const char* tag[4] = {"R9", "R1-open", "R1-close", "R9-dense"};
  for (int ci = 0; ci < 4; ++ci) {
    const uint32_t R = cfg[ci][0], pos0 = cfg[ci][1];
    const uint32_t nbmax = (pos0 + R) / RATIO, ILD = TOPB * RATIO + RATIO;
    Buf iqk(R * LD), wq(HD), wk(HD), iq(R * NH * HD);
    Buf ik((pos0 + R) * HD, 2), bk((nbmax + 1) * HD), score(R * nbmax), idx(R * ILD), cnt(R);
    iqk.fill(-3, 3), wq.fill(0.5f, 1.5f), wk.fill(0.5f, 1.5f);
    for (size_t i = 0; i < ik.n; ++i) ik.b.as<_Float16>()[i] = _Float16(urand(-3, 3));  // history + future rows
    bk.fill(-9, 9);  // stale blocks: only the ones these dispatches complete may change, on either side
    auto init_out = [&] {
      for (size_t i = 0; i < score.n; ++i) score.f()[i] = 0;
      for (size_t i = 0; i < R * ILD; ++i) idx.b.as<uint32_t>()[i] = 0xffffffffu;
      for (uint32_t i = 0; i < R; ++i) cnt.b.as<uint32_t>()[i] = 0;
    };
    init_out();
    const std::vector<uint16_t> ik0(ik.b.as<uint16_t>(), ik.b.as<uint16_t>() + ik.n);
    const std::vector<float> bk0(bk.f(), bk.f() + bk.n);
    QsaP p{R, NH, HD, ROT, pos0, RATIO, LD, 1e-6f, theta};
    struct { uint32_t R, NH, HD, pos0, ratio, topb, s_ld, i_ld; } ps{R, NH, HD, pos0, RATIO, TOPB, nbmax, ILD};
    run("qsa_prep", [&](Encoder& e) {
      e.buf(0, iqk.b); e.buf(1, wq.b); e.buf(2, iq.b); e.buf(3, ik.b); e.val(4, p); e.buf(5, iq.b); e.val(6, mr);
    }, NH + 1, R, 1, HD);
    run("qsa_block", [&](Encoder& e) {
      e.buf(0, ik.b); e.buf(1, wk.b); e.buf(2, bk.b); e.val(3, p); e.buf(4, iq.b); e.val(5, mr);
    }, R, 1, 1, HD);
    run("qsa_score", [&](Encoder& e) { e.buf(0, iq.b); e.buf(1, bk.b); e.buf(2, score.b); e.val(3, ps); },
        (nbmax + 255) / 256, R, 1, 256);
    run("qsa_select", [&](Encoder& e) { e.buf(0, score.b); e.buf(1, idx.b); e.buf(2, cnt.b); e.val(3, ps); }, R, 1, 1, 1024);
    const std::vector<uint16_t> ikc(ik.b.as<uint16_t>(), ik.b.as<uint16_t>() + ik.n);
    const std::vector<float> bkc(bk.f(), bk.f() + bk.n), scorec(score.f(), score.f() + score.n);
    const std::vector<uint32_t> idxc(idx.b.as<uint32_t>(), idx.b.as<uint32_t>() + size_t(R) * ILD);
    const std::vector<uint32_t> cntc(cnt.b.as<uint32_t>(), cnt.b.as<uint32_t>() + R);
    // restore the in-place inputs and the outputs, then run the fused chain on the identical state
    std::copy(ik0.begin(), ik0.end(), ik.b.as<uint16_t>());
    std::copy(bk0.begin(), bk0.end(), bk.f());
    init_out();
    run("qsa_all", [&](Encoder& e) {
      e.buf(0, iqk.b); e.buf(1, wq.b); e.buf(2, wk.b); e.buf(3, ik.b); e.buf(4, bk.b); e.buf(5, score.b);
      e.buf(6, idx.b); e.buf(7, cnt.b); e.val(8, p); e.buf(9, iq.b); e.val(10, mr); e.val(11, ps);
    }, R, 1, 1, 1024);
    auto diff = [](const void* a, const void* b, size_t bytes) {
      size_t d = 0;
      for (const uint8_t *x = (const uint8_t*)a, *y = (const uint8_t*)b, *end = x + bytes; x != end; ++x, ++y) d += *x != *y;
      return double(d);
    };
    // max |a - b| / max |b| over a buffer: the fused chain's block keys (and the scores read from them) can differ
    // from qsa_block's in the last bits (#574: <= 2.4e-7 on keys up to ~9, in the pass-through dims as well as the
    // rope'd ones, both paths equally close to an fp64 reference): fast math may contract or reassociate the same
    // source expressions differently in two kernels. A stale or mis-indexed block is off by O(1)
    auto rel = [](const float* a, const float* b, size_t n) {
      double d = 0, m = 0;
      for (size_t i = 0; i < n; ++i) d = std::max(d, double(std::fabs(a[i] - b[i]))), m = std::max(m, double(std::fabs(b[i])));
      return d / m;
    };
    char nm[64];
    snprintf(nm, sizeof nm, "qsa_all %s: ik (bytes)", tag[ci]);
    report(nm, diff(ik.b.as<uint16_t>(), ikc.data(), ik.n * 2), 0);
    snprintf(nm, sizeof nm, "qsa_all %s: bk", tag[ci]);
    report(nm, rel(bk.f(), bkc.data(), bk.n), 1e-5);
    snprintf(nm, sizeof nm, "qsa_all %s: score", tag[ci]);
    report(nm, rel(score.f(), scorec.data(), score.n), 1e-5);
    snprintf(nm, sizeof nm, "qsa_all %s: idx (bytes)", tag[ci]);
    report(nm, diff(idx.b.as<uint32_t>(), idxc.data(), size_t(R) * ILD * 4), 0);
    snprintf(nm, sizeof nm, "qsa_all %s: cnt (bytes)", tag[ci]);
    report(nm, diff(cnt.b.as<uint32_t>(), cntc.data(), R * 4), 0);
  }
}

// the 3-row half-precision verify GEMV on the Flash-Next shapes against fp64 (qmvht_t3_r4; Apple10: qmvhts_t3_r4, as
// Qwen35 picks it there)
void test_qmvht() {
  struct Shape { uint32_t N, K; };
  const bool a10 = Gpu::get().apple10();
  const char* kn = getenv("FN_QMVHT") ? getenv("FN_QMVHT") : a10 ? "qmvhts_t3_r4" : "qmvht_t3_r4";
  const uint32_t R = uint32_t(strstr(kn, "_t")[2] - '0');
  // qmvht itself returns NaN under fast math on Apple10 (unused there): printed, not failed. Its half products are
  // 1-5% off fp64 on these random inputs (relative to max(|y|, 0.1); the same with NI_MATH_SAFE=1), so only a
  // non-finite row fails
  const bool unused = a10 && !strncmp(kn, "qmvht_", 6);
  for (Shape sh : {Shape{320, 10240}, Shape{10240, 320}, Shape{48, 2560}, Shape{2560, 6144}}) {
    const uint32_t N = sh.N, K = sh.K, G = K / 64;
    Buf w(size_t(N) * K / 8), sc(size_t(N) * G, 2), bi(size_t(N) * G, 2), x(R * K), y(R * N);
    for (size_t i = 0; i < w.n; ++i) w.b.as<uint32_t>()[i] = uint32_t(rng());
    sc.fill_bf16(0.001f, 0.01f), bi.fill_bf16(-0.05f, 0.0f), x.fill(-1, 1);
    struct { uint32_t N, K, R, x_ld, y_ld, add; } p{N, K, R, K, N, 0};
    run(kn, [&](Encoder& e) {
      e.buf(0, w.b); e.buf(1, sc.b); e.buf(2, bi.b); e.buf(3, x.b); e.buf(4, y.b); e.val(5, p);
    }, (N + 31) / 32, 1, 1, 256);
    double err = 0;
    for (uint32_t r = 0; r < R; ++r)
      for (uint32_t n = 0; n < N; ++n) {
        double acc = 0;
        for (uint32_t k = 0; k < K; ++k) {
          const uint32_t q = (w.b.as<uint32_t>()[(size_t(n) * K + k) / 8] >> (4 * (k % 8))) & 15;
          acc += (bf(sc, n * G + k / 64) * q + bf(bi, n * G + k / 64)) * x.f()[r * K + k];
        }
        const double d = std::fabs(acc - y.f()[r * N + n]) / std::max(std::fabs(acc), 0.1);
        err = std::isfinite(d) ? std::max(err, d) : INFINITY;
        if (getenv("FN_DEBUG") && n < 2) printf("  r %u n %u: %f vs %f\n", r, n, acc, y.f()[r * N + n]);
      }
    char name[96];
    snprintf(name, sizeof name, "%s %ux%u%s", kn, N, K, unused ? " (unused on Apple10)" : "");
    if (unused) printf("%-34s max rel err %.2e\n", name, err);
    else if (std::isfinite(err)) printf("%-34s max rel err %.2e (finite)\n", name, err);
    else report(name, err, 1.0);
  }
}

// the 512-expert router top-k against a CPU softmax + top-k (both renormalizations)
void test_topk512() {
  const uint32_t R = 3, E = 512, K = 10;
  for (uint32_t norm = 0; norm < 2; ++norm) {
    Buf lg(R * E), sh(R), idx(R * K), wts(R * (K + 1));
    lg.fill(-4, 4), sh.fill(-2, 2);
    struct { uint32_t E, k, norm; } p{E, K, norm};
    run("moe_topk512", [&](Encoder& e) { e.buf(0, lg.b); e.buf(1, sh.b); e.buf(2, idx.b); e.buf(3, wts.b); e.val(4, p); }, R, 1, 1, 32);
    int bad = 0;
    double err = 0;
    for (uint32_t r = 0; r < R; ++r) {
      std::vector<std::pair<float, uint32_t>> v;
      double mx = -1e30, tot = 0;
      for (uint32_t i = 0; i < E; ++i) v.push_back({lg.f()[r * E + i], i}), mx = std::max(mx, double(lg.f()[r * E + i]));
      for (auto& x : v) tot += std::exp(x.first - mx);
      std::stable_sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.first > b.first; });
      double sel = 0;
      for (uint32_t q = 0; q < K; ++q) sel += std::exp(v[q].first - mx);
      for (uint32_t q = 0; q < K; ++q) {
        bad += idx.b.as<uint32_t>()[r * K + q] != v[q].second;
        const double w = std::exp(v[q].first - mx) / (norm ? sel : tot);
        err = std::max(err, std::fabs(w - wts.f()[r * (K + 1) + q]) / w);
      }
      err = std::max(err, std::fabs(sigm(sh.f()[r]) - wts.f()[r * (K + 1) + K]));
    }
    report(norm ? "moe_topk512 (renormalized): ids" : "moe_topk512: ids", bad, 0);
    report(norm ? "moe_topk512 (renormalized): weights" : "moe_topk512: weights", err, 1e-5);
  }
}

// the weight-format kernels (wfmt.metal) on Qwen3.8-Flash-Next's 6-bit shapes against fp64: the decode GEMV
// (qmvg_f6g64_r2) and the prefill GEMM (qrf_f6g64_*, half inputs)
void test_wfmt6() {
  struct Shape { uint32_t N, K; };
  const int bits = 6;
  for (Shape sh : {Shape{320, 10240}, Shape{10240, 320}, Shape{12288, 2560}, Shape{2560, 6144}}) {
    const uint32_t N = sh.N, K = sh.K, G = K / 64;
    Buf W(size_t(N) * K * bits / 32), Sc(size_t(N) * G, 2), Bi(size_t(N) * G, 2);
    for (size_t i = 0; i < W.n; ++i) W.b.as<uint32_t>()[i] = uint32_t(rng());
    Sc.fill_bf16(0.0005f, 0.002f), Bi.fill_bf16(-0.06f, 0.0f);
    auto deq = [&](uint32_t n, uint32_t k) {
      const uint64_t bit = (uint64_t(n) * K + k) * uint64_t(bits);
      const uint32_t* w = W.b.as<uint32_t>();
      uint64_t v = w[bit / 32];
      if (bit % 32 + bits > 32) v |= uint64_t(w[bit / 32 + 1]) << 32;
      return double(bf(Sc, n * G + k / 64)) * double(uint32_t(v >> (bit % 32)) & 63u) + bf(Bi, n * G + k / 64);
    };
    // decode GEMV, one row
    {
      Buf x(K), y(N);
      x.fill(-1, 1);
      struct { uint32_t N, K, xs, ys; float eps; } p{N, K, K, N, 1e-6f};
      run("qmvg_f6g64_r2", [&](Encoder& e) {
        e.buf(0, W.b); e.buf(1, Sc.b); e.buf(2, Bi.b); e.buf(3, x.b); e.buf(4, y.b); e.val(5, p); e.buf(6, y.b); e.buf(7, y.b);
      }, (N + 15) / 16, 1, 1, 256);
      double err = 0;
      for (uint32_t n = 0; n < N; n += (N > 2048 ? 7 : 1)) {
        double a = 0;
        for (uint32_t k = 0; k < K; ++k) a += deq(n, k) * x.f()[k];
        err = std::max(err, std::fabs(a - y.f()[n]) / std::max(std::fabs(a), 0.05));
      }
      char name[64];
      snprintf(name, sizeof name, "qmvg_f6g64_r2 %ux%u", N, K);
      report(name, err, 1e-4);
    }
    // prefill GEMM, M rows of half inputs (FN_ONEHOT: row m = the unit vector of input 7m + 3)
    for (uint32_t M : {5u, 37u, 160u}) {
      Buf xh(size_t(M) * K, 2), y(size_t(M) * N);
      for (size_t i = 0; i < xh.n; ++i) xh.b.as<_Float16>()[i] = _Float16(getenv("FN_ONEHOT") ? 0.0f : urand(-1, 1));
      if (getenv("FN_ONEHOT"))
        for (uint32_t m = 0; m < M; ++m) xh.b.as<_Float16>()[size_t(m) * K + (7 * m + 3) % K] = _Float16(1.0f);
      std::string n = "qrf_f6g64";
      uint32_t bm = 32, bn = 128, th = 128;
      if (M <= 8) n += "_m8_n64", bm = 8, bn = 64, th = 64;
      else if (M < 64 || N <= 64) n += "_m16_n64", bm = 16, bn = 64;
      else n += "_m32_n128";
      struct { uint32_t M, N, K, x_ld, y_ld, add, gs, ws, ss; } p{M, N, K, K, N, 0, 64, 0, 0};
      run(n.c_str(), [&](Encoder& e) {
        e.buf(0, xh.b); e.buf(1, W.b); e.buf(2, Sc.b); e.buf(3, Bi.b); e.buf(4, y.b); e.val(5, p);
      }, (N + bn - 1) / bn, (M + bm - 1) / bm, 1, th);
      double err = 0;
      for (uint32_t m = 0; m < M; m += (M > 40 ? 13 : 1))
        for (uint32_t nn = 0; nn < N; nn += (N > 2048 ? 11 : 1)) {
          double a = 0;
          for (uint32_t k = 0; k < K; ++k) a += deq(nn, k) * double(float(xh.b.as<_Float16>()[size_t(m) * K + k]));
          err = std::max(err, std::fabs(a - y.f()[size_t(m) * N + nn]) / std::max(std::fabs(a), 0.05));
        }
      char name[64];
      snprintf(name, sizeof name, "%s %ux%u M=%u", n.c_str(), N, K, M);
      report(name, err, 2e-3);
      if (getenv("FN_ONEHOT") && M == 5)
        for (uint32_t m = 0; m < 2; ++m)
          for (uint32_t nn = 0; nn < 3; ++nn)
            printf("   m %u n %u: got %.6f want %.6f\n", m, nn, y.f()[size_t(m) * N + nn], deq(nn, (7 * m + 3) % K));
    }
  }
}


// Verify kernels (qwen4v.metal) against the one-row kernels decode runs, bit for bit: the multi-row folded gated
// residual (hc_down_skr + hc_up_mixr) against R one-row threadgroups of hc_down_sk2 + hc_up_mix2
void test_hc_rows(int bits) {
  const uint32_t D = 768, S = 4, SD = S * D, RK = 128, G = 64, KC = SD / 1024;
  for (uint32_t R = 1; R <= 4; ++R)
    for (int mode = 0; mode < 2; ++mode) {
      Buf X(R * SD), y(R * D), injp(R * S), w(SD), winj(S * SD, 2);
      Buf Wd(size_t(RK) * SD * bits / 32), Sd(RK * SD / G, 2), Bd(RK * SD / G, 2);
      Buf Wu(size_t(SD) * RK * bits / 32), Su(SD * RK / G, 2), Bu(SD * RK / G, 2);
      X.fill(-2, 2), y.fill(-1, 1), injp.fill(0, 2), w.fill(0.5f, 1.5f), winj.fill_bf16(-0.05f, 0.05f);
      for (size_t i = 0; i < Wd.n; ++i) Wd.b.as<uint32_t>()[i] = uint32_t(rng());
      for (size_t i = 0; i < Wu.n; ++i) Wu.b.as<uint32_t>()[i] = uint32_t(rng());
      const float sc = bits == 4 ? 0.004f : 0.0003f;
      Sd.fill_bf16(sc / 2, sc), Bd.fill_bf16(-sc * (1 << bits) / 2, 0), Su.fill_bf16(sc, 2 * sc), Bu.fill_bf16(-sc * (1 << bits), 0);
      struct Out {
        Buf X, y, u, inj, part, X2, stat;
        Out(uint32_t R, uint32_t SD, uint32_t D, uint32_t S, uint32_t KC, uint32_t RK)
            : X(R * SD), y(R * D), u(R * D), inj(R * S), part(KC * R * RK * 2), X2(R * SD), stat(R * KC * 10) {}
      };
      Out a(R, SD, D, S, KC, RK), b(R, SD, D, S, KC, RK), c4(R, SD, D, S, KC, RK);
      for (Out* o : {&a, &b, &c4}) {
        memcpy(o->X.f(), X.f(), R * SD * 4);
        memcpy(o->y.f(), y.f(), R * D * 4);
      }
      struct { uint32_t R, D, S, rank, kc, u_ld, has_y, has_inj; float eps; } p{R, D, S, RK, KC, D, uint32_t(mode), uint32_t(mode), 1e-6f};
      const std::string sfx = "_b" + std::to_string(bits);
      auto bind_down = [&](Out& o) {
        return [&](Encoder& e) {
          e.buf(0, o.X.b); e.buf(1, o.y.b); e.buf(2, injp.b); e.buf(3, w.b); e.buf(4, winj.b); e.buf(5, Wd.b);
          e.buf(6, Sd.b); e.buf(7, Bd.b); e.buf(8, o.part.b); e.buf(9, o.X2.b); e.buf(10, o.stat.b); e.val(11, p);
        };
      };
      auto bind_up = [&](Out& o) {
        return [&](Encoder& e) {
          e.buf(0, o.part.b); e.buf(1, o.X2.b); e.buf(2, o.stat.b); e.buf(3, w.b); e.buf(4, Wu.b); e.buf(5, Su.b);
          e.buf(6, Bu.b); e.buf(7, o.u.b); e.buf(8, o.X.b); e.buf(9, o.y.b); e.buf(10, o.inj.b); e.val(11, p);
        };
      };
      run(("hc_down_sk2" + sfx).c_str(), bind_down(a), RK / 8, KC, R, 256);
      run(("hc_up_mix2" + sfx).c_str(), bind_up(a), D / 16, R, 1, 256);
      constexpr int nr = 2, cpt = 2;
      run(("hc_down_sk3_n" + std::to_string(nr) + sfx).c_str(), bind_down(b), RK / (8 * nr), KC, 1, 256);
      run(("hc_up_mix3_c" + std::to_string(cpt) + sfx).c_str(), bind_up(b), D / (16 * cpt), 1, 1, 256);
      run(("hc_down_sk3_n" + std::to_string(nr) + sfx).c_str(), bind_down(c4), RK / (8 * nr), KC, 1, 256);
      run(("hc_up_mix4_c4" + sfx).c_str(), bind_up(c4), D / (16 * 4), 1, 1, 256);
      auto diff = [](Buf& x, Buf& z) {
        size_t bad = 0;
        double mx = 0;
        for (size_t i = 0; i < x.n; ++i)
          if (memcmp(x.f() + i, z.f() + i, 4)) ++bad, mx = std::max(mx, double(std::fabs(x.f()[i] - z.f()[i])));
        if (bad && getenv("FN_VERBOSE")) printf("   %zu of %zu differ (max %.3g)\n", bad, x.n, mx);
        return bad ? 1.0 : 0.0;
      };
      const double e = diff(a.X, b.X) + diff(a.y, b.y) + diff(a.u, b.u) + (mode ? diff(a.inj, b.inj) : 0.0) +
                       diff(a.part, b.part) + diff(a.X2, b.X2) + diff(a.stat, b.stat);
      char nm[96];
      snprintf(nm, sizeof nm, "hc_*3%s R=%u %s == one-row", sfx.c_str(), R, mode ? "inject" : "plain");
      report(nm, e, 0);
      const double e4 = diff(a.X, c4.X) + diff(a.y, c4.y) + diff(a.u, c4.u) + (mode ? diff(a.inj, c4.inj) : 0.0) +
                        diff(a.part, c4.part) + diff(a.X2, c4.X2) + diff(a.stat, c4.stat);
      snprintf(nm, sizeof nm, "hc_up_mix4%s R=%u %s == one-row", sfx.c_str(), R, mode ? "inject" : "plain");
      report(nm, e4, 0);
    }
}

// The multi-segment row GEMV (qmvs_multi: 4- and 8-bit segments sharing x) against qmv2_multi_r2 (decode) per row
void test_qmvs_rows() {
  const uint32_t K = 1536, G = 64;
  const uint32_t Ns[4] = {96, 40, 8, 4}, bits[4] = {4, 4, 8, 8};
  std::vector<Buf*> Wq, Sq, Bq;
  for (int i = 0; i < 4; ++i) {
    Wq.push_back(new Buf(size_t(Ns[i]) * K * bits[i] / 32));
    Sq.push_back(new Buf(Ns[i] * K / G, 2));
    Bq.push_back(new Buf(Ns[i] * K / G, 2));
    for (size_t j = 0; j < Wq[i]->n; ++j) Wq[i]->b.as<uint32_t>()[j] = uint32_t(rng());
    Sq[i]->fill_bf16(0.001f, 0.01f), Bq[i]->fill_bf16(-0.05f, 0.0f);
  }
  for (uint32_t R = 2; R <= 8; ++R) {
    Buf x(R * K);
    x.fill(-2, 2);
    std::vector<Buf*> ya, yb;
    for (int i = 0; i < 4; ++i) ya.push_back(new Buf(R * Ns[i])), yb.push_back(new Buf(R * Ns[i]));
    // decode kernel, one row at a time
    for (uint32_t r = 0; r < R; ++r) {
      struct { uint32_t K, nseg; float eps; uint32_t norm, write_xn, N[4], tg_begin[5], bits[4]; } p{};
      p.K = K, p.nseg = 4;
      uint32_t tg = 0;
      for (int i = 0; i < 4; ++i) p.N[i] = Ns[i], p.bits[i] = bits[i], p.tg_begin[i] = tg, tg += (Ns[i] + 15) / 16;
      p.tg_begin[4] = tg;
      run("qmv2_multi_r2", [&](Encoder& e) {
        for (int i = 0; i < 4; ++i) {
          e.buf(i * 4, Wq[i]->b); e.buf(i * 4 + 1, Sq[i]->b); e.buf(i * 4 + 2, Bq[i]->b);
          e.buf(i * 4 + 3, ya[i]->b, size_t(r) * Ns[i] * 4);
        }
        e.buf(16, x.b, size_t(r) * K * 4); e.val(17, p); e.buf(18, x.b); e.buf(19, x.b);
      }, tg, 1, 1, 256);
    }
    struct { uint32_t K, R, nseg, x_ld, N[4], tg_begin[5], bits[4], y_ld[4], add[4]; } q{};
    q.K = K, q.R = R, q.nseg = 4, q.x_ld = K;
    uint32_t tg = 0;
    for (int i = 0; i < 4; ++i) q.N[i] = Ns[i], q.bits[i] = bits[i], q.y_ld[i] = Ns[i], q.tg_begin[i] = tg, tg += (Ns[i] + 7) / 8;
    q.tg_begin[4] = tg;
    run("qmvs_multi", [&](Encoder& e) {
      for (int i = 0; i < 4; ++i) {
        e.buf(i * 4, Wq[i]->b); e.buf(i * 4 + 1, Sq[i]->b); e.buf(i * 4 + 2, Bq[i]->b); e.buf(i * 4 + 3, yb[i]->b);
      }
      e.buf(16, x.b); e.val(17, q);
    }, R, tg, 1, 64);
    double e = 0;
    for (int i = 0; i < 4; ++i) e += memcmp(ya[i]->f(), yb[i]->f(), size_t(R) * Ns[i] * 4) == 0 ? 0 : 1;
    char nm[64];
    snprintf(nm, sizeof nm, "qmvs_multi R=%u == qmv2_multi_r2", R);
    report(nm, e, 0);
    for (int i = 0; i < 4; ++i) memset(yb[i]->f(), 0, size_t(R) * Ns[i] * 4);
    run("qmvp_multi", [&](Encoder& e) {
      for (int i = 0; i < 4; ++i) {
        e.buf(i * 4, Wq[i]->b); e.buf(i * 4 + 1, Sq[i]->b); e.buf(i * 4 + 2, Bq[i]->b); e.buf(i * 4 + 3, yb[i]->b);
      }
      e.buf(16, x.b); e.val(17, q);
    }, tg, 1, 1, 64 * ((R + 1) / 2));
    // (two token rows per simdgroup: the compiler's fast math orders the sums differently.) The error is
    // max |a - b| / max |b| per segment: the sum-order noise is absolute (~1e-5 on outputs up to ~100), so dividing
    // by each element (cmp) let the smallest output of a draw decide the gate (#576)
    e = 0;
    for (int i = 0; i < 4; ++i) {
      double d = 0, m = 0;
      for (size_t j = 0; j < size_t(R) * Ns[i]; ++j)
        d = std::max(d, std::fabs(double(yb[i]->f()[j]) - ya[i]->f()[j])), m = std::max(m, std::fabs(double(ya[i]->f()[j])));
      e = std::max(e, d / m);
    }
    snprintf(nm, sizeof nm, "qmvp_multi R=%u ~ qmv2_multi_r2", R);
    report(nm, e, 1e-5);
    for (int i = 0; i < 4; ++i) delete ya[i], delete yb[i];
  }
  for (int i = 0; i < 4; ++i) delete Wq[i], delete Sq[i], delete Bq[i];
}

// The verify MoE (moe4_unique + moe4_gate_up + moe4_down) against the decode kernels (moe_gate_up3_r2 + moe_down2_w)
// run row by row on the same top-k, bit for bit; small = the moe4_down_s instance only (shader validation)
void test_moe4_rows(bool small) {
  const uint32_t D = 512, I = 128, SI = 128, k = 10, E = 24, G = 64;
  Buf gw(size_t(E) * I * D / 8), gs(E * I * D / G, 2), gb(E * I * D / G, 2);
  Buf uw(size_t(E) * I * D / 8), us(E * I * D / G, 2), ub(E * I * D / G, 2);
  Buf dw(size_t(E) * D * I / 8), ds(E * D * I / G, 2), db(E * D * I / G, 2);
  Buf sgw(size_t(SI) * D / 8), sgs(SI * D / G, 2), sgb(SI * D / G, 2);
  Buf suw(size_t(SI) * D / 8), sus(SI * D / G, 2), sub(SI * D / G, 2);
  Buf sdw(size_t(D) * SI / 8), sds(D * SI / G, 2), sdb(D * SI / G, 2);
  for (Buf* b : {&gw, &uw, &dw, &sgw, &suw, &sdw})
    for (size_t i = 0; i < b->n; ++i) b->b.as<uint32_t>()[i] = uint32_t(rng());
  for (Buf* b : {&gs, &us, &ds, &sgs, &sus, &sds}) b->fill_bf16(0.002f, 0.01f);
  for (Buf* b : {&gb, &ub, &db, &sgb, &sub, &sdb}) b->fill_bf16(-0.06f, 0.0f);
  struct { uint32_t D, I, SI, k; float eps; } mp{D, I, SI, k, 1e-6f};
  for (uint32_t R = 1; R <= 4; ++R) {
    Buf x(R * D), idx(R * k), wts(R * (k + 1)), ya(R * D), yb(R * D), h1((k + 1) * I), hR(R * (k + 1) * I);
    Buf ue(64), umask(64), uslot(64), un(4), dummy(64);
    x.fill(-2, 2), wts.fill(0.0f, 0.3f), ya.fill(-1, 1);
    memcpy(yb.f(), ya.f(), R * D * 4);
    for (uint32_t r = 0; r < R; ++r) {  // distinct experts per row, overlapping across rows
      std::vector<uint32_t> pool(E);
      for (uint32_t i = 0; i < E; ++i) pool[i] = i;
      std::shuffle(pool.begin(), pool.end(), rng);
      for (uint32_t s2 = 0; s2 < k; ++s2) idx.b.as<uint32_t>()[r * k + s2] = pool[s2];
    }
    if (!small) {
      for (uint32_t r = 0; r < R; ++r) {
        uint32_t Ev = E;
        run("moe_gate_up3_r2", [&](Encoder& e) {
          e.buf(0, gw.b); e.buf(1, gs.b); e.buf(2, gb.b); e.buf(3, uw.b); e.buf(4, us.b); e.buf(5, ub.b);
          e.buf(6, sgw.b); e.buf(7, sgs.b); e.buf(8, sgb.b); e.buf(9, suw.b); e.buf(10, sus.b); e.buf(11, sub.b);
          e.buf(12, x.b, size_t(r) * D * 4); e.buf(13, dummy.b); e.buf(14, dummy.b); e.buf(15, h1.b); e.val(16, mp);
          e.buf(17, idx.b, size_t(r) * k * 4); e.buf(18, wts.b, size_t(r) * (k + 1) * 4); e.val(19, Ev);
        }, I / 16, k + 1, 1, 256);
        run("moe_down2_w", [&](Encoder& e) {
          e.buf(0, dw.b); e.buf(1, ds.b); e.buf(2, db.b); e.buf(3, sdw.b); e.buf(4, sds.b); e.buf(5, sdb.b);
          e.buf(6, h1.b); e.buf(7, idx.b, size_t(r) * k * 4); e.buf(8, wts.b, size_t(r) * (k + 1) * 4);
          e.buf(9, ya.b, size_t(r) * D * 4); e.val(10, mp);
        }, D / 16, 1, 1, 256);
      }
    }
    const struct { uint32_t R, k; } rk{R, k};
    run("moe4_unique", [&](Encoder& e) {
      e.buf(0, idx.b); e.buf(1, ue.b); e.buf(2, umask.b); e.buf(3, uslot.b); e.buf(4, un.b); e.val(5, rk);
    }, 1, 1, 1, 32);
    run("moe4_gate_up", [&](Encoder& e) {
      e.buf(0, gw.b); e.buf(1, gs.b); e.buf(2, gb.b); e.buf(3, uw.b); e.buf(4, us.b); e.buf(5, ub.b);
      e.buf(6, sgw.b); e.buf(7, sgs.b); e.buf(8, sgb.b); e.buf(9, suw.b); e.buf(10, sus.b); e.buf(11, sub.b);
      e.buf(12, x.b); e.buf(13, hR.b); e.val(14, mp); e.buf(15, ue.b); e.buf(16, umask.b); e.buf(17, uslot.b);
      e.buf(18, un.b); e.val(19, R);
    }, R, I / 16, R * k + 1, 256);
    run(small ? "moe4_down_s" : "moe4_down", [&](Encoder& e) {
      e.buf(0, dw.b); e.buf(1, ds.b); e.buf(2, db.b); e.buf(3, sdw.b); e.buf(4, sds.b); e.buf(5, sdb.b);
      e.buf(6, hR.b); e.buf(7, idx.b); e.buf(8, wts.b); e.buf(9, yb.b); e.val(10, mp);
    }, R, D / 16, 1, 256);
    char nm[64];
    if (small) {
      bool finite = true;
      for (size_t i = 0; i < yb.n; ++i) finite = finite && std::isfinite(yb.f()[i]);
      snprintf(nm, sizeof nm, "moe4 R=%u (validation run) finite", R);
      report(nm, finite ? 0.0 : 1.0, 0);
    } else {
      snprintf(nm, sizeof nm, "moe4 R=%u == decode MoE per row", R);
      report(nm, memcmp(ya.f(), yb.f(), size_t(R) * D * 4) == 0 ? 0.0 : 1.0, 0);
    }
  }
}

// The row kernels (qmvs_multi / qmvs_multi_f6g64, one segment) against the decode kernels Flash-Next runs for single
// matrices: qmv3_b4_r2 (K > 4096: GDN out), qmv2_b4_g64_r4 (the LM head), qmvg_f6g64_r2 (attention o), per row
void test_qmvs_single() {
  struct Case { const char* dec; int bits; uint32_t N, K, rows, nsg; };
  const Case cs[] = {{"qmv3_b4_r2", 4, 96, 6144, 2, 8}, {"qmv2_b4_g64_r4", 4, 128, 2560, 4, 4}, {"qmvg_f6g64_r2", 6, 64, 3072, 2, 8},
                     {"qmv3_b8_r2", 8, 48, 6144, 2, 8}};
  for (const Case& c : cs) {
    const uint32_t G = 64, R = 4;
    Buf W(size_t(c.N) * c.K * c.bits / 32 + 4), S(c.N * c.K / G, 2), B(c.N * c.K / G, 2), x(R * c.K), ya(R * c.N), yb(R * c.N);
    for (size_t j = 0; j < W.n; ++j) W.b.as<uint32_t>()[j] = uint32_t(rng());
    S.fill_bf16(0.001f, 0.01f), B.fill_bf16(-0.05f, 0.0f), x.fill(-2, 2);
    for (uint32_t r = 0; r < R; ++r) {
      struct { uint32_t N, K, xs, ys; float eps; } q{c.N, c.K, c.K, c.N, 1e-6f};
      run(c.dec, [&](Encoder& e) {
        e.buf(0, W.b); e.buf(1, S.b); e.buf(2, B.b); e.buf(3, x.b, size_t(r) * c.K * 4); e.buf(4, ya.b, size_t(r) * c.N * 4);
        e.val(5, q); e.buf(6, ya.b); e.buf(7, ya.b);
      }, (c.N + c.rows * c.nsg - 1) / (c.rows * c.nsg), 1, 1, 32 * c.nsg);
    }
    struct { uint32_t K, R, nseg, x_ld, N[4], tg_begin[5], bits[4], y_ld[4], add[4]; } q{};
    q.K = c.K, q.R = R, q.nseg = 1, q.x_ld = c.K, q.N[0] = c.N, q.bits[0] = uint32_t(c.bits), q.y_ld[0] = c.N;
    q.tg_begin[1] = (c.N + 7) / 8;
    run(c.bits == 6 ? "qmvs_multi_f6g64" : "qmvs_multi", [&](Encoder& e) {
      for (int i = 0; i < 4; ++i) { e.buf(i * 4, W.b); e.buf(i * 4 + 1, S.b); e.buf(i * 4 + 2, B.b); e.buf(i * 4 + 3, yb.b); }
      e.buf(16, x.b); e.val(17, q);
    }, R, q.tg_begin[1], 1, 64);
    char nm[64];
    snprintf(nm, sizeof nm, "qmvs R=4 == %s", c.dec);
    report(nm, memcmp(ya.f(), yb.f(), size_t(R) * c.N * 4) == 0 ? 0.0 : 1.0, 0);
  }
}


// hc-time: the folded gated residual at Flash-Next's shape (D 2560, S 4, rank 320, 6-bit) for R = 1..4 rows, over
// NL distinct weight sets (so that the weights stream from DRAM as in a step, not from the caches): per variant
// reps x NL down+up pairs behind one commit, best of 3; then every variant's outputs against the one-row kernels
void test_hc_timing() {
  const uint32_t D = 2560, S = 4, SD = S * D, RK = 320, G = 64, KC = SD / 1024, NL = 96, bits = 6;
  const int reps = getenv("FN_REPS") ? atoi(getenv("FN_REPS")) : 4;
  std::vector<Buf*> Wd, Sd, Bd, Wu, Su, Bu;
  for (uint32_t l = 0; l < NL; ++l) {
    Wd.push_back(new Buf(size_t(RK) * SD * bits / 32)); Sd.push_back(new Buf(RK * SD / G, 2)); Bd.push_back(new Buf(RK * SD / G, 2));
    Wu.push_back(new Buf(size_t(SD) * RK * bits / 32)); Su.push_back(new Buf(SD * RK / G, 2)); Bu.push_back(new Buf(SD * RK / G, 2));
    for (size_t i = 0; i < Wd[l]->n; ++i) Wd[l]->b.as<uint32_t>()[i] = uint32_t(rng());
    for (size_t i = 0; i < Wu[l]->n; ++i) Wu[l]->b.as<uint32_t>()[i] = uint32_t(rng());
    const float sc = 0.0003f;
    Sd[l]->fill_bf16(sc / 2, sc), Bd[l]->fill_bf16(-sc * 32, 0), Su[l]->fill_bf16(sc, 2 * sc), Bu[l]->fill_bf16(-sc * 64, 0);
  }
  struct V { const char *down, *up; uint32_t nr, cpt; bool one_row; };
  const V vs[] = {{"hc_down_sk2_b6", "hc_up_mix2_b6", 1, 1, true},
                  {"hc_down_sk3_n2_b6", "hc_up_mix3_c2_b6", 2, 2, false},
                  {"hc_down_sk3_n2_b6", "hc_up_mix4_c2_b6", 2, 2, false},
                  {"hc_down_sk3_n2_b6", "hc_up_mix4_c4_b6", 2, 4, false}};
  constexpr int NV = sizeof vs / sizeof vs[0];
  for (uint32_t R = 1; R <= 4; ++R) {
    Buf X(R * SD), y(R * D), injp(R * S), w(SD), winj(S * SD, 2);
    X.fill(-2, 2), y.fill(-1, 1), injp.fill(0, 2), w.fill(0.5f, 1.5f), winj.fill_bf16(-0.05f, 0.05f);
    struct { uint32_t R, D, S, rank, kc, u_ld, has_y, has_inj; float eps; } p{R, D, S, RK, KC, D, 1, 1, 1e-6f};
    std::vector<Buf*> out;  // per variant: X, y, u, inj, part, X2, stat
    printf("R=%u:", R);
    for (int v = 0; v < NV; ++v) {
      Buf *oX = new Buf(R * SD), *oy = new Buf(R * D), *ou = new Buf(R * D), *oi = new Buf(R * S);
      Buf *op = new Buf(KC * R * RK * 2), *o2 = new Buf(R * SD), *os = new Buf(R * KC * 10);
      const V& c = vs[v];
      auto& gpu = Gpu::get();
      const Pipeline& pd = gpu.pipeline(c.down);
      const Pipeline& pu = gpu.pipeline(c.up);
      const char* only = getenv("FN_HC_ONLY");  // "down" or "up": time one kernel of the pair
      auto pass = [&](Encoder& e, uint32_t l) {
        // the residual stream is reset to X each pair (the up kernel writes X = X2 and y = 0)
        if (only && !strcmp(only, "up")) goto up;
        e.use(pd);
        e.buf(0, X.b); e.buf(1, y.b); e.buf(2, injp.b); e.buf(3, w.b); e.buf(4, winj.b); e.buf(5, Wd[l]->b);
        e.buf(6, Sd[l]->b); e.buf(7, Bd[l]->b); e.buf(8, op->b); e.buf(9, o2->b); e.buf(10, os->b); e.val(11, p);
        if (c.one_row) e.dispatch(RK / 8, KC, R, 256); else e.dispatch(RK / (8 * c.nr), KC, 1, 256);
        if (only && !strcmp(only, "down")) return;
      up:
        e.use(pu);
        e.buf(0, op->b); e.buf(1, o2->b); e.buf(2, os->b); e.buf(3, w.b); e.buf(4, Wu[l]->b); e.buf(5, Su[l]->b);
        e.buf(6, Bu[l]->b); e.buf(7, ou->b); e.buf(8, oX->b); e.buf(9, oy->b); e.buf(10, oi->b); e.val(11, p);
        if (c.one_row) e.dispatch(D / 16, R, 1, 256); else e.dispatch(D / (16 * c.cpt), 1, 1, 256);
      };
      double best = 1e30;
      for (int t = 0; t < 4; ++t) {
        auto e = gpu.begin();
        const int n = t == 0 ? 1 : reps;
        for (int k = 0; k < n; ++k)
          for (uint32_t l = 0; l < NL; ++l) pass(*e, l);
        const auto t0 = std::chrono::steady_clock::now();
        e->commit_and_wait();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (t) best = std::min(best, ms * 1000.0 / (n * NL));
      }
      printf("  %s+%s %.1f us", c.down + 8, c.up + 6, best);
      for (Buf* b : {oX, oy, ou, oi, op, o2, os}) out.push_back(b);
    }
    printf("\n");
    for (int v = 1; v < NV; ++v) {
      double bad = 0;
      for (int k = 0; k < 7; ++k) {
        size_t nd = 0, first = 0;
        for (size_t i = 0; i < out[k]->n; ++i)
          if (memcmp(out[k]->f() + i, out[v * 7 + k]->f() + i, 4)) { if (!nd) first = i; ++nd; }
        if (nd) {
          bad += 1;
          if (getenv("FN_VERBOSE"))
            printf("   buf %d: %zu of %zu differ, first %zu: %g vs %g\n", k, nd, out[k]->n, first, out[k]->f()[first],
                   out[v * 7 + k]->f()[first]);
        }
      }
      char nm[96];
      snprintf(nm, sizeof nm, "R=%u %s+%s == one-row", R, vs[v].down + 8, vs[v].up + 6);
      report(nm, bad, 0);
    }
  }
}

int kernels() {
  if (getenv("FN_WFMT")) {
    test_wfmt6();
    return g_fail ? 1 : 0;
  }
  if (getenv("FN_ROWS")) {  // only the verify kernels (zqwen4v.metal) against the decode kernels
    test_hc_rows(4);
    test_hc_rows(6);
    test_hc_rows(8);
    // (qmv2_multi_r2: 16 KB of threadgroup memory, past what shader validation accepts)
    if (!getenv("MTL_SHADER_VALIDATION")) test_qmvs_rows(), test_qmvs_single();
    test_moe4_rows(getenv("MTL_SHADER_VALIDATION") != nullptr);
    printf("%s\n", g_fail ? "FAILED" : "all kernels match");
    return g_fail ? 1 : 0;
  }
  test_topk512();
  // (24 KB of threadgroup memory: past what shader validation, which doubles it, accepts)
  if (!getenv("MTL_SHADER_VALIDATION")) test_qmvht();
  test_hc();
  test_hc_row(4);
  test_hc_row(8);
  test_hc_mix_sk(4);
  test_hc_mix_sk(6);
  test_hc_mix_sk(8);
  test_hc_fold(4);
  test_hc_fold(6);
  test_hc_fold(8);
  test_hc_rows(4);
  test_hc_rows(6);
  test_hc_rows(8);
  // (qmv2_multi_r2: 16 KB of threadgroup memory, past what shader validation accepts)
  if (!getenv("MTL_SHADER_VALIDATION")) test_qmvs_rows(), test_qmvs_single();
  test_moe4_rows(getenv("MTL_SHADER_VALIDATION") != nullptr);
  test_ple();
  test_qsa();
  test_qsa_rows();
  test_qsa_sparse();
  test_qsa_all();
  printf("%s\n", g_fail ? "FAILED" : "all kernels match");
  return g_fail ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: fn_check ngram @ids | selftest | kernels | compile-only | qsa-time\n");
    return 2;
  }
  const std::string cmd = argv[1];
  // The zero-dispatch compile gate (qwen4.metal:345-348's four QSA constants aborted the A9 rider's first pass on a
  // source fault after 1 s and zero dispatches): compile the library a model and the kernels would use, then stop.
  if (cmd == "--compile-only" || cmd == "compile-only") {
    const auto t0 = std::chrono::steady_clock::now();
    Gpu::get().compile();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("compile-only: the kernel library compiled in %.0f ms, no dispatch\n", ms);
    return 0;
  }
  if (cmd == "selftest") return selftest();
  if (cmd == "kernels") return kernels();
  if (cmd == "--qsa-time" || cmd == "qsa-time") { test_qsa_timing(); return g_fail ? 1 : 0; }
  if (cmd == "hc-time") { test_hc_timing(); return g_fail ? 1 : 0; }
  if (cmd == "ngram" && argc >= 3) {
    const std::vector<int> tok = read_ids(argv[2]);
    const NgramHash h = flash_hash();
    std::vector<int64_t> ids(size_t(h.heads));
    for (size_t t = 0; t < tok.size(); ++t) {
      h.ids(tok.data(), int(t), ids.data());
      printf("%zu:", t);
      for (int64_t v : ids) printf(" %lld", (long long)v);
      printf("\n");
    }
    return 0;
  }
  fprintf(stderr, "fn_check: unknown command %s\n", cmd.c_str());
  return 2;
}
