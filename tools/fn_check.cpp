// Qwen3.8-Flash-Next checks.
//   fn_check ngram @ids             the PLE n-gram ids of every position ("t: id0 .. id15"), CPU only: compare with a
//                                   port of the reference's hash (transformers' Qwen4ExpTextNGramEmbedding)
//   fn_check selftest               the hash on a built-in sequence against ids computed by that port (CPU only)
//   fn_check kernels                the Flash-Next kernels (src/kernels/qwen4.metal) on small random inputs against a CPU
//                                   reference in double; one dispatch per command buffer, so it suits
//                                   MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1
#include "metal/gpu.h"
#include "model/ngram.h"
#include "util/bf16.h"
#include "util/log.h"

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
    }), 1e-5);
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
  run("qsa_prep", [&](Encoder& e) { e.buf(0, iqk.b); e.buf(1, wq.b); e.buf(2, iq.b); e.buf(3, ik.b); e.val(4, p); }, NH + 1, R, 1, HD);
  run("qsa_block", [&](Encoder& e) { e.buf(0, ik.b); e.buf(1, wk.b); e.buf(2, bk.b); e.val(3, p); }, R, 1, 1, HD);
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

// the 3-row half-precision verify GEMV (qmvht_t3_r4) on the Flash-Next shapes against fp64
void test_qmvht() {
  struct Shape { uint32_t N, K; };
  const char* kn = getenv("FN_QMVHT") ? getenv("FN_QMVHT") : "qmvht_t3_r4";
  const uint32_t R = uint32_t(kn[7] - '0');
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
    char name[64];
    snprintf(name, sizeof name, "%s %ux%u", kn, N, K);
    report(name, err, 5e-3);
  }
}

int kernels() {
  // (24 KB of threadgroup memory: past what shader validation, which doubles it, accepts)
  if (!getenv("MTL_SHADER_VALIDATION")) test_qmvht();
  test_hc();
  test_ple();
  test_qsa();
  test_qsa_sparse();
  printf("%s\n", g_fail ? "FAILED" : "all kernels match");
  return g_fail ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: fn_check ngram @ids | selftest | kernels\n");
    return 2;
  }
  const std::string cmd = argv[1];
  if (cmd == "selftest") return selftest();
  if (cmd == "kernels") return kernels();
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
