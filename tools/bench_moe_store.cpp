// Correctness and per-expert timing of the routed experts' 8-row verify kernels for MLX-affine stores of any bit width and
// scale group (moe8g_gu_f<B>g<G>_f8g64, moe8g_down_f<B>g<G>_f8g64 in wfmt.metal), on random weights, no model files:
//   bench_moe_store --synth BITS:GS[,BITS:GS...] [--check] [--trials N]
// One layer (E=256, D=2048, I=512) of random routed experts at the given width and group (64 or 128) with an 8-bit group-64
// shared expert. A 3:128 spec also builds its group-64 control: the same codes with every scale and bias written twice,
// run through the f3g64 kernels, and compares the two bitwise. Parts:
//   check   three random 8-row routings through gate/up, down and combine against a double-precision CPU reference
//   sweep   (skipped with --check) N distinct experts per call (8 rows, 8 of N each), disjoint chunks of the layer's
//           experts rotating call to call: gate/up, down and their pair per format, median and range over the trials,
//           plus a linear fit in microseconds per distinct expert
// Output: human lines plus "RESULT ..." lines.
#include "gguf/gguf.h"
#include "gguf/quant.h"
#include "loader/safetensors.h"
#include "metal/gpu.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;
namespace gg = ni::gguf;

namespace {

struct Qmm8GuParams { uint32_t K, N, w_stride, s_stride, moe, h_stride, hs_stride; };
struct Moe8DownParams { uint32_t K, N, w_stride, s_stride, h_stride, hs_stride, k; };
struct X8PrepParams { uint32_t K, R, x_ld, norm, write_xn; float eps; uint32_t xn_ld, write_lo, unroll, write_t; };
struct Moe8CombineParams { uint32_t N, k; };

constexpr uint32_t D = 2048, I = 512, E = 256, K8 = 8;
constexpr uint32_t NONE = 0xffffffffu;

std::mt19937 rng(7);
float urand(float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); }
double hround(double v) { return double(_Float16(float(v))); }
double silu(double v) { return v / (1.0 + std::exp(-v)); }
Pipeline pipe(const std::string& n) { return Gpu::get().pipeline(n); }

// ---------------------------------------------------------------------------------------------------- weights
struct Q8 {  // MLX 8-bit group-64 matrix (the shared expert)
  Tensor w, s, b;
  uint32_t N = 0, K = 0;
  double at(uint32_t n, uint32_t k) const {
    const size_t g = size_t(n) * (K / 64) + k / 64;
    return double(gg::from_bf16(s.buf.as<uint16_t>()[g])) * w.buf.as<uint8_t>()[size_t(n) * K + k] +
           double(gg::from_bf16(b.buf.as<uint16_t>()[g]));
  }
};

struct Store {  // one layer's routed experts (codes, scales, biases) and its shared expert
  Tensor gw, uw, dw;  // packed codes
  Tensor gs, gb, us, ub, ds, db;
  uint32_t wstride = 0, sstride = 0, dwstride = 0, dsstride = 0;
  Q8 sg, su, sd;
  int bits = 3, gsz = 64;  // code width and scale group
  std::string label;
};

// ---------------------------------------------------------------------------------------------------- plans
struct Route {
  uint32_t R = 0;
  std::vector<uint32_t> idx;  // [R][8]
};

struct Plan {  // one tile list, as moe8_plan writes it (ascending expert ids, then the shared expert's tile) + its h tables
  Buffer tiles, nt, slot, args, hT, hS;
  uint32_t n = 0;  // tiles
};

Plan make_plan(const Route& r) {
  Gpu& g = Gpu::get();
  std::vector<uint32_t> tiles, slot;
  for (uint32_t e = 0; e < E; ++e) {
    uint32_t s[8];
    bool used = false;
    for (uint32_t t = 0; t < 8; ++t) {
      s[t] = NONE;
      if (t < r.R)
        for (uint32_t j = 0; j < K8; ++j)
          if (r.idx[t * K8 + j] == e) s[t] = j, used = true;
    }
    if (!used) continue;
    tiles.push_back(e);
    slot.insert(slot.end(), s, s + 8);
  }
  tiles.push_back(NONE);
  for (uint32_t t = 0; t < 8; ++t) slot.push_back(t < r.R ? K8 : NONE);
  Plan p;
  p.n = uint32_t(tiles.size());
  const size_t cap = 66;
  p.tiles = g.alloc(cap * 4), p.slot = g.alloc(cap * 8 * 4), p.nt = g.alloc(16), p.args = g.alloc(64);
  memcpy(p.tiles.ptr(), tiles.data(), tiles.size() * 4);
  memcpy(p.slot.ptr(), slot.data(), slot.size() * 4);
  uint32_t* nt = p.nt.as<uint32_t>();
  nt[0] = p.n, nt[1] = 0, nt[2] = 0;
  uint32_t* a = p.args.as<uint32_t>();
  const uint32_t y = std::max(p.n, 1u);
  a[0] = I / 64, a[1] = y, a[2] = 1, a[3] = D / 64, a[4] = y, a[5] = 1;
  p.hT = g.alloc(size_t(y) * I * 8 * 2), p.hS = g.alloc(size_t(y) * I / 64 * 8 * 4);
  return p;
}

// ---------------------------------------------------------------------------------------------------- shared inputs
struct Shared {
  Buffer x, table, sums, tlo, slo, out, sh, wts;
  std::vector<float> x0;
};

std::string kernel(const char* stage, const Store& s) {
  return std::string("moe8g_") + stage + "_f" + std::to_string(s.bits) + "g" + std::to_string(s.gsz) + "_f8g64";
}

void enc_gu(Encoder& e, const Store& s, const Shared& c, const Plan& p) {
  const Qmm8GuParams gp{D, I, s.wstride, s.sstride, 1u, I * 8, I / 64 * 8};
  e.use(pipe(kernel("gu", s)));
  e.buf(0, s.gw.buf); e.buf(1, s.gs.buf); e.buf(2, s.gb.buf);
  e.buf(3, s.uw.buf); e.buf(4, s.us.buf); e.buf(5, s.ub.buf);
  e.buf(6, s.sg.w.buf); e.buf(7, s.sg.s.buf); e.buf(8, s.sg.b.buf);
  e.buf(9, s.su.w.buf); e.buf(10, s.su.s.buf); e.buf(11, s.su.b.buf);
  e.buf(12, c.table); e.buf(13, c.sums); e.buf(14, p.hT); e.buf(15, p.hS);
  e.val(16, gp);
  e.buf(17, p.tiles); e.buf(18, p.nt);
  e.dispatch_indirect(p.args, 0, 512);
}

void enc_down(Encoder& e, const Store& s, const Shared& c, const Plan& p) {
  const Moe8DownParams dp{I, D, s.dwstride, s.dsstride, I * 8, I / 64 * 8, K8};
  e.use(pipe(kernel("down", s)));
  e.buf(0, s.dw.buf); e.buf(1, s.ds.buf); e.buf(2, s.db.buf);
  e.buf(3, s.sd.w.buf); e.buf(4, s.sd.s.buf); e.buf(5, s.sd.b.buf);
  e.buf(6, p.hT); e.buf(7, p.hS); e.buf(8, c.out); e.buf(9, c.sh);
  e.val(10, dp);
  e.buf(11, p.tiles); e.buf(12, p.nt); e.buf(13, p.slot);
  e.dispatch_indirect(p.args, 12, 128);
}

Shared make_shared() {
  Gpu& g = Gpu::get();
  Shared c;
  c.x = g.alloc(size_t(8) * D * 4);
  c.table = g.alloc(size_t(D) * 8 * 2);
  c.sums = g.alloc(size_t(D) / 64 * 8 * 4);
  c.tlo = g.alloc(size_t(D) * 8 * 2);
  c.slo = g.alloc(size_t(D) / 64 * 8 * 4);
  c.out = g.alloc(size_t(8) * K8 * D * 4);
  c.sh = g.alloc(size_t(8) * D * 4);
  c.wts = g.alloc(size_t(8) * (K8 + 1) * 4);
  c.x0.assign(size_t(8) * D, 0.0f);
  for (auto& v : c.x0) v = urand(-2.0f, 2.0f);
  memcpy(c.x.ptr(), c.x0.data(), c.x0.size() * 4);
  const X8PrepParams xp{D, 8, D, 0, 0, 1e-6f, D, 0, 1, 0};
  auto e = g.begin();
  e->use(pipe("x8_prep2"));
  e->buf(0, c.x); e->buf(1, c.x); e->buf(2, c.table); e->buf(3, c.sums); e->buf(4, c.x); e->val(5, xp);
  e->buf(6, c.tlo); e->buf(7, c.slo); e->buf(8, c.tlo);
  e->dispatch(D / 256, 1, 1, 256);
  e->commit_and_wait();
  return c;
}

// ---------------------------------------------------------------------------------------------------- statistics / timing
struct Stat { double med, mn, mx, p10, p90; };
Stat stat(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  auto at = [&](double q) { return v[size_t(q * (v.size() - 1) + 0.5)]; };
  return Stat{at(0.5), v.front(), v.back(), at(0.1), at(0.9)};
}

int g_trials = 40;
double load1() {
  double l[3] = {0, 0, 0};
  getloadavg(l, 3);
  return l[0];
}

struct Var {
  std::string name;
  std::function<void(Encoder&, int)> enc;  // enc(e, call index)
  int calls;
  std::vector<double> ms;  // per call, per trial
};

void time_vars(std::vector<Var>& vs) {
  Gpu& g = Gpu::get();
  for (auto& v : vs) {  // warm-up
    auto e = g.begin();
    for (int i = 0; i < v.calls; ++i) v.enc(*e, i);
    e->commit_and_wait();
  }
  for (int t = 0; t < g_trials; ++t)  // variants interleaved per trial
    for (auto& v : vs) {
      auto e = g.begin();
      for (int i = 0; i < v.calls; ++i) v.enc(*e, t * 7 + i);
      e->commit_and_wait();
      v.ms.push_back(e->gpu_ms() / v.calls);
    }
}

// ---------------------------------------------------------------------------------------------------- reference / check
uint32_t unpack(const uint32_t* row, uint32_t col, int bits) {
  const uint32_t p = col * uint32_t(bits), wi = p / 32, sh = p % 32;
  uint64_t v = row[wi] >> sh;
  if (sh + uint32_t(bits) > 32) v |= uint64_t(row[wi + 1]) << (32 - sh);
  return uint32_t(v & ((1u << bits) - 1));
}

// dequantized row grow of an affine matrix of Kv columns
std::vector<double> dequant_row(const Store& s, const Tensor& w, const Tensor& sc, const Tensor& bi, size_t grow, uint32_t Kv) {
  std::vector<double> out(Kv);
  const uint32_t* wr = w.buf.as<uint32_t>() + grow * (Kv * uint32_t(s.bits) / 32);
  for (uint32_t c = 0; c < Kv; ++c) {
    const size_t g = grow * (Kv / uint32_t(s.gsz)) + c / uint32_t(s.gsz);
    out[c] = double(gg::from_bf16(sc.buf.as<uint16_t>()[g])) * unpack(wr, c, s.bits) + double(gg::from_bf16(bi.buf.as<uint16_t>()[g]));
  }
  return out;
}

// full chain (gate/up, down, combine) on a routing, compared with fp64 on the dequantized weights; returns the relative error
double check_chain(const Store& s, Shared& c, const Route& r, const std::vector<float>& wts) {
  Gpu& g = Gpu::get();
  Plan p = make_plan(r);
  memcpy(c.x.ptr(), c.x0.data(), c.x0.size() * 4);
  memcpy(c.wts.ptr(), wts.data(), wts.size() * 4);
  auto e = g.begin();
  enc_gu(*e, s, c, p);
  enc_down(*e, s, c, p);
  const Moe8CombineParams cp{D, K8};
  e->use(pipe("moe8_combine"));
  e->buf(0, c.out); e->buf(1, c.sh); e->buf(2, c.wts); e->buf(3, c.x);
  e->val(4, cp);
  e->dispatch((D / 4 + 63) / 64, r.R, 1, 64);
  e->commit_and_wait();
  double max_abs = 0, max_ref = 0;
  for (uint32_t t = 0; t < 2; ++t) {  // rows 0 and 1
    std::vector<double> acc(D, 0.0);
    for (uint32_t sl = 0; sl <= K8; ++sl) {  // the 8 routed experts, then the shared one
      const double w = wts[t * (K8 + 1) + sl];
      const uint32_t ex = sl == K8 ? 0 : r.idx[t * K8 + sl];
      std::vector<double> hh(I);
      for (uint32_t n = 0; n < I; ++n) {
        std::vector<double> wg, wu;
        if (sl < K8) {
          wg = dequant_row(s, s.gw, s.gs, s.gb, size_t(ex) * I + n, D);
          wu = dequant_row(s, s.uw, s.us, s.ub, size_t(ex) * I + n, D);
        }
        double a = 0, b = 0;
        for (uint32_t cc = 0; cc < D; ++cc) {
          const double xh = hround(c.x0[size_t(t) * D + cc]);
          a += (sl == K8 ? s.sg.at(n, cc) : wg[cc]) * xh, b += (sl == K8 ? s.su.at(n, cc) : wu[cc]) * xh;
        }
        hh[n] = hround(silu(a) * b);
      }
      for (uint32_t n = 0; n < D; ++n) {
        std::vector<double> wd;
        if (sl < K8) wd = dequant_row(s, s.dw, s.ds, s.db, size_t(ex) * D + n, I);
        double a = 0;
        for (uint32_t cc = 0; cc < I; ++cc) a += (sl == K8 ? s.sd.at(n, cc) : wd[cc]) * hh[cc];
        acc[n] += w * a;
      }
    }
    for (uint32_t n = 0; n < D; ++n) {
      const double got = double(c.x.as<float>()[size_t(t) * D + n]) - c.x0[size_t(t) * D + n];
      max_abs = std::max(max_abs, std::fabs(got - acc[n]));
      max_ref = std::max(max_ref, std::fabs(acc[n]));
    }
  }
  return max_abs / max_ref;
}

// ---------------------------------------------------------------------------------------------------- synthetic stores
Tensor synth_tensor(DType dt, std::vector<int64_t> shape, size_t bytes) {
  Tensor t;
  t.buf = Gpu::get().alloc(bytes + 64);
  memset(t.buf.ptr(), 0, bytes + 64);
  t.dtype = dt;
  t.shape = std::move(shape);
  return t;
}
uint16_t bf16_of(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return uint16_t(u >> 16);
}
Tensor synth_codes(size_t rows, uint32_t Kv, int bits) {
  const size_t words = size_t(Kv) * size_t(bits) / 32;
  Tensor t = synth_tensor(DType::U32, {int64_t(rows), int64_t(words)}, rows * words * 4);
  uint32_t* p = t.buf.as<uint32_t>();
  for (size_t i = 0; i < rows * words; ++i) p[i] = uint32_t(rng());
  return t;
}
// scales / biases [rows, groups]: w = s q + b centred near zero, so no group dominates the dot products
void synth_sb(size_t rows, size_t groups, int bits, Tensor& s, Tensor& b) {
  s = synth_tensor(DType::BF16, {int64_t(rows), int64_t(groups)}, rows * groups * 2);
  b = synth_tensor(DType::BF16, {int64_t(rows), int64_t(groups)}, rows * groups * 2);
  const float qmax = float((1u << bits) - 1);
  for (size_t i = 0; i < rows * groups; ++i) {
    const float sc = urand(0.05f, 0.15f) * 7.0f / qmax;
    s.buf.as<uint16_t>()[i] = bf16_of(sc);
    b.buf.as<uint16_t>()[i] = bf16_of(-sc * qmax * 0.5f + urand(-0.01f, 0.01f));
  }
}
Tensor dup_scales(const Tensor& t) {
  const size_t n = size_t(t.numel());
  Tensor o = synth_tensor(DType::BF16, {t.shape[0], t.shape[1] * 2}, n * 4);
  for (size_t i = 0; i < n; ++i) o.buf.as<uint16_t>()[2 * i] = o.buf.as<uint16_t>()[2 * i + 1] = t.buf.as<uint16_t>()[i];
  return o;
}
Q8 synth_q8(uint32_t Nn, uint32_t Kv) {
  Q8 q;
  q.N = Nn, q.K = Kv;
  q.w = synth_tensor(DType::U8, {int64_t(Nn), int64_t(Kv)}, size_t(Nn) * Kv);
  for (size_t i = 0; i < size_t(Nn) * Kv; ++i) q.w.buf.as<uint8_t>()[i] = uint8_t(rng());
  q.s = synth_tensor(DType::BF16, {int64_t(Nn), int64_t(Kv / 64)}, size_t(Nn) * (Kv / 64) * 2);
  q.b = synth_tensor(DType::BF16, {int64_t(Nn), int64_t(Kv / 64)}, size_t(Nn) * (Kv / 64) * 2);
  for (size_t i = 0; i < size_t(Nn) * (Kv / 64); ++i) {
    const float sc = urand(0.0005f, 0.0015f);
    q.s.buf.as<uint16_t>()[i] = bf16_of(sc);
    q.b.buf.as<uint16_t>()[i] = bf16_of(-sc * 127.5f);
  }
  return q;
}
// native: random codes + scales at group gsz; control (share != null): its codes, scales written twice (group 64)
std::unique_ptr<Store> synth_store(int bits, int gsz, const Store* share, const Q8& g, const Q8& u, const Q8& d) {
  auto s = std::make_unique<Store>();
  s->bits = bits, s->gsz = share ? 64 : gsz;
  s->label = "f" + std::to_string(bits) + "g" + std::to_string(s->gsz) + (share ? " (dup of g128)" : "");
  if (share) {
    s->gw = share->gw, s->uw = share->uw, s->dw = share->dw;
    s->gs = dup_scales(share->gs), s->gb = dup_scales(share->gb), s->us = dup_scales(share->us);
    s->ub = dup_scales(share->ub), s->ds = dup_scales(share->ds), s->db = dup_scales(share->db);
  } else {
    s->gw = synth_codes(size_t(E) * I, D, bits), s->uw = synth_codes(size_t(E) * I, D, bits), s->dw = synth_codes(size_t(E) * D, I, bits);
    synth_sb(size_t(E) * I, D / size_t(gsz), bits, s->gs, s->gb);
    synth_sb(size_t(E) * I, D / size_t(gsz), bits, s->us, s->ub);
    synth_sb(size_t(E) * D, I / size_t(gsz), bits, s->ds, s->db);
  }
  s->wstride = uint32_t(size_t(I) * D * bits / 32);
  s->sstride = I * (D / uint32_t(s->gsz));
  s->dwstride = uint32_t(size_t(D) * I * bits / 32);
  s->dsstride = D * (I / uint32_t(s->gsz));
  s->sg = g, s->su = u, s->sd = d;
  return s;
}

Route synth_route() {
  Route r;
  r.R = 8;
  for (uint32_t t = 0; t < 8; ++t) {
    std::set<uint32_t> used;
    while (used.size() < K8) used.insert(rng() % E);
    for (uint32_t e : used) r.idx.push_back(e);
  }
  return r;
}

struct Snap { std::vector<uint8_t> hT, hS, out, sh; };
Snap run_pass(const Store& s, Shared& c, const Plan& p) {
  memset(c.out.ptr(), 0, c.out.size);
  memset(c.sh.ptr(), 0, c.sh.size);
  memset(p.hT.ptr(), 0, p.hT.size);
  memset(p.hS.ptr(), 0, p.hS.size);
  auto e = Gpu::get().begin();
  enc_gu(*e, s, c, p);
  enc_down(*e, s, c, p);
  e->commit_and_wait();
  auto cp = [](const Buffer& b) { return std::vector<uint8_t>((uint8_t*)b.ptr(), (uint8_t*)b.ptr() + b.size); };
  return Snap{cp(p.hT), cp(p.hS), cp(c.out), cp(c.sh)};
}

int run_synth(const std::string& spec, bool check_only) {
  std::vector<std::pair<int, int>> fm;
  {
    std::istringstream is(spec);
    std::string tok;
    while (std::getline(is, tok, ',')) {
      int b = 0, gz = 0;
      if (sscanf(tok.c_str(), "%d:%d", &b, &gz) != 2 || b < 2 || b > 8 || (gz != 64 && gz != 128)) fatal("bad --synth item %s (BITS:GS, GS 64 or 128)", tok.c_str());
      fm.push_back({b, gz});
    }
  }
  Shared c = make_shared();
  const Q8 qg = synth_q8(I, D), qu = synth_q8(I, D), qd = synth_q8(D, I);
  std::vector<std::unique_ptr<Store>> stores;
  for (auto [b, gz] : fm) {
    stores.push_back(synth_store(b, gz, nullptr, qg, qu, qd));
    if (gz == 128 && b != 2) stores.push_back(synth_store(b, gz, stores.back().get(), qg, qu, qd));  // group-64 control (3-bit)
  }
  std::vector<float> wts(8 * (K8 + 1));
  for (uint32_t t = 0; t < 8; ++t) {
    float sum = 0;
    for (uint32_t j = 0; j < K8; ++j) sum += (wts[t * (K8 + 1) + j] = urand(0.05f, 1.0f));
    for (uint32_t j = 0; j < K8; ++j) wts[t * (K8 + 1) + j] /= sum;
    wts[t * (K8 + 1) + K8] = urand(0.1f, 0.9f);
  }
  bool all_ok = true;
  printf("synthetic correctness (one layer, E=%u D=%u I=%u, 8 rows x top-8 + shared, 3 random routes per format; fp64 CPU reference, pass < 2e-3)\n", E, D, I);
  for (auto& s : stores) {
    for (int rep = 0; rep < 3; ++rep) {
      const Route r = synth_route();
      const double err = check_chain(*s, c, r, wts);
      printf("  check %-22s route %d: max |got - ref| / max |ref| = %.2e%s\n", s->label.c_str(), rep, err, err > 2e-3 ? "  FAILED" : "");
      printf("RESULT synthcheck fmt=\"%s\" route=%d relerr=%.3e\n", s->label.c_str(), rep, err);
      if (err > 2e-3 || !(err == err)) all_ok = false;
    }
  }
  // bitwise: native g128 vs the duplicated-scale g64 path on identical data (h tables, h sums, down outputs, shared output)
  for (size_t i = 0; i + 1 < stores.size(); ++i) {
    if (stores[i + 1]->label.find("dup") == std::string::npos) continue;
    int bad = 0;
    for (int rep = 0; rep < 4; ++rep) {
      const Route r = synth_route();
      Plan p = make_plan(r);
      const Snap a = run_pass(*stores[i], c, p), b = run_pass(*stores[i + 1], c, p);
      const bool same = a.hT == b.hT && a.hS == b.hS && a.out == b.out && a.sh == b.sh;
      if (!same) ++bad;
    }
    printf("  bitwise %s vs %s: %d of 4 routes differ (h tables, h sums, down outputs, shared output)\n", stores[i]->label.c_str(), stores[i + 1]->label.c_str(), bad);
    printf("RESULT synthbitwise native=\"%s\" control=\"%s\" routes_differ=%d of=4\n", stores[i]->label.c_str(), stores[i + 1]->label.c_str(), bad);
    if (bad) all_ok = false;
  }
  if (check_only) return all_ok ? 0 : 1;

  std::vector<int> ns = {8, 12, 16, 24, 32, 40, 48, 56, 64};
  const int nf = int(stores.size());
  struct Pt { int n; std::vector<Stat> gu, dn, pair; };
  std::vector<Pt> pts;
  printf("\nsynthetic sweep: 8 rows, N distinct experts, disjoint chunks rotating, %d trials, formats interleaved per trial\n", g_trials);
  const double load_a = load1();
  printf("load before: %.2f\n", load_a);
  for (int n : ns) {
    std::vector<uint32_t> perm(E);
    std::iota(perm.begin(), perm.end(), 0u);
    std::shuffle(perm.begin(), perm.end(), rng);
    const int nchunks = std::min<int>(E / n, 32);
    std::vector<Plan> plans;
    for (int ch = 0; ch < nchunks; ++ch) {
      Route r;
      r.R = 8;
      for (uint32_t t = 0; t < 8; ++t)
        for (uint32_t j = 0; j < K8; ++j) r.idx.push_back(perm[size_t(ch) * n + (t * 8 + j) % n]);
      plans.push_back(make_plan(r));
    }
    const int calls = std::max(64, nchunks * 2);
    std::vector<Var> vs;
    for (int f = 0; f < nf; ++f) {
      const Store* sp = stores[size_t(f)].get();
      vs.push_back({sp->label + " gu", [&, sp](Encoder& e, int i) { enc_gu(e, *sp, c, plans[size_t(i) % plans.size()]); }, calls, {}});
      vs.push_back({sp->label + " down", [&, sp](Encoder& e, int i) { enc_down(e, *sp, c, plans[size_t(i) % plans.size()]); }, calls, {}});
      vs.push_back({sp->label + " pair", [&, sp](Encoder& e, int i) {
                      enc_gu(e, *sp, c, plans[size_t(i) % plans.size()]);
                      enc_down(e, *sp, c, plans[size_t(i) % plans.size()]);
                    }, calls, {}});
    }
    time_vars(vs);
    Pt pt;
    pt.n = n;
    printf("  n=%2d (%d chunks, %d calls x %d trials)  ms/call median [min-max]:\n", n, nchunks, calls, g_trials);
    for (int f = 0; f < nf; ++f) {
      pt.gu.push_back(stat(vs[size_t(f) * 3 + 0].ms)), pt.dn.push_back(stat(vs[size_t(f) * 3 + 1].ms)), pt.pair.push_back(stat(vs[size_t(f) * 3 + 2].ms));
      const Stat &g1 = pt.gu.back(), &d1 = pt.dn.back(), &p1 = pt.pair.back();
      printf("     %-22s gu %.4f [%.4f-%.4f]  down %.4f [%.4f-%.4f]  pair %.4f [%.4f-%.4f] range %.1f%%\n", stores[size_t(f)]->label.c_str(), g1.med, g1.mn,
             g1.mx, d1.med, d1.mn, d1.mx, p1.med, p1.mn, p1.mx, 100.0 * (p1.mx - p1.mn) / p1.med);
      printf("RESULT synth n=%d fmt=\"%s\" gu=%.5f gu_min=%.5f gu_max=%.5f dn=%.5f dn_min=%.5f dn_max=%.5f pair=%.5f pair_p10=%.5f pair_p90=%.5f pair_min=%.5f pair_max=%.5f\n", n,
             stores[size_t(f)]->label.c_str(), g1.med, g1.mn, g1.mx, d1.med, d1.mn, d1.mx, p1.med, p1.p10, p1.p90, p1.mn, p1.mx);
    }
    pts.push_back(std::move(pt));
  }
  const double load_b = load1();
  printf("load after: %.2f\n", load_b);
  printf("RESULT synthload before=%.2f after=%.2f\n", load_a, load_b);
  // linear fit ms = a + us_per_expert * n over the sweep points (the shared expert and launch cost are the intercept)
  auto fit = [&](int f, int which) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    const double m = double(pts.size());
    for (auto& p : pts) {
      const double x = p.n, y = (which == 0 ? p.gu[size_t(f)] : which == 1 ? p.dn[size_t(f)] : p.pair[size_t(f)]).med * 1000.0;
      sx += x, sy += y, sxx += x * x, sxy += x * y;
    }
    const double slope = (m * sxy - sx * sy) / (m * sxx - sx * sx);
    return std::pair<double, double>(slope, (sy - slope * sx) / m);
  };
  for (int f = 0; f < nf; ++f) {
    const auto g1 = fit(f, 0), d1 = fit(f, 1), p1 = fit(f, 2);
    printf("  fit %-22s us per distinct expert: gu %.2f  down %.2f  pair %.2f   (intercepts us: %.1f %.1f %.1f)\n", stores[size_t(f)]->label.c_str(), g1.first,
           d1.first, p1.first, g1.second, d1.second, p1.second);
    printf("RESULT synthfit fmt=\"%s\" us_gu=%.3f us_dn=%.3f us_pair=%.3f int_gu=%.2f int_dn=%.2f int_pair=%.2f\n", stores[size_t(f)]->label.c_str(), g1.first,
           d1.first, p1.first, g1.second, d1.second, p1.second);
  }
  return all_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  bool check = false;
  std::string synth_spec;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--synth" && i + 1 < argc) synth_spec = argv[++i];
    else if (a == "--trials" && i + 1 < argc) g_trials = atoi(argv[++i]);
    else if (a == "--check") check = true;
    else fatal("unknown argument %s", a.c_str());
  }
  if (synth_spec.empty()) fatal("usage: bench_moe_store --synth BITS:GS[,BITS:GS...] [--check] [--trials N]");
  Gpu& g = Gpu::get();
  printf("%s, %d GPU cores, synthetic %s\n", g.name().c_str(), g.core_count(), synth_spec.c_str());
  return run_synth(synth_spec, check);
}
