// Clover's denoising-pass kernels (src/kernels/clover.metal, attn_prefill11gs_bidi) against CPU fp64 transcriptions of
// the reference implementation (Diffuspace/Clover-35B-A3B-Preview's modeling_clover_moe.py and sampling_clover_moe.py)
// on random data at the model's shapes (35B-A3B: 16 key / 32 value heads of 128, 16 query / 2 KV heads of 256, 256
// experts, the 248320-token vocabulary). No model is read.
//   clover_check [T ...]   (rows: default 32, the block, and 8, a self-speculative cycle)
// The kernels other than the attention hold at most 4224 bytes of threadgroup memory, so they run under
// MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1; attn_prefill11gs_bidi holds 20480 bytes, which the debug layer counts
// twice against the 32768-byte cap (AGENTS.md): CLOVER_NO_ATTN=1 leaves it out for such a run. Returns 1 if any check
// fails. It takes the GPU lock (BENCH_GPU_LOCK, default /tmp/kapitsa-gpu.lock) unless its caller holds it
// (NI_LOCK_HELD=1 / NACE_GPU_LOCK_HELD=1).
#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/kparams.h"
#include "util/bf16.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace ni;

namespace {

struct CloverPrep {
  uint32_t T, Hk, Hv, C;
};
struct CloverMem {
  uint32_t T, Hk, Hv;
  float lr;
};
struct CloverCoreset {
  uint32_t T, E, k;
  float top_p;
};
struct CloverPick {
  uint32_t V, ld;
  uint32_t mask_id;
  uint32_t nbias;
  float frequency, presence, repetition, temperature;
  uint32_t seed, step;
  uint32_t skip, ncand, sample_out;
};
struct Row {
  int tok, smp;
  float conf, conf_smp, pc, ent, pad0, pad1;
};

constexpr int Hk = 16, Hv = 32, Dh = 128, C = 2 * Hk * Dh + Hv * Dh;
int fails = 0;

void report(const char* what, double err, double tol) {
  const bool ok = err <= tol;
  printf("%-44s max error %.3g (tolerance %.3g) %s\n", what, err, tol, ok ? "ok" : "FAIL");
  if (!ok) ++fails;
}

double silu(double x) { return x / (1 + std::exp(-x)); }
double softplus(double x) { return x > 20 ? x : std::log1p(std::exp(x)); }

void check_gdn(Gpu& g, std::mt19937& rng, int T) {
  std::normal_distribution<float> nd(0.0f, 1.0f);
  Buffer qkv = g.alloc(size_t(T) * C * 4), a = g.alloc(size_t(T) * Hv * 4), cs = g.alloc(3 * C * 4);
  Buffer cw = g.alloc(C * 4 * 2), cr = g.alloc(C * 3 * 2), Al = g.alloc(Hv * 2), dt = g.alloc(Hv * 2);
  Buffer q = g.alloc(size_t(T) * Hk * Dh * 4), k = g.alloc(size_t(T) * Hk * Dh * 4), v = g.alloc(size_t(T) * Hv * Dh * 4);
  Buffer gg = g.alloc(size_t(T) * Hv * 4), st = g.alloc(size_t(Hv) * Dh * Dh * 4), sc = g.alloc(3 * Hv * 4);
  Buffer out = g.alloc(size_t(T) * Hv * Dh * 4);
  for (int i = 0; i < T * C; ++i) qkv.as<float>()[i] = nd(rng);
  for (int i = 0; i < T * Hv; ++i) a.as<float>()[i] = nd(rng);
  for (int i = 0; i < 3 * C; ++i) cs.as<float>()[i] = nd(rng);
  for (int i = 0; i < C * 4; ++i) cw.as<uint16_t>()[i] = f32_to_bf16(nd(rng) * 0.4f);
  for (int i = 0; i < C * 3; ++i) cr.as<uint16_t>()[i] = f32_to_bf16(nd(rng) * 0.3f);
  for (int i = 0; i < Hv; ++i) Al.as<uint16_t>()[i] = f32_to_bf16(nd(rng)), dt.as<uint16_t>()[i] = f32_to_bf16(nd(rng));
  for (size_t i = 0; i < size_t(Hv) * Dh * Dh; ++i) st.as<float>()[i] = nd(rng) * 0.1f;
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for (int h = 0; h < Hv; ++h) {
    sc.as<float>()[h] = 5 + 10 * u(rng);           // logit scale (the reference initializes it at 10)
    sc.as<float>()[Hv + h] = 6 * u(rng);           // history bias (5)
    sc.as<float>()[2 * Hv + h] = 2 * u(rng);       // history c (1)
  }
  const float lr = std::log(float(T) / 32.0f);
  auto e = g.begin();
  const CloverPrep pp{uint32_t(T), Hk, Hv, C};
  e->use(g.pipeline("clover_gdn_prep"));
  e->buf(0, qkv); e->buf(1, a); e->buf(2, cs); e->buf(3, cw); e->buf(4, cr); e->buf(5, Al); e->buf(6, dt);
  e->buf(7, q); e->buf(8, k); e->buf(9, v); e->buf(10, gg);
  e->val(11, pp);
  e->dispatch(C / 128 + 1, 1, 1, 128);
  const CloverMem mp{uint32_t(T), Hk, Hv, lr};
  e->use(g.pipeline("clover_gdn_mem"));
  e->buf(0, q); e->buf(1, k); e->buf(2, v); e->buf(3, gg); e->buf(4, st); e->buf(5, sc); e->buf(6, out);
  e->val(7, mp);
  e->dispatch(Hv, 4, 1, 128);
  e->commit_and_wait();

  // bd_gdn_conv1d: the left taps over [3 committed rows | the block], the right taps over the block's next rows; SiLU
  std::vector<double> mixed(size_t(T) * C);
  auto x = [&](int t, int c) { return t < 0 ? double(cs.as<float>()[(t + 3) * C + c]) : double(qkv.as<float>()[t * C + c]); };
  for (int t = 0; t < T; ++t)
    for (int c = 0; c < C; ++c) {
      double acc = 0;
      for (int j = 0; j < 4; ++j) acc += double(bf16_to_f32(cw.as<uint16_t>()[c * 4 + j])) * x(t - 3 + j, c);
      for (int d = 1; d < 4 && t + d < T; ++d) acc += double(bf16_to_f32(cr.as<uint16_t>()[c * 3 + d - 1])) * x(t + d, c);
      mixed[size_t(t) * C + c] = silu(acc);
    }
  // q, k l2-normalized (eps 1e-6 in the sum); q as the kernels keep it, / sqrt(128)
  std::vector<double> rq(size_t(T) * Hk * Dh), rk(rq.size()), rv(size_t(T) * Hv * Dh), rg(size_t(T) * Hv);
  double eq = 0, ev = 0, eg = 0;
  for (int t = 0; t < T; ++t) {
    for (int hd = 0; hd < 2 * Hk; ++hd) {
      double ss = 0;
      for (int i = 0; i < Dh; ++i) ss += mixed[size_t(t) * C + hd * Dh + i] * mixed[size_t(t) * C + hd * Dh + i];
      const double inv = 1 / std::sqrt(ss + 1e-6);
      for (int i = 0; i < Dh; ++i) {
        const double val = mixed[size_t(t) * C + hd * Dh + i] * inv;
        if (hd < Hk) {
          rq[(size_t(t) * Hk + hd) * Dh + i] = val / std::sqrt(double(Dh));
          eq = std::max(eq, std::abs(val / std::sqrt(double(Dh)) - q.as<float>()[(size_t(t) * Hk + hd) * Dh + i]) * std::sqrt(double(Dh)));
        } else {
          rk[(size_t(t) * Hk + hd - Hk) * Dh + i] = val;
          eq = std::max(eq, std::abs(val - k.as<float>()[(size_t(t) * Hk + hd - Hk) * Dh + i]));
        }
      }
    }
    for (int i = 0; i < Hv * Dh; ++i) {
      rv[size_t(t) * Hv * Dh + i] = mixed[size_t(t) * C + 2 * Hk * Dh + i];
      ev = std::max(ev, std::abs(rv[size_t(t) * Hv * Dh + i] - v.as<float>()[size_t(t) * Hv * Dh + i]));
    }
    for (int h = 0; h < Hv; ++h) {
      const double A = std::exp(double(bf16_to_f32(Al.as<uint16_t>()[h])));
      const double gv = -A * softplus(a.as<float>()[t * Hv + h] + double(bf16_to_f32(dt.as<uint16_t>()[h])));
      rg[size_t(t) * Hv + h] = std::exp(gv);
      eg = std::max(eg, std::abs(std::exp(gv) - gg.as<float>()[t * Hv + h]));
    }
  }
  char name[96];
  snprintf(name, sizeof name, "clover_gdn_prep T=%d: q, k (unit vectors)", T);
  report(name, eq, 2e-5);
  snprintf(name, sizeof name, "clover_gdn_prep T=%d: v", T);
  report(name, ev, 2e-5);
  snprintf(name, sizeof name, "clover_gdn_prep T=%d: decay exp(g)", T);
  report(name, eg, 2e-6);

  // bd_gdn_local_memory on the kernel's own q / k / v / decay (so this check is the memory alone)
  double em = 0, mag = 0;
  for (int h = 0; h < Hv; ++h) {
    const int hk = h / (Hv / Hk);
    for (int l = 0; l < T; ++l) {
      const float* ql = q.as<float>() + (size_t(l) * Hk + hk) * Dh;
      std::vector<double> lg(size_t(T) + 1);
      for (int s = 0; s < T; ++s) {
        const float* ks = k.as<float>() + (size_t(s) * Hk + hk) * Dh;
        double d = 0;
        for (int i = 0; i < Dh; ++i) d += double(ql[i]) * std::sqrt(double(Dh)) * ks[i];
        lg[size_t(s)] = d * sc.as<float>()[h];
      }
      lg[size_t(T)] = sc.as<float>()[Hv + h] + sc.as<float>()[2 * Hv + h] * double(lr);
      const double m = *std::max_element(lg.begin(), lg.end());
      double z = 0;
      for (double& w : lg) z += (w = std::exp(w - m));
      for (double& w : lg) w /= z;
      for (int dv = 0; dv < Dh; ++dv) {
        double hist = 0;
        for (int i = 0; i < Dh; ++i) hist += double(st.as<float>()[(size_t(h) * Dh + dv) * Dh + i]) * ql[i];
        double o = lg[size_t(T)] * hist * gg.as<float>()[l * Hv + h];
        for (int s = 0; s < T; ++s) o += lg[size_t(s)] * v.as<float>()[(size_t(s) * Hv + h) * Dh + dv];
        em = std::max(em, std::abs(o - out.as<float>()[(size_t(l) * Hv + h) * Dh + dv]));
        mag = std::max(mag, std::abs(o));
      }
    }
  }
  snprintf(name, sizeof name, "clover_gdn_mem T=%d (outputs up to %.2f)", T, mag);
  report(name, em, 2e-5 * std::max(1.0, mag));
}

void check_coreset(Gpu& g, std::mt19937& rng, int T) {
  constexpr int E = 256, K = 8;
  const float top_p = 0.9f;
  std::normal_distribution<float> nd(0.0f, 2.0f);
  Buffer lg = g.alloc(size_t(T) * E * 4);
  std::vector<float> src(size_t(T) * E);
  for (float& x : src) x = nd(rng);
  std::copy(src.begin(), src.end(), lg.as<float>());
  auto e = g.begin();
  const CloverCoreset cp{uint32_t(T), E, K, top_p};
  e->use(g.pipeline("clover_coreset"));
  e->buf(0, lg);
  e->val(1, cp);
  e->dispatch(1, 1, 1, E);
  e->commit_and_wait();
  // coreset_torch: every row's top-k softmax mass per expert, normalized, kept while the mass before is < top_p
  std::vector<double> tot(E, 0.0);
  for (int t = 0; t < T; ++t) {
    const float* r = src.data() + size_t(t) * E;
    std::vector<int> ids(E);
    for (int i = 0; i < E; ++i) ids[size_t(i)] = i;
    std::stable_sort(ids.begin(), ids.end(), [&](int x, int y) { return r[x] > r[y]; });
    const double m = r[ids[0]];
    double z = 0;
    for (int i = 0; i < E; ++i) z += std::exp(double(r[i]) - m);
    for (int j = 0; j < K; ++j) tot[size_t(ids[size_t(j)])] += std::exp(double(r[ids[size_t(j)]]) - m) / z;
  }
  double sum = 0;
  for (double x : tot) sum += x;
  std::vector<int> order(E);
  for (int i = 0; i < E; ++i) order[size_t(i)] = i;
  std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return tot[size_t(x)] > tot[size_t(y)]; });
  std::vector<bool> keep(E, false);
  double before = 0;
  int kept = 0, borderline = 0, wrong = 0;
  for (int j = 0; j < E; ++j) {
    const double p = tot[size_t(order[size_t(j)])] / sum;
    keep[size_t(order[size_t(j)])] = j == 0 || before < top_p;
    if (std::abs(before - top_p) < 1e-5) ++borderline;
    kept += keep[size_t(order[size_t(j)])];
    before += p;
  }
  for (int t = 0; t < T; ++t)
    for (int i = 0; i < E; ++i) {
      const bool masked = lg.as<float>()[size_t(t) * E + i] < -1e29f;
      const bool same = masked ? !keep[size_t(i)] : keep[size_t(i)] && lg.as<float>()[size_t(t) * E + i] == src[size_t(t) * E + i];
      wrong += !same;
    }
  char name[96];
  snprintf(name, sizeof name, "clover_coreset T=%d: %d of %d experts kept", T, kept, E);
  report(name, borderline ? 0 : wrong, 0);
}

void check_pick(Gpu& g, std::mt19937& rng, int T) {
  constexpr int V = 248320, MASK = 151669;
  const uint32_t ids[3] = {248044, 248046, 248069};
  std::normal_distribution<float> nd(0.0f, 2.0f);
  Buffer lg = g.alloc(size_t(T) * V * 4), cnt = g.alloc(size_t(V) * 4), bid = g.alloc(64), bias = g.alloc(32 * 4 * 4);
  Buffer tok = g.alloc(64 * 4), rows = g.alloc(32 * sizeof(Row)), cand = g.alloc(32 * 4);
  std::uniform_int_distribution<int> any(0, V - 1);
  for (size_t i = 0; i < size_t(T) * V; ++i) lg.as<float>()[i] = nd(rng);
  std::fill(cnt.as<uint32_t>(), cnt.as<uint32_t>() + V, 0u);
  for (int t = 0; t < T; ++t) {
    const int id = any(rng);
    lg.as<float>()[size_t(t) * V + id] += t % 2 ? 25.0f : 9.0f;            // confident rows and unsure ones
    lg.as<float>()[size_t(t) * V + MASK] = 40.0f;                          // the mask token never wins
    if (t % 4 == 1) cnt.as<uint32_t>()[id] = 2;                            // a repeated token loses its lead
    cand.as<int>()[t] = t % 3 ? id : any(rng);                             // a verify row's draft: its argmax or not
  }
  for (int i = 0; i < 200; ++i) cnt.as<uint32_t>()[any(rng)] += 1;
  for (int i = 0; i < 3; ++i) bid.as<uint32_t>()[i] = ids[i];
  for (int t = 0; t < T; ++t)
    for (int i = 0; i < 4; ++i) bias.as<float>()[t * 4 + i] = i < 3 ? float((t * 7 + i * 3) % 11) : 0.0f;
  for (int i = 0; i < 64; ++i) tok.as<int>()[i] = -7;
  const uint32_t ncand = uint32_t(T) - 1;
  int wrong = 0, wrong_out = 0;
  double ec = 0, ep = 0, eh = 0, es = 0;
  for (const float temp : {0.0f, 0.8f}) {
    const CloverPick pk{V, V, MASK, 3, 0.1f, 0.9f, 1.05f, temp, 1, 0, 1, ncand, temp > 0 ? 1u : 0u};
    auto e = g.begin();
    e->use(g.pipeline("clover_pick"));
    e->buf(0, lg); e->buf(1, cnt); e->buf(2, bid); e->buf(3, bias); e->buf(4, tok, 4 * 4); e->buf(5, rows);
    e->val(6, pk);
    e->buf(7, cand);
    e->dispatch(uint32_t(T), 1, 1, 1024);
    e->commit_and_wait();
    for (int t = 0; t < T; ++t) {
      std::vector<double> l(V);
      for (int v = 0; v < V; ++v) {
        double x = lg.as<float>()[size_t(t) * V + v];
        if (v == MASK) x = -INFINITY;
        const uint32_t c = cnt.as<uint32_t>()[v];
        x -= double(pk.frequency) * c;
        if (c) x -= pk.presence;
        if (c) x = x < 0 ? x * pk.repetition : x / pk.repetition;
        for (int i = 0; i < 3; ++i)
          if (uint32_t(v) == ids[i]) x += bias.as<float>()[t * 4 + i];
        l[size_t(v)] = x;
      }
      const int am = int(std::max_element(l.begin(), l.end()) - l.begin());
      double z = 0, s1 = 0;
      for (double x : l)
        if (x > -INFINITY) z += std::exp(x - l[size_t(am)]), s1 += std::exp(x - l[size_t(am)]) * (x - l[size_t(am)]);
      const Row& r = rows.as<Row>()[t];
      wrong += r.tok != am;
      ec = std::max(ec, std::abs(1 / z - r.conf));
      eh = std::max(eh, std::abs(std::log(z) - s1 / z - r.ent));
      if (uint32_t(t) < ncand)
        ep = std::max(ep, std::abs(std::exp(l[size_t(cand.as<int>()[t])] - l[size_t(am)]) / z - r.pc));
      if (temp == 0.0f) wrong += r.smp != am;
      if (r.smp < 0 || r.smp >= V || r.smp == MASK) ++wrong;
      else es = std::max(es, std::abs(std::exp(l[size_t(r.smp)] - l[size_t(am)]) / z - r.conf_smp));
      // tok_out: rows from skip (1) on, the draw with a temperature, else the argmax; row 0 untouched
      const int want = t < 1 ? -7 : temp > 0 ? r.smp : r.tok;
      wrong_out += tok.as<int>()[4 + t] != want;
    }
    wrong_out += tok.as<int>()[3] != -7 || tok.as<int>()[4 + T] != -7;
  }
  char name[96];
  snprintf(name, sizeof name, "clover_pick T=%d: argmax and draws", T);
  report(name, wrong, 0);
  snprintf(name, sizeof name, "clover_pick T=%d: tokens written (row 0 skipped)", T);
  report(name, wrong_out, 0);
  snprintf(name, sizeof name, "clover_pick T=%d: argmax probability", T);
  report(name, ec, 1e-5);
  snprintf(name, sizeof name, "clover_pick T=%d: candidate probability", T);
  report(name, ep, 1e-5);
  snprintf(name, sizeof name, "clover_pick T=%d: draw probability", T);
  report(name, es, 1e-5);
  snprintf(name, sizeof name, "clover_pick T=%d: entropy", T);
  report(name, eh, 1e-4);
}

void check_attn(Gpu& g, std::mt19937& rng, int T, int pos0, uint32_t ns) {
  constexpr int Hq = 16, Hkv = 2, D = 256, cap = 1024;
  std::normal_distribution<float> nd(0.0f, 1.0f);
  Buffer q = g.alloc(size_t(T) * Hq * D * 4), qg = g.alloc(size_t(T) * Hq * 2 * D * 4);
  Buffer kc = g.alloc(size_t(Hkv) * cap * D * 2), vc = g.alloc(size_t(Hkv) * cap * D * 2);
  Buffer out = g.alloc(size_t(T) * Hq * D * 4), po = g.alloc(size_t(Hq) * 1024 * D * 4), pml = g.alloc(size_t(Hq) * 1024 * 8);
  for (size_t i = 0; i < size_t(T) * Hq * D; ++i) q.as<float>()[i] = nd(rng);
  for (size_t i = 0; i < size_t(T) * Hq * 2 * D; ++i) qg.as<float>()[i] = nd(rng);
  for (size_t i = 0; i < size_t(Hkv) * cap * D; ++i)
    kc.as<_Float16>()[i] = _Float16(nd(rng)), vc.as<_Float16>()[i] = _Float16(nd(rng));
  const kp::AttnPref fp{uint32_t(T), Hq, Hkv, uint32_t(pos0), cap, 1.0f / 16.0f};
  const uint32_t zero = 0;
  auto e = g.begin();
  e->use(g.pipeline("attn_prefill11gs_bidi"));
  e->buf(0, q); e->buf(1, kc); e->buf(2, vc); e->buf(3, qg); e->buf(4, out);
  e->val(5, fp);
  e->buf(6, po); e->buf(7, pml);
  e->val(8, ns);
  e->dispatch((uint32_t(T) * (Hq / Hkv) + 31) / 32, Hkv, ns, 256);
  e->use(g.pipeline("attn11_reduce"));
  e->buf(0, po); e->buf(1, pml); e->buf(2, qg); e->buf(3, out);
  e->val(4, fp);
  e->val(5, ns);
  e->val(6, zero);
  e->dispatch(uint32_t(T), Hq, 1, D);
  e->commit_and_wait();
  // every row over keys [0, pos0 + T), the block's later rows included; q rounded to half as the kernel loads it
  double err = 0;
  const int n = pos0 + T;
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < Hq; ++h) {
      const int kvh = h / (Hq / Hkv);
      std::vector<double> s(static_cast<size_t>(n));
      for (int j = 0; j < n; ++j) {
        double d = 0;
        for (int i = 0; i < D; ++i)
          d += double(_Float16(q.as<float>()[(size_t(t) * Hq + h) * D + i] / 16.0f)) *
               double(kc.as<_Float16>()[(size_t(kvh) * cap + j) * D + i]);
        s[size_t(j)] = d;
      }
      const double m = *std::max_element(s.begin(), s.end());
      double z = 0;
      for (double& x : s) z += (x = std::exp(x - m));
      for (int i = 0; i < D; ++i) {
        double o = 0;
        for (int j = 0; j < n; ++j) o += s[size_t(j)] * double(vc.as<_Float16>()[(size_t(kvh) * cap + j) * D + i]);
        const double gate = qg.as<float>()[(size_t(t) * Hq + h) * 2 * D + D + i];
        o = o / z / (1 + std::exp(-gate));
        err = std::max(err, std::abs(o - out.as<float>()[(size_t(t) * Hq + h) * D + i]));
      }
    }
  char name[96];
  snprintf(name, sizeof name, "attn_prefill11gs_bidi T=%d pos0=%d, %u split%s", T, pos0, ns, ns > 1 ? "s" : "");
  report(name, err, 3e-3);  // the kernel's softmax probabilities meet V as half fragments
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> Ts;
  for (int i = 1; i < argc; ++i) Ts.push_back(atoi(argv[i]));
  if (Ts.empty()) Ts = {32, 8};
  bench::GpuLock lock;
  if (!bench::GpuLock::held_by_caller() && !lock.acquire([](const std::string& who) {
        fprintf(stderr, "waiting for the GPU lock %s%s\n", bench::GpuLock::path().c_str(),
                who.empty() ? "" : (", held by " + who).c_str());
      })) {
    fprintf(stderr, "cannot take the GPU lock %s: %s\n", bench::GpuLock::path().c_str(), strerror(errno));
    return 2;
  }
  Gpu& g = Gpu::get();
  std::mt19937 rng(11);
  for (int T : Ts) {
    if (T < 1 || T > 32) {
      fprintf(stderr, "T: 1 to 32\n");
      return 2;
    }
    check_gdn(g, rng, T);
    check_coreset(g, rng, T);
    check_pick(g, rng, T);
    if (!getenv("CLOVER_NO_ATTN")) {
      check_attn(g, rng, T, 0, 1);
      check_attn(g, rng, T, 333, 1);
      check_attn(g, rng, T, 333, 3);
    }
  }
  printf("%s\n", fails ? "FAILED" : "all checks passed");
  return fails ? 1 : 0;
}
