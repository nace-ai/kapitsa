// Nemotron 3.5 Lightning's new kernels against CPU references at small odd sizes: the DSpark drafter's causal block
// attention (nh_dsp_attn, a ring that wraps), its mask rows (nh_dsp_mask_rows) and Markov walk over candidates
// (nh_dsp_walk, NVFP4 W2), the MoE sums of layers without latent projections (nh_moe2_sum2_add,
// nh_moe_combine_add, the fused one-token layer's nh_moe_ah / nh_moe_dh), and the 8-row BF16 product (nh_xb_v8:
// Lightning's attention in verify) and the F32 router of verify rows (nh_mv_f32_sk), the verify's sampling passes on
// simdgroup reductions against the threadgroup ones. One dispatch per command buffer; run it under
// MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1.
//   nh_dsp_check
#include "metal/gpu.h"
#include "model/dspark.h"
#include "model/nemotron_h.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace ni {
extern const char* const kNhKernelSource;
}
using namespace ni;

namespace {

uint64_t g_rng = 0x243F6A8885A308D3ull;
uint64_t rnd() {
  uint64_t z = (g_rng += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
float nrand() { return (float(rnd() & 0xFFFFFF) / 16777215.0f - 0.5f) * 2.0f; }

Buffer up(const void* p, size_t n, const char* label) {
  Buffer b = Gpu::get().alloc(std::max<size_t>(n, 16), label);
  memcpy(b.ptr(), p, n);
  return b;
}
template <typename T>
Buffer up(const std::vector<T>& v, const char* label) {
  return up(v.data(), v.size() * sizeof(T), label);
}

int g_fail = 0;
void report(const char* what, double err, double tol) {
  const bool ok = err <= tol;
  if (!ok) ++g_fail;
  printf("%-22s rel.err %.2e %s\n", what, err, ok ? "ok" : "FAIL");
}

double e4m3(uint8_t x) {
  const int e = (x >> 3) & 0xF, m = x & 7;
  const double v = e == 0 ? std::ldexp(double(m), -9) : std::ldexp(1.0 + double(m) / 8.0, e - 7);
  return (x & 0x80) ? -v : v;
}
const double kE2M1[16] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6};

// the block attention (nh_dsp_attn_part + nh_dsp_attn_fin): raw q|k|v rows, q / k norm + RoPE, a ring that wraps,
// context pieces (some empty), per-row window starts, against a double reference
void check_attn() {
  struct Case {
    int P, lo, W, NS;
  };
  for (const Case cs : {Case{37, 13, 24, 16}, Case{5, 0, 24, 16}, Case{200, 177, 24, 4}, Case{0, 0, 24, 16}, Case{700, 400, 300, 4}, Case{1100, 76, 1024, 16}}) {
    const int HD = 128, Hq = 4, Hkv = 2, B = 8, G = Hq / Hkv, P = cs.P, W = cs.W, NS = cs.NS;
    const int ld = (Hq + 2 * Hkv) * HD;
    std::vector<float> qkv(size_t(B) * ld), qn(HD), kn(HD), sinks(Hq);
    std::vector<__fp16> rk(size_t(Hkv) * W * HD), rv(rk.size());
    for (auto& x : qkv) x = nrand() * 2.0f;
    for (auto& x : qn) x = 1.0f + 0.5f * nrand();
    for (auto& x : kn) x = 1.0f + 0.5f * nrand();
    for (auto& x : rk) x = __fp16(nrand());
    for (auto& x : rv) x = __fp16(nrand());
    for (auto& x : sinks) x = nrand();
    const float scale = 1.0f / std::sqrt(float(HD)), eps = 1e-6f, theta = 10000.0f;
    struct {
      uint32_t P, lo, W, Hq, Hkv, B, NS, ld;
      float scale, eps, theta;
    } p{uint32_t(P), uint32_t(cs.lo), uint32_t(W), uint32_t(Hq), uint32_t(Hkv), uint32_t(B), uint32_t(NS), uint32_t(ld),
        scale, eps, theta};
    Gpu& g = Gpu::get();
    Buffer bq = up(qkv, "qkv"), bqn = up(qn, "qn"), bkn = up(kn, "kn"), brk = up(rk, "rk"), brv = up(rv, "rv"),
           bs = up(sinks, "s");
    Buffer part = g.alloc(size_t(B) * Hq * NS * (HD + 2) * 4, "part"), out = g.alloc(size_t(B) * Hq * HD * 2, "out"),
           tab = g.alloc(size_t(B) * HD / 2 * 8, "rope");
    auto e0 = g.begin();
    e0->use(NemotronH::pipeline("nh_dsp_rope_tab"));
    e0->buf(0, tab);
    e0->bytes(1, &p, sizeof p);
    e0->dispatch(1, 1, 1, uint32_t(B * HD / 2));
    e0->commit_and_wait();
    auto e = g.begin();
    e->use(NemotronH::pipeline("nh_dsp_attn_part"));
    e->buf(0, bq);
    e->buf(1, bqn);
    e->buf(2, brk);
    e->buf(3, brv);
    e->buf(4, part);
    e->bytes(5, &p, sizeof p);
    e->buf(6, tab);
    e->dispatch(uint32_t(Hkv), uint32_t(B), uint32_t(NS), uint32_t(32 * G));
    e->commit_and_wait();
    auto e2 = g.begin();
    e2->use(NemotronH::pipeline("nh_dsp_attn_fin"));
    e2->buf(0, bq);
    e2->buf(1, bqn);
    e2->buf(2, bkn);
    e2->buf(3, bs);
    e2->buf(4, part);
    e2->buf(5, out);
    e2->bytes(6, &p, sizeof p);
    e2->buf(7, tab);
    e2->dispatch(uint32_t(Hkv), uint32_t(B), 1, uint32_t(32 * G));
    e2->commit_and_wait();
    // reference: per-head RMSNorm, NeoX RoPE
    auto norm_rope = [&](const float* x, const std::vector<float>& w, int pos) {
      double ss = 0;
      for (int i = 0; i < HD; ++i) ss += double(x[i]) * x[i];
      const double inv = 1.0 / std::sqrt(ss / HD + eps);
      std::vector<double> n(HD), o(HD);
      for (int i = 0; i < HD; ++i) n[size_t(i)] = x[i] * inv * w[size_t(i)];
      for (int d = 0; d < HD; ++d) {
        const int i = d % (HD / 2);
        const double ang = double(pos) * std::pow(double(theta), -2.0 * i / HD);
        o[size_t(d)] = d < HD / 2 ? n[size_t(d)] * std::cos(ang) - n[size_t(d + HD / 2)] * std::sin(ang)
                                  : n[size_t(d)] * std::cos(ang) + n[size_t(d - HD / 2)] * std::sin(ang);
      }
      return o;
    };
    double err = 0, mx = 0;
    for (int r = 0; r < B; ++r)
      for (int h = 0; h < Hq; ++h) {
        const int kvh = h / G, qpos = P + r;
        std::vector<double> q = norm_rope(&qkv[size_t(r) * ld + size_t(h) * HD], qn, qpos);
        std::vector<std::vector<double>> keys, vals;
        for (int t = std::max(cs.lo, qpos + 1 - W); t < P; ++t) {
          std::vector<double> k(HD), v(HD);
          for (int i = 0; i < HD; ++i) {
            k[size_t(i)] = float(rk[(size_t(kvh) * W + size_t(t % W)) * HD + size_t(i)]);
            v[size_t(i)] = float(rv[(size_t(kvh) * W + size_t(t % W)) * HD + size_t(i)]);
          }
          keys.push_back(k);
          vals.push_back(v);
        }
        for (int j = 0; j <= r; ++j) {
          keys.push_back(norm_rope(&qkv[size_t(j) * ld + size_t(Hq + kvh) * HD], kn, P + j));
          std::vector<double> v(HD);
          for (int i = 0; i < HD; ++i) v[size_t(i)] = qkv[size_t(j) * ld + size_t(Hq + Hkv + kvh) * HD + size_t(i)];
          vals.push_back(v);
        }
        std::vector<double> sc;
        double m = sinks[size_t(h)];
        for (auto& k : keys) {
          double d = 0;
          for (int i = 0; i < HD; ++i) d += q[size_t(i)] * scale * k[size_t(i)];
          sc.push_back(d);
          m = std::max(m, d);
        }
        double den = std::exp(double(sinks[size_t(h)]) - m);
        std::vector<double> o(HD, 0.0);
        for (size_t j = 0; j < keys.size(); ++j) {
          const double w = std::exp(sc[j] - m);
          den += w;
          for (int i = 0; i < HD; ++i) o[size_t(i)] += w * vals[j][size_t(i)];
        }
        for (int i = 0; i < HD; ++i) {
          const double ref = o[size_t(i)] / den;
          err = std::max(err, std::fabs(ref - double(out.as<__fp16>()[(size_t(r) * Hq + h) * HD + size_t(i)])));
          mx = std::max(mx, std::fabs(ref));
        }
      }
    char name[64];
    snprintf(name, sizeof name, "nh_dsp_attn P%d NS%d", P, NS);
    report(name, err / mx, 2e-3);
    g.release(part);
    g.release(out);
    g.release(tab);
  }
}

void check_mask_rows() {
  const uint32_t D = 300, B = 8;
  std::vector<float> x(size_t(B) * D), row(D);
  for (auto& v : x) v = nrand();
  for (auto& v : row) v = nrand();
  Gpu& g = Gpu::get();
  Buffer bx = up(x, "x"), br = up(row, "row");
  auto e = g.begin();
  e->use(NemotronH::pipeline("nh_dsp_mask_rows"));
  e->buf(0, bx);
  e->buf(1, br);
  e->val(2, D);
  e->dispatch(B - 1, (D + 255) / 256, 1, 256);
  e->commit_and_wait();
  double err = 0;
  for (uint32_t r = 0; r < B; ++r)
    for (uint32_t d = 0; d < D; ++d) err = std::max(err, double(std::fabs(bx.as<float>()[r * D + d] - (r ? row[d] : x[d]))));
  report("nh_dsp_mask_rows", err, 0);
}

std::string walk_name(const char* what, int V, int R, int K) {
  char b[96];
  snprintf(b, sizeof b, "%s V%d R%d K%d", what, V, R, K);
  return b;
}

void check_walk(const int V, const int R, const int K) {
  // V tokens, rank R (NVFP4 W2 in the loader's layout), S slots of K candidates
  const int S = 7, P = 3;
  std::vector<uint16_t> w1(size_t(V) * R);
  std::vector<float> w1f(w1.size());
  for (size_t i = 0; i < w1.size(); ++i) {
    const float f = nrand();
    uint32_t u;
    memcpy(&u, &f, 4);
    w1[i] = uint16_t(u >> 16);
    u = uint32_t(w1[i]) << 16;
    memcpy(&w1f[i], &u, 4);
  }
  // NVFP4 rows: ModelOpt order (byte j = elements 2j, 2j + 1), then the loader's repack
  std::vector<uint8_t> src(size_t(V) * R / 2), q(src.size()), sc(size_t(V) * R / 16);
  for (auto& b : src) b = uint8_t(rnd());
  for (auto& b : sc) b = uint8_t(0x30 + rnd() % 0x20);
  nh_repack_rows(NhFmt::FP4, R, src.data(), V, q.data(), nullptr);
  const float gs = 0.37f;
  std::vector<int> cand(size_t(S) * K), tok(16, 0);
  std::vector<float> clp(size_t(S) * K);
  for (int s = 0; s < S; ++s) {
    float l = -0.5f;
    for (int c = 0; c < K; ++c) {
      cand[size_t(s) * K + c] = int(rnd() % V);
      clp[size_t(s) * K + c] = l;
      l -= 0.3f + 0.2f * float(rnd() % 4);
    }
  }
  tok[P] = 11;
  auto w2 = [&](int id, int k) {  // dequantized W2[id][k] from the stored bytes
    const uint8_t b = src[size_t(id) * R / 2 + size_t(k / 2)];
    return kE2M1[(k & 1) ? b >> 4 : b & 15] * e4m3(sc[size_t(id) * R / 16 + size_t(k / 16)]) * gs;
  };
  std::vector<int> ref_tok(S);
  std::vector<double> ref_q(size_t(S) * K);
  int prev = tok[P];
  for (int s = 0; s < S; ++s) {
    std::vector<double> sc2(K);
    int best = 0;
    double base = 0;
    for (int c = 0; c < K; ++c) {
      double d = 0;
      for (int k = 0; k < R; ++k) d += w2(cand[size_t(s) * K + c], k) * w1f[size_t(prev) * R + k];
      sc2[size_t(c)] = clp[size_t(s) * K + c] + d;
      base += std::exp(double(clp[size_t(s) * K + c]));
      if (sc2[size_t(c)] > sc2[size_t(best)]) best = c;
    }
    double den = std::max(0.0, 1.0 - base);
    for (int c = 0; c < K; ++c) den += std::exp(sc2[size_t(c)]);
    for (int c = 0; c < K; ++c) ref_q[size_t(s) * K + c] = sc2[size_t(c)] - std::log(den);
    prev = ref_tok[size_t(s)] = cand[size_t(s) * K + best];
  }
  Gpu& g = Gpu::get();
  Buffer bc = up(cand, "cand"), bl = up(clp, "clp"), b1 = up(w1, "w1"), bq = up(q, "w2q"), bsc = up(sc, "w2s"),
         bt = up(tok, "tok"), bp = g.alloc(64, "picks"), bql = g.alloc(size_t(S) * K * 4, "qlp");
  struct {
    uint32_t S, K, R, P;
    float gs;
  } p{uint32_t(S), uint32_t(K), uint32_t(R), uint32_t(P), gs};
  auto e = g.begin();
  e->use(NemotronH::pipeline("nh_dsp_walk"));
  e->buf(0, bc);
  e->buf(1, bl);
  e->buf(2, b1);
  e->buf(3, bq);
  e->buf(4, bsc);
  e->buf(5, bt);
  e->buf(6, bp);
  e->buf(7, bql);
  e->bytes(8, &p, sizeof p);
  e->dispatch(1, 1, 1, 32 * K);
  e->commit_and_wait();
  int bad = 0;
  double err = 0;
  for (int s = 0; s < S; ++s) {
    bad += bt.as<int>()[P + 1 + s] != ref_tok[size_t(s)];
    for (int c = 0; c < K; ++c) err = std::max(err, std::fabs(ref_q[size_t(s) * K + c] - bql.as<float>()[s * K + c]));
  }
  report(walk_name("nh_dsp_walk tokens", V, R, K).c_str(), double(bad), 0);
  // the two-pass walk (nh_dsp_walk_dots + nh_dsp_walk_pick): bitwise the one-pass kernel's tokens, picks, log-probs
  {
    Buffer bt2 = up(tok, "tok2"), bp2 = g.alloc(64, "picks2"), bq2 = g.alloc(size_t(S) * K * 4, "qlp2"),
           bd = g.alloc(size_t(S) * K * K * 4, "dots");
    auto e2 = g.begin();
    e2->use(NemotronH::pipeline("nh_dsp_walk_dots"));
    e2->buf(0, bc);
    e2->buf(1, b1);
    e2->buf(2, bq);
    e2->buf(3, bsc);
    e2->buf(4, bt2);
    e2->buf(5, bd);
    e2->bytes(6, &p, sizeof p);
    e2->dispatch(uint32_t(S), 1, 1, uint32_t(32 * K));
    e2->commit_and_wait();
    auto e3 = g.begin();
    e3->use(NemotronH::pipeline("nh_dsp_walk_pick"));
    e3->buf(0, bc);
    e3->buf(1, bl);
    e3->buf(2, bd);
    e3->buf(3, bt2);
    e3->buf(4, bp2);
    e3->buf(5, bq2);
    e3->bytes(6, &p, sizeof p);
    e3->dispatch(1, 1, 1, 32);
    e3->commit_and_wait();
    const bool same = !memcmp(bt.ptr(), bt2.ptr(), 64) && !memcmp(bp.ptr(), bp2.ptr(), size_t(S) * 4) &&
                      !memcmp(bql.ptr(), bq2.ptr(), size_t(S) * K * 4);
    report(walk_name("nh_dsp_walk two-pass", V, R, K).c_str(), same ? 0.0 : 1.0, 0);
  }
  report(walk_name("nh_dsp_walk log-probs", V, R, K).c_str(), err, 1e-4);
}

void check_sums() {
  const uint32_t R = 3, k = 5, D = 200;
  std::vector<float> part(size_t(R) * k * D), x(size_t(R) * D), ew(size_t(R) * k);
  for (auto& v : part) v = nrand();
  for (auto& v : x) v = nrand();
  for (auto& v : ew) v = nrand();
  Gpu& g = Gpu::get();
  {
    Buffer bp = up(part, "part"), bx = up(x, "x");
    const uint32_t live[4] = {2, 0, 0, 0};  // row 2 is past the live rows: untouched
    Buffer bl = up(live, sizeof live, "live");
    const uint32_t mp[8] = {0, D, k, 0, 0, 0, 0, 0};
    auto e = g.begin();
    e->use(NemotronH::pipeline("nh_moe2_sum2_add"));
    e->buf(0, bp);
    e->buf(1, bx);
    e->bytes(2, mp, sizeof mp);
    e->buf(3, bl);
    e->dispatch((D + 63) / 64, R, 1, 64);
    e->commit_and_wait();
    double err = 0;
    for (uint32_t r = 0; r < R; ++r)
      for (uint32_t d = 0; d < D; ++d) {
        double ref = x[r * D + d];
        if (r < 2)
          for (uint32_t j = 0; j < k; ++j) ref += part[(size_t(r) * k + j) * D + d];
        err = std::max(err, std::fabs(ref - bx.as<float>()[r * D + d]));
      }
    report("nh_moe2_sum2_add", err, 1e-5);
  }
  {
    Buffer bp = up(part, "out"), bw = up(ew, "ew"), bx = up(x, "x");
    const uint32_t cp[2] = {D, k};
    auto e = g.begin();
    e->use(NemotronH::pipeline("nh_moe_combine_add"));
    e->buf(0, bp);
    e->buf(1, bw);
    e->buf(2, bx);
    e->bytes(3, cp, sizeof cp);
    e->dispatch(R, 1, 1, 256);
    e->commit_and_wait();
    double err = 0;
    for (uint32_t r = 0; r < R; ++r)
      for (uint32_t d = 0; d < D; ++d) {
        double ref = x[r * D + d];
        for (uint32_t j = 0; j < k; ++j) ref += double(ew[r * k + j]) * part[(size_t(r) * k + j) * D + d];
        err = std::max(err, std::fabs(ref - bx.as<float>()[r * D + d]));
      }
    report("nh_moe_combine_add", err, 1e-5);
  }
}

void check_xb_v8() {
  // N outputs (a partial 16-row tile), K = 3 x 128, R rows of half X, every epilogue but relu^2
  const uint32_t N = 37, K = 384;
  for (uint32_t R : {1u, 5u, 8u})
    for (uint32_t epi : {0u, 1u, 3u}) {
      std::vector<uint16_t> w(size_t(N) * K);
      std::vector<__fp16> x(size_t(R) * K);
      std::vector<float> y0(size_t(R) * N);
      for (auto& v : w) {
        const float f = nrand() * 0.5f;
        uint32_t u;
        memcpy(&u, &f, 4);
        v = uint16_t(u >> 16);
      }
      for (auto& v : x) v = __fp16(nrand());
      for (auto& v : y0) v = nrand();
      Gpu& g = Gpu::get();
      Buffer bw = up(w, "w"), bx = up(x, "x"), by = up(y0, "y");
      struct {
        uint32_t N, K, x_ld, y_ld, R;
        float gs;
        uint32_t epi;
      } p{N, K, K, N, R, 1.0f, epi};
      auto e = g.begin();
      e->use(NemotronH::pipeline("nh_xb_v8"));
      e->buf(0, bx);
      e->buf(1, bw);
      e->buf(2, bw);
      e->buf(3, by);
      e->bytes(4, &p, sizeof p);
      e->dispatch((N + 15) / 16, 1, 1, 256);
      e->commit_and_wait();
      double err = 0, mx = 0;
      for (uint32_t r = 0; r < R; ++r)
        for (uint32_t n = 0; n < N; ++n) {
          double ref = 0;
          for (uint32_t k = 0; k < K; ++k) {
            const uint32_t u = uint32_t(w[size_t(n) * K + k]) << 16;
            float f;
            memcpy(&f, &u, 4);
            ref += double(__fp16(f)) * double(x[size_t(r) * K + k]);
          }
          double got;
          if (epi == 3) got = double(((__fp16*)by.ptr())[size_t(r) * N + n]);
          else got = by.as<float>()[size_t(r) * N + n] - (epi == 1 ? y0[size_t(r) * N + n] : 0.0f);
          err = std::max(err, std::fabs(got - ref));
          mx = std::max(mx, std::fabs(ref));
        }
      char name[64];
      snprintf(name, sizeof name, "nh_xb_v8 R%u epi%u", R, epi);
      report(name, err / mx, epi == 3 ? 2e-3 : 1e-5);
    }
}

void check_moe_ah() {
  // E experts (2 router threadgroups), top-k, H inputs, S shared-expert rows (NVFP4 in the loader's layout)
  const uint32_t E = 20, k = 3, H = 96, S = 37;
  std::vector<float> xn(H), rw(size_t(E) * H), bias(E);
  for (auto& v : xn) v = nrand();
  for (auto& v : rw) v = nrand() * 0.3f;
  for (auto& v : bias) v = nrand() * 0.1f;
  std::vector<uint8_t> src(size_t(S) * H / 2), q(src.size()), sc(size_t(S) * H / 16);
  for (auto& b : src) b = uint8_t(rnd());
  for (auto& b : sc) b = uint8_t(0x28 + rnd() % 0x10);
  nh_repack_rows(NhFmt::FP4, H, src.data(), S, q.data(), nullptr);
  const float gs = 0.8f, scale = 2.5f;
  Gpu& g = Gpu::get();
  Buffer bx = up(xn, "xn"), brw = up(rw, "rw"), bq = up(q, "uq"), bs = up(sc, "us"), bb = up(bias, "bias");
  Buffer lg = g.alloc(E * 4, "logits"), sh = g.alloc(S * 4, "sh"), eid = g.alloc(k * 4, "eid"), ew = g.alloc(k * 4, "ew");
  Buffer ctr = g.alloc(16, "ctr");
  memset(ctr.ptr(), 0, 16);
  struct {
    uint32_t H, E, k, norm;
    float scale;
    uint32_t L, S, I, uqs, uss, tg_r, tg_l, tg_u;
  } p{H, E, k, 1, scale, H, S, 1, 0, 0, (E + 15) / 16, 0, 0};
  auto e = g.begin();
  e->use(NemotronH::pipeline("nh_moe_ah"));
  e->buf(0, bx);
  e->buf(1, brw);
  e->buf(4, bq);
  e->buf(5, bs);
  e->buf(6, lg);
  e->buf(8, sh);
  e->buf(9, bb);
  e->buf(10, eid);
  e->buf(11, ew);
  e->buf(12, ctr);
  e->bytes(13, &p, sizeof p);
  e->bytes(14, &gs, 4);
  e->dispatch(p.tg_r + (S + 15) / 16, 1, 1, 128);
  e->commit_and_wait();
  // reference: logits, sigmoid + bias selection, normalized weights x scale; shared up relu^2
  std::vector<double> prob(E), sel(E);
  double lerr = 0, lmx = 0;
  for (uint32_t i = 0; i < E; ++i) {
    double l = 0;
    for (uint32_t j = 0; j < H; ++j) l += double(rw[size_t(i) * H + j]) * xn[j];
    lerr = std::max(lerr, std::fabs(l - lg.as<float>()[i]));
    lmx = std::max(lmx, std::fabs(l));
    prob[i] = 1.0 / (1.0 + std::exp(-l));
    sel[i] = prob[i] + bias[i];
  }
  std::vector<int> ids;
  double sum = 1e-20;
  for (uint32_t j = 0; j < k; ++j) {
    int b = -1;
    for (uint32_t i = 0; i < E; ++i)
      if (std::find(ids.begin(), ids.end(), int(i)) == ids.end() && (b < 0 || sel[i] > sel[size_t(b)])) b = int(i);
    ids.push_back(b);
    sum += prob[size_t(b)];
  }
  int bad = 0;
  double werr = 0;
  for (uint32_t j = 0; j < k; ++j) {
    bad += eid.as<int>()[j] != ids[j];
    werr = std::max(werr, std::fabs(prob[size_t(ids[j])] / sum * scale - ew.as<float>()[j]));
  }
  double serr = 0, smx = 0;
  for (uint32_t n = 0; n < S; ++n) {
    double d = 0;
    for (uint32_t j = 0; j < H; ++j) {
      const uint8_t b = src[size_t(n) * H / 2 + j / 2];
      d += kE2M1[(j & 1) ? b >> 4 : b & 15] * e4m3(sc[size_t(n) * H / 16 + j / 16]) * xn[j];
    }
    d = std::max(d * gs, 0.0);
    serr = std::max(serr, std::fabs(d * d - sh.as<float>()[n]));
    smx = std::max(smx, d * d);
  }
  report("nh_moe_ah logits", lerr / lmx, 1e-5);
  report("nh_moe_ah experts", double(bad), 0);
  report("nh_moe_ah weights", werr, 1e-5);
  report("nh_moe_ah shared up", serr / smx, 1e-5);
  // nh_moe_dh: x += sum of k partials + sd
  std::vector<float> part(size_t(k) * H), sd(H), x(H);
  for (auto& v : part) v = nrand();
  for (auto& v : sd) v = nrand();
  for (auto& v : x) v = nrand();
  Buffer bp = up(part, "part"), bsd = up(sd, "sd"), bxx = up(x, "x");
  e = g.begin();
  e->use(NemotronH::pipeline("nh_moe_dh"));
  e->buf(0, bp);
  e->buf(1, bsd);
  e->buf(2, bxx);
  e->bytes(3, &p, sizeof p);
  e->dispatch((H + 255) / 256, 1, 1, 256);
  e->commit_and_wait();
  double derr = 0;
  for (uint32_t n = 0; n < H; ++n) {
    double ref = double(x[n]) + sd[n];
    for (uint32_t j = 0; j < k; ++j) ref += part[size_t(j) * H + n];
    derr = std::max(derr, std::fabs(ref - bxx.as<float>()[n]));
  }
  report("nh_moe_dh", derr, 1e-5);
}

void check_f32_sk() {
  const uint32_t N = 13, K = 300;
  for (uint32_t R : {2u, 5u, 8u}) {
    std::vector<float> w(size_t(N) * K), x(size_t(R) * (K + 8));
    for (auto& v : w) v = nrand();
    for (auto& v : x) v = nrand();
    Gpu& g = Gpu::get();
    Buffer bw = up(w, "w"), bx = up(x, "x"), by = g.alloc(size_t(R) * (N + 3) * 4, "y");
    struct {
      uint32_t N, K, xs, ys;
      float gs;
      uint32_t R;
    } p{N, K, K + 8, N + 3, 1.0f, R};
    auto e = g.begin();
    e->use(NemotronH::pipeline("nh_mv_f32_sk"));
    e->buf(0, bw);
    e->buf(2, bx);
    e->buf(3, by);
    e->bytes(4, &p, sizeof p);
    e->dispatch(N, 1, 1, 256);
    e->commit_and_wait();
    double err = 0, mx = 0;
    for (uint32_t r = 0; r < R; ++r)
      for (uint32_t n = 0; n < N; ++n) {
        double ref = 0;
        for (uint32_t k = 0; k < K; ++k) ref += double(w[size_t(n) * K + k]) * x[size_t(r) * (K + 8) + k];
        err = std::max(err, std::fabs(ref - by.as<float>()[size_t(r) * (N + 3) + n]));
        mx = std::max(mx, std::fabs(ref));
      }
    char name[64];
    snprintf(name, sizeof name, "nh_mv_f32_sk R%u", R);
    report(name, err / mx, 1e-6);
  }
}

// the candidates (nh_cand1 + nh_cand2, LiLiCorr's and DSpark's): S rows of V logits with many ties (a few distinct
// values) and a partial last chunk, against a CPU top-K by (value descending, id ascending); log-probs against a
// double log-softmax; V 5000 (3 chunks) and 131000 (64 chunks, two per lane of the merge, as Lightning's vocabulary)
void check_cand() {
  for (const uint32_t V : {5000u, 131000u})
  for (const uint32_t K : {16u, 32u}) {
    const uint32_t S = 3, nch = (V + 2047) / 2048;
    std::vector<float> lg(size_t(S) * V);
    for (auto& x : lg) x = float(int(rnd() % 40)) * 0.25f - 5.0f;
    Gpu& g = Gpu::get();
    Buffer bl = up(lg, "logits");
    Buffer cv = g.alloc(size_t(S) * nch * K * 4, "cv"), ci = g.alloc(size_t(S) * nch * K * 4, "ci"),
           ms = g.alloc(size_t(S) * nch * 8, "ms"), cand = g.alloc(size_t(S) * K * 4, "cand"),
           clp = g.alloc(size_t(S) * K * 4, "clp");
    const struct {
      uint32_t V, K, nchunks, row0, ld;
    } p{V, K, nch, 0, V};
    auto e = g.begin();
    e->use(NemotronH::pipeline("nh_cand1"));
    e->buf(0, bl);
    e->buf(1, cv);
    e->buf(2, ci);
    e->buf(3, ms);
    e->bytes(4, &p, sizeof p);
    e->dispatch(nch, S, 1, 256);
    e->commit_and_wait();
    auto e2 = g.begin();
    e2->use(NemotronH::pipeline("nh_cand2"));
    e2->buf(0, cv);
    e2->buf(1, ci);
    e2->buf(2, ms);
    e2->buf(3, cand);
    e2->buf(4, clp);
    e2->bytes(5, &p, sizeof p);
    e2->dispatch(S, 1, 1, 32);
    e2->commit_and_wait();
    int bad = 0;
    double err = 0;
    for (uint32_t s = 0; s < S; ++s) {
      const float* l = &lg[size_t(s) * V];
      std::vector<uint32_t> idx(V);
      for (uint32_t i = 0; i < V; ++i) idx[i] = i;
      std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return l[a] > l[b]; });
      double m = -1e30, den = 0;
      for (uint32_t i = 0; i < V; ++i) m = std::max(m, double(l[i]));
      for (uint32_t i = 0; i < V; ++i) den += std::exp(double(l[i]) - m);
      for (uint32_t r = 0; r < K; ++r) {
        if (uint32_t(cand.as<int>()[s * K + r]) != idx[r]) ++bad;
        err = std::max(err, std::fabs(double(l[idx[r]]) - m - std::log(den) - clp.as<float>()[s * K + r]));
      }
    }
    char name[64];
    snprintf(name, sizeof name, "nh_cand V%u K%u ids", V, K);
    report(name, double(bad), 0);
    snprintf(name, sizeof name, "nh_cand V%u K%u log-probs", V, K);
    report(name, err, 1e-5);
  }
}

// DSpark's fused kernels against the kernels they replace (bitwise): nh_dsp_swiglu_half = the scaled products,
// nh_swiglu, nh_rows_half_scaled; nh_dsp_scale_add = nh_rows_scale, nh_add; nh_dsp_commit_kv = nh_qk_norm_rope (k),
// nh_lili_ring_write
void check_dsp_fused() {
  Gpu& g = Gpu::get();
  const uint32_t R = 5, I = 300, D = 200;
  {
    std::vector<float> x(size_t(R) * 2 * I);
    for (auto& v : x) v = nrand() * 300.0f;  // some rows past half's range after the product
    const float gs_g = 0.37f, gs_u = 1.9f;
    std::vector<float> a(size_t(R) * I), b(size_t(R) * I);
    for (uint32_t r = 0; r < R; ++r)
      for (uint32_t k = 0; k < I; ++k) {
        a[r * I + k] = x[r * 2 * I + k] * gs_g;
        b[r * I + k] = x[r * 2 * I + I + k] * gs_u;
      }
    Buffer bx = up(x, "x"), ba = up(a, "a"), bb = up(b, "b");
    Buffer y1 = g.alloc(size_t(R) * I * 2, "y1"), y2 = g.alloc(size_t(R) * I * 2, "y2"), s1 = g.alloc(R * 4, "s1"),
           s2 = g.alloc(R * 4, "s2");
    const struct {
      uint32_t I, ld;
      float gs_g, gs_u;
    } sw{I, 2 * I, gs_g, gs_u};
    auto e = g.begin();
    e->use(NemotronH::pipeline("nh_dsp_swiglu_half"));
    e->buf(0, bx);
    e->buf(1, y1);
    e->buf(2, s1);
    e->bytes(3, &sw, sizeof sw);
    e->dispatch(R, 1, 1, 1024);
    e->commit_and_wait();
    const uint32_t n = R * I;
    auto e2 = g.begin();
    e2->use(NemotronH::pipeline("nh_swiglu"));
    e2->buf(0, ba);
    e2->buf(1, bb);
    e2->buf(2, ba);
    e2->bytes(3, &n, 4);
    e2->dispatch((n + 255) / 256, 1, 1, 256);
    e2->commit_and_wait();
    auto e3 = g.begin();
    e3->use(NemotronH::pipeline("nh_rows_half_scaled"));
    e3->buf(0, ba);
    e3->buf(1, y2);
    e3->buf(2, s2);
    e3->bytes(3, &I, 4);
    e3->dispatch(R, 1, 1, 1024);
    e3->commit_and_wait();
    // (a, b computed on the CPU: the GEMM's stores of acc x gs are the same fp32 products)
    const bool same = !memcmp(y1.ptr(), y2.ptr(), size_t(R) * I * 2) && !memcmp(s1.ptr(), s2.ptr(), R * 4);
    int big = 0;
    for (uint32_t r = 0; r < R; ++r) big += s1.as<float>()[r] > 1.0f;
    report(big ? "nh_dsp_swiglu_half" : "nh_dsp_swiglu_half (no scaled row)", same ? 0.0 : 1.0, 0);
  }
  {
    std::vector<float> x(size_t(R) * D), t(size_t(R) * D), sc(R);
    for (auto& v : x) v = nrand();
    for (auto& v : t) v = nrand();
    for (auto& v : sc) v = 1.0f + std::fabs(nrand()) * 7.0f;
    Buffer x1 = up(x, "x1"), x2 = up(x, "x2"), t1 = up(t, "t1"), t2 = up(t, "t2"), bs = up(sc, "sc");
    auto e = g.begin();
    e->use(NemotronH::pipeline("nh_dsp_scale_add"));
    e->buf(0, x1);
    e->buf(1, t1);
    e->buf(2, bs);
    e->bytes(3, &D, 4);
    e->dispatch(R, (D + 255) / 256, 1, 256);
    e->commit_and_wait();
    auto e2 = g.begin();
    e2->use(NemotronH::pipeline("nh_rows_scale"));
    e2->buf(0, t2);
    e2->buf(1, bs);
    e2->bytes(2, &D, 4);
    e2->dispatch(R, (D + 255) / 256, 1, 256);
    e2->commit_and_wait();
    const uint32_t n = R * D;
    auto e3 = g.begin();
    e3->use(NemotronH::pipeline("nh_add"));
    e3->buf(0, x2);
    e3->buf(1, t2);
    e3->buf(2, x2);
    e3->bytes(3, &n, 4);
    e3->dispatch((n + 255) / 256, 1, 1, 256);
    e3->commit_and_wait();
    report("nh_dsp_scale_add", memcmp(x1.ptr(), x2.ptr(), size_t(n) * 4) ? 1.0 : 0.0, 0);
  }
  {
    const uint32_t Hkv = 2, HD = 128, W = 7, pos0 = 11, Rk = 4;
    std::vector<float> kv(size_t(Rk) * 2 * Hkv * HD), w(HD);
    for (auto& v : kv) v = nrand() * 3.0f;
    for (auto& v : w) v = nrand();
    std::vector<float> k(size_t(Rk) * Hkv * HD), v(k.size());
    for (uint32_t r = 0; r < Rk; ++r)
      for (uint32_t i = 0; i < Hkv * HD; ++i) {
        k[r * Hkv * HD + i] = kv[r * 2 * Hkv * HD + i];
        v[r * Hkv * HD + i] = kv[r * 2 * Hkv * HD + Hkv * HD + i];
      }
    Buffer bkv = up(kv, "kv"), bw = up(w, "w"), bk = up(k, "k"), bv = up(v, "v");
    std::vector<uint16_t> zero(size_t(Hkv) * W * HD, 0x7e00);
    Buffer rk1 = up(zero, "rk1"), rv1 = up(zero, "rv1"), rk2 = up(zero, "rk2"), rv2 = up(zero, "rv2");
    const float eps = 1e-6f, theta = 10000.0f;
    const struct {
      uint32_t pos0, W, Hkv, HD, ld;
      float eps, theta;
    } kp{pos0, W, Hkv, HD, 2 * Hkv * HD, eps, theta};
    auto e = g.begin();
    e->use(NemotronH::pipeline("nh_dsp_commit_kv"));
    e->buf(0, bkv);
    e->buf(1, bw);
    e->buf(2, rk1);
    e->buf(3, rv1);
    e->bytes(4, &kp, sizeof kp);
    e->dispatch(Rk, 2 * Hkv, 1, HD);
    e->commit_and_wait();
    const struct {
      uint32_t H, HD, pos0, rope;
      float eps, theta;
    } qp{Hkv, HD, pos0, 1, eps, theta};
    auto e2 = g.begin();
    e2->use(NemotronH::pipeline("nh_qk_norm_rope"));
    e2->buf(0, bk);
    e2->buf(1, bw);
    e2->bytes(2, &qp, sizeof qp);
    e2->dispatch(Rk, Hkv, 1, HD);
    e2->commit_and_wait();
    const struct {
      uint32_t pos0, W, Hkv, HD;
    } rp{pos0, W, Hkv, HD};
    auto e3 = g.begin();
    e3->use(NemotronH::pipeline("nh_lili_ring_write"));
    e3->buf(0, bk);
    e3->buf(1, bv);
    e3->buf(2, rk2);
    e3->buf(3, rv2);
    e3->bytes(4, &rp, sizeof rp);
    e3->dispatch(Rk, 2 * Hkv, 1, HD);
    e3->commit_and_wait();
    const bool same = !memcmp(rk1.ptr(), rk2.ptr(), zero.size() * 2) && !memcmp(rv1.ptr(), rv2.ptr(), zero.size() * 2);
    report("nh_dsp_commit_kv", same ? 0.0 : 1.0, 0);
  }
}

// nh_xq8_v8 (Q8, repacked codes) at small odd sizes against a double product of the Q8 values (epilogues store, add)
void check_xq8() {
  Gpu& g = Gpu::get();
  for (const uint32_t R : {1u, 5u, 8u}) {
    const uint32_t N = 40, K = 384;
    std::vector<uint8_t> q(size_t(N) * K);
    std::vector<__fp16> sc(size_t(N) * K / 32);
    std::vector<float> w(size_t(N) * K);
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t b = 0; b < K / 32; ++b) {
        sc[n * (K / 32) + b] = __fp16(0.01f + 0.02f * std::fabs(nrand()));
        for (uint32_t i = 0; i < 32; ++i) {
          const int8_t v = int8_t(int(rnd() % 256) - 128);
          q[size_t(n) * K + b * 32 + i] = uint8_t(v);
          w[size_t(n) * K + b * 32 + i] = float(sc[n * (K / 32) + b]) * float(v);
        }
      }
    std::vector<uint8_t> rq = q;
    DSpark::repack_xq8(rq.data(), N, K);
    std::vector<__fp16> x(size_t(R) * K);
    for (auto& v : x) v = __fp16(nrand());
    std::vector<float> y0(size_t(R) * N);
    for (auto& v : y0) v = nrand();
    for (const uint32_t epi : {0u, 1u}) {
      Buffer bq = up(rq, "q"), bs = up(sc, "s"), bx = up(x, "x"), by = up(y0, "y");
      const struct {
        uint32_t N, K, x_ld, y_ld, R;
        float gs;
        uint32_t epi;
      } p{N, K, K, N, R, 1.0f, epi};
      auto e = g.begin();
      e->use(NemotronH::pipeline("nh_xq8_v8"));
      e->buf(0, bx);
      e->buf(1, bq);
      e->buf(2, bs);
      e->buf(3, by);
      e->bytes(4, &p, sizeof p);
      e->dispatch((N + 15) / 16, 1, 1, 256);
      e->commit_and_wait();
      double err = 0, mx = 0;
      for (uint32_t r = 0; r < R; ++r)
        for (uint32_t n = 0; n < N; ++n) {
          double ref = epi ? y0[r * N + n] : 0.0;
          for (uint32_t k = 0; k < K; ++k) ref += double(w[size_t(n) * K + k]) * double(float(x[size_t(r) * K + k]));
          err = std::max(err, std::fabs(ref - by.as<float>()[r * N + n]));
          mx = std::max(mx, std::fabs(ref));
        }
      char name[64];
      snprintf(name, sizeof name, "nh_xq8_v8 R%u epi%u", R, epi);
      report(name, err / mx, 2e-3);
    }
  }
}

}  // namespace

// a verify's two sampling passes on simdgroup reductions (topk_stage1_rows_sg + sample_stage2_rows_sg) against
// topk_stage1_rows + sample_stage2_rows: stage 1's candidates and the sampled tokens the same to the bit, logits with
// many ties (a few distinct values) or distinct, V 5000 (a partial last chunk) and 131072, K 1 / 20 / 32, T 0 / 0.6
void check_sample_sg() {
  Gpu& g = Gpu::get();
  struct TopkP {
    uint32_t V, K, chunk;
  };
  struct SampleP {
    uint32_t n, K;
    float temp, top_p;
    uint32_t seed, step, out_index, remap_from, remap_base, table_from;
  };
  const uint32_t R = 3;
  for (uint32_t V : {5000u, 131072u})
    for (int ties = 0; ties < 2; ++ties)
      for (uint32_t K : {1u, 20u, 32u}) {
        std::vector<float> lg(size_t(R) * V);
        for (auto& v : lg) v = ties ? float(rnd() % 6) * 0.5f : nrand() * 3.0f;
        const uint32_t ntg = (V + 2047) / 2048;
        Buffer bl = up(lg, "logits");
        Buffer cv[2] = {g.alloc(size_t(R) * ntg * K * 4, "cv"), g.alloc(size_t(R) * ntg * K * 4, "cv")};
        Buffer ci[2] = {g.alloc(size_t(R) * ntg * K * 4, "ci"), g.alloc(size_t(R) * ntg * K * 4, "ci")};
        Buffer tok[2] = {g.alloc(64 * 4, "tok"), g.alloc(64 * 4, "tok")};
        bool same = true;
        for (float temp : {0.0f, 0.6f})
          for (uint32_t step = 0; step < 40 && same; ++step) {
            for (int v = 0; v < 2; ++v) {
              const TopkP tp{V, K, 2048};
              const SampleP sp{ntg * K, K, temp, 0.95f, 1234u, step, 0, 0, 0, 0};
              auto e = g.begin();
              e->use(NemotronH::pipeline(v ? "topk_stage1_rows_sg" : "topk_stage1_rows"));
              e->buf(0, bl);
              e->buf(1, cv[v]);
              e->buf(2, ci[v]);
              e->bytes(3, &tp, sizeof tp);
              e->dispatch(ntg, R, 1, 256);
              e->use(NemotronH::pipeline(v ? "sample_stage2_rows_sg" : "sample_stage2_rows"));
              e->buf(0, cv[v]);
              e->buf(1, ci[v]);
              e->buf(2, tok[v]);
              e->bytes(3, &sp, sizeof sp);
              e->buf(4, ci[v]);
              e->dispatch(R, 1, 1, v ? 256 : 1024);
              e->commit_and_wait();
            }
            same = memcmp(cv[0].ptr(), cv[1].ptr(), cv[0].size) == 0 && memcmp(ci[0].ptr(), ci[1].ptr(), ci[0].size) == 0 &&
                   memcmp(tok[0].ptr(), tok[1].ptr(), R * 4) == 0;
          }
        char name[64];
        snprintf(name, sizeof name, "sample_sg V%u K%u%s", V, K, ties ? " ties" : "");
        report(name, same ? 0.0 : 1.0, 0.0);
        for (Buffer* b : {&bl, &cv[0], &cv[1], &ci[0], &ci[1], &tok[0], &tok[1]}) g.release(*b);
      }
}

int main() {
  Gpu& g = Gpu::get();
  g.compile_extra("Nemotron-H", kNhKernelSource);
  check_attn();
  check_mask_rows();
  check_walk(77, 48, 5);
  check_walk(2000, 512, 16);  // (Lightning's Markov rank and default candidates; NI_DSPARK_K's maximum)
  check_walk(2000, 512, 32);
  check_sums();
  check_xb_v8();
  check_moe_ah();
  check_f32_sk();
  check_cand();
  check_dsp_fused();
  check_xq8();
  check_sample_sg();
  printf("%s\n", g_fail ? "FAILED" : "all ok");
  return g_fail ? 1 : 0;
}
