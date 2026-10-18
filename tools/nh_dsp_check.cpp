// Nemotron 3.5 Lightning's new kernels against CPU references at small odd sizes: the DSpark drafter's causal block
// attention (nh_dsp_attn, a ring that wraps), its mask rows (nh_dsp_mask_rows) and Markov walk over candidates
// (nh_dsp_walk, NVFP4 W2), the MoE sums of layers without latent projections (nh_moe2_sum2_add,
// nh_moe_combine_add, the fused one-token layer's nh_moe_ah / nh_moe_dh), and the 8-row BF16 product (nh_xb_v8:
// Lightning's attention in verify) and the F32 router of verify rows (nh_mv_f32_sk). One dispatch per command buffer; run it under MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1.
//   nh_dsp_check
#include "metal/gpu.h"
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

void check_attn() {
  // P context positions in a ring of W slots (the ring holds [P - W, P)), B = 8 block rows, causal within the block
  const int HD = 128, Hq = 4, Hkv = 2, W = 24, B = 8, P = 37, G = Hq / Hkv;
  std::vector<float> q(size_t(B) * Hq * HD), kb(size_t(B) * Hkv * HD), vb(kb.size()), sinks(Hq);
  std::vector<__fp16> rk(size_t(Hkv) * W * HD), rv(rk.size());
  for (auto& x : q) x = nrand();
  for (auto& x : kb) x = nrand();
  for (auto& x : vb) x = nrand();
  for (auto& x : rk) x = __fp16(nrand());
  for (auto& x : rv) x = __fp16(nrand());
  for (auto& x : sinks) x = nrand();
  const float scale = 1.0f / std::sqrt(float(HD));
  struct {
    uint32_t P, lo, W, Hq, Hkv, B;
    float scale;
  } p{uint32_t(P), uint32_t(P - W), uint32_t(W), uint32_t(Hq), uint32_t(Hkv), uint32_t(B), scale};
  Gpu& g = Gpu::get();
  Buffer bq = up(q, "q"), bk = up(kb, "k"), bv = up(vb, "v"), brk = up(rk, "rk"), brv = up(rv, "rv"), bs = up(sinks, "s");
  Buffer out = g.alloc(q.size() * 4, "out");
  auto e = g.begin();
  e->use(NemotronH::pipeline("nh_dsp_attn"));
  e->buf(0, bq);
  e->buf(1, bk);
  e->buf(2, bv);
  e->buf(3, brk);
  e->buf(4, brv);
  e->buf(5, bs);
  e->buf(6, out);
  e->bytes(7, &p, sizeof p);
  e->dispatch(uint32_t(Hkv), uint32_t(B), 1, uint32_t(32 * G));
  e->commit_and_wait();
  double err = 0, mx = 0;
  for (int r = 0; r < B; ++r)
    for (int h = 0; h < Hq; ++h) {
      const int kvh = h / G, qpos = P + r;
      std::vector<double> s;
      std::vector<const float*> vs;
      std::vector<std::vector<float>> vtmp;
      auto qd = [&](const float* k) {
        double d = 0;
        for (int i = 0; i < HD; ++i) d += double(q[(size_t(r) * Hq + h) * HD + i]) * scale * k[i];
        return d;
      };
      std::vector<std::vector<float>> keys, vals;
      for (int t = std::max(P - W, qpos + 1 - W); t < P; ++t) {
        std::vector<float> k(HD), v(HD);
        for (int i = 0; i < HD; ++i) {
          k[size_t(i)] = float(rk[(size_t(kvh) * W + size_t(t % W)) * HD + size_t(i)]);
          v[size_t(i)] = float(rv[(size_t(kvh) * W + size_t(t % W)) * HD + size_t(i)]);
        }
        keys.push_back(k);
        vals.push_back(v);
      }
      for (int j = 0; j <= r; ++j) {
        keys.push_back(std::vector<float>(&kb[(size_t(j) * Hkv + kvh) * HD], &kb[(size_t(j) * Hkv + kvh) * HD] + HD));
        vals.push_back(std::vector<float>(&vb[(size_t(j) * Hkv + kvh) * HD], &vb[(size_t(j) * Hkv + kvh) * HD] + HD));
      }
      double m = sinks[size_t(h)];
      for (auto& k : keys) m = std::max(m, qd(k.data()));
      double den = std::exp(double(sinks[size_t(h)]) - m);
      std::vector<double> o(HD, 0.0);
      for (size_t j = 0; j < keys.size(); ++j) {
        const double w = std::exp(qd(keys[j].data()) - m);
        den += w;
        for (int i = 0; i < HD; ++i) o[size_t(i)] += w * vals[j][size_t(i)];
      }
      for (int i = 0; i < HD; ++i) {
        const double ref = o[size_t(i)] / den;
        err = std::max(err, std::fabs(ref - out.as<float>()[(size_t(r) * Hq + h) * HD + i]));
        mx = std::max(mx, std::fabs(ref));
      }
    }
  report("nh_dsp_attn", err / mx, 1e-4);
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

void check_walk() {
  // V tokens, rank R (NVFP4 W2 in the loader's layout), S slots of K candidates
  const int V = 77, R = 48, S = 7, K = 5, P = 3;
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
  report("nh_dsp_walk tokens", double(bad), 0);
  report("nh_dsp_walk log-probs", err, 1e-4);
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

}  // namespace

int main() {
  Gpu& g = Gpu::get();
  g.compile_extra("Nemotron-H", kNhKernelSource);
  check_attn();
  check_mask_rows();
  check_walk();
  check_sums();
  check_xb_v8();
  check_moe_ah();
  check_f32_sk();
  printf("%s\n", g_fail ? "FAILED" : "all ok");
  return g_fail ? 1 : 0;
}
