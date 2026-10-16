// Small-T dense GEMM A/B (T <= 64 token rows, 4-bit g64): register-streaming qmms (qmms.metal) vs
// the qmr tiles (qmm3.metal) the engine picks at that T, on the projection shapes of the 35B / 27B.
//   bench_qmms
// Each case is a group of projections sharing the half input X[T, K]: one matrix (qkv, out, ...),
// the GDN in-proj (qkv, z, b, a) or attention q/k/v (qmms runs a group in ONE dispatch; qmr runs one
// dispatch per projection), or gate/up + SwiGLU (GU: qmms_gu / qmr_gu, half output).
// Weights are rotated over copies totalling >= ROT_MB (default 96) MB so that the repeated
// dispatches of a timing loop stream from DRAM as in a real prefill pass (the SLC would otherwise
// hold the smaller 35B matrices). Kernels are timed interleaved, TRIALS rounds (default 5); prints
// the median per call, effective weight GB/s and TFLOP/s, and the max relative error of 48 sampled
// outputs per projection vs a double reference on the dequantized halfs (as qmm: half(s*q + b)).
// env: SHAPES=35|27 (default both); TS="8 16 23" token counts; CASES=substring filter on case names;
//      KERNELS="auto qmms ..." where auto = the engine's qmr tile (m8 for T <= 8, m16 below 64),
//      qmms / qmmsi = register-streaming (engine K split; qmms:NSG:KS overrides), or a qmr kernel
//      name (qmr_k64_w14_m32_n64, qmr_gu_k64_w14_m16_n64, ...: tile/threads parsed from the name).
#include "metal/gpu.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;

static uint16_t f2h(float f) {
  __fp16 h = (__fp16)f;
  uint16_t u;
  memcpy(&u, &h, 2);
  return u;
}
static float h2f(uint16_t u) {
  __fp16 h;
  memcpy(&h, &u, 2);
  return float(h);
}
static std::vector<std::string> words(const char* env, const char* def) {
  const char* v = getenv(env);
  std::stringstream ss(v && *v ? v : def);
  std::vector<std::string> r;
  std::string w;
  while (ss >> w) r.push_back(w);
  return r;
}

struct Case {
  std::string name;
  uint32_t K;
  std::vector<uint32_t> N;  // projections sharing X
  bool gu;                  // gate/up + SwiGLU (two matrices of N[0])
  int bits = 4;
};

struct Mat {  // one projection's weights, ncopy identical copies
  std::vector<Buffer> W, S, B;
  uint32_t N = 0;
};

int main() {
  Gpu& g = Gpu::get();
  std::mt19937 rng(7);
  std::vector<Case> cases;
  const char* sh = getenv("SHAPES");
  if (!sh || !strcmp(sh, "35"))
    cases.insert(cases.end(), {{"35B gdn in (qkv,z,b,a)", 2048, {8192, 4096, 32, 32}, false},
                               {"35B gdn out", 4096, {2048}, false},
                               {"35B attn q,k,v", 2048, {8192, 512, 512}, false},
                               {"35B attn o", 4096, {2048}, false},
                               {"35B shared gate/up", 2048, {512}, true},
                               {"35B shared down", 512, {2048}, false},
                               {"35B router+gate (8-bit)", 2048, {256, 1}, false, 8}});
  if (!sh || !strcmp(sh, "27"))
    cases.insert(cases.end(), {{"27B gdn in (qkv,z,b,a)", 5120, {10240, 6144, 48, 48}, false},
                               {"27B gdn out", 6144, {5120}, false},
                               {"27B attn q,k,v", 5120, {12288, 1024, 1024}, false},
                               {"27B attn o", 6144, {5120}, false},
                               {"27B mlp gate/up", 5120, {17408}, true},
                               {"27B mlp down", 17408, {5120}, false}});
  std::vector<int> Ts;
  for (auto& w : words("TS", "1 4 8 12 16 23 24 32 40 48 58 64")) Ts.push_back(std::stoi(w));
  const auto kspecs = words("KERNELS", "auto qmms");
  const int trials = getenv("TRIALS") ? atoi(getenv("TRIALS")) : 5;
  const double rot_mb = getenv("ROT_MB") ? atof(getenv("ROT_MB")) : 96;
  const char* filt = getenv("CASES");
  const std::regex tile(".*_w([0-9])([0-9])_m([0-9]+)_n([0-9]+)$");

  for (const Case& c : cases) {
    if (filt && *filt && c.name.find(filt) == std::string::npos) continue;
    const uint32_t K = c.K, Tmax = 128;
    const int nmat = c.gu ? 2 : int(c.N.size());
    double bytes = 0;
    const int bits = c.bits;
    for (int m = 0; m < nmat; ++m) bytes += double(c.N[c.gu ? 0 : m]) * K * bits / 8 + double(c.N[c.gu ? 0 : m]) * K / 64 * 4;
    const int ncopy = std::clamp(int(std::ceil(rot_mb * 1e6 / bytes)), 1, 32);
    std::vector<Mat> mats(nmat);
    std::uniform_real_distribution<float> d(-1, 1);
    for (int m = 0; m < nmat; ++m) {
      Mat& M = mats[m];
      M.N = c.N[c.gu ? 0 : m];
      const size_t ww = size_t(M.N) * K * bits / 32, gr = size_t(M.N) * K / 64;
      for (int k = 0; k < ncopy; ++k) {
        M.W.push_back(g.alloc(ww * 4));
        M.S.push_back(g.alloc(gr * 2));
        M.B.push_back(g.alloc(gr * 2));
        if (k == 0) {
          for (size_t i = 0; i < ww; ++i) M.W[0].as<uint32_t>()[i] = rng();
          for (size_t i = 0; i < gr; ++i) {
            M.S[0].as<uint16_t>()[i] = f32_to_bf16((bits == 4 ? 0.01f : 0.0008f) * (1.0f + d(rng)));
            M.B[0].as<uint16_t>()[i] = f32_to_bf16(0.05f * d(rng));
          }
        } else {
          memcpy(M.W[k].ptr(), M.W[0].ptr(), ww * 4);
          memcpy(M.S[k].ptr(), M.S[0].ptr(), gr * 2);
          memcpy(M.B[k].ptr(), M.B[0].ptr(), gr * 2);
        }
      }
    }
    Buffer X = g.alloc(size_t(Tmax) * K * 2);
    for (size_t i = 0; i < size_t(Tmax) * K; ++i) X.as<uint16_t>()[i] = f2h(d(rng));
    std::vector<Buffer> Y(nmat);
    for (int m = 0; m < nmat; ++m) Y[m] = g.alloc(size_t(Tmax) * mats[m].N * 4);  // GU: Y[0] holds half H
    auto wval = [&](const Mat& M, uint32_t n, uint32_t k) {
      uint32_t q = bits == 4 ? (M.W[0].as<uint32_t>()[(size_t(n) * K + k) / 8] >> (4 * (k % 8))) & 15
                             : (M.W[0].as<uint32_t>()[(size_t(n) * K + k) / 4] >> (8 * (k % 4))) & 255;
      float wv = bf16_to_f32(M.S[0].as<uint16_t>()[(size_t(n) * K + k) / 64]) * q +
                 bf16_to_f32(M.B[0].as<uint16_t>()[(size_t(n) * K + k) / 64]);
      return double(float(__fp16(wv)));
    };
    double flop1 = 0;  // per token row
    for (int m = 0; m < nmat; ++m) flop1 += 2.0 * mats[m].N * K;
    printf("---- %s: K=%u N=", c.name.c_str(), K);
    for (int m = 0; m < nmat; ++m) printf("%u%s", mats[m].N, m + 1 < nmat ? "," : "");
    printf(" (%.1f MB, %d copies) ----\n", bytes / 1e6, ncopy);

    for (int T : Ts) {
      struct K1 {
        std::string label;
        std::function<void(Encoder&, int)> enc;  // copy index
        bool gu_half;
      };
      std::vector<K1> ks;
      for (const std::string& spec : kspecs) {
        if (spec.rfind("qmms", 0) == 0) {
          const bool il = spec.rfind("qmmsi", 0) == 0;
          const int TB = (T + 7) / 8;
          if (TB > 8 || (il && (TB < 2 || TB > 4))) continue;
          uint32_t nsg = 8, ksplit = K >= 4096 ? 8 : 4;
          if (spec.find(':') != std::string::npos) sscanf(spec.c_str() + spec.find(':'), ":%u:%u", &nsg, &ksplit);
          const uint32_t gpr = K / 64;
          while (ksplit > 1 && (gpr / ksplit < 2 || nsg % ksplit)) ksplit /= 2;
          if (il && bits != 4) continue;
          std::string kn = std::string(il ? "qmmsi" : "qmms") + (c.gu ? "_gu_b" : "_b") + std::to_string(bits) + "_t" +
                           std::to_string(TB);
          Pipeline pl = g.pipeline(kn);
          kp::Qmms p{};
          p.K = K; p.T = uint32_t(T); p.x_ld = K; p.ks = ksplit;
          const uint32_t rbs = nsg / ksplit;
          uint32_t tg = 0;
          const int nseg = c.gu ? 2 : nmat;
          p.nseg = uint32_t(nseg);
          for (int s = 0; s < nseg; ++s) {
            p.N[s] = mats[c.gu ? 0 : s].N;
            p.y_ld[s] = p.N[s];
            p.tg_begin[s] = tg;
            if (!c.gu || s == 0) tg += ((p.N[s] + 7) / 8 + rbs - 1) / rbs;
            else p.tg_begin[s] = p.tg_begin[0] + tg;  // unused by qmms_gu
          }
          if (c.gu) p.tg_begin[1] = p.tg_begin[2] = tg;
          p.tg_begin[nseg] = tg;
          ks.push_back({kn + " nsg" + std::to_string(nsg) + " ks" + std::to_string(ksplit),
                        [&, pl, p, tg, nsg, nseg](Encoder& e, int cp) {
                          e.use(pl);
                          for (int s = 0; s < 4; ++s) {
                            const int m = std::min(s, nmat - 1);
                            e.buf(s * 4 + 0, mats[m].W[cp]);
                            e.buf(s * 4 + 1, mats[m].S[cp]);
                            e.buf(s * 4 + 2, mats[m].B[cp]);
                            e.buf(s * 4 + 3, Y[c.gu ? 0 : m]);
                          }
                          e.buf(16, X);
                          e.val(17, p);
                          e.dispatch(tg, 1, 1, 32 * nsg);
                          (void)nseg;
                        },
                        c.gu});
          continue;
        }
        if (spec.rfind("qmrk_", 0) == 0) {  // split-K qmr ("qmrk_..._nBN:KS") + qmr_sk_reduce, one projection
          if (c.gu || bits != 4 || nmat != 1) continue;
          const std::string kn = spec.substr(0, spec.find(':'));
          const uint32_t ksp = spec.find(':') != std::string::npos ? std::stoi(spec.substr(spec.find(':') + 1)) : 4;
          if ((K / 64) % ksp) continue;
          std::smatch mt;
          if (!std::regex_match(kn, mt, tile)) fatal("bad kernel %s", kn.c_str());
          const uint32_t th = 32 * std::stoi(mt[1]) * std::stoi(mt[2]), bm = std::stoi(mt[3]), bn = std::stoi(mt[4]);
          Pipeline pl = g.pipeline(kn), red = g.pipeline("qmr_sk_reduce");
          static Buffer part;
          if (!part) part = g.alloc(size_t(16) * 64 * 32768 * 4);
          kp::Qmm p{uint32_t(T), mats[0].N, K, K, mats[0].N, 0, 64, ksp, 0};
          ks.push_back({spec, [&, pl, red, p, th, bm, bn, ksp, T](Encoder& e, int cp) {
                          e.use(pl);
                          e.buf(0, X); e.buf(1, mats[0].W[cp]); e.buf(2, mats[0].S[cp]); e.buf(3, mats[0].B[cp]);
                          e.buf(4, part);
                          e.val(5, p);
                          e.dispatch((mats[0].N + bn - 1) / bn, (T + bm - 1) / bm, ksp, th);
                          e.use(red);
                          e.buf(0, part); e.buf(1, Y[0]);
                          e.val(2, p);
                          const uint32_t n4 = uint32_t(T) * mats[0].N / 4;
                          e.dispatch((n4 + 255) / 256, 1, 1, 256);
                        },
                        false});
          continue;
        }
        if (spec.rfind("qmrs_", 0) == 0) {  // multi-projection qmr: the whole group in one dispatch
          if (c.gu || bits != 4) continue;
          std::smatch mt;
          if (!std::regex_match(spec, mt, tile)) fatal("bad kernel %s", spec.c_str());
          const uint32_t th = 32 * std::stoi(mt[1]) * std::stoi(mt[2]), bm = std::stoi(mt[3]), bn = std::stoi(mt[4]);
          Pipeline pl = g.pipeline(spec);
          kp::Qmms p{};
          p.K = K; p.T = uint32_t(T); p.x_ld = K; p.nseg = uint32_t(nmat);
          uint32_t tg = 0;
          for (int m = 0; m < nmat; ++m) {
            p.N[m] = mats[m].N;
            p.y_ld[m] = mats[m].N;
            p.tg_begin[m] = tg;
            tg += (mats[m].N + bn - 1) / bn;
          }
          p.tg_begin[nmat] = tg;
          ks.push_back({spec, [&, pl, p, tg, th, bm, T](Encoder& e, int cp) {
                          e.use(pl);
                          for (int s4 = 0; s4 < 4; ++s4) {
                            const int m = std::min(s4, nmat - 1);
                            e.buf(s4 * 4 + 0, mats[m].W[cp]);
                            e.buf(s4 * 4 + 1, mats[m].S[cp]);
                            e.buf(s4 * 4 + 2, mats[m].B[cp]);
                            e.buf(s4 * 4 + 3, Y[m]);
                          }
                          e.buf(16, X);
                          e.val(17, p);
                          e.dispatch(tg, (T + bm - 1) / bm, 1, th);
                        },
                        false});
          continue;
        }
        // qmr: "auto" = the engine's choice at this T, else a kernel name
        std::string kn = spec;
        if (spec == "auto" && bits == 8) kn = "qmm_b8_m32_n32_w22";  // the old kernel (128 threads, 32 x 32 tiles)
        else if (spec == "auto") {
          if (c.gu) kn = T <= 8 ? "qmr_gu_k64_w12_m8_n32" : T < 64 ? "qmr_gu_k64_w14_m16_n64" : "qmr_gu_k32_w14_m32_n64";
          else kn = T <= 8 ? "qmr_k64_w12_m8_n64" : T < 64 ? "qmr_k64_w14_m16_n64" : "qmr_k32_w14_m32_n128";
        }
        const bool kgu = kn.find("_gu_") != std::string::npos;
        if (kgu != c.gu || (bits == 8) != (kn.rfind("qmm_b8", 0) == 0)) continue;
        uint32_t th = 128, bm = 32, bn = 32;
        if (kn == "qmm_b8_m32_n32_w22") {
          kn = "qmm_b8_m32_n32";
        } else {
          std::smatch mt;
          if (!std::regex_match(kn, mt, tile)) fatal("bad kernel %s", kn.c_str());
          th = 32 * std::stoi(mt[1]) * std::stoi(mt[2]), bm = std::stoi(mt[3]), bn = std::stoi(mt[4]);
        }
        Pipeline pl = g.pipeline(kn);
        ks.push_back({(spec == "auto" ? "auto " : "") + kn,
                      [&, pl, th, bm, bn, T](Encoder& e, int cp) {
                        e.use(pl);
                        if (c.gu) {
                          kp::Qmm p{uint32_t(T), mats[0].N, K, K, mats[0].N, 0, 64, 0, 0};
                          e.buf(0, X); e.buf(1, mats[0].W[cp]); e.buf(2, mats[0].S[cp]); e.buf(3, mats[0].B[cp]);
                          e.buf(4, Y[0]);
                          e.val(5, p);
                          e.buf(6, mats[1].W[cp]); e.buf(7, mats[1].S[cp]); e.buf(8, mats[1].B[cp]);
                          e.dispatch((mats[0].N + bn - 1) / bn, (T + bm - 1) / bm, 1, th);
                          return;
                        }
                        for (int m = 0; m < nmat; ++m) {
                          // the engine's rule: N <= 64 always takes the m16 tile (unless M <= 8)
                          kp::Qmm p{uint32_t(T), mats[m].N, K, K, mats[m].N, 0, 64, 0, 0};
                          e.buf(0, X); e.buf(1, mats[m].W[cp]); e.buf(2, mats[m].S[cp]); e.buf(3, mats[m].B[cp]);
                          e.buf(4, Y[m]);
                          e.val(5, p);
                          e.dispatch((mats[m].N + bn - 1) / bn, (T + bm - 1) / bm, 1, th);
                        }
                      },
                      false});
      }
      // time interleaved
      const int reps = std::max(4, std::min(64, int(2e-3 / std::max(bytes / 220e9, flop1 * T / 6e12))));
      std::vector<std::vector<double>> ms(ks.size());
      auto run = [&](size_t ki, int n) {
        auto e = g.begin();
        for (int r = 0; r < n; ++r) ks[ki].enc(*e, r % ncopy);
        e->commit_and_wait();
        return e->gpu_ms() / n;
      };
      for (size_t ki = 0; ki < ks.size(); ++ki) run(ki, 2);
      for (int t = 0; t < trials; ++t)
        for (size_t ki = 0; ki < ks.size(); ++ki) ms[ki].push_back(run(ki, reps));
      for (size_t ki = 0; ki < ks.size(); ++ki) {
        for (int m = 0; m < nmat; ++m) memset(Y[m].ptr(), 0, size_t(Tmax) * mats[m].N * 4);
        run(ki, 1);
        double maxrel = 0;
        std::mt19937 r2(11);
        for (int m = 0; m < (c.gu ? 1 : nmat); ++m)
          for (int sidx = 0; sidx < 48; ++sidx) {
            const uint32_t t = sidx == 0 ? T - 1 : r2() % T, n = sidx == 0 ? mats[m].N - 1 : r2() % mats[m].N;
            double ref = 0, got;
            for (uint32_t k = 0; k < K; ++k) ref += wval(mats[m], n, k) * h2f(X.as<uint16_t>()[size_t(t) * K + k]);
            if (c.gu) {
              double u = 0;
              for (uint32_t k = 0; k < K; ++k) u += wval(mats[1], n, k) * h2f(X.as<uint16_t>()[size_t(t) * K + k]);
              ref = ref / (1.0 + std::exp(-ref)) * u;
              got = h2f(Y[0].as<uint16_t>()[size_t(t) * mats[0].N + n]);
            } else {
              got = Y[m].as<float>()[size_t(t) * mats[m].N + n];
            }
            maxrel = std::max(maxrel, std::fabs(got - ref) / (std::fabs(ref) + 1.0));
          }
        std::vector<double> v = ms[ki];
        std::sort(v.begin(), v.end());
        const double med = v[v.size() / 2];
        printf("T=%2d %-40s %8.1f us  %6.1f GB/s  %5.2f TFLOP/s  err=%.1e%s\n", T, ks[ki].label.c_str(), med * 1000,
               bytes / (med * 1e-3) / 1e9, flop1 * T / (med * 1e-3) / 1e12, maxrel,
               maxrel > (c.gu ? 4e-3 : 1e-3) ? "  FAIL" : "");
      }
      fflush(stdout);
    }
    for (auto& M : mats)
      for (int k = 0; k < ncopy; ++k) {
        g.release(M.W[k]);
        g.release(M.S[k]);
        g.release(M.B[k]);
      }
    g.release(X);
    for (auto& y : Y) g.release(y);
  }
}
