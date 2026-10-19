// S1: exactness of the on-GPU acceptance on synthetic inputs,
// no model needed.
//   dflash_accept_test [trials=40000] [K=3]
// 1. dfl_dist_rows against a CPU transcription of the plain sampler's distribution (sample_stage2_rows:
//    top-k ordered (value desc, id asc), top-p on the untempered probabilities, tempered on the kept
//    prefix) on random candidate lists, including ties.
// 2. The speculative loop on prefix-conditional synthetic p(.|prefix) and q(.|prefix): chains drawn from
//    q on the host, dfl_accept on the GPU, a second cycle from the emitted token when only one token came
//    out; the joint law of the first two emitted tokens is tested against p(t1) p(t2|t1) (chi-square,
//    cells with expected count < 5 pooled), in several regimes: random p and q, q = p, q disjoint from p
//    (always rejected: residual = p), point masses, p(d) = 0 for some proposals, K = 1..7, and with the
//    dynamic verify depth (dfl_depth before dfl_accept): its exact T > 0 rules (0, and 2, the earlier one) must
//    pass; the rule that also uses the proposal's own confidence (rule 1, not used by the engine at T > 0) must fail in the depth-sensitive
//    regime (q = p at the first proposal, q = 0.9 / 0.1 vs p = 1/2 at the second: that rule emits 0.55 / 0.45).
#include "metal/gpu.h"
#include "model/kparams.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <vector>

using namespace ni;

namespace {

// ---- chi-square p-value (regularized upper incomplete gamma) ----
double gamma_q(double a, double x) {
  if (x <= 0) return 1.0;
  const double gln = std::lgamma(a);
  if (x < a + 1) {  // series for P
    double ap = a, sum = 1.0 / a, del = sum;
    for (int n = 0; n < 1000; ++n) {
      ap += 1;
      del *= x / ap;
      sum += del;
      if (std::fabs(del) < std::fabs(sum) * 1e-14) break;
    }
    return 1.0 - sum * std::exp(-x + a * std::log(x) - gln);
  }
  double b = x + 1 - a, c = 1e300, d = 1 / b, h = d;  // continued fraction for Q
  for (int i = 1; i < 1000; ++i) {
    const double an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (std::fabs(d) < 1e-300) d = 1e-300;
    c = b + an / c;
    if (std::fabs(c) < 1e-300) c = 1e-300;
    d = 1 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1) < 1e-14) break;
  }
  return std::exp(-x + a * std::log(x) - gln) * h;
}

// chi-square of observed counts vs expected probabilities (pooling cells with expected < 5)
void chi2(const std::map<long, long>& obs, const std::map<long, double>& expp, long n, double* stat, int* dof, double* pval,
          long* outside) {
  double pool_o = 0, pool_e = 0, s = 0;
  int cells = 0;
  *outside = 0;
  for (auto& [k, c] : obs)
    if (!expp.count(k) || expp.at(k) <= 0) *outside += c;
  for (auto& [k, pr] : expp) {
    const double e = pr * n;
    const double o = obs.count(k) ? double(obs.at(k)) : 0.0;
    if (e < 5) {
      pool_o += o;
      pool_e += e;
      continue;
    }
    s += (o - e) * (o - e) / e;
    ++cells;
  }
  if (pool_e > 0) {
    if (pool_e >= 5 || cells == 0) {
      s += (pool_o - pool_e) * (pool_o - pool_e) / pool_e;
      ++cells;
    } else {
      s += (pool_o - pool_e) * (pool_o - pool_e) / std::max(pool_e, 1e-9);  // tiny tail: still counted
      ++cells;
    }
  }
  *stat = s;
  *dof = std::max(1, cells - 1);
  *pval = gamma_q(*dof / 2.0, s / 2.0);
}

uint64_t mix(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

// ---- synthetic conditional distributions ----
constexpr int kV = 24;       // token universe: ids kBase + 0 .. kBase + kV - 1
constexpr int kBase = 1000;
struct Dist {
  std::vector<int> id;
  std::vector<float> p;
};
enum Regime { kRandom, kEqual, kDisjoint, kPoint, kMixed, kBias };
const char* regime_name(int r) {
  static const char* n[] = {"random p, random q", "q = p", "q disjoint from p", "point masses", "mixed (p(d)=0, flat q)",
                            "depth-sensitive"};
  return n[r];
}

uint64_t key_of(const std::vector<int>& prefix) {
  uint64_t h = 0x9e3779b97f4a7c15ULL;
  for (int t : prefix) h = mix(h ^ uint64_t(t + 1));
  return h;
}

// p(.|prefix): support of 1..10 tokens of the universe, probabilities sorted descending (like a top-k list)
Dist p_of(const std::vector<int>& prefix, int regime) {
  std::mt19937_64 r(key_of(prefix) ^ 0x1111);
  Dist d;
  if (regime == kBias) {
    // every position: two tokens at 1/2 (which two depends on the prefix); q (below) equals p at the
    // first proposal, so d_0 is always accepted and position 1 is always decided inside the cycle
    const int a = int(r() % kV), b = (a + 1 + int(r() % (kV - 1))) % kV;
    d.id = {kBase + a, kBase + b};
    d.p = {0.5f, 0.5f};
    return d;
  }
  int ns = regime == kPoint ? 1 + int(r() % 2) : 1 + int(r() % 10);
  std::vector<int> u(kV);
  for (int i = 0; i < kV; ++i) u[i] = i;
  std::shuffle(u.begin(), u.end(), r);
  std::vector<float> w(ns);
  float tot = 0;
  for (int i = 0; i < ns; ++i) {
    w[i] = std::pow(float((r() % 1000) + 1) / 1000.0f, 2.0f);
    tot += w[i];
  }
  std::vector<int> ord(ns);
  for (int i = 0; i < ns; ++i) ord[i] = i;
  std::sort(ord.begin(), ord.end(), [&](int a, int b) { return w[a] > w[b]; });
  for (int i : ord) {
    d.id.push_back(kBase + u[i]);
    d.p.push_back(w[i] / tot);
  }
  return d;
}

// q(.|prefix): 16 distinct candidates with probabilities (the walk's softmax over its top-16)
Dist q_of(const std::vector<int>& prefix, int regime, const Dist& p) {
  std::mt19937_64 r(key_of(prefix) ^ 0x2222);
  Dist q;
  if (regime == kBias) {
    // q = p for the first proposal (prefix of length <= 1 here: the anchor's cycle starts at prefix ()),
    // else 0.9 / 0.1 on p's two tokens: a confidence rule that uses c_p verifies only the 0.9 draw
    q.id = p.id;
    if (prefix.size() % 2 == 0) q.p = {0.5f, 0.5f};
    else q.p = {0.9f, 0.1f};
    for (int i = 0; i < kV && q.id.size() < 16; ++i)
      if (std::find(q.id.begin(), q.id.end(), kBase + i) == q.id.end()) {
        q.id.push_back(kBase + i);
        q.p.push_back(0.0f);
      }
    return q;
  }
  std::vector<int> u(kV);
  for (int i = 0; i < kV; ++i) u[i] = i;
  std::shuffle(u.begin(), u.end(), r);
  if (regime == kEqual || (regime == kMixed && r() % 3 == 0)) {
    // q = p on p's support, the other candidates at 0
    for (size_t i = 0; i < p.id.size(); ++i) {
      q.id.push_back(p.id[i]);
      q.p.push_back(p.p[i]);
    }
    for (int i = 0; i < kV && q.id.size() < 16; ++i)
      if (std::find(q.id.begin(), q.id.end(), kBase + u[i]) == q.id.end()) {
        q.id.push_back(kBase + u[i]);
        q.p.push_back(0.0f);
      }
    return q;
  }
  if (regime == kDisjoint) {
    for (int i = 0; i < kV && q.id.size() < 16; ++i)
      if (std::find(p.id.begin(), p.id.end(), kBase + u[i]) == p.id.end()) q.id.push_back(kBase + u[i]);
    while (q.id.size() < 16) q.id.push_back(kBase + kV + int(q.id.size()));  // outside the universe
  } else {
    for (int i = 0; i < 16; ++i) q.id.push_back(kBase + u[i]);
  }
  std::vector<float> w(16);
  float tot = 0;
  const bool point = regime == kPoint && r() % 2 == 0;
  const bool flat = regime == kMixed && r() % 2 == 0;
  for (int i = 0; i < 16; ++i) {
    w[i] = point ? (i == 0 ? 1.0f : 0.0f) : flat ? 1.0f : std::pow(float((r() % 1000) + 1) / 1000.0f, 3.0f);
    tot += w[i];
  }
  for (int i = 0; i < 16; ++i) q.p.push_back(w[i] / tot);
  return q;
}

int draw(const Dist& q, std::mt19937_64& r, int* k_out) {
  std::uniform_real_distribution<double> U(0, 1);
  double u = U(r), c = 0;
  int k = -1;
  for (size_t i = 0; i < q.p.size(); ++i) {
    if (q.p[i] <= 0) continue;
    c += q.p[i];
    k = int(i);
    if (u < c) break;
  }
  *k_out = k;
  return q.id[k];
}

// per-trial GPU inputs (one dispatch each)
struct Slot {
  std::vector<int> tokens;  // [16]
  std::vector<int> cand;    // [7 * 16]
  std::vector<float> qp;    // [7 * 16]
  std::vector<int> wk;      // [8]
  std::vector<int> did;     // [8 * 32]
  std::vector<float> dp;    // [8 * 32]
};

// Inputs of one cycle with the anchor at position P (tokens[P] = anchor, proposals at P+1..): the
// prefix (the tokens emitted so far) conditions p and q.
void fill(Slot& s, const std::vector<int>& prefix, int P, int K, int regime, std::mt19937_64& r, std::vector<int>* chain) {
  s.tokens.assign(16, 0);
  if (P > 0) s.tokens[P] = prefix.back();
  s.cand.assign(7 * 16, -3);
  s.qp.assign(7 * 16, 0.0f);
  s.wk.assign(8, 0);
  s.did.assign(8 * 32, -1);
  s.dp.assign(8 * 32, 0.0f);
  std::vector<int> pre = prefix;
  chain->clear();
  for (int i = 0; i <= K; ++i) {
    const Dist p = p_of(pre, regime);
    for (size_t j = 0; j < p.id.size(); ++j) {
      s.did[i * 32 + j] = p.id[j];
      s.dp[i * 32 + j] = p.p[j];
    }
    if (i == K) break;
    const Dist q = q_of(pre, regime, p);
    int k;
    const int d = draw(q, r, &k);
    for (int j = 0; j < 16; ++j) {
      s.cand[i * 16 + j] = q.id[j];
      s.qp[i * 16 + j] = q.p[j];
    }
    s.wk[i] = k;
    s.tokens[P + 1 + i] = d;
    chain->push_back(d);
    pre.push_back(d);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const long trials = argc > 1 ? atol(argv[1]) : 40000;
  const int Kmain = argc > 2 ? atoi(argv[2]) : 3;
  const uint64_t salt = argc > 3 ? uint64_t(atoll(argv[3])) : 0;  // seed stream offset (repeat with others)
  const int only = argc > 4 ? atoi(argv[4]) : -1;                   // run only case #only
  Gpu& g = Gpu::get();
  int fails = 0;

  // ---- 1. dfl_dist_rows vs the CPU transcription of sample_stage2_rows ----
  {
    const uint32_t R = 8, n = 122 * 20, Kk = 20;
    Buffer cv = g.alloc(size_t(R) * n * 4), ci = g.alloc(size_t(R) * n * 4);
    Buffer did = g.alloc(R * 32 * 4), dp = g.alloc(R * 32 * 4), dk = g.alloc(R * 4);
    std::mt19937_64 r(7);
    for (uint32_t row = 0; row < R; ++row)
      for (uint32_t i = 0; i < n; ++i) {
        // a few exact ties and a peaked head
        float v = float(int(r() % 4000)) / 400.0f;
        if (i % 97 == 0) v = 12.0f;
        cv.as<float>()[row * n + i] = v;
        ci.as<uint32_t>()[row * n + i] = uint32_t((i * 7919 + row * 13) % 248320);
      }
    const float temps[] = {0.6f, 1.0f, 0.3f};
    const float tops[] = {0.95f, 0.8f, 1.0f};
    double max_rel = 0;
    int id_bad = 0;
    for (int t = 0; t < 3; ++t) {
      kp::DflDist p{n, Kk, temps[t], tops[t]};
      auto e = g.begin();
      e->use(g.pipeline("dfl_dist_rows"));
      e->buf(0, cv); e->buf(1, ci); e->buf(2, did); e->buf(3, dp); e->buf(4, dk);
      e->val(5, p);
      e->dispatch(R, 1, 1, 1024);
      e->commit_and_wait();
      for (uint32_t row = 0; row < R; ++row) {
        std::vector<std::pair<float, uint32_t>> c(n);
        for (uint32_t i = 0; i < n; ++i) c[i] = {cv.as<float>()[row * n + i], ci.as<uint32_t>()[row * n + i]};
        std::sort(c.begin(), c.end(), [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        float probs[32], tot = 0;
        for (uint32_t q = 0; q < Kk; ++q) {
          probs[q] = std::exp(c[q].first - c[0].first);
          tot += probs[q];
        }
        uint32_t keep = Kk;
        float cum = 0;
        for (uint32_t q = 0; q < Kk; ++q) {
          cum += probs[q] / tot;
          if (cum >= tops[t]) {
            keep = q + 1;
            break;
          }
        }
        float kt = 0;
        for (uint32_t q = 0; q < keep; ++q) {
          probs[q] = std::exp((c[q].first - c[0].first) / temps[t]);
          kt += probs[q];
        }
        if (dk.as<uint32_t>()[row] != keep) ++id_bad;
        for (uint32_t q = 0; q < keep; ++q) {
          if (uint32_t(did.as<int>()[row * 32 + q]) != c[q].second) ++id_bad;
          const double ref = probs[q] / kt, got = dp.as<float>()[row * 32 + q];
          max_rel = std::max(max_rel, std::fabs(got - ref) / ref);
        }
      }
    }
    const bool ok = id_bad == 0 && max_rel < 1e-5;
    printf("[dist] dfl_dist_rows vs CPU sampler distribution, 3 settings x 8 rows: %d id/keep mismatches, max rel. "
           "prob error %.2e: %s\n", id_bad, max_rel, ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
  }

  // ---- 2. the speculative loop: joint law of the first two emitted tokens ----
  struct Case {
    int regime, K;
    float temp;
    float tau = 0.0f;  // > 0: dynamic verify depth (dfl_depth) with this threshold
    int rule = 0;      // dfl_depth rule (1 = own confidence at T > 0: biased, expected to fail; 2 = the earlier rule)
  };
  std::vector<Case> cases = {{kRandom, Kmain, 0.6f}, {kEqual, Kmain, 0.6f}, {kDisjoint, Kmain, 0.6f},
                             {kPoint, Kmain, 0.6f},  {kMixed, Kmain, 0.6f}, {kRandom, 1, 0.6f},
                             {kRandom, 7, 1.0f},     {kMixed, 7, 1.0f},
                             {kRandom, 3, 0.6f, 0.1f}, {kRandom, 7, 1.0f, 0.05f}, {kMixed, 7, 0.6f, 0.2f},
                             {kEqual, 7, 0.6f, 0.3f},  {kBias, 3, 0.6f, 0.3f},       {kBias, 3, 0.6f, 0.3f, 1},
                             {kBias, 3, 0.6f, 0.3f, 2}, {kMixed, 7, 0.6f, 0.2f, 2}};
  Pipeline acc = g.pipeline("dfl_accept"), dep = g.pipeline("dfl_depth");
  const int B = 4096;  // trials per command buffer
  Buffer tok = g.alloc(size_t(B) * 16 * 4), cand = g.alloc(size_t(B) * 112 * 4), qp = g.alloc(size_t(B) * 112 * 4);
  Buffer wk = g.alloc(size_t(B) * 8 * 4), did = g.alloc(size_t(B) * 256 * 4), dp = g.alloc(size_t(B) * 256 * 4);
  Buffer res = g.alloc(size_t(B) * 4 * 4), samp = g.alloc(64), scratch = g.alloc(64), kd = g.alloc(size_t(B) * 16);
  Buffer ws = g.alloc(size_t(B) * 112 * 4);  // the walk's scores: log q, so softmax(S_j) = q_j (dfl_depth)
  for (size_t ci = 0; ci < cases.size(); ++ci) {
    if (only >= 0 && int(ci) != only) continue;
    const Case& cs = cases[ci];
    std::map<long, long> obs, obs1;
    long done = 0;
    double mean_a = 0;
    std::mt19937_64 hr(1234 + cs.regime * 17 + cs.K + salt * 1000003);
    while (done < trials) {
      const int nb = int(std::min<long>(B, trials - done));
      // first cycle of each trial: anchor at position 0, prefix = (); a trial that emits a single token
      // continues with a second cycle anchored at position 1 on that token (its uniforms then come from
      // positions 2.., as in the engine)
      std::vector<std::vector<int>> chains(nb);
      auto upload = [&](int i, const Slot& s) {
        memcpy(tok.as<int>() + i * 16, s.tokens.data(), 64);
        memcpy(cand.as<int>() + i * 112, s.cand.data(), 112 * 4);
        memcpy(qp.as<float>() + i * 112, s.qp.data(), 112 * 4);
        for (int j = 0; j < 112; ++j) ws.as<float>()[i * 112 + j] = s.qp[j] > 0 ? std::log(s.qp[j]) : -1e30f;
        memcpy(wk.as<int>() + i * 8, s.wk.data(), 32);
        memcpy(did.as<int>() + i * 256, s.did.data(), 256 * 4);
        memcpy(dp.as<float>() + i * 256, s.dp.data(), 256 * 4);
      };
      auto run = [&](int count, uint32_t P, const std::vector<uint32_t>& seed) {
        auto e = g.begin();
        for (int i = 0; i < count; ++i) {
          if (cs.tau > 0) {  // the drafter's depth decision from this trial's q and walk choices
            kp::DflDepth dpp{uint32_t(cs.K), cs.tau, cs.temp, uint32_t(cs.rule)};
            e->use(dep);
            e->buf(0, ws, size_t(i) * 448); e->buf(1, wk, size_t(i) * 32); e->buf(2, qp, size_t(i) * 448);
            e->buf(3, kd, size_t(i) * 16);
            e->val(4, dpp);
            e->dispatch(1, 1, 1, 32);
          }
          kp::DflAccept ap{P, uint32_t(cs.K), cs.temp, seed[i], cs.tau > 0 ? 1u : 0u};
          e->use(acc);
          e->buf(0, tok, size_t(i) * 64);
          e->buf(1, samp); e->buf(2, did, size_t(i) * 1024); e->buf(3, dp, size_t(i) * 1024);
          e->buf(4, cand, size_t(i) * 448); e->buf(5, qp, size_t(i) * 448); e->buf(6, wk, size_t(i) * 32);
          e->buf(7, res, size_t(i) * 16);
          e->val(8, ap);
          e->buf(9, kd, size_t(i) * 16);  // dynamic depth (read only with ap.dyn)
          e->buf(10, scratch);  // drafter state for a GPU-driven next pass (unused here)
          e->dispatch(1, 1, 1, 32);
        }
        e->commit_and_wait();
      };
      std::vector<uint32_t> seeds(nb);
      for (int i = 0; i < nb; ++i) {
        Slot s;
        fill(s, {}, 0, cs.K, cs.regime, hr, &chains[i]);
        upload(i, s);
        seeds[i] = uint32_t(mix(uint64_t(done + i) + (uint64_t(ci) << 40) + (salt << 48)));
      }
      run(nb, 0, seeds);
      std::vector<int> t1(nb), t2(nb, -1), second;
      for (int i = 0; i < nb; ++i) {
        const int a = res.as<int>()[i * 4];
        const int y = res.as<int>()[i * 4 + 1];
        mean_a += a;
        std::vector<int> em(chains[i].begin(), chains[i].begin() + a);
        em.push_back(y);
        t1[i] = em[0];
        if (em.size() >= 2) t2[i] = em[1];
        else second.push_back(i);
      }
      if (!second.empty()) {
        std::vector<uint32_t> s1;
        std::vector<std::vector<int>> ch2(second.size());
        for (size_t j = 0; j < second.size(); ++j) {
          Slot s;
          fill(s, {t1[second[j]]}, 1, cs.K, cs.regime, hr, &ch2[j]);
          upload(int(j), s);
          s1.push_back(seeds[second[j]]);
        }
        run(int(second.size()), 1, s1);
        for (size_t j = 0; j < second.size(); ++j) {
          const int a = res.as<int>()[j * 4];
          const int y = res.as<int>()[j * 4 + 1];
          t2[second[j]] = a >= 1 ? ch2[j][0] : y;
        }
      }
      for (int i = 0; i < nb; ++i) {
        obs[long(t1[i]) * 100000 + t2[i]]++;
        obs1[t1[i]]++;
      }
      done += nb;
    }
    // expected law
    std::map<long, double> ex, ex1;
    const Dist p0 = p_of({}, cs.regime);
    for (size_t i = 0; i < p0.id.size(); ++i) {
      ex1[p0.id[i]] = p0.p[i];
      const Dist p1 = p_of({p0.id[i]}, cs.regime);
      for (size_t j = 0; j < p1.id.size(); ++j) ex[long(p0.id[i]) * 100000 + p1.id[j]] += double(p0.p[i]) * p1.p[j];
    }
    double st, pv, st1, pv1;
    int dof, dof1;
    long out, out1;
    chi2(obs, ex, trials, &st, &dof, &pv, &out);
    chi2(obs1, ex1, trials, &st1, &dof1, &pv1, &out1);
    const bool ok = pv > 0.01 && pv1 > 0.01 && out == 0 && out1 == 0;
    char dyn[64] = "";
    if (cs.tau > 0) snprintf(dyn, sizeof dyn, ", depth tau %.2f rule %d", cs.tau, cs.rule);
    const bool expect_fail = cs.rule == 1;
    printf("[S1] %-24s K=%d T=%.1f%s: %ld trials, mean a %.2f; joint chi2 %.1f (%d dof) p=%.3f; first token chi2 "
           "%.1f (%d dof) p=%.3f; outside support %ld: %s%s\n",
           regime_name(cs.regime), cs.K, cs.temp, dyn, trials, mean_a / trials, st, dof, pv, st1, dof1, pv1,
           out + out1, ok ? "PASS" : "FAIL", expect_fail ? (ok ? " (biased rule NOT detected)" : " (biased rule detected, as expected)") : "");
    if (ok == expect_fail) ++fails;
  }
  printf(fails ? "FAILED: %d\n" : "ALL PASSED\n", fails);
  return fails ? 1 : 0;
}
