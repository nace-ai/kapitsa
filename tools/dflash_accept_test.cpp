// S1: exactness of the on-GPU acceptance on synthetic inputs,
// no model needed. (Cases are numbered in order: the chain's, then the tree's.)
//   dflash_accept_test [trials=40000] [K=3]
//   dflash_accept_test --cpu   (the budgets' checks alone, no GPU: KfBudget's fit, cache and costs, KfBatch)
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
// 3. KFlash tree verifies: the shared builder (kf_build) takes a best-first tree from a synthetic DFlash2 lattice
//    (random, or aligned with the target's likely path; the identity or a random calibration; 1-8 live rows of an
//    8-row verify, one count allowed, or every count 1-8 under the size rule, n per trial: the MoE targets'; the
//    wide verifies' 16 and 32 rows, and the size rule choosing among 8 | 16 | 32 by cost), the target's rows of the
//    tree's paths are sampled by sample_stage2_rows_tree (top-k, top-p, temperature as the engine's sampler, each row
//    with the uniform of its token's position; the live rows alone), the shared kf_accept walks them; a second cycle
//    runs when one token came out.
//    The joint law of the first two tokens is tested as in 2, and every trial's two tokens must equal plain decoding's
//    with the same position-keyed uniforms (the sampler transcribed on the CPU); T = 0: equal to the argmax decoding.
#include "metal/gpu.h"
#include "model/kflash_budget.h"
#include "model/kparams.h"
#include "util/log.h"

#include <dirent.h>
#include <nlohmann/json.hpp>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <thread>
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

// ---- tree drafting (section 3) ----
// The target's rows: stage-1 candidates over the whole universe (kV ids), values log p(.|prefix) on p's support and
// lower ones elsewhere, sampled by sample_stage2_rows_tree with top-k / top-p / temperature like the engine.
constexpr int kTreeN = kV;
void target_row(const std::vector<int>& prefix, int regime, float* v, uint32_t* id) {
  const Dist p = p_of(prefix, regime == kPoint ? kPoint : kRandom);
  std::mt19937_64 r(key_of(prefix) ^ 0x3333);
  float lo = 0;
  for (size_t i = 0; i < p.id.size(); ++i) lo = std::min(lo, std::log(p.p[i]));
  int n = 0;
  for (size_t i = 0; i < p.id.size(); ++i) {
    v[n] = std::log(p.p[i]);
    id[n++] = uint32_t(p.id[i]);
  }
  for (int t = 0; t < kV && n < kTreeN; ++t) {
    if (std::find(p.id.begin(), p.id.end(), kBase + t) != p.id.end()) continue;
    v[n] = lo - 4.0f - float(r() % 8) * 0.5f;  // (ties on purpose: the sampler orders them by id)
    id[n++] = uint32_t(kBase + t);
  }
}

// The plain sampler's law of a row (sample_stage2_rows, transcribed): top-k by (value desc, id asc), top-p on the
// untempered probabilities, then the temperature on the kept prefix; T = 0: the first of that order.
struct Law {
  std::vector<uint32_t> id;
  std::vector<float> p;  // the kernel's float arithmetic (unnormalized probs; kt = their sum in order)
  float kt = 1;
};
Law row_law(const float* v, const uint32_t* id, int n, int K, float top_p, float temp) {
  std::vector<std::pair<float, uint32_t>> c(n);
  for (int i = 0; i < n; ++i) c[i] = {v[i], id[i]};
  std::sort(c.begin(), c.end(),
            [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
  Law L;
  if (temp <= 0) {
    L.id = {c[0].second};
    L.p = {1.0f};
    return L;
  }
  K = std::min(K, n);
  std::vector<float> probs(K);
  float tot = 0;
  for (int q = 0; q < K; ++q) {
    probs[q] = std::exp(c[q].first - c[0].first);
    tot += probs[q];
  }
  int keep = K;
  float cum = 0;
  for (int q = 0; q < K; ++q) {
    cum += probs[q] / tot;
    if (cum >= top_p) {
      keep = q + 1;
      break;
    }
  }
  L.kt = 0;
  for (int q = 0; q < keep; ++q) {
    L.id.push_back(c[q].second);
    L.p.push_back(std::exp((c[q].first - c[0].first) / temp));
    L.kt += L.p.back();
  }
  return L;
}
uint32_t pcg(uint32_t v) {
  const uint32_t state = v * 747796405u + 2891336453u;
  const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
// the draw sample_stage2_rows makes with RNG step `key` (the position-keyed uniform)
int law_draw(const Law& L, uint32_t seed, uint32_t key) {
  if (L.id.size() == 1) return int(L.id[0]);
  const float u = float(pcg(seed ^ pcg(key)) & 0xffffff) / 16777216.0f * L.kt;
  float acc = 0;
  for (size_t q = 0; q < L.id.size(); ++q) {
    acc += L.p[q];
    if (u < acc) return int(L.id[q]);
  }
  return int(L.id.back());
}

// A synthetic lattice (cand_i [7][16] from the universe, cand_v, edges [7][16][16]) for a cycle after `prefix`:
// random, or aligned with the target's likely path (the walk's chain is then mostly right and the siblings are p's
// next tokens), drawn from r (independent of the target's uniforms, as a drafter's lattice is).
void lattice(const std::vector<int>& prefix, int regime, std::mt19937_64& r, int* ci, float* cv, float* E) {
  std::normal_distribution<float> N(0, 1);
  std::vector<int> path = prefix;
  for (int d = 0; d < 7; ++d) {
    std::vector<int> u(kV);
    for (int i = 0; i < kV; ++i) u[i] = kBase + i;
    std::shuffle(u.begin(), u.end(), r);
    if (regime == kEqual) {  // aligned: p's support of the main path's prefix first, values log p + noise
      const Dist p = p_of(path, kRandom);
      std::vector<int> c(p.id.begin(), p.id.end());
      for (int t : u)
        if (c.size() < 16 && std::find(c.begin(), c.end(), t) == c.end()) c.push_back(t);
      for (int k = 0; k < 16; ++k) {
        ci[d * 16 + k] = c[k];
        cv[d * 16 + k] = k < int(p.id.size()) ? std::log(p.p[k]) + 0.3f * N(r) : -6.0f + N(r);
      }
      path.push_back(p.id[0]);
    } else {
      for (int k = 0; k < 16; ++k) {
        ci[d * 16 + k] = u[k];
        cv[d * 16 + k] = 1.5f * N(r);
      }
    }
    for (int j = 0; j < 16; ++j)
      for (int k = 0; k < 16; ++k) E[d * 256 + j * 16 + k] = regime == kEqual ? 0.0f : 0.7f * N(r);
  }
}

int tree_cases(long trials, uint64_t salt, int only) {
  Gpu& g = Gpu::get();
  struct TCase {
    int regime;     // lattice: kRandom or kEqual (aligned); target: kRandom, or kPoint (1-2 tokens: deep paths)
    float temp, top_p;
    int top_k;
    bool cal;  // a random per-depth calibration (alpha, beta) instead of the identity (the law must not depend on it)
    int R;     // live rows (the budget allows this count alone; 16 / 32: a wide verify of that many rows); 0: every
               // count 1 .. 8, the size rule picks n per trial; -1: the size rule over 8 | 16 | 32 (the wide verifies'
               // counts) with costs that make every count win in some trials
    float rate = 0.0f;  // the rule's cases: the recent tokens per ms (E(n) - rate cost[n]; 0: E(n) / cost[n])
  };
  const std::vector<TCase> cases = {{kRandom, 0.6f, 0.95f, 20, false, 8}, {kEqual, 0.6f, 0.95f, 20, true, 8},
                                    {kEqual, 1.0f, 1.0f, 32, true, 5},    {kPoint, 0.6f, 0.95f, 20, true, 3},
                                    {kRandom, 0.0f, 1.0f, 1, false, 8},   {kEqual, 0.0f, 1.0f, 1, true, 6},
                                    {kEqual, 0.6f, 0.95f, 20, true, 1},   {kEqual, 0.6f, 0.95f, 20, true, 0},
                                    {kRandom, 0.6f, 0.95f, 20, false, 0}, {kEqual, 0.0f, 1.0f, 1, true, 0},
                                    {kEqual, 0.6f, 0.95f, 20, true, 0, 0.1f},
                                    {kRandom, 0.0f, 1.0f, 1, false, 0, 0.06f},
                                    // the wide verifies (dense targets): 16 and 32 rows, and the rule over 8 | 16 | 32
                                    {kEqual, 0.6f, 0.95f, 20, true, 16},  {kRandom, 0.6f, 0.95f, 20, false, 32},
                                    {kPoint, 0.6f, 0.95f, 20, true, 32},  {kEqual, 1.0f, 1.0f, 32, true, 32},
                                    {kEqual, 0.0f, 1.0f, 1, true, 32},    {kRandom, 0.0f, 1.0f, 1, false, 16},
                                    {kEqual, 0.6f, 0.95f, 20, true, -1},  {kEqual, 0.0f, 1.0f, 1, true, -1},
                                    {kRandom, 0.6f, 0.95f, 20, false, -1}};
  Pipeline table_p = g.pipeline("dfl_kf_table"), build_p = g.pipeline("kf_build"),
           samp_p = g.pipeline("sample_stage2_rows_tree"), acc_p = g.pipeline("kf_accept");
  const int B = 2048, W = kf::kWords, RM = kf::kMaxRows;
  constexpr int TS = 48;  // tokens per trial: the anchor's position P <= 1, its rows P .. P + 31, the token after them
  Buffer ci = g.alloc(size_t(B) * 112 * 4), cv = g.alloc(size_t(B) * 112 * 4), E = g.alloc(size_t(B) * 1792 * 4);
  Buffer clq = g.alloc(size_t(B) * 1792 * 4), ctok = g.alloc(size_t(B) * 1792 * 4), next = g.alloc(1792 * 4);
  Buffer tok = g.alloc(size_t(B) * TS * 4), tree = g.alloc(size_t(B) * W * 4), live = g.alloc(size_t(B) * 16);
  Buffer rv = g.alloc(size_t(B) * RM * kTreeN * 4);
  Buffer rid = g.alloc(size_t(B) * RM * kTreeN * 4), samp = g.alloc(size_t(B) * RM * 4), res = g.alloc(size_t(B) * 16);
  Buffer gst = g.alloc(size_t(B) * 64), path = g.alloc(size_t(B) * kf::kPathWords * 4);
  {
    const std::vector<int32_t> nx = kf::lattice_next(7, 16);
    memcpy(next.ptr(), nx.data(), nx.size() * 4);
  }
  int fails = 0;
  for (size_t cix = 0; cix < cases.size(); ++cix) {
    if (only >= 0 && int(cix) != only) continue;
    const TCase& cs = cases[cix];
    const int tgt = cs.regime == kPoint ? kPoint : kRandom;
    const int lat = cs.regime == kEqual ? kEqual : kRandom;
    std::map<long, long> obs, obs1;
    long done = 0, path_bad = 0, compared = 0, nhist[kf::kMaxRows + 1] = {};
    double mean_a = 0;
    std::mt19937_64 hr(4321 + cix * 7 + salt * 1000003);
    KfParams kp{};
    const bool wide_rule = cs.R < 0;  // the size rule over 8 | 16 | 32
    const int cap = wide_rule ? RM : std::max(8, cs.R);  // the verify's rows (the builder's max)
    {  // the budget's parameters the builder reads: the identity, or a random calibration; this case's rows alone
      std::uniform_real_distribution<float> U(0, 1);
      for (int d = 0; d <= kf::kMaxRows; ++d) {
        kp.alpha[d] = cs.cal ? -1.5f + 2.0f * U(hr) : 0.0f;
        kp.beta[d] = cs.cal ? 0.5f + 1.0f * U(hr) : 1.0f;
        kp.cost[d] = 1.0f;
      }
      kp.force_n = 0;
      // the size rule's cases: every count 1 .. 8, a cycle's cost growing with its rows (a MoE verify's experts); or
      // 8 | 16 | 32, the wider verifies costing more, so a tree's breadth decides (each count gets picked)
      kp.allowed = wide_rule ? (1u << 7) | (1u << 15) | (1u << 31) : cs.R ? 1u << (cs.R - 1) : 0xffu;
      if (!cs.R)
        for (int nn = 1; nn <= 8; ++nn) kp.cost[nn] = 8.0f + float(nn);
      if (wide_rule) {
        kp.cost[16] = 1.25f;
        kp.cost[32] = 1.6f;
      }
      kp.max_rows = cap;
      kp.cap = cap;
      kp.rate = cs.rate;
    }
    // the rule's choice from the descriptor's E(n) (kf_build's: the first n maximizing E(n) - rate cost[n], rate 0:
    // E(n) / cost[n], over the allowed counts within the rows built)
    auto rule_n = [&](const int* tr) {
      const int m = tr[kf::kNodes];
      float best = -INFINITY;
      int n = 0;
      for (int nn = 1; nn <= m; ++nn) {
        float E;
        memcpy(&E, &tr[kf::kE + nn], 4);
        const float v = kp.rate > 0.0f ? E - kp.rate * kp.cost[nn] : E / kp.cost[nn];
        if (((kp.allowed >> (nn - 1)) & 1u) && v > best) {
          best = v;
          n = nn;
        }
      }
      return n ? n : m;
    };
    // one cycle for trials [0, n) after prefix pre[i] (the anchor at position P = pre[i].size()); the emitted tokens
    auto cycle = [&](const std::vector<std::vector<int>>& pre, const std::vector<uint32_t>& seed) {
      const int n = int(pre.size());
      for (int i = 0; i < n; ++i) {
        lattice(pre[i], lat, hr, ci.as<int>() + i * 112, cv.as<float>() + i * 112, E.as<float>() + i * 1792);
        int* tk = tok.as<int>() + i * TS;
        for (int j = 0; j < TS; ++j) tk[j] = 0;  // tokens[0]: the first anchor (outside the universe)
        for (size_t j = 0; j < pre[i].size(); ++j) tk[1 + j] = pre[i][j];  // the anchor at P = pre.size()
      }
      {  // the trees: the lattice's tables, then the shared builder
        auto e = g.begin();
        for (int i = 0; i < n; ++i) {
          const uint32_t P = uint32_t(pre[i].size());
          e->use(table_p);
          e->buf(0, ci, size_t(i) * 448); e->buf(1, cv, size_t(i) * 448); e->buf(2, E, size_t(i) * 1792 * 4);
          e->buf(3, clq, size_t(i) * 1792 * 4); e->buf(4, ctok, size_t(i) * 1792 * 4);
          e->dispatch(7 * 16, 1, 1, 32);
          const kp::KfBuild bp{16u, P, 0u, 0u};
          e->use(build_p);
          e->buf(0, clq, size_t(i) * 1792 * 4); e->buf(1, ctok, size_t(i) * 1792 * 4); e->buf(2, next);
          e->val(3, kp);
          e->buf(4, tree, size_t(i) * W * 4); e->buf(5, tok, size_t(i) * TS * 4); e->buf(6, live, size_t(i) * 16);
          e->val(7, bp);
          e->dispatch(1, 1, 1, 32);
        }
        e->commit_and_wait();
      }
      for (int i = 0; i < n; ++i) {  // the target's rows: slot t's path from the tree (the verify's rows, all cap)
        const int* tk = tok.as<int>() + i * TS;
        const int* tr = tree.as<int>() + i * W;
        const int P = int(pre[i].size());
        for (int t = 0; t < cap; ++t) {
          std::vector<int> nodes;
          for (int v = t; v > 0; v = tr[kf::kPar + v]) nodes.push_back(tk[P + v]);
          std::vector<int> pp = pre[i];
          pp.insert(pp.end(), nodes.rbegin(), nodes.rend());
          target_row(pp, tgt, rv.as<float>() + (size_t(i) * RM + t) * kTreeN,
                     rid.as<uint32_t>() + (size_t(i) * RM + t) * kTreeN);
        }
      }
      {
        auto e = g.begin();
        for (int i = 0; i < n; ++i) {
          const uint32_t P = uint32_t(pre[i].size());
          kp::SampleRows sr{uint32_t(kTreeN), uint32_t(cs.top_k), cs.temp, cs.top_p, seed[i], P, 0, 0, 0};
          e->use(samp_p);
          e->buf(0, rv, size_t(i) * RM * kTreeN * 4); e->buf(1, rid, size_t(i) * RM * kTreeN * 4);
          e->buf(2, samp, size_t(i) * RM * 4);
          e->val(3, sr);
          e->buf(4, rid); e->buf(5, tree, size_t(i) * W * 4);
          e->dispatch(uint32_t(cap), 1, 1, 1024);
          const kp::KfAccept ap{P, uint32_t(cap), 1u, 0u};
          e->use(acc_p);
          e->buf(0, samp, size_t(i) * RM * 4); e->buf(1, tok, size_t(i) * TS * 4); e->buf(2, tree, size_t(i) * W * 4);
          e->buf(3, res, size_t(i) * 16);
          e->val(4, ap);
          e->buf(5, live, size_t(i) * 16); e->buf(6, path, size_t(i) * kf::kPathWords * 4); e->buf(7, gst, size_t(i) * 64);
          e->dispatch(1, 1, 1, 32);
        }
        e->commit_and_wait();
      }
      std::vector<std::vector<int>> out(n);
      for (int i = 0; i < n; ++i) {
        const int a = res.as<int>()[i * 4], P = int(pre[i].size());
        const int* tr = tree.as<int>() + i * W;
        for (int j = 1; j <= a + 1; ++j) out[i].push_back(tok.as<int>()[i * TS + P + j]);
        if (out[i].back() != res.as<int>()[i * 4 + 1]) ++path_bad;  // the token after the path
        const int nx = cs.R > 0 ? cs.R : rule_n(tr);
        if (tr[kf::kN] != nx || int(live.as<uint32_t>()[i * 4]) != nx || tr[kf::kA] != a || tr[kf::kObs] != nx)
          ++path_bad;
        if (a > nx - 1) ++path_bad;
        if (P == 0) nhist[std::clamp(nx, 0, int(kf::kMaxRows))]++;
        if (int(gst.as<uint32_t>()[i * 16]) != P + a + 1 || int(gst.as<uint32_t>()[i * 16 + 1]) != a + 1) ++path_bad;
        for (int j = 0; j <= a; ++j)  // the path: rows of the accepted branch, the drafter state's and the sequence's
          if (path.as<int>()[i * kf::kPathWords + j] != tr[kf::kPath + j] ||
              int(gst.as<uint32_t>()[i * 16 + 4 + j]) != tr[kf::kPath + j])
            ++path_bad;
        mean_a += P == 0 ? a : 0;
      }
      return out;
    };
    while (done < trials) {
      const int nb = int(std::min<long>(B, trials - done));
      std::vector<uint32_t> seeds(nb);
      for (int i = 0; i < nb; ++i)
        seeds[i] = uint32_t(mix(uint64_t(done + i) + (uint64_t(cix + 100) << 40) + (salt << 48)));
      std::vector<std::vector<int>> pre0(nb);
      auto em = cycle(pre0, seeds);
      std::vector<int> second;
      for (int i = 0; i < nb; ++i)
        if (em[i].size() < 2) second.push_back(i);
      if (!second.empty()) {
        std::vector<std::vector<int>> pre1;
        std::vector<uint32_t> s1;
        for (int i : second) {
          pre1.push_back({em[i][0]});
          s1.push_back(seeds[i]);
        }
        auto em2 = cycle(pre1, s1);
        for (size_t j = 0; j < second.size(); ++j) em[second[j]].push_back(em2[j][0]);
      }
      for (int i = 0; i < nb; ++i) {
        const int t1 = em[i][0], t2 = em[i][1];
        obs[long(t1) * 100000 + t2]++;
        obs1[t1]++;
        // pathwise: plain decoding with position-keyed uniforms writes the same two tokens
        float v[kTreeN];
        uint32_t id[kTreeN];
        target_row({}, tgt, v, id);
        const int p1 = law_draw(row_law(v, id, kTreeN, cs.top_k, cs.top_p, cs.temp), seeds[i], 0);
        target_row({p1}, tgt, v, id);
        const int p2 = law_draw(row_law(v, id, kTreeN, cs.top_k, cs.top_p, cs.temp), seeds[i], 1);
        ++compared;
        if (p1 != t1 || p2 != t2) ++path_bad;
      }
      done += nb;
    }
    // expected law of (t1, t2)
    std::map<long, double> ex, ex1;
    float v[kTreeN];
    uint32_t id[kTreeN];
    target_row({}, tgt, v, id);
    const Law l0 = row_law(v, id, kTreeN, cs.top_k, cs.top_p, cs.temp);
    for (size_t i = 0; i < l0.id.size(); ++i) {
      const double p0 = l0.p[i] / l0.kt;
      ex1[long(l0.id[i])] += p0;
      target_row({int(l0.id[i])}, tgt, v, id);
      const Law l1 = row_law(v, id, kTreeN, cs.top_k, cs.top_p, cs.temp);
      for (size_t j = 0; j < l1.id.size(); ++j) ex[long(l0.id[i]) * 100000 + long(l1.id[j])] += p0 * l1.p[j] / l1.kt;
    }
    double st, pv, st1, pv1;
    int dof, dof1;
    long out, out1;
    chi2(obs, ex, trials, &st, &dof, &pv, &out);
    chi2(obs1, ex1, trials, &st1, &dof1, &pv1, &out1);
    const bool greedy = cs.temp <= 0;
    // the rule's cases must reach several counts (else they test one count again)
    int counts = 0;
    for (int nn = 1; nn <= kf::kMaxRows; ++nn) counts += nhist[nn] > 0;
    const bool ok = (greedy || (pv > 0.01 && pv1 > 0.01)) && out == 0 && out1 == 0 && path_bad == 0 &&
                    (cs.R > 0 || counts >= (wide_rule ? 2 : 3));
    char rows[160];
    if (cs.R > 0) {
      snprintf(rows, sizeof rows, "%d rows", cs.R);
    } else if (wide_rule) {
      snprintf(rows, sizeof rows, "rule 8|16|32 (%ld/%ld/%ld)", nhist[8], nhist[16], nhist[32]);
    } else {
      int o = cs.rate > 0 ? snprintf(rows, sizeof rows, "rate rule (%.2f) over 1-8 rows (trials by n", cs.rate)
                          : snprintf(rows, sizeof rows, "rule over 1-8 rows (trials by n");
      for (int nn = 1; nn <= 8 && o < int(sizeof rows) - 12; ++nn)
        if (nhist[nn]) o += snprintf(rows + o, sizeof rows - o, " %d:%ld", nn, nhist[nn]);
      snprintf(rows + o, sizeof rows - o, ")");
    }
    printf("[S1 tree] lattice %-8s target %-6s T=%.1f top-p %.2f top-k %d %s, %s: %ld trials, mean a %.2f; "
           "joint chi2 %.1f (%d dof) p=%.3f; first token p=%.3f; outside support %ld; plain-decoding mismatches %ld of "
           "%ld: %s\n",
           lat == kEqual ? "aligned" : "random", tgt == kPoint ? "1-2" : "random", cs.temp, cs.top_p, cs.top_k,
           cs.cal ? "calibrated" : "identity", rows, trials, mean_a / trials, st, dof, pv, pv1, out + out1, path_bad,
           compared, ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
  }
  return fails;
}

// The shared budget's calibration fit (KfBudget::fit): records drawn from a known per-depth calibration recover it.
int fit_check(uint64_t salt) {
  std::mt19937_64 r(99 + salt);
  std::uniform_real_distribution<float> ul(-6.0f, -0.01f), u01(0.0f, 1.0f);
  int bad = 0;
  for (int t = 0; t < 6; ++t) {
    const float ta = -1.2f + 0.4f * float(t), tb = 0.6f + 0.25f * float(t);
    std::vector<float> x;
    std::vector<uint8_t> y;
    for (int i = 0; i < 2048; ++i) {
      const float lq = ul(r), xi = KfBudget::logit_q(lq);
      x.push_back(xi);
      y.push_back(u01(r) < 1.0f / (1.0f + std::exp(-(ta + tb * xi))));
    }
    float a = 0.0f, b = 1.0f;
    KfBudget::fit(x, y, a, b, 8);
    const bool ok = std::fabs(a - ta) < 0.3f && std::fabs(b - tb) < 0.25f;
    if (!ok) ++bad;
    printf("[fit] alpha %.2f beta %.2f from 2048 records drawn with alpha %.2f beta %.2f: %s\n", a, b, ta, tb,
           ok ? "PASS" : "FAIL");
  }
  return bad ? 1 : 0;
}

// The shared budget's cache file (KfBudget::save, load; KAPITSA_HOME/kflash): nothing before 32 cycles, nor without
// new ones since the last save, nor before kSaveEvery new ones when asked for that many; written whole (no temporary
// file left); its table (cost_of: the cycle's GPU ms) comes back as measured, a table without cost_of is measured
// again with the calibration kept; saves from another thread while cycles are observed leave a whole file.
int cache_check() {
  const char* t = getenv("TMPDIR");
  std::string dir = std::string(t && *t ? t : "/tmp") + "/kf_cache_XXXXXX";
  if (!mkdtemp(dir.data())) {
    printf("[cache] no temporary folder: FAIL\n");
    return 1;
  }
  const char* h = getenv("KAPITSA_HOME");
  const std::string old_home = h ? h : "";
  setenv("KAPITSA_HOME", dir.c_str(), 1);
  const std::string kdir = dir + "/kflash", file = kdir + "/t.json";
  auto exists = [](const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
  };
  auto files = [&] {
    std::vector<std::string> v;
    if (DIR* d = opendir(kdir.c_str())) {
      while (dirent* e = readdir(d))
        if (e->d_name[0] != '.') v.push_back(e->d_name);
      closedir(d);
    }
    std::sort(v.begin(), v.end());
    return v;
  };
  int bad = 0;
  auto check = [&](bool ok, const char* what) {
    printf("[cache] %s: %s\n", what, ok ? "PASS" : "FAIL");
    bad += !ok;
  };
  std::vector<int32_t> tree(kf::kWords, 0);
  tree[kf::kN] = 8;
  {
    KfBudget b("t");
    b.set_rows(1u << 7, 8);
    for (int i = 0; i < 31; ++i) b.observe(tree.data(), 30.0);
    b.save();
    check(!exists(file), "31 cycles, save(): no file (32 needed)");
    b.observe(tree.data(), 30.0);
    b.save(KfBudget::kSaveEvery);
    check(!exists(file), "32 cycles, save(kSaveEvery): no file (kSaveEvery new cycles needed)");
    b.save();
    check(exists(file), "32 cycles, save(): written");
    unlink(file.c_str());
    b.save();
    check(!exists(file), "no cycle since the last save, save(): not written again");
    b.observe(tree.data(), 30.0);
  }  // the destructor saves the 33rd
  check(files() == std::vector<std::string>{"t.json"}, "the destructor wrote it, whole (no temporary file left)");
  {
    KfBudget b("t");
    b.set_rows(1u << 7, 8);
    KfParams p{};
    b.write(p);
    check(p.force_n == 0 && std::fabs(p.cost[8] - 30.0f) < 1e-4f, "loaded: the cost table counts as measured");
  }
  {
    nlohmann::json j = nlohmann::json::parse(std::ifstream(file));
    j.erase("cost_of");
    j["alpha"][1] = 0.5;
    std::ofstream(file) << j.dump();
    KfBudget b("t");
    b.set_rows(1u << 7, 8);
    KfParams p{};
    b.write(p);
    check(p.force_n == 8 && p.cost[8] == 1.0f && p.alpha[1] == 0.5f,
          "a table without cost_of: measured again (8 rows forced), the calibration kept");
  }
  {
    const int lvl = log_level();
    log_level() = 0;  // (one "budget saved" line per save)
    KfBudget b("t2");
    b.set_rows(1u << 7, 8);
    std::atomic<bool> done{false};
    std::thread saver([&] {
      while (!done) b.save();
    });
    for (int i = 0; i < 20000; ++i) b.observe(tree.data(), 30.0 + i % 7);
    done = true;
    saver.join();
    b.save();
    log_level() = lvl;
    bool whole = false;
    try {
      const nlohmann::json j = nlohmann::json::parse(std::ifstream(kdir + "/t2.json"));
      whole = j.value("cost_of", "") == "cycle GPU ms" && j.at("cost").at(8).get<double>() > 30.0;
    } catch (...) {
    }
    check(whole && files() == std::vector<std::string>({"t.json", "t2.json"}),
          "saves from another thread while cycles are observed: one whole file");
  }
  for (const auto& f : files()) unlink((kdir + "/" + f).c_str());
  rmdir(kdir.c_str());
  rmdir(dir.c_str());
  if (h) setenv("KAPITSA_HOME", old_home.c_str(), 1);
  else unsetenv("KAPITSA_HOME");
  return bad ? 1 : 0;
}

// The shared budget's costs and rate (KfBudget, every count 1 .. 8 allowed, as on MoE targets): the counts explored
// kExplore times each, then cycles of 8 rows whose cost rises with the context (+7 ms over 400 cycles): every count's
// cost follows (a common base), not only the count the cycles verify; the rate is the tokens per ms they gave; a
// process loading the cache verifies a count it holds first and its cost sets the base (+7 ms more here).
int cost_check() {
  const char* t = getenv("TMPDIR");
  std::string dir = std::string(t && *t ? t : "/tmp") + "/kf_cost_XXXXXX";
  if (!mkdtemp(dir.data())) {
    printf("[costs] no temporary folder: FAIL\n");
    return 1;
  }
  const char* h = getenv("KAPITSA_HOME");
  const std::string old_home = h ? h : "";
  setenv("KAPITSA_HOME", dir.c_str(), 1);
  int bad = 0;
  auto check = [&](bool ok, const char* what, double got, double want) {
    printf("[costs] %s: %.3f (expected %.3f): %s\n", what, got, want, ok ? "PASS" : "FAIL");
    bad += !ok;
  };
  const int lvl = log_level();
  log_level() = 0;
  auto cost = [](int n, double base) { return base + 1.1 * (n - 1); };  // a cycle's ms: 1.1 ms per row
  std::vector<int32_t> tree(kf::kWords, 0);
  auto cycle = [&](KfBudget& b, int n, double ms, int a) {
    tree[kf::kObs] = n;
    const float e = 1.0f + float(n) * 0.4f;
    memcpy(&tree[kf::kObs + 1], &e, 4);
    tree[kf::kN] = 8;  // (the next cycle's choice already)
    tree[kf::kA] = a;
    b.observe(tree.data(), ms);
  };
  {
    KfBudget b("c");
    b.set_rows(0xffu, 8);
    int explored = 0;
    for (int i = 0; i < 64; ++i) {
      KfParams p{};
      b.write(p);
      if (!p.force_n) break;
      cycle(b, p.force_n, cost(p.force_n, 14.0), 1);
      ++explored;
    }
    check(explored == 8 * KfBudget::kExplore, "exploration: forced cycles", explored, 8 * KfBudget::kExplore);
    for (int i = 0; i < 400; ++i) cycle(b, 8, cost(8, 14.0 + 7.0 * (i + 1) / 400.0), 2);
    KfParams p{};
    b.write(p);
    check(std::fabs(p.cost[8] - cost(8, 21.0)) < 0.3, "8 rows (verified) after the drift", p.cost[8], cost(8, 21.0));
    check(std::fabs(p.cost[1] - cost(1, 21.0)) < 0.6, "1 row (not verified since) after the drift", p.cost[1],
          cost(1, 21.0));
    check(std::fabs(p.cost[8] - p.cost[1] - 7.7) < 0.6, "8 rows over 1 row", p.cost[8] - p.cost[1], 7.7);
    // (the rate's sums decay by 1/64 a cycle: their ms lag the drift by ~63 cycles)
    const double rate = 3.0 / cost(8, 21.0 - 7.0 * 63.0 / 400.0);
    check(std::fabs(p.rate - rate) < 0.002, "rate (tokens per ms)", p.rate, rate);
  }  // saved
  {
    KfBudget b("c");
    b.set_rows(0xffu, 8);
    KfParams p{};
    b.write(p);
    check(p.force_n == 8 && p.rate == 0.0f, "loaded: the first cycle verifies a count the cache holds", p.force_n, 8);
    cycle(b, 8, cost(8, 28.0), 2);
    b.write(p);
    check(p.force_n == 0 && std::fabs(p.cost[1] - cost(1, 28.0)) < 0.6, "loaded, +7 ms: 1 row's cost from the base",
          p.cost[1], cost(1, 28.0));
  }
  log_level() = lvl;
  const std::string kdir = dir + "/kflash";
  unlink((kdir + "/c.json").c_str());
  rmdir(kdir.c_str());
  rmdir(dir.c_str());
  if (h) setenv("KAPITSA_HOME", old_home.c_str(), 1);
  else unsetenv("KAPITSA_HOME");
  return bad ? 1 : 0;
}

// KFlash's batch budget (KfBatch). The rule (KfBatch::allocate) on random batches of 2-4 members (falling survivals
// with exact ties, caps, convex costs; the rate rule, the ratio rule, a floor of one draft, forced counts): bounds (rows
// <= 8, caps, floors, forced counts); optimal under the rate rule (brute force over every allocation); causal (a
// member's draft p is in or out whatever its survivals past p are: a sampled member's draw at p moves only those, so
// its speculative sampling stays exact). The cost line (an outlier, a falling table). The budget: the exploration, its
// costs, the refresh of a count measured while busy, the cache, greedy records to KfBudget and sampled ones to its own
// calibration, a floor.
int batch_check(uint64_t salt) {
  constexpr int D = KfBatch::kDrafts, R = KfBatch::kRows;
  std::mt19937_64 r(4242 + salt);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  long bounds_bad = 0, opt_bad = 0, causal_bad = 0, opt_n = 0, causal_n = 0;
  auto chain = [&](float* sv, int g, int from) {  // survivals of member g from position `from` on (1: an exact tie)
    for (int p = from; p < D; ++p)
      sv[g * D + p] = float((p ? double(sv[g * D + p - 1]) : 1.0) * (r() % 6 == 0 ? 1.0 : u01(r)));
  };
  for (int t = 0; t < 20000; ++t) {
    const int M = 2 + int(r() % 3), floor = t % 5 == 0, force = t % 7 == 0 ? M + int(r() % (R - M + 1)) : 0;
    float sv[R * D];
    int kmax[R], k[R], fl = 0, room = 0;
    for (int g = 0; g < M; ++g) {
      chain(sv, g, 0);
      kmax[g] = r() % 4 == 0 ? int(r() % (D + 1)) : D;
      fl += std::min(floor, kmax[g]);
      room += kmax[g];
    }
    double cost[R + 1];
    const double base = 10.0 + 60.0 * u01(r), slope = 2.0 * u01(r), curv = 0.3 * u01(r);
    for (int n = 0; n <= R; ++n) cost[n] = base + slope * n + curv * n * n;
    const double rate = t % 3 == 0 ? 0.0 : 0.02 + 0.6 * u01(r);
    const int n = KfBatch::allocate(sv, M, kmax, cost, rate, floor, force, k);
    auto value = [&](const int* kk) {  // expected tokens - rate x cost (-inf past 8 rows)
      double e = 0;
      int nn = M;
      for (int g = 0; g < M; ++g) {
        for (int p = 0; p < kk[g]; ++p) e += sv[g * D + p];
        nn += kk[g];
      }
      return nn > R ? -1e30 : e - rate * cost[nn];
    };
    int sum = 0;
    bool ok = true;
    for (int g = 0; g < M; ++g) sum += k[g], ok = ok && k[g] <= kmax[g] && k[g] >= std::min(floor, kmax[g]);
    bounds_bad += !(ok && n == M + sum && n <= R && (!force || n == std::max(std::min(force, M + room), M + fl)));
    if (rate > 0 && !force) {  // brute force: every allocation within the caps and the floors
      double best = -1e30;
      int kk[R] = {};
      std::function<void(int)> rec = [&](int g) {
        if (g == M) return void(best = std::max(best, value(kk)));
        for (kk[g] = std::min(floor, kmax[g]); kk[g] <= kmax[g]; ++kk[g]) rec(g + 1);
      };
      rec(0);
      ++opt_n;
      opt_bad += value(k) < best - 1e-6;
    }
    for (int g = 0; g < M; ++g)  // member g's survivals past p redrawn (still falling): draft p in or out as before
      for (int p = 0; p < kmax[g]; ++p) {
        float s2[R * D];
        memcpy(s2, sv, sizeof s2);
        chain(s2, g, p + 1);
        int k2[R];
        KfBatch::allocate(s2, M, kmax, cost, rate, floor, force, k2);
        ++causal_n;
        causal_bad += (k2[g] > p) != (k[g] > p);
      }
  }
  int bad = 0;
  auto check = [&](bool ok, const std::string& what) {
    printf("[batch] %s: %s\n", what.c_str(), ok ? "PASS" : "FAIL");
    bad += !ok;
  };
  check(bounds_bad == 0, "rule on 20000 random batches: bounds and forced counts (" + std::to_string(bounds_bad) + " broken)");
  check(opt_bad == 0 && opt_n > 1000, "rate rule optimal against every allocation: " + std::to_string(opt_bad) +
                                          " of " + std::to_string(opt_n) + " not");
  check(causal_bad == 0 && causal_n > 10000, "a draft's place independent of its member's later survivals: " +
                                                 std::to_string(causal_bad) + " of " + std::to_string(causal_n) + " moved");
  {
    double c[R + 1] = {}, out[R + 1] = {}, f[R + 1] = {}, fo[R + 1] = {}, one[R + 1] = {}, oo[R + 1] = {};
    for (int n = 2; n <= R; ++n) c[n] = 20.0 + 1.2 * n, f[n] = 30.0 - 0.5 * n;
    c[3] += 10.0;  // an outlier
    one[5] = 25.0;
    KfBatch::cost_line(c, 2, R, out);
    KfBatch::cost_line(f, 2, R, fo);
    KfBatch::cost_line(one, 2, R, oo);
    bool ok = true;
    for (int n = 2; n <= R; ++n) ok = ok && std::fabs(out[n] - (20.0 + 1.2 * n)) < 0.05 && fo[n] == fo[2] && oo[n] == 25.0;
    check(ok, "cost line: an outlier at 3 rows leaves 20 + 1.2 n; a falling table is flat; one count: flat");
  }
  const char* tmp = getenv("TMPDIR");
  std::string dir = std::string(tmp && *tmp ? tmp : "/tmp") + "/kf_batch_XXXXXX";
  if (!mkdtemp(dir.data())) return printf("[batch] no temporary folder: FAIL\n"), 1;
  const char* h = getenv("KAPITSA_HOME");
  const std::string old_home = h ? h : "";
  setenv("KAPITSA_HOME", dir.c_str(), 1);
  const int lvl = log_level();
  log_level() = 0;
  auto ms_of = [](int M, int n) { return 20.0 + 3.0 * M + 1.2 * n; };
  // cycles of M members with drafts of confidence c (random below 0: 0.3 .. 0.99), accepting a = k * acc of k
  auto run = [&](KfBatch& kb, KfBudget& gr, int M, int cycles, bool sampled, double c, double acc, double extra,
                 int* seen) {
    for (int i = 0; i < cycles; ++i) {
      KfBatch::Member m[R];
      int k[R], a[R];
      for (int g = 0; g < M; ++g) {
        for (int p = 0; p < D; ++p) m[g].lc[p] = float(std::log(c > 0 ? c : 0.3 + 0.69 * u01(r)));
        m[g].sampled = sampled;
      }
      const int n = kb.plan(m, M, &gr, k);
      if (seen && n >= 1 && n <= R) ++seen[n];
      for (int g = 0; g < M; ++g) a[g] = int(k[g] * acc);
      kb.observe(m, M, k, a, ms_of(M, n) + (n == 3 ? extra : 0.0), &gr);
    }
  };
  float ga[D + 1], gb[D + 1];
  {
    KfBatch kb("bt");
    KfBudget greedy("");
    int seen[R + 1] = {};
    run(kb, greedy, 2, 7 * KfBatch::kExplore, false, -1, 0.5, 0, seen);
    bool all = true;
    for (int n = 2; n <= R; ++n) all = all && seen[n] == KfBatch::kExplore;
    check(all, "2 members: rows 2 .. 8 explored 3 times each");
    run(kb, greedy, 2, 64, false, -1, 0.0, 0, nullptr);  // every draft missed: the greedy calibration learns it
    check(std::fabs(kb.cost(2, 8) - ms_of(2, 8)) < 0.05 && std::fabs(kb.cost(2, 3) - ms_of(2, 3)) < 0.05,
          "2 members: the costs of 3 and 8 rows as measured");
    greedy.calibration(ga, gb, D + 1);
    check(ga[1] < -1.0f && kb.summary().find("by depth (0.00, 1.00)") != std::string::npos,
          "greedy members' records reach the drafter's budget (alpha < -1 at depth 1), not the sampled calibration");
  }  // saved
  {
    bool whole = false;
    try {
      const nlohmann::json j = nlohmann::json::parse(std::ifstream(dir + "/kflash/bt.json"));
      whole = j.value("cost_of", "") == "batched cycle GPU ms" && j.at("cost").at(2).at(8).get<double>() > 0;
    } catch (...) {
    }
    KfBatch kb("bt");
    KfBudget greedy("");
    int seen[R + 1] = {};
    run(kb, greedy, 2, 1, true, -1, 0.0, 0, seen);
    check(whole && seen[R] == 1, "the cache: written whole; read back, the first cycle verifies 8 rows (a count it holds)");
    run(kb, greedy, 2, 64, true, -1, 0.0, 0.0, nullptr);
    greedy.calibration(ga, gb, D + 1);
    check(ga[1] == 0.0f && kb.summary().find("by depth (0.00, 1.00)") == std::string::npos,
          "sampled members' records reach the batch's own calibration, not the drafter's");
  }
  {
    KfBatch kb("");
    KfBudget greedy("");
    run(kb, greedy, 2, 7 * KfBatch::kExplore, false, 0.95, 1.0, 30.0, nullptr);  // 3 rows measured 30 ms too dear
    const double busy = kb.cost(2, 3);
    int seen[R + 1] = {};
    run(kb, greedy, 2, 7 * KfBatch::kRefresh + 64, false, 0.95, 1.0, 0.0, seen);
    bool spread = true;
    for (int n = 2; n <= R; ++n) spread = spread && seen[n] > 0;
    check(busy > ms_of(2, 3) + 20.0 && std::fabs(kb.cost(2, 3) - ms_of(2, 3)) < 6.0 && spread &&
              seen[R] * 10 > 9 * (7 * KfBatch::kRefresh + 64),
          "refresh: every count measured again, 3 rows' cost back from +30 ms; 8 rows in 9 cycles of 10 meanwhile");
  }
  {
    KfBatch kb("", 1);
    KfBudget greedy("");
    bool floor_ok = true;
    for (int i = 0; i < 40; ++i) {
      KfBatch::Member m[4];
      int k[4];
      const int a[4] = {0, 1, 0, 1};
      for (auto& x : m)
        for (float& l : x.lc) l = float(std::log(0.3 + 0.69 * u01(r)));
      const int n = kb.plan(m, 4, &greedy, k);
      for (int g = 0; g < 4; ++g) floor_ok = floor_ok && k[g] >= 1;
      kb.observe(m, 4, k, a, ms_of(4, n), &greedy);
    }
    check(floor_ok, "a floor of one draft: 4 members verify one draft each at least");
  }
  log_level() = lvl;
  unlink((dir + "/kflash/bt.json").c_str());
  rmdir((dir + "/kflash").c_str());
  rmdir(dir.c_str());
  if (h) setenv("KAPITSA_HOME", old_home.c_str(), 1);
  else unsetenv("KAPITSA_HOME");
  return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  // --cpu: the budgets' checks alone (no GPU): the calibration fit, the cache, the costs, the batch budget
  if (argc > 1 && !strcmp(argv[1], "--cpu")) {
    int fails = fit_check(0) + cache_check() + cost_check() + batch_check(0);
    printf(fails ? "FAILED: %d\n" : "ALL PASSED\n", fails);
    return fails ? 1 : 0;
  }
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
        std::sort(c.begin(), c.end(),
            [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
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
  // ---- 3. KFlash tree verifies: kf_build + sample_stage2_rows_tree + kf_accept ----
  if (only < 0 || only >= int(cases.size()))
    fails += tree_cases(trials, salt, only < 0 ? -1 : only - int(cases.size()));
  if (only < 0) fails += fit_check(salt);
  if (only < 0) fails += cache_check();
  if (only < 0) fails += cost_check();
  if (only < 0) fails += batch_check(salt);
  printf(fails ? "FAILED: %d\n" : "ALL PASSED\n", fails);
  return fails ? 1 : 0;
}
