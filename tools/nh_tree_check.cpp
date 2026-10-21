// KFlash tree verify on a Nemotron-H target (src/model/kflash_budget.h), checked against plain decoding:
//  1. rows: for random trees over random tokens, each tree row's logits equal the last row of a chain verify of the
//     row's root path (the path, then filler: 8 rows like the tree)
//  2. acceptance: trees holding the target's own continuation on one branch (a random depth, at a random place among
//     random siblings and subtrees) accept exactly that branch; decoding tree after tree writes plain decoding's
//     tokens, greedy and sampled (T = 0.8, top-k 20, top-p 0.95, a seed: the plain sampler's position-keyed draws);
//     the state afterwards decodes on as plain decoding's does
//  3. the tree builder (the shared kf_build on the lattice's tables) against a CPU reference on random lattices,
//     calibrations and cost tables
//  4. with a drafter (<dir>/dflash): its ring after committing a tree verify's accepted path (encode_commit_path)
//     equals its ring after committing the same tokens verified as a chain; then whole KFlash decoding loops (the
//     drafter's tree under the budget) write plain decoding's tokens, greedy and sampled
//   nh_tree_check <nemotron_h_dir>   (tools/nh_synth.py <dir> --dspark writes a small random one)
#include "metal/gpu.h"
#include "model/kflash_budget.h"
#include "model/nemotron_h.h"
#include "model/nh_drafter.h"
#include "model/lilicorr.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace ni;

namespace {

std::mt19937 rng(7);
int rnd(int n) { return int(rng() % uint32_t(n)); }
int g_fail = 0;

void check(bool ok, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
void check(bool ok, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  printf("%s ", ok ? "PASS" : "FAIL");
  vprintf(fmt, ap);
  printf("\n");
  va_end(ap);
  if (!ok) ++g_fail;
}

constexpr int R8 = NemotronH::kMaxRows;

// the tree buffer for parents par[0 .. n) (par[0] = -1), and the verify's live rows
void write_tree(NemotronH& m, const std::vector<int>& par) {
  int32_t* t = m.tree_buf().as<int32_t>();
  const int n = int(par.size());
  int dep[8] = {}, mask[8] = {};
  for (int r = 0; r < R8; ++r) {
    const int p = r < n ? par[size_t(r)] : r - 1;
    dep[r] = r == 0 ? 0 : dep[p] + 1;
    mask[r] = (r == 0 ? 0 : mask[p]) | (1 << r);
    t[kf::kPar + r] = r == 0 ? -1 : p;
    t[kf::kDep + r] = dep[r];
    t[kf::kMask + r] = mask[r];
    t[kf::kLq + r] = 0;
  }
  t[kf::kN] = n;
  m.live_rows().as<uint32_t>()[0] = uint32_t(n);
}

std::vector<int> depths(const std::vector<int>& par) {
  std::vector<int> d(par.size(), 0);
  for (size_t r = 1; r < par.size(); ++r) d[r] = d[size_t(par[r])] + 1;
  return d;
}

void prefill(NemotronH& m, Seq& s, const std::vector<int>& prompt, const SamplingParams& sp) {
  memcpy(s.tokens(), prompt.data(), prompt.size() * 4);
  auto e = Gpu::get().begin();
  m.encode_prefill(*e, s, 0, int(prompt.size()), sp, 0, true);
  e->commit_and_wait();
}

// plain decoding of n tokens after position P (tokens[P] is fed first): tokens[P + 1 .. P + n]
std::vector<int> plain(NemotronH& m, Seq& s, int P, int n, const SamplingParams& sp) {
  for (int i = 0; i < n; ++i) {
    auto e = Gpu::get().begin();
    m.encode_step(*e, s, P + i, sp, uint32_t(P + i), true);
    e->commit_and_wait();
  }
  return std::vector<int>(s.tokens() + P + 1, s.tokens() + P + 1 + n);
}

int tree_cycle(NemotronH& m, Seq& s, int P, const struct Tree& t, const SamplingParams& sp, NhDrafter* dr);

// plain decoding through the verify pass (one-row trees: the row-0 sample each cycle), so its logits come from the
// same multi-row products as every verify's (a one-row decode step uses other kernels, whose rounding can move a
// sampled token)
std::vector<int> plain_verify(NemotronH& m, Seq& s, int P, int n, const SamplingParams& sp);

// a random tree of n rows whose branch 0 -> ... carries want[0 .. k) at depths 1 .. k (the target's continuation),
// placed among random siblings; tokens of the other nodes random and different from their siblings'
struct Tree {
  std::vector<int> par, tok;  // rows (tok[0]: the anchor, not written)
  int expect = 0;             // the drafts the acceptance must keep
};
Tree oracle_tree(int n, const std::vector<int>& want, int k, int V) {
  Tree t;
  t.par = {-1};
  t.tok = {-1};
  std::vector<int> spine = {0};  // rows of the oracle branch by depth
  // the oracle branch's rows are interleaved with others: build in insertion order
  while (int(t.par.size()) < n) {
    const int r = int(t.par.size());
    const int sd = int(spine.size()) - 1;  // depth of the spine's tip
    const bool grow = sd < k && (rnd(3) != 0 || n - r <= k - sd);
    if (grow) {
      t.par.push_back(spine.back());
      t.tok.push_back(want[size_t(sd)]);
      spine.push_back(r);
      continue;
    }
    const int p = rnd(r);
    // a token unlike p's other children and unlike the continuation at p's depth (so the expected path is the spine)
    const std::vector<int> dep = depths(t.par);
    int tok;
    for (;;) {
      tok = 3 + rnd(V - 3);
      bool clash = dep[size_t(p)] < int(want.size()) && tok == want[size_t(dep[size_t(p)])];
      for (int c = 1; c < r; ++c) clash |= t.par[size_t(c)] == p && t.tok[size_t(c)] == tok;
      if (!clash) break;
    }
    t.par.push_back(p);
    t.tok.push_back(tok);
  }
  t.expect = int(spine.size()) - 1;
  return t;
}

void put_tokens(Seq& s, int P, const Tree& t) {
  for (size_t r = 1; r < t.tok.size(); ++r) s.tokens()[P + int(r)] = t.tok[r];
  for (int r = int(t.tok.size()); r < R8; ++r) s.tokens()[P + r] = 5;
}

// one tree cycle at P: verify, accept, commit; returns the accepted drafts
int tree_cycle(NemotronH& m, Seq& s, int P, const Tree& t, const SamplingParams& sp, NhDrafter* dr) {
  write_tree(m, t.par);
  put_tokens(s, P, t);
  auto e = Gpu::get().begin();
  if (dr) dr->arm_verify(P);
  m.set_dynamic_rows(true);
  m.set_tree(true);
  m.encode_verify(*e, s, P, R8);
  if (dr) dr->disarm();
  m.encode_accept_tree(*e, s, P, sp);
  m.set_tree(false);
  m.set_dynamic_rows(false);
  e->commit_and_wait();
  const int a = m.accepted();
  m.commit_rows_tree(s, a + 1);
  return a;
}

std::vector<int> plain_verify(NemotronH& m, Seq& s, int P, int n, const SamplingParams& sp) {
  Tree root;
  root.par = {-1};
  root.tok = {-1};
  for (int i = 0; i < n; ++i) tree_cycle(m, s, P + i, root, sp, nullptr);
  return std::vector<int>(s.tokens() + P + 1, s.tokens() + P + 1 + n);
}

// ---- 3. the builder's CPU reference ----
float softplus_f(float x) { return x > 20.0f ? x : std::log(1.0f + std::exp(x)); }
float lcal(float lq, float a, float b) {
  const float q = std::min(std::exp(lq), 0.999999f);
  const float z = a + b * (lq - std::log(1.0f - q));
  return -softplus_f(-z);
}
struct RefTree {
  int n = 1, m = 1;
  int par[8], dep[8], tok[8];
  float score[8];
};
RefTree build_ref(const std::vector<float>& tlq, const std::vector<int>& cand, const KfParams& kp, int S, int K, int P,
                  const int* anchor_tok, float* min_gap) {
  (void)P;
  (void)anchor_tok;
  struct F {
    float s, lq;
    int par, slot, c;
    bool taken;
  };
  std::vector<F> f;
  for (int c = 0; c < K; ++c) f.push_back({lcal(tlq[size_t(c)], kp.alpha[1], kp.beta[1]), tlq[size_t(c)], 0, 0, c, false});
  RefTree t;
  t.par[0] = -1;
  t.dep[0] = 0;
  t.score[0] = 0;
  *min_gap = 1e9f;
  for (int r = 1; r < 8; ++r) {
    int bi = -1;
    float second = -INFINITY;
    for (int i = 0; i < int(f.size()); ++i) {
      if (f[size_t(i)].taken) continue;
      if (bi < 0 || f[size_t(i)].s > f[size_t(bi)].s) {
        if (bi >= 0) second = std::max(second, f[size_t(bi)].s);
        bi = i;
      } else {
        second = std::max(second, f[size_t(i)].s);
      }
    }
    if (bi < 0) break;
    if (second > -INFINITY) *min_gap = std::min(*min_gap, f[size_t(bi)].s - second);
    f[size_t(bi)].taken = true;
    const F b = f[size_t(bi)];  // (a copy: the pushes below may move f)
    t.par[r] = b.par;
    t.dep[r] = b.slot + 1;
    t.tok[r] = cand[size_t(b.slot * K + b.c)];
    t.score[r] = b.s;
    t.m = r + 1;
    if (b.slot + 1 < S)
      for (int c2 = 0; c2 < K; ++c2) {
        const float lq = tlq[size_t(((b.slot + 1) * K + b.c) * K + c2)];
        f.push_back({b.s + lcal(lq, kp.alpha[b.slot + 2], kp.beta[b.slot + 2]), lq, r, b.slot + 1, c2, false});
      }
  }
  if (kp.force_n > 0) {
    t.n = std::min(kp.force_n, t.m);
  } else {  // E(n) - rate cost[n] (rate 0: E(n) / cost[n])
    auto val = [&](float E, int n) { return kp.rate > 0.0f ? E - kp.rate * kp.cost[n] : E / kp.cost[n]; };
    float E = 1.0f, best = val(E, 1);
    t.n = 1;
    for (int r = 1; r < t.m; ++r) {
      E += std::exp(t.score[r]);
      const float v = val(E, r + 1);
      if (v > best) best = v, t.n = r + 1;
    }
  }
  return t;
}

void test_builder(NemotronH& m, Seq& s) {
  Gpu& g = Gpu::get();
  const int S = 7;
  int ok = 0, ties = 0, bad = 0;
  for (int trial = 0; trial < 300; ++trial) {
    const int K = trial % 3 == 0 ? 8 : trial % 3 == 1 ? 16 : 32;
    std::vector<float> tlq(size_t(S * K * K));
    std::vector<int> cand(size_t(S * K));
    std::normal_distribution<float> nd(0.0f, 2.0f + float(trial % 5));
    for (int sl = 0; sl < S; ++sl) {
      for (int c = 0; c < K; ++c) cand[size_t(sl * K + c)] = 100 + sl * 1000 + c;
      for (int p = 0; p < K; ++p) {
        float lg[32], mx = -INFINITY, den = 0.0f;
        for (int c = 0; c < K; ++c) mx = std::max(mx, lg[c] = nd(rng));
        for (int c = 0; c < K; ++c) den += std::exp(lg[c] - mx);
        const float out = 0.05f * float(rnd(10));  // the mass outside the candidates
        for (int c = 0; c < K; ++c)
          tlq[size_t((sl * K + p) * K + c)] = lg[c] - mx - std::log(den) + std::log(1.0f - out);
      }
    }
    KfParams kp{};
    for (int d = 0; d < 8; ++d) {
      kp.alpha[d] = trial < 50 ? 0.0f : -1.0f + 0.02f * float(rnd(100));
      kp.beta[d] = trial < 50 ? 1.0f : 0.5f + 0.015f * float(rnd(100));
    }
    kp.cost[0] = 1.0f;
    for (int n = 1; n <= 8; ++n) kp.cost[n] = trial % 4 == 0 ? 1.0f : 1.0f + 0.03f * float(n - 1) * float(1 + trial % 7);
    kp.force_n = trial % 10 == 9 ? 1 + rnd(8) : 0;
    kp.rate = trial % 2 ? 0.0f : 0.1f + 0.01f * float(rnd(40));  // the rate rule on even trials
    kp.allowed = 0xffu;  // every row count 1 .. 8 (a MoE target's rows cost experts)
    kp.max_rows = R8;
    // the shared builder's children tables: tokens per children-row, the lattice's constant successors
    std::vector<int> ctok(size_t(S * K * K));
    for (int sl = 0; sl < S; ++sl)
      for (int pp = 0; pp < K; ++pp)
        for (int c = 0; c < K; ++c) ctok[size_t((sl * K + pp) * K + c)] = cand[size_t(sl * K + c)];
    const std::vector<int32_t> next = kf::lattice_next(S, K);
    Buffer bt = g.alloc(tlq.size() * 4, "t_tlq"), bc = g.alloc(ctok.size() * 4, "t_ctok"),
           bn = g.alloc(next.size() * 4, "t_next"), bk = g.alloc(sizeof(KfParams), "t_kp");
    memcpy(bt.ptr(), tlq.data(), tlq.size() * 4);
    memcpy(bc.ptr(), ctok.data(), ctok.size() * 4);
    memcpy(bn.ptr(), next.data(), next.size() * 4);
    memcpy(bk.ptr(), &kp, sizeof kp);
    const int P = 20;
    const struct {
      uint32_t K, P, root, pad;
    } bp{uint32_t(K), uint32_t(P), 0u, 0u};
    auto e = g.begin();
    e->use(NemotronH::pipeline("kf_build"));
    e->buf(0, bt);
    e->buf(1, bc);
    e->buf(2, bn);
    e->buf(3, bk);
    e->buf(4, m.tree_buf());
    e->buf(5, s.tok);
    e->buf(6, m.live_rows());
    e->bytes(7, &bp, sizeof bp);
    e->dispatch(1, 1, 1, 32);
    e->commit_and_wait();
    float gap;
    const RefTree ref = build_ref(tlq, cand, kp, S, K, P, nullptr, &gap);
    const int32_t* t = m.tree_buf().as<int32_t>();
    bool same = t[kf::kN] == ref.n && int(m.live_rows().as<uint32_t>()[0]) == ref.n;
    for (int r = 1; r < ref.m && same; ++r)
      same = t[kf::kPar + r] == ref.par[r] && t[kf::kDep + r] == ref.dep[r] && s.tokens()[P + r] == ref.tok[r];
    for (int r = 0; r < R8 && same; ++r) {
      const int p = t[kf::kPar + r];
      same = r == 0 ? (p == -1 && t[kf::kMask] == 1) : (p >= 0 && p < r && t[kf::kMask + r] == (t[kf::kMask + p] | (1 << r)));
    }
    if (same) ++ok;
    else if (gap < 1e-4f) ++ties;
    else ++bad;
    if (!same && gap >= 1e-4f && bad <= 3) {
      printf("  builder mismatch (trial %d, K %d, force %d): GPU n %d rows", trial, K, kp.force_n, t[kf::kN]);
      for (int r = 1; r < R8; ++r) printf(" %d:%d/%d/%d", r, t[kf::kPar + r], t[kf::kDep + r], s.tokens()[P + r]);
      printf("\n  CPU n %d m %d rows", ref.n, ref.m);
      for (int r = 1; r < ref.m; ++r) printf(" %d:%d/%d/%d(%.3f)", r, ref.par[r], ref.dep[r], ref.tok[r], ref.score[r]);
      printf("\n");
    }
    g.release(bt);
    g.release(bc);
    g.release(bn);
    g.release(bk);
  }
  check(bad == 0, "builder: %d of 300 random lattices identical to the CPU reference, %d differ at near-ties (< 1e-4), %d differ",
        ok, ties, bad);
}

// the drafter's whole context (ring K/V, anchor slots) of two sequences: identical?
// ---- the drafters' lattice tables against CPU references on random inputs ----
float ue4m3(uint8_t b) {  // nh_ue4m3: UE4M3 times 0.5 (the NVFP4 lut is doubled)
  if (b == 0x7F) return 0.0f;
  const uint16_t h = uint16_t(b) << 7;
  __fp16 v;
  memcpy(&v, &h, 2);
  return float(v) * 128.0f;
}
const float kLut[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
float bf16f(uint16_t b) {
  const uint32_t u = uint32_t(b) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

void test_tables() {
  Gpu& g = Gpu::get();
  const int S = 7, V = 512, R = 512;
  double worst_d = 0, worst_l = 0;
  for (int trial = 0; trial < 6; ++trial) {
    const int K = trial % 2 ? 16 : 8;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<int> cand(size_t(S * K));
    std::vector<float> clp(size_t(S * K));
    for (int sl = 0; sl < S; ++sl)
      for (int c = 0; c < K; ++c) {
        cand[size_t(sl * K + c)] = (sl * 37 + c * 11 + trial) % V;
        clp[size_t(sl * K + c)] = -1.0f - 0.3f * float(c) - 0.1f * float(rnd(10));
      }
    // DSpark: W1 BF16 [V][R], W2 NVFP4 [V][R / 2] codes, [V][R / 16] UE4M3 scales
    std::vector<uint16_t> w1(size_t(V) * R);
    std::vector<uint8_t> w2q(size_t(V) * R / 2), w2s(size_t(V) * R / 16);
    for (auto& x : w1) {
      const float f = 0.05f * nd(rng);
      uint32_t u;
      memcpy(&u, &f, 4);
      x = uint16_t(u >> 16);
    }
    for (auto& x : w2q) x = uint8_t(rnd(256));
    for (auto& x : w2s) x = uint8_t(0x30 + rnd(16));
    const float gs = 0.02f;
    const int P = 3;
    std::vector<int> toks(16, 0);
    toks[P] = rnd(V);
    Buffer bc = g.alloc(cand.size() * 4, "t_c"), bl = g.alloc(clp.size() * 4, "t_l"), b1 = g.alloc(w1.size() * 2, "t_w1"),
           b2 = g.alloc(w2q.size(), "t_w2"), bs = g.alloc(w2s.size(), "t_ws"), bt = g.alloc(64, "t_tok"),
           bo = g.alloc(size_t(S * K * K) * 4, "t_tlq");
    memcpy(bc.ptr(), cand.data(), cand.size() * 4);
    memcpy(bl.ptr(), clp.data(), clp.size() * 4);
    memcpy(b1.ptr(), w1.data(), w1.size() * 2);
    memcpy(b2.ptr(), w2q.data(), w2q.size());
    memcpy(bs.ptr(), w2s.data(), w2s.size());
    memcpy(bt.ptr(), toks.data(), 64);
    const struct {
      uint32_t S, K, R, P;
      float gs;
    } tp{uint32_t(S), uint32_t(K), uint32_t(R), uint32_t(P), gs};
    {
      auto e = g.begin();
      e->use(NemotronH::pipeline("nh_dsp_table"));
      e->buf(0, bc);
      e->buf(1, bl);
      e->buf(2, b1);
      e->buf(3, b2);
      e->buf(4, bs);
      e->buf(5, bt);
      e->buf(6, bo);
      e->bytes(7, &tp, sizeof tp);
      e->dispatch(uint32_t(1 + (S - 1) * K), 1, 1, uint32_t(32 * K));
      e->commit_and_wait();
    }
    const float* got = bo.as<float>();
    for (int sl = 0; sl < S; ++sl)
      for (int pp = 0; pp < (sl == 0 ? 1 : K); ++pp) {
        const int prev = sl == 0 ? toks[P] : cand[size_t((sl - 1) * K + pp)];
        double sc[32], mx = -1e30, den = 0, base = 0;
        for (int c = 0; c < K; ++c) {
          const int id = cand[size_t(sl * K + c)];
          double d = 0;
          for (int b = 0; b < R / 16; ++b) {
            double acc = 0;
            for (int j = 0; j < 8; ++j) {
              const uint8_t byte = w2q[size_t(id) * (R / 2) + size_t(b) * 8 + size_t(j)];
              acc += kLut[byte & 15] * bf16f(w1[size_t(prev) * R + size_t(b) * 16 + size_t(j)]) +
                     kLut[byte >> 4] * bf16f(w1[size_t(prev) * R + size_t(b) * 16 + size_t(j) + 8]);
            }
            d += acc * ue4m3(w2s[size_t(id) * (R / 16) + size_t(b)]);
          }
          sc[c] = clp[size_t(sl * K + c)] + d * gs;
          mx = std::max(mx, sc[c]);
        }
        for (int c = 0; c < K; ++c) den += std::exp(sc[c] - mx), base += std::exp(double(clp[size_t(sl * K + c)]));
        den += std::max(0.0, 1.0 - base) * std::exp(-mx);
        const double lden = mx + std::log(den);
        for (int c = 0; c < K; ++c)
          worst_d = std::max(worst_d, std::fabs(double(got[(sl * K + pp) * K + c]) - (sc[c] - lden)));
      }
    // LiLiCorr: in / out vectors [S * K][D], the anchor's [D]
    const int D = 256;
    const float ls = 8.0f;
    std::vector<float> in(static_cast<size_t>(S * K * D)), outv(static_cast<size_t>(S * K * D)),
        aout(static_cast<size_t>(D));
    for (auto& x : in) x = 0.06f * nd(rng);
    for (auto& x : outv) x = 0.06f * nd(rng);
    for (auto& x : aout) x = 0.06f * nd(rng);
    Buffer bi = g.alloc(in.size() * 4, "t_in"), bv = g.alloc(outv.size() * 4, "t_out"), ba = g.alloc(aout.size() * 4, "t_a");
    memcpy(bi.ptr(), in.data(), in.size() * 4);
    memcpy(bv.ptr(), outv.data(), outv.size() * 4);
    memcpy(ba.ptr(), aout.data(), aout.size() * 4);
    const struct {
      uint32_t S, K, D;
      float ls;
    } lp{uint32_t(S), uint32_t(K), uint32_t(D), ls};
    {
      auto e = g.begin();
      e->use(NemotronH::pipeline("nh_lili_table"));
      e->buf(0, bi);
      e->buf(1, bv);
      e->buf(2, ba);
      e->buf(3, bl);
      e->buf(4, bo);
      e->bytes(5, &lp, sizeof lp);
      e->dispatch(uint32_t(1 + (S - 1) * K), 1, 1, uint32_t(32 * K));
      e->commit_and_wait();
    }
    for (int sl = 0; sl < S; ++sl)
      for (int pp = 0; pp < (sl == 0 ? 1 : K); ++pp) {
        double sc[32], mx = -1e30, den = 0, mass = 0;
        for (int c = 0; c < K; ++c) {
          const float* a = in.data() + size_t(sl * K + c) * D;
          const float* b = sl == 0 ? aout.data() : outv.data() + size_t((sl - 1) * K + pp) * D;
          double d = 0;
          for (int o = 0; o < D; ++o) d += double(a[o]) * b[o];
          sc[c] = ls * d;
          mx = std::max(mx, sc[c]);
        }
        for (int c = 0; c < K; ++c) den += std::exp(sc[c] - mx), mass += std::exp(double(clp[size_t(sl * K + c)]));
        const double l = mx + std::log(den) - std::log(std::min(1.0, mass));
        for (int c = 0; c < K; ++c)
          worst_l = std::max(worst_l, std::fabs(double(got[(sl * K + pp) * K + c]) - (sc[c] - l)));
      }
    for (Buffer* b : {&bc, &bl, &b1, &b2, &bs, &bt, &bo, &bi, &bv, &ba}) g.release(*b);
  }
  check(worst_d < 1e-3 && worst_l < 1e-3, "lattice tables: DSpark (Markov W1 . W2) max |log q difference| %.2g, LiLiCorr "
        "(transitions) %.2g, against CPU references", worst_d, worst_l);
}

bool same_ring(const LiliSeq& a, const LiliSeq& b, double* maxd) {
  bool same = true;
  *maxd = 0;
  for (const Buffer* x : {&a.ring_k, &a.ring_v}) {
    const Buffer& y = x == &a.ring_k ? b.ring_k : b.ring_v;
    const __fp16* p = x->as<__fp16>();
    const __fp16* q = y.as<__fp16>();
    for (size_t i = 0; i < x->size / 2; ++i) *maxd = std::max(*maxd, std::fabs(double(p[i]) - double(q[i])));
    same &= memcmp(x->ptr(), y.ptr(), x->size) == 0;
  }
  if (a.anchor && b.anchor) {
    const float* p = a.anchor.as<float>();
    const float* q = b.anchor.as<float>();
    for (size_t i = 0; i < a.anchor.size / 4; ++i) *maxd = std::max(*maxd, std::fabs(double(p[i]) - double(q[i])));
    same &= memcmp(a.anchor.ptr(), b.anchor.ptr(), a.anchor.size) == 0;
  }
  return same;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: nh_tree_check <nemotron_h_dir>\n");
    return 2;
  }
  const std::string dir = argv[1];
  Gpu& g = Gpu::get();
  NemotronH m(dir, 64);
  const int V = m.vocab();
  SamplingParams greedy;
  SamplingParams sampled;
  sampled.temperature = 0.8f;
  sampled.top_k = 20;
  sampled.top_p = 0.95f;
  sampled.seed = 1234;
  std::vector<int> prompt(40);
  for (int& t : prompt) t = 3 + rnd(V - 3);
  const int P0 = int(prompt.size());

  // ---- 1. tree rows against chain verifies of their root paths ----
  {
    auto A = m.new_seq(512), B = m.new_seq(512);
    prefill(m, *A, prompt, greedy);
    prefill(m, *B, prompt, greedy);
    int rows = 0, bitwise = 0;
    double maxd = 0, maxrel = 0;
    for (int trial = 0; trial < 40; ++trial) {
      const int n = 2 + rnd(7);
      Tree t;
      t.par = {-1};
      t.tok = {-1};
      for (int r = 1; r < n; ++r) {
        t.par.push_back(rnd(r));
        t.tok.push_back(3 + rnd(V - 3));
      }
      A->n_pend = 0;
      A->pend_tree = false;
      write_tree(m, t.par);
      put_tokens(*A, P0, t);
      {
        auto e = g.begin();
        m.set_dynamic_rows(true);
        m.set_tree(true);
        m.encode_verify(*e, *A, P0, R8);
        m.set_tree(false);
        m.set_dynamic_rows(false);
        e->commit_and_wait();
      }
      std::vector<float> tl(m.logits(), m.logits() + size_t(n) * size_t(V));
      const std::vector<int> dep = depths(t.par);
      for (int r = 0; r < n; ++r) {
        std::vector<int> path;
        for (int x = r; x > 0; x = t.par[size_t(x)]) path.push_back(x);
        std::reverse(path.begin(), path.end());
        for (int j = 0; j < R8 - 1; ++j) B->tokens()[P0 + 1 + j] = j < int(path.size()) ? t.tok[size_t(path[size_t(j)])] : 7;
        B->n_pend = 0;
        auto e = g.begin();
        m.encode_verify(*e, *B, P0, R8);
        e->commit_and_wait();
        const float* cl = m.logits() + size_t(dep[size_t(r)]) * size_t(V);
        const float* tr = tl.data() + size_t(r) * size_t(V);
        double d = 0, scale = 0;
        for (int v = 0; v < V; ++v) d = std::max(d, double(std::fabs(cl[v] - tr[v]))), scale = std::max(scale, double(std::fabs(cl[v])));
        maxd = std::max(maxd, d);
        maxrel = std::max(maxrel, d / std::max(scale, 1e-6));
        bitwise += memcmp(cl, tr, size_t(V) * 4) == 0;
        ++rows;
      }
    }
    check(maxrel < 1e-4, "rows: %d tree rows of 40 random trees against chain verifies of their paths: %d bit-identical, "
          "max |difference| %.3g (%.3g of the largest logit)", rows, bitwise, maxd, maxrel);
  }

  // ---- 2. acceptance and decoding against plain decoding ----
  for (int mode = 0; mode < 2; ++mode) {
    const SamplingParams& sp = mode ? sampled : greedy;
    auto A = m.new_seq(512), B = m.new_seq(512);
    prefill(m, *A, prompt, sp);
    prefill(m, *B, prompt, sp);
    const int N = 120;
    // greedy: plain one-row decoding; sampled: plain decoding through the verify pass (the same products)
    const std::vector<int> ref = mode ? plain_verify(m, *B, P0, N + 8, sp) : plain(m, *B, P0, N + 8, sp);
    int P = P0, cycles = 0, wrong_a = 0, wrong_tok = 0, emitted = 0;
    std::vector<int> want(8);
    while (P < P0 + N) {
      const int n = 1 + rnd(8);
      for (int j = 0; j < 8; ++j) want[size_t(j)] = ref[size_t(P - P0 + j)];
      const int k = std::min(n - 1, rnd(8));
      const Tree t = oracle_tree(n, want, k, V);
      const int a = tree_cycle(m, *A, P, t, sp, nullptr);
      wrong_a += a != t.expect;
      for (int j = 1; j <= a + 1; ++j) wrong_tok += A->tokens()[P + j] != ref[size_t(P - P0 + j - 1)];
      emitted += a + 1;
      P += a + 1;
      ++cycles;
    }
    // the state after the trees: flush, then plain decoding continues as the reference's
    const std::vector<int> more = mode ? plain_verify(m, *A, P, 4, sp) : plain(m, *A, P, 4, sp);
    int wrong_more = 0;
    for (int j = 0; j < 4; ++j) wrong_more += more[size_t(j)] != ref[size_t(P - P0 + j)];
    check(wrong_a == 0 && wrong_tok == 0 && wrong_more == 0,
          "acceptance (%s): %d tree cycles, %d tokens: %d cycles accepted other than the planted branch, %d tokens "
          "differ from plain decoding, %d of the 4 plain tokens after them differ",
          mode ? "sampled T 0.8" : "greedy", cycles, emitted, wrong_a, wrong_tok, wrong_more);
  }

  // ---- 3. the builder, and the budget's calibration fit (records drawn from a known calibration) ----
  {
    auto A = m.new_seq(512);
    test_builder(m, *A);
    test_tables();
    std::vector<float> x;
    std::vector<uint8_t> y;
    const float ta = -0.6f, tb = 1.4f;
    std::uniform_real_distribution<float> ul(-6.0f, -0.01f), u01(0.0f, 1.0f);
    for (int i = 0; i < 2048; ++i) {
      const float lq = ul(rng), xi = KfBudget::logit_q(lq);
      x.push_back(xi);
      y.push_back(u01(rng) < 1.0f / (1.0f + std::exp(-(ta + tb * xi))));
    }
    float a = 0.0f, b = 1.0f;
    KfBudget::fit(x, y, a, b, 8);
    check(std::fabs(a - ta) < 0.25f && std::fabs(b - tb) < 0.2f,
          "calibration fit: alpha %.2f beta %.2f from 2048 records drawn with alpha %.2f beta %.2f", a, b, ta, tb);
  }

  // ---- 4. a drafter: path commits, then whole KFlash loops ----
  const std::string ddir = NhDrafter::find(dir);
  if (ddir.empty()) {
    printf("(no drafter in %s/dflash: tests 4 skipped)\n", dir.c_str());
  } else {
    auto dr = NhDrafter::open(m, ddir);
    auto A = m.new_seq(512), B = m.new_seq(512), C = m.new_seq(512);
    auto d1 = dr->new_seq(), d2 = dr->new_seq();
    NhDrafter::reset(*d1);
    NhDrafter::reset(*d2);
    memcpy(A->tokens(), prompt.data(), prompt.size() * 4);
    {
      dr->begin_prefill(P0, 0);
      auto e = g.begin();
      m.encode_prefill(*e, *A, 0, P0, greedy, 0, true);
      dr->encode_prefill_commit(*e, *d1);
      e->commit_and_wait();
      dr->end_prefill();
      auto e2 = g.begin();
      dr->encode_copy(*e2, *d1, *d2);
      e2->commit_and_wait();
      d2->ctx = d1->ctx;
      d2->hi = d1->hi;
    }
    prefill(m, *B, prompt, greedy);
    prefill(m, *C, prompt, greedy);
    const std::vector<int> ref = plain(m, *C, P0, 16, greedy);
    std::vector<int> want(ref.begin(), ref.begin() + 8);
    int same = 0, trials = 0;
    double worst = 0;
    int P = P0;
    for (int trial = 0; trial < 6; ++trial) {
      // a tree whose planted branch is 1-6 deep, not first among its siblings
      Tree t = oracle_tree(8, want, 1 + trial % 6, V);
      const int a = tree_cycle(m, *A, P, t, greedy, dr.get());
      {
        auto e = g.begin();
        dr->encode_commit_path(*e, *d1, a + 1, A->pend_path);
        e->commit_and_wait();
      }
      // the same tokens as a chain on B, committed as rows 0 .. a
      B->tokens()[P] = A->tokens()[P];  // (the anchor: the last cycle's bonus token)
      for (int j = 0; j < R8 - 1; ++j) B->tokens()[P + 1 + j] = j < a ? A->tokens()[P + 1 + j] : 7;
      {
        auto e = g.begin();
        dr->arm_verify(P);
        m.encode_verify(*e, *B, P, R8);
        dr->disarm();
        dr->encode_commit(*e, *d2, a + 1);
        e->commit_and_wait();
      }
      double md;
      same += same_ring(*d1, *d2, &md);
      worst = std::max(worst, md);
      ++trials;
      // both back to the same state: the tree's commit stays (B redoes it as a chain); next trial from P + a + 1
      m.commit_rows(*B, a + 1);
      P += a + 1;
      for (int j = 0; j < 8; ++j) want[size_t(j)] = j + P - P0 < int(ref.size()) ? ref[size_t(P - P0 + j)] : 5;
      if (P - P0 + 8 > int(ref.size())) break;
    }
    check(same == trials, "drafter %s: ring after %d tree path commits identical to the chain commits in %d (max |difference| %.3g)",
          dr->name(), trials, same, worst);

    // whole loops: plain, the walk's chain, the KFlash tree under the budget
    for (int mode = 0; mode < 2; ++mode) {
      const SamplingParams& sp = mode ? sampled : greedy;
      const int N = 96;
      auto R = m.new_seq(512);
      prefill(m, *R, prompt, sp);
      const std::vector<int> pref = mode ? plain_verify(m, *R, P0, N + 8, sp) : plain(m, *R, P0, N + 8, sp);
      for (int tree = 0; tree < 2; ++tree) {
        auto S = m.new_seq(512);
        auto d = dr->new_seq();
        NhDrafter::reset(*d);
        memcpy(S->tokens(), prompt.data(), prompt.size() * 4);
        dr->begin_prefill(P0, 0);
        {
          auto e = g.begin();
          m.encode_prefill(*e, *S, 0, P0, sp, 0, true);
          dr->encode_prefill_commit(*e, *d);
          e->commit_and_wait();
        }
        dr->end_prefill();
        KfBudget budget;
        Buffer kp = g.alloc(sizeof(KfParams), "t_kfp");
        int Pq = P0, mprev = 0, cycles = 0;
        while (Pq < P0 + N) {
          auto e1 = g.begin();
          if (mprev > 0) tree ? dr->encode_commit_path(*e1, *d, mprev, S->pend_path) : dr->encode_commit(*e1, *d, mprev);
          if (tree) {
            budget.write(*kp.as<KfParams>());
            dr->encode_draft_tree(*e1, *S, *d, Pq, kp);
          } else {
            dr->encode_draft(*e1, *S, *d, Pq);
          }
          e1->commit();
          auto e2 = g.begin();
          dr->arm_verify(Pq);
          m.set_dynamic_rows(true);
          m.set_tree(tree != 0);
          m.encode_verify(*e2, *S, Pq, R8);
          dr->disarm();
          if (tree) m.encode_accept_tree(*e2, *S, Pq, sp);
          else m.encode_accept(*e2, *S, Pq, R8, R8 - 1, sp);
          m.set_tree(false);
          m.set_dynamic_rows(false);
          e2->commit_and_wait();
          if (tree) budget.observe(m.tree_buf().as<int32_t>(), 1.0);
          const int a = m.accepted();
          if (tree) m.commit_rows_tree(*S, a + 1);
          else m.commit_rows(*S, a + 1);
          mprev = a + 1;
          Pq += a + 1;
          ++cycles;
        }
        int wrong = 0;
        for (int j = 0; j < N; ++j) wrong += S->tokens()[P0 + 1 + j] != pref[size_t(j)];
        check(wrong == 0, "%s loop (%s): %d tokens in %d cycles, %d differ from plain decoding",
              tree ? "KFlash tree" : "chain", mode ? "sampled" : "greedy", N, cycles, wrong);
        g.release(kp);
      }
    }
  }
  printf("%s\n", g_fail ? "FAILED" : "all passed");
  return g_fail ? 1 : 0;
}
