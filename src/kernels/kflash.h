// KFlash: the one tree descriptor, tree builder and tree acceptance that every backend shares (Qwen's DFlash2 /
// DFlash drafters, Nemotron-H's LiLiCorr / DSpark); only the verify kernels differ per backend. Both Metal libraries
// embed this file after common.h (cmake/embed_kernels*.cmake) and the host includes it (src/model/kflash_budget.h):
// the layout and KfParams are read on both sides, the kernels are Metal only.
//
// The tree descriptor (int32 words; floats stored as their bits). Rows are the tree's nodes in insertion order: row
// 0 is the root (the anchor, tokens[P]), every row's parent comes before it, row r's token is tokens[P + r] and it
// sits at position P + depth(r). The first n rows are the verify's live rows; the rows past them, up to the verify's
// capacity, are a well-formed filler chain (never live). Rows are best first: the first n rows are the best tree of n
// rows under the builder's model, for every n.
//   [kKfN]           live rows n (the size rule's choice)
//   [kKfPar + r]     parent row (-1: the root)
//   [kKfDep + r]     depth (the root: 0)
//   [kKfMask + r]    bit j set: row j is r or one of its ancestors
//   [kKfLq + r]      the path model's log-probability of the node given its parent (float bits; the root: 0)
//   [kKfA]           accepted drafts a (the acceptance)
//   [kKfPath + i]    the accepted path's rows, i = 0 .. a (path[0] = 0)
//   [kKfNrec]        calibration records: the nodes whose parent is on the accepted path
//   [kKfRecDep + i], [kKfRecLq + i], [kKfRecHit + i]: their depth, log-probability (float bits), 1 if accepted
//   [kKfScore + r]   the calibrated log-probability of the node's whole path (float bits; the root: 0)
//   [kKfE + n]       E(n) = 1 + the summed calibrated path probabilities of rows 1 .. n - 1 (float bits): the tokens a
//                    verify of n rows is expected to emit (the accepted path plus one), n = 1 .. nodes built
//   [kKfJump]        bit r set: row r's parent is not row r - 1 (a verify may replay such a row's ancestry)
//   [kKfNodes]       rows built m (1 + the nodes taken), n <= m
//   [kKfCap]         the verify's row capacity this cycle (KfParams::cap, else max_rows): a backend that verifies
//                    counts past 8 as one wide verify may verify an 8-row tree of a wider build with its 8-row verify
//   [kKfObs + i]     what the budget reads of the cycle the acceptance walked, copied by kf_accept (in a double-
//                    buffered loop the next cycle's builder may rewrite the words above before the host reads them):
//                    i = 0 its live rows n, 1 its E(n) (float bits), 2 its capacity, 3 the rows built, 4 .. 6 E(8),
//                    E(16), E(32) (float bits; 0 past the rows built)
#ifdef __METAL_VERSION__
#define KF_WORD(name, v) constant constexpr uint kKf##name = v
typedef uint kf_uint;
typedef int kf_int;
#else
#include <cstdint>
#include <vector>
#define KF_WORD(name, v) constexpr int k##name = v
typedef uint32_t kf_uint;
typedef int32_t kf_int;
namespace ni::kf {
#endif
KF_WORD(MaxRows, 32);
KF_WORD(N, 0);
KF_WORD(Par, 1);
KF_WORD(Dep, 33);
KF_WORD(Mask, 65);
KF_WORD(Lq, 97);
KF_WORD(A, 129);
KF_WORD(Path, 130);
KF_WORD(Nrec, 163);
KF_WORD(RecDep, 164);
KF_WORD(RecLq, 196);
KF_WORD(RecHit, 228);
KF_WORD(Score, 260);
KF_WORD(E, 292);
KF_WORD(Jump, 325);
KF_WORD(Nodes, 326);
KF_WORD(Cap, 327);
KF_WORD(Obs, 328);
KF_WORD(Words, 336);
KF_WORD(PathWords, 40);  // Seq::pend_path: the accepted path's rows (int32), up to kKfMaxRows + 1
#undef KF_WORD

// What the budget (src/model/kflash_budget.h, KfBudget) gives the builder each cycle, from past cycles only:
// the per-depth calibration of the path model's probabilities (p = sigmoid(alpha_d + beta_d logit(q)); 0 and 1 give q
// itself), the measured cost of a cycle that verifies n rows (its GPU ms), which row counts the backend's verify runs
// at a cost of their own (allowed: bit n - 1), the verify's row capacity, the cost table's exploration (force_n > 0:
// verify that many) and the tokens per ms the recent cycles gave (rate; 0: none yet).
struct KfParams {
  float alpha[33], beta[33];  // depth d = 1 .. 32 (index d)
  float cost[33];             // n = 1 .. 32 rows (index n), ms
  kf_int force_n;
  kf_uint allowed;
  kf_int max_rows;
  kf_int cap;  // the verify's row capacity this cycle (0: max_rows); the allowed counts stay within it
  float rate;
};
#ifndef __METAL_VERSION__
// A lattice's children table (the first-order drafters: DFlash2, LiLiCorr, DSpark): children-row (s * K + p) holds
// slot s's K candidates after candidate p of slot s - 1 (slot 0: after the root, p = 0); child c's own children-row is
// (s + 1) * K + c, none past the last slot. Constant for a drafter: filled once.
inline std::vector<kf_int> lattice_next(int S, int K) {
  std::vector<kf_int> v(size_t(S) * K * K);
  for (int s = 0; s < S; ++s)
    for (int p = 0; p < K; ++p)
      for (int c = 0; c < K; ++c) v[size_t((s * K + p) * K + c)] = s + 1 < S ? (s + 1) * K + c : -1;
  return v;
}
}  // namespace ni::kf
namespace ni {
using kf::KfParams;
}
#else

// The live rows of a tree verify of R rows (the builder's n, at least 1): a verify's row-scaled work beyond them is
// skipped (their outputs are garbage; no live row reads them).
NI_INLINE uint kf_live(const device int* tree, uint R) { return min(R, uint(max(tree[kKfN], 1))); }

// ---- the tree builder: best-first over a path model's children tables ----
// The path model is three tables over "children-rows" (any indexing the model likes): clq[row][c] the log-probability
// of child c given the node the row belongs to, ctok[row][c] its token, cnext[row][c] the children-row of that child
// (-1: a leaf; a node without scored children is never expanded). A first-order lattice indexes rows by (slot,
// candidate) (kf::lattice_next); a candidate decoder (KFlash M3) by its own node ids. The root's children-row is p.root.
// A node's score is the sum over its root path of log p, p = sigmoid(alpha_d + beta_d logit(q)) the calibrated
// probability that the target accepts a node of depth d whose model probability is q. Nodes are taken in decreasing
// score (a child never scores above its parent, so rows are in insertion order; ties: the earlier frontier entry) up
// to the verify's capacity; the live rows n maximize E(n) - rate cost[n] over the allowed n (a row is worth the tokens
// per ms it costs at the long-run rate; rate 0: E(n) / cost[n]; force_n: that n; no allowed n within the rows built:
// all of them). The nodes' tokens go to tokens[P + r], the descriptor to tree, n to live[0].
// grid: 1 x 32 threads. Threadgroup memory: 12 KB + (every chip).
struct KfBuildP {
  uint K;     // children per row (<= 32)
  uint P;     // anchor position
  uint root;  // the root's children-row
  uint pad;
};
NI_INLINE float kf_lcal(float lq, float a, float b) {  // log of the calibrated acceptance probability
  const float q = min(metal::precise::exp(lq), 0.999999f);
  const float z = a + b * (lq - metal::precise::log(1.0f - q));
  return -softplus(-z);
}
// fs: the frontier entries' path scores (-inf once taken), fq their own log q, fe parent row | candidate << 5 | the
// parent's children-row << 10 (kKfMaxRows * 32 each); ns / tdep the rows' path scores and depths (kKfMaxRows)
NI_INLINE void kf_build_impl(const device float* clq, const device int* ctok, const device int* cnext,
                             constant KfParams& kp, device int* tree, device int* tokens, device uint* live, uint Kin,
                             uint P, uint root, threadgroup float* fs, threadgroup float* fq, threadgroup uint* fe,
                             threadgroup float* ns, threadgroup uint* tdep, threadgroup uint& nf,
                             threadgroup uint& nodes, uint lane) {
  const uint K = min(Kin, 32u), R = uint(clamp(kp.max_rows, 1, int(kKfMaxRows)));
  for (uint c = lane; c < K; c += 32) {  // the root's children
    const float lq = clq[root * K + c];
    fs[c] = kf_lcal(lq, kp.alpha[1], kp.beta[1]);
    fq[c] = lq;
    fe[c] = (c << 5) | (root << 10);
  }
  if (lane == 0) {
    nf = K;
    nodes = 1;
    ns[0] = 0.0f;
    tdep[0] = 0;
    tree[kKfPar] = -1;
    tree[kKfDep] = 0;
    tree[kKfMask] = 1;
    tree[kKfLq] = 0;
    tree[kKfScore] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = 1; r < R; ++r) {
    float bv = -INFINITY;
    uint bi = 0xffffffffu;
    for (uint i = lane; i < nf; i += 32)
      if (fs[i] > bv) {
        bv = fs[i];
        bi = i;
      }
    for (uint off = 16; off > 0; off >>= 1) {
      const float ov = simd_shuffle_xor(bv, off);
      const uint oi = simd_shuffle_xor(bi, off);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
    if (bv == -INFINITY) break;
    const uint e = fe[bi], pr = e & 31u, c = (e >> 5) & 31u, crow = e >> 10;
    const int next = cnext[crow * K + c];
    const uint dep = tdep[pr] + 1;
    const uint base = nf;
    if (lane == 0) {
      fs[bi] = -INFINITY;
      tdep[r] = dep;
      tree[kKfPar + r] = int(pr);
      tree[kKfDep + r] = int(dep);
      tree[kKfMask + r] = tree[kKfMask + pr] | int(1u << r);
      tree[kKfLq + r] = as_type<int>(fq[bi]);
      tree[kKfScore + r] = as_type<int>(bv);
      tokens[P + r] = ctok[crow * K + c];
      ns[r] = bv;
      nodes = r + 1;
    }
    const bool push = next >= 0 && r + 1 < R;  // (the last row's children would never be taken)
    if (push) {
      const float a = kp.alpha[min(dep + 1, 32u)], b = kp.beta[min(dep + 1, 32u)];
      for (uint c2 = lane; c2 < K; c2 += 32) {
        const float lq = clq[uint(next) * K + c2];
        fs[base + c2] = bv + kf_lcal(lq, a, b);
        fq[base + c2] = lq;
        fe[base + c2] = r | (c2 << 5) | (uint(next) << 10);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0 && push) nf = base + K;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) {
    const uint m = nodes;
    float E = 1.0f, best = -INFINITY;
    uint n = 0;
    tree[kKfE] = 0;
    for (uint nn = 1; nn <= m; ++nn) {
      if (nn > 1) E += metal::precise::exp(ns[nn - 1]);
      tree[kKfE + nn] = as_type<int>(E);
      const float v = kp.rate > 0.0f ? E - kp.rate * kp.cost[nn] : E / kp.cost[nn];
      if (((kp.allowed >> (nn - 1)) & 1u) && v > best) {
        best = v;
        n = nn;
      }
    }
    for (uint nn = m + 1; nn <= kKfMaxRows; ++nn) tree[kKfE + nn] = 0;
    if (kp.force_n > 0) n = min(uint(kp.force_n), m);
    if (n == 0) n = m;
    uint jump = 0;
    for (uint r = 1; r < m; ++r) jump |= tree[kKfPar + r] != int(r) - 1 ? 1u << r : 0u;
    for (uint r = m; r < R; ++r) {  // rows past the tree: a filler chain (never live)
      tree[kKfPar + r] = int(r - 1);
      tree[kKfDep + r] = tree[kKfDep + r - 1] + 1;
      tree[kKfMask + r] = tree[kKfMask + r - 1] | int(1u << r);
      tree[kKfLq + r] = 0;
      tree[kKfScore + r] = as_type<int>(-INFINITY);
      tokens[P + r] = ctok[root * K];
    }
    tree[kKfN] = int(n);
    tree[kKfNodes] = int(m);
    tree[kKfJump] = int(jump);
    tree[kKfCap] = kp.cap > 0 ? kp.cap : int(R);
    live[0] = n;
  }
}
[[kernel]] void kf_build(const device float* clq [[buffer(0)]], const device int* ctok [[buffer(1)]],
                         const device int* cnext [[buffer(2)]], constant KfParams& kp [[buffer(3)]],
                         device int* tree [[buffer(4)]], device int* tokens [[buffer(5)]],
                         device uint* live [[buffer(6)]], constant KfBuildP& p [[buffer(7)]],
                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float fs[kKfMaxRows * 32], fq[kKfMaxRows * 32];
  threadgroup uint fe[kKfMaxRows * 32];
  threadgroup float ns[kKfMaxRows];
  threadgroup uint tdep[kKfMaxRows];
  threadgroup uint nf, nodes;
  kf_build_impl(clq, ctok, cnext, kp, tree, tokens, live, p.K, p.P, p.root, fs, fq, fe, ns, tdep, nf, nodes, lane);
}

// A first-order lattice's children tokens: ctok[(s * K + p) * K + c] = cand[s * K + c] (slot s's candidate c, whatever
// the candidate before). grid: S * K * K threads.
struct KfLatticeP {
  uint S, K;
};
[[kernel]] void kf_lattice_tokens(const device int* cand [[buffer(0)]], device int* ctok [[buffer(1)]],
                                  constant KfLatticeP& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.S * p.K * p.K) return;
  ctok[i] = cand[(i / (p.K * p.K)) * p.K + i % p.K];
}

// ---- the tree's acceptance (greedy and sampled alike) ----
// samp[r] = the target's own token after row r: its argmax at T = 0, at T > 0 the plain sampler's draw with the uniform
// of position P + depth(r), so each row's sample is what plain decoding draws after the row's root path. From the
// root, descend into the child whose token equals the current row's sample while there is one: the emitted tokens are
// then exactly plain decoding's, whatever the tree (which the drafter fixed before any sample was drawn). Writes
// tokens[P + 1 .. P + a] = the path's tokens and tokens[P + 1 + a] = its last row's sample; res = {a, that token}; the
// path (kKfA, kKfPath, and path[i] = Seq::pend_path for the next pass's pending rows); the calibration records (every
// live node whose parent is on the path: depth, log q, 1 if it is on the path), the live rows and their E (kKfObs);
// with p.gst, the drafter state of a GPU-driven loop (gst[0] = P + a + 1, gst[1] = a + 1, gst[2] = P, gst[4 + i] =
// path[i]: DFlash2's). The live rows are min(tree[kKfN], live[0], p.R). grid: 1 x 32.
struct KfAcceptP {
  uint P;    // anchor position
  uint R;    // the verify's rows
  uint gst;  // 1: write the drafter state
  uint pad;
};
[[kernel]] void kf_accept(const device int* samp [[buffer(0)]], device int* tokens [[buffer(1)]],
                          device int* tree [[buffer(2)]], device int* res [[buffer(3)]],
                          constant KfAcceptP& p [[buffer(4)]], const device uint* live [[buffer(5)]],
                          device int* path [[buffer(6)]], device uint* gst [[buffer(7)]],
                          uint lane [[thread_index_in_simdgroup]]) {
  const uint n = min(min(uint(tree[kKfN]), live[0]), min(p.R, kKfMaxRows));
  const int par = lane < n ? tree[kKfPar + lane] : -2;
  const int tok = lane < n ? tokens[p.P + lane] : -1;
  uint v = 0, a = 0, mine = 0, onpath = 1u;  // lane i: the row of the path's i-th entry
  int y = samp[0];
  while (a + 1 < n) {
    const bool hit = lane > 0 && lane < n && par == int(v) && tok == y;
    const uint c = simd_min(hit ? lane : 32u);
    if (c == 32u) break;
    ++a;
    if (lane == a) mine = c;
    v = c;
    y = samp[v];
    onpath |= 1u << c;
  }
  const int ptok = simd_shuffle(tok, ushort(mine));
  if (lane >= 1 && lane <= a) tokens[p.P + lane] = ptok;
  if (lane <= a) {
    tree[kKfPath + lane] = int(mine);
    path[lane] = int(mine);
    if (p.gst) gst[4 + lane] = mine;
  }
  const bool rec = lane >= 1 && lane < n && par >= 0 && ((onpath >> uint(par)) & 1u);
  const uint idx = simd_prefix_exclusive_sum(rec ? 1u : 0u), nrec = simd_sum(rec ? 1u : 0u);
  if (rec) {
    tree[kKfRecDep + idx] = tree[kKfDep + lane];
    tree[kKfRecLq + idx] = tree[kKfLq + lane];
    tree[kKfRecHit + idx] = tok == samp[par] ? 1 : 0;
  }
  if (lane == 0) {
    tokens[p.P + 1 + a] = y;
    res[0] = int(a);
    res[1] = y;
    tree[kKfA] = int(a);
    tree[kKfNrec] = int(nrec);
    const int m = tree[kKfNodes];
    tree[kKfObs + 0] = int(n);
    tree[kKfObs + 1] = tree[kKfE + n];
    tree[kKfObs + 2] = tree[kKfCap];
    tree[kKfObs + 3] = m;
    tree[kKfObs + 4] = m >= 8 ? tree[kKfE + 8] : 0;
    tree[kKfObs + 5] = m >= 16 ? tree[kKfE + 16] : 0;
    tree[kKfObs + 6] = m >= 32 ? tree[kKfE + 32] : 0;
    if (p.gst) {
      gst[0] = p.P + a + 1;
      gst[1] = a + 1;
      gst[2] = p.P;
    }
  }
}
#endif
