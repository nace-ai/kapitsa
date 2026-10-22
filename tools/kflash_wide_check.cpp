// KFlash tree verifies wider than 8 rows (Qwen35::verify_wide_ok: 16 / 32 rows, GPU-driven row tiles) against plain
// decoding, on a real dense Qwen model:
//   kflash_wide_check <model_dir> [prompt: @ids file | comma-separated ids] [cases=6] [seed=1]
// After a prefill of the prompt, each case writes a tree descriptor of 32 rows (a 32-deep chain, random trees, a bushy
// best-first-like tree) with a live row count n (the size rule's choice: the verify runs 8 ceil(n / 8) rows), runs the
// target's tree verify of capacity 32, and compares every live row's logits with plain decoding of that row's root path
// (encode_step token by token from the prefill's state): max |logit diff|, its share of the row's logit range, argmax
// agreement (a differing argmax counts only when plain decoding's top two are further apart than the error). Also:
//  - the first 8 rows against the same tree verified at capacity 8 (the 8-row tree verify): equal up to fp rounding (bit-equal with NI_V8_TREE_ATTN=0:
//    the 8-row verify's default zone attention rounds differently);
//  - the next pass after an accepted path through the later tiles (K/V compaction, the GDN tape and conv window fold
//    of tape rows past 8): a plain step after encode_flush, and a second tree verify that folds the path itself
//    (its live rows against plain decoding of path + their own root paths).
// Exit code 0 when everything passes. Takes the GPU lock ($BENCH_GPU_LOCK, default /tmp/kapitsa-gpu.lock) unless
// NI_LOCK_HELD=1 or NACE_GPU_LOCK_HELD=1.
#include "bench/bench.h"
#include "kernels/kflash.h"
#include "metal/gpu.h"
#include "model/qwen35.h"
#include "model/qwen4_exp.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;

namespace {

struct Tree {
  std::vector<int> par, dep, tok;  // rows 0 .. 31 (row 0: the anchor)
  int n = 32;                      // live rows (the size rule's choice)
};

// the descriptor words kf_build writes for these rows (parents before children)
void write_desc(const Tree& t, int32_t* d) {
  memset(d, 0, size_t(kf::kWords) * 4);
  const int R = int(t.par.size());
  uint32_t jump = 0;
  for (int r = 0; r < R; ++r) {
    d[kf::kPar + r] = t.par[r];
    d[kf::kDep + r] = t.dep[r];
    d[kf::kMask + r] = int32_t((r ? uint32_t(d[kf::kMask + t.par[r]]) : 0u) | (1u << r));
    if (r > 0 && t.par[r] != r - 1) jump |= 1u << r;
  }
  d[kf::kN] = t.n;
  d[kf::kNodes] = R;
  d[kf::kJump] = int32_t(jump);
}

std::vector<int> root_path(const Tree& t, int r) {  // rows from the root to r
  std::vector<int> p;
  for (int v = r; v >= 0; v = v ? t.par[v] : -1) p.push_back(v);
  std::reverse(p.begin(), p.end());
  return p;
}

struct Cmp {
  double max_abs = 0, max_rel = 0;
  int argmax_bad = 0, rows = 0;
};
// a against b (b: plain decoding's logits)
void compare(const float* a, const float* b, int V, Cmp& c) {
  double mx = 0, lo = 1e30, hi = -1e30;
  int ia = 0, ib = 0;
  for (int v = 0; v < V; ++v) {
    mx = std::max(mx, double(std::fabs(a[v] - b[v])));
    lo = std::min(lo, double(b[v]));
    hi = std::max(hi, double(b[v]));
    if (a[v] > a[ia]) ia = v;
    if (b[v] > b[ib]) ib = v;
  }
  float second = -1e30f;
  for (int v = 0; v < V; ++v)
    if (v != ib) second = std::max(second, b[v]);
  if (ia != ib && double(b[ib] - second) > 2 * mx) ++c.argmax_bad;
  c.max_abs = std::max(c.max_abs, mx);
  c.max_rel = std::max(c.max_rel, mx / std::max(1e-6, hi - lo));
  ++c.rows;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: kflash_wide_check <model_dir> [@ids | ids] [cases=6] [seed=1]\n");
    return 1;
  }
  const std::string dir = argv[1];
  std::vector<int> ids;
  {
    std::string arg = argc > 2 ? argv[2] : "1,2,3,4,5,6,7,8";
    if (arg[0] == '@') {
      std::ifstream f(arg.substr(1));
      arg.assign(std::istreambuf_iterator<char>(f), {});
    }
    std::stringstream ss(arg);
    std::string t;
    while (std::getline(ss, t, ','))
      if (!t.empty()) ids.push_back(std::stoi(t));
  }
  const int ncases = argc > 3 ? atoi(argv[3]) : 6;
  std::mt19937_64 rng(argc > 4 ? strtoull(argv[4], nullptr, 10) : 1);

  bench::GpuLock lock;
  const char* held = getenv("NI_LOCK_HELD");
  const char* held2 = getenv("NACE_GPU_LOCK_HELD");
  if (!(held && !strcmp(held, "1")) && !(held2 && !strcmp(held2, "1"))) {
    if (!lock.acquire([](const std::string& who) {
          fprintf(stderr, "waiting for the GPU lock %s%s\n", bench::GpuLock::path().c_str(),
                  who.empty() ? "" : (", held by " + who).c_str());
        }))
      fatal("cannot take the GPU lock %s: %s", bench::GpuLock::path().c_str(), strerror(errno));
  }
  Gpu& g = Gpu::get();
  std::unique_ptr<Qwen35> mp = load_qwen(dir, 2048);
  Qwen35& m = *mp;
  if (!m.verify_wide_ok()) fatal("%s: this model's verify runs 8-row trees only", dir.c_str());
  constexpr int RW = Qwen35::kMaxTreeRows;
  m.set_tree_rows(RW);
  const int V = m.vocab(), n = int(ids.size());
  auto s = m.new_seq(n + 256), s2 = m.new_seq(n + 256);
  SamplingParams greedy;
  greedy.temperature = 0;
  // the prefill (both sequences), the anchor sampled at position n; checkpoints of the state after it
  for (Seq* q : {s.get(), s2.get()}) {
    for (int i = 0; i < n; ++i) q->tokens()[i] = ids[i];
    for (int p = 0; p < n; p += m.prefill_chunk()) {
      auto e = g.begin();
      const int T = std::min(m.prefill_chunk(), n - p);
      m.encode_prefill(*e, *q, p, T, greedy, 0, p + T == n);
      e->commit_and_wait();
    }
  }
  const int P = n, anchor = s->tokens()[P];
  Checkpoint ck = m.new_checkpoint(), ck2 = m.new_checkpoint();
  {
    auto e = g.begin();
    m.save(*e, *s, ck, P);
    m.save(*e, *s2, ck2, P);
    e->commit_and_wait();
  }
  Buffer tb = g.alloc(size_t(kf::kWords) * 4, "check_tree");
  s->pend_path = g.alloc(size_t(kf::kPathWords) * 4, "pend_path");
  auto restore = [&](Seq& q, const Checkpoint& c) {
    auto e = g.begin();
    m.restore(*e, q, c);
    e->commit_and_wait();
    q.n_pend = 0;
    q.pend_tree = false;
  };
  // plain decoding of `toks` at positions P.. from the prefill's state: the logits after its last token
  auto plain = [&](const std::vector<int>& toks) {
    restore(*s2, ck2);
    for (size_t i = 0; i < toks.size(); ++i) s2->tokens()[P + int(i)] = toks[i];
    auto e = g.begin();
    for (size_t i = 0; i < toks.size(); ++i)
      m.encode_step(*e, *s2, P + int(i), greedy, 0, i + 1 == toks.size());
    e->commit_and_wait();
    return std::vector<float>(m.logits(), m.logits() + V);
  };
  // the tree's rows verified at P0 (tokens[P0 + r]); capacity R; the live rows' logits
  auto verify = [&](const Tree& t, int P0, int R) {
    write_desc(t, tb.as<int32_t>());
    for (int r = 0; r < R; ++r) s->tokens()[P0 + r] = t.tok[r];
    m.set_verify_tree(&tb);
    auto e = g.begin();
    m.encode_verify(*e, *s, P0, R, greedy, uint32_t(P0), true);
    e->commit_and_wait();
    m.set_verify_tree(nullptr);
    const int rows = R <= 8 ? R : std::min(R, (t.n + 7) / 8 * 8);
    return std::vector<float>(m.logits(), m.logits() + size_t(rows) * V);
  };
  auto random_tok = [&] { return ids[rng() % ids.size()]; };
  // trees: 0 a 32-deep chain, 1 random parents, 2 bushy (parents among the first rows: many jumps), then random;
  // live rows 32, 16, 24 (three tiles), 9 (two tiles, one live row past 8), 32 ...
  const int live[] = {32, 32, 16, 24, 9, 32, 17, 32};
  int fails = 0;
  for (int c = 0; c < ncases; ++c) {
    Tree t;
    t.par.assign(RW, -1);
    t.dep.assign(RW, 0);
    t.tok.assign(RW, 0);
    t.n = live[c % 8];
    t.tok[0] = anchor;
    for (int r = 1; r < RW; ++r) {
      const int kind = c % 3;
      t.par[r] = kind == 0 ? r - 1 : kind == 1 ? int(rng() % uint64_t(r)) : int(rng() % uint64_t(std::min(r, 4)));
      t.dep[r] = t.dep[t.par[r]] + 1;
      t.tok[r] = random_tok();
    }
    int maxdep = 0;
    for (int r = 0; r < RW; ++r) maxdep = std::max(maxdep, t.dep[r]);
    restore(*s, ck);
    const std::vector<float> w = verify(t, P, RW);
    const int rows = int(w.size() / size_t(V));
    Cmp cw;
    for (int r = 0; r < rows; ++r) {
      std::vector<int> toks;
      for (int v : root_path(t, r)) toks.push_back(t.tok[v]);
      const std::vector<float> ref = plain(toks);
      compare(w.data() + size_t(r) * V, ref.data(), V, cw);
    }
    // the first 8 rows at capacity 8 (the 8-row tree verify)
    Tree t8 = t;
    t8.par.resize(8);
    t8.dep.resize(8);
    t8.tok.resize(8);
    t8.n = 8;
    restore(*s, ck);
    const std::vector<float> w8 = verify(t8, P, 8);
    double d8 = 0;
    for (size_t i = 0; i < w8.size(); ++i) d8 = std::max(d8, double(std::fabs(w8[i] - w[i])));
    // the 8-row verify's attention is the tree-zone kernel by default (NI_V8_TREE_ATTN=2), a wide verify's the tree
    // kernel (no zone path for >8 rows): the same math, rounded differently (a diff of plain decoding's size);
    // NI_V8_TREE_ATTN=0 runs both on the tree kernel: bit-equal
    const char* ta = getenv("NI_V8_TREE_ATTN");
    // the bound: bit-equal when both run the tree kernel, else twice the wide verify's own distance to plain decoding
    // (at least 2e-2): two roundings of the same math, each within that distance of plain decoding's
    const double tol8 = ta && atoi(ta) == 0 ? 1e-3 : std::max(2e-2, 2 * cw.max_abs);
    const bool ok = cw.argmax_bad == 0 && cw.max_rel < 0.02 && d8 < tol8;
    printf("[tree %d] %s, depth up to %d, %d live rows (%d verified): vs plain decoding max |diff| %.4f (%.4f of the "
           "range), argmax mismatches %d of %d; rows 0-7 vs the 8-row verify max |diff| %.2g: %s\n",
           c, c % 3 == 0 ? "chain" : c % 3 == 1 ? "random" : "bushy", maxdep, t.n, rows, cw.max_abs, cw.max_rel,
           cw.argmax_bad, cw.rows, d8, ok ? "PASS" : "FAIL");
    if (!ok) ++fails;

    // the next pass after an accepted path to the deepest live row past the first tile
    int r_end = 0;
    for (int r = std::min(8, rows - 1); r < rows; ++r)
      if (t.dep[r] > t.dep[r_end]) r_end = r;
    const std::vector<int> path = root_path(t, r_end);
    const int a = int(path.size()) - 1;
    if (a + 1 > kf::kPathWords - 1) continue;
    std::vector<int> ptoks;
    for (int v : path) ptoks.push_back(t.tok[v]);
    const int next = random_tok();
    auto accept = [&] {  // what kf_accept leaves: the path (descriptor, Seq::pend_path), K/V compaction, pending rows
      restore(*s, ck);
      (void)verify(t, P, RW);
      int32_t* d = tb.as<int32_t>();
      d[kf::kA] = a;
      for (int i = 0; i <= a; ++i) d[kf::kPath + i] = path[size_t(i)], s->pend_path.as<int32_t>()[i] = path[size_t(i)];
      auto e = g.begin();
      m.encode_kv_tree_compact(*e, *s, P, tb);
      e->commit_and_wait();
      m.commit_rows(*s, a + 1);
      s->pend_tree = true;
      for (int i = 0; i <= a; ++i) s->tokens()[P + i] = ptoks[size_t(i)];
    };
    // (i) encode_flush, then a plain step of the next token
    accept();
    {
      s->tokens()[P + a + 1] = next;
      auto e = g.begin();
      m.encode_flush(*e, *s);
      m.encode_step(*e, *s, P + a + 1, greedy, 0, true);
      e->commit_and_wait();
    }
    std::vector<float> got(m.logits(), m.logits() + V);
    std::vector<int> toks = ptoks;
    toks.push_back(next);
    Cmp cf;
    compare(got.data(), plain(toks).data(), V, cf);
    // (ii) a second tree verify folds the path itself: a random 32-row tree from the next token, its live rows
    accept();
    Tree t2;
    t2.par.assign(RW, -1);
    t2.dep.assign(RW, 0);
    t2.tok.assign(RW, 0);
    t2.n = 32;
    t2.tok[0] = next;
    for (int r = 1; r < RW; ++r) {
      t2.par[r] = int(rng() % uint64_t(r));
      t2.dep[r] = t2.dep[t2.par[r]] + 1;
      t2.tok[r] = random_tok();
    }
    const std::vector<float> w2 = verify(t2, P + a + 1, RW);
    Cmp c2;
    for (int r = 0; r < RW; r += 5) {  // every fifth row (each needs a plain decoding of path + its root path)
      std::vector<int> tk = ptoks;
      for (int v : root_path(t2, r)) tk.push_back(t2.tok[v]);
      compare(w2.data() + size_t(r) * V, plain(tk).data(), V, c2);
    }
    const bool ok2 = cf.argmax_bad == 0 && cf.max_rel < 0.02 && c2.argmax_bad == 0 && c2.max_rel < 0.02;
    printf("  next pass after the accepted path to row %d (%d tokens): flush + step max |diff| %.4f; the next tree "
           "verify's rows max |diff| %.4f, argmax mismatches %d of %d: %s\n",
           r_end, a + 1, cf.max_abs, c2.max_abs, c2.argmax_bad, c2.rows, ok2 ? "PASS" : "FAIL");
    if (!ok2) ++fails;
  }
  printf("%s\n", fails ? "FAIL" : "all passed");
  return fails ? 1 : 0;
}
