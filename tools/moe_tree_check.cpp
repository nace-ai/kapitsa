// Bitwise check of a MoE (or any Qwen35) target's KFlash tree verify: a position's logits must not depend on the
// tree that verified it (its size, its shape, the slot its path takes) nor on the cycle anchor (where the history /
// block split of the attention and the committed-fold / in-block split of the GDN fall).
//
//   moe_tree_check <model_dir> @ids [n_tok=96] [schedule ...]
//
// 1. Prefill the prompt, then n_tok plain greedy decode steps: the teacher-forced token stream toks.
// 2. For every schedule: a fresh sequence and the stream verified cycle by cycle through tree verifies the schedule
//    builds (the real path: tree descriptor, live rows, jump rows, the K/V compaction, the next cycle's fold of the
//    accepted path). Row path[i] of a cycle anchored at P predicts position P + i + 1; its logits (fp32, whole
//    vocabulary) and, with NI_TREE_CHECK_LAYERS=1 (default), the hidden state after every layer (the bf16 capture) are
//    hashed per position.
// 3. All pairs of schedules must agree on every position's logits bit for bit. The report gives the classes, the
//    first position at which two schedules differ, the max |logit diff| there, and the first layer whose hidden state
//    differs (the bisect: which layer, hence which op, first sees the dependence).
//
// Schedules (comma-free words): cN a chain of N live rows (N = 1..8); tN N rows, the path first then junk siblings
// after it (the path keeps its slots); jN N rows, junk rows interleaved before every path row (jump rows: the path
// sits in other slots and replays its ancestry); rN:SEED N rows of a random tree with a random root path;
// mixS a changing size 1..8 and shape cycle by cycle (seed S). Default: c8 c1 c3 c5 t5 j5 j8 r8:1 r6:2 mix1 mix2.
// NI_TREE_CHECK_SKIP=k: the first k positions are not compared (nothing special; the prefill's tail).
#include "metal/gpu.h"
#include "model/kflash_budget.h"
#include "model/qwen35.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;

namespace {

constexpr int R8 = 8;

struct Shape {
  std::vector<int> par;   // parent row (row 0: -1)
  std::vector<int> path;  // the true path's rows (path[0] = 0), root to leaf
};

Shape chain_shape(int n) {
  Shape s;
  for (int r = 0; r < n; ++r) {
    s.par.push_back(r - 1);
    s.path.push_back(r);
  }
  return s;
}

// the path first (rows 0 .. k-1), then junk siblings of the path's nodes
Shape tail_shape(int n) {
  Shape s;
  const int k = std::max(1, (n + 1) / 2);
  for (int r = 0; r < k; ++r) {
    s.par.push_back(r - 1);
    s.path.push_back(r);
  }
  for (int r = k; r < n; ++r) s.par.push_back(std::max(0, r - k));  // siblings of path nodes (parent r - k)
  return s;
}

// junk before every path row: 0 root, 1 junk (child of 0), 2 path (child of 0), 3 junk (child of 2), 4 path, ...
Shape jump_shape(int n) {
  Shape s;
  s.par.push_back(-1);
  s.path.push_back(0);
  int last = 0;
  for (int r = 1; r < n; ++r) {
    if (r % 2 == 1) {
      s.par.push_back(last);  // junk
    } else {
      s.par.push_back(last);  // path node: child of the last path node
      last = r;
      s.path.push_back(r);
    }
  }
  return s;
}

Shape random_shape(int n, std::mt19937& rng) {
  Shape s;
  s.par.push_back(-1);
  for (int r = 1; r < n; ++r) s.par.push_back(int(rng() % uint32_t(r)));
  int v = 0;
  s.path.push_back(0);
  for (;;) {
    std::vector<int> kids;
    for (int r = 1; r < n; ++r)
      if (s.par[size_t(r)] == v) kids.push_back(r);
    if (kids.empty()) break;
    v = kids[rng() % kids.size()];
    s.path.push_back(v);
  }
  return s;
}

struct Schedule {
  std::string name;
  int kind = 0;  // 0 chain, 1 tail, 2 jump, 3 random, 4 mix
  int n = 8;
  uint32_t seed = 0;
};

Schedule parse_schedule(const std::string& w) {
  Schedule s;
  s.name = w;
  if (w.rfind("mix", 0) == 0) {
    s.kind = 4;
    s.seed = uint32_t(atoi(w.c_str() + 3));
    return s;
  }
  switch (w[0]) {
    case 'c': s.kind = 0; break;
    case 't': s.kind = 1; break;
    case 'j': s.kind = 2; break;
    case 'r': s.kind = 3; break;
    default: fatal("moe_tree_check: schedule '%s'", w.c_str());
  }
  s.n = std::clamp(atoi(w.c_str() + 1), 1, R8);
  if (const char* c = strchr(w.c_str(), ':')) s.seed = uint32_t(atoi(c + 1));
  return s;
}

Shape shape_for(const Schedule& sc, std::mt19937& rng) {
  switch (sc.kind) {
    case 0: return chain_shape(sc.n);
    case 1: return tail_shape(sc.n);
    case 2: return jump_shape(sc.n);
    case 3: return random_shape(sc.n, rng);
    default: {
      const int n = 1 + int(rng() % 8);
      switch (rng() % 4) {
        case 0: return chain_shape(n);
        case 1: return tail_shape(n);
        case 2: return jump_shape(n);
        default: return random_shape(n, rng);
      }
    }
  }
}

uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  const uint8_t* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

void prefill(Qwen35& m, Seq& s, const std::vector<int>& ids, const SamplingParams& sp) {
  Gpu& g = Gpu::get();
  const int n = int(ids.size()), C = m.prefill_chunk();
  for (int i = 0; i < n; ++i) s.tokens()[i] = ids[i];
  for (int p = 0; p < n; p += C) {
    const int T = std::min(C, n - p);
    auto e = g.begin();
    m.encode_prefill(*e, s, p, T, sp, 0, p + T == n);
    e->commit_and_wait();
    if (!m.ane_finish_chunk()) fatal("the NI_ANE prefill split failed; rerun with NI_ANE=0");
  }
}

struct PosRec {
  uint64_t logit = 0;          // hash of the row's logits
  std::vector<uint64_t> hid;   // hash of the hidden state after each layer
  int argmax = -1;
  int P = 0, row = 0, n = 0;   // the cycle's anchor, the row, the live rows (for the report)
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: moe_tree_check <model_dir> @ids [n_tok=96] [schedule ...]\n");
    return 1;
  }
  std::vector<int> ids;
  {
    std::string arg = argv[2];
    if (!arg.empty() && arg[0] == '@') {
      std::ifstream f(arg.substr(1));
      arg.assign(std::istreambuf_iterator<char>(f), {});
    }
    std::stringstream ss(arg);
    std::string t;
    while (std::getline(ss, t, ',')) ids.push_back(std::stoi(t));
  }
  const int n_tok = argc > 3 ? atoi(argv[3]) : 96;
  std::vector<Schedule> scheds;
  for (int i = 4; i < argc; ++i) scheds.push_back(parse_schedule(argv[i]));
  if (scheds.empty())
    for (const char* w : {"c8", "c1", "c3", "c5", "t5", "j5", "j8", "r8:1", "r6:2", "mix1", "mix2"})
      scheds.push_back(parse_schedule(w));
  const bool layers_on = !getenv("NI_TREE_CHECK_LAYERS") || atoi(getenv("NI_TREE_CHECK_LAYERS")) != 0;

  Gpu& g = Gpu::get();
  Qwen35 m(argv[1], 2048);
  if (!m.verify_tree_ok()) fatal("moe_tree_check: the model's verify has no tree path");
  const int V = m.config().vocab, n = int(ids.size()), H = m.config().hidden, L = m.config().layers;
  SamplingParams sp;  // greedy
  const int cap = n + n_tok + 64;

  // ---- the teacher-forced stream: plain greedy decoding ----
  auto s1 = m.new_seq(cap);
  prefill(m, *s1, ids, sp);
  for (int i = 0; i < n_tok; ++i) {
    auto e = g.begin();
    m.encode_step(*e, *s1, n + i, sp, uint32_t(i + 1), true);
    e->commit_and_wait();
  }
  std::vector<int> toks(s1->tokens(), s1->tokens() + n + n_tok + 1);
  s1.reset();

  Buffer tree = g.alloc(size_t(kf::kWords) * 4, "mtc_tree");
  Buffer live = g.alloc(16, "mtc_live");
  Buffer cbuf;
  CaptureHook hook;
  if (layers_on) {
    cbuf = g.alloc(size_t(R8) * L * H * 2, "mtc_capt");
    hook.dst = cbuf;
    hook.ld = uint32_t(L * H);
    hook.rows = R8;
    hook.slot.resize(size_t(L));
    for (int i = 0; i < L; ++i) hook.slot[size_t(i)] = i;
  }

  std::vector<std::vector<PosRec>> recs(scheds.size(), std::vector<PosRec>(size_t(n_tok)));
  std::vector<std::vector<float>> keep;  // schedule 0's logits per position (the diff magnitude)
  keep.resize(size_t(n_tok));

  for (size_t si = 0; si < scheds.size(); ++si) {
    const Schedule& sc = scheds[si];
    std::mt19937 rng(sc.seed * 2654435761u + 12345u);
    auto s = m.new_seq(cap);
    prefill(m, *s, ids, sp);
    for (int i = n; i <= n + n_tok; ++i) s->tokens()[i] = toks[size_t(i)];
    s->pend_path = g.alloc(size_t(kf::kPathWords) * 4, "mtc_path");
    int P = n;
    while (P < n + n_tok) {
      Shape sh = shape_for(sc, rng);
      const int nl = int(sh.par.size());
      // keep the path inside the stream
      while (int(sh.path.size()) > n + n_tok - P && sh.path.size() > 1) {
        // truncate: drop the last path node and every row from it on is not needed; simplest is a shorter chain
        sh = chain_shape(std::max(1, n + n_tok - P));
        break;
      }
      const int live_n = int(sh.par.size());
      (void)nl;
      std::vector<char> on_path(R8, 0);
      std::vector<int> dep(R8, 0);
      for (int r : sh.path) on_path[size_t(r)] = 1;
      int32_t* t = tree.as<int32_t>();
      memset(t, 0, size_t(kf::kWords) * 4);
      int mask[R8] = {};
      uint32_t jump = 0;
      for (int r = 0; r < R8; ++r) {
        const int p = r < live_n ? sh.par[size_t(r)] : r - 1;
        dep[size_t(r)] = r == 0 ? 0 : dep[size_t(p)] + 1;
        mask[r] = (r == 0 ? 0 : mask[p]) | (1 << r);
        t[kf::kPar + r] = r == 0 ? -1 : p;
        t[kf::kDep + r] = dep[size_t(r)];
        t[kf::kMask + r] = mask[r];
        if (r >= 1 && r < live_n && p != r - 1) jump |= 1u << r;
      }
      t[kf::kN] = live_n;
      t[kf::kNodes] = live_n;
      t[kf::kJump] = int(jump);
      const int a = int(sh.path.size()) - 1;
      t[kf::kA] = a;
      for (int i = 0; i <= a; ++i) t[kf::kPath + i] = sh.path[size_t(i)];
      live.as<uint32_t>()[0] = uint32_t(live_n);
      // the rows' tokens: a path node its depth's true token, a junk node a token that is not (deterministic)
      for (int r = 0; r < R8; ++r) {
        const int d = dep[size_t(r)];
        const int truth = P + d <= n + n_tok ? toks[size_t(P + d)] : toks[size_t(n + n_tok)];
        s->tokens()[P + r] = (r < live_n && !on_path[size_t(r)]) ? (truth + 1 + r * 7919) % V : truth;
      }
      if (s->tokens()[P] != toks[size_t(P)]) fatal("anchor token");
      auto e = g.begin();
      if (layers_on) {
        hook.pos_base = P;
        m.set_capture(&hook);
      }
      m.set_verify_tree(&tree);
      m.set_verify_row_limit(&live);
      m.set_verify_moe_exact(true);
      m.encode_verify(*e, *s, P, R8, sp, uint32_t(P), /*v8=*/true);
      m.set_verify_moe_exact(false);
      m.set_verify_row_limit(nullptr);
      m.set_verify_tree(nullptr);
      m.set_capture(nullptr);
      e->commit_and_wait();
      // acceptance (kf_accept's writes): this cycle's path becomes the pending rows of the next pass; K / V of the path
      // move to their positions
      for (int i = 0; i <= a; ++i) s->pend_path.as<int32_t>()[i] = sh.path[size_t(i)];
      {
        auto e2 = g.begin();
        m.encode_kv_tree_compact(*e2, *s, P, tree);
        e2->commit_and_wait();
      }
      s->pend_tree = true;
      for (int i = 0; i <= a && P + i < n + n_tok; ++i) {
        const int row = sh.path[size_t(i)];
        const float* lg = m.logits() + size_t(row) * V;
        PosRec& pr = recs[si][size_t(P + i - n)];
        pr.logit = fnv(lg, size_t(V) * 4);
        pr.argmax = int(std::max_element(lg, lg + V) - lg);
        pr.P = P;
        pr.row = row;
        pr.n = live_n;
        if (layers_on) {
          pr.hid.resize(size_t(L));
          const uint16_t* c = cbuf.as<uint16_t>() + size_t(row) * L * H;
          for (int l = 0; l < L; ++l) pr.hid[size_t(l)] = fnv(c + size_t(l) * H, size_t(H) * 2);
        }
        if (si == 0) keep[size_t(P + i - n)].assign(lg, lg + V);
      }
      m.commit_rows(*s, a + 1);
      P += a + 1;
    }
    fprintf(stderr, "schedule %s done\n", sc.name.c_str());
  }

  // ---- all pairs ----
  int bad_pairs = 0;
  std::vector<int> cls(scheds.size(), -1);
  int ncls = 0;
  for (size_t i = 0; i < scheds.size(); ++i) {
    for (size_t j = 0; j < i; ++j) {
      int first = -1;
      for (int p = 0; p < n_tok; ++p)
        if (recs[i][size_t(p)].logit != recs[j][size_t(p)].logit) {
          first = p;
          break;
        }
      if (first < 0) {
        if (cls[i] < 0) cls[i] = cls[j];
        continue;
      }
      ++bad_pairs;
      const PosRec& a = recs[j][size_t(first)];
      const PosRec& b = recs[i][size_t(first)];
      int fl = -1;
      if (layers_on)
        for (int l = 0; l < L; ++l)
          if (a.hid.size() == size_t(L) && b.hid.size() == size_t(L) && a.hid[size_t(l)] != b.hid[size_t(l)]) {
            fl = l;
            break;
          }
      printf("DIFF %-6s vs %-6s: position %d (token %d) first differing layer %d | %s: anchor %d row %d of %d live, "
             "%s: anchor %d row %d of %d live | argmax %d / %d\n",
             scheds[j].name.c_str(), scheds[i].name.c_str(), first, n + first + 1, fl, scheds[j].name.c_str(), a.P - n,
             a.row, a.n, scheds[i].name.c_str(), b.P - n, b.row, b.n, a.argmax, b.argmax);
    }
    if (cls[i] < 0) cls[i] = ncls++;
  }
  printf("classes:");
  for (int c = 0; c < ncls; ++c) {
    printf(" [");
    bool f = true;
    for (size_t i = 0; i < scheds.size(); ++i)
      if (cls[i] == c) {
        printf("%s%s", f ? "" : " ", scheds[i].name.c_str());
        f = false;
      }
    printf("]");
  }
  printf("\n");
  int argmax_diff = 0;
  for (int p = 0; p < n_tok; ++p)
    for (size_t i = 1; i < scheds.size(); ++i)
      if (recs[i][size_t(p)].argmax != recs[0][size_t(p)].argmax) {
        ++argmax_diff;
        break;
      }
  printf("%d of %zu schedules' pairs differ bitwise (%zu schedules, %d positions); positions where an argmax differs "
         "from %s: %d\n",
         bad_pairs, scheds.size() * (scheds.size() - 1) / 2, scheds.size(), n_tok, scheds[0].name.c_str(), argmax_diff);
  printf(bad_pairs == 0 ? "PASS: every schedule's logits are bitwise equal\n" : "FAIL: logits depend on the tree / anchor\n");
  return bad_pairs == 0 ? 0 : 2;
}
