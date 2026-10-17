// Splits the multi-row verify's deviation from plain decoding into (a) the per-step error of the
// verify kernels on the PLAIN state and (b) the accumulated drift when the verify path builds the state.
//   ni_v8drift <model_dir> @ids [n_tok=512] [R=8]
// Pass 1: prefill + n_tok plain greedy steps, logits kept.
// Pass 2 (fresh): same prefill; at every P = n + R k, encode_verify of the plain tokens P..P+R-1 on the plain
//   state (n_pend = 0: the verify leaves the recurrent state alone; its KV rows are rewritten by the plain steps
//   that follow), compare its R rows, then R plain steps (their logits must equal pass 1 bit for bit).
// Pass 3 (accumulated, = ni_v8check): same prefill; verify R rows at a time, all committed.
// Metrics per row (row r predicts position P + r + 1): |dlogit| at the plain argmax, the change of the plain
// top-2 gap |(a1 - a2) - (b1 - b2)| (shift invariant: what decides an argmax flip), argmax flips.
#include "metal/gpu.h"
#include "model/qwen35.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;

static void prefill(Qwen35& m, Seq& s, const std::vector<int>& ids, const SamplingParams& sp) {
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

struct Stats {
  std::vector<double> top, gap;
  int flips = 0, rows = 0;
  std::vector<std::string> notes;
  void add(const float* a, const float* b, int V, int pos) {
    int am = 0, bm = 0, a2 = -1;
    for (int v = 0; v < V; ++v) {
      if (a[v] > a[am]) am = v;
      if (b[v] > b[bm]) bm = v;
    }
    for (int v = 0; v < V; ++v)
      if (v != am && (a2 < 0 || a[v] > a[a2])) a2 = v;
    top.push_back(std::fabs(a[am] - b[am]));
    gap.push_back(std::fabs((a[am] - a[a2]) - (b[am] - b[a2])));
    ++rows;
    if (am != bm) {
      ++flips;
      char buf[256];
      snprintf(buf, sizeof buf, "pos %d: plain %d gap %.4f -> verify %d (verify gap of plain top-2 %.4f)", pos, am,
               a[am] - a[a2], bm, b[am] - b[a2]);
      notes.push_back(buf);
    }
  }
  static double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[size_t(q * (v.size() - 1))];
  }
  void print(const char* name) const {
    double mt = 0;
    for (double x : top) mt += x;
    printf("%s: %d rows, argmax flips %d; |dlogit@argmax| mean %.4f median %.4f p95 %.4f max %.4f; |dgap top-2| median "
           "%.4f p95 %.4f p99 %.4f max %.4f\n",
           name, rows, flips, mt / std::max(rows, 1), pct(top, 0.5), pct(top, 0.95), pct(top, 1.0), pct(gap, 0.5),
           pct(gap, 0.95), pct(gap, 0.99), pct(gap, 1.0));
    for (auto& s : notes) printf("    %s\n", s.c_str());
  }
};

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: ni_v8drift <model_dir> @ids [n_tok] [R]\n");
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
  const int n_tok = argc > 3 ? atoi(argv[3]) : 512, R = argc > 4 ? atoi(argv[4]) : 8;
  Gpu& g = Gpu::get();
  Qwen35 m(argv[1], 2048);
  const int V = m.config().vocab, n = int(ids.size());
  SamplingParams sp;
  const int cap = n + n_tok + 64;
  // pass 1
  auto s1 = m.new_seq(cap);
  prefill(m, *s1, ids, sp);
  std::vector<float> plain(size_t(n_tok) * V);
  for (int i = 0; i < n_tok; ++i) {
    auto e = g.begin();
    m.encode_step(*e, *s1, n + i, sp, uint32_t(i + 1), true);
    e->commit_and_wait();
    memcpy(plain.data() + size_t(i) * V, m.logits(), size_t(V) * 4);
  }
  std::vector<int> toks(s1->tokens(), s1->tokens() + n + n_tok + 1);
  // pass 2: fresh verifies on the plain state
  Stats fresh;
  long replay_bad = 0;
  {
    auto s2 = m.new_seq(cap);
    prefill(m, *s2, ids, sp);
    for (int i = n; i <= n + n_tok; ++i) s2->tokens()[i] = toks[i];
    for (int P = n; P + R <= n + n_tok; P += R) {
      {
        auto e = g.begin();
        m.encode_verify(*e, *s2, P, R, sp, uint32_t(P));
        e->commit_and_wait();
      }
      m.commit_rows(*s2, 0);
      for (int r = 0; r < R; ++r) fresh.add(plain.data() + size_t(P - n + r) * V, m.logits() + size_t(r) * V, V, P + r + 1);
      for (int i = 0; i < n_tok; ++i) s2->tokens()[n + i + 1] = toks[n + i + 1];  // the verify wrote samples
      for (int q = P; q < P + R; ++q) {
        auto e = g.begin();
        m.encode_step(*e, *s2, q, sp, uint32_t(q - n + 1), true);
        e->commit_and_wait();
        if (memcmp(m.logits(), plain.data() + size_t(q - n) * V, size_t(V) * 4) != 0) ++replay_bad;
        s2->tokens()[q + 1] = toks[q + 1];
      }
    }
  }
  // pass 3: accumulated (the verify path builds the state)
  Stats acc;
  {
    auto s3 = m.new_seq(cap);
    prefill(m, *s3, ids, sp);
    for (int i = n; i <= n + n_tok; ++i) s3->tokens()[i] = toks[i];
    for (int P = n; P + R <= n + n_tok; P += R) {
      auto e = g.begin();
      m.encode_verify(*e, *s3, P, R, sp, uint32_t(P));
      e->commit_and_wait();
      for (int r = 0; r < R; ++r) acc.add(plain.data() + size_t(P - n + r) * V, m.logits() + size_t(r) * V, V, P + r + 1);
      for (int i = n; i <= n + n_tok; ++i) s3->tokens()[i] = toks[i];
      m.commit_rows(*s3, R);
    }
  }
  printf("prompt %d, %d generated rows, R=%d (plain replay mismatches in pass 2: %ld of %d steps)\n", n, n_tok, R,
         replay_bad, fresh.rows);
  fresh.print("fresh (verify on the plain state)");
  acc.print("accumulated (verify path builds the state)");
  return 0;
}
