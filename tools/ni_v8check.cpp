// Teacher-forced check of the multi-row verify against plain decoding, plus timing.
//   ni_v8check <model_dir> @ids [n_tok=64] [R=8] [reps=1]
// 1. Prefill the prompt, then n_tok plain greedy decode steps (one command buffer each), keeping
//    every step's logits and GPU time.
// 2. Fresh sequence, same prefill; then verify the SAME tokens R rows at a time (encode_verify,
//    all rows committed, so the GDN tape folds R rows per cycle) and compare every row's logits with
//    the plain step that predicts the same position: max |diff| over the vocabulary, the diff at the
//    plain argmax, and argmax agreement (with the plain top-2 gap wherever they differ).
// R <= 4 uses the MTP-era kernels unless NI_V8=1. reps > 1 re-runs the verify sweep (timing only;
// the KV/GDN state is rebuilt by a fresh prefill each time).
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
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: ni_v8check <model_dir> @ids [n_tok] [R] [reps]\n");
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
  const int n_tok = argc > 3 ? atoi(argv[3]) : 64, R = argc > 4 ? atoi(argv[4]) : 8, reps = argc > 5 ? atoi(argv[5]) : 1;
  Gpu& g = Gpu::get();
  Qwen35 m(argv[1], 2048);
  const int V = m.config().vocab, n = int(ids.size());
  SamplingParams sp;  // greedy
  const int cap = n + n_tok + 64;

  // ---- plain decode ----
  auto s1 = m.new_seq(cap);
  prefill(m, *s1, ids, sp);
  std::vector<float> plain(size_t(n_tok) * V);
  std::vector<double> plain_ms;
  for (int i = 0; i < n_tok; ++i) {
    auto e = g.begin();
    m.encode_step(*e, *s1, n + i, sp, uint32_t(i + 1), true);
    e->commit_and_wait();
    plain_ms.push_back(e->gpu_ms());
    memcpy(plain.data() + size_t(i) * V, m.logits(), size_t(V) * 4);
  }
  std::vector<int> toks(s1->tokens(), s1->tokens() + n + n_tok + 1);
  if (Encoder::profiling()) {
    fprintf(stderr, "==== plain decode profile ====\n");
    Encoder::profile_report(n_tok);
  }

  // ---- teacher-forced verify ----
  std::vector<double> ver_ms;
  int agree = 0, rows = 0, flips = 0;
  double max_all = 0, sum_top = 0, max_top = 0;
  std::vector<double> top_diffs;
  double ustat_sum = 0;
  int ustat_n = 0;
  for (int rep = 0; rep < reps; ++rep) {
    auto s2 = m.new_seq(cap);
    prefill(m, *s2, ids, sp);
    for (int i = n; i <= n + n_tok; ++i) s2->tokens()[i] = toks[i];
    if (Encoder::profiling()) {
      fprintf(stderr, "==== (prefill, ignore) ====\n");
      Encoder::profile_report(1);
    }
    for (int P = n; P + R <= n + n_tok; P += R) {
      auto e = g.begin();
      m.encode_verify(*e, *s2, P, R, sp, uint32_t(P));
      e->commit_and_wait();
      ver_ms.push_back(e->gpu_ms());
      if (rep) {
        m.commit_rows(*s2, R);
        continue;
      }
      if (const uint32_t* u = m.verify8_experts()) {
        double su = 0;
        for (int l = 0; l < m.config().layers; ++l) su += u[l];
        ustat_sum += su / m.config().layers;
        ++ustat_n;
      }
      const float* lv = m.logits();
      for (int r = 0; r < R; ++r) {
        const float* a = plain.data() + size_t(P - n + r) * V;
        const float* b = lv + size_t(r) * V;
        int am = 0, bm = 0, a2 = -1;
        double md = 0;
        for (int v = 0; v < V; ++v) {
          md = std::max(md, double(std::fabs(a[v] - b[v])));
          if (a[v] > a[am]) am = v;
          if (b[v] > b[bm]) bm = v;
        }
        for (int v = 0; v < V; ++v)
          if (v != am && (a2 < 0 || a[v] > a[a2])) a2 = v;
        const double td = std::fabs(a[am] - b[am]);
        top_diffs.push_back(td);
        max_all = std::max(max_all, md);
        max_top = std::max(max_top, td);
        sum_top += td;
        ++rows;
        if (am == bm) ++agree;
        else {
          ++flips;
          printf("  row at pos %d: plain argmax %d (%.4f, 2nd %d %.4f, gap %.4f) vs verify argmax %d (%.4f vs %.4f)\n",
                 P + r + 1, am, a[am], a2, a[a2], a[am] - a[a2], bm, b[bm], b[am]);
        }
        if (m.samples()[r] != bm) printf("  row %d: sampled %d != argmax %d\n", P + r + 1, m.samples()[r], bm);
      }
      m.commit_rows(*s2, R);
    }
  }
  if (Encoder::profiling()) {
    fprintf(stderr, "==== verify R=%d profile (per cycle) ====\n", R);
    Encoder::profile_report(int(ver_ms.size()));
  }
  std::sort(top_diffs.begin(), top_diffs.end());
  auto pct = [&](double q) { return top_diffs.empty() ? 0.0 : top_diffs[size_t(q * (top_diffs.size() - 1))]; };
  double pm = 0, vm = 0;
  for (double x : plain_ms) pm += x;
  std::sort(ver_ms.begin(), ver_ms.end());
  for (double x : ver_ms) vm += x;
  pm /= plain_ms.size();
  vm /= ver_ms.size();
  printf("prompt %d, %d rows checked, R=%d: argmax agree %d/%d (%d differ); |logit diff| at plain argmax: mean %.4f "
         "median %.4f p95 %.4f max %.4f; max over vocab %.4f\n",
         n, rows, R, agree, rows, flips, sum_top / std::max(rows, 1), pct(0.5), pct(0.95), max_top, max_all);
  if (ustat_n) printf("MoE: %.1f distinct routed experts per layer and verify (mean over %d verifies)\n", ustat_sum / ustat_n, ustat_n);
  printf("GPU ms: plain step %.2f (mean of %zu); verify R=%d mean %.2f median %.2f (%zu cycles) = %.2fx a plain step, "
         "%.2f ms/row\n",
         pm, plain_ms.size(), R, vm, ver_ms[ver_ms.size() / 2], ver_ms.size(), vm / pm, vm / R);
  return 0;
}
