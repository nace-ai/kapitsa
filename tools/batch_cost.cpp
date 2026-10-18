// What decoding several requests in one forward pass would cost, measured on one sequence: after a prefill, the
// median GPU time of a plain decode step and of a verify pass of R = 1..8 rows (the multi-row path speculative decoding
// verifies with) whose tokens are
//   - unrelated (random ids): rows routed as R separate requests' tokens would be (MoE: the experts' union), and
//   - the sequence's own continuation (its greedy tokens): rows routed as one request's drafted chain.
// A pass of R rows from R requests would read each weight once for all of them; its per-sequence parts (attention,
// recurrent state) cost what they cost here. Nothing is committed (the state stays at the prompt + 8 tokens).
//   batch_cost <model_dir> @ids [reps=9]      (Qwen3.5-family or Nemotron-H; NI_ANE=0 keeps the prefill on the GPU)
#include "metal/gpu.h"
#include "model/nemotron_h.h"
#include "model/qwen35.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v[v.size() / 2];
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: batch_cost <model_dir> @ids [reps]\n");
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
  const int reps = argc > 3 ? std::max(1, atoi(argv[3])) : 9;
  Gpu& g = Gpu::get();
  const bool nemo = NemotronH::is_model(argv[1]);
  std::unique_ptr<Model> model;
  Qwen35* q = nullptr;
  NemotronH* nh = nullptr;
  if (nemo) {
    auto m = std::make_unique<NemotronH>(argv[1], 2048);
    nh = m.get();
    model = std::move(m);
  } else {
    auto m = std::make_unique<Qwen35>(argv[1], 2048);
    q = m.get();
    model = std::move(m);
  }
  const int n = int(ids.size()), V = model->vocab();
  auto s = model->new_seq(n + 64);
  SamplingParams sp;  // greedy
  for (int i = 0; i < n; ++i) s->tokens()[i] = ids[i];
  for (int p = 0, C = model->prefill_chunk(); p < n; p += C) {
    const int T = std::min(C, n - p);
    auto e = g.begin();
    model->encode_prefill(*e, *s, p, T, sp, 0, p + T == n);
    e->commit_and_wait();
    if (!model->ane_finish_chunk()) fatal("the Neural Engine prefill split failed; rerun with NI_ANE=0");
  }
  // plain decode: 8 steps (their tokens are the continuation below), then `reps` more for the timing
  std::vector<double> step_ms;
  int pos = n;
  for (int i = 0; i < 8 + reps; ++i, ++pos) {
    auto e = model->grouped_passes() ? g.begin_grouped() : g.begin();
    model->encode_step(*e, *s, pos, sp, uint32_t(pos), true);
    e->commit_and_wait();
    if (i >= 8) step_ms.push_back(e->gpu_ms());
  }
  const int P = pos;  // the verify rows sit here; the continuation is the first 8 decoded tokens
  std::vector<int> cont(s->tokens() + n, s->tokens() + n + 8);
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> any(1000, std::min(V, 150000) - 1);
  printf("%s, prompt %d tokens: plain decode step %.2f ms GPU (median of %d)\n", argv[1], n, median(step_ms), reps);
  printf("verify of R rows, ms GPU (median of %d): unrelated rows (R requests) | the sequence's own continuation\n", reps);
  for (int R = 1; R <= 8; ++R) {
    double ms[2];
    for (int kind = 0; kind < 2; ++kind) {
      std::vector<double> v;
      for (int k = 0; k < reps; ++k) {
        for (int r = 0; r < R; ++r) s->tokens()[P + r] = kind == 0 ? any(rng) : cont[size_t(r)];
        auto e = model->grouped_passes() ? g.begin_grouped() : g.begin();
        if (q) q->encode_verify(*e, *s, P, R, sp, uint32_t(P), /*v8=*/true);
        else nh->encode_verify(*e, *s, P, R);
        e->commit_and_wait();
        v.push_back(e->gpu_ms());
        if (q) q->commit_rows(*s, 0);
        else nh->commit_rows(*s, 0);
      }
      ms[kind] = median(v);
    }
    printf("  R=%d  %7.2f ms (%.2f per row) | %7.2f ms\n", R, ms[0], ms[0] / R, ms[1]);
  }
  // Numerics: the logits of rows 0..3 (the continuation) verified (a) with the continuation's rows 4..7, (b) with
  // unrelated rows 4..7 (another request's rows in the same pass), (c) as a 4-row pass. Rows are causal, so 0..3 see
  // the same inputs every time: a difference is the pass's own arithmetic (expert grouping, attention split).
  auto logits4 = [&](int R, bool other) {
    for (int r = 0; r < R; ++r) s->tokens()[P + r] = r >= 4 && other ? any(rng) : cont[size_t(r)];
    auto e = model->grouped_passes() ? g.begin_grouped() : g.begin();
    if (q) q->encode_verify(*e, *s, P, R, sp, uint32_t(P), /*v8=*/true);
    else nh->encode_verify(*e, *s, P, R);
    e->commit_and_wait();
    const float* l = q ? q->logits() : nh->logits();
    std::vector<float> out(l, l + size_t(4) * V);
    if (q) q->commit_rows(*s, 0);
    else nh->commit_rows(*s, 0);
    return out;
  };
  const std::vector<float> a = logits4(8, false), b = logits4(8, true), c = logits4(4, false);
  auto cmp = [&](const std::vector<float>& x, const char* what) {
    double md = 0;
    int argmax_same = 0, bitwise = 1;
    for (int r = 0; r < 4; ++r) {
      const float* p = a.data() + size_t(r) * V;
      const float* o = x.data() + size_t(r) * V;
      int ia = 0, io = 0;
      for (int v = 0; v < V; ++v) {
        md = std::max(md, double(std::fabs(p[v] - o[v])));
        bitwise &= p[v] == o[v];
        if (p[v] > p[ia]) ia = v;
        if (o[v] > o[io]) io = v;
      }
      argmax_same += ia == io;
    }
    printf("  rows 0-3 %s: %s, max |logit diff| %.4g, argmax equal in %d of 4 rows\n", what,
           bitwise ? "bitwise equal" : "differ", md, argmax_same);
  };
  printf("numerics of rows 0-3 against the 8-row pass of the sequence's own continuation:\n");
  cmp(b, "with another request's rows 4-7 in the pass");
  cmp(c, "as a 4-row pass");
  return 0;
}
