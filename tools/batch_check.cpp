// Batched decode check: N sequences verified one at a time (encode_verify, the single-sequence pass) against the
// same rows of all of them in one encode_verify_multi pass. Two identical sets of sequences (the same prompts,
// prefilled the same way) take the same teacher-forced tokens; each cycle every sequence verifies R rows and keeps
// m = 1 + (cycle mod R) of them (the rest of its tape is dropped, as after a partial acceptance), so the pending-row
// folding is checked too. Reports, over all rows: how many logit rows are bitwise equal, the largest |logit diff|,
// how often the argmax agrees, and the GPU time of the single passes (summed) against the batched one.
//   batch_check <model_dir> @ids [N=2] [R=1] [cycles=16] [prompt=256]
// (Qwen3.5-family or Nemotron-H; NI_ANE=0 keeps the prefill on the GPU)
// R may instead give each sequence its own rows, as the batch budget (KfBatch) does, one layout per cycle in turn:
// "5,2,1;1,7;4,4" (layouts of N entries each, N then ignored; at most 8 rows a layout). NI_BATCH_SPLIT=1: the batched
// pass's first layer is a command buffer of its own (encode_verify_multi's split_after, as the batch budget runs it).
#include "metal/gpu.h"
#include "model/nemotron_h.h"
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

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: batch_check <model_dir> @ids [N=2] [R=1] [cycles=16] [prompt=256]\n");
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
  int N = argc > 3 ? std::clamp(atoi(argv[3]), 1, 8) : 2;
  // layouts: rows per sequence, one layout per cycle in turn (a plain R: every sequence R rows)
  std::vector<std::vector<int>> lay;
  {
    const std::string a = argc > 4 ? argv[4] : "1";
    if (a.find_first_of(",;") != std::string::npos) {
      std::stringstream ls(a);
      std::string one;
      while (std::getline(ls, one, ';')) {
        std::vector<int> v;
        std::stringstream rs(one);
        std::string t;
        while (std::getline(rs, t, ',')) v.push_back(std::clamp(std::stoi(t), 1, 8));
        if (!v.empty()) lay.push_back(v);
      }
      if (lay.empty()) fatal("no layout in \"%s\"", a.c_str());
      N = int(lay[0].size());
      for (const auto& v : lay) {
        int sum = 0;
        for (int r : v) sum += r;
        if (int(v.size()) != N || sum > 8) fatal("layouts need %d sequences and at most 8 rows each", N);
      }
    } else {
      lay.push_back(std::vector<int>(size_t(N), std::clamp(atoi(a.c_str()), 1, 8)));
      if (N * lay[0][0] > 8) fatal("N x R = %d rows > 8", N * lay[0][0]);
    }
  }
  int R = 1;  // the most rows a sequence verifies in a cycle
  for (const auto& v : lay)
    for (int r : v) R = std::max(R, r);
  const int cycles = argc > 5 ? std::max(1, atoi(argv[5])) : 16;
  const int plen = argc > 6 ? std::max(8, atoi(argv[6])) : 256;
  static const int split_after = getenv("NI_BATCH_SPLIT") && atoi(getenv("NI_BATCH_SPLIT")) ? 1 : 0;
  const int span = plen + cycles * R + 8;  // tokens a sequence reads: its prompt, then the teacher-forced rows
  if (int(ids.size()) < N * span) fatal("%zu ids for %d sequences of %d tokens", ids.size(), N, span);
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
  const int V = model->vocab();
  auto begin = [&] { return model->grouped_passes() ? g.begin_grouped() : g.begin(); };
  // two identical sets: A verifies one sequence at a time, B all of them in one pass
  std::vector<std::unique_ptr<Seq>> A, B;
  std::vector<int> P(size_t(N), plen - 1);
  SamplingParams sp;  // greedy
  for (int set = 0; set < 2; ++set)
    for (int i = 0; i < N; ++i) {
      auto s = model->new_seq(plen + cycles * R + 64);
      for (int t = 0; t < span; ++t) s->tokens()[t] = ids[size_t(i) * span + t];
      // the prompt but its last token (the first verify's anchor)
      for (int p = 0, C = model->prefill_chunk(); p < plen - 1; p += C) {
        const int T = std::min(C, plen - 1 - p);
        auto e = g.begin();
        model->encode_prefill(*e, *s, p, T, sp, 0, false);
        e->commit_and_wait();
        if (!model->ane_finish_chunk()) fatal("the Neural Engine prefill split failed; rerun with NI_ANE=0");
      }
      (set == 0 ? A : B).push_back(std::move(s));
    }
  std::vector<float> ref(size_t(N) * R * V);
  long rows = 0, bitwise = 0, argmax_same = 0;
  double maxdiff = 0, t_single = 0, t_multi = 0;
  auto argmax = [&](const float* l) { return int(std::max_element(l, l + V) - l); };
  for (int c = 0; c < cycles; ++c) {
    const std::vector<int>& Rs = lay[size_t(c) % lay.size()];
    std::vector<size_t> r0(size_t(N), 0);  // each sequence's first row in the batched pass
    for (int i = 1; i < N; ++i) r0[size_t(i)] = r0[size_t(i - 1)] + size_t(Rs[size_t(i - 1)]);
    for (int i = 0; i < N; ++i) {  // A: one sequence at a time
      const int Ri = Rs[size_t(i)];
      auto e = begin();
      if (q) q->encode_verify(*e, *A[size_t(i)], P[size_t(i)], Ri, sp, uint32_t(P[size_t(i)]), /*v8=*/true);
      else nh->encode_verify(*e, *A[size_t(i)], P[size_t(i)], Ri);
      e->commit_and_wait();
      t_single += e->gpu_ms();
      const float* l = q ? q->logits() : nh->logits();
      std::copy(l, l + size_t(Ri) * V, ref.begin() + long(r0[size_t(i)] * V));
    }
    {  // B: all of them in one pass (grouped, as the engine encodes it: each stage's per-sequence dispatches side by
       // side; NI_BATCH_SERIAL=1: a serial encoder)
      static const bool serial = getenv("NI_BATCH_SERIAL") && atoi(getenv("NI_BATCH_SERIAL")) != 0;
      auto e = serial ? begin() : g.begin_grouped();
      if (q) {
        std::vector<Qwen35::VerifyGroup> gr(static_cast<size_t>(N));
        for (int i = 0; i < N; ++i) gr[size_t(i)] = Qwen35::VerifyGroup{B[size_t(i)].get(), P[size_t(i)], Rs[size_t(i)], sp};
        q->encode_verify_multi(*e, gr, split_after);
      } else {
        std::vector<NemotronH::VerifyGroup> gr(static_cast<size_t>(N));
        for (int i = 0; i < N; ++i)
          gr[size_t(i)] = NemotronH::VerifyGroup{B[size_t(i)].get(), P[size_t(i)], Rs[size_t(i)], sp};
        nh->encode_verify_multi(*e, gr);
      }
      e->commit_and_wait();
      t_multi += e->gpu_ms();
    }
    const float* l = q ? q->logits() : nh->logits();
    const int nrows = int(r0[size_t(N - 1)]) + Rs[size_t(N - 1)];
    for (int r = 0; r < nrows; ++r) {
      const float* a = ref.data() + size_t(r) * V;
      const float* b = l + size_t(r) * V;
      double md = 0;
      bool bw = true;
      for (int v = 0; v < V; ++v) {
        md = std::max(md, double(std::fabs(a[v] - b[v])));
        bw &= a[v] == b[v];
      }
      ++rows;
      bitwise += bw;
      argmax_same += argmax(a) == argmax(b);
      maxdiff = std::max(maxdiff, md);
    }
    for (int i = 0; i < N; ++i) {  // keep m rows of each, as after a partial acceptance
      const int m = 1 + c % Rs[size_t(i)];
      if (q) {
        q->commit_rows(*A[size_t(i)], m);
        q->commit_rows(*B[size_t(i)], m);
      } else {
        nh->commit_rows(*A[size_t(i)], m);
        nh->commit_rows(*B[size_t(i)], m);
      }
      P[size_t(i)] += m;
    }
  }
  printf("%s: %d sequences x %s rows, %d cycles (prompts of %d tokens): %ld rows, %ld bitwise equal, max |logit diff| "
         "%.4g, argmax equal in %ld\n", argv[1], N, argc > 4 ? argv[4] : "1", cycles, plen, rows, bitwise, maxdiff,
         argmax_same);
  printf("GPU ms per cycle: %.2f for the %d single passes, %.2f for the batched pass\n", t_single / cycles, N,
         t_multi / cycles);
  return 0;
}
