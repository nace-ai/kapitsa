// Batched decode check: N sequences verified one at a time (encode_verify, the single-sequence pass) against the
// same rows of all of them in one encode_verify_multi pass. Two identical sets of sequences (the same prompts,
// prefilled the same way) take the same teacher-forced tokens; each cycle every sequence verifies R rows and keeps
// m = 1 + (cycle mod R) of them (the rest of its tape is dropped, as after a partial acceptance), so the pending-row
// folding is checked too. Reports, over all rows: how many logit rows are bitwise equal, the largest |logit diff|,
// how often the argmax agrees, and the GPU time of the single passes (summed) against the batched one.
//   batch_check <model_dir> @ids [N=2] [R=1] [cycles=16] [prompt=256]
// (Qwen3.5-family or Nemotron-H; NI_ANE=0 keeps the prefill on the GPU)
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
  const int N = argc > 3 ? std::clamp(atoi(argv[3]), 1, 8) : 2;
  const int R = argc > 4 ? std::clamp(atoi(argv[4]), 1, 8) : 1;
  const int cycles = argc > 5 ? std::max(1, atoi(argv[5])) : 16;
  const int plen = argc > 6 ? std::max(8, atoi(argv[6])) : 256;
  if (N * R > 8) fatal("N x R = %d rows > 8", N * R);
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
    const int m = 1 + c % R;
    for (int i = 0; i < N; ++i) {  // A: one sequence at a time
      auto e = begin();
      if (q) q->encode_verify(*e, *A[size_t(i)], P[size_t(i)], R, sp, uint32_t(P[size_t(i)]), /*v8=*/true);
      else nh->encode_verify(*e, *A[size_t(i)], P[size_t(i)], R);
      e->commit_and_wait();
      t_single += e->gpu_ms();
      const float* l = q ? q->logits() : nh->logits();
      std::copy(l, l + size_t(R) * V, ref.begin() + long(size_t(i) * R * V));
    }
    {  // B: all of them in one pass (grouped, as the engine encodes it: each stage's per-sequence dispatches side by
       // side; NI_BATCH_SERIAL=1: a serial encoder)
      static const bool serial = getenv("NI_BATCH_SERIAL") && atoi(getenv("NI_BATCH_SERIAL")) != 0;
      auto e = serial ? begin() : g.begin_grouped();
      if (q) {
        std::vector<Qwen35::VerifyGroup> gr(static_cast<size_t>(N));
        for (int i = 0; i < N; ++i) gr[size_t(i)] = Qwen35::VerifyGroup{B[size_t(i)].get(), P[size_t(i)], R, sp};
        q->encode_verify_multi(*e, gr);
      } else {
        std::vector<NemotronH::VerifyGroup> gr(static_cast<size_t>(N));
        for (int i = 0; i < N; ++i) gr[size_t(i)] = NemotronH::VerifyGroup{B[size_t(i)].get(), P[size_t(i)], R, sp};
        nh->encode_verify_multi(*e, gr);
      }
      e->commit_and_wait();
      t_multi += e->gpu_ms();
    }
    const float* l = q ? q->logits() : nh->logits();
    for (int r = 0; r < N * R; ++r) {
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
  printf("%s: %d sequences x %d rows, %d cycles (prompts of %d tokens): %ld rows, %ld bitwise equal, max |logit diff| "
         "%.4g, argmax equal in %ld\n", argv[1], N, R, cycles, plen, rows, bitwise, maxdiff, argmax_same);
  printf("GPU ms per cycle: %.2f for the %d single passes, %.2f for the batched pass\n", t_single / cycles, N,
         t_multi / cycles);
  return 0;
}
