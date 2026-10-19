// In-pipeline timing of the 8-row verify on a real model, teacher-forced on its own greedy continuation (as
// ni_v8check): the prompt is prefilled, n_tok tokens are decoded plainly, then a fresh sequence verifies them R at a
// time; each block is verified `reps` times per variant before its rows are committed, `inner` verifies back to back
// per command buffer (an uncommitted verify can be repeated: the same work with the real routing).
//   bench_verify8 <model_dir> @ids [n_tok=64] [reps=3] [inner=4] [R=8]
// BENCH_AB=key=val,... (V8Knobs keys, e.g. emu=0): every block alternates command buffers with the defaults (A) and
// those knobs (B), A B A B ... (interleaved A/B under the same GPU conditions). BENCH_EXACT=1: sampled verifies' MoE numerics
// (Qwen35::set_verify_moe_exact). NI_PROFILE=1 works as in ni_v8check (per-kernel isolated times).
#include "metal/gpu.h"
#include "model/qwen35.h"
#include "util/log.h"

#include <algorithm>
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
    fprintf(stderr, "usage: bench_verify8 <model_dir> @ids [reps] [inner] [R]\n");
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
  const int n_tok = argc > 3 ? atoi(argv[3]) : 64, reps = argc > 4 ? atoi(argv[4]) : 3,
            inner = argc > 5 ? atoi(argv[5]) : 4, R = argc > 6 ? atoi(argv[6]) : 8;
  Gpu& g = Gpu::get();
  Qwen35 m(argv[1], 2048);
  SamplingParams sp;
  const int n = int(ids.size()), cap = n + n_tok + 64;
  auto prefill = [&](Seq& s) {
    for (int i = 0; i < n; ++i) s.tokens()[i] = ids[i];
    for (int p = 0; p < n; p += m.prefill_chunk()) {
      const int T = std::min(m.prefill_chunk(), n - p);
      auto e = g.begin();
      m.encode_prefill(*e, s, p, T, sp, 0, p + T == n);
      e->commit_and_wait();
      if (!m.ane_finish_chunk()) fatal("the NI_ANE prefill split failed; rerun with NI_ANE=0");
    }
  };
  auto s1 = m.new_seq(cap);
  prefill(*s1);
  for (int i = 0; i < n_tok; ++i) {
    auto e = g.begin();
    m.encode_step(*e, *s1, n + i, sp, uint32_t(i + 1), true);
    e->commit_and_wait();
  }
  if (getenv("BENCH_EXACT") && atoi(getenv("BENCH_EXACT")) > 0) m.set_verify_moe_exact(true);
  auto s = m.new_seq(cap);
  prefill(*s);
  for (int i = n; i <= n + n_tok; ++i) s->tokens()[i] = s1->tokens()[i];
  if (Encoder::profiling()) {
    fprintf(stderr, "==== (prefill and plain decode, ignore) ====\n");
    Encoder::profile_report(1);
  }
  const V8Knobs ka = v8_knobs();
  const bool ab = getenv("BENCH_AB") && *getenv("BENCH_AB");
  const V8Knobs kb = ab ? v8_knobs_parse(getenv("BENCH_AB"), ka) : ka;
  std::vector<double> ms[2];
  double ustat = 0;
  int nblk = 0;
  for (int P = n; P + R <= n + n_tok; P += R, ++nblk) {
    for (int r = -1; r < reps; ++r)  // one warm-up pair per block
      for (int v = 0; v < (ab && !getenv("BENCH_REPLAY") ? 2 : 1); ++v) {
        v8_knobs() = v ? kb : ka;
        auto e = g.begin();
        for (int i = 0; i < inner; ++i) m.encode_verify(*e, *s, P, R, sp, uint32_t(P));
        e->commit_and_wait();
        if (r >= 0) ms[v].push_back(e->gpu_ms() / inner);
      }
    if (const uint32_t* u = m.verify8_experts()) {
      double su = 0;
      for (int l = 0; l < m.config().layers; ++l) su += u[l];
      ustat += su / m.config().layers;
    }
    m.commit_rows(*s, R);
  }
  if (const char* rp = getenv("BENCH_REPLAY")) {  // the expert passes alone: the last block's plan, every layer's weights
    const int mask = atoi(rp), L = m.config().layers;
    for (int v = 0; v < 2; ++v) ms[v].clear();
    for (int r = -2; r < reps * 8; ++r)
      for (int v = 0; v < (ab ? 2 : 1); ++v) {
        v8_knobs() = v ? kb : ka;
        auto e = getenv("BENCH_GROUPED") ? g.begin_grouped() : g.begin();
        for (int i = 0; i < inner; ++i)
          for (int li = 0; li < L; ++li)
            if (m.config().moe) m.moe8_replay(*e, li, mask);
        e->commit_and_wait();
        if (r >= 0) ms[v].push_back(e->gpu_ms() / inner);
      }
    printf("replay mask %d (ms per %d layers):\n", mask, L);
  }
  v8_knobs() = ka;
  if (Encoder::profiling()) {
    Encoder::profile_report(nblk * (reps + 1) * inner * (ab ? 2 : 1));
    return 0;
  }
  for (int v = 0; v < (ab ? 2 : 1); ++v) {
    std::vector<double> x = ms[v];
    std::sort(x.begin(), x.end());
    double sum = 0;
    for (double y : x) sum += y;
    printf("%-24s verify R=%d %.3f ms median, %.3f mean, %.3f min (%zu x %d)\n", v ? getenv("BENCH_AB") : "base", R,
           x[x.size() / 2], sum / x.size(), x[0], x.size(), inner);
  }
  printf("MoE: %.1f distinct routed experts per layer and verify (%d blocks)\n", ustat / std::max(nblk, 1), nblk);
  if (ab) {
    std::vector<double> d;
    for (size_t i = 0; i < std::min(ms[0].size(), ms[1].size()); ++i) d.push_back(ms[1][i] - ms[0][i]);
    std::sort(d.begin(), d.end());
    printf("B - A: median %+.3f ms per verify\n", d[d.size() / 2]);
  }
  return 0;
}
