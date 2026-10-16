// Runs a prompt (token ids) through the engine and generates tokens.
//   ni_run <model_dir> <ids: comma-separated> [n_gen=32] [temp=0] [--dump-logits file]
//          [--spec D] [--dflash <drafter dir>]   (NI_DFLASH=<dir> works too)
// Prints generated ids, prompt/decode speeds; optionally dumps the logits of the first
// generated position (raw float32) for comparison against a reference implementation.
#include "metal/gpu.h"
#include "model/dflash2.h"
#include "model/qwen35.h"
#include "util/log.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: ni_run <model_dir> <ids> [n_gen] [temp] [--dump-logits f]\n");
    return 1;
  }
  std::string dir = argv[1];
  std::vector<int> ids;
  {
    std::string arg = argv[2];
    if (!arg.empty() && arg[0] == '@') {  // @file with comma-separated ids
      std::ifstream f(arg.substr(1));
      arg.assign(std::istreambuf_iterator<char>(f), {});
    }
    std::stringstream ss(arg);
    std::string t;
    while (std::getline(ss, t, ',')) ids.push_back(std::stoi(t));
  }
  int n_gen = argc > 3 ? atoi(argv[3]) : 32;
  float temp = argc > 4 ? float(atof(argv[4])) : 0.0f;
  const char* dump = nullptr;
  bool tbt = false;
  int spec = 0;
  for (int i = 1; i < argc; ++i) {
    if (i + 1 < argc && !strcmp(argv[i], "--dump-logits")) dump = argv[i + 1];
    if (!strcmp(argv[i], "--tbt")) tbt = true;
    if (i + 1 < argc && !strcmp(argv[i], "--spec")) spec = atoi(argv[i + 1]);
  }
  std::string dflash_dir = getenv("NI_DFLASH") ? getenv("NI_DFLASH") : "";
  for (int i = 1; i + 1 < argc; ++i)
    if (!strcmp(argv[i], "--dflash")) dflash_dir = argv[i + 1];

  Gpu& g = Gpu::get();
  Qwen35 m(dir, getenv("NI_CHUNK") ? atoi(getenv("NI_CHUNK")) : 2048);
  auto seq = m.new_seq(std::max(8192, int(ids.size()) + n_gen + 64));
  Seq& s = *seq;
  SamplingParams sp;
  sp.temperature = temp;
  sp.top_p = 0.95f;
  sp.top_k = 20;
  sp.seed = 42;

  int n = int(ids.size());
  for (int i = 0; i < n; ++i) s.tokens()[i] = ids[i];

  // Opt-in DFlash2 block drafter: the target's verify (currently <= Qwen35::kMaxVerify rows)
  // checks the first NI_DFLASH_VERIFY (default kMaxVerify - 1) of the 7 proposals.
  std::unique_ptr<DFlash2> dfl;
  std::unique_ptr<DFlashSeq> dseq;
  if (!dflash_dir.empty()) {
    dfl = std::make_unique<DFlash2>(m, dflash_dir);
    dseq = dfl->new_seq();
    dfl->begin_prefill(int(ids.size()));
  }

  int last_chunk = 0;
  double t0 = now_ms();
  if (Encoder::profiling()) fprintf(stderr, "(profiling: prompt timings not meaningful)\n");
  if (tbt) {
    // token by token through the decode path (reference for the batched prefill)
    for (int p = 0; p < n; p += 32) {
      auto e = g.begin();
      for (int q = p; q < std::min(n, p + 32); ++q) m.encode_step(*e, s, q, sp, 0, q == n - 1);
      e->commit_and_wait();
    }
  } else {
    int C = m.prefill_chunk();
    for (int p = 0; p < n; p += C) {
      int T = std::min(C, n - p);
      auto e = g.begin();
      m.encode_prefill(*e, s, p, T, sp, 0, p + T == n);
      // MTP history rows i in [p, min(p+T, n-1)): pairs (h_i, t_{i+1})
      if (spec && m.has_mtp()) m.encode_mtp_prefill(*e, s, p, std::min(p + T, n - 1) - p, 0);
      e->commit_and_wait();
      last_chunk = p;
    }
  }
  if (dfl) {
    dfl->end_prefill();
    auto e = g.begin();
    dfl->encode_prefill_commit(*e, *dseq);
    e->commit_and_wait();
  }
  double t_prompt = now_ms() - t0;
  if (dump) {
    FILE* f = fopen(dump, "wb");
    fwrite(m.logits(), 4, m.config().vocab, f);
    fclose(f);
  }
  std::vector<int> out;
  out.push_back(s.tokens()[n]);

  if (dfl) {
    const int K = std::clamp(getenv("NI_DFLASH_VERIFY") ? atoi(getenv("NI_DFLASH_VERIFY")) : Qwen35::kMaxVerify - 1, 1,
                             Qwen35::kMaxVerify - 1);
    double ts = now_ms();
    int P = n, m_prev = 0, cycles = 0;
    std::vector<int> acc_hist(K + 1, 0);
    double gpu_sum = 0;
    bool stop2 = false;
    if (Encoder::profiling()) Encoder::profile_report(1);
    while (int(out.size()) < n_gen && !stop2) {
      double tc0 = now_ms();
      auto e = g.begin();
      if (m_prev) dfl->encode_commit(*e, *dseq, m_prev);
      dfl->encode_draft(*e, s, *dseq, P, sp);
      dfl->arm_verify(P);
      m.encode_verify(*e, s, P, K + 1, sp, uint32_t(P));
      dfl->disarm();
      e->commit_and_wait();
      gpu_sum += e->gpu_ms();
      const int* y = m.samples();
      int* tk = s.tokens();
      int a = 0;
      while (a < K && tk[P + 1 + a] == y[a]) ++a;
      if (getenv("NI_TRACE"))
        fprintf(stderr, "P=%d draft=%d,%d,%d y0=%d a=%d wall=%.1fms gpu=%.1fms\n", P, tk[P + 1], tk[P + 2], tk[P + 3],
                y[0], a, now_ms() - tc0, e->gpu_ms());
      tk[P + a + 1] = y[a];
      for (int i = 1; i <= a + 1 && int(out.size()) < n_gen; ++i) {
        out.push_back(tk[P + i]);
        for (int eos : m.config().eos)
          if (tk[P + i] == eos) stop2 = true;
        if (stop2) break;
      }
      m.commit_rows(s, a + 1);
      m_prev = a + 1;
      ++cycles;
      acc_hist[a]++;
      P += a + 1;
    }
    double td = now_ms() - ts;
    if (Encoder::profiling()) Encoder::profile_report(cycles);
    printf("ids:");
    for (int t : out) printf(" %d", t);
    printf("\n");
    fprintf(stderr, "prompt %d tok in %.0f ms (%.1f tok/s); dflash decode (verify %d of 7): %zu tok in %.0f ms (%.1f tok/s), "
            "%d cycles, %.2f tok/cycle, %.2f ms GPU/cycle, hist",
            n, t_prompt, n / (t_prompt / 1000), K, out.size() - 1, td, (out.size() - 1) / (td / 1000), cycles,
            double(out.size() - 1) / cycles, gpu_sum / cycles);
    for (int a = 0; a <= K; ++a) fprintf(stderr, " %d", acc_hist[a]);
    fprintf(stderr, "\n");
    return 0;
  }

  if (spec && m.has_mtp()) {
    const int D = spec;
    double ts = now_ms();
    int P = n, c = 1, h_row0 = n - 1 - last_chunk;
    int cycles = 0, accepted = 0;
    if (Encoder::profiling()) Encoder::profile_report(1);  // drop prefill records
    std::vector<int> acc_hist(D + 1, 0);
    bool stop2 = false;
    while (int(out.size()) < n_gen && !stop2) {
      double tc0 = now_ms();
      auto e = g.begin();
      m.encode_mtp(*e, s, P - c, c, P - c + 1, m.hidden_rows(), h_row0, P + 1);
      for (int k = 2; k <= D; ++k)
        m.encode_mtp(*e, s, P + k - 2, 1, P + k - 1, m.mtp_post(), k == 2 ? c - 1 : 0, P + k, false);
      m.encode_verify(*e, s, P, D + 1, sp, uint32_t(P));
      e->commit_and_wait();
      const int* y = m.samples();
      int* tk = s.tokens();
      int a = 0;
      while (a < D && tk[P + 1 + a] == y[a]) ++a;
      if (getenv("NI_TRACE"))
        fprintf(stderr, "P=%d c=%d draft=%d y0=%d a=%d wall=%.1fms gpu=%.1fms\n", P, c, tk[P + 1], y[0], a,
                now_ms() - tc0, e->gpu_ms());
      tk[P + a + 1] = y[a];
      for (int i = 1; i <= a + 1 && int(out.size()) < n_gen; ++i) {
        out.push_back(tk[P + i]);
        for (int eos : m.config().eos)
          if (tk[P + i] == eos) stop2 = true;
        if (stop2) break;
      }
      m.commit_rows(s, a + 1);
      ++cycles;
      accepted += a;
      acc_hist[a]++;
      c = a + 1;
      h_row0 = 0;
      P += a + 1;
    }
    double td = now_ms() - ts;
    if (Encoder::profiling()) Encoder::profile_report(cycles);
    printf("ids:");
    for (int t : out) printf(" %d", t);
    printf("\n");
    fprintf(stderr, "prompt %d tok in %.0f ms (%.1f tok/s); spec decode D=%d: %zu tok in %.0f ms (%.1f tok/s), %d cycles, %.2f tok/cycle, accepted/drafted %.2f, hist",
            n, t_prompt, n / (t_prompt / 1000), D, out.size() - 1, td, (out.size() - 1) / (td / 1000), cycles,
            double(out.size() - 1) / cycles, double(accepted) / (cycles * D));
    for (int a = 0; a <= D; ++a) fprintf(stderr, " %d", acc_hist[a]);
    fprintf(stderr, "\n");
    return 0;
  }

  // Decode: keep up to 2 command buffers in flight; the GPU writes each sampled id into the
  // token buffer, which the next step's embedding reads, so the CPU never waits per token.
  double t1 = now_ms();
  std::deque<std::unique_ptr<Encoder>> inflight;
  int pos = n;
  int done = 1;
  bool stop = false;
  double gpu_total = 0;
  int gpu_n = 0;
  for (int step = 1; step < n_gen && !stop; ++step) {
    auto e = g.begin();
    m.encode_step(*e, s, pos, sp, uint32_t(step), true);
    e->commit();
    inflight.push_back(std::move(e));
    ++pos;
    while (inflight.size() > 2 || (step == n_gen - 1 && !inflight.empty())) {
      inflight.front()->wait();
      gpu_total += inflight.front()->gpu_ms();
      ++gpu_n;
      inflight.pop_front();
      int tok = s.tokens()[n + done];
      out.push_back(tok);
      ++done;
      for (int eos : m.config().eos)
        if (tok == eos) stop = true;
    }
  }
  while (!inflight.empty()) {
    inflight.front()->wait();
    inflight.pop_front();
    out.push_back(s.tokens()[n + done]);
    ++done;
  }
  double t_dec = now_ms() - t1;
  if (Encoder::profiling()) Encoder::profile_report(int(out.size()) - 1);
  if (gpu_n) fprintf(stderr, "mean GPU time per step: %.2f ms (%d steps)\n", gpu_total / gpu_n, gpu_n);
  printf("ids:");
  for (int t : out) printf(" %d", t);
  printf("\n");
  fprintf(stderr, "prompt %d tok in %.0f ms (%.1f tok/s); decode %zu tok in %.0f ms (%.1f tok/s)\n", n, t_prompt,
          n / (t_prompt / 1000), out.size() - 1, t_dec, (out.size() - 1) / (t_dec / 1000));
  return 0;
}
