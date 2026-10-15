// Runs a prompt (token ids) through the engine and generates tokens.
//   ni_run <model_dir> <ids: comma-separated> [n_gen=32] [temp=0] [--dump-logits file]
// Prints generated ids, prompt/decode speeds; optionally dumps the logits of the first
// generated position (raw float32) for comparison against a reference implementation.
#include "metal/gpu.h"
#include "model/qwen35.h"
#include "util/log.h"

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
  for (int i = 1; i < argc; ++i) {
    if (i + 1 < argc && !strcmp(argv[i], "--dump-logits")) dump = argv[i + 1];
    if (!strcmp(argv[i], "--tbt")) tbt = true;
  }

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
      e->commit_and_wait();
    }
  }
  double t_prompt = now_ms() - t0;
  if (dump) {
    FILE* f = fopen(dump, "wb");
    fwrite(m.logits(), 4, m.config().vocab, f);
    fclose(f);
  }
  std::vector<int> out;
  out.push_back(s.tokens()[n]);

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
