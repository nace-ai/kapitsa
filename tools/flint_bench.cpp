// Flint timing: prefill of a prompt from bench/corpus.txt, then plain greedy decoding, each pass one command buffer as
// the engine's decode loop commits them (the next step committed before the host waits on the last one).
//   flint_bench <model_dir> [--prompt N] [--steps N] [--chunk C] [--reps R] [--rows V]
// --rows V (2-8): each decode step a verify pass of V rows instead (its cost; nothing is accepted)
// NI_PROFILE=1: every dispatch in its own command buffer, the decode steps' GPU time per kernel (Encoder::profile_report).
// Takes the GPU lock ($BENCH_GPU_LOCK, default ~/.cache/nace-gpu.lock) unless NI_LOCK_HELD=1 or NACE_GPU_LOCK_HELD=1.
#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/flint.h"
#include "tokenizer/tokenizer.h"
#include "util/log.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: flint_bench <model_dir> [--prompt N] [--steps N] [--chunk C] [--reps R]\n");
    return 1;
  }
  const std::string dir = argv[1];
  int prompt = 1024, steps = 256, chunk = 2048, reps = 3, vrows = 1;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    const int v = atoi(argv[i + 1]);
    if (a == "--prompt") prompt = v;
    else if (a == "--steps") steps = v;
    else if (a == "--chunk") chunk = v;
    else if (a == "--reps") reps = v;
    else if (a == "--rows") vrows = v;
    else fatal("unknown option %s", a.c_str());
  }
  std::ifstream cf("bench/corpus.txt");
  std::stringstream ss;
  ss << cf.rdbuf();
  Tokenizer tok;
  tok.load(dir + "/tokenizer.json");
  std::vector<int> ids = tok.encode(ss.str());
  while (int(ids.size()) < prompt) ids.insert(ids.end(), ids.begin(), ids.end());
  ids.resize(size_t(prompt));

  bench::GpuLock lock;
  if (!env_on("NI_LOCK_HELD") && !env_on("NACE_GPU_LOCK_HELD") &&
      !lock.acquire([](const std::string& who) {
        fprintf(stderr, "waiting for the GPU lock%s\n", who.empty() ? "" : (", held by " + who).c_str());
      }))
    fatal("cannot take the GPU lock: %s", strerror(errno));
  Gpu& g = Gpu::get();
  Flint m(dir, chunk);
  auto s = m.new_seq(prompt + steps + 64 + vrows);
  std::copy(ids.begin(), ids.end(), s->tokens());
  const SamplingParams greedy;
  const bool prof = Encoder::profiling();
  for (int rep = 0; rep < reps; ++rep) {
    m.reset(*s);
    const double t0 = Encoder::host_ms();
    double gpu_pf = 0;
    for (int p0 = 0; p0 < prompt; p0 += chunk) {
      const int T = std::min(chunk, prompt - p0);
      auto e = m.grouped_passes() ? g.begin_grouped() : g.begin();
      m.encode_prefill(*e, *s, p0, T, greedy, 0, p0 + T == prompt);
      e->commit_and_wait();
      gpu_pf += e->gpu_ms();
    }
    const double t1 = Encoder::host_ms();
    if (prof) Encoder::profile_report(1);  // (the prefill's dispatches, then the decode's alone)
    // decode: step i+1 committed before waiting on step i, as the engine's plain loop does
    std::unique_ptr<Encoder> prev;
    double gpu_dec = 0;
    for (int i = 0; i < steps; ++i) {
      const int pos = prompt + i;
      auto e = m.grouped_passes() ? g.begin_grouped() : g.begin();
      if (vrows > 1) m.encode_verify(*e, *s, pos, vrows);
      else m.encode_step(*e, *s, pos, greedy, uint32_t(pos), true);
      e->commit();
      if (prev) {
        prev->wait();
        gpu_dec += prev->gpu_ms();
      }
      prev = std::move(e);
    }
    prev->wait();
    gpu_dec += prev->gpu_ms();
    const double t2 = Encoder::host_ms();
    if (prof) {
      Encoder::profile_report(steps);
      continue;
    }
    printf("rep %d: prefill %d tokens %.1f ms (%.0f tok/s, GPU %.1f ms); decode %d steps %.1f ms = %.2f ms/step "
           "(%.1f tok/s; GPU %.2f ms/step)\n",
           rep, prompt, t1 - t0, prompt / (t1 - t0) * 1000, gpu_pf, steps, t2 - t1, (t2 - t1) / steps,
           steps / (t2 - t1) * 1000, gpu_dec / steps);
    fflush(stdout);
  }
  return 0;
}
