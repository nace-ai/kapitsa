// Hy4-flash checks against a reference: every row's logits of a prompt, and greedy tokens after it.
//   hy4_dump <model> --ids FILE [--out F.f32] [--rows N] [--gen N] [--chunk R]
// FILE: comma- or whitespace-separated token ids. --out writes float32 [n, vocab] for rows 0..n-1 (n = --rows, default
// every prompt row): row p is the distribution of token p + 1. --gen N: N greedy tokens after the prompt (decode
// passes), printed as ids with tok/s. Takes the GPU lock (BENCH_GPU_LOCK, default /tmp/kapitsa-gpu.lock) unless the
// caller holds it (NI_LOCK_HELD=1).
#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/hy4_flash.h"
#include "util/log.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;

static std::vector<int> read_ids(const std::string& path) {
  std::ifstream f(path);
  if (!f) fatal("cannot read %s", path.c_str());
  std::string s((std::istreambuf_iterator<char>(f)), {});
  for (char& c : s)
    if (c == ',' || c == '[' || c == ']') c = ' ';
  std::stringstream ss(s);
  std::vector<int> v;
  int x;
  while (ss >> x) v.push_back(x);
  if (!ss.eof()) {  // a word that is not an integer: refuse it rather than drop the rest of the prompt
    ss.clear();
    std::string w;
    ss >> w;
    fatal("%s: \"%s\" (after %zu ids) is not a token id", path.c_str(), w.c_str(), v.size());
  }
  return v;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: hy4_dump <model> --ids FILE [--out F.f32] [--rows N] [--gen N] [--chunk R]\n");
    return 1;
  }
  std::string dir = argv[1], ids_file, out;
  int rows = -1, gen = 0, chunk = 512;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) fatal("%s needs a value", a.c_str());
      return argv[++i];
    };
    if (a == "--ids") ids_file = next();
    else if (a == "--out") out = next();
    else if (a == "--rows") rows = atoi(next().c_str());
    else if (a == "--gen") gen = atoi(next().c_str());
    else if (a == "--chunk") chunk = atoi(next().c_str());
    else fatal("unknown argument %s", a.c_str());
  }
  const std::vector<int> ids = read_ids(ids_file);
  const int n = int(ids.size());
  if (n < 1) fatal("no ids");
  if (rows < 0 || rows > n) rows = n;

  if (chunk < 1) fatal("--chunk %d: a pass needs at least 1 row", chunk);
  if (gen < 0) fatal("--gen %d: the number of tokens to generate is 0 or more", gen);

  bench::GpuLock lock;  // released at exit
  if (!bench::GpuLock::held_by_caller() && !lock.acquire([](const std::string& who) {
        fprintf(stderr, "waiting for the GPU lock %s%s\n", bench::GpuLock::path().c_str(),
                who.empty() ? "" : (", held by " + who).c_str());
      }))
    fatal("cannot take the GPU lock %s: %s", bench::GpuLock::path().c_str(), strerror(errno));

  const double t0 = now_ms();
  Hy4Flash m(dir, chunk);
  fprintf(stderr, "load %.1f s\n", (now_ms() - t0) / 1000);
  const int V = m.vocab();
  for (int i = 0; i < n; ++i)
    if (ids[size_t(i)] < 0 || ids[size_t(i)] >= V)
      fatal("id %d (at %d) is outside the vocabulary (0 .. %d)", ids[size_t(i)], i, V - 1);
  Gpu& g = Gpu::get();
  auto s = m.new_seq(n + gen + 32);
  memcpy(s->tokens(), ids.data(), size_t(n) * 4);
  Buffer lg = g.alloc(size_t(chunk) * V * 4, "hy4_dump_logits");
  FILE* fo = out.empty() ? nullptr : fopen(out.c_str(), "wb");
  if (!out.empty() && !fo) fatal("cannot write %s", out.c_str());
  SamplingParams sp;
  sp.temperature = 0.0f;
  const double tp = now_ms();
  for (int pos0 = 0; pos0 < n; pos0 += chunk) {
    const int T = std::min(chunk, n - pos0);
    const bool last = pos0 + T == n;
    auto e = g.begin();
    m.encode_prefill(*e, *s, pos0, T, sp, 0, last && gen > 0);
    if (fo && pos0 < rows) m.encode_logits_rows(*e, T, lg);
    e->commit_and_wait();
    m.write_selection();  // NI_HY4_DUMP_SEL (debug); a no-op without it
    if (fo && pos0 < rows) {
      const int w = std::min(T, rows - pos0);
      fwrite(lg.ptr(), 4, size_t(w) * V, fo);
    }
    fprintf(stderr, "prefill rows %d-%d: %.1f s (GPU %.1f ms)\n", pos0, pos0 + T - 1, (now_ms() - tp) / 1000,
            e->gpu_end_host_ms() - e->gpu_start_host_ms());
    if (Encoder::profiling()) Encoder::profile_report(1);  // NI_PROFILE=1: this chunk's kernels
  }
  if (fo) fclose(fo);
  const double pf = now_ms() - tp;
  fprintf(stderr, "prefill %d tokens: %.1f s (%.1f tok/s)\n", n, pf / 1000, n * 1000.0 / pf);
  if (gen > 0) {
    std::vector<int> outs = {s->tokens()[n]};
    const double td = now_ms();
    double gpu = 0;  // the steps' GPU time (start to end of each command buffer), against their wall time
    for (int k = 1; k < gen; ++k) {
      const int pos = n + k - 1;
      auto e = g.begin();
      m.encode_step(*e, *s, pos, sp, uint32_t(k));
      e->commit_and_wait();
      gpu += e->gpu_end_host_ms() - e->gpu_start_host_ms();
      outs.push_back(s->tokens()[pos + 1]);
    }
    const double dt = now_ms() - td;
    if (gen > 1) fprintf(stderr, "decode: GPU %.2f of %.2f ms a step\n", gpu / (gen - 1), dt / (gen - 1));
    if (Encoder::profiling() && gen > 1) Encoder::profile_report(gen - 1);  // NI_PROFILE=1: kernels per step
    printf("gen:");
    for (int t : outs) printf(" %d", t);
    printf("\n");
    if (gen > 1) fprintf(stderr, "decode %d tokens: %.1f s (%.2f tok/s)\n", gen - 1, dt / 1000, (gen - 1) * 1000.0 / dt);
  }
  return 0;
}
