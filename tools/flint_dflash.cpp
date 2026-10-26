// A DFlash2 drafter on Flint's dense Llama (Nanbeige4.3): greedy decoding of a prompt plainly, then with the drafter
// (a chain of K drafts verified in one pass of K + 1 rows), the tokens per cycle, the speed of each and whether the
// two texts agree.
//   flint_dflash <model_dir> <drafter_dir> <ids_file> [--tokens N] [--k K] [--bits 4|8]
// ids_file: the prompt's token ids (whitespace-separated), e.g. a chat prompt rendered by its template.
// Takes the GPU lock ($BENCH_GPU_LOCK, default /tmp/kapitsa-gpu.lock) unless NI_LOCK_HELD=1 or NACE_GPU_LOCK_HELD=1.
#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/flint.h"
#include "model/flint_dfl.h"
#include "util/log.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace ni;

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: flint_dflash <model_dir> <drafter_dir> <ids_file> [--tokens N] [--k K] [--bits 4|8]\n");
    return 1;
  }
  const std::string dir = argv[1], ddir = argv[2];
  int n_tok = 256, K = 7, bits = 4;
  for (int i = 4; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    const int v = atoi(argv[i + 1]);
    if (a == "--tokens") n_tok = v;
    else if (a == "--k") K = v;
    else if (a == "--bits") bits = v;
    else fatal("unknown option %s", a.c_str());
  }
  if (K < 1 || K > 7) fatal("--k: 1 to 7");
  std::vector<int> ids;
  {
    std::ifstream f(argv[3]);
    for (int t; f >> t;) ids.push_back(t);
  }
  if (ids.empty()) fatal("%s: no token ids", argv[3]);
  const int n = int(ids.size());

  bench::GpuLock lock;
  if (!env_on("NI_LOCK_HELD") && !env_on("NACE_GPU_LOCK_HELD") &&
      !lock.acquire([](const std::string& who) {
        fprintf(stderr, "waiting for the GPU lock%s\n", who.empty() ? "" : (", held by " + who).c_str());
      }))
    fatal("cannot take the GPU lock: %s", strerror(errno));
  Gpu& g = Gpu::get();
  Flint m(dir, 2048);
  FlintDflTarget target(m);
  DFlash2 dr(target, ddir, bits, false);
  const SamplingParams greedy;
  auto s = m.new_seq(n + n_tok + 64);
  auto prefill = [&](bool capture) {
    m.reset(*s);
    std::copy(ids.begin(), ids.end(), s->tokens());
    if (capture) dr.begin_prefill(n, 0);
    for (int p0 = 0; p0 < n; p0 += 2048) {
      const int T = std::min(2048, n - p0);
      auto e = g.begin();
      m.encode_prefill(*e, *s, p0, T, greedy, 0, p0 + T == n);
      e->commit_and_wait();
    }
  };

  // plain greedy decoding: the reference text and speed
  prefill(false);
  std::vector<int> plain;
  double t0 = Encoder::host_ms();
  for (int i = 0; i < n_tok; ++i) {
    auto e = g.begin();
    m.encode_step(*e, *s, n + i, greedy, uint32_t(n + i), true);
    e->commit_and_wait();
  }
  const double plain_ms = Encoder::host_ms() - t0;
  plain.assign(s->tokens() + n, s->tokens() + n + n_tok + 1);

  // the drafter: prefill with captures, the context ring, then cycles of [commit the last cycle's rows, draft from the
  // anchor, verify K + 1 rows with captures, accept] in one command buffer
  prefill(true);
  auto d = dr.new_seq();
  {
    auto e = g.begin();
    dr.encode_prefill_commit(*e, *d);
    e->commit_and_wait();
    dr.end_prefill();
  }
  int P = n, made = 1, cycles = 0;
  t0 = Encoder::host_ms();
  // NI_DFL_SPLIT=1: the drafting half (commit + draft) and the verify half (verify + accept) in command buffers of their
  // own, with each one's GPU time summed (a timing split; the default runs a cycle as one command buffer)
  const bool split = env_on("NI_DFL_SPLIT");
  double gpu_draft = 0, gpu_verify = 0;
  // The fallback (NI_DFL_GATE=0 turns it off). Drafting cycles keep a moving average (1/8 per cycle) of the tokens
  // each keeps; under the floor (NI_DFL_FLOOR, default 1.50: an 8-row verify plus the drafting pass cost 1.36 decode
  // steps on the M5 Max) the next `hold` tokens decode plainly, as 1-row cycles without drafts (the decode step's
  // kernels and sampler: the same text), whose captures keep the drafter's context whole. Then a probe: 4 drafting
  // cycles, judged on their own mean; a probe that passes goes back to drafting, one that fails doubles the hold (16
  // .. 256 tokens), so a prose stretch stops paying for probes while code keeps its bursts. Token counts only, no
  // clock: the same request decodes the same text whatever the load.
  const bool gate = !getenv("NI_DFL_GATE") || atoi(getenv("NI_DFL_GATE")) != 0;
  const double floor_tpc = getenv("NI_DFL_FLOOR") ? atof(getenv("NI_DFL_FLOOR")) : 1.50;
  const int probe = 4;
  double avg = 2.0;  // prior: a drafting cycle keeps 2 tokens
  int hold = 16, plain_left = 0, probe_left = 0, probe_sum = 0, spec_cycles = 0, plain_cycles = 0;
  // The drafter's context: rows captured into its capture buffer and not yet committed (pend, positions cap_base ..),
  // committed before a drafting pass, or every 8 rows of a plain stretch (one commit for 8 tokens, not one a token)
  int pend = 0, cap_base = n;
  while (made <= n_tok && P + 16 < n + n_tok + 64) {
    const bool drafting = !gate || plain_left == 0;
    const int k = drafting ? K : 0;
    auto e = g.begin();
    if (pend && (drafting || pend >= 8)) dr.encode_commit(*e, *d, pend), pend = 0, cap_base = P;
    if (drafting) dr.encode_draft(*e, *s, *d, P, greedy);
    if (split) {
      e->commit_and_wait();
      gpu_draft += e->gpu_ms();
      e = g.begin();
    }
    if (drafting) dr.arm_verify(P, k + 1);
    else dr.arm_verify(cap_base, 8);  // a plain row lands in row P - cap_base
    m.encode_verify(*e, *s, P, k + 1);
    dr.disarm();
    m.encode_accept(*e, *s, P, k + 1, k, greedy);
    e->commit_and_wait();
    if (split) gpu_verify += e->gpu_ms();
    const int a = m.accepted();
    if (drafting) {
      ++spec_cycles;
      if (probe_left > 0) {  // a probe: its own mean decides
        probe_sum += a + 1;
        if (--probe_left == 0) {
          const double mean = double(probe_sum) / probe;
          if (mean >= floor_tpc) avg = mean, hold = 16;
          else plain_left = hold, hold = std::min(2 * hold, 256);
        }
      } else {
        avg = 0.875 * avg + 0.125 * (a + 1);
        if (gate && avg < floor_tpc) plain_left = hold, hold = std::min(2 * hold, 256);
      }
    } else {
      ++plain_cycles;
      if (--plain_left == 0) probe_left = probe, probe_sum = 0;
    }
    if (drafting) cap_base = P, pend = a + 1;
    else ++pend;
    P += a + 1;
    made += a + 1;
    ++cycles;
  }
  const double spec_ms = Encoder::host_ms() - t0;
  const int got = made - 1;
  int same = 0;
  while (same < std::min(got, n_tok) && s->tokens()[n + 1 + same] == plain[size_t(1 + same)]) ++same;
  printf("prompt %d tokens; plain: %d tokens in %.0f ms (%.1f tok/s)\n", n, n_tok, plain_ms, n_tok / plain_ms * 1000);
  printf("drafter K=%d: %d tokens in %d cycles (%.2f tokens/cycle), %.0f ms (%.1f tok/s, %.2fx); first %d tokens "
         "identical to plain decoding\n",
         K, got, cycles, double(got) / std::max(cycles, 1), spec_ms, got / spec_ms * 1000,
         (got / spec_ms) / (n_tok / plain_ms), same);
  printf("fallback %s: %d drafting cycles, %d plain cycles\n", gate ? "on" : "off", spec_cycles, plain_cycles);
  if (split)
    printf("per cycle: drafting %.2f ms, verify + accept %.2f ms (GPU)\n", gpu_draft / std::max(cycles, 1),
           gpu_verify / std::max(cycles, 1));
  return 0;
}
