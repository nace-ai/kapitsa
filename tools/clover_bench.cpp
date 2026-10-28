// Clover's self-speculative decoding (src/model/clover.h) on a prompt, outside the engine: the cycles of the engine's
// loop (draft, verify, the acceptance, the kept rows' commit), each half timed as its own command buffer, with the
// tokens per cycle and the reply. Takes the GPU lock (BENCH_GPU_LOCK, default /tmp/kapitsa-gpu.lock) unless its caller
// holds it (NI_LOCK_HELD=1 / NACE_GPU_LOCK_HELD=1).
//   clover_bench <model_dir> [max_tokens=256] [prompt]   (NI_CLOVER_* as the server: draft rows, acceptance)
// NI_PROFILE=1: every dispatch timed apart, the draft's under "d:", the verify's under "v:" (much slower).
// CB_WARM=n: cycles left out of the timings (default 4).
#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/clover.h"
#include "tokenizer/tokenizer.h"
#include "util/log.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace ni;

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: clover_bench <model_dir> [max_tokens=256] [prompt]\n");
    return 2;
  }
  const std::string dir = argv[1];
  const int max_tokens = argc > 2 ? atoi(argv[2]) : 256;
  const std::string text = argc > 3 ? argv[3]
                                    : "Write a C++ function that parses an ISO 8601 date (YYYY-MM-DD) into a struct with "
                                      "year, month and day, validating each field, with a short explanation.";
  bench::GpuLock lock;
  if (!bench::GpuLock::held_by_caller() && !lock.acquire([](const std::string& who) {
        fprintf(stderr, "waiting for the GPU lock %s%s\n", bench::GpuLock::path().c_str(),
                who.empty() ? "" : (", held by " + who).c_str());
      }))
    fatal("cannot take the GPU lock %s: %s", bench::GpuLock::path().c_str(), strerror(errno));
  Clover m(dir, 2048);
  if (!m.self_spec()) fatal("clover_bench times the self-speculative cycles (NI_CLOVER_DECODE is diffusion)");
  Tokenizer tok;
  tok.load(dir + "/tokenizer.json");
  const std::vector<int> ids =
      tok.encode("<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
  const int n = int(ids.size()), W = m.draft_rows();
  auto s = m.new_seq(n + max_tokens + 64);
  std::copy(ids.begin(), ids.end(), s->tokens());
  Gpu& g = Gpu::get();
  SamplingParams sp;  // greedy
  std::vector<int> stops = m.eos_ids();
  if (const int im = tok.find("<|im_end|>"); im >= 0 && std::find(stops.begin(), stops.end(), im) == stops.end())
    stops.push_back(im);
  const double t0 = now_ms();
  {
    auto e = g.begin();
    for (int p = 0; p < n; p += m.prefill_chunk())
      m.encode_prefill(*e, *s, p, std::min(m.prefill_chunk(), n - p), sp, 0, p + m.prefill_chunk() >= n);
    e->commit_and_wait();
  }
  const double t_prefill = now_ms() - t0;
  if (Encoder::profiling()) Encoder::profile_report(1);  // (drops the prefill's records)
  CloverReply reply;
  reply.counts.assign(size_t(m.vocab()), 0);
  int* tk = s->tokens();
  int P = n, gen = 1, cycles = 0, timed = 0, timed_tokens = 0;
  reply.commit(tk + n, 1);
  bool go = std::find(stops.begin(), stops.end(), tk[n]) == stops.end();
  double gpu_d = 0, gpu_v = 0, wall = 0;
  const int warm = getenv("CB_WARM") ? atoi(getenv("CB_WARM")) : 4;
  std::vector<int> hist(size_t(W) + 1, 0);
  while (go && gen < max_tokens) {
    const double c0 = now_ms();
    Encoder::profile_tag("d:");
    auto e1 = g.begin();
    m.encode_draft(*e1, *s, P, W, reply, stops, sp, uint32_t(P));
    e1->commit();
    Encoder::profile_tag("v:");
    auto e2 = g.begin();
    m.encode_check(*e2, *s, P, W, reply, stops, sp, uint32_t(P));
    e2->commit_and_wait();
    e1->wait();
    Encoder::profile_tag(nullptr);
    const CloverRow* rw = m.rows();
    int a = 0;
    while (a < W - 1 && m.accept(rw[a], tk[P + 1 + a])) ++a;
    tk[P + a + 1] = rw[a].tok;
    int k = 0;
    while (k < a + 1 && gen < max_tokens) {
      ++k, ++gen;
      if (std::find(stops.begin(), stops.end(), tk[P + k]) != stops.end()) {
        go = false;
        break;
      }
    }
    reply.commit(tk + P + 1, k);
    m.commit_rows(*s, k);
    P += k;
    ++hist[size_t(a + 1)];
    if (++cycles > warm) {
      ++timed;
      timed_tokens += k;
      gpu_d += e1->gpu_ms();
      gpu_v += e2->gpu_ms();
      wall += now_ms() - c0;
    }
  }
  {
    auto e = g.begin();
    m.encode_flush(*e, *s);
    e->commit_and_wait();
  }
  if (Encoder::profiling()) Encoder::profile_report(std::max(timed, 1));
  const std::vector<int> out(tk + n, tk + n + gen);
  printf("%s\n---\n", tok.decode(out).c_str());
  printf("prompt %d tokens, prefill %.1f ms; %d tokens in %d cycles (%d-row draft and verify): %.2f tokens/cycle\n", n,
         t_prefill, gen, cycles, W, double(gen - 1) / std::max(cycles, 1));
  if (timed > 0)
    printf("timed %d cycles: draft %.2f ms + verify %.2f ms GPU, %.2f ms wall a cycle; %.1f tok/s (%d tokens)\n", timed,
           gpu_d / timed, gpu_v / timed, wall / timed, timed_tokens / (wall / 1000), timed_tokens);
  printf("tokens per cycle:");
  for (int i = 1; i <= W; ++i) printf(" %d:%d", i, hist[size_t(i)]);
  printf("\n");
  return 0;
}
