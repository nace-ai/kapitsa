// Runs a prompt (token ids) through the engine and generates tokens.
//   ni_run <model_dir> <ids: comma-separated> [n_gen=32] [temp=0] [--dump-logits file]
// Prints generated ids, prompt/decode speeds; optionally dumps the logits of the first
// generated position (raw float32) for comparison against a reference implementation.
#include "metal/gpu.h"
#include "model/qwen35.h"
#include "util/log.h"

#include <cmath>
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
  int pf_reps = 1;  // --pf-reps R: time the prefill R times (state reset in between), report each
  // --prefix P: prefill ids[0, P) once untimed and checkpoint it; the timed prefill then restores that
  // checkpoint and runs only ids[P, n) (a cached-turn suffix). --split B (P < B < n): a turn boundary
  // inside the suffix, checkpointed like the server does (split pass, or in-pass with NI_SHORTPF);
  // afterwards the in-pass checkpoint is compared with a separate prefill of ids[P, B).
  int prefix = 0, split = 0;
  for (int i = 1; i < argc; ++i) {
    if (i + 1 < argc && !strcmp(argv[i], "--dump-logits")) dump = argv[i + 1];
    if (!strcmp(argv[i], "--tbt")) tbt = true;
    if (i + 1 < argc && !strcmp(argv[i], "--spec")) spec = atoi(argv[i + 1]);
    if (i + 1 < argc && !strcmp(argv[i], "--pf-reps")) pf_reps = atoi(argv[i + 1]);
    if (i + 1 < argc && !strcmp(argv[i], "--prefix")) prefix = atoi(argv[i + 1]);
    if (i + 1 < argc && !strcmp(argv[i], "--split")) split = atoi(argv[i + 1]);
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
    const int C = m.prefill_chunk();
    Checkpoint ck0 = m.new_checkpoint(), ck1 = m.new_checkpoint();
    if (prefix > 0) {
      if (prefix >= n) { fprintf(stderr, "--prefix must be < prompt length\n"); return 1; }
      for (int p = 0; p < prefix; p += C) {
        int T = std::min(C, prefix - p);
        auto e = g.begin();
        m.encode_prefill(*e, s, p, T, sp, 0, false);
        if (spec && m.has_mtp()) m.encode_mtp_prefill(*e, s, p, T, 0);
        if (p + T == prefix) m.save(*e, s, ck0, prefix);
        e->commit_and_wait();
      }
      if (Encoder::profiling()) Encoder::profile_report(1);
      fprintf(stderr, "prefix %d tok prefilled; timing ids[%d, %d)\n", prefix, prefix, n);
      t0 = now_ms();
    }
    if (split && (split <= prefix || split >= n)) { fprintf(stderr, "--split must be in (prefix, n)\n"); return 1; }
    // one timed prefill of ids[prefix, n) as the server runs it (checkpoint at --split); returns GPU ms
    auto prefill_pass = [&]() {
      double gpu = 0;
      int p = prefix;
      auto e = g.begin();
      if (prefix) m.restore(*e, s, ck0);
      else m.reset(s);
      while (p < n) {
        int T = std::min(C, n - p);
        Checkpoint* snap = nullptr;
        if (split > p && split < p + T) {
          if (m.can_snapshot()) snap = &ck1;
          else T = split - p;
        }
        m.encode_prefill(*e, s, p, T, sp, 0, p + T == n, snap, snap ? split - p : 0);
        // MTP history rows i in [p, min(p+T, n-1)): pairs (h_i, t_{i+1})
        if (spec && m.has_mtp()) m.encode_mtp_prefill(*e, s, p, std::min(p + T, n - 1) - p, 0);
        last_chunk = p;
        p += T;
        if (p == split) m.save(*e, s, ck1, split);
        e->commit_and_wait();
        gpu += e->gpu_ms();
        if (p < n) e = g.begin();
      }
      return gpu;
    };
    for (int rep = 0; rep < pf_reps; ++rep) {
      if (rep) t0 = now_ms();
      const double gpu = prefill_pass();
      if (pf_reps > 1)
        fprintf(stderr, "prefill rep %d: %d tok in %.2f ms (GPU %.2f ms)\n", rep, n - prefix, now_ms() - t0, gpu);
      if (pf_reps > 1 && Encoder::profiling()) Encoder::profile_report(1);
    }
    if (split && m.can_snapshot()) {
      // reference: the state after a separate prefill of ids[prefix, split)
      std::vector<std::vector<float>> conv, rec;
      for (size_t i = 0; i < ck1.rec.size(); ++i) {
        conv.emplace_back(ck1.conv[i] ? std::vector<float>(ck1.conv[i].as<float>(), ck1.conv[i].as<float>() + ck1.conv[i].size / 4) : std::vector<float>());
        rec.emplace_back(ck1.rec[i] ? std::vector<float>(ck1.rec[i].as<float>(), ck1.rec[i].as<float>() + ck1.rec[i].size / 4) : std::vector<float>());
      }
      auto e = g.begin();
      if (prefix) m.restore(*e, s, ck0);
      else m.reset(s);
      for (int p = prefix; p < split; p += C) m.encode_prefill(*e, s, p, std::min(C, split - p), sp, 0, false);
      e->commit_and_wait();
      double dc = 0, dr = 0, mr = 0;
      for (size_t i = 0; i < rec.size(); ++i) {
        if (rec[i].empty()) continue;
        for (size_t j = 0; j < conv[i].size(); ++j) dc = std::max(dc, double(std::fabs(conv[i][j] - s.conv[i].as<float>()[j])));
        for (size_t j = 0; j < rec[i].size(); ++j) {
          dr = std::max(dr, double(std::fabs(rec[i][j] - s.rec[i].as<float>()[j])));
          mr = std::max(mr, double(std::fabs(s.rec[i].as<float>()[j])));
        }
      }
      fprintf(stderr, "snapshot at %d vs separate prefill: conv max|d| %.3g, state max|d| %.3g (max|state| %.3g)\n",
              split, dc, dr, mr);
      // the reference prefill replaced the live state: run the timed pass again before decoding
      const double tk = now_ms();
      prefill_pass();
      t0 += now_ms() - tk;
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
