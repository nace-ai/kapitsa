// Runs a prompt (token ids) through the engine and generates tokens.
//   ni_run <model_dir> <ids: comma-separated> [n_gen=32] [temp=0] [--dump-logits file]
//          [--spec D] [--dflash <drafter dir>]   (NI_DFLASH=<dir> works too) [--ignore-eos]
// --dflash: DFlash2 speculative decoding (research/inference/dflash2/SPEC.md), one command buffer per
// cycle: drafting pass, 8-row target verify with captures, on-GPU acceptance (greedy, or exact
// speculative sampling at temp > 0), ring commit of the kept rows at the start of the next cycle.
// --ignore-eos: keep generating past end-of-sequence tokens (benchmarks, equality tests).
// --ref <file with an "ids:" line> (with --dflash, T = 0): at the first token that differs from that
// reference (plain greedy), prints the target's logits of both tokens three ways (SPEC §12 G2): the
// DFlash verify row that produced it, plain token-by-token decoding, one multi-row prefill of the prefix.
// Prints generated ids, prompt/decode speeds; optionally dumps the logits of the first
// generated position (raw float32) for comparison against a reference implementation.
#include "metal/gpu.h"
#include "model/dflash2.h"
#include "model/qwen35.h"
#include "util/log.h"

#include <algorithm>
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
  std::vector<int> ref;  // --ref: reference output ids
  bool tbt = false, ignore_eos = false;
  int spec = 0;
  for (int i = 1; i < argc; ++i) {
    if (i + 1 < argc && !strcmp(argv[i], "--dump-logits")) dump = argv[i + 1];
    if (i + 1 < argc && !strcmp(argv[i], "--ref")) {
      std::ifstream f(argv[i + 1]);
      std::string all((std::istreambuf_iterator<char>(f)), {}), line;
      const size_t at = all.find("ids:");
      if (at == std::string::npos) fatal("--ref %s: no ids: line", argv[i + 1]);
      std::stringstream ss(all.substr(at + 4, all.find('\n', at) - at - 4));
      int t;
      while (ss >> t) ref.push_back(t);
    }
    if (!strcmp(argv[i], "--tbt")) tbt = true;
    if (!strcmp(argv[i], "--ignore-eos")) ignore_eos = true;
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

  // Opt-in DFlash2 block drafter; the prompt rows are captured for the drafter's context during the
  // prefill below.
  std::unique_ptr<DFlash2> dfl;
  std::unique_ptr<DFlashSeq> dseq;
  if (dflash_dir == "auto") {
    dflash_dir = DFlash2::find(dir, m.config());
    if (dflash_dir.empty()) fatal("--dflash auto: no DFlash2 drafter found for %s", dir.c_str());
  }
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
    // NI_DFLASH_VERIFY=k: verify only the first k of the 7 proposals (lossless; fewer verify rows)
    const int K = DFlash2::verify_depth();
    const bool trace = getenv("NI_TRACE") != nullptr;
    // NI_DFLASH_CONF=<file>: per cycle "a c_0 .. c_6", c_p = softmax over the 16 candidates of the walk's
    // scores S_p at the chosen candidate (the drafter's confidence; calibration data for a dynamic depth)
    FILE* conf = getenv("NI_DFLASH_CONF") ? fopen(getenv("NI_DFLASH_CONF"), "a") : nullptr;
    double ts = now_ms();
    int P = n, m_prev = 0, cycles = 0, emitted = 0;
    int div_at = -1, div_tok[2] = {0, 0}, div_row = 0, div_P = 0;
    long verified_sum = 0;
    float div_logit[2] = {0, 0};
    std::vector<int> acc_hist(K + 1, 0), reached(K, 0), acc_pos(K, 0);
    double gpu_sum = 0, wall_sum = 0;
    bool stop2 = false;
    if (Encoder::profiling()) Encoder::profile_report(1);
    // NI_DFLASH_CB: 1 = the whole cycle in one command buffer; 2 = its drafting and verify halves as two
    // command buffers committed back to back (the verify is encoded while the draft runs); 3 (default)
    // = double-buffered: the next cycle's drafting half (GPU-driven: anchor and ring rows from this
    // cycle's acceptance) is committed right behind each verify, before the host waits for it.
    // NI_TRACE / NI_DFLASH_CONF read drafter buffers after each cycle, so they run mode 2; NI_PROFILE
    // (one command buffer per dispatch) runs mode 1.
    int mode = getenv("NI_DFLASH_CB") && *getenv("NI_DFLASH_CB") ? atoi(getenv("NI_DFLASH_CB")) : 3;
    if ((trace || conf) && mode == 3) mode = 2;
    if (Encoder::profiling()) mode = 1;
    std::unique_ptr<Encoder> next_draft;  // mode 3: the committed drafting half of the coming cycle
    if (mode == 3) {
      dfl->gpu_state_reset(P);  // the GPU is idle here
      next_draft = g.begin();
      dfl->encode_cycle_draft_gpu(*next_draft, s, *dseq, P, sp);
      next_draft->commit();
    }
    while (int(out.size()) < n_gen && !stop2) {
      if (P + DFlash2::kB + 2 >= s.cap) break;
      const double tc0 = now_ms();
      std::unique_ptr<Encoder> e1;
      auto e = g.begin();
      if (mode == 1) {
        dfl->encode_cycle(*e, s, *dseq, P, m_prev, K, sp);
        e->commit();
      } else if (mode == 2) {
        e1 = std::move(e);
        dfl->encode_cycle_draft(*e1, s, *dseq, P, m_prev, sp);
        e1->commit();
        e = g.begin();
        dfl->encode_cycle_verify(*e, s, P, K, sp);
        e->commit();
      } else {
        if (!next_draft) break;  // no room for another drafting pass
        e1 = std::move(next_draft);
        dfl->encode_cycle_verify(*e, s, P, K, sp);
        e->commit();
        // the next anchor is at most P + 8: commit its drafting half now (it waits on this verify)
        if (P + 8 + DFlash2::kB + 2 < s.cap) {
          next_draft = g.begin();
          dfl->encode_cycle_draft_gpu(*next_draft, s, *dseq, P + 8, sp);
          next_draft->commit();
        }
      }
      e->wait();
      const double wall = now_ms() - tc0;
      const double gms = e->gpu_ms() + (e1 ? e1->gpu_ms() : 0.0);
      gpu_sum += gms;
      wall_sum += wall;
      const int a = dfl->accepted();
      const int* tk = s.tokens();
      if (trace)
        fprintf(stderr, "P=%d a=%d chain=%d,%d,%d,%d,%d,%d,%d next=%d wall=%.2fms gpu=%.2fms\n", P, a, tk[P + 1],
                tk[P + 2], tk[P + 3], tk[P + 4], tk[P + 5], tk[P + 6], tk[P + 7], tk[P + 1 + a], wall, gms);
      if (conf) {
        fprintf(conf, "%d", a);
        for (int p = 0; p < 7; ++p) {
          const float* S = dfl->walk_scores() + p * 16;
          float mx = S[0], z = 0;
          for (int k = 1; k < 16; ++k) mx = std::max(mx, S[k]);
          for (int k = 0; k < 16; ++k) z += std::exp(S[k] - mx);
          fprintf(conf, " %.4f", std::exp(S[dfl->walk_k()[p]] - mx) / z);
        }
        fprintf(conf, "\n");
      }
      int mm = 0;  // tokens emitted = rows committed this cycle
      for (int i = 1; i <= a + 1 && int(out.size()) < n_gen; ++i) {
        if (!ref.empty() && div_at < 0 && out.size() < ref.size() && tk[P + i] != ref[out.size()]) {
          div_at = int(out.size());
          div_tok[0] = ref[out.size()];
          div_tok[1] = tk[P + i];
          const int V = m.config().vocab;
          const float* row = m.logits() + size_t(i - 1) * V;  // verify row i-1 predicts position P+i
          div_logit[0] = row[div_tok[0]];
          div_logit[1] = row[div_tok[1]];
          div_row = i - 1;
          div_P = P;
        }
        out.push_back(tk[P + i]);
        ++mm;
        if (!ignore_eos)
          for (int eos : m.config().eos)
            if (tk[P + i] == eos) stop2 = true;
        if (stop2) break;
      }
      const int kv = dfl->verified(K);  // proposals this cycle verified
      verified_sum += kv;
      for (int i = 0; i <= std::min(a, kv - 1); ++i) {
        reached[i]++;
        if (i < a) acc_pos[i]++;
      }
      m.commit_rows(s, mm);
      m_prev = mm;
      emitted += mm;
      ++cycles;
      acc_hist[a]++;
      P += mm;
      if (mode == 3) dseq->ctx = P;  // the GPU committed the kept rows to the ring
    }
    if (next_draft) next_draft->wait();
    double td = now_ms() - ts;
    if (conf) fclose(conf);
    if (Encoder::profiling()) Encoder::profile_report(cycles);
    printf("ids:");
    for (int t : out) printf(" %d", t);
    printf("\n");
    fprintf(stderr,
            "prompt %d tok in %.0f ms (%.1f tok/s); dflash decode (verify %d of 7, %s): %zu tok in %.0f ms (%.1f tok/s), "
            "%d cycles, %.3f tok/cycle, %.2f ms GPU/cycle, %.2f ms wall/cycle (mode %d), hist",
            n, t_prompt, n / (t_prompt / 1000), K, sp.temperature > 0 ? "sampled" : "greedy", out.size() - 1, td,
            (out.size() - 1) / (td / 1000), cycles, double(emitted) / std::max(cycles, 1), gpu_sum / std::max(cycles, 1),
            td / std::max(cycles, 1), mode);
    for (int a = 0; a <= K; ++a) fprintf(stderr, " %d", acc_hist[a]);
    fprintf(stderr, "; %.2f proposals verified per cycle; acceptance by position", double(verified_sum) / std::max(cycles, 1));
    for (int i = 0; i < K; ++i) fprintf(stderr, " %.2f", reached[i] ? double(acc_pos[i]) / reached[i] : 0.0);
    fprintf(stderr, "\n");
    if (!ref.empty()) {
      const size_t cmpn = std::min(ref.size(), out.size());
      if (div_at < 0) {
        printf("ref: identical to the reference over %zu tokens\n", cmpn);
        return 0;
      }
      // the same position computed by plain decoding and by one prefill of the prefix (fresh sequences)
      const int V = m.config().vocab, pos = n + div_at;  // tokens[pos] is the differing token
      auto top2 = [&](const float* l, int* i0, int* i1) {
        *i0 = 0;
        for (int v = 1; v < V; ++v)
          if (l[v] > l[*i0]) *i0 = v;
        *i1 = *i0 ? 0 : 1;
        for (int v = 0; v < V; ++v)
          if (v != *i0 && l[v] > l[*i1]) *i1 = v;
      };
      SamplingParams gsp;
      float plain[2], pre[2];
      int p0, p1, q0, q1;
      {
        auto s2 = m.new_seq(pos + 64);
        for (int i = 0; i < n; ++i) s2->tokens()[i] = ids[i];
        for (int i = 0; i < div_at; ++i) s2->tokens()[n + i] = out[i];
        for (int p = 0; p < n; p += m.prefill_chunk()) {
          auto e = g.begin();
          m.encode_prefill(*e, *s2, p, std::min(m.prefill_chunk(), n - p), gsp, 0, false);
          e->commit_and_wait();
        }
        for (int q = n; q < pos; ++q) {  // feed out[0 .. div_at-1] token by token; the last step's logits
          auto e = g.begin();
          m.encode_step(*e, *s2, q, gsp, 0, q == pos - 1);
          e->commit_and_wait();
        }
        if (pos == n) {  // the first generated token: the prefill's own logits
          auto e = g.begin();
          m.reset(*s2);
          for (int p = 0; p < n; p += m.prefill_chunk()) {
            const int T = std::min(m.prefill_chunk(), n - p);
            m.encode_prefill(*e, *s2, p, T, gsp, 0, p + T == n);
          }
          e->commit_and_wait();
        }
        plain[0] = m.logits()[div_tok[0]];
        plain[1] = m.logits()[div_tok[1]];
        top2(m.logits(), &p0, &p1);
        printf("G2 plain decode: top-2 %d (%.4f) %d (%.4f), gap %.4f\n", p0, m.logits()[p0], p1, m.logits()[p1],
               m.logits()[p0] - m.logits()[p1]);
      }
      {
        auto s3 = m.new_seq(pos + 64);
        for (int i = 0; i < n; ++i) s3->tokens()[i] = ids[i];
        for (int i = 0; i < div_at; ++i) s3->tokens()[n + i] = out[i];
        for (int p = 0; p < pos; p += m.prefill_chunk()) {
          const int T = std::min(m.prefill_chunk(), pos - p);
          auto e = g.begin();
          m.encode_prefill(*e, *s3, p, T, gsp, 0, p + T == pos);
          e->commit_and_wait();
        }
        pre[0] = m.logits()[div_tok[0]];
        pre[1] = m.logits()[div_tok[1]];
        top2(m.logits(), &q0, &q1);
      }
      printf("G2 first difference at output token %d (position %d): reference %d, dflash %d\n", div_at, pos, div_tok[0],
             div_tok[1]);
      printf("G2   logits (reference token / dflash token): plain decode %.4f / %.4f (gap %+.4f); dflash verify "
             "row %d at P=%d %.4f / %.4f (gap %+.4f); one prefill of the prefix %.4f / %.4f (gap %+.4f, argmax %d)\n",
             plain[0], plain[1], plain[0] - plain[1], div_row, div_P, div_logit[0], div_logit[1],
             div_logit[0] - div_logit[1], pre[0], pre[1], pre[0] - pre[1], q0);
    }
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
        if (!ignore_eos)
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
      if (!ignore_eos)
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
