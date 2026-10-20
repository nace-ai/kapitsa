// Nemotron-H checks: teacher-forced next-token distributions against a reference file, raw greedy
// generation with timings.
//   nh_eval <dir> --ref F @ids1 [@ids2 ...]      KL(ref || model) over the reference's top-K (+ the lumped
//                                                       rest), top-1 agreement, perplexity; F in ni_eval's NIEVREF2
//                                                       format (tools/ni_eval.cpp; llama.cpp's logits through
//                                                       bench/quality/llama_ref.cpp)
//   nh_eval <dir> --dump-ref F @ids...             writes such a file from this model
//   nh_eval <dir> --vcheck N F @ids                prefill of the first N ids, then the rest teacher-forced in verify
//                                                       passes of 8 rows (the last one shorter) (all rows, no drafter); writes their logits (fp32
//                                                       [rows][vocab]) to F, and with NH_VCHECK_STEP=1 the one-row
//                                                       decode path's instead (tools/nh_vcheck.py compares two)
//   nh_eval <dir> --gen N [--temp T] [--spec] @ids  prefill + N decoded tokens: ids, prefill and decode speed;
//                                                       --spec decodes with the block drafter (LiLiCorr, DSpark)
//                                                       found next to the model or named by --drafter DIR (tokens
//                                                       per cycle, the drafts' acceptance)
//   --image F (repeatable): the images of the sequences, in order of their runs of <image> tokens (each run as long
//   as the model's processor makes that image; its <img> / </img> marks already in the ids); --skip N[,N2,...]:
//   statistics from position N on (e.g. an answer after an image prompt), per sequence or one for all
#include "metal/gpu.h"
#include "model/dspark.h"
#include "model/lilicorr.h"
#include "model/nh_drafter.h"
#include "model/nemotron_h.h"
#include "util/log.h"
#include "vision/encoder.h"

#include <mach/mach.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace ni;

namespace {

std::vector<int> read_ids(const std::string& arg0) {
  std::string arg = arg0;
  if (!arg.empty() && arg[0] == '@') {
    std::ifstream f(arg.substr(1));
    if (!f) fatal("cannot read %s", arg.c_str() + 1);
    arg.assign(std::istreambuf_iterator<char>(f), {});
  }
  std::vector<int> v;
  std::stringstream ss(arg);
  std::string tok;
  while (std::getline(ss, tok, ','))
    if (tok.find_first_not_of(" \n\r\t") != std::string::npos) v.push_back(std::stoi(tok));
  return v;
}

int log_softmax(float* x, int V) {
  float m = x[0];
  int am = 0;
  for (int i = 1; i < V; ++i)
    if (x[i] > m) m = x[i], am = i;
  double s = 0.0;
  for (int i = 0; i < V; ++i) s += std::exp(double(x[i] - m));
  const float lse = m + float(std::log(s));
  for (int i = 0; i < V; ++i) x[i] -= lse;
  return am;
}

struct RefPos {
  float lp_next = 0;
  std::vector<std::pair<uint32_t, float>> top;
};

double kl_topk(const RefPos& r, const float* q) {
  double pk = 0, qk = 0, kl = 0;
  for (auto& [id, lp] : r.top) {
    const double p = std::exp(double(lp));
    pk += p;
    qk += std::exp(double(q[id]));
    kl += p * (double(lp) - double(q[id]));
  }
  const double pr = std::max(1.0 - pk, 1e-12), qr = std::max(1.0 - qk, 1e-12);
  return kl + pr * (std::log(pr) - std::log(qr));
}

double pct(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, size_t(q * double(v.size() - 1) + 0.5))];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: nh_eval <dir> [--ref F | --dump-ref F | --gen N [--temp T]] [--chunk R] [--topk K] @ids...\n");
    return 1;
  }
  std::string model = argv[1], ref_file, dump_file, drafter;
  int gen = -1, chunk = 512, topk = 64, vcheck = -1;
  std::string vcheck_file;
  std::vector<int> skips;
  std::vector<std::string> image_files;
  bool spec = false;
  float temp = 0.0f;
  std::vector<std::string> seqs;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) fatal("%s needs a value", a.c_str());
      return argv[++i];
    };
    if (a == "--ref") ref_file = next();
    else if (a == "--dump-ref") dump_file = next();
    else if (a == "--gen") gen = std::stoi(next());
    else if (a == "--vcheck") vcheck = std::stoi(next()), vcheck_file = next();
    else if (a == "--temp") temp = std::stof(next());
    else if (a == "--spec") spec = true;
    else if (a == "--drafter") drafter = next(), spec = true;
    else if (a == "--chunk") chunk = std::stoi(next());
    else if (a == "--topk") topk = std::stoi(next());
    else if (a == "--skip") {
      std::stringstream ss(next());
      std::string t;
      while (std::getline(ss, t, ',')) skips.push_back(std::stoi(t));
    }
    else if (a == "--image") image_files.push_back(next());
    else seqs.push_back(a);
  }
  if (seqs.empty()) fatal("no sequences");
  std::vector<std::vector<int>> ids;
  for (auto& s : seqs) ids.push_back(read_ids(s));

  NemotronH m(model, chunk);
  const int V = m.vocab();
  Gpu& g = Gpu::get();
  // --image: each run of the encoder's placeholder takes the next image's rows (encoded at the processor's own sizes)
  std::unique_ptr<vision::ImageEncoder> enc;
  if (!image_files.empty() && !(enc = vision::open_encoder(model))) fatal("%s takes no images", model.c_str());
  size_t next_image = 0;
  std::vector<Buffer> image_bufs;
  auto set_images = [&](Seq& s, const std::vector<int>& seq, std::vector<vision::EmbedRun>& runs) {
    if (!enc) return;
    const int ph = enc->placeholder(), n = int(seq.size());
    for (int i = 0; i < n;) {
      if (seq[i] != ph) {
        ++i;
        continue;
      }
      int j = i;
      while (j < n && seq[j] == ph) ++j;
      if (next_image >= image_files.size()) fatal("more image runs than --image files");
      const std::string& f = image_files[next_image++];
      std::ifstream in(f, std::ios::binary);
      const std::string bytes((std::istreambuf_iterator<char>(in)), {});
      const vision::Prepared p = enc->prepare(vision::decode(bytes));
      if (p.tokens != j - i) fatal("%s: %d image tokens in the sequence, the processor makes %d", f.c_str(), j - i, p.tokens);
      Buffer b = g.alloc(size_t(p.tokens) * enc->out_dim() * 2 + 16, "image_embeddings");
      const double t0 = now_ms();
      enc->encode(p, b);
      fprintf(stderr, "image %s: %dx%d -> %d tokens, encoded in %.0f ms\n", f.c_str(), p.width, p.height, p.tokens,
              now_ms() - t0);
      image_bufs.push_back(b);
      runs.push_back(vision::EmbedRun{i, p.tokens, b, 0, enc->out_dim()});
      i = j;
    }
    s.embeds = runs.empty() ? nullptr : &runs;
  };

  if (vcheck >= 0) {
    const std::vector<int>& p = ids[0];
    const int n = vcheck, L = int(p.size());
    if (n < 1 || n >= L) fatal("--vcheck %d: the sequence has %d ids", n, L);
    auto s = m.new_seq(L + 64);
    memcpy(s->tokens(), p.data(), p.size() * 4);
    std::vector<vision::EmbedRun> runs;
    set_images(*s, p, runs);
    SamplingParams sp;
    for (int p0 = 0; p0 < n; p0 += m.prefill_chunk()) {
      const int T = std::min(m.prefill_chunk(), n - p0);
      auto e = g.begin();
      m.encode_prefill(*e, *s, p0, T, sp, 0, p0 + T == n);
      e->commit_and_wait();
      if (!m.ane_finish_chunk()) fatal("the Neural Engine split failed on a chunk (rerun with NI_ANE=0)");
    }
    memcpy(s->tokens(), p.data(), p.size() * 4);  // (the prefill sampled tokens[n]: the given one goes back)
    const bool step = getenv("NH_VCHECK_STEP") && atoi(getenv("NH_VCHECK_STEP"));
    FILE* f = fopen(vcheck_file.c_str(), "wb");
    if (!f) fatal("cannot write %s", vcheck_file.c_str());
    Buffer out = g.alloc(size_t(8) * V * 4, "vcheck_logits");
    int rows = 0;
    for (int P = n; P < L; P += 8) {
      const int R = std::min(8, L - P);
      if (step) {
        for (int r = 0; r < R; ++r) {
          auto e = g.begin();
          m.encode_logits_rows(*e, *s, P + r, 1, out.view(size_t(r) * V * 4, size_t(V) * 4));
          e->commit_and_wait();
        }
        fwrite(out.as<float>(), 4, size_t(R) * V, f);
      } else {
        auto e = g.begin();
        m.encode_verify(*e, *s, P, R);
        e->commit_and_wait();
        fwrite(m.logits(), 4, size_t(R) * V, f);
        m.commit_rows(*s, R);
      }
      rows += R;
    }
    fclose(f);
    printf("vcheck: %d rows of %d logits from position %d (%s) -> %s\n", rows, V, n, step ? "one-row decode" : "8-row verify",
           vcheck_file.c_str());
    return 0;
  }

  if (gen >= 0) {
    const std::vector<int>& p = ids[0];
    const int n = int(p.size());
    auto s = m.new_seq(n + gen + 64);
    memcpy(s->tokens(), p.data(), p.size() * 4);
    SamplingParams sp;
    sp.temperature = temp;
    sp.top_p = 0.95f;
    sp.top_k = 20;
    sp.seed = getenv("NI_SEED") ? uint32_t(strtoul(getenv("NI_SEED"), nullptr, 10)) : 1234;
    std::unique_ptr<NhDrafter> lili;
    std::unique_ptr<LiliSeq> d;
    if (spec) {
      const std::string dp = drafter.empty() ? NhDrafter::find(model) : drafter;
      if (dp.empty()) fatal("no block drafter next to %s", model.c_str());
      lili = NhDrafter::open(m, dp);
      d = lili->new_seq();
      lili->begin_prefill(n, 0);
    }
    // NH_PF_REPS=k: k prefills first (state reset between them), each printed with its GPU start latency and GPU time
    // (where a prefill's wall time goes)
    const int reps = getenv("NH_PF_REPS") ? atoi(getenv("NH_PF_REPS")) : 0;
    for (int rep = 0; rep < reps; ++rep) {
      m.reset(*s);
      const double r0 = now_ms();
      double start_lat = 0, gpu = 0;
      for (int p0 = 0; p0 < n; p0 += m.prefill_chunk()) {
        const int T = std::min(m.prefill_chunk(), n - p0);
        auto e = g.begin();
        m.encode_prefill(*e, *s, p0, T, sp, 0, p0 + T == n);
        e->commit();
        const double hc = Encoder::host_ms();
        e->wait();
        start_lat += e->gpu_start_host_ms() - hc;
        gpu += e->gpu_ms();
        if (!m.ane_finish_chunk()) fatal("the Neural Engine split failed on a chunk (rerun with NI_ANE=0)");
      }
      printf("prefill rep %d: %.1f ms wall, GPU start latency %.1f ms, GPU %.1f ms\n", rep, now_ms() - r0, start_lat, gpu);
    }
    if (reps) m.reset(*s);
    double t0 = now_ms();
    for (int p0 = 0; p0 < n; p0 += m.prefill_chunk()) {
      const int T = std::min(m.prefill_chunk(), n - p0);
      auto e = g.begin();
      m.encode_prefill(*e, *s, p0, T, sp, 0, p0 + T == n);
      if (lili && p0 + T == n) {
        lili->end_prefill();
        lili->encode_prefill_commit(*e, *d);
      }
      e->commit_and_wait();
      if (!m.ane_finish_chunk()) fatal("the Neural Engine split failed on a chunk (rerun with NI_ANE=0)");
    }
    const double t_pf = now_ms() - t0;
    if (lili && getenv("NH_FEATURE_STATS"))
      if (auto* lc = dynamic_cast<LiliCorr*>(lili.get())) lc->debug_feature_stats();
    t0 = now_ms();
    int produced = 1, cycles = 0;
    double spec_gpu_ms = 0, draft_gpu_ms = 0;
    if (!lili) {
      // as the engine: the next step is encoded and queued while the current one runs
      std::unique_ptr<Encoder> next;
      for (int i = 0; i < gen - 1; ++i) {
        auto e = m.grouped_passes() ? g.begin_grouped() : g.begin();
        m.encode_step(*e, *s, n + i, sp, uint32_t(n + i), true);
        e->commit();
        if (next) next->wait();
        next = std::move(e);
      }
      if (next) next->wait();
      produced = gen;
    } else {
      constexpr int B = NhDrafter::kB;
      int P = n, m_prev = 0;
      double gpu_ms = 0;  // GPU time of the drafting and verify command buffers (the rest of the wall time: host gaps)
      // as the engine: a drafter that chains() commits the next cycle's drafting half right behind each verify (it
      // reads the acceptance on the GPU), so the GPU does not wait for the host between cycles; otherwise the drafting
      // half is encoded once the host has read the acceptance
      // NH_SPEC_LOG=<file>: per cycle a JSON line of the DSpark pass (candidates, their log-probs, the Markov head's
      // corrected log-probs and picks), the verified rows and the accepted count (the drafting halves unchained, so
      // the pass's buffers still hold the cycle's)
      FILE* slog = getenv("NH_SPEC_LOG") ? fopen(getenv("NH_SPEC_LOG"), "w") : nullptr;
      if (getenv("NH_SPEC_LOG") && !slog) fatal("NH_SPEC_LOG: cannot write %s", getenv("NH_SPEC_LOG"));
      DSpark* dsp = dynamic_cast<DSpark*>(lili.get());
      if (slog && !dsp) fatal("NH_SPEC_LOG: not a DSpark drafter");
      const bool chain = lili->chains() && !slog;
      std::unique_ptr<Encoder> next;  // the coming cycle's committed drafting half
      while (produced < gen) {
        if (!next) {
          next = g.begin();
          Encoder::profile_tag("d:");  // (NI_PROFILE: the drafting half's kernels apart)
          if (m_prev > 0) lili->encode_commit(*next, *d, m_prev);
          m_prev = 0;
          lili->encode_draft(*next, *s, *d, P);
          Encoder::profile_tag(nullptr);
          next->commit();
        }
        const std::unique_ptr<Encoder> e1 = std::move(next);
        auto e2 = m.grouped_passes() ? g.begin_grouped() : g.begin();
        lili->arm_verify(P);
        m.set_dynamic_rows(true);
        m.encode_verify(*e2, *s, P, B);
        lili->disarm();
        m.encode_accept(*e2, *s, P, B, B - 1, sp);
        m.set_dynamic_rows(false);
        e2->commit();
        if (chain && produced + 1 < gen) {
          next = g.begin();
          Encoder::profile_tag("d:");
          lili->encode_draft_chained(*next, *s, *d, P);
          Encoder::profile_tag(nullptr);
          next->commit();
        }
        e2->wait();
        gpu_ms += e1->gpu_ms() + e2->gpu_ms();
        draft_gpu_ms += e1->gpu_ms();
        const int a = m.accepted();
        if (slog) {
          std::vector<int> cand, picks;
          std::vector<float> clp, qlp;
          dsp->debug_read(cand, clp, qlp, picks);
          auto arr = [&](const auto& v) {
            std::string o = "[";
            for (size_t i = 0; i < v.size(); ++i) o += (i ? "," : "") + std::to_string(v[i]);
            return o + "]";
          };
          // (T > 0) the target's distribution of rows 0..6 as its sampler draws from it: top-k by logit, top-p on the
          // untempered probabilities, then the temperature (in double: a candidate right at the top-p boundary can
          // differ from the GPU sampler's float sums); null for the rows the cycle did not verify (past "live")
          std::string tp = "[";
          if (sp.temperature > 0) {
            const int V = m.vocab(), live = std::min(int(m.live_rows().as<uint32_t>()[0]), B);
            for (int r = 0; r < B - 1; ++r) {
              if (r >= live) {
                tp += r ? ",null" : "null";
                continue;
              }
              const float* l = m.logits() + size_t(r) * V;
              std::vector<int> idx{}; idx.resize(size_t(V));
              for (int i = 0; i < V; ++i) idx[size_t(i)] = i;
              const int kk = std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32);
              std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                                [&](int x, int y) { return l[x] > l[y] || (l[x] == l[y] && x < y); });
              std::vector<double> pr{}; pr.resize(size_t(kk));
              double tot = 0;
              for (int q = 0; q < kk; ++q) tot += pr[size_t(q)] = std::exp(double(l[idx[size_t(q)]]) - l[idx[0]]);
              int keep = kk;
              double cum = 0;
              for (int q = 0; q < kk; ++q)
                if ((cum += pr[size_t(q)] / tot) >= sp.top_p) {
                  keep = q + 1;
                  break;
                }
              double kt = 0;
              for (int q = 0; q < keep; ++q) kt += pr[size_t(q)] = std::exp((double(l[idx[size_t(q)]]) - l[idx[0]]) / sp.temperature);
              tp += r ? ",[" : "[";
              for (int q = 0; q < keep; ++q)
                tp += (q ? "," : "") + std::to_string(idx[size_t(q)]) + "," + std::to_string(pr[size_t(q)] / kt);
              tp += "]";
            }
          }
          tp += "]";
          std::vector<int> drafts(s->tokens() + P + 1, s->tokens() + P + B);
          fprintf(slog, "{\"P\":%d,\"a\":%d,\"live\":%d,\"K\":%d,\"cand\":%s,\"clp\":%s,\"qlp\":%s,\"picks\":%s,\"tp\":%s}\n", P,
                  a, int(m.live_rows().as<uint32_t>()[0]), dsp->cand_k(), arr(cand).c_str(), arr(clp).c_str(),
                  arr(qlp).c_str(), arr(picks).c_str(), tp.c_str());
        }
        const int mm = std::min(a + 1, gen - produced);
        produced += mm;
        ++cycles;
        m.commit_rows(*s, mm);
        if (next) {  // (the chained half committed the a + 1 rows to the ring)
          d->ctx = P + mm;
          d->hi = std::max(d->hi, P + a + 1);
        } else {
          m_prev = mm;
        }
        P += mm;
      }
      if (slog && (ferror(slog) | fclose(slog))) fatal("NH_SPEC_LOG: writing %s failed", getenv("NH_SPEC_LOG"));
      auto e3 = g.begin();
      Encoder::profile_tag("d:");  // (the last cycle's rows into the drafter's ring: drafting work too)
      if (m_prev > 0) lili->encode_commit(*e3, *d, m_prev);
      Encoder::profile_tag(nullptr);
      m.encode_flush(*e3, *s);
      e3->commit_and_wait();
      spec_gpu_ms = gpu_ms;
    }
    const double t_dec = now_ms() - t0;
    printf("generated:");
    for (int i = 0; i < produced; ++i) printf("%s%d", i ? "," : " ", s->tokens()[n + i]);
    printf("\n");
    printf("prefill %d tokens in %.0f ms (%.1f tok/s); decode %d tokens in %.0f ms (%.2f tok/s)", n, t_pf,
           n / (t_pf / 1000), produced - 1, t_dec, produced > 1 ? (produced - 1) / (t_dec / 1000) : 0.0);
    if (cycles)
      printf("; %d cycles, %.2f tokens per cycle, %.1f ms per cycle (GPU %.1f: drafting %.2f, verify %.2f)", cycles,
             double(produced - 1) / cycles, t_dec / cycles, spec_gpu_ms / cycles, draft_gpu_ms / cycles,
             (spec_gpu_ms - draft_gpu_ms) / cycles);
    const AneStats& as = m.ane_stats();
    if (as.chunks) printf("; ANE %d chunks, busy %.0f ms", as.chunks, as.ane_ms);
    task_vm_info_data_t vi;
    mach_msg_type_number_t vn = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vi, &vn) == KERN_SUCCESS)
      printf("; footprint %.2f GiB (peak %.2f)", vi.phys_footprint / 1073741824.0, vi.ledger_phys_footprint_peak / 1073741824.0);
    printf("; thermal %s\n", Gpu::thermal_state());
    if (Encoder::profiling()) Encoder::profile_report(cycles ? cycles : produced - 1);
    return 0;
  }

  std::vector<std::vector<RefPos>> refpos;
  if (!ref_file.empty()) {
    std::ifstream f(ref_file, std::ios::binary);
    char magic[8];
    uint32_t nseq = 0, RV = 0, K = 0;
    f.read(magic, 8);
    f.read((char*)&nseq, 4);
    f.read((char*)&RV, 4);
    f.read((char*)&K, 4);
    if (!f || memcmp(magic, "NIEVREF2", 8) || nseq != ids.size()) fatal("%s: not a reference for these sequences", ref_file.c_str());
    refpos.resize(nseq);
    for (uint32_t q = 0; q < nseq; ++q) {
      uint32_t n = 0;
      f.read((char*)&n, 4);
      std::vector<int> rid(n);
      f.read((char*)rid.data(), n * 4);
      if (rid != ids[q]) fatal("%s: sequence %u differs", ref_file.c_str(), q);
      refpos[q].resize(n > 0 ? n - 1 : 0);
      for (auto& rp : refpos[q]) {
        f.read((char*)&rp.lp_next, 4);
        rp.top.resize(K);
        for (auto& t : rp.top) {
          f.read((char*)&t.first, 4);
          f.read((char*)&t.second, 4);
        }
      }
    }
    if (!f) fatal("%s: truncated", ref_file.c_str());
  }
  std::ofstream dump;
  if (!dump_file.empty()) {
    dump.open(dump_file, std::ios::binary);
    const uint32_t nseq = uint32_t(ids.size()), Vu = uint32_t(V), Ku = uint32_t(topk);
    dump.write("NIEVREF2", 8);
    dump.write((const char*)&nseq, 4);
    dump.write((const char*)&Vu, 4);
    dump.write((const char*)&Ku, 4);
  }
  Buffer lg = g.alloc(size_t(chunk) * V * 4, "eval_logits");
  std::vector<double> all_kl;
  long all_agree = 0, all_n = 0;
  double all_nll = 0, all_nll_ref = 0;
  for (size_t q = 0; q < ids.size(); ++q) {
    const std::vector<int>& seq = ids[q];
    const int n = int(seq.size());
    auto s = m.new_seq(n + 16);
    memcpy(s->tokens(), seq.data(), seq.size() * 4);
    std::vector<vision::EmbedRun> runs;
    set_images(*s, seq, runs);
    const int skip = skips.empty() ? 0 : skips.size() == 1 ? skips[0] : skips.at(q);
    std::vector<double> kl;
    long agree = 0, cnt = 0;
    double nll = 0, nll_ref = 0;
    std::vector<RefPos> mine(n > 0 ? n - 1 : 0);
    for (int p0 = 0; p0 < n; p0 += chunk) {
      const int T = std::min(chunk, n - p0);
      auto e = g.begin();
      m.encode_logits_rows(*e, *s, p0, T, lg);
      e->commit_and_wait();
      if (!m.ane_finish_chunk()) fatal("the Neural Engine split failed on a chunk (rerun with NI_ANE=0)");
      for (int r = 0; r < T; ++r) {
        const int p = p0 + r;
        if (p >= n - 1) continue;
        float* row = lg.as<float>() + size_t(r) * V;
        const int am = log_softmax(row, V);
        if (p >= skip) {
          nll -= row[seq[p + 1]];
          ++cnt;
        }
        if (dump) {
          mine[p].lp_next = row[seq[p + 1]];
          std::vector<uint32_t> idx(V);
          for (int i = 0; i < V; ++i) idx[i] = uint32_t(i);
          std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(), [&](uint32_t a, uint32_t b) { return row[a] > row[b]; });
          mine[p].top.resize(topk);
          for (int i = 0; i < topk; ++i) mine[p].top[i] = {idx[i], row[idx[i]]};
        }
        if (p >= skip && !refpos.empty()) {
          const RefPos& rp = refpos[q][p];
          kl.push_back(std::max(kl_topk(rp, row), 0.0));
          nll_ref -= rp.lp_next;
          agree += uint32_t(am) == rp.top[0].first;
        }
      }
    }
    if (dump) {
      const uint32_t nn = uint32_t(n);
      dump.write((const char*)&nn, 4);
      dump.write((const char*)seq.data(), size_t(n) * 4);
      for (auto& rp : mine) {
        dump.write((const char*)&rp.lp_next, 4);
        for (auto& t : rp.top) {
          dump.write((const char*)&t.first, 4);
          dump.write((const char*)&t.second, 4);
        }
      }
    }
    double mean = 0;
    for (double v : kl) mean += v;
    mean = kl.empty() ? 0 : mean / double(kl.size());
    printf("{\"seq\": \"%s\", \"positions\": %ld, \"kl_mean\": %.6g, \"kl_median\": %.6g, \"kl_p99\": %.6g, \"kl_max\": "
           "%.6g, \"top1_agree\": %.5f, \"ppl\": %.5f, \"ppl_ref\": %.5f}\n",
           seqs[q].c_str(), cnt, mean, pct(kl, 0.5), pct(kl, 0.99), pct(kl, 1.0), cnt ? double(agree) / cnt : 0.0,
           cnt ? std::exp(nll / cnt) : 0.0, cnt ? std::exp(nll_ref / cnt) : 0.0);
    fflush(stdout);
    all_kl.insert(all_kl.end(), kl.begin(), kl.end());
    all_agree += agree;
    all_n += cnt;
    all_nll += nll;
    all_nll_ref += nll_ref;
  }
  double mean = 0;
  for (double v : all_kl) mean += v;
  mean = all_kl.empty() ? 0 : mean / double(all_kl.size());
  printf("{\"seq\": \"total\", \"positions\": %ld, \"kl_mean\": %.6g, \"kl_median\": %.6g, \"kl_p99\": %.6g, \"kl_max\": "
         "%.6g, \"top1_agree\": %.5f, \"ppl\": %.5f, \"ppl_ref\": %.5f}\n",
         all_n, mean, pct(all_kl, 0.5), pct(all_kl, 0.99), pct(all_kl, 1.0), all_n ? double(all_agree) / all_n : 0.0,
         all_n ? std::exp(all_nll / all_n) : 0.0, all_n ? std::exp(all_nll_ref / all_n) : 0.0);
  return 0;
}
