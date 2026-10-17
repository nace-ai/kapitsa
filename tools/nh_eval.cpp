// Nemotron-H checks: teacher-forced next-token distributions against a reference file, raw greedy
// generation with timings.
//   nh_eval <dir> --ref F @ids1 [@ids2 ...]      KL(ref || model) over the reference's top-K (+ the lumped
//                                                       rest), top-1 agreement, perplexity; F in ni_eval's NIEVREF2
//                                                       format (tools/ni_eval.cpp; llama.cpp's logits through
//                                                       bench/quality/llama_ref.cpp)
//   nh_eval <dir> --dump-ref F @ids...             writes such a file from this model
//   nh_eval <dir> --gen N [--temp T] [--spec] @ids  prefill + N decoded tokens: ids, prefill and decode speed;
//                                                       --spec decodes with the LiLiCorr drafter found next to the
//                                                       model (tokens per cycle, the drafts' acceptance)
#include "metal/gpu.h"
#include "model/lilicorr.h"
#include "model/nemotron_h.h"
#include "util/log.h"

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
  std::string model = argv[1], ref_file, dump_file;
  int gen = -1, chunk = 512, topk = 64;
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
    else if (a == "--temp") temp = std::stof(next());
    else if (a == "--spec") spec = true;
    else if (a == "--chunk") chunk = std::stoi(next());
    else if (a == "--topk") topk = std::stoi(next());
    else seqs.push_back(a);
  }
  if (seqs.empty()) fatal("no sequences");
  std::vector<std::vector<int>> ids;
  for (auto& s : seqs) ids.push_back(read_ids(s));

  NemotronH m(model, chunk);
  const int V = m.vocab();
  Gpu& g = Gpu::get();

  if (gen >= 0) {
    const std::vector<int>& p = ids[0];
    const int n = int(p.size());
    auto s = m.new_seq(n + gen + 64);
    memcpy(s->tokens(), p.data(), p.size() * 4);
    SamplingParams sp;
    sp.temperature = temp;
    sp.top_p = 0.95f;
    sp.top_k = 20;
    sp.seed = 1234;
    std::unique_ptr<LiliCorr> lili;
    std::unique_ptr<LiliSeq> d;
    if (spec) {
      const std::string dp = LiliCorr::find(model);
      if (dp.empty()) fatal("no LiLiCorr drafter next to %s", model.c_str());
      lili = std::make_unique<LiliCorr>(m, dp);
      d = lili->new_seq();
      lili->begin_prefill(n, 0);
    }
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
    if (lili && getenv("NH_FEATURE_STATS")) lili->debug_feature_stats();
    t0 = now_ms();
    int produced = 1, cycles = 0;
    if (!lili) {
      for (int i = 0; i < gen - 1; ++i) {
        auto e = g.begin();
        m.encode_step(*e, *s, n + i, sp, uint32_t(n + i), true);
        e->commit_and_wait();
      }
      produced = gen;
    } else {
      constexpr int B = LiliCorr::kB;
      int P = n, m_prev = 0;
      while (produced < gen) {
        auto e1 = g.begin();
        if (m_prev > 0) lili->encode_commit(*e1, *d, m_prev);
        lili->encode_draft(*e1, *s, *d, P);
        e1->commit();
        auto e2 = g.begin();
        lili->arm_verify(P);
        m.set_dynamic_rows(true);
        m.encode_verify(*e2, *s, P, B);
        lili->disarm();
        m.encode_accept(*e2, *s, P, B, B - 1, sp);
        m.set_dynamic_rows(false);
        e2->commit_and_wait();
        const int a = m.accepted();
        const int mm = std::min(a + 1, gen - produced);
        produced += mm;
        ++cycles;
        m.commit_rows(*s, mm);
        m_prev = mm;
        P += mm;
      }
      auto e3 = g.begin();
      lili->encode_commit(*e3, *d, m_prev);
      m.encode_flush(*e3, *s);
      e3->commit_and_wait();
    }
    const double t_dec = now_ms() - t0;
    printf("generated:");
    for (int i = 0; i < produced; ++i) printf("%s%d", i ? "," : " ", s->tokens()[n + i]);
    printf("\n");
    printf("prefill %d tokens in %.0f ms (%.1f tok/s); decode %d tokens in %.0f ms (%.2f tok/s)", n, t_pf,
           n / (t_pf / 1000), produced - 1, t_dec, produced > 1 ? (produced - 1) / (t_dec / 1000) : 0.0);
    if (cycles) printf("; %d cycles, %.2f tokens per cycle", cycles, double(produced - 1) / cycles);
    printf("\n");
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
        nll -= row[seq[p + 1]];
        ++cnt;
        if (dump) {
          mine[p].lp_next = row[seq[p + 1]];
          std::vector<uint32_t> idx(V);
          for (int i = 0; i < V; ++i) idx[i] = uint32_t(i);
          std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(), [&](uint32_t a, uint32_t b) { return row[a] > row[b]; });
          mine[p].top.resize(topk);
          for (int i = 0; i < topk; ++i) mine[p].top[i] = {idx[i], row[idx[i]]};
        }
        if (!refpos.empty()) {
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
