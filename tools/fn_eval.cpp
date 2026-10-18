// Qwen3.8-Flash-Next: teacher-forced logits of a token sequence through one of the model's passes, and the residual
// stream after each layer, as raw dumps for a reference comparison (the numpy reference of a synthetic pack, or the
// converter's dumps of the real one).
//   fn_eval <model> @ids --out F [--mode prefill|decode|verify] [--chunk C] [--rows R] [--xs F2]
//     --mode prefill (default): prefill chunks of C rows (default 512), logits of every row on the prefill GEMMs
//     --mode decode: one decode step per token (the decode kernels), logits of each step's sampled row
//     --mode verify: verify passes of R rows (default 3; MTP verify kernels)
//     --mode mtp: the MTP head's logits at every position (NI_FULL_DRAFT=1 for the full vocabulary): a prefill of the
//                 sequence, the head's history of rows [0, P) in one pass (--hist P, default 0), then one draft per
//                 position (row p = the head's distribution of token p + 2 from the state of p and token p + 1)
//     --xs F2: the residual (4 streams) of every position after the embedding and after each layer (one prefill per
//              layer count; float32 [L + 1, n, 4 * hidden])
//   F: float32 logits [n, vocab] (row p = the distribution after token p)
//   --dump-ref F  instead: ni_eval's NIEVREF2 reference file (top-K log-probs per position, --topk, default 64)
//   --ref F       compares with such a file: KL over its top-K (+ the lumped rest), top-1 agreement, perplexity
//                 (--seq i: the file's i-th sequence, for ni_eval's / the converter's files of several; --split P:
//                 also the mean KL of positions before P and from P, e.g. 2051 = where the indexer starts selecting)
#include "metal/gpu.h"
#include "model/qwen4_exp.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;

// log-softmax of one row in place; returns the argmax
static int log_softmax(float* x, int V) {
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

static std::vector<int> read_ids(const std::string& arg0) {
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

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: fn_eval <model> @ids --out F [--mode prefill|decode|verify] [--chunk C] [--rows R] [--xs F2]\n");
    return 2;
  }
  const std::string dir = argv[1];
  const std::vector<int> ids = read_ids(argv[2]);
  std::string out, xs_out, mode = "prefill", dump_ref, ref;
  int chunk = 512, rows = 3, hist = 0, topk = 64, seq = 0, split = 0;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) fatal("%s needs a value", a.c_str());
      return argv[++i];
    };
    if (a == "--out") out = next();
    else if (a == "--mode") mode = next();
    else if (a == "--chunk") chunk = std::stoi(next());
    else if (a == "--rows") rows = std::stoi(next());
    else if (a == "--xs") xs_out = next();
    else if (a == "--hist") hist = std::stoi(next());
    else if (a == "--dump-ref") dump_ref = next();
    else if (a == "--ref") ref = next();
    else if (a == "--topk") topk = std::stoi(next());
    else if (a == "--seq") seq = std::stoi(next());
    else if (a == "--split") split = std::stoi(next());
    else fatal("unknown option %s", a.c_str());
  }
  const int n = int(ids.size());
  Qwen4Exp m(dir, std::max(chunk, n));
  const int V = m.config().vocab, D = m.config().hidden, SD = m.config4().hc * D, L = m.config().layers;
  Gpu& g = Gpu::get();
  SamplingParams sp;
  if (!xs_out.empty()) {
    std::ofstream f(xs_out, std::ios::binary);
    for (int l = 0; l <= L; ++l) {
      m.set_debug_layers(l);
      auto s = m.new_seq(n + 16);
      memcpy(s->tokens(), ids.data(), ids.size() * 4);
      auto e = g.begin();
      m.encode_prefill(*e, *s, 0, n, sp, 0, false);
      e->commit_and_wait();
      f.write(reinterpret_cast<const char*>(m.residual().ptr()), std::streamsize(size_t(n) * SD * 4));
    }
    m.set_debug_layers(-1);
    fprintf(stderr, "wrote %s\n", xs_out.c_str());
  }
  if (out.empty() && dump_ref.empty() && ref.empty()) return 0;
  std::vector<float> logits(size_t(n) * V);
  auto s = m.new_seq(n + 16);
  memcpy(s->tokens(), ids.data(), ids.size() * 4);
  const double t0 = now_ms();
  if (mode == "prefill") {
    Buffer lg = g.alloc(size_t(chunk) * V * 4, "eval_logits");
    for (int p0 = 0; p0 < n; p0 += chunk) {
      const int T = std::min(chunk, n - p0);
      auto e = g.begin();
      m.encode_prefill(*e, *s, p0, T, sp, 0, false);
      m.encode_logits_rows(*e, T, lg);
      e->commit_and_wait();
      memcpy(logits.data() + size_t(p0) * V, lg.ptr(), size_t(T) * V * 4);
    }
  } else if (mode == "decode") {
    for (int p = 0; p < n; ++p) {
      auto e = g.begin();
      m.encode_step(*e, *s, p, sp, uint32_t(p), true);
      e->commit_and_wait();
      memcpy(logits.data() + size_t(p) * V, m.logits(), size_t(V) * 4);
      s->tokens()[p + 1] = p + 1 < n ? ids[size_t(p + 1)] : 0;  // teacher forcing
      m.ple_forget();
    }
  } else if (mode == "verify") {
    for (int p = 0; p < n; p += rows) {
      const int R = std::min(rows, n - p);
      auto e = g.begin();
      m.encode_verify(*e, *s, p, R, sp, uint32_t(p));
      e->commit_and_wait();
      memcpy(logits.data() + size_t(p) * V, m.logits(), size_t(R) * V * 4);
      m.commit_rows(*s, R);
    }
    auto e = g.begin();
    m.encode_flush(*e, *s);
    e->commit_and_wait();
  } else if (mode == "mtp") {
    auto e = g.begin();
    m.encode_prefill(*e, *s, 0, n, sp, 0, false);
    if (hist > 0) m.encode_mtp_prefill(*e, *s, 0, hist, 0);
    e->commit_and_wait();
    for (int p = hist; p + 1 < n; ++p) {
      auto e2 = g.begin();
      m.encode_mtp(*e2, *s, p, 1, p + 1, m.hidden_rows(), p, n + 4);
      e2->commit_and_wait();
      memcpy(logits.data() + size_t(p) * V, m.draft_logits().ptr(), size_t(V) * 4);
    }
  } else {
    fatal("unknown mode %s", mode.c_str());
  }
  fprintf(stderr, "%s: %d positions in %.0f ms\n", mode.c_str(), n, now_ms() - t0);
  if (!out.empty()) {
    std::ofstream f(out, std::ios::binary);
    f.write(reinterpret_cast<const char*>(logits.data()), std::streamsize(logits.size() * 4));
  }
  // positions [p_lo, n - 1) score (mtp: rows from --hist; its row p predicts token p + 2)
  const int p_lo = mode == "mtp" ? hist : 0, shift = mode == "mtp" ? 2 : 1;
  if (!dump_ref.empty()) {
    std::ofstream f(dump_ref, std::ios::binary);
    const uint32_t one = 1, Vu = uint32_t(V), Ku = uint32_t(topk), nn = uint32_t(n);
    f.write("NIEVREF2", 8);
    f.write((const char*)&one, 4); f.write((const char*)&Vu, 4); f.write((const char*)&Ku, 4);
    f.write((const char*)&nn, 4);
    f.write((const char*)ids.data(), std::streamsize(ids.size() * 4));
    std::vector<uint32_t> idx(static_cast<size_t>(V));
    for (int p = 0; p + 1 < n; ++p) {
      float* row = logits.data() + size_t(p) * V;
      log_softmax(row, V);
      const float lp = p + shift < n ? row[ids[size_t(p + shift)]] : 0.0f;
      for (int i = 0; i < V; ++i) idx[size_t(i)] = uint32_t(i);
      std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(), [&](uint32_t a, uint32_t b) { return row[a] > row[b]; });
      f.write((const char*)&lp, 4);
      for (int i = 0; i < topk; ++i) {
        f.write((const char*)&idx[size_t(i)], 4);
        f.write((const char*)&row[idx[size_t(i)]], 4);
      }
    }
    fprintf(stderr, "wrote %s\n", dump_ref.c_str());
  }
  if (!ref.empty()) {
    std::ifstream f(ref, std::ios::binary);
    char magic[8];
    uint32_t nseq = 0, Vr = 0, K = 0, nr = 0;
    f.read(magic, 8); f.read((char*)&nseq, 4); f.read((char*)&Vr, 4); f.read((char*)&K, 4);
    if (!f || memcmp(magic, "NIEVREF2", 8) || seq < 0 || uint32_t(seq) >= nseq) fatal("%s: no sequence %d", ref.c_str(), seq);
    for (int q = 0; q < seq; ++q) {  // skip: n, ids, n - 1 records of (lp, K x (id, lp))
      f.read((char*)&nr, 4);
      f.seekg(std::streamoff(nr) * 4 + std::streamoff(nr - 1) * (4 + 8 * std::streamoff(K)), std::ios::cur);
    }
    f.read((char*)&nr, 4);
    if (!f || int(nr) != n) fatal("%s: not a reference for this sequence", ref.c_str());
    std::vector<int> rid(nr);
    f.read((char*)rid.data(), nr * 4);
    if (rid != ids) fatal("%s: other token ids", ref.c_str());
    std::vector<double> kls;
    double kl_lo = 0, kl_hi = 0;
    long n_lo = 0, n_hi = 0;
    double nll = 0, nll_ref = 0;
    long agree = 0, cnt = 0;
    std::vector<std::pair<uint32_t, float>> top(K);
    for (int p = 0; p + 1 < n; ++p) {
      float lp_ref = 0;
      f.read((char*)&lp_ref, 4);
      for (auto& t : top) f.read((char*)&t.first, 4), f.read((char*)&t.second, 4);
      if (p < p_lo || p + shift >= n) continue;
      float* row = logits.data() + size_t(p) * V;
      const int am = log_softmax(row, V);
      double pk = 0, qk = 0, kl = 0;
      for (auto& [id, lpr] : top) {
        const double pr = std::exp(double(lpr));
        pk += pr;
        qk += std::exp(double(row[id]));
        kl += pr * (double(lpr) - double(row[id]));
      }
      const double rp = std::max(1.0 - pk, 1e-12), rq = std::max(1.0 - qk, 1e-12);
      kls.push_back(std::max(0.0, kl + rp * (std::log(rp) - std::log(rq))));
      (p < split ? kl_lo : kl_hi) += kls.back();
      ++(p < split ? n_lo : n_hi);
      agree += uint32_t(am) == top[0].first;
      nll -= row[ids[size_t(p + shift)]];
      nll_ref -= lp_ref;
      ++cnt;
    }
    double mean = 0;
    for (double k : kls) mean += k;
    mean /= std::max<size_t>(kls.size(), 1);
    std::vector<double> srt = kls;
    std::sort(srt.begin(), srt.end());
    auto pct = [&](double q) { return srt.empty() ? 0.0 : srt[std::min(srt.size() - 1, size_t(q * double(srt.size() - 1) + 0.5))]; };
    printf("{\"mode\": \"%s\", \"positions\": %ld, \"kl_mean\": %.6g, \"kl_median\": %.6g, \"kl_p99\": %.6g, \"kl_max\": %.6g, "
           "\"top1_agree\": %.5f, \"ppl\": %.5f, \"ppl_ref\": %.5f}\n",
           mode.c_str(), cnt, mean, pct(0.5), pct(0.99), pct(1.0), cnt ? double(agree) / double(cnt) : 0.0,
           cnt ? std::exp(nll / double(cnt)) : 0.0, cnt ? std::exp(nll_ref / double(cnt)) : 0.0);
    if (split > 0)
      printf("{\"kl_mean_before_%d\": %.6g, \"kl_mean_from_%d\": %.6g, \"positions_from\": %ld}\n", split,
             n_lo ? kl_lo / double(n_lo) : 0.0, split, n_hi ? kl_hi / double(n_hi) : 0.0, n_hi);
  }
  return 0;
}
