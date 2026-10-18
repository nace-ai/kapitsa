// Teacher-forced next-token distributions over token sequences: the quality metric of a quantization
// (bench/quality/README.md).
//   ni_eval <model> [--ref-model <model>] [--ref <file>] [--dump-ref <file>] [--topk K] [--chunk R] [--skip S]
//           @ids1 [@ids2 ...]
//  --ref-model M  both models run in this process; per position the exact KL(ref || model) over the vocabulary
//  --dump-ref F   writes this model's per-position top-K log-probs (default K = 256) and the realized next
//                 token's log-prob: a reference file (also written by bench/quality/eval_stream.py)
//  --ref F        compares with such a file: KL over the reference's top-K with the rest of the mass lumped
//                 (a lower bound of the exact KL; with --ref-model both are printed)
// Scores positions [skip, n - 1) of each sequence (n tokens: position p predicts token p + 1). Prints one JSON
// line per sequence and a total: KL mean / median / p95 / p99 / max, top-1 agreement with the reference,
// perplexity of the model (and of the reference). Prefill chunks of R rows (default 512); the logits come from
// the prefill GEMM over the final-norm rows. Set NI_ANE=0 to keep the Neural Engine split out of the numbers.
//  --image F      (repeatable) the images of the sequences, in the order of their runs of image placeholder tokens:
//                 each run takes the image's embeddings and M-RoPE positions, as the server does (src/vision)
#include "metal/gpu.h"
#include "model/qwen35.h"
#include "model/qwen4_exp.h"
#include "util/log.h"
#include "vision/encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
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

// log-softmax of one row in place; returns argmax
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
  std::vector<std::pair<uint32_t, float>> top;  // (id, log-prob), descending
};

struct Stats {
  std::vector<double> kl, kl_top;
  double nll = 0, nll_ref = 0;
  long agree = 0, n = 0;
  void add(const Stats& o) {
    kl.insert(kl.end(), o.kl.begin(), o.kl.end());
    kl_top.insert(kl_top.end(), o.kl_top.begin(), o.kl_top.end());
    nll += o.nll;
    nll_ref += o.nll_ref;
    agree += o.agree;
    n += o.n;
  }
  static double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(q * double(v.size() - 1) + 0.5))];
  }
  std::string json(const std::string& name) const {
    auto mean = [](const std::vector<double>& v) {
      double s = 0;
      for (double x : v) s += x;
      return v.empty() ? 0.0 : s / double(v.size());
    };
    char b[1024];
    const std::vector<double>& k = kl.empty() ? kl_top : kl;
    snprintf(b, sizeof b,
             "{\"seq\": \"%s\", \"positions\": %ld, \"kl_mean\": %.6g, \"kl_median\": %.6g, \"kl_p95\": %.6g, "
             "\"kl_p99\": %.6g, \"kl_max\": %.6g, \"kl_topk_mean\": %.6g, \"top1_agree\": %.5f, \"ppl\": %.5f, "
             "\"ppl_ref\": %.5f, \"exact\": %s}",
             name.c_str(), n, mean(k), pct(k, 0.5), pct(k, 0.95), pct(k, 0.99), pct(k, 1.0), mean(kl_top),
             n ? double(agree) / double(n) : 0.0, n ? std::exp(nll / double(n)) : 0.0,
             n ? std::exp(nll_ref / double(n)) : 0.0, kl.empty() ? "false" : "true");
    return b;
  }
};

// KL(ref || q) over the reference's top-K plus the lumped rest; q = full log-probs
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

void top_of(const float* lp, int V, int K, RefPos& out) {
  std::vector<uint32_t> idx(V);
  for (int i = 0; i < V; ++i) idx[i] = uint32_t(i);
  std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](uint32_t a, uint32_t b) { return lp[a] > lp[b]; });
  out.top.resize(K);
  for (int i = 0; i < K; ++i) out.top[i] = {idx[i], lp[idx[i]]};
}

// --image: the images of the sequences, and their encoder
struct Images {
  std::vector<std::string> files;
  std::unique_ptr<vision::ImageEncoder> enc;
  size_t next = 0;  // images are taken in order across the sequences
  std::vector<Buffer> bufs;
};

// The image rows of ids (runs of the placeholder token, one per image, in order): embeddings and positions into s.
void set_images(Qwen35& m, Seq& s, const std::vector<int>& ids, Images& im, std::vector<vision::EmbedRun>& runs) {
  if (im.files.empty()) return;
  std::vector<vision::ImageSpan> spans;
  const int ph = im.enc->placeholder(), n = int(ids.size());
  for (int i = 0; i < n;) {
    if (ids[i] != ph) {
      ++i;
      continue;
    }
    int j = i;
    while (j < n && ids[j] == ph) ++j;
    if (im.next >= im.files.size()) fatal("more image runs than --image files");
    const std::string& f = im.files[im.next++];
    std::ifstream in(f, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(in)), {});
    vision::Prepared p = im.enc->prepare(vision::decode(bytes));
    if (p.tokens != j - i) fatal("%s: %d image tokens in the sequence, the processor makes %d", f.c_str(), j - i, p.tokens);
    if (!im.enc->loaded()) im.enc->load();
    Buffer b = Gpu::get().alloc(size_t(p.tokens) * im.enc->out_dim() * 2 + 16, "image_embeddings");
    const double t0 = now_ms();
    im.enc->encode(p, b);
    fprintf(stderr, "image %s: %dx%d -> %d tokens, encoded in %.0f ms\n", f.c_str(), p.width, p.height, p.tokens,
            now_ms() - t0);
    im.bufs.push_back(b);
    runs.push_back(vision::EmbedRun{i, p.tokens, b, 0, im.enc->out_dim()});
    spans.push_back(vision::ImageSpan{i, p.tokens, p.grid_t, p.grid_h, p.grid_w});
    i = j;
  }
  m.set_rope_positions(s, n, spans);
  s.embeds = runs.empty() ? nullptr : &runs;
}

// Runs `m` over the sequence chunk by chunk; calls fn(row pointer, position) for every position < n - 1.
template <class Fn>
void run_model(Qwen35& m, const std::vector<int>& ids, int chunk, const Buffer& logits, Images& im, Fn&& fn) {
  const int n = int(ids.size()), V = m.config().vocab;
  auto s = m.new_seq(n + 16);
  memcpy(s->tokens(), ids.data(), ids.size() * 4);
  std::vector<vision::EmbedRun> runs;
  set_images(m, *s, ids, im, runs);
  SamplingParams sp;
  for (int pos0 = 0; pos0 < n; pos0 += chunk) {
    const int T = std::min(chunk, n - pos0);
    auto e = Gpu::get().begin();
    m.encode_prefill(*e, *s, pos0, T, sp, 0, false);
    m.encode_logits_rows(*e, T, logits);
    e->commit_and_wait();
    if (!m.ane_finish_chunk()) fatal("Neural Engine split failed on a chunk; run with NI_ANE=0");
    for (int r = 0; r < T; ++r)
      if (pos0 + r < n - 1) fn(logits.as<float>() + size_t(r) * V, pos0 + r);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: ni_eval <model> [--ref-model M] [--ref F] [--dump-ref F] [--topk K] [--chunk R] [--skip S] "
                    "@ids...\n");
    return 1;
  }
  std::string model_dir = argv[1], ref_model, ref_file, dump_file;
  int topk = 256, chunk = 512, skip = 0;
  std::vector<std::string> seqs;
  Images images;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) fatal("%s needs a value", a.c_str());
      return argv[++i];
    };
    if (a == "--ref-model") ref_model = next();
    else if (a == "--ref") ref_file = next();
    else if (a == "--dump-ref") dump_file = next();
    else if (a == "--topk") topk = std::stoi(next());
    else if (a == "--chunk") chunk = std::stoi(next());
    else if (a == "--skip") skip = std::stoi(next());
    else if (a == "--image") images.files.push_back(next());
    else seqs.push_back(a);
  }
  if (seqs.empty()) fatal("no sequences");
  std::vector<std::vector<int>> ids;
  for (auto& s : seqs) ids.push_back(read_ids(s));

  // reference file
  std::vector<std::vector<RefPos>> refpos;
  if (!ref_file.empty()) {
    std::ifstream f(ref_file, std::ios::binary);
    char magic[8];
    uint32_t nseq = 0, V = 0, K = 0;
    f.read(magic, 8);
    f.read((char*)&nseq, 4);
    f.read((char*)&V, 4);
    f.read((char*)&K, 4);
    if (!f || memcmp(magic, "NIEVREF2", 8) || nseq != ids.size()) fatal("%s: not a reference for these sequences", ref_file.c_str());
    refpos.resize(nseq);
    for (uint32_t q = 0; q < nseq; ++q) {
      uint32_t n = 0;
      f.read((char*)&n, 4);
      std::vector<int> rid(n);
      f.read((char*)rid.data(), n * 4);
      if (rid != ids[q]) fatal("%s: sequence %u differs from %s", ref_file.c_str(), q, seqs[q].c_str());
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

  std::unique_ptr<Qwen35> ref;
  if (!ref_model.empty()) ref = load_qwen(ref_model, chunk);
  std::unique_ptr<Qwen35> mp = load_qwen(model_dir, chunk);
  Qwen35& m = *mp;
  if (!images.files.empty() && !(images.enc = vision::open_encoder(model_dir))) fatal("%s takes no images", model_dir.c_str());
  if (ref && !images.files.empty()) fatal("--image with --ref-model is not supported");
  const int V = m.config().vocab;
  if (ref && ref->config().vocab != V) fatal("vocabularies differ");
  Gpu& g = Gpu::get();
  Buffer lg = g.alloc(size_t(chunk) * V * 4, "eval_logits"), lr;
  if (ref) lr = g.alloc(size_t(chunk) * V * 4, "eval_logits_ref");

  std::ofstream dump;
  if (!dump_file.empty()) {
    dump.open(dump_file, std::ios::binary);
    const uint32_t nseq = uint32_t(ids.size()), Vu = uint32_t(V), Ku = uint32_t(topk);
    dump.write("NIEVREF2", 8);
    dump.write((const char*)&nseq, 4);
    dump.write((const char*)&Vu, 4);
    dump.write((const char*)&Ku, 4);
  }
  Stats total;
  const unsigned nth = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
  for (size_t q = 0; q < ids.size(); ++q) {
    const std::vector<int>& seq = ids[q];
    const int n = int(seq.size());
    // the reference's log-probs per position (full rows only within a chunk: compared chunk by chunk)
    std::vector<RefPos> mine(n > 0 ? n - 1 : 0);
    Stats st;
    if (ref) {
      // both models chunk by chunk: exact KL from the two logit blocks
      auto sr = ref->new_seq(n + 16), sm = m.new_seq(n + 16);
      memcpy(sr->tokens(), seq.data(), seq.size() * 4);
      memcpy(sm->tokens(), seq.data(), seq.size() * 4);
      SamplingParams sp;
      for (int pos0 = 0; pos0 < n; pos0 += chunk) {
        const int T = std::min(chunk, n - pos0);
        for (int w = 0; w < 2; ++w) {
          Qwen35& mm = w ? m : *ref;
          auto e = g.begin();
          mm.encode_prefill(*e, w ? *sm : *sr, pos0, T, sp, 0, false);
          mm.encode_logits_rows(*e, T, w ? lg : lr);
          e->commit_and_wait();
          if (!mm.ane_finish_chunk()) fatal("Neural Engine split failed on a chunk; run with NI_ANE=0");
        }
        std::vector<Stats> part(nth);
        std::vector<std::thread> th;
        for (unsigned t = 0; t < nth; ++t)
          th.emplace_back([&, t] {
            for (int r = int(t); r < T; r += int(nth)) {
              const int p = pos0 + r;
              if (p >= n - 1 || p < skip) continue;
              float* a = lr.as<float>() + size_t(r) * V;
              float* b = lg.as<float>() + size_t(r) * V;
              const int am_r = log_softmax(a, V), am_m = log_softmax(b, V);
              double kl = 0;
              for (int i = 0; i < V; ++i) {
                const double pr = std::exp(double(a[i]));
                if (pr > 0) kl += pr * (double(a[i]) - double(b[i]));
              }
              RefPos rp;
              top_of(a, V, std::min(topk, V), rp);
              Stats& s = part[t];
              s.kl.push_back(std::max(kl, 0.0));
              s.kl_top.push_back(std::max(kl_topk(rp, b), 0.0));
              s.nll -= b[seq[p + 1]];
              s.nll_ref -= a[seq[p + 1]];
              s.agree += am_r == am_m;
              ++s.n;
            }
          });
        for (auto& t : th) t.join();
        for (auto& s : part) st.add(s);
      }
    } else {
      run_model(m, seq, chunk, lg, images, [&](float* row, int p) {
        const int am = log_softmax(row, V);
        if (dump) {
          mine[p].lp_next = row[seq[p + 1]];
          top_of(row, V, std::min(topk, V), mine[p]);
        }
        if (p < skip) return;
        st.nll -= row[seq[p + 1]];
        ++st.n;
        if (!refpos.empty()) {
          const RefPos& rp = refpos[q][p];
          st.kl_top.push_back(std::max(kl_topk(rp, row), 0.0));
          st.nll_ref -= rp.lp_next;
          st.agree += uint32_t(am) == rp.top[0].first;
        }
      });
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
    printf("%s\n", st.json(seqs[q]).c_str());
    fflush(stdout);
    total.add(st);
  }
  printf("%s\n", total.json("total").c_str());
  return 0;
}
