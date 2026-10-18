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
#include "metal/gpu.h"
#include "model/qwen4_exp.h"
#include "util/log.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;

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
  std::string out, xs_out, mode = "prefill";
  int chunk = 512, rows = 3, hist = 0;
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
  if (out.empty()) return 0;
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
  std::ofstream f(out, std::ios::binary);
  f.write(reinterpret_cast<const char*>(logits.data()), std::streamsize(logits.size() * 4));
  return 0;
}
