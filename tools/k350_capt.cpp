// #350 drafter fine-tune: training features from the target, one model load for many sequences.
//   k350_capt <model_dir> <list_file>
// Takes no GPU lock itself: run it under `lockf -k /tmp/kapitsa-gpu.lock`.
// Each list line is "<mode> <ids_file> <out_file>" (ids comma-separated):
//   c: prefill all ids with the drafter's capture hook; writes the drafter's context rows
//      c(j) = RMSNorm(x_ctx W_fc^T; hidden_norm) of the last min(n, 2047) positions as bf16 [rows][H].
//   d: prefill ids[0, n-1), anchor ids[n-1]; writes c rows as above and prints one drafting pass at P = n-1:
//      "DRAFT <ids_file> P <P> chain t1..t7 cand <7x16 ids>".
// K350_RAW=1 also writes the raw prompt captures (bf16 [rows][8*H]) to <out_file>.raw.
// The drafter is <model_dir>/dflash2 (DFlash2::find); NI_DFLASH_BITS picks its projections' bits (default 4).
#include "metal/gpu.h"
#include "model/dflash2.h"
#include "model/qwen35.h"
#include "model/qwen4_exp.h"
#include "util/bf16.h"
#include "util/log.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;

static std::vector<int> read_ids(const std::string& path) {
  std::ifstream f(path);
  std::string a((std::istreambuf_iterator<char>(f)), {}), t;
  std::stringstream ss(a);
  std::vector<int> v;
  while (std::getline(ss, t, ','))
    if (!t.empty() && t != "\n") v.push_back(std::stoi(t));
  return v;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: k350_capt <model_dir> <list_file>\n");
    return 1;
  }
  const std::string dir = argv[1];
  Gpu& g = Gpu::get();
  std::unique_ptr<Qwen35> mp = load_qwen(dir, 2048);
  Qwen35& m = *mp;
  const std::string ddir = DFlash2::find(dir, m.config());
  if (ddir.empty()) fatal("no DFlash2 drafter for %s", dir.c_str());
  DFlash2 dr(m, ddir);
  dr.set_pf8(0);  // every prompt commit through the 512-row path, which keeps c(j) in ctx_c()
  const int H = dr.config().H;
  const size_t nH = dr.config().target_layers.size() * size_t(H);
  const bool raw = getenv("K350_RAW") && atoi(getenv("K350_RAW"));
  fprintf(stderr, "k350_capt: drafter %s, %d-bit\n", ddir.c_str(), dr.bits());
  SamplingParams greedy;
  greedy.temperature = 0;
  std::ifstream lf(argv[2]);
  std::string line;
  int done = 0;
  long rows_total = 0;
  const double t0 = now_ms();
  std::vector<uint16_t> ob;
  while (std::getline(lf, line)) {
    std::stringstream ls(line);
    std::string mode, idf, outf;
    ls >> mode >> idf >> outf;
    if (mode.empty() || mode[0] == '#') continue;
    std::vector<int> ids = read_ids(idf);
    const bool draft = mode == "d";
    const int n = int(ids.size()) - (draft ? 1 : 0);
    if (n < 2) { fprintf(stderr, "skip %s: %d ids\n", idf.c_str(), n); continue; }
    auto seq = m.new_seq(n + 64);
    Seq& s = *seq;
    for (int i = 0; i < n; ++i) s.tokens()[i] = ids[i];
    auto d = dr.new_seq();
    dr.begin_prefill(n);
    for (int p = 0; p < n; p += m.prefill_chunk()) {
      const int T = std::min(m.prefill_chunk(), n - p);
      auto e = g.begin();
      m.encode_prefill(*e, s, p, T, greedy, 0, p + T == n);
      e->commit_and_wait();
      if (!m.ane_finish_chunk()) fatal("the NI_ANE prefill split failed; rerun with NI_ANE=0");
    }
    dr.end_prefill();
    const int rows = std::min(n, int(DFlash2::kCtxKeep));
    if (raw) {
      FILE* f = fopen((outf + ".raw").c_str(), "wb");
      if (!f) fatal("cannot write %s.raw", outf.c_str());
      fwrite(dr.prompt_captures().ptr(), 2, size_t(rows) * nH, f);
      fclose(f);
    }
    {
      auto e = g.begin();
      dr.encode_prefill_commit(*e, *d);
      e->commit_and_wait();
    }
    const float* c = dr.ctx_c().as<float>();
    ob.resize(size_t(rows) * H);
    for (size_t i = 0; i < ob.size(); ++i) ob[i] = f32_to_bf16(c[i]);
    FILE* f = fopen(outf.c_str(), "wb");
    if (!f) fatal("cannot write %s", outf.c_str());
    fwrite(ob.data(), 2, ob.size(), f);
    fclose(f);
    if (draft) {
      s.tokens()[n] = ids[n];
      auto e = g.begin();
      dr.encode_draft(*e, s, *d, n, greedy);
      e->commit_and_wait();
      printf("DRAFT %s P %d chain", idf.c_str(), n);
      for (int p = 0; p < 7; ++p) printf(" %d", s.tokens()[n + 1 + p]);
      printf(" cand");
      for (int i = 0; i < 7 * 16; ++i) printf(" %d", dr.cand_ids()[i]);
      printf("\n");
      fflush(stdout);
    }
    ++done;
    rows_total += rows;
    if (done % 50 == 0)
      fprintf(stderr, "k350_capt: %d sequences, %ld rows, %.0f s\n", done, rows_total, (now_ms() - t0) / 1000);
  }
  fprintf(stderr, "k350_capt: done %d sequences, %ld rows in %.0f s\n", done, rows_total, (now_ms() - t0) / 1000);
  return 0;
}
