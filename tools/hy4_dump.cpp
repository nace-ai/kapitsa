// Hy4-flash checks against a reference: every row's logits of a prompt, and greedy tokens after it.
//   hy4_dump <model> --ids FILE [--out F.f32] [--rows N] [--gen N] [--chunk R] [--verify] [--spec N]
// --verify: prefill all but the last 6 prompt tokens, then one 6-row verify pass over them; --out gets the verify's 6
// logit rows (rows n-6 .. n-1 of a plain run's). --spec N: N tokens greedy with the DFly2 drafter after the prompt
// (prints them, the cycles, accepted tokens per cycle and tok/s).
// FILE: comma- or whitespace-separated token ids. --out writes float32 [n, vocab] for rows 0..n-1 (n = --rows, default
// every prompt row): row p is the distribution of token p + 1. --gen N: N greedy tokens after the prompt (decode
// passes), printed as ids with tok/s. Takes the GPU lock (BENCH_GPU_LOCK, default /tmp/kapitsa-gpu.lock) unless the
// caller holds it (NI_LOCK_HELD=1).
#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/hy4_flash.h"
#include "util/log.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;

static std::vector<int> read_ids(const std::string& path) {
  std::ifstream f(path);
  if (!f) fatal("cannot read %s", path.c_str());
  std::string s((std::istreambuf_iterator<char>(f)), {});
  for (char& c : s)
    if (c == ',' || c == '[' || c == ']') c = ' ';
  std::stringstream ss(s);
  std::vector<int> v;
  int x;
  while (ss >> x) v.push_back(x);
  if (!ss.eof()) {  // a word that is not an integer: refuse it rather than drop the rest of the prompt
    ss.clear();
    std::string w;
    ss >> w;
    fatal("%s: \"%s\" (after %zu ids) is not a token id", path.c_str(), w.c_str(), v.size());
  }
  return v;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: hy4_dump <model> --ids FILE [--out F.f32] [--rows N] [--gen N] [--chunk R]\n");
    return 1;
  }
  std::string dir = argv[1], ids_file, out;
  int rows = -1, gen = 0, chunk = 512, spec = 0;
  std::string spec_check;  // --spec-check F.f32: teacher-forced logits of the prompt + these tokens (a debugging aid)
  bool verify = false, verify_draft = false;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) fatal("%s needs a value", a.c_str());
      return argv[++i];
    };
    if (a == "--ids") ids_file = next();
    else if (a == "--out") out = next();
    else if (a == "--rows") rows = atoi(next().c_str());
    else if (a == "--gen") gen = atoi(next().c_str());
    else if (a == "--chunk") chunk = atoi(next().c_str());
    else if (a == "--verify") verify = true;
    else if (a == "--verify-draft") verify = verify_draft = true;  // the drafter runs first (its tokens discarded)
    else if (a == "--spec") spec = atoi(next().c_str());
    else if (a == "--spec-check") spec_check = next();
    else fatal("unknown argument %s", a.c_str());
  }
  const std::vector<int> ids = read_ids(ids_file);
  const int n = int(ids.size());
  if (n < 1) fatal("no ids");
  if (rows < 0 || rows > n) rows = n;

  if (chunk < 1) fatal("--chunk %d: a pass needs at least 1 row", chunk);
  if (gen < 0) fatal("--gen %d: the number of tokens to generate is 0 or more", gen);

  bench::GpuLock lock;  // released at exit
  if (!bench::GpuLock::held_by_caller() && !lock.acquire([](const std::string& who) {
        fprintf(stderr, "waiting for the GPU lock %s%s\n", bench::GpuLock::path().c_str(),
                who.empty() ? "" : (", held by " + who).c_str());
      }))
    fatal("cannot take the GPU lock %s: %s", bench::GpuLock::path().c_str(), strerror(errno));

  const double t0 = now_ms();
  Hy4Flash m(dir, chunk);
  fprintf(stderr, "load %.1f s\n", (now_ms() - t0) / 1000);
  const int V = m.vocab();
  for (int i = 0; i < n; ++i)
    if (ids[size_t(i)] < 0 || ids[size_t(i)] >= V)
      fatal("id %d (at %d) is outside the vocabulary (0 .. %d)", ids[size_t(i)], i, V - 1);
  Gpu& g = Gpu::get();
  auto s = m.new_seq(n + gen + spec + 32);
  memcpy(s->tokens(), ids.data(), size_t(n) * 4);
  Buffer lg = g.alloc(size_t(chunk) * V * 4, "hy4_dump_logits");
  FILE* fo = out.empty() ? nullptr : fopen(out.c_str(), "wb");
  if (!out.empty() && !fo) fatal("cannot write %s", out.c_str());
  SamplingParams sp;
  sp.temperature = 0.0f;
  const double tp = now_ms();
  if (verify) {
    constexpr int R = Hy4Flash::kSpecRows;
    if (n <= R) fatal("--verify needs more than %d ids", R);
    const int np = n - R;
    for (int pos0 = 0; pos0 < np; pos0 += chunk) {
      auto e = g.begin();
      m.encode_prefill(*e, *s, pos0, std::min(chunk, np - pos0), sp, 0, false);
      e->commit_and_wait();
    }
    if (verify_draft) {  // the prefill's last row must sample t0 for the drafter: rerun the last prompt row sampled
      auto e0 = g.begin();
      m.encode_prefill(*e0, *s, np - 1, 1, sp, 0, true);
      e0->commit_and_wait();
      memcpy(s->tokens(), ids.data(), size_t(n) * 4);
      if (!getenv("HY4_DUMP_SKIP_DRAFT")) {  // (the control: the re-prefilled row alone)
        auto ed = g.begin();
        m.encode_draft(*ed, *s, np);
        ed->commit_and_wait();
      }
      memcpy(s->tokens(), ids.data(), size_t(n) * 4);
    }
    auto e = g.begin();
    m.encode_verify(*e, *s, np, R);
    m.encode_accept(*e, *s, np, R, sp);
    e->commit_and_wait();
    if (fo) {
      fwrite(m.verify_logits().ptr(), 4, size_t(R) * V, fo);
      fclose(fo);
    }
    printf("verify at %d: accepted %d (the prompt's own tokens as drafts)\n", np, m.accepted());
    return 0;
  }
  for (int pos0 = 0; pos0 < n; pos0 += chunk) {
    const int T = std::min(chunk, n - pos0);
    const bool last = pos0 + T == n;
    auto e = g.begin();
    m.encode_prefill(*e, *s, pos0, T, sp, 0, last && (gen > 0 || spec > 0));
    if (fo && pos0 < rows) m.encode_logits_rows(*e, T, lg);
    e->commit_and_wait();
    m.write_selection();  // NI_HY4_DUMP_SEL (debug); a no-op without it
    if (fo && pos0 < rows) {
      const int w = std::min(T, rows - pos0);
      fwrite(lg.ptr(), 4, size_t(w) * V, fo);
    }
    fprintf(stderr, "prefill rows %d-%d: %.1f s\n", pos0, pos0 + T - 1, (now_ms() - tp) / 1000);
  }
  if (fo) fclose(fo);
  const double pf = now_ms() - tp;
  fprintf(stderr, "prefill %d tokens: %.1f s (%.1f tok/s)\n", n, pf / 1000, n * 1000.0 / pf);
  if (gen > 0) {
    std::vector<int> outs = {s->tokens()[n]};
    const double td = now_ms();
    for (int k = 1; k < gen; ++k) {
      const int pos = n + k - 1;
      auto e = g.begin();
      m.encode_step(*e, *s, pos, sp, uint32_t(k));
      e->commit_and_wait();
      outs.push_back(s->tokens()[pos + 1]);
    }
    const double dt = now_ms() - td;
    printf("gen:");
    for (int t : outs) printf(" %d", t);
    printf("\n");
    if (gen > 1) fprintf(stderr, "decode %d tokens: %.1f s (%.2f tok/s)\n", gen - 1, dt / 1000, (gen - 1) * 1000.0 / dt);
  }
  if (spec > 0) {
    if (!m.has_dfly()) fatal("--spec: no DFly2 drafter in %s", dir.c_str());
    constexpr int B = Hy4Flash::kSpecRows;
    int P = n, cycles = 0;
    const double td = now_ms();
    double tdraft = 0;
    std::vector<int> hist(B + 1, 0);
    while (P - n < spec) {
      const double c0 = now_ms();
      auto e1 = g.begin();
      m.encode_draft(*e1, *s, P);
      e1->commit_and_wait();
      tdraft += now_ms() - c0;
      auto e2 = g.begin();
      m.encode_verify(*e2, *s, P, B);
      m.encode_accept(*e2, *s, P, B, sp);
      e2->commit_and_wait();
      const int a = m.accepted();
      if (!spec_check.empty()) {  // each verify row against the teacher-forced row of its position
        FILE* rf = fopen(spec_check.c_str(), "rb");
        std::vector<float> ref(static_cast<size_t>(V));
        const float* vl = m.verify_logits().as<float>();
        fprintf(stderr, "cycle %d P %d accepted %d:", cycles, P, a);
        for (int r = 0; r < B && rf; ++r) {
          if (fseeko(rf, off_t(P + r) * V * 4, SEEK_SET) || fread(ref.data(), 4, size_t(V), rf) != size_t(V)) break;
          const float* c = vl + size_t(r) * V;
          double ma = -1e30, mc = -1e30, sa = 0, sc = 0, kl = 0;
          int aa = 0, ac = 0;
          for (int v = 0; v < V; ++v) {
            if (ref[size_t(v)] > ma) ma = ref[size_t(v)], aa = v;
            if (c[v] > mc) mc = c[v], ac = v;
          }
          for (int v = 0; v < V; ++v) sa += exp(ref[size_t(v)] - ma), sc += exp(c[v] - mc);
          for (int v = 0; v < V; ++v) {
            const double la = ref[size_t(v)] - ma - log(sa), lc = c[v] - mc - log(sc);
            kl += exp(la) * (la - lc);
          }
          fprintf(stderr, " [r%d KL %.1e%s]", r, kl, aa == ac ? "" : " MISS");
        }
        fprintf(stderr, "\n");
        if (rf) fclose(rf);
      }
      ++hist[size_t(a + 1)];
      P += a + 1;
      ++cycles;
    }
    const double dt = now_ms() - td;
    printf("spec:");
    for (int i = n; i < P; ++i) printf(" %d", s->tokens()[i]);
    printf("\n");
    fprintf(stderr, "spec %d tokens in %d cycles (%.2f tokens/cycle; per-cycle histogram 1..6:", P - n, cycles,
            double(P - n) / cycles);
    for (int k = 1; k <= B; ++k) fprintf(stderr, " %d", hist[size_t(k)]);
    fprintf(stderr, "): %.1f s, %.2f tok/s, draft %.1f ms/cycle\n", dt / 1000, (P - n) * 1000.0 / dt, tdraft / cycles);
  }
  return 0;
}
