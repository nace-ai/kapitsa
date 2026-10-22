// S2: end-to-end exactness of the sampled DFlash2 loop on the real
// target and drafter. After a prompt, plain greedy steps move the start point to the first position whose
// sampler distribution is uncertain (>= 3 kept tokens, top probability <= 0.8); with the anchor y0 there,
// every trial runs DFlash2 cycles (fresh seed, state restored) until two tokens are out, and the empirical
// joint law of (t1, t2) is tested against p(t1) p(t2 | t1) from the same 8-row verify path (chi-square,
// cells with expected count < 5 pooled), as dflash2_ref.py `stat` does for the MLX reference.
//   dflash_stat <model_dir> @<ids file> [trials=2000] [temp=0.6] [--dflash <dir|auto>] [--min-warm N]
//               [--max-warm N] [--top-p 0.95] [--top-k 20]
#include "metal/gpu.h"
#include "model/dflash2.h"
#include "model/qwen35.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;

namespace {

double gamma_q(double a, double x) {
  if (x <= 0) return 1.0;
  const double gln = std::lgamma(a);
  if (x < a + 1) {
    double ap = a, sum = 1.0 / a, del = sum;
    for (int n = 0; n < 1000; ++n) {
      ap += 1;
      del *= x / ap;
      sum += del;
      if (std::fabs(del) < std::fabs(sum) * 1e-14) break;
    }
    return 1.0 - sum * std::exp(-x + a * std::log(x) - gln);
  }
  double b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
  for (int i = 1; i < 1000; ++i) {
    const double an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (std::fabs(d) < 1e-300) d = 1e-300;
    c = b + an / c;
    if (std::fabs(c) < 1e-300) c = 1e-300;
    d = 1 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1) < 1e-14) break;
  }
  return std::exp(-x + a * std::log(x) - gln) * h;
}

void chi2(const std::map<long, long>& obs, const std::map<long, double>& expp, long n, double* stat, int* dof,
          double* pval, long* outside) {
  double pool_o = 0, pool_e = 0, s = 0;
  int cells = 0;
  *outside = 0;
  for (auto& [k, c] : obs)
    if (!expp.count(k) || expp.at(k) <= 0) *outside += c;
  for (auto& [k, pr] : expp) {
    const double e = pr * n, o = obs.count(k) ? double(obs.at(k)) : 0.0;
    if (e < 5) {
      pool_o += o;
      pool_e += e;
      continue;
    }
    s += (o - e) * (o - e) / e;
    ++cells;
  }
  if (pool_e > 0) {
    s += (pool_o - pool_e) * (pool_o - pool_e) / std::max(pool_e, 1e-9);
    ++cells;
  }
  *stat = s;
  *dof = std::max(1, cells - 1);
  *pval = gamma_q(*dof / 2.0, s / 2.0);
}

uint32_t mix32(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return uint32_t(x);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: dflash_stat <model_dir> @<ids> [trials] [temp] [--dflash dir|auto] [--min-warm N] [--max-warm N]\n");
    return 1;
  }
  const std::string dir = argv[1];
  std::vector<int> ids;
  {
    std::string arg = argv[2];
    if (!arg.empty() && arg[0] == '@') {
      std::ifstream f(arg.substr(1));
      arg.assign(std::istreambuf_iterator<char>(f), {});
    }
    std::stringstream ss(arg);
    std::string t;
    while (std::getline(ss, t, ',')) ids.push_back(std::stoi(t));
  }
  const long trials = argc > 3 && argv[3][0] != '-' ? atol(argv[3]) : 2000;
  const float temp = argc > 4 && argv[4][0] != '-' ? float(atof(argv[4])) : 0.6f;
  std::string ddir = "auto";
  int min_warm = 8, max_warm = 400;
  float top_p = 0.95f;
  int top_k = 20;
  for (int i = 3; i + 1 < argc; ++i) {
    if (!strcmp(argv[i], "--dflash")) ddir = argv[i + 1];
    if (!strcmp(argv[i], "--min-warm")) min_warm = atoi(argv[i + 1]);
    if (!strcmp(argv[i], "--max-warm")) max_warm = atoi(argv[i + 1]);
    if (!strcmp(argv[i], "--top-p")) top_p = float(atof(argv[i + 1]));
    if (!strcmp(argv[i], "--top-k")) top_k = atoi(argv[i + 1]);
  }
  Gpu& g = Gpu::get();
  Qwen35 m(dir, 2048);
  if (ddir == "auto") ddir = DFlash2::find(dir, m.config());
  if (ddir.empty()) fatal("no DFlash2 drafter for %s", dir.c_str());
  DFlash2 dfl(m, ddir);
  auto d = dfl.new_seq();
  const int n = int(ids.size());
  auto seq = m.new_seq(n + max_warm + 64);
  Seq& s = *seq;
  for (int i = 0; i < n; ++i) s.tokens()[i] = ids[i];
  SamplingParams greedy;  // temperature 0
  SamplingParams sp;
  sp.temperature = temp;
  sp.top_p = top_p;
  sp.top_k = top_k;
  const int K = dfl.verify_depth();
  // prefill with the drafter's prompt capture
  dfl.begin_prefill(n, 0);
  for (int p = 0; p < n; p += m.prefill_chunk()) {
    const int T = std::min(m.prefill_chunk(), n - p);
    auto e = g.begin();
    m.encode_prefill(*e, s, p, T, greedy, 0, p + T == n);
    if (p + T == n) {
      dfl.end_prefill();
      dfl.encode_prefill_commit(*e, *d);
    }
    e->commit_and_wait();
    if (!m.ane_finish_chunk()) fatal("the NI_ANE prefill split failed; rerun with NI_ANE=0");
  }
  // the target's sampler distribution of the row after `pos` (an 8-row verify at pos: rows 0 and 1)
  struct Dist {
    std::vector<int> id;
    std::vector<float> p;
  };
  auto verify_dists = [&](int pos, Dist* d0, Dist* d1) {
    auto e = g.begin();
    dfl.arm_verify(pos);
    m.set_verify_moe_exact(dfl.exact_moe() && sp.temperature > 0);  // as the cycle verifies
    m.encode_verify(*e, s, pos, K + 1, sp, uint32_t(pos), true);
    m.set_verify_moe_exact(false);
    dfl.disarm();
    dfl.encode_dist(*e, 2, sp);
    e->commit_and_wait();
    m.commit_rows(s, 0);
    for (int r = 0; r < 2; ++r) {
      Dist* o = r == 0 ? d0 : d1;
      if (!o) continue;
      o->id.clear();
      o->p.clear();
      for (int q = 0; q < 32; ++q) {
        const int id = dfl.dist_ids()[r * 32 + q];
        if (id < 0) break;
        o->id.push_back(id);
        o->p.push_back(dfl.dist_probs()[r * 32 + q]);
      }
    }
  };
  // warm-up: greedy steps (committed to the target and the ring) to the first uncertain position
  int P = n, y0 = s.tokens()[n];
  Dist first;
  for (int w = 0;; ++w) {
    if (w >= min_warm) {
      s.tokens()[P] = y0;
      for (int i = 1; i <= 8; ++i) s.tokens()[P + i] = y0;
      verify_dists(P, &first, nullptr);
      const float top = first.p.empty() ? 1.0f : *std::max_element(first.p.begin(), first.p.end());
      if (first.id.size() >= 3 && top <= 0.8f) break;
      if (w >= max_warm) fatal("no uncertain position within %d tokens", max_warm);
    }
    auto e = g.begin();
    s.tokens()[P] = y0;
    dfl.arm_verify(P);
    m.encode_step(*e, s, P, greedy, uint32_t(P), true);  // feeds y0 (captured), samples the next greedy token
    dfl.disarm();
    dfl.encode_commit(*e, *d, 1);
    e->commit_and_wait();
    y0 = s.tokens()[P + 1];
    ++P;
  }
  // exact joint law from the verify path: p(t1) = row 0 at P; p(t2 | t1) = row 1 with tokens[P+1] = t1
  std::map<long, double> exact, marg;
  for (size_t i = 0; i < first.id.size(); ++i) {
    marg[first.id[i]] = first.p[i];
    s.tokens()[P] = y0;
    s.tokens()[P + 1] = first.id[i];
    Dist r1;
    verify_dists(P, nullptr, &r1);
    for (size_t j = 0; j < r1.id.size(); ++j) exact[long(first.id[i]) * 1000000 + r1.id[j]] = double(first.p[i]) * r1.p[j];
  }
  printf("# S2 %s T=%.2f top_p %.2f top_k %d: start position %d (warm %d), anchor %d, support t1 %zu, pairs %zu\n",
         dir.c_str(), temp, top_p, top_k, P, P - n, y0, first.id.size(), exact.size());
  fflush(stdout);
  Checkpoint ck = m.new_checkpoint();
  {
    auto e = g.begin();
    m.save(*e, s, ck, P);
    e->commit_and_wait();
  }
  const int ctx0 = d->ctx;
  std::map<long, long> obs, obs1;
  long a0_sum = 0, two_cycles = 0;
  const double t0 = now_ms();
  for (long t = 0; t < trials; ++t) {
    sp.seed = mix32(uint64_t(t) + 0x5eed0000ULL);
    s.tokens()[P] = y0;
    auto e = g.begin();
    dfl.encode_cycle(*e, s, *d, P, 0, K, sp);
    e->commit_and_wait();
    int a = dfl.accepted();
    a0_sum += a;
    const int t1 = s.tokens()[P + 1];
    int t2;
    if (a >= 1) {
      t2 = s.tokens()[P + 2];
    } else {
      ++two_cycles;
      m.commit_rows(s, 1);
      auto e2 = g.begin();
      dfl.encode_cycle(*e2, s, *d, P + 1, 1, K, sp);
      e2->commit_and_wait();
      t2 = s.tokens()[P + 2];
    }
    obs[long(t1) * 1000000 + t2]++;
    obs1[t1]++;
    // restore: target recurrent state, no pending rows, ring back to the start
    m.commit_rows(s, 0);
    auto e3 = g.begin();
    m.restore(*e3, s, ck);
    e3->commit_and_wait();
    DFlash2::rewind(*d, ctx0);
  }
  const double secs = (now_ms() - t0) / 1000;
  double st, pv, st1, pv1;
  int dof, dof1;
  long out, out1;
  chi2(obs, exact, trials, &st, &dof, &pv, &out);
  chi2(obs1, marg, trials, &st1, &dof1, &pv1, &out1);
  printf("# %ld trials in %.0f s, mean a of the first cycle %.2f, second cycle needed in %ld\n", trials, secs,
         double(a0_sum) / trials, two_cycles);
  printf("S2 joint chi2 %.1f (%d dof) p=%.3f; first token chi2 %.1f (%d dof) p=%.3f; outside support %ld: %s\n", st, dof,
         pv, st1, dof1, pv1, out + out1, pv > 0.01 && pv1 > 0.01 && out + out1 == 0 ? "PASS" : "FAIL");
  // the largest cells
  std::vector<std::pair<long, long>> top(obs.begin(), obs.end());
  std::sort(top.begin(), top.end(), [](auto& a, auto& b) { return a.second > b.second; });
  for (size_t i = 0; i < std::min<size_t>(8, top.size()); ++i)
    printf("  (%ld, %ld): observed %ld, expected %.1f\n", top[i].first / 1000000, top[i].first % 1000000, top[i].second,
           exact.count(top[i].first) ? exact[top[i].first] * trials : 0.0);
  return pv > 0.01 && pv1 > 0.01 && out + out1 == 0 ? 0 : 1;
}
