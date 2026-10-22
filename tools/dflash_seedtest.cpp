// Sampled exactness through the REAL decode loops (complements dflash_stat, which tests one-command-buffer
// cycles against the verify path's own law): many seeds, compared with the exact law of the PLAIN sampler.
//   dflash_seedtest <model_dir> @<ids> --arm plain|dflash [--dflash dir] [--trials N] [--temp T] [--len L]
//                [--seed-base B] [--warm W] [--out samples.txt] [--law-out law.txt]   (--trials 0: laws only)
// Start point: prompt + W greedy tokens (W = auto: first greedy position whose T-distribution keeps >= 3
// tokens with top <= 0.8 and whose greedy successor keeps >= 2 with top <= 0.9).
// Exact law (plain decode path): p(t1) from a 1-token prefill of the last prompt row, p(t2|t1) and
// p(t3|t1,t2) from plain decode steps; the sampler distribution is rebuilt on the CPU from the logits
// exactly as sample_stage2 does (global top-k by (value desc, id asc), top-p on the untempered
// probabilities, then temperature).
// Trials: restore the checkpoint, 1-token prefill (sampled, as the engine's prefill), then either the
// engine's plain decode steps or the engine's double-buffered DFlash2 loop (copied from Engine::chat)
// until L tokens exist. Writes one line per trial "t1 t2 .. tL cycles a0".
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
    for (int n = 0; n < 2000; ++n) {
      ap += 1;
      del *= x / ap;
      sum += del;
      if (std::fabs(del) < std::fabs(sum) * 1e-15) break;
    }
    return 1.0 - sum * std::exp(-x + a * std::log(x) - gln);
  }
  double b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
  for (int i = 1; i < 2000; ++i) {
    const double an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (std::fabs(d) < 1e-300) d = 1e-300;
    c = b + an / c;
    if (std::fabs(c) < 1e-300) c = 1e-300;
    d = 1 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1) < 1e-15) break;
  }
  return std::exp(-x + a * std::log(x) - gln) * h;
}

uint32_t mix32(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return uint32_t(x);
}

struct Dist {
  std::vector<int> id;
  std::vector<double> p;
};

// sample_stage2's distribution from full logits (CPU, fp32 arithmetic in the kernel's order)
Dist plain_dist(const float* l, int V, const SamplingParams& sp) {
  const int K = std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32);
  std::vector<int> idx(V);
  for (int i = 0; i < V; ++i) idx[i] = i;
  std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) {
    return l[a] > l[b] || (l[a] == l[b] && a < b);
  });
  Dist d;
  float probs[64], tot = 0.0f;
  for (int q = 0; q < K; ++q) {
    probs[q] = std::exp(l[idx[q]] - l[idx[0]]);
    tot += probs[q];
  }
  int keep = K;
  float cum = 0.0f;
  for (int q = 0; q < K; ++q) {
    cum += probs[q] / tot;
    if (cum >= sp.top_p) {
      keep = q + 1;
      break;
    }
  }
  float kt = 0.0f;
  for (int q = 0; q < keep; ++q) {
    probs[q] = std::exp((l[idx[q]] - l[idx[0]]) / sp.temperature);
    kt += probs[q];
  }
  for (int q = 0; q < keep; ++q) {
    d.id.push_back(idx[q]);
    d.p.push_back(double(probs[q]) / kt);
  }
  return d;
}

struct Chi {
  double stat = 0, p = 1;
  int dof = 0, cells = 0;
  long outside = 0;
};
Chi chi2(const std::map<std::vector<int>, long>& obs, const std::map<std::vector<int>, double>& ex, long n) {
  Chi c;
  for (auto& [k, v] : obs)
    if (!ex.count(k)) n -= v;
  double po = 0, pe = 0;
  for (auto& [k, v] : obs)
    if (!ex.count(k)) c.outside += v;
  for (auto& [k, pr] : ex) {
    const double e = pr * n, o = obs.count(k) ? double(obs.at(k)) : 0.0;
    if (e < 5) {
      po += o;
      pe += e;
      continue;
    }
    c.stat += (o - e) * (o - e) / e;
    ++c.cells;
  }
  // samples outside the exact support (e.g. a top-p boundary moved by the verify numerics) are reported,
  // not tested: n counts only the samples inside the support
  if (pe > 0) {
    c.stat += (po - pe) * (po - pe) / std::max(pe, 1e-9);
    ++c.cells;
  }
  c.dof = std::max(1, c.cells - 1);
  c.p = gamma_q(c.dof / 2.0, c.stat / 2.0);
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: dflash_seedtest <model_dir> @<ids> --arm plain|dflash [--dflash dir] [--trials N] ...\n");
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
  std::string arm = "dflash", ddir = "auto", out_path, law_path;
  long trials = 2000;
  int L = 3, warm = -1, max_warm = 400;
  uint64_t seed_base = 1000;
  float temp = 0.6f;
  for (int i = 3; i + 1 < argc; ++i) {
    if (!strcmp(argv[i], "--arm")) arm = argv[i + 1];
    if (!strcmp(argv[i], "--dflash")) ddir = argv[i + 1];
    if (!strcmp(argv[i], "--trials")) trials = atol(argv[i + 1]);
    if (!strcmp(argv[i], "--temp")) temp = float(atof(argv[i + 1]));
    if (!strcmp(argv[i], "--len")) L = atoi(argv[i + 1]);
    if (!strcmp(argv[i], "--seed-base")) seed_base = strtoull(argv[i + 1], nullptr, 10);
    if (!strcmp(argv[i], "--warm")) warm = atoi(argv[i + 1]);
    if (!strcmp(argv[i], "--out")) out_path = argv[i + 1];
    if (!strcmp(argv[i], "--law-out")) law_path = argv[i + 1];
  }
  if (L < 2 || L > 3) fatal("--len 2 or 3");
  const bool dfl_arm = arm == "dflash";
  Gpu& g = Gpu::get();
  Qwen35 m(dir, 2048);
  const int V = m.config().vocab;
  SamplingParams greedy, sp;
  sp.temperature = temp;
  sp.top_p = 0.95f;
  sp.top_k = 20;
  auto seq = m.new_seq(int(ids.size()) + max_warm + 64);
  Seq& s = *seq;

  // ---- start point: prompt + W greedy tokens (plain decode, logits per step) ----
  std::vector<int> pr = ids;
  {
    for (size_t i = 0; i < ids.size(); ++i) s.tokens()[i] = ids[i];
    const int n0 = int(ids.size());
    std::vector<std::vector<float>> lg;  // logits predicting position n0 + w
    for (int p = 0; p < n0; p += m.prefill_chunk()) {
      const int T = std::min(m.prefill_chunk(), n0 - p);
      auto e = g.begin();
      m.encode_prefill(*e, s, p, T, greedy, 0, p + T == n0);
      e->commit_and_wait();
      if (!m.ane_finish_chunk()) fatal("the NI_ANE prefill split failed; rerun with NI_ANE=0");
    }
    lg.emplace_back(m.logits(), m.logits() + V);
    const int wmax = warm >= 0 ? warm + 1 : max_warm;
    for (int w = 0; w < wmax; ++w) {
      auto e = g.begin();
      m.encode_step(*e, s, n0 + w, greedy, uint32_t(n0 + w), true);
      e->commit_and_wait();
      lg.emplace_back(m.logits(), m.logits() + V);
    }
    int W = warm;
    if (W < 0) {
      for (int w = 0; w + 1 < int(lg.size()); ++w) {
        Dist a = plain_dist(lg[w].data(), V, sp), b = plain_dist(lg[w + 1].data(), V, sp);
        const double ta = *std::max_element(a.p.begin(), a.p.end()), tb = *std::max_element(b.p.begin(), b.p.end());
        if (w >= 4 && a.id.size() >= 3 && ta <= 0.8 && b.id.size() >= 2 && tb <= 0.9) {
          W = w;
          break;
        }
      }
      if (W < 0) fatal("no uncertain start within %d tokens", max_warm);
    }
    for (int w = 0; w < W; ++w) pr.push_back(s.tokens()[n0 + w]);
    m.reset(s);
  }
  const int n = int(pr.size());
  for (int i = 0; i < n; ++i) s.tokens()[i] = pr[i];

  std::unique_ptr<DFlash2> dfl;
  std::unique_ptr<DFlashSeq> d;
  if (dfl_arm) {
    if (ddir == "auto") ddir = DFlash2::find(dir, m.config());
    dfl = std::make_unique<DFlash2>(m, ddir);
    d = dfl->new_seq();
  }
  // ---- prefill [0, n-1) (+ drafter capture), checkpoint at n-1 ----
  if (dfl) dfl->begin_prefill(n - 1, 0);
  for (int p = 0; p < n - 1; p += m.prefill_chunk()) {
    const int T = std::min(m.prefill_chunk(), n - 1 - p);
    auto e = g.begin();
    m.encode_prefill(*e, s, p, T, greedy, 0, false);
    if (dfl && p + T == n - 1) {
      dfl->end_prefill();
      dfl->encode_prefill_commit(*e, *d);
    }
    e->commit_and_wait();
    if (!m.ane_finish_chunk()) fatal("the NI_ANE prefill split failed; rerun with NI_ANE=0");
  }
  Checkpoint ck = m.new_checkpoint();
  {
    auto e = g.begin();
    m.save(*e, s, ck, n - 1);
    e->commit_and_wait();
  }
  auto restore = [&]() {
    m.commit_rows(s, 0);
    auto e = g.begin();
    m.restore(*e, s, ck);
    e->commit_and_wait();
    for (int i = 0; i < n; ++i) s.tokens()[i] = pr[i];
  };
  // last prompt row: 1-token prefill (sampled with sp at step 0, as Engine::chat's prefill)
  auto last_row = [&](const SamplingParams& x, bool capture) {
    if (capture) {
      DFlash2::rewind(*d, n - 1);
      dfl->begin_prefill(n, n - 1);
    }
    auto e = g.begin();
    m.encode_prefill(*e, s, n - 1, 1, x, 0, true);
    if (capture) {
      dfl->end_prefill();
      dfl->encode_prefill_commit(*e, *d);
    }
    e->commit_and_wait();
  };

  // ---- exact law under plain decoding ----
  std::map<std::vector<int>, double> ex1, ex2, ex3;
  {
    restore();
    last_row(greedy, false);
    Dist p1 = plain_dist(m.logits(), V, sp);
    for (size_t i = 0; i < p1.id.size(); ++i) {
      ex1[{p1.id[i]}] = p1.p[i];
      restore();
      last_row(greedy, false);
      s.tokens()[n] = p1.id[i];
      {
        auto e = g.begin();
        m.encode_step(*e, s, n, greedy, uint32_t(n), true);
        e->commit_and_wait();
      }
      Dist p2 = plain_dist(m.logits(), V, sp);
      for (size_t j = 0; j < p2.id.size(); ++j) {
        const double pij = p1.p[i] * p2.p[j];
        ex2[{p1.id[i], p2.id[j]}] = pij;
        if (L < 3 || pij * (trials > 0 ? trials : 4000) < 2) continue;  // third token only where the pair is visible
        restore();
        last_row(greedy, false);
        s.tokens()[n] = p1.id[i];
        {
          auto e = g.begin();
          m.encode_step(*e, s, n, greedy, uint32_t(n), false);
          e->commit_and_wait();
        }
        s.tokens()[n + 1] = p2.id[j];
        {
          auto e = g.begin();
          m.encode_step(*e, s, n + 1, greedy, uint32_t(n + 1), true);
          e->commit_and_wait();
        }
        Dist p3 = plain_dist(m.logits(), V, sp);
        for (size_t k = 0; k < p3.id.size(); ++k) ex3[{p1.id[i], p2.id[j], p3.id[k]}] = pij * p3.p[k];
      }
    }
    // --law-out: the same law with the second / third token's distribution taken from the 8-row verify (rows 0
    // and 1 of a verify at the start point, row-independent MoE numerics as the sampled cycle uses), i.e. the
    // law a DFlash2 cycle is exact against; samples can then be tested against both (two numerics, one algorithm)
    if (!law_path.empty()) {
      auto vdist = [&](int t1, int t2, int row) {
        restore();
        last_row(greedy, false);
        s.tokens()[n] = t1;
        for (int i = 1; i <= 8; ++i) s.tokens()[n + i] = t2 >= 0 ? t2 : t1;
        auto e = g.begin();
        m.set_verify_moe_exact(true);
        m.encode_verify(*e, s, n, 8, sp, uint32_t(n), true);
        m.set_verify_moe_exact(false);
        e->commit_and_wait();
        m.commit_rows(s, 0);
        return plain_dist(m.logits() + size_t(row) * V, V, sp);
      };
      FILE* fl = fopen(law_path.c_str(), "w");
      for (auto& [k, v] : ex1) fprintf(fl, "1 %d %.9g\n", k[0], v);
      for (auto& [k1, v1] : ex1) {
        Dist pv = vdist(k1[0], -1, 0);
        std::map<int, double> pvm;
        for (size_t j = 0; j < pv.id.size(); ++j) pvm[pv.id[j]] = pv.p[j];
        std::map<int, double> ppm;
        for (auto& [k, v] : ex2)
          if (k[0] == k1[0]) ppm[k[1]] = v / v1;
        for (auto& [t2, pp] : ppm) fprintf(fl, "2 %d %d %.9g %.9g\n", k1[0], t2, pp, pvm.count(t2) ? pvm[t2] : 0.0);
        for (auto& [t2, pq] : pvm)
          if (!ppm.count(t2)) fprintf(fl, "2 %d %d 0 %.9g\n", k1[0], t2, pq);
      }
      std::map<std::vector<int>, bool> pairs3;
      for (auto& [k, v] : ex3) pairs3[{k[0], k[1]}] = true;
      for (auto& [pk, unused] : pairs3) {
        (void)unused;
        Dist pv = vdist(pk[0], pk[1], 1);
        std::map<int, double> pvm, ppm;
        for (size_t j = 0; j < pv.id.size(); ++j) pvm[pv.id[j]] = pv.p[j];
        const double pij = ex2[pk];
        for (auto& [k, v] : ex3)
          if (k[0] == pk[0] && k[1] == pk[1]) ppm[k[2]] = v / pij;
        for (auto& [t3, pp] : ppm) fprintf(fl, "3 %d %d %d %.9g %.9g\n", pk[0], pk[1], t3, pp, pvm.count(t3) ? pvm[t3] : 0.0);
        for (auto& [t3, pq] : pvm)
          if (!ppm.count(t3)) fprintf(fl, "3 %d %d %d 0 %.9g\n", pk[0], pk[1], t3, pq);
      }
      fclose(fl);
    }
    printf("# %s T=%.2f start position %d (prompt %zu + warm %d), support t1 %zu, pairs %zu, triples %zu\n",
           dir.c_str(), temp, n, ids.size(), n - int(ids.size()), ex1.size(), ex2.size(), ex3.size());
    for (auto& [k, v] : ex1) printf("#   p(t1=%d) = %.4f\n", k[0], v);
    fflush(stdout);
  }

  if (trials <= 0) return 0;
  // ---- trials ----
  FILE* fo = out_path.empty() ? nullptr : fopen(out_path.c_str(), "w");
  std::map<std::vector<int>, long> ob1, ob2, ob3;
  long cyc_sum = 0, a0_hist[9] = {0};
  const int K = dfl->verify_depth();
  const double t0 = now_ms();
  for (long t = 0; t < trials; ++t) {
    sp.seed = mix32(seed_base + uint64_t(t));
    restore();
    last_row(sp, dfl_arm);
    std::vector<int> out = {s.tokens()[n]};
    int cycles = 0, a0 = -1;
    if (!dfl_arm) {
      for (int q = n; int(out.size()) < L; ++q) {  // Engine::chat plain loop: step = position
        auto e = g.begin();
        m.encode_step(*e, s, q, sp, uint32_t(q), true);
        e->commit_and_wait();
        out.push_back(s.tokens()[q + 1]);
      }
    } else {
      // Engine::chat's DFlash2 loop (double-buffered), cut at L tokens
      DFlashSeq& dd = *d;
      int P = n, mm = 0;
      std::unique_ptr<Encoder> next_draft;
      dfl->gpu_state_reset(P);
      next_draft = g.begin();
      dfl->encode_cycle_draft_gpu(*next_draft, s, dd, P, sp);
      next_draft->commit();
      while (int(out.size()) < L) {
        next_draft.reset();
        auto e2 = g.begin();
        dfl->encode_cycle_verify(*e2, s, P, K, sp);
        e2->commit();
        next_draft = g.begin();
        dfl->encode_cycle_draft_gpu(*next_draft, s, dd, P + 8, sp);
        next_draft->commit();
        e2->wait();
        const int a = dfl->accepted();
        if (a0 < 0) a0 = a;
        const int* tk = s.tokens();
        mm = 0;
        while (mm < a + 1 && int(out.size()) < L) {
          ++mm;
          out.push_back(tk[P + mm]);
        }
        ++cycles;
        m.commit_rows(s, mm);
        P += mm;
        dd.ctx = P;
      }
      if (next_draft) next_draft->wait();
    }
    cyc_sum += cycles;
    if (a0 >= 0 && a0 <= 8) a0_hist[a0]++;
    ob1[{out[0]}]++;
    ob2[{out[0], out[1]}]++;
    if (L >= 3) ob3[{out[0], out[1], out[2]}]++;
    if (fo) {
      for (int x : out) fprintf(fo, "%d ", x);
      fprintf(fo, "%d %d\n", cycles, a0);
    }
  }
  if (fo) fclose(fo);
  const double secs = (now_ms() - t0) / 1000;
  Chi c1 = chi2(ob1, ex1, trials), c2 = chi2(ob2, ex2, trials);
  printf("# arm %s: %ld trials in %.0f s, %.2f cycles/trial, first-cycle a histogram", arm.c_str(), trials, secs,
         double(cyc_sum) / trials);
  for (int a = 0; a <= 7; ++a) printf(" %ld", a0_hist[a]);
  printf("\n");
  printf("t1: chi2 %.2f (%d dof) p=%.4f outside %ld\n", c1.stat, c1.dof, c1.p, c1.outside);
  printf("t1,t2: chi2 %.2f (%d dof) p=%.4f outside %ld\n", c2.stat, c2.dof, c2.p, c2.outside);
  if (L >= 3) {
    // third token only over the pairs whose p3 was computed (renormalized to their mass)
    std::map<std::vector<int>, double> e3;
    std::map<std::vector<int>, long> o3;
    std::map<std::vector<int>, bool> covered;
    double mass = 0;
    for (auto& [k, v] : ex3) {
      mass += v;
      covered[{k[0], k[1]}] = true;
    }
    // mass of the covered pairs (ex3 sums to it exactly up to the third token's normalization)
    double pmass = 0;
    for (auto& [k, v] : covered) pmass += ex2[k];
    long n3 = 0;
    for (auto& [k, v] : ob3)
      if (covered.count({k[0], k[1]})) {
        o3[k] += v;
        n3 += v;
      }
    for (auto& [k, v] : ex3) e3[k] = v / pmass;
    Chi c3 = chi2(o3, e3, n3);
    printf("t1,t2,t3 (pairs with p3, %.4f of the mass, %ld samples): chi2 %.2f (%d dof) p=%.4f outside %ld\n", mass, n3,
           c3.stat, c3.dof, c3.p, c3.outside);
  }
  std::vector<std::pair<std::vector<int>, long>> top(ob2.begin(), ob2.end());
  std::sort(top.begin(), top.end(), [](auto& a, auto& b) { return a.second > b.second; });
  for (size_t i = 0; i < std::min<size_t>(10, top.size()); ++i)
    printf("  (%d, %d): observed %ld, expected %.1f\n", top[i].first[0], top[i].first[1], top[i].second,
           ex2.count(top[i].first) ? ex2[top[i].first] * trials : 0.0);
  return 0;
}
