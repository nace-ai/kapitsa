// A request's top-k on the GPU samplers (kapitsa#1286; src/model/sampler.h). Synthetic logits, no model.
//   sample_topk_test --cpu                 which top_k the two-pass sampler takes and which go to the whole vocabulary
//   sample_topk_test [--draws N (4000)] [--two-pass]
// The GPU cases sample one logits row (Qwen3.5's 248,320 tokens, logit -0.02 * rank over a random permutation of the
// ids) N times, each draw with its own RNG step, by the route a model's plain step takes: sample_full when
// full_vocab(sp), else topk_stage1 + sample_stage2 with topk_k(sp) (stage 1's chunk: topk_chunk(K)); a two-pass case
// also draws with topk_stage1_rows_sg as stage 1 (Qwen's step on the M5, K >= 8) and must give the same tokens. Each case checks the drawn ids against the plain
// sampler's law computed on the CPU in double (the K best by value desc, id asc; the top-p prefix of the untempered
// probabilities; exp((l - max) / T) over it): no draw outside the kept set, the number of distinct ids, and a
// chi-square test (cells with expected count < 5 pooled). top_k 0 must reach ranks past 32, top_k 40 and 64 (the two
// passes since the cap went to 64) their 40 and 64, and top_k 100 its 100.
// --two-pass routes every case through the two-pass sampler with the clamp main had before #1286
// (clamp(top_k <= 0 ? 32 : top_k, 1, 32)): top_k 0 and 100 then fail, which is the bug.
// Takes the GPU lock (NI_LOCK_HELD=1 when the caller holds it).
#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/kparams.h"
#include "model/sampler.h"
#include "api/anthropic.h"
#include "api/ollama.h"
#include "api/responses.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ni;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  printf("  %s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++failures;
}

// ---- chi-square p-value (regularized upper incomplete gamma, as dflash_accept_test) ----
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

// The plain sampler's law over ids (0 outside the kept set), in double.
std::vector<double> law(const std::vector<float>& l, int top_k, float top_p, float temp, int* kept) {
  const int V = int(l.size());
  std::vector<int> ord(static_cast<size_t>(V));
  std::iota(ord.begin(), ord.end(), 0);
  auto val = [&](int i) { return std::isnan(l[size_t(i)]) ? -INFINITY : l[size_t(i)]; };
  std::sort(ord.begin(), ord.end(), [&](int a, int b) { return val(a) > val(b) || (val(a) == val(b) && a < b); });
  int K = top_k > 0 && top_k < V ? top_k : V;
  const double m = val(ord[0]);
  double tot = 0;
  for (int q = 0; q < K; ++q) tot += std::exp(double(val(ord[size_t(q)])) - m);
  int keep = K;
  if (top_p < 1.0f) {
    double cum = 0;
    for (int q = 0; q < K; ++q) {
      cum += std::exp(double(val(ord[size_t(q)])) - m) / tot;
      if (cum >= top_p) {
        keep = q + 1;
        break;
      }
    }
  }
  std::vector<double> p(size_t(V), 0.0);
  double kt = 0;
  for (int q = 0; q < keep; ++q) kt += p[size_t(ord[size_t(q)])] = std::exp((double(val(ord[size_t(q)])) - m) / temp);
  for (double& x : p) x /= kt;
  *kept = 0;
  for (double x : p) *kept += x > 0;
  return p;
}

uint32_t pcg_hash(uint32_t v) {  // sample.metal's
  const uint32_t state = v * 747796405u + 2891336453u;
  const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

struct Case {
  const char* name;
  int top_k;
  float top_p, temp;
  int ties;         // > 0: the ranks [top_k - ties, top_k + ties) share one logit (a cut through a tie)
  bool masked;      // every 7th id -inf, every 11th NaN
  int min_distinct; // distinct ids the draws must reach (0: no check)
  float slope = 0.02f;  // logit of rank r: -slope * r
};

struct Gpu1 {
  Gpu& g = Gpu::get();
  Pipeline st1 = g.pipeline("topk_stage1"), st1sg = g.pipeline("topk_stage1_rows_sg"), st2 = g.pipeline("sample_stage2");
  Buffer logits, cand_v, cand_i, tokens, sprob;
  Gpu1(uint32_t V, int draws) {
    logits = g.alloc(size_t(V) * 4);
    cand_v = g.alloc(size_t((V + 2047) / 2048) * 64 * 4);
    cand_i = g.alloc(size_t((V + 2047) / 2048) * 64 * 4);
    tokens = g.alloc(size_t(draws) * 4);
    sprob = g.alloc(64 * 4);
  }
  // the route of a model's plain step (Qwen35 / Flint / Dory / NemotronH::enc_sample)
  void draw(const SamplingParams& sp, uint32_t V, int n, bool two_pass, bool force_full = false, bool sg = false) {
    const int chunk = 512;  // (command buffers of 512 draws)
    for (int t0 = 0; t0 < n; t0 += chunk) {
      auto e = g.begin();
      for (int t = t0; t < std::min(n, t0 + chunk); ++t) {
        if (force_full || (!two_pass && full_vocab(sp))) {
          enc_sample_full(*e, logits, V, tokens, t, sprob, sp, uint32_t(t));
          continue;
        }
        const uint32_t K = two_pass ? (sp.temperature > 0 ? uint32_t(std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32)) : 1u)
                                    : topk_k(sp);
        const kp::TopkStage ts{V, K, topk_chunk(K)};
        const uint32_t ntg = topk_lists(V, K);
        e->use(sg ? st1sg : st1);
        e->buf(0, logits); e->buf(1, cand_v); e->buf(2, cand_i);
        e->val(3, ts);
        e->dispatch(ntg, 1, 1, topk_threads(K));
        const kp::Sample p2{ntg * K, K, sp.temperature, sp.top_p, sp.seed, uint32_t(t), uint32_t(t)};
        e->use(st2);
        e->buf(0, cand_v); e->buf(1, cand_i); e->buf(2, tokens); e->buf(3, sprob);
        e->val(4, p2);
        e->dispatch(1, 1, 1, 1024);
      }
      e->commit_and_wait();
    }
  }
};

void run_case(Gpu1& G, const Case& c, uint32_t V, int draws, bool two_pass) {
  printf("[%s] top_k %d top_p %.2f T %.2f%s%s, %d draws\n", c.name, c.top_k, c.top_p, c.temp,
         c.ties ? ", tie at the cut" : "", c.masked ? ", -inf / NaN ids" : "", draws);
  std::mt19937 rng(1286u + uint32_t(c.top_k) * 7u + uint32_t(c.ties));
  std::vector<int> perm(V);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  std::vector<float> l(V);
  for (uint32_t r = 0; r < V; ++r) {
    int rr = int(r);
    if (c.ties && rr >= c.top_k - c.ties && rr < c.top_k + c.ties) rr = c.top_k - c.ties;
    l[size_t(perm[r])] = -c.slope * float(rr);
  }
  if (c.masked)
    for (uint32_t i = 0; i < V; ++i) {
      if (i % 7 == 0) l[i] = -INFINITY;
      else if (i % 11 == 0) l[i] = std::numeric_limits<float>::quiet_NaN();
    }
  memcpy(G.logits.as<float>(), l.data(), size_t(V) * 4);
  int kept = 0;
  const std::vector<double> p = law(l, c.top_k, c.top_p, c.temp, &kept);
  SamplingParams sp;
  sp.temperature = c.temp;
  sp.top_p = c.top_p;
  sp.top_k = c.top_k;
  sp.seed = 0x5eed1286u;
  printf("  route: %s; kept set %d ids\n",
         two_pass ? "two-pass (main's clamp)" : full_vocab(sp) ? "sample_full" : "two-pass", kept);
  G.draw(sp, V, draws, two_pass);
  const int* tk = G.tokens.as<int>();
  std::vector<long> obs(V, 0);
  long outside = 0, bad = 0;
  for (int t = 0; t < draws; ++t) {
    if (tk[t] < 0 || uint32_t(tk[t]) >= V) {
      ++bad;
      continue;
    }
    if (p[size_t(tk[t])] <= 0) ++outside;
    ++obs[size_t(tk[t])];
  }
  long distinct = 0, past32 = 0;
  for (uint32_t i = 0; i < V; ++i) distinct += obs[i] > 0;
  // ranks of the draws (rank of an id in the sorted order)
  {
    std::vector<int> rank(V);
    for (uint32_t r = 0; r < V; ++r) rank[size_t(perm[r])] = int(r);
    for (uint32_t i = 0; i < V; ++i)
      if (rank[i] >= 32) past32 += obs[i];
  }
  double stat = 0, pool_o = 0, pool_e = 0;
  int cells = 0;
  for (uint32_t i = 0; i < V; ++i) {
    const double e = p[i] * draws;
    if (e <= 0) continue;
    if (e < 5) {
      pool_o += double(obs[i]), pool_e += e;
      continue;
    }
    stat += (double(obs[i]) - e) * (double(obs[i]) - e) / e;
    ++cells;
  }
  if (pool_e > 0) stat += (pool_o - pool_e) * (pool_o - pool_e) / pool_e, ++cells;
  const double pv = cells > 1 ? gamma_q(0.5 * (cells - 1), 0.5 * stat) : 1.0;
  printf("  distinct ids %ld, draws ranked past 32: %ld (%.1f%%), chi-square %.1f over %d cells, p = %.4f\n", distinct,
         past32, 100.0 * double(past32) / draws, stat, cells, pv);
  check(bad == 0, "every draw is a token id");
  check(outside == 0, "no draw outside the kept set (" + std::to_string(outside) + " outside)");
  if (c.min_distinct && draws >= 4000)  // (fewer draws: the law's checks alone)
    check(distinct >= c.min_distinct,
          "draws reach at least " + std::to_string(c.min_distinct) + " distinct ids (" + std::to_string(distinct) + ")");
  check(pv > 1e-4, "the draws follow the sampler's law (chi-square p > 1e-4)");
  if (two_pass) return;
  if (!full_vocab(sp)) {
    const std::vector<int> st2(tk, tk + draws);
    if (sp.top_k >= 8) {  // stage 1 on simdgroup reductions (Qwen35::enc_sample on the M5): the same lists, the same tokens
      G.draw(sp, V, draws, false, false, true);
      long dsg = 0;
      for (int t = 0; t < draws; ++t) dsg += G.tokens.as<int>()[t] != st2[size_t(t)];
      check(dsg == 0, "topk_stage1_rows_sg draws topk_stage1's token at every step (" + std::to_string(dsg) + " differ)");
    }
    // sample_full on the same request: its list path runs sample_stage2's rules over the same top-k in the same order
    G.draw(sp, V, draws, false, true);
    long diff = 0;
    for (int t = 0; t < draws; ++t) diff += G.tokens.as<int>()[t] != st2[size_t(t)];
    check(diff == 0, "sample_full draws sample_stage2's token at every step (" + std::to_string(diff) + " differ)");
    return;
  }
  // sample_full's draw itself, in double: the uniform of its step walked over the kept law. The list path (top-k of at
  // most 1024, or a top-p prefix of at most 1024 tokens) walks the kept tokens in value order (desc, id asc) as
  // sample_stage2 does; the bisection path in id order. A draw within 1e-5 of a boundary may take either side (the GPU
  // sums in float).
  const int V_ = int(V);
  const int Kc = c.top_k > 0 && c.top_k < V_ ? c.top_k : V_;
  const bool list = Kc <= 1024 || (Kc == V_ && c.top_p < 1.0f && kept <= 1024);
  std::vector<uint32_t> walk;
  for (uint32_t i = 0; i < V; ++i)
    if (p[i] > 0) walk.push_back(i);
  if (list)
    std::sort(walk.begin(), walk.end(), [&](uint32_t a, uint32_t b) { return p[a] > p[b] || (p[a] == p[b] && a < b); });
  std::vector<double> cum(walk.size());
  double a = 0;
  for (size_t q = 0; q < walk.size(); ++q) cum[q] = a += p[walk[q]];
  std::vector<int> pos(V, -1);
  for (size_t q = 0; q < walk.size(); ++q) pos[walk[q]] = int(q);
  long wrong = 0;
  for (int t = 0; t < draws; ++t) {
    const double u = double(pcg_hash(sp.seed ^ pcg_hash(uint32_t(t))) & 0xffffff) / 16777216.0;
    const int want = int(std::upper_bound(cum.begin(), cum.end(), u) - cum.begin());
    const int got = pos[size_t(tk[t])];
    if (got == want || got < 0) continue;  // (got < 0: counted as outside above)
    const int lo = std::min(got, want);
    if (std::fabs(cum[size_t(lo)] - u) < 1e-5) continue;
    if (++wrong <= 3) printf("  draw %d: u %.7f got walk index %d want %d\n", t, u, got, want);
  }
  printf("  draw order: %s\n", list ? "value (list path)" : "id (bisection path)");
  check(wrong == 0, "every draw is the id its uniform selects in the kept law (" + std::to_string(wrong) + " not)");
}

int cpu_checks() {
  printf("[cpu] routing (sampler.h, kTopkMax %d)\n", kTopkMax);
  auto sp = [](float t, int k) {
    SamplingParams s;
    s.temperature = t;
    s.top_k = k;
    return s;
  };
  check(full_vocab(sp(1.0f, 0)), "top_k 0 at T > 0: the whole vocabulary");
  check(full_vocab(sp(0.6f, -1)), "top_k -1 at T > 0: the whole vocabulary");
  check(full_vocab(sp(1.0f, 65)) && full_vocab(sp(1.0f, 100)), "top_k 65 and 100: the whole vocabulary");
  check(!full_vocab(sp(1.0f, 1)) && !full_vocab(sp(1.0f, 20)) && !full_vocab(sp(1.0f, 32)) && !full_vocab(sp(1.0f, 40)) &&
            !full_vocab(sp(1.0f, 64)),
        "top_k 1, 20, 32, 40, 64: the two-pass sampler");
  check(!full_vocab(sp(0.0f, 0)) && !full_vocab(sp(0.0f, 100)), "greedy (T = 0), any top_k: the two-pass sampler");
  check(topk_k(sp(1.0f, 20)) == 20 && topk_k(sp(1.0f, 40)) == 40 && topk_k(sp(1.0f, 64)) == 64 &&
            topk_k(sp(1.0f, 1)) == 1,
        "topk_k: top_k 1..64 unchanged");
  check(topk_chunk(32) == 2048 && topk_threads(32) == 256 && topk_lists(248320, 32) == 122,
        "top_k up to 32: stage 1 as before (2048-logit chunks, 256 threads, 122 lists for 248,320 ids)");
  check(topk_chunk(33) == 4096 && topk_threads(64) == 512 && topk_lists(248320, 64) * 64 <= 4096 &&
            topk_lists(151936, 64) * 64 <= 4096 && topk_lists(131072, 64) * 64 <= 4096,
        "top_k 33..64: 4096-logit chunks, at most 4096 candidates for stage 2 (Qwen, Flint and Nemotron vocabularies)");
  check(topk_k(sp(0.0f, 0)) == 1 && topk_k(sp(0.0f, 100)) == 1, "topk_k: greedy keeps 1");

  // A request that leaves top_k out keeps the two-pass sampler (and the speculative loops): kapitsa#1286 changes only
  // what an explicit top_k <= 0 or above 64 does.
  printf("[cpu] omitted top_k\n");
  check(omitted_top_k(20) == 20 && omitted_top_k(32) == 32 && omitted_top_k(1) == 1,
        "generation_config top_k 1..32: the publisher's");
  check(omitted_top_k(0) == 32 && omitted_top_k(-1) == 32 && omitted_top_k(100) == 32 && omitted_top_k(40) == 40,
        "generation_config without top_k, or top_k <= 0 / above 64: 32, as before #1286; 40: its 40");
  auto sampled = [](const ChatRequest& r) {
    SamplingParams s;
    s.temperature = r.temperature > 0 ? r.temperature : 0;
    s.top_k = r.top_k;
    return s;
  };
  const ChatRequest qwen;  // Qwen3.5/3.6/3.8: the engine's defaults are ChatRequest's
  check(qwen.top_k == 20 && !full_vocab(sampled(qwen)), "Qwen defaults: top_k 20, two-pass");
  ChatRequest pub;  // Dory / Nemotron: the publisher's temperature 1.0, no top_k
  pub.temperature = 1.0f;
  pub.top_k = omitted_top_k(0);
  const ojson msgs = ojson::parse(R"([{"role": "user", "content": "hi"}])");
  for (const ChatRequest* d : {static_cast<const ChatRequest*>(&qwen), static_cast<const ChatRequest*>(&pub)}) {
    const std::string who = d == &qwen ? "Qwen defaults" : "publisher defaults without top_k";
    ojson b{{"model", "m"}, {"messages", msgs}, {"options", {{"temperature", 0.7}}}};
    const ChatRequest oc = api::ollama_chat_request(b, *d);
    check(oc.top_k == d->top_k && !full_vocab(sampled(oc)), "Ollama /api/chat, top_k omitted (" + who + "): " +
                                                                 std::to_string(oc.top_k) + ", two-pass");
    ojson g{{"model", "m"}, {"prompt", "hi"}, {"options", {{"temperature", 0.7}}}};
    const ChatRequest og = api::ollama_generate_request(g, *d);
    check(og.top_k == d->top_k && !full_vocab(sampled(og)), "Ollama /api/generate, top_k omitted (" + who + "): " +
                                                                 std::to_string(og.top_k) + ", two-pass");
    b["options"]["top_k"] = 0;
    check(full_vocab(sampled(api::ollama_chat_request(b, *d))), "Ollama, explicit top_k 0: the whole vocabulary");
  }
  {
    ojson b{{"model", "m"}, {"max_tokens", 64}, {"temperature", 0.7}, {"messages", msgs}};
    const ChatRequest a = api::messages_request(b).chat;
    check(a.top_k == 20 && !full_vocab(sampled(a)), "Anthropic /v1/messages, top_k omitted: 20, two-pass");
    b["top_k"] = 0;
    check(full_vocab(sampled(api::messages_request(b).chat)), "Anthropic, explicit top_k 0: the whole vocabulary");
    api::ResponseStore store;
    ojson rb{{"model", "m"}, {"input", "hi"}, {"temperature", 0.7}};
    const ChatRequest o = api::responses_request(rb, store).chat;
    check(o.top_k == 20 && !full_vocab(sampled(o)), "OpenAI /v1/responses, top_k omitted: 20, two-pass");
  }
  // (/v1/chat/completions: parse_request in src/server/main.cpp starts from the engine's defaults, as Ollama's do;
  // tests/test_topk_omitted.py checks it against a running server)
  return failures;
}

}  // namespace

int main(int argc, char** argv) {
  int draws = 4000;
  bool two_pass = false, cpu = false;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--cpu")) cpu = true;
    else if (!strcmp(argv[i], "--two-pass")) two_pass = true;
    else if (!strcmp(argv[i], "--draws") && i + 1 < argc) draws = atoi(argv[++i]);
    else fatal("usage: sample_topk_test [--cpu] [--draws N] [--two-pass]");
  }
  cpu_checks();
  if (cpu) {
    printf("sample_topk_test --cpu: %d failures\n", failures);
    return failures ? 1 : 0;
  }
  bench::GpuLock lock;
  if (!bench::GpuLock::held_by_caller() && !lock.acquire()) fatal("cannot acquire GPU lock");
  const uint32_t V = 248320;  // Qwen3.5's vocabulary
  Gpu1 G(V, draws);
  const Case cases[] = {
      {"top_k 0", 0, 1.0f, 1.0f, 0, false, 200},
      {"top_k 100", 100, 1.0f, 1.0f, 0, false, 100},
      {"top_k 20 (two-pass, unchanged)", 20, 1.0f, 1.0f, 0, false, 20},
      {"top_k 0, top_p 0.9", 0, 0.9f, 1.0f, 0, false, 0},
      {"top_k 100, top_p 0.5, T 0.7", 100, 0.5f, 0.7f, 0, false, 0},
      {"top_k 64, tie at the cut", 64, 1.0f, 1.0f, 8, false, 64},
      {"top_k 40 (two-pass, 4096-logit lists)", 40, 1.0f, 1.0f, 0, false, 40},
      {"top_k 40, top_p 0.95, T 0.7, masked and NaN ids", 40, 0.95f, 0.7f, 0, true, 0},
      {"top_k 65 (sample_full)", 65, 1.0f, 1.0f, 0, false, 65},
      {"top_k 0, masked and NaN ids", 0, 0.95f, 0.6f, 0, true, 0},
      {"top_k 0, top_p 0.95, flat (prefix past the 1024 list)", 0, 0.95f, 1.0f, 0, false, 1000, 0.002f},
      {"top_k 2000, top_p 0.9, flat (bisection)", 2000, 0.9f, 1.0f, 0, false, 800, 0.002f},
      {"top_k 1024, tie at the cut", 1024, 1.0f, 1.0f, 8, false, 0, 0.002f},
      {"top_k 20 at sample_full's list", 20, 0.95f, 0.8f, 0, false, 0},
  };
  for (const Case& c : cases) run_case(G, c, V, draws, two_pass);
  printf("sample_topk_test%s: %d failures\n", two_pass ? " --two-pass" : "", failures);
  return failures ? 1 : 0;
}
