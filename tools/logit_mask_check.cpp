// Structured output's GPU side (logit_mask_rows in src/kernels/sample.metal, enc_logit_mask in src/model/logit_mask.cpp),
// at small sizes first (run it with MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 after a kernel change):
//  - masked logits rows against the CPU: every left-out token -inf, every other logit and the row padding untouched,
//    for vocabularies around word boundaries, several rows, a row stride past the vocabulary and an offset mask row;
//  - the shared-event wait: a pass whose masks the host has not signalled does not finish until it does;
//  - the sampler after a mask (topk_stage1 + sample_stage2, Qwen's vocabulary size): greedy gives the best allowed
//    token, sampling (T 0.7, top-k 20, top-p 0.95) draws allowed tokens only, a single allowed token always.
//   logit_mask_check      exit status 1 if any check fails
#include "metal/gpu.h"
#include "model/model.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace ni;

namespace {

int fails = 0, passes = 0;
void expect(const std::string& name, bool ok, const std::string& detail = "") {
  printf("%s %s%s\n", ok ? "PASS" : "FAIL", name.c_str(), ok || detail.empty() ? "" : ("  " + detail).c_str());
  ok ? ++passes : ++fails;
}

struct TopkP {
  uint32_t V, K, chunk;
};
struct SampleP {
  uint32_t n, K;
  float temp, top_p;
  uint32_t seed, step, out_index;
};

}  // namespace

int main() {
  Gpu& g = Gpu::get();
  std::mt19937 rng(42);
  // ---- 1. masked rows against the CPU ----
  for (const int V : {1, 31, 32, 33, 100, 257}) {
    for (const int R : {1, 3, 8}) {
      const int ld = V + (V % 3 == 0 ? 5 : 0), W = (V + 31) / 32, mrow0 = R == 3 ? 2 : 0;
      Buffer lg = g.alloc(size_t(R + 1) * ld * 4, "lg");
      Seq s;
      s.mask_words = W;
      s.mask_rows = R + mrow0;
      s.mask = g.alloc(size_t(R + mrow0) * W * 4, "mask");
      std::vector<float> ref(size_t(R + 1) * ld);
      for (auto& x : ref) x = std::uniform_real_distribution<float>(-5, 5)(rng);
      memcpy(lg.ptr(), ref.data(), ref.size() * 4);
      uint32_t* mw = s.mask.as<uint32_t>();
      for (int i = 0; i < (R + mrow0) * W; ++i) {
        const int k = int(rng() % 4);
        mw[i] = k == 0 ? 0xffffffffu : k == 1 ? 0u : uint32_t(rng());
      }
      auto e = g.begin();
      enc_logit_mask(*e, s, lg, R, V, ld, mrow0);
      e->commit_and_wait();
      const float* got = lg.as<float>();
      bool ok = true;
      for (int r = 0; r <= R && ok; ++r)
        for (int t = 0; t < ld && ok; ++t) {
          const float before = ref[size_t(r) * ld + t];
          const bool masked = r < R && t < V && !(mw[size_t(r + mrow0) * W + (t >> 5)] >> (t & 31) & 1);
          const float want = masked ? -INFINITY : before;
          ok = got[size_t(r) * ld + t] == want;
          if (!ok) printf("   row %d token %d: %g, want %g\n", r, t, got[size_t(r) * ld + t], want);
        }
      expect("masked rows: vocabulary " + std::to_string(V) + ", " + std::to_string(R) + " rows, stride " +
                std::to_string(ld) + ", mask rows from " + std::to_string(mrow0), ok);
      g.release(lg);
      g.release(s.mask);
    }
  }
  // no masks: nothing encoded
  {
    Seq s;
    Buffer lg = g.alloc(64 * 4, "lg");
    for (int i = 0; i < 64; ++i) lg.as<float>()[i] = float(i);
    auto e = g.begin();
    enc_logit_mask(*e, s, lg, 1, 64, 64);
    e->commit_and_wait();
    bool ok = true;
    for (int i = 0; i < 64; ++i) ok = ok && lg.as<float>()[i] == float(i);
    expect("a sequence without masks: logits untouched", ok);
    g.release(lg);
  }
  // ---- 2. the event: the pass waits for the host ----
  {
    const int V = 64, W = 2;
    Seq s;
    s.mask_words = W;
    s.mask_rows = 1;
    s.mask = g.alloc(W * 4, "mask");
    Buffer lg = g.alloc(V * 4, "lg");
    for (int i = 0; i < V; ++i) lg.as<float>()[i] = 1.0f;
    s.mask_event = g.new_shared_event();
    s.mask_value = 3;
    auto e = g.begin();
    enc_logit_mask(*e, s, lg, 1, V, V);
    e->commit();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    const bool waited = !e->wait_for(0);
    s.mask.as<uint32_t>()[0] = 0x0000ffffu;  // written after the commit, before the signal: the pass reads this
    s.mask.as<uint32_t>()[1] = 0xffffffffu;
    Gpu::signal_event_cpu(s.mask_event, 3);
    e->wait();
    bool ok = true;
    for (int i = 0; i < V; ++i) ok = ok && lg.as<float>()[i] == (i >= 16 && i < 32 ? -INFINITY : 1.0f);
    expect("the event: the pass waits for the host's signal", waited);
    expect("the event: the masks written after the commit are the ones applied", ok);
    g.release(lg);
    g.release(s.mask);
  }
  // ---- 3. the sampler after a mask, at Qwen's vocabulary size ----
  {
    const uint32_t V = 248320, W = V / 32, chunk = 2048, ntg = (V + chunk - 1) / chunk;
    Buffer lg = g.alloc(size_t(V) * 4, "lg"), cv = g.alloc(size_t(ntg) * 64 * 4, "cv"), ci = g.alloc(size_t(ntg) * 64 * 4, "ci");
    Buffer tok = g.alloc(64 * 4, "tok"), prob = g.alloc(16, "prob");
    Seq s;
    s.mask_words = int(W);
    s.mask_rows = 1;
    s.mask = g.alloc(size_t(W) * 4, "mask");
    static const Pipeline st1 = g.pipeline("topk_stage1"), st2 = g.pipeline("sample_stage2");
    std::vector<float> lv(V);
    auto run = [&](float temp, uint32_t step) {
      memcpy(lg.ptr(), lv.data(), size_t(V) * 4);
      const uint32_t K = temp > 0 ? 20 : 1;
      auto e = g.begin();
      enc_logit_mask(*e, s, lg, 1, int(V), int(V));
      const TopkP ts{V, K, chunk};
      e->use(st1);
      e->buf(0, lg), e->buf(1, cv), e->buf(2, ci);
      e->val(3, ts);
      e->dispatch(ntg, 1, 1, 256);
      const SampleP sp{ntg * K, K, temp, 0.95f, 1234, step, 0};
      e->use(st2);
      e->buf(0, cv), e->buf(1, ci), e->buf(2, tok), e->buf(3, prob);
      e->val(4, sp);
      e->dispatch(1, 1, 1, 1024);
      e->commit_and_wait();
      return tok.as<int>()[0];
    };
    for (int trial = 0; trial < 6; ++trial) {
      for (auto& x : lv) x = std::normal_distribution<float>(0, 3)(rng);
      uint32_t* mw = s.mask.as<uint32_t>();
      // a sparse mask: ~1 token in 500 allowed (the case of a grammar position); trial 5: one token alone
      for (uint32_t w = 0; w < W; ++w) mw[w] = 0;
      std::vector<int> allowed;
      for (uint32_t t = 0; t < V; ++t)
        if (trial == 5 ? t == 123457 : rng() % 500 == 0) mw[t >> 5] |= 1u << (t & 31), allowed.push_back(int(t));
      int best = allowed[0];
      for (int t : allowed)
        if (lv[size_t(t)] > lv[size_t(best)]) best = t;
      const int got = run(0, 0);
      expect("greedy after a mask (" + std::to_string(allowed.size()) + " allowed): the best allowed token", got == best,
            std::to_string(got) + " vs " + std::to_string(best));
      bool all_allowed = true;
      for (uint32_t step = 0; step < 200 && all_allowed; ++step) {
        const int t = run(0.7f, step);
        all_allowed = t >= 0 && t < int(V) && (mw[uint32_t(t) >> 5] >> (t & 31) & 1);
      }
      expect("sampling after a mask (" + std::to_string(allowed.size()) + " allowed): 200 draws, all allowed", all_allowed);
    }
  }
  printf("%d passed, %d failed\n", passes, fails);
  return fails ? 1 : 0;
}
