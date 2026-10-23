// A model split across Macs by layers (src/cluster/pipe.h) against the same model whole: the whole model in this
// process, and the split one between this process (the head) and `kapitsa stage` processes, on this Mac or others. Both
// run the same passes in lockstep (prefill chunks, one with a checkpoint inside it, decode steps, the MTP head's
// drafts, verify passes with pending rows, a flush, a restore, a reset), and every pass's logits, sampled tokens and
// residual rows are compared bit for bit.
//   split_check DIR --split HOST:PORT[,HOST:PORT...] [--seed N] [--steps N] [--repeat N]
// For a 6-layer synthetic pack (tools/fn_synth.py), with the stage on this Mac (it takes the layers the placement gives
// it, or with --layers 4-6 those):
//   kapitsa stage --model DIR --port 9331 &
//   split_check DIR --split 127.0.0.1:9331
// --repeat N: then N decode steps, 3-row verifies and 2048-row prefill chunks on each, interleaved, timed (the hops'
// cost on this link).
#include "cluster/pipe.h"
#include "engine/doctor.h"
#include "engine/engine.h"
#include "metal/gpu.h"
#include "model/qwen4_exp.h"
#include "util/log.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace ni;

namespace {

int g_fail = 0, g_checks = 0;

// a and b, n bytes: equal, or the first differing float and how many differ
void same(const char* what, const void* a, const void* b, size_t n) {
  ++g_checks;
  if (!memcmp(a, b, n)) return;
  ++g_fail;
  const float* fa = static_cast<const float*>(a);
  const float* fb = static_cast<const float*>(b);
  size_t first = 0, diff = 0;
  for (size_t i = 0; i < n / 4; ++i)
    if (memcmp(fa + i, fb + i, 4)) {
      if (!diff) first = i;
      ++diff;
    }
  fprintf(stderr, "FAIL %s: %zu of %zu values differ, the first at %zu: %.9g whole vs %.9g split\n", what, diff, n / 4,
          first, double(fa[first]), double(fb[first]));
}

void same_int(const char* what, int a, int b) {
  ++g_checks;
  if (a == b) return;
  ++g_fail;
  fprintf(stderr, "FAIL %s: %d whole vs %d split\n", what, a, b);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argv[1][0] == '-') {
    fprintf(stderr, "split_check DIR --split HOST:PORT[,...] [--seed N] [--steps N] [--repeat N]\n");
    return 2;
  }
  const std::string dir = argv[1];
  std::vector<std::string> stages;
  int seed = 1, steps = 12, repeat = 0;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) fatal("%s needs a value", a.c_str());
      return argv[++i];
    };
    if (a == "--split") {
      const std::string v = next();
      for (size_t at = 0; at <= v.size();) {
        const size_t c = std::min(v.find(',', at), v.size());
        if (c > at) stages.push_back(v.substr(at, c - at));
        at = c + 1;
      }
    } else if (a == "--seed") seed = std::stoi(next());
    else if (a == "--steps") steps = std::stoi(next());
    else if (a == "--repeat") repeat = std::stoi(next());
    else fatal("unknown option %s", a.c_str());
  }
  if (stages.empty()) fatal("--split: the stages' host:port");
  std::string why;
  cluster::HeadInfo me;
  me.name = "split_check";
  me.mem = double(default_mem_budget()) / 2;  // (the whole model shares this process)
  me.gbs = bandwidth_gbs(Gpu::get().name(), Gpu::get().core_count());
  auto link = cluster::Link::connect(stages, dir, me, &why);
  if (!link) fatal("--split: %s", why.c_str());
  for (const cluster::StageInfo& s : link->stages())
    fprintf(stderr, "stage %s (%s): layers %d-%d, up to %d rows a pass\n", s.name.c_str(), s.addr.c_str(), s.lo + 1, s.hi,
            s.rows);
  fprintf(stderr, "ring round trip %.2f ms\n", link->rtt_ms());

  const int chunk = 2048;
  Qwen4Exp::Split none, head_split{0, link->first(), link};
  Qwen4Exp whole(dir, chunk, 0, &none);
  Qwen4Exp head(dir, chunk, 0, &head_split);
  if (!whole.has_mtp()) fatal("%s has no MTP head (mtp.safetensors): the check drafts with it", dir.c_str());
  const int V = whole.config().vocab, SD = whole.config4().hc * whole.config().hidden;
  const size_t xrow = size_t(SD) * 4;
  Gpu& g = Gpu::get();
  SamplingParams sp;  // greedy

  // the token sequence (eos left out), the same in both sequences
  const int T1 = 37, T2 = 300, snap_row = 120, limit = 4096;
  std::mt19937 rng{uint32_t(seed)};
  std::vector<int> ids(size_t(limit + 8));
  for (int& t : ids) t = int(rng() % uint32_t(V - 2));
  auto a = whole.new_seq(limit), b = head.new_seq(limit);
  memcpy(a->tokens(), ids.data(), ids.size() * 4);
  memcpy(b->tokens(), ids.data(), ids.size() * 4);
  Checkpoint ca = whole.new_checkpoint(), cb = head.new_checkpoint();

  // runs the same encoding on both models, each in its own command buffer
  auto both = [&](const std::function<void(Qwen4Exp&, Seq&, Checkpoint&, Encoder&)>& f) {
    auto e = g.begin();
    f(whole, *a, ca, *e);
    e->commit_and_wait();
    if (getenv("SPLIT_TRACE")) fprintf(stderr, "  whole done; split next\n");
    auto e2 = g.begin();
    f(head, *b, cb, *e2);
    e2->commit_and_wait();
  };
  const bool trace = getenv("SPLIT_TRACE") != nullptr;
  auto check_pass = [&](const char* what, int rows_l, int rows_x) {
    if (trace) fprintf(stderr, "checked: %s\n", what);
    if (rows_l) same((std::string(what) + " logits").c_str(), whole.logits(), head.logits(), size_t(rows_l) * V * 4);
    if (rows_x) same((std::string(what) + " residual").c_str(), whole.residual().ptr(), head.residual().ptr(), rows_x * xrow);
  };

  // 1. a prefill chunk, sampling its next token
  both([&](Qwen4Exp& m, Seq& s, Checkpoint&, Encoder& e) { m.encode_prefill(e, s, 0, T1, sp, 0, true); });
  check_pass("prefill 1", 1, T1);
  same_int("prefill 1 token", a->tokens()[T1], b->tokens()[T1]);
  // 2. a long chunk (a command buffer per layer) with a checkpoint saved inside it
  both([&](Qwen4Exp& m, Seq& s, Checkpoint& c, Encoder& e) {
    m.encode_prefill(e, s, T1, T2, sp, 1, true, &c, snap_row);
  });
  check_pass("prefill 2 (checkpoint inside)", 1, T2);
  int P = T1 + T2;
  same_int("prefill 2 token", a->tokens()[P], b->tokens()[P]);
  // 3. decode steps, each feeding its own sample
  for (int i = 0; i < steps; ++i, ++P) {
    both([&](Qwen4Exp& m, Seq& s, Checkpoint&, Encoder& e) { m.encode_step(e, s, P, sp, uint32_t(P), true); });
    check_pass("decode step", 1, 1);
    same_int("decode token", a->tokens()[P + 1], b->tokens()[P + 1]);
  }
  // 4. the MTP head's draft from the last step's rows (they came back from the stages on the split model)
  both([&](Qwen4Exp& m, Seq& s, Checkpoint&, Encoder& e) { m.encode_mtp(e, s, P - 1, 1, P, m.hidden_rows(), 0, limit + 4); });
  same_int("MTP draft", a->tokens()[limit + 4], b->tokens()[limit + 4]);
  // 5. verify passes of 3 rows (the next sample and two drafts), keeping 2 rows each: pending rows the next pass folds
  for (int i = 0; i < 3; ++i) {
    const int R = 3, keep = 2;
    a->tokens()[P + 1] = b->tokens()[P + 1] = int(rng() % uint32_t(V - 2));
    a->tokens()[P + 2] = b->tokens()[P + 2] = int(rng() % uint32_t(V - 2));
    both([&](Qwen4Exp& m, Seq& s, Checkpoint&, Encoder& e) { m.encode_verify(e, s, P, R, sp, uint32_t(P)); });
    check_pass("verify", R, R);
    for (int r = 0; r < R; ++r) same_int("verify sample", whole.samples()[r], head.samples()[r]);
    whole.commit_rows(*a, keep);
    head.commit_rows(*b, keep);
    P += keep;
    a->tokens()[P] = b->tokens()[P] = whole.samples()[keep - 1];
  }
  // 6. the pending rows folded, then a plain step
  both([&](Qwen4Exp& m, Seq& s, Checkpoint&, Encoder& e) { m.encode_flush(e, s); });
  both([&](Qwen4Exp& m, Seq& s, Checkpoint&, Encoder& e) { m.encode_step(e, s, P, sp, uint32_t(P), true); });
  check_pass("step after the flush", 1, 1);
  same_int("token after the flush", a->tokens()[P + 1], b->tokens()[P + 1]);
  // 7. back to the checkpoint inside chunk 2, then the rest of that chunk's tokens again
  const int at = T1 + snap_row;
  both([&](Qwen4Exp& m, Seq& s, Checkpoint& c, Encoder& e) {
    m.restore(e, s, c);
    m.encode_prefill(e, s, at, 64, sp, 2, true);
  });
  check_pass("prefill from the restored checkpoint", 1, 64);
  same_int("token from the restored checkpoint", a->tokens()[at + 64], b->tokens()[at + 64]);
  // 8. a reset, and the first chunk again: the same as before on both
  whole.reset(*a);
  head.reset(*b);
  memcpy(a->tokens(), ids.data(), ids.size() * 4);
  memcpy(b->tokens(), ids.data(), ids.size() * 4);
  both([&](Qwen4Exp& m, Seq& s, Checkpoint&, Encoder& e) { m.encode_prefill(e, s, 0, T1, sp, 0, true); });
  check_pass("prefill after a reset", 1, T1);
  same_int("token after a reset", a->tokens()[T1], b->tokens()[T1]);
  // 9. a conversation whose caches grow chunk by chunk, as the engine grows them (reserve before each pass); the stages
  // round their growth by their own budgets, so what they hold is tracked apart from the head's cap
  {
    auto a2 = whole.new_seq(limit, 0), b2 = head.new_seq(limit, 0);
    memcpy(a2->tokens(), ids.data(), ids.size() * 4);
    memcpy(b2->tokens(), ids.data(), ids.size() * 4);
    for (int p = 0, T = 0; p + 1 < 3200; p += T) {
      T = std::min(700, 3200 - 1 - p);
      const bool ok_a = whole.reserve(*a2, p + T + 1), ok_b = head.reserve(*b2, p + T + 1);
      same_int("growing caches: reserve", ok_a, ok_b);
      if (!ok_a || !ok_b) break;
      if (b2->rcap < p + T + 1) same_int("growing caches: the stages hold the positions", b2->rcap, p + T + 1);
      auto e = g.begin();
      whole.encode_prefill(*e, *a2, p, T, sp, 3, true);
      e->commit_and_wait();
      auto e2 = g.begin();
      head.encode_prefill(*e2, *b2, p, T, sp, 3, true);
      e2->commit_and_wait();
      check_pass("growing caches: prefill", 1, T);
      same_int("growing caches: token", a2->tokens()[p + T], b2->tokens()[p + T]);
    }
  }
  if (!head.lost().empty()) fatal("the split broke: %s", head.lost().c_str());
  printf("%s: %d checks, %d failed (split: layers 1-%d here, the rest on %zu stage%s)\n", g_fail ? "FAIL" : "PASS",
         g_checks, g_fail, link->first(), link->stages().size(), link->stages().size() > 1 ? "s" : "");

  // timing: whole and split interleaved (A B A B), one command buffer each, the median of each kind: a decode step, a
  // verify of 3 rows (the MTP head's two drafts), a prefill chunk of `chunk` rows. The split's extra time is what the
  // hops cost on this link.
  if (repeat > 0) {
    struct Kind {
      const char* name;
      int rows;
    };
    for (const Kind k : {Kind{"decode step", 1}, Kind{"verify, 3 rows", 3}, Kind{"prefill chunk", chunk}}) {
      std::vector<double> t[2];
      const int P0 = T1 + T2 + 64;
      for (int i = 0; i < repeat; ++i)
        for (int w = 0; w < 2; ++w) {
          Qwen4Exp& m = w ? head : whole;
          Seq& s = w ? *b : *a;
          const double t0 = now_ms();
          auto e = g.begin();
          if (k.rows == 1) m.encode_step(*e, s, P0, sp, uint32_t(P0), true);
          else if (k.rows == 3) m.encode_verify(*e, s, P0, 3, sp, uint32_t(P0));
          else m.encode_prefill(*e, s, P0, k.rows, sp, 0, false);
          e->commit_and_wait();
          if (k.rows == 3) m.commit_rows(s, 0);
          t[w].push_back(now_ms() - t0);
        }
      for (auto& v : t) std::sort(v.begin(), v.end());
      const double mw = t[0][t[0].size() / 2], ms = t[1][t[1].size() / 2];
      printf("%-15s %4d rows: whole %8.2f ms, split %8.2f ms (+%.2f ms; %.1f KB a row each way)\n", k.name, k.rows, mw,
             ms, ms - mw, double(head.hidden_row_bytes() + size_t(head.config().hidden) * 4 + 16) / 1e3);
    }
  }
  return g_fail ? 1 : 0;
}
