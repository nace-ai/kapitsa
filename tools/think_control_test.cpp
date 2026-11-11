// The thinking budget's row masks (src/engine/think_control.h) on the CPU: chains and KFlash trees (parents), a drafted
// </think> or tool call ending the control on its path, the closing text forced across rows and cycles, rows never
// accepted, a cap of 0, a template that leaves <think> to the model (whitespace before it keeps the control waiting,
// any other token ends it), the first reasoning block only, fill() against push() on random chains and trees, and
// whole replies under each decode loop's gating, the block counted from the emitted tokens.
// The cap counts the forced close: the close starts at cap minus its length (think_close_for), so a cap of 6 with </think>
// alone forces at reasoning position 5.
#include "engine/think_control.h"

#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace ni;

namespace {

int fails = 0, passes = 0;

void check(bool ok, const std::string& name, const std::string& detail = "") {
  printf("%s %s%s\n", ok ? "PASS" : "FAIL", name.c_str(), ok || detail.empty() ? "" : ("\n     " + detail).c_str());
  ok ? ++passes : ++fails;
}

constexpr int kOpen = 10, kClose = 11, kCall = 12, kA = 40, kB = 41, kT1 = 50, kT2 = 51, kNl = 60, kSp = 61;
constexpr size_t W = 3;  // 96 tokens

// whitespace-only tokens: a line break and a space
const std::vector<uint8_t>* spaces() {
  static const std::vector<uint8_t> s = [] {
    std::vector<uint8_t> v(96, 0);
    v[kNl] = v[kSp] = 1;
    return v;
  }();
  return &s;
}

// the one token a row allows (-1: every token, -2: something else)
int allowed(const std::vector<uint32_t>& m, int r) {
  int n = 0, tok = -1;
  for (size_t w = 0; w < W; ++w)
    for (int b = 0; b < 32; ++b)
      if (m[size_t(r) * W + w] >> b & 1u) ++n, tok = int(w * 32 + b);
  return n == int(W * 32) ? -1 : n == 1 ? tok : -2;
}

std::vector<int> rows_of(ThinkControl& tc, const std::vector<int>& tokens, int P, int rows, const int* parent,
                         int live) {
  std::vector<uint32_t> m(size_t(rows) * W, 0x12345678u);
  tc.fill(m.data(), W, tokens.data(), P, rows, parent, live);
  std::vector<int> a;
  for (int r = 0; r < rows; ++r) a.push_back(allowed(m, r));
  return a;
}

std::string str(const std::vector<int>& v) {
  std::string s;
  for (int x : v) s += std::to_string(x) + " ";
  return s;
}

// fill() over a chain or a tree against push(): each row's mask must be what push() leaves after the tokens on the
// row's path (a copy of the control, those tokens pushed, one row filled). Rows that cannot be accepted (an ancestor
// forces a token and the path's next token differs) are not compared.
bool equivalent(ThinkControl& tc, const std::vector<int>& tokens, int rows, const int* parent, int live,
                std::string* why) {
  const auto got = rows_of(tc, tokens, 0, rows, parent, live);
  for (int r = 0; r < live; ++r) {
    std::vector<int> path;
    bool ok = true;
    for (int x = r; x > 0;) {
      const int par = parent ? parent[x] : x - 1;
      if (got[size_t(par)] >= 0 && got[size_t(par)] != tokens[size_t(x)]) ok = false;
      path.insert(path.begin(), tokens[size_t(x)]);
      x = par;
    }
    if (!ok) continue;
    ThinkControl seq = tc;
    for (int t : path) seq.push(t);
    const int want = rows_of(seq, {0}, 0, 1, nullptr, 1)[0];
    if (got[size_t(r)] != want) {
      *why = "row " + std::to_string(r) + ": fill " + std::to_string(got[size_t(r)]) + ", push " +
             std::to_string(want) + "; path " + str(path);
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  {  // a chain of 4 verify rows reaching a cap of 6 with no closing text (</think> at position 5): row r samples
     // reasoning position 3 + r
    ThinkControl tc;
    tc.start(6, {kClose}, kOpen, kClose, -1, true);
    for (int i = 0; i < 3; ++i) tc.push(kA);
    check(tc.count() == 3 && tc.armed() && !tc.near(2) && tc.near(3), "count and near() before the cap");
    const std::vector<int> tk = {0, 0, 0, kA, kB, kB, kClose};  // tokens[P = 3] handled; drafts at 4..6
    const auto a = rows_of(tc, tk, 3, 4, nullptr, 4);
    check(a == std::vector<int>({-1, -1, kClose, -1}), "chain: the row at the cap forces </think>, the row after a "
          "drafted close is free", str(a));
  }
  {  // a drafted </think> before the cap ends the control on that path
    ThinkControl tc;
    tc.start(5, {kClose}, kOpen, kClose, -1, true);
    for (int i = 0; i < 3; ++i) tc.push(kA);
    const std::vector<int> tk = {0, 0, 0, kA, kClose, kB, kB};
    const auto a = rows_of(tc, tk, 3, 4, nullptr, 4);
    check(a == std::vector<int>({-1, -1, -1, -1}), "chain: a drafted </think> ends the control on its path", str(a));
  }
  {  // closing text across rows and across two cycles
    ThinkControl tc;
    tc.start(5, {kT1, kT2, kClose}, kOpen, kClose, -1, true);  // the close at positions 2, 3, 4
    tc.push(kA);
    const std::vector<int> tk = {kA, kB, kT1, kT2, kClose, kB};  // P = 0
    auto a = rows_of(tc, tk, 0, 3, nullptr, 3);
    check(a == std::vector<int>({-1, kT1, kT2}), "closing text: forced in order from the cap", str(a));
    tc.push(kB);
    tc.push(kT1);  // the cycle kept row 1's forced token; the next cycle starts after it
    check(tc.count() == 3 && tc.forced(), "closing text: the forced tokens count as reasoning");
    const std::vector<int> tk2 = {kA, kB, kT1, kT2, kClose, kB};
    a = rows_of(tc, tk2, 2, 3, nullptr, 3);
    check(a == std::vector<int>({kT2, kClose, -1}), "closing text: the next cycle goes on where the last stopped", str(a));
    tc.push(kT2);
    tc.push(kClose);
    check(!tc.armed() && tc.count() == 5 && tc.count() <= tc.cap(), "the forced </think> ends the control: the reasoning "
          "holds exactly the cap, the close included");
    a = rows_of(tc, tk2, 4, 2, nullptr, 2);
    check(a == std::vector<int>({-1, -1}), "after the close every row is free", str(a));
  }
  {  // a KFlash tree: row r's depth is its parent's + 1; rows from `live` on are never accepted
    ThinkControl tc;
    tc.start(3, {kClose}, kOpen, kClose, -1, true);
    // rows: 0 root; 1 (parent 0); 2 (parent 0, a drafted </think>); 3 (parent 1); 4 (parent 2); 5 past live
    const int parent[] = {-1, 0, 0, 1, 2, 3};
    const std::vector<int> tk = {kA, kA, kClose, kB, kB, kB};  // P = 0: tokens[r] is row r's drafted token
    const auto a = rows_of(tc, tk, 0, 6, parent, 5);
    check(a == std::vector<int>({-1, -1, -1, kClose, -1, -1}),
          "tree: depth 2 forces, a branch through a drafted </think> and rows past live do not", str(a));
  }
  {  // a tool call inside the reasoning ends the control when end_on_tool_call is on
    ThinkControl tc;
    tc.start(3, {kClose}, kOpen, kClose, kCall, true);
    const std::vector<int> tk = {kA, kCall, kB, kClose};
    auto a = rows_of(tc, tk, 0, 4, nullptr, 4);
    check(a == std::vector<int>({-1, -1, -1, -1}), "end_on_tool_call: a drafted <tool_call> ends it on its path", str(a));
    ThinkControl off;
    off.start(3, {kClose}, kOpen, kClose, -1, true);
    a = rows_of(off, tk, 0, 4, nullptr, 4);
    check(a == std::vector<int>({-1, -1, kClose, -1}), "end_on_tool_call off: the cap still applies", str(a));
    tc.push(kCall);
    check(!tc.armed(), "end_on_tool_call: a handled <tool_call> ends it");
  }
  {  // a cap of 0 (the engine turns thinking off first; this is the fallback for a template that opens the reasoning all
     // the same): the first sample is the close
    ThinkControl tc;
    tc.start(0, {kClose}, kOpen, kClose, -1, true);
    check(tc.near(1), "cap 0: near() at once");
    const std::vector<int> tk = {0};
    check(rows_of(tc, tk, 0, 1, nullptr, 1) == std::vector<int>({kClose}), "cap 0: row 0 forces </think>");
  }
  {  // a template that leaves <think> to the model: whitespace keeps the control waiting, counting starts after <think>
    ThinkControl tc;
    tc.start(2, {kClose}, kOpen, kClose, -1, false, spaces());
    tc.push(kNl);
    tc.push(kSp);
    check(tc.armed() && !tc.in_reasoning() && tc.count() == 0 &&
              rows_of(tc, {0}, 0, 1, nullptr, 1) == std::vector<int>({-1}),
          "whitespace before <think>: still waiting, nothing forced");
    tc.push(kOpen);
    tc.push(kA);
    check(tc.count() == 1 && rows_of(tc, {0}, 0, 1, nullptr, 1) == std::vector<int>({kClose}),
          "a late <think> after whitespace: the cap counts from it");
  }
  {  // the reply starts outside a reasoning block (any token but whitespace or <think>): the control ends
    ThinkControl tc;
    tc.start(4, {kClose}, kOpen, kClose, -1, false, spaces());
    tc.push(kNl);
    tc.push(kA);
    check(!tc.armed() && !tc.near(1000) && rows_of(tc, {0}, 0, 1, nullptr, 1) == std::vector<int>({-1}),
          "content before <think>: disarmed, nothing forced, never near (the request may batch)");
    tc.push(kOpen);
    for (int i = 0; i < 6; ++i) tc.push(kA);
    check(!tc.armed() && tc.count() == 0 && !tc.forced(), "content before <think>: a later <think> is not capped");
    ThinkControl f;  // the same on a verify's rows
    f.start(2, {kClose}, kOpen, kClose, -1, false, spaces());
    check(rows_of(f, {0, kNl, kOpen, kA, kClose}, 0, 5, nullptr, 5) == std::vector<int>({-1, -1, -1, kClose, -1}),
          "rows: whitespace then a drafted <think>: the cap applies from it");
    check(rows_of(f, {0, kA, kOpen, kA, kA}, 0, 5, nullptr, 5) == std::vector<int>({-1, -1, -1, -1, -1}),
          "rows: a drafted content token then <think>: nothing forced");
  }
  {  // the first reasoning block only: after its </think> a second <think> is neither counted nor capped
    ThinkControl tc;
    tc.start(3, {kClose}, kOpen, kClose, -1, true, spaces());
    tc.push(kA);
    tc.push(kClose);
    tc.push(kOpen);
    for (int i = 0; i < 5; ++i) tc.push(kA);
    check(!tc.armed() && tc.count() == 2 && !tc.forced() &&
              rows_of(tc, {kA, kA, kA}, 0, 3, nullptr, 3) == std::vector<int>({-1, -1, -1}),
          "a second reasoning block: not capped, the count stays the first block's");
  }
  {  // a cap shorter than the closing text: the text is dropped, </think> alone fits (the reasoning stays within it)
    ThinkControl tc;
    tc.start(2, {kT1, kT2, kClose}, kOpen, kClose, -1, true);
    tc.push(kA);
    check(tc.near(1) && rows_of(tc, {kA}, 0, 1, nullptr, 1) == std::vector<int>({kClose}),
          "short cap: </think> alone at position 1");
    tc.push(kClose);
    check(!tc.armed() && tc.count() == 2 && tc.forced(), "short cap: 2 reasoning tokens for a cap of 2");
    const ThinkClose c1 = think_close_for(1, {kT1, kT2}, kClose);
    const ThinkClose c4 = think_close_for(4, {kT1, kT2}, kClose);
    const ThinkClose c0 = think_close_for(0, {kT1}, kClose);
    check(c1.tokens == std::vector<int>({kClose}) && c1.at == 0 && c4.tokens == std::vector<int>({kT1, kT2, kClose}) &&
              c4.at == 1 && c0.tokens == std::vector<int>({kClose}) && c0.at == 0,
          "think_close_for: the close fits the cap (text dropped when it does not), from cap minus its length");
  }
  {  // think_cap_of: thinking_control's cap over budget_tokens; a budget of 0 (thinking disabled) is no cap
    check(think_cap_of(-1, -1) == -1 && think_cap_of(-1, 0) == -1 && think_cap_of(-1, 4096) == 4096 &&
              think_cap_of(300, 4096) == 300 && think_cap_of(0, 4096) == 0,
          "think_cap_of: precedence and a cap of 0");
  }
  {  // off
    ThinkControl tc;
    tc.start(-1, {kClose}, kOpen, kClose, -1, true);
    check(!tc.on() && !tc.armed() && !tc.near(1000), "a cap of -1 is off");
  }
  // the review of nace-ai/kapitsa#950: a generated <think> on a verify's row
  {  // <think> on row 1 of a chain, cap 4: row 4 samples reasoning position 3, the close
    ThinkControl tc;
    tc.start(4, {kClose}, kOpen, kClose, -1, false);
    const auto a = rows_of(tc, {kA, kOpen, kA, kA, kA, kA, kA, kA}, 0, 8, nullptr, 8);
    check(a[0] == -1 && a[1] == -1 && a[2] == -1 && a[3] == -1 && a[4] == kClose,
          "chain, a drafted <think> at row 1, cap 4: row 4 forces </think>", str(a));
  }
  {  // the same on a tree: the branch under <think> is capped, its sibling without <think> is not
    ThinkControl tc;
    tc.start(3, {kClose}, kOpen, kClose, -1, false);
    const int parent[] = {-1, 0, 0, 1, 3, 2, 4};
    const auto a = rows_of(tc, {kA, kOpen, kB, kA, kA, kB, kA}, 0, 7, parent, 7);
    check(a[4] == kClose && a[6] == kClose && a[1] == -1 && a[3] == -1 && a[2] == -1 &&
              a[5] == -1,
          "tree, a drafted <think>: its descendants from depth 2 force, the sibling branch is free", str(a));
  }
  {  // cap 1: the row whose token is <think> samples reasoning position 0, which is the close
    ThinkControl tc;
    tc.start(1, {kClose}, kOpen, kClose, -1, false);
    check(rows_of(tc, {kA, kOpen, kA}, 0, 3, nullptr, 3)[1] == kClose, "cap 1: the <think> row forces </think>");
  }
  {  // a drafted <think> after a drafted </think> does not re-arm the path
    ThinkControl tc;
    tc.start(2, {kClose}, kOpen, kClose, -1, true);
    tc.push(kA);
    check(rows_of(tc, {kA, kClose, kOpen, kA, kA}, 0, 5, nullptr, 5) == std::vector<int>({kClose, -1, -1, -1, -1}),
          "rows: no re-arm after a drafted </think>");
  }
  {  // closing text after a drafted <think>; a drafted <tool_call> after one ends the path
    ThinkControl tc;
    tc.start(3, {kT1, kClose}, kOpen, kClose, -1, false);
    const auto a = rows_of(tc, {kA, kOpen, kA, kT1, kClose, kA}, 0, 6, nullptr, 6);
    check(a[2] == kT1 && a[3] == kClose && a[4] == -1, "closing text after a drafted <think>: in order", str(a));
    ThinkControl c;
    c.start(3, {kClose}, kOpen, kClose, kCall, false);
    const auto b = rows_of(c, {kA, kOpen, kCall, kA, kA, kA}, 0, 6, nullptr, 6);
    check(b[2] == -1 && b[3] == -1 && b[4] == -1, "a drafted <tool_call> after a drafted <think>: free", str(b));
  }
  {  // rows from `live` on are never accepted: not forced even past the cap
    ThinkControl tc;
    tc.start(3, {kClose}, kOpen, kClose, -1, true);
    check(rows_of(tc, {kA, kA, kA, kA}, 0, 4, nullptr, 2) == std::vector<int>({-1, -1, -1, -1}),
          "rows past live: free");
  }
  {  // a forced token past the mask's width: the row allows everything (fail-open) and unmasked() counts it
    ThinkControl tc;
    tc.start(1, {100}, kOpen, 100, -1, true);
    check(rows_of(tc, {0}, 0, 1, nullptr, 1) == std::vector<int>({-1}) && tc.unmasked() == 1,
          "a forced id past the mask width: counted for the engine's warning");
  }
  {  // random chains and trees: fill() agrees with push(), and a control not near(rows) forces none of its rows
    std::mt19937 rng(949);
    const int alphabet[] = {kA, kB, kOpen, kClose, kCall, kT1, kNl, kSp};
    int bad = 0, near_bad = 0;
    std::string first, near_first;
    for (int it = 0; it < 20000; ++it) {
      ThinkControl tc;
      const int cap = int(rng() % 10);
      const bool text = rng() % 3 == 0;
      const int call = rng() % 2 ? kCall : -1;
      tc.start(cap, text ? std::vector<int>{kT1, kClose} : std::vector<int>{kClose}, kOpen, kClose, call,
               rng() % 2 == 0, spaces());
      for (int i = int(rng() % 5); i > 0; --i) tc.push(alphabet[rng() % 8]);
      const int rows = 1 + int(rng() % 9), live = 1 + int(rng() % unsigned(rows));
      std::vector<int> tk(static_cast<size_t>(rows)), par(static_cast<size_t>(rows), -1);
      for (auto& t : tk) t = alphabet[rng() % 8];
      const bool tree = rng() % 2;
      for (int r = 1; r < rows; ++r) par[size_t(r)] = tree ? int(rng() % unsigned(r)) : r - 1;
      std::string why;
      if (!equivalent(tc, tk, rows, tree ? par.data() : nullptr, live, &why) && !bad++)
        first = "cap " + std::to_string(cap) + (tree ? " tree: " : " chain: ") + why;
      if (!tc.near(rows)) {
        std::vector<uint32_t> m(size_t(rows) * W);
        if (tc.fill(m.data(), W, tk.data(), 0, rows, tree ? par.data() : nullptr, live) > 0 && !near_bad++)
          near_first = "cap " + std::to_string(cap) + ", rows " + std::to_string(rows) + ": " + str(tk);
      }
    }
    check(bad == 0, "random chains and trees: fill() agrees with push() on every acceptable row", first);
    check(near_bad == 0, "random: a control not near(rows) forces none of its rows", near_first);
  }
  {  // whole replies under each decode loop's gating: a target that does not
     // close its reasoning by itself and a drafter that mostly predicts it, so accepted runs are long. Whatever the
     // loop, the capped block must end within the cap (a cap of 0 inside the reasoning: by its one forced token).
     // Gating as in src/engine/engine.cpp: 0 a DFlash2 / KFlash cycle masked when near(its rows); 1 a loop that cannot
     // mask, handing over to masked plain steps at near(33) after a handled token; 2 the plain loop, one unhandled
     // step in flight, handing over at near(2); 3 batched plain passes, the next pass encoded (masked when near(9))
     // before the host handled the pass in flight; 4 a batched cycle of up to 8 rows masked when near(9).
    std::mt19937 rng(950);
    const int alphabet[] = {kA, kB, kOpen, kClose, kCall, kT1, kNl, kSp};
    auto state_of = [](const ThinkControl& tc) {
      ThinkControl::State s;
      s.phase = !tc.armed() ? ThinkControl::kEnded : tc.in_reasoning() ? ThinkControl::kThink : ThinkControl::kPre;
      s.think = tc.count();
      return s;
    };
    auto want = [&](const ThinkControl::State& s) -> int {  // the unmasked target's sample after a path in state s
      if (s.phase == ThinkControl::kPre) return rng() % 2 ? kOpen : rng() % 2 ? kNl : kSp;
      if (s.phase == ThinkControl::kThink) return rng() % 32 ? kA : alphabet[rng() % 8];
      return kB;
    };
    // one verify of `rows` rows after the handled tokens (masked: under fill()'s masks): the tokens the cycle writes,
    // the accepted drafts and the sample after them
    auto cycle = [&](const ThinkControl& tc, int rows, bool masked, bool tree) {
      const size_t n = static_cast<size_t>(rows);
      std::vector<int> par(n, -1), tk(n, 0), wants(n, 0), smp(n, 0);
      std::vector<ThinkControl::State> st(n);
      const int live = tree ? 1 + int(rng() % unsigned(rows)) : rows;
      st[0] = state_of(tc);
      wants[0] = want(st[0]);
      for (int r = 1; r < rows; ++r) {
        const size_t i = static_cast<size_t>(r);
        par[i] = tree ? int(rng() % unsigned(r)) : r - 1;
        const size_t p = static_cast<size_t>(par[i]);
        tk[i] = rng() % 10 ? wants[p] : alphabet[rng() % 8];
        st[i] = tc.step(st[p], tk[i]);
        wants[i] = want(st[i]);
      }
      std::vector<uint32_t> m(n * W, 0xffffffffu);
      if (masked) tc.fill(m.data(), W, tk.data(), 0, rows, tree ? par.data() : nullptr, live);
      for (int r = 0; r < rows; ++r) {
        const int a = allowed(m, r);
        smp[static_cast<size_t>(r)] = a >= 0 ? a : wants[static_cast<size_t>(r)];
      }
      std::vector<int> out;
      int cur = 0;
      for (;;) {
        int kid = -1;
        for (int r = 1; r < live && kid < 0; ++r) {
          const size_t i = static_cast<size_t>(r);
          if (par[i] == cur && tk[i] == smp[static_cast<size_t>(cur)]) kid = r;
        }
        if (kid < 0) break;
        cur = kid;
        out.push_back(tk[static_cast<size_t>(cur)]);
      }
      out.push_back(smp[static_cast<size_t>(cur)]);
      return out;
    };
    int bad = 0, open_end = 0;
    std::string first;
    for (int it = 0; it < 50000; ++it) {
      ThinkControl tc;
      const int cap = int(rng() % (it % 2 ? 80u : 12u));
      const bool in_think = cap == 0 || rng() % 2 == 0;  // (the engine starts a cap of 0 only inside the reasoning)
      std::vector<int> forced(static_cast<size_t>(rng() % 3), int(kT1));
      forced.push_back(kClose);
      const int call = rng() % 3 ? -1 : kCall;
      tc.start(cap, forced, kOpen, kClose, call, in_think, spaces());
      // the oracle: the first reasoning block counted from the emitted tokens alone, not by the control
      int block = 0;
      bool in_block = in_think, over = false;
      auto emit = [&](int t) {
        if (!over && in_block) {
          ++block;
          over = t == kClose || t == call;
        } else if (!over) {
          in_block = t == kOpen;
          over = !in_block && t != kNl && t != kSp;
        }
        tc.push(t);
      };
      const int mode = it % 5;
      auto plain_masked = [&] {  // the constrained loop: every step under its mask
        for (int k = 0; tc.armed() && k < 400; ++k) emit(cycle(tc, 1, true, false)[0]);
      };
      // the prompt's last chunk samples the first token, masked when the prompt is in the reasoning at the cap
      const int first_tok = cycle(tc, 1, in_think && tc.near(1), false)[0];
      if (mode == 0) {
        const int mrows = 4 << (rng() % 4);  // 4, 8, 16, 32
        const bool tree = rng() % 2 == 0;
        emit(first_tok);
        for (int k = 0; tc.armed() && k < 400; ++k) {
          const int rows = tree ? 1 + int(rng() % unsigned(mrows)) : mrows;
          for (int t : cycle(tc, rows, tc.near(mrows), tree)) emit(t);
        }
      } else if (mode == 1) {
        const int R = 2 << (rng() % 4);  // 2, 4, 8, 16
        bool brk = tc.near(33);          // (tc_direct: the constrained loop from the first token)
        emit(first_tok);
        brk = brk || tc.near(33);
        for (int k = 0; !brk && tc.armed() && k < 400; ++k)
          for (int t : cycle(tc, 1 + int(rng() % unsigned(R)), false, R == 16)) {
            emit(t);
            if (tc.near(33)) {
              brk = true;
              break;  // (the rest of the accepted run is dropped, as the loops drop it)
            }
          }
        if (brk) plain_masked();
      } else if (mode == 2) {
        if (tc.near(2)) {
          emit(first_tok);
          plain_masked();
        } else {
          int cur = first_tok;
          for (int k = 0; k < 400; ++k) {
            // the step in flight was encoded unmasked before `cur` is handled; its sample follows `cur`
            ThinkControl after = tc;
            after.push(cur);
            const int next = cycle(after, 1, false, false)[0];
            emit(cur);
            if (tc.near(2)) {
              emit(next);  // tc_unhandled: the constrained loop handles the sample in flight first
              plain_masked();
              break;
            }
            if (!tc.armed()) break;
            cur = next;
          }
        }
      } else if (mode == 3) {
        emit(first_tok);
        for (int run = 0; tc.armed() && run < 60; ++run) {
          // pass 0: encoded, and its mask filled, with nothing in flight
          int in_flight = cycle(tc, 1, tc.near(9), false)[0];
          for (int c = 0; c < 8; ++c) {
            const bool have_next = c + 1 < 8;
            const bool mk_next = have_next && tc.near(9);  // pass c + 1 is encoded before pass c's token is handled
            emit(in_flight);
            if (!have_next) break;
            in_flight = cycle(tc, 1, mk_next, false)[0];  // (its mask: the state after pass c's token, member_masks)
          }
        }
      } else {
        emit(first_tok);
        for (int k = 0; tc.armed() && k < 400; ++k)
          for (int t : cycle(tc, 1 + int(rng() % 8), tc.near(9), false)) emit(t);
      }
      const int most = cap == 0 ? 1 : cap;
      if (tc.armed()) ++open_end;
      if ((block > most || tc.count() != block) && !bad++)
        first = "loop " + std::to_string(mode) + ", cap " + std::to_string(cap) +
                (in_think ? ", in the reasoning" : ", before <think>") + ": " + std::to_string(block) +
                " reasoning tokens, count() " + std::to_string(tc.count());
    }
    check(bad == 0, "whole replies: the capped block ends within the cap under every loop's gating", first);
    check(open_end == 0, "whole replies: every control ended (the cap was reached or the reasoning closed)",
          std::to_string(open_end) + " still armed");
  }
  printf("%d passed, %d failed\n", passes, fails);
  return fails ? 1 : 0;
}
