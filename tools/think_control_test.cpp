// The thinking budget's row masks (src/engine/think_control.h) on the CPU: chains and KFlash trees (parents), a drafted
// </think> or tool call ending the control on its path, the closing text forced across rows and cycles, rows never
// accepted, a cap of 0, a template that leaves <think> to the model, and the reasoning count against the engine's.
// The cap counts the forced close: the close starts at cap minus its length (think_close_for), so a cap of 6 with </think>
// alone forces at reasoning position 5.
#include "engine/think_control.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace ni;

namespace {

int fails = 0, passes = 0;

void check(bool ok, const std::string& name, const std::string& detail = "") {
  printf("%s %s%s\n", ok ? "PASS" : "FAIL", name.c_str(), ok || detail.empty() ? "" : ("\n     " + detail).c_str());
  ok ? ++passes : ++fails;
}

constexpr int kOpen = 10, kClose = 11, kCall = 12, kA = 40, kB = 41, kT1 = 50, kT2 = 51;
constexpr size_t W = 3;  // 96 tokens

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
  {  // a template that leaves <think> to the model: counting starts after the generated <think>
    ThinkControl tc;
    tc.start(2, {kClose}, kOpen, kClose, -1, false);
    tc.push(kA);
    check(tc.count() == 0 && rows_of(tc, {0}, 0, 1, nullptr, 1) == std::vector<int>({-1}), "no <think> yet: nothing forced");
    tc.push(kOpen);
    tc.push(kA);
    check(tc.count() == 1 && rows_of(tc, {0}, 0, 1, nullptr, 1) == std::vector<int>({kClose}),
          "after a generated <think>: the cap counts from it");
  }
  {  // the reply starts outside a reasoning block (thinking off, or the model skipped <think>): end() disarms it
    ThinkControl tc;
    tc.start(4, {kClose}, kOpen, kClose, -1, false);
    tc.push(kA);
    check(tc.armed() && !tc.in_reasoning(), "no <think> yet: still armed, not in the reasoning");
    tc.end();
    check(!tc.armed() && !tc.near(1000) && rows_of(tc, {0}, 0, 1, nullptr, 1) == std::vector<int>({-1}),
          "end(): disarmed, nothing forced, never near (the request may batch)");
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
  {  // think_cap_of: thinking_control's cap over budget_tokens; budget 0 (thinking off) is a cap of 0
    check(think_cap_of(-1, -1) == -1 && think_cap_of(-1, 0) == 0 && think_cap_of(-1, 4096) == 4096 &&
              think_cap_of(300, 4096) == 300 && think_cap_of(0, 4096) == 0,
          "think_cap_of: precedence and a cap of 0");
  }
  {  // off
    ThinkControl tc;
    tc.start(-1, {kClose}, kOpen, kClose, -1, true);
    check(!tc.on() && !tc.armed() && !tc.near(1000), "a cap of -1 is off");
  }
  printf("%d passed, %d failed\n", passes, fails);
  return fails ? 1 : 0;
}
