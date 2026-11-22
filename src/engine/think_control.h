// Thinking budget: a hard cap on the reasoning, enforced while decoding (Anthropic thinking.budget_tokens, or a
// request's thinking_control.hard_cap). The cap counts every reasoning token, the forced close included: the capped
// block (ChatResult::think_tokens, which counts its closing </think>) never holds more than `cap` tokens; the API's
// reasoning_tokens also counts later blocks and a tool call held inside the reasoning. When it reaches
// cap minus the close's length, the next tokens are forced: an optional closing text, then </think>; the model then
// writes its reply (or a tool call) in the same turn. Forcing is a token mask on the verify or step rows (Seq::mask,
// enc_logit_mask), so speculative decoding stays exact: a draft that is not the forced token is rejected, and the row's
// masked sample, the forced token, is what the cycle keeps. A cap of 0 is thinking turned off (think_cap_of, Engine::chat
// renders the prompt as thinking off), so the reply carries no empty reasoning block.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ni {

// The request's cap: thinking_control.hard_cap, else a positive budget_tokens (messages_prompt makes budget_tokens 0 a
// cap of 0, thinking off; thinking disabled or absent is no cap); -1: none.
inline int think_cap_of(int think_cap, int thinking_budget) {
  return think_cap >= 0 ? think_cap : thinking_budget > 0 ? thinking_budget : -1;
}

// What a cap forces and from which reasoning position, so that the reasoning, the close included, holds at most `cap`
// tokens: `text` (the closing text's tokens) then `close` (</think>). A cap shorter than the whole close drops the text
// and forces </think> alone. A cap of 0 forces </think> as the first token: the fallback for a template that thinking
// off does not reach (the engine turns thinking off for a cap of 0 first), the one case past the cap, by that one token.
struct ThinkClose {
  std::vector<int> tokens;
  int at = 0;  // the reasoning position the first forced token takes
};
ThinkClose think_close_for(int cap, std::vector<int> text, int close);

class ThinkControl {
 public:
  // The control's state after a path of tokens. push() (the handled tokens) and fill() (each verify row's path) take
  // the same step(), so a row is masked exactly as the token that reaches it would leave the control.
  //   kPre:   the reasoning has not started: the template left the opening tag to the model. Tokens marked in
  //           `space` (whitespace-only tokens, or a leading role header such as Inkling-2's <|message_model|>) keep
  //           the control waiting; any other token is the reply starting outside a reasoning block, which ends it.
  //   kThink: in the reasoning; `think` reasoning tokens so far, the row's sample takes reasoning position `think`.
  //   kEnded: </think>, a tool call (with end_on_tool_call), or a reply outside a reasoning block. Final: the cap
  //           covers the first reasoning block only (a later <think> in the same reply is not capped).
  enum Phase : uint8_t { kPre, kThink, kEnded };
  struct State {
    Phase phase = kEnded;
    int think = 0;
  };

  // cap: the most reasoning tokens, the close included (< 0: off). forced: the closing text's tokens then </think>
  // (its last token), as think_close_for shortens it for the cap. open / close / call: the <think>, </think> and
  // <tool_call> token ids; call < 0: a tool call inside the reasoning does not end the control. in_think: the prompt
  // ends inside the reasoning (its template opened the think block), else the reasoning starts at a generated <think>.
  // space: per token id, 1 when a token keeps kPre waiting before `open` (whitespace, role headers; null or short: none).
  void start(int cap, std::vector<int> forced, int open, int close, int call, bool in_think,
             const std::vector<uint8_t>* space = nullptr);
  bool on() const { return cap_ >= 0 && !forced_.empty(); }
  // the control still applies: before or in the first reasoning block
  bool armed() const { return on() && st_.phase != kEnded; }
  bool in_reasoning() const { return st_.phase == kThink; }
  int count() const { return st_.think; }  // reasoning tokens handled so far (the closing </think> included)
  bool forced() const { return forced_n_ > 0; }
  int cap() const { return cap_; }
  // forced rows fill() could not mask (the token id past the mask's width: such a row allows every token)
  int unmasked() const { return unmasked_; }

  // A generated token, in order, after the engine handled it.
  void push(int t);
  // Whether one of the next `rows` samples (reasoning positions count() .. count() + rows - 1) can reach the cap:
  // a loop sampling `rows` tokens without masks must hand over before that (or mask its rows).
  bool near(int rows) const { return armed() && st_.think + rows > at_; }

  // Masks for a verify's rows (uint32 words, W per row; bit t set: token t allowed), row r sampling the token after
  // row r's path: row 0 after the handled tokens, row r > 0 after tokens[P + r] under its parent (parent[r]: a tree's,
  // null: a chain's, r - 1; rows from `live` on are never accepted). A row in kPre with cap 0 forbids `open`; a row in
  // the reasoning at or past the cap allows only the forced token at its position; every other row allows every token.
  // Returns the rows constrained by the control.
  int fill(uint32_t* mask, size_t W, const int* tokens, int P, int rows, const int* parent, int live) const;
  // fill() for a DFlash2 cycle: tree, its KFlash tree descriptor (src/kernels/kflash.h: the parents and the live rows),
  // null: a chain of `rows` live rows.
  int fill_cycle(uint32_t* mask, size_t W, const int* tokens, int P, int rows, const int* tree) const;

  // A batched member's rows in one cycle (at most 8) and the plain passes encoded before the one in flight is handled:
  // a member whose control is near(kBatchNear) is batched masked (Engine::BatchJob::masked).
  static constexpr int kBatchNear = 9;

  // the state after token t (the transition push() and fill() share)
  State step(State s, int t) const;

 private:
  int forced_at(int pos) const;  // the token forced at reasoning position pos (-1: none)
  bool space(int t) const { return space_ && t >= 0 && size_t(t) < space_->size() && (*space_)[size_t(t)]; }
  int cap_ = -1;
  int at_ = 0;  // the reasoning position of the first forced token (think_close_for)
  std::vector<int> forced_;
  int open_ = -1, close_ = -1, call_ = -1;
  const std::vector<uint8_t>* space_ = nullptr;
  State st_;
  int forced_n_ = 0;          // forced tokens handled (usage: whether the cap was reached)
  mutable int unmasked_ = 0;  // (fill's fail-open rows, for the engine's warning)
};

// The decode loops that cannot mask their rows hand the reply to the constrained loop when near() this many rows after a
// handled token (Engine::chat's tc_rows): a speculative loop, kf::kMaxRows + 1 (it waits for each verify and handles
// its tokens in order, so it stops at the token before the cap; the margin hands over a cycle early); the plain loop,
// 2 (its next step is in flight, unmasked, when a token is handled).
int think_handover_rows(bool spec_loop);

// A batched member's masks for `rows` chain rows from P (Engine::member_masks): its grammar's (structured output, once
// the reasoning is over: grammar_on, grammar_fill), else its thinking budget's while armed, else every token allowed.
template <class GrammarFill>
void think_member_masks(bool grammar_on, GrammarFill&& grammar_fill, const ThinkControl* tc, uint32_t* mask, size_t W,
                        const int* tokens, int P, int rows) {
  if (grammar_on) grammar_fill();
  else if (tc && tc->armed()) tc->fill(mask, W, tokens, P, rows, nullptr, rows);
  else memset(mask, 0xff, size_t(rows) * W * 4);
}

}  // namespace ni
