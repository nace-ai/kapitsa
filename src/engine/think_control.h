// Thinking budget: a hard cap on the reasoning, enforced while decoding (Anthropic thinking.budget_tokens, or a
// request's thinking_control.hard_cap). The cap counts every reasoning token, the forced close included: the reasoning
// (ChatResult::reasoning_tokens, which counts the closing </think>) never holds more than `cap` tokens. When it reaches
// cap minus the close's length, the next tokens are forced: an optional closing text, then </think>; the model then
// writes its reply (or a tool call) in the same turn. Forcing is a token mask on the verify or step rows (Seq::mask,
// enc_logit_mask), so speculative decoding stays exact: a draft that is not the forced token is rejected, and the row's
// masked sample, the forced token, is what the cycle keeps. A cap of 0 is thinking turned off (think_cap_of, Engine::chat
// renders the prompt as thinking off), so the reply carries no empty reasoning block.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ni {

// The request's cap: thinking_control.hard_cap, else Anthropic's budget_tokens (0 there is thinking off, which a cap of
// 0 means too); -1: none.
inline int think_cap_of(int think_cap, int thinking_budget) {
  return think_cap >= 0 ? think_cap : thinking_budget >= 0 ? thinking_budget : -1;
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
  // cap: the most reasoning tokens, the close included (< 0: off). forced: the closing text's tokens then </think>
  // (its last token), as think_close_for shortens it for the cap. open / close / call: the <think>, </think> and
  // <tool_call> token ids; call < 0: a tool call inside the reasoning does not end the control. in_think: the prompt
  // ends inside the reasoning (its template opened the think block), else the reasoning starts at a generated <think>.
  void start(int cap, std::vector<int> forced, int open, int close, int call, bool in_think);
  bool on() const { return cap_ >= 0 && !forced_.empty(); }
  // the control still applies: in the reasoning (or before it), not ended by </think> or a tool call
  bool armed() const { return on() && !ended_; }
  // the reasoning has started (the prompt opened it, or a generated <think> did)
  bool in_reasoning() const { return in_; }
  // The reply started outside a reasoning block (thinking off, or a template whose model skipped <think>): nothing is
  // left to cap, so the request decodes as if it had no control (batched, unmasked).
  void end() { ended_ = true; }
  int count() const { return count_; }  // reasoning tokens handled so far (the closing </think> included)
  bool forced() const { return forced_n_ > 0; }
  int cap() const { return cap_; }

  // A generated token, in order, after the engine handled it.
  void push(int t);
  // Whether one of the next `rows` samples (reasoning positions count() .. count() + rows - 1) can reach the cap:
  // a loop sampling `rows` tokens without masks must hand over before that (or mask its rows).
  bool near(int rows) const { return armed() && count_ + rows > at_; }

  // Masks for a verify's rows (uint32 words, W per row; bit t set: token t allowed), row r sampling the token after
  // row r's path: row 0 after the handled tokens, row r > 0 after tokens[P + r] under its parent (parent[r]: a tree's,
  // null: a chain's, r - 1; rows from `live` on are never accepted). A row past the cap allows only the forced token
  // at its depth; every other row allows every token. Returns the rows that force a token.
  int fill(uint32_t* mask, size_t W, const int* tokens, int P, int rows, const int* parent, int live) const;

 private:
  int forced_at(int pos) const;  // the token forced at reasoning position pos (-1: none)
  int cap_ = -1;
  int at_ = 0;  // the reasoning position of the first forced token (think_close_for)
  std::vector<int> forced_;
  int open_ = -1, close_ = -1, call_ = -1;
  bool in_ = false, ended_ = false;
  int count_ = 0;
  int forced_n_ = 0;  // forced tokens handled (usage: whether the cap was reached)
};

}  // namespace ni
