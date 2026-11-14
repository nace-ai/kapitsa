#include "engine/think_control.h"

#include <algorithm>
#include <cstring>
#include <utility>

namespace ni {

ThinkClose think_close_for(int cap, std::vector<int> text, int close) {
  ThinkClose c;
  c.tokens = std::move(text);
  c.tokens.push_back(close);
  if (cap < int(c.tokens.size())) c.tokens = {close};  // (no room for the text: the close alone)
  c.at = std::max(0, cap - int(c.tokens.size()));
  return c;
}

void ThinkControl::start(int cap, std::vector<int> forced, int open, int close, int call, bool in_think,
                         const std::vector<uint8_t>* space) {
  cap_ = cap;
  at_ = 0;
  if (cap >= 0 && !forced.empty()) {  // forced's last token is </think>, the text before it
    std::vector<int> text(forced.begin(), forced.end() - 1);
    ThinkClose c = think_close_for(cap, std::move(text), forced.back());
    forced = std::move(c.tokens);
    at_ = c.at;
  }
  forced_ = std::move(forced);
  open_ = open;
  close_ = close;
  call_ = call;
  space_ = space;
  st_ = State();
  st_.phase = in_think ? kThink : kPre;
  forced_n_ = 0;
  unmasked_ = 0;
}

ThinkControl::State ThinkControl::step(State s, int t) const {
  switch (s.phase) {
    case kPre:  // (a template that leaves <think> to the model: the reasoning starts after it)
      if (t == open_) s.phase = kThink;
      else if (!space(t)) s.phase = kEnded;  // the reply began outside a reasoning block: nothing left to cap
      break;
    case kThink:  // as the engine counts reasoning tokens (ChatResult::reasoning_tokens): </think> too
      ++s.think;
      if (t == close_ || (call_ >= 0 && t == call_)) s.phase = kEnded;
      break;
    case kEnded:
      break;
  }
  return s;
}

void ThinkControl::push(int t) {
  if (!armed()) return;
  if (st_.phase == kThink && forced_at(st_.think) == t) ++forced_n_;
  st_ = step(st_, t);
}

int ThinkControl::forced_at(int pos) const {
  if (cap_ < 0 || pos < at_ || forced_.empty()) return -1;
  // past the last forced token the path holds </think> (the control ended), so the clamp is never read but keeps the
  // index in range
  const size_t k = std::min(size_t(pos - at_), forced_.size() - 1);
  return forced_[k];
}

int ThinkControl::fill(uint32_t* mask, size_t W, const int* tokens, int P, int rows, const int* parent, int live) const {
  // each row's state after its path, by step() from the handled tokens' (ok[r] 0: the row is never accepted, or the
  // control does not apply on its path)
  thread_local std::vector<State> rs;
  thread_local std::vector<uint8_t> ok;
  rs.assign(size_t(rows), State());
  ok.assign(size_t(rows), 0);
  int n = 0;
  for (int r = 0; r < rows; ++r) {
    if (r == 0) {
      rs[0] = st_;
      ok[0] = armed() ? 1 : 0;
    } else {
      const int par = parent ? parent[r] : r - 1;
      if (r < live && par >= 0 && par < r && ok[size_t(par)]) {
        rs[size_t(r)] = step(rs[size_t(par)], tokens[P + r]);
        ok[size_t(r)] = rs[size_t(r)].phase != kEnded;
      }
    }
    uint32_t* const w = mask + size_t(r) * W;
    const int f = ok[size_t(r)] && rs[size_t(r)].phase == kThink ? forced_at(rs[size_t(r)].think) : -1;
    if (f < 0 || size_t(f) >= W * 32) {
      if (f >= 0) ++unmasked_;  // (the mask is narrower than the vocabulary: the engine warns)
      memset(w, 0xff, W * 4);
      continue;
    }
    memset(w, 0, W * 4);
    w[size_t(f) / 32] = 1u << (unsigned(f) % 32);
    ++n;
  }
  return n;
}

}  // namespace ni
