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

void ThinkControl::start(int cap, std::vector<int> forced, int open, int close, int call, bool in_think) {
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
  in_ = in_think;
  ended_ = false;
  count_ = 0;
  forced_n_ = 0;
}

void ThinkControl::push(int t) {
  if (!armed()) return;
  if (!in_) {  // (a template that leaves <think> to the model: the reasoning starts after it)
    in_ = t == open_;
    return;
  }
  // as the engine counts reasoning tokens (ChatResult::reasoning_tokens): every token inside the block, </think> too
  if (forced_at(count_) == t) ++forced_n_;
  ++count_;
  if (t == close_ || (call_ >= 0 && t == call_)) ended_ = true;
}

int ThinkControl::forced_at(int pos) const {
  if (cap_ < 0 || pos < at_ || forced_.empty()) return -1;
  // past the last forced token the path holds </think> (the control ended), so the clamp is never read but keeps the
  // index in range
  const size_t k = std::min(size_t(pos - at_), forced_.size() - 1);
  return forced_[k];
}

int ThinkControl::fill(uint32_t* mask, size_t W, const int* tokens, int P, int rows, const int* parent, int live) const {
  // depth[r]: tokens on row r's path after the handled ones (-1: the row is not controlled: its path left the
  // reasoning, or it is never accepted)
  thread_local std::vector<int> depth;
  depth.assign(size_t(rows), -1);
  int n = 0;
  for (int r = 0; r < rows; ++r) {
    if (r == 0) {
      depth[0] = armed() && in_ ? 0 : -1;
    } else {
      const int par = parent ? parent[r] : r - 1;
      const int t = tokens[P + r];
      if (r < live && par >= 0 && par < r && depth[size_t(par)] >= 0 && t != close_ && !(call_ >= 0 && t == call_))
        depth[size_t(r)] = depth[size_t(par)] + 1;
    }
    uint32_t* const w = mask + size_t(r) * W;
    const int f = depth[size_t(r)] >= 0 ? forced_at(count_ + depth[size_t(r)]) : -1;
    if (f < 0 || size_t(f) >= W * 32) {
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
