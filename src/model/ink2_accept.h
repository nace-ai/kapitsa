#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <span>

namespace ni {

struct Ink2Choice {
  std::array<int, 8> tokens{};
  int accepted = 0;
  int size() const { return accepted + 1; }
};

// Target row i scores the successor of input i; the last row supplies the bonus.
inline Ink2Choice ink2_choose(std::span<const int> draft, std::span<const int> target) {
  assert(draft.size() <= 7 && target.size() == draft.size() + 1);
  Ink2Choice out;
  while (std::size_t(out.accepted) < draft.size() && draft[out.accepted] == target[out.accepted]) {
    out.tokens[out.accepted] = draft[out.accepted];
    ++out.accepted;
  }
  out.tokens[out.accepted] = target[out.accepted];
  return out;
}

}  // namespace ni
