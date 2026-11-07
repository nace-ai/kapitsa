// Slot selection rules that do not depend on model state or GPU resources.
#pragma once

#include <cstddef>

namespace ni {

inline bool should_allocate_new_conversation(size_t best_lcp, size_t slot_rows, size_t prompt_rows) {
  const bool extends_live_slot = best_lcp == slot_rows && best_lcp < prompt_rows;
  return best_lcp < 64 && !extends_live_slot;
}

}  // namespace ni
