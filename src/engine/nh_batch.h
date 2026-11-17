#pragma once
// Which Nemotron-H models batch the decode of requests at once, and how many rows each member gets
// (Engine::batch_rows), pinned on the CPU by nh_batch_test.
#include <algorithm>
#include <vector>

namespace ni {

// a dense Nemotron-H model (no LatentMoE layer, 'E') without a block drafter: requests at once batch plain rows
inline bool nh_batches(const std::vector<char>& kind, bool block_drafter) {
  return !block_drafter && std::find(kind.begin(), kind.end(), 'E') == kind.end();
}

// rows per member for n members, with no drafter: 1 (every member's pending token), 0 = take turns.
// mode: NI_BATCH (-1 unset, 0 take turns); env_rows: NI_BATCH_ROWS's entry for n (-1 unset)
inline int nh_batch_rows(int n, int mode, int env_rows) {
  return mode == 0 || n < 2 || n > 8 || env_rows == 0 ? 0 : 1;
}

}  // namespace ni
