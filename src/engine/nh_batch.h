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

// rows per member with no drafter, once Engine::batch_rows has ruled out NI_BATCH=0 and fewer than 2 or more than 8
// members: 1 (every member's pending token), or 0 = take turns when NI_BATCH_ROWS gives 0 for this count.
// env_rows: NI_BATCH_ROWS's entry for the member count (-1 unset; a drafter's row count still means one plain row)
inline int nh_batch_rows(int env_rows) { return env_rows == 0 ? 0 : 1; }

}  // namespace ni
