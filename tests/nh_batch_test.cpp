// Which Nemotron-H models batch requests at once and the rows each member gets (src/engine/nh_batch.h).
// No model, no GPU.
#include "engine/nh_batch.h"

#include <cstdio>

static int failures = 0;
#define CHECK(condition, what)                                                                       \
  do {                                                                                               \
    if (!(condition)) {                                                                              \
      std::fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__, what, #condition);          \
      ++failures;                                                                                    \
    }                                                                                                \
  } while (false)

int main() {
  const std::vector<char> dense = {'M', 'D', 'M', 'A', 'M', 'D'};  // Nano 4B, 9B v2
  const std::vector<char> moe = {'M', 'E', 'M', 'A', 'M', 'E'};    // LatentMoE layers
  CHECK(ni::nh_batches(dense, false), "a dense model without a drafter batches");
  CHECK(!ni::nh_batches(dense, true), "a block drafter keeps its own path");
  CHECK(!ni::nh_batches(moe, false), "a LatentMoE model takes turns");
  CHECK(!ni::nh_batches(moe, true), "a LatentMoE model with a drafter keeps its own path");

  // (NI_BATCH=0 and member counts outside 2 .. 8 return 0 in Engine::batch_rows before this rule)
  CHECK(ni::nh_batch_rows(-1) == 1, "NI_BATCH_ROWS unset: one row each");
  CHECK(ni::nh_batch_rows(0) == 0, "NI_BATCH_ROWS=n:0 takes turns at n");
  CHECK(ni::nh_batch_rows(3) == 1, "a drafter's row count gives plain rows one row");

  if (failures) return 1;
  std::puts("nh_batch_test: ok");
  return 0;
}
