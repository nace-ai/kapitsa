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

  for (int n = 2; n <= 8; ++n) CHECK(ni::nh_batch_rows(n, -1, -1) == 1, "2 to 8 members: one row each");
  CHECK(ni::nh_batch_rows(1, -1, -1) == 0, "one request decodes alone");
  CHECK(ni::nh_batch_rows(9, -1, -1) == 0, "more than 8 members take turns");
  CHECK(ni::nh_batch_rows(4, 0, -1) == 0, "NI_BATCH=0 takes turns");
  CHECK(ni::nh_batch_rows(4, 1, -1) == 1, "NI_BATCH=1 batches");
  CHECK(ni::nh_batch_rows(4, -1, 0) == 0, "NI_BATCH_ROWS=4:0 takes turns at 4");
  CHECK(ni::nh_batch_rows(4, -1, 3) == 1, "a drafter's row count gives plain rows one row");

  if (failures) return 1;
  std::puts("nh_batch_test: ok");
  return 0;
}
