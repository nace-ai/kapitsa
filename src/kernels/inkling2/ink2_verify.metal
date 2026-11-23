#include "common.h"

struct Ink2VerifyP {
  uint visits, kv, hidden, rows, keep;
};

kernel void ink2_verify_commit(device const uint* before [[buffer(0)]], device const uint* raw [[buffer(1)]],
                               device uint* history [[buffer(2)]], constant Ink2VerifyP& p [[buffer(3)]],
                               uint i [[thread_position_in_grid]]) {
  const uint width = 2 * p.kv + 2 * p.hidden;
  if (i >= p.visits * width) return;
  const uint visit = i / width;
  const uint q = i % width;
  const uint channels = q < 2 * p.kv ? p.kv : p.hidden;
  const uint prefix = q < 2 * p.kv ? (q / p.kv) * p.kv : 2 * p.kv + ((q - 2 * p.kv) / p.hidden) * p.hidden;
  const uint c = q - prefix;
  const uint old_base = visit * 3 * width + 3 * prefix;
  const uint raw_base = visit * p.rows * width + p.rows * prefix;
  for (uint r = 0; r < 3; ++r) {
    const uint j = p.keep + r;
    history[old_base + r * channels + c] =
        j < 3 ? before[old_base + j * channels + c] : raw[raw_base + (j - 3) * channels + c];
  }
}
