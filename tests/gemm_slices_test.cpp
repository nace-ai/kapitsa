#include "model/gemm_slices.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define CHECK(condition) do { if (!(condition)) { \
  std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); std::abort(); \
} } while (false)

using ni::Buffer;
using ni::gemm_slices::Plan;

struct alignas(16) Block { uint8_t bytes[16]; };

static Buffer buffer(std::vector<Block>& storage, size_t offset, size_t bytes) {
  return {storage.data(), reinterpret_cast<uint8_t*>(storage.data()), offset, bytes};
}

template<class F> static void rejects(F&& action) {
  bool rejected = false;
  try { action(); } catch (const std::exception&) { rejected = true; }
  CHECK(rejected);
}

static void coverage(size_t rows, size_t output_bytes, bool extra_capacity) {
  constexpr size_t K = 64, N = 65, offset = 32;
  const size_t in_bytes = rows * K * 2, out_bytes = rows * N * output_bytes;
  const size_t extra = extra_capacity ? 4096 : 0;
  std::vector<Block> xs((offset + in_bytes + extra + 15) / 16);
  std::vector<Block> ys((offset + out_bytes + extra + 15) / 16);
  Buffer x = buffer(xs, offset, in_bytes + extra), y = buffer(ys, offset, out_bytes + extra);
  const Plan plan(rows, K, N, x, y, output_bytes);
  CHECK(plan.count() == rows / 2048 + (rows % 2048 != 0));
  size_t covered = 0;
  for (size_t i = 0; i < plan.count(); ++i) {
    const auto s = plan.slice(i);
    CHECK(s.row_start == covered);
    CHECK(s.rows > 0 && s.rows <= 2048);
    CHECK(s.rows == std::min(size_t(2048), rows - covered));
    CHECK(s.input.mtl == x.mtl && s.input.base == x.base);
    CHECK(s.output.mtl == y.mtl && s.output.base == y.base);
    CHECK(s.input.offset == x.offset + covered * K * 2);
    CHECK(s.output.offset == y.offset + covered * N * output_bytes);
    CHECK(s.input.offset % 16 == 0);
    CHECK(s.input.size == s.rows * K * 2);
    CHECK(s.output.size == s.rows * N * output_bytes);
    covered += s.rows;
  }
  CHECK(covered == rows);
  rejects([&] { (void)plan.slice(plan.count()); });
  if (rows) {
    auto short_x = x, short_y = y;
    short_x.size = in_bytes - 1;
    short_y.size = out_bytes - 1;
    rejects([&] { Plan bad(rows, K, N, short_x, y, output_bytes); });
    rejects([&] { Plan bad(rows, K, N, x, short_y, output_bytes); });
  }
}

int main() {
  for (size_t rows : {0, 1, 63, 64, 2047, 2048, 2049, 4096, 4097, 8192, 16384})
    for (size_t bytes : {2, 4})
      for (bool extra : {false, true}) coverage(rows, bytes, extra);

  using ni::gemm_slices::eligible;
  CHECK(eligible(true, false, false, 64, 5120, 17408, 2049));
  CHECK(!eligible(false, false, false, 64, 5120, 17408, 4096));
  CHECK(!eligible(true, true, false, 64, 5120, 17408, 4096));
  CHECK(!eligible(true, false, true, 64, 5120, 17408, 4096));
  CHECK(!eligible(true, false, false, 63, 5120, 17408, 4096));
  CHECK(!eligible(true, false, false, 64, 5121, 17408, 4096));
  CHECK(!eligible(true, false, false, 64, 5120, 17409, 4096));
  for (int rows : {-1, 0, 1, 63, 64, 2047, 2048}) CHECK(!eligible(true, false, false, 64, 5120, 17408, rows));

  alignas(16) uint8_t storage[256]{};
  Buffer b{storage, storage, 0, sizeof storage};
  const size_t max = std::numeric_limits<size_t>::max();
  rejects([&] { Plan bad(1, max, 1, b, b, 2); });
  rejects([&] { Plan bad(1, 64, max, b, b, 4); });
  rejects([&] { Plan bad(max, 64, 1, b, b, 2); });
  rejects([&] { Plan bad(2, 64, max / 4, b, b, 4); });
  Buffer wrapping = b; wrapping.offset = max - 15; wrapping.size = 32;
  rejects([&] { Plan bad(0, 64, 1, wrapping, b, 2); });
  rejects([&] { Plan bad(0, 64, 1, b, wrapping, 4); });
  Buffer misaligned = b; misaligned.offset = 2;
  rejects([&] { Plan bad(1, 64, 1, misaligned, b, 2); });
  rejects([&] { Plan bad(1, 64, 1, b, misaligned, 4); });
  Buffer misaligned_base = b; misaligned_base.base += 2;
  rejects([&] { Plan bad(1, 64, 1, misaligned_base, b, 2); });
  rejects([&] { Plan bad(1, 64, 1, b, misaligned_base, 4); });
  rejects([&] { Plan bad(1, 1, 1, b, b, 2); });
  rejects([&] { Plan bad(1, 0, 1, b, b, 2); });
  rejects([&] { Plan bad(1, 64, 0, b, b, 2); });
  rejects([&] { Plan bad(1, 64, 1, b, b, 8); });
  Buffer half_output = b; half_output.offset = 2;
  CHECK(Plan(1, 64, 1, b, half_output, 2).slice(0).output.offset == 2);
  std::puts("gemm_slices_test: all checks passed");
}
