// Prints and checks the element layout of the int32 destination cooperative tensor qmti8s (src/kernels/tensor/qmti8.metal)
// takes from matmul2d: the 32 x 16 x 64 int8 x int8 (right transposed) product of one simdgroup. The layout is the
// implementation's, so a new GPU can differ; qmti8s reads it from elements 0, 1 and 4 of each lane and assumes element i
// sits at (column c0 + dc * (i & 3), row r0 + dr * (i >> 2)), dc = column(1) - column(0), dr = row(4) - row(0).
// This checks that assumption for every lane and element, and that the 32 lanes cover the 32 x 16 tile exactly once.
//   coop_layout [-v]   (-v: every lane's (column, row) list)
// Exit status 0 when qmti8s's assumption holds on this GPU, 1 when it does not, 2 without Metal 4 tensor ops.
#include "metal/gpu.h"
#include "util/log.h"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace ni;

static const char* kSrc = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// out[0]: capacity; out[1 + (lane * 32 + i) * 2 + {0, 1}]: element i's (column, row)
[[kernel]] void coop_layout_i8(device int* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int BK = 64, SM = 32, HN = 16;
  threadgroup int8_t xs[SM * BK], ws[HN * BK];
  constexpr auto desc =
      matmul2d_descriptor(SM, HN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  typedef tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tg_t;
  auto tX = tg_t(xs, dextents<int32_t, 2>(BK, SM));
  auto tW = tg_t(ws, dextents<int32_t, 2>(BK, HN));
  auto c = op.template get_destination_cooperative_tensor<tg_t, tg_t, int32_t>();
  if (lane == 0) out[0] = int(c.get_capacity());
  for (uint16_t i = 0; i < c.get_capacity() && i < 32; ++i) {
    const auto d = c.get_multidimensional_index(i);
    out[1 + (lane * 32 + i) * 2] = int(d[0]);
    out[2 + (lane * 32 + i) * 2] = int(d[1]);
  }
}
)MSL";

int main(int argc, char** argv) {
  const bool verbose = argc > 1 && !strcmp(argv[1], "-v");
  Gpu& g = Gpu::get();
  printf("%s, Apple%d\n", g.name().c_str(), g.apple_family());
  g.compile_extra("coop_layout", kSrc, true);
  Pipeline p = g.pipeline("coop_layout_i8", false);
  if (!p.pso) {
    printf("no pipeline (needs Metal 4 tensor ops)\n");
    return 2;
  }
  Buffer out = g.alloc((1 + 32 * 32 * 2) * 4);
  memset(out.ptr(), 0xff, out.size);
  auto e = g.begin();
  e->use(p);
  e->buf(0, out);
  e->dispatch(1, 1, 1, 32);
  e->commit_and_wait();
  const int* o = out.as<int>();
  const int cap = o[0];
  auto col = [&](int lane, int i) { return o[1 + (lane * 32 + i) * 2]; };
  auto row = [&](int lane, int i) { return o[2 + (lane * 32 + i) * 2]; };
  int bad = 0;
  if (cap != 16) {
    printf("capacity %d, qmti8s needs 16\n", cap);
    bad++;
  }
  std::vector<int> hits(32 * 16, 0);
  for (int lane = 0; lane < 32 && cap == 16; ++lane) {
    const int c0 = col(lane, 0), r0 = row(lane, 0), dc = col(lane, 1) - c0, dr = row(lane, 4) - r0;
    if (verbose) printf("lane %2d: c0 %2d r0 %2d dc %d dr %d:", lane, c0, r0, dc, dr);
    for (int i = 0; i < cap; ++i) {
      const int c = col(lane, i), r = row(lane, i);
      if (verbose) printf(" (%d,%d)", c, r);
      if (c != c0 + dc * (i & 3) || r != r0 + dr * (i >> 2)) {
        if (bad++ < 8) printf("\nlane %d element %d at (%d, %d), qmti8s assumes (%d, %d)\n", lane, i, c, r,
                              c0 + dc * (i & 3), r0 + dr * (i >> 2));
      }
      if (c < 0 || c >= 16 || r < 0 || r >= 32) {
        if (bad++ < 8) printf("lane %d element %d at (%d, %d), outside the 16 x 32 tile\n", lane, i, c, r);
      } else {
        hits[r * 16 + c]++;
      }
    }
    if (verbose) printf("\n");
  }
  if (cap == 16)
    for (int k = 0; k < 32 * 16; ++k)
      if (hits[k] != 1 && bad++ < 8) printf("column %d row %d held by %d elements\n", k % 16, k / 16, hits[k]);
  if (cap == 16) printf("lane 0: c0 %d r0 %d dc %d dr %d\n", col(0, 0), row(0, 0), col(0, 1) - col(0, 0), row(0, 4) - row(0, 0));
  printf(bad ? "qmti8s layout assumption FAILS (%d mismatches): keep NI_QMT_I8 off on this GPU\n"
             : "qmti8s layout assumption holds\n",
         bad);
  return bad ? 1 : 0;
}
