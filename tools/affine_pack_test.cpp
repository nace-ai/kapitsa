// The loader's pure helpers for MLX affine packs (src/model/affine_pack.h), on tiny synthetic tensors:
//  - affine_bits: the width from a tensor's shapes (3/4/5/6/8 bits, group 64), the real MTPLX 6-bit sidecar's shapes
//    (mtp.layers.0.self_attn.q_proj [12288, 960] words / [12288, 80] groups, mlp.down_proj 3264 / 272), shapes that fit
//    no whole width -> 0;
//  - to_bf16: float16 / float32 -> bf16 counts the values that change. Every float16 value (subnormals, the maximum)
//    stays finite and within bf16's rounding error; a bf16 number that float16 holds exactly comes back unchanged (an
//    MTPLX pack exported from bf16 weights casts with 0 inexact values); a float16 with more than 8 significant bits
//    is counted.
//   affine_pack_test      (no model, no GPU)
#include "model/affine_pack.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    ++g_fail;
    if (g_fail <= 20) printf("  FAILED: %s\n", what);
  }
}
_Float16 f16_from_bits(uint16_t b) {
  _Float16 h;
  memcpy(&h, &b, 2);
  return h;
}
}  // namespace

int main() {
  using namespace ni;
  // widths: words = K * bits / 32 per row, groups = K / 64
  for (int bits : {2, 3, 4, 5, 6, 8})
    for (int64_t K : {64, 128, 5120, 17408})
      if (K * bits % 32 == 0) check(affine_bits(K * bits / 32, K / 64, 64) == bits, "width from shapes");
  check(affine_bits(960, 80, 64) == 6, "q_proj of the MTPLX 6-bit sidecar");
  check(affine_bits(3264, 272, 64) == 6, "down_proj of the MTPLX 6-bit sidecar");
  check(affine_bits(640, 80, 64) == 4 && affine_bits(1280, 80, 64) == 8, "4- and 8-bit rows of the same K");
  check(affine_bits(961, 80, 64) == 0, "words that fit no width");
  check(affine_bits(0, 80, 64) == 0 && affine_bits(960, 0, 64) == 0 && affine_bits(960, 80, 0) == 0, "empty shapes");
  check(affine_bits(8000, 80, 64) == 0, "more than 8 bits");

  // float16 -> bf16, every bit pattern
  size_t finite = 0, inexact = 0;
  for (uint32_t b = 0; b < 65536; ++b) {
    const _Float16 h = f16_from_bits(uint16_t(b));
    const float v = float(h);
    if (std::isnan(v)) continue;
    uint16_t out = 0;
    const size_t lossy = to_bf16(&h, &out, 1);
    const float back = bf16_to_f32(out);
    if (std::isinf(v)) {
      check(back == v && !lossy, "inf stays inf");
      continue;
    }
    ++finite;
    inexact += lossy;
    check(std::isfinite(back), "finite float16 stays finite in bf16");
    check((back == v) == (lossy == 0), "lossy counts exactly the changed values");
    // bf16: 8 significant bits, so a relative error of at most 2^-8 (the subnormal float16s are normal bf16 numbers)
    check(std::fabs(back - v) <= std::fabs(v) * 0.00390625f, "within bf16's rounding error");
    if (v != 0.0f) check(back != 0.0f, "no value flushes to zero");
  }
  check(finite == 63488, "all 63488 finite float16 patterns were visited");
  check(inexact > 0, "float16 has values bf16 cannot hold");
  {
    const _Float16 h[3] = {f16_from_bits(0x0001), f16_from_bits(0x7bff), f16_from_bits(0x3c01)};  // 2^-24, 65504, 1 + 2^-10
    uint16_t out[3];
    check(to_bf16(h, out, 3) == 2, "min subnormal is exact; 65504 and 1 + 2^-10 are not");
    check(bf16_to_f32(out[0]) == std::ldexp(1.0f, -24), "2^-24 survives");
    check(bf16_to_f32(out[1]) == 65536.0f, "65504 rounds to 65536");
    check(bf16_to_f32(out[2]) == 1.0f, "1 + 2^-10 rounds to 1");
  }
  // a pack exported from bf16 weights: every bf16 number that float16 holds exactly casts with no loss
  size_t from_bf16 = 0;
  for (uint32_t b = 0; b < 65536; ++b) {
    const float v = bf16_to_f32(uint16_t(b));
    if (!std::isfinite(v)) continue;
    const _Float16 h = _Float16(v);
    if (float(h) != v) continue;  // out of float16's range or precision: such a pack was not exported through float16
    ++from_bf16;
    uint16_t out = 0;
    check(to_bf16(&h, &out, 1) == 0 && out == uint16_t(b), "a bf16 value that float16 holds round-trips");
  }
  check(from_bf16 > 7000, "many bf16 values pass through float16 exactly");
  // float32
  {
    const float f[3] = {1.0f, 1.0f + 0x1p-10f, 3.0e-39f};
    uint16_t out[3];
    check(to_bf16(f, out, 3) == 2 && out[0] == 0x3f80, "float32: exact, inexact, inexact");
  }
  if (g_fail) {
    printf("affine_pack_test: %d failures\n", g_fail);
    return 1;
  }
  printf("affine_pack_test: ALL PASSED (%zu finite float16 values, %zu not bf16 numbers; %zu bf16 values via float16)\n",
         finite, inexact, from_bf16);
  return 0;
}
