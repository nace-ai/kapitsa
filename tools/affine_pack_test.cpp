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
#include "model/affine_coeff_fixed.h"

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
  // Check the interleaved record address independently, including partial output blocks, multiple row groups,
  // compressed and raw coefficient headers. The raw tail is compared too: its header offsets are absolute.
  size_t interleaves = 0;
  for (uint32_t N : {1u, 17u, 64u, 128u, 129u, 256u})
    for (uint32_t G : {1u, 4u, 8u, 16u, 40u, 64u, 192u}) {
      std::vector<uint32_t> w(size_t(N) * G * 8);
      for (size_t i = 0; i < w.size(); ++i) w[i] = uint32_t(i * 2654435761u);
      std::vector<uint16_t> sc(size_t(N) * G, 0x3b80), bi(size_t(N) * G, 0xbd80);
      for (size_t i = 0; i < bi.size(); i += 97) bi[i] = 0x3f80;  // same sign forces raw coefficients
      const auto original = pack_affine_packed_coeff(w.data(), sc.data(), bi.data(), N, G);
      for (uint32_t ks : {1u, 2u, 4u, 8u}) for (uint32_t rg : {0u, 1u, 2u, 8u, 32u}) {
        auto got = original, expected = original;
        const size_t pairs = (G + 1) / 2, rbs = (N + 15) / 16;
        if (ks >= 2 && pairs % ks == 0 && rg && rbs % rg == 0)
          for (size_t dst = 0; dst < original.blocks; ++dst) {
            const size_t group = dst / (rg * pairs), within = dst % (rg * pairs);
            const size_t slice = within % ks, row = (within / ks) % rg, pair = within / (ks * rg);
            const size_t src = (group * rg + row) * pairs + slice * (pairs / ks) + pair;
            std::copy_n(original.words.data() + src * 272, 272, expected.words.data() + dst * 272);
            expected.headers[dst] = original.headers[src];
          }
        interleave_packed_slices(got, N, G, ks, rg);
        check(got.words == expected.words && got.headers == expected.headers, "packed interleave, records and raw tail");
        check(got.blocks == original.blocks && got.compressed_blocks == original.compressed_blocks,
              "interleave preserves codec counts");
        ++interleaves;
      }
    }
  printf("packed interleave: %zu cases\n", interleaves);
  check(affine_kernel_group_size(5, 128) == 64, "5-bit group 128 reuses group-64 kernels");
  check(affine_kernel_group_size(8, 128) == 64, "8-bit group 128 keeps its group-64 path");
  for (int bits : {3, 4, 6})
    check(affine_kernel_group_size(bits, 128) == 128, "other group-128 formats stay unsupported");
  for (int bits : {3, 4, 5, 6, 8})
    for (int gs : {32, 64}) check(affine_kernel_group_size(bits, gs) == gs, "other groups stay unchanged");
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

  check(affine_bits(8, int64_t(1) << 58, 64) == 0, "group product overflow");
  check(affine_bits(std::numeric_limits<int64_t>::max(), 1, 64) == 0, "packed word product overflow");
  check(affine_bits(8, -1, 64) == 0, "negative dimension");
  auto shape = [](std::vector<int64_t> w, std::vector<int64_t> sc, std::vector<int64_t> bi, int64_t gs = 64) {
    return affine_matrix_shape(w, sc, bi, gs);
  };
  for (int bits : {2, 3, 4, 5, 6, 8}) {
    check(shape({3, 2 * bits}, {3, 1}, {3, 1}), "affine matrix");
    check(shape({2, 3, 2 * bits}, {2, 3, 1}, {2, 3, 1}), "stacked affine experts");
  }
  check(!shape({}, {1, 1}, {1, 1}), "scalar weight");
  check(!shape({8}, {1, 1}, {1, 1}), "rank-one weight");
  check(!shape({1, 1, 1, 8}, {1, 1, 1, 1}, {1, 1, 1, 1}), "rank-four weight");
  check(!shape({2, 8}, {}, {2, 1}), "scalar scales");
  check(!shape({2, 8}, {2, 1}, {}), "scalar biases");
  check(!shape({2, 8}, {1, 1}, {2, 1}), "scale rows differ");
  check(!shape({2, 8}, {2, 1}, {2, 2}), "bias groups differ");
  check(!shape({2, 3, 8}, {1, 3, 1}, {1, 3, 1}), "expert count differs");
  check(!shape({0, 8}, {0, 1}, {0, 1}), "zero rows");
  check(!shape({2, 0}, {2, 0}, {2, 0}), "zero width");
  check(!shape({-1, 8}, {-1, 1}, {-1, 1}), "negative rows");
  check(!shape({2, 8}, {2, 1}, {2, 1}, 0), "zero group size");
  check(!shape({2, 9}, {2, 1}, {2, 1}), "fractional bit width");
  const int64_t u32 = std::numeric_limits<uint32_t>::max();
  check(!shape({u32 + 1, 8}, {u32 + 1, 1}, {u32 + 1, 1}), "rows exceed kernel dimension");
  check(!shape({2, (u32 / 64 + 1) * 8}, {2, u32 / 64 + 1}, {2, u32 / 64 + 1}), "input width exceeds kernel dimension");
  check(shape({2, u32 / 64 * 8}, {2, u32 / 64}, {2, u32 / 64}), "largest group-aligned kernel width");

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
