// A short prefill pass's routed experts (qmms_moe3_*, NI_SHORTPF_EXACT) must give a row the same bits whatever number
// of rows share its expert: a tile of 1-2 rows runs the FMA form, a larger one an MMA that reproduces it. Checked on
// synthetic 4-bit group-64 experts of the 35B's shapes (hidden 2048, intermediate 512), on the GPU at hand:
//   * every row of a tile of R = 1..8 rows (the MMA path from 3) against the same row alone in a 1-row tile (the FMA
//     path), gate/up+SwiGLU (half output) and down (fp32 output), bit for bit;
//   * the down outputs against a CPU reference of the FMA form (fp32 FMAs of half products in each 16-code slot's
//     order, the four slots added as (q0 + q1) + (q2 + q3)), bit for bit.
// The MMA side rests on simdgroup_multiply_accumulate adding its 8 products to C one at a time in k order (tools/mma_probe)
// and on the slot sums keeping their association under fast math (qc_sum). Exit status 1 on any difference.
//   moe_exact_check
#include "metal/gpu.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace ni;

namespace {

struct Experts {
  uint32_t N, K;
  Buffer W[2], S[2], B[2];  // gate/up (down: [0] only), one expert
};

Experts make(Gpu& g, std::mt19937& rng, uint32_t N, uint32_t K) {
  std::normal_distribution<float> nd(0, 1);
  Experts x{N, K, {}, {}, {}};
  for (int m = 0; m < 2; ++m) {
    x.W[m] = g.alloc(size_t(N) * K / 8 * 4);
    x.S[m] = g.alloc(size_t(N) * K / 64 * 2);
    x.B[m] = g.alloc(size_t(N) * K / 64 * 2);
    for (size_t i = 0; i < size_t(N) * K / 8; ++i) x.W[m].as<uint32_t>()[i] = uint32_t(rng());
    for (size_t i = 0; i < size_t(N) * K / 64; ++i) {
      x.S[m].as<uint16_t>()[i] = f32_to_bf16(0.01f * (1 + std::fabs(nd(rng))));
      x.B[m].as<uint16_t>()[i] = f32_to_bf16(-0.08f * (1 + 0.1f * nd(rng)));
    }
  }
  return x;
}

// out rows [0, R): the R rows of X as one tile (one = false) or as R tiles of one row
void run(Gpu& g, const Experts& x, bool down, const Buffer& X, uint32_t R, bool one, const Buffer& out) {
  Buffer rows = g.alloc(64 * 4), tiles = g.alloc(8 * 3 * 4), nt = g.alloc(16);
  for (uint32_t i = 0; i < R; ++i) rows.as<uint32_t>()[i] = i;
  const uint32_t n = one ? R : 1;
  for (uint32_t t = 0; t < n; ++t) {
    tiles.as<uint32_t>()[t * 3 + 0] = 0;
    tiles.as<uint32_t>()[t * 3 + 1] = one ? t : 0;
    tiles.as<uint32_t>()[t * 3 + 2] = one ? t + 1 : R;
  }
  nt.as<uint32_t>()[0] = n;
  kp::Qmm2 p{x.N, x.K, x.K, x.N, 64, 0, 0, 1};  // expert 0; the shared expert (1) never appears
  auto e = g.begin();
  e->use(g.pipeline(down ? "qmms_moe3_dn_b4" : "qmms_moe3_gu_b4"));
  e->buf(0, X);
  if (down) {
    e->buf(1, x.W[0]); e->buf(2, x.S[0]); e->buf(3, x.B[0]); e->buf(4, out);
    e->val(5, p);
    e->buf(6, tiles); e->buf(7, nt); e->buf(8, rows);
    e->buf(9, x.W[0]); e->buf(10, x.S[0]); e->buf(11, x.B[0]);
  } else {
    e->buf(1, x.W[0]); e->buf(2, x.S[0]); e->buf(3, x.B[0]);
    e->buf(4, x.W[1]); e->buf(5, x.S[1]); e->buf(6, x.B[1]);
    e->buf(7, out);
    e->val(8, p);
    e->buf(9, tiles); e->buf(10, nt); e->buf(11, rows);
    e->buf(12, x.W[0]); e->buf(13, x.S[0]); e->buf(14, x.B[0]);
    e->buf(15, x.W[1]); e->buf(16, x.S[1]); e->buf(17, x.B[1]);
  }
  e->dispatch(x.N / 32, n, 1, 128);  // 4 simdgroups of 8 output columns per threadgroup
  e->commit_and_wait();
}

// the FMA form of qmms_moe2 / qmms_moe3 for output column n of row r (one matrix)
float reference(const Experts& x, const Buffer& X, uint32_t r, uint32_t n) {
  const uint32_t gpr = x.K / 64, wpr = x.K / 8;
  float a[4];
  for (int q = 0; q < 4; ++q) {
    float acc = 0.0f;
    for (uint32_t g = 0; g < gpr; ++g) {
      const __fp16 s = (__fp16)bf16_to_f32(x.S[0].as<uint16_t>()[n * gpr + g]);
      const __fp16 b = (__fp16)bf16_to_f32(x.B[0].as<uint16_t>()[n * gpr + g]);
      for (int f = 0; f < 8; ++f) {
        const uint32_t word = x.W[0].as<uint32_t>()[n * wpr + g * 8 + 2 * q + f / 4];
        for (int e = 0; e < 2; ++e) {  // codes f % 4 and f % 4 + 4 of the word
          const uint32_t code = (word >> (4 * (f % 4) + 16 * e)) & 0xF;
          const __fp16 w = (__fp16)(double(code) * double(s) + double(b));  // one rounding, as the half FMA
          const __fp16 v = X.as<__fp16>()[r * x.K + g * 64 + 16 * q + 8 * (f / 4) + f % 4 + 4 * e];
          acc = fmaf(float(w), float(v), acc);
        }
      }
    }
    a[q] = acc;
  }
  return (a[0] + a[1]) + (a[2] + a[3]);
}

}  // namespace

int main() {
  Gpu& g = Gpu::get();
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0, 1);
  long bad = 0;
  for (int down = 0; down < 2; ++down) {
    const uint32_t K = down ? 512 : 2048, N = down ? 2048 : 512, esz = down ? 4 : 2;
    const Experts x = make(g, rng, N, K);
    for (uint32_t R = 1; R <= 8; ++R) {
      Buffer X = g.alloc(size_t(R) * K * 2), tile = g.alloc(size_t(R) * N * esz), alone = g.alloc(size_t(R) * N * esz);
      for (size_t i = 0; i < size_t(R) * K; ++i) X.as<__fp16>()[i] = (__fp16)(0.5f * nd(rng));
      memset(tile.ptr(), 0xff, tile.size);
      memset(alone.ptr(), 0x7f, alone.size);
      run(g, x, down, X, R, false, tile);
      run(g, x, down, X, R, true, alone);
      const long diff_alone = memcmp(tile.ptr(), alone.ptr(), size_t(R) * N * esz) != 0;
      long diff_ref = 0;
      if (down)
        for (uint32_t r = 0; r < R; ++r)
          for (uint32_t n = 0; n < N; ++n) {
            const float v = reference(x, X, r, n);
            diff_ref += memcmp(&v, tile.as<float>() + r * N + n, 4) != 0;
          }
      printf("%s, %u rows: tile vs rows alone %s", down ? "down" : "gate/up", R, diff_alone ? "DIFFER" : "bitwise equal");
      if (down) printf(", vs the CPU FMA form: %ld of %u outputs differ", diff_ref, R * N);
      printf("\n");
      bad += diff_alone + diff_ref;
    }
  }
  printf(bad ? "FAIL\n" : "OK\n");
  return bad ? 1 : 0;
}
