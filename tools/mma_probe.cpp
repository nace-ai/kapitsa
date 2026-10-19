// Probe of simdgroup_multiply_accumulate (half x half + float -> float) rounding semantics.
// One simdgroup, 8x8 A (half), 8x8 B (half), 8x8 C (float) per trial; compares the GPU's C' with models.
#include "metal/gpu.h"
#include "util/log.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

using namespace ni;

static const char* kSrc = R"MSL(
[[kernel]] void mma_probe(const device half* A [[buffer(0)]], const device half* B [[buffer(1)]],
                          const device float* C [[buffer(2)]], device float* D [[buffer(3)]],
                          uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  simdgroup_half8x8 a, b;
  simdgroup_float8x8 c;
  simdgroup_load(a, A + tg * 64, 8);
  simdgroup_load(b, B + tg * 64, 8);
  simdgroup_load(c, C + tg * 64, 8);
  simdgroup_multiply_accumulate(c, a, b, c);
  simdgroup_store(c, D + tg * 64, 8);
}
// chain of 2 MMAs: C += A0 B0; C += A1 B1
[[kernel]] void mma_probe2(const device half* A [[buffer(0)]], const device half* B [[buffer(1)]],
                           const device float* C [[buffer(2)]], device float* D [[buffer(3)]],
                           uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  simdgroup_half8x8 a, b, a1, b1;
  simdgroup_float8x8 c;
  simdgroup_load(a, A + tg * 128, 8);
  simdgroup_load(b, B + tg * 128, 8);
  simdgroup_load(a1, A + tg * 128 + 64, 8);
  simdgroup_load(b1, B + tg * 128 + 64, 8);
  simdgroup_load(c, C + tg * 64, 8);
  simdgroup_multiply_accumulate(c, a, b, c);
  simdgroup_multiply_accumulate(c, a1, b1, c);
  simdgroup_store(c, D + tg * 64, 8);
}
)MSL";

static float h2f(uint16_t u) { __fp16 h; memcpy(&h, &u, 2); return float(h); }
static uint16_t f2h(float f) { __fp16 h = (__fp16)f; uint16_t u; memcpy(&u, &h, 2); return u; }

// exact sum via long double is not enough on arm64; use __int128 fixed point with scale 2^S
static const int S = 80;
static __int128 fx(double v) {  // v must be exactly representable as int * 2^-S
  int e; double m = frexp(v, &e);  // v = m 2^e, m in [0.5,1)
  // m * 2^53 integer
  long long mi = (long long)ldexp(m, 53);
  int sh = e - 53 + S;
  __int128 r = mi;
  if (sh >= 0) r <<= sh; else r >>= -sh;  // assume exact
  return r;
}
static float rnd_fx(__int128 v) {  // round v * 2^-S to float (RNE)
  if (v == 0) return 0.0f;
  bool neg = v < 0; unsigned __int128 a = neg ? -(unsigned __int128)v : v;
  int msb = 127; while (!((a >> msb) & 1)) --msb;
  int drop = msb - 23;
  unsigned __int128 m;
  if (drop > 0) {
    m = a >> drop;
    unsigned __int128 rem = a & ((((unsigned __int128)1) << drop) - 1), half = ((unsigned __int128)1) << (drop - 1);
    if (rem > half || (rem == half && (m & 1))) ++m;
  } else { m = a << -drop; drop = 0; }
  double r = ldexp((double)(unsigned long long)m, drop - S);
  return float(neg ? -r : r);
}

int main(int argc, char** argv) {
  const int N = argc > 1 ? atoi(argv[1]) : 256;
  const int erange = argc > 2 ? atoi(argv[2]) : 6;
  Gpu& g = Gpu::get();
  g.compile(kSrc);
  std::mt19937 rng(123);
  auto rh = [&]() {
    std::uniform_int_distribution<int> m(1024, 2047), e(-erange, erange), s(0, 1);
    float v = ldexp(float(m(rng)) / 1024.0f, e(rng)) * (s(rng) ? -1 : 1);
    return f2h(v);
  };
  auto rf = [&]() {
    std::uniform_int_distribution<int> m(1 << 23, (1 << 24) - 1), e(-2 * erange, 2 * erange), s(0, 1);
    return ldexp(float(m(rng)) / float(1 << 23), e(rng)) * (s(rng) ? -1 : 1);
  };
  for (int mode = 1; mode <= 2; ++mode) {
    const int nm = mode;
    Buffer A = g.alloc(size_t(N) * 64 * nm * 2), B = g.alloc(size_t(N) * 64 * nm * 2), C = g.alloc(size_t(N) * 64 * 4),
           D = g.alloc(size_t(N) * 64 * 4);
    for (size_t i = 0; i < size_t(N) * 64 * nm; ++i) A.as<uint16_t>()[i] = rh(), B.as<uint16_t>()[i] = rh();
    for (size_t i = 0; i < size_t(N) * 64; ++i) C.as<float>()[i] = rf();
    auto e = g.begin();
    e->use(g.pipeline(mode == 1 ? "mma_probe" : "mma_probe2"));
    e->buf(0, A); e->buf(1, B); e->buf(2, C); e->buf(3, D);
    e->dispatch(N, 1, 1, 32);
    e->commit_and_wait();
    // models
    const char* names[] = {"seq k0..7 from c", "seq k7..0 from c", "exact sum (single round)", "products sum seq then +c",
                           "pairwise tree then +c", "c + tree", "exact products-sum rounded, then +c"};
    const int NM = 7;
    std::vector<long> ok(NM, 0);
    long total = 0;
    for (int t = 0; t < N; ++t)
      for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) {
          float c = C.as<float>()[t * 64 + i * 8 + j];
          float got = D.as<float>()[t * 64 + i * 8 + j];
          float m[NM];
          for (int k = 0; k < NM; ++k) m[k] = c;
          for (int mm = 0; mm < nm; ++mm) {
            float p[8];
            for (int k = 0; k < 8; ++k)
              p[k] = h2f(A.as<uint16_t>()[(t * nm + mm) * 64 + i * 8 + k]) * h2f(B.as<uint16_t>()[(t * nm + mm) * 64 + k * 8 + j]);
            // 0: seq forward
            for (int k = 0; k < 8; ++k) m[0] = m[0] + p[k];
            for (int k = 7; k >= 0; --k) m[1] = m[1] + p[k];
            __int128 ex = fx(m[2]);
            for (int k = 0; k < 8; ++k) ex += fx(p[k]);
            m[2] = rnd_fx(ex);
            float s = 0; for (int k = 0; k < 8; ++k) s += p[k];
            m[3] = m[3] + s;
            float tr = ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]));
            m[4] = m[4] + tr;
            m[5] = m[5] + tr;
            __int128 ep = 0; for (int k = 0; k < 8; ++k) ep += fx(p[k]);
            m[6] = m[6] + rnd_fx(ep);
          }
          ++total;
          for (int k = 0; k < NM; ++k) ok[k] += memcmp(&m[k], &got, 4) == 0;
        }
    printf("mode %d (%d MMA chain): %ld outputs\n", mode, nm, total);
    for (int k = 0; k < NM; ++k) printf("  %-40s %ld / %ld\n", names[k], ok[k], total);
  }
  return 0;
}
