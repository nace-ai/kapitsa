// The dense Nemotron-H models' kernels (Nemotron Nano 4B / 9B v2: Mamba-2, attention and dense MLP layers): checks of
// the kernels they add, against fp64 references or bit for bit against their twins, then timings of the products they
// run at their shapes.
//   nh_mlp_bench [--check] [--time] [--reps n] [filter ...]   (neither --check nor --time: both)
// Checks (small sizes: they also run under MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1, except attn5, whose 17 KB of
// threadgroup memory validation doubles past 32 KB):
//   tail   nh_x4_v8[_l0]_t, nh_x4_mm16/24/32[_l0]_t and nh_xb_v8_t at K % 128 = 32, 64, 96 against fp64 products of the
//          kernels' operands; at K % 128 = 0 bit for bit against nh_x4_v8 / nh_x4_mm* / nh_xb_v8
//   relu2  nh_x4_mv_relu2 (layout L2) bit for bit against nh_x4_mv_l0_relu2 (layout L0), and fp64
//   ssm80  nh_ssm_h80 (a 21-row prefill chunk) and nh_ssm1 at head dim 80 (one decode row; a verify block that first
//          commits pending rows) against an fp64 scan
//   attn5  nh_attn_pf2_q1_h5_k32 bit for bit against nh_attn_prefill (40 query heads over 8 KV heads)
//   nh_mlp_bench --dump F: the outputs of the MoE models' and Dory's kernels whose source the dense models' changed
//          (nh_x4_v8 / nh_x4_mm16/24/32 in both layouts, nh_xb_v8, nh_x4_mv*) on seeded inputs, every byte into F: a
//          build before the change and one after must write the same file
// Timings: back-to-back calls in one command buffer over rotating weight copies (~200 MB per shape, so no call finds
// its weights in the system cache), median of --reps buffers; us per call and GB/s of weight bytes.
//   mv     one row: the Nano 4B's BF16 shapes (nh_mv_bf16_r1), the 9B v2's NVFP4 ones (nh_x4_mv in layouts L2 / L0),
//          its BF16 layers' and its LM head (BF16 and Q8_0: nh_mv_q8_r1)
//   v8     8 rows: nh_xb_v8[_t] at the 4B's shapes, nh_x4_v8[_t] at the 9B's
//   gemm   a 2048-row prefill chunk: nh_gemm_bf16w_m64 at the 4B's shapes, nh_gemm_fp4l2_m64 at the 9B's (TFLOPS)
// (filter: substrings of the case label)
#include "metal/gpu.h"
#include "model/nemotron_h.h"
#include "util/bf16.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace ni {
extern const char* const kNhKernelSource;
}
using namespace ni;

namespace {

uint64_t g_rng = 0x9E3779B97F4A7C15ull;
uint64_t rnd() {  // splitmix64
  uint64_t z = (g_rng += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
float nrand() {  // ~N(0, 1)
  float s = 0;
  for (int i = 0; i < 4; ++i) s += float(rnd() & 0xFFFF) / 65535.0f;
  return (s - 2.0f) * 1.7320508f;
}
float urand() { return float(rnd() & 0xFFFFFF) / 16777216.0f; }

double e4m3(uint8_t x) {
  if (x == 0x7F) return 0.0;
  const int e = (x >> 3) & 0xF, m = x & 7;
  return e == 0 ? std::ldexp(double(m), -9) : std::ldexp(1.0 + double(m) / 8.0, e - 7);
}
const double kE2M1[16] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6};

uint16_t f2bf(float f) {  // round to nearest even
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return uint16_t(u >> 16);
}
float half_round(float f) { return float(__fp16(f)); }

void parallel_for(size_t n, const std::function<void(size_t, size_t)>& f) {
  const size_t nt = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  for (size_t t = 0; t < nt; ++t) th.emplace_back([&, t] { f(n * t / nt, n * (t + 1) / nt); });
  for (auto& x : th) x.join();
}

// A weight matrix in the engine's repacked layout, `copies` times (rotation), and its values as the MMA kernels take
// them (w: NVFP4 exact; BF16 rounded to half, as nh_xb_v8 converts) when kept for a check
struct Mat {
  NhFmt fmt = NhFmt::F32;
  uint32_t N = 0, K = 0;
  float gs = 1.0f;
  bool l2 = false;
  Buffer q, s;
  size_t qb = 0, sb = 0;  // bytes of one copy
  int copies = 1;
  std::vector<float> w;
  size_t bytes() const { return qb + sb; }
};

void release(Mat& m) {
  Gpu::get().release(m.q);
  if (m.sb) Gpu::get().release(m.s);
  m.w.clear();
}

// NVFP4 rows as ModelOpt writes them (byte j of a 16-group: elements 2j | 2j + 1 << 4; E4M3 block scales), through
// the engine's repack (nh_repack_rows) into layout L0 or L2
Mat make_fp4(uint32_t N, uint32_t K, bool l2, int copies, bool keep) {
  Mat m;
  m.fmt = NhFmt::FP4;
  m.N = N, m.K = K, m.l2 = l2, m.copies = copies;
  m.gs = 0.0037f;
  m.qb = size_t(N) * K / 2;
  m.sb = size_t(N) * K / 16;
  std::vector<uint8_t> src(m.qb), sc(m.sb);
  for (size_t i = 0; i < src.size(); i += 8) {
    const uint64_t r = rnd();
    memcpy(src.data() + i, &r, 8);
  }
  for (size_t i = 0; i < sc.size(); ++i) {
    const uint64_t r = rnd();
    const int sel = int(r & 63);
    uint8_t b = sel == 0 ? uint8_t((r >> 8) & 7) : sel == 1 ? uint8_t(0x70 + ((r >> 8) % 15)) : uint8_t(0x20 + ((r >> 8) % 0x50));
    sc[i] = b == 0x7F ? 0x7E : b;
  }
  Gpu& g = Gpu::get();
  m.q = g.alloc(m.qb * size_t(copies), "mlp_q", false);
  m.s = g.alloc(m.sb * size_t(copies), "mlp_s", false);
  nh_repack_rows(NhFmt::FP4, K, src.data(), N, m.q.as<uint8_t>(), nullptr, l2);
  memcpy(m.s.as<uint8_t>(), sc.data(), m.sb);
  for (int c = 1; c < copies; ++c) {
    memcpy(m.q.as<uint8_t>() + size_t(c) * m.qb, m.q.as<uint8_t>(), m.qb);
    memcpy(m.s.as<uint8_t>() + size_t(c) * m.sb, m.s.as<uint8_t>(), m.sb);
  }
  if (keep) {
    m.w.resize(size_t(N) * K);
    for (size_t i = 0; i < m.w.size(); ++i) {
      const uint8_t code = (src[i / 2] >> (4 * (i & 1))) & 15;
      m.w[i] = float(kE2M1[code] * e4m3(sc[i / 16]));  // (gs applied to the products)
    }
  }
  return m;
}

Mat make_bf16(uint32_t N, uint32_t K, int copies, bool keep) {
  Mat m;
  m.fmt = NhFmt::BF16;
  m.N = N, m.K = K, m.copies = copies;
  m.qb = size_t(N) * K * 2;
  Gpu& g = Gpu::get();
  m.q = g.alloc(m.qb * size_t(copies), "mlp_bf16", false);
  m.s = m.q;
  uint16_t* p = m.q.as<uint16_t>();
  parallel_for(N, [&](size_t n0, size_t n1) {
    uint64_t st = 0x1234567ull + n0 * 0x9E3779B97F4A7C15ull;
    for (size_t i = n0 * K; i < n1 * K; ++i) {
      st = st * 6364136223846793005ull + 1442695040888963407ull;
      const float u = float((st >> 33) & 0xFFFFFF) / 16777216.0f - 0.5f;
      p[i] = f2bf(u * 0.08f);
    }
  });
  for (int c = 1; c < copies; ++c) memcpy(m.q.as<uint8_t>() + size_t(c) * m.qb, m.q.as<uint8_t>(), m.qb);
  if (keep) {
    m.w.resize(size_t(N) * K);
    for (size_t i = 0; i < m.w.size(); ++i) m.w[i] = half_round(bf16_to_f32(p[i]));
  }
  return m;
}

Mat make_q8(uint32_t N, uint32_t K, int copies) {
  Mat m;
  m.fmt = NhFmt::Q8;
  m.N = N, m.K = K, m.copies = copies;
  nh_row_bytes(NhFmt::Q8, K, m.qb, m.sb);
  m.qb *= N;
  m.sb *= N;
  Gpu& g = Gpu::get();
  m.q = g.alloc(m.qb * size_t(copies), "mlp_q8", false);
  m.s = g.alloc(m.sb * size_t(copies), "mlp_q8s", false);
  for (size_t i = 0; i < m.qb; i += 8) {
    const uint64_t r = rnd();
    memcpy(m.q.as<uint8_t>() + i, &r, std::min<size_t>(8, m.qb - i));
  }
  for (size_t i = 0; i < m.sb / 2; ++i) {
    const __fp16 h = __fp16(0.0005f + 0.001f * urand());
    memcpy(m.s.as<uint8_t>() + 2 * i, &h, 2);
  }
  for (int c = 1; c < copies; ++c) {
    memcpy(m.q.as<uint8_t>() + size_t(c) * m.qb, m.q.as<uint8_t>(), m.qb);
    memcpy(m.s.as<uint8_t>() + size_t(c) * m.sb, m.s.as<uint8_t>(), m.sb);
  }
  return m;
}

struct MvP {
  uint32_t N, K, xs, ys;
  float gs;
  uint32_t R;
};
struct V8P {
  uint32_t N, K, x_ld, y_ld, R;
  float gs;
  uint32_t epi;
};

// One product: kernel `name` over rows [0, R) of x (fp32 for the one-row kernels, half for the rest) into y
void enc_product(Encoder& e, const std::string& name, const Mat& m, int copy, const Buffer& x, const Buffer& y,
                 uint32_t R, uint32_t epi, const Buffer& part, const Buffer& cnt) {
  Gpu& g = Gpu::get();
  e.use(g.pipeline(name));
  const size_t qo = size_t(copy) * m.qb, so = m.sb ? size_t(copy) * m.sb : qo;
  if (name.rfind("nh_x4_mv", 0) == 0 || name.rfind("nh_mv_", 0) == 0) {
    const MvP p{m.N, m.K, m.K, m.N, m.gs, R};
    e.buf(0, m.q, qo);
    e.buf(1, m.s, so);
    e.buf(2, x);
    e.buf(3, y);
    e.bytes(4, &p, sizeof p);
    if (name.rfind("nh_x4_mv", 0) == 0) e.dispatch((m.N + 7) / 8, 1, 1, 64);
    else {
      const uint32_t rk = R <= 1 ? 1 : R <= 2 ? 2 : R <= 4 ? 4 : 8, nr = rk <= 2 ? 4 : rk == 4 ? 2 : 1;
      e.dispatch((m.N + nr * 4 - 1) / (nr * 4), 1, 1, 128);
    }
    return;
  }
  const V8P p{m.N, m.K, m.K, m.N, R, m.gs, epi};
  e.buf(0, x);
  e.buf(1, m.q, qo);
  e.buf(2, m.s, so);
  e.buf(3, y);
  e.bytes(4, &p, sizeof p);
  if (name.rfind("qrf_f16_", 0) == 0) {  // the base library's dense BF16 GEMMs (wfmt.metal, QmmParams; Y fp32)
    const struct {
      uint32_t M, N, K, x_ld, y_ld, add, gs, we, se;
    } qp{R, m.N, m.K, m.K, m.N, epi == 1 ? 1u : 0u, 64, 0, 0};
    e.buf(0, x);
    e.buf(1, m.q, qo);
    e.buf(2, m.q, qo);
    e.buf(3, m.q, qo);
    e.buf(4, y);
    e.bytes(5, &qp, sizeof qp);
    const bool mf = name.size() > 3 && name.compare(name.size() - 3, 3, "_mf") == 0;
    const uint32_t bm = name.find("_m32_") != std::string::npos ? 32 : name.find("_m64_") != std::string::npos ? 64 : 16,
                   bn = name.find("_n128") != std::string::npos ? 128 : 64;
    const uint32_t mt = (R + bm - 1) / bm, nt = (m.N + bn - 1) / bn;
    if (mf) e.dispatch(mt, nt, 1, 128);
    else e.dispatch(nt, mt, 1, 128);
    return;
  }
  if (name.rfind("nh_gemm_", 0) == 0) {  // the tile GEMMs (NhGemmP), R rows
    const uint32_t bm = name.size() > 3 && name.compare(name.size() - 3, 3, "m16") == 0 ? 16
                        : name.compare(name.size() - 3, 3, "m32") == 0                ? 32
                                                                                       : 64;
    const struct {
      uint32_t M, N, K, x_ld, y_ld;
      float gs;
      uint32_t epi, qe, se, k;
    } gp{R, m.N, m.K, m.K, m.N, m.gs, epi, 0, 0, 1};
    e.bytes(4, &gp, sizeof gp);
    e.buf(5, part);
    e.buf(6, part);
    e.buf(7, part);
    e.buf(8, part);
    e.dispatch((m.N + 63) / 64, (R + bm - 1) / bm, 1, 128);
    return;
  }
  if (name.rfind("nh_x4_mm", 0) == 0) {
    const uint32_t sk = m.K <= 8192 ? 4 : 8;
    e.buf(5, part);
    e.buf(6, cnt);
    e.dispatch(sk, (m.N + 63) / 64, 1, 128);
  } else {
    e.dispatch((m.N + 15) / 16, 1, 1, 256);
  }
}

int g_fail = 0;
void report(const std::string& what, bool ok, const std::string& detail) {
  printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", what.c_str(), detail.empty() ? "" : "  ", detail.c_str());
  if (!ok) ++g_fail;
}

bool want(const std::vector<std::string>& filters, const std::string& label) {
  if (filters.empty()) return true;
  for (auto& f : filters)
    if (label.find(f) != std::string::npos) return true;
  return false;
}

// ---- checks ----

// y (R rows, fp32 or half per epi) against the fp64 products of m's operands with x (half rows) times gs
void check_product(const std::string& name, const Mat& m, uint32_t R, uint32_t epi, Buffer& part, Buffer& cnt) {
  Gpu& g = Gpu::get();
  const bool mv = name.rfind("nh_x4_mv", 0) == 0, half_in = !mv;
  std::vector<float> xf(size_t(R) * m.K);
  for (auto& v : xf) v = half_round(nrand());
  Buffer x = g.alloc(xf.size() * (half_in ? 2 : 4), "mlp_x");
  for (size_t i = 0; i < xf.size(); ++i) {
    if (half_in) x.as<__fp16>()[i] = __fp16(xf[i]);
    else x.as<float>()[i] = xf[i];
  }
  const bool half_out = !mv && epi >= 2;  // (the one-row kernels store fp32, their epilogue in their name)
  Buffer y = g.alloc(size_t(R) * m.N * 4 + 64, "mlp_y");
  memset(y.ptr(), 0, y.size);
  auto e = g.begin();
  enc_product(*e, name, m, 0, x, y, R, epi, part, cnt);
  e->commit_and_wait();
  double err = 0, mx = 0;
  int bad_nan = 0;
  for (uint32_t t = 0; t < R; ++t)
    for (uint32_t n = 0; n < m.N; ++n) {
      double a = 0;
      const float* wr = m.w.data() + size_t(n) * m.K;
      const float* xr = xf.data() + size_t(t) * m.K;
      for (uint32_t k = 0; k < m.K; ++k) a += double(wr[k]) * double(xr[k]);
      a *= m.gs;
      if (epi == 2) a = a > 0 ? a * a : 0.0;
      const double v = half_out ? double(float(y.as<__fp16>()[size_t(t) * m.N + n])) : double(y.as<float>()[size_t(t) * m.N + n]);
      if (!std::isfinite(v)) ++bad_nan;
      err = std::max(err, std::fabs(v - a));
      mx = std::max(mx, std::fabs(a));
    }
  const double rel = mx > 0 ? err / mx : err;
  const double tol = half_out ? 4e-3 : 2e-5;
  char d[160];
  snprintf(d, sizeof d, "N %u K %u (K %% 128 = %u) R %u epi %u: rel.err %.2e%s", m.N, m.K, m.K % 128, R, epi, rel,
           bad_nan ? " NON-FINITE" : "");
  report(name, rel < tol && !bad_nan, d);
  g.release(x);
  g.release(y);
}

// two kernels on the same operands: outputs equal bit for bit
void check_twins(const std::string& a, const Mat& ma, const std::string& b, const Mat& mb, uint32_t R, uint32_t epi,
                 Buffer& part, Buffer& cnt) {
  Gpu& g = Gpu::get();
  const bool half_in = a.rfind("nh_x4_mv", 0) != 0;
  Buffer x = g.alloc(size_t(R) * ma.K * (half_in ? 2 : 4), "mlp_x");
  for (size_t i = 0; i < size_t(R) * ma.K; ++i) {
    // subnormal and tiny inputs too (layout L2's products must stay L0's)
    float v = nrand();
    if ((i & 63) == 0) v *= 1e-30f;
    if (half_in) x.as<__fp16>()[i] = __fp16(v);
    else x.as<float>()[i] = v;
  }
  Buffer ya = g.alloc(size_t(R) * ma.N * 4 + 64, "mlp_ya"), yb = g.alloc(size_t(R) * ma.N * 4 + 64, "mlp_yb");
  memset(ya.ptr(), 0, ya.size);
  memset(yb.ptr(), 0, yb.size);
  auto e = g.begin();
  enc_product(*e, a, ma, 0, x, ya, R, epi, part, cnt);
  enc_product(*e, b, mb, 0, x, yb, R, epi, part, cnt);
  e->commit_and_wait();
  const size_t nb = size_t(R) * ma.N * (epi >= 2 ? 2 : 4);
  size_t diff = 0;
  for (size_t i = 0; i < nb; ++i) diff += ya.as<uint8_t>()[i] != yb.as<uint8_t>()[i];
  char d[128];
  snprintf(d, sizeof d, "N %u K %u R %u epi %u: %zu of %zu bytes differ", ma.N, ma.K, R, epi, diff, nb);
  report(a + " == " + b, diff == 0, d);
  g.release(x);
  g.release(ya);
  g.release(yb);
}

void checks_tail(const std::vector<std::string>& f) {
  Gpu& g = Gpu::get();
  Buffer part = g.alloc(size_t(8) * 16 * 2048 * 4, "mlp_part"), cnt = g.alloc(4096, "mlp_cnt");
  memset(cnt.ptr(), 0, cnt.size);
  const uint32_t N = 160;  // 2.5 row tiles of 64, 10 of 16: partial tiles at both sizes
  for (uint32_t K : {1056u, 1088u, 1120u, 3136u}) {
    for (bool l2 : {true, false}) {
      Mat m = make_fp4(N, K, l2, 1, true);
      const std::string sfx = std::string(l2 ? "" : "_l0") + "_t";
      if (want(f, "tail+fp4"))
        for (uint32_t R : {1u, 3u, 8u})
          for (uint32_t epi : {0u, 2u}) check_product("nh_x4_v8" + sfx, m, R, epi, part, cnt);
      if (want(f, "tail+fp4"))
        for (auto [kn, R] : {std::pair<const char*, uint32_t>{"nh_x4_mm16", 9}, {"nh_x4_mm16", 16}, {"nh_x4_mm24", 17},
                             {"nh_x4_mm24", 24}, {"nh_x4_mm32", 25}, {"nh_x4_mm32", 32}})
          check_product(std::string(kn) + sfx, m, R, R == 16 ? 2u : 0u, part, cnt);
      release(m);
    }
    if (want(f, "tail+bf16")) {
      Mat b = make_bf16(N, K, 1, true);
      for (uint32_t R : {1u, 2u, 5u, 8u})
        for (uint32_t epi : {0u, 2u}) check_product("nh_xb_v8_t", b, R, epi, part, cnt);
      release(b);
    }
  }
  // K % 128 == 0: the tail kernels are their twins
  if (want(f, "tail+twin")) {
    for (bool l2 : {true, false}) {
      Mat m = make_fp4(N, 1024, l2, 1, false);
      const std::string l = l2 ? "" : "_l0";
      check_twins("nh_x4_v8" + l + "_t", m, "nh_x4_v8" + l, m, 8, 0, part, cnt);
      check_twins("nh_x4_mm32" + l + "_t", m, "nh_x4_mm32" + l, m, 30, 0, part, cnt);
      release(m);
    }
    Mat b = make_bf16(N, 1024, 1, false);
    check_twins("nh_xb_v8_t", b, "nh_xb_v8", b, 7, 0, part, cnt);
    release(b);
  }
  g.release(part);
  g.release(cnt);
}

void checks_relu2(const std::vector<std::string>& f) {
  if (!want(f, "relu2")) return;
  Gpu& g = Gpu::get();
  Buffer dummy = g.alloc(4096, "mlp_dummy");
  g_rng = 77;
  Mat a = make_fp4(1000, 4480, true, 1, true);
  g_rng = 77;
  Mat b = make_fp4(1000, 4480, false, 1, false);
  check_twins("nh_x4_mv_relu2", a, "nh_x4_mv_l0_relu2", b, 1, 0, dummy, dummy);
  check_twins("nh_x4_mv_add", a, "nh_x4_mv_l0_add", b, 1, 0, dummy, dummy);
  check_product("nh_x4_mv_relu2", a, 1, 2, dummy, dummy);
  release(a);
  release(b);
  g.release(dummy);
}

// ---- the Mamba-2 scan at head dim 80 ----
struct SsmP {
  uint32_t T, pos0, write, zx_stride, NH, HD, DS, NG;
};

double softplus_d(double x) { return x > 20 ? x : std::log1p(std::exp(x)); }
double silu_d(double x) { return x / (1.0 + std::exp(-x)); }

void checks_ssm(const std::vector<std::string>& f) {
  if (!want(f, "ssm80")) return;
  Gpu& g = Gpu::get();
  const uint32_t NH = 16, HD = 80, NG = 2, DS = 128, DI = NH * HD, CH = DI + 2 * NG * DS, IP = 2 * DI + 2 * NG * DS + NH;
  const int T = 21, pos0 = 37;
  // inputs: zx rows, the ring (positions pos0 - 3 .. pos0 - 1), conv, A (negative), D, dt bias, the state
  std::vector<float> zx(size_t(T) * IP), ring(4 * CH), cw(size_t(CH) * 4), cb(CH), A(NH), D(NH), dtb(NH),
      st0(size_t(NH) * HD * DS);
  for (auto& v : zx) v = nrand();
  for (auto& v : ring) v = nrand();
  for (auto& v : cw) v = 0.5f * nrand();
  for (auto& v : cb) v = 0.1f * nrand();
  for (uint32_t h = 0; h < NH; ++h) A[h] = -std::exp(0.5f * nrand()), D[h] = nrand(), dtb[h] = 0.5f * nrand() - 1.0f;
  for (auto& v : st0) v = 0.3f * nrand();
  // fp64 reference over rows [0, n) of zx: y per row, the state after each row
  auto ref = [&](int n, std::vector<double>& y, std::vector<double>& st) {
    st.assign(st0.begin(), st0.end());
    y.assign(size_t(n) * DI, 0.0);
    auto raw = [&](int t, uint32_t c) -> double {
      return t >= 0 ? zx[size_t(t) * IP + DI + c] : ring[((pos0 + t + 4) & 3) * CH + c];
    };
    auto conv = [&](int t, uint32_t c) {
      double v = cb[c];
      for (int j = 0; j < 4; ++j) v += double(cw[c * 4 + j]) * raw(t - 3 + j, c);
      return silu_d(v);
    };
    for (int t = 0; t < n; ++t)
      for (uint32_t h = 0; h < NH; ++h) {
        const uint32_t gg = h / (NH / NG);
        const double dt = softplus_d(double(zx[size_t(t) * IP + DI + CH + h]) + dtb[h]), dA = std::exp(dt * A[h]);
        std::vector<double> B(DS), C(DS);
        for (uint32_t s = 0; s < DS; ++s) B[s] = conv(t, DI + gg * DS + s), C[s] = conv(t, DI + NG * DS + gg * DS + s);
        for (uint32_t i = 0; i < HD; ++i) {
          const double xv = conv(t, h * HD + i);
          double acc = 0;
          for (uint32_t s = 0; s < DS; ++s) {
            double& sv = st[(size_t(h) * HD + i) * DS + s];
            sv = sv * dA + B[s] * xv * dt;
            acc += sv * C[s];
          }
          y[size_t(t) * DI + h * HD + i] = (acc + D[h] * xv) * silu_d(zx[size_t(t) * IP + h * HD + i]);
        }
      }
  };
  auto upload = [&](const std::vector<float>& v, const char* l) {
    Buffer b = g.alloc(std::max<size_t>(v.size() * 4, 16), l);
    memcpy(b.ptr(), v.data(), v.size() * 4);
    return b;
  };
  Buffer bzx = upload(zx, "zx"), bcw = upload(cw, "cw"), bcb = upload(cb, "cb"), bA = upload(A, "A"),
         bD = upload(D, "D"), bdt = upload(dtb, "dtb");
  Buffer live = g.alloc(16, "live");
  live.as<uint32_t>()[0] = 0xffffffffu;
  auto compare = [&](const std::string& what, const std::vector<double>& ry, const Buffer& y, int y0, int ny,
                     const std::vector<double>* rst, const Buffer* st) {
    double err = 0, mx = 0;
    for (int t = 0; t < ny; ++t)
      for (uint32_t c = 0; c < DI; ++c) {
        const double a = ry[size_t(y0 + t) * DI + c], v = y.as<float>()[size_t(t) * DI + c];
        err = std::max(err, std::isfinite(v) ? std::fabs(v - a) : 1e30);
        mx = std::max(mx, std::fabs(a));
      }
    double serr = 0, smx = 0;
    if (rst)
      for (size_t i = 0; i < rst->size(); ++i) {
        const double v = st->as<float>()[i];
        serr = std::max(serr, std::isfinite(v) ? std::fabs(v - (*rst)[i]) : 1e30);
        smx = std::max(smx, std::fabs((*rst)[i]));
      }
    char d[160];
    snprintf(d, sizeof d, "y rel.err %.2e, state rel.err %.2e", err / mx, rst ? serr / smx : 0.0);
    report(what, err / mx < 1e-4 && (!rst || serr / smx < 1e-4), d);
  };
  std::vector<double> ry, rst;
  // nh_ssm_h80: the 21-row chunk (3 staging blocks), state written
  {
    ref(T, ry, rst);
    Buffer bring = upload(ring, "ring"), bst = upload(st0, "st"), y = g.alloc(size_t(T) * DI * 4, "y");
    const SsmP p{uint32_t(T), uint32_t(pos0), 1, IP, NH, HD, DS, NG};
    auto e = g.begin();
    e->use(g.pipeline("nh_ssm_h80"));
    e->buf(0, bzx);
    e->buf(1, bring);
    e->buf(2, bst);
    e->buf(3, y);
    e->buf(4, bcw);
    e->buf(5, bcb);
    e->buf(6, bA);
    e->buf(7, bD);
    e->buf(8, bdt);
    e->bytes(9, &p, sizeof p);
    e->dispatch(NH, 1, 1, 4 * HD);
    e->commit_and_wait();
    compare("nh_ssm_h80 (21 rows)", ry, y, 0, T, &rst, &bst);
  }
  // nh_ssm1 at head dim 80: one decode row; then a verify block of 5 rows after 3 pending rows (state after them
  // written, the block's rows not); nh_ssm1r5 / nh_ssm1r10 (several rows per simdgroup) equal to it bit for bit
  for (auto [np, nt, write] : {std::tuple<int, int, int>{0, 1, 1}, {3, 5, 0}}) {
    std::vector<std::vector<uint8_t>> outs;  // per kernel: y, then the state, then the ring
    for (const char* kn : {"nh_ssm1", "nh_ssm1r5", "nh_ssm1r10"}) {
      Buffer bring = upload(ring, "ring"), bst = upload(st0, "st"), y = g.alloc(size_t(nt) * DI * 4, "y");
      memset(y.ptr(), 0, y.size);
      const SsmP p{uint32_t(nt), uint32_t(pos0 + np), uint32_t(write), IP, NH, HD, DS, NG};
      const uint32_t npu = uint32_t(np), rows_tg = std::string(kn) == "nh_ssm1" ? 8 : std::string(kn) == "nh_ssm1r5" ? 40 : 80;
      auto e = g.begin();
      e->use(g.pipeline(kn));
      e->buf(0, bzx, size_t(np) * IP * 4);
      e->buf(1, bring);
      e->buf(2, bst);
      e->buf(3, y);
      e->buf(4, bcw);
      e->buf(5, bcb);
      e->buf(6, bA);
      e->buf(7, bD);
      e->buf(8, bdt);
      e->bytes(9, &p, sizeof p);
      e->buf(10, live);
      e->buf(11, bzx);
      e->bytes(12, &npu, 4);
      e->dispatch(NH * HD / rows_tg, 1, 1, 256);
      e->commit_and_wait();
      std::vector<uint8_t> o(y.as<uint8_t>(), y.as<uint8_t>() + y.size);
      o.insert(o.end(), bst.as<uint8_t>(), bst.as<uint8_t>() + bst.size);
      o.insert(o.end(), bring.as<uint8_t>(), bring.as<uint8_t>() + bring.size);
      outs.push_back(o);
      g.release(bring);
      g.release(bst);
      g.release(y);
    }
    for (size_t k = 1; k < outs.size(); ++k) {
      size_t diff = 0;
      for (size_t i = 0; i < outs[0].size(); ++i) diff += outs[0][i] != outs[k][i];
      char d[96];
      snprintf(d, sizeof d, "%d pending + %d rows: %zu of %zu bytes differ", np, nt, diff, outs[0].size());
      report(std::string(k == 1 ? "nh_ssm1r5" : "nh_ssm1r10") + " == nh_ssm1 at head dim 80", diff == 0, d);
    }
  }
  for (auto [np, nt, write] : {std::tuple<int, int, int>{0, 1, 1}, {3, 5, 0}}) {
    ref(np + nt, ry, rst);
    std::vector<double> rst_np;
    if (np) {
      std::vector<double> y2;
      ref(np, y2, rst_np);
    }
    Buffer bring = upload(ring, "ring"), bst = upload(st0, "st"), y = g.alloc(size_t(nt) * DI * 4, "y");
    // the pass's rows: zx rows np.., the pending ones zx rows 0..np - 1, at positions pos0 - np ..
    const SsmP p{uint32_t(nt), uint32_t(pos0 + np), uint32_t(write), IP, NH, HD, DS, NG};
    const uint32_t npu = uint32_t(np);
    auto e = g.begin();
    e->use(g.pipeline("nh_ssm1"));
    e->buf(0, bzx, size_t(np) * IP * 4);
    e->buf(1, bring);
    e->buf(2, bst);
    e->buf(3, y);
    e->buf(4, bcw);
    e->buf(5, bcb);
    e->buf(6, bA);
    e->buf(7, bD);
    e->buf(8, bdt);
    e->bytes(9, &p, sizeof p);
    e->buf(10, live);
    e->buf(11, bzx);
    e->bytes(12, &npu, 4);
    e->dispatch(NH * HD / 8, 1, 1, 256);
    e->commit_and_wait();
    const std::string what = "nh_ssm1 at head dim 80 (" + std::to_string(np) + " pending + " + std::to_string(nt) + " rows)";
    compare(what, ry, y, np, nt, np ? &rst_np : &rst, &bst);
    g.release(bring);
    g.release(bst);
    g.release(y);
  }
}

// ---- prefill attention with 5 query heads per KV head ----
void checks_attn(const std::vector<std::string>& f) {
  if (!want(f, "attn5")) return;
  Gpu& g = Gpu::get();
  const uint32_t Hq = 40, Hkv = 8, HD = 128, T = 77, pos0 = 45, cap = 160;
  Buffer q = g.alloc(size_t(T) * Hq * HD * 4, "q"), kc = g.alloc(size_t(Hkv) * cap * HD * 2, "kc"),
         vc = g.alloc(size_t(Hkv) * cap * HD * 2, "vc"), o1 = g.alloc(size_t(T) * Hq * HD * 2, "o1"),
         o2 = g.alloc(size_t(T) * Hq * HD * 2, "o2");
  for (size_t i = 0; i < size_t(T) * Hq * HD; ++i) q.as<float>()[i] = nrand();
  for (size_t i = 0; i < size_t(Hkv) * cap * HD; ++i) kc.as<__fp16>()[i] = __fp16(nrand()), vc.as<__fp16>()[i] = __fp16(nrand());
  memset(o1.ptr(), 0, o1.size);
  memset(o2.ptr(), 0, o2.size);
  struct {
    uint32_t T, pos0, cap, Hq, Hkv;
    float scale;
  } p1{T, pos0, cap, Hq, Hkv, 1.0f / std::sqrt(float(HD))};
  struct {
    uint32_t T, pos0, cap, Hq, Hkv;
    float scale;
    uint32_t qb0;
  } p2{T, pos0, cap, Hq, Hkv, 1.0f / std::sqrt(float(HD)), 0};
  auto e = g.begin();
  e->use(g.pipeline("nh_attn_prefill"));
  e->buf(0, q);
  e->buf(1, kc);
  e->buf(2, vc);
  e->buf(3, o1);
  e->bytes(4, &p1, sizeof p1);
  e->dispatch((T + 31) / 32, Hq, 1, 128);
  // in two pieces (as a long pass), row blocks [0, 4) and [4, 10)
  e->use(g.pipeline("nh_attn_pf2_q1_h5_k32"));
  e->buf(0, q);
  e->buf(1, kc);
  e->buf(2, vc);
  e->buf(3, o2);
  e->bytes(4, &p2, sizeof p2);
  e->dispatch(4, Hq / 5, 1, 160);
  p2.qb0 = 4;
  e->bytes(4, &p2, sizeof p2);
  e->dispatch((T + 7) / 8 - 4, Hq / 5, 1, 160);
  e->commit_and_wait();
  size_t diff = 0;
  for (size_t i = 0; i < size_t(T) * Hq * HD; ++i) diff += o1.as<uint16_t>()[i] != o2.as<uint16_t>()[i];
  char d[96];
  snprintf(d, sizeof d, "T %u after %u positions: %zu of %zu values differ", T, pos0, diff, size_t(T) * Hq * HD);
  report("nh_attn_pf2_q1_h5_k32 == nh_attn_prefill", diff == 0, d);
  for (Buffer* b : {&q, &kc, &vc, &o1, &o2}) g.release(*b);
}

// ---- the existing kernels' outputs, for a byte comparison between builds ----
void dump(const std::string& path) {
  Gpu& g = Gpu::get();
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) fatal("cannot write %s", path.c_str());
  Buffer part = g.alloc(size_t(8) * 64 * 2048 * 4, "mlp_part"), cnt = g.alloc(4096, "mlp_cnt");
  memset(cnt.ptr(), 0, cnt.size);
  size_t n = 0;
  auto run = [&](const std::string& name, const Mat& m, uint32_t R, uint32_t epi) {
    const bool mv = name.rfind("nh_x4_mv", 0) == 0;
    Buffer x = g.alloc(size_t(R) * m.K * 4, "mlp_x"), y = g.alloc(size_t(R) * m.N * 4 + 64, "mlp_y");
    for (size_t i = 0; i < size_t(R) * m.K; ++i) {
      float v = nrand();
      if ((i & 63) == 0) v *= 1e-30f;
      if (mv) x.as<float>()[i] = v;
      else x.as<__fp16>()[i] = __fp16(v);
    }
    for (size_t i = 0; i < size_t(R) * m.N; ++i) y.as<float>()[i] = 0.25f * float(i % 7);  // (the add epilogues)
    auto e = g.begin();
    enc_product(*e, name, m, 0, x, y, R, epi, part, cnt);
    e->commit_and_wait();
    fwrite(y.ptr(), 1, size_t(R) * m.N * 4, f);
    ++n;
    g.release(x);
    g.release(y);
  };
  for (auto [N, K] : {std::pair<uint32_t, uint32_t>{1000, 2048}, {4096, 4096}, {2688, 1856 * 2}, {1024, 12288}})
    for (bool l2 : {true, false}) {
      g_rng = 1000 + N + K + l2;
      Mat m = make_fp4(N, K, l2, 1, false);
      const std::string l = l2 ? "" : "_l0";
      for (uint32_t R : {1u, 2u, 5u, 8u})
        for (uint32_t epi : {0u, 1u, 2u, 3u}) run("nh_x4_v8" + l, m, R, epi);
      for (auto [kn, R] : {std::pair<const char*, uint32_t>{"nh_x4_mm16", 9}, {"nh_x4_mm16", 16}, {"nh_x4_mm24", 20},
                           {"nh_x4_mm32", 25}, {"nh_x4_mm32", 32}})
        for (uint32_t epi : {0u, 1u, 2u}) run(std::string(kn) + l, m, R, epi);
      for (const char* e : {"", "_add"}) run("nh_x4_mv" + l + e, m, 1, 0);
      if (!l2) run("nh_x4_mv_l0_relu2", m, 1, 0);
      release(m);
    }
  for (auto [N, K] : {std::pair<uint32_t, uint32_t>{4096, 2688}, {256, 2688}, {2688, 4096}}) {
    g_rng = 7 + N + K;
    Mat m = make_bf16(N, K, 1, false);
    for (uint32_t R : {1u, 3u, 8u})
      for (uint32_t epi : {0u, 1u, 2u, 3u}) run("nh_xb_v8", m, R, epi);
    release(m);
  }
  fclose(f);
  g.release(part);
  g.release(cnt);
  printf("%zu outputs written to %s\n", n, path.c_str());
}

// ---- timings ----
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

void time_case(const std::string& label, const std::string& name, const Mat& m, uint32_t R, uint32_t epi, int reps,
               Buffer& part, Buffer& cnt) {
  Gpu& g = Gpu::get();
  const bool half_in = name.rfind("nh_x4_mv", 0) != 0 && name.rfind("nh_mv_", 0) != 0;
  Buffer x = g.alloc(size_t(R) * m.K * 4, "mlp_x"), y = g.alloc(size_t(R) * m.N * 4 + 64, "mlp_y");
  for (size_t i = 0; i < size_t(R) * m.K; ++i) {
    if (half_in) x.as<__fp16>()[i] = __fp16(nrand());
    else x.as<float>()[i] = nrand();
  }
  const int calls = R > 32 ? 4 : std::max(10, m.copies * 2);
  std::vector<double> t;
  for (int r = 0; r < reps + 1; ++r) {
    auto e = g.begin();
    for (int c = 0; c < calls; ++c) enc_product(*e, name, m, c % m.copies, x, y, R, epi, part, cnt);
    e->commit_and_wait();
    if (r) t.push_back(e->gpu_ms() * 1000.0 / calls);  // (the first buffer warms up)
  }
  const double us = median(t);
  if (R > 32)
    printf("%-34s %-22s R %u  %8.1f us  %6.2f TFLOPS\n", label.c_str(), name.c_str(), R, us,
           2.0 * R * m.N * m.K / (us * 1e6));
  else
    printf("%-34s %-22s R %u  %8.1f us  %6.1f GB/s\n", label.c_str(), name.c_str(), R, us, m.bytes() / (us * 1e3));
  fflush(stdout);
  g.release(x);
  g.release(y);
}

int copies_for(size_t bytes) { return int(std::max<size_t>(1, std::min<size_t>(16, (size_t(200) << 20) / bytes + 1))); }

void timings(const std::vector<std::string>& f, int reps) {
  Gpu& g = Gpu::get();
  Buffer part = g.alloc(size_t(8) * 512 * 2048 * 4, "mlp_part"), cnt = g.alloc(4096, "mlp_cnt");
  memset(cnt.ptr(), 0, cnt.size);
  struct S {
    const char* label;
    uint32_t N, K;
    uint32_t epi;
  };
  // Nano 4B (BF16): every matrix
  const S b4[] = {{"4B in_proj", 17504, 3136, 0},  {"4B out_proj", 3136, 7680, 1}, {"4B mlp up", 12544, 3136, 2},
                  {"4B mlp down", 3136, 12544, 1}, {"4B q", 5120, 3136, 0},       {"4B k|v", 1024, 3136, 0},
                  {"4B o", 3136, 5120, 1},         {"4B head", 131072, 3136, 0}};
  // Nano 9B v2: NVFP4 layers, BF16 ones (attention, layers 0 / 1 / 54 / 55), head
  const S n9[] = {{"9B in_proj", 22656, 4480, 0}, {"9B out_proj", 4480, 10240, 1}, {"9B mlp up", 15680, 4480, 2},
                  {"9B mlp down", 4480, 15680, 1}};
  const S b9[] = {{"9B bf16 in_proj", 22656, 4480, 0}, {"9B bf16 mlp up", 15680, 4480, 2}, {"9B bf16 mlp down", 4480, 15680, 1},
                  {"9B q", 5120, 4480, 0},             {"9B k|v", 1024, 4480, 0},          {"9B o", 4480, 5120, 1},
                  {"9B head bf16", 131072, 4480, 0}};
  for (const S& s : b4) {
    const std::string l = std::string(s.label) + " mv";
    if (!want(f, l + "+bf16")) continue;
    Mat m = make_bf16(s.N, s.K, copies_for(size_t(s.N) * s.K * 2), false);
    const char* ep = s.epi == 1 ? "_add" : s.epi == 2 ? "_relu2" : "";
    time_case(l, std::string("nh_mv_bf16_r1") + ep, m, 1, 0, reps, part, cnt);
    release(m);
  }
  for (const S& s : b9) {
    const std::string l = std::string(s.label) + " mv";
    if (!want(f, l + "+bf16")) continue;
    Mat m = make_bf16(s.N, s.K, copies_for(size_t(s.N) * s.K * 2), false);
    time_case(l, std::string("nh_mv_bf16_r1") + (s.epi == 1 ? "_add" : s.epi == 2 ? "_relu2" : ""), m, 1, 0, reps, part, cnt);
    release(m);
  }
  if (want(f, "9B head q8 mv")) {
    Mat m = make_q8(131072, 4480, 1);
    time_case("9B head q8 mv", "nh_mv_q8_r1", m, 1, 0, reps, part, cnt);
    release(m);
  }
  for (const S& s : n9)
    for (bool l2 : {true, false}) {
      const std::string l = std::string(s.label) + (l2 ? " mv L2" : " mv L0");
      if (!want(f, l + "+fp4")) continue;
      Mat m = make_fp4(s.N, s.K, l2, copies_for(size_t(s.N) * s.K * 9 / 16), false);
      const std::string base = l2 ? "nh_x4_mv" : "nh_x4_mv_l0";
      time_case(l, base + (s.epi == 1 ? "_add" : s.epi == 2 ? "_relu2" : ""), m, 1, 0, reps, part, cnt);
      release(m);
    }
  for (const S& s : b4) {
    const std::string l = std::string(s.label) + " v8";
    if (!want(f, l + "+bf16")) continue;
    Mat m = make_bf16(s.N, s.K, copies_for(size_t(s.N) * s.K * 2), false);
    time_case(l, s.K % 128 ? "nh_xb_v8_t" : "nh_xb_v8", m, 8, s.epi == 2 ? 2 : 0, reps, part, cnt);
    release(m);
  }
  for (const S& s : n9) {
    const std::string l = std::string(s.label) + " v8";
    if (!want(f, l + "+fp4")) continue;
    Mat m = make_fp4(s.N, s.K, true, copies_for(size_t(s.N) * s.K * 9 / 16), false);
    time_case(l, s.K % 128 ? "nh_x4_v8_t" : "nh_x4_v8", m, 8, s.epi == 2 ? 2 : 0, reps, part, cnt);
    release(m);
  }
  for (const S& s : b4) {
    const std::string l = std::string(s.label) + " gemm";
    if (!want(f, l + "+bf16") || s.N > 32768) continue;
    Mat m = make_bf16(s.N, s.K, 1, false);
    time_case(l, "nh_gemm_bf16w_m64", m, 2048, s.epi == 2 ? 2 : 0, reps, part, cnt);
    time_case(l, "qrf_f16_m32_n128_mf", m, 2048, 0, reps, part, cnt);
    release(m);
  }
  for (const S& s : n9) {
    const std::string l = std::string(s.label) + " gemm";
    if (!want(f, l + "+fp4")) continue;
    for (bool l2 : {true, false}) {
      Mat m = make_fp4(s.N, s.K, l2, 1, false);
      time_case(l, l2 ? "nh_gemm_fp4l2_m64" : "nh_gemm_fp4_m64", m, 2048, s.epi == 2 ? 2 : 0, reps, part, cnt);
      release(m);
    }
  }
  // the decode scan (one row, state written) at the Nano 4B's (96 heads) and 9B v2's (128) sizes
  for (uint32_t NH : {96u, 128u}) {
    const std::string l = std::string(NH == 96 ? "4B" : "9B") + " ssm decode";
    if (!want(f, l)) continue;
    const uint32_t HD = 80, NG = 8, DS = 128, DI = NH * HD, CH = DI + 2 * NG * DS, IP = 2 * DI + 2 * NG * DS + NH;
    const int copies = 8;  // the states of 8 layers, rotating (no state found in the cache)
    Buffer zx = g.alloc(size_t(IP) * 4, "zx"), ring = g.alloc(size_t(4) * CH * 4 * copies, "ring"),
           st = g.alloc(size_t(NH) * HD * DS * 4 * copies, "st"), y = g.alloc(size_t(DI) * 4, "y"),
           cw = g.alloc(size_t(CH) * 16, "cw"), cb = g.alloc(size_t(CH) * 4, "cb"), A = g.alloc(NH * 4, "A"),
           D = g.alloc(NH * 4, "D"), dtb = g.alloc(NH * 4, "dtb"), live = g.alloc(16, "live");
    for (Buffer* b : {&zx, &ring, &st, &cw, &cb, &D, &dtb})
      for (size_t i = 0; i < b->size / 4; ++i) b->as<float>()[i] = 0.1f * nrand();
    for (uint32_t h = 0; h < NH; ++h) A.as<float>()[h] = -1.0f;
    live.as<uint32_t>()[0] = 0xffffffffu;
    for (const char* kn : {"nh_ssm1", "nh_ssm1r5", "nh_ssm1r10"}) {
      const uint32_t rows_tg = std::string(kn) == "nh_ssm1" ? 8 : std::string(kn) == "nh_ssm1r5" ? 40 : 80;
      const SsmP p{1, 7, 1, IP, NH, HD, DS, NG};
      const uint32_t np = 0;
      std::vector<double> t;
      for (int r = 0; r < reps + 1; ++r) {
        auto e = g.begin();
        for (int c = 0; c < 40; ++c) {
          e->use(g.pipeline(kn));
          e->buf(0, zx);
          e->buf(1, ring, size_t(c % copies) * 4 * CH * 4);
          e->buf(2, st, size_t(c % copies) * NH * HD * DS * 4);
          e->buf(3, y);
          e->buf(4, cw);
          e->buf(5, cb);
          e->buf(6, A);
          e->buf(7, D);
          e->buf(8, dtb);
          e->bytes(9, &p, sizeof p);
          e->buf(10, live);
          e->buf(11, zx);
          e->bytes(12, &np, 4);
          e->dispatch(NH * HD / rows_tg, 1, 1, 256);
        }
        e->commit_and_wait();
        if (r) t.push_back(e->gpu_ms() * 1000.0 / 40);
      }
      const double us = median(t);
      printf("%-34s %-22s       %8.1f us  %6.1f GB/s (the state read and written)\n", l.c_str(), kn, us,
             2.0 * NH * HD * DS * 4 / (us * 1e3));
    }
    for (Buffer* b : {&zx, &ring, &st, &y, &cw, &cb, &A, &D, &dtb, &live}) g.release(*b);
  }
  g.release(part);
  g.release(cnt);
}

}  // namespace

int main(int argc, char** argv) {
  bool check = false, timing = false;
  int reps = 5;
  std::string dump_file;
  std::vector<std::string> filters;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--check") check = true;
    else if (a == "--time") timing = true;
    else if (a == "--reps" && i + 1 < argc) reps = std::max(1, atoi(argv[++i]));
    else if (a == "--dump" && i + 1 < argc) dump_file = argv[++i];
    else filters.push_back(a);
  }
  if (!check && !timing) check = timing = true;
  Gpu& g = Gpu::get();
  g.compile_extra("Nemotron-H", kNhKernelSource);
  printf("GPU: %s, %d cores, Apple family %d\n", g.name().c_str(), g.core_count(), g.apple_family());
  if (!dump_file.empty()) {
    dump(dump_file);
    return 0;
  }
  if (check) {
    checks_tail(filters);
    checks_relu2(filters);
    checks_ssm(filters);
    checks_attn(filters);
  }
  if (timing) timings(filters, reps);
  if (check) printf("%s\n", g_fail ? ("FAILED: " + std::to_string(g_fail)).c_str() : "ALL CHECKS PASSED");
  return g_fail ? 1 : 0;
}
