// bench_fn8t -- the G2 bench vehicle for fn8t-1b (docs/preregistration-fn8t-1b.md, FN8T-IMPL-NOTES.md).
//
// Runs each preregistered (pair, shape) cell three ways -- the tensor path, the scalar path and an fp64 CPU
// reference -- and writes /tmp/fn8t-gate/g2.json (contract: FN8T-IMPL-NOTES.md). Cells: qmm6t vs
// qmvs_multi_f6g64 at the four dense shapes plus M / N tail classes, each qmt8v stage vs its moe_q20v scalar
// kernel at the Flash-Next q20 expert shape (arena), and the full R = 8 / R = 5 chains. Gates evaluated per
// cell: half_steps_max <= 1.0, the fp64 errors within 2x of each other, hilo <= single, layer_clean,
// static_tg_bytes <= 16384.
//
// layer_clean: the parent respawns itself as `bench_fn8t layercheck <idx>` under MTL_SHADER_VALIDATION=1
// MTL_DEBUG_LAYER=1 and reads the child's exit code (gpu.mm's commit_and_wait fatals on any command-buffer
// error, so a clean child = clean dispatches). If the full child fails, a tensor-only child (skipping the two
// shipped scalar kernels' dispatches, whose dynamic / 16 KiB static threadgroup memory is shipped code
// outside this branch's change surface) decides, and the table says which mode passed.
//
// Run under the GPU lock:  lockf -k /tmp/kapitsa-gpu.lock build/bench_fn8t
#include "metal/gpu.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>
#include <sys/stat.h>

namespace ni {
extern const char* const kTensorKernelSource;
}
using namespace ni;

namespace {

// ---- the shape constants: Flash-Next q20 (config.json), expert pool 128, arena chunks of 64 ----
constexpr uint32_t kD = 2560, kI = 640, kSI = 640, kTopK = 10, kE = 128;
constexpr uint32_t kImax = 640;  // max(I, SI)
constexpr uint32_t kRk = 8 * kTopK;

// QmvtParams's mirror (zqwen4v.metal) -- byte-identical on the host
struct QmvtP {
  uint32_t K, R, nseg, x_ld;
  uint32_t N[4];
  uint32_t tg_begin[5];
  uint32_t bits[4];
  uint32_t y_ld[4];
  uint32_t add[4];
};

// Q20Slot's mirror (moe_q20.cpp's SlotParams)
struct SlotP {
  uint32_t gc, gd, uc, ud, dc, dd;
};

// q20::slot_layout(D, I), computed locally so the tool stays header-only on the q20 side
struct Lay {
  uint32_t gc, gd, uc, ud, dc, dd, bytes;
};
Lay slot_layout(uint32_t D, uint32_t I) {
  const uint32_t cb = I * D / 4, db = I * D / 32;
  return Lay{0u, 3 * cb, cb, 3 * cb + db, 2 * cb, 3 * cb + 2 * db, (3 * (cb + db) + 16383) / 16384 * 16384};
}

struct Buf {
  Buffer b;
  Buf(size_t bytes = 16) {
    b = Gpu::get().alloc(bytes < 16 ? 16 : bytes, "bench_fn8t");
    memset(b.ptr(), 0, b.size);
  }
  float* f() const { return b.as<float>(); }
  uint32_t* u() const { return b.as<uint32_t>(); }
  uint16_t* u16() const { return b.as<uint16_t>(); }
  uint8_t* u8() const { return b.as<uint8_t>(); }
  __fp16* h() const { return b.as<__fp16>(); }
};
void zero(const Buf& x) { memset(x.b.ptr(), 0, x.b.size); }

Pipeline pipe(const std::string& n) {
  static std::map<std::string, Pipeline> c;
  const auto it = c.find(n);
  if (it != c.end()) return it->second;
  if (n.rfind("qmm6t", 0) == 0 || n.rfind("qmm8t", 0) == 0 || n.rfind("qmt8v", 0) == 0)
    Gpu::get().compile_extra("tensor", kTensorKernelSource, true);
  Pipeline p = Gpu::get().pipeline(n);
  c.emplace(n, p);
  return p;
}

void run(const std::string& kern, const std::function<void(Encoder&)>& bind, uint32_t gx, uint32_t gy, uint32_t gz,
         uint32_t tx) {
  auto e = Gpu::get().begin();
  e->use(pipe(kern));
  bind(*e);
  e->dispatch(gx, gy, gz, tx);
  e->commit_and_wait();
}

std::mt19937 rng(0x5EED1234u);
float urand(float a, float b) { return a + (b - a) * float(double(rng()) / double(rng.max())); }

// ---- the weight formats' CPU decoders (wfmt.metal / zq2moe.metal's layouts) ----

// f6g64: contiguous 6-bit codes, code kk at the row's bit 6*kk (wf_codes16<6>'s window)
inline uint32_t code6(const uint32_t* wrow, uint32_t kk) {
  const uint32_t bit = kk * 6, w0 = bit / 32, s = bit % 32;
  uint32_t v = wrow[w0] >> s;
  if (s + 6 > 32) v |= wrow[w0 + 1] << (32 - s);
  return v & 63u;
}
inline uint32_t code4(const uint32_t* wrow, uint32_t kk) { return (wrow[kk / 8] >> (4 * (kk % 8))) & 15u; }
inline uint32_t code2(const uint32_t* wrow, uint32_t kk) { return (wrow[kk / 16] >> (2 * (kk % 16))) & 3u; }

inline double dsilu(double g) { return g / (1.0 + exp(-g)); }

void deq6_row(const uint32_t* w, const uint16_t* s, const uint16_t* bb, uint32_t row, uint32_t K, double* out) {
  const uint32_t wpr = K * 6 / 32, gpr = K / 64;
  const uint32_t* wr = w + size_t(row) * wpr;
  for (uint32_t kk = 0; kk < K; ++kk) {
    const size_t gi = size_t(row) * gpr + kk / 64;
    out[kk] = double(bf16_to_f32(s[gi])) * double(code6(wr, kk)) + double(bf16_to_f32(bb[gi]));
  }
}

void deq4_mat(const uint32_t* w, const uint16_t* s, const uint16_t* bb, uint32_t N, uint32_t K, double* out) {
  const uint32_t wpr = K / 8, gpr = K / 64;
  for (uint32_t n = 0; n < N; ++n) {
    const uint32_t* wr = w + size_t(n) * wpr;
    for (uint32_t kk = 0; kk < K; ++kk) {
      const size_t gi = size_t(n) * gpr + kk / 64;
      out[size_t(n) * K + kk] =
          double(bf16_to_f32(s[gi])) * double(code4(wr, kk)) + double(bf16_to_f32(bb[gi]));
    }
  }
}

// Q2_0: weight = (code - 1) * d, d fp16 per 64-block; codes plane [N][K/16] uint32, d plane [N][K/64] half
void deq20_mat(const uint32_t* cod, const __fp16* d, uint32_t N, uint32_t K, double* out) {
  for (uint32_t n = 0; n < N; ++n) {
    const uint32_t* cr = cod + size_t(n) * (K / 16);
    const __fp16* dr = d + size_t(n) * (K / 64);
    for (uint32_t kk = 0; kk < K; ++kk)
      out[size_t(n) * K + kk] = (double(code2(cr, kk)) - 1.0) * double(float(dr[kk / 64]));
  }
}

// ---- the cells ----

struct Cell {
  const char* pair;
  const char* shape;
  int kind;  // 0 dense, 1 gu20, 2 gu4, 3 dn20, 4 dn4, 5 the full chain
  uint32_t N, R;
  // the results
  double hs = 0, e_t = 0, e_s = 0, e_hilo = 0, e_single = 0;
  uint32_t tg = 0;
  bool layer = false, layer_all = false;
};

Cell cells[] = {
    {"qmm6t|qmvs_multi_f6g64", "R8 K2560 N512", 0, 512, 8},
    {"qmm6t|qmvs_multi_f6g64", "R8 K2560 N640", 0, 640, 8},
    {"qmm6t|qmvs_multi_f6g64", "R8 K2560 N3584", 0, 3584, 8},
    {"qmm6t|qmvs_multi_f6g64", "R8 K2560 N2560", 0, 2560, 8},
    {"qmm6t|qmvs_multi_f6g64", "R8 K2560 N648 tailN16", 0, 648, 8},
    {"qmm6t|qmvs_multi_f6g64", "R5 K2560 N512 tailM5", 0, 512, 5},
    {"qmt8v_gu20|moe_q20v_gate_up", "R8 I640 D2560 k10 arena", 1, 0, 8},
    {"qmt8v_gu4|moe_q20v_gate_up", "R8 SI640 D2560 k10 arena shared", 2, 0, 8},
    {"qmt8v_dn20|moe_q20_down", "R8 D2560 I640 k10 arena onehot", 3, 0, 8},
    {"qmt8v_dn4|moe_q20_down", "R8 D2560 SI640 k10 arena shared onehot", 4, 0, 8},
    {"qmt8v_chain|moe_q20v+moe_q20_down", "R8 D2560 I640 k10 arena full", 5, 0, 8},
    {"qmt8v_chain|moe_q20v+moe_q20_down", "R5 D2560 I640 k10 arena full tailM5", 5, 0, 5},
};
constexpr int kCells = int(sizeof(cells) / sizeof(cells[0]));

// ---- the dense cells ----

struct Dense {
  uint32_t N = 0, R = 0;
  Buf w, s, bb, x, ys, yt, yt1, t8, t8l, s8, s8l, v8, v8l;
};

void dense_build(Dense& d, uint32_t N, uint32_t R) {
  const uint32_t K = kD;
  d.N = N;
  d.R = R;
  d.w = Buf(size_t(N) * (K * 6 / 32) * 4 + 64);
  d.s = Buf(size_t(N) * (K / 64) * 2);
  d.bb = Buf(size_t(N) * (K / 64) * 2);
  d.x = Buf(8 * K * 4);
  d.ys = Buf(size_t(8) * N * 4);
  d.yt = Buf(size_t(8) * N * 4);
  d.yt1 = Buf(size_t(8) * N * 4);
  d.t8 = Buf(8 * K * 2);
  d.t8l = Buf(8 * K * 2);
  d.s8 = Buf((K / 64) * 8 * 4);
  d.s8l = Buf((K / 64) * 8 * 4);
  d.v8 = Buf(8 * K * 2);
  d.v8l = Buf(8 * K * 2);
  for (size_t i = 0; i < size_t(N) * (K * 6 / 32); ++i) d.w.u()[i] = uint32_t(rng());
  for (size_t i = 0; i < size_t(N) * (K / 64); ++i) {
    d.s.u16()[i] = f32_to_bf16(urand(-0.002f, 0.002f));
    d.bb.u16()[i] = f32_to_bf16(urand(-0.01f, 0.01f));
  }
  for (size_t i = 0; i < size_t(8) * K; ++i) d.x.f()[i] = urand(-1.5f, 1.5f);
}

// the dispatches of a dense cell: x8_prep2, qmm6tp_b6_nt16 (hilo), qmm6t1p_b6_nt16 (single), the scalar
void dense_dispatch(const Dense& d, bool scalar_too) {
  const uint32_t K = kD, N = d.N, R = d.R;
  const kp::X8Prep pp{K, R, K, 0u, 0u, 1e-6f, K, 1u, 0u, 1u, 1u};
  run("x8_prep2",
      [&](Encoder& e) {
        e.buf(0, d.x.b);
        e.buf(1, d.x.b);  // the norm weight: unused (norm = 0)
        e.buf(2, d.v8.b);
        e.buf(3, d.s8.b);
        e.buf(4, d.x.b);  // xn: unused (write_xn = 0)
        e.val(5, pp);
        e.buf(6, d.v8l.b);
        e.buf(7, d.s8l.b);
        e.buf(8, d.t8.b);
        e.buf(9, d.t8l.b);
      },
      K / 256, 1, 1, 256);
  // the GEMM: K = 2560 -> gpr 40 -> ks 4, pf 1, nt 16 (enc_rows8t's computed rules at these shapes)
  const uint32_t tg = (N + 15) / 16;
  kp::Qmm8 p{};
  p.K = K;
  p.R = R;
  p.nseg = 1;
  p.ks = 4;
  p.N[0] = N;
  p.y_ld[0] = N;
  p.add[0] = 0;
  p.tg_begin[1] = tg;
  const auto bindt = [&](const Buffer& y) {
    return [&](Encoder& e) {
      for (int i = 0; i < 4; ++i) {  // clamped: the kernel reads no segment past nseg
        e.buf(i * 4 + 0, d.w.b);
        e.buf(i * 4 + 1, d.s.b);
        e.buf(i * 4 + 2, d.bb.b);
        e.buf(i * 4 + 3, y);
      }
      e.buf(16, d.t8.b);
      e.buf(17, d.s8.b);
      e.buf(18, d.t8l.b);
      e.buf(19, d.s8l.b);
      e.val(20, p);
    };
  };
  run("qmm6tp_b6_nt16", bindt(d.yt.b), tg, 1, 1, 128);
  run("qmm6t1p_b6_nt16", bindt(d.yt1.b), tg, 1, 1, 128);
  if (!scalar_too) return;
  QmvtP qp{};
  qp.K = K;
  qp.R = R;
  qp.nseg = 1;
  qp.x_ld = K;
  qp.N[0] = N;
  qp.tg_begin[1] = (N + 7) / 8;
  qp.bits[0] = 6;
  qp.y_ld[0] = N;
  run("qmvs_multi_f6g64",
      [&](Encoder& e) {
        for (int i = 0; i < 4; ++i) {
          e.buf(i * 4 + 0, d.w.b);
          e.buf(i * 4 + 1, d.s.b);
          e.buf(i * 4 + 2, d.bb.b);
          e.buf(i * 4 + 3, d.ys.b);
        }
        e.buf(16, d.x.b);
        e.val(17, qp);
      },
      R, (N + 7) / 8, 1, 64);
}

void dense_ref(Dense& d, Cell& c) {
  const uint32_t K = kD, N = d.N, R = d.R;
  std::vector<double> wr(K);
  double hmax = 0, hdev = 0;
  for (uint32_t n = 0; n < N; ++n) {
    deq6_row(d.w.u(), d.s.u16(), d.bb.u16(), n, K, wr.data());
    for (uint32_t r = 0; r < R; ++r) {
      const float* xr = d.x.f() + size_t(r) * K;
      double ref = 0;
      for (uint32_t kk = 0; kk < K; ++kk) ref += wr[kk] * double(xr[kk]);
      const size_t o = size_t(r) * N + n;
      const double sv = double(d.ys.f()[o]);
      hmax = std::max(hmax, fabs(sv));
      hdev = std::max(hdev, fabs(double(d.yt.f()[o]) - sv));
      c.e_s = std::max(c.e_s, fabs(sv - ref));
      c.e_hilo = c.e_t = std::max(c.e_hilo, fabs(double(d.yt.f()[o]) - ref));
      c.e_single = std::max(c.e_single, fabs(double(d.yt1.f()[o]) - ref));
    }
  }
  c.hs = hdev / ldexp(hmax ? hmax : 1.0, -11);
  c.tg = pipe("qmm6tp_b6_nt16").tg_mem;
}

// ---- the expert fixture: an arena of 128 Q2_0 experts + the shared 4-bit expert + unique's outputs ----

struct Uniq {
  Buf ue, umask, uslot, un;
  std::vector<uint32_t> ueC, umaskC;
  std::vector<uint8_t> uslotC;
  uint32_t n = 0;
};

struct Tables {
  const Buf *ht, *htl, *hs, *hsl, *hst, *hstl, *hss, *hssl;
};

struct Fix {
  Lay L = slot_layout(kD, kI);
  Buf a0, a1, slots;
  Buf gw, gs, gb, uw, us, ub, dw, ds, db;  // the shared expert's 4-bit planes
  Buf xn, ridx, wts;
  std::vector<uint32_t> ridxC;
  Uniq u8, u5;
  Buf he8, he8b, out8, out8b;
  Buf ht8, ht8l, hs8, hs8l, hst8, hst8l, hss8, hss8l;          // the hilo chains' down tables
  Buf ht8b, ht8lb, hs8b, hs8lb, hst8b, hst8lb, hss8b, hss8lb;  // the single chains'
  Buf t8x, t8xl, s8x, s8xl, v8x, v8xl;                         // x8_prep2's outputs (K = D)
  uint8_t* plane(int e) const { return (e < 64 ? a0.u8() : a1.u8()) + size_t(e % 64) * L.bytes; }
  Tables ta() const { return Tables{&ht8, &ht8l, &hs8, &hs8l, &hst8, &hst8l, &hss8, &hss8l}; }
  Tables tb() const { return Tables{&ht8b, &ht8lb, &hs8b, &hs8lb, &hst8b, &hst8lb, &hss8b, &hss8lb}; }
};

void unique_dispatch(const Fix& F, Uniq& U, uint32_t R) {
  U.ue = Buf(512);
  U.umask = Buf(512);
  U.uslot = Buf(1024);
  U.un = Buf(16);
  const struct {
    uint32_t R, k;
  } rk{R, kTopK};
  run("moe_q20_unique",
      [&](Encoder& e) {
        e.buf(0, F.ridx.b);
        e.buf(1, U.ue.b);
        e.buf(2, U.umask.b);
        e.buf(3, U.uslot.b);
        e.buf(4, U.un.b);
        e.val(5, rk);
      },
      1, 1, 1, 32);
  U.n = U.un.u()[0];
  U.ueC.assign(U.ue.u(), U.ue.u() + U.n);
  U.umaskC.assign(U.umask.u(), U.umask.u() + kE);
  U.uslotC.assign(U.uslot.u8(), U.uslot.u8() + size_t(U.n) * 8);
}

void fix_build(Fix& F) {
  const uint32_t pages = F.L.bytes / 16384;
  F.a0 = Buf(size_t(64) * F.L.bytes);
  F.a1 = Buf(size_t(64) * F.L.bytes);
  F.slots = Buf(kE * 4);
  for (uint32_t e = 0; e < kE; ++e) {
    uint8_t* b = F.plane(int(e));
    uint32_t* gc = (uint32_t*)(b + F.L.gc);
    uint32_t* uc = (uint32_t*)(b + F.L.uc);
    uint32_t* dc = (uint32_t*)(b + F.L.dc);
    __fp16* gd = (__fp16*)(b + F.L.gd);
    __fp16* ud = (__fp16*)(b + F.L.ud);
    __fp16* dd = (__fp16*)(b + F.L.dd);
    for (size_t i = 0; i < size_t(kI) * (kD / 16); ++i) gc[i] = uint32_t(rng());
    for (size_t i = 0; i < size_t(kI) * (kD / 16); ++i) uc[i] = uint32_t(rng());
    for (size_t i = 0; i < size_t(kD) * (kI / 16); ++i) dc[i] = uint32_t(rng());
    for (size_t i = 0; i < size_t(kI) * (kD / 64); ++i) gd[i] = __fp16(urand(0.004f, 0.03f));
    for (size_t i = 0; i < size_t(kI) * (kD / 64); ++i) ud[i] = __fp16(urand(0.004f, 0.03f));
    for (size_t i = 0; i < size_t(kD) * (kI / 64); ++i) dd[i] = __fp16(urand(0.004f, 0.03f));
    F.slots.u()[e] = ((e / 64) << 28) | ((e % 64) * pages);
  }
  F.gw = Buf(size_t(kSI) * (kD / 8) * 4);
  F.gs = Buf(size_t(kSI) * (kD / 64) * 2);
  F.gb = Buf(size_t(kSI) * (kD / 64) * 2);
  F.uw = Buf(size_t(kSI) * (kD / 8) * 4);
  F.us = Buf(size_t(kSI) * (kD / 64) * 2);
  F.ub = Buf(size_t(kSI) * (kD / 64) * 2);
  F.dw = Buf(size_t(kD) * (kSI / 8) * 4);
  F.ds = Buf(size_t(kD) * (kSI / 64) * 2);
  F.db = Buf(size_t(kD) * (kSI / 64) * 2);
  for (size_t i = 0; i < size_t(kSI) * (kD / 8); ++i) F.gw.u()[i] = uint32_t(rng());
  for (size_t i = 0; i < size_t(kSI) * (kD / 8); ++i) F.uw.u()[i] = uint32_t(rng());
  for (size_t i = 0; i < size_t(kD) * (kSI / 8); ++i) F.dw.u()[i] = uint32_t(rng());
  for (size_t i = 0; i < size_t(kSI) * (kD / 64); ++i) {
    F.gs.u16()[i] = f32_to_bf16(urand(-0.002f, 0.002f));
    F.gb.u16()[i] = f32_to_bf16(urand(-0.01f, 0.01f));
    F.us.u16()[i] = f32_to_bf16(urand(-0.002f, 0.002f));
    F.ub.u16()[i] = f32_to_bf16(urand(-0.01f, 0.01f));
  }
  for (size_t i = 0; i < size_t(kD) * (kSI / 64); ++i) {
    F.ds.u16()[i] = f32_to_bf16(urand(-0.002f, 0.002f));
    F.db.u16()[i] = f32_to_bf16(urand(-0.01f, 0.01f));
  }
  F.xn = Buf(8 * kD * 4);
  for (size_t i = 0; i < size_t(8) * kD; ++i) F.xn.f()[i] = urand(-1.5f, 1.5f);
  F.ridx = Buf(8 * kTopK * 4);
  F.ridxC.resize(8 * kTopK);
  // a real router's top-k is distinct experts per token, and uslot holds one slot per (u, t):
  // duplicates inside a token would leave a slot with no plane in the unique layout
  for (uint32_t t = 0; t < 8; ++t) {
    std::vector<uint32_t> pool(kE);
    for (uint32_t e = 0; e < kE; ++e) pool[e] = e;
    for (uint32_t s = 0; s < kTopK; ++s) {
      std::swap(pool[s], pool[s + uint32_t(rng()) % (kE - s)]);
      F.ridxC[size_t(t) * kTopK + s] = pool[s];
      F.ridx.u()[size_t(t) * kTopK + s] = pool[s];
    }
  }
  F.wts = Buf(8 * (kTopK + 1) * 4);
  for (size_t i = 0; i < size_t(8) * (kTopK + 1); ++i) F.wts.f()[i] = urand(0.02f, 0.15f);
  F.he8 = Buf(size_t(kRk + 1) * 8 * kImax * 4);
  F.he8b = Buf(size_t(kRk + 1) * 8 * kImax * 4);
  F.out8 = Buf(size_t(8) * (kTopK + 1) * kD * 4);
  F.out8b = Buf(size_t(8) * (kTopK + 1) * kD * 4);
  F.ht8 = Buf(size_t(kRk) * 8 * kI * 2);
  F.ht8l = Buf(size_t(kRk) * 8 * kI * 2);
  F.hs8 = Buf(size_t(kRk) * (kI / 64) * 8 * 4);
  F.hs8l = Buf(size_t(kRk) * (kI / 64) * 8 * 4);
  F.hst8 = Buf(size_t(8) * kSI * 2);
  F.hst8l = Buf(size_t(8) * kSI * 2);
  F.hss8 = Buf(size_t(kSI / 64) * 8 * 4);
  F.hss8l = Buf(size_t(kSI / 64) * 8 * 4);
  F.ht8b = Buf(size_t(kRk) * 8 * kI * 2);
  F.ht8lb = Buf(size_t(kRk) * 8 * kI * 2);
  F.hs8b = Buf(size_t(kRk) * (kI / 64) * 8 * 4);
  F.hs8lb = Buf(size_t(kRk) * (kI / 64) * 8 * 4);
  F.hst8b = Buf(size_t(8) * kSI * 2);
  F.hst8lb = Buf(size_t(8) * kSI * 2);
  F.hss8b = Buf(size_t(kSI / 64) * 8 * 4);
  F.hss8lb = Buf(size_t(kSI / 64) * 8 * 4);
  F.t8x = Buf(8 * kD * 2);
  F.t8xl = Buf(8 * kD * 2);
  F.s8x = Buf((kD / 64) * 8 * 4);
  F.s8xl = Buf((kD / 64) * 8 * 4);
  F.v8x = Buf(8 * kD * 2);
  F.v8xl = Buf(8 * kD * 2);
  unique_dispatch(F, F.u8, 8);
  unique_dispatch(F, F.u5, 5);
}

kp::Qmt8v qparams(const Fix& F, uint32_t R, bool lo) {
  kp::Qmt8v p{};
  p.D = kD;
  p.I = kI;
  p.SI = kSI;
  p.k = kTopK;
  p.R = R;
  p.ks_gu = 4;  // enc_moe_experts8t's ks_for at these shapes: D/64 = 40 -> 4, I/64 = SI/64 = 10 -> 8
  p.ks_d20 = 8;
  p.ks_d4 = 8;
  p.arena = 1;
  p.Imax = kImax;
  p.lo = lo ? 1u : 0u;
  p.lgc = F.L.gc;
  p.lgd = F.L.gd;
  p.luc = F.L.uc;
  p.lud = F.L.ud;
  p.ldc = F.L.dc;
  p.ldd = F.L.dd;
  return p;
}

// ---- the tensor steps (enc_moe_experts8t's dispatches, mirrored) ----

void f_prep(const Fix& F, uint32_t R) {
  const kp::X8Prep pp{kD, R, kD, 0u, 0u, 1e-6f, kD, 1u, 0u, 1u, 1u};
  run("x8_prep2",
      [&](Encoder& e) {
        e.buf(0, F.xn.b);
        e.buf(1, F.xn.b);
        e.buf(2, F.v8x.b);
        e.buf(3, F.s8x.b);
        e.buf(4, F.xn.b);
        e.val(5, pp);
        e.buf(6, F.v8xl.b);
        e.buf(7, F.s8xl.b);
        e.buf(8, F.t8x.b);
        e.buf(9, F.t8xl.b);
      },
      kD / 256, 1, 1, 256);
}

void f_gu20(const Fix& F, const Uniq& U, bool single, const Buf& he) {
  const kp::Qmt8v p = qparams(F, 8, !single);
  run(single ? "qmt8v_gu201p" : "qmt8v_gu20p",
      [&](Encoder& e) {
        e.buf(0, F.a0.b);
        e.buf(1, F.a1.b);
        e.buf(2, F.slots.b);
        e.buf(3, F.slots.b);
        e.buf(4, F.t8x.b);
        e.buf(5, F.s8x.b);
        e.buf(6, F.t8xl.b);
        e.buf(7, F.s8xl.b);
        e.buf(8, he.b);
        e.buf(9, U.ue.b);
        e.buf(10, U.un.b);
        e.val(11, p);
      },
      kI / 16, 1, kRk, 128);
}

void f_gu4(const Fix& F, bool single, const Buf& he) {
  const kp::Qmt8v p = qparams(F, 8, !single);
  run(single ? "qmt8v_gu41p" : "qmt8v_gu4p",
      [&](Encoder& e) {
        e.buf(0, F.gw.b);
        e.buf(1, F.gs.b);
        e.buf(2, F.gb.b);
        e.buf(3, F.uw.b);
        e.buf(4, F.us.b);
        e.buf(5, F.ub.b);
        e.buf(6, F.t8x.b);
        e.buf(7, F.s8x.b);
        e.buf(8, F.t8xl.b);
        e.buf(9, F.s8xl.b);
        e.buf(10, he.b);
        e.val(11, p);
      },
      kSI / 16, 1, 1, 128);
}

void f_hsum(const Fix& F, const Uniq& U, const Buf& he, Tables T) {
  const kp::Qmt8v p = qparams(F, 8, true);
  run("qmt8v_hsum",
      [&](Encoder& e) {
        e.buf(0, he.b);
        e.buf(1, T.ht->b);
        e.buf(2, T.htl->b);
        e.buf(3, T.hs->b);
        e.buf(4, T.hsl->b);
        e.buf(5, T.hst->b);
        e.buf(6, T.hstl->b);
        e.buf(7, T.hss->b);
        e.buf(8, T.hssl->b);
        e.buf(9, U.un.b);
        e.val(10, p);
      },
      (kImax + 255) / 256, 1, kRk + 1, 256);
}

void f_dn20(const Fix& F, const Uniq& U, bool single, Tables T, const Buf& out) {
  const kp::Qmt8v p = qparams(F, 8, !single);
  run(single ? "qmt8v_dn201" : "qmt8v_dn20",
      [&](Encoder& e) {
        e.buf(0, F.a0.b);
        e.buf(1, F.a1.b);
        e.buf(2, F.slots.b);
        e.buf(3, T.ht->b);
        e.buf(4, T.hs->b);
        e.buf(5, T.htl->b);
        e.buf(6, T.hsl->b);
        e.buf(7, out.b);
        e.buf(8, U.umask.b);
        e.buf(9, U.uslot.b);
        e.buf(10, U.ue.b);
        e.buf(11, U.un.b);
        e.val(12, p);
      },
      kD / 16, 1, kRk, 256);
}

void f_dn4(const Fix& F, bool single, Tables T, const Buf& out) {
  const kp::Qmt8v p = qparams(F, 8, !single);
  run(single ? "qmt8v_dn41" : "qmt8v_dn4",
      [&](Encoder& e) {
        e.buf(0, F.dw.b);
        e.buf(1, F.ds.b);
        e.buf(2, F.db.b);
        e.buf(3, T.hst->b);
        e.buf(4, T.hss->b);
        e.buf(5, T.hstl->b);
        e.buf(6, T.hssl->b);
        e.buf(7, out.b);
        e.val(8, p);
      },
      kD / 16, 1, 1, 256);
}

void f_combine(const Fix& F, uint32_t R, const Buf& out, const Buf& wts, const Buf& y) {
  const kp::Qmt8v p = qparams(F, R, true);
  run("qmt8v_combine",
      [&](Encoder& e) {
        e.buf(0, out.b);
        e.buf(1, wts.b);
        e.buf(2, y.b);
        e.val(3, p);
      },
      (R * kD + 255) / 256, 1, 1, 256);
}

// ---- the scalar steps (q20::enc_verify's dispatches, mirrored) ----

void s_gu(const Fix& F, const Uniq& U, uint32_t R, const Buf& h) {
  const kp::Moe mp{kD, kI, kSI, kTopK, 0.0f};
  const SlotP L{F.L.gc, F.L.gd, F.L.uc, F.L.ud, F.L.dc, F.L.dd};
  run("moe_q20v_gate_up_slot",
      [&](Encoder& e) {
        e.buf(0, F.a0.b);
        e.buf(1, F.a1.b);
        e.buf(2, F.slots.b);
        e.buf(3, F.slots.b);
        e.buf(4, F.gw.b);
        e.buf(5, F.gs.b);
        e.buf(6, F.gb.b);
        e.buf(7, F.uw.b);
        e.buf(8, F.us.b);
        e.buf(9, F.ub.b);
        e.buf(10, F.xn.b);
        e.buf(11, h.b);
        e.val(12, mp);
        e.buf(13, U.ue.b);
        e.buf(14, U.umask.b);
        e.buf(15, U.uslot.b);
        e.buf(16, U.un.b);
        e.val(17, R);
        e.val(18, L);
      },
      R, (kImax + 31) / 32, R * kTopK + 1, 256);
}

void s_dn(const Fix& F, uint32_t R, const Buf& h, const Buf& idx, const Buf& wts, const Buf& y) {
  const kp::Moe mp{kD, kI, kSI, kTopK, 0.0f};
  const SlotP L{F.L.gc, F.L.gd, F.L.uc, F.L.ud, F.L.dc, F.L.dd};
  run("moe_q20_down_slot",
      [&](Encoder& e) {
        e.buf(0, F.a0.b);
        e.buf(1, F.a1.b);
        e.buf(2, F.slots.b);
        e.buf(3, F.dw.b);
        e.buf(4, F.ds.b);
        e.buf(5, F.db.b);
        e.buf(6, h.b);
        e.buf(7, idx.b);
        e.buf(8, wts.b);
        e.buf(9, y.b);
        e.val(10, mp);
        e.val(11, L);
        e.tgmem(0, size_t(kTopK + 1) * kImax * 4);
      },
      R, (kD + 31) / 32, 1, 256);
}

// ---- the expert cells: dispatches shared with the layercheck child, metrics behind `metrics` ----

void cell_gu20(Fix& F, Cell& c, bool metrics) {
  f_prep(F, 8);
  zero(F.he8);
  zero(F.he8b);
  f_gu20(F, F.u8, false, F.he8);
  f_gu20(F, F.u8, true, F.he8b);
  Buf hs(size_t(8) * (kTopK + 1) * kImax * 4);
  s_gu(F, F.u8, 8, hs);
  if (!metrics) return;
  const Uniq& U = F.u8;
  std::vector<double> gW(size_t(kI) * kD), uW(size_t(kI) * kD);
  double hmax = 0, hdev = 0;
  for (uint32_t u = 0; u < U.n; ++u) {
    const uint8_t* b = F.plane(int(U.ueC[u]));
    deq20_mat((const uint32_t*)(b + F.L.gc), (const __fp16*)(b + F.L.gd), kI, kD, gW.data());
    deq20_mat((const uint32_t*)(b + F.L.uc), (const __fp16*)(b + F.L.ud), kI, kD, uW.data());
    for (uint32_t t = 0; t < 8; ++t) {
      if (!((U.umaskC[u] >> t) & 1u)) continue;
      const uint32_t slot = U.uslotC[size_t(u) * 8 + t];
      const float* xr = F.xn.f() + size_t(t) * kD;
      const float* htr = F.he8.f() + (size_t(u) * 8 + t) * kImax;
      const float* htr1 = F.he8b.f() + (size_t(u) * 8 + t) * kImax;
      const float* hsr = hs.f() + (size_t(t) * (kTopK + 1) + slot) * kImax;
      for (uint32_t i = 0; i < kI; ++i) {
        const double* gw = gW.data() + size_t(i) * kD;
        const double* uw = uW.data() + size_t(i) * kD;
        double g = 0, up = 0;
        for (uint32_t kk = 0; kk < kD; ++kk) {
          const double xv = double(xr[kk]);
          g += gw[kk] * xv;
          up += uw[kk] * xv;
        }
        const double ref = dsilu(g) * up;
        const double sv = double(hsr[i]);
        hmax = std::max(hmax, fabs(sv));
        hdev = std::max(hdev, fabs(double(htr[i]) - sv));
        c.e_s = std::max(c.e_s, fabs(sv - ref));
        c.e_hilo = c.e_t = std::max(c.e_hilo, fabs(double(htr[i]) - ref));
        c.e_single = std::max(c.e_single, fabs(double(htr1[i]) - ref));
      }
    }
  }
  c.hs = hdev / ldexp(hmax ? hmax : 1.0, -11);
  c.tg = pipe("qmt8v_gu20p").tg_mem;
}

void cell_gu4(Fix& F, Cell& c, bool metrics) {
  f_prep(F, 8);
  zero(F.he8);
  zero(F.he8b);
  f_gu4(F, false, F.he8);
  f_gu4(F, true, F.he8b);
  Buf hs(size_t(8) * (kTopK + 1) * kImax * 4);
  s_gu(F, F.u8, 8, hs);
  if (!metrics) return;
  std::vector<double> gW(size_t(kSI) * kD), uW(size_t(kSI) * kD);
  deq4_mat(F.gw.u(), F.gs.u16(), F.gb.u16(), kSI, kD, gW.data());
  deq4_mat(F.uw.u(), F.us.u16(), F.ub.u16(), kSI, kD, uW.data());
  double hmax = 0, hdev = 0;
  for (uint32_t t = 0; t < 8; ++t) {
    const float* xr = F.xn.f() + size_t(t) * kD;
    const float* htr = F.he8.f() + (size_t(kRk) * 8 + t) * kImax;
    const float* htr1 = F.he8b.f() + (size_t(kRk) * 8 + t) * kImax;
    const float* hsr = hs.f() + (size_t(t) * (kTopK + 1) + kTopK) * kImax;
    for (uint32_t i = 0; i < kSI; ++i) {
      const double* gw = gW.data() + size_t(i) * kD;
      const double* uw = uW.data() + size_t(i) * kD;
      double g = 0, up = 0;
      for (uint32_t kk = 0; kk < kD; ++kk) {
        const double xv = double(xr[kk]);
        g += gw[kk] * xv;
        up += uw[kk] * xv;
      }
      const double ref = dsilu(g) * up;
      const double sv = double(hsr[i]);
      hmax = std::max(hmax, fabs(sv));
      hdev = std::max(hdev, fabs(double(htr[i]) - sv));
      c.e_s = std::max(c.e_s, fabs(sv - ref));
      c.e_hilo = c.e_t = std::max(c.e_hilo, fabs(double(htr[i]) - ref));
      c.e_single = std::max(c.e_single, fabs(double(htr1[i]) - ref));
    }
  }
  c.hs = hdev / ldexp(hmax ? hmax : 1.0, -11);
  c.tg = pipe("qmt8v_gu4p").tg_mem;
}

// permutes the scalar h layout (per row, per slot) into he8's plane layout (per unique expert, 8 tokens)
void permute_h_into_he8(const Fix& F, const float* hs) {
  zero(F.he8);
  const Uniq& U = F.u8;
  for (uint32_t u = 0; u < U.n; ++u)
    for (uint32_t t = 0; t < 8; ++t) {
      if (!((U.umaskC[u] >> t) & 1u)) continue;
      const uint32_t slot = U.uslotC[size_t(u) * 8 + t];
      memcpy(F.he8.f() + (size_t(u) * 8 + t) * kImax, hs + (size_t(t) * (kTopK + 1) + slot) * kImax,
             kImax * sizeof(float));
    }
  for (uint32_t t = 0; t < 8; ++t)
    memcpy(F.he8.f() + (size_t(kRk) * 8 + t) * kImax, hs + (size_t(t) * (kTopK + 1) + kTopK) * kImax,
           kImax * sizeof(float));
}

Buf synth_h() {
  Buf hs(size_t(8) * (kTopK + 1) * kImax * 4);
  for (size_t i = 0; i < size_t(8) * (kTopK + 1) * kImax; ++i) hs.f()[i] = urand(-2.0f, 2.0f);
  return hs;
}

void cell_dn20(Fix& F, Cell& c, bool metrics) {
  Buf hs = synth_h();  // h straight from the host: isolates dn from gu's error
  permute_h_into_he8(F, hs.f());
  f_hsum(F, F.u8, F.he8, F.ta());
  zero(F.out8);
  zero(F.out8b);
  f_dn20(F, F.u8, false, F.ta(), F.out8);
  f_dn20(F, F.u8, true, F.ta(), F.out8b);
  // the scalar side: one one-hot run per slot, so y[t] is exactly slot j's down output
  std::vector<Buf> ys;
  for (uint32_t j = 0; j <= kTopK; ++j) {
    Buf wj(size_t(8) * (kTopK + 1) * 4);
    for (uint32_t t = 0; t < 8; ++t) wj.f()[size_t(t) * (kTopK + 1) + j] = 1.0f;
    ys.emplace_back(size_t(8) * kD * 4);
    s_dn(F, 8, hs, F.ridx, wj, ys.back());
  }
  if (!metrics) return;
  const Uniq& U = F.u8;
  std::vector<double> dW(size_t(kD) * kI);
  double hmax = 0, hdev = 0;
  for (uint32_t t = 0; t < 8; ++t)
    for (uint32_t j = 0; j < kTopK; ++j) {
      uint32_t u = 0;
      bool found = false;
      for (uint32_t v = 0; v < U.n; ++v)
        if ((U.umaskC[v] >> t) & 1u && U.uslotC[size_t(v) * 8 + t] == j) {
          u = v;
          found = true;
          break;
        }
      if (!found) fatal("bench_fn8t: no plane for (t %u, slot %u)", t, j);
      const uint8_t* b = F.plane(int(U.ueC[u]));
      deq20_mat((const uint32_t*)(b + F.L.dc), (const __fp16*)(b + F.L.dd), kD, kI, dW.data());
      const float* hr = hs.f() + (size_t(t) * (kTopK + 1) + j) * kImax;
      const float* otr = F.out8.f() + (size_t(t) * (kTopK + 1) + j) * kD;
      const float* otr1 = F.out8b.f() + (size_t(t) * (kTopK + 1) + j) * kD;
      const float* osr = ys[j].f() + size_t(t) * kD;
      for (uint32_t d = 0; d < kD; ++d) {
        const double* wr = dW.data() + size_t(d) * kI;
        double ref = 0;
        for (uint32_t i = 0; i < kI; ++i) ref += wr[i] * double(hr[i]);
        const double sv = double(osr[d]);
        hmax = std::max(hmax, fabs(sv));
        hdev = std::max(hdev, fabs(double(otr[d]) - sv));
        c.e_s = std::max(c.e_s, fabs(sv - ref));
        c.e_hilo = c.e_t = std::max(c.e_hilo, fabs(double(otr[d]) - ref));
        c.e_single = std::max(c.e_single, fabs(double(otr1[d]) - ref));
      }
    }
  c.hs = hdev / ldexp(hmax ? hmax : 1.0, -11);
  c.tg = pipe("qmt8v_dn20").tg_mem;
}

void cell_dn4(Fix& F, Cell& c, bool metrics) {
  Buf hs = synth_h();
  permute_h_into_he8(F, hs.f());
  f_hsum(F, F.u8, F.he8, F.ta());
  zero(F.out8);
  zero(F.out8b);
  f_dn4(F, false, F.ta(), F.out8);
  f_dn4(F, true, F.ta(), F.out8b);
  Buf wj(size_t(8) * (kTopK + 1) * 4);
  for (uint32_t t = 0; t < 8; ++t) wj.f()[size_t(t) * (kTopK + 1) + kTopK] = 1.0f;
  Buf ys(size_t(8) * kD * 4);
  s_dn(F, 8, hs, F.ridx, wj, ys);
  if (!metrics) return;
  std::vector<double> dW(size_t(kD) * kSI);
  deq4_mat(F.dw.u(), F.ds.u16(), F.db.u16(), kD, kSI, dW.data());
  double hmax = 0, hdev = 0;
  for (uint32_t t = 0; t < 8; ++t) {
    const float* hr = hs.f() + (size_t(t) * (kTopK + 1) + kTopK) * kImax;
    const float* otr = F.out8.f() + (size_t(t) * (kTopK + 1) + kTopK) * kD;
    const float* otr1 = F.out8b.f() + (size_t(t) * (kTopK + 1) + kTopK) * kD;
    const float* osr = ys.f() + size_t(t) * kD;
    for (uint32_t d = 0; d < kD; ++d) {
      const double* wr = dW.data() + size_t(d) * kSI;
      double ref = 0;
      for (uint32_t i = 0; i < kSI; ++i) ref += wr[i] * double(hr[i]);
      const double sv = double(osr[d]);
      hmax = std::max(hmax, fabs(sv));
      hdev = std::max(hdev, fabs(double(otr[d]) - sv));
      c.e_s = std::max(c.e_s, fabs(sv - ref));
      c.e_hilo = c.e_t = std::max(c.e_hilo, fabs(double(otr[d]) - ref));
      c.e_single = std::max(c.e_single, fabs(double(otr1[d]) - ref));
    }
  }
  c.hs = hdev / ldexp(hmax ? hmax : 1.0, -11);
  c.tg = pipe("qmt8v_dn4").tg_mem;
}

void cell_full(Fix& F, Cell& c, bool metrics) {
  const uint32_t R = c.R;
  const Uniq& U = R == 8 ? F.u8 : F.u5;
  f_prep(F, R);
  zero(F.he8);
  zero(F.he8b);
  zero(F.out8);
  zero(F.out8b);
  Buf yt(size_t(8) * kD * 4), yt1(size_t(8) * kD * 4), ys(size_t(8) * kD * 4);
  // the hilo chain
  f_gu20(F, U, false, F.he8);
  f_gu4(F, false, F.he8);
  f_hsum(F, U, F.he8, F.ta());
  f_dn20(F, U, false, F.ta(), F.out8);
  f_dn4(F, false, F.ta(), F.out8);
  f_combine(F, R, F.out8, F.wts, yt);
  // the single-term chain (its own h and tables, so every stage runs its LO = 0 form)
  f_gu20(F, U, true, F.he8b);
  f_gu4(F, true, F.he8b);
  f_hsum(F, U, F.he8b, F.tb());
  f_dn20(F, U, true, F.tb(), F.out8b);
  f_dn4(F, true, F.tb(), F.out8b);
  f_combine(F, R, F.out8b, F.wts, yt1);
  // the scalar chain
  Buf hs(size_t(8) * (kTopK + 1) * kImax * 4);
  s_gu(F, U, R, hs);
  s_dn(F, R, hs, F.ridx, F.wts, ys);
  if (!metrics) return;
  // the end-to-end fp64 reference: h in double, down in double
  std::vector<double> gW(size_t(kI) * kD), uW(size_t(kI) * kD), dW(size_t(kD) * kI);
  std::vector<double> shG(size_t(kSI) * kD), shU(size_t(kSI) * kD), shD(size_t(kD) * kSI);
  deq4_mat(F.gw.u(), F.gs.u16(), F.gb.u16(), kSI, kD, shG.data());
  deq4_mat(F.uw.u(), F.us.u16(), F.ub.u16(), kSI, kD, shU.data());
  deq4_mat(F.dw.u(), F.ds.u16(), F.db.u16(), kD, kSI, shD.data());
  std::vector<double> hh(kImax), refY(size_t(R) * kD, 0.0);
  for (uint32_t t = 0; t < R; ++t) {
    const float* xr = F.xn.f() + size_t(t) * kD;
    for (uint32_t j = 0; j <= kTopK; ++j) {
      const double *gp, *up_, *dp;
      uint32_t NI;
      if (j == kTopK) {
        gp = shG.data();
        up_ = shU.data();
        dp = shD.data();
        NI = kSI;
      } else {
        const uint8_t* b = F.plane(int(F.ridxC[size_t(t) * kTopK + j]));
        deq20_mat((const uint32_t*)(b + F.L.gc), (const __fp16*)(b + F.L.gd), kI, kD, gW.data());
        deq20_mat((const uint32_t*)(b + F.L.uc), (const __fp16*)(b + F.L.ud), kI, kD, uW.data());
        deq20_mat((const uint32_t*)(b + F.L.dc), (const __fp16*)(b + F.L.dd), kD, kI, dW.data());
        gp = gW.data();
        up_ = uW.data();
        dp = dW.data();
        NI = kI;
      }
      for (uint32_t i = 0; i < NI; ++i) {
        const double* gw = gp + size_t(i) * kD;
        const double* uw = up_ + size_t(i) * kD;
        double g = 0, u = 0;
        for (uint32_t kk = 0; kk < kD; ++kk) {
          const double xv = double(xr[kk]);
          g += gw[kk] * xv;
          u += uw[kk] * xv;
        }
        hh[i] = dsilu(g) * u;
      }
      const double wt = double(F.wts.f()[size_t(t) * (kTopK + 1) + j]);
      for (uint32_t d = 0; d < kD; ++d) {
        const double* wr = dp + size_t(d) * NI;
        double acc = 0;
        for (uint32_t i = 0; i < NI; ++i) acc += wr[i] * hh[i];
        refY[size_t(t) * kD + d] += wt * acc;
      }
    }
  }
  double hmax = 0, hdev = 0;
  for (uint32_t t = 0; t < R; ++t)
    for (uint32_t d = 0; d < kD; ++d) {
      const size_t o = size_t(t) * kD + d;
      const double sv = double(ys.f()[o]), ref = refY[o];
      hmax = std::max(hmax, fabs(sv));
      hdev = std::max(hdev, fabs(double(yt.f()[o]) - sv));
      c.e_s = std::max(c.e_s, fabs(sv - ref));
      c.e_hilo = c.e_t = std::max(c.e_hilo, fabs(double(yt.f()[o]) - ref));
      c.e_single = std::max(c.e_single, fabs(double(yt1.f()[o]) - ref));
    }
  c.hs = hdev / ldexp(hmax ? hmax : 1.0, -11);
  c.tg = std::max({pipe("qmt8v_gu20p").tg_mem, pipe("qmt8v_gu4p").tg_mem, pipe("qmt8v_hsum").tg_mem,
                   pipe("qmt8v_dn20").tg_mem, pipe("qmt8v_dn4").tg_mem, pipe("qmt8v_combine").tg_mem});
}

void run_cell(int idx, Fix& F, bool metrics) {
  Cell& c = cells[idx];
  switch (c.kind) {
    case 1: cell_gu20(F, c, metrics); break;
    case 2: cell_gu4(F, c, metrics); break;
    case 3: cell_dn20(F, c, metrics); break;
    case 4: cell_dn4(F, c, metrics); break;
    case 5: cell_full(F, c, metrics); break;
    default: fatal("bench_fn8t: cell %d is dense", idx);
  }
}

void dispatch_cell(int idx, bool tensor_only) {
  const Cell& c = cells[idx];
  if (c.kind == 0) {
    Dense d;
    dense_build(d, c.N, c.R);
    dense_dispatch(d, !tensor_only);
    return;
  }
  Fix F;
  fix_build(F);
  run_cell(idx, F, false);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 3 && strcmp(argv[1], "layercheck") == 0) {
    const int idx = atoi(argv[2]);
    if (idx < 0 || idx >= kCells) return 2;
    const bool tensor_only = argc >= 4 && strcmp(argv[3], "tensoronly") == 0;
    dispatch_cell(idx, tensor_only);
    printf("layercheck cell %d (%s): dispatches clean\n", idx, tensor_only ? "tensor-only" : "all");
    return 0;
  }
  printf("== bench_fn8t: the fn8t-1b G2 battery (%d cells) ==\n", kCells);
  Fix F;
  bool fix_built = false;
  for (int i = 0; i < kCells; ++i) {
    Cell& c = cells[i];
    if (c.kind == 0) {
      Dense d;
      dense_build(d, c.N, c.R);
      dense_dispatch(d, true);
      dense_ref(d, c);
    } else {
      if (!fix_built) {
        fix_build(F);
        fix_built = true;
      }
      run_cell(i, F, true);
    }
    // layer_clean: a child process under the debug layer; gpu.mm's commit_and_wait fatals on any error
    const std::string exe = strchr(argv[0], '/') ? std::string(argv[0]) : "./" + std::string(argv[0]);
    const std::string base =
        "MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 '" + exe + "' layercheck " + std::to_string(i);
    c.layer_all = system((base + " >/dev/null").c_str()) == 0;
    if (c.layer_all)
      c.layer = true;
    else
      c.layer = system((base + " tensoronly >/dev/null").c_str()) == 0;
    printf("cell %2d %-36s %-40s hsteps %8.4f  e64 t/s %.3e/%.3e  hi/sg %.3e/%.3e  tg %5u  layer %s\n", i, c.pair,
           c.shape, c.hs, c.e_t, c.e_s, c.e_hilo, c.e_single, c.tg,
           c.layer_all ? "clean(all)" : (c.layer ? "clean(tensor-only)" : "FAILED"));
    fflush(stdout);
  }
  // the gates
  bool all_ok = true;
  for (int i = 0; i < kCells; ++i) {
    const Cell& c = cells[i];
    const bool g1 = c.hs <= 1.0 + 1e-9;
    const bool g2 = std::max(c.e_t, c.e_s) <= 2.0 * std::min(c.e_t, c.e_s) + 1e-12;
    const bool g3 = c.e_hilo <= c.e_single * (1.0 + 1e-9) + 1e-15;
    const bool g4 = c.layer;
    const bool g5 = c.tg <= 16384;
    const bool ok = g1 && g2 && g3 && g4 && g5;
    all_ok &= ok;
    printf("cell %2d gates: half_steps%s fp64_2x%s hilo<=single%s layer%s tg%s -> %s\n", i, g1 ? " ok" : " FAIL",
           g2 ? " ok" : " FAIL", g3 ? " ok" : " FAIL", g4 ? " ok" : " FAIL", g5 ? " ok" : " FAIL",
           ok ? "PASS" : "FAIL");
  }
  // the JSON (FN8T-IMPL-NOTES.md's contract)
  mkdir("/tmp/fn8t-gate", 0777);
  FILE* f = fopen("/tmp/fn8t-gate/g2.json", "w");
  if (!f) fatal("bench_fn8t: cannot write /tmp/fn8t-gate/g2.json");
  fprintf(f, "{\n  \"cells\": [\n");
  for (int i = 0; i < kCells; ++i) {
    const Cell& c = cells[i];
    fprintf(f,
            "    {\"pair\": \"%s\", \"shape\": \"%s\", \"half_steps_max\": %.6g, \"fp64_err_tensor\": %.6g, "
            "\"fp64_err_scalar\": %.6g, \"fp64_err_hilo\": %.6g, \"fp64_err_single\": %.6g, \"layer_clean\": %s, "
            "\"static_tg_bytes\": %u}%s\n",
            c.pair, c.shape, c.hs, c.e_t, c.e_s, c.e_hilo, c.e_single, c.layer ? "true" : "false", c.tg,
            i + 1 < kCells ? "," : "");
  }
  fprintf(f, "  ]\n}\n");
  fclose(f);
  printf("G2 JSON: /tmp/fn8t-gate/g2.json (%d cells)\nOVERALL: %s\n", kCells, all_ok ? "PASS" : "FAIL");
  return all_ok ? 0 : 1;
}
