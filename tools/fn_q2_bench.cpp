// Q2_0 routed experts of Qwen3.8-Flash-Next (src/kernels/zq2moe.metal, dispatched by src/model/moe_q20.h) on synthetic
// weights of the real shapes (hidden 2560, 512 experts of 640, top-10, a 4-bit shared expert of 640).
//   fn_q2_bench check   decode, verify (R = 1, 3, 8, rows sharing experts) and prefill (8-row tiles at T = 5 and 40,
//                       32-row tiles at T = 100; skewed routing, so tiles of 1 to 32 rows occur) against a CPU reference
//                       in double. Q2_0 weights are exact halfs, so the tolerances cover only fp32 sums, the fast-math
//                       exp and, at prefill, the half h between gate/up and down. Then each path again through an arena
//                       of two chunks with shuffled slots: the outputs must equal the stacked path's bit for bit.
//   fn_q2_bench bench   GPU time per call, stacked and through the slot table, beside the same calls on 4-bit experts
//                       (the engine's kernels): decode (1 row), verify (3 and 8 rows), prefill (T = 32 on 8-row tiles,
//                       T = 512 on 32-row tiles). Uniform routing, a new routing per call (64 in rotation, so the experts
//                       come from DRAM); medians of --rounds rounds of 16 calls in one command buffer.
//   fn_q2_bench hash4   the 4-bit MoE kernels the engine runs for Flash-Next (decode, verify, both prefill paths) on fixed
//                       synthetic inputs: a hash of their outputs, for comparing two builds bit for bit. Builds without
//                       src/model/moe_q20.h have only this mode.
//   fn_q2_bench tune    the decode kernels' shapes (values a lane, rows a simdgroup) compiled from zq2moe.metal's macros
//                       next to the library's: GPU time and agreement with the default kernels
//   fn_q2_bench pack DIR [--layer L]   the checks of `check` on layer L of a pack converted by tools/qwen4_exp_q20.py,
//                       its tensors read through Kapitsa's loader with qlin's format checks (shapes and top-k from it)
//   fn_q2_bench store [--layers N] [--passes N]   the streamed-expert store (src/model/expert_store.h) under the kernels
//                       at the small shapes: every layer's output through the store's arena, read from a temporary pack
//                       file mid-pass, against the stacked planes' bit for bit (see run_store)
//   --seed N   --rounds N (default 7)   --small: hidden 256, 64 experts of 128, top-8 (with check: for shader validation,
//   MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1)
// Run it under the GPU lock: lockf -k -t 600 /tmp/kapitsa-gpu.lock build/fn_q2_bench check
#include "metal/gpu.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"
#include "loader/safetensors.h"
#include "nlohmann/json.hpp"
#if __has_include("model/moe_q20.h")
#include "model/expert_store.h"
#include "model/moe_q20.h"
#define FN_Q20 1
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

using namespace ni;

namespace {

// Qwen3.8-Flash-Next's shapes; `pack` mode sets them from the pack it reads
uint32_t D = 2560, I = 640, SI = 640, E = 512, TOPK = 10;
std::mt19937 rng(1234);
int g_rounds = 7;

float urand(float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); }
uint32_t rand_u32() { return uint32_t(rng()); }

Buffer wbuf(size_t bytes, const char* label) { return Gpu::get().alloc(bytes + 16, label, /*tracked=*/false); }
Buffer abuf(size_t bytes, const char* label) {
  Buffer b = Gpu::get().alloc(bytes + 16, label, /*tracked=*/true);
  std::memset(b.ptr(), 0, bytes + 16);
  return b;
}

void parallel_for(size_t n, const std::function<void(size_t)>& f) {
  const unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; ++t)
    th.emplace_back([&, t] {
      for (size_t i = t; i < n; i += nt) f(i);
    });
  for (auto& t : th) t.join();
}

uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  const uint8_t* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

double silu(double v) { return v / (1.0 + std::exp(-v)); }

// ---- 4-bit experts (MLX affine, group 64), stacked [ne][N][K] ----
struct Lin4 {
  Buffer w, s, b;
  uint32_t ne = 0, N = 0, K = 0;
  // the weights of row n of expert e as doubles
  void row(uint32_t e, uint32_t n, double* out) const {
    const uint32_t* wp = w.as<uint32_t>() + (size_t(e) * N + n) * (K / 8);
    const uint16_t* sp = s.as<uint16_t>() + (size_t(e) * N + n) * (K / 64);
    const uint16_t* bp = b.as<uint16_t>() + (size_t(e) * N + n) * (K / 64);
    for (uint32_t k = 0; k < K; ++k)
      out[k] = double(bf16_to_f32(sp[k / 64])) * double((wp[k / 8] >> (4 * (k % 8))) & 15u) + double(bf16_to_f32(bp[k / 64]));
  }
};

Lin4 make4(uint32_t ne, uint32_t N, uint32_t K, float smin, float smax) {
  Lin4 L;
  L.ne = ne; L.N = N; L.K = K;
  const size_t words = size_t(ne) * N * (K / 8), groups = size_t(ne) * N * (K / 64);
  L.w = wbuf(words * 4, "w4");
  L.s = wbuf(groups * 2, "s4");
  L.b = wbuf(groups * 2, "b4");
  uint32_t* w = L.w.as<uint32_t>();
  for (size_t i = 0; i < words; ++i) w[i] = rand_u32();
  uint16_t *s = L.s.as<uint16_t>(), *b = L.b.as<uint16_t>();
  for (size_t i = 0; i < groups; ++i) {
    const float sc = urand(smin, smax);
    s[i] = f32_to_bf16(sc);
    b[i] = f32_to_bf16(-7.5f * sc);  // centred codes
  }
  return L;
}

// ---- Q2_0 experts: codes [E][N][K/16], d [E][N][K/64] half ----
struct Plane2 {
  Buffer c, d;
  uint32_t N = 0, K = 0;
  size_t code_bytes() const { return size_t(N) * K / 4; }   // one expert
  size_t d_bytes() const { return size_t(N) * K / 32; }
  void row(uint32_t e, uint32_t n, double* out) const {
    const uint32_t* cp = c.as<uint32_t>() + (size_t(e) * N + n) * (K / 16);
    const _Float16* dp = d.as<_Float16>() + (size_t(e) * N + n) * (K / 64);
    for (uint32_t k = 0; k < K; ++k) out[k] = (double((cp[k / 16] >> (2 * (k % 16))) & 3u) - 1.0) * double(dp[k / 64]);
  }
};

Plane2 make2(uint32_t N, uint32_t K, float dmin, float dmax) {
  Plane2 P;
  P.N = N; P.K = K;
  const size_t words = size_t(E) * N * (K / 16), groups = size_t(E) * N * (K / 64);
  P.c = wbuf(words * 4, "q20_codes");
  P.d = wbuf(groups * 2, "q20_d");
  uint32_t* c = P.c.as<uint32_t>();
  for (size_t i = 0; i < words; ++i) c[i] = rand_u32();
  _Float16* d = P.d.as<_Float16>();
  for (size_t i = 0; i < groups; ++i) d[i] = _Float16(urand(dmin, dmax) * (rand_u32() & 1 ? 1.0f : -1.0f));
  return P;
}

struct Model {
  Plane2 g2, u2, d2;  // Q2_0 routed experts
  Lin4 sg, su, sd;    // 4-bit shared expert
  Lin4 g4, u4, d4;    // 4-bit routed experts (bench, hash4)
  bool have4 = false, have2 = false;
};

void make_shared(Model& m) {
  m.sg = make4(1, SI, D, 0.002f, 0.006f);
  m.su = make4(1, SI, D, 0.002f, 0.006f);
  m.sd = make4(1, D, SI, 0.004f, 0.012f);
}

#ifdef FN_Q20
// The same experts as slots of an arena of two chunks: slots in a shuffled order, chunks alternating, a one-page gap
// after every 7th slot of a chunk.
struct Arena {
  Buffer a0, a1, slots;
  q20::SlotLayout lay;
};

Arena make_arena(const Model& m) {
  Arena A;
  A.lay = q20::slot_layout(D, I);
  const uint32_t pages = A.lay.bytes / 16384;
  std::vector<uint32_t> order(E);
  std::iota(order.begin(), order.end(), 0u);
  std::shuffle(order.begin(), order.end(), rng);
  std::vector<uint32_t> word(E);
  uint32_t next[2] = {0, 0}, count[2] = {0, 0};
  for (uint32_t i = 0; i < E; ++i) {
    const uint32_t e = order[i], ch = i % 2;
    word[e] = (ch << 28) | next[ch];
    next[ch] += pages + ((++count[ch] % 7) == 0 ? 1 : 0);
  }
  A.a0 = wbuf(size_t(next[0]) * 16384, "arena0");
  A.a1 = wbuf(size_t(next[1]) * 16384, "arena1");
  A.slots = wbuf(E * 4, "slots");
  std::memcpy(A.slots.ptr(), word.data(), E * 4);
  const Plane2* P[3] = {&m.g2, &m.u2, &m.d2};
  const uint32_t off_c[3] = {A.lay.gc, A.lay.uc, A.lay.dc}, off_d[3] = {A.lay.gd, A.lay.ud, A.lay.dd};
  for (uint32_t e = 0; e < E; ++e) {
    uint8_t* base = static_cast<uint8_t*>((word[e] >> 28) ? A.a1.ptr() : A.a0.ptr()) + size_t(word[e] & 0x0FFFFFFFu) * 16384;
    for (int j = 0; j < 3; ++j) {
      std::memcpy(base + off_c[j], static_cast<const uint8_t*>(P[j]->c.ptr()) + e * P[j]->code_bytes(), P[j]->code_bytes());
      std::memcpy(base + off_d[j], static_cast<const uint8_t*>(P[j]->d.ptr()) + e * P[j]->d_bytes(), P[j]->d_bytes());
    }
  }
  return A;
}

q20::Experts stacked(const Model& m) {
  q20::Experts x;
  x.gc = m.g2.c; x.gd = m.g2.d; x.uc = m.u2.c; x.ud = m.u2.d; x.dc = m.d2.c; x.dd = m.d2.d;
  return x;
}

q20::Experts slotted(const Arena& A) {
  q20::Experts x;
  x.arena = true;
  x.a0 = A.a0; x.a1 = A.a1; x.slots = A.slots;
  x.lay = A.lay;
  return x;
}

q20::Shared shared_of(const Model& m) {
  return q20::Shared{m.sg.w, m.sg.s, m.sg.b, m.su.w, m.su.s, m.su.b, m.sd.w, m.sd.s, m.sd.b};
}
#endif

// ---- routings ----
// R rows of top-k: distinct within a row; rows draw from a pool of TOPK + 6 experts so they share many
std::vector<uint32_t> route_rows(int R, bool shared_pool) {
  std::vector<uint32_t> pool(E);
  std::iota(pool.begin(), pool.end(), 0u);
  std::shuffle(pool.begin(), pool.end(), rng);
  const size_t n = shared_pool ? std::min<size_t>(TOPK + 6, E) : E;
  std::vector<uint32_t> idx;
  for (int r = 0; r < R; ++r) {
    std::vector<uint32_t> p(pool.begin(), pool.begin() + long(n));
    std::shuffle(p.begin(), p.end(), rng);
    idx.insert(idx.end(), p.begin(), p.begin() + TOPK);
  }
  return idx;
}

// T tokens of top-k for prefill. skew: slot 0 = expert t % 3, slot 1 = 3 + t % 5 (tiles of many rows), the rest
// distinct from 8 .. E-1; else uniform.
std::vector<uint32_t> route_tokens(int T, bool skew) {
  std::vector<uint32_t> idx;
  for (int t = 0; t < T; ++t) {
    std::vector<uint32_t> row;
    const bool sk = skew && E >= 8 + 2 * TOPK;  // too few experts: uniform
    if (sk) {
      row.push_back(uint32_t(t % 3));
      row.push_back(3u + uint32_t(t % 5));
    }
    while (row.size() < TOPK) {
      const uint32_t e = sk ? 8u + rand_u32() % (E - 8) : rand_u32() % E;
      if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e);
    }
    idx.insert(idx.end(), row.begin(), row.end());
  }
  return idx;
}

std::vector<float> rand_wts(int R) {
  std::vector<float> w(size_t(R) * (TOPK + 1));
  for (auto& v : w) v = urand(0.02f, 0.2f);
  return w;
}

std::vector<float> rand_vec(size_t n, float a) {
  std::vector<float> v(n);
  for (auto& x : v) x = urand(-a, a);
  return v;
}

Buffer upload(const void* p, size_t bytes, const char* label) {
  Buffer b = abuf(bytes, label);
  std::memcpy(b.ptr(), p, bytes);
  return b;
}

// ---- references ----
using RowFn = std::function<void(uint32_t e, uint32_t n, double* out)>;

// rows r of x [R, D] through the experts idx [R, k] (and the shared expert as slot k): y [R, D] += sum_slot
// wts[r, slot] * down(silu(gate x) * up x)
std::vector<double> ref_rows(int R, const std::vector<float>& x, const std::vector<uint32_t>& idx,
                             const std::vector<float>& wts, const RowFn& gate, const RowFn& up, const RowFn& down,
                             const Model& m) {
  // (row, slot) pairs by expert (slot k: the shared expert, keyed E)
  std::map<uint32_t, std::vector<std::pair<int, int>>> by_e;
  for (int r = 0; r < R; ++r) {
    for (uint32_t s = 0; s < TOPK; ++s) by_e[idx[size_t(r) * TOPK + s]].push_back({r, int(s)});
    by_e[E].push_back({r, int(TOPK)});
  }
  std::vector<std::pair<uint32_t, std::vector<std::pair<int, int>>>> items(by_e.begin(), by_e.end());
  std::vector<std::vector<double>> part(items.size(), std::vector<double>(size_t(R) * D, 0.0));
  parallel_for(items.size(), [&](size_t it) {
    const uint32_t e = items[it].first;
    const auto& uses = items[it].second;
    const bool sh = e == E;
    const uint32_t Nh = sh ? SI : I;
    std::vector<double> wg(D), wu(D), wd(Nh), h(uses.size() * Nh);
    for (uint32_t n = 0; n < Nh; ++n) {
      if (sh) {
        m.sg.row(0, n, wg.data());
        m.su.row(0, n, wu.data());
      } else {
        gate(e, n, wg.data());
        up(e, n, wu.data());
      }
      for (size_t j = 0; j < uses.size(); ++j) {
        const float* xr = x.data() + size_t(uses[j].first) * D;
        double g = 0, u = 0;
        for (uint32_t k = 0; k < D; ++k) {
          g += wg[k] * xr[k];
          u += wu[k] * xr[k];
        }
        h[j * Nh + n] = silu(g) * u;
      }
    }
    for (uint32_t d = 0; d < D; ++d) {
      if (sh) m.sd.row(0, d, wd.data());
      else down(e, d, wd.data());
      for (size_t j = 0; j < uses.size(); ++j) {
        double a = 0;
        for (uint32_t n = 0; n < Nh; ++n) a += wd[n] * h[j * Nh + n];
        const int r = uses[j].first, s = uses[j].second;
        part[it][size_t(r) * D + d] += double(wts[size_t(r) * (TOPK + 1) + size_t(s)]) * a;
      }
    }
  });
  std::vector<double> y(size_t(R) * D, 0.0);
  for (auto& p : part)
    for (size_t i = 0; i < y.size(); ++i) y[i] += p[i];
  return y;
}

// prefill pairs p (token p / k, expert idx[p]) of half rows X [T, D]: Y[p] = down(half(silu(gate x) * up x))
std::map<size_t, std::vector<double>> ref_pairs(const std::vector<_Float16>& X, const std::vector<uint32_t>& idx,
                                                const std::vector<size_t>& pairs, const RowFn& gate, const RowFn& up,
                                                const RowFn& down) {
  std::map<uint32_t, std::vector<size_t>> by_e;
  for (size_t p : pairs) by_e[idx[p]].push_back(p);
  std::vector<std::pair<uint32_t, std::vector<size_t>>> items(by_e.begin(), by_e.end());
  std::vector<std::vector<std::vector<double>>> out(items.size());
  parallel_for(items.size(), [&](size_t it) {
    const uint32_t e = items[it].first;
    const auto& ps = items[it].second;
    std::vector<double> wg(D), wu(D), wd(I), h(ps.size() * I);
    for (uint32_t n = 0; n < I; ++n) {
      gate(e, n, wg.data());
      up(e, n, wu.data());
      for (size_t j = 0; j < ps.size(); ++j) {
        const _Float16* xr = X.data() + (ps[j] / TOPK) * D;
        double g = 0, u = 0;
        for (uint32_t k = 0; k < D; ++k) {
          g += wg[k] * double(xr[k]);
          u += wu[k] * double(xr[k]);
        }
        h[j * I + n] = double(_Float16(silu(g) * u));  // the kernels keep h in half
      }
    }
    out[it].assign(ps.size(), std::vector<double>(D));
    for (uint32_t d = 0; d < D; ++d) {
      down(e, d, wd.data());
      for (size_t j = 0; j < ps.size(); ++j) {
        double a = 0;
        for (uint32_t n = 0; n < I; ++n) a += wd[n] * h[j * I + n];
        out[it][j][d] = a;
      }
    }
  });
  std::map<size_t, std::vector<double>> r;
  for (size_t it = 0; it < items.size(); ++it)
    for (size_t j = 0; j < items[it].second.size(); ++j) r[items[it].second[j]] = out[it][j];
  return r;
}

// max |gpu - ref| relative to max |ref| (and the plain maximum)
struct Err {
  double rel = 0, abs = 0, ref_max = 0;
};
Err compare(const float* gpu, const double* ref, size_t n) {
  Err e;
  for (size_t i = 0; i < n; ++i) {
    e.abs = std::max(e.abs, std::fabs(double(gpu[i]) - ref[i]));
    e.ref_max = std::max(e.ref_max, std::fabs(ref[i]));
  }
  e.rel = e.ref_max > 0 ? e.abs / e.ref_max : e.abs;
  return e;
}

int g_fail = 0;
void report(const char* what, const Err& e, double tol) {
  const bool ok = e.rel <= tol && std::isfinite(e.rel);
  if (!ok) ++g_fail;
  std::printf("  %-44s max err %.2e of max |ref| %.3g (rel %.2e, tol %.0e)  %s\n", what, e.abs, e.ref_max, e.rel, tol,
              ok ? "ok" : "FAIL");
}

// ---- the bucketing (moe_bucket3, format-independent) for prefill ----
struct Tiles {
  Buffer idx, yrows, xrows, tiles, ntiles;
  uint32_t P = 0, max_tiles = 0;
};

Tiles bucket(Encoder& e, const std::vector<uint32_t>& idx, int T, uint32_t bm) {
  Tiles t;
  t.P = uint32_t(T) * TOPK;
  t.max_tiles = (t.P + bm - 1) / bm + E;
  t.idx = upload(idx.data(), idx.size() * 4, "ridx");
  t.yrows = abuf(size_t(t.P) * 4, "yrows");
  t.xrows = abuf(size_t(t.P) * 4, "xrows");
  t.tiles = abuf(size_t(t.max_tiles) * 12, "tiles");
  t.ntiles = abuf(16, "ntiles");
  const kp::Bucket2 bp{t.P, TOPK, E, bm, 0u};
  e.use(Gpu::get().pipeline("moe_bucket3"));
  e.buf(0, t.idx); e.buf(1, t.yrows); e.buf(2, t.xrows); e.buf(3, t.tiles); e.buf(4, t.ntiles);
  e.val(5, bp);
  e.dispatch(1, 1, 1, 1024);
  return t;
}

// ---- the engine's 4-bit dispatches (qwen4_exp.cpp enc_moe4 / enc_moe4_rows, qwen35.cpp prefill) ----
void enc4_gate_up_decode(Encoder& e, const Model& m, const Buffer& xn, const Buffer& idx, const Buffer& wts,
                         const Buffer& h) {
  const kp::Moe mp{D, I, SI, TOPK, 0.0f};
  e.use(Gpu::get().pipeline("moe_gate_up3_r2"));
  e.buf(0, m.g4.w); e.buf(1, m.g4.s); e.buf(2, m.g4.b);
  e.buf(3, m.u4.w); e.buf(4, m.u4.s); e.buf(5, m.u4.b);
  e.buf(6, m.sg.w); e.buf(7, m.sg.s); e.buf(8, m.sg.b);
  e.buf(9, m.su.w); e.buf(10, m.su.s); e.buf(11, m.su.b);
  e.buf(12, xn); e.buf(13, xn); e.buf(14, xn); e.buf(15, h);
  e.val(16, mp);
  e.buf(17, idx); e.buf(18, wts); e.val(19, E);
  e.dispatch((I + 15) / 16, TOPK + 1, 1, 256);
}

void enc4_down(Encoder& e, const Model& m, const char* kernel, int R, const Buffer& h, const Buffer& idx,
               const Buffer& wts, const Buffer& y) {
  const kp::Moe mp{D, I, SI, TOPK, 0.0f};
  e.use(Gpu::get().pipeline(kernel));
  e.buf(0, m.d4.w); e.buf(1, m.d4.s); e.buf(2, m.d4.b);
  e.buf(3, m.sd.w); e.buf(4, m.sd.s); e.buf(5, m.sd.b);
  e.buf(6, h); e.buf(7, idx); e.buf(8, wts); e.buf(9, y);
  e.val(10, mp);
  if (R == 0) e.dispatch((D + 15) / 16, 1, 1, 256);  // moe_down2_w
  else e.dispatch(uint32_t(R), (D + 15) / 16, 1, 256);  // moe4_down
}

void enc4_verify(Encoder& e, const Model& m, int R, const Buffer& xn, const Buffer& idx, const Buffer& wts,
                 const Buffer& h, const Buffer& y, const Buffer& ue, const Buffer& umask, const Buffer& uslot,
                 const Buffer& un) {
  const struct {
    uint32_t R, k;
  } rk{uint32_t(R), TOPK};
  e.use(Gpu::get().pipeline("moe4_unique"));
  e.buf(0, idx); e.buf(1, ue); e.buf(2, umask); e.buf(3, uslot); e.buf(4, un);
  e.val(5, rk);
  e.dispatch(1, 1, 1, 32);
  const kp::Moe mp{D, I, SI, TOPK, 0.0f};
  e.use(Gpu::get().pipeline("moe4_gate_up"));
  e.buf(0, m.g4.w); e.buf(1, m.g4.s); e.buf(2, m.g4.b);
  e.buf(3, m.u4.w); e.buf(4, m.u4.s); e.buf(5, m.u4.b);
  e.buf(6, m.sg.w); e.buf(7, m.sg.s); e.buf(8, m.sg.b);
  e.buf(9, m.su.w); e.buf(10, m.su.s); e.buf(11, m.su.b);
  e.buf(12, xn); e.buf(13, h);
  e.val(14, mp);
  e.buf(15, ue); e.buf(16, umask); e.buf(17, uslot); e.buf(18, un);
  e.val(19, uint32_t(R));
  e.dispatch(uint32_t(R), (I + 15) / 16, uint32_t(R) * TOPK + 1, 256);
  enc4_down(e, m, "moe4_down", R, h, idx, wts, y);
}

void enc4_prefill(Encoder& e, const Model& m, const Tiles& t, bool small, const Buffer& xh, const Buffer& H,
                  const Buffer& Y) {
  const kp::Qmm2 pg{I, D, D, I, 64u, I * D / 8, I * D / 64, E + 1};
  const kp::Qmm2 pd{D, I, I, D, 64u, D * I / 8, D * I / 64, E + 1};
  e.use(Gpu::get().pipeline(small ? "qmms_moe2_gu_b4" : "qrm_gu_k32_m32_n64"));
  e.buf(0, xh);
  e.buf(1, m.g4.w); e.buf(2, m.g4.s); e.buf(3, m.g4.b);
  e.buf(4, m.u4.w); e.buf(5, m.u4.s); e.buf(6, m.u4.b);
  e.buf(7, H);
  e.val(8, pg);
  e.buf(9, t.tiles); e.buf(10, t.ntiles); e.buf(11, t.xrows);
  e.buf(12, m.g4.w); e.buf(13, m.g4.s); e.buf(14, m.g4.b);
  e.buf(15, m.u4.w); e.buf(16, m.u4.s); e.buf(17, m.u4.b);
  e.dispatch(small ? I / 32 : I / 64, t.max_tiles, 1, 128);
  e.use(Gpu::get().pipeline(small ? "qmms_moe2_dn_b4" : "qrm_dn_k32_m32_n128"));
  e.buf(0, H);
  e.buf(1, m.d4.w); e.buf(2, m.d4.s); e.buf(3, m.d4.b);
  e.buf(4, Y);
  e.val(5, pd);
  e.buf(6, t.tiles); e.buf(7, t.ntiles); e.buf(8, t.yrows);
  e.buf(9, m.d4.w); e.buf(10, m.d4.s); e.buf(11, m.d4.b);
  e.dispatch(small ? D / 32 : D / 128, t.max_tiles, 1, 128);
}

// scratch of the verify paths (ue, umask, uslot, un) and the activations
struct Scratch {
  Buffer ue, umask, uslot, un, h, xh, H, Y;
  Scratch(int Tmax) {
    ue = abuf(512 * 4, "ue");
    umask = abuf(512 * 4, "umask");
    uslot = abuf(4096, "uslot");
    un = abuf(16, "un");
    h = abuf(size_t(8) * (TOPK + 1) * std::max(I, SI) * 4, "h");
    xh = abuf(size_t(Tmax) * D * 2, "xh");
    H = abuf(size_t(Tmax) * TOPK * I * 2, "H");
    Y = abuf(size_t(Tmax) * TOPK * D * 4, "Y");
  }
};

#ifdef FN_Q20
// ---- check ----
void run_check(const Model& m, const Arena& A) {
  Gpu& g = Gpu::get();
  const q20::Shape shape{D, I, SI, TOPK, E};
  const q20::Shared sh = shared_of(m);
  const q20::Experts xs = stacked(m), xa = slotted(A);
  std::string why;
  if (!q20::shapes_ok(shape, &why)) fatal("shapes: %s", why.c_str());
  const RowFn gate = [&](uint32_t e, uint32_t n, double* o) { m.g2.row(e, n, o); };
  const RowFn up = [&](uint32_t e, uint32_t n, double* o) { m.u2.row(e, n, o); };
  const RowFn down = [&](uint32_t e, uint32_t n, double* o) { m.d2.row(e, n, o); };
  Scratch s(112);

  // decode and verify: y = base + the MoE; both through enc_decode (R = 1) and enc_verify
  for (int R : {1, 3, 8}) {
    for (int mode = 0; mode < (R == 1 ? 2 : 1); ++mode) {  // R = 1: decode, then verify of one row
      const bool decode = R == 1 && mode == 0;
      const std::vector<float> x = rand_vec(size_t(R) * D, 1.0f), base = rand_vec(size_t(R) * D, 0.5f);
      const std::vector<uint32_t> idx = route_rows(R, R > 1);
      const std::vector<float> wts = rand_wts(R);
      // compared: what the MoE added (y - base), so the base does not dilute the error
      const std::vector<double> ref = ref_rows(R, x, idx, wts, gate, up, down, m);
      const Buffer xb = upload(x.data(), x.size() * 4, "x"), ib = upload(idx.data(), idx.size() * 4, "idx"),
                   wb = upload(wts.data(), wts.size() * 4, "wts");
      std::vector<float> out[2];
      for (int a = 0; a < 2; ++a) {
        const Buffer yb = upload(base.data(), base.size() * 4, "y");
        auto e = g.begin();
        if (decode) q20::enc_decode(*e, a ? xa : xs, sh, shape, xb, ib, wb, s.h, yb);
        else q20::enc_verify(*e, a ? xa : xs, sh, shape, R, xb, ib, wb, s.h, yb, s.ue, s.umask, s.uslot, s.un);
        e->commit();
        e->wait();
        out[a].assign(yb.as<float>(), yb.as<float>() + size_t(R) * D);
      }
      std::vector<float> added(out[0].size());
      for (size_t i = 0; i < added.size(); ++i) added[i] = float(double(out[0][i]) - double(base[i]));
      char what[96];
      std::snprintf(what, sizeof what, "%s, %d row%s (stacked)", decode ? "decode" : "verify", R, R > 1 ? "s" : "");
      report(what, compare(added.data(), ref.data(), ref.size()), 2e-4);
      const bool same = std::memcmp(out[0].data(), out[1].data(), out[0].size() * 4) == 0;
      if (!same) ++g_fail;
      std::printf("  %-44s %s\n", "  through the slot table", same ? "bit for bit equal" : "DIFFERS");
    }
  }

  // prefill: 8-row tiles at T = 5 and 40, 32-row tiles at T = 100
  for (auto [T, bm] : {std::pair<int, uint32_t>{5, 8}, {40, 8}, {100, 32}}) {
    std::vector<_Float16> X(size_t(T) * D);
    for (auto& v : X) v = _Float16(urand(-1.0f, 1.0f));
    const std::vector<uint32_t> idx = route_tokens(T, true);
    std::memcpy(s.xh.ptr(), X.data(), X.size() * 2);
    const uint32_t P = uint32_t(T) * TOPK;
    // pairs checked: all when few, else every expert's first and last pair plus a random set
    std::vector<size_t> pairs;
    if (P <= 160) {
      for (size_t p = 0; p < P; ++p) pairs.push_back(p);
    } else {
      std::map<uint32_t, std::pair<size_t, size_t>> fl;
      for (size_t p = 0; p < P; ++p) {
        auto it = fl.find(idx[p]);
        if (it == fl.end()) fl[idx[p]] = {p, p};
        else it->second.second = p;
      }
      for (auto& [e, f] : fl)
        if (e < 8) {
          pairs.push_back(f.first);
          pairs.push_back(f.second);
        }
      while (pairs.size() < 120) pairs.push_back(rand_u32() % P);
      std::sort(pairs.begin(), pairs.end());
      pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    }
    const auto ref = ref_pairs(X, idx, pairs, gate, up, down);
    std::vector<float> out[2];
    uint32_t ntiles = 0, rows_max = 0;
    for (int a = 0; a < 2; ++a) {
      std::memset(s.Y.ptr(), 0, size_t(P) * D * 4);
      auto e = g.begin();
      const Tiles t = bucket(*e, idx, T, bm);
      q20::enc_prefill(*e, a ? xa : xs, shape, int(bm), t.max_tiles, s.xh, s.H, s.Y, t.tiles, t.ntiles, t.xrows,
                       t.yrows);
      e->commit();
      e->wait();
      out[a].assign(s.Y.as<float>(), s.Y.as<float>() + size_t(P) * D);
      ntiles = t.ntiles.as<uint32_t>()[0];
      for (uint32_t i = 0; i < ntiles; ++i)
        rows_max = std::max(rows_max, t.tiles.as<uint32_t>()[i * 3 + 2] - t.tiles.as<uint32_t>()[i * 3 + 1]);
    }
    Err worst;
    for (auto& [p, r] : ref) {
      const Err e = compare(out[0].data() + p * D, r.data(), D);
      if (e.rel > worst.rel) worst = e;
    }
    char what[96];
    std::snprintf(what, sizeof what, "prefill T=%d, %u-row tiles (%u tiles, <= %u rows), %zu pairs", T, bm, ntiles,
                  rows_max, ref.size());
    report(what, worst, 4e-3);
    const bool same = std::memcmp(out[0].data(), out[1].data(), out[0].size() * 4) == 0;
    if (!same) ++g_fail;
    std::printf("  %-44s %s\n", "  through the slot table", same ? "bit for bit equal" : "DIFFERS");
  }
}

// ---- bench ----
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

void run_bench(const Model& m, const Arena& A) {
  Gpu& g = Gpu::get();
  const q20::Shape shape{D, I, SI, TOPK, E};
  const q20::Shared sh = shared_of(m);
  const q20::Experts xs = stacked(m), xa = slotted(A);
  constexpr int NROUT = 64, CALLS = 16;
  Scratch s(512);
  // routings and inputs in rotation
  std::vector<Buffer> idx1, wts1, x1, idx3, wts3, x3, idx8, wts8, x8;
  for (int i = 0; i < NROUT; ++i) {
    auto put = [&](int R, std::vector<Buffer>& ib, std::vector<Buffer>& wb, std::vector<Buffer>& xb) {
      const auto idx = route_rows(R, false);
      const auto w = rand_wts(R);
      const auto x = rand_vec(size_t(R) * D, 1.0f);
      ib.push_back(upload(idx.data(), idx.size() * 4, "idx"));
      wb.push_back(upload(w.data(), w.size() * 4, "wts"));
      xb.push_back(upload(x.data(), x.size() * 4, "x"));
    };
    put(1, idx1, wts1, x1);
    put(3, idx3, wts3, x3);
    put(8, idx8, wts8, x8);
  }
  const Buffer y = abuf(size_t(8) * D * 4, "y");
  std::vector<_Float16> X(size_t(512) * D);
  for (auto& v : X) v = _Float16(urand(-1.0f, 1.0f));
  std::memcpy(s.xh.ptr(), X.data(), X.size() * 2);

  // ms per call: rounds of CALLS calls in one command buffer
  auto time = [&](const std::function<void(Encoder&, int)>& call) {
    {
      auto e = g.begin();
      for (int c = 0; c < 4; ++c) call(*e, c);
      e->commit();
      e->wait();
    }
    std::vector<double> ms;
    int k = 0;
    for (int r = 0; r < g_rounds; ++r) {
      auto e = g.begin();
      for (int c = 0; c < CALLS; ++c) call(*e, k++);
      e->commit();
      e->wait();
      ms.push_back(e->gpu_ms() / CALLS);
    }
    return median(ms);
  };
  const double slot_mb = q20::slot_layout(D, I).bytes / 1e6;
  const double q20_e = (size_t(3) * I * D / 4 + size_t(3) * I * D / 32) / 1e6;     // one expert, MB
  const double q4_e = (size_t(3) * I * D / 2 + size_t(3) * I * D / 64 * 4) / 1e6;   // codes + scales + biases
  const double sh_mb = (size_t(3) * SI * D / 2 + size_t(3) * SI * D / 64 * 4) / 1e6;
  std::printf("one expert: Q2_0 %.3f MB (slot %.3f MB), 4-bit %.3f MB; the shared expert %.3f MB\n", q20_e, slot_mb,
              q4_e, sh_mb);
  std::printf("%-34s %10s %10s %10s %9s\n", "call", "Q2_0", "Q2_0 slot", "4-bit", "Q2_0 GB/s");
  auto row = [&](const char* what, double mb2, double t2, double t2s, double t4) {
    std::printf("%-34s %8.3f ms %8.3f ms %8.3f ms %9.0f\n", what, t2, t2s, t4, mb2 / t2);
  };
  auto distinct = [&](const std::vector<Buffer>& ib, int R) {
    double n = 0;
    for (const Buffer& b : ib) {
      std::vector<uint32_t> v(b.as<uint32_t>(), b.as<uint32_t>() + size_t(R) * TOPK);
      std::sort(v.begin(), v.end());
      n += double(std::unique(v.begin(), v.end()) - v.begin());
    }
    return n / double(ib.size());
  };
  // decode
  {
    auto q2 = [&](const q20::Experts& x) {
      return time([&](Encoder& e, int c) {
        q20::enc_decode(e, x, sh, shape, x1[c % NROUT], idx1[c % NROUT], wts1[c % NROUT], s.h, y);
      });
    };
    const double t4 = m.have4 ? time([&](Encoder& e, int c) {
      enc4_gate_up_decode(e, m, x1[c % NROUT], idx1[c % NROUT], wts1[c % NROUT], s.h);
      enc4_down(e, m, "moe_down2_w", 0, s.h, idx1[c % NROUT], wts1[c % NROUT], y);
    }) : NAN;
    row("decode, 1 row", TOPK * q20_e + sh_mb, q2(xs), q2(xa), t4);
  }
  // verify
  for (int R : {3, 8}) {
    auto& ib = R == 3 ? idx3 : idx8;
    auto& wb = R == 3 ? wts3 : wts8;
    auto& xb = R == 3 ? x3 : x8;
    auto q2 = [&](const q20::Experts& x) {
      return time([&](Encoder& e, int c) {
        q20::enc_verify(e, x, sh, shape, R, xb[c % NROUT], ib[c % NROUT], wb[c % NROUT], s.h, y, s.ue, s.umask,
                        s.uslot, s.un);
      });
    };
    const double t4 = m.have4 && R <= 4 ? time([&](Encoder& e, int c) {
      enc4_verify(e, m, R, xb[c % NROUT], ib[c % NROUT], wb[c % NROUT], s.h, y, s.ue, s.umask, s.uslot, s.un);
    }) : NAN;
    char what[64];
    std::snprintf(what, sizeof what, "verify, %d rows (%.1f experts)", R, distinct(ib, R));
    row(what, distinct(ib, R) * q20_e + sh_mb, q2(xs), q2(xa), t4);
  }
  // prefill: the bucketing once per routing (outside the timing), then the expert GEMMs
  for (auto [T, bm] : {std::pair<int, uint32_t>{32, 8}, {512, 32}}) {
    std::vector<Tiles> tl;
    {
      auto e = g.begin();
      for (int i = 0; i < 4; ++i) tl.push_back(bucket(*e, route_tokens(T, false), T, bm));
      e->commit();
      e->wait();
    }
    auto q2 = [&](const q20::Experts& x) {
      return time([&](Encoder& e, int c) {
        const Tiles& t = tl[size_t(c) % tl.size()];
        q20::enc_prefill(e, x, shape, int(bm), t.max_tiles, s.xh, s.H, s.Y, t.tiles, t.ntiles, t.xrows, t.yrows);
      });
    };
    const double t4 = m.have4 ? time([&](Encoder& e, int c) {
      const Tiles& t = tl[size_t(c) % tl.size()];
      enc4_prefill(e, m, t, bm == 8, s.xh, s.H, s.Y);
    }) : NAN;
    double used = 0;
    for (const Tiles& t : tl) {
      std::vector<uint32_t> v(t.idx.as<uint32_t>(), t.idx.as<uint32_t>() + t.P);
      std::sort(v.begin(), v.end());
      used += double(std::unique(v.begin(), v.end()) - v.begin());
    }
    used /= double(tl.size());
    const double t2 = q2(xs), t2s = q2(xa);
    char what[64];
    std::snprintf(what, sizeof what, "prefill T=%d, %u-row tiles", T, bm);
    row(what, used * q20_e, t2, t2s, t4);
    std::printf("%-34s %8.0f tok/s %6.0f tok/s %6.0f tok/s  (the expert GEMMs alone, one layer)\n", "", T / t2 * 1e3,
                T / t2s * 1e3, T / t4 * 1e3);
  }
}
#endif

#ifdef FN_Q20
// ---- tune: the decode kernels' shapes (VPL values a lane, ROWS rows a simdgroup), compiled from zq2moe.metal's macros ----
const char* kTuneSrc = R"MSL(
NI_Q20_GU(tq_gu_v16_r2, false, 16, 2)
NI_Q20_GU(tq_gu_v32_r2, false, 32, 2)
NI_Q20_GU(tq_gu_v32_r4, false, 32, 4)
NI_Q20_GU(tq_gu_v64_r2, false, 64, 2)
NI_Q20_DN(tq_dn_v16_r2, false, 16, 2)
NI_Q20_DN(tq_dn_v32_r2, false, 32, 2)
NI_Q20_DN(tq_dn_v32_r4, false, 32, 4)
NI_Q20_DN(tq_dn_v16_r8, false, 16, 8)
NI_Q20_GU(tq_gu_v16_r1, false, 16, 1)
)MSL";

void run_tune(const Model& m) {
  Gpu& g = Gpu::get();
  constexpr int NROUT = 64, CALLS = 16;
  const q20::Experts x = stacked(m);
  const kp::Moe mp{D, I, SI, TOPK, 0.0f};
  const q20::SlotLayout lay = q20::slot_layout(D, I);
  const struct {
    uint32_t gc, gd, uc, ud, dc, dd;
  } L{lay.gc, lay.gd, lay.uc, lay.ud, lay.dc, lay.dd};
  std::vector<Buffer> ib, wb, xb;
  for (int i = 0; i < NROUT; ++i) {
    const auto idx = route_rows(1, false);
    const auto w = rand_wts(1);
    const auto xv = rand_vec(D, 1.0f);
    ib.push_back(upload(idx.data(), idx.size() * 4, "idx"));
    wb.push_back(upload(w.data(), w.size() * 4, "wts"));
    xb.push_back(upload(xv.data(), xv.size() * 4, "x"));
  }
  const Buffer h = abuf(size_t(TOPK + 1) * I * 4, "h"), y = abuf(size_t(D) * 4, "y");
  auto gu = [&](Encoder& e, const char* k, int rows, int c) {
    e.use(g.pipeline(k));
    e.buf(0, x.gc); e.buf(1, x.gd); e.buf(2, x.uc); e.buf(3, x.ud);
    e.buf(4, m.sg.w); e.buf(5, m.sg.s); e.buf(6, m.sg.b); e.buf(7, m.su.w); e.buf(8, m.su.s); e.buf(9, m.su.b);
    e.buf(10, xb[size_t(c) % NROUT]); e.buf(11, h);
    e.val(12, mp);
    e.buf(13, ib[size_t(c) % NROUT]);
    e.val(14, L);
    e.dispatch((I + 8 * rows - 1) / (8 * rows), TOPK + 1, 1, 256);
  };
  auto dn = [&](Encoder& e, const char* k, int rows, int c) {
    e.use(g.pipeline(k));
    e.buf(0, x.dc); e.buf(1, x.dd); e.buf(2, x.dc);
    e.buf(3, m.sd.w); e.buf(4, m.sd.s); e.buf(5, m.sd.b);
    e.buf(6, h); e.buf(7, ib[size_t(c) % NROUT]); e.buf(8, wb[size_t(c) % NROUT]); e.buf(9, y);
    e.val(10, mp);
    e.val(11, L);
    e.tgmem(0, size_t(TOPK + 1) * std::max(I, SI) * 4);
    e.dispatch(1, (D + 8 * rows - 1) / (8 * rows), 1, 256);
  };
  auto time = [&](const std::function<void(Encoder&, int)>& call) {
    std::vector<double> ms;
    for (int r = 0; r < g_rounds + 1; ++r) {
      auto e = g.begin();
      for (int c = 0; c < CALLS; ++c) call(*e, r * CALLS + c);
      e->commit();
      e->wait();
      if (r) ms.push_back(e->gpu_ms() / CALLS);
    }
    return median(ms);
  };
  // h and y of the default kernels on routing 0, for the variants' agreement
  auto outputs = [&](const char* gk, int gr, const char* dk, int dr) {
    std::memset(y.ptr(), 0, size_t(D) * 4);
    auto e = g.begin();
    gu(*e, gk, gr, 0);
    dn(*e, dk, dr, 0);
    e->commit();
    e->wait();
    return std::make_pair(std::vector<float>(h.as<float>(), h.as<float>() + size_t(TOPK + 1) * I),
                          std::vector<float>(y.as<float>(), y.as<float>() + D));
  };
  const auto ref = outputs("moe_q20_gate_up", 4, "moe_q20_down", 4);
  auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
    double mx = 0, e = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      mx = std::max(mx, std::fabs(double(b[i])));
      e = std::max(e, std::fabs(double(a[i]) - double(b[i])));
    }
    return mx > 0 ? e / mx : e;
  };
  // variants round-robin within each round (the GPU's clock and the machine's state shift between rounds), and the
  // whole decode (gate/up + down) for the shapes that win
  struct V {
    std::string name;
    std::function<void(Encoder&, int)> call;
    double mb;
    std::vector<double> ms;
  };
  std::vector<V> vs;
  for (auto [k, r] : {std::pair<const char*, int>{"moe_q20_gate_up", 4}, {"tq_gu_v16_r2", 2}, {"tq_gu_v32_r2", 2},
                      {"tq_gu_v32_r4", 4}, {"tq_gu_v16_r1", 1}})
    vs.push_back({std::string("gate/up ") + k, [&, k = k, r = r](Encoder& e, int c) { gu(e, k, r, c); }, 11.06, {}});
  for (auto [k, r] : {std::pair<const char*, int>{"moe_q20_down", 4}, {"tq_dn_v16_r2", 2}, {"tq_dn_v32_r4", 4},
                      {"tq_dn_v16_r8", 8}})
    vs.push_back({std::string("down ") + k, [&, k = k, r = r](Encoder& e, int c) { dn(e, k, r, c); }, 5.53, {}});
  vs.push_back({"decode (the library's: rows 4)", [&](Encoder& e, int c) {
                  gu(e, "moe_q20_gate_up", 4, c);
                  dn(e, "moe_q20_down", 4, c);
                }, 16.59, {}});
  vs.push_back({"decode, rows 2", [&](Encoder& e, int c) {
                  gu(e, "tq_gu_v16_r2", 2, c);
                  dn(e, "tq_dn_v16_r2", 2, c);
                }, 16.59, {}});
  for (int r = 0; r < g_rounds + 1; ++r)
    for (V& v : vs) {
      auto e = g.begin();
      for (int c = 0; c < CALLS; ++c) v.call(*e, r * CALLS + c);
      e->commit();
      e->wait();
      if (r) v.ms.push_back(e->gpu_ms() / CALLS);
    }
  for (V& v : vs) {
    const double t = median(v.ms);
    std::printf("  %-34s %7.1f us  %5.0f GB/s  (p10-p90 %.1f-%.1f)\n", v.name.c_str(), t * 1e3, v.mb / t,
                [&] { auto w = v.ms; std::sort(w.begin(), w.end()); return w[w.size() / 10] * 1e3; }(),
                [&] { auto w = v.ms; std::sort(w.begin(), w.end()); return w[w.size() * 9 / 10] * 1e3; }());
  }
  // agreement with the default kernels (routing 0)
  for (auto [k, r] : {std::pair<const char*, int>{"tq_gu_v16_r2", 2}, {"tq_gu_v32_r4", 4}})
    std::printf("  %-34s h vs the default %.1e\n", k, rel(outputs(k, r, "moe_q20_down", 4).first, ref.first));
  for (auto [k, r] : {std::pair<const char*, int>{"tq_dn_v16_r2", 2}, {"tq_dn_v16_r8", 8}})
    std::printf("  %-34s y vs the default %.1e\n", k, rel(outputs("moe_q20_gate_up", 4, k, r).second, ref.second));
}
#endif

#ifdef FN_Q20
// ---- pack: layer L of a converted pack (tools/qwen4_exp_q20.py) through Kapitsa's loader, and the checks of `check` ----
// The experts as the engine's qlin reads them (config mode q2_0, U32 codes [E, N, K/16], F16 scales [E, N, K/64], no
// biases), the 4-bit shared expert; the shapes and top-k from the pack.
void run_pack(const std::string& dir, int layer) {
  using json = nlohmann::json;
  const json cfg = json::parse(std::ifstream(dir + "/config.json"));
  const json idx = json::parse(std::ifstream(dir + "/model.safetensors.index.json"))["weight_map"];
  const std::string pre = "language_model.model.layers." + std::to_string(layer) + ".mlp.";
  const std::string file = idx.at(pre + "switch_mlp.gate_proj.weight").get<std::string>();
  WeightStore ws;
  ws.open_dir(dir, [&](const std::string& n) { return n.rfind(pre, 0) == 0; },
              [&](const std::string& f) { return f == file; });
  auto q2 = [&](const char* mat) {
    const std::string base = pre + "switch_mlp." + mat + "_proj";
    const auto& q = cfg["quantization"];
    if (!q.contains(base) || q[base].value("mode", "") != "q2_0" || q[base].value("bits", 0) != 2 ||
        q[base].value("group_size", 0) != 64)
      fatal("%s: config.json has no q2_0 entry", base.c_str());
    const Tensor &w = ws.get(base + ".weight"), &sc = ws.get(base + ".scales");
    if (ws.find(base + ".biases")) fatal("%s: Q2_0 has no biases", base.c_str());
    if (w.dtype != DType::U32 || sc.dtype != DType::F16 || w.shape.size() != 3 || sc.shape.size() != 3)
      fatal("%s: not uint32 codes and fp16 scales [E, N, ...]", base.c_str());
    Plane2 p;
    p.c = w.buf;
    p.d = sc.buf;
    p.N = uint32_t(w.shape[1]);
    p.K = uint32_t(w.shape[2] * 16);
    if (sc.shape[2] * 64 != int64_t(p.K) || sc.shape[0] != w.shape[0] || sc.shape[1] != w.shape[1])
      fatal("%s: scales [%lld, %lld, %lld] do not match the codes", base.c_str(), (long long)sc.shape[0],
            (long long)sc.shape[1], (long long)sc.shape[2]);
    E = uint32_t(w.shape[0]);
    return p;
  };
  auto l4 = [&](const char* mat) {
    const std::string base = pre + "shared_expert." + mat + "_proj";
    const Tensor &w = ws.get(base + ".weight"), &sc = ws.get(base + ".scales"), &b = ws.get(base + ".biases");
    Lin4 L;
    L.w = w.buf;
    L.s = sc.buf;
    L.b = b.buf;
    L.ne = 1;
    L.N = uint32_t(w.shape[0]);
    L.K = uint32_t(w.shape[1] * 8);
    if (w.dtype != DType::U32 || sc.shape[1] * 64 != int64_t(L.K)) fatal("%s: not 4-bit group 64", base.c_str());
    return L;
  };
  Model m;
  m.g2 = q2("gate");
  m.u2 = q2("up");
  m.d2 = q2("down");
  m.sg = l4("gate");
  m.su = l4("up");
  m.sd = l4("down");
  m.have2 = true;
  D = m.g2.K;
  I = m.g2.N;
  SI = m.sg.N;
  const json& tc = cfg.contains("text_config") ? cfg["text_config"] : cfg;
  TOPK = uint32_t(tc.value("num_experts_per_tok", 0));
  if (TOPK == 0) TOPK = std::min(10u, E / 2);
  if (m.u2.N != I || m.u2.K != D || m.d2.N != D || m.d2.K != I) fatal("layer %d: gate/up/down shapes disagree", layer);
  std::printf("pack %s, layer %d (%s): %u experts of [%u, %u] Q2_0, shared expert %u 4-bit, top-%u\n", dir.c_str(),
              layer, file.c_str(), E, I, D, SI, TOPK);
  const Arena A = make_arena(m);
  run_check(m, A);
}

// ---- store: the streamed-expert store (src/model/expert_store.h) under the kernels ----
// Small shapes (the --small ones). The routed experts of `layers` layers are written to a temporary pack file (stacked
// planes, some at odd offsets) and served by an ExpertStore whose arena holds 3 E of the layers * E experts, so turns
// read and evict. Each pass (decode rows, verify windows of 3 and 8 rows, prefill chunks on 8- and 32-row tiles) runs
// every layer in one command buffer twice: on the stacked planes, then through the store, its routing copied on the GPU
// into one ids buffer just before each layer's handshake (as the router's top-k writes it), with next-layer predictions
// (decode / verify: the true ids with two of them wrong; prefill chunks of 64+ rows: every expert of the next layer).
// Every layer's output must be the stacked one's bit for bit.
void run_store(int layers, int passes) {
  Gpu& g = Gpu::get();
  const q20::Shape shape{D, I, SI, TOPK, E};
  std::string why;
  if (!q20::shapes_ok(shape, &why)) fatal("shapes: %s", why.c_str());
  Model m;
  make_shared(m);
  const q20::Shared sh = shared_of(m);
  std::vector<Model> lm(static_cast<size_t>(layers));
  for (Model& x : lm) {
    x.g2 = make2(I, D, 0.004f, 0.012f);
    x.u2 = make2(I, D, 0.004f, 0.012f);
    x.d2 = make2(D, I, 0.008f, 0.024f);
  }
  // the pack file: layer by layer, the six planes; odd layers' planes at odd offsets (widened or unaligned reads)
  const char* td = getenv("TMPDIR");
  std::string path = std::string(td && *td ? td : "/tmp") + "/fn_q2_store.XXXXXX";
  std::vector<char> pb(path.begin(), path.end());
  pb.push_back('\0');
  const int fd = mkstemp(pb.data());
  if (fd < 0) fatal("mkstemp failed");
  path = pb.data();
  std::vector<q20::LayerSrc> src(static_cast<size_t>(layers));
  uint64_t off = 16384;
  for (int l = 0; l < layers; ++l) {
    const Plane2* P[3] = {&lm[size_t(l)].g2, &lm[size_t(l)].u2, &lm[size_t(l)].d2};
    for (int j = 0; j < 6; ++j) {
      const Plane2& pl = *P[j / 2];
      const Buffer& b = j % 2 ? pl.d : pl.c;
      const size_t per = j % 2 ? pl.d_bytes() : pl.code_bytes();
      off = l % 2 ? off + 37 : (off + 16383) / 16384 * 16384;
      q20::PlaneSrc& ps = src[size_t(l)].p[j];
      ps.file = 0;
      ps.off = off;
      ps.stride = per;
      ps.len = uint32_t(per);
      if (pwrite(fd, b.ptr(), per * E, off_t(off)) != ssize_t(per * E)) fatal("writing %s failed", path.c_str());
      off += per * E;
    }
  }
  close(fd);
  const q20::SlotLayout lay = q20::slot_layout(D, I);
  q20::ExpertStore::Options o;
  o.bytes = size_t(3) * E * lay.bytes;
  o.cache.readers = 8;
  q20::ExpertStore store({path}, src, E, lay, o);
  std::printf("store: %d layers of %u experts [%u, %u] Q2_0 (top-%u), %u slots in the arena\n", layers, E, I, D, TOPK,
              store.slots());
  Scratch s(112);
  const Buffer ids = abuf(size_t(128) * TOPK * 4, "ids"), pids = abuf(size_t(8) * TOPK * 4, "pred_ids");
  auto gpu_copy = [&](Encoder& e, const Buffer& from, const Buffer& to, size_t bytes) {
    static Pipeline cp = g.pipeline("copy_u32");
    const uint32_t n = uint32_t((bytes + 15) / 16);
    e.use(cp);
    e.buf(0, from);
    e.buf(1, to);
    e.val(2, n);
    e.dispatch((n + 255) / 256, 1, 1, 256);
  };
  auto padded = [&](const std::vector<uint32_t>& v, const char* label) {
    std::vector<uint32_t> p(v);
    p.resize((p.size() + 3) / 4 * 4, 0u);
    return upload(p.data(), p.size() * 4, label);
  };
  int bad = 0, cases = 0;
  for (int pass = 0; pass < passes; ++pass) {
    // decode and verify windows: rows R through every layer, then prefill chunks of T rows
    for (int R : {1, 3, 8}) {
      std::vector<Buffer> xb, ib, wb, ybs, ybr, pidb;
      std::vector<std::vector<uint32_t>> idx(static_cast<size_t>(layers));
      for (int l = 0; l < layers; ++l) idx[size_t(l)] = route_rows(R, R > 1);
      for (int l = 0; l < layers; ++l) {
        const std::vector<float> x = rand_vec(size_t(R) * D, 1.0f), base = rand_vec(size_t(R) * D, 0.5f);
        const std::vector<float> wts = rand_wts(R);
        xb.push_back(upload(x.data(), x.size() * 4, "x"));
        ib.push_back(padded(idx[size_t(l)], "idx"));
        wb.push_back(upload(wts.data(), wts.size() * 4, "wts"));
        ybs.push_back(upload(base.data(), base.size() * 4, "y_stacked"));
        ybr.push_back(upload(base.data(), base.size() * 4, "y_store"));
        std::vector<uint32_t> pred = l + 1 < layers ? idx[size_t(l + 1)] : std::vector<uint32_t>(size_t(R) * TOPK, 0u);
        for (int j = 0; j < 2; ++j) pred[size_t(rand_u32() % pred.size())] = rand_u32() % E;
        pidb.push_back(padded(pred, "pred"));
      }
      for (int a = 0; a < 2; ++a) {
        auto e = g.begin();
        for (int l = 0; l < layers; ++l) {
          const bool st = a == 1;
          q20::Experts x = stacked(lm[size_t(l)]);
          const Buffer& rid = st ? ids : ib[size_t(l)];
          if (st) {
            gpu_copy(*e, ib[size_t(l)], ids, size_t(R) * TOPK * 4);
            gpu_copy(*e, pidb[size_t(l)], pids, size_t(R) * TOPK * 4);
            store.begin(*e, l, ids, uint32_t(R) * TOPK, &pids, uint32_t(R) * TOPK, l + 1 < layers ? l + 1 : -1, false,
                        false);
            store.end(*e);
            x = store.experts(l);
          }
          const Buffer& yb = st ? ybr[size_t(l)] : ybs[size_t(l)];
          if (R == 1 && pass % 2 == 0) q20::enc_decode(*e, x, sh, shape, xb[size_t(l)], rid, wb[size_t(l)], s.h, yb);
          else q20::enc_verify(*e, x, sh, shape, R, xb[size_t(l)], rid, wb[size_t(l)], s.h, yb, s.ue, s.umask, s.uslot, s.un);
        }
        e->commit();
        e->wait();
      }
      for (int l = 0; l < layers; ++l) {
        ++cases;
        if (std::memcmp(ybs[size_t(l)].ptr(), ybr[size_t(l)].ptr(), size_t(R) * D * 4)) ++bad;
      }
    }
    for (auto [T, bm] : {std::pair<int, uint32_t>{5, 8}, {40, 8}, {100, 32}}) {
      const uint32_t P = uint32_t(T) * TOPK;
      std::vector<Buffer> xh, ys, yr;
      std::vector<std::vector<uint32_t>> idx;
      for (int l = 0; l < layers; ++l) {
        std::vector<_Float16> X(size_t(T) * D);
        for (auto& v : X) v = _Float16(urand(-1.0f, 1.0f));
        xh.push_back(upload(X.data(), X.size() * 2, "xh"));
        idx.push_back(route_tokens(T, true));
        ys.push_back(abuf(size_t(P) * D * 4, "Y_stacked"));
        yr.push_back(abuf(size_t(P) * D * 4, "Y_store"));
      }
      for (int a = 0; a < 2; ++a) {
        auto e = g.begin();
        for (int l = 0; l < layers; ++l) {
          const bool st = a == 1;
          const Tiles t = bucket(*e, idx[size_t(l)], T, bm);  // t.idx: the routing the GPU reads
          q20::Experts x = stacked(lm[size_t(l)]);
          if (st) {
            gpu_copy(*e, t.idx, ids, size_t(P) * 4);
            const bool all = T >= 64 && l + 1 < layers;
            store.begin(*e, l, ids, P, nullptr, 0, all ? l + 1 : -1, all, T > 16);
            store.end(*e);
            x = store.experts(l);
          }
          q20::enc_prefill(*e, x, shape, int(bm), t.max_tiles, xh[size_t(l)], s.H, st ? yr[size_t(l)] : ys[size_t(l)],
                           t.tiles, t.ntiles, t.xrows, t.yrows);
        }
        e->commit();
        e->wait();
      }
      for (int l = 0; l < layers; ++l) {
        ++cases;
        if (std::memcmp(ys[size_t(l)].ptr(), yr[size_t(l)].ptr(), size_t(P) * D * 4)) ++bad;
      }
    }
  }
  const q20::StoreStats st = store.stats();
  unlink(path.c_str());
  std::printf("  %d layer outputs through the store (decode, verify 3 / 8 rows, prefill T = 5 / 40 / 100): %d differ "
              "from the stacked planes'\n",
              cases, bad);
  std::printf("  %llu turns: %llu lookups, %llu hits, %llu read ahead and waited for, %llu misses; %llu read ahead (%llu "
              "routed), %llu evictions, %.1f MB read; stall %.3f ms a turn (reads %.3f)\n",
              (unsigned long long)st.handshakes, (unsigned long long)st.lookups, (unsigned long long)st.hits,
              (unsigned long long)st.late, (unsigned long long)st.misses, (unsigned long long)st.prefetches,
              (unsigned long long)st.prefetch_used, (unsigned long long)st.evictions, st.bytes / 1e6,
              st.stall_ms / double(std::max<uint64_t>(st.handshakes, 1)),
              st.read_wait_ms / double(std::max<uint64_t>(st.handshakes, 1)));
  if (bad || st.evictions == 0 || st.misses == 0) ++g_fail;
}
#endif

// ---- hash4 ----
void run_hash4(const Model& m) {
  Gpu& g = Gpu::get();
  Scratch s(112);
  uint64_t all = 1469598103934665603ull;
  // the hash, and the largest output (minus the base y for decode / verify: what the kernels added)
  auto show = [&](const char* what, const void* p, size_t n, const float* base = nullptr) {
    const uint64_t h = fnv(p, n);
    all = fnv(&h, 8, all);
    double mx = 0;
    for (size_t i = 0; i < n / 4; ++i) mx = std::max(mx, std::fabs(double(static_cast<const float*>(p)[i]) - (base ? base[i] : 0.0)));
    std::printf("  %-40s %016llx  (max |added| %.3g)\n", what, (unsigned long long)h, mx);
  };
  {
    const std::vector<float> x = rand_vec(D, 1.0f), base = rand_vec(D, 0.5f), wts = rand_wts(1);
    const std::vector<uint32_t> idx = route_rows(1, false);
    const Buffer xb = upload(x.data(), x.size() * 4, "x"), ib = upload(idx.data(), idx.size() * 4, "idx"),
                 wb = upload(wts.data(), wts.size() * 4, "wts"), yb = upload(base.data(), base.size() * 4, "y");
    auto e = g.begin();
    enc4_gate_up_decode(*e, m, xb, ib, wb, s.h);
    enc4_down(*e, m, "moe_down2_w", 0, s.h, ib, wb, yb);
    e->commit();
    e->wait();
    show("decode (moe_gate_up3_r2, moe_down2_w)", yb.ptr(), size_t(D) * 4, base.data());
  }
  {
    const int R = 3;
    const std::vector<float> x = rand_vec(size_t(R) * D, 1.0f), base = rand_vec(size_t(R) * D, 0.5f), wts = rand_wts(R);
    const std::vector<uint32_t> idx = route_rows(R, true);
    const Buffer xb = upload(x.data(), x.size() * 4, "x"), ib = upload(idx.data(), idx.size() * 4, "idx"),
                 wb = upload(wts.data(), wts.size() * 4, "wts"), yb = upload(base.data(), base.size() * 4, "y");
    auto e = g.begin();
    enc4_verify(*e, m, R, xb, ib, wb, s.h, yb, s.ue, s.umask, s.uslot, s.un);
    e->commit();
    e->wait();
    show("verify, 3 rows (moe4_*)", yb.ptr(), size_t(R) * D * 4, base.data());
  }
  for (auto [T, bm] : {std::pair<int, uint32_t>{40, 8}, {70, 32}}) {
    std::vector<_Float16> X(size_t(T) * D);
    for (auto& v : X) v = _Float16(urand(-1.0f, 1.0f));
    std::memcpy(s.xh.ptr(), X.data(), X.size() * 2);
    const std::vector<uint32_t> idx = route_tokens(T, true);
    std::memset(s.Y.ptr(), 0, size_t(T) * TOPK * D * 4);
    auto e = g.begin();
    const Tiles t = bucket(*e, idx, T, bm);
    enc4_prefill(*e, m, t, bm == 8, s.xh, s.H, s.Y);
    e->commit();
    e->wait();
    show(bm == 8 ? "prefill T=40 (qmms_moe2_*_b4)" : "prefill T=70 (qrm_*_k32_m32)", s.Y.ptr(), size_t(T) * TOPK * D * 4);
  }
  std::printf("  %-40s %016llx\n", "all", (unsigned long long)all);
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode = "check", pack_dir;
  int pack_layer = 0, store_layers = 4, store_passes = 6;
  bool small = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--seed" && i + 1 < argc) rng.seed(uint32_t(std::atoi(argv[++i])));
    else if (a == "--rounds" && i + 1 < argc) g_rounds = std::max(1, std::atoi(argv[++i]));
    else if (a == "--small") small = true;
    else if (a == "--layers" && i + 1 < argc) store_layers = std::max(2, std::atoi(argv[++i]));
    else if (a == "--passes" && i + 1 < argc) store_passes = std::max(1, std::atoi(argv[++i]));
    else if (a == "check" || a == "bench" || a == "hash4" || a == "tune" || a == "all" || a == "store") mode = a;
    else if (a == "pack" && i + 1 < argc) {
      mode = a;
      pack_dir = argv[++i];
    } else if (a == "--layer" && i + 1 < argc) pack_layer = std::atoi(argv[++i]);
    else {
      std::fprintf(stderr, "fn_q2_bench: unknown argument %s (see the comment at the top of tools/fn_q2_bench.cpp)\n",
                   a.c_str());
      return 2;
    }
  }
#ifndef FN_Q20
  if (mode != "hash4") fatal("this build has no Q2_0 kernels: only hash4");
#endif
  Gpu& g = Gpu::get();
#ifdef FN_Q20
  if (mode == "tune") g.compile(kTuneSrc);  // the variants beside the library's kernels (before any pipeline)
#endif
  std::printf("%s, %d GPU cores\n", g.name().c_str(), g.core_count());
#ifdef FN_Q20
  if (mode == "pack") {
    run_pack(pack_dir, pack_layer);
    std::printf("%s\n", g_fail ? "FAILED" : "all checks passed");
    return g_fail ? 1 : 0;
  }
#endif
  if (small || mode == "store") {  // kernel-legal small shapes (shader-validation runs)
    D = 256, I = 128, SI = 128, E = 64, TOPK = 8;
    std::printf("small shapes: hidden %u, %u experts of %u, shared expert %u, top-%u\n", D, E, I, SI, TOPK);
  }
#ifdef FN_Q20
  if (mode == "store") {
    run_store(store_layers, store_passes);
    std::printf("%s\n", g_fail ? "FAILED" : "all checks passed");
    return g_fail ? 1 : 0;
  }
#endif
  Model m;
  make_shared(m);
  const bool need4 = mode == "bench" || mode == "hash4" || mode == "all";
  if (need4) {
    m.g4 = make4(E, I, D, 0.002f, 0.006f);
    m.u4 = make4(E, I, D, 0.002f, 0.006f);
    m.d4 = make4(E, D, I, 0.004f, 0.012f);
    m.have4 = true;
  }
  if (mode == "hash4") {
    std::printf("hash4: the engine's 4-bit MoE kernels on fixed inputs\n");
    run_hash4(m);
    return 0;
  }
#ifdef FN_Q20
  m.g2 = make2(I, D, 0.004f, 0.012f);
  m.u2 = make2(I, D, 0.004f, 0.012f);
  m.d2 = make2(D, I, 0.008f, 0.024f);
  m.have2 = true;
  const Arena A = make_arena(m);
  if (mode == "check" || mode == "all") {
    std::printf("check: Q2_0 against a CPU reference in double, the slot table against the stacked planes\n");
    run_check(m, A);
    std::printf("%s\n", g_fail ? "FAILED" : "all checks passed");
  }
  if (mode == "tune") {
    std::printf("tune: decode kernel shapes, GPU time per call (median of %d rounds of 16)\n", g_rounds);
    run_tune(m);
  }
  if (mode == "bench" || mode == "all") {
    std::printf("bench: GPU time per call (median of %d rounds of 16)\n", g_rounds);
    run_bench(m, A);
  }
#endif
  return g_fail ? 1 : 0;
}
