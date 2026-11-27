// Checks the MLX affine routed experts (3- or 4-bit, group 32 / 64 / 128) with the clamped SwiGLU (src/model/moe_a4.h,
// src/kernels/hy4/a4moe.metal) against a CPU reference in double, and times them. No model: the experts are random weights
// quantized as mx.quantize does (affine, each row's codes a little-endian bit stream, bf16 scales and biases, value =
// code * scale + bias).
//   check  (default) enc_rows on stacked planes and through a slot arena (shuffled slot words, two chunks) for
//          R = 1, 8, 512 on both paths: max abs / rel error against the reference, stacked == arena bit for bit; for
//          every format (gate / up and down alike, then two mixed pairs) or the one --fmt names
//   time   R = 1 (auto) and 8 .. 512 (both paths) at the given shape, every expert resident (stacked and arena)
//   store  a fake pack in /tmp (model-experts-LNN.safetensors + model.safetensors.index.json), a4::Store over it with
//          an arena of 3 E slots (evictions), decode / verify / prefill passes through begin / end / enc_rows against
//          the reference and the stacked planes (bit for bit); --mixed: the layers cycle through four formats
// Options: --small (D = I = 256, E = 64, top-16; for shader validation), --hy4 (D = I = 1536, E = 512, top-16: the
// default for time), --seed N, --layers N (store: default 4; 2 layers fit the minimum arena whole: no evictions),
// --fmt B/G[:B/G] (gate / up's bits and group, then down's; default 4/64).
// Takes no lock: run it under lockf -k -t 600 /tmp/kapitsa-gpu.lock build/a4moe_check ...
#include "metal/gpu.h"
#include "model/hy4_kernels.h"
#include "model/moe_a4.h"
#include "util/bf16.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace ni;

namespace {

a4::Shape SH{256, 256, 16, 64, 10.0f};
a4::Fmt FGU, FDN;  // the formats of check / time / store (all formats when check has no --fmt)
uint32_t g_seed = 1234;
int g_fail = 0;

#define CHECK(c, ...)                         \
  do {                                        \
    if (!(c)) {                               \
      std::fprintf(stderr, "FAIL: " __VA_ARGS__); \
      std::fprintf(stderr, "\n");             \
      ++g_fail;                               \
    }                                         \
  } while (0)

void parallel_for(size_t n, const std::function<void(size_t)>& f) {
  const unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
  std::atomic<size_t> next{0};
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; ++t)
    th.emplace_back([&] {
      for (size_t i; (i = next.fetch_add(1)) < n;) f(i);
    });
  for (auto& t : th) t.join();
}

Buffer abuf(size_t bytes, const char* label) {
  Buffer b = Gpu::get().alloc(bytes + 256, label, /*tracked=*/false);
  std::memset(b.ptr(), 0, bytes + 256);
  return b;
}

// One matrix's stacked planes [E][N][K]: codes K * bits / 32 words a row (a little-endian bit stream), scales /
// biases K / gs bf16 a row
struct Mat {
  uint32_t N = 0, K = 0;
  a4::Fmt f;
  Buffer w, s, b;
  size_t wrow() const { return size_t(K) * f.bits / 8; }  // bytes of a row's codes
  size_t wbytes() const { return size_t(N) * wrow(); }    // one expert
  size_t sbytes() const { return size_t(N) * (K / f.gs) * 2; }  // one expert's scales (biases the same)
  const uint32_t* wp(uint32_t e) const { return w.as<uint32_t>() + size_t(e) * wbytes() / 4; }
  const uint16_t* sp(uint32_t e) const { return s.as<uint16_t>() + size_t(e) * sbytes() / 2; }
  const uint16_t* bp(uint32_t e) const { return b.as<uint16_t>() + size_t(e) * sbytes() / 2; }
  // row n of expert e dequantized; the codes read byte by byte, LSB first (MLX: for 3 bits eight values in three
  // bytes, value j at bits 3j of the 24)
  void row(uint32_t e, uint32_t n, double* out) const {
    const uint8_t* w0 = reinterpret_cast<const uint8_t*>(wp(e)) + size_t(n) * wrow();
    const uint32_t gpr = K / f.gs;
    const uint16_t *s0 = sp(e) + size_t(n) * gpr, *b0 = bp(e) + size_t(n) * gpr;
    for (uint32_t k = 0; k < K; ++k) {
      uint32_t c = 0;
      for (uint32_t bit = 0; bit < f.bits; ++bit) {
        const uint32_t at = k * f.bits + bit;
        c |= uint32_t((w0[at / 8] >> (at % 8)) & 1u) << bit;
      }
      out[k] = double(bf16_to_f32(s0[k / f.gs])) * double(c) + double(bf16_to_f32(b0[k / f.gs]));
    }
  }
};

// mx.quantize(w, group_size=gs, bits=bits) of one group of gs values: codes into the row's bit stream at value k0, and
// the bf16 scale and bias
void quantize_group(const float* w, const a4::Fmt& f, uint32_t k0, uint32_t* row, uint16_t& s16, uint16_t& b16) {
  const int n = int(f.gs);
  const float bins = float((1u << f.bits) - 1u);
  float mx = w[0], mn = w[0];
  for (int i = 1; i < n; ++i) mx = std::max(mx, w[i]), mn = std::min(mn, w[i]);
  const bool mask = std::fabs(mn) > std::fabs(mx);
  float scale = std::max((mx - mn) / bins, 1e-7f);
  scale = mask ? scale : -scale;
  const float edge = mask ? mn : mx;
  const float q0 = std::nearbyint(edge / scale);
  if (q0 != 0.0f) scale = edge / q0;
  const float bias = q0 == 0.0f ? 0.0f : edge;
  s16 = f32_to_bf16(scale);
  b16 = f32_to_bf16(bias);
  for (int i = 0; i < n; ++i) {
    const uint32_t q = uint32_t(std::clamp(std::nearbyint((w[i] - bias) / scale), 0.0f, bins));
    const uint64_t at = uint64_t(k0 + uint32_t(i)) * f.bits;
    row[at / 32] |= q << (at % 32);
    if (at % 32 + f.bits > 32) row[at / 32 + 1] |= q >> (32 - at % 32);
  }
}

Mat make_mat(uint32_t E, uint32_t N, uint32_t K, const a4::Fmt& f, float sigma, uint32_t salt) {
  Mat m;
  m.N = N, m.K = K, m.f = f;
  m.w = abuf(size_t(E) * m.wbytes(), "a4_w");
  m.s = abuf(size_t(E) * m.sbytes(), "a4_s");
  m.b = abuf(size_t(E) * m.sbytes(), "a4_b");
  parallel_for(size_t(E) * N, [&](size_t en) {
    std::mt19937 rng(g_seed * 7919u + salt * 104729u + uint32_t(en));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // rows of different magnitudes, groups with a mean offset (bias codes off-centre)
    const float rs = sigma * (0.5f + float(en % 7) * 0.25f);
    uint32_t* w = m.w.as<uint32_t>() + en * m.wrow() / 4;
    const uint32_t gpr = K / f.gs;
    uint16_t *s = m.s.as<uint16_t>() + en * gpr, *b = m.b.as<uint16_t>() + en * gpr;
    std::vector<float> g(f.gs);
    for (uint32_t gi = 0; gi < gpr; ++gi) {
      const float off = rs * 0.5f * nd(rng);
      for (float& v : g) v = off + rs * nd(rng);
      quantize_group(g.data(), f, gi * f.gs, w, s[gi], b[gi]);
    }
  });
  return m;
}

struct Model {
  uint32_t E = 0;
  Mat g, u, d;
  a4::Experts stacked() const {
    a4::Experts x;
    x.gw = g.w, x.gs = g.s, x.gb = g.b, x.uw = u.w, x.us = u.s, x.ub = u.b, x.dw = d.w, x.ds = d.s, x.db = d.b;
    x.gu = g.f, x.dn = d.f;
    return x;
  }
};

Model make_model(uint32_t E, uint32_t salt, const a4::Fmt& gu, const a4::Fmt& dn) {
  Model m;
  m.E = E;
  m.g = make_mat(E, SH.I, SH.D, gu, 0.03f, salt * 3 + 0);
  m.u = make_mat(E, SH.I, SH.D, gu, 0.03f, salt * 3 + 1);
  m.d = make_mat(E, SH.D, SH.I, dn, 0.03f, salt * 3 + 2);
  return m;
}

// The same experts as slots of an arena of two chunks: slots in a shuffled order, chunks alternating, a one-page gap
// after every 5th slot of a chunk
struct Arena {
  Buffer a0, a1, slots;
};

Arena make_arena(const Model& m) {
  const a4::SlotLayout L = a4::slot_layout(SH, m.g.f, m.d.f);
  const uint32_t pages = L.bytes / 16384;
  std::vector<uint32_t> order(m.E);
  std::iota(order.begin(), order.end(), 0u);
  std::mt19937 rng(g_seed + 5);
  std::shuffle(order.begin(), order.end(), rng);
  std::vector<uint32_t> word(m.E);
  uint32_t next[2] = {0, 0};
  for (uint32_t i = 0; i < m.E; ++i) {
    const uint32_t c = i % 2, e = order[i];
    word[e] = (c << 28) | next[c];
    next[c] += pages + ((i / 2) % 5 == 4 ? 1 : 0);
  }
  Arena a;
  a.a0 = abuf(size_t(std::max(next[0], 1u)) * 16384, "a4_arena0");
  a.a1 = abuf(size_t(std::max(next[1], 1u)) * 16384, "a4_arena1");
  a.slots = abuf(size_t(m.E) * 4, "a4_slots");
  std::memcpy(a.slots.ptr(), word.data(), size_t(m.E) * 4);
  const Mat* M[3] = {&m.g, &m.u, &m.d};
  parallel_for(m.E, [&](size_t e) {
    uint8_t* base = (word[e] >> 28 ? a.a1 : a.a0).as<uint8_t>() + size_t(word[e] & 0x0FFFFFFFu) * 16384;
    for (int mi = 0; mi < 3; ++mi) {
      const Mat& X = *M[mi];
      std::memcpy(base + L.off[3 * mi], X.wp(uint32_t(e)), X.wbytes());
      std::memcpy(base + L.off[3 * mi + 1], X.sp(uint32_t(e)), X.sbytes());
      std::memcpy(base + L.off[3 * mi + 2], X.bp(uint32_t(e)), X.sbytes());
    }
  });
  return a;
}

a4::Experts arena_experts(const Arena& a, const Model& m) {
  a4::Experts x;
  x.arena = true;
  x.a0 = a.a0, x.a1 = a.a1, x.slots = a.slots;
  const a4::SlotLayout L = a4::slot_layout(SH, m.g.f, m.d.f);
  for (int i = 0; i < 9; ++i) x.off[i] = L.off[i];
  x.gu = m.g.f, x.dn = m.d.f;
  return x;
}

// A batch of R rows: inputs, routing, weights
struct Rows {
  int R = 0;
  std::vector<float> x, w;
  std::vector<uint32_t> idx;
  Buffer bx, bidx, bw;
};

Rows make_rows(int R, uint32_t E, std::mt19937& rng) {
  Rows r;
  r.R = R;
  r.x.resize(size_t(R) * SH.D);
  r.w.resize(size_t(R) * SH.k);
  r.idx.resize(size_t(R) * SH.k);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::uniform_real_distribution<float> ud(0.0f, 1.0f);
  for (int i = 0; i < R; ++i) {
    // rows of very different sizes, so some gate / up values pass the clamp
    const float sc = 0.3f + 12.0f * ud(rng) * ud(rng);
    for (uint32_t k = 0; k < SH.D; ++k) r.x[size_t(i) * SH.D + k] = sc * nd(rng);
    std::vector<uint32_t> all(E);
    std::iota(all.begin(), all.end(), 0u);
    for (uint32_t j = 0; j < SH.k; ++j) {
      const uint32_t p = j + uint32_t(rng() % (E - j));
      std::swap(all[j], all[p]);
      r.idx[size_t(i) * SH.k + j] = all[j];
    }
    float sum = 0;
    for (uint32_t j = 0; j < SH.k; ++j) sum += (r.w[size_t(i) * SH.k + j] = 0.05f + ud(rng));
    for (uint32_t j = 0; j < SH.k; ++j) r.w[size_t(i) * SH.k + j] *= 3.998f / sum;
  }
  r.bx = abuf(r.x.size() * 4, "a4_x");
  r.bidx = abuf(r.idx.size() * 4, "a4_idx");
  r.bw = abuf(r.w.size() * 4, "a4_wts");
  std::memcpy(r.bx.ptr(), r.x.data(), r.x.size() * 4);
  std::memcpy(r.bidx.ptr(), r.idx.data(), r.idx.size() * 4);
  std::memcpy(r.bw.ptr(), r.w.data(), r.w.size() * 4);
  return r;
}

std::atomic<uint64_t> g_clamped{0};

// the reference output of row i (fp64)
std::vector<double> ref_row(const Model& m, const Rows& r, int i) {
  const uint32_t D = SH.D, I = SH.I;
  std::vector<double> y(D, 0.0), wr(std::max(D, I)), h(I);
  const float* x = r.x.data() + size_t(i) * D;
  uint64_t clamped = 0;
  for (uint32_t j = 0; j < SH.k; ++j) {
    const uint32_t e = r.idx[size_t(i) * SH.k + j];
    for (uint32_t n = 0; n < I; ++n) {
      double g = 0, u = 0;
      m.g.row(e, n, wr.data());
      for (uint32_t k = 0; k < D; ++k) g += wr[k] * x[k];
      m.u.row(e, n, wr.data());
      for (uint32_t k = 0; k < D; ++k) u += wr[k] * x[k];
      if (SH.limit > 0) {
        if (g > SH.limit || u > SH.limit || u < -SH.limit) ++clamped;
        g = std::min(g, double(SH.limit));
        u = std::clamp(u, -double(SH.limit), double(SH.limit));
      }
      h[n] = g / (1.0 + std::exp(-g)) * u;
    }
    const double wt = r.w[size_t(i) * SH.k + j];
    for (uint32_t n = 0; n < D; ++n) {
      m.d.row(e, n, wr.data());
      double a = 0;
      for (uint32_t k = 0; k < I; ++k) a += wr[k] * h[k];
      y[n] += wt * a;
    }
  }
  g_clamped += clamped;
  return y;
}

struct Err {
  double abs = 0, scale = 0;
  double rel() const { return scale > 0 ? abs / scale : 0; }
};

// GPU output y [R, D] against the reference at rows `rows`
Err compare(const std::vector<std::vector<double>>& ref, const std::vector<int>& rows, const float* y) {
  Err e;
  for (size_t q = 0; q < rows.size(); ++q)
    for (uint32_t n = 0; n < SH.D; ++n) {
      const double rv = ref[q][n], gv = y[size_t(rows[q]) * SH.D + n];
      if (!std::isfinite(gv)) e.abs = INFINITY;
      e.abs = std::max(e.abs, std::fabs(gv - rv));
      e.scale = std::max(e.scale, std::fabs(rv));
    }
  return e;
}

const char* path_name(a4::Path p) { return p == a4::Path::Rows ? "rows" : p == a4::Path::Tiles ? "tiles" : "auto"; }

double run_once(const a4::Experts& x, const Rows& r, const Buffer& scratch, const Buffer& y, a4::Path p) {
  auto e = Gpu::get().begin();
  a4::enc_rows(*e, x, SH, r.R, r.bx, r.bidx, r.bw, scratch, y, p);
  e->commit_and_wait();
  return e->gpu_ms();
}

// ---- check ----
void run_check(int tmax_rows, const a4::Fmt& gu, const a4::Fmt& dn) {
  std::string why;
  if (!a4::formats_ok(SH, gu, dn, &why)) fatal("formats: %s", why.c_str());
  std::printf("-- gate / up %s, down %s (slot %u pages)\n", a4::fmt_name(gu).c_str(), a4::fmt_name(dn).c_str(),
              a4::slot_layout(SH, gu, dn).bytes / 16384);
  const Model m = make_model(SH.E, 1, gu, dn);
  const Arena A = make_arena(m);
  const a4::Experts xs = m.stacked(), xa = arena_experts(A, m);
  const int Rs[3] = {1, 8, std::min(512, tmax_rows)};
  const Buffer scratch = abuf(a4::scratch_bytes(SH, 512), "a4_scratch");
  const Buffer y1 = abuf(size_t(512) * SH.D * 4, "a4_y1"), y2 = abuf(size_t(512) * SH.D * 4, "a4_y2");
  std::mt19937 rng(g_seed);
  for (int R : Rs) {
    const Rows r = make_rows(R, SH.E, rng);
    // reference rows: every row up to 64, else 32 spread over the batch
    std::vector<int> rows;
    if (R <= 64) {
      for (int i = 0; i < R; ++i) rows.push_back(i);
    } else {
      for (int i = 0; i < 32; ++i) rows.push_back(int((size_t(i) * 2654435761u) % size_t(R)));
      rows[0] = 0, rows[1] = R - 1;
    }
    std::vector<std::vector<double>> ref(rows.size());
    g_clamped = 0;
    parallel_for(rows.size(), [&](size_t q) { ref[q] = ref_row(m, r, rows[q]); });
    for (a4::Path p : {a4::Path::Rows, a4::Path::Tiles}) {
      std::memset(y1.ptr(), 0xFF, size_t(R) * SH.D * 4);  // NaN: an unwritten output shows
      std::memset(y2.ptr(), 0xFF, size_t(R) * SH.D * 4);
      run_once(xs, r, scratch, y1, p);
      run_once(xa, r, scratch, y2, p);
      const Err e = compare(ref, rows, y1.as<float>());
      const bool same = !std::memcmp(y1.ptr(), y2.ptr(), size_t(R) * SH.D * 4);
      // fp32 rows; the tiles round x and h to half
      const double gate = p == a4::Path::Rows ? 1e-4 : 4e-3;
      std::printf("R = %3d %-5s: max abs %.3e, max |ref| %.3e, rel %.3e (gate %.0e), %zu rows checked, %llu clamped "
                  "gate/up values; arena %s stacked\n",
                  R, path_name(p), e.abs, e.scale, e.rel(), gate, rows.size(), (unsigned long long)g_clamped.load(),
                  same ? "==" : "!=");
      CHECK(e.rel() <= gate, "%s/%s R = %d %s: rel %.3e over %.0e", a4::fmt_name(gu).c_str(), a4::fmt_name(dn).c_str(),
            R, path_name(p), e.rel(), gate);
      CHECK(same, "%s/%s R = %d %s: the arena's output differs from the stacked planes'", a4::fmt_name(gu).c_str(),
            a4::fmt_name(dn).c_str(), R, path_name(p));
    }
  }
}

// ---- time ----
void run_time() {
  std::printf("-- gate / up %s, down %s\n", a4::fmt_name(FGU).c_str(), a4::fmt_name(FDN).c_str());
  const Model m = make_model(SH.E, 2, FGU, FDN);
  const Arena A = make_arena(m);
  const a4::Experts xs = m.stacked(), xa = arena_experts(A, m);
  const Buffer scratch = abuf(a4::scratch_bytes(SH, 512), "a4_scratch");
  const Buffer y = abuf(size_t(512) * SH.D * 4, "a4_y");
  std::mt19937 rng(g_seed + 77);
  const double expert_mb = (2.0 * (m.g.wbytes() + 2 * m.g.sbytes()) + m.d.wbytes() + 2 * m.d.sbytes()) / 1e6;
  for (int R : {1, 8, 32, 64, 128, 256, 512}) {
    const int n = R <= 8 ? 40 : 8;
    std::vector<Rows> rs;
    for (int i = 0; i < n; ++i) rs.push_back(make_rows(R, SH.E, rng));
    for (int arena = 0; arena < 2; ++arena) {
      std::vector<a4::Path> paths = {a4::Path::Auto};
      if (R > 1) paths = {a4::Path::Rows, a4::Path::Tiles};
      for (a4::Path p : paths) {
        run_once(arena ? xa : xs, rs[0], scratch, y, p);  // warm-up (pipelines)
        std::vector<double> ms;
        for (int i = 0; i < n; ++i) ms.push_back(run_once(arena ? xa : xs, rs[size_t(i)], scratch, y, p));
        std::sort(ms.begin(), ms.end());
        std::set<uint32_t> ex;
        for (uint32_t v : rs[0].idx) ex.insert(v);
        std::printf("time R = %3d %-5s %-7s: median %.3f ms, min %.3f ms over %d runs (%zu experts, %.0f MB of weights: "
                    "%.0f GB/s at the median)\n",
                    R, path_name(p), arena ? "arena" : "stacked", ms[ms.size() / 2], ms[0], n, ex.size(),
                    ex.size() * expert_mb, ex.size() * expert_mb / ms[ms.size() / 2]);
      }
    }
  }
}

// ---- store ----
void write_safetensors(const std::string& path, const std::vector<std::pair<std::string, std::pair<std::string,
                       std::vector<int64_t>>>>& meta, const std::vector<std::pair<const void*, size_t>>& data) {
  nlohmann::json h = nlohmann::json::object();
  size_t off = 0;
  for (size_t i = 0; i < meta.size(); ++i) {
    h[meta[i].first] = {{"dtype", meta[i].second.first}, {"shape", meta[i].second.second},
                        {"data_offsets", {off, off + data[i].second}}};
    off += data[i].second;
  }
  h["__metadata__"] = {{"format", "mlx"}};
  const std::string s = h.dump();
  std::ofstream f(path, std::ios::binary);
  const uint64_t n = s.size();
  f.write(reinterpret_cast<const char*>(&n), 8);
  f.write(s.data(), std::streamsize(s.size()));
  for (const auto& d : data) f.write(static_cast<const char*>(d.first), std::streamsize(d.second));
  if (!f) fatal("writing %s failed", path.c_str());
}

void run_store(int nlayers, bool mixed) {
  Gpu& g = Gpu::get();
  char tmpl[] = "/tmp/a4store.XXXXXX";
  if (!mkdtemp(tmpl)) fatal("mkdtemp failed");
  const std::string dir = tmpl;
  std::vector<int> layers;
  std::vector<Model> lm;
  nlohmann::json index = {{"metadata", {{"total_size", 0}}}, {"weight_map", nlohmann::json::object()}};
  for (int l = 0; l < nlayers; ++l) {
    const int trunk = 1 + 2 * l;  // trunk layers 1, 3, 5, ...
    layers.push_back(trunk);
    // --mixed: the four formats in turn (the slots are the largest's)
    static const a4::Fmt kMix[4][2] = {{{3, 64}, {3, 64}}, {{4, 64}, {4, 64}}, {{3, 128}, {4, 32}}, {{3, 32}, {3, 128}}};
    const a4::Fmt gu = mixed ? kMix[l % 4][0] : FGU, dn = mixed ? kMix[l % 4][1] : FDN;
    lm.push_back(make_model(SH.E, 10 + uint32_t(l), gu, dn));
    const Model& m = lm.back();
    char fn[64];
    std::snprintf(fn, sizeof fn, "model-experts-L%02d.safetensors", trunk);
    std::vector<std::pair<std::string, std::pair<std::string, std::vector<int64_t>>>> meta;
    std::vector<std::pair<const void*, size_t>> data;
    const Mat* M[3] = {&m.g, &m.u, &m.d};
    const char* names[3] = {"gate", "up", "down"};
    // down first in the file and a stray tensor in front: the store must go by the header, not by an order
    static const float stray[3] = {1, 2, 3};
    meta.push_back({"model.layers." + std::to_string(trunk) + ".mlp.gate.weight", {"F32", {3}}});
    data.push_back({stray, 12});
    for (int mi : {2, 0, 1}) {
      const Mat& X = *M[mi];
      const std::string base = "model.layers." + std::to_string(trunk) + ".mlp.switch_mlp." + names[mi] + "_proj.";
      const std::vector<int64_t> ws = {int64_t(SH.E), int64_t(X.N), int64_t(X.K * X.f.bits / 32)},
                                 ss = {int64_t(SH.E), int64_t(X.N), int64_t(X.K / X.f.gs)};
      meta.push_back({base + "weight", {"U32", ws}});
      data.push_back({X.w.ptr(), size_t(SH.E) * X.wbytes()});
      meta.push_back({base + "scales", {"BF16", ss}});
      data.push_back({X.s.ptr(), size_t(SH.E) * X.sbytes()});
      meta.push_back({base + "biases", {"BF16", ss}});
      data.push_back({X.b.ptr(), size_t(SH.E) * X.sbytes()});
    }
    write_safetensors(dir + "/" + fn, meta, data);
    for (const auto& mt : meta) index["weight_map"][mt.first] = fn;
  }
  std::ofstream(dir + "/model.safetensors.index.json") << index.dump(1);
  uint32_t slot = 0;
  for (const Model& m : lm) slot = std::max(slot, a4::slot_layout(SH, m.g.f, m.d.f).bytes);
  const size_t arena = size_t(3) * SH.E * slot;
  a4::Store st(dir, layers, SH, arena);
  std::printf("store: %d layers of %u experts (D = I = %u), %s; complete: %s\n", nlayers, SH.E, SH.D,
              st.describe().c_str(), st.complete() ? "yes" : "no");
  g.make_resident(st.buffers());
  const int tmax = 96;
  const Buffer scratch = abuf(a4::scratch_bytes(SH, tmax), "a4_scratch");
  std::vector<Buffer> ys, yr;
  for (int l = 0; l < nlayers; ++l) {
    ys.push_back(abuf(size_t(tmax) * SH.D * 4, "a4_ys"));
    yr.push_back(abuf(size_t(tmax) * SH.D * 4, "a4_yr"));
  }
  std::mt19937 rng(g_seed + 99);
  int bad = 0;
  Err worst;
  const int passes = 12;
  for (int pass = 0; pass < passes; ++pass) {
    const int kind = pass % 3;  // 0 decode, 1 verify, 2 prefill chunk
    const int R = kind == 0 ? 1 : kind == 1 ? 8 : tmax;
    std::vector<Rows> rs;
    for (int l = 0; l < nlayers; ++l) rs.push_back(make_rows(R, SH.E, rng));
    // predictions: the next layer's true ids with three replaced
    std::vector<Buffer> preds;
    for (int l = 0; l < nlayers; ++l) {
      std::vector<uint32_t> p(rs[size_t((l + 1) % nlayers)].idx);
      for (int j = 0; j < 3 && j < int(p.size()); ++j) p[size_t(j)] = uint32_t(rng() % SH.E);
      Buffer b = abuf(p.size() * 4, "a4_pred");
      std::memcpy(b.ptr(), p.data(), p.size() * 4);
      preds.push_back(b);
    }
    auto e = g.begin();
    for (int l = 0; l < nlayers; ++l) {
      const Rows& r = rs[size_t(l)];
      const uint32_t n = uint32_t(R) * SH.k;
      st.begin(*e, layers[size_t(l)], r.bidx, n, kind == 2 ? nullptr : &preds[size_t(l)], n, kind == 2);
      st.end(*e);
      a4::enc_rows(*e, st.experts(layers[size_t(l)]), SH, R, r.bx, r.bidx, r.bw, scratch, ys[size_t(l)]);
      a4::enc_rows(*e, lm[size_t(l)].stacked(), SH, R, r.bx, r.bidx, r.bw, scratch, yr[size_t(l)]);
    }
    e->commit_and_wait();
    for (int l = 0; l < nlayers; ++l) {
      const Rows& r = rs[size_t(l)];
      if (std::memcmp(ys[size_t(l)].ptr(), yr[size_t(l)].ptr(), size_t(R) * SH.D * 4)) ++bad;
      std::vector<int> rows;
      for (int i = 0; i < std::min(R, 8); ++i) rows.push_back(i * (R / std::min(R, 8)));
      std::vector<std::vector<double>> ref(rows.size());
      parallel_for(rows.size(), [&](size_t q) { ref[q] = ref_row(lm[size_t(l)], r, rows[q]); });
      const Err er = compare(ref, rows, ys[size_t(l)].as<float>());
      const double gate = R <= a4::kRowsMax ? 1e-4 : 4e-3;
      CHECK(er.rel() <= gate, "store pass %d layer %d: rel %.3e", pass, l, er.rel());
      if (er.rel() > worst.rel()) worst = er;
    }
  }
  const a4::Store::Counts c = st.counts();
  std::printf("store: %d passes (decode, verify 8 rows, prefill %d rows): %llu handshakes, %llu lookups, %llu misses, "
              "%llu read ahead, %llu evictions, %.1f MB read, stall %.3f ms a turn; worst rel error %.3e; %d layer "
              "outputs differ from the stacked planes'\n",
              passes, tmax, (unsigned long long)c.handshakes, (unsigned long long)c.lookups,
              (unsigned long long)c.misses, (unsigned long long)c.prefetches, (unsigned long long)c.evictions,
              c.mb_read, c.stall_ms / double(std::max<uint64_t>(c.handshakes, 1)), worst.rel(), bad);
  CHECK(bad == 0, "store: %d layer outputs differ from the stacked planes'", bad);
  if (!st.complete()) CHECK(c.evictions > 0 && c.misses > 0, "store: no evictions or misses");
  else CHECK(c.handshakes == 0, "store: a complete store made handshakes");
  for (int l = 0; l < nlayers; ++l) {
    char fn[64];
    std::snprintf(fn, sizeof fn, "/model-experts-L%02d.safetensors", layers[size_t(l)]);
    unlink((dir + fn).c_str());
  }
  unlink((dir + "/model.safetensors.index.json").c_str());
  rmdir(dir.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode = "check";
  bool small = false, hy4 = false;
  int nlayers = 4;
  bool mixed = false, fmt_given = false;
  auto parse_fmt = [](const std::string& v, a4::Fmt& f) {
    unsigned b = 0, g = 0;
    if (std::sscanf(v.c_str(), "%u/%u", &b, &g) != 2) fatal("--fmt: %s is not BITS/GROUP", v.c_str());
    f = a4::Fmt{b, g};
  };
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--small") small = true;
    else if (a == "--hy4") hy4 = true;
    else if (a == "--seed" && i + 1 < argc) g_seed = uint32_t(std::atoi(argv[++i]));
    else if (a == "--layers" && i + 1 < argc) nlayers = std::max(1, std::atoi(argv[++i]));
    else if (a == "--mixed") mixed = true;
    else if (a == "--fmt" && i + 1 < argc) {
      const std::string v = argv[++i];
      const size_t c = v.find(':');
      parse_fmt(v.substr(0, c), FGU);
      FDN = FGU;
      if (c != std::string::npos) parse_fmt(v.substr(c + 1), FDN);
      fmt_given = true;
    }
    else if (a == "check" || a == "time" || a == "store") mode = a;
    else {
      std::fprintf(stderr, "a4moe_check: unknown argument %s (see the comment at the top of tools/a4moe_check.cpp)\n",
                   a.c_str());
      return 2;
    }
  }
  if (hy4 || (mode == "time" && !small)) SH = a4::Shape{1536, 1536, 16, 512, 10.0f};
  std::string why;
  if (!a4::formats_ok(SH, FGU, FDN, &why)) fatal("shapes / formats: %s", why.c_str());
  Gpu& g = Gpu::get();
  std::printf("%s, %d GPU cores; D %u, I %u, E %u, top-%u, limit %.1f; slot %u pages\n", g.name().c_str(),
              g.core_count(), SH.D, SH.I, SH.E, SH.k, SH.limit, a4::slot_layout(SH, FGU, FDN).bytes / 16384);
  for (const std::string& k : a4::kernel_names()) {
    const Pipeline p = hy4::pipeline(k);
    if (p.tg_mem > 16384) {
      std::fprintf(stderr, "FAIL: %s uses %u B of static threadgroup memory (over 16384)\n", k.c_str(), p.tg_mem);
      ++g_fail;
    }
  }
  if (mode == "check") {
    if (fmt_given) {
      run_check(512, FGU, FDN);
    } else {
      for (uint32_t bits : {4u, 3u})
        for (uint32_t gs : {64u, 32u, 128u}) run_check(512, a4::Fmt{bits, gs}, a4::Fmt{bits, gs});
      run_check(512, a4::Fmt{3, 64}, a4::Fmt{4, 64});
      run_check(512, a4::Fmt{4, 128}, a4::Fmt{3, 32});
    }
  } else if (mode == "time") {
    run_time();
  } else {
    run_store(nlayers, mixed);
  }
  std::printf("%s\n", g_fail ? "FAILED" : "all checks passed");
  return g_fail ? 1 : 0;
}
