#include "model/moe_exl3.h"

#include "util/log.h"

#include <cmath>
#include <cstring>

namespace ni::exl3 {

namespace {

struct Params {  // Exl3Params
  uint32_t D, I, k;
  uint32_t ng, nu, nd;
  uint32_t mask;
  uint32_t rows;
};

Params params(const Experts& x, const Shape& sh, uint32_t rows) {
  return Params{sh.D, sh.I, sh.k, x.g.n, x.u.n, x.d.n, (1u << sh.window) - 1u, rows};
}

float h2f(uint16_t h) {
  _Float16 v;
  memcpy(&v, &h, 2);
  return float(v);
}

uint16_t f2h(float f) {
  const _Float16 v = _Float16(f);
  uint16_t h;
  memcpy(&h, &v, 2);
  return h;
}

float round16(float f) { return h2f(f2h(f)); }

}  // namespace

uint32_t rate_of(const std::vector<int64_t>& shape) {
  if (shape.size() != 4) return 0;
  const int64_t n = shape[3];
  return n >= 16 && n <= 64 && n % 2 == 0 ? uint32_t(n) : 0u;
}

bool shapes_ok(const Shape& sh, const Experts& x, std::string* why) {
  auto no = [why](const std::string& m) {
    if (why) *why = m;
    return false;
  };
  if (sh.D % 512 || sh.I % 128) return no("hidden must be a multiple of 512 and the expert width of 128 (H128 blocks)");
  if (sh.D > 4096 || sh.I > 4096) return no("hidden or expert width over 4096 (the rows kernels stage x in threadgroup memory)");
  if (sh.k == 0 || sh.k > 16) return no("top-k must be 1-16");
  if (sh.E > 512) return no("more than 512 experts (moe_bucket3)");
  if (sh.window < 8 || sh.window > 16) return no("a codeword window of " + std::to_string(sh.window) + " bits (8-16)");
  for (const Proj* p : {&x.g, &x.u, &x.d})
    if (p->n < 16 || p->n > 64 || p->n % 2) return no("a rate of " + std::to_string(p->n) + " halfwords a tile (16-64: K1-K4)");
  // the buffers hold what the kernels index from these shapes (a config that disagrees with the tensors, a short shard)
  auto holds = [&](const Proj& p, uint32_t in, uint32_t out) {
    return p.t.size >= size_t(sh.E) * (in / 16) * (out / 16) * p.n * 2 && p.suh.size >= size_t(sh.E) * in * 2 &&
           p.svh.size >= size_t(sh.E) * out * 2;
  };
  if (!holds(x.g, sh.D, sh.I) || !holds(x.u, sh.D, sh.I) || !holds(x.d, sh.I, sh.D))
    return no("the trellis, suh or svh tensors are smaller than the experts' shapes need");
  return true;
}

std::vector<std::string> kernel_names() {
  return {"exl3_gu_rows", "exl3_mid", "exl3_dn_rows", "exl3_fin_rows", "exl3_gu_mm", "exl3_dn_mm", "exl3_fin_pairs"};
}

void enc_rows(Encoder& e, const Experts& x, const Shape& sh, int R, const Buffer& xn, const Buffer& idx,
              const Buffer& wts, const Buffer& ys, const Buffer& hg, const Buffer& hu, const Buffer& md,
              const Buffer& yd, const Buffer& y) {
  if (R < 1 || R > 8) fatal("exl3::enc_rows: %d rows", R);
  const uint32_t S = uint32_t(R) * sh.k;
  const Params p = params(x, sh, S);
  Gpu& g = Gpu::get();
  e.use(g.pipeline("exl3_gu_rows"));
  e.buf(0, x.g.t); e.buf(1, x.g.suh); e.buf(2, x.u.t); e.buf(3, x.u.suh);
  e.buf(4, xn); e.buf(5, idx); e.buf(6, hg); e.buf(7, hu);
  e.val(8, p);
  e.tgmem(0, size_t(sh.D) * 4);
  e.dispatch((sh.I / 16 + 7) / 8, S, 2, 256);
  const uint32_t sorted = 0;
  e.use(g.pipeline("exl3_mid"));
  e.buf(0, hg); e.buf(1, hu); e.buf(2, x.g.svh); e.buf(3, x.u.svh); e.buf(4, x.d.suh); e.buf(5, idx); e.buf(6, idx);
  e.buf(7, md);
  e.val(8, p);
  e.val(9, sorted);
  e.dispatch(sh.I / 128, (S + 3) / 4, 1, 128);
  e.use(g.pipeline("exl3_dn_rows"));
  e.buf(0, x.d.t); e.buf(1, md); e.buf(2, idx); e.buf(3, yd);
  e.val(4, p);
  e.tgmem(0, size_t(sh.I) * 4);
  e.dispatch((sh.D / 16 + 7) / 8, S, 1, 256);
  e.use(g.pipeline("exl3_fin_rows"));
  e.buf(0, yd); e.buf(1, x.d.svh); e.buf(2, idx); e.buf(3, wts); e.buf(4, ys); e.buf(5, y);
  e.val(6, p);
  e.dispatch((sh.D + 511) / 512, uint32_t(R), 1, 128);
}

void enc_prefill(Encoder& e, const Experts& x, const Shape& sh, uint32_t P, uint32_t max_tiles, const Buffer& xh,
                 const Buffer& tiles, const Buffer& ntiles, const Buffer& xrows, const Buffer& yrows, const Buffer& idx,
                 const Buffer& hg, const Buffer& hu, const Buffer& md, const Buffer& yd) {
  const Params p = params(x, sh, P);
  Gpu& g = Gpu::get();
  e.use(g.pipeline("exl3_gu_mm"));
  e.buf(0, x.g.t); e.buf(1, x.g.suh); e.buf(2, x.u.t); e.buf(3, x.u.suh);
  e.buf(4, xh); e.buf(5, tiles); e.buf(6, ntiles); e.buf(7, xrows); e.buf(8, hg); e.buf(9, hu);
  e.val(10, p);
  e.dispatch(sh.I / 64, max_tiles, 2, 128);
  const uint32_t sorted = 1;
  e.use(g.pipeline("exl3_mid"));
  e.buf(0, hg); e.buf(1, hu); e.buf(2, x.g.svh); e.buf(3, x.u.svh); e.buf(4, x.d.suh); e.buf(5, idx);
  e.buf(6, yrows); e.buf(7, md);
  e.val(8, p);
  e.val(9, sorted);
  e.dispatch(sh.I / 128, (P + 3) / 4, 1, 128);
  e.use(g.pipeline("exl3_dn_mm"));
  e.buf(0, x.d.t); e.buf(1, md); e.buf(2, tiles); e.buf(3, ntiles); e.buf(4, yrows); e.buf(5, yd);
  e.val(6, p);
  e.dispatch(sh.D / 64, max_tiles, 1, 128);
  e.use(g.pipeline("exl3_fin_pairs"));
  e.buf(0, yd); e.buf(1, x.d.svh); e.buf(2, idx);
  e.val(3, p);
  e.dispatch(sh.D / 128, (P + 3) / 4, 1, 128);
}

// ---- CPU reference: Sushi's expert_exl3.zig (decodeMcg, unpackTile, tensorCorePerm, hadamard128, project) ----

uint16_t mcg(uint32_t cw) {
  const uint32_t pair = 0x3B603B60u ^ ((cw * 0xCBAC1FEDu) & 0x8FFF8FFFu);
  return f2h(h2f(uint16_t(pair)) + h2f(uint16_t(pair >> 16)));
}

void unpack_tile(const uint16_t* words, uint32_t n, uint16_t* cw) {
  // bit b of the stream: bit 31 - b % 32 of word b / 32 (word i: halfwords 2i, 2i + 1, little-endian)
  const uint32_t total = 16 * n;
  auto bit = [&](uint32_t b) {
    b %= total;
    const uint32_t w = uint32_t(words[2 * (b / 32)]) | (uint32_t(words[2 * (b / 32) + 1]) << 16);
    return (w >> (31 - b % 32)) & 1u;
  };
  for (uint32_t t = 0; t < 256; ++t) {
    const uint32_t end = ((t + 1) * n) >> 4;
    uint32_t v = 0;
    for (uint32_t b = end + total - 16; b < end + total; ++b) v = (v << 1) | bit(b);
    cw[t] = uint16_t(v);
  }
}

uint32_t perm(uint32_t i) {
  const uint32_t lane = i / 8, m = i % 8, q = lane % 4, c = lane / 4;
  const uint32_t rows[4] = {2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9};
  return rows[m % 4] * 16 + c + (m / 4) * 8;
}

void decode_tile(const uint16_t* words, uint32_t n, uint32_t window, float* w) {
  uint16_t cw[256];
  unpack_tile(words, n, cw);
  const uint32_t mask = (1u << window) - 1u;
  for (uint32_t i = 0; i < 256; ++i) w[perm(i)] = h2f(mcg(cw[i] & mask));
}

void h128(float* v) {
  float t[128];
  for (int r = 0; r < 128; ++r) {
    float acc = 0.0f;
    for (int c = 0; c < 128; ++c) acc += (__builtin_popcount(unsigned(r & c)) % 2 ? -1.0f : 1.0f) * 0.08838834764831845f * v[c];
    t[r] = acc;
  }
  memcpy(v, t, sizeof t);
}

void project(const uint16_t* trellis, const uint16_t* suh, const uint16_t* svh, uint32_t in, uint32_t out, uint32_t n,
             uint32_t window, const float* x, float* y) {
  std::vector<float> xt(in), inner(out, 0.0f);
  for (uint32_t i = 0; i < in; ++i) xt[i] = round16(x[i]) * h2f(suh[i]);
  for (uint32_t b = 0; b < in; b += 128) {
    h128(&xt[b]);
    for (uint32_t i = b; i < b + 128; ++i) xt[i] = round16(xt[i]);
  }
  float w[256];
  const uint32_t ot = out / 16;
  for (uint32_t tk = 0; tk < in / 16; ++tk)
    for (uint32_t tn = 0; tn < ot; ++tn) {
      decode_tile(trellis + (size_t(tk) * ot + tn) * n, n, window, w);
      for (uint32_t r = 0; r < 16; ++r)
        for (uint32_t c = 0; c < 16; ++c) inner[tn * 16 + c] += xt[tk * 16 + r] * w[r * 16 + c];
    }
  for (uint32_t b = 0; b < out; b += 128) h128(&inner[b]);
  for (uint32_t c = 0; c < out; ++c) y[c] = round16(inner[c] * h2f(svh[c]));
}

}  // namespace ni::exl3
