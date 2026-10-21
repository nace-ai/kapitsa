#include "model/moe_q20.h"

#include "model/kparams.h"
#include "util/log.h"

#include <algorithm>

namespace ni::q20 {

namespace {

// rows a simdgroup of the decode / verify kernels (zq2moe.metal's instances)
constexpr uint32_t kRows = 4;

struct SlotParams {  // Q20Slot
  uint32_t gc, gd, uc, ud, dc, dd;
};

SlotParams slot_params(const Experts& x) {
  return SlotParams{x.lay.gc, x.lay.gd, x.lay.uc, x.lay.ud, x.lay.dc, x.lay.dd};
}

Pipeline pipe(const std::string& base, const Experts& x) { return Gpu::get().pipeline(x.arena ? base + "_slot" : base); }

// the routed experts' gate/up at buffers i .. i + 3: stacked planes, or the arena chunks and the slot table
void bind_gu(Encoder& e, const Experts& x, int i) {
  if (x.arena) {
    e.buf(i, x.a0); e.buf(i + 1, x.a1); e.buf(i + 2, x.slots); e.buf(i + 3, x.slots);
  } else {
    e.buf(i, x.gc); e.buf(i + 1, x.gd); e.buf(i + 2, x.uc); e.buf(i + 3, x.ud);
  }
}

// the routed experts' down at buffers i .. i + 2
void bind_dn(Encoder& e, const Experts& x, int i) {
  if (x.arena) {
    e.buf(i, x.a0); e.buf(i + 1, x.a1); e.buf(i + 2, x.slots);
  } else {
    e.buf(i, x.dc); e.buf(i + 1, x.dd); e.buf(i + 2, x.dc);
  }
}

void bind_shared_gu(Encoder& e, const Shared& s, int i) {
  e.buf(i, s.gw); e.buf(i + 1, s.gs); e.buf(i + 2, s.gb);
  e.buf(i + 3, s.uw); e.buf(i + 4, s.us); e.buf(i + 5, s.ub);
}

// down of R rows (moe_q20_down: decode is R = 1)
void enc_down(Encoder& e, const Experts& x, const Shared& s, const Shape& sh, int R, const Buffer& h,
              const Buffer& idx, const Buffer& wts, const Buffer& y) {
  const kp::Moe mp{sh.D, sh.I, sh.SI, sh.k, 0.0f};
  const SlotParams L = slot_params(x);
  e.use(pipe("moe_q20_down", x));
  bind_dn(e, x, 0);
  e.buf(3, s.dw); e.buf(4, s.ds); e.buf(5, s.db);
  e.buf(6, h); e.buf(7, idx); e.buf(8, wts); e.buf(9, y);
  e.val(10, mp);
  e.val(11, L);
  e.tgmem(0, size_t(sh.k + 1) * std::max(sh.I, sh.SI) * 4);  // h of the k + 1 slots (shapes_ok: <= 7168 floats)
  e.dispatch(uint32_t(R), (sh.D + 8 * kRows - 1) / (8 * kRows), 1, 256);
}

}  // namespace

SlotLayout slot_layout(uint32_t D, uint32_t I) {
  // gate and up are [I, D], down [D, I]: each has I * D / 4 bytes of codes and I * D / 32 of d
  const uint32_t cb = I * D / 4, db = I * D / 32;
  SlotLayout L;
  L.gc = 0;
  L.uc = cb;
  L.dc = 2 * cb;
  L.gd = 3 * cb;
  L.ud = 3 * cb + db;
  L.dd = 3 * cb + 2 * db;
  L.bytes = (3 * (cb + db) + 16383) / 16384 * 16384;
  return L;
}

std::vector<std::string> kernel_names(bool slot) {
  std::vector<std::string> v;
  for (const char* n : {"moe_q20_gate_up", "moe_q20_down", "moe_q20v_gate_up", "qmms_moe_gu_q20", "qmms_moe_dn_q20",
                        "qrm_gu_q20_m32_n64", "qrm_dn_q20_m32_n128"})
    v.push_back(slot ? std::string(n) + "_slot" : std::string(n));
  v.push_back("moe_q20_unique");
  return v;
}

bool shapes_ok(const Shape& sh, std::string* why) {
  auto no = [why](const char* m) {
    if (why) *why = m;
    return false;
  };
  if (sh.D % 128 || sh.I % 64) return no("hidden must be a multiple of 128 and the expert width of 64");
  if (sh.D > 4096) return no("hidden over 4096 (the gate/up kernels stage x in threadgroup memory)");
  if ((sh.k + 1) * std::max(sh.I, sh.SI) > 7168) return no("(k + 1) * max(I, SI) over 7168 (the down kernel's staging)");
  if (sh.k == 0 || sh.k > 16) return no("top-k must be 1-16");
  if (sh.E > 512) return no("more than 512 experts (moe_q20_unique, moe_bucket3)");
  if (sh.SI % 16) return no("shared expert width not a multiple of 16");
  return true;
}

void enc_decode(Encoder& e, const Experts& x, const Shared& s, const Shape& sh, const Buffer& xn, const Buffer& idx,
                const Buffer& wts, const Buffer& h, const Buffer& y) {
  const kp::Moe mp{sh.D, sh.I, sh.SI, sh.k, 0.0f};
  const SlotParams L = slot_params(x);
  constexpr uint32_t nsg = 8, rows = kRows;
  const uint32_t Imax = std::max(sh.I, sh.SI);
  e.use(pipe("moe_q20_gate_up", x));
  bind_gu(e, x, 0);
  bind_shared_gu(e, s, 4);
  e.buf(10, xn); e.buf(11, h);
  e.val(12, mp);
  e.buf(13, idx);
  e.val(14, L);
  e.dispatch((Imax + rows * nsg - 1) / (rows * nsg), sh.k + 1, 1, 32 * nsg);
  enc_down(e, x, s, sh, 1, h, idx, wts, y);
}

void enc_verify(Encoder& e, const Experts& x, const Shared& s, const Shape& sh, int R, const Buffer& xn,
                const Buffer& idx, const Buffer& wts, const Buffer& h, const Buffer& y, const Buffer& ue,
                const Buffer& umask, const Buffer& uslot, const Buffer& un) {
  if (R < 1 || R > 8 || uint32_t(R) * sh.k > 128) fatal("q20::enc_verify: %d rows of top-%u", R, sh.k);
  const kp::Moe mp{sh.D, sh.I, sh.SI, sh.k, 0.0f};
  const SlotParams L = slot_params(x);
  const uint32_t Ru = uint32_t(R), Imax = std::max(sh.I, sh.SI);
  const struct {
    uint32_t R, k;
  } rk{Ru, sh.k};
  e.use(Gpu::get().pipeline("moe_q20_unique"));
  e.buf(0, idx); e.buf(1, ue); e.buf(2, umask); e.buf(3, uslot); e.buf(4, un);
  e.val(5, rk);
  e.dispatch(1, 1, 1, 32);
  e.use(pipe("moe_q20v_gate_up", x));
  bind_gu(e, x, 0);
  bind_shared_gu(e, s, 4);
  e.buf(10, xn); e.buf(11, h);
  e.val(12, mp);
  e.buf(13, ue); e.buf(14, umask); e.buf(15, uslot); e.buf(16, un);
  e.val(17, Ru);
  e.val(18, L);
  e.dispatch(Ru, (Imax + 8 * kRows - 1) / (8 * kRows), Ru * sh.k + 1, 256);
  enc_down(e, x, s, sh, R, h, idx, wts, y);
}

void enc_prefill(Encoder& e, const Experts& x, const Shape& sh, int bm, uint32_t max_tiles, const Buffer& xh,
                 const Buffer& H, const Buffer& Y, const Buffer& tiles, const Buffer& ntiles, const Buffer& xrows,
                 const Buffer& yrows) {
  if (bm != 8 && bm != 32) fatal("q20::enc_prefill: tiles of %d rows (8 or 32)", bm);
  const bool small = bm == 8;
  const SlotParams L = slot_params(x);
  // N, K, x_ld, y_ld, gs, expert strides (words, halfs), shared_e (no shared tiles)
  const kp::Qmm2 pg{sh.I, sh.D, sh.D, sh.I, 64u, sh.I * (sh.D / 16), sh.I * (sh.D / 64), sh.E + 1};
  const kp::Qmm2 pd{sh.D, sh.I, sh.I, sh.D, 64u, sh.D * (sh.I / 16), sh.D * (sh.I / 64), sh.E + 1};
  constexpr uint32_t nsg = 4;  // the small-T kernels: 4 simdgroups of 8 weight rows
  e.use(pipe(small ? "qmms_moe_gu_q20" : "qrm_gu_q20_m32_n64", x));
  e.buf(0, xh);
  bind_gu(e, x, 1);
  e.buf(5, H);
  e.val(6, pg);
  e.buf(7, tiles); e.buf(8, ntiles); e.buf(9, xrows);
  e.val(10, L);
  e.dispatch(small ? sh.I / (8 * nsg) : sh.I / 64, max_tiles, 1, small ? 32 * nsg : 128);
  e.use(pipe(small ? "qmms_moe_dn_q20" : "qrm_dn_q20_m32_n128", x));
  e.buf(0, H);
  bind_dn(e, x, 1);
  e.buf(4, Y);
  e.val(5, pd);
  e.buf(6, tiles); e.buf(7, ntiles); e.buf(8, yrows);
  e.val(9, L);
  e.dispatch(small ? sh.D / (8 * nsg) : sh.D / 128, max_tiles, 1, small ? 32 * nsg : 128);
}

}  // namespace ni::q20
