// Hy4-flash GPU blocks: host side of src/kernels/hy4/hy4.metal (and attn.metal's attn_prefill11_impl with HY). See hy4_ops.h.
#include "model/hy4_ops.h"

#include "model/hy4_kernels.h"
#include "model/kparams.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace ni::hy4 {
namespace {

Pipeline pipe(const char* name) { return hy4::pipeline(name); }

uint32_t dt(const Tensor& t, const char* what) {
  switch (t.dtype) {
    case DType::F32: return 0;
    case DType::BF16: return 1;
    case DType::F16: return 2;
    default: fatal("hy4: %s has an unsupported dtype (fp32, bf16 or fp16 expected)", what);
  }
}

const Tensor& need(const Tensor* t, const char* what) {
  if (!t || !*t) fatal("hy4: %s is missing", what);
  return *t;
}

uint32_t threads_for(uint32_t n) {  // a row kernel's threadgroup: 1024, fewer for short rows (multiple of 32)
  return std::max(32u, std::min(1024u, (n + 31) / 32 * 32));
}

// Decode chunking: at most kMaxChunks chunks of at least 32 keys (a multiple of 4)
constexpr uint32_t kMaxChunks = 64;
void dec_chunks(uint32_t n, uint32_t* chunk, uint32_t* nchunks) {
  uint32_t c = std::max(32u, (n + kMaxChunks - 1) / kMaxChunks);
  c = (c + 3) / 4 * 4;
  *chunk = c;
  *nchunks = (n + c - 1) / c;
}

// The validation layer counts static threadgroup memory twice (AGENTS.md): under it, the 16 KB prefill variant
bool small_tg() {
  static const bool on = [] {
    for (const char* v : {"MTL_SHADER_VALIDATION", "MTL_DEBUG_LAYER", "NI_HY4_ATTN_K16"}) {
      const char* s = getenv(v);
      if (s && *s && strcmp(s, "0") != 0) return true;
    }
    return false;
  }();
  return on;
}

}  // namespace

std::vector<std::string> kernel_names() {
  return {"hy4_hc_pre",   "hy4_hc_inject",       "hy4_hc_expand", "hy4_rmsnorm",
          "hy4_add",      "hy4_swiglu",          "hy4_router",    "hy4_engram",
          "hy4_attn_prep", "hy4_attn_dec",       "hy4_attn_dec_reduce",
          small_tg() ? "hy4_attn_prefill_k16" : "hy4_attn_prefill", "kv_copy_prefix",
          "hy4_mla_write", "hy4_mla_qprep", "hy4_mla_gemm", "hy4_mla_kr", "hy4_mla_attn", "hy4_mla_reduce",
          "hy4_mla_pack", "hy4_mla_uk", "hy4_mla_uv", "hy4_mla_expand",
          "hy4_mla_attn_sel", "hy4_ix_tok", "hy4_ix_pool", "hy4_ix_q", "hy4_ix_score", "hy4_ix_select", "hy4_ix_pick"};
}

// ---- hyper-connections ----
void enc_hc_pre(Encoder& e, const Hc& w, const HcConst& c, int R, const Buffer& X, const Buffer* y,
                const Buffer* post_prev, const Buffer& u, bool u_half, const Buffer* post_out) {
  const Tensor& fn = need(w.fn, "hc fn");
  const Tensor& base = need(w.base, "hc base");
  const Tensor& scale = need(w.scale, "hc scale");
  if (w.nout != 4 && w.nout != 8) fatal("hy4: hc nout %u (4 or 8)", w.nout);
  if (fn.numel() != int64_t(w.nout) * kStreams * c.D || base.numel() != w.nout)
    fatal("hy4: hc fn/base shape does not match nout %u, D %u", w.nout, c.D);
  if ((y != nullptr) != (post_prev != nullptr)) fatal("hy4: hc_pre needs y and post_prev together");
  const bool has_post = w.nout == 8 && post_out;
  if (w.nout == 8 && !post_out) fatal("hy4: hc_pre nout 8 needs post_out");
  const kp::Hy4Hc p{c.D, w.nout, y ? 1u : 0u, u_half ? 1u : 0u, has_post ? 1u : 0u, dt(fn, "hc fn"),
                    dt(base, "hc base"), dt(scale, "hc scale"), c.rms_eps, c.hc_eps, c.mag};
  e.use(pipe("hy4_hc_pre"));
  e.buf(0, X);
  e.buf(1, y ? *y : X);
  e.buf(2, post_prev ? *post_prev : X);
  e.buf(3, fn.buf);
  e.buf(4, base.buf);
  e.buf(5, scale.buf);
  e.buf(6, u);
  e.buf(7, has_post ? *post_out : X);
  e.val(8, p);
  e.dispatch(uint32_t(R), 1, 1, threads_for(c.D));
}

void enc_hc_inject(Encoder& e, uint32_t D, int R, const Buffer& X, const Buffer& y, const Buffer& post) {
  e.use(pipe("hy4_hc_inject"));
  e.buf(0, X);
  e.buf(1, y);
  e.buf(2, post);
  e.val(3, D);
  e.dispatch((D + 255) / 256, uint32_t(R), 1, 256);
}

void enc_hc_expand(Encoder& e, uint32_t D, int R, const Buffer& emb, const Buffer& X) {
  e.use(pipe("hy4_hc_expand"));
  e.buf(0, emb);
  e.buf(1, X);
  e.val(2, D);
  e.dispatch((D + 255) / 256, uint32_t(R), 1, 256);
}

// ---- rows ----
void enc_rmsnorm(Encoder& e, uint32_t D, int R, const Buffer& x, const Tensor& w, float eps, const Buffer& y,
                 bool y_half) {
  if (w.numel() != D) fatal("hy4: rmsnorm weight has %lld elements, D %u", (long long)w.numel(), D);
  const kp::Hy4Norm p{D, dt(w, "rmsnorm weight"), y_half ? 1u : 0u, eps};
  e.use(pipe("hy4_rmsnorm"));
  e.buf(0, x);
  e.buf(1, w.buf);
  e.buf(2, y);
  e.val(3, p);
  e.dispatch(uint32_t(R), 1, 1, threads_for(D));
}

void enc_add(Encoder& e, uint32_t n_per_row, int R, const Buffer& x, const Buffer& y) {
  const uint32_t n = n_per_row * uint32_t(R);
  e.use(pipe("hy4_add"));
  e.buf(0, x);
  e.buf(1, y);
  e.val(2, n);
  e.dispatch((n + 255) / 256, 1, 1, 256);
}

void enc_swiglu(Encoder& e, uint32_t n, int R, const Buffer& g, const Buffer& u, const Buffer& h, float limit,
                bool h_half) {
  const kp::Hy4Swiglu p{n * uint32_t(R), h_half ? 1u : 0u, limit};
  e.use(pipe("hy4_swiglu"));
  e.buf(0, g);
  e.buf(1, u);
  e.buf(2, h);
  e.val(3, p);
  e.dispatch((p.n + 255) / 256, 1, 1, 256);
}

// ---- router ----
void enc_router(Encoder& e, uint32_t E, uint32_t k, float scale, int R, const Buffer& logits, const Tensor& bias,
                const Buffer& idx, const Buffer& wts) {
  if (E > 512 || k > 16 || k == 0 || k > E) fatal("hy4: router E %u k %u (E <= 512, 1 <= k <= 16)", E, k);
  if (bias.numel() != E) fatal("hy4: router bias has %lld elements, E %u", (long long)bias.numel(), E);
  const kp::Hy4Router p{uint32_t(R), E, k, dt(bias, "router bias"), scale};
  e.use(pipe("hy4_router"));
  e.buf(0, logits);
  e.buf(1, bias.buf);
  e.buf(2, idx);
  e.buf(3, wts);
  e.val(4, p);
  e.dispatch((uint32_t(R) + 3) / 4, 1, 1, 128);
}

// ---- Engram ----
void enc_engram(Encoder& e, const EngramW& w, uint32_t D, float engram_eps, int R, const Buffer& vraw,
                const Buffer& kraw, const Buffer& X, const Buffer* mask) {
  auto group_t = [&](const Tensor* const* ts, const char* what) {
    const uint32_t t = dt(need(ts[0], what), what);
    for (uint32_t s = 0; s < kStreams; ++s) {
      const Tensor& x = need(ts[s], what);
      if (dt(x, what) != t || x.numel() != D) fatal("hy4: engram %s: mixed dtypes or not [D]", what);
    }
    return t;
  };
  const Tensor& vb = need(w.vb, "engram value bias");
  const Tensor& pg = need(w.pre_gamma, "engram pre_gamma");
  if (vb.numel() != D || pg.numel() != D) fatal("hy4: engram value bias / pre_gamma not [D]");
  const kp::Hy4Engram p{D,
                        uint32_t(R),
                        dt(vb, "engram value bias"),
                        group_t(w.kb, "key bias"),
                        group_t(w.nk, "norm_key"),
                        group_t(w.nq, "norm_query"),
                        dt(pg, "engram pre_gamma"),
                        group_t(w.hc_scale, "per_hc_scale"),
                        mask ? 1u : 0u,
                        engram_eps,
                        float(1.0 / std::sqrt(double(D)))};
  e.use(pipe("hy4_engram"));
  e.buf(0, X);
  e.buf(1, vraw);
  e.buf(2, kraw);
  e.buf(3, vb.buf);
  for (uint32_t s = 0; s < kStreams; ++s) {
    e.buf(4 + s, w.kb[s]->buf);
    e.buf(8 + s, w.nk[s]->buf);
    e.buf(12 + s, w.nq[s]->buf);
    e.buf(17 + s, w.hc_scale[s]->buf);
  }
  e.buf(16, pg.buf);
  e.val(21, p);
  e.buf(22, mask ? *mask : X);  // unread without a mask
  e.dispatch(uint32_t(R), 1, 1, threads_for(D));
}

// ---- attention ----
size_t kv_bytes_per_pos(uint32_t H) { return 2 * size_t(H) * kv_row_bytes(256); }

KvCache alloc_kv(uint32_t H, uint32_t cap, const char* label, bool ring) {
  KvCache kv;
  kv.H = H;
  kv.cap = cap;
  kv.ring = ring;
  const size_t n = size_t(H) * cap * kv_row_bytes(256);
  Gpu& g = Gpu::get();
  kv.k = g.alloc(n, label);
  kv.v = g.alloc(n, label);
  // zeros: a ring's rows are all written before they are read, but a tile can touch a never-written row
  memset(kv.k.ptr(), 0, n);
  memset(kv.v.ptr(), 0, n);
  return kv;
}

void free_kv(KvCache& kv) {
  Gpu& g = Gpu::get();
  g.release(kv.k);
  g.release(kv.v);
  kv.cap = 0;
}

void grow_kv(KvCache& kv, uint32_t cap, uint32_t keep) {
  if (kv.ring) fatal("hy4: grow_kv on a ring cache");
  if (cap <= kv.cap) return;
  keep = std::min(keep, kv.cap);
  Gpu& g = Gpu::get();
  const size_t n = size_t(kv.H) * cap * kv_row_bytes(256);
  Buffer nk = g.alloc(n, "hy4_k_cache"), nv = g.alloc(n, "hy4_v_cache");
  if (keep > 0) {
    const uint32_t w = uint32_t(kv_row_bytes(256) / 16);  // 16-byte words a row
    const uint32_t p[4] = {keep * w, kv.cap * w, cap * w, 0};  // (n, src_cap, dst_cap) per head, uint3 = 16 bytes
    auto e = g.begin();
    for (int i = 0; i < 2; ++i) {
      e->use(pipe("kv_copy_prefix"));
      e->buf(0, i ? kv.v : kv.k);
      e->buf(1, i ? nv : nk);
      e->bytes(2, p, sizeof p);
      e->dispatch((p[0] + 255) / 256, kv.H, 1, 256);
    }
    e->commit_and_wait();
  }
  g.release(kv.k);
  g.release(kv.v);
  kv.k = nk;
  kv.v = nv;
  kv.cap = cap;
}

size_t attn_scratch_bytes(uint32_t Hmax, int Tmax, uint32_t /*cap*/) {
  const size_t qo = size_t(std::max(Tmax, 1)) * Hmax * 256 * sizeof(float);
  const size_t part = size_t(Hmax) * kMaxChunks * (256 + 2) * sizeof(float);
  return std::max(qo, size_t(Hmax) * 256 * sizeof(float) + part);
}

void enc_attn(Encoder& e, const AttnLayer& L, KvCache& kv, int pos0, int T, const Buffer& q, const Buffer& kvb,
              const Buffer& kr, uint32_t kr_ld, uint32_t kr_off, const Buffer& gate, const Buffer& out, bool out_half,
              const Buffer& scratch) {
  const Tensor& sinks = need(L.sinks, "attention sinks");
  if (sinks.numel() != L.H) fatal("hy4: sinks have %lld elements, H %u", (long long)sinks.numel(), L.H);
  if (kv.H != L.H) fatal("hy4: cache has %u heads, layer %u", kv.H, L.H);
  if (T < 1 || pos0 < 0) fatal("hy4: enc_attn T %d pos0 %d", T, pos0);
  const uint32_t last = uint32_t(pos0 + T - 1);
  if (kv.ring) {
    if (!L.window || kv.cap < L.window + uint32_t(T) - 1)
      fatal("hy4: ring cache of %u rows for window %u and %d rows (needs window + T - 1)", kv.cap, L.window, T);
  } else if (last >= kv.cap) {
    fatal("hy4: position %u past the cache's %u", last, kv.cap);
  }
  if (scratch.size && scratch.size < attn_scratch_bytes(L.H, T))
    fatal("hy4: attention scratch %zu B, needs %zu", scratch.size, attn_scratch_bytes(L.H, T));
  const uint32_t sink_t = dt(sinks, "sinks");
  const float scale = 1.0f / 16.0f;  // 1 / sqrt(qk_head_dim 256)
  const Buffer& qo = scratch;      // fp32 [T, H, 256]

  const kp::Hy4AttnPrep pp{uint32_t(T), L.H,  uint32_t(pos0), kv.cap, kv.ring ? 1u : 0u,
                           kr_ld,       kr_off, L.rope_bf16 ? 1u : 0u, L.theta};
  e.use(pipe("hy4_attn_prep"));
  e.buf(0, q);
  e.buf(1, kvb);
  e.buf(2, kr);
  e.buf(3, qo);
  e.buf(4, kv.k);
  e.buf(5, kv.v);
  e.val(6, pp);
  e.dispatch(L.H, uint32_t(T), 1, 256);

  if (T == 1) {
    const uint32_t Lk = last + 1;
    const uint32_t lo = L.window && Lk > L.window ? Lk - L.window : 0;
    uint32_t chunk, nch;
    dec_chunks(Lk - lo, &chunk, &nch);
    const size_t qo_bytes = size_t(L.H) * 256 * sizeof(float);
    const size_t po_bytes = size_t(L.H) * nch * 256 * sizeof(float);
    const kp::Hy4AttnDec dp{L.H, Lk, lo, chunk, nch, kv.cap, kv.ring ? 1u : 0u, sink_t, out_half ? 1u : 0u, scale};
    e.use(pipe("hy4_attn_dec"));
    e.buf(0, qo);
    e.buf(1, kv.k);
    e.buf(2, kv.v);
    e.buf(3, scratch, qo_bytes);
    e.buf(4, scratch, qo_bytes + po_bytes);
    e.val(5, dp);
    e.dispatch((nch + 3) / 4, L.H, 1, 128);
    e.use(pipe("hy4_attn_dec_reduce"));
    e.buf(0, scratch, qo_bytes);
    e.buf(1, scratch, qo_bytes + po_bytes);
    e.buf(2, gate);
    e.buf(3, sinks.buf);
    e.buf(4, out);
    e.val(5, dp);
    e.dispatch(L.H, 1, 1, 256);
    return;
  }
  const kp::AttnPref ap{uint32_t(T), L.H, L.H, uint32_t(pos0), kv.cap, scale};
  const kp::Hy4AttnHy hy{L.window, kv.ring ? kv.cap : 0u, sink_t, out_half ? 1u : 0u};
  e.use(pipe(small_tg() ? "hy4_attn_prefill_k16" : "hy4_attn_prefill"));
  e.buf(0, qo);
  e.buf(1, kv.k);
  e.buf(2, kv.v);
  e.buf(3, gate);
  e.buf(4, out);
  e.val(5, ap);
  e.buf(6, sinks.buf);
  e.val(7, hy);
  e.dispatch((uint32_t(T) + 31) / 32, L.H, 1, 256);
}

// ---- latent MLA ----
namespace {
constexpr int kAbsorbMaxT = 16;
constexpr uint32_t kMlaTile = 16;     // hy4_mla_attn's key tile
constexpr uint32_t kMlaPartRows = 8192;  // nsplit * rows bound: part_o is at most 8192 x 2 KB
constexpr size_t al256(size_t n) { return (n + 255) / 256 * 256; }

bool absorbed_prefill() {
  static const bool on = [] {
    const char* s = getenv("NI_HY4_MLA_PREFILL");
    return s && !strcmp(s, "absorbed");
  }();
  return on;
}
// splits of the key range: ~kMlaTgs threadgroups, at least 4 key tiles a split, part_o bounded
constexpr uint32_t kMlaTgs = 256;
uint32_t mla_nsplit(uint32_t R, uint32_t ntiles) {
  const uint32_t n = std::min({64u, kMlaPartRows / R, (kMlaTgs * 16 + R - 1) / R, (ntiles + 3) / 4});
  return std::max(1u, n);
}
uint32_t mla_nsplit_max(uint32_t R) { return std::max(1u, std::min({64u, kMlaPartRows / R, (kMlaTgs * 16 + R - 1) / R})); }
// absorbed: qh half [R][576] | olat fp32 [R][512] | part_o fp32 [nsplit*R][512] | part_ml float2 [nsplit*R]
size_t absorbed_bytes(uint32_t H, int T) {
  const size_t R = size_t(H) * T;
  const uint32_t ns = mla_nsplit_max(uint32_t(R));
  const size_t parts = ns > 1 ? ns * R : 0;
  return al256(R * kLatentRowBytes) + al256(R * 512 * 4) + al256(parts * 512 * 4) + al256(parts * 8);
}
// expanded: qo fp32 [T][H][256] | K half [H][M][256] | V same | (NI_KV_I8) K, V rows
size_t expand_bytes(uint32_t H, int T, uint32_t M) {
  const size_t kv = al256(size_t(H) * M * 512);
  return al256(size_t(T) * H * 1024) + 2 * kv + (kv_i8() ? 2 * al256(size_t(H) * M * kv_row_bytes(256)) : 0);
}
}  // namespace

LatentCache alloc_latent(uint32_t cap, const char* label, bool ring) {
  LatentCache c;
  c.cap = (std::max(cap, 1u) + 15) / 16 * 16;
  c.ring = ring;
  const size_t n = size_t(c.cap) * kLatentRowBytes;
  c.rows = Gpu::get().alloc(n, label);
  memset(c.rows.ptr(), 0, n);  // a tile can read a row before any position wrote it (masked, but must be finite)
  return c;
}

void free_latent(LatentCache& c) {
  Gpu::get().release(c.rows);
  c.cap = 0;
}

void grow_latent(LatentCache& c, uint32_t cap, uint32_t keep) {
  if (c.ring) fatal("hy4: grow_latent on a ring cache");
  cap = (cap + 15) / 16 * 16;
  if (cap <= c.cap) return;
  keep = std::min(keep, c.cap);
  Gpu& g = Gpu::get();
  Buffer nr = g.alloc(size_t(cap) * kLatentRowBytes, "hy4_latent_cache");
  memset(nr.ptr(), 0, size_t(cap) * kLatentRowBytes);
  if (keep > 0) {
    const uint32_t w = uint32_t(kLatentRowBytes / 16);
    const uint32_t p[4] = {keep * w, c.cap * w, cap * w, 0};
    auto e = g.begin();
    e->use(pipe("kv_copy_prefix"));
    e->buf(0, c.rows);
    e->buf(1, nr);
    e->bytes(2, p, sizeof p);
    e->dispatch((p[0] + 255) / 256, 1, 1, 256);
    e->commit_and_wait();
  }
  g.release(c.rows);
  c.rows = nr;
  c.cap = cap;
}

void enc_latent_write(Encoder& e, const AttnLayer& L, LatentCache& c, int pos0, int T, const Buffer& lat,
                      uint32_t lat_ld, const Buffer& kr, uint32_t kr_ld, uint32_t kr_off) {
  if (T < 1 || pos0 < 0) fatal("hy4: enc_latent_write T %d pos0 %d", T, pos0);
  if (!c.ring && uint32_t(pos0 + T) > c.cap) fatal("hy4: position %d past the latent cache's %u", pos0 + T - 1, c.cap);
  if (c.ring && uint32_t(T) > c.cap) fatal("hy4: %d rows into a ring of %u", T, c.cap);
  const kp::Hy4MlaWrite p{uint32_t(pos0), c.cap, c.ring ? 1u : 0u, lat_ld, kr_ld, kr_off, L.rope_bf16 ? 1u : 0u,
                          L.theta};
  e.use(pipe("hy4_mla_write"));
  e.buf(0, lat);
  e.buf(1, kr);
  e.buf(2, c.rows);
  e.val(3, p);
  e.dispatch(uint32_t(T), 1, 1, 64);
}

size_t latent_scratch_bytes(uint32_t Hmax, int Tmax, uint32_t max_keys) {
  // absorbed at every T: an indexer selection (sel) runs the absorbed path whatever T is
  size_t n = absorbed_bytes(Hmax, std::max(Tmax, 1));
  if (Tmax > kAbsorbMaxT && !absorbed_prefill()) n = std::max(n, expand_bytes(Hmax, Tmax, max_keys + 32));
  return n;
}

void enc_attn_latent(Encoder& e, const AttnLayer& L, LatentCache& c, int pos0, int T, const Buffer& q, uint32_t q_ld,
                     const Buffer& lat, uint32_t lat_ld, const Buffer& kr, uint32_t kr_ld, uint32_t kr_off,
                     const Tensor& kv_b, const Buffer& gate, const Buffer& out, bool out_half, const Buffer& scratch,
                     const Buffer* sel, bool sel_one) {
  const Tensor& sinks = need(L.sinks, "attention sinks");
  const uint32_t H = L.H;
  if (H == 0 || H % 16) fatal("hy4: enc_attn_latent needs H %% 16 == 0 (H %u)", H);
  if (sinks.numel() != H) fatal("hy4: sinks have %lld elements, H %u", (long long)sinks.numel(), H);
  if (kv_b.dtype != DType::BF16 || kv_b.numel() != int64_t(H) * 448 * 512)
    fatal("hy4: kv_b must be bf16 [%u, 512]", H * 448);
  if (T < 1 || pos0 < 0) fatal("hy4: enc_attn_latent T %d pos0 %d", T, pos0);
  const uint32_t last = uint32_t(pos0 + T - 1);
  if (c.ring) {
    if (!L.window || c.cap < L.window + uint32_t(T) - 1)
      fatal("hy4: ring latent cache of %u rows for window %u and %d rows (needs window + T - 1)", c.cap, L.window, T);
  } else if (last >= c.cap) {
    fatal("hy4: position %u past the latent cache's %u", last, c.cap);
  }
  const uint32_t sink_t = dt(sinks, "sinks");
  const float scale = 1.0f / 16.0f;
  const uint32_t ring = c.ring ? 1u : 0u;
  enc_latent_write(e, L, c, pos0, T, lat, lat_ld, kr, kr_ld, kr_off);

  const auto lo_of = [&](uint32_t kend) { return L.window && kend > L.window ? kend - L.window : 0u; };
  if (sel && (c.ring || L.window)) fatal("hy4: an indexer selection on a window layer");
  if (sel && sel_one && pos0 < 1) fatal("hy4: a shared selection row at pos0 0");
  if (T <= kAbsorbMaxT || absorbed_prefill() || sel) {
    const uint32_t R = H * uint32_t(T);
    const size_t o_qh = 0, o_olat = al256(size_t(R) * kLatentRowBytes), o_part = o_olat + al256(size_t(R) * 2048);
    const uint32_t ntiles = sel && sel_one ? (std::min(uint32_t(pos0 - 1) / kIndexRatio, kIndexGroups - 1) + 1 +
                                              (uint32_t(T) + 3) / 4 + 3) / 4
                            : sel ? (std::min(last / kIndexRatio, kIndexGroups - 1) + 1 + 3) / 4
                                : (last + 1 + kMlaTile - 1) / kMlaTile - lo_of(uint32_t(pos0) + 1) / kMlaTile;
    const uint32_t nsplit = mla_nsplit(R, ntiles);
    const size_t parts = nsplit > 1 ? size_t(nsplit) * R : 0;
    const size_t o_ml = o_part + al256(parts * 2048), o_end = o_ml + al256(parts * 8);
    if (scratch.size < o_end) fatal("hy4: latent scratch %zu B, needs %zu", scratch.size, o_end);
    const bool small = T <= kAbsorbMaxT;
    const float qscale = scale * 1.4426950408889634f;
    // q: nope through W_UK (mode 0), rope dims roped (qprep mode 0), both times log2(e)/16, half
    const kp::Hy4MlaQ qp{uint32_t(T), H, uint32_t(pos0), q_ld, 0, L.rope_bf16 ? 1u : 0u, L.theta, qscale};
    e.use(pipe("hy4_mla_qprep"));
    e.buf(0, q);
    e.buf(1, scratch, o_qh);
    e.val(2, qp);
    e.dispatch(H, uint32_t(T), 1, 256);
    const kp::Hy4MlaSmall sp{uint32_t(T), H, q_ld, out_half ? 1u : 0u, H * 256, H * 256, qscale};
    if (small) {
      e.use(pipe("hy4_mla_uk"));
      e.buf(0, q);
      e.buf(1, kv_b.buf);
      e.buf(2, scratch, o_qh);
      e.val(3, sp);
      e.dispatch(4, H, 1, 128);
    } else if (!small) {
      const kp::Hy4MlaGemm g0{uint32_t(T), 512, 192, H, q_ld, 256, 0, 0, 0, 0, qscale};
      e.use(pipe("hy4_mla_gemm"));
      e.buf(0, q);
      e.buf(1, kv_b.buf);
      e.buf(2, scratch, o_qh);
      e.buf(4, gate);
      e.val(5, g0);
      e.dispatch(512 / 32, (uint32_t(T) + 31) / 32, H, 128);
    }
    const uint32_t per = (ntiles + nsplit - 1) / nsplit;
    const kp::Hy4MlaAttn ap{uint32_t(T), H, uint32_t(pos0), c.cap, ring, L.window, nsplit, per, sink_t, sel && sel_one ? 1u : 0u};
    e.use(pipe(sel ? "hy4_mla_attn_sel" : "hy4_mla_attn"));
    e.buf(0, scratch, o_qh);
    e.buf(1, c.rows);
    e.buf(2, sinks.buf);
    e.buf(3, scratch, o_olat);
    e.buf(4, scratch, nsplit > 1 ? o_part : o_olat);  // unused at nsplit 1
    e.buf(5, scratch, nsplit > 1 ? o_ml : o_olat);
    e.val(6, ap);
    e.buf(7, sel ? *sel : scratch);  // unused without sel
    e.dispatch(R / 16, nsplit, 1, 256);
    if (nsplit > 1) {
      e.use(pipe("hy4_mla_reduce"));
      e.buf(0, scratch, o_part);
      e.buf(1, scratch, o_ml);
      e.buf(2, sinks.buf);
      e.buf(3, scratch, o_olat);
      e.val(4, ap);
      e.dispatch(R, 1, 1, 128);
    }
    if (small) {
      e.use(pipe("hy4_mla_uv"));
      e.buf(0, scratch, o_olat);
      e.buf(1, kv_b.buf);
      e.buf(2, out);
      e.buf(3, gate);
      e.val(4, sp);
      e.dispatch(16, H, 1, 128);
      return;
    }
    const kp::Hy4MlaGemm g1{uint32_t(T), 256, 512, H, H * 512, 512, 1, out_half ? 1u : 0u, H * 256, H * 256, 1.0f};
    e.use(pipe("hy4_mla_gemm"));
    e.buf(0, scratch, o_olat);
    e.buf(1, kv_b.buf);
    e.buf(2, out);
    e.buf(4, gate);
    e.val(5, g1);
    e.dispatch(256 / 32, (uint32_t(T) + 31) / 32, H, 128);
    return;
  }
  // T > 16: expand the attended keys' K/V from the latent rows (positions kbase.., kbase 32-aligned so every key
  // tile the prefill kernel reads is expanded), then enc_attn's prefill kernel on positions relative to kbase.
  const uint32_t kbase = lo_of(uint32_t(pos0) + 1) / 32 * 32;
  const uint32_t M = last + 1 - kbase;
  const size_t o_qo = 0, kvb_bytes = al256(size_t(H) * M * 512);
  const size_t o_k = al256(size_t(T) * H * 1024), o_v = o_k + kvb_bytes, o_end = o_v + kvb_bytes;
  const size_t need_b = o_end + (kv_i8() ? 2 * al256(size_t(H) * M * kv_row_bytes(256)) : 0);
  if (scratch.size < need_b) fatal("hy4: latent scratch %zu B, needs %zu", scratch.size, need_b);
  const kp::Hy4MlaQ qp{uint32_t(T), H, uint32_t(pos0), q_ld, 1, L.rope_bf16 ? 1u : 0u, L.theta, 1.0f};
  e.use(pipe("hy4_mla_qprep"));
  e.buf(0, q);
  e.buf(1, scratch, o_qo);
  e.val(2, qp);
  e.dispatch(H, uint32_t(T), 1, 256);
  const kp::Hy4MlaExp xp{M, c.cap, ring, kbase};
  e.use(pipe("hy4_mla_expand"));
  e.buf(0, c.rows);
  e.buf(1, kv_b.buf);
  e.buf(2, scratch, o_k);
  e.buf(3, scratch, o_v);
  e.val(4, xp);
  e.dispatch(448 / 64, (M + 63) / 64, H, 128);
  const kp::Hy4MlaKr kp_{M, c.cap, ring, kbase};
  e.use(pipe("hy4_mla_kr"));
  e.buf(0, c.rows);
  e.buf(1, scratch, o_k);
  e.val(2, kp_);
  e.dispatch((M * 64 + 255) / 256, H, 1, 256);
  size_t ok = o_k, ov = o_v;
  if (kv_i8()) {
    const size_t pk = o_end, pv = o_end + al256(size_t(H) * M * kv_row_bytes(256));
    for (int i = 0; i < 2; ++i) {
      e.use(pipe("hy4_mla_pack"));
      e.buf(0, scratch, i ? o_v : o_k);
      e.buf(1, scratch, i ? pv : pk);
      e.dispatch(H * M, 1, 1, 256);
    }
    ok = pk;
    ov = pv;
  }
  const kp::AttnPref ap{uint32_t(T), H, H, uint32_t(pos0) - kbase, M, scale};
  const kp::Hy4AttnHy hy{L.window, 0u, sink_t, out_half ? 1u : 0u};
  e.use(pipe(small_tg() ? "hy4_attn_prefill_k16" : "hy4_attn_prefill"));
  e.buf(0, scratch, o_qo);
  e.buf(1, scratch, ok);
  e.buf(2, scratch, ov);
  e.buf(3, gate);
  e.buf(4, out);
  e.val(5, ap);
  e.buf(6, sinks.buf);
  e.val(7, hy);
  e.dispatch((uint32_t(T) + 31) / 32, H, 1, 256);
}

// ---- compressed indexer ----
namespace {
constexpr size_t kIxRowBytes = 256;  // 128 half
constexpr float kIxEps = 1e-5f;      // rms_norm_eps
}  // namespace

IndexerCache alloc_indexer(uint32_t cap, const char* label) {
  IndexerCache c;
  c.cap = (std::max(cap, 1u) + 15) / 16 * 16;
  c.tok = Gpu::get().alloc(size_t(c.cap) * kIxRowBytes, label);
  c.grp = Gpu::get().alloc(size_t(c.cap / kIndexRatio) * kIxRowBytes, label);
  memset(c.tok.ptr(), 0, size_t(c.cap) * kIxRowBytes);
  memset(c.grp.ptr(), 0, size_t(c.cap / kIndexRatio) * kIxRowBytes);
  return c;
}

void free_indexer(IndexerCache& c) {
  Gpu::get().release(c.tok);
  Gpu::get().release(c.grp);
  c.cap = 0;
}

void grow_indexer(IndexerCache& c, uint32_t cap, uint32_t keep) {
  cap = (cap + 15) / 16 * 16;
  if (cap <= c.cap) return;
  keep = std::min(keep, c.cap);
  Gpu& g = Gpu::get();
  IndexerCache n = alloc_indexer(cap, "hy4_indexer_cache");
  const uint32_t w = uint32_t(kIxRowBytes / 16);
  auto e = g.begin();
  const struct {
    const Buffer *src, *dst;
    uint32_t rows, cap_old, cap_new;
  } parts[2] = {{&c.tok, &n.tok, keep, c.cap, cap},
                {&c.grp, &n.grp, keep / kIndexRatio, c.cap / kIndexRatio, cap / kIndexRatio}};
  for (auto& q : parts) {
    if (!q.rows) continue;
    const uint32_t p[4] = {q.rows * w, q.cap_old * w, q.cap_new * w, 0};
    e->use(pipe("kv_copy_prefix"));
    e->buf(0, *q.src);
    e->buf(1, *q.dst);
    e->bytes(2, p, sizeof p);
    e->dispatch((p[0] + 255) / 256, 1, 1, 256);
  }
  e->commit_and_wait();
  free_indexer(c);
  c = n;
}

void enc_index_write(Encoder& e, const AttnLayer& L, const IndexerWeights& w, IndexerCache& c, int pos0, int T,
                     const Buffer& tk, uint32_t tk_ld, uint32_t tk_off) {
  const Tensor& tkn = need(w.tk_norm, "token_key_norm");
  const Tensor& wv = need(w.wv, "key_value_projection");
  const Tensor& wg = need(w.wg, "key_gate_projection");
  const Tensor& ape = need(w.ape, "intra_group_ape");
  const Tensor& kn = need(w.key_norm, "key_norm");
  if (wv.dtype != DType::F32 || wg.dtype != DType::F32 || ape.dtype != DType::F32 || wv.numel() != 128 * 128 ||
      wg.numel() != 128 * 128 || ape.numel() != 4 * 128 || tkn.numel() != 128 || kn.numel() != 128)
    fatal("hy4: indexer pooling weights: f32 [128,128] x 2, f32 [4,128], norms [128]");
  if (T < 1 || pos0 < 0 || uint32_t(pos0 + T) > c.cap)
    fatal("hy4: enc_index_write pos0 %d T %d past the indexer cache's %u", pos0, T, c.cap);
  const kp::Hy4IxTok tp{uint32_t(pos0), tk_ld, tk_off, dt(tkn, "token_key_norm"), kIxEps};
  e.use(pipe("hy4_ix_tok"));
  e.buf(0, tk);
  e.buf(1, tkn.buf);
  e.buf(2, c.tok);
  e.val(3, tp);
  e.dispatch(uint32_t(T), 1, 1, 128);
  const uint32_t g0 = uint32_t(pos0) / kIndexRatio, g1 = uint32_t(pos0 + T) / kIndexRatio;
  if (g1 <= g0) return;
  const kp::Hy4IxPool pp{g0, dt(kn, "key_norm"), kIxEps, L.theta, L.rope_bf16 ? 1u : 0u};
  e.use(pipe("hy4_ix_pool"));
  e.buf(0, c.tok);
  e.buf(1, wv.buf);
  e.buf(2, wg.buf);
  e.buf(3, ape.buf);
  e.buf(4, kn.buf);
  e.buf(5, c.grp);
  e.val(6, pp);
  e.dispatch(g1 - g0, 1, 1, 128);
}

// scratch: qx half [T][16][128] | hwx fp32 [T][16] | logits fp32 [T][gld], gld = max_pos / 4 rounded up to 32
size_t index_scratch_bytes(int Tmax, uint32_t max_pos) {
  const size_t T = size_t(std::max(Tmax, 1)), gld = (max_pos / kIndexRatio + 31) / 32 * 32;
  return al256(T * 16 * 128 * 2) + al256(T * 16 * 4) + al256(T * gld * 4);
}

void enc_index_select(Encoder& e, const AttnLayer& L, const IndexerCache& c, int pos0, int T, const Buffer& qi,
                      uint32_t qi_ld, uint32_t qi_off, const Buffer& hw, uint32_t hw_ld, uint32_t hw_off,
                      const Buffer& sel, const Buffer& scratch) {
  if (T < 1 || pos0 < 0 || uint32_t(pos0 + T) > c.cap) fatal("hy4: enc_index_select pos0 %d T %d", pos0, T);
  const uint32_t last = uint32_t(pos0 + T - 1);
  const uint32_t gld = (last / kIndexRatio + 31) / 32 * 32;
  const size_t o_q = 0, o_hw = al256(size_t(T) * 16 * 128 * 2), o_lg = o_hw + al256(size_t(T) * 16 * 4);
  const size_t need_b = o_lg + al256(size_t(T) * gld * 4);
  if (scratch.size < need_b) fatal("hy4: indexer scratch %zu B, needs %zu", scratch.size, need_b);
  if (sel.size < size_t(T) * kIndexGroups * 4) fatal("hy4: sel holds %zu B, needs %d x 512 uint", sel.size, T);
  const kp::Hy4IxQ qp{uint32_t(pos0), qi_ld, qi_off, hw_ld, hw_off, L.rope_bf16 ? 1u : 0u, L.theta};
  e.use(pipe("hy4_ix_q"));
  e.buf(0, qi);
  e.buf(1, hw);
  e.buf(2, scratch, o_q);
  e.buf(3, scratch, o_hw);
  e.val(4, qp);
  e.dispatch(16, uint32_t(T), 1, 128);
  const uint32_t gmax = last / kIndexRatio;
  if (gmax > kIndexGroups - 1) {
    const kp::Hy4IxScore sp{uint32_t(T), uint32_t(pos0), gld};
    e.use(pipe("hy4_ix_score"));
    e.buf(0, scratch, o_q);
    e.buf(1, scratch, o_hw);
    e.buf(2, c.grp);
    e.buf(3, scratch, o_lg);
    e.val(4, sp);
    e.dispatch((gmax + 31) / 32, (uint32_t(T) + 3) / 4, 1, 128);
  }
  const kp::Hy4IxSel lp{uint32_t(pos0), gld};
  e.use(pipe("hy4_ix_select"));
  e.buf(0, scratch, o_lg);
  e.buf(1, sel);
  e.val(2, lp);
  e.dispatch(uint32_t(T), 1, 1, 256);
}

void enc_index_pick(Encoder& e, const Buffer& sel, int T, const Buffer* idx, int row, const Buffer& dst) {
  if (T < 1 || (!idx && (row < 0 || row >= T))) fatal("hy4: enc_index_pick row %d of %d", row, T);
  const kp::Hy4IxPick p{uint32_t(T), uint32_t(std::max(row, 0)), idx ? 1u : 0u};
  e.use(pipe("hy4_ix_pick"));
  e.buf(0, sel);
  e.buf(1, idx ? *idx : sel);
  e.buf(2, dst);
  e.val(3, p);
  e.dispatch(1, 1, 1, kIndexGroups);
}

}  // namespace ni::hy4
