// Dory prefill attention: Kapitsa's attn_prefill11 (src/kernels/attn.metal: one query head per threadgroup, 32
// queries, 8 simdgroups = 4 row groups x 2 halves of the 256 head dims, Q / O in registers, 32-key tiles) with what
// Dory's layers need instead of Qwen's gate:
//  - keys from two places: positions before the chunk from the layer's cache (a full-attention layer's linear cache
//    or a sliding-window layer's ring, slot = position % rows), the chunk's own positions from its K / V buffer
//    ([Hkv][T][256], this loop's keys: last-loop KV caching keeps earlier steps' final-loop keys in the cache while a
//    step's tokens see their own loop's), so a ring needs no room for the chunk;
//  - a sliding window (row t sees keys (pos - window, pos]), tiles from the block's first visible key;
//  - out: fp32 [T][Hq * 256] = softmax(q k^T / 16) v (Dory's gate and norm follow in dory_gate_norm).
// q: fp32 [T][Hq * 256], roped, normalized and scaled (dory_prep). grid: (ceil(T / 32), Hq) x 256.

struct DoryAttnPfP {
  uint T, pos0, Hq, Hkv;
  uint rows, ring, window;
  float scale;
};

constant constexpr uint kDBQ = 32;
constant constexpr uint kDHalf = 128;           // head dims per simdgroup
constant constexpr uint kDVLD = kDD + 8;        // V tile row stride (halves)
constant constexpr float kDNeg = -1.0e30f;
// the K / V tile buffer (halves) for kDBK-key tiles: K transposed [256][kDBK + 8] or V [kDBK][264]
template <uint kDBK> constexpr uint dory_kv_halves() {
  return kDD * (kDBK + 8) > kDBK * kDVLD ? kDD * (kDBK + 8) : kDBK * kDVLD;
}

#ifndef NI_UNROLL
#define NI_UNROLL _Pragma("clang loop unroll(full)")
#endif

// the 512-byte K / V row of key position `key` for kv head kvh
NI_INLINE const device half* dory_kv_row(const device half* cache, const device half* own, uint key,
                                         constant DoryAttnPfP& p, uint kvh) {
  if (key >= p.pos0) return own + (size_t(kvh) * p.T + (key - p.pos0)) * kDD;
  const uint slot = p.ring ? key % p.rows : key;
  return cache + (size_t(kvh) * p.rows + slot) * kDD;
}

template <uint kDBK>
NI_INLINE void dory_attn_prefill_impl(const device float* q, const device half* kc, const device half* vc,
                                      const device half* ck, const device half* cv, device float* out,
                                      constant DoryAttnPfP& p, uint2 tg, uint tid, uint sg, uint lane,
                                      threadgroup half* kv) {
  constexpr uint kDKLD = kDBK + 8;
  constexpr uint NC = kDBK / 8;
  const uint G = p.Hq / p.Hkv;
  const uint rg = sg / 2, dh = sg % 2;
  const uint qd = lane / 4;
  const uint frow = (qd & 4) + (lane / 2) % 4;
  const uint fcol = (qd & 2) * 2 + (lane % 2) * 2;
  const uint h = tg.y, kvh = h / G;
  const uint nqb = (p.T + kDBQ - 1) / kDBQ;
  const uint t0 = (nqb - 1 - tg.x) * kDBQ;          // heaviest (latest) blocks first
  const uint t = t0 + rg * 8 + frow;
  const uint qpos = p.pos0 + t;
  const uint kend = p.pos0 + min(t0 + kDBQ, p.T);   // keys needed by the block (exclusive)
  const uint kfirst = p.window && p.pos0 + t0 + 1 > p.window ? p.pos0 + t0 + 1 - p.window : 0;
  const uint klo = p.window && qpos + 1 > p.window ? qpos + 1 - p.window : 0;  // this row's first key
  const uint it0 = kfirst / kDBK, it1 = (kend + kDBK - 1) / kDBK;

  simdgroup_half8x8 qf[kDHalf / 8];
  {
    const device float* qr = q + (size_t(min(t, p.T - 1)) * p.Hq + h) * kDD + dh * kDHalf + fcol;
    NI_UNROLL for (uint d = 0; d < kDHalf / 8; ++d) {
      const half2 v = t < p.T ? half2(*(const device float2*)(qr + d * 8) * p.scale) : half2(0);
      qf[d].thread_elements()[0] = v.x;
      qf[d].thread_elements()[1] = v.y;
    }
  }
  simdgroup_float8x8 o[kDHalf / 8];
  NI_UNROLL for (uint j = 0; j < kDHalf / 8; ++j) o[j] = simdgroup_float8x8(0);
  float m_row = kDNeg, l_row = 0.0f;
  threadgroup float2* xw = (threadgroup float2*)kv;
  const threadgroup half* kfr = kv + (dh * kDHalf + frow) * kDKLD + fcol;
  const threadgroup half* vfr = kv + frow * kDVLD + dh * kDHalf + fcol;

  for (uint it = it0; it < it1; ++it) {
    const uint k0 = it * kDBK;
    // K rows of the tile: thread = (key tid % 32, a 32-dim slice tid / 32)
    uint4 r[kDBK / 8];
    {
      const uint key = k0 + tid % kDBK;
      const bool ok = key < kend;
      const device uint4* src = (const device uint4*)(dory_kv_row(kc, ck, ok ? key : p.pos0, p, kvh) + (tid / kDBK) * kDBK);
      NI_UNROLL for (uint u = 0; u < kDBK / 8; ++u) r[u] = ok ? src[u] : uint4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      threadgroup half* dst = kv + (tid / kDBK) * kDBK * kDKLD + tid % kDBK;
      NI_UNROLL for (uint u = 0; u < kDBK / 8; ++u)
        NI_UNROLL for (uint w = 0; w < 4; ++w) {
          const half2 h2 = as_type<half2>(r[u][w]);
          dst[(u * 8 + w * 2) * kDKLD] = h2.x;
          dst[(u * 8 + w * 2 + 1) * kDKLD] = h2.y;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 s[NC];
    NI_UNROLL for (uint c = 0; c < NC; ++c) s[c] = simdgroup_float8x8(0);
    NI_UNROLL for (uint d = 0; d < kDHalf / 8; ++d) {
      simdgroup_half8x8 kf[NC];
      NI_UNROLL for (uint c = 0; c < NC; ++c) {
        const half2 k2 = *(const threadgroup half2*)(kfr + d * 8 * kDKLD + c * 8);
        kf[c].thread_elements()[0] = k2.x;
        kf[c].thread_elements()[1] = k2.y;
      }
      NI_UNROLL for (uint c = 0; c < NC; ++c) simdgroup_multiply_accumulate(s[c], qf[d], kf[c], s[c]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // K reads done: the exchange reuses the buffer
    NI_UNROLL for (uint c = 0; c < NC; ++c)
      xw[(sg * NC + c) * 32 + lane] = float2(s[c].thread_elements()[0], s[c].thread_elements()[1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    NI_UNROLL for (uint c = 0; c < NC; ++c) {
      const float2 o2 = xw[((sg ^ 1) * NC + c) * 32 + lane];
      s[c].thread_elements()[0] += o2.x;
      s[c].thread_elements()[1] += o2.y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // exchange reads done: V may overwrite
    {
      // V rows: thread = (key tid / 8, a 32-dim slice tid % 8)
      const uint key = k0 + tid / (kDD / kDBK);
      const bool ok = key < kend;
      const device uint4* src =
          (const device uint4*)(dory_kv_row(vc, cv, ok ? key : p.pos0, p, kvh) + (tid % (kDD / kDBK)) * kDBK);
      threadgroup uint4* dst = (threadgroup uint4*)(kv + (tid / (kDD / kDBK)) * kDVLD + (tid % (kDD / kDBK)) * kDBK);
      NI_UNROLL for (uint u = 0; u < kDBK / 8; ++u) dst[u] = ok ? src[u] : uint4(0);
    }
    float mx = kDNeg;
    NI_UNROLL for (uint c = 0; c < NC; ++c)
      NI_UNROLL for (uint i = 0; i < 2; ++i) {
        const uint key = k0 + c * 8 + fcol + i;
        float v = s[c].thread_elements()[i] * M_LOG2E_F;
        if (key > qpos || key < klo || key >= kend) v = kDNeg;
        s[c].thread_elements()[i] = v;
        mx = max(mx, v);
      }
    mx = max(mx, simd_shuffle_xor(mx, 1));
    mx = max(mx, simd_shuffle_xor(mx, 8));
    const float m_new = max(m_row, mx);
    const float corr = metal::fast::exp2(m_row - m_new);
    float ps = 0.0f;
    NI_UNROLL for (uint c = 0; c < NC; ++c)
      NI_UNROLL for (uint i = 0; i < 2; ++i) {
        const float v = s[c].thread_elements()[i];
        const float e = v <= 0.5f * kDNeg ? 0.0f : metal::fast::exp2(v - m_new);
        s[c].thread_elements()[i] = e;
        ps += e;
      }
    ps += simd_shuffle_xor(ps, 1);
    ps += simd_shuffle_xor(ps, 8);
    l_row = l_row * corr + ps;
    m_row = m_new;
    NI_UNROLL for (uint j = 0; j < kDHalf / 8; ++j) {
      o[j].thread_elements()[0] *= corr;
      o[j].thread_elements()[1] *= corr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // V tile visible
    NI_UNROLL for (uint j = 0; j < kDHalf / 8; ++j) {
      simdgroup_half8x8 vf[NC];
      NI_UNROLL for (uint c = 0; c < NC; ++c) {
        const half2 v2 = *(const threadgroup half2*)(vfr + c * 8 * kDVLD + j * 8);
        vf[c].thread_elements()[0] = v2.x;
        vf[c].thread_elements()[1] = v2.y;
      }
      NI_UNROLL for (uint c = 0; c < NC; ++c) simdgroup_multiply_accumulate(o[j], s[c], vf[c], o[j]);
    }
  }
  if (t < p.T) {
    const float inv = l_row > 0.0f ? 1.0f / l_row : 0.0f;
    device float* dst = out + (size_t(t) * p.Hq + h) * kDD + dh * kDHalf + fcol;
    NI_UNROLL for (uint j = 0; j < kDHalf / 8; ++j)
      *(device float2*)(dst + j * 8) = float2(o[j].thread_elements()[0], o[j].thread_elements()[1]) * inv;
  }
}

// 32-key tiles (20 KB of threadgroup memory); _k16: 16-key tiles (12 KB), the one GPU shader validation can run (its
// instrumentation doubles a kernel's threadgroup memory, over the 32 KB limit at 20 KB)
#define DORY_ATTN_PF(NAME, BK)                                                                                       \
  [[kernel, max_total_threads_per_threadgroup(256)]] void NAME(                                                     \
      const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]], \
      const device half* ck [[buffer(3)]], const device half* cv [[buffer(4)]], device float* out [[buffer(5)]],      \
      constant DoryAttnPfP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],                              \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],                          \
      uint lane [[thread_index_in_simdgroup]]) {                                                                     \
    threadgroup uint4 kvm[dory_kv_halves<BK>() / 8];                                                                 \
    dory_attn_prefill_impl<BK>(q, kc, vc, ck, cv, out, p, tg, tid, sg, lane, (threadgroup half*)kvm);               \
  }
DORY_ATTN_PF(dory_attn_prefill, 32)
DORY_ATTN_PF(dory_attn_prefill_k16, 16)
