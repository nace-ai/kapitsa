#include "common.h"

#ifndef NI_UNROLL
#define NI_UNROLL _Pragma("clang loop unroll(full)")
#endif

// ==== Image encoders (src/vision) ====
// The building blocks a ViT needs besides the GEMMs (which are the bf16 prefill GEMMs, qrf_f16_*): LayerNorm, bias +
// activation, non-causal attention over one image's patches. Then the Qwen3.5 ViT's own steps (qvit_*): patches from
// the resized image, position embeddings interpolated to the grid, 2D RoPE.

// ---- LayerNorm ----
struct VitLnParams {
  uint rows, D, x_ld, y_ld;
  float eps;
  uint has_pb;   // add a pending residual bias first (the previous linear's bias, folded in here)
};

// x[r] += pb (written back when has_pb), y[r] = half((x[r] - mean) / sqrt(var + eps) * w + b). One simdgroup per row.
[[kernel]] void vit_ln(device float* x [[buffer(0)]], const device bfloat* pb [[buffer(1)]],
                       const device bfloat* w [[buffer(2)]], const device bfloat* b [[buffer(3)]],
                       device half* y [[buffer(4)]], constant VitLnParams& p [[buffer(5)]],
                       uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
  const uint r = tg * nsg + sg;
  if (r >= p.rows) return;
  device float* xr = x + size_t(r) * p.x_ld;
  float s = 0.0f;
  for (uint i = lane; i < p.D; i += 32) {
    float v = xr[i];
    if (p.has_pb) {
      v += float(pb[i]);
      xr[i] = v;
    }
    s += v;
  }
  const float mean = simd_sum(s) / float(p.D);
  float q = 0.0f;
  for (uint i = lane; i < p.D; i += 32) {
    const float d = xr[i] - mean;
    q += d * d;
  }
  const float rstd = metal::precise::rsqrt(simd_sum(q) / float(p.D) + p.eps);
  device half* yr = y + size_t(r) * p.y_ld;
  for (uint i = lane; i < p.D; i += 32) yr[i] = half((xr[i] - mean) * rstd * float(w[i]) + float(b[i]));
}

// ---- bias + activation ----
struct VitActParams {
  uint rows, N, ld;  // y rows of ld floats; columns [N, ld) of the output are written 0
  uint act;          // 0 none, 1 GELU (tanh approximation), 2 GELU (erf)
  uint out_ld;       // output row stride (elements)
};

// erf to 1.5e-7 (Abramowitz & Stegun 7.1.26): Metal has no erf
NI_INLINE float vit_erf(float x) {
  const float z = metal::abs(x), t = 1.0f / (1.0f + 0.3275911f * z);
  const float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  const float r = 1.0f - poly * metal::precise::exp(-z * z);
  return x >= 0.0f ? r : -r;
}

NI_INLINE float vit_act(float v, uint act) {
  if (act == 1) return 0.5f * v * (1.0f + metal::precise::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
  if (act == 2) return 0.5f * v * (1.0f + vit_erf(v * 0.7071067811865476f));
  return v;
}

[[kernel]] void vit_bias_act_h(const device float* y [[buffer(0)]], const device bfloat* bias [[buffer(1)]],
                               device half* out [[buffer(2)]], constant VitActParams& p [[buffer(3)]],
                               uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, r = gid.y;
  if (c >= p.ld || r >= p.rows) return;
  out[size_t(r) * p.out_ld + c] = c < p.N ? half(vit_act(y[size_t(r) * p.ld + c] + float(bias[c]), p.act)) : half(0.0h);
}

[[kernel]] void vit_bias_act_b(const device float* y [[buffer(0)]], const device bfloat* bias [[buffer(1)]],
                               device bfloat* out [[buffer(2)]], constant VitActParams& p [[buffer(3)]],
                               uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, r = gid.y;
  if (c >= p.N || r >= p.rows) return;
  out[size_t(r) * p.out_ld + c] = bfloat(vit_act(y[size_t(r) * p.ld + c] + float(bias[c]), p.act));
}

// fp32 rows (+ bias) copied out as fp32 (tests: the hidden state of a block)
[[kernel]] void vit_bias_f32(const device float* y [[buffer(0)]], const device bfloat* bias [[buffer(1)]],
                             device float* out [[buffer(2)]], constant VitActParams& p [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, r = gid.y;
  if (c >= p.N || r >= p.rows) return;
  out[size_t(r) * p.out_ld + c] = y[size_t(r) * p.ld + c] + (p.act ? float(bias[c]) : 0.0f);
}

// ---- non-causal attention over the patches of one image ----
// Q, K, V: half [heads][N][D] (head-major); out: half [N][ld], head h at columns h*D... One threadgroup per (block of
// NSG*RB*8 queries, head); each simdgroup owns RB blocks of 8 query rows over all D dims. Key tiles of BK keys: K^T and V in
// threadgroup memory, S = Q K^T and O += P V on simdgroup matrices, online softmax in the log2 domain.
struct VitAttnParams {
  uint N;       // patches (keys and queries)
  uint ld;      // out row stride (halves)
  uint qb0;     // first query block of this dispatch
  float scale;  // softmax scale (1/sqrt(D))
};

constant constexpr float kVitNeg = -1.0e30f;

template <uint D, uint NSG, uint RB, uint BK>
NI_INLINE void vit_attn_impl(const device half* Q, const device half* K, const device half* V, device half* O,
                             constant VitAttnParams& p, uint2 tg, uint tid, uint sg, uint lane,
                             threadgroup half* kt, threadgroup half* vs) {
  constexpr uint ND = D / 8, NC = BK / 8, NT = NSG * 32;
  constexpr uint KLD = BK + 8;  // K^T: one row per dim, BK keys
  constexpr uint VLD = D + 8;   // V: one row per key, D dims
  const uint h = tg.y;
  const uint q0 = (p.qb0 + tg.x) * NSG * 8 * RB + sg * 8 * RB;  // this simdgroup's RB blocks of 8 queries
  const uint qd = lane / 4;
  const uint frow = (qd & 4) + (lane / 2) % 4, fcol = (qd & 2) * 2 + (lane % 2) * 2;
  const device half* Qh = Q + size_t(h) * p.N * D;
  const device half* Kh = K + size_t(h) * p.N * D;
  const device half* Vh = V + size_t(h) * p.N * D;

  simdgroup_half8x8 qf[RB][ND];
  NI_UNROLL for (uint b = 0; b < RB; ++b) {
    const device half* qr = Qh + size_t(min(q0 + b * 8 + frow, p.N - 1)) * D + fcol;
    NI_UNROLL for (uint d = 0; d < ND; ++d) {
      const half2 v = *(const device half2*)(qr + d * 8);
      qf[b][d].thread_elements()[0] = v.x;
      qf[b][d].thread_elements()[1] = v.y;
    }
  }
  simdgroup_float8x8 o[RB][ND];
  NI_UNROLL for (uint b = 0; b < RB; ++b)
    NI_UNROLL for (uint d = 0; d < ND; ++d) o[b][d] = simdgroup_float8x8(0);
  float m_row[RB], l_row[RB];
  NI_UNROLL for (uint b = 0; b < RB; ++b) m_row[b] = kVitNeg, l_row[b] = 0.0f;
  const float sl2 = p.scale * M_LOG2E_F;
  const uint ntiles = (p.N + BK - 1) / BK;
  for (uint it = 0; it < ntiles; ++it) {
    const uint k0 = it * BK;
    threadgroup_barrier(mem_flags::mem_threadgroup);  // the previous tile's reads are done
    // tile loads: chunk c = (key c % BK, 8 dims c / BK); consecutive lanes take consecutive keys
    for (uint c = tid; c < BK * ND; c += NT) {
      const uint key = c % BK, d8 = c / BK;
      const bool ok = k0 + key < p.N;
      const uint4 kv = ok ? *(const device uint4*)(Kh + size_t(k0 + key) * D + d8 * 8) : uint4(0);
      const uint4 vv = ok ? *(const device uint4*)(Vh + size_t(k0 + key) * D + d8 * 8) : uint4(0);
      threadgroup half* kc = kt + (d8 * 8) * KLD + key;
      const half2 a = as_type<half2>(kv.x), b = as_type<half2>(kv.y), e = as_type<half2>(kv.z), f = as_type<half2>(kv.w);
      kc[0] = a.x;
      kc[KLD] = a.y;
      kc[2 * KLD] = b.x;
      kc[3 * KLD] = b.y;
      kc[4 * KLD] = e.x;
      kc[5 * KLD] = e.y;
      kc[6 * KLD] = f.x;
      kc[7 * KLD] = f.y;
      *(threadgroup uint4*)(vs + key * VLD + d8 * 8) = vv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S (8 queries x BK keys per block) = Q K^T: each K^T fragment serves the RB blocks
    simdgroup_float8x8 s[RB][NC];
    NI_UNROLL for (uint b = 0; b < RB; ++b)
      NI_UNROLL for (uint c = 0; c < NC; ++c) s[b][c] = simdgroup_float8x8(0);
    NI_UNROLL for (uint d = 0; d < ND; ++d) {
      NI_UNROLL for (uint c = 0; c < NC; ++c) {
        simdgroup_half8x8 kf;
        const half2 k2 = *(const threadgroup half2*)(kt + (d * 8 + frow) * KLD + c * 8 + fcol);
        kf.thread_elements()[0] = k2.x;
        kf.thread_elements()[1] = k2.y;
        NI_UNROLL for (uint b = 0; b < RB; ++b) simdgroup_multiply_accumulate(s[b][c], qf[b][d], kf, s[b][c]);
      }
    }
    // online softmax of this lane's rows (a row's other lanes: lane ^ 1, lane ^ 8)
    NI_UNROLL for (uint b = 0; b < RB; ++b) {
      float mx = kVitNeg;
      NI_UNROLL for (uint c = 0; c < NC; ++c)
        NI_UNROLL for (uint i = 0; i < 2; ++i) {
          float v = s[b][c].thread_elements()[i] * sl2;
          if (k0 + c * 8 + fcol + i >= p.N) v = kVitNeg;
          s[b][c].thread_elements()[i] = v;
          mx = max(mx, v);
        }
      mx = max(mx, simd_shuffle_xor(mx, 1));
      mx = max(mx, simd_shuffle_xor(mx, 8));
      const float m_new = max(m_row[b], mx);
      const float corr = metal::fast::exp2(m_row[b] - m_new);
      float ps = 0.0f;
      NI_UNROLL for (uint c = 0; c < NC; ++c)
        NI_UNROLL for (uint i = 0; i < 2; ++i) {
          const float e = metal::fast::exp2(s[b][c].thread_elements()[i] - m_new);
          s[b][c].thread_elements()[i] = e;
          ps += e;
        }
      ps += simd_shuffle_xor(ps, 1);
      ps += simd_shuffle_xor(ps, 8);
      l_row[b] = l_row[b] * corr + ps;
      m_row[b] = m_new;
      NI_UNROLL for (uint d = 0; d < ND; ++d) {
        o[b][d].thread_elements()[0] *= corr;
        o[b][d].thread_elements()[1] *= corr;
      }
    }
    // O += P V: each V fragment serves the RB blocks
    NI_UNROLL for (uint d = 0; d < ND; ++d) {
      NI_UNROLL for (uint c = 0; c < NC; ++c) {
        simdgroup_half8x8 vf;
        const half2 v2 = *(const threadgroup half2*)(vs + (c * 8 + frow) * VLD + d * 8 + fcol);
        vf.thread_elements()[0] = v2.x;
        vf.thread_elements()[1] = v2.y;
        NI_UNROLL for (uint b = 0; b < RB; ++b) simdgroup_multiply_accumulate(o[b][d], s[b][c], vf, o[b][d]);
      }
    }
  }
  NI_UNROLL for (uint b = 0; b < RB; ++b) {
    const uint q = q0 + b * 8 + frow;
    if (q < p.N) {
      const float inv = 1.0f / l_row[b];
      device half* dst = O + size_t(q) * p.ld + h * D + fcol;
      NI_UNROLL for (uint d = 0; d < ND; ++d)
        *(device half2*)(dst + d * 8) = half2(o[b][d].thread_elements()[0] * inv, o[b][d].thread_elements()[1] * inv);
    }
  }
}

#define NI_VIT_ATTN(NAME, D, NSG, RB, BK)                                                                      \
  [[kernel, max_total_threads_per_threadgroup(NSG * 32)]] void NAME(                                           \
      const device half* Q [[buffer(0)]], const device half* K [[buffer(1)]], const device half* V [[buffer(2)]], \
      device half* O [[buffer(3)]], constant VitAttnParams& p [[buffer(4)]],                                  \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                     \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                   \
    threadgroup half kt[D * (BK + 8)];                                                                         \
    threadgroup half vs[BK * (D + 8)];                                                                         \
    vit_attn_impl<D, NSG, RB, BK>(Q, K, V, O, p, tg, tid, sg, lane, kt, vs);                                   \
  }
// threadgroup = NSG simdgroups x RB blocks of 8 queries each; key tiles of 32 (or 64: _k64)
NI_VIT_ATTN(vit_attn_d72_s4, 72, 4, 1, 32)
NI_VIT_ATTN(vit_attn_d72_s8, 72, 8, 1, 32)
NI_VIT_ATTN(vit_attn_d72_s4r2, 72, 4, 2, 32)
NI_VIT_ATTN(vit_attn_d72_s8r2, 72, 8, 2, 32)
NI_VIT_ATTN(vit_attn_d72_s4k64, 72, 4, 1, 64)
NI_VIT_ATTN(vit_attn_d72_s8k64, 72, 8, 1, 64)

// ==== the Qwen3.5 ViT ====

// Patch rows from the resized image: X[r][k] (half [patches, T*P*P*3]) for patch r in merge-block order (2x2 blocks of
// patches, row-major blocks), k = ((frame * P + row) * P + column) * 3 + channel (the MLX Conv3d weight's order); a
// still image fills every frame. lut[channel][value]: the processor's rescale + normalize, computed on the CPU.
struct QvitPatchParams {
  uint W;        // image width (pixels)
  uint gw;       // patches per row
  uint P, T;     // patch size, frames
  uint merge;    // spatial merge (2)
};
[[kernel]] void qvit_patchify(const device uchar* rgb [[buffer(0)]], const device float* lut [[buffer(1)]],
                              device half* X [[buffer(2)]], constant QvitPatchParams& p [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
  const uint k = gid.x, r = gid.y, PP3 = p.P * p.P * 3;
  if (k >= PP3) return;
  const uint mm = p.merge * p.merge, b = r / mm, bw = p.gw / p.merge;
  const uint prow = (b / bw) * p.merge + (r % mm) / p.merge, pcol = (b % bw) * p.merge + r % p.merge;
  const uint c = k % 3, px = (k / 3) % p.P, py = k / (3 * p.P);
  const uint v = rgb[((prow * p.P + py) * p.W + pcol * p.P + px) * 3 + c];
  const half x = half(lut[c * 256 + v]);
  device half* row = X + size_t(r) * PP3 * p.T;
  for (uint t = 0; t < p.T; ++t) row[t * PP3 + k] = x;
}

// x[r] += bias + the learned position embedding bilinearly interpolated at patch r's (row, column) of the grid
// (align corners: grid row i samples table row i * (side - 1) / (rows - 1)). rowinfo / colinfo: per grid row / column
// (floor, ceil, 1 - frac, frac) as floats.
struct QvitEmbedParams {
  uint N, D, gw, side, merge;
};
[[kernel]] void qvit_embed(device float* x [[buffer(0)]], const device bfloat* bias [[buffer(1)]],
                           const device bfloat* pos [[buffer(2)]], const device float4* rowinfo [[buffer(3)]],
                           const device float4* colinfo [[buffer(4)]], constant QvitEmbedParams& p [[buffer(5)]],
                           uint2 gid [[thread_position_in_grid]]) {
  const uint i = gid.x, r = gid.y;
  if (i >= p.D || r >= p.N) return;
  const uint mm = p.merge * p.merge, b = r / mm, bw = p.gw / p.merge;
  const uint prow = (b / bw) * p.merge + (r % mm) / p.merge, pcol = (b % bw) * p.merge + r % p.merge;
  const float4 hr = rowinfo[prow], wc = colinfo[pcol];
  const uint h0 = uint(hr.x), h1 = uint(hr.y), w0 = uint(wc.x), w1 = uint(wc.y);
  // the reference's tap order: (h0, w0), (h0, w1), (h1, w0), (h1, w1)
  float e = float(pos[(h0 * p.side + w0) * p.D + i]) * (hr.z * wc.z);
  e += float(pos[(h0 * p.side + w1) * p.D + i]) * (hr.z * wc.w);
  e += float(pos[(h1 * p.side + w0) * p.D + i]) * (hr.w * wc.z);
  e += float(pos[(h1 * p.side + w1) * p.D + i]) * (hr.w * wc.w);
  device float* xr = x + size_t(r) * p.D;
  xr[i] = xr[i] + float(bias[i]) + e;
}

// qkv rows (fp32 [rows, 3 * heads * HD], from the GEMM) + bias -> 2D RoPE on q and k -> half Q, K, V [heads][N][HD].
// RoPE: dims d and d + HD/2 rotate together by angle pos * inv_freq[d % (HD/4)], pos = the patch's grid row for
// d < HD/4, its column after (the reference's cat(h_freqs, w_freqs), repeated over both halves). Thread = (row, head,
// pair d < HD/2).
struct QvitRopeParams {
  uint rows, r0, N, heads, HD, gw, merge;
};
[[kernel]] void qvit_qkv_rope(const device float* y [[buffer(0)]], const device bfloat* bias [[buffer(1)]],
                              const device float* inv_freq [[buffer(2)]], device half* Q [[buffer(3)]],
                              device half* K [[buffer(4)]], device half* V [[buffer(5)]],
                              constant QvitRopeParams& p [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  const uint half_d = p.HD / 2, quarter = p.HD / 4;
  const uint d = gid % half_d, h = (gid / half_d) % p.heads, rr = gid / (half_d * p.heads);
  if (rr >= p.rows) return;
  const uint r = p.r0 + rr, mm = p.merge * p.merge, b = r / mm, bw = p.gw / p.merge;
  const uint prow = (b / bw) * p.merge + (r % mm) / p.merge, pcol = (b % bw) * p.merge + r % p.merge;
  const float ang = float(d < quarter ? prow : pcol) * inv_freq[d % quarter];
  const float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
  const uint W = p.heads * p.HD;
  const device float* yr = y + size_t(rr) * 3 * W;
  const uint c0 = h * p.HD + d, c1 = c0 + half_d;
  const size_t o = (size_t(h) * p.N + r) * p.HD + d;
  for (uint m = 0; m < 2; ++m) {  // q, then k
    const float a = yr[m * W + c0] + float(bias[m * W + c0]), b2 = yr[m * W + c1] + float(bias[m * W + c1]);
    device half* dst = m == 0 ? Q : K;
    dst[o] = half(a * cs - b2 * sn);
    dst[o + half_d] = half(b2 * cs + a * sn);
  }
  V[o] = half(yr[2 * W + c0] + float(bias[2 * W + c0]));
  V[o + half_d] = half(yr[2 * W + c1] + float(bias[2 * W + c1]));
}
