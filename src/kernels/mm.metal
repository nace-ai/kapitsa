#include "common.h"

// ==== Images in the language model's passes (src/vision/embed.h) ====

// Prompt rows that take image embeddings: x[dst_row0 + r] (fp32, D) = src[src_row0 + r] (bf16, src_ld per row).
struct EmbedRowsParams {
  uint D, src_ld, dst_row0, src_row0;
};
[[kernel]] void embed_rows_bf16(const device bfloat* src [[buffer(0)]], device float* x [[buffer(1)]],
                                constant EmbedRowsParams& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint i = gid.x * 4, r = gid.y;
  if (i >= p.D) return;
  const device bfloat* s = src + size_t(p.src_row0 + r) * p.src_ld + i;
  device float* d = x + size_t(p.dst_row0 + r) * p.D + i;
  for (uint k = 0; k < 4 && i + k < p.D; ++k) d[k] = float(s[k]);
}

// attn_prep with Qwen's interleaved M-RoPE (attn.metal; same buffers 0-8): the rotary angle of frequency i takes the
// row's temporal, height or width position (interleaved: i % 3 == 1 below sec1 -> height, i % 3 == 2 below sec2 ->
// width, else temporal). Rows at cache positions below mn take (t, h, w) from mpos; later rows take position + delta
// for all three. A text row's three positions are equal, so it rotates exactly as attn_prep rotates that position.
struct MropeParams {
  uint mn;
  int delta;
  uint sec1, sec2;
};

[[kernel]] void attn_prep_m(const device float* qg [[buffer(0)]], const device float* kr [[buffer(1)]],
                            const device float* vr [[buffer(2)]], const device bfloat* qn_w [[buffer(3)]],
                            const device bfloat* kn_w [[buffer(4)]], device float* q_out [[buffer(5)]],
                            device half* kc [[buffer(6)]], device half* vc [[buffer(7)]],
                            constant AttnPrepParams& p [[buffer(8)]], const device int* mpos [[buffer(9)]],
                            constant MropeParams& mr [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],
                            uint2 tid2 [[thread_position_in_threadgroup]]) {
  const uint tid = tid2.x;
  threadgroup float scratch[32];
  threadgroup float row[512];
  // One entry per rotary pair; rot <= D <= 512 (row[]), so half_rot <= 256.
  threadgroup float cosine[256], sine[256];
  const uint h = tg.x, t = tg.y, D = p.D;
  const uint pos = p.pos + t;  // cache row
  if (h >= p.Hq + p.Hkv) {     // value head: plain copy into the cache
    uint hv = h - p.Hq - p.Hkv;
    vc[(hv * p.max_seq + pos) * D + tid] = half(vr[(t * p.Hkv + hv) * D + tid]);
    return;
  }
  const bool is_q = h < p.Hq;
  float x = is_q ? qg[(t * p.Hq + h) * 2 * D + tid] : kr[(t * p.Hkv + (h - p.Hq)) * D + tid];
  float ss = tg_sum(x * x, scratch, tid, D);
  x = x * metal::precise::rsqrt(ss / float(D) + p.eps) * float(is_q ? qn_w[tid] : kn_w[tid]);
  row[tid] = x;
  const uint half_rot = p.rot / 2;
  if (tid < half_rot) {
    uint i = tid;
    int rp;
    if (pos < mr.mn) {
      const uint a = (i % 3 == 1 && i < mr.sec1) ? 1u : (i % 3 == 2 && i < mr.sec2) ? 2u : 0u;
      rp = mpos[pos * 3 + a];
    } else {
      rp = int(pos) + mr.delta;
    }
    float inv_freq = metal::precise::pow(p.theta, -2.0f * float(i) / float(p.rot));
    float ang = float(rp) * inv_freq;
    // Both coordinates of a rotary pair use the same precise angle.
    cosine[tid] = metal::precise::cos(ang);
    sine[tid] = metal::precise::sin(ang);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < p.rot) {
    float cs = cosine[tid % half_rot], sn = sine[tid % half_rot];
    if (tid < half_rot) x = row[tid] * cs - row[tid + half_rot] * sn;
    else x = row[tid] * cs + row[tid - half_rot] * sn;
  }
  if (is_q) q_out[(t * p.Hq + h) * D + tid] = x;
  else kc[((h - p.Hq) * p.max_seq + pos) * D + tid] = half(x);
}
