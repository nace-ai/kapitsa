#include "common.h"

// ---- Gated full attention (Qwen3.5/3.6/3.8): q/k RMSNorm, partial RoPE, KV cache write ----
// q_proj output per head is [query (D) | gate (D)]. KV cache: half, [Hkv, max_seq, D].
struct AttnPrepParams {
  uint T;
  uint Hq;
  uint Hkv;
  uint D;        // head dim (256)
  uint rot;      // rotary dims (64)
  uint pos;      // position of the first token
  uint max_seq;
  float eps;
  float theta;
};

// grid: x = Hq + 2*Hkv (q heads, k heads, v heads), y = T; threadgroup = D threads
[[kernel]] void attn_prep(const device float* qg [[buffer(0)]], const device float* kr [[buffer(1)]],
                          const device float* vr [[buffer(2)]], const device bfloat* qn_w [[buffer(3)]],
                          const device bfloat* kn_w [[buffer(4)]], device float* q_out [[buffer(5)]],
                          device half* kc [[buffer(6)]], device half* vc [[buffer(7)]],
                          constant AttnPrepParams& p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint2 tid2 [[thread_position_in_threadgroup]]) {
  const uint tid = tid2.x;
  threadgroup float scratch[32];
  threadgroup float row[512];
  const uint h = tg.x, t = tg.y, D = p.D;
  const uint pos = p.pos + t;
  if (h >= p.Hq + p.Hkv) {  // value head: plain copy into the cache
    uint hv = h - p.Hq - p.Hkv;
    vc[(hv * p.max_seq + pos) * D + tid] = half(vr[(t * p.Hkv + hv) * D + tid]);
    return;
  }
  const bool is_q = h < p.Hq;
  float x = is_q ? qg[(t * p.Hq + h) * 2 * D + tid] : kr[(t * p.Hkv + (h - p.Hq)) * D + tid];
  float ss = tg_sum(x * x, scratch, tid, D);
  x = x * metal::precise::rsqrt(ss / float(D) + p.eps) * float(is_q ? qn_w[tid] : kn_w[tid]);
  row[tid] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint half_rot = p.rot / 2;
  if (tid < p.rot) {
    uint i = tid % half_rot;
    float inv_freq = metal::precise::pow(p.theta, -2.0f * float(i) / float(p.rot));
    float ang = float(pos) * inv_freq;
    float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
    if (tid < half_rot) x = row[tid] * cs - row[tid + half_rot] * sn;
    else x = row[tid] * cs + row[tid - half_rot] * sn;
  }
  if (is_q) q_out[(t * p.Hq + h) * D + tid] = x;
  else kc[((h - p.Hq) * p.max_seq + pos) * D + tid] = half(x);
}

// ---- KV cache growth (Qwen35::reserve): positions [0, old_cap) of every kv head of a [H, old_cap, D]
// cache into a [H, new_cap, D] one, in 16-byte words. p = (old_cap, new_cap) in words per head, H.
// grid: x = ceil(p.x / 256) threadgroups of 256 threads, y = H.
[[kernel]] void kv_regrow(const device uint4* src [[buffer(0)]], device uint4* dst [[buffer(1)]],
                          constant uint3& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.x) return;
  dst[gid.y * p.y + gid.x] = src[gid.y * p.x + gid.x];
}

// ---- KV prefix copy between two conversations' caches (Model::copy_prefix): positions [0, n) of every kv head of a
// [H, src_cap, D] cache into a [H, dst_cap, D] one, in 16-byte words. p = (n, src_cap, dst_cap) in words per head.
// grid: x = ceil(p.x / 256) threadgroups of 256 threads, y = H.
[[kernel]] void kv_copy_prefix(const device uint4* src [[buffer(0)]], device uint4* dst [[buffer(1)]],
                               constant uint3& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.x) return;
  dst[gid.y * p.z + gid.x] = src[gid.y * p.y + gid.x];
}

// ---- Decode attention, split-K over the cache ("flash decoding") ----
// One threadgroup per (chunk, kv head); its G = Hq/Hkv simdgroups are the query heads sharing that
// kv head, so K/V tiles are read from memory once and reused from threadgroup memory.
struct AttnDecParams {
  uint Hq;
  uint Hkv;
  uint L;       // number of valid cache positions (including the current token)
  uint chunk;   // positions per threadgroup
  uint max_seq;
  float scale;
  uint nchunks;
};

constant constexpr uint kTile = 16;
constant constexpr uint kHD = 256;

[[kernel]] void attn_decode_partial(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                    const device half* vc [[buffer(2)]], device float* part_o [[buffer(3)]],
                                    device float* part_ml [[buffer(4)]], constant AttnDecParams& p [[buffer(5)]],
                                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                    uint2 ntg2 [[threads_per_threadgroup]]) {
  const uint ntg = ntg2.x;
  threadgroup half ktile[kTile * kHD];
  threadgroup half vtile[kTile * kHD];
  const uint chunk = tg.x, kvh = tg.y;
  const uint G = p.Hq / p.Hkv;
  const uint hq = kvh * G + sg;
  const uint start = chunk * p.chunk;
  const uint end = min(start + p.chunk, p.L);

  float qv[8], acc[8];
  for (int i = 0; i < 8; ++i) {
    qv[i] = q[hq * kHD + lane * 8 + i] * p.scale;
    acc[i] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  const device half* kbase = kc + kvh * p.max_seq * kHD;
  const device half* vbase = vc + kvh * p.max_seq * kHD;

  for (uint t0 = start; t0 < end; t0 += kTile) {
    uint n = min(kTile, end - t0);
    // cooperative load: n*256 halfs each for K and V, as half4
    for (uint i = tid; i < n * kHD / 4; i += ntg) {
      ((threadgroup half4*)ktile)[i] = ((const device half4*)(kbase + t0 * kHD))[i];
      ((threadgroup half4*)vtile)[i] = ((const device half4*)(vbase + t0 * kHD))[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < G) {
      for (uint j = 0; j < n; ++j) {
        const threadgroup half* kr = ktile + j * kHD + lane * 8;
        float s = 0.0f;
        for (int i = 0; i < 8; ++i) s += qv[i] * float(kr[i]);
        s = simd_sum(s);
        float mn = max(m, s);
        float corr = metal::fast::exp(m - mn);
        float e = metal::fast::exp(s - mn);
        l = l * corr + e;
        const threadgroup half* vr = vtile + j * kHD + lane * 8;
        for (int i = 0; i < 8; ++i) acc[i] = acc[i] * corr + e * float(vr[i]);
        m = mn;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (sg < G) {
    uint base = (hq * p.nchunks + chunk);
    for (int i = 0; i < 8; ++i) part_o[base * kHD + lane * 8 + i] = acc[i];
    if (lane == 0) {
      part_ml[base * 2 + 0] = m;
      part_ml[base * 2 + 1] = l;
    }
  }
}

// grid: Hq threadgroups x 256 threads. out[h*D + d] = softmax-combined o * sigmoid(gate)
[[kernel]] void attn_decode_reduce(const device float* part_o [[buffer(0)]], const device float* part_ml [[buffer(1)]],
                                   const device float* qg [[buffer(2)]], device float* out [[buffer(3)]],
                                   constant AttnDecParams& p [[buffer(4)]], uint h [[threadgroup_position_in_grid]],
                                   uint d [[thread_position_in_threadgroup]]) {
  float M = -INFINITY;
  for (uint c = 0; c < p.nchunks; ++c) M = max(M, part_ml[(h * p.nchunks + c) * 2]);
  float num = 0.0f, den = 0.0f;
  for (uint c = 0; c < p.nchunks; ++c) {
    float mc = part_ml[(h * p.nchunks + c) * 2];
    if (mc == -INFINITY) continue;
    float w = metal::fast::exp(mc - M);
    num += w * part_o[(h * p.nchunks + c) * kHD + d];
    den += w * part_ml[(h * p.nchunks + c) * 2 + 1];
  }
  float gate = qg[h * 2 * kHD + kHD + d];
  out[h * kHD + d] = (num / den) * sigmoid(gate);
}

// ---- Prefill attention: causal flash attention with simdgroup matrices ----
// One threadgroup per (block of 4 queries, kv head); its 4 simdgroups each own one query and the
// G=8 query heads that share the kv head (8 rows of an 8x8 fragment), so K/V tiles are loaded once
// for 32 rows and every row in a simdgroup has the same causal limit.
// q: [T, Hq, D] fp32 (normed + roped); kc/vc: [Hkv, max_seq, D] half; qg: [T, Hq, 2D] (gate half).
// out: [T, Hq*D] = softmax(q k^T * scale) v * sigmoid(gate).
struct AttnPrefParams {
  uint T;
  uint Hq;
  uint Hkv;
  uint pos0;     // absolute position of query 0
  uint max_seq;
  float scale;
};

constant constexpr uint kQB = 4;    // queries per threadgroup
constant constexpr uint kKB = 16;   // keys per tile
constant constexpr uint kQLD = kHD + 8;

[[kernel]] void attn_prefill(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                             const device half* vc [[buffer(2)]], const device float* qg [[buffer(3)]],
                             device float* out [[buffer(4)]], constant AttnPrefParams& p [[buffer(5)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half qs[kQB * 8 * kQLD];   // 32 rows x 264
  threadgroup half kv[kKB * kQLD];       // 16 keys x 264 (K, then V)
  threadgroup float sst[kQB][8 * kKB];   // per-simdgroup S staging (and output staging)
  threadgroup half pst[kQB][8 * kKB];    // per-simdgroup P (probabilities) as half
  threadgroup float diag[kQB][64];
  const uint G = p.Hq / p.Hkv;  // must be 8
  const uint kvh = tg.y;
  const uint t0 = tg.x * kQB;
  const uint qi = t0 + sg;                    // this simdgroup's query index
  const bool qvalid = qi < p.T;
  const uint qpos = p.pos0 + min(qi, p.T - 1);
  const uint last_q = min(t0 + kQB, p.T) - 1;
  const uint kend = p.pos0 + last_q + 1;      // keys needed by the block

  // load Q rows: row r = s*8 + g  -> q[t0+s, kvh*G+g, :]
  for (uint i = tid; i < kQB * 8 * kHD / 4; i += 128) {
    uint r = i / (kHD / 4), c4 = i % (kHD / 4);
    uint s = r / 8, g = r % 8;
    uint t = min(t0 + s, p.T - 1);
    float4 v = *(const device float4*)(q + (t * p.Hq + kvh * G + g) * kHD + c4 * 4);
    *(threadgroup half4*)(qs + r * kQLD + c4 * 4) = half4(v * p.scale);
  }
  for (uint i = lane; i < 64; i += 32) diag[sg][i] = 0.0f;

  simdgroup_float8x8 o[kHD / 8];
  for (uint j = 0; j < kHD / 8; ++j) o[j] = simdgroup_float8x8(0);
  // row stats: lane handles row (lane / 4), cols (lane % 4)*4 .. +3 of the 8x16 S tile
  const uint my_row = lane / 4, my_c0 = (lane % 4) * 4;
  float m_row = -INFINITY, l_row = 0.0f;

  const device half* kb = kc + kvh * p.max_seq * kHD;
  const device half* vb = vc + kvh * p.max_seq * kHD;
  for (uint k0 = 0; k0 < kend; k0 += kKB) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < kKB * kHD / 4; i += 128) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      half4 v = k0 + r < kend ? *(const device half4*)(kb + (k0 + r) * kHD + c4 * 4) : half4(0);
      *(threadgroup half4*)(kv + r * kQLD + c4 * 4) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // S = Q K^T  (8 x 16)
    simdgroup_float8x8 s0 = simdgroup_float8x8(0), s1 = simdgroup_float8x8(0);
    for (uint d = 0; d < kHD; d += 8) {
      simdgroup_half8x8 a, b0, b1;
      simdgroup_load(a, qs + sg * 8 * kQLD + d, kQLD);
      simdgroup_load(b0, kv + 0 * kQLD + d, kQLD, ulong2(0, 0), true);
      simdgroup_load(b1, kv + 8 * kQLD + d, kQLD, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(s0, a, b0, s0);
      simdgroup_multiply_accumulate(s1, a, b1, s1);
    }
    simdgroup_store(s0, sst[sg], kKB);
    simdgroup_store(s1, sst[sg] + 8, kKB);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // online softmax on this lane's 4 values of row my_row
    float sv[4];
    float mx = -INFINITY;
    for (uint c = 0; c < 4; ++c) {
      uint key = k0 + my_c0 + c;
      sv[c] = (key <= qpos && key < kend) ? sst[sg][my_row * kKB + my_c0 + c] : -INFINITY;
      mx = max(mx, sv[c]);
    }
    mx = max(mx, simd_shuffle_xor(mx, 1));
    mx = max(mx, simd_shuffle_xor(mx, 2));
    float m_new = max(m_row, mx);
    float corr = m_new == -INFINITY ? 1.0f : metal::fast::exp(m_row - m_new);
    float ps = 0.0f;
    for (uint c = 0; c < 4; ++c) {
      float e = sv[c] == -INFINITY ? 0.0f : metal::fast::exp(sv[c] - m_new);
      ps += e;
      pst[sg][my_row * kKB + my_c0 + c] = half(e);
    }
    ps += simd_shuffle_xor(ps, 1);
    ps += simd_shuffle_xor(ps, 2);
    l_row = l_row * corr + ps;
    m_row = m_new;
    if (lane % 4 == 0) diag[sg][my_row * 9] = corr;
    // load V into the same tile buffer
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < kKB * kHD / 4; i += 128) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      half4 v = k0 + r < kend ? *(const device half4*)(vb + (k0 + r) * kHD + c4 * 4) : half4(0);
      *(threadgroup half4*)(kv + r * kQLD + c4 * 4) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // O = diag(corr) O + P V
    simdgroup_float8x8 dg;
    simdgroup_half8x8 p0, p1;
    simdgroup_load(dg, diag[sg], 8);
    simdgroup_load(p0, pst[sg], kKB);
    simdgroup_load(p1, pst[sg] + 8, kKB);
    for (uint j = 0; j < kHD / 8; ++j) {
      simdgroup_half8x8 v0, v1;
      simdgroup_load(v0, kv + 0 * kQLD + j * 8, kQLD);
      simdgroup_load(v1, kv + 8 * kQLD + j * 8, kQLD);
      simdgroup_multiply(o[j], dg, o[j]);
      simdgroup_multiply_accumulate(o[j], p0, v0, o[j]);
      simdgroup_multiply_accumulate(o[j], p1, v1, o[j]);
    }
  }
  // finalize: o / l, times sigmoid(gate)
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // row sums to diag (1/l)
  if (lane % 4 == 0) diag[sg][my_row * 9] = l_row > 0.0f ? 1.0f / l_row : 0.0f;
  simdgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 dg;
  simdgroup_load(dg, diag[sg], 8);
  for (uint j = 0; j < kHD / 8; ++j) {
    simdgroup_multiply(o[j], dg, o[j]);
    simdgroup_store(o[j], sst[sg], 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (qvalid) {
      for (uint e = lane; e < 64; e += 32) {
        uint g = e / 8, d = j * 8 + e % 8;
        uint h = kvh * G + g;
        float gate = qg[(qi * p.Hq + h) * 2 * kHD + kHD + d];
        out[qi * p.Hq * kHD + h * kHD + d] = sst[sg][e] * sigmoid(gate);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ---- Prefill attention v2 ----
// Threadgroup = 8 simdgroups = 8 consecutive queries x the 8 query heads of one kv head (64 rows);
// Q fragments live in registers, K and V tiles of 32 keys go through one threadgroup buffer.
constant constexpr uint kQB2 = 8;
constant constexpr uint kKB2 = 32;
constant constexpr uint kLD2 = kHD + 8;

[[kernel]] void attn_prefill2(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                              const device half* vc [[buffer(2)]], const device float* qg [[buffer(3)]],
                              device float* out [[buffer(4)]], constant AttnPrefParams& p [[buffer(5)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half kv[kKB2 * kLD2];            // 32 keys x 264 (K, then V); also stages Q at start
  threadgroup float sst[kQB2][8 * kKB2];       // S per simdgroup (8 x 32)
  threadgroup half pst[kQB2][8 * kKB2];        // P per simdgroup
  threadgroup float diag[kQB2][64];
  const uint G = p.Hq / p.Hkv;
  const uint kvh = tg.y;
  const uint t0 = tg.x * kQB2;
  const uint qi = t0 + sg;
  const bool qvalid = qi < p.T;
  const uint qpos = p.pos0 + min(qi, p.T - 1);
  const uint last_q = min(t0 + kQB2, p.T) - 1;
  const uint kend = p.pos0 + last_q + 1;

  // Q -> registers, staged through `kv` 4 queries (32 rows) at a time
  simdgroup_half8x8 qf[kHD / 8];
  for (uint half_ = 0; half_ < 2; ++half_) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 32 * kHD / 4; i += 256) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      uint s = half_ * 4 + r / 8, g = r % 8;
      uint t = min(t0 + s, p.T - 1);
      float4 v = *(const device float4*)(q + (t * p.Hq + kvh * G + g) * kHD + c4 * 4);
      *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = half4(v * p.scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg / 4 == half_)
      for (uint d = 0; d < kHD / 8; ++d) simdgroup_load(qf[d], kv + (sg % 4) * 8 * kLD2 + d * 8, kLD2);
  }
  for (uint i = lane; i < 64; i += 32) diag[sg][i] = 0.0f;

  simdgroup_float8x8 o[kHD / 8];
  for (uint j = 0; j < kHD / 8; ++j) o[j] = simdgroup_float8x8(0);
  // lane owns row lane/4 and columns (lane%4)*8 .. +7 of the 8x32 tile
  const uint my_row = lane / 4, my_c0 = (lane % 4) * 8;
  float m_row = -INFINITY, l_row = 0.0f;
  const device half* kb = kc + kvh * p.max_seq * kHD;
  const device half* vb = vc + kvh * p.max_seq * kHD;

  for (uint k0 = 0; k0 < kend; k0 += kKB2) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < kKB2 * kHD / 4; i += 256) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      half4 v = k0 + r < kend ? *(const device half4*)(kb + (k0 + r) * kHD + c4 * 4) : half4(0);
      *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool active = k0 <= qpos;  // whole tile beyond this query: skip compute (still sync)
    if (active) {
      simdgroup_float8x8 s[4] = {simdgroup_float8x8(0), simdgroup_float8x8(0), simdgroup_float8x8(0),
                                 simdgroup_float8x8(0)};
      for (uint d = 0; d < kHD / 8; ++d) {
        for (uint c = 0; c < 4; ++c) {
          simdgroup_half8x8 b;
          simdgroup_load(b, kv + c * 8 * kLD2 + d * 8, kLD2, ulong2(0, 0), true);
          simdgroup_multiply_accumulate(s[c], qf[d], b, s[c]);
        }
      }
      for (uint c = 0; c < 4; ++c) simdgroup_store(s[c], sst[sg] + c * 8, kKB2);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      float sv[8];
      float mx = -INFINITY;
      for (uint c = 0; c < 8; ++c) {
        uint key = k0 + my_c0 + c;
        sv[c] = key <= qpos ? sst[sg][my_row * kKB2 + my_c0 + c] : -INFINITY;
        mx = max(mx, sv[c]);
      }
      mx = max(mx, simd_shuffle_xor(mx, 1));
      mx = max(mx, simd_shuffle_xor(mx, 2));
      float m_new = max(m_row, mx);
      float corr = metal::fast::exp(m_row - m_new);
      float ps = 0.0f;
      for (uint c = 0; c < 8; ++c) {
        float e = sv[c] == -INFINITY ? 0.0f : metal::fast::exp(sv[c] - m_new);
        ps += e;
        pst[sg][my_row * kKB2 + my_c0 + c] = half(e);
      }
      ps += simd_shuffle_xor(ps, 1);
      ps += simd_shuffle_xor(ps, 2);
      l_row = l_row * corr + ps;
      m_row = m_new;
      if (lane % 4 == 0) diag[sg][my_row * 9] = corr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < kKB2 * kHD / 4; i += 256) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      half4 v = k0 + r < kend ? *(const device half4*)(vb + (k0 + r) * kHD + c4 * 4) : half4(0);
      *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      simdgroup_float8x8 dg;
      simdgroup_load(dg, diag[sg], 8);
      simdgroup_half8x8 pf[4];
      for (uint c = 0; c < 4; ++c) simdgroup_load(pf[c], pst[sg] + c * 8, kKB2);
      for (uint j = 0; j < kHD / 8; ++j) {
        simdgroup_multiply(o[j], dg, o[j]);
        for (uint c = 0; c < 4; ++c) {
          simdgroup_half8x8 vf;
          simdgroup_load(vf, kv + c * 8 * kLD2 + j * 8, kLD2);
          simdgroup_multiply_accumulate(o[j], pf[c], vf, o[j]);
        }
      }
    }
  }
  if (lane % 4 == 0) diag[sg][my_row * 9] = l_row > 0.0f ? 1.0f / l_row : 0.0f;
  simdgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 dg;
  simdgroup_load(dg, diag[sg], 8);
  for (uint j = 0; j < kHD / 8; ++j) {
    simdgroup_multiply(o[j], dg, o[j]);
    simdgroup_store(o[j], sst[sg], 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (qvalid) {
      for (uint e = lane; e < 64; e += 32) {
        uint g = e / 8, d = j * 8 + e % 8;
        uint h = kvh * G + g;
        float gate = qg[(qi * p.Hq + h) * 2 * kHD + kHD + d];
        out[qi * p.Hq * kHD + h * kHD + d] = sst[sg][e] * sigmoid(gate);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ---- Prefill attention v3: softmax in registers ----
// Same tiling as v2 (8 queries x 8 heads per threadgroup, Q in registers, 32-key tiles), but the
// online softmax works on the fragments' own elements. Lane layout of an 8x8 simdgroup matrix:
// row = (q & 4) + (lane / 2) % 4, cols = (q & 2) * 2 + (lane % 2) * 2 + {0, 1}, q = lane / 4;
// the 4 lanes of a row are lane ^ 1 and lane ^ 8 (and both).
[[kernel]] void attn_prefill3(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                              const device half* vc [[buffer(2)]], const device float* qg [[buffer(3)]],
                              device float* out [[buffer(4)]], constant AttnPrefParams& p [[buffer(5)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half kv[kKB2 * kLD2];
  const uint G = p.Hq / p.Hkv;
  const uint kvh = tg.y;
  const uint t0 = tg.x * kQB2;
  const uint qi = t0 + sg;
  const bool qvalid = qi < p.T;
  const uint qpos = p.pos0 + min(qi, p.T - 1);
  const uint last_q = min(t0 + kQB2, p.T) - 1;
  const uint kend = p.pos0 + last_q + 1;
  const uint qd = lane / 4;
  const uint frow = (qd & 4) + (lane / 2) % 4;      // head within the kv group
  const uint fcol = (qd & 2) * 2 + (lane % 2) * 2;  // first of the two columns this lane owns

  simdgroup_half8x8 qf[kHD / 8];
  for (uint hh = 0; hh < 2; ++hh) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 32 * kHD / 4; i += 256) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      uint s = hh * 4 + r / 8, g = r % 8;
      uint t = min(t0 + s, p.T - 1);
      float4 v = *(const device float4*)(q + (t * p.Hq + kvh * G + g) * kHD + c4 * 4);
      *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = half4(v * p.scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg / 4 == hh)
      for (uint d = 0; d < kHD / 8; ++d) simdgroup_load(qf[d], kv + (sg % 4) * 8 * kLD2 + d * 8, kLD2);
  }

  simdgroup_float8x8 o[kHD / 8];
  for (uint j = 0; j < kHD / 8; ++j) o[j] = simdgroup_float8x8(0);
  float m_row = -INFINITY, l_row = 0.0f;
  const device half* kb = kc + kvh * p.max_seq * kHD;
  const device half* vb = vc + kvh * p.max_seq * kHD;

  for (uint k0 = 0; k0 < kend; k0 += kKB2) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < kKB2 * kHD / 4; i += 256) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      half4 v = k0 + r < kend ? *(const device half4*)(kb + (k0 + r) * kHD + c4 * 4) : half4(0);
      *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool active = k0 <= qpos;
    simdgroup_half8x8 pf[4];
    if (active) {
      simdgroup_float8x8 s[4] = {simdgroup_float8x8(0), simdgroup_float8x8(0), simdgroup_float8x8(0),
                                 simdgroup_float8x8(0)};
      for (uint d = 0; d < kHD / 8; ++d)
        for (uint c = 0; c < 4; ++c) {
          simdgroup_half8x8 b;
          simdgroup_load(b, kv + c * 8 * kLD2 + d * 8, kLD2, ulong2(0, 0), true);
          simdgroup_multiply_accumulate(s[c], qf[d], b, s[c]);
        }
      float sv[8];
      float mx = -INFINITY;
      for (uint c = 0; c < 4; ++c) {
        float2 e = float2(s[c].thread_elements()[0], s[c].thread_elements()[1]);
        uint key = k0 + c * 8 + fcol;
        sv[2 * c] = key <= qpos ? e.x : -INFINITY;
        sv[2 * c + 1] = key + 1 <= qpos ? e.y : -INFINITY;
        mx = max(mx, max(sv[2 * c], sv[2 * c + 1]));
      }
      mx = max(mx, simd_shuffle_xor(mx, 1));
      mx = max(mx, simd_shuffle_xor(mx, 8));
      const float m_new = max(m_row, mx);
      const float corr = metal::fast::exp(m_row - m_new);
      float ps = 0.0f;
      for (uint c = 0; c < 4; ++c) {
        float a = metal::fast::exp(sv[2 * c] - m_new), b = metal::fast::exp(sv[2 * c + 1] - m_new);
        ps += a + b;
        pf[c].thread_elements()[0] = half(a);
        pf[c].thread_elements()[1] = half(b);
      }
      ps += simd_shuffle_xor(ps, 1);
      ps += simd_shuffle_xor(ps, 8);
      l_row = l_row * corr + ps;
      m_row = m_new;
      for (uint j = 0; j < kHD / 8; ++j) {
        o[j].thread_elements()[0] *= corr;
        o[j].thread_elements()[1] *= corr;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < kKB2 * kHD / 4; i += 256) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      half4 v = k0 + r < kend ? *(const device half4*)(vb + (k0 + r) * kHD + c4 * 4) : half4(0);
      *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      for (uint j = 0; j < kHD / 8; ++j)
        for (uint c = 0; c < 4; ++c) {
          simdgroup_half8x8 vf;
          simdgroup_load(vf, kv + c * 8 * kLD2 + j * 8, kLD2);
          simdgroup_multiply_accumulate(o[j], pf[c], vf, o[j]);
        }
    }
  }
  if (qvalid) {
    const float inv = l_row > 0.0f ? 1.0f / l_row : 0.0f;
    const uint h = kvh * G + frow;
    const device float* gate = qg + (qi * p.Hq + h) * 2 * kHD + kHD;
    device float* dst = out + qi * p.Hq * kHD + h * kHD;
    for (uint j = 0; j < kHD / 8; ++j) {
      uint d = j * 8 + fcol;
      dst[d] = o[j].thread_elements()[0] * inv * sigmoid(gate[d]);
      dst[d + 1] = o[j].thread_elements()[1] * inv * sigmoid(gate[d + 1]);
    }
  }
}

[[kernel]] void attn_prefill4(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                              const device half* vc [[buffer(2)]], const device float* qg [[buffer(3)]],
                              device float* out [[buffer(4)]], constant AttnPrefParams& p [[buffer(5)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half kv[kKB2 * kLD2];
  threadgroup float diag[kQB2][64];
  const uint G = p.Hq / p.Hkv;
  const uint kvh = tg.y;
  const uint t0 = tg.x * kQB2;
  const uint qi = t0 + sg;
  const bool qvalid = qi < p.T;
  const uint qpos = p.pos0 + min(qi, p.T - 1);
  const uint last_q = min(t0 + kQB2, p.T) - 1;
  const uint kend = p.pos0 + last_q + 1;
  const uint qd = lane / 4;
  const uint frow = (qd & 4) + (lane / 2) % 4;      // head within the kv group
  const uint fcol = (qd & 2) * 2 + (lane % 2) * 2;  // first of the two columns this lane owns

  simdgroup_half8x8 qf[kHD / 8];
  for (uint hh = 0; hh < 2; ++hh) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 32 * kHD / 4; i += 256) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      uint s = hh * 4 + r / 8, g = r % 8;
      uint t = min(t0 + s, p.T - 1);
      float4 v = *(const device float4*)(q + (t * p.Hq + kvh * G + g) * kHD + c4 * 4);
      *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = half4(v * p.scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg / 4 == hh)
      for (uint d = 0; d < kHD / 8; ++d) simdgroup_load(qf[d], kv + (sg % 4) * 8 * kLD2 + d * 8, kLD2);
  }

  for (uint i = lane; i < 64; i += 32) diag[sg][i] = 0.0f;
  simdgroup_float8x8 o[kHD / 8];
  for (uint j = 0; j < kHD / 8; ++j) o[j] = simdgroup_float8x8(0);
  float m_row = -INFINITY, l_row = 0.0f;
  const device half* kb = kc + kvh * p.max_seq * kHD;
  const device half* vb = vc + kvh * p.max_seq * kHD;

  for (uint k0 = 0; k0 < kend; k0 += kKB2) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < kKB2 * kHD / 4; i += 256) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      half4 v = k0 + r < kend ? *(const device half4*)(kb + (k0 + r) * kHD + c4 * 4) : half4(0);
      *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool active = k0 <= qpos;
    simdgroup_half8x8 pf[4];
    if (active) {
      simdgroup_float8x8 s[4] = {simdgroup_float8x8(0), simdgroup_float8x8(0), simdgroup_float8x8(0),
                                 simdgroup_float8x8(0)};
      for (uint d = 0; d < kHD / 8; ++d)
        for (uint c = 0; c < 4; ++c) {
          simdgroup_half8x8 b;
          simdgroup_load(b, kv + c * 8 * kLD2 + d * 8, kLD2, ulong2(0, 0), true);
          simdgroup_multiply_accumulate(s[c], qf[d], b, s[c]);
        }
      float sv[8];
      float mx = -INFINITY;
      for (uint c = 0; c < 4; ++c) {
        float2 e = float2(s[c].thread_elements()[0], s[c].thread_elements()[1]);
        uint key = k0 + c * 8 + fcol;
        sv[2 * c] = key <= qpos ? e.x : -INFINITY;
        sv[2 * c + 1] = key + 1 <= qpos ? e.y : -INFINITY;
        mx = max(mx, max(sv[2 * c], sv[2 * c + 1]));
      }
      mx = max(mx, simd_shuffle_xor(mx, 1));
      mx = max(mx, simd_shuffle_xor(mx, 8));
      const float m_new = max(m_row, mx);
      const float corr = metal::fast::exp(m_row - m_new);
      float ps = 0.0f;
      for (uint c = 0; c < 4; ++c) {
        float a = metal::fast::exp(sv[2 * c] - m_new), b = metal::fast::exp(sv[2 * c + 1] - m_new);
        ps += a + b;
        pf[c].thread_elements()[0] = half(a);
        pf[c].thread_elements()[1] = half(b);
      }
      ps += simd_shuffle_xor(ps, 1);
      ps += simd_shuffle_xor(ps, 8);
      l_row = l_row * corr + ps;
      m_row = m_new;
      if ((lane & 9) == 0) diag[sg][frow * 9] = corr;  // one lane per row
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < kKB2 * kHD / 4; i += 256) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      half4 v = k0 + r < kend ? *(const device half4*)(vb + (k0 + r) * kHD + c4 * 4) : half4(0);
      *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      simdgroup_float8x8 dg;
      simdgroup_load(dg, diag[sg], 8);
      for (uint j = 0; j < kHD / 8; ++j) {
        simdgroup_multiply(o[j], dg, o[j]);
        for (uint c = 0; c < 4; ++c) {
          simdgroup_half8x8 vf;
          simdgroup_load(vf, kv + c * 8 * kLD2 + j * 8, kLD2);
          simdgroup_multiply_accumulate(o[j], pf[c], vf, o[j]);
        }
      }
    }
  }
  if (qvalid) {
    const float inv = l_row > 0.0f ? 1.0f / l_row : 0.0f;
    const uint h = kvh * G + frow;
    const device float* gate = qg + (qi * p.Hq + h) * 2 * kHD + kHD;
    device float* dst = out + qi * p.Hq * kHD + h * kHD;
    for (uint j = 0; j < kHD / 8; ++j) {
      uint d = j * 8 + fcol;
      dst[d] = o[j].thread_elements()[0] * inv * sigmoid(gate[d]);
      dst[d + 1] = o[j].thread_elements()[1] * inv * sigmoid(gate[d + 1]);
    }
  }
}

// ---- Prefill attention v5: head dim split across simdgroup pairs ----
// Threadgroup = 8 simdgroups = 4 queries (x 8 heads of one kv head = 32 rows) x 2 halves of the
// head dimension. Each simdgroup keeps Q and O for its 128 dims in registers, computes partial
// scores, and the pair sums them through threadgroup memory (the layout MLX uses for D=256).
constant constexpr uint kQB5 = 4;
constant constexpr uint kKB5 = 32;
constant constexpr uint kDH = kHD / 2;

[[kernel]] void attn_prefill5(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                              const device half* vc [[buffer(2)]], const device float* qg [[buffer(3)]],
                              device float* out [[buffer(4)]], constant AttnPrefParams& p [[buffer(5)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half kv[kKB5 * kLD2];          // K tile, then V tile; also stages Q
  threadgroup float xchg[8][32 * 8];          // partial scores per simdgroup (8 per lane)
  const uint G = p.Hq / p.Hkv;
  const uint kvh = tg.y;
  const uint rg = sg / 2, dh = sg % 2;        // row group (query), head-dim half
  const uint t0 = tg.x * kQB5;
  const uint qi = t0 + rg;
  const bool qvalid = qi < p.T;
  const uint qpos = p.pos0 + min(qi, p.T - 1);
  const uint last_q = min(t0 + kQB5, p.T) - 1;
  const uint kend = p.pos0 + last_q + 1;
  const uint qd = lane / 4;
  const uint frow = (qd & 4) + (lane / 2) % 4;
  const uint fcol = (qd & 2) * 2 + (lane % 2) * 2;

  // Q (4 queries x 8 heads = 32 rows) -> threadgroup -> registers (own half)
  for (uint i = tid; i < 32 * kHD / 4; i += 256) {
    uint r = i / (kHD / 4), c4 = i % (kHD / 4);
    uint s = r / 8, g = r % 8;
    uint t = min(t0 + s, p.T - 1);
    float4 v = *(const device float4*)(q + (t * p.Hq + kvh * G + g) * kHD + c4 * 4);
    *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = half4(v * p.scale);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_half8x8 qf[kDH / 8];
  for (uint d = 0; d < kDH / 8; ++d) simdgroup_load(qf[d], kv + rg * 8 * kLD2 + dh * kDH + d * 8, kLD2);

  simdgroup_float8x8 o[kDH / 8];
  for (uint j = 0; j < kDH / 8; ++j) o[j] = simdgroup_float8x8(0);
  float m_row = -INFINITY, l_row = 0.0f;
  const device half* kb = kc + kvh * p.max_seq * kHD;
  const device half* vb = vc + kvh * p.max_seq * kHD;

  for (uint k0 = 0; k0 < kend; k0 += kKB5) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < kKB5 * kHD / 4; i += 256) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      half4 v = k0 + r < kend ? *(const device half4*)(kb + (k0 + r) * kHD + c4 * 4) : half4(0);
      *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 s[4] = {simdgroup_float8x8(0), simdgroup_float8x8(0), simdgroup_float8x8(0),
                               simdgroup_float8x8(0)};
    for (uint d = 0; d < kDH / 8; ++d)
      for (uint c = 0; c < 4; ++c) {
        simdgroup_half8x8 b;
        simdgroup_load(b, kv + c * 8 * kLD2 + dh * kDH + d * 8, kLD2, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(s[c], qf[d], b, s[c]);
      }
    // exchange partial scores with the other half
    for (uint c = 0; c < 4; ++c) {
      xchg[sg][lane * 8 + 2 * c] = s[c].thread_elements()[0];
      xchg[sg][lane * 8 + 2 * c + 1] = s[c].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sv[8];
    float mx = -INFINITY;
    for (uint c = 0; c < 4; ++c) {
      uint key = k0 + c * 8 + fcol;
      float a = s[c].thread_elements()[0] + xchg[sg ^ 1][lane * 8 + 2 * c];
      float b = s[c].thread_elements()[1] + xchg[sg ^ 1][lane * 8 + 2 * c + 1];
      sv[2 * c] = key <= qpos ? a : -INFINITY;
      sv[2 * c + 1] = key + 1 <= qpos ? b : -INFINITY;
      mx = max(mx, max(sv[2 * c], sv[2 * c + 1]));
    }
    mx = max(mx, simd_shuffle_xor(mx, 1));
    mx = max(mx, simd_shuffle_xor(mx, 8));
    const float m_new = max(m_row, mx);
    const float corr = m_new == -INFINITY ? 1.0f : metal::fast::exp(m_row - m_new);
    float ps = 0.0f;
    simdgroup_half8x8 pf[4];
    for (uint c = 0; c < 4; ++c) {
      float a = sv[2 * c] == -INFINITY ? 0.0f : metal::fast::exp(sv[2 * c] - m_new);
      float b = sv[2 * c + 1] == -INFINITY ? 0.0f : metal::fast::exp(sv[2 * c + 1] - m_new);
      ps += a + b;
      pf[c].thread_elements()[0] = half(a);
      pf[c].thread_elements()[1] = half(b);
    }
    ps += simd_shuffle_xor(ps, 1);
    ps += simd_shuffle_xor(ps, 8);
    l_row = l_row * corr + ps;
    m_row = m_new;
    for (uint j = 0; j < kDH / 8; ++j) {
      o[j].thread_elements()[0] *= corr;
      o[j].thread_elements()[1] *= corr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < kKB5 * kHD / 4; i += 256) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      half4 v = k0 + r < kend ? *(const device half4*)(vb + (k0 + r) * kHD + c4 * 4) : half4(0);
      *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < kDH / 8; ++j)
      for (uint c = 0; c < 4; ++c) {
        simdgroup_half8x8 vf;
        simdgroup_load(vf, kv + c * 8 * kLD2 + dh * kDH + j * 8, kLD2);
        simdgroup_multiply_accumulate(o[j], pf[c], vf, o[j]);
      }
  }
  if (qvalid) {
    const float inv = l_row > 0.0f ? 1.0f / l_row : 0.0f;
    const uint h = kvh * G + frow;
    const device float* gate = qg + (qi * p.Hq + h) * 2 * kHD + kHD + dh * kDH;
    device float* dst = out + qi * p.Hq * kHD + h * kHD + dh * kDH;
    for (uint j = 0; j < kDH / 8; ++j) {
      uint d = j * 8 + fcol;
      dst[d] = o[j].thread_elements()[0] * inv * sigmoid(gate[d]);
      dst[d + 1] = o[j].thread_elements()[1] * inv * sigmoid(gate[d + 1]);
    }
  }
}

[[kernel]] void attn_prefill6(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                              const device half* vc [[buffer(2)]], const device float* qg [[buffer(3)]],
                              device float* out [[buffer(4)]], constant AttnPrefParams& p [[buffer(5)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half kv[kHD * (kKB5 + 8)];     // K^T tile [256][40], then V tile [32][264]; also stages Q
  threadgroup float xchg[8][32 * 8];          // partial scores per simdgroup (8 per lane)
  const uint G = p.Hq / p.Hkv;
  const uint kvh = tg.y;
  const uint rg = sg / 2, dh = sg % 2;        // row group (query), head-dim half
  const uint t0 = tg.x * kQB5;
  const uint qi = t0 + rg;
  const bool qvalid = qi < p.T;
  const uint qpos = p.pos0 + min(qi, p.T - 1);
  const uint last_q = min(t0 + kQB5, p.T) - 1;
  const uint kend = p.pos0 + last_q + 1;
  const uint qd = lane / 4;
  const uint frow = (qd & 4) + (lane / 2) % 4;
  const uint fcol = (qd & 2) * 2 + (lane % 2) * 2;

  // Q (4 queries x 8 heads = 32 rows) -> threadgroup -> registers (own half)
  for (uint i = tid; i < 32 * kHD / 4; i += 256) {
    uint r = i / (kHD / 4), c4 = i % (kHD / 4);
    uint s = r / 8, g = r % 8;
    uint t = min(t0 + s, p.T - 1);
    float4 v = *(const device float4*)(q + (t * p.Hq + kvh * G + g) * kHD + c4 * 4);
    *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = half4(v * p.scale);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_half8x8 qf[kDH / 8];
  for (uint d = 0; d < kDH / 8; ++d) simdgroup_load(qf[d], kv + rg * 8 * kLD2 + dh * kDH + d * 8, kLD2);

  simdgroup_float8x8 o[kDH / 8];
  for (uint j = 0; j < kDH / 8; ++j) o[j] = simdgroup_float8x8(0);
  float m_row = -INFINITY, l_row = 0.0f;
  const device half* kb = kc + kvh * p.max_seq * kHD;
  const device half* vb = vc + kvh * p.max_seq * kHD;

  for (uint k0 = 0; k0 < kend; k0 += kKB5) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < kKB5 * kHD / 4; i += 256) {
      uint r = i % kKB5, c4 = i / kKB5;  // consecutive threads take consecutive keys
      half4 v = k0 + r < kend ? *(const device half4*)(kb + (k0 + r) * kHD + c4 * 4) : half4(0);
      for (uint e = 0; e < 4; ++e) kv[(c4 * 4 + e) * (kKB5 + 8) + r] = v[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 s[4] = {simdgroup_float8x8(0), simdgroup_float8x8(0), simdgroup_float8x8(0),
                               simdgroup_float8x8(0)};
    for (uint d = 0; d < kDH / 8; ++d)
      for (uint c = 0; c < 4; ++c) {
        simdgroup_half8x8 b;
        simdgroup_load(b, kv + (dh * kDH + d * 8) * (kKB5 + 8) + c * 8, kKB5 + 8);
        simdgroup_multiply_accumulate(s[c], qf[d], b, s[c]);
      }
    // exchange partial scores with the other half
    for (uint c = 0; c < 4; ++c) {
      xchg[sg][lane * 8 + 2 * c] = s[c].thread_elements()[0];
      xchg[sg][lane * 8 + 2 * c + 1] = s[c].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sv[8];
    float mx = -INFINITY;
    for (uint c = 0; c < 4; ++c) {
      uint key = k0 + c * 8 + fcol;
      float a = s[c].thread_elements()[0] + xchg[sg ^ 1][lane * 8 + 2 * c];
      float b = s[c].thread_elements()[1] + xchg[sg ^ 1][lane * 8 + 2 * c + 1];
      sv[2 * c] = key <= qpos ? a : -INFINITY;
      sv[2 * c + 1] = key + 1 <= qpos ? b : -INFINITY;
      mx = max(mx, max(sv[2 * c], sv[2 * c + 1]));
    }
    mx = max(mx, simd_shuffle_xor(mx, 1));
    mx = max(mx, simd_shuffle_xor(mx, 8));
    const float m_new = max(m_row, mx);
    const float corr = m_new == -INFINITY ? 1.0f : metal::fast::exp(m_row - m_new);
    float ps = 0.0f;
    simdgroup_half8x8 pf[4];
    for (uint c = 0; c < 4; ++c) {
      float a = sv[2 * c] == -INFINITY ? 0.0f : metal::fast::exp(sv[2 * c] - m_new);
      float b = sv[2 * c + 1] == -INFINITY ? 0.0f : metal::fast::exp(sv[2 * c + 1] - m_new);
      ps += a + b;
      pf[c].thread_elements()[0] = half(a);
      pf[c].thread_elements()[1] = half(b);
    }
    ps += simd_shuffle_xor(ps, 1);
    ps += simd_shuffle_xor(ps, 8);
    l_row = l_row * corr + ps;
    m_row = m_new;
    for (uint j = 0; j < kDH / 8; ++j) {
      o[j].thread_elements()[0] *= corr;
      o[j].thread_elements()[1] *= corr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < kKB5 * kHD / 4; i += 256) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      half4 v = k0 + r < kend ? *(const device half4*)(vb + (k0 + r) * kHD + c4 * 4) : half4(0);
      *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < kDH / 8; ++j)
      for (uint c = 0; c < 4; ++c) {
        simdgroup_half8x8 vf;
        simdgroup_load(vf, kv + c * 8 * kLD2 + dh * kDH + j * 8, kLD2);
        simdgroup_multiply_accumulate(o[j], pf[c], vf, o[j]);
      }
  }
  if (qvalid) {
    const float inv = l_row > 0.0f ? 1.0f / l_row : 0.0f;
    const uint h = kvh * G + frow;
    const device float* gate = qg + (qi * p.Hq + h) * 2 * kHD + kHD + dh * kDH;
    device float* dst = out + qi * p.Hq * kHD + h * kHD + dh * kDH;
    for (uint j = 0; j < kDH / 8; ++j) {
      uint d = j * 8 + fcol;
      dst[d] = o[j].thread_elements()[0] * inv * sigmoid(gate[d]);
      dst[d + 1] = o[j].thread_elements()[1] * inv * sigmoid(gate[d + 1]);
    }
  }
}

[[kernel]] void attn_prefill7(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                              const device half* vc [[buffer(2)]], const device float* qg [[buffer(3)]],
                              device float* out [[buffer(4)]], constant AttnPrefParams& p [[buffer(5)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half kv[kKB5 * kLD2];          // K tile, then V tile; also stages Q
  threadgroup float xchg[8][32 * 8];          // partial scores per simdgroup (8 per lane)
  const uint G = p.Hq / p.Hkv;
  const uint kvh = tg.y;
  const uint rg = sg / 2, dh = sg % 2;        // row group (query), head-dim half
  const uint t0 = tg.x * kQB5;
  const uint qi = t0 + rg;
  const bool qvalid = qi < p.T;
  const uint qpos = p.pos0 + min(qi, p.T - 1);
  const uint last_q = min(t0 + kQB5, p.T) - 1;
  const uint kend = p.pos0 + last_q + 1;
  const uint qd = lane / 4;
  const uint frow = (qd & 4) + (lane / 2) % 4;
  const uint fcol = (qd & 2) * 2 + (lane % 2) * 2;

  // Q (4 queries x 8 heads = 32 rows) -> threadgroup -> registers (own half)
  for (uint i = tid; i < 32 * kHD / 4; i += 256) {
    uint r = i / (kHD / 4), c4 = i % (kHD / 4);
    uint s = r / 8, g = r % 8;
    uint t = min(t0 + s, p.T - 1);
    float4 v = *(const device float4*)(q + (t * p.Hq + kvh * G + g) * kHD + c4 * 4);
    *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = half4(v * p.scale);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_half8x8 qf[kDH / 8];
  for (uint d = 0; d < kDH / 8; ++d) simdgroup_load(qf[d], kv + rg * 8 * kLD2 + dh * kDH + d * 8, kLD2);

  simdgroup_float8x8 o[kDH / 8];
  for (uint j = 0; j < kDH / 8; ++j) o[j] = simdgroup_float8x8(0);
  float m_row = -INFINITY, l_row = 0.0f;
  const device half* kb = kc + kvh * p.max_seq * kHD;
  const device half* vb = vc + kvh * p.max_seq * kHD;

  for (uint k0 = 0; k0 < kend; k0 += kKB5) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < kKB5 * kHD / 4; i += 256) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      half4 v = k0 + r < kend ? *(const device half4*)(kb + (k0 + r) * kHD + c4 * 4) : half4(0);
      *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 s[4] = {simdgroup_float8x8(0), simdgroup_float8x8(0), simdgroup_float8x8(0),
                               simdgroup_float8x8(0)};
    for (uint d = 0; d < kDH / 8; ++d)
      for (uint c = 0; c < 4; ++c) {
        simdgroup_half8x8 b;
        simdgroup_load(b, kv + c * 8 * kLD2 + dh * kDH + d * 8, kLD2, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(s[c], qf[d], b, s[c]);
      }
    // exchange partial scores with the other half
    for (uint c = 0; c < 4; ++c) {
      xchg[sg][lane * 8 + 2 * c] = s[c].thread_elements()[0];
      xchg[sg][lane * 8 + 2 * c + 1] = s[c].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sfin[8];
    for (uint c = 0; c < 4; ++c) {
      sfin[2 * c] = s[c].thread_elements()[0] + xchg[sg ^ 1][lane * 8 + 2 * c];
      sfin[2 * c + 1] = s[c].thread_elements()[1] + xchg[sg ^ 1][lane * 8 + 2 * c + 1];
    }
    // Every simdgroup is past the exchange barrier, so K is dead: start loading V now and let the
    // softmax below overlap with it.
    for (uint i = tid; i < kKB5 * kHD / 4; i += 256) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      half4 v = k0 + r < kend ? *(const device half4*)(vb + (k0 + r) * kHD + c4 * 4) : half4(0);
      *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = v;
    }
    float sv[8];
    float mx = -INFINITY;
    for (uint c = 0; c < 4; ++c) {
      uint key = k0 + c * 8 + fcol;
      sv[2 * c] = key <= qpos ? sfin[2 * c] : -INFINITY;
      sv[2 * c + 1] = key + 1 <= qpos ? sfin[2 * c + 1] : -INFINITY;
      mx = max(mx, max(sv[2 * c], sv[2 * c + 1]));
    }
    mx = max(mx, simd_shuffle_xor(mx, 1));
    mx = max(mx, simd_shuffle_xor(mx, 8));
    const float m_new = max(m_row, mx);
    const float corr = m_new == -INFINITY ? 1.0f : metal::fast::exp(m_row - m_new);
    float ps = 0.0f;
    simdgroup_half8x8 pf[4];
    for (uint c = 0; c < 4; ++c) {
      float a = sv[2 * c] == -INFINITY ? 0.0f : metal::fast::exp(sv[2 * c] - m_new);
      float b = sv[2 * c + 1] == -INFINITY ? 0.0f : metal::fast::exp(sv[2 * c + 1] - m_new);
      ps += a + b;
      pf[c].thread_elements()[0] = half(a);
      pf[c].thread_elements()[1] = half(b);
    }
    ps += simd_shuffle_xor(ps, 1);
    ps += simd_shuffle_xor(ps, 8);
    l_row = l_row * corr + ps;
    m_row = m_new;
    for (uint j = 0; j < kDH / 8; ++j) {
      o[j].thread_elements()[0] *= corr;
      o[j].thread_elements()[1] *= corr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < kDH / 8; ++j)
      for (uint c = 0; c < 4; ++c) {
        simdgroup_half8x8 vf;
        simdgroup_load(vf, kv + c * 8 * kLD2 + dh * kDH + j * 8, kLD2);
        simdgroup_multiply_accumulate(o[j], pf[c], vf, o[j]);
      }
  }
  if (qvalid) {
    const float inv = l_row > 0.0f ? 1.0f / l_row : 0.0f;
    const uint h = kvh * G + frow;
    const device float* gate = qg + (qi * p.Hq + h) * 2 * kHD + kHD + dh * kDH;
    device float* dst = out + qi * p.Hq * kHD + h * kHD + dh * kDH;
    for (uint j = 0; j < kDH / 8; ++j) {
      uint d = j * 8 + fcol;
      dst[d] = o[j].thread_elements()[0] * inv * sigmoid(gate[d]);
      dst[d + 1] = o[j].thread_elements()[1] * inv * sigmoid(gate[d + 1]);
    }
  }
}


template <uint QB>
NI_INLINE void attn_prefill8_impl(const device float* q, const device half* kc,
                              const device half* vc, const device float* qg,
                              device float* out, constant AttnPrefParams& p,
                              uint2 tg, uint tid,
                              uint sg, uint lane, threadgroup half* kv, threadgroup float (*xchg)[32 * 8]) {
  constexpr uint NT = QB * 64;
  const uint G = p.Hq / p.Hkv;
  const uint kvh = tg.y;
  const uint rg = sg / 2, dh = sg % 2;        // row group (query), head-dim half
  const uint t0 = tg.x * QB;
  const uint qi = t0 + rg;
  const bool qvalid = qi < p.T;
  const uint qpos = p.pos0 + min(qi, p.T - 1);
  const uint last_q = min(t0 + QB, p.T) - 1;
  const uint kend = p.pos0 + last_q + 1;
  const uint qd = lane / 4;
  const uint frow = (qd & 4) + (lane / 2) % 4;
  const uint fcol = (qd & 2) * 2 + (lane % 2) * 2;

  // Q (4 queries x 8 heads = 32 rows) -> threadgroup -> registers (own half)
  simdgroup_half8x8 qf[kDH / 8];
  for (uint part = 0; part < QB / 4; ++part) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 32 * kHD / 4; i += NT) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      uint s = part * 4 + r / 8, g = r % 8;
      uint t = min(t0 + s, p.T - 1);
      float4 v = *(const device float4*)(q + (t * p.Hq + kvh * G + g) * kHD + c4 * 4);
      *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = half4(v * p.scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rg / 4 == part)
      for (uint d = 0; d < kDH / 8; ++d) simdgroup_load(qf[d], kv + (rg % 4) * 8 * kLD2 + dh * kDH + d * 8, kLD2);
  }

  simdgroup_float8x8 o[kDH / 8];
  for (uint j = 0; j < kDH / 8; ++j) o[j] = simdgroup_float8x8(0);
  float m_row = -INFINITY, l_row = 0.0f;
  const device half* kb = kc + kvh * p.max_seq * kHD;
  const device half* vb = vc + kvh * p.max_seq * kHD;

  for (uint k0 = 0; k0 < kend; k0 += kKB5) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < kKB5 * kHD / 8; i += NT) {
      uint r = i / (kHD / 8), c8 = i % (kHD / 8);
      uint4 v = k0 + r < kend ? *(const device uint4*)(kb + (k0 + r) * kHD + c8 * 8) : uint4(0);
      *(threadgroup uint4*)(kv + r * kLD2 + c8 * 8) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 s[4] = {simdgroup_float8x8(0), simdgroup_float8x8(0), simdgroup_float8x8(0),
                               simdgroup_float8x8(0)};
    for (uint d = 0; d < kDH / 8; ++d)
      for (uint c = 0; c < 4; ++c) {
        simdgroup_half8x8 b;
        simdgroup_load(b, kv + c * 8 * kLD2 + dh * kDH + d * 8, kLD2, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(s[c], qf[d], b, s[c]);
      }
    // exchange partial scores with the other half
    for (uint c = 0; c < 4; ++c) {
      xchg[sg][lane * 8 + 2 * c] = s[c].thread_elements()[0];
      xchg[sg][lane * 8 + 2 * c + 1] = s[c].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sfin[8];
    for (uint c = 0; c < 4; ++c) {
      sfin[2 * c] = s[c].thread_elements()[0] + xchg[sg ^ 1][lane * 8 + 2 * c];
      sfin[2 * c + 1] = s[c].thread_elements()[1] + xchg[sg ^ 1][lane * 8 + 2 * c + 1];
    }
    // Every simdgroup is past the exchange barrier, so K is dead: start loading V now and let the
    // softmax below overlap with it.
    for (uint i = tid; i < kKB5 * kHD / 8; i += NT) {
      uint r = i / (kHD / 8), c8 = i % (kHD / 8);
      uint4 v = k0 + r < kend ? *(const device uint4*)(vb + (k0 + r) * kHD + c8 * 8) : uint4(0);
      *(threadgroup uint4*)(kv + r * kLD2 + c8 * 8) = v;
    }
    float sv[8];
    float mx = -INFINITY;
    for (uint c = 0; c < 4; ++c) {
      uint key = k0 + c * 8 + fcol;
      sv[2 * c] = key <= qpos ? sfin[2 * c] : -INFINITY;
      sv[2 * c + 1] = key + 1 <= qpos ? sfin[2 * c + 1] : -INFINITY;
      mx = max(mx, max(sv[2 * c], sv[2 * c + 1]));
    }
    mx = max(mx, simd_shuffle_xor(mx, 1));
    mx = max(mx, simd_shuffle_xor(mx, 8));
    const float m_new = max(m_row, mx);
    const float corr = m_new == -INFINITY ? 1.0f : metal::fast::exp(m_row - m_new);
    float ps = 0.0f;
    simdgroup_half8x8 pf[4];
    for (uint c = 0; c < 4; ++c) {
      float a = sv[2 * c] == -INFINITY ? 0.0f : metal::fast::exp(sv[2 * c] - m_new);
      float b = sv[2 * c + 1] == -INFINITY ? 0.0f : metal::fast::exp(sv[2 * c + 1] - m_new);
      ps += a + b;
      pf[c].thread_elements()[0] = half(a);
      pf[c].thread_elements()[1] = half(b);
    }
    ps += simd_shuffle_xor(ps, 1);
    ps += simd_shuffle_xor(ps, 8);
    l_row = l_row * corr + ps;
    m_row = m_new;
    for (uint j = 0; j < kDH / 8; ++j) {
      o[j].thread_elements()[0] *= corr;
      o[j].thread_elements()[1] *= corr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < kDH / 8; ++j)
      for (uint c = 0; c < 4; ++c) {
        simdgroup_half8x8 vf;
        simdgroup_load(vf, kv + c * 8 * kLD2 + dh * kDH + j * 8, kLD2);
        simdgroup_multiply_accumulate(o[j], pf[c], vf, o[j]);
      }
  }
  if (qvalid) {
    const float inv = l_row > 0.0f ? 1.0f / l_row : 0.0f;
    const uint h = kvh * G + frow;
    const device float* gate = qg + (qi * p.Hq + h) * 2 * kHD + kHD + dh * kDH;
    device float* dst = out + qi * p.Hq * kHD + h * kHD + dh * kDH;
    for (uint j = 0; j < kDH / 8; ++j) {
      uint d = j * 8 + fcol;
      dst[d] = o[j].thread_elements()[0] * inv * sigmoid(gate[d]);
      dst[d + 1] = o[j].thread_elements()[1] * inv * sigmoid(gate[d + 1]);
    }
  }
}


#define NI_ATTN8(NAME, QB)                                                                                  \
  [[kernel]] void NAME(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],            \
                       const device half* vc [[buffer(2)]], const device float* qg [[buffer(3)]],           \
                       device float* out [[buffer(4)]], constant AttnPrefParams& p [[buffer(5)]],           \
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half kv[kKB5 * kLD2];                                                                       \
    threadgroup float xchg[QB * 2][32 * 8];                                                                 \
    attn_prefill8_impl<QB>(q, kc, vc, qg, out, p, tg, tid, sg, lane, kv, xchg);                             \
  }
NI_ATTN8(attn_prefill8_q4, 4)

template <uint QB, uint BK>
NI_INLINE void attn_prefill9_impl(const device float* q, const device half* kc,
                              const device half* vc, const device float* qg,
                              device float* out, constant AttnPrefParams& p,
                              uint2 tg, uint tid,
                              uint sg, uint lane, threadgroup half* kv, threadgroup float (*xchg)[32 * (BK / 4)]) {
  constexpr uint NT = QB * 64;
  const uint G = p.Hq / p.Hkv;
  const uint kvh = tg.y;
  const uint rg = sg / 2, dh = sg % 2;        // row group (query), head-dim half
  const uint t0 = tg.x * QB;
  const uint qi = t0 + rg;
  const bool qvalid = qi < p.T;
  const uint qpos = p.pos0 + min(qi, p.T - 1);
  const uint last_q = min(t0 + QB, p.T) - 1;
  const uint kend = p.pos0 + last_q + 1;
  const uint qd = lane / 4;
  const uint frow = (qd & 4) + (lane / 2) % 4;
  const uint fcol = (qd & 2) * 2 + (lane % 2) * 2;

  // Q (4 queries x 8 heads = 32 rows) -> threadgroup -> registers (own half)
  simdgroup_half8x8 qf[kDH / 8];
  constexpr uint QPP = BK / 8;  // queries staged per pass (BK rows of the tile buffer)
  for (uint part = 0; part < QB / QPP; ++part) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < BK * kHD / 4; i += NT) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      uint s = part * QPP + r / 8, g = r % 8;
      uint t = min(t0 + s, p.T - 1);
      float4 v = *(const device float4*)(q + (t * p.Hq + kvh * G + g) * kHD + c4 * 4);
      *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = half4(v * p.scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rg / QPP == part)
      for (uint d = 0; d < kDH / 8; ++d) simdgroup_load(qf[d], kv + (rg % QPP) * 8 * kLD2 + dh * kDH + d * 8, kLD2);
  }

  simdgroup_float8x8 o[kDH / 8];
  for (uint j = 0; j < kDH / 8; ++j) o[j] = simdgroup_float8x8(0);
  float m_row = -INFINITY, l_row = 0.0f;
  const device half* kb = kc + kvh * p.max_seq * kHD;
  const device half* vb = vc + kvh * p.max_seq * kHD;

  for (uint k0 = 0; k0 < kend; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < BK * kHD / 8; i += NT) {
      uint r = i / (kHD / 8), c8 = i % (kHD / 8);
      uint4 v = k0 + r < kend ? *(const device uint4*)(kb + (k0 + r) * kHD + c8 * 8) : uint4(0);
      *(threadgroup uint4*)(kv + r * kLD2 + c8 * 8) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr uint NC = BK / 8;
    simdgroup_float8x8 s[NC];
    for (uint c = 0; c < NC; ++c) s[c] = simdgroup_float8x8(0);
    for (uint d = 0; d < kDH / 8; ++d)
      for (uint c = 0; c < NC; ++c) {
        simdgroup_half8x8 b;
        simdgroup_load(b, kv + c * 8 * kLD2 + dh * kDH + d * 8, kLD2, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(s[c], qf[d], b, s[c]);
      }
    // exchange partial scores with the other half
    for (uint c = 0; c < NC; ++c) {
      xchg[sg][lane * (BK / 4) + 2 * c] = s[c].thread_elements()[0];
      xchg[sg][lane * (BK / 4) + 2 * c + 1] = s[c].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sfin[BK / 4];
    for (uint c = 0; c < NC; ++c) {
      sfin[2 * c] = s[c].thread_elements()[0] + xchg[sg ^ 1][lane * (BK / 4) + 2 * c];
      sfin[2 * c + 1] = s[c].thread_elements()[1] + xchg[sg ^ 1][lane * (BK / 4) + 2 * c + 1];
    }
    // Every simdgroup is past the exchange barrier, so K is dead: start loading V now and let the
    // softmax below overlap with it.
    for (uint i = tid; i < BK * kHD / 8; i += NT) {
      uint r = i / (kHD / 8), c8 = i % (kHD / 8);
      uint4 v = k0 + r < kend ? *(const device uint4*)(vb + (k0 + r) * kHD + c8 * 8) : uint4(0);
      *(threadgroup uint4*)(kv + r * kLD2 + c8 * 8) = v;
    }
    float sv[BK / 4];
    float mx = -INFINITY;
    for (uint c = 0; c < NC; ++c) {
      uint key = k0 + c * 8 + fcol;
      sv[2 * c] = key <= qpos ? sfin[2 * c] : -INFINITY;
      sv[2 * c + 1] = key + 1 <= qpos ? sfin[2 * c + 1] : -INFINITY;
      mx = max(mx, max(sv[2 * c], sv[2 * c + 1]));
    }
    mx = max(mx, simd_shuffle_xor(mx, 1));
    mx = max(mx, simd_shuffle_xor(mx, 8));
    const float m_new = max(m_row, mx);
    const float corr = m_new == -INFINITY ? 1.0f : metal::fast::exp(m_row - m_new);
    float ps = 0.0f;
    simdgroup_half8x8 pf[NC];
    for (uint c = 0; c < NC; ++c) {
      float a = sv[2 * c] == -INFINITY ? 0.0f : metal::fast::exp(sv[2 * c] - m_new);
      float b = sv[2 * c + 1] == -INFINITY ? 0.0f : metal::fast::exp(sv[2 * c + 1] - m_new);
      ps += a + b;
      pf[c].thread_elements()[0] = half(a);
      pf[c].thread_elements()[1] = half(b);
    }
    ps += simd_shuffle_xor(ps, 1);
    ps += simd_shuffle_xor(ps, 8);
    l_row = l_row * corr + ps;
    m_row = m_new;
    for (uint j = 0; j < kDH / 8; ++j) {
      o[j].thread_elements()[0] *= corr;
      o[j].thread_elements()[1] *= corr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < kDH / 8; ++j)
      for (uint c = 0; c < NC; ++c) {
        simdgroup_half8x8 vf;
        simdgroup_load(vf, kv + c * 8 * kLD2 + dh * kDH + j * 8, kLD2);
        simdgroup_multiply_accumulate(o[j], pf[c], vf, o[j]);
      }
  }
  if (qvalid) {
    const float inv = l_row > 0.0f ? 1.0f / l_row : 0.0f;
    const uint h = kvh * G + frow;
    const device float* gate = qg + (qi * p.Hq + h) * 2 * kHD + kHD + dh * kDH;
    device float* dst = out + qi * p.Hq * kHD + h * kHD + dh * kDH;
    for (uint j = 0; j < kDH / 8; ++j) {
      uint d = j * 8 + fcol;
      dst[d] = o[j].thread_elements()[0] * inv * sigmoid(gate[d]);
      dst[d + 1] = o[j].thread_elements()[1] * inv * sigmoid(gate[d + 1]);
    }
  }
}



#define NI_ATTN9(NAME, QB, BK)                                                                              \
  [[kernel]] void NAME(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],            \
                       const device half* vc [[buffer(2)]], const device float* qg [[buffer(3)]],           \
                       device float* out [[buffer(4)]], constant AttnPrefParams& p [[buffer(5)]],           \
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half kv[BK * kLD2];                                                                         \
    threadgroup float xchg[QB * 2][32 * (BK / 4)];                                                          \
    attn_prefill9_impl<QB, BK>(q, kc, vc, qg, out, p, tg, tid, sg, lane, kv, xchg);                         \
  }
NI_ATTN9(attn_prefill9_q4_k32, 4, 32)
NI_ATTN9(attn_prefill9_q8_k16, 8, 16)
NI_ATTN9(attn_prefill9_q4_k16, 4, 16)
// attn_prefill9_q8_k32 (QB 8, BK 32) is not instantiated: its static threadgroup memory is 32 * kLD2 halfs
// (kLD2 = kDH + 8 = 264 at D = 256) plus QB * 2 * 32 * (BK / 4) floats, 33,280 B in all, and every Apple GPU caps
// a threadgroup at 32,768 B, so the driver refuses to create the pipeline (AGXMetalG17X, error 3).

template <uint QB, uint BK>
NI_INLINE void attn_prefill10_impl(const device float* q, const device half* kc,
                              const device half* vc, const device float* qg,
                              device float* out, constant AttnPrefParams& p,
                              uint2 tg, uint tid,
                              uint sg, uint lane, threadgroup half* kv, threadgroup float (*xchg)[32 * (BK / 4)]) {
  constexpr uint NT = QB * 64;
  const uint G = p.Hq / p.Hkv;
  const uint kvh = tg.y;
  const uint rg = sg / 2, dh = sg % 2;        // row group (query), head-dim half
  const uint t0 = tg.x * QB;
  const uint qi = t0 + rg;
  const bool qvalid = qi < p.T;
  const uint qpos = p.pos0 + min(qi, p.T - 1);
  const uint last_q = min(t0 + QB, p.T) - 1;
  const uint kend = p.pos0 + last_q + 1;
  const uint qd = lane / 4;
  const uint frow = (qd & 4) + (lane / 2) % 4;
  const uint fcol = (qd & 2) * 2 + (lane % 2) * 2;

  // Q (4 queries x 8 heads = 32 rows) -> threadgroup -> registers (own half)
  simdgroup_half8x8 qf[kDH / 8];
  constexpr uint QPP = BK / 8;  // queries staged per pass (BK rows of the tile buffer)
  for (uint part = 0; part < QB / QPP; ++part) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < BK * kHD / 4; i += NT) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      uint s = part * QPP + r / 8, g = r % 8;
      uint t = min(t0 + s, p.T - 1);
      float4 v = g < G ? *(const device float4*)(q + (t * p.Hq + kvh * G + g) * kHD + c4 * 4) : float4(0);
      *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = half4(v * p.scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rg / QPP == part)
      for (uint d = 0; d < kDH / 8; ++d) simdgroup_load(qf[d], kv + (rg % QPP) * 8 * kLD2 + dh * kDH + d * 8, kLD2);
  }

  simdgroup_float8x8 o[kDH / 8];
  for (uint j = 0; j < kDH / 8; ++j) o[j] = simdgroup_float8x8(0);
  float m_row = -INFINITY, l_row = 0.0f;
  const device half* kb = kc + kvh * p.max_seq * kHD;
  const device half* vb = vc + kvh * p.max_seq * kHD;

  for (uint k0 = 0; k0 < kend; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < BK * kHD / 8; i += NT) {
      uint r = i % BK, c8 = i / BK;  // consecutive threads -> consecutive keys (conflict-free stores)
      uint4 v = k0 + r < kend ? *(const device uint4*)(kb + (k0 + r) * kHD + c8 * 8) : uint4(0);
      thread const half* hv = (thread const half*)&v;
      for (uint e = 0; e < 8; ++e) kv[(c8 * 8 + e) * (BK + 8) + r] = hv[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr uint NC = BK / 8;
    simdgroup_float8x8 s[NC];
    for (uint c = 0; c < NC; ++c) s[c] = simdgroup_float8x8(0);
    for (uint d = 0; d < kDH / 8; ++d)
      for (uint c = 0; c < NC; ++c) {
        simdgroup_half8x8 b;
        half2 kp2 = *(threadgroup const half2*)(kv + (dh * kDH + d * 8 + frow) * (BK + 8) + c * 8 + fcol);
        b.thread_elements()[0] = kp2.x;
        b.thread_elements()[1] = kp2.y;
        simdgroup_multiply_accumulate(s[c], qf[d], b, s[c]);
      }
    // exchange partial scores with the other half
    for (uint c = 0; c < NC; ++c) {
      xchg[sg][lane * (BK / 4) + 2 * c] = s[c].thread_elements()[0];
      xchg[sg][lane * (BK / 4) + 2 * c + 1] = s[c].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sfin[BK / 4];
    for (uint c = 0; c < NC; ++c) {
      sfin[2 * c] = s[c].thread_elements()[0] + xchg[sg ^ 1][lane * (BK / 4) + 2 * c];
      sfin[2 * c + 1] = s[c].thread_elements()[1] + xchg[sg ^ 1][lane * (BK / 4) + 2 * c + 1];
    }
    // Every simdgroup is past the exchange barrier, so K is dead: start loading V now and let the
    // softmax below overlap with it.
    for (uint i = tid; i < BK * kHD / 8; i += NT) {
      uint r = i / (kHD / 8), c8 = i % (kHD / 8);
      uint4 v = k0 + r < kend ? *(const device uint4*)(vb + (k0 + r) * kHD + c8 * 8) : uint4(0);
      *(threadgroup uint4*)(kv + r * kLD2 + c8 * 8) = v;
    }
    float sv[BK / 4];
    float mx = -INFINITY;
    for (uint c = 0; c < NC; ++c) {
      uint key = k0 + c * 8 + fcol;
      sv[2 * c] = key <= qpos ? sfin[2 * c] : -INFINITY;
      sv[2 * c + 1] = key + 1 <= qpos ? sfin[2 * c + 1] : -INFINITY;
      mx = max(mx, max(sv[2 * c], sv[2 * c + 1]));
    }
    mx = max(mx, simd_shuffle_xor(mx, 1));
    mx = max(mx, simd_shuffle_xor(mx, 8));
    const float m_new = max(m_row, mx);
    const float corr = m_new == -INFINITY ? 1.0f : metal::fast::exp(m_row - m_new);
    float ps = 0.0f;
    simdgroup_half8x8 pf[NC];
    for (uint c = 0; c < NC; ++c) {
      float a = sv[2 * c] == -INFINITY ? 0.0f : metal::fast::exp(sv[2 * c] - m_new);
      float b = sv[2 * c + 1] == -INFINITY ? 0.0f : metal::fast::exp(sv[2 * c + 1] - m_new);
      ps += a + b;
      pf[c].thread_elements()[0] = half(a);
      pf[c].thread_elements()[1] = half(b);
    }
    ps += simd_shuffle_xor(ps, 1);
    ps += simd_shuffle_xor(ps, 8);
    l_row = l_row * corr + ps;
    m_row = m_new;
    for (uint j = 0; j < kDH / 8; ++j) {
      o[j].thread_elements()[0] *= corr;
      o[j].thread_elements()[1] *= corr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < kDH / 8; ++j)
      for (uint c = 0; c < NC; ++c) {
        simdgroup_half8x8 vf;
        half2 vp2 = *(threadgroup const half2*)(kv + (c * 8 + frow) * kLD2 + dh * kDH + j * 8 + fcol);
        vf.thread_elements()[0] = vp2.x;
        vf.thread_elements()[1] = vp2.y;
        simdgroup_multiply_accumulate(o[j], pf[c], vf, o[j]);
      }
  }
  if (qvalid && frow < G) {
    const float inv = l_row > 0.0f ? 1.0f / l_row : 0.0f;
    const uint h = kvh * G + frow;
    const device float* gate = qg + (qi * p.Hq + h) * 2 * kHD + kHD + dh * kDH;
    device float* dst = out + qi * p.Hq * kHD + h * kHD + dh * kDH;
    for (uint j = 0; j < kDH / 8; ++j) {
      uint d = j * 8 + fcol;
      dst[d] = o[j].thread_elements()[0] * inv * sigmoid(gate[d]);
      dst[d + 1] = o[j].thread_elements()[1] * inv * sigmoid(gate[d + 1]);
    }
  }
}




#define NI_ATTN10(NAME, QB, BK)                                                                             \
  [[kernel]] void NAME(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],            \
                       const device half* vc [[buffer(2)]], const device float* qg [[buffer(3)]],           \
                       device float* out [[buffer(4)]], constant AttnPrefParams& p [[buffer(5)]],           \
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half kv[kHD * (BK + 8) > BK * kLD2 ? kHD * (BK + 8) : BK * kLD2];                          \
    threadgroup float xchg[QB * 2][32 * (BK / 4)];                                                          \
    attn_prefill10_impl<QB, BK>(q, kc, vc, qg, out, p, tg, tid, sg, lane, kv, xchg);                        \
  }
NI_ATTN10(attn_prefill10_q4_k32, 4, 32)

// ---- Prefill attention v10u: v10 with its fixed-trip loops fully unrolled (A/B for v11) ----
// Nothing else changes. Left to itself the compiler does not fully unroll v10's 16 x 4 fragment
// loop nests, so o[], qf[], s[], pf[] ... are indexed at run time and live in per-thread stack
// memory: 6912 bytes per thread for v10 (v1-v9: 6.4-13 KB; see ni_compile NI_ARCHIVE_DIR), and
// every MMA moves its operands through memory. Unrolled: 96 B of stack, 96 GPRs.
#define NI_UNROLL _Pragma("clang loop unroll(full)")

template <uint QB, uint BK>
NI_INLINE void attn_prefill10u_impl(const device float* q, const device half* kc,
                              const device half* vc, const device float* qg,
                              device float* out, constant AttnPrefParams& p,
                              uint2 tg, uint tid,
                              uint sg, uint lane, threadgroup half* kv, threadgroup float (*xchg)[32 * (BK / 4)]) {
  constexpr uint NT = QB * 64;
  const uint G = p.Hq / p.Hkv;
  const uint kvh = tg.y;
  const uint rg = sg / 2, dh = sg % 2;        // row group (query), head-dim half
  const uint t0 = tg.x * QB;
  const uint qi = t0 + rg;
  const bool qvalid = qi < p.T;
  const uint qpos = p.pos0 + min(qi, p.T - 1);
  const uint last_q = min(t0 + QB, p.T) - 1;
  const uint kend = p.pos0 + last_q + 1;
  const uint qd = lane / 4;
  const uint frow = (qd & 4) + (lane / 2) % 4;
  const uint fcol = (qd & 2) * 2 + (lane % 2) * 2;

  // Q (4 queries x 8 heads = 32 rows) -> threadgroup -> registers (own half)
  simdgroup_half8x8 qf[kDH / 8];
  constexpr uint QPP = BK / 8;  // queries staged per pass (BK rows of the tile buffer)
  NI_UNROLL for (uint part = 0; part < QB / QPP; ++part) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < BK * kHD / 4; i += NT) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      uint s = part * QPP + r / 8, g = r % 8;
      uint t = min(t0 + s, p.T - 1);
      float4 v = g < G ? *(const device float4*)(q + (t * p.Hq + kvh * G + g) * kHD + c4 * 4) : float4(0);
      *(threadgroup half4*)(kv + r * kLD2 + c4 * 4) = half4(v * p.scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rg / QPP == part)
      NI_UNROLL for (uint d = 0; d < kDH / 8; ++d) simdgroup_load(qf[d], kv + (rg % QPP) * 8 * kLD2 + dh * kDH + d * 8, kLD2);
  }

  simdgroup_float8x8 o[kDH / 8];
  NI_UNROLL for (uint j = 0; j < kDH / 8; ++j) o[j] = simdgroup_float8x8(0);
  float m_row = -INFINITY, l_row = 0.0f;
  const device half* kb = kc + kvh * p.max_seq * kHD;
  const device half* vb = vc + kvh * p.max_seq * kHD;

  for (uint k0 = 0; k0 < kend; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < BK * kHD / 8; i += NT) {
      uint r = i % BK, c8 = i / BK;  // consecutive threads -> consecutive keys (conflict-free stores)
      uint4 v = k0 + r < kend ? *(const device uint4*)(kb + (k0 + r) * kHD + c8 * 8) : uint4(0);
      thread const half* hv = (thread const half*)&v;
      NI_UNROLL for (uint e = 0; e < 8; ++e) kv[(c8 * 8 + e) * (BK + 8) + r] = hv[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr uint NC = BK / 8;
    simdgroup_float8x8 s[NC];
    NI_UNROLL for (uint c = 0; c < NC; ++c) s[c] = simdgroup_float8x8(0);
    NI_UNROLL for (uint d = 0; d < kDH / 8; ++d)
      NI_UNROLL for (uint c = 0; c < NC; ++c) {
        simdgroup_half8x8 b;
        half2 kp2 = *(threadgroup const half2*)(kv + (dh * kDH + d * 8 + frow) * (BK + 8) + c * 8 + fcol);
        b.thread_elements()[0] = kp2.x;
        b.thread_elements()[1] = kp2.y;
        simdgroup_multiply_accumulate(s[c], qf[d], b, s[c]);
      }
    // exchange partial scores with the other half
    NI_UNROLL for (uint c = 0; c < NC; ++c) {
      xchg[sg][lane * (BK / 4) + 2 * c] = s[c].thread_elements()[0];
      xchg[sg][lane * (BK / 4) + 2 * c + 1] = s[c].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sfin[BK / 4];
    NI_UNROLL for (uint c = 0; c < NC; ++c) {
      sfin[2 * c] = s[c].thread_elements()[0] + xchg[sg ^ 1][lane * (BK / 4) + 2 * c];
      sfin[2 * c + 1] = s[c].thread_elements()[1] + xchg[sg ^ 1][lane * (BK / 4) + 2 * c + 1];
    }
    // Every simdgroup is past the exchange barrier, so K is dead: start loading V now and let the
    // softmax below overlap with it.
    for (uint i = tid; i < BK * kHD / 8; i += NT) {
      uint r = i / (kHD / 8), c8 = i % (kHD / 8);
      uint4 v = k0 + r < kend ? *(const device uint4*)(vb + (k0 + r) * kHD + c8 * 8) : uint4(0);
      *(threadgroup uint4*)(kv + r * kLD2 + c8 * 8) = v;
    }
    float sv[BK / 4];
    float mx = -INFINITY;
    NI_UNROLL for (uint c = 0; c < NC; ++c) {
      uint key = k0 + c * 8 + fcol;
      sv[2 * c] = key <= qpos ? sfin[2 * c] : -INFINITY;
      sv[2 * c + 1] = key + 1 <= qpos ? sfin[2 * c + 1] : -INFINITY;
      mx = max(mx, max(sv[2 * c], sv[2 * c + 1]));
    }
    mx = max(mx, simd_shuffle_xor(mx, 1));
    mx = max(mx, simd_shuffle_xor(mx, 8));
    const float m_new = max(m_row, mx);
    const float corr = m_new == -INFINITY ? 1.0f : metal::fast::exp(m_row - m_new);
    float ps = 0.0f;
    simdgroup_half8x8 pf[NC];
    NI_UNROLL for (uint c = 0; c < NC; ++c) {
      float a = sv[2 * c] == -INFINITY ? 0.0f : metal::fast::exp(sv[2 * c] - m_new);
      float b = sv[2 * c + 1] == -INFINITY ? 0.0f : metal::fast::exp(sv[2 * c + 1] - m_new);
      ps += a + b;
      pf[c].thread_elements()[0] = half(a);
      pf[c].thread_elements()[1] = half(b);
    }
    ps += simd_shuffle_xor(ps, 1);
    ps += simd_shuffle_xor(ps, 8);
    l_row = l_row * corr + ps;
    m_row = m_new;
    NI_UNROLL for (uint j = 0; j < kDH / 8; ++j) {
      o[j].thread_elements()[0] *= corr;
      o[j].thread_elements()[1] *= corr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    NI_UNROLL for (uint j = 0; j < kDH / 8; ++j)
      NI_UNROLL for (uint c = 0; c < NC; ++c) {
        simdgroup_half8x8 vf;
        half2 vp2 = *(threadgroup const half2*)(kv + (c * 8 + frow) * kLD2 + dh * kDH + j * 8 + fcol);
        vf.thread_elements()[0] = vp2.x;
        vf.thread_elements()[1] = vp2.y;
        simdgroup_multiply_accumulate(o[j], pf[c], vf, o[j]);
      }
  }
  if (qvalid && frow < G) {
    const float inv = l_row > 0.0f ? 1.0f / l_row : 0.0f;
    const uint h = kvh * G + frow;
    const device float* gate = qg + (qi * p.Hq + h) * 2 * kHD + kHD + dh * kDH;
    device float* dst = out + qi * p.Hq * kHD + h * kHD + dh * kDH;
    NI_UNROLL for (uint j = 0; j < kDH / 8; ++j) {
      uint d = j * 8 + fcol;
      dst[d] = o[j].thread_elements()[0] * inv * sigmoid(gate[d]);
      dst[d + 1] = o[j].thread_elements()[1] * inv * sigmoid(gate[d + 1]);
    }
  }
}

#define NI_ATTN10U(NAME, QB, BK)                                                                            \
  [[kernel]] void NAME(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],            \
                       const device half* vc [[buffer(2)]], const device float* qg [[buffer(3)]],           \
                       device float* out [[buffer(4)]], constant AttnPrefParams& p [[buffer(5)]],           \
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half kv[kHD * (BK + 8) > BK * kLD2 ? kHD * (BK + 8) : BK * kLD2];                          \
    threadgroup float xchg[QB * 2][32 * (BK / 4)];                                                          \
    attn_prefill10u_impl<QB, BK>(q, kc, vc, qg, out, p, tg, tid, sg, lane, kv, xchg);                       \
  }
NI_ATTN10U(attn_prefill10u_q4_k32, 4, 32)

// ---- Prefill attention v11: one query head per threadgroup (MLX steel-attention layout) ----
// Threadgroup = 32 consecutive queries of ONE query head, 256 threads; grid = (ceil(T/32), Hq).
// 8 simdgroups = 4 row groups (8 queries, one fragment row each) x 2 halves of the head dim; each
// keeps Q (its 128 dims, half) and O (its 128 dims, fp32) in registers, and the pair sums partial
// scores through threadgroup memory. Every fixed-trip loop is fully unrolled, so all fragment
// arrays stay in registers (v1-v10 keep them on the stack, see v10u). Against v10 also:
//  - rows are queries, not GQA heads: no padded rows when G < 8 (the 27B has G = 6, a quarter of
//    v10's MMAs are padding there), any G works, and only tiles that reach past a row group's
//    first query need the causal mask;
//  - softmax in the log2 domain (exp2 on scores scaled by log2e) with a finite sentinel for masked
//    scores, so no -inf special cases; P stays fp32 as the A operand of the P.V multiply;
//  - each lane fetches 2*BK contiguous bytes of one K row (v10: 16 bytes from 4 places);
//  - the score exchange reuses the K tile (XS = false: 20 KB at BK = 32, as MLX, two more
//    barriers per tile) or has its own buffer (XS = true);
//  - V is fetched before the softmax (MLX order); heaviest query blocks are dispatched first.
// Not taken from MLX: simdgroup_barrier fences between fragment loads and MMAs (here they raise
// an isolated P.V loop from 52 to 92 GPRs plus spills) and max_total_threads_per_threadgroup
// (v10 compiles to identical machine code with it; v11 keeps it, it is what the host dispatches).
// Registers (ni_compile NI_ARCHIVE_DIR): BK = 32 sits at the compiler's ~92-96 GPR ceiling with
// ~100 B of residual spill; BK = 16 halves S and the tiles at twice the per-key barrier/softmax work.
// q: [T, Hq, D] fp32 (normed + roped); kc/vc: [Hkv, max_seq, D] half; qg: [T, Hq, 2D] (gate half).
// out: [T, Hq*D] = softmax(q k^T * scale) v * sigmoid(gate), causal at absolute positions pos0 + t.
constant constexpr uint kBQ11 = 32;            // queries per threadgroup
constant constexpr uint kVLD11 = kHD + 8;      // V tile row stride (halves)
constant constexpr float kNeg11 = -1.0e30f;    // masked score: exp2 underflows to 0, no inf arithmetic

// K^T row stride (halves; conflict-free half2 fragment reads), tile buffer (halves), exchange (float2)
template <uint BK> constexpr uint attn11_kld() { return BK + 8; }
template <uint BK> constexpr uint attn11_kv() { return kHD * (BK + 8) > BK * kVLD11 ? kHD * (BK + 8) : BK * kVLD11; }
template <uint BK> constexpr uint attn11_xs() { return 8 * (BK / 8) * 32; }

// K rows of one tile into registers: thread = (key tid % BK, BK-dim slice tid / BK), 2*BK bytes.
template <uint BK>
NI_INLINE void attn11_fetch_k(thread uint4* r, const device half* kb, uint k0, uint kend, uint tid) {
  const uint key = tid % BK;
  const bool ok = k0 + key < kend;
  const device uint4* src = (const device uint4*)(kb + (k0 + key) * kHD + (tid / BK) * BK);
  NI_UNROLL for (uint u = 0; u < BK / 8; ++u) r[u] = ok ? src[u] : uint4(0);
}

// K^T store: for each dim, consecutive lanes write consecutive keys.
template <uint BK>
NI_INLINE void attn11_store_k(threadgroup half* kt, thread const uint4* r, uint tid) {
  threadgroup half* dst = kt + (tid / BK) * BK * attn11_kld<BK>() + tid % BK;
  NI_UNROLL for (uint u = 0; u < BK / 8; ++u)
    NI_UNROLL for (uint w = 0; w < 4; ++w) {
      const half2 h2 = as_type<half2>(r[u][w]);
      dst[(u * 8 + w * 2) * attn11_kld<BK>()] = h2.x;
      dst[(u * 8 + w * 2 + 1) * attn11_kld<BK>()] = h2.y;
    }
}

// V rows of one tile: thread = (key, BK-dim slice); 256 / BK lanes cover one 512-byte row.
template <uint BK>
NI_INLINE void attn11_fetch_v(thread uint4* r, const device half* vb, uint k0, uint kend, uint tid) {
  const uint key = tid / (kHD / BK);
  const bool ok = k0 + key < kend;
  const device uint4* src = (const device uint4*)(vb + (k0 + key) * kHD + (tid % (kHD / BK)) * BK);
  NI_UNROLL for (uint u = 0; u < BK / 8; ++u) r[u] = ok ? src[u] : uint4(0);
}

template <uint BK>
NI_INLINE void attn11_store_v(threadgroup half* vt, thread const uint4* r, uint tid) {
  threadgroup uint4* dst = (threadgroup uint4*)(vt + (tid / (kHD / BK)) * kVLD11 + (tid % (kHD / BK)) * BK);
  NI_UNROLL for (uint u = 0; u < BK / 8; ++u) dst[u] = r[u];
}

// SPLIT (attn_prefill11gs, NI_SHORTPF): the block's key tiles are divided over grid.z = nsplit
// threadgroups, each writing its unnormalized O rows, running max and sum to part_o / part_ml
// ([split][t][head]); attn11_reduce combines them. For few queries over a long cached prefix,
// where (ceil(T/32) x Hq) threadgroups alone cannot fill the GPU.
// GQ (attn_prefill11g, NI_SHORTPF): rows are (query, head) pairs of one KV head's G query heads,
// row r = query r / G, head kvh*G + r % G; grid = (ceil(T*G / 32), Hkv[, nsplit]). For few queries
// the 32-row blocks then hold G queries' worth of rows instead of padding T up to 32.
// MASK (attn_prefill11m, Qwen3.8-Flash-Next's sparse attention; BK = 32, not GQ): query t attends only to the keys
// whose bit is set in its mask row (mask[t * mwords + key / 32]), causally; key tiles no query of the block selected
// are skipped (tflag: 2 uints of threadgroup memory).
template <uint BK, bool XS, bool SPLIT = false, bool GQ = false, bool RSKIP = false, bool MASK = false>
NI_INLINE void attn_prefill11_impl(const device float* q, const device half* kc, const device half* vc,
                                   const device float* qg, device float* out, constant AttnPrefParams& p,
                                   uint2 tg, uint tid, uint sg, uint lane, threadgroup half* kv,
                                   threadgroup float2* xs, uint split = 0, uint nsplit = 1,
                                   device float* part_o = nullptr, device float2* part_ml = nullptr,
                                   const device uint* mask = nullptr, uint mwords = 0,
                                   threadgroup uint* tflag = nullptr) {
  constexpr uint NC = BK / 8;                     // key fragments per tile
  constexpr uint KLD = attn11_kld<BK>();
  const uint G = p.Hq / p.Hkv;
  const uint rg = sg / 2, dh = sg % 2;            // row group (8 queries), head-dim half
  const uint qd = lane / 4;
  const uint frow = (qd & 4) + (lane / 2) % 4;    // this lane's fragment row and first column
  const uint fcol = (qd & 2) * 2 + (lane % 2) * 2;
  uint h, kvh, t, qpos_rg, kend, tq0 = 0;
  if (GQ) {
    kvh = tg.y;
    const uint nrb = (p.T * G + kBQ11 - 1) / kBQ11;
    const uint r0 = (nrb - 1 - tg.x) * kBQ11;     // heaviest (latest) blocks first
    const uint r = r0 + rg * 8 + frow;
    t = r / G;
    h = kvh * G + r % G;
    qpos_rg = p.pos0 + (r0 + rg * 8) / G;
    kend = p.pos0 + min((r0 + kBQ11 - 1) / G + 1, p.T);
  } else {
    h = tg.y;
    kvh = h / G;
    const uint nqb = (p.T + kBQ11 - 1) / kBQ11;
    const uint t0 = (nqb - 1 - tg.x) * kBQ11;     // heaviest (latest) blocks first: short last wave
    tq0 = t0;
    t = t0 + rg * 8 + frow;                       // this lane's query
    qpos_rg = p.pos0 + t0 + rg * 8;               // first query position of the row group
    kend = p.pos0 + min(t0 + kBQ11, p.T);         // keys needed by the block
  }
  const uint qpos = p.pos0 + t;
  const uint ntiles = (kend + BK - 1) / BK;
  uint it_begin = 0, it_end = ntiles;
  if (SPLIT) {
    const uint per = (ntiles + nsplit - 1) / nsplit;
    it_begin = min(split * per, ntiles);
    it_end = min(it_begin + per, ntiles);
  }

  // Q straight from device into fragments (own 128 dims), scaled by 1/sqrt(D) (a power of two:
  // the half rounding is v10's). Rows past T are zero and never stored.
  simdgroup_half8x8 qf[kDH / 8];
  {
    const device float* qr = q + (min(t, p.T - 1) * p.Hq + h) * kHD + dh * kDH + fcol;
    NI_UNROLL for (uint d = 0; d < kDH / 8; ++d) {
      const half2 v = t < p.T ? half2(*(const device float2*)(qr + d * 8) * p.scale) : half2(0);
      qf[d].thread_elements()[0] = v.x;
      qf[d].thread_elements()[1] = v.y;
    }
  }

  simdgroup_float8x8 o[kDH / 8];
  NI_UNROLL for (uint j = 0; j < kDH / 8; ++j) o[j] = simdgroup_float8x8(0);
  float m_row = kNeg11, l_row = 0.0f;             // running max (log2 domain) and sum of the lane's row
  const device half* kb = kc + kvh * p.max_seq * kHD;
  const device half* vb = vc + kvh * p.max_seq * kHD;
  threadgroup float2* xw = XS ? xs : (threadgroup float2*)kv;
  const threadgroup half* kfr = kv + (dh * kDH + frow) * KLD + fcol;     // K^T fragment base
  const threadgroup half* vfr = kv + frow * kVLD11 + dh * kDH + fcol;   // V fragment base

  for (uint it = it_begin; it < it_end; ++it) {
    const uint k0 = it * BK;
    uint mword = 0;  // MASK: this lane's query's selection bits in the tile
    if (MASK) {
      if (sg == 0) {  // the block's 32 queries: any selected key in the tile?
        const uint tq = tq0 + lane;
        const uint any = simd_or(tq < p.T ? mask[tq * mwords + k0 / 32] : 0u);
        if (lane == 0) tflag[it & 1] = any;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tflag[it & 1] == 0) continue;  // (uniform over the threadgroup)
      mword = mask[min(t, p.T - 1) * mwords + k0 / 32];
    }
    uint4 r[BK / 8];
    attn11_fetch_k<BK>(r, kb, k0, kend, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);   // previous P.V is done with the buffer
    attn11_store_k<BK>(kv, r, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S (8 queries x BK keys) = Q K^T over this simdgroup's 128 dims
    simdgroup_float8x8 s[NC];
    NI_UNROLL for (uint c = 0; c < NC; ++c) s[c] = simdgroup_float8x8(0);
    NI_UNROLL for (uint d = 0; d < kDH / 8; ++d) {
      simdgroup_half8x8 kf[NC];
      NI_UNROLL for (uint c = 0; c < NC; ++c) {
        const half2 k2 = *(const threadgroup half2*)(kfr + d * 8 * KLD + c * 8);
        kf[c].thread_elements()[0] = k2.x;
        kf[c].thread_elements()[1] = k2.y;
      }
      NI_UNROLL for (uint c = 0; c < NC; ++c) simdgroup_multiply_accumulate(s[c], qf[d], kf[c], s[c]);
    }

    // sum the two head-dim halves: one float2 per lane per fragment, lane-consecutive
    if (!XS) threadgroup_barrier(mem_flags::mem_threadgroup);  // K reads done: the exchange reuses it
    NI_UNROLL for (uint c = 0; c < NC; ++c)
      xw[(sg * NC + c) * 32 + lane] = float2(s[c].thread_elements()[0], s[c].thread_elements()[1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    NI_UNROLL for (uint c = 0; c < NC; ++c) {
      const float2 o2 = xw[((sg ^ 1) * NC + c) * 32 + lane];
      s[c].thread_elements()[0] += o2.x;
      s[c].thread_elements()[1] += o2.y;
    }
    if (!XS) threadgroup_barrier(mem_flags::mem_threadgroup);  // exchange reads done: V may overwrite
    // V load issued before the softmax so the two overlap
    attn11_fetch_v<BK>(r, vb, k0, kend, tid);
    attn11_store_v<BK>(kv, r, tid);

    // online softmax on this lane's scores (row frow; the row's other lanes are lane^1, lane^8)
    const bool diag = k0 + BK - 1 > qpos_rg;      // tile reaches past the row group's first query
    float mx = kNeg11;
    NI_UNROLL for (uint c = 0; c < NC; ++c)
      NI_UNROLL for (uint i = 0; i < 2; ++i) {
        float v = s[c].thread_elements()[i] * M_LOG2E_F;
        if (diag && k0 + c * 8 + fcol + i > qpos) v = kNeg11;
        if (MASK && !((mword >> (c * 8 + fcol + i)) & 1u)) v = kNeg11;
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
        const float e = metal::fast::exp2(s[c].thread_elements()[i] - m_new);
        s[c].thread_elements()[i] = e;
        ps += e;
      }
    ps += simd_shuffle_xor(ps, 1);
    ps += simd_shuffle_xor(ps, 8);
    l_row = l_row * corr + ps;
    m_row = m_new;
    // RSKIP: no row of the simdgroup raised its max (corr == 1 exactly): the rescale is x * 1.0
    if (!RSKIP || !simd_all(corr == 1.0f))
      NI_UNROLL for (uint j = 0; j < kDH / 8; ++j) {
        o[j].thread_elements()[0] *= corr;
        o[j].thread_elements()[1] *= corr;
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // V tile visible

    // O += P V (P: the fp32 fragments as they came out of the softmax)
    NI_UNROLL for (uint j = 0; j < kDH / 8; ++j) {
      simdgroup_half8x8 vf[NC];
      NI_UNROLL for (uint c = 0; c < NC; ++c) {
        const half2 v2 = *(const threadgroup half2*)(vfr + c * 8 * kVLD11 + j * 8);
        vf[c].thread_elements()[0] = v2.x;
        vf[c].thread_elements()[1] = v2.y;
      }
      NI_UNROLL for (uint c = 0; c < NC; ++c) simdgroup_multiply_accumulate(o[j], s[c], vf[c], o[j]);
    }
  }

  if (SPLIT) {
    if (t < p.T) {
      const uint row = (split * p.T + t) * p.Hq + h;
      device float* dst = part_o + row * kHD + dh * kDH + fcol;
      NI_UNROLL for (uint j = 0; j < kDH / 8; ++j)
        *(device float2*)(dst + j * 8) = float2(o[j].thread_elements()[0], o[j].thread_elements()[1]);
      if (dh == 0 && fcol == 0) part_ml[row] = float2(m_row, l_row);
    }
    return;
  }
  if (t < p.T) {
    const float inv = 1.0f / l_row;   // l_row >= 1: the row's maximum contributed exp2(0)
    const device float* gate = qg + (t * p.Hq + h) * 2 * kHD + kHD + dh * kDH + fcol;
    device float* dst = out + (t * p.Hq + h) * kHD + dh * kDH + fcol;
    NI_UNROLL for (uint j = 0; j < kDH / 8; ++j) {
      const float2 g = *(const device float2*)(gate + j * 8);
      const float2 r = float2(o[j].thread_elements()[0], o[j].thread_elements()[1]) * inv;
      *(device float2*)(dst + j * 8) = float2(r.x * sigmoid(g.x), r.y * sigmoid(g.y));
    }
  }
}

#define NI_ATTN11(NAME, BK, XS)                                                                             \
  [[kernel, max_total_threads_per_threadgroup(256)]] void NAME(                                             \
      const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],                            \
      const device half* vc [[buffer(2)]], const device float* qg [[buffer(3)]],                           \
      device float* out [[buffer(4)]], constant AttnPrefParams& p [[buffer(5)]],                           \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                 \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {               \
    threadgroup uint4 kvm[attn11_kv<BK>() / 8];                                                              \
    threadgroup float2 xs[XS ? attn11_xs<BK>() : 1];                                                         \
    attn_prefill11_impl<BK, XS>(q, kc, vc, qg, out, p, tg, tid, sg, lane, (threadgroup half*)kvm, xs);      \
  }
// GQA-row form with the keys split over grid.z = nsplit threadgroups (grid (ceil(T*G/32), Hkv, nsplit)).
// 11g/11gs skip unit O rescales like attn_prefill11r (bitwise neutral).
[[kernel, max_total_threads_per_threadgroup(256)]] void attn_prefill11gs(
    const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],
    const device float* qg [[buffer(3)]], device float* out [[buffer(4)]], constant AttnPrefParams& p [[buffer(5)]],
    device float* part_o [[buffer(6)]], device float2* part_ml [[buffer(7)]], constant uint& nsplit [[buffer(8)]],
    uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint4 kvm[attn11_kv<32>() / 8];
  threadgroup float2 xs[1];
  attn_prefill11_impl<32, false, true, true, true>(q, kc, vc, qg, out, p, tg.xy, tid, sg, lane, (threadgroup half*)kvm,
                                                   xs, tg.z, nsplit, part_o, part_ml);
}

// GQA-row form without the key split (grid (ceil(T*G/32), Hkv)).
[[kernel, max_total_threads_per_threadgroup(256)]] void attn_prefill11g(
    const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],
    const device float* qg [[buffer(3)]], device float* out [[buffer(4)]], constant AttnPrefParams& p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint4 kvm[attn11_kv<32>() / 8];
  threadgroup float2 xs[1];
  attn_prefill11_impl<32, false, false, true, true>(q, kc, vc, qg, out, p, tg, tid, sg, lane, (threadgroup half*)kvm, xs);
}

// Combines attn_prefill11gs partials: out[t, h*D + d] = (sum_s 2^(m_s - m) o_s[d]) / (sum_s 2^(m_s - m) l_s)
// * sigmoid(gate). grid: (T, Hq) x 256 threads (one per head dim).
[[kernel]] void attn11_reduce(const device float* part_o [[buffer(0)]], const device float2* part_ml [[buffer(1)]],
                              const device float* qg [[buffer(2)]], device float* out [[buffer(3)]],
                              constant AttnPrefParams& p [[buffer(4)]], constant uint& nsplit [[buffer(5)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
  const uint t = tg.x, h = tg.y;
  float m = kNeg11;
  for (uint s = 0; s < nsplit; ++s) m = max(m, part_ml[(s * p.T + t) * p.Hq + h].x);
  float l = 0.0f, o = 0.0f;
  for (uint s = 0; s < nsplit; ++s) {
    const uint row = (s * p.T + t) * p.Hq + h;
    const float2 ml = part_ml[row];
    const float w = ml.y > 0.0f ? metal::fast::exp2(ml.x - m) : 0.0f;
    l += w * ml.y;
    o += w * part_o[row * kHD + d];
  }
  const float g = qg[(t * p.Hq + h) * 2 * kHD + kHD + d];
  out[(t * p.Hq + h) * kHD + d] = o / l * sigmoid(g);
}

// attn_prefill11m: attn_prefill11r over the keys each query's mask row selects (sparse attention; grid as 11r)
[[kernel, max_total_threads_per_threadgroup(256)]] void attn_prefill11m(
    const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],
    const device float* qg [[buffer(3)]], device float* out [[buffer(4)]], constant AttnPrefParams& p [[buffer(5)]],
    const device uint* mask [[buffer(6)]], constant uint& mwords [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint4 kvm[attn11_kv<32>() / 8];
  threadgroup float2 xs[1];
  threadgroup uint tflag[2];
  attn_prefill11_impl<32, false, false, false, true, true>(q, kc, vc, qg, out, p, tg, tid, sg, lane,
                                                           (threadgroup half*)kvm, xs, 0, 1, nullptr, nullptr, mask,
                                                           mwords, tflag);
}

// attn_prefill11r (NI_ATTN=attn_prefill11r): v11 with the O rescale skipped when corr == 1 for all rows
[[kernel, max_total_threads_per_threadgroup(256)]] void attn_prefill11r(
    const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],
    const device float* qg [[buffer(3)]], device float* out [[buffer(4)]], constant AttnPrefParams& p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint4 kvm[attn11_kv<32>() / 8];
  threadgroup float2 xs[1];
  attn_prefill11_impl<32, false, false, false, true>(q, kc, vc, qg, out, p, tg, tid, sg, lane,
                                                     (threadgroup half*)kvm, xs);
}
NI_ATTN11(attn_prefill11, 32, false)         // 20 KB, exchange in the K tile (MLX)
NI_ATTN11(attn_prefill11_x, 32, true)        // 28 KB, own exchange buffer, 2 barriers fewer per tile
NI_ATTN11(attn_prefill11_k16, 16, false)     // 16-key tiles: 12 KB, fewer registers
NI_ATTN11(attn_prefill11_k16x, 16, true)
// attn_prefill11_k16xr: attn_prefill11_k16x with 11r's skipped unit O rescales (bitwise equal to k16x)
[[kernel, max_total_threads_per_threadgroup(256)]] void attn_prefill11_k16xr(
    const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],
    const device float* qg [[buffer(3)]], device float* out [[buffer(4)]], constant AttnPrefParams& p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint4 kvm[attn11_kv<16>() / 8];
  threadgroup float2 xs[attn11_xs<16>()];
  attn_prefill11_impl<16, true, false, false, true>(q, kc, vc, qg, out, p, tg, tid, sg, lane,
                                                    (threadgroup half*)kvm, xs);
}
