// Recursive Flint (FlintForCausalLM, model_type "flint"): the kernels its layers need beyond the base library's (whose
// 4-bit matrix-vector, GEMM, expert and sampling kernels run its projections, MoE, dense MLP and LM head):
//  - flint_prep: per-head RMSNorm of q and k, RoPE on the sliding-window layers (NeoX halves, all 128 dims; the
//    full-attention layers have no positional encoding), k and v into the layer's cache: a sliding-window layer's
//    ring (slot = position % rows) or a full-attention layer's linear cache, or a prefill chunk's own K / V buffer;
//  - flint_attn_d2: one decode row's attention in one dispatch (norms, RoPE, the cache write, split-K, the merge);
//  - flint_attn_partial / flint_attn_reduce: split-K attention of R <= 8 rows over a cache (verify passes), GQA, the
//    window; the per-head sink logits join the softmax in the reduce;
//  - flint_attn_prefill: a prefill chunk's attention, keys before the chunk from the cache, the chunk's own from its
//    K / V buffer, every row's softmax starting from its head's sink;
//  - flint_kv_store: a chunk's K / V into the cache (a ring keeps the last rows);
//  - flint_router_topk / flint_router_shared: the router's logits (bf16 weights, rounded to bf16 as the reference's
//    matmul output) and the sigmoid top-k, its weights renormalized, for decode and verify rows (flint_router_shared:
//    one row, with the shared expert's gate / up); flint_route: the top-k of a prefill chunk's logits. The ungated
//    shared expert's weight is 1;
//  - flint_swiglu_scaled / flint_add_scaled: the prefill's SwiGLU into half rows scaled where they would overflow;
//  - flint_accept / flint_kv_tree_compact: speculative decoding's acceptance and a tree's accepted keys.
// Head dim 128 throughout. K / V caches: half [Hkv][rows][128].
//
// flint_router_topk, flint_router_shared and flint_attn_d2 merge in the threadgroup that finishes last (an arrival
// counter behind device-scope fences, atomic_thread_fence with thread_scope_device). Those fences come with Metal 3.2,
// which macOS 14 does not have: there the library compiles without them (FLINT_FENCES 0), the three kernels only write
// their partial results, and flint_router_topk_merge / flint_router_shared_merge / flint_attn_d2_merge run the same
// merge as a dispatch of their own (p.split; Flint's split_merges(), NI_FLINT_SPLIT=1 on a newer macOS). A GGUF
// import's (IQ experts') do so before Apple9 (M1, M2) too, where the last threadgroup read stale data behind the
// fences (the M1 Pro).

#ifndef NI_UNROLL
#define NI_UNROLL _Pragma("clang loop unroll(full)")
#endif

#if __METAL_VERSION__ >= 320
#define FLINT_FENCES 1
#else
#define FLINT_FENCES 0
#endif

constant constexpr uint kFD = 128;  // head dim

struct FlintPrepP {
  uint R;       // rows
  uint pos0;    // position of row 0
  uint Hq, Hkv;
  uint rows;    // the cache's rows (a ring's, or the linear cache's capacity)
  uint ring;    // 1: slot = position % rows
  uint rope;    // 1: RoPE (sliding-window layers)
  uint chunk;   // 1: k / v into the chunk's buffers [Hkv][R][128] instead of the cache (prefill)
  uint qkv_ld;  // floats per row of qkv
  float eps;
  uint tree;    // 1: the rows are a KFlash tree (buffer 8): row t rotates at position pos0 + depth(t); its k / v still
                // go to cache row pos0 + t until the acceptance moves the accepted path's (flint_kv_tree_compact)
};

// a cache's rows written at positions pos0 + t, t >= t0, of T (flint_kv_store; flint_kv_tree_compact: pos0 alone)
struct FlintStoreP {
  uint T, t0, pos0, Hkv, rows, ring;
};

NI_INLINE size_t flint_kv_index(constant FlintPrepP& p, uint kvh, uint t, uint d) {
  if (p.chunk) return (size_t(kvh) * p.R + t) * kFD + d;
  const uint pos = p.pos0 + t;
  return (size_t(kvh) * p.rows + (p.ring ? pos % p.rows : pos)) * kFD + d;
}

// grid: (Hq + 2 Hkv, R) x 128 threads. qkv: fp32 [R][qkv_ld] (q heads | k heads | v heads); q_out: fp32 [R][Hq * 128];
// inv: fp32 [64], RoPE's inverse frequencies.
[[kernel]] void flint_prep(const device float* qkv [[buffer(0)]], device float* q_out [[buffer(1)]],
                           const device bfloat* qn [[buffer(2)]], const device bfloat* kn [[buffer(3)]],
                           const device float* inv [[buffer(4)]], device half* kc [[buffer(5)]],
                           device half* vc [[buffer(6)]], constant FlintPrepP& p [[buffer(7)]],
                           const device int* tree [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                           uint d [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  threadgroup float row[kFD];
  const uint h = tg.x, t = tg.y;
  const device float* src = qkv + size_t(t) * p.qkv_ld;
  const uint QD = p.Hq * kFD, KD = p.Hkv * kFD;
  if (h >= p.Hq + p.Hkv) {  // a value head: into the cache as it is
    const uint vh = h - p.Hq - p.Hkv;
    vc[flint_kv_index(p, vh, t, d)] = half(src[QD + KD + vh * kFD + d]);
    return;
  }
  const bool is_q = h < p.Hq;
  float x = src[(is_q ? 0 : QD) + (is_q ? h : h - p.Hq) * kFD + d];
  const float ss = tg_sum(x * x, scratch, d, kFD);
  x = x * metal::precise::rsqrt(ss / float(kFD) + p.eps) * float(is_q ? qn[d] : kn[d]);
  if (p.rope) {
    row[d] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float ang = float(p.pos0 + (p.tree ? uint(tree[kKfDep + t]) : t)) * inv[d % (kFD / 2)];
    const float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
    x = d < kFD / 2 ? row[d] * cs - row[d + kFD / 2] * sn : row[d] * cs + row[d - kFD / 2] * sn;
  }
  if (is_q) q_out[(size_t(t) * p.Hq + h) * kFD + d] = x;
  else kc[flint_kv_index(p, h - p.Hq, t, d)] = half(x);
}

// ---- decode / verify attention: split-K over the cache ----
struct FlintAttnP {
  uint R, pos0, Hq, Hkv;
  uint rows, ring;  // the cache's rows; ring: slot = position % rows
  uint window;      // keys a row sees, its own included (0: all)
  uint k0;          // the first key of chunk 0 (row 0's first visible key)
  uint chunk, nchunks;
  float scale;
};
// One threadgroup per (chunk, kv head, row); its G = Hq / Hkv simdgroups are the query heads sharing the kv head, the
// K / V tiles (KT keys) staged once for them; a lane holds 4 of the 128 dims. part_o / part_ml: per (row, head, chunk)
// the unnormalized output, the running max and sum (natural-log domain). grid: (nchunks, Hkv, R) x (32 G) threads.
// TREE (KFlash tree verifies): the rows are a tree (the descriptor's depths and ancestor masks): row r sits at position
// pos0 + depth(r) and sees the history and, of the block (cache rows pos0 .. pos0 + r), itself and its ancestors.
template <uint KT, bool TREE>
NI_INLINE void flint_attn_partial_impl(const device float* q, const device half* kc, const device half* vc,
                                       device float* part_o, device float* part_ml, constant FlintAttnP& p,
                                       const device int* tree, uint3 tg, uint tid, uint sg, uint lane, uint ntg,
                                       threadgroup half4* kt, threadgroup half4* vt) {
  constexpr uint kFKT = KT;
  const uint chunk = tg.x, kvh = tg.y, r = tg.z;
  const uint G = p.Hq / p.Hkv, hq = kvh * G + sg;
  const uint qpos = p.pos0 + (TREE ? uint(tree[kKfDep + r]) : r);
  const uint mrow = TREE ? uint(tree[kKfMask + r]) : 0u;
  const uint lo = p.window && qpos + 1 > p.window ? qpos + 1 - p.window : 0u;  // this row's first key
  const uint start = p.k0 + chunk * p.chunk, end = min(start + p.chunk, p.pos0 + r + 1);
  const float4 qv = ((const device float4*)(q + (size_t(r) * p.Hq + hq) * kFD))[lane] * p.scale;
  float4 acc = 0.0f;
  float m = -INFINITY, l = 0.0f;
  const device half4* kb = (const device half4*)(kc + size_t(kvh) * p.rows * kFD);
  const device half4* vb = (const device half4*)(vc + size_t(kvh) * p.rows * kFD);
  for (uint t0 = start; t0 < end; t0 += kFKT) {
    const uint n = min(kFKT, end - t0);
    for (uint i = tid; i < n * kFD / 4; i += ntg) {
      const uint key = t0 + i / (kFD / 4);
      const uint at = (p.ring ? key % p.rows : key) * (kFD / 4) + i % (kFD / 4);
      kt[i] = kb[at];
      vt[i] = vb[at];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < n; ++j) {
      if (t0 + j < lo) continue;
      if (TREE && t0 + j >= p.pos0 && !((mrow >> (t0 + j - p.pos0)) & 1u)) continue;
      const float s = simd_sum(dot(qv, float4(kt[j * kFD / 4 + lane])));
      const float mn = max(m, s);
      const float corr = metal::fast::exp(m - mn), e = metal::fast::exp(s - mn);
      l = l * corr + e;
      acc = acc * corr + e * float4(vt[j * kFD / 4 + lane]);
      m = mn;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const size_t base = (size_t(r) * p.Hq + hq) * p.nchunks + chunk;
  ((device float4*)(part_o + base * kFD))[lane] = acc;
  if (lane == 0) {
    part_ml[base * 2 + 0] = m;
    part_ml[base * 2 + 1] = l;
  }
}

// 32-key tiles (16 KB of threadgroup memory); _k16: 16-key tiles, the one GPU shader validation can run (its
// instrumentation doubles a kernel's threadgroup memory)
#define FLINT_ATTN_PARTIAL(NAME, KT, TREE)                                                                            \
  [[kernel]] void NAME(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],                     \
                       const device half* vc [[buffer(2)]], device float* part_o [[buffer(3)]],                      \
                       device float* part_ml [[buffer(4)]], constant FlintAttnP& p [[buffer(5)]],                    \
                       const device int* tree [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],              \
                       uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],         \
                       uint lane [[thread_index_in_simdgroup]], uint3 ntg3 [[threads_per_threadgroup]]) {            \
    threadgroup half4 kt[KT * kFD / 4];                                                                              \
    threadgroup half4 vt[KT * kFD / 4];                                                                              \
    flint_attn_partial_impl<KT, TREE>(q, kc, vc, part_o, part_ml, p, tree, tg, tid, sg, lane, ntg3.x, kt, vt);     \
  }
FLINT_ATTN_PARTIAL(flint_attn_partial, 32, false)
FLINT_ATTN_PARTIAL(flint_attn_partial_k16, 16, false)
FLINT_ATTN_PARTIAL(flint_attn_partial_tree, 32, true)
FLINT_ATTN_PARTIAL(flint_attn_partial_tree_k16, 16, true)

// KFlash: after a tree verify's acceptance, the accepted path's K / V (verify rows path[1 .. a], written at cache row
// pos0 + path[i]) move to pos0 + i, where plain decoding puts them, in a ring's slots or a linear cache's rows. Each
// thread copies one element of one head in path order (a destination never holds a later entry's source: path[j] >= j).
// grid: 2 Hkv threadgroups x 128 threads
[[kernel]] void flint_kv_tree_compact(device half* kc [[buffer(0)]], device half* vc [[buffer(1)]],
                                      const device int* tree [[buffer(2)]], constant FlintStoreP& p [[buffer(3)]],
                                      uint tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
  const uint hh = tg % p.Hkv;
  device half* c = (tg >= p.Hkv ? vc : kc) + size_t(hh) * p.rows * kFD;
  const uint a = uint(tree[kKfA]);
  for (uint i = 1; i <= a; ++i) {
    const uint src = uint(tree[kKfPath + i]);
    if (src == i) continue;
    const uint ps = p.pos0 + src, pd = p.pos0 + i;
    c[size_t(p.ring ? pd % p.rows : pd) * kFD + d] = c[size_t(p.ring ? ps % p.rows : ps) * kFD + d];
  }
}

// The chunks combined with the head's sink: softmax over [scores, sink], the sink's column dropped.
// out: fp32 [R][Hq * 128]. grid: (Hq, R) x 128 threads.
[[kernel]] void flint_attn_reduce(const device float* part_o [[buffer(0)]], const device float* part_ml [[buffer(1)]],
                                  const device bfloat* sinks [[buffer(2)]], device float* out [[buffer(3)]],
                                  constant FlintAttnP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                                  uint d [[thread_index_in_threadgroup]]) {
  const uint h = tg.x, r = tg.y;
  const size_t base = (size_t(r) * p.Hq + h) * p.nchunks;
  const float sink = float(sinks[h]);
  float M = sink;
  for (uint c = 0; c < p.nchunks; ++c) M = max(M, part_ml[(base + c) * 2]);
  float num = 0.0f, den = metal::precise::exp(sink - M);
  for (uint c = 0; c < p.nchunks; ++c) {
    const float mc = part_ml[(base + c) * 2];
    if (mc == -INFINITY) continue;
    const float w = metal::precise::exp(mc - M);
    num += w * part_o[(base + c) * kFD + d];
    den += w * part_ml[(base + c) * 2 + 1];
  }
  out[(size_t(r) * p.Hq + h) * kFD + d] = num / den;
}

// ---- prefill attention ----
struct FlintAttnPfP {
  uint T, pos0, Hq, Hkv;
  uint rows, ring, window;
  float scale;
};
constant constexpr uint kFPLD = kFD + 8;

// A threadgroup = 8 query rows x the G = 8 query heads of one KV head (a simdgroup each, its 8 rows a fragment's), the
// 32-key K / V tiles staged once for them; online softmax in base 2 with fp32 statistics, every row starting from its
// head's sink (max = sink, sum = 1: the sink's own term). Keys before the chunk come from the cache (a ring: slot =
// position % rows; the window keeps them in it), the chunk's from ck / cv ([Hkv][T][128]). Row t sees keys
// (pos - window, pos] (all up to pos without a window). q: fp32 [T][Hq * 128]; out: half [T][Hq * 128].
// grid: (ceil(T / 8), Hkv) x (32 G) threads.
template <uint BK>
NI_INLINE void flint_attn_prefill_impl(const device float* q, const device half* kc, const device half* vc,
                                       const device half* ck, const device half* cv, const device bfloat* sinks,
                                       device half* out, constant FlintAttnPfP& p, uint2 tg, uint tid, uint sg,
                                       uint lane, uint ntg, threadgroup half* kt, threadgroup half* vt) {
  constexpr uint kFPBK = BK;
  const uint G = p.Hq / p.Hkv, kvh = tg.y, h = kvh * G + sg;
  const uint t0 = tg.x * 8;
  const float qscale = p.scale * M_LOG2E_F;
  const uint qq = lane / 4;
  const uint rr = (qq & 4) + (lane / 2) % 4, cc = (qq & 2) * 2 + (lane % 2) * 2;
  const uint my_t = t0 + rr;
  simdgroup_half8x8 qf[kFD / 8];
  {
    const device float* qr = q + (size_t(min(my_t, p.T - 1)) * p.Hq + h) * kFD + cc;
    NI_UNROLL for (uint d = 0; d < kFD / 8; ++d) {
      const float2 v = *(const device float2*)(qr + d * 8) * qscale;
      qf[d].thread_elements()[0] = half(v.x);
      qf[d].thread_elements()[1] = half(v.y);
    }
  }
  simdgroup_float8x8 o[kFD / 8];
  NI_UNROLL for (uint d = 0; d < kFD / 8; ++d) o[d] = simdgroup_float8x8(0);
  float m = float(sinks[h]) * M_LOG2E_F, l = 1.0f;
  const uint qpos = p.pos0 + my_t;
  const uint lo = p.window && qpos + 1 > p.window ? qpos + 1 - p.window : 0u;
  const uint kend = p.pos0 + min(t0 + 8, p.T);  // keys the block needs (exclusive)
  const uint kfirst = p.window && p.pos0 + t0 + 1 > p.window ? p.pos0 + t0 + 1 - p.window : 0u;
  const device half* kb = kc + size_t(kvh) * p.rows * kFD;
  const device half* vb = vc + size_t(kvh) * p.rows * kFD;
  const device half* ckb = ck + size_t(kvh) * p.T * kFD;
  const device half* cvb = cv + size_t(kvh) * p.T * kFD;
  for (uint k0 = kfirst / kFPBK * kFPBK; k0 < kend; k0 += kFPBK) {
    for (uint i = tid; i < kFPBK * kFD / 8; i += ntg) {
      const uint r = i / (kFD / 8), c = (i % (kFD / 8)) * 8, key = k0 + r;
      uint4 kv4 = uint4(0), vv4 = uint4(0);
      if (key >= kfirst && key < kend) {  // (keys outside are zeros: a ring slot past the window holds another's)
        const size_t at = key >= p.pos0 ? size_t(key - p.pos0) * kFD + c : size_t(p.ring ? key % p.rows : key) * kFD + c;
        kv4 = *(const device uint4*)((key >= p.pos0 ? ckb : kb) + at);
        vv4 = *(const device uint4*)((key >= p.pos0 ? cvb : vb) + at);
      }
      *(threadgroup uint4*)(kt + r * kFPLD + c) = kv4;
      *(threadgroup uint4*)(vt + r * kFPLD + c) = vv4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 sf[kFPBK / 8];
    NI_UNROLL for (uint j = 0; j < kFPBK / 8; ++j) {
      sf[j] = simdgroup_float8x8(0);
      NI_UNROLL for (uint d = 0; d < kFD / 8; ++d) {
        simdgroup_half8x8 kf;
        simdgroup_load(kf, kt + (j * 8) * kFPLD + d * 8, kFPLD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(sf[j], qf[d], kf, sf[j]);
      }
    }
    float tmax = -1e30f;
    NI_UNROLL for (uint j = 0; j < kFPBK / 8; ++j)
      NI_UNROLL for (uint e = 0; e < 2; ++e) {
        float v = sf[j].thread_elements()[e];
        const uint key = k0 + j * 8 + cc + e;
        if (key > qpos || key < lo || key >= kend) v = -1e30f;
        sf[j].thread_elements()[e] = v;
        tmax = max(tmax, v);
      }
    tmax = max(tmax, simd_shuffle_xor(tmax, 1));
    tmax = max(tmax, simd_shuffle_xor(tmax, 8));
    const float mn = max(m, tmax);
    const float corr = exp2(m - mn);
    float psum = 0.0f;
    simdgroup_half8x8 pf[kFPBK / 8];
    NI_UNROLL for (uint j = 0; j < kFPBK / 8; ++j)
      NI_UNROLL for (uint e = 0; e < 2; ++e) {
        const float v = sf[j].thread_elements()[e];
        const float pv = v <= -1e29f ? 0.0f : exp2(v - mn);
        psum += pv;
        pf[j].thread_elements()[e] = half(pv);
      }
    psum += simd_shuffle_xor(psum, 1);
    psum += simd_shuffle_xor(psum, 8);
    l = l * corr + psum;
    m = mn;
    NI_UNROLL for (uint d = 0; d < kFD / 8; ++d) {
      o[d].thread_elements()[0] *= corr;
      o[d].thread_elements()[1] *= corr;
    }
    NI_UNROLL for (uint j = 0; j < kFPBK / 8; ++j)
      NI_UNROLL for (uint d = 0; d < kFD / 8; ++d) {
        simdgroup_half8x8 vf;
        simdgroup_load(vf, vt + (j * 8) * kFPLD + d * 8, kFPLD);
        simdgroup_multiply_accumulate(o[d], pf[j], vf, o[d]);
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (my_t < p.T) {
    const float inv = 1.0f / l;
    device half* dst = out + (size_t(my_t) * p.Hq + h) * kFD;
    NI_UNROLL for (uint d = 0; d < kFD / 8; ++d)
      NI_UNROLL for (uint e = 0; e < 2; ++e) dst[d * 8 + cc + e] = half(o[d].thread_elements()[e] * inv);
  }
}

// 32-key tiles (17 KB of threadgroup memory); _k16: 16-key tiles, for GPU shader validation (as flint_attn_partial_k16)
#define FLINT_ATTN_PREFILL(NAME, BK)                                                                                  \
  [[kernel, max_total_threads_per_threadgroup(256)]] void NAME(                                                      \
      const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]], \
      const device half* ck [[buffer(3)]], const device half* cv [[buffer(4)]],                                      \
      const device bfloat* sinks [[buffer(5)]], device half* out [[buffer(6)]], constant FlintAttnPfP& p [[buffer(7)]], \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                           \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                           \
      uint2 ntg2 [[threads_per_threadgroup]]) {                                                                      \
    threadgroup half kt[BK * kFPLD];                                                                                 \
    threadgroup half vt[BK * kFPLD];                                                                                 \
    flint_attn_prefill_impl<BK>(q, kc, vc, ck, cv, sinks, out, p, tg, tid, sg, lane, ntg2.x, kt, vt);               \
  }
FLINT_ATTN_PREFILL(flint_attn_prefill, 32)
FLINT_ATTN_PREFILL(flint_attn_prefill_k16, 16)

// A chunk's K / V (half [Hkv][T][128]) into the cache at positions pos0 + t, t >= t0 (a ring keeps its last rows).
// grid: (T - t0, 2 Hkv) x 32 threads
[[kernel]] void flint_kv_store(const device half4* ck [[buffer(0)]], const device half4* cv [[buffer(1)]],
                               device half4* kc [[buffer(2)]], device half4* vc [[buffer(3)]],
                               constant FlintStoreP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_threadgroup]]) {
  const uint t = p.t0 + tg.x, hh = tg.y % p.Hkv;
  const bool isv = tg.y >= p.Hkv;
  const uint pos = p.pos0 + t;
  const size_t dst = (size_t(hh) * p.rows + (p.ring ? pos % p.rows : pos)) * (kFD / 4) + lane;
  const size_t src = (size_t(hh) * p.T + t) * (kFD / 4) + lane;
  if (isv) vc[dst] = cv[src];
  else kc[dst] = ck[src];
}

// ---- speculative decoding: the drafts tokens[P + 1 .. P + K] accepted while they equal the verify rows' samples, then
// the first differing row's sample (or the one after the last draft) at tokens[P + 1 + a]; acc[0] = a. grid: 1 x 1 ----
struct FlintAcceptP {
  uint P, K;
};
[[kernel]] void flint_accept(const device int* samples [[buffer(0)]], device int* tokens [[buffer(1)]],
                             device int* acc [[buffer(2)]], constant FlintAcceptP& p [[buffer(3)]]) {
  uint a = 0;
  while (a < p.K && samples[a] == tokens[p.P + 1 + a]) ++a;
  tokens[p.P + 1 + a] = samples[a];
  acc[0] = int(a);
}

// ---- MoE routing ----
struct FlintRouteP {
  uint R, k;
};
// Sigmoid top-k of R rows of 256 logits (each rounded to bf16 first): idx[r k + j] the experts, lowest index on ties;
// wts[r (k + 1) + j] their scores over the sum of the k (+ 1e-20, as the reference), wts[r (k + 1) + k] = 1 (the
// shared expert's). grid: ceil(R / 8) threadgroups x 256 threads (a row per simdgroup).
[[kernel]] void flint_route(const device float* logits [[buffer(0)]], device uint* idx [[buffer(1)]],
                            device float* wts [[buffer(2)]], constant FlintRouteP& p [[buffer(3)]],
                            uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint per = 8;
  const uint r = tg * 8 + sg;
  if (r >= p.R) return;
  float v[per];
  NI_UNROLL for (uint i = 0; i < per; ++i) {
    const float z = float(bfloat(logits[size_t(r) * 256 + lane * per + i]));
    v[i] = 1.0f / (1.0f + metal::precise::exp(-z));
  }
  uint taken = 0;
  float sel_sum = 0.0f, my_w[8];
  uint my_i[8];
  for (uint j = 0; j < p.k; ++j) {
    float best = -1.0f;
    uint bi = 0xffffffffu;
    NI_UNROLL for (uint i = 0; i < per; ++i)
      if (!(taken & (1u << i)) && v[i] > best) {
        best = v[i];
        bi = lane * per + i;
      }
    const float gbest = simd_max(best);
    const uint win = simd_min(best == gbest ? bi : 0xffffffffu);
    if (win != 0xffffffffu && win / per == lane) taken |= 1u << (win % per);
    my_w[j] = gbest;
    my_i[j] = win;
    sel_sum += gbest;
  }
  if (lane == 0) {
    const float s = sel_sum + 1e-20f;
    for (uint j = 0; j < p.k; ++j) {
      idx[r * p.k + j] = my_i[j];
      wts[r * (p.k + 1) + j] = my_w[j] / s;
    }
    wts[r * (p.k + 1) + p.k] = 1.0f;
  }
}

// ---- prefill SwiGLU into half rows that cannot overflow ----
// Flint's activations reach past half's range in places (a routed expert's silu(gate) * up of 75776 at layer 46 of a chat
// prompt, where bf16 holds it): the prefill's GEMMs read half rows, so a row whose largest |h| passes 2^15 is stored
// divided by a power of two s (exact), and s multiplies what the row's product is read with instead: mode 0, row p of
// a top-k routing's pairs (p = t k + j: its routing weight wts[p + p / k]); mode 1, the shared expert's row t (its
// weight wts[t (k + 1) + k]); mode 2, a dense MLP's row t (scale[t], applied by flint_add_scaled); mode 3, the products
// silu(gate) * up themselves in g (u unused) in moe_bucket3's sorted order (IQ experts: qrmi_gu_*_f32), row m the pair
// rows[m] (its weight wts[q + q / k], q = rows[m]).
// g, u: fp32 [rows][I]; h: half [rows][I]. grid: rows x 256 threads.
struct FlintSwigluP {
  uint I, mode, k;
};
[[kernel]] void flint_swiglu_scaled(const device float* g [[buffer(0)]], const device float* u [[buffer(1)]],
                                    device half* h [[buffer(2)]], device float* wts [[buffer(3)]],
                                    device float* scale [[buffer(4)]], constant FlintSwigluP& p [[buffer(5)]],
                                    const device uint* rows [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const device float* gr = g + size_t(row) * p.I;
  const device float* ur = u + size_t(row) * p.I;
  const bool prod = p.mode == 3;
  float m = 0.0f;
  for (uint i = tid; i < p.I; i += ntg) m = max(m, fabs(prod ? gr[i] : silu(gr[i]) * ur[i]));
  m = tg_max(m, scratch, tid, ntg);
  const float s = m > 32768.0f ? exp2(ceil(log2(m / 32768.0f))) : 1.0f;
  const float inv = 1.0f / s;
  for (uint i = tid; i < p.I; i += ntg) h[size_t(row) * p.I + i] = half((prod ? gr[i] : silu(gr[i]) * ur[i]) * inv);
  if (tid == 0) {
    if (p.mode == 3) wts[rows[row] + rows[row] / p.k] *= s;
    else if (p.mode == 0) wts[row + row / p.k] *= s;
    else if (p.mode == 1) wts[row * (p.k + 1) + p.k] *= s;
    else scale[row] = s;
  }
}

// x[t] += scale[t] y[t] (rows of D floats): a dense MLP's down product of rows flint_swiglu_scaled stored scaled.
// grid: T x 256 threads
[[kernel]] void flint_add_scaled(device float* x [[buffer(0)]], const device float* y [[buffer(1)]],
                                 const device float* scale [[buffer(2)]], constant uint& D [[buffer(3)]],
                                 uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                 uint ntg [[threads_per_threadgroup]]) {
  const float s = scale[t];
  for (uint i = tid; i < D; i += ntg) x[size_t(t) * D + i] += s * y[size_t(t) * D + i];
}

// ---- the router and its top-k in one dispatch ----
// Router logits of R <= 8 rows (rmsnorm(x_r) w against the bf16 router rows, rounded to bf16), the norm folded in as
// qmvf folds it (each simdgroup sums x^2 over the inputs it walks anyway: no prologue, no barrier), then the sigmoid
// top-k (flint_route's) by the threadgroup that finishes last (an arrival counter, reset by it); with xn != null that
// threadgroup also writes the normalized rows (fp32 [R][D], the experts' gate / up input).
// grid: E / NSG threadgroups x 32 NSG threads (a router row per simdgroup). D a multiple of 256.
// split (and without FLINT_FENCES): threadgroup 0 writes the rows' norm scales after the logits (logits[R E + r]) and
// flint_router_topk_merge (1 threadgroup x 32 NSG threads) routes and writes xn instead.
struct FlintRouterTopkP {
  uint R, D, E, k;
  float eps;
  uint write_xn;
  uint split;
};
NI_INLINE void flint_topk_row(const device float* logits, uint E, uint k, uint lane, device uint* idx,
                              device float* wts) {
  float v[8];
  for (uint i = 0; i < 8; ++i) {
    const float z = float(bfloat(logits[lane * 8 + i]));
    v[i] = 1.0f / (1.0f + metal::precise::exp(-z));
  }
  uint taken = 0;
  float sel_sum = 0.0f, my_w[8];
  uint my_i[8];
  for (uint j = 0; j < k; ++j) {
    float best = -1.0f;
    uint bi = 0xffffffffu;
    for (uint i = 0; i < 8; ++i)
      if (!(taken & (1u << i)) && v[i] > best) {
        best = v[i];
        bi = lane * 8 + i;
      }
    const float gbest = simd_max(best);
    const uint win = simd_min(best == gbest ? bi : 0xffffffffu);
    if (win != 0xffffffffu && win / 8 == lane) taken |= 1u << (win % 8);
    my_w[j] = gbest;
    my_i[j] = win;
    sel_sum += gbest;
  }
  if (lane == 0) {
    const float s = sel_sum + 1e-20f;
    for (uint j = 0; j < k; ++j) {
      idx[j] = my_i[j];
      wts[j] = my_w[j] / s;
    }
    wts[k] = 1.0f;
  }
}
[[kernel]] void flint_router_topk(const device float* x [[buffer(0)]], const device bfloat* nw [[buffer(1)]],
                                  const device bfloat* W [[buffer(2)]], device float* logits [[buffer(3)]],
                                  device float* xn [[buffer(4)]], constant FlintRouterTopkP& p [[buffer(5)]],
                                  device uint* idx [[buffer(6)]], device float* wts [[buffer(7)]],
                                  device atomic_uint* cnt [[buffer(8)]], uint tg [[threadgroup_position_in_grid]],
                                  uint ntgs [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup uint last;
  threadgroup float inv_s[8];
  const uint e = tg * nsg + sg;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float ss[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint k0 = lane * 8; k0 < p.D; k0 += 256) {
    const device bfloat* wr = W + size_t(e) * p.D + k0;
    float wv[8];
    NI_UNROLL for (uint i = 0; i < 8; ++i) wv[i] = float(wr[i]) * float(nw[k0 + i]);
    NI_UNROLL for (uint r = 0; r < 8; ++r) {
      if (r >= p.R) break;
      const device float* xr = x + size_t(r) * p.D + k0;
      const float4 a = *(const device float4*)xr, b = *(const device float4*)(xr + 4);
      acc[r] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b.x + wv[5] * b.y + wv[6] * b.z +
                wv[7] * b.w;
      ss[r] += dot(a, a) + dot(b, b);
    }
  }
  NI_UNROLL for (uint r = 0; r < 8; ++r) {
    if (r >= p.R) break;
    const float inv = metal::precise::rsqrt(simd_sum(ss[r]) / float(p.D) + p.eps);
    const float v = simd_sum(acc[r]) * inv;
    if (lane == 0) logits[r * p.E + e] = v;
    if (sg == 0 && lane == 0) inv_s[r] = inv;
    if ((!FLINT_FENCES || p.split) && tg == 0 && sg == 0 && lane == 0) logits[p.R * p.E + r] = inv;
  }
  if (!FLINT_FENCES || p.split) return;
#if FLINT_FENCES
  // the last threadgroup to finish: every row's logits are written
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == 0) last = atomic_fetch_add_explicit(cnt, 1u, memory_order_relaxed) == ntgs - 1 ? 1u : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!last) return;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  for (uint r = sg; r < p.R; r += nsg)
    flint_topk_row(logits + size_t(r) * p.E, p.E, p.k, lane, idx + r * p.k, wts + r * (p.k + 1));
  if (p.write_xn)
    for (uint i = tid; i < p.R * p.D; i += ntg) {
      const uint r = i / p.D, k = i % p.D;
      xn[i] = x[size_t(r) * p.D + k] * inv_s[r] * float(nw[k]);
    }
  if (tid == 0) atomic_store_explicit(cnt, 0u, memory_order_relaxed);
#endif
}
// flint_router_topk's last threadgroup's work as a dispatch of its own (split): the logits and norm scales it wrote
[[kernel]] void flint_router_topk_merge(const device float* x [[buffer(0)]], const device bfloat* nw [[buffer(1)]],
                                        const device float* logits [[buffer(3)]], device float* xn [[buffer(4)]],
                                        constant FlintRouterTopkP& p [[buffer(5)]], device uint* idx [[buffer(6)]],
                                        device float* wts [[buffer(7)]], uint tid [[thread_index_in_threadgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                        uint ntg [[threads_per_threadgroup]]) {
  for (uint r = sg; r < p.R; r += nsg)
    flint_topk_row(logits + size_t(r) * p.E, p.E, p.k, lane, idx + r * p.k, wts + r * (p.k + 1));
  if (p.write_xn)
    for (uint i = tid; i < p.R * p.D; i += ntg) {
      const uint r = i / p.D, k = i % p.D;
      xn[i] = x[size_t(r) * p.D + k] * logits[p.R * p.E + r] * float(nw[k]);
    }
}

// ---- one decode row's router and shared expert in one dispatch ----
// Threadgroups [0, nr): flint_router_topk's work for the row (logits, the norm folded in; the last of them to finish
// routes and writes the normalized row xn). Threadgroups [nr, ..): the shared expert's gate / up rows (ROWS per
// simdgroup), the normalized row staged once per threadgroup: h[k hstride + n] = silu(gate . xn) (up . xn). The routed
// experts' gate / up (moe_gate_up3 over slots [0, k)) follow; the router's latency overlaps the shared expert's weights.
// grid: nr + ceil(SI / (ROWS nsg)) threadgroups x 32 nsg threads.
// split (and without FLINT_FENCES): threadgroup 0 writes the row's norm scale after the logits (logits[E]) and
// flint_router_shared_merge (1 threadgroup x 256 threads) routes and writes xn instead.
struct FlintRouterSharedP {
  uint D, E, k, SI, hstride, nr;
  float eps;
  uint split;
};
template <int ROWS>
NI_INLINE void flint_router_shared_impl(const device float* x, const device bfloat* nw, const device bfloat* W,
                                        device float* logits, device float* xn, constant FlintRouterSharedP& p,
                                        device uint* idx, device float* wts, device atomic_uint* cnt,
                                        const device uint32_t* sgw, const device bfloat* sgs, const device bfloat* sgb,
                                        const device uint32_t* suw, const device bfloat* sus, const device bfloat* sub,
                                        device float* h, threadgroup float* xs, threadgroup float* red,
                                        threadgroup uint* last, uint tg, uint tid, uint ntg, uint sg, uint nsg,
                                        uint lane) {
  if (tg < p.nr) {
    // D = 2048 (Flint::enc_moe_rows takes this kernel only then): a lane's 8 chunks of 8 inputs, every load issued
    // before the arithmetic
    constexpr uint NCH = 2048 / 256;
    const uint e = tg * nsg + sg;
    float acc = 0.0f, ss = 0.0f;
    uint4 wq[NCH], nq[NCH];
    float4 xa[NCH], xb[NCH];
    NI_UNROLL for (uint c = 0; c < NCH; ++c) {
      const uint k0 = lane * 8 + c * 256;
      wq[c] = *(const device uint4*)(W + size_t(e) * p.D + k0);
      nq[c] = *(const device uint4*)(nw + k0);
      xa[c] = *(const device float4*)(x + k0);
      xb[c] = *(const device float4*)(x + k0 + 4);
    }
    NI_UNROLL for (uint c = 0; c < NCH; ++c) {
      const float xv[8] = {xa[c].x, xa[c].y, xa[c].z, xa[c].w, xb[c].x, xb[c].y, xb[c].z, xb[c].w};
      NI_UNROLL for (uint i = 0; i < 8; ++i) {
        const ushort wb = ushort(wq[c][i / 2] >> (16 * (i % 2))), nb = ushort(nq[c][i / 2] >> (16 * (i % 2)));
        acc += float(as_type<bfloat>(wb)) * float(as_type<bfloat>(nb)) * xv[i];
      }
      ss += dot(xa[c], xa[c]) + dot(xb[c], xb[c]);
    }
    const float inv = metal::precise::rsqrt(simd_sum(ss) / float(p.D) + p.eps);
    const float v = simd_sum(acc) * inv;
    if (lane == 0) logits[e] = v;
    if (!FLINT_FENCES || p.split) {
      if (tg == 0 && sg == 0 && lane == 0) logits[p.E] = inv;  // (read by flint_router_shared_merge)
      return;
    }
#if FLINT_FENCES
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) *last = atomic_fetch_add_explicit(cnt, 1u, memory_order_relaxed) == p.nr - 1 ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!*last) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    if (sg == 0) flint_topk_row(logits, p.E, p.k, lane, idx, wts);
    for (uint i = tid; i < p.D; i += ntg) xn[i] = x[i] * inv * float(nw[i]);
    if (tid == 0) atomic_store_explicit(cnt, 0u, memory_order_relaxed);
#endif
    return;
  }
  stage_x(x, p.D, true, p.eps, nw, xs, red, tid, ntg);
  const uint row0 = ((tg - p.nr) * nsg + sg) * ROWS;
  if (row0 >= p.SI) return;
  float ag[ROWS], au[ROWS];
  qmv2_rows<4, 16, ROWS, 64>(sgw, sgs, sgb, xs, p.SI, p.D, row0, lane, ag);
  qmv2_rows<4, 16, ROWS, 64>(suw, sus, sub, xs, p.SI, p.D, row0, lane, au);
  if (lane == 0)
    for (int r = 0; r < ROWS; ++r)
      if (row0 + r < p.SI) h[p.k * p.hstride + row0 + r] = silu(ag[r]) * au[r];
}
[[kernel]] void flint_router_shared(const device float* x [[buffer(0)]], const device bfloat* nw [[buffer(1)]],
                                    const device bfloat* W [[buffer(2)]], device float* logits [[buffer(3)]],
                                    device float* xn [[buffer(4)]], constant FlintRouterSharedP& p [[buffer(5)]],
                                    device uint* idx [[buffer(6)]], device float* wts [[buffer(7)]],
                                    device atomic_uint* cnt [[buffer(8)]], const device uint32_t* sgw [[buffer(9)]],
                                    const device bfloat* sgs [[buffer(10)]], const device bfloat* sgb [[buffer(11)]],
                                    const device uint32_t* suw [[buffer(12)]], const device bfloat* sus [[buffer(13)]],
                                    const device bfloat* sub [[buffer(14)]], device float* h [[buffer(15)]],
                                    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                    uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                    uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[4096];
  threadgroup float red[32];
  threadgroup uint last;
  flint_router_shared_impl<2>(x, nw, W, logits, xn, p, idx, wts, cnt, sgw, sgs, sgb, suw, sus, sub, h, xs, red, &last,
                              tg, tid, ntg, sg, nsg, lane);
}
// flint_router_shared's routing as a dispatch of its own (split): the logits and the norm scale its router wrote
[[kernel]] void flint_router_shared_merge(const device float* x [[buffer(0)]], const device bfloat* nw [[buffer(1)]],
                                          const device float* logits [[buffer(3)]], device float* xn [[buffer(4)]],
                                          constant FlintRouterSharedP& p [[buffer(5)]], device uint* idx [[buffer(6)]],
                                          device float* wts [[buffer(7)]], uint tid [[thread_index_in_threadgroup]],
                                          uint ntg [[threads_per_threadgroup]],
                                          uint sg [[simdgroup_index_in_threadgroup]],
                                          uint lane [[thread_index_in_simdgroup]]) {
  if (sg == 0) flint_topk_row(logits, p.E, p.k, lane, idx, wts);
  const float inv = logits[p.E];
  for (uint i = tid; i < p.D; i += ntg) xn[i] = x[i] * inv * float(nw[i]);
}

// ---- one decode row's attention, v2: K / V staged in 32-key tiles, fp32 fragments ----
// A threadgroup = one split of the visible keys [k0, pos] of one KV head, in 32-key tiles: K and V staged in
// threadgroup memory by coalesced 16-byte loads, the next tile's loads issued into registers before the current one is
// computed. Simdgroup j takes keys 8 j .. 8 j + 7 of a tile: S = Q K^T with the G = 8 query heads as the fragments'
// rows, online softmax in base 2, O += P V, the products in fp32 (the cache's half K / V converted). The 4 simdgroups
// merge in threadgroup memory; with several splits each writes its partial and the threadgroup of the KV head that
// finishes last merges them with the heads' sinks (an arrival counter per KV head, reset by it). The split holding the
// row's own position computes its k (normed, rotated) and v into the cache and its tile (no other split reads it).
// qkv: fp32 [Hq * 128 | Hkv * 128 | Hkv * 128] raw projections; rope: the position's (cos, sin) per frequency;
// out: fp32 [Hq * 128]. grid: (S, Hkv) x 128 threads; Hq / Hkv == 8. _t16: 16-key tiles (two simdgroups take the
// blocks), the one GPU shader validation can run (its instrumentation doubles a kernel's threadgroup memory).
struct FlintAttnD2P {
  uint pos, Hq, Hkv;
  uint rows, ring, rope;
  uint k0;   // the first visible key
  uint tps;  // tiles per split
  uint S;    // splits
  float eps, scale;
  uint split;  // with S > 1 (and always without FLINT_FENCES): the partials alone; flint_attn_d2_merge merges them
};

// a KV head's S partials (part_o / part_ml) and its heads' sinks into out: thread tid of the head's 128 merges head
// tid / 16's dims (tid % 16) 8 .. + 7. The same statements as flint_attn_d2_impl's last threadgroup (kept apart, so
// the fused kernel stays as it was measured)
NI_INLINE void flint_attn_d2_merge_impl(const device float* part_o, const device float* part_ml,
                                        const device bfloat* sinks, device float* out, uint S, uint kvh, uint tid) {
  const uint h = tid / 16, dd = (tid % 16) * 8, hq = kvh * 8 + h;
  const float s2 = float(sinks[hq]) * M_LOG2E_F;
  const device float* ml = part_ml + size_t(hq) * S * 2;
  const device float4* po = (const device float4*)(part_o + size_t(hq) * S * kFD + dd);
  float Mx = s2;
  for (uint t = 0; t < S; ++t) Mx = max(Mx, ml[t * 2]);
  float den = exp2(s2 - Mx);
  float4 n0 = 0.0f, n1 = 0.0f;
  _Pragma("clang loop unroll_count(4)") for (uint t = 0; t < S; ++t) {
    const float wt = exp2(ml[t * 2] - Mx);
    den += wt * ml[t * 2 + 1];
    n0 += wt * po[t * (kFD / 4)];
    n1 += wt * po[t * (kFD / 4) + 1];
  }
  ((device float4*)(out + hq * kFD + dd))[0] = n0 / den;
  ((device float4*)(out + hq * kFD + dd))[1] = n1 / den;
}

// part tid % 16 of the tile's keys tid / 16 + 8 j, j < TK / 8 (zero past ke, and for the row's own position: written
// apart)
template <uint TK>
NI_INLINE void flint_d2_load(const device uint4* kb, const device uint4* vb, uint t0, uint ke, uint own, uint rows,
                             uint ring, uint tid, thread uint4* kr, thread uint4* vr) {
  NI_UNROLL for (uint j = 0; j < TK / 8; ++j) {
    const uint key = t0 + tid / 16 + 8 * j;
    kr[j] = uint4(0);
    vr[j] = uint4(0);
    if (key < ke && key != own) {
      const uint at = (ring ? key % rows : key) * (kFD / 8) + tid % 16;
      kr[j] = kb[at];
      vr[j] = vb[at];
    }
  }
}

NI_INLINE float4 flint_d2_norm_rope(float4 v, const device bfloat* w, uint lane, float eps, uint rope,
                                    constant float2* cs) {
  const float ss = simd_sum(dot(v, v));
  v *= metal::precise::rsqrt(ss / float(kFD) + eps);
  v *= float4(float(w[lane * 4]), float(w[lane * 4 + 1]), float(w[lane * 4 + 2]), float(w[lane * 4 + 3]));
  if (rope) {  // dims d and d + 64: lanes l and l ^ 16
    const float4 o = simd_shuffle_xor(v, 16);
    float4 r;
    NI_UNROLL for (uint e = 0; e < 4; ++e) {
      const uint d = lane * 4 + e;
      const float2 c = cs[d % (kFD / 2)];
      r[e] = d < kFD / 2 ? v[e] * c.x - o[e] * c.y : v[e] * c.x + o[e] * c.y;
    }
    v = r;
  }
  return v;
}

template <uint TK>
NI_INLINE void flint_attn_d2_impl(const device float* qkv, const device bfloat* qn, const device bfloat* kn,
                                  constant float2* cs, device half* kc, device half* vc, const device bfloat* sinks,
                                  device float* out, device float* part_o, device float* part_ml,
                                  device atomic_uint* cnt, constant FlintAttnD2P& p, uint2 tg, uint tid, uint sg,
                                  uint lane, threadgroup uint4* kvt, threadgroup float* qs, threadgroup float* xml,
                                  threadgroup uint& last) {
  constexpr uint G = 8, NK = TK / 8;  // NK simdgroups take the tiles' 8-key blocks
  const uint split = tg.x, kvh = tg.y;
  const uint QD = p.Hq * kFD, KD = p.Hkv * kFD;
  const float l2e = M_LOG2E_F;
  const uint ks = p.k0 + split * p.tps * TK, ke = min(ks + p.tps * TK, p.pos + 1);
  const uint ntiles = (ke - ks + TK - 1) / TK;
  const bool own = p.pos >= ks && p.pos < ke;
  device half* kbh = kc + size_t(kvh) * p.rows * kFD;
  device half* vbh = vc + size_t(kvh) * p.rows * kFD;
  const device uint4* kb = (const device uint4*)kbh;
  const device uint4* vb = (const device uint4*)vbh;
  uint4 kr[NK], vr[NK];
  flint_d2_load<TK>(kb, vb, ks, ke, p.pos, p.rows, p.ring, tid, kr, vr);
  // q of the G heads (two per simdgroup), normalized, rotated, scaled to base 2
  NI_UNROLL for (uint j = 0; j < 2; ++j) {
    const uint h = sg * 2 + j;
    float4 v = ((const device float4*)(qkv + (kvh * G + h) * kFD))[lane];
    v = flint_d2_norm_rope(v, qn, lane, p.eps, p.rope, cs) * (p.scale * l2e);
    ((threadgroup float4*)(qs + h * kFD))[lane] = v;
  }
  // the row's own k / v (the split holding its position): into the cache, and into the tile below
  half4 own_kv = half4(0.0h);
  if (own && sg < 2) {
    const uint slot = p.ring ? p.pos % p.rows : p.pos;
    if (sg == 0) {
      const float4 v = ((const device float4*)(qkv + QD + kvh * kFD))[lane];
      own_kv = half4(flint_d2_norm_rope(v, kn, lane, p.eps, p.rope, cs));
      ((device half4*)(kbh + size_t(slot) * kFD))[lane] = own_kv;
    } else {
      own_kv = half4(((const device float4*)(qkv + QD + KD + kvh * kFD))[lane]);
      ((device half4*)(vbh + size_t(slot) * kFD))[lane] = own_kv;
    }
  }
  simdgroup_float8x8 o[kFD / 8];
  NI_UNROLL for (uint d = 0; d < kFD / 8; ++d) o[d] = simdgroup_float8x8(0);
  const uint qq = lane / 4;
  const uint cc = (qq & 2) * 2 + (lane % 2) * 2;  // this lane's columns (keys) of its fragment row (head)
  const uint row = (qq & 4) + (lane / 2) % 4;
  float m = -1e30f, l = 0.0f;
  const threadgroup half* kth = (const threadgroup half*)kvt + sg * 8 * kFD;
  const threadgroup half* vth = (const threadgroup half*)kvt + TK * kFD + sg * 8 * kFD;
  for (uint t = 0; t < ntiles; ++t) {
    const uint t0 = ks + t * TK;
    NI_UNROLL for (uint j = 0; j < NK; ++j) {
      if (t0 + tid / 16 + 8 * j == p.pos) continue;  // (written below)
      kvt[(tid / 16 + 8 * j) * (kFD / 8) + tid % 16] = kr[j];
      kvt[TK * kFD / 8 + (tid / 16 + 8 * j) * (kFD / 8) + tid % 16] = vr[j];
    }
    if (own && sg < 2 && p.pos >= t0 && p.pos < t0 + TK)
      ((threadgroup half4*)kvt)[(sg * TK + p.pos - t0) * (kFD / 4) + lane] = own_kv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t + 1 < ntiles) flint_d2_load<TK>(kb, vb, t0 + TK, ke, p.pos, p.rows, p.ring, tid, kr, vr);
    if (sg < NK) {  // (the others only load)
    simdgroup_float8x8 s4[4];
    NI_UNROLL for (uint i = 0; i < 4; ++i) s4[i] = simdgroup_float8x8(0);
    NI_UNROLL for (uint d = 0; d < kFD / 8; ++d) {
      simdgroup_float8x8 qf, kf;
      simdgroup_half8x8 kh;
      simdgroup_load(qf, qs + d * 8, kFD);
      simdgroup_load(kh, kth + d * 8, kFD, ulong2(0, 0), true);
      kf.thread_elements()[0] = float(kh.thread_elements()[0]);
      kf.thread_elements()[1] = float(kh.thread_elements()[1]);
      simdgroup_multiply_accumulate(s4[d % 4], qf, kf, s4[d % 4]);
    }
    float sv[2], tmax = -1e30f;
    NI_UNROLL for (uint e = 0; e < 2; ++e) {
      const uint key = t0 + sg * 8 + cc + e;
      sv[e] = s4[0].thread_elements()[e] + s4[1].thread_elements()[e] + s4[2].thread_elements()[e] +
              s4[3].thread_elements()[e];
      if (key >= ke) sv[e] = -1e30f;
      tmax = max(tmax, sv[e]);
    }
    tmax = max(tmax, simd_shuffle_xor(tmax, 1));
    tmax = max(tmax, simd_shuffle_xor(tmax, 8));
    const float mn = max(m, tmax);
    const float corr = exp2(m - mn);
    float psum = 0.0f;
    simdgroup_float8x8 pf;
    NI_UNROLL for (uint e = 0; e < 2; ++e) {
      const float pv = sv[e] <= -1e29f ? 0.0f : exp2(sv[e] - mn);
      psum += pv;
      pf.thread_elements()[e] = pv;
    }
    psum += simd_shuffle_xor(psum, 1);
    psum += simd_shuffle_xor(psum, 8);
    l = l * corr + psum;
    m = mn;
    NI_UNROLL for (uint d = 0; d < kFD / 8; ++d) {
      simdgroup_half8x8 vh;
      simdgroup_float8x8 vf;
      simdgroup_load(vh, vth + d * 8, kFD);
      vf.thread_elements()[0] = float(vh.thread_elements()[0]);
      vf.thread_elements()[1] = float(vh.thread_elements()[1]);
      o[d].thread_elements()[0] *= corr;
      o[d].thread_elements()[1] *= corr;
      simdgroup_multiply_accumulate(o[d], pf, vf, o[d]);
    }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // the simdgroups merged: their O scaled to the split's max into xo [NK][G][128] (the tiles' memory)
  if (cc == 0 && sg < NK) {
    xml[(sg * G + row) * 2 + 0] = m;
    xml[(sg * G + row) * 2 + 1] = l;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float M = -1e30f;
  NI_UNROLL for (uint j = 0; j < NK; ++j) M = max(M, xml[(j * G + row) * 2]);
  const float w = exp2(m - M);
  threadgroup float* xo = (threadgroup float*)kvt;
  if (sg < NK)
    NI_UNROLL for (uint d = 0; d < kFD / 8; ++d)
      NI_UNROLL for (uint e = 0; e < 2; ++e) xo[(sg * G + row) * kFD + d * 8 + cc + e] = o[d].thread_elements()[e] * w;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // thread tid: head h = tid / 16, dims dd .. dd + 7
  const uint h = tid / 16, dd = (tid % 16) * 8, hq = kvh * G + h;
  float Mh = -1e30f, Lh = 0.0f;
  NI_UNROLL for (uint j = 0; j < NK; ++j) Mh = max(Mh, xml[(j * G + h) * 2]);
  NI_UNROLL for (uint j = 0; j < NK; ++j) Lh += xml[(j * G + h) * 2 + 1] * exp2(xml[(j * G + h) * 2] - Mh);
  float4 o0 = 0.0f, o1 = 0.0f;
  NI_UNROLL for (uint j = 0; j < NK; ++j) {
    o0 += ((const threadgroup float4*)(xo + (j * G + h) * kFD + dd))[0];
    o1 += ((const threadgroup float4*)(xo + (j * G + h) * kFD + dd))[1];
  }
  const float s2 = float(sinks[hq]) * l2e;
  if (p.S == 1) {
    const float Mt = max(Mh, s2);
    const float sc = exp2(Mh - Mt) / (Lh * exp2(Mh - Mt) + exp2(s2 - Mt));
    ((device float4*)(out + hq * kFD + dd))[0] = o0 * sc;
    ((device float4*)(out + hq * kFD + dd))[1] = o1 * sc;
    return;
  }
  // several splits: this split's partial, then the KV head's last threadgroup merges them
  const size_t pb = size_t(hq) * p.S + split;
  ((device float4*)(part_o + pb * kFD + dd))[0] = o0;
  ((device float4*)(part_o + pb * kFD + dd))[1] = o1;
  if (tid % 16 == 0) {
    part_ml[pb * 2 + 0] = Mh;
    part_ml[pb * 2 + 1] = Lh;
  }
  if (!FLINT_FENCES || p.split) return;
#if FLINT_FENCES
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == 0) last = atomic_fetch_add_explicit(cnt + kvh, 1u, memory_order_relaxed) == p.S - 1 ? 1u : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!last) return;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  const device float* ml = part_ml + size_t(hq) * p.S * 2;
  const device float4* po = (const device float4*)(part_o + size_t(hq) * p.S * kFD + dd);
  float Mx = s2;
  for (uint t = 0; t < p.S; ++t) Mx = max(Mx, ml[t * 2]);
  float den = exp2(s2 - Mx);
  float4 n0 = 0.0f, n1 = 0.0f;
  _Pragma("clang loop unroll_count(4)") for (uint t = 0; t < p.S; ++t) {
    const float wt = exp2(ml[t * 2] - Mx);
    den += wt * ml[t * 2 + 1];
    n0 += wt * po[t * (kFD / 4)];
    n1 += wt * po[t * (kFD / 4) + 1];
  }
  ((device float4*)(out + hq * kFD + dd))[0] = n0 / den;
  ((device float4*)(out + hq * kFD + dd))[1] = n1 / den;
  if (tid == 0) atomic_store_explicit(cnt + kvh, 0u, memory_order_relaxed);
#endif
}
// flint_attn_d2's merge as a dispatch of its own (split, S > 1): grid Hkv threadgroups x 128 threads
[[kernel]] void flint_attn_d2_merge(const device bfloat* sinks [[buffer(6)]], device float* out [[buffer(7)]],
                                    const device float* part_o [[buffer(8)]], const device float* part_ml [[buffer(9)]],
                                    constant FlintAttnD2P& p [[buffer(11)]], uint kvh [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
  flint_attn_d2_merge_impl(part_o, part_ml, sinks, out, p.S, kvh, tid);
}
#define FLINT_ATTN_D2(NAME, TK)                                                                                        \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                       \
      const device float* qkv [[buffer(0)]], const device bfloat* qn [[buffer(1)]], const device bfloat* kn [[buffer(2)]], \
      constant float2* cs [[buffer(3)]], device half* kc [[buffer(4)]], device half* vc [[buffer(5)]],                \
      const device bfloat* sinks [[buffer(6)]], device float* out [[buffer(7)]], device float* part_o [[buffer(8)]],  \
      device float* part_ml [[buffer(9)]], device atomic_uint* cnt [[buffer(10)]],                                    \
      constant FlintAttnD2P& p [[buffer(11)]], uint2 tg [[threadgroup_position_in_grid]],                             \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],                          \
      uint lane [[thread_index_in_simdgroup]]) {                                                                      \
    threadgroup uint4 kvt[2 * TK * kFD / 8]; /* the K tile then the V tile, [TK][16] uint4 each; then the merge's O */ \
    threadgroup float qs[8 * kFD];                                                                                    \
    threadgroup float xml[TK / 8 * 8 * 2];                                                                            \
    threadgroup uint last;                                                                                            \
    flint_attn_d2_impl<TK>(qkv, qn, kn, cs, kc, vc, sinks, out, part_o, part_ml, cnt, p, tg, tid, sg, lane, kvt, qs, \
                           xml, last);                                                                                \
  }
FLINT_ATTN_D2(flint_attn_d2, 32)
FLINT_ATTN_D2(flint_attn_d2_t16, 16)
