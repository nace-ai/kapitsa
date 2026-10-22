// ---- Fused single-token (decode) MoE layer: 4 dispatches instead of 9 ----
//   A: router (BF16) | latent down (Q8) | shared up (IQ2, relu^2), row segments of one grid over the normed hidden
//      state; the last router threadgroup to finish routes (sigmoid + selection bias, top-k, normalized weights).
//   B: routed up (IQ2, relu^2) of the k selected experts | shared down (IQ2) into sd.
//   C: routed down (nh_moe2_dn_iq2: weighted partials per selected expert).
//   D: latent up (Q8) of lo = sum of C's partials, + sd, added into the residual.
// Same per-row arithmetic as the separate kernels (nh_mv_dot / nh_x2_rows / nh_moe_route1).
// NVFP4 experts and shared expert (Super-VL): nh_moe_a4 (router | latent down | shared up, NVFP4), nh_moe_b4 (routed
// up with each expert's tensor scale | shared down, NVFP4), nh_x4_moe_dn, nh_moe_d1 (as nh_x4_mv / nh_x4_moe_*).

struct NhMoeDecP {
  uint H;            // hidden (A's input, D's output)
  uint E, k, norm;   // experts, used, normalize
  float scale;       // routed scaling factor
  uint L;            // latent
  uint S;            // shared expert width
  uint I;            // routed expert width
  uint up_q_stride, up_s_stride;  // bytes between experts' up codes / scales
  uint tg_r, tg_l;   // A: threadgroups of the router and latent segments (the rest: shared up)
  uint tg_u;         // B: threadgroups of the routed up segment (the rest: shared down)
};

// grid: tg_r + tg_l + ceil(S / 16) threadgroups x 128 threads (4 simdgroups x 4 rows)
[[kernel]] void nh_moe_a1(const device float* xn [[buffer(0)]], const device uchar* rw [[buffer(1)]],
                          const device uchar* lq [[buffer(2)]], const device uchar* ls [[buffer(3)]],
                          const device uchar* uq [[buffer(4)]], const device uchar* us [[buffer(5)]],
                          device float* logits [[buffer(6)]], device float* lat [[buffer(7)]],
                          device float* sh [[buffer(8)]], const device float* bias [[buffer(9)]],
                          device int* eid [[buffer(10)]], device float* ew [[buffer(11)]],
                          device atomic_uint* counter [[buffer(12)]], constant NhMoeDecP& p [[buffer(13)]],
                          uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint arrival;
  const uint tg = tgp.x;
  if (tg < p.tg_r) {  // router rows (BF16)
    const uint row0 = (tg * nsg + sg) * 4;
    float acc[4][1] = {{0}, {0}, {0}, {0}};
    if (row0 < p.E) nh_mv_dot<4, 4, 1>(rw, rw, xn, p.H, p.H, 1, row0, p.E, lane, nullptr, acc);
    nh_mv_store<4, 1, 0>(acc, logits, p.E, p.E, 1, 1.0f, row0, lane);
    // the last router threadgroup to arrive routes
#if NH_FENCES
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      arrival = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (arrival != p.tg_r - 1) return;
    if (sg == 0) nh_route_row((const device float*)logits, bias, p.E, p.k, p.norm, p.scale, lane, eid, ew);
    if (tid == 0) atomic_store_explicit(counter, 0u, memory_order_relaxed);
#endif
    return;
  }
  if (tg < p.tg_r + p.tg_l) {  // latent down rows (Q8)
    const uint row0 = ((tg - p.tg_r) * nsg + sg) * 4;
    float acc[4][1] = {{0}, {0}, {0}, {0}};
    if (row0 < p.L) nh_mv_dot<0, 4, 1>(lq, ls, xn, p.H, p.H, 1, row0, p.L, lane, nullptr, acc);
    nh_mv_store<4, 1, 0>(acc, lat, p.L, p.L, 1, 1.0f, row0, lane);
    return;
  }
  // shared up rows (IQ2, relu^2)
  nh_x2_rows<4, 2>(uq, us, xn, sh, p.S, p.H, 1.0f, ((tg - p.tg_r - p.tg_l) * nsg + sg) * 4, lane);
}

// grid: tg_u = k * ceil(I / 16) + ceil(H / 16) threadgroups x 128 threads
[[kernel]] void nh_moe_b1(const device uchar* eq [[buffer(0)]], const device uchar* es [[buffer(1)]],
                          const device float* lat [[buffer(2)]], device float* h [[buffer(3)]],
                          const device int* eid [[buffer(4)]], const device uchar* dq [[buffer(5)]],
                          const device uchar* ds [[buffer(6)]], const device float* sh [[buffer(7)]],
                          device float* sd [[buffer(8)]], constant NhMoeDecP& p [[buffer(9)]],
                          uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint tg = tgp.x;
  if (tg < p.tg_u) {  // routed up: pair j = tg / tiles per expert
    const uint per = (p.I + 15) / 16, j = tg / per, e = uint(eid[j]);
    nh_x2_rows<4, 2>(eq + size_t(e) * p.up_q_stride, es + size_t(e) * p.up_s_stride, lat, h + size_t(j) * p.I, p.I,
                      p.L, 1.0f, ((tg % per) * nsg + sg) * 4, lane);
    return;
  }
  // shared down rows (IQ2) into sd
  nh_x2_rows<4, 0>(dq, ds, sh, sd, p.H, p.S, 1.0f, ((tg - p.tg_u) * nsg + sg) * 4, lane);
}

// x[n] += W_up[n, :] . (sum_j part[j, :]) + sd[n]. grid: ceil(H / 16) threadgroups x 128 threads
[[kernel]] void nh_moe_d1(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                          const device float* part [[buffer(2)]], const device float* sd [[buffer(3)]],
                          device float* x [[buffer(4)]], constant NhMoeDecP& p [[buffer(5)]],
                          uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint row0 = (tgp.x * nsg + sg) * 4;
  float acc[4] = {0, 0, 0, 0};
  if (row0 < p.H) {
    uint rows[4];
    for (int j = 0; j < 4; ++j) rows[j] = min(row0 + j, p.H - 1);
    for (uint k0 = lane * 32; k0 < p.L; k0 += 32 * 32) {
      float4 xv[8];
      for (int i = 0; i < 8; ++i) xv[i] = float4(0.0f);
      for (uint e = 0; e < p.k; ++e) {  // lo = the experts' weighted partials, summed in expert order
        const device float4* pp = (const device float4*)(part + size_t(e) * p.L + k0);
        for (int i = 0; i < 8; ++i) xv[i] += pp[i];
      }
      for (int j = 0; j < 4; ++j) {
        const size_t n = rows[j];
        const uint4 c0 = *(const device uint4*)(q + n * p.L + k0), c1 = *(const device uint4*)(q + n * p.L + k0 + 16);
        const float sc = float(((const device half*)s)[n * (p.L / 32) + k0 / 32]);
        const uint w[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        float d = 0.0f;
        for (int i = 0; i < 8; ++i) {
          const uint ww = w[i];
          d += dot(float4(float(as_type<char>(uchar(ww))), float(as_type<char>(uchar(ww >> 8))),
                          float(as_type<char>(uchar(ww >> 16))), float(as_type<char>(uchar(ww >> 24)))),
                   xv[i]);
        }
        acc[j] += sc * d;
      }
    }
  }
  for (int j = 0; j < 4; ++j) {
    const float v = simd_sum(acc[j]);
    const uint n = row0 + j;
    if (lane == 0 && n < p.H) x[n] += v + sd[n];
  }
}

// ---- NVFP4 experts and shared expert: 4 rows per simdgroup, any simdgroup count (the engine: 8, 32 rows per
// threadgroup, as nh_x4_moe_*) ----
// grid: tg_r + tg_l + ceil(S / rows) threadgroups
[[kernel]] void nh_moe_a4(const device float* xn [[buffer(0)]], const device uchar* rw [[buffer(1)]],
                          const device uchar* lq [[buffer(2)]], const device uchar* ls [[buffer(3)]],
                          const device uchar* uq [[buffer(4)]], const device uchar* us [[buffer(5)]],
                          device float* logits [[buffer(6)]], device float* lat [[buffer(7)]],
                          device float* sh [[buffer(8)]], const device float* bias [[buffer(9)]],
                          device int* eid [[buffer(10)]], device float* ew [[buffer(11)]],
                          device atomic_uint* counter [[buffer(12)]], constant NhMoeDecP& p [[buffer(13)]],
                          constant float& gs_up [[buffer(14)]], uint3 tgp [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint arrival;
  threadgroup half2 tlh[256];
  const uint tg = tgp.x;
  if (tg < p.tg_r) {  // router rows (BF16)
    const uint row0 = (tg * nsg + sg) * 4;
    float acc[4][1] = {{0}, {0}, {0}, {0}};
    if (row0 < p.E) nh_mv_dot<4, 4, 1>(rw, rw, xn, p.H, p.H, 1, row0, p.E, lane, nullptr, acc);
    nh_mv_store<4, 1, 0>(acc, logits, p.E, p.E, 1, 1.0f, row0, lane);
    // the last router threadgroup to arrive routes
#if NH_FENCES
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      arrival = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (arrival != p.tg_r - 1) return;
    if (sg == 0) nh_route_row((const device float*)logits, bias, p.E, p.k, p.norm, p.scale, lane, eid, ew);
    if (tid == 0) atomic_store_explicit(counter, 0u, memory_order_relaxed);
#endif
    return;
  }
  if (tg < p.tg_r + p.tg_l) {  // latent down rows (Q8)
    const uint row0 = ((tg - p.tg_r) * nsg + sg) * 4;
    float acc[4][1] = {{0}, {0}, {0}, {0}};
    if (row0 < p.L) nh_mv_dot<0, 4, 1>(lq, ls, xn, p.H, p.H, 1, row0, p.L, lane, nullptr, acc);
    nh_mv_store<4, 1, 0>(acc, lat, p.L, p.L, 1, 1.0f, row0, lane);
    return;
  }
  // shared up rows (NVFP4, relu^2)
  nh_x4_lut_to_tg(tlh, tid, nsg * 32);
  nh_x4_mv_impl<0, 2>(uq, us, xn, sh, p.S, p.H, gs_up, ((tg - p.tg_r - p.tg_l) * nsg + sg) * 4, lane, tlh);
}

// grid: tg_u = k * ceil(I / rows) + ceil(H / rows) threadgroups. gse: the routed experts' up scales.
// LE: the routed experts' layout (L0, or L2: nh_moe_b4_l2; the shared down stays L0)
#define NH_MOE_B4(NAME, LE)                                                                                        \
  [[kernel]] void NAME(const device uchar* eq [[buffer(0)]], const device uchar* es [[buffer(1)]],                 \
                       const device float* lat [[buffer(2)]], device float* h [[buffer(3)]],                      \
                       const device int* eid [[buffer(4)]], const device uchar* dq [[buffer(5)]],                  \
                       const device uchar* ds [[buffer(6)]], const device float* sh [[buffer(7)]],                 \
                       device float* sd [[buffer(8)]], constant NhMoeDecP& p [[buffer(9)]],                       \
                       const device float* gse [[buffer(10)]], constant float& gs_dn [[buffer(11)]],               \
                       uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],      \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],           \
                       uint tid [[thread_index_in_threadgroup]]) {                                                 \
    threadgroup half2 tlh[256];                                                                                    \
    nh_x4_lut_to_tg(tlh, tid, nsg * 32);                                                                           \
    const uint tg = tgp.x;                                                                                         \
    if (tg < p.tg_u) { /* routed up: pair j = tg / tiles per expert */                                             \
      const uint per = (p.I + 4 * nsg - 1) / (4 * nsg), j = tg / per, e = uint(eid[j]);                            \
      nh_x4_mv_impl<LE, 2>(eq + size_t(e) * p.up_q_stride, es + size_t(e) * p.up_s_stride, lat,                    \
                           h + size_t(j) * p.I, p.I, p.L, gse[e], ((tg % per) * nsg + sg) * 4, lane, tlh);         \
      return;                                                                                                      \
    }                                                                                                              \
    /* shared down rows (NVFP4) into sd */                                                                         \
    nh_x4_mv_impl<0, 0>(dq, ds, sh, sd, p.H, p.S, gs_dn, ((tg - p.tg_u) * nsg + sg) * 4, lane, tlh);               \
  }
NH_MOE_B4(nh_moe_b4, 0)
NH_MOE_B4(nh_moe_b4_l2, 2)

// x[n] += W_up[n, :] . lo + sd[n] (lo: the summed partials, nh_x4_moe_dn_sum). grid: ceil(H / rows) threadgroups
[[kernel]] void nh_moe_d2(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                          const device float* lo [[buffer(2)]], const device float* sd [[buffer(3)]],
                          device float* x [[buffer(4)]], constant NhMoeDecP& p [[buffer(5)]],
                          uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint row0 = (tgp.x * nsg + sg) * 4;
  float acc[4][1] = {{0}, {0}, {0}, {0}};
  if (row0 < p.H) nh_mv_dot<0, 4, 1>(q, s, lo, p.L, p.L, 1, row0, p.H, lane, nullptr, acc);
  for (int j = 0; j < 4; ++j) {
    const float v = simd_sum(acc[j][0]);
    const uint n = row0 + j;
    if (lane == 0 && n < p.H) x[n] += v + sd[n];
  }
}
