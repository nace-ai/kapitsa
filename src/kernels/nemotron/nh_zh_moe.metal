// MoE layers without latent projections (moe_latent_size null: Nemotron 3.5 Lightning): the routed experts read the
// normed hidden state and their weighted sum is added into the residual stream directly, where LatentMoE layers
// project it up first (nh_moe2_sum / nh_moe_combine + the latent up projection).

// x[r, d] += sum_j part[r * k + j, d] (decode and verify rows; nh_moe2_sum2's additions in its order).
// grid: (ceil(D / 64), R) x 64 threads. Rows >= live[0] are left as they are.
[[kernel]] void nh_moe2_sum2_add(const device float* part [[buffer(0)]], device float* x [[buffer(1)]],
                                 constant NhMoeParams& p [[buffer(2)]], const device uint* live [[buffer(3)]],
                                 uint2 tg [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]],
                                 uint2 ntg [[threads_per_threadgroup]]) {
  const uint r = tg.y, d = tg.x * ntg.x + tid.x;
  if (r >= live[0] || d >= p.D) return;
  float v = 0.0f;
  for (uint j = 0; j < p.k; ++j) v += part[(size_t(r) * p.k + j) * p.D + d];
  x[r * p.D + d] += v;
}

// x[r, d] = (x[r, d] + sum_j part[r * k + j, d]) + sd[r, d]: nh_moe2_sum2_add, then the shared expert's down product
// sd (stored, not added, by its product) added as that product's add epilogue adds it. grid as nh_moe2_sum2_add
[[kernel]] void nh_moe2_sum2_add_sd(const device float* part [[buffer(0)]], device float* x [[buffer(1)]],
                                    constant NhMoeParams& p [[buffer(2)]], const device uint* live [[buffer(3)]],
                                    const device float* sd [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                                    uint2 tid [[thread_position_in_threadgroup]], uint2 ntg [[threads_per_threadgroup]]) {
  const uint r = tg.y, d = tg.x * ntg.x + tid.x;
  if (r >= live[0] || d >= p.D) return;
  float v = 0.0f;
  for (uint j = 0; j < p.k; ++j) v += part[(size_t(r) * p.k + j) * p.D + d];
  const float t = x[r * p.D + d] + v;
  x[r * p.D + d] = t + sd[r * p.D + d];
}

// x[t, d] += sum_j w[t, j] * out[t * k + j, d] (prefill; nh_moe_combine's sums, kept in fp32).
// grid: T threadgroups x 256 threads
[[kernel]] void nh_moe_combine_add(const device float* out [[buffer(0)]], const device float* ew [[buffer(1)]],
                                   device float* x [[buffer(2)]], constant NhCombineP& p [[buffer(3)]],
                                   uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                                   uint ntg [[threads_per_threadgroup]]) {
  for (uint d = tid; d < p.D; d += ntg) {
    float v = 0.0f;
    for (uint j = 0; j < p.k; ++j) v += ew[t * p.k + j] * out[(size_t(t) * p.k + j) * p.D + d];
    x[size_t(t) * p.D + d] += v;
  }
}

// ---- one token's MoE layer in 4 dispatches (nh_zdec.metal's design without the latent projections) ----
//   A nh_moe_ah: router (F32 rows) | shared up (NVFP4, relu^2) over the normed hidden state; the last router
//     threadgroup to finish routes (nh_route_row)
//   B nh_moe_b4: routed up (each expert's scale) of the selected experts on the hidden state | shared down into sd
//   C nh_x4_moe_dn: the routed experts' weighted partials
//   D nh_moe_dh: x += the partials' sum (expert order, as nh_moe2_sum2_add) + sd
// grid A: tg_r + ceil(S / 16) threadgroups x 128 threads (4 simdgroups x 4 rows); p.L = p.H, p.tg_l = 0
[[kernel]] void nh_moe_ah(const device float* xn [[buffer(0)]], const device float* rw [[buffer(1)]],
                          const device uchar* uq [[buffer(4)]], const device uchar* us [[buffer(5)]],
                          device float* logits [[buffer(6)]], device float* sh [[buffer(8)]],
                          const device float* bias [[buffer(9)]], device int* eid [[buffer(10)]],
                          device float* ew [[buffer(11)]], device atomic_uint* counter [[buffer(12)]],
                          constant NhMoeDecP& p [[buffer(13)]], constant float& gs_up [[buffer(14)]],
                          uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint arrival;
  threadgroup half2 tlh[256];
  const uint tg = tgp.x;
  if (tg < p.tg_r) {  // router rows (F32)
    const uint row0 = (tg * nsg + sg) * 4;
    float acc[4][1] = {{0}, {0}, {0}, {0}};
    const device uchar* r8 = (const device uchar*)rw;
    if (row0 < p.E) nh_mv_dot<3, 4, 1>(r8, r8, xn, p.H, p.H, 1, row0, p.E, lane, nullptr, acc);
    nh_mv_store<4, 1, 0>(acc, logits, p.E, p.E, 1, 1.0f, row0, lane);
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
  // shared up rows (NVFP4, relu^2)
  nh_x4_lut_to_tg(tlh, tid, nsg * 32);
  nh_x4_mv_impl<0, 2>(uq, us, xn, sh, p.S, p.H, gs_up, ((tg - p.tg_r) * nsg + sg) * 4, lane, tlh);
}

// x[n] += sum_j part[j, n] + sd[n] (one token). grid: ceil(H / 256) x 256
[[kernel]] void nh_moe_dh(const device float* part [[buffer(0)]], const device float* sd [[buffer(1)]],
                          device float* x [[buffer(2)]], constant NhMoeDecP& p [[buffer(3)]],
                          uint n [[thread_position_in_grid]]) {
  if (n >= p.H) return;
  float v = 0.0f;
  for (uint j = 0; j < p.k; ++j) v += part[size_t(j) * p.H + n];
  x[n] += v + sd[n];
}

// ---- F32 matrix (Lightning's router [128][2688]) x up to 8 rows (verify): one threadgroup per output row, its 256
// threads over K in float4 steps, fp32 sums reduced over the threadgroup (nh_mv_f32_r8 ran 128 simdgroups in all:
// 31 us per layer). A step's 8 input loads are issued before its products (rows past R load row R - 1, unused): 23
// layers' logits in one command buffer 6.9 -> 4.9 us per layer (M5 Max), the same sums in the same order.
// NhMvParams [[4]] (x_stride / y_stride in floats; 1 <= R <= 8). grid: N threadgroups x 256 threads ----
[[kernel]] void nh_mv_f32_sk(const device float* W [[buffer(0)]], const device float* x [[buffer(2)]],
                             device float* y [[buffer(3)]], constant NhMvParams& p [[buffer(4)]],
                             uint n [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                             uint ntg [[threads_per_threadgroup]]) {
  threadgroup float red[8][8];
  nh_mv_f32_rows(W, x, y, p.K, p.x_stride, p.y_stride, p.gs, p.R, n, tid, sg, lane, ntg, red);
}
