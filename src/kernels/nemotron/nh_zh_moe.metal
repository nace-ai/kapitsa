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
