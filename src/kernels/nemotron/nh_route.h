// Shared by the Nemotron-H library and the tensor library (the verify's fused router pass, nh_v8t.metal).

// Sigmoid router selection of one row by one simdgroup (E <= 32 V: V experts per lane in registers): k argmax rounds
// over prob + bias (lowest index on ties), weights = the selected probs (normalized by their sum + 1e-20 if norm),
// times scale. The per-lane arrays are only indexed with constants (a dynamic index would move them to memory). Any V
// with E <= 32 V selects the same experts with the same weights (the extra slots are -inf, never selected): V 4 for
// E <= 128 (Lightning) takes a verify's routing from 8.1 to 4.8 us per layer (M5 Max, 23 layers in one command buffer)
// (returns lane j < K's selected expert, eid[j])
template <uint V>
NI_INLINE uint nh_route_row_v(const device float* lg, const device float* bias, uint E, uint K, uint norm, float scale,
                              uint lane, device int* eid, device float* ew) {
  float prob[V], sel[V];
  for (uint j = 0; j < V; ++j) {
    const uint e = lane + 32 * j;
    const bool live = e < E;
    prob[j] = live ? 1.0f / (1.0f + metal::precise::exp(-lg[e])) : 0.0f;
    sel[j] = live ? prob[j] + bias[e] : -INFINITY;
  }
  float sum = 1e-20f, mine = 0.0f;
  uint mine_i = 0;
  for (uint k = 0; k < K; ++k) {
    float v = -INFINITY;
    uint i = 0xffffffffu;
    for (uint j = 0; j < V; ++j)
      if (sel[j] > v) { v = sel[j]; i = lane + 32 * j; }  // ascending index: the lowest wins ties
    for (uint off = 16; off > 0; off >>= 1) {
      const float ov = simd_shuffle_xor(v, off);
      const uint oi = simd_shuffle_xor(i, off);
      if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    float w = 0.0f;
    const bool own = i % 32 == lane;
    const uint jj = i / 32;
    for (uint j = 0; j < V; ++j)
      if (own && j == jj) {
        w = prob[j];
        sel[j] = -INFINITY;
      }
    w = simd_shuffle(w, ushort(i % 32));
    sum += w;
    if (lane == k) { mine = w; mine_i = i; }
  }
  if (lane < K) {
    eid[lane] = int(mine_i);
    ew[lane] = (norm ? mine / sum : mine) * scale;
  }
  return mine_i;
}

NI_INLINE uint nh_route_row(const device float* lg, const device float* bias, uint E, uint K, uint norm, float scale,
                            uint lane, device int* eid, device float* ew) {
  if (E <= 128) return nh_route_row_v<4>(lg, bias, E, K, norm, scale, lane, eid, ew);
  return nh_route_row_v<16>(lg, bias, E, K, norm, scale, lane, eid, ew);
}

// One output row n of an F32 matrix (Lightning's router [128][2688]) for up to 8 rows: the threadgroup's threads over
// K in float4 steps, fp32 sums reduced over the threadgroup (nh_mv_f32_sk). A step's 8 input loads are issued before
// its products (rows past R load row R - 1, unused).
NI_INLINE void nh_mv_f32_rows(const device float* W, const device float* x, device float* y, uint K, uint x_stride,
                              uint y_stride, float gs, uint R, uint n, uint tid, uint sg, uint lane, uint ntg,
                              threadgroup float (*red)[8]) {
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const device float4* w4 = (const device float4*)(W + size_t(n) * K);
  for (uint c = tid; c < K / 4; c += ntg) {
    const float4 w = w4[c];
    float4 xv[8];
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < 8; ++r) xv[r] = ((const device float4*)(x + size_t(min(r, R - 1)) * x_stride))[c];
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < 8; ++r)
      if (r < R) acc[r] += dot(w, xv[r]);
  }
  for (uint r = 0; r < 8; ++r) {
    const float v = simd_sum(acc[r]);
    if (lane == 0) red[sg][r] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < R) {
    float v = 0.0f;
    for (uint j = 0; j < ntg / 32; ++j) v += red[j][tid];
    y[size_t(tid) * y_stride + n] = v * gs;
  }
}
