// ---- Fused single-token (decode) MoE layer: 4 dispatches instead of 9 ----
//   A: router (BF16) | latent down (Q8) | shared up (IQ2, relu^2), row segments of one grid over the normed hidden
//      state; the last router threadgroup to finish routes (sigmoid + selection bias, top-k, normalized weights).
//   B: routed up (IQ2, relu^2) of the k selected experts | shared down (IQ2) into sd.
//   C: routed down (nh_moe2_dn_iq2: weighted partials per selected expert).
//   D: latent up (Q8) of lo = sum of C's partials, + sd, added into the residual.
// Same per-row arithmetic as the separate kernels (nh_mv_dot / nh_mv1_iq2v / nh_moe_route1).

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

// One simdgroup argmax top-k over E <= 512 (16 per lane), as nh_moe_route1.
NI_INLINE void nh_route_sg(const device float* logits, const device float* bias, constant NhMoeDecP& p, uint lane,
                           device int* eid, device float* ew) {
  constexpr uint V = 16;
  float prob[V], sel[V];
  for (uint j = 0; j < V; ++j) {
    const uint e = lane + 32 * j;
    const bool live = e < p.E;
    prob[j] = live ? 1.0f / (1.0f + metal::precise::exp(-logits[e])) : 0.0f;
    sel[j] = live ? prob[j] + bias[e] : -INFINITY;
  }
  float sum = 1e-20f, mine = 0.0f;
  uint mine_i = 0;
  for (uint k = 0; k < p.k; ++k) {
    float v = -INFINITY;
    uint i = 0xffffffffu;
    for (uint j = 0; j < V; ++j)
      if (sel[j] > v) { v = sel[j]; i = lane + 32 * j; }
    for (uint off = 16; off > 0; off >>= 1) {
      const float ov = simd_shuffle_xor(v, off);
      const uint oi = simd_shuffle_xor(i, off);
      if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    float w = 0.0f;
    if (i % 32 == lane) {
      w = prob[i / 32];
      sel[i / 32] = -INFINITY;
    }
    w = simd_shuffle(w, ushort(i % 32));
    sum += w;
    if (lane == k) { mine = w; mine_i = i; }
  }
  if (lane < p.k) {
    eid[lane] = int(mine_i);
    ew[lane] = (p.norm ? mine / sum : mine) * p.scale;
  }
}

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
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      arrival = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (arrival != p.tg_r - 1) return;
    if (sg == 0) nh_route_sg((const device float*)logits, bias, p, lane, eid, ew);
    if (tid == 0) atomic_store_explicit(counter, 0u, memory_order_relaxed);
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
  nh_mv1_iq2v<4, 0, 2>(uq, us, xn, sh, p.S, p.H, 1.0f, ((tg - p.tg_r - p.tg_l) * nsg + sg) * 4, lane, nullptr);
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
    nh_mv1_iq2v<4, 0, 2>(eq + size_t(e) * p.up_q_stride, es + size_t(e) * p.up_s_stride, lat, h + size_t(j) * p.I, p.I,
                         p.L, 1.0f, ((tg % per) * nsg + sg) * 4, lane, nullptr);
    return;
  }
  // shared down rows (IQ2) into sd
  nh_mv1_iq2v<4, 0, 0>(dq, ds, sh, sd, p.H, p.S, 1.0f, ((tg - p.tg_u) * nsg + sg) * 4, lane, nullptr);
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

// ---- Mamba out_proj (FP4) of one token with the gated group RMSNorm applied on load: x[n] += gs * W[n, :] . yn,
// yn = y * rsqrt(mean_group(y^2) + eps) * w (groups of G channels). Each threadgroup computes the group scales
// itself (the input is 32 KB). grid: ceil(N / 16) threadgroups x 128 threads (4 simdgroups x 4 rows). ----
struct NhOut1P {
  uint N, K, G;
  float eps, gs;
};
[[kernel]] void nh_out1_fp4(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                            const device float* y [[buffer(2)]], const device float* w [[buffer(3)]],
                            device float* x [[buffer(4)]], constant NhOut1P& p [[buffer(5)]],
                            uint3 tgp [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                            uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float part[128];
  threadgroup float inv[16];
  const uint ng = p.K / p.G, per = p.K / 128;  // elements per thread (one group's slice)
  {
    float ss = 0.0f;
    const device float4* yp = (const device float4*)(y + tid * per);
    for (uint i = 0; i < per / 4; ++i) {
      const float4 v = yp[i];
      ss += dot(v, v);
    }
    part[tid] = ss;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < ng) {
    const uint tpg = 128 / ng;  // threads per group
    float ss = 0.0f;
    for (uint t = 0; t < tpg; ++t) ss += part[tid * tpg + t];
    inv[tid] = 1.0f / metal::precise::sqrt(ss / float(p.G) + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint row0 = (tgp.x * nsg + sg) * 4;
  float acc[4] = {0, 0, 0, 0};
  if (row0 < p.N) {
    uint rows[4];
    for (int j = 0; j < 4; ++j) rows[j] = min(row0 + j, p.N - 1);
    for (uint k0 = lane * 32; k0 < p.K; k0 += 32 * 32) {
      const float gi = inv[k0 / p.G];
      float xv[32];
      const device float4* yp = (const device float4*)(y + k0);
      const device float4* wp = (const device float4*)(w + k0);
      for (int i = 0; i < 8; ++i) {
        const float4 v = yp[i] * wp[i] * gi;
        xv[4 * i] = v.x; xv[4 * i + 1] = v.y; xv[4 * i + 2] = v.z; xv[4 * i + 3] = v.w;
      }
      for (int j = 0; j < 4; ++j) {
        const size_t n = rows[j];
        const uint4 c = *(const device uint4*)(q + n * (p.K / 2) + k0 / 2);
        const uint sb = *(const device ushort*)(s + n * (p.K / 16) + k0 / 16);
        // sub-block 0: bytes c.x, c.y (elements b, b + 8 of byte b), sub-block 1: c.z, c.w
        const uint wv[4] = {c.x, c.y, c.z, c.w};
        float d0 = 0.0f, d1 = 0.0f;
        for (int wi = 0; wi < 4; ++wi)
          for (int b = 0; b < 4; ++b) {
            const uint byte = (wv[wi] >> (8 * b)) & 0xFFu;
            const uint i = (wi & 1) * 4 + b, base = (wi >> 1) * 16;
            const float t = nh_fp4_lut[byte & 15u] * xv[base + i] + nh_fp4_lut[byte >> 4] * xv[base + i + 8];
            if (wi < 2) d0 += t;
            else d1 += t;
          }
        acc[j] += d0 * nh_ue4m3(sb & 0xFFu) + d1 * nh_ue4m3(sb >> 8);
      }
    }
  }
  for (int j = 0; j < 4; ++j) {
    const float v = simd_sum(acc[j]) * p.gs;
    const uint n = row0 + j;
    if (lane == 0 && n < p.N) x[n] += v;
  }
}
