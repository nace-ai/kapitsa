// Dory (NVIDIA Nemotron-Labs Dory1.0, DoryForCausalLM): the normalized-GPT pieces between the matrix products,
// which run on the Nemotron-H weight kernels (src/kernels/nemotron/: BF16, NVFP4, Q8 matvecs and GEMMs).
//
// The math follows NVIDIA's implementation (vLLM PR #60149, dory.py): justnorm(x) = x * rsqrt(sum x^2 + 1e-6), an L2
// normalization (not RMS), in fp32; a residual is the slerp-like step h = justnorm(justnorm(h) + alpha (justnorm(y) -
// justnorm(h))). Activations are fp32 rows; the prefill GEMMs read half copies of their inputs.
//
// Attention (head dim 256, GQA): rope (neox halves) then justnorm of q and k per head, q times sqk^2; scores scaled by
// 1/16; an output gate sigmoid(g * s_gate^2) and a justnorm over all heads (4096) before o_proj. Caches are half
// [Hkv][rows][256]: a full-attention layer's holds positions 0 .. rows - 1 (slot = position), a sliding-window layer's
// is a ring of `rows` (>= window + 8) positions (slot = position % rows).

#define DORY_EPS 1e-6f
constant constexpr uint kDD = 256;   // head dim
constant constexpr uint kDH = 128;   // half of it (the rope pairs)

// ---- q / k / v of R rows: rope, justnorm, q x sqk^2; k and v (half) into a cache at their slots ----
// The rows come from the attention's joint projection (q | k | v | g columns of one matrix, run with tensor scale 1):
// each part is scaled by its own NVFP4 tensor scale here (1 for BF16).
struct DoryPrepP {
  uint R, pos0;      // rows, position of row 0
  uint Hq, Hkv;
  uint rows;         // the destination's rows (ring size, linear cap, or a prefill chunk's T)
  uint ring;         // 1: slot = position % rows; 0: slot = position - base
  uint base;         // (linear) position of slot 0: 0 for a cache, pos0 for a prefill chunk's own K / V buffer
  uint in_ld;        // floats per input row
  uint q_off, k_off, v_off;  // columns of q, k, v in an input row
  float sq, sk, sv;  // the parts' tensor scales
};
// grid: (Hq + 2 Hkv, R) x 128 threads; thread t holds elements t and t + 128 of its head (a rope pair).
// q out: fp32 [R][Hq x 256]; inv_freq: fp32 [128], 1 / theta^(2i / 256) of the layer's rope.
[[kernel]] void dory_prep(const device float* in [[buffer(0)]], device float* qo [[buffer(1)]],
                          const device float* sqk2 [[buffer(3)]], const device float* inv_freq [[buffer(4)]],
                          device half* kc [[buffer(5)]], device half* vc [[buffer(6)]],
                          constant DoryPrepP& p [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint t [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  const uint head = tg.x, r = tg.y;
  const uint pos = p.pos0 + r;
  const uint slot = p.ring ? pos % p.rows : pos - p.base;
  const device float* row = in + size_t(r) * p.in_ld;
  if (head >= p.Hq + p.Hkv) {  // v: copied
    const uint vh = head - p.Hq - p.Hkv;
    const device float* src = row + p.v_off + vh * kDD;
    device half* dst = vc + (size_t(vh) * p.rows + slot) * kDD;
    dst[t] = half(src[t] * p.sv);
    dst[t + kDH] = half(src[t + kDH] * p.sv);
    return;
  }
  const bool isq = head < p.Hq;
  const uint hh = isq ? head : head - p.Hq;
  const device float* src = row + (isq ? p.q_off : p.k_off) + hh * kDD;
  const float sc = isq ? p.sq : p.sk;
  const float x1 = src[t] * sc, x2 = src[t + kDH] * sc;
  const float ang = float(pos) * inv_freq[t];
  const float c = metal::precise::cos(ang), s = metal::precise::sin(ang);
  const float y1 = x1 * c - x2 * s, y2 = x2 * c + x1 * s;
  const float ss = tg_sum(y1 * y1 + y2 * y2, scratch, t, kDH);
  const float inv = metal::precise::rsqrt(ss + DORY_EPS);
  if (isq) {
    device float* dst = qo + (size_t(r) * p.Hq + hh) * kDD;
    dst[t] = sqk2[hh * kDD + t] * (y1 * inv);
    dst[t + kDH] = sqk2[hh * kDD + t + kDH] * (y2 * inv);
  } else {
    device half* dst = kc + (size_t(hh) * p.rows + slot) * kDD;
    dst[t] = half(y1 * inv);
    dst[t + kDH] = half(y2 * inv);
  }
}

// ---- decode / verify attention of R <= 8 rows over a cache (the rows' own keys written first) ----
// Row r (position pos0 + r) sees keys [lo_r, pos0 + r], lo_r = window ? max(0, pos0 + r + 1 - window) : 0. Split
// over chunks of keys: one threadgroup per (chunk, kv head, row), its G = Hq / Hkv simdgroups the query heads of
// the kv head; K / V tiles of 16 keys staged once for them; lanes hold 8 dims. Partial (o, max, sum) per chunk.
// grid: (nchunks, Hkv, R) x 32 G threads; chunk c covers keys [k0 + c chunk, k0 + (c + 1) chunk), k0 = lo_0.
struct DoryAttnP {
  uint R, pos0, Hq, Hkv;
  uint rows, ring, window;
  uint chunk, nchunks, k0;
  float scale;
};
constant constexpr uint kDoryKT = 16;

[[kernel]] void dory_attn_partial(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                  const device half* vc [[buffer(2)]], device float* part_o [[buffer(3)]],
                                  device float* part_ml [[buffer(4)]], constant DoryAttnP& p [[buffer(5)]],
                                  uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                  uint3 ntg3 [[threads_per_threadgroup]]) {
  const uint ntg = ntg3.x;
  threadgroup half4 kt[kDoryKT * kDD / 4];
  threadgroup half4 vt[kDoryKT * kDD / 4];
  const uint chunk = tg.x, kvh = tg.y, r = tg.z;
  const uint G = p.Hq / p.Hkv, hq = kvh * G + sg;
  const uint qp = p.pos0 + r;
  const uint lo = p.window && qp + 1 > p.window ? qp + 1 - p.window : 0;
  const uint c0 = p.k0 + chunk * p.chunk;
  const uint start = max(c0, lo), end = min(c0 + p.chunk, qp + 1);
  const device float4* qr = (const device float4*)(q + (size_t(r) * p.Hq + hq) * kDD) + lane * 2;
  const float4 q0 = qr[0] * p.scale, q1 = qr[1] * p.scale;
  float4 a0 = 0.0f, a1 = 0.0f;
  float m = -INFINITY, l = 0.0f;
  const device half4* kb = (const device half4*)(kc + size_t(kvh) * p.rows * kDD);
  const device half4* vb = (const device half4*)(vc + size_t(kvh) * p.rows * kDD);
  // (every threadgroup of the dispatch runs the same tile loop shape: no early return before the barriers)
  for (uint t0 = start; t0 < end; t0 += kDoryKT) {
    const uint n = min(kDoryKT, end - t0);
    for (uint i = tid; i < n * kDD / 4; i += ntg) {
      const uint key = t0 + i / (kDD / 4), slot = p.ring ? key % p.rows : key;
      kt[i] = kb[size_t(slot) * (kDD / 4) + i % (kDD / 4)];
      vt[i] = vb[size_t(slot) * (kDD / 4) + i % (kDD / 4)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < n; ++j) {
      const float s = simd_sum(dot(q0, float4(kt[j * kDD / 4 + lane * 2])) + dot(q1, float4(kt[j * kDD / 4 + lane * 2 + 1])));
      const float mn = max(m, s);
      const float corr = metal::precise::exp(m - mn), e = metal::precise::exp(s - mn);
      l = l * corr + e;
      a0 = a0 * corr + e * float4(vt[j * kDD / 4 + lane * 2]);
      a1 = a1 * corr + e * float4(vt[j * kDD / 4 + lane * 2 + 1]);
      m = mn;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const size_t base = (size_t(r) * p.Hq + hq) * p.nchunks + chunk;
  device float4* po = (device float4*)(part_o + base * kDD) + lane * 2;
  po[0] = a0;
  po[1] = a1;
  if (lane == 0) {
    part_ml[base * 2 + 0] = m;
    part_ml[base * 2 + 1] = l;
  }
}

// grid: (Hq, R) x 256 threads. out[r][h * 256 + d] (fp32, not yet gated)
[[kernel]] void dory_attn_reduce(const device float* part_o [[buffer(0)]], const device float* part_ml [[buffer(1)]],
                                 device float* out [[buffer(2)]], constant DoryAttnP& p [[buffer(3)]],
                                 uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
  const uint h = tg.x, r = tg.y;
  const size_t base = (size_t(r) * p.Hq + h) * p.nchunks;
  float M = -INFINITY;
  for (uint c = 0; c < p.nchunks; ++c) M = max(M, part_ml[(base + c) * 2]);
  float num = 0.0f, den = 0.0f;
  for (uint c = 0; c < p.nchunks; ++c) {
    const float mc = part_ml[(base + c) * 2];
    if (mc == -INFINITY) continue;
    const float w = metal::precise::exp(mc - M);
    num += w * part_o[(base + c) * kDD + d];
    den += w * part_ml[(base + c) * 2 + 1];
  }
  out[(size_t(r) * p.Hq + h) * kDD + d] = num / den;
}

// The decode attention's partials combined and gated (dory_attn_reduce, then the gate of dory_gate_norm) in one pass,
// without the norm over all heads: o_proj is linear and the residual normalizes its output (dory_slerp's
// justnorm(y)), so that norm's scale cancels. out[r] = o * sigmoid(g * sg * s_gate^2), fp32 [R][Hq x 256].
// grid: (Hq, R) x 256 (a thread per head dim)
struct DoryRedGateP {
  uint Hq, nchunks, g_ld;
  float sg;
};
[[kernel]] void dory_attn_reduce_gate(const device float* part_o [[buffer(0)]], const device float* part_ml [[buffer(1)]],
                                      const device float* g [[buffer(2)]], const device float* sg2 [[buffer(3)]],
                                      device float* out [[buffer(4)]], constant DoryRedGateP& p [[buffer(5)]],
                                      uint2 tg [[threadgroup_position_in_grid]], uint2 tp [[thread_position_in_threadgroup]]) {
  const uint h = tg.x, r = tg.y, d = tp.x;
  const size_t base = (size_t(r) * p.Hq + h) * p.nchunks;
  float M = -INFINITY;
  for (uint c = 0; c < p.nchunks; ++c) M = max(M, part_ml[(base + c) * 2]);
  float num = 0.0f, den = 0.0f;
  for (uint c = 0; c < p.nchunks; ++c) {
    const float mc = part_ml[(base + c) * 2];
    if (mc == -INFINITY) continue;
    const float w = metal::precise::exp(mc - M);
    num += w * part_o[(base + c) * kDD + d];
    den += w * part_ml[(base + c) * 2 + 1];
  }
  const uint i = h * kDD + d;
  const float gv = (g[size_t(r) * p.g_ld + i] * p.sg) * sg2[i];
  out[size_t(r) * p.Hq * kDD + i] = (num / den) * (1.0f / (1.0f + metal::precise::exp(-gv)));
}

// ---- row kernels ----
struct DoryRowP {
  uint D;
  uint x_ld, y_ld;  // floats (or halves) per row
  uint out_half;    // 1: y is half
  uint g_ld;        // floats per row of the second input (the gate rows / the MLP's up rows)
  float s1, s2;     // tensor scales of the first and second input (NVFP4 parts; 1 for BF16)
};

// Attention output gate and norm: y = justnorm(a * sigmoid(g * sg2)) over D (all heads). grid: R x 1024
[[kernel]] void dory_gate_norm(const device float* a [[buffer(0)]], const device float* g [[buffer(1)]],
                               const device float* sg2 [[buffer(2)]], device void* y [[buffer(3)]],
                               constant DoryRowP& p [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const device float* ar = a + size_t(r) * p.x_ld;
  const device float* gr = g + size_t(r) * p.g_ld;
  float ss = 0.0f;
  for (uint i = tid; i < p.D; i += ntg) {
    const float v = ar[i] * (1.0f / (1.0f + metal::precise::exp(-((gr[i] * p.s2) * sg2[i]))));
    ss += v * v;
  }
  ss = tg_sum(ss, scratch, tid, ntg);
  const float inv = metal::precise::rsqrt(ss + DORY_EPS);
  for (uint i = tid; i < p.D; i += ntg) {
    const float v = ar[i] * (1.0f / (1.0f + metal::precise::exp(-((gr[i] * p.s2) * sg2[i])))) * inv;
    if (p.out_half) ((device half*)y)[size_t(r) * p.y_ld + i] = half(v);
    else ((device float*)y)[size_t(r) * p.y_ld + i] = v;
  }
}

// The MLP's activation for decode rows (fp32), without the norm over the FFN width: down_proj is linear and the
// residual normalizes its output, so that norm's scale cancels. y = silu(g s1 sg) (u s2 su). grid: (ceil(D / 256), R)
// x 256
[[kernel]] void dory_swiglu(const device float* g [[buffer(0)]], const device float* u [[buffer(1)]],
                            const device float* sg [[buffer(2)]], const device float* su [[buffer(3)]],
                            device float* y [[buffer(4)]], constant DoryRowP& p [[buffer(5)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint2 tp [[thread_position_in_threadgroup]]) {
  const uint i = tg.x * 256 + tp.x, r = tg.y;
  if (i >= p.D) return;
  const float a = (g[size_t(r) * p.x_ld + i] * p.s1) * sg[i], b = (u[size_t(r) * p.g_ld + i] * p.s2) * su[i];
  ((device float*)y)[size_t(r) * p.y_ld + i] = a / (1.0f + metal::precise::exp(-a)) * b;
}

// MLP activation and norm: y = justnorm(silu(g * sg) * (u * su)) over D (the FFN width). grid: R x 1024
[[kernel]] void dory_swiglu_norm(const device float* g [[buffer(0)]], const device float* u [[buffer(1)]],
                                 const device float* sg [[buffer(2)]], const device float* su [[buffer(3)]],
                                 device void* y [[buffer(4)]], constant DoryRowP& p [[buffer(5)]],
                                 uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                                 uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const device float* gr = g + size_t(r) * p.x_ld;
  const device float* ur = u + size_t(r) * p.g_ld;
  float ss = 0.0f;
  for (uint i = tid; i < p.D; i += ntg) {
    const float a = (gr[i] * p.s1) * sg[i], b = (ur[i] * p.s2) * su[i];
    const float v = a / (1.0f + metal::precise::exp(-a)) * b;
    ss += v * v;
  }
  ss = tg_sum(ss, scratch, tid, ntg);
  const float inv = metal::precise::rsqrt(ss + DORY_EPS);
  for (uint i = tid; i < p.D; i += ntg) {
    const float a = (gr[i] * p.s1) * sg[i], b = (ur[i] * p.s2) * su[i];
    const float v = a / (1.0f + metal::precise::exp(-a)) * b * inv;
    if (p.out_half) ((device half*)y)[size_t(r) * p.y_ld + i] = half(v);
    else ((device float*)y)[size_t(r) * p.y_ld + i] = v;
  }
}

// The residual step: h = justnorm(a + alpha (b - a)), a = justnorm(h), b = justnorm(y); then, by p.mode,
//   1: h = justnorm(h)                   (the end of a loop: state = justnorm(h))
//   2: h = justnorm(justnorm(h) + inp)   (the end of a loop with another to follow: its input justnorm(state + inp))
// With xh: a half copy of the result (the next GEMM's input). grid: R x 1024 (D <= 1024 * 4)
struct DorySlerpP {
  uint D, mode, write_half;
};
[[kernel]] void dory_slerp(device float* h [[buffer(0)]], const device float* y [[buffer(1)]],
                           const device float* alpha [[buffer(2)]], const device float* inp [[buffer(3)]],
                           device half* xh [[buffer(4)]], constant DorySlerpP& p [[buffer(5)]],
                           uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                           uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  device float* hr = h + size_t(r) * p.D;
  const device float* yr = y + size_t(r) * p.D;
  float hv[4], yv[4];
  float sa = 0.0f, sb = 0.0f;
  for (uint j = 0; j < 4; ++j) {
    const uint i = tid + j * ntg;
    hv[j] = i < p.D ? hr[i] : 0.0f;
    yv[j] = i < p.D ? yr[i] : 0.0f;
    sa += hv[j] * hv[j];
    sb += yv[j] * yv[j];
  }
  sa = tg_sum(sa, scratch, tid, ntg);
  sb = tg_sum(sb, scratch, tid, ntg);
  const float ia = metal::precise::rsqrt(sa + DORY_EPS), ib = metal::precise::rsqrt(sb + DORY_EPS);
  float sc = 0.0f;
  for (uint j = 0; j < 4; ++j) {
    const uint i = tid + j * ntg;
    const float a = hv[j] * ia, b = yv[j] * ib;
    hv[j] = i < p.D ? a + alpha[i] * (b - a) : 0.0f;
    sc += hv[j] * hv[j];
  }
  sc = tg_sum(sc, scratch, tid, ntg);
  float ic = metal::precise::rsqrt(sc + DORY_EPS);
  for (uint j = 0; j < 4; ++j) hv[j] *= ic;
  if (p.mode >= 1) {  // state = justnorm(h)
    float s2 = 0.0f;
    for (uint j = 0; j < 4; ++j) s2 += hv[j] * hv[j];
    s2 = tg_sum(s2, scratch, tid, ntg);
    ic = metal::precise::rsqrt(s2 + DORY_EPS);
    for (uint j = 0; j < 4; ++j) hv[j] *= ic;
  }
  if (p.mode == 2) {  // the next loop's input: justnorm(state + inp)
    const device float* ir = inp + size_t(r) * p.D;
    float s3 = 0.0f;
    for (uint j = 0; j < 4; ++j) {
      const uint i = tid + j * ntg;
      hv[j] = i < p.D ? hv[j] + ir[i] : 0.0f;
      s3 += hv[j] * hv[j];
    }
    s3 = tg_sum(s3, scratch, tid, ntg);
    ic = metal::precise::rsqrt(s3 + DORY_EPS);
    for (uint j = 0; j < 4; ++j) hv[j] *= ic;
  }
  for (uint j = 0; j < 4; ++j) {
    const uint i = tid + j * ntg;
    if (i < p.D) {
      hr[i] = hv[j];
      if (p.write_half) xh[size_t(r) * p.D + i] = half(hv[j]);
    }
  }
}

// The first loop's input: h = justnorm(h + h) (state = input), inp = the input rows; with xh a half copy.
// grid: R x 1024
[[kernel]] void dory_loop_start(device float* h [[buffer(0)]], device float* inp [[buffer(1)]],
                                device half* xh [[buffer(2)]], constant DorySlerpP& p [[buffer(3)]],
                                uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                                uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  device float* hr = h + size_t(r) * p.D;
  float hv[4];
  float s = 0.0f;
  for (uint j = 0; j < 4; ++j) {
    const uint i = tid + j * ntg;
    const float x = i < p.D ? hr[i] : 0.0f;
    if (i < p.D) inp[size_t(r) * p.D + i] = x;
    hv[j] = x + x;
    s += hv[j] * hv[j];
  }
  s = tg_sum(s, scratch, tid, ntg);
  const float ic = metal::precise::rsqrt(s + DORY_EPS);
  for (uint j = 0; j < 4; ++j) {
    const uint i = tid + j * ntg;
    if (i < p.D) {
      hr[i] = hv[j] * ic;
      if (p.write_half) xh[size_t(r) * p.D + i] = half(hv[j] * ic);
    }
  }
}

// Self-speculation's verify: the loop after a drafted loop 0, from the draft's rows: h = justnorm(state + inp) (the
// next loop's input, as dory_slerp's mode 2 makes it) and inp = the input rows. grid: R x 1024
[[kernel]] void dory_loop_resume(const device float* state [[buffer(0)]], const device float* inp0 [[buffer(1)]],
                                 device float* h [[buffer(2)]], device float* inp [[buffer(3)]],
                                 constant DorySlerpP& p [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                                 uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  float hv[4];
  float s = 0.0f;
  for (uint j = 0; j < 4; ++j) {
    const uint i = tid + j * ntg;
    const float a = i < p.D ? state[size_t(r) * p.D + i] : 0.0f, b = i < p.D ? inp0[size_t(r) * p.D + i] : 0.0f;
    if (i < p.D) inp[size_t(r) * p.D + i] = b;
    hv[j] = i < p.D ? a + b : 0.0f;
    s += hv[j] * hv[j];
  }
  s = tg_sum(s, scratch, tid, ntg);
  const float ic = metal::precise::rsqrt(s + DORY_EPS);
  for (uint j = 0; j < 4; ++j) {
    const uint i = tid + j * ntg;
    if (i < p.D) h[size_t(r) * p.D + i] = hv[j] * ic;
  }
}

// rows -> half (the first layer's GEMM input: the raw embedding rows). grid: R x 256
[[kernel]] void dory_rows_half(const device float* x [[buffer(0)]], device half* y [[buffer(1)]],
                               constant uint& D [[buffer(2)]], uint r [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  for (uint i = tid; i < D; i += ntg) y[size_t(r) * D + i] = half(x[size_t(r) * D + i]);
}

// logits[r][v] *= scale[v] (lm_head.logit_scale / initializer_range). grid: (ceil(V / 256), R) x 256
struct DoryScaleP {
  uint V, ld;
};
[[kernel]] void dory_logit_scale(device float* lg [[buffer(0)]], const device float* sc [[buffer(1)]],
                                 constant DoryScaleP& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x < p.V) lg[size_t(gid.y) * p.ld + gid.x] *= sc[gid.x];
}

// A prefill chunk's own K / V ([Hkv][T][256]) into a cache at their slots: rows [r0, T) (a ring keeps the last
// `rows` positions). grid: (T - r0, 2 Hkv) x 64 threads (16-byte words)
struct DoryStoreP {
  uint T, r0, pos0, Hkv, rows, ring;
};
[[kernel]] void dory_kv_store(const device uint4* ck [[buffer(0)]], const device uint4* cv [[buffer(1)]],
                              device uint4* kc [[buffer(2)]], device uint4* vc [[buffer(3)]],
                              constant DoryStoreP& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                              uint w [[thread_index_in_threadgroup]]) {
  const uint t = p.r0 + tg.x, hh = tg.y % p.Hkv;
  const bool isv = tg.y >= p.Hkv;
  const uint pos = p.pos0 + t, slot = p.ring ? pos % p.rows : pos;
  const device uint4* src = (isv ? cv : ck) + (size_t(hh) * p.T + t) * (kDD / 8);
  device uint4* dst = (isv ? vc : kc) + (size_t(hh) * p.rows + slot) * (kDD / 8);
  if (w < kDD / 8) dst[w] = src[w];
}

// A cache's ring (or its first `n` rows) copied: checkpoints of the sliding-window layers. grid: ceil(n / 256) x 256
[[kernel]] void dory_copy4(const device uint4* src [[buffer(0)]], device uint4* dst [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) dst[i] = src[i];
}
