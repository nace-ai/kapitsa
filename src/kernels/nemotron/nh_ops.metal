// Nemotron-H (GGUF) kernels: embedding, norms, the Mamba-2 mixer (conv1d + selective scan + gate), attention
// for decode / verify rows, MoE routing.

// ---- embedding (Q8 table) ----
// out[r, :] = embedding row of tokens[tok_off + r]. grid: R threadgroups x 256 threads.
struct NhEmbedParams {
  uint D, tok_off;
};
[[kernel]] void nh_embed_q8(const device char* q [[buffer(0)]], const device half* s [[buffer(1)]],
                            const device int* tokens [[buffer(2)]], device float* out [[buffer(3)]],
                            constant NhEmbedParams& p [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  const size_t id = size_t(tokens[p.tok_off + r]);
  for (uint k = tid; k < p.D; k += ntg) out[size_t(r) * p.D + k] = float(q[id * p.D + k]) * float(s[id * (p.D / 32) + k / 32]);
}

[[kernel]] void nh_embed_bf16(const device ushort* w [[buffer(0)]], const device int* tokens [[buffer(2)]],
                              device float* out [[buffer(3)]], constant NhEmbedParams& p [[buffer(4)]],
                              uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                              uint ntg [[threads_per_threadgroup]]) {
  const size_t id = size_t(tokens[p.tok_off + r]);
  for (uint k = tid; k < p.D; k += ntg) out[size_t(r) * p.D + k] = as_type<float>(uint(w[id * p.D + k]) << 16);
}

// ---- RMSNorm: y = x / sqrt(mean(x^2) + eps) * w. grid: R threadgroups x 1024 threads ----
struct NhNormParams {
  uint D;
  float eps;
  uint x_stride, y_stride;
};
[[kernel]] void nh_rmsnorm(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                           device float* y [[buffer(2)]], constant NhNormParams& p [[buffer(3)]],
                           uint r [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                           uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const device float* xr = x + size_t(r) * p.x_stride;
  float ss = 0.0f;
  for (uint k = tid; k < p.D; k += ntg) ss += xr[k] * xr[k];
  ss = tg_sum(ss, scratch, tid, ntg);
  const float inv = 1.0f / metal::precise::sqrt(ss / float(p.D) + p.eps);
  for (uint k = tid; k < p.D; k += ntg) y[size_t(r) * p.y_stride + k] = xr[k] * inv * w[k];
}

// nh_rmsnorm that also writes the rows as half (yh, row stride D): the verify's MoE layers, whose router reads the fp32
// rows and whose expert GEMMs the half ones; each value as nh_rmsnorm + nh_rows_half write it, in one pass
[[kernel]] void nh_rmsnorm_h(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                             device float* y [[buffer(2)]], constant NhNormParams& p [[buffer(3)]],
                             device half* yh [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const device float* xr = x + size_t(r) * p.x_stride;
  float ss = 0.0f;
  for (uint k = tid; k < p.D; k += ntg) ss += xr[k] * xr[k];
  ss = tg_sum(ss, scratch, tid, ntg);
  const float inv = 1.0f / metal::precise::sqrt(ss / float(p.D) + p.eps);
  for (uint k = tid; k < p.D; k += ntg) {
    const float v = xr[k] * inv * w[k];
    y[size_t(r) * p.y_stride + k] = v;
    yh[size_t(r) * p.D + k] = half(v);
  }
}

// Grouped RMSNorm of the Mamba output: groups of G channels each normalized on their own, times w.
// grid: (R * ngroups) threadgroups x 256 threads
[[kernel]] void nh_gnorm(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                         device float* y [[buffer(2)]], constant NhGnormParams& p [[buffer(3)]],
                         uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                         uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const uint ng = p.D / p.G, r = tg / ng, g = tg % ng;
  const device float* xr = x + size_t(r) * p.D + g * p.G;
  float ss = 0.0f;
  for (uint k = tid; k < p.G; k += ntg) ss += xr[k] * xr[k];
  ss = tg_sum(ss, scratch, tid, ntg);
  const float inv = 1.0f / metal::precise::sqrt(ss / float(p.G) + p.eps);
  for (uint k = tid; k < p.G; k += ntg) y[size_t(r) * p.D + g * p.G + k] = xr[k] * inv * w[g * p.G + k];
}

// ---- Mamba-2 mixer: causal conv1d (k = 4) + SiLU, selective scan, D skip, SiLU(z) gate ----
// in_proj rows zx[t] = [ z (DI) | x (DI) | B (NG*DS) | C (NG*DS) | dt (NH) ], DI = NH * HD.
// The conv's input history lives in a ring of the last 4 raw xBC rows, slot = position & 3, so a pass never
// reads a slot it writes (the ring is written by nh_conv_ring after the scan).
// state: fp32 [NH][HD][DS]. With write = 0 (verify) the state is read but not written.
// grid: NH threadgroups x 256 threads (HD = 64 rows x 4 lanes of DS / 4 = 32 state columns each).
struct NhSsmParams {
  uint T, pos0, write;
  uint zx_stride;   // floats per zx row
  uint NH, HD, DS, NG;
};
constant constexpr uint kSsmNB = 8;  // time steps staged per barrier

// raw xBC input of channel c at row t of the pass (t < 0: the ring, positions before the pass)
NI_INLINE float nh_raw(const device float* zx, const device float* ring, constant NhSsmParams& p, uint DI, uint CH,
                       int t, uint c) {
  return t >= 0 ? zx[size_t(t) * p.zx_stride + DI + c] : ring[((p.pos0 + uint(t + 4)) & 3u) * CH + c];
}

NI_INLINE float nh_conv(const device float* zx, const device float* ring, const device float* conv_w,
                        const device float* conv_b, constant NhSsmParams& p, uint DI, uint CH, int t, uint c) {
  const float4 w = *(const device float4*)(conv_w + c * 4);
  const float v = conv_b[c] + w.x * nh_raw(zx, ring, p, DI, CH, t - 3, c) + w.y * nh_raw(zx, ring, p, DI, CH, t - 2, c) +
                  w.z * nh_raw(zx, ring, p, DI, CH, t - 1, c) + w.w * nh_raw(zx, ring, p, DI, CH, t, c);
  return silu(v);
}

[[kernel]] void nh_ssm(const device float* zx [[buffer(0)]], device float* ring [[buffer(1)]],
                       device float* state [[buffer(2)]], device float* y [[buffer(3)]],
                       const device float* conv_w [[buffer(4)]], const device float* conv_b [[buffer(5)]],
                       const device float* A [[buffer(6)]], const device float* Dsk [[buffer(7)]],
                       const device float* dt_bias [[buffer(8)]], constant NhSsmParams& p [[buffer(9)]],
                       uint h [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float sbc[kSsmNB][256];   // conv'd B (0..127) and C (128..255) of the head's group
  threadgroup float sx[kSsmNB][64];     // conv'd x of the head's 64 channels
  threadgroup float sdt[kSsmNB][2];     // softplus(dt), exp(dt * A)
  const uint DI = p.NH * p.HD, CH = DI + 2 * p.NG * p.DS;  // conv channels
  const uint g = h / (p.NH / p.NG);
  const uint row = tid >> 2, qq = tid & 3;
  // conv channel this thread stages: B / C of group g, and (tid < 64) x of head h
  const uint cbc = DI + (tid < 128 ? g * p.DS + tid : p.NG * p.DS + g * p.DS + (tid - 128));
  const uint cx = h * p.HD + (tid & 63);
  float st[32];
  device float* sp = state + (size_t(h) * p.HD + row) * p.DS + qq * 32;
  for (int i = 0; i < 8; ++i) {
    float4 v = ((device float4*)sp)[i];
    st[i * 4 + 0] = v.x; st[i * 4 + 1] = v.y; st[i * 4 + 2] = v.z; st[i * 4 + 3] = v.w;
  }
  const float Ah = A[h], Dh = Dsk[h], dtb = dt_bias[h];
  for (uint t0 = 0; t0 < p.T; t0 += kSsmNB) {
    const uint nb = min(kSsmNB, p.T - t0);
    for (uint i = 0; i < nb; ++i) {
      sbc[i][tid] = nh_conv(zx, ring, conv_w, conv_b, p, DI, CH, int(t0 + i), cbc);
      if (tid < 64) sx[i][tid] = nh_conv(zx, ring, conv_w, conv_b, p, DI, CH, int(t0 + i), cx);
      if (tid == 64) {
        const float dt = softplus(zx[size_t(t0 + i) * p.zx_stride + DI + CH + h] + dtb);
        sdt[i][0] = dt;
        sdt[i][1] = metal::precise::exp(dt * Ah);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < nb; ++i) {
      const float dt = sdt[i][0], dA = sdt[i][1];
      const float xv = sx[i][row];
      const float xdt = xv * dt;
      float acc = 0.0f;
      for (int n = 0; n < 32; ++n) {
        st[n] = st[n] * dA + sbc[i][qq * 32 + n] * xdt;
        acc += st[n] * sbc[i][128 + qq * 32 + n];
      }
      acc += simd_shuffle_xor(acc, 1);
      acc += simd_shuffle_xor(acc, 2);
      if (qq == 0) {
        const size_t t = t0 + i;
        const float z = zx[t * p.zx_stride + h * p.HD + row];
        y[t * DI + h * p.HD + row] = (acc + Dh * xv) * silu(z);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (p.write)
    for (int i = 0; i < 8; ++i) ((device float4*)sp)[i] = float4(st[i * 4], st[i * 4 + 1], st[i * 4 + 2], st[i * 4 + 3]);
  // one row (decode): the ring slot of this position is read by no head in this pass (they read the 3 before it),
  // so the conv input goes in here (x channels by their head, B / C by the group's first head) and nh_conv_ring is
  // not needed
  if (p.write && p.T == 1) {
    const uint slot = (p.pos0 & 3u) * CH;
    if (tid < 64) ring[slot + cx] = zx[DI + cx];
    if (h % (p.NH / p.NG) == 0) ring[slot + cbc] = zx[DI + cbc];
  }
}

// A long pass's conv (nh_conv of every channel and row) and dt (softplus, exp(dt A)) up front, in parallel: cv[t][c]
// for the CH conv channels, then dta[t][h] = (dt, exp(dt A)) after T x CH floats, the values nh_ssm stages itself
// (each B / C channel once per head of its group); nh_ssd.metal scans them.
// grid: (T, ceil((CH + NH) / 256)) x 256 threads
[[kernel]] void nh_ssm_conv(const device float* zx [[buffer(0)]], const device float* ring [[buffer(1)]],
                            device float* cv [[buffer(3)]], const device float* conv_w [[buffer(4)]],
                            const device float* conv_b [[buffer(5)]], const device float* A [[buffer(6)]],
                            const device float* dt_bias [[buffer(8)]], constant NhSsmParams& p [[buffer(9)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]]) {
  const uint DI = p.NH * p.HD, CH = DI + 2 * p.NG * p.DS;
  const uint t = tg.x, c = tg.y * 256 + tid.x;
  if (c < CH) {
    cv[size_t(t) * CH + c] = nh_conv(zx, ring, conv_w, conv_b, p, DI, CH, int(t), c);
  } else if (c < CH + p.NH) {
    const uint h = c - CH;
    const float dt = softplus(zx[size_t(t) * p.zx_stride + DI + CH + h] + dt_bias[h]);
    device float* d = cv + size_t(p.T) * CH + (size_t(t) * p.NH + h) * 2;
    d[0] = dt;
    d[1] = metal::precise::exp(dt * A[h]);
  }
}

// The same scan for another head dim HD (Nemotron Nano 4B / 9B v2: 80): a threadgroup of 4 x HD threads per head
// (HD state rows x 4 lanes of 32 columns), the head group's 256 B / C channels staged by the first 256 threads, the
// head's x channels by the first HD, dt by thread HD; the same update and sums as nh_ssm.
// grid: NH threadgroups x 4 HD threads.
template <uint HD>
NI_INLINE void nh_ssm_hd(const device float* zx, device float* ring, device float* state, device float* y,
                         const device float* conv_w, const device float* conv_b, const device float* A,
                         const device float* Dsk, const device float* dt_bias, constant NhSsmParams& p, uint h, uint tid,
                         threadgroup float* sbc, threadgroup float* sx, threadgroup float* sdt) {
  const uint DI = p.NH * p.HD, CH = DI + 2 * p.NG * p.DS;
  const uint g = h / (p.NH / p.NG);
  const uint row = tid >> 2, qq = tid & 3;
  const uint cbc = DI + (tid < 128 ? g * p.DS + tid : p.NG * p.DS + g * p.DS + (tid - 128));
  const uint cx = h * p.HD + tid % HD;
  float st[32];
  device float* sp = state + (size_t(h) * p.HD + row) * p.DS + qq * 32;
  for (int i = 0; i < 8; ++i) {
    float4 v = ((device float4*)sp)[i];
    st[i * 4 + 0] = v.x; st[i * 4 + 1] = v.y; st[i * 4 + 2] = v.z; st[i * 4 + 3] = v.w;
  }
  const float Ah = A[h], Dh = Dsk[h], dtb = dt_bias[h];
  for (uint t0 = 0; t0 < p.T; t0 += kSsmNB) {
    const uint nb = min(kSsmNB, p.T - t0);
    for (uint i = 0; i < nb; ++i) {
      if (tid < 256) sbc[i * 256 + tid] = nh_conv(zx, ring, conv_w, conv_b, p, DI, CH, int(t0 + i), cbc);
      if (tid < HD) sx[i * HD + tid] = nh_conv(zx, ring, conv_w, conv_b, p, DI, CH, int(t0 + i), cx);
      if (tid == HD) {
        const float dt = softplus(zx[size_t(t0 + i) * p.zx_stride + DI + CH + h] + dtb);
        sdt[i * 2] = dt;
        sdt[i * 2 + 1] = metal::precise::exp(dt * Ah);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < nb; ++i) {
      const float dt = sdt[i * 2], dA = sdt[i * 2 + 1];
      const float xv = sx[i * HD + row];
      const float xdt = xv * dt;
      float acc = 0.0f;
      for (int n = 0; n < 32; ++n) {
        st[n] = st[n] * dA + sbc[i * 256 + qq * 32 + n] * xdt;
        acc += st[n] * sbc[i * 256 + 128 + qq * 32 + n];
      }
      acc += simd_shuffle_xor(acc, 1);
      acc += simd_shuffle_xor(acc, 2);
      if (qq == 0) {
        const size_t t = t0 + i;
        const float z = zx[t * p.zx_stride + h * p.HD + row];
        y[t * DI + h * p.HD + row] = (acc + Dh * xv) * silu(z);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (p.write)
    for (int i = 0; i < 8; ++i) ((device float4*)sp)[i] = float4(st[i * 4], st[i * 4 + 1], st[i * 4 + 2], st[i * 4 + 3]);
  // one row: this position's conv input into its ring slot (as nh_ssm)
  if (p.write && p.T == 1) {
    const uint slot = (p.pos0 & 3u) * CH;
    if (tid < HD) ring[slot + cx] = zx[DI + cx];
    if (h % (p.NH / p.NG) == 0 && tid < 256) ring[slot + cbc] = zx[DI + cbc];
  }
}

[[kernel, max_total_threads_per_threadgroup(320)]] void nh_ssm_h80(
    const device float* zx [[buffer(0)]], device float* ring [[buffer(1)]], device float* state [[buffer(2)]],
    device float* y [[buffer(3)]], const device float* conv_w [[buffer(4)]], const device float* conv_b [[buffer(5)]],
    const device float* A [[buffer(6)]], const device float* Dsk [[buffer(7)]],
    const device float* dt_bias [[buffer(8)]], constant NhSsmParams& p [[buffer(9)]],
    uint h [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float sbc[kSsmNB * 256], sx[kSsmNB * 80], sdt[kSsmNB * 2];
  nh_ssm_hd<80>(zx, ring, state, y, conv_w, conv_b, A, Dsk, dt_bias, p, h, tid, sbc, sx, sdt);
}

// Short passes (decode, verify, the pending-row flush; T <= 8): 8 state rows of one head per threadgroup, one
// simdgroup per row with lane l holding states 4l .. 4l + 3 (coalesced float4 loads and stores); the conv of the
// head group's B and C channels (one per thread), the rows' x channels and dt are staged in threadgroup memory once.
// Same update as nh_ssm, sums in another order.
// Before the pass's T rows (zx, positions pos0 ..) it can commit np pending rows (zp, positions pos0 - np ..: a
// verify block's accepted rows): their update is written to the state, then the T rows run on in registers
// (written only with p.write). The conv reads raw xBC of positions < pos0 - np from the ring, the pending rows from zp.
// grid: NH * HD / 8 threadgroups x 256 threads.
// The pending rows are rows 0 .. np - 1 of zp, or with ptree (a KFlash tree verify's accepted path) rows ppath[0 .. np).
struct NhSsm1P {
  uint np;     // pending rows (zp)
  uint ptree;  // pending row c is zp row ppath[c]
};
NI_INLINE uint nh_prow(const device int* ppath, uint ptree, int c) { return ptree ? uint(ppath[c]) : uint(c); }
NI_INLINE float nh_raw1(const device float* zx, const device float* zp, const device float* ring,
                        constant NhSsmParams& p, uint np, uint DI, uint CH, int c, uint ch, const device int* ppath,
                        uint ptree) {
  // c: combined row (pending rows 0 .. np - 1, then the pass's rows); c < 0: the ring
  if (c >= int(np)) return zx[size_t(c - int(np)) * p.zx_stride + DI + ch];
  if (c >= 0) return zp[size_t(nh_prow(ppath, ptree, c)) * p.zx_stride + DI + ch];
  return ring[((p.pos0 - np + uint(c + 4)) & 3u) * CH + ch];
}
NI_INLINE float nh_conv1(const device float* zx, const device float* zp, const device float* ring,
                         const device float* conv_w, const device float* conv_b, constant NhSsmParams& p, uint np,
                         uint DI, uint CH, int c, uint ch, const device int* ppath, uint ptree) {
  const float4 w = *(const device float4*)(conv_w + ch * 4);
  const float v = conv_b[ch] + w.x * nh_raw1(zx, zp, ring, p, np, DI, CH, c - 3, ch, ppath, ptree) +
                  w.y * nh_raw1(zx, zp, ring, p, np, DI, CH, c - 2, ch, ppath, ptree) +
                  w.z * nh_raw1(zx, zp, ring, p, np, DI, CH, c - 1, ch, ppath, ptree) +
                  w.w * nh_raw1(zx, zp, ring, p, np, DI, CH, c, ch, ppath, ptree);
  return silu(v);
}
[[kernel]] void nh_ssm1(const device float* zx [[buffer(0)]], device float* ring [[buffer(1)]],
                        device float* state [[buffer(2)]], device float* y [[buffer(3)]],
                        const device float* conv_w [[buffer(4)]], const device float* conv_b [[buffer(5)]],
                        const device float* A [[buffer(6)]], const device float* Dsk [[buffer(7)]],
                        const device float* dt_bias [[buffer(8)]], constant NhSsmParams& p [[buffer(9)]],
                        const device uint* live [[buffer(10)]], const device float* zp [[buffer(11)]],
                        constant NhSsm1P& q [[buffer(12)]], const device int* ppath [[buffer(13)]],
                        uint tgi [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 sbc[2 * kSsmNB][64];  // conv'd B (0..31) and C (32..63) of the head's group, as float4
  threadgroup float sx[2 * kSsmNB][8];     // conv'd x of the 8 rows
  threadgroup float sdt[2 * kSsmNB][2];    // softplus(dt), exp(dt * A)
  const uint DI = p.NH * p.HD, CH = DI + 2 * p.NG * p.DS;
  const uint tg_per_head = p.HD / 8;
  const uint h = tgi / tg_per_head, row0 = (tgi % tg_per_head) * 8, row = row0 + sg;
  const uint g = h / (p.NH / p.NG);
  const uint cx = h * p.HD + row;
  // conv channel of this thread: B (tid < 128) or C (tid >= 128) of group g
  const uint cbc = DI + (tid < 128 ? g * p.DS + tid : p.NG * p.DS + g * p.DS + (tid - 128));
  const float Ah = A[h], Dh = Dsk[h], dtb = dt_bias[h];
  const uint np = q.np;
  const uint T = p.write ? p.T : min(p.T, live[0]);  // a verify block's rows past the drafter's depth are not needed
  const uint n = np + T;
  const uint ptree = q.ptree;
  for (uint c = 0; c < n; ++c) {
    ((threadgroup float*)sbc[c])[tid] = nh_conv1(zx, zp, ring, conv_w, conv_b, p, np, DI, CH, int(c), cbc, ppath, ptree);
    if (tid < 8)
      sx[c][tid] = nh_conv1(zx, zp, ring, conv_w, conv_b, p, np, DI, CH, int(c), h * p.HD + row0 + tid, ppath, ptree);
    if (tid == 8) {
      const float zdt = c < np ? zp[size_t(nh_prow(ppath, ptree, int(c))) * p.zx_stride + DI + CH + h]
                               : zx[size_t(c - np) * p.zx_stride + DI + CH + h];
      const float dt = softplus(zdt + dtb);
      sdt[c][0] = dt;
      sdt[c][1] = metal::precise::exp(dt * Ah);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float4* sp = (device float4*)(state + (size_t(h) * p.HD + row) * p.DS) + lane;
  float4 st = *sp;
  for (uint c = 0; c < n; ++c) {
    const float xv = sx[c][sg], dt = sdt[c][0], dA = sdt[c][1];
    st = st * dA + sbc[c][lane] * (xv * dt);
    if (c + 1 == np) *sp = st;  // the pending rows committed
    if (c < np) continue;
    const float acc = simd_sum(dot(st, sbc[c][32 + lane]));
    if (lane == 0) {
      const uint t = c - np;
      const float z = zx[size_t(t) * p.zx_stride + cx];
      y[size_t(t) * DI + cx] = (acc + Dh * xv) * silu(z);
    }
  }
  if (p.write && T > 0) {
    *sp = st;
    // one row (decode): the conv input goes into this position's ring slot here (read by no head in this pass)
    if (p.T == 1 && np == 0) {
      const uint slot = (p.pos0 & 3u) * CH;
      if (lane == 0) ring[slot + cx] = zx[DI + cx];
      if (h % (p.NH / p.NG) == 0 && row0 == 0) ring[slot + cbc] = zx[DI + cbc];
    }
  }
}

// KFlash tree verify: nh_ssm1's short pass over a verify block whose T rows form a tree (the KFlash tree buffer: row
// 0 the root, every row's parent before it). A row's state is its parent's (the root's: the state after the pending
// rows), so each row's output is what plain decoding of its root path gives; its conv window is its own raw input and
// its three nearest ancestors' (past the root: the pending rows, then the ring). The pending rows commit as in
// nh_ssm1; the tree rows' states are not written (the accepted path becomes the next pass's pending rows).
// grid: NH * HD / 8 threadgroups x 256 threads.
NI_INLINE int nh_tpred(const device int* tree, uint np, int c) {
  // combined row c's predecessor: a tree row's parent (the root's: the last pending row), else the row before
  return c > int(np) ? int(np) + tree[kKfPar + uint(c) - np] : c - 1;
}
NI_INLINE float nh_tconv(const device float* zx, const device float* zp, const device float* ring,
                         const device float* conv_w, const device float* conv_b, constant NhSsmParams& p, uint np,
                         uint DI, uint CH, int c, uint ch, const device int* ppath, uint ptree, const device int* tree) {
  const int c1 = nh_tpred(tree, np, c), c2 = nh_tpred(tree, np, c1), c3 = nh_tpred(tree, np, c2);
  const float4 w = *(const device float4*)(conv_w + ch * 4);
  const float v = conv_b[ch] + w.x * nh_raw1(zx, zp, ring, p, np, DI, CH, c3, ch, ppath, ptree) +
                  w.y * nh_raw1(zx, zp, ring, p, np, DI, CH, c2, ch, ppath, ptree) +
                  w.z * nh_raw1(zx, zp, ring, p, np, DI, CH, c1, ch, ppath, ptree) +
                  w.w * nh_raw1(zx, zp, ring, p, np, DI, CH, c, ch, ppath, ptree);
  return silu(v);
}
[[kernel]] void nh_ssm1_tree(const device float* zx [[buffer(0)]], device float* ring [[buffer(1)]],
                             device float* state [[buffer(2)]], device float* y [[buffer(3)]],
                             const device float* conv_w [[buffer(4)]], const device float* conv_b [[buffer(5)]],
                             const device float* A [[buffer(6)]], const device float* Dsk [[buffer(7)]],
                             const device float* dt_bias [[buffer(8)]], constant NhSsmParams& p [[buffer(9)]],
                             const device uint* live [[buffer(10)]], const device float* zp [[buffer(11)]],
                             constant NhSsm1P& q [[buffer(12)]], const device int* ppath [[buffer(13)]],
                             const device int* tree [[buffer(14)]],
                             uint tgi [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 sbc[2 * kSsmNB][64];  // conv'd B (0..31) and C (32..63) of the head's group, as float4
  threadgroup float sx[2 * kSsmNB][8];     // conv'd x of the 8 rows
  threadgroup float sdt[2 * kSsmNB][2];    // softplus(dt), exp(dt * A)
  const uint DI = p.NH * p.HD, CH = DI + 2 * p.NG * p.DS;
  const uint tg_per_head = p.HD / 8;
  const uint h = tgi / tg_per_head, row0 = (tgi % tg_per_head) * 8, row = row0 + sg;
  const uint g = h / (p.NH / p.NG);
  const uint cx = h * p.HD + row;
  const uint cbc = DI + (tid < 128 ? g * p.DS + tid : p.NG * p.DS + g * p.DS + (tid - 128));
  const float Ah = A[h], Dh = Dsk[h], dtb = dt_bias[h];
  const uint np = q.np, ptree = q.ptree;
  const uint T = min(min(p.T, live[0]), uint(tree[kKfN]));
  const uint n = np + T;
  for (uint c = 0; c < n; ++c) {
    ((threadgroup float*)sbc[c])[tid] = nh_tconv(zx, zp, ring, conv_w, conv_b, p, np, DI, CH, int(c), cbc, ppath, ptree, tree);
    if (tid < 8)
      sx[c][tid] = nh_tconv(zx, zp, ring, conv_w, conv_b, p, np, DI, CH, int(c), h * p.HD + row0 + tid, ppath, ptree, tree);
    if (tid == 8) {
      const float zdt = c < np ? zp[size_t(nh_prow(ppath, ptree, int(c))) * p.zx_stride + DI + CH + h]
                               : zx[size_t(c - np) * p.zx_stride + DI + CH + h];
      const float dt = softplus(zdt + dtb);
      sdt[c][0] = dt;
      sdt[c][1] = metal::precise::exp(dt * Ah);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float4* sp = (device float4*)(state + (size_t(h) * p.HD + row) * p.DS) + lane;
  float4 st = *sp;
  for (uint c = 0; c < np; ++c) {  // the pending rows (a chain) are committed
    st = st * sdt[c][1] + sbc[c][lane] * (sx[c][sg] * sdt[c][0]);
    if (c + 1 == np) *sp = st;
  }
  float4 ts[8];
  _Pragma("clang loop unroll(full)")
  for (uint t = 0; t < 8; ++t) {
    if (t < T) {
      const int par = t == 0 ? -1 : tree[kKfPar + t];
      float4 base = st;
      _Pragma("clang loop unroll(full)")
      for (uint j = 0; j < 8; ++j)
        if (j < t && int(j) == par) base = ts[j];
      const uint c = np + t;
      const float xv = sx[c][sg];
      ts[t] = base * sdt[c][1] + sbc[c][lane] * (xv * sdt[c][0]);
      const float acc = simd_sum(dot(ts[t], sbc[c][32 + lane]));
      if (lane == 0) {
        const float z = zx[size_t(t) * p.zx_stride + cx];
        y[size_t(t) * DI + cx] = (acc + Dh * xv) * silu(z);
      }
    }
  }
}

// The same short-pass scan with RPS state rows per simdgroup: 8 RPS rows of a head per threadgroup (the dense Nemotron
// Nano models: head dim 80, 40 or 80 rows), so the head group's B / C conv, the rows' x and dt are staged once for 8 RPS
// rows instead of 8 (nh_ssm1 at head dim 80: ten threadgroups per head each stage the group's 256 B / C channels, ~70
// GB/s of state traffic on an M1 Pro). Each row's update is nh_ssm1's, in the same order: the same outputs to the bit.
// grid: NH * HD / (8 RPS) threadgroups x 256 threads; HD % (8 RPS) == 0.
template <int RPS, bool SZ = false>
NI_INLINE void nh_ssm1r_impl(const device float* zx, device float* ring, device float* state, device float* y,
                             const device float* conv_w, const device float* conv_b, const device float* A,
                             const device float* Dsk, const device float* dt_bias, constant NhSsmParams& p,
                             const device uint* live, const device float* zp, uint np, const device int* ppath,
                             uint ptree, uint tgi, uint tid, uint sg, uint lane, threadgroup float4 (*sbc)[64],
                             threadgroup float* sx, threadgroup float (*sdt)[2], threadgroup float* sz = nullptr) {
  constexpr uint RT = 8 * RPS;  // state rows per threadgroup
  const uint DI = p.NH * p.HD, CH = DI + 2 * p.NG * p.DS;
  const uint tg_per_head = p.HD / RT;
  const uint h = tgi / tg_per_head, row0 = (tgi % tg_per_head) * RT;
  const uint g = h / (p.NH / p.NG);
  const uint cbc = DI + (tid < 128 ? g * p.DS + tid : p.NG * p.DS + g * p.DS + (tid - 128));
  const float Ah = A[h], Dh = Dsk[h], dtb = dt_bias[h];
  const uint T = p.write ? p.T : min(p.T, live[0]);
  const uint n = np + T;
  for (uint c = 0; c < n; ++c) {
    ((threadgroup float*)sbc[c])[tid] = nh_conv1(zx, zp, ring, conv_w, conv_b, p, np, DI, CH, int(c), cbc, ppath, ptree);
    if (tid < RT)
      sx[c * RT + tid] =
          nh_conv1(zx, zp, ring, conv_w, conv_b, p, np, DI, CH, int(c), h * p.HD + row0 + tid, ppath, ptree);
    // SZ: the gate z of the pass's rows staged too (the steps below then wait on no device load)
    if (SZ && tid < RT && c >= np) sz[(c - np) * RT + tid] = zx[size_t(c - np) * p.zx_stride + h * p.HD + row0 + tid];
    if (tid == RT) {
      const float zdt = c < np ? zp[size_t(nh_prow(ppath, ptree, int(c))) * p.zx_stride + DI + CH + h]
                               : zx[size_t(c - np) * p.zx_stride + DI + CH + h];
      const float dt = softplus(zdt + dtb);
      sdt[c][0] = dt;
      sdt[c][1] = metal::precise::exp(dt * Ah);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float4* sp[RPS];
  float4 st[RPS];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < RPS; ++i) {
    sp[i] = (device float4*)(state + (size_t(h) * p.HD + row0 + sg * RPS + uint(i)) * p.DS) + lane;
    st[i] = *sp[i];
  }
  if (SZ) {
    // per step the RPS rows' updates, then their sums together (RPS independent shuffle chains), then their outputs
    // (z from threadgroup memory): each value as below. Lightning, NI_PROFILE us per layer: verify at depth 7 33.3 ->
    // 24.8, depth 3 21.4 -> 18.9, decode 12.6 -> 11.6; the same outputs to the bit
    for (uint c = 0; c < n; ++c) {
      const float dt = sdt[c][0], dA = sdt[c][1];
      const float4 bv = sbc[c][lane], cv = sbc[c][32 + lane];
      float xv[RPS], acc[RPS];
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < RPS; ++i) {
        xv[i] = sx[c * RT + sg * RPS + uint(i)];
        st[i] = st[i] * dA + bv * (xv[i] * dt);
      }
      if (c + 1 == np)
        _Pragma("clang loop unroll(full)")
        for (int i = 0; i < RPS; ++i) *sp[i] = st[i];  // the pending rows committed
      if (c < np) continue;
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < RPS; ++i) acc[i] = dot(st[i], cv);
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < RPS; ++i) acc[i] = simd_sum(acc[i]);
      if (lane == 0)
        _Pragma("clang loop unroll(full)")
        for (int i = 0; i < RPS; ++i) {
          const uint r = sg * RPS + uint(i), t = c - np, cx = h * p.HD + row0 + r;
          y[size_t(t) * DI + cx] = (acc[i] + Dh * xv[i]) * silu(sz[t * RT + r]);
        }
    }
  } else
  for (uint c = 0; c < n; ++c) {
    const float dt = sdt[c][0], dA = sdt[c][1];
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < RPS; ++i) {
      const uint r = sg * RPS + uint(i);
      const float xv = sx[c * RT + r];
      st[i] = st[i] * dA + sbc[c][lane] * (xv * dt);
      if (c + 1 == np) *sp[i] = st[i];  // the pending rows committed
      if (c < np) continue;
      const float acc = simd_sum(dot(st[i], sbc[c][32 + lane]));
      if (lane == 0) {
        const uint t = c - np, cx = h * p.HD + row0 + r;
        const float z = zx[size_t(t) * p.zx_stride + cx];
        y[size_t(t) * DI + cx] = (acc + Dh * xv) * silu(z);
      }
    }
  }
  if (p.write && T > 0) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < RPS; ++i) *sp[i] = st[i];
    // one row (decode): the conv input goes into this position's ring slot here (read by no head in this pass)
    if (p.T == 1 && np == 0) {
      const uint slot = (p.pos0 & 3u) * CH;
      if (tid < RT) ring[slot + h * p.HD + row0 + tid] = zx[DI + h * p.HD + row0 + tid];
      if (h % (p.NH / p.NG) == 0 && row0 == 0) ring[slot + cbc] = zx[DI + cbc];
    }
  }
}

#define NH_SSM1R(NAME, RPS)                                                                                        \
  [[kernel]] void NAME(const device float* zx [[buffer(0)]], device float* ring [[buffer(1)]],                     \
                       device float* state [[buffer(2)]], device float* y [[buffer(3)]],                          \
                       const device float* conv_w [[buffer(4)]], const device float* conv_b [[buffer(5)]],        \
                       const device float* A [[buffer(6)]], const device float* Dsk [[buffer(7)]],                \
                       const device float* dt_bias [[buffer(8)]], constant NhSsmParams& p [[buffer(9)]],          \
                       const device uint* live [[buffer(10)]], const device float* zp [[buffer(11)]],             \
                       constant NhSsm1P& q [[buffer(12)]], const device int* ppath [[buffer(13)]],             \
                       uint tgi [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],    \
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {     \
    threadgroup float4 sbc[2 * kSsmNB][64];                                                                        \
    threadgroup float sx[2 * kSsmNB * 8 * RPS];                                                                    \
    threadgroup float sdt[2 * kSsmNB][2];                                                                          \
    nh_ssm1r_impl<RPS>(zx, ring, state, y, conv_w, conv_b, A, Dsk, dt_bias, p, live, zp, q.np, ppath, q.ptree, tgi,  \
                       tid, sg, lane, sbc, sx, sdt);                                                               \
  }
NH_SSM1R(nh_ssm1r2, 2)    // 16 rows per threadgroup
NH_SSM1R(nh_ssm1r8, 8)    // 64 rows: a head per threadgroup at head dim 64
NH_SSM1R(nh_ssm1r5, 5)    // 40 rows per threadgroup
NH_SSM1R(nh_ssm1r10, 10)  // 80 rows: a head per threadgroup at head dim 80
// 32 rows (the M5's MoE models), z staged (SZ)
[[kernel]] void nh_ssm1r4(const device float* zx [[buffer(0)]], device float* ring [[buffer(1)]],
                          device float* state [[buffer(2)]], device float* y [[buffer(3)]],
                          const device float* conv_w [[buffer(4)]], const device float* conv_b [[buffer(5)]],
                          const device float* A [[buffer(6)]], const device float* Dsk [[buffer(7)]],
                          const device float* dt_bias [[buffer(8)]], constant NhSsmParams& p [[buffer(9)]],
                          const device uint* live [[buffer(10)]], const device float* zp [[buffer(11)]],
                          constant NhSsm1P& q [[buffer(12)]], const device int* ppath [[buffer(13)]],
                          uint tgi [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 sbc[2 * kSsmNB][64];
  threadgroup float sx[2 * kSsmNB * 8 * 4];
  threadgroup float sdt[2 * kSsmNB][2];
  threadgroup float sz[kSsmNB * 8 * 4];
  nh_ssm1r_impl<4, true>(zx, ring, state, y, conv_w, conv_b, A, Dsk, dt_bias, p, live, zp, q.np, ppath, q.ptree, tgi,
                         tid, sg, lane, sbc, sx, sdt, sz);
}

// The conv ring after a pass of T rows at pos0: the raw xBC of its last min(T, 3) rows into their slots.
// grid: ceil(CH * min(T, 3) / 256) x 256 threads
// With ptree the rows are zx rows ppath[0 .. T) (a KFlash tree verify's accepted path committed as pending rows).
struct NhRingParams {
  uint T, pos0, zx_stride, DI, CH, ptree;
};
[[kernel]] void nh_conv_ring(const device float* zx [[buffer(0)]], device float* ring [[buffer(1)]],
                             constant NhRingParams& p [[buffer(2)]], const device int* ppath [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
  const uint nr = min(p.T, 3u);
  if (i >= nr * p.CH) return;
  const uint t = p.T - nr + i / p.CH, c = i % p.CH;
  const uint row = p.ptree ? uint(ppath[t]) : t;
  ring[((p.pos0 + t) & 3u) * p.CH + c] = zx[size_t(row) * p.zx_stride + p.DI + c];
}

// nh_gnorm_half (nemotron/nh_gemm.metal) that also does a verify's nh_conv_ring of the pending rows it committed
// (zp [[4]], ring [[5]], NhRingParams [[6]]: the scan before it has read the ring), spread over the grid's threads:
// one pass less per Mamba layer. Each output as the two passes write it. grid: (R * groups) x 256 threads
[[kernel]] void nh_gnorm_half_ring(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                                   device half* y [[buffer(2)]], constant NhGnormParams& p [[buffer(3)]],
                                   const device float* zp [[buffer(4)]], device float* ring [[buffer(5)]],
                                   constant NhRingParams& rp [[buffer(6)]], uint tg [[threadgroup_position_in_grid]],
                                   uint ngrid [[threadgroups_per_grid]], uint tid [[thread_position_in_threadgroup]],
                                   uint ntg [[threads_per_threadgroup]]) {
  const uint nr = min(rp.T, 3u);
  for (uint i = tg * ntg + tid; i < nr * rp.CH; i += ngrid * ntg) {
    const uint t = rp.T - nr + i / rp.CH, c = i % rp.CH;
    ring[((rp.pos0 + t) & 3u) * rp.CH + c] = zp[size_t(t) * rp.zx_stride + rp.DI + c];
  }
  threadgroup float scratch[32];
  const uint ng = p.D / p.G, r = tg / ng, g = tg % ng;
  const device float* xr = x + size_t(r) * p.D + g * p.G;
  float ss = 0.0f;
  for (uint k = tid; k < p.G; k += ntg) ss += xr[k] * xr[k];
  ss = tg_sum(ss, scratch, tid, ntg);
  const float inv = 1.0f / metal::precise::sqrt(ss / float(p.G) + p.eps);
  for (uint k = tid; k < p.G; k += ntg) y[size_t(r) * p.D + g * p.G + k] = half(xr[k] * inv * w[g * p.G + k]);
}

// ---- attention (no RoPE: Nemotron-H's attention layers carry no positional encoding) ----
// K/V caches: half [Hkv][cap][HD]. Rows r of a pass sit at positions pos0 + r.
struct NhKvParams {
  uint R, pos0, cap, Hkv, HD;
};
// grid: (R, 2 * Hkv) threadgroups x HD threads
[[kernel]] void nh_kv_write(const device float* k [[buffer(0)]], const device float* v [[buffer(1)]],
                            device half* kc [[buffer(2)]], device half* vc [[buffer(3)]],
                            constant NhKvParams& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                            uint2 d2 [[thread_position_in_threadgroup]]) {
  const uint d = d2.x;
  const uint r = tg.x, hh = tg.y % p.Hkv;
  const bool isv = tg.y >= p.Hkv;
  const float val = (isv ? v : k)[size_t(r) * p.Hkv * p.HD + hh * p.HD + d];
  (isv ? vc : kc)[(size_t(hh) * p.cap + p.pos0 + r) * p.HD + d] = half(val);
}

// Split-K decode attention for R query rows (row r sees positions [0, pos0 + r]). One threadgroup per
// (chunk, kv head, row): its G = Hq / Hkv simdgroups are the query heads sharing the kv head; K/V tiles are
// staged once for them. Lanes hold HD / 32 = 4 dims. grid: (nchunks, Hkv, R) x (32 * G) threads.
struct NhAttnParams {
  uint R, pos0, cap, Hq, Hkv, chunk, nchunks;
  float scale;
};
constant constexpr uint kAttnKT = 32;  // keys per tile

// KFlash tree verify (nh_attn_partial_tree): the block's rows form a tree (the KFlash tree buffer), so row r sees
// the history [0, pos0) and, of the block, only itself and its ancestors (its mask's bits).
template <bool TREE>
NI_INLINE void nh_attn_partial_impl(const device float* q, const device half* kc, const device half* vc,
                                    device float* part_o, device float* part_ml, constant NhAttnParams& p,
                                    const device int* tree, uint3 tg, uint tid, uint sg, uint lane, uint ntg,
                                    threadgroup half4* kt, threadgroup half4* vt) {
  constexpr uint HD = 128;
  const uint chunk = tg.x, kvh = tg.y, r = tg.z;
  const uint G = p.Hq / p.Hkv, hq = kvh * G + sg;
  const uint L = p.pos0 + r + 1;  // keys visible to this row
  const uint start = chunk * p.chunk, end = min(start + p.chunk, L);
  const uint mrow = TREE ? uint(tree[kKfMask + r]) : 0u;
  const float4 qv = ((const device float4*)(q + (size_t(r) * p.Hq + hq) * HD))[lane] * p.scale;
  float4 acc = 0.0f;
  float m = -INFINITY, l = 0.0f;
  const device half4* kb = (const device half4*)(kc + size_t(kvh) * p.cap * HD);
  const device half4* vb = (const device half4*)(vc + size_t(kvh) * p.cap * HD);
  for (uint t0 = start; t0 < end; t0 += kAttnKT) {
    const uint n = min(kAttnKT, end - t0);
    for (uint i = tid; i < n * HD / 4; i += ntg) {
      kt[i] = kb[t0 * HD / 4 + i];
      vt[i] = vb[t0 * HD / 4 + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < n; ++j) {
      if (TREE && t0 + j >= p.pos0 && !((mrow >> (t0 + j - p.pos0)) & 1u)) continue;
      const float s = simd_sum(dot(qv, float4(kt[j * HD / 4 + lane])));
      const float mn = max(m, s);
      const float corr = metal::fast::exp(m - mn), e = metal::fast::exp(s - mn);
      l = l * corr + e;
      acc = acc * corr + e * float4(vt[j * HD / 4 + lane]);
      m = mn;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const size_t base = (size_t(r) * p.Hq + hq) * p.nchunks + chunk;
  ((device float4*)(part_o + base * HD))[lane] = acc;
  if (lane == 0) {
    part_ml[base * 2 + 0] = m;
    part_ml[base * 2 + 1] = l;
  }
}

[[kernel]] void nh_attn_partial(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                const device half* vc [[buffer(2)]], device float* part_o [[buffer(3)]],
                                device float* part_ml [[buffer(4)]], constant NhAttnParams& p [[buffer(5)]],
                                uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                uint3 ntg3 [[threads_per_threadgroup]]) {
  threadgroup half4 kt[kAttnKT * 128 / 4];
  threadgroup half4 vt[kAttnKT * 128 / 4];
  nh_attn_partial_impl<false>(q, kc, vc, part_o, part_ml, p, nullptr, tg, tid, sg, lane, ntg3.x, kt, vt);
}

[[kernel]] void nh_attn_partial_tree(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                     const device half* vc [[buffer(2)]], device float* part_o [[buffer(3)]],
                                     device float* part_ml [[buffer(4)]], constant NhAttnParams& p [[buffer(5)]],
                                     const device int* tree [[buffer(6)]],
                                     uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                     uint3 ntg3 [[threads_per_threadgroup]]) {
  threadgroup half4 kt[kAttnKT * 128 / 4];
  threadgroup half4 vt[kAttnKT * 128 / 4];
  nh_attn_partial_impl<true>(q, kc, vc, part_o, part_ml, p, tree, tg, tid, sg, lane, ntg3.x, kt, vt);
}

// KFlash: after a tree verify's acceptance, the accepted path's K/V (verify rows path[1 .. a], written at pos0 +
// path[i]) move to pos0 + i, where plain decoding puts them. Each thread copies one element of one head in path order
// (a destination pos0 + i never holds a later row's source: path[j] >= j). grid: 2 * Hkv threadgroups x HD threads.
[[kernel]] void nh_kv_tree_compact(device half* kc [[buffer(0)]], device half* vc [[buffer(1)]],
                                   const device int* tree [[buffer(2)]], constant NhKvParams& p [[buffer(3)]],
                                   uint tg [[threadgroup_position_in_grid]], uint d [[thread_position_in_threadgroup]]) {
  const uint hh = tg % p.Hkv;
  device half* c = (tg >= p.Hkv ? vc : kc) + size_t(hh) * p.cap * p.HD;
  const uint a = uint(tree[kKfA]);
  for (uint i = 1; i <= a; ++i) {
    const uint src = uint(tree[kKfPath + i]);
    if (src != i) c[size_t(p.pos0 + i) * p.HD + d] = c[size_t(p.pos0 + src) * p.HD + d];
  }
}

// grid: (Hq, R) threadgroups x HD threads. out[r, h * HD + d]; nh_attn_reduce_h: as half (the verify's o_proj input,
// what nh_rows_half made of the fp32 rows)
NI_INLINE float nh_attn_reduce_v(const device float* part_o, const device float* part_ml, constant NhAttnParams& p,
                                 uint h, uint r, uint d) {
  constexpr uint HD = 128;
  const size_t base = (size_t(r) * p.Hq + h) * p.nchunks;
  const uint L = p.pos0 + r + 1, nc = min(p.nchunks, (L + p.chunk - 1) / p.chunk);
  float M = -INFINITY;
  for (uint c = 0; c < nc; ++c) M = max(M, part_ml[(base + c) * 2]);
  float num = 0.0f, den = 0.0f;
  for (uint c = 0; c < nc; ++c) {
    const float mc = part_ml[(base + c) * 2];
    if (mc == -INFINITY) continue;
    const float w = metal::fast::exp(mc - M);
    num += w * part_o[(base + c) * HD + d];
    den += w * part_ml[(base + c) * 2 + 1];
  }
  return num / den;
}
[[kernel]] void nh_attn_reduce(const device float* part_o [[buffer(0)]], const device float* part_ml [[buffer(1)]],
                               device float* out [[buffer(2)]], constant NhAttnParams& p [[buffer(3)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint2 d2 [[thread_position_in_threadgroup]]) {
  out[(size_t(tg.y) * p.Hq + tg.x) * 128 + d2.x] = nh_attn_reduce_v(part_o, part_ml, p, tg.x, tg.y, d2.x);
}
[[kernel]] void nh_attn_reduce_h(const device float* part_o [[buffer(0)]], const device float* part_ml [[buffer(1)]],
                                 device half* out [[buffer(2)]], constant NhAttnParams& p [[buffer(3)]],
                                 uint2 tg [[threadgroup_position_in_grid]], uint2 d2 [[thread_position_in_threadgroup]]) {
  out[(size_t(tg.y) * p.Hq + tg.x) * 128 + d2.x] = half(nh_attn_reduce_v(part_o, part_ml, p, tg.x, tg.y, d2.x));
}

// ---- MoE routing: sigmoid scores, + selection bias, top-k, the k scores renormalized and scaled ----
// grid: R threadgroups x 512 threads (E <= 512)
struct NhRouteParams {
  uint E, k, norm;
  float scale;
  uint logit_stride;
  uint rows;  // rows of the pass (nh_moe_route1)
};

// (nh_route_row: nh_route.h)

// A verify's routed-expert grids from its live rows (dynamic rows: the drafter wrote live[0]): the up and down
// products' indirect arguments {gx, min(live, R) k, 1} at args[0] and args[4], so the pairs of rows past live get no
// threadgroups at all. grid: 1 x 1
struct NhPairArgsP {
  uint gx_up, gx_dn, k, R;
};
[[kernel]] void nh_moe_pair_args(const device uint* live [[buffer(0)]], device uint* args [[buffer(1)]],
                                 constant NhPairArgsP& p [[buffer(2)]]) {
  const uint n = max(1u, min(live[0], p.R)) * p.k;
  args[0] = p.gx_up;
  args[1] = n;
  args[2] = 1;
  args[4] = p.gx_dn;
  args[5] = n;
  args[6] = 1;
}

// One simdgroup per row, 8 rows per threadgroup. Same selection and weights as nh_moe_route.
// grid: ceil(rows / 8) threadgroups x 256 threads.
[[kernel]] void nh_moe_route1(const device float* logits [[buffer(0)]], const device float* bias [[buffer(1)]],
                              device int* eid [[buffer(2)]], device float* ew [[buffer(3)]],
                              constant NhRouteParams& p [[buffer(4)]], const device uint* live [[buffer(5)]],
                              uint tg [[threadgroup_position_in_grid]], uint sgi [[simdgroup_index_in_threadgroup]],
                              uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg * nsg + sgi;
  if (r >= p.rows || r >= live[0]) return;
  nh_route_row(logits + size_t(r) * p.logit_stride, bias, p.E, p.k, p.norm, p.scale, lane, eid + r * p.k, ew + r * p.k);
}

[[kernel]] void nh_moe_route(const device float* logits [[buffer(0)]], const device float* bias [[buffer(1)]],
                             device int* eid [[buffer(2)]], device float* ew [[buffer(3)]],
                             constant NhRouteParams& p [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float sv[32];
  threadgroup uint si[32];
  threadgroup float wsel[16];
  threadgroup uint isel[16];
  const bool live = tid < p.E;
  const float prob = live ? 1.0f / (1.0f + metal::precise::exp(-logits[size_t(r) * p.logit_stride + tid])) : 0.0f;
  float sel = live ? prob + bias[tid] : -INFINITY;
  for (uint j = 0; j < p.k; ++j) {
    float v = sel;
    uint i = live ? tid : 0xffffffffu;
    // argmax, lowest index on ties
    for (uint off = 16; off > 0; off >>= 1) {
      const float ov = simd_shuffle_down(v, off);
      const uint oi = simd_shuffle_down(i, off);
      if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    if ((tid & 31) == 0) { sv[tid >> 5] = v; si[tid >> 5] = i; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
      const uint nsg = (ntg + 31) / 32;
      v = tid < nsg ? sv[tid] : -INFINITY;
      i = tid < nsg ? si[tid] : 0xffffffffu;
      for (uint off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_down(v, off);
        const uint oi = simd_shuffle_down(i, off);
        if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
      }
      if (tid == 0) isel[j] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == isel[j]) {
      wsel[j] = prob;
      sel = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    float sum = 1e-20f;  // as NVIDIA's modeling code: weights / (sum + 1e-20)
    for (uint j = 0; j < p.k; ++j) sum += wsel[j];
    for (uint j = 0; j < p.k; ++j) {
      eid[r * p.k + j] = int(isel[j]);
      ew[r * p.k + j] = (p.norm ? wsel[j] / sum : wsel[j]) * p.scale;
    }
  }
}

// ---- small helpers ----
// y[i] = a[i] + b[i] (fp32), n elements. grid: ceil(n / 256) x 256
[[kernel]] void nh_add(const device float* a [[buffer(0)]], const device float* b [[buffer(1)]],
                       device float* y [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i < n) y[i] = a[i] + b[i];
}
