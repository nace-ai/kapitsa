// ---- qmti8: W4A8 dense prefill GEMM on the M5's int8 matrix units (NI_QMT_I8) ----
// The int8 x int8 -> int32 matmul2d runs at ~64 TOPS on the M5 Pro against ~32 TFLOP/s for half (tools/bench_tops:
// tp_32_32_64_i8 64.5, tp_32_32_64 32.3, and 58.3 with a per-K-step fp32 rescale, tp_rs_32_32_64).
// X is quantized first by qx8_g64: int8 per row per group of 64 (scale absmax / 127, round to nearest), the k order
// inside each 8 permuted to [0,2,4,6,1,3,5,7] (the order a packed 4-bit word splits into with two byte masks, so
// the weight tile needs no shuffle), plus float2 (sx, sx * sum q) per row-group. With w = q_w * s + b per group:
//   Y[m, n] = sum_g ( sx * s * C_g + sx * sum(q_x) * b ),  C_g = the int32 dot of the group's 64 codes.
// Then qmti8s: BM x BN outputs, simdgroups of 32 x 32, BK = 64 (one weight group per step, a fresh int32 product per
// step rescaled into fp32), double-buffered as qmt_tile. Not bitwise qmt's (activation rounding).

[[kernel]] void qx8_g64(const device half* X [[buffer(0)]], device char* Q [[buffer(1)]], device float2* XS [[buffer(2)]],
                        constant packed_uint3& p [[buffer(3)]],  // M, K, x_ld (halves); 12 bytes, as the host passes
                        uint j [[thread_position_in_grid]]) {
  const uint G = p.y / 64, gi = j / 8, sub = j % 8, m = gi / G, g = gi % G;
  const bool ok = m < p.x;
  float4 a = 0, b = 0;
  if (ok) {
    const device half4* src = (const device half4*)(X + size_t(m) * p.z + g * 64 + sub * 8);
    a = float4(src[0]);
    b = float4(src[1]);
  }
  const float4 t = max(abs(a), abs(b));
  float am = max(max(t.x, t.y), max(t.z, t.w));
  am = max(am, simd_shuffle_xor(am, 1));
  am = max(am, simd_shuffle_xor(am, 2));
  am = max(am, simd_shuffle_xor(am, 4));
  const float inv = am > 0 ? 127.0f / am : 0.0f;
  const int4 qa = int4(clamp(rint(a * inv), -127.0f, 127.0f)), qb = int4(clamp(rint(b * inv), -127.0f, 127.0f));
  int s = qa.x + qa.y + qa.z + qa.w + qb.x + qb.y + qb.z + qb.w;
  s += simd_shuffle_xor(s, 1);
  s += simd_shuffle_xor(s, 2);
  s += simd_shuffle_xor(s, 4);
  if (!ok) return;
  device char4* dst = (device char4*)(Q + size_t(m) * p.y + g * 64 + sub * 8);
  dst[0] = char4(qa.x, qa.z, qb.x, qb.z);
  dst[1] = char4(qa.y, qa.w, qb.y, qb.w);
  if (sub == 0) {
    const float sx = am / 127.0f;
    XS[size_t(m) * G + g] = float2(sx, sx * float(s));
  }
}

// qmti8s: each simdgroup's 32 x 32 as two 32 x 16 products on alternating int32 tiles, so the rescale of one runs
// while the matrix unit computes the other (one product per step makes every step wait for its own result).
template <int BM, int BN>
NI_INLINE void qmti8s_tile(const device char* XQ, const device float2* XS, const device uint* W, const device bfloat* S,
                           const device bfloat* B, device float* Y, constant QmtParams& p, uint m0, uint n0, uint tid,
                           uint sg, threadgroup int8_t* xs, threadgroup int8_t* ws, threadgroup float2* xsc,
                           threadgroup float2* wsc) {
  // simdgroups of 32 x 32 (each two 32 x 16 products): (BM / 32) x (BN / 32) of them
  constexpr int BK = 64, SM = 32, SN = 32, HN = 16, CAP = SM * HN / 32, NSN = BN / 32, NT = BM * BN / 32;
  const uint sm = sg / NSN, sn = sg % NSN, G = p.K / 64, wld = p.K / 8, sld = p.K / 64;
  constexpr auto desc =
      matmul2d_descriptor(SM, HN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  typedef tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tg_t;
  auto tX0 = tg_t(xs + sm * SM * BK, dextents<int32_t, 2>(BK, SM));
  auto tX1 = tg_t(xs + BM * BK + sm * SM * BK, dextents<int32_t, 2>(BK, SM));
  auto tW00 = tg_t(ws + sn * SN * BK, dextents<int32_t, 2>(BK, HN));
  auto tW01 = tg_t(ws + (sn * SN + HN) * BK, dextents<int32_t, 2>(BK, HN));
  auto tW10 = tg_t(ws + BN * BK + sn * SN * BK, dextents<int32_t, 2>(BK, HN));
  auto tW11 = tg_t(ws + BN * BK + (sn * SN + HN) * BK, dextents<int32_t, 2>(BK, HN));
  auto c0 = op.template get_destination_cooperative_tensor<tg_t, tg_t, int32_t>();
  auto c1 = op.template get_destination_cooperative_tensor<tg_t, tg_t, int32_t>();
  float a0[CAP], a1[CAP];
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < CAP; ++i) a0[i] = a1[i] = 0.0f;
  // Where element i of this lane's 32 x 16 int32 product sits, read from the destination tensor (c0 and c1 share
  // its layout): column cl + dc * (i & 3), row rl + dr * (i >> 2), the four numbers from elements 0, 1 and 4. The M5
  // has dc 1, dr 8; tools/coop_layout checks that a GPU's layout has this form. Reading every element's index instead
  // makes the rescale load 32 scales from threadgroup memory rather than 8: 1-2% slower qmti8s_m64_n128.
  const auto ix0 = c0.get_multidimensional_index(uint16_t(0)), ix1 = c0.get_multidimensional_index(uint16_t(1)),
             ix4 = c0.get_multidimensional_index(uint16_t(4));
  const uint cl = sn * SN + ix0[0], rl = sm * SM + ix0[1], dc = ix1[0] - ix0[0], dr = ix4[1] - ix0[1];
  constexpr int XP = (BM * BK / 16 + NT - 1) / NT, WP = (BN * BK / 32 + NT - 1) / NT;
  uint xr[XP], xk[XP], xrow[XP];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < XP; ++i) {
    const uint j = tid + NT * i;
    xr[i] = j / 4;
    xk[i] = (j % 4) * 16;
    xrow[i] = j < uint(BM * BK / 16) && m0 + xr[i] < p.M ? m0 + xr[i] : 0xffffffffu;
  }
  uint wr[WP], wk[WP], wn[WP];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < WP; ++i) {
    const uint j = tid + NT * i;
    wr[i] = j / 2;
    wk[i] = (j % 2) * 32;
    wn[i] = min(n0 + wr[i], p.N - 1);
  }
  const uint srow = m0 + tid;
  uint4 xv[XP], wv[WP];
  float2 xsv = 0, wsv[WP];
  auto load = [&](uint st) {
    const uint k0 = st * BK;
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < XP; ++i)
      xv[i] = xrow[i] != 0xffffffffu ? *(const device uint4*)(XQ + size_t(xrow[i]) * p.x_ld + k0 + xk[i]) : uint4(0);
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < WP; ++i) {
      wv[i] = *(const device uint4*)(W + size_t(wn[i]) * wld + (k0 + wk[i]) / 8);
      const size_t gi = size_t(wn[i]) * sld + st;
      wsv[i] = float2(float(S[gi]), float(B[gi]));
    }
    if (tid < uint(BM)) xsv = srow < p.M ? XS[size_t(srow) * G + st] : float2(0);
  };
  auto store = [&](uint b, uint sb) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < XP; ++i)
      if (!((BM * BK / 16) % NT) || tid + NT * i < uint(BM * BK / 16))
        *(threadgroup uint4*)(xs + b * BM * BK + xr[i] * BK + xk[i]) = xv[i];
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < WP; ++i) {
      if ((BN * BK / 32) % NT && tid + NT * i >= uint(BN * BK / 32)) continue;
      const uint4 w = wv[i];
      threadgroup uint4* d = (threadgroup uint4*)(ws + b * BN * BK + wr[i] * BK + wk[i]);
      d[0] = uint4(w.x & 0x0F0F0F0Fu, (w.x >> 4) & 0x0F0F0F0Fu, w.y & 0x0F0F0F0Fu, (w.y >> 4) & 0x0F0F0F0Fu);
      d[1] = uint4(w.z & 0x0F0F0F0Fu, (w.z >> 4) & 0x0F0F0F0Fu, w.w & 0x0F0F0F0Fu, (w.w >> 4) & 0x0F0F0F0Fu);
      if (wk[i] == 0) wsc[sb * BN + wr[i]] = wsv[i];
    }
    if (tid < uint(BM)) xsc[sb * BM + tid] = xsv;
  };
  auto rescale = [&](thread decltype(c0)& cc, thread float* acc, uint sb, uint coff) {
    float2 xr4[4], wc4[4];
    _Pragma("clang loop unroll(full)")
    for (int a = 0; a < 4; ++a) xr4[a] = xsc[sb * BM + rl + dr * a];
    _Pragma("clang loop unroll(full)")
    for (int a = 0; a < 4; ++a) wc4[a] = wsc[sb * BN + cl + dc * a + coff];
    _Pragma("clang loop unroll(full)")
    for (uint16_t i = 0; i < CAP; ++i) {
      const float2 xa = xr4[i >> 2], wb = wc4[i & 3];
      acc[i] = fma(float(cc[i]), xa.x * wb.x, fma(xa.y, wb.y, acc[i]));
    }
  };
  load(0);
  store(0, 0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint st = 0; st < G; ++st) {
    const bool more = st + 1 < G;
    const uint b = st & 1, sb = st % 3;
    if (more) load(st + 1);
    _Pragma("clang loop unroll(full)")
    for (uint16_t i = 0; i < CAP; ++i) c0[i] = 0;
    if (b) op.run(tX1, tW10, c0);
    else op.run(tX0, tW00, c0);
    if (st) rescale(c1, a1, (st + 2) % 3, HN);
    _Pragma("clang loop unroll(full)")
    for (uint16_t i = 0; i < CAP; ++i) c1[i] = 0;
    if (b) op.run(tX1, tW11, c1);
    else op.run(tX0, tW01, c1);
    rescale(c0, a0, sb, 0);
    if (more) store(b ^ 1, (st + 1) % 3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  rescale(c1, a1, (G - 1) % 3, HN);
  _Pragma("clang loop unroll(full)")
  for (uint16_t i = 0; i < CAP; ++i) {
    const uint m = m0 + rl + dr * (i >> 2);
    if (m >= p.M) continue;
    const uint n = n0 + cl + dc * (i & 3), n1 = n + HN;
    const size_t o = size_t(m) * p.y_ld + n;
    if (n < p.N) Y[o] = p.add ? Y[o] + a0[i] : a0[i];
    if (n1 < p.N) Y[o + HN] = p.add ? Y[o + HN] + a1[i] : a1[i];
  }
}

#define NI_QMTI8S(NAME, BM, BN)                                                                                    \
  [[kernel, max_total_threads_per_threadgroup(BM * BN / 32)]] void NAME(                                             \
      const device char* XQ [[buffer(0)]], const device uint* W [[buffer(1)]], const device bfloat* S [[buffer(2)]], \
      const device bfloat* B [[buffer(3)]], device float* Y [[buffer(4)]], constant QmtParams& p [[buffer(5)]],      \
      const device float2* XS [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],                             \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {                       \
    threadgroup int8_t xs[2 * BM * 64];                                                                              \
    threadgroup int8_t ws[2 * BN * 64];                                                                              \
    threadgroup float2 xsc[3 * BM];                                                                                  \
    threadgroup float2 wsc[3 * BN];                                                                                  \
    qmti8s_tile<BM, BN>(XQ, XS, W, S, B, Y, p, tg.y * BM, tg.x * BN, tid, sg, xs, ws, xsc, wsc);                    \
  }
// 64 x 128: NI_QMT_I8's tile (enc_qmm_large). Threadgroup memory 29,184 B, past the Metal validation layer's
// doubled-count cap, so the 32 x 64 instantiation (14,592 B) is the one that runs the same code under the layer.
NI_QMTI8S(qmti8s_m64_n128, 64, 128)
NI_QMTI8S(qmti8s_m32_n64, 32, 64)
