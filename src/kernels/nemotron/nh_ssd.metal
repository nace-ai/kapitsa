// ---- Mamba-2's scan of a long pass in chunks of 32 rows as matrix products (the SSD form; M5 prefill) ----
//
// On nh_ssm_conv's values (cv: conv'd x / B / C [T][CH], then (dt, exp(dt A)) [T][NH]), head dim 64, state 128.
// Within a chunk, with lam_i = sum_{k <= i} dt_k A (the chunk's cumulative log decay) and S the state entering it:
//   y_i[p] = sum_{j <= i} M_ij dt_j x_j[p] + exp(lam_i) sum_n C_i[n] S[p][n] + D x_i[p],
//            M_ij = (C_i . B_j) exp(lam_i - lam_j)
//   state after the chunk = exp(lam_end) S + sum_j exp(lam_end - lam_j) dt_j x_j[p] B_j[n]
// (every exp(lam_a - lam_b) as a product of the rows' exp(dt A), as nh_ssm applies them)
// the same recurrence as nh_ssm's step by step, its sums as fp32 8 x 8 simdgroup products. nh_ssd_m computes every
// chunk's M dt (M'), exp(lam_i) and exp(lam_end - lam_j) dt_j at once (a threadgroup per (group, chunk): C B^T once
// for the group's heads); nh_ssd_scan walks the chunks in order with the state in registers (a threadgroup per (head,
// 32 state rows; a simdgroup per 8 of them), a chunk's x / B / C staged once in threadgroup memory.
// Sequential depth T / 32 chunk steps instead of T row steps.
// sb per (head, chunk): M' [32][32], exp(lam) [32], exp(lam_end - lam_j) dt_j [32].

constant constexpr uint kSdL = 32, kSdB = kSdL * kSdL + 2 * kSdL;  // rows per chunk, sb floats per (head, chunk)
struct NhSsdP {
  uint nch;
};

// grid: NG x nch threadgroups x 128 threads
[[kernel, max_total_threads_per_threadgroup(128)]] void nh_ssd_m(
    const device float* A [[buffer(6)]], constant NhSsmParams& p [[buffer(9)]], const device float* cv [[buffer(11)]],
    device float* sb [[buffer(12)]], constant NhSsdP& sp [[buffer(13)]], uint tgi [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float cs[kSdL * 64], bs[kSdL * 64];  // C / B rows, half of the state per pass
  threadgroup float gm[kSdL * kSdL];
  const uint DI = p.NH * p.HD, CH = DI + 2 * p.NG * p.DS;
  const uint g = tgi / sp.nch, c = tgi % sp.nch, t0 = c * kSdL;
  const uint cb = DI + g * 128, cc = DI + p.NG * 128 + g * 128;
  const uint qd = lane / 4, fr = (qd & 4) + (lane / 2) % 4, fc = (qd & 2) * 2 + (lane % 2) * 2;
  // G = C B^T [32][32]: simdgroup sg the rows [8 sg, 8 sg + 8)
  simdgroup_float8x8 gt[4];
  _Pragma("clang loop unroll(full)")
  for (int q = 0; q < 4; ++q) gt[q] = simdgroup_float8x8(0);
  _Pragma("clang loop unroll(full)")
  for (uint hf = 0; hf < 2; ++hf) {
    for (uint k = tid; k < kSdL * 64; k += 128) {
      const uint i = k / 64, t = t0 + i, n = hf * 64 + k % 64;
      cs[k] = t < p.T ? cv[size_t(t) * CH + cc + n] : 0.0f;
      bs[k] = t < p.T ? cv[size_t(t) * CH + cb + n] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < 64; k += 8) {
      simdgroup_float8x8 a;
      simdgroup_load(a, cs + 8 * sg * 64 + k, 64);
      _Pragma("clang loop unroll(full)")
      for (int q = 0; q < 4; ++q) {
        simdgroup_float8x8 b;
        simdgroup_load(b, bs + 8 * q * 64 + k, 64, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(gt[q], a, b, gt[q]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  _Pragma("clang loop unroll(full)")
  for (int q = 0; q < 4; ++q)
    _Pragma("clang loop unroll(full)")
    for (int e = 0; e < 2; ++e) gm[(8 * sg + fr) * kSdL + 8 * q + fc + e] = gt[q].thread_elements()[e];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // per head of the group (simdgroup sg: heads sg, sg + 4, ...): lane i = row i
  const device float* dta = cv + size_t(p.T) * CH;
  const uint hpg = p.NH / p.NG, t = t0 + lane;
  for (uint hh = sg; hh < hpg; hh += 4) {
    const uint h = g * hpg + hh;
    // the decays as products of the rows' exp(dt A) (nh_ssm's factors; 1 past T): exp of differences of cumulative
    // sums lost up to 1e-4 of a fast-decaying head's factors (|lam| ~ 1e3; tools/nh_pf_bench ssm). (exp of the sums
    // for the decays from the chunk's start, more accurate against fp64 for slow heads, moved Lightning's logits
    // further from nh_ssm's after an 86-token prefill: KL mean 5.2e-4 against 1.2e-4, max 0.030 against 0.009)
    const float dt = t < p.T ? dta[(size_t(t) * p.NH + h) * 2] : 0.0f;
    const float dA = t < p.T ? dta[(size_t(t) * p.NH + h) * 2 + 1] : 1.0f;
    device float* mh = sb + (size_t(h) * sp.nch + c) * kSdB;
    float run = 1.0f, wd = 1.0f;  // prod_{j < k <= lane} dA_k, prod_{lane < k < 32} dA_k
    for (int j = int(kSdL) - 1; j >= 0; --j) {
      const float dj = simd_shuffle(dt, ushort(j)), aj = simd_shuffle(dA, ushort(j));
      mh[lane * kSdL + uint(j)] = uint(j) <= lane ? gm[lane * kSdL + uint(j)] * run * dj : 0.0f;
      if (uint(j) <= lane) run *= aj;
      if (uint(j) > lane) wd *= aj;
    }
    mh[kSdL * kSdL + lane] = run;                     // prod_{k <= i} dA_k
    mh[kSdL * kSdL + kSdL + lane] = wd * dt;           // prod_{j < k} dA_k dt_j
  }
}

// grid: NH x 64 / (8 NS) threadgroups x 32 NS threads (NS <= 8: rows [8 NS tg, 8 NS tg + 8 NS) of the (head, state
// row) rows, a simdgroup per 8; NS 8: a head per threadgroup); a chunk's x / B / C rows staged once for them
template <uint NS>
NI_INLINE void nh_ssd_scan_impl(const device float* zx, device float* state, device float* y, const device float* Dsk,
                                 constant NhSsmParams& p, const device float* cv, const device float* sb,
                                 constant NhSsdP& sp, uint tgi, uint tid, uint sg, uint lane, threadgroup float* cs,
                                 threadgroup float* bs, threadgroup float* xt, threadgroup float* xw) {
  constexpr uint NT = 32 * NS, RW = 8 * NS;
  const uint DI = p.NH * p.HD, CH = DI + 2 * p.NG * p.DS;
  // the threadgroup's state rows [R0, R0 + RW) of the (head, state row) rows
  const uint R0 = tgi * RW, g = (R0 / 64) / (p.NH / p.NG), h = (R0 + 8 * sg) / 64;  // h: this simdgroup's head
  const uint cb = DI + g * 128, cc = DI + p.NG * 128 + g * 128, cx = R0;  // B, C, x channels
  const uint qd = lane / 4, fr = (qd & 4) + (lane / 2) % 4, fc = (qd & 2) * 2 + (lane % 2) * 2;
  device float* stp = state + (size_t(R0) + 8 * sg) * 128;
  simdgroup_float8x8 st[16];  // the state's rows [8 sg, 8 sg + 8) of this block, 16 column tiles
  _Pragma("clang loop unroll(full)")
  for (int q = 0; q < 16; ++q) simdgroup_load(st[q], stp + 8 * q, 128);
  const float Dh = Dsk[h];
  for (uint c = 0; c < sp.nch; ++c) {
    const uint t0 = c * kSdL;
    const device float* mc = sb + (size_t(h) * sp.nch + c) * kSdB;
    for (uint k = tid; k < kSdL * RW; k += NT) {
      const uint j = k / RW, t = t0 + j;
      const float v = t < p.T ? cv[size_t(t) * CH + cx + k % RW] : 0.0f;
      xt[k] = v;
      xw[k] = sb[(size_t((R0 + k % RW) / 64) * sp.nch + c) * kSdB + kSdL * kSdL + kSdL + j] * v;
    }
    simdgroup_float8x8 yi[4], ye[4];
    _Pragma("clang loop unroll(full)")
    for (int q = 0; q < 4; ++q) yi[q] = ye[q] = simdgroup_float8x8(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Yi = x^T M'^T (M' read from sb)
    _Pragma("clang loop unroll(full)")
    for (uint k = 0; k < kSdL; k += 8) {
      simdgroup_float8x8 a;
      simdgroup_load(a, xt + k * RW + 8 * sg, RW, ulong2(0, 0), true);
      _Pragma("clang loop unroll(full)")
      for (int q = 0; q < 4; ++q) {
        simdgroup_float8x8 b;
        simdgroup_load(b, mc + 8 * q * kSdL + k, kSdL, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(yi[q], a, b, yi[q]);
      }
    }
    const float ed = mc[kSdL * kSdL + kSdL - 1];
    _Pragma("clang loop unroll(full)")
    for (uint hf = 0; hf < 2; ++hf) {
      for (uint k = tid; k < kSdL * 64; k += NT) {
        const uint i = k / 64, t = t0 + i, n = hf * 64 + k % 64;
        cs[k] = t < p.T ? cv[size_t(t) * CH + cc + n] : 0.0f;
        bs[k] = t < p.T ? cv[size_t(t) * CH + cb + n] : 0.0f;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // Ye += S C^T over this half's 8 column tiles (the state entering the chunk), then those tiles updated:
      // S = exp(lam_end) S + xw^T B
      _Pragma("clang loop unroll(full)")
      for (int q = 0; q < 8; ++q)
        _Pragma("clang loop unroll(full)")
        for (int it = 0; it < 4; ++it) {
          simdgroup_float8x8 b;
          simdgroup_load(b, cs + 8 * it * 64 + 8 * q, 64, ulong2(0, 0), true);
          simdgroup_multiply_accumulate(ye[it], st[8 * hf + q], b, ye[it]);
        }
      _Pragma("clang loop unroll(full)")
      for (int q = 0; q < 8; ++q) {
        st[8 * hf + q].thread_elements()[0] *= ed;
        st[8 * hf + q].thread_elements()[1] *= ed;
      }
      _Pragma("clang loop unroll(full)")
      for (uint k = 0; k < kSdL; k += 8) {
        simdgroup_float8x8 a;
        simdgroup_load(a, xw + k * RW + 8 * sg, RW, ulong2(0, 0), true);
        _Pragma("clang loop unroll(full)")
        for (int q = 0; q < 8; ++q) {
          simdgroup_float8x8 b;
          simdgroup_load(b, bs + k * 64 + 8 * q, 64);
          simdgroup_multiply_accumulate(st[8 * hf + q], a, b, st[8 * hf + q]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // y = (Yi + exp(lam_i) Ye + D x) silu(z): element (row p = 8 sg + fr, column i = 8 it + fc + e)
    const uint pp = cx + 8 * sg + fr;
    _Pragma("clang loop unroll(full)")
    for (int it = 0; it < 4; ++it)
      _Pragma("clang loop unroll(full)")
      for (int e = 0; e < 2; ++e) {
        const uint i = 8 * it + fc + e, t = t0 + i;
        if (t >= p.T) continue;
        const float v = yi[it].thread_elements()[e] + mc[kSdL * kSdL + i] * ye[it].thread_elements()[e];
        const float xv = xt[i * RW + 8 * sg + fr], z = zx[size_t(t) * p.zx_stride + pp];
        y[size_t(t) * DI + pp] = (v + Dh * xv) * silu(z);
      }
    // (every simdgroup's reads of xt above before the next chunk overwrites it: without this barrier a simdgroup
    // ahead staged the next chunk's x under a slower one's output loop, and a prefill's y differed from run to run)
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (p.write)
    _Pragma("clang loop unroll(full)")
    for (int q = 0; q < 16; ++q) simdgroup_store(st[q], stp + 8 * q, 128);
}
#define NH_SSD_SCAN(NAME, NS)                                                                                      \
  [[kernel, max_total_threads_per_threadgroup(32 * NS)]] void NAME(                                               \
      const device float* zx [[buffer(0)]], device float* state [[buffer(2)]], device float* y [[buffer(3)]],    \
      const device float* Dsk [[buffer(7)]], constant NhSsmParams& p [[buffer(9)]],                             \
      const device float* cv [[buffer(11)]], const device float* sb [[buffer(12)]],                             \
      constant NhSsdP& sp [[buffer(13)]], uint tgi [[threadgroup_position_in_grid]],                            \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],                    \
      uint lane [[thread_index_in_simdgroup]]) {                                                               \
    threadgroup float cs[kSdL * 64], bs[kSdL * 64], xt[kSdL * 8 * NS], xw[kSdL * 8 * NS];                         \
    nh_ssd_scan_impl<NS>(zx, state, y, Dsk, p, cv, sb, sp, tgi, tid, sg, lane, cs, bs, xt, xw);                 \
  }
// a head per threadgroup (2048 rows: 0.77 ms per Lightning layer; 32 state rows per threadgroup 0.89, 16 rows 1.41)
NH_SSD_SCAN(nh_ssd_scan, 8)
