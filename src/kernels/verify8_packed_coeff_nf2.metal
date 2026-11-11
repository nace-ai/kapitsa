#include "common.h"

// Two output fragments retain the original NF2 grid and K reduction order.
NI_INLINE void v8pc_run_nf2(const device uint* payload, const device uint* headers, uint first_block,
                            const device half* table, const device float* sums, uint g0, uint g1,
                            uint lane, uint fm, uint fn, thread float2* acc) {
  for (uint g = g0; g < g1; g += 2) {
    V8Ops2<2> c;
    _Pragma("clang loop unroll(full)")
    for (int h = 0; h < 2; ++h) {
      const device uint4* tb = (const device uint4*)(table + ((g + h) * 32 + lane) * 16);
      c.t[h][0] = tb[0]; c.t[h][1] = tb[1];
      c.sx[h] = *(const device float2*)(sums + (g + h) * kV8 + fn);
    }
    const uint block = first_block + g / 2, header = headers[block];
    const device uint* record = payload + block * 272;
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < 2; ++f) {
      c.w[0][f] = *(const device uint2*)(record + f * 64 + lane * 2);
      c.w[1][f] = *(const device uint2*)(record + (2 + f) * 64 + lane * 2);
      const uint2 sb = v8pc_coeff(payload, record, header, fm + f * 8);
      c.s[f] = sb.x; c.b[f] = sb.y;
    }
    v8_compute2<2, 1>(c, acc);
  }
}

[[kernel]] void qmm8pc_b4_nf2(
    const device uint* W0 [[buffer(0)]], const device ushort* S0 [[buffer(1)]],
    const device ushort* B0 [[buffer(2)]], device float* Y0 [[buffer(3)]],
    const device uint* W1 [[buffer(4)]], const device ushort* S1 [[buffer(5)]],
    const device ushort* B1 [[buffer(6)]], device float* Y1 [[buffer(7)]],
    const device uint* W2 [[buffer(8)]], const device ushort* S2 [[buffer(9)]],
    const device ushort* B2 [[buffer(10)]], device float* Y2 [[buffer(11)]],
    const device uint* W3 [[buffer(12)]], const device ushort* S3 [[buffer(13)]],
    const device ushort* B3 [[buffer(14)]], device float* Y3 [[buffer(15)]],
    const device half* table [[buffer(16)]], const device float* sums [[buffer(17)]],
    constant Qmm8Params& p [[buffer(18)]],
    const device uint* C0 [[buffer(21)]], const device uint* C1 [[buffer(22)]],
    const device uint* C2 [[buffer(23)]], const device uint* C3 [[buffer(24)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float2 red[2 * 8 * 32];
  const device uint* Wv[4] = {W0,W1,W2,W3};
  const device uint* Cv[4] = {C0,C1,C2,C3};
  device float* Yv[4] = {Y0,Y1,Y2,Y3};
  uint si = 0;
  while (si + 1 < p.nseg && tg >= p.tg_begin[si + 1]) ++si;
  const uint cs = nsg / p.ks, kslice = sg % p.ks;
  const uint n0 = ((tg - p.tg_begin[si]) * cs + sg / p.ks) * 16;
  const uint N = p.N[si], groups = p.K / 64;
  const uint2 rc = v8_lane(lane);
  const uint gper = (groups + p.ks - 1) / p.ks;
  const uint g0 = min(kslice * gper, groups), g1 = min(g0 + gper, groups);
  float2 acc[2] = {float2(0.0f), float2(0.0f)};
  const uint first_block = (min(n0,N-1) / 16) * ((groups + 1) / 2);
  v8pc_run_nf2(Wv[si], Cv[si], first_block, table, sums, g0, g1, lane, rc.x, rc.y, acc);
  if (p.ks > 1) {
    v8_reduce<2>(acc, red, p.ks, sg, lane);
    if (kslice) return;
  }
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < 2; ++f) {
    const uint n = n0 + f * 8 + rc.x;
    if (n >= N) continue;
    _Pragma("clang loop unroll(full)")
    for (int e = 0; e < 2; ++e) {
      const uint t = rc.y + e;
      if (t < p.R) {
        device float* dst = Yv[si] + t * p.y_ld[si] + n;
        *dst = p.add[si] ? *dst + acc[f][e] : acc[f][e];
      }
    }
  }
}
