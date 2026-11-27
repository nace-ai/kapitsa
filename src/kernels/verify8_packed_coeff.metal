#include "common.h"

// Restore the original BF16 coefficient bits. The CPU encoder proves that
// neither signed halfword sum carries into the next halfword.
NI_INLINE uint2 v8pc_decode(uint code, uint header) {
  const uint sp = (code & 0x80ff80ffu) + header;
  const uint bp = sp + ((code >> 8) & 0x007f007fu) + 0x01630163u;
  return uint2(sp, bp ^ 0x80008000u);
}
NI_INLINE uint2 v8pc_coeff(const device uint* payload, const device uint* record, uint header, uint slot) {
  if (header & 0x80000000u)
    return *(const device uint2*)(payload + (header & 0x7fffffffu) + slot * 2);
  return v8pc_decode(record[256 + slot], header);
}
[[kernel]] void v8pc_check(const device ushort* s [[buffer(0)]], const device ushort* b [[buffer(1)]],
                          const device uint* payload [[buffer(2)]], device uint* out [[buffer(3)]],
                          constant uint2& shape [[buffer(4)]], const device uint* headers [[buffer(5)]],
                          uint i [[thread_position_in_grid]]) {
  if (i >= shape.x * shape.y) return;
  const uint row = i / shape.y, g = i % shape.y;
  const uint block = (row / 16) * ((shape.y + 1) / 2) + g / 2;
  const uint2 sb = v8pc_coeff(payload, payload + block * 272, headers[block], row % 16);
  out[i] = ((sb.x >> (16 * (g & 1u))) & 0xffffu) | (((sb.y >> (16 * (g & 1u))) & 0xffffu) << 16);
}

// Each 1088-byte record contains four lane-contiguous 256-byte Q4 spans and
// 64 bytes of coefficient codes. Headers and uncommon raw slabs stay separate.
NI_INLINE void v8pc_run(const device uint* payload, const device uint* headers, uint first_block,
                        const device half* table, const device float* sums, uint g0, uint g1,
                        uint lane, uint fm, uint fn, uint frag_index, thread float2* acc, uint bstep) {
  // bstep 1: a block row's records in pair order; ks: interleaved for ks k-slices (first_block: this slice's first)
  for (uint g = g0, bi = 0; g < g1; g += 2, bi += bstep) {
    V8Ops2<1> c;
    _Pragma("clang loop unroll(full)")
    for (int h = 0; h < 2; ++h) {
      const device uint4* tb = (const device uint4*)(table + ((g + h) * 32 + lane) * 16);
      c.t[h][0] = tb[0]; c.t[h][1] = tb[1];
      c.sx[h] = *(const device float2*)(sums + (g + h) * kV8 + fn);
    }
    const uint block = first_block + bi, header = headers[block];
    const device uint* record = payload + block * 272;
    // The other eight output channels are owned by a separate SIMD group.
    c.w[0][0] = *(const device uint2*)(record + frag_index * 64 + lane * 2);
    c.w[1][0] = *(const device uint2*)(record + (2 + frag_index) * 64 + lane * 2);
    const uint2 sb = v8pc_coeff(payload, record, header, fm + frag_index * 8);
    c.s[0] = sb.x; c.b[0] = sb.y;
    v8_compute2<1, 1>(c, acc);
  }
}

[[kernel]] void qmm8pc_b4_nf1(
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
    constant uint* ilv [[buffer(25)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float2 red[8 * 32];
  const device uint* Wv[4] = {W0,W1,W2,W3};
  const device uint* Cv[4] = {C0,C1,C2,C3};
  device float* Yv[4] = {Y0,Y1,Y2,Y3};
  uint si = 0;
  while (si + 1 < p.nseg && tg >= p.tg_begin[si + 1]) ++si;
  const uint cs = nsg / p.ks, kslice = sg % p.ks;
  const uint n0 = ((tg - p.tg_begin[si]) * cs + sg / p.ks) * 8;
  const uint N = p.N[si], groups = p.K / 64;
  const uint2 rc = v8_lane(lane);
  const uint gper = (groups + p.ks - 1) / p.ks;
  const uint g0 = min(kslice * gper, groups), g1 = min(g0 + gper, groups);
  float2 acc[1] = {float2(0.0f)};
  // record order (ilv[si], interleave_packed_slices): 0 pairs in order; rg > 0: the k-slices of rg block rows interleaved
  const uint rb = min(n0, N - 1) / 16, pairs = (groups + 1) / 2, rg = ilv[si];
  const uint first_block = rg ? (rb / rg) * rg * pairs + (rb % rg) * p.ks + kslice : rb * pairs + g0 / 2;
  v8pc_run(Wv[si], Cv[si], first_block, table, sums, g0, g1, lane, rc.x, rc.y, (n0 / 8) & 1u, acc, rg ? rg * p.ks : 1u);
  if (p.ks > 1) {
    v8_reduce<1>(acc, red, p.ks, sg, lane);
    if (kslice) return;
  }
  const uint n = n0 + rc.x;
  if (n >= N) return;
  _Pragma("clang loop unroll(full)")
  for (int e = 0; e < 2; ++e) {
    const uint t = rc.y + e;
    if (t < p.R) {
      device float* dst = Yv[si] + t * p.y_ld[si] + n;
      *dst = p.add[si] ? *dst + acc[0][e] : acc[0][e];
    }
  }
}
