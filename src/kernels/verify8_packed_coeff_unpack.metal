#include "common.h"

// Reverse the historical pair-order affine image into the exact original
// Q4 words and BF16 coefficient bits. One thread owns one row's 64-value group.
// This is only a byte-layout conversion; no floating-point arithmetic is used.
[[kernel]] void qmm8pc_unpack_affine(
    const device uint* payload [[buffer(0)]],
    const device uint* headers [[buffer(1)]],
    device uint* weights [[buffer(2)]],
    device ushort* scales [[buffer(3)]],
    device ushort* biases [[buffer(4)]],
    constant uint2& shape [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
  const uint N = shape.x, K = shape.y, groups = K / 64;
  if (!N || !groups || K % 64 || ulong(i) >= ulong(N) * groups) return;
  const uint row = i / groups, g = i % groups;
  const uint block = (row / 16) * ((groups + 1) / 2) + g / 2;
  const device uint* record = payload + block * 272;
  const uint fm = row & 7u, f = (row >> 3) & 1u;
  const uint offset = ((g & 1u) * 2 + f) * 64 + ((fm & 4u) << 3) + ((fm & 3u) << 2);
  const uint4 lo = *(const device uint4*)(record + offset);
  const uint4 hi = *(const device uint4*)(record + offset + 16);
  device uint4* dst = (device uint4*)(weights + ulong(i) * 8);
  dst[0] = lo;
  dst[1] = hi;
  const uint2 sb = v8pc_coeff(payload, record, headers[block], row & 15u);
  const uint shift = (g & 1u) * 16;
  scales[i] = ushort(sb.x >> shift);
  biases[i] = ushort(sb.y >> shift);
}
