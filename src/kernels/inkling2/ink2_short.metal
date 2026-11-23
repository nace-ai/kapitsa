NI_INLINE float ink2_short_nvfp4_dot(uint2 pw, const thread float* xf) {
  const thread uchar* bw = (const thread uchar*)&pw;
  float d = 0.0f;
  for (uint b = 0; b < 8; ++b) d += kE2M1[bw[b] & 15u] * xf[2 * b] + kE2M1[bw[b] >> 4] * xf[2 * b + 1];
  return d;
}

[[kernel, max_total_threads_per_threadgroup(128)]] void ink2_moe_nvfp4_w13_shared(
    const device float* x [[buffer(0)]], const device uint* eid [[buffer(1)]],
    const device uchar* w13 [[buffer(2)]], const device uchar* sc [[buffer(3)]],
    const device float* outer [[buffer(4)]], device float* act [[buffer(5)]],
    constant Ink2MoeP& p [[buffer(6)]], const device uint* sorted [[buffer(7)]],
    const device uint4* tiles [[buffer(8)]], const device uint* ntiles [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (tg.y >= ntiles[0]) return;
  const uint j0 = tg.x * 4u;
  if (j0 >= p.I) return;
  const uint nr = min(4u, p.I - j0);
  const uint4 tile = tiles[tg.y];
  const uint e = tile.x;
  const uint two_i = 2u * p.I, half_h = p.H / 2u, blocks = p.H / 16u;
  const size_t wg0_row = size_t(e) * two_i + 2u * j0;
  const uint tid = sg * 32u + lane;
  threadgroup uint2 packed[8 * 144];
  threadgroup uchar scales[8 * 144];
  for (uint i = tid; i < 2u * nr * blocks; i += 128u) {
    const uint wr = i / blocks, blk = i % blocks;
    packed[wr * 144u + blk] = *(const device uint2*)(w13 + (wg0_row + wr) * half_h + blk * 8u);
    scales[wr * 144u + blk] = sc[(wg0_row + wr) * blocks + blk];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint row = sg; row < tile.z; row += 4u) {
    const uint pair = sorted[tile.y + row];
    const uint t = pair / p.K_top, slot = pair % p.K_top;
    const device float* xp = x + size_t(t) * p.H;
    float ag[4] = {0.0f, 0.0f, 0.0f, 0.0f}, au[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint blk = lane; blk < blocks; blk += 32u) {
      const uint kbase = blk * 16u;
      const float4 xv[4] = {
          *(const device float4*)(xp + kbase + 0u), *(const device float4*)(xp + kbase + 4u),
          *(const device float4*)(xp + kbase + 8u), *(const device float4*)(xp + kbase + 12u),
      };
      const thread float* xf = (const thread float*)xv;
      for (uint q = 0u; q < nr; ++q) {
        ag[q] += ink2_short_nvfp4_dot(packed[(2u * q) * 144u + blk], xf) * ink2_e4m3(scales[(2u * q) * 144u + blk]);
        au[q] += ink2_short_nvfp4_dot(packed[(2u * q + 1u) * 144u + blk], xf) * ink2_e4m3(scales[(2u * q + 1u) * 144u + blk]);
      }
    }
    const float os = outer[e];
    const size_t obase = (size_t(t) * p.K_top + slot) * p.I + j0;
    for (uint q = 0u; q < nr; ++q) {
      const float g = simd_sum(ag[q]) * os, u = simd_sum(au[q]) * os;
      if (lane == 0u) {
        act[obase + q] = (4.0f * metal::precise::tanh(g * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g))) *
                         (25.0f * metal::precise::tanh(u * 0.04f));
      }
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(128)]] void ink2_moe_mxfp8_w13_shared(
    const device float* x [[buffer(0)]], const device uint* eid [[buffer(1)]],
    const device uchar* w13 [[buffer(2)]], const device uchar* sc [[buffer(3)]],
    device float* act [[buffer(4)]], constant Ink2MoeP& p [[buffer(5)]],
    const device uint* sorted [[buffer(6)]], const device uint4* tiles [[buffer(7)]],
    const device uint* ntiles [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  if (tg.y >= ntiles[0]) return;
  const uint j = tg.x;
  if (j >= p.I) return;
  const uint4 tile = tiles[tg.y];
  const uint e = tile.x;
  const uint two_i = 2 * p.I;
  const uint blocks = p.H / 32, chunks = p.H / 16;
  const size_t wg_row = size_t(e) * two_i + 2 * j;
  const uint tid = sg * 32u + lane;
  threadgroup uint4 packed[2 * 144];
  threadgroup uchar scales[2 * 72];
  for (uint i = tid; i < 2u * chunks; i += 128u) {
    const uint wr = i / chunks, chunk = i % chunks;
    packed[wr * 144u + chunk] = *(const device uint4*)(w13 + (wg_row + wr) * p.H + chunk * 16u);
  }
  for (uint i = tid; i < 2u * blocks; i += 128u) {
    const uint wr = i / blocks, blk = i % blocks;
    scales[wr * 72u + blk] = sc[(wg_row + wr) * blocks + blk];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const threadgroup uchar* weights = (const threadgroup uchar*)packed;

  for (uint row = sg; row < tile.z; row += 4u) {
    const uint pair = sorted[tile.y + row];
    const uint t = pair / p.K_top, slot = pair % p.K_top;
    const device float* xp = x + size_t(t) * p.H;
    float ag = 0.0f, au = 0.0f;
    for (uint blk = lane; blk < blocks; blk += 32) {
      const float sg_b = ink2_ue8m0(scales[blk]);
      const float su_b = ink2_ue8m0(scales[72u + blk]);
      const threadgroup uchar* pg = weights + blk * 32;
      const threadgroup uchar* pu = weights + 2304u + blk * 32;
      const uint kbase = blk * 32;
      float dg = 0.0f, du = 0.0f;
      for (uint b = 0; b < 32; ++b) {
        const float xv = xp[kbase + b];
        dg += ink2_e4m3(pg[b]) * xv;
        du += ink2_e4m3(pu[b]) * xv;
      }
      ag += dg * sg_b;
      au += du * su_b;
    }
    const float g = simd_sum(ag);
    const float u = simd_sum(au);
    if (lane == 0) {
      const float out_v =
          (4.0f * metal::precise::tanh(g * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g))) *
          (25.0f * metal::precise::tanh(u * 0.04f));
      act[(size_t(t) * p.K_top + slot) * p.I + j] = out_v;
    }
  }
}
