// Shared pieces of the Nemotron-H kernels: NVFP4 / IQ2_XXS decoding, parameter structs used by several files. (Headers
// come before the .metal files in the embedded source; the IQ2_XXS lattice nh_iq2xxs_grid is in ../iq.h, before them.)

// NVFP4 values doubled (E2M1 x 2); nh_ue4m3 halves the scale to match
constant float nh_fp4_lut[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

NI_INLINE float nh_ue4m3(uint b) {
  // UE4M3 (bias 7) times 0.5: as_type<half>(b << 7) is 2^-8 of the E4M3 value
  // (normal and subnormal alike); 0x7F (NaN) and 0 are 0
  return (b == 0x7Fu) ? 0.0f : float(as_type<half>(ushort(b << 7))) * 128.0f;
}


NI_INLINE void nh_iq2_grid_to_tg(threadgroup ulong* tgrid, uint tid, uint ntg) {
  for (uint i = tid; i < 256; i += ntg) tgrid[i] = nh_iq2xxs_grid[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// 8 weights (one grid point) of an IQ2_XXS sub-block, times db = (0.5 + s4) / 4 (the block's d is applied later)
NI_INLINE void nh_iq2_decode8(uint idx, uint s7, float db, threadgroup const ulong* tgrid, thread float* w) {
  const ulong g = tgrid[idx];
  const uint sg = s7 | ((popcount(s7) & 1u) << 7);
  for (int i = 0; i < 8; ++i) {
    const float m = float(uint(g >> (8 * i)) & 0xFFu) * db;
    w[i] = (sg >> i) & 1u ? -m : m;
  }
}


struct NhGnormParams {
  uint D, G;
  float eps;
};

// (the KFlash tree descriptor's layout, kKf*, comes from src/kernels/kflash.h, shared with every backend)
