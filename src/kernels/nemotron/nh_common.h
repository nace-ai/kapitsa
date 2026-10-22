// Shared pieces of the Nemotron-H kernels: NVFP4 / IQ2_XXS decoding, parameter structs used by several files. (Headers
// come before the .metal files in the embedded source; the IQ2_XXS lattice nh_iq2xxs_grid is in ../iq.h, before them.)

// Device-scope fences (atomic_thread_fence with memory_order_seq_cst, thread_scope_device) come with Metal 3.2, which
// macOS 14 does not have (Gpu::compile_library: 3.1 there). Without them (NH_FENCES 0) the kernels that merge in the
// threadgroup to arrive last compile without the merge: the fused one-token MoE layers' routing (nh_moe_a1, nh_moe_a4,
// nh_moe_ah) is left out, nh_x4_moe_dn_sum writes only its partials and the split-K GEMMs (nh_x4_mm*, nh_x8_mm*)
// write nothing for SK > 1. The engine does not run those then (NemotronH::split_merges(): macOS 14, or
// NI_NH_SPLIT=1): the separate routing kernels, nh_x4_moe_dn + nh_moe2_sum2 and SK = 1. With Metal 3.2+ every kernel
// is the same source as before.
#if __METAL_VERSION__ >= 320
#define NH_FENCES 1
#else
#define NH_FENCES 0
#endif

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
