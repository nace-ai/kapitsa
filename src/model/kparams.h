// Host mirrors of the kernel parameter structs (must match src/kernels/*.metal field for field).
#pragma once
#include <cstdint>

namespace ni::kp {

struct Qmv {
  uint32_t N, K, x_stride, y_stride;
  float eps;
};

struct QmvMulti {
  uint32_t K, nseg;
  float eps;
  uint32_t N[4];
  uint32_t tg_begin[5];
  uint32_t bits[4];
};

struct QmvMulti2 {
  uint32_t K, nseg;
  float eps;
  uint32_t norm, write_xn;
  uint32_t N[4];
  uint32_t tg_begin[5];
  uint32_t bits[4];
};

struct Embed {
  uint32_t D, gs, tok_off;
};

struct GdnPrep {
  uint32_t T, Hk, Hv, C;
};

struct GdnStep {
  uint32_t T, Hk, Hv;
  float eps;
};

struct AttnPrep {
  uint32_t T, Hq, Hkv, D, rot, pos, max_seq;
  float eps, theta;
};

struct AttnDec {
  uint32_t Hq, Hkv, L, chunk, max_seq;
  float scale;
  uint32_t nchunks;
};

struct Topk {
  uint32_t E, k, norm;
};

struct Moe {
  uint32_t D, I, SI, k;
  float eps;
};

struct TopkStage {
  uint32_t V, K, chunk;
};

struct Sample {
  uint32_t n, K;
  float temp, top_p;
  uint32_t seed, step, out_index;
};

struct Qmm {
  uint32_t M, N, K, x_ld, y_ld, add, gs, w_expert_stride, s_expert_stride;
};

struct AttnPref {
  uint32_t T, Hq, Hkv, pos0, max_seq;
  float scale;
};

struct Bucket {
  uint32_t P, k, E, BM;
};

struct Bucket2 {
  uint32_t P, k, E, BM, S;
};

struct Qmm2 {
  uint32_t N, K, x_ld, y_ld, gs, w_expert_stride, s_expert_stride, shared_e;
};

struct Rows {
  uint32_t D, norm;
  float eps;
};

struct Combine {
  uint32_t D, k;
};

struct RowsNorm {
  uint32_t D;
  float eps;
};

struct Qmvr {
  uint32_t N, K, R, x_ld, y_ld, add;
};

struct AttnRows {
  uint32_t Hq, Hkv, pos0, chunk, max_seq;
  float scale;
  uint32_t nchunks;
};

struct MoeRowsDown {
  uint32_t D, I, SI, k, shared_bits;
};

struct GdnVerify {
  uint32_t R, n_pend, Hk, Hv, C;
  float eps;
};

struct SampleRows {
  uint32_t n, K;
  float temp, top_p;
  uint32_t seed, step, out_index, remap_from, remap_base;
};

struct QmvrMulti {
  uint32_t K, R, nseg, norm, write_xn;
  float eps;
  uint32_t N[4];
  uint32_t tg_begin[5];
  uint32_t bits[4];
  uint32_t add[4];
};

struct QmvrcMulti {
  uint32_t K, R, nseg, norm, write_xn;
  float eps;
  uint32_t N[4];
  uint32_t tg_begin[5];
  uint32_t bits[4];
};


// ---- 8-row verify (src/kernels/verify8.metal) ----
struct X8Prep {
  uint32_t K, R, x_ld, norm, write_xn;
  float eps;
  uint32_t xn_ld, write_lo;
};

struct Qmm8 {
  uint32_t K, R, nseg, ks;
  uint32_t N[4];
  uint32_t y_ld[4];
  uint32_t add[4];
  uint32_t tg_begin[5];
};

struct Qmm8Gu {
  uint32_t K, N, w_stride, s_stride, moe, h_stride, hs_stride;
};

struct Moe8Plan {
  uint32_t R, k, E, gu_gx, dn_gx, layer, small, sgu_gx, sdn_gx;
};

struct Moe8s {
  uint32_t K, N, w_stride, s_stride, k, x_ld;
};

struct Moe8Down {
  uint32_t K, N, w_stride, s_stride, h_stride, hs_stride, k;
};

struct Moe8Combine {
  uint32_t N, k;
};

struct Capture8 {
  uint32_t D, ld, slot;
};

} // namespace ni::kp
