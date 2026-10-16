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

} // namespace ni::kp
