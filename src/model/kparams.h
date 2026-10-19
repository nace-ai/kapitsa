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

struct QmmLd {
  Qmm q;
  uint32_t w_ld, s_ld;
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

struct Qmms {
  uint32_t K, T, x_ld, nseg, ks;
  uint32_t N[4], y_ld[4], add[4], tg_begin[5];
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
  uint32_t table_from;  // 0, or: indices >= table_from map through the table at buffer 4 (gathered draft rows)
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

// qmvf.metal: small-batch matvecs with a folded RMSNorm, up to 4 matrices (or a SwiGLU gate / up pair) per dispatch
struct Qmvf {
  uint32_t K, R, x_ld, nseg;
  float eps;
  uint32_t N[4], y_ld[4], add[4], tg_begin[5];
  uint32_t write_xn;
};

// ---- DFlash2 drafter (src/kernels/spec_dflash.metal) ----
struct DflCapture {
  uint32_t H, ld, col0, dst_row0;
};

struct DflEmbed {
  uint32_t H, gs, tok_off, mask;
  float eps;
};

struct DflNorm {
  uint32_t H;
  float eps;
  float xscale;
  uint32_t write_n;
};

struct DflConv {
  uint32_t H, G;
  float eps;
};

struct DflQk {
  uint32_t P;
  float eps;
  uint32_t q_ld, kv_ld;
};

struct DflAttn {
  uint32_t P, lo, chunk, nsplit;
  float scale;
};

struct DflAttnRed {
  uint32_t nsplit;
  uint32_t P, lo, chunk;
};

struct DflCommit {
  uint32_t rows, pos0, kv_ld;
  float eps;
};

struct DflX8 {
  uint32_t K, m, src_ld;
  float scale;
};

struct DflRows {
  uint32_t K;
  float scale;
};

struct DflQmm8 {
  uint32_t K, nseg, ks, x_ld;
  uint32_t tg_begin[5];
  uint32_t N[4];
  uint32_t bits[4];
  uint32_t y_ld[4];
  float oscale[4];
};

struct DflMerge {
  uint32_t n;
};

struct DflTop {
  uint32_t n, ld, nchunks;
};

struct DflSelect {
  uint32_t P;
  float temp;
  uint32_t seed, out_index;
  float qtemp;  // the walk's temperature at temp > 0 (0: argmax walk, one-hot q)
};

struct DflDist {
  uint32_t n, K;
  float temp, top_p;
};

struct DflAccept {
  uint32_t P, K;
  float temp;
  uint32_t seed;
  uint32_t dyn;  // 1: K = min(K, kdyn[0]) (dynamic verify depth)
};

struct DflDepth {
  uint32_t kmax;
  float tau, temp;
  uint32_t rule;  // 0: the engine's rule; 1: own confidence at any T (tests only); 2: the earlier T > 0 rule (dfl_depth)
};

struct DflPerm {
  uint32_t K;
};

// ---- 8-row verify (src/kernels/verify8.metal) ----
struct X8Prep {
  uint32_t K, R, x_ld, norm, write_xn;
  float eps;
  uint32_t xn_ld, write_lo;
  uint32_t unroll;  // x8_prep2 norm: 4 independent sum-of-squares chains
  uint32_t write_t;  // x8_prep2: also the T table (buffer 8) of qmm8t (src/kernels/tensor/verify8t.metal)
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

struct SwigluX8 {
  uint32_t K, R;
};

struct Moe8Plan {
  uint32_t R, k, E, gu_gx, dn_gx, layer, small, sgu_gx, sdn_gx, rlim, split, shs, sep, edn_gx, topk;
};

struct Moe8s {
  uint32_t K, N, w_stride, s_stride, k, x_ld;
};

struct Moe8e {
  uint32_t K, N, w_stride, s_stride, k;
};

struct Moe8Down {
  uint32_t K, N, w_stride, s_stride, h_stride, hs_stride, k;
};

struct Moe8Combine {
  uint32_t N, k;
};

struct Moe8CombineX8 {
  uint32_t N, k, R;
  float eps;
  uint32_t unroll, write_t;
};

} // namespace ni::kp
