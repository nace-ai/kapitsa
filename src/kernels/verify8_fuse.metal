#include "common.h"

// ---- 8-row verify: fused small passes (fewer dependent dispatches per layer) ----
// Each kernel here does the work of two or more verify8.metal passes in one dispatch, with the same arithmetic in
// the same order (bitwise the same outputs): a dependent dispatch costs its launch plus a pipeline drain on top of
// its own latency, and the verify chains ~600 of them.

// MoE combine + the next X8 table: x[t] += sum_j w[t][j] out[t k + j] + w[t][k] sh_out[t] (moe8_combine's order),
// then x8_prep2's RMSNorm table of the new rows (its sum-of-squares order, its group-sum order). One threadgroup per
// token row (the norm needs the whole row): the combined row is staged in threadgroup memory, simdgroup 0 forms the
// row's norm, every thread then writes 8 consecutive inputs into the table (2-byte stores: a row is every other half
// of the table's 16-byte lane runs). Rows t >= R get zeros. With write_t, also the row of qmm8t's T table (x8_prep2's
// write_t; the halfs staged in threadgroup memory, 16-byte stores). grid 8 x N / 4 threads (N <= 2048).
struct Moe8CombineX8Params {
  uint N, k;     // hidden, routed experts per row
  uint R;        // live rows
  float eps;
  uint unroll;   // x8_prep2's unroll (V8Knobs::prepu): 4 independent sum-of-squares chains
  uint write_t;  // 1: also the T table (buffer 8)
};
[[kernel]] void moe8_combine_x8(const device float4* out [[buffer(0)]], const device float4* sh_out [[buffer(1)]],
                                const device float* wts [[buffer(2)]], device float4* x [[buffer(3)]],
                                const device bfloat* w [[buffer(4)]], device half* table [[buffer(5)]],
                                device float* sums [[buffer(6)]], constant Moe8CombineX8Params& p [[buffer(7)]],
                                device half* table_t [[buffer(8)]], uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                uint ntg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 xs[512];
  threadgroup half hs[2048];
  threadgroup float inv_s;
  const uint n4 = p.N / 4, K = p.N;
  const bool live = t < p.R;
  if (live) {
    // moe8_combine's code as it is (one float4 per thread): written otherwise, the compiler orders the sum otherwise
    const uint i = tid;
    if (i < n4) {
      float4 acc = float4(0.0f);
      for (uint j = 0; j < p.k; ++j) acc += wts[t * (p.k + 1) + j] * out[(t * p.k + j) * n4 + i];
      acc += wts[t * (p.k + 1) + p.k] * sh_out[t * n4 + i];
      x[t * n4 + i] += acc;
      xs[i] = x[t * n4 + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float ss = 0.0f;
      if (p.unroll && K % 512 == 0) {
        float4 a0 = float4(0.0f), a1 = float4(0.0f), a2 = float4(0.0f), a3 = float4(0.0f);
        for (uint i = lane; i < K / 4; i += 128) {
          const float4 v0 = xs[i], v1 = xs[i + 32], v2 = xs[i + 64], v3 = xs[i + 96];
          a0 = fma(v0, v0, a0);
          a1 = fma(v1, v1, a1);
          a2 = fma(v2, v2, a2);
          a3 = fma(v3, v3, a3);
        }
        const float4 a = (a0 + a1) + (a2 + a3);
        ss = (a.x + a.y) + (a.z + a.w);
      } else {
        for (uint i = lane; i < K / 4; i += 32) {
          const float4 v = xs[i];
          ss += dot(v, v);
        }
      }
      ss = simd_sum(ss);
      if (lane == 0) inv_s = metal::precise::rsqrt(ss / float(K) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // thread q < K / 8: inputs 8q .. 8q + 7; a group's 8 octets are 8 consecutive lanes of one simdgroup, added in
  // x8_prep2's order. A thread's 8 halfs are summed in sequence with an identity shuffle between the additions: fast
  // math would otherwise add them as two 4-vectors in some contexts (a different rounding).
  const float inv = live ? inv_s : 0.0f;
  const threadgroup float* xf = (const threadgroup float*)xs;
  const uint q = tid, k0 = q * 8;
  const bool on = q < K / 8;
  float v[8];
  for (int i = 0; i < 8; ++i) v[i] = 0.0f;
  if (live && on)
    for (int i = 0; i < 8; ++i) v[i] = xf[k0 + i] * inv * float(w[k0 + i]);
  half hv[8];
  for (int i = 0; i < 8; ++i) hv[i] = half(v[i]);
  float sh = float(hv[0]);
  for (int i = 1; i < 8; ++i) sh = simd_shuffle(sh, ushort(lane)) + float(hv[i]);
  sh += simd_shuffle_xor(sh, 1u);
  sh += simd_shuffle_xor(sh, 2u);
  sh += simd_shuffle_xor(sh, 4u);
  if (on) {
    const uint g = k0 / 64;
    if ((lane & 7) == 0) sums[g * kV8 + t] = sh;
    for (int i = 0; i < 8; ++i) table[v8_tab(g, (k0 % 64) + i, t)] = hv[i];
  }
  if (p.write_t) {  // slots 8q .. 8q + 7 of row t
    if (on)
      for (int i = 0; i < 8; ++i) hs[k0 + i] = hv[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (on) {
      const uint gb = k0 & ~63u;
      half o[8];
      for (int i = 0; i < 8; ++i) o[i] = hs[gb + v8t_input_of_slot((k0 + i) & 63)];
      *(device uint4*)(table_t + t * K + k0) = uint4(as_type<uint>(half2(o[0], o[1])), as_type<uint>(half2(o[2], o[3])),
                                                    as_type<uint>(half2(o[4], o[5])), as_type<uint>(half2(o[6], o[7])));
    }
  }
}

// gdn_prep_verify with the rows' q / k norms reduced together: each row's conv output first (all R in registers), then
// one threadgroup exchange for the R sums of squares (tg_sum's two stages per row: simd_sum, then the simd_sum of the
// simdgroups' partials) instead of R tg_sums in sequence (two barriers each). Bitwise gdn_prep_verify.
// grid: C/128 threadgroups x 128 threads; R <= 8.
[[kernel]] void gdn_prep_verify8(const device float* qkv [[buffer(0)]], const device float* a [[buffer(1)]],
                                 const device float* bb [[buffer(2)]], device float* conv_state [[buffer(3)]],
                                 const device bfloat* conv_w [[buffer(4)]], const device bfloat* A_log [[buffer(5)]],
                                 const device bfloat* dt_bias [[buffer(6)]], device float* q_out [[buffer(7)]],
                                 device float* k_out [[buffer(8)]], device float* v_out [[buffer(9)]],
                                 device float* g_out [[buffer(10)]], device float* beta_out [[buffer(11)]],
                                 device float* pend_qkv [[buffer(12)]], constant GdnVerifyParams& p [[buffer(13)]],
                                 uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[8][4];
  constexpr uint D = 128;
  const uint c = tg * D + tid;
  const uint C = p.C;
  const uint qk = p.Hk * D;
  const bool is_q = c < qk, is_k = c >= qk && c < 2 * qk;
  float in[8];
  for (uint t = 0; t < 8; ++t) in[t] = t < p.R ? qkv[t * C + c] : 0.0f;
  // fold pending rows into the committed window
  float w3[3] = {conv_state[0 * C + c], conv_state[1 * C + c], conv_state[2 * C + c]};
  for (uint t = 0; t < p.n_pend; ++t) {
    w3[0] = w3[1];
    w3[1] = w3[2];
    w3[2] = pend_qkv[t * C + c];
  }
  for (uint r = 0; r < 3; ++r) conv_state[r * C + c] = w3[r];
  float w0 = conv_w[c * 4 + 0], w1 = conv_w[c * 4 + 1], w2 = conv_w[c * 4 + 2], wq = conv_w[c * 4 + 3];
  float h0 = w3[0], h1 = w3[1], h2 = w3[2];
  float o[8];
  for (uint t = 0; t < 8; ++t) {
    if (t >= p.R) break;
    float in3 = in[t];
    pend_qkv[t * C + c] = in3;
    o[t] = silu(w0 * h0 + w1 * h1 + w2 * h2 + wq * in3);
    h0 = h1;
    h1 = h2;
    h2 = in3;
  }
  if (is_q || is_k) {  // (uniform per threadgroup: 128 channels = one head)
    for (uint t = 0; t < 8; ++t) {
      if (t >= p.R) break;
      const float v = simd_sum(o[t] * o[t]);
      if (lane == 0) scratch[t][sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < 8; ++t) {
      if (t >= p.R) break;
      float r = lane < D / 32 ? scratch[t][lane] : 0.0f;
      r = simd_sum(r);
      const float ot = o[t] * metal::precise::rsqrt(r + 1e-6f);
      if (is_q) q_out[t * qk + c] = ot * metal::precise::rsqrt(float(D));
      else k_out[t * qk + (c - qk)] = ot;
    }
  } else {
    for (uint t = 0; t < 8; ++t) {
      if (t >= p.R) break;
      v_out[t * (C - 2 * qk) + (c - 2 * qk)] = o[t];
    }
  }
  if (tg == 0 && tid < p.Hv) {
    float A = metal::precise::exp(float(A_log[tid]));
    float dtb = float(dt_bias[tid]);
    for (uint t = 0; t < p.R; ++t) {
      g_out[t * p.Hv + tid] = metal::precise::exp(-A * softplus(a[t * p.Hv + tid] + dtb));
      beta_out[t * p.Hv + tid] = sigmoid(bb[t * p.Hv + tid]);
    }
  }
}
