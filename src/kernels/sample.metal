#include "common.h"

// ---- On-GPU sampling: two-stage top-k, then temperature / top-p sampling ----
// The sampled token id is written straight into the token buffer so the next step's embedding
// lookup reads it without a CPU round trip.

struct TopkStageParams {
  uint V;      // vocab
  uint K;      // candidates kept per threadgroup (<= 64)
  uint chunk;  // logits per threadgroup (threads * 8)
};

NI_INLINE void tg_argmax(thread float& v, thread uint& i, threadgroup float* sv, threadgroup uint* si, uint tid,
                         uint ntg) {
  uint lane = tid & 31, sg = tid >> 5, nsg = (ntg + 31) >> 5;
  for (uint off = 16; off > 0; off >>= 1) {
    float ov = simd_shuffle_down(v, off);
    uint oi = simd_shuffle_down(i, off);
    if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
  }
  if (lane == 0) { sv[sg] = v; si[sg] = i; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    v = lane < nsg ? sv[lane] : -INFINITY;
    i = lane < nsg ? si[lane] : 0xffffffff;
    for (uint off = 16; off > 0; off >>= 1) {
      float ov = simd_shuffle_down(v, off);
      uint oi = simd_shuffle_down(i, off);
      if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    if (lane == 0) { sv[0] = v; si[0] = i; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  v = sv[0];
  i = si[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// grid: ceil(V / chunk) threadgroups of 256 threads; out: [ntg, K] (value, index), descending.
[[kernel]] void topk_stage1(const device float* logits [[buffer(0)]], device float* cv [[buffer(1)]],
                            device uint* ci [[buffer(2)]], constant TopkStageParams& p [[buffer(3)]],
                            uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                            uint ntg [[threads_per_threadgroup]]) {
  threadgroup float sv[32];
  threadgroup uint si[32];
  float v[8];
  uint base = tg * p.chunk + tid * 8;
  for (int j = 0; j < 8; ++j) v[j] = base + j < p.V ? logits[base + j] : -INFINITY;
  for (uint r = 0; r < p.K; ++r) {
    float bv = -INFINITY;
    uint bi = 0xffffffff;
    for (int j = 0; j < 8; ++j)
      if (v[j] > bv) { bv = v[j]; bi = base + j; }
    tg_argmax(bv, bi, sv, si, tid, ntg);
    if (bi >= base && bi < base + 8) v[bi - base] = -INFINITY;
    if (tid == 0) {
      cv[tg * p.K + r] = bv;
      ci[tg * p.K + r] = bi;
    }
  }
}

struct SampleParams {
  uint n;        // candidates from stage 1
  uint K;        // top-k
  float temp;    // 0 = greedy
  float top_p;
  uint seed;
  uint step;
  uint out_index; // tokens[out_index] = sampled id
};

NI_INLINE uint pcg_hash(uint v) {
  uint state = v * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

// One threadgroup of 1024 threads. The RNG step of the draw: p.step, plus dstep[p.out_index] for sample_stage2_dstep
// (a KFlash tree verify: row r draws at position P + depth(r), so the draw is the plain sampler's for that position).
NI_INLINE void sample_stage2_impl(const device float* cv, const device uint* ci, device int* tokens,
                                  device float* out_probs, constant SampleParams& p, uint step, uint tid, uint ntg,
                                  threadgroup float* sv, threadgroup uint* si, threadgroup float* topv,
                                  threadgroup uint* topi) {
  constexpr int PER = 4;
  float v[PER];
  uint id[PER];
  for (int j = 0; j < PER; ++j) {
    uint k = tid * PER + j;
    v[j] = k < p.n ? cv[k] : -INFINITY;
    id[j] = k < p.n ? ci[k] : 0xffffffff;
  }
  uint K = max(1u, min(p.K, 64u));
  if (p.temp <= 0.0f) K = 1;
  for (uint r = 0; r < K; ++r) {
    float bv = -INFINITY;
    uint bi = 0xffffffff;
    int bj = -1;
    for (int j = 0; j < PER; ++j)
      if (v[j] > bv || (v[j] == bv && id[j] < bi)) { bv = v[j]; bi = id[j]; bj = j; }
    float wv = bv;
    uint wi = bi;
    tg_argmax(wv, wi, sv, si, tid, ntg);
    if (bj >= 0 && bi == wi && bv == wv) v[bj] = -INFINITY;
    if (tid == 0) { topv[r] = wv; topi[r] = wi; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint chosen = topi[0];
    if (p.temp > 0.0f && K > 1) {
      // top-p is taken on the untempered distribution (llama.cpp's and mlx-lm's order: top-k,
      // top-p, then temperature), then the kept candidates are sampled at temperature p.temp.
      float probs[64];
      float tot = 0.0f;
      for (uint r = 0; r < K; ++r) {
        probs[r] = metal::precise::exp(topv[r] - topv[0]);
        tot += probs[r];
      }
      // top-p over the sorted candidates (always keep the first)
      uint keep = K;
      float cum = 0.0f;
      for (uint r = 0; r < K; ++r) {
        cum += probs[r] / tot;
        if (cum >= p.top_p) { keep = r + 1; break; }
      }
      float kt = 0.0f;
      for (uint r = 0; r < keep; ++r) {
        probs[r] = metal::precise::exp((topv[r] - topv[0]) / p.temp);
        kt += probs[r];
      }
      float u = float(pcg_hash(p.seed ^ pcg_hash(step)) & 0xffffff) / 16777216.0f * kt;
      chosen = topi[keep - 1];
      float acc = 0.0f;
      for (uint r = 0; r < keep; ++r) {
        acc += probs[r];
        if (u < acc) { chosen = topi[r]; break; }
      }
    }
    tokens[p.out_index] = int(chosen);
    out_probs[0] = topv[0];
  }
}

[[kernel]] void sample_stage2(const device float* cv [[buffer(0)]], const device uint* ci [[buffer(1)]],
                              device int* tokens [[buffer(2)]], device float* out_probs [[buffer(3)]],
                              constant SampleParams& p [[buffer(4)]], uint tid [[thread_position_in_threadgroup]],
                              uint ntg [[threads_per_threadgroup]]) {
  threadgroup float sv[32];
  threadgroup uint si[32];
  threadgroup float topv[64];
  threadgroup uint topi[64];
  sample_stage2_impl(cv, ci, tokens, out_probs, p, p.step, tid, ntg, sv, si, topv, topi);
}

[[kernel]] void sample_stage2_dstep(const device float* cv [[buffer(0)]], const device uint* ci [[buffer(1)]],
                                    device int* tokens [[buffer(2)]], device float* out_probs [[buffer(3)]],
                                    constant SampleParams& p [[buffer(4)]], const device int* dstep [[buffer(5)]],
                                    uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float sv[32];
  threadgroup uint si[32];
  threadgroup float topv[64];
  threadgroup uint topi[64];
  sample_stage2_impl(cv, ci, tokens, out_probs, p, p.step + uint(dstep[p.out_index]), tid, ntg, sv, si, topv, topi);
}
