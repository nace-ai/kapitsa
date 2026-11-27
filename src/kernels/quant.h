// Shared helpers for MLX affine-quantized dot products (included after common.h).

template <int NT>
struct NormCtx {
  float inv[NT];
};

// Per-simdgroup RMS of each input vector (K floats; every lane reads K/32 of them).
template <int NT>
NI_INLINE NormCtx<NT> rms_ctx(const device float* x, uint x_stride, uint K, float eps, uint lane) {
  NormCtx<NT> c;
  for (int t = 0; t < NT; ++t) {
    const device float4* xp = (const device float4*)(x + t * x_stride);
    float ss = 0.0f;
    for (uint i = lane; i < K / 4; i += 32) {
      float4 v = xp[i];
      ss += dot(v, v);
    }
    ss = simd_sum(ss);
    c.inv[t] = metal::precise::rsqrt(ss / float(K) + eps);
  }
  return c;
}

template <int BITS, int VPL, int NT>
struct XChunk {
  float xs[NT][VPL];
  float xsum[NT];
};

template <int BITS, int VPL, int NT, bool NORM>
NI_INLINE void load_x(thread XChunk<BITS, VPL, NT>& c, const device float* x, uint x_stride, uint k0,
                      thread const NormCtx<NT>& nc, const device bfloat* nw) {
  constexpr int PER_WORD = 32 / BITS;
  for (int t = 0; t < NT; ++t) {
    const device float4* xp = (const device float4*)(x + t * x_stride + k0);
    c.xsum[t] = 0.0f;
    for (int i = 0; i < VPL / 4; ++i) {
      float4 v = xp[i];
      if (NORM) {
        float4 w = float4(nw[k0 + i * 4 + 0], nw[k0 + i * 4 + 1], nw[k0 + i * 4 + 2], nw[k0 + i * 4 + 3]);
        v = v * nc.inv[t] * w;
      }
      c.xsum[t] += v.x + v.y + v.z + v.w;
      for (int e = 0; e < 4; ++e) {
        int j = (i * 4 + e) % PER_WORD;
        c.xs[t][i * 4 + e] = v[e] * float(1.0f / float(1u << (BITS * j)));
      }
    }
  }
}

// acc[t] += dot(row chunk, x chunk) for one row; wp points at the lane's first word.
template <int BITS, int VPL, int NT>
NI_INLINE void dot_row(thread const XChunk<BITS, VPL, NT>& c, const device uint32_t* wp, float s, float b,
                       thread float* acc) {
  constexpr int PER_WORD = 32 / BITS;
  constexpr int WORDS = VPL / PER_WORD;
  constexpr uint MASK = (1u << BITS) - 1u;
  uint32_t wv[WORDS];
  if (WORDS == 4) {
    uint4 v = *(const device uint4*)wp;
    wv[0] = v.x; wv[1] = v.y; wv[2] = v.z; wv[3] = v.w;
  } else if (WORDS == 2) {
    uint2 v = *(const device uint2*)wp;
    wv[0] = v.x; wv[1] = v.y;
  } else {
    for (int i = 0; i < WORDS; ++i) wv[i] = wp[i];
  }
  for (int t = 0; t < NT; ++t) {
    float d = 0.0f;
    for (int i = 0; i < WORDS; ++i)
      for (int j = 0; j < PER_WORD; ++j) d += float(wv[i] & (MASK << (BITS * j))) * c.xs[t][i * PER_WORD + j];
    acc[t] += s * d + b * c.xsum[t];
  }
}


// Shared body for v2 kernels whose x is already staged in threadgroup memory.
template <int BITS, int VPL, int ROWS, int GS>
NI_INLINE void qmv2_rows(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                         threadgroup const float* xs, uint N, uint K, uint row0, uint lane, thread float* acc) {
  constexpr int PER_WORD = 32 / BITS;
  constexpr uint MASK = (1u << BITS) - 1u;
  constexpr int WORDS = VPL / PER_WORD;
  const uint words_per_row = K / PER_WORD;
  const uint groups_per_row = K / GS;
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    acc[r] = 0.0f;
    rows[r] = min(row0 + r, N - 1);
  }
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    float xv[VPL];
    float xsum = 0.0f;
    for (int i = 0; i < VPL / 4; ++i) {
      float4 v = ((threadgroup const float4*)(xs + k0))[i];
      xsum += v.x + v.y + v.z + v.w;
      for (int e = 0; e < 4; ++e) xv[i * 4 + e] = v[e] * float(1.0f / float(1u << (BITS * ((i * 4 + e) % PER_WORD))));
    }
    const uint g = k0 / GS, wo = k0 / PER_WORD;
    for (int r = 0; r < ROWS; ++r) {
      const device uint32_t* wp = w + rows[r] * words_per_row + wo;
      uint32_t wv[WORDS];
      if (WORDS == 4) {
        uint4 v = *(const device uint4*)wp;
        wv[0] = v.x; wv[1] = v.y; wv[2] = v.z; wv[3] = v.w;
      } else if (WORDS == 2) {
        uint2 v = *(const device uint2*)wp;
        wv[0] = v.x; wv[1] = v.y;
      } else {
        for (int i = 0; i < WORDS; ++i) wv[i] = wp[i];
      }
      float d = 0.0f;
      for (int i = 0; i < WORDS; ++i)
        for (int j = 0; j < PER_WORD; ++j) d += float(wv[i] & (MASK << (BITS * j))) * xv[i * PER_WORD + j];
      acc[r] += float(scales[rows[r] * groups_per_row + g]) * d + float(biases[rows[r] * groups_per_row + g]) * xsum;
    }
  }
  for (int r = 0; r < ROWS; ++r) acc[r] = simd_sum(acc[r]);
}

// qmv2_rows with x in device memory (read through the caches instead of staged): the same arithmetic
template <int BITS, int VPL, int ROWS, int GS>
NI_INLINE void qmv2_rows_dx(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                            const device float* xs, uint N, uint K, uint row0, uint lane, thread float* acc) {
  constexpr int PER_WORD = 32 / BITS;
  constexpr uint MASK = (1u << BITS) - 1u;
  constexpr int WORDS = VPL / PER_WORD;
  const uint words_per_row = K / PER_WORD;
  const uint groups_per_row = K / GS;
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    acc[r] = 0.0f;
    rows[r] = min(row0 + r, N - 1);
  }
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    float xv[VPL];
    float xsum = 0.0f;
    for (int i = 0; i < VPL / 4; ++i) {
      float4 v = ((const device float4*)(xs + k0))[i];
      xsum += v.x + v.y + v.z + v.w;
      for (int e = 0; e < 4; ++e) xv[i * 4 + e] = v[e] * float(1.0f / float(1u << (BITS * ((i * 4 + e) % PER_WORD))));
    }
    const uint g = k0 / GS, wo = k0 / PER_WORD;
    for (int r = 0; r < ROWS; ++r) {
      const device uint32_t* wp = w + rows[r] * words_per_row + wo;
      uint32_t wv[WORDS];
      if (WORDS == 4) {
        uint4 v = *(const device uint4*)wp;
        wv[0] = v.x; wv[1] = v.y; wv[2] = v.z; wv[3] = v.w;
      } else if (WORDS == 2) {
        uint2 v = *(const device uint2*)wp;
        wv[0] = v.x; wv[1] = v.y;
      } else {
        for (int i = 0; i < WORDS; ++i) wv[i] = wp[i];
      }
      float d = 0.0f;
      for (int i = 0; i < WORDS; ++i)
        for (int j = 0; j < PER_WORD; ++j) d += float(wv[i] & (MASK << (BITS * j))) * xv[i * PER_WORD + j];
      acc[r] += float(scales[rows[r] * groups_per_row + g]) * d + float(biases[rows[r] * groups_per_row + g]) * xsum;
    }
  }
  for (int r = 0; r < ROWS; ++r) acc[r] = simd_sum(acc[r]);
}

// Stages x (optionally normalized) into threadgroup memory.
NI_INLINE void stage_x(const device float* x, uint K, bool norm, float eps, const device bfloat* nw,
                       threadgroup float* xs, threadgroup float* red, uint tid, uint ntg) {
  float ss = 0.0f;
  for (uint i = tid; i < K / 4; i += ntg) {
    float4 v = ((const device float4*)x)[i];
    ((threadgroup float4*)xs)[i] = v;
    ss += dot(v, v);
  }
  if (norm) {
    ss = tg_sum(ss, red, tid, ntg);
    float inv = metal::precise::rsqrt(ss / float(K) + eps);
    for (uint i = tid; i < K; i += ntg) xs[i] = xs[i] * inv * float(nw[i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

