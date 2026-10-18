// ---- 8-row MMA matrix products (the verify8 design, for Nemotron-H's formats) ----
// Up to 8 token rows at once (a speculative verify block, the drafter's block): every
// projection runs as Y^T[n, t] = W[n, :] . X^T[:, t] on 8x8x8 simdgroup MMAs, the weights as the A operand decoded
// straight into registers as exact halves, the 8 token rows as B:
//   Q8   int8 codes; one MMA chain per 32-element block, closed by c += scale[n] * chain (fp32)
//   FP4  E2M1 x 2 integers times the block's UE4M3 scale x 0.5: exact in half (<= 8 significant bits), so no epilogue;
//        the tensor scale gs at the end
//   IQ2  signed grid magnitudes (8, 25, 43) from the grid as halves (buffer 8: nh_iq2_fill_tables2's first 4 KB,
//        constant address space) with the signs XORed in; one chain per 32-element sub-block, closed by c += db[n] * chain
// Within each 32-element block, MMA step s (0..3) and k-slot j (0..7) read the physical input
//   Q8 / IQ2: k(s, j) = (j >> 1) * 8 + 2 s + (j & 1)   (lane (fm, fn)'s A elements: 8 consecutive weights of row fm,
//             one IQ2 grid point)
//   FP4:      k(s, j) = 16 (j >> 2) + 4 ((j >> 1) & 1) + s + 8 (j & 1)   (the two nibbles of 4 bytes per lane: the
//             repacked order holds elements i and i + 8 in byte i of a 16-block; a threadgroup LUT maps a byte to both)
// B is gathered on the fly from half rows of X (2 loads per step: tokens fn, fn + 1 at k(s, fm)).
// The lane layout and the integer codes as MMA operands follow src/kernels/verify8.metal.
//
// 8x8 fragment lane layout: lane holds row fm, columns fn, fn + 1 with q = lane >> 2,
// fm = (q & 4) | ((lane >> 1) & 3), fn = ((q & 2) << 1) | ((lane & 1) << 1).

struct NhV8P {
  uint N, K;      // outputs, inputs
  uint x_ld;      // halfs per X row
  uint y_ld;      // elements per Y row
  uint R;         // live token rows
  float gs;       // tensor scale (FP4), else 1
  uint epi;       // 0 store fp32, 1 add fp32, 2 relu^2 -> half, 3 store half
};

NI_INLINE void nh_v8_mma(thread float2& c, half2 a, half2 b) {
  simdgroup_half8x8 A, B;
  simdgroup_float8x8 C;
  A.thread_elements()[0] = a.x;
  A.thread_elements()[1] = a.y;
  B.thread_elements()[0] = b.x;
  B.thread_elements()[1] = b.y;
  C.thread_elements()[0] = c.x;
  C.thread_elements()[1] = c.y;
  simdgroup_multiply_accumulate(C, A, B, C);
  c = float2(C.thread_elements()[0], C.thread_elements()[1]);
}

// E2M1 nibble -> its value x 2 as half bits: magnitudes 0, 1, 2, 3, 4, 6, 8, 12 (code m: 0, 0x3C00, (m + 30) << 9
// from m = 2 up), sign bit 3 -> bit 15
NI_INLINE ushort nh_fp4_h2(uint nib) {
  const uint m = nib & 7u;
  const uint mag = m < 2u ? m * 0x3C00u : (m + 30u) << 9;
  return ushort(mag | ((nib & 8u) << 12));
}

// The lane's A operands of one 32-element block for the 4 MMA steps (a[s] = elements 2s, 2s + 1 of its 8), and the
// block's scale (Q8: the code scale; IQ2: db; FP4: 1, the scales are inside the halves).
// FP4 byte -> (low nibble, high nibble) values x 2 as half2 (threadgroup LUT, 1 KB)
NI_INLINE void nh_v8_fp4_lut(threadgroup half2* lut, uint tid, uint ntg) {
  for (uint b = tid; b < 256; b += ntg)
    lut[b] = half2(as_type<half>(nh_fp4_h2(b & 15u)), as_type<half>(nh_fp4_h2(b >> 4)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <int FMT>
NI_INLINE float nh_v8_a(const device uchar* q, const device uchar* s, uint K, uint n, uint g, uint part,
                        thread half2 (&a)[4], threadgroup const half2* lut, constant uint4* gh) {
  if (FMT == 0) {  // Q8: codes [N][K], half scales [N][K/32]
    const uint2 w = *(const device uint2*)(q + size_t(n) * K + g * 32 + part * 8);
    for (int st = 0; st < 4; ++st) {
      const uint word = st < 2 ? w.x : w.y, sh = 16 * (st & 1);
      a[st] = half2(half(as_type<char>(uchar(word >> sh))), half(as_type<char>(uchar(word >> (sh + 8)))));
    }
    return float(((const device half*)s)[size_t(n) * (K / 32) + g]);
  } else if (FMT == 1) {  // FP4: codes [N][K/2] (16-blocks: byte i = elements i, i + 8), UE4M3 [N][K/16]
    // the lane's pairs are the two nibbles of 4 bytes: 16-block part >> 1, bytes (part & 1) * 4 + step
    const uint blk = g * 2 + (part >> 1);
    const uint w = *(const device uint*)(q + size_t(n) * (K / 2) + blk * 8 + (part & 1u) * 4);
    const half sc = half(nh_ue4m3(s[size_t(n) * (K / 16) + blk]));
    for (int st = 0; st < 4; ++st) a[st] = lut[(w >> (8 * st)) & 0xFFu] * sc;
    return 1.0f;
  } else {  // IQ2: codes [N][K/256][64] (per 32: grid indices, signs + scale), half d [N][K/256]
    const uint2 aux = *(const device uint2*)(q + size_t(n) * (K / 4) + g * 8);
    const uint4 m = gh[(aux.x >> (8 * part)) & 0xFFu];  // the grid point's 8 magnitudes as halves
    const uint s7 = (aux.y >> (7 * part)) & 127u;
    const uint sg = s7 | ((popcount(s7) & 1u) << 7);
    // sign bits 2s, 2s + 1 -> bits 15, 31 of word s
    const uint4 sm = ((uint4(sg) >> uint4(0, 2, 4, 6)) & 1u) << 15 | ((uint4(sg) >> uint4(1, 3, 5, 7)) & 1u) << 31;
    const uint4 w = m ^ sm;
    for (int st = 0; st < 4; ++st) a[st] = as_type<half2>(w[st]);
    return float(((const device half*)s)[size_t(n) * (K / 256) + g / 8]) * (0.5f + float(aux.y >> 28)) * 0.25f;
  }
}

// Y^T for X rows 0..R-1 (B columns; rows >= R are zero). The threadgroup's simdgroups share one block of 8 NB output
// rows and split K (interleaved 32-blocks), reduced through threadgroup memory (red: nsg * NB * 32 float2).
// grid: ceil(N / (8 NB)) x (32 nsg).
template <int FMT, int NB>
NI_INLINE void nh_v8_impl(const device half* X, const device uchar* q, const device uchar* s, device void* Y,
                          constant NhV8P& p, uint tg, uint sg, uint nsg, uint lane, threadgroup const half2* lut,
                          constant uint4* gh, threadgroup float2* red) {
  const uint q4 = lane >> 2;
  const uint fm = (q4 & 4u) | ((lane >> 1) & 3u), fn = ((q4 & 2u) << 1) | ((lane & 1u) << 1);
  // the B rows of this lane's two columns (fn, fn + 1); a missing column reads row 0 and is masked
  uint xr[2];
  bool live[2];
  for (int c = 0; c < 2; ++c) {
    const uint j = fn + uint(c);
    live[c] = j < p.R;
    xr[c] = live[c] ? j : 0u;
  }
  const uint nb0 = tg * (8 * NB);
  if (nb0 >= p.N) return;
  float2 acc[NB];
  for (int b = 0; b < NB; ++b) acc[b] = float2(0.0f);
  // B: the lane's k-slot j = fm reads k(s, fm): Q8 / IQ2 (fm >> 1) * 8 + 2 s + (fm & 1) (8 consecutive weights per
  // lane), FP4 16 ((fm >> 1) >> 1) + ((fm >> 1) & 1) 4 + s + 8 (fm & 1) (the two nibbles of 4 bytes per lane)
  const uint part = fm >> 1;
  const uint G = p.K / 32;
  for (uint g = sg; g < G; g += nsg) {
    half2 bv[4];
    for (int st = 0; st < 4; ++st) {
      const uint k = FMT == 1 ? g * 32 + 16 * (part >> 1) + (part & 1u) * 4 + uint(st) + 8 * (fm & 1u)
                              : g * 32 + part * 8 + 2 * st + (fm & 1u);
      const half x0 = live[0] ? X[size_t(xr[0]) * p.x_ld + k] : half(0);
      const half x1 = live[1] ? X[size_t(xr[1]) * p.x_ld + k] : half(0);
      bv[st] = half2(x0, x1);
    }
    for (int b = 0; b < NB; ++b) {
      const uint n = min(nb0 + 8 * b + fm, p.N - 1);
      half2 av[4];
      const float sc = nh_v8_a<FMT>(q, s, p.K, n, g, fn >> 1, av, lut, gh);
      if (FMT == 1) {
        for (int st = 0; st < 4; ++st) nh_v8_mma(acc[b], av[st], bv[st]);
      } else {
        float2 ch = float2(0.0f);
        for (int st = 0; st < 4; ++st) nh_v8_mma(ch, av[st], bv[st]);
        acc[b] += sc * ch;
      }
    }
  }
  for (int b = 0; b < NB; ++b) red[(sg * NB + uint(b)) * 32 + lane] = acc[b];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  for (int b = 0; b < NB; ++b)
    for (uint j = 1; j < nsg; ++j) acc[b] += red[(j * NB + uint(b)) * 32 + lane];
  // C: row n = nb0 + 8 b + fm, columns (tokens) fn, fn + 1
  for (int b = 0; b < NB; ++b) {
    const uint n = nb0 + 8 * b + fm;
    if (n >= p.N) continue;
    for (int c = 0; c < 2; ++c) {
      if (!live[c]) continue;
      float v = acc[b][c] * p.gs;
      const size_t o = size_t(fn + uint(c)) * p.y_ld + n;
      if (p.epi == 0) ((device float*)Y)[o] = v;
      else if (p.epi == 1) ((device float*)Y)[o] += v;
      else if (p.epi == 2) { v = max(v, 0.0f); ((device half*)Y)[o] = half(v * v); }
      else ((device half*)Y)[o] = half(v);
    }
  }
}

#define NH_V8S(NAME, FMT, NB)                                                                                       \
  [[kernel]] void NAME(const device half* X [[buffer(0)]], const device uchar* Wq [[buffer(1)]],                    \
                       const device uchar* Ws [[buffer(2)]], device void* Y [[buffer(3)]],                         \
                       constant NhV8P& p [[buffer(4)]], constant uint4* gh [[buffer(8)]],                          \
                       uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],      \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],           \
                       uint tid [[thread_index_in_threadgroup]]) {                                                  \
    threadgroup half2 lut[FMT == 1 ? 256 : 1];                                                                      \
    threadgroup float2 red[8 * NB * 32];                                                                            \
    if (FMT == 1) nh_v8_fp4_lut(lut, tid, 32 * nsg);                                                                \
    nh_v8_impl<FMT, NB>(X, Wq, Ws, Y, p, tg.x, sg, nsg, lane, lut, gh, red);                                        \
  }
// split-K: grid ceil(N / (8 NB)) threadgroups (dense) x (32 nsg), nsg <= 8. The engine runs NB 4 for 4k-32k outputs and
// NB 2 otherwise (enc_gemm); NB 1 and FP4's NB 8 are kept for nh_bench.
NH_V8S(nh_v8_q8_s1, 0, 1)
NH_V8S(nh_v8_q8_s2, 0, 2)
NH_V8S(nh_v8_q8_s4, 0, 4)
NH_V8S(nh_v8_fp4_s1, 1, 1)
NH_V8S(nh_v8_fp4_s2, 1, 2)
NH_V8S(nh_v8_fp4_s4, 1, 4)
NH_V8S(nh_v8_fp4_s8, 1, 8)
NH_V8S(nh_v8_iq2_s1, 2, 1)
NH_V8S(nh_v8_iq2_s2, 2, 2)
NH_V8S(nh_v8_iq2_s4, 2, 4)
