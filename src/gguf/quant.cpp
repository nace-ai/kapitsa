// GGUF block decoders (written from the format's layout: ggml-common.h's block structs) and MLX affine encoders. The
// IQ1 / IQ2 / IQ3 decoders follow ggml's dequantize_row_iq* (ggml/src/ggml-quants.c, MIT License, Copyright (c)
// 2023-2026 The ggml authors; see THIRD_PARTY_NOTICES), with its codebooks in iq_grids.h.
#include "gguf/quant.h"

#include "gguf/gguf.h"
#include "gguf/iq_grids.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace ni::gguf {

uint16_t to_bf16(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x7fffffu)) return uint16_t((u >> 16) | 0x40);  // quiet NaN
  u += 0x7fffu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}

float from_bf16(uint16_t h) {
  const uint32_t u = uint32_t(h) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

float from_f16(uint16_t h) {
  _Float16 f;
  memcpy(&f, &h, 2);
  return float(f);
}

namespace {

inline float f16_at(const uint8_t* p) {
  uint16_t h;
  memcpy(&h, p, 2);
  return from_f16(h);
}

// Q4_K / Q5_K: eight 6-bit scales and mins packed in 12 bytes
inline void scale_min_k4(int j, const uint8_t* q, uint8_t* sc, uint8_t* m) {
  if (j < 4) {
    *sc = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *sc = uint8_t((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    *m = uint8_t((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

// IQ4_NL / IQ4_XS: the format's 16 non-linear levels
constexpr int8_t kIq4Levels[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
// E2M1 (FP4) magnitudes doubled, as MXFP4 / NVFP4 blocks are stored with a halved scale
constexpr int8_t kFp4x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

inline float e8m0_half(uint8_t e) {  // 2^(e - 127) / 2
  return std::ldexp(1.0f, int(e) - 128);
}

inline float ue4m3_half(uint8_t x) {  // unsigned E4M3 (bias 7), halved; 0 and 0x7F (NaN) read as 0
  if (x == 0 || x == 0x7F) return 0.0f;
  const int e = (x >> 3) & 0xF, m = x & 7;
  const float v = e == 0 ? std::ldexp(float(m), -9) : std::ldexp(1.0f + float(m) / 8.0f, e - 7);
  return v * 0.5f;
}

void dq_q4_0(const uint8_t* b, float* y) {
  const float d = f16_at(b);
  const uint8_t* q = b + 2;
  for (int j = 0; j < 16; ++j) {
    y[j] = d * float(int(q[j] & 0xF) - 8);
    y[j + 16] = d * float(int(q[j] >> 4) - 8);
  }
}

void dq_q4_1(const uint8_t* b, float* y) {
  const float d = f16_at(b), m = f16_at(b + 2);
  const uint8_t* q = b + 4;
  for (int j = 0; j < 16; ++j) {
    y[j] = d * float(q[j] & 0xF) + m;
    y[j + 16] = d * float(q[j] >> 4) + m;
  }
}

void dq_q5_0(const uint8_t* b, float* y) {
  const float d = f16_at(b);
  uint32_t qh;
  memcpy(&qh, b + 2, 4);
  const uint8_t* q = b + 6;
  for (int j = 0; j < 16; ++j) {
    const int lo = int(q[j] & 0xF) | int(((qh >> j) & 1u) << 4);
    const int hi = int(q[j] >> 4) | int(((qh >> (j + 16)) & 1u) << 4);
    y[j] = d * float(lo - 16);
    y[j + 16] = d * float(hi - 16);
  }
}

void dq_q5_1(const uint8_t* b, float* y) {
  const float d = f16_at(b), m = f16_at(b + 2);
  uint32_t qh;
  memcpy(&qh, b + 4, 4);
  const uint8_t* q = b + 8;
  for (int j = 0; j < 16; ++j) {
    const int lo = int(q[j] & 0xF) | int(((qh >> j) & 1u) << 4);
    const int hi = int(q[j] >> 4) | int(((qh >> (j + 16)) & 1u) << 4);
    y[j] = d * float(lo) + m;
    y[j + 16] = d * float(hi) + m;
  }
}

void dq_q8_0(const uint8_t* b, float* y) {
#if defined(__aarch64__)
  uint16_t h;
  memcpy(&h, b, 2);
  const uintptr_t src = reinterpret_cast<uintptr_t>(b), dst = reinterpret_cast<uintptr_t>(y);
  const bool overlap = src <= dst ? dst - src < 34 : src - dst < 32 * sizeof(float);
  if ((h & 0x7c00u) != 0x7c00u && !overlap) {
    // Finite half x int8 fits exactly in fp32. Keep exceptional scales and sequential overlapping writes scalar.
    const float32x4_t d = vdupq_n_f32(from_f16(h));
    const int8x16_t a = vld1q_s8(reinterpret_cast<const int8_t*>(b + 2));
    const int8x16_t z = vld1q_s8(reinterpret_cast<const int8_t*>(b + 18));
    const int16x8_t a0 = vmovl_s8(vget_low_s8(a)), a1 = vmovl_s8(vget_high_s8(a));
    const int16x8_t z0 = vmovl_s8(vget_low_s8(z)), z1 = vmovl_s8(vget_high_s8(z));
    vst1q_f32(y, vmulq_f32(d, vcvtq_f32_s32(vmovl_s16(vget_low_s16(a0)))));
    vst1q_f32(y + 4, vmulq_f32(d, vcvtq_f32_s32(vmovl_s16(vget_high_s16(a0)))));
    vst1q_f32(y + 8, vmulq_f32(d, vcvtq_f32_s32(vmovl_s16(vget_low_s16(a1)))));
    vst1q_f32(y + 12, vmulq_f32(d, vcvtq_f32_s32(vmovl_s16(vget_high_s16(a1)))));
    vst1q_f32(y + 16, vmulq_f32(d, vcvtq_f32_s32(vmovl_s16(vget_low_s16(z0)))));
    vst1q_f32(y + 20, vmulq_f32(d, vcvtq_f32_s32(vmovl_s16(vget_high_s16(z0)))));
    vst1q_f32(y + 24, vmulq_f32(d, vcvtq_f32_s32(vmovl_s16(vget_low_s16(z1)))));
    vst1q_f32(y + 28, vmulq_f32(d, vcvtq_f32_s32(vmovl_s16(vget_high_s16(z1)))));
    return;
  }
#endif
  const float d = f16_at(b);
  const int8_t* q = reinterpret_cast<const int8_t*>(b + 2);
  for (int j = 0; j < 32; ++j) y[j] = d * float(q[j]);
}

// 256 values in 16 sub-blocks of 16; 2-bit codes, a 4-bit scale and 4-bit min per sub-block
void dq_q2_k(const uint8_t* b, float* y) {
  const uint8_t* sc = b;
  const uint8_t* qs = b + 16;
  const float d = f16_at(b + 80), dmin = f16_at(b + 82);
  for (int k = 0; k < 16; ++k) {
    const int half = k / 8, shift = 2 * ((k % 8) / 2), second = k % 2;
    const uint8_t* q = qs + 32 * half + 16 * second;
    const float dl = d * float(sc[k] & 0xF), ml = dmin * float(sc[k] >> 4);
    for (int l = 0; l < 16; ++l) y[16 * k + l] = dl * float((q[l] >> shift) & 3) - ml;
  }
}

// 256 values in 16 sub-blocks of 16; 3-bit codes (2 low bits + a high-bit mask), signed 6-bit scales
void dq_q3_k(const uint8_t* b, float* y) {
  const uint8_t* hm = b;
  const uint8_t* qs = b + 32;
  const uint8_t* s = b + 96;
  const float d = f16_at(b + 108);
  int scale[16];
  for (int i = 0; i < 4; ++i) {
    scale[i] = (s[i] & 0xF) | (((s[8 + i] >> 0) & 3) << 4);
    scale[4 + i] = (s[4 + i] & 0xF) | (((s[8 + i] >> 2) & 3) << 4);
    scale[8 + i] = (s[i] >> 4) | (((s[8 + i] >> 4) & 3) << 4);
    scale[12 + i] = (s[4 + i] >> 4) | (((s[8 + i] >> 6) & 3) << 4);
  }
#if defined(__aarch64__)
  const uintptr_t src = reinterpret_cast<uintptr_t>(b), dst = reinterpret_cast<uintptr_t>(y);
  const bool overlap = src <= dst ? dst - src < 110 : src - dst < 256 * sizeof(float);
  if (std::isfinite(d) && !overlap) {
    // A finite half times a signed 6-bit scale and a 3-bit value fits exactly in fp32.
    for (int k = 0; k < 16; ++k) {
      const int half = k / 8, j = (k % 8) / 2, second = k % 2;
      const uint8x16_t low = vandq_u8(vshlq_u8(vld1q_u8(qs + 32 * half + 16 * second),
                                               vdupq_n_s8(int8_t(-2 * j))), vdupq_n_u8(3));
      const uint8x16_t high = vandq_u8(vshlq_u8(vld1q_u8(hm + 16 * second),
                                                vdupq_n_s8(int8_t(-4 * half - j))), vdupq_n_u8(1));
      const int8x16_t values = vreinterpretq_s8_u8(vsubq_u8(low, vshlq_n_u8(veorq_u8(high, vdupq_n_u8(1)), 2)));
      const int16x8_t a = vmovl_s8(vget_low_s8(values)), z = vmovl_s8(vget_high_s8(values));
      const float32x4_t dl = vdupq_n_f32(d * float(scale[k] - 32));
      vst1q_f32(y + 16 * k, vmulq_f32(dl, vcvtq_f32_s32(vmovl_s16(vget_low_s16(a)))));
      vst1q_f32(y + 16 * k + 4, vmulq_f32(dl, vcvtq_f32_s32(vmovl_s16(vget_high_s16(a)))));
      vst1q_f32(y + 16 * k + 8, vmulq_f32(dl, vcvtq_f32_s32(vmovl_s16(vget_low_s16(z)))));
      vst1q_f32(y + 16 * k + 12, vmulq_f32(dl, vcvtq_f32_s32(vmovl_s16(vget_high_s16(z)))));
    }
    return;
  }
#endif
  for (int k = 0; k < 16; ++k) {
    const int half = k / 8, j = (k % 8) / 2, second = k % 2;
    const int shift = 2 * j, bit = 4 * half + j;
    const uint8_t* q = qs + 32 * half + 16 * second;
    const uint8_t* h = hm + 16 * second;
    const float dl = d * float(scale[k] - 32);
    for (int l = 0; l < 16; ++l) {
      const int v = int((q[l] >> shift) & 3) - ((h[l] >> bit) & 1 ? 0 : 4);
      y[16 * k + l] = dl * float(v);
    }
  }
}

// 256 values in 8 sub-blocks of 32; 4-bit codes, 6-bit scale and min per sub-block
void q4_k_parts(const uint8_t* b, uint8_t* codes, float* scale, float* offset) {
  const float d = f16_at(b), dmin = f16_at(b + 2);
  const uint8_t* sm = b + 4;
  const uint8_t* qs = b + 16;
  for (int p = 0; p < 4; ++p) {
    for (int h = 0; h < 2; ++h) {
      uint8_t sc, m;
      scale_min_k4(2 * p + h, sm, &sc, &m);
      scale[2 * p + h] = d * float(sc);
      offset[2 * p + h] = -dmin * float(m);
      for (int l = 0; l < 32; ++l) codes[64 * p + 32 * h + l] = h ? qs[32 * p + l] >> 4 : qs[32 * p + l] & 0xF;
    }
  }
}

void dq_q4_k(const uint8_t* b, float* y) {
  uint8_t c[256];
  float s[8], o[8];
  q4_k_parts(b, c, s, o);
  for (int i = 0; i < 256; ++i) y[i] = s[i / 32] * float(c[i]) + o[i / 32];
}

// 256 values in 8 sub-blocks of 32; 5-bit codes (4 low bits + a high-bit plane), 6-bit scale and min
void dq_q5_k(const uint8_t* b, float* y) {
  const float d = f16_at(b), dmin = f16_at(b + 2);
  const uint8_t* sm = b + 4;
  const uint8_t* qh = b + 16;
  const uint8_t* qs = b + 48;
  for (int p = 0; p < 4; ++p)
    for (int h = 0; h < 2; ++h) {
      uint8_t sc, m;
      scale_min_k4(2 * p + h, sm, &sc, &m);
      const float dl = d * float(sc), ml = dmin * float(m);
      const int bit = 2 * p + h;
      for (int l = 0; l < 32; ++l) {
        const int lo = h ? qs[32 * p + l] >> 4 : qs[32 * p + l] & 0xF;
        y[64 * p + 32 * h + l] = dl * float(lo + (((qh[l] >> bit) & 1) << 4)) - ml;
      }
    }
}

// 256 values in 16 sub-blocks of 16; 6-bit codes (4 low + 2 high bits), signed 8-bit scales
void dq_q6_k(const uint8_t* b, float* y) {
#if defined(__aarch64__)
  uint16_t h;
  memcpy(&h, b + 208, 2);
  const uintptr_t src = reinterpret_cast<uintptr_t>(b), dst = reinterpret_cast<uintptr_t>(y);
  const bool overlap = src <= dst ? dst - src < 210 : src - dst < 256 * sizeof(float);
  if ((h & 0x7c00u) != 0x7c00u && !overlap) {
    const float d = from_f16(h);
    const uint8x16_t mask = vdupq_n_u8(15), high_mask = vdupq_n_u8(48);
    auto store = [](uint8x16_t code, float scale, float* out) {
      const int8x16_t q = vreinterpretq_s8_u8(vsubq_u8(code, vdupq_n_u8(32)));
      const int16x8_t lo = vmovl_s8(vget_low_s8(q)), hi = vmovl_s8(vget_high_s8(q));
      vst1q_f32(out, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), scale));
      vst1q_f32(out + 4, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), scale));
      vst1q_f32(out + 8, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), scale));
      vst1q_f32(out + 12, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), scale));
    };
    // Keep the scalar order: half scale times signed sub-block scale, then the six-bit value.
    for (int half = 0; half < 2; ++half) {
      const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192 + 8 * half);
      for (int s = 0; s < 2; ++s) {
        const uint8x16_t lo = vld1q_u8(b + 64 * half + 16 * s);
        const uint8x16_t hi = vld1q_u8(b + 64 * half + 16 * s + 32);
        const uint8x16_t qh = vld1q_u8(b + 128 + 32 * half + 16 * s);
        float* out = y + 128 * half + 16 * s;
        store(vorrq_u8(vandq_u8(lo, mask), vandq_u8(vshlq_n_u8(qh, 4), high_mask)), d * float(sc[s]), out);
        store(vorrq_u8(vandq_u8(hi, mask), vandq_u8(vshlq_n_u8(qh, 2), high_mask)), d * float(sc[s + 2]), out + 32);
        store(vorrq_u8(vshrq_n_u8(lo, 4), vandq_u8(qh, high_mask)), d * float(sc[s + 4]), out + 64);
        store(vorrq_u8(vshrq_n_u8(hi, 4), vandq_u8(vshrq_n_u8(qh, 2), high_mask)), d * float(sc[s + 6]), out + 96);
      }
    }
    return;
  }
#endif
  const uint8_t* ql = b;
  const uint8_t* qh = b + 128;
  const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
  const float d = f16_at(b + 208);
  for (int half = 0; half < 2; ++half) {
    const uint8_t* l0 = ql + 64 * half;
    const uint8_t* h0 = qh + 32 * half;
    const int8_t* s0 = sc + 8 * half;
    float* y0 = y + 128 * half;
    for (int l = 0; l < 32; ++l) {
      const int is = l / 16;
      const int q1 = int((l0[l] & 0xF) | (((h0[l] >> 0) & 3) << 4)) - 32;
      const int q2 = int((l0[l + 32] & 0xF) | (((h0[l] >> 2) & 3) << 4)) - 32;
      const int q3 = int((l0[l] >> 4) | (((h0[l] >> 4) & 3) << 4)) - 32;
      const int q4 = int((l0[l + 32] >> 4) | (((h0[l] >> 6) & 3) << 4)) - 32;
      y0[l] = d * float(s0[is]) * float(q1);
      y0[l + 32] = d * float(s0[is + 2]) * float(q2);
      y0[l + 64] = d * float(s0[is + 4]) * float(q3);
      y0[l + 96] = d * float(s0[is + 6]) * float(q4);
    }
  }
}

void dq_iq4_nl_scalar(const uint8_t* b, float* y) {
  const float d = f16_at(b);
  const uint8_t* q = b + 2;
  for (int j = 0; j < 16; ++j) {
    y[j] = d * float(kIq4Levels[q[j] & 0xF]);
    y[j + 16] = d * float(kIq4Levels[q[j] >> 4]);
  }
}


void dq_iq4_nl(const uint8_t* b, float* y) {
#if defined(__aarch64__)
  uint16_t h;
  memcpy(&h, b, 2);
  const uintptr_t src = reinterpret_cast<uintptr_t>(b), dst = reinterpret_cast<uintptr_t>(y);
  const bool overlap = dst >= src ? dst - src < 18 : src - dst < 32 * sizeof(float);
  if ((h & 0x7c00) != 0x7c00 && !overlap) {
    const float d = from_f16(h);
    const int8x16_t levels = vld1q_s8(kIq4Levels);
    const uint8x16_t q = vld1q_u8(b + 2);
    auto store = [&](uint8x16_t indices, float* out) {
      const int8x16_t v = vqtbl1q_s8(levels, indices);
      const int16x8_t lo = vmovl_s8(vget_low_s8(v)), hi = vmovl_s8(vget_high_s8(v));
      vst1q_f32(out, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), d));
      vst1q_f32(out + 4, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), d));
      vst1q_f32(out + 8, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), d));
      vst1q_f32(out + 12, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), d));
    };
    store(vandq_u8(q, vdupq_n_u8(15)), y);
    store(vshrq_n_u8(q, 4), y + 16);
    return;
  }
#endif
  dq_iq4_nl_scalar(b, y);
}

void dq_iq4_xs(const uint8_t* b, float* y) {
#if defined(__aarch64__)
  uint16_t h, high;
  memcpy(&h, b, 2);
  memcpy(&high, b + 2, 2);
  const uintptr_t src = reinterpret_cast<uintptr_t>(b), dst = reinterpret_cast<uintptr_t>(y);
  const bool overlap = dst >= src ? dst - src < 136 : src - dst < 256 * sizeof(float);
  if ((h & 0x7c00) != 0x7c00 && !overlap) {
    const float d = from_f16(h);
    const int8x16_t levels = vld1q_s8(kIq4Levels);
    auto store = [&](uint8x16_t indices, float scale, float* out) {
      const int8x16_t v = vqtbl1q_s8(levels, indices);
      const int16x8_t lo = vmovl_s8(vget_low_s8(v)), hi = vmovl_s8(vget_high_s8(v));
      vst1q_f32(out, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), scale));
      vst1q_f32(out + 4, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), scale));
      vst1q_f32(out + 8, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), scale));
      vst1q_f32(out + 12, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), scale));
    };
    for (int ib = 0; ib < 8; ++ib) {
      const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((high >> (2 * ib)) & 3) << 4);
      const float dl = d * float(ls - 32);
      const uint8x16_t q = vld1q_u8(b + 8 + 16 * ib);
      store(vandq_u8(q, vdupq_n_u8(15)), dl, y + 32 * ib);
      store(vshrq_n_u8(q, 4), dl, y + 32 * ib + 16);
    }
    return;
  }
#endif
  const float d = f16_at(b);
  uint16_t sh;
  memcpy(&sh, b + 2, 2);
  const uint8_t* sl = b + 4;
  const uint8_t* qs = b + 8;
  for (int ib = 0; ib < 8; ++ib) {
    const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
    const float dl = d * float(ls - 32);
    const uint8_t* q = qs + 16 * ib;
    for (int j = 0; j < 16; ++j) {
      y[32 * ib + j] = dl * float(kIq4Levels[q[j] & 0xF]);
      y[32 * ib + j + 16] = dl * float(kIq4Levels[q[j] >> 4]);
    }
  }
}

// ---- IQ1 / IQ2 / IQ3: 256 values per block, 8-value groups looked up in a codebook (iq_grids.h)

// IQ2 / IQ3 signs: 7 stored bits per 8 values, the 8th the parity of the 7 (ggml's ksigns_iq2xs)
inline uint32_t iq_signs(uint32_t s7) { return s7 | (uint32_t(__builtin_popcount(s7) & 1) << 7); }
inline float iq_sign(uint32_t signs, int j) { return (signs >> j) & 1 ? -1.0f : 1.0f; }
inline float grid_byte(uint64_t g, int j) { return float(uint8_t(g >> (8 * j))); }
inline float grid_int8(uint64_t g, int j) { return float(int8_t(uint8_t(g >> (8 * j)))); }

// 2.06 bits: per 32 values 4 one-byte grid indices, then 4 x 7 sign bits and a 4-bit scale
void dq_iq2_xxs(const uint8_t* b, float* y) {
  const float d = f16_at(b);
  for (int ib = 0; ib < 8; ++ib) {
    uint32_t idx, aux;
    memcpy(&idx, b + 2 + 8 * ib, 4);
    memcpy(&aux, b + 6 + 8 * ib, 4);
    const float db = d * (0.5f + float(aux >> 28)) * 0.25f;
    for (int l = 0; l < 4; ++l) {
      const uint64_t g = iq::kIq2xxs[(idx >> (8 * l)) & 255];
      const uint32_t s = iq_signs((aux >> (7 * l)) & 127);
      for (int j = 0; j < 8; ++j) y[32 * ib + 8 * l + j] = db * grid_byte(g, j) * iq_sign(s, j);
    }
  }
}

// 2.31 bits: 16-bit entries (9-bit grid index, 7 sign bits), two 4-bit scales per 32 values
void dq_iq2_xs(const uint8_t* b, float* y) {
  const float d = f16_at(b);
  const uint8_t* sc = b + 66;
  for (int ib = 0; ib < 8; ++ib) {
    const float db[2] = {d * (0.5f + float(sc[ib] & 0xF)) * 0.25f, d * (0.5f + float(sc[ib] >> 4)) * 0.25f};
    for (int l = 0; l < 4; ++l) {
      uint16_t v;
      memcpy(&v, b + 2 + 2 * (4 * ib + l), 2);
      const uint64_t g = iq::kIq2xs[v & 511];
      const uint32_t s = iq_signs(v >> 9);
      for (int j = 0; j < 8; ++j) y[32 * ib + 8 * l + j] = db[l / 2] * grid_byte(g, j) * iq_sign(s, j);
    }
  }
}

// 2.56 bits: 10-bit grid indices (8 low bits + 2 in qh), 8 stored sign bits per 8 values, two 4-bit scales per 32
void dq_iq2_s(const uint8_t* b, float* y) {
#if defined(__aarch64__)
  uint16_t half;
  memcpy(&half, b, 2);
  const uintptr_t src = reinterpret_cast<uintptr_t>(b), dst = reinterpret_cast<uintptr_t>(y);
  const bool overlap = src <= dst ? dst - src < 82 : src - dst < 256 * sizeof(float);
  if ((half & 0x7c00u) != 0x7c00u && !overlap) {
    const float d = from_f16(half);
    const uint8x8_t sign_bits = {1, 2, 4, 8, 16, 32, 64, 128};
    for (int group = 0; group < 8; ++group) {
      for (int lane = 0; lane < 4; ++lane) {
        const unsigned nib = (b[74 + group] >> (4 * (lane / 2))) & 15;
        const float32x4_t scale = vdupq_n_f32(d * (0.5f + float(nib)) * 0.25f);
        const unsigned index = b[2 + 4 * group + lane] | ((unsigned(b[66 + group]) << (8 - 2 * lane)) & 0x300);
        const uint8x8_t grid = vreinterpret_u8_u64(vcreate_u64(iq::kIq2s[index]));
        const int16x8_t values = vreinterpretq_s16_u16(vmovl_u8(grid));
        const uint8x8_t positive = vceq_u8(vand_u8(vdup_n_u8(b[34 + 4 * group + lane]), sign_bits), vdup_n_u8(0));
        const uint16x8_t mask = vceqq_u16(vmovl_u8(positive), vdupq_n_u16(255));
        const int16x8_t signed_values = vbslq_s16(mask, values, vnegq_s16(values));
        float* out = y + 32 * group + 8 * lane;
        vst1q_f32(out, vmulq_f32(scale, vcvtq_f32_s32(vmovl_s16(vget_low_s16(signed_values)))));
        vst1q_f32(out + 4, vmulq_f32(scale, vcvtq_f32_s32(vmovl_s16(vget_high_s16(signed_values)))));
      }
    }
    return;
  }
#endif
  const float d = f16_at(b);
  const uint8_t* qs = b + 2;
  const uint8_t* signs = b + 34;
  const uint8_t* qh = b + 66;
  const uint8_t* sc = b + 74;
  for (int ib = 0; ib < 8; ++ib) {
    const float db[2] = {d * (0.5f + float(sc[ib] & 0xF)) * 0.25f, d * (0.5f + float(sc[ib] >> 4)) * 0.25f};
    for (int l = 0; l < 4; ++l) {
      const uint64_t g = iq::kIq2s[qs[4 * ib + l] | ((uint32_t(qh[ib]) << (8 - 2 * l)) & 0x300)];
      const uint32_t s = signs[4 * ib + l];
      for (int j = 0; j < 8; ++j) y[32 * ib + 8 * l + j] = db[l / 2] * grid_byte(g, j) * iq_sign(s, j);
    }
  }
}

// 3.06 bits: per 32 values 8 one-byte indices of 4-value grid points, then 4 x 7 sign bits and a 4-bit scale
void dq_iq3_xxs(const uint8_t* b, float* y) {
#if defined(__aarch64__)
  uint16_t half;
  memcpy(&half, b, 2);
  const uintptr_t src = reinterpret_cast<uintptr_t>(b), dst = reinterpret_cast<uintptr_t>(y);
  const bool overlap = src <= dst ? dst - src < 98 : src - dst < 256 * sizeof(float);
  if ((half & 0x7c00u) != 0x7c00u && !overlap) {
    const float d = from_f16(half);
    const uint8x8_t sign_bits = {1, 2, 4, 8, 16, 32, 64, 128};
    for (int group = 0; group < 8; ++group) {
      uint32_t aux;
      memcpy(&aux, b + 66 + 4 * group, 4);
      const float32x4_t scale = vdupq_n_f32(d * (0.5f + float(aux >> 28)) * 0.5f);
      for (int lane = 0; lane < 4; ++lane) {
        const uint32_t a = iq::kIq3xxs[b[2 + 8 * group + 2 * lane]];
        const uint32_t z = iq::kIq3xxs[b[3 + 8 * group + 2 * lane]];
        const uint8x8_t grid = vreinterpret_u8_u64(vcreate_u64(uint64_t(a) | (uint64_t(z) << 32)));
        const int16x8_t values = vreinterpretq_s16_u16(vmovl_u8(grid));
        const uint8x8_t signs = vdup_n_u8(uint8_t(iq_signs((aux >> (7 * lane)) & 127)));
        const uint8x8_t positive = vceq_u8(vand_u8(signs, sign_bits), vdup_n_u8(0));
        const uint16x8_t mask = vceqq_u16(vmovl_u8(positive), vdupq_n_u16(255));
        const int16x8_t signed_values = vbslq_s16(mask, values, vnegq_s16(values));
        float* out = y + 32 * group + 8 * lane;
        // Finite half x odd scale x grid magnitude / 4 is exact in fp32, including signed zero.
        vst1q_f32(out, vmulq_f32(scale, vcvtq_f32_s32(vmovl_s16(vget_low_s16(signed_values)))));
        vst1q_f32(out + 4, vmulq_f32(scale, vcvtq_f32_s32(vmovl_s16(vget_high_s16(signed_values)))));
      }
    }
    return;
  }
#endif
  const float d = f16_at(b);
  const uint8_t* qs = b + 2;
  const uint8_t* sas = b + 66;
  for (int ib = 0; ib < 8; ++ib) {
    uint32_t aux;
    memcpy(&aux, sas + 4 * ib, 4);
    const float db = d * (0.5f + float(aux >> 28)) * 0.5f;
    for (int l = 0; l < 4; ++l) {
      const uint32_t s = iq_signs((aux >> (7 * l)) & 127);
      const uint32_t g1 = iq::kIq3xxs[qs[8 * ib + 2 * l]], g2 = iq::kIq3xxs[qs[8 * ib + 2 * l + 1]];
      for (int j = 0; j < 4; ++j) {
        y[32 * ib + 8 * l + j] = db * grid_byte(g1, j) * iq_sign(s, j);
        y[32 * ib + 8 * l + j + 4] = db * grid_byte(g2, j) * iq_sign(s, j + 4);
      }
    }
  }
}

// 3.44 bits: 9-bit indices of 4-value grid points (8 low bits + 1 in qh), 8 stored sign bits per 8 values, a 4-bit
// scale per 32 values (two per byte)
void dq_iq3_s(const uint8_t* b, float* y) {
#if defined(__aarch64__)
  uint16_t half;
  memcpy(&half, b, 2);
  const uintptr_t src = reinterpret_cast<uintptr_t>(b), dst = reinterpret_cast<uintptr_t>(y);
  const bool overlap = src <= dst ? dst - src < 110 : src - dst < 256 * sizeof(float);
  if ((half & 0x7c00u) != 0x7c00u && !overlap) {
    const float d = from_f16(half);
    const uint8x8_t sign_bits = {1, 2, 4, 8, 16, 32, 64, 128};
    for (int group = 0; group < 8; ++group) {
      const int nib = (b[106 + group / 2] >> (4 * (group % 2))) & 15;
      const float32x4_t scale = vdupq_n_f32(d * float(1 + 2 * nib));
      const unsigned high = b[66 + group];
      for (int lane = 0; lane < 4; ++lane) {
        const uint32_t a = iq::kIq3s[b[2 + 8 * group + 2 * lane] | (((high >> (2 * lane)) & 1) << 8)];
        const uint32_t z = iq::kIq3s[b[3 + 8 * group + 2 * lane] | (((high >> (2 * lane + 1)) & 1) << 8)];
        const uint8x8_t grid = vreinterpret_u8_u64(vcreate_u64(uint64_t(a) | (uint64_t(z) << 32)));
        const int16x8_t values = vreinterpretq_s16_u16(vmovl_u8(grid));
        const uint8x8_t positive = vceq_u8(vand_u8(vdup_n_u8(b[74 + 4 * group + lane]), sign_bits), vdup_n_u8(0));
        const uint16x8_t mask = vceqq_u16(vmovl_u8(positive), vdupq_n_u16(255));
        const int16x8_t signed_values = vbslq_s16(mask, values, vnegq_s16(values));
        float* out = y + 32 * group + 8 * lane;
        // Finite half x odd scale x grid magnitude is exact in fp32, including signed zero.
        vst1q_f32(out, vmulq_f32(scale, vcvtq_f32_s32(vmovl_s16(vget_low_s16(signed_values)))));
        vst1q_f32(out + 4, vmulq_f32(scale, vcvtq_f32_s32(vmovl_s16(vget_high_s16(signed_values)))));
      }
    }
    return;
  }
#endif
  const float d = f16_at(b);
  const uint8_t* qs = b + 2;
  const uint8_t* qh = b + 66;
  const uint8_t* signs = b + 74;
  const uint8_t* sc = b + 106;
  for (int ib = 0; ib < 8; ++ib) {
    const int nib = (sc[ib / 2] >> (4 * (ib % 2))) & 0xF;
    const float db = d * float(1 + 2 * nib);
    const uint32_t h = qh[ib];
    for (int l = 0; l < 4; ++l) {
      const uint32_t g1 = iq::kIq3s[qs[8 * ib + 2 * l] | ((h << (8 - 2 * l)) & 256)];
      const uint32_t g2 = iq::kIq3s[qs[8 * ib + 2 * l + 1] | ((h << (7 - 2 * l)) & 256)];
      const uint32_t s = signs[4 * ib + l];
      for (int j = 0; j < 4; ++j) {
        y[32 * ib + 8 * l + j] = db * grid_byte(g1, j) * iq_sign(s, j);
        y[32 * ib + 8 * l + j + 4] = db * grid_byte(g2, j) * iq_sign(s, j + 4);
      }
    }
  }
}

// 1.56 bits: 11-bit indices of 8-value ternary grid points, a 3-bit scale and a +-1/8 shift per 32 values
void dq_iq1_s(const uint8_t* b, float* y) {
  const float d = f16_at(b);
  const uint8_t* qs = b + 2;
  for (int ib = 0; ib < 8; ++ib) {
    uint16_t qh;
    memcpy(&qh, b + 34 + 2 * ib, 2);
    const float dl = d * float(2 * ((qh >> 12) & 7) + 1);
    const float delta = qh & 0x8000 ? -0.125f : 0.125f;
    for (int l = 0; l < 4; ++l) {
      const uint64_t g = iq::kIq1s[qs[4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)];
      for (int j = 0; j < 8; ++j) y[32 * ib + 8 * l + j] = dl * (grid_int8(g, j) + delta);
    }
  }
}

// 1.75 bits: as IQ1_S with a 3-bit scale per 16 values and a shift per 8; the block's fp16 scale is spread over the
// top nibbles of the four scale words
void dq_iq1_m(const uint8_t* b, float* y) {
  const uint8_t* qs = b;
  const uint8_t* qh = b + 32;
  uint16_t sc[4];
  memcpy(sc, b + 48, 8);
  const uint16_t dbits = uint16_t((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
  const float d = from_f16(dbits);
  for (int ib = 0; ib < 8; ++ib) {
    const float dl1 = d * float(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 7) + 1);
    const float dl2 = d * float(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1);
    const uint8_t h0 = qh[2 * ib], h1 = qh[2 * ib + 1];
    const uint32_t idx[4] = {qs[4 * ib + 0] | ((uint32_t(h0) << 8) & 0x700), qs[4 * ib + 1] | ((uint32_t(h0) << 4) & 0x700),
                             qs[4 * ib + 2] | ((uint32_t(h1) << 8) & 0x700), qs[4 * ib + 3] | ((uint32_t(h1) << 4) & 0x700)};
    const float delta[4] = {h0 & 0x08 ? -0.125f : 0.125f, h0 & 0x80 ? -0.125f : 0.125f, h1 & 0x08 ? -0.125f : 0.125f,
                            h1 & 0x80 ? -0.125f : 0.125f};
    for (int l = 0; l < 4; ++l) {
      const uint64_t g = iq::kIq1s[idx[l]];
      const float dl = l < 2 ? dl1 : dl2;
      for (int j = 0; j < 8; ++j) y[32 * ib + 8 * l + j] = dl * (grid_int8(g, j) + delta[l]);
    }
  }
}

void dq_mxfp4(const uint8_t* b, float* y) {
  const float d = e8m0_half(b[0]);
  const uint8_t* q = b + 1;
  for (int j = 0; j < 16; ++j) {
    y[j] = d * float(kFp4x2[q[j] & 0xF]);
    y[j + 16] = d * float(kFp4x2[q[j] >> 4]);
  }
}

void dq_nvfp4(const uint8_t* b, float* y) {
  const uint8_t* q = b + 4;
  for (int s = 0; s < 4; ++s) {
    const float d = ue4m3_half(b[s]);
    for (int j = 0; j < 8; ++j) {
      y[16 * s + j] = d * float(kFp4x2[q[8 * s + j] & 0xF]);
      y[16 * s + j + 8] = d * float(kFp4x2[q[8 * s + j] >> 4]);
    }
  }
}

}  // namespace

bool can_dequantize(uint32_t t) {
  switch (t) {
    case kF32: case kF16: case kBF16: case kQ4_0: case kQ4_1: case kQ5_0: case kQ5_1: case kQ8_0: case kQ2_K:
    case kQ3_K: case kQ4_K: case kQ5_K: case kQ6_K: case kIQ4_NL: case kIQ4_XS: case kMXFP4: case kNVFP4:
    case kIQ2_XXS: case kIQ2_XS: case kIQ2_S: case kIQ3_XXS: case kIQ3_S: case kIQ1_S: case kIQ1_M:
      return true;
    default:
      return false;
  }
}

void dequantize(uint32_t t, const uint8_t* src, float* dst, size_t n) {
  if (t == kF32) {
    memcpy(dst, src, n * 4);
    return;
  }
  if (t == kF16 || t == kBF16) {
    for (size_t i = 0; i < n; ++i) {
      uint16_t h;
      memcpy(&h, src + 2 * i, 2);
      dst[i] = t == kF16 ? from_f16(h) : from_bf16(h);
    }
    return;
  }
  uint32_t be = 0, bb = 0;
  if (!type_block(t, &be, &bb) || !can_dequantize(t)) throw Error("internal: no decoder for " + type_name(t));
  void (*f)(const uint8_t*, float*) = nullptr;
  switch (t) {
    case kQ4_0: f = dq_q4_0; break;
    case kQ4_1: f = dq_q4_1; break;
    case kQ5_0: f = dq_q5_0; break;
    case kQ5_1: f = dq_q5_1; break;
    case kQ8_0: f = dq_q8_0; break;
    case kQ2_K: f = dq_q2_k; break;
    case kQ3_K: f = dq_q3_k; break;
    case kQ4_K: f = dq_q4_k; break;
    case kQ5_K: f = dq_q5_k; break;
    case kQ6_K: f = dq_q6_k; break;
    case kIQ4_NL: f = dq_iq4_nl; break;
    case kIQ4_XS: f = dq_iq4_xs; break;
    case kMXFP4: f = dq_mxfp4; break;
    case kNVFP4: f = dq_nvfp4; break;
    case kIQ2_XXS: f = dq_iq2_xxs; break;
    case kIQ2_XS: f = dq_iq2_xs; break;
    case kIQ2_S: f = dq_iq2_s; break;
    case kIQ3_XXS: f = dq_iq3_xxs; break;
    case kIQ3_S: f = dq_iq3_s; break;
    case kIQ1_S: f = dq_iq1_s; break;
    case kIQ1_M: f = dq_iq1_m; break;
  }
  for (size_t i = 0; i < n / be; ++i) f(src + i * bb, dst + i * be);
}

bool exact4(uint32_t t) { return t == kQ4_0 || t == kQ4_1 || t == kQ4_K; }

void decode4(uint32_t t, const uint8_t* src, uint8_t* codes, float* scale, float* offset, size_t n) {
  if (t == kQ4_K) {
    for (size_t i = 0; i < n / 256; ++i) q4_k_parts(src + 144 * i, codes + 256 * i, scale + 8 * i, offset + 8 * i);
    return;
  }
  const size_t bb = t == kQ4_0 ? 18 : 20;
#if defined(__aarch64__)
  const uintptr_t in = reinterpret_cast<uintptr_t>(src), out = reinterpret_cast<uintptr_t>(codes);
  const bool overlap = out >= in ? out - in < n / 32 * bb : in - out < n / 32 * 32;
#endif
  for (size_t i = 0; i < n / 32; ++i) {
    const uint8_t* b = src + bb * i;
    const float d = f16_at(b);
    scale[i] = d;
    offset[i] = t == kQ4_0 ? -8.0f * d : f16_at(b + 2);
    const uint8_t* q = b + (t == kQ4_0 ? 2 : 4);
#if defined(__aarch64__)
    if ((t == kQ4_0 || t == kQ4_1) && !overlap) {
      const uint8x16_t packed = vld1q_u8(q);
      vst1q_u8(codes + 32 * i, vandq_u8(packed, vdupq_n_u8(15)));
      vst1q_u8(codes + 32 * i + 16, vshrq_n_u8(packed, 4));
      continue;
    }
#endif
    for (int j = 0; j < 16; ++j) {
      codes[32 * i + j] = q[j] & 0xF;
      codes[32 * i + j + 16] = q[j] >> 4;
    }
  }
}

void pack_codes(const uint8_t* codes, size_t n, int bits, uint32_t* words) {
  if (bits == 5 && n % 32 == 0) {
    const uintptr_t src = reinterpret_cast<uintptr_t>(codes), dst = reinterpret_cast<uintptr_t>(words);
    const bool overlap = dst >= src ? dst - src < n : src - dst < n / 32 * 20;
    if (!overlap) {
      // A complete 160-bit tile writes each output word once, without clearing the row first.
      for (size_t j = 0; j < n; j += 32, words += 5) {
        const auto q = [&](size_t k) { return uint32_t(codes[j + k] & 31); };
        words[0] = q(0) | q(1) << 5 | q(2) << 10 | q(3) << 15 | q(4) << 20 | q(5) << 25 | q(6) << 30;
        words[1] = q(6) >> 2 | q(7) << 3 | q(8) << 8 | q(9) << 13 | q(10) << 18 | q(11) << 23 | q(12) << 28;
        words[2] = q(12) >> 4 | q(13) << 1 | q(14) << 6 | q(15) << 11 | q(16) << 16 | q(17) << 21 |
                   q(18) << 26 | q(19) << 31;
        words[3] = q(19) >> 1 | q(20) << 4 | q(21) << 9 | q(22) << 14 | q(23) << 19 | q(24) << 24 | q(25) << 29;
        words[4] = q(25) >> 3 | q(26) << 2 | q(27) << 7 | q(28) << 12 | q(29) << 17 | q(30) << 22 | q(31) << 27;
      }
      return;
    }
  }
  if (bits == 8) {
    memcpy(words, codes, n);  // byte j of the little-endian stream is code j
    return;
  }
  if (bits == 4) {
    size_t start = 0;
#if defined(__aarch64__)
    const size_t used = n / 8 * 8;
    const uintptr_t src = reinterpret_cast<uintptr_t>(codes), dst = reinterpret_cast<uintptr_t>(words);
    const bool overlap = dst >= src ? dst - src < used : src - dst < used / 2;
    if (!overlap) {
      const uint8x16_t mask = vdupq_n_u8(15);
      for (; used - start >= 32; start += 32) {
        const uint8x16_t a = vandq_u8(vld1q_u8(codes + start), mask);
        const uint8x16_t b = vandq_u8(vld1q_u8(codes + start + 16), mask);
        const uint8x16_t packed = vorrq_u8(vuzp1q_u8(a, b), vshlq_n_u8(vuzp2q_u8(a, b), 4));
        vst1q_u8(reinterpret_cast<uint8_t*>(words) + start / 2, packed);
      }
    }
#endif
    for (size_t w = start / 8; w < n / 8; ++w) {
      uint32_t v = 0;
      for (int k = 0; k < 8; ++k) v |= uint32_t(codes[8 * w + k] & 0xF) << (4 * k);
      words[w] = v;
    }
    return;
  }
  const size_t nw = n * size_t(bits) / 32;
  std::fill(words, words + nw, 0u);
  const uint32_t mask = (1u << bits) - 1u;
  for (size_t j = 0; j < n; ++j) {
    const uint64_t bit = uint64_t(j) * uint64_t(bits);
    const uint32_t c = codes[j] & mask, s = uint32_t(bit % 32);
    words[bit / 32] |= c << s;
    if (s + uint32_t(bits) > 32) words[bit / 32 + 1] |= c >> (32 - s);
  }
}

namespace {

// codes of one group for a (bf16) grid, and the squared error
double fit_codes(const float* x, int gs, int n, float s, float b, uint8_t* q) {
  double e = 0;
  const float inv = s != 0.0f ? 1.0f / s : 0.0f;
  for (int i = 0; i < gs; ++i) {
    float c = std::nearbyint((x[i] - b) * inv);
    c = std::min(std::max(c, 0.0f), float(n));
    q[i] = uint8_t(c);
    const double d = double(c) * s + b - x[i];
    e += d * d;
  }
  return e;
}

}  // namespace

void quantize(const float* x, size_t total, int bits, int gs, uint8_t* codes, uint16_t* scales, uint16_t* biases) {
  const int n = (1 << bits) - 1;
  uint8_t q[256], qb[256];
  for (size_t g = 0; g < total / size_t(gs); ++g) {
    const float* v = x + g * size_t(gs);
    float lo = v[0], hi = v[0];
    for (int i = 1; i < gs; ++i) lo = std::min(lo, v[i]), hi = std::max(hi, v[i]);
    double best = INFINITY;
    uint16_t bs = 0, bb = 0;
    // the grid (s, b) rounded to bf16: its codes into q; kept when its error is the lowest so far
    auto consider = [&](float s, float b) {
      if (std::fabs(s) < 1e-30f) s = 1e-30f;
      const uint16_t s16 = to_bf16(s), b16 = to_bf16(b);
      const double e = fit_codes(v, gs, n, from_bf16(s16), from_bf16(b16), q);
      if (e < best) {
        best = e, bs = s16, bb = b16;
        memcpy(qb, q, size_t(gs));
      }
    };
    if (!(hi > lo)) {  // a constant group (or non-finite values): code 0 and the value as bias
      consider(1.0f, lo);
    } else {
      const float s0 = (hi - lo) / float(n);
      // symmetric: codes centred on (n + 1) / 2, as Q8_0 / Q6_K / Q3_K grids are
      const float am = std::max(std::fabs(lo), std::fabs(hi));
      const int half = (n + 1) / 2;
      const float s1 = am / float(half - 1);
      consider(s1, -float(half) * from_bf16(to_bf16(s1)));
      // the min-max grid, then least-squares refits of it (each from the previous grid's codes)
      consider(s0, lo);
      for (int it = 0; it < 4; ++it) {
        double sq = 0, sx = 0, sqq = 0, sqx = 0;
        for (int i = 0; i < gs; ++i) sq += q[i], sx += v[i], sqq += double(q[i]) * q[i], sqx += double(q[i]) * v[i];
        const double m = gs, var = sqq - sq * sq / m;
        if (!(var > 0)) break;
        const double ns = (sqx - sq * sx / m) / var, nb = (sx - ns * sq) / m;
        if (!std::isfinite(ns) || !std::isfinite(nb)) break;
        consider(float(ns), float(nb));
      }
    }
    memcpy(codes + g * size_t(gs), qb, size_t(gs));
    scales[g] = bs;
    biases[g] = bb;
  }
}


// ---- IQ blocks in the GPU layout of src/kernels/wfmt_iq.metal ----

bool iq_gpu_type(uint32_t type) {
  return type == kIQ2_XXS || type == kIQ2_S || type == kIQ3_S || type == kIQ3_XXS || type == kIQ1_S;
}

size_t iq_block_bytes(uint32_t type) {
  uint32_t e = 0, b = 0;
  return type_block(type, &e, &b) && e == 256 ? b : 0;
}

int iq_bits(uint32_t type) { return type == kIQ3_S || type == kIQ3_XXS ? 3 : type == kIQ1_S ? 1 : 2; }

std::string iq_suffix(uint32_t type) {
  std::string s;
  for (char c : type_name(type))
    if (c != '_') s += char(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

uint32_t iq_mode_type(const std::string& mode) {
  for (uint32_t t : {kIQ1_S, kIQ2_XXS, kIQ2_S, kIQ3_XXS, kIQ3_S}) {
    std::string m = type_name(t);
    for (auto& c : m) c = char(std::tolower(static_cast<unsigned char>(c)));
    if (m == mode) return t;
  }
  return 0;
}

IqLayout iq_layout(uint32_t type, size_t rows, size_t K) {
  // per 32 values: Q 8 bytes (IQ1_S 4); IQ2_S: X1 qh | scales << 8 (2); IQ3_S: X1 signs (4), X2 qh (1), X3 scales (a
  // byte per 64); IQ3_XXS: X1 signs | scale (4); IQ1_S: X1 qh (2)
  IqLayout L;
  const size_t sb = rows * (K / 32);
  const size_t x1 = type == kIQ2_S || type == kIQ1_S ? 2 : type == kIQ3_S || type == kIQ3_XXS ? 4 : 0;
  const size_t x2 = type == kIQ3_S ? 1 : 0;
  L.x1 = sb * (type == kIQ1_S ? 4 : 8);
  L.x2 = L.x1 + sb * x1;
  L.x3 = L.x2 + sb * x2;
  L.d = L.x3 + (type == kIQ3_S ? sb / 2 : 0);
  L.bytes = L.d + rows * (K / 256) * 2;
  return L;
}

void iq_repack(uint32_t type, const uint8_t* src, size_t rows, size_t K, size_t r0, size_t r1, uint8_t* dst) {
  const IqLayout L = iq_layout(type, rows, K);
  const size_t nb = K / 256, bb = iq_block_bytes(type);
  for (size_t r = r0; r < r1; ++r)
    for (size_t b = 0; b < nb; ++b) {
      const uint8_t* blk = src + ((r - r0) * nb + b) * bb;
      const size_t blkid = r * nb + b, sb0 = blkid * 8;  // the block's first sub-block
      memcpy(dst + L.d + blkid * 2, blk, 2);
      if (type == kIQ3_S) memcpy(dst + L.x3 + sb0 / 2, blk + 106, 4);  // scales: sub-blocks 2m, 2m + 1 in byte m
      for (size_t j = 0; j < 8; ++j) {
        uint8_t* q = dst + (sb0 + j) * (type == kIQ1_S ? 4 : 8);
        if (type == kIQ2_XXS) {
          memcpy(q, blk + 2 + 8 * j, 8);
        } else if (type == kIQ3_XXS) {
          memcpy(q, blk + 2 + 8 * j, 8);                            // index bytes
          memcpy(dst + L.x1 + (sb0 + j) * 4, blk + 66 + 4 * j, 4);  // signs | scale << 28
        } else if (type == kIQ1_S) {
          memcpy(q, blk + 2 + 4 * j, 4);                            // index low bytes
          memcpy(dst + L.x1 + (sb0 + j) * 2, blk + 34 + 2 * j, 2);  // qh
        } else if (type == kIQ2_S) {
          memcpy(q, blk + 2 + 4 * j, 4);       // index low bytes
          memcpy(q + 4, blk + 34 + 4 * j, 4);  // sign bytes
          const uint16_t e = uint16_t(blk[66 + j] | (blk[74 + j] << 8));  // qh | scales << 8
          memcpy(dst + L.x1 + (sb0 + j) * 2, &e, 2);
        } else {
          memcpy(q, blk + 2 + 8 * j, 8);                            // index low bytes
          memcpy(dst + L.x1 + (sb0 + j) * 4, blk + 74 + 4 * j, 4);  // signs
          dst[L.x2 + sb0 + j] = blk[66 + j];                        // qh
        }
      }
    }
}

}  // namespace ni::gguf
