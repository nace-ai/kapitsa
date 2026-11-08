#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
__attribute__((noinline)) void candidate_q8(const float* x, int64_t K, uint8_t* q, uint8_t* s) {
  for (int64_t b = 0; b < K / 32; ++b) {
    const float* block = x + b * 32;
    auto scalar = [&] {
      float amax = 0.0f;
      for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
      const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
      const __fp16 dh = __fp16(d);
      memcpy(s + b * 2, &dh, 2);
      for (int i = 0; i < 32; ++i) {
        const int v = int(std::lround(x[b * 32 + i] * id));
        q[b * 32 + i] = uint8_t(int8_t(std::clamp(v, -127, 127)));
      }
    };
    // Keep sequential source reads when a destination overlaps this input block.
    auto overlaps = [](const void* a, size_t an, const void* c, size_t cn) {
      const uintptr_t av = reinterpret_cast<uintptr_t>(a), cv = reinterpret_cast<uintptr_t>(c);
      return av <= cv ? cv - av < an : av - cv < cn;
    };
    if (overlaps(block, 32 * sizeof(float), q + b * 32, 32) ||
        overlaps(block, 32 * sizeof(float), s + b * 2, 2)) {
      scalar();
      continue;
    }
    float32x4_t values[8];
    uint32x4_t nonfinite = vdupq_n_u32(0);
    const uint32x4_t exponent = vdupq_n_u32(0x7f800000u);
    for (int i = 0; i < 8; ++i) {
      values[i] = vld1q_f32(block + i * 4);
      const uint32x4_t bits = vreinterpretq_u32_f32(values[i]);
      nonfinite = vorrq_u32(nonfinite, vceqq_u32(vandq_u32(bits, exponent), exponent));
    }
    // Classify with integer bits before any floating operations, including for signaling NaNs.
    if (vmaxvq_u32(nonfinite)) {
      scalar();
      continue;
    }
    float32x4_t maximum = vdupq_n_f32(0.0f);
    for (const float32x4_t v : values) maximum = vmaxq_f32(maximum, vabsq_f32(v));
    const float amax = vmaxvq_f32(maximum);
    const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
    if (!std::isfinite(id)) {
      scalar();
      continue;
    }
    const __fp16 dh = __fp16(d);
    memcpy(s + b * 2, &dh, 2);
    for (int i = 0; i < 8; i += 2) {
      int32x4_t lo = vcvtaq_s32_f32(vmulq_n_f32(values[i], id));
      int32x4_t hi = vcvtaq_s32_f32(vmulq_n_f32(values[i + 1], id));
      lo = vminq_s32(vmaxq_s32(lo, vdupq_n_s32(-127)), vdupq_n_s32(127));
      hi = vminq_s32(vmaxq_s32(hi, vdupq_n_s32(-127)), vdupq_n_s32(127));
      const int16x8_t pair = vcombine_s16(vmovn_s32(lo), vmovn_s32(hi));
      vst1_s8(reinterpret_cast<int8_t*>(q + b * 32 + i * 4), vmovn_s16(pair));
    }
  }
}
