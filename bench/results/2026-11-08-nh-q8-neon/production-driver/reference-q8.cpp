#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
__attribute__((noinline)) void reference_q8(const float* x, int64_t K, uint8_t* q, uint8_t* s) {
  for (int64_t b = 0; b < K / 32; ++b) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
    const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
    const __fp16 dh = __fp16(d);
    memcpy(s + b * 2, &dh, 2);
    for (int i = 0; i < 32; ++i) {
      const int v = int(std::lround(x[b * 32 + i] * id));
      q[b * 32 + i] = uint8_t(int8_t(std::clamp(v, -127, 127)));
    }
  }
}
