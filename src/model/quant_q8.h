#pragma once
#include <cstdint>

namespace ni {

// Quantizes a K-element fp32 row to Q8_0 (int8 codes + fp16 block-32 scales, d = max|x| / 127).
// Implemented with NEON in src/model/nemotron_h.cpp.
void nh_quant_q8(const float* x, int64_t K, uint8_t* q, uint8_t* s);

}  // namespace ni
