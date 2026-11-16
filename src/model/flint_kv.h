// NI_KV_I8 and Flint (#1257): Flint's attention kernels read the int8 KV rows wrongly (verify returns NaN at every
// position on a 4-bit model, a ~58K-token prompt page-faults the GPU), so the loader refuses the switch until that is
// fixed. Plain C++ so a CPU test can check the message.
#pragma once
#include "metal/gpu.h"

namespace ni {

// The one message a Flint load prints when int8 KV is asked for; nullptr: nothing to refuse.
inline const char* flint_kv_i8_refusal(bool int8_kv) {
  return int8_kv ? "int8 KV cache isn't supported with Flint models yet; unset NI_KV_I8" : nullptr;
}

// What the Flint constructor asks: the answer the Metal library was compiled with.
inline const char* flint_kv_i8_refusal() { return flint_kv_i8_refusal(kv_i8()); }

// What unsupported_model asks, before any engine is built or released (a model switch is refused, the old model stays
// up): the environment's answer now.
inline const char* flint_kv_i8_requested_refusal() { return flint_kv_i8_refusal(kv_i8_requested()); }

}  // namespace ni
