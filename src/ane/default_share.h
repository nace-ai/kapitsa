// The Neural Engine's default share of a dense FFN when no calibration says (NI_ANE), kept apart from ffn_split.cpp
// so the CPU test (device_profile_test) can pin its values without the ANE runtime.
#pragma once

#include <cstdint>
#include <string>

namespace ni::ane {

// Default share (units of 512 channels) for a chip without a calibration, for an FFN of `total` units
// on a GPU of `cores` cores; 0 = the GPU alone.
uint32_t default_units(const std::string& chip, int cores, uint32_t total, bool wave, bool gpu_mma = false);
// The smallest prefill chunk that splits when no calibration says: 512 rows, or 1024 on an M1 or M2 whose plain split
// at `units` of `total` would be slower than the GPU alone (chunks under 1024 rows run the plain split).
uint32_t default_min_rows(const std::string& chip, int cores, uint32_t units, uint32_t total);

}  // namespace ni::ane
