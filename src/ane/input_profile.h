// Profile policy for Qwen's dense ANE input geometry. Runtime availability, format,
// memory and startup checks still decide whether an eligible split can run.
#pragma once

#include "engine/profile.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace ni::ane {

inline bool input_profile_enabled(uint32_t segment) {
  if (segment == 2560) {
    const char* v = std::getenv("NI_ANE");
    return !v || std::strcmp(v, "0") != 0;  // preserve the existing 4B/27B defaults
  }
  if (segment != 2048) return false;
  // The profile table's ane_segmented_fast row covers usable ANEs. Hardware and
  // calibration checks remain in init_ane; cache keys use this same policy.
  static const bool on = profile_switch("NI_ANE", true, "dense ANE FFN with 2048-channel inputs");
  return on;
}

}  // namespace ni::ane
