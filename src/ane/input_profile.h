// Profile policy for Qwen's dense ANE input geometry. Runtime availability, format,
// memory and startup checks still decide whether an eligible split can run.
#pragma once

#include "engine/device_profile.h"
#include "engine/profile.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace ni::ane {

inline bool input_profile_enabled(uint32_t segment, bool matrix) {
  if (segment == 2560) {
    const char* v = std::getenv("NI_ANE");
    return !v || std::strcmp(v, "0") != 0;  // preserve the existing 4B/27B defaults
  }
  if (segment != 2048) return false;
  // Matrix-unit Macs keep the GPU default; an explicit NI_ANE override still wins.
  return profile_switch("NI_ANE", !matrix, "dense ANE FFN with 2048-channel inputs");
}

inline bool input_profile_enabled(uint32_t segment) {
  if (segment != 2048) return input_profile_enabled(segment, false);
  static const bool on = [] {
    const char* v = std::getenv("NI_ANE");
    // Explicit overrides and precise need no device read (including CPU planning checks).
    const bool automatic = (!v || !*v) && profile() == Profile::kFast;
    return input_profile_enabled(2048, automatic && matrix_units());
  }();
  return on;
}

}  // namespace ni::ane
