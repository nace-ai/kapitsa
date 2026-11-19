// Hy4-flash's Metal kernels (src/kernels/hy4/) are a library of their own (cmake/embed_kernels_hy4.cmake), compiled
// when the first Hy4 pipeline is asked for: every other model's load compiles the base library without them.
#pragma once

#include "metal/gpu.h"

#include <mutex>
#include <string>

namespace ni {

extern const char* const kHy4KernelSource;

namespace hy4 {

inline void compile_library() {
  static std::once_flag once;
  std::call_once(once, [] { Gpu::get().compile_extra("Hy4", kHy4KernelSource); });
}

// A pipeline from the Hy4 library or the base library (pipeline() looks in the base library first)
inline Pipeline pipeline(const std::string& name, bool required = true) {
  compile_library();
  return Gpu::get().pipeline(name, required);
}

}  // namespace hy4
}  // namespace ni
