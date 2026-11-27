#pragma once
#include "metal/gpu.h"

#include <string>

namespace ni {

extern const char* const kInk2KernelSource;

namespace ink2 {

inline void compile_kernels() {
  Gpu::get().compile_extra("Ink2", kInk2KernelSource);
}

inline Pipeline pipeline(const std::string& name, bool required = true) {
  compile_kernels();
  return Gpu::get().pipeline(name, required);
}

}  // namespace ink2
}  // namespace ni
