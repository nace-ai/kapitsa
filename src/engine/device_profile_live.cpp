// The one place, besides Gpu itself, that reads the live device into DeviceFacts. The CPU test does not link this.
#include "engine/device_profile.h"

#include "metal/gpu.h"

namespace ni {

DeviceFacts facts_from_gpu(long ram_bytes, bool ane) {
  Gpu& g = Gpu::get();
  DeviceFacts f;
  f.apple_family = g.apple_family();
  f.cores = g.core_count();
  f.cores_measured = g.core_count_measured();
  f.name = g.name();
  f.ram_bytes = ram_bytes;
  f.ane = ane;
  return f;
}

bool matrix_units() { return Gpu::get().apple_family() >= 10; }

bool legacy_gpu() { return Gpu::get().apple_family() <= 7; }

bool m1_max_32() { return Gpu::get().name() == "Apple M1 Max" && Gpu::get().core_count() == 32; }

bool wide_cores() { return matrix_units() && Gpu::get().core_count() >= 40; }

bool named_m5_max() { return Gpu::get().name().find("M5 Max") != std::string::npos; }

bool m4_ten_core() { return Gpu::get().name() == "Apple M4" && Gpu::get().core_count() <= 10; }

}  // namespace ni
