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

const DeviceProfile& current_device() {
  // Resolved once, the way the call sites cached their own checks. No default_on row keys on RAM or the ANE, so they stay
  // unset here; doctor resolves its own with them filled for its print.
  static const DeviceProfile p = resolve_device(facts_from_gpu(0, false));
  return p;
}

bool default_on(Knob k, const ModelFacts& m) { return default_on(k, current_device(), m); }

bool matrix_units() { return Gpu::get().apple_family() >= 10; }

bool legacy_gpu() { return Gpu::get().apple_family() <= 7; }

bool m1_max_32() { return Gpu::get().name() == "Apple M1 Max" && Gpu::get().core_count() == 32; }

bool wide_cores() { return matrix_units() && Gpu::get().core_count() >= 40; }

bool m4_10() { return Gpu::get().name() == "Apple M4" && Gpu::get().core_count() == 10; }

bool m4_pro_gpu() { return Gpu::get().name() == "Apple M4 Pro"; }

bool m4_pro_20() { return Gpu::get().name() == "Apple M4 Pro" && Gpu::get().core_count() == 20; }

bool ultra_gpu() { return Gpu::get().name().find("Ultra") != std::string::npos; }

bool pre_m3_gpu() { return Gpu::get().apple_family() < 9; }

bool m4_ten_core() { return Gpu::get().name() == "Apple M4" && Gpu::get().core_count() <= 10; }

}  // namespace ni
