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

}  // namespace ni
