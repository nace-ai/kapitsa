// What this Mac is, as a few classes, and the table of defaults those classes decide.
//
// Call sites still decide today. This module is what they move to: one resolve from injected facts (so a CPU test
// can build an M5 Max without a GPU), one table, and `kapitsa doctor --print-profile` printing the result before it
// scans a model. A chip name, a core count or a RAM size is a row here, not a new check in a model file.
#pragma once

#include <string>

namespace ni {

// Four classes. Family 7 and below is the M1 path (the k64 kernels). Family 8 and 9 have no tensor units (M2, M3,
// M4). Family 10 and up has matrix units; 40 cores or more is the wide GPU the 35B wave was measured on. A name, a
// core count and a RAM size are not classes.
enum class ChipClass { kLegacy, kSimd, kMatrix, kWide };

enum class ModelClass { kOther, kQwen35A3b };

// A row's device half. A class matches only itself. kName is a narrow exception: the name must match, and the row
// has to cite a win of at least 10% that the wider class does not get.
enum class DefaultScope { kLegacy, kSimd, kMatrix, kWide, kName };

struct DeviceFacts {
  int apple_family = 0;  // Gpu::apple_family(): 7 M1, 8 M2, 9 M3 and M4, 10 M5. 7 is also the fallback.
  int cores = 0;         // Gpu::core_count(), which honors NI_GPU_CORES and the unread-IORegistry fallback.
  bool cores_measured = false;
  std::string name;
  long ram_bytes = 0;
  bool ane = false;
};

struct ModelFacts {
  bool moe = false;
  int hidden = 0;
  int layers = 0;
  int experts = 0;
  int top_k = 0;
  int moe_i = 0;
  int shared_i = 0;
  bool clover = false;  // same 35B-A3B shape; not measured, so not kQwen35A3b
};

struct DeviceProfile {
  ChipClass chip = ChipClass::kLegacy;
  int cores = 0;
  bool cores_measured = false;
  std::string name;
  long ram_bytes = 0;
  bool ane = false;
};

struct DefaultRow {
  const char* id;
  ModelClass model;
  DefaultScope scope;
  const char* chip_name;  // kName only
  const char* value;      // what the table would turn on, once the call site reads it
  const char* measured_on;
  bool live;  // false: the call site still has its own check
};

DeviceProfile resolve_device(const DeviceFacts&);
ModelClass resolve_model(const ModelFacts&);
const char* chip_class_name(ChipClass);
const char* model_class_name(ModelClass);

int default_count();
const DefaultRow& default_at(int i);
bool default_matches(const DefaultRow&, const DeviceProfile&);

// The doctor lines. as_json is the same fields for programs.
std::string format_profile(const DeviceProfile&, bool as_json);

// The one live read. Doctor passes RAM and the Neural Engine; this reads the GPU. Not linked into the CPU test.
DeviceFacts facts_from_gpu(long ram_bytes, bool ane);

}  // namespace ni
