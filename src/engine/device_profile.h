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

// A row's device half. A class matches its own chips, and kMatrix also matches kWide (a wide chip has matrix
// units). kName is a narrow exception: the name must match, and the row
// has to cite a win of at least 10% that the wider class does not get.
enum class DefaultScope { kLegacy, kSimd, kMatrix, kWide, kName };

struct DeviceFacts {
  int apple_family = 0;  // Gpu::apple_family(): 7 M1, 8 M2, 9 M3 and M4, 10 M5, higher when a later family exists.
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
  const char* value;  // what the table would turn on, once the call site reads it
  int win_pct;        // lower bound of the cited win, in percent. 0 if the row cites none.
  int win_pct_hi;     // 0 when win_pct is a single number, else the upper bound
  bool exclusive;     // true only when the row cites a miss on the wider class. A class row is false.
  const char* where;  // the Mac the number was measured on, not a sentence
  bool live;          // false: the call site still has its own check
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

// The duplicated checks. The no-argument forms read the live GPU by calling the same Gpu methods the sites used. The
// DeviceFacts forms are those predicates for the CPU test. The table above is still not what the sites read.
bool matrix_units(const DeviceFacts&);  // apple_family() >= 10
bool legacy_gpu(const DeviceFacts&);    // apple_family() <= 7: the M1 path
bool m1_max_32(const DeviceFacts&);     // name "Apple M1 Max" and 32 cores
bool wide_cores(const DeviceFacts&);    // matrix units and 40 cores or more
bool m4_10(const DeviceFacts&);         // name "Apple M4" and exactly 10 cores
bool m4_pro_gpu(const DeviceFacts&);    // name "Apple M4 Pro" (14 or 20 cores)
bool m4_pro_20(const DeviceFacts&);     // name "Apple M4 Pro" and exactly 20 cores
bool ultra_gpu(const DeviceFacts&);     // the name contains "Ultra": two dies
bool pre_m3_gpu(const DeviceFacts&);    // apple_family() < 9: M1 and M2
bool m4_ten_core(const DeviceFacts&);   // name "Apple M4" and 10 cores or fewer

bool matrix_units();
bool legacy_gpu();
bool m1_max_32();
bool wide_cores();
bool m4_10();
bool m4_pro_gpu();
bool m4_pro_20();
bool ultra_gpu();
bool pre_m3_gpu();
bool m4_ten_core();

}  // namespace ni
