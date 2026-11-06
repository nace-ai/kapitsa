// What this Mac is, as a few classes, and the table of defaults those classes decide.
//
// The table decides: a gated call site reads its row through default_on() after its env override, one resolve from
// injected facts (so a CPU test can build an M5 Max without a GPU), and `kapitsa doctor --print-profile` printing
// the same table before it scans a model. A chip name, a core count or a RAM size is a row here, not a new check in
// a model file.
#pragma once

#include <string>

namespace ni {

// Four classes. Family 7 and below is the M1 path (the k64 kernels). Family 8 and 9 have no tensor units (M2, M3,
// M4). Family 10 and up has matrix units; 40 cores or more is the wide GPU the 35B wave was measured on. A name, a
// core count and a RAM size are not classes.
enum class ChipClass { kLegacy, kSimd, kMatrix, kWide };

// The model half of a row. kOther matches every model. kQwen35Shape is the 35B-A3B's prefill shape (MoE, hidden
// 2048, 40 layers), which Clover-35B-A3B shares; kQwen35A3b is the full shape with the drafter fields, Clover out.
enum class ModelClass { kOther, kQwen35Shape, kQwen35A3b };

// The knobs the table decides, in the table's order. A call site passes the enum, not the string: a typo is a
// compile error, not a default that silently reads as off. kNone is a documentation row (a class or a guard the
// call site still reads itself); default_on never looks one up.
enum class Knob { kPfWave, kKflashHist, kGemmSlices, kHcWide, kMoe4Dx, kTreeG6, kKflashNarrow, kNone };

// A row's device half. A class matches its own chips, and kMatrix also matches kWide (a wide chip has matrix
// units). kName is a narrow exception: the name must match, and the row
// has to cite a win of at least 10% that the wider class does not get.
// kAneNonMatrix requires an ANE and excludes matrix/wide GPUs; runtime share checks still apply.
// kGuard is not that. It documents a predicate the call site still reads: a shape, a RAM size, or a
// correctness miss on the wider chip. It is not a class, and it does not claim a 10% win.
enum class DefaultScope { kLegacy, kSimd, kMatrix, kWide, kAneNonMatrix, kName, kGuard };

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
  bool clover = false;  // DiffuSpace's Clover: the 35B-A3B shape; kQwen35A3b leaves it out, nothing has been run on it (#369)
};

struct DeviceProfile {
  ChipClass chip = ChipClass::kLegacy;
  int apple_family = 0;  // kept so a guard that keys on the family (pre_m3_gpu) matches in doctor
  int cores = 0;
  bool cores_measured = false;
  std::string name;
  long ram_bytes = 0;
  bool ane = false;
};

struct DefaultRow {
  const char* id;
  Knob knob;
  ModelClass model;
  DefaultScope scope;
  const char* chip_name;  // kName, and a kGuard row that has an exact name. Null when the guard matches a substring.
  const char* value;  // what the call site turns on when the row matches. A kGuard row is not read that way yet.
  int win_pct;        // lower bound of the cited win, in percent. 0 if the row cites none.
  int win_pct_hi;     // 0 when win_pct is a single number, else the upper bound
  bool exclusive;     // true only when the row cites a miss on the wider class. A class row is false.
  const char* where;  // the measured Mac. Append "; widened by judgment, sweep 1" when that class
                      // take was not measured.
  bool live;          // true: the call site's default is this row (its env override and runtime
                      // conditions - a slot count, a prompt size - stay at the site)
};

DeviceProfile resolve_device(const DeviceFacts&);
ModelClass resolve_model(const ModelFacts&);
const char* chip_class_name(ChipClass);
const char* model_class_name(ModelClass);

int default_count();
const DefaultRow& default_at(int i);
bool default_matches(const DefaultRow&, const DeviceProfile&);  // the row's device half
bool model_class_matches(ModelClass, const ModelFacts&);        // the row's model half; kOther matches all

// The one read a call site makes after its env override: the row's value is "on" and both halves match. False when
// the knob has no row. The 3-fact form is the pure one the CPU test links; the 2-fact form resolves the live device.
bool default_on(Knob, const DeviceProfile&, const ModelFacts&);
bool default_on(Knob, const ModelFacts&);

// The doctor lines. as_json is the same fields for programs.
std::string format_profile(const DeviceProfile&, bool as_json);

// The one live read. Doctor passes RAM and the Neural Engine; this reads the GPU. Not linked into the CPU test.
DeviceFacts facts_from_gpu(long ram_bytes, bool ane);

// The live device, resolved once. RAM and the ANE stay 0/false in it: no row keys on them, and doctor resolves its
// own with them filled for its print.
const DeviceProfile& current_device();

// The named checks the remaining gates call. The no-argument forms read the live GPU by calling the same Gpu methods
// the sites used. The DeviceFacts forms are those predicates for the CPU test.
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
