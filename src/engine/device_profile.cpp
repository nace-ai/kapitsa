#include "engine/device_profile.h"

#include <cstdio>
#include <sstream>

namespace ni {
namespace {

// The 35B-A3B shape the wave and the KFlash lanes were measured on. Clover has the same numbers and is excluded by
// the caller (config.json model_type), not by a second chip check.
bool qwen35_a3b(const ModelFacts& m) {
  return m.moe && !m.clover && m.hidden == 2048 && m.layers == 40 && m.experts == 256 && m.top_k == 8 &&
         m.moe_i == 512 && m.shared_i == 512;
}

const DefaultRow kDefaults[] = {
    // Wide class, not a chip name: apple10 and 40 cores or more. The call site also wants one slot and a 4-bit
    // group-64 expert pack; those are not a device class, so they stay at the call site when it moves.
    {"pf_wave", ModelClass::kQwen35A3b, DefaultScope::kWide, nullptr, "on",
     "M5 Max, 16,288 ids 2,767 -> 2,620 ms (-5.3%). Not a name.", false},
    // Narrow: the name, not the matrix class. +12 to +15% on the M5 Pro, and not shown slower on the wider class,
    // so the name stays until a later row cites that miss or widens. The selector update uses the same gate.
    {"kflash_hist", ModelClass::kQwen35A3b, DefaultScope::kName, "Apple M5 Pro", "on",
     "M5 Pro, +12 to +15% tok/s in 3 of 3. Not shown slower on the wider matrix class, so the name stays.", false},
};

}  // namespace

DeviceProfile resolve_device(const DeviceFacts& f) {
  DeviceProfile p;
  p.cores = f.cores;
  p.cores_measured = f.cores_measured;
  p.name = f.name;
  p.ram_bytes = f.ram_bytes;
  p.ane = f.ane;
  if (f.apple_family >= 10 && f.cores >= 40) p.chip = ChipClass::kWide;
  else if (f.apple_family >= 10) p.chip = ChipClass::kMatrix;
  else if (f.apple_family <= 7) p.chip = ChipClass::kLegacy;
  else p.chip = ChipClass::kSimd;
  return p;
}

ModelClass resolve_model(const ModelFacts& m) { return qwen35_a3b(m) ? ModelClass::kQwen35A3b : ModelClass::kOther; }

const char* chip_class_name(ChipClass c) {
  switch (c) {
    case ChipClass::kLegacy: return "legacy";
    case ChipClass::kSimd: return "simd";
    case ChipClass::kMatrix: return "matrix";
    case ChipClass::kWide: return "wide";
  }
  return "legacy";
}

const char* model_class_name(ModelClass m) {
  return m == ModelClass::kQwen35A3b ? "qwen35-a3b" : "other";
}

int default_count() { return int(sizeof(kDefaults) / sizeof(kDefaults[0])); }

const DefaultRow& default_at(int i) { return kDefaults[i]; }

bool default_matches(const DefaultRow& row, const DeviceProfile& dev) {
  switch (row.scope) {
    case DefaultScope::kLegacy: return dev.chip == ChipClass::kLegacy;
    case DefaultScope::kSimd: return dev.chip == ChipClass::kSimd;
    case DefaultScope::kMatrix: return dev.chip == ChipClass::kMatrix;
    case DefaultScope::kWide: return dev.chip == ChipClass::kWide;
    case DefaultScope::kName: return row.chip_name && dev.name == row.chip_name;
  }
  return false;
}

namespace {

const char* scope_name(const DefaultRow& row) {
  if (row.scope == DefaultScope::kName) return row.chip_name ? row.chip_name : "name";
  switch (row.scope) {
    case DefaultScope::kLegacy: return "legacy";
    case DefaultScope::kSimd: return "simd";
    case DefaultScope::kMatrix: return "matrix";
    case DefaultScope::kWide: return "wide";
    case DefaultScope::kName: return "name";
  }
  return "name";
}

std::string mac_line(const DeviceProfile& p) {
  char buf[160];
  std::snprintf(buf, sizeof(buf), "%s, %d cores%s, %.0f GB RAM, Neural Engine %s", p.name.c_str(), p.cores,
                p.cores_measured ? "" : " (not measured: the tier's smallest)", p.ram_bytes / 1073741824.0,
                p.ane ? "on" : "off");
  return buf;
}

}  // namespace

std::string format_profile(const DeviceProfile& p, bool as_json) {
  if (as_json) {
    std::ostringstream o;
    o << "{\"chip_class\":\"" << chip_class_name(p.chip) << "\",\"name\":\"" << p.name << "\",\"cores\":" << p.cores
      << ",\"cores_measured\":" << (p.cores_measured ? "true" : "false") << ",\"ram_bytes\":" << p.ram_bytes
      << ",\"ane\":" << (p.ane ? "true" : "false") << ",\"defaults\":[";
    for (int i = 0; i < default_count(); ++i) {
      const DefaultRow& r = default_at(i);
      if (i) o << ',';
      o << "{\"id\":\"" << r.id << "\",\"model\":\"" << model_class_name(r.model) << "\",\"scope\":\"" << scope_name(r)
        << "\",\"value\":\"" << r.value << "\",\"matches\":" << (default_matches(r, p) ? "true" : "false")
        << ",\"live\":" << (r.live ? "true" : "false") << ",\"measured_on\":\"" << r.measured_on << "\"}";
    }
    o << "]}\n";
    return o.str();
  }
  std::ostringstream o;
  o << "Kapitsa profile\n";
  o << "  chip class: " << chip_class_name(p.chip) << "\n";
  o << "  Mac: " << mac_line(p) << "\n";
  o << "  defaults (call sites still decide; this is the table they move to):\n";
  for (int i = 0; i < default_count(); ++i) {
    const DefaultRow& r = default_at(i);
    o << "    " << r.id << "  " << model_class_name(r.model) << "  " << scope_name(r) << "  "
      << (default_matches(r, p) ? "matches" : "no") << "  " << r.measured_on << "\n";
  }
  return o.str();
}

bool matrix_units(const DeviceFacts& f) { return f.apple_family >= 10; }

bool apple10_gpu(const DeviceFacts& f) { return f.apple_family >= 10; }

bool m1_max_32(const DeviceFacts& f) { return f.name == "Apple M1 Max" && f.cores == 32; }

bool wide_cores(const DeviceFacts& f) { return f.apple_family >= 10 && f.cores >= 40; }

bool named_m5_max(const DeviceFacts& f) { return f.name.find("M5 Max") != std::string::npos; }

bool m4_ten_core(const DeviceFacts& f) { return f.name == "Apple M4" && f.cores <= 10; }

}  // namespace ni
