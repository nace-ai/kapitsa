#include "engine/device_profile.h"

#include <cstdio>
#include <sstream>

namespace ni {
namespace {

// The 35B-A3B shape the wave and the KFlash lanes were run on. Clover has the same numbers and is left out by the
// caller (config.json model_type), not by a second chip check: no run on Clover shows a loss or a break, so this is a
// narrower gate than the shape, for #369 to widen.
bool qwen35_a3b(const ModelFacts& m) {
  return m.moe && !m.clover && m.hidden == 2048 && m.layers == 40 && m.experts == 256 && m.top_k == 8 &&
         m.moe_i == 512 && m.shared_i == 512;
}

const DefaultRow kDefaults[] = {
    // Wide class, not a chip name: apple10 and 40 cores or more. M5 Max, 16,288 ids, 2,767 -> 2,620 ms (-5.3%).
    // The call site also wants one slot and a 4-bit group-64 expert pack; those stay at the call site when it moves.
    {"pf_wave", ModelClass::kQwen35A3b, DefaultScope::kWide, nullptr, "on", 5, 0, false, "M5 Max", false},
    // Matrix class (any M5): +12 to +15% on the M5 Pro, no measured miss on the others, so the class takes it, not the
    // name. The selector update uses the same gate.
    {"kflash_hist", ModelClass::kQwen35A3b, DefaultScope::kMatrix, nullptr, "on", 12, 15, false, "M5 Pro", false},
    // NI_GEMM_SLICES. Drex Max v1, matrix class. M5 Pro, 16K 33.5-34.0 -> 30.3-30.7 s (-9.6%).
    // The M5 Max was not measured.
    {"gemm_slices", ModelClass::kOther, DefaultScope::kMatrix, nullptr, "on", 9, 0, false,
     "M5 Pro; widened by judgment, sweep 1", false},
    // NI_HC_WIDE. Flash-Next, wide class. M5 Max, decode 105.9 -> 108.1 tok/s (+2.0%). Under 10%, so not a name gate.
    {"hc_wide", ModelClass::kOther, DefaultScope::kWide, nullptr, "on", 2, 0, false,
     "M5 Max; widened by judgment, sweep 1", false},
    // NI_MOE4_DX. Flash-Next, wide class. M5 Max, decode 108.2 -> 109.3 tok/s (+1.0%). Under 10%, so not a name gate.
    {"moe4_dx", ModelClass::kOther, DefaultScope::kWide, nullptr, "on", 1, 0, false,
     "M5 Max; widened by judgment, sweep 1", false},
    // NI_V8_TREE_G6. Qwen3.8 27B, wide class. M5 Max, ni_run, decode 48.6 -> 62.2 tok/s (27, the integer part).
    // The wide class today is that Mac.
    {"tree_g6", ModelClass::kOther, DefaultScope::kWide, nullptr, "on", 27, 0, false, "M5 Max", false},
    // NI_KFLASH_NARROW. Qwen3.8 27B, wide class. M5 Max, 131K 56.1 -> 70.1 tok/s (24, the integer part);
    // 32K 80.5 -> 85.6.
    {"kflash_narrow", ModelClass::kOther, DefaultScope::kWide, nullptr, "on", 24, 0, false, "M5 Max", false},
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
    case DefaultScope::kMatrix: return dev.chip == ChipClass::kMatrix || dev.chip == ChipClass::kWide;
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

void append_json_string(std::ostream& o, const char* s) {
  o << '"';
  if (s) {
    for (const auto* p = reinterpret_cast<const unsigned char*>(s); *p; ++p) {
      if (*p == '"' || *p == '\\') o << '\\' << char(*p);
      else if (*p < 0x20) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "\\u%04x", *p);
        o << buf;
      } else {
        o << char(*p);
      }
    }
  }
  o << '"';
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
    o << "{\"chip_class\":";
    append_json_string(o, chip_class_name(p.chip));
    o << ",\"name\":";
    append_json_string(o, p.name.c_str());
    o << ",\"cores\":" << p.cores << ",\"cores_measured\":" << (p.cores_measured ? "true" : "false")
      << ",\"ram_bytes\":" << p.ram_bytes << ",\"ane\":" << (p.ane ? "true" : "false") << ",\"defaults\":[";
    for (int i = 0; i < default_count(); ++i) {
      const DefaultRow& r = default_at(i);
      if (i) o << ',';
      o << "{\"id\":";
      append_json_string(o, r.id);
      o << ",\"model\":";
      append_json_string(o, model_class_name(r.model));
      o << ",\"scope\":";
      append_json_string(o, scope_name(r));
      o << ",\"value\":";
      append_json_string(o, r.value);
      o << ",\"matches\":" << (default_matches(r, p) ? "true" : "false") << ",\"live\":" << (r.live ? "true" : "false")
        << ",\"where\":";
      append_json_string(o, r.where);
      o << ",\"win_pct\":" << r.win_pct << ",\"win_pct_hi\":" << r.win_pct_hi
        << ",\"exclusive\":" << (r.exclusive ? "true" : "false") << '}';
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
      << (default_matches(r, p) ? "matches" : "no") << "  " << (r.where ? r.where : "") << "  win " << r.win_pct;
    if (r.win_pct_hi) o << '-' << r.win_pct_hi;
    o << '%';
    if (r.scope == DefaultScope::kName) o << (r.exclusive ? "  exclusive" : "  not exclusive");
    o << "  live " << (r.live ? "yes" : "no") << "\n";
  }
  return o.str();
}

bool matrix_units(const DeviceFacts& f) { return f.apple_family >= 10; }

bool legacy_gpu(const DeviceFacts& f) { return f.apple_family <= 7; }

bool m1_max_32(const DeviceFacts& f) { return f.name == "Apple M1 Max" && f.cores == 32; }

bool wide_cores(const DeviceFacts& f) { return f.apple_family >= 10 && f.cores >= 40; }

bool m4_10(const DeviceFacts& f) { return f.name == "Apple M4" && f.cores == 10; }

bool m4_pro_gpu(const DeviceFacts& f) { return f.name == "Apple M4 Pro"; }

bool m4_pro_20(const DeviceFacts& f) { return f.name == "Apple M4 Pro" && f.cores == 20; }

bool ultra_gpu(const DeviceFacts& f) { return f.name.find("Ultra") != std::string::npos; }

bool pre_m3_gpu(const DeviceFacts& f) { return f.apple_family < 9; }

bool m4_ten_core(const DeviceFacts& f) { return f.name == "Apple M4" && f.cores <= 10; }

}  // namespace ni
