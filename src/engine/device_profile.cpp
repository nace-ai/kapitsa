#include "engine/device_profile.h"

#include <cstdio>
#include <cstring>
#include <sstream>

namespace ni {
namespace {

// The 35B-A3B's prefill shape: MoE, hidden 2048, 40 layers. Clover-35B-A3B has the same numbers; a row keyed on this
// class decides for both, the way the wave's call site always has (Clover's instance flag keeps the wave off there:
// nothing has been run on it).
bool qwen35_shape(const ModelFacts& m) { return m.moe && m.hidden == 2048 && m.layers == 40; }

// The 35B-A3B shape the KFlash lanes were run on. Clover has the same numbers and is left out by the caller
// (config.json model_type), not by a second chip check: no run on Clover shows a loss or a break, so this is a
// narrower gate than the shape, for #369 to widen.
bool qwen35_a3b(const ModelFacts& m) {
  return qwen35_shape(m) && !m.clover && m.experts == 256 && m.top_k == 8 && m.moe_i == 512 && m.shared_i == 512;
}

}  // namespace

bool model_class_matches(ModelClass c, const ModelFacts& m) {
  switch (c) {
    case ModelClass::kOther: return true;
    case ModelClass::kQwen35Shape: return qwen35_shape(m);
    case ModelClass::kQwen35A3b: return qwen35_a3b(m);
  }
  return false;
}

namespace {

const DefaultRow kDefaults[] = {
    // PR #775 proposes fast membership for dense Qwen 2048-channel inputs through hidden 8192.
    // Non-matrix ANE class; init_ane retains availability/calibration, weight and memory checks.
    // Current-head M4 Pro -62.77% TTFT, calibrated M3 Ultra -23.63%: bench/results/2026-11-07-pr775-m4-default-share
    // and 2026-11-07-pr775-m3-current. Both pass the 576-row bar; the extra M3 prefill check exceeds MLX.
    // This remains a proposed member with that caveat, not an unconditional all-path admission claim.
    {"ane_segmented_fast", ModelClass::kOther, DefaultScope::kAneNonMatrix, nullptr, "fast profile", 24, 63, false,
     "M4 built-in share; non-matrix Macs with calibrated share (M3 Ultra); #775 review 6690600497; matrix excluded", false},
    // Wide class, not a chip name: apple10 and 40 cores or more. M5 Max, 16,288 ids, 2,767 -> 2,620 ms (-5.3%).
    // The call site also wants one slot and a 4-bit group-64 expert pack; those stay at the call site.
    {"pf_wave", Knob::kPfWave, ModelClass::kQwen35Shape, DefaultScope::kWide, nullptr, "on", 5, 0, false, "M5 Max",
     true},
    // Matrix class (any M5): +12 to +15% on the M5 Pro, no measured miss on the others, so the class takes it, not the
    // name. The selector update reads the same row.
    {"kflash_hist", Knob::kKflashHist, ModelClass::kQwen35A3b, DefaultScope::kMatrix, nullptr, "on", 12, 15, false,
     "M5 Pro", true},
    // NI_GEMM_SLICES. Drex Max v1, matrix class. M5 Pro, 16K 33.5-34.0 -> 30.3-30.7 s (-9.6%).
    // The M5 Max was not measured. The shape check that keeps it to Drex Max v1 stays at the call site.
    {"gemm_slices", Knob::kGemmSlices, ModelClass::kOther, DefaultScope::kMatrix, nullptr, "on", 9, 0, false,
     "M5 Pro; widened by judgment, sweep 1", true},
    // NI_HC_WIDE. Flash-Next, wide class. M5 Max, decode 105.9 -> 108.1 tok/s (+2.0%). Under 10%, so not a name gate.
    {"hc_wide", Knob::kHcWide, ModelClass::kOther, DefaultScope::kWide, nullptr, "on", 2, 0, false,
     "M5 Max; widened by judgment, sweep 1", true},
    // NI_MOE4_DX. Flash-Next, wide class. M5 Max, decode 108.2 -> 109.3 tok/s (+1.0%). Under 10%, so not a name gate.
    {"moe4_dx", Knob::kMoe4Dx, ModelClass::kOther, DefaultScope::kWide, nullptr, "on", 1, 0, false,
     "M5 Max; widened by judgment, sweep 1", true},
    // NI_V8_TREE_G6. Qwen3.8 27B, wide class. M5 Max, ni_run, decode 48.6 -> 62.2 tok/s (27, the integer part).
    // The wide class today is that Mac.
    {"tree_g6", Knob::kTreeG6, ModelClass::kOther, DefaultScope::kWide, nullptr, "on", 27, 0, false, "M5 Max", true},
    // NI_KFLASH_NARROW. Qwen3.8 27B, wide class. M5 Max, 131K 56.1 -> 70.1 tok/s (24, the integer part);
    // 32K 80.5 -> 85.6. The 16-row drafting block it applies to stays at the call site.
    {"kflash_narrow", Knob::kKflashNarrow, ModelClass::kOther, DefaultScope::kWide, nullptr, "on", 24, 0, false,
     "M5 Max", true},
    // Capability classes the call sites still read. Not a name gate: the predicate is the class. The site count is
    // calls outside this module; tests/test_profile_gates.py checks the where string against that count.
    {"matrix_units", Knob::kNone, ModelClass::kOther, DefaultScope::kMatrix, nullptr, "class", 0, 0, false,
     "46 sites; capability class: apple_family >= 10, the matrix half of ChipClass", false},
    {"legacy_gpu", Knob::kNone, ModelClass::kOther, DefaultScope::kLegacy, nullptr, "class", 0, 0, false,
     "11 sites; capability class: apple_family <= 7, the M1 path", false},
    {"wide_cores", Knob::kNone, ModelClass::kOther, DefaultScope::kWide, nullptr, "class", 0, 0, false,
     "2 sites; capability class: matrix units and 40 cores or more", false},
    // Name and correctness predicates the call sites still read. Not kName: none of these cites a 10% win the wider
    // class misses. The id is the predicate. chip_name is the exact name when the predicate has one.
    // Knob::kNone: not a lookup. A shape guard stays because the wider class is the wrong shape. A correctness
    // guard stays because the fused path failed on that class.
    {"m1_max_32", Knob::kNone, ModelClass::kOther, DefaultScope::kGuard, "Apple M1 Max", "64", 0, 0, false,
     "4 sites; shape guard: 32 cores; 35B group of 64 (hidden 2048, intermediate 512, else 32); also the 35B 4096-row prefill chunk, the 35B q4-head v8 default, and the drafter CORE4 path", false},
    {"m4_10", Knob::kNone, ModelClass::kOther, DefaultScope::kGuard, "Apple M4", "on", 0, 0, false,
     "1 site; shape guard: 10 cores and 24 GB; 4B packed-coeff shape only", false},
    {"m4_pro_gpu", Knob::kNone, ModelClass::kOther, DefaultScope::kGuard, "Apple M4 Pro", "pack", 0, 0, false,
     "1 site; shape guard: Qwen 4B and 35B vision-tower shapes; other shapes keep bf16", false},
    {"m4_pro_20", Knob::kNone, ModelClass::kOther, DefaultScope::kGuard, "Apple M4 Pro", "on", 0, 0, false,
     "2 sites; shape guard: 20 cores and 48 GB for the 4B or 9B packed coeff; 35B X8 share on 5-6-bit affine projections", false},
    {"m4_ten_core", Knob::kNone, ModelClass::kOther, DefaultScope::kGuard, "Apple M4", "narrow", 0, 0, false,
     "6 sites; shape guard: 10 cores or fewer; low-rank 768, GDN blocks of 128, attention splits of at least 128 keys, tree-zone fuse, lazy GDN fold, depth-first tree rows", false},
    {"ultra_gpu", Knob::kNone, ModelClass::kOther, DefaultScope::kGuard, nullptr, "split", 0, 0, true,
     "1 site; correctness guard: two dies; fused merges failed past a 32-key tile on the old arrival (#421); the new arrival's fused merges are unverified on an Ultra", false},
    {"pre_m3_gpu", Knob::kNone, ModelClass::kOther, DefaultScope::kGuard, nullptr, "split", 0, 0, true,
     "1 site; correctness guard: apple_family under 9; fused merges produced garbage on an M1 Max on 0.2.15, old arrival (#478); the new arrival is unverified on M1 and M2", false},
    // Fast profile only (docs/profiles.md): Flash-Next's sparse prefill attention on the matrix units, Q and P as hi + lo
    // halves (#818). Matrix class, not the name: the kernel needs matrix units and nothing was measured slower on them.
    // M5 Pro, 32K and 64K prompts at 2048-row chunks: prefill +8 to +10%, decode the same per draft cycle; 4 top-1
    // flips in 72 rows (Splash 1.5.0: 16). The call site is
    // tattn() in qwen4_exp.cpp, whose matrix-unit check predates this table. Knob::kNone: not a default_on lookup.
    {"tattn_fast", Knob::kNone, ModelClass::kOther, DefaultScope::kMatrix, nullptr, "fast profile", 8, 10, false,
     "M5 Pro", false},
};

bool guard_matches(const char* id, const DeviceProfile& dev) {
  if (!id) return false;
  DeviceFacts f;
  f.apple_family = dev.apple_family;
  f.name = dev.name;
  f.cores = dev.cores;
  f.cores_measured = dev.cores_measured;
  if (std::strcmp(id, "m1_max_32") == 0) return m1_max_32(f);
  if (std::strcmp(id, "m4_10") == 0) return m4_10(f);
  if (std::strcmp(id, "m4_pro_gpu") == 0) return m4_pro_gpu(f);
  if (std::strcmp(id, "m4_pro_20") == 0) return m4_pro_20(f);
  if (std::strcmp(id, "m4_ten_core") == 0) return m4_ten_core(f);
  if (std::strcmp(id, "ultra_gpu") == 0) return ultra_gpu(f);
  if (std::strcmp(id, "pre_m3_gpu") == 0) return pre_m3_gpu(f);
  return false;
}

}  // namespace

DeviceProfile resolve_device(const DeviceFacts& f) {
  DeviceProfile p;
  p.apple_family = f.apple_family;
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

ModelClass resolve_model(const ModelFacts& m) {
  if (qwen35_a3b(m)) return ModelClass::kQwen35A3b;
  if (qwen35_shape(m)) return ModelClass::kQwen35Shape;
  return ModelClass::kOther;
}

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
  switch (m) {
    case ModelClass::kQwen35A3b: return "qwen35-a3b";
    case ModelClass::kQwen35Shape: return "qwen35-shape";
    case ModelClass::kOther: return "other";
  }
  return "other";
}

bool default_on(Knob k, const DeviceProfile& d, const ModelFacts& m) {
  for (int i = 0; i < default_count(); ++i) {
    const DefaultRow& r = default_at(i);
    if (r.knob == k)
      return std::string(r.value) == "on" && default_matches(r, d) && model_class_matches(r.model, m);
  }
  return false;
}

int default_count() { return int(sizeof(kDefaults) / sizeof(kDefaults[0])); }

const DefaultRow& default_at(int i) { return kDefaults[i]; }

bool default_matches(const DefaultRow& row, const DeviceProfile& dev) {
  switch (row.scope) {
    case DefaultScope::kAneNonMatrix:
      return dev.ane && dev.chip != ChipClass::kMatrix && dev.chip != ChipClass::kWide;
    case DefaultScope::kLegacy: return dev.chip == ChipClass::kLegacy;
    case DefaultScope::kSimd: return dev.chip == ChipClass::kSimd;
    case DefaultScope::kMatrix: return dev.chip == ChipClass::kMatrix || dev.chip == ChipClass::kWide;
    case DefaultScope::kWide: return dev.chip == ChipClass::kWide;
    case DefaultScope::kName: return row.chip_name && dev.name == row.chip_name;
    case DefaultScope::kGuard: return guard_matches(row.id, dev);
  }
  return false;
}

namespace {

const char* scope_name(const DefaultRow& row) {
  if (row.scope == DefaultScope::kName) return row.chip_name ? row.chip_name : "name";
  switch (row.scope) {
    case DefaultScope::kAneNonMatrix: return "ane-nonmatrix";
    case DefaultScope::kLegacy: return "legacy";
    case DefaultScope::kSimd: return "simd";
    case DefaultScope::kMatrix: return "matrix";
    case DefaultScope::kWide: return "wide";
    case DefaultScope::kName: return "name";
    case DefaultScope::kGuard: return "guard";
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
  o << "  defaults (the call sites read this table; env overrides apply at the site):\n";
  for (int i = 0; i < default_count(); ++i) {
    const DefaultRow& r = default_at(i);
    o << "    " << r.id << "  " << model_class_name(r.model) << "  " << scope_name(r) << "  "
      << (default_matches(r, p) ? "matches" : "no") << "  " << (r.where ? r.where : "") << "  win " << r.win_pct;
    if (r.win_pct_hi) o << '-' << r.win_pct_hi;
    o << '%';
    if (r.scope == DefaultScope::kName || r.scope == DefaultScope::kGuard)
      o << (r.exclusive ? "  exclusive" : "  not exclusive");
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
