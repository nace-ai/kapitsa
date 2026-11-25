#include "engine/device_profile.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace ni {
namespace {

// The 35B-A3B's prefill shape: MoE, hidden 2048, 40 layers. Clover-35B-A3B has the same numbers; a row keyed on this
// class decides for both, the way the wave's call site always has (Clover's instance flag keeps the wave off there:
// nothing has been run on it).
bool qwen35_shape(const ModelFacts& m) { return m.moe && m.hidden == 2048 && m.layers == 40; }

// The 35B-A3B shape the KFlash lanes were run on. Clover has the same numbers and is left out by the caller
// (config.json model_type): with the history lane and the selector update, Clover's sampled decode is slower
// (M5 Pro, interleaved, 6 of 6 pairs slower at temperature 0.6 (-4%), faster at 0 (+5%; #1094)), so this gate is narrower than the shape on a measured miss.
bool qwen35_a3b(const ModelFacts& m) {
  return qwen35_shape(m) && !m.clover && m.experts == 256 && m.top_k == 8 && m.moe_i == 512 && m.shared_i == 512;
}

// The Qwen3.8 27B's shape (ModelClass::kQwen27bDense)
bool qwen27b_dense(const ModelFacts& m) { return !m.moe && m.hidden == 5120 && m.layers == 64; }

}  // namespace

// The generation N of an "Apple M<N>" chip name ("Apple M6 Pro": 6), or 0 for any other name.
int ane_generation(const std::string& chip) {
  const std::string prefix = "Apple M";
  if (chip.compare(0, prefix.size(), prefix) != 0) return 0;
  size_t i = prefix.size();
  int gen = 0;
  while (i < chip.size() && chip[i] >= '0' && chip[i] <= '9' && gen < 1000) gen = gen * 10 + (chip[i++] - '0');
  if (i == prefix.size() || (i < chip.size() && chip[i] != ' ')) return 0;
  return gen;
}

// The M4 and every later generation ("Apple M4", "Apple M5 Pro", "Apple M6", "Apple M10 Max"). A substring test on
// "M4"/"M5" left the M6 and later with no default share, so their Neural Engine prefill split stayed off without a
// calibration, while the GPU half (matrix units, apple_family >= 10) already treats them as M5-class.
bool ane_builtin_share(const std::string& chip) { return ane_generation(chip) >= 4; }

// The plain split's default share (no wavefront: Drex Max's noncausal pass, or NI_ANE_WAVE=0) was output-checked on
// the M4 Pro and the M5 Pro only; the M6 and later keep that pass on the GPU until it is measured there (review of
// nace-ai/kapitsa#1259), as M1 and M2 do. Their wavefront share (every causal prefill) is unaffected.
bool ane_plain_share(const std::string& chip) {
  const int gen = ane_generation(chip);
  return gen >= 4 && gen <= 5;
}

int ane_pre_m3_share(const std::string& chip) {
  for (int family : {1, 2}) {
    const std::string name = "Apple M" + std::to_string(family);
    if (chip.compare(0, name.size(), name) == 0 && (chip.size() == name.size() || chip[name.size()] == ' ')) return family;
  }
  return 0;
}

bool ane_segmented_default(const DeviceFacts& facts) {
  return ane_builtin_share(facts.name) && !matrix_units(facts);
}

bool model_class_matches(ModelClass c, const ModelFacts& m) {
  switch (c) {
    case ModelClass::kOther: return true;
    case ModelClass::kQwen35Shape: return qwen35_shape(m);
    case ModelClass::kQwen35A3b: return qwen35_a3b(m);
    case ModelClass::kQwen27bDense: return qwen27b_dense(m);
    case ModelClass::kInkling2: return m.inkling2;
  }
  return false;
}

namespace {

const DefaultRow kDefaults[] = {
    // Inkling-2 architecture, all Macs. Automatic fast selection keeps the member eligibility below unchanged:
    // tensor prefill needs matrix units; Q8 dense packing runs on every Apple Silicon GPU. Evidence and the
    // quantization-relative quality gate: bench/results/inkling-fast-default-20261124/README.md.
    {"ink2_profile_fast", Knob::kProfileFast, ModelClass::kInkling2, DefaultScope::kAll, nullptr, "on", 0, 0, false,
     "Inkling-2; automatic fast profile (explicit precise wins)", true},
    // Mixed 4/5-bit 9B uses the matrix path. The SIMD packed-coefficient path accepts only 4-bit weights.
    // #493 keeps the legacy pack on non-matrix Macs; this is a format policy, not an M3/M4 speed measurement.
    {"qwen9b_q4", Knob::kQwen9bQ4, ModelClass::kOther, DefaultScope::kMatrix, nullptr, "on", 0, 0, false,
     "#493: matrix mixed-format path; non-matrix keeps legacy 4-bit", true},
    // Dense Qwen 2048-channel inputs through hidden 8192: built-in ANE share, non-matrix GPUs (M4 today).
    // M4 Pro -61.94% loaded TTFT and all-path bar pass: bench/results/2026-11-07-pr775-rebase-3f3b.
    // Calibrated M3 prefill exceeds MLX (mean KL 1.846e-4 vs 1.472e-4); calibration-only chips stay off.
    // Matrix GPUs stay off per #775 review 6690600497; built-in-share restriction: review 6698446394.
    {"ane_segmented_fast", Knob::kNone, ModelClass::kOther, DefaultScope::kAneBuiltinNonMatrix, nullptr, "fast profile", 61, 0, false,
     "M4 built-in share; M3 prefill bar failure excludes calibration-only chips; matrix GPUs excluded", false},
    // Matrix class (any M5). M5 Max, 16,288 ids, 2,767 -> 2,620 ms (-5.3%); M5 Pro (#393), 16,288 ids in 8,192-row
    // chunks against 4,096: the 35B-A3B 5,874 -> 5,618 ms (-4.4%), Clover 5,699 -> 5,526 (-3.0%), the same text.
    // The call site also wants one slot and a 4-bit group-64 expert pack; those stay at the call site.
    {"pf_wave", Knob::kPfWave, ModelClass::kQwen35Shape, DefaultScope::kMatrix, nullptr, "on", 3, 5, false,
     "M5 Max, M5 Pro", true},
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
    // NI_KFLASH_BLOCK's default on a server of more than one slot: the 16-row drafting block, as one slot has it, where
    // the drafting pass runs on the matrix units. Requests decoding together then take turns instead of sharing an
    // 8-row batched pass (Engine::batch_rows). Mac mini M6 (12 cores), 27B KQuant, serve's default 2 slots, the opencode agent replay:
    // writing 36.2 -> 44.3 tok/s (+22%), the session 159.8 -> 147.7 s (-7.6%); two sessions at once 294.3 -> 279.0 s
    // (-5.2%); the 2 slots' drafter rings 0.25 -> 0.46 GB. M5 Pro (#1262's review lane), 27B 4-bit, serve's default
    // 4 slots: writing 68.7 -> 79.8 tok/s (+16%), the session 88.6 -> 82.0 s (-7.4%). One slot already drafts 16 rows;
    // the call site keeps its checks (DFlash2::plan_block_rows).
    {"kflash_block16", Knob::kKflashBlock16, ModelClass::kOther, DefaultScope::kMatrix, nullptr, "on", 22, 0, false,
     "Mac mini M6, M5 Pro", true},
    // Capability classes the call sites still read. Not a name gate: the predicate is the class. The site count is
    // calls outside this module; tests/test_profile_gates.py checks the where string against that count.
    {"matrix_units", Knob::kNone, ModelClass::kOther, DefaultScope::kMatrix, nullptr, "class", 0, 0, false,
     "46 sites; capability class: apple_family >= 10, the matrix half of ChipClass", false},
    {"legacy_gpu", Knob::kNone, ModelClass::kOther, DefaultScope::kLegacy, nullptr, "class", 0, 0, false,
     "11 sites; capability class: apple_family <= 7, the M1 path", false},
    {"wide_cores", Knob::kNone, ModelClass::kOther, DefaultScope::kWide, nullptr, "class", 0, 0, false,
     "1 site; capability class: matrix units and 40 cores or more", false},
    // Capability class: apple_family == 9 (M3, M4; not M2 which is family 8). Fold of m4_10, m4_pro_gpu, m4_pro_20,
    // m4_ten_core (#393): low-rank 768, GDN 128, tree-fuse, lazy GDN, DFS, packed coeff, VIT prepack, NORM X8,
    // SHARE X8 widened from narrow M4 gates. m1_max_32 kept as its own guard row.
    {"simd_gpu", Knob::kNone, ModelClass::kOther, DefaultScope::kSimd, nullptr, "class", 0, 0, false,
     "9 sites; capability class: apple_family == 9 (M3, M4); fold of m4_10, m4_pro_gpu, m4_pro_20, m4_ten_core (#393)", false},
    {"m1_max_32", Knob::kNone, ModelClass::kOther, DefaultScope::kGuard, "Apple M1 Max", "64", 0, 0, false,
     "4 sites; shape guard: 32 cores; 35B group of 64 (hidden 2048, intermediate 512, else 32); also the 35B 4096-row prefill chunk, the 35B q4-head v8 default, and the drafter CORE4 path (measured on M5 and the 32-core M1 Max)", false},
    {"ultra_gpu", Knob::kNone, ModelClass::kOther, DefaultScope::kGuard, nullptr, "split", 0, 0, true,
     "1 site; correctness guard: two dies; Flint's fused merges failed past a 32-key tile on the old arrival (#421), the new arrival's are unverified on an Ultra; Nemotron-H's fused merges answered garbage on Super-VL's plain decoding and doubled tokens under LiLiCorr on an M5 Ultra (macOS 27.2.1), NI_NH_SPLIT=1 fixed both (kapitsa#1301; nh_merge_path)", false},
    {"pre_m3_gpu", Knob::kNone, ModelClass::kOther, DefaultScope::kGuard, nullptr, "split", 0, 0, true,
     "1 site; correctness guard: apple_family under 9; fused merges produced garbage on an M1 Max on 0.2.15, old arrival (#478); the new arrival is unverified on M1 and M2", false},
    // Fast profile only (docs/profiles.md): Flash-Next's sparse prefill attention on the matrix units, Q and P as hi + lo
    // halves (#818). Matrix class, not the name: the kernel needs matrix units and nothing was measured slower on them.
    // M5 Pro, 32K and 64K prompts at 2048-row chunks: prefill +8 to +10%, decode the same per draft cycle; 4 top-1
    // flips in 72 rows (Splash 1.5.0: 16). The call site is
    // tattn() in qwen4_exp.cpp, whose matrix-unit check predates this table. Knob::kNone: not a default_on lookup.
    {"tattn_fast", Knob::kNone, ModelClass::kOther, DefaultScope::kMatrix, nullptr, "fast profile", 8, 10, false,
     "M5 Pro", false},
    // Fast profile only (docs/profiles.md): the token mixers' input projections on the Neural Engine beside the FFN
    // split's wavefront (NI_ANE_PROJ, Qwen35::proj_on). A narrow row on a measured miss: the gain needs a GPU slower than
    // the Neural Engine at the projections, so it is the chip and model measured, not the matrix class. Mac mini M6
    // (12 cores), the 27B: 4K and 16K prefill +20%, 1024-row chunks +3.5% (ni_run, 3 interleaved pairs); 192 step rows
    // 0 flips, mean KL 2.2e-4 against precise, inside its bar (mlx-lm: 0 flips, 2.9e-4). M5 Pro (48 GB, review of
    // #1275): 4K 750 -> 563 tok/s (-25%), 16K 688 -> 542 (-21%), 3 rounds: slower on the wider class. The 4B on the
    // M6: faster (+6 to +9%) but over its bar (4 flips, 7.7e-4 against mlx-lm's 2, 7.1e-4). Not measured: the M5 Max,
    // the M5, the M6 Pro and Max (their GPUs are faster than the M6's), and the Macs without matrix units.
    {"ane_proj_fast", Knob::kAneProjFast, ModelClass::kQwen27bDense, DefaultScope::kName, "Apple M6", "on", 20, 0, true,
     "Mac mini M6 (27B); slower on the M5 Pro (-21 to -25%); fast profile only", true},
    // Fast profile only (docs/profiles.md): Hy4-flash's routed experts' prefill GEMMs on the matrix units
    // (src/kernels/tensor/qmt_moe_a4.metal, NI_A4_QTM; a4::Store, Hy4's alone, is the model half). The matrix class,
    // as tattn_fast: the tile needs matrix units, and the measured gain is the expert GEMMs' simdgroup-to-tensor-unit
    // step every M5 takes. M5 Max, kapitsa bench: prefill 4K 596 -> 930 tok/s (+56%), 16K 625 -> 989 (+58%), decode
    // unchanged; 1,024 rows against precise: 86 top-1 flips, mean KL 0.137, inside its bar (MLX on the same pack).
    {"a4_qtm_fast", Knob::kA4QtmFast, ModelClass::kOther, DefaultScope::kMatrix, nullptr, "on", 56, 58, false,
     "M5 Max; fast profile only", true},
    // Fast profile only (docs/profiles.md): Inkling-2-Mini prefill GEMMs (BF16 dense + grouped NVFP4/MXFP8 MoE) on
    // Apple10+ matrix units (src/kernels/tensor/ink2_t.metal, NI_INK2_TENSOR). Mac mini M6, chunk 512, interleaved:
    // 512 tok 70.4 -> 493.7/495.3 tok/s (+603%), 2048 tok 67.4 -> 382.6 tok/s (+468%); 8-window logit_dump against
    // precise: step passes --reorder (2 near-tie flips / 192, mean KL 1.03e-3), prefill 0 flips / 8 (mean KL 3.60e-3).
    {"ink2_tensor_pf", Knob::kInk2TensorPf, ModelClass::kOther, DefaultScope::kMatrix, nullptr, "on", 468, 603, false,
     "Mac mini M6; fast profile only", true},
    // Fast profile only (docs/profiles.md): Inkling-2-Mini load-time Q8_0 packing for BF16 dense projections and
    // unembed (NI_INK2_Q8). All Apple Silicon (uses scalar/SIMD + simdgroup_half8x8 kernels): M6 32 GB decode
    // 20.25 -> 26.17 tok/s (+29.2%), resident text weights 18.89 -> 17.85 GB (-1.04 GB); M5 Pro 48 GB decode
    // +20-22%, prefill +8-16%; 192 step rows 1 flip, mean KL 2.0e-3 (within the fast fallback bar).
    {"ink2_q8_fast", Knob::kInk2Q8Fast, ModelClass::kOther, DefaultScope::kAll, nullptr, "on", 20, 29, false,
     "Mac mini M6 (+29.2%, -1.04 GB), M5 Pro (+20-22%); fast profile only", true},
    // Fast profile only: Inkling-2-Mini's M1 stack (inkling2.cpp ink2_knob; one switch each, its env var wins): tensor
    // prefill with the M1 cooperative-tensor layout and the simdgroup FlashAttention, MessageGate GEMMs, Q8 matvecs for
    // 1-8 rows beside #1381's packing with BF16 kept for prefill, repacked NVFP4 MoE decode, folded norms / short
    // convs / rel bias / softcap, last-threadgroup attention and router reductions, ring-modulo-free and tiled-K decode
    // attention, verify rows through the decode attention, prompt lookup with a table policy. Mechanism (M1 Max
    // profiles, lab rounds 2-17): decode on Apple7 is issue- and latency-bound (MoE decode at 220-307 GB/s of ~355,
    // decode attention not DRAM-bound, a removed stage worth its ~2.4 us barrier), so the changes cut ALU work, stages
    // and dependent hops rather than bytes; the chunk sizes (KW_SWACH 128, KW_MAXCH 48) are the 32-core M1 Max's wave
    // quantization. M1 Max 64 GB, served 1 slot: decode 46.0 / 40.3 / 34.3 tok/s at 8k / 40k / 84k, prefill ~430 / 380
    // / 330 (lab13 base; round 1 on the #1379 + #1387 tree: 25.4 / 22.8 / 19.6, 292 / 256 / 218). Not measured on
    // other classes: they keep the shipped path.
    {"ink2_m1_stack", Knob::kInk2M1Stack, ModelClass::kInkling2, DefaultScope::kLegacy, nullptr, "on", 75, 81, false,
     "M1 Max 64 GB (32 cores); fast profile only", true},
};

bool guard_matches(const char* id, const DeviceProfile& dev) {
  if (!id) return false;
  DeviceFacts f;
  f.apple_family = dev.apple_family;
  f.name = dev.name;
  f.cores = dev.cores;
  f.cores_measured = dev.cores_measured;
  if (std::strcmp(id, "m1_max_32") == 0) return m1_max_32(f);
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
  if (m.inkling2) return ModelClass::kInkling2;
  if (qwen35_a3b(m)) return ModelClass::kQwen35A3b;
  if (qwen35_shape(m)) return ModelClass::kQwen35Shape;
  if (qwen27b_dense(m)) return ModelClass::kQwen27bDense;
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
    case ModelClass::kQwen27bDense: return "qwen-27b-dense";
    case ModelClass::kInkling2: return "inkling2";
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

Profile default_profile(const DeviceProfile& d, const ModelFacts& m) {
  return default_on(Knob::kProfileFast, d, m) ? Profile::kFast : Profile::kPrecise;
}

int default_count() { return int(sizeof(kDefaults) / sizeof(kDefaults[0])); }

const DefaultRow& default_at(int i) { return kDefaults[i]; }

bool default_matches(const DefaultRow& row, const DeviceProfile& dev) {
  switch (row.scope) {
    case DefaultScope::kAll: return true;
    case DefaultScope::kAneBuiltinNonMatrix:
      return dev.ane && ane_builtin_share(dev.name) && dev.chip != ChipClass::kMatrix && dev.chip != ChipClass::kWide;
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
    case DefaultScope::kAll: return "all";
    case DefaultScope::kAneBuiltinNonMatrix: return "ane-builtin-nonmatrix";
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

bool wide_cores(const DeviceFacts& f) { return f.apple_family >= 10 && f.cores >= 40; }

bool m1_max_32(const DeviceFacts& f) { return f.name == "Apple M1 Max" && f.cores == 32; }

bool ultra_gpu(const DeviceFacts& f) { return f.name.find("Ultra") != std::string::npos; }

bool pre_m3_gpu(const DeviceFacts& f) { return f.apple_family < 9; }

bool simd_gpu(const DeviceFacts& f) { return f.apple_family == 9; }  // M3, M4; not M2 (family 8)

NhMergePath nh_merge_path(const DeviceFacts& f, bool fences, const char* env) {
  if (!fences) return {true, "macOS 14: no device fences in the library"};
  if (env) return atoi(env) != 0 ? NhMergePath{true, "NI_NH_SPLIT=1"} : NhMergePath{false, "NI_NH_SPLIT=0"};
  if (ultra_gpu(f)) return {true, "an Ultra: two dies, kapitsa#1301"};
  return {false, "NI_NH_SPLIT=1 splits them"};
}

}  // namespace ni
