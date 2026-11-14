#include "ane/default_share.h"
#include "engine/device_profile.h"
#include "metal/gpu.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK(condition)                                                                                               \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition);                             \
      std::abort();                                                                                                    \
    }                                                                                                                  \
  } while (false)

static ni::DeviceFacts mac(int family, int cores, const char* name, long ram) {
  ni::DeviceFacts f;
  f.apple_family = family;
  f.cores = cores;
  f.cores_measured = true;
  f.name = name;
  f.ram_bytes = ram;
  f.ane = true;
  return f;
}

static const ni::DefaultRow* row(const char* id) {
  for (int i = 0; i < ni::default_count(); ++i)
    if (std::string(ni::default_at(i).id) == id) return &ni::default_at(i);
  return nullptr;
}

int main() {
  using ni::ChipClass;
  const long g48 = 48L << 30;
  CHECK(ni::resolve_device(mac(7, 32, "Apple M1 Max", 64L << 30)).chip == ChipClass::kLegacy);
  CHECK(ni::resolve_device(mac(8, 10, "Apple M2", 24L << 30)).chip == ChipClass::kSimd);
  CHECK(ni::resolve_device(mac(9, 20, "Apple M4 Pro", g48)).chip == ChipClass::kSimd);
  CHECK(ni::resolve_device(mac(9, 40, "Apple M4 Max", 64L << 30)).chip == ChipClass::kSimd);
  CHECK(ni::resolve_device(mac(10, 20, "Apple M5 Pro", g48)).chip == ChipClass::kMatrix);
  CHECK(ni::resolve_device(mac(10, 40, "Apple M5 Max", 128L << 30)).chip == ChipClass::kWide);
  // A family past 10 is still matrix or wide. apple_family() must not clamp it to 10.
  CHECK(ni::resolve_device(mac(11, 20, "Apple M6 Pro", g48)).chip == ChipClass::kMatrix);
  CHECK(ni::resolve_device(mac(11, 40, "Apple M6 Max", 128L << 30)).chip == ChipClass::kWide);
  // An unread core count falls back to 24 for a Max. That is matrix, not wide, which is what the wave gate does today.
  ni::DeviceFacts unread = mac(10, 24, "Apple M5 Max", 128L << 30);
  unread.cores_measured = false;
  CHECK(ni::resolve_device(unread).chip == ChipClass::kMatrix);

  ni::ModelFacts a3b;
  a3b.moe = true;
  a3b.hidden = 2048;
  a3b.layers = 40;
  a3b.experts = 256;
  a3b.top_k = 8;
  a3b.moe_i = 512;
  a3b.shared_i = 512;
  CHECK(ni::resolve_model(a3b) == ni::ModelClass::kQwen35A3b);
  a3b.clover = true;
  CHECK(ni::resolve_model(a3b) == ni::ModelClass::kQwen35Shape);  // Clover has the wave's shape, not the drafter's
  ni::ModelFacts shape_only;
  shape_only.moe = true;
  shape_only.hidden = 2048;
  shape_only.layers = 40;
  CHECK(ni::resolve_model(shape_only) == ni::ModelClass::kQwen35Shape);

  const ni::DefaultRow* wave = row("pf_wave");
  const ni::DefaultRow* hist = row("kflash_hist");
  CHECK(wave && hist);
  CHECK(wave->live && hist->live);
  CHECK(wave->win_pct == 3 && wave->win_pct_hi == 5 && !wave->exclusive);
  CHECK(hist->win_pct == 12 && hist->win_pct_hi == 15 && !hist->exclusive);
  const auto m5pro = ni::resolve_device(mac(10, 20, "Apple M5 Pro", g48));
  const auto m5max = ni::resolve_device(mac(10, 40, "Apple M5 Max", 128L << 30));
  CHECK(ni::default_matches(*wave, m5pro));  // matrix class since #393
  CHECK(ni::default_matches(*wave, m5max));
  CHECK(ni::default_matches(*hist, m5pro));
  CHECK(ni::default_matches(*hist, m5max));  // the matrix class includes the wide chips
  CHECK(!ni::default_matches(*hist, ni::resolve_device(mac(9, 20, "Apple M4 Pro", g48))));
  // NI_GPU_CORES can make an M5 Pro look wide. Both rows follow the class, not the name.
  const auto faked = ni::resolve_device(mac(10, 40, "Apple M5 Pro", g48));
  CHECK(faked.chip == ChipClass::kWide);
  CHECK(ni::default_matches(*wave, faked));
  CHECK(ni::default_matches(*hist, faked));

  CHECK(ni::default_count() == 19);  // +1: 9B mixed-format catalog policy (#493); +1: kflash_block16; +1: m5_pro_gpu
  // kflash_block16: a multi-slot server drafts 16 rows on the matrix class (M5 and the M6), not on M1 to M4
  const ni::DefaultRow* b16 = row("kflash_block16");
  CHECK(b16 && b16->scope == ni::DefaultScope::kMatrix && b16->live && !b16->exclusive && b16->win_pct == 22);
  CHECK(ni::default_on(ni::Knob::kKflashBlock16, ni::resolve_device(mac(11, 12, "Apple M6", 32L << 30)), {}));
  CHECK(ni::default_on(ni::Knob::kKflashBlock16, ni::resolve_device(mac(10, 40, "Apple M5 Max", 128L << 30)), {}));
  CHECK(!ni::default_on(ni::Knob::kKflashBlock16, ni::resolve_device(mac(9, 20, "Apple M4 Pro", g48)), {}));
  CHECK(!ni::default_on(ni::Knob::kKflashBlock16, ni::resolve_device(mac(7, 32, "Apple M1 Max", 64L << 30)), {}));
  const auto* segmented = row("ane_segmented_fast");
  CHECK(segmented && segmented->scope == ni::DefaultScope::kAneBuiltinNonMatrix && !segmented->exclusive);
  for (const auto& facts : {mac(7, 8, "Apple M1", g48), mac(8, 10, "Apple M2", g48),
                            mac(9, 60, "Apple M3 Ultra", g48), mac(9, 10, "Apple M4", g48),
                            mac(9, 20, "Apple M4 Pro", g48), mac(9, 40, "Apple M4 Max", g48),
                            mac(10, 20, "Apple M5 Pro", g48), mac(10, 40, "Apple M5 Max", g48)}) {
    const bool expected = facts.name == "Apple M4" || facts.name == "Apple M4 Pro" || facts.name == "Apple M4 Max";
    CHECK(ni::ane_segmented_default(facts) == expected);
    CHECK(ni::ane_builtin_share(facts.name) == (expected || facts.apple_family == 10));
    auto with_ane = ni::resolve_device(facts);
    with_ane.ane = true;
    CHECK(ni::default_matches(*segmented, with_ane) == expected);
    with_ane.ane = false;
    CHECK(!ni::default_matches(*segmented, with_ane));
    // M1 and M2 get default_units()'s own share; it stays apart from the built-in share and the segmented default
    CHECK(ni::ane_pre_m3_share(facts.name) == (facts.apple_family < 9 ? facts.apple_family - 6 : 0));
  }
  // The built-in share is the M4 and every later generation, by the name's generation number, not a substring.
  CHECK(ni::ane_builtin_share("Apple M6") && ni::ane_builtin_share("Apple M6 Pro") && ni::ane_builtin_share("Apple M7 Max"));
  CHECK(ni::ane_builtin_share("Apple M10") && ni::ane_builtin_share("Apple M5 Ultra"));
  CHECK(!ni::ane_builtin_share("Apple M3 Max") && !ni::ane_builtin_share("Apple M1") && !ni::ane_builtin_share("Apple M2 Ultra"));
  CHECK(!ni::ane_builtin_share("Apple M") && !ni::ane_builtin_share("Apple M6X") && !ni::ane_builtin_share("Paravirtual M5"));
  // An M6 is matrix class: the share's gpu_mma branch applies, and the fast profile's segmented inputs stay off.
  CHECK(!ni::ane_segmented_default(mac(11, 12, "Apple M6", 32L << 30)));
  CHECK(ni::ane::default_units("Apple M6", 12, 34, true, true) == 29 && ni::ane::default_units("Apple M6", 12, 18, true, true) == 15);
  // ... but the plain split (Drex Max's noncausal pass, NI_ANE_WAVE=0) keeps no default share on M6 and later until it
  // is output-checked there; M4 and M5 keep theirs
  CHECK(ni::ane::default_units("Apple M6", 12, 34, false, true) == 0 && ni::ane::default_units("Apple M7 Max", 40, 34, false) == 0);
  CHECK(ni::ane::default_units("Apple M5 Pro", 20, 34, false, true) > 0 && ni::ane::default_units("Apple M4 Pro", 20, 34, false) > 0);
  CHECK(ni::ane_generation("Apple M6 Pro") == 6 && ni::ane_generation("Apple M10") == 10 && ni::ane_generation("AMD M6") == 0);
  CHECK(ni::ane_pre_m3_share("Apple M1 Ultra") == 1 && ni::ane_pre_m3_share("Apple M2 Pro") == 2);
  CHECK(ni::ane_pre_m3_share("Apple M2 Max") == 2 && ni::ane_pre_m3_share("Apple M3 Max") == 0);
  CHECK(ni::ane_pre_m3_share("Apple M10") == 0 && ni::ane_pre_m3_share("Apple M20 Pro") == 0);
  // The shares the M1/M2 serve logs printed (bench/results/2026-11-12-m1-m2-ane-share): the 27B's 34 units and the
  // 4B's 18, wavefront; the 32-unit compile cap; the M1 Max share #1194 measured around; no noncausal share; the M3
  // still needs a calibration.
  using ni::ane::default_units;
  using ni::ane::default_min_rows;
  CHECK(default_units("Apple M1 Ultra", 64, 34, true) == 20 && default_units("Apple M2 Pro", 19, 34, true) == 30);
  CHECK(default_units("Apple M1 Ultra", 64, 18, true) == 11 && default_units("Apple M2 Pro", 19, 18, true) == 16);
  CHECK(default_units("Apple M1 Max", 32, 34, true) == 28);
  CHECK(default_units("Apple M1", 8, 34, true) == 32 && default_units("Apple M2", 10, 34, true) == 32);
  CHECK(default_units("Apple M1 Ultra", 64, 34, false) == 0 && default_units("Apple M2 Pro", 19, 34, false) == 0);
  CHECK(default_units("Apple M3 Pro", 18, 34, true) == 0 && default_units("Apple M3 Max", 40, 34, false) == 0);
  CHECK(default_min_rows("Apple M1 Ultra", 64, 20, 34) == 1024 && default_min_rows("Apple M2 Pro", 19, 30, 34) == 512);
  CHECK(default_min_rows("Apple M4 Pro", 20, 27, 34) == 512 && default_min_rows("Apple M5 Max", 40, 14, 34) == 512);
  // The fast profile's first member: matrix class, so every M5, not the M5 Pro it was measured on.
  const ni::DefaultRow* tattn = row("tattn_fast");
  CHECK(tattn && tattn->scope == ni::DefaultScope::kMatrix && !tattn->exclusive && tattn->win_pct == 8);
  CHECK(ni::default_matches(*tattn, m5pro) && ni::default_matches(*tattn, m5max));
  CHECK(!ni::default_matches(*tattn, ni::resolve_device(mac(9, 20, "Apple M4 Pro", g48))));
  const ni::DefaultRow* slices = row("gemm_slices");
  const ni::DefaultRow* hc = row("hc_wide");
  const ni::DefaultRow* moe = row("moe4_dx");
  const ni::DefaultRow* tree = row("tree_g6");
  const ni::DefaultRow* narrow = row("kflash_narrow");
  CHECK(slices && hc && moe && tree && narrow);
  CHECK(slices->scope == ni::DefaultScope::kMatrix && slices->win_pct == 9 && slices->live && !slices->exclusive);
  CHECK(hc->scope == ni::DefaultScope::kWide && hc->win_pct == 2 && moe->win_pct == 1);
  CHECK(tree->win_pct == 27 && narrow->win_pct == 24 && tree->scope == ni::DefaultScope::kWide);
  CHECK(std::string(slices->where).find("widened by judgment, sweep 1") != std::string::npos);
  CHECK(std::string(hc->where).find("widened by judgment, sweep 1") != std::string::npos);
  CHECK(std::string(moe->where).find("widened by judgment, sweep 1") != std::string::npos);
  CHECK(std::string(tree->where) == "M5 Max" && std::string(narrow->where) == "M5 Max");
  CHECK(ni::default_matches(*slices, m5pro) && ni::default_matches(*slices, m5max));
  CHECK(!ni::default_matches(*slices, ni::resolve_device(mac(9, 20, "Apple M4 Pro", g48))));
  CHECK(ni::default_matches(*hc, m5max) && !ni::default_matches(*hc, m5pro));
  CHECK(ni::default_matches(*tree, m5max) && !ni::default_matches(*narrow, m5pro));

  const char* guards[] = {"ultra_gpu", "pre_m3_gpu"};
  for (const char* id : guards) {
    const ni::DefaultRow* g = row(id);
    CHECK(g && g->scope == ni::DefaultScope::kGuard && g->win_pct == 0 && !g->live);
    CHECK(!ni::default_matches(*g, m5pro) && !ni::default_matches(*g, m5max));
  }
  const auto m1p = ni::resolve_device(mac(7, 32, "Apple M1 Max", 64L << 30));
  const auto m4p = ni::resolve_device(mac(9, 10, "Apple M4", 24L << 30));
  const auto m4_8 = ni::resolve_device(mac(9, 8, "Apple M4", 16L << 30));
  const auto m4pro14 = ni::resolve_device(mac(9, 14, "Apple M4 Pro", g48));
  const auto m4pro20 = ni::resolve_device(mac(9, 20, "Apple M4 Pro", g48));
  const auto ultra = ni::resolve_device(mac(10, 80, "Apple M5 Ultra", 512L << 30));
  CHECK(ni::default_matches(*row("ultra_gpu"), ultra) && row("ultra_gpu")->exclusive);
  CHECK(std::string(row("ultra_gpu")->where).find("#421") != std::string::npos);
  CHECK(ni::format_profile(ultra, false).find("exclusive") != std::string::npos);
  const auto m2p = ni::resolve_device(mac(8, 10, "Apple M2", 24L << 30));
  CHECK(ni::default_matches(*row("pre_m3_gpu"), m1p) && ni::default_matches(*row("pre_m3_gpu"), m2p));
  CHECK(!ni::default_matches(*row("pre_m3_gpu"), m4p) && row("pre_m3_gpu")->exclusive);
  CHECK(std::string(row("pre_m3_gpu")->where).find("#478") != std::string::npos);
  CHECK(row("matrix_units") && row("matrix_units")->scope == ni::DefaultScope::kMatrix && !row("matrix_units")->live);
  CHECK(ni::default_matches(*row("matrix_units"), m5pro) && !ni::default_matches(*row("matrix_units"), m4pro20));
  CHECK(row("legacy_gpu")->scope == ni::DefaultScope::kLegacy);
  CHECK(ni::default_matches(*row("legacy_gpu"), m1p) && !ni::default_matches(*row("legacy_gpu"), m5pro));
  CHECK(row("wide_cores")->scope == ni::DefaultScope::kWide);
  CHECK(ni::default_matches(*row("wide_cores"), m5max) && !ni::default_matches(*row("wide_cores"), m5pro));
  // The simd_gpu class row (#393): matches kSimd chips (apple_family 8 and 9), not kMatrix or kLegacy.
  const ni::DefaultRow* simd_row = row("simd_gpu");
  CHECK(simd_row && simd_row->scope == ni::DefaultScope::kSimd && !simd_row->live);
  CHECK(ni::default_matches(*simd_row, m4p) && ni::default_matches(*simd_row, m4pro20));
  CHECK(!ni::default_matches(*simd_row, m5pro) && !ni::default_matches(*simd_row, m1p));

  const auto m5pro_f = mac(10, 20, "Apple M5 Pro", g48);
  const auto m5max_f = mac(10, 40, "Apple M5 Max", 128L << 30);
  const auto faked_f = mac(10, 40, "Apple M5 Pro", g48);
  const auto m1 = mac(7, 32, "Apple M1 Max", 64L << 30);
  const auto m4 = mac(9, 10, "Apple M4", 16L << 30);
  const auto m4pro = mac(9, 20, "Apple M4 Pro", g48);
  CHECK(ni::matrix_units(m5pro_f) && !ni::matrix_units(m4pro));
  CHECK(ni::legacy_gpu(m1) && !ni::legacy_gpu(m4pro) && !ni::legacy_gpu(m5pro_f));
  CHECK(ni::wide_cores(m5max_f) && !ni::wide_cores(m5pro_f) && !ni::wide_cores(unread));
  CHECK(ni::wide_cores(faked_f));  // NI_GPU_CORES=40 on a Pro: the class follows the cores, not the name
  CHECK(ni::ultra_gpu(mac(10, 80, "Apple M5 Ultra", 512L << 30)) && !ni::ultra_gpu(m5max_f));
  CHECK(ni::pre_m3_gpu(m1) && ni::pre_m3_gpu(mac(8, 10, "Apple M2", 24L << 30)) && !ni::pre_m3_gpu(m4pro));
  // simd_gpu: apple_family == 9 (M3, M4), not 7, 8 or 10
  CHECK(ni::simd_gpu(m4) && ni::simd_gpu(m4pro));
  CHECK(!ni::simd_gpu(m1) && !ni::simd_gpu(m5pro_f) && !ni::simd_gpu(mac(8, 10, "Apple M2", 24L << 30)));
  // m1_max_32: kept as guard (#393)
  CHECK(ni::m1_max_32(m1));  // m1 is an M1 Max 32-core
  CHECK(!ni::m1_max_32(m4) && !ni::m1_max_32(m5pro_f) && !ni::m1_max_32(mac(7, 16, "Apple M1 Pro", 32L << 30)));
  // m5_pro_gpu: the 27B's 64-row tensor tile keeps 64 rows there (a measured miss at 17); the name, not the cores
  CHECK(ni::m5_pro_gpu(m5pro_f) && ni::m5_pro_gpu(faked_f) && !ni::m5_pro_gpu(m5max_f) && !ni::m5_pro_gpu(m4pro));
  CHECK(!ni::m5_pro_gpu(mac(10, 12, "Apple M6", 32L << 30)) && !ni::m5_pro_gpu(mac(10, 10, "Apple M5", 24L << 30)));
  CHECK(row("m5_pro_gpu")->scope == ni::DefaultScope::kGuard && row("m5_pro_gpu")->exclusive);
  CHECK(ni::default_matches(*row("m5_pro_gpu"), m5pro) && !ni::default_matches(*row("m5_pro_gpu"), m5max));

  const std::string text = ni::format_profile(m5pro, false);
  CHECK(text.find("chip class: matrix") != std::string::npos);
  CHECK(text.find("the call sites read this table") != std::string::npos);
  CHECK(text.find("pf_wave") != std::string::npos);
  const std::string json = ni::format_profile(m5max, true);
  CHECK(json.find("\"chip_class\":\"wide\"") != std::string::npos);
  CHECK(json.find("\"id\":\"pf_wave\"") != std::string::npos);
  CHECK(json.find("\"win_pct\":3") != std::string::npos);
  CHECK(json.find("\"exclusive\":false") != std::string::npos);
  CHECK(json.find("measured_on") == std::string::npos);
  ni::DeviceProfile quoted = m5pro;
  quoted.name = "Apple \"M5\"\\Pro\n";
  const std::string escaped = ni::format_profile(quoted, true);
  CHECK(escaped.find("\"name\":\"Apple \\\"M5\\\"\\\\Pro\\u000a\"") != std::string::npos);
  CHECK(ni::apple_family_probed([](int n) { return n == 10; }) == 10);
  CHECK(ni::apple_family_probed([](int n) { return n == 11 || n == 10; }) == 11);
  CHECK(ni::apple_family_probed([](int) { return false; }) == 7);
  CHECK(ni::apple_family_probed([](int n) { return n == 8; }) == 8);
  CHECK(ni::apple_family_probed([](int n) { return n <= 16; }) == ni::kAppleFamilyProbeMax);

  // Every knob has exactly one row, and the row's id stays the string the lint and the doctor print.
  const ni::Knob knobs[] = {ni::Knob::kPfWave,      ni::Knob::kKflashHist,  ni::Knob::kGemmSlices, ni::Knob::kHcWide,
                            ni::Knob::kMoe4Dx,      ni::Knob::kTreeG6,      ni::Knob::kKflashNarrow};
  const char* const ids[] = {"pf_wave", "kflash_hist", "gemm_slices", "hc_wide", "moe4_dx", "tree_g6", "kflash_narrow"};
  for (int k = 0; k < 7; ++k) {
    int n = 0;
    for (int i = 0; i < ni::default_count(); ++i)
      if (ni::default_at(i).knob == knobs[k]) {
        ++n;
        CHECK(std::string(ni::default_at(i).id) == ids[k]);
        CHECK(ni::default_at(i).live);
      }
    CHECK(n == 1);
  }

  // The proof the move changes nothing: default_on() must agree, on every (device, model) pair, with the inline
  // expression the call site had before it read the table. These lambdas are those expressions, copied verbatim.
  const auto old_pf_wave = [](const ni::DeviceFacts& d, const ni::ModelFacts& m) {  // matrix class since #393
    return m.moe && m.hidden == 2048 && m.layers == 40 && d.apple_family >= 10;
  };
  const auto old_hist = [](const ni::DeviceFacts& d, const ni::ModelFacts& m) {
    return d.apple_family >= 10 && m.moe && !m.clover && m.hidden == 2048 && m.layers == 40 && m.experts == 256 &&
           m.top_k == 8 && m.moe_i == 512 && m.shared_i == 512;
  };
  const auto old_matrix = [](const ni::DeviceFacts& d, const ni::ModelFacts&) { return d.apple_family >= 10; };
  const auto old_wide = [](const ni::DeviceFacts& d, const ni::ModelFacts&) {
    return d.apple_family >= 10 && d.cores >= 40;
  };
  ni::ModelFacts models[5];
  models[0] = ni::ModelFacts{};  // a model nothing keys on
  models[1].moe = true;  // the 35B-A3B
  models[1].hidden = 2048;
  models[1].layers = 40;
  models[1].experts = 256;
  models[1].top_k = 8;
  models[1].moe_i = 512;
  models[1].shared_i = 512;
  models[2] = models[1];
  models[2].clover = true;  // Clover-35B-A3B
  models[3].moe = true;  // the wave's shape without the drafter fields (engine.cpp's chunk pick fills only these)
  models[3].hidden = 2048;
  models[3].layers = 40;
  models[4].hidden = 3584;  // a dense 27B
  models[4].layers = 64;
  const int families[] = {5, 7, 8, 9, 10, 11, 16};
  const int cores[] = {0, 8, 10, 20, 24, 32, 39, 40, 48, 64, 80};
  for (int fam : families)
    for (int c : cores)
      for (const ni::ModelFacts& m : models) {
        const ni::DeviceFacts d = mac(fam, c, "Apple M5 Pro", g48);
        const ni::DeviceProfile p = ni::resolve_device(d);
        CHECK(ni::default_on(ni::Knob::kPfWave, p, m) == old_pf_wave(d, m));
        CHECK(ni::default_on(ni::Knob::kKflashHist, p, m) == old_hist(d, m));
        CHECK(ni::default_on(ni::Knob::kGemmSlices, p, m) == old_matrix(d, m));
        CHECK(ni::default_on(ni::Knob::kHcWide, p, m) == old_wide(d, m));
        CHECK(ni::default_on(ni::Knob::kMoe4Dx, p, m) == old_wide(d, m));
        CHECK(ni::default_on(ni::Knob::kTreeG6, p, m) == old_wide(d, m));
        CHECK(ni::default_on(ni::Knob::kKflashNarrow, p, m) == old_wide(d, m));
      }

  std::printf("device_profile_test: ok (%d defaults)\n", ni::default_count());
  return 0;
}
