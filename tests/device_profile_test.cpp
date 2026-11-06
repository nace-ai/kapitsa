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
  CHECK(ni::resolve_model(a3b) == ni::ModelClass::kOther);

  const ni::DefaultRow* wave = row("pf_wave");
  const ni::DefaultRow* hist = row("kflash_hist");
  CHECK(wave && hist);
  CHECK(!wave->live && !hist->live);
  CHECK(wave->win_pct == 5 && wave->win_pct_hi == 0 && !wave->exclusive);
  CHECK(hist->win_pct == 12 && hist->win_pct_hi == 15 && !hist->exclusive);
  const auto m5pro = ni::resolve_device(mac(10, 20, "Apple M5 Pro", g48));
  const auto m5max = ni::resolve_device(mac(10, 40, "Apple M5 Max", 128L << 30));
  CHECK(!ni::default_matches(*wave, m5pro));
  CHECK(ni::default_matches(*wave, m5max));
  CHECK(ni::default_matches(*hist, m5pro));
  CHECK(ni::default_matches(*hist, m5max));  // the matrix class includes the wide chips
  CHECK(!ni::default_matches(*hist, ni::resolve_device(mac(9, 20, "Apple M4 Pro", g48))));
  // NI_GPU_CORES can make an M5 Pro look wide. Both rows follow the class, not the name.
  const auto faked = ni::resolve_device(mac(10, 40, "Apple M5 Pro", g48));
  CHECK(faked.chip == ChipClass::kWide);
  CHECK(ni::default_matches(*wave, faked));
  CHECK(ni::default_matches(*hist, faked));

  const auto m5pro_f = mac(10, 20, "Apple M5 Pro", g48);
  const auto m5max_f = mac(10, 40, "Apple M5 Max", 128L << 30);
  const auto faked_f = mac(10, 40, "Apple M5 Pro", g48);
  const auto m1 = mac(7, 32, "Apple M1 Max", 64L << 30);
  const auto m1_24 = mac(7, 24, "Apple M1 Max", 64L << 30);
  const auto m4 = mac(9, 10, "Apple M4", 16L << 30);
  const auto m4pro = mac(9, 20, "Apple M4 Pro", g48);
  CHECK(ni::matrix_units(m5pro_f) && !ni::matrix_units(m4pro));
  CHECK(ni::legacy_gpu(m1) && !ni::legacy_gpu(m4pro) && !ni::legacy_gpu(m5pro_f));
  CHECK(ni::m1_max_32(m1) && !ni::m1_max_32(m1_24) && !ni::m1_max_32(m5max_f));
  CHECK(ni::wide_cores(m5max_f) && !ni::wide_cores(m5pro_f) && !ni::wide_cores(unread));
  CHECK(ni::wide_cores(faked_f));  // NI_GPU_CORES=40 on a Pro: the class follows the cores, not the name
  CHECK(ni::m4_10(m4) && !ni::m4_10(mac(9, 8, "Apple M4", 16L << 30)) && !ni::m4_10(m4pro));
  CHECK(ni::m4_pro_gpu(m4pro) && ni::m4_pro_gpu(mac(9, 14, "Apple M4 Pro", g48)) && !ni::m4_pro_gpu(m4));
  CHECK(ni::m4_pro_20(m4pro) && !ni::m4_pro_20(mac(9, 14, "Apple M4 Pro", g48)));
  CHECK(ni::ultra_gpu(mac(10, 80, "Apple M5 Ultra", 512L << 30)) && !ni::ultra_gpu(m5max_f));
  CHECK(ni::pre_m3_gpu(m1) && ni::pre_m3_gpu(mac(8, 10, "Apple M2", 24L << 30)) && !ni::pre_m3_gpu(m4pro));
  CHECK(ni::m4_ten_core(m4) && !ni::m4_ten_core(m4pro));

  const std::string text = ni::format_profile(m5pro, false);
  CHECK(text.find("chip class: matrix") != std::string::npos);
  CHECK(text.find("call sites still decide") != std::string::npos);
  CHECK(text.find("pf_wave") != std::string::npos);
  const std::string json = ni::format_profile(m5max, true);
  CHECK(json.find("\"chip_class\":\"wide\"") != std::string::npos);
  CHECK(json.find("\"id\":\"pf_wave\"") != std::string::npos);
  CHECK(json.find("\"win_pct\":5") != std::string::npos);
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
  std::printf("device_profile_test: ok (%d defaults)\n", ni::default_count());
  return 0;
}
