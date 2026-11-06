#include "engine/device_profile.h"

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
  const auto m5pro = ni::resolve_device(mac(10, 20, "Apple M5 Pro", g48));
  const auto m5max = ni::resolve_device(mac(10, 40, "Apple M5 Max", 128L << 30));
  CHECK(!ni::default_matches(*wave, m5pro));
  CHECK(ni::default_matches(*wave, m5max));
  CHECK(ni::default_matches(*hist, m5pro));
  CHECK(!ni::default_matches(*hist, m5max));
  // NI_GPU_CORES can make an M5 Pro look wide. The wave follows cores; the lane follows the name.
  const auto faked = ni::resolve_device(mac(10, 40, "Apple M5 Pro", g48));
  CHECK(faked.chip == ChipClass::kWide);
  CHECK(ni::default_matches(*wave, faked));
  CHECK(ni::default_matches(*hist, faked));

  const std::string text = ni::format_profile(m5pro, false);
  CHECK(text.find("chip class: matrix") != std::string::npos);
  CHECK(text.find("call sites still decide") != std::string::npos);
  CHECK(text.find("pf_wave") != std::string::npos);
  const std::string json = ni::format_profile(m5max, true);
  CHECK(json.find("\"chip_class\":\"wide\"") != std::string::npos);
  CHECK(json.find("\"id\":\"pf_wave\"") != std::string::npos);
  std::printf("device_profile_test: ok (%d defaults)\n", ni::default_count());
  return 0;
}
