#include "engine/device_profile.h"
#include "hub/hub.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #condition); std::abort(); } } while (false)

// Inject the live-device seam: this target links no Metal code and opens no GPU queue.
namespace ni {
DeviceProfile test_device;
const DeviceProfile& catalog_device() { return test_device; }
const DeviceProfile& current_device() { std::abort(); }  // Catalog lookup must not initialize the GPU singleton.
}

int main() {
  using namespace ni::hub;
  const auto baseline = catalog(ni::resolve_device({.apple_family = 10, .cores = 20}));
  for (int family : {0, 7, 8, 9, 10, 11}) {
    for (int cores : {8, 20, 40, 60}) {
      ni::test_device = ni::resolve_device({.apple_family = family, .cores = cores});
      const bool mixed = family >= 10;
      const auto& selected = catalog();
      CHECK(selected.size() == baseline.size());
      for (size_t i = 0; i < selected.size(); ++i) {
        const auto& model = selected[i];
        const auto& original = baseline[i];
        CHECK(model.name == original.name && model.aliases == original.aliases);
        CHECK(model.default_variant == original.default_variant);
        CHECK(model.variants.size() == original.variants.size());
        for (size_t j = 0; j < model.variants.size(); ++j) {
          const auto& v = model.variants[j];
          const auto& before = original.variants[j];
          CHECK(v.name == before.name);
          CHECK(v.companions.size() == before.companions.size());
          for (size_t k = 0; k < v.companions.size(); ++k) {
            const auto& a = v.companions[k];
            const auto& b = before.companions[k];
            CHECK(a.kind == b.kind && a.repo == b.repo && a.revision == b.revision);
            CHECK(a.files == b.files && a.subdir == b.subdir && a.sha256 == b.sha256 && a.size == b.size);
          }
          if (model.name != "qwen3.5-9b") {
            CHECK(v.repo == before.repo && v.revision == before.revision);
            CHECK(v.quant == before.quant && v.download_gb == before.download_gb);
          }
        }
      }
      const auto* nine = find_model("9b");
      CHECK(nine && nine == find_model("qwen3.5-9b"));
      const auto* v = nine->variant();
      CHECK(v && v == nine->variant("4bit"));
      CHECK(v->repo == (mixed ? "nace-ai/Qwen3.5-9B-MLX-Q4-KQuant" : "nace-ai/Qwen3.5-9B-MLX-4bit-KQuant"));
      CHECK(v->revision == (mixed ? "ff7b61b8c655186cdd50ddc4f485d3d5b93e1af4" : "867adaa1c6d84d1fd9b4f37f43836019c877ba59"));
      CHECK(v->download_gb == (mixed ? 9.30 : 9.05));
      CHECK(v->quant == (mixed ? "MLX affine mixed 4/5-bit (KQuant Q4)" : "MLX affine 4-bit, group 64 (KQuant)"));
      CHECK(find_variant_by_repo(v->repo, v->revision, nullptr) == v);
      for (const char* alias : {"9b", "qwen3.5-9b", "9b:4bit"}) {
        const auto r = resolve(alias);
        CHECK(r.repo == v->repo && r.revision == v->revision);
        CHECK(r.variant == v && r.model == nine);
      }
      CHECK(ni::format_profile(ni::test_device, true).find("qwen9b_q4") != std::string::npos);
    }
  }
  std::puts("catalog_profile_test: 24 device fixtures PASS; aliases, pins, companions and other models unchanged");
}
