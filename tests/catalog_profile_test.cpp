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

// `--model <combined KQuant repo>` in every form a user types takes the catalog variant that pins the same pack, with
// its drafter and its MTP head where the variant has one (#1210). `mixed`: the 9B's Q4 pack is `4bit` on Macs with
// matrix units and `kquant` on the others, where `4bit` is the legacy pack.
static void combined_ids(bool mixed) {
  using namespace ni::hub;
  struct Form { std::string id; const char* model; const char* variant; const char* repo; const char* drafter; bool mtp; };
  const std::string q4b = "nace-ai/Qwen3.5-4B-MLX-KQuant", r4b = "84168d4d60c807a3c8372ee60f9d2034d5166964";
  const std::string q9 = "nace-ai/Qwen3.5-9B-MLX-KQuant", r9 = "02b8c7669c8dee02a5efa475a478bcca966a24d1";
  const std::string q27 = "nace-ai/Qwen3.8-27B-MLX-KQuant", r27 = "105530e56743ac3515a34b274b3865965dc83bc6";
  const std::string q35 = "nace-ai/Qwen3.6-35B-A3B-MLX-KQuant", r35 = "f82ecb7f83f578e2d599e36e27177cc97d02c1f1";
  const std::string fn = "nace-ai/Qwen3.8-Flash-Next-MLX-KQuant", rfn = "8fda104737236c4bbe1d6fb2784fb499ffd0a4cd";
  const char* nine_q4 = mixed ? "4bit" : "kquant";
  const char *d4 = "z-lab/Qwen3.5-4B-DFlash", *d9 = "z-lab/Qwen3.5-9B-DFlash", *d27 = "nace-ai/Qwen3.8-27B-DFlash2-Kapitsa",
             *d35 = "incoai/Qwen3.6-35B-A3B-DFlash2";
  const std::vector<Form> forms = {
      {q4b, "qwen3.5-4b", "4bit", "nace-ai/Qwen3.5-4B-MLX-Q4-KQuant", d4, true},
      {q4b + "/Q4", "qwen3.5-4b", "4bit", "nace-ai/Qwen3.5-4B-MLX-Q4-KQuant", d4, true},
      {q4b + "@" + r4b, "qwen3.5-4b", "4bit", "nace-ai/Qwen3.5-4B-MLX-Q4-KQuant", d4, true},
      {q4b + "/Q4@" + r4b, "qwen3.5-4b", "4bit", "nace-ai/Qwen3.5-4B-MLX-Q4-KQuant", d4, true},
      {"https://huggingface.co/" + q4b + "/Q4", "qwen3.5-4b", "4bit", "nace-ai/Qwen3.5-4B-MLX-Q4-KQuant", d4, true},
      {q9, "qwen3.5-9b", nine_q4, "nace-ai/Qwen3.5-9B-MLX-Q4-KQuant", d9, true},
      {q9 + "/Q4", "qwen3.5-9b", nine_q4, "nace-ai/Qwen3.5-9B-MLX-Q4-KQuant", d9, true},
      {q9 + "@" + r9, "qwen3.5-9b", nine_q4, "nace-ai/Qwen3.5-9B-MLX-Q4-KQuant", d9, true},
      {q9 + "/Q4@" + r9, "qwen3.5-9b", nine_q4, "nace-ai/Qwen3.5-9B-MLX-Q4-KQuant", d9, true},
      {q9 + "/Q5", "qwen3.5-9b", "kquant5", "nace-ai/Qwen3.5-9B-MLX-Q5-KQuant", d9, true},
      {q9 + "/Q5@" + r9, "qwen3.5-9b", "kquant5", "nace-ai/Qwen3.5-9B-MLX-Q5-KQuant", d9, true},
      {"hf.co/" + q9 + "/Q5/", "qwen3.5-9b", "kquant5", "nace-ai/Qwen3.5-9B-MLX-Q5-KQuant", d9, true},
      {q27, "qwen3.8-27b", "kquant", "nace-ai/Qwen3.8-27B-MLX-Q4-KQuant", d27, false},
      {q27 + "/Q4", "qwen3.8-27b", "kquant", "nace-ai/Qwen3.8-27B-MLX-Q4-KQuant", d27, false},
      {q27 + "/Q4@" + r27, "qwen3.8-27b", "kquant", "nace-ai/Qwen3.8-27B-MLX-Q4-KQuant", d27, false},
      {q27 + "/Q5", "qwen3.8-27b", "kquant5", "nace-ai/Qwen3.8-27B-MLX-Q5-KQuant", d27, true},
      {q27 + "/Q5@" + r27, "qwen3.8-27b", "kquant5", "nace-ai/Qwen3.8-27B-MLX-Q5-KQuant", d27, true},
      {q35, "qwen3.6-35b-a3b", "4bit", "nace-ai/Qwen3.6-35B-A3B-MLX-Q4-KQuant", d35, false},
      {q35 + "/Q4", "qwen3.6-35b-a3b", "4bit", "nace-ai/Qwen3.6-35B-A3B-MLX-Q4-KQuant", d35, false},
      {q35 + "/Q4@" + r35, "qwen3.6-35b-a3b", "4bit", "nace-ai/Qwen3.6-35B-A3B-MLX-Q4-KQuant", d35, false},
      {q35 + "/Q5", "qwen3.6-35b-a3b", "kquant5", "nace-ai/Qwen3.6-35B-A3B-MLX-Q5-KQuant", d35, true},
      {q35 + "/Q5@" + r35, "qwen3.6-35b-a3b", "kquant5", "nace-ai/Qwen3.6-35B-A3B-MLX-Q5-KQuant", d35, true},
      {fn, "qwen3.8-flash-next", "kquant", "nace-ai/Qwen3.8-Flash-Next-MLX-Q4-KQuant", nullptr, false},
      {fn + "@" + rfn, "qwen3.8-flash-next", "kquant", "nace-ai/Qwen3.8-Flash-Next-MLX-Q4-KQuant", nullptr, false},
      // the standalone ids still name their variants, as on main
      {"nace-ai/Qwen3.5-9B-MLX-Q5-KQuant", "qwen3.5-9b", "kquant5", "nace-ai/Qwen3.5-9B-MLX-Q5-KQuant", d9, true},
      {"nace-ai/Qwen3.5-4B-MLX-Q4-KQuant", "qwen3.5-4b", "4bit", "nace-ai/Qwen3.5-4B-MLX-Q4-KQuant", d4, true},
  };
  for (const auto& f : forms) {
    const auto r = resolve(f.id);
    if (r.kind != ModelRef::kCatalog || r.model->name != f.model || r.variant->name != f.variant || r.repo != f.repo) {
      std::fprintf(stderr, "%s -> %s:%s %s\n", f.id.c_str(), r.model ? r.model->name.c_str() : "(not catalog)",
                   r.variant ? r.variant->name.c_str() : "", r.repo.c_str());
      CHECK(false);
    }
    bool drafter = false, mtp = false;
    for (const auto& c : r.variant->companions) {
      drafter = drafter || (c.drafter() && f.drafter && c.repo == f.drafter);
      mtp = mtp || c.kind == "mtp" || c.kind == "mtp-sidecar";
    }
    CHECK(drafter == (f.drafter != nullptr) && mtp == f.mtp);
    CHECK(r.revision == r.variant->revision && r.variant_given);
  }
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
      combined_ids(mixed);
    }
  }
  // a combined repo's folder or revision that no catalog entry pins is refused, never a plain download (#1210)
  for (const char* id : {"nace-ai/Qwen3.8-Flash-Next-MLX-KQuant/Q4", "nace-ai/Qwen3.5-4B-MLX-KQuant/Q5",
                         "nace-ai/Qwen3.5-9B-MLX-KQuant/Q6", "nace-ai/Qwen3.5-9B-MLX-KQuant@main",
                         "nace-ai/Qwen3.5-9B-MLX-KQuant/Q5@0123456789abcdef0123456789abcdef01234567"}) {
    bool refused = false;
    try {
      resolve(id);
    } catch (const Error&) {
      refused = true;
    }
    if (!refused) std::fprintf(stderr, "not refused: %s\n", id);
    CHECK(refused);
  }
  std::puts("catalog_profile_test: 24 device fixtures PASS; aliases, pins, companions and other models unchanged");
}
