#include "engine/device_profile.h"
#include "hub/hub.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#include <sys/stat.h>
#include <unistd.h>

#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #condition); std::abort(); } } while (false)

// Inject the live-device seam: this target links no Metal code and opens no GPU queue.
namespace ni {
DeviceProfile test_device;
const DeviceProfile& catalog_device() { return test_device; }
const DeviceProfile& current_device() { std::abort(); }  // Catalog lookup must not initialize the GPU singleton.
}

// `--model <combined Nace repo>` in every form a user types takes the catalog variant that pins the same pack, with
// its drafter and its MTP head where the variant has one (#1210). `mixed`: the 9B's Q4 pack is `4bit` on Macs with
// matrix units and `q4` on the others, where `4bit` is the legacy pack.
static void combined_ids(bool mixed) {
  using namespace ni::hub;
  struct Form { std::string id; const char* model; const char* variant; const char* repo; const char* drafter; bool mtp; };
  const std::string q4b = "nace-ai/Qwen3.5-4B-MLX", r4b = "84168d4d60c807a3c8372ee60f9d2034d5166964";
  const std::string q9 = "nace-ai/Qwen3.5-9B-MLX", r9 = "02b8c7669c8dee02a5efa475a478bcca966a24d1";
  const std::string q27 = "nace-ai/Qwen3.8-27B-MLX", r27 = "105530e56743ac3515a34b274b3865965dc83bc6";
  const std::string q35 = "nace-ai/Qwen3.6-35B-A3B-MLX", r35 = "f82ecb7f83f578e2d599e36e27177cc97d02c1f1";
  const std::string fn = "nace-ai/Qwen3.8-Flash-Next-MLX", rfn = "8fda104737236c4bbe1d6fb2784fb499ffd0a4cd";
  // the revisions the catalog pins in those repos' roots (the standalone repos' own, kept through the rename)
  const std::string p4b = "9f7d3278cdab59ea0d8c23c25f89987b36e8cd89", p9 = "ff7b61b8c655186cdd50ddc4f485d3d5b93e1af4",
                    p27 = "8e62e9e0d989f71ce2d0ab5c4b8e0249c1e33364", p35 = "37e804f90e368a4de866bc34650b9dde67a78c8e";
  const char* nine_q4 = mixed ? "4bit" : "q4";
  const char *d4 = "z-lab/Qwen3.5-4B-DFlash", *d9 = "z-lab/Qwen3.5-9B-DFlash", *d27 = "nace-ai/Qwen3.8-27B-DFlash2-Kapitsa",
             *d35 = "incoai/Qwen3.6-35B-A3B-DFlash2";
  const std::vector<Form> forms = {
      {q4b, "qwen3.5-4b", "4bit", "nace-ai/Qwen3.5-4B-MLX", d4, true},
      {q4b + "/Q4", "qwen3.5-4b", "4bit", "nace-ai/Qwen3.5-4B-MLX", d4, true},
      {q4b + "@" + r4b, "qwen3.5-4b", "4bit", "nace-ai/Qwen3.5-4B-MLX", d4, true},
      {q4b + "/Q4@" + r4b, "qwen3.5-4b", "4bit", "nace-ai/Qwen3.5-4B-MLX", d4, true},
      {"https://huggingface.co/" + q4b + "/Q4", "qwen3.5-4b", "4bit", "nace-ai/Qwen3.5-4B-MLX", d4, true},
      {q9, "qwen3.5-9b", nine_q4, "nace-ai/Qwen3.5-9B-MLX", d9, true},
      {q9 + "/Q4", "qwen3.5-9b", nine_q4, "nace-ai/Qwen3.5-9B-MLX", d9, true},
      {q9 + "@" + r9, "qwen3.5-9b", nine_q4, "nace-ai/Qwen3.5-9B-MLX", d9, true},
      {q9 + "/Q4@" + r9, "qwen3.5-9b", nine_q4, "nace-ai/Qwen3.5-9B-MLX", d9, true},
      {q9 + "/Q5", "qwen3.5-9b", "q5", "nace-ai/Qwen3.5-9B-MLX-Q5", d9, true},
      {q9 + "/Q5@" + r9, "qwen3.5-9b", "q5", "nace-ai/Qwen3.5-9B-MLX-Q5", d9, true},
      {"hf.co/" + q9 + "/Q5/", "qwen3.5-9b", "q5", "nace-ai/Qwen3.5-9B-MLX-Q5", d9, true},
      {q27, "qwen3.8-27b", "q4", "nace-ai/Qwen3.8-27B-MLX", d27, false},
      {q27 + "/Q4", "qwen3.8-27b", "q4", "nace-ai/Qwen3.8-27B-MLX", d27, false},
      {q27 + "/Q4@" + r27, "qwen3.8-27b", "q4", "nace-ai/Qwen3.8-27B-MLX", d27, false},
      {q27 + "/Q5", "qwen3.8-27b", "q5", "nace-ai/Qwen3.8-27B-MLX-Q5", d27, true},
      {q27 + "/Q5@" + r27, "qwen3.8-27b", "q5", "nace-ai/Qwen3.8-27B-MLX-Q5", d27, true},
      {q35, "qwen3.6-35b-a3b", "4bit", "nace-ai/Qwen3.6-35B-A3B-MLX", d35, false},
      {q35 + "/Q4", "qwen3.6-35b-a3b", "4bit", "nace-ai/Qwen3.6-35B-A3B-MLX", d35, false},
      {q35 + "/Q4@" + r35, "qwen3.6-35b-a3b", "4bit", "nace-ai/Qwen3.6-35B-A3B-MLX", d35, false},
      {q35 + "/Q5", "qwen3.6-35b-a3b", "q5", "nace-ai/Qwen3.6-35B-A3B-MLX-Q5", d35, true},
      {q35 + "/Q5@" + r35, "qwen3.6-35b-a3b", "q5", "nace-ai/Qwen3.6-35B-A3B-MLX-Q5", d35, true},
      {fn, "qwen3.8-flash-next", "q4", "nace-ai/Qwen3.8-Flash-Next-MLX", nullptr, false},
      {fn + "@" + rfn, "qwen3.8-flash-next", "q4", "nace-ai/Qwen3.8-Flash-Next-MLX", nullptr, false},
      // the revision a variant downloads is one of its repo's own: accepted beside the combined revision (#513)
      {q4b + "@" + p4b, "qwen3.5-4b", "4bit", "nace-ai/Qwen3.5-4B-MLX", d4, true},
      {q9 + "@" + p9, "qwen3.5-9b", nine_q4, "nace-ai/Qwen3.5-9B-MLX", d9, true},
      {q27 + "@" + p27, "qwen3.8-27b", "q4", "nace-ai/Qwen3.8-27B-MLX", d27, false},
      {q35 + "@" + p35, "qwen3.6-35b-a3b", "4bit", "nace-ai/Qwen3.6-35B-A3B-MLX", d35, false},
      // the standalone ids still name their variants: the Q5 repos exist, and the Hub redirects the old Q4 ones to the
      // combined repos, so those downloads now name the combined repo (#513)
      {"nace-ai/Qwen3.5-9B-MLX-Q5", "qwen3.5-9b", "q5", "nace-ai/Qwen3.5-9B-MLX-Q5", d9, true},
      {"nace-ai/Qwen3.5-4B-MLX-Q4-KQuant", "qwen3.5-4b", "4bit", "nace-ai/Qwen3.5-4B-MLX", d4, true},  // redirected old id
      {"nace-ai/Qwen3.5-4B-MLX-Q4-KQuant@" + p4b, "qwen3.5-4b", "4bit", "nace-ai/Qwen3.5-4B-MLX", d4, true},  // redirected old id
      {"nace-ai/Qwen3.5-9B-MLX-Q4-KQuant", "qwen3.5-9b", nine_q4, "nace-ai/Qwen3.5-9B-MLX", d9, true},  // redirected old id
      {"nace-ai/Qwen3.8-27B-MLX-Q4-KQuant", "qwen3.8-27b", "q4", "nace-ai/Qwen3.8-27B-MLX", d27, false},  // redirected old id
      {"nace-ai/Qwen3.6-35B-A3B-MLX-Q4-KQuant@" + p35, "qwen3.6-35b-a3b", "4bit", "nace-ai/Qwen3.6-35B-A3B-MLX", d35,  // redirected old id
       false},
      {"nace-ai/Qwen3.8-Flash-Next-MLX-Q4-KQuant", "qwen3.8-flash-next", "q4", "nace-ai/Qwen3.8-Flash-Next-MLX", nullptr,  // redirected old id
       false},
      {"nace-ai/Qwen3.5-9B-MLX-KQuant/Q5", "qwen3.5-9b", "q5", "nace-ai/Qwen3.5-9B-MLX-Q5", d9, true},  // redirected old id
      {"nace-ai/Qwen3.8-27B-MLX-KQuant", "qwen3.8-27b", "q4", "nace-ai/Qwen3.8-27B-MLX", d27, false},  // redirected old id
      {"nace-ai/Qwen3.8-Flash-Next-MLX-KQuant@" + rfn, "qwen3.8-flash-next", "q4", "nace-ai/Qwen3.8-Flash-Next-MLX", nullptr,  // redirected old id
       false},
      {"nace-ai/Qwen3.5-9B-MLX-Q5-KQuant", "qwen3.5-9b", "q5", "nace-ai/Qwen3.5-9B-MLX-Q5", d9, true},  // redirected old id
      {"nace-ai/Qwen3.8-27B-MLX-Q5-KQuant", "qwen3.8-27b", "q5", "nace-ai/Qwen3.8-27B-MLX-Q5", d27, true},  // redirected old id
      {"nace-ai/Qwen3.6-35B-A3B-MLX-Q5-KQuant", "qwen3.6-35b-a3b", "q5", "nace-ai/Qwen3.6-35B-A3B-MLX-Q5", d35, true},  // redirected old id
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

// The variants renamed on 13 Nov 2026 (#530) keep their old names as hidden aliases, and a pack downloaded under an old
// name is read where it is: no folder is renamed or moved, and the pack is not downloaded again.
static void renamed_variants() {
  using namespace ni::hub;
  ni::test_device = ni::resolve_device({.apple_family = 10, .cores = 20});
  const auto* m27 = find_model("27b");
  CHECK(m27 && m27->default_variant == "q4" && m27->variant("kquant") == m27->variant("q4") && m27->variant("q4"));
  CHECK(m27->variant("KQuant5") == m27->variant("q5") && m27->variant("q5"));
  CHECK(!m27->variant("kquant6"));
  for (const auto& m : catalog())
    for (const auto& v : m.variants) CHECK(v.name != "kquant" && v.name != "kquant5");  // hidden: never listed
  for (const char* id : {"27b:kquant", "27B:KQUANT", "qwen3.8-27b:kquant"}) CHECK(resolve(id).variant == m27->variant("q4"));
  CHECK(resolve("9b:kquant5").variant == find_model("9b")->variant("q5"));
  CHECK(resolve("flash:kquant").variant == find_model("flash")->variant("q4"));
  CHECK(resolve("35b:kquant5").variant == find_model("35b")->variant("q5"));
  bool refused = false;
  try {
    resolve("35b:kquant");  // the 35B has no Q4 variant (its `4bit` is the Nace pack)
  } catch (const Error&) {
    refused = true;
  }
  CHECK(refused);
}

static void write_file(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "w");
  CHECK(f);
  std::fputs("x", f);
  std::fclose(f);
}

static bool exists(const std::string& path) {
  struct stat st {};
  return stat(path.c_str(), &st) == 0;
}

static void renamed_dirs() {
  using namespace ni::hub;
  char tmpl[] = "/tmp/kapitsa-catalog-test-XXXXXX";
  const std::string home = mkdtemp(tmpl);
  setenv("KAPITSA_HOME", home.c_str(), 1);
  const std::string models = home + "/models";
  mkdir(models.c_str(), 0755);
  // the 27B downloaded as `kquant` is `q4` now: its folder is used in place, and no folder is created or renamed
  mkdir((models + "/qwen3.8-27b-kquant").c_str(), 0755);
  write_file(models + "/qwen3.8-27b-kquant/model.safetensors");
  for (const char* id : {"27b", "27b:q4", "27b:kquant"}) {
    const auto r = resolve(id);
    CHECK(r.dir == models + "/qwen3.8-27b-kquant");
    CHECK(exists(models + "/qwen3.8-27b-kquant/model.safetensors") && !exists(models + "/qwen3.8-27b-q4"));
  }
  // a Q5 pack the same way, named by its old variant
  mkdir((models + "/qwen3.6-35b-a3b-kquant5").c_str(), 0755);
  write_file(models + "/qwen3.6-35b-a3b-kquant5/model.safetensors");
  CHECK(resolve("35b:kquant5").dir == models + "/qwen3.6-35b-a3b-kquant5");
  CHECK(resolve("35b:q5").dir == models + "/qwen3.6-35b-a3b-kquant5");
  CHECK(exists(models + "/qwen3.6-35b-a3b-kquant5/model.safetensors") && !exists(models + "/qwen3.6-35b-a3b-q5"));
  // the old name does not map to another variant: a Q5 folder is not a Q4 pack
  CHECK(resolve("35b:q5").dir != models + "/qwen3.6-35b-a3b-kquant" && legacy_dir_name(*find_model("35b"), *find_model("35b")->variant("4bit")).empty());
  // a new folder is never replaced by an old one
  mkdir((models + "/qwen3.5-9b-q4").c_str(), 0755);
  mkdir((models + "/qwen3.5-9b-kquant").c_str(), 0755);
  CHECK(resolve("9b:q4").dir == models + "/qwen3.5-9b-q4" && exists(models + "/qwen3.5-9b-kquant"));
  CHECK(resolve("9b:kquant").dir == models + "/qwen3.5-9b-q4");
  // no folder of either name: the new one, to be downloaded
  CHECK(resolve("flash:q4").dir == models + "/qwen3.8-flash-next-q4" && !exists(models + "/qwen3.8-flash-next-q4"));
  std::string cmd = "rm -rf '" + home + "'";
  CHECK(std::system(cmd.c_str()) == 0);
  unsetenv("KAPITSA_HOME");
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
      CHECK(v->repo == (mixed ? "nace-ai/Qwen3.5-9B-MLX" : "nace-ai/Qwen3.5-9B-MLX-4bit"));
      CHECK(v->revision == (mixed ? "ff7b61b8c655186cdd50ddc4f485d3d5b93e1af4" : "867adaa1c6d84d1fd9b4f37f43836019c877ba59"));
      CHECK(v->download_gb == (mixed ? 9.30 : 9.05));
      CHECK(v->quant == (mixed ? "MLX affine mixed 4/5-bit (Nace Q4)" : "MLX affine 4-bit, group 64 (Nace)"));
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
  for (const char* id : {"nace-ai/Qwen3.8-Flash-Next-MLX/Q4", "nace-ai/Qwen3.5-4B-MLX/Q5",
                         "nace-ai/Qwen3.5-9B-MLX/Q6", "nace-ai/Qwen3.5-9B-MLX@main",
                         "nace-ai/Qwen3.5-9B-MLX/Q5@0123456789abcdef0123456789abcdef01234567",
                         // the Q5 variant's own revision is not one of the combined repo's
                         "nace-ai/Qwen3.5-9B-MLX/Q5@bbddf39f41d3e0f682f2293351c67a9616d5941d"}) {
    bool refused = false;
    try {
      resolve(id);
    } catch (const Error&) {
      refused = true;
    }
    if (!refused) std::fprintf(stderr, "not refused: %s\n", id);
    CHECK(refused);
  }
  renamed_variants();
  renamed_dirs();
  std::puts("catalog_profile_test: 24 device fixtures PASS; aliases, pins, companions and other models unchanged");
}
