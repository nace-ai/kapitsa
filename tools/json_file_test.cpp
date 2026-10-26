// CPU test of the parsed-JSON cache (src/util/json_file.h): no GPU, no model. A synthetic config.json the size of
// Nemotron 3.5 Super-VL's (8 MB, mostly quantization_config's per-layer lists) is written to a temporary folder: it
// reads back as nlohmann's own parse of the file, a second read is the same value without a parse, and 2,000 reads (what
// kapitsa doctor's plans ask of one model) take well under a second. A file written again is read again; a missing,
// empty or broken file is null; files past the cache's size are parsed again, the same; threads reading at once agree.
//   json_file_test
#include "util/json_file.h"

#include <nlohmann/json.hpp>

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace ni;
using json = nlohmann::json;

namespace {

int g_fail = 0;
#define CHECK(c, ...)                                      \
  do {                                                     \
    if (!(c)) {                                            \
      ++g_fail;                                            \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__);                        \
      fprintf(stderr, "\n");                               \
    }                                                      \
  } while (0)

double ms_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

void write(const std::string& path, const std::string& text) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f << text;
}

// a Nemotron-H config: the text model's fields under llm_config, and quantized_layers / config_groups entries for each
// of 88 layers' 512 experts' projections
std::string big_config(int tag) {
  json q = json::object(), groups = json::object();
  for (int l = 0; l < 88; ++l)
    for (int e = 0; e < 512; ++e)
      for (const char* p : {"up_proj", "down_proj"}) {
        const std::string n =
            "language_model.backbone.layers." + std::to_string(l) + ".mixer.experts." + std::to_string(e) + "." + p;
        q[n] = {{"quant_algo", "NVFP4"}, {"group_size", 16}};
        if (e % 8 == 0) groups["group_" + std::to_string(l * 64 + e / 8)] = {{"targets", {n}}, {"num_bits", 4}};
      }
  json j = {{"architectures", {"NemotronH_Nano_VL_V2"}},
            {"llm_config", {{"model_type", "nemotron_h"}, {"hidden_size", 4096}, {"num_hidden_layers", 88}, {"tag", tag}}},
            {"quantization_config", {{"quant_method", "modelopt"}, {"quantized_layers", q}, {"config_groups", groups}}}};
  return j.dump(2);
}

}  // namespace

int main() {
  char tmpl[] = "/tmp/json_file_test.XXXXXX";
  const char* d = mkdtemp(tmpl);
  if (!d) {
    perror("mkdtemp");
    return 1;
  }
  const std::string dir = d, path = dir + "/config.json";

  // missing, empty, broken: null
  CHECK(json_file(dir + "/missing.json")->is_null(), "a missing file is not null");
  write(dir + "/empty.json", "");
  CHECK(json_file(dir + "/empty.json")->is_null(), "an empty file is not null");
  write(dir + "/broken.json", "{\"model_type\": \"nemotron_h\",");
  CHECK(json_file(dir + "/broken.json")->is_null(), "a broken file is not null");
  CHECK(json_file(dir)->is_null(), "a folder is not null");

  // the big file: as nlohmann parses it, then the same value without a parse
  write(path, big_config(1));
  std::ifstream f(path);
  const json want = json::parse(f);
  auto t = std::chrono::steady_clock::now();
  const auto a = json_file(path);
  const double first = ms_since(t);
  CHECK(*a == want, "the cached value differs from nlohmann's parse");
  CHECK((*a)["llm_config"]["model_type"] == "nemotron_h", "llm_config.model_type");
  t = std::chrono::steady_clock::now();
  const auto b = json_file(path);
  const double second = ms_since(t);
  CHECK(a.get() == b.get(), "a second read parsed the file again");
  t = std::chrono::steady_clock::now();
  for (int i = 0; i < 2000; ++i)
    if ((*json_file(path))["llm_config"]["tag"] != 1) CHECK(false, "read %d: the wrong value", i);
  const double many = ms_since(t);
  CHECK(many < 1000, "2000 reads took %.0f ms", many);
  printf("%zu-byte config: first read %.1f ms, second %.3f ms, 2000 reads %.1f ms\n", want.dump().size(), first, second,
         many);

  // written again (another size): read again; the old value stays valid while held
  write(path, big_config(22));
  const auto c = json_file(path);
  CHECK(c.get() != a.get() && (*c)["llm_config"]["tag"] == 22, "a rewritten file kept its old value");
  CHECK((*a)["llm_config"]["tag"] == 1, "a held value changed");
  // the same size, another modification time: read again
  write(dir + "/small.json", "{\"v\": 1}");
  CHECK((*json_file(dir + "/small.json"))["v"] == 1, "small.json v 1");
  usleep(20000);
  write(dir + "/small.json", "{\"v\": 2}");
  CHECK((*json_file(dir + "/small.json"))["v"] == 2, "a rewrite of the same size kept its old value");

  // more files than the cache keeps: the first parsed again, the same value
  for (int i = 0; i < 12; ++i) {
    const std::string p = dir + "/f" + std::to_string(i) + ".json";
    write(p, "{\"i\": " + std::to_string(i) + "}");
    CHECK((*json_file(p))["i"] == i, "f%d", i);
  }
  const auto e = json_file(path);
  CHECK(*e == *c, "the big file read back after eviction differs");

  // threads reading at once agree
  std::vector<std::thread> th;
  std::vector<int> bad(8, 0);
  for (int k = 0; k < 8; ++k)
    th.emplace_back([&, k] {
      for (int i = 0; i < 200; ++i) {
        const std::string p = i % 2 ? path : dir + "/f" + std::to_string((k + i) % 12) + ".json";
        const auto v = json_file(p);
        if (i % 2 ? (*v)["llm_config"]["tag"] != 22 : (*v)["i"] != (k + i) % 12) ++bad[size_t(k)];
      }
    });
  for (auto& x : th) x.join();
  for (int k = 0; k < 8; ++k) CHECK(bad[size_t(k)] == 0, "thread %d: %d wrong values", k, bad[size_t(k)]);

  const std::string rm = "rm -rf '" + dir + "'";
  if (system(rm.c_str()) != 0) fprintf(stderr, "could not remove %s\n", dir.c_str());
  if (g_fail) {
    fprintf(stderr, "%d check(s) failed\n", g_fail);
    return 1;
  }
  printf("json_file_test: all checks passed\n");
  return 0;
}
