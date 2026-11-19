// Hy4-flash's Engram CPU lookup (src/model/hy4_engram.h) on a pack folder, without the GPU or the model: the rows a
// pass would read for positions pos0 .. pos0 + R - 1 of a token sequence, and how long the lookups take on a real pack.
// hy4_engram_test checks the lookup itself on a synthetic pack.
//
//   hy4_engram_check <pack_dir> <layer_id> <tokens.txt> <pos0> <R> [out.bin]
//
// tokens.txt: whitespace-separated token ids (at least pos0 + R). layer_id: the 1-based config key (2, 12, 22, 32).
// out.bin: fp32 [R, 1536] rows, then R mask bytes, then int64 [R, 16] table rows (ids(), masked rows included).
// Takes no GPU lock: it uses no GPU.
#include "model/hy4_engram.h"
#include "util/json_file.h"
#include "util/log.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace ni;

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: hy4_engram_check <pack_dir> <layer_id> <tokens.txt> <pos0> <R> [out.bin]\n");
    return 2;
  }
  const std::string dir = argv[1];
  const int layer = atoi(argv[2]), pos0 = atoi(argv[4]), R = atoi(argv[5]);
  std::vector<int32_t> toks;
  {
    std::ifstream f(argv[3]);
    for (long v; f >> v;) toks.push_back(int32_t(v));
  }
  if (pos0 < 0 || R < 1 || size_t(pos0) + size_t(R) > toks.size())
    fatal("positions %d .. %d of %zu tokens", pos0, pos0 + R - 1, toks.size());
  const auto cfg = json_file(dir + "/config.json");
  if (!cfg) fatal("%s/config.json: missing or not JSON", dir.c_str());
  const nlohmann::json& tc = cfg->contains("text_config") ? (*cfg)["text_config"] : *cfg;

  auto t0 = std::chrono::steady_clock::now();
  hy4::EngramCpu eng(dir, tc, layer);
  auto t1 = std::chrono::steady_clock::now();
  const int W = eng.width(), H = eng.heads();
  std::vector<float> out(size_t(R) * W);
  std::vector<uint8_t> mask(static_cast<size_t>(R));
  eng.rows(toks.data(), pos0, R, out.data(), mask.data());
  auto t2 = std::chrono::steady_clock::now();
  std::vector<int64_t> ids(size_t(R) * H);
  for (int r = 0; r < R; ++r) eng.ids(toks.data(), pos0 + r, ids.data() + size_t(r) * H);
  int live = 0;
  for (uint8_t m : mask) live += m;
  printf("layer %d: table %s; open %.1f ms; rows %d..%d (%d with an n-gram) in %.2f ms\n", layer,
         eng.table().describe().c_str(), std::chrono::duration<double, std::milli>(t1 - t0).count(), pos0,
         pos0 + R - 1, live, std::chrono::duration<double, std::milli>(t2 - t1).count());
  if (argc > 6) {
    std::ofstream o(argv[6], std::ios::binary);
    o.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size() * 4));
    o.write(reinterpret_cast<const char*>(mask.data()), std::streamsize(mask.size()));
    o.write(reinterpret_cast<const char*>(ids.data()), std::streamsize(ids.size() * 8));
    if (!o) fatal("cannot write %s", argv[6]);
  }
  return 0;
}
