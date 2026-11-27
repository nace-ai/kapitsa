#include "bench/bench.h"
#include "vision/encoder.h"
#include "util/bf16.h"
#include "util/log.h"
#include <nlohmann/json.hpp>
#include <sys/resource.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

using namespace ni;
using namespace ni::vision;

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  bench::GpuLock lock;
  if (!bench::GpuLock::held_by_caller() && !lock.acquire(nullptr, 0)) return 75;
  try {
    const std::filesystem::path dir(argv[2]);
    if (!std::filesystem::create_directory(dir)) throw std::runtime_error("output directory already exists");
    auto enc = open_encoder(argv[1]);
    if (!enc) throw std::runtime_error("no encoder");
    auto& gpu = Gpu::get();
    nlohmann::json report = {{"model", argv[1]}, {"encoder", enc->describe()}, {"runs", nlohmann::json::array()}};
    std::vector<std::vector<uint8_t>> first;
    for (int reload = 0; reload < 2; ++reload) {
      if (enc->loaded()) throw std::runtime_error("encoder did not unload");
      const double start = now_ms();
      enc->load();
      const double load_ms = now_ms() - start;
      if (!enc->loaded()) throw std::runtime_error("encoder did not load");
      int case_id = 0;
      for (const auto [width, height] : {std::pair{64, 64}, std::pair{224, 224}, std::pair{384, 256}}) {
        Image im;
        im.width = width; im.height = height; im.rgb.resize(size_t(width) * height * 3);
        for (size_t i = 0; i < im.rgb.size(); ++i) im.rgb[i] = uint8_t((i * 71 + (i / 17) * 29 + 13) % 256);
        const auto prepared = enc->prepare(im);
        if (prepared.tokens <= 0) throw std::runtime_error("empty prepared image");
        const size_t bytes = size_t(prepared.tokens) * enc->out_dim() * 2;
        Buffer storage = gpu.alloc(bytes + 32, "guarded vision output");
        memset(storage.ptr(), 0xa5, bytes + 32);
        const double encode_start = now_ms();
        enc->encode(prepared, storage.view(16, bytes));
        const double encode_ms = now_ms() - encode_start;
        const uint8_t* data = storage.as<uint8_t>();
        for (size_t i = 0; i < 16; ++i)
          if (data[i] != 0xa5 || data[16 + bytes + i] != 0xa5) throw std::runtime_error("output guard changed");
        const uint16_t* values = reinterpret_cast<const uint16_t*>(data + 16);
        bool nonzero = false;
        for (size_t i = 0; i < bytes / 2; ++i) {
          const float v = bf16_to_f32(values[i]);
          if (!std::isfinite(v)) throw std::runtime_error("non-finite embedding");
          nonzero |= v != 0;
        }
        if (!nonzero) throw std::runtime_error("all-zero embedding");
        std::vector<uint8_t> output(data + 16, data + 16 + bytes);
        if (!reload) first.push_back(output);
        else if (output != first.at(size_t(case_id))) throw std::runtime_error("reload output mismatch");
        const std::string name = "reload-" + std::to_string(reload) + "-case-" + std::to_string(case_id) + ".bf16";
        std::ofstream f(dir / name, std::ios::binary);
        f.write(reinterpret_cast<const char*>(output.data()), output.size());
        if (!f) throw std::runtime_error("output write failed");
        struct rusage ru{}; getrusage(RUSAGE_SELF, &ru);
        report["runs"].push_back({{"reload", reload}, {"case", case_id}, {"width", width}, {"height", height},
          {"prepared_width", prepared.width}, {"prepared_height", prepared.height}, {"tokens", prepared.tokens},
          {"bytes", bytes}, {"file", name}, {"guards", true}, {"finite_nonzero", true},
          {"load_ms", load_ms}, {"encode_ms", encode_ms}, {"peak_rss_bytes", ru.ru_maxrss}});
        gpu.release(storage);
        ++case_id;
      }
      enc->unload();
    }
    report["reload_bytes_equal"] = true;
    report["scope"] = "complete image encoder; not language-model logits or controlled performance timing";
    std::ofstream(dir / "report.json") << report.dump(2) << '\n';
    printf("PASS: 6 complete encoder outputs, guarded finite nonzero data; exact reload equality\n");
  } catch (const std::exception& e) {
    fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
  }
}
