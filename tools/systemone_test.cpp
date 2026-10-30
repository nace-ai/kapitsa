#include "systemone/systemone.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

void write_head(const std::filesystem::path& path, uint32_t version, uint32_t hidden, uint32_t dim,
                const std::vector<float>& values) {
  std::ofstream file(path, std::ios::binary);
  file.write("KPS1", 4);
  for (uint32_t value : {version, hidden, dim}) file.write(reinterpret_cast<const char*>(&value), 4);
  file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
  require(bool(file), "write head");
}

void rejected(const std::filesystem::path& path) {
  bool failed = false;
  try { ni::systemone::PointerHead::load(path.string()); }
  catch (const std::runtime_error&) { failed = true; }
  require(failed, "invalid head accepted");
}
}

int main() {
  const auto path = std::filesystem::temp_directory_path() / ("kapitsa-systemone-head-" + std::to_string(getpid()));
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::error_code ec; std::filesystem::remove(path, ec); }
  } cleanup{path};
  try {
    write_head(path, 2, 3, 2, {1, -2, .5f, 4, 0, 3, 2});
    auto readout = ni::systemone::PointerHead::load(path.string());
    require(readout.is_readout() && readout.hidden() == 3 && readout.dim() == 2, "readout shape/kind");
    const float h[] = {2, 1, -2};
    std::vector<float> logits;
    readout.logits(h, {nullptr, nullptr}, &logits);
    require(logits.size() == 2 && logits[0] == -.5f && logits[1] == 1, "readout formula");
    readout.logits(h, {nullptr}, &logits);
    require(logits.size() == 1 && logits[0] == -.5f, "first N readout outputs");
    for (size_t n : {size_t(0), size_t(3)}) {
      bool failed = false;
      try { readout.logits(h, std::vector<const float*>(n, nullptr), &logits); }
      catch (const std::runtime_error&) { failed = true; }
      require(failed, "invalid option count accepted");
    }

    write_head(path, 1, 2, 1, {1, 2, .5f, 3, 4, -1, 2});
    auto projected = ni::systemone::PointerHead::load(path.string());
    const float decide[] = {2, 1}, option[] = {1, 2};
    projected.logits(decide, {option}, &logits);
    require(!projected.is_readout() && logits.size() == 1 && logits[0] == 22.5f, "projected formula");

    for (uint32_t version : {0u, 3u}) { write_head(path, version, 1, 1, {1, 1}); rejected(path); }
    write_head(path, 2, 0, 1, {1}); rejected(path);
    write_head(path, 2, 1, 256, {}); rejected(path);
    write_head(path, 2, 1, 1, {1}); rejected(path);
    write_head(path, 2, 1, 1, {1, 1, 1}); rejected(path);
    for (float temperature : {0.f, -1.f}) {
      write_head(path, 2, 1, 1, {1, temperature});
      ni::systemone::PointerHead::load(path.string()).logits(h, {nullptr}, &logits);
      require(logits.size() == 1 && logits[0] == (temperature == 0 ? 2.f : -2.f), "readout temperature");
      write_head(path, 1, 1, 1, {1, 0, 1, 0, temperature}); rejected(path);
    }
    write_head(path, 2, 1, 1, {std::numeric_limits<float>::infinity(), 1}); rejected(path);
    write_head(path, 2, 1, 1, {1, std::numeric_limits<float>::quiet_NaN()}); rejected(path);
    // option count against the head's width is a 400 decided before any prefill (tools/systemone.cpp score_max)
    ni::systemone::Encoded enc;
    enc.branches.resize(2);
    enc.branches[0].group.options = {0, 1};
    enc.branches[1].group.options = {0, 1, 2};
    ni::systemone::check_options(enc, 3);
    bool bad = false;
    try { ni::systemone::check_options(enc, 2); }
    catch (const ni::systemone::BadRequest& e) { bad = std::string(e.what()).find("1 to 2 options") != std::string::npos; }
    require(bad, "more options than the head's width is a BadRequest");
    std::cout << "System One projected/readout head checks passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
