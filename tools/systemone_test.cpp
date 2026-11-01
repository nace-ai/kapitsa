#include "systemone/systemone.h"
#include "systemone_max_dump.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
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

void dump_roundtrip(const std::filesystem::path& head_path) {
  using namespace ni::systemone;
  char pattern[] = "/tmp/kapitsa-max-dump-test-XXXXXX";
  const char* made = mkdtemp(pattern);
  require(made != nullptr, "create dump fixture root");
  const std::filesystem::path root(made);
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::error_code ec; std::filesystem::remove_all(path, ec); }
  } cleanup{root};
  write_head(head_path, 2, 3, 3, {1, -2, .5f, 4, 0, 3, -1, 2, .25f, 2});
  auto head = PointerHead::load(head_path.string());
  const float hidden[] = {2, 1, -2}, norm[] = {1, .5f, 2};
  std::vector<float> normalized;
  final_norm_row(hidden, norm, 3, 1e-6f, &normalized);
  Encoded enc;
  enc.meta = {{"two", "choice", {"yes", "no"}, {}}, {"three", "choice", {"a", "b", "c"}, {}}};
  enc.branches = {{{11, 12, 13}, {2, {0, 1}}}, {{21, 22, 23, 24}, {3, {0, 1, 2}}}};
  std::vector<std::vector<float>> groups(2);
  for (size_t i = 0; i < groups.size(); ++i) {
    head.logits(normalized.data(), std::vector<const float*>(enc.meta[i].keys.size(), nullptr), &groups[i]);
  }
  const auto answers = format_answers(groups, enc.meta);
  const auto unchanged = groups;
  MaxDump dump(root / "complete");
  dump.begin_request(enc);
  for (const auto& group : groups) dump.append(group);
  dump.begin_request(enc);
  for (const auto& group : groups) dump.append(group);
  dump.finish();
  require(groups == unchanged && format_answers(groups, enc.meta) == answers, "dump changed stock answers");
  std::ifstream mf(root / "complete/manifest.json");
  const auto manifest = ojson::parse(mf);
  require(manifest.at("schema_version") == 1 && manifest.at("complete") == true &&
          manifest.at("payload_bytes") == 40 && manifest.at("rows").size() == 4, "complete dump manifest");
  std::ifstream payload(root / "complete/logits.f32", std::ios::binary);
  size_t offset = 0;
  for (size_t request = 0; request < 2; ++request) {
    for (size_t i = 0; i < groups.size(); ++i) {
      const auto& row = manifest.at("rows").at(request * 2 + i);
      const auto& meta = enc.meta[i];
      require(row.at("request_index") == request && row.at("branch_index") == i && row.at("byte_offset") == offset &&
              row.at("width") == groups[i].size() && row.at("tokens") == ojson(enc.branches[i].tokens) &&
              row.at("decide") == enc.branches[i].group.decide && row.at("option_order") == ojson(meta.keys) &&
              row.at("question") == ojson{{"id", meta.id}, {"type", meta.type}, {"keys", meta.keys}, {"legend", meta.legend}},
              "dump row identity/order");
      std::vector<float> restored(groups[i].size());
      const size_t bytes = restored.size() * sizeof(float);
      payload.read(reinterpret_cast<char*>(restored.data()), std::streamsize(bytes));
      require(bool(payload) && std::memcmp(restored.data(), groups[i].data(), bytes) == 0, "stock readout byte roundtrip");
      offset += bytes;
    }
  }
  require(payload.peek() == std::char_traits<char>::eof(), "trailing dump bytes");
  auto rejects = [&](auto action, const char* message) {
    bool failed = false;
    try { action(); } catch (const std::exception&) { failed = true; }
    require(failed, message);
  };
  rejects([&] { MaxDump existing(root / "complete"); }, "dump overwrote existing directory");
  MaxDump partial(root / "partial");
  partial.begin_request(enc);
  rejects([&] { partial.append({1}); }, "dump accepted wrong active width");
  rejects([&] { partial.append({1, std::numeric_limits<float>::infinity()}); }, "dump accepted nonfinite");
  partial.append(groups[0]);
  rejects([&] { partial.begin_request(enc); }, "dump accepted unfinished request");
  rejects([&] { partial.finish(); }, "dump published incomplete request");
  require(!std::filesystem::exists(root / "partial/manifest.json"), "partial dump marked complete");
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
    dump_roundtrip(path);
    std::cout << "System One projected/readout head and Max dump checks passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
