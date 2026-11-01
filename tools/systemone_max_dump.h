#pragma once

#include "systemone/systemone.h"

#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace ni::systemone {

class MaxDump {
 public:
  explicit MaxDump(const std::filesystem::path& dir) : dir_(dir) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    static_assert(std::endian::native == std::endian::little);
    if (!std::filesystem::create_directory(dir_)) throw std::runtime_error("Max dump directory must be fresh");
    payload_.exceptions(std::ios::failbit | std::ios::badbit);
    payload_.open(dir_ / "logits.f32", std::ios::binary);
  }

  void begin_request(const Encoded& enc) {
    if (finished_ || next_ != rows_.size()) throw std::runtime_error("unfinished Max dump request");
    if (!enc.state.empty() || enc.branches.empty() || enc.branches.size() != enc.meta.size()) {
      throw std::runtime_error("invalid Max dump request layout");
    }
    for (size_t i = 0; i < enc.branches.size(); ++i) {
      const auto& b = enc.branches[i];
      const auto& m = enc.meta[i];
      if (b.tokens.empty() || b.group.decide < 0 || size_t(b.group.decide) >= b.tokens.size() ||
          b.group.options.empty() || b.group.options.size() != m.keys.size()) {
        throw std::runtime_error("invalid Max dump branch layout");
      }
      for (size_t j = 0; j < b.group.options.size(); ++j) {
        if (b.group.options[j] != int(j)) throw std::runtime_error("Max dump needs readout option indices");
      }
      rows_.push_back(ojson{{"request_index", requests_}, {"branch_index", i},
                            {"question", {{"id", m.id}, {"type", m.type}, {"keys", m.keys}, {"legend", m.legend}}},
                            {"option_order", m.keys}, {"tokens", b.tokens}, {"decide", b.group.decide},
                            {"width", b.group.options.size()}});
    }
    ++requests_;
  }

  void append(const std::vector<float>& group) {
    if (finished_ || next_ >= rows_.size() || group.size() != rows_[next_].at("width").get<size_t>()) {
      throw std::runtime_error("Max dump active-option width mismatch");
    }
    for (float value : group) {
      if (!std::isfinite(value)) throw std::runtime_error("nonfinite Max dump logit");
    }
    rows_[next_]["byte_offset"] = bytes_;
    const size_t n = group.size() * sizeof(float);
    payload_.write(reinterpret_cast<const char*>(group.data()), std::streamsize(n));
    bytes_ += n;
    ++next_;
  }

  void finish() {
    if (finished_ || rows_.empty() || next_ != rows_.size()) throw std::runtime_error("incomplete Max dump");
    payload_.close();
    const ojson manifest{{"schema_version", 1}, {"complete", true}, {"payload_bytes", bytes_}, {"rows", rows_}};
    std::ofstream file;
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.open(dir_ / "manifest.tmp");
    file << manifest.dump(2) << '\n';
    file.close();
    std::filesystem::rename(dir_ / "manifest.tmp", dir_ / "manifest.json");
    finished_ = true;
  }

 private:
  std::filesystem::path dir_;
  std::ofstream payload_;
  ojson rows_ = ojson::array();
  size_t next_ = 0, requests_ = 0, bytes_ = 0;
  bool finished_ = false;
};

}
