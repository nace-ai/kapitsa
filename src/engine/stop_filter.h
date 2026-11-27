#pragma once

#include <algorithm>
#include <string>
#include <vector>

namespace ni {

// Parsed content only. Keep a possible stop prefix until the next delta decides it, so neither a streamed reply
// nor its accumulated result can contain the beginning of a matched stop string.
class StopFilter {
 public:
  explicit StopFilter(const std::vector<std::string>& stops) : stops_(stops) {}

  void filter(std::string& text, bool final = false) {
    if (!matched_.empty()) {
      text.clear();
      return;
    }
    if (stops_.empty()) return;
    pending_ += text;
    size_t first = std::string::npos;
    for (const auto& stop : stops_) {
      if (stop.empty()) continue;
      const size_t at = pending_.find(stop);
      if (at < first) first = at, matched_ = stop;
    }
    if (first != std::string::npos) {
      text.assign(pending_, 0, first);
      pending_.clear();
      return;
    }
    size_t keep = 0;
    if (!final) {
      for (const auto& stop : stops_) {
        if (stop.empty()) continue;
        for (size_t n = std::min(pending_.size(), stop.size() - 1); n > keep; --n) {
          if (pending_.compare(pending_.size() - n, n, stop, 0, n) == 0) {
            keep = n;
            break;
          }
        }
      }
    }
    text.assign(pending_, 0, pending_.size() - keep);
    pending_.erase(0, pending_.size() - keep);
  }

  const std::string& matched() const { return matched_; }

 private:
  const std::vector<std::string>& stops_;
  std::string pending_, matched_;
};

}  // namespace ni
