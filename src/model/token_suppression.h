#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ni {

struct TokenSuppressionMask {
  std::vector<uint32_t> words;
  size_t count = 0;
};

inline TokenSuppressionMask token_suppression_from_ids(const std::vector<int64_t>& ids, int vocab,
                                                       const std::vector<int>& protected_ids) {
  if (vocab <= 0 || ids.empty())
    throw std::runtime_error("token suppression: expected a nonempty list of token IDs");
  std::vector<uint32_t> words((size_t(vocab) + 31) / 32, 0xffffffffu);
  for (int64_t raw : ids) {
    if (raw < 0 || raw >= vocab)
      throw std::runtime_error("token suppression: token ID is out of range for the model vocabulary");
    const int t = int(raw);
    if (std::find(protected_ids.begin(), protected_ids.end(), t) != protected_ids.end())
      throw std::runtime_error("token suppression: cannot suppress a special or stop token");
    const uint32_t bit = 1u << (t % 32);
    if (!(words[size_t(t) / 32] & bit)) throw std::runtime_error("token suppression: duplicate token ID");
    words[size_t(t) / 32] &= ~bit;
  }
  if (ids.size() == size_t(vocab)) throw std::runtime_error("token suppression: no allowed tokens");
  return {std::move(words), ids.size()};
}

inline TokenSuppressionMask token_suppression_mask(const nlohmann::json& j, int vocab,
                                                   const std::vector<int>& protected_ids) {
  const nlohmann::json* arr = nullptr;
  if (j.is_array()) {
    arr = &j;
  } else if (j.is_object() && j.contains("target_token_ids") && j["target_token_ids"].is_array()) {
    arr = &j["target_token_ids"];
    if (j.contains("n_target") && (!j["n_target"].is_number_integer() || j["n_target"] != arr->size()))
      throw std::runtime_error("token suppression: n_target does not match target_token_ids");
  } else {
    throw std::runtime_error("token suppression: expected a JSON array or object with target_token_ids");
  }
  if (arr->empty()) throw std::runtime_error("token suppression: expected a nonempty list of token IDs");
  std::vector<int64_t> ids;
  ids.reserve(arr->size());
  for (const auto& id : *arr) {
    if (!id.is_number_integer())
      throw std::runtime_error("token suppression: token ID is not an integer in the model vocabulary");
    ids.push_back(id.get<int64_t>());
  }
  return token_suppression_from_ids(ids, vocab, protected_ids);
}

inline TokenSuppressionMask token_suppression_mask(const std::string& content, int vocab,
                                                   const std::vector<int>& protected_ids) {
  size_t pos = 0;
  while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) ++pos;
  if (pos == content.size()) throw std::runtime_error("token suppression: mask file is empty");
  if (content[pos] == '[' || content[pos] == '{') {
    const auto j = nlohmann::json::parse(content);
    return token_suppression_mask(j, vocab, protected_ids);
  }
  std::vector<int64_t> ids;
  std::istringstream in(content);
  std::string line;
  while (std::getline(in, line)) {
    const size_t hash = line.find('#');
    std::string_view sv(line.data(), hash == std::string::npos ? line.size() : hash);
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front()))) sv.remove_prefix(1);
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) sv.remove_suffix(1);
    if (sv.empty()) continue;
    int64_t v = 0;
    const auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), v);
    if (ec != std::errc{} || ptr != sv.data() + sv.size())
      throw std::runtime_error("token suppression: invalid integer line in mask file");
    ids.push_back(v);
  }
  return token_suppression_from_ids(ids, vocab, protected_ids);
}

}
