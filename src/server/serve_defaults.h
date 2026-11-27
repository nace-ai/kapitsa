#pragma once
// Bare `kapitsa serve` and the agent's Homebrew-service hint read the same argument file. Explicit command lines
// stay self-contained; this is only a source of arguments for the existing serve parser, not another option schema.
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ni {

inline std::vector<std::string> serve_defaults(const std::string& home) {
  const std::string path = home + "/serve.json";
  std::error_code ec;
  const auto status = std::filesystem::status(path, ec);
  if (status.type() == std::filesystem::file_type::not_found) return {};
  if (ec || !std::filesystem::is_regular_file(status))
    throw std::runtime_error("Can't read " + path + " as a regular file.");
  std::ifstream f(path);
  if (!f) throw std::runtime_error("Can't read " + path + ".");
  const auto config = nlohmann::json::parse(f, nullptr, false);
  if (config.is_discarded() || !config.is_array())
    throw std::runtime_error(path + " must contain a JSON array of argument strings.");
  std::vector<std::string> args;
  for (const auto& value : config) {
    if (!value.is_string()) throw std::runtime_error(path + " must contain only argument strings.");
    std::string a = value.get<std::string>();
    if (a.find('\0') != std::string::npos) throw std::runtime_error(path + " arguments must not contain NUL bytes.");
    args.push_back(std::move(a));
  }
  return args;
}

}  // namespace ni
