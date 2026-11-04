// A local instance's control plane. Independent of the engine and the terminal:
// CPU tests use the same discovery, requests and event replay as the app and CLI.
#pragma once
#include "engine/chat.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ni::runtime {
using json = nlohmann::ordered_json;
inline constexpr int kProtocol = 1;

bool valid_id(const std::string& id);
std::string new_id();

// Records contain a private control credential. Never print them as status.
std::vector<json> discover(const std::string& directory);
json call(const json& record, const std::string& method, const std::string& path,
          const json& body = json::object());

class Instance {
 public:
  Instance(const std::string& directory, const std::string& id, const std::string& name,
           const std::string& api_url, const std::string& model, const std::string& version,
           std::function<void()> stop);
  ~Instance();
  Instance(const Instance&) = delete;
  Instance& operator=(const Instance&) = delete;

  void update(const json& fields);
  void details(std::function<json()> read);
  void event(const std::string& type, const json& data);
  std::string begin_request(const std::string& client, std::shared_ptr<LiveDecode> live);
  void finish_request(const std::string& id, const ChatResult& result);
  json snapshot();
  const json& record() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The observer owns no engine and must not keep an instance alive after shutdown.
void observe(const std::shared_ptr<Instance>& instance);
std::shared_ptr<Instance> observed();
void log_event(int level, const char* message);
}  // namespace ni::runtime
