// CPU-only ABI double: real native threads, callbacks and allocated outputs; no model/GPU.
#include "kapitsa.h"
#include <nlohmann/json.hpp>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
using json = nlohmann::ordered_json;
using namespace std::chrono_literals;
static std::atomic<int> engines{0}, requests{0}, strings{0};
struct State {
  std::atomic<bool> failed{false};
  State() { ++engines; }
  ~State() { --engines; }
};
struct kapitsa_engine { std::shared_ptr<State> state; };
struct kapitsa_request {
  std::shared_ptr<State> state;
  json input;
  std::atomic<bool> cancelled{false};
  ~kapitsa_request() { --requests; }
};
static char* copy(const std::string& s) { ++strings; return strdup(s.c_str()); }
extern "C" {
uint32_t kapitsa_abi_version() { return KAPITSA_ABI_VERSION; }
kapitsa_status kapitsa_engine_create(const char* input, kapitsa_engine** out, char** error) {
  *out = nullptr; *error = nullptr;
  std::this_thread::sleep_for(40ms);
  auto options = json::parse(input);
  if (options.value("model_dir", "") == "bad") {
    *error = copy("fixture load error"); return KAPITSA_ERROR;
  }
  if (engines) { *error = copy("only one engine"); return KAPITSA_BUSY; }
  *out = new kapitsa_engine{std::make_shared<State>()};
  return KAPITSA_OK;
}
void kapitsa_engine_destroy(kapitsa_engine* e) { std::this_thread::sleep_for(40ms); delete e; }
kapitsa_status kapitsa_request_create(kapitsa_engine* e, const char* input, kapitsa_request** out, char** error) {
  *out = nullptr; *error = nullptr;
  auto request = json::parse(input);
  if (!request.contains("messages") || request["messages"].empty()) {
    *error = copy("messages required"); return KAPITSA_INVALID_ARGUMENT;
  }
  *out = new kapitsa_request{e->state, request}; ++requests;
  return KAPITSA_OK;
}
kapitsa_status kapitsa_generate(kapitsa_request* r, kapitsa_delta_callback cb, void* user,
                               char** result, char** error) {
  *result = nullptr; *error = nullptr;
  if (r->state->failed) { *error = copy("engine unusable"); return KAPITSA_ERROR; }
  if (r->cancelled) return KAPITSA_CANCELLED;
  const auto mode = r->input["messages"][0].value("content", "");
  const auto calls = json::array({{{"id", "c1"}, {"name", "weather"}, {"arguments", "{\"city\":\"北京\"}"}}});
  std::string content;
  int count = 0;
  for (; count < 6 && !r->cancelled; ++count) {
    std::this_thread::sleep_for(10ms);
    content += "猫🐈";
    auto delta = json{{"content", "猫🐈"}, {"reasoning", "想"}, {"tool_calls", calls}}.dump();
    if (cb && !cb(delta.data(), delta.size(), user)) r->cancelled = true;
    if (mode == "deferred") { r->state->failed = true; break; }
  }
  auto reason = r->state->failed ? "error" : r->cancelled ? "cancelled" : "stop";
  auto message = r->state->failed ? "deferred pipeline error" : "";
  *result = copy(json{{"content", content}, {"reasoning", "想"}, {"tool_calls", calls},
    {"finish_reason", reason}, {"error", message}, {"prompt_tokens", 3}, {"cached_tokens", 0},
    {"completion_tokens", count}, {"prefill_ms", 1.5}, {"decode_ms", 60.0}}.dump());
  if (r->state->failed) { *error = copy(message); return KAPITSA_ERROR; }
  return r->cancelled ? KAPITSA_CANCELLED : KAPITSA_OK;
}
void kapitsa_request_cancel(kapitsa_request* r) { r->cancelled = true; }
void kapitsa_request_destroy(kapitsa_request* r) { delete r; }
void kapitsa_string_free(char* s) { if (s) { --strings; free(s); } }
int fixture_live() { return engines + requests + strings; }
}
