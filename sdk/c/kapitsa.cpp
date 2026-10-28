#include "kapitsa.h"
#include "options.h"
#include <cerrno>
#include "bench/bench.h"
#include "util/log.h"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>

namespace {
std::mutex lifetime_mutex;
bool engine_live = false;
struct State {
  bool claimed = false;
  ni::bench::GpuLock gpu_lock;
  std::unique_ptr<ni::Gpu::AllocationScope> allocations;
  std::unique_ptr<ni::Engine> engine;
  ~State() {
    engine.reset();
    allocations.reset();
    gpu_lock.release();
    if (claimed) {
      std::lock_guard<std::mutex> lock(lifetime_mutex);
      engine_live = false;
    }
  }
};
char* copy(const std::string& value) {
  auto* p = static_cast<char*>(std::malloc(value.size() + 1));
  if (!p) throw std::bad_alloc();
  std::memcpy(p, value.c_str(), value.size() + 1);
  return p;
}
kapitsa_status fail(char** error, kapitsa_status status, const char* message) noexcept {
  if (error) {
    try { *error = copy(message); } catch (...) { *error = nullptr; }
  }
  return status;
}
}
struct kapitsa_engine { std::shared_ptr<State> state; };
struct kapitsa_request {
  std::shared_ptr<State> state;
  ni::ChatRequest request;
  std::atomic<bool> cancelled{false}, started{false};
};
extern "C" {
uint32_t kapitsa_abi_version(void) { return KAPITSA_ABI_VERSION; }
kapitsa_status kapitsa_engine_create(const char* json, kapitsa_engine** out, char** error) {
  if (out) *out = nullptr;
  if (error) *error = nullptr;
  if (!json || !out || !error) return fail(error, KAPITSA_INVALID_ARGUMENT, "null argument");
  try {
    auto options = kapitsa_sdk::options(ni::ojson::parse(json));
    auto state = std::make_shared<State>();
    {
      std::lock_guard<std::mutex> lock(lifetime_mutex);
      if (engine_live) {
        // Do not let the unclaimed State reset another engine's lifetime flag.
        state.reset();
        return fail(error, KAPITSA_BUSY, "only one live SDK engine per process is supported");
      }
      engine_live = true;
      state->claimed = true;
    }
    if (!ni::bench::GpuLock::held_by_caller() && !state->gpu_lock.acquire(nullptr, options.gpu_lock_timeout_ms))
      return fail(error, errno == EWOULDBLOCK ? KAPITSA_BUSY : KAPITSA_ERROR, "could not acquire Kapitsa GPU lock");
    state->allocations = std::make_unique<ni::Gpu::AllocationScope>();
    try {
      ni::FatalRecovery recover;
      state->engine = std::make_unique<ni::Engine>(options.engine);
    } catch (const std::exception& e) { return fail(error, KAPITSA_ERROR, e.what()); }
    *out = new kapitsa_engine{std::move(state)};
    return KAPITSA_OK;
  } catch (const ni::ojson::exception& e) { return fail(error, KAPITSA_INVALID_ARGUMENT, e.what());
  } catch (const std::invalid_argument& e) { return fail(error, KAPITSA_INVALID_ARGUMENT, e.what());
  } catch (const std::exception& e) { return fail(error, KAPITSA_ERROR, e.what());
  } catch (...) { return fail(error, KAPITSA_ERROR, "unknown native exception"); }
}
void kapitsa_engine_destroy(kapitsa_engine* engine) { delete engine; }
kapitsa_status kapitsa_request_create(kapitsa_engine* engine, const char* json, kapitsa_request** out, char** error) {
  if (out) *out = nullptr;
  if (error) *error = nullptr;
  if (!engine || !json || !out || !error) return fail(error, KAPITSA_INVALID_ARGUMENT, "null argument");
  try {
    auto r = std::make_unique<kapitsa_request>();
    r->request = kapitsa_sdk::request(ni::ojson::parse(json), engine->state->engine->request_defaults());
    r->state = engine->state;
    *out = r.release();
    return KAPITSA_OK;
  } catch (const ni::ojson::exception& e) { return fail(error, KAPITSA_INVALID_ARGUMENT, e.what());
  } catch (const std::invalid_argument& e) { return fail(error, KAPITSA_INVALID_ARGUMENT, e.what());
  } catch (const std::exception& e) { return fail(error, KAPITSA_ERROR, e.what());
  } catch (...) { return fail(error, KAPITSA_ERROR, "unknown native exception"); }
}
kapitsa_status kapitsa_generate(kapitsa_request* r, kapitsa_delta_callback callback, void* user,
                                char** result, char** error) {
  if (result) *result = nullptr;
  if (error) *error = nullptr;
  if (!r || !result || !error) return fail(error, KAPITSA_INVALID_ARGUMENT, "null argument");
  if (r->started.exchange(true)) return fail(error, KAPITSA_BUSY, "request is one-shot");
  if (r->cancelled.load()) return KAPITSA_CANCELLED;
  try {
    const auto cancelled = [r] { return r->cancelled.load(); };
    auto response = r->state->engine->chat(r->request, [&](const ni::ChatDelta& delta) {
      if (cancelled()) return false;
      if (!callback) return true;
      auto json = kapitsa_sdk::dump(ni::ojson{{"content", delta.content}, {"reasoning", delta.reasoning},
                            {"tool_calls", kapitsa_sdk::calls(delta.tool_calls)}});
      if (!callback(json.data(), json.size(), user)) r->cancelled.store(true);
      return !cancelled();
    }, nullptr, cancelled);
    *result = copy(kapitsa_sdk::dump(ni::ojson{{"content", response.content}, {"reasoning", response.reasoning},
      {"tool_calls", kapitsa_sdk::calls(response.tool_calls)}, {"finish_reason", response.finish_reason},
      {"error", response.error}, {"prompt_tokens", response.prompt_tokens},
      {"cached_tokens", response.cached_tokens}, {"completion_tokens", response.completion_tokens},
      {"prefill_ms", response.prefill_ms}, {"decode_ms", response.decode_ms}}));
    if (response.finish_reason == "error")
      return fail(error, response.busy ? KAPITSA_BUSY : response.invalid_request ? KAPITSA_INVALID_ARGUMENT : KAPITSA_ERROR,
                  response.error.c_str());
    return response.finish_reason == "cancelled" ? KAPITSA_CANCELLED : KAPITSA_OK;
  } catch (const std::exception& e) { return fail(error, KAPITSA_ERROR, e.what());
  } catch (...) { return fail(error, KAPITSA_ERROR, "unknown native exception"); }
}
void kapitsa_request_cancel(kapitsa_request* r) { if (r) r->cancelled.store(true); }
void kapitsa_request_destroy(kapitsa_request* r) { delete r; }
void kapitsa_string_free(char* string) { std::free(string); }
}
