// Kapitsa C API — implementation wrapping the Engine.
//
// kapitsa_load() creates an Engine with the model directory, context and memory budget from the options.
// kapitsa_generate() builds a ChatRequest, calls Engine::chat() and streams each delta's content through the
// callback. kapitsa_cancel() sets an atomic flag the engine's cancellation callback reads.
#include "../../include/kapitsa.h"

#include "engine/engine.h"
#include "util/log.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#ifndef KAPITSA_VERSION
#define KAPITSA_VERSION "0.2.31"
#endif

// A thread-local last-error message for the global (NULL ctx) case.
static thread_local std::string g_last_error;

struct kapitsa_ctx {
  std::unique_ptr<ni::Engine> engine;
  std::string last_error;
  std::atomic<bool> cancelled{false};
};

// ---- lifecycle ----

kapitsa_status kapitsa_init(void) {
  // The Engine's constructor initialises Metal and the thread pool on first use; nothing global is needed here.
  static bool done = false;
  if (done) return KAPITSA_OK;
  done = true;
  return KAPITSA_OK;
}

kapitsa_status kapitsa_load(const kapitsa_load_opts* opts, kapitsa_ctx** ctx) {
  if (!opts || !ctx) return KAPITSA_ERROR_INVALID_ARG;
  *ctx = nullptr;
  if (!opts->model_path || !*opts->model_path) {
    g_last_error = "model_path is required";
    return KAPITSA_ERROR_INVALID_ARG;
  }
  auto c = std::make_unique<kapitsa_ctx>();
  try {
    ni::EngineOptions eo;
    eo.model_dir = opts->model_path;
    if (opts->context_length > 0) eo.context = opts->context_length;
    if (opts->mem_budget_bytes > 0) eo.mem_budget_gb = double(opts->mem_budget_bytes) / (1024.0 * 1024.0 * 1024.0);
    eo.slots = 1;          // the C API drives one conversation at a time
    eo.disk_cache = "0";   // no SSD cache for the library (the caller manages state)
    c->engine = std::make_unique<ni::Engine>(eo);
  } catch (const std::exception& e) {
    g_last_error = e.what();
    // Classify the error by its message when we can
    const std::string msg = e.what();
    if (msg.find("not found") != std::string::npos || msg.find("No such file") != std::string::npos)
      return KAPITSA_ERROR_IO;
    if (msg.find("insufficient memory") != std::string::npos || msg.find("does not fit") != std::string::npos)
      return KAPITSA_ERROR_OOM;
    if (msg.find("unsupported") != std::string::npos || msg.find("model_type") != std::string::npos)
      return KAPITSA_ERROR_MODEL;
    return KAPITSA_ERROR_INTERNAL;
  }
  *ctx = c.release();
  return KAPITSA_OK;
}

// ---- generation ----

kapitsa_status kapitsa_generate(kapitsa_ctx* ctx, const kapitsa_generate_opts* opts, kapitsa_token_cb cb,
                                void* userdata) {
  if (!ctx || !ctx->engine || !opts || !opts->prompt) return KAPITSA_ERROR_INVALID_ARG;
  ctx->cancelled.store(false, std::memory_order_relaxed);
  ctx->last_error.clear();

  // Build the messages array: an optional system message and the user prompt.
  ni::ojson messages = ni::ojson::array();
  if (opts->system && *opts->system)
    messages.push_back({{"role", "system"}, {"content", opts->system}});
  messages.push_back({{"role", "user"}, {"content", opts->prompt}});

  ni::ChatRequest r;
  r.messages = std::move(messages);
  r.temperature = opts->temperature;
  if (opts->max_tokens > 0) r.max_tokens = opts->max_tokens;

  try {
    ni::ChatResult res = ctx->engine->chat(
        r,
        // on_delta: stream each text fragment (reasoning and content) through the callback
        [&](const ni::ChatDelta& d) -> bool {
          if (ctx->cancelled.load(std::memory_order_relaxed)) return false;
          if (cb) {
            if (!d.reasoning.empty() && !cb(d.reasoning.c_str(), d.reasoning.size(), userdata)) return false;
            if (!d.content.empty() && !cb(d.content.c_str(), d.content.size(), userdata)) return false;
          }
          return true;
        },
        nullptr,  // on_prompt
        // cancelled: polled while waiting for a slot or memory
        [&]() -> bool { return ctx->cancelled.load(std::memory_order_relaxed); });

    if (res.finish_reason == "cancelled") return KAPITSA_ERROR_CANCELLED;
    if (res.finish_reason == "error") {
      ctx->last_error = res.error;
      return KAPITSA_ERROR_INTERNAL;
    }
    // "stop", "length", "tool_calls" are all successful completions
    return KAPITSA_OK;
  } catch (const std::exception& e) {
    ctx->last_error = e.what();
    return KAPITSA_ERROR_INTERNAL;
  }
}

kapitsa_status kapitsa_cancel(kapitsa_ctx* ctx) {
  if (!ctx) return KAPITSA_ERROR_INVALID_ARG;
  ctx->cancelled.store(true, std::memory_order_release);
  return KAPITSA_OK;
}

// ---- cleanup ----

void kapitsa_free(kapitsa_ctx* ctx) { delete ctx; }

void kapitsa_free_string(char* s) { free(s); }

// ---- info ----

const char* kapitsa_version(void) { return KAPITSA_VERSION; }

const char* kapitsa_last_error(kapitsa_ctx* ctx) {
  if (ctx) return ctx->last_error.c_str();
  return g_last_error.c_str();
}
