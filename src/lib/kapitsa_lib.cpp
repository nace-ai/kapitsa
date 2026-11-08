// Kapitsa C API — stub implementation.
//
// This file provides the symbols declared in include/kapitsa.h. The current implementation returns stub values so that
// the header's contract can be validated by a linker and by Swift's C importer. The real implementation (wrapping the
// Engine, the model loader and the KV cache) is a follow-up.
#include "../../include/kapitsa.h"

#include <cstring>
#include <string>

#ifndef KAPITSA_VERSION
#define KAPITSA_VERSION "0.2.31"
#endif

// A thread-local last-error message for the global (NULL ctx) case.
static thread_local std::string g_last_error;

struct kapitsa_ctx {
  std::string last_error;
  bool cancelled = false;
};

// ---- lifecycle ----

kapitsa_status kapitsa_init(void) {
  // TODO: one-time Metal device setup, thread pool, etc.
  return KAPITSA_OK;
}

kapitsa_status kapitsa_load(const kapitsa_load_opts* opts, kapitsa_ctx** ctx) {
  if (!opts || !ctx) return KAPITSA_ERROR_INVALID_ARG;
  if (!opts->model_path || !*opts->model_path) {
    g_last_error = "model_path is required";
    if (ctx) *ctx = nullptr;
    return KAPITSA_ERROR_INVALID_ARG;
  }
  // TODO: load the model via Engine, apply context_length / mem_budget_bytes
  *ctx = new kapitsa_ctx;
  (*ctx)->last_error = "stub: model not actually loaded (implementation pending)";
  return KAPITSA_OK;
}

// ---- generation ----

kapitsa_status kapitsa_generate(kapitsa_ctx* ctx, const kapitsa_generate_opts* opts, kapitsa_token_cb cb,
                                void* userdata) {
  if (!ctx || !opts || !opts->prompt) return KAPITSA_ERROR_INVALID_ARG;
  if (ctx->cancelled) {
    ctx->cancelled = false;
    return KAPITSA_ERROR_CANCELLED;
  }
  // TODO: build a ChatRequest from opts, run the engine, stream tokens through cb
  ctx->last_error = "stub: generation not implemented";
  return KAPITSA_ERROR_INTERNAL;
}

kapitsa_status kapitsa_cancel(kapitsa_ctx* ctx) {
  if (!ctx) return KAPITSA_ERROR_INVALID_ARG;
  ctx->cancelled = true;
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
