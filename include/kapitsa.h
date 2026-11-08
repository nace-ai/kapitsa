// Kapitsa C API — a minimal interface for embedding Kapitsa inference in C, Objective-C and Swift programs.
//
// This header is the public contract. The implementation (src/lib/kapitsa_lib.cpp) wraps the Engine. A Swift package
// (Package.swift at the repo root) exposes it as a Swift module. See docs/c-api.md for the design.
//
// Thread safety: a kapitsa_ctx is not thread-safe. Create one per thread, or serialise access. The callback from
// kapitsa_generate may be called from any thread; the caller must handle that.
//
// Memory: every function that returns a char* returns a heap-allocated string that the caller must free with
// kapitsa_free_string(). The context is freed with kapitsa_free().
#ifndef KAPITSA_H
#define KAPITSA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- types ----

// An opaque inference context. Holds the loaded model, the KV cache and the engine state.
typedef struct kapitsa_ctx kapitsa_ctx;

// Status codes returned by every fallible function.
typedef enum {
  KAPITSA_OK = 0,
  KAPITSA_ERROR_INVALID_ARG = 1,   // a NULL pointer or out-of-range value
  KAPITSA_ERROR_IO = 2,            // model file not found or unreadable
  KAPITSA_ERROR_OOM = 3,           // not enough memory (RAM or GPU)
  KAPITSA_ERROR_MODEL = 4,         // unsupported model format or architecture
  KAPITSA_ERROR_CANCELLED = 5,     // generation stopped by kapitsa_cancel() or the callback
  KAPITSA_ERROR_INTERNAL = 6,      // an unexpected engine error
} kapitsa_status;

// Options for loading a model. Pass to kapitsa_load().
typedef struct {
  const char* model_path;   // path to the model directory (required)
  int context_length;       // context window in tokens; 0 = the model's default
  int threads;              // CPU threads; 0 = automatic
  size_t mem_budget_bytes;  // GPU memory budget in bytes; 0 = automatic
} kapitsa_load_opts;

// Options for a generation request.
typedef struct {
  const char* prompt;       // the user prompt (required; UTF-8)
  float temperature;        // sampling temperature; 0.0 = greedy
  int max_tokens;           // maximum tokens to generate; 0 = model default
  const char* system;       // optional system prompt; NULL = none
} kapitsa_generate_opts;

// Callback for streaming tokens. Return true to continue, false to stop.
// `token` is a UTF-8 fragment (one or more characters); `userdata` is the pointer passed to kapitsa_generate().
typedef bool (*kapitsa_token_cb)(const char* token, size_t token_len, void* userdata);

// ---- lifecycle ----

// Initialise the library. Call once before any other function.
// Returns KAPITSA_OK on success. Safe to call more than once (subsequent calls are no-ops).
kapitsa_status kapitsa_init(void);

// Load a model and return a ready-to-use context. The context holds the model on the GPU.
// On failure *ctx is set to NULL and the return value says why.
kapitsa_status kapitsa_load(const kapitsa_load_opts* opts, kapitsa_ctx** ctx);

// ---- generation ----

// Generate a completion. Calls `cb` for each token. Blocks until generation finishes, the callback returns false,
// or kapitsa_cancel() is called from another thread.
kapitsa_status kapitsa_generate(kapitsa_ctx* ctx, const kapitsa_generate_opts* opts, kapitsa_token_cb cb, void* userdata);

// Cancel a running generation from any thread. The kapitsa_generate() call returns KAPITSA_ERROR_CANCELLED.
kapitsa_status kapitsa_cancel(kapitsa_ctx* ctx);

// ---- cleanup ----

// Free a context and release its GPU memory. NULL is safe.
void kapitsa_free(kapitsa_ctx* ctx);

// Free a string returned by this API. NULL is safe.
void kapitsa_free_string(char* s);

// ---- info ----

// The library version string (e.g. "0.2.31"). The returned pointer is static; do not free it.
const char* kapitsa_version(void);

// The last error message for `ctx`, or a global message when ctx is NULL. Returns "" when there is none.
// The returned pointer is valid until the next API call on the same ctx.
const char* kapitsa_last_error(kapitsa_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif // KAPITSA_H
