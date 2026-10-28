#ifndef KAPITSA_SDK_H
#define KAPITSA_SDK_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define KAPITSA_ABI_VERSION 1
#if defined(__GNUC__)
#define KAPITSA_API __attribute__((visibility("default")))
#else
#define KAPITSA_API
#endif
typedef struct kapitsa_engine kapitsa_engine;
typedef struct kapitsa_request kapitsa_request;
typedef enum kapitsa_status {
  KAPITSA_OK = 0, KAPITSA_INVALID_ARGUMENT = 1, KAPITSA_BUSY = 2,
  KAPITSA_CANCELLED = 3, KAPITSA_ERROR = 4
} kapitsa_status;
/* UTF-8 JSON valid only during callback. Return 0 to request cancellation.
 * Incomplete UTF-8 is replaced with U+FFFD, as in the HTTP API.
 * Called synchronously on the thread running kapitsa_generate. Do not unwind
 * across C or destroy/reenter this request from its callback. */
typedef int (*kapitsa_delta_callback)(const char* json, size_t length, void* user);
KAPITSA_API uint32_t kapitsa_abi_version(void);
/* Options: model_dir (required), context, slots, prefill_chunk, spec_depth,
 * mem_budget_gb, dflash, disk_cache, gpu_lock_timeout_ms. No HTTP, downloads or child processes.
 * gpu_lock_timeout_ms: integer 0..2147483647; default 0 returns BUSY immediately on contention.
 * Positive values wait up to that many ms for the lock, then return BUSY if still unavailable.
 * This bounds lock waiting only; model loading blocks separately. Other lock failures return ERROR.
 * The lock is held while the engine or retained requests exist, even when idle.
 * One live SDK engine per process; concurrent requests use its native slots.
 * Creating a second live engine returns BUSY immediately regardless of the lock timeout.
 * Synchronous model-loading and Metal initialization failures return ERROR with a diagnostic.
 * Failed construction releases its allocations and lock, permitting another create attempt.
 * Fatal errors during generation or in independent background services can still terminate
 * the host process; applications needing crash isolation should use a worker process.
 * Errors/output strings are owned by the caller; free with kapitsa_string_free.
 * Output pointer arguments must be non-null; the returned handle/error may be null.
 * They are initialized on entry. Never pass a stale handle. */
KAPITSA_API kapitsa_status kapitsa_engine_create(const char* options_json,
                                                kapitsa_engine** out, char** error);
/* Existing requests retain the engine, even after this handle is released.
 * Destroy must not race a request_create on the same handle. */
KAPITSA_API void kapitsa_engine_destroy(kapitsa_engine* engine);
/* Request: messages (required), tools, template_kwargs, max_tokens,
 * temperature, top_p, top_k, seed, stop. Unknown fields are rejected.
 * Requests are one-shot. Distinct requests may generate on different threads.
 * Destroy must not race generate or cancel; cancellation does not wait for completion. */
KAPITSA_API kapitsa_status kapitsa_request_create(kapitsa_engine* engine,
                                                 const char* request_json,
                                                 kapitsa_request** out, char** error);
/* Blocking call. callback may be null. Result/error outputs must be distinct and non-null.
 * Any returned strings must be freed, including a partial result returned with an error.
 * Pre-cancelled requests return CANCELLED with no result; later cancellations may return
 * a partial result. A second call returns BUSY. Once terminal, cancellation has no effect.
 * Result JSON: content, reasoning, tool_calls, finish_reason, error, prompt_tokens,
 * cached_tokens, completion_tokens, prefill_ms, decode_ms. Tool-call arguments are JSON text.
 * Invalid UTF-8 is replaced in both deltas and the final result. */
KAPITSA_API kapitsa_status kapitsa_generate(kapitsa_request* request,
                                           kapitsa_delta_callback callback, void* user,
                                           char** result_json, char** error);
/* Thread-safe with generate; cancellation is cooperative at engine boundaries. */
KAPITSA_API void kapitsa_request_cancel(kapitsa_request* request);
KAPITSA_API void kapitsa_request_destroy(kapitsa_request* request);
KAPITSA_API void kapitsa_string_free(char* string);
#ifdef __cplusplus
}
#endif
#endif
