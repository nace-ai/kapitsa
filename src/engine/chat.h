#pragma once
#include <nlohmann/json.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ni {

using ojson = nlohmann::ordered_json;
class Grammar;
namespace vision { struct Prepared; }

// One request's progress while it runs, for a live display (`kapitsa serve`'s dashboard in a terminal): the caller
// passes it in ChatRequest::live and reads it from any thread; the engine writes it with relaxed atomic stores between
// decode cycles, never on the GPU's path.
struct LiveDecode {
  std::atomic<int> phase{0};               // 0 waiting for the engine, 1 reading the prompt, 2 writing, 3 done
  std::atomic<int> prompt{0}, cached{0};   // prompt tokens, of which reused from the cache
  std::atomic<int> read{0};                // prompt tokens read so far: the cached ones, then each prefill chunk the
                                           // GPU has finished
  std::atomic<int> tokens{0};              // tokens written
  std::atomic<int> cycles{0};              // decode cycles: a speculative cycle's verify, or a plain step
  std::atomic<int> last{0};                // tokens the last cycle wrote (its accepted drafts + 1)
  std::atomic<int> spec{0};                // 0 plain decoding, 1 MTP, 2 DFlash2, 3 LiLiCorr
};

struct ChatRequest {
  ojson messages = ojson::array();
  ojson tools = ojson::array();
  ojson template_kwargs = ojson::object(); // e.g. enable_thinking, reasoning_effort
  int max_tokens = 8192;
  float temperature = 0.6f;
  float top_p = 0.95f;
  int top_k = 20;
  uint32_t seed = 0;
  // The request's seed as Splash 1.4.0's frontend takes it (server/frontend.py:1153-1162): absent or -1 is a random seed,
  // any other integer in [-2^63, 2^64) is its 64 bits (0 included), and anything else (a float, a boolean, null, a string,
  // a larger integer) is refused. Only the Splash backend reads these; the native engines keep `seed`.
  std::optional<uint64_t> splash_seed;
  bool splash_seed_invalid = false;  // the body's seed is not such an integer: the Splash backend answers 400
  // Sampling options only the Splash backend honours (OpenAI's names; vLLM's min_p and repetition_penalty, ignore_eos). Every
  // other engine leaves them out, as before; the Splash backend sends them on, or refuses the request naming the field.
  float presence_penalty = 0.0f, frequency_penalty = 0.0f, repetition_penalty = 1.0f, min_p = 0.0f;
  bool ignore_eos = false;
  ojson cache_salt;  // the body's "cache_salt" (null: none); only the Splash backend uses it, as its cache key
  std::vector<std::string> stop;
  std::shared_ptr<LiveDecode> live;  // the request's progress for a display, or null
  // Looped models (Dory): the loop count asked for ("loops" in the body, 0: none), else derived from the reasoning
  // effort (OpenAI's reasoning_effort / reasoning.effort: minimal and low 1, medium 2, high and xhigh 4; none, which
  // only turns thinking off, the default) or Anthropic's thinking (disabled 1, a budget by its size); "" / -1: the
  // server's default
  int loops = 0;
  std::string effort;
  int thinking_budget = -1;     // Anthropic thinking.budget_tokens; 0: thinking disabled
  // Structured output (src/engine/grammar.h: a JSON schema, any JSON object, tool calls the request forces or strict
  // tools): what the reply must follow after its reasoning, or null. The engine samples each token under its mask.
  std::shared_ptr<const Grammar> grammar;
};

struct ToolCall {
  std::string id, name;
  std::string arguments; // JSON object text
};

// Tags the model writes as text (a tokenizer without <think> / tool-call tokens: Nemotron Nano 9B v2's), which the
// output parser matches in the decoded text instead of by token id.
struct TextTags {
  bool think = false;             // <think> / </think>
  std::string tc_open, tc_close;  // a tool call's tags, e.g. <TOOLCALL> / </TOOLCALL> (empty: none)
  bool json_calls = false;        // the call's body: a JSON list of {"name": ..., "arguments": ...} (Nano 9B v2's)
  bool any() const { return think || !tc_open.empty(); }
};

struct ChatDelta {
  std::string reasoning;
  std::string content;
  std::vector<ToolCall> tool_calls; // complete calls, emitted when their closing tag arrives
};

struct ChatResult {
  std::string reasoning, content;
  std::vector<ToolCall> tool_calls;
  std::string finish_reason;  // stop | length | tool_calls | cancelled | error
  std::string error;
  std::string stop_sequence;  // "stop" at one of the request's stop strings: that string
  bool context_full = false;  // "length" because the context window (or the memory for it) ran out, not max_tokens
  int prompt_limit = 0;       // "error" because the prompt does not fit: the most prompt tokens that would
  bool busy = false;          // "error" because the engine has no capacity for it now: a 503 with Retry-After, not a 400
  bool invalid_request = false;  // "error" because of the request's content (an image that cannot be read, a video)
  bool engine_down = false;   // "error" because the engine process has exited: a 503 without Retry-After (nothing relaunches it)
  int prompt_tokens = 0, cached_tokens = 0, completion_tokens = 0;
  int reasoning_tokens = 0;   // completion tokens inside the think block
  double prefill_ms = 0, decode_ms = 0;
  // prefill_ms is the time to read the uncached prompt tokens: no image rows (their encoder runs before the clock
  // starts) and no turn given to another request in between (--slots > 1); a rate from it is shown only then
  bool prefill_exact = false;
  // engine bookkeeping for prefill_history: the slot, its use count and the request count at the reply, the
  // request's images (what the processor made of them, without pixels)
  int slot = -1;
  uint64_t slot_gen = 0, req_seq = 0;
  std::shared_ptr<const std::vector<vision::Prepared>> images;
  // concurrency: time this request waited for a free slot, and the GPU turns it gave to other requests
  double queue_ms = 0;
  int yields = 0;
  int batched = 0;  // decode cycles it ran in one pass with other requests' (batched decode)
  int loops = 0;    // looped models: the loop count the request ran (0: not a looped model)
  // KFlash around the decode (src/model/kflash_wrap.h): the proposal source ("DFlash2", "MTP", "prompt lookup", "none":
  // plain decoding, 1 row per cycle) and the cycles it ran; empty source: the wrapper is off (NI_KFLASH=0)
  std::string kflash_source;
  int kflash_cycles = 0;
};

}
