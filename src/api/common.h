// What the Anthropic Messages and OpenAI Responses endpoints share: errors, SSE frames, ids, reasoning carried
// through an opaque field, reasoning effort, and the stream writer interface the server drives (Chat Completions uses
// it too).
#pragma once
#include "engine/chat.h"
#include "engine/grammar.h"

#include <stdexcept>
#include <string>

namespace ni::api {

// A request refused before the engine runs (HTTP status, the API's error type, message, and for Responses a code
// such as "context_length_exceeded" and the parameter at fault).
struct ApiError : std::runtime_error {
  int status;
  std::string type, code, param;
  ApiError(int status_, std::string type_, const std::string& message, std::string code_ = "", std::string param_ = "")
      : std::runtime_error(message), status(status_), type(std::move(type_)), code(std::move(code_)),
        param(std::move(param_)) {}
};

// JSON text with any invalid UTF-8 replaced (a model's output is not guaranteed to be valid UTF-8 at its very end)
std::string dump(const ojson& j);
// One server-sent event: "event: <name>\ndata: <json>\n\n"
std::string sse(const std::string& event, const ojson& data);
// A random id: prefix + 24 letters and digits (thread-safe)
std::string new_id(const std::string& prefix);

std::string base64_encode(const std::string& in);
bool base64_decode(const std::string& in, std::string* out);
// The reasoning of a turn in an opaque field the client returns unchanged (an Anthropic thinking block's signature,
// a Responses reasoning item's encrypted_content), so the next request renders it exactly as generated.
std::string pack_reasoning(const std::string& text);
// The text, if `blob` is one of ours
bool unpack_reasoning(const std::string& blob, std::string* text);

// A reasoning effort into the chat template's kwargs: "none" and "minimal" turn thinking off, "low" and "medium" pass
// as reasoning_effort, and "high", "xhigh" and "max" leave the template's default, its highest level (Qwen3.8 takes
// low, medium and xhigh, the default, and raises on anything else; Qwen3.6 has no levels). Other values are left out
// (returns false).
bool apply_effort(const std::string& effort, ojson& kwargs);

// A request's own loop count for looped models (Dory): "loops" in the body (the OpenAI clients' extra_body), 1-4,
// else 0. Throws ApiError (400) for another value.
int request_loops_field(const ojson& body);
// A request's thinking budget beyond the API's own field (src/engine/think_control.h), on Messages, Chat Completions
// and Responses (not the Ollama routes): "thinking_control" in the body, {"hard_cap": tokens (0 or more),
// "close_text": text forced before </think> at the cap, "end_on_tool_call": a tool call inside the reasoning ends the
// control (default: when the request has tools)}, into r.
// Throws ApiError (400) for a malformed one.
void request_think_field(const ojson& body, ChatRequest& r);
// The usage fields that report them: {"loops": n} for a looped model, {"thinking": {...}} for a request with a thinking
// budget; null when neither applies
ojson kapitsa_usage(const ChatResult& res);

// Appends a text piece, with a newline between pieces that have none at the junction
void join_text(std::string& acc, const std::string& piece);

// Tool call arguments as the chat template takes them: a JSON object (a JSON text is parsed; anything that is not
// an object is kept as {"arguments": <text>})
ojson arguments_object(const ojson& args);

// Structured output (src/engine/grammar.h). A JSON Schema format's own fields, {"schema": ..., "strict": ...} (Chat's
// response_format.json_schema, Responses' text.format), into spec; ApiError (400, param) for a malformed one.
void schema_format(const ojson& f, OutputSpec& spec, const std::string& param);
// The grammar of `spec` over the request's tools as the prompt declares them (output_grammar), or null when nothing
// constrains the reply; ApiError (400) for one that cannot be held, naming the field at fault (format_param: the
// API's name for the format).
std::shared_ptr<const Grammar> request_grammar(const OutputSpec& spec, const ojson& tools,
                                               const std::string& format_param = "response_format");

// How far the engine has read a request's prompt (llama.cpp's prompt_progress): its tokens, of which reused from the
// cache, read so far (the cached ones included), and the milliseconds since the cached prefix was known
struct PromptProgress {
  int total = 0, cache = 0, processed = 0;
  long long time_ms = 0;
};

// The events of one streamed reply. The server calls begin() once the engine has counted the prompt and matched it
// against its cache (prompt_tokens and cached_tokens set), or at once, without them, for a stream that starts before
// the request runs (Chat Completions), delta() for each engine delta, keepalive() when nothing was sent for a while,
// progress() while the prompt is read when the request asked for it, and end() with the finished result. Each returns
// the SSE text to send (possibly empty).
class StreamWriter {
 public:
  virtual ~StreamWriter() = default;
  virtual std::string begin(const ChatResult& prompt) = 0;
  virtual std::string delta(const ChatDelta& d) = 0;
  virtual std::string keepalive() = 0;
  virtual std::string progress(const PromptProgress&) { return {}; }  // only Chat Completions has an event for it
  virtual std::string end(const ChatResult& res) = 0;
};

}  // namespace ni::api
