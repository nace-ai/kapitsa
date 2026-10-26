// The OpenAI Chat Completions API's replies (POST /v1/chat/completions; main.cpp reads the request, which is a chat
// request as it stands): a reply's usage and tool calls, its errors, and its stream's chunks.
#pragma once
#include "api/common.h"

#include <utility>
#include <vector>

namespace ni::api {

// {"prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details": {"cached_tokens"}}, and for looped
// models "kapitsa": {"loops"}
ojson chat_usage(const ChatResult& res);
// The tool_calls array ({"id", "type": "function", "function": {"name", "arguments"}}); with_index (a stream's chunks):
// each call's "index", from index0
ojson chat_tool_calls(const std::vector<ToolCall>& tcs, size_t index0, bool with_index);

// {"error":{"message":..,"type":..,"param":..,"code":..}}
ojson chat_error(const std::string& message, const std::string& type = "invalid_request_error",
                 const std::string& code = "", const std::string& param = "");
// A failed reply: HTTP status 400 and its error. A prompt the context (or the memory for it) cannot hold says so as
// OpenAI does (code context_length_exceeded, "maximum context length"), which clients answer by compacting.
std::pair<int, ojson> chat_failure(const ChatResult& res);

// The reply's token limit: max_completion_tokens, else max_tokens, else 16384. A null field counts as absent, as
// OpenAI's API has it (SDKs send unset fields as null); another non-number throws (a 400).
int chat_max_tokens(const ojson& body);

// Structured output of a request (src/engine/grammar.h): response_format (json_object; json_schema with its strict),
// tool_choice (required, or a named function), parallel_tool_calls and strict tools ("strict": true in a function),
// over `tools`, the request's as the prompt declares them (none for tool_choice "none"). Null when nothing constrains
// the reply. Throws ApiError (400) for a malformed or unsupported one.
std::shared_ptr<const Grammar> chat_grammar(const ojson& body, const ojson& tools);

// The stream: a chat.completion.chunk with the assistant's role at begin(), one per delta's reasoning
// (reasoning_content), text (content) and tool calls (tool_calls, indexed across the reply), then one with
// finish_reason and usage, and "data: [DONE]". begin() takes no counts (the stream starts before the request has a
// slot); a failure after it is chat_failure's error as an event before [DONE].
class ChatCompletionsStream : public StreamWriter {
 public:
  ChatCompletionsStream(std::string id, std::string model, long created)
      : id_(std::move(id)), model_(std::move(model)), created_(created) {}
  std::string begin(const ChatResult& prompt) override;
  std::string delta(const ChatDelta& d) override;
  std::string keepalive() override;
  std::string end(const ChatResult& res) override;

 private:
  std::string chunk(const ojson& delta, const ojson& finish, const ojson* usage = nullptr) const;
  std::string id_, model_;
  long created_;
  size_t tools_ = 0;  // tool calls sent: the next one's index
};

}  // namespace ni::api
