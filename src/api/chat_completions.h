// The OpenAI Chat Completions API's replies (POST /v1/chat/completions; main.cpp reads the request, which is a chat
// request as it stands): a reply's usage and tool calls, and its stream's chunks.
#pragma once
#include "api/common.h"

#include <vector>

namespace ni::api {

// {"prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details": {"cached_tokens"}}, and for looped
// models "kapitsa": {"loops"}
ojson chat_usage(const ChatResult& res);
// The tool_calls array ({"id", "type": "function", "function": {"name", "arguments"}}); with_index (a stream's chunks):
// each call's "index", from index0
ojson chat_tool_calls(const std::vector<ToolCall>& tcs, size_t index0, bool with_index);

// The stream: a chat.completion.chunk with the assistant's role at begin(), one per delta's reasoning
// (reasoning_content), text (content) and tool calls (tool_calls, indexed across the reply), then one with
// finish_reason and usage, and "data: [DONE]". It starts before the prompt is counted (begin() takes no counts), so
// a failure, a prompt the context cannot hold included, is an {"error": ...} event before [DONE].
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
