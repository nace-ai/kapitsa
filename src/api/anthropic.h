// The Anthropic Messages API on the engine: POST /v1/messages (streamed or not) and /v1/messages/count_tokens.
// The request becomes an OpenAI-style chat request (system text first, thinking blocks as the assistant turn's
// reasoning, tool_use as tool calls, tool_result as tool messages), and the engine's deltas become Anthropic's
// content blocks and stream events.
#pragma once
#include "api/common.h"

#include <utility>

namespace ni::api {

struct MessagesRequest {
  ChatRequest chat;
  bool stream = false;
  bool show_thinking = true;  // false for thinking.display "omitted": thinking blocks carry only their signature
};

// The request as a chat request. Throws ApiError (400) for a malformed one.
MessagesRequest messages_request(const ojson& body);
// count_tokens: the prompt of a Messages request (max_tokens and stream not needed)
ChatRequest messages_prompt(const ojson& body, bool* show_thinking = nullptr);

// {"type":"error","error":{"type":..,"message":..}}
ojson messages_error(const std::string& type, const std::string& message);
// A reply that failed before its first event: HTTP status and error body ("prompt is too long: N tokens > M maximum"
// for a prompt the context cannot hold, which Claude Code answers by compacting)
std::pair<int, ojson> messages_failure(const ChatResult& res);

// stop_reason and stop_sequence of a finished reply
std::pair<std::string, ojson> messages_stop(const ChatResult& res);
// The usage object: input_tokens = prompt minus cached, cache_read_input_tokens = cached
ojson messages_usage(const ChatResult& res);

// The whole response of a non-streamed request
ojson messages_response(const std::string& id, const std::string& model, const ChatResult& res, bool show_thinking);

// The stream: message_start and ping at begin(), a content block per run of reasoning, text or a tool call
// (thinking blocks end with their signature_delta), then message_delta and message_stop.
class MessagesStream : public StreamWriter {
 public:
  MessagesStream(std::string id, std::string model, bool show_thinking)
      : id_(std::move(id)), model_(std::move(model)), show_(show_thinking) {}
  std::string begin(const ChatResult& prompt) override;
  std::string delta(const ChatDelta& d) override;
  std::string keepalive() override;
  std::string end(const ChatResult& res) override;

 private:
  enum Kind { kNone, kThinking, kText };
  std::string open(Kind k);
  std::string close();
  std::string id_, model_;
  bool show_;
  Kind open_ = kNone;
  int index_ = -1;         // the open block's index
  int blocks_ = 0;         // blocks started
  std::string reasoning_;  // the open thinking block's text (its signature)
};

}  // namespace ni::api
