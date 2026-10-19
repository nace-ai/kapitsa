// The OpenAI Responses API on the engine: POST /v1/responses (streamed or not), with previous_response_id.
// Input items become an OpenAI-style chat request (instructions and leading system/developer messages as the system
// message, reasoning items as the next assistant turn's reasoning, function calls as tool calls, their outputs as tool
// messages), and the engine's deltas become output items and the stream's events.
#pragma once
#include "api/common.h"

#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include <utility>

namespace ni::api {

// Recent stored responses, for previous_response_id: each one's conversation through its output (the items it was
// given, earlier responses' included, then its output items). Bounded: the oldest are dropped.
class ResponseStore {
 public:
  explicit ResponseStore(size_t capacity = 64) : cap_(capacity) {}
  void put(const std::string& id, ojson items);
  bool get(const std::string& id, ojson* items) const;

 private:
  mutable std::mutex mu_;
  size_t cap_;
  std::deque<std::string> order_;
  std::unordered_map<std::string, ojson> items_;
};

struct ResponsesRequest {
  ChatRequest chat;
  bool stream = false, store = true;
  ojson items = ojson::array();  // the conversation as items: the previous response's, then this request's input
  ojson echo = ojson::object();  // request fields the response repeats (instructions, tools, temperature ...)
  std::map<std::string, std::pair<std::string, std::string>> namespaced;  // tool name in the prompt -> (namespace, name)
  std::set<std::string> custom;  // freeform tools (type "custom"): their calls go out as custom_tool_call items
};

// The request as a chat request. Throws ApiError: 400 for a malformed request or an unknown previous_response_id.
ResponsesRequest responses_request(const ojson& body, const ResponseStore& store);

// {"error":{"message":..,"type":..,"param":..,"code":..}}
ojson responses_error(const std::string& message, const std::string& type, const std::string& code = "",
                      const std::string& param = "");
// A reply that failed before its first event: HTTP status and error body (code context_length_exceeded for a prompt
// the context cannot hold)
std::pair<int, ojson> responses_failure(const ChatResult& res);

// The stream: response.created and response.in_progress at begin(); an output item per run of reasoning (content
// reasoning_text, and encrypted_content with the same text), text (a message) or a tool call (function_call, or
// custom_tool_call for a freeform tool), each with its part and delta events; then response.completed, or
// response.incomplete after a max_output_tokens cut. Every event carries sequence_number.
class ResponsesStream : public StreamWriter {
 public:
  // keeps what it needs of the request (the engine's thread may still call it after the request is gone)
  ResponsesStream(const ResponsesRequest& rq, std::string id, std::string model, long created)
      : echo_(rq.echo), namespaced_(rq.namespaced), custom_(rq.custom), store_(rq.store), id_(std::move(id)),
        model_(std::move(model)), created_(created) {}
  std::string begin(const ChatResult& prompt) override;
  std::string delta(const ChatDelta& d) override;
  std::string keepalive() override;
  std::string end(const ChatResult& res) override;
  const ojson& output() const { return output_; }  // the finished output items
  const ojson& final_response() const { return final_; }  // the response end() sent

 private:
  enum Kind { kNone, kReasoning, kMessage };
  std::string event(const std::string& type, ojson data);
  std::string item_id(const char* kind) const;
  std::string open(Kind k);
  std::string close(bool cut = false);
  ojson response(const std::string& status, const ojson& usage, const ojson& error, const ojson& incomplete) const;
  ojson echo_;
  std::map<std::string, std::pair<std::string, std::string>> namespaced_;
  std::set<std::string> custom_;
  bool store_;
  std::string id_, model_;
  long created_;
  int seq_ = 0;
  Kind open_ = kNone;
  std::string item_id_, text_;  // the open item
  ojson output_ = ojson::array();
  ojson final_;
};

// The whole response of a non-streamed request: the output items a stream would produce
ojson responses_response(const ResponsesRequest& rq, const std::string& id, const std::string& model, long created,
                         const ChatResult& res, ojson* output = nullptr);

}  // namespace ni::api
