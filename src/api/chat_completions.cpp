#include "api/chat_completions.h"

namespace ni::api {

ojson chat_usage(const ChatResult& res) {
  ojson u{{"prompt_tokens", res.prompt_tokens},
          {"completion_tokens", res.completion_tokens},
          {"total_tokens", res.prompt_tokens + res.completion_tokens},
          {"prompt_tokens_details", {{"cached_tokens", res.cached_tokens}}}};
  if (res.loops > 0) u["kapitsa"] = kapitsa_usage(res);  // looped models: the loop count the request ran
  return u;
}

ojson chat_tool_calls(const std::vector<ToolCall>& tcs, size_t index0, bool with_index) {
  ojson a = ojson::array();
  for (size_t i = 0; i < tcs.size(); ++i) {
    ojson t{{"id", tcs[i].id}, {"type", "function"}, {"function", {{"name", tcs[i].name}, {"arguments", tcs[i].arguments}}}};
    if (with_index) t["index"] = index0 + i;
    a.push_back(t);
  }
  return a;
}

ojson chat_error(const std::string& message, const std::string& type, const std::string& code, const std::string& param) {
  return {{"error",
           {{"message", message}, {"type", type}, {"param", param.empty() ? ojson() : ojson(param)},
            {"code", code.empty() ? ojson() : ojson(code)}}}};
}

std::pair<int, ojson> chat_failure(const ChatResult& res) {
  if (res.prompt_limit > 0)
    return {400, chat_error("This model's maximum context length is " + std::to_string(res.prompt_limit) +
                                " tokens. However, your messages resulted in " + std::to_string(res.prompt_tokens) +
                                " tokens. Please reduce the length of the messages.",
                            "invalid_request_error", "context_length_exceeded", "messages")};
  return {400, chat_error(res.error.empty() ? "the request failed" : res.error)};
}

std::string ChatCompletionsStream::chunk(const ojson& delta, const ojson& finish, const ojson* usage) const {
  ojson c{{"id", id_},
          {"object", "chat.completion.chunk"},
          {"created", created_},
          {"model", model_},
          {"choices", ojson::array({ojson{{"index", 0}, {"delta", delta}, {"finish_reason", finish}}})}};
  if (usage) c["usage"] = *usage;
  return "data: " + dump(c) + "\n\n";
}

std::string ChatCompletionsStream::begin(const ChatResult&) {
  return chunk(ojson{{"role", "assistant"}, {"content", ""}}, nullptr);
}

std::string ChatCompletionsStream::delta(const ChatDelta& d) {
  std::string s;
  if (!d.reasoning.empty()) s += chunk(ojson{{"reasoning_content", d.reasoning}}, nullptr);
  if (!d.content.empty()) s += chunk(ojson{{"content", d.content}}, nullptr);
  if (!d.tool_calls.empty()) {
    s += chunk(ojson{{"tool_calls", chat_tool_calls(d.tool_calls, tools_, true)}}, nullptr);
    tools_ += d.tool_calls.size();
  }
  return s;
}

std::string ChatCompletionsStream::keepalive() {
  // A chunk with an empty delta, not an SSE comment. The idle timers that drop a stream during a long prefill count
  // either the body's bytes (Node's fetch: undici's 300 s bodyTimeout, which Cline hit on local models; Bun's 300 s
  // socket timer; opencode's chunkTimeout: 300 s in 1.x, 900 s in 2.x's own OpenAI-compatible client) or its parsed
  // SSE events, which leave comments out (opencode 2's AI SDK path moves its deadline only on an event). A chunk
  // resets both kinds, a comment only the first, and parsers that read every line as "data: " JSON fail on a comment.
  // The delta carries nothing, not even empty content, so no client takes it for the start of text, wherever it lands
  // (a reply also goes quiet mid-reasoning while the parser holds back a tool call, or while it waits for memory).
  return chunk(ojson::object(), nullptr);
}

std::string ChatCompletionsStream::end(const ChatResult& res) {
  if (res.finish_reason == "error") return "data: " + dump(chat_failure(res).second) + "\n\ndata: [DONE]\n\n";
  const ojson u = chat_usage(res);
  return chunk(ojson::object(), res.finish_reason, &u) + "data: [DONE]\n\n";
}

}  // namespace ni::api
