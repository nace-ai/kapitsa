// Ollama-compatible API: /api/chat, /api/generate, /api/tags, /api/show, /api/ps, /api/version, /api/embed.
// Streams newline-delimited JSON (one JSON object per line), not SSE.
#pragma once
#include "api/common.h"
#include "hub/hub.h"
#include <string>

namespace ni::api {

class OllamaChatStream : public StreamWriter {
 public:
  OllamaChatStream(std::string model) : model_(std::move(model)) {}
  std::string begin(const ChatResult& prompt) override;
  std::string delta(const ChatDelta& d) override;
  std::string keepalive() override;
  std::string end(const ChatResult& res) override;
 private:
  std::string model_;
};

class OllamaGenerateStream : public StreamWriter {
 public:
  OllamaGenerateStream(std::string model) : model_(std::move(model)) {}
  std::string begin(const ChatResult& prompt) override;
  std::string delta(const ChatDelta& d) override;
  std::string keepalive() override;
  std::string end(const ChatResult& res) override;
 private:
  std::string model_;
};

ChatRequest ollama_chat_request(const ojson& body, const ChatRequest& defaults);
ChatRequest ollama_generate_request(const ojson& body, const ChatRequest& defaults);
std::string ollama_now();
void ollama_eval_stats(ojson& j, const ChatResult& res);
std::string ollama_resolve_model(const std::string& name);
ojson ollama_tags_response(const std::string& model_id, const std::string& model_name, time_t started);
ojson ollama_show_response(const std::string& model_id, const std::string& model_name);
ojson ollama_ps_response(const std::string& model_id, const std::string& model_name, time_t started);

}  // namespace ni::api
