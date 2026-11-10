#include "server/pre_route.h"

#include "server/engine_view.h"
#include "server/ws_upgrade.h"

namespace ni::server {

httplib::Server::HandlerResponse pre_route(const httplib::Request& req, httplib::Response& res,
                                           const std::string& api_key) {
  // a WebSocket upgrade on /v1/responses (Codex Desktop's probe, #772): 426 before the key check, so the client
  // falls back to HTTP at once instead of retrying a 404 with backoff
  if (refuse_ws_upgrade(req, res)) return httplib::Server::HandlerResponse::Handled;
  // the engine view (docs/control-socket.md, "The engine view"): read-only, this Mac only, and this run's key in
  // place of the API key (src/server/engine_view.h)
  if (const auto gate = engine_gate(req, res)) return *gate;
  // the chat page asks for the key once and sends it with its API calls
  if (api_key.empty() || req.path == "/health" || req.path == "/cluster" ||
      req.path == "/api/version" ||
      (req.method == "GET" && (req.path == "/" || req.path == "/favicon.ico")))
    return httplib::Server::HandlerResponse::Unhandled;
  // OpenAI clients send "Authorization: Bearer KEY", Anthropic ones "x-api-key: KEY" (Claude Code may send both,
  // with an Anthropic key of its own in x-api-key)
  if (req.get_header_value("Authorization") == "Bearer " + api_key || req.get_header_value("x-api-key") == api_key)
    return httplib::Server::HandlerResponse::Unhandled;
  res.status = 401;
  if (req.path.rfind("/v1/messages", 0) == 0)
    res.set_content(R"({"type":"error","error":{"type":"authentication_error","message":"invalid api key"}})",
                    "application/json");  // (api::messages_error's form, as a literal: this unit links alone)
  else
    res.set_content(R"({"error":{"message":"invalid api key","type":"auth_error"}})", "application/json");
  return httplib::Server::HandlerResponse::Handled;
}

}  // namespace ni::server
