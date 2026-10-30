// Refusing a WebSocket handshake on the Responses API (issue #772): Codex Desktop probes ws://<host>/v1/responses
// before it falls back to HTTP; a 404 retries with backoff ("Reconnecting /5"), a 426 falls back at once.
#pragma once

#include <httplib/httplib.h>

#include <cctype>
#include <cstdint>
#include <string>

namespace ni::server {

// True and `res` set when the request is a WebSocket upgrade for /v1/responses: 426 and a JSON reason. Runs before
// the API key check so a probe without a key gets the same reply.
inline bool refuse_ws_upgrade(const httplib::Request& req, httplib::Response& res) {
  if (req.path != "/v1/responses") return false;
  std::string up = req.get_header_value("Upgrade");
  for (auto& c : up) c = char(std::tolower(uint8_t(c)));
  if (up.find("websocket") == std::string::npos) return false;
  res.status = 426;
  res.set_content(R"({"error":{"message":"Kapitsa uses the HTTP Responses transport","type":"invalid_request_error"}})",
                  "application/json");
  return true;
}

}  // namespace ni::server
