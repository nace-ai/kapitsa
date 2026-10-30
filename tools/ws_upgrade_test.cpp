// Codex Desktop opens a WebSocket handshake on /v1/responses before falling back to HTTP (issue #772): the probe
// must get 426 so the client falls back at once; every other request passes through untouched. Runs the production
// refusal (src/server/ws_upgrade.h, wired in common_routes' pre-routing handler) on a stand-in httplib server.
// No model, no GPU.
//   ws_upgrade_test     exit status 1 if any check fails
#include "server/ws_upgrade.h"

#include <httplib/httplib.h>

#include <cstdio>
#include <string>
#include <thread>

using namespace ni;

namespace {

int fails = 0, passes = 0;

void check(bool ok, const std::string& name, const std::string& detail = "") {
  if (ok) {
    ++passes;
    printf("ok   %s\n", name.c_str());
  } else {
    ++fails;
    printf("FAIL %s%s%s\n", name.c_str(), detail.empty() ? "" : ": ", detail.c_str());
  }
}

std::string describe(const httplib::Result& r) {
  return r ? std::to_string(r->status) + " " + r->body : std::string("no reply");
}

}  // namespace

int main() {
  httplib::Server srv;
  // the same wiring as common_routes in src/server/main.cpp: the refusal runs first, routes answer the rest
  srv.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
    if (server::refuse_ws_upgrade(req, res)) return httplib::Server::HandlerResponse::Handled;
    return httplib::Server::HandlerResponse::Unhandled;
  });
  srv.Post("/v1/responses", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(R"({"status":"completed"})", "application/json");
  });
  const int port = srv.bind_to_any_port("127.0.0.1");
  std::thread th([&] { srv.listen_after_bind(); });

  httplib::Client cli("127.0.0.1", port);
  // the issue's probe, byte for byte
  const httplib::Headers probe = {{"Connection", "Upgrade"}, {"Upgrade", "websocket"}, {"Sec-WebSocket-Version", "13"},
                                  {"Sec-WebSocket-Key", "x3JJHMbDL1EzLkh9GBhXDw=="}};

  const auto ws = cli.Get("/v1/responses", probe);
  check(ws && ws->status == 426 && ws->body.find("HTTP Responses transport") != std::string::npos &&
            ws->get_header_value("Content-Type") == "application/json",
        "websocket upgrade on /v1/responses: 426 and a JSON reason, not 404", describe(ws));

  const auto plain = cli.Get("/v1/responses");
  check(plain && plain->status == 404, "a plain GET is not the probe: untouched (404, no GET route exists)",
        describe(plain));

  const auto post = cli.Post("/v1/responses", "{}", "application/json");
  check(post && post->status == 200 && post->body.find("completed") != std::string::npos,
        "POST /v1/responses passes through to its handler", describe(post));

  const auto other = cli.Get("/v1/chat/completions", probe);
  check(other && other->status == 404, "the same probe on another endpoint is untouched", describe(other));

  const auto mixed = cli.Get("/v1/responses", {{"Upgrade", "WebSocket"}});
  check(mixed && mixed->status == 426, "the Upgrade value is read case-insensitively", describe(mixed));

  srv.stop();
  th.join();
  printf("%d passed, %d failed\n", passes, fails);
  return fails ? 1 : 0;
}
