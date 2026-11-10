#include "server/engine_view.h"

#include <atomic>

namespace ni::server {

static void reply(httplib::Response& res, const ControlReply& r) {
  res.status = r.status;
  if (r.retry_after_s > 0) res.set_header("Retry-After", std::to_string(r.retry_after_s));
  res.set_content(r.body.dump(), "application/json");
}

static bool is_engine_path(const std::string& p) { return p == "/engine" || p.rfind("/engine/", 0) == 0; }

std::string engine_refusal(const httplib::Request& req) {
  const std::string& a = req.remote_addr;
  if (a != "127.0.0.1" && a != "::1" && a != "::ffff:127.0.0.1") return "the engine view answers this Mac only";
  std::string host = req.get_header_value("Host");
  host = !host.empty() && host[0] == '[' ? host.substr(0, host.find(']') + 1) : host.substr(0, host.find(':'));
  if (host != "127.0.0.1" && host != "localhost" && host != "[::1]")
    return "the engine view answers http://127.0.0.1, http://localhost or http://[::1] only";
  if (req.method == "POST" && req.get_header_value("Content-Type").rfind("application/json", 0) != 0)
    return "a stop or switch is sent as application/json";
  return "";
}

std::optional<httplib::Server::HandlerResponse> engine_gate(const httplib::Request& req, httplib::Response& res) {
  if (!is_engine_path(req.path)) return std::nullopt;
  if (const std::string why = engine_refusal(req); !why.empty()) {
    reply(res, control_error(403, "forbidden_peer", why));
    return httplib::Server::HandlerResponse::Handled;
  }
  if (req.method == "GET") return httplib::Server::HandlerResponse::Unhandled;
  return std::nullopt;
}

void engine_routes(httplib::Server& srv, std::function<ControlServer*()> control, const char* page) {
  srv.Get("/engine", [page](const httplib::Request&, httplib::Response& res) {
    res.set_header("Content-Security-Policy",
                   "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                   "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'");
    res.set_header("X-Content-Type-Options", "nosniff");
    res.set_header("Referrer-Policy", "no-referrer");
    res.set_header("Cache-Control", "no-cache");
    res.set_content(page, "text/html; charset=utf-8");
  });
  const auto no_socket = [](httplib::Response& res) {
    reply(res, control_error(503, "internal", "this server has no control socket (another server holds its instance "
                                              "name, or run/ cannot be made)"));
  };
  srv.Get("/engine/status", [control, no_socket](const httplib::Request&, httplib::Response& res) {
    ControlServer* c = control();
    if (!c) return no_socket(res);
    reply(res, c->handle("GET", "/status", ""));
  });
  srv.Post(R"(/engine/(stop|model))", [control, no_socket](const httplib::Request& req, httplib::Response& res) {
    ControlServer* c = control();
    if (!c) return no_socket(res);
    reply(res, c->handle("POST", "/" + req.matches[1].str(), req.body));
  });
  static std::atomic<int> open{0};
  srv.Get("/engine/events", [control, no_socket](const httplib::Request& req, httplib::Response& res) {
    ControlServer* c = control();
    if (!c) return no_socket(res);
    ControlReply refused = control_error(503, "busy", std::to_string(kMaxEngineViews) + " engine views are open",
                                         {{"retry_after_s", 5}});
    refused.retry_after_s = 5;
    ControlServer::EventStream es;
    if (open.fetch_add(1) < kMaxEngineViews) es = c->open_events(req.get_header_value("Last-Event-ID"), &refused);
    if (!es.s) {
      open.fetch_sub(1);
      return reply(res, refused);
    }
    res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider(
        "text/event-stream",
        [c, es](size_t, httplib::DataSink& sink) {
          // (a viewer that left: is_writable() sees the closed socket within the stream's 1 s wait)
          c->pump_events(es, [&](const std::string& s) { return sink.write(s.data(), s.size()); },
                         [&] { return !sink.is_writable(); });
          sink.done();
          return true;
        },
        [](bool) { open.fetch_sub(1); });
  });
}

}  // namespace ni::server
