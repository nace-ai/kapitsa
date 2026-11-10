#include "server/engine_view.h"

#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <memory>

namespace ni::server {

// Every /engine answer: not kept by a cache, sends no referrer, no sniffing
static void harden(httplib::Response& res) {
  res.set_header("Cache-Control", "no-store");
  res.set_header("Referrer-Policy", "no-referrer");
  res.set_header("X-Content-Type-Options", "nosniff");
}

static void reply(httplib::Response& res, const ControlReply& r) {
  harden(res);
  res.status = r.status;
  if (r.retry_after_s > 0) res.set_header("Retry-After", std::to_string(r.retry_after_s));
  res.set_content(r.body.dump(), "application/json");
}

static bool is_engine_path(const std::string& p) { return p == "/engine" || p.rfind("/engine/", 0) == 0; }

const std::string& engine_token() {
  static const std::string token = [] {
    unsigned char b[24];
    arc4random_buf(b, sizeof b);
    static const char* abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string s;
    for (size_t i = 0; i < sizeof b; i += 3) {
      const unsigned v = b[i] << 16 | b[i + 1] << 8 | b[i + 2];
      for (int k = 18; k >= 0; k -= 6) s += abc[v >> k & 63];
    }
    return s;
  }();
  return token;
}

std::string write_engine_token(const std::string& path) {
  unlink(path.c_str());  // (a stale file of an earlier run, or a link someone left: never written through)
  const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
  if (fd < 0) return "cannot write " + path + ": " + strerror(errno);
  const std::string& t = engine_token();
  const bool ok = write(fd, t.data(), t.size()) == ssize_t(t.size()) && fchmod(fd, 0600) == 0;
  close(fd);
  if (!ok) unlink(path.c_str());
  return ok ? "" : "cannot write " + path;
}

std::string engine_refusal(const httplib::Request& req) {
  const std::string& a = req.remote_addr;
  if (a != "127.0.0.1" && a != "::1" && a != "::ffff:127.0.0.1") return "the engine view answers this Mac only";
  const std::string host_port = req.get_header_value("Host");
  const std::string host = !host_port.empty() && host_port[0] == '['
                               ? host_port.substr(0, host_port.find(']') + 1)
                               : host_port.substr(0, host_port.find(':'));
  if (host != "127.0.0.1" && host != "localhost" && host != "[::1]")
    return "the engine view answers http://127.0.0.1, http://localhost or http://[::1] only";
  if (req.has_header("Origin") && req.get_header_value("Origin") != "http://" + host_port)
    return "the engine view answers its own page only";
  for (const char* h : {"Forwarded", "X-Forwarded-For", "X-Forwarded-Host", "Via"})
    if (req.has_header(h)) return "the engine view is not served through a proxy";
  if (req.has_header("Content-Length") && atoll(req.get_header_value("Content-Length").c_str()) > 64 * 1024)
    return "the engine view takes no body";
  return "";
}

static bool same_key(const std::string& a, const std::string& b) {  // (in time that does not depend on where they differ)
  unsigned char d = a.size() != b.size();
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) d |= a[i] ^ b[i];
  return d == 0;
}

// Content-Type application/json, parameters (charset) aside
static bool json_body(const httplib::Request& req) {
  std::string t = req.get_header_value("Content-Type");
  t = t.substr(0, t.find(';'));
  while (!t.empty() && t.back() == ' ') t.pop_back();
  for (char& ch : t) ch = char(tolower(static_cast<unsigned char>(ch)));
  return t == "application/json";
}

std::optional<httplib::Server::HandlerResponse> engine_gate(const httplib::Request& req, httplib::Response& res) {
  if (!is_engine_path(req.path)) return std::nullopt;
  const auto refuse = [&](const ControlReply& r) {
    reply(res, r);
    return httplib::Server::HandlerResponse::Handled;
  };
  if (const std::string why = engine_refusal(req); !why.empty()) return refuse(control_error(403, "forbidden_peer", why));
  const bool page = req.path == "/engine", action = req.path == "/engine/stop" || req.path == "/engine/model";
  if (!page && !action && req.path != "/engine/status" && req.path != "/engine/events")
    return refuse(control_error(404, "not_found", "the engine view has /engine, /engine/status, /engine/events, "
                                                  "/engine/stop and /engine/model"));
  if (action ? req.method != "POST" : req.method != "GET" && req.method != "HEAD") {
    res.set_header("Allow", action ? "POST" : "GET, HEAD");
    return refuse(control_error(405, "method_not_allowed",
                                action ? req.path + " takes POST" : "the page, its status and its events only read"));
  }
  if (!page && !same_key(req.get_header_value("Authorization"), "Bearer " + engine_token()))
    return refuse(control_error(403, "forbidden_peer", "the engine view needs this run's key: open the link kapitsa "
                                                       "serve printed, or read it from run/<instance>.token"));
  if (action && !json_body(req))  // (a form another page posts cannot send this, nor a key)
    return refuse(control_error(400, "bad_request", "a stop or a switch is a JSON body: Content-Type: application/json"));
  return httplib::Server::HandlerResponse::Unhandled;
}

void engine_routes(httplib::Server& srv, std::function<ControlServer*()> control, const char* page) {
  srv.Get("/engine", [page](const httplib::Request&, httplib::Response& res) {
    harden(res);
    res.set_header("Content-Security-Policy",
                   "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:; "
                   "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'");
    res.set_header("X-Frame-Options", "DENY");
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
  // stop and switch: the socket's own checks and answers. A stop's after_stop() runs once its answer is sent, as on
  // the socket: the content provider's releaser runs after the last byte, or when the client has gone.
  srv.Post("/engine/stop", [control, no_socket](const httplib::Request& req, httplib::Response& res) {
    ControlServer* c = control();
    if (!c) return no_socket(res);
    const ControlReply r = c->handle("POST", "/stop", req.body);
    if (r.status != 202) return reply(res, r);
    harden(res);
    res.status = 202;
    auto body = std::make_shared<const std::string>(r.body.dump());
    res.set_content_provider(
        body->size(), "application/json",
        [body](size_t off, size_t n, httplib::DataSink& sink) { return sink.write(body->data() + off, n); },
        [c](bool) { c->stop_answered(); });
  });
  srv.Post("/engine/model", [control, no_socket](const httplib::Request& req, httplib::Response& res) {
    ControlServer* c = control();
    if (!c) return no_socket(res);
    reply(res, c->handle("POST", "/model", req.body));  // (holds this thread while the model loads, as on the socket)
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
    harden(res);
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
