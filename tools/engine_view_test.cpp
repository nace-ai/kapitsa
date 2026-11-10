// The engine view's HTTP side (src/server/engine_view.h) on a real httplib server on 127.0.0.1, in front of a real
// control server and bus with a stand-in status: the page, the status mirror, the event stream and its cap, the Host
// and JSON checks, the API key on POSTs only, and the 503 without a control socket. CPU only.
#include "server/activity.h"
#include "server/engine_view.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace ni::server;
using nlohmann::json;
#define CHECK(x) do { if (!(x)) { std::cerr << __LINE__ << ": " #x "\n"; std::abort(); } } while (false)

struct Stub : ControlApi {
  Events& bus;
  Activity& act;
  Identity id;
  Stub(Events& b, Activity& a) : bus(b), act(a) {
    id.instance_id = "code", id.version = "0.2.33", id.started_at = "2026-11-10T01:00:00Z";
    id.lifecycle = "foreground", id.api_url = "http://127.0.0.1:8100";
    id.config = {{"model", "27b"}, {"port", 8100}, {"private", true}};
  }
  json status() override { return instance_status(id, bus, act, last_state()); }
};

static std::string error_code(const httplib::Result& r) {
  return r ? json::parse(r->body, nullptr, false).value("error", json::object()).value("code", "") : "";
}

int main() {
  Events bus("7f3a9c2e5b1d4e60");
  Activity act(bus, 1000);
  Stub stub(bus, act);
  ControlServer control(stub, bus);
  bus.publish("state", {{"state", "starting"}, {"model", nullptr}});
  std::atomic<bool> socket_up{true};

  httplib::Server srv;
  srv.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
    if (const auto gate = engine_gate(req, res)) return *gate;
    if (req.get_header_value("Authorization") == "Bearer k") return httplib::Server::HandlerResponse::Unhandled;
    res.status = 401;  // (kapitsa serve's key check, as main.cpp runs it after the gate)
    return httplib::Server::HandlerResponse::Handled;
  });
  engine_routes(srv, [&] { return socket_up ? &control : nullptr; }, "<!doctype html><title>Kapitsa engine</title>");
  const int port = srv.bind_to_any_port("127.0.0.1");
  CHECK(port > 0);
  std::thread serving([&] { srv.listen_after_bind(); });
  srv.wait_until_ready();
  httplib::Client cl("127.0.0.1", port);
  cl.set_read_timeout(10, 0);

  // the page and the status, without the key
  auto r = cl.Get("/engine");
  CHECK(r && r->status == 200 && r->body.find("Kapitsa engine") != std::string::npos);
  CHECK(r->get_header_value("Content-Security-Policy").find("connect-src 'self'") != std::string::npos);
  r = cl.Get("/engine/status");
  CHECK(r && r->status == 200);
  const json st = json::parse(r->body);
  CHECK(st["state"] == "starting" && st["run_id"] == "7f3a9c2e5b1d4e60");
  std::cout << "page and status ok\n";

  // a name rebound to 127.0.0.1 by a page elsewhere: refused, and so is a POST that is not JSON
  r = cl.Get("/engine/status", {{"Host", "evil.example:" + std::to_string(port)}});
  CHECK(r && r->status == 403 && error_code(r) == "forbidden_peer");
  for (const char* h : {"localhost", "[::1]"}) {
    r = cl.Get("/engine/status", {{"Host", std::string(h) + ":" + std::to_string(port)}});
    CHECK(r && r->status == 200);
  }
  r = cl.Post("/engine/stop", {{"Authorization", "Bearer k"}}, "{}", "text/plain");
  CHECK(r && r->status == 403 && error_code(r) == "forbidden_peer");
  // a POST needs the key; with it, the control server answers it as over the socket
  r = cl.Post("/engine/stop", "{}", "application/json");
  CHECK(r && r->status == 401);
  r = cl.Post("/engine/stop", {{"Authorization", "Bearer k"}}, "{}", "application/json");
  CHECK(r && r->status == control.handle("POST", "/stop", "{}").status && !error_code(r).empty());
  // the peer check, on a request as httplib fills it in
  httplib::Request far;
  far.remote_addr = "192.168.1.20", far.method = "GET", far.headers.emplace("Host", "127.0.0.1:8100");
  CHECK(!engine_refusal(far).empty());
  far.remote_addr = "::1";
  CHECK(engine_refusal(far).empty());
  std::cout << "host, json, key and peer checks ok\n";

  // events: what the bus publishes after the cursor, as SSE frames
  {
    std::string got;
    std::thread later([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
      bus.publish("log", {{"level", "info"}, {"message", "hello"}});
    });
    r = cl.Get("/engine/events", {{"Last-Event-ID", st["last_event_id"].get<std::string>()}},
               [&](const char* d, size_t n) {
                 got.append(d, n);
                 return got.find("hello") == std::string::npos;
               });
    later.join();
    CHECK(got.rfind("id: 7f3a9c2e5b1d4e60.", 0) == 0 && got.find("event: log\n") != std::string::npos);
    // an old run's cursor: a reset with a fresh status first
    got.clear();
    r = cl.Get("/engine/events", {{"Last-Event-ID", "0000000000000000.5"}}, [&](const char* d, size_t n) {
      got.append(d, n);
      return got.find("\n\n") == std::string::npos;
    });
    CHECK(got.rfind("id: 7f3a9c2e5b1d4e60.", 0) == 0 && got.find("event: reset\n") != std::string::npos &&
          got.find("run_changed") != std::string::npos);
  }
  std::cout << "events ok\n";

  // at most kMaxEngineViews streams: one more gets 503 with Retry-After, and a slot frees when a viewer leaves
  {
    std::atomic<bool> hold{true};
    std::atomic<int> open{0};
    std::vector<std::thread> views;
    for (int i = 0; i < kMaxEngineViews; ++i)
      views.emplace_back([&] {
        httplib::Client c("127.0.0.1", port);
        c.set_read_timeout(20, 0);
        bool counted = false;
        c.Get("/engine/events", [&](const char*, size_t) {
          if (!counted) counted = true, ++open;
          return hold.load();
        });
      });
    bus.publish("log", {{"level", "info"}, {"message", "open"}});
    for (int i = 0; i < 100 && open < kMaxEngineViews; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      bus.publish("log", {{"level", "info"}, {"message", "open"}});
    }
    CHECK(open == kMaxEngineViews);
    r = cl.Get("/engine/events");
    CHECK(r && r->status == 503 && r->get_header_value("Retry-After") == "5" && error_code(r) == "busy");
    hold = false;
    bus.publish("log", {{"level", "info"}, {"message", "leave"}});
    for (auto& t : views) t.join();
    bool freed = false;
    for (int i = 0; i < 50 && !freed; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      std::string got;
      std::thread later([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        bus.publish("log", {{"level", "info"}, {"message", "back"}});
      });
      int status = 0;
      cl.Get(
          "/engine/events",
          [&](const httplib::Response& res) {
            status = res.status;
            return status == 200;
          },
          [&](const char* d, size_t n) {
            got.append(d, n);
            return false;
          });
      later.join();
      freed = status == 200 && got.find("event: log\n") != std::string::npos;
    }
    CHECK(freed);
  }
  std::cout << "stream cap ok\n";

  // no control socket (another server holds the instance name): 503
  socket_up = false;
  r = cl.Get("/engine/status");
  CHECK(r && r->status == 503 && error_code(r) == "internal");
  r = cl.Get("/engine");
  CHECK(r && r->status == 200);
  std::cout << "no socket ok\n";

  srv.stop();
  serving.join();
  control.stop();
  std::cout << "engine_view_test ok\n";
}
