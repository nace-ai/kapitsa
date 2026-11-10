// The engine view's HTTP side (src/server/engine_view.h) on a real httplib server on 127.0.0.1, in front of a real
// control server and bus with a stand-in status, behind main.cpp's order (the gate, then an API key check): the page,
// the run's key in place of the API key, read-only (404 and 405 as JSON), the Host, Origin, proxy and peer checks, the
// key's file, the event stream and its cap, and the 503 without a control socket. CPU only.
#include "server/activity.h"
#include "server/engine_view.h"

#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <fstream>
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
  // the control socket stops and switches (#1125); nothing on the inference port may reach either
  std::atomic<int> admin{0};
  ControlReply stop() override { ++admin; return {202, {{"run_id", bus.run_id()}, {"state", "stopping"}}}; }
  ControlReply model_switch(const json&) override { ++admin; return {202, json::object()}; }
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
    res.status = 401;  // (kapitsa serve's API key check, as main.cpp runs it after the gate: no /engine path reaches it)
    return httplib::Server::HandlerResponse::Handled;
  });
  engine_routes(srv, [&] { return socket_up ? &control : nullptr; }, "<!doctype html><title>Kapitsa engine</title>");
  const int port = srv.bind_to_any_port("127.0.0.1");
  CHECK(port > 0);
  std::thread serving([&] { srv.listen_after_bind(); });
  srv.wait_until_ready();
  httplib::Client cl("127.0.0.1", port);
  cl.set_read_timeout(10, 0);

  // the page needs no key; status and events need this run's key, and it stands in for the API key ("k" here)
  const httplib::Headers run_key = {{"Authorization", "Bearer " + engine_token()}};
  CHECK(engine_token().size() == 32 && engine_token().find_first_not_of(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_") == std::string::npos);
  auto r = cl.Get("/engine");
  CHECK(r && r->status == 200 && r->body.find("Kapitsa engine") != std::string::npos);
  CHECK(r->get_header_value("Content-Security-Policy").find("frame-ancestors 'none'") != std::string::npos);
  CHECK(r->get_header_value("X-Frame-Options") == "DENY" && r->get_header_value("Cache-Control") == "no-store");
  for (const httplib::Headers& h : {httplib::Headers{}, httplib::Headers{{"Authorization", "Bearer k"}},
                                    httplib::Headers{{"Authorization", "Bearer " + engine_token() + "x"}}}) {
    r = cl.Get("/engine/status", h);
    CHECK(r && r->status == 403 && error_code(r) == "forbidden_peer");
    r = cl.Get("/engine/events", h);
    CHECK(r && r->status == 403 && error_code(r) == "forbidden_peer");
  }
  r = cl.Get("/engine/status", run_key);
  CHECK(r && r->status == 200 && r->get_header_value("Cache-Control") == "no-store" &&
        r->get_header_value("Referrer-Policy") == "no-referrer");
  const json st = json::parse(r->body);
  CHECK(st["state"] == "starting" && st["run_id"] == "7f3a9c2e5b1d4e60");
  cl.set_default_headers(run_key);
  std::cout << "page, status and the run's key ok\n";

  // read-only: no stop or switch over HTTP, and every other method or path is a JSON error, never the API key's 401
  for (const char* path : {"/engine/stop", "/engine/model", "/engine/", "/engine/status/", "/engine/../v1/models"}) {
    r = cl.Post(path, "{}", "application/json");
    CHECK(r && r->status == 404 && error_code(r) == "not_found");
    r = cl.Get(path);
    CHECK(r && r->status == 404 && error_code(r) == "not_found");
  }
  for (const char* path : {"/engine", "/engine/status", "/engine/events"}) {
    r = cl.Post(path, "{}", "application/json");
    CHECK(r && r->status == 405 && error_code(r) == "method_not_allowed" && r->get_header_value("Allow") == "GET, HEAD");
    r = cl.Delete(path);
    CHECK(r && r->status == 405);
    r = cl.Options(path);
    CHECK(r && r->status == 405 && r->get_header_value("Access-Control-Allow-Origin").empty());
  }
  r = cl.Head("/engine/status");
  CHECK(r && r->status == 200);
  std::cout << "read-only ok\n";

  // this Mac only: a name rebound to 127.0.0.1, another site's page, a proxy, a body; the page itself too
  const std::string here = "127.0.0.1:" + std::to_string(port);
  for (const httplib::Headers& h : {
           httplib::Headers{{"Host", "evil.example:" + std::to_string(port)}},
           httplib::Headers{{"Host", "127.0.0.1.evil.example"}},
           httplib::Headers{{"Origin", "http://evil.example"}},
           httplib::Headers{{"Origin", "null"}},
           httplib::Headers{{"Origin", "https://" + here}},
           httplib::Headers{{"X-Forwarded-For", "203.0.113.9"}},
           httplib::Headers{{"Forwarded", "for=203.0.113.9"}},
           httplib::Headers{{"Via", "1.1 proxy"}},
           httplib::Headers{{"Content-Length", "1000000"}}}) {
    for (const char* path : {"/engine", "/engine/status"}) {
      httplib::Headers with = h;
      with.insert(run_key.begin(), run_key.end());
      r = cl.Get(path, with);
      CHECK(r && r->status == 403 && error_code(r) == "forbidden_peer");
    }
  }
  for (const char* h : {"localhost", "[::1]"}) {
    r = cl.Get("/engine/status", {{"Host", std::string(h) + ":" + std::to_string(port)}});
    CHECK(r && r->status == 200);
  }
  r = cl.Get("/engine/status", {{"Origin", "http://" + here}});
  CHECK(r && r->status == 200);
  // the peer check, on a request as httplib fills it in
  httplib::Request far;
  far.remote_addr = "192.168.1.20", far.method = "GET", far.headers.emplace("Host", "127.0.0.1:8100");
  CHECK(!engine_refusal(far).empty());
  far.remote_addr = "::1";
  CHECK(engine_refusal(far).empty());
  std::cout << "host, origin, proxy and peer checks ok\n";

  // the key's file: 0600, the key, and a link left at its path is replaced, not written through
  {
    char dir[] = "/tmp/engine_view_test.XXXXXX";
    CHECK(mkdtemp(dir));
    const std::string path = std::string(dir) + "/default.token", target = std::string(dir) + "/target";
    CHECK(symlink(target.c_str(), path.c_str()) == 0);
    CHECK(write_engine_token(path).empty());
    struct stat sb;
    CHECK(lstat(path.c_str(), &sb) == 0 && S_ISREG(sb.st_mode) && (sb.st_mode & 0777) == 0600);
    CHECK(access(target.c_str(), F_OK) != 0);
    std::ifstream f(path);
    std::string got;
    std::getline(f, got);
    CHECK(got == engine_token());
    unlink(path.c_str());
    rmdir(dir);
  }
  std::cout << "key file ok\n";

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
        c.set_default_headers(run_key);
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
  CHECK(stub.admin == 0);  // no HTTP request stopped or switched the model
  std::cout << "engine_view_test ok\n";
}
