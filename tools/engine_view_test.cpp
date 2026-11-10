// The engine view's HTTP side (src/server/engine_view.h) on a real httplib server on 127.0.0.1, in front of a real
// control server and bus with a stand-in status, behind main.cpp's order (the gate, then an API key check): the page,
// the run's key in place of the API key, the methods (404 and 405 as JSON), stop and switch only with the key and a JSON
// body, the Host, Origin, proxy and peer checks, the key's file, the event stream and its cap, and the 503 without a
// control socket. CPU only.
#include "server/activity.h"
#include "server/engine_view.h"
#include "server/pre_route.h"

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
  // what reached the control API: a refused request must reach neither
  std::atomic<int> stops{0}, stopped{0}, switches{0};
  std::string switched_to;
  ControlReply stop() override { ++stops; return {202, {{"run_id", bus.run_id()}, {"state", "stopping"}}}; }
  void after_stop() override { ++stopped; }
  ControlReply model_switch(const json& r) override {
    switched_to = r["model"];
    ++switches;
    return {200, {{"run_id", bus.run_id()}, {"model", {{"id", switched_to}}}}};
  }
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
  // kapitsa serve's own pre-routing (src/server/pre_route.h), with --api-key k
  srv.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) { return pre_route(req, res, "k"); });
  srv.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) { res.set_content("{}", "application/json"); });
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
  // the API key still guards the rest of the port, and opens no /engine path (above)
  httplib::Client plain("127.0.0.1", port);  // (cl sends the run's key on every request from here on)
  r = plain.Get("/v1/models", httplib::Headers{{"Authorization", "Bearer k"}});
  CHECK(r && r->status == 200);
  r = plain.Get("/v1/models", run_key);
  CHECK(r && r->status == 401);
  r = plain.Post("/v1/messages", "{}", "application/json");
  CHECK(r && r->status == 401 &&
        r->body == R"({"type":"error","error":{"type":"authentication_error","message":"invalid api key"}})");
  std::cout << "page, status and the run's key ok\n";

  // the methods: every other method or path is a JSON error, never the API key's 401
  for (const char* path : {"/engine/", "/engine/status/", "/engine/stop/", "/engine/../v1/models"}) {
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
  for (const char* path : {"/engine/stop", "/engine/model"}) {
    r = cl.Get(path);
    CHECK(r && r->status == 405 && error_code(r) == "method_not_allowed" && r->get_header_value("Allow") == "POST");
    r = cl.Options(path);  // (a cross-site fetch's preflight: no CORS header, so the browser sends nothing after it)
    CHECK(r && r->status == 405 && r->get_header_value("Access-Control-Allow-Origin").empty());
  }
  r = cl.Head("/engine/status");
  CHECK(r && r->status == 200);
  // stop and switch need this run's key, the API key opens neither, and a body another page's form can send (not
  // application/json) is refused even with the key
  for (const char* path : {"/engine/stop", "/engine/model"}) {
    for (const httplib::Headers& h : {httplib::Headers{}, httplib::Headers{{"Authorization", "Bearer k"}},
                                      httplib::Headers{{"Authorization", "Bearer " + engine_token() + "x"}}}) {
      r = plain.Post(path, h, R"({"model":"4b"})", "application/json");
      CHECK(r && r->status == 403 && error_code(r) == "forbidden_peer");
    }
    for (const char* type : {"text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x",
                             "application/jsonx"}) {
      r = cl.Post(path, R"({"model":"4b"})", type);
      CHECK(r && r->status == 400 && error_code(r) == "bad_request");
    }
  }
  r = cl.Post("/engine/model", std::string(70 * 1024, ' '), "application/json");
  CHECK(r && r->status == 403 && error_code(r) == "forbidden_peer");  // (over 64 KiB: refused before it is read)
  CHECK(stub.stops == 0 && stub.switches == 0);
  std::cout << "methods, and stop and switch only with the run's key ok\n";

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
    for (const char* path : {"/engine/stop", "/engine/model"}) {
      if (h.count("Content-Length")) continue;  // (a real body's length: below, with the key)
      r = cl.Post(path, h, R"({"model":"4b"})", "application/json");
      CHECK(r && r->status == 403 && error_code(r) == "forbidden_peer");
    }
  }
  CHECK(stub.stops == 0 && stub.switches == 0);
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
        bool streaming = false, counted = false;  // (only a 200's stream counts: a 503's body is not a view)
        c.Get(
            "/engine/events",
            [&](const httplib::Response& head) {
              streaming = head.status == 200;
              return true;
            },
            [&](const char*, size_t) {
              if (streaming && !counted) counted = true, ++open;
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

  // stop and switch with the key: the socket's answers and checks (run_id, the fields), then the control API
  {
    const std::string run = R"("run_id":"7f3a9c2e5b1d4e60")";
    r = cl.Post("/engine/model", "{" + run + R"(,"model":"4b"})", "application/json; charset=utf-8");
    CHECK(r && r->status == 200 && stub.switches == 1 && stub.switched_to == "4b");
    CHECK(json::parse(r->body)["model"]["id"] == "4b" && r->get_header_value("Cache-Control") == "no-store");
    r = cl.Post("/engine/model", R"({"run_id":"0000000000000000","model":"4b"})", "application/json");
    CHECK(r && r->status == 409 && error_code(r) == "run_id_mismatch");
    r = cl.Post("/engine/model", R"({"context":4096})", "application/json");
    CHECK(r && r->status == 400 && error_code(r) == "invalid_value");
    r = cl.Post("/engine/stop", R"({"force":true})", "application/json");
    CHECK(r && r->status == 400 && error_code(r) == "unknown_field");
    CHECK(stub.switches == 1 && stub.stops == 0);
    r = cl.Post("/engine/stop", "{" + run + "}", "application/json");
    CHECK(r && r->status == 202 && json::parse(r->body)["state"] == "stopping" && stub.stops == 1);
    for (int i = 0; i < 100 && stub.stopped == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK(stub.stopped == 1);  // after_stop(), once the answer is sent
    r = cl.Post("/engine/stop", "{}", "application/json");
    CHECK(r && r->status == 202);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    CHECK(stub.stopped == 1);  // once only
    r = cl.Post("/engine/model", R"({"model":"9b"})", "application/json");
    CHECK(r && r->status == 503 && error_code(r) == "stopping" && stub.switches == 1);
  }
  std::cout << "stop and switch ok\n";

  // no control socket (another server holds the instance name): 503
  socket_up = false;
  r = cl.Get("/engine/status");
  CHECK(r && r->status == 503 && error_code(r) == "internal");
  r = cl.Post("/engine/stop", "{}", "application/json");
  CHECK(r && r->status == 503 && error_code(r) == "internal");
  r = cl.Get("/engine");
  CHECK(r && r->status == 200);
  std::cout << "no socket ok\n";

  srv.stop();
  serving.join();
  control.stop();
  CHECK(stub.stops == 2 && stub.stopped == 1 && stub.switches == 1);  // the allowed ones only
  std::cout << "engine_view_test ok\n";
}
