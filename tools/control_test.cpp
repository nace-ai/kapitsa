// The control socket (src/server/control.h) over a real Unix socket in a temporary folder, with a stand-in for the
// server: the start lock, every endpoint's checks and errors, the peer check, the event stream's resets and its cap,
// and the socket file's removal. --print: one line per message kind, {"schema", "message"}, for
// tests/test_control_socket.py to check against schema/control/. CPU only.
#include "server/activity.h"
#include "server/control.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

using namespace ni::server;
using nlohmann::json;
#define CHECK(x) do { if (!(x)) { std::cerr << __LINE__ << ": " #x "\n"; std::abort(); } } while (false)

static const std::string kRun = "7f3a9c2e5b1d4e60";

// A stand-in server: the real status assembly over a real bus and activity, switches that succeed or fail by name
struct Stub : ControlApi {
  Events& bus;
  Activity& act;
  std::atomic<int> stops{0};
  Identity id;
  Stub(Events& b, Activity& a) : bus(b), act(a) {
    id.instance_id = "code", id.version = "0.2.33", id.started_at = "2026-11-10T01:00:00Z";
    id.lifecycle = "foreground", id.api_url = "http://127.0.0.1:8100";
    id.config = {{"model", "27b"}, {"port", 8100}, {"private", true}};
  }
  json status() override { return instance_status(id, bus, act, last_state()); }
  ControlReply stop() override { return {202, {{"run_id", bus.run_id()}, {"state", "stopping"}}}; }
  void after_stop() override { ++stops; }
  ControlReply model_switch(const json& r) override {
    if (ControlReply no = switch_refusal(id.config, r); no.status) return no;
    const json prev = model_json("qwen3.6-27b", 131072, 2);
    const std::string m = r["model"];
    if (m == "nope") return control_error(404, "model_not_found", "no model named nope", {{"model", m}});
    if (m == "oom")
      return control_error(500, "load_failed", "Engine initialization failed: out of memory",
                           {{"model", m}, {"state", "ready"}});
    return {200, {{"run_id", bus.run_id()}, {"model", model_json(m, 65536, 1)}, {"previous_model", prev}}};
  }
};

struct Answer {
  int status = 0;
  std::string head, body;
};

static int dial(const std::string& path) {
  const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  sockaddr_un a{};
  a.sun_family = AF_UNIX;
  memcpy(a.sun_path, path.c_str(), path.size());
  if (connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0) {
    close(fd);
    return -1;
  }
  return fd;
}

static std::string read_all(int fd) {
  std::string out;
  char buf[4096];
  for (ssize_t n; (n = recv(fd, buf, sizeof buf, 0)) > 0;) out.append(buf, size_t(n));
  return out;
}

// One request as raw bytes, and its answer
static Answer raw(const std::string& sock, const std::string& req) {
  const int fd = dial(sock);
  CHECK(fd >= 0);
  CHECK(send(fd, req.data(), req.size(), 0) == ssize_t(req.size()));
  const std::string raw = read_all(fd);
  close(fd);
  Answer a;
  const size_t split = raw.find("\r\n\r\n");
  CHECK(split != std::string::npos && raw.compare(0, 9, "HTTP/1.1 ") == 0);
  a.status = atoi(raw.c_str() + 9);
  a.head = raw.substr(0, split);
  a.body = raw.substr(split + 4);
  return a;
}

static Answer ask(const std::string& sock, const std::string& method, const std::string& path,
                  const std::string& body = "") {
  return raw(sock, method + " " + path + " HTTP/1.1\r\nHost: kapitsa\r\nContent-Length: " +
                       std::to_string(body.size()) + "\r\n\r\n" + body);
}

static std::string code_of(const Answer& a) { return json::parse(a.body)["error"]["code"]; }

// Reads an event stream until `frames` frames have come; each frame as {id, type, data}
static json frames_of(int fd, int frames) {
  std::string in;
  char buf[4096];
  json out = json::array();
  timeval tv{5, 0};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  while (int(out.size()) < frames) {
    size_t end;
    while ((end = in.find("\n\n")) != std::string::npos && int(out.size()) < frames) {
      const std::string f = in.substr(0, end);
      in.erase(0, end + 2);
      if (f.rfind("HTTP/1.1", 0) == 0) {
        const size_t body = f.find("\r\n\r\n");
        if (body == std::string::npos) continue;
        in.insert(0, f.substr(body + 4) + (f.size() > body + 4 ? "\n\n" : ""));
        continue;
      }
      json fr;
      size_t at = 0;
      while (at < f.size()) {
        const size_t nl = f.find('\n', at);
        const std::string line = f.substr(at, nl == std::string::npos ? std::string::npos : nl - at);
        at = nl == std::string::npos ? f.size() : nl + 1;
        if (line.rfind("id: ", 0) == 0) fr["id"] = line.substr(4);
        else if (line.rfind("event: ", 0) == 0) fr["type"] = line.substr(7);
        else if (line.rfind("data: ", 0) == 0) fr["data"] = json::parse(line.substr(6));
      }
      if (!fr.is_null()) out.push_back(fr);
    }
    if (int(out.size()) >= frames) break;
    const ssize_t n = recv(fd, buf, sizeof buf, 0);
    if (n <= 0) break;
    in.append(buf, size_t(n));
  }
  return out;
}

static int open_stream(const std::string& sock, const std::string& cursor) {
  const int fd = dial(sock);
  CHECK(fd >= 0);
  const std::string req = "GET /events HTTP/1.1\r\nHost: kapitsa\r\n" +
                          (cursor.empty() ? std::string() : "Last-Event-ID: " + cursor + "\r\n") + "\r\n";
  CHECK(send(fd, req.data(), req.size(), 0) == ssize_t(req.size()));
  return fd;
}

int main(int argc, char** argv) {
  const bool print = argc > 1 && std::string(argv[1]) == "--print";
  char tmpl[] = "/tmp/kapitsa-control-XXXXXX";
  CHECK(mkdtemp(tmpl));
  const std::string home = tmpl, dir = control_dir(home);
  struct stat st;
  CHECK(!dir.empty() && stat(dir.c_str(), &st) == 0 && (st.st_mode & 0777) == 0700);
  // a home whose run/ cannot be a folder: "" (the server then serves without a socket), never a path at the root
  const std::string blocked = home + "/blocked";
  CHECK(mkdir(blocked.c_str(), 0700) == 0 && close(open((blocked + "/run").c_str(), O_CREAT | O_WRONLY, 0600)) == 0);
  CHECK(control_dir(blocked).empty());
  unlink((blocked + "/run").c_str());
  rmdir(blocked.c_str());
  {  // POST /model's manager: none before attach, the owner's while it lives, none once the owner lets it go
    Attached<int> slot;
    CHECK(!slot.get());
    auto owner = std::make_shared<int>(7);
    slot.attach(owner);
    CHECK(slot.get() && *slot.get() == 7);
    std::weak_ptr<int> seen = owner;
    owner.reset();
    CHECK(seen.expired() && !slot.get());
  }
  CHECK(api_url("", 8100) == "http://127.0.0.1:8100" && api_url("::", 1) == "http://127.0.0.1:1" &&
        api_url("::1", 8100) == "http://[::1]:8100" && api_url("10.0.0.2", 80) == "http://10.0.0.2:80");
  CHECK(valid_instance_id("code") && valid_instance_id("a_B-9") && !valid_instance_id("") &&
        !valid_instance_id(std::string(33, 'a')) && !valid_instance_id("a.b") && !valid_instance_id("a/b"));

  // the start lock: one holder; a second server's wait ends without it
  const int lock = take_instance_lock(dir + "/code.lock", 1000);
  CHECK(lock >= 0 && take_instance_lock(dir + "/code.lock", 100) < 0);

  Events bus(kRun);
  Activity act(bus, 1000);
  Stub stub(bus, act);
  ControlServer srv(stub, bus);
  publish_state("starting", nullptr);  // (no process bus here: last_state() keeps its first value)
  bus.publish("state", {{"state", "starting"}, {"model", nullptr}});
  CHECK(srv.start(dir + "/" + std::string(120, 'x') + ".sock").find("longer than macOS allows") != std::string::npos);
  const std::string sock = dir + "/code.sock";
  CHECK(srv.start(sock).empty());
  CHECK(stat(sock.c_str(), &st) == 0 && (st.st_mode & 0777) == 0600);

  json out = json::array();
  const auto keep = [&](const char* schema, const Answer& a) { out.push_back({{"schema", schema}, {"message", json::parse(a.body)}}); };

  Answer a = ask(sock, "GET", "/status");
  CHECK(a.status == 200);
  json s = json::parse(a.body);
  CHECK(s["run_id"] == kRun && s["instance_id"] == "code" && s["name"] == "code" && s["state"] == "starting" &&
        s["last_event_id"] == kRun + ".1" && s["requests"].empty() && s["pid"] == int(getpid()));
  keep("status", a);
  auto r = act.start("openai", "curl/8.7.1");
  a = ask(sock, "GET", "/status");
  CHECK(json::parse(a.body)["requests"].size() == 1 && json::parse(a.body)["last_event_id"] == kRun + ".2");
  keep("status", a);
  act.finish(r, ni::ChatResult{});
  // last_event_id is the seq read with the state, not the bus's last event when the status is assembled
  CHECK(instance_status(stub.id, bus, act, {{"state", "loading"}, {"seq", 1}})["last_event_id"] == kRun + ".1");

  // paths, methods and bodies
  CHECK(code_of(a = ask(sock, "GET", "/nope")) == "not_found" && a.status == 404);
  keep("error", a);
  CHECK(code_of(a = ask(sock, "POST", "/status")) == "method_not_allowed" && a.status == 405);
  keep("error", a);
  CHECK(code_of(ask(sock, "GET", "/stop")) == "method_not_allowed");
  CHECK(code_of(a = ask(sock, "POST", "/stop", "{not json")) == "bad_request" && a.status == 400);
  keep("error", a);
  CHECK(code_of(ask(sock, "POST", "/stop", "[1]")) == "bad_request");
  // over 64 KiB: refused on its Content-Length, before the body is read
  CHECK(code_of(raw(sock, "POST /model HTTP/1.1\r\nContent-Length: 70000\r\n\r\n")) == "bad_request");
  CHECK(srv.handle("POST", "/model", std::string(65 * 1024, ' ')).body["error"]["code"] == "bad_request");
  a = ask(sock, "POST", "/stop", R"({"mem_budget_gb": 40})");
  CHECK(a.status == 400 && code_of(a) == "unknown_field" && json::parse(a.body)["error"]["details"]["field"] == "mem_budget_gb");
  keep("error", a);
  a = ask(sock, "POST", "/stop", R"({"run_id": "XYZ"})");
  CHECK(a.status == 400 && code_of(a) == "invalid_value");
  keep("error", a);
  a = ask(sock, "POST", "/model", R"({"run_id": "0123456789abcdef", "model": "35b"})");
  CHECK(a.status == 409 && code_of(a) == "run_id_mismatch" && json::parse(a.body)["error"]["details"]["run_id"] == kRun);
  keep("error", a);
  CHECK(code_of(ask(sock, "POST", "/model", "{}")) == "invalid_value");

  // a switch: the stand-in's answers come through with their status and body
  a = ask(sock, "POST", "/model", R"({"model": "qwen3.6-35b-a3b", "run_id": ")" + kRun + R"("})");
  CHECK(a.status == 200 && json::parse(a.body)["model"]["id"] == "qwen3.6-35b-a3b");
  keep("model-switch-response", a);
  CHECK((a = ask(sock, "POST", "/model", R"({"model": "nope"})")).status == 404 && code_of(a) == "model_not_found");
  keep("error", a);
  CHECK((a = ask(sock, "POST", "/model", R"({"model": "oom"})")).status == 500 && code_of(a) == "load_failed");
  keep("error", a);
  a = ask(sock, "POST", "/model", R"({"model": "35b", "context": "8k"})");
  CHECK(a.status == 400 && code_of(a) == "invalid_value" && json::parse(a.body)["error"]["details"]["field"] == "context");
  // a Splash instance serves its splash_model folder: a switch is refused, not a restart of the same engine
  stub.id.config["engine"] = "splash";
  a = ask(sock, "POST", "/model", R"({"model": "35b"})");
  CHECK(a.status == 400 && code_of(a) == "invalid_value" && json::parse(a.body)["error"]["details"]["field"] == "model");
  keep("error", a);
  stub.id.config.erase("engine");

  // another user's connection gets 403 (the socket's directory keeps them out already; this is the second check)
  srv.set_owner(geteuid() + 1);
  CHECK((a = ask(sock, "GET", "/status")).status == 403 && code_of(a) == "forbidden_peer");
  keep("error", a);
  srv.set_owner(geteuid());

  // events: live from the next one; a cursor that cannot continue starts with a reset carrying a fresh status
  int live = open_stream(sock, "");
  usleep(200 * 1000);
  bus.publish("log", {{"level", "info"}, {"message", "hello"}});
  json f = frames_of(live, 1);
  CHECK(f.size() == 1 && f[0]["type"] == "log" && f[0]["id"] == kRun + ".4");
  close(live);
  int bad = open_stream(sock, "garbage");
  f = frames_of(bad, 1);
  CHECK(f.size() == 1 && f[0]["type"] == "reset" && f[0]["data"]["reason"] == "invalid_cursor" &&
        f[0]["id"] == f[0]["data"]["status"]["last_event_id"]);
  out.push_back({{"schema", "event"}, {"message", f[0]}});
  close(bad);
  int other = open_stream(sock, "00000000deadbeef.3");
  f = frames_of(other, 1);
  CHECK(f.size() == 1 && f[0]["data"]["reason"] == "run_changed");
  close(other);
  int again = open_stream(sock, kRun + ".3");  // replay: the event after the cursor
  f = frames_of(again, 1);
  CHECK(f.size() == 1 && f[0]["id"] == kRun + ".4");
  close(again);

  // a reader that never reads: once the socket's buffers fill, its stream's send gives up, and the slot frees
  int stalled = open_stream(sock, "");
  usleep(300 * 1000);
  const std::string line(4000, 'x');
  for (int i = 0; i < 600; ++i) bus.publish("log", {{"level", "info"}, {"message", line}});
  for (int i = 0; i < 200 && srv.connections() > 0; ++i) usleep(50 * 1000);
  CHECK(srv.connections() == 0);
  close(stalled);

  // the stream cap: 16 open, a 17th answers 503 busy with Retry-After
  usleep(1500 * 1000);  // (the streams closed above end within a second)
  int fds[Events::kMaxStreams];
  for (int& fd : fds) {
    fd = open_stream(sock, "");
    frames_of(fd, 0);
  }
  usleep(300 * 1000);
  a = ask(sock, "GET", "/events");
  CHECK(a.status == 503 && code_of(a) == "busy" && a.head.find("Retry-After: 5") != std::string::npos);
  keep("error", a);
  for (int fd : fds) close(fd);

  // stop: 202, then the server's stop runs once the answer is out; stop() removes the socket file
  a = ask(sock, "POST", "/stop", R"({"run_id": ")" + kRun + R"("})");
  CHECK(a.status == 202 && json::parse(a.body)["state"] == "stopping");
  keep("stop-response", a);
  CHECK(stub.stops == 1);
  CHECK(ask(sock, "POST", "/stop", "").status == 202 && stub.stops == 1);  // again (no body: any run): no second stop
  CHECK((a = ask(sock, "POST", "/model", R"({"model": "35b"})")).status == 503 && code_of(a) == "stopping");
  keep("error", a);
  srv.stop();
  CHECK(stat(sock.c_str(), &st) != 0 && dial(sock) < 0);

  // the process bus: last_state() reads the state and the bus's last seq under the lock state events publish under
  start_activity();
  publish_state("ready", model_json("qwen3.6-27b", 131072, 2));
  events()->publish("log", {{"level", "info"}, {"message", "after"}});
  CHECK(last_state()["state"] == "ready" && last_state()["seq"] == events()->last_seq());

  close(lock);
  unlink((dir + "/code.lock").c_str());
  rmdir(dir.c_str());
  rmdir(home.c_str());
  if (print)
    for (const auto& m : out) std::cout << m.dump() << "\n";
  else
    std::cout << "control_test: ok\n";
  return 0;
}
