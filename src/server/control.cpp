#include "server/control.h"

#include "server/activity.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <thread>

namespace ni::server {

using nlohmann::json;

namespace {

constexpr int kMaxConnections = 64;  // above the 16 event streams plus requests; a further one is closed at once
constexpr int kReadTimeoutS = 5;     // a request's head and body
constexpr int kKeepaliveMs = 15000;
constexpr int kSendTimeoutS = 5;     // a stream whose reader stops reading ends, and frees its slot

const char* reason_phrase(int status) {
  switch (status) {
    case 200: return "OK";
    case 202: return "Accepted";
    case 400: return "Bad Request";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 409: return "Conflict";
    case 503: return "Service Unavailable";
    default: return "Internal Server Error";
  }
}

bool send_all(int fd, const std::string& s) {
  size_t off = 0;
  while (off < s.size()) {
    const ssize_t n = ::send(fd, s.data() + off, s.size() - off, 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    off += size_t(n);
  }
  return true;
}

void send_reply(int fd, const ControlReply& r) {
  const std::string body = r.body.dump();
  std::string head = "HTTP/1.1 " + std::to_string(r.status) + " " + reason_phrase(r.status) +
                     "\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) +
                     "\r\nConnection: close\r\n";
  if (r.retry_after_s > 0) head += "Retry-After: " + std::to_string(r.retry_after_s) + "\r\n";
  send_all(fd, head + "\r\n" + body);
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return s;
}

std::string trim(const std::string& s) {
  const size_t a = s.find_first_not_of(" \t"), b = s.find_last_not_of(" \t\r");
  return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

struct Head {
  std::string method, path, last_event_id;
  long long length = 0;
};

// The request line and the headers this server reads; false when they do not parse
bool parse_head(const std::string& text, Head* h) {
  size_t eol = text.find("\r\n");
  const std::string line = text.substr(0, eol);
  const size_t a = line.find(' '), b = line.rfind(' ');
  if (a == std::string::npos || b <= a || line.compare(b + 1, 5, "HTTP/") != 0) return false;
  h->method = line.substr(0, a);
  h->path = line.substr(a + 1, b - a - 1);
  if (const size_t q = h->path.find('?'); q != std::string::npos) h->path.resize(q);
  while (eol != std::string::npos && eol + 2 < text.size()) {
    const size_t start = eol + 2;
    eol = text.find("\r\n", start);
    const std::string hl = text.substr(start, eol == std::string::npos ? std::string::npos : eol - start);
    const size_t c = hl.find(':');
    if (c == std::string::npos) continue;
    const std::string name = lower(trim(hl.substr(0, c))), value = trim(hl.substr(c + 1));
    if (name == "content-length") {
      char* end = nullptr;
      h->length = std::strtoll(value.c_str(), &end, 10);
      if (value.empty() || *end || h->length < 0) return false;
    } else if (name == "last-event-id") {
      h->last_event_id = value;
    }
  }
  return true;
}

bool known_path(const std::string& p) { return p == "/status" || p == "/events" || p == "/stop" || p == "/model"; }

const char* method_of(const std::string& p) { return p == "/status" || p == "/events" ? "GET" : "POST"; }

}  // namespace

ControlReply control_error(int status, const std::string& code, const std::string& message, const json& details) {
  json e = {{"code", code}, {"message", message}};
  if (!details.is_null()) e["details"] = details;
  return {status, {{"error", e}}};
}

json instance_status(const Identity& id, const Events& bus, const Activity& act, const json& state) {
  json j = {{"protocol", kControlProtocol}, {"instance_id", id.instance_id}, {"run_id", bus.run_id()},
            {"name", id.name.empty() ? id.instance_id : id.name}, {"pid", int(getpid())}, {"version", id.version},
            {"started_at", id.started_at}, {"lifecycle", id.lifecycle}, {"state", state.value("state", "starting")},
            {"api_url", id.api_url}, {"model", state.value("model", json())}, {"config", id.config},
            {"tick_ms", act.tick_ms()},
            {"last_event_id", bus.id(state.contains("seq") ? state["seq"].get<uint64_t>() : bus.last_seq())},
            {"requests", act.requests()}};  // (read after the seq: a request started in between is shown, not missed)
  if (id.parent_pid > 1) j["parent_pid"] = id.parent_pid;
  // (only with the state it was read with: a switch frees the engine, so there is none to show outside ready)
  if (j["state"] == "ready")
    if (json e = act.engine(); e.is_object()) j["engine"] = std::move(e);
  return j;
}

std::string api_url(const std::string& host, int port) {
  const bool any = host.empty() || host == "0.0.0.0" || host == "::";
  const std::string h = any ? "127.0.0.1" : host.find(':') != std::string::npos ? "[" + host + "]" : host;
  return "http://" + h + ":" + std::to_string(port);
}

ControlReply switch_refusal(const json& config, const json& request) {
  if (config.value("engine", "native") == "splash")
    return control_error(400, "invalid_value",
                         "a Splash instance serves its splash_model folder only: start another instance for this model",
                         {{"field", "model"}});
  for (auto it = request.begin(); it != request.end(); ++it)
    if (it.key() != "model" && it.key() != "run_id")
      return control_error(400, "invalid_value", "this server does not take " + it.key() + " on a switch yet",
                           {{"field", it.key()}});
  return {0, nullptr};
}

bool valid_instance_id(const std::string& id) {
  if (id.empty() || id.size() > 32) return false;
  return std::all_of(id.begin(), id.end(), [](unsigned char c) { return std::isalnum(c) || c == '_' || c == '-'; });
}

std::string control_dir(const std::string& home) {
  const std::string dir = home + "/run";
  if (mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST) return "";
  struct stat st;
  if (stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode) || st.st_uid != geteuid()) return "";
  if ((st.st_mode & 0777) != 0700 && chmod(dir.c_str(), 0700) != 0) return "";
  return dir;
}

int take_instance_lock(const std::string& path, int wait_ms) {
  const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
  if (fd < 0) return -1;
  const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
  for (;;) {
    if (flock(fd, LOCK_EX | LOCK_NB) == 0) return fd;
    if (errno != EWOULDBLOCK && errno != EINTR) break;
    if (std::chrono::steady_clock::now() >= until) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  ::close(fd);
  return -1;
}

std::string ControlServer::start(const std::string& socket_path) {
  std::lock_guard<std::mutex> lk(mu_);
  if (started_) return "the control socket is already listening";
  sockaddr_un addr{};
  if (socket_path.size() >= sizeof addr.sun_path)
    return "the control socket's path is longer than macOS allows (" + std::to_string(sizeof addr.sun_path - 1) +
           " bytes): " + socket_path;
  const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return std::string("socket: ") + strerror(errno);
  fcntl(fd, F_SETFD, FD_CLOEXEC);
  addr.sun_family = AF_UNIX;
  memcpy(addr.sun_path, socket_path.c_str(), socket_path.size());
  ::unlink(socket_path.c_str());  // an old one: the caller holds the instance's lock
  const mode_t old = umask(0177);  // created 0600, never briefly wider
  const int rc = ::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr);
  umask(old);
  if (rc != 0 || ::listen(fd, 16) != 0) {
    const std::string why = std::string("cannot listen on ") + socket_path + ": " + strerror(errno);
    ::close(fd);
    return why;
  }
  path_ = socket_path;
  listen_fd_ = fd;
  started_ = true;
  std::thread([this] { accept_loop(); }).detach();
  return "";
}

void ControlServer::stop() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (!started_ || stopping_.exchange(true)) return;
    ::unlink(path_.c_str());
  }
  // the accept loop and every connection see stopping_ within a poll (accept, a stream) or their read timeout
  const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(kReadTimeoutS + 1);
  while ((connections_.load() > 0 || listen_fd_ >= 0) && std::chrono::steady_clock::now() < until)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void ControlServer::accept_loop() {
  pollfd p{listen_fd_, POLLIN, 0};
  while (!stopping_.load()) {
    if (::poll(&p, 1, 200) <= 0) continue;
    const int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) continue;
    fcntl(fd, F_SETFD, FD_CLOEXEC);
    if (connections_.fetch_add(1) >= kMaxConnections) {
      connections_.fetch_sub(1);
      ::close(fd);
      continue;
    }
    std::thread([this, fd] {
      serve(fd);
      ::close(fd);
      connections_.fetch_sub(1);
    }).detach();
  }
  ::close(listen_fd_);
  listen_fd_ = -1;
}

void ControlServer::serve(int fd) {
  const int on = 1;
  setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof on);
  timeval tv{kReadTimeoutS, 0};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  uid_t uid = 0;
  gid_t gid = 0;
  if (getpeereid(fd, &uid, &gid) != 0 || uid != owner_)
    return send_reply(fd, control_error(403, "forbidden_peer", "this instance belongs to another user"));

  std::string in;
  size_t head_end = std::string::npos;
  char buf[4096];
  while ((head_end = in.find("\r\n\r\n")) == std::string::npos) {
    if (in.size() > 16 * 1024) return send_reply(fd, control_error(400, "bad_request", "the request's headers are too long"));
    const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return;  // gone, or silent past the timeout
    in.append(buf, size_t(n));
  }
  Head h;
  if (!parse_head(in.substr(0, head_end), &h))
    return send_reply(fd, control_error(400, "bad_request", "the request does not parse as HTTP/1.1"));
  if (h.length > static_cast<long long>(kMaxControlBody))
    return send_reply(fd, control_error(400, "bad_request", "the body is over 64 KiB"));
  std::string body = in.substr(head_end + 4);
  while (body.size() < size_t(h.length)) {
    const ssize_t n = ::recv(fd, buf, std::min(sizeof buf, size_t(h.length) - body.size()), 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return;
    body.append(buf, size_t(n));
  }
  body.resize(size_t(h.length));
  if (h.path == "/events" && h.method == "GET") return stream_events(fd, h.last_event_id);
  const ControlReply r = handle(h.method, h.path, body);
  send_reply(fd, r);
  if (r.status == 202 && h.path == "/stop") stop_answered();
}

ControlReply ControlServer::handle(const std::string& method, const std::string& path, const std::string& body) {
  if (!known_path(path)) return control_error(404, "not_found", "no such path: " + path);
  if (method != method_of(path)) return control_error(405, "method_not_allowed", path + " takes " + method_of(path));
  if (method == "GET") {
    if (path == "/status") return {200, api_.status()};
    return control_error(400, "bad_request", "GET /events streams only over the socket");
  }
  if (body.size() > kMaxControlBody) return control_error(400, "bad_request", "the body is over 64 KiB");
  json req = json::object();
  if (!body.empty()) {
    req = json::parse(body, nullptr, false);
    if (!req.is_object()) return control_error(400, "bad_request", "the body is not a JSON object");
  }
  static const char* const kStop[] = {"run_id"};
  static const char* const kModel[] = {"run_id", "model",         "model_id",   "context",   "slots",
                                       "spec",   "dflash",        "prefill_chunk", "mem_budget", "disk_cache"};
  const bool stop = path == "/stop";
  for (auto it = req.begin(); it != req.end(); ++it) {
    const auto known = [&](const auto& names) {
      return std::any_of(std::begin(names), std::end(names), [&](const char* n) { return it.key() == n; });
    };
    if (stop ? !known(kStop) : !known(kModel))
      return control_error(400, "unknown_field", "unknown field " + it.key(), {{"field", it.key()}});
  }
  if (req.contains("run_id")) {
    const json& id = req["run_id"];
    const bool shaped = id.is_string() && id.get_ref<const std::string&>().size() >= 16 &&
                        id.get_ref<const std::string&>().size() <= 64 &&
                        std::all_of(id.get_ref<const std::string&>().begin(), id.get_ref<const std::string&>().end(),
                                    [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
    if (!shaped) return control_error(400, "invalid_value", "run_id is 16 to 64 lowercase hex digits", {{"field", "run_id"}});
    if (id != bus_.run_id())
      return control_error(409, "run_id_mismatch", "the instance restarted: its run is " + bus_.run_id(),
                           {{"run_id", bus_.run_id()}});
  }
  if (stop) return api_.stop();  // (again while it stops: 202, and the stop already under way goes on)
  if (stopping_.load() || stop_asked_.load()) return control_error(503, "stopping", "the instance is stopping");
  if (!req.contains("model") || !req["model"].is_string() || req["model"].get_ref<const std::string&>().empty())
    return control_error(400, "invalid_value", "model is required: a catalog name, a folder or a GGUF file",
                         {{"field", "model"}});
  return api_.model_switch(req);
}

ControlServer::EventStream ControlServer::open_events(const std::string& last_event_id, ControlReply* refused) {
  EventStream es;
  es.s = stopping_.load() ? nullptr : bus_.open(last_event_id, &es.how);
  if (!es.s) {
    *refused = control_error(503, "busy", "16 event streams are open", {{"retry_after_s", 5}});
    refused->retry_after_s = 5;
  }
  return es;
}

void ControlServer::pump_events(const EventStream& es, const std::function<bool(const std::string&)>& write,
                                const std::function<bool()>& gone) {
  const auto frame = [&](const std::string& id, const std::string& type, const std::string& data) {
    return "id: " + id + "\nevent: " + type + "\ndata: " + data + "\n\n";
  };
  if (const char* reason = reset_reason(es.how); *reason)
    if (!write(frame(bus_.id(es.s->start()), "reset", json{{"reason", reason}, {"status", api_.status()}}.dump())))
      return;
  auto quiet_since = std::chrono::steady_clock::now();
  while (!stopping_.load() && !gone()) {
    if (auto e = es.s->next(1000)) {
      if (!write(frame(bus_.id(e->seq), e->type, e->data))) return;
      quiet_since = std::chrono::steady_clock::now();
    } else if (es.s->closed()) {
      return;  // fell kMaxUnsent behind: the client reconnects with Last-Event-ID
    } else if (std::chrono::steady_clock::now() - quiet_since >= std::chrono::milliseconds(kKeepaliveMs)) {
      if (!write(": keepalive\n\n")) return;
      quiet_since = std::chrono::steady_clock::now();
    }
  }
}

void ControlServer::stream_events(int fd, const std::string& last_event_id) {
  ControlReply refused;
  const EventStream es = open_events(last_event_id, &refused);
  if (!es.s) return send_reply(fd, refused);
  // a reader that stops reading: send() gives up, the stream ends and its slot frees (without this the thread would wait
  // in send() forever, and the bus's kMaxUnsent cut-off would never be seen)
  timeval tv{kSendTimeoutS, 0};
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
  if (!send_all(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                    "Connection: close\r\n\r\n"))
    return;
  // a client that hung up frees its stream within a second, not at the next keepalive's failed write
  pump_events(es, [fd](const std::string& s) { return send_all(fd, s); }, [fd] {
    pollfd p{fd, POLLIN, 0};
    char c;
    return ::poll(&p, 1, 0) > 0 && ::recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT) == 0;
  });
}

}  // namespace ni::server
