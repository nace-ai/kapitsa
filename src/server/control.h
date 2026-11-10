// The control socket (docs/control-socket.md): HTTP/1.1 over a Unix domain socket in $KAPITSA_HOME/run, answering
// GET /status and GET /events for the user who started the server (POST /stop and POST /model: the next change). It is its own small HTTP
// server rather than httplib's: the peer check needs each accepted socket (getpeereid), which httplib keeps private.
// One request per connection (`Connection: close`), so there is no keep-alive state to get wrong.
#pragma once
#include "server/events.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <unistd.h>

namespace ni::server {

struct ControlReply {
  int status = 200;
  nlohmann::json body;
  int retry_after_s = 0;  // > 0: a Retry-After header
};

// The error body every non-2xx answer has (schema/control/error.schema.json)
ControlReply control_error(int status, const std::string& code, const std::string& message,
                           const nlohmann::json& details = nullptr);

// What the socket asks of the server
class ControlApi {
 public:
  virtual ~ControlApi() = default;
  virtual nlohmann::json status() = 0;
};

constexpr int kControlProtocol = 1;

class Activity;

// What GET /status says about the instance itself, fixed for the run
struct Identity {
  std::string instance_id, name, version, started_at, lifecycle, api_url;
  int parent_pid = 0;  // attached only
  nlohmann::json config = nlohmann::json::object();
};
// GET /status from the identity, the bus (run_id), the activity (tick_ms, requests) and the last state event's
// {state, model, seq} (last_state(): seq is the bus's last event when the state was read, so last_event_id never passes
// a state event the status does not show)
nlohmann::json instance_status(const Identity& id, const Events& bus, const Activity& act, const nlohmann::json& state);
constexpr size_t kMaxControlBody = 64 * 1024;

// The inference API's URL for `kapitsa serve --host host --port port`: a wildcard host as 127.0.0.1, IPv6 in brackets
std::string api_url(const std::string& host, int port);

bool valid_instance_id(const std::string& id);  // 1 to 32 of A-Z a-z 0-9 _ -
std::string control_dir(const std::string& home);  // <home>/run, made 0700; "" when it cannot be made

// The instance's start lock: an exclusive flock on `path`, waiting at most `wait_ms`. The descriptor (O_CLOEXEC, so
// no child keeps it) is held until the process exits. -1: another server holds it.
int take_instance_lock(const std::string& path, int wait_ms);

class ControlServer {
 public:
  ControlServer(ControlApi& api, Events& bus) : api_(api), bus_(bus) {}
  ~ControlServer() { stop(); }
  ControlServer(const ControlServer&) = delete;
  ControlServer& operator=(const ControlServer&) = delete;

  // Removes an old socket file, binds and listens (0600, O_CLOEXEC). "" or why it cannot.
  std::string start(const std::string& socket_path);
  // Stops accepting, ends the event streams, removes the socket file
  void stop();
  const std::string& path() const { return path_; }

  // One request's answer, everything but GET /events (tests call it without a socket)
  ControlReply handle(const std::string& method, const std::string& path, const std::string& body);
  void set_owner(uid_t uid) { owner_ = uid; }  // whose connections are served (tests); the server's user otherwise
  int connections() const { return connections_.load(); }  // open connections, event streams included

 private:
  void accept_loop();
  void serve(int fd);
  void stream_events(int fd, const std::string& last_event_id);

  ControlApi& api_;
  Events& bus_;
  std::string path_;
  std::atomic<int> listen_fd_{-1};  // -1 once the accept loop has closed it
  uid_t owner_ = geteuid();
  std::atomic<bool> stopping_{false};
  std::atomic<int> connections_{0};
  std::mutex mu_;  // start/stop
  bool started_ = false;
};

}  // namespace ni::server
