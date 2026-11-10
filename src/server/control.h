// The control socket (docs/control-socket.md): HTTP/1.1 over a Unix domain socket in $KAPITSA_HOME/run, answering
// GET /status, GET /events, POST /stop and POST /model for the user who started the server. It is its own small HTTP
// server rather than httplib's: the peer check needs each accepted socket (getpeereid), which httplib keeps private.
// One request per connection (`Connection: close`), so there is no keep-alive state to get wrong.
#pragma once
#include "server/events.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <functional>
#include <memory>
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

// What the socket asks of the server. Requests reach it already checked: a known path and method, a JSON object
// with only the request's fields, and a run_id (when given) that is this run's.
class ControlApi {
 public:
  virtual ~ControlApi() = default;
  virtual nlohmann::json status() = 0;
  virtual ControlReply stop() = 0;  // 202: after_stop() runs once the first one's answer is sent, never again
  virtual void after_stop() {}
  virtual ControlReply model_switch(const nlohmann::json& request) = 0;
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

// Why POST /model `request` cannot switch an instance serving `config` (status 0: it can): a Splash engine serves its
// splash_model folder and nothing else, and this server takes no serve field but `model` on a switch yet
ControlReply switch_refusal(const nlohmann::json& config, const nlohmann::json& request);

// What POST /model switches: the server's manager, held weakly so the socket never keeps it (and its engine, whose
// destructor saves state) alive after its owner lets it go. get() is null before attach() and after that.
template <class T>
class Attached {
 public:
  void attach(const std::shared_ptr<T>& p) {
    std::lock_guard<std::mutex> lk(mu_);
    p_ = p;
  }
  std::shared_ptr<T> get() const {
    std::lock_guard<std::mutex> lk(mu_);
    return p_.lock();
  }

 private:
  mutable std::mutex mu_;
  std::weak_ptr<T> p_;
};

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

  // GET /events for another transport (the engine view's HTTP mirror, src/server/main.cpp): the stream, or null with
  // the 503 to answer when 16 are open. pump_events() then sends its frames, a reset first when the cursor needs one,
  // and keepalives through `write` until `write` fails, `gone()` says the reader left, the stream falls behind, or the
  // server stops.
  struct EventStream {
    std::shared_ptr<Events::Stream> s;
    Resume how = Resume::live;
  };
  EventStream open_events(const std::string& last_event_id, ControlReply* refused);
  void pump_events(const EventStream& es, const std::function<bool(const std::string&)>& write,
                   const std::function<bool()>& gone);
  void set_owner(uid_t uid) { owner_ = uid; }  // whose connections are served (tests); the server's user otherwise
  // A stop began (POST /stop, a signal): POST /model answers 503 stopping, a further POST /stop 202 and nothing more
  void mark_stopping() { stop_asked_ = true; }
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
  std::atomic<bool> stopping_{false}, stop_asked_{false};
  std::atomic<int> connections_{0};
  std::mutex mu_;  // start/stop
  bool started_ = false;
};

}  // namespace ni::server
