// The Splash engine as a child process: `splash serve-native <model dir> <ctx|auto> <mem|auto> ...` speaking the native
// protocol of splash/protocol.h over its stdin and stdout. A C++ counterpart of splash-1.4.0/server/runtime.py, without its
// relaunch: a child that dies fails every request it held with an engine_unhealthy error and the Runtime stays dead, so the
// owner decides whether to start another (docs/splash-backend.md).
//
// One reader thread decodes the child's events and calls a request's Sink from that thread, so a Sink returns quickly and
// never calls back into the Runtime's submit() or shutdown(); cancel() is allowed. Exactly one of on_done / on_error ends
// a request, whatever the child does.
#pragma once

#include <sys/types.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "splash/protocol.h"

namespace ni::splash {

// A request's events, in order: on_start once, then on_progress and on_tokens any number of times, then on_done or on_error.
class Sink {
 public:
  virtual ~Sink() = default;
  virtual void on_start(const Start&) {}
  virtual void on_progress(const PromptProgress&) {}
  virtual void on_tokens(const Tokens&) = 0;
  virtual void on_done(const Done&) = 0;
  virtual void on_error(const Error&) = 0;
};

// The engine refused or lost the work, and how: a request_error is the request's own, the others are the engine's.
struct EngineUnhealthy : std::runtime_error { using std::runtime_error::runtime_error; };
struct PendingLimitExceeded : std::runtime_error { using std::runtime_error::runtime_error; };

struct RuntimeOptions {
  std::vector<std::string> argv;  // argv[0] is the splash binary; the rest as server.py's _native_command builds it
  std::vector<std::string> env;   // "NAME=value" added to this process's environment
  double ready_timeout_s = 600;   // a cold start reads and calibrates the whole model
  double stop_grace_s = 10;       // after closing the child's stdin, before SIGTERM; and again before SIGKILL
  double write_timeout_s = 30;    // a frame the child does not take within this long: the child is wedged and is killed
  std::function<void(const std::string&)> log;
};

class Runtime {
 public:
  explicit Runtime(RuntimeOptions o);
  ~Runtime();  // shutdown()
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  // Spawns the child and waits for READY. Throws EngineUnhealthy when it does not come (exec failure, exit, protocol
  // failure, timeout), after the child is reaped.
  void start();

  const Ready& ready() const { return ready_; }  // valid after start()
  bool alive() const;

  // Queues the request and returns the id the engine knows it by. The request's id and both deadlines are stamped here:
  // the absolute one is now + remaining, as runtime.py's _wire_deadline does. Throws PendingLimitExceeded when the engine's
  // concurrent-request capacity is full, EngineUnhealthy when it is not running, ProtocolError when the engine would refuse
  // the request's own fields.
  uint64_t submit(Request r, std::shared_ptr<Sink> sink, double deadline_s);

  // Asks the engine to stop the request. It still ends with on_done (cancelled, or finished first) or on_error.
  void cancel(uint64_t request_id);

  // The engine's status JSON, or "" when it does not answer within timeout_s.
  std::string status(double timeout_s = 10);

  // Closes the child's stdin and waits for it to drain and exit, then TERM, then KILL, each after stop_grace_s; the signals
  // go to the child's process group (it is spawned as its own group leader), so a descendant that holds its stdout does not
  // keep this from returning. A writer blocked on a child that does not read is released within 100 ms. Pending requests
  // end with an engine_unhealthy error. Idempotent. Not to be called from a Sink.
  void shutdown();

  pid_t pid() const { return pid_; }  // -1 once reaped

 private:
  struct Pending {
    std::shared_ptr<Sink> sink;
    bool started = false;
  };
  void read_loop();
  void fail_all(const std::string& code, const std::string& message);
  void write_frame(const std::vector<uint8_t>& frame);
  void reap(double grace_s);
  void signal_child(int sig);  // to the child's process group, only while it is not reaped
  void say(const std::string& s) const { if (o_.log) o_.log(s); }

  RuntimeOptions o_;
  std::atomic<pid_t> pid_{-1};
  int in_fd_ = -1, out_fd_ = -1;  // the child's stdin (written) and stdout (read)
  std::thread reader_;

  mutable std::mutex m_;  // state below
  std::condition_variable cv_;
  enum class State { kIdle, kStarting, kReady, kDead } state_ = State::kIdle;
  std::string dead_reason_;
  Ready ready_;
  uint64_t next_id_ = 1;
  std::map<uint64_t, Pending> pending_;
  std::map<uint64_t, std::string> status_replies_;  // correlation id -> reply, for a status() still waiting
  uint64_t next_corr_ = 1;
  std::atomic<bool> stopping_{false};  // shutdown() ran: a dead child is expected
  std::atomic<bool> abandon_{false};   // shutdown() gave up waiting for the child's stdout to close: the reader stops reading
  std::mutex pid_m_;  // pid_ is set and cleared, and signalled through, only under this: a signal never reaches a reaped pid
  std::mutex write_m_;  // one writer at a time on the child's stdin
  std::mutex cancel_m_;            // cancels_ only, never held across a write
  std::vector<uint64_t> cancels_;  // cancel frames waiting for the writer
  void write_locked(const std::vector<uint8_t>& frame);  // write_m_ held
  void flush_cancels();
};

}  // namespace ni::splash
