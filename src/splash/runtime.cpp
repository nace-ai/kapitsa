#include "splash/runtime.h"

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

extern char** environ;

namespace ni::splash {
namespace {

using Clock = std::chrono::steady_clock;

uint64_t unix_micros() {
  using namespace std::chrono;
  return uint64_t(duration_cast<microseconds>(system_clock::now().time_since_epoch()).count());
}

// runtime.py's _wire_deadline: remaining = max(1, int(seconds * 1e6)), saturating for an infinite or huge deadline.
uint64_t remaining_micros(double s) {
  if (!(s < 1.8e13)) return UINT64_MAX;  // also catches +inf and NaN
  if (s <= 0) return 1;
  return std::max<uint64_t>(1, uint64_t(s * 1e6));
}

Error unhealthy(uint64_t id, const std::string& code, const std::string& message) {
  Error e;
  e.failure = FailureClass::kEngineUnhealthy;
  e.request_id = id;
  e.code = code;
  e.message = message;
  return e;
}

}  // namespace

Runtime::Runtime(RuntimeOptions o) : o_(std::move(o)) {}

Runtime::~Runtime() { shutdown(); }

bool Runtime::alive() const {
  std::lock_guard<std::mutex> l(m_);
  return state_ == State::kReady;
}

void Runtime::start() {
  {
    std::lock_guard<std::mutex> l(m_);
    if (state_ != State::kIdle) throw std::logic_error("splash runtime started twice");
    if (o_.argv.empty()) throw std::logic_error("splash runtime has no command");
    state_ = State::kStarting;
  }
  signal(SIGPIPE, SIG_IGN);  // a write to a dead child is an EPIPE this class handles, not a signal that ends the server

  int to_child[2], from_child[2];
  if (pipe(to_child) != 0 || pipe(from_child) != 0) throw EngineUnhealthy(std::string("pipe: ") + std::strerror(errno));
  for (int fd : {to_child[0], to_child[1], from_child[0], from_child[1]}) fcntl(fd, F_SETFD, FD_CLOEXEC);

  posix_spawn_file_actions_t fa;
  posix_spawnattr_t at;
  posix_spawn_file_actions_init(&fa);
  posix_spawnattr_init(&at);
  // Only what is named below reaches the child: stdin and stdout are the pipes, stderr is ours (as runtime.py's Popen).
  posix_spawnattr_setflags(&at, POSIX_SPAWN_CLOEXEC_DEFAULT);
  posix_spawn_file_actions_adddup2(&fa, to_child[0], 0);
  posix_spawn_file_actions_adddup2(&fa, from_child[1], 1);
  posix_spawn_file_actions_addinherit_np(&fa, 2);

  std::vector<std::string> envs;
  for (char** e = environ; e && *e; e++) {
    std::string kv = *e;
    bool overridden = false;
    for (const auto& add : o_.env) {
      auto eq = add.find('=');
      if (eq != std::string::npos && kv.compare(0, eq + 1, add, 0, eq + 1) == 0) overridden = true;
    }
    if (!overridden) envs.push_back(kv);
  }
  envs.insert(envs.end(), o_.env.begin(), o_.env.end());
  std::vector<char*> argv, envp;
  for (auto& a : o_.argv) argv.push_back(const_cast<char*>(a.c_str()));
  for (auto& e : envs) envp.push_back(const_cast<char*>(e.c_str()));
  argv.push_back(nullptr);
  envp.push_back(nullptr);

  pid_t child = -1;
  int rc = posix_spawn(&child, o_.argv[0].c_str(), &fa, &at, argv.data(), envp.data());
  posix_spawn_file_actions_destroy(&fa);
  posix_spawnattr_destroy(&at);
  close(to_child[0]);
  close(from_child[1]);
  if (rc != 0) {
    close(to_child[1]);
    close(from_child[0]);
    std::lock_guard<std::mutex> l(m_);
    state_ = State::kDead;
    dead_reason_ = std::string("cannot start ") + o_.argv[0] + ": " + std::strerror(rc);
    throw EngineUnhealthy(dead_reason_);
  }
  pid_ = child;
  in_fd_ = to_child[1];
  out_fd_ = from_child[0];
  say("splash engine pid " + std::to_string(pid_) + " started");
  reader_ = std::thread([this] { read_loop(); });

  std::unique_lock<std::mutex> l(m_);
  bool ok = cv_.wait_for(l, std::chrono::duration<double>(o_.ready_timeout_s), [&] { return state_ != State::kStarting; });
  if (ok && state_ == State::kReady) return;
  std::string why = ok ? dead_reason_ : "the engine was not ready within " + std::to_string(int(o_.ready_timeout_s)) + " s";
  l.unlock();
  if (!ok) kill(pid_, SIGKILL);
  shutdown();
  throw EngineUnhealthy(why);
}

void Runtime::write_frame(const std::vector<uint8_t>& f) {
  std::lock_guard<std::mutex> l(write_m_);
  if (in_fd_ < 0) throw EngineUnhealthy("the splash engine is not running");
  size_t off = 0;
  while (off < f.size()) {
    ssize_t n = write(in_fd_, f.data() + off, f.size() - off);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) throw EngineUnhealthy(std::string("write to the splash engine failed: ") + std::strerror(errno));
    off += size_t(n);
  }
}

uint64_t Runtime::submit(Request r, std::shared_ptr<Sink> sink, double deadline_s) {
  uint64_t id;
  {
    std::lock_guard<std::mutex> l(m_);
    if (state_ != State::kReady) throw EngineUnhealthy("the splash engine is not running" + (dead_reason_.empty() ? "" : ": " + dead_reason_));
    if (pending_.size() >= ready_.max_concurrent_requests) throw PendingLimitExceeded("the splash engine's request capacity is full");
    id = next_id_++;
  }
  r.request_id = id;
  r.remaining_deadline_micros = remaining_micros(deadline_s);
  uint64_t now = unix_micros();
  r.absolute_deadline_unix_micros = r.remaining_deadline_micros > UINT64_MAX - now ? UINT64_MAX : now + r.remaining_deadline_micros;
  std::vector<uint8_t> frame = encode_request(r);  // may throw ProtocolError: nothing is registered yet
  {
    std::lock_guard<std::mutex> l(m_);
    if (state_ != State::kReady) throw EngineUnhealthy("the splash engine is not running" + (dead_reason_.empty() ? "" : ": " + dead_reason_));
    pending_[id] = Pending{std::move(sink), false};  // before the write: the first event can follow it at once
  }
  try {
    write_frame(frame);
  } catch (...) {
    std::lock_guard<std::mutex> l(m_);
    pending_.erase(id);  // if the reader already failed it, it is gone and this is a no-op
    throw;
  }
  return id;
}

void Runtime::cancel(uint64_t id) {
  {
    std::lock_guard<std::mutex> l(m_);
    if (!pending_.count(id)) return;  // done already
  }
  try {
    write_frame(encode_cancel(id));
  } catch (const EngineUnhealthy&) {
    // dead: the reader ends the request
  }
}

std::string Runtime::status(double timeout_s) {
  uint64_t corr;
  {
    std::lock_guard<std::mutex> l(m_);
    if (state_ != State::kReady) return "";
    corr = next_corr_++;
    status_replies_[corr];
  }
  try {
    write_frame(encode_status_request(corr));
  } catch (const EngineUnhealthy&) {
    std::lock_guard<std::mutex> l(m_);
    status_replies_.erase(corr);
    return "";
  }
  std::unique_lock<std::mutex> l(m_);
  cv_.wait_for(l, std::chrono::duration<double>(timeout_s), [&] { return !status_replies_[corr].empty() || state_ == State::kDead; });
  std::string out = std::move(status_replies_[corr]);
  status_replies_.erase(corr);
  return out;
}

void Runtime::fail_all(const std::string& code, const std::string& message) {
  std::map<uint64_t, Pending> held;
  {
    std::lock_guard<std::mutex> l(m_);
    held.swap(pending_);
    if (state_ != State::kDead) {
      state_ = State::kDead;
      dead_reason_ = code + ": " + message;
    }
  }
  cv_.notify_all();
  for (auto& [id, p] : held) p.sink->on_error(unhealthy(id, code, message));
}

void Runtime::reap(double grace_s) {
  pid_t pid = pid_;
  if (pid <= 0) return;
  int st = 0;
  auto until = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(grace_s));
  while (true) {
    pid_t r = waitpid(pid, &st, WNOHANG);
    if (r == pid || (r < 0 && errno != EINTR)) break;
    if (Clock::now() >= until) {
      kill(pid, SIGKILL);
      waitpid(pid, &st, 0);
      break;
    }
    usleep(10000);
  }
  if (WIFEXITED(st)) say("splash engine exited with status " + std::to_string(WEXITSTATUS(st)));
  else if (WIFSIGNALED(st)) say("splash engine killed by signal " + std::to_string(WTERMSIG(st)));
  pid_ = -1;
}

void Runtime::read_loop() {
  FrameParser parser;
  std::vector<uint8_t> buf(1 << 16);
  std::string code = "engine_exited", message = "the splash engine closed its output";
  bool fatal = false;
  try {
    while (true) {
      ssize_t n = read(out_fd_, buf.data(), buf.size());
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) break;
      parser.feed(buf.data(), size_t(n));
      Event ev;
      while (parser.next(ev)) {
        // Handle one event. A Sink runs here, outside every lock.
        if (auto* e = std::get_if<Ready>(&ev)) {
          std::lock_guard<std::mutex> l(m_);
          if (state_ != State::kStarting) throw ProtocolError(Issue{FailureClass::kProtocolFatal, IssueCode::kInvalidEnumValue, 0, "a second ready event"});
          ready_ = *e;
          state_ = State::kReady;
          cv_.notify_all();
          continue;
        }
        {
          std::lock_guard<std::mutex> l(m_);
          if (state_ == State::kStarting) throw ProtocolError(Issue{FailureClass::kProtocolFatal, IssueCode::kInvalidEnumValue, 0, "an event before ready"});
        }
        if (auto* e = std::get_if<StatusJson>(&ev)) {
          std::lock_guard<std::mutex> l(m_);
          auto it = status_replies_.find(e->correlation_id);
          if (it != status_replies_.end()) { it->second = e->json; cv_.notify_all(); }
          continue;
        }
        if (std::holds_alternative<MaskRequest>(ev))
          throw ProtocolError(Issue{FailureClass::kProtocolFatal, IssueCode::kInvalidConstraint, std::get<MaskRequest>(ev).request_id,
                                    "the engine asked for a token mask on an unconstrained request"});
        if (auto* e = std::get_if<Error>(&ev); e && e->failure != FailureClass::kRequestError) {
          code = e->code;
          message = e->message;
          fatal = true;
          break;
        }
        uint64_t id = std::visit([](auto& x) -> uint64_t {
          using T = std::decay_t<decltype(x)>;
          if constexpr (std::is_same_v<T, Ready> || std::is_same_v<T, StatusJson>) return 0;
          else return x.request_id;
        }, ev);
        std::shared_ptr<Sink> sink;
        bool terminal = std::holds_alternative<Done>(ev) || std::holds_alternative<Error>(ev);
        {
          std::lock_guard<std::mutex> l(m_);
          auto it = pending_.find(id);
          if (it == pending_.end()) { say("splash engine event for unknown request " + std::to_string(id)); continue; }
          sink = it->second.sink;
          if (std::holds_alternative<Start>(ev)) it->second.started = true;
          if (terminal) pending_.erase(it);
        }
        std::visit([&](auto& x) {
          using T = std::decay_t<decltype(x)>;
          if constexpr (std::is_same_v<T, Start>) sink->on_start(x);
          else if constexpr (std::is_same_v<T, PromptProgress>) sink->on_progress(x);
          else if constexpr (std::is_same_v<T, Tokens>) sink->on_tokens(x);
          else if constexpr (std::is_same_v<T, Done>) sink->on_done(x);
          else if constexpr (std::is_same_v<T, Error>) sink->on_error(x);
        }, ev);
        cv_.notify_all();  // a slot is free for a waiting submit-er
      }
      if (fatal) break;
    }
  } catch (const ProtocolError& e) {
    code = "protocol_error";
    message = e.what();
    fatal = true;
    say(std::string("splash protocol failure: ") + e.what());
  } catch (const std::exception& e) {
    code = "internal_error";
    message = e.what();
    fatal = true;
  }
  if (fatal && pid_ > 0) kill(pid_, SIGKILL);  // an untrusted stream: do not read more of it
  reap(fatal ? 5 : std::max(0.1, o_.stop_grace_s));
  bool stopping;
  { std::lock_guard<std::mutex> l(m_); stopping = stopping_; }
  if (stopping) { code = "engine_stopped"; message = "the splash engine was shut down"; }
  fail_all(code, message);
}

void Runtime::shutdown() {
  {
    std::lock_guard<std::mutex> l(m_);
    if (state_ == State::kIdle) return;
    stopping_ = true;
  }
  {
    std::lock_guard<std::mutex> l(write_m_);
    if (in_fd_ >= 0) { close(in_fd_); in_fd_ = -1; }  // EOF on its stdin: it drains and exits
  }
  if (reader_.joinable()) {
    std::unique_lock<std::mutex> l(m_);
    auto grace = std::chrono::duration<double>(o_.stop_grace_s);
    if (!cv_.wait_for(l, grace, [&] { return state_ == State::kDead; }) && pid_ > 0) {
      say("splash engine did not exit: SIGTERM");
      kill(pid_, SIGTERM);
      if (!cv_.wait_for(l, grace, [&] { return state_ == State::kDead; }) && pid_ > 0) kill(pid_, SIGKILL);
    }
    l.unlock();
    reader_.join();
  }
  if (out_fd_ >= 0) { close(out_fd_); out_fd_ = -1; }
}

}  // namespace ni::splash
