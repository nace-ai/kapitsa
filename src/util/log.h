#pragma once
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace ni {

// Only the SDK's synchronous engine construction opts in. A worker must explicitly
// inherit this setting and return its exception to the joining thread (LoadThreads).
class FatalRecovery {
 public:
  explicit FatalRecovery(bool enable = true) : previous_(enabled()) { enabled() = enable; }
  ~FatalRecovery() { enabled() = previous_; }
  FatalRecovery(const FatalRecovery&) = delete;
  FatalRecovery& operator=(const FatalRecovery&) = delete;
  static bool& enabled() { static thread_local bool value = false; return value; }
 private:
  bool previous_;
};

class FatalError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

inline double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Feature flags: unset, empty or "0" means off.
inline bool env_on(const char* name) {
  const char* v = getenv(name);
  return v && *v && !(v[0] == '0' && v[1] == 0);
}

inline int& log_level() {
  static int lvl = getenv("NI_LOG") ? atoi(getenv("NI_LOG")) : 1;
  return lvl;
}

// Where logi, logd and fatal lines go. None (the default): stderr, as "[kapitsa] <message>" lines (what the Nace Agent
// app, the bench harness and `kapitsa claude`'s server log read). `kapitsa serve` in a terminal sets one
// (src/server/ui.cpp): a log file, with warnings and errors on screen. `level`: 0 fatal, 1 logi, 2 logd; `msg` has
// no "[kapitsa] " prefix and no newline. Called from any thread.
using LogSink = void (*)(int level, const char* msg);
inline LogSink& log_sink() {
  static LogSink s = nullptr;
  return s;
}

inline void log_to_sink(LogSink s, int level, const char* fmt, va_list ap) {
  char buf[1024];
  va_list aq;
  va_copy(aq, ap);
  const int n = vsnprintf(buf, sizeof buf, fmt, aq);
  va_end(aq);
  if (n < int(sizeof buf)) {
    s(level, buf);
    return;
  }
  std::string big(size_t(n) + 1, '\0');
  vsnprintf(big.data(), big.size(), fmt, ap);
  big.resize(size_t(n));
  s(level, big.c_str());
}

// Exits with status 1 (NI_ABORT=1: abort() instead, for a backtrace in a debugger or crash report). Most fatal
// errors are the user's to fix (a missing file, not enough memory), and abort()'s "Abort trap" and crash report
// would only bury the message.
[[noreturn]] __attribute__((format(printf, 1, 2))) inline void fatal(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  if (FatalRecovery::enabled()) {
    char local[1024];
    va_list aq;
    va_copy(aq, ap);
    const int n = vsnprintf(local, sizeof local, fmt, aq);
    va_end(aq);
    std::string message;
    try {
      if (n < 0) message = "could not format engine error";
      else if (n < int(sizeof local)) message = local;
      else {
        message.resize(size_t(n) + 1);
        vsnprintf(message.data(), message.size(), fmt, ap);
        message.resize(size_t(n));
      }
    } catch (...) { va_end(ap); throw; }
    va_end(ap);
    throw FatalError(message);
  }
  if (LogSink s = log_sink()) {
    log_to_sink(s, 0, fmt, ap);
  } else {
    fprintf(stderr, "[kapitsa] fatal: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
  }
  va_end(ap);
  if (env_on("NI_ABORT")) abort();
  fflush(stdout);
  _exit(1);
}

__attribute__((format(printf, 1, 2))) inline void logi(const char* fmt, ...) {
  if (log_level() < 1) return;
  va_list ap;
  va_start(ap, fmt);
  if (LogSink s = log_sink()) {
    log_to_sink(s, 1, fmt, ap);
    va_end(ap);
    return;
  }
  fprintf(stderr, "[kapitsa] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

__attribute__((format(printf, 1, 2))) inline void logd(const char* fmt, ...) {
  if (log_level() < 2) return;
  va_list ap;
  va_start(ap, fmt);
  if (LogSink s = log_sink()) {
    log_to_sink(s, 2, fmt, ap);
    va_end(ap);
    return;
  }
  fprintf(stderr, "[kapitsa] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

} // namespace ni
