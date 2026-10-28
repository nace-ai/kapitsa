#pragma once
#include "util/log.h"
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace ni {

// Joined loading/conversion jobs: a failure must not escape a thread entry point,
// and even failure to start a later thread must join the jobs already using buffers.
class LoadThreads {
 public:
  ~LoadThreads() { join_all(); }
  template<class F, class... A> void emplace_back(F&& f, A&&... args) {
    threads_.emplace_back([this, recover = FatalRecovery::enabled(),
                           work = std::bind(std::forward<F>(f), std::forward<A>(args)...)]() mutable {
      FatalRecovery scope(recover);
      try { work(); }
      catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!error_) error_ = std::current_exception();
      }
    });
  }
  void join() {
    join_all();
    if (error_) std::rethrow_exception(error_);
  }
 private:
  void join_all() { for (auto& t : threads_) if (t.joinable()) t.join(); }
  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::exception_ptr error_;
};

}  // namespace ni
