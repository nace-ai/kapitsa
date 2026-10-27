#include "engine/execution_session_internal.h"

#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace ni {
namespace {
thread_local const void* active_execution = nullptr;
using detail::ChatObserver;
using detail::StopToken;
}

struct ExecutionSession::State {
  enum class Phase { Ready, Running, Joined };
  const detail::ExecutionFactory::Work work;
  std::optional<ChatRequest> request;
  const std::shared_ptr<ChatObserver> observer;
  ChatResult result;
  std::string emergency_reason = "error", emergency_error = "execution failed";
  const std::shared_ptr<std::atomic<bool>> stopped = std::make_shared<std::atomic<bool>>(false);
  std::exception_ptr callback_error;
  std::mutex join_mu, terminal_mu;
  std::atomic<bool> terminal{false};
  std::thread worker;
  Phase phase = Phase::Ready;

  State(detail::ExecutionFactory::Work w, std::shared_ptr<ChatObserver> events)
      : work(std::move(w)), observer(std::move(events)) {}

  StopToken token() const noexcept { return {stopped}; }
  void cancel() noexcept {
    if (stopped->exchange(true) || terminal.load()) return;
    std::lock_guard<std::mutex> lk(terminal_mu);
    if (terminal.load()) return;
    const auto previous = active_execution;
    active_execution = this;
    observer->cancelled();
    active_execution = previous;
  }

  struct Callbacks : ChatObserver {
    State& s;
    explicit Callbacks(State& state) : s(state) {}
    template <class F> bool deliver(F f) {
      if (s.stopped->load()) return false;
      try {
        if (f()) return true;
      } catch (...) {
        s.callback_error = std::current_exception();
      }
      s.cancel();
      return false;
    }
    bool accepted(const ChatResult& r, StopToken token) override {
      return deliver([&] { return s.observer->accepted(r, token); });
    }
    void prompt(const ChatResult& r) override {
      deliver([&] { s.observer->prompt(r); return true; });
    }
    bool delta(const ChatDelta& d) override { return deliver([&] { return s.observer->delta(d); }); }
  };

  void failure(std::exception_ptr error) noexcept {
    if (result.finish_reason != "error") result.finish_reason.swap(emergency_reason);
    if (result.error.empty()) result.error.swap(emergency_error);
    try {
      std::rethrow_exception(error);
    } catch (const std::exception& e) {
      try { if (*e.what()) result.error = e.what(); } catch (...) {}
    } catch (...) {}
  }

  void finish() noexcept {
    std::lock_guard<std::mutex> lk(terminal_mu);
    terminal = true;
    try { observer->finished(result, token()); }
    catch (...) { failure(std::current_exception()); }
  }

  void run() noexcept {
    active_execution = this;
    try {
      if (stopped->load()) result.finish_reason = "cancelled";
      else {
        Callbacks callbacks(*this);
        result = work(*request, callbacks, token());
      }
    } catch (...) {
      failure(std::current_exception());
    }
    if (callback_error) failure(callback_error);
    finish();
    active_execution = nullptr;
  }
};

ExecutionSession::ExecutionSession(std::unique_ptr<State> state) : state_(std::move(state)) {}
ExecutionSession::~ExecutionSession() {
  if (active_execution == state_.get()) std::terminate();
  if (state_->phase == State::Phase::Running) cancel();
  std::lock_guard<std::mutex> lk(state_->join_mu);
  if (state_->worker.joinable()) state_->worker.join();
}

void ExecutionSession::start(ChatRequest request) {
  if (active_execution == state_.get()) throw std::logic_error("execution cannot start itself");
  auto& s = *state_;
  std::lock_guard<std::mutex> lk(s.join_mu);
  if (s.phase != State::Phase::Ready) throw std::logic_error("execution already started");
  s.request.emplace(std::move(request));
  try {
    s.worker = std::thread([state = &s] { state->run(); });
    s.phase = State::Phase::Running;
  } catch (...) {
    s.failure(std::current_exception());
    const auto previous = active_execution;
    active_execution = &s;
    s.finish();
    active_execution = previous;
    s.phase = State::Phase::Joined;
  }
}

void ExecutionSession::cancel() noexcept { state_->cancel(); }

const ChatResult& ExecutionSession::wait() {
  auto& s = *state_;
  if (active_execution == &s) throw std::logic_error("execution cannot wait on its own callback");
  std::lock_guard<std::mutex> lk(s.join_mu);
  if (s.phase == State::Phase::Ready) throw std::logic_error("execution has not started");
  if (s.worker.joinable()) s.worker.join();
  s.phase = State::Phase::Joined;
  return s.result;
}

std::shared_ptr<ExecutionSession> detail::ExecutionFactory::bind(Work work, std::shared_ptr<ChatObserver> observer) {
  if (!work || !observer) throw std::invalid_argument("execution requires bound work and observer");
  return std::shared_ptr<ExecutionSession>(new ExecutionSession(
      std::make_unique<ExecutionSession::State>(std::move(work), std::move(observer))));
}

}
