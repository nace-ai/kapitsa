#pragma once
#include "engine/execution_session.h"

#include <atomic>
#include <functional>

namespace ni::detail {

struct StopToken {
  const std::shared_ptr<const std::atomic<bool>> stopped;
  bool stop_requested() const noexcept { return stopped->load(); }
};

class ChatObserver {
 public:
  virtual ~ChatObserver() = default;
  virtual bool accepted(const ChatResult&, StopToken) { return true; }
  virtual void prompt(const ChatResult&) {}
  virtual bool delta(const ChatDelta&) { return true; }
  virtual void cancelled() noexcept {}
  virtual void finished(const ChatResult&, StopToken) {}
};

struct ExecutionFactory {
  using Work = std::function<ChatResult(const ChatRequest&, ChatObserver&, StopToken)>;
  static std::shared_ptr<ExecutionSession> bind(Work work, std::shared_ptr<ChatObserver> observer);
};

}
