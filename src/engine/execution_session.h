#pragma once
#include "engine/chat.h"

#include <memory>

namespace ni {
namespace detail { struct ExecutionFactory; }

class ExecutionSession final {
 public:
  ~ExecutionSession();
  ExecutionSession(const ExecutionSession&) = delete;
  ExecutionSession& operator=(const ExecutionSession&) = delete;
  // One-shot start belongs to the control owner, as does destruction. Only cancel/wait may race after start.
  void start(ChatRequest request);
  void cancel() noexcept;  // latched before start, idempotent, callback-safe
  const ChatResult& wait();  // repeatable synchronized join; self-wait and wait-before-start throw

 private:
  struct State;
  explicit ExecutionSession(std::unique_ptr<State> state);
  friend struct detail::ExecutionFactory;
  std::unique_ptr<State> state_;
};

}
