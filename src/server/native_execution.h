#pragma once
#include "engine/execution_session_internal.h"

namespace ni {
class RequestGuard;

// Streaming remembers before its end frame; nonstream routes keep ModelManager's existing ordering.
enum class NativeReplyMemory { Manager, BeforeTerminal };
std::shared_ptr<ExecutionSession> native_execution(std::shared_ptr<RequestGuard> guard,
                                                std::shared_ptr<detail::ChatObserver> observer,
                                                NativeReplyMemory memory = NativeReplyMemory::Manager);

}  // namespace ni
