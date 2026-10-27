#include "server/native_execution.h"
#include "engine/execution_session_internal.h"
#include "server/model_manager.h"

#include <stdexcept>

namespace ni {

std::shared_ptr<ExecutionSession> native_execution(std::shared_ptr<RequestGuard> guard,
                                                std::shared_ptr<detail::ChatObserver> observer, NativeReplyMemory memory) {
  if (!guard) throw std::invalid_argument("native execution requires a request guard");
  return detail::ExecutionFactory::bind(
      [guard = std::move(guard), memory](const ChatRequest& r, detail::ChatObserver& events, detail::StopToken stop) {
        auto& engine = guard->engine();
        // the engine is bound to one backend when it is built: Splash's child process (chat_splash) or this process's
        // model (chat, which refuses in Splash mode); no request moves from one to the other
        const auto run = engine.splash_mode() ? &Engine::chat_splash : &Engine::chat;
        auto out = (engine.*run)(r,
            [&](const ChatDelta& d) { return events.delta(d); },
            [&](const ChatResult& prompt) { events.prompt(prompt); },
            [stop] { return stop.stop_requested(); },
            [&](const ChatResult& prompt) { return events.accepted(prompt, stop); });
        if (memory == NativeReplyMemory::BeforeTerminal) engine.remember_reply(r, out);
        return out;
      }, std::move(observer));
}

}  // namespace ni
