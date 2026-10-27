// Server-private HTTP framing. Native binding and endpoint JSON errors stay with the caller.
#pragma once
#include "engine/execution_session_internal.h"

#include <functional>
#include <memory>

namespace httplib { struct Response; }
namespace ni::api { class StreamWriter; }

namespace ni::server {

using BindExecution = std::function<std::shared_ptr<ExecutionSession>(std::shared_ptr<detail::ChatObserver>)>;
struct StreamMeter {
  std::function<void()> token;
  std::function<void(const ChatResult&)> done;
};

// Meter exceptions are contained. token runs on the worker; done is attempted once on the connection thread after join.
// failure owns the pre-stream response. Otherwise the response owns the provider and its execution.
void stream_reply(const ChatRequest& request, std::shared_ptr<api::StreamWriter> writer,
                  httplib::Response& response, BindExecution bind,
                  std::function<void(const ChatResult&, httplib::Response&)> failure,
                  std::function<void(const ChatResult&)> after, StreamMeter meter,
                  bool early = false, bool telemetry = false);

}  // namespace ni::server
