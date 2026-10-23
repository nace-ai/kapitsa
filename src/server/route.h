// Several Macs in `kapitsa serve` (docs/usage.md, "Several Macs"): the name a model goes by between Macs, whether this
// Mac holds a model whole, and the server that sends its requests to a Mac on the network that runs its model whole.
#pragma once
#include "cluster/discovery.h"
#include "engine/engine.h"
#include "hub/hub.h"

#include <httplib/httplib.h>

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace ni {

// The name a model goes by between Macs: "qwen3.8-flash-next:4bit" for a catalog model, also for a folder named as the
// store names one (<catalog name>-<variant>: a Q2_0 pack in qwen3.8-flash-next-q2_0 is "qwen3.8-flash-next:q2_0");
// else its API id
std::string model_key(const hub::ModelRef& ref);

// This Mac holds `ref` whole: the engine would start it within `budget` (start_need, as serve checks it before a
// download) and no routed experts would stream from the SSD. False when it cannot be planned (not downloaded and the
// Hub out of reach); true for a folder that cannot be (serve then says what is wrong with it).
bool runs_whole_here(const hub::ModelRef& ref, const EngineOptions& opt, size_t budget);

// Requests sent on to a Mac that runs the model whole: every /v1 request forwarded as it came (headers, API key
// included), the reply relayed as it comes back (an SSE stream chunk by chunk). The Mac is picked for each request from
// what the browser sees (cluster::pick_server); while none serves the model, a request gets 503 and says why.
class Router {
 public:
  // head_id: the server whose stage this Mac is (pair.h): requests go there while it answers, whole or not
  Router(std::shared_ptr<cluster::Browser> browser, std::string model, std::string head_id = "");
  // Where requests go now, if anywhere
  std::optional<cluster::MacInfo> peer() const;
  // The Mac requests went to last (for messages while it is away)
  cluster::MacInfo last() const;
  const std::string& model() const { return model_; }
  // Forwards GET and POST /v1/... on `srv`; `done` gets each forwarded request's status, seconds and Mac ("" none)
  using Done = std::function<void(const httplib::Request&, int status, double secs, const std::string& mac)>;
  void add_routes(httplib::Server& srv, Done done);
  // What a request gets while no Mac serves the model
  std::string away() const;

 private:
  void forward(const httplib::Request& req, httplib::Response& res, const Done& done);
  std::shared_ptr<cluster::Browser> browser_;
  std::string model_, head_id_;
  mutable std::mutex mu_;
  mutable cluster::MacInfo last_;
};

}  // namespace ni
