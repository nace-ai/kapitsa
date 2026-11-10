// The engine view (src/server/engine.html, docs/control-socket.md "The engine view"): a page kapitsa serve serves at
// /engine, and its mirror of the control socket's endpoints over HTTP under /engine/ (a browser cannot open the Unix
// socket). This Mac only, by peer address and by Host.
#pragma once

#include "server/control.h"

#include <httplib/httplib.h>

#include <functional>
#include <optional>
#include <string>

namespace ni::server {

// At most this many GET /engine/events at once: each holds one of httplib's threads, which the API's requests share
constexpr int kMaxEngineViews = 4;

// Why the engine view refuses this request, or "": a peer other than this Mac, a Host that is not a loopback name (a
// page elsewhere that rebinds its name to 127.0.0.1), a POST that is not JSON (a cross-origin form cannot send JSON)
std::string engine_refusal(const httplib::Request& req);

// For the server's pre-routing handler, ahead of its API key check. Not an /engine path: nullopt. Refused: a 403 in
// `res`, Handled. A GET: Unhandled, without the key (an EventSource sends no headers; the page shows what the control
// socket shows). A POST (stop, switch): nullopt, so the key check runs as for the API.
std::optional<httplib::Server::HandlerResponse> engine_gate(const httplib::Request& req, httplib::Response& res);

// GET /engine (`page`), GET /engine/status, GET /engine/events, POST /engine/stop and POST /engine/model, answered as
// the control socket answers them. `control()` is null while the server has no control socket: a 503.
void engine_routes(httplib::Server& srv, std::function<ControlServer*()> control, const char* page);

}  // namespace ni::server
