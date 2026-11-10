// The engine view (src/server/engine.html, docs/control-socket.md "The engine view"): a page kapitsa serve serves at
// /engine, and a mirror of the control socket under /engine/ (a browser cannot open the Unix socket): GET /status,
// GET /events, POST /stop and POST /model, answered by the socket's own handlers.
//
// The socket admits its own user only (getpeereid). The mirror keeps that with a key made for this run and kept in a
// file only that user can read ($KAPITSA_HOME/run/<instance>.token, 0600 in the 0700 run/); kapitsa serve prints the
// page's link with it. Every path but the page needs the key, the API key opens none of them. And it answers this Mac only: a loopback peer, a loopback Host, no other site's Origin, nothing
// a proxy adds.
#pragma once

#include "server/control.h"

#include <httplib/httplib.h>

#include <functional>
#include <optional>
#include <string>

namespace ni::server {

// At most this many GET /engine/events at once: each holds one of httplib's threads (max(8, cores - 1)), which the
// API's requests share
constexpr int kMaxEngineViews = 3;

// This process's key for the engine view: 24 random bytes, base64url (32 characters), made on first use
const std::string& engine_token();

// Writes engine_token() to `path`, mode 0600, replacing a stale file (a link is not followed). "" or why not.
std::string write_engine_token(const std::string& path);

// Why the engine view refuses this request before its key is looked at, or "": a peer other than this Mac, a Host that
// is not a loopback name (a page elsewhere that rebinds its name to 127.0.0.1), an Origin that is not the page's own, a
// header a proxy adds (Forwarded, X-Forwarded-For, Via), or a body over 64 KiB
std::string engine_refusal(const httplib::Request& req);

// For the server's pre-routing handler, ahead of its API key check. Not an /engine path: nullopt. Refused (above), a
// path that is not the view's (404), a method other than GET or HEAD on the page, status and events or POST on stop
// and model (405), no `Authorization: Bearer <key>` with this run's key (403, all but the page), or a stop or switch
// whose Content-Type is not application/json (400): the JSON error in `res`, Handled. Otherwise Unhandled: the run's
// key stands in for the API key, which the page does not have. The page itself needs no key: it holds no data.
std::optional<httplib::Server::HandlerResponse> engine_gate(const httplib::Request& req, httplib::Response& res);

// GET /engine (`page`), GET /engine/status, GET /engine/events, POST /engine/stop and POST /engine/model, answered as
// the control socket answers them. `control()` is null while the server has no control socket: a 503.
void engine_routes(httplib::Server& srv, std::function<ControlServer*()> control, const char* page);

}  // namespace ni::server
