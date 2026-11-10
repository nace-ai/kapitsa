// What every request to kapitsa serve's HTTP port meets before its route, in order: a WebSocket upgrade on
// /v1/responses refused (426), the engine view's gate (src/server/engine_view.h), then the API key. Its own unit so
// tools/engine_view_test.cpp runs the server's order, not a copy of it.
#pragma once
#include <httplib/httplib.h>

#include <string>

namespace ni::server {

httplib::Server::HandlerResponse pre_route(const httplib::Request& req, httplib::Response& res,
                                           const std::string& api_key);

}  // namespace ni::server
