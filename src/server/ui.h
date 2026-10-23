// The command line's screens in a terminal (term::fancy()): the home screen, styled --help, `kapitsa serve`'s loading,
// ready card and request lines, list, pull, rm, and the coding agents' launcher. Output that is not a terminal keeps
// the plain text of main.cpp and launch.cpp.
#pragma once
#include "cluster/cluster.h"
#include "engine/engine.h"
#include "hub/hub.h"

#include <httplib/httplib.h>

#include <memory>
#include <string>

namespace ni::ui {

// `kapitsa`, `kapitsa --help`
void home(const std::string& version);
// A command's --help, from its plain text: usage, options and their descriptions, examples, wrapped to the terminal
void help(const std::string& plain);
// `kapitsa list`
void list();
// `kapitsa pull --dry-run MODEL` (not --json); a GGUF variant's import on disk (hub::import_size; 0: not planned)
void plan(const hub::ModelRef& r, uint64_t imported_bytes = 0);

// The engine's log lines are left out of a screen (they are for `kapitsa serve`'s log); fatal errors still show
void quiet();
// Lines: "✓ text" (the accent mark), "› text" (dim), "! text" (yellow mark), on stderr unless `out`
void ok(const std::string& s, bool out = false);
void note(const std::string& s, bool out = false);
void warning(const std::string& s);

// ---- kapitsa serve

// Before anything else: the engine's log goes to memory, and from serve_log_file() on to
// ~/.cache/kapitsa/logs/serve-<port>.log (and on screen too, dim, with `verbose` or NI_LOG); warnings and errors show on
// screen either way
void serve_begin(bool verbose);
// The header, once the server will start (a mistyped command shows its error alone)
void serve_header(const std::string& version);
// Once the port is known to be free (another server's log is not cut): the log file, with what came before. Its path.
std::string serve_log_file(int port);
// A spinner while the engine loads, showing its latest log line; then "✓ Loaded <what> in 6.1s"
void loading_begin(const std::string& what);
void loading_end(const std::string& what);
// The server on the network (several Macs, docs/usage.md)
struct Network {
  bool visible = false;  // advertised to Kapitsa on the other Macs (not --private, not a loopback --host)
  std::string name;      // as they see it: this Mac's computer name
  std::string url;       // http://<this Mac's address>:<port>
  bool api_key = false;  // the API needs its key
  int others = -1;       // other Macs found running Kapitsa (-1: not looked)
};
// The line about it under the card: "Visible to Kapitsa on this network as ... (no password; --private hides it)"
std::string network_line(const Network& n);
struct Ready {
  std::string model, quant, base, api_key_note, log;
  size_t budget = 0, used = 0;
  int context = 0, context_fit = 0, slots = 0;
  Network net;
};
// The card once the server listens
void ready(const Ready& r);
// The card of a server that sends its requests to another Mac (route.h), once it listens
struct Routed {
  std::string model;  // the model as that Mac serves it ("qwen3.8-flash-next:4bit")
  cluster::MacInfo mac;
  std::string base, api_key_note, log;
  Network net;
};
void routed(const Routed& r);
// "qwen3.8-flash-next runs on Studio (M3 Ultra, 512 GB) · Wi-Fi 4 ms: this Mac sends its requests there. --local runs
// it here."
std::string route_line(const std::string& model, const cluster::MacInfo& mac);
// One request sent on to the other Mac: its line in the request log (the log file without the terminal UI)
void forwarded(const httplib::Request& req, int status, double secs, const std::string& mac);
// The decode rate of the latest request (tokens/s; 0 before any), which the server tells the other Macs
double recent_tok_s();

// One API request, for its line in the request log: the client (from the User-Agent), the time from its arrival to
// its first token, the images it carries. nullptr when the terminal UI is off; the functions take that too.
struct Meter;
std::shared_ptr<Meter> meter(const httplib::Request& req, const ChatRequest& r);
// What the engine updates while it writes the request (ChatRequest::live), for the dashboard; null without a meter
std::shared_ptr<LiveDecode> live_of(const std::shared_ptr<Meter>& m);
void token(const std::shared_ptr<Meter>& m);
void done(const std::shared_ptr<Meter>& m, const ChatResult& r);
// What the request log shows of the time before the first token (`--timing`): "first" (time to the first token), "read/s"
// (prefill tokens per second, the uncached prompt tokens only) or both (the default). False when `mode` is none of them.
bool set_timing(const std::string& mode);

}  // namespace ni::ui
