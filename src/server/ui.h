// The command line's screens in a terminal (term::fancy()): the home screen, styled --help, `kapitsa serve`'s loading,
// ready card and request lines, list, pull, rm, and the coding agents' launcher. Output that is not a terminal keeps
// the plain text of main.cpp and launch.cpp.
#pragma once
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
struct Ready {
  std::string model, quant, base, api_key_note, log;
  size_t budget = 0, used = 0;
  int context = 0, context_fit = 0, slots = 0;
};
// The card once the server listens
void ready(const Ready& r);

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
