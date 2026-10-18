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
// `kapitsa pull --dry-run MODEL` (not --json)
void plan(const hub::ModelRef& r);

// The engine's log lines are left out of a screen (they are for `kapitsa serve`'s log); fatal errors still show
void quiet();
// Lines: "✓ text" (the accent mark), "› text" (dim), "! text" (yellow mark), on stderr unless `out`
void ok(const std::string& s, bool out = false);
void note(const std::string& s, bool out = false);
void warning(const std::string& s);
// "kapitsa: <msg>" errors: a red ✗ with the problem and the fix on its own dim line
int fail(const std::string& msg, int code);

// ---- kapitsa serve

// From here on, the engine's log goes to ~/.cache/kapitsa/logs/serve-<port>.log (and on screen too, dim, with
// `verbose`); warnings and errors show on screen either way. Returns the log's path.
std::string serve_log(int port, bool verbose);
// The header line of `kapitsa serve`
void serve_header(const std::string& version);
// A spinner while the engine loads, showing its latest log line; done() with its load time
void loading_begin(const std::string& what);
void loading_end(bool ok);
struct Ready {
  std::string model, variant, quant, base, api_key_note, log;
  size_t budget = 0;
  int context = 0, context_fit = 0, slots = 0;
};
// The card once the server listens
void ready(const Ready& r);

// One API request, for its line in the request log: the client (from the User-Agent), the time from its arrival to
// its first token, the images it carries. nullptr when the terminal UI is off; the functions take that too.
struct Meter;
std::shared_ptr<Meter> meter(const httplib::Request& req, const ojson& messages);
std::shared_ptr<Meter> meter(const httplib::Request& req, const ChatRequest& r);
void token(const std::shared_ptr<Meter>& m);
void done(const std::shared_ptr<Meter>& m, const ChatResult& r);

}  // namespace ni::ui
