// The command line's terminal look, without dependencies: whether output is styled, colors, the width, UTF-8
// display widths, text layout, a status line (spinner, progress bar) and styled errors.
//
// Styled output needs stdout and stderr to be terminals. Anything else (a pipe, a file: the Nace Agent app, the bench
// harness, tests, users' scripts) gets the plain text the commands always printed, byte for byte: every caller checks
// fancy() and keeps its plain branch as it was. In a terminal:
// - colors: truecolor when COLORTERM is truecolor or 24bit (or the terminal is known to have it), 256 when TERM says
//   so, else the 16 ANSI colors; none with NO_COLOR (or CLICOLOR=0) or TERM=dumb. FORCE_COLOR (1, 2, 3) or
//   CLICOLOR_FORCE turn colors on despite those and set their minimum depth; they never style a pipe.
// - TERM=dumb: no styling, unless forced; even then nothing redraws in place (no spinner, no live progress).
// - the accent is Nace green, brighter on dark backgrounds (#3FB55A) and the brand's own (#308442) on light ones; the
//   background comes from COLORFGBG or the terminal's answer to an OSC 11 query (only on a local terminal of the
//   foreground job, fenced by a DA1 query so no reply is left behind), else colors that read on both.
#pragma once
#include <cstdio>
#include <string>
#include <vector>

namespace ni::term {

// The styled look is on (both streams are terminals)
bool fancy();
// A status line may redraw itself in place (fancy() and TERM is not dumb)
bool live();
// Colors are on (fancy() and not NO_COLOR / dumb, unless forced)
bool color();
// The terminal's width in columns (TIOCGWINSZ, else COLUMNS, else 80); read again on each call
int columns();

// Styles: `s` wrapped in its escape codes, or `s` as is when styling is off. They nest, except bold inside dim (or
// the reverse): both end with the same code.
std::string bold(const std::string& s);
std::string dim(const std::string& s);
std::string italic(const std::string& s);
std::string accent(const std::string& s);  // Nace green: success, the marks, what is ready
std::string warn(const std::string& s);    // yellow
std::string red(const std::string& s);     // errors
std::string badge(const std::string& s);   // the wordmark: bold white on the brand green

// Glyphs (one column each)
extern const char* const kOk;       // ✓
extern const char* const kFail;     // ✗
extern const char* const kOn;       // ●
extern const char* const kOff;      // ○
extern const char* const kStar;     // ★
extern const char* const kArrow;    // ›
extern const char* const kDot;      // ·
extern const char* const kWarn;     // !

// Display width in columns: UTF-8 aware (wide East Asian characters and emoji take 2, combining marks 0) and blind to
// escape sequences
int width(const std::string& s);
// `s` padded with spaces to `w` columns (left-aligned / right-aligned); never shortened
std::string left(const std::string& s, int w);
std::string right(const std::string& s, int w);
// `s` cut to at most `w` columns, ending in "…" when cut; escape sequences are kept (and closed)
std::string clip(const std::string& s, int w);
// Plain text wrapped at spaces into lines of at most `w` columns (a longer word is split)
std::vector<std::string> wrap(const std::string& text, int w);

// "0.4s", "12s", "3m 05s", "1h 02m"
std::string duration(double seconds);
// 812, 29.6K, 1.2M
std::string count(double n);
// A context size: 32K, 128K (a multiple of 1024 counts in K of 1024, as contexts are named), else as count()
std::string context(int n);

// Output that coexists with a status line: the status line is cleared, the line written (with a newline) to `f`,
// and the status line drawn again below it. Thread-safe; every styled line should go through it.
void line(const std::string& s, FILE* f = stdout);
// Sets the status line drawn under the output on stderr ("" removes it). Drawn only when live(), with the cursor
// hidden (shown again when it goes, at exit, and on a signal that ends the process).
void status(const std::string& s);
// A status block of several rows (an animation's frame), redrawn in one write
void status(const std::vector<std::string>& rows);
// Leaves the status block on screen as it is, as ordinary output, and forgets it
void keep_status();
// Removes the status block for good (before the process exits: a fatal error)
void end_status();
// The handlers that take the status block off the screen and show the cursor when a signal ends the process: put in
// with the first status block, or before it by code whose own handlers pass the signal on to them
void guard_signals();

// A spinner on the status line: "⠹ <label>  <seconds>  <detail>" until done()
class Spinner {
 public:
  explicit Spinner(const std::string& label);
  ~Spinner();
  void label(const std::string& s);
  void detail(const std::string& s);  // shown dim after the time (e.g. the latest log line)
  double seconds() const;
  // Stops, and leaves `final` in its place ("" leaves nothing)
  void done(const std::string& final = "");

 private:
  struct Impl;
  Impl* impl_;
};

// A progress bar of `w` columns: done part in the accent, the rest dim
std::string bar(double fraction, int w);

// Text layout for the screens: rows of "key  description" (keys in a column styled by `key`, descriptions wrapped
// beside them, or under them when the column leaves too little room), and paragraphs wrapped at an indent
using Style = std::string (*)(const std::string&);
std::string plain(const std::string& s);
void two_columns(std::vector<std::string>& out, const std::vector<std::pair<std::string, std::string>>& rows, int indent,
                 int w, Style key = bold, Style desc = plain, int min_key = 0);
void paragraph(std::vector<std::string>& out, const std::string& text, int indent, int w, Style style = plain);
// A command's --help from its plain text: usage, options and their descriptions, examples, wrapped to the terminal
// (stdout)
void help(const std::string& plain_text);

// "kapitsa: <msg>" for a terminal: a red ✗ with the problem, then what to do on its own dim line(s). The message's
// first sentence is the problem, the rest the fix; `hint` (if any) is a last dim line. On stderr.
void error(const std::string& msg, const std::string& hint = "");

}  // namespace ni::term
