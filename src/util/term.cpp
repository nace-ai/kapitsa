// The command line's terminal look (term.h).
#include "util/term.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

namespace ni::term {

const char* const kOk = "✓";
const char* const kFail = "✗";
const char* const kOn = "●";
const char* const kOff = "○";
const char* const kStar = "★";
const char* const kArrow = "›";
const char* const kDot = "·";
const char* const kWarn = "!";

namespace {

std::string env(const char* k) {
  const char* v = getenv(k);
  return v ? v : "";
}

enum Depth { kNone = 0, k16 = 1, k256 = 2, kTrue = 3 };
enum Bg { kAny, kDark, kLight };

struct Caps {
  bool fancy = false, live = false;
  int depth = kNone;
};

// FORCE_COLOR: -1 unset; 0 for "0" / "false"; 1 for "", "true", "1"; 2; 3. CLICOLOR_FORCE (set, not "0"): at least 1.
int forced_depth() {
  int f = -1;
  if (const char* v = getenv("FORCE_COLOR")) {
    const std::string s = v;
    if (s == "0" || s == "false") f = 0;
    else if (s.empty() || s == "true") f = 1;
    else f = std::clamp(atoi(s.c_str()), 1, 3);
  }
  if (const char* v = getenv("CLICOLOR_FORCE"); v && *v && strcmp(v, "0") != 0) f = std::max(f, 1);
  return f;
}

int detected_depth() {
  std::string ct = env("COLORTERM");
  for (auto& ch : ct) ch = char(tolower(uint8_t(ch)));
  if (ct == "truecolor" || ct == "24bit") return kTrue;
  const std::string t = env("TERM"), prog = env("TERM_PROGRAM");
  if (prog == "iTerm.app" || prog == "WezTerm" || prog == "ghostty" || prog == "vscode" || t == "xterm-kitty" ||
      t == "xterm-ghostty" || t.find("-direct") != std::string::npos)
    return kTrue;
  if (t.find("256") != std::string::npos || prog == "Apple_Terminal") return k256;
  return k16;
}

const Caps& caps() {
  static const Caps c = [] {
    Caps k;
    const bool dumb = env("TERM") == "dumb";
    const int force = forced_depth();
    k.fancy = isatty(1) && isatty(2) && (!dumb || force > 0);
    k.live = k.fancy && !dumb;
    if (k.fancy) {
      if (force > 0) k.depth = std::max(force, dumb ? int(k16) : detected_depth());
      else if (force == 0 || dumb || !env("NO_COLOR").empty() || env("CLICOLOR") == "0") k.depth = kNone;
      else k.depth = detected_depth();
    }
    return k;
  }();
  return c;
}

// The terminal's background from its answer to OSC 11 ("ESC ] 11 ; rgb:RRRR/GGGG/BBBB"), fenced by DA1 ("ESC [ c",
// which every terminal answers, in order): once the DA1 reply is in, nothing else is coming. A reply that comes after
// the wait would be typed into the shell (or the coding agent), so only terminals known to answer at once are asked:
// local ones (not over SSH, not through tmux or screen), of the foreground job. Elsewhere (a web terminal through
// ttyd answers in about half a second) the colors that read on both backgrounds are used.
Bg query_background() {
  if (!caps().live || getenv("TMUX") || getenv("STY") || getenv("SSH_TTY") || getenv("SSH_CONNECTION")) return kAny;
  const std::string prog = env("TERM_PROGRAM"), t = env("TERM");
  if (prog != "iTerm.app" && prog != "Apple_Terminal" && prog != "ghostty" && prog != "WezTerm" && prog != "vscode" &&
      t != "xterm-kitty" && t != "xterm-ghostty")
    return kAny;
  const int fd = open("/dev/tty", O_RDWR | O_NOCTTY | O_CLOEXEC);
  if (fd < 0) return kAny;
  Bg out = kAny;
  termios old {};
  if (tcgetpgrp(fd) == getpgrp() && tcgetattr(fd, &old) == 0) {
    termios raw = old;
    raw.c_lflag &= ~tcflag_t(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(fd, TCSANOW, &raw) == 0) {
      static const char q[] = "\033]11;?\033\\\033[c";
      std::string r;
      if (write(fd, q, sizeof q - 1) == ssize_t(sizeof q - 1)) {
        const auto t0 = std::chrono::steady_clock::now();
        for (;;) {
          const int left = 1000 - int(std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::steady_clock::now() - t0).count());
          if (left <= 0) break;
          // select, not poll: macOS's poll answers POLLNVAL at once on /dev/tty, which ended the wait before any reply
          fd_set in;
          FD_ZERO(&in);
          FD_SET(fd, &in);
          timeval tv{left / 1000, (left % 1000) * 1000};
          const int n = select(fd + 1, &in, nullptr, nullptr, &tv);
          if (n < 0 && errno == EINTR) continue;
          if (n <= 0) break;
          char buf[256];
          const ssize_t got = read(fd, buf, sizeof buf);
          if (got <= 0) break;
          r.append(buf, size_t(got));
          const size_t d = r.find("\033[?");
          if (d != std::string::npos && r.find('c', d) != std::string::npos) break;
        }
      }
      tcsetattr(fd, TCSANOW, &old);
      // "rgb:" and three fields of 1-4 hex digits
      if (size_t p = r.find("]11;rgb:"); p != std::string::npos) {
        p += 8;
        double v[3];
        int k = 0;
        for (; k < 3; ++k) {
          size_t e = p;
          while (e < r.size() && isxdigit(uint8_t(r[e]))) ++e;
          if (e == p || e - p > 4) break;
          v[k] = double(strtoul(r.substr(p, e - p).c_str(), nullptr, 16)) / (std::pow(16.0, double(e - p)) - 1);
          p = e + 1;
          if (k < 2 && (e >= r.size() || r[e] != '/')) break;
        }
        if (k == 3) out = 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2] > 0.5 ? kLight : kDark;
      }
    }
  }
  close(fd);
  return out;
}

Bg background() {
  static const Bg b = [] {
    // COLORFGBG ("15;0", "0;15", rxvt's "15;default;0"): the background is the last field, an ANSI color number
    const std::string v = env("COLORFGBG");
    const size_t p = v.find_last_of(';');
    if (p != std::string::npos && p + 1 < v.size() && isdigit(uint8_t(v[p + 1]))) {
      const int i = atoi(v.c_str() + p + 1);
      return i == 7 || (i >= 9 && i <= 15) ? kLight : kDark;
    }
    return query_background();
  }();
  return b;
}

struct Rgb {
  int r, g, b;
  int x256 = -1;  // the 256-color index, when the nearest one reads worse (greener, more contrast)
};

// The nearest of xterm's 256 colors: the 6x6x6 cube or the gray ramp
int to256(Rgb c) {
  auto level = [](int v) { return v < 48 ? 0 : v < 115 ? 1 : (v - 35) / 40; };
  static const int steps[] = {0, 95, 135, 175, 215, 255};
  const int r = level(c.r), g = level(c.g), b = level(c.b);
  auto dist = [&](int x, int y, int z) { return (x - c.r) * (x - c.r) + (y - c.g) * (y - c.g) + (z - c.b) * (z - c.b); };
  const int cube = dist(steps[r], steps[g], steps[b]);
  const int avg = (c.r + c.g + c.b) / 3;
  const int gi = std::clamp((avg - 8 + 5) / 10, 0, 23), gv = 8 + 10 * gi;
  return dist(gv, gv, gv) < cube ? 232 + gi : 16 + 36 * r + 6 * g + b;
}

std::string fg(Rgb c) {
  char b[32];
  if (caps().depth == kTrue) snprintf(b, sizeof b, "\033[38;2;%d;%d;%dm", c.r, c.g, c.b);
  else snprintf(b, sizeof b, "\033[38;5;%dm", c.x256 >= 0 ? c.x256 : to256(c));
  return b;
}

enum Role { kAccentRole, kWarnRole, kErrRole };

// Dark backgrounds: bright enough to read (#3FB55A on #1E1E1E is 6.3:1); light ones: the brand green (4.7:1 on
// white) and darker yellow and red; unknown: between the two (3.2:1 on white, 5.2:1 on #1E1E1E).
Rgb rgb(Role role, Bg bg) {
  static const Rgb dark[] = {{0x3F, 0xB5, 0x5A}, {0xD2, 0x99, 0x22}, {0xF8, 0x51, 0x49}};
  // (the nearest 256-color of the brand green is a grayish #5F875F: #008700 instead)
  static const Rgb light[] = {{0x30, 0x84, 0x42, 28}, {0x9A, 0x67, 0x00}, {0xCF, 0x22, 0x2E}};
  static const Rgb any[] = {{0x2E, 0xA4, 0x4F}, {0xB9, 0x83, 0x06}, {0xDA, 0x36, 0x33}};
  return (bg == kDark ? dark : bg == kLight ? light : any)[role];
}

std::string paint(Role role, const std::string& s) {
  if (!color() || s.empty()) return s;
  if (caps().depth == k16) return std::string(role == kAccentRole ? "\033[32m" : role == kWarnRole ? "\033[33m" : "\033[31m") + s + "\033[39m";
  return fg(rgb(role, background())) + s + "\033[39m";
}

// UTF-8 decoding: the code point at s[i], and where the next one starts (an invalid byte is one character)
char32_t decode(const std::string& s, size_t i, size_t* next) {
  const uint8_t c = uint8_t(s[i]);
  const int n = c < 0x80 ? 0 : (c & 0xE0) == 0xC0 ? 1 : (c & 0xF0) == 0xE0 ? 2 : (c & 0xF8) == 0xF0 ? 3 : -1;
  if (n < 0 || i + size_t(n) >= s.size()) {
    *next = i + 1;
    return c;
  }
  char32_t cp = n == 0 ? c : n == 1 ? c & 0x1F : n == 2 ? c & 0x0F : c & 0x07;
  for (int k = 1; k <= n; ++k) {
    const uint8_t x = uint8_t(s[i + size_t(k)]);
    if ((x & 0xC0) != 0x80) {
      *next = i + 1;
      return c;
    }
    cp = (cp << 6) | (x & 0x3F);
  }
  *next = i + size_t(n) + 1;
  return cp;
}

// Columns a code point takes (wcwidth): 0 for controls, combining marks and joiners; 2 for East Asian wide and
// fullwidth characters and emoji; 1 otherwise
int cp_width(char32_t c) {
  if (c < 0x20 || (c >= 0x7F && c < 0xA0)) return 0;
  if (c < 0x300) return 1;
  if ((c >= 0x300 && c <= 0x36F) || (c >= 0x483 && c <= 0x489) || (c >= 0x591 && c <= 0x5BD) ||
      (c >= 0x610 && c <= 0x61A) || (c >= 0x64B && c <= 0x65F) || (c >= 0x1AB0 && c <= 0x1AFF) ||
      (c >= 0x1DC0 && c <= 0x1DFF) || (c >= 0x200B && c <= 0x200F) || (c >= 0x20D0 && c <= 0x20FF) ||
      (c >= 0xFE00 && c <= 0xFE0F) || (c >= 0xFE20 && c <= 0xFE2F) || c == 0xFEFF || (c >= 0xE0100 && c <= 0xE01EF))
    return 0;
  static const char32_t wide[][2] = {
      {0x1100, 0x115F}, {0x231A, 0x231B}, {0x2329, 0x232A}, {0x23E9, 0x23EC}, {0x23F0, 0x23F0}, {0x23F3, 0x23F3},
      {0x25FD, 0x25FE}, {0x2614, 0x2615}, {0x2648, 0x2653}, {0x267F, 0x267F}, {0x2693, 0x2693}, {0x26A1, 0x26A1},
      {0x26AA, 0x26AB}, {0x26BD, 0x26BE}, {0x26C4, 0x26C5}, {0x26CE, 0x26CE}, {0x26D4, 0x26D4}, {0x26EA, 0x26EA},
      {0x26F2, 0x26F3}, {0x26F5, 0x26F5}, {0x26FA, 0x26FA}, {0x26FD, 0x26FD}, {0x2705, 0x2705}, {0x270A, 0x270B},
      {0x2728, 0x2728}, {0x274C, 0x274C}, {0x274E, 0x274E}, {0x2753, 0x2755}, {0x2757, 0x2757}, {0x2795, 0x2797},
      {0x27B0, 0x27B0}, {0x27BF, 0x27BF}, {0x2B1B, 0x2B1C}, {0x2B50, 0x2B50}, {0x2B55, 0x2B55}, {0x2E80, 0x303E},
      {0x3041, 0x33FF}, {0x3400, 0x4DBF}, {0x4E00, 0x9FFF}, {0xA000, 0xA4CF}, {0xA960, 0xA97F}, {0xAC00, 0xD7A3},
      {0xF900, 0xFAFF}, {0xFE10, 0xFE19}, {0xFE30, 0xFE6F}, {0xFF00, 0xFF60}, {0xFFE0, 0xFFE6}, {0x16FE0, 0x18CFF},
      {0x1B000, 0x1B2FF}, {0x1F004, 0x1F004}, {0x1F0CF, 0x1F0CF}, {0x1F18E, 0x1F18E}, {0x1F191, 0x1F19A},
      {0x1F200, 0x1F251}, {0x1F300, 0x1F64F}, {0x1F680, 0x1F6FF}, {0x1F7E0, 0x1F7EB}, {0x1F90C, 0x1F9FF},
      {0x1FA70, 0x1FAFF}, {0x20000, 0x2FFFD}, {0x30000, 0x3FFFD}};
  for (const auto& w : wide)
    if (c >= w[0] && c <= w[1]) return 2;
  return 1;
}

// Where the escape sequence at s[i] (an ESC) ends: CSI to its final byte, OSC to BEL or ST, else two bytes
size_t skip_escape(const std::string& s, size_t i) {
  if (i + 1 >= s.size()) return s.size();
  if (s[i + 1] == '[') {
    size_t j = i + 2;
    while (j < s.size() && !(uint8_t(s[j]) >= 0x40 && uint8_t(s[j]) <= 0x7E)) ++j;
    return std::min(j + 1, s.size());
  }
  if (s[i + 1] == ']') {
    for (size_t j = i + 2; j < s.size(); ++j) {
      if (s[j] == '\a') return j + 1;
      if (s[j] == '\033' && j + 1 < s.size() && s[j + 1] == '\\') return j + 2;
    }
    return s.size();
  }
  return i + 2;
}

std::string repeat(const char* g, int n) {
  std::string s;
  for (int i = 0; i < n; ++i) s += g;
  return s;
}

// Output shared by the status block (a spinner's line, an animation's rows) and the lines above it
std::mutex g_mu;
std::vector<std::string> g_status;
int g_drawn = 0;  // rows of the status block on screen (the cursor is at the end of its last one)
volatile sig_atomic_t g_rows_on_screen = 0;  // the same, for a signal handler
bool g_ended = false, g_hidden = false;

void put(const std::string& s) {
  fputs(s.c_str(), stderr);
  fflush(stderr);
}

// The cursor is hidden while a status block is drawn, and shown again however the process ends: at exit, and on the
// signals that would otherwise kill it with the cursor hidden (when nothing else handles them)
void restore_cursor() {
  if (g_hidden) {
    static const char show[] = "\033[?25h";
    if (write(2, show, sizeof show - 1) < 0) {
    }
  }
}

// A signal that ends the process (Ctrl+C): the status block off the screen, the cursor back (write() only)
void on_signal(int sig) {
  const int n = g_rows_on_screen;
  if (n > 0 && (sig == SIGINT || sig == SIGTERM || sig == SIGHUP)) {
    char b[32];
    int k = 0;
    b[k++] = '\r';
    if (n > 1) {
      b[k++] = '\033', b[k++] = '[';
      char d[8];
      int nd = 0;
      for (int v = n - 1; v > 0 && nd < 7; v /= 10) d[nd++] = char('0' + v % 10);
      while (nd > 0) b[k++] = d[--nd];
      b[k++] = 'A';
    }
    b[k++] = '\033', b[k++] = '[', b[k++] = 'J';
    if (write(2, b, size_t(k)) < 0) {
    }
  }
  restore_cursor();
  signal(sig, SIG_DFL);
  raise(sig);
}

void hide_cursor() {
  if (g_hidden) return;
  guard_signals();
  g_hidden = true;
  put("\033[?25l");
}

void show_cursor() {
  if (!g_hidden) return;
  g_hidden = false;
  put("\033[?25h");
}

// The escape codes that take the cursor back to the start of the status block and clear it
std::string erase_status() {
  if (g_drawn == 0) return "";
  std::string s = "\r";
  if (g_drawn > 1) s += "\033[" + std::to_string(g_drawn - 1) + "A";
  g_drawn = 0;
  g_rows_on_screen = 0;
  return s + "\033[J";
}

// The status block, each row cut to the width (a wrapped row would break the redraw)
std::string status_rows() {
  if (g_status.empty() || g_ended || !caps().live) return "";
  const int w = std::max(10, columns() - 1);
  std::string s;
  for (size_t i = 0; i < g_status.size(); ++i) s += (i ? "\n" : "") + clip(g_status[i], w) + "\033[K";
  g_drawn = int(g_status.size());
  g_rows_on_screen = g_drawn;
  return s;
}

// One write per change, inside synchronized output (terminals without it ignore the mode): no flicker
void redraw(const std::string& before = "") {
  std::string s = erase_status() + before + status_rows();
  if (s.empty()) return;
  if (g_drawn > 0) hide_cursor();
  put("\033[?2026h" + s + "\033[?2026l");
  if (g_drawn == 0) show_cursor();
}

}  // namespace

void guard_signals() {
  static bool once = [] {
    atexit(restore_cursor);
    for (int sig : {SIGINT, SIGTERM, SIGHUP, SIGQUIT, SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE}) {
      struct sigaction old {};
      sigaction(sig, nullptr, &old);
      if (old.sa_handler == SIG_DFL) signal(sig, on_signal);
    }
    return true;
  }();
  (void)once;
}

bool fancy() { return caps().fancy; }
bool live() { return caps().live; }
bool color() { return caps().depth != kNone; }

int columns() {
  winsize ws {};
  for (int fd : {1, 2})
    if (ioctl(fd, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) return ws.ws_col;
  if (const int c = atoi(env("COLUMNS").c_str()); c > 0) return c;
  return 80;
}

std::string bold(const std::string& s) { return fancy() && !s.empty() ? "\033[1m" + s + "\033[22m" : s; }
std::string dim(const std::string& s) { return fancy() && !s.empty() ? "\033[2m" + s + "\033[22m" : s; }
std::string italic(const std::string& s) { return fancy() && !s.empty() ? "\033[3m" + s + "\033[23m" : s; }
std::string accent(const std::string& s) { return paint(kAccentRole, s); }
std::string warn(const std::string& s) { return paint(kWarnRole, s); }
std::string red(const std::string& s) { return paint(kErrRole, s); }

std::string badge(const std::string& s) {
  if (!fancy()) return s;
  if (!color()) return bold(s);
  // 16 colors: black on the theme's green reads on bright and dark greens alike
  if (caps().depth == k16) return "\033[1;30;42m" + s + "\033[22;39;49m";
  const Rgb brand{0x30, 0x84, 0x42, 28};  // white on it: 4.7:1
  char b[32];
  if (caps().depth == kTrue) snprintf(b, sizeof b, "\033[48;2;%d;%d;%dm", brand.r, brand.g, brand.b);
  else snprintf(b, sizeof b, "\033[48;5;%dm", brand.x256);
  const std::string bg = b;
  return "\033[1m" + bg + (caps().depth == kTrue ? "\033[38;2;255;255;255m" : "\033[38;5;231m") + s + "\033[22;39;49m";
}

int width(const std::string& s) {
  int w = 0;
  for (size_t i = 0; i < s.size();) {
    if (s[i] == '\033') {
      i = skip_escape(s, i);
      continue;
    }
    size_t next;
    w += cp_width(decode(s, i, &next));
    i = next;
  }
  return w;
}

std::string left(const std::string& s, int w) {
  const int n = width(s);
  return n >= w ? s : s + std::string(size_t(w - n), ' ');
}

std::string right(const std::string& s, int w) {
  const int n = width(s);
  return n >= w ? s : std::string(size_t(w - n), ' ') + s;
}

std::string clip(const std::string& s, int w) {
  if (width(s) <= w) return s;
  if (w <= 0) return "";
  std::string out;
  int used = 0;
  bool styled = false;
  for (size_t i = 0; i < s.size();) {
    if (s[i] == '\033') {
      const size_t j = skip_escape(s, i);
      out.append(s, i, j - i);
      styled = true;
      i = j;
      continue;
    }
    size_t next;
    const int cw = cp_width(decode(s, i, &next));
    if (used + cw > w - 1) break;
    out.append(s, i, next - i);
    used += cw;
    i = next;
  }
  out += "…";
  if (styled) out += "\033[0m";
  return out;
}

std::vector<std::string> wrap(const std::string& text, int w) {
  std::vector<std::string> lines;
  w = std::max(w, 8);
  std::string cur;
  int cw = 0;
  size_t i = 0;
  while (i < text.size()) {
    while (i < text.size() && text[i] == ' ') ++i;
    if (i >= text.size()) break;
    size_t j = i;
    while (j < text.size() && text[j] != ' ') ++j;
    std::string word = text.substr(i, j - i);
    i = j;
    int ww = width(word);
    if (cw > 0 && cw + 1 + ww <= w) {
      cur += " " + word;
      cw += 1 + ww;
      continue;
    }
    if (cw > 0) lines.push_back(cur);
    cur.clear();
    cw = 0;
    while (ww > w) {  // a word longer than the line: split it
      size_t n = 0;
      for (int used = 0; n < word.size();) {
        size_t next;
        const int c = cp_width(decode(word, n, &next));
        if (used + c > w) break;
        used += c;
        n = next;
      }
      lines.push_back(word.substr(0, n));
      word = word.substr(n);
      ww = width(word);
    }
    cur = word;
    cw = ww;
  }
  if (cw > 0 || lines.empty()) lines.push_back(cur);
  return lines;
}

std::string duration(double s) {
  char b[32];
  if (!(s >= 0) || !std::isfinite(s)) return "?";
  if (s < 9.95) snprintf(b, sizeof b, "%.1fs", s);
  else if (s < 59.5) snprintf(b, sizeof b, "%.0fs", s);
  else if (s < 3600) snprintf(b, sizeof b, "%dm %02ds", int(s + 0.5) / 60, int(s + 0.5) % 60);
  else snprintf(b, sizeof b, "%dh %02dm", int(s) / 3600, int(s) % 3600 / 60);
  return b;
}

std::string count(double n) {
  char b[32];
  if (n < 1000) snprintf(b, sizeof b, "%.0f", n);
  else if (n < 99950) snprintf(b, sizeof b, "%.1fK", n / 1e3);
  else if (n < 999500) snprintf(b, sizeof b, "%.0fK", n / 1e3);
  else snprintf(b, sizeof b, "%.1fM", n / 1e6);
  return b;
}

std::string context(int n) {
  if (n > 0 && n % 1048576 == 0) return std::to_string(n / 1048576) + "M";
  return n > 0 && n % 1024 == 0 ? std::to_string(n / 1024) + "K" : count(n);
}

void line(const std::string& s, FILE* f) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (f == stderr) {
    redraw(s + "\n");  // the block off, the line, the block again: one write
    return;
  }
  // stdout: the block off (stderr) first, then the line, then the block again
  if (g_drawn > 0) put(erase_status());
  fputs((s + "\n").c_str(), f);
  fflush(f);
  redraw();
}

void status(const std::string& s) { status(s.empty() ? std::vector<std::string>{} : std::vector<std::string>{s}); }

void status(const std::vector<std::string>& rows) {
  std::lock_guard<std::mutex> lk(g_mu);
  g_status = rows;
  redraw();
  if (g_status.empty()) show_cursor();
}

void keep_status() {
  std::lock_guard<std::mutex> lk(g_mu);
  if (g_drawn > 0) put("\n");
  g_drawn = 0;
  g_rows_on_screen = 0;
  g_status.clear();
  show_cursor();
}

void end_status() {
  std::lock_guard<std::mutex> lk(g_mu);
  put(erase_status());
  g_status.clear();
  g_ended = true;
  show_cursor();
}

struct Spinner::Impl {
  std::mutex mu;
  std::string label, detail;
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  std::atomic<bool> stop{false};
  std::thread th;
  bool finished = false;

  double secs() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  std::string render(int frame) {
    static const char* const frames[] = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
    std::lock_guard<std::mutex> lk(mu);
    std::string s = "  " + accent(frames[frame % 10]) + " " + label + "  " + dim(duration(secs()));
    if (!detail.empty()) s += "  " + dim(detail);
    return s;
  }
};

Spinner::Spinner(const std::string& label) : impl_(new Impl) {
  impl_->label = label;
  if (!live()) return;
  impl_->th = std::thread([this] {
    for (int i = 0; !impl_->stop; ++i) {
      status(impl_->render(i));
      for (int k = 0; k < 8 && !impl_->stop; ++k) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  });
}

Spinner::~Spinner() {
  done();
  delete impl_;
}

void Spinner::label(const std::string& s) {
  std::lock_guard<std::mutex> lk(impl_->mu);
  impl_->label = s;
}

void Spinner::detail(const std::string& s) {
  std::lock_guard<std::mutex> lk(impl_->mu);
  impl_->detail = s;
}

double Spinner::seconds() const { return impl_->secs(); }

void Spinner::done(const std::string& final) {
  if (impl_->finished) return;
  impl_->finished = true;
  impl_->stop = true;
  if (impl_->th.joinable()) impl_->th.join();
  status("");
  if (!final.empty()) line(final, stderr);
}

std::string bar(double f, int w) {
  f = std::clamp(f, 0.0, 1.0);
  const double cells = f * w;
  int full = int(cells);
  const bool half = full < w && cells - full >= 0.5;
  std::string done = repeat("━", full) + (half ? "╸" : "");
  const int rest = w - full - (half ? 1 : 0);
  return accent(done) + dim(repeat("━", rest));
}

void error(const std::string& msg, const std::string& hint) {
  // the problem: the first line, up to the end of its first sentence; the rest is the fix
  std::string first = msg, rest;
  if (const size_t nl = msg.find('\n'); nl != std::string::npos) first = msg.substr(0, nl), rest = msg.substr(nl + 1);
  int depth = 0;
  for (size_t i = 0; i + 2 < first.size(); ++i) {
    if (first[i] == '(') ++depth;
    else if (first[i] == ')') depth = std::max(0, depth - 1);
    else if (depth == 0 && (first[i] == '.' || first[i] == ':' || first[i] == '?') && first[i + 1] == ' ' &&
             (isupper(uint8_t(first[i + 2])) || first[i + 2] == '`' || first[i + 2] == '(')) {
      rest = first.substr(i + 2) + (rest.empty() ? "" : "\n" + rest);
      first = first.substr(0, i + 1);
      break;
    }
  }
  const int w = std::max(30, std::min(columns(), 100) - 4);
  std::vector<std::string> head = wrap(first, w);
  std::string out;
  for (size_t i = 0; i < head.size(); ++i) out += (i ? "    " : "  " + red(bold(kFail)) + " ") + head[i] + "\n";
  size_t a = 0;
  while (a < rest.size()) {
    size_t b = rest.find('\n', a);
    if (b == std::string::npos) b = rest.size();
    std::string l = rest.substr(a, b - a);
    a = b + 1;
    const bool cmd = l.rfind("  ", 0) == 0;  // an indented line: a command to run, kept as is
    if (cmd) {
      while (!l.empty() && l[0] == ' ') l.erase(l.begin());
      std::string alt;  // "cmd      (or: other cmd)": the other one on a line of its own
      if (const size_t p = l.find("  ("); p != std::string::npos && l.back() == ')') {
        alt = l.substr(p + 3, l.size() - p - 4);
        l.resize(p);
        while (!l.empty() && l.back() == ' ') l.pop_back();
      }
      out += "      " + bold(clip(l, w - 2)) + "\n";
      if (!alt.empty()) out += "      " + dim(clip(alt, w - 2)) + "\n";
      continue;
    }
    for (const auto& x : wrap(l, w)) out += "    " + dim(x) + "\n";
  }
  if (!hint.empty())
    for (const auto& x : wrap(hint, w)) out += "    " + dim(x) + "\n";
  fflush(stdout);
  std::lock_guard<std::mutex> lk(g_mu);
  redraw(out);
}


// Rows of "key  description": the keys in a column (styled by `key`), the descriptions wrapped beside them (styled per
// line by `desc`), or under them when the column leaves too little room.
std::string plain(const std::string& s) { return s; }

void two_columns(std::vector<std::string>& out, const std::vector<std::pair<std::string, std::string>>& rows, int indent,
                 int w, Style key, Style desc, int min_key) {
  int kw = min_key;
  for (const auto& [k, d] : rows) kw = std::max(kw, width(k));
  const int dw = w - indent - kw - 3;
  const std::string pad(size_t(indent), ' ');
  for (const auto& [k, d] : rows) {
    if (d.empty()) {
      out.push_back(pad + key(k));
      continue;
    }
    if (dw < 20) {  // too narrow: the description under its key
      out.push_back(pad + key(k));
      for (const auto& l : wrap(d, w - indent - 2)) out.push_back(pad + "  " + desc(l));
      continue;
    }
    const auto lines = wrap(d, dw);
    for (size_t i = 0; i < lines.size(); ++i)
      out.push_back(pad + (i ? std::string(size_t(kw), ' ') : left(key(k), kw)) + "   " + desc(lines[i]));
  }
}

// A paragraph wrapped to the width at an indent, each line styled
void paragraph(std::vector<std::string>& out, const std::string& text, int indent, int w, Style style) {
  for (const auto& l : wrap(text, w - indent)) out.push_back(std::string(size_t(indent), ' ') + style(l));
}



void help(const std::string& text) {
  const int W = std::min(columns(), 100);
  std::vector<std::string> lines;
  for (size_t a = 0; a <= text.size();) {
    size_t b = text.find('\n', a);
    if (b == std::string::npos) b = text.size();
    lines.push_back(text.substr(a, b - a));
    a = b + 1;
  }
  while (!lines.empty() && lines.back().empty()) lines.pop_back();
  std::vector<std::string> o;
  o.push_back("");
  // the usage: "kapitsa <command> ..." up to its first ": " or the end of its paragraph
  size_t i = 0;
  {
    std::string usage;
    for (; i < lines.size() && !lines[i].empty() && lines[i][0] != ' '; ++i) usage += (usage.empty() ? "" : " ") + lines[i];
    std::string rest;
    if (const size_t c = usage.find(": "); c != std::string::npos) rest = usage.substr(c + 2), usage.resize(c);
    // the command words in bold, its options as they are
    size_t cmd_end = 0;
    for (size_t k = 0; k < usage.size();) {
      size_t e = usage.find(' ', k);
      if (e == std::string::npos) e = usage.size();
      const std::string word = usage.substr(k, e - k);
      if (word.empty() || word[0] == '[' || word[0] == '-' || isupper(uint8_t(word[0]))) break;
      cmd_end = e;
      k = e + 1;
    }
    const auto wrapped = wrap(usage, W - 4);
    for (size_t k = 0; k < wrapped.size(); ++k) {
      std::string l = wrapped[k];
      if (k == 0 && cmd_end > 0 && cmd_end <= l.size()) l = bold(l.substr(0, cmd_end)) + l.substr(cmd_end);
      o.push_back((k ? "    " : "  ") + l);
    }
    if (!rest.empty()) {
      o.push_back("");
      paragraph(o, rest, 2, W);
    }
  }
  // then: option rows ("  --flag ARG   what it does", continued by deeper indented lines), examples (other indented
  // lines) and paragraphs. The descriptions start at one column (a flag as wide as it is followed by one space only):
  // the most common column after two spaces.
  size_t desc_col = 0;
  {
    std::vector<size_t> cols;
    for (size_t k = i; k < lines.size(); ++k) {
      const std::string& l = lines[k];
      size_t ind = 0;
      while (ind < l.size() && l[ind] == ' ') ++ind;
      if (ind < 2 || ind >= 6 || ind >= l.size() || l[ind] != '-') continue;
      if (const size_t g = l.find("  ", ind); g != std::string::npos) {
        size_t d = g;
        while (d < l.size() && l[d] == ' ') ++d;
        cols.push_back(d);
      }
    }
    size_t best = 0;
    for (size_t c : cols)
      if (std::count(cols.begin(), cols.end(), c) > std::count(cols.begin(), cols.end(), best)) best = c;
    desc_col = best;
  }
  std::vector<std::pair<std::string, std::string>> opts;
  auto flush_opts = [&] {
    if (opts.empty()) return;
    two_columns(o, opts, 4, W);
    opts.clear();
  };
  std::string para;
  auto flush_para = [&] {
    if (para.empty()) return;
    paragraph(o, para, 2, W);
    para.clear();
  };
  for (; i < lines.size(); ++i) {
    const std::string& l = lines[i];
    if (l.empty()) {
      flush_opts();
      flush_para();
      o.push_back("");
      continue;
    }
    size_t ind = 0;
    while (ind < l.size() && l[ind] == ' ') ++ind;
    if (ind >= 6 && !opts.empty()) {  // an option's description, continued
      opts.back().second += " " + l.substr(ind);
      continue;
    }
    if (ind >= 2 && ind < 6 && l[ind] == '-' && l.size() > ind + 1 && l[ind + 1] != ' ') {
      flush_para();
      size_t gap = l.find("  ", ind);
      if (desc_col > ind && desc_col < l.size() && l[desc_col - 1] == ' ' && l[desc_col] != ' ' && (gap == std::string::npos || gap > desc_col)) {
        std::string k = l.substr(ind, desc_col - ind);
        while (!k.empty() && k.back() == ' ') k.pop_back();
        opts.push_back({k, l.substr(desc_col)});
      } else if (gap == std::string::npos) {
        opts.push_back({l.substr(ind), ""});
      } else {
        size_t d = gap;
        while (d < l.size() && l[d] == ' ') ++d;
        opts.push_back({l.substr(ind, gap - ind), l.substr(d)});
      }
      continue;
    }
    if (ind >= 2) {  // an example: kept as is, the command in bold
      flush_opts();
      flush_para();
      o.push_back("    " + bold(l.substr(ind)));
      continue;
    }
    flush_opts();
    // a sentence that ends a line and a capital that starts the next: a new paragraph
    if (!para.empty() && para.back() == '.' && isupper(uint8_t(l[0]))) flush_para();
    para += (para.empty() ? "" : " ") + l;
  }
  flush_opts();
  flush_para();
  o.push_back("");
  std::string out;
  for (const auto& l : o) out += l + "\n";
  fputs(out.c_str(), stdout);
  fflush(stdout);
}

}  // namespace ni::term
