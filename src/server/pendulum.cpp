// Kapitsa's pendulum (pendulum.h).
#include "server/pendulum.h"

#include "util/log.h"
#include "util/term.h"

#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

namespace ni::ui {

using namespace term;

namespace {

// ---- the physics: lengths in rod lengths, time in sqrt(l / g)

// theta from the upward vertical; the pivot at a cos(w t): theta'' = (1 - a w^2 cos(w t)) sin(theta) - b theta'.
// Upright is stable when a^2 w^2 / 2 > 1 (here 5.8): the vibration's average effect outweighs gravity.
struct Sim {
  double th = 0, om = 0, t = 0;
  double a = 0.1, w = 34;  // the pivot's amplitude and frequency
  double vib = 0;          // how much of the vibration is on (0 .. 1)
  double b = 0.05;         // damping

  double pivot() const { return a * vib * std::cos(w * t); }
  double acc(double th_, double om_, double t_) const {
    return (1.0 - a * vib * w * w * std::cos(w * t_)) * std::sin(th_) - b * om_;
  }
  // Runge-Kutta 4 in steps of 0.002 (about 90 per period of the vibration)
  void advance(double dt) {
    const int n = std::max(1, int(std::ceil(dt / 0.002)));
    const double h = dt / n;
    for (int i = 0; i < n; ++i) {
      const double k1t = om, k1o = acc(th, om, t);
      const double k2t = om + h / 2 * k1o, k2o = acc(th + h / 2 * k1t, om + h / 2 * k1o, t + h / 2);
      const double k3t = om + h / 2 * k2o, k3o = acc(th + h / 2 * k2t, om + h / 2 * k2o, t + h / 2);
      const double k4t = om + h * k3o, k4o = acc(th + h * k3t, om + h * k3o, t + h);
      th += h / 6 * (k1t + 2 * k2t + 2 * k3t + k4t);
      om += h / 6 * (k1o + 2 * k2o + 2 * k3o + k4o);
      t += h;
    }
  }
};

constexpr double kScale = 0.2;  // seconds of animation per time unit
constexpr int kFps = 30;

// ---- drawing: quadrant blocks (▘▝▖▗ and their unions), 2 x 2 pixels a character: solid where braille's dots would
// look faint. Lengths in character widths; a pixel is half a width wide and one width tall (a character is about twice
// as tall as wide), so circles stay round.

constexpr int kW = kArtCols * 2, kH = kArtRows * 2;  // pixels
constexpr double kPx = kArtCols - 3.5, kPy = 3.0;     // the pivot at rest (y up from the bottom)
constexpr double kL = 9.5, kR = 1.45;                 // the rod's length and the bob's radius

struct Canvas {
  uint8_t main[kH][kW] = {}, faint[kH][kW] = {};

  void dot(bool m, double x, double y) {
    const long xi = long(std::floor(x / 0.5)), yi = kH - 1 - long(std::floor(y));
    if (xi >= 0 && xi < kW && yi >= 0 && yi < kH) (m ? main : faint)[yi][xi] = 1;
  }
  void line(bool m, double x0, double y0, double x1, double y1) {
    const int n = std::max(1, int(std::ceil(std::max(std::fabs(x1 - x0) * 2, std::fabs(y1 - y0)) * 3)));
    for (int i = 0; i <= n; ++i) dot(m, x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n);
  }
  void disk(double cx, double cy, double r) {  // the pixels whose centers are inside
    for (int yi = 0; yi < kH; ++yi)
      for (int xi = 0; xi < kW; ++xi) {
        const double x = (xi + 0.5) * 0.5, y = kH - 1 - yi + 0.5;
        if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r) main[yi][xi] = 1;
      }
  }
  void bar(bool m, double cx, double y, double half) {
    for (double x = cx - half; x <= cx + half + 1e-9; x += 0.5) dot(m, x, y);
  }
  // The rows: the pendulum in the accent, the trail and the base's blur faint
  std::vector<std::string> rows() const {
    static const char* const glyph[16] = {" ", "▘", "▝", "▀", "▖", "▌", "▞", "▛",
                                          "▗", "▚", "▐", "▜", "▄", "▙", "▟", "█"};
    std::vector<std::string> out;
    for (int r = 0; r < kArtRows; ++r) {
      std::string row, run;
      int style = 0;  // of `run`: 0 blank, 1 main, 2 faint
      auto flush = [&] {
        row += style == 1 ? accent(run) : style == 2 ? dim(accent(run)) : run;
        run.clear();
      };
      for (int c = 0; c < kArtCols; ++c) {
        int m = 0, f = 0;
        for (int y = 0; y < 2; ++y)
          for (int x = 0; x < 2; ++x) {
            const int bit = 1 << (y * 2 + x);  // ▘ ▝ ▖ ▗
            if (main[r * 2 + y][c * 2 + x]) m |= bit;
            if (faint[r * 2 + y][c * 2 + x]) f |= bit;
          }
        const int st = m ? 1 : f ? 2 : 0;
        if (st != style) flush(), style = st;
        run += glyph[m ? m : f];
      }
      flush();
      out.push_back(row);
    }
    return out;
  }
};

// The scene: a fixed ground, the shaker's platform under the pivot (its extremes faint while it vibrates), the bob's
// trail, the rod and the bob
void draw(Canvas& c, const Sim& s, const std::deque<std::pair<double, double>>& trail) {
  const double py = kPy + s.pivot() * kL;
  c.bar(false, kPx, 0.5, 2.5);
  if (s.vib > 0.3) c.bar(false, kPx, kPy - 1 + s.a * kL, 1.5), c.bar(false, kPx, kPy - 1 - s.a * kL, 1.5);
  c.bar(true, kPx, py - 1, 1.5);
  for (size_t i = 0; i + 1 < trail.size(); ++i)
    c.line(false, trail[i].first, trail[i].second, trail[i + 1].first, trail[i + 1].second);
  const double bx = kPx + kL * std::sin(s.th), by = py + kL * std::cos(s.th);
  c.line(true, kPx, py, bx, by);
  c.disk(bx, by, kR);
}

std::pair<double, double> bob(const Sim& s) {
  return {kPx + kL * std::sin(s.th), kPy + s.pivot() * kL + kL * std::cos(s.th)};
}

std::vector<std::string> frame(const Sim& s, const std::deque<std::pair<double, double>>& trail) {
  Canvas c;
  draw(c, s, trail);
  return c.rows();
}

int term_rows() {
  winsize ws {};
  for (int fd : {1, 2, 0})
    if (ioctl(fd, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0) return ws.ws_row;
  return 24;
}

// Keys during the intro: no echo, no line editing, no signals (Ctrl+C is a key that skips it); the modes come back
// at the end, and on a signal that ends the process
termios g_saved;
std::atomic<bool> g_raw{false};

void restore_tty(int sig) {
  if (g_raw.exchange(false)) tcsetattr(0, TCSANOW, &g_saved);
  static const char show[] = "\033[?25h";
  if (write(2, show, sizeof show - 1) < 0) {
  }
  signal(sig, SIG_DFL);
  raise(sig);
}

struct Keys {
  struct sigaction old_term {}, old_hup {};
  Keys() {
    if (tcgetattr(0, &g_saved) != 0) return;
    termios raw = g_saved;
    raw.c_lflag &= ~tcflag_t(ICANON | ECHO | ISIG);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    struct sigaction sa {};
    sa.sa_handler = restore_tty;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGTERM, &sa, &old_term);
    sigaction(SIGHUP, &sa, &old_hup);
    if (tcsetattr(0, TCSANOW, &raw) == 0) g_raw = true;
  }
  ~Keys() {
    if (g_raw.exchange(false)) tcsetattr(0, TCSANOW, &g_saved);
    sigaction(SIGTERM, &old_term, nullptr);
    sigaction(SIGHUP, &old_hup, nullptr);
  }
  bool pressed() {
    if (!g_raw) return false;
    pollfd p{0, POLLIN, 0};
    if (poll(&p, 1, 0) <= 0) return false;
    char b[64];
    return read(0, b, sizeof b) > 0;
  }
};

}  // namespace

bool can_animate() {
  if (!live() || !color() || !isatty(0)) return false;
  if (const char* ci = getenv("CI"); ci && *ci && strcmp(ci, "0") != 0 && strcmp(ci, "false") != 0) return false;
  if (env_on("KAPITSA_NO_ANIMATION")) return false;
  if (tcgetpgrp(1) != getpgrp()) return false;  // a background job: the terminal is someone else's
  return columns() >= 60 && term_rows() >= kArtRows + 6;
}

std::vector<std::string> beside(const std::vector<std::string>& art, const std::vector<std::string>& text, int at) {
  std::vector<std::string> out;
  const size_t n = std::max(art.size(), size_t(at) + text.size());
  for (size_t i = 0; i < n; ++i) {
    std::string l = "  " + left(i < art.size() ? art[i] : "", kArtCols);
    if (i >= size_t(at) && i - size_t(at) < text.size()) l += "   " + text[i - size_t(at)];
    while (!l.empty() && l.back() == ' ') l.pop_back();
    out.push_back(l);
  }
  return out;
}

std::vector<std::string> still_art() {
  Sim s;
  s.vib = 1;
  s.t = M_PI / 2 / s.w;  // the platform midway, its extremes faint
  return frame(s, {});
}

void intro(const std::vector<std::string>& text, int at) {
  if (!can_animate()) {
    std::string out;
    for (const auto& l : beside(still_art(), text, at)) out += l + "\n";
    fputs(out.c_str(), stdout);
    fflush(stdout);
    return;
  }
  (void)accent(" ");  // the terminal's colors are asked for before the keys are read
  Keys keys;
  // tilted a little: it falls; the vibration comes on and brings it back up, where it settles
  Sim s;
  s.th = -0.12;
  const double kick = 0.45, total = 1.45;
  std::deque<std::pair<double, double>> trail;
  const auto t0 = std::chrono::steady_clock::now();
  for (int f = 0;; ++f) {
    const double tr = double(f) / kFps;
    if (tr >= total || keys.pressed()) break;
    const double ramp = std::clamp((tr - kick) / 0.15, 0.0, 1.0);
    s.vib = ramp;
    s.b = 0.05 + 1.75 * ramp;
    if (f > 0) s.advance(1.0 / kFps / kScale);
    status(beside(frame(s, trail), text, at));
    trail.push_back(bob(s));
    if (trail.size() > 6) trail.pop_front();
    std::this_thread::sleep_until(t0 + std::chrono::microseconds(1000000LL * (f + 1) / kFps));
  }
  status(beside(still_art(), text, at));
  keep_status();
}

struct Wobble::Impl {
  std::function<std::vector<std::string>()> text;
  int at = 0;
  std::atomic<double> progress{0};
  std::atomic<bool> stop{false};
  std::thread th;
};

Wobble::Wobble(std::function<std::vector<std::string>()> text, int at) : impl_(new Impl) {
  impl_->text = std::move(text);
  impl_->at = at;
  if (!can_animate()) return;
  impl_->th = std::thread([this] {
    Sim s;
    s.vib = 1;
    s.b = 0.35;
    s.th = -0.26;  // 15 degrees off: it wobbles back and forth around upright, less and less
    std::deque<std::pair<double, double>> trail;
    int side = 1;
    const auto t0 = std::chrono::steady_clock::now();
    for (int f = 0; !impl_->stop; ++f) {
      if (f > 0) s.advance(1.0 / kFps / kScale);
      // calm again: a nudge, smaller as the load goes on
      const double p = std::max(impl_->progress.load(), 1 - std::exp(-double(f) / kFps / 5));
      if (std::fabs(s.th) < 0.03 && std::fabs(s.om) < 0.15 && p < 0.95) s.om += side * 0.6 * (1 - p), side = -side;
      status(beside(frame(s, trail), impl_->text(), impl_->at));
      trail.push_back(bob(s));
      if (trail.size() > 6) trail.pop_front();
      std::this_thread::sleep_until(t0 + std::chrono::microseconds(1000000LL * (f + 1) / kFps));
    }
  });
}

Wobble::~Wobble() {
  impl_->stop = true;
  if (impl_->th.joinable()) impl_->th.join();
  delete impl_;
}

void Wobble::progress(double p) { impl_->progress = std::clamp(p, 0.0, 1.0); }

void Wobble::done(const std::vector<std::string>& final) {
  const bool animated = impl_->th.joinable();
  impl_->stop = true;
  if (animated) impl_->th.join();
  const auto rows = beside(still_art(), final, impl_->at);
  if (animated) {
    status(rows);
    keep_status();
    return;
  }
  for (const auto& l : rows) line(l, stderr);
}

}  // namespace ni::ui
