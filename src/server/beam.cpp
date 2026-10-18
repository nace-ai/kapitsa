// Kapitsa's look: the Kapitza-Dirac effect (beam.h).
#include "server/beam.h"

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
#include <mutex>
#include <thread>

namespace ni::ui {

using namespace term;

namespace {

constexpr int kFps = 30;

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

void sleep_frame(std::chrono::steady_clock::time_point t0, int f) {
  std::this_thread::sleep_until(t0 + std::chrono::microseconds(1000000LL * (f + 1) / kFps));
}

}  // namespace

std::vector<std::string> diffraction(int cols, int rows, const Beam& b) {
  const int W = cols * 2, H = rows * 4;  // braille: 2 x 4 dots a character
  std::vector<uint8_t> main(size_t(W * H), 0), faint(size_t(W * H), 0);
  const double yc = (H - 1) / 2.0, lim = b.reach * W;
  auto dot = [&](bool m, double x, double y) {
    const long xi = long(std::floor(x + 0.5)), yi = long(std::floor(y + 0.5));
    if (xi < 0 || xi >= W || yi < 0 || yi >= H || x > lim) return;
    (m ? main : faint)[size_t(yi * W + xi)] = 1;
  };
  // a beam from (x0, y0) to (x1, y1), `thick` dots across; dashes moving right with `flow`
  const double shift = b.t * 24;
  auto beam = [&](bool m, double x0, double y0, double x1, double y1, int thick, bool flow) {
    const int n = std::max(1, int(std::ceil(std::max(std::fabs(x1 - x0), std::fabs(y1 - y0)) * 2)));
    for (int i = 0; i <= n; ++i) {
      const double x = x0 + (x1 - x0) * i / n, y = y0 + (y1 - y0) * i / n;
      if (flow && std::fmod(x - shift + 1e4, 5.0) >= 3.0) continue;  // a dash, a gap
      for (int d = 0; d < thick; ++d) dot(m, x, y - (thick - 1) / 2.0 + d);
    }
  };
  // the standing wave: four fringes three dots apart, light and dark in turn (the dark ones faint); they light up one
  // after another with `light`, and with `shimmer` the light ones drift between bright and faint
  const int nf = 4, b0 = int(std::lround(W * b.band));
  for (int i = 0; i < nf; ++i) {
    if (b.light * nf < i + 0.5) break;
    bool bright = i % 2 == 0;
    if (b.shimmer) bright = std::sin(2 * M_PI * 0.7 * b.t + 1.9 * i) > (i % 2 == 0 ? -0.5 : 0.6);
    const int x = b0 + 3 * i;
    for (int y = 0; y < H; ++y)
      if (std::fabs(y - yc) <= H * 0.46) {
        const long at = long(y) * W + x;
        if (x < W) (bright ? main : faint)[size_t(at)] = 1;  // the wave stands whatever the beam's reach
      }
  }
  const int b1 = b0 + 3 * (nf - 1) + 1;
  beam(true, 0, yc, b1, yc, 2, b.flow);  // in, two dots thick (H is even: centered)
  if (b.k > 0) {
    // out: k beams spread over the height; the middle one thick, the outer ones (higher orders) fainter; many beams
    // are one dot thin so that they stay apart
    const double half = H / 2.0 - 1, step = b.k > 1 ? std::min(half, 2 * half / (b.k - 1)) : 0;
    for (int i = 0; i < b.k; ++i) {
      const double o = (i - (b.k - 1) / 2.0) * step, order = std::fabs(i - (b.k - 1) / 2.0);
      beam(order < 1.6, b1, yc, W - 1, yc + o, order < 0.6 || b.k <= 3 ? 2 : 1, false);
    }
  }
  static const int bit[4][2] = {{0x01, 0x08}, {0x02, 0x10}, {0x04, 0x20}, {0x40, 0x80}};
  std::vector<std::string> out;
  for (int r = 0; r < rows; ++r) {
    std::string row, run;
    int style = 0;  // of `run`: 0 blank, 1 main, 2 faint
    auto flush = [&] {
      row += style == 1 ? accent(run) : style == 2 ? dim(accent(run)) : run;
      run.clear();
    };
    for (int c = 0; c < cols; ++c) {
      int m = 0, f = 0;
      for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 2; ++x) {
          const size_t at = size_t((r * 4 + y) * W + c * 2 + x);
          if (main[at]) m |= bit[y][x];
          if (faint[at]) f |= bit[y][x];
        }
      const int st = m ? 1 : f ? 2 : 0;
      if (st != style) flush(), style = st;
      if (!st) {
        run += ' ';
        continue;
      }
      const int v = m | f;
      run += char(0xE2), run += char(0xA0 | (v >> 6)), run += char(0x80 | (v & 0x3F));  // U+2800 + v
    }
    flush();
    out.push_back(row);
  }
  return out;
}

std::vector<std::string> logo() { return diffraction(kLogoCols, kLogoRows, Beam{}); }

bool can_animate() {
  if (!live() || !color() || !isatty(0)) return false;
  if (const char* ci = getenv("CI"); ci && *ci && strcmp(ci, "0") != 0 && strcmp(ci, "false") != 0) return false;
  if (env_on("KAPITSA_NO_ANIMATION")) return false;
  if (tcgetpgrp(1) != getpgrp()) return false;  // a background job: the terminal is someone else's
  return columns() >= 60 && term_rows() >= kLogoRows + 6;
}

std::vector<std::string> beside(const std::vector<std::string>& art, int art_cols, const std::vector<std::string>& text,
                                int at) {
  std::vector<std::string> out;
  const size_t n = std::max(art.size(), size_t(at) + text.size());
  for (size_t i = 0; i < n; ++i) {
    std::string l = "  " + left(i < art.size() ? art[i] : "", art_cols);
    if (i >= size_t(at) && i - size_t(at) < text.size()) l += "   " + text[i - size_t(at)];
    while (!l.empty() && l.back() == ' ') l.pop_back();
    out.push_back(l);
  }
  return out;
}

void intro(const std::vector<std::string>& text, int at) {
  if (!can_animate()) {
    std::string out;
    for (const auto& l : beside(logo(), kLogoCols, text, at)) out += l + "\n";
    fputs(out.c_str(), stdout);
    fflush(stdout);
    return;
  }
  (void)accent(" ");  // the terminal's colors are asked for before the keys are read
  Keys keys;
  // the wave lights up, the beam comes in from the left, crosses it and fans out into three
  const auto t0 = std::chrono::steady_clock::now();
  const double total = 1.35;
  for (int f = 0;; ++f) {
    const double t = double(f) / kFps;
    if (t >= total || keys.pressed()) break;
    Beam b;
    b.t = t;
    b.shimmer = t < 1.1;
    b.light = std::clamp(t / 0.35, 0.0, 1.0);
    b.reach = t < 0.25 ? 0 : std::min(1.0, (t - 0.25) / 0.85);  // at the wave by 0.65 s, out by 1.1 s
    b.k = b.reach > b.band + 0.12 ? 3 : 0;
    status(beside(diffraction(kLogoCols, kLogoRows, b), kLogoCols, text, at));
    sleep_frame(t0, f);
  }
  status(beside(logo(), kLogoCols, text, at));
  keep_status();
}

struct Loader::Impl {
  std::function<std::vector<std::string>()> text;
  int at = 0;
  double expected = 8;
  std::atomic<double> progress{0};
  std::atomic<bool> stop{false};
  std::thread th;
};

Loader::Loader(std::function<std::vector<std::string>()> text, int at, double expected) : impl_(new Impl) {
  impl_->text = std::move(text);
  impl_->at = at;
  impl_->expected = expected;
  if (!can_animate()) return;
  impl_->th = std::thread([this] {
    const auto t0 = std::chrono::steady_clock::now();
    for (int f = 0; !impl_->stop; ++f) {
      const double t = double(f) / kFps;
      // the fringes light up and the beam builds up toward the wave as the load goes on; dashes keep it moving
      const double p = std::max(impl_->progress.load(), 1 - std::exp(-t / (impl_->expected / 2)));
      Beam b;
      b.t = t;
      b.flow = b.shimmer = true;
      b.light = 0.2 + 0.8 * p;
      b.reach = 0.08 + (b.band + 0.04) * p;
      b.k = 0;
      status(beside(diffraction(kLogoCols, kLogoRows, b), kLogoCols, impl_->text(), impl_->at));
      sleep_frame(t0, f);
    }
  });
}

Loader::~Loader() {
  impl_->stop = true;
  if (impl_->th.joinable()) impl_->th.join();
  delete impl_;
}

void Loader::progress(double p) { impl_->progress = std::clamp(p, 0.0, 1.0); }

void Loader::done(const std::vector<std::string>& final) {
  const bool animated = impl_->th.joinable();
  impl_->stop = true;
  if (animated) {
    impl_->th.join();
    // ready: the beam crosses the wave and fans out (0.3 s), then stays
    const auto t0 = std::chrono::steady_clock::now();
    for (int f = 0; f < kFps * 3 / 10; ++f) {
      Beam b;
      b.t = double(f) / kFps;
      b.reach = b.band + (1 - b.band) * (f + 1) / (kFps * 0.3);
      status(beside(diffraction(kLogoCols, kLogoRows, b), kLogoCols, final, impl_->at));
      sleep_frame(t0, f);
    }
    status(beside(logo(), kLogoCols, final, impl_->at));
    keep_status();
    return;
  }
  for (const auto& l : beside(logo(), kLogoCols, final, impl_->at)) line(l, stderr);
}

}  // namespace ni::ui
