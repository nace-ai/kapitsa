// Kapitsa's pendulum (Pyotr Kapitsa, 1951): an inverted pendulum stands upright when its pivot vibrates fast enough,
// stability through fast vibration. Simulated (theta'' = (g + pivot acceleration) sin(theta) / l, the pivot at
// a cos(w t) with a^2 w^2 > 2 g l) and drawn in braille, 8 rows by 13 columns, in the accent color with a faint
// trail. The help screen plays it as an intro beside the wordmark; `kapitsa serve` and the coding agents' launcher
// show it wobbling while the model loads, upright once it is ready.
//
// It plays only in an interactive terminal (stdin, stdout and stderr, the foreground job) with colors, room for it,
// and none of CI or KAPITSA_NO_ANIMATION=1; never for a pipe (the plain output does not change). Any key or Ctrl+C
// skips the intro. The cursor and the terminal's modes come back however it ends.
#pragma once
#include <functional>
#include <string>
#include <vector>

namespace ni::ui {

constexpr int kArtRows = 8, kArtCols = 13;

// Whether the pendulum may play (see above)
bool can_animate();

// The art (kArtRows rows) with `text` beside it: text row i beside art row `at` + i
std::vector<std::string> beside(const std::vector<std::string>& art, const std::vector<std::string>& text, int at);

// The pendulum upright and still, its base vibrating
std::vector<std::string> still_art();

// The help screen's header: the intro where it may play (about 1.4 s), then its last frame stays; elsewhere that
// frame alone. Prints to stdout.
void intro(const std::vector<std::string>& text, int at);

// The pendulum wobbling while something loads, beside text() (called every frame, from the animation's thread). It
// calms as time goes on (a load takes 5-10 s), or as progress() in [0, 1] says. done() stands it upright and leaves it
// on screen beside `final`.
class Wobble {
 public:
  Wobble(std::function<std::vector<std::string>()> text, int at);
  ~Wobble();
  void progress(double p);
  void done(const std::vector<std::string>& final);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace ni::ui
