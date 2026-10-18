// Kapitsa's look: the Kapitza-Dirac effect (Kapitsa and Dirac predicted it in 1933; it was seen in 2001). An electron
// beam that crosses a standing wave of light comes out as several beams, the way one step of speculative decoding (the
// drafter, then the verify) comes out as several tokens. Drawn in braille: the beam from the left, the wave's fringes
// (light and dark, standing), the diffracted beams to the right, the outer ones fainter like higher orders. Its
// three-beam frame reads as a K.
//
// - The help screen's header: the beam's intro (about 1.4 s; any key or Ctrl+C skips it), whose last frame stays as
//   the logo beside the wordmark.
// - `kapitsa serve` and the coding agents' launcher: the logo as a loader (the fringes light up and the beam builds up
//   as the model loads; at "ready" it fans out and stays).
// - `kapitsa serve`'s dashboard: one live beam per request being written, fanning out into the tokens its last step
//   accepted.
//
// Animations play only in an interactive terminal (stdin, stdout and stderr, the foreground job) with colors, room
// for them, and none of CI or KAPITSA_NO_ANIMATION=1; never for a pipe (the plain output does not change). The cursor
// and the terminal's modes come back however they end.
#pragma once
#include <functional>
#include <string>
#include <vector>

namespace ni::ui {

constexpr int kLogoRows = 5, kLogoCols = 14;

// One frame of the drawing
struct Beam {
  int k = 3;              // beams out of the wave (0: none yet)
  double reach = 1;       // how far from the left the beam has come, 0..1 (0.4: at the wave)
  double light = 1;       // how lit the wave is, 0..1: its fringes light up one after another
  double t = 0;           // seconds: the fringes' shimmer and the dashes' flow
  bool flow = false;      // the beam coming in as dashes moving right
  bool shimmer = false;   // the fringes' brightness drifting
  double band = 0.36;     // where the wave stands, as a share of the width
};
std::vector<std::string> diffraction(int cols, int rows, const Beam& b);

// The logo: kLogoRows x kLogoCols, the beam diffracted into three
std::vector<std::string> logo();

// Whether an animation may play (see above)
bool can_animate();

// `art` (`art_cols` wide) with `text` beside it, text row i beside art row `at` + i
std::vector<std::string> beside(const std::vector<std::string>& art, int art_cols, const std::vector<std::string>& text,
                                int at);

// The help screen's header: the logo with `text` beside it (from row `at`), its intro first where it may play. To
// stdout.
void intro(const std::vector<std::string>& text, int at);

// The logo as a loader beside text() (called each frame, from the animation's thread): the fringes light up and the
// beam builds up with progress() in [0, 1] (or with time, over about `expected` seconds). done() fans it out and
// leaves it on screen beside `final`. Where no animation may play: nothing until done(), which prints the logo.
class Loader {
 public:
  Loader(std::function<std::vector<std::string>()> text, int at, double expected = 8);
  ~Loader();
  void progress(double p);
  void done(const std::vector<std::string>& final);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace ni::ui
