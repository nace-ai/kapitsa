// CPU test of the split's placement (src/cluster/place.h): no GPU, no model. Two equal Macs take half the layers each;
// a Mac with twice the bandwidth takes more of them; memory bounds a run; the head holds its own tensors and the
// n-gram block's layers; with more members than layers, or too little memory, nothing fits.
//   split_place_test
#include "cluster/place.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace ni::cluster;

namespace {

int g_fail = 0;
#define CHECK(c, ...)                                      \
  do {                                                     \
    if (!(c)) {                                            \
      ++g_fail;                                            \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__);                        \
      fprintf(stderr, "\n");                               \
    }                                                      \
  } while (0)

// 48 layers of 0.8 GB, a decode step reading 60 MB of each (Flash-Next's Q2_0 pack, roughly); the head's tensors 1 GB
std::vector<LayerCost> layers48() { return std::vector<LayerCost>(48, LayerCost{0.8e9, 60e6}); }
const LayerCost kHead{1e9, 300e6};

}  // namespace

int main() {
  {  // two equal Macs: the head's tensors make its run a little shorter
    const Placement p = place(layers48(), kHead, 2, {{"a", 30e9, 273}, {"b", 30e9, 273}});
    CHECK(p.fits, "two 30 GB Macs hold 39.4 GB");
    CHECK(p.lo.size() == 3 && p.lo[0] == 0 && p.lo[2] == 48, "runs cover the layers");
    CHECK(p.lo[1] >= 23 && p.lo[1] <= 25, "an even split, not %d", p.lo[1]);
    CHECK(p.held[0] <= 30e9 && p.held[1] <= 30e9, "within memory");
    // the time: every byte read once, at 273 GB/s
    const double want = (48 * 60e6 + 300e6) / 273e6;
    CHECK(std::fabs(p.ms - want) < 1e-6, "%.4f ms, want %.4f", p.ms, want);
    CHECK(std::fabs(whole_ms(layers48(), kHead, 273) - want) < 1e-6, "whole_ms");
  }
  {  // the faster Mac takes as much as its memory allows
    const Placement p = place(layers48(), kHead, 2, {{"a", 30e9, 273}, {"fast", 30e9, 546}});
    CHECK(p.fits && p.lo[1] < 24, "the head keeps the minimum the fast Mac's memory leaves it: %d", p.lo[1]);
    CHECK(p.held[1] <= 30e9, "within the fast Mac's memory");
  }
  {  // memory bounds the runs: a 12 GB Mac holds at most 15 layers
    const Placement p = place(layers48(), kHead, 2, {{"head", 40e9, 273}, {"small", 12e9, 546}});
    CHECK(p.fits && 48 - p.lo[1] <= 15, "the small Mac holds %d layers", 48 - p.lo[1]);
  }
  {  // the head keeps the n-gram block's layers even when the stage is far faster
    const Placement p = place(layers48(), kHead, 2, {{"head", 40e9, 100}, {"fast", 40e9, 800}});
    CHECK(p.fits && p.lo[1] >= 2, "the head holds layers 1-2, not %d", p.lo[1]);
  }
  {  // three Macs, each at least one layer, in ring order
    const Placement p = place(layers48(), kHead, 2, {{"a", 14e9, 273}, {"b", 14e9, 273}, {"c", 14e9, 273}});
    CHECK(p.fits && p.lo.size() == 4 && p.lo[0] == 0 && p.lo[1] < p.lo[2] && p.lo[2] < p.lo[3] && p.lo[3] == 48,
          "three runs");
  }
  {  // too little memory, too many members
    CHECK(!place(layers48(), kHead, 2, {{"a", 15e9, 273}, {"b", 15e9, 273}}).fits,
          "two 15 GB Macs do not hold 39.4 GB");
    const std::vector<LayerCost> two(2, LayerCost{1, 1});
    CHECK(!place(two, kHead, 1, {{"a", 9e9, 1}, {"b", 9e9, 1}, {"c", 9e9, 1}}).fits, "three Macs, two layers");
  }
  {  // unknown bandwidth: the slowest known member's
    const Placement p = place(layers48(), kHead, 2, {{"a", 30e9, 0}, {"b", 30e9, 273}});
    CHECK(p.fits && p.lo[1] >= 23 && p.lo[1] <= 25, "an unknown Mac counts as the known one: %d", p.lo[1]);
  }
  printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
