// Which dense FFN layers get the Neural Engine split's matrix-unit GPU share (src/model/ffn_share.h), as plan() decides
// it from a pack's safetensors shapes and init_ane() from the loaded projections:
//  - from_shape: an MLX affine projection's bits and group size from its U32 words and scale groups per row, on the
//    Qwen3.5-4B and Qwen3.8-27B FFN shapes (3-bit, 4-bit g64 / g32, 6-bit, 8-bit), and shapes that fit no format;
//  - layer_mma: one format for gate, up and down, 4-bit g64 or one NI_QMTF runs. A mixed layer (mlx-lm's mixed
//    recipes: 3-bit gate/up with a 4-bit g64 or 6-bit down) keeps the older share, because enc_mlp_gpu_part_qmt runs
//    it on the generic kernels; an NI_QMTF format with that format switched off does too.
//   ffn_share_test      (no model, no GPU)
#include "model/ffn_share.h"

#include <cstdio>
#include <cstdlib>
#include <utility>

#define CHECK(condition) do { if (!(condition)) { \
  std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); std::abort(); \
} } while (false)

using ni::ffn_share::Fmt;
using ni::ffn_share::Qmtf;
using ni::ffn_share::from_shape;
using ni::ffn_share::layer_mma;

// a projection's U32 words and scale groups per row for K inputs in format f
struct Shape { int64_t words, groups; };
static Shape shape_of(int64_t K, Fmt f) { return {K * f.bits / 32, K / f.gs}; }

// plan()'s decision for one layer of a model with hidden width H and FFN width I: gate and up read H, down reads I
static bool plan_layer(int64_t H, int64_t I, Fmt gate, Fmt up, Fmt down, Qmtf q) {
  const Shape g = shape_of(H, gate), u = shape_of(H, up), d = shape_of(I, down);
  return layer_mma(from_shape(H, g.words, g.groups), from_shape(H, u.words, u.groups), from_shape(I, d.words, d.groups), q);
}

int main() {
  const Fmt b3{3, 64}, b4{4, 64}, b4g32{4, 32}, b6{6, 64}, b8{8, 64};
  const Qmtf on{true, true, true}, off{};

  // from_shape on the 4B (hidden 2560, FFN 9216) and the 27B (5120, 17408)
  for (int64_t K : {2560, 9216, 5120, 17408})
    for (Fmt f : {b3, b4, b4g32, b6, b8}) {
      const Shape s = shape_of(K, f);
      CHECK(from_shape(K, s.words, s.groups) == f);
    }
  CHECK(from_shape(2560, 240, 40) == b3);       // 3-bit: 2560 * 3 / 32 words, 2560 / 64 groups
  CHECK(from_shape(9216, 1152, 288) == b4g32);  // a GGUF Q4_K import's down projection
  CHECK(from_shape(2560, 241, 40) == Fmt{});    // words that fit no whole width
  CHECK(from_shape(2560, 320, 41) == Fmt{});    // groups that do not divide K
  CHECK(from_shape(2560, 0, 40) == Fmt{});      // not an affine U32 pack
  CHECK(from_shape(0, 320, 40) == Fmt{});

  for (auto [H, I] : {std::pair<int64_t, int64_t>{2560, 9216}, {5120, 17408}}) {
    // one format throughout: the matrix-unit share
    CHECK(plan_layer(H, I, b4, b4, b4, off));
    CHECK(plan_layer(H, I, b4, b4, b4, on));
    CHECK(plan_layer(H, I, b3, b3, b3, on));
    CHECK(plan_layer(H, I, b8, b8, b8, on));
    CHECK(plan_layer(H, I, b4g32, b4g32, b4g32, on));
    // an NI_QMTF format whose tile is off, and a format with no tile at all: the older share
    CHECK(!plan_layer(H, I, b3, b3, b3, off));
    CHECK(!plan_layer(H, I, b3, b3, b3, Qmtf{false, true, true}));
    CHECK(!plan_layer(H, I, b8, b8, b8, Qmtf{true, false, true}));
    CHECK(!plan_layer(H, I, b4g32, b4g32, b4g32, Qmtf{true, true, false}));
    CHECK(!plan_layer(H, I, b6, b6, b6, on));
    // mixed layers: the older share (the review's case first: 3-bit gate/up, 4-bit g64 down)
    CHECK(!plan_layer(H, I, b3, b3, b4, on));
    CHECK(!plan_layer(H, I, b3, b3, b6, on));
    CHECK(!plan_layer(H, I, b4, b4, b8, on));
    CHECK(!plan_layer(H, I, b3, b3, b8, on));
    CHECK(!plan_layer(H, I, b3, b4, b3, on));
    CHECK(!plan_layer(H, I, b4, b4g32, b4, on));
  }
  // what init_ane passes for a projection that is not an affine pack (GGUF blocks, Q2_0, EXL3)
  CHECK(!layer_mma(Fmt{}, Fmt{}, Fmt{}, on));
  std::printf("ffn_share_test: ok\n");
  return 0;
}
