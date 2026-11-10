// The kernels the Neural Engine split's GPU part runs a dense FFN layer on (src/model/ffn_share.h: gpu_part), the one
// rule plan() (from a pack's safetensors shapes), init_ane() and enc_mlp_gpu_part_qmt decide by:
//  - from_shape: an MLX affine projection's bits and group size from its U32 words and scale groups per row, on the
//    Qwen3.5-4B and Qwen3.8-27B FFN shapes (3-bit, 4-bit g64 / g32, 6-bit, 8-bit), and shapes that fit no format;
//  - gpu_part through plan()'s path (shapes -> formats): one format throughout takes the matrix units (4-bit g64: qmt;
//    3-bit, 8-bit, 4-bit g32: qmtf); a GGUF Q4_K_M import's layer (Q4_K gate/up -> 4-bit g32, Q6_K down -> 8-bit g64,
//    src/gguf/import.h) takes qmtf; 3-bit gate/up with a 4-bit g64 down (mlx-lm's mixed recipes), a 6-bit down, gate
//    and up in different formats, or a format whose tile is off keep the generic kernels and the older share;
//  - matrix_unit_share (plan()'s and init_ane()'s ffn_mma) over a whole model's layers: the 27B Q4_K_M's real mix
//    (33 layers Q4_K/Q4_K/Q6_K, 32 Q4_K throughout, read from its GGUF header) takes it; one odd layer drops it.
//   ffn_share_test      (no model, no GPU)
#include "model/ffn_share.h"

#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#define CHECK(condition) do { if (!(condition)) { \
  std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); std::abort(); \
} } while (false)

using ni::ffn_share::Fmt;
using ni::ffn_share::LayerFmt;
using ni::ffn_share::matrix_unit_share;
using ni::ffn_share::Path;
using ni::ffn_share::Qmtf;
using ni::ffn_share::from_shape;
using ni::ffn_share::gpu_part;

// a projection's U32 words and scale groups per row for K inputs in format f
struct Shape { int64_t words, groups; };
static Shape shape_of(int64_t K, Fmt f) { return {K * f.bits / 32, K / f.gs}; }

// plan()'s decision for one layer of a model with hidden width H and FFN width I: gate and up read H, down reads I
static Path plan_layer(int64_t H, int64_t I, Fmt gate, Fmt up, Fmt down, Qmtf q) {
  const Shape g = shape_of(H, gate), u = shape_of(H, up), d = shape_of(I, down);
  return gpu_part(from_shape(H, g.words, g.groups), from_shape(H, u.words, u.groups), from_shape(I, d.words, d.groups), q);
}

int main() {
  const Fmt b3{3, 64}, b4{4, 64}, q4k{4, 32}, b6{6, 64}, b8{8, 64};  // q4k: a GGUF Q4_0 / Q4_K import; b8: Q8_0 / Q6_K
  const Qmtf on{true, true, true}, off{};

  // from_shape on the 4B (hidden 2560, FFN 9216) and the 27B (5120, 17408)
  for (int64_t K : {2560, 9216, 5120, 17408})
    for (Fmt f : {b3, b4, q4k, b6, b8}) {
      const Shape s = shape_of(K, f);
      CHECK(from_shape(K, s.words, s.groups) == f);
    }
  CHECK(from_shape(2560, 240, 40) == b3);     // 3-bit: 2560 * 3 / 32 words, 2560 / 64 groups
  CHECK(from_shape(9216, 1152, 288) == q4k);  // a Q4_K down projection's import
  CHECK(from_shape(2560, 241, 40) == Fmt{});  // words that fit no whole width
  CHECK(from_shape(2560, 320, 41) == Fmt{});  // groups that do not divide K
  CHECK(from_shape(2560, 0, 40) == Fmt{});    // not an affine U32 pack
  CHECK(from_shape(0, 320, 40) == Fmt{});

  for (auto [H, I] : {std::pair<int64_t, int64_t>{2560, 9216}, {5120, 17408}}) {
    // one format throughout
    CHECK(plan_layer(H, I, b4, b4, b4, off) == Path::qmt);
    CHECK(plan_layer(H, I, b4, b4, b4, on) == Path::qmt);
    CHECK(plan_layer(H, I, b3, b3, b3, on) == Path::qmtf);
    CHECK(plan_layer(H, I, b8, b8, b8, on) == Path::qmtf);
    CHECK(plan_layer(H, I, q4k, q4k, q4k, on) == Path::qmtf);
    // a GGUF Q4_K_M import: Q4_K gate/up, Q6_K down (and its Q4_K-down layers above)
    CHECK(plan_layer(H, I, q4k, q4k, b8, on) == Path::qmtf);
    CHECK(plan_layer(H, I, b3, b3, b8, on) == Path::qmtf);
    // 3-bit gate/up with a 4-bit g64 down: the generic kernels, the older share
    CHECK(plan_layer(H, I, b3, b3, b4, on) == Path::generic);
    CHECK(plan_layer(H, I, b3, b3, b6, on) == Path::generic);
    CHECK(plan_layer(H, I, b4, b4, b8, on) == Path::generic);
    CHECK(plan_layer(H, I, b4, b4, q4k, on) == Path::generic);
    CHECK(plan_layer(H, I, b3, b4, b3, on) == Path::generic);
    CHECK(plan_layer(H, I, q4k, b8, b8, on) == Path::generic);
    CHECK(plan_layer(H, I, b6, b6, b6, on) == Path::generic);
    // a format whose tile is off (NI_QMTF=0, NI_QMM8=0)
    CHECK(plan_layer(H, I, b3, b3, b3, off) == Path::generic);
    CHECK(plan_layer(H, I, b8, b8, b8, Qmtf{true, false, true}) == Path::generic);
    CHECK(plan_layer(H, I, q4k, q4k, b8, Qmtf{true, false, true}) == Path::generic);
    CHECK(plan_layer(H, I, q4k, q4k, q4k, Qmtf{true, true, false}) == Path::generic);
  }
  // what init_ane passes for a projection that is not an affine pack (GGUF blocks, Q2_0, EXL3)
  CHECK(gpu_part(Fmt{}, Fmt{}, Fmt{}, on) == Path::generic);

  // whole models (plan()'s and init_ane()'s ffn_mma)
  auto model = [](int n, LayerFmt l) { return std::vector<LayerFmt>(size_t(n), l); };
  CHECK(matrix_unit_share(model(64, {b4, b4, b4}), off));     // all 4-bit g64
  CHECK(matrix_unit_share(model(32, {b3, b3, b3}), on));      // all qmtf, gate/up alike
  CHECK(matrix_unit_share(model(32, {b8, b8, b8}), on));
  std::vector<LayerFmt> q4km;                                 // the 27B Q4_K_M's GGUF header, layer by layer
  for (int i = 0; i < 65; ++i) q4km.push_back(i % 2 ? LayerFmt{q4k, q4k, q4k} : LayerFmt{q4k, q4k, b8});
  CHECK(matrix_unit_share(q4km, on));
  CHECK(!matrix_unit_share(q4km, Qmtf{true, false, true}));   // 8-bit tile off: its Q6_K downs go generic
  CHECK(!matrix_unit_share(model(32, {b3, b3, b4}), on));     // 3-bit gate/up + 4-bit g64 down
  CHECK(!matrix_unit_share(model(32, {q4k, b8, b8}), on));    // gate and up in different qmtf formats
  CHECK(!matrix_unit_share(model(32, {b3, b8, b3}), on));
  for (LayerFmt odd : {LayerFmt{b3, b3, b4}, LayerFmt{b6, b6, b6}, LayerFmt{b3, b8, b3}, LayerFmt{}})
    for (size_t at : {size_t(0), size_t(17), size_t(31)}) {   // one odd layer anywhere drops the whole model's share
      auto m4 = model(32, {b4, b4, b4}), m3 = model(32, {b3, b3, b3});
      m4[at] = m3[at] = odd;
      CHECK(!matrix_unit_share(m4, on));
      CHECK(!matrix_unit_share(m3, on));
    }
  CHECK(!matrix_unit_share({}, on));                          // no dense FFN layers
  std::printf("ffn_share_test: ok\n");
  return 0;
}
