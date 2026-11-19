// Routed experts in MLX affine 3- or 4-bit, group 32 / 64 / 128, streamed from SSD through an arena of slots (Hy4-flash:
// 41 MoE layers x 512 experts x 3 matrices of 1536 x 1536, 83.6 GB at 4-bit group 64; src/model/expert_store.h holds
// the cache). The format is per layer, and within a layer gate / up share one and down may have another. The experts'
// SwiGLU clamps: gate <= limit, -limit <= up <= limit (Hy4's swiglu_limit 10). The router, the shared expert, the
// latent down/up projections and the combine norm stay with the caller.
#pragma once
#include "metal/gpu.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ni::a4 {

struct Shape {
  uint32_t D = 1536;  // the experts' input / output width (Hy4: the latent, routed_expert_hidden_size)
  uint32_t I = 1536;  // moe_intermediate_size
  uint32_t k = 16;    // experts per row
  uint32_t E = 512;
  float limit = 10.0f;  // swiglu_limit; <= 0: no clamp
};

// A matrix's MLX affine format: bits 3 or 4, group size 32, 64 or 128
struct Fmt {
  uint32_t bits = 4, gs = 64;
  bool operator==(const Fmt& o) const { return bits == o.bits && gs == o.gs; }
  bool operator!=(const Fmt& o) const { return !(*this == o); }
};
std::string fmt_name(const Fmt& f);  // "3-bit group 64"

// One layer's experts: stacked planes (tests; gate/up w [E, I, D * bits / 32] uint32, s/b [E, I, D / gs] bf16, down
// likewise [E, D, I * bits / 32] / [E, D, I / gs]) or slots of the store's arena through the layer's slot table.
struct Experts {
  Buffer gw, gs, gb, uw, us, ub, dw, ds, db;
  Buffer a0, a1, slots;
  bool arena = false;
  uint32_t off[9] = {};  // plane byte offsets inside a slot, in the order gw gs gb uw us ub dw ds db
  Fmt gu, dn;            // gate / up's format, down's
};

// every pipeline enc_rows can use (all formats), or those of the given gate / up and down formats
std::vector<std::string> kernel_names();
std::vector<std::string> kernel_names(const Fmt& gu, const Fmt& dn);

// Whether the kernels take these shapes (why: the first reason they don't): D and I multiples of 64, D <= 2048 (the
// rows path stages x in threadgroup memory), 1 <= k <= 16, E <= 512 (the prefill bucketing)
bool shapes_ok(const Shape& sh, std::string* why = nullptr);
// shapes_ok and the formats: bits 3 or 4, group 32 / 64 / 128 dividing the matrices' K (D for gate / up, I for down)
bool formats_ok(const Shape& sh, const Fmt& gu, const Fmt& dn, std::string* why = nullptr);

// An expert's slot in the store's arena: the nine planes' byte offsets (Experts::off order) and the slot's bytes in
// whole 16 KiB pages. The three code planes come first, each on a page boundary, then the six scale / bias planes back
// to back. Hy4 at 4-bit group 64: code planes of 72 pages, scale / bias planes of 73,728 B, 243 pages a slot; 3-bit
// group 64: 54-page code planes, 189 pages; 3-bit group 128: 176 pages.
struct SlotLayout {
  uint32_t off[9] = {}, len[9] = {};
  uint32_t bytes = 0;
};
SlotLayout slot_layout(const Shape& sh, const Fmt& gu = Fmt{}, const Fmt& dn = Fmt{});

// Rows (decode R = 1, any R up to tmax): y fp32 [R, D] = sum_j wts[r, j] * down_e(silu(clamp(gate_e x)) * clamp(up_e x))
// with e = idx[r, j]; x fp32 [R, D]; idx uint32 [R, k]; wts fp32 [R, k]. y is overwritten. scratch: scratch_bytes.
void enc_rows(Encoder& e, const Experts& x, const Shape& sh, int R, const Buffer& xin, const Buffer& idx,
              const Buffer& wts, const Buffer& scratch, const Buffer& y);
size_t scratch_bytes(const Shape& sh, int tmax);
// enc_rows with the path chosen: Rows (one matvec per (row, expert): decode, verify), Tiles (the pairs sorted by expert
// into tiles of 32 rows, a GEMM per tile: prefill), Auto (enc_rows: Rows up to kRowsMax rows; on the M5 Pro at Hy4's
// shapes with uniform routing the rows path is ahead up to ~96 rows: 9.5 vs 12.6 ms at 64, 18.2 vs 14.5 at 128). The ids
// must be < E.
enum class Path { Auto, Rows, Tiles };
constexpr int kRowsMax = 64;
void enc_rows(Encoder& e, const Experts& x, const Shape& sh, int R, const Buffer& xin, const Buffer& idx,
              const Buffer& wts, const Buffer& scratch, const Buffer& y, Path path);

// The pack's streamed experts. dir: the model folder (MLX safetensors; the planes are
// model.layers.{L}.mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}, [E, out, in * bits / 32] and
// [E, out, in / gs]). layers: the trunk layer indices that have routed experts. Each layer's formats come from its
// planes' shapes (checked against config.json's quantization entries when the folder has one); every slot is the
// largest layer's (a layer in a smaller format leaves the rest of its slot unused). arena_bytes 0: decide from free
// memory (NI_EXPERT_STORE=<GB>|all overrides).
class Store {
 public:
  Store(const std::string& dir, const std::vector<int>& layers, const Shape& sh, size_t arena_bytes);
  ~Store();
  // After the dispatches that write layer's routing ids (n uint32 in GPU memory); work encoded between begin and end
  // overlaps the reads. pred: ids predicted for the next MoE layer (read ahead), or nullptr. scan: a prefill chunk's.
  // layer: a trunk layer index of the constructor's list. The command buffer must be committed for the handshake to
  // run (the GPU signals, the store's thread answers); end() before the layer's enc_rows.
  void begin(Encoder& e, int layer, const Buffer& ids, uint32_t n, const Buffer* pred, uint32_t npred, bool scan);
  void end(Encoder& e);
  Experts experts(int layer) const;
  std::vector<Buffer> buffers() const;  // residency
  std::vector<std::string> kernel_names() const;  // the pipelines of the layers' formats
  std::string describe() const;
  size_t arena_bytes() const;
  bool complete() const;  // every expert in the arena: begin / end encode nothing
  struct Counts {
    uint64_t handshakes = 0, lookups = 0, misses = 0, prefetches = 0, evictions = 0;
    double mb_read = 0, stall_ms = 0;
  };
  Counts counts() const;  // since the store was made

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ni::a4
