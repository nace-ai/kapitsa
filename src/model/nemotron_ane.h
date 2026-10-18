// GPU + Neural Engine split of Nemotron-H's dense prefill work (NI_ANE). Per prefill chunk of T rows (T >= min rows):
//   Mamba layer: in_proj and out_proj split by output rows: the GPU computes the first g rows (its GEMMs), the ANE the
//                rest; the ANE's rows are joined in fp32 (in_proj into the zx rows, out_proj added into the residual).
//   MoE layer:   the shared expert (up, relu^2, down) on the ANE while the GPU runs the router and routed experts;
//                its output is added into the residual.
// The ANE runs W8A8 (int8 weights and activations after block Hadamard rotations, per-row weight and per-token input
// scales) as one multi-function MIL program; the GPU stages each op's int8 weights one op ahead from the checkpoint's
// NVFP4 / IQ2_XXS rows (nothing stored twice: the per-row scales are computed once and cached on disk). GPU and ANE
// are ordered through two shared events (ane::Handoff) with a watchdog; a failed job retires the split and the
// engine reruns the chunk on the GPU (Model::ane_finish_chunk).
// The scheme is src/ane/ffn_split's (the Qwen dense FFN split), applied to Nemotron-H's projections.
#pragma once
#include "ane/ane.h"
#include "metal/gpu.h"

#include <memory>
#include <string>
#include <vector>

namespace ni {

class NemotronH;
struct NhLayer;

struct NhAneConfig {
  double in_share = 0.7;    // fraction of in_proj rows on the ANE
  double out_share = 0.7;   // fraction of out_proj rows on the ANE
  bool ffn = true;          // the shared expert on the ANE
  bool ffn_a16 = true;      // its intermediate in fp16 (W8A16 down projection) instead of int8 per token
  uint32_t min_rows = 512;  // smallest chunk that splits
  std::vector<uint32_t> fn_rows;  // ANE functions (rows); default 512, 1024, 1536, 2048
  std::string cache_dir, model_key;
  bool compile_only = false;  // only compile the program into the ANE daemon's cache
  bool in_first = true;       // the ANE takes in_proj's first rows (z, then x), the GPU the rest (x, B, C, dt)
};

class NhAne {
 public:
  static constexpr uint32_t kSeg = 2048, kMaxRows = 2048;
  // Builds the surfaces, the row scales (or reads them from the cache) and the program. Throws std::runtime_error.
  NhAne(NemotronH& m, const NhAneConfig& c);
  ~NhAne();

  bool splits(int T) const;
  uint32_t gpu_in() const { return gin_; }    // in_proj rows the GPU computes
  uint32_t gpu_in0() const { return cfg_.in_first ? ain_ : 0; }  // the first of them
  uint32_t gpu_out() const { return gout_; }  // out_proj rows the GPU computes
  bool ffn() const { return cfg_.ffn; }
  // Starts a chunk of T rows and stages the first layers' weights.
  void begin_chunk(Encoder& e, int T);
  // Mamba layer li: xh = its normed rows (half [T][hidden]); zx = in_proj's fp32 rows [T][in_proj] (the ANE's rows go
  // to columns [0, ain) with in_first, else [gpu_in, in_proj)); yh = the gated norm's output (half [T][d_inner]); x = the residual (fp32 [T][hidden]).
  void in_launch(Encoder& e, int li, const Buffer& xh, int T);
  void in_join(Encoder& e, int li, const Buffer& zx, int T);
  void out_launch(Encoder& e, int li, const Buffer& yh, int T);
  void out_join(Encoder& e, int li, const Buffer& x, int T);
  // MoE layer li: the shared expert of xh (half [T][hidden]) added into x.
  void ffn_launch(Encoder& e, int li, const Buffer& xh, int T);
  void ffn_join(Encoder& e, int li, const Buffer& x, int T);
  // After the chunk's command buffer completed: false if a job failed (the chunk must rerun on the GPU alone).
  bool finish_chunk(double* ane_ms = nullptr);
  void retire(const std::string& why);
  bool retired() const;
  std::string reason() const;
  int last_jobs() const { return last_jobs_; }
  size_t metal_bytes() const { return bytes_; }
  const std::string& note() const { return note_; }
  double startup_ms() const { return startup_ms_; }

 private:
  struct Op;  // one op type's surfaces, bindings and per-layer row scales
  void build();
  void program(std::string& mil, std::vector<uint8_t>& blob);
  std::string fn_proj(const char* name, const Op& op, uint32_t R) const;
  std::string fn_ffn(uint32_t R) const;
  void row_scales();
  void stage(Encoder& e, int which, int li);  // which: 0 in, 1 out, 2 ffn; li: the model layer
  void pack(Encoder& e, const Buffer& xh, uint32_t D, int T, bool hidden);
  void launch(Encoder& e, int which, int T);
  void wait_join(Encoder& e);
  int next_layer(int which, int li) const;  // the next layer after li with this op, -1 if none
  int fn_for(int T) const;
  uint32_t ain0() const { return cfg_.in_first ? 0 : gin_; }  // the ANE's first in_proj row

  NemotronH& m_;
  NhAneConfig cfg_;
  uint32_t H_ = 0, DI_ = 0, NIN_ = 0, I_ = 0;
  uint32_t gin_ = 0, gout_ = 0, ain_ = 0, aout_ = 0;  // GPU / ANE rows of in_proj and out_proj
  std::vector<int> mamba_, moe_;                       // model layers with each op
  std::vector<uint32_t> fn_rows_;
  std::vector<uint32_t> dseg_;                         // the shared down's input segments (of I)
  std::unique_ptr<Op> in_, out_, ffn_;
  ane::Surface xs_[2], txs_, ys_[4], tys_;             // packed inputs: hidden (2 x 2048), d_inner (4 x 2048)
  Buffer rot_, signs_, status_;
  std::vector<float> signs_host_;
  std::unique_ptr<ane::Program> prog_;
  std::unique_ptr<ane::Handoff> h_;
  uint64_t pending_ = 0;  // the done value of the launched job not yet joined
  int fn_ = -1, T_ = 0, jobs_ = 0, last_jobs_ = 0;
  size_t bytes_ = 0;
  double startup_ms_ = 0;
  std::string note_;
};

}  // namespace ni
