// A dense prefill projection on the Neural Engine (NI_ANE_PROJ): y = W x for every row of a chunk, W8A8 like the FFN
// split (ffn_split.h): the input rows rotated in 128-blocks and packed to int8 with a per-row scale, the weights rotated
// alike and staged as int8 under per-row scales one layer ahead into two double-buffered weight sets. A layer's
// projection may be several matrices over the same input (DeltaNet's in_proj_qkv and in_proj_z, attention's q, k, v),
// stacked into one output; unpack() writes each part's rows to its own fp32 buffer [T][N_part] for the GPU. Layers may
// differ in their parts: the program is built for the largest stack and a smaller one leaves the rest unused.
//   GPU: pack(lane, xh)  -> launch(lane, li)  ... other work ...  unpack(lane, outs)  (waits for the job)
#pragma once
#include "ane/ane.h"
#include "ane/ffn_split.h"
#include "metal/gpu.h"

#include <memory>
#include <string>
#include <vector>

namespace ni::ane {

struct ProjLayer {
  std::vector<Proj> parts;  // 1-3 matrices of the same K; rows a multiple of 32
};

struct ProjConfig {
  std::vector<uint32_t> fn_rows;  // ANE functions (rows); empty: 512, 1024, 2048
  std::string cache_dir, model_key, tag = "proj";
  int lanes = 2;
};

class ProjSplit {
 public:
  static constexpr uint32_t kMaxRows = FfnSplit::kMaxRows;
  // layers: one entry per model layer; layers with no parts are skipped (staging, launch throw for them).
  ProjSplit(const ProjConfig& c, std::vector<ProjLayer> layers);
  ~ProjSplit();
  ProjSplit(const ProjSplit&) = delete;
  ProjSplit& operator=(const ProjSplit&) = delete;

  uint32_t rows_out() const { return N_; }
  size_t metal_bytes() const { return bytes_; }
  std::string startup_note() const { return note_; }
  const std::vector<uint32_t>& fn_rows() const { return fn_rows_; }

  void begin_chunk();
  void pack(Encoder& e, int lane, const Buffer& xh, int T);
  void launch(Encoder& e, int lane, int li, int T);  // weight set: stage_set(li); ends the command buffer
  // waits for the lane's job (layer li's); writes part p's channels of T rows to outs[p] (fp32 [T][N_p])
  void unpack(Encoder& e, int lane, int li, const std::vector<Buffer>& outs, int T);
  void stage_layer(Encoder& e, int li, int set);
  int stage_set(int li) const { return set_of_.at(li); }
  bool finish_chunk(double* ane_ms = nullptr);
  int last_jobs() const { return last_jobs_; }
  void retire(const std::string& why);
  bool retired() const;
  std::string reason() const;
  double eval_sync(int set, int lane, int T);  // diagnostics: the ANE alone

 private:
  void init_resources();
  void free_all();
  std::string function(uint32_t R) const;
  void compute_row_scales();
  Buffer row_scales(int li, int part) const;
  int fn_for(int T) const;

  ProjConfig cfg_;
  std::vector<ProjLayer> layers_;
  std::vector<int> set_of_;      // weight set of each layer with parts (alternating in layer order), -1 otherwise
  std::vector<size_t> rs_off_;   // per layer: element offset of its row scales in rs_
  std::vector<std::vector<uint32_t>> part_n_;  // per layer
  uint32_t K_ = 0, N_ = 0, seg_ = 0, rot_sg_ = 8;
  std::vector<uint32_t> fn_rows_;
  struct Lane {
    std::vector<Surface> x;  // int8 [seg][2048] per input segment
    Surface tx, y;           // fp16 [1][2048], fp16 [N][2048]
    uint64_t done = 0;
  } lane_[2];
  struct Set {
    std::vector<Surface> w;  // int8 [N][seg] per input segment
    Surface s;               // fp16 [N][1]
  } set_[2];
  Buffer rot_, signs_, status_, rs_;
  std::vector<float> signs_host_;
  std::unique_ptr<Program> prog_;
  std::vector<std::shared_ptr<Program::Binding>> bind_[2][2];  // [set][lane][function]
  std::unique_ptr<Handoff> h_;
  int jobs_ = 0, last_jobs_ = 0;
  size_t bytes_ = 0;
  std::string note_;
};

// Metal memory of a ProjSplit of the largest stack of N rows over K inputs, `scale_rows` weight rows over all layers
// (their row scales) and `lanes` activation lanes (ProjSplit::metal_bytes)
size_t proj_bytes(uint32_t K, uint32_t N, size_t scale_rows, int lanes);

}  // namespace ni::ane
