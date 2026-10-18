// What the image encoders' towers share on the GPU (qwen_vit.cpp, radio_vit.cpp): their bf16 linears, read once from
// the model's files into buffers of their own; the bf16 prefill GEMMs; vit.metal's LayerNorm, bias + activation,
// q / k / v split and attention; command buffers of bounded size.
#pragma once
#include "loader/safetensors.h"
#include "metal/gpu.h"

#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <vector>

namespace ni::vision {

constexpr int kVitRowChunk = 4096;     // rows per GEMM pass of an MLP or q / k / v (bounds the scratch memory)
constexpr double kVitCmdGflop = 150;   // work per command buffer (~25 ms): long command buffers hold up the display

Pipeline vit_pipe(const std::string& name);  // created once, then cached
nlohmann::json vit_read_json(const std::string& path);  // an empty object if the file is missing or not JSON
// The tensor names of dir's safetensors (from model.safetensors.index.json, else the files' headers)
std::vector<std::string> vit_tensor_names(const std::string& dir);

// y = x W^T + b: W bf16 [N, K] (K padded with zero columns, N with zero rows), b bf16 [N] (zeros without a bias)
struct VitLin {
  Buffer w, b;
  uint32_t N = 0, K = 0;
};

// The tensors of one vision tower (the names under `prefix` in dir's safetensors), read into buffers it owns
class VitWeights {
 public:
  ~VitWeights();
  void open(const std::string& dir, const std::string& prefix);  // the headers only (of the files that hold them)
  void close();                                                  // the headers go, the buffers stay
  const Tensor* find(const std::string& name) const;             // by the name after the prefix
  // A bf16 copy of a tensor (bf16, f16 or f32 in the file) into dst: `rows` rows of k values become rows of ld values
  // (zeros after k).
  void read_bf16(const std::string& name, uint16_t* dst, size_t rows, size_t k, size_t ld) const;
  // `name`.weight [N, K] (+ `name`.bias with `bias`, else zeros), padded to pad_n rows and pad_k columns
  VitLin lin(const std::string& name, uint32_t pad_n = 0, uint32_t pad_k = 0, bool bias = true);
  Buffer vec(const std::string& name);  // a bf16 copy of any tensor
  Buffer alloc(size_t bytes, const char* label);  // an owned buffer of zeros
  void make_resident();
  void release();  // every buffer

 private:
  std::unique_ptr<WeightStore> ws_;
  std::string prefix_;
  std::vector<Buffer> owned_;
};

// Y (+)= X W^T on the bf16 prefill GEMMs (qrf_f16_*; X half rows of x_ld, Y fp32 rows of y_ld); the bias is not added
void vit_gemm(Encoder& e, const VitLin& L, const Buffer& x, uint32_t x_ld, const Buffer& y, uint32_t y_ld, int M,
              bool add);

// GPU work in command buffers of about kVitCmdGflop each, committed as they fill; finish() waits for all of them
struct VitCmds {
  std::vector<std::unique_ptr<Encoder>> done;
  std::unique_ptr<Encoder> cur;
  double work = 0;
  Encoder& at(double gflop);
  void finish();
};

// LayerNorm (eps) of `rows` fp32 rows of x (x_ld floats apart) into half rows of out; with `pending`, that bf16 bias is
// added to x first (and written back: a linear's bias folded into the next norm)
void vit_ln(Encoder& e, const Buffer& x, uint32_t x_ld, const Buffer* pending, const Buffer& w, const Buffer& b,
            const Buffer& out, uint32_t out_ld, int rows, int D, float eps);
// The same LayerNorm in place, the rows staying fp32 (vit_ln_f32)
void vit_ln_inplace(Encoder& e, const Buffer& x, uint32_t x_ld, const Buffer* pending, const Buffer& w, const Buffer& b,
                    int rows, int D, float eps);
// rows of fp32 y (ld apart) + bias -> activation (0 none, 1 GELU tanh, 2 GELU erf, 3 ReLU squared) -> dst:
// vit_bias_act_h (half; columns [n, ld) written 0), vit_bias_act_b (bf16), vit_bias_f32 (fp32; act != 0 adds the bias)
void vit_act(Encoder& e, const char* kernel, const Buffer& y, uint32_t ld, const Buffer& bias, const Buffer& dst,
             uint32_t out_ld, int rows, int n, uint32_t act);
// fp32 q / k / v rows [rows, 3 * heads * HD] (from the GEMM) + bias -> half Q, K, V [heads][N][HD] at rows r0..
void vit_qkv_split(Encoder& e, const Buffer& y, const Buffer& bias, const Buffer& Q, const Buffer& K, const Buffer& V,
                   int rows, int r0, int N, int heads, int HD);
// Non-causal attention of N rows: Q, K, V half [heads][N][HD] -> out half [N][ld] (head h at columns h * HD), in
// command buffers of bounded work. The kernel: vit_attn_d<HD>_s4 (NI_VIT_ATTN names another of that head dimension).
void vit_attention(VitCmds& cs, const Buffer& Q, const Buffer& K, const Buffer& V, const Buffer& out, int N, int ld,
                   int heads, int HD);

}  // namespace ni::vision
