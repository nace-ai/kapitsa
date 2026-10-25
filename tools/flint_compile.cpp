// Compile check of the Flint kernels (src/kernels/flint): builds the library and every flint_* pipeline, and the base
// library's pipelines Flint runs (no GPU work), a dense Llama's on its backend (Nanbeige4.3: 8-bit and bf16 forms)
// included; on an Apple10 GPU under macOS 26 the tensor library's Flint attention too. flint_compile [kernel ...]
// limits it to those kernels.
#include "metal/gpu.h"
#include "util/log.h"

#include <cstdio>
#include <string>

namespace ni {
extern const char* const kFlintKernelSource;
extern const char* const kTensorKernelSource;
}
using namespace ni;

int main(int argc, char** argv) {
  Gpu& g = Gpu::get();
  g.compile_extra("Flint", kFlintKernelSource);
  bool tensor = false;  // (as Flint's qmt_available: Metal 4's tensor_ops)
  if (__builtin_available(macOS 26.0, *)) tensor = g.apple_family() >= 10;
  if (tensor) g.compile_extra("tensor", kTensorKernelSource, true);
  std::vector<std::string> names;
  for (int i = 1; i < argc; ++i) names.push_back(argv[i]);
  if (names.empty()) {  // every Flint kernel of the library, and the base library's that src/model/flint.cpp runs
    for (const std::string& n : g.function_names())
      if (n.rfind("flint_", 0) == 0) names.push_back(n);
    for (const char* n : {"qmvf_b4_t1_r4", "qmvf_b4_t1_r4_norm", "qmvf_gu_b4_t1_r4_norm", "embed_f8g64", "embed_q4",
                          "rows_to_half", "qmr_k32_w14_m32_n128", "qmr_k64_w14_m16_n64", "qmr_k64_w12_m8_n64",
                          "qrf_f16_m32_n128", "qrf_f16_m16_n64", "qrf_f16_m8_n64", "moe_bucket3", "qrm_gu_k32_m32_n64",
                          "qrm_dn_k32_m32_n128", "swiglu_half", "moe_combine", "moe_gate_up3_r2", "moe_down2_w", "moe_gate_up", "moe_down", "moe4_unique", "moe4_gate_up",
                          "moe4_down", "moe_bucket", "qmm_moe_b4_m16_n64",
                          "topk_stage1", "sample_stage2", "copy_u32", "kv_regrow", "flint_route",
                          // a dense Llama's 8-bit and bf16 forms
                          "qmvf_b8_t1_r4", "qmvf_b8_t1_r4_norm", "qmvf_gu_b8_t1_r4_norm", "qmr8_k32_w14_m32_n128",
                          "qmr8_k32_w14_m16_n64", "embed_f16", "qmvg_f16_r2_add", "qmvg_f16_r2_norm", "qmvg_f16_r4_norm",
                          "qmvg_multi_f16", "mlp_gate_up_f16"})
      names.push_back(n);
    if (tensor)
      for (const char* n : {"qmt_m64_n64", "qattn_flint_s8k32p", "qattn_flint_s4k16", "qattn_flint_s8k32p_g7",
                            "qattn_flint_s4k16_g7", "qmtm_g_m64_n64", "qmtm_g_m64_n64_o"})
        names.push_back(n);
    // a GGUF import's IQ experts (src/kernels/wfmt_iq.metal): the forms Flint runs, in every format
    for (const std::string f : {"iq1s", "iq2xxs", "iq2s", "iq3xxs", "iq3s"})
      for (const std::string& n : {"moei_gu_rows_" + f + "_r2_s4", "moei_down_rows_" + f + "_r8_s4",
                                   "qrmi_gu_" + f + "_m32_n64_k32_f32", "qrmi_dn_" + f + "_m32_n64"})
        names.push_back(n);
  }
  int bad = 0;
  for (auto& n : names) {
    Pipeline p = g.pipeline(n, false);
    if (!p.pso) {
      ++bad;
      printf("%-28s FAILED\n", n.c_str());
      continue;
    }
    printf("%-28s max threads %4u, threadgroup memory %5u B\n", n.c_str(), p.max_threads, p.tg_mem);
  }
  printf("%d kernels failed\n", bad);
  return bad ? 1 : 0;
}
