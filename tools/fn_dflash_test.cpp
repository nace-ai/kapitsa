// Qwen3.8-Flash-Next's DFlash2 plumbing without a model.
//   fn_dflash_test          CPU: a target's residual streams from its config.json (hc_count), the reduction a draft
//                           must state for it (dflash_config.target_hidden_reduction: stream_mean for a hyper-connected
//                           target, none for another, absent = the target's) through DFlash2's load-time config checks
//   fn_dflash_test --gpu    also the capture kernel (hc_capture_mean, small rows) against a CPU reference: the mean of
//                           a row's 4 streams after the layer (X + inj * y, the block output not yet injected) in bf16
//                           columns of a wider capture row; under the GPU lock, best with MTL_SHADER_VALIDATION=1
#include "metal/gpu.h"
#include "model/dflash2.h"
#include "model/qwen35.h"
#include "util/bf16.h"
#include "util/log.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace ni;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_fail;
}

// A DFlash2 draft config for a target of hidden 2560 and vocabulary 248320 (Flash-Next's), with `reduction` as its
// dflash_config.target_hidden_reduction ("" leaves the key out)
std::string draft_config(const std::string& reduction) {
  std::string red = reduction.empty() ? "" : "\"target_hidden_reduction\": \"" + reduction + "\", ";
  return R"({"architectures": ["DFlash2DraftModel"], "is_causal": false,
  "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": 248077,
    "selector_rank": 256, "selector_top_k": 16, )" +
         red + R"("target_layer_ids": [3, 17, 31, 47]},
  "head_dim": 128, "hidden_size": 2560, "intermediate_size": 5120, "num_attention_heads": 32,
  "num_key_value_heads": 8, "num_hidden_layers": 3,
  "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention"], "sliding_window": 2048,
  "rms_norm_eps": 1e-06, "rope_theta": 10000000, "tie_word_embeddings": false, "vocab_size": 248320})";
}

bool has(const std::vector<std::string>& v, const std::string& s) {
  for (const std::string& x : v)
    if (x.find(s) != std::string::npos) return true;
  return false;
}

// a text config of Flash-Next's shapes, hyper-connected (hc_count 4) or not
std::string target_config(bool hc) {
  return std::string(R"({"model_type": "qwen4_exp", "text_config": {"model_type": "qwen4_exp_text", "hidden_size": 2560,
  "num_hidden_layers": 48, "vocab_size": 248320, "linear_num_key_heads": 16, "linear_num_value_heads": 48,
  "linear_key_head_dim": 128, "linear_value_head_dim": 128, "num_attention_heads": 24, "num_key_value_heads": 2,
  "head_dim": 256, "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
  "shared_expert_intermediate_size": 640, "eos_token_id": 248044)") +
         (hc ? R"(, "hc_count": 4)" : "") + "}}";
}

void run_cpu() {
  std::printf("target streams and the draft's reduction (DFlash2 config checks)\n");
  const char* td = getenv("TMPDIR");
  std::string base = std::string(td && *td ? td : "/tmp") + "/fn_dflash_test.XXXXXX";
  std::vector<char> b(base.begin(), base.end());
  b.push_back('\0');
  const char* dir = mkdtemp(b.data());
  if (!dir) fatal("mkdtemp failed");
  Qwen35Config flash, plain;
  for (int hc = 0; hc < 2; ++hc) {
    const std::string path = std::string(dir) + "/config.json";
    std::ofstream(path) << target_config(hc == 1);
    (hc ? flash : plain) = Qwen35::read_config(dir);
    unlink(path.c_str());
  }
  rmdir(dir);
  check(flash.hc == 4 && plain.hc == 1, "config.json's hc_count: 4 streams (Flash-Next), 1 without it");
  check(DFlash2::target_reduction(flash) == "stream_mean" && DFlash2::target_reduction(plain) == "none",
        "a hyper-connected target's reduction is stream_mean, another's none");
  const auto ok_sm = DFlash2::config_check(draft_config("stream_mean"), flash);
  check(ok_sm.empty(), "Flash-Next + a draft stating stream_mean: loads" +
                           (ok_sm.empty() ? std::string() : " (problems: " + ok_sm[0] + ")"));
  check(DFlash2::config_check(draft_config(""), flash).empty(), "Flash-Next + a draft not stating it: loads (the target's)");
  const auto bad_none = DFlash2::config_check(draft_config("none"), flash);
  check(bad_none.size() == 1 && has(bad_none, "target_hidden_reduction stream_mean"),
        "Flash-Next + a draft stating none: refused, for that reason only");
  const auto bad_other = DFlash2::config_check(draft_config("stream_sum"), flash);
  check(has(bad_other, "target_hidden_reduction"), "Flash-Next + a draft stating another reduction: refused");
  const auto bad_sm = DFlash2::config_check(draft_config("stream_mean"), plain);
  check(bad_sm.size() == 1 && has(bad_sm, "target_hidden_reduction none"),
        "a target without streams + a draft stating stream_mean: refused");
  check(DFlash2::config_check(draft_config("none"), plain).empty() && DFlash2::config_check(draft_config(""), plain).empty(),
        "a target without streams + none or nothing: loads");
  check(!DFlash2::config_check("[1, 2]", flash).empty(), "not a config: refused");

  // The drafting block the memory plan reserves rows for: a pure decision on (target, slots, NI_DFLASH_BLOCK), asked
  // several times in one process here, as an --allow-switch server does (each model decides its own at its load).
  std::printf("drafting block (DFlash2::plan_block_rows)\n");
  Qwen35Config dense = plain, moe = plain;
  dense.moe = false;
  moe.moe = true;
  check(DFlash2::plan_block_rows(dense, 1, "", true) == 16, "one slot, dense target, unset: 16 rows");
  check(DFlash2::plan_block_rows(dense, 1, "auto", true) == 16, "NI_DFLASH_BLOCK=auto is the default: 16 rows");
  check(DFlash2::plan_block_rows(dense, 1, "8", true) == 8, "NI_DFLASH_BLOCK=8: 8 rows");
  check(DFlash2::plan_block_rows(dense, 1, "16", true) == 16, "NI_DFLASH_BLOCK=16, dense target: 16 rows");
  check(DFlash2::plan_block_rows(dense, 2, "", true) == 8, "two slots: 8 rows (a 16-row block has no batched cycle)");
  check(DFlash2::plan_block_rows(moe, 1, "", true) == 8, "one slot, MoE target: 8 rows (no wide tree verify to run)");
  check(DFlash2::plan_block_rows(moe, 1, "16", true) == 16,
        "NI_DFLASH_BLOCK=16, MoE target: 16 rows (the load-time check refuses it)");
  setenv("NI_DFLASH_BLOCK", "8", 1);
  const int env8 = DFlash2::plan_block_rows(dense, 1, nullptr, true);
  unsetenv("NI_DFLASH_BLOCK");
  check(env8 == 8 && DFlash2::plan_block_rows(dense, 1, nullptr, true) == 16, "the environment is read when no value is passed: 16, 8, 16");
  const int first = DFlash2::plan_block_rows(dense, 1, "", true), moe_next = DFlash2::plan_block_rows(moe, 1, "", true),
            back = DFlash2::plan_block_rows(dense, 1, "", true);
  check(first == 16 && moe_next == 8 && back == 16,
        "16 then 8 then 16 in one process: the next model decides for itself");
  // the cheap half of rows16_auto_ok: the knobs that make the load keep 8 rows take the plan with them, so a start
  // that cannot run the block does not reserve its rows
  setenv("NI_KFLASH", "0", 1);
  const int kf0 = DFlash2::plan_block_rows(dense, 1, "", true);
  unsetenv("NI_KFLASH");
  setenv("NI_DFLASH_VERIFY", "7", 1);
  const int shallow = DFlash2::plan_block_rows(dense, 1, "", true);
  unsetenv("NI_DFLASH_VERIFY");
  setenv("NI_DFLASH_ATTN", "scalar", 1);
  const int scalar = DFlash2::plan_block_rows(dense, 1, "", true);
  unsetenv("NI_DFLASH_ATTN");
  check(kf0 == 8 && shallow == 8 && scalar == 8,
        "NI_KFLASH=0, NI_DFLASH_VERIFY=7, NI_DFLASH_ATTN=scalar: the plan keeps 8 rows");
  check(DFlash2::plan_block_rows(dense, 1, "", true) == 16, "and with the knobs back off: 16 rows");
  check(DFlash2::plan_block_rows(dense, 1, "", false) == 8,
        "no matrix units on this Mac (an M1-M4, or CI's virtual GPU): the plan keeps 8 rows too");
  // incoai/Qwen3.8-Flash-Next-DFlash2 @ a22a1744 (Apache-2.0), its config.json's fields that the checks read: 5 draft
  // layers over target layers 5, 15, 25, 35, 45, rope theta 1e4, no target_hidden_reduction (the target's: stream_mean)
  const std::string incoai = R"({"architectures": ["DFlash2DraftModel"], "is_causal": false,
  "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": 248077,
    "selector_rank": 256, "selector_top_k": 16, "target_layer_ids": [5, 15, 25, 35, 45]},
  "head_dim": 128, "hidden_size": 2560, "intermediate_size": 7680, "num_attention_heads": 32, "num_hidden_layers": 5,
  "num_key_value_heads": 8, "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
  "sliding_attention", "sliding_attention"], "sliding_window": 2048, "rms_norm_eps": 1e-06,
  "rope_parameters": {"rope_theta": 10000, "rope_type": "default"}, "tie_word_embeddings": false, "vocab_size": 248320})";
  const auto inc = DFlash2::config_check(incoai, flash);
  check(inc.empty(), "incoai's Flash-Next DFlash2 config (rope theta 1e4): loads for Flash-Next" +
                         (inc.empty() ? std::string() : " (problems: " + inc[0] + ")"));
}

// hc_capture_mean against the CPU: R rows of S streams of D, the block output y and its injection weights, into slot
// `slot` of capture rows ld = 3 D wide from row dst_row0
void run_gpu() {
  std::printf("hc_capture_mean (small rows) against a CPU reference\n");
  constexpr uint32_t R = 5, D = 256, S = 4, slots = 3, slot = 1, row0 = 2;
  const uint32_t ld = slots * D;
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> u(-2.0f, 2.0f);
  Gpu& g = Gpu::get();
  Buffer X = g.alloc(size_t(R) * S * D * 4, "X"), y = g.alloc(size_t(R) * D * 4, "y"),
         inj = g.alloc(size_t(R) * S * 4, "inj"), dst = g.alloc(size_t(row0 + R + 1) * ld * 2, "dst");
  for (size_t i = 0; i < size_t(R) * S * D; ++i) X.as<float>()[i] = u(rng);
  for (size_t i = 0; i < size_t(R) * D; ++i) y.as<float>()[i] = u(rng);
  for (size_t i = 0; i < size_t(R) * S; ++i) inj.as<float>()[i] = u(rng) * 0.5f;
  for (int has_y = 0; has_y < 2; ++has_y) {
    std::memset(dst.ptr(), 0xFF, dst.size);
    const struct {
      uint32_t R, D, S, ld, col0, dst_row0, has_y;
    } p{R, D, S, ld, slot * D, row0, uint32_t(has_y)};
    auto e = g.begin();
    e->use(g.pipeline("hc_capture_mean"));
    e->buf(0, X); e->buf(1, y); e->buf(2, inj); e->buf(3, dst);
    e->val(4, p);
    e->dispatch((D + 255) / 256, R, 1, 256);
    e->commit_and_wait();
    int bad = 0, outside = 0;
    const uint16_t* o = dst.as<uint16_t>();
    for (uint32_t r = 0; r < row0 + R + 1; ++r)
      for (uint32_t c = 0; c < ld; ++c) {
        const bool in = r >= row0 && r < row0 + R && c >= slot * D && c < (slot + 1) * D;
        const uint16_t got = o[size_t(r) * ld + c];
        if (!in) {
          outside += got != 0xFFFF;
          continue;
        }
        const uint32_t rr = r - row0, cc = c - slot * D;
        double sum = 0;
        for (uint32_t s = 0; s < S; ++s)
          sum += double(X.as<float>()[(rr * S + s) * D + cc]) +
                 (has_y ? double(inj.as<float>()[rr * S + s]) * double(y.as<float>()[rr * D + cc]) : 0.0);
        const float want = float(sum / S), have = bf16_to_f32(got);
        // bf16 keeps 8 bits: within one unit in the last place of the value (fp32 sums vs the double reference)
        const float ulp = std::ldexp(1.0f, std::ilogb(std::fabs(want) + 1e-30f) - 7);
        bad += !(std::fabs(have - want) <= ulp);
      }
    check(bad == 0 && outside == 0, std::string(has_y ? "with" : "without") + " the pending block output: " +
                                        std::to_string(R * D - uint32_t(bad)) + " of " + std::to_string(R * D) +
                                        " values within a bf16 unit, nothing written outside its slot");
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool gpu = false;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--gpu")) gpu = true;
    else {
      std::fprintf(stderr, "usage: fn_dflash_test [--gpu]\n");
      return 2;
    }
  }
  run_cpu();
  if (gpu) run_gpu();
  std::printf("%s\n", g_fail ? "FAILED" : "all checks passed");
  return g_fail ? 1 : 0;
}
