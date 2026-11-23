#include "model/ink2_draft.h"
#include "model/ink2_draft_plan.h"
#include "util/json_file.h"

#include <cmath>
#include <filesystem>

namespace ni::ink2draft {
namespace {
void check(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(std::string("ink2 draft: ") + what);
}
}
DraftPlan read_plan(const std::string& path) {
  auto root = json_file(path + "/config.json");
  check(root && root->contains("mtp_config"), "missing mtp_config");
  const auto& c = root->at("mtp_config");
  check(c.at("hidden_size") == H && c.at("num_hidden_layers") == 3 && c.at("num_attention_heads") == HQ &&
      c.at("num_key_value_heads") == HKV && c.at("head_dim") == D && c.at("intermediate_size") == I &&
      c.at("markov_rank") == R && c.at("block_size") == WIDTH && c.at("markov_head_type") == "rnn_hidden" &&
      c.at("q_bias") == false && c.at("o_bias") == false &&
      c.at("local_layer_ids") == nlohmann::json::array({0, 2}) && c.at("sliding_window_size") == 1024 &&
      c.at("target_layer_ids") == nlohmann::json::array({5, 23, 41, 48}), "unsupported mtp_config");
  const float eps = c.at("rms_norm_eps").get<float>(), theta = c.at("rope_theta").get<float>();
  check(std::isfinite(eps) && eps > 0 && std::isfinite(theta) && theta > 0, "invalid norm/rope parameters");
  const auto& tc = root->contains("text_config") ? root->at("text_config") : *root;
  auto positive = [&](const char* key) {
    const auto& v = tc.at(key);
    check(v.is_number_integer() && v.get<int64_t>() > 0 && v.get<uint64_t>() <= UINT32_MAX, key);
    return v.get<uint32_t>();
  };
  const uint32_t vocab = positive("unpadded_vocab_size"), padded = positive("vocab_size");
  check(tc.at("hidden_size") == H && vocab <= padded && padded <= uint32_t(INT32_MAX), "target vocabulary/hidden size");
  check(vocab <= 4096u * topk_chunk(1), "vocabulary exceeds greedy sampler capacity");
  auto weights = WeightStore::plan_file(path + "/mtp.safetensors");
  auto require = [&](const std::string& name, std::vector<int64_t> shape) {
    const auto it = weights.tensors.find(name);
    check(it != weights.tensors.end() && it->second.dtype == DType::BF16 && it->second.shape == shape, name.c_str());
  };
  require("markov_head.hidden_out.weight", {H, R});
  require("markov_head.hidden_out.bias", {H});
  require("fuse_weight", {4, 4});
  require("norm.weight", {H});
  require("mask_embed", {H});
  require("markov_head.markov_w1.weight", {padded, R});
  require("markov_head.dflare_norm.weight", {H});
  require("markov_head.rnn_proj.bias", {3 * R});
  require("markov_head.rnn_proj.weight", {3 * R, 2 * R + 2 * H});
  for (int i = 0; i < 3; ++i) {
    const std::string s = "layers." + std::to_string(i) + ".";
    require(s + "dflare_norm.weight", {H});
    require(s + "self_attn.k_proj_ctx.weight", {HKV * D, H});
    require(s + "self_attn.v_proj_ctx.weight", {HKV * D, H});
    require(s + "self_attn.k_norm_ctx.weight", {D});
    require(s + "self_attn.q_proj.weight", {HQ * D, H});
    require(s + "self_attn.k_proj.weight", {HKV * D, H});
    require(s + "self_attn.v_proj.weight", {HKV * D, H});
    require(s + "self_attn.q_norm.weight", {D});
    require(s + "self_attn.k_norm.weight", {D});
    require(s + "self_attn.o_proj.weight", {H, HQ * D});
    require(s + "input_layernorm.weight", {H});
    require(s + "post_attention_layernorm.weight", {H});
    require(s + "mlp.gate_proj.weight", {I, H});
    require(s + "mlp.up_proj.weight", {I, H});
    require(s + "mlp.down_proj.weight", {H, I});
  }
  SharedLayout layout(vocab);
  (void)add(weights.bytes, layout.total());
  return {vocab, padded, eps, theta, std::move(weights), layout};
}
}  // namespace ni::ink2draft

namespace ni {
Ink2DraftMemory Ink2Draft::memory_plan(const std::string& path) {
  if (std::filesystem::symlink_status(path + "/mtp.safetensors").type() == std::filesystem::file_type::not_found)
    return {0, 0};
  const auto p = ink2draft::read_plan(path);
  return {ink2draft::add(p.weights.bytes, p.layout.total()), ink2draft::empty_seq_bytes()};
}
}  // namespace ni
