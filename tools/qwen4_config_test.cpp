// What a Qwen 4-family config.json asks that the Qwen3.8-Flash-Next path does not run (src/model/qwen4_config.h), on
// Flash-Next's published config: it runs as it is and with Sushi's YaRN (named, not applied), and each setting the
// path does not run is refused alone. CPU only, no model.
#include "model/qwen4_config.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using json = nlohmann::json;

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

// Qwen/Qwen3.8-Flash-Next's config.json (2026-08-27), its vision tower aside
json flash_next() {
  json t = json::parse(R"({"attention_bias": false, "attention_dropout": 0.0, "bos_token_id": 248044,
    "dtype": "bfloat16", "eos_token_id": 248044, "full_attention_interval": 4, "hc_count": 4, "hc_lowrank": 320,
    "head_dim": 256, "heads_per_ngram": 8, "hidden_act": "silu", "hidden_size": 2560, "indexer_budget": 2048,
    "indexer_compress_ratio": 4, "indexer_head_dim": 128, "indexer_kv_heads": 1, "indexer_n_heads": 4,
    "initializer_range": 0.02, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
    "linear_num_value_heads": 48, "linear_value_head_dim": 128, "make_ngram_vocab_size_divisible_by": 128,
    "mamba_ssm_dtype": "float32", "max_position_embeddings": 262144, "model_type": "qwen4_exp_text",
    "moe_intermediate_size": 640, "mtp": {"hybrid": true, "layer_types": ["full_attention"],
    "mtp_use_hidden_state_from_layer": null, "num_hidden_layers": 1, "rope_theta": 10000000},
    "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false, "ngram_size": 3,
    "ngram_vocab_size_base": 20000000, "num_attention_heads": 24, "num_experts": 512, "num_experts_per_tok": 10,
    "num_hidden_layers": 48, "num_key_value_heads": 2, "output_gate_type": "sigmoid", "output_router_logits": false,
    "pad_token_id": null, "partial_rotary_factor": 0.25, "ple_conv_kernel_size": 4, "ple_embed_dim": 2560,
    "ple_layer_ids": [2], "rms_norm_eps": 1e-06, "rope_parameters": {"mrope_interleaved": true,
    "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
    "router_aux_loss_coef": 0.001, "shared_expert_intermediate_size": 640, "split_ngram_parts": 128,
    "tie_word_embeddings": false, "use_cache": true, "vocab_size": 248320})");
  t["layer_types"] = json::array();
  for (int i = 0; i < 48; ++i) t["layer_types"].push_back(i % 4 == 3 ? "full_attention" : "linear_attention");
  return {{"architectures", {"Qwen4ExpForConditionalGeneration"}}, {"model_type", "qwen4_exp"},
          {"image_token_id", 248056}, {"video_token_id", 248057}, {"vision_start_token_id", 248053},
          {"vision_end_token_id", 248054}, {"language_model_only", false}, {"tie_word_embeddings", false},
          {"transformers_version", "5.8.0.dev0"}, {"text_config", t}};
}

}  // namespace

int main() {
  const json fn = flash_next();
  CHECK(ni::qwen4_family(fn), "Flash-Next is Qwen 4-family");
  CHECK(ni::qwen4_refusal(fn, "x").empty(), "Flash-Next runs: %s", ni::qwen4_refusal(fn, "x").c_str());
  CHECK(ni::qwen4_unapplied(fn).empty(), "Flash-Next: nothing unapplied (%zu)", ni::qwen4_unapplied(fn).size());

  // Sushi's 2bpw pack (beamster/Qwen3.8-Flash-Next-Sushi-2bpw@cb6c85e, the catalog's 64gb variant): YaRN for 1M,
  // which runs unscaled as before; refusing it would fail a pack that loads today
  json sushi = fn;
  sushi["text_config"]["rope_parameters"].update(
      {{"rope_type", "yarn"}, {"factor", 4.0}, {"original_max_position_embeddings", 262144}});
  sushi["text_config"]["max_position_embeddings"] = 1048576;
  const std::vector<std::string> su = ni::qwen4_unapplied(sushi);
  CHECK(ni::qwen4_refusal(sushi, "x").empty(), "Sushi's YaRN runs: %s", ni::qwen4_refusal(sushi, "x").c_str());
  CHECK(su.size() == 1 && su[0] == "rope_type yarn (RoPE runs unscaled)", "Sushi: one unapplied line (%zu)", su.size());

  // as Qwen4Exp::is_model takes them: a config without a text_config runs as model_type qwen4_exp (with Nace AI's
  // pack's ple_storage), and is refused as qwen4_exp_text, which the loader does not take at the top level
  json flat = fn["text_config"];
  flat["model_type"] = "qwen4_exp";
  flat["ple_storage"] = json::object();
  CHECK(ni::qwen4_refusal(flat, "x").empty(), "a flat qwen4_exp config runs: %s", ni::qwen4_refusal(flat, "x").c_str());
  flat["model_type"] = "qwen4_exp_text";
  const std::vector<std::string> fv = ni::qwen4_unsupported(flat);
  CHECK(fv.size() == 1 && fv[0].rfind("model_type qwen4_exp_text without a text_config", 0) == 0,
        "a flat qwen4_exp_text config is refused for its type alone (%zu)", fv.size());

  // not Qwen 4
  CHECK(!ni::qwen4_family(json{{"model_type", "qwen3_5_moe"}, {"text_config", {{"model_type", "qwen3_5_moe_text"}}}}),
        "Qwen3.5 is not Qwen 4-family");

  // a new Qwen 4 model_type with Flash-Next's settings: refused for its type alone, its new settings named
  json moe = fn;
  moe["model_type"] = "qwen4_moe";
  moe["text_config"]["model_type"] = "qwen4_moe_text";
  moe["text_config"]["gr_dtype"] = "fp8";
  const std::string mr = ni::qwen4_refusal(moe, "x");
  CHECK(ni::qwen4_unsupported(moe).size() == 1 && mr.find("x is a qwen4_moe model (Qwen 4)") == 0 &&
            mr.find("does not have: gr_dtype") != std::string::npos,
        "qwen4_moe refused for its type, gr_dtype named: %s", mr.c_str());

  // each setting the path does not run, alone on Flash-Next's config: exactly one refusal line
  const std::vector<std::pair<const char*, std::function<void(json&)>>> one = {
      {"hidden_size 5120", [](json& t) { t["hidden_size"] = 5120; t["ple_embed_dim"] = 5120; }},
      {"hc_count 2", [](json& t) { t["hc_count"] = 2; }},
      {"head_dim 128", [](json& t) { t["head_dim"] = 128; }},
      {"no num_experts", [](json& t) { t.erase("num_experts"); }},
      {"hidden_act gelu", [](json& t) { t["hidden_act"] = "gelu"; }},
      {"output_gate_type tanh", [](json& t) { t["output_gate_type"] = "tanh"; }},
      {"linear_conv_kernel_dim 3", [](json& t) { t["linear_conv_kernel_dim"] = 3; }},
      {"layer_types has sliding_attention", [](json& t) { t["layer_types"][3] = "sliding_attention"; }},
      {"mrope_interleaved false", [](json& t) { t["rope_parameters"]["mrope_interleaved"] = false; }},
      {"no indexer_* settings", [](json& t) {
         for (const char* k : {"indexer_n_heads", "indexer_kv_heads", "indexer_head_dim", "indexer_budget",
                               "indexer_compress_ratio"})
           t.erase(k);
       }},
      {"indexer settings without indexer_budget", [](json& t) { t.erase("indexer_budget"); }},
      {"indexer_head_dim 64", [](json& t) { t["indexer_head_dim"] = 64; }},
      {"indexer_n_heads 16", [](json& t) { t["indexer_n_heads"] = 16; }},
      {"indexer_compress_ratio 8", [](json& t) { t["indexer_compress_ratio"] = 8; }},
      {"indexer_kv_heads 2", [](json& t) { t["indexer_kv_heads"] = 2; }},
      {"ple_layer_ids [2,30]", [](json& t) { t["ple_layer_ids"] = {2, 30}; }},
      {"ple_conv_kernel_size 8", [](json& t) { t["ple_conv_kernel_size"] = 8; }},
      {"ple_embed_dim 1280", [](json& t) { t["ple_embed_dim"] = 1280; }},
      {"ple_embedding_mode per_token", [](json& t) { t["ple_embedding_mode"] = "per_token"; }},
  };
  for (const auto& [what, edit] : one) {
    json c = fn;
    edit(c["text_config"]);
    const std::vector<std::string> v = ni::qwen4_unsupported(c);
    CHECK(v.size() == 1 && v[0].rfind(what, 0) == 0, "%s: one refusal line, got %zu (%s)", what, v.size(),
          v.empty() ? "" : v[0].c_str());
  }

  printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
