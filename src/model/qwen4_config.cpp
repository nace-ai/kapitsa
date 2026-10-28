#include "model/qwen4_config.h"

#include <algorithm>
#include <cstdio>
#include <set>

namespace ni {

using json = nlohmann::json;

namespace {

const json& text_of(const json& c) {
  return c.contains("text_config") && c["text_config"].is_object() ? c["text_config"] : c;
}

std::string type_of(const json& c) {
  return c.is_object() && c.contains("model_type") && c["model_type"].is_string() ? c["model_type"].get<std::string>()
                                                                                  : std::string();
}

// a number setting, `def` when it is absent or null
long long num(const json& t, const char* k, long long def) {
  return t.contains(k) && t[k].is_number() ? t[k].get<long long>() : def;
}

std::string str(const json& t, const char* k) {
  return t.contains(k) && t[k].is_string() ? t[k].get<std::string>() : std::string();
}

bool present(const json& t, const char* k) { return t.contains(k) && !t[k].is_null(); }

std::string fmt(const char* f, long long a) {
  char b[160];
  snprintf(b, sizeof b, f, a);
  return b;
}

}  // namespace

bool qwen4_family(const json& c) {
  if (!c.is_object()) return false;
  return type_of(c).rfind("qwen4", 0) == 0 || type_of(text_of(c)).rfind("qwen4", 0) == 0;
}

std::vector<std::string> qwen4_unsupported(const json& c) {
  std::vector<std::string> v;
  if (!c.is_object()) return {"config.json is not a JSON object"};
  const json& t = text_of(c);
  const std::string mt = type_of(c), tt = type_of(t);
  // Qwen4Exp::is_model's: model_type qwen4_exp, or a text_config of qwen4_exp_text
  const bool nested = &t != &c;
  if (mt == "qwen4_exp_text" && !nested)
    v.push_back("model_type qwen4_exp_text without a text_config: Kapitsa loads Qwen3.8-Flash-Next's config.json "
                "(model_type qwen4_exp, or its text settings in a text_config)");
  else if (mt != "qwen4_exp" && !(nested && tt == "qwen4_exp_text"))
    v.push_back("model_type " + (mt.empty() ? tt : mt) +
                ": Kapitsa runs Qwen 4's preview architecture, qwen4_exp (Qwen3.8-Flash-Next), and a new one once its "
                "differences are read");
  // what model/qwen4_exp.cpp's load() and its kernels (qwen4.metal) take
  const long long hidden = num(t, "hidden_size", 0);
  if (hidden <= 0 || hidden % 128 || hidden > 4096)
    v.push_back(fmt("hidden_size %lld: 128 to 4096 in steps of 128 (the gated residual's kernels)", hidden));
  if (const long long hc = num(t, "hc_count", 4); hc != 4)
    v.push_back(fmt("hc_count %lld: 4 residual streams", hc));
  const long long hq = num(t, "num_attention_heads", 0);
  if (const long long hd = num(t, "head_dim", hq > 0 ? hidden / hq : 0); hd != 256)  // (Qwen35's default when absent)
    v.push_back(fmt("head_dim %lld: the sparse attention's kernels take 256", hd));
  if (num(t, "num_experts", 0) <= 0) v.push_back("no num_experts: dense MLP layers (the qwen4_exp path is MoE only)");
  if (const std::string act = str(t, "hidden_act"); !act.empty() && act != "silu")
    v.push_back("hidden_act " + act + ": SiLU");
  if (const std::string g = str(t, "output_gate_type"); !g.empty() && g != "sigmoid" && g != "silu")
    v.push_back("output_gate_type " + g + ": sigmoid or SiLU");
  if (const long long k = num(t, "linear_conv_kernel_dim", 4); k != 4)
    v.push_back(fmt("linear_conv_kernel_dim %lld: 4", k));
  if (t.contains("layer_types") && t["layer_types"].is_array()) {
    const json& lt = t["layer_types"];
    if (num(t, "num_hidden_layers", 0) != (long long)lt.size())
      v.push_back(fmt("layer_types names %lld layers, not num_hidden_layers", (long long)lt.size()));
    std::set<std::string> other;
    for (const json& x : lt)
      if (!x.is_string() || (x != "linear_attention" && x != "full_attention" && x != "indexed_attention"))
        other.insert(x.is_string() ? x.get<std::string>() : x.dump());
    for (const std::string& o : other) v.push_back("layer_types has " + o + " layers");
  }
  // RoPE: interleaved M-RoPE for image rows (scaling: qwen4_unapplied)
  const json rp = t.contains("rope_parameters") && t["rope_parameters"].is_object() ? t["rope_parameters"] : json::object();
  if (rp.contains("mrope_interleaved") && rp["mrope_interleaved"].is_boolean() && !rp["mrope_interleaved"].get<bool>())
    v.push_back("mrope_interleaved false: interleaved M-RoPE");
  // Qwen Sparse Attention: Transformers runs the full-attention layers dense without the indexer settings; this path
  // always selects. (indexer_kv_heads, which Transformers requires to be 1, is not read: one key head.)
  static const char* kIx[] = {"indexer_n_heads", "indexer_head_dim", "indexer_budget", "indexer_compress_ratio"};
  std::string missing;
  int have = 0;
  for (const char* k : kIx) {
    if (present(t, k)) ++have;
    else missing += std::string(missing.empty() ? "" : ", ") + k;
  }
  if (!have) {
    v.push_back("no indexer_* settings: full attention without Qwen Sparse Attention's indexer");
  } else if (!missing.empty()) {
    v.push_back("indexer settings without " + missing);
  } else {
    if (const long long d = num(t, "indexer_head_dim", 0); d != 128) v.push_back(fmt("indexer_head_dim %lld: 128", d));
    if (const long long h = num(t, "indexer_n_heads", 0); h < 1 || h > 8)
      v.push_back(fmt("indexer_n_heads %lld: 1 to 8", h));
    if (const long long r = num(t, "indexer_compress_ratio", 0); r != 4)
      v.push_back(fmt("indexer_compress_ratio %lld: 4", r));
    if (const long long kv = num(t, "indexer_kv_heads", 1); kv != 1) v.push_back(fmt("indexer_kv_heads %lld: 1", kv));
    if (const long long b = num(t, "indexer_budget", 0); b <= 0 || b % 4)
      v.push_back(fmt("indexer_budget %lld: whole blocks of 4", b));
  }
  // the n-gram embedding (PLE)
  if (t.contains("ple_layer_ids") && t["ple_layer_ids"].is_array() && !t["ple_layer_ids"].empty()) {
    if (t["ple_layer_ids"].size() > 1) v.push_back("ple_layer_ids " + t["ple_layer_ids"].dump() + ": one n-gram layer");
    if (const long long k = num(t, "ple_conv_kernel_size", 4); k != 4)
      v.push_back(fmt("ple_conv_kernel_size %lld: 4", k));
    if (const long long d = num(t, "ple_embed_dim", hidden); d != hidden)
      v.push_back(fmt("ple_embed_dim %lld: the hidden size", d));
    if (const std::string m = str(t, "ple_embedding_mode"); !m.empty() && m != "ngram")
      v.push_back("ple_embedding_mode " + m + ": the hashed n-gram table");
  }
  return v;
}

std::vector<std::string> qwen4_unapplied(const json& c) {
  // Flash-Next's text_config (Qwen/Qwen3.8-Flash-Next, its FP8 copy, Nace AI's and Sushi's packs), Transformers'
  // Qwen4ExpTextConfig and PreTrainedConfig, what converters add (ple_storage, quantization), and the top level's of a
  // config without a text_config
  static const std::set<std::string> kKnown = {
      "_name_or_path", "architectures", "attention_bias", "attention_dropout", "bos_token_id",
      "chunk_size_feed_forward", "dtype", "eos_token_id", "full_attention_interval", "hc_count", "hc_lowrank",
      "head_dim", "heads_per_ngram", "hidden_act", "hidden_size", "id2label", "image_token_id", "indexer_budget",
      "indexer_compress_ratio", "indexer_head_dim", "indexer_kv_heads", "indexer_n_heads", "initializer_range",
      "intermediate_size", "is_encoder_decoder", "label2id", "language_model_only", "layer_types",
      "linear_conv_kernel_dim",
      "linear_key_head_dim", "linear_num_key_heads", "linear_num_value_heads", "linear_value_head_dim",
      "make_ngram_vocab_size_divisible_by", "mamba_ssm_dtype", "max_position_embeddings", "model_type",
      "moe_intermediate_size", "mtp", "mtp_num_hidden_layers", "mtp_use_dedicated_embeddings", "ngram_size",
      "ngram_vocab_size_base", "norm_topk_prob", "num_attention_heads", "num_experts", "num_experts_per_tok",
      "num_hidden_layers", "num_key_value_heads", "output_attentions", "output_gate_type", "output_hidden_states",
      "output_router_logits", "pad_token_id", "partial_rotary_factor", "ple_conv_kernel_size", "ple_embed_dim",
      "ple_embedding_mode", "ple_layer_ids", "ple_storage", "problem_type", "quantization", "quantization_config",
      "return_dict", "rms_norm_eps", "rope_parameters", "rope_scaling", "rope_theta", "router_aux_loss_coef", "seed",
      "shared_expert_intermediate_size", "split_ngram_parts", "tie_word_embeddings", "tokenizer_class", "torch_dtype",
      "transformers_version", "use_cache", "video_token_id", "vision_config", "vision_end_token_id",
      "vision_start_token_id", "vocab_size"};
  std::vector<std::string> v;
  const json& t = text_of(c);
  if (!t.is_object()) return v;
  const json rp = t.contains("rope_parameters") && t["rope_parameters"].is_object() ? t["rope_parameters"] : json::object();
  if (const std::string rt = str(rp, "rope_type"); !rt.empty() && rt != "default")
    v.push_back("rope_type " + rt + " (RoPE runs unscaled)");
  if (present(t, "rope_scaling")) v.push_back("rope_scaling (RoPE runs unscaled)");
  std::vector<std::string> unknown;
  for (auto it = t.begin(); it != t.end(); ++it)
    if (!kKnown.count(it.key())) unknown.push_back(it.key());
  std::sort(unknown.begin(), unknown.end());
  if (!unknown.empty()) {
    std::string names;
    for (const std::string& n : unknown) names += (names.empty() ? "" : ", ") + n;
    v.push_back("settings Qwen3.8-Flash-Next does not have: " + names);
  }
  return v;
}

std::string qwen4_refusal(const json& c, const std::string& what) {
  const std::vector<std::string> v = qwen4_unsupported(c);
  if (v.empty()) return "";
  const json& t = text_of(c);
  std::string type = type_of(c);
  if (type.empty()) type = type_of(t);
  std::string m = what + " is a " + type +
                  (type == "qwen4_exp" || type == "qwen4_exp_text"
                       ? " model with settings Kapitsa's Qwen3.8-Flash-Next path does not run: "
                       : " model (Qwen 4), which Kapitsa does not run yet: ");
  for (size_t i = 0; i < v.size(); ++i) m += (i ? "; " : "") + v[i];
  m += ".";
  const std::vector<std::string> u = qwen4_unapplied(c);
  if (!u.empty()) {
    m += " Not applied either: ";
    for (size_t i = 0; i < u.size(); ++i) m += (i ? "; " : "") + u[i];
    m += ".";
  }
  return m;
}

}  // namespace ni
