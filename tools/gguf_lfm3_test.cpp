// The LFM3 GGUF import (src/gguf/import.cpp: read_hp_lfm3, check_runs_lfm3, build_lfm3, make_config_lfm3 and
// make_tokenizer's lfm2 case) on a synthetic GGUF built here: the real schedule's prefix (conv, conv, full
// attention, conv, sliding attention), small dimensions, the real gate values (head_dim 128, conv_L_cache 7), and
// llama.cpp's baked rope_freqs.weight for the partial rotation. Import runs through the public import_model, in
// plan mode (config.json and the safetensors headers, no tensor data) and once fully into a KAPITSA_HOME sandbox.
// CPU only, no model, no GPU.
#include "gguf/import.h"
#include "tokenizer/tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using json = nlohmann::json;
using ni::gguf::ImportOptions;
using ni::gguf::import_model;

namespace {

int g_checks = 0, g_fail = 0;

void ok(const std::string& name, bool cond, const std::string& detail = "") {
  ++g_checks;
  if (cond) {
    printf("ok %s\n", name.c_str());
  } else {
    ++g_fail;
    fprintf(stderr, "FAIL %s%s%s\n", name.c_str(), detail.empty() ? "" : ": ", detail.c_str());
  }
}

std::string read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

json read_json(const std::string& p) { return json::parse(read_file(p), nullptr, false); }

// GPT-2's byte-to-unicode, as tokenizer.json's vocabulary spells its bytes
std::string byte_unicode(int b) {
  static const auto map = [] {
    std::vector<uint32_t> m(256);
    for (int x = 0, n = 0; x < 256; ++x) {
      const bool keep = (x >= '!' && x <= '~') || (x >= 0xA1 && x <= 0xAC) || (x >= 0xAE && x <= 0xFF);
      m[x] = keep ? uint32_t(x) : uint32_t(256 + n++);
    }
    return m;
  }();
  const uint32_t c = map[b];
  std::string s;
  if (c < 0x80) s += char(c);
  else s += {char(0xC0 | (c >> 6)), char(0x80 | (c & 0x3F))};
  return s;
}

void put_str(std::string& out, const std::string& s) {
  uint64_t n = s.size();
  out.append(reinterpret_cast<const char*>(&n), 8);
  out += s;
}

void put_kv(std::string& out, const std::string& k, uint32_t t, const std::string& v) {
  put_str(out, k);
  out.append(reinterpret_cast<const char*>(&t), 4);
  out += v;
}

void put_u32(std::string& v, uint32_t x) { v.append(reinterpret_cast<const char*>(&x), 4); }
void put_i64(std::string& v, int64_t x) { v.append(reinterpret_cast<const char*>(&x), 8); }
void put_f32(std::string& v, float x) { v.append(reinterpret_cast<const char*>(&x), 4); }

// A GGUF v3 file: the KVs as one built blob, then the tensors (F32 values or Q4_1 blocks of zeros)
struct TensorDef {
  std::string name;
  std::vector<uint64_t> ne;  // innermost first
  uint32_t type;             // 0 F32, 3 Q4_1
  std::string f32;           // F32: little-endian floats
};

std::string write_gguf(const std::string& path, std::string kvs, int nkv, const std::vector<TensorDef>& tensors) {
  std::string infos, data;
  uint64_t off = 0;
  for (const auto& t : tensors) {
    put_str(infos, t.name);
    put_u32(infos, uint32_t(t.ne.size()));
    for (uint64_t d : t.ne) put_i64(infos, int64_t(d));
    put_u32(infos, t.type);
    put_i64(infos, int64_t(off));
    if (t.type == 0) data += t.f32;
    else data.append(size_t(t.ne[0]) * t.ne[1] / 32 * 20, '\0');  // Q4_1: zero scale, zero min, zero codes
    off = uint64_t(data.size());
  }

  std::string out = "GGUF";
  const uint32_t ver = 3;
  out.append(reinterpret_cast<const char*>(&ver), 4);
  const uint64_t nt = tensors.size(), nk = uint64_t(nkv);
  out.append(reinterpret_cast<const char*>(&nt), 8);
  out.append(reinterpret_cast<const char*>(&nk), 8);
  out += kvs;
  out += infos;
  while (out.size() % 32) out += '\0';
  out += data;
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f << out;
  return path;
}

struct Spec {
  int64_t H = 128, layers = 5, hq = 2, hkv = 1, head_dim = 128, ffn = 256, vocab = 512, ctx = 8192;
  int64_t window = 256, l_cache = 7;
  double theta = 1e6, theta_swa = 1e4, eps = 1e-5;
  bool rope_freqs = true;      // the baked zero-tail table: 32 nonzeros of 64
  int64_t rope_nz = 32;
  std::vector<int64_t> attn = {0, 0, 1, 0, 1}, swa = {0, 0, 0, 0, 1};
};

std::string make_gguf(const std::string& dir, const std::string& name, const Spec& s,
                      const std::vector<std::pair<std::string, int64_t>>& overrides = {}) {
  std::string kvs;
  int nkv = 0;
  auto kv_str = [&](const std::string& k, const std::string& v) {
    std::string b;
    put_str(b, v);
    put_kv(kvs, k, 8, b);
    ++nkv;
  };
  auto kv_i64 = [&](const std::string& k, int64_t x) {
    std::string b;
    put_i64(b, x);
    put_kv(kvs, k, 10, b);
    ++nkv;
  };
  auto kv_f32 = [&](const std::string& k, float x) {
    std::string b;
    put_f32(b, x);
    put_kv(kvs, k, 6, b);
    ++nkv;
  };
  auto kv_i64s = [&](const std::string& k, const std::vector<int64_t>& xs) {
    std::string b;
    put_u32(b, 10);
    put_i64(b, int64_t(xs.size()));
    for (int64_t x : xs) put_i64(b, x);
    put_kv(kvs, k, 9, b);
    ++nkv;
  };
  auto kv_strs = [&](const std::string& k, const std::vector<std::string>& xs) {
    std::string b;
    put_u32(b, 8);
    put_i64(b, int64_t(xs.size()));
    for (const std::string& x : xs) put_str(b, x);
    put_kv(kvs, k, 9, b);
    ++nkv;
  };

  kv_str("general.architecture", "lfm3");
  kv_str("general.name", "LFM3 " + name);
  kv_i64("lfm3.embedding_length", s.H);
  kv_i64("lfm3.block_count", s.layers);
  kv_i64("lfm3.attention.head_count", s.hq);
  kv_i64("lfm3.attention.key_length", s.head_dim);
  kv_i64("lfm3.attention.value_length", s.head_dim);
  kv_i64("lfm3.rope.dimension_count", s.head_dim);
  kv_i64("lfm3.feed_forward_length", s.ffn);
  kv_i64("lfm3.attention.sliding_window", s.window);
  kv_i64("lfm3.shortconv.l_cache", s.l_cache);
  kv_i64("lfm3.context_length", s.ctx);
  kv_f32("lfm3.attention.layer_norm_rms_epsilon", float(s.eps));
  kv_f32("lfm3.rope.freq_base", float(s.theta));
  kv_f32("lfm3.rope.freq_base_swa", float(s.theta_swa));
  kv_i64s("lfm3.attention.head_count_kv", s.attn);
  kv_i64s("lfm3.attention.sliding_window_pattern", s.swa);
  kv_i64("tokenizer.ggml.bos_token_id", s.vocab - 6);
  kv_i64("tokenizer.ggml.eos_token_id", s.vocab - 5);
  kv_i64("tokenizer.ggml.padding_token_id", s.vocab - 4);
  kv_i64("tokenizer.ggml.add_bos_token", 0);
  kv_str("tokenizer.ggml.model", "gpt2");
  kv_str("tokenizer.ggml.pre", "lfm2");
  kv_str("tokenizer.chat_template", "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}");
  for (const auto& [k, x] : overrides) kv_i64(k, x);

  // the tokenizer: 256 byte tokens, filler, six special tokens — exactly the embedding's vocabulary
  std::vector<std::string> toks;
  for (int b = 0; b < 256; ++b) toks.push_back(byte_unicode(b));
  for (int i = 256; i + 6 < int(s.vocab); ++i) toks.push_back("t" + std::to_string(i));
  for (const char* a : {"<|im_start|>", "<|im_end|>", "<|pad|>", "<think>", "</think>", "<tool_call>"})
    toks.push_back(a);
  std::vector<int64_t> types(size_t(s.vocab), 1);
  for (int64_t i = s.vocab - 6; i < s.vocab; ++i) types[size_t(i)] = 3;
  kv_strs("tokenizer.ggml.tokens", toks);
  kv_i64s("tokenizer.ggml.token_type", types);
  std::vector<std::string> mg;
  for (size_t i = 0; i + 1 < toks.size(); ++i) mg.push_back(toks[i] + " " + toks[i + 1]);
  kv_strs("tokenizer.ggml.merges", mg);

  std::vector<TensorDef> ts;
  auto f32s = [&](const std::string& n, size_t count, float x) {
    std::string d;
    for (size_t i = 0; i < count; ++i) put_f32(d, x);
    ts.push_back({n, {count}, 0, d});
  };
  auto q41 = [&](const std::string& n, uint64_t K, uint64_t N) { ts.push_back({n, {K, N}, 3, ""}); };

  f32s("output_norm.weight", size_t(s.H), 1.0f);
  q41("token_embd.weight", s.H, uint64_t(s.vocab));
  for (int64_t i = 0; i < s.layers; ++i) {
    const std::string b = "blk." + std::to_string(i) + ".";
    for (const char* n : {"attn_norm.weight", "post_attention_norm.weight", "ffn_norm.weight", "post_ffw_norm.weight"})
      f32s(b + n, size_t(s.H), 1.0f);
    q41(b + "ffn_gate.weight", s.H, uint64_t(s.ffn));
    q41(b + "ffn_up.weight", s.H, uint64_t(s.ffn));
    q41(b + "ffn_down.weight", uint64_t(s.ffn), s.H);
    if (s.attn[size_t(i)]) {
      const uint64_t QD = uint64_t(s.hq * s.head_dim), KD = uint64_t(s.hkv * s.head_dim);
      q41(b + "attn_q.weight", s.H, QD);
      q41(b + "attn_k.weight", s.H, KD);
      q41(b + "attn_v.weight", s.H, KD);
      q41(b + "attn_output.weight", QD, s.H);
      q41(b + "attn_gate.weight", s.H, uint64_t(s.hq));
      f32s(b + "attn_q_norm.weight", size_t(s.head_dim), 1.0f);
      f32s(b + "attn_k_norm.weight", size_t(s.head_dim), 1.0f);
    } else {
      std::string d;
      for (int64_t k = 0; k < s.l_cache; ++k) put_f32(d, 0.25f * float(k + 1));
      for (size_t c = 1; c < size_t(s.H); ++c) d += d.substr(0, size_t(4 * s.l_cache));
      ts.push_back({b + "shortconv.conv.weight", {uint64_t(s.l_cache), uint64_t(s.H)}, 0, d});
      q41(b + "shortconv.in_proj.weight", s.H, uint64_t(3 * s.H));
      q41(b + "shortconv.out_proj.weight", s.H, s.H);
    }
  }
  if (s.rope_freqs) {
    std::string d;
    for (int64_t i = 0; i < s.head_dim / 2; ++i) put_f32(d, i < s.rope_nz ? 1.0f : 0.0f);
    ts.push_back({"rope_freqs.weight", {uint64_t(s.head_dim / 2)}, 0, d});
  }

  return write_gguf(dir + "/" + name + ".gguf", std::move(kvs), nkv, ts);
}

std::string sandbox() {
  const char* tmp = getenv("TMPDIR");
  std::string base = std::string(tmp && *tmp ? tmp : "/tmp") + "/kapitsa_gguf_lfm3_test";
  std::filesystem::remove_all(base);
  std::filesystem::create_directories(base + "/home");
  setenv("KAPITSA_HOME", (base + "/home").c_str(), 1);
  return base;
}

std::string first_error(const std::string& file, const ImportOptions& opt) {
  try {
    import_model(opt);
    return "";
  } catch (const std::exception& e) {
    return e.what();
  }
}

}  // namespace

int main() {
  const std::string base = sandbox();
  Spec spec;

  // plan mode: config.json and the safetensors headers, no tensor data
  const std::string gguf = make_gguf(base, "lfm3-test", spec);
  const ImportOptions plan{.file = gguf, .name = "lfm3-test", .plan_dir = base + "/plan", .progress = false,
                           .drafter = false};
  std::string err = first_error(gguf, plan);
  ok("plan-mode import of a synthetic lfm3 GGUF runs", err.empty(), err);
  if (err.empty()) {
    const json cfg = read_json(base + "/plan/config.json");
    ok("config: model_type lfm3", cfg.value("model_type", "") == "lfm3");
    const json lt = cfg.value("layer_types", json::array());
    ok("config: the schedule from the per-layer arrays",
       lt == json::array({"conv", "conv", "full_attention", "conv", "sliding_attention"}), lt.dump());
    ok("config: rope_parameters per attention type",
       cfg["rope_parameters"]["full_attention"].value("partial_rotary_factor", 0.0) == 0.5 &&
           cfg["rope_parameters"]["full_attention"].value("rope_theta", 0.0) == 1e6 &&
           cfg["rope_parameters"]["sliding_attention"].value("rope_theta", 0.0) == 1e4,
       cfg["rope_parameters"].dump());
    ok("config: conv_L_cache 7, bias false, sandwich norms, tied embeddings",
       cfg.value("conv_L_cache", 0) == 7 && cfg.value("conv_bias", true) == false &&
           cfg.value("use_sandwich_norm", false) == true && cfg.value("tie_word_embeddings", false) == true);
    ok("config: the shape", cfg.value("hidden_size", 0) == 128 && cfg.value("head_dim", 0) == 128 &&
                               cfg.value("num_attention_heads", 0) == 2 && cfg.value("num_key_value_heads", 0) == 1 &&
                               cfg.value("intermediate_size", 0) == 256 && cfg.value("num_hidden_layers", 0) == 5 &&
                               cfg.value("sliding_window", 0) == 256 && cfg.value("norm_eps", 0.0) == 1e-5);
    const json q = cfg.value("quantization", json());
    ok("config: the file's codes kept (4-bit group 32)",
       q.value("bits", 0) == 4 && q.value("group_size", 0) == 32 && q.value("mode", "") == "affine", q.dump());

    const std::string hdr = read_file(base + "/plan/model.safetensors");
    const uint64_t hl = *reinterpret_cast<const uint64_t*>(hdr.data());
    const json h = json::parse(hdr.substr(8, hl), nullptr, false);
    auto has = [&](const std::string& n, const char* dt, std::vector<int64_t> shape) {
      const json& t = h.value(n, json());
      bool good = t.is_object() && t.value("dtype", "") == dt && t.value("shape", json::array()) == json(shape);
      ok("header: " + n + " " + dt, good, t.dump());
    };
    has("model.embed_tokens.weight", "U32", {512, 16});
    has("model.norm.weight", "BF16", {128});
    has("model.layers.0.conv.conv.weight", "BF16", {128, 7});
    has("model.layers.0.conv.in_proj.weight", "U32", {384, 16});
    has("model.layers.0.conv.out_proj.weight", "U32", {128, 16});
    has("model.layers.0.mlp.w1.weight", "U32", {256, 16});
    has("model.layers.0.mlp.w2.weight", "U32", {128, 32});
    has("model.layers.2.self_attn.q_proj.weight", "U32", {256, 16});
    has("model.layers.2.self_attn.k_proj.weight", "U32", {128, 16});
    has("model.layers.2.self_attn.o_proj.weight", "U32", {128, 32});
    has("model.layers.2.self_attn.gate_proj.weight", "U32", {2, 16});
    has("model.layers.2.self_attn.q_norm.weight", "BF16", {128});
    has("model.layers.2.operator_norm.weight", "BF16", {128});
    has("model.layers.2.post_ffn_norm.weight", "BF16", {128});
    ok("header: no lm_head (the embedding is tied)", !h.contains("lm_head.weight"));
    ok("header: rope_freqs.weight not converted", !h.contains("rope_freqs.weight"));

    const json imp = read_json(base + "/plan/import.json");
    ok("import.json: a planned write", imp.value("write", uint64_t(0)) > 0);
  }

  // full mode: the model folder, its tokenizer.json with LFM2's split, and its chat template
  const ImportOptions full{.file = gguf, .name = "lfm3-test", .progress = false, .drafter = false};
  err = first_error(gguf, full);
  ok("full import runs", err.empty(), err);
  if (err.empty()) {
    const std::string dir = base + "/home/models/lfm3-test";
    const json tj = read_json(dir + "/tokenizer.json");
    const json& pt = tj["pre_tokenizer"]["pretokenizers"];
    ok("tokenizer.json: LFM2's one Isolated split and a ByteLevel without its regex",
       pt.size() == 2 && pt[0]["pattern"].value("Regex", "") == ni::Tokenizer::kLfm2Split &&
           pt[0].value("behavior", "") == "Isolated" && pt[1].value("type", "") == "ByteLevel" &&
           !pt[1].value("use_regex", true), pt.dump());
    ni::Tokenizer tok;
    bool loaded = true;
    try {
      tok.load(dir + "/tokenizer.json");
    } catch (const std::exception& e) {
      loaded = false;
      ok("the generated tokenizer.json loads", false, e.what());
    }
    if (loaded) ok("the generated tokenizer.json loads", true);
    const json tc = read_json(dir + "/tokenizer_config.json");
    ok("tokenizer_config.json: the GGUF's chat template",
       tc.contains("chat_template") && !tc["chat_template"].get<std::string>().empty());
    ok("the model folder has config.json and model.safetensors",
       !read_json(dir + "/config.json").is_discarded() &&
           std::ifstream(dir + "/model.safetensors").good());
  }

  // each shape the path does not run, refused alone
  const std::vector<std::pair<const char*, Spec>> bad = {
      {"an attention head size of 64 (128 needed)", {.head_dim = 64}},
      {"a short convolution of 4 (7 needed)", {.l_cache = 4}},
      {"a grouped-query ratio above 16", {.hq = 32}},
      {"a width that is not a multiple of 64", {.H = 96}},
  };
  for (const auto& [why, s] : bad) {
    const std::string g = make_gguf(base, "lfm3-bad", s);
    const ImportOptions p{.file = g, .name = "lfm3-bad-" + std::to_string(g_checks), .plan_dir = base + "/plan",
                          .progress = false, .drafter = false};
    const std::string e = first_error(g, p);
    ok(std::string("refused: ") + why, e.find(why) != std::string::npos, e);
  }
  {
    Spec s;
    s.attn = {0, 0, 1, 0};  // one entry short of the layer count
    const std::string g = make_gguf(base, "lfm3-bad-arrays", s);
    const ImportOptions p{.file = g, .name = "lfm3-bad-arrays", .plan_dir = base + "/plan", .progress = false,
                          .drafter = false};
    const std::string e = first_error(g, p);
    ok("refused: a per-layer array that is not one entry per layer",
       e.find("has no per-layer lfm3.") != std::string::npos, e);
  }
  {
    Spec s;
    s.attn = {0, 0, 0, 0, 0};  // all convolutions
    const std::string g = make_gguf(base, "lfm3-bad-noattn", s);
    const ImportOptions p{.file = g, .name = "lfm3-bad-noattn", .plan_dir = base + "/plan", .progress = false,
                          .drafter = false};
    const std::string e = first_error(g, p);
    ok("refused: no attention layers", e.find("no attention layers") != std::string::npos, e);
  }

  // the partial rotation from the baked zero-tail table, and Lfm3Config's default without it
  {
    Spec s;
    s.rope_freqs = false;
    const std::string g = make_gguf(base, "lfm3-nofreqs", s);
    const ImportOptions p{.file = g, .name = "lfm3-nofreqs", .plan_dir = base + "/plan2", .progress = false,
                          .drafter = false};
    err = first_error(g, p);
    const double prf = err.empty() ? read_json(base + "/plan2/config.json")["rope_parameters"]["full_attention"]
                                         .value("partial_rotary_factor", 0.0)
                                   : 0.0;
    ok("no rope_freqs.weight: Lfm3Config's 0.5 default", err.empty() && prf == 0.5, err);
  }
  {
    Spec s;
    s.rope_nz = 64;  // a full-dim table: no partial rotation
    const std::string g = make_gguf(base, "lfm3-fullfreqs", s);
    const ImportOptions p{.file = g, .name = "lfm3-fullfreqs", .plan_dir = base + "/plan3", .progress = false,
                          .drafter = false};
    err = first_error(g, p);
    const double prf = err.empty() ? read_json(base + "/plan3/config.json")["rope_parameters"]["full_attention"]
                                         .value("partial_rotary_factor", 0.0)
                                   : 0.0;
    ok("a table of all nonzero frequencies: full rotation", err.empty() && prf == 1.0, err);
  }

  printf("%s (%d checks, %d failed)\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
  return g_fail ? 1 : 0;
}
