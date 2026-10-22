#include "model/nemotron_h.h"
#include "model/kflash_budget.h"
#include "ane/ffn_split.h"
#include "util/bf16.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace ni {

extern const char* const kNhKernelSource;
extern const char* const kTensorKernelSource;

namespace {

// host mirrors of the kernel parameter structs (src/kernels/nemotron/*.metal)
struct MvP {
  uint32_t N, K, x_stride, y_stride;
  float gs;
  uint32_t R;
};
struct MoeP {
  uint32_t I, D, k, up_q_stride, up_s_stride, dn_q_stride, dn_s_stride, h_stride;
};
struct EmbedP {
  uint32_t D, tok_off;
};
struct NormP {
  uint32_t D;
  float eps;
  uint32_t x_stride, y_stride;
};
struct GnormP {
  uint32_t D, G;
  float eps;
};
struct SsmP {
  uint32_t T, pos0, write, zx_stride, NH, HD, DS, NG;
};
struct RingP {
  uint32_t T, pos0, zx_stride, DI, CH, ptree;
};
struct Ssm1P {
  uint32_t np, ptree;
};
struct KvP {
  uint32_t R, pos0, cap, Hkv, HD;
};
struct AttnP {
  uint32_t R, pos0, cap, Hq, Hkv, chunk, nchunks;
  float scale;
};
struct RouteP {
  uint32_t E, k, norm;
  float scale;
  uint32_t logit_stride;
  uint32_t rows;
};
struct GemmP {
  uint32_t M, N, K, x_ld, y_ld;
  float gs;
  uint32_t epi, q_estride, s_estride, k;
};
struct GemmF32P {
  uint32_t M, N, K, x_ld, y_ld;
};
struct MoeDecP {  // nh_zdec.metal
  uint32_t H, E, k, norm;
  float scale;
  uint32_t L, S, I, up_q_stride, up_s_stride, tg_r, tg_l, tg_u;
};
struct V8P {  // nh_v8.metal
  uint32_t N, K, x_ld, y_ld, R;
  float gs;
  uint32_t epi;
};
struct BucketP {
  uint32_t P, E, BM;
  uint32_t k;  // with k: the pairs of rows >= live[0] are left out
};
struct CombineP {
  uint32_t D, k;
};
struct HalfP {
  uint32_t D, norm;
  float eps;
  uint32_t x_ld;
};
struct AttnPfP {
  uint32_t T, pos0, cap, Hq, Hkv;
  float scale;
};
struct TopkStageP {  // sample.metal (base library)
  uint32_t V, K, chunk;
};
struct SampleP {
  uint32_t n, K;
  float temp, top_p;
  uint32_t seed, step, out_index;
};
struct SampleRowsP {  // verify.metal (base library)
  uint32_t n, K;
  float temp, top_p;
  uint32_t seed, step, out_index, remap_from, remap_base, table_from;
};

Pipeline pipe(const std::string& n) {
  static std::mutex mu;
  static std::unordered_map<std::string, Pipeline> cache;
  std::lock_guard<std::mutex> lk(mu);
  auto it = cache.find(n);
  if (it == cache.end()) it = cache.emplace(n, Gpu::get().pipeline(n)).first;
  return it->second;
}

size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

}  // namespace

// ---- discovery ----
namespace {

nlohmann::json read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) return {};
  try {
    return nlohmann::json::parse(f);
  } catch (...) {
    return {};
  }
}

// The language model's part of a checkpoint's config.json: llm_config (Nemotron Omni releases), text_config, or the
// whole file (text-only releases).
const nlohmann::json& text_config(const nlohmann::json& j) {
  for (const char* k : {"llm_config", "text_config"})
    if (j.contains(k) && j[k].is_object()) return j[k];
  return j;
}

bool nh_type(const nlohmann::json& t) {
  if (!t.is_object() || !t.contains("model_type") || !t["model_type"].is_string()) return false;
  const std::string mt = t["model_type"];
  return mt == "nemotron_h" || mt == "nemotron_h_puzzle";
}

// Tensors the language model does not use (vision tower, projector, sound encoder) stay out of the index.
bool not_lm(const std::string& n) {
  return n.find("backbone.") == std::string::npos && n.find("lm_head.") == std::string::npos;
}

}  // namespace

bool NemotronH::is_model(const std::string& dir) {
  struct stat st;
  if (stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return false;
  return nh_type(text_config(read_json(dir + "/config.json")));
}

// ---- config ----
static NhConfig read_config(const std::string& dir) {
  const nlohmann::json j = read_json(dir + "/config.json");
  const nlohmann::json& t = text_config(j);
  if (!nh_type(t)) fatal("%s: config.json is not a Nemotron-H model", dir.c_str());
  auto num = [&](const nlohmann::json& o, const char* k, double def) {
    return o.contains(k) && o[k].is_number() ? o[k].get<double>() : def;
  };
  auto str = [&](const char* k, const char* def) {
    return t.contains(k) && t[k].is_string() ? t[k].get<std::string>() : std::string(def);
  };
  NhConfig c;
  c.hidden = int(num(t, "hidden_size", 0));
  c.vocab = int(num(t, "vocab_size", 0));
  c.n_head = int(num(t, "mamba_num_heads", 0));
  c.head_dim = int(num(t, "mamba_head_dim", 0));
  c.d_inner = c.n_head * c.head_dim;
  c.d_state = int(num(t, "ssm_state_size", 0));
  c.n_group = int(num(t, "n_groups", 1));
  c.conv_k = int(num(t, "conv_kernel", 4));
  c.hq = int(num(t, "num_attention_heads", 0));
  c.hkv = int(num(t, "num_key_value_heads", 0));
  c.hd = int(num(t, "head_dim", c.hq ? c.hidden / c.hq : 0));
  c.latent = int(num(t, "moe_latent_size", c.hidden));
  c.shared_I = int(num(t, "moe_shared_expert_intermediate_size", 0)) * int(num(t, "n_shared_experts", 1));
  c.w_scale = float(num(t, "routed_scaling_factor", 1.0));
  c.w_norm = !t.contains("norm_topk_prob") || !t["norm_topk_prob"].is_boolean() || t["norm_topk_prob"].get<bool>();
  c.eps = float(num(t, "layer_norm_epsilon", num(t, "rms_norm_eps", num(t, "norm_eps", 1e-5))));
  if (str("scoring_func", "sigmoid") != "sigmoid" || num(t, "n_group", 1) > 1 || num(t, "topk_group", 1) > 1)
    fatal("%s: expert routing other than sigmoid without groups is not supported", dir.c_str());
  if (str("mamba_hidden_act", "silu") != "silu" || str("mlp_hidden_act", str("hidden_act", "relu2").c_str()) != "relu2")
    fatal("%s: activations other than silu (Mamba) / relu2 (experts) are not supported", dir.c_str());
  auto flag = [&](const char* k, bool def) { return t.contains(k) && t[k].is_boolean() ? t[k].get<bool>() : def; };
  if (flag("attention_bias", false) || flag("mamba_proj_bias", false) || flag("mlp_bias", false) ||
      flag("use_qk_norm", false) || !flag("use_conv_bias", true))
    fatal("%s: attention / projection biases, qk-norm or a conv1d without bias are not supported", dir.c_str());
  if (t.contains("time_step_limit") && t["time_step_limit"].is_array() && t["time_step_limit"].size() == 2 &&
      t["time_step_limit"][0].is_number() && t["time_step_limit"][0].get<double>() > 0.0)
    fatal("%s: a time_step_limit above 0 is not supported", dir.c_str());
  if (t.contains("time_step_limit") && t["time_step_limit"].is_array() && t["time_step_limit"].size() == 2 &&
      t["time_step_limit"][1].is_number() && std::isfinite(t["time_step_limit"][1].get<double>()))
    fatal("%s: a finite time_step_limit is not supported", dir.c_str());
  // layer kinds: Puzzle block_configs (with per-layer expert counts and widths), layers_block_type, or the pattern
  std::vector<nlohmann::json> blocks;
  if (t.contains("block_configs") && t["block_configs"].is_array()) {
    for (auto& b : t["block_configs"]) blocks.push_back(b);
  } else if (t.contains("layers_block_type") && t["layers_block_type"].is_array()) {
    for (auto& b : t["layers_block_type"]) blocks.push_back({{"block_type", b}});
  } else if (t.contains("hybrid_override_pattern") && t["hybrid_override_pattern"].is_string()) {
    for (char ch : t["hybrid_override_pattern"].get<std::string>())
      blocks.push_back({{"block_type", ch == 'M' ? "mamba" : ch == '*' ? "attention" : ch == 'E' ? "moe" : "mlp"}});
  }
  c.layers = int(blocks.size());
  c.kind.resize(size_t(c.layers));
  c.n_exp.assign(size_t(c.layers), 0);
  c.top_k.assign(size_t(c.layers), 0);
  c.exp_I.assign(size_t(c.layers), 0);
  c.mlp_I.assign(size_t(c.layers), 0);
  int n_mlp = 0;  // dense MLP layers so far (an intermediate_size list has one entry per MLP layer)
  for (int i = 0; i < c.layers; ++i) {
    const nlohmann::json& b = blocks[size_t(i)];
    const std::string bt = b.contains("block_type") && b["block_type"].is_string() ? b["block_type"].get<std::string>() : "";
    if (bt == "mamba") {
      c.kind[size_t(i)] = 'M';
    } else if (bt == "attention") {
      c.kind[size_t(i)] = 'A';
    } else if (bt == "moe") {
      c.kind[size_t(i)] = 'E';
      c.n_exp[size_t(i)] = int(num(b, "n_routed_experts", num(t, "n_routed_experts", num(t, "num_experts", num(t, "num_local_experts", 0)))));
      c.top_k[size_t(i)] = int(num(b, "num_experts_per_tok", num(t, "num_experts_per_tok", 0)));
      c.exp_I[size_t(i)] = int(num(b, "moe_intermediate_size", num(t, "moe_intermediate_size", 0)));
      c.max_E = std::max(c.max_E, c.n_exp[size_t(i)]);
      c.max_k = std::max(c.max_k, c.top_k[size_t(i)]);
      c.max_I = std::max(c.max_I, c.exp_I[size_t(i)]);
    } else if (bt == "mlp") {
      // a dense MLP block (NVIDIA's NemotronHMLP: down_proj(relu^2(up_proj(x))), no bias); its width: the block's own
      // intermediate_size, else config.json's (one number, or a list with one entry per MLP layer as NVIDIA's
      // modeling code reads it)
      c.kind[size_t(i)] = 'D';
      int w = int(num(b, "intermediate_size", 0));
      if (w <= 0 && t.contains("intermediate_size") && t["intermediate_size"].is_array()) {
        const nlohmann::json& a = t["intermediate_size"];
        const size_t j = a.size() == 1 ? 0 : size_t(n_mlp);
        if (j < a.size() && a[j].is_number()) w = a[j].get<int>();
      } else if (w <= 0) {
        w = int(num(t, "intermediate_size", 0));
      }
      if (w <= 0) fatal("%s: MLP layer %d has no intermediate_size", dir.c_str(), i);
      c.mlp_I[size_t(i)] = w;
      c.max_mlp_I = std::max(c.max_mlp_I, w);
      ++n_mlp;
    } else {
      fatal("%s: layer %d is a '%s' block, which is not supported", dir.c_str(), i, bt.c_str());
    }
  }
  // end of sequence: generation_config.json's ids (</s> and <|im_end|> for the Nemotron 3.5 releases), else config's
  auto add_ids = [&](const nlohmann::json& v, std::vector<int>& out) {
    if (v.is_number_integer()) out.push_back(v.get<int>());
    if (v.is_array())
      for (auto& x : v)
        if (x.is_number_integer() && std::find(out.begin(), out.end(), x.get<int>()) == out.end()) out.push_back(x.get<int>());
  };
  const nlohmann::json g = read_json(dir + "/generation_config.json");
  if (g.is_object() && g.contains("eos_token_id")) add_ids(g["eos_token_id"], c.eos);
  if (c.eos.empty() && t.contains("eos_token_id")) add_ids(t["eos_token_id"], c.eos);
  if (j.contains("eos_token_id")) add_ids(j["eos_token_id"], c.eos);
  c.bos = int(num(t, "bos_token_id", -1));
  // (Mamba head dim 64 (Nemotron 3.5) or 80 (Nemotron Nano 4B / 9B v2): the prefill scan's instantiations, nh_ssm /
  // nh_ssm_h80; the short passes' nh_ssm1 takes any multiple of 8)
  if (c.hidden <= 0 || c.layers <= 0 || c.vocab <= 0 || c.d_inner <= 0 || c.n_head <= 0 || c.d_state != 128 ||
      (c.head_dim != 64 && c.head_dim != 80) || c.conv_k != 4 || c.n_group <= 0 || c.n_head % c.n_group)
    fatal("%s: unsupported Nemotron-H shape (hidden %d, Mamba inner %d, state %d, heads %d, conv %d)", dir.c_str(),
          c.hidden, c.d_inner, c.d_state, c.n_head, c.conv_k);
  if (std::count(c.kind.begin(), c.kind.end(), 'A') && (c.hd != 128 || !c.hkv || c.hq % c.hkv))
    fatal("%s: attention head dim %d is not supported", dir.c_str(), c.hd);
  // (the router keeps 16 experts per lane in registers: E <= 512, k <= 32)
  if (c.max_E > 512 || c.max_k > 32) fatal("%s: %d experts / top-%d is not supported", dir.c_str(), c.max_E, c.max_k);
  return c;
}

namespace {

// A model without MoE layers (Mamba-2, attention and dense MLP layers: Nemotron Nano 4B / 9B v2). The choices below
// that only these make (the LM head's format, the NVFP4 layout, the kernels for K % 128 != 0) leave the MoE models'
// paths as they were.
bool dense_model(const NhConfig& c) { return std::find(c.kind.begin(), c.kind.end(), 'E') == c.kind.end(); }

// The LM head as Q8_0 rows (quantized at load, half the bytes a decoded token reads) for a dense model whose own
// linears are NVFP4 or FP8 (Nemotron Nano 9B v2's NVFP4 and Nano 4B's FP8 checkpoints, both of which NVIDIA quantized
// with the head left BF16), as Dory runs its NVFP4 checkpoint's head; NI_NH_HEAD=bf16 keeps it BF16, =q8 also
// quantizes a BF16 checkpoint's (a dense model's).
bool head_q8(const NhConfig& c, const SafetensorsIndex& idx, const std::string& pre) {
  if (!dense_model(c)) return false;
  if (const char* v = getenv("NI_NH_HEAD"); v && *v) {
    if (strcmp(v, "bf16") && strcmp(v, "q8")) fatal("NI_NH_HEAD=%s: expected bf16 or q8", v);
    return !strcmp(v, "q8");
  }
  for (int i = 0; i < c.layers; ++i)
    for (const char* n : {"in_proj", "up_proj"}) {
      const std::string w = pre + "backbone.layers." + std::to_string(i) + ".mixer." + n + ".weight";
      const SafetensorsIndex::Entry* s = idx.find(w + "_scale");
      if (s && s->dtype == DType::F8E4M3) return true;  // NVFP4 block scales
      const SafetensorsIndex::Entry* q = idx.find(w);
      if (q && q->dtype == DType::F8E4M3) return true;  // FP8 E4M3 weights (ModelOpt FP8)
    }
  return false;
}

}  // namespace

// ---- weight formats ----
// Tensors as NVIDIA's checkpoints store them (hf_quant_config.json "MIXED_PRECISION"): Q8_0 and IQ2_XXS as GGUF
// blocks in uint8 tensors [N][K / block][block bytes]; NVFP4 as ModelOpt writes it (uint8 [N][K / 2], low nibble =
// even element; F8_E4M3 scales [N][K / 16]; F32 weight_scale_2, per expert in an expert stack); FP8 as ModelOpt
// writes it (F8_E4M3 [N][K], one F32 weight_scale per tensor; the static input_scale is not used: W8A16); BF16 for
// embeddings, router, norms, Mamba vectors and latent projections.
void nh_row_bytes(NhFmt f, int64_t K, size_t& qb, size_t& sb) {
  switch (f) {
    case NhFmt::Q8: qb = size_t(K); sb = size_t(K / 32) * 2; break;
    case NhFmt::FP4: qb = size_t(K / 2); sb = size_t(K / 16); break;
    case NhFmt::IQ2: qb = size_t(K / 4); sb = size_t(K / 256) * 2; break;
    case NhFmt::BF16: qb = size_t(K) * 2; sb = 0; break;
    case NhFmt::F32: qb = size_t(K) * 4; sb = 0; break;
    case NhFmt::FP8: qb = size_t(K); sb = 0; break;
  }
}

size_t nh_src_row_bytes(NhFmt f, int64_t K) {
  switch (f) {
    case NhFmt::Q8: return size_t(K / 32) * 34;
    case NhFmt::IQ2: return size_t(K / 256) * 66;
    case NhFmt::FP4: return size_t(K / 2);
    case NhFmt::BF16: return size_t(K) * 2;
    case NhFmt::F32: return size_t(K) * 4;
    case NhFmt::FP8: return size_t(K);
  }
  return 0;
}

namespace {
// nibble i of x -> the low nibble of byte i
inline uint64_t spread_nibbles(uint32_t x) {
  uint64_t v = x;
  v = (v | (v << 16)) & 0x0000FFFF0000FFFFull;
  v = (v | (v << 8)) & 0x00FF00FF00FF00FFull;
  v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0Full;
  return v;
}
}  // namespace

void nh_repack_rows(NhFmt f, int64_t K, const uint8_t* src, int64_t rows, uint8_t* q, uint8_t* s, bool fp4_l2) {
  size_t qb, sb;
  nh_row_bytes(f, K, qb, sb);
  const size_t rb = nh_src_row_bytes(f, K);
  for (int64_t r = 0; r < rows; ++r) {
    uint8_t* qr = q + size_t(r) * qb;
    uint8_t* sr = s ? s + size_t(r) * sb : nullptr;
    const uint8_t* b = src + size_t(r) * rb;
    switch (f) {
      case NhFmt::Q8:
        for (int64_t i = 0; i < K / 32; ++i, b += 34) {
          memcpy(sr + i * 2, b, 2);
          memcpy(qr + i * 32, b + 2, 32);
        }
        break;
      case NhFmt::IQ2:
        for (int64_t i = 0; i < K / 256; ++i, b += 66) {
          memcpy(sr + i * 2, b, 2);
          memcpy(qr + i * 64, b + 2, 64);
        }
        break;
      case NhFmt::FP4:
        // ModelOpt order (byte j of a 16-element group: elements 2j, 2j + 1) -> L0, the kernels' (byte i: i, i + 8);
        // or L2 (word h: nibble j = element 4h + j, nibble 4 + j = element 8 + 4h + j, codes as (m2 m1 m0 s)): read
        // as one little-endian word, ModelOpt's nibble e is element e, so L2 swaps the nibble quads of elements 4-7
        // and 8-11 and rotates each code's sign bit to the bottom
        for (int64_t i = 0; i < K / 16; ++i) {
          uint64_t v;
          memcpy(&v, b + i * 8, 8);
          uint64_t o;
          if (fp4_l2) {
            o = (v & 0xFFFF00000000FFFFull) | ((v >> 16) & 0x00000000FFFF0000ull) | ((v << 16) & 0x0000FFFF00000000ull);
            o = ((o & 0x7777777777777777ull) << 1) | ((o >> 3) & 0x1111111111111111ull);
          } else {
            o = spread_nibbles(uint32_t(v)) | (spread_nibbles(uint32_t(v >> 32)) << 4);
          }
          memcpy(qr + i * 8, &o, 8);
        }
        break;
      case NhFmt::FP8:
        // 16-element groups: byte 4 m + (0, 1, 2, 3) = elements 2 m, 2 m + 1, 2 m + 8, 2 m + 9, so the bytes 0, 2 /
        // 1, 3 of a word are the pairs (i, i + 8) the multi-row kernels' operands take (nh_x_fp8.metal)
        for (int64_t i = 0; i < K / 16; ++i)
          for (int m = 0; m < 4; ++m) {
            const uint8_t* g = b + i * 16;
            uint8_t* o = qr + i * 16 + 4 * m;
            o[0] = g[2 * m];
            o[1] = g[2 * m + 1];
            o[2] = g[2 * m + 8];
            o[3] = g[2 * m + 9];
          }
        break;
      case NhFmt::BF16:
      case NhFmt::F32: memcpy(qr, b, qb); break;
    }
  }
}

void nh_quant_q8(const float* x, int64_t K, uint8_t* q, uint8_t* s) {
  for (int64_t b = 0; b < K / 32; ++b) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
    const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
    const __fp16 dh = __fp16(d);
    memcpy(s + b * 2, &dh, 2);
    for (int i = 0; i < 32; ++i) {
      const int v = int(std::lround(x[b * 32 + i] * id));
      q[b * 32 + i] = uint8_t(int8_t(std::clamp(v, -127, 127)));
    }
  }
}

// The format and shape of a weight matrix from its safetensors entries (scale: NVFP4's weight_scale, else null).
bool nh_describe(const SafetensorsIndex::Entry& w, const SafetensorsIndex::Entry* scale, NhFmt& f, uint32_t& N,
                 uint32_t& K) {
  const auto& sh = w.shape;
  if (w.dtype == DType::U8 && sh.size() == 3 && (sh[2] == 34 || sh[2] == 66)) {
    f = sh[2] == 34 ? NhFmt::Q8 : NhFmt::IQ2;
    N = uint32_t(sh[0]);
    K = uint32_t(sh[1] * (sh[2] == 34 ? 32 : 256));
    return true;
  }
  if (w.dtype == DType::U8 && sh.size() == 2 && scale && scale->dtype == DType::F8E4M3 && scale->shape.size() == 2 &&
      scale->shape[0] == sh[0] && scale->shape[1] * 8 == sh[1]) {
    f = NhFmt::FP4;
    N = uint32_t(sh[0]);
    K = uint32_t(sh[1] * 2);
    return true;
  }
  if (w.dtype == DType::F8E4M3 && sh.size() == 2 && sh[1] % 16 == 0 && scale && scale->dtype == DType::F32 &&
      scale->numel() == 1) {
    f = NhFmt::FP8;
    N = uint32_t(sh[0]);
    K = uint32_t(sh[1]);
    return true;
  }
  if ((w.dtype == DType::BF16 || w.dtype == DType::F32) && sh.size() == 2) {
    f = w.dtype == DType::BF16 ? NhFmt::BF16 : NhFmt::F32;
    N = uint32_t(sh[0]);
    K = uint32_t(sh[1]);
    return true;
  }
  return false;
}

bool NemotronH::split_merges() {
  static const bool split = [] {
    // macOS 14 always splits: its library is compiled without the merges (NH_FENCES 0), so NI_NH_SPLIT=0 may only
    // turn the split off where the fences exist (Gpu::compile_library: Metal 3.2+, macOS 15)
    if (!__builtin_available(macOS 15.0, *)) return true;
    if (const char* v = getenv("NI_NH_SPLIT")) return atoi(v) != 0;
    return false;
  }();
  return split;
}

NemotronH::NemotronH(const std::string& path, int prefill_chunk) {
  path_ = path;
  while (path_.size() > 1 && path_.back() == '/') path_.pop_back();
  if (!is_model(path_)) fatal("%s: not a Nemotron-H checkpoint folder (config.json)", path.c_str());
  cfg_ = read_config(path_);
  dense_ = dense_model(cfg_);
  Gpu::get().compile_extra("Nemotron-H", kNhKernelSource);
  // the M5's matrix units (Apple10, Metal 4 on macOS 26) for the 8-row NVFP4 GEMMs
  if (Gpu::get().apple_family() >= 10 && !(getenv("NI_NH_V8T") && atoi(getenv("NI_NH_V8T")) == 0))
    if (__builtin_available(macOS 26.0, *)) {
      Gpu::get().compile_extra("tensor", kTensorKernelSource, true);
      v8t_ = true;
    }
  load(prefill_chunk);
  prewarm_pipelines();
  init_ane();
}

// The pipelines of the decode, verify, prefill and drafter passes, created up front in parallel so the first request
// does not create them inside its prompt. NI_PREWARM=0 skips it.
void NemotronH::prewarm_pipelines() {
  if (getenv("NI_PREWARM") && !strcmp(getenv("NI_PREWARM"), "0")) return;
  const double t0 = now_ms();
  Gpu& g = Gpu::get();
  static const char* prefixes[] = {"nh_embed", "nh_rmsnorm", "nh_mv1_fp4_n8", "nh_mv_", "nh_ssm", "nh_moe",
                                   "nh_kv_write", "nh_attn", "nh_mv1_iq2_c", "nh_v8_q8_s", "nh_v8_fp4_s", "nh_v8_iq2_s",
                                   "nh_rows", "nh_gnorm", "nh_conv_ring", "nh_accept", "nh_capture", "nh_gemm_",
                                   "nh_lili", "nh_cand", "nh_dconv", "nh_qk_norm", "nh_swiglu", "nh_add", "nh_bias",
                                   "nh_block_tok", "nh_anchor", "nh_iq2_fill_tables2", "topk_stage1", "sample_stage2",
                                   "copy_u32", "kv_regrow", "nh_x4_", "nh_x2_", "nh_x8_", "nh_xb_", "nh_dsp_",
                                   "nh_kv_tree", "nh_kf_", "nh_x4t", "nh_x2t", "nh_ssd", "nh_xq8_"};
  // the kernels only a dense model runs (the tail variants, head dim 80's scan, 5-head prefill attention, layout L2's
  // relu^2 decode, the short scans of several rows per simdgroup but the M5's nh_ssm1r4): not created for the MoE
  // models, whose prewarm stays as it was
  const bool m5 = g.apple10();
  auto dense_only = [m5](const std::string& n) {
    return (n.size() > 2 && n.compare(n.size() - 2, 2, "_t") == 0) || n == "nh_ssm_h80" ||
           n == "nh_attn_pf2_q1_h5_k32" || n == "nh_x4_mv_relu2" ||
           (n.rfind("nh_ssm1r", 0) == 0 && !(m5 && n == "nh_ssm1r4"));
  };
  // (the routed experts' layout L2 twins: only when a loaded layer's experts are in L2)
  const bool any_l2 =
      std::any_of(layers_.begin(), layers_.end(), [](const NhLayer& L) { return L.up_e.l2 || L.down_e.l2; });
  std::vector<std::string> v;
  for (const std::string& n : g.function_names())
    for (const char* pre : prefixes)
      if (n.rfind(pre, 0) == 0) {
        const bool el2 = n.find("fp4l2_moe") != std::string::npos ||
                         (n.rfind("nh_x4_moe", 0) == 0 && n.size() > 3 && n.compare(n.size() - 3, 3, "_l2") == 0) ||
                         n == "nh_moe_b4_l2";
        // (the prefill's chunked scan nh_ssd_*: only with the M5's tile GEMMs, the switch that selects it)
        if ((dense_ || !dense_only(n)) && (any_l2 || !el2) && (tg_ || n.rfind("nh_ssd", 0) != 0)) v.push_back(n);
        break;
      }
  const unsigned nt = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  for (unsigned i = 0; i < nt; ++i)
    th.emplace_back([&, i] {
      for (size_t j = i; j < v.size(); j += nt) g.pipeline(v[j], false);
    });
  for (auto& t : th) t.join();
  logi("Nemotron-H: prewarmed %zu pipelines in %.0f ms", v.size(), now_ms() - t0);
}

// ---- GPU + Neural Engine prefill split (nemotron_ane.h), on by default ----
//   NI_ANE=0            the GPU runs the prefill alone
//   NI_ANE_IN / NI_ANE_OUT=<f>   fraction of in_proj / out_proj rows on the ANE (default 0.7)
//   NI_ANE_FFN=0        the shared expert stays on the GPU
//   NI_ANE_IN_FIRST=0   the ANE takes in_proj's last rows (x, B, C, dt) instead of its first (z, x)
//   NI_ANE_FFN_A16=0    the shared expert's intermediate as int8 per token on the ANE (default fp16: relu^2 is
//                       heavy-tailed; KL to the fp32 reference 1.96e-3 against 3.67e-3, at the same speed)
//   NI_ANE_MIN_ROWS=<r> smallest prefill chunk that splits (default 512)
//   NI_ANE_FNS=a,b,..   ANE program functions (rows; default 512, 1024, 1536, 2048)
void NemotronH::init_ane() {
  if (const char* on = getenv("NI_ANE"); on && !atoi(on)) return;
  if (Encoder::profiling()) {
    logi("NI_ANE: off under NI_PROFILE");
    return;
  }
  NhAneConfig c;
  if (const char* v = getenv("NI_ANE_IN")) c.in_share = atof(v);
  if (const char* v = getenv("NI_ANE_OUT")) c.out_share = atof(v);
  if (const char* v = getenv("NI_ANE_FFN")) c.ffn = atoi(v) != 0;
  if (const char* v = getenv("NI_ANE_IN_FIRST")) c.in_first = atoi(v) != 0;
  if (const char* v = getenv("NI_ANE_FFN_A16")) c.ffn_a16 = atoi(v) != 0;
  if (const char* v = getenv("NI_ANE_MIN_ROWS")) c.min_rows = uint32_t(atoi(v));
  if (const char* v = getenv("NI_ANE_FNS")) {
    for (std::string t = v; !t.empty();) {
      const size_t q = t.find(',');
      c.fn_rows.push_back(uint32_t(std::stoul(t.substr(0, q))));
      t = q == std::string::npos ? "" : t.substr(q + 1);
    }
  }
  c.cache_dir = ane::cache_dir();
  c.model_key = ane::model_key(path_);
  ane_cfg_ = c;
  if (getenv("NI_ANE_ASYNC") && atoi(getenv("NI_ANE_ASYNC"))) {
    // the server: the ANE daemon compiles the program in the background (tens of seconds when it evicted it, ~0.1 s
    // when it still holds it); prefill runs on the GPU alone until ane_poll() sees it done and builds the split
    NhAneConfig cc = c;
    cc.compile_only = true;
    auto st = std::make_shared<std::atomic<int>>(1);
    ane_async_ = st;
    ane_thread_ = std::thread([this, st, cc] {
      const double t0 = now_ms();
      try {
        NhAne warm(*this, cc);
        logi("NI_ANE: Neural Engine program ready in the background (%s, %.1f s)", warm.note().c_str(),
             (now_ms() - t0) / 1000);
        st->store(2);
      } catch (const std::exception& e) {
        logi("NI_ANE: Neural Engine split unavailable (%s); the GPU runs the prefill alone", e.what());
        st->store(3);
      }
    });
    logi("NI_ANE: compiling the Neural Engine program in the background; prefill uses the GPU alone until it is ready");
    return;
  }
  ane_build();
}

void NemotronH::ane_poll() {
  if (!ane_async_) return;
  const int st = ane_async_->load();
  if (st == 1) return;  // still compiling
  if (ane_thread_.joinable()) ane_thread_.join();
  ane_async_.reset();
  if (st == 2) ane_build();
}

void NemotronH::ane_build() {
  try {
    ane_ = std::make_unique<NhAne>(*this, ane_cfg_);
  } catch (const std::exception& e) {
    ane::Program::unpark_all();
    logi("NI_ANE: Neural Engine split unavailable (%s); the GPU runs the prefill alone", e.what());
    ane_.reset();
    return;
  }
  ane::Program::unpark_all();  // (a background compile parks its program for this build)
  if (mem_budget_ && Gpu::get().allocated_bytes() > mem_budget_) {
    logi("NI_ANE: the Neural Engine split's %.0f MB would exceed the memory budget (%.1f GB); GPU only",
         ane_->metal_bytes() / 1e6, mem_budget_ / 1e9);
    ane_.reset();
    return;
  }
  logi("NI_ANE: Neural Engine split: in_proj %u of %d rows, out_proj %u of %d, shared expert %s; %s; %.0f MB; ready "
       "in %.1f s",
       uint32_t(cfg_.in_proj()) - ane_->gpu_in(), cfg_.in_proj(), uint32_t(cfg_.hidden) - ane_->gpu_out(), cfg_.hidden,
       ane_->ffn() ? "on the ANE" : "on the GPU", ane_->note().c_str(), ane_->metal_bytes() / 1e6,
       ane_->startup_ms() / 1000);
}

bool NemotronH::ane_splits(int T) const { return ane_ && T > kMaxRows && ane_->splits(T); }

bool NemotronH::iq2_experts() const {
  for (const NhLayer& L : layers_)
    if (L.kind == 'E' && L.up_e.fmt == NhFmt::IQ2) return true;
  return false;
}

bool NemotronH::ane_finish_chunk() {
  if (!ane_chunk_) return true;
  ane_chunk_ = false;
  double ms = 0;
  const bool ok = ane_->finish_chunk(&ms);
  if (ok) {
    ane_stats_.ane_ms += ms;
    ane_stats_.evals += ane_->last_jobs();
  } else {
    ++ane_stats_.reruns;
    logi("NI_ANE: Neural Engine split stopped (%s); this chunk reruns on the GPU, which runs the prefill alone from now "
         "on", ane_->reason().c_str());
  }
  return ok;
}

void NemotronH::ane_abort(const std::string& why) {
  if (ane_chunk_ && ane_) ane_->retire(why);
}

bool NemotronH::ane_release() {
  if (ane_async_) {  // a compile still running in the background: never build the split
    if (ane_thread_.joinable()) ane_thread_.detach();
    ane_async_.reset();
  }
  if (!ane_) return false;
  logi("NI_ANE: releasing the Neural Engine split (%.0f MB) for KV cache; prefill runs on the GPU alone",
       ane_->metal_bytes() / 1e6);
  ane_.reset();
  return true;
}

NemotronH::~NemotronH() {
  if (ane_thread_.joinable()) ane_thread_.join();
  ane_.reset();
  if (embed_map_) munmap(embed_map_, embed_map_len_);
}

namespace {

// One conversion job: a byte range of a source tensor into its place in a weight buffer.
struct Job {
  const SafetensorsIndex::Entry* t = nullptr;
  enum Kind { Rows, Copy, Bf16ToF32, NegExpBf16, Bf16ToQ8 } kind = Copy;
  size_t src = 0, len = 0;  // bytes of the source tensor
  NhFmt fmt = NhFmt::F32;   // Rows: the matrix format
  int64_t K = 0, rows = 0;  // Rows
  uint8_t* q = nullptr;     // destination (codes)
  uint8_t* s = nullptr;     // destination (scales; Rows of Q8 / IQ2, Bf16ToQ8)
  bool l2 = false;          // Rows of NVFP4: codes in layout L2
};

void run_job(const SafetensorsIndex& idx, const Job& j, std::vector<uint8_t>& raw) {
  raw.resize(j.len);
  idx.read(*j.t, j.src, raw.data(), j.len);
  switch (j.kind) {
    case Job::Rows: nh_repack_rows(j.fmt, j.K, raw.data(), j.rows, j.q, j.s, j.l2); break;
    case Job::Bf16ToQ8: {
      std::vector<float> row(size_t(j.K));
      size_t qb, sb;
      nh_row_bytes(NhFmt::Q8, j.K, qb, sb);
      for (int64_t r = 0; r < j.rows; ++r) {
        const uint8_t* src = raw.data() + size_t(r) * size_t(j.K) * 2;
        for (int64_t i = 0; i < j.K; ++i) {
          uint16_t h;
          memcpy(&h, src + 2 * i, 2);
          row[size_t(i)] = bf16_to_f32(h);
        }
        nh_quant_q8(row.data(), j.K, j.q + size_t(r) * qb, j.s + size_t(r) * sb);
      }
      break;
    }
    case Job::Copy: memcpy(j.q, raw.data(), j.len); break;
    case Job::Bf16ToF32:
    case Job::NegExpBf16:
      for (size_t i = 0; i < j.len / 2; ++i) {
        uint16_t h;
        memcpy(&h, raw.data() + 2 * i, 2);
        const float v = j.kind == Job::Bf16ToF32 ? bf16_to_f32(h) : -std::exp(bf16_to_f32(h));
        memcpy(j.q + 4 * i, &v, 4);
      }
      break;
  }
}

}  // namespace

void NemotronH::load(int prefill_chunk) {
  const double t0 = now_ms();
  tmax_ = std::max(kMaxRows, prefill_chunk);
  Gpu& g = Gpu::get();
  if (const char* v = getenv("NI_NH_MAP_EMBED")) map_embed_ = atoi(v) != 0;
  st_ = std::make_unique<SafetensorsIndex>(path_, not_lm);
  const SafetensorsIndex& idx = *st_;
  const std::string pre = idx.find("language_model.backbone.embeddings.weight") ? "language_model." : "";
  // NVFP4 layout: L2 for a dense model on the M1's GPU family (Apple7), where L2's decode is a few integer ops in place
  // of L0's threadgroup table (Dory's shapes on an M1 Pro: one-row products 12-16% faster, the same outputs to the
  // bit), L0 elsewhere; NI_NH_FP4=l0 / l2 overrides. L2 needs the x4 kernels for every NVFP4 product of up to 32 rows
  // (NI_NH_X4, NI_NH_V8 and NI_NH_ROWS_GEMM on); the MoE models keep L0 (their expert kernels read L0 only).
  {
    const char* fv = getenv("NI_NH_FP4");
    if (fv && *fv && strcmp(fv, "l0") && strcmp(fv, "l2")) fatal("NI_NH_FP4=%s: expected l0 or l2", fv);
    auto on = [](const char* k) {
      const char* v = getenv(k);
      return !v || atoi(v) != 0;
    };
    fp4_l2_ = dense_model(cfg_) && on("NI_NH_X4") && on("NI_NH_V8") && on("NI_NH_ROWS_GEMM") &&
              (fv && *fv ? !strcmp(fv, "l2") : g.apple_family() <= 7);
    // NVFP4 routed experts without latent projections (Lightning): L2 on the M5, where a verify's expert products
    // (per pair, nh_x4_moe_up / _dn) were bound by L0's table decode, not by memory: up + down + sum 196.4 against
    // 223.0 us per layer (548 / 483 GB/s of distinct experts, tools/nh_moe_bench; a read-only pass of the same
    // bytes measured as fast as L2); NI_NH_FP4_EXPERTS=l0 / l2 overrides. Their kernels (decode, verify, prefill) have
    // L2 twins with the same outputs to the bit; the latent models' (nh_x4_moe_dn_sum) do not, nor the IQ2 ones
    const char* ev = getenv("NI_NH_FP4_EXPERTS");
    if (ev && *ev && strcmp(ev, "l0") && strcmp(ev, "l2")) fatal("NI_NH_FP4_EXPERTS=%s: expected l0 or l2", ev);
    experts_l2_ = ev && *ev ? !strcmp(ev, "l2") : g.apple10();
  }
  // 1. what goes where: every tensor's converted size, placed in 1 GB weight buffers (bigger tensors get their own)
  struct Want {
    NhMat* m = nullptr;                                  // a matrix (q, s placed)
    std::vector<const SafetensorsIndex::Entry*> parts;   // its source tensors: one, or one per expert
    std::vector<const SafetensorsIndex::Entry*> scales;  // NVFP4 block scales, one per part
    Buffer* vec = nullptr;                               // an fp32 vector
    Job::Kind vk = Job::Copy;
    std::vector<float> gse;                              // NVFP4 expert stacks: the experts' weight_scale_2
    bool q8 = false;                                     // a BF16 matrix quantized to Q8 rows at load
  };
  std::vector<Want> wants;
  // names: the parts' tensor names ("x.weight"); q8: a BF16 matrix becomes Q8 rows (NI_NH_BF16=1 keeps it BF16)
  static const bool keep_bf16 = getenv("NI_NH_BF16") && atoi(getenv("NI_NH_BF16"));
  static const bool attn_q8 = getenv("NI_NH_ATTN_Q8") && atoi(getenv("NI_NH_ATTN_Q8"));
  auto mat_from = [&](NhMat& m, std::vector<const SafetensorsIndex::Entry*> parts, const std::vector<std::string>& names,
                      bool q8 = false) {
    auto base = [](const std::string& n) { return n.substr(0, n.size() - 7); };  // "x.weight" -> "x"
    const SafetensorsIndex::Entry* sc = idx.find(base(names[0]) + ".weight_scale");
    uint32_t N = 0, K = 0;
    if (!nh_describe(*parts[0], sc, m.fmt, N, K))
      fatal("%s: tensor %s has a format Kapitsa does not run", path_.c_str(), names[0].c_str());
    for (auto* p : parts)
      if (p->shape != parts[0]->shape || p->dtype != parts[0]->dtype)
        fatal("%s: the experts of %s differ in shape", path_.c_str(), names[0].c_str());
    m.N = N;
    m.K = K;
    m.E = uint32_t(parts.size());
    Want w{&m, parts, {}, nullptr, Job::Copy, {}, false};
    if (q8 && m.fmt == NhFmt::BF16 && !keep_bf16 && K % 32 == 0 && m.E == 1) {
      m.fmt = NhFmt::Q8;
      w.q8 = true;
    }
    m.l2 = fp4_l2_ && m.fmt == NhFmt::FP4;
    nh_row_bytes(m.fmt, K, m.q_row, m.s_row);
    auto scalar = [&](const std::string& n, const char* what) {
      float v = 1.0f;
      if (const SafetensorsIndex::Entry* t = idx.find(n)) {
        if (t->dtype != DType::F32 || t->len != 4) fatal("%s: %s is not one F32", path_.c_str(), n.c_str());
        idx.read(*t, 0, &v, 4);
      } else if (what) {
        fatal("%s: %s has no %s", path_.c_str(), n.c_str(), what);
      }
      return v;
    };
    if (m.fmt == NhFmt::FP4) {
      // block scales and the tensor scale of every part (an expert stack: each expert's own)
      for (size_t e = 0; e < parts.size(); ++e) {
        const SafetensorsIndex::Entry* se = e == 0 ? sc : idx.find(base(names[e]) + ".weight_scale");
        if (!se || se->len != m.s_row * m.N) fatal("%s: %s has no NVFP4 scales of its size", path_.c_str(), names[e].c_str());
        w.scales.push_back(se);
        w.gse.push_back(scalar(base(names[e]) + ".weight_scale_2", nullptr));
      }
      if (m.E == 1) {
        m.gs = w.gse[0];
        w.gse.clear();
      }
    } else if (m.fmt == NhFmt::FP8) {
      if (m.E != 1) fatal("%s: FP8 expert stacks are not supported", path_.c_str());
      m.gs = scalar(base(names[0]) + ".weight_scale", "FP8 weight_scale") * 256.0f;
    }
    wants.push_back(std::move(w));
  };
  auto mat = [&](NhMat& m, const std::string& name, bool q8 = false) { mat_from(m, {&idx.get(name)}, {name}, q8); };
  auto experts = [&](NhMat& m, const std::string& prefix, int E, const char* which) {
    std::vector<const SafetensorsIndex::Entry*> parts;
    std::vector<std::string> names;
    for (int e = 0; e < E; ++e) {
      names.push_back(prefix + std::to_string(e) + "." + which + ".weight");
      parts.push_back(&idx.get(names.back()));
    }
    mat_from(m, std::move(parts), names);
  };
  auto vec = [&](Buffer& b, const std::string& name, bool neg_exp = false) {
    const SafetensorsIndex::Entry& t = idx.get(name);
    if (t.dtype != DType::BF16 && t.dtype != DType::F32)
      fatal("%s: tensor %s is neither BF16 nor F32", path_.c_str(), name.c_str());
    if (neg_exp && t.dtype != DType::BF16) fatal("%s: tensor %s is not BF16", path_.c_str(), name.c_str());
    wants.push_back({nullptr, {&t}, {}, &b, neg_exp ? Job::NegExpBf16 : t.dtype == DType::BF16 ? Job::Bf16ToF32 : Job::Copy,
                     {}, false});
  };
  const std::string bb = pre + "backbone.";
  layers_.resize(size_t(cfg_.layers));
  mat(embed_, bb + "embeddings.weight");
  mat(head_, idx.find(pre + "lm_head.weight") ? pre + "lm_head.weight" : bb + "embeddings.weight", head_q8(cfg_, idx, pre));
  vec(out_norm_, bb + "norm_f.weight");
  mamba_idx_.assign(size_t(cfg_.layers), -1);
  int nm = 0;
  for (int i = 0; i < cfg_.layers; ++i) {
    NhLayer& L = layers_[size_t(i)];
    const std::string b = bb + "layers." + std::to_string(i) + ".", mx = b + "mixer.";
    L.kind = cfg_.kind[size_t(i)];
    vec(L.norm, b + "norm.weight");
    if (L.kind == 'M') {
      mamba_idx_[size_t(i)] = nm++;
      mat(L.in, mx + "in_proj.weight");
      mat(L.out, mx + "out_proj.weight");
      vec(L.conv_w, mx + "conv1d.weight");
      vec(L.conv_b, mx + "conv1d.bias");
      vec(L.A, mx + "A_log", true);  // A = -exp(A_log)
      vec(L.D, mx + "D");
      vec(L.dt_bias, mx + "dt_bias");
      vec(L.gnorm, mx + "norm.weight");
      if (int(L.in.N) != cfg_.in_proj() || int(L.in.K) != cfg_.hidden || int(L.out.K) != cfg_.d_inner)
        fatal("%s: layer %d's Mamba projections disagree with config.json", path_.c_str(), i);
    } else if (L.kind == 'A') {
      // BF16 projections (Lightning's) stay BF16: as Q8_0 rows, decode is 3.5% faster but the KL to the fp32
      // reference grows from 2.8e-3 to 3.7e-3 (NI_NH_ATTN_Q8=1 quantizes them)
      mat(L.q, mx + "q_proj.weight", attn_q8);
      mat(L.k, mx + "k_proj.weight", attn_q8);
      mat(L.v, mx + "v_proj.weight", attn_q8);
      mat(L.o, mx + "o_proj.weight", attn_q8);
    } else if (L.kind == 'D') {
      // dense MLP: up / down in the shared expert's fields, as stored (BF16, NVFP4; FP8 and Q8 run too)
      mat(L.up_s, mx + "up_proj.weight");
      mat(L.down_s, mx + "down_proj.weight");
      const int I = cfg_.mlp_I[size_t(i)];
      if (int(L.up_s.N) != I || int(L.up_s.K) != cfg_.hidden || int(L.down_s.N) != cfg_.hidden || int(L.down_s.K) != I)
        fatal("%s: layer %d's MLP projections disagree with config.json", path_.c_str(), i);
      for (const NhMat* m : {&L.up_s, &L.down_s})
        if (m->fmt == NhFmt::IQ2 || m->fmt == NhFmt::F32 || m->K % 32)
          fatal("%s: layer %d's MLP projections are not BF16, NVFP4, FP8 or Q8_0 rows of whole 32s", path_.c_str(), i);
    } else {
      L.E = cfg_.n_exp[size_t(i)];
      L.k_used = cfg_.top_k[size_t(i)];
      L.I = cfg_.exp_I[size_t(i)];
      mat(L.router, mx + "gate.weight");
      vec(L.exp_bias, mx + "gate.e_score_correction_bias");
      // latent projections (LatentMoE); without them (moe_latent_size null: Lightning) the experts act on the hidden
      // state itself
      if (idx.find(mx + "fc1_latent_proj.weight") || cfg_.latent != cfg_.hidden) {
        mat(L.lat_down, mx + "fc1_latent_proj.weight", true);
        mat(L.lat_up, mx + "fc2_latent_proj.weight", true);
      }
      experts(L.up_e, mx + "experts.", L.E, "up_proj");
      experts(L.down_e, mx + "experts.", L.E, "down_proj");
      for (NhMat* m : {&L.up_e, &L.down_e}) m->l2 = experts_l2_ && m->fmt == NhFmt::FP4 && !L.lat_down;
      mat(L.up_s, mx + "shared_experts.up_proj.weight");
      mat(L.down_s, mx + "shared_experts.down_proj.weight");
      // (NVFP4 experts run on K % 32 designs: per-pair matvecs and the prefill GEMMs; Lightning's down K is 1856)
      if (!((L.up_e.fmt == NhFmt::IQ2 && L.down_e.fmt == NhFmt::IQ2) ||
            (L.up_e.fmt == NhFmt::FP4 && L.down_e.fmt == NhFmt::FP4 && L.up_e.K % 32 == 0 && L.down_e.K % 32 == 0)))
        fatal("%s: layer %d's experts are neither IQ2_XXS nor NVFP4 (K %% 32)", path_.c_str(), i);
      if (int(L.router.N) != L.E || int(L.up_e.N) != L.I || int(L.up_e.K) != cfg_.latent || int(L.down_e.K) != L.I)
        fatal("%s: layer %d's expert shapes disagree with config.json", path_.c_str(), i);
    }
  }
  if (embed_.fmt != NhFmt::BF16 && embed_.fmt != NhFmt::Q8)
    fatal("%s: the embedding is neither BF16 nor Q8_0", path_.c_str());
  // 2. place, and cut the reads into jobs of ~16 MB. Placement: every tensor's codes / scales / vector packed into
  // buffers of at most 1 GiB, biggest first, each into the first buffer with room, and each buffer allocated at the
  // size it holds (filling 1 GiB buffers in load order left ~3.4 GB of their tails unused but resident).
  const size_t kChunk = size_t(1) << 30, kJob = size_t(16) << 20;
  struct Slot {
    Buffer* dst;
    size_t bytes, chunk = 0, off = 0;
  };
  std::vector<Slot> slots;
  const bool map_embed = map_embed_;
  for (Want& w : wants) {
    if (w.m == &embed_ && embed_.fmt == NhFmt::BF16 && map_embed) continue;
    if (w.vec) {
      slots.push_back({w.vec, size_t(w.parts[0]->numel()) * 4});
      continue;
    }
    NhMat& m = *w.m;
    slots.push_back({&m.q, m.q_row * m.N * m.E});
    if (m.s_row) slots.push_back({&m.s, m.s_row * m.N * m.E});
    if (!w.gse.empty()) slots.push_back({&m.gse, w.gse.size() * 4});
  }
  for (Slot& sl : slots) sl.bytes = align_up(std::max<size_t>(sl.bytes, 16), 256);
  {
    std::vector<size_t> order(slots.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return slots[a].bytes > slots[b].bytes; });
    std::vector<size_t> used;  // per buffer
    for (size_t i : order) {
      Slot& sl = slots[i];
      size_t c = 0;
      while (c < used.size() && (used[c] + sl.bytes > std::max(kChunk, sl.bytes) || (sl.bytes > kChunk && used[c])))
        ++c;
      if (c == used.size()) used.push_back(0);
      sl.chunk = c;
      sl.off = used[c];
      used[c] += sl.bytes;
    }
    for (size_t c = 0; c < used.size(); ++c) arena_.push_back(g.alloc(used[c], "nh_weights", false));
    for (Slot& sl : slots) *sl.dst = arena_[sl.chunk].view(sl.off, sl.bytes);
  }
  std::vector<Job> jobs;
  uint64_t src_bytes = 0;
  for (Want& w : wants) {
    if (w.m == &embed_ && embed_.fmt == NhFmt::BF16 && map_embed_) {
      // the embedding is mapped from its file (no copy): only the rows a conversation uses become resident, as clean
      // file pages; NI_NH_MAP_EMBED=0 reads it into GPU memory with the rest
      const SafetensorsIndex::Entry& t = *w.parts[0];
      const size_t page = size_t(getpagesize()), off0 = t.off / page * page;
      const size_t len = (t.off + t.len - off0 + page - 1) / page * page;
      const int fd = open(idx.path(t).c_str(), O_RDONLY);
      if (fd < 0) fatal("open %s failed", idx.path(t).c_str());
      // shared and read-only: a private mapping's pages count as dirty in the process footprint once Metal wires them
      void* p = mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, off_t(off0));
      close(fd);
      if (p == MAP_FAILED) fatal("mapping the embedding of %s failed", idx.path(t).c_str());
      // read ahead in the background: the first prompt then finds its rows in the page cache (clean file pages,
      // outside the process footprint)
      madvise(p, len, MADV_WILLNEED);
      embed_map_ = p;
      embed_map_len_ = len;
      Buffer b = g.wrap(p, len, "nh_embedding (mapped)", false);
      embed_.q = embed_.s = b.view(t.off - off0, t.len);
      continue;
    }
    if (w.vec) {
      const SafetensorsIndex::Entry& t = *w.parts[0];
      Job j;
      j.t = &t;
      j.kind = w.vk;
      j.len = t.len;
      j.q = w.vec->as<uint8_t>();
      jobs.push_back(j);
      src_bytes += t.len;
      weight_bytes_ += w.vec->size;
      continue;
    }
    NhMat& m = *w.m;
    if (!m.s_row) m.s = m.q;
    weight_bytes_ += m.q.size + (m.s_row ? m.s.size : 0);
    if (!w.gse.empty()) memcpy(m.gse.ptr(), w.gse.data(), w.gse.size() * 4);
    const size_t rb = nh_src_row_bytes(w.q8 ? NhFmt::BF16 : m.fmt, m.K);
    const int64_t per = std::max<int64_t>(1, int64_t(kJob / std::max<size_t>(rb, 1)));
    for (size_t e = 0; e < w.parts.size(); ++e) {
      const SafetensorsIndex::Entry& t = *w.parts[e];
      if (t.len != rb * m.N) fatal("%s: a tensor's size disagrees with its shape", path_.c_str());
      for (int64_t r0 = 0; r0 < int64_t(m.N); r0 += per) {
        Job j;
        j.t = &t;
        j.kind = w.q8 ? Job::Bf16ToQ8 : Job::Rows;
        j.fmt = m.fmt;
        j.K = m.K;
        j.rows = std::min<int64_t>(per, int64_t(m.N) - r0);
        j.src = rb * size_t(r0);
        j.len = rb * size_t(j.rows);
        const size_t row = e * m.N + size_t(r0);
        j.q = m.q.as<uint8_t>() + m.q_row * row;
        j.s = m.s_row && m.fmt != NhFmt::FP4 ? m.s.as<uint8_t>() + m.s_row * row : nullptr;
        j.l2 = m.l2;
        jobs.push_back(j);
      }
      src_bytes += t.len;
    }
    for (size_t e = 0; e < w.scales.size(); ++e) {  // NVFP4 scales: the E4M3 bytes as they are, per part
      const SafetensorsIndex::Entry& t = *w.scales[e];
      if (t.len != m.s_row * m.N) fatal("%s: NVFP4 scales of the wrong size", path_.c_str());
      for (size_t o = 0; o < t.len; o += kJob) {
        Job j;
        j.t = &t;
        j.kind = Job::Copy;
        j.src = o;
        j.len = std::min(kJob, t.len - o);
        j.q = m.s.as<uint8_t>() + e * m.s_exp() + o;
        jobs.push_back(j);
      }
      src_bytes += t.len;
    }
  }
  if (mem_budget_ && g.allocated_bytes() > mem_budget_)
    fatal("insufficient memory: the weights of %s take %.1f GB", path_.c_str(), g.allocated_bytes() / 1e9);
  // 3. read and convert in parallel (biggest first, so the tail is short)
  std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) { return a.len > b.len; });
  std::atomic<size_t> next{0};
  const unsigned nth = std::max(2u, std::min(10u, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  for (unsigned i = 0; i < nth; ++i)
    th.emplace_back([&] {
      std::vector<uint8_t> raw;
      for (size_t k; (k = next.fetch_add(1)) < jobs.size();) run_job(idx, jobs[k], raw);
    });
  for (auto& t : th) t.join();
  g.make_resident(arena_);
  alloc_buffers();
  const int n_d = int(std::count(cfg_.kind.begin(), cfg_.kind.end(), 'D'));
  // (layout L2 is chosen before the weights are read; only matrices loaded as NVFP4 use it: none in a BF16 checkpoint)
  const bool any_l2 = std::any_of(layers_.begin(), layers_.end(), [](const NhLayer& L) {
    return L.in.l2 || L.out.l2 || L.q.l2 || L.k.l2 || L.v.l2 || L.o.l2 || L.up_s.l2 || L.down_s.l2;
  });
  const bool experts_l2 = std::any_of(layers_.begin(), layers_.end(), [](const NhLayer& L) { return L.up_e.l2; });
  logi("Nemotron-H: %d layers (%d Mamba-2, %d attention, %d %s), %.2f GB of weights read in %.1f s (%s)%s", cfg_.layers,
       int(std::count(cfg_.kind.begin(), cfg_.kind.end(), 'M')), int(std::count(cfg_.kind.begin(), cfg_.kind.end(), 'A')),
       n_d ? n_d : int(std::count(cfg_.kind.begin(), cfg_.kind.end(), 'E')), n_d ? "MLP" : "MoE", src_bytes / 1e9,
       (now_ms() - t0) / 1000, formats_summary().c_str(),
       any_l2 ? "; NVFP4 codes in layout L2" : experts_l2 ? "; NVFP4 experts in layout L2" : "");
}

namespace {
const char* fmt_name(NhFmt f) {
  switch (f) {
    case NhFmt::Q8: return "Q8_0";
    case NhFmt::FP4: return "NVFP4";
    case NhFmt::IQ2: return "IQ2_XXS";
    case NhFmt::BF16: return "BF16";
    case NhFmt::F32: return "F32";
    case NhFmt::FP8: return "FP8";
  }
  return "?";
}
}  // namespace

std::string NemotronH::formats_summary() const {
  if (dense_model(cfg_)) {
    // each part's formats with their layer counts where they differ, e.g. the Nano 9B v2's "Mamba NVFP4 25 / BF16 2,
    // MLP NVFP4 23 / BF16 2, attention BF16, Q8_0 head, BF16 embedding"
    std::string r;
    for (auto [kind, part] : {std::pair<char, const char*>{'M', "Mamba"}, {'D', "MLP"}, {'A', "attention"}}) {
      std::vector<std::pair<NhFmt, int>> f;
      for (const NhLayer& L : layers_) {
        if (L.kind != kind) continue;
        const NhFmt x = kind == 'M' ? L.in.fmt : kind == 'D' ? L.up_s.fmt : L.q.fmt;
        // (an init-capture: Clang 15 cannot capture a structured binding)
        auto it = std::find_if(f.begin(), f.end(), [x = x](auto& y) { return y.first == x; });
        if (it == f.end()) f.push_back({x, 1});
        else ++it->second;
      }
      if (f.empty()) continue;
      r += std::string(r.empty() ? "" : ", ") + part + " ";
      for (size_t i = 0; i < f.size(); ++i)
        r += std::string(i ? " / " : "") + fmt_name(f[i].first) + (f.size() > 1 ? " " + std::to_string(f[i].second) : "");
    }
    return r + ", " + fmt_name(head_.fmt) + " head, " + fmt_name(embed_.fmt) + " embedding";
  }
  // the format of each part (first layer of its kind), e.g. Bolt's "IQ2_XXS experts, NVFP4 Mamba, Q8_0 attention /
  // latent / head, BF16 embedding / router"
  NhFmt ex = NhFmt::F32, sh = NhFmt::F32, mb = NhFmt::F32, at = NhFmt::F32, lt = NhFmt::F32, rt = NhFmt::F32;
  bool has_lt = false;
  for (const NhLayer& L : layers_) {
    if (L.kind == 'M') mb = L.in.fmt;
    if (L.kind == 'A') at = L.q.fmt;
    if (L.kind == 'E') ex = L.up_e.fmt, sh = L.up_s.fmt, lt = L.lat_down.fmt, rt = L.router.fmt, has_lt = bool(L.lat_down);
  }
  std::string r = std::string(fmt_name(ex)) + " experts";
  if (sh != ex) r += std::string(", ") + fmt_name(sh) + " shared expert";
  r += std::string(", ") + fmt_name(mb) + " Mamba";
  // attention, latent and head grouped by format
  std::vector<std::pair<NhFmt, std::string>> g;
  for (auto [f, n] : {std::pair<NhFmt, const char*>{at, "attention"}, {lt, "latent"}, {head_.fmt, "head"}}) {
    if (!has_lt && !strcmp(n, "latent")) continue;
    // an init-capture: Clang 15 (the Command Line Tools on macOS 14) cannot capture a structured binding
    auto it = std::find_if(g.begin(), g.end(), [f = f](auto& x) { return x.first == f; });
    if (it == g.end()) g.push_back({f, n});
    else it->second += std::string(" / ") + n;
  }
  for (auto& [f, n] : g) r += std::string(", ") + fmt_name(f) + " " + n;
  r += std::string(", ") + fmt_name(embed_.fmt) + " embedding" + (rt == embed_.fmt ? " / router" : std::string(", ") + fmt_name(rt) + " router");
  return r;
}

void NemotronH::alloc_buffers() {
  Gpu& g = Gpu::get();
  const size_t R = kMaxRows, H = size_t(cfg_.hidden);
  x_ = g.alloc(R * H * 4, "nh_x");
  xn_ = g.alloc(R * H * 4, "nh_xn");
  zx_ = g.alloc(R * size_t(cfg_.in_proj()) * 4, "nh_zx");
  y_ = g.alloc(R * size_t(cfg_.d_inner) * 4, "nh_y");
  yn_ = g.alloc(R * size_t(cfg_.d_inner) * 4, "nh_yn");
  q_ = g.alloc(R * size_t(std::max(1, cfg_.hq * cfg_.hd)) * 4, "nh_q");
  k_ = g.alloc(R * size_t(std::max(1, cfg_.hkv * cfg_.hd)) * 4, "nh_k");
  v_ = g.alloc(R * size_t(std::max(1, cfg_.hkv * cfg_.hd)) * 4, "nh_v");
  ao_ = g.alloc(R * size_t(std::max(1, cfg_.hq * cfg_.hd)) * 4, "nh_ao");
  part_o_ = g.alloc(R * size_t(std::max(1, cfg_.hq)) * attn_max_chunks_ * size_t(cfg_.hd) * 4, "nh_part_o");
  part_ml_ = g.alloc(R * size_t(std::max(1, cfg_.hq)) * attn_max_chunks_ * 2 * 4, "nh_part_ml");
  rl_ = g.alloc(R * size_t(std::max(1, cfg_.max_E)) * 4, "nh_router");
  eid_ = g.alloc(R * size_t(std::max(1, cfg_.max_k)) * 4, "nh_eid");
  ew_ = g.alloc(R * size_t(std::max(1, cfg_.max_k)) * 4, "nh_ew");
  lat_ = g.alloc(R * size_t(std::max(1, cfg_.latent)) * 4, "nh_lat");
  h_ = g.alloc(R * size_t(std::max(1, cfg_.max_k)) * size_t(std::max(1, cfg_.max_I)) * 4, "nh_h");
  lo_ = g.alloc(R * size_t(std::max(1, cfg_.latent)) * 4, "nh_lo");
  part_ = g.alloc(R * size_t(std::max(1, cfg_.max_k)) * size_t(std::max(1, cfg_.latent)) * 4, "nh_moe_part");
  sh_ = g.alloc(R * size_t(std::max(1, cfg_.ffn_I())) * 4, "nh_sh");  // (the shared expert's or a dense MLP's)
  logits_ = g.alloc(R * size_t(cfg_.vocab) * 4, "nh_logits");
  const size_t nchunk1 = (size_t(cfg_.vocab) + 2047) / 2048;
  cand_v_ = g.alloc(size_t(kMaxRows) * nchunk1 * 64 * 4, "nh_cand_v");  // (a verify's rows sample together)
  cand_i_ = g.alloc(size_t(kMaxRows) * nchunk1 * 64 * 4, "nh_cand_i");
  sprob_ = g.alloc(64 * 4, "nh_sprob");
  samp_ = g.alloc(64 * 4, "nh_samples");
  xnh_ = g.alloc(R * std::max(H, size_t(cfg_.d_inner)) * 2, "nh_xnh");
  aoh_ = g.alloc(R * size_t(std::max(1, cfg_.hq * cfg_.hd)) * 2, "nh_aoh");
  loh_ = g.alloc(R * size_t(std::max(1, cfg_.latent)) * 2, "nh_loh");
  shh_ = g.alloc(R * size_t(std::max(1, cfg_.ffn_I())) * 2, "nh_shh");
  if (const char* v = getenv("NI_NH_ROWS_GEMM")) gemm_rows_ = atoi(v) != 0;
  if (const char* v = getenv("NI_NH_V8")) v8_ = atoi(v) != 0;
  // attention's keys per chunk (threadgroup) before the 64-chunk cap doubles it: on the M5 64 (Lightning's verify at
  // ~600 positions: partial + reduce 27.0 against 36.8 us per layer with 256, NI_PROFILE; 32: 26.9)
  attn_chunk_ = Gpu::get().apple10() ? 64 : 256;
  if (const char* v = getenv("NI_NH_ATTN_CHUNK")) attn_chunk_ = std::max(32, atoi(v));
  if (const char* v = getenv("NI_NH_MERGE_FLUSH")) merge_flush_ = atoi(v) != 0;
  acc_ = g.alloc(64, "nh_accepted");
  tree_ = g.alloc(kf::kWords * 4, "nh_kf_tree");
  memset(tree_.ptr(), 0, tree_.size);
  route_ctr_ = g.alloc(16, "nh_route_counter");
  memset(route_ctr_.ptr(), 0, 16);
  moecnt_ = g.alloc(size_t(kMaxRows) * ((size_t(std::max(1, cfg_.latent)) + 31) / 32) * 4 + 16, "nh_moe_counters");
  memset(moecnt_.ptr(), 0, moecnt_.size);
  sd_ = g.alloc(size_t(cfg_.hidden) * 4, "nh_shared_down");
  if (const char* v = getenv("NI_NH_FUSE")) fuse_dec_ = atoi(v) != 0;
  if (split_) fuse_dec_ = false;  // (its routing merges in the last router threadgroup)
  if (const char* v = getenv("NI_NH_ROUTE_BUCKET")) route_bucket_ = atoi(v) != 0;
  // a verify's IQ2 routed experts' down projection: 4 simdgroups per threadgroup on the M5 (16 rows of a tile; tools/
  // nh_x2moe_bench, a 5-row harness verify's 29 experts: 52.0 against 59.1 us with 1, 52.8 with 2, 52.0 with 8; the
  // up projection 46.4-46.6 with 1-4: left at 1), the same outputs to the bit. NI_NH_X2_SGD (1-8) overrides.
  x2_sgd_ = g.apple10() ? 4u : 1u;
  if (const char* v = getenv("NI_NH_X2_SGD")) x2_sgd_ = uint32_t(std::clamp(atoi(v), 1, 8));
  // a verify's NVFP4 routed experts without latent projections (Lightning): 4 simdgroups per threadgroup on the M5
  // (tools/nh_moe_bench: up + down + sum 223 against 230 us per layer with 8; the same outputs to the bit)
  moe_thr_ = g.apple10() ? 128u : 256u;
  if (const char* v = getenv("NI_NH_MOE_THR")) moe_thr_ = std::clamp(uint32_t(atoi(v)) / 32u * 32u, 32u, 1024u);
  if (const char* v = getenv("NI_NH_GROUP")) group_ = atoi(v) != 0;
  // the M5's verify (Lightning): router and routing beside the shared expert (enc_moe)
  rsu_ = true;
  if (const char* v = getenv("NI_NH_RSU")) rsu_ = atoi(v) != 0;
  // the routed experts' indirect grids of a dynamic verify (NVFP4 experts, no latent projections, one shape); on the
  // M5, where it was measured
  moe_args_ = g.alloc(32, "nh_moe_pair_args");
  memset(moe_args_.ptr(), 0, 32);
  pair_ind_ = g.apple10();
  if (const char* v = getenv("NI_NH_PAIR_IND")) pair_ind_ = atoi(v) != 0;
  moe_lightning_ = true;
  for (const NhLayer& L : layers_) {
    if (L.kind != 'E') continue;
    if (L.lat_up || L.up_e.fmt != NhFmt::FP4 || L.k_used <= 0 ||
        (moe_I_ && (uint32_t(L.I) != moe_I_ || uint32_t(L.k_used) != moe_k_)))
      moe_lightning_ = false;
    moe_I_ = uint32_t(L.I);
    moe_k_ = uint32_t(L.k_used);
  }
  if (!moe_I_) moe_lightning_ = false;
  sample_sg_ = g.apple10();
  if (const char* v = getenv("NI_NH_SAMPLE_SG")) sample_sg_ = atoi(v) != 0;
  sdv_ = g.alloc(size_t(kMaxRows) * cfg_.hidden * 4, "nh_shared_down_rows");
  // the M5: a verify's 8-row FP8 / BF16 products and the drafter's Q8 at 8 weight rows per threadgroup (nh_*_v8_n1),
  // twice the threadgroups of 16 (each row's sums as before)
  v8_n1_ = g.apple10();
  if (const char* v = getenv("NI_NH_V8_N1")) v8_n1_ = atoi(v) != 0;
  live_ = g.alloc(16, "nh_live_rows");
  live_all_ = g.alloc(16, "nh_live_all");
  live_.as<uint32_t>()[0] = 0xffffffffu;
  live_all_.as<uint32_t>()[0] = 0xffffffffu;
  // NVFP4 9-32-row kernels split K in 4 (K <= 8192) or 8 parts: partials of the largest FP4 matrix (in_proj) at 32
  // rows, and one arrival counter per 64-row tile (zero now; every call leaves them zero)
  if (const char* v = getenv("NI_NH_X4")) x4_ = atoi(v) != 0;
  if (const char* v = getenv("NI_NH_X8")) x8_ = atoi(v) != 0;
  x4part_ = g.alloc(size_t(4) * ((size_t(cfg_.in_proj()) + 63) / 64) * 2048 * 4, "nh_x4_partials");
  x4cnt_ = g.alloc(4096, "nh_x4_counters");
  memset(x4cnt_.ptr(), 0, 4096);
  if (const char* v = getenv("NI_NH_X2")) x2_ = atoi(v) != 0;
  tg_ = [] {
    if (const char* v = getenv("NI_NH_TG"); v && !atoi(v)) return false;
    if (Gpu::get().apple_family() < 10) return false;
    if (__builtin_available(macOS 26.0, *)) {
      Gpu::get().compile_extra("tensor", kTensorKernelSource, true);
      return Gpu::get().pipeline("nh_gemm_t_fp8_m64", false).pso != nullptr;
    }
    return false;
  }();
  {
    const size_t P = size_t(kMaxRows) * size_t(std::max(1, cfg_.max_k));  // (row, expert) pairs of a verify block
    x2sorted_ = g.alloc(P * 4, "nh_x2_sorted");
    x2tiles_ = g.alloc(P * 16, "nh_x2_tiles");
    x2args_ = g.alloc(64, "nh_x2_dispatch_args");
    x2rank_ = g.alloc(P * 4, "nh_x2_rank");
  }
  // the IQ2_XXS grid as halves (nh_iq2_fill_tables2), read by the 8-row and GEMM kernels from the constant space
  iq2tab_ = g.alloc(4096, "nh_iq2_grid_half");
  {
    auto e = g.begin();
    e->use(pipe("nh_iq2_fill_tables2"));
    e->buf(0, iq2tab_);
    e->dispatch(1, 1, 1, 256);
    e->commit_and_wait();
  }
}

int NemotronH::wide_chunk(const std::string& path) {
  // Lightning (nh_eval, 4K / 16K-token prompts, M5 Max): 752 / 3048 against 790 / 3198 ms with 2048-row chunks, 8192
  // rows 3113 at 16K; NI_NH_WIDE_CHUNK=0: 2048. The MoE models whose prefill stays on the GPU alone (Lightning).
  if (const char* v = getenv("NI_NH_WIDE_CHUNK"); v && !atoi(v)) return 0;
  if (const char* v = getenv("NI_NH_TG"); v && !atoi(v)) return 0;
  if (Gpu::get().apple_family() < 10) return 0;
  if (__builtin_available(macOS 26.0, *)) {
    // not where the Neural Engine split runs the prefill (Bolt, Super-VL): it takes chunks of up to 2048 rows
    const NhConfig c = read_config(path);
    return dense_model(c) || ane_plan_bytes(c, 0) ? 0 : 4096;
  }
  return 0;
}

MemPlan NemotronH::plan(const std::string& path, int prefill_chunk) {
  MemPlan m;
  if (!is_model(path)) return m;
  const NhConfig c = read_config(path);
  const SafetensorsIndex idx(path, not_lm);
  m.hidden = c.hidden;
  m.layers = c.layers;
  const bool keep_bf16 = getenv("NI_NH_BF16") && atoi(getenv("NI_NH_BF16"));
  const bool attn_q8 = getenv("NI_NH_ATTN_Q8") && atoi(getenv("NI_NH_ATTN_Q8"));
  // a dense NVFP4 or FP8 model's BF16 LM head becomes Q8_0 rows at load (head_q8)
  const std::string pre = idx.find("language_model.backbone.embeddings.weight") ? "language_model." : "";
  const bool head8 = head_q8(c, idx, pre);
  std::string fmts;  // the matrix formats present (in tensor-name order)
  bool ane_fmts = true;  // the Neural Engine split stages NVFP4 / FP8 Mamba projections, an IQ2_XXS / NVFP4 shared expert
  for (const auto& [name, t] : idx.all()) {
    // as loaded: GGUF blocks, NVFP4 and FP8 keep their size (codes and scales apart), BF16 vectors become fp32, BF16
    // latent projections (and with NI_NH_ATTN_Q8 attention projections) Q8_0 (34 bytes per 32)
    const bool vector = t.shape.size() == 1 || name.find("conv1d") != std::string::npos;
    const bool embedding = name.find("embeddings.") != std::string::npos;
    // the BF16 embedding is mapped from its file (NI_NH_MAP_EMBED): clean file pages, the rows in use
    const bool mapped = embedding && t.dtype == DType::BF16 && !(getenv("NI_NH_MAP_EMBED") && !atoi(getenv("NI_NH_MAP_EMBED")));
    const bool to_q8 = !keep_bf16 && t.dtype == DType::BF16 && t.shape.size() == 2 && t.shape[1] % 32 == 0 &&
                       (name.find("latent_proj.weight") != std::string::npos || (head8 && name == pre + "lm_head.weight") ||
                        (attn_q8 && name.find("mixer.") != std::string::npos && name.find("_proj.weight") != std::string::npos &&
                         (name.find(".q_proj.") != std::string::npos || name.find(".k_proj.") != std::string::npos ||
                          name.find(".v_proj.") != std::string::npos || name.find(".o_proj.") != std::string::npos)));
    const size_t loaded = to_q8 ? t.len / 2 * 34 / 32 : t.dtype == DType::BF16 && vector ? 2 * t.len : t.len;
    if (!mapped) m.weights += loaded;
    if (embedding) continue;  // one row per token
    const size_t lp = name.find("layers.");
    const int li = lp == std::string::npos ? -1 : std::atoi(name.c_str() + lp + 7);
    NhFmt f;
    uint32_t N = 0, K = 0;
    const size_t dot = name.rfind(".weight");
    const SafetensorsIndex::Entry* sc =
        dot != std::string::npos && dot + 7 == name.size() ? idx.find(name.substr(0, dot) + ".weight_scale") : nullptr;
    const bool matrix = nh_describe(t, sc, f, N, K) && name.find("weight_scale") == std::string::npos;
    if (matrix && t.shape.size() >= 2 && !vector) {
      const NhFmt lf = to_q8 ? NhFmt::Q8 : f;
      if (fmts.find(fmt_name(lf)) == std::string::npos) fmts += std::string(fmts.empty() ? "" : " / ") + fmt_name(lf);
      if ((name.find("mixer.in_proj.") != std::string::npos || name.find("mixer.out_proj.") != std::string::npos) &&
          f != NhFmt::FP4 && f != NhFmt::FP8)
        ane_fmts = false;
      if (name.find("shared_experts.") != std::string::npos && f != NhFmt::IQ2 && f != NhFmt::FP4) ane_fmts = false;
    }
    if (name.find(".experts.") != std::string::npos && li >= 0 && c.n_exp[size_t(li)] > 0) {
      const double used = double(c.top_k[size_t(li)]) / c.n_exp[size_t(li)];  // each expert: k / E of the tokens
      m.decode_bytes += size_t(double(t.len) * used);
      if (matrix) m.decode_params += size_t(double(N) * K * used);
    } else {
      m.decode_bytes += to_q8 ? loaded : t.len;
      if (matrix) m.decode_params += size_t(N) * K;
    }
    if (name.find("lm_head.") != std::string::npos && matrix) {
      size_t qb = 0, sb = 0;
      nh_row_bytes(to_q8 ? NhFmt::Q8 : f, K, qb, sb);
      m.head_row_bytes = qb + sb;
    }
  }
  const int nm = int(std::count(c.kind.begin(), c.kind.end(), 'M'));
  const int na = int(std::count(c.kind.begin(), c.kind.end(), 'A'));
  const size_t state = size_t(nm) * (size_t(c.n_head) * c.head_dim * c.d_state * 4 + 4 * size_t(c.conv_ch()) * 4);
  const size_t pend = size_t(nm) * kMaxRows * size_t(c.in_proj()) * 4;
  m.slot_state = 3 * state + pend;  // live state + two turn-boundary checkpoints, pending verify rows
  m.kv_per_pos = size_t(na) * 2 * c.hkv * c.hd * 2;
  m.activations = size_t(kMaxRows) * (size_t(c.in_proj()) + 4 * size_t(c.hidden) + 2 * size_t(c.d_inner) +
                                      size_t(c.max_k) * c.max_I + size_t(c.vocab)) * 4 + (size_t(64) << 20);
  // prefill buffers (alloc_prefill) for chunks of prefill_chunk rows
  const size_t R = size_t(std::max(kMaxRows, prefill_chunk)), P = R * size_t(std::max(1, c.max_k));
  const size_t H = size_t(c.hidden), DI = size_t(c.d_inner), QD = size_t(c.hq) * c.hd, KD = size_t(c.hkv) * c.hd;
  m.activations += R * (2 * H * 4 + std::max(H, DI) * 2 + size_t(c.in_proj()) * 4 + DI * 6 + QD * 6 + KD * 8 +
                        size_t(c.max_E) * 4 + size_t(c.latent) * 4 + size_t(c.ffn_I()) * 2) +
                   P * (16 + size_t(c.max_I) * 2 + size_t(c.latent) * 4) + (P / 16 + size_t(c.max_E) + 2) * 16;
  // the M5's chunked scan (nh_ssd, head dim 64, state 128): its conv rows and per-chunk states in nh_pf_out when they
  // fit, else in buffers of their own (alloc_prefill); counted on every Mac
  if (c.head_dim == 64 && c.d_state == 128) {
    const size_t cvb = R * (size_t(c.conv_ch()) + 2 * size_t(c.n_head)) * 4, nch = (R + 31) / 32;
    const size_t sbb = size_t(c.n_head) * nch * (32 * 32 + 64) * 4, pout = P * size_t(std::max(1, c.latent)) * 4;
    m.activations += (pout >= cvb ? 0 : cvb) + (pout >= (cvb + 255) / 256 * 256 + sbb ? 0 : sbb);
  }
  m.activations += size_t(4) * ((size_t(c.in_proj()) + 63) / 64) * 2048 * 4 + 4096;  // x4 split-K partials, counters
  m.activations += size_t(kMaxRows) * c.hidden * 4;  // the shared expert's down rows of a verify (sdv_, every Mac)
  if (!m.head_row_bytes) m.head_row_bytes = size_t(c.hidden) + size_t(c.hidden / 32) * 2;
  m.dense = dense_model(c);  // (Nemotron Nano: MLP layers, no experts; the doctor's label)
  m.ffn_ane = false;
  m.formats = fmts;
  m.ane = ane_fmts ? ane_plan_bytes(c, state) : 0;
  return m;
}

// The GPU + Neural Engine split (on unless NI_ANE=0): its surfaces at the default shares (int8 inputs and weights
// per 2048-channel segment, fp16 outputs for 2048 rows), the rotation scratch, and the recurrent-state checkpoint the
// engine keeps to rerun a chunk on the GPU.
size_t NemotronH::ane_plan_bytes(const NhConfig& c, size_t state) {
  if (const char* v = getenv("NI_ANE"); v && !atoi(v)) return 0;
  if (c.hidden % 2048 || c.d_inner % 2048) return 0;  // (NhAne needs whole 2048-channel segments)
  const size_t R = 2048, H = size_t(c.hidden), DI = size_t(c.d_inner), NIN = size_t(c.in_proj()), I = size_t(c.shared_I);
  const size_t ain = NIN * 7 / 10, aout = H * 7 / 10;
  const size_t surfaces = H * R + DI * R + ain * H + ain * R * 2 + aout * DI + aout * R * 2 + I * H + H * I + (H + 1) * R * 2;
  return surfaces + R * std::max(H, DI) * 2 + state;
}

// ---- sequences, KV caches, checkpoints ----
std::unique_ptr<Seq> NemotronH::new_seq(int limit, int initial) {
  Gpu& g = Gpu::get();
  auto s = std::make_unique<Seq>();
  s->limit = limit;
  s->tok = g.alloc(size_t(limit + 8) * 4, "tokens");
  const size_t CH = size_t(cfg_.conv_ch());
  for (int i = 0; i < cfg_.layers; ++i) {
    s->kc.push_back(Buffer{});
    s->vc.push_back(Buffer{});
    if (cfg_.kind[size_t(i)] == 'M') {
      s->conv.push_back(g.alloc(4 * CH * 4, "nh_conv_ring"));
      s->rec.push_back(g.alloc(size_t(cfg_.n_head) * cfg_.head_dim * cfg_.d_state * 4, "nh_ssm_state"));
      s->pend_qkv.push_back(g.alloc(2 * size_t(kMaxRows) * cfg_.in_proj() * 4, "nh_pend_zx"));  // 2 halves
    } else {
      s->conv.push_back(Buffer{});
      s->rec.push_back(Buffer{});
      s->pend_qkv.push_back(Buffer{});
    }
  }
  s->pend_path = g.alloc(size_t(kf::kPathWords) * 4, "nh_pend_path");  // (KFlash tree verifies: the accepted path)
  const int cap = initial < 0 ? limit : std::min(initial, limit);
  if (cap > 0 && !reserve(*s, cap)) fatal("insufficient memory for a %d-token KV cache", cap);
  reset(*s);
  return s;
}

size_t NemotronH::kv_bytes_per_pos() const {
  const int na = int(std::count(cfg_.kind.begin(), cfg_.kind.end(), 'A'));
  return size_t(na) * 2 * cfg_.hkv * cfg_.hd * 2;
}

bool NemotronH::reserve(Seq& s, int need) {
  if (need <= s.cap) return true;
  if (need > s.limit) return false;
  Gpu& g = Gpu::get();
  const size_t per = kv_bytes_per_pos();
  auto round_up = [](int v, int m) { return (v + m - 1) / m * m; };
  auto fits = [&](int c) {
    const size_t copy = std::min(size_t(256) << 20, size_t(s.cap) * per);
    return !mem_budget_ || g.allocated_bytes() + size_t(c - s.cap) * per + copy <= mem_budget_;
  };
  int target = std::min(s.limit, round_up(std::max(need, s.cap + s.cap / 4), 1024));
  if (!fits(target)) target = std::min(s.limit, round_up(need, 256));
  if (!fits(target)) return false;
  const int old = s.cap;
  const size_t nb = size_t(cfg_.hkv) * size_t(target) * cfg_.hd * 2;
  for (int i = 0; i < cfg_.layers; ++i) {
    if (cfg_.kind[size_t(i)] != 'A') continue;
    for (Buffer* b : {&s.kc[size_t(i)], &s.vc[size_t(i)]}) {
      Buffer nbuf = g.alloc(nb, "nh_kv");
      if (old > 0 && *b) {
        const uint32_t n16 = uint32_t(cfg_.hd) * 2 / 16;
        const uint32_t p[4] = {uint32_t(old) * n16, uint32_t(target) * n16, uint32_t(cfg_.hkv), 0};
        auto e = g.begin();
        e->use(pipe("kv_regrow"));
        e->buf(0, *b);
        e->buf(1, nbuf);
        e->bytes(2, p, sizeof p);
        e->dispatch((p[0] + 255) / 256, uint32_t(cfg_.hkv), 1, 256);
        e->commit_and_wait();
      }
      g.release(*b);
      *b = nbuf;
    }
  }
  s.cap = target;
  return true;
}

void NemotronH::drop_kv(Seq& s) {
  Gpu& g = Gpu::get();
  for (auto* v : {&s.kc, &s.vc})
    for (auto& b : *v) g.release(b);
  s.cap = 0;
}

void NemotronH::reset(Seq& s) {
  s.n_pend = 0;
  s.pend_tree = false;
  for (auto& b : s.conv)
    if (b) memset(b.ptr(), 0, b.size);
  for (auto& b : s.rec)
    if (b) memset(b.ptr(), 0, b.size);
}

Checkpoint NemotronH::new_checkpoint() {
  Gpu& g = Gpu::get();
  Checkpoint c;
  for (int i = 0; i < cfg_.layers; ++i) {
    const bool m = cfg_.kind[size_t(i)] == 'M';
    c.conv.push_back(m ? g.alloc(4 * size_t(cfg_.conv_ch()) * 4, "nh_ckpt_ring") : Buffer{});
    c.rec.push_back(m ? g.alloc(size_t(cfg_.n_head) * cfg_.head_dim * cfg_.d_state * 4, "nh_ckpt_state") : Buffer{});
  }
  return c;
}

static void enc_copy(Encoder& e, const Buffer& src, const Buffer& dst) {
  const uint32_t n = uint32_t(std::min(src.size, dst.size) / 16);
  e.use(pipe("copy_u32"));
  e.buf(0, src);
  e.buf(1, dst);
  e.val(2, n);
  e.dispatch((n + 255) / 256, 1, 1, 256);
}

void NemotronH::save(Encoder& e, const Seq& s, Checkpoint& c, int pos) {
  if (s.n_pend) fatal("NemotronH::save with pending verify rows");
  for (int i = 0; i < cfg_.layers; ++i)
    if (cfg_.kind[size_t(i)] == 'M') {
      enc_copy(e, s.conv[size_t(i)], c.conv[size_t(i)]);
      enc_copy(e, s.rec[size_t(i)], c.rec[size_t(i)]);
    }
  c.pos = pos;
}

void NemotronH::restore(Encoder& e, Seq& s, const Checkpoint& c) {
  s.n_pend = 0;
  s.pend_tree = false;
  for (int i = 0; i < cfg_.layers; ++i)
    if (cfg_.kind[size_t(i)] == 'M') {
      enc_copy(e, c.conv[size_t(i)], s.conv[size_t(i)]);
      enc_copy(e, c.rec[size_t(i)], s.rec[size_t(i)]);
    }
}

// ---- encoders ----
Pipeline NemotronH::pipeline(const std::string& name) { return pipe(name); }

void NemotronH::enc_norm(Encoder& e, const Buffer& x, const Buffer& w, const Buffer& y, int R, int D, float eps) {
  const NormP p{uint32_t(D), eps < 0 ? cfg_.eps : eps, uint32_t(D), uint32_t(D)};
  e.use(pipe("nh_rmsnorm"));
  e.buf(0, x);
  e.buf(1, w);
  e.buf(2, y);
  e.val(3, p);
  e.dispatch(uint32_t(R), 1, 1, 1024);
}

void NemotronH::enc_mv(Encoder& e, const NhMat& W, const Buffer& x, const Buffer& y, int R, int epi, uint32_t xs,
                       uint32_t ys) {
  if (W.E != 1) fatal("enc_mv on an expert stack");
  // one FP8 row: the x8 kernel (4 rows per simdgroup, 2 simdgroups)
  if (R == 1 && W.fmt == NhFmt::FP8 && x8_ && W.K % 32 == 0 && (xs == 0 || xs == W.K) && (ys == 0 || ys == W.N)) {
    const MvP p{W.N, W.K, W.K, W.N, W.gs, 1};
    // (the M5: 2 rows per simdgroup, twice the threadgroups, each row's sums as before)
    const bool n2 = v8_n1_ && epi <= 1;
    e.use(pipe(epi == 2   ? "nh_x8_mv_relu2"
               : epi == 1 ? (n2 ? "nh_x8_mv_add_n2" : "nh_x8_mv_add")
                          : (n2 ? "nh_x8_mv_n2" : "nh_x8_mv")));
    e.buf(0, W.q);
    e.buf(1, W.q);
    e.buf(2, x);
    e.buf(3, y);
    e.val(4, p);
    e.dispatch(n2 ? (W.N + 3) / 4 : (W.N + 7) / 8, 1, 1, 64);
    return;
  }
  // one FP4 row: the x4 kernel (8 rows per 64-thread threadgroup; in_proj 248, out_proj 228 GB/s on fresh weights), in
  // the matrix's layout
  if (R == 1 && W.fmt == NhFmt::FP4 && x4_ && epi <= 2 && W.K % 32 == 0 && (xs == 0 || xs == W.K) &&
      (ys == 0 || ys == W.N)) {
    const MvP p{W.N, W.K, W.K, W.N, W.gs, 1};
    e.use(pipe(W.l2 ? (epi == 2 ? "nh_x4_mv_relu2" : epi ? "nh_x4_mv_add" : "nh_x4_mv")
                    : (epi == 2 ? "nh_x4_mv_l0_relu2" : epi ? "nh_x4_mv_l0_add" : "nh_x4_mv_l0")));
    e.buf(0, W.q);
    e.buf(1, W.s);
    e.buf(2, x);
    e.buf(3, y);
    e.val(4, p);
    e.dispatch((W.N + 7) / 8, 1, 1, 64);
    return;
  }
  // one IQ2 row: the x2 kernels (one simdgroup per 4 rows, 3 for the add; shared up 182, down 169 GB/s vs 104 / 95)
  if (R == 1 && W.fmt == NhFmt::IQ2 && x2_ && W.K % 256 == 0 && (xs == 0 || xs == W.K) && (ys == 0 || ys == W.N)) {
    const MvP p{W.N, W.K, W.K, W.N, W.gs, 1};
    const uint32_t nr = epi == 1 ? 3 : 4;
    e.use(pipe(epi == 2 ? "nh_x2_mv4_relu2" : epi == 1 ? "nh_x2_mv3_add" : "nh_x2_mv4"));
    e.buf(0, W.q);
    e.buf(1, W.s);
    e.buf(2, x);
    e.buf(3, y);
    e.val(4, p);
    e.dispatch((W.N + nr - 1) / nr, 1, 1, 32);
    return;
  }
  if (W.l2) fatal("NemotronH: NVFP4 rows in layout L2 need the x4 kernels (%d rows, K %u, epilogue %d)", R, W.K, epi);
  // one row: FP4 with K <= 4096 (Mamba in_proj) and IQ2 on the nh_mv1 kernels (the FP4 byte-pair table, the IQ2 grid
  // from the constant table); the rest on the generic kernels (FP4 out_proj, K 8192: nh_mv_fp4_r1 242 against 189 GB/s)
  if (R == 1 && ((W.fmt == NhFmt::FP4 && W.K <= 4096 && epi <= 1) || (W.fmt == NhFmt::IQ2 && epi <= 2)) &&
      (xs == 0 || xs == W.K) && (ys == 0 || ys == W.N)) {
    const bool fp4 = W.fmt == NhFmt::FP4;
    std::string name = fp4 ? "nh_mv1_fp4_n8" : "nh_mv1_iq2_c";
    if (epi == 1) name += "_add";
    if (epi == 2) name += "_relu2";
    const uint32_t rows_per_tg = fp4 ? 32 : 16;
    const MvP p{W.N, W.K, W.K, W.N, W.gs, 1};
    e.use(pipe(name));
    e.buf(0, W.q);
    e.buf(1, W.s);
    e.buf(2, x);
    e.buf(3, y);
    e.val(4, p);
    e.dispatch((W.N + rows_per_tg - 1) / rows_per_tg, 1, 1, 128);
    return;
  }
  // F32 (Lightning's router) for 2-8 rows: one threadgroup per output row over K (verify: 0.38 ms less per cycle)
  if (R >= 2 && W.fmt == NhFmt::F32 && epi == 0 && W.K % 4 == 0 && (xs ? xs : W.K) % 4 == 0) {
    const MvP p{W.N, W.K, xs ? xs : W.K, ys ? ys : W.N, W.gs, uint32_t(R)};
    e.use(pipe("nh_mv_f32_sk"));
    e.buf(0, W.q);
    e.buf(2, x);
    e.buf(3, y);
    e.val(4, p);
    e.dispatch(W.N, 1, 1, 256);
    return;
  }
  // (BF16 one row, the dense models' decode: nh_mv_bf16_r1 streams at 165-175 GB/s on an M1 Pro, the Nano 4B's shapes,
  // tools/nh_mlp_bench; one-row kernels with 4 or 8 rows per simdgroup and fma chains measured the same or 1-4% slower,
  // and NVFP4's x4 decode with 8 rows per simdgroup 1.5% slower end to end on the Nano 9B v2. For its prefill
  // nh_gemm_bf16w_m64 reaches 3.4 TFLOPS there, as wfmt.metal's qrf_f16 does; qrf's design with these epilogues 3.0-3.3.)
  const int rk = R <= 1 ? 1 : R <= 2 ? 2 : R <= 4 ? 4 : 8;
  const char* fm = W.fmt == NhFmt::Q8    ? "q8"
                   : W.fmt == NhFmt::FP4 ? "fp4"
                   : W.fmt == NhFmt::IQ2 ? "iq2"
                   : W.fmt == NhFmt::BF16 ? "bf16"
                   : W.fmt == NhFmt::FP8 ? "fp8"
                                          : "f32";
  // BF16 for 2-8 rows on the M5 (the router in verify): nh_mv_bf16_rt, the same outputs to the bit, 2 weight rows x 2
  // activation rows per simdgroup (Bolt's router 12.2 against 46 us per layer)
  if (W.fmt == NhFmt::BF16 && R >= 2 && epi == 0 && W.K % 256 == 0 && Gpu::get().apple10()) {
    const MvP p{W.N, W.K, xs ? xs : W.K, ys ? ys : W.N, W.gs, uint32_t(R)};
    e.use(pipe("nh_mv_bf16_rt"));
    e.buf(0, W.q);
    e.buf(2, x);
    e.buf(3, y);
    e.val(4, p);
    e.dispatch((W.N + 7) / 8, (uint32_t(R) + 1) / 2, 1, 128);
    return;
  }
  // (one BF16 row on the M5: 2 weight rows per simdgroup, the same sums)
  const bool n2 = R == 1 && W.fmt == NhFmt::BF16 && v8_n1_ && epi <= 1;
  const std::string name = std::string("nh_mv_") + fm + "_r" + std::to_string(rk) +
                           (epi == 1 ? "_add" : epi == 2 ? "_relu2" : "") + (n2 ? "_n2" : "");
  const uint32_t nr = n2 ? 2 : rk <= 2 ? 4 : rk == 4 ? 2 : 1, nsg = 4;
  const MvP p{W.N, W.K, xs ? xs : W.K, ys ? ys : W.N, W.gs, uint32_t(R)};
  e.use(pipe(name));
  e.buf(0, W.q);
  e.buf(1, W.s);
  e.buf(2, x);
  e.buf(3, y);
  e.val(4, p);
  e.dispatch((W.N + nr * nsg - 1) / (nr * nsg), 1, 1, 32 * nsg);
}

void NemotronH::enc_mv_rows(Encoder& e, const NhMat& W, const Buffer& x, const Buffer& y, int M, int epi) {
  for (int r0 = 0; r0 < M; r0 += kMaxRows) {
    const int R = std::min(kMaxRows, M - r0);
    enc_mv(e, W, x.view(size_t(r0) * W.K * 4, size_t(R) * W.K * 4), y.view(size_t(r0) * W.N * 4, size_t(R) * W.N * 4), R,
           epi);
  }
}

void NemotronH::enc_rows_half(Encoder& e, const Buffer& x, const Buffer& y, int R, int D) {
  const HalfP hp{uint32_t(D), 0u, cfg_.eps, uint32_t(D)};
  e.use(pipe("nh_rows_half"));
  e.buf(0, x);
  e.buf(1, x);
  e.buf(2, y);
  e.val(3, hp);
  e.dispatch(uint32_t(R), 1, 1, 1024);
}

void NemotronH::enc_norm_half(Encoder& e, const Buffer& x, const Buffer& w, const Buffer& yh, int R, int D) {
  const HalfP hp{uint32_t(D), 1u, cfg_.eps, uint32_t(D)};
  e.use(pipe("nh_rows_half"));
  e.buf(0, x);
  e.buf(1, w);
  e.buf(2, yh);
  e.val(3, hp);
  e.dispatch(uint32_t(R), 1, 1, 1024);
}

void NemotronH::enc_norm2(Encoder& e, const Buffer& x, const Buffer& w, const Buffer& y, const Buffer& yh, int R,
                          int D) {
  const NormP p{uint32_t(D), cfg_.eps, uint32_t(D), uint32_t(D)};
  e.use(pipe("nh_rmsnorm_h"));
  e.buf(0, x);
  e.buf(1, w);
  e.buf(2, y);
  e.val(3, p);
  e.buf(4, yh);
  e.dispatch(uint32_t(R), 1, 1, 1024);
}

void NemotronH::enc_embed(Encoder& e, const Buffer& tokens, int tok_off, int R, const Buffer& out, const Buffer* params,
                          size_t params_off) {
  const EmbedP ep{uint32_t(cfg_.hidden), uint32_t(tok_off)};
  e.use(pipe(embed_.fmt == NhFmt::BF16 ? "nh_embed_bf16" : "nh_embed_q8"));
  e.buf(0, embed_.q);
  e.buf(1, embed_.s);
  e.buf(2, tokens);
  e.buf(3, out);
  if (params) e.buf(4, *params, params_off);
  else e.val(4, ep);
  e.dispatch(uint32_t(R), 1, 1, 256);
}

// block-drafter capture: x's rows [0, R) sit at positions pos0..; the ones in the hook's window are copied
void NemotronH::enc_capture(Encoder& e, int layer, const Buffer& x, int pos0, int R) {
  if (!cap_ || layer >= int(cap_->slot.size()) || cap_->slot[size_t(layer)] < 0) return;
  const int lo = std::max(pos0, cap_->pos_base), hi = std::min(pos0 + R, cap_->pos_base + cap_->rows);
  if (hi <= lo) return;
  struct {
    uint32_t D, ld, slot, dst_row0, src_row0;
  } p{uint32_t(cfg_.hidden), cap_->ld, uint32_t(cap_->slot[size_t(layer)]), uint32_t(lo - cap_->pos_base),
      uint32_t(lo - pos0)};
  e.use(pipe("nh_capture"));
  e.buf(0, x);
  e.buf(1, cap_->feat);
  e.bytes(2, &p, sizeof p);
  e.dispatch(uint32_t(hi - lo), (uint32_t(cfg_.hidden) + 255) / 256, 1, 256);
}

void NemotronH::enc_ssm(Encoder& e, Seq& s, int li, const Buffer& zx, int T, int pos0, bool write, const Buffer& y,
                        const Buffer* zp, int np, int parts) {
  const NhLayer& L = layers_[size_t(li)];
  const SsmP p{uint32_t(T), uint32_t(pos0), write ? 1u : 0u, uint32_t(cfg_.in_proj()), uint32_t(cfg_.n_head),
               uint32_t(cfg_.head_dim), uint32_t(cfg_.d_state), uint32_t(cfg_.n_group)};
  // KFlash: a tree verify's rows (each row's state its parent's: nh_ssm1_tree), and pending rows that are the last tree
  // verify's accepted path (zp rows s.pend_path[0 .. np))
  const bool tree = tree_on_ && !write && T > 0;
  const uint32_t ptree = s.pend_tree && np > 0 ? 1u : 0u;
  // short passes (decode, verify, flush): one simdgroup per state row (NI_NH_SSM1=0: the head-per-threadgroup kernel)
  static const bool ssm1 = !getenv("NI_NH_SSM1") || atoi(getenv("NI_NH_SSM1")) != 0;
  const bool short_pass = (ssm1 || np > 0 || tree) && T <= kMaxRows && cfg_.head_dim % 8 == 0 && cfg_.d_state == 128;
  if (tree && !short_pass) fatal("NemotronH: a tree verify of %d rows needs the short-pass scan", T);
  // a dense model's short passes (head dim 80): nh_ssm1r5, 5 rows per simdgroup, so the group's B / C conv is staged
  // once per 40 state rows instead of 8, the same outputs to the bit (one decode row on an M1 Pro: the Nano 4B's 96
  // heads 62 against 110 us, the 9B v2's 128 heads 79 against 144; decoding 1.5% / 4% faster; 10 rows per simdgroup
  // 79 / 103 us). On the M5 the MoE models' too, at 4 rows per simdgroup (head dim 64: 32 rows, half a head, per
  // threadgroup): Lightning's 8-row verify 22.7 against 35.3 us per layer (NI_PROFILE; 2 rows 27.8, 8 rows 24.8).
  // NI_NH_SSM1R=0 / 2 / 4 / 5 / 8 / 10 picks nh_ssm1 / nh_ssm1r<n>.
  static const int rps_env = getenv("NI_NH_SSM1R") ? atoi(getenv("NI_NH_SSM1R")) : -1;
  const int rps_want = rps_env >= 0 ? rps_env : dense_ ? 5 : Gpu::get().apple10() ? 4 : 0;
  const int rps = short_pass && !tree && (rps_want == 2 || rps_want == 4 || rps_want == 5 || rps_want == 8 || rps_want == 10) &&
                          cfg_.head_dim % (8 * rps_want) == 0
                      ? rps_want
                      : 0;
  // M5 (tg_), head dim 64, state 128, a pass of 33+ rows: the scan in chunks of 32 rows as fp32 matrix products
  // (nh_ssd.metal) on the conv and dt of every row computed first (nh_ssm_conv). Per 2048-row Lightning chunk
  // (NI_PROFILE): 0.99 ms (conv 0.15, nh_ssd_m 0.07, nh_ssd_scan 0.77) against nh_ssm's 1.70. NI_NH_SSD=0: nh_ssm.
  static const bool ssd = !getenv("NI_NH_SSD") || atoi(getenv("NI_NH_SSD")) != 0;
  const uint32_t nch = (uint32_t(T) + 31) / 32;
  const bool pc = ssd && tg_ && !short_pass && cfg_.head_dim == 64 && cfg_.d_state == 128 && nch >= 2 && np == 0 &&
                  (parts & 1) && cfg_.n_head % cfg_.n_group == 0 &&
                  pcv_.size >= (size_t(T) * cfg_.conv_ch() + size_t(T) * cfg_.n_head * 2) * 4 &&
                  psb_.size >= size_t(cfg_.n_head) * nch * (32 * 32 + 64) * 4;
  if (pc) {
    const uint32_t CH = uint32_t(cfg_.conv_ch()), sp[1] = {nch};
    e.use(pipe("nh_ssm_conv"));
    e.buf(0, zx);
    e.buf(1, s.conv[size_t(li)]);
    e.buf(3, pcv_);
    e.buf(4, L.conv_w);
    e.buf(5, L.conv_b);
    e.buf(6, L.A);
    e.buf(8, L.dt_bias);
    e.val(9, p);
    e.dispatch(uint32_t(T), (CH + uint32_t(cfg_.n_head) + 255) / 256, 1, 256);
    e.use(pipe("nh_ssd_m"));
    e.buf(6, L.A);
    e.val(9, p);
    e.buf(11, pcv_);
    e.buf(12, psb_);
    e.bytes(13, sp, sizeof sp);
    e.dispatch(uint32_t(cfg_.n_group) * nch, 1, 1, 128);
    e.use(pipe("nh_ssd_scan"));
    e.buf(0, zx);
    e.buf(2, s.rec[size_t(li)]);
    e.buf(3, y);
    e.buf(7, L.D);
    e.val(9, p);
    e.buf(11, pcv_);
    e.buf(12, psb_);
    e.bytes(13, sp, sizeof sp);
    e.dispatch(uint32_t(cfg_.n_head), 1, 1, 256);  // (a threadgroup per head, a simdgroup per 8 state rows)
  }
  if ((parts & 1) && !pc) {
  // (the long passes: a head per threadgroup of 4 x head_dim threads; nh_ssm_h80 for the Nano models' head dim 80)
  e.use(pipe(tree ? "nh_ssm1_tree"
                  : rps ? "nh_ssm1r" + std::to_string(rps)
                        : short_pass ? "nh_ssm1" : cfg_.head_dim == 80 ? "nh_ssm_h80" : "nh_ssm"));
  e.buf(0, zx);
  e.buf(1, s.conv[size_t(li)]);
  e.buf(2, s.rec[size_t(li)]);
  e.buf(3, y);
  e.buf(4, L.conv_w);
  e.buf(5, L.conv_b);
  e.buf(6, L.A);
  e.buf(7, L.D);
  e.buf(8, L.dt_bias);
  e.val(9, p);
  e.buf(10, write ? live_all_ : live());
  const Ssm1P q{uint32_t(np), ptree};
  e.buf(11, zp ? *zp : zx);
  e.val(12, q);
  e.buf(13, s.pend_path);
  if (tree) e.buf(14, tree_);
  if (short_pass) e.dispatch(uint32_t(cfg_.n_head * cfg_.head_dim / (8 * (rps ? rps : 1))), 1, 1, 256);
  else e.dispatch(uint32_t(cfg_.n_head), 1, 1, uint32_t(4 * cfg_.head_dim));
  }
  if (!(parts & 2)) return;
  if (np > 0) {  // the committed pending rows' raw xBC into the ring (after every head has read it)
    const uint32_t CH = uint32_t(cfg_.conv_ch());
    const RingP rp{uint32_t(np), uint32_t(pos0 - np), uint32_t(cfg_.in_proj()), uint32_t(cfg_.d_inner), CH, ptree};
    const uint32_t nn = CH * uint32_t(std::min(np, 3));
    e.use(pipe("nh_conv_ring"));
    e.buf(0, *zp);
    e.buf(1, s.conv[size_t(li)]);
    e.val(2, rp);
    e.buf(3, s.pend_path);
    e.dispatch((nn + 255) / 256, 1, 1, 256);
  }
  if (write && T > 1) {  // (one row: nh_ssm writes its ring slot itself)
    const uint32_t CH = uint32_t(cfg_.conv_ch());
    const RingP rp{uint32_t(T), uint32_t(pos0), uint32_t(cfg_.in_proj()), uint32_t(cfg_.d_inner), CH, 0u};
    const uint32_t n = CH * uint32_t(std::min(T, 3));
    e.use(pipe("nh_conv_ring"));
    e.buf(0, zx);
    e.buf(1, s.conv[size_t(li)]);
    e.val(2, rp);
    e.buf(3, s.pend_path);
    e.dispatch((n + 255) / 256, 1, 1, 256);
  }
}

// 2-8 rows of BF16 q, k, v (Lightning's verify): one nh_xb_v8_qkv pass for the three, each output as nh_xb_v8's;
// false where the three do not fit it (enc_gemm's three passes then)
bool NemotronH::enc_qkv(Encoder& e, const NhLayer& L, int R) {
  if (!v8_ || R > 8 || L.q.fmt != NhFmt::BF16 || L.k.fmt != NhFmt::BF16 || L.v.fmt != NhFmt::BF16 ||
      L.q.K % 128 != 0 || L.k.K != L.q.K || L.v.K != L.q.K || L.k.N != L.v.N || L.q.E != 1 || L.k.E != 1 || L.v.E != 1)
    return false;
  const V8P pq{L.q.N, L.q.K, L.q.K, L.q.N, uint32_t(R), L.q.gs, 0}, pk{L.k.N, L.k.K, L.k.K, L.k.N, uint32_t(R), L.k.gs, 0};
  e.use(pipe(v8_n1_ ? "nh_xb_v8_qkv_n1" : "nh_xb_v8_qkv"));
  e.buf(0, xnh_);
  e.buf(1, L.q.q);
  e.buf(2, L.k.q);
  e.buf(3, q_);
  e.val(4, pq);
  e.buf(5, L.v.q);
  e.buf(6, k_);
  e.buf(7, v_);
  e.val(8, pk);
  if (v8_n1_) e.dispatch((L.q.N + 7) / 8 + 2 * ((L.k.N + 7) / 8), 1, 1, 256);
  else e.dispatch((L.q.N + 15) / 16 + 2 * ((L.k.N + 15) / 16), 1, 1, 256);
  return true;
}

void NemotronH::enc_attn(Encoder& e, Seq& s, int li, int pos0, int R) {
  const NhLayer& L = layers_[size_t(li)];
  const bool gm = gemm_rows_ && R >= 2;  // xnh_ holds the normed rows in half
  e.group_begin();  // (a grouped encoder runs the three projections together)
  if (gm && enc_qkv(e, L, R)) {
  } else if (gm) {
    enc_gemm(e, L.q, xnh_, q_, R, 0);
    enc_gemm(e, L.k, xnh_, k_, R, 0);
    enc_gemm(e, L.v, xnh_, v_, R, 0);
  } else if (R == 1 && v8_n1_ && L.q.fmt == NhFmt::BF16 && L.k.fmt == NhFmt::BF16 && L.v.fmt == NhFmt::BF16 &&
             L.k.N == L.v.N && L.k.K == L.q.K && L.v.K == L.q.K && L.q.K % 8 == 0 && L.q.E == 1 && L.k.E == 1 &&
             L.v.E == 1) {
    // one row on the M5: the three projections in one dispatch (nh_mv_bf16_qkv_n2, each row as nh_mv_bf16_r1_n2)
    const MvP pq{L.q.N, L.q.K, L.q.K, L.q.N, L.q.gs, 1}, pk{L.k.N, L.k.K, L.k.K, L.k.N, L.k.gs, 1};
    e.use(pipe("nh_mv_bf16_qkv_n2"));
    e.buf(0, L.q.q);
    e.buf(1, L.k.q);
    e.buf(2, xn_);
    e.buf(3, q_);
    e.val(4, pq);
    e.buf(5, L.v.q);
    e.buf(6, k_);
    e.buf(7, v_);
    e.val(8, pk);
    e.dispatch((L.q.N + 7) / 8 + 2 * ((L.k.N + 7) / 8), 1, 1, 128);
  } else {
    enc_mv(e, L.q, xn_, q_, R, 0);
    enc_mv(e, L.k, xn_, k_, R, 0);
    enc_mv(e, L.v, xn_, v_, R, 0);
  }
  e.group_end();
  const KvP kp{uint32_t(R), uint32_t(pos0), uint32_t(s.cap), uint32_t(cfg_.hkv), uint32_t(cfg_.hd)};
  e.use(pipe("nh_kv_write"));
  e.buf(0, k_);
  e.buf(1, v_);
  e.buf(2, s.kc[size_t(li)]);
  e.buf(3, s.vc[size_t(li)]);
  e.val(4, kp);
  e.dispatch(uint32_t(R), uint32_t(2 * cfg_.hkv), 1, uint32_t(cfg_.hd));
  const int L_max = pos0 + R;
  uint32_t chunk = attn_chunk_;
  while ((uint32_t(L_max) + chunk - 1) / chunk > uint32_t(attn_max_chunks_)) chunk *= 2;
  const uint32_t nchunks = (uint32_t(L_max) + chunk - 1) / chunk;
  const AttnP ap{uint32_t(R), uint32_t(pos0), uint32_t(s.cap), uint32_t(cfg_.hq), uint32_t(cfg_.hkv), chunk, nchunks,
                 1.0f / std::sqrt(float(cfg_.hd))};
  e.use(pipe(tree_on_ ? "nh_attn_partial_tree" : "nh_attn_partial"));  // (a KFlash tree: rows see their ancestors)
  e.buf(0, q_);
  e.buf(1, s.kc[size_t(li)]);
  e.buf(2, s.vc[size_t(li)]);
  e.buf(3, part_o_);
  e.buf(4, part_ml_);
  e.val(5, ap);
  if (tree_on_) e.buf(6, tree_);
  e.dispatch(nchunks, uint32_t(cfg_.hkv), uint32_t(R), uint32_t(32 * (cfg_.hq / cfg_.hkv)));
  // 2+ rows: the heads' outputs straight to half (o_proj's GEMM input)
  e.use(pipe(gm ? "nh_attn_reduce_h" : "nh_attn_reduce"));
  e.buf(0, part_o_);
  e.buf(1, part_ml_);
  e.buf(2, gm ? aoh_ : ao_);
  e.val(3, ap);
  e.dispatch(uint32_t(cfg_.hq), uint32_t(R), 1, uint32_t(cfg_.hd));
  if (gm) {
    enc_gemm(e, L.o, aoh_, x_, R, 1);
  } else {
    enc_mv(e, L.o, ao_, x_, R, 1);
  }
}

void NemotronH::enc_moe(Encoder& e, const NhLayer& L, int R) {
  const uint32_t k = uint32_t(L.k_used);
  if (R == 1 && fuse_dec_ && L.router.fmt == NhFmt::BF16 && L.lat_down.fmt == NhFmt::Q8 && L.lat_up.fmt == NhFmt::Q8 &&
      L.up_s.fmt == NhFmt::IQ2 && L.down_s.fmt == NhFmt::IQ2 && L.up_e.fmt == NhFmt::IQ2 && L.E <= 512 &&
      L.k_used <= 32) {
    // one token: 4 dispatches (nh_zdec.metal)
    const uint32_t H = uint32_t(cfg_.hidden), Lt = uint32_t(cfg_.latent), S = uint32_t(cfg_.shared_I), I = uint32_t(L.I);
    const uint32_t tg_r = (uint32_t(L.E) + 15) / 16, tg_l = (Lt + 15) / 16, tg_u = k * ((I + 15) / 16);
    const MoeDecP dp{H, uint32_t(L.E), k, cfg_.w_norm ? 1u : 0u, cfg_.w_scale, Lt, S, I, uint32_t(L.up_e.q_exp()),
                     uint32_t(L.up_e.s_exp()), tg_r, tg_l, tg_u};
    e.use(pipe("nh_moe_a1"));
    e.buf(0, xn_);
    e.buf(1, L.router.q);
    e.buf(2, L.lat_down.q);
    e.buf(3, L.lat_down.s);
    e.buf(4, L.up_s.q);
    e.buf(5, L.up_s.s);
    e.buf(6, rl_);
    e.buf(7, lat_);
    e.buf(8, sh_);
    e.buf(9, L.exp_bias);
    e.buf(10, eid_);
    e.buf(11, ew_);
    e.buf(12, route_ctr_);
    e.val(13, dp);
    e.dispatch(tg_r + tg_l + (S + 15) / 16, 1, 1, 128);
    e.use(pipe("nh_moe_b1"));
    e.buf(0, L.up_e.q);
    e.buf(1, L.up_e.s);
    e.buf(2, lat_);
    e.buf(3, h_);
    e.buf(4, eid_);
    e.buf(5, L.down_s.q);
    e.buf(6, L.down_s.s);
    e.buf(7, sh_);
    e.buf(8, sd_);
    e.val(9, dp);
    e.dispatch(tg_u + (H + 15) / 16, 1, 1, 128);
    const MoeP mp{I, Lt, k, uint32_t(L.up_e.q_exp()), uint32_t(L.up_e.s_exp()), uint32_t(L.down_e.q_exp()),
                  uint32_t(L.down_e.s_exp()), I};
    e.use(pipe(x2_ ? "nh_x2_moe_dn3" : "nh_moe2_dn_iq2"));
    e.buf(0, L.down_e.q);
    e.buf(1, L.down_e.s);
    e.buf(2, h_);
    e.buf(3, part_);
    e.buf(4, eid_);
    e.buf(5, ew_);
    e.val(6, mp);
    e.buf(8, live_all_);
    if (x2_) e.dispatch((Lt + 2) / 3, k, 1, 32);
    else e.dispatch((Lt + 15) / 16, k, 1, 128);
    e.use(pipe("nh_moe_d1"));
    e.buf(0, L.lat_up.q);
    e.buf(1, L.lat_up.s);
    e.buf(2, part_);
    e.buf(3, sd_);
    e.buf(4, x_);
    e.val(5, dp);
    e.dispatch((H + 15) / 16, 1, 1, 128);
    return;
  }
  if (R == 1 && fuse_dec_ && L.router.fmt == NhFmt::BF16 && L.lat_down.fmt == NhFmt::Q8 && L.lat_up.fmt == NhFmt::Q8 &&
      L.up_s.fmt == NhFmt::FP4 && L.down_s.fmt == NhFmt::FP4 && L.up_e.fmt == NhFmt::FP4 && L.E <= 512 &&
      L.k_used <= 32 && cfg_.hidden % 32 == 0 && cfg_.shared_I % 32 == 0 && cfg_.latent % 32 == 0) {
    // one token, NVFP4 experts and shared expert: 4 dispatches (nh_zdec.metal)
    // 32 rows (8 simdgroups) per threadgroup
    const uint32_t H = uint32_t(cfg_.hidden), Lt = uint32_t(cfg_.latent), S = uint32_t(cfg_.shared_I), I = uint32_t(L.I);
    const uint32_t tg_r = (uint32_t(L.E) + 31) / 32, tg_l = (Lt + 31) / 32, tg_u = k * ((I + 31) / 32);
    const MoeDecP dp{H, uint32_t(L.E), k, cfg_.w_norm ? 1u : 0u, cfg_.w_scale, Lt, S, I, uint32_t(L.up_e.q_exp()),
                     uint32_t(L.up_e.s_exp()), tg_r, tg_l, tg_u};
    e.use(pipe("nh_moe_a4"));
    e.buf(0, xn_);
    e.buf(1, L.router.q);
    e.buf(2, L.lat_down.q);
    e.buf(3, L.lat_down.s);
    e.buf(4, L.up_s.q);
    e.buf(5, L.up_s.s);
    e.buf(6, rl_);
    e.buf(7, lat_);
    e.buf(8, sh_);
    e.buf(9, L.exp_bias);
    e.buf(10, eid_);
    e.buf(11, ew_);
    e.buf(12, route_ctr_);
    e.val(13, dp);
    e.val(14, L.up_s.gs);
    e.dispatch(tg_r + tg_l + (S + 31) / 32, 1, 1, 256);
    e.use(pipe("nh_moe_b4"));
    e.buf(0, L.up_e.q);
    e.buf(1, L.up_e.s);
    e.buf(2, lat_);
    e.buf(3, h_);
    e.buf(4, eid_);
    e.buf(5, L.down_s.q);
    e.buf(6, L.down_s.s);
    e.buf(7, sh_);
    e.buf(8, sd_);
    e.val(9, dp);
    e.buf(10, L.up_e.gse);
    e.val(11, L.down_s.gs);
    e.dispatch(tg_u + (H + 31) / 32, 1, 1, 256);
    const MoeP mp{I, Lt, k, uint32_t(L.up_e.q_exp()), uint32_t(L.up_e.s_exp()), uint32_t(L.down_e.q_exp()),
                  uint32_t(L.down_e.s_exp()), I};
    e.use(pipe("nh_x4_moe_dn_sum"));
    e.buf(0, L.down_e.q);
    e.buf(1, L.down_e.s);
    e.buf(2, h_);
    e.buf(3, part_);
    e.buf(4, eid_);
    e.buf(5, ew_);
    e.val(6, mp);
    e.buf(8, live_all_);
    e.buf(9, L.down_e.gse);
    e.buf(10, lo_);
    e.buf(11, moecnt_);
    e.dispatch((Lt + 31) / 32, k, 1, 256);
    e.use(pipe("nh_moe_d2"));
    e.buf(0, L.lat_up.q);
    e.buf(1, L.lat_up.s);
    e.buf(2, lo_);
    e.buf(3, sd_);
    e.buf(4, x_);
    e.val(5, dp);
    e.dispatch((H + 15) / 16, 1, 1, 128);
    return;
  }
  if (R == 1 && fuse_dec_ && !L.lat_up && L.router.fmt == NhFmt::F32 && L.up_s.fmt == NhFmt::FP4 &&
      L.down_s.fmt == NhFmt::FP4 && L.up_e.fmt == NhFmt::FP4 && L.E <= 512 && L.k_used <= 32 && cfg_.hidden % 32 == 0 &&
      cfg_.shared_I % 32 == 0) {
    // one token, no latent projections (Lightning): 4 dispatches (nh_zh_moe.metal)
    const uint32_t H = uint32_t(cfg_.hidden), S = uint32_t(cfg_.shared_I), I = uint32_t(L.I);
    const uint32_t tg_r = (uint32_t(L.E) + 15) / 16, tg_u = k * ((I + 15) / 16);
    const MoeDecP dp{H, uint32_t(L.E), k, cfg_.w_norm ? 1u : 0u, cfg_.w_scale, H, S, I, uint32_t(L.up_e.q_exp()),
                     uint32_t(L.up_e.s_exp()), tg_r, 0, tg_u};
    e.use(pipe("nh_moe_ah"));
    e.buf(0, xn_);
    e.buf(1, L.router.q);
    e.buf(4, L.up_s.q);
    e.buf(5, L.up_s.s);
    e.buf(6, rl_);
    e.buf(8, sh_);
    e.buf(9, L.exp_bias);
    e.buf(10, eid_);
    e.buf(11, ew_);
    e.buf(12, route_ctr_);
    e.val(13, dp);
    e.val(14, L.up_s.gs);
    e.dispatch(tg_r + (S + 15) / 16, 1, 1, 128);
    e.use(pipe(L.up_e.l2 ? "nh_moe_b4_l2" : "nh_moe_b4"));
    e.buf(0, L.up_e.q);
    e.buf(1, L.up_e.s);
    e.buf(2, xn_);
    e.buf(3, h_);
    e.buf(4, eid_);
    e.buf(5, L.down_s.q);
    e.buf(6, L.down_s.s);
    e.buf(7, sh_);
    e.buf(8, sd_);
    e.val(9, dp);
    e.buf(10, L.up_e.gse);
    e.val(11, L.down_s.gs);
    e.dispatch(tg_u + (H + 15) / 16, 1, 1, 128);
    const MoeP mp{I, H, k, uint32_t(L.up_e.q_exp()), uint32_t(L.up_e.s_exp()), uint32_t(L.down_e.q_exp()),
                  uint32_t(L.down_e.s_exp()), I};
    e.use(pipe(L.down_e.l2 ? "nh_x4_moe_dn_l2" : "nh_x4_moe_dn"));
    e.buf(0, L.down_e.q);
    e.buf(1, L.down_e.s);
    e.buf(2, h_);
    e.buf(3, part_);
    e.buf(4, eid_);
    e.buf(5, ew_);
    e.val(6, mp);
    e.buf(8, live_all_);
    e.buf(9, L.down_e.gse);
    e.dispatch((H + 7) / 8, k, 1, 64);
    e.use(pipe("nh_moe_dh"));
    e.buf(0, part_);
    e.buf(1, sd_);
    e.buf(2, x_);
    e.val(3, dp);
    e.dispatch((H + 255) / 256, 1, 1, 256);
    return;
  }
  const bool gm = gemm_rows_ && R >= 2;
  // without latent projections (Lightning) the experts read the normed hidden state, their sum goes into x_
  const bool lat = bool(L.lat_up);
  const Buffer& lin = lat ? lat_ : xn_;
  // With a grouped encoder (NI_NH_GROUP): router, latent down and shared up run together, then routed down and shared
  // down; a serial encoder runs them one after another in the same order. (The batched decode's grouped passes keep the
  // serial order unless NI_NH_GROUP is on, as a single request's passes do: the shared expert's sum then comes after
  // the routed experts', so a batched pass's rows round as their single passes do.)
  const bool grouped = e.grouped() && group_;
  // The M5's verify (Lightning): the router beside the shared expert's up product (nh_moe_ru_v8t), the routing beside
  // its down product (nh_moe_rd_v8t), stored into sdv_ and added after the routed experts' sum (nh_moe2_sum2_add_sd):
  // each value and each addition as the separate passes below make them
  const bool rsu = rsu_ && gm && !grouped && !lat && R <= kMaxRows && L.router.fmt == NhFmt::F32 && L.E <= 512 &&
                   L.k_used <= 32 && L.up_e.fmt == NhFmt::FP4 && v8t_ && v8_ && x4_ && L.up_s.fmt == NhFmt::FP4 &&
                   !L.up_s.l2 && L.up_s.K % 64 == 0 && L.down_s.fmt == NhFmt::FP4 && !L.down_s.l2 &&
                   L.down_s.K % 64 == 0 && L.router.K % 4 == 0 && cfg_.hidden % 4 == 0 && !getenv("NI_NH_V8T_KS");
  if (rsu) {
    const V8P pu{L.up_s.N, L.up_s.K, L.up_s.K, L.up_s.N, uint32_t(R), L.up_s.gs, 2},
        pd{L.down_s.N, L.down_s.K, L.down_s.K, L.down_s.N, uint32_t(R), L.down_s.gs, 0};
    const struct {
      uint32_t E, K, x_stride, lg_stride, R, k, norm;
      float scale, gs;
    } rp{uint32_t(L.E), L.router.K, uint32_t(cfg_.hidden), uint32_t(cfg_.max_E), uint32_t(R), k, cfg_.w_norm ? 1u : 0u,
         cfg_.w_scale, L.router.gs};
    e.use(pipe("nh_moe_ru_v8t"));
    e.buf(0, xnh_);
    e.buf(1, L.up_s.q);
    e.buf(2, L.up_s.s);
    e.buf(3, shh_);
    e.val(4, pu);
    e.buf(5, L.router.q);
    e.buf(6, xn_);
    e.buf(7, rl_);
    e.bytes(13, &rp, sizeof rp);
    e.dispatch(uint32_t(L.E) + (L.up_s.N + 15) / 16, 1, 1, 256);
    e.use(pipe("nh_moe_rd_v8t"));
    e.buf(0, shh_);
    e.buf(1, L.down_s.q);
    e.buf(2, L.down_s.s);
    e.buf(3, sdv_);
    e.val(4, pd);
    e.buf(7, rl_);
    e.buf(8, L.exp_bias);
    e.buf(9, eid_);
    e.buf(10, ew_);
    e.buf(12, live());
    e.bytes(13, &rp, sizeof rp);
    e.dispatch(1 + (L.down_s.N + 15) / 16, 1, 1, 256);
  }
  // router: logits (BF16 weights, fp32 products), sigmoid + selection bias, top-k
  if (grouped) e.group_begin();
  if (!rsu) enc_mv(e, L.router, xn_, rl_, R, 0, uint32_t(cfg_.hidden), uint32_t(cfg_.max_E));
  if (grouped) {
    if (lat && gm) enc_gemm(e, L.lat_down, xnh_, lat_, R, 0);
    else if (lat) enc_mv(e, L.lat_down, xn_, lat_, R, 0);
    if (gm) enc_gemm(e, L.up_s, xnh_, shh_, R, 2);
    else enc_mv(e, L.up_s, xn_, sh_, R, 2);
    e.group_end();
  }
  const RouteP rp{uint32_t(L.E), k, cfg_.w_norm ? 1u : 0u, cfg_.w_scale, uint32_t(cfg_.max_E), uint32_t(R)};
  const bool route1 = L.E <= 512 && L.k_used <= 32;
  // IQ2 experts for 2-8 rows (Bolt's verify) on the M5: the routing and nh_x2_bucket's tiles in one dispatch
  // (nh_x2_route_bucket: the same experts, weights and tiles)
  const bool x2e = L.up_e.fmt != NhFmt::FP4 && x2_ && L.E <= 1024 && L.I % 256 == 0 && cfg_.latent % 256 == 0;
  const bool up8 = cfg_.latent <= 1024;
  const struct {
    uint32_t P, E, BM, k, up_x, dn_x;
  } bp{uint32_t(R) * k, uint32_t(L.E), 2, k, (uint32_t(L.I) + (up8 ? 7 : 3)) / (up8 ? 8 : 4),
       (uint32_t(cfg_.latent) + 4 * x2_sgd_ - 1) / (4 * x2_sgd_)};
  const bool fused_bucket = x2e && R >= 2 && route1 && Gpu::get().apple10() && route_bucket_;
  if (fused_bucket) {
    e.use(pipe("nh_x2_route_bucket"));
    e.buf(0, rl_);
    e.buf(1, L.exp_bias);
    e.buf(2, eid_);
    e.buf(3, ew_);
    e.val(4, rp);
    e.buf(5, live());
    e.buf(6, x2sorted_);
    e.buf(7, x2tiles_);
    e.buf(8, x2args_);
    e.bytes(9, &bp, sizeof bp);
    e.dispatch(1, 1, 1, 256);
  } else if (!rsu) {
    e.use(pipe(route1 ? "nh_moe_route1" : "nh_moe_route"));
    e.buf(0, rl_);
    e.buf(1, L.exp_bias);
    e.buf(2, eid_);
    e.buf(3, ew_);
    e.val(4, rp);
    e.buf(5, live());
    if (route1) e.dispatch((uint32_t(R) + 7) / 8, 1, 1, 256);
    else e.dispatch(uint32_t(R), 1, 1, 512);
  }
  // latent projection; routed experts: per (row, expert) pair matvecs (a verify block's 72 pairs average ~1.4 rows per expert, so
  // grouping the pairs by expert into 8-row MMA tiles (1.02 ms per layer) or 16-row GEMM tiles (~1.8) measured slower
  // than 0.82 ms per layer for the pairs; so did decoding each expert row once for all of its pairs, which spills)
  if (!grouped && lat) {
    if (gm) enc_gemm(e, L.lat_down, xnh_, lat_, R, 0);
    else enc_mv(e, L.lat_down, xn_, lat_, R, 0);
  }
  const MoeP mp{uint32_t(L.I), uint32_t(cfg_.latent), k, uint32_t(L.up_e.q_exp()), uint32_t(L.up_e.s_exp()),
                uint32_t(L.down_e.q_exp()), uint32_t(L.down_e.s_exp()), uint32_t(cfg_.max_I)};
  // the down projection per (pair, 16 rows) into weighted partials, summed per row (nh_moe2_*)
  const uint32_t npairs = uint32_t(R) * k, nrows = uint32_t(R);
  {
    const bool fp4 = L.up_e.fmt == NhFmt::FP4;
    const bool x2 = x2e;
    if (fp4) {
      // NVFP4 experts: per (row, expert) pair, each expert's tensor scale applied before relu^2 (nh_x4_moe_*)
      // (nh_x4_moe_dn_sum's counters assume 8 simdgroups: the latent path keeps them)
      const uint32_t thr = lat ? 256u : moe_thr_, rows_tg = thr / 8;
      e.use(pipe(L.up_e.l2 ? "nh_x4_moe_up_l2" : "nh_x4_moe_up"));
      e.buf(0, L.up_e.q);
      e.buf(1, L.up_e.s);
      e.buf(2, lin);
      e.buf(3, h_);
      e.buf(4, eid_);
      e.val(5, mp);
      e.buf(7, live());
      e.buf(9, L.up_e.gse);
      // (dynamic rows: the live rows' pairs only, nh_moe_pair_args' grid)
      const bool ind = pair_args_ && pair_thr_ == thr && uint32_t(R) == pair_rows_ && k == pair_k_ &&
                       (uint32_t(L.I) + rows_tg - 1) / rows_tg == pair_gx_up_ &&
                       (uint32_t(cfg_.latent) + rows_tg - 1) / rows_tg == pair_gx_dn_;
      if (ind) e.dispatch_indirect(moe_args_, 0, thr);
      else e.dispatch((uint32_t(L.I) + rows_tg - 1) / rows_tg, npairs, 1, thr);
      if (grouped) e.group_begin();
      // (the rows' sums into lo_: no nh_moe2_sum2 below; without latent projections the partials, then their sums
      // added into the residual by nh_moe2_sum2_add)
      // (split_: nh_x4_moe_dn, its products to the bit, and nh_moe2_sum2 below)
      e.use(pipe(lat && !split_ ? "nh_x4_moe_dn_sum" : L.down_e.l2 ? "nh_x4_moe_dn_l2" : "nh_x4_moe_dn"));
      e.buf(0, L.down_e.q);
      e.buf(1, L.down_e.s);
      e.buf(2, h_);
      e.buf(3, part_);
      e.buf(4, eid_);
      e.buf(5, ew_);
      e.val(6, mp);
      e.buf(8, live());
      e.buf(9, L.down_e.gse);
      e.buf(10, lo_);
      e.buf(11, moecnt_);
      if (ind) e.dispatch_indirect(moe_args_, 16, thr);
      else e.dispatch((uint32_t(cfg_.latent) + rows_tg - 1) / rows_tg, npairs, 1, thr);
    } else if (x2 && R >= 2) {
      // the pairs grouped by expert into tiles of <= 2 (each expert row decoded once for both tokens), the up and down
      // dispatches sized by the bucketing (indirect): a verify layer 0.36-0.52 vs 0.55-0.88 ms; up with 8 rows per
      // simdgroup when the latent fits in registers (K <= 1024: 3-5 % per layer)
      if (!fused_bucket) {
        e.use(pipe("nh_x2_bucket"));
        e.buf(0, eid_);
        e.buf(1, x2sorted_);
        e.buf(2, x2tiles_);
        e.buf(3, x2args_);
        e.buf(4, x2rank_);
        e.bytes(5, &bp, sizeof bp);
        e.buf(6, live());
        e.dispatch(1, 1, 1, 1024);
      }
      e.use(pipe(up8 ? "nh_x2_moeb_up8" : "nh_x2_moeb_up"));
      e.buf(0, L.up_e.q);
      e.buf(1, L.up_e.s);
      e.buf(2, lin);
      e.buf(3, h_);
      e.buf(4, x2tiles_);
      e.buf(5, x2sorted_);
      e.val(6, mp);
      e.dispatch_indirect(x2args_, 0, 32, 1, 1);
      if (grouped) e.group_begin();
      e.use(pipe("nh_x2_moeb_dn"));
      e.buf(0, L.down_e.q);
      e.buf(1, L.down_e.s);
      e.buf(2, h_);
      e.buf(3, part_);
      e.buf(4, x2tiles_);
      e.buf(5, x2sorted_);
      e.buf(6, ew_);
      e.val(7, mp);
      e.dispatch_indirect(x2args_, 16, 32 * x2_sgd_, 1, 1);
    } else {
      e.use(pipe(x2 ? "nh_x2_moe_up4" : "nh_moe2_up_iq2"));
      e.buf(0, L.up_e.q);
      e.buf(1, L.up_e.s);
      e.buf(2, lin);
      e.buf(3, h_);
      e.buf(4, eid_);
      e.val(5, mp);
      e.buf(7, live());
      if (x2) e.dispatch((uint32_t(L.I) + 3) / 4, npairs, 1, 32);
      else e.dispatch((uint32_t(L.I) + 15) / 16, npairs, 1, 128);
      if (grouped) e.group_begin();
      e.use(pipe(x2 ? "nh_x2_moe_dn3" : "nh_moe2_dn_iq2"));
      e.buf(0, L.down_e.q);
      e.buf(1, L.down_e.s);
      e.buf(2, h_);
      e.buf(3, part_);
      e.buf(4, eid_);
      e.buf(5, ew_);
      e.val(6, mp);
      e.buf(8, live());
      if (x2) e.dispatch((uint32_t(cfg_.latent) + 2) / 3, npairs, 1, 32);
      else e.dispatch((uint32_t(cfg_.latent) + 15) / 16, npairs, 1, 128);
    }
    if (grouped) {
      if (gm) enc_gemm(e, L.down_s, shh_, x_, R, 1);
      else enc_mv(e, L.down_s, sh_, x_, R, 1);
      e.group_end();
    }
    // the experts' weighted sum: into lo_ for the latent up projection (NVFP4: summed by nh_x4_moe_dn_sum above),
    // else added into the residual
    if (!lat) {
      e.use(pipe(rsu ? "nh_moe2_sum2_add_sd" : "nh_moe2_sum2_add"));
      e.buf(0, part_);
      e.buf(1, x_);
      e.val(2, mp);
      e.buf(3, live());
      e.buf(4, sdv_);
      e.dispatch((uint32_t(cfg_.latent) + 63) / 64, nrows, 1, 64);
    } else if (fp4 && split_) {  // (nh_x4_moe_dn_sum's sums: the same additions in the same order)
      e.use(pipe("nh_moe2_sum2"));
      e.buf(0, part_);
      e.buf(1, lo_);
      e.val(2, mp);
      e.buf(3, live());
      e.dispatch((uint32_t(cfg_.latent) + 63) / 64, nrows, 1, 64);
    } else if (!fp4) {
      e.use(pipe("nh_moe2_sum"));
      e.buf(0, part_);
      e.buf(1, lo_);
      e.val(2, mp);
      e.buf(3, live());
      e.dispatch(nrows, 1, 1, 256);
    }
  }
  // shared expert on the full hidden state; both branches add into the residual stream
  if (rsu) {  // (the shared expert ran above)
  } else if (grouped) {  // (the shared expert ran above)
    if (lat && gm) {
      enc_rows_half(e, lo_, loh_, R, cfg_.latent);
      enc_gemm(e, L.lat_up, loh_, x_, R, 1);
    } else if (lat) {
      enc_mv(e, L.lat_up, lo_, x_, R, 1);
    }
  } else if (gm) {
    enc_gemm(e, L.up_s, xnh_, shh_, R, 2);
    if (lat) {
      enc_rows_half(e, lo_, loh_, R, cfg_.latent);
      enc_gemm(e, L.lat_up, loh_, x_, R, 1);
    }
    enc_gemm(e, L.down_s, shh_, x_, R, 1);
  } else {
    enc_mv(e, L.up_s, xn_, sh_, R, 2);
    if (lat) enc_mv(e, L.lat_up, lo_, x_, R, 1);
    enc_mv(e, L.down_s, sh_, x_, R, 1);
  }
}

// A dense MLP layer ('D'): x += W_down relu^2(W_up xn) over R <= kMaxRows rows, on the shared expert's kernels and
// epilogues: relu^2 in the up product (into sh_, or half shh_ for the down GEMM), the residual add in the down product.
// One row: the one-row kernels on xn_ (fp32); 2+ rows: the 8-row GEMMs on xnh_ (the normed rows as half).
void NemotronH::enc_mlp(Encoder& e, const NhLayer& L, int R) {
  if (gemm_rows_ && R >= 2) {
    enc_gemm(e, L.up_s, xnh_, shh_, R, 2);
    enc_gemm(e, L.down_s, shh_, x_, R, 1);
  } else {
    enc_mv(e, L.up_s, xn_, sh_, R, 2);
    enc_mv(e, L.down_s, sh_, x_, R, 1);
  }
}

void NemotronH::enc_rows(Encoder& e, Seq& s, int pos0, int R, bool commit) {
  if (R < 1 || R > kMaxRows) fatal("NemotronH: %d rows in one pass", R);
  if (pos0 + R >= s.cap) fatal("context full (%d)", s.cap);
  enc_embed(e, s.tok, pos0, R, x_);
  if (s.embeds) vision::enc_embed_runs(e, *s.embeds, pos0, R, x_, cfg_.hidden);  // image rows (a prompt's prefill)
  // a dynamic verify (rows past live[0] dropped): the routed experts' grids of the live rows' pairs (nh_moe_pair_args)
  pair_args_ = false;
  if (pair_ind_ && dyn_rows_ && !commit && gemm_rows_ && R >= 2 && moe_lightning_) {
    const uint32_t thr = moe_thr_, rows_tg = thr / 8;
    pair_thr_ = thr;
    pair_rows_ = uint32_t(R);
    pair_k_ = moe_k_;
    pair_gx_up_ = (moe_I_ + rows_tg - 1) / rows_tg;
    pair_gx_dn_ = (uint32_t(cfg_.latent) + rows_tg - 1) / rows_tg;
    const struct {
      uint32_t gx_up, gx_dn, k, R;
    } pp{pair_gx_up_, pair_gx_dn_, pair_k_, uint32_t(R)};
    e.use(pipe("nh_moe_pair_args"));
    e.buf(0, live());
    e.buf(1, moe_args_);
    e.val(2, pp);
    e.dispatch(1, 1, 1, 1);
    pair_args_ = true;
  }
  for (int i = 0; i < cfg_.layers; ++i) {
    const NhLayer& L = layers_[size_t(i)];
    if (cap_) enc_capture(e, i, x_, pos0, R);
    const bool gm = gemm_rows_ && R >= 2;
    // 2+ rows: the normed rows as half in one pass (the MoE layers' router also reads them in fp32)
    if (!gm) enc_norm(e, x_, L.norm, xn_, R, cfg_.hidden);
    else if (L.kind == 'E') enc_norm2(e, x_, L.norm, xn_, xnh_, R, cfg_.hidden);
    else enc_norm_half(e, x_, L.norm, xnh_, R, cfg_.hidden);
    if (L.kind == 'M') {
      // a verify block's rows go to the half of the pending buffer the pending rows (if any) do not use; the scan
      // commits those first (merged flush)
      const size_t half = size_t(kMaxRows) * cfg_.in_proj() * 4;
      const Buffer pb = s.pend_qkv[size_t(i)];
      const Buffer zx = commit ? zx_ : pb.view(size_t(1 - s.pend_half) * half, half);
      if (gm) enc_gemm(e, L.in, xnh_, zx, R, 0);
      else enc_mv(e, L.in, xn_, zx, R, 0);
      // a verify committing pending rows: their ring writes go with the grouped norm (nh_gnorm_half_ring)
      const bool merged = !commit && s.n_pend > 0 && merge_flush_, ring_gn = merged && gm;
      const Buffer zp = pb.view(size_t(s.pend_half) * half, half);
      if (merged) enc_ssm(e, s, i, zx, R, pos0, false, y_, &zp, s.n_pend, ring_gn ? 1 : 3);
      else enc_ssm(e, s, i, zx, R, pos0, commit, y_);
      const GnormP gp{uint32_t(cfg_.d_inner), uint32_t(cfg_.d_inner / cfg_.n_group), cfg_.eps};
      e.use(pipe(ring_gn ? "nh_gnorm_half_ring" : gm ? "nh_gnorm_half" : "nh_gnorm"));
      e.buf(0, y_);
      e.buf(1, L.gnorm);
      e.buf(2, gm ? xnh_ : yn_);
      e.val(3, gp);
      if (ring_gn) {
        const RingP rp{uint32_t(s.n_pend), uint32_t(pos0 - s.n_pend), uint32_t(cfg_.in_proj()), uint32_t(cfg_.d_inner),
                       uint32_t(cfg_.conv_ch())};
        e.buf(4, zp);
        e.buf(5, s.conv[size_t(i)]);
        e.val(6, rp);
      }
      e.dispatch(uint32_t(R * cfg_.n_group), 1, 1, 256);
      if (gm) enc_gemm(e, L.out, xnh_, x_, R, 1);
      else enc_mv(e, L.out, yn_, x_, R, 1);
    } else if (L.kind == 'A') {
      enc_attn(e, s, i, pos0, R);
    } else if (L.kind == 'D') {
      enc_mlp(e, L, R);
    } else {
      enc_moe(e, L, R);
    }
  }
  pair_args_ = false;
  if (cap_) enc_capture(e, cfg_.layers, x_, pos0, R);  // (a drafter reading the trunk's output: slot `layers`)
  if (!commit) {
    s.n_pend = R;  // until commit_rows() / commit_rows_tree() says which were accepted
    s.pend_tree = false;
    s.pend_pos = pos0;
    s.pend_half = 1 - s.pend_half;
  }
}

void NemotronH::encode_flush(Encoder& e, Seq& s) {
  if (s.n_pend <= 0) {
    s.n_pend = 0;
    return;
  }
  const size_t half = size_t(kMaxRows) * cfg_.in_proj() * 4;
  for (int i = 0; i < cfg_.layers; ++i) {
    if (cfg_.kind[size_t(i)] != 'M') continue;
    const Buffer zp = s.pend_qkv[size_t(i)].view(size_t(s.pend_half) * half, half);
    // a tree verify's accepted path: committed as pending rows of an empty pass (the scan reads them by path)
    if (s.pend_tree) enc_ssm(e, s, i, zp, 0, s.pend_pos + s.n_pend, false, y_, &zp, s.n_pend);
    else enc_ssm(e, s, i, zp, s.n_pend, s.pend_pos, true, y_);
  }
  s.n_pend = 0;
  s.pend_tree = false;
}

void NemotronH::enc_head(Encoder& e, int R, const Buffer& out, size_t out_row_stride) {
  if (gemm_rows_ && R >= 2) {
    enc_norm_half(e, x_, out_norm_, xnh_, R, cfg_.hidden);
    enc_gemm(e, head_, xnh_, out, R, 0, uint32_t(out_row_stride));
  } else {
    enc_norm(e, x_, out_norm_, xn_, R, cfg_.hidden);
    enc_mv(e, head_, xn_, out, R, 0, uint32_t(cfg_.hidden), uint32_t(out_row_stride));
  }
}

void NemotronH::enc_sample(Encoder& e, const Buffer& lrow, const Buffer& tokens, int out_index, const SamplingParams& sp,
                           uint32_t step, const Buffer* dstep, size_t dstep_off) {
  const uint32_t K = sp.temperature > 0 ? uint32_t(std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32)) : 1u;
  const TopkStageP ts{uint32_t(cfg_.vocab), K, 2048};
  const uint32_t ntg = (uint32_t(cfg_.vocab) + 2047) / 2048;
  e.use(pipe("topk_stage1"));
  e.buf(0, lrow);
  e.buf(1, cand_v_);
  e.buf(2, cand_i_);
  e.val(3, ts);
  e.dispatch(ntg, 1, 1, 256);
  const SampleP p2{ntg * K, K, sp.temperature, sp.top_p, sp.seed, step, uint32_t(out_index)};
  e.use(pipe(dstep ? "sample_stage2_dstep" : "sample_stage2"));
  e.buf(0, cand_v_);
  e.buf(1, cand_i_);
  e.buf(2, tokens);
  e.buf(3, sprob_);
  e.val(4, p2);
  if (dstep) e.buf(5, *dstep, dstep_off);
  e.dispatch(1, 1, 1, 1024);
}

void NemotronH::encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample) {
  encode_flush(e, s);
  enc_rows(e, s, pos, 1, true);
  if (!sample) return;
  enc_head(e, 1, logits_, size_t(cfg_.vocab));
  enc_logit_mask(e, s, logits_, 1, cfg_.vocab, cfg_.vocab);  // structured output (nothing without masks)
  enc_sample(e, logits_, s.tok, pos + 1, sp, step);
}

void NemotronH::encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step,
                               bool sample, Checkpoint* snap, int snap_row) {
  (void)snap;
  (void)snap_row;
  encode_flush(e, s);
  if (T > tmax_) fatal("NemotronH: prefill of %d rows > chunk %d", T, tmax_);
  Buffer last;
  if (T <= kMaxRows) {
    enc_rows(e, s, pos0, T, true);
    last = x_.view(size_t(T - 1) * cfg_.hidden * 4, size_t(cfg_.hidden) * 4);
  } else {
    if (ane_splits(T)) {
      ane_->begin_chunk(e, T);
      ane_chunk_ = true;
      ++ane_stats_.chunks;
    }
    enc_prefill_gemm(e, s, pos0, T);
    last = px_.view(size_t(T - 1) * cfg_.hidden * 4, size_t(cfg_.hidden) * 4);
  }
  if (!sample) return;
  enc_norm(e, last, out_norm_, xn_, 1, cfg_.hidden);
  enc_mv(e, head_, xn_, logits_, 1, 0);
  enc_logit_mask(e, s, logits_, 1, cfg_.vocab, cfg_.vocab);  // structured output (nothing without masks)
  enc_sample(e, logits_, s.tok, pos0 + T, sp, step);
}

void NemotronH::encode_verify(Encoder& e, Seq& s, int pos, int R) {
  if (!merge_flush_) encode_flush(e, s);  // (merged: each Mamba layer's scan commits the pending rows first)
  else if (s.n_pend > 0 && s.pend_pos + s.n_pend != pos) fatal("NemotronH: verify at %d after pending rows at %d", pos, s.pend_pos);
  enc_rows(e, s, pos, R, false);
  enc_head(e, R, logits_, size_t(cfg_.vocab));
}

// Batched decode (see the header): enc_rows' layer loop over the rows of several sequences. The Mamba in_proj, the
// attention projections, the MoE and the LM head run once over all the rows; each group's in_proj rows go into its
// sequence's pending buffer (the half its pending rows do not use) before its scan, which commits those pending rows
// first (merged flush); its KV cache write and attention run on views of the shared row buffers.
void NemotronH::encode_verify_multi(Encoder& e, const std::vector<VerifyGroup>& groups) {
  int R = 0;
  for (const VerifyGroup& gr : groups) {
    if (!gr.s || gr.R < 1) fatal("NemotronH: an empty verify group");
    Seq& s = *gr.s;
    if (gr.pos + gr.R >= s.cap) fatal("context full (%d)", s.cap);
    if (!merge_flush_) encode_flush(e, s);
    else if (s.n_pend > 0 && s.pend_pos + s.n_pend != gr.pos)
      fatal("NemotronH: verify at %d after pending rows at %d", gr.pos, s.pend_pos);
    R += gr.R;
  }
  if (groups.empty() || R > kMaxRows) fatal("NemotronH: %d rows in one pass", R);
  const bool dyn = dyn_rows_;
  dyn_rows_ = false;  // every row of every group is live
  const size_t H = size_t(cfg_.hidden), IP = size_t(cfg_.in_proj()), DI = size_t(cfg_.d_inner);
  const size_t QW = size_t(std::max(1, cfg_.hq * cfg_.hd)), KW = size_t(std::max(1, cfg_.hkv * cfg_.hd));
  if ((IP * 4) % 16) fatal("NemotronH: in_proj width %zu", IP);
  std::vector<size_t> r0(groups.size());
  for (size_t g = 0, r = 0; g < groups.size(); ++g) {
    r0[g] = r;
    r += size_t(groups[g].R);
  }
  auto rows = [&](const Buffer& b, size_t g, size_t w) {
    return b.view(r0[g] * w * 4, size_t(groups[g].R) * w * 4);
  };
  // (with a grouped encoder each stage's per-sequence dispatches run side by side: disjoint rows of the shared
  // buffers, their own sequence's state and pending rows, their own attention partials)
  e.group_begin();
  for (size_t g = 0; g < groups.size(); ++g)
    enc_embed(e, groups[g].s->tok, groups[g].pos, groups[g].R, rows(x_, g, H));
  e.group_end();
  const bool gm = gemm_rows_ && R >= 2;
  const size_t half = size_t(kMaxRows) * IP * 4;
  // the drafters' captures of layer i's input (i = layers: the trunk's output), by row, into each group's own hook
  auto capture = [&](int i) {
    e.group_begin();
    for (size_t g = 0; g < groups.size(); ++g) {
      const NhCapture* cap = groups[g].cap;
      if (!cap || i >= int(cap->slot.size()) || cap->slot[size_t(i)] < 0) continue;
      struct {
        uint32_t D, ld, slot, dst_row0, src_row0;
      } p{uint32_t(H), cap->ld, uint32_t(cap->slot[size_t(i)]), uint32_t(groups[g].cap_row0), uint32_t(r0[g])};
      e.use(pipe("nh_capture"));
      e.buf(0, x_);
      e.buf(1, cap->feat);
      e.bytes(2, &p, sizeof p);
      e.dispatch(uint32_t(groups[g].R), uint32_t((H + 255) / 256), 1, 256);
    }
    e.group_end();
  };
  for (int i = 0; i < cfg_.layers; ++i) {
    const NhLayer& L = layers_[size_t(i)];
    capture(i);
    if (!gm) enc_norm(e, x_, L.norm, xn_, R, cfg_.hidden);
    else if (L.kind == 'E') enc_norm2(e, x_, L.norm, xn_, xnh_, R, cfg_.hidden);
    else enc_norm_half(e, x_, L.norm, xnh_, R, cfg_.hidden);
    if (L.kind == 'M') {
      if (gm) enc_gemm(e, L.in, xnh_, zx_, R, 0);
      else enc_mv(e, L.in, xn_, zx_, R, 0);
      for (int part = 0; part < 3; ++part) {  // every sequence's copy into its pending rows, then scans, then rings
        e.group_begin();
        for (size_t g = 0; g < groups.size(); ++g) {
          Seq& s = *groups[g].s;
          const Buffer pb = s.pend_qkv[size_t(i)];
          const Buffer zx = pb.view(size_t(1 - s.pend_half) * half, half);
          if (part == 0) {
            enc_copy(e, rows(zx_, g, IP), zx.view(0, size_t(groups[g].R) * IP * 4));
          } else if (s.n_pend > 0 && merge_flush_) {
            const Buffer zp = pb.view(size_t(s.pend_half) * half, half);
            enc_ssm(e, s, i, zx, groups[g].R, groups[g].pos, false, rows(y_, g, DI), &zp, s.n_pend, part);
          } else {
            enc_ssm(e, s, i, zx, groups[g].R, groups[g].pos, false, rows(y_, g, DI), nullptr, 0, part);
          }
        }
        e.group_end();
      }
      const GnormP gp{uint32_t(cfg_.d_inner), uint32_t(cfg_.d_inner / cfg_.n_group), cfg_.eps};
      e.use(pipe(gm ? "nh_gnorm_half" : "nh_gnorm"));
      e.buf(0, y_);
      e.buf(1, L.gnorm);
      e.buf(2, gm ? xnh_ : yn_);
      e.val(3, gp);
      e.dispatch(uint32_t(R * cfg_.n_group), 1, 1, 256);
      if (gm) enc_gemm(e, L.out, xnh_, x_, R, 1);
      else enc_mv(e, L.out, yn_, x_, R, 1);
    } else if (L.kind == 'A') {
      e.group_begin();
      if (gm) {
        enc_gemm(e, L.q, xnh_, q_, R, 0);
        enc_gemm(e, L.k, xnh_, k_, R, 0);
        enc_gemm(e, L.v, xnh_, v_, R, 0);
      } else {
        enc_mv(e, L.q, xn_, q_, R, 0);
        enc_mv(e, L.k, xn_, k_, R, 0);
        enc_mv(e, L.v, xn_, v_, R, 0);
      }
      e.group_end();
      std::vector<AttnP> ap(groups.size());
      std::vector<size_t> pbase(groups.size());
      for (size_t g = 0, base = 0; g < groups.size(); ++g) {
        const int pos0 = groups[g].pos, Rg = groups[g].R;
        const int L_max = pos0 + Rg;
        uint32_t chunk = attn_chunk_;
        while ((uint32_t(L_max) + chunk - 1) / chunk > uint32_t(attn_max_chunks_)) chunk *= 2;
        const uint32_t nchunks = (uint32_t(L_max) + chunk - 1) / chunk;
        ap[g] = AttnP{uint32_t(Rg), uint32_t(pos0), uint32_t(groups[g].s->cap), uint32_t(cfg_.hq), uint32_t(cfg_.hkv),
                      chunk, nchunks, 1.0f / std::sqrt(float(cfg_.hd))};
        pbase[g] = base;  // its partials: rows (r * Hq + h) * nchunks + c of a region of its own
        base += size_t(Rg) * size_t(cfg_.hq) * nchunks;
      }
      auto part = [&](const Buffer& b, size_t g, size_t w) {
        return b.view(pbase[g] * w * 4, size_t(groups[g].R) * size_t(cfg_.hq) * ap[g].nchunks * w * 4);
      };
      e.group_begin();
      for (size_t g = 0; g < groups.size(); ++g) {
        Seq& s = *groups[g].s;
        const KvP kp{uint32_t(groups[g].R), uint32_t(groups[g].pos), uint32_t(s.cap), uint32_t(cfg_.hkv),
                     uint32_t(cfg_.hd)};
        e.use(pipe("nh_kv_write"));
        e.buf(0, rows(k_, g, KW));
        e.buf(1, rows(v_, g, KW));
        e.buf(2, s.kc[size_t(i)]);
        e.buf(3, s.vc[size_t(i)]);
        e.val(4, kp);
        e.dispatch(uint32_t(groups[g].R), uint32_t(2 * cfg_.hkv), 1, uint32_t(cfg_.hd));
      }
      e.group_end();
      e.group_begin();
      for (size_t g = 0; g < groups.size(); ++g) {
        Seq& s = *groups[g].s;
        e.use(pipe("nh_attn_partial"));
        e.buf(0, rows(q_, g, QW));
        e.buf(1, s.kc[size_t(i)]);
        e.buf(2, s.vc[size_t(i)]);
        e.buf(3, part(part_o_, g, size_t(cfg_.hd)));
        e.buf(4, part(part_ml_, g, 2));
        e.val(5, ap[g]);
        e.dispatch(ap[g].nchunks, uint32_t(cfg_.hkv), uint32_t(groups[g].R), uint32_t(32 * (cfg_.hq / cfg_.hkv)));
      }
      e.group_end();
      e.group_begin();
      for (size_t g = 0; g < groups.size(); ++g) {
        e.use(pipe(gm ? "nh_attn_reduce_h" : "nh_attn_reduce"));
        e.buf(0, part(part_o_, g, size_t(cfg_.hd)));
        e.buf(1, part(part_ml_, g, 2));
        e.buf(2, gm ? aoh_.view(r0[g] * QW * 2, size_t(groups[g].R) * QW * 2) : rows(ao_, g, QW));
        e.val(3, ap[g]);
        e.dispatch(uint32_t(cfg_.hq), uint32_t(groups[g].R), 1, uint32_t(cfg_.hd));
      }
      e.group_end();
      if (gm) {
        enc_gemm(e, L.o, aoh_, x_, R, 1);
      } else {
        enc_mv(e, L.o, ao_, x_, R, 1);
      }
    } else if (L.kind == 'D') {
      enc_mlp(e, L, R);
    } else {
      enc_moe(e, L, R);
    }
  }
  capture(cfg_.layers);  // (a drafter reading the trunk's output: DSpark)
  for (const VerifyGroup& gr : groups) {
    gr.s->n_pend = gr.R;  // until commit_rows() says how many were accepted
    gr.s->pend_tree = false;
    gr.s->pend_pos = gr.pos;
    gr.s->pend_half = 1 - gr.s->pend_half;
  }
  enc_head(e, R, logits_, size_t(cfg_.vocab));
  const size_t V = size_t(cfg_.vocab);
  for (size_t g = 0; g < groups.size(); ++g)
    for (int j = 0; j < groups[g].R; ++j) {
      const size_t r = r0[g] + size_t(j);
      if (groups[g].tok_out)
        enc_sample(e, logits_.view(r * V * 4, V * 4), groups[g].s->tok, groups[g].pos + 1 + j, groups[g].sp,
                   uint32_t(groups[g].pos + j));
      else
        enc_sample(e, logits_.view(r * V * 4, V * 4), samp_, int(r), groups[g].sp, uint32_t(groups[g].pos + j));
    }
  dyn_rows_ = dyn;
}

void NemotronH::encode_accept(Encoder& e, Seq& s, int pos, int R, int K, const SamplingParams& sp) {
  // the R rows' samples in two passes (topk_stage1_rows, sample_stage2_rows: each row's candidates and RNG step pos + r
  // as enc_sample's two passes per row give them)
  const uint32_t Kc = sp.temperature > 0 ? uint32_t(std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32)) : 1u;
  const TopkStageP ts{uint32_t(cfg_.vocab), Kc, 2048};
  const uint32_t ntg = (uint32_t(cfg_.vocab) + 2047) / 2048;
  // (the M5: on simdgroup reductions, the same candidates and samples: topk_stage1_rows_sg, sample_stage2_rows_sg)
  const bool sg = sample_sg_ && ntg <= 64 && Kc <= 32;
  e.use(pipe(sg ? "topk_stage1_rows_sg" : "topk_stage1_rows"));
  e.buf(0, logits_);
  e.buf(1, cand_v_);
  e.buf(2, cand_i_);
  e.val(3, ts);
  e.dispatch(ntg, uint32_t(R), 1, 256);
  const SampleRowsP p2{ntg * Kc, Kc, sp.temperature, sp.top_p, sp.seed, uint32_t(pos), 0, 0, 0, 0};
  e.use(pipe(sg ? "sample_stage2_rows_sg" : "sample_stage2_rows"));
  e.buf(0, cand_v_);
  e.buf(1, cand_i_);
  e.buf(2, samp_);
  e.val(3, p2);
  e.buf(4, cand_i_);  // (no table)
  e.dispatch(uint32_t(R), 1, 1, sg ? 256 : 1024);
  struct {
    uint32_t P, K;
  } ap{uint32_t(pos), uint32_t(K)};
  e.use(pipe("nh_accept"));
  e.buf(0, samp_);
  e.buf(1, s.tok);
  e.buf(2, acc_);
  e.bytes(3, &ap, sizeof ap);
  e.buf(4, live());
  e.dispatch(1, 1, 1, 1);
}

void NemotronH::encode_accept_tree(Encoder& e, Seq& s, int pos, const SamplingParams& sp) {
  // row r draws at position pos + depth(r): the tree buffer's depths
  for (int r = 0; r < kMaxRows; ++r)
    enc_sample(e, logits_.view(size_t(r) * cfg_.vocab * 4, size_t(cfg_.vocab) * 4), samp_, r, sp, uint32_t(pos), &tree_,
               size_t(kf::kDep) * 4);
  const struct {
    uint32_t P, R, gst, pad;
  } ap{uint32_t(pos), uint32_t(kMaxRows), 0u, 0u};
  e.use(pipe("kf_accept"));  // (shared: src/kernels/kflash.h)
  e.buf(0, samp_);
  e.buf(1, s.tok);
  e.buf(2, tree_);
  e.buf(3, acc_);
  e.bytes(4, &ap, sizeof ap);
  e.buf(5, live());
  e.buf(6, s.pend_path);
  e.buf(7, live_all_);  // (no drafter state written: gst 0)
  e.dispatch(1, 1, 1, 32);
  // the accepted path's K/V into place (positions pos + 1 .. pos + a)
  const KvP kp{uint32_t(kMaxRows), uint32_t(pos), uint32_t(s.cap), uint32_t(cfg_.hkv), uint32_t(cfg_.hd)};
  for (int i = 0; i < cfg_.layers; ++i) {
    if (cfg_.kind[size_t(i)] != 'A') continue;
    e.use(pipe("nh_kv_tree_compact"));
    e.buf(0, s.kc[size_t(i)]);
    e.buf(1, s.vc[size_t(i)]);
    e.buf(2, tree_);
    e.val(3, kp);
    e.dispatch(uint32_t(2 * cfg_.hkv), 1, 1, uint32_t(cfg_.hd));
  }
}

void NemotronH::encode_logits_rows(Encoder& e, Seq& s, int pos0, int T, const Buffer& out) {
  encode_flush(e, s);
  if (T <= kMaxRows) {
    enc_rows(e, s, pos0, T, true);
    enc_head(e, T, out, size_t(cfg_.vocab));
    return;
  }
  if (ane_splits(T)) {
    ane_->begin_chunk(e, T);
    ane_chunk_ = true;
    ++ane_stats_.chunks;
  }
  enc_prefill_gemm(e, s, pos0, T);
  // final norm of every row into half, the LM head as a GEMM
  const HalfP hp{uint32_t(cfg_.hidden), 1u, cfg_.eps, uint32_t(cfg_.hidden)};
  e.use(pipe("nh_rows_half"));
  e.buf(0, px_);
  e.buf(1, out_norm_);
  e.buf(2, pxh_);
  e.val(3, hp);
  e.dispatch(uint32_t(T), 1, 1, 1024);
  enc_gemm(e, head_, pxh_, out, T, 0, uint32_t(cfg_.vocab));
}

// ---- prefill with GEMMs ----
void NemotronH::alloc_prefill(int T) {
  if (pf_rows_ >= T) return;
  Gpu& g = Gpu::get();
  if (pcv_ && pcv_.mtl != pout_.mtl) g.release(pcv_);
  if (psb_ && psb_.mtl != pout_.mtl) g.release(psb_);
  pcv_ = psb_ = Buffer{};
  for (Buffer* b : {&px_, &pxn_, &pxh_, &pzx_, &py_, &pyh_, &pq_, &pk_, &pv_, &poh_, &prl_, &peid_, &pew_, &plat_, &ph_,
                    &pout_, &plo_, &psh_, &psorted_, &ptiles_, &pntiles_, &prank_})
    g.release(*b);
  const size_t R = size_t(tmax_), H = size_t(cfg_.hidden), k = size_t(std::max(1, cfg_.max_k));
  const size_t P = R * k;
  px_ = g.alloc(R * H * 4, "nh_pf_x");
  pxn_ = g.alloc(R * H * 4, "nh_pf_xn");
  pxh_ = g.alloc(R * std::max(H, size_t(cfg_.d_inner)) * 2, "nh_pf_xh");
  pzx_ = g.alloc(R * size_t(cfg_.in_proj()) * 4, "nh_pf_zx");
  py_ = g.alloc(R * size_t(cfg_.d_inner) * 4, "nh_pf_y");
  pyh_ = g.alloc(R * size_t(cfg_.d_inner) * 2, "nh_pf_yh");
  pq_ = g.alloc(R * size_t(std::max(1, cfg_.hq * cfg_.hd)) * 4, "nh_pf_q");
  pk_ = g.alloc(R * size_t(std::max(1, cfg_.hkv * cfg_.hd)) * 4, "nh_pf_k");
  pv_ = g.alloc(R * size_t(std::max(1, cfg_.hkv * cfg_.hd)) * 4, "nh_pf_v");
  poh_ = g.alloc(R * size_t(std::max(1, cfg_.hq * cfg_.hd)) * 2, "nh_pf_oh");
  prl_ = g.alloc(R * size_t(std::max(1, cfg_.max_E)) * 4, "nh_pf_router");
  peid_ = g.alloc(P * 4, "nh_pf_eid");
  pew_ = g.alloc(P * 4, "nh_pf_ew");
  plat_ = g.alloc(R * size_t(std::max(1, cfg_.latent)) * 2, "nh_pf_lat");
  ph_ = g.alloc(P * size_t(std::max(1, cfg_.max_I)) * 2, "nh_pf_h");
  pout_ = g.alloc(P * size_t(std::max(1, cfg_.latent)) * 4, "nh_pf_out");
  plo_ = g.alloc(R * size_t(std::max(1, cfg_.latent)) * 2, "nh_pf_lo");
  psh_ = g.alloc(R * size_t(std::max(1, cfg_.ffn_I())) * 2, "nh_pf_sh");
  psorted_ = g.alloc(P * 4, "nh_pf_sorted");
  prank_ = g.alloc(P * 4, "nh_pf_rank");
  ptiles_ = g.alloc((P / 16 + size_t(cfg_.max_E) + 2) * 16, "nh_pf_tiles");
  // nh_ssm_conv's rows (M5): in pout_ when it is large enough (the experts' outputs, free during a Mamba layer)
  if (tg_) {
    const size_t cvb = R * (size_t(cfg_.conv_ch()) + 2 * size_t(cfg_.n_head)) * 4;
    pcv_ = pout_.size >= cvb ? pout_ : g.alloc(cvb, "nh_pf_conv");
    // nh_ssd's per-chunk M and decays: the rest of pout_ if it is large enough
    const size_t nch = (R + 31) / 32, sbb = size_t(cfg_.n_head) * nch * (32 * 32 + 64) * 4;
    const size_t off = (cvb + 255) / 256 * 256;
    psb_ = pout_.size >= off + sbb ? pout_.view(off, sbb) : g.alloc(sbb, "nh_pf_ssd_states");
  }
  pntiles_ = g.alloc(16, "nh_pf_ntiles");
  pf_rows_ = tmax_;
}

void NemotronH::enc_gemm(Encoder& e, const NhMat& W, const Buffer& xh, const Buffer& y, int M, int epi, uint32_t y_ld,
                         uint32_t x_ld) {
  if (W.fmt == NhFmt::F32 || W.E != 1) fatal("enc_gemm: unsupported matrix");
  // FP8 up to 32 rows: the x8 kernels (nh_x4's designs; 9-32 rows share x4's split-K buffers); a dense model's
  // K % 128 != 0 (the Nano 4B's hidden 3136: Mamba in_proj, MLP up) on their tail variants (*_t: the last K % 128
  // inputs as one more, masked step) instead of nh_gemm_fp8_m16 / m32 (M1 Pro, 2-8 rows: in_proj 0.337 vs 0.815 ms,
  // MLP up 0.245 vs 0.624, as one row's nh_x8_mv takes 0.331 / 0.274; 9-24 rows 21-27% faster, 25-32 within 2%)
  const bool x8_tail = dense_ && W.K % 128 != 0 && W.K % 32 == 0;
  if (W.fmt == NhFmt::FP8 && x8_ && M <= 32 && (W.K % 128 == 0 || x8_tail) && (x_ld ? x_ld : W.K) % 8 == 0) {
    const V8P p{W.N, W.K, x_ld ? x_ld : W.K, y_ld ? y_ld : W.N, uint32_t(M), W.gs, uint32_t(epi)};
    const uint32_t tb = M <= 8 ? 1 : M <= 16 ? 2 : M <= 24 ? 3 : 4, sk = split_ ? 1 : W.K <= 8192 ? 4 : 8,
                   tiles = (W.N + 63) / 64;
    if (tb == 1 || (size_t(sk) * tiles * 2048 * tb <= x4part_.size && tiles * 4 <= x4cnt_.size)) {
      const bool n1 = tb == 1 && v8_n1_ && W.K % 128 == 0;
      std::string name = tb == 1   ? (n1 ? "nh_x8_v8_n1" : "nh_x8_v8")
                         : tb == 2 ? "nh_x8_mm16"
                         : tb == 3 ? "nh_x8_mm24"
                                   : "nh_x8_mm32";
      if (W.K % 128) name += "_t";
      e.use(pipe(name));
      e.buf(0, xh);
      e.buf(1, W.q);
      e.buf(2, W.q);
      e.buf(3, y);
      e.val(4, p);
      if (tb == 1) {
        e.dispatch(n1 ? (W.N + 7) / 8 : (W.N + 15) / 16, 1, 1, 256);
      } else {
        e.buf(5, x4part_);
        e.buf(6, x4cnt_);
        e.dispatch(sk, tiles, 1, 128);
      }
      return;
    }
  }
  // NVFP4 up to 8 rows on the M5's matrix units (v8t_): nh_x4t_v8_l0, 16 weight rows per threadgroup of 8 simdgroups
  // taking K's 64-input groups in turn. tools/nh_v8t_bench (us per call, weights rotating out of the cache, against
  // nh_x4_v8_l0): LM head 131072x2688 379 vs 467, shared expert up 3712x2688 14.5 vs 18.6, down 2688x3712 15.6 vs
  // 19.0, the drafter's 6144x2688 21.4 vs 29.3 (4 simdgroups 401 / 15.2 / 16.4 / 23.3). NI_NH_V8T_KS (1-8) overrides.
  if (W.fmt == NhFmt::FP4 && v8t_ && v8_ && x4_ && M <= 8 && !W.l2 && W.K % 64 == 0) {
    static const int ks_env = getenv("NI_NH_V8T_KS") ? atoi(getenv("NI_NH_V8T_KS")) : 8;
    const V8P p{W.N, W.K, x_ld ? x_ld : W.K, y_ld ? y_ld : W.N, uint32_t(M), W.gs, uint32_t(epi)};
    e.use(pipe("nh_x4t_v8_l0"));
    e.buf(0, xh);
    e.buf(1, W.q);
    e.buf(2, W.s);
    e.buf(3, y);
    e.val(4, p);
    e.dispatch((W.N + 15) / 16, 1, 1, 32 * uint32_t(std::clamp(ks_env, 1, 8)));
    return;
  }
  // NVFP4 up to 32 rows: the x4 kernels; up to 8 rows 8 simdgroups split K (in_proj 0.207 vs 0.260 ms, out_proj
  // 0.097 vs 0.141), 9-32 rows SK parts of K summed in a fixed order by the last part to arrive (in_proj 32 rows
  // 0.72 vs 0.91 ms, out_proj 0.34 vs 0.56). In the matrix's layout; a dense model's K % 128 != 0 (the Nano 9B v2's
  // MLP down, K 15680) on their tail variants (*_t: the last K % 128 inputs as one more, masked step)
  const bool x4_tail = dense_ && W.K % 128 != 0 && W.K % 32 == 0;
  if (W.fmt == NhFmt::FP4 && x4_ && M <= 32 && (W.K % 128 == 0 || x4_tail) && (x_ld ? x_ld : W.K) % 8 == 0 &&
      (M > 8 || v8_)) {
    const V8P p{W.N, W.K, x_ld ? x_ld : W.K, y_ld ? y_ld : W.N, uint32_t(M), W.gs, uint32_t(epi)};
    const uint32_t tb = M <= 8 ? 1 : M <= 16 ? 2 : M <= 24 ? 3 : 4, sk = split_ ? 1 : W.K <= 8192 ? 4 : 8,
                   tiles = (W.N + 63) / 64;
    if (tb == 1 || (size_t(sk) * tiles * 2048 * tb <= x4part_.size && tiles * 4 <= x4cnt_.size)) {
      std::string name = tb == 1 ? "nh_x4_v8" : tb == 2 ? "nh_x4_mm16" : tb == 3 ? "nh_x4_mm24" : "nh_x4_mm32";
      if (!W.l2) name += "_l0";
      if (W.K % 128) name += "_t";
      e.use(pipe(name));
      e.buf(0, xh);
      e.buf(1, W.q);
      e.buf(2, W.s);
      e.buf(3, y);
      e.val(4, p);
      if (tb == 1) {
        e.dispatch((W.N + 15) / 16, 1, 1, 256);
      } else {
        e.buf(5, x4part_);
        e.buf(6, x4cnt_);
        e.dispatch(sk, tiles, 1, 128);
      }
      return;
    }
  }
  // IQ2 up to 8 rows on the M5's matrix units (v8t_; Bolt's shared expert in verify): nh_x2t_v8, nh_x4t_v8_l0's design
  // with the IQ2_XXS decode. tools/nh_x2t_bench (us per call, weights rotating out of the cache, against nh_x2_v8):
  // up 5376x4096 24.8 vs 35.4, down 4096x5376 26.6 vs 41.6 (4 simdgroups; 2: 25.2 / 28.5, 8: 31.5 / 29.8); Bolt's
  // greedy harness decode 29.5 -> 27.8 ms per cycle at the same 4.48 tokens per cycle. NI_NH_X2T_KS (1-8) overrides.
  if (W.fmt == NhFmt::IQ2 && v8t_ && v8_ && x2_ && M <= 8 && W.K % 256 == 0) {
    static const int ks_env = getenv("NI_NH_X2T_KS") ? atoi(getenv("NI_NH_X2T_KS")) : 4;
    const V8P p{W.N, W.K, x_ld ? x_ld : W.K, y_ld ? y_ld : W.N, uint32_t(M), W.gs, uint32_t(epi)};
    e.use(pipe("nh_x2t_v8"));
    e.buf(0, xh);
    e.buf(1, W.q);
    e.buf(2, W.s);
    e.buf(3, y);
    e.val(4, p);
    e.dispatch((W.N + 15) / 16, 1, 1, 32 * uint32_t(std::clamp(ks_env, 1, 8)));
    return;
  }
  // IQ2 up to 32 rows (the shared expert in verify): the x2 kernels, one simdgroup per 8 rows on 8x8 MMAs with the
  // weights as fp16 subnormals (8 rows: up 0.083 vs 0.107 ms, down 0.090 vs 0.111; 32 rows 0.25 vs 0.35)
  if (W.fmt == NhFmt::IQ2 && x2_ && M <= 32 && W.K % 256 == 0 && (M > 8 || v8_)) {
    const V8P p{W.N, W.K, x_ld ? x_ld : W.K, y_ld ? y_ld : W.N, uint32_t(M), W.gs, uint32_t(epi)};
    e.use(pipe(M <= 8 ? "nh_x2_v8" : M <= 16 ? "nh_x2_v16" : "nh_x2_v32"));
    e.buf(0, xh);
    e.buf(1, W.q);
    e.buf(2, W.s);
    e.buf(3, y);
    e.val(4, p);
    e.dispatch((W.N + 7) / 8, 1, 1, 32);
    return;
  }
  // BF16 up to 8 rows (Lightning's attention in verify): nh_xb_v8, the x8 design with the weights converted to half
  // as nh_gemm_bf16w converts them (a verify's 24 projections 0.70 against 2.47 ms on nh_gemm_bf16w_m16); a dense
  // model's K % 128 != 0 (the Nano 4B's hidden 3136) on its tail variant nh_xb_v8_t
  if (W.fmt == NhFmt::BF16 && v8_ && M <= 8 && (W.K % 128 == 0 || (dense_ && W.K % 32 == 0)) &&
      (x_ld ? x_ld : W.K) % 8 == 0) {
    const V8P p{W.N, W.K, x_ld ? x_ld : W.K, y_ld ? y_ld : W.N, uint32_t(M), W.gs, uint32_t(epi)};
    const bool n1 = v8_n1_ && W.K % 128 == 0;
    e.use(pipe(W.K % 128 ? "nh_xb_v8_t" : n1 ? "nh_xb_v8_n1" : "nh_xb_v8"));
    e.buf(0, xh);
    e.buf(1, W.q);
    e.buf(2, W.q);
    e.buf(3, y);
    e.val(4, p);
    e.dispatch(n1 ? (W.N + 7) / 8 : (W.N + 15) / 16, 1, 1, 256);
    return;
  }
  // up to 8 rows (verify, the drafter's block): the 8-row MMA kernels (weights decoded into registers; NVFP4 in
  // layout L0)
  if (M <= 8 && v8_ && W.fmt != NhFmt::FP8 && W.fmt != NhFmt::BF16 && !W.l2) {
    // split-K: 4 simdgroups share 8 NB rows and split K (measured 6-35% over a block of rows per simdgroup); 32 rows
    // (NB 4) for 4k-32k outputs (FP4 in_proj 0.263 vs 0.315 ms, out_proj 0.139 vs 0.153, IQ2 shared expert 0.105 vs
    // 0.114, Q8 4096-row 0.077 vs 0.085), 16 for the 1024-row latent and the 131k-row head
    const char* fm = W.fmt == NhFmt::Q8 ? "q8" : W.fmt == NhFmt::FP4 ? "fp4" : "iq2";
    const uint32_t nb = W.N >= 4096 && W.N <= 32768 ? 4 : 2, rows_tg = 8 * nb;
    const V8P p{W.N, W.K, x_ld ? x_ld : W.K, y_ld ? y_ld : W.N, uint32_t(M), W.gs, uint32_t(epi)};
    e.use(pipe(std::string("nh_v8_") + fm + "_s" + std::to_string(nb)));
    e.buf(0, xh);
    e.buf(1, W.q);
    e.buf(2, W.s);
    e.buf(3, y);
    e.val(4, p);
    e.buf(8, iq2tab_);  // the IQ2 grid as halves
    e.dispatch((W.N + rows_tg - 1) / rows_tg, 1, 1, 128);
    return;
  }
  // rows per tile: 16 / 32 for short inputs (less padding), else 64. Over 32 rows on the M5 (tg_): the same tiles on
  // the matrix units (nh_gemm_t_*: a 2048-row chunk's Mamba in_proj 1.90-1.95 against 7.48 ms, out_proj 0.76-0.77 /
  // 3.07, the shared expert's up 0.78-0.80 / 2.82-2.83, attention q 0.76-0.82 / 3.13-3.26; tools/nh_pf_bench)
  const int bm = M <= 16 ? 16 : M <= 32 ? 32 : 64;
  const bool tg = tg_ && M > 32 && (W.fmt == NhFmt::FP8 || W.fmt == NhFmt::BF16 || W.fmt == NhFmt::FP4) &&
                  W.K % 32 == 0 && (x_ld ? x_ld : W.K) % 8 == 0;
  std::string name = std::string(W.fmt == NhFmt::Q8    ? "nh_gemm_q8"
                                       : W.fmt == NhFmt::FP4 ? (W.l2 ? "nh_gemm_fp4l2" : "nh_gemm_fp4")
                                       : W.fmt == NhFmt::FP8 ? "nh_gemm_fp8"
                                       : W.fmt == NhFmt::BF16 ? "nh_gemm_bf16w"
                                                              : "nh_gemm_iq2") +
                           "_m" + std::to_string(bm);
  if (tg) name.insert(8, "t_");  // nh_gemm_t_<fmt>_m64
  const GemmP p{uint32_t(M), W.N, W.K, x_ld ? x_ld : W.K, y_ld ? y_ld : W.N, W.gs, uint32_t(epi), 0, 0, 1};
  e.use(pipe(name));
  e.buf(0, xh);
  e.buf(1, W.q);
  e.buf(2, W.s);
  e.buf(3, y);
  e.val(4, p);
  e.buf(8, iq2tab_);  // the IQ2 grid as halves
  e.dispatch((W.N + 63) / 64, (uint32_t(M) + uint32_t(bm) - 1) / uint32_t(bm), 1, 128);
}

void NemotronH::enc_prefill_gemm(Encoder& e, Seq& s, int pos0, int T) {
  if (pos0 + T >= s.cap) fatal("context full (%d)", s.cap);
  alloc_prefill(T);
  const uint32_t H = uint32_t(cfg_.hidden);
  enc_embed(e, s.tok, pos0, T, px_);
  if (s.embeds) vision::enc_embed_runs(e, *s.embeds, pos0, T, px_, cfg_.hidden);  // image rows
  for (int i = 0; i < cfg_.layers; ++i) {
    const NhLayer& L = layers_[size_t(i)];
    if (cap_) enc_capture(e, i, px_, pos0, T);
    const bool moe = L.kind == 'E';
    if (moe) enc_norm(e, px_, L.norm, pxn_, T, cfg_.hidden);  // the router reads the fp32 normalized rows
    const HalfP hp{H, moe ? 0u : 1u, cfg_.eps, H};
    e.use(pipe("nh_rows_half"));
    e.buf(0, moe ? pxn_ : px_);
    e.buf(1, L.norm);
    e.buf(2, pxh_);
    e.val(3, hp);
    e.dispatch(uint32_t(T), 1, 1, 1024);
    const bool split = ane_chunk_;
    if (L.kind == 'M') {
      if (split && ane_->gpu_in() < uint32_t(cfg_.in_proj())) {
        // in_proj: the ANE's rows while the GPU computes rows [gpu_in0, gpu_in0 + gpu_in)
        ane_->in_launch(e, i, pxh_, T);
        NhMat W = L.in;
        const uint32_t r0 = ane_->gpu_in0();
        W.N = ane_->gpu_in();
        W.q = W.q.view(size_t(r0) * W.q_row, size_t(W.N) * W.q_row);
        W.s = W.s.view(size_t(r0) * W.s_row, size_t(W.N) * W.s_row);
        if (W.N) enc_gemm(e, W, pxh_, pzx_.view(size_t(r0) * 4, pzx_.size - size_t(r0) * 4), T, 0, uint32_t(cfg_.in_proj()));
        ane_->in_join(e, i, pzx_, T);
      } else {
        enc_gemm(e, L.in, pxh_, pzx_, T, 0);
      }
      enc_ssm(e, s, i, pzx_, T, pos0, true, py_);
      const GnormP gp{uint32_t(cfg_.d_inner), uint32_t(cfg_.d_inner / cfg_.n_group), cfg_.eps};
      e.use(pipe("nh_gnorm_half"));
      e.buf(0, py_);
      e.buf(1, L.gnorm);
      e.buf(2, pyh_);
      e.val(3, gp);
      e.dispatch(uint32_t(T * cfg_.n_group), 1, 1, 256);
      if (split && ane_->gpu_out() < uint32_t(cfg_.hidden)) {
        ane_->out_launch(e, i, pyh_, T);
        NhMat W = L.out;
        W.N = ane_->gpu_out();
        if (W.N) enc_gemm(e, W, pyh_, px_, T, 1, uint32_t(cfg_.hidden));
        ane_->out_join(e, i, px_, T);
      } else {
        enc_gemm(e, L.out, pyh_, px_, T, 1);
      }
    } else if (L.kind == 'A') {
      enc_gemm(e, L.q, pxh_, pq_, T, 0);
      enc_gemm(e, L.k, pxh_, pk_, T, 0);
      enc_gemm(e, L.v, pxh_, pv_, T, 0);
      const KvP kp{uint32_t(T), uint32_t(pos0), uint32_t(s.cap), uint32_t(cfg_.hkv), uint32_t(cfg_.hd)};
      e.use(pipe("nh_kv_write"));
      e.buf(0, pk_);
      e.buf(1, pv_);
      e.buf(2, s.kc[size_t(i)]);
      e.buf(3, s.vc[size_t(i)]);
      e.val(4, kp);
      e.dispatch(uint32_t(T), uint32_t(2 * cfg_.hkv), 1, uint32_t(cfg_.hd));
      // heads per threadgroup of nh_attn_pf2: 4, or 5 (the Nano models' 40 query heads over 8 KV heads)
      const uint32_t pf_nsg = (cfg_.hq / cfg_.hkv) % 4 == 0 ? 4 : (cfg_.hq / cfg_.hkv) % 5 == 0 ? 5 : 0;
      // M5 (tg_), 16 query heads per KV head of 128 dims (Lightning): nh_attn_t_s4 on the matrix units, a token's 16
      // heads per simdgroup, 4 tokens per threadgroup (2048 rows after 14336 positions: 12.8 against nh_attn_pf2's
      // 37.1 ms, tools/nh_pf_bench)
      const bool att_t = tg_ && cfg_.hq == 16 * cfg_.hkv && cfg_.hd == 128;
      if (att_t || pf_nsg) {
        // 4 heads per threadgroup (nh_attn_pf2: the same values as nh_attn_prefill); a pass longer than ~25 ms (2048
        // rows after 14336 positions: ~330 ms on the M5 Max's 40 cores by nh_attn_pf2's estimate, nh_attn_t's from
        // its measured 12.8 ms) in pieces of equal work over the row blocks, each in a command buffer of its own
        // (outside a Neural Engine split chunk)
        const uint32_t bq = att_t ? 4 : 8, nb = (uint32_t(T) + bq - 1) / bq;
        const double work = double(T) * (double(pos0) + T / 2.0) * cfg_.hq;  // row x key x head
        const double est_ms = work * (att_t ? 1.25e-8 : 3.3e-7) * 40.0 / std::max(1, Gpu::get().core_count());
        const int pieces = split ? 1 : std::clamp(int(std::ceil(est_ms / 25.0)), 1, int(nb));
        uint32_t b0 = 0;
        for (int pc = 0; pc < pieces; ++pc) {
          // row blocks [b0, b1): the work of rows [0, bq b) grows as bq b (pos0 + bq b / 2)
          uint32_t b1 = nb;
          if (pc + 1 < pieces) {
            const double target = (pc + 1) * (double(T) * (double(pos0) + T / 2.0)) / pieces;
            b1 = b0;
            while (b1 < nb && double(bq) * b1 * (double(pos0) + bq * b1 / 2.0) < target) ++b1;
            b1 = std::max(b1, b0 + 1);
          }
          if (b1 > nb) b1 = nb;
          if (b1 <= b0) break;
          const struct {
            uint32_t T, pos0, cap, Hq, Hkv;
            float scale;
            uint32_t qb0;
          } ap{uint32_t(T), uint32_t(pos0), uint32_t(s.cap), uint32_t(cfg_.hq), uint32_t(cfg_.hkv),
               1.0f / std::sqrt(float(cfg_.hd)), b0};
          if (pc > 0) e.split();
          e.use(pipe(att_t ? "nh_attn_t_s4" : pf_nsg == 4 ? "nh_attn_pf2_q1_h4_k32" : "nh_attn_pf2_q1_h5_k32"));
          e.buf(0, pq_);
          e.buf(1, s.kc[size_t(i)]);
          e.buf(2, s.vc[size_t(i)]);
          e.buf(3, poh_);
          e.bytes(4, &ap, sizeof ap);
          if (att_t) e.dispatch(b1 - b0, uint32_t(cfg_.hkv), 1, 128);
          else e.dispatch(b1 - b0, uint32_t(cfg_.hq) / pf_nsg, 1, 32 * pf_nsg);
          b0 = b1;
        }
      } else {
        const AttnPfP ap{uint32_t(T), uint32_t(pos0), uint32_t(s.cap), uint32_t(cfg_.hq), uint32_t(cfg_.hkv),
                         1.0f / std::sqrt(float(cfg_.hd))};
        e.use(pipe("nh_attn_prefill"));
        e.buf(0, pq_);
        e.buf(1, s.kc[size_t(i)]);
        e.buf(2, s.vc[size_t(i)]);
        e.buf(3, poh_);
        e.val(4, ap);
        e.dispatch((uint32_t(T) + 31) / 32, uint32_t(cfg_.hq), 1, 128);
      }
      enc_gemm(e, L.o, poh_, px_, T, 1);
    } else if (L.kind == 'D') {
      // dense MLP (on the GPU, also in a Neural Engine split chunk): the normed rows pxh_ up with relu^2 into half
      // rows, down added into the residual
      enc_gemm(e, L.up_s, pxh_, psh_, T, 2);
      enc_gemm(e, L.down_s, psh_, px_, T, 1);
    } else {
      const uint32_t k = uint32_t(L.k_used), P = uint32_t(T) * k;
      const bool ffn_ane = split && ane_->ffn();  // the shared expert on the ANE while the GPU runs the experts
      // router (bf16 or fp32 weights, fp32 products), routing
      const GemmF32P rp{uint32_t(T), uint32_t(L.E), H, H, uint32_t(cfg_.max_E)};
      e.use(pipe(L.router.fmt == NhFmt::BF16 ? "nh_gemm_bf16" : "nh_gemm_f32"));
      e.buf(0, pxn_);
      e.buf(1, L.router.q);
      e.buf(2, prl_);
      e.val(3, rp);
      e.dispatch((uint32_t(L.E) + 31) / 32, (uint32_t(T) + 31) / 32, 1, 128);
      const RouteP rtp{uint32_t(L.E), k, cfg_.w_norm ? 1u : 0u, cfg_.w_scale, uint32_t(cfg_.max_E), uint32_t(T)};
      e.use(pipe(L.E <= 512 ? "nh_moe_route1" : "nh_moe_route"));
      e.buf(0, prl_);
      e.buf(1, L.exp_bias);
      e.buf(2, peid_);
      e.buf(3, pew_);
      e.val(4, rtp);
      e.buf(5, live_all_);
      if (L.E <= 512) e.dispatch((uint32_t(T) + 7) / 8, 1, 1, 256);
      else e.dispatch(uint32_t(T), 1, 1, 512);
      // latent, the pairs grouped by expert, routed experts, combine (without latent projections the experts read the
      // normed rows pxh_ and their sum is added into px_)
      const bool lat = bool(L.lat_up);
      if (lat) enc_gemm(e, L.lat_down, pxh_, plat_, T, 3);
      // rows per expert tile: as many as the pairs per expert make worth it
      // (NVFP4 experts: nh_x4_moe_*, tiles of 32 or 16 rows)
      // (on the M5's matrix units, tge: NVFP4 experts in tiles of 16, 32 or 64 rows on nh_gemm_t_fp4_moe_*)
      const bool fp4e = L.up_e.fmt == NhFmt::FP4 && x4_;
      const bool tge = tg_ && L.up_e.fmt == NhFmt::FP4 && L.up_e.l2 == L.down_e.l2 && L.up_e.K % 32 == 0 &&
                       L.down_e.K % 32 == 0;
      const uint32_t per = P / uint32_t(L.E), bm = per >= 40 && (!fp4e || tge) ? 64 : per >= 12 ? 32 : 16;
      const std::string ms = "_m" + std::to_string(bm);
      const BucketP bp{P, uint32_t(L.E), bm, 0};
      e.use(pipe("nh_moe_bucket"));
      e.buf(0, peid_);
      e.buf(1, psorted_);
      e.buf(2, ptiles_);
      e.buf(3, pntiles_);
      e.buf(4, prank_);
      e.val(5, bp);
      e.buf(6, live_all_);
      e.dispatch(1, 1, 1, 1024);
      // launched here, after the routing, not before the router: under a busy desktop (the UI's GPU work), a launch
      // before the router GEMM left the GPU idle 0.8-2.3 s per 2048-row chunk (other processes' frames ran in the
      // gaps; 100-190 ms each, measured with command-buffer timestamps), here 0.03-0.2 s (4k prefill 10.0 -> 8.4 s)
      if (ffn_ane) ane_->ffn_launch(e, i, pxh_, T);
      const uint32_t max_tiles = P / bm + uint32_t(L.E) + 1;
      {
        const GemmP gp{P, uint32_t(L.I), uint32_t(cfg_.latent), uint32_t(cfg_.latent), uint32_t(L.I), 1.0f, 2,
                       uint32_t(L.up_e.q_exp()), uint32_t(L.up_e.s_exp()), k};
        const bool fp4 = L.up_e.fmt == NhFmt::FP4;
        // (nh_x4_moe_* need K % 128: Lightning's down projection, K 1856, stays on nh_gemm_fp4_moe_dn)
        const bool x4m = fp4 && x4_ && L.up_e.K % 128 == 0;
        e.use(pipe(tge   ? (L.up_e.l2 ? "nh_gemm_t_fp4l2_moe_up" : "nh_gemm_t_fp4_moe_up") + ms
                   : x4m ? "nh_x4_moe_up" + ms + (L.up_e.l2 ? "_l2" : "")
                         : (fp4 ? (L.up_e.l2 ? "nh_gemm_fp4l2_moe_up" : "nh_gemm_fp4_moe_up") : "nh_gemm_iq2_moe_up") + ms));
        e.buf(0, lat ? plat_ : pxh_);
        e.buf(1, L.up_e.q);
        e.buf(2, L.up_e.s);
        e.buf(3, ph_);
        e.val(4, gp);
        e.buf(5, ptiles_);
        e.buf(6, pntiles_);
        e.buf(7, psorted_);
        e.buf(8, iq2tab_);
        if (fp4) e.buf(9, L.up_e.gse);
        e.dispatch((uint32_t(L.I) + 63) / 64, max_tiles, 1, 128);
      }
      {
        const GemmP gp{P, uint32_t(cfg_.latent), uint32_t(L.I), uint32_t(L.I), uint32_t(cfg_.latent), 1.0f, 0,
                       uint32_t(L.down_e.q_exp()), uint32_t(L.down_e.s_exp()), k};
        const bool fp4 = L.down_e.fmt == NhFmt::FP4;
        const bool x4m = fp4 && x4_ && L.down_e.K % 128 == 0;
        // (64-row tiles on the matrix units: 128 columns per threadgroup, 2048 rows 3.54-3.66 against 3.94-3.97 ms
        // with 64, tools/nh_pf_bench; the up projection's 64 columns are faster there, 3.09 against 3.57-3.63)
        const bool n128 = tge && bm == 64;
        e.use(pipe(tge   ? (L.down_e.l2 ? "nh_gemm_t_fp4l2_moe_dn" : "nh_gemm_t_fp4_moe_dn") + ms + (n128 ? "n128" : "")
                   : x4m ? "nh_x4_moe_dn" + ms + (L.down_e.l2 ? "_l2" : "")
                         : (fp4 ? (L.down_e.l2 ? "nh_gemm_fp4l2_moe_dn" : "nh_gemm_fp4_moe_dn") : "nh_gemm_iq2_moe_dn") + ms));
        e.buf(0, ph_);
        e.buf(1, L.down_e.q);
        e.buf(2, L.down_e.s);
        e.buf(3, pout_);
        e.val(4, gp);
        e.buf(5, ptiles_);
        e.buf(6, pntiles_);
        e.buf(7, psorted_);
        e.buf(8, iq2tab_);
        if (fp4) e.buf(9, L.down_e.gse);
        e.dispatch((uint32_t(cfg_.latent) + (n128 ? 127 : 63)) / (n128 ? 128 : 64), max_tiles, 1, 128);
      }
      const CombineP cp{uint32_t(cfg_.latent), k};
      e.use(pipe(lat ? "nh_moe_combine" : "nh_moe_combine_add"));
      e.buf(0, pout_);
      e.buf(1, pew_);
      e.buf(2, lat ? plo_ : px_);
      e.val(3, cp);
      e.dispatch(uint32_t(T), 1, 1, 256);
      // shared expert; both branches add into the residual stream
      if (ffn_ane) {
        if (lat) enc_gemm(e, L.lat_up, plo_, px_, T, 1);
        ane_->ffn_join(e, i, px_, T);
      } else {
        enc_gemm(e, L.up_s, pxh_, psh_, T, 2);
        if (lat) enc_gemm(e, L.lat_up, plo_, px_, T, 1);
        enc_gemm(e, L.down_s, psh_, px_, T, 1);
      }
    }
    // a command buffer per layer of a long chunk (~80 ms at 2048 rows instead of seconds), so the desktop's own GPU
    // work is not held back behind one long buffer (a split chunk already ends one at every Neural Engine handoff)
    if (!split && T >= 256 && i + 1 < cfg_.layers) e.split();
  }
  if (cap_) enc_capture(e, cfg_.layers, px_, pos0, T);
}

}  // namespace ni
