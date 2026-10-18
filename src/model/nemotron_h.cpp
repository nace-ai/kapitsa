#include "model/nemotron_h.h"
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
  uint32_t T, pos0, zx_stride, DI, CH;
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
  for (int i = 0; i < c.layers; ++i) {
    const nlohmann::json& b = blocks[size_t(i)];
    const std::string bt = b.contains("block_type") && b["block_type"].is_string() ? b["block_type"].get<std::string>() : "";
    if (bt == "mamba") {
      c.kind[size_t(i)] = 'M';
    } else if (bt == "attention") {
      c.kind[size_t(i)] = 'A';
    } else if (bt == "moe") {
      c.kind[size_t(i)] = 'E';
      c.n_exp[size_t(i)] = int(num(b, "n_routed_experts", num(t, "n_routed_experts", 0)));
      c.top_k[size_t(i)] = int(num(b, "num_experts_per_tok", num(t, "num_experts_per_tok", 0)));
      c.exp_I[size_t(i)] = int(num(b, "moe_intermediate_size", num(t, "moe_intermediate_size", 0)));
      c.max_E = std::max(c.max_E, c.n_exp[size_t(i)]);
      c.max_k = std::max(c.max_k, c.top_k[size_t(i)]);
      c.max_I = std::max(c.max_I, c.exp_I[size_t(i)]);
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
  if (c.hidden <= 0 || c.layers <= 0 || c.vocab <= 0 || c.d_inner <= 0 || c.n_head <= 0 || c.d_state != 128 ||
      c.head_dim != 64 || c.conv_k != 4 || c.n_group <= 0 || c.n_head % c.n_group)
    fatal("%s: unsupported Nemotron-H shape (hidden %d, Mamba inner %d, state %d, heads %d, conv %d)", dir.c_str(),
          c.hidden, c.d_inner, c.d_state, c.n_head, c.conv_k);
  if (std::count(c.kind.begin(), c.kind.end(), 'A') && (c.hd != 128 || !c.hkv || c.hq % c.hkv))
    fatal("%s: attention head dim %d is not supported", dir.c_str(), c.hd);
  if (c.max_E > 512 || c.max_k > 16) fatal("%s: %d experts / top-%d is not supported", dir.c_str(), c.max_E, c.max_k);
  return c;
}

// ---- weight formats ----
// Tensors as NVIDIA's checkpoint stores them (hf_quant_config.json "MIXED_PRECISION"): Q8_0 and IQ2_XXS as ggml
// blocks in uint8 tensors [N][K / block][block bytes]; NVFP4 as ModelOpt writes it (uint8 [N][K / 2], low nibble =
// even element; F8_E4M3 scales [N][K / 16]; F32 weight_scale_2); BF16 for embeddings, router, norms and Mamba vectors.
void nh_row_bytes(NhFmt f, int64_t K, size_t& qb, size_t& sb) {
  switch (f) {
    case NhFmt::Q8: qb = size_t(K); sb = size_t(K / 32) * 2; break;
    case NhFmt::FP4: qb = size_t(K / 2); sb = size_t(K / 16); break;
    case NhFmt::IQ2: qb = size_t(K / 4); sb = size_t(K / 256) * 2; break;
    case NhFmt::BF16: qb = size_t(K) * 2; sb = 0; break;
    case NhFmt::F32: qb = size_t(K) * 4; sb = 0; break;
  }
}

size_t nh_src_row_bytes(NhFmt f, int64_t K) {
  switch (f) {
    case NhFmt::Q8: return size_t(K / 32) * 34;
    case NhFmt::IQ2: return size_t(K / 256) * 66;
    case NhFmt::FP4: return size_t(K / 2);
    case NhFmt::BF16: return size_t(K) * 2;
    case NhFmt::F32: return size_t(K) * 4;
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

void nh_repack_rows(NhFmt f, int64_t K, const uint8_t* src, int64_t rows, uint8_t* q, uint8_t* s) {
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
        // ModelOpt order (byte j of a 16-element group: elements 2j, 2j + 1) -> the kernels' (byte i: i, i + 8)
        for (int64_t i = 0; i < K / 16; ++i) {
          uint64_t v;
          memcpy(&v, b + i * 8, 8);
          const uint64_t o = spread_nibbles(uint32_t(v)) | (spread_nibbles(uint32_t(v >> 32)) << 4);
          memcpy(qr + i * 8, &o, 8);
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
  if ((w.dtype == DType::BF16 || w.dtype == DType::F32) && sh.size() == 2) {
    f = w.dtype == DType::BF16 ? NhFmt::BF16 : NhFmt::F32;
    N = uint32_t(sh[0]);
    K = uint32_t(sh[1]);
    return true;
  }
  return false;
}

NemotronH::NemotronH(const std::string& path, int prefill_chunk) {
  path_ = path;
  while (path_.size() > 1 && path_.back() == '/') path_.pop_back();
  if (!is_model(path_)) fatal("%s: not a Nemotron-H checkpoint folder (config.json)", path.c_str());
  cfg_ = read_config(path_);
  Gpu::get().compile_extra("Nemotron-H", kNhKernelSource);
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
                                   "copy_u32", "kv_regrow", "nh_x4_", "nh_x2_"};
  std::vector<std::string> v;
  for (const std::string& n : g.function_names())
    for (const char* pre : prefixes)
      if (n.rfind(pre, 0) == 0) {
        v.push_back(n);
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
  enum Kind { Rows, Copy, Bf16ToF32, NegExpBf16 } kind = Copy;
  size_t src = 0, len = 0;  // bytes of the source tensor
  NhFmt fmt = NhFmt::F32;   // Rows: the matrix format
  int64_t K = 0, rows = 0;  // Rows
  uint8_t* q = nullptr;     // destination (codes)
  uint8_t* s = nullptr;     // destination (scales; Rows of Q8 / IQ2)
};

void run_job(const SafetensorsIndex& idx, const Job& j, std::vector<uint8_t>& raw) {
  raw.resize(j.len);
  idx.read(*j.t, j.src, raw.data(), j.len);
  switch (j.kind) {
    case Job::Rows: nh_repack_rows(j.fmt, j.K, raw.data(), j.rows, j.q, j.s); break;
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
  // 1. what goes where: every tensor's converted size, placed in 1 GB weight buffers (bigger tensors get their own)
  struct Want {
    NhMat* m = nullptr;                                 // a matrix (q, s placed)
    std::vector<const SafetensorsIndex::Entry*> parts;  // its source tensors: one, or one per expert
    const SafetensorsIndex::Entry* scale = nullptr;     // NVFP4 scales
    Buffer* vec = nullptr;                              // an fp32 vector
    Job::Kind vk = Job::Copy;
  };
  std::vector<Want> wants;
  auto mat_from = [&](NhMat& m, std::vector<const SafetensorsIndex::Entry*> parts, const std::string& name) {
    const std::string base = name.substr(0, name.size() - 7);  // "x.weight" -> "x"
    const SafetensorsIndex::Entry* sc = idx.find(base + ".weight_scale");
    uint32_t N = 0, K = 0;
    if (!nh_describe(*parts[0], sc, m.fmt, N, K))
      fatal("%s: tensor %s has a format Kapitsa does not run", path_.c_str(), name.c_str());
    for (auto* p : parts)
      if (p->shape != parts[0]->shape || p->dtype != parts[0]->dtype)
        fatal("%s: the experts of %s differ in shape", path_.c_str(), name.c_str());
    m.N = N;
    m.K = K;
    m.E = uint32_t(parts.size());
    nh_row_bytes(m.fmt, K, m.q_row, m.s_row);
    if (m.fmt == NhFmt::FP4) {
      float gs = 1.0f;
      if (const SafetensorsIndex::Entry* s2 = idx.find(base + ".weight_scale_2")) {
        if (s2->dtype != DType::F32 || s2->len != 4) fatal("%s: %s.weight_scale_2 is not one F32", path_.c_str(), base.c_str());
        idx.read(*s2, 0, &gs, 4);
      }
      m.gs = gs;
    }
    wants.push_back({&m, std::move(parts), m.fmt == NhFmt::FP4 ? sc : nullptr, nullptr, Job::Copy});
  };
  auto mat = [&](NhMat& m, const std::string& name) { mat_from(m, {&idx.get(name)}, name); };
  auto experts = [&](NhMat& m, const std::string& prefix, int E, const char* which) {
    std::vector<const SafetensorsIndex::Entry*> parts;
    for (int e = 0; e < E; ++e) parts.push_back(&idx.get(prefix + std::to_string(e) + "." + which + ".weight"));
    mat_from(m, std::move(parts), prefix + "0." + which + ".weight");
  };
  auto vec = [&](Buffer& b, const std::string& name, bool neg_exp = false) {
    const SafetensorsIndex::Entry& t = idx.get(name);
    if (t.dtype != DType::BF16 && t.dtype != DType::F32)
      fatal("%s: tensor %s is neither BF16 nor F32", path_.c_str(), name.c_str());
    if (neg_exp && t.dtype != DType::BF16) fatal("%s: tensor %s is not BF16", path_.c_str(), name.c_str());
    wants.push_back({nullptr, {&t}, nullptr, &b,
                     neg_exp ? Job::NegExpBf16 : t.dtype == DType::BF16 ? Job::Bf16ToF32 : Job::Copy});
  };
  const std::string bb = pre + "backbone.";
  layers_.resize(size_t(cfg_.layers));
  mat(embed_, bb + "embeddings.weight");
  mat(head_, idx.find(pre + "lm_head.weight") ? pre + "lm_head.weight" : bb + "embeddings.weight");
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
      mat(L.q, mx + "q_proj.weight");
      mat(L.k, mx + "k_proj.weight");
      mat(L.v, mx + "v_proj.weight");
      mat(L.o, mx + "o_proj.weight");
    } else {
      L.E = cfg_.n_exp[size_t(i)];
      L.k_used = cfg_.top_k[size_t(i)];
      L.I = cfg_.exp_I[size_t(i)];
      mat(L.router, mx + "gate.weight");
      vec(L.exp_bias, mx + "gate.e_score_correction_bias");
      mat(L.lat_down, mx + "fc1_latent_proj.weight");
      mat(L.lat_up, mx + "fc2_latent_proj.weight");
      experts(L.up_e, mx + "experts.", L.E, "up_proj");
      experts(L.down_e, mx + "experts.", L.E, "down_proj");
      mat(L.up_s, mx + "shared_experts.up_proj.weight");
      mat(L.down_s, mx + "shared_experts.down_proj.weight");
      if (L.up_e.fmt != NhFmt::IQ2 || L.down_e.fmt != NhFmt::IQ2)
        fatal("%s: layer %d's experts are not IQ2_XXS", path_.c_str(), i);
      if (int(L.router.N) != L.E || int(L.up_e.N) != L.I || int(L.up_e.K) != cfg_.latent || int(L.down_e.K) != L.I)
        fatal("%s: layer %d's expert shapes disagree with config.json", path_.c_str(), i);
    }
  }
  if (embed_.fmt != NhFmt::BF16 && embed_.fmt != NhFmt::Q8)
    fatal("%s: the embedding is neither BF16 nor Q8_0", path_.c_str());
  // 2. place, and cut the reads into jobs of ~16 MB
  const size_t kChunk = size_t(1) << 30, kJob = size_t(16) << 20;
  size_t used = kChunk;  // in the current arena buffer
  auto place = [&](size_t bytes, const char* label) -> Buffer {
    bytes = align_up(std::max<size_t>(bytes, 16), 256);
    if (bytes > kChunk) {
      arena_.push_back(g.alloc(bytes, label, false));
      return arena_.back().view(0, bytes);
    }
    if (used + bytes > kChunk) {
      arena_.push_back(g.alloc(kChunk, "nh_weights", false));
      used = 0;
    }
    Buffer b = arena_.back().view(used, bytes);
    used += bytes;
    return b;
  };
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
      *w.vec = place(size_t(t.numel()) * 4, "nh_vectors");
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
    m.q = place(m.q_row * m.N * m.E, "nh_codes");
    m.s = m.s_row ? place(m.s_row * m.N * m.E, "nh_scales") : m.q;
    weight_bytes_ += m.q.size + (m.s_row ? m.s.size : 0);
    const size_t rb = nh_src_row_bytes(m.fmt, m.K);
    const int64_t per = std::max<int64_t>(1, int64_t(kJob / std::max<size_t>(rb, 1)));
    for (size_t e = 0; e < w.parts.size(); ++e) {
      const SafetensorsIndex::Entry& t = *w.parts[e];
      if (t.len != rb * m.N) fatal("%s: a tensor's size disagrees with its shape", path_.c_str());
      for (int64_t r0 = 0; r0 < int64_t(m.N); r0 += per) {
        Job j;
        j.t = &t;
        j.kind = Job::Rows;
        j.fmt = m.fmt;
        j.K = m.K;
        j.rows = std::min<int64_t>(per, int64_t(m.N) - r0);
        j.src = rb * size_t(r0);
        j.len = rb * size_t(j.rows);
        const size_t row = e * m.N + size_t(r0);
        j.q = m.q.as<uint8_t>() + m.q_row * row;
        j.s = m.s_row && m.fmt != NhFmt::FP4 ? m.s.as<uint8_t>() + m.s_row * row : nullptr;
        jobs.push_back(j);
      }
      src_bytes += t.len;
    }
    if (w.scale) {  // NVFP4 scales: the E4M3 bytes as they are
      const SafetensorsIndex::Entry& t = *w.scale;
      if (t.len != m.s_row * m.N) fatal("%s: NVFP4 scales of the wrong size", path_.c_str());
      for (size_t o = 0; o < t.len; o += kJob) {
        Job j;
        j.t = &t;
        j.kind = Job::Copy;
        j.src = o;
        j.len = std::min(kJob, t.len - o);
        j.q = m.s.as<uint8_t>() + o;
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
  logi("Nemotron-H: %d layers (%d Mamba-2, %d attention, %d MoE), %.2f GB of weights read in %.1f s (%s)", cfg_.layers,
       int(std::count(cfg_.kind.begin(), cfg_.kind.end(), 'M')), int(std::count(cfg_.kind.begin(), cfg_.kind.end(), 'A')),
       int(std::count(cfg_.kind.begin(), cfg_.kind.end(), 'E')), src_bytes / 1e9, (now_ms() - t0) / 1000,
       formats_summary().c_str());
}

std::string NemotronH::formats_summary() const {
  return "IQ2_XXS experts, NVFP4 Mamba, Q8_0 attention / latent / head, BF16 embedding / router";
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
  sh_ = g.alloc(R * size_t(std::max(1, cfg_.shared_I)) * 4, "nh_sh");
  logits_ = g.alloc(R * size_t(cfg_.vocab) * 4, "nh_logits");
  const size_t nchunk1 = (size_t(cfg_.vocab) + 2047) / 2048;
  cand_v_ = g.alloc(nchunk1 * 64 * 4, "nh_cand_v");
  cand_i_ = g.alloc(nchunk1 * 64 * 4, "nh_cand_i");
  sprob_ = g.alloc(64 * 4, "nh_sprob");
  samp_ = g.alloc(64 * 4, "nh_samples");
  xnh_ = g.alloc(R * std::max(H, size_t(cfg_.d_inner)) * 2, "nh_xnh");
  aoh_ = g.alloc(R * size_t(std::max(1, cfg_.hq * cfg_.hd)) * 2, "nh_aoh");
  loh_ = g.alloc(R * size_t(std::max(1, cfg_.latent)) * 2, "nh_loh");
  shh_ = g.alloc(R * size_t(std::max(1, cfg_.shared_I)) * 2, "nh_shh");
  if (const char* v = getenv("NI_NH_ROWS_GEMM")) gemm_rows_ = atoi(v) != 0;
  if (const char* v = getenv("NI_NH_V8")) v8_ = atoi(v) != 0;
  if (const char* v = getenv("NI_NH_MERGE_FLUSH")) merge_flush_ = atoi(v) != 0;
  acc_ = g.alloc(64, "nh_accepted");
  route_ctr_ = g.alloc(16, "nh_route_counter");
  memset(route_ctr_.ptr(), 0, 16);
  sd_ = g.alloc(size_t(cfg_.hidden) * 4, "nh_shared_down");
  if (const char* v = getenv("NI_NH_FUSE")) fuse_dec_ = atoi(v) != 0;
  if (const char* v = getenv("NI_NH_GROUP")) group_ = atoi(v) != 0;
  live_ = g.alloc(16, "nh_live_rows");
  live_all_ = g.alloc(16, "nh_live_all");
  live_.as<uint32_t>()[0] = 0xffffffffu;
  live_all_.as<uint32_t>()[0] = 0xffffffffu;
  // NVFP4 9-32-row kernels split K in 4 (K <= 8192) or 8 parts: partials of the largest FP4 matrix (in_proj) at 32
  // rows, and one arrival counter per 64-row tile (zero now; every call leaves them zero)
  if (const char* v = getenv("NI_NH_X4")) x4_ = atoi(v) != 0;
  x4part_ = g.alloc(size_t(4) * ((size_t(cfg_.in_proj()) + 63) / 64) * 2048 * 4, "nh_x4_partials");
  x4cnt_ = g.alloc(4096, "nh_x4_counters");
  memset(x4cnt_.ptr(), 0, 4096);
  if (const char* v = getenv("NI_NH_X2")) x2_ = atoi(v) != 0;
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

MemPlan NemotronH::plan(const std::string& path, int prefill_chunk) {
  MemPlan m;
  if (!is_model(path)) return m;
  const NhConfig c = read_config(path);
  const SafetensorsIndex idx(path, not_lm);
  m.hidden = c.hidden;
  m.layers = c.layers;
  for (const auto& [name, t] : idx.all()) {
    // as loaded: ggml blocks and NVFP4 keep their size (codes and scales apart), BF16 vectors become fp32
    const bool vector = t.shape.size() == 1 || name.find("conv1d") != std::string::npos;
    const bool embedding = name.find("embeddings.") != std::string::npos;
    // the BF16 embedding is mapped from its file (NI_NH_MAP_EMBED): clean file pages, the rows in use
    const bool mapped = embedding && t.dtype == DType::BF16 && !(getenv("NI_NH_MAP_EMBED") && !atoi(getenv("NI_NH_MAP_EMBED")));
    if (!mapped) m.weights += t.dtype == DType::BF16 && vector ? 2 * t.len : t.len;
    if (embedding) continue;  // one row per token
    const size_t lp = name.find("layers.");
    const int li = lp == std::string::npos ? -1 : std::atoi(name.c_str() + lp + 7);
    NhFmt f;
    uint32_t N = 0, K = 0;
    const size_t dot = name.rfind(".weight");
    const SafetensorsIndex::Entry* sc =
        dot != std::string::npos && dot + 7 == name.size() ? idx.find(name.substr(0, dot) + ".weight_scale") : nullptr;
    const bool matrix = nh_describe(t, sc, f, N, K) && name.find("weight_scale") == std::string::npos;
    if (name.find(".experts.") != std::string::npos && li >= 0 && c.n_exp[size_t(li)] > 0) {
      const double used = double(c.top_k[size_t(li)]) / c.n_exp[size_t(li)];  // each expert: k / E of the tokens
      m.decode_bytes += size_t(double(t.len) * used);
      if (matrix) m.decode_params += size_t(double(N) * K * used);
    } else {
      m.decode_bytes += t.len;
      if (matrix) m.decode_params += size_t(N) * K;
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
                        size_t(c.max_E) * 4 + size_t(c.latent) * 4 + size_t(c.shared_I) * 2) +
                   P * (16 + size_t(c.max_I) * 2 + size_t(c.latent) * 4) + (P / 16 + size_t(c.max_E) + 2) * 16;
  m.activations += size_t(4) * ((size_t(c.in_proj()) + 63) / 64) * 2048 * 4 + 4096;  // x4 split-K partials, counters
  m.head_row_bytes = size_t(c.hidden) + size_t(c.hidden / 32) * 2;
  m.dense = false;
  m.ffn_ane = false;
  m.formats = "IQ2_XXS / NVFP4 / Q8_0 / BF16";
  m.ane = ane_plan_bytes(c, state);
  return m;
}

// The GPU + Neural Engine split (on unless NI_ANE=0): its surfaces at the default shares (int8 inputs and weights
// per 2048-channel segment, fp16 outputs for 2048 rows), the rotation scratch, and the recurrent-state checkpoint the
// engine keeps to rerun a chunk on the GPU.
size_t NemotronH::ane_plan_bytes(const NhConfig& c, size_t state) {
  if (const char* v = getenv("NI_ANE"); v && !atoi(v)) return 0;
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
  // one FP4 row: the x4 kernel (8 rows per 64-thread threadgroup; in_proj 248, out_proj 228 GB/s on fresh weights)
  if (R == 1 && W.fmt == NhFmt::FP4 && x4_ && epi <= 1 && W.K % 32 == 0 && (xs == 0 || xs == W.K) &&
      (ys == 0 || ys == W.N)) {
    const MvP p{W.N, W.K, W.K, W.N, W.gs, 1};
    e.use(pipe(epi ? "nh_x4_mv_l0_add" : "nh_x4_mv_l0"));
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
  const int rk = R <= 1 ? 1 : R <= 2 ? 2 : R <= 4 ? 4 : 8;
  const char* fm = W.fmt == NhFmt::Q8    ? "q8"
                   : W.fmt == NhFmt::FP4 ? "fp4"
                   : W.fmt == NhFmt::IQ2 ? "iq2"
                   : W.fmt == NhFmt::BF16 ? "bf16"
                                          : "f32";
  const std::string name = std::string("nh_mv_") + fm + "_r" + std::to_string(rk) +
                           (epi == 1 ? "_add" : epi == 2 ? "_relu2" : "");
  const uint32_t nr = rk <= 2 ? 4 : rk == 4 ? 2 : 1, nsg = 4;
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

void NemotronH::enc_embed(Encoder& e, const Buffer& tokens, int tok_off, int R, const Buffer& out) {
  const EmbedP ep{uint32_t(cfg_.hidden), uint32_t(tok_off)};
  e.use(pipe(embed_.fmt == NhFmt::BF16 ? "nh_embed_bf16" : "nh_embed_q8"));
  e.buf(0, embed_.q);
  e.buf(1, embed_.s);
  e.buf(2, tokens);
  e.buf(3, out);
  e.val(4, ep);
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
                        const Buffer* zp, int np) {
  const NhLayer& L = layers_[size_t(li)];
  const SsmP p{uint32_t(T), uint32_t(pos0), write ? 1u : 0u, uint32_t(cfg_.in_proj()), uint32_t(cfg_.n_head),
               uint32_t(cfg_.head_dim), uint32_t(cfg_.d_state), uint32_t(cfg_.n_group)};
  // short passes (decode, verify, flush): one simdgroup per state row (NI_NH_SSM1=0: the head-per-threadgroup kernel)
  static const bool ssm1 = !getenv("NI_NH_SSM1") || atoi(getenv("NI_NH_SSM1")) != 0;
  const bool short_pass = (ssm1 || np > 0) && T <= kMaxRows && cfg_.head_dim % 8 == 0 && cfg_.d_state == 128;
  e.use(pipe(short_pass ? "nh_ssm1" : "nh_ssm"));
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
  const uint32_t npu = uint32_t(np);
  e.buf(11, zp ? *zp : zx);
  e.val(12, npu);
  if (short_pass) e.dispatch(uint32_t(cfg_.n_head * cfg_.head_dim / 8), 1, 1, 256);
  else e.dispatch(uint32_t(cfg_.n_head), 1, 1, 256);
  if (np > 0) {  // the committed pending rows' raw xBC into the ring (after every head has read it)
    const uint32_t CH = uint32_t(cfg_.conv_ch());
    const RingP rp{uint32_t(np), uint32_t(pos0 - np), uint32_t(cfg_.in_proj()), uint32_t(cfg_.d_inner), CH};
    const uint32_t nn = CH * uint32_t(std::min(np, 3));
    e.use(pipe("nh_conv_ring"));
    e.buf(0, *zp);
    e.buf(1, s.conv[size_t(li)]);
    e.val(2, rp);
    e.dispatch((nn + 255) / 256, 1, 1, 256);
  }
  if (write && T > 1) {  // (one row: nh_ssm writes its ring slot itself)
    const uint32_t CH = uint32_t(cfg_.conv_ch());
    const RingP rp{uint32_t(T), uint32_t(pos0), uint32_t(cfg_.in_proj()), uint32_t(cfg_.d_inner), CH};
    const uint32_t n = CH * uint32_t(std::min(T, 3));
    e.use(pipe("nh_conv_ring"));
    e.buf(0, zx);
    e.buf(1, s.conv[size_t(li)]);
    e.val(2, rp);
    e.dispatch((n + 255) / 256, 1, 1, 256);
  }
}

void NemotronH::enc_attn(Encoder& e, Seq& s, int li, int pos0, int R) {
  const NhLayer& L = layers_[size_t(li)];
  const bool gm = gemm_rows_ && R >= 2;  // xnh_ holds the normed rows in half
  e.group_begin();  // (a grouped encoder runs the three projections together)
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
  const KvP kp{uint32_t(R), uint32_t(pos0), uint32_t(s.cap), uint32_t(cfg_.hkv), uint32_t(cfg_.hd)};
  e.use(pipe("nh_kv_write"));
  e.buf(0, k_);
  e.buf(1, v_);
  e.buf(2, s.kc[size_t(li)]);
  e.buf(3, s.vc[size_t(li)]);
  e.val(4, kp);
  e.dispatch(uint32_t(R), uint32_t(2 * cfg_.hkv), 1, uint32_t(cfg_.hd));
  const int L_max = pos0 + R;
  uint32_t chunk = 256;
  while ((uint32_t(L_max) + chunk - 1) / chunk > uint32_t(attn_max_chunks_)) chunk *= 2;
  const uint32_t nchunks = (uint32_t(L_max) + chunk - 1) / chunk;
  const AttnP ap{uint32_t(R), uint32_t(pos0), uint32_t(s.cap), uint32_t(cfg_.hq), uint32_t(cfg_.hkv), chunk, nchunks,
                 1.0f / std::sqrt(float(cfg_.hd))};
  e.use(pipe("nh_attn_partial"));
  e.buf(0, q_);
  e.buf(1, s.kc[size_t(li)]);
  e.buf(2, s.vc[size_t(li)]);
  e.buf(3, part_o_);
  e.buf(4, part_ml_);
  e.val(5, ap);
  e.dispatch(nchunks, uint32_t(cfg_.hkv), uint32_t(R), uint32_t(32 * (cfg_.hq / cfg_.hkv)));
  e.use(pipe("nh_attn_reduce"));
  e.buf(0, part_o_);
  e.buf(1, part_ml_);
  e.buf(2, ao_);
  e.val(3, ap);
  e.dispatch(uint32_t(cfg_.hq), uint32_t(R), 1, uint32_t(cfg_.hd));
  if (gm) {
    enc_rows_half(e, ao_, aoh_, R, cfg_.hq * cfg_.hd);
    enc_gemm(e, L.o, aoh_, x_, R, 1);
  } else {
    enc_mv(e, L.o, ao_, x_, R, 1);
  }
}

void NemotronH::enc_moe(Encoder& e, const NhLayer& L, int R) {
  const uint32_t k = uint32_t(L.k_used);
  if (R == 1 && fuse_dec_ && L.router.fmt == NhFmt::BF16 && L.lat_down.fmt == NhFmt::Q8 && L.lat_up.fmt == NhFmt::Q8 &&
      L.up_s.fmt == NhFmt::IQ2 && L.down_s.fmt == NhFmt::IQ2 && L.E <= 512 && L.k_used <= 32) {
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
  const bool gm = gemm_rows_ && R >= 2;
  // With a grouped encoder (NI_NH_GROUP): router, latent down and shared up run together, then routed down and shared
  // down; a serial encoder runs them one after another in the same order.
  const bool grouped = e.grouped();
  // router: logits (BF16 weights, fp32 products), sigmoid + selection bias, top-k
  if (grouped) e.group_begin();
  enc_mv(e, L.router, xn_, rl_, R, 0, uint32_t(cfg_.hidden), uint32_t(cfg_.max_E));
  if (grouped) {
    if (gm) enc_gemm(e, L.lat_down, xnh_, lat_, R, 0);
    else enc_mv(e, L.lat_down, xn_, lat_, R, 0);
    if (gm) enc_gemm(e, L.up_s, xnh_, shh_, R, 2);
    else enc_mv(e, L.up_s, xn_, sh_, R, 2);
    e.group_end();
  }
  const RouteP rp{uint32_t(L.E), k, cfg_.w_norm ? 1u : 0u, cfg_.w_scale, uint32_t(cfg_.max_E), uint32_t(R)};
  const bool route1 = L.E <= 512 && L.k_used <= 32;
  e.use(pipe(route1 ? "nh_moe_route1" : "nh_moe_route"));
  e.buf(0, rl_);
  e.buf(1, L.exp_bias);
  e.buf(2, eid_);
  e.buf(3, ew_);
  e.val(4, rp);
  e.buf(5, live());
  if (route1) e.dispatch((uint32_t(R) + 7) / 8, 1, 1, 256);
  else e.dispatch(uint32_t(R), 1, 1, 512);
  // latent projection; routed experts: per (row, expert) pair matvecs (a verify block's 72 pairs average ~1.4 rows per expert, so
  // grouping the pairs by expert into 8-row MMA tiles (1.02 ms per layer) or 16-row GEMM tiles (~1.8) measured slower
  // than 0.82 ms per layer for the pairs; so did decoding each expert row once for all of its pairs, which spills)
  if (!grouped) {
    if (gm) enc_gemm(e, L.lat_down, xnh_, lat_, R, 0);
    else enc_mv(e, L.lat_down, xn_, lat_, R, 0);
  }
  const MoeP mp{uint32_t(L.I), uint32_t(cfg_.latent), k, uint32_t(L.up_e.q_exp()), uint32_t(L.up_e.s_exp()),
                uint32_t(L.down_e.q_exp()), uint32_t(L.down_e.s_exp()), uint32_t(cfg_.max_I)};
  // the down projection per (pair, 16 rows) into weighted partials, summed per row (nh_moe2_*)
  const uint32_t npairs = uint32_t(R) * k, nrows = uint32_t(R);
  {
    const bool x2 = x2_ && L.E <= 1024 && L.I % 256 == 0 && cfg_.latent % 256 == 0;
    if (x2 && R >= 2) {
      // the pairs grouped by expert into tiles of <= 2 (each expert row decoded once for both tokens), the up and down
      // dispatches sized by the bucketing (indirect): a verify layer 0.36-0.52 vs 0.55-0.88 ms
      const struct {
        uint32_t P, E, BM, k, up_x, dn_x;
      } bp{npairs, uint32_t(L.E), 2, k, (uint32_t(L.I) + 3) / 4, (uint32_t(cfg_.latent) + 3) / 4};
      e.use(pipe("nh_x2_bucket"));
      e.buf(0, eid_);
      e.buf(1, x2sorted_);
      e.buf(2, x2tiles_);
      e.buf(3, x2args_);
      e.buf(4, x2rank_);
      e.bytes(5, &bp, sizeof bp);
      e.buf(6, live());
      e.dispatch(1, 1, 1, 1024);
      e.use(pipe("nh_x2_moeb_up"));
      e.buf(0, L.up_e.q);
      e.buf(1, L.up_e.s);
      e.buf(2, lat_);
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
      e.dispatch_indirect(x2args_, 16, 32, 1, 1);
    } else {
      e.use(pipe(x2 ? "nh_x2_moe_up4" : "nh_moe2_up_iq2"));
      e.buf(0, L.up_e.q);
      e.buf(1, L.up_e.s);
      e.buf(2, lat_);
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
    e.use(pipe("nh_moe2_sum"));
    e.buf(0, part_);
    e.buf(1, lo_);
    e.val(2, mp);
    e.buf(3, live());
    e.dispatch(nrows, 1, 1, 256);
  }
  // shared expert on the full hidden state; both branches add into the residual stream
  if (grouped) {  // (the shared expert ran above)
    if (gm) {
      enc_rows_half(e, lo_, loh_, R, cfg_.latent);
      enc_gemm(e, L.lat_up, loh_, x_, R, 1);
    } else {
      enc_mv(e, L.lat_up, lo_, x_, R, 1);
    }
  } else if (gm) {
    enc_gemm(e, L.up_s, xnh_, shh_, R, 2);
    enc_rows_half(e, lo_, loh_, R, cfg_.latent);
    enc_gemm(e, L.lat_up, loh_, x_, R, 1);
    enc_gemm(e, L.down_s, shh_, x_, R, 1);
  } else {
    enc_mv(e, L.up_s, xn_, sh_, R, 2);
    enc_mv(e, L.lat_up, lo_, x_, R, 1);
    enc_mv(e, L.down_s, sh_, x_, R, 1);
  }
}

void NemotronH::enc_rows(Encoder& e, Seq& s, int pos0, int R, bool commit) {
  if (R < 1 || R > kMaxRows) fatal("NemotronH: %d rows in one pass", R);
  if (pos0 + R >= s.cap) fatal("context full (%d)", s.cap);
  enc_embed(e, s.tok, pos0, R, x_);
  for (int i = 0; i < cfg_.layers; ++i) {
    const NhLayer& L = layers_[size_t(i)];
    if (cap_) enc_capture(e, i, x_, pos0, R);
    const bool gm = gemm_rows_ && R >= 2;
    enc_norm(e, x_, L.norm, xn_, R, cfg_.hidden);
    if (gm && L.kind != 'E') enc_rows_half(e, xn_, xnh_, R, cfg_.hidden);
    if (gm && L.kind == 'E') enc_rows_half(e, xn_, xnh_, R, cfg_.hidden);
    if (L.kind == 'M') {
      // a verify block's rows go to the half of the pending buffer the pending rows (if any) do not use; the scan
      // commits those first (merged flush)
      const size_t half = size_t(kMaxRows) * cfg_.in_proj() * 4;
      const Buffer pb = s.pend_qkv[size_t(i)];
      const Buffer zx = commit ? zx_ : pb.view(size_t(1 - s.pend_half) * half, half);
      if (gm) enc_gemm(e, L.in, xnh_, zx, R, 0);
      else enc_mv(e, L.in, xn_, zx, R, 0);
      if (!commit && s.n_pend > 0 && merge_flush_) {
        const Buffer zp = pb.view(size_t(s.pend_half) * half, half);
        enc_ssm(e, s, i, zx, R, pos0, false, y_, &zp, s.n_pend);
      } else {
        enc_ssm(e, s, i, zx, R, pos0, commit, y_);
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
      enc_attn(e, s, i, pos0, R);
    } else {
      enc_moe(e, L, R);
    }
  }
  if (!commit) {
    s.n_pend = R;  // until commit_rows() says how many were accepted
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
  for (int i = 0; i < cfg_.layers; ++i)
    if (cfg_.kind[size_t(i)] == 'M')
      enc_ssm(e, s, i, s.pend_qkv[size_t(i)].view(size_t(s.pend_half) * half, half), s.n_pend, s.pend_pos, true, y_);
  s.n_pend = 0;
}

void NemotronH::enc_head(Encoder& e, int R, const Buffer& out, size_t out_row_stride) {
  enc_norm(e, x_, out_norm_, xn_, R, cfg_.hidden);
  if (gemm_rows_ && R >= 2) {
    enc_rows_half(e, xn_, xnh_, R, cfg_.hidden);
    enc_gemm(e, head_, xnh_, out, R, 0, uint32_t(out_row_stride));
  } else {
    enc_mv(e, head_, xn_, out, R, 0, uint32_t(cfg_.hidden), uint32_t(out_row_stride));
  }
}

void NemotronH::enc_sample(Encoder& e, const Buffer& lrow, const Buffer& tokens, int out_index, const SamplingParams& sp,
                           uint32_t step) {
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
  e.use(pipe("sample_stage2"));
  e.buf(0, cand_v_);
  e.buf(1, cand_i_);
  e.buf(2, tokens);
  e.buf(3, sprob_);
  e.val(4, p2);
  e.dispatch(1, 1, 1, 1024);
}

void NemotronH::encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample) {
  encode_flush(e, s);
  enc_rows(e, s, pos, 1, true);
  if (!sample) return;
  enc_head(e, 1, logits_, size_t(cfg_.vocab));
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
  enc_sample(e, logits_, s.tok, pos0 + T, sp, step);
}

void NemotronH::encode_verify(Encoder& e, Seq& s, int pos, int R) {
  if (!merge_flush_) encode_flush(e, s);  // (merged: each Mamba layer's scan commits the pending rows first)
  else if (s.n_pend > 0 && s.pend_pos + s.n_pend != pos) fatal("NemotronH: verify at %d after pending rows at %d", pos, s.pend_pos);
  enc_rows(e, s, pos, R, false);
  enc_head(e, R, logits_, size_t(cfg_.vocab));
}

void NemotronH::encode_accept(Encoder& e, Seq& s, int pos, int R, int K, const SamplingParams& sp) {
  for (int r = 0; r < R; ++r)
    enc_sample(e, logits_.view(size_t(r) * cfg_.vocab * 4, size_t(cfg_.vocab) * 4), samp_, r, sp, uint32_t(pos + r));
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
  psh_ = g.alloc(R * size_t(std::max(1, cfg_.shared_I)) * 2, "nh_pf_sh");
  psorted_ = g.alloc(P * 4, "nh_pf_sorted");
  prank_ = g.alloc(P * 4, "nh_pf_rank");
  ptiles_ = g.alloc((P / 16 + size_t(cfg_.max_E) + 2) * 16, "nh_pf_tiles");
  pntiles_ = g.alloc(16, "nh_pf_ntiles");
  pf_rows_ = tmax_;
}

void NemotronH::enc_gemm(Encoder& e, const NhMat& W, const Buffer& xh, const Buffer& y, int M, int epi, uint32_t y_ld,
                         uint32_t x_ld) {
  if (W.fmt == NhFmt::F32 || W.fmt == NhFmt::BF16 || W.E != 1) fatal("enc_gemm: unsupported matrix");
  // NVFP4 up to 32 rows: the x4 kernels; up to 8 rows 8 simdgroups split K (in_proj 0.207 vs 0.260 ms, out_proj
  // 0.097 vs 0.141), 9-32 rows SK parts of K summed in a fixed order by the last part to arrive (in_proj 32 rows
  // 0.72 vs 0.91 ms, out_proj 0.34 vs 0.56)
  if (W.fmt == NhFmt::FP4 && x4_ && M <= 32 && W.K % 128 == 0 && (x_ld ? x_ld : W.K) % 8 == 0 && (M > 8 || v8_)) {
    const V8P p{W.N, W.K, x_ld ? x_ld : W.K, y_ld ? y_ld : W.N, uint32_t(M), W.gs, uint32_t(epi)};
    const uint32_t tb = M <= 8 ? 1 : M <= 16 ? 2 : M <= 24 ? 3 : 4, sk = W.K <= 8192 ? 4 : 8, tiles = (W.N + 63) / 64;
    if (tb == 1 || (size_t(sk) * tiles * 2048 * tb <= x4part_.size && tiles * 4 <= x4cnt_.size)) {
      e.use(pipe(tb == 1 ? "nh_x4_v8_l0" : tb == 2 ? "nh_x4_mm16_l0" : tb == 3 ? "nh_x4_mm24_l0" : "nh_x4_mm32_l0"));
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
  // up to 8 rows (verify, the drafter's block): the 8-row MMA kernels (weights decoded into registers)
  if (M <= 8 && v8_) {
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
  // rows per tile: 16 / 32 for short inputs (less padding), else 64
  const int bm = M <= 16 ? 16 : M <= 32 ? 32 : 64;
  const std::string name = std::string(W.fmt == NhFmt::Q8 ? "nh_gemm_q8" : W.fmt == NhFmt::FP4 ? "nh_gemm_fp4" : "nh_gemm_iq2") +
                           "_m" + std::to_string(bm);
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
      const AttnPfP ap{uint32_t(T), uint32_t(pos0), uint32_t(s.cap), uint32_t(cfg_.hq), uint32_t(cfg_.hkv),
                       1.0f / std::sqrt(float(cfg_.hd))};
      e.use(pipe("nh_attn_prefill"));
      e.buf(0, pq_);
      e.buf(1, s.kc[size_t(i)]);
      e.buf(2, s.vc[size_t(i)]);
      e.buf(3, poh_);
      e.val(4, ap);
      e.dispatch((uint32_t(T) + 31) / 32, uint32_t(cfg_.hq), 1, 128);
      enc_gemm(e, L.o, poh_, px_, T, 1);
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
      // latent, the pairs grouped by expert, routed experts, combine
      enc_gemm(e, L.lat_down, pxh_, plat_, T, 3);
      // rows per expert tile: as many as the pairs per expert make worth it
      const uint32_t per = P / uint32_t(L.E), bm = per >= 40 ? 64 : per >= 12 ? 32 : 16;
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
        e.use(pipe("nh_gemm_iq2_moe_up" + ms));
        e.buf(0, plat_);
        e.buf(1, L.up_e.q);
        e.buf(2, L.up_e.s);
        e.buf(3, ph_);
        e.val(4, gp);
        e.buf(5, ptiles_);
        e.buf(6, pntiles_);
        e.buf(7, psorted_);
        e.buf(8, iq2tab_);
        e.dispatch((uint32_t(L.I) + 63) / 64, max_tiles, 1, 128);
      }
      {
        const GemmP gp{P, uint32_t(cfg_.latent), uint32_t(L.I), uint32_t(L.I), uint32_t(cfg_.latent), 1.0f, 0,
                       uint32_t(L.down_e.q_exp()), uint32_t(L.down_e.s_exp()), k};
        e.use(pipe("nh_gemm_iq2_moe_dn" + ms));
        e.buf(0, ph_);
        e.buf(1, L.down_e.q);
        e.buf(2, L.down_e.s);
        e.buf(3, pout_);
        e.val(4, gp);
        e.buf(5, ptiles_);
        e.buf(6, pntiles_);
        e.buf(7, psorted_);
        e.buf(8, iq2tab_);
        e.dispatch((uint32_t(cfg_.latent) + 63) / 64, max_tiles, 1, 128);
      }
      const CombineP cp{uint32_t(cfg_.latent), k};
      e.use(pipe("nh_moe_combine"));
      e.buf(0, pout_);
      e.buf(1, pew_);
      e.buf(2, plo_);
      e.val(3, cp);
      e.dispatch(uint32_t(T), 1, 1, 256);
      // shared expert; both branches add into the residual stream
      if (ffn_ane) {
        enc_gemm(e, L.lat_up, plo_, px_, T, 1);
        ane_->ffn_join(e, i, px_, T);
      } else {
        enc_gemm(e, L.up_s, pxh_, psh_, T, 2);
        enc_gemm(e, L.lat_up, plo_, px_, T, 1);
        enc_gemm(e, L.down_s, psh_, px_, T, 1);
      }
    }
    // a command buffer per layer of a long chunk (~80 ms at 2048 rows instead of seconds), so the desktop's own GPU
    // work is not held back behind one long buffer (a split chunk already ends one at every Neural Engine handoff)
    if (!split && T >= 256 && i + 1 < cfg_.layers) e.split();
  }
}

}  // namespace ni
