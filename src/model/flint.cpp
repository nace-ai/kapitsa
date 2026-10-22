#include "model/flint.h"
#include "model/kparams.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>
#include <unordered_map>

namespace ni {

extern const char* const kFlintKernelSource;
extern const char* const kTensorKernelSource;

namespace {

// host mirrors of src/kernels/flint/flint.metal's parameter structs
struct PrepP {
  uint32_t R, pos0, Hq, Hkv, rows, ring, rope, chunk, qkv_ld;
  float eps;
  uint32_t tree;
};
struct AttnP {
  uint32_t R, pos0, Hq, Hkv, rows, ring, window, k0, chunk, nchunks;
  float scale;
};
struct AttnPfP {
  uint32_t T, pos0, Hq, Hkv, rows, ring, window;
  float scale;
};
struct StoreP {
  uint32_t T, t0, pos0, Hkv, rows, ring;
};
struct RouteP {
  uint32_t R, k;
};
struct RouterTopkP {
  uint32_t R, D, E, k;
  float eps;
  uint32_t write_xn;
};
struct AcceptP {
  uint32_t P, K;
};
struct SwigluP {
  uint32_t I, mode, k;
};

Pipeline pipe(const std::string& n) {
  static std::mutex mu;
  static std::unordered_map<std::string, Pipeline> cache;
  std::lock_guard<std::mutex> lk(mu);
  auto it = cache.find(n);
  if (it == cache.end()) it = cache.emplace(n, Gpu::get().pipeline(n)).first;
  return it->second;
}

nlohmann::json read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) return {};
  try {
    return nlohmann::json::parse(f);
  } catch (...) {
    return {};
  }
}

// GPU shader validation's instrumentation doubles a kernel's threadgroup memory: the 16-key attention tiles under it
// (NI_FLINT_K16=1 selects them too)
bool small_tiles() {
  static const bool v = getenv("MTL_SHADER_VALIDATION") || env_on("NI_FLINT_K16");
  return v;
}

uint32_t gpu_cores() {
  static const uint32_t c = uint32_t(std::max(1, Gpu::get().core_count()));
  return c;
}

// Metal 4's tensor_ops GEMMs for the dense 4-bit prefill products (the M5's matrix units; Qwen35's NI_QMT): macOS 26 and
// an Apple10 GPU. NI_QMT=0 turns them off.
bool qmt_available() {
  const char* v = getenv("NI_QMT");
  if ((v && !strcmp(v, "0")) || Gpu::get().apple_family() < 10) return false;
  if (__builtin_available(macOS 26.0, *)) return true;
  return false;
}

}  // namespace

// ---- config ----
bool Flint::is_model(const std::string& dir) {
  const nlohmann::json j = read_json(dir + "/config.json");
  return j.is_object() && j.value("model_type", std::string()) == "flint";
}

FlintConfig Flint::read_config(const std::string& dir) {
  const nlohmann::json j = read_json(dir + "/config.json");
  if (!j.is_object()) fatal("%s: no readable config.json", dir.c_str());
  auto num = [&](const char* k, double d) {
    const auto it = j.find(k);
    return it != j.end() && it->is_number() ? it->get<double>() : d;
  };
  FlintConfig c;
  c.hidden = int(num("hidden_size", 0));
  c.vocab = int(num("vocab_size", 0));
  c.layers = int(num("num_hidden_layers", 0));
  c.hq = int(num("num_attention_heads", 0));
  c.hkv = int(num("num_key_value_heads", 0));
  c.hd = int(num("head_dim", c.hq ? c.hidden / c.hq : 0));
  c.ffn = int(num("intermediate_size", 0));
  c.moe_I = int(num("moe_intermediate_size", 0));
  c.shared_I = int(num("shared_expert_intermediate_size", 0));
  c.experts = int(num("num_experts", num("num_local_experts", 0)));
  c.top_k = int(num("num_experts_per_tok", 0));
  c.window = int(num("sliding_window", 0));
  c.eps = float(num("rms_norm_eps", 1e-6));
  c.theta = num("rope_theta", 1e4);
  if (j.contains("rope_parameters") && j["rope_parameters"].is_object()) {
    const auto& rp = j["rope_parameters"];
    if (rp.value("rope_type", std::string("default")) != "default")
      fatal("%s: Flint RoPE other than the default is not supported", dir.c_str());
    c.theta = rp.value("rope_theta", c.theta);
  }
  const std::string score = j.value("router_score_function", j.value("scoring_func", std::string("sigmoid")));
  if (score != "sigmoid") fatal("%s: Flint routing by %s is not supported (sigmoid)", dir.c_str(), score.c_str());
  if (!j.value("norm_topk_prob", true)) fatal("%s: Flint without norm_topk_prob is not supported", dir.c_str());
  if (j.value("router_expert_bias", false)) fatal("%s: Flint with a router expert bias is not supported", dir.c_str());
  if (j.value("softmax_type", std::string("learnable")) != "learnable")
    fatal("%s: Flint attention without learned sinks is not supported", dir.c_str());
  if (j.value("tie_word_embeddings", false)) fatal("%s: Flint with tied embeddings is not supported", dir.c_str());
  std::vector<int> norope, dense;
  if (j.contains("no_rope_layer_indices")) norope = j["no_rope_layer_indices"].get<std::vector<int>>();
  if (j.contains("mlp_only_layers")) dense = j["mlp_only_layers"].get<std::vector<int>>();
  const nlohmann::json lt = j.value("layer_types", nlohmann::json::array());
  for (int l = 0; l < c.layers; ++l) {
    const std::string k = l < int(lt.size()) ? lt[size_t(l)].get<std::string>() : std::string("full_attention");
    if (k != "sliding_attention" && k != "full_attention")
      fatal("%s: layer %d is %s (Flint: sliding_attention or full_attention)", dir.c_str(), l, k.c_str());
    c.kind.push_back(k == "sliding_attention" ? 'S' : 'F');
    c.rope.push_back(std::find(norope.begin(), norope.end(), l) == norope.end());
    c.dense.push_back(std::find(dense.begin(), dense.end(), l) != dense.end());
  }
  if (j.contains("eos_token_id")) {
    const auto& e = j["eos_token_id"];
    if (e.is_array()) c.eos = e.get<std::vector<int>>();
    else if (e.is_number()) c.eos = {e.get<int>()};
  }
  c.bos = int(num("bos_token_id", -1));
  const nlohmann::json q = j.value("quantization", nlohmann::json::object());
  if (q.is_object() && !q.empty()) {
    c.bits = q.value("bits", 4);
    c.gs = q.value("group_size", 64);
    const nlohmann::json x = j.value("mlx_export", nlohmann::json::object());
    c.embed_bits = x.is_object() ? x.value("embedding_bits", c.bits) : c.bits;
    if (x.is_object() && x.value("routed_expert_bits", c.bits) != c.bits)
      fatal("%s: routed experts in %d bits (Flint: the projections' %d)", dir.c_str(), x.value("routed_expert_bits", 0),
            c.bits);
  } else {
    fatal("%s: an unquantized Flint checkpoint; Kapitsa runs Flint's MLX export (RecursiveSI/flint-int4)", dir.c_str());
  }
  if (c.bits != 4 || c.gs != 64 || (c.embed_bits != 4 && c.embed_bits != 8))
    fatal("%s: Flint in %d-bit group %d is not supported (MLX 4-bit group 64, as RecursiveSI/flint-int4)", dir.c_str(),
          c.bits, c.gs);
  if (c.hidden <= 0 || c.hidden % 256 || c.hidden > 4096 || c.vocab <= 0 || c.layers <= 0 || c.hq <= 0 || c.hkv <= 0 ||
      c.hq % c.hkv || c.hq / c.hkv > 8 || c.hd != 128 || c.experts != 256 || c.top_k < 1 || c.top_k > 8 ||
      c.window <= 0 || c.moe_I % 64 || c.shared_I % 64 || c.ffn % 64 ||
      size_t(c.top_k + 1) * size_t(std::max(c.moe_I, c.shared_I)) > 7168)
    fatal("%s: a Flint shape Kapitsa does not run (hidden %d, heads %d / %d of %d, %d experts top-%d, window %d)",
          dir.c_str(), c.hidden, c.hq, c.hkv, c.hd, c.experts, c.top_k, c.window);
  return c;
}

size_t Flint::kv_per_pos(const FlintConfig& c) {
  const size_t full = size_t(std::count(c.kind.begin(), c.kind.end(), 'F'));
  return full * 2 * size_t(c.hkv) * size_t(c.hd) * 2;
}

size_t Flint::ring_bytes(const FlintConfig& c) {
  const size_t swa = size_t(std::count(c.kind.begin(), c.kind.end(), 'S'));
  return swa * 2 * size_t(c.hkv) * size_t((c.window + kRingSpare + 7) / 8 * 8) * size_t(c.hd) * 2;
}

MemPlan Flint::plan(const std::string& path, int prefill_chunk) {
  MemPlan m;
  const FlintConfig c = read_config(path);
  const SafetensorsIndex idx(path);
  for (const auto& [name, t] : idx.all()) m.weights += t.len;
  const size_t D = size_t(c.hidden), T = size_t(std::max(kMaxRows, prefill_chunk)), k = size_t(c.top_k);
  const size_t QKV = size_t(c.hq + 2 * c.hkv) * size_t(c.hd), QD = size_t(c.hq) * c.hd, I = size_t(c.moe_I);
  const size_t P = T * k;
  m.activations = T * D * 4 + T * D * 2 + T * QKV * 4 + T * QD * 4 + 2 * T * size_t(c.hkv) * c.hd * 2 + T * QD * 2 +
                  2 * T * size_t(c.ffn) * 4 + T * size_t(c.ffn) * 2 + T * size_t(c.experts) * 4 + P * 4 +
                  T * (k + 1) * 4 + 2 * P * 4 + (P + size_t(c.experts) + 1) * 12 + P * I * 2 + P * D * 4 +
                  2 * T * size_t(c.shared_I) * 4 + T * size_t(c.shared_I) * 2 + T * D * 4 +
                  size_t(kMaxRows) * size_t(c.vocab) * 4;
  m.slot_state = 3 * ring_bytes(c);  // the rings and their copies in the slot's two checkpoints
  m.kv_per_pos = kv_per_pos(c);
  // one plain decode step: attention, the dense layers, router, shared and k routed experts per MoE layer, the head
  const double bpw = c.bits == 16 ? 2.0 : (c.bits + 32.0 / c.gs) / 8.0;
  size_t params = 0;
  for (int l = 0; l < c.layers; ++l) {
    params += D * (QKV + QD);
    if (c.dense[size_t(l)]) params += 3 * D * size_t(c.ffn);
    else params += 3 * D * (k * I + size_t(c.shared_I));
  }
  params += D * size_t(c.vocab);
  m.decode_params = params;
  size_t router = 0;
  for (int l = 0; l < c.layers; ++l)
    if (!c.dense[size_t(l)]) router += D * size_t(c.experts) * 2;
  m.decode_bytes = size_t(double(params) * bpw) + router;
  m.head_row_bytes = size_t(double(D) * bpw);
  m.hidden = c.hidden;
  m.layers = c.layers;
  m.ffn = 0;
  m.dense = false;
  m.ffn_ane = false;
  m.formats = c.bits == 16 ? "bf16" : std::to_string(c.bits) + "-bit group " + std::to_string(c.gs) + " + " +
                                          (c.embed_bits == 16 ? std::string("bf16") : std::to_string(c.embed_bits) + "-bit") +
                                          " embedding, bf16 router";
  return m;
}

// ---- load ----
Flint::Flint(const std::string& path, int prefill_chunk) {
  path_ = path;
  while (path_.size() > 1 && path_.back() == '/') path_.pop_back();
  if (!is_model(path_)) fatal("%s: not a Flint checkpoint folder (config.json)", path.c_str());
  cfg_ = read_config(path_);
  Gpu::get().compile_extra("Flint", kFlintKernelSource);
  qmt_ = cfg_.bits == 4 && qmt_available();
  if (qmt_) Gpu::get().compile_extra("tensor", kTensorKernelSource, true);
  load(prefill_chunk);
}

Flint::~Flint() = default;

void Flint::load(int prefill_chunk) {
  const double t0 = now_ms();
  tmax_ = std::max(kMaxRows, prefill_chunk);
  ring_rows_ = (cfg_.window + kRingSpare + 7) / 8 * 8;  // (a multiple of 8: flint_attn_dec's 8-key blocks never wrap)
  ws_.open_dir(path_);
  // a linear's tensors: codes (uint8 or uint32 words of the same stream) with bf16 scales / biases of the config's groups,
  // or bf16 weights; `E` stacked matrices for the experts
  auto mat = [&](const std::string& base, uint32_t E = 1, bool required = true) {
    FlintMat m;
    const Tensor* w = ws_.find(base + ".weight");
    if (!w) {
      if (required) fatal("%s: tensor %s.weight missing", path_.c_str(), base.c_str());
      return m;
    }
    const size_t nd = w->shape.size();
    if (nd != (E > 1 ? 3u : 2u) || (E > 1 && w->shape[0] != int64_t(E)))
      fatal("%s: %s.weight has an unexpected shape", path_.c_str(), base.c_str());
    m.w = w;
    m.E = E;
    m.N = uint32_t(w->shape[nd - 2]);
    const Tensor* s = ws_.find(base + ".scales");
    if (!s) {
      if (w->dtype != DType::BF16) fatal("%s: %s.weight is neither quantized nor bf16", path_.c_str(), base.c_str());
      m.bits = 16;
      m.gs = 0;
      m.K = uint32_t(w->shape[nd - 1]);
      return m;
    }
    m.s = s;
    m.b = ws_.find(base + ".biases");
    if (!m.b || (w->dtype != DType::U8 && w->dtype != DType::U32) || s->shape.size() != nd)
      fatal("%s: %s is not an MLX affine-quantized linear", path_.c_str(), base.c_str());
    m.gs = cfg_.gs;
    m.K = uint32_t(s->shape[nd - 1]) * uint32_t(m.gs);
    const size_t code_bits = size_t(w->shape[nd - 1]) * (w->dtype == DType::U8 ? 8 : 32);
    if (code_bits % m.K) fatal("%s: %s's codes do not match its scales", path_.c_str(), base.c_str());
    m.bits = int(code_bits / m.K);
    if (m.bits != 4 && m.bits != 8) fatal("%s: %s is %d-bit (Flint: 4 or 8)", path_.c_str(), base.c_str(), m.bits);
    return m;
  };
  auto vec = [&](const std::string& name) {
    const Tensor* t = ws_.find(name);
    if (!t || t->dtype != DType::BF16) fatal("%s: bf16 tensor %s missing", path_.c_str(), name.c_str());
    return t;
  };
  const FlintConfig& c = cfg_;
  embed_ = mat("model.embed_tokens");
  head_ = mat("lm_head");
  final_norm_ = vec("model.norm.weight");
  if (embed_.N != uint32_t(c.vocab) || embed_.K != uint32_t(c.hidden) || head_.N != uint32_t(c.vocab) ||
      head_.K != uint32_t(c.hidden) || embed_.bits == 16 || head_.bits == 16)
    fatal("%s: embedding or LM head shape differs from the config", path_.c_str());
  layers_.resize(size_t(c.layers));
  for (int l = 0; l < c.layers; ++l) {
    FlintLayer& L = layers_[size_t(l)];
    const std::string p = "model.layers." + std::to_string(l) + ".";
    L.kind = c.kind[size_t(l)];
    L.rope = c.rope[size_t(l)];
    L.dense = c.dense[size_t(l)];
    L.in_norm = vec(p + "input_layernorm.weight");
    L.post_norm = vec(p + "post_attention_layernorm.weight");
    L.qn = vec(p + "self_attn.q_norm.weight");
    L.kn = vec(p + "self_attn.k_norm.weight");
    L.sinks = vec(p + "self_attn.softmax_offset");
    L.q = mat(p + "self_attn.q_proj");
    L.k = mat(p + "self_attn.k_proj");
    L.v = mat(p + "self_attn.v_proj");
    L.o = mat(p + "self_attn.o_proj");
    if (L.q.N != uint32_t(c.hq * c.hd) || L.k.N != uint32_t(c.hkv * c.hd) || L.v.N != L.k.N ||
        L.o.K != uint32_t(c.hq * c.hd) || L.o.N != uint32_t(c.hidden) || L.sinks->numel() != c.hq)
      fatal("%s: layer %d's attention shapes differ from the config", path_.c_str(), l);
    if (L.dense) {
      L.mg = mat(p + "mlp.gate_proj");
      L.mu = mat(p + "mlp.up_proj");
      L.md = mat(p + "mlp.down_proj");
      if (L.mg.N != uint32_t(c.ffn) || L.mu.N != L.mg.N || L.md.K != L.mg.N)
        fatal("%s: layer %d's MLP shapes differ from the config", path_.c_str(), l);
    } else {
      const uint32_t E = uint32_t(c.experts);
      L.router = mat(p + "mlp.gate");
      if (L.router.bits != 16 || L.router.N != E || L.router.K != uint32_t(c.hidden))
        fatal("%s: layer %d's router is not bf16 [%d, %d]", path_.c_str(), l, c.experts, c.hidden);
      L.eg = mat(p + "mlp.switch_mlp.gate_proj", E);
      L.eu = mat(p + "mlp.switch_mlp.up_proj", E);
      L.ed = mat(p + "mlp.switch_mlp.down_proj", E);
      L.sg = mat(p + "mlp.shared_expert.gate_proj");
      L.su = mat(p + "mlp.shared_expert.up_proj");
      L.sd = mat(p + "mlp.shared_expert.down_proj");
      if (L.eg.N != uint32_t(c.moe_I) || L.ed.N != uint32_t(c.hidden) || L.sg.N != uint32_t(c.shared_I) ||
          L.sd.K != uint32_t(c.shared_I))
        fatal("%s: layer %d's MoE shapes differ from the config", path_.c_str(), l);
      for (const FlintMat* m : {&L.eg, &L.eu, &L.ed, &L.sg, &L.su, &L.sd})
        if (m->bits != 4 || m->gs != 64)
          fatal("%s: layer %d's experts are not 4-bit group 64 (the MoE kernels' format)", path_.c_str(), l);
    }
  }
  // RoPE's inverse frequencies, 1 / theta^(2i / D) (MLX's nn.RoPE, traditional=False)
  Gpu& g = Gpu::get();
  inv_freq_ = g.alloc(size_t(c.hd / 2) * 4, "flint_rope", false);
  for (int i = 0; i < c.hd / 2; ++i)
    inv_freq_.as<float>()[i] = float(1.0 / std::pow(c.theta, double(2 * i) / double(c.hd)));
  alloc_buffers();
  g.make_resident(ws_.buffers());
  logi("Flint: %d layers (%d full attention), %s, loaded in %.1f s", c.layers,
       int(std::count(c.kind.begin(), c.kind.end(), 'F')), formats_summary().c_str(), (now_ms() - t0) / 1000.0);
}

std::string Flint::formats_summary() const {
  auto f = [](int bits) { return bits == 16 ? std::string("bf16") : std::to_string(bits) + "-bit group 64"; };
  return f(cfg_.bits) + " (embedding " + f(embed_.bits) + ", LM head " + f(head_.bits) + ", router bf16)";
}

void Flint::alloc_buffers() {
  Gpu& g = Gpu::get();
  const FlintConfig& c = cfg_;
  const size_t R = kMaxRows, D = size_t(c.hidden), QD = size_t(c.hq) * c.hd, V = size_t(c.vocab);
  const size_t QKV = QD + 2 * size_t(c.hkv) * c.hd, k = size_t(c.top_k);
  const size_t max_chunks = 64;
  x_ = g.alloc(R * D * 4, "flint_x");
  xn_ = g.alloc(R * D * 4, "flint_xn");
  qkv_ = g.alloc(R * QKV * 4, "flint_qkv");
  q_ = g.alloc(R * QD * 4, "flint_q");
  ao_ = g.alloc(R * QD * 4, "flint_ao");
  mh_ = g.alloc(R * size_t(std::max(c.ffn, 1)) * 4, "flint_mlp_h");
  part_o_ = g.alloc(R * size_t(c.hq) * max_chunks * c.hd * 4, "flint_part_o");
  part_ml_ = g.alloc(R * size_t(c.hq) * max_chunks * 2 * 4, "flint_part_ml");
  rlog_ = g.alloc(R * size_t(c.experts) * 4, "flint_rlog");
  eidx_ = g.alloc(R * k * 4 + 16, "flint_eidx");
  ewts_ = g.alloc(R * (k + 1) * 4 + 16, "flint_ewts");
  eh_ = g.alloc(R * (k + 1) * size_t(std::max(c.moe_I, c.shared_I)) * 4, "flint_eh");
  uq_ = g.alloc(1024, "flint_unique_experts");
  logits_ = g.alloc(R * V * 4, "flint_logits");
  cand_v_ = g.alloc(R * (V / 2048 + 2) * 64 * 4, "flint_cand_v");
  cand_i_ = g.alloc(R * (V / 2048 + 2) * 64 * 4, "flint_cand_i");
  sprob_ = g.alloc(64 * 4, "flint_sprob");
  samp_ = g.alloc(64 * 4, "flint_samples");
  acc_ = g.alloc(16, "flint_accepted");
  rcnt_ = g.alloc(16, "flint_router_arrivals");
  memset(rcnt_.ptr(), 0, 16);
  acnt_ = g.alloc(size_t(std::max(c.hkv, 4)) * 4, "flint_attn_arrivals");
  memset(acnt_.ptr(), 0, acnt_.size);
}

void Flint::alloc_prefill() {
  if (prefill_alloc_) return;
  Gpu& g = Gpu::get();
  const FlintConfig& c = cfg_;
  const size_t T = size_t(tmax_), D = size_t(c.hidden), QD = size_t(c.hq) * c.hd, KVD = size_t(c.hkv) * c.hd;
  const size_t k = size_t(c.top_k), P = T * k, F = size_t(std::max(c.ffn, 1)), SI = size_t(c.shared_I);
  px_ = g.alloc(T * D * 4, "flint_px");
  pxh_ = g.alloc(T * std::max(D, QD) * 2, "flint_pxh");
  pqkv_ = g.alloc(T * (QD + 2 * KVD) * 4, "flint_pqkv");
  pq_ = g.alloc(T * QD * 4, "flint_pq");
  pck_ = g.alloc(T * KVD * 2, "flint_pck");
  pcv_ = g.alloc(T * KVD * 2, "flint_pcv");
  pao_ = g.alloc(T * QD * 2, "flint_pao");
  const size_t GU = std::max(T * F, P * size_t(c.moe_I));  // a dense MLP's rows, or the routed pairs'
  phg_ = g.alloc(GU * 4, "flint_phg");
  phu_ = g.alloc(GU * 4, "flint_phu");
  phh_ = g.alloc(T * F * 2, "flint_phh");
  prlog_ = g.alloc(T * size_t(c.experts) * 4, "flint_prlog");
  pidx_ = g.alloc(P * 4 + 16, "flint_pidx");
  pwts_ = g.alloc(T * (k + 1) * 4 + 16, "flint_pwts");
  spair_ = g.alloc(P * 4 + 16, "flint_spair");
  stok_ = g.alloc(P * 4 + 16, "flint_stok");
  tiles_ = g.alloc((P + size_t(c.experts) + 1) * 3 * 4, "flint_tiles");
  ntiles_ = g.alloc(16, "flint_ntiles");
  ehh_ = g.alloc(P * size_t(c.moe_I) * 2, "flint_ehh");
  yd_ = g.alloc(P * D * 4, "flint_yd");
  sgb_ = g.alloc(T * SI * 4, "flint_sgb");
  sub_ = g.alloc(T * SI * 4, "flint_sub");
  shh_ = g.alloc(T * SI * 2, "flint_shh");
  ys_ = g.alloc(T * D * 4, "flint_ys");
  pscale_ = g.alloc(T * 4, "flint_pscale");
  prefill_alloc_ = true;
}

// ---- sequences, KV caches, checkpoints ----
std::unique_ptr<Seq> Flint::new_seq(int limit, int initial) {
  Gpu& g = Gpu::get();
  auto s = std::make_unique<Seq>();
  s->limit = limit;
  s->tok = g.alloc(size_t(limit + 8) * 4, "tokens");
  s->kc.resize(size_t(cfg_.layers));
  s->vc.resize(size_t(cfg_.layers));
  const int cap = initial < 0 ? limit : std::min(initial, limit);
  if (cap > 0 && !reserve(*s, cap)) fatal("insufficient memory for a %d-token KV cache", cap);
  return s;
}

// The sliding-window layers' rings (allocated with the first cache). False when the budget does not allow them.
bool Flint::ensure_rings(Seq& s) {
  if (s.cap <= 0) return true;
  Gpu& g = Gpu::get();
  const size_t row = size_t(cfg_.hkv) * cfg_.hd * 2;
  size_t need = 0;
  for (int l = 0; l < cfg_.layers; ++l)
    if (cfg_.kind[size_t(l)] == 'S' && !s.kc[size_t(l)]) need += 2 * row * size_t(ring_rows_);
  if (!need) return true;
  if (mem_budget_ && g.allocated_bytes() + need > mem_budget_) return false;
  for (int l = 0; l < cfg_.layers; ++l)
    if (cfg_.kind[size_t(l)] == 'S' && !s.kc[size_t(l)]) {
      // zeroed: flint_attn_dec's last block reads slots no position has written yet (its scores masked, its values
      // multiplied by 0, which a NaN left in fresh memory would not survive)
      s.kc[size_t(l)] = g.alloc(row * size_t(ring_rows_), "flint_kv_ring");
      s.vc[size_t(l)] = g.alloc(row * size_t(ring_rows_), "flint_kv_ring");
      memset(s.kc[size_t(l)].ptr(), 0, s.kc[size_t(l)].size);
      memset(s.vc[size_t(l)].ptr(), 0, s.vc[size_t(l)].size);
    }
  return true;
}

bool Flint::reserve(Seq& s, int need) {
  if (need <= s.cap) return ensure_rings(s);
  if (need > s.limit) return false;
  Gpu& g = Gpu::get();
  const size_t per = kv_per_pos(cfg_), row = size_t(cfg_.hkv) * cfg_.hd * 2;
  auto round_up = [](int v, int m) { return (v + m - 1) / m * m; };
  const size_t rings = s.cap > 0 ? 0 : ring_bytes(cfg_);
  auto fits = [&](int c) {
    const size_t copy = std::min(size_t(256) << 20, size_t(s.cap) * per);
    return !mem_budget_ || g.allocated_bytes() + size_t(c - s.cap) * per + copy + rings <= mem_budget_;
  };
  int target = std::min(s.limit, round_up(std::max(need, s.cap + s.cap / 4), 1024));
  if (!fits(target)) target = std::min(s.limit, round_up(need, 256));
  if (!fits(target)) return false;
  const int old = s.cap;
  for (int l = 0; l < cfg_.layers; ++l) {
    if (cfg_.kind[size_t(l)] != 'F') continue;
    for (Buffer* b : {&s.kc[size_t(l)], &s.vc[size_t(l)]}) {
      Buffer nbuf = g.alloc(row * size_t(target), "flint_kv");
      memset(nbuf.ptr(), 0, nbuf.size);  // (as the rings: flint_attn_dec reads past the last position)
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
      if (*b) g.release(*b);
      *b = nbuf;
    }
  }
  s.cap = target;
  return ensure_rings(s);
}

void Flint::drop_kv(Seq& s) {
  Gpu& g = Gpu::get();
  for (auto* v : {&s.kc, &s.vc})
    for (auto& b : *v) {
      g.release(b);
      b = Buffer{};
    }
  s.cap = 0;
}

void Flint::reset(Seq& s) { (void)s; }  // no recurrent state: every position is written before it is read

// A checkpoint holds copies of the sliding-window rings (made on the first save): the linear caches keep every earlier
// position, the rings only the last window + 8.
Checkpoint Flint::new_checkpoint() { return Checkpoint{}; }

void Flint::save(Encoder& e, const Seq& s, Checkpoint& c, int pos) {
  Gpu& g = Gpu::get();
  c.pos = -1;
  const size_t n = size_t(cfg_.layers);
  if (c.rec.size() != 2 * n) c.rec.assign(2 * n, Buffer{});
  for (size_t i = 0; i < n; ++i)
    for (int kv = 0; kv < 2; ++kv) {
      const Buffer& src = kv ? s.vc[i] : s.kc[i];
      Buffer& dst = c.rec[2 * i + size_t(kv)];
      if (cfg_.kind[i] != 'S' || !src) {
        if (dst) g.release(dst), dst = Buffer{};
        continue;
      }
      if (!dst) {
        if (mem_budget_ && g.allocated_bytes() + src.size > mem_budget_) return;  // no room: no checkpoint
        dst = g.alloc(src.size, "flint_ring_ckpt");
      }
      const uint32_t words = uint32_t(src.size / 16);
      e.use(pipe("copy_u32"));
      e.buf(0, src);
      e.buf(1, dst);
      e.val(2, words);
      e.dispatch((words + 255) / 256, 1, 1, 256);
    }
  c.pos = pos;
}

void Flint::restore(Encoder& e, Seq& s, const Checkpoint& c) {
  for (size_t i = 0; i < size_t(cfg_.layers) && 2 * i + 1 < c.rec.size(); ++i)
    for (int kv = 0; kv < 2; ++kv) {
      const Buffer& src = c.rec[2 * i + size_t(kv)];
      const Buffer& dst = kv ? s.vc[i] : s.kc[i];
      if (!src || !dst) continue;
      const uint32_t words = uint32_t(std::min(src.size, dst.size) / 16);
      e.use(pipe("copy_u32"));
      e.buf(0, src);
      e.buf(1, dst);
      e.val(2, words);
      e.dispatch((words + 255) / 256, 1, 1, 256);
    }
}

std::string Flint::disk_format() const {
  char b[256];
  snprintf(b, sizeof b, "flint %s; K, V f16 [%d][cap][%d]; rings of %d; %s", std::string(cfg_.kind.begin(), cfg_.kind.end()).c_str(),
           cfg_.hkv, cfg_.hd, ring_rows_, formats_summary().c_str());
  return b;
}

void Flint::disk_rows(const Seq& s, bool mtp, std::vector<KvRows>& out) const {
  (void)mtp;
  for (int l = 0; l < cfg_.layers; ++l)
    if (cfg_.kind[size_t(l)] == 'F')
      for (const Buffer* b : {&s.kc[size_t(l)], &s.vc[size_t(l)]}) out.push_back({*b, cfg_.hkv, size_t(cfg_.hd) * 2});
}

void Flint::disk_state(const Seq& s, const Checkpoint* ck, std::vector<Buffer>& out) const {
  for (int l = 0; l < cfg_.layers; ++l)
    if (cfg_.kind[size_t(l)] == 'S')
      for (size_t kv = 0; kv < 2; ++kv)
        out.push_back(!ck ? (kv ? s.vc[size_t(l)] : s.kc[size_t(l)])
                          : 2 * size_t(l) + kv < ck->rec.size() ? ck->rec[2 * size_t(l) + kv] : Buffer{});
}

// ---- encoders ----
void Flint::enc_embed(Encoder& e, const Buffer& tokens, int tok_off, int R, const Buffer& out) {
  const uint32_t D = uint32_t(cfg_.hidden);
  const kp::Embed ep{D, uint32_t(embed_.gs), uint32_t(tok_off)};
  e.use(pipe(embed_.bits == 4 ? std::string("embed_q4") : "embed_f" + std::to_string(embed_.bits) + "g64"));
  e.buf(0, embed_.w->buf);
  e.buf(1, embed_.s->buf);
  e.buf(2, embed_.b->buf);
  e.buf(3, tokens);
  e.buf(4, out);
  e.val(5, ep);
  e.dispatch((D / 8 + 63) / 64, uint32_t(R), 1, 64);
}

void Flint::enc_mv(Encoder& e, const std::vector<Seg>& segs, const Buffer& x, int R, const Tensor* norm, uint32_t y_ld,
                   bool gu) {
  if (segs.empty() || segs.size() > 4 || R < 1 || R > kMaxRows || (gu && segs.size() != 2)) fatal("Flint: bad matvec");
  const int bits = segs[0].M->bits;
  const uint32_t K = segs[0].M->K;
  for (const Seg& s : segs)
    if (s.M->bits != bits || s.M->K != K || (bits != 4 && bits != 8) || s.M->gs != 64)
      fatal("Flint: matvec segments in another format than 4/8-bit group 64 (bf16 checkpoints are not wired yet)");
  // one threadgroup per token row and 8 weight rows (2 simdgroups x 4 rows), the rows of a weight block adjacent
  constexpr int rows = 4, nsg = 2;
  char name[64];
  snprintf(name, sizeof name, "qmvf%s_b%d_t1_r%d%s", gu ? "_gu" : "", bits, rows, norm ? "_norm" : "");
  e.use(pipe(name));
  kp::Qmvf p{};
  p.K = K;
  p.R = uint32_t(R);
  p.x_ld = K;
  p.nseg = uint32_t(segs.size());
  p.eps = cfg_.eps;
  p.write_xn = 0;
  const uint32_t per_tg = uint32_t(rows * nsg);
  uint32_t tg = 0;
  for (size_t i = 0; i < 4; ++i) {
    const Seg& s = segs[std::min(i, segs.size() - 1)];
    if (i < segs.size()) {
      p.N[i] = s.M->N;
      p.y_ld[i] = y_ld ? y_ld : s.M->N;
      p.add[i] = s.add ? 1u : 0u;
      p.tg_begin[i] = tg;
      if (!gu || i == 0) tg += (s.M->N + per_tg - 1) / per_tg;
    }
    e.buf(int(i * 4 + 0), s.M->w->buf);
    e.buf(int(i * 4 + 1), s.M->s->buf);
    e.buf(int(i * 4 + 2), s.M->b->buf);
    e.buf(int(i * 4 + 3), s.y);
  }
  if (gu) tg = p.tg_begin[1];
  p.tg_begin[segs.size()] = tg;
  e.buf(16, x);
  e.val(17, p);
  e.buf(18, norm ? norm->buf : x);
  e.buf(19, x);
  e.dispatch(uint32_t(R), tg, 1, 32 * uint32_t(nsg));
}

void Flint::enc_rows_half(Encoder& e, const Buffer& x, const Tensor* norm, const Buffer& y, int T) {
  const kp::Rows p{uint32_t(cfg_.hidden), norm ? 1u : 0u, cfg_.eps};
  e.use(pipe("rows_to_half"));
  e.buf(0, x);
  e.buf(1, norm ? norm->buf : x);
  e.buf(2, y);
  e.val(3, p);
  e.dispatch(uint32_t(T), 1, 1, 256);
}

void Flint::enc_gemm(Encoder& e, const FlintMat& W, const Buffer& xh, const Buffer& y, int M, bool add, uint32_t y_ld) {
  const kp::Qmm p{uint32_t(M), W.N, W.K, W.K, y_ld ? y_ld : W.N, add ? 1u : 0u, uint32_t(std::max(W.gs, 1)), 0, 0};
  std::string n;
  uint32_t bm = 32, bn = 128, th = 128;
  if (W.bits == 16) {  // bf16 (the router): qmr's tiles for bf16
    if (M <= 8) n = "qrf_f16_m8_n64", bm = 8, bn = 64, th = 64;
    else if (M < 64 || W.N <= 64) n = "qrf_f16_m16_n64", bm = 16, bn = 64;
    else n = "qrf_f16_m32_n128";
  } else if (W.bits == 4 && qmt_ && M >= 64) {
    n = "qmt_m64_n64", bm = 64, bn = 64;
  } else if (W.bits == 4) {
    if (M <= 8) n = "qmr_k64_w12_m8_n64", bm = 8, bn = 64, th = 64;
    else if (M < 64 || W.N <= 64) n = "qmr_k64_w14_m16_n64", bm = 16, bn = 64;
    else if (Gpu::get().apple_family() <= 7) n = "qmr_k64_w12_m32_n64", bn = 64, th = 64;
    else n = "qmr_k32_w14_m32_n128";
  } else {  // 8-bit group 64
    if (M < 64 || W.N <= 64) n = "qmr8_k32_w14_m16_n64", bm = 16, bn = 64;
    else n = "qmr8_k32_w14_m32_n128";
  }
  e.use(pipe(n));
  e.buf(0, xh);
  e.buf(1, W.w->buf);
  e.buf(2, W.bits == 16 ? W.w->buf : W.s->buf);
  e.buf(3, W.bits == 16 ? W.w->buf : W.b->buf);
  e.buf(4, y);
  e.val(5, p);
  e.dispatch((W.N + bn - 1) / bn, (uint32_t(M) + bm - 1) / bm, 1, th);
}

const float* Flint::rope_cs(int pos) {
  if (pos != rope_pos_) {
    const int n = cfg_.hd / 2;
    rope_cs_.resize(size_t(2 * n));
    const float* inv = inv_freq_.as<float>();
    for (int i = 0; i < n; ++i) {
      const float ang = float(pos) * inv[i];  // as flint_prep's
      rope_cs_[size_t(2 * i)] = float(std::cos(double(ang)));
      rope_cs_[size_t(2 * i + 1)] = float(std::sin(double(ang)));
    }
    rope_pos_ = pos;
  }
  return rope_cs_.data();
}

// One layer of R <= kMaxRows rows (decode, verify, short prompts): x_ in place
void Flint::enc_layer_rows(Encoder& e, Seq& s, int li, int pos0, int R) {
  const FlintConfig& c = cfg_;
  const FlintLayer& L = layers_[size_t(li)];
  const uint32_t QD = uint32_t(c.hq * c.hd), KD = uint32_t(c.hkv * c.hd), QKV = QD + 2 * KD;
  const bool ring = L.kind == 'S';
  const uint32_t rows = uint32_t(cache_rows(s, li));
  if (!s.kc[size_t(li)]) fatal("Flint: layer %d's cache is not allocated", li);
  if (!ring && pos0 + R > s.cap) fatal("context full (%d)", s.cap);
  // q | k | v of the normalized rows, then their norms, RoPE and the cache
  enc_mv(e, {{&L.q, qkv_}, {&L.k, qkv_.view(size_t(QD) * 4, qkv_.size - size_t(QD) * 4)},
             {&L.v, qkv_.view(size_t(QD + KD) * 4, qkv_.size - size_t(QD + KD) * 4)}},
         x_, R, L.in_norm, QKV);
  const uint32_t window = ring ? uint32_t(c.window) : 0u;
  const uint32_t k0 = window && uint32_t(pos0) + 1 > window ? uint32_t(pos0) + 1 - window : 0u;
  // NI_FLINT_ATTN=0: a decode row on the verify rows' three dispatches instead (prep, partial, reduce)
  static const bool attn1 = !getenv("NI_FLINT_ATTN") || atoi(getenv("NI_FLINT_ATTN")) != 0;
  if (R == 1 && attn1 && c.hq / c.hkv == 8 && c.hd == 128) {
    // one decode row in one dispatch: q / k norms, RoPE, the cache write, split-K attention over 32-key tiles (16 under
    // shader validation) and the merge (M4 Pro, 1K context: 9 against 19 us a sliding-window layer, 15 against 29 a
    // full-attention one for prep + partial + reduce; 7.95 against 8.3 ms a token). At most 12 splits per KV head up
    // to 64 tiles, then 24: a full-attention layer at 16K / 32K keys 142 / 264 us (236 / 248 GB/s) against 150 / 281
    // with 12 and 155 / 285 with 48
    const uint32_t tile = small_tiles() ? 16u : 32u;
    const uint32_t T = (uint32_t(pos0) + 1 - k0 + tile - 1) / tile;
    uint32_t S = std::min(T, T <= 64 ? 12u : 24u);
    const uint32_t tps = (T + S - 1) / S;
    S = (T + tps - 1) / tps;
    const struct {
      uint32_t pos, Hq, Hkv, rows, ring, rope, k0, tps, S;
      float eps, scale;
    } ap{uint32_t(pos0), uint32_t(c.hq), uint32_t(c.hkv), rows, ring ? 1u : 0u, L.rope ? 1u : 0u, k0, tps, S, c.eps,
         1.0f / std::sqrt(float(c.hd))};
    e.use(pipe(small_tiles() ? "flint_attn_d2_t16" : "flint_attn_d2"));
    e.buf(0, qkv_);
    e.buf(1, L.qn->buf);
    e.buf(2, L.kn->buf);
    e.bytes(3, rope_cs(pos0), size_t(c.hd) * 4);
    e.buf(4, s.kc[size_t(li)]);
    e.buf(5, s.vc[size_t(li)]);
    e.buf(6, L.sinks->buf);
    e.buf(7, ao_);
    e.buf(8, part_o_);
    e.buf(9, part_ml_);
    e.buf(10, acnt_);
    e.bytes(11, &ap, sizeof ap);
    e.dispatch(S, uint32_t(c.hkv), 1, 128);
  } else {
    const PrepP pp{uint32_t(R), uint32_t(pos0), uint32_t(c.hq), uint32_t(c.hkv), rows, ring ? 1u : 0u,
                   L.rope ? 1u : 0u, 0, QKV, c.eps, 0};
    e.use(pipe("flint_prep"));
    e.buf(0, qkv_);
    e.buf(1, q_);
    e.buf(2, L.qn->buf);
    e.buf(3, L.kn->buf);
    e.buf(4, inv_freq_);
    e.buf(5, s.kc[size_t(li)]);
    e.buf(6, s.vc[size_t(li)]);
    e.val(7, pp);
    e.buf(8, inv_freq_);  // (no tree)
    e.dispatch(uint32_t(c.hq + 2 * c.hkv), uint32_t(R), 1, 128);
    // keys [k0, pos0 + R) in chunks: one per GPU core and kv head, at least 64 keys, at most 64 chunks
    const uint32_t Lk = uint32_t(pos0 + R) - k0;
    uint32_t chunk = std::max<uint32_t>(64, (Lk + gpu_cores() - 1) / gpu_cores());
    chunk = std::max(chunk, (Lk + 63) / 64);
    chunk = (chunk + 31) / 32 * 32;
    const uint32_t nch = (Lk + chunk - 1) / chunk;
    const AttnP ap{uint32_t(R), uint32_t(pos0), uint32_t(c.hq), uint32_t(c.hkv), rows, ring ? 1u : 0u, window, k0,
                   chunk, nch, 1.0f / std::sqrt(float(c.hd))};
    e.use(pipe(small_tiles() ? "flint_attn_partial_k16" : "flint_attn_partial"));
    e.buf(0, q_);
    e.buf(1, s.kc[size_t(li)]);
    e.buf(2, s.vc[size_t(li)]);
    e.buf(3, part_o_);
    e.buf(4, part_ml_);
    e.val(5, ap);
    e.buf(6, part_ml_);  // (no tree)
    e.dispatch(nch, uint32_t(c.hkv), uint32_t(R), 32 * uint32_t(c.hq / c.hkv));
    e.use(pipe("flint_attn_reduce"));
    e.buf(0, part_o_);
    e.buf(1, part_ml_);
    e.buf(2, L.sinks->buf);
    e.buf(3, ao_);
    e.val(4, ap);
    e.dispatch(uint32_t(c.hq), uint32_t(R), 1, uint32_t(c.hd));
  }
  enc_mv(e, {{&L.o, x_, true}}, ao_, R, nullptr);
  // MLP
  if (L.dense) {
    enc_mv(e, {{&L.mg, mh_}, {&L.mu, mh_}}, x_, R, L.post_norm, 0, true);
    enc_mv(e, {{&L.md, x_, true}}, mh_, R, nullptr);
  } else {
    enc_moe_rows(e, L, R);
  }
}

// MoE of R <= kMaxRows rows: router logits and the normalized rows, sigmoid top-k, the experts' gate / up (routed +
// shared) and down (weighted, into the residual)
void Flint::enc_moe_rows(Encoder& e, const FlintLayer& L, int R) {
  const FlintConfig& c = cfg_;
  const uint32_t D = uint32_t(c.hidden), E = uint32_t(c.experts), k = uint32_t(c.top_k);
  const uint32_t I = uint32_t(std::max(c.moe_I, c.shared_I));
  const kp::Moe mp{D, uint32_t(c.moe_I), uint32_t(c.shared_I), k, c.eps};
  if (R == 1 && !small_tiles() && D == 2048) {
    // one decode row: the router with the shared expert's gate / up (its last threadgroup routes; the router's
    // latency overlaps the shared expert's weights; its router rows are 2048 wide, so another hidden size takes the
    // rows' path below), the routed experts' gate / up, then the down products of every slot into the residual
    const struct {
      uint32_t D, E, k, SI, hstride, nr;
      float eps;
    } rp{D, E, k, uint32_t(c.shared_I), I, E / 8, c.eps};
    e.use(pipe("flint_router_shared"));
    e.buf(0, x_); e.buf(1, L.post_norm->buf); e.buf(2, L.router.w->buf); e.buf(3, rlog_); e.buf(4, xn_);
    e.bytes(5, &rp, sizeof rp);
    e.buf(6, eidx_); e.buf(7, ewts_); e.buf(8, rcnt_);
    e.buf(9, L.sg.w->buf); e.buf(10, L.sg.s->buf); e.buf(11, L.sg.b->buf);
    e.buf(12, L.su.w->buf); e.buf(13, L.su.s->buf); e.buf(14, L.su.b->buf); e.buf(15, eh_);
    e.dispatch(E / 8 + (uint32_t(c.shared_I) + 15) / 16, 1, 1, 256);
    e.use(pipe("moe_gate_up3_r2"));
    e.buf(0, L.eg.w->buf); e.buf(1, L.eg.s->buf); e.buf(2, L.eg.b->buf);
    e.buf(3, L.eu.w->buf); e.buf(4, L.eu.s->buf); e.buf(5, L.eu.b->buf);
    e.buf(6, L.sg.w->buf); e.buf(7, L.sg.s->buf); e.buf(8, L.sg.b->buf);
    e.buf(9, L.su.w->buf); e.buf(10, L.su.s->buf); e.buf(11, L.su.b->buf);
    e.buf(12, xn_); e.buf(13, rlog_); e.buf(14, rlog_); e.buf(15, eh_);
    e.val(16, mp);
    e.buf(17, eidx_); e.buf(18, ewts_); e.val(19, E);
    e.dispatch((uint32_t(c.moe_I) + 15) / 16, k, 1, 256);  // the routed slots [0, k) alone
    e.use(pipe("moe_down"));
    e.buf(0, L.ed.w->buf); e.buf(1, L.ed.s->buf); e.buf(2, L.ed.b->buf);
    e.buf(3, L.sd.w->buf); e.buf(4, L.sd.s->buf); e.buf(5, L.sd.b->buf);
    e.buf(6, eh_); e.buf(7, eidx_); e.buf(8, ewts_); e.buf(9, x_);
    e.val(10, mp);
    e.dispatch((D + 7) / 8, 1, 1, 64);  // 2 simdgroups x 4 rows per threadgroup (bench: 201 against 171 GB/s at 8)
    return;
  }
  // logits, top-k and the normalized rows in one dispatch (the last threadgroup to finish routes)
  constexpr uint32_t rnsg = 4;
  const RouterTopkP rp{uint32_t(R), D, E, k, c.eps, 1};
  e.use(pipe("flint_router_topk"));
  e.buf(0, x_);
  e.buf(1, L.post_norm->buf);
  e.buf(2, L.router.w->buf);
  e.buf(3, rlog_);
  e.buf(4, xn_);
  e.val(5, rp);
  e.buf(6, eidx_);
  e.buf(7, ewts_);
  e.buf(8, rcnt_);
  e.dispatch(E / rnsg, 1, 1, 32 * rnsg);
  if (small_tiles()) {
    // under GPU shader validation, which doubles a kernel's threadgroup memory (moe_gate_up3's 16 KB of staged input,
    // moe_down2_w's and moe4_down's 28 KB of staged hidden rows): the first decode kernels, which stage nothing (the
    // same products, the norm of x computed in each threadgroup)
    constexpr uint32_t nsg = 8;
    for (int r = 0; r < R; ++r) {
      const size_t xo = size_t(r) * D * 4;
      e.use(pipe("moe_gate_up"));
      e.buf(0, L.eg.w->buf); e.buf(1, L.eg.s->buf); e.buf(2, L.eg.b->buf);
      e.buf(3, L.eu.w->buf); e.buf(4, L.eu.s->buf); e.buf(5, L.eu.b->buf);
      e.buf(6, L.sg.w->buf); e.buf(7, L.sg.s->buf); e.buf(8, L.sg.b->buf);
      e.buf(9, L.su.w->buf); e.buf(10, L.su.s->buf); e.buf(11, L.su.b->buf);
      e.buf(12, x_, xo); e.buf(13, L.post_norm->buf); e.buf(14, eidx_, size_t(r) * k * 4); e.buf(15, eh_);
      e.val(16, mp);
      e.dispatch((I + 4 * nsg - 1) / (4 * nsg), k + 1, 1, 32 * nsg);
      e.use(pipe("moe_down"));
      e.buf(0, L.ed.w->buf); e.buf(1, L.ed.s->buf); e.buf(2, L.ed.b->buf);
      e.buf(3, L.sd.w->buf); e.buf(4, L.sd.s->buf); e.buf(5, L.sd.b->buf);
      e.buf(6, eh_); e.buf(7, eidx_, size_t(r) * k * 4); e.buf(8, ewts_, size_t(r) * (k + 1) * 4); e.buf(9, x_, xo);
      e.val(10, mp);
      e.dispatch((D + 4 * nsg - 1) / (4 * nsg), 1, 1, 32 * nsg);
    }
    return;
  }
  if (R <= 4 && (k + 1) * I <= 7168) {
    // 2-4 rows (a prompt-lookup verify): the distinct experts of the rows' top-k, each one's gate / up for the rows that
    // chose it (their threadgroups adjacent: the weights shared through the cache), then every row's down products
    // (moe4_*, the decode kernels' arithmetic: moe_gate_up3's and moe_down2_w's; M4 Pro, 1K context: a pass of 2 / 4
    // rows 13.0 / 22.7 ms against 14.3 / 24.6 with the per-row kernels below)
    const struct {
      uint32_t R, k;
    } rk{uint32_t(R), k};
    const Buffer ue = uq_.view(0, 256), umask = uq_.view(256, 256), uslot = uq_.view(512, 256), un = uq_.view(768, 16);
    e.use(pipe("moe4_unique"));
    e.buf(0, eidx_); e.buf(1, ue); e.buf(2, umask); e.buf(3, uslot); e.buf(4, un);
    e.val(5, rk);
    e.dispatch(1, 1, 1, 32);
    e.use(pipe("moe4_gate_up"));
    e.buf(0, L.eg.w->buf); e.buf(1, L.eg.s->buf); e.buf(2, L.eg.b->buf);
    e.buf(3, L.eu.w->buf); e.buf(4, L.eu.s->buf); e.buf(5, L.eu.b->buf);
    e.buf(6, L.sg.w->buf); e.buf(7, L.sg.s->buf); e.buf(8, L.sg.b->buf);
    e.buf(9, L.su.w->buf); e.buf(10, L.su.s->buf); e.buf(11, L.su.b->buf);
    e.buf(12, xn_); e.buf(13, eh_);
    e.val(14, mp);
    e.buf(15, ue); e.buf(16, umask); e.buf(17, uslot); e.buf(18, un);
    e.val(19, uint32_t(R));
    e.dispatch(uint32_t(R), (I + 15) / 16, uint32_t(R) * k + 1, 256);
    e.use(pipe("moe4_down"));
    e.buf(0, L.ed.w->buf); e.buf(1, L.ed.s->buf); e.buf(2, L.ed.b->buf);
    e.buf(3, L.sd.w->buf); e.buf(4, L.sd.s->buf); e.buf(5, L.sd.b->buf);
    e.buf(6, eh_); e.buf(7, eidx_); e.buf(8, ewts_); e.buf(9, x_);
    e.val(10, mp);
    e.dispatch(uint32_t(R), (D + 15) / 16, 1, 256);
    return;
  }
  // 5-8 rows: each row's gate / up (routed + shared) and down products in turn
  constexpr uint32_t nsg = 8, rows = 2;
  for (int r = 0; r < R; ++r) {
    const size_t xo = size_t(r) * D * 4;
    e.use(pipe("moe_gate_up3_r2"));
    e.buf(0, L.eg.w->buf); e.buf(1, L.eg.s->buf); e.buf(2, L.eg.b->buf);
    e.buf(3, L.eu.w->buf); e.buf(4, L.eu.s->buf); e.buf(5, L.eu.b->buf);
    e.buf(6, L.sg.w->buf); e.buf(7, L.sg.s->buf); e.buf(8, L.sg.b->buf);
    e.buf(9, L.su.w->buf); e.buf(10, L.su.s->buf); e.buf(11, L.su.b->buf);
    e.buf(12, xn_, xo); e.buf(13, rlog_); e.buf(14, rlog_); e.buf(15, eh_);
    e.val(16, mp);
    e.buf(17, eidx_, size_t(r) * k * 4); e.buf(18, ewts_, size_t(r) * (k + 1) * 4); e.val(19, E);
    e.dispatch((I + rows * nsg - 1) / (rows * nsg), k + 1, 1, 32 * nsg);
    e.use(pipe("moe_down2_w"));
    e.buf(0, L.ed.w->buf); e.buf(1, L.ed.s->buf); e.buf(2, L.ed.b->buf);
    e.buf(3, L.sd.w->buf); e.buf(4, L.sd.s->buf); e.buf(5, L.sd.b->buf);
    e.buf(6, eh_); e.buf(7, eidx_, size_t(r) * k * 4); e.buf(8, ewts_, size_t(r) * (k + 1) * 4); e.buf(9, x_, xo);
    e.val(10, mp);
    e.dispatch((D + rows * nsg - 1) / (rows * nsg), 1, 1, 32 * nsg);
  }
}

void Flint::enc_rows(Encoder& e, Seq& s, int pos0, int R) {
  if (R < 1 || R > kMaxRows) fatal("Flint: %d rows in one pass", R);
  enc_embed(e, s.tok, pos0, R, x_);
  for (int l = 0; l < cfg_.layers; ++l) enc_layer_rows(e, s, l, pos0, R);
}

// One layer of a prefill chunk of T rows (GEMMs on half rows): px_ in place. The chunk's keys go into the layer's cache
// after its attention (a ring holds only the window's positions, so the chunk reads its own keys from pck_ / pcv_).
void Flint::enc_layer_prefill(Encoder& e, Seq& s, int li, int pos0, int T) {
  const FlintConfig& c = cfg_;
  const FlintLayer& L = layers_[size_t(li)];
  const uint32_t QD = uint32_t(c.hq * c.hd), KD = uint32_t(c.hkv * c.hd), QKV = QD + 2 * KD;
  const bool ring = L.kind == 'S';
  const uint32_t rows = uint32_t(cache_rows(s, li));
  if (!s.kc[size_t(li)]) fatal("Flint: layer %d's cache is not allocated", li);
  if (!ring && pos0 + T > s.cap) fatal("context full (%d)", s.cap);
  enc_rows_half(e, px_, L.in_norm, pxh_, T);
  enc_gemm(e, L.q, pxh_, pqkv_, T, false, QKV);
  enc_gemm(e, L.k, pxh_, pqkv_.view(size_t(QD) * 4, pqkv_.size - size_t(QD) * 4), T, false, QKV);
  enc_gemm(e, L.v, pxh_, pqkv_.view(size_t(QD + KD) * 4, pqkv_.size - size_t(QD + KD) * 4), T, false, QKV);
  const PrepP pp{uint32_t(T), uint32_t(pos0), uint32_t(c.hq), uint32_t(c.hkv), uint32_t(T), 0, L.rope ? 1u : 0u, 1,
                 QKV, c.eps, 0};
  e.use(pipe("flint_prep"));
  e.buf(0, pqkv_);
  e.buf(1, pq_);
  e.buf(2, L.qn->buf);
  e.buf(3, L.kn->buf);
  e.buf(4, inv_freq_);
  e.buf(5, pck_);
  e.buf(6, pcv_);
  e.val(7, pp);
  e.buf(8, inv_freq_);  // (no tree)
  e.dispatch(uint32_t(c.hq + 2 * c.hkv), uint32_t(T), 1, 128);
  const AttnPfP ap{uint32_t(T), uint32_t(pos0), uint32_t(c.hq), uint32_t(c.hkv), rows, ring ? 1u : 0u,
                   ring ? uint32_t(c.window) : 0u, 1.0f / std::sqrt(float(c.hd))};
  e.use(pipe(small_tiles() ? "flint_attn_prefill_k16" : "flint_attn_prefill"));
  e.buf(0, pq_);
  e.buf(1, s.kc[size_t(li)]);
  e.buf(2, s.vc[size_t(li)]);
  e.buf(3, pck_);
  e.buf(4, pcv_);
  e.buf(5, L.sinks->buf);
  e.buf(6, pao_);
  e.val(7, ap);
  e.dispatch((uint32_t(T) + 7) / 8, uint32_t(c.hkv), 1, 32 * uint32_t(c.hq / c.hkv));
  const uint32_t t0 = ring && uint32_t(T) > rows ? uint32_t(T) - rows : 0u;
  const StoreP sp{uint32_t(T), t0, uint32_t(pos0), uint32_t(c.hkv), rows, ring ? 1u : 0u};
  e.use(pipe("flint_kv_store"));
  e.buf(0, pck_);
  e.buf(1, pcv_);
  e.buf(2, s.kc[size_t(li)]);
  e.buf(3, s.vc[size_t(li)]);
  e.val(4, sp);
  e.dispatch(uint32_t(T) - t0, 2 * uint32_t(c.hkv), 1, 32);
  enc_gemm(e, L.o, pao_, px_, T, true);
  enc_rows_half(e, px_, L.post_norm, pxh_, T);
  if (L.dense) {
    enc_gemm(e, L.mg, pxh_, phg_, T, false);
    enc_gemm(e, L.mu, pxh_, phu_, T, false);
    enc_swiglu(e, phg_, phu_, phh_, T, L.mg.N, 2);
    enc_gemm(e, L.md, phh_, ys_, T, false);
    const uint32_t D = uint32_t(c.hidden);
    e.use(pipe("flint_add_scaled"));
    e.buf(0, px_);
    e.buf(1, ys_);
    e.buf(2, pscale_);
    e.val(3, D);
    e.dispatch(uint32_t(T), 1, 1, 256);
  } else {
    enc_moe_prefill(e, L, T);
  }
}

// MoE of a prefill chunk: the router GEMM (bf16), sigmoid top-k per row, the (row, expert) pairs grouped by expert, the
// experts' gate and up as grouped GEMMs (fp32), their SwiGLU into half rows scaled where they would overflow
// (flint_swiglu_scaled: the scale goes into the pair's routing weight), the down products, the shared expert on the
// dense GEMMs the same way, then the weighted sum into the residual
void Flint::enc_moe_prefill(Encoder& e, const FlintLayer& L, int T) {
  const FlintConfig& c = cfg_;
  const uint32_t D = uint32_t(c.hidden), E = uint32_t(c.experts), k = uint32_t(c.top_k), P = uint32_t(T) * k;
  constexpr uint32_t bm = 16;
  enc_gemm(e, L.router, pxh_, prlog_, T, false);
  const RouteP tp{uint32_t(T), k};
  e.use(pipe("flint_route"));
  e.buf(0, prlog_);
  e.buf(1, pidx_);
  e.buf(2, pwts_);
  e.val(3, tp);
  e.dispatch((uint32_t(T) + 7) / 8, 1, 1, 256);
  const kp::Bucket bp{P, k, E, bm};
  e.use(pipe("moe_bucket"));
  e.buf(0, pidx_);
  e.buf(1, spair_);
  e.buf(2, stok_);
  e.buf(3, tiles_);
  e.buf(4, ntiles_);
  e.val(5, bp);
  e.dispatch(1, 1, 1, 1024);
  const uint32_t max_tiles = (P + bm - 1) / bm + E;
  // a grouped GEMM over the sorted pairs: X rows xrows[sorted row], Y rows yrows[sorted row] (pair ids)
  auto moe_gemm = [&](const FlintMat& W, const Buffer& xh, const Buffer& y, const Buffer& xrows) {
    const kp::Qmm p{P, W.N, W.K, W.K, W.N, 0, uint32_t(W.gs), uint32_t(size_t(W.N) * W.row_bytes() / 4),
                    W.N * (W.K / uint32_t(W.gs))};
    e.use(pipe("qmm_moe_b4_m16_n64"));
    e.buf(0, xh);
    e.buf(1, W.w->buf);
    e.buf(2, W.s->buf);
    e.buf(3, W.b->buf);
    e.buf(4, y);
    e.val(5, p);
    e.buf(6, tiles_);
    e.buf(7, ntiles_);
    e.buf(8, xrows);
    e.buf(9, spair_);
    e.dispatch((W.N + 63) / 64, max_tiles, 1, 128);
  };
  moe_gemm(L.eg, pxh_, phg_, stok_);
  moe_gemm(L.eu, pxh_, phu_, stok_);
  enc_swiglu(e, phg_, phu_, ehh_, int(P), L.eg.N, 0);
  moe_gemm(L.ed, ehh_, yd_, spair_);
  // the shared expert on every row
  enc_gemm(e, L.sg, pxh_, sgb_, T, false);
  enc_gemm(e, L.su, pxh_, sub_, T, false);
  enc_swiglu(e, sgb_, sub_, shh_, T, L.sg.N, 1);
  enc_gemm(e, L.sd, shh_, ys_, T, false);
  const kp::Combine cp{D, k};
  e.use(pipe("moe_combine"));
  e.buf(0, yd_);
  e.buf(1, ys_);
  e.buf(2, pwts_);
  e.buf(3, px_);
  e.val(4, cp);
  e.dispatch((D / 4 + 63) / 64, uint32_t(T), 1, 64);
}

void Flint::enc_swiglu(Encoder& e, const Buffer& g, const Buffer& u, const Buffer& h, int rows, uint32_t I, uint32_t mode) {
  const SwigluP p{I, mode, uint32_t(cfg_.top_k)};
  e.use(pipe("flint_swiglu_scaled"));
  e.buf(0, g);
  e.buf(1, u);
  e.buf(2, h);
  e.buf(3, pwts_);
  e.buf(4, pscale_);
  e.val(5, p);
  e.dispatch(uint32_t(rows), 1, 1, 256);
}

void Flint::enc_prefill_rows(Encoder& e, Seq& s, int pos0, int T) {
  alloc_prefill();
  enc_embed(e, s.tok, pos0, T, px_);
  for (int l = 0; l < cfg_.layers; ++l) {
    enc_layer_prefill(e, s, l, pos0, T);
    // a command buffer per layer of a long chunk, so the desktop's own GPU work is not held back behind one long one
    if (T >= 256 && l + 1 < cfg_.layers) e.split();
  }
}

void Flint::enc_head(Encoder& e, const Buffer& x, int R, const Buffer& out, size_t ld) {
  enc_mv(e, {{&head_, out}}, x, R, final_norm_, uint32_t(ld));
}

void Flint::enc_sample(Encoder& e, const Buffer& lrow, const Buffer& tokens, int out_index, const SamplingParams& sp,
                       uint32_t step) {
  const uint32_t K = sp.temperature > 0 ? uint32_t(std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32)) : 1u;
  const kp::TopkStage ts{uint32_t(cfg_.vocab), K, 2048};
  const uint32_t ntg = (uint32_t(cfg_.vocab) + 2047) / 2048;
  e.use(pipe("topk_stage1"));
  e.buf(0, lrow);
  e.buf(1, cand_v_);
  e.buf(2, cand_i_);
  e.val(3, ts);
  e.dispatch(ntg, 1, 1, 256);
  const kp::Sample p2{ntg * K, K, sp.temperature, sp.top_p, sp.seed, step, uint32_t(out_index)};
  e.use(pipe("sample_stage2"));
  e.buf(0, cand_v_);
  e.buf(1, cand_i_);
  e.buf(2, tokens);
  e.buf(3, sprob_);
  e.val(4, p2);
  e.dispatch(1, 1, 1, 1024);
}

void Flint::encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample) {
  enc_rows(e, s, pos, 1);
  if (!sample) return;
  enc_head(e, x_, 1, logits_, size_t(cfg_.vocab));
  enc_logit_mask(e, s, logits_, 1, cfg_.vocab, cfg_.vocab);  // structured output (nothing without masks)
  enc_sample(e, logits_, s.tok, pos + 1, sp, step);
}

void Flint::encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                           Checkpoint* snap, int snap_row) {
  (void)snap;
  (void)snap_row;
  if (T > tmax_) fatal("Flint: prefill of %d rows > chunk %d", T, tmax_);
  Buffer last;
  const size_t row = size_t(cfg_.hidden) * 4;
  if (T <= kMaxRows) {  // the decode rows' kernels
    enc_rows(e, s, pos0, T);
    last = x_.view(size_t(T - 1) * row, row);
  } else {
    enc_prefill_rows(e, s, pos0, T);
    last = px_.view(size_t(T - 1) * row, row);
  }
  if (!sample) return;
  enc_head(e, last, 1, logits_, size_t(cfg_.vocab));
  enc_logit_mask(e, s, logits_, 1, cfg_.vocab, cfg_.vocab);  // structured output (nothing without masks)
  enc_sample(e, logits_, s.tok, pos0 + T, sp, step);
}

// ---- speculative decoding ----
void Flint::encode_verify(Encoder& e, Seq& s, int pos, int R) {
  enc_rows(e, s, pos, R);
  enc_head(e, x_, R, logits_, size_t(cfg_.vocab));
}

void Flint::encode_accept(Encoder& e, Seq& s, int pos, int R, int K, const SamplingParams& sp) {
  const size_t V = size_t(cfg_.vocab);
  for (int r = 0; r < R; ++r) enc_sample(e, logits_.view(size_t(r) * V * 4, V * 4), samp_, r, sp, uint32_t(pos + r));
  const AcceptP ap{uint32_t(pos), uint32_t(std::min(K, R - 1))};
  e.use(pipe("flint_accept"));
  e.buf(0, samp_);
  e.buf(1, s.tok);
  e.buf(2, acc_);
  e.val(3, ap);
  e.dispatch(1, 1, 1, 1);
}

void Flint::encode_logits_rows(Encoder& e, Seq& s, int pos0, int T, const Buffer& out) {
  const size_t row = size_t(cfg_.hidden) * 4, V = size_t(cfg_.vocab);
  if (T <= kMaxRows) {
    enc_rows(e, s, pos0, T);
    enc_head(e, x_, T, out, V);
    return;
  }
  enc_prefill_rows(e, s, pos0, T);
  for (int r0 = 0; r0 < T; r0 += kMaxRows) {
    const int R = std::min(kMaxRows, T - r0);
    enc_head(e, px_.view(size_t(r0) * row, size_t(R) * row), R, out.view(size_t(r0) * V * 4, size_t(R) * V * 4), V);
  }
}

}  // namespace ni
