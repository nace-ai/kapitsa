#include "model/lilicorr.h"
#include "util/bf16.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace ni {

namespace {

Pipeline pipe(const std::string& n) { return NemotronH::pipeline(n); }

struct DconvP {
  uint32_t D, gs, side, R;
};
struct QkP {
  uint32_t H, HD, pos0, rope;
  float eps, theta;
};
struct RingKvP {
  uint32_t pos0, W, Hkv, HD;
};
struct LiliAttnP {
  uint32_t P, lo, W, Hq, Hkv, B;
  float scale;
};
struct CandP {
  uint32_t V, K, nchunks, row0, ld;
};
struct FeatP {
  uint32_t K, Hh;
  float eps;
};
struct BiasP {
  uint32_t D, n, act;
};
struct MhaP {
  uint32_t N, Hh, heads;
  float scale;
};
struct L2P {
  uint32_t D;
  float eps;
};
struct WalkP {
  uint32_t S, K, D, P;
  float ls;
};
struct LiliTableP {
  uint32_t S, K, D;
  float ls;
};
struct BlockTokP {
  uint32_t P, mask, B;
};
struct AnchorP {
  uint32_t pos0, R, D, nslots;
};
struct DepthP {
  uint32_t S, K;
  float ratio;
  uint32_t mode;
};

}  // namespace

// ---- discovery and plan ----
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

// A LiLiCorr drafter folder: config.json of a DFlash drafter with the LiLiCorr projector, and model.safetensors.
bool is_lilicorr(const std::string& dir, int hidden) {
  const nlohmann::json c = read_json(dir + "/config.json");
  if (!c.is_object() || !c.contains("dflash_config") || !c["dflash_config"].is_object()) return false;
  const nlohmann::json& d = c["dflash_config"];
  const bool lili = (d.contains("lilicorr_enabled") && d["lilicorr_enabled"].is_boolean() && d["lilicorr_enabled"].get<bool>()) ||
                    (d.contains("projector_type") && d["projector_type"] == "lilicorr");
  struct stat st;
  return lili && c.value("hidden_size", 0) == hidden && stat((dir + "/model.safetensors").c_str(), &st) == 0;
}

// as loaded: NVFP4 as stored, BF16 matrices quantized to Q8 (34 bytes per 32), BF16 vectors as fp32
size_t loaded_bytes(const SafetensorsIndex& idx) {
  size_t w = 0;
  for (const auto& [name, t] : idx.all()) {
    if (t.dtype == DType::BF16 && t.shape.size() == 2 && t.shape[1] % 32 == 0 && t.shape[1] >= 256)
      w += size_t(t.numel()) * 34 / 32;
    else if (t.dtype == DType::BF16)
      w += size_t(t.numel()) * 4;
    else
      w += t.len;
  }
  return w;
}

}  // namespace

std::string LiliCorr::find(const std::string& model_dir) {
  int hidden = 0;
  {
    // the language model's config: llm_config (Nemotron Omni releases), text_config, or the whole file
    const nlohmann::json j = read_json(model_dir + "/config.json");
    const nlohmann::json* t = &j;
    for (const char* k : {"llm_config", "text_config"})
      if (j.is_object() && j.contains(k) && j[k].is_object()) {
        t = &j[k];
        break;
      }
    if (t->is_object()) hidden = t->value("hidden_size", 0);
  }
  for (const char* sub : {"/dflash", "/lilicorr"})
    if (is_lilicorr(model_dir + sub, hidden)) return model_dir + sub;
  return "";
}

size_t LiliCorr::plan_bytes(const std::string& dir, size_t* per_slot) {
  const nlohmann::json c = read_json(dir + "/config.json");
  const SafetensorsIndex idx(dir);
  const size_t w = loaded_bytes(idx);
  const nlohmann::json& d = c.contains("dflash_config") ? c["dflash_config"] : c;
  const int64_t H = c.value("hidden_size", 4096), NL = c.value("num_hidden_layers", 5);
  const int64_t W = d.value("swa_window_size", c.value("sliding_window", 1024)), hkv = c.value("num_key_value_heads", 2);
  const int64_t hd = c.value("head_dim", 128);
  const size_t ring = size_t(NL) * 2 * size_t(hkv) * size_t(W) * size_t(hd) * 2 + size_t(kAnchors) * size_t(H) * 4;
  if (per_slot) *per_slot = ring;
  const int64_t nl = d.contains("target_layer_ids") ? int64_t(d["target_layer_ids"].size()) : 6;
  // prompt capture (window rows of every target layer) and its half copy, commit buffers (window rows), pass buffers
  const size_t bufs = size_t(W) * size_t(nl) * size_t(H) * 6 + size_t(W) * (size_t(H) * 6 + size_t(hkv) * size_t(hd) * 8) +
                      (size_t(96) << 20);
  return w + bufs;
}

// ---- loading ----
// NVFP4 (ModelOpt) as the target's kernels run it; BF16 matrices quantized to Q8 rows (drafts only propose: the
// target verifies every token).
NhMat LiliCorr::mat(const std::string& name) {
  const SafetensorsIndex& idx = *st_;
  const SafetensorsIndex::Entry& t = idx.get(name);
  const std::string base = name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0 ? name.substr(0, name.size() - 7) : name;
  const SafetensorsIndex::Entry* sc = idx.find(base + ".weight_scale");
  Gpu& g = Gpu::get();
  NhMat m;
  uint32_t N = 0, K = 0;
  if (!nh_describe(t, sc, m.fmt, N, K) || (m.fmt != NhFmt::FP4 && m.fmt != NhFmt::BF16))
    fatal("%s: tensor %s has a format the drafter does not run", path_.c_str(), name.c_str());
  m.N = N;
  m.K = K;
  std::vector<uint8_t> raw(t.len);
  idx.read(t, 0, raw.data(), t.len);
  if (m.fmt == NhFmt::FP4) {
    nh_row_bytes(m.fmt, m.K, m.q_row, m.s_row);
    m.q = g.alloc(m.q_row * N, name.c_str(), false);
    m.s = g.alloc(m.s_row * N, "scales", false);
    nh_repack_rows(m.fmt, m.K, raw.data(), N, m.q.as<uint8_t>(), nullptr);
    idx.read(*sc, 0, m.s.ptr(), m.s_row * N);
    if (const SafetensorsIndex::Entry* s2 = idx.find(base + ".weight_scale_2")) idx.read(*s2, 0, &m.gs, 4);
  } else {
    if (m.K % 32) fatal("%s: %s cannot be quantized (K = %u)", path_.c_str(), name.c_str(), m.K);
    m.fmt = NhFmt::Q8;
    nh_row_bytes(m.fmt, m.K, m.q_row, m.s_row);
    m.q = g.alloc(m.q_row * N, name.c_str(), false);
    m.s = g.alloc(m.s_row * N, "scales", false);
    std::vector<float> row(m.K);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
    for (uint32_t r = 0; r < N; ++r) {
      for (uint32_t i = 0; i < m.K; ++i) row[i] = bf16_to_f32(src[size_t(r) * m.K + i]);
      nh_quant_q8(row.data(), m.K, m.q.as<uint8_t>() + size_t(r) * m.q_row, m.s.as<uint8_t>() + size_t(r) * m.s_row);
    }
  }
  owned_.push_back(m.q);
  owned_.push_back(m.s);
  weight_bytes_ += m.q.size + m.s.size;
  return m;
}

std::vector<float> LiliCorr::host(const std::string& name) {
  const SafetensorsIndex::Entry& t = st_->get(name);
  if (t.dtype != DType::BF16 && t.dtype != DType::F32)
    fatal("%s: %s is neither BF16 nor F32", path_.c_str(), name.c_str());
  std::vector<uint8_t> raw(t.len);
  st_->read(t, 0, raw.data(), t.len);
  std::vector<float> v(size_t(t.numel()));
  for (size_t i = 0; i < v.size(); ++i) {
    if (t.dtype == DType::F32) {
      memcpy(&v[i], raw.data() + 4 * i, 4);
    } else {
      uint16_t h;
      memcpy(&h, raw.data() + 2 * i, 2);
      v[i] = bf16_to_f32(h);
    }
  }
  return v;
}

Buffer LiliCorr::upload(const std::vector<float>& v, const char* label) {
  Buffer b = Gpu::get().alloc(std::max<size_t>(v.size() * 4, 16), label, false);
  memcpy(b.ptr(), v.data(), v.size() * 4);
  owned_.push_back(b);
  weight_bytes_ += b.size;
  return b;
}

Buffer LiliCorr::vec(const std::string& name) { return upload(host(name), name.c_str()); }

LiliCorr::LiliCorr(NemotronH& target, const std::string& dir) : tg_(target), path_(dir) {
  while (path_.size() > 1 && path_.back() == '/') path_.pop_back();
  const NhConfig& tc = tg_.config();
  if (!is_lilicorr(path_, tc.hidden)) fatal("%s is not a LiLiCorr drafter for this model", dir.c_str());
  const nlohmann::json j = read_json(path_ + "/config.json");
  const nlohmann::json& d = j["dflash_config"];
  LiliConfig& c = cfg_;
  c.H = j.value("hidden_size", 0);
  c.NL = j.value("num_hidden_layers", 0);
  c.I = j.value("intermediate_size", 0);
  c.hq = j.value("num_attention_heads", 0);
  c.hkv = j.value("num_key_value_heads", 0);
  c.hd = j.value("head_dim", 128);
  c.block = j.value("block_size", 8);
  c.window = d.value("swa_window_size", j.value("sliding_window", 1024));
  c.conv_k = d.value("conv_kernel_size", 2);
  c.conv_gs = d.value("conv_group_size", 16);
  c.eps = j.value("rms_norm_eps", 1e-6f);
  c.theta = j.value("rope_theta", 10000.0f);
  c.mask_id = d.value("mask_token_id", -1);
  // HF target_layer_ids name layers whose OUTPUTS feed fc: the inputs of the next layers, which the target captures
  for (auto& v : d["target_layer_ids"]) c.target_layers.push_back(v.get<int>() + 1);
  c.lh = d.value("lilicorr_hidden_size", 0);
  c.topk = d.value("lilicorr_candidate_topk", 0);
  c.heads = d.value("lilicorr_num_heads", 0);
  c.lnl = d.value("lilicorr_num_layers", 0);
  c.factor = d.value("lilicorr_factor_dim", 0);
  c.mlp = int(std::lround(c.lh * d.value("lilicorr_mlp_ratio", 2.0)));
  c.vec_eps = d.value("lilicorr_vector_eps", 1e-4f);
  c.logit_scale = d.value("lilicorr_logit_scale", 8.0f);
  const bool causal = d.value("causal", false);
  if (c.H != tc.hidden || c.block != kB || c.hd != 128 || c.conv_k != 2 || c.mask_id < 0 || c.target_layers.empty() ||
      c.topk != 8 || c.lh != c.factor || !c.heads || c.lh / c.heads != 128 || c.window <= 0 || causal)
    fatal("%s: unsupported drafter shape (hidden %d, block %d, top-k %d)", dir.c_str(), c.H, c.block, c.topk);
  for (int l : c.target_layers)
    if (l < 1 || l >= tc.layers) fatal("%s: target layer %d out of range", dir.c_str(), l - 1);
  // reduced draft head (NI_LILI_HEAD=n rows; 0 or >= the vocabulary: the full head)
  head_rows_ = tg_.vocab();
  if (const char* v = getenv("NI_LILI_HEAD"); v && *v && atoi(v) > 0) head_rows_ = std::min(tg_.vocab(), atoi(v));
  else head_rows_ = std::min(tg_.vocab(), kHeadRows);
  const double t0 = now_ms();
  st_ = std::make_unique<SafetensorsIndex>(path_);
  has_own_head_ = j.value("has_own_lm_head", false) && st_->find("lm_head.weight");
  if (has_own_head_ && j.value("draft_vocab_size", tg_.vocab()) != tg_.vocab())
    fatal("%s: a draft vocabulary other than the target's is not supported", dir.c_str());
  load();
  st_.reset();
  alloc_buffers();
  logi("LiLiCorr drafter: %d layers, window %d, %d KV heads, %.2f GB (NVFP4 as stored, BF16 matrices as Q8), draft head "
       "%d of %d rows (%s), loaded in %.1f s",
       c.NL, c.window, c.hkv, weight_bytes_ / 1e9, head_rows_, tg_.vocab(), own_head_ ? "its own" : "the target's",
       (now_ms() - t0) / 1000);
}

LiliCorr::~LiliCorr() {
  Gpu& g = Gpu::get();
  for (auto& b : owned_) g.release(b);
}

void LiliCorr::load() {
  fc_ = mat("fc.weight");
  enc_norm_ = vec("hidden_norm.weight");
  out_norm_ = vec("norm.weight");
  dl_.resize(size_t(cfg_.NL));
  for (int i = 0; i < cfg_.NL; ++i) {
    DLayer& L = dl_[size_t(i)];
    const std::string b = "layers." + std::to_string(i) + ".";
    L.attn_norm = vec(b + "input_layernorm.weight");
    L.ffn_norm = vec(b + "post_attention_layernorm.weight");
    L.qn = vec(b + "self_attn.q_norm.weight");
    L.kn = vec(b + "self_attn.k_norm.weight");
    // attention sinks (Bolt's drafter; Super-VL's has none: a logit of -1e30 adds nothing to the softmax)
    if (st_->find(b + "self_attn.attention_sink_bias")) L.sinks = vec(b + "self_attn.attention_sink_bias");
    else L.sinks = upload(std::vector<float>(size_t(cfg_.hq), -1e30f), "lili_no_sinks");
    L.ac_base = vec(b + "attention_conv.base_kernel");
    L.fc_base = vec(b + "mlp_conv.base_kernel");
    L.ac_proj = mat(b + "attention_conv.kernel_projection.weight");
    L.fc_proj = mat(b + "mlp_conv.kernel_projection.weight");
    L.q = mat(b + "self_attn.q_proj.weight");
    L.k = mat(b + "self_attn.k_proj.weight");
    L.v = mat(b + "self_attn.v_proj.weight");
    L.o = mat(b + "self_attn.o_proj.weight");
    L.gate = mat(b + "mlp.gate_proj.weight");
    L.up = mat(b + "mlp.up_proj.weight");
    L.down = mat(b + "mlp.down_proj.weight");
  }
  // its own draft head (Super-VL's: an NVFP4 copy of the target's LM head), else the target's
  if (has_own_head_) own_head_ = mat("lm_head.weight");
  const std::string h = "lilicorr.";
  tok_proj_ = mat(h + "token_proj.weight");
  tok_b_ = vec(h + "token_proj.bias");
  pass_proj_ = mat(h + "pass_hidden_proj.weight");
  pass_b_ = vec(h + "pass_hidden_proj.bias");
  ctx_proj_ = mat(h + "context_proj.weight");
  ctx_b_ = vec(h + "context_proj.bias");
  feat_ln_w_ = vec(h + "feature_norm.weight");
  feat_ln_b_ = vec(h + "feature_norm.bias");
  mlp1_w_ = vec(h + "feature_mlp.up_proj.weight");  // [hidden][5]
  mlp1_b_ = vec(h + "feature_mlp.up_proj.bias");
  mlp3_ = mat(h + "feature_mlp.down_proj.weight");
  mlp3_b_ = vec(h + "feature_mlp.down_proj.bias");
  slot_emb_ = vec(h + "slot_embedding");  // [slots][hidden]
  rank_emb_ = vec(h + "rank_embedding");  // [top-k][hidden]
  {
    // the head's attention bias over the (slot, rank) nodes, [heads][N][N]: relative_slot_bias[h][slot_i - slot_j +
    // block - 1], plus same_slot_bias[h] within a slot
    const std::vector<float> rel = host(h + "relative_slot_bias"), same = host(h + "same_slot_bias");
    const int S = kB - 1, K = cfg_.topk, N = S * K, R = 2 * kB - 1;
    if (int(rel.size()) != cfg_.heads * R || int(same.size()) != cfg_.heads)
      fatal("%s: LiLiCorr slot biases of an unexpected size", path_.c_str());
    std::vector<float> bias(size_t(cfg_.heads) * N * N);
    for (int hh = 0; hh < cfg_.heads; ++hh)
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
          const int si = i / K, sj = j / K;
          bias[(size_t(hh) * N + i) * N + j] = rel[size_t(hh) * R + (si - sj + kB - 1)] + (si == sj ? same[size_t(hh)] : 0.0f);
        }
    attn_bias_ = upload(bias, "lili_attn_bias");
  }
  hl_.resize(size_t(cfg_.lnl));
  for (int i = 0; i < cfg_.lnl; ++i) {
    HLayer& L = hl_[size_t(i)];
    const std::string b = h + "layers." + std::to_string(i) + ".";
    L.attn_norm = vec(b + "attn_norm.weight");
    L.mlp_norm = vec(b + "mlp_norm.weight");
    L.in_proj = mat(b + "attn.in_proj_weight");
    L.in_b = vec(b + "attn.in_proj_bias");
    L.out_proj = mat(b + "attn.out_proj.weight");
    L.out_b = vec(b + "attn.out_proj.bias");
    L.mlp0 = mat(b + "mlp.up_proj.weight");
    L.mlp0_b = vec(b + "mlp.up_proj.bias");
    L.mlp2 = mat(b + "mlp.down_proj.weight");
    L.mlp2_b = vec(b + "mlp.down_proj.bias");
  }
  lout_norm_ = vec(h + "output_norm.weight");
  anchor_norm_ = vec(h + "anchor_norm.weight");
  factor_in_ = mat(h + "factor_input_proj.weight");
  factor_b_ = vec(h + "factor_input_proj.bias");
  out_head_ = mat(h + "out_head.weight");
  out_b_ = vec(h + "out_head.bias");
  in_head_ = mat(h + "in_head.weight");
  in_b_ = vec(h + "in_head.bias");
  anchor_head_ = mat(h + "anchor_out_head.weight");
  anchor_b_ = vec(h + "anchor_out_head.bias");
}

void LiliCorr::alloc_buffers() {
  Gpu& g = Gpu::get();
  const size_t B = kB, H = size_t(cfg_.H), S = B - 1, K = size_t(cfg_.topk), N = S * K, Hh = size_t(cfg_.lh);
  const size_t nl = cfg_.target_layers.size(), W = size_t(cfg_.window), V = size_t(tg_.vocab());
  auto al = [&](size_t bytes, const char* label) {
    Buffer b = g.alloc(std::max<size_t>(bytes, 16), label);
    owned_.push_back(b);
    return b;
  };
  pfeat_ = al(W * nl * H * 4, "lili_pfeat");
  pfeat_h_ = al(W * nl * H * 2, "lili_pfeat_h");
  vfeat_ = al(B * nl * H * 4, "lili_vfeat");
  btok_ = al(B * 4, "lili_btok");
  x_ = al(B * H * 4, "lili_x");
  n_ = al(B * H * 4, "lili_n");
  dyn_ = al(B * H / size_t(cfg_.conv_gs) * 4 * 4, "lili_dyn");
  t1_ = al(B * H * 4, "lili_t1");
  t2_ = al(B * H * 4, "lili_t2");
  q_ = al(B * size_t(cfg_.hq) * 128 * 4, "lili_q");
  k_ = al(B * size_t(cfg_.hkv) * 128 * 4, "lili_k");
  v_ = al(B * size_t(cfg_.hkv) * 128 * 4, "lili_v");
  ao_ = al(B * size_t(cfg_.hq) * 128 * 4, "lili_ao");
  gt_ = al(B * size_t(cfg_.I) * 4, "lili_gate");
  upb_ = al(B * size_t(cfg_.I) * 4, "lili_up");
  hfin_ = al(B * H * 4, "lili_hfin");
  logits_ = al(B * V * 4, "lili_logits");
  const size_t nch = (V + 2047) / 2048;
  cv_ = al(S * nch * K * 4, "lili_cv");
  ci_ = al(S * nch * K * 4, "lili_ci");
  ms_ = al(S * nch * 8, "lili_ms");
  cand_ = al(N * 4, "lili_cand");
  clp_ = al(N * 4, "lili_clp");
  ce_ = al(N * H * 4, "lili_ce");
  ts_ = al(N * Hh * 4, "lili_ts");
  ps_ = al(B * Hh * 4, "lili_ps");
  h1_ = al(N * Hh * 4, "lili_h1");
  fs_ = al(N * Hh * 4, "lili_fs");
  hx_ = al(N * Hh * 4, "lili_hx");
  hxn_ = al(N * Hh * 4, "lili_hxn");
  qkv_ = al(N * 3 * Hh * 4, "lili_qkv");
  hao_ = al(N * Hh * 4, "lili_hao");
  ho_ = al(N * Hh * 4, "lili_ho");
  hm0_ = al(N * size_t(cfg_.mlp) * 4, "lili_hm0");
  hm2_ = al(N * Hh * 4, "lili_hm2");
  as_ = al(Hh * 4, "lili_as");
  an_ = al(Hh * 4, "lili_an");
  fi_ = al(N * 3 * Hh * 4, "lili_fi");
  fct_ = al(N * Hh * 4, "lili_fct");
  outv_ = al(N * Hh * 4, "lili_out");
  inv_ = al(N * Hh * 4, "lili_in");
  aout_ = al(Hh * 4, "lili_aout");
  scores_ = al(S * K * 4, "lili_scores");
  tlq_ = al(S * K * K * 4, "lili_kf_lattice");
  picks_ = al(S * 4, "lili_picks");
  if (const char* v = getenv("NI_LILI_DYN")) dyn_ratio_ = float(atof(v));
  if (const char* v = getenv("NI_LILI_DYN_P")) dyn_mode_ = uint32_t(atoi(v));
  cg_ = al(W * H * 4, "lili_cg");
  cgh_ = al(W * H * 2, "lili_cgh");
  ck_ = al(W * size_t(cfg_.hkv) * 128 * 4, "lili_ck");
  cv2_ = al(W * size_t(cfg_.hkv) * 128 * 4, "lili_cv2");
  rscale_ = al(W * 4, "lili_rscale");
  xh_ = al(std::max<size_t>(64 * 12288, size_t(kB) * nl * H) * 2, "lili_xh");
  int* bt = btok_.as<int>();
  for (size_t i = 0; i < B; ++i) bt[i] = cfg_.mask_id;
  // capture hooks: the target layers' inputs, in fc's column order
  for (NhCapture* h : {&hook_pf_, &hook_v_}) {
    h->slot.assign(size_t(tg_.config().layers), -1);
    for (size_t j = 0; j < nl; ++j) h->slot[size_t(cfg_.target_layers[j])] = int(j);
    h->ld = uint32_t(nl * H);
  }
  hook_pf_.feat = pfeat_;
  hook_v_.feat = vfeat_;
}

std::unique_ptr<LiliSeq> LiliCorr::new_seq() {
  Gpu& g = Gpu::get();
  auto d = std::make_unique<LiliSeq>();
  const size_t ring = size_t(cfg_.NL) * cfg_.hkv * cfg_.window * 128 * 2;
  d->ring_k = g.alloc(ring, "lili_ring_k");
  d->ring_v = g.alloc(ring, "lili_ring_v");
  d->anchor = g.alloc(size_t(kAnchors) * cfg_.H * 4, "lili_anchor");
  memset(d->anchor.ptr(), 0, d->anchor.size);
  return d;
}

void LiliCorr::encode_copy(Encoder& e, const LiliSeq& src, LiliSeq& dst) {
  for (int i = 0; i < 3; ++i) {
    const Buffer& a = i == 0 ? src.ring_k : i == 1 ? src.ring_v : src.anchor;
    const Buffer& b = i == 0 ? dst.ring_k : i == 1 ? dst.ring_v : dst.anchor;
    const uint32_t n = uint32_t(std::min(a.size, b.size) / 16);
    e.use(pipe("copy_u32"));
    e.buf(0, a);
    e.buf(1, b);
    e.val(2, n);
    e.dispatch((n + 255) / 256, 1, 1, 256);
  }
  dst.ctx = src.ctx;
  dst.hi = src.hi;
}

// ---- context ----
void LiliCorr::begin_prefill(int n, int start) {
  pf_n_ = n;
  pf_lo_ = std::max(start, n - cfg_.window);
  hook_pf_.pos_base = pf_lo_;
  hook_pf_.rows = n - pf_lo_;
  tg_.set_capture(&hook_pf_);
}

void LiliCorr::end_prefill() { tg_.set_capture(nullptr); }

void LiliCorr::arm_verify(int P) {
  hook_v_.pos_base = P;
  hook_v_.rows = kB;
  tg_.set_capture(&hook_v_);
}

void LiliCorr::disarm() { tg_.set_capture(nullptr); }

void LiliCorr::enc_bias(Encoder& e, const Buffer& x, const Buffer& b, int rows, int D, int act) {
  const BiasP p{uint32_t(D), uint32_t(rows * D), uint32_t(act)};
  e.use(pipe("nh_bias_act"));
  e.buf(0, x);
  e.buf(1, b);
  e.val(2, p);
  e.dispatch((p.n + 255) / 256, 1, 1, 256);
}

void LiliCorr::lin(Encoder& e, const NhMat& W, const Buffer& x, const Buffer& y, int M, int epi, bool scaled) {
  if (M == 1 || W.fmt == NhFmt::F32) {
    tg_.enc_mv_rows(e, W, x, y, M, epi);
    return;
  }
  const Buffer& xh = M * size_t(W.K) * 2 <= xh_.size ? xh_ : pfeat_h_;
  if (scaled) {
    const uint32_t pk[1] = {W.K};
    e.use(pipe("nh_rows_half_scaled"));
    e.buf(0, x);
    e.buf(1, xh);
    e.buf(2, rscale_);
    e.bytes(3, pk, sizeof pk);
    e.dispatch(uint32_t(M), 1, 1, 1024);
    tg_.enc_gemm(e, W, xh, y, M, epi);
    const uint32_t pn[1] = {W.N};
    e.use(pipe("nh_rows_scale"));
    e.buf(0, y);
    e.buf(1, rscale_);
    e.bytes(2, pn, sizeof pn);
    e.dispatch(uint32_t(M), (W.N + 255) / 256, 1, 256);
  } else {
    tg_.enc_rows_half(e, x, xh, M, int(W.K));
    tg_.enc_gemm(e, W, xh, y, M, epi);
  }
}

void LiliCorr::enc_commit_rows(Encoder& e, LiliSeq& d, const Buffer& feat, int R, int pos0) {
  if (R <= 0) return;
  const uint32_t H = uint32_t(cfg_.H), F = uint32_t(cfg_.target_layers.size()) * H;
  // g = RMSNorm(fc(features)); the raw residual-stream features go to half with a per-row scale
  lin(e, fc_, feat, cg_, R, 0, true);
  tg_.enc_norm(e, cg_, enc_norm_, cg_, R, int(H), cfg_.eps);
  const AnchorP ap{uint32_t(pos0), uint32_t(R), H, uint32_t(kAnchors)};
  e.use(pipe("nh_anchor_write"));
  e.buf(0, cg_);
  e.buf(1, d.anchor);
  e.val(2, ap);
  e.dispatch(uint32_t(R), (H + 255) / 256, 1, 256);
  if (R > 1) tg_.enc_rows_half(e, cg_, cgh_, R, int(H));
  const size_t ring_layer = size_t(cfg_.hkv) * cfg_.window * 128 * 2;
  for (int l = 0; l < cfg_.NL; ++l) {
    const DLayer& L = dl_[size_t(l)];
    if (R == 1) {
      tg_.enc_mv(e, L.k, cg_, ck_, R, 0);
      tg_.enc_mv(e, L.v, cg_, cv2_, R, 0);
    } else {
      tg_.enc_gemm(e, L.k, cgh_, ck_, R, 0);
      tg_.enc_gemm(e, L.v, cgh_, cv2_, R, 0);
    }
    const QkP qp{uint32_t(cfg_.hkv), 128, uint32_t(pos0), 1, cfg_.eps, cfg_.theta};
    e.use(pipe("nh_qk_norm_rope"));
    e.buf(0, ck_);
    e.buf(1, L.kn);
    e.val(2, qp);
    e.dispatch(uint32_t(R), uint32_t(cfg_.hkv), 1, 128);
    const RingKvP rp{uint32_t(pos0), uint32_t(cfg_.window), uint32_t(cfg_.hkv), 128};
    e.use(pipe("nh_lili_ring_write"));
    e.buf(0, ck_);
    e.buf(1, cv2_);
    e.buf(2, d.ring_k, size_t(l) * ring_layer);
    e.buf(3, d.ring_v, size_t(l) * ring_layer);
    e.val(4, rp);
    e.dispatch(uint32_t(R), uint32_t(2 * cfg_.hkv), 1, 128);
  }
}

void LiliCorr::debug_feature_stats() const {
  const size_t F = cfg_.target_layers.size() * size_t(cfg_.H);
  const int rows = pf_n_ - pf_lo_;
  for (size_t j = 0; j < cfg_.target_layers.size(); ++j) {
    float mx = 0;
    double ss = 0;
    for (int r = 0; r < rows; ++r)
      for (int d = 0; d < cfg_.H; ++d) {
        const float v = pfeat_.as<float>()[size_t(r) * F + j * cfg_.H + d];
        mx = std::max(mx, std::fabs(v));
        ss += double(v) * v;
      }
    logi("LiLiCorr features: layer %d input: max |x| %.1f, rms %.3f over %d rows", cfg_.target_layers[j], mx,
         std::sqrt(ss / std::max(1.0, double(rows) * cfg_.H)), rows);
  }
}

void LiliCorr::encode_prefill_commit(Encoder& e, LiliSeq& d) {
  // rows [pf_lo_, pf_n_) captured; in chunks the commit buffers hold (window rows)
  enc_commit_rows(e, d, pfeat_, pf_n_ - pf_lo_, pf_lo_);
  d.ctx = pf_n_;
  d.hi = std::max(d.hi, d.ctx);
}

void LiliCorr::encode_commit(Encoder& e, LiliSeq& d, int m) {
  enc_commit_rows(e, d, vfeat_, m, d.ctx);
  d.ctx += m;
  d.hi = std::max(d.hi, d.ctx);
}

// ---- drafting pass ----
void LiliCorr::encode_pass(Encoder& e, Seq& s, LiliSeq& d, int P) {
  if (d.ctx != P) fatal("LiliCorr: drafting at %d with context %d", P, d.ctx);
  const uint32_t H = uint32_t(cfg_.H), B = kB, S = B - 1, K = uint32_t(cfg_.topk), N = S * K, Hh = uint32_t(cfg_.lh);
  const int R = int(B);
  {
    const BlockTokP bp{uint32_t(P), uint32_t(cfg_.mask_id), B};
    e.use(pipe("nh_block_tok"));
    e.buf(0, s.tok);
    e.buf(1, btok_);
    e.val(2, bp);
    e.dispatch(1, 1, 1, B);
  }
  tg_.enc_embed(e, btok_, 0, R, x_);
  const size_t ring_layer = size_t(cfg_.hkv) * cfg_.window * 128 * 2;
  const uint32_t G = H / uint32_t(cfg_.conv_gs);
  auto dconv = [&](const Buffer& in, const Buffer& base, uint32_t side, const Buffer& out) {
    const DconvP p{H, uint32_t(cfg_.conv_gs), side, B};
    e.use(pipe("nh_dconv"));
    e.buf(0, in);
    e.buf(1, dyn_);
    e.buf(2, base);
    e.buf(3, out);
    e.val(4, p);
    e.dispatch(B, (H + 255) / 256, 1, 256);
  };
  auto add = [&](const Buffer& a, const Buffer& b, const Buffer& y, uint32_t n) {
    e.use(pipe("nh_add"));
    e.buf(0, a);
    e.buf(1, b);
    e.buf(2, y);
    e.val(3, n);
    e.dispatch((n + 255) / 256, 1, 1, 256);
  };
  (void)G;
  for (int l = 0; l < cfg_.NL; ++l) {
    const DLayer& L = dl_[size_t(l)];
    tg_.enc_norm(e, x_, L.attn_norm, n_, R, int(H), cfg_.eps);
    lin(e, L.ac_proj, n_, dyn_, R, 0);
    dconv(n_, L.ac_base, 0, t1_);
    tg_.enc_rows_half(e, t1_, xh_, R, int(H));  // shared by q, k, v
    tg_.enc_gemm(e, L.q, xh_, q_, R, 0);
    tg_.enc_gemm(e, L.k, xh_, k_, R, 0);
    tg_.enc_gemm(e, L.v, xh_, v_, R, 0);
    {
      const QkP qp{uint32_t(cfg_.hq), 128, uint32_t(P), 1, cfg_.eps, cfg_.theta};
      e.use(pipe("nh_qk_norm_rope"));
      e.buf(0, q_);
      e.buf(1, L.qn);
      e.val(2, qp);
      e.dispatch(B, uint32_t(cfg_.hq), 1, 128);
      const QkP kp{uint32_t(cfg_.hkv), 128, uint32_t(P), 1, cfg_.eps, cfg_.theta};
      e.use(pipe("nh_qk_norm_rope"));
      e.buf(0, k_);
      e.buf(1, L.kn);
      e.val(2, kp);
      e.dispatch(B, uint32_t(cfg_.hkv), 1, 128);
    }
    {
      const LiliAttnP ap{uint32_t(P), uint32_t(std::max({0, P - cfg_.window, d.hi - cfg_.window})), uint32_t(cfg_.window),
                         uint32_t(cfg_.hq),
                         uint32_t(cfg_.hkv), B, 1.0f / std::sqrt(128.0f)};
      e.use(pipe("nh_lili_attn"));
      e.buf(0, q_);
      e.buf(1, k_);
      e.buf(2, v_);
      e.buf(3, d.ring_k, size_t(l) * ring_layer);
      e.buf(4, d.ring_v, size_t(l) * ring_layer);
      e.buf(5, L.sinks);
      e.buf(6, ao_);
      e.val(7, ap);
      e.dispatch(uint32_t(cfg_.hkv), B, 1, uint32_t(32 * (cfg_.hq / cfg_.hkv)));
    }
    lin(e, L.o, ao_, t2_, R, 0);
    dconv(t2_, L.ac_base, 1, t1_);
    add(x_, t1_, x_, B * H);
    tg_.enc_norm(e, x_, L.ffn_norm, n_, R, int(H), cfg_.eps);
    lin(e, L.fc_proj, n_, dyn_, R, 0);
    dconv(n_, L.fc_base, 0, t1_);
    tg_.enc_rows_half(e, t1_, xh_, R, int(H));  // shared by gate, up
    tg_.enc_gemm(e, L.gate, xh_, gt_, R, 0);
    tg_.enc_gemm(e, L.up, xh_, upb_, R, 0);
    {
      const uint32_t n = B * uint32_t(cfg_.I);
      e.use(pipe("nh_swiglu"));
      e.buf(0, gt_);
      e.buf(1, upb_);
      e.buf(2, gt_);
      e.val(3, n);
      e.dispatch((n + 255) / 256, 1, 1, 256);
    }
    lin(e, L.down, gt_, t2_, R, 0, true);  // (Super-VL's drafter: silu(gate) * up reaches 2.6e5, past half)
    dconv(t2_, L.fc_base, 1, t1_);
    add(x_, t1_, x_, B * H);
  }
  tg_.enc_norm(e, x_, out_norm_, hfin_, R, int(H), cfg_.eps);
  // slot logits (rows 1..7) with the target's LM head, their top-K candidates
  // the draft head: the drafter's own or the target's LM head, rows [0, head_rows_) (ids in the tokenizer's order:
  // special tokens first, then the BPE vocabulary in merge order, most frequent first), a view without a copy
  NhMat head = own_head_ ? own_head_ : tg_.lm_head();
  head.N = uint32_t(head_rows_);
  const uint32_t V = head.N;
  lin(e, head, hfin_.view(size_t(H) * 4, size_t(S) * H * 4), logits_, int(S), 0);
  const uint32_t nch = (V + 2047) / 2048;
  const CandP cp{V, K, nch, 0, V};
  e.use(pipe("nh_cand1"));
  e.buf(0, logits_);
  e.buf(1, cv_);
  e.buf(2, ci_);
  e.buf(3, ms_);
  e.val(4, cp);
  e.dispatch(nch, S, 1, 256);
  e.use(pipe("nh_cand2"));
  e.buf(0, cv_);
  e.buf(1, ci_);
  e.buf(2, ms_);
  e.buf(3, cand_);
  e.buf(4, clp_);
  e.val(5, cp);
  e.dispatch(S, 1, 1, 32);
  // LiLiCorr head: node states
  tg_.enc_embed(e, cand_, 0, int(N), ce_);
  lin(e, tok_proj_, ce_, ts_, int(N), 0);
  lin(e, pass_proj_, hfin_.view(size_t(H) * 4, size_t(S) * H * 4), ps_, int(S), 0);
  const FeatP fp{K, Hh, 1e-5f};
  e.use(pipe("nh_lili_feat1"));
  e.buf(0, clp_);
  e.buf(1, feat_ln_w_);
  e.buf(2, feat_ln_b_);
  e.buf(3, mlp1_w_);
  e.buf(4, mlp1_b_);
  e.buf(5, h1_);
  e.val(6, fp);
  e.dispatch(N, 1, 1, 256);
  lin(e, mlp3_, h1_, fs_, int(N), 0);
  e.use(pipe("nh_lili_x0"));
  e.buf(0, ts_);
  e.buf(1, tok_b_);
  e.buf(2, ps_);
  e.buf(3, pass_b_);
  e.buf(4, fs_);
  e.buf(5, mlp3_b_);
  e.buf(6, slot_emb_);
  e.buf(7, rank_emb_);
  e.buf(8, hx_);
  e.val(9, fp);
  e.dispatch(N, 1, 1, 256);
  // anchor: the projected context feature of position P - 1
  const Buffer anc = d.anchor.view(size_t((P - 1) % kAnchors) * H * 4, size_t(H) * 4);
  tg_.enc_mv(e, ctx_proj_, anc, as_, 1, 0);
  enc_bias(e, as_, ctx_b_, 1, int(Hh), 0);
  for (int i = 0; i < cfg_.lnl; ++i) {
    const HLayer& L = hl_[size_t(i)];
    tg_.enc_norm(e, hx_, L.attn_norm, hxn_, int(N), int(Hh), cfg_.eps);
    lin(e, L.in_proj, hxn_, qkv_, int(N), 0);
    enc_bias(e, qkv_, L.in_b, int(N), int(3 * Hh), 0);
    const MhaP mp{N, Hh, uint32_t(cfg_.heads), 1.0f / std::sqrt(float(Hh / uint32_t(cfg_.heads)))};
    e.use(pipe("nh_lili_mha"));
    e.buf(0, qkv_);
    e.buf(1, attn_bias_);
    e.buf(2, hao_);
    e.val(3, mp);
    e.dispatch(uint32_t(cfg_.heads), N, 1, 32);
    lin(e, L.out_proj, hao_, ho_, int(N), 0);
    {
      const BiasP bp{Hh, N * Hh, 0};
      e.use(pipe("nh_add_bias"));
      e.buf(0, hx_);
      e.buf(1, ho_);
      e.buf(2, L.out_b);
      e.val(3, bp);
      e.dispatch((bp.n + 255) / 256, 1, 1, 256);
    }
    tg_.enc_norm(e, hx_, L.mlp_norm, hxn_, int(N), int(Hh), cfg_.eps);
    lin(e, L.mlp0, hxn_, hm0_, int(N), 0);
    enc_bias(e, hm0_, L.mlp0_b, int(N), cfg_.mlp, 1);
    lin(e, L.mlp2, hm0_, hm2_, int(N), 0);
    {
      const BiasP bp{Hh, N * Hh, 0};
      e.use(pipe("nh_add_bias"));
      e.buf(0, hx_);
      e.buf(1, hm2_);
      e.buf(2, L.mlp2_b);
      e.val(3, bp);
      e.dispatch((bp.n + 255) / 256, 1, 1, 256);
    }
  }
  tg_.enc_norm(e, hx_, lout_norm_, hxn_, int(N), int(Hh), cfg_.eps);
  tg_.enc_norm(e, as_, anchor_norm_, an_, 1, int(Hh), cfg_.eps);
  e.use(pipe("nh_lili_factor_in"));
  e.buf(0, hxn_);
  e.buf(1, an_);
  e.buf(2, fi_);
  e.val(3, fp);
  e.dispatch(N, 1, 1, 256);
  lin(e, factor_in_, fi_, fct_, int(N), 0);
  enc_bias(e, fct_, factor_b_, int(N), int(Hh), 1);
  const L2P lp{Hh, cfg_.vec_eps};
  tg_.enc_rows_half(e, fct_, xh_, int(N), int(Hh));  // shared by the out and in heads
  tg_.enc_gemm(e, out_head_, xh_, outv_, int(N), 0);
  tg_.enc_gemm(e, in_head_, xh_, inv_, int(N), 0);
  tg_.enc_mv(e, anchor_head_, an_, aout_, 1, 0);
  for (int w = 0; w < 3; ++w) {
    const Buffer& xb = w == 0 ? outv_ : w == 1 ? inv_ : aout_;
    const Buffer& bb = w == 0 ? out_b_ : w == 1 ? in_b_ : anchor_b_;
    e.use(pipe("nh_bias_l2norm"));
    e.buf(0, xb);
    e.buf(1, bb);
    e.val(2, lp);
    e.dispatch(w == 2 ? 1 : N, 1, 1, 256);
  }
}

void LiliCorr::encode_draft(Encoder& e, Seq& s, LiliSeq& d, int P) {
  encode_pass(e, s, d, P);
  const uint32_t S = kB - 1, K = uint32_t(cfg_.topk), Hh = uint32_t(cfg_.lh);
  const WalkP wp{S, K, Hh, uint32_t(P), cfg_.logit_scale};
  e.use(pipe("nh_lili_walk"));
  e.buf(0, inv_);
  e.buf(1, outv_);
  e.buf(2, aout_);
  e.buf(3, cand_);
  e.buf(4, s.tok);
  e.buf(5, scores_);
  e.val(6, wp);
  e.buf(7, picks_);
  e.dispatch(1, 1, 1, 32 * K);
  // the coming verify's depth (NI_LILI_DYN: the per-row cost ratio; 0 verifies every draft)
  const DepthP dp{S, K, dyn_ratio_, dyn_mode_};
  e.use(pipe("nh_lili_depth"));
  e.buf(0, clp_);
  e.buf(1, picks_);
  e.buf(2, tg_.live_rows());
  e.val(3, dp);
  e.buf(4, scores_);
  e.dispatch(1, 1, 1, 1);
}

// KFlash: the LiLiCorr head's transition probabilities between every pair of candidates of consecutive slots
// (nh_lili_table), then the best-first tree under the budget (nh_kf_build)
void LiliCorr::encode_draft_tree(Encoder& e, Seq& s, LiliSeq& d, int P, const Buffer& kfp) {
  encode_pass(e, s, d, P);
  const uint32_t S = kB - 1, K = uint32_t(cfg_.topk), Hh = uint32_t(cfg_.lh);
  const LiliTableP tp{S, K, Hh, cfg_.logit_scale};
  e.use(pipe("nh_lili_table"));
  e.buf(0, inv_);
  e.buf(1, outv_);
  e.buf(2, aout_);
  e.buf(3, clp_);
  e.buf(4, tlq_);
  e.val(5, tp);
  e.dispatch(1 + (S - 1) * K, 1, 1, 32 * K);
  enc_kf_build(e, tg_, s, P, cand_, tlq_, int(K), kfp);
}

}  // namespace ni
