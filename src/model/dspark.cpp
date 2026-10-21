#include "model/dspark.h"
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

// host mirrors of the kernel parameter structs (nh_lili.metal, nh_dspark.metal)
struct RingKvP {
  uint32_t pos0, W, Hkv, HD;
};
struct AttnP {
  uint32_t P, lo, W, Hq, Hkv, B, NS, ld;
  float scale, eps, theta;
};
constexpr uint32_t kAttnSplit = 16;  // pieces of the context in the block attention (nh_dsp_attn_part)
struct CandP {
  uint32_t V, K, nchunks, row0, ld;
};
struct WalkP {
  uint32_t S, K, R, P;
  float gs;
};
struct DepthP {
  uint32_t S, K;
  float ratio;
  uint32_t mode;
};
struct DspTableP {
  uint32_t S, K, R, P;
  float gs;
};
struct CycP {
  uint32_t P, hi, W, D, Hq, Hkv, B, S, K, R, NS, ld;
  float eps, theta, scale, gs;
};
// byte offsets of nh_dsp_cycle's blocks in cyc_
constexpr size_t kCycEmbed = 0, kCycAttn = 256, kCycWalk = 512, kCycRing = 768;

nlohmann::json read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) return {};
  try {
    return nlohmann::json::parse(f);
  } catch (...) {
    return {};
  }
}

// as loaded: NVFP4 as stored, BF16 matrices quantized to Q8 (34 bytes per 32), BF16 vectors as fp32, the Markov
// embedding as stored; of the embedding table only the mask token's row
size_t loaded_bytes(const SafetensorsIndex& idx, int hidden) {
  size_t w = 0;
  for (const auto& [name, t] : idx.all()) {
    if (name.find("embed_tokens.") != std::string::npos) w += size_t(hidden) * 4;
    else if (name.find("markov_w1") != std::string::npos) w += t.len;
    else if (t.dtype == DType::BF16 && t.shape.size() == 2 && t.shape[1] % 32 == 0) w += size_t(t.numel()) * 34 / 32;
    else if (t.dtype == DType::BF16) w += size_t(t.numel()) * 4;
    else w += t.len;
  }
  return w;
}

}  // namespace

bool DSpark::is_dspark(const std::string& dir, int hidden) {
  const nlohmann::json c = read_json(dir + "/config.json");
  if (!c.is_object() || !c.contains("dflash_config") || !c["dflash_config"].is_object()) return false;
  const bool markov = c.value("dspark_fixup_head_type", std::string()) == "markov" || c.value("markov_rank", 0) > 0;
  struct stat st;
  return markov && c.value("hidden_size", 0) == hidden && stat((dir + "/model.safetensors").c_str(), &st) == 0;
}

size_t DSpark::plan_bytes(const std::string& dir, size_t* per_slot) {
  const nlohmann::json c = read_json(dir + "/config.json");
  const SafetensorsIndex idx(dir);
  const nlohmann::json& d = c.contains("dflash_config") ? c["dflash_config"] : c;
  const int64_t H = c.value("hidden_size", 2688), NL = c.value("num_hidden_layers", 6);
  const int64_t W = d.value("swa_window_size", c.value("sliding_window", 1024)), hkv = c.value("num_key_value_heads", 2);
  const int64_t hd = c.value("head_dim", 128), V = c.value("vocab_size", 131072), I = c.value("intermediate_size", 6144);
  const size_t w = loaded_bytes(idx, int(H));
  const size_t ring = size_t(NL) * 2 * size_t(hkv) * size_t(W) * size_t(hd) * 2 + 256;
  if (per_slot) *per_slot = ring;
  const int64_t nl = d.contains("target_layer_ids") ? int64_t(d["target_layer_ids"].size()) : 6;
  // prompt capture (window rows of every tap) and its half copy, commit buffers (window rows), the pass (8 rows:
  // activations, the slots' logits, candidates)
  const int64_t hq = c.value("num_attention_heads", 32);
  // (+ the block attention's pieces)
  const size_t bufs = size_t(W) * size_t(nl) * size_t(H) * 6 + size_t(W) * (size_t(H) * 6 + size_t(hkv) * size_t(hd) * 8 + 4) +
                      size_t(kB) * size_t(hq) * kAttnSplit * (size_t(hd) + 2) * 4 +
                      size_t(kB) * (size_t(V) * 4 + size_t(I) * 8 + size_t(H) * 24 + size_t(nl) * size_t(H) * 4) +
                      std::max<size_t>(64 * 12288, size_t(kB) * size_t(nl) * size_t(H)) * 2 + (size_t(4) << 20);
  // the M5's nh_xq8_v8 path keeps a copy of each layer's k|v codes in the Q8 layout (counted on every Mac)
  const size_t kv_std = size_t(NL) * 2 * size_t(hkv) * size_t(hd) * size_t(H);
  return w + bufs + kv_std;
}

// ---- loading ----
// NVFP4 (ModelOpt) as the target's kernels run it; BF16 matrices quantized to Q8 rows (drafts only propose)
NhMat DSpark::mat(const std::string& name) {
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

NhMat DSpark::mat_cat(const std::vector<std::string>& names, std::vector<float>& gs, bool shared) {
  std::vector<NhMat> parts;
  for (const std::string& n : names) parts.push_back(mat(n));
  NhMat m = parts[0];
  m.N = 0;
  gs.clear();
  for (const NhMat& p : parts) {
    if (p.fmt != m.fmt || p.K != m.K || p.q_row != m.q_row || p.s_row != m.s_row)
      fatal("%s: %s and %s do not share a format", path_.c_str(), names[0].c_str(), names[1].c_str());
    m.N += p.N;
    gs.push_back(p.gs);
  }
  // shared: the parts' common tensor scale on the whole (q|k|v: its products and views apply m.gs); else 1, the
  // caller applying each part's own (gate|up: gs_g / gs_u in the SwiGLU)
  m.gs = shared ? gs[0] : 1.0f;
  Gpu& g = Gpu::get();
  m.q = g.alloc(m.q_row * m.N, names[0].c_str(), false);
  m.s = g.alloc(m.s_row * m.N, "scales", false);
  size_t oq = 0, os = 0;
  for (NhMat& p : parts) {
    memcpy(m.q.as<uint8_t>() + oq, p.q.ptr(), p.q_row * p.N);
    memcpy(m.s.as<uint8_t>() + os, p.s.ptr(), p.s_row * p.N);
    oq += p.q_row * p.N;
    os += p.s_row * p.N;
    for (Buffer* b : {&p.q, &p.s}) {
      owned_.erase(std::find_if(owned_.begin(), owned_.end(), [&](const Buffer& o) { return o.mtl == b->mtl; }));
      weight_bytes_ -= b->size;
      g.release(*b);
    }
  }
  owned_.push_back(m.q);
  owned_.push_back(m.s);
  weight_bytes_ += m.q.size + m.s.size;
  return m;
}

void DSpark::repack_xq8(uint8_t* q, size_t rows, uint32_t K) {
  for (size_t g = 0; g < rows * (K / 16); ++g) {
    uint8_t* b = q + g * 16;
    uint8_t t[16];
    for (int m = 0; m < 4; ++m) {
      t[4 * m + 0] = b[2 * m];
      t[4 * m + 1] = b[2 * m + 1];
      t[4 * m + 2] = b[2 * m + 8];
      t[4 * m + 3] = b[2 * m + 9];
    }
    memcpy(b, t, 16);
  }
}

NhMat DSpark::rows(const NhMat& m, uint32_t r0, uint32_t n) {
  NhMat v = m;
  v.N = n;
  v.q = m.q.view(size_t(r0) * m.q_row, size_t(n) * m.q_row);
  v.s = m.s.view(size_t(r0) * m.s_row, size_t(n) * m.s_row);
  return v;
}

std::vector<float> DSpark::host(const std::string& name) {
  const SafetensorsIndex::Entry& t = st_->get(name);
  if (t.dtype != DType::BF16 && t.dtype != DType::F32) fatal("%s: %s is neither BF16 nor F32", path_.c_str(), name.c_str());
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

Buffer DSpark::upload(const void* p, size_t bytes, const char* label) {
  Buffer b = Gpu::get().alloc(std::max<size_t>(bytes, 16), label, false);
  memcpy(b.ptr(), p, bytes);
  owned_.push_back(b);
  weight_bytes_ += b.size;
  return b;
}

Buffer DSpark::vec(const std::string& name) {
  const std::vector<float> v = host(name);
  return upload(v.data(), v.size() * 4, name.c_str());
}

DSpark::DSpark(NemotronH& target, const std::string& dir) : tg_(target), path_(dir) {
  while (path_.size() > 1 && path_.back() == '/') path_.pop_back();
  const NhConfig& tc = tg_.config();
  if (!is_dspark(path_, tc.hidden)) fatal("%s is not a DSpark drafter for this model", dir.c_str());
  const nlohmann::json j = read_json(path_ + "/config.json");
  const nlohmann::json& d = j["dflash_config"];
  H_ = j.value("hidden_size", 0);
  NL_ = j.value("num_hidden_layers", 0);
  I_ = j.value("intermediate_size", 0);
  hq_ = j.value("num_attention_heads", 0);
  hkv_ = j.value("num_key_value_heads", 0);
  window_ = d.value("swa_window_size", j.value("sliding_window", 1024));
  eps_ = j.value("rms_norm_eps", 1e-6f);
  theta_ = j.value("rope_theta", 10000.0f);
  mask_id_ = d.value("mask_token_id", j.value("mask_token_id", -1));
  rank_ = j.value("dspark_markov_rank", j.value("markov_rank", 0));
  // the outputs of target_layer_ids are the inputs of the next layers; the last layer's output (an id of layers - 1)
  // is captured after the trunk (NemotronH: capture slot `layers`)
  if (d.contains("target_layer_ids"))
    for (auto& v : d["target_layer_ids"]) taps_.push_back(v.get<int>() + 1);
  // block_size counts the anchor when dspark_bonus_anchor is set (8 rows: the anchor + 7 proposals)
  const int block = j.value("block_size", 8) + (j.value("dspark_bonus_anchor", false) ? 0 : 1);
  const bool causal = d.value("causal", j.value("dflash_query_causal", false));
  const std::string rope = j.contains("rope_parameters") && !j["rope_parameters"].is_null() ? "scaled" : "";
  if (H_ != tc.hidden || NL_ <= 0 || block != kB || j.value("head_dim", 128) != 128 || hq_ <= 0 || hkv_ <= 0 ||
      hq_ % hkv_ || mask_id_ < 0 || mask_id_ >= tg_.vocab() || taps_.empty() || !causal || rank_ <= 0 || rank_ % 16 ||
      rank_ > 1024 || tg_.vocab() > 64 * 2048 /* the candidates kernel's 64 chunks */ || d.value("sample_from_anchor", j.value("sample_from_anchor", true)) || j.value("has_lm_head", false) ||
      !rope.empty() || j.value("vocab_size", 0) != tg_.vocab() || window_ <= 0)
    fatal("%s: unsupported DSpark drafter (hidden %d, block %d, Markov rank %d)", dir.c_str(), H_, block, rank_);
  for (int l : taps_)
    if (l < 1 || l > tc.layers) fatal("%s: target layer %d out of range", dir.c_str(), l - 1);
  K_ = 16;
  if (const char* v = getenv("NI_DSPARK_K")) K_ = std::clamp(atoi(v), 1, kMaxCand);
  head_rows_ = tg_.vocab();
  if (const char* v = getenv("NI_DSPARK_HEAD"); v && atoi(v) > 0) head_rows_ = std::min(tg_.vocab(), atoi(v));
  if (Gpu::get().apple10()) dyn_ratio_ = 0.3f;
  if (const char* v = getenv("NI_DSPARK_DYN")) dyn_ratio_ = float(atof(v));
  if (const char* v = getenv("NI_DSPARK_CHAIN")) chain_ = atoi(v) != 0;
  xq8_ = Gpu::get().apple10();
  if (const char* v = getenv("NI_DSPARK_XQ8")) xq8_ = atoi(v) != 0;
  if (H_ % 128) xq8_ = false;  // (nh_xq8_v8: K % 128 == 0; o's K, heads x 128, always is)
  if (const char* v = getenv("NI_DSPARK_DEPTH")) fixed_depth_ = std::clamp(atoi(v), 0, kB - 1);
  const double t0 = now_ms();
  st_ = std::make_unique<SafetensorsIndex>(path_);
  load();
  st_.reset();
  alloc_buffers();
  logi("DSpark drafter: %d layers, window %d, %d KV heads, Markov rank %d over %d candidates, %.2f GB (NVFP4 as "
       "stored, BF16 matrices as Q8, the target's embedding and LM head), loaded in %.1f s",
       NL_, window_, hkv_, rank_, K_, weight_bytes_ / 1e9, (now_ms() - t0) / 1000);
}

DSpark::~DSpark() {
  Gpu& g = Gpu::get();
  for (auto& b : owned_) g.release(b);
}

void DSpark::load() {
  fc_ = mat("fc.weight");
  if (int(fc_.K) != int(taps_.size()) * H_ || int(fc_.N) != H_) fatal("%s: fc of an unexpected shape", path_.c_str());
  enc_norm_ = vec("hidden_norm.weight");
  out_norm_ = vec("norm.weight");
  dl_.resize(size_t(NL_));
  for (int i = 0; i < NL_; ++i) {
    DLayer& L = dl_[size_t(i)];
    const std::string b = "layers." + std::to_string(i) + ".";
    L.attn_norm = vec(b + "input_layernorm.weight");
    L.ffn_norm = vec(b + "post_attention_layernorm.weight");
    L.qn = vec(b + "self_attn.q_norm.weight");
    L.kn = vec(b + "self_attn.k_norm.weight");
    if (st_->find(b + "self_attn.attention_sink_bias")) {
      L.sinks = vec(b + "self_attn.attention_sink_bias");
    } else {
      const std::vector<float> none(size_t(hq_), -1e30f);  // no sinks: a logit that adds nothing
      L.sinks = upload(none.data(), none.size() * 4, "dspark_no_sinks");
    }
    std::vector<float> gs;
    L.qkv = mat_cat({b + "self_attn.q_proj.weight", b + "self_attn.k_proj.weight", b + "self_attn.v_proj.weight"}, gs,
                    true);
    const uint32_t nq = uint32_t(hq_) * 128, nk = uint32_t(hkv_) * 128;
    if (L.qkv.N != nq + 2 * nk) fatal("%s: attention projections of an unexpected shape", path_.c_str());
    if (gs[0] != gs[1] || gs[1] != gs[2])  // (q|k|v run as one product with one tensor scale)
      fatal("%s: q, k and v projections with different NVFP4 tensor scales are not supported", path_.c_str());
    L.kv = rows(L.qkv, nq, 2 * nk);
    L.o = mat(b + "self_attn.o_proj.weight");
    // (xq8_: the Q8 matrices only, BF16 checkpoints' attention; gemm_q8 sends NVFP4 ones to enc_gemm)
    L.kv_std = L.kv;
    if (xq8_ && L.qkv.fmt == NhFmt::Q8) {
      // k|v's rows in the Q8 layout as well (one-row and prompt commits), then q|k|v's codes repacked
      L.kv_std.q = upload(L.kv.q.ptr(), L.kv.q.size, "dspark_kv_std");
      repack_xq8(L.qkv.q.as<uint8_t>(), L.qkv.N, L.qkv.K);
    }
    if (xq8_ && L.o.fmt == NhFmt::Q8) repack_xq8(L.o.q.as<uint8_t>(), L.o.N, L.o.K);
    L.gu = mat_cat({b + "mlp.gate_proj.weight", b + "mlp.up_proj.weight"}, gs);
    L.gs_g = gs[0];
    L.gs_u = gs[1];
    if (int(L.gu.N) != 2 * I_) fatal("%s: gate / up of an unexpected shape", path_.c_str());
    L.down = mat(b + "mlp.down_proj.weight");
  }
  // the embedding: the drafter's table equals the target's except the mask token's (trained) row
  {
    const SafetensorsIndex::Entry* t = st_->find("embed_tokens.weight");
    std::vector<float> row(static_cast<size_t>(H_));
    if (t) {
      if (t->dtype != DType::BF16 || t->shape.size() != 2 || int(t->shape[1]) != H_)
        fatal("%s: embed_tokens of an unexpected format", path_.c_str());
      std::vector<uint16_t> raw(static_cast<size_t>(H_));
      st_->read(*t, size_t(mask_id_) * size_t(H_) * 2, raw.data(), raw.size() * 2);
      for (int i = 0; i < H_; ++i) row[size_t(i)] = bf16_to_f32(raw[size_t(i)]);
    } else {
      fatal("%s: no embed_tokens (the mask token's row)", path_.c_str());
    }
    mask_row_ = upload(row.data(), row.size() * 4, "dspark_mask_row");
  }
  // Markov head: W1 BF16 [V][rank] as stored, W2 NVFP4 [V][rank]
  {
    const SafetensorsIndex::Entry& t = st_->get("markov_head.markov_w1.weight");
    if (t.dtype != DType::BF16 || t.shape.size() != 2 || int(t.shape[0]) != tg_.vocab() || int(t.shape[1]) != rank_)
      fatal("%s: markov_w1 of an unexpected format", path_.c_str());
    Buffer b = Gpu::get().alloc(t.len, "dspark_markov_w1", false);
    st_->read(t, 0, b.ptr(), t.len);
    owned_.push_back(b);
    weight_bytes_ += b.size;
    w1_ = b;
  }
  w2_ = mat("markov_head.markov_w2.weight");
  if (w2_.fmt != NhFmt::FP4 || int(w2_.N) != tg_.vocab() || int(w2_.K) != rank_)
    fatal("%s: markov_w2 of an unexpected format", path_.c_str());
}

void DSpark::alloc_buffers() {
  Gpu& g = Gpu::get();
  const size_t B = kB, H = size_t(H_), S = B - 1, K = size_t(kMaxCand);
  const size_t nl = taps_.size(), W = size_t(window_), V = size_t(tg_.vocab());
  auto al = [&](size_t bytes, const char* label) {
    Buffer b = g.alloc(std::max<size_t>(bytes, 16), label);
    owned_.push_back(b);
    return b;
  };
  pfeat_ = al(W * nl * H * 4, "dspark_pfeat");
  pfeat_h_ = al(W * nl * H * 2, "dspark_pfeat_h");
  vfeat_ = al(B * nl * H * 4, "dspark_vfeat");
  x_ = al(B * H * 4, "dspark_x");
  t2_ = al(B * H * 4, "dspark_t2");
  q_ = al(B * size_t(hq_ + 2 * hkv_) * 128 * 4, "dspark_qkv");
  ao_ = al(B * size_t(hq_) * kAttnSplit * (128 + 2) * 4, "dspark_attn_part");
  gt_ = al(B * size_t(I_) * 2 * 4, "dspark_gate_up");
  logits_ = al(S * V * 4, "dspark_logits");
  const size_t nch = (V + 2047) / 2048;
  cv_ = al(S * nch * K * 4, "dspark_cv");
  ci_ = al(S * nch * K * 4, "dspark_ci");
  ms_ = al(S * nch * 8, "dspark_ms");
  cand_ = al(S * K * 4, "dspark_cand");
  clp_ = al(S * K * 4, "dspark_clp");
  qlp_ = al(S * K * 4, "dspark_qlp");
  tlq_ = al(S * K * K * 4, "dspark_kf_lattice");
  picks_ = al(S * 4, "dspark_picks");
  cg_ = al(W * H * 4, "dspark_cg");
  cgh_ = al(W * H * 2, "dspark_cgh");
  ck_ = al(W * 2 * size_t(hkv_) * 128 * 4, "dspark_ckv");  // a commit's k|v rows
  rscale_ = al(W * 4, "dspark_rscale");
  xh_ = al(std::max<size_t>(64 * 12288, B * nl * H) * 2, "dspark_xh");
  cyc_ = al(2048, "dspark_cycle");
  rope_ = al(B * 64 * 8, "dspark_rope");
  walk_d_ = al(S * K * K * 4, "dspark_walk_dots");
  // capture hooks: the taps' inputs (the last one: the trunk's output, slot `layers`), in fc's column order
  for (NhCapture* h : {&hook_pf_, &hook_v_}) {
    h->slot.assign(size_t(tg_.config().layers) + 1, -1);
    for (size_t j = 0; j < nl; ++j) h->slot[size_t(taps_[j])] = int(j);
    h->ld = uint32_t(nl * H);
  }
  hook_pf_.feat = pfeat_;
  hook_v_.feat = vfeat_;
}

std::unique_ptr<LiliSeq> DSpark::new_seq() {
  Gpu& g = Gpu::get();
  auto d = std::make_unique<LiliSeq>();
  const size_t ring = size_t(NL_) * hkv_ * window_ * 128 * 2;
  d->ring_k = g.alloc(ring, "dspark_ring_k");
  d->ring_v = g.alloc(ring, "dspark_ring_v");
  d->anchor = g.alloc(16, "dspark_unused");
  return d;
}

void DSpark::encode_copy(Encoder& e, const LiliSeq& src, LiliSeq& dst) {
  for (int i = 0; i < 2; ++i) {
    const Buffer& a = i == 0 ? src.ring_k : src.ring_v;
    const Buffer& b = i == 0 ? dst.ring_k : dst.ring_v;
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
void DSpark::begin_prefill(int n, int start) {
  pf_n_ = n;
  pf_lo_ = std::max(start, n - window_);
  hook_pf_.pos_base = pf_lo_;
  hook_pf_.rows = n - pf_lo_;
  tg_.set_capture(&hook_pf_);
}

void DSpark::end_prefill() { tg_.set_capture(nullptr); }

void DSpark::arm_verify(int P) {
  hook_v_.pos_base = P;
  hook_v_.rows = kB;
  tg_.set_capture(&hook_v_);
}

void DSpark::disarm() { tg_.set_capture(nullptr); }

void DSpark::lin(Encoder& e, const NhMat& W, const Buffer& x, const Buffer& y, int M, int epi, bool scaled) {
  if (M == 1) {
    tg_.enc_mv_rows(e, W, x, y, M, epi);
    return;
  }
  const Buffer& xh = M * size_t(W.K) * 2 <= xh_.size ? xh_ : pfeat_h_;
  if (scaled) {
    // rows to half with a per-row scale (inputs that may leave half's range), the product scaled back
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

void DSpark::gemm_q8(Encoder& e, const NhMat& W, const Buffer& xh, const Buffer& y, int M, int epi) {
  if (!xq8_ || M > 8 || W.fmt != NhFmt::Q8) {
    tg_.enc_gemm(e, W, xh, y, M, epi);
    return;
  }
  const struct {
    uint32_t N, K, x_ld, y_ld, R;
    float gs;
    uint32_t epi;
  } p{W.N, W.K, W.K, W.N, uint32_t(M), W.gs, uint32_t(epi)};
  // (nh_xq8_v8_n1: 8 weight rows per threadgroup, the same outputs; NI_NH_V8_N1=0 as for the target's products)
  static const bool n1 = getenv("NI_NH_V8_N1") ? atoi(getenv("NI_NH_V8_N1")) != 0 : true;
  e.use(pipe(n1 ? "nh_xq8_v8_n1" : "nh_xq8_v8"));
  e.buf(0, xh);
  e.buf(1, W.q);
  e.buf(2, W.s);
  e.buf(3, y);
  e.val(4, p);
  e.dispatch(n1 ? (W.N + 7) / 8 : (W.N + 15) / 16, 1, 1, 256);
}

void DSpark::norm_half(Encoder& e, const Buffer& x, const Buffer& w, const Buffer& y, int R) {
  // nh_rows_half normalizing: the values of nh_rmsnorm then nh_rows_half in one pass
  const struct {
    uint32_t D, norm;
    float eps;
    uint32_t x_ld;
  } hp{uint32_t(H_), 1u, eps_, uint32_t(H_)};
  e.use(pipe("nh_rows_half"));
  e.buf(0, x);
  e.buf(1, w);
  e.buf(2, y);
  e.val(3, hp);
  e.dispatch(uint32_t(R), 1, 1, 1024);
}

void DSpark::enc_commit_rows(Encoder& e, LiliSeq& d, const Buffer& feat, int R, int pos0) {
  enc_commit(e, d, feat, R, pos0, false);
}

void DSpark::enc_commit(Encoder& e, LiliSeq& d, const Buffer& feat, int R, int pos0, bool chained) {
  if (R <= 0) return;
  // g = RMSNorm(fc(features)); the raw residual-stream features go to half with a per-row scale
  lin(e, fc_, feat, cg_, R, 0, true);
  tg_.enc_norm(e, cg_, enc_norm_, cg_, R, H_, eps_);
  if (R > 1) tg_.enc_rows_half(e, cg_, cgh_, R, H_);
  const size_t ring_layer = size_t(hkv_) * window_ * 128 * 2;
  for (int l = 0; l < NL_; ++l) {
    const DLayer& L = dl_[size_t(l)];
    // k|v in one product, then k's norm + RoPE and both into the ring in one pass
    if (R == 1) tg_.enc_mv(e, L.kv_std, cg_, ck_, R, 0);
    else if (R <= 8) gemm_q8(e, L.kv, cgh_, ck_, R, 0);
    else tg_.enc_gemm(e, L.kv_std, cgh_, ck_, R, 0);
    const struct {
      uint32_t pos0, W, Hkv, HD, ld;
      float eps, theta;
    } kp{uint32_t(pos0), uint32_t(window_), uint32_t(hkv_), 128, L.kv.N, eps_, theta_};
    e.use(pipe("nh_dsp_commit_kv"));
    e.buf(0, ck_);
    e.buf(1, L.kn);
    e.buf(2, d.ring_k, size_t(l) * ring_layer);
    e.buf(3, d.ring_v, size_t(l) * ring_layer);
    e.val(4, kp);
    if (chained) e.dispatch_indirect(cyc_, kCycRing, 128);  // the accepted rows only
    else e.dispatch(uint32_t(R), uint32_t(2 * hkv_), 1, 128);
  }
}

void DSpark::encode_prefill_commit(Encoder& e, LiliSeq& d) {
  enc_commit_rows(e, d, pfeat_, pf_n_ - pf_lo_, pf_lo_);
  d.ctx = pf_n_;
  d.hi = std::max(d.hi, d.ctx);
}

void DSpark::encode_commit(Encoder& e, LiliSeq& d, int m) {
  enc_commit_rows(e, d, vfeat_, m, d.ctx);
  d.ctx += m;
  d.hi = std::max(d.hi, d.ctx);
}

// ---- drafting pass ----
void DSpark::encode_pass(Encoder& e, Seq& s, LiliSeq& d, int P) {
  if (d.ctx != P) fatal("DSpark: drafting at %d with context %d", P, d.ctx);
  enc_draft(e, s, d, P, false, false);
}

void DSpark::encode_draft(Encoder& e, Seq& s, LiliSeq& d, int P) {
  if (d.ctx != P) fatal("DSpark: drafting at %d with context %d", P, d.ctx);
  enc_draft(e, s, d, P, false, true);
}

// Chained behind the verify at P: the parameters from its acceptance (nh_dsp_cycle), the commit of all 8 verify
// rows' K/V (the GEMMs' rows are independent; the ring takes the accepted ones), the drafting pass from the anchor.
void DSpark::encode_draft_chained(Encoder& e, Seq& s, LiliSeq& d, int P) {
  if (d.ctx != P) fatal("DSpark: chained drafting behind a verify at %d with context %d", P, d.ctx);
  const CycP cp{uint32_t(P),     uint32_t(d.hi),  uint32_t(window_),  uint32_t(H_), uint32_t(hq_), uint32_t(hkv_),
                kB,              kB - 1,          uint32_t(K_),       uint32_t(rank_), kAttnSplit, dl_[0].qkv.N,
                eps_,            theta_,          1.0f / std::sqrt(128.0f), w2_.gs};
  e.use(pipe("nh_dsp_cycle"));
  e.buf(0, tg_.accepted_buf());
  e.buf(1, cyc_);
  e.val(2, cp);
  e.dispatch(1, 1, 1, 1);
  enc_commit(e, d, vfeat_, kB, P, true);
  enc_draft(e, s, d, P, true, true);
}

void DSpark::enc_draft(Encoder& e, Seq& s, LiliSeq& d, int P, bool chained, bool walk) {
  const uint32_t H = uint32_t(H_), B = kB, S = B - 1, K = uint32_t(K_);
  const int R = int(B);
  // the block: the anchor's embedding (the target's table), the mask token's row below it
  tg_.enc_embed(e, s.tok, P, 1, x_, chained ? &cyc_ : nullptr, kCycEmbed);
  e.use(pipe("nh_dsp_mask_rows"));
  e.buf(0, x_);
  e.buf(1, mask_row_);
  e.val(2, H);
  e.dispatch(S, (H + 255) / 256, 1, 256);
  const size_t ring_layer = size_t(hkv_) * window_ * 128 * 2;
  for (int l = 0; l < NL_; ++l) {
    const DLayer& L = dl_[size_t(l)];
    norm_half(e, x_, L.attn_norm, xh_, R);
    gemm_q8(e, L.qkv, xh_, q_, R, 0);  // q|k|v [R][(Hq + 2 Hkv) 128]
    // the block attention (q / k norm + RoPE inside): the context in kAttnSplit pieces, then the pieces with the
    // block's keys and the sinks, into half rows for o
    const AttnP ap{uint32_t(P), uint32_t(std::max({0, P - window_, d.hi - window_})), uint32_t(window_), uint32_t(hq_),
                   uint32_t(hkv_), B, kAttnSplit, L.qkv.N, 1.0f / std::sqrt(128.0f), eps_, theta_};
    if (l == 0) {  // the block's RoPE table (positions P..P+7), for every layer
      e.use(pipe("nh_dsp_rope_tab"));
      e.buf(0, rope_);
      if (chained) e.buf(1, cyc_, kCycAttn);
      else e.val(1, ap);
      e.dispatch(1, 1, 1, B * 64);
    }
    e.use(pipe("nh_dsp_attn_part"));
    e.buf(0, q_);
    e.buf(1, L.qn);
    e.buf(2, d.ring_k, size_t(l) * ring_layer);
    e.buf(3, d.ring_v, size_t(l) * ring_layer);
    e.buf(4, ao_);
    if (chained) e.buf(5, cyc_, kCycAttn);
    else e.val(5, ap);
    e.buf(6, rope_);
    e.dispatch(uint32_t(hkv_), B, kAttnSplit, uint32_t(32 * (hq_ / hkv_)));
    e.use(pipe("nh_dsp_attn_fin"));
    e.buf(0, q_);
    e.buf(1, L.qn);
    e.buf(2, L.kn);
    e.buf(3, L.sinks);
    e.buf(4, ao_);
    e.buf(5, xh_);
    if (chained) e.buf(6, cyc_, kCycAttn);
    else e.val(6, ap);
    e.buf(7, rope_);
    e.dispatch(uint32_t(hkv_), B, 1, uint32_t(32 * (hq_ / hkv_)));
    gemm_q8(e, L.o, xh_, x_, R, 1);  // the residual add in the GEMM's epilogue
    norm_half(e, x_, L.ffn_norm, xh_, R);
    tg_.enc_gemm(e, L.gu, xh_, gt_, R, 0);  // gate|up [R][2 I]
    // silu(gate) * up to half with a per-row scale (it may leave half's range), the down product, scaled back
    // into the residual
    const struct {
      uint32_t I, ld;
      float gs_g, gs_u;
    } sw{uint32_t(I_), L.gu.N, L.gs_g, L.gs_u};
    e.use(pipe("nh_dsp_swiglu_half"));
    e.buf(0, gt_);
    e.buf(1, xh_);
    e.buf(2, rscale_);
    e.val(3, sw);
    e.dispatch(B, 1, 1, 1024);
    tg_.enc_gemm(e, L.down, xh_, t2_, R, 0);
    e.use(pipe("nh_dsp_scale_add"));
    e.buf(0, x_);
    e.buf(1, t2_);
    e.buf(2, rscale_);
    e.val(3, H);
    e.dispatch(B, (H + 255) / 256, 1, 256);
  }
  // slot logits (rows 1..7, normalized into half) with the target's LM head (rows [0, head_rows_): NI_DSPARK_HEAD),
  // their top-K candidates
  norm_half(e, x_.view(size_t(H) * 4, size_t(S) * H * 4), out_norm_, xh_, int(S));
  NhMat head = tg_.lm_head();
  head.N = uint32_t(head_rows_);
  const uint32_t V = head.N;
  tg_.enc_gemm(e, head, xh_, logits_, int(S), 0);
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
  if (!walk) return;  // KFlash: the candidates alone (encode_pass)
  // the Markov head's walk from the anchor: tokens[P + 1 .. P + 7]
  const WalkP wp{S, K, uint32_t(rank_), uint32_t(P), w2_.gs};
  // (two passes: every candidate pair's Markov dot at once, then the walk on one simdgroup)
  e.use(pipe("nh_dsp_walk_dots"));
  e.buf(0, cand_);
  e.buf(1, w1_);
  e.buf(2, w2_.q);
  e.buf(3, w2_.s);
  e.buf(4, s.tok);
  e.buf(5, walk_d_);
  if (chained) e.buf(6, cyc_, kCycWalk);
  else e.val(6, wp);
  e.dispatch(S, 1, 1, 32 * K);
  e.use(pipe("nh_dsp_walk_pick"));
  e.buf(0, cand_);
  e.buf(1, clp_);
  e.buf(2, walk_d_);
  e.buf(3, s.tok);
  e.buf(4, picks_);
  e.buf(5, qlp_);
  if (chained) e.buf(6, cyc_, kCycWalk);
  else e.val(6, wp);
  e.dispatch(1, 1, 1, 32);
  // the coming verify's depth from the corrected probabilities of the picks (NI_DSPARK_DYN; 0 verifies every draft)
  if (fixed_depth_ >= 0) {
    const uint32_t fd = uint32_t(fixed_depth_);
    e.use(pipe("nh_dsp_fixed_depth"));
    e.buf(0, tg_.live_rows());
    e.val(1, fd);
    e.dispatch(1, 1, 1, 1);
    return;
  }
  const DepthP dp{S, K, dyn_ratio_, 0};
  e.use(pipe("nh_lili_depth"));
  e.buf(0, qlp_);
  e.buf(1, picks_);
  e.buf(2, tg_.live_rows());
  e.val(3, dp);
  e.buf(4, qlp_);
  e.dispatch(1, 1, 1, 1);
}

// KFlash: the Markov head's corrected probabilities of every candidate after every candidate of the slot before
// (nh_dsp_table), then the best-first tree under the budget (nh_kf_build)
void DSpark::encode_draft_tree(Encoder& e, Seq& s, LiliSeq& d, int P, const Buffer& kfp) {
  encode_pass(e, s, d, P);
  const uint32_t S = kB - 1, K = uint32_t(K_);
  const DspTableP tp{S, K, uint32_t(rank_), uint32_t(P), w2_.gs};
  e.use(pipe("nh_dsp_table"));
  e.buf(0, cand_);
  e.buf(1, clp_);
  e.buf(2, w1_);
  e.buf(3, w2_.q);
  e.buf(4, w2_.s);
  e.buf(5, s.tok);
  e.buf(6, tlq_);
  e.val(7, tp);
  e.dispatch(1 + (S - 1) * K, 1, 1, 32 * K);
  enc_kf_build(e, tg_, s, P, cand_, tlq_, int(K), kfp);
}

void DSpark::debug_read(std::vector<int>& cand, std::vector<float>& clp, std::vector<float>& qlp,
                        std::vector<int>& picks) const {
  const size_t n = size_t(kB - 1) * size_t(K_);
  cand.assign(cand_.as<int>(), cand_.as<int>() + n);
  clp.assign(clp_.as<float>(), clp_.as<float>() + n);
  qlp.assign(qlp_.as<float>(), qlp_.as<float>() + n);
  picks.assign(picks_.as<int>(), picks_.as<int>() + (kB - 1));
}

}  // namespace ni
