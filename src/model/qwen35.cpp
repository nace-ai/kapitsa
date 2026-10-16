#include "model/qwen35.h"
#include "model/kparams.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <unordered_map>

namespace ni {

using json = nlohmann::json;

static json read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) fatal("cannot read %s", path.c_str());
  return json::parse(f);
}

static json quant_cfg_;
static Pipeline pipe(const std::string& n);

// NI_MOE_FUSED (read once): MoE prefill through the fused grouped kernels in qmm_moe.metal.
// 1: gate/up/SwiGLU in one GEMM, down GEMM, shared expert folded in as one more expert;
// 2 (default): same, shared expert on the dense path (35B 4k: 785 -> 862 tok/s); 0: the original path.
static int moe_fused() {
  static const int v = getenv("NI_MOE_FUSED") && *getenv("NI_MOE_FUSED") ? atoi(getenv("NI_MOE_FUSED")) : 2;
  return v;
}

void Qwen35::load_config(const std::string& dir) {
  json root = read_json(dir + "/config.json");
  json t = root.contains("text_config") ? root["text_config"] : root;
  quant_cfg_ = root.contains("quantization") ? root["quantization"] : json::object();
  cfg_.group_size = quant_cfg_.value("group_size", 64);
  cfg_.hidden = t["hidden_size"];
  cfg_.layers = t["num_hidden_layers"];
  cfg_.vocab = t["vocab_size"];
  cfg_.eps = t.value("rms_norm_eps", 1e-6f);
  cfg_.lin_hk = t["linear_num_key_heads"];
  cfg_.lin_hv = t["linear_num_value_heads"];
  cfg_.lin_dk = t["linear_key_head_dim"];
  cfg_.lin_dv = t["linear_value_head_dim"];
  cfg_.conv_k = t.value("linear_conv_kernel_dim", 4);
  cfg_.hq = t["num_attention_heads"];
  cfg_.hkv = t["num_key_value_heads"];
  cfg_.head_dim = t.value("head_dim", cfg_.hidden / cfg_.hq);
  float prf = 0.25f;
  if (t.contains("rope_parameters")) {
    auto& rp = t["rope_parameters"];
    cfg_.theta = rp.value("rope_theta", 1e7f);
    prf = rp.value("partial_rotary_factor", 0.25f);
  } else {
    cfg_.theta = t.value("rope_theta", 1e7f);
    prf = t.value("partial_rotary_factor", 0.25f);
  }
  cfg_.rot_dims = int(cfg_.head_dim * prf);
  int interval = t.value("full_attention_interval", 4);
  cfg_.is_linear.resize(cfg_.layers);
  if (t.contains("layer_types") && t["layer_types"].is_array()) {
    for (int i = 0; i < cfg_.layers; ++i) cfg_.is_linear[i] = t["layer_types"][i] == "linear_attention";
  } else {
    for (int i = 0; i < cfg_.layers; ++i) cfg_.is_linear[i] = (i + 1) % interval != 0;
  }
  cfg_.experts = t.value("num_experts", 0);
  cfg_.moe = cfg_.experts > 0;
  cfg_.top_k = t.value("num_experts_per_tok", 0);
  cfg_.moe_I = t.value("moe_intermediate_size", 0);
  cfg_.shared_I = t.value("shared_expert_intermediate_size", 0);
  cfg_.dense_I = t.value("intermediate_size", 0);
  cfg_.norm_topk = t.value("norm_topk_prob", true);
  auto eos = root.contains("eos_token_id") ? root["eos_token_id"] : t["eos_token_id"];
  if (eos.is_array())
    for (auto& e : eos) cfg_.eos.push_back(e);
  else
    cfg_.eos.push_back(eos);
}

QLinear Qwen35::qlin(const std::string& name, bool required) {
  QLinear L;
  std::string full = prefix_ + name;
  L.w = ws_.find(full + ".weight");
  if (!L.w) {
    if (required) fatal("missing %s.weight", full.c_str());
    return L;
  }
  L.s = ws_.find(full + ".scales");
  L.b = ws_.find(full + ".biases");
  if (!L.s || !L.b) fatal("%s is not MLX affine-quantized (bf16 linears not supported yet)", full.c_str());
  int bits = quant_cfg_.value("bits", 4), gs = quant_cfg_.value("group_size", 64);
  // per-module overrides are keyed by the module path without ".weight"
  for (auto key : {full, name}) {
    if (quant_cfg_.contains(key) && quant_cfg_[key].is_object()) {
      bits = quant_cfg_[key].value("bits", bits);
      gs = quant_cfg_[key].value("group_size", gs);
    }
  }
  L.bits = bits;
  L.gs = gs;
  int64_t words = L.w->shape.back();
  L.K = uint32_t(words * 32 / bits);
  L.N = uint32_t(L.w->shape[L.w->shape.size() - 2]);
  if (int64_t(L.K / gs) != L.s->shape.back()) fatal("%s: bits/group mismatch", full.c_str());
  return L;
}

Qwen35::Qwen35(const std::string& dir, int prefill_chunk) : dir_(dir), tmax_(prefill_chunk) {
  double t0 = now_ms();
  load_config(dir);
  // Text-only engine for now: skip the vision tower and (until speculative decoding lands) MTP.
  ws_.open_dir(dir, [](const std::string& n) {
    return n.find("vision_tower") == std::string::npos && n.find("visual.") == std::string::npos &&
           n.rfind("mtp.", 0) != 0;
  });
  if (ws_.has("language_model.model.embed_tokens.weight")) prefix_ = "language_model.";
  else if (ws_.has("model.embed_tokens.weight")) prefix_ = "";
  else fatal("unrecognized weight naming");

  embed_ = qlin("model.embed_tokens");
  lm_head_ = qlin("lm_head");
  final_norm_ = &ws_.get(prefix_ + "model.norm.weight");
  layers_.resize(cfg_.layers);
  for (int i = 0; i < cfg_.layers; ++i) {
    Layer& L = layers_[i];
    std::string p = "model.layers." + std::to_string(i) + ".";
    L.linear = cfg_.is_linear[i];
    L.in_norm = &ws_.get(prefix_ + p + "input_layernorm.weight");
    L.post_norm = &ws_.get(prefix_ + p + "post_attention_layernorm.weight");
    if (L.linear) {
      std::string a = p + "linear_attn.";
      L.qkv = qlin(a + "in_proj_qkv");
      L.z = qlin(a + "in_proj_z");
      L.a = qlin(a + "in_proj_a");
      L.b = qlin(a + "in_proj_b");
      L.out = qlin(a + "out_proj");
      L.conv_w = &ws_.get(prefix_ + a + "conv1d.weight");
      L.A_log = &ws_.get(prefix_ + a + "A_log");
      L.dt_bias = &ws_.get(prefix_ + a + "dt_bias");
      L.gnorm = &ws_.get(prefix_ + a + "norm.weight");
    } else {
      std::string a = p + "self_attn.";
      L.q = qlin(a + "q_proj");
      L.k = qlin(a + "k_proj");
      L.v = qlin(a + "v_proj");
      L.o = qlin(a + "o_proj");
      L.q_norm = &ws_.get(prefix_ + a + "q_norm.weight");
      L.k_norm = &ws_.get(prefix_ + a + "k_norm.weight");
    }
    if (cfg_.moe) {
      std::string m = p + "mlp.";
      L.router = qlin(m + "gate");
      L.shared_gate = qlin(m + "shared_expert_gate");
      L.eg = qlin(m + "switch_mlp.gate_proj");
      L.eu = qlin(m + "switch_mlp.up_proj");
      L.ed = qlin(m + "switch_mlp.down_proj");
      L.sg = qlin(m + "shared_expert.gate_proj");
      L.su = qlin(m + "shared_expert.up_proj");
      L.sd = qlin(m + "shared_expert.down_proj");
    } else {
      std::string m = p + "mlp.";
      L.mg = qlin(m + "gate_proj");
      L.mu = qlin(m + "up_proj");
      L.md = qlin(m + "down_proj");
    }
  }
  // Checks the kernels' assumptions.
  if (cfg_.lin_dk != 128 || cfg_.lin_dv != 128) fatal("GDN head dims must be 128");
  if (cfg_.head_dim != 256) fatal("attention head_dim must be 256");
  if (cfg_.hq / cfg_.hkv > 8) fatal("GQA ratio > 8 unsupported");
  if (cfg_.conv_k != 4) fatal("conv kernel must be 4");
  if (cfg_.moe && (cfg_.moe_I % 512 || cfg_.shared_I % 512)) fatal("expert sizes must be multiples of 512");
  if (cfg_.moe && (cfg_.top_k + 1) * std::max(cfg_.moe_I, cfg_.shared_I) > 4608) fatal("MoE decode: (k+1)*I > 4608");
  if (cfg_.moe && (cfg_.experts % 32 || cfg_.experts > 512 || cfg_.top_k > 16)) fatal("MoE: unsupported expert count");
  if (cfg_.moe && (cfg_.hidden > 4096 || cfg_.lin_hv * cfg_.lin_dv > 4096 || cfg_.hq * cfg_.head_dim > 4096))
    fatal("MoE decode kernels stage at most 4096 inputs");
  if (embed_.bits != 4) fatal("embedding must be 4-bit");

  if (!getenv("NI_NO_MTP")) load_mtp(dir);
  Gpu::get().make_resident(ws_.buffers());
  alloc_state();
  logi("loaded %s: %d layers, %s, %.1f GB mapped in %.0f ms", dir.c_str(), cfg_.layers,
       cfg_.moe ? "MoE" : "dense", ws_.total_bytes() / 1e9, now_ms() - t0);
}

void Qwen35::alloc_state() {
  Gpu& g = Gpu::get();
  const int D = cfg_.hidden;
  const int T = tmax_; // activation buffers hold a whole prefill chunk; decode uses row 0
  const int C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  const int VD = cfg_.lin_hv * cfg_.lin_dv;
  x_ = g.alloc(size_t(T) * D * 4, "x");
  xn_ = g.alloc(size_t(D) * 4, "xn");
  logits_ = g.alloc(size_t(cfg_.vocab) * kMaxVerify * 4, "logits");
  qkv_ = g.alloc(size_t(T) * C * 4);
  zb_ = g.alloc(size_t(T) * VD * 4);
  ab_ = g.alloc(size_t(T) * cfg_.lin_hv * 4);
  bb_ = g.alloc(size_t(T) * cfg_.lin_hv * 4);
  gq_ = g.alloc(size_t(T) * cfg_.lin_hk * cfg_.lin_dk * 4);
  gk_ = g.alloc(size_t(T) * cfg_.lin_hk * cfg_.lin_dk * 4);
  gv_ = g.alloc(size_t(T) * VD * 4);
  gg_ = g.alloc(size_t(T) * cfg_.lin_hv * 4);
  gbeta_ = g.alloc(size_t(T) * cfg_.lin_hv * 4);
  gout_ = g.alloc(size_t(T) * VD * 4);
  qg_ = g.alloc(size_t(T) * cfg_.hq * cfg_.head_dim * 2 * 4);
  kr_ = g.alloc(size_t(T) * cfg_.hkv * cfg_.head_dim * 4);
  vr_ = g.alloc(size_t(T) * cfg_.hkv * cfg_.head_dim * 4);
  qo_ = g.alloc(size_t(T) * cfg_.hq * cfg_.head_dim * 4);
  attn_out_ = g.alloc(size_t(T) * cfg_.hq * cfg_.head_dim * 4);
  const int max_chunks = 1024;
  part_o_ = g.alloc(size_t(cfg_.hq) * max_chunks * cfg_.head_dim * 4);
  part_ml_ = g.alloc(size_t(cfg_.hq) * max_chunks * 2 * 4);
  eidx_ = g.alloc(64 * 4);
  ewts_ = g.alloc(64 * 4);
  eh_ = g.alloc(size_t(kMaxVerify * cfg_.top_k + 1) * kMaxVerify * std::max(cfg_.moe_I, cfg_.shared_I) * 4 + 16);
  ue_ = g.alloc(size_t(kMaxVerify) * 64 * 4, "uniq_experts");
  uw_ = g.alloc(size_t(kMaxVerify) * 64 * kMaxVerify * 4, "uniq_weights");
  un_ = g.alloc(16, "uniq_count");
  cand_v_ = g.alloc(size_t(kMaxVerify) * (cfg_.vocab / 2048 + 2) * 64 * 4);
  cand_i_ = g.alloc(size_t(kMaxVerify) * (cfg_.vocab / 2048 + 2) * 64 * 4);
  mtp_e_ = g.alloc(size_t(T) * D * 4, "mtp_e");
  mtp_u_ = g.alloc(size_t(T) * 2 * D * 4, "mtp_u");
  mtp_x_ = g.alloc(size_t(T) * D * 4, "mtp_x");
  mtp_post_ = g.alloc(size_t(kMaxVerify) * D * 4, "mtp_post");
  xrn_ = g.alloc(size_t(kMaxVerify) * std::max(D, VD) * 4, "xrn");
  samp_ = g.alloc(64, "samples");
  dlog_ = g.alloc(size_t(cfg_.vocab) * 4, "draft_logits");
  mh_ = g.alloc(size_t(kMaxVerify) * std::max(cfg_.dense_I, 1) * 4, "mlp_h");
  sprob_ = g.alloc(64 * 4);
  {
    const size_t P = size_t(T) * std::max(cfg_.top_k, 1);
    const int I = std::max({cfg_.moe_I, cfg_.shared_I, cfg_.dense_I, 1});
    // fused MoE prefill: T shared-expert rows after the P pair rows (hg/hu stay: a layer the
    // fused kernels do not support falls back to the original path)
    const size_t S = cfg_.moe && moe_fused() ? size_t(T) : 0;
    const int W = std::max({D, VD, cfg_.hq * cfg_.head_dim, cfg_.dense_I}); // widest half input row
    xh_ = g.alloc(size_t(T) * W * 2, "xh");
    hh_ = g.alloc(size_t(T) * W * 2, "hh");
    ridx_ = g.alloc(P * 4);
    rwts_ = g.alloc(size_t(T) * (cfg_.top_k + 1) * 4 + 16);
    spair_ = g.alloc((P + S) * 4);
    stok_ = g.alloc((P + S) * 4);
    tiles_ = g.alloc((P + S + size_t(std::max(cfg_.experts, 1)) + 1) * 3 * 4);
    ntiles_ = g.alloc(16);
    hg_ = g.alloc(P * I * 4);
    hu_ = g.alloc(P * I * 4);
    hhalf_ = g.alloc((P + S) * I * 2);
    yd_ = g.alloc((P + S) * D * 4);
    sgb_ = g.alloc(size_t(T) * I * 4);
    sub_ = g.alloc(size_t(T) * I * 4);
    ys_ = g.alloc(size_t(T) * D * 4);
    rlog_ = g.alloc(size_t(T) * std::max(cfg_.experts, 1) * 4);
    slog_ = g.alloc(size_t(T) * 4 + 16);
  }
}

std::unique_ptr<Seq> Qwen35::new_seq(int cap) {
  Gpu& g = Gpu::get();
  auto s = std::make_unique<Seq>();
  s->cap = cap;
  s->tok = g.alloc(size_t(cap + 8) * 4, "tokens");
  const int C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  size_t bytes = 0;
  for (int i = 0; i < cfg_.layers; ++i) {
    if (cfg_.is_linear[i]) {
      s->conv.push_back(g.alloc(size_t(3) * C * 4, "conv_state"));
      s->rec.push_back(g.alloc(size_t(cfg_.lin_hv) * cfg_.lin_dv * cfg_.lin_dk * 4, "gdn_state"));
      s->kc.push_back(Buffer{});
      s->vc.push_back(Buffer{});
      bytes += s->conv.back().size + s->rec.back().size;
      s->pend_qkv.push_back(g.alloc(size_t(kMaxVerify) * C * 4, "pend_qkv"));
      s->pend_k.push_back(g.alloc(size_t(kMaxVerify) * cfg_.lin_hv * cfg_.lin_dk * 4, "pend_k"));
      s->pend_g.push_back(g.alloc(size_t(kMaxVerify) * cfg_.lin_hv * 4, "pend_g"));
      s->pend_d.push_back(g.alloc(size_t(kMaxVerify) * cfg_.lin_hv * cfg_.lin_dv * 4, "pend_d"));
    } else {
      size_t kvb = size_t(cfg_.hkv) * cap * cfg_.head_dim * 2;
      s->conv.push_back(Buffer{});
      s->rec.push_back(Buffer{});
      s->kc.push_back(g.alloc(kvb, "k_cache"));
      s->vc.push_back(g.alloc(kvb, "v_cache"));
      bytes += 2 * kvb;
      s->pend_qkv.push_back(Buffer{});
      s->pend_k.push_back(Buffer{});
      s->pend_g.push_back(Buffer{});
      s->pend_d.push_back(Buffer{});
    }
  }
  if (mtp_loaded_) {
    size_t kvb = size_t(cfg_.hkv) * cap * cfg_.head_dim * 2;
    s->mtp_kc = g.alloc(kvb, "mtp_k_cache");
    s->mtp_vc = g.alloc(kvb, "mtp_v_cache");
    bytes += 2 * kvb;
  }
  logd("sequence: %.0f MB (KV for %d tokens)", bytes / 1e6, cap);
  reset(*s);
  return s;
}

void Qwen35::reset(Seq& s) {
  s.n_pend = 0;
  for (auto& b : s.conv)
    if (b) memset(b.ptr(), 0, b.size);
  for (auto& b : s.rec)
    if (b) memset(b.ptr(), 0, b.size);
}

Checkpoint Qwen35::new_checkpoint() {
  Gpu& g = Gpu::get();
  Checkpoint c;
  const int C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  for (int i = 0; i < cfg_.layers; ++i) {
    bool lin = cfg_.is_linear[i];
    c.conv.push_back(lin ? g.alloc(size_t(3) * C * 4, "ckpt_conv") : Buffer{});
    c.rec.push_back(lin ? g.alloc(size_t(cfg_.lin_hv) * cfg_.lin_dv * cfg_.lin_dk * 4, "ckpt_rec") : Buffer{});
  }
  return c;
}

void Qwen35::enc_copy(Encoder& e, const Buffer& src, const Buffer& dst) {
  static Pipeline cp = Gpu::get().pipeline("copy_u32");
  uint32_t n = uint32_t(std::min(src.size, dst.size) / 16);
  e.use(cp);
  e.buf(0, src);
  e.buf(1, dst);
  e.val(2, n);
  e.dispatch((n + 255) / 256, 1, 1, 256);
}

void Qwen35::save(Encoder& e, const Seq& s, Checkpoint& c, int pos) {
  for (int i = 0; i < cfg_.layers; ++i)
    if (cfg_.is_linear[i]) {
      enc_copy(e, s.conv[i], c.conv[i]);
      enc_copy(e, s.rec[i], c.rec[i]);
    }
  c.pos = pos;
}

void Qwen35::restore(Encoder& e, Seq& s, const Checkpoint& c) {
  for (int i = 0; i < cfg_.layers; ++i)
    if (cfg_.is_linear[i]) {
      enc_copy(e, c.conv[i], s.conv[i]);
      enc_copy(e, c.rec[i], s.rec[i]);
    }
}

// ---- encoders ----
static constexpr uint32_t kRows = 4, kSg = 2; // qmv: rows per simdgroup, simdgroups per threadgroup

void Qwen35::enc_qmv(Encoder& e, const QLinear& L, const Buffer& x, const Buffer& y, const Buffer* resid,
                     const Tensor* norm) {
  // Tuned on M4 Pro: huge matrices (lm_head) like 4 rows x 4 simdgroups, the rest 2 rows x 8.
  const bool huge = L.N >= 65536;
  uint32_t rows = huge ? 4 : 2, nsg = huge ? 4 : 8;
  if (L.gs != 64) fatal("qmv: group size %d unsupported", L.gs);
  std::string name = "qmv2_b" + std::to_string(L.bits) + "_g64_r" + std::to_string(rows);
  if (L.K > 4096) {  // wide inputs: x staged in 4096-wide tiles
    if (L.bits == 8 || resid) rows = 2;
    name = "qmv3_b" + std::to_string(L.bits) + "_r" + std::to_string(rows);
  }
  if (resid) name += "_add";
  if (norm) name += "_norm";
  static std::unordered_map<std::string, Pipeline> cache;
  auto it = cache.find(name);
  if (it == cache.end()) it = cache.emplace(name, Gpu::get().pipeline(name)).first;
  e.use(it->second);
  kp::Qmv p{L.N, L.K, L.K, L.N, cfg_.eps};
  e.buf(0, L.w->buf);
  e.buf(1, L.s->buf);
  e.buf(2, L.b->buf);
  e.buf(3, x);
  e.buf(4, y);
  e.val(5, p);
  e.buf(6, resid ? *resid : y);
  e.buf(7, norm ? norm->buf : y);
  e.dispatch((L.N + rows * nsg - 1) / (rows * nsg), 1, 1, 32 * nsg);
}

void Qwen35::enc_qmv_multi(Encoder& e, std::vector<Seg> segs, const Buffer& x, const Tensor* norm,
                           const Buffer* xn_out) {
  static Pipeline pp = Gpu::get().pipeline("qmv2_multi_r2");
  static Pipeline pw = Gpu::get().pipeline("qmv3_multi_r2");
  constexpr uint32_t rows = 2, nsg = 8;
  e.use(segs[0].L->K > 4096 ? pw : pp);
  kp::QmvMulti2 p{};
  p.K = segs[0].L->K;
  p.nseg = uint32_t(segs.size());
  p.eps = cfg_.eps;
  p.norm = norm ? 1u : 0u;
  p.write_xn = xn_out ? 1u : 0u;
  uint32_t tg = 0;
  for (size_t i = 0; i < 4; ++i) {
    const Seg& s = segs[std::min(i, segs.size() - 1)];
    if (i < segs.size()) {
      if (s.L->K != p.K || s.L->gs != 64) fatal("qmv_multi: segments must share K and group 64");
      p.N[i] = s.L->N;
      p.bits[i] = uint32_t(s.L->bits);
      p.tg_begin[i] = tg;
      tg += (s.L->N + rows * nsg - 1) / (rows * nsg);
    }
    e.buf(int(i * 4 + 0), s.L->w->buf);
    e.buf(int(i * 4 + 1), s.L->s->buf);
    e.buf(int(i * 4 + 2), s.L->b->buf);
    e.buf(int(i * 4 + 3), s.y);
  }
  p.tg_begin[segs.size()] = tg;
  e.buf(16, x);
  e.val(17, p);
  e.buf(18, norm ? norm->buf : x);
  e.buf(19, xn_out ? *xn_out : x);
  e.dispatch(tg, 1, 1, 32 * nsg);
}

void Qwen35::enc_gdn(Encoder& e, Seq& s, int li, const Layer& L) {
  static Pipeline prep = Gpu::get().pipeline("gdn_prep");
  static Pipeline step = Gpu::get().pipeline("gdn_step");
  enc_qmv_multi(e, {{&L.qkv, qkv_}, {&L.z, zb_}, {&L.b, bb_}, {&L.a, ab_}}, x_, L.in_norm);
  const uint32_t C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  kp::GdnPrep pp{1, uint32_t(cfg_.lin_hk), uint32_t(cfg_.lin_hv), C};
  e.use(prep);
  e.buf(0, qkv_); e.buf(1, ab_); e.buf(2, bb_); e.buf(3, s.conv[li]);
  e.buf(4, L.conv_w->buf); e.buf(5, L.A_log->buf); e.buf(6, L.dt_bias->buf);
  e.buf(7, gq_); e.buf(8, gk_); e.buf(9, gv_); e.buf(10, gg_); e.buf(11, gbeta_);
  e.val(12, pp);
  e.dispatch(C / 128, 1, 1, 128);
  kp::GdnStep sp{1, uint32_t(cfg_.lin_hk), uint32_t(cfg_.lin_hv), cfg_.eps};
  e.use(step);
  e.buf(0, gq_); e.buf(1, gk_); e.buf(2, gv_); e.buf(3, gg_); e.buf(4, gbeta_);
  e.buf(5, s.rec[li]); e.buf(6, zb_); e.buf(7, L.gnorm->buf); e.buf(8, gout_);
  e.val(9, sp);
  e.dispatch(cfg_.lin_hv, 1, 1, 1024);
  enc_qmv(e, L.out, gout_, x_, &x_, nullptr);
}

void Qwen35::enc_attn(Encoder& e, Seq& s, int li, const Layer& L, int pos) {
  static Pipeline prep = Gpu::get().pipeline("attn_prep");
  static Pipeline part = Gpu::get().pipeline("attn_decode_partial");
  static Pipeline red = Gpu::get().pipeline("attn_decode_reduce");
  enc_qmv_multi(e, {{&L.q, qg_}, {&L.k, kr_}, {&L.v, vr_}}, x_, L.in_norm);
  kp::AttnPrep ap{1, uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(cfg_.head_dim), uint32_t(cfg_.rot_dims),
                  uint32_t(pos), uint32_t(s.cap), cfg_.eps, cfg_.theta};
  e.use(prep);
  e.buf(0, qg_); e.buf(1, kr_); e.buf(2, vr_); e.buf(3, L.q_norm->buf); e.buf(4, L.k_norm->buf);
  e.buf(5, qo_); e.buf(6, s.kc[li]); e.buf(7, s.vc[li]);
  e.val(8, ap);
  e.dispatch(cfg_.hq + 2 * cfg_.hkv, 1, 1, cfg_.head_dim);

  const uint32_t Lk = uint32_t(pos + 1);
  // aim for >= ~40 threadgroups overall, chunks of at least 64 positions
  uint32_t chunk = std::max<uint32_t>(64, (Lk + 19) / 20);
  chunk = (chunk + 15) / 16 * 16;
  uint32_t nch = (Lk + chunk - 1) / chunk;
  kp::AttnDec dp{uint32_t(cfg_.hq), uint32_t(cfg_.hkv), Lk, chunk, uint32_t(s.cap),
                 1.0f / std::sqrt(float(cfg_.head_dim)), nch};
  e.use(part);
  e.buf(0, qo_); e.buf(1, s.kc[li]); e.buf(2, s.vc[li]); e.buf(3, part_o_); e.buf(4, part_ml_);
  e.val(5, dp);
  e.dispatch(nch, cfg_.hkv, 1, 32 * (cfg_.hq / cfg_.hkv));
  e.use(red);
  e.buf(0, part_o_); e.buf(1, part_ml_); e.buf(2, qg_); e.buf(3, attn_out_);
  e.val(4, dp);
  e.dispatch(cfg_.hq, 1, 1, cfg_.head_dim);
  enc_qmv(e, L.o, attn_out_, x_, &x_, nullptr);
}

void Qwen35::enc_moe(Encoder& e, const Layer& L) {
  // NI_GU: 2 = fused top-k (v2), 3 = separate top-k dispatch + v3 rows 2, 4 = v3 rows 4
  static int variant = getenv("NI_GU") ? atoi(getenv("NI_GU")) : 3;
  static Pipeline gu = Gpu::get().pipeline(variant == 2 ? "moe_gate_up2" : variant == 3 ? "moe_gate_up3_r2" : "moe_gate_up3_r4");
  static Pipeline dn = Gpu::get().pipeline("moe_down2");
  static Pipeline topk = Gpu::get().pipeline(cfg_.experts == 256 && cfg_.top_k <= 8 ? "moe_topk256" : "moe_topk");
  constexpr uint32_t nsg = 8;
  const uint32_t gu_rows = variant == 4 ? 4 : 2, rows = 2;
  enc_qmv_multi(e, {{&L.router, rlog_}, {&L.shared_gate, slog_}}, x_, L.post_norm, &xn_);
  kp::Moe mp{uint32_t(cfg_.hidden), uint32_t(cfg_.moe_I), uint32_t(cfg_.shared_I), uint32_t(cfg_.top_k), cfg_.eps};
  uint32_t E = uint32_t(cfg_.experts);
  if (variant != 2) {
    kp::Topk tp{E, uint32_t(cfg_.top_k), cfg_.norm_topk ? 1u : 0u};
    e.use(topk);
    e.buf(0, rlog_); e.buf(1, slog_); e.buf(2, eidx_); e.buf(3, ewts_);
    e.val(4, tp);
    e.dispatch(1, 1, 1, 32);
  }
  e.use(gu);
  e.buf(0, L.eg.w->buf); e.buf(1, L.eg.s->buf); e.buf(2, L.eg.b->buf);
  e.buf(3, L.eu.w->buf); e.buf(4, L.eu.s->buf); e.buf(5, L.eu.b->buf);
  e.buf(6, L.sg.w->buf); e.buf(7, L.sg.s->buf); e.buf(8, L.sg.b->buf);
  e.buf(9, L.su.w->buf); e.buf(10, L.su.s->buf); e.buf(11, L.su.b->buf);
  e.buf(12, xn_); e.buf(13, rlog_); e.buf(14, slog_); e.buf(15, eh_);
  e.val(16, mp);
  e.buf(17, eidx_); e.buf(18, ewts_); e.val(19, E);
  uint32_t I = uint32_t(std::max(cfg_.moe_I, cfg_.shared_I));
  e.dispatch((I + gu_rows * nsg - 1) / (gu_rows * nsg), cfg_.top_k + 1, 1, 32 * nsg);
  e.use(dn);
  e.buf(0, L.ed.w->buf); e.buf(1, L.ed.s->buf); e.buf(2, L.ed.b->buf);
  e.buf(3, L.sd.w->buf); e.buf(4, L.sd.s->buf); e.buf(5, L.sd.b->buf);
  e.buf(6, eh_); e.buf(7, eidx_); e.buf(8, ewts_); e.buf(9, x_);
  e.val(10, mp);
  e.dispatch((cfg_.hidden + rows * nsg - 1) / (rows * nsg), 1, 1, 32 * nsg);
}

void Qwen35::enc_mlp(Encoder& e, const Layer& L) {
  static Pipeline gu = Gpu::get().pipeline("mlp_gate_up");
  constexpr uint32_t rows = 2, nsg = 8;
  kp::Qmv p{L.mg.N, L.mg.K, L.mg.K, L.mg.N, cfg_.eps};
  e.use(gu);
  e.buf(0, L.mg.w->buf); e.buf(1, L.mg.s->buf); e.buf(2, L.mg.b->buf);
  e.buf(3, L.mu.w->buf); e.buf(4, L.mu.s->buf); e.buf(5, L.mu.b->buf);
  e.buf(6, x_); e.buf(7, L.post_norm->buf); e.buf(8, mh_);
  e.val(9, p);
  e.dispatch((L.mg.N + rows * nsg - 1) / (rows * nsg), 1, 1, 32 * nsg);
  enc_qmv(e, L.md, mh_, x_, &x_, nullptr);
}

void Qwen35::enc_mlp_prefill(Encoder& e, const Layer& L, int T, const Buffer& xbuf) {
  const int D = cfg_.hidden;
  enc_rows_half(e, xbuf, L.post_norm, xh_, T, D);
  enc_qmm(e, L.mg, xh_, hg_, T, false);
  enc_qmm(e, L.mu, xh_, hu_, T, false);
  uint32_t n = uint32_t(T) * L.mg.N;
  e.use(pipe("swiglu_half"));
  e.buf(0, hg_); e.buf(1, hu_); e.buf(2, hhalf_); e.val(3, n);
  e.dispatch((n + 255) / 256, 1, 1, 256);
  enc_qmm(e, L.md, hhalf_, xbuf, T, true);
}

void Qwen35::encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample) {
  static Pipeline emb = Gpu::get().pipeline("embed_q4");
  static Pipeline st1 = Gpu::get().pipeline("topk_stage1");
  static Pipeline st2 = Gpu::get().pipeline("sample_stage2");
  if (pos + 1 >= s.cap) fatal("context full (%d)", s.cap);
  kp::Embed ep{uint32_t(cfg_.hidden), uint32_t(embed_.gs), uint32_t(pos)};
  e.use(emb);
  e.buf(0, embed_.w->buf); e.buf(1, embed_.s->buf); e.buf(2, embed_.b->buf); e.buf(3, s.tok); e.buf(4, x_);
  e.val(5, ep);
  e.dispatch((cfg_.hidden / 8 + 63) / 64, 1, 1, 64);
  static int dbg_layers = getenv("NI_DEBUG_LAYERS") ? atoi(getenv("NI_DEBUG_LAYERS")) : -1;
  // profiling aid: NI_SKIP bitmask 1=moe 2=gdn 4=attn 8=head (output becomes garbage)
  static int skip = getenv("NI_SKIP") ? atoi(getenv("NI_SKIP")) : 0;
  for (int i = 0; i < cfg_.layers; ++i) {
    if (i == dbg_layers) return;
    const Layer& L = layers_[i];
    if (L.linear) {
      if (!(skip & 2)) enc_gdn(e, s, i, L);
    } else if (!(skip & 4)) {
      enc_attn(e, s, i, L, pos);
    }
    if (i + 1 == dbg_layers && getenv("NI_DEBUG_HALF")) return;
    if (skip & 1) continue;
    if (cfg_.moe) enc_moe(e, L);
    else enc_mlp(e, L);
    if (cap_) enc_capture(e, i, pos, 1);
  }
  if (!sample || (skip & 8)) return;
  enc_sample(e, s, x_, pos + 1, sp, step);
}

void Qwen35::enc_sample(Encoder& e, Seq& s, const Buffer& xrow, int out_index, const SamplingParams& sp, uint32_t step) {
  static Pipeline st1 = Gpu::get().pipeline("topk_stage1");
  static Pipeline st2 = Gpu::get().pipeline("sample_stage2");
  enc_qmv(e, lm_head_, xrow, logits_, nullptr, final_norm_);
  // stage 2 holds 4 candidates per thread (4096): 122 threadgroups x 32 fits
  uint32_t K = sp.temperature > 0 ? uint32_t(std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32)) : 1u;
  kp::TopkStage ts{uint32_t(cfg_.vocab), K, 2048};
  uint32_t ntg = (cfg_.vocab + 2047) / 2048;
  e.use(st1);
  e.buf(0, logits_); e.buf(1, cand_v_); e.buf(2, cand_i_);
  e.val(3, ts);
  e.dispatch(ntg, 1, 1, 256);
  kp::Sample sp2{ntg * K, K, sp.temperature, sp.top_p, sp.seed, step, uint32_t(out_index)};
  e.use(st2);
  e.buf(0, cand_v_); e.buf(1, cand_i_); e.buf(2, s.tok); e.buf(3, sprob_);
  e.val(4, sp2);
  e.dispatch(1, 1, 1, 1024);
}

// ---- prefill ----
static Pipeline pipe(const std::string& n) {
  static std::unordered_map<std::string, Pipeline> cache;
  auto it = cache.find(n);
  if (it == cache.end()) it = cache.emplace(n, Gpu::get().pipeline(n)).first;
  return it->second;
}

void Qwen35::enc_rows_half(Encoder& e, const Buffer& x, const Tensor* norm, const Buffer& y, int T, int D) {
  e.use(pipe("rows_to_half"));
  kp::Rows p{uint32_t(D), norm ? 1u : 0u, cfg_.eps};
  e.buf(0, x);
  e.buf(1, norm ? norm->buf : x);
  e.buf(2, y);
  e.val(3, p);
  e.dispatch(T, 1, 1, 256);
}

void Qwen35::enc_qmm(Encoder& e, const QLinear& L, const Buffer& xh, const Buffer& y, int M, bool add) {
  const bool big = M >= 64 && L.bits == 4;
  const int BM = big ? 64 : 32, BN = big ? 64 : 32;
  std::string n = "qmm_b" + std::to_string(L.bits) + (big ? "_m64_n64" : "_m32_n32");
  e.use(pipe(n));
  kp::Qmm p{uint32_t(M), L.N, L.K, L.K, L.N, add ? 1u : 0u, uint32_t(L.gs), 0, 0};
  e.buf(0, xh); e.buf(1, L.w->buf); e.buf(2, L.s->buf); e.buf(3, L.b->buf); e.buf(4, y);
  e.val(5, p);
  e.dispatch((L.N + BN - 1) / BN, (M + BM - 1) / BM, 1, 128);
}

void Qwen35::enc_qmm_moe(Encoder& e, const QLinear& L, const Buffer& xh, const Buffer& y, const Buffer& xrows,
                         const Buffer& yrows, int P, int bm) {
  std::string n = "qmm_moe_b4_m" + std::to_string(bm) + "_n64";
  e.use(pipe(n));
  // expert matrices are stacked [E, N, K]: strides in words / groups
  uint32_t wstride = L.N * (L.K * L.bits / 32), sstride = L.N * (L.K / L.gs);
  kp::Qmm p{uint32_t(P), L.N, L.K, L.K, L.N, 0, uint32_t(L.gs), wstride, sstride};
  e.buf(0, xh); e.buf(1, L.w->buf); e.buf(2, L.s->buf); e.buf(3, L.b->buf); e.buf(4, y);
  e.val(5, p);
  e.buf(6, tiles_); e.buf(7, ntiles_); e.buf(8, xrows); e.buf(9, yrows);
  uint32_t max_tiles = uint32_t((P + bm - 1) / bm + cfg_.experts);
  e.dispatch((L.N + 63) / 64, max_tiles, 1, 128);
}

void Qwen35::enc_moe_prefill(Encoder& e, const Layer& L, int T, const Buffer& xbuf) {
  if (moe_fused() && enc_moe_prefill_fused(e, L, T, xbuf)) return;
  const int D = cfg_.hidden, k = cfg_.top_k, P = T * k;
  enc_rows_half(e, xbuf, L.post_norm, xh_, T, D);
  enc_qmm(e, L.router, xh_, rlog_, T, false);
  enc_qmm(e, L.shared_gate, xh_, slog_, T, false);
  kp::Topk tp{uint32_t(cfg_.experts), uint32_t(k), cfg_.norm_topk ? 1u : 0u};
  e.use(pipe("moe_topk_rows"));
  e.buf(0, rlog_); e.buf(1, slog_); e.buf(2, ridx_); e.buf(3, rwts_);
  e.val(4, tp);
  e.dispatch(T, 1, 1, 32);
  const int per_expert = P / cfg_.experts;
  // 16-row tiles measured fastest for expert GEMMs at every chunk size (32-row tiles hit a
  // 2-3 TFLOP/s pathology on M4 Pro; 64-row tiles waste more rows in partial tiles).
  int bm = 16;
  (void)per_expert;
  if (getenv("NI_BM")) bm = atoi(getenv("NI_BM"));
  kp::Bucket bp{uint32_t(P), uint32_t(k), uint32_t(cfg_.experts), uint32_t(bm)};
  e.use(pipe("moe_bucket"));
  e.buf(0, ridx_); e.buf(1, spair_); e.buf(2, stok_); e.buf(3, tiles_); e.buf(4, ntiles_);
  e.val(5, bp);
  e.dispatch(1, 1, 1, 1024);
  enc_qmm_moe(e, L.eg, xh_, hg_, stok_, spair_, P, bm);
  enc_qmm_moe(e, L.eu, xh_, hu_, stok_, spair_, P, bm);
  uint32_t n = uint32_t(P * cfg_.moe_I);
  e.use(pipe("swiglu_half"));
  e.buf(0, hg_); e.buf(1, hu_); e.buf(2, hhalf_); e.val(3, n);
  e.dispatch((n + 255) / 256, 1, 1, 256);
  enc_qmm_moe(e, L.ed, hhalf_, yd_, spair_, spair_, P, bm);
  // shared expert on all rows
  enc_qmm(e, L.sg, xh_, sgb_, T, false);
  enc_qmm(e, L.su, xh_, sub_, T, false);
  uint32_t ns = uint32_t(T * cfg_.shared_I);
  e.use(pipe("swiglu_half"));
  e.buf(0, sgb_); e.buf(1, sub_); e.buf(2, hh_); e.val(3, ns);
  e.dispatch((ns + 255) / 256, 1, 1, 256);
  enc_qmm(e, L.sd, hh_, ys_, T, false);
  kp::Combine cp{uint32_t(D), uint32_t(k)};
  e.use(pipe("moe_combine"));
  e.buf(0, yd_); e.buf(1, ys_); e.buf(2, rwts_); e.buf(3, xbuf);
  e.val(4, cp);
  e.dispatch((D / 4 + 63) / 64, T, 1, 64);
}

// Opt-in MoE prefill (NI_MOE_FUSED): router and top-k as above, then
//   moe_bucket2 -> qmm_moe_gu (gate+up+SwiGLU -> half h, sorted rows) -> qmm_moe_dn (-> yd by pair)
// with the shared expert as expert index E over the T token rows (mode 1, when its shapes and
// quantization match the routed experts') or on the dense path (mode 2). Per row the arithmetic
// matches the original path (same dequantized halfs, same MMA order over K). Returns false when
// the layer does not fit the kernels' assumptions (the caller then runs the original path).
bool Qwen35::enc_moe_prefill_fused(Encoder& e, const Layer& L, int T, const Buffer& xbuf) {
  const int D = cfg_.hidden, k = cfg_.top_k, P = T * k, E = cfg_.experts;
  const QLinear &G = L.eg, &U = L.eu, &Dn = L.ed;
  // NI_MOE_BM: rows per tile (16, 32 or 64); partial tiles skip their empty 8-row fragments
  static const int bm = getenv("NI_MOE_BM") ? atoi(getenv("NI_MOE_BM")) : 64;
  constexpr int kBnGu = 32, kBnDn = 64;
  auto fits = [](const QLinear& q) { return q.bits == 4 && q.gs % 32 == 0 && q.K % 32 == 0; };
  // gate and up share K, strides and the xh_ row stride (D) in one dispatch
  if (!fits(G) || !fits(U) || !fits(Dn) || G.N % kBnGu || Dn.N % kBnDn || U.N != G.N || U.K != G.K ||
      U.gs != G.gs || G.K != uint32_t(D) || Dn.K != G.N || Dn.N != uint32_t(D) || (bm != 16 && bm != 32 && bm != 64)) {
    static bool warned = false;
    if (!warned) logi("NI_MOE_FUSED: layer shapes/quantization not supported, using the original MoE path");
    warned = true;
    return false;
  }
  const bool fold = moe_fused() == 1 && fits(L.sg) && fits(L.su) && fits(L.sd) && L.sg.N == G.N && L.su.N == G.N &&
                    L.sg.K == G.K && L.su.K == G.K && L.sd.K == Dn.K && L.sd.N == Dn.N && L.sg.gs == G.gs &&
                    L.su.gs == G.gs && L.sd.gs == Dn.gs;
  const int S = fold ? T : 0;
  enc_rows_half(e, xbuf, L.post_norm, xh_, T, D);
  enc_qmm(e, L.router, xh_, rlog_, T, false);
  enc_qmm(e, L.shared_gate, xh_, slog_, T, false);
  kp::Topk tp{uint32_t(E), uint32_t(k), cfg_.norm_topk ? 1u : 0u};
  e.use(pipe("moe_topk_rows"));
  e.buf(0, rlog_); e.buf(1, slog_); e.buf(2, ridx_); e.buf(3, rwts_);
  e.val(4, tp);
  e.dispatch(T, 1, 1, 32);
  kp::Bucket2 bp{uint32_t(P), uint32_t(k), uint32_t(E), uint32_t(bm), uint32_t(S)};
  e.use(pipe("moe_bucket2"));
  e.buf(0, ridx_); e.buf(1, spair_); e.buf(2, stok_); e.buf(3, tiles_); e.buf(4, ntiles_);
  e.val(5, bp);
  e.dispatch(1, 1, 1, 1024);
  const uint32_t max_tiles = uint32_t((P + bm - 1) / bm + E + (S + bm - 1) / bm);
  const std::string m = "_b4_m" + std::to_string(bm);
  // gate + up + SwiGLU: hhalf_ row r = sorted row r (pairs grouped by expert, then shared rows)
  {
    // expert matrices are stacked [E, N, K]: strides in words / groups
    kp::Qmm2 p{G.N, G.K, G.K, G.N, uint32_t(G.gs), G.N * (G.K / 8), G.N * (G.K / G.gs), uint32_t(E)};
    const QLinear &SG = fold ? L.sg : G, &SU = fold ? L.su : U;
    e.use(pipe("qmm_moe_gu" + m + "_n" + std::to_string(kBnGu)));
    e.buf(0, xh_);
    e.buf(1, G.w->buf); e.buf(2, G.s->buf); e.buf(3, G.b->buf);
    e.buf(4, U.w->buf); e.buf(5, U.s->buf); e.buf(6, U.b->buf);
    e.buf(7, hhalf_);
    e.val(8, p);
    e.buf(9, tiles_); e.buf(10, ntiles_); e.buf(11, stok_);
    e.buf(12, SG.w->buf); e.buf(13, SG.s->buf); e.buf(14, SG.b->buf);
    e.buf(15, SU.w->buf); e.buf(16, SU.s->buf); e.buf(17, SU.b->buf);
    e.dispatch(G.N / kBnGu, max_tiles, 1, 128);
  }
  // down: yd_ row = pair id (shared rows: P + token)
  {
    kp::Qmm2 p{Dn.N, Dn.K, Dn.K, Dn.N, uint32_t(Dn.gs), Dn.N * (Dn.K / 8), Dn.N * (Dn.K / Dn.gs), uint32_t(E)};
    const QLinear& SD = fold ? L.sd : Dn;
    e.use(pipe("qmm_moe_dn" + m + "_n" + std::to_string(kBnDn)));
    e.buf(0, hhalf_);
    e.buf(1, Dn.w->buf); e.buf(2, Dn.s->buf); e.buf(3, Dn.b->buf);
    e.buf(4, yd_);
    e.val(5, p);
    e.buf(6, tiles_); e.buf(7, ntiles_); e.buf(8, spair_);
    e.buf(9, SD.w->buf); e.buf(10, SD.s->buf); e.buf(11, SD.b->buf);
    e.dispatch(Dn.N / kBnDn, max_tiles, 1, 128);
  }
  if (!fold) {
    // shared expert on all rows (as in enc_moe_prefill)
    enc_qmm(e, L.sg, xh_, sgb_, T, false);
    enc_qmm(e, L.su, xh_, sub_, T, false);
    uint32_t ns = uint32_t(T * cfg_.shared_I);
    e.use(pipe("swiglu_half"));
    e.buf(0, sgb_); e.buf(1, sub_); e.buf(2, hh_); e.val(3, ns);
    e.dispatch((ns + 255) / 256, 1, 1, 256);
    enc_qmm(e, L.sd, hh_, ys_, T, false);
  }
  kp::Combine cp{uint32_t(D), uint32_t(k)};
  e.use(pipe("moe_combine"));
  e.buf(0, yd_);
  e.buf(1, fold ? yd_.view(size_t(P) * D * 4, size_t(T) * D * 4) : ys_);
  e.buf(2, rwts_); e.buf(3, xbuf);
  e.val(4, cp);
  e.dispatch((D / 4 + 63) / 64, T, 1, 64);
  return true;
}

// NI_GDN_PAR=1 (read once): token-parallel GDN prefill. gdn_prep_par runs the conv/norm with
// a simdgroup per 8 tokens x 128 channels instead of a T-long loop per channel; the recurrence
// spreads each head's value rows over simdgroups/threadgroups with no barriers (gdn_step_par_l*),
// and gdn_gnorm applies the gated RMSNorm, writing out_proj's half input directly (replaces
// rows_to_half). Same arithmetic order as the default kernels. NI_GDN_STEP picks the recurrence
// layout (gdn_step_par_l{4,8,16,32}: lanes per value row), NI_GDN_SGS its simdgroups per threadgroup.
// On by default (35B 4k prefill +5.5%); NI_GDN_PAR=0 runs the per-channel gdn_prep/gdn_step path.
static bool gdn_par() {
  static const bool on = !getenv("NI_GDN_PAR") || !*getenv("NI_GDN_PAR") || atoi(getenv("NI_GDN_PAR")) > 0;
  return on;
}

void Qwen35::enc_gdn_prefill_par(Encoder& e, Seq& s, int li, const Layer& L, int T) {
  static const std::string step_name = getenv("NI_GDN_STEP") ? getenv("NI_GDN_STEP") : "gdn_step_par_l16";
  static const uint32_t sgs = getenv("NI_GDN_SGS") ? uint32_t(atoi(getenv("NI_GDN_SGS"))) : 2u;
  static const uint32_t lanes = uint32_t(atoi(step_name.c_str() + step_name.rfind("_l") + 2));
  const uint32_t C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  const uint32_t Hv = uint32_t(cfg_.lin_hv);
  // rows per threadgroup = sgs * 32/lanes must split the 128 value rows of a head evenly
  const uint32_t rows_tg = lanes ? sgs * (32 / lanes) : 0;
  if (!rows_tg || 32 % lanes || 128 % rows_tg || sgs > 32)
    fatal("NI_GDN_STEP=%s with NI_GDN_SGS=%u: bad layout", step_name.c_str(), sgs);
  kp::GdnPrep pp{uint32_t(T), uint32_t(cfg_.lin_hk), Hv, C};
  e.use(pipe("gdn_prep_par"));
  e.buf(0, qkv_); e.buf(1, ab_); e.buf(2, bb_); e.buf(3, s.conv[li]);
  e.buf(4, L.conv_w->buf); e.buf(5, L.A_log->buf); e.buf(6, L.dt_bias->buf);
  e.buf(7, gq_); e.buf(8, gk_); e.buf(9, gv_); e.buf(10, gg_); e.buf(11, gbeta_);
  e.val(12, pp);
  e.dispatch((T + 31) / 32, C / 128 + 1, 1, 128);
  kp::GdnStep sp{uint32_t(T), uint32_t(cfg_.lin_hk), Hv, cfg_.eps};
  e.use(pipe(step_name));
  e.buf(0, gq_); e.buf(1, gk_); e.buf(2, gv_); e.buf(3, gg_); e.buf(4, gbeta_);
  e.buf(5, s.rec[li]); e.buf(6, gout_);
  e.val(7, sp);
  e.dispatch(Hv * (128 / rows_tg), 1, 1, 32 * sgs);
  e.use(pipe("gdn_gnorm"));
  e.buf(0, gout_); e.buf(1, zb_); e.buf(2, L.gnorm->buf); e.buf(3, hh_);
  e.val(4, sp);
  e.dispatch((uint32_t(T) * Hv + 7) / 8, 1, 1, 256);
}

// Causal attention of T new query rows (qo_, gate qg_) against the cache (kc, vc) -> attn_out_.
// attn_prefill11 (default): MLX-style layout, 32 queries of one query head per threadgroup,
// fragments in registers (2.1x v10 on the 35B shape, 3.1x on the 27B's G=6). NI_ATTN overrides.
void Qwen35::enc_attn_prefill(Encoder& e, const Buffer& kc, const Buffer& vc, int T, int pos0, int cap) {
  kp::AttnPref fp{uint32_t(T), uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(pos0), uint32_t(cap),
                  1.0f / std::sqrt(float(cfg_.head_dim))};
  static const std::string an = getenv("NI_ATTN") && *getenv("NI_ATTN") ? getenv("NI_ATTN") : "attn_prefill11";
  uint32_t qb = 4, th = 256, gy = cfg_.hkv;
  if (an.rfind("attn_prefill11", 0) == 0) { qb = 32; gy = cfg_.hq; }  // 32 queries of one q head
  else if (an == "attn_prefill") th = 128;
  else if (an == "attn_prefill2" || an == "attn_prefill3" || an == "attn_prefill4") qb = 8;
  else if (an.find("_q8") != std::string::npos) { qb = 8; th = 512; }
  e.use(pipe(an));
  e.buf(0, qo_); e.buf(1, kc); e.buf(2, vc); e.buf(3, qg_); e.buf(4, attn_out_);
  e.val(5, fp);
  e.dispatch((T + qb - 1) / qb, gy, 1, th);
}

void Qwen35::encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample) {
  if (T > tmax_) fatal("prefill chunk %d > %d", T, tmax_);
  if (pos0 + T >= s.cap) fatal("context full (%d)", s.cap);
  const int D = cfg_.hidden;
  kp::Embed ep{uint32_t(D), uint32_t(embed_.gs), uint32_t(pos0)};
  e.use(pipe("embed_q4"));
  e.buf(0, embed_.w->buf); e.buf(1, embed_.s->buf); e.buf(2, embed_.b->buf); e.buf(3, s.tok); e.buf(4, x_);
  e.val(5, ep);
  e.dispatch((D / 8 + 63) / 64, T, 1, 64);
  const uint32_t C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  const int VD = cfg_.lin_hv * cfg_.lin_dv;
  for (int i = 0; i < cfg_.layers; ++i) {
    const Layer& L = layers_[i];
    enc_rows_half(e, x_, L.in_norm, xh_, T, D);
    if (L.linear) {
      enc_qmm(e, L.qkv, xh_, qkv_, T, false);
      enc_qmm(e, L.z, xh_, zb_, T, false);
      enc_qmm(e, L.b, xh_, bb_, T, false);
      enc_qmm(e, L.a, xh_, ab_, T, false);
      if (gdn_par()) {
        enc_gdn_prefill_par(e, s, i, L, T);
      } else {
        kp::GdnPrep pp{uint32_t(T), uint32_t(cfg_.lin_hk), uint32_t(cfg_.lin_hv), C};
        e.use(pipe("gdn_prep"));
        e.buf(0, qkv_); e.buf(1, ab_); e.buf(2, bb_); e.buf(3, s.conv[i]);
        e.buf(4, L.conv_w->buf); e.buf(5, L.A_log->buf); e.buf(6, L.dt_bias->buf);
        e.buf(7, gq_); e.buf(8, gk_); e.buf(9, gv_); e.buf(10, gg_); e.buf(11, gbeta_);
        e.val(12, pp);
        e.dispatch(C / 128, 1, 1, 128);
        kp::GdnStep sp2{uint32_t(T), uint32_t(cfg_.lin_hk), uint32_t(cfg_.lin_hv), cfg_.eps};
        e.use(pipe("gdn_step"));
        e.buf(0, gq_); e.buf(1, gk_); e.buf(2, gv_); e.buf(3, gg_); e.buf(4, gbeta_);
        e.buf(5, s.rec[i]); e.buf(6, zb_); e.buf(7, L.gnorm->buf); e.buf(8, gout_);
        e.val(9, sp2);
        e.dispatch(cfg_.lin_hv, 1, 1, 1024);
        enc_rows_half(e, gout_, nullptr, hh_, T, VD);
      }
      enc_qmm(e, L.out, hh_, x_, T, true);
    } else {
      enc_qmm(e, L.q, xh_, qg_, T, false);
      enc_qmm(e, L.k, xh_, kr_, T, false);
      enc_qmm(e, L.v, xh_, vr_, T, false);
      kp::AttnPrep ap{uint32_t(T), uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(cfg_.head_dim),
                      uint32_t(cfg_.rot_dims), uint32_t(pos0), uint32_t(s.cap), cfg_.eps, cfg_.theta};
      e.use(pipe("attn_prep"));
      e.buf(0, qg_); e.buf(1, kr_); e.buf(2, vr_); e.buf(3, L.q_norm->buf); e.buf(4, L.k_norm->buf);
      e.buf(5, qo_); e.buf(6, s.kc[i]); e.buf(7, s.vc[i]);
      e.val(8, ap);
      e.dispatch(cfg_.hq + 2 * cfg_.hkv, T, 1, cfg_.head_dim);
      enc_attn_prefill(e, s.kc[i], s.vc[i], T, pos0, s.cap);
      enc_rows_half(e, attn_out_, nullptr, hh_, T, cfg_.hq * cfg_.head_dim);
      enc_qmm(e, L.o, hh_, x_, T, true);
    }
    if (cfg_.moe) enc_moe_prefill(e, L, T, x_);
    else enc_mlp_prefill(e, L, T, x_);
    if (cap_) enc_capture(e, i, pos0, T);
  }
  if (sample) enc_sample(e, s, x_.view(size_t(T - 1) * D * 4, size_t(D) * 4), pos0 + T, sp, step);
}

void Qwen35::enc_capture(Encoder& e, int layer, int pos0, int T) {
  if (!cap_ || layer >= int(cap_->slot.size()) || cap_->slot[layer] < 0) return;
  const int lo = std::max(pos0, cap_->pos_base), hi = std::min(pos0 + T, cap_->pos_base + cap_->rows);
  if (lo >= hi) return;
  const uint32_t D = uint32_t(cfg_.hidden);
  kp::DflCapture p{D, cap_->ld, uint32_t(cap_->slot[layer]) * D, uint32_t(lo - cap_->pos_base)};
  e.use(pipe("dfl_capture"));
  e.buf(0, x_, size_t(lo - pos0) * D * 4);
  e.buf(1, cap_->dst);
  e.val(2, p);
  e.dispatch((D / 4 + 255) / 256, uint32_t(hi - lo), 1, 256);
}

Buffer Qwen35::debug_buf(const std::string& n) {
  if (n == "rlog") return rlog_;
  if (n == "slog") return slog_;
  if (n == "eidx") return eidx_;
  if (n == "ewts") return ewts_;
  if (n == "eh") return eh_;
  if (n == "gout") return gout_;
  if (n == "qkv") return qkv_;
  if (n == "gq") return gq_;
  if (n == "gv") return gv_;
  if (n == "gg") return gg_;
  if (n == "gbeta") return gbeta_;
  if (n == "zb") return zb_;
  return x_;
}

} // namespace ni
