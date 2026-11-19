#include "model/hy4_dfly.h"

#include "model/hy4_kernels.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <sys/stat.h>

#include <cstring>
#include <fstream>

namespace ni {

using nlohmann::json;

namespace {

constexpr uint32_t kLat = 512, kRope = 64, kHead = 256, kKvbRows = 448;
const char* const kPre = "model.dfly_layers.";
const char* const kFile = "model-dfly.safetensors";

struct MvP {  // DflyMvP
  uint32_t N, K, R, x_ld, y_ld, r_ld, add;
};
struct QmvP {  // DflyQmvP
  uint32_t N, K;
};
struct EmbP {  // DflyEmbP
  uint32_t K, V, R, nids, fill;
};

Pipeline pipe(const char* n) { return hy4::pipeline(n); }

void check_q(const QLinear& L, const char* what) {
  if (!L || L.iq || L.q20 || L.exl3 || L.gs != 64 || (L.bits != 4 && L.bits != 8))
    fatal("hy4 dfly: the target's %s is %s; the drafter takes affine 4- or 8-bit group 64", what, L ? L.name().c_str() : "missing");
}

}  // namespace

Hy4DflyConfig Hy4DflyConfig::from_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) fatal("hy4 dfly: cannot read %s", path.c_str());
  const json c = json::parse(f, nullptr, false);
  if (c.is_discarded() || !c.is_object()) fatal("hy4 dfly: bad %s", path.c_str());
  const json& t = c.contains("text_config") ? c["text_config"] : c;
  Hy4DflyConfig k;
  if (!c.contains("dfly_target_kv_layers")) fatal("hy4 dfly: %s has no dfly_target_kv_layers", path.c_str());
  k.kv_layers = c["dfly_target_kv_layers"].get<std::vector<int>>();
  if (c.contains("dfly_target_index_layer") && c["dfly_target_index_layer"].get<std::vector<int>>() != k.kv_layers)
    fatal("hy4 dfly: dfly_target_index_layer differs from dfly_target_kv_layers");
  if (c.value("dfly_num_hidden_layers", int(k.kv_layers.size())) != int(k.kv_layers.size()))
    fatal("hy4 dfly: dfly_num_hidden_layers %d, %zu kv layers", c.value("dfly_num_hidden_layers", 0), k.kv_layers.size());
  k.mask_token = c.value("dfly_mask_token_id", 120009u);
  if (c.contains("dfly_context_local_window") && !c["dfly_context_local_window"].is_null() &&
      c["dfly_context_local_window"].get<int>() != 0)
    fatal("hy4 dfly: dfly_context_local_window %d is not supported", c["dfly_context_local_window"].get<int>());
  if (!c.value("dfly_enable_hidden_correction", true)) fatal("hy4 dfly: a drafter without the hidden correction");
  if (c.value("dfly_mtp_fusion", std::string("slot0")) != "slot0")
    fatal("hy4 dfly: dfly_mtp_fusion %s (only slot0)", c.value("dfly_mtp_fusion", std::string()).c_str());
  k.D = t.value("hidden_size", 3072u);
  k.H = t.value("num_attention_heads", 32u);
  k.eps = t.value("rms_norm_eps", 1e-5f);
  if (t.value("kv_lora_rank", 0) != int(kLat) || t.value("qk_nope_head_dim", 0) != 192 ||
      t.value("qk_rope_head_dim", 0) != int(kRope) || t.value("v_head_dim", 0) != int(kHead))
    fatal("hy4 dfly: unexpected attention dims in %s", path.c_str());
  if (t.contains("rope_parameters")) {
    const json& rp = t["rope_parameters"];
    const json& rs = rp.contains("deepseek_sparse_attention") ? rp["deepseek_sparse_attention"]
                     : rp.contains("indexed_attention")       ? rp["indexed_attention"]
                                                              : rp;
    k.theta = rs.value("rope_theta", 1e7f);
  }
  return k;
}

bool Hy4Dfly::present(const std::string& dir) {
  struct stat st;
  return stat((dir + "/" + kFile).c_str(), &st) == 0;
}

Hy4Dfly::Lin Hy4Dfly::lin(const std::string& name, uint32_t r0, uint32_t n) {
  const Tensor& t = ws_.get(kPre + name);
  if (t.dtype != DType::BF16 || t.shape.size() != 2) fatal("hy4 dfly: %s is not a 2-D bf16 matrix", name.c_str());
  Lin L;
  L.K = uint32_t(t.shape[1]);
  const uint32_t rows = uint32_t(t.shape[0]);
  if (!n) n = rows - r0;
  if (r0 + n > rows || L.K % 8) fatal("hy4 dfly: %s [%u, %u]: rows %u+%u or K", name.c_str(), rows, L.K, r0, n);
  L.N = n;
  L.w = t.buf.view(size_t(r0) * L.K * 2, size_t(n) * L.K * 2);
  if (L.w.offset % 16) fatal("hy4 dfly: %s is not 16-byte aligned", name.c_str());
  return L;
}

Hy4Dfly::Lin Hy4Dfly::cols(const std::string& name, uint32_t c0, uint32_t n) {
  const Lin src = lin(name);
  if (c0 + n > src.K || n % 8) fatal("hy4 dfly: %s columns %u+%u of %u", name.c_str(), c0, n, src.K);
  Lin L;
  L.N = src.N;
  L.K = n;
  L.w = Gpu::get().alloc(size_t(L.N) * n * 2, "dfly_cols", false);
  own_bufs_.push_back(L.w);
  weight_bytes_ += L.w.size;
  for (uint32_t r = 0; r < L.N; ++r)
    std::memcpy(L.w.as<uint16_t>() + size_t(r) * n, src.w.as<uint16_t>() + size_t(r) * src.K + c0, size_t(n) * 2);
  return L;
}

const Tensor* Hy4Dfly::vec(const std::string& name, uint32_t n) {
  const Tensor& t = ws_.get(kPre + name);
  if (t.numel() != int64_t(n)) fatal("hy4 dfly: %s has %lld elements, not %u", name.c_str(), (long long)t.numel(), n);
  return &t;
}

Hy4Dfly::Hy4Dfly(const std::string& dir, const Hy4DflyConfig& cfg, const QLinear& embed, const QLinear& lm_head)
    : cfg_(cfg), embed_(embed), lm_head_(lm_head) {
  check_q(embed, "embedding");
  check_q(lm_head, "LM head");
  const uint32_t D = cfg_.D, H = cfg_.H;
  if (embed.K != D || lm_head.K != D || embed.N != lm_head.N)
    fatal("hy4 dfly: embedding [%u, %u] / LM head [%u, %u] for hidden %u", embed.N, embed.K, lm_head.N, lm_head.K, D);
  V_ = lm_head.N;
  if (D % 64 || V_ % 4 || cfg_.mask_token >= V_ || cfg_.kv_layers.empty() || H % 16)
    fatal("hy4 dfly: hidden %u, vocab %u, mask %u, %zu layers, %u heads", D, V_, cfg_.mask_token, cfg_.kv_layers.size(), H);
  ws_.open_dir(
      dir,
      [](const std::string& n) {
        return n.rfind(kPre, 0) == 0 && n.find("confidence_head") == std::string::npos;
      },
      [](const std::string& f) { return f == kFile; });
  for (const Buffer& b : ws_.buffers()) weight_bytes_ += b.size;
  const int nl = int(cfg_.kv_layers.size());
  L_.resize(size_t(nl));
  for (int i = 0; i < nl; ++i) {
    Layer& l = L_[size_t(i)];
    const std::string p = std::to_string(i) + ".";
    l.in_norm = vec(p + "input_layernorm.weight", D);
    l.post_norm = vec(p + "post_attention_layernorm.weight", D);
    l.q_a = lin(p + "self_attn.q_a_proj.weight");
    l.q_a_norm = vec(p + "self_attn.q_a_layernorm.weight", l.q_a.N);
    l.q_b = lin(p + "self_attn.q_b_proj.weight");
    l.kv_lat = lin(p + "self_attn.kv_a_proj_with_mqa.weight", 0, kLat);
    l.kv_rope = lin(p + "self_attn.kv_a_proj_with_mqa.weight", kLat, kRope);
    l.kv_a_norm = vec(p + "self_attn.kv_a_layernorm.weight", kLat);
    l.kv_b = &ws_.get(kPre + p + "self_attn.kv_b_proj.weight");
    l.sinks = vec(p + "self_attn.learnable_sink_param", H);
    l.gate = lin(p + "self_attn.linear_gate.weight");
    l.o = lin(p + "self_attn.o_proj.weight");
    l.post_proj_norm = vec(p + "self_attn.post_proj_norm.weight", D);
    l.mg = lin(p + "mlp.gate_proj.weight");
    l.mu = lin(p + "mlp.up_proj.weight");
    l.md = lin(p + "mlp.down_proj.weight");
    const uint32_t I = l.mg.N;
    if (l.q_a.K != D || l.q_b.K != l.q_a.N || l.q_b.N != H * kHead || l.kv_lat.K != D ||
        l.kv_b->dtype != DType::BF16 || l.kv_b->shape != std::vector<int64_t>{int64_t(H * kKvbRows), int64_t(kLat)} ||
        l.gate.K != D || l.gate.N != H * kHead || l.o.K != H * kHead || l.o.N != D || l.mg.K != D || l.mu.N != I ||
        l.mu.K != D || l.md.K != I || l.md.N != D)
      fatal("hy4 dfly: layer %d's shapes do not fit hidden %u, %u heads", i, D, H);
    l.at.H = H;
    l.at.theta = cfg_.theta;
    l.at.window = 0;
    l.at.sinks = l.sinks;
    l.at.rope_bf16 = cfg_.rope_bf16;
  }
  enorm_ = vec("mtp_fusion.enorm.weight", D);
  hnorm_ = vec("mtp_fusion.hnorm.weight", D);
  fgate_ = vec("mtp_fusion.gate", D);
  if (fgate_->dtype != DType::BF16) fatal("hy4 dfly: mtp_fusion.gate is not bf16");
  eh_ = lin("mtp_fusion.eh_proj.weight");
  final_norm_ = vec("final_norm.weight", D);
  hc_hnorm_ = vec("hidden_correction.hidden_norm.weight", D);
  hc_enorm_ = vec("hidden_correction.embed_norm.weight", D);
  // the correction's gate / up see concat(hidden_norm(z), embed_norm(e)): their halves as separate matrices, so the
  // hidden half runs once for the 5 rows and each step reads only the embedding half
  hc_gh_ = cols("hidden_correction.gate_proj.weight", 0, D);
  hc_ge_ = cols("hidden_correction.gate_proj.weight", D, D);
  hc_uh_ = cols("hidden_correction.up_proj.weight", 0, D);
  hc_ue_ = cols("hidden_correction.up_proj.weight", D, D);
  hc_down_ = lin("hidden_correction.down_proj.weight");
  if (eh_.K != 2 * D || eh_.N != D || hc_gh_.N != D || hc_uh_.N != D || hc_down_.N != D || hc_down_.K != D)
    fatal("hy4 dfly: fusion / correction shapes do not fit hidden %u", D);

  Gpu& g = Gpu::get();
  const size_t R = kBlock, I = L_[0].mg.N, QA = L_[0].q_a.N, HD = size_t(H) * kHead;
  auto act = [&](size_t floats, const char* label) {
    Buffer b = g.alloc(floats * 4, label);
    std::memset(b.ptr(), 0, floats * 4);
    own_bufs_.push_back(b);
    return b;
  };
  X_ = act(R * D, "dfly_x");
  u_ = act(R * D, "dfly_u");
  qa_ = act(R * QA, "dfly_qa");
  q_ = act(R * HD, "dfly_q");
  lat_ = act(R * kLat, "dfly_lat");
  kr_ = act(R * kRope, "dfly_kr");
  gt_ = act(R * HD, "dfly_gate");
  att_ = act(R * HD, "dfly_att");
  y_ = act(R * D, "dfly_y");
  g_ = act(R * I, "dfly_g");
  up_ = act(R * I, "dfly_up");
  cat_ = act(2 * D, "dfly_cat");
  f_ = act(D, "dfly_f");
  hb_ = act(D, "dfly_hb");
  z_ = act(R * D, "dfly_z");
  hn_ = act(kDraft * D, "dfly_hn");
  Gh_ = act(kDraft * D, "dfly_gh");
  Uh_ = act(kDraft * D, "dfly_uh");
  e1_ = act(D, "dfly_e");
  en_ = act(D, "dfly_en");
  hg_ = act(D, "dfly_hg");
  hu_ = act(D, "dfly_hu");
  zc_ = act(D, "dfly_zc");
  logits_ = act(size_t(kDraft) * V_, "dfly_logits");
  const size_t sb = hy4::latent_scratch_bytes(H, kBlock, 2048 + kBlock);
  attn_scratch_ = g.alloc(std::max<size_t>(sb, 256), "dfly_attn_scratch");
  own_bufs_.push_back(attn_scratch_);
  logi("hy4 dfly: %d layers over target layers %s, MLP %zu, %.2f GB of bf16 weights", nl,
       [&] {
         std::string s;
         for (int v : cfg_.kv_layers) s += (s.empty() ? "" : "/") + std::to_string(v);
         return s;
       }().c_str(),
       I, weight_bytes_ / 1e9);
}

Hy4Dfly::~Hy4Dfly() = default;

std::vector<Buffer> Hy4Dfly::buffers() const {
  std::vector<Buffer> v = ws_.buffers();
  v.insert(v.end(), own_bufs_.begin(), own_bufs_.end());
  return v;
}

std::vector<std::string> Hy4Dfly::kernel_names() const {
  return {"dfly_mv_t1_r4", "dfly_mv_t6_r2", "dfly_mv_t6_r4", lm_head_.bits == 8 ? "dfly_qmv_b8" : "dfly_qmv_b4",
          embed_.bits == 8 ? "dfly_embed_b8" : "dfly_embed_b4", "dfly_argmax", "dfly_round_bf16", "dfly_gate_add"};
}

void Hy4Dfly::mv(Encoder& e, const Lin& L, const Buffer& x, int R, const Buffer& y, const Buffer* resid, uint32_t x_ld,
                 uint32_t y_ld, uint32_t r_ld) {
  if (R < 1 || R > kBlock) fatal("hy4 dfly: mv over %d rows", R);
  const MvP p{L.N, L.K, uint32_t(R), x_ld ? x_ld : L.K, y_ld ? y_ld : L.N, r_ld ? r_ld : L.N, resid ? 1u : 0u};
  const int rows = R == 1 ? 4 : L.N >= 4096 ? 4 : 2;
  e.use(pipe(R == 1 ? "dfly_mv_t1_r4" : rows == 4 ? "dfly_mv_t6_r4" : "dfly_mv_t6_r2"));
  e.buf(0, L.w);
  e.buf(1, x);
  e.buf(2, resid ? *resid : y);
  e.buf(3, y);
  e.val(4, p);
  const uint32_t per_tg = 8u * uint32_t(rows);
  e.dispatch((L.N + per_tg - 1) / per_tg, 1, 1, 256);
}

void Hy4Dfly::embed(Encoder& e, const Buffer& ids, uint32_t nids, int R, const Buffer& y) {
  const EmbP p{cfg_.D, V_, uint32_t(R), nids, cfg_.mask_token};
  const uint32_t words = cfg_.D / (32u / uint32_t(embed_.bits));
  e.use(pipe(embed_.bits == 8 ? "dfly_embed_b8" : "dfly_embed_b4"));
  e.buf(0, embed_.w->buf);
  e.buf(1, embed_.s->buf);
  e.buf(2, embed_.b->buf);
  e.buf(3, ids);
  e.buf(4, y);
  e.val(5, p);
  e.dispatch((words + 63) / 64, uint32_t(R), 1, 64);
}

void Hy4Dfly::head(Encoder& e, const Buffer& x, const Buffer& y) {
  const QmvP p{V_, cfg_.D};
  e.use(pipe(lm_head_.bits == 8 ? "dfly_qmv_b8" : "dfly_qmv_b4"));
  e.buf(0, lm_head_.w->buf);
  e.buf(1, lm_head_.s->buf);
  e.buf(2, lm_head_.b->buf);
  e.buf(3, x);
  e.buf(4, y);
  e.val(5, p);
  e.dispatch((V_ + 31) / 32, 1, 1, 256);
}

void Hy4Dfly::draft(Encoder& e, const In& in, const Buffer& out) {
  const uint32_t D = cfg_.D, H = cfg_.H, HD = H * kHead;
  const int R = kBlock;
  const float eps = cfg_.eps;
  if (in.caches.size() != L_.size()) fatal("hy4 dfly: %zu caches for %zu layers", in.caches.size(), L_.size());
  if (in.a < 0) fatal("hy4 dfly: position %d", in.a);
  for (const hy4::LatentCache* c : in.caches)
    if (!c || c->ring || c->cap < uint32_t(in.a + R))
      fatal("hy4 dfly: a target cache is %s (cap %u) for a block at %d", !c ? "missing" : c->ring ? "a ring" : "short",
            c ? c->cap : 0u, in.a);
  const size_t rowb = size_t(D) * 4;
  auto row = [&](const Buffer& b, int r) { return b.view(size_t(r) * rowb, rowb); };

  // the block [t0, M, M, M, M, M] and the slot-0 fusion with the target hidden (held in bf16 between cycles)
  embed(e, in.t0, 1, R, X_);
  {
    const uint32_t n = D;
    e.use(pipe("dfly_round_bf16"));
    e.buf(0, in.h);
    e.buf(1, hb_);
    e.val(2, n);
    e.dispatch((n + 255) / 256, 1, 1, 256);
  }
  hy4::enc_rmsnorm(e, D, 1, row(X_, 0), *enorm_, eps, cat_.view(0, rowb));
  hy4::enc_rmsnorm(e, D, 1, hb_, *hnorm_, eps, cat_.view(rowb, rowb));
  mv(e, eh_, cat_, 1, f_);
  {
    const uint32_t n = D;
    e.use(pipe("dfly_gate_add"));
    e.buf(0, X_);
    e.buf(1, fgate_->buf);
    e.buf(2, f_);
    e.val(3, n);
    e.dispatch((n + 255) / 256, 1, 1, 256);
  }

  for (size_t i = 0; i < L_.size(); ++i) {
    const Layer& l = L_[i];
    const uint32_t I = l.mg.N;
    hy4::enc_rmsnorm(e, D, R, X_, *l.in_norm, eps, u_);
    mv(e, l.q_a, u_, R, qa_);
    hy4::enc_rmsnorm(e, l.q_a.N, R, qa_, *l.q_a_norm, eps, qa_);
    mv(e, l.q_b, qa_, R, q_);
    mv(e, l.kv_lat, u_, R, lat_);
    hy4::enc_rmsnorm(e, kLat, R, lat_, *l.kv_a_norm, eps, lat_);
    mv(e, l.kv_rope, u_, R, kr_);
    mv(e, l.gate, u_, R, gt_);
    hy4::enc_attn_latent(e, l.at, *in.caches[i], in.a, R, q_, HD, lat_, kLat, kr_, kRope, 0, *l.kv_b, gt_, att_, false,
                         attn_scratch_, in.sel.empty() ? nullptr : in.sel[i], !in.sel.empty());
    mv(e, l.o, att_, R, y_);
    hy4::enc_rmsnorm(e, D, R, y_, *l.post_proj_norm, eps, y_);
    hy4::enc_add(e, D, R, y_, X_);
    hy4::enc_rmsnorm(e, D, R, X_, *l.post_norm, eps, u_);
    mv(e, l.mg, u_, R, g_);
    mv(e, l.mu, u_, R, up_);
    hy4::enc_swiglu(e, I, R, g_, up_, g_, 0.0f, false);
    mv(e, l.md, g_, R, X_, &X_);
  }
  hy4::enc_rmsnorm(e, D, R, X_, *final_norm_, eps, z_);

  // the head: zc_k = z_k + down(silu(gate g) * up g), g = [hidden_norm(z_k) | embed_norm(e(prev))]; prev = t0, d1, ..
  const Buffer z5 = z_.view(rowb, size_t(kDraft) * rowb);
  hy4::enc_rmsnorm(e, D, kDraft, z5, *hc_hnorm_, eps, hn_);
  mv(e, hc_gh_, hn_, kDraft, Gh_);
  mv(e, hc_uh_, hn_, kDraft, Uh_);
  for (int k = 1; k <= kDraft; ++k) {
    const Buffer prev = k == 1 ? in.t0 : out.view(size_t(k - 2) * 4, 4);
    embed(e, prev, 1, 1, e1_);
    hy4::enc_rmsnorm(e, D, 1, e1_, *hc_enorm_, eps, en_);
    const Buffer gh = row(Gh_, k - 1), uh = row(Uh_, k - 1), zk = row(z_, k);
    mv(e, hc_ge_, en_, 1, hg_, &gh);
    mv(e, hc_ue_, en_, 1, hu_, &uh);
    hy4::enc_swiglu(e, D, 1, hg_, hu_, hg_, 0.0f, false);
    mv(e, hc_down_, hg_, 1, zc_, &zk);
    const Buffer lg = logits_.view(size_t(k - 1) * V_ * 4, size_t(V_) * 4);
    head(e, zc_, lg);
    e.use(pipe("dfly_argmax"));
    e.buf(0, lg);
    e.buf(1, out.view(size_t(k - 1) * 4, 4));
    e.val(2, V_);
    e.dispatch(1, 1, 1, 1024);
  }
}

}  // namespace ni
