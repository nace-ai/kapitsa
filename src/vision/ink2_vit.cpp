// Thinking Machines Inkling-2 Vision Encoder (see ink2_vit.h).
#include "vision/ink2_vit.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"
#include "vision/qwen_vit.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using json = nlohmann::json;

namespace ni::vision {

namespace {
struct Ink2VitPatchParams {
  uint32_t src_w, src_h, cols, patch, merge, K0, K_pad;
};
struct Ink2VitEmbedParams {
  uint32_t NP, D, ph, pw, cols, side, merge;
};
struct Ink2VitRopeParams {
  uint32_t rows, r0, NP, heads, HD, cols, merge;
};
struct VitLsResParams {
  uint32_t rows, D;
};

constexpr const char* kInk2VisPrefix = "model.visual.";
}  // namespace

bool Ink2ViT::supports(const std::string& dir, Ink2ViTConfig* c) {
  const json j = vit_read_json(dir + "/config.json");
  if (!j.contains("vision_config") || !j["vision_config"].is_object()) return false;
  const json& v = j["vision_config"];
  if (v.value("model_type", "") != "inkling2_vision") return false;

  Ink2ViTConfig cfg;
  cfg.depth = v.value("num_hidden_layers", cfg.depth);
  cfg.hidden = v.value("hidden_size", cfg.hidden);
  cfg.heads = v.value("num_attention_heads", cfg.heads);
  if (v.contains("intermediate_size") && v["intermediate_size"].is_number_integer()) {
    cfg.inter = v["intermediate_size"].get<int>();
  } else {
    cfg.inter = int(std::llround(double(cfg.hidden) * v.value("mlp_ratio", 4.0)));
  }
  cfg.patch = v.value("native_patch_size", v.value("patch_size", cfg.patch));
  cfg.merge = v.value("adapter_merge_size", v.value("spatial_merge_size", cfg.merge));
  cfg.side = v.value("posemb_grid_size", cfg.side);
  cfg.proj_hidden = v.value("projection_hidden_dim", cfg.proj_hidden);
  cfg.proj_bias = v.value("projector_bias", cfg.proj_bias);
  cfg.ln_eps = v.value("layer_norm_eps", cfg.ln_eps);
  if (v.contains("rope_parameters") && v["rope_parameters"].is_object())
    cfg.rope_theta = v["rope_parameters"].value("rope_theta", cfg.rope_theta);
  else if (v.contains("rope_theta"))
    cfg.rope_theta = v.value("rope_theta", cfg.rope_theta);

  if (v.contains("text_hidden_size")) cfg.out_hidden = v.value("text_hidden_size", cfg.out_hidden);
  else if (j.contains("text_config") && j["text_config"].is_object())
    cfg.out_hidden = j["text_config"].value("hidden_size", cfg.out_hidden);

  if (j.contains("image_token_id") && j["image_token_id"].is_number_integer())
    cfg.image_token = j["image_token_id"].get<int>();

  if (cfg.hidden <= 0 || cfg.heads <= 0 || cfg.hidden % cfg.heads != 0) return false;
  if (!vit_attention_has(cfg.hidden / cfg.heads)) return false;

  const json pc = vit_read_json(dir + "/processor_config.json");
  if (pc.is_object()) {
    const json& ip = pc.contains("image_processor") && pc["image_processor"].is_object() ? pc["image_processor"] : pc;
    if (ip.contains("rescale_factor") && ip["rescale_factor"].is_number())
      cfg.rescale = ip["rescale_factor"].get<double>();
    if (ip.contains("pad_value") && ip["pad_value"].is_number())
      cfg.pad_value = ip["pad_value"].get<double>();
    if (ip.contains("image_mean") && ip["image_mean"].is_array() && ip["image_mean"].size() == 3)
      for (int i = 0; i < 3; ++i) cfg.mean[i] = ip["image_mean"][size_t(i)].get<float>();
    if (ip.contains("image_std") && ip["image_std"].is_array() && ip["image_std"].size() == 3)
      for (int i = 0; i < 3; ++i) cfg.stdv[i] = ip["image_std"][size_t(i)].get<float>();
  }

  bool has_patch = false;
  for (const std::string& n : vit_tensor_names(dir)) {
    if (n == std::string(kInk2VisPrefix) + "backbone.embeddings.patch_embedding.weight") {
      has_patch = true;
      break;
    }
  }
  if (!has_patch) return false;
  if (c) *c = cfg;
  return true;
}

Ink2ViT::Ink2ViT(const std::string& dir, const Ink2ViTConfig& c) : dir_(dir), c_(c) {
  const uint32_t k0 = uint32_t(c_.patch * c_.patch * 3);
  patch_k_pad_ = ((k0 + 31u) / 32u) * 32u;
  inter_pad_ = ((uint32_t(c_.inter) + 63u) / 64u) * 64u;
  proj_pad_ = ((uint32_t(c_.proj_hidden) + 63u) / 64u) * 64u;
}

Ink2ViT::~Ink2ViT() { unload(); }

std::string Ink2ViT::describe() const {
  char buf[192];
  if (c_.max_pixels > 0) {
    snprintf(buf, sizeof buf, "%s (%dx%d, patch %d, merge %dx%d, up to %.1f MP)", name().c_str(), c_.depth,
             c_.hidden, c_.patch, c_.merge, c_.merge, c_.max_pixels / 1e6);
  } else {
    snprintf(buf, sizeof buf, "%s (%dx%d, patch %d, merge %dx%d, native size)", name().c_str(), c_.depth, c_.hidden,
             c_.patch, c_.merge, c_.merge);
  }
  return buf;
}

Prepared Ink2ViT::prepare(const Image& im) const {
  if (im.width <= 0 || im.height <= 0 || im.rgb.size() != size_t(im.width) * im.height * 3)
    throw Error("invalid image dimensions");
  int ow = im.width, oh = im.height;
  PixelBuffer px;
  if (c_.max_pixels > 0 && double(im.width) * double(im.height) > double(c_.max_pixels)) {
    const double scale = std::sqrt(double(c_.max_pixels) / (double(im.width) * double(im.height)));
    ow = std::max(1, int(std::floor(double(im.width) * scale)));
    oh = std::max(1, int(std::floor(double(im.height) * scale)));
    px = pil_resize_bicubic(im.rgb, im.width, im.height, ow, oh);
  } else {
    px.resize(im.rgb.size());
    memcpy(px.data(), im.rgb.data(), im.rgb.size());
  }
  const int tile = c_.patch * c_.merge;
  const int rows = (oh + tile - 1) / tile;
  const int cols = (ow + tile - 1) / tile;
  Prepared p;
  p.width = ow;
  p.height = oh;
  p.grid_t = 1;
  p.grid_h = rows;
  p.grid_w = cols;
  p.tokens = rows * cols;
  p.pixels = std::move(px);

  uint8_t key[32 + 16];
  memcpy(key, im.digest.data(), 32);
  const int32_t dims[2] = {oh, ow};
  memcpy(key + 32, dims, 8);
  memcpy(key + 40, "ink2vit1", 8);
  p.digest = sha256(key, sizeof key);
  return p;
}

std::vector<float> Ink2ViT::pixel_values(const Prepared& p) const {
  const int tile = c_.patch * c_.merge;
  const int rows = p.grid_h, cols = p.grid_w;
  if (p.pixels.size() != size_t(p.width) * p.height * 3) fatal("vision: pixel_values without pixels");
  std::vector<float> out(size_t(p.tokens) * tile * tile * 3);
  const uint8_t* src = p.pixels.data();
  for (int tr = 0; tr < rows; ++tr) {
    for (int tc = 0; tc < cols; ++tc) {
      float* dst = out.data() + size_t(tr * cols + tc) * tile * tile * 3;
      for (int ty = 0; ty < tile; ++ty) {
        const int gy = tr * tile + ty;
        for (int tx = 0; tx < tile; ++tx) {
          const int gx = tc * tile + tx;
          const bool in = gy < p.height && gx < p.width;
          const uint8_t* sp = in ? src + (size_t(gy) * p.width + gx) * 3 : nullptr;
          float* dp = dst + (size_t(ty) * tile + tx) * 3;
          for (int ch = 0; ch < 3; ++ch) {
            const double v = in ? double(sp[ch]) : c_.pad_value;
            dp[ch] = float((v * c_.rescale - double(c_.mean[ch])) / double(c_.stdv[ch]));
          }
        }
      }
    }
  }
  return out;
}

size_t Ink2ViT::weight_bytes() const {
  const size_t H = size_t(c_.hidden), I = size_t(inter_pad_), H4 = H * c_.merge * c_.merge;
  const size_t PH = size_t(proj_pad_), O = size_t(c_.out_hidden);
  const size_t emb = H * patch_k_pad_ + size_t(c_.side) * c_.side * H + 4 * H;
  const size_t blk = 6 * H + (3 * H * H + 3 * H) + (H * H + H) + (I * H + I) + (H * I + H);
  const size_t proj = (PH * H4 + PH) + (O * PH + O) + H4;
  return 2 * (emb + size_t(c_.depth) * blk + proj);
}

void Ink2ViT::load() {
  if (loaded_) return;
  const uint32_t H = uint32_t(c_.hidden);
  w_.open(dir_, kInk2VisPrefix);

  patch_ = w_.lin("backbone.embeddings.patch_embedding", 0, patch_k_pad_, /*bias=*/false);
  pos_ = w_.vec("backbone.embeddings.position_embedding");
  cls_ = w_.vec("backbone.embeddings.class_embedding");
  cls_pos_ = w_.vec("backbone.embeddings.cls_position_embedding");
  ln_pre_w_ = w_.vec("backbone.ln_pre.weight");
  ln_pre_b_ = w_.vec("backbone.ln_pre.bias");

  if (const Tensor* t = w_.find("backbone.embeddings.position_embedding");
      !t || t->numel() != int64_t(c_.side) * c_.side * c_.hidden)
    fatal("vision: Inkling-2 position_embedding has unexpected size");

  blocks_.resize(size_t(c_.depth));
  for (int l = 0; l < c_.depth; ++l) {
    Block& B = blocks_[size_t(l)];
    const std::string pre = "backbone.layers." + std::to_string(l) + ".";
    B.n1w = w_.vec(pre + "ln_1.weight");
    B.n1b = w_.vec(pre + "ln_1.bias");
    B.n2w = w_.vec(pre + "ln_2.weight");
    B.n2b = w_.vec(pre + "ln_2.bias");
    B.ls1 = w_.vec(pre + "ls_1.gamma");
    B.ls2 = w_.vec(pre + "ls_2.gamma");

    B.qkv.N = 3u * H;
    B.qkv.K = H;
    B.qkv.w = w_.alloc(size_t(3u * H) * H * 2, "vit_w");
    B.qkv.b = w_.alloc(size_t(3u * H) * 2, "vit_b");
    uint16_t* qw = B.qkv.w.as<uint16_t>();
    uint16_t* qb = B.qkv.b.as<uint16_t>();
    w_.read_bf16(pre + "self_attn.q_proj.weight", qw, H, H, H);
    w_.read_bf16(pre + "self_attn.k_proj.weight", qw + size_t(H) * H, H, H, H);
    w_.read_bf16(pre + "self_attn.v_proj.weight", qw + size_t(2u * H) * H, H, H, H);
    w_.read_bf16(pre + "self_attn.q_proj.bias", qb, 1, H, H);
    w_.read_bf16(pre + "self_attn.k_proj.bias", qb + H, 1, H, H);
    w_.read_bf16(pre + "self_attn.v_proj.bias", qb + 2u * H, 1, H, H);

    B.proj = w_.lin(pre + "self_attn.out_proj", 0, 0, true);
    B.fc1 = w_.lin(pre + "mlp.c_fc", inter_pad_, 0, true);
    B.fc2 = w_.lin(pre + "mlp.c_proj", 0, inter_pad_, true);
  }
  m1_ = w_.lin("proj_in", proj_pad_, 0, c_.proj_bias);
  m2_ = w_.lin("proj_out", 0, proj_pad_, c_.proj_bias);
  zero_h4_ = w_.alloc(size_t(c_.hidden) * c_.merge * c_.merge * 2, "vit_z");

  w_.close();
  w_.make_resident();
  loaded_ = true;
}

void Ink2ViT::unload() {
  w_.release();
  blocks_.clear();
  patch_ = VitLin{};
  m1_ = VitLin{};
  m2_ = VitLin{};
  pos_ = cls_ = cls_pos_ = ln_pre_w_ = ln_pre_b_ = zero_h4_ = Buffer{};
  loaded_ = false;
}

size_t Ink2ViT::work_bytes(const Prepared& p) const {
  const int mm = c_.merge * c_.merge;
  const size_t N = size_t(p.tokens) * mm;
  const size_t NP = 1 + N;
  const size_t H = size_t(c_.hidden), I = size_t(inter_pad_), PH = size_t(proj_pad_), O = size_t(c_.out_hidden);
  const size_t RC = std::min<size_t>(NP, kVitRowChunk);
  const size_t ym = std::max({size_t(3) * H, I, PH, O});
  const size_t hm = std::max({I, size_t(mm) * H, PH});
  return pixel_buffer_bytes(p.pixels.size()) + N * patch_k_pad_ * 2 /* pix */ + NP * H * 4 /* x */ +
         NP * H * 2 /* xn */ + 3 * NP * H * 2 /* qkv */ + RC * ym * 4 /* y */ + RC * hm * 2 /* hh */ +
         64 * 1024 /* lut + invfreq */;
}

void Ink2ViT::encode(const Prepared& p, const Buffer& out) {
  if (!loaded_) load();
  if (p.pixels.size() != size_t(p.width) * p.height * 3) fatal("vision: encode without pixels");
  const int mg = c_.merge, mm = mg * mg;
  const int cols = p.grid_w, rows = p.grid_h, pw = cols * mg, ph = rows * mg;
  const int N = p.tokens * mm, NP = 1 + N, M = p.tokens;
  const int H = c_.hidden, NH = c_.heads, HD = H / NH, I = int(inter_pad_), H4 = H * mm;
  const int PH = int(proj_pad_), O = c_.out_hidden;
  const uint32_t K0 = uint32_t(c_.patch * c_.patch * 3);
  const int RC = std::min(NP, kVitRowChunk);
  const int YM = std::max({3 * H, I, PH, O});
  const int HM = std::max({I, H4, PH});

  Gpu& g = Gpu::get();
  std::vector<Buffer> scratch;
  auto alloc = [&](size_t bytes, const char* label) {
    scratch.push_back(g.alloc(bytes + 16, label));
    return scratch.back();
  };

  Buffer rgb = g.wrap(const_cast<uint8_t*>(p.pixels.data()), pixel_buffer_bytes(p.pixels.size()), "vit_pixels", true);
  scratch.push_back(rgb);

  Buffer lut = alloc(3 * 257 * 4, "ink2_vit_lut");
  {
    float* lf = lut.as<float>();
    for (int ch = 0; ch < 3; ++ch) {
      for (int v = 0; v < 256; ++v)
        lf[ch * 257 + v] = float((double(v) * c_.rescale - double(c_.mean[ch])) / double(c_.stdv[ch]));
      lf[ch * 257 + 256] = float((c_.pad_value * c_.rescale - double(c_.mean[ch])) / double(c_.stdv[ch]));
    }
  }
  Buffer invf = alloc(size_t(HD / 4) * 4, "ink2_vit_invfreq");
  for (int j = 0; j < HD / 4; ++j)
    invf.as<float>()[j] = 1.0f / std::pow(c_.rope_theta, float(2 * j) / float(HD / 2));

  Buffer pix = alloc(size_t(N) * patch_k_pad_ * 2, "ink2_vit_pix");
  Buffer x = alloc(size_t(NP) * H * 4, "ink2_vit_x");
  Buffer xn = alloc(size_t(NP) * H * 2, "ink2_vit_xn");
  Buffer qkv = alloc(size_t(3) * NP * H * 2, "ink2_vit_qkv");
  Buffer y = alloc(size_t(RC) * YM * 4, "ink2_vit_y");
  Buffer hh = alloc(size_t(RC) * HM * 2, "ink2_vit_h");
  const Buffer Qb = qkv.view(0, size_t(NP) * H * 2);
  const Buffer Kb = qkv.view(size_t(NP) * H * 2, size_t(NP) * H * 2);
  const Buffer Vb = qkv.view(size_t(2) * NP * H * 2, size_t(NP) * H * 2);

  VitCmds cs;
  // Step 1: patchify + patch_embedding + CLS + interpolated pos_embed + pre_layernorm
  {
    Encoder& e = cs.at(2.0 * N * patch_k_pad_ * H / 1e9);
    const Ink2VitPatchParams pp{uint32_t(p.width), uint32_t(p.height), uint32_t(cols),
                                uint32_t(c_.patch), uint32_t(mg), K0, patch_k_pad_};
    e.use(vit_pipe("ink2_vit_patchify"));
    e.buf(0, rgb);
    e.buf(1, lut);
    e.buf(2, pix);
    e.val(3, pp);
    e.dispatch((patch_k_pad_ + 255u) / 256u, uint32_t(N), 1, 256);

    vit_gemm(e, patch_, pix, patch_k_pad_, x.view(size_t(H) * 4, size_t(N) * H * 4), uint32_t(H), N, false);

    const Ink2VitEmbedParams ep{uint32_t(NP), uint32_t(H), uint32_t(ph), uint32_t(pw),
                                uint32_t(cols), uint32_t(c_.side), uint32_t(mg)};
    e.use(vit_pipe("ink2_vit_embed"));
    e.buf(0, x);
    e.buf(1, pos_);
    e.buf(2, cls_);
    e.buf(3, cls_pos_);
    e.val(4, ep);
    e.dispatch((uint32_t(H) + 255u) / 256u, uint32_t(NP), 1, 256);

    vit_ln_inplace(e, x, uint32_t(H), nullptr, ln_pre_w_, ln_pre_b_, NP, H, c_.ln_eps);
  }

  // Step 2: 23 ViT encoder layers with post-branch LayerScale
  const int stop = dbg_stop_ >= 0 ? std::min(dbg_stop_, c_.depth) : c_.depth;
  for (int l = 0; l < stop; ++l) {
    const Block& B = blocks_[size_t(l)];
    vit_ln(cs.at(0.01 * NP * H / 1e6), x, uint32_t(H), nullptr, B.n1w, B.n1b, xn, uint32_t(H), NP, H, c_.ln_eps);
    for (int r0 = 0; r0 < NP; r0 += RC) {
      const int rc = std::min(RC, NP - r0);
      Encoder& e = cs.at(2.0 * rc * H * 3 * H / 1e9);
      vit_gemm(e, B.qkv, xn.view(size_t(r0) * H * 2, size_t(rc) * H * 2), uint32_t(H), y, uint32_t(3 * H), rc, false);
      const Ink2VitRopeParams rp{uint32_t(rc), uint32_t(r0), uint32_t(NP), uint32_t(NH),
                                 uint32_t(HD), uint32_t(cols), uint32_t(mg)};
      e.use(vit_pipe("ink2_vit_qkv_rope"));
      e.buf(0, y);
      e.buf(1, B.qkv.b);
      e.buf(2, invf);
      e.buf(3, Qb);
      e.buf(4, Kb);
      e.buf(5, Vb);
      e.val(6, rp);
      const uint32_t nthreads = uint32_t(rc) * uint32_t(NH) * uint32_t(HD / 2);
      e.dispatch((nthreads + 255u) / 256u, 1, 1, 256);
    }
    vit_attention(cs, Qb, Kb, Vb, xn, NP, H, NH, HD);
    for (int r0 = 0; r0 < NP; r0 += RC) {
      const int rc = std::min(RC, NP - r0);
      Encoder& e = cs.at(2.0 * rc * H * H / 1e9);
      vit_gemm(e, B.proj, xn.view(size_t(r0) * H * 2, size_t(rc) * H * 2), uint32_t(H), y, uint32_t(H), rc, false);
      const VitLsResParams lp{uint32_t(rc), uint32_t(H)};
      e.use(vit_pipe("vit_layerscale_res"));
      e.buf(0, x.view(size_t(r0) * H * 4, size_t(rc) * H * 4));
      e.buf(1, y);
      e.buf(2, B.proj.b);
      e.buf(3, B.ls1);
      e.val(4, lp);
      e.dispatch((uint32_t(H) + 255u) / 256u, uint32_t(rc), 1, 256);
    }
    vit_ln(cs.at(0.01 * NP * H / 1e6), x, uint32_t(H), nullptr, B.n2w, B.n2b, xn, uint32_t(H), NP, H, c_.ln_eps);
    for (int r0 = 0; r0 < NP; r0 += RC) {
      const int rc = std::min(RC, NP - r0);
      Encoder& e = cs.at(4.0 * rc * H * I / 1e9);
      vit_gemm(e, B.fc1, xn.view(size_t(r0) * H * 2, size_t(rc) * H * 2), uint32_t(H), y, uint32_t(I), rc, false);
      vit_act(e, "vit_bias_act_h", y, uint32_t(I), B.fc1.b, hh, uint32_t(I), rc, c_.inter, 2);
      vit_gemm(e, B.fc2, hh, uint32_t(I), y, uint32_t(H), rc, false);
      const VitLsResParams lp{uint32_t(rc), uint32_t(H)};
      e.use(vit_pipe("vit_layerscale_res"));
      e.buf(0, x.view(size_t(r0) * H * 4, size_t(rc) * H * 4));
      e.buf(1, y);
      e.buf(2, B.fc2.b);
      e.buf(3, B.ls2);
      e.val(4, lp);
      e.dispatch((uint32_t(H) + 255u) / 256u, uint32_t(rc), 1, 256);
    }
  }

  if (dbg_stop_ >= 0) {
    Encoder& e = cs.at(0.01);
    vit_act(e, "vit_bias_f32", x, uint32_t(H), zero_h4_, out, uint32_t(H), NP, H, 0);
    cs.finish();
    for (Buffer& b : scratch) g.release(b);
    return;
  }

  // Step 3: Drop CLS (row 0); patches [1..N] are already in 2x2 merge-block order -> [M, H4] projector
  const int MC = std::max(1, std::min(M, RC / mm));
  for (int m0 = 0; m0 < M; m0 += MC) {
    const int mc = std::min(MC, M - m0);
    Encoder& e = cs.at(2.0 * mc * (double(H4) * PH + double(PH) * O) / 1e9);
    const Buffer x_slice = x.view(size_t(1 + size_t(m0) * mm) * H * 4, size_t(mc) * H4 * 4);
    vit_act(e, "vit_bias_act_h", x_slice, uint32_t(H4), zero_h4_, hh, uint32_t(H4), mc, H4, 0);
    vit_gemm(e, m1_, hh, uint32_t(H4), y, uint32_t(PH), mc, false);
    vit_act(e, "vit_bias_act_h", y, uint32_t(PH), m1_.b, hh, uint32_t(PH), mc, c_.proj_hidden, 2);
    vit_gemm(e, m2_, hh, uint32_t(PH), y, uint32_t(O), mc, false);
    vit_act(e, "vit_bias_act_b", y, uint32_t(O), m2_.b, out.view(size_t(m0) * O * 2, size_t(mc) * O * 2),
            uint32_t(O), mc, O, 0);
  }
  cs.finish();
  for (Buffer& b : scratch) g.release(b);
}

}  // namespace ni::vision
