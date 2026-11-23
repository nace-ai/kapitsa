// Inkling-2's image encoder (see ink2_vit.h).
#include "vision/ink2_vit.h"
#include "util/bf16.h"
#include "util/log.h"
#include "vision/qwen_vit.h"  // pil_resize_bicubic: images over the cap

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ni::vision {

using json = nlohmann::json;

namespace {
constexpr const char* kPrefix = "model.visual.";  // the tower's tensors in an Inkling-2 checkpoint

struct IvitPatchParams {
  uint32_t W, H, gw, P, merge, K, K_pad;
  float pad;
};
struct QvitEmbedParams {  // vit.metal's qvit_embed (the same interpolated table, our taps)
  uint32_t N, D, gw, side, merge;
};
struct IvitRopeParams {
  uint32_t rows, r0, N, heads, HD, gw, merge;
};
struct IvitScaleParams {
  uint32_t rows, D, y_ld;
};
}  // namespace

// ---- configuration ----

bool Ink2ViT::supports(const std::string& dir, Ink2ViTConfig* c) {
  const json root = vit_read_json(dir + "/config.json");
  if (!root.contains("vision_config") || !root["vision_config"].is_object()) return false;
  const json& v = root["vision_config"];
  if (v.value("model_type", std::string()) != "inkling2_vision") return false;
  Ink2ViTConfig r;
  r.depth = v.value("num_hidden_layers", r.depth);
  r.hidden = v.value("hidden_size", r.hidden);
  r.heads = v.value("num_attention_heads", r.heads);
  r.inter = v.contains("intermediate_size") && v["intermediate_size"].is_number_integer()
                ? v["intermediate_size"].get<int>()
                : int(std::lround(r.hidden * v.value("mlp_ratio", 4.0)));
  r.patch = v.value("native_patch_size", v.value("patch_size", r.patch));
  r.merge = v.value("adapter_merge_size", v.value("spatial_merge_size", r.merge));
  r.in_ch = v.value("num_channels", r.in_ch);
  r.side = v.value("posemb_grid_size", r.side);
  r.proj_hidden = v.value("projection_hidden_dim", r.proj_hidden);
  r.proj_bias = v.value("projector_bias", r.proj_bias);
  r.ln_eps = v.value("layer_norm_eps", r.ln_eps);
  const json t = root.value("text_config", json::object());
  r.out_hidden = v.value("text_hidden_size", t.value("hidden_size", r.out_hidden));
  r.image_token = root.value("image_token_id", r.image_token);
  if (v.contains("rope_parameters") && v["rope_parameters"].is_object())
    r.theta = v["rope_parameters"].value("rope_theta", r.theta);
  else
    r.theta = v.value("rope_theta", r.theta);
  const int hd = r.heads > 0 ? r.hidden / r.heads : 0;
  auto why_not = [&]() -> std::string {
    if (v.value("hidden_act", std::string("gelu")) != "gelu") return "an activation other than GELU";
    if (v.contains("rope_parameters") && v["rope_parameters"].is_object() &&
        v["rope_parameters"].value("rope_type", std::string("axial")) != "axial")
      return "a RoPE other than axial";
    if (r.in_ch != 3) return "channels other than RGB";
    if (r.heads <= 0 || r.hidden % r.heads || !vit_attention_has(hd) || hd % 4) return "an unexpected head dimension";
    if (r.merge < 1 || r.hidden % 64 || r.inter % 64 || r.proj_hidden % 64 || r.out_hidden % 64 || r.side < 2)
      return "an unexpected shape";
    return "";
  };
  if (const std::string w = why_not(); !w.empty()) {
    logi("vision: %s has an Inkling-2 vision tower with %s, which Kapitsa does not run: images are off", dir.c_str(),
         w.c_str());
    return false;
  }
  bool tower = false, proj = false;
  for (const std::string& n : vit_tensor_names(dir)) {
    tower = tower || n == std::string(kPrefix) + "backbone.embeddings.patch_embedding.weight";
    proj = proj || n == std::string(kPrefix) + "proj_out.weight";
  }
  if (!tower || !proj) return false;  // a text-only conversion of the model (or a GGUF import)
  // the processor: processor_config.json's image_processor (or preprocessor_config.json)
  json pp = vit_read_json(dir + "/preprocessor_config.json");
  const json pc = vit_read_json(dir + "/processor_config.json");
  if (pc.contains("image_processor") && pc["image_processor"].is_object())
    for (auto& [k, val] : pc["image_processor"].items()) pp[k] = val;
  const json mean = pp.value("image_mean", json()), stdv = pp.value("image_std", json());
  if (mean.is_array() && mean.size() == 3 && stdv.is_array() && stdv.size() == 3)
    for (int i = 0; i < 3; ++i) r.mean[i] = mean[i].get<float>(), r.stdv[i] = stdv[i].get<float>();
  r.rescale = pp.value("rescale_factor", r.rescale);
  r.do_rescale = pp.value("do_rescale", true);
  r.do_normalize = pp.value("do_normalize", true);
  if (pp.contains("size") && pp["size"].is_object()) {
    const int side = r.patch * r.merge;
    if (pp["size"].value("height", side) != side || pp["size"].value("width", side) != side) {
      logi("vision: %s: the image processor's token size differs from the vision tower's %d pixels: images are off",
           dir.c_str(), side);
      return false;
    }
  }
  if (c) *c = r;
  return true;
}

Ink2ViT::Ink2ViT(const std::string& dir, const Ink2ViTConfig& c) : dir_(dir), c_(c) {
  patch_k_ = (uint32_t(3 * c_.patch * c_.patch) + 63) / 64 * 64;
}

Ink2ViT::~Ink2ViT() { unload(); }

std::string Ink2ViT::describe() const {
  return name() + " (" + std::to_string(c_.depth) + " blocks, width " + std::to_string(c_.hidden) + ", " +
         std::to_string(c_.patch) + "-pixel patches, " + std::to_string(c_.patch * c_.merge) +
         "-pixel tokens at the image's own resolution" +
         (c_.max_pixels > 0 ? ", images over " + std::to_string(c_.max_pixels / 1000) + "K pixels scaled down" : "") +
         ")";
}

// ---- the processor (CPU) ----

Prepared Ink2ViT::prepare(const Image& im) const {
  const int F = c_.patch * c_.merge;
  int w = im.width, h = im.height;
  Prepared p;
  if (c_.max_pixels > 0 && long(w) * h > c_.max_pixels) {
    // Kapitsa's cap (the reference processor has none): scaled down to about max_pixels, aspect kept, to whole tokens
    const double s = std::sqrt(double(c_.max_pixels) / (double(w) * h));
    const int ow = std::max(F, int(std::floor(w * s / F)) * F), oh = std::max(F, int(std::floor(h * s / F)) * F);
    p.pixels = pil_resize_bicubic(im.rgb, w, h, ow, oh);
    w = ow;
    h = oh;
  } else {
    p.pixels = PixelBuffer(im.rgb.begin(), im.rgb.end());
  }
  p.width = w;
  p.height = h;
  p.grid_t = 1;
  p.grid_h = (h + F - 1) / F;
  p.grid_w = (w + F - 1) / F;
  p.tokens = p.grid_h * p.grid_w;
  uint8_t key[32 + 8 + 8];
  memcpy(key, im.digest.data(), 32);
  const int32_t dims[2] = {h, w};
  memcpy(key + 32, dims, 8);
  memcpy(key + 40, "ink2-vit", 8);
  p.digest = sha256(key, sizeof key);
  return p;
}

// value v of channel c after rescale and normalize, as the reference processor computes it: transformers' torchvision
// backend fuses the two into one normalize with mean and std scaled by 1 / rescale_factor (float32 steps); [256]: the
// padding's -1 on the 0-255 scale
void Ink2ViT::build_lut(float* lut) const {
  const float rs = float(c_.rescale), inv = float(1.0 / c_.rescale);
  for (int ch = 0; ch < 3; ++ch) {
    const bool fused = c_.do_rescale && c_.do_normalize;
    const float mean = fused ? c_.mean[ch] * inv : c_.mean[ch], stdv = fused ? c_.stdv[ch] * inv : c_.stdv[ch];
    for (int v = 0; v <= 256; ++v) {
      float x = v == 256 ? -1.0f : float(v);
      if (c_.do_rescale && !fused) x = x * rs;
      if (c_.do_normalize) {
        x = x - mean;
        x = x / stdv;
      }
      lut[ch * 257 + v] = x;
    }
  }
}

std::vector<float> Ink2ViT::pixel_values(const Prepared& p) const {
  float lut[3 * 257];
  build_lut(lut);
  const int F = c_.patch * c_.merge, cols = grid_cols(p);
  std::vector<float> out(size_t(p.tokens) * F * F * 3);
  for (int t = 0; t < p.tokens; ++t) {
    const int ty = t / cols, tx = t % cols;
    float* o = out.data() + size_t(t) * F * F * 3;
    for (int y = 0; y < F; ++y)
      for (int x = 0; x < F; ++x) {
        const int px = tx * F + x, py = ty * F + y;
        const bool in = px < p.width && py < p.height;
        for (int ch = 0; ch < 3; ++ch)
          o[(y * F + x) * 3 + ch] = lut[ch * 257 + (in ? p.pixels[(size_t(py) * p.width + px) * 3 + ch] : 256)];
      }
  }
  return out;
}

// ---- weights ----

size_t Ink2ViT::weight_bytes() const {
  const size_t H = size_t(c_.hidden), I = size_t(c_.inter), PH = size_t(c_.proj_hidden), O = size_t(c_.out_hidden);
  const size_t HM = H * c_.merge * c_.merge;
  const size_t block = 6 * H + (3 * H * H + 3 * H) + (H * H + H) + (I * H + I) + (H * I + H);
  const size_t embed = H * patch_k_ + H + size_t(c_.side) * c_.side * H + 2 * H;
  const size_t merger = PH * HM + PH + O * PH + O;
  return 2 * (block * c_.depth + embed + merger);
}

void Ink2ViT::load() {
  if (loaded_) return;
  w_.open(dir_, kPrefix);
  const uint32_t H = uint32_t(c_.hidden);
  patch_ = w_.lin("backbone.embeddings.patch_embedding", 0, patch_k_, /*bias=*/false);
  if (patch_.N != H || patch_.K != patch_k_) fatal("vision: Inkling-2 patch embedding of an unexpected shape");
  pos_ = w_.vec("backbone.embeddings.position_embedding");
  if (const Tensor* t = w_.find("backbone.embeddings.position_embedding");
      !t || t->numel() != int64_t(c_.side) * c_.side * c_.hidden)
    fatal("vision: Inkling-2 position table of an unexpected size");
  {  // the class token's row: class_embedding + cls_position_embedding, in fp32
    std::vector<uint16_t> a(H), b(H);
    w_.read_bf16("backbone.embeddings.class_embedding", a.data(), 1, H, H);
    w_.read_bf16("backbone.embeddings.cls_position_embedding", b.data(), 1, H, H);
    cls_.resize(H);
    for (uint32_t i = 0; i < H; ++i) cls_[i] = bf16_to_f32(a[i]) + bf16_to_f32(b[i]);
  }
  lnw_ = w_.vec("backbone.ln_pre.weight");
  lnb_ = w_.vec("backbone.ln_pre.bias");
  blocks_.resize(size_t(c_.depth));
  for (int i = 0; i < c_.depth; ++i) {
    Block& B = blocks_[size_t(i)];
    const std::string b = "backbone.layers." + std::to_string(i) + ".";
    B.n1w = w_.vec(b + "ln_1.weight");
    B.n1b = w_.vec(b + "ln_1.bias");
    B.n2w = w_.vec(b + "ln_2.weight");
    B.n2b = w_.vec(b + "ln_2.bias");
    B.g1 = w_.vec(b + "ls_1.gamma");
    B.g2 = w_.vec(b + "ls_2.gamma");
    // q, k, v as one linear [3 * H, H]: rows [0, H) q, [H, 2H) k, [2H, 3H) v
    B.qkv.N = 3 * H;
    B.qkv.K = H;
    B.qkv.w = w_.alloc(size_t(3) * H * H * 2, "vit_w");
    B.qkv.b = w_.alloc(size_t(3) * H * 2, "vit_b");
    const char* names[3] = {"q_proj", "k_proj", "v_proj"};
    for (int j = 0; j < 3; ++j) {
      const std::string n = b + "self_attn." + names[j];
      if (const Tensor* t = w_.find(n + ".weight"); !t || t->shape.size() != 2 || t->shape[0] != int64_t(H) || t->shape[1] != int64_t(H))
        fatal("vision: %s%s.weight missing or not [%u, %u]", kPrefix, n.c_str(), H, H);
      w_.read_bf16(n + ".weight", B.qkv.w.as<uint16_t>() + size_t(j) * H * H, H, H, H);
      w_.read_bf16(n + ".bias", B.qkv.b.as<uint16_t>() + size_t(j) * H, 1, H, H);
    }
    B.proj = w_.lin(b + "self_attn.out_proj");
    B.fc1 = w_.lin(b + "mlp.c_fc");
    B.fc2 = w_.lin(b + "mlp.c_proj");
    if (B.proj.N != H || B.proj.K != H || B.fc1.N != uint32_t(c_.inter) || B.fc1.K != H ||
        B.fc2.N != H || B.fc2.K != uint32_t(c_.inter))
      fatal("vision: Inkling-2 block %d of an unexpected shape", i);
  }
  m1_ = w_.lin("proj_in", 0, 0, c_.proj_bias);
  m2_ = w_.lin("proj_out", 0, 0, c_.proj_bias);
  if (m1_.K != H * uint32_t(c_.merge * c_.merge) || m1_.N != uint32_t(c_.proj_hidden) ||
      m2_.K != uint32_t(c_.proj_hidden) || m2_.N != uint32_t(c_.out_hidden))
    fatal("vision: Inkling-2 merger of an unexpected shape");
  w_.close();
  w_.make_resident();
  loaded_ = true;
}

void Ink2ViT::unload() {
  w_.release();
  blocks_.clear();
  cls_.clear();
  loaded_ = false;
}

// ---- the tower on the GPU ----

size_t Ink2ViT::work_bytes(const Prepared& p) const {
  const size_t H = size_t(c_.hidden), N = size_t(patches(p)), NP = N + 1;
  const size_t rc = std::min<size_t>(NP, kVitRowChunk);
  const size_t ym = std::max<size_t>({3 * H, size_t(c_.inter), size_t(c_.proj_hidden)});
  const size_t hm = std::max<size_t>(size_t(c_.inter), size_t(c_.proj_hidden));
  return pixel_buffer_bytes(p.pixels.size()) + N * patch_k_ * 2 /* patches */ + NP * H * 4 /* x */ +
         NP * H * 2 /* xn */ + 3 * NP * H * 2 /* q, k, v */ + rc * ym * 4 /* y */ + rc * hm * 2 /* h */ +
         64 * 1024 /* tables */;
}

void Ink2ViT::encode(const Prepared& p, const Buffer& out) {
  if (!loaded_) load();
  if (p.pixels.size() != size_t(p.width) * p.height * 3) fatal("vision: encode without pixels");
  Gpu& g = Gpu::get();
  const int P = c_.patch, mg = c_.merge, F = P * mg;
  const int rows = grid_rows(p), cols = grid_cols(p), gh = rows * mg, gw = cols * mg, N = gh * gw, NP = N + 1;
  const int M = rows * cols;  // tokens
  const int H = c_.hidden, NH = c_.heads, HD = H / NH, I = c_.inter, HM = H * mg * mg, PH = c_.proj_hidden,
            O = c_.out_hidden, K0 = 3 * P * P, KP = int(patch_k_);
  const int RC = std::min(NP, kVitRowChunk), MC = std::min(M, RC);
  const int YM = std::max({3 * H, I, PH}), HW = std::max(I, PH);
  std::vector<Buffer> scratch;
  auto alloc = [&](size_t bytes, const char* label) {
    scratch.push_back(g.alloc(bytes + 16, label));
    return scratch.back();
  };
  // prepare() wrote page-aligned CPU storage that outlives cs.finish(); ivit_patchify only reads it
  Buffer pix = g.wrap(const_cast<uint8_t*>(p.pixels.data()), pixel_buffer_bytes(p.pixels.size()), "vit_pixels", true);
  scratch.push_back(pix);
  Buffer lut = alloc(3 * 257 * 4, "vit_lut");
  build_lut(lut.as<float>());
  // position-embedding taps per grid row / column: (floor, floor + 1 clamped, their weights), the reference's bilinear
  // resampling with align_corners false and border padding: source index (i + 0.5) * side / n - 0.5 in float32
  auto axis = [&](int n, const char* label) {
    Buffer b = alloc(size_t(n) * 16, label);
    float* f = b.as<float>();
    for (int i = 0; i < n; ++i) {
      float src = float(i) + 0.5f;
      src = src * float(c_.side);
      src = src / float(n);
      src = src - 0.5f;
      const float fl = std::floor(src), frac = src - fl;
      f[4 * i] = float(std::clamp(int(fl), 0, c_.side - 1));
      f[4 * i + 1] = float(std::clamp(int(fl) + 1, 0, c_.side - 1));
      f[4 * i + 2] = std::max(1.0f - std::fabs(frac), 0.0f);
      f[4 * i + 3] = std::max(1.0f - std::fabs(frac - 1.0f), 0.0f);
    }
    return b;
  };
  Buffer rowinfo = axis(gh, "vit_rows"), colinfo = axis(gw, "vit_cols");
  Buffer invf = alloc(size_t(HD / 4) * 4, "vit_invfreq");
  for (int j = 0; j < HD / 4; ++j)  // the reference's float32 1 / theta^(2j / (HD/2))
    invf.as<float>()[j] = 1.0f / std::pow(c_.theta, float(2 * j) / float(HD / 2));
  Buffer patches = alloc(size_t(N) * KP * 2, "vit_patches");
  Buffer x = alloc(size_t(NP) * H * 4, "vit_x");
  Buffer xn = alloc(size_t(NP) * H * 2, "vit_xn");
  Buffer qkv = alloc(size_t(3) * NP * H * 2, "vit_qkv");
  Buffer y = alloc(size_t(RC) * YM * 4, "vit_y");
  Buffer hh = alloc(size_t(RC) * HW * 2, "vit_h");
  const Buffer Qb = qkv.view(0, size_t(NP) * H * 2), Kb = qkv.view(size_t(NP) * H * 2, size_t(NP) * H * 2),
               Vb = qkv.view(size_t(2) * NP * H * 2, size_t(NP) * H * 2);
  const Buffer xp = x.view(size_t(H) * 4, size_t(N) * H * 4);  // the patches' rows (after the class token's)
  memcpy(x.ptr(), cls_.data(), size_t(H) * 4);                  // row 0: the class token
  VitCmds cs;
  auto scale_add = [&](Encoder& e, int r0, int rc, const Buffer& bias, const Buffer& gamma) {
    const IvitScaleParams sp{uint32_t(rc), uint32_t(H), uint32_t(H)};
    e.use(vit_pipe("ivit_scale_add"));
    e.buf(0, x.view(size_t(r0) * H * 4, size_t(rc) * H * 4));
    e.buf(1, y);
    e.buf(2, bias);
    e.buf(3, gamma);
    e.val(4, sp);
    e.dispatch((uint32_t(H) + 255) / 256, uint32_t(rc), 1, 256);
  };
  // patches -> patch embedding + the interpolated position embedding into x's rows 1.., then ln_pre over every row
  {
    Encoder& e = cs.at(2.0 * N * KP * H / 1e9);
    const IvitPatchParams pp{uint32_t(p.width), uint32_t(p.height), uint32_t(gw), uint32_t(P), uint32_t(mg),
                             uint32_t(K0), uint32_t(KP), lut.as<float>()[256]};
    e.use(vit_pipe("ivit_patchify"));
    e.buf(0, pix);
    e.buf(1, lut);
    e.buf(2, patches);
    e.val(3, pp);
    e.dispatch((uint32_t(KP) + 255) / 256, uint32_t(N), 1, 256);
    vit_gemm(e, patch_, patches, uint32_t(KP), xp, uint32_t(H), N, false);
    const QvitEmbedParams ep{uint32_t(N), uint32_t(H), uint32_t(gw), uint32_t(c_.side), uint32_t(mg)};
    e.use(vit_pipe("qvit_embed"));
    e.buf(0, xp);
    e.buf(1, patch_.b);  // zeros: no bias
    e.buf(2, pos_);
    e.buf(3, rowinfo);
    e.buf(4, colinfo);
    e.val(5, ep);
    e.dispatch((uint32_t(H) + 255) / 256, uint32_t(N), 1, 256);
    vit_ln_inplace(e, x, uint32_t(H), nullptr, lnw_, lnb_, NP, H, c_.ln_eps);
  }
  const int stop = dbg_stop_ >= 0 ? std::min(dbg_stop_, c_.depth) : c_.depth;
  for (int L = 0; L < stop; ++L) {
    const Block& B = blocks_[size_t(L)];
    vit_ln(cs.at(0.01 * NP * H / 1e6), x, uint32_t(H), nullptr, B.n1w, B.n1b, xn, uint32_t(H), NP, H, c_.ln_eps);
    for (int r0 = 0; r0 < NP; r0 += RC) {
      const int rc = std::min(RC, NP - r0);
      Encoder& e = cs.at(2.0 * rc * H * 3 * H / 1e9);
      vit_gemm(e, B.qkv, xn.view(size_t(r0) * H * 2, size_t(rc) * H * 2), uint32_t(H), y, uint32_t(3 * H), rc, false);
      const IvitRopeParams rp{uint32_t(rc), uint32_t(r0), uint32_t(NP), uint32_t(NH), uint32_t(HD), uint32_t(gw),
                              uint32_t(mg)};
      e.use(vit_pipe("ivit_qkv_rope"));
      e.buf(0, y);
      e.buf(1, B.qkv.b);
      e.buf(2, invf);
      e.buf(3, Qb);
      e.buf(4, Kb);
      e.buf(5, Vb);
      e.val(6, rp);
      const uint32_t nthreads = uint32_t(rc) * NH * (HD / 2);
      e.dispatch((nthreads + 255) / 256, 1, 1, 256);
    }
    vit_attention(cs, Qb, Kb, Vb, xn, NP, H, NH, HD);
    for (int r0 = 0; r0 < NP; r0 += RC) {  // x += ls_1 * (out_proj(attention) + bias)
      const int rc = std::min(RC, NP - r0);
      Encoder& e = cs.at(2.0 * rc * H * H / 1e9);
      vit_gemm(e, B.proj, xn.view(size_t(r0) * H * 2, size_t(rc) * H * 2), uint32_t(H), y, uint32_t(H), rc, false);
      scale_add(e, r0, rc, B.proj.b, B.g1);
    }
    vit_ln(cs.at(0.01 * NP * H / 1e6), x, uint32_t(H), nullptr, B.n2w, B.n2b, xn, uint32_t(H), NP, H, c_.ln_eps);
    for (int r0 = 0; r0 < NP; r0 += RC) {  // x += ls_2 * (c_proj(GELU(c_fc(xn))) + bias)
      const int rc = std::min(RC, NP - r0);
      Encoder& e = cs.at(4.0 * rc * H * I / 1e9);
      vit_gemm(e, B.fc1, xn.view(size_t(r0) * H * 2, size_t(rc) * H * 2), uint32_t(H), y, uint32_t(I), rc, false);
      vit_act(e, "vit_bias_act_h", y, uint32_t(I), B.fc1.b, hh, uint32_t(I), rc, I, 2);
      vit_gemm(e, B.fc2, hh, uint32_t(I), y, uint32_t(H), rc, false);
      scale_add(e, r0, rc, B.fc2.b, B.g2);
    }
  }
  if (dbg_stop_ >= 0) {  // tests: every row's hidden state after `stop` blocks, fp32
    Encoder& e = cs.at(0.01);
    vit_act(e, "vit_bias_f32", x, uint32_t(H), patch_.b, out, uint32_t(H), NP, H, 0u);
    cs.finish();
    for (Buffer& b : scratch) g.release(b);
    return;
  }
  // the merger: a token's merge x merge patches side by side (the patch rows as half, 4 * H per token), proj_in, GELU,
  // proj_out -> out (bf16)
  vit_act(cs.at(0.01 * N * H / 1e6), "vit_bias_act_h", xp, uint32_t(H), patch_.b, xn, uint32_t(H), N, H, 0u);
  for (int m0 = 0; m0 < M; m0 += MC) {
    const int mc = std::min(MC, M - m0);
    Encoder& e = cs.at(2.0 * mc * (double(HM) * PH + double(PH) * O) / 1e9);
    vit_gemm(e, m1_, xn.view(size_t(m0) * HM * 2, size_t(mc) * HM * 2), uint32_t(HM), y, uint32_t(PH), mc, false);
    vit_act(e, "vit_bias_act_h", y, uint32_t(PH), m1_.b, hh, uint32_t(PH), mc, PH, 2);
    vit_gemm(e, m2_, hh, uint32_t(PH), y, uint32_t(O), mc, false);
    vit_act(e, "vit_bias_act_b", y, uint32_t(O), m2_.b, out.view(size_t(m0) * O * 2, size_t(mc) * O * 2), uint32_t(O),
            mc, O, 0);
  }
  cs.finish();
  for (Buffer& b : scratch) g.release(b);
}

}  // namespace ni::vision
