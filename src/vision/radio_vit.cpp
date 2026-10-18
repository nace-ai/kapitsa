// Nemotron 3.5 Bolt's image encoder (see radio_vit.h).
#include "vision/radio_vit.h"
#include "util/bf16.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>

namespace ni::vision {

using json = nlohmann::json;

namespace {
constexpr const char* kTower = "vision_model.radio_model.model.";  // the tower's tensors (timm names)
constexpr const char* kProj = "mlp1.";                              // the projector's: 0 RMSNorm, 1 and 3 linears

// rows [0, n) in parallel
template <class F>
void parallel_rows(int n, F f) {
  const int nt = std::max(1, std::min<int>(int(std::thread::hardware_concurrency()), 8));
  if (n < 64 || nt == 1) return f(0, n);
  std::vector<std::thread> th;
  for (int t = 0; t < nt; ++t) th.emplace_back([&, t] { f(n * t / nt, n * (t + 1) / nt); });
  for (auto& t : th) t.join();
}
}  // namespace

// ---- configuration ----

bool RadioViT::supports(const std::string& dir, RadioViTConfig* c) {
  const json root = vit_read_json(dir + "/config.json");
  if (!root.contains("vision_config") || !root["vision_config"].is_object()) return false;
  const json& v = root["vision_config"];
  if (v.value("model_type", std::string()) != "radio") return false;
  RadioViTConfig r;
  r.depth = v.value("num_hidden_layers", r.depth);
  r.hidden = v.value("hidden_size", r.hidden);
  r.heads = v.value("num_attention_heads", r.heads);
  r.inter = int(std::lround(r.hidden * v.value("mlp_ratio", 4.0)));
  r.patch = v.value("patch_size", r.patch);
  r.prefix = v.value("num_cls_tokens", 3) + v.value("num_registers", 7);
  r.side = v.value("max_img_size", 2048) / r.patch;
  r.ln_eps = v.value("layer_norm_eps", 1e-6f);
  r.proj_hidden = root.value("projector_hidden_size", r.proj_hidden);
  r.image_token = root.value("img_context_token_id", r.image_token);
  const json t = root.contains("llm_config") ? root["llm_config"] : root.value("text_config", json::object());
  r.out_hidden = t.value("hidden_size", r.out_hidden);
  auto why_not = [&]() -> std::string {
    if (v.value("use_swiglu_ffn", false)) return "a SwiGLU MLP";
    if (v.value("hidden_act", std::string("gelu")) != "gelu") return "an activation other than GELU";
    if (!v.value("qkv_bias", true)) return "no q / k / v bias";
    if (r.hidden % r.heads || r.hidden / r.heads != 80 || r.hidden % 64 || r.inter % 64) return "an unexpected shape";
    if (std::fabs(root.value("downsample_ratio", 0.5) - 0.5) > 1e-9) return "a downsample ratio other than 0.5";
    if (root.value("ps_version", std::string("v2")) != "v2") return "pixel shuffle v1";
    if (root.value("vision_final_layernorm", false)) return "a final LayerNorm on the tower";
    if (root.value("image_tag_type", std::string("internvl")) != "internvl") return "another image tag type";
    return "";
  };
  if (const std::string w = why_not(); !w.empty()) {
    logi("vision: %s has a RADIO tower with %s, which Kapitsa does not run: images are off", dir.c_str(), w.c_str());
    return false;
  }
  bool tower = false, proj = false;
  for (const std::string& n : vit_tensor_names(dir)) {
    tower = tower || n == std::string(kTower) + "patch_generator.embedder.weight";
    proj = proj || n == std::string(kProj) + "1.weight";
  }
  if (!tower || !proj) return false;  // a text-only conversion of the model
  // the processor: preprocessor_config.json (or processor_config.json's image_processor), else config.json
  json pp = vit_read_json(dir + "/preprocessor_config.json");
  const json pc = vit_read_json(dir + "/processor_config.json");
  if (pc.contains("image_processor") && pc["image_processor"].is_object())
    for (auto& [k, val] : pc["image_processor"].items()) pp[k] = val;
  r.min_patches = pp.value("min_num_patches", v.value("min_num_patches", r.min_patches));
  r.max_patches = pp.value("max_num_patches", v.value("max_num_patches", r.max_patches));
  const json mean = pp.contains("norm_mean") ? pp["norm_mean"] : root.value("norm_mean", json());
  const json stdv = pp.contains("norm_std") ? pp["norm_std"] : root.value("norm_std", json());
  if (mean.is_array() && mean.size() == 3 && stdv.is_array() && stdv.size() == 3)
    for (int i = 0; i < 3; ++i) r.mean[i] = mean[i].get<float>(), r.stdv[i] = stdv[i].get<float>();
  if (pp.value("patch_size", r.patch) != r.patch || std::fabs(pp.value("downsample_ratio", 0.5) - 0.5) > 1e-9) {
    logi("vision: %s: the image processor's patch size or downsampling does not match the tower: images are off",
         dir.c_str());
    return false;
  }
  if (c) *c = r;
  return true;
}

RadioViT::RadioViT(const std::string& dir, const RadioViTConfig& c) : dir_(dir), c_(c) {}

RadioViT::~RadioViT() { unload(); }

int RadioViT::patch_budget() const {
  const long cap = c_.max_pixels > 0 ? c_.max_pixels / (long(c_.patch) * c_.patch) : long(c_.max_patches);
  return int(std::clamp<long>(cap, c_.min_patches, c_.max_patches));
}

std::string RadioViT::describe() const {
  return name() + " (" + std::to_string(c_.depth) + " blocks, width " + std::to_string(c_.hidden) + ", " +
         std::to_string(c_.patch) + "-pixel patches; images of " + std::to_string(c_.min_patches) + " to " +
         std::to_string(patch_budget()) + " patches)";
}

// ---- the processor (CPU) ----

std::pair<int, int> radio_target_patches(int w, int h, int patch, int budget, int min_patches) {
  // closest grid: Python's round(x / patch + 0.5) (half to even)
  const long ch = long(std::nearbyint(double(h) / patch + 0.5)), cw = long(std::nearbyint(double(w) / patch + 0.5));
  // down to the budget
  const double factor = std::min(std::sqrt(double(budget) / double(ch * cw)), 1.0);
  long th = long(std::floor(factor * double(ch))), tw = long(std::floor(factor * double(cw)));
  // up to the minimum
  if (budget > min_patches && th * tw < min_patches) {
    const double up = std::sqrt(double(min_patches) / double(th * tw));
    th = long(std::ceil(up * double(th)));
    tw = long(std::ceil(up * double(tw)));
  }
  // even sides (the 2x2 pixel shuffle), within the budget when rounding up
  if (th % 2) th = (th + 1) * tw <= budget ? th + 1 : std::max(2L, th - 1);
  if (tw % 2) tw = th * (tw + 1) <= budget ? tw + 1 : std::max(2L, tw - 1);
  return {int(tw), int(th)};
}

namespace {
// torch's antialiased bicubic filter (a = -0.5), in float
inline float aa_cubic(float x) {
  const float a = -0.5f;
  x = std::fabs(x);
  if (x < 1.0f) return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
  if (x < 2.0f) return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
  return 0.0f;
}

// The taps of every output index along one axis, as torch's _compute_index_ranges_weights computes them for a float
// tensor (antialias, align_corners false): float arithmetic where torch's is float, double where it mixes in 0.5.
struct AaTaps {
  std::vector<int> start, size;
  std::vector<float> w;  // `width` weights per output index
  int width = 0;
};
AaTaps aa_taps(int in, int out) {
  AaTaps t;
  const float scale = float(in) / float(out);
  const float support = scale >= 1.0f ? float(2.0 * double(scale)) : 2.0f;
  const float invscale = scale >= 1.0f ? float(1.0 / double(scale)) : 1.0f;
  t.width = int(std::ceil(support)) * 2 + 1;
  t.start.resize(size_t(out));
  t.size.resize(size_t(out));
  t.w.assign(size_t(out) * t.width, 0.0f);
  for (int i = 0; i < out; ++i) {
    const float center = float(double(scale) * (double(i) + 0.5));
    const int64_t xmin = std::max<int64_t>(int64_t(double(center - support) + 0.5), 0);
    int64_t xsize = std::min<int64_t>(int64_t(double(center + support) + 0.5), in) - xmin;
    xsize = std::clamp<int64_t>(xsize, 0, t.width);
    float* w = t.w.data() + size_t(i) * t.width;
    float total = 0.0f;
    for (int64_t j = 0; j < xsize; ++j) {
      w[j] = aa_cubic(float((double(float(j + xmin) - center) + 0.5) * double(invscale)));
      total += w[j];
    }
    if (total != 0.0f)
      for (int64_t j = 0; j < xsize; ++j) w[j] /= total;
    t.start[size_t(i)] = int(xmin);
    t.size[size_t(i)] = int(xsize);
  }
  return t;
}
}  // namespace

std::vector<float> torch_resize_bicubic_aa(const std::vector<uint8_t>& rgb, int w, int h, int ow, int oh) {
  // channel planes of the input as float
  std::vector<float> src(size_t(3) * h * w);
  for (int c = 0; c < 3; ++c)
    for (size_t i = 0; i < size_t(h) * w; ++i) src[size_t(c) * h * w + i] = float(rgb[i * 3 + c]);
  if (ow != w) {  // the width pass
    const AaTaps t = aa_taps(w, ow);
    std::vector<float> dst(size_t(3) * h * ow);
    parallel_rows(3 * h, [&](int r0, int r1) {
      for (int r = r0; r < r1; ++r) {
        const float* s = src.data() + size_t(r) * w;
        float* d = dst.data() + size_t(r) * ow;
        for (int x = 0; x < ow; ++x) {
          const float* wt = t.w.data() + size_t(x) * t.width;
          const float* sp = s + t.start[size_t(x)];
          float acc = sp[0] * wt[0];
          for (int j = 1; j < t.size[size_t(x)]; ++j) acc = std::fma(sp[j], wt[j], acc);
          d[x] = acc;
        }
      }
    });
    src.swap(dst);
    w = ow;
  }
  if (oh != h) {  // the height pass
    const AaTaps t = aa_taps(h, oh);
    std::vector<float> dst(size_t(3) * oh * w);
    parallel_rows(3 * oh, [&](int r0, int r1) {
      for (int r = r0; r < r1; ++r) {
        const int c = r / oh, y = r % oh;
        const float* wt = t.w.data() + size_t(y) * t.width;
        const float* plane = src.data() + size_t(c) * h * w;
        float* d = dst.data() + size_t(r) * w;
        for (int x = 0; x < w; ++x) {
          const float* sp = plane + size_t(t.start[size_t(y)]) * w + x;
          float acc = sp[0] * wt[0];
          for (int j = 1; j < t.size[size_t(y)]; ++j) acc = std::fma(sp[size_t(j) * w], wt[j], acc);
          d[x] = acc;
        }
      }
    });
    src.swap(dst);
  }
  return src;
}

Prepared RadioViT::prepare(const Image& im) const {
  const auto [wp, hp] = radio_target_patches(im.width, im.height, c_.patch, patch_budget(), c_.min_patches);
  const int W = wp * c_.patch, H = hp * c_.patch;
  std::vector<float> px = torch_resize_bicubic_aa(im.rgb, im.width, im.height, W, H);
  for (int c = 0; c < 3; ++c) {  // the reference's float32 steps: (v / 255 - mean) / std
    float* p = px.data() + size_t(c) * H * W;
    for (size_t i = 0; i < size_t(H) * W; ++i) p[i] = (p[i] / 255.0f - c_.mean[c]) / c_.stdv[c];
  }
  Prepared p;
  p.width = W;
  p.height = H;
  p.grid_t = 1;
  p.grid_h = hp / 2;
  p.grid_w = wp / 2;
  p.tokens = p.grid_h * p.grid_w;
  p.pixels.resize(px.size() * 4);
  memcpy(p.pixels.data(), px.data(), p.pixels.size());
  uint8_t key[32 + 8 + 8];
  memcpy(key, im.digest.data(), 32);
  const int32_t dims[2] = {H, W};
  memcpy(key + 32, dims, 8);
  memcpy(key + 40, "radiovit", 8);
  p.digest = sha256(key, sizeof key);
  return p;
}

std::vector<float> RadioViT::pixel_values(const Prepared& p) const {
  std::vector<float> out(p.pixels.size() / 4);
  memcpy(out.data(), p.pixels.data(), out.size() * 4);
  return out;
}

// ---- weights ----

size_t RadioViT::weight_bytes() const {
  const size_t H = size_t(c_.hidden), I = size_t(c_.inter), P3 = size_t(3) * c_.patch * c_.patch, H4 = 4 * H;
  const size_t block = 4 * H + (3 * H * H + 3 * H) + (H * H + H) + (I * H + I) + (H * I + H);
  const size_t tower = block * c_.depth + (H * P3 + H) + size_t(c_.side) * c_.side * H + size_t(c_.prefix) * H;
  const size_t proj = H4 + (size_t(c_.proj_hidden) * H4 + c_.proj_hidden) +
                      (size_t(c_.out_hidden) * c_.proj_hidden + c_.out_hidden);
  return 2 * (tower + proj);
}

void RadioViT::load() {
  if (loaded_) return;
  tw_.open(dir_, kTower);
  pw_.open(dir_, kProj);
  patch_ = tw_.lin("patch_generator.embedder", 0, 0, /*bias=*/false);
  if (patch_.K != uint32_t(3 * c_.patch * c_.patch) || patch_.N != uint32_t(c_.hidden))
    fatal("vision: RADIO patch embedder of an unexpected shape");
  pos_ = tw_.vec("patch_generator.pos_embed");
  cls_ = tw_.vec("patch_generator.cls_token.token");
  if (const Tensor* t = tw_.find("patch_generator.pos_embed"); !t || t->numel() != int64_t(c_.side) * c_.side * c_.hidden)
    fatal("vision: RADIO position table of an unexpected size");
  if (const Tensor* t = tw_.find("patch_generator.cls_token.token"); !t || t->numel() != int64_t(c_.prefix) * c_.hidden)
    fatal("vision: RADIO cls / register tokens of an unexpected size");
  blocks_.resize(size_t(c_.depth));
  for (int i = 0; i < c_.depth; ++i) {
    Block& B = blocks_[size_t(i)];
    const std::string b = "blocks." + std::to_string(i) + ".";
    B.n1w = tw_.vec(b + "norm1.weight");
    B.n1b = tw_.vec(b + "norm1.bias");
    B.n2w = tw_.vec(b + "norm2.weight");
    B.n2b = tw_.vec(b + "norm2.bias");
    B.qkv = tw_.lin(b + "attn.qkv");
    B.proj = tw_.lin(b + "attn.proj");
    B.fc1 = tw_.lin(b + "mlp.fc1");
    B.fc2 = tw_.lin(b + "mlp.fc2");
  }
  mnorm_ = pw_.vec("0.weight");
  m1_ = pw_.lin("1", 0, 0, false);
  m2_ = pw_.lin("3", 0, 0, false);
  if (m1_.K != uint32_t(4 * c_.hidden) || m1_.N != uint32_t(c_.proj_hidden) || m2_.N != uint32_t(c_.out_hidden))
    fatal("vision: RADIO projector of an unexpected shape");
  tw_.close();
  pw_.close();
  tw_.make_resident();
  pw_.make_resident();
  loaded_ = true;
}

void RadioViT::unload() {
  tw_.release();
  pw_.release();
  blocks_.clear();
  loaded_ = false;
}

// ---- the tower on the GPU ----

namespace {
struct RadioPosParams {
  uint32_t N, D, cols, side, prefix;
  float scale;
};
struct RadioShuffleParams {
  uint32_t D, cols, prefix, t0;
  float eps;
};
struct EmbedRowsParams {
  uint32_t D, src_ld, dst_row0, src_row0;
};
// projector rows per command buffer (its two GEMMs: ~190 MFLOP per row)
int proj_rows(const RadioViTConfig& c, int M) {
  const double per = 2.0 * (4.0 * c.hidden * c.proj_hidden + double(c.proj_hidden) * c.out_hidden);
  return std::max(1, std::min(M, std::max(32, int(kVitCmdGflop * 1e9 / per) / 32 * 32)));
}
}  // namespace

size_t RadioViT::work_bytes(const Prepared& p) const {
  const size_t H = size_t(c_.hidden), I = size_t(c_.inter), N = size_t(p.height / c_.patch) * (p.width / c_.patch);
  const size_t NP = N + size_t(c_.prefix), rc = std::min<size_t>(NP, kVitRowChunk), M = N / 4;
  const size_t mc = size_t(proj_rows(c_, int(M))), PH = size_t(c_.proj_hidden);
  return N * 3 * c_.patch * c_.patch * 2 /* patches */ + NP * H * 4 /* x */ + NP * H * 2 /* xn */ +
         3 * NP * H * 2 /* q, k, v */ + std::max(rc * std::max(3 * H, I) * 4, mc * PH * 4) /* y */ +
         std::max(rc * I * 2, mc * PH * 2) /* h */ + mc * 4 * H * 2 /* shuffled */ + 64 * 1024;
}

void RadioViT::encode(const Prepared& p, const Buffer& out) {
  if (!loaded_) load();
  const int P = c_.patch, rows = p.height / P, cols = p.width / P, N = rows * cols, NP = N + c_.prefix;
  if (p.pixels.size() != size_t(3) * p.height * p.width * 4) fatal("vision: encode without pixels");
  const int H = c_.hidden, NH = c_.heads, HD = H / NH, I = c_.inter, H4 = 4 * H, PH = c_.proj_hidden,
            O = c_.out_hidden, K0 = 3 * P * P, M = N / 4;
  const int RC = std::min(NP, kVitRowChunk), MC = proj_rows(c_, M);
  Gpu& g = Gpu::get();
  std::vector<Buffer> scratch;
  auto alloc = [&](size_t bytes, const char* label) {
    scratch.push_back(g.alloc(bytes + 16, label));
    return scratch.back();
  };
  // patch rows (half [N, 3 * P * P], (channel, row, column) order) from the processor's channel planes
  Buffer pix = alloc(size_t(N) * K0 * 2, "radio_patches");
  {
    const float* src = reinterpret_cast<const float*>(p.pixels.data());
    _Float16* dst = reinterpret_cast<_Float16*>(pix.ptr());
    const int W = p.width, Hh = p.height;
    parallel_rows(N, [&](int r0, int r1) {
      for (int r = r0; r < r1; ++r) {
        const int pr = r / cols, pc = r % cols;
        _Float16* d = dst + size_t(r) * K0;
        for (int c = 0; c < 3; ++c)
          for (int y = 0; y < P; ++y) {
            const float* s = src + (size_t(c) * Hh + size_t(pr * P + y)) * W + size_t(pc * P);
            for (int x = 0; x < P; ++x) d[(c * P + y) * P + x] = _Float16(s[x]);
          }
      }
    });
  }
  Buffer x = alloc(size_t(NP) * H * 4, "radio_x");
  Buffer xn = alloc(size_t(NP) * H * 2, "radio_xn");
  Buffer qkv = alloc(size_t(3) * NP * H * 2, "radio_qkv");
  Buffer y = alloc(std::max(size_t(RC) * std::max(3 * H, I), size_t(MC) * PH) * 4, "radio_y");
  Buffer hh = alloc(std::max(size_t(RC) * I, size_t(MC) * PH) * 2, "radio_h");
  Buffer sh = alloc(size_t(MC) * H4 * 2, "radio_shuffled");
  const Buffer Qb = qkv.view(0, size_t(NP) * H * 2), Kb = qkv.view(size_t(NP) * H * 2, size_t(NP) * H * 2),
               Vb = qkv.view(size_t(2) * NP * H * 2, size_t(NP) * H * 2);
  VitCmds cs;
  // cls / register tokens, then the patches' embeddings + their position embeddings
  {
    Encoder& e = cs.at(2.0 * N * K0 * H / 1e9);
    const EmbedRowsParams ep{uint32_t(H), uint32_t(H), 0, 0};
    e.use(vit_pipe("embed_rows_bf16"));
    e.buf(0, cls_);
    e.buf(1, x);
    e.val(2, ep);
    e.dispatch((uint32_t(H) / 4 + 255) / 256, uint32_t(c_.prefix), 1, 256);
    vit_gemm(e, patch_, pix, uint32_t(K0), x.view(size_t(c_.prefix) * H * 4, size_t(N) * H * 4), uint32_t(H), N, false);
    const RadioPosParams pp{uint32_t(N), uint32_t(H), uint32_t(cols), uint32_t(c_.side), uint32_t(c_.prefix),
                            float(c_.side) / float(std::max(rows, cols))};
    e.use(vit_pipe("radio_pos"));
    e.buf(0, x);
    e.buf(1, pos_);
    e.val(2, pp);
    e.dispatch((uint32_t(H) + 255) / 256, uint32_t(N), 1, 256);
  }
  const int stop = dbg_stop_ >= 0 ? std::min(dbg_stop_, c_.depth) : c_.depth;
  const Buffer* pending = nullptr;  // the residual bias still to add to x (folded into the next norm)
  for (int L = 0; L < stop; ++L) {
    const Block& B = blocks_[size_t(L)];
    vit_ln(cs.at(0.01 * NP * H / 1e6), x, uint32_t(H), pending, B.n1w, B.n1b, xn, uint32_t(H), NP, H, c_.ln_eps);
    for (int r0 = 0; r0 < NP; r0 += RC) {
      const int rc = std::min(RC, NP - r0);
      Encoder& e = cs.at(2.0 * rc * H * 3 * H / 1e9);
      vit_gemm(e, B.qkv, xn.view(size_t(r0) * H * 2, size_t(rc) * H * 2), uint32_t(H), y, uint32_t(3 * H), rc, false);
      vit_qkv_split(e, y, B.qkv.b, Qb, Kb, Vb, rc, r0, NP, NH, HD);
    }
    vit_attention(cs, Qb, Kb, Vb, xn, NP, H, NH, HD);
    for (int r0 = 0; r0 < NP; r0 += RC) {
      const int rc = std::min(RC, NP - r0);
      vit_gemm(cs.at(2.0 * rc * H * H / 1e9), B.proj, xn.view(size_t(r0) * H * 2, size_t(rc) * H * 2), uint32_t(H),
               x.view(size_t(r0) * H * 4, size_t(rc) * H * 4), uint32_t(H), rc, true);
    }
    vit_ln(cs.at(0.01 * NP * H / 1e6), x, uint32_t(H), &B.proj.b, B.n2w, B.n2b, xn, uint32_t(H), NP, H, c_.ln_eps);
    for (int r0 = 0; r0 < NP; r0 += RC) {
      const int rc = std::min(RC, NP - r0);
      Encoder& e = cs.at(4.0 * rc * H * I / 1e9);
      vit_gemm(e, B.fc1, xn.view(size_t(r0) * H * 2, size_t(rc) * H * 2), uint32_t(H), y, uint32_t(I), rc, false);
      vit_act(e, "vit_bias_act_h", y, uint32_t(I), B.fc1.b, hh, uint32_t(I), rc, I, 2);
      vit_gemm(e, B.fc2, hh, uint32_t(I), x.view(size_t(r0) * H * 4, size_t(rc) * H * 4), uint32_t(H), rc, true);
    }
    pending = &B.fc2.b;
  }
  if (dbg_stop_ >= 0) {  // tests: every row's hidden state after `stop` blocks, fp32
    Encoder& e = cs.at(0.01);
    vit_act(e, "vit_bias_f32", x, uint32_t(H), pending ? *pending : patch_.b, out, uint32_t(H), NP, H,
            pending ? 1u : 0u);
    cs.finish();
    for (Buffer& b : scratch) g.release(b);
    return;
  }
  // the projector: 2x2 patches as one row (+ the last block's pending bias), RMSNorm, linear, ReLU^2, linear -> bf16
  for (int m0 = 0; m0 < M; m0 += MC) {
    const int mc = std::min(MC, M - m0);
    Encoder& e = cs.at(2.0 * mc * (double(H4) * PH + double(PH) * O) / 1e9);
    const RadioShuffleParams sp{uint32_t(H), uint32_t(cols), uint32_t(c_.prefix), uint32_t(m0), c_.rms_eps};
    e.use(vit_pipe("radio_shuffle_rms"));
    e.buf(0, x);
    e.buf(1, *pending);
    e.buf(2, mnorm_);
    e.buf(3, sh);
    e.val(4, sp);
    e.dispatch(uint32_t(mc), 1, 1, 256);
    vit_gemm(e, m1_, sh, uint32_t(H4), y, uint32_t(PH), mc, false);
    vit_act(e, "vit_bias_act_h", y, uint32_t(PH), m1_.b, hh, uint32_t(PH), mc, PH, 3);
    vit_gemm(e, m2_, hh, uint32_t(PH), y, uint32_t(O), mc, false);
    vit_act(e, "vit_bias_act_b", y, uint32_t(O), m2_.b, out.view(size_t(m0) * O * 2, size_t(mc) * O * 2), uint32_t(O),
            mc, O, 0);
  }
  cs.finish();
  for (Buffer& b : scratch) g.release(b);
}

}  // namespace ni::vision
