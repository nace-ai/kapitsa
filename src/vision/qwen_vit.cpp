// The Qwen3.5-family image encoder (see qwen_vit.h).
#include "vision/qwen_vit.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>

namespace ni::vision {

using json = nlohmann::json;

// ---- configuration ----

// The tensor-name prefix of the vision tower in dir's safetensors ("vision_tower." in MLX conversions,
// "model.visual." in Hugging Face checkpoints), "" if there is none.
static std::string vision_prefix(const std::string& dir) {
  const std::vector<std::string> names = vit_tensor_names(dir);
  for (const char* p : {"vision_tower.", "model.visual."})
    for (const std::string& n : names)
      if (n.rfind(std::string(p) + "patch_embed.", 0) == 0) return p;
  return "";
}

bool QwenViT::supports(const std::string& dir, QwenViTConfig* c) {
  const json root = vit_read_json(dir + "/config.json");
  if (!root.contains("vision_config") || !root["vision_config"].is_object()) return false;
  const std::string mt = root.value("model_type", std::string());
  if (mt.rfind("qwen3_5", 0) != 0) return false;
  const json& v = root["vision_config"];
  QwenViTConfig q;
  q.depth = v.value("depth", q.depth);
  q.hidden = v.value("hidden_size", q.hidden);
  q.inter = v.value("intermediate_size", q.inter);
  q.heads = v.value("num_heads", q.heads);
  q.out_hidden = v.value("out_hidden_size", q.out_hidden);
  q.patch = v.value("patch_size", q.patch);
  q.temporal = v.value("temporal_patch_size", q.temporal);
  q.merge = v.value("spatial_merge_size", q.merge);
  q.in_ch = v.value("in_channels", q.in_ch);
  q.pos_embeds = v.value("num_position_embeddings", q.pos_embeds);
  q.gelu_tanh = v.value("hidden_act", std::string("gelu_pytorch_tanh")) != "gelu";
  q.image_token = root.value("image_token_id", q.image_token);
  if (v.contains("deepstack_visual_indexes") && v["deepstack_visual_indexes"].is_array() &&
      !v["deepstack_visual_indexes"].empty()) {
    logi("vision: %s has deepstack vision layers, which Kapitsa does not run: images are off", dir.c_str());
    return false;
  }
  const json t = root.contains("text_config") ? root["text_config"] : root;
  if (q.out_hidden != t.value("hidden_size", 0) || q.hidden % q.heads || (q.hidden / q.heads) != 72 || q.in_ch != 3 ||
      q.merge != 2 || q.hidden % 64) {
    logi("vision: %s has a vision tower of a shape Kapitsa does not run (hidden %d, %d heads, out %d): images are off",
         dir.c_str(), q.hidden, q.heads, q.out_hidden);
    return false;
  }
  const int side = int(std::lround(std::sqrt(double(q.pos_embeds))));
  if (side * side != q.pos_embeds) return false;
  if (vision_prefix(dir).empty()) return false;  // a text-only conversion of the model
  // the image processor: preprocessor_config.json, then processor_config.json's image_processor over it (as mlx_vlm)
  json pp = vit_read_json(dir + "/preprocessor_config.json");
  const json pc = vit_read_json(dir + "/processor_config.json");
  if (pc.contains("image_processor") && pc["image_processor"].is_object())
    for (auto& [k, val] : pc["image_processor"].items()) pp[k] = val;
  if (pp.contains("size") && pp["size"].is_object()) {
    q.min_pixels = pp["size"].value("shortest_edge", q.min_pixels);
    q.max_pixels = pp["size"].value("longest_edge", q.max_pixels);
  }
  if (pp.contains("min_pixels") && pp["min_pixels"].is_number()) q.min_pixels = pp["min_pixels"].get<long>();
  if (pp.contains("max_pixels") && pp["max_pixels"].is_number()) q.max_pixels = pp["max_pixels"].get<long>();
  if (pp.value("patch_size", q.patch) != q.patch || pp.value("merge_size", q.merge) != q.merge ||
      pp.value("temporal_patch_size", q.temporal) != q.temporal) {
    logi("vision: %s: the image processor's patch geometry differs from the vision tower's: images are off", dir.c_str());
    return false;
  }
  for (int i = 0; i < 3; ++i) {
    if (pp.contains("image_mean") && pp["image_mean"].size() == 3) q.mean[i] = pp["image_mean"][i].get<float>();
    if (pp.contains("image_std") && pp["image_std"].size() == 3) q.stdv[i] = pp["image_std"][i].get<float>();
  }
  q.rescale = pp.value("rescale_factor", q.rescale);
  q.do_rescale = pp.value("do_rescale", true);
  q.do_normalize = pp.value("do_normalize", true);
  if (c) *c = q;
  return true;
}

QwenViT::QwenViT(const std::string& dir, const QwenViTConfig& c) : dir_(dir), c_(c) {
  prefix_ = vision_prefix(dir);
  inter_pad_ = uint32_t((c_.inter + 63) / 64 * 64);
}

std::string QwenViT::describe() const {
  return name() + " (" + std::to_string(c_.depth) + " blocks, width " + std::to_string(c_.hidden) + ", " +
         std::to_string(c_.patch) + "-pixel patches; images of " + std::to_string(c_.min_pixels) + " to " +
         std::to_string(c_.max_pixels) + " pixels)";
}

QwenViT::~QwenViT() { unload(); }

// ---- the processor (CPU) ----

std::pair<int, int> qwen_smart_resize(int h, int w, int factor, long min_pixels, long max_pixels) {
  if (h < 1 || w < 1) throw Error("an image has no pixels");
  if (double(std::max(h, w)) / double(std::min(h, w)) > 200)
    throw Error("an image is " + std::to_string(w) + " x " + std::to_string(h) +
                ": its aspect ratio must be smaller than 200");
  // Python's round(): halves to even, as nearbyint in the default rounding mode
  long hb = long(std::nearbyint(double(h) / factor)) * factor;
  long wb = long(std::nearbyint(double(w) / factor)) * factor;
  if (hb * wb > max_pixels) {
    const double beta = std::sqrt(double(long(h) * long(w)) / double(max_pixels));
    hb = std::max<long>(factor, long(std::floor(double(h) / beta / factor)) * factor);
    wb = std::max<long>(factor, long(std::floor(double(w) / beta / factor)) * factor);
  } else if (hb * wb < min_pixels) {
    const double beta = std::sqrt(double(min_pixels) / double(long(h) * long(w)));
    hb = long(std::ceil(double(h) * beta / factor)) * factor;
    wb = long(std::ceil(double(w) * beta / factor)) * factor;
  }
  return {int(hb), int(wb)};
}

// Pillow's resampling (libImaging/Resample.c), for 8-bit RGB and the bicubic filter (a = -0.5): coefficients in
// double, normalized, as 22-bit fixed point; horizontal pass over the rows the vertical pass needs, then vertical;
// each pass rounds to 8 bits.
namespace {
constexpr int kPrecisionBits = 32 - 8 - 2;

double bicubic_filter(double x) {
  const double a = -0.5;
  if (x < 0.0) x = -x;
  if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
  if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
  return 0.0;
}

struct Coeffs {
  int ksize = 0;
  std::vector<int> bounds;  // per output: xmin, count
  std::vector<int> kk;      // per output: ksize fixed-point coefficients
};

Coeffs precompute_coeffs(int in_size, int out_size) {
  const float in0 = 0.0f, in1 = float(in_size);
  double filterscale, scale;
  filterscale = scale = double(in1 - in0) / out_size;
  if (filterscale < 1.0) filterscale = 1.0;
  const double support = 2.0 * filterscale;
  Coeffs c;
  c.ksize = int(std::ceil(support)) * 2 + 1;
  std::vector<double> kk(size_t(out_size) * c.ksize, 0.0);
  c.bounds.resize(size_t(out_size) * 2);
  for (int xx = 0; xx < out_size; ++xx) {
    const double center = in0 + (xx + 0.5) * scale;
    double ww = 0.0;
    const double ss = 1.0 / filterscale;
    int xmin = int(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = int(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;
    double* k = &kk[size_t(xx) * c.ksize];
    for (int x = 0; x < xmax; ++x) {
      const double w = bicubic_filter((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (int x = 0; x < xmax; ++x)
      if (ww != 0.0) k[x] /= ww;
    c.bounds[size_t(xx) * 2] = xmin;
    c.bounds[size_t(xx) * 2 + 1] = xmax;
  }
  c.kk.resize(kk.size());
  for (size_t i = 0; i < kk.size(); ++i)
    c.kk[i] = kk[i] < 0 ? int(-0.5 + kk[i] * (1 << kPrecisionBits)) : int(0.5 + kk[i] * (1 << kPrecisionBits));
  return c;
}

inline uint8_t clip8(int in) {
  if (in >= (1 << kPrecisionBits << 8)) return 255;
  if (in <= 0) return 0;
  return uint8_t(in >> kPrecisionBits);
}

// rows [y0, y1) of the output, in parallel
template <class F>
void parallel_rows(int n, F f) {
  const int nt = std::max(1, std::min<int>(8, int(std::thread::hardware_concurrency())));
  if (n < 64 || nt == 1) {
    f(0, n);
    return;
  }
  std::vector<std::thread> th;
  const int per = (n + nt - 1) / nt;
  for (int t = 0; t < nt; ++t)
    if (t * per < n) th.emplace_back(f, t * per, std::min(n, (t + 1) * per));
  for (auto& t : th) t.join();
}
}  // namespace

// One pass of 8-bit RGB: horizontal (output column xx reads input columns bounds(xx) - off of the same row) over
// `rows` rows from row y0 of src, or vertical (output row yy reads input rows bounds(yy) - off) over `cols` columns
// from column x0. Rows of `sw` pixels in, of `ow` (horizontal) or `cols` (vertical) out.
static void hpass(const uint8_t* src, int sw, int y0, int rows, const Coeffs& c, int ow, int off, uint8_t* out) {
  parallel_rows(rows, [&](int r0, int r1) {
    for (int yy = r0; yy < r1; ++yy) {
      const uint8_t* in = src + size_t(yy + y0) * sw * 3;
      uint8_t* o = out + size_t(yy) * ow * 3;
      for (int xx = 0; xx < ow; ++xx) {
        const int xmin = c.bounds[size_t(xx) * 2] - off, xmax = c.bounds[size_t(xx) * 2 + 1];
        const int* k = &c.kk[size_t(xx) * c.ksize];
        int s0 = 1 << (kPrecisionBits - 1), s1 = s0, s2 = s0;
        for (int x = 0; x < xmax; ++x) {
          const uint8_t* px = in + size_t(x + xmin) * 3;
          s0 += px[0] * k[x];
          s1 += px[1] * k[x];
          s2 += px[2] * k[x];
        }
        o[xx * 3] = clip8(s0), o[xx * 3 + 1] = clip8(s1), o[xx * 3 + 2] = clip8(s2);
      }
    }
  });
}

static void vpass(const uint8_t* src, int sw, int x0, int cols, const Coeffs& c, int oh, int off, uint8_t* out) {
  parallel_rows(oh, [&](int r0, int r1) {
    for (int yy = r0; yy < r1; ++yy) {
      const int ymin = c.bounds[size_t(yy) * 2] - off, ymax = c.bounds[size_t(yy) * 2 + 1];
      const int* k = &c.kk[size_t(yy) * c.ksize];
      uint8_t* o = out + size_t(yy) * cols * 3;
      for (int xx = 0; xx < cols; ++xx) {
        int s0 = 1 << (kPrecisionBits - 1), s1 = s0, s2 = s0;
        for (int y = 0; y < ymax; ++y) {
          const uint8_t* px = src + (size_t(y + ymin) * sw + x0 + xx) * 3;
          s0 += px[0] * k[y];
          s1 += px[1] * k[y];
          s2 += px[2] * k[y];
        }
        o[xx * 3] = clip8(s0), o[xx * 3 + 1] = clip8(s1), o[xx * 3 + 2] = clip8(s2);
      }
    }
  });
}

std::vector<uint8_t> pil_resize_bicubic(const std::vector<uint8_t>& rgb, int w, int h, int ow, int oh) {
  if (ow == w && oh == h) return rgb;
  const Coeffs ch = precompute_coeffs(w, ow), cv = precompute_coeffs(h, oh);
  std::vector<uint8_t> out;
  if (ow == w || oh == h) {  // one pass
    out.resize(size_t(ow) * oh * 3);
    if (ow != w) hpass(rgb.data(), w, 0, h, ch, ow, 0, out.data());
    else vpass(rgb.data(), w, 0, w, cv, oh, 0, out.data());
    return out;
  }
  // Pillow runs the horizontal pass first (over the rows the vertical pass reads), except for images more than 100
  // times taller than wide: vertical first (over the columns the horizontal pass reads). Each pass rounds to 8 bits,
  // so the order shows in the result.
  std::vector<uint8_t> tmp;
  out.resize(size_t(ow) * oh * 3);
  if (h > 100 * w) {
    const int x0 = ch.bounds[0], x1 = ch.bounds[size_t(ow) * 2 - 2] + ch.bounds[size_t(ow) * 2 - 1];
    tmp.resize(size_t(x1 - x0) * oh * 3);
    vpass(rgb.data(), w, x0, x1 - x0, cv, oh, 0, tmp.data());
    hpass(tmp.data(), x1 - x0, 0, oh, ch, ow, x0, out.data());
  } else {
    const int y0 = cv.bounds[0], y1 = cv.bounds[size_t(oh) * 2 - 2] + cv.bounds[size_t(oh) * 2 - 1];
    tmp.resize(size_t(ow) * (y1 - y0) * 3);
    hpass(rgb.data(), w, y0, y1 - y0, ch, ow, 0, tmp.data());
    vpass(tmp.data(), ow, 0, ow, cv, oh, y0, out.data());
  }
  return out;
}

Prepared QwenViT::prepare(const Image& im) const {
  const int factor = c_.patch * c_.merge;
  const auto [rh, rw] = qwen_smart_resize(im.height, im.width, factor, c_.min_pixels, c_.max_pixels);
  Prepared p;
  p.width = rw;
  p.height = rh;
  p.pixels = pil_resize_bicubic(im.rgb, im.width, im.height, rw, rh);
  p.grid_t = 1;
  p.grid_h = rh / c_.patch / c_.merge;
  p.grid_w = rw / c_.patch / c_.merge;
  p.tokens = p.grid_t * p.grid_h * p.grid_w;
  uint8_t key[32 + 8 + 8];
  memcpy(key, im.digest.data(), 32);
  const int32_t dims[2] = {rh, rw};
  memcpy(key + 32, dims, 8);
  memcpy(key + 40, "qwen-vit", 8);
  p.digest = sha256(key, sizeof key);
  return p;
}

// value v of channel c after rescale and normalize, as the reference processors compute it (float32 steps)
static void build_lut(const QwenViTConfig& c, float* lut) {
  const float rs = float(c.rescale);
  for (int ch = 0; ch < 3; ++ch)
    for (int v = 0; v < 256; ++v) {
      float x = float(v);
      if (c.do_rescale) x = x * rs;
      if (c.do_normalize) {
        x = x - c.mean[ch];
        x = x / c.stdv[ch];
      }
      lut[ch * 256 + v] = x;
    }
}

// patch r (merge-block order) -> its (row, column) in the patch grid
static inline void patch_rc(int r, int gw, int merge, int& prow, int& pcol) {
  const int mm = merge * merge, b = r / mm, bw = gw / merge;
  prow = (b / bw) * merge + (r % mm) / merge;
  pcol = (b % bw) * merge + r % merge;
}

std::vector<float> QwenViT::pixel_values(const Prepared& p) const {
  float lut[3 * 256];
  build_lut(c_, lut);
  const int P = c_.patch, T = c_.temporal, gh = p.height / P, gw = p.width / P, N = gh * gw;
  const size_t K = size_t(3) * T * P * P;
  std::vector<float> out(size_t(N) * K);
  for (int r = 0; r < N; ++r) {
    int prow, pcol;
    patch_rc(r, gw, c_.merge, prow, pcol);
    float* o = out.data() + size_t(r) * K;
    for (int ch = 0; ch < 3; ++ch)
      for (int t = 0; t < T; ++t)
        for (int y = 0; y < P; ++y)
          for (int x = 0; x < P; ++x)
            o[((size_t(ch) * T + t) * P + y) * P + x] =
                lut[ch * 256 + p.pixels[(size_t(prow * P + y) * p.width + pcol * P + x) * 3 + ch]];
  }
  return out;
}

// ---- weights ----

size_t QwenViT::weight_bytes() const {
  const size_t H = size_t(c_.hidden), I = inter_pad_, K0 = size_t(c_.temporal) * c_.patch * c_.patch * c_.in_ch;
  const size_t block = (H * 3 * H + 3 * H) + (H * H + H) + (I * H + I) + (H * I + H) + 4 * H;
  const size_t merger = 4 * H * 4 * H + 4 * H + size_t(c_.out_hidden) * 4 * H + c_.out_hidden + 2 * H;
  return 2 * (block * c_.depth + merger + H * K0 + H + size_t(c_.pos_embeds) * H);
}

void QwenViT::load() {
  if (loaded_) return;
  w_.open(dir_, prefix_);
  const uint32_t H = uint32_t(c_.hidden), P = uint32_t(c_.patch), T = uint32_t(c_.temporal);
  const uint32_t K0 = T * P * P * uint32_t(c_.in_ch);
  // patch embedding: Conv3d [hidden, T, P, P, C] (MLX) or [hidden, C, T, P, P] (Hugging Face), as a linear over
  // (frame, row, column, channel)
  {
    const Tensor* t = w_.find("patch_embed.proj.weight");
    if (!t || t->numel() != int64_t(H) * K0) fatal("vision: patch_embed.proj.weight missing or of an unexpected size");
    patch_.N = H;
    patch_.K = K0;
    patch_.w = w_.alloc(size_t(H) * K0 * 2, "vit_w");
    w_.read_bf16("patch_embed.proj.weight", patch_.w.as<uint16_t>(), H, K0, K0);
    const bool channels_first = t->shape.size() == 5 && t->shape[1] == c_.in_ch && t->shape[4] != c_.in_ch;
    if (channels_first) {  // (c, t, y, x) -> (t, y, x, c)
      std::vector<uint16_t> row(K0);
      for (uint32_t o = 0; o < H; ++o) {
        uint16_t* w = patch_.w.as<uint16_t>() + size_t(o) * K0;
        for (uint32_t c = 0; c < 3; ++c)
          for (uint32_t f = 0; f < T; ++f)
            for (uint32_t y = 0; y < P; ++y)
              for (uint32_t x = 0; x < P; ++x) row[((f * P + y) * P + x) * 3 + c] = w[((c * T + f) * P + y) * P + x];
        memcpy(w, row.data(), K0 * 2);
      }
    }
    patch_.b = w_.alloc(size_t(H) * 2, "vit_b");
    w_.read_bf16("patch_embed.proj.bias", patch_.b.as<uint16_t>(), 1, H, H);
  }
  pos_ = w_.vec("pos_embed.weight");
  blocks_.resize(size_t(c_.depth));
  for (int i = 0; i < c_.depth; ++i) {
    Block& B = blocks_[size_t(i)];
    const std::string b = "blocks." + std::to_string(i) + ".";
    B.n1w = w_.vec(b + "norm1.weight");
    B.n1b = w_.vec(b + "norm1.bias");
    B.n2w = w_.vec(b + "norm2.weight");
    B.n2b = w_.vec(b + "norm2.bias");
    B.qkv = w_.lin(b + "attn.qkv");
    B.proj = w_.lin(b + "attn.proj");
    B.fc1 = w_.lin(b + "mlp.linear_fc1", inter_pad_, 0);  // [4352, hidden]: zero rows past 4304
    B.fc2 = w_.lin(b + "mlp.linear_fc2", 0, inter_pad_);  // [hidden, 4352]: zero columns past 4304
  }
  mnw_ = w_.vec("merger.norm.weight");
  mnb_ = w_.vec("merger.norm.bias");
  m1_ = w_.lin("merger.linear_fc1");
  m2_ = w_.lin("merger.linear_fc2");
  w_.close();
  w_.make_resident();
  loaded_ = true;
}

void QwenViT::unload() {
  w_.release();
  blocks_.clear();
  loaded_ = false;
}

// ---- the tower on the GPU ----

namespace {
struct QvitPatchParams {
  uint32_t W, gw, P, T, merge;
};
struct QvitEmbedParams {
  uint32_t N, D, gw, side, merge;
};
struct QvitRopeParams {
  uint32_t rows, r0, N, heads, HD, gw, merge;
};
}  // namespace

size_t QwenViT::work_bytes(const Prepared& p) const {
  const size_t H = size_t(c_.hidden), N = size_t(p.height / c_.patch) * size_t(p.width / c_.patch);
  const size_t rc = std::min<size_t>(N, kVitRowChunk);
  const size_t ym = std::max<size_t>(3 * H, inter_pad_);
  return p.pixels.size() + N * H * 4 /* x */ + N * H * 2 /* xn */ + 3 * N * H * 2 /* q, k, v */ +
         rc * ym * 4 /* y */ + rc * inter_pad_ * 2 /* h */ + 64 * 1024 /* tables */;
}

void QwenViT::encode(const Prepared& p, const Buffer& out) {
  if (!loaded_) load();
  if (p.pixels.size() != size_t(p.width) * p.height * 3) fatal("vision: encode without pixels");
  Gpu& g = Gpu::get();
  const int P = c_.patch, T = c_.temporal, mg = c_.merge;
  const int gh = p.height / P, gw = p.width / P, N = gh * gw, M = N / (mg * mg);
  const int H = c_.hidden, NH = c_.heads, HD = H / NH, I = int(inter_pad_), H4 = H * mg * mg, O = c_.out_hidden;
  const int K0 = T * P * P * 3;
  const int RC = std::min(N, kVitRowChunk);
  const int YM = std::max(3 * H, I);
  std::vector<Buffer> scratch;
  auto alloc = [&](size_t bytes, const char* label) {
    scratch.push_back(g.alloc(bytes + 16, label));
    return scratch.back();
  };
  Buffer pix = alloc(p.pixels.size(), "vit_pixels");
  memcpy(pix.ptr(), p.pixels.data(), p.pixels.size());
  Buffer lut = alloc(3 * 256 * 4, "vit_lut");
  build_lut(c_, lut.as<float>());
  // position-embedding interpolation per grid row / column: (floor, ceil, 1 - frac, frac), align corners
  const int side = int(std::lround(std::sqrt(double(c_.pos_embeds))));
  auto axis = [&](int n, const char* label) {
    Buffer b = alloc(size_t(n) * 16, label);
    float* f = b.as<float>();
    for (int i = 0; i < n; ++i) {
      const float src = float(i * (side - 1)) / float(std::max(n - 1, 1));
      const float fl = std::floor(src), fr = src - fl;
      f[4 * i] = fl;
      f[4 * i + 1] = std::min(fl + 1.0f, float(side - 1));
      f[4 * i + 2] = 1.0f - fr;
      f[4 * i + 3] = fr;
    }
    return b;
  };
  Buffer rowinfo = axis(gh, "vit_rows"), colinfo = axis(gw, "vit_cols");
  Buffer invf = alloc(size_t(HD / 4) * 4, "vit_invfreq");
  for (int j = 0; j < HD / 4; ++j)  // the reference's float32 1 / 10000^(2j / (HD/2))
    invf.as<float>()[j] = 1.0f / std::pow(10000.0f, float(2 * j) / float(HD / 2));
  Buffer x = alloc(size_t(N) * H * 4, "vit_x");
  Buffer xn = alloc(size_t(N) * H * 2, "vit_xn");
  Buffer qkv = alloc(size_t(3) * N * H * 2, "vit_qkv");
  Buffer y = alloc(size_t(RC) * YM * 4, "vit_y");
  Buffer hh = alloc(size_t(RC) * I * 2, "vit_h");
  const Buffer Qb = qkv.view(0, size_t(N) * H * 2), Kb = qkv.view(size_t(N) * H * 2, size_t(N) * H * 2),
               Vb = qkv.view(size_t(2) * N * H * 2, size_t(N) * H * 2);
  VitCmds cs;
  auto ln = [&](Encoder& e, const Buffer* pb, const Buffer& w, const Buffer& b) {
    vit_ln(e, x, uint32_t(H), pb, w, b, xn, uint32_t(H), N, H, 1e-6f);
  };
  const auto& act = vit_act;
  // patches -> patch embedding (+ bias, + interpolated position embedding) into x
  {
    Encoder& e = cs.at(2.0 * N * K0 * H / 1e9);
    const QvitPatchParams pp{uint32_t(p.width), uint32_t(gw), uint32_t(P), uint32_t(T), uint32_t(mg)};
    e.use(vit_pipe("qvit_patchify"));
    e.buf(0, pix);
    e.buf(1, lut);
    e.buf(2, qkv);
    e.val(3, pp);
    e.dispatch((uint32_t(P * P * 3) + 255) / 256, uint32_t(N), 1, 256);
    vit_gemm(e, patch_, qkv, uint32_t(K0), x, uint32_t(H), N, false);
    const QvitEmbedParams ep{uint32_t(N), uint32_t(H), uint32_t(gw), uint32_t(side), uint32_t(mg)};
    e.use(vit_pipe("qvit_embed"));
    e.buf(0, x);
    e.buf(1, patch_.b);
    e.buf(2, pos_);
    e.buf(3, rowinfo);
    e.buf(4, colinfo);
    e.val(5, ep);
    e.dispatch((uint32_t(H) + 255) / 256, uint32_t(N), 1, 256);
  }
  const int stop = dbg_stop_ >= 0 ? std::min(dbg_stop_, c_.depth) : c_.depth;
  const Buffer* pending = nullptr;  // the residual bias still to add to x (folded into the next LayerNorm)
  for (int L = 0; L < stop; ++L) {
    const Block& B = blocks_[size_t(L)];
    ln(cs.at(0.01 * N * H / 1e6), pending, B.n1w, B.n1b);
    for (int r0 = 0; r0 < N; r0 += RC) {
      const int rc = std::min(RC, N - r0);
      Encoder& e = cs.at(2.0 * rc * H * 3 * H / 1e9);
      vit_gemm(e, B.qkv, xn.view(size_t(r0) * H * 2, size_t(rc) * H * 2), uint32_t(H), y, uint32_t(3 * H), rc, false);
      const QvitRopeParams rp{uint32_t(rc), uint32_t(r0), uint32_t(N), uint32_t(NH), uint32_t(HD), uint32_t(gw),
                              uint32_t(mg)};
      e.use(vit_pipe("qvit_qkv_rope"));
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
    vit_attention(cs, Qb, Kb, Vb, xn, N, H, NH, HD);
    for (int r0 = 0; r0 < N; r0 += RC) {
      const int rc = std::min(RC, N - r0);
      vit_gemm(cs.at(2.0 * rc * H * H / 1e9), B.proj, xn.view(size_t(r0) * H * 2, size_t(rc) * H * 2), uint32_t(H),
           x.view(size_t(r0) * H * 4, size_t(rc) * H * 4), uint32_t(H), rc, true);
    }
    ln(cs.at(0.01 * N * H / 1e6), &B.proj.b, B.n2w, B.n2b);
    for (int r0 = 0; r0 < N; r0 += RC) {
      const int rc = std::min(RC, N - r0);
      Encoder& e = cs.at(4.0 * rc * H * I / 1e9);
      vit_gemm(e, B.fc1, xn.view(size_t(r0) * H * 2, size_t(rc) * H * 2), uint32_t(H), y, uint32_t(I), rc, false);
      act(e, "vit_bias_act_h", y, uint32_t(I), B.fc1.b, hh, uint32_t(I), rc, c_.inter, c_.gelu_tanh ? 1u : 2u);
      vit_gemm(e, B.fc2, hh, uint32_t(I), x.view(size_t(r0) * H * 4, size_t(rc) * H * 4), uint32_t(H), rc, true);
    }
    pending = &B.fc2.b;
  }
  if (dbg_stop_ >= 0) {  // tests: the hidden state after `stop` blocks, fp32
    Encoder& e = cs.at(0.01);
    act(e, "vit_bias_f32", x, uint32_t(H), pending ? *pending : patch_.b, out, uint32_t(H), N, H, pending ? 1u : 0u);
    cs.finish();
    for (Buffer& b : scratch) g.release(b);
    return;
  }
  // merger: LayerNorm per patch, 2x2 patches as one row of 4 * hidden, fc1, GELU, fc2 -> out (bf16)
  ln(cs.at(0.01 * N * H / 1e6), pending, mnw_, mnb_);
  const int MC = std::max(1, RC / (mg * mg));
  for (int m0 = 0; m0 < M; m0 += MC) {
    const int mc = std::min(MC, M - m0);
    Encoder& e = cs.at(2.0 * mc * H4 * (H4 + O) / 1e9);
    vit_gemm(e, m1_, xn.view(size_t(m0) * H4 * 2, size_t(mc) * H4 * 2), uint32_t(H4), y, uint32_t(H4), mc, false);
    act(e, "vit_bias_act_h", y, uint32_t(H4), m1_.b, hh, uint32_t(H4), mc, H4, 2);
    vit_gemm(e, m2_, hh, uint32_t(H4), y, uint32_t(O), mc, false);
    act(e, "vit_bias_act_b", y, uint32_t(O), m2_.b, out.view(size_t(m0) * O * 2, size_t(mc) * O * 2), uint32_t(O), mc,
        O, 0);
  }
  cs.finish();
  for (Buffer& b : scratch) g.release(b);
}

}  // namespace ni::vision
