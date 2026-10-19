// GPU + Neural Engine split of Nemotron-H's dense prefill work (see nemotron_ane.h).
#include "model/nemotron_ane.h"
#include "model/nemotron_h.h"
#include "util/log.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>

namespace ni {
namespace {

Pipeline pipe(const std::string& n) { return NemotronH::pipeline(n); }

std::string fp16lit(double v) {
  char t[48];
  snprintf(t, sizeof t, "fp16(%a)", v);
  return t;
}
std::string dims(uint64_t r, uint64_t w) { return "[1, 1, " + std::to_string(r) + ", " + std::to_string(w) + "]"; }
std::string tens(const char* ty, uint64_t r, uint64_t w) { return std::string("tensor<") + ty + ", " + dims(r, w) + ">"; }

struct RotP {
  uint32_t T, D;
};
struct PackP {
  uint32_t T, D, ch0, stride;
};
struct WP {  // NhAneWP
  uint32_t row0, col0, width, K, stride, sstride;
  float gs;
};
struct JoinP {  // NhAneJoinP
  uint32_t T, rows, ystride, ld, col0;
};
struct FfnJoinP {  // ane_join's AneJoinParams
  uint32_t T, D, ystride;
};

constexpr uint32_t kRotBlock = 256;  // the shared expert's intermediate rotates in 256-blocks (5376 = 21 x 256)

}  // namespace

struct NhAne::Op {
  uint32_t rows = 0;                  // ANE output rows (in / out: projection rows; ffn: the intermediate)
  std::vector<ane::Surface> w, wd;    // weights per input segment (ffn: up; wd: down per intermediate segment)
  ane::Surface s, sd, y;              // row scales, down row scales (ffn), output
  Buffer rs, rsd;                     // per layer row scales [layers][rows] (ffn down: [layers][hidden])
  std::vector<std::shared_ptr<ane::Program::Binding>> bind;  // per function (fn_rows_)
  std::vector<int> layers;            // model layers with this op
};

NhAne::NhAne(NemotronH& m, const NhAneConfig& c) : m_(m), cfg_(c) {
  const double t0 = now_ms();
  if (const std::string& why = ane::unavailable(); !why.empty()) throw std::runtime_error(why);
  const NhConfig& mc = m.config();
  H_ = uint32_t(mc.hidden);
  DI_ = uint32_t(mc.d_inner);
  NIN_ = uint32_t(mc.in_proj());
  I_ = uint32_t(mc.shared_I);
  if (H_ % kSeg || DI_ % kSeg || H_ / 128 > 64 || DI_ / 128 > 64)
    throw std::runtime_error("hidden / Mamba sizes are not whole 2048-channel segments");
  if (cfg_.ffn && (I_ % kRotBlock || I_ > 16384)) throw std::runtime_error("shared expert width not supported");
  auto rows_of = [](uint32_t n, double share) {
    const uint32_t a = std::min(n, uint32_t(std::lround(n * std::clamp(share, 0.0, 1.0) / 64.0)) * 64);
    return a;
  };
  ain_ = rows_of(NIN_, cfg_.in_share);
  aout_ = rows_of(H_, cfg_.out_share);
  gin_ = NIN_ - ain_;
  gout_ = H_ - aout_;
  if (!ain_ && !aout_ && !cfg_.ffn) throw std::runtime_error("nothing on the Neural Engine");
  for (int i = 0; i < mc.layers; ++i) {
    if (mc.kind[size_t(i)] == 'M') mamba_.push_back(i);
    if (mc.kind[size_t(i)] == 'E') moe_.push_back(i);
  }
  // the staging kernels read NVFP4 or FP8 Mamba projections and an IQ2_XXS or NVFP4 shared expert, one format each
  // for every layer (Bolt: NVFP4 / IQ2_XXS; Super-VL: FP8 / NVFP4)
  auto fmt_of = [](NhFmt f) -> const char* {
    return f == NhFmt::FP4 ? "fp4" : f == NhFmt::FP8 ? "fp8" : f == NhFmt::IQ2 ? "iq2" : "";
  };
  for (int i : mamba_) {
    const NhLayer& l = m.layers()[size_t(i)];
    const std::string fi = fmt_of(l.in.fmt), fo = fmt_of(l.out.fmt);
    if (mfmt_.empty()) mfmt_ = fi;
    if ((fi != "fp4" && fi != "fp8") || fi != fo || fi != mfmt_)
      throw std::runtime_error("the Mamba projections are neither NVFP4 nor FP8");
    if (l.in.l2 || l.out.l2)  // (the staging kernels read layout L0; a dense model on the M1 repacks to L2)
      throw std::runtime_error("the Mamba projections' NVFP4 codes are in layout L2");
  }
  for (int i : moe_) {
    const NhLayer& l = m.layers()[size_t(i)];
    const std::string fu = fmt_of(l.up_s.fmt), fd = fmt_of(l.down_s.fmt);
    if (sfmt_.empty()) sfmt_ = fu;
    if (cfg_.ffn && ((fu != "iq2" && fu != "fp4") || fu != fd || fu != sfmt_))
      throw std::runtime_error("the shared expert is neither IQ2_XXS nor NVFP4");
  }
  fn_rows_ = cfg_.fn_rows.empty() ? std::vector<uint32_t>{512, 1024, 1536, 2048} : cfg_.fn_rows;
  std::sort(fn_rows_.begin(), fn_rows_.end());
  fn_rows_.erase(std::remove_if(fn_rows_.begin(), fn_rows_.end(),
                                [&](uint32_t r) { return r < std::min(cfg_.min_rows, kMaxRows) || r > kMaxRows || r % 32; }),
                 fn_rows_.end());
  if (fn_rows_.empty() || fn_rows_.back() != kMaxRows) fn_rows_.push_back(kMaxRows);
  for (uint32_t b = 0; b < I_; b += kSeg) dseg_.push_back(std::min(kSeg, I_ - b));
  try {
    build();
  } catch (...) {
    for (auto* op : {in_.get(), out_.get(), ffn_.get()})
      if (op) {
        for (auto& s : op->w) ane::free_surface(s);
        for (auto& s : op->wd) ane::free_surface(s);
        for (auto* s : {&op->s, &op->sd, &op->y})
          if (s->ref) ane::free_surface(*s);
      }
    for (auto& s : xs_)
      if (s.ref) ane::free_surface(s);
    for (auto& s : ys_)
      if (s.ref) ane::free_surface(s);
    if (txs_.ref) ane::free_surface(txs_);
    if (tys_.ref) ane::free_surface(tys_);
    throw;
  }
  startup_ms_ = now_ms() - t0;
}

NhAne::~NhAne() {
  h_.reset();
  for (auto* op : {in_.get(), out_.get(), ffn_.get()})
    if (op) op->bind.clear();
  prog_.reset();
  Gpu& g = Gpu::get();
  for (auto* op : {in_.get(), out_.get(), ffn_.get()})
    if (op) {
      for (auto& s : op->w) ane::free_surface(s);
      for (auto& s : op->wd) ane::free_surface(s);
      for (auto* s : {&op->s, &op->sd, &op->y})
        if (s->ref) ane::free_surface(*s);
      g.release(op->rs);
      g.release(op->rsd);
    }
  for (auto& s : xs_)
    if (s.ref) ane::free_surface(s);
  for (auto& s : ys_)
    if (s.ref) ane::free_surface(s);
  if (txs_.ref) ane::free_surface(txs_);
  if (tys_.ref) ane::free_surface(tys_);
  g.release(rot_);
  g.release(signs_);
  g.release(status_);
}

void NhAne::build() {
  Gpu& g = Gpu::get();
  const bool real = !cfg_.compile_only;
  auto surf = [&](uint32_t r, uint32_t w, bool f16) {
    ane::Surface s = real ? ane::make_surface(r, w, f16) : ane::surface_layout(r, w, f16);
    bytes_ += s.bytes;
    return s;
  };
  // inputs: the hidden rows (in_proj, shared expert) and the d_inner rows (out_proj), packed per 2048 channels
  for (uint32_t k = 0; k < H_ / kSeg; ++k) xs_[k] = surf(kSeg, kMaxRows, false);
  txs_ = surf(1, kMaxRows, true);
  if (aout_) {
    for (uint32_t k = 0; k < DI_ / kSeg; ++k) ys_[k] = surf(kSeg, kMaxRows, false);
    tys_ = surf(1, kMaxRows, true);
  }
  auto proj = [&](uint32_t rows, uint32_t K, const std::vector<int>& layers) {
    auto op = std::make_unique<Op>();
    op->rows = rows;
    op->layers = layers;
    for (uint32_t k = 0; k < K / kSeg; ++k) op->w.push_back(surf(rows, kSeg, false));
    op->s = surf(rows, 1, true);
    op->y = surf(rows, kMaxRows, true);
    return op;
  };
  if (ain_) in_ = proj(ain_, H_, mamba_);
  if (aout_) out_ = proj(aout_, DI_, mamba_);
  if (cfg_.ffn) {
    ffn_ = std::make_unique<Op>();
    ffn_->rows = I_;
    ffn_->layers = moe_;
    for (uint32_t k = 0; k < H_ / kSeg; ++k) ffn_->w.push_back(surf(I_, kSeg, false));
    for (uint32_t d : dseg_) ffn_->wd.push_back(surf(H_, d, false));
    ffn_->s = surf(I_, 1, true);
    ffn_->sd = surf(H_, 1, true);
    ffn_->y = surf(H_ + 1, kMaxRows, true);
  }
  // signs of the rotations (one fixed vector: 128-blocks use its first 128, 256-blocks its first 256)
  {
    std::mt19937 gen(20261018);
    signs_host_.resize(512);
    for (float& v : signs_host_) v = (gen() & 1) ? -1.f : 1.f;
  }
  if (real) {
    rot_ = g.alloc(size_t(kMaxRows) * std::max(H_, DI_) * 2, "nh ane rotated input");
    signs_ = g.alloc(512 * 4, "nh ane signs");
    status_ = g.alloc(16, "nh ane status");
    memcpy(signs_.ptr(), signs_host_.data(), 512 * 4);
    memset(status_.ptr(), 0, 16);
    bytes_ += rot_.size + signs_.size + status_.size;
    std::vector<Buffer> res = {rot_, signs_, status_};
    for (auto& s : xs_)
      if (s.ref) res.push_back(s.buf);
    for (auto& s : ys_)
      if (s.ref) res.push_back(s.buf);
    for (auto* s : {&txs_, &tys_})
      if (s->ref) res.push_back(s->buf);
    for (auto* op : {in_.get(), out_.get(), ffn_.get()})
      if (op) {
        for (auto& s : op->w) res.push_back(s.buf);
        for (auto& s : op->wd) res.push_back(s.buf);
        for (auto* s : {&op->s, &op->sd, &op->y})
          if (s->ref) res.push_back(s->buf);
      }
    g.make_resident(res);
    const double t1 = now_ms();
    row_scales();
    note_ = "row scales " + std::to_string(int(now_ms() - t1)) + " ms";
  }
  // the program: one function per op and row count
  std::string mil;
  std::vector<uint8_t> blob;
  program(mil, blob);
  const double t2 = now_ms();
  prog_ = std::make_unique<ane::Program>(mil, blob, "kapitsa-nh-ane");
  note_ += std::string(note_.empty() ? "" : ", ") + "program " +
           (prog_->adopted ? "from the background compile" : prog_->was_cached ? "cached" : "compiled") + " in " +
           std::to_string(int(now_ms() - t2)) + " ms (" + std::to_string(3 * fn_rows_.size()) + " functions, MIL " +
           std::to_string(mil.size() / 1024) + " KB)";
  if (!real) {
    ane::Program::park(std::move(prog_));  // the real build takes it over (the daemon may not keep the compilation)
    return;
  }
  // bindings, by input name
  struct In {
    std::string name;
    const ane::Surface* s;
  };
  auto bind_all = [&](Op& op, const char* fname, const std::vector<In>& ins) {
    for (uint32_t r : fn_rows_) {
      const int p = prog_->proc(std::string(fname) + std::to_string(r));
      if (p < 0) throw std::runtime_error(std::string("ANE program lacks ") + fname + std::to_string(r));
      std::vector<const ane::Surface*> v;
      for (const std::string& n : prog_->inputs(p)) {
        auto it = std::find_if(ins.begin(), ins.end(), [&](const In& x) { return x.name == n; });
        if (it == ins.end()) throw std::runtime_error("ANE program has an unknown input " + n);
        v.push_back(it->s);
      }
      op.bind.push_back(prog_->bind(p, v, op.y));
    }
  };
  if (in_) {
    std::vector<In> ins;
    for (uint32_t k = 0; k < H_ / kSeg; ++k) ins.push_back({"x" + std::to_string(k), &xs_[k]});
    for (uint32_t k = 0; k < H_ / kSeg; ++k) ins.push_back({"w" + std::to_string(k), &in_->w[k]});
    ins.push_back({"s", &in_->s});
    bind_all(*in_, "in", ins);
  }
  if (out_) {
    std::vector<In> ins;
    for (uint32_t k = 0; k < DI_ / kSeg; ++k) ins.push_back({"x" + std::to_string(k), &ys_[k]});
    for (uint32_t k = 0; k < DI_ / kSeg; ++k) ins.push_back({"w" + std::to_string(k), &out_->w[k]});
    ins.push_back({"s", &out_->s});
    bind_all(*out_, "out", ins);
  }
  if (ffn_) {
    std::vector<In> ins;
    for (uint32_t k = 0; k < H_ / kSeg; ++k) ins.push_back({"x" + std::to_string(k), &xs_[k]});
    ins.push_back({"tx", &txs_});
    for (uint32_t k = 0; k < H_ / kSeg; ++k) ins.push_back({"u" + std::to_string(k), &ffn_->w[k]});
    ins.push_back({"su", &ffn_->s});
    for (size_t i = 0; i < dseg_.size(); ++i) ins.push_back({"d" + std::to_string(i), &ffn_->wd[i]});
    ins.push_back({"sd", &ffn_->sd});
    bind_all(*ffn_, "ffn", ins);
  }
  h_ = std::make_unique<ane::Handoff>(2.0);
}

// ---- the program ----
// A projection's ANE rows over R tokens (layout [1, 1, channels, tokens]): y = (sum over the input segments of
// W_k . x_k) * s, int8 codes read as code / 128, s the rows' scales; the GPU join multiplies by the tokens' scales.
std::string NhAne::fn_proj(const char* name, const Op& op, uint32_t R) const {
  const uint32_t C = op.rows, nseg = uint32_t(op.w.size());
  std::string par, b;
  auto line = [&](const std::string& t) { b += "        " + t + ";\n"; };
  auto f16 = [&](const std::string& n, uint64_t r, uint64_t w, const std::string& e) {
    line(tens("fp16", r, w) + " " + n + " = " + e);
  };
  auto input = [&](const std::string& nm, const ane::Surface& s, uint32_t rows, uint32_t width) {
    par += (par.empty() ? "" : ", ") + s.buffer_type(rows, width) + " " + nm;
    line(tens(s.mil_type().c_str(), rows, width) + " " + nm + "_t = tensor_buffer_to_tensor<ios17>(input = " + nm + ")");
  };
  const ane::Surface* xin = std::string(name) == "in" ? xs_ : ys_;
  for (uint32_t k = 0; k < nseg; ++k) input("x" + std::to_string(k), xin[k], kSeg, R);
  for (uint32_t k = 0; k < nseg; ++k) input("w" + std::to_string(k), op.w[k], C, kSeg);
  input("s", op.s, C, 1);
  const std::string unit = fp16lit(1.0 / 128);
  for (uint32_t k = 0; k < nseg; ++k) {
    const std::string ks = std::to_string(k);
    f16("x" + ks + "_d", kSeg, R, "dequantize(input = x" + ks + "_t, scale = " + unit + ")");
    f16("w" + ks + "_d", C, kSeg, "dequantize(input = w" + ks + "_t, scale = " + unit + ")");
    line(tens("fp16", C, R) + " m" + ks + " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = w" + ks +
         "_d, y = x" + ks + "_d)");
  }
  std::string t = "m0";
  for (uint32_t k = 1; k < nseg; ++k) {
    const std::string nx = "ms" + std::to_string(k);
    f16(nx, C, R, "add(x = " + t + ", y = m" + std::to_string(k) + ")");
    t = nx;
  }
  f16("yt", C, R, "mul(x = " + t + ", y = s_t)");
  line(op.y.buffer_type(C, R) + " y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, "
       "1, 1]), strides = tensor<int64, [4]>(" + op.y.strides(op.y.rows) + "))");
  return "    func " + std::string(name) + std::to_string(R) + "<ios18>(" + par + ") {\n" + b + "    } -> (y);\n";
}

// The shared expert over R tokens: u = (sum_k U_k . x_k) * su (the true u is u * tx), h' = relu(u)^2 (the true h is
// h' * tx^2), rotated in 256-blocks (a grouped 1x1 conv against the program's constant) and quantized per token;
// the output is down's partial before the tokens' scales plus a last row holding hscale * tx, so the GPU join
// (ane_join: x += partial * row[H] * tx) gives partial * hscale * tx^2.
std::string NhAne::fn_ffn(uint32_t R) const {
  const Op& op = *ffn_;
  const uint32_t C = I_, nseg = H_ / kSeg;
  std::string par, b;
  auto line = [&](const std::string& t) { b += "        " + t + ";\n"; };
  auto f16 = [&](const std::string& n, uint64_t r, uint64_t w, const std::string& e) {
    line(tens("fp16", r, w) + " " + n + " = " + e);
  };
  auto input = [&](const std::string& nm, const ane::Surface& s, uint32_t rows, uint32_t width) {
    par += (par.empty() ? "" : ", ") + s.buffer_type(rows, width) + " " + nm;
    line(tens(s.mil_type().c_str(), rows, width) + " " + nm + "_t = tensor_buffer_to_tensor<ios17>(input = " + nm + ")");
  };
  for (uint32_t k = 0; k < nseg; ++k) input("x" + std::to_string(k), xs_[k], kSeg, R);
  input("tx", txs_, 1, R);
  for (uint32_t k = 0; k < nseg; ++k) input("u" + std::to_string(k), op.w[k], C, kSeg);
  input("su", op.s, C, 1);
  for (size_t i = 0; i < dseg_.size(); ++i) input("d" + std::to_string(i), op.wd[i], H_, dseg_[i]);
  input("sd", op.sd, H_, 1);
  const std::string unit = fp16lit(1.0 / 128);
  for (uint32_t k = 0; k < nseg; ++k) {
    const std::string ks = std::to_string(k);
    f16("x" + ks + "_d", kSeg, R, "dequantize(input = x" + ks + "_t, scale = " + unit + ")");
    f16("u" + ks + "_d", C, kSeg, "dequantize(input = u" + ks + "_t, scale = " + unit + ")");
    line(tens("fp16", C, R) + " um" + ks + " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = u" + ks +
         "_d, y = x" + ks + "_d)");
  }
  std::string t = "um0";
  for (uint32_t k = 1; k < nseg; ++k) {
    const std::string nx = "ums" + std::to_string(k);
    f16(nx, C, R, "add(x = " + t + ", y = um" + std::to_string(k) + ")");
    t = nx;
  }
  f16("us", C, R, "mul(x = " + t + ", y = su_t)");
  f16("ur", C, R, "relu(x = us)");
  f16("h", C, R, "mul(x = ur, y = ur)");
  const std::string cs = std::to_string(C), rs = std::to_string(R);
  line("tensor<fp16, [1, " + cs + ", 1, " + rs + "]> h4 = reshape(x = h, shape = tensor<int32, [4]>([1, " + cs + ", 1, " +
       rs + "]))");
  line("tensor<fp16, [" + cs + ", " + std::to_string(kRotBlock) + ", 1, 1]> rot = const()[name = string(\"rot\"), val = "
       "tensor<fp16, [" + cs + ", " + std::to_string(kRotBlock) +
       ", 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(64)))]");
  line("tensor<fp16, [1, " + cs + ", 1, " + rs + "]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32(" +
       std::to_string(C / kRotBlock) + "), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), " +
       "strides = tensor<int32, [2]>([1, 1]), weight = rot, x = h4)");
  f16("hr", C, R, "reshape(x = hr4, shape = tensor<int32, [4]>(" + dims(C, R) + "))");
  // with ffn_a16 the intermediate stays fp16 (relu^2 is heavy-tailed: one int8 scale per token for it costs most of
  // the split's error); the down projection then reads hr and the scale row is tx
  const std::string hin = cfg_.ffn_a16 ? "hr" : "hd", hsc = cfg_.ffn_a16 ? "tx_t" : "hsc";
  if (!cfg_.ffn_a16) {
    f16("ha", C, R, "abs(x = hr)");
    f16("pk", 1, R, "reduce_max(x = ha, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))");
    f16("fl", 1, R, "maximum(x = pk, y = " + fp16lit(0x1p-9) + ")");
    f16("iv", 1, R, "real_div(x = " + fp16lit(127.0) + ", y = fl)");
    f16("hs", C, R, "mul(x = hr, y = iv)");
    line(tens("int8", C, R) + " hq = quantize(input = hs, scale = fp16(1), output_dtype = string(\"int8\"))");
    f16("hd", C, R, "dequantize(input = hq, scale = " + unit + ")");
    f16("hsc0", 1, R, "mul(x = fl, y = " + fp16lit(128.0 / 127.0) + ")");
    f16("hsc", 1, R, "mul(x = hsc0, y = tx_t)");
  }
  uint32_t begin = 0;
  for (size_t i = 0; i < dseg_.size(); ++i) {
    const uint32_t w = dseg_[i];
    const std::string is = std::to_string(i);
    f16("hd" + is, w, R, "slice_by_size(x = " + hin + ", begin = tensor<int32, [4]>([0, 0, " + std::to_string(begin) +
                             ", 0]), size = tensor<int32, [4]>(" + dims(w, R) + "))");
    f16("d" + is + "_d", H_, w, "dequantize(input = d" + is + "_t, scale = " + unit + ")");
    line(tens("fp16", H_, R) + " dm" + is + " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = d" + is +
         "_d, y = hd" + is + ")");
    begin += w;
  }
  t = "dm0";
  for (size_t i = 1; i < dseg_.size(); ++i) {
    const std::string nx = "dms" + std::to_string(i);
    f16(nx, H_, R, "add(x = " + t + ", y = dm" + std::to_string(i) + ")");
    t = nx;
  }
  f16("ds", H_, R, "mul(x = " + t + ", y = sd_t)");
  f16("yt", H_ + 1, R, "concat(axis = int32(2), interleave = bool(false), values = (ds, " + hsc + "))");
  line(op.y.buffer_type(H_ + 1, R) + " y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, "
       "[4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>(" + op.y.strides(op.y.rows) + "))");
  return "    func ffn" + std::to_string(R) + "<ios18>(" + par + ") {\n" + b + "    } -> (y);\n";
}

void NhAne::program(std::string& mil, std::vector<uint8_t>& blob) {
  mil = "program(1.3)\n{\n";
  for (uint32_t r : fn_rows_) {
    if (in_) mil += fn_proj("in", *in_, r);
    if (out_) mil += fn_proj("out", *out_, r);
    if (ffn_) mil += fn_ffn(r);
  }
  mil += "}\n";
  // the rotation of the shared expert's intermediate: channel c = 256-block row c % 256, D H / sqrt(256)
  const uint32_t C = ffn_ ? I_ : kRotBlock;
  std::vector<_Float16> rot(size_t(C) * kRotBlock);
  const float nrm = 1.f / std::sqrt(float(kRotBlock));
  for (uint32_t ch = 0; ch < C; ++ch)
    for (uint32_t i = 0; i < kRotBlock; ++i) {
      const uint32_t o = ch % kRotBlock;
      const float h = (std::popcount(o & i) & 1) ? -1.f : 1.f;
      rot[size_t(ch) * kRotBlock + i] = _Float16(signs_host_[o] * h * nrm);
    }
  blob = ane::fp16_blob(rot.data(), rot.size());
}

// ---- row scales: computed on the GPU once per (model, shares), cached on disk ----
void NhAne::row_scales() {
  namespace fs = std::filesystem;
  Gpu& g = Gpu::get();
  const size_t nm = mamba_.size(), ne = moe_.size();
  if (in_) in_->rs = g.alloc(std::max<size_t>(16, nm * ain_ * 2), "nh ane rs in");
  if (out_) out_->rs = g.alloc(std::max<size_t>(16, nm * aout_ * 2), "nh ane rs out");
  if (ffn_) {
    ffn_->rs = g.alloc(std::max<size_t>(16, ne * I_ * 2), "nh ane rs up");
    ffn_->rsd = g.alloc(std::max<size_t>(16, ne * H_ * 2), "nh ane rs down");
  }
  std::vector<Buffer*> bufs;
  for (auto* op : {in_.get(), out_.get(), ffn_.get()})
    if (op) {
      bufs.push_back(&op->rs);
      if (op->rsd) bufs.push_back(&op->rsd);
    }
  for (Buffer* b : bufs) bytes_ += b->size;
  std::string path;
  if (!cfg_.cache_dir.empty() && !cfg_.model_key.empty())
    path = cfg_.cache_dir + "/nh-rowscales-" + cfg_.model_key + "-" + std::to_string(ain_) + (cfg_.in_first ? "f" : "") +
           "-" + std::to_string(aout_) + (ffn_ ? "-ffn" : "") + ".bin";
  if (!path.empty()) {
    std::ifstream f(path, std::ios::binary);
    char magic[8] = {0};
    f.read(magic, 8);
    bool ok = bool(f) && !memcmp(magic, "NHANERS1", 8);
    for (Buffer* b : bufs) {
      uint64_t len = 0;
      f.read((char*)&len, 8);
      ok = ok && f && len == b->size;
      if (ok) f.read((char*)b->ptr(), std::streamsize(b->size));
      ok = ok && bool(f);
    }
    if (ok) {
      note_ = "row scales from cache";
      return;
    }
  }
  auto e = g.begin();
  const auto& L = m_.layers();
  for (size_t j = 0; j < nm; ++j) {
    const NhLayer& l = L[size_t(mamba_[j])];
    if (in_) {
      const WP p{ain0(), 0, H_, H_, 0, 0, l.in.gs};
      e->use(pipe("nh_ane_rs_" + mfmt_ + "_b128"));
      e->buf(0, l.in.q);
      e->buf(1, l.in.s);
      e->buf(2, in_->rs, j * ain_ * 2);
      e->buf(3, signs_);
      e->val(4, p);
      e->dispatch(ain_ / 8, 1, 1, 256);
    }
    if (out_) {
      const WP p{gout_, 0, DI_, DI_, 0, 0, l.out.gs};
      e->use(pipe("nh_ane_rs_" + mfmt_ + "_b128"));
      e->buf(0, l.out.q);
      e->buf(1, l.out.s);
      e->buf(2, out_->rs, j * aout_ * 2);
      e->buf(3, signs_);
      e->val(4, p);
      e->dispatch(aout_ / 8, 1, 1, 256);
    }
  }
  for (size_t j = 0; ffn_ && j < ne; ++j) {
    const NhLayer& l = L[size_t(moe_[j])];
    const WP pu{0, 0, H_, H_, 0, 0, l.up_s.gs};
    e->use(pipe("nh_ane_rs_" + sfmt_ + "_b128"));
    e->buf(0, l.up_s.q);
    e->buf(1, l.up_s.s);
    e->buf(2, ffn_->rs, j * I_ * 2);
    e->buf(3, signs_);
    e->val(4, pu);
    e->dispatch(I_ / 8, 1, 1, 256);
    const WP pd{0, 0, I_, I_, 0, 0, l.down_s.gs};
    e->use(pipe("nh_ane_rs_" + sfmt_ + "_b256"));
    e->buf(0, l.down_s.q);
    e->buf(1, l.down_s.s);
    e->buf(2, ffn_->rsd, j * H_ * 2);
    e->buf(3, signs_);
    e->val(4, pd);
    e->dispatch(H_ / 8, 1, 1, 256);
  }
  e->commit_and_wait();
  if (!path.empty()) {
    std::error_code ec;
    fs::create_directories(cfg_.cache_dir, ec);
    const std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    f.write("NHANERS1", 8);
    for (Buffer* b : bufs) {
      const uint64_t len = b->size;
      f.write((const char*)&len, 8);
      f.write((const char*)b->ptr(), std::streamsize(b->size));
    }
    f.close();
    if (f) fs::rename(tmp, path, ec);
  }
}

// ---- per chunk ----
bool NhAne::splits(int T) const {
  return h_ && !h_->retired() && T >= int(std::min(cfg_.min_rows, kMaxRows)) && T <= int(kMaxRows);
}

int NhAne::fn_for(int T) const {
  for (size_t i = 0; i < fn_rows_.size(); ++i)
    if (fn_rows_[i] >= uint32_t(T)) return int(i);
  throw std::logic_error("NhAne: no function of " + std::to_string(T) + " rows");
}

int NhAne::next_layer(int which, int li) const {
  const std::vector<int>& v = which == 2 ? moe_ : mamba_;
  for (int l : v)
    if (l > li) return l;
  return -1;
}

void NhAne::stage(Encoder& e, int which, int li) {
  if (li < 0) return;
  const NhLayer& l = m_.layers()[size_t(li)];
  if (which == 0 || which == 1) {
    Op& op = which == 0 ? *in_ : *out_;
    const NhMat& W = which == 0 ? l.in : l.out;
    const uint32_t g0 = which == 0 ? ain0() : gout_;
    const std::vector<int>& lv = mamba_;
    const size_t j = size_t(std::find(lv.begin(), lv.end(), li) - lv.begin());
    e.use(pipe("nh_ane_stage_" + mfmt_ + "_b128"));
    for (size_t k = 0; k < op.w.size(); ++k) {
      const WP p{g0, uint32_t(k * kSeg), kSeg, W.K, op.w[k].stride, op.s.stride / 2, W.gs};
      e.buf(0, W.q);
      e.buf(1, W.s);
      e.buf(2, op.rs, j * op.rows * 2);
      e.buf(3, op.w[k].buf);
      e.buf(4, op.s.buf);
      e.buf(5, signs_);
      e.val(6, p);
      e.dispatch(op.rows / 8, kSeg / 128, 1, 256);
    }
    return;
  }
  Op& op = *ffn_;
  const size_t j = size_t(std::find(moe_.begin(), moe_.end(), li) - moe_.begin());
  e.use(pipe("nh_ane_stage_" + sfmt_ + "_b128"));
  for (size_t k = 0; k < op.w.size(); ++k) {
    const WP p{0, uint32_t(k * kSeg), kSeg, H_, op.w[k].stride, op.s.stride / 2, l.up_s.gs};
    e.buf(0, l.up_s.q);
    e.buf(1, l.up_s.s);
    e.buf(2, op.rs, j * I_ * 2);
    e.buf(3, op.w[k].buf);
    e.buf(4, op.s.buf);
    e.buf(5, signs_);
    e.val(6, p);
    e.dispatch(I_ / 8, kSeg / 128, 1, 256);
  }
  e.use(pipe("nh_ane_stage_" + sfmt_ + "_b256"));
  uint32_t col = 0;
  for (size_t i = 0; i < op.wd.size(); ++i) {
    const WP p{0, col, dseg_[i], I_, op.wd[i].stride, op.sd.stride / 2, l.down_s.gs};
    e.buf(0, l.down_s.q);
    e.buf(1, l.down_s.s);
    e.buf(2, op.rsd, j * H_ * 2);
    e.buf(3, op.wd[i].buf);
    e.buf(4, op.sd.buf);
    e.buf(5, signs_);
    e.val(6, p);
    e.dispatch(H_ / 8, dseg_[i] / (2 * 128), 1, 256);
    col += dseg_[i];
  }
}

void NhAne::begin_chunk(Encoder& e, int T) {
  if (T < 1 || T > int(kMaxRows)) throw std::logic_error("NhAne: chunk rows out of range");
  fn_ = fn_for(T);
  T_ = T;
  jobs_ = 0;
  pending_ = 0;
  *status_.as<uint32_t>() = 0;
  if (in_ && !mamba_.empty()) stage(e, 0, mamba_.front());
  if (out_ && !mamba_.empty()) stage(e, 1, mamba_.front());
  if (ffn_ && !moe_.empty()) stage(e, 2, moe_.front());
}

void NhAne::pack(Encoder& e, const Buffer& xh, uint32_t D, int T, bool hidden) {
  const uint32_t R = fn_rows_[size_t(fn_)], blocks = D / 128;
  uint32_t nsg = 8;
  while (nsg > 1 && (blocks % nsg || blocks / nsg > 8)) --nsg;
  ane::Surface* segs = hidden ? xs_ : ys_;
  const ane::Surface& tx = hidden ? txs_ : tys_;
  e.use(pipe("ane_rotate"));
  e.buf(0, xh);
  e.buf(1, signs_);
  e.buf(2, rot_);
  e.buf(3, tx.buf);
  e.val(4, RotP{uint32_t(T), D});
  e.dispatch(R, 1, 1, 32 * nsg);
  e.use(pipe("ane_pack"));
  for (uint32_t k = 0; k < D / kSeg; ++k) {
    e.buf(0, rot_);
    e.buf(1, segs[k].buf);
    e.val(2, PackP{uint32_t(T), D, k * kSeg, segs[k].stride});
    e.dispatch(R / 32, kSeg / 32, 1, 32, 8);
  }
}

void NhAne::launch(Encoder& e, int which, int T) {
  (void)T;
  Op& op = which == 0 ? *in_ : which == 1 ? *out_ : *ffn_;
  const uint64_t ready = h_->next_ready(), done = h_->next_done();
  std::shared_ptr<ane::Program::Binding> b = op.bind[size_t(fn_)];
  ane::Program* prog = prog_.get();
  h_->queue(
      ready, done,
      [prog, b](void* rev, uint64_t w, void* dev, uint64_t s, ane::Handoff::Report rep) {
        prog->enqueue(*b, rev, w, dev, s, std::move(rep));
      },
      which);
  ++jobs_;
  pending_ = done;
  e.signal_event(h_->ready_event(), ready);
}

void NhAne::wait_join(Encoder& e) {
  if (pending_) {
    e.wait_event(h_->done_event(), pending_);
    if (void* t = h_->trace_event()) e.signal_event(t, pending_);  // NI_ANE_TRACE: when the GPU went on
  }
  pending_ = 0;
}

void NhAne::in_launch(Encoder& e, int li, const Buffer& xh, int T) {
  (void)li;
  pack(e, xh, H_, T, true);
  launch(e, 0, T);
}

void NhAne::in_join(Encoder& e, int li, const Buffer& zx, int T) {
  wait_join(e);
  const JoinP p{uint32_t(T), ain_, in_->y.stride / 2, NIN_, ain0()};
  e.use(pipe("nh_ane_join_store"));
  e.buf(0, zx);
  e.buf(1, in_->y.buf);
  e.buf(2, txs_.buf);
  e.buf(3, status_);
  e.val(4, p);
  e.dispatch((uint32_t(T) + 31) / 32, (ain_ + 31) / 32, 1, 32, 8);
  stage(e, 0, next_layer(0, li));
}

void NhAne::out_launch(Encoder& e, int li, const Buffer& yh, int T) {
  (void)li;
  pack(e, yh, DI_, T, false);
  launch(e, 1, T);
}

void NhAne::out_join(Encoder& e, int li, const Buffer& x, int T) {
  wait_join(e);
  const JoinP p{uint32_t(T), aout_, out_->y.stride / 2, H_, gout_};
  e.use(pipe("nh_ane_join_add"));
  e.buf(0, x);
  e.buf(1, out_->y.buf);
  e.buf(2, tys_.buf);
  e.buf(3, status_);
  e.val(4, p);
  e.dispatch((uint32_t(T) + 31) / 32, (aout_ + 31) / 32, 1, 32, 8);
  stage(e, 1, next_layer(1, li));
}

void NhAne::ffn_launch(Encoder& e, int li, const Buffer& xh, int T) {
  (void)li;
  pack(e, xh, H_, T, true);
  launch(e, 2, T);
}

void NhAne::ffn_join(Encoder& e, int li, const Buffer& x, int T) {
  wait_join(e);
  const FfnJoinP p{uint32_t(T), H_, ffn_->y.stride / 2};
  e.use(pipe("ane_join"));
  e.buf(0, x);
  e.buf(1, ffn_->y.buf);
  e.buf(2, txs_.buf);
  e.buf(3, status_);
  e.val(4, p);
  e.dispatch((uint32_t(T) + 31) / 32, H_ / 32, 1, 32, 8);
  stage(e, 2, next_layer(2, li));
}

bool NhAne::finish_chunk(double* ane_ms) {
  double ms = 0;
  bool ok = h_->finish(&ms);
  if (*status_.as<uint32_t>()) {
    h_->retire("the Neural Engine's output or its scales were not finite");
    ok = false;
  }
  if (ane_ms) *ane_ms = ms;
  fn_ = -1;
  last_jobs_ = jobs_;
  return ok;
}

void NhAne::retire(const std::string& why) {
  if (h_) h_->retire(why);
}
bool NhAne::retired() const { return !h_ || h_->retired(); }
std::string NhAne::reason() const { return h_ ? h_->reason() : "not built"; }

}  // namespace ni
