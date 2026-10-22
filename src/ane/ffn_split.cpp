// GPU + Neural Engine FFN split (see ffn_split.h).
#include "ane/ffn_split.h"
#include "util/log.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>

#include <mach/mach.h>
#include <sys/sysctl.h>

namespace ni::ane {
namespace {

constexpr uint32_t kBlock = 512;  // intermediate rotation block (and the channel unit)

struct RotP {
  uint32_t T, D;
};
struct PackP {
  uint32_t T, D, ch0, stride;
};
struct WP {
  uint32_t row0, col0, width, wpr, gpr, stride, sstride;
};
struct JoinP {
  uint32_t T, D, ystride;
};

Pipeline pipe(const char* n) { return Gpu::get().pipeline(n); }

// The staging kernels of a projection's format (`base`_b128 / _b512 for 4-bit group 64, `base`_<fmt>_<blk> from
// wfmt.metal's library for the others) and its row strides in words / scale groups.
Pipeline wpipe(const char* base, const Proj& P, const char* blk) {
  std::string n = base;
  if (P.bits == 16) n += "_f16";
  else if (P.bits != 4 || P.gs != 64) n += "_f" + std::to_string(P.bits) + "g" + std::to_string(P.gs);
  return Gpu::get().pipeline(n + "_" + blk);
}
uint32_t wpr(const Proj& P) { return P.bits == 16 ? P.K / 2 : P.K * uint32_t(P.bits) / 32; }
uint32_t gpr(const Proj& P) { return P.bits == 16 ? 1u : P.K / uint32_t(P.gs); }

}  // namespace

// System-wide wired memory (MB): the ANE service's buffers are wired kernel memory that no
// process footprint shows, so the split reports the change around its program's load and first
// evaluations (other processes add noise).
double wired_mb() {
  vm_statistics64_data_t v;
  mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&v, &n) != KERN_SUCCESS) return 0;
  return double(v.wire_count) * double(vm_kernel_page_size) / 1e6;
}

namespace {

std::string fp16lit(double v) {
  char t[48];
  snprintf(t, sizeof t, "fp16(%a)", v);
  return t;
}
std::string dims(uint64_t r, uint64_t w) { return "[1, 1, " + std::to_string(r) + ", " + std::to_string(w) + "]"; }
std::string tens(const char* ty, uint64_t r, uint64_t w) { return std::string("tensor<") + ty + ", " + dims(r, w) + ">"; }

std::vector<float> make_signs() {
  std::mt19937 g(20261016);
  std::vector<float> s(kBlock);
  for (float& v : s) v = (g() & 1) ? -1.f : 1.f;
  return s;
}

}  // namespace

FfnSplit::FfnSplit(const FfnConfig& c, std::vector<FfnLayer> layers) : layers_(std::move(layers)), cfg_(c) {
  const double t0 = now_ms();
  if (const std::string& why = unavailable(); !why.empty()) throw std::runtime_error(why);
  if (layers_.empty()) throw std::runtime_error("no layers");
  hidden_ = layers_[0].gate.K;
  inter_ = layers_[0].gate.N;
  for (const FfnLayer& L : layers_)
    if (L.gate.K != hidden_ || L.up.K != hidden_ || L.gate.N != inter_ || L.up.N != inter_ || L.down.N != hidden_ ||
        L.down.K != inter_)
      throw std::runtime_error("FFN layers differ in shape");
  if (hidden_ % kSeg) throw std::runtime_error("hidden size is not whole 2560-channel segments");
  // ane_rotate: the hidden's 128-blocks split evenly over at most 8 simdgroups, at most 8 blocks each
  // (27B: 40 blocks, 8 x 5; 4B: 20 blocks, 5 x 4)
  rot_sg_ = rotate_simdgroups(hidden_);
  if (!rot_sg_) throw std::runtime_error("hidden size ane_rotate does not take");
  if (inter_ % kUnit || inter_ < 2 * kUnit) throw std::runtime_error("intermediate size is not whole 512-channel units");
  if (c.units < 1 || c.units > inter_ / kUnit)
    throw std::runtime_error("units must be in 1.." + std::to_string(inter_ / kUnit));
  units_ = c.units;
  ane_ = units_ * kUnit;
  gpu_ = inter_ - ane_;
  for (uint32_t b = 0; b < ane_; b += kSeg) dseg_.push_back(std::min(kSeg, ane_ - b));
  min_rows_ = std::clamp<uint32_t>(c.min_rows, 1, kMaxRows);
  fn_rows_ = c.fn_rows;
  if (fn_rows_.empty())
    for (uint32_t r = std::max<uint32_t>(128, (min_rows_ + 127) / 128 * 128); r <= kMaxRows; r += 128) fn_rows_.push_back(r);
  std::sort(fn_rows_.begin(), fn_rows_.end());
  if (fn_rows_.back() != kMaxRows) fn_rows_.push_back(kMaxRows);
  for (uint32_t r : fn_rows_)
    if (r % 32 || r > kMaxRows) throw std::runtime_error("function rows must be multiples of 32 up to 2048");

  if (c.compile_only) {
    compile_program_only();
    startup_ms_ = now_ms() - t0;
    return;
  }
  // Everything below allocates: a failure (e.g. the ANE refuses the program) must free what was
  // made so far, since the caller falls back to the GPU and keeps running.
  try {
    init_resources();
  } catch (...) {
    free_all();
    throw;
  }
  startup_ms_ = now_ms() - t0;
  // NI_ANE_KEEP_S (default 300): see keep_loop. 0 turns the keepalive off; NI_ANE_KEEP_MS sets its period.
  keep_window_s_ = getenv("NI_ANE_KEEP_S") ? atof(getenv("NI_ANE_KEEP_S")) : 300;
  if (getenv("NI_ANE_KEEP_MS")) keep_period_ms_ = std::max(100.0, atof(getenv("NI_ANE_KEEP_MS")));
  last_use_ = now_ms();
  if (keep_window_s_ > 0) keep_ = std::thread([this, st = keep_st_, prog = prog_.get(), b = bind_[0][0][0]] { keep_loop(st, prog, b); });
}

void FfnSplit::init_resources() {
  const FfnConfig& c = cfg_;
  // ---- memory ----
  Gpu& g = Gpu::get();
  const uint32_t nin = hidden_ / kSeg;
  auto surf = [&](uint32_t r, uint32_t w, bool f16) {
    Surface s = make_surface(r, w, f16);
    bytes_ += s.bytes;
    return s;
  };
  if (c.lanes < 1 || c.lanes > 2) throw std::runtime_error("lanes must be 1 or 2");
  for (int l = 0; l < c.lanes; ++l) {
    for (uint32_t k = 0; k < nin; ++k) lane_[l].x.push_back(surf(kSeg, kMaxRows, false));
    lane_[l].tx = surf(1, kMaxRows, true);
    lane_[l].y = surf(hidden_ + 1, kMaxRows, true);
  }
  for (Set& s : set_) {
    for (uint32_t k = 0; k < nin; ++k) {
      s.wg.push_back(surf(ane_, kSeg, false));
      s.wu.push_back(surf(ane_, kSeg, false));
    }
    for (uint32_t d : dseg_) s.wd.push_back(surf(hidden_, d, false));
    s.sg = surf(ane_, 1, true);
    s.su = surf(ane_, 1, true);
    s.sd = surf(hidden_, 1, true);
  }
  rot_ = g.alloc(size_t(kMaxRows) * hidden_ * 2, "ane rotated input");
  signs_ = g.alloc(kBlock * 4, "ane signs");
  status_ = g.alloc(16, "ane status");
  rs_ = g.alloc(layers_.size() * (2 * ane_ + hidden_) * 2, "ane row scales");
  bytes_ += rot_.size + signs_.size + status_.size + rs_.size;
  signs_host_ = make_signs();
  memcpy(signs_.ptr(), signs_host_.data(), kBlock * 4);
  memset(status_.ptr(), 0, 16);
  g.make_resident({rot_, signs_, status_, rs_});
  {
    std::vector<Buffer> sb;
    for (int l = 0; l < c.lanes; ++l) {
      for (auto& s : lane_[l].x) sb.push_back(s.buf);
      sb.push_back(lane_[l].tx.buf);
      sb.push_back(lane_[l].y.buf);
    }
    for (Set& s : set_) {
      for (auto* v : {&s.wg, &s.wu, &s.wd})
        for (auto& x : *v) sb.push_back(x.buf);
      sb.push_back(s.sg.buf);
      sb.push_back(s.su.buf);
      sb.push_back(s.sd.buf);
    }
    g.make_resident(sb);
  }

  // ---- per-row int8 scales (cached on disk) ----
  const double t1 = now_ms();
  compute_row_scales();
  const double scales_ms = now_ms() - t1;

  // ---- the program: one function per row count ----
  std::string mil;
  std::vector<uint8_t> blob;
  program_source(mil, blob);
  const double t2 = now_ms();
  wired0_ = wired_mb();
  prog_ = std::make_unique<Program>(mil, blob, "nace-infer-ane");
  wired_load_ = wired_mb() - wired0_;
  const double prog_ms = now_ms() - t2;

  // ---- bindings: inputs in each procedure's order, per weight set and lane ----
  struct In {
    std::string name;
    const Surface* s[2];
  };
  for (int lane = 0; lane < c.lanes; ++lane) {
    const Lane& La = lane_[lane];
    std::vector<In> ins;
    for (uint32_t k = 0; k < nin; ++k) ins.push_back({"x" + std::to_string(k), {&La.x[k], &La.x[k]}});
    ins.push_back({"tx", {&La.tx, &La.tx}});
    for (uint32_t k = 0; k < nin; ++k) {
      ins.push_back({"wg" + std::to_string(k), {&set_[0].wg[k], &set_[1].wg[k]}});
      ins.push_back({"wu" + std::to_string(k), {&set_[0].wu[k], &set_[1].wu[k]}});
    }
    ins.push_back({"sg", {&set_[0].sg, &set_[1].sg}});
    ins.push_back({"su", {&set_[0].su, &set_[1].su}});
    for (size_t i = 0; i < dseg_.size(); ++i) ins.push_back({"wd" + std::to_string(i), {&set_[0].wd[i], &set_[1].wd[i]}});
    ins.push_back({"sd", {&set_[0].sd, &set_[1].sd}});
    for (uint32_t r : fn_rows_) {
      const int p = prog_->proc("ffn" + std::to_string(r));
      if (p < 0) throw std::runtime_error("ANE program lacks ffn" + std::to_string(r));
      for (int s = 0; s < 2; ++s) {
        std::vector<const Surface*> v;
        for (const std::string& n : prog_->inputs(p)) {
          auto it = std::find_if(ins.begin(), ins.end(), [&](const In& x) { return x.name == n; });
          if (it == ins.end()) throw std::runtime_error("ANE program has an unknown input " + n);
          v.push_back(it->s[s]);
        }
        bind_[s][lane].push_back(prog_->bind(p, v, La.y));
      }
    }
  }
  h_ = std::make_unique<Handoff>(2.0);
  char buf[256];
  snprintf(buf, sizeof buf, "row scales %.0f ms, program %s %.0f ms + load %.0f ms (%zu functions, MIL %zu KB)",
           scales_ms, prog_->was_cached ? "cached" : "compiled", prog_->compile_ms, prog_->load_ms, fn_rows_.size(),
           mil.size() / 1024);
  note_ = buf;
  (void)prog_ms;
}

FfnSplit::~FfnSplit() {
  {
    std::lock_guard lk(keep_st_->m);
    keep_st_->stop = true;
  }
  keep_st_->cv.notify_all();
  if (keep_.joinable()) {
    // Program::evaluate has no timeout. If the keepalive is wedged in it, do not hang the process on exit or model
    // unload: detach the thread and leak what its evaluation uses rather than free it underneath.
    const double t0 = now_ms();
    while (!keep_st_->done && now_ms() - t0 < 2000) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (keep_st_->done) {
      keep_.join();
    } else {
      fprintf(stderr, "ANE split: the keepalive evaluation did not return; leaking the program and its surfaces\n");
      keep_.detach();
      (void)prog_.release();
      for (auto& b : bind_)
        for (auto& l : b) (void)new std::vector<std::shared_ptr<Program::Binding>>(std::move(l));
      (void)h_.release();
      return;
    }
  }
  free_all();
}

void FfnSplit::keep_loop(std::shared_ptr<KeepState> st, Program* prog, std::shared_ptr<Program::Binding> bind) {
  // After the evaluation returns, only `st` is touched before the stop check: the FfnSplit may be gone by then.
  const double period_ms = keep_period_ms_, window_ms = keep_window_s_ * 1000;
  for (;;) {
    {
      std::unique_lock lk(st->m);
      st->cv.wait_for(lk, std::chrono::duration<double, std::milli>(period_ms), [&] { return st->stop; });
      if (st->stop) break;
    }
    if (h_->retired()) break;  // the Neural Engine misbehaved: leave it alone
    std::lock_guard ev(st->eval_mu);  // begin_chunk sets in_chunk_ and then takes this lock: check after taking it
    if (in_chunk_ || fn_ >= 0) continue;
    const double idle_ms = now_ms() - last_use_;
    if (idle_ms < period_ms || idle_ms > window_ms) continue;  // between chunks of a request, or asleep for good
    try {
      prog->evaluate(*bind);
    } catch (...) {
    }
  }
  st->done = true;
}

uint32_t FfnSplit::rotate_simdgroups(uint32_t hidden) {
  if (hidden % kSeg) return 0;
  for (uint32_t g = 8; g >= 1; --g)
    if ((hidden / 128) % g == 0 && hidden / 128 / g <= 8) return g;
  return 0;
}

void FfnSplit::program_source(std::string& mil, std::vector<uint8_t>& blob) {
  mil = "program(1.3)\n{\n";
  for (uint32_t r : fn_rows_) mil += function(r);
  mil += "}\n";
  std::vector<_Float16> rot(size_t(ane_) * kBlock);
  const float nrm = 1.f / std::sqrt(float(kBlock));
  for (uint32_t ch = 0; ch < ane_; ++ch)
    for (uint32_t i = 0; i < kBlock; ++i) {
      const uint32_t o = ch % kBlock;
      const float h = (std::popcount(o & i) & 1) ? -1.f : 1.f;
      rot[size_t(ch) * kBlock + i] = _Float16(signs_host_[o] * h * nrm);
    }
  blob = fp16_blob(rot.data(), rot.size());
}

// FfnConfig::compile_only: the same MIL and blob as init_resources (so the same daemon cache key),
// generated from layout-only surfaces; compiles (unless the daemon already holds it), loads, and
// unloads again. No Metal or IOSurface allocation, so it can run on a background thread.
void FfnSplit::compile_program_only() {
  const uint32_t nin = hidden_ / kSeg;
  Lane& L = lane_[0];
  for (uint32_t k = 0; k < nin; ++k) L.x.push_back(surface_layout(kSeg, kMaxRows, false));
  L.tx = surface_layout(1, kMaxRows, true);
  L.y = surface_layout(hidden_ + 1, kMaxRows, true);
  Set& S = set_[0];
  for (uint32_t k = 0; k < nin; ++k) {
    S.wg.push_back(surface_layout(ane_, kSeg, false));
    S.wu.push_back(surface_layout(ane_, kSeg, false));
  }
  for (uint32_t d : dseg_) S.wd.push_back(surface_layout(hidden_, d, false));
  S.sg = surface_layout(ane_, 1, true);
  S.su = surface_layout(ane_, 1, true);
  S.sd = surface_layout(hidden_, 1, true);
  signs_host_ = make_signs();
  std::string mil;
  std::vector<uint8_t> blob;
  program_source(mil, blob);
  {
    Program p(mil, blob, "nace-infer-ane");
    note_ = p.was_cached ? "program was cached" : "program compiled in " + std::to_string(int(p.compile_ms)) + " ms";
  }
  L = Lane{};
  S = Set{};
}

void FfnSplit::free_all() {
  h_.reset();
  for (auto& b : bind_)
    for (auto& l : b) l.clear();
  prog_.reset();
  for (Lane& l : lane_) {
    for (auto& s : l.x) free_surface(s);
    if (l.tx.ref) free_surface(l.tx);
    if (l.y.ref) free_surface(l.y);
  }
  for (Set& s : set_) {
    for (auto* v : {&s.wg, &s.wu, &s.wd})
      for (auto& x : *v) free_surface(x);
    free_surface(s.sg);
    free_surface(s.su);
    free_surface(s.sd);
  }
  Gpu& g = Gpu::get();
  g.release(rot_);
  g.release(signs_);
  g.release(status_);
  g.release(rs_);
}

// The ANE's share of the FFN over R tokens (layout [1,1,channels,tokens], tokens innermost).
// int8 values are read as code/128 so fp16 sums cannot overflow; per-row weight scales and the
// per-token input scale carry the magnitude back. silu(g) = g/2 (1 + tanh(g/2)) (the
// ANE's sigmoid is much less accurate for small gate values). The intermediate is rotated in
// 512-blocks (a grouped 1x1 conv against the program's only constant) and quantized per token;
// the output is down's partial before its token scales plus a last row holding the intermediate's
// per-token scale, applied with the input's by the GPU join in fp32. Up's token scale is left out
// (applied in the join): the true output is partial * hscale * tx.
std::string FfnSplit::function(uint32_t R) const {
  const uint32_t C = ane_, nseg = hidden_ / kSeg;
  std::string par, b;
  auto line = [&](const std::string& t) { b += "        " + t + ";\n"; };
  auto f16 = [&](const std::string& n, uint64_t r, uint64_t w, const std::string& e) {
    line(tens("fp16", r, w) + " " + n + " = " + e);
  };
  auto input = [&](const std::string& name, const Surface& s, uint32_t rows, uint32_t width) {
    par += (par.empty() ? "" : ", ") + s.buffer_type(rows, width) + " " + name;
    line(tens(s.mil_type().c_str(), rows, width) + " " + name + "_t = tensor_buffer_to_tensor<ios17>(input = " + name +
         ")");
  };
  for (uint32_t k = 0; k < nseg; ++k) input("x" + std::to_string(k), lane_[0].x[k], kSeg, R);
  input("tx", lane_[0].tx, 1, R);
  for (uint32_t k = 0; k < nseg; ++k) {
    input("wg" + std::to_string(k), set_[0].wg[k], C, kSeg);
    input("wu" + std::to_string(k), set_[0].wu[k], C, kSeg);
  }
  input("sg", set_[0].sg, C, 1);
  input("su", set_[0].su, C, 1);
  for (size_t i = 0; i < dseg_.size(); ++i) input("wd" + std::to_string(i), set_[0].wd[i], hidden_, dseg_[i]);
  input("sd", set_[0].sd, hidden_, 1);

  const std::string unit = fp16lit(1.0 / 128);
  auto matmul = [&](const std::string& n, const std::string& w, const std::string& x, uint64_t r, uint64_t k) {
    f16(w + "_d", r, k, "dequantize(input = " + w + "_t, scale = " + unit + ")");
    line(tens("fp16", r, R) + " " + n + " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + w +
         "_d, y = " + x + ")");
  };
  auto sum = [&](const std::string& p, size_t n, uint64_t r) {
    std::string t = p + "0";
    for (size_t k = 1; k < n; ++k) {
      std::string nx = p + "_s" + std::to_string(k);
      f16(nx, r, R, "add(x = " + t + ", y = " + p + std::to_string(k) + ")");
      t = nx;
    }
    return t;
  };
  for (uint32_t k = 0; k < nseg; ++k)
    f16("x" + std::to_string(k) + "_d", kSeg, R, "dequantize(input = x" + std::to_string(k) + "_t, scale = " + unit + ")");
  for (const char* p : {"g", "u"}) {
    for (uint32_t k = 0; k < nseg; ++k)
      matmul(std::string(p) + "m" + std::to_string(k), std::string("w") + p + std::to_string(k),
             "x" + std::to_string(k) + "_d", C, kSeg);
    f16(std::string(p) + "s", C, R, "mul(x = " + sum(std::string(p) + "m", nseg, C) + ", y = s" + p + "_t)");
  }
  f16("gt", C, R, "mul(x = gs, y = tx_t)");
  f16("gh", C, R, "mul(x = gt, y = " + fp16lit(0.5) + ")");
  f16("th", C, R, "tanh(x = gh)");
  f16("tp", C, R, "add(x = th, y = " + fp16lit(1.0) + ")");
  f16("si", C, R, "mul(x = gh, y = tp)");
  f16("h", C, R, "mul(x = si, y = us)");
  const std::string cs = std::to_string(C), rs = std::to_string(R);
  line("tensor<fp16, [1, " + cs + ", 1, " + rs + "]> h4 = reshape(x = h, shape = tensor<int32, [4]>([1, " + cs + ", 1, " +
       rs + "]))");
  line("tensor<fp16, [" + cs + ", 512, 1, 1]> rot = const()[name = string(\"rot\"), val = tensor<fp16, [" + cs +
       ", 512, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(64)))]");
  line("tensor<fp16, [1, " + cs + ", 1, " + rs + "]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32(" +
       std::to_string(C / kBlock) + "), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), " +
       "strides = tensor<int32, [2]>([1, 1]), weight = rot, x = h4)");
  f16("hr", C, R, "reshape(x = hr4, shape = tensor<int32, [4]>(" + dims(C, R) + "))");
  f16("ha", C, R, "abs(x = hr)");
  f16("pk", 1, R, "reduce_max(x = ha, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))");
  f16("fl", 1, R, "maximum(x = pk, y = " + fp16lit(0x1p-9) + ")");
  f16("iv", 1, R, "real_div(x = " + fp16lit(127.0) + ", y = fl)");
  f16("hs", C, R, "mul(x = hr, y = iv)");
  line(tens("int8", C, R) + " hq = quantize(input = hs, scale = fp16(1), output_dtype = string(\"int8\"))");
  f16("hd", C, R, "dequantize(input = hq, scale = " + unit + ")");
  f16("hsc", 1, R, "mul(x = fl, y = " + fp16lit(128.0 / 127.0) + ")");
  uint32_t begin = 0;
  for (size_t i = 0; i < dseg_.size(); ++i) {
    const uint32_t w = dseg_[i];
    const std::string s = "hd" + std::to_string(i);
    f16(s, w, R, "slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, " + std::to_string(begin) +
                     ", 0]), size = tensor<int32, [4]>(" + dims(w, R) + "))");
    matmul("dm" + std::to_string(i), "wd" + std::to_string(i), s, hidden_, w);
    begin += w;
  }
  f16("ds", hidden_, R, "mul(x = " + sum("dm", dseg_.size(), hidden_) + ", y = sd_t)");
  f16("yt", hidden_ + 1, R, "concat(axis = int32(2), interleave = bool(false), values = (ds, hsc))");
  const Surface& Y = lane_[0].y;
  line(Y.buffer_type(hidden_ + 1, R) +
       " y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
       "strides = tensor<int64, [4]>(" + Y.strides(Y.rows) + "))");
  return "    func ffn" + std::to_string(R) + "<ios18>(" + par + ") {\n" + b + "    } -> (y);\n";
}

Buffer FfnSplit::row_scales(int li, int which) const {
  const size_t per = 2 * ane_ + hidden_;
  const size_t off = (size_t(li) * per + size_t(which) * ane_) * 2;
  return rs_.view(off, size_t(which == 2 ? hidden_ : ane_) * 2);
}

void FfnSplit::compute_row_scales() {
  namespace fs = std::filesystem;
  const size_t n = rs_.size;
  std::string path;
  if (!cfg_.cache_dir.empty() && !cfg_.model_key.empty()) {
    path = cfg_.cache_dir + "/rowscales-" + cfg_.model_key + "-u" + std::to_string(units_) + ".bin";
    std::ifstream f(path, std::ios::binary);
    if (f) {
      char magic[8] = {0};
      uint64_t len = 0;
      f.read(magic, 8);
      f.read((char*)&len, 8);
      if (f && !memcmp(magic, "NIANERS1", 8) && len == n) {
        f.read((char*)rs_.ptr(), n);
        if (f) {
          note_ = "row scales from cache";
          return;
        }
      }
    }
  }
  auto e = Gpu::get().begin();
  for (size_t li = 0; li < layers_.size(); ++li) {
    const FfnLayer& L = layers_[li];
    for (int which = 0; which < 3; ++which) {
      const Proj& P = which == 0 ? L.gate : which == 1 ? L.up : L.down;
      WP w{};
      uint32_t rows;
      if (which < 2) {
        w = WP{gpu_, 0, hidden_, wpr(P), gpr(P), 0, 0};
        rows = ane_;
        e->use(wpipe("ane_row_scale", P, "b128"));
      } else {
        w = WP{0, gpu_, ane_, wpr(P), gpr(P), 0, 0};
        rows = hidden_;
        e->use(wpipe("ane_row_scale", P, "b512"));
      }
      e->buf(0, P.w);
      e->buf(1, P.s);
      e->buf(2, P.b);
      e->buf(3, row_scales(int(li), which));
      e->buf(4, signs_);
      e->val(5, w);
      e->dispatch(rows / 8, 1, 1, 256);
    }
  }
  e->commit_and_wait();
  if (!path.empty()) {
    std::error_code ec;
    fs::create_directories(cfg_.cache_dir, ec);
    const std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    const uint64_t len = n;
    f.write("NIANERS1", 8);
    f.write((const char*)&len, 8);
    f.write((const char*)rs_.ptr(), n);
    f.close();
    if (f) fs::rename(tmp, path, ec);
  }
}

bool FfnSplit::splits(int T) const { return T >= int(min_rows_) && T <= int(kMaxRows) && !h_->retired(); }

int FfnSplit::fn_for(int T) const {
  for (size_t i = 0; i < fn_rows_.size(); ++i)
    if (fn_rows_[i] >= uint32_t(T)) return int(i);
  throw std::logic_error("ANE split: no function of " + std::to_string(T) + " rows");
}

void FfnSplit::begin_chunk(int T) {
  if (T < 1 || T > int(kMaxRows)) throw std::logic_error("ANE split: chunk rows out of range");
  in_chunk_ = true;
  struct Reset {  // a chunk that never starts must not leave the keepalive stopped
    std::atomic<bool>& f;
    bool armed = true;
    ~Reset() { if (armed) f = false; }
  } reset{in_chunk_};
  {
    // Waits out a keepalive evaluation in flight (about 8 ms). Program::evaluate has no timeout, so the wait is bounded:
    // if the Neural Engine is wedged in it, retire the split and the chunk falls back to the GPU alone.
    std::unique_lock lk(keep_st_->eval_mu, std::defer_lock);
    if (!lk.try_lock_for(std::chrono::seconds(2))) h_->retire("the keepalive evaluation did not return");
  }
  fn_ = fn_for(T);
  reset.armed = false;
  T_ = T;
  jobs_ = 0;
  lane_[0].done = lane_[1].done = 0;
  *status_.as<uint32_t>() = 0;
}

void FfnSplit::stage_layer(Encoder& e, int li, int set) {
  const FfnLayer& L = layers_.at(li);
  const Set& S = set_[set];
  for (int which = 0; which < 2; ++which) {
    const Proj& P = which == 0 ? L.gate : L.up;
    e.use(wpipe("ane_stage", P, "b128"));
    for (size_t k = 0; k < S.wg.size(); ++k) {
      const Surface& out = which == 0 ? S.wg[k] : S.wu[k];
      const Surface& so = which == 0 ? S.sg : S.su;
      WP w{gpu_, uint32_t(k * kSeg), kSeg, wpr(P), gpr(P), out.stride, so.stride / 2};
      e.buf(0, P.w);
      e.buf(1, P.s);
      e.buf(2, P.b);
      e.buf(3, row_scales(li, which));
      e.buf(4, out.buf);
      e.buf(5, so.buf);
      e.buf(6, signs_);
      e.val(7, w);
      e.dispatch(ane_ / 8, kSeg / 128, 1, 256);
    }
  }
  e.use(wpipe("ane_stage", L.down, "b512"));
  uint32_t col = gpu_;
  for (size_t i = 0; i < S.wd.size(); ++i) {
    WP w{0, col, dseg_[i], wpr(L.down), gpr(L.down), S.wd[i].stride, S.sd.stride / 2};
    e.buf(0, L.down.w);
    e.buf(1, L.down.s);
    e.buf(2, L.down.b);
    e.buf(3, row_scales(li, 2));
    e.buf(4, S.wd[i].buf);
    e.buf(5, S.sd.buf);
    e.buf(6, signs_);
    e.val(7, w);
    e.dispatch(hidden_ / 8, dseg_[i] / 512, 1, 256);
    col += dseg_[i];
  }
}

void FfnSplit::pack(Encoder& e, int lane, const Buffer& xh, int T) {
  const Lane& La = lane_[lane];
  const uint32_t R = fn_rows_[fn_for(T)], tiles = R / 32;
  e.use(pipe("ane_rotate"));
  e.buf(0, xh);
  e.buf(1, signs_);
  e.buf(2, rot_);
  e.buf(3, La.tx.buf);
  e.val(4, RotP{uint32_t(T), hidden_});
  e.dispatch(R, 1, 1, 32 * rot_sg_);
  e.use(pipe("ane_pack"));
  for (size_t k = 0; k < La.x.size(); ++k) {
    e.buf(0, rot_);
    e.buf(1, La.x[k].buf);
    e.val(2, PackP{uint32_t(T), hidden_, uint32_t(k * kSeg), La.x[k].stride});
    e.dispatch(tiles, kSeg / 32, 1, 32, 8);
  }
}

void FfnSplit::launch(Encoder& e, int lane, int li, int T) {
  const uint64_t ready = h_->next_ready(), done = h_->next_done();
  const bool fault = cfg_.fault_layer == li && !fault_used_;
  if (fault) fault_used_ = true;
  std::shared_ptr<Program::Binding> b = bind_[li & 1][lane][fn_for(T)];
  Program* prog = prog_.get();
  h_->queue(ready, done, [prog, b, fault](void* rev, uint64_t w, void* dev, uint64_t s, Handoff::Report rep) {
    if (fault) return;  // injected fault: never starts, so the watchdog must retire the split
    prog->enqueue(*b, rev, w, dev, s, std::move(rep));
  });
  ++jobs_;
  lane_[lane].done = done;
  e.signal_event(h_->ready_event(), ready);
}

void FfnSplit::join(Encoder& e, int lane, const Buffer& x, int T) {
  Lane& La = lane_[lane];
  if (La.done) e.wait_event(h_->done_event(), La.done);
  La.done = 0;
  e.use(pipe("ane_join"));
  e.buf(0, x);
  e.buf(1, La.y.buf);
  e.buf(2, La.tx.buf);
  e.buf(3, status_);
  e.val(4, JoinP{uint32_t(T), hidden_, La.y.stride / 2});
  e.dispatch((uint32_t(T) + 31) / 32, hidden_ / 32, 1, 32, 8);
}

void FfnSplit::encode_layer(Encoder& e, int li, const Buffer& xh, const Buffer& x,
                            const std::function<void(Encoder&)>& gpu_part, bool ane, bool stage) {
  if (fn_ < 0) throw std::logic_error("ANE split: begin_chunk() first");
  if (stage && li == 0) stage_layer(e, 0, 0);
  pack(e, 0, xh, T_);
  if (ane) launch(e, 0, li, T_);
  gpu_part(e);
  if (stage && li + 1 < int(layers_.size())) stage_layer(e, li + 1, (li + 1) & 1);
  join(e, 0, x, T_);
}

bool FfnSplit::finish_chunk(double* ane_ms) {
  double ms = 0;
  bool ok = h_->finish(&ms);
  if (*status_.as<uint32_t>()) {
    h_->retire("the Neural Engine's output or its scales were not finite");
    ok = false;
  }
  if (ane_ms) *ane_ms = ms;
  fn_ = -1;
  last_use_ = now_ms();
  in_chunk_ = false;
  last_jobs_ = jobs_;
  if (jobs_ && wired_run_ == 0) wired_run_ = wired_mb() - wired0_;
  return ok;
}

void FfnSplit::retire(const std::string& why) { h_->retire(why); }
bool FfnSplit::retired() const { return h_->retired(); }
std::string FfnSplit::reason() const { return h_->reason(); }

void FfnSplit::fill_partial(uint16_t bits) {
  for (int l = 0; l < cfg_.lanes; ++l) {
    uint16_t* p = lane_[l].y.buf.as<uint16_t>();
    std::fill(p, p + lane_[l].y.bytes / 2, bits);
  }
}

void FfnSplit::fill_status(uint32_t v) { *status_.as<uint32_t>() = v; }

double FfnSplit::eval_sync(int set) {
  if (fn_ < 0) throw std::logic_error("ANE split: begin_chunk() first");
  return prog_->evaluate(*bind_[set][0][fn_]);
}

// ---- persisted configuration ----
std::string cache_dir() {
  const char* h = getenv("HOME");
  return std::string(h && *h ? h : "/tmp") + "/.cache/nace-infer-models/ane-cache";
}

std::string model_key(const std::string& dir) {
  namespace fs = std::filesystem;
  std::error_code ec;
  const std::string real = fs::weakly_canonical(dir, ec).string();
  uint64_t h = fnv1a(real.data(), real.size());
  std::vector<std::string> files;
  for (auto& de : fs::directory_iterator(dir, ec))
    if (de.path().extension() == ".safetensors") files.push_back(de.path().string());
  std::sort(files.begin(), files.end());
  for (const std::string& f : files) {
    const uint64_t sz = fs::file_size(f, ec);
    const auto mt = fs::last_write_time(f, ec).time_since_epoch().count();
    h = fnv1a(f.data(), f.size(), h);
    h = fnv1a(&sz, sizeof sz, h);
    h = fnv1a(&mt, sizeof mt, h);
  }
  char hex[17];
  snprintf(hex, sizeof hex, "%016llx", (unsigned long long)h);
  return hex;
}

std::string os_build() {
  char b[64] = {0};
  size_t n = sizeof b;
  if (sysctlbyname("kern.osversion", b, &n, nullptr, 0)) return "unknown";
  return b;
}

std::string machine_key(const std::string& chip, const std::string& model) {
  std::string c = chip;
  for (char& ch : c)
    if (ch == ' ' || ch == '|') ch = '_';
  return c + "|" + os_build() + "|" + model;
}

static std::string calib_path() { return cache_dir() + "/calibration.txt"; }

bool read_calibration(const std::string& key, Calibration& out) {
  std::ifstream f(calib_path());
  std::string line;
  bool found = false;
  while (std::getline(f, line)) {  // the last line for the key wins
    char k[512], tag[32];
    unsigned u = 0, r = 0;
    int used = 0;
    if (sscanf(line.c_str(), "%31s %511s %u %u%n", tag, k, &u, &r, &used) == 4 && !strcmp(tag, "nace-ane-calib-1") &&
        key == k) {
      out.units = u;
      out.min_rows = r;
      out.detail = line.substr(size_t(used));
      found = true;
    }
  }
  return found;
}

bool write_calibration(const std::string& key, const Calibration& c) {
  std::error_code ec;
  std::filesystem::create_directories(cache_dir(), ec);
  std::ofstream f(calib_path(), std::ios::app);
  f << "nace-ane-calib-1 " << key << " " << c.units << " " << c.min_rows << " " << c.detail << "\n";
  return bool(f);
}

size_t split_bytes(uint32_t hidden, uint32_t inter, uint32_t units, int layers, int lanes) {
  (void)inter;
  const size_t nin = hidden / FfnSplit::kSeg, ane = size_t(units) * FfnSplit::kUnit, R = FfnSplit::kMaxRows;
  auto surf = [](size_t rows, size_t width, bool f16) {  // rows padded to 64 bytes, size to 16 KB pages
    const size_t stride = (width * (f16 ? 2 : 1) + 63) / 64 * 64;
    return (rows * stride + 16383) / 16384 * 16384;
  };
  size_t b = 0;
  for (int l = 0; l < lanes; ++l) b += nin * surf(FfnSplit::kSeg, R, false) + surf(1, R, true) + surf(hidden + 1, R, true);
  size_t dseg = 0;
  for (size_t c = 0; c < ane; c += FfnSplit::kSeg) dseg += surf(hidden, std::min<size_t>(FfnSplit::kSeg, ane - c), false);
  b += 2 * (2 * nin * surf(ane, FfnSplit::kSeg, false) + dseg + 2 * surf(ane, 1, true) + surf(hidden, 1, true));
  b += R * hidden * 2 + 512 * 4 + 16 + size_t(layers) * (2 * ane + hidden) * 2;  // rotation scratch, signs, row scales
  return b;
}

uint32_t default_units(const std::string& chip, int cores, uint32_t total, bool wave, bool gpu_mma) {
  // Measured on the M4 Pro (20-core GPU, 16-core ANE, 27B): the wavefront hides the ANE's
  // FFN of one row half under the GPU's mixer of the other, and the GPU stays the bottleneck up to the
  // whole FFN (34 of 34 units: 7.20 s per 2048-row chunk vs 7.40 at 33, 8.40 at 31); the plain split
  // balances at 27 of 34. Per layer and chunk, the ANE takes ~1.94 ms per unit (fixed for the chip family)
  // and the GPU ~100 ms of mixer plus 6.2 ms per unit it keeps, both scaling with 20 / cores. So the GPU
  // stays the bottleneck at the whole FFN up to ~30 cores (MacBook Airs: 8-10), and a bigger GPU (M4 Max:
  // 40 cores) balances at u = (100 + 6.2 x 34) x 20 / cores / (1.94 + 6.2 x 20 / cores) ~ 31 of 34. A
  // larger ANE share also keeps a fanless Air cooler: the ANE draws a fraction of the GPU's power for the
  // same work. M4 and M5 family only (the W8A8 program was verified on the M4's ANE; older ANEs are left to
  // a calibration: ane_calib --wave --write).
  if (!total || (chip.find("M4") == std::string::npos && chip.find("M5") == std::string::npos)) return 0;
  const double c = std::max(1, cores), k = 20.0 / c;
  if (gpu_mma) {
    // gpu_mma: the GPU's share runs on the M5's matrix units (Qwen35's NI_QMT, 4-bit group 64), ~3.5x cheaper per
    // channel than its simdgroup GEMMs. Measured on the M5 Pro (20 cores, 27B, ane_calib, 2048 rows): the ANE
    // ~1.68 ms per unit and the GPU 1.52 ms per unit it keeps; the GPU-alone chunk is ~4.8 s, ~75 ms per layer, ~50
    // of them the FFN, so ~25 ms of mixer. Wavefront balance u = (25 + 1.52 x 34) k / (1.68 + 1.52 k): 24 of 34 at
    // 20 cores (ane_calib --wave: 22-28 within noise at 2.99-3.30 s per chunk, 34 4.0 s); plain split
    // u = 1.52 k U / (1.68 + 1.52 k): 16 of 34 (measured best 16).
    const double g = 1.52 * k, a = 1.68;
    const double frac = std::min(1.0, wave ? (25.0 * k + g * 34) / (a + g) / 34 : g / (a + g));
    return std::max<uint32_t>(1, std::min<uint32_t>(total, uint32_t(std::lround(frac * total))));
  }
  const double u34 = (100.0 + 6.2 * 34) * k / (1.94 + 6.2 * k);  // on the 27B's 34 units
  // plain split: ANE and GPU part concurrent per layer, balanced at u = g U / (a + g), matched to the 27 of 34
  // measured at 20 cores
  const double plain = 27.0 / 34 * (6.2 * k / (1.94 + 6.2 * k)) / (6.2 / (1.94 + 6.2));
  const double frac = std::min(1.0, wave ? u34 / 34 : plain);
  return std::max<uint32_t>(1, std::min<uint32_t>(total, uint32_t(std::lround(frac * total))));
}

}  // namespace ni::ane
