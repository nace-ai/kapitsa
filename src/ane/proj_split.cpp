// A dense prefill projection on the Neural Engine (see proj_split.h).
#include "ane/proj_split.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>

namespace ni::ane {
namespace {

// the FFN split's kernels' parameter blocks (ffn_split.cpp)
struct RotP {
  uint32_t T, D;
};
struct PackP {
  uint32_t T, D, ch0, stride;
};
struct WP {
  uint32_t row0, col0, width, wpr, gpr, stride, sstride;
};
struct UnpackP {
  uint32_t T, N0, N01, ystride;
};

Pipeline pipe(const char* n) { return Gpu::get().pipeline(n); }
Pipeline wpipe(const char* base, const Proj& P) {
  std::string n = base;
  if (P.bits == 16) n += "_f16";
  else if (P.bits != 4 || P.gs != 64) n += "_f" + std::to_string(P.bits) + "g" + std::to_string(P.gs);
  return Gpu::get().pipeline(n + "_b128");
}
uint32_t wpr(const Proj& P) { return P.bits == 16 ? P.K / 2 : P.K * uint32_t(P.bits) / 32; }
uint32_t gpr(const Proj& P) { return P.bits == 16 ? 1u : P.K / uint32_t(P.gs); }

std::string dims(uint64_t r, uint64_t w) { return "[1, 1, " + std::to_string(r) + ", " + std::to_string(w) + "]"; }
std::string tens(const char* ty, uint64_t r, uint64_t w) { return std::string("tensor<") + ty + ", " + dims(r, w) + ">"; }
std::string fp16lit(double v) {
  char t[48];
  snprintf(t, sizeof t, "fp16(%a)", v);
  return t;
}

// the same signs as the FFN split (any fixed signs work; one set keeps one rotation per process)
std::vector<float> make_signs() {
  std::mt19937 g(20261016);
  std::vector<float> s(512);
  for (float& v : s) v = (g() & 1) ? -1.f : 1.f;
  return s;
}

}  // namespace

ProjSplit::ProjSplit(const ProjConfig& c, std::vector<ProjLayer> layers) : cfg_(c), layers_(std::move(layers)) {
  if (const std::string& why = unavailable(); !why.empty()) throw std::runtime_error(why);
  int nset = 0;
  for (const ProjLayer& L : layers_) {
    std::vector<uint32_t> ns;
    uint32_t n = 0;
    for (const Proj& P : L.parts) {
      if (!K_) K_ = P.K;
      if (P.K != K_) throw std::runtime_error("projection parts differ in K");
      if (P.N % 32) throw std::runtime_error("projection rows are not a multiple of 32");
      ns.push_back(P.N);
      n += P.N;
    }
    if (ns.size() > 3) throw std::runtime_error("at most three parts per projection");
    N_ = std::max(N_, n);
    part_n_.push_back(ns);
    set_of_.push_back(ns.empty() ? -1 : (nset++ & 1));
  }
  if (!nset) throw std::runtime_error("no projection layers");
  seg_ = FfnSplit::input_segment(K_);
  rot_sg_ = FfnSplit::rotate_simdgroups(K_);
  if (!seg_ || !rot_sg_) throw std::runtime_error("input size is not supported by the input segments");
  if (c.lanes < 1 || c.lanes > 2) throw std::runtime_error("lanes must be 1 or 2");
  fn_rows_ = c.fn_rows.empty() ? std::vector<uint32_t>{512, 1024, 2048} : c.fn_rows;
  std::sort(fn_rows_.begin(), fn_rows_.end());
  if (fn_rows_.back() != kMaxRows) fn_rows_.push_back(kMaxRows);
  for (uint32_t r : fn_rows_)
    if (r % 32 || r > kMaxRows) throw std::runtime_error("function rows must be multiples of 32 up to 2048");
  try {
    init_resources();
  } catch (...) {
    free_all();
    throw;
  }
}

ProjSplit::~ProjSplit() { free_all(); }

void ProjSplit::free_all() {
  h_.reset();
  for (auto& b : bind_)
    for (auto& l : b) l.clear();
  prog_.reset();
  for (Lane& l : lane_) {
    for (auto& s : l.x) free_surface(s);
    l.x.clear();
    if (l.tx.ref) free_surface(l.tx);
    if (l.y.ref) free_surface(l.y);
  }
  for (Set& s : set_) {
    for (auto& w : s.w) free_surface(w);
    s.w.clear();
    if (s.s.ref) free_surface(s.s);
  }
  Gpu& g = Gpu::get();
  for (Buffer* b : {&rot_, &signs_, &status_, &rs_})
    if (b->mtl) g.release(*b);
}

void ProjSplit::init_resources() {
  Gpu& g = Gpu::get();
  const uint32_t nin = K_ / seg_;
  auto surf = [&](uint32_t r, uint32_t w, bool f16) {
    Surface s = make_surface(r, w, f16);
    bytes_ += s.bytes;
    return s;
  };
  for (int l = 0; l < cfg_.lanes; ++l) {
    for (uint32_t k = 0; k < nin; ++k) lane_[l].x.push_back(surf(seg_, kMaxRows, false));
    lane_[l].tx = surf(1, kMaxRows, true);
    lane_[l].y = surf(N_, kMaxRows, true);
  }
  for (Set& s : set_) {
    for (uint32_t k = 0; k < nin; ++k) s.w.push_back(surf(N_, seg_, false));
    s.s = surf(N_, 1, true);
  }
  size_t nrs = 0;
  for (const auto& ns : part_n_) {
    rs_off_.push_back(nrs);
    for (uint32_t n : ns) nrs += n;
  }
  rot_ = g.alloc(size_t(kMaxRows) * K_ * 2, "ane proj rotated input");
  signs_ = g.alloc(512 * 4, "ane proj signs");
  status_ = g.alloc(16, "ane proj status");
  rs_ = g.alloc(std::max<size_t>(nrs, 1) * 2, "ane proj row scales");
  bytes_ += rot_.size + signs_.size + status_.size + rs_.size;
  signs_host_ = make_signs();
  memcpy(signs_.ptr(), signs_host_.data(), 512 * 4);
  memset(status_.ptr(), 0, 16);
  g.make_resident({rot_, signs_, status_, rs_});
  {
    std::vector<Buffer> sb;
    for (int l = 0; l < cfg_.lanes; ++l) {
      for (auto& s : lane_[l].x) sb.push_back(s.buf);
      sb.push_back(lane_[l].tx.buf);
      sb.push_back(lane_[l].y.buf);
    }
    for (Set& s : set_) {
      for (auto& w : s.w) sb.push_back(w.buf);
      sb.push_back(s.s.buf);
    }
    g.make_resident(sb);
  }
  const double t1 = now_ms();
  compute_row_scales();
  const double scales_ms = now_ms() - t1;

  std::string mil = "program(1.3)\n{\n";
  for (uint32_t r : fn_rows_) mil += function(r);
  mil += "}\n";
  const _Float16 one = 1;
  prog_ = std::make_unique<Program>(mil, fp16_blob(&one, 1), "nace-infer-ane-" + cfg_.tag);

  for (int lane = 0; lane < cfg_.lanes; ++lane) {
    const Lane& La = lane_[lane];
    for (uint32_t r : fn_rows_) {
      const int p = prog_->proc("proj" + std::to_string(r));
      if (p < 0) throw std::runtime_error("ANE program lacks proj" + std::to_string(r));
      for (int s = 0; s < 2; ++s) {
        std::vector<const Surface*> v;
        for (const std::string& n : prog_->inputs(p)) {
          if (n == "tx") v.push_back(&La.tx);
          else if (n == "s") v.push_back(&set_[s].s);
          else if (n[0] == 'x') v.push_back(&La.x.at(std::stoul(n.substr(1))));
          else if (n[0] == 'w') v.push_back(&set_[s].w.at(std::stoul(n.substr(1))));
          else throw std::runtime_error("ANE program has an unknown input " + n);
        }
        bind_[s][lane].push_back(prog_->bind(p, v, La.y));
      }
    }
  }
  h_ = std::make_unique<Handoff>(2.0);
  char buf[256];
  snprintf(buf, sizeof buf, "row scales %.0f ms, program %s %.0f ms + load %.0f ms (%zu functions)", scales_ms,
           prog_->was_cached ? "cached" : "compiled", prog_->compile_ms, prog_->load_ms, fn_rows_.size());
  note_ = buf;
}

// y [N][R] = (sum over input segments of W_k x_k) * s: int8 codes read as code/128 (as the FFN split), the per-row
// weight scales applied here and the per-token input scale by unpack() on the GPU. The tx input is unused by the
// arithmetic but bound, so the lanes' surfaces stay the FFN split's layout.
std::string ProjSplit::function(uint32_t R) const {
  const uint32_t nseg = K_ / seg_;
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
  for (uint32_t k = 0; k < nseg; ++k) input("x" + std::to_string(k), lane_[0].x[k], seg_, R);
  for (uint32_t k = 0; k < nseg; ++k) input("w" + std::to_string(k), set_[0].w[k], N_, seg_);
  input("s", set_[0].s, N_, 1);
  const std::string unit = fp16lit(1.0 / 128);
  for (uint32_t k = 0; k < nseg; ++k) {
    const std::string ks = std::to_string(k);
    f16("x" + ks + "_d", seg_, R, "dequantize(input = x" + ks + "_t, scale = " + unit + ")");
    f16("w" + ks + "_d", N_, seg_, "dequantize(input = w" + ks + "_t, scale = " + unit + ")");
    line(tens("fp16", N_, R) + " m" + ks + " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = w" + ks +
         "_d, y = x" + ks + "_d)");
  }
  std::string t = "m0";
  for (uint32_t k = 1; k < nseg; ++k) {
    const std::string nx = "ms" + std::to_string(k);
    f16(nx, N_, R, "add(x = " + t + ", y = m" + std::to_string(k) + ")");
    t = nx;
  }
  f16("yt", N_, R, "mul(x = " + t + ", y = s_t)");
  const Surface& Y = lane_[0].y;
  line(Y.buffer_type(N_, R) +
       " y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
       "strides = tensor<int64, [4]>(" + Y.strides(Y.rows) + "))");
  return "    func proj" + std::to_string(R) + "<ios18>(" + par + ") {\n" + b + "    } -> (y);\n";
}

Buffer ProjSplit::row_scales(int li, int part) const {
  size_t off = rs_off_.at(li);
  for (int p = 0; p < part; ++p) off += part_n_.at(li)[p];
  return rs_.view(off * 2, size_t(part_n_.at(li)[part]) * 2);
}

void ProjSplit::compute_row_scales() {
  namespace fs = std::filesystem;
  const size_t n = rs_.size;
  std::string path;
  if (!cfg_.cache_dir.empty() && !cfg_.model_key.empty()) {
    path = cfg_.cache_dir + "/rowscales-" + cfg_.tag + "-" + cfg_.model_key + ".bin";
    std::ifstream f(path, std::ios::binary);
    char magic[8] = {0};
    uint64_t len = 0;
    if (f && f.read(magic, 8) && f.read((char*)&len, 8) && !memcmp(magic, "NIANEPR1", 8) && len == n &&
        f.read((char*)rs_.ptr(), n))
      return;
  }
  auto e = Gpu::get().begin();
  for (size_t li = 0; li < layers_.size(); ++li)
    for (size_t p = 0; p < layers_[li].parts.size(); ++p) {
      const Proj& P = layers_[li].parts[p];
      e->use(wpipe("ane_row_scale", P));
      e->buf(0, P.w);
      e->buf(1, P.s);
      e->buf(2, P.b);
      e->buf(3, row_scales(int(li), int(p)));
      e->buf(4, signs_);
      e->val(5, WP{0, 0, K_, wpr(P), gpr(P), 0, 0});
      e->dispatch(P.N / 8, 1, 1, 256);
    }
  e->commit_and_wait();
  if (!path.empty()) {
    std::error_code ec;
    fs::create_directories(cfg_.cache_dir, ec);
    const std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    const uint64_t len = n;
    f.write("NIANEPR1", 8);
    f.write((const char*)&len, 8);
    f.write((const char*)rs_.ptr(), n);
    f.close();
    if (f) fs::rename(tmp, path, ec);
  }
}

int ProjSplit::fn_for(int T) const {
  for (size_t i = 0; i < fn_rows_.size(); ++i)
    if (fn_rows_[i] >= uint32_t(T)) return int(i);
  throw std::logic_error("ANE proj: no function of " + std::to_string(T) + " rows");
}

void ProjSplit::begin_chunk() {
  jobs_ = 0;
  lane_[0].done = lane_[1].done = 0;
  *status_.as<uint32_t>() = 0;
}

void ProjSplit::stage_layer(Encoder& e, int li, int set) {
  const ProjLayer& L = layers_.at(li);
  const Set& S = set_[set];
  uint32_t row = 0;
  for (size_t p = 0; p < L.parts.size(); ++p) {
    const Proj& P = L.parts[p];
    e.use(wpipe("ane_stage", P));
    for (size_t k = 0; k < S.w.size(); ++k) {
      const Surface& out = S.w[k];
      e.buf(0, P.w);
      e.buf(1, P.s);
      e.buf(2, P.b);
      e.buf(3, row_scales(li, int(p)));
      e.buf(4, out.buf.view(size_t(row) * out.stride, size_t(P.N) * out.stride));
      e.buf(5, S.s.buf.view(size_t(row) * S.s.stride, size_t(P.N) * S.s.stride));
      e.buf(6, signs_);
      e.val(7, WP{0, uint32_t(k * seg_), seg_, wpr(P), gpr(P), out.stride, S.s.stride / 2});
      e.dispatch(P.N / 8, seg_ / 128, 1, 256);
    }
    row += P.N;
  }
}

void ProjSplit::pack(Encoder& e, int lane, const Buffer& xh, int T) {
  const Lane& La = lane_[lane];
  const uint32_t R = fn_rows_[fn_for(T)], tiles = R / 32;
  e.use(pipe("ane_rotate"));
  e.buf(0, xh);
  e.buf(1, signs_);
  e.buf(2, rot_);
  e.buf(3, La.tx.buf);
  e.val(4, RotP{uint32_t(T), K_});
  e.dispatch(R, 1, 1, 32 * rot_sg_);
  e.use(pipe("ane_pack"));
  for (size_t k = 0; k < La.x.size(); ++k) {
    e.buf(0, rot_);
    e.buf(1, La.x[k].buf);
    e.val(2, PackP{uint32_t(T), K_, uint32_t(k * seg_), La.x[k].stride});
    e.dispatch(tiles, seg_ / 32, 1, 32, 8);
  }
}

void ProjSplit::launch(Encoder& e, int lane, int li, int T) {
  const uint64_t ready = h_->next_ready(), done = h_->next_done();
  std::shared_ptr<Program::Binding> b = bind_[set_of_.at(li)][lane][fn_for(T)];
  Program* prog = prog_.get();
  h_->queue(ready, done, [prog, b](void* rev, uint64_t w, void* dev, uint64_t s, Handoff::Report rep) {
    prog->enqueue(*b, rev, w, dev, s, std::move(rep));
  }, 1);
  ++jobs_;
  lane_[lane].done = done;
  e.signal_event(h_->ready_event(), ready);
}

void ProjSplit::unpack(Encoder& e, int lane, int li, const std::vector<Buffer>& outs, int T) {
  Lane& La = lane_[lane];
  const std::vector<uint32_t>& ns = part_n_.at(li);
  if (outs.size() != ns.size()) throw std::logic_error("ANE proj: one output per part");
  if (La.done) e.wait_event(h_->done_event(), La.done);
  La.done = 0;
  const uint32_t n0 = ns[0], n01 = ns.size() > 1 ? n0 + ns[1] : n0;
  uint32_t used = 0;
  for (uint32_t n : ns) used += n;
  e.use(pipe("ane_proj_unpack"));
  e.buf(0, La.y.buf);
  e.buf(1, La.tx.buf);
  e.buf(2, outs[0]);
  e.buf(3, outs.size() > 1 ? outs[1] : outs[0]);
  e.buf(4, outs.size() > 2 ? outs[2] : outs[0]);
  e.buf(5, status_);
  e.val(6, UnpackP{uint32_t(T), n0, n01, La.y.stride / 2});
  e.dispatch((uint32_t(T) + 31) / 32, used / 32, 1, 32, 8);
}

bool ProjSplit::finish_chunk(double* ane_ms) {
  double ms = 0;
  bool ok = h_->finish(&ms);
  if (*status_.as<uint32_t>()) {
    h_->retire("the Neural Engine's projection output was not finite");
    ok = false;
  }
  if (ane_ms) *ane_ms = ms;
  last_jobs_ = jobs_;
  return ok;
}

void ProjSplit::retire(const std::string& why) { h_->retire(why); }
bool ProjSplit::retired() const { return h_->retired(); }
std::string ProjSplit::reason() const { return h_->reason(); }

double ProjSplit::eval_sync(int set, int lane, int T) { return prog_->evaluate(*bind_[set][lane][fn_for(T)]); }

size_t proj_bytes(uint32_t K, uint32_t N, size_t scale_rows, int lanes) {
  const size_t seg = FfnSplit::input_segment(K);
  if (!seg) return 0;
  const size_t nin = K / seg, R = ProjSplit::kMaxRows;
  auto surf = [](size_t rows, size_t width, bool f16) {  // as make_surface: rows padded to 64 bytes, size to 16 KB pages
    const size_t stride = (width * (f16 ? 2 : 1) + 63) / 64 * 64;
    return (rows * stride + 16383) / 16384 * 16384;
  };
  size_t b = 0;
  for (int l = 0; l < lanes; ++l) b += nin * surf(seg, R, false) + surf(1, R, true) + surf(N, R, true);
  b += 2 * (nin * surf(N, seg, false) + surf(N, 1, true));
  b += R * K * 2 + 512 * 4 + 16 + std::max<size_t>(scale_rows, 1) * 2;  // rotation scratch, signs, status, row scales
  return b;
}

}  // namespace ni::ane
