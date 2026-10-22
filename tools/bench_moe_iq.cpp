// The MoE kernels of routed experts in llama.cpp's IQ blocks (src/kernels/wfmt_iq.metal: Qwen GGUF imports) at the
// real shapes of Qwen3.6-35B-A3B (D 2048, I 512, 256 experts, top-8, an 8-bit group-64 shared expert of 512): a CPU
// fp64 reference on the dequantized weights (llama.cpp's dequantization, src/gguf/quant.cpp) for every kernel, and GPU
// time per call.
//   bench_moe_iq [--tensors DIR] [--validate] [--trials N] [--fmt gu:dn] [case ...]
// cases (substrings): "rows R=<r>" (decode / small verify: the per-row gate/up and down, R = 1 2 4), "v8 R=<r>" (the
// 8-row verify: x8_prep2, moe8_plan, moe8i_gu, moe8i_down, moe8_combine; R = 2 4 8), "pf T=<t>" (prefill:
// moe_bucket3, qrmi_gu, qrmi_dn; T = 64 512 2048 4096); default: every case.
//   --tensors SRC  real expert tensors from a GGUF file (SRC ending in .gguf) or a folder of raw tensors as tools
//                  fetch them (SRC/<tensor>.bin, SRC/meta.json): blk.<L>.ffn_{gate,up,down}_exps.weight (and the
//                  shared expert's tensors, made 8-bit group 64 as the importer makes them); else synthetic blocks
//   --layer L      the layer of --tensors (default 0: IQ2_XXS gate/up, IQ2_S down; 39: IQ3_S down)
//   --fmt gu:dn    synthetic formats (iq1s, iq2xxs, iq2s, iq3xxs, iq3s; default iq2xxs:iq2s)
//   --topk K       experts per token (default 8)
//   --shared SI    the shared expert's width when it differs from the routed experts' (Flint: --topk 6 --shared 1024;
//                  the rows and prefill cases: the 8-row verify's kernels take a shared expert as wide as the others)
//   --shared-bits 4  a 4-bit group-64 shared expert (Flint's GGUF imports: the rows kernels' _s4 forms) instead of 8
// The prefill case also checks qrmi_gu_*_f32 (fp32 SwiGLU rows, which Flint stores as scaled half rows).
//   --validate     small shapes, one kernel run per command buffer: run it under MTL_SHADER_VALIDATION=1
//                  MTL_DEBUG_LAYER=1 to catch out-of-bounds access
//   --variants     the tilings of each case side by side (rows: NR x simdgroups; v8: down NF; pf: tile widths),
//                  and each case's kernels timed alone
//   --src FILE     Metal source appended to the library (kernel experiments built on wfmt_iq.metal's helpers)
// Errors: max |got - ref| / max |ref| over the outputs checked; the MMA paths (v8, pf) take their inputs as halves (as
// the model does), so their reference rounds the inputs the same way ("exact" = vs that, "vs fp64" = vs unrounded).
// Timing: variants round-robin per trial, each trial the mean of a command buffer of calls on rotating routings (the
// experts' bytes come from DRAM, as in a model), medians printed.
#include "gguf/gguf.h"
#include "gguf/quant.h"
#include "metal/gpu.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace ni;
namespace gg = ni::gguf;

namespace {

// kernel parameter mirrors (wfmt_iq.metal, verify8.metal, qmm_moe.metal)
struct IqSec {
  uint32_t x1, x2, x3, d;
};
struct MoeIqParams {
  uint32_t D, I, SI, k;
  IqSec gu, dn;
};
struct Moe8IqParams {
  uint32_t K, N, h_stride, hs_stride, k;
  IqSec sec;
};
struct QrmIqParams {
  uint32_t N, K, x_ld, y_ld;
  IqSec sec;
};
struct X8PrepParams {
  uint32_t K, R, x_ld, norm, write_xn;
  float eps;
  uint32_t xn_ld, write_lo, unroll, write_t;
};
struct Moe8PlanParams {
  uint32_t R, k, E, gu_gx, dn_gx, layer, small, sgu_gx, sdn_gx, rlim, split, shs, sep, edn_gx, topk;
};
struct TopkParams {
  uint32_t E, k, norm;
};
struct Moe8CombineParams {
  uint32_t N, k;
};
struct Bucket2Params {
  uint32_t P, k, E, BM, S;
};

std::mt19937 rng(7);
float urand(float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); }

uint16_t f2h(float f) {
  _Float16 h = _Float16(f);
  uint16_t u;
  memcpy(&u, &h, 2);
  return u;
}
float h2f(uint16_t u) {
  _Float16 h;
  memcpy(&h, &u, 2);
  return float(h);
}
double hround(double v) { return double(_Float16(float(v))); }

std::string fmt_name(uint32_t t) { return gg::iq_suffix(t); }
uint32_t fmt_type(const std::string& s) {
  for (uint32_t t : {gg::kIQ1_S, gg::kIQ2_XXS, gg::kIQ2_S, gg::kIQ3_XXS, gg::kIQ3_S})
    if (s == gg::iq_suffix(t)) return t;
  fatal("unknown format %s", s.c_str());
  return 0;
}
size_t block_bytes(uint32_t t) { return gg::iq_block_bytes(t); }

// A stacked IQ matrix [E, N, K]: the GGUF blocks (host, for the reference) and the GPU layout
struct IqMat {
  uint32_t type = 0, E = 0, N = 0, K = 0;
  std::vector<uint8_t> raw;
  Buffer buf;
  gg::IqLayout L;
  IqSec sec() const { return IqSec{uint32_t(L.x1), uint32_t(L.x2), uint32_t(L.x3), uint32_t(L.d)}; }
  size_t row_raw() const { return size_t(K) / 256 * block_bytes(type); }
  void row(uint32_t e, uint32_t n, std::vector<double>& w) const {
    std::vector<float> f(K);
    gg::dequantize(type, raw.data() + (size_t(e) * N + n) * row_raw(), f.data(), K);
    w.assign(f.begin(), f.end());
  }
  void upload() {
    const size_t rows = size_t(E) * N;
    L = gg::iq_layout(type, rows, K);
    buf = Gpu::get().alloc(L.bytes + 256, "iq_mat", false);
    gg::iq_repack(type, raw.data(), rows, K, 0, rows, buf.as<uint8_t>());
  }
};

IqMat synth_iq(uint32_t type, uint32_t E, uint32_t N, uint32_t K) {
  IqMat m;
  m.type = type, m.E = E, m.N = N, m.K = K;
  const size_t nblk = size_t(E) * N * K / 256, bb = block_bytes(type);
  m.raw.resize(nblk * bb);
  std::uniform_int_distribution<int> byte(0, 255);
  for (size_t b = 0; b < nblk; ++b) {
    uint8_t* p = m.raw.data() + b * bb;
    for (size_t i = 2; i < bb; ++i) p[i] = uint8_t(byte(rng));
    // d at about the GGUFs' value scale (IQ3_S's d multiplies 1-15, IQ1_S's 1-15 x 1.125)
    const uint16_t d = f2h(urand(2e-5f, 2.5e-3f) * (type == gg::kIQ3_S ? 4.0f : type == gg::kIQ1_S ? 40.0f : 1.0f));
    memcpy(p, &d, 2);
  }
  m.upload();
  return m;
}

// MLX 8-bit (or 4-bit) group-64 matrix [N, K]
struct Q8Mat {
  uint32_t N = 0, K = 0;
  int bits = 8;
  std::vector<uint8_t> q;
  std::vector<uint16_t> s, b;  // bf16
  Buffer w, sb, bb;
  double at(uint32_t n, uint32_t k) const {
    const size_t g = size_t(n) * (K / 64) + k / 64;
    auto bf = [](uint16_t u) { return double(gg::from_bf16(u)); };
    return bf(s[g]) * q[size_t(n) * K + k] + bf(b[g]);
  }
  void upload() {
    Gpu& g = Gpu::get();
    w = g.alloc(q.size() + 64, "q8_w", false);
    sb = g.alloc(s.size() * 2 + 64, "q8_s", false);
    bb = g.alloc(b.size() * 2 + 64, "q8_b", false);
    if (bits == 4) gg::pack_codes(q.data(), q.size(), 4, w.as<uint32_t>());
    else memcpy(w.ptr(), q.data(), q.size());
    memcpy(sb.ptr(), s.data(), s.size() * 2);
    memcpy(bb.ptr(), b.data(), b.size() * 2);
  }
};

int g_shared_bits = 8;

Q8Mat q8_from(const std::vector<float>& x, uint32_t N, uint32_t K) {
  Q8Mat m;
  m.N = N, m.K = K, m.bits = g_shared_bits;
  m.q.resize(size_t(N) * K);
  m.s.resize(size_t(N) * K / 64);
  m.b.resize(size_t(N) * K / 64);
  for (uint32_t n = 0; n < N; ++n)
    gg::quantize(x.data() + size_t(n) * K, K, m.bits, 64, m.q.data() + size_t(n) * K, m.s.data() + size_t(n) * K / 64,
                 m.b.data() + size_t(n) * K / 64);
  m.upload();
  return m;
}

Q8Mat synth_q8(uint32_t N, uint32_t K) {
  std::vector<float> x(size_t(N) * K);
  for (auto& v : x) v = urand(-0.06f, 0.06f);
  return q8_from(x, N, K);
}

struct Layer {
  uint32_t D = 0, I = 0, E = 0, k = 0, SI = 0;  // SI: the shared expert's width
  IqMat g, u, d;
  Q8Mat sg, su, sd;
};

std::vector<uint8_t> read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) fatal("cannot read %s", p.c_str());
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// layer L of a GGUF file (path ending in .gguf) or of a folder of raw tensors (DIR/<tensor>.bin, DIR/meta.json)
Layer real_layer(const std::string& dir, int L) {
  const bool gguf = dir.size() > 5 && dir.compare(dir.size() - 5, 5, ".gguf") == 0;
  std::unique_ptr<gg::File> gf;
  nlohmann::json meta;
  if (gguf) {
    gf = std::make_unique<gg::File>(dir);
  } else {
    std::ifstream mf(dir + "/meta.json");
    if (!mf) fatal("no %s/meta.json", dir.c_str());
    meta = nlohmann::json::parse(mf);
  }
  const std::string b = "blk." + std::to_string(L) + ".";
  auto has = [&](const std::string& n) { return gguf ? gf->find(b + n) != nullptr : meta.contains(b + n); };
  auto tensor = [&](const std::string& n, std::vector<uint64_t>& ne, uint32_t& type) {
    if (gguf) {
      const gg::Tensor* t = gf->find(b + n);
      if (!t) fatal("%s has no %s", dir.c_str(), (b + n).c_str());
      ne = t->ne;
      type = t->type;
      std::vector<uint8_t> v(t->bytes);
      gf->read(*t, 0, v.data(), v.size());
      return v;
    }
    const auto& m = meta.at(b + n);
    ne = m.at("ne").get<std::vector<uint64_t>>();
    const std::string t = m.at("type");
    type = t == "IQ2_XXS" ? gg::kIQ2_XXS : t == "IQ2_S" ? gg::kIQ2_S : t == "IQ3_S" ? gg::kIQ3_S
           : t == "IQ3_XXS" ? gg::kIQ3_XXS : t == "IQ1_S" ? gg::kIQ1_S
           : t == "Q5_K" ? gg::kQ5_K : t == "Q6_K" ? gg::kQ6_K : t == "Q8_0" ? gg::kQ8_0 : t == "F32" ? gg::kF32 : 999;
    return read_file(dir + "/" + b + n + ".bin");
  };
  Layer l;
  auto iq = [&](const std::string& n) {
    IqMat m;
    std::vector<uint64_t> ne;
    m.raw = tensor(n, ne, m.type);
    if (!gg::iq_gpu_type(m.type)) fatal("%s is not IQ1_S / IQ2_XXS / IQ2_S / IQ3_XXS / IQ3_S", n.c_str());
    m.K = uint32_t(ne[0]), m.N = uint32_t(ne[1]), m.E = uint32_t(ne[2]);
    m.upload();
    return m;
  };
  l.g = iq("ffn_gate_exps.weight");
  l.u = iq("ffn_up_exps.weight");
  l.d = iq("ffn_down_exps.weight");
  l.D = l.g.K, l.I = l.g.N, l.E = l.g.E, l.k = 8, l.SI = l.I;
  auto q8 = [&](const std::string& n, uint32_t N, uint32_t K) {
    if (!has(n)) {
      logi("%s: not in %s, synthetic", n.c_str(), dir.c_str());
      return synth_q8(N, K);
    }
    std::vector<uint64_t> ne;
    uint32_t type = 0;
    const std::vector<uint8_t> raw = tensor(n, ne, type);
    N = uint32_t(ne[1]), K = uint32_t(ne[0]);  // the file's shape (Flint's shared expert is wider than its experts)
    std::vector<float> x(size_t(N) * K);
    gg::dequantize(type, raw.data(), x.data(), x.size());
    return q8_from(x, N, K);
  };
  l.sg = q8("ffn_gate_shexp.weight", l.I, l.D);
  l.su = q8("ffn_up_shexp.weight", l.I, l.D);
  l.sd = q8("ffn_down_shexp.weight", l.D, l.I);
  l.SI = l.sg.N;
  return l;
}

Layer synth_layer(uint32_t D, uint32_t I, uint32_t E, uint32_t k, uint32_t tgu, uint32_t tdn, uint32_t SI) {
  Layer l;
  l.D = D, l.I = I, l.E = E, l.k = k, l.SI = SI;
  l.g = synth_iq(tgu, E, I, D);
  l.u = synth_iq(tgu, E, I, D);
  l.d = synth_iq(tdn, E, D, I);
  l.sg = synth_q8(SI, D);
  l.su = synth_q8(SI, D);
  l.sd = synth_q8(D, SI);
  return l;
}

double silu(double v) { return v / (1.0 + std::exp(-v)); }

// routing: per row k distinct experts, weights (k routed + shared) as moe_topk*_rows writes them
struct Routing {
  std::vector<uint32_t> idx;  // [R][k]
  std::vector<float> wts;     // [R][k + 1]
};
Routing make_routing(uint32_t R, uint32_t E, uint32_t k) {
  Routing r;
  for (uint32_t t = 0; t < R; ++t) {
    std::vector<uint32_t> perm(E);
    for (uint32_t i = 0; i < E; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    float sum = 0;
    std::vector<float> w(k);
    for (uint32_t j = 0; j < k; ++j) w[j] = urand(0.05f, 1.0f), sum += w[j];
    for (uint32_t j = 0; j < k; ++j) {
      r.idx.push_back(perm[j]);
      r.wts.push_back(w[j] / sum);
    }
    r.wts.push_back(urand(0.1f, 0.9f));  // shared expert gate (sigmoid)
  }
  return r;
}

struct Err {
  double max_abs = 0, max_ref = 0, max_abs2 = 0;  // vs the reference; max_abs2: vs the unrounded fp64 one
  void add(double got, double ref, double ref2) {
    max_abs = std::max(max_abs, std::fabs(got - ref));
    max_abs2 = std::max(max_abs2, std::fabs(got - ref2));
    max_ref = std::max(max_ref, std::fabs(ref2));
  }
  double rel() const { return max_ref > 0 ? max_abs / max_ref : max_abs; }
  double rel2() const { return max_ref > 0 ? max_abs2 / max_ref : max_abs2; }
};

bool g_validate = false, g_variants = false;
int g_trials = 7;

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v[v.size() / 2];
}

// One timed variant: enc(e, rotation index) encodes one call
struct Variant {
  std::string name;
  std::function<void(Encoder&, int)> enc;
  std::vector<double> ms;
};

void time_variants(std::vector<Variant>& vs, int calls, int rotations, double bytes_per_call, const char* unit_label) {
  if (g_validate) return;
  Gpu& g = Gpu::get();
  for (auto& v : vs) {  // warm-up
    auto e = g.begin();
    for (int i = 0; i < calls; ++i) v.enc(*e, i % rotations);
    e->commit_and_wait();
  }
  for (int t = 0; t < g_trials; ++t)
    for (auto& v : vs) {
      auto e = g.begin();
      for (int i = 0; i < calls; ++i) v.enc(*e, (t * calls + i) % rotations);
      e->commit_and_wait();
      v.ms.push_back(e->gpu_ms() / calls);
    }
  for (auto& v : vs) {
    const double m = median(v.ms);
    std::vector<double> s = v.ms;
    std::sort(s.begin(), s.end());
    printf("    %-40s %8.4f ms (%.4f-%.4f)", v.name.c_str(), m, s.front(), s.back());
    if (bytes_per_call > 0) printf("  %6.1f %s", bytes_per_call / m / 1e6, unit_label);
    printf("\n");
  }
}

Pipeline pipe(const std::string& n) { return Gpu::get().pipeline(n); }

// ---------------------------------------------------------------------------------------------------------------- rows
void case_rows(const Layer& l, uint32_t R) {
  Gpu& g = Gpu::get();
  const uint32_t D = l.D, I = l.I, SI = l.SI, k = l.k, Imax = std::max(I, SI);
  const int rot = g_validate ? 1 : 16;
  std::vector<Routing> rts;
  for (int i = 0; i < rot; ++i) rts.push_back(make_routing(R, l.E, k));
  Buffer xn = g.alloc(size_t(R) * D * 4), h = g.alloc(size_t(R) * (k + 1) * Imax * 4), x = g.alloc(size_t(R) * D * 4);
  Buffer idx = g.alloc(size_t(rot) * R * k * 4), wts = g.alloc(size_t(rot) * R * (k + 1) * 4);
  for (int i = 0; i < rot; ++i) {
    memcpy(idx.as<uint32_t>() + size_t(i) * R * k, rts[i].idx.data(), size_t(R) * k * 4);
    memcpy(wts.as<float>() + size_t(i) * R * (k + 1), rts[i].wts.data(), size_t(R) * (k + 1) * 4);
  }
  std::vector<float> x0(size_t(R) * D);
  for (auto& v : x0) v = urand(-2.0f, 2.0f);
  memcpy(xn.ptr(), x0.data(), x0.size() * 4);
  MoeIqParams p{D, I, SI, k, l.g.sec(), l.d.sec()};
  const std::string gf = fmt_name(l.g.type), df = fmt_name(l.d.type);
  // gate/up: nr rows per simdgroup, nsg simdgroups; down: dnr rows per threadgroup of k + 1 simdgroups, t16: the
  // lanes of a row share its h (I = 512)
  // BENCH_GU="name:nr:nsg,...": extra gate/up kernels (--src), timed beside the default
  auto enc_gu = [&](Encoder& e, int r, const std::string& name, int nr, uint32_t nsg) {
    e.use(pipe(name));
    e.buf(0, l.g.buf); e.buf(1, l.u.buf);
    e.buf(2, l.sg.w); e.buf(3, l.sg.sb); e.buf(4, l.sg.bb);
    e.buf(5, l.su.w); e.buf(6, l.su.sb); e.buf(7, l.su.bb);
    e.buf(8, xn); e.buf(9, h);
    e.val(10, p);
    e.buf(11, idx, size_t(r) * R * k * 4);
    e.dispatch((Imax + nr * nsg - 1) / (nr * nsg), k + 1, R, 32 * nsg);
  };
  auto enc = [&](Encoder& e, int r, int nr, uint32_t nsg, int dnr, bool t16, int parts = 3) {
    const std::string s4 = g_shared_bits == 4 ? "_s4" : "";
    if (parts & 1) {
      e.use(pipe("moei_gu_rows_" + gf + "_r" + std::to_string(nr) + s4));
      e.buf(0, l.g.buf); e.buf(1, l.u.buf);
      e.buf(2, l.sg.w); e.buf(3, l.sg.sb); e.buf(4, l.sg.bb);
      e.buf(5, l.su.w); e.buf(6, l.su.sb); e.buf(7, l.su.bb);
      e.buf(8, xn); e.buf(9, h);
      e.val(10, p);
      e.buf(11, idx, size_t(r) * R * k * 4);
      e.dispatch((Imax + nr * nsg - 1) / (nr * nsg), k + 1, R, 32 * nsg);
    }
    if (!(parts & 2)) return;
    e.use(pipe("moei_down_rows_" + df + "_r" + std::to_string(dnr) + (t16 ? "_t16" : "") + s4));
    e.buf(0, l.d.buf);
    e.buf(1, l.sd.w); e.buf(2, l.sd.sb); e.buf(3, l.sd.bb);
    e.buf(4, h); e.buf(5, idx, size_t(r) * R * k * 4); e.buf(6, wts, size_t(r) * R * (k + 1) * 4); e.buf(7, x);
    e.val(8, p);
    e.dispatch((D + dnr - 1) / dnr, R, 1, 32 * (k + 1));
  };
  // check (rotation 0): h and x
  struct Cfg {
    int nr;
    uint32_t nsg;
    int dnr;
    bool t16;
  };
  const bool t16 = I == 512 && SI == 512;
  std::vector<Cfg> cfgs = {{2, 2, 8, t16}};  // the model's (Qwen35::enc_moe_iq_rows)
  if (g_validate) cfgs = {{2, 2, 8, t16}, {4, 1, 8, false}};
  if (g_shared_bits == 4) cfgs = {{2, 2, 8, false}};  // the _s4 forms: Flint's (Flint::enc_moe_iq_rows)
  if (g_variants && g_shared_bits == 8) {
    cfgs = {{4, 2, 8, false}, {4, 2, 8, t16}, {2, 2, 8, t16}, {4, 2, 16, t16}, {8, 1, 16, t16}, {8, 2, 16, t16}};
  }
  const Routing& rt = rts[0];
  // reference
  std::vector<double> href(size_t(R) * (k + 1) * Imax), xref(size_t(R) * D, 0.0);
  std::vector<double> wg, wu;
  for (uint32_t t = 0; t < R; ++t)
    for (uint32_t s = 0; s <= k; ++s)
      for (uint32_t n = 0; n < (s == k ? SI : I); ++n) {
        double a = 0, b = 0;
        if (s == k) {
          for (uint32_t c = 0; c < D; ++c) a += l.sg.at(n, c) * x0[size_t(t) * D + c], b += l.su.at(n, c) * x0[size_t(t) * D + c];
        } else {
          const uint32_t e = rt.idx[t * k + s];
          l.g.row(e, n, wg);
          l.u.row(e, n, wu);
          for (uint32_t c = 0; c < D; ++c) a += wg[c] * x0[size_t(t) * D + c], b += wu[c] * x0[size_t(t) * D + c];
        }
        href[(size_t(t) * (k + 1) + s) * Imax + n] = silu(a) * b;
      }
  // down reference on the GPU's own h (so the two checks are independent)
  for (const Cfg& c : cfgs) {
    memset(x.ptr(), 0, x.size);
    auto e = g.begin();
    enc(*e, 0, c.nr, c.nsg, c.dnr, c.t16);
    e->commit_and_wait();
    Err eh, ex;
    const float* hg = h.as<float>();
    for (uint32_t t = 0; t < R; ++t)
      for (uint32_t s = 0; s <= k; ++s)
        for (uint32_t n = 0; n < (s == k ? SI : I); ++n) {
          const size_t i = (size_t(t) * (k + 1) + s) * Imax + n;
          eh.add(hg[i], href[i], href[i]);
        }
    std::vector<double> wd;
    for (uint32_t t = 0; t < R; ++t) {
      std::vector<double> acc(D, 0.0);
      for (uint32_t s = 0; s <= k; ++s) {
        const double w = rt.wts[t * (k + 1) + s];
        const float* hs = hg + (size_t(t) * (k + 1) + s) * Imax;
        for (uint32_t n = 0; n < D; ++n) {
          double a = 0;
          if (s == k) {
            for (uint32_t c2 = 0; c2 < SI; ++c2) a += l.sd.at(n, c2) * hs[c2];
          } else {
            l.d.row(rt.idx[t * k + s], n, wd);
            for (uint32_t c2 = 0; c2 < I; ++c2) a += wd[c2] * hs[c2];
          }
          acc[n] += w * a;
        }
      }
      for (uint32_t n = 0; n < D; ++n) ex.add(x.as<float>()[size_t(t) * D + n], acc[n], acc[n]);
    }
    printf("  rows R=%u %s/%s gu nr %d x %u sg, down nr %d%s: gate/up rel.err %.2e, down rel.err %.2e%s\n", R,
           gf.c_str(), df.c_str(), c.nr, c.nsg, c.dnr, c.t16 ? " t16" : "", eh.rel(), ex.rel(),
           eh.rel() > 1e-5 || ex.rel() > 1e-5 ? "  FAILED" : "");
    if (!g_variants && !g_validate) break;
  }
  // bytes per call: the touched experts (distinct per row) + the shared expert
  const double eb = double(l.g.L.bytes + l.u.L.bytes) / l.E + double(l.d.L.bytes) / l.E;
  const double sb = double(l.sg.q.size() + l.su.q.size() + l.sd.q.size()) * (1.0 + 4.0 / 64);
  auto label = [](const Cfg& c) {
    return "gu " + std::to_string(c.nr) + "x" + std::to_string(c.nsg) + " dn " + std::to_string(c.dnr) +
           (c.t16 ? " t16" : "");
  };
  std::vector<Variant> vs;
  for (const Cfg& c : cfgs) vs.push_back({label(c), [&, c](Encoder& e, int r) { enc(e, r, c.nr, c.nsg, c.dnr, c.t16); }});
  time_variants(vs, 20, rot, R * k * eb + sb, "GB/s (experts read)");
  if (const char* bg = getenv("BENCH_GU")) {  // experimental gate/up kernels against the default, alone
    std::vector<Variant> gv;
    gv.push_back({"gate/up default", [&](Encoder& e, int r) { enc(e, r, cfgs[0].nr, cfgs[0].nsg, cfgs[0].dnr, cfgs[0].t16, 1); }});
    std::string spec = bg;
    for (size_t a = 0; a < spec.size();) {
      size_t b = spec.find(',', a);
      if (b == std::string::npos) b = spec.size();
      const std::string it = spec.substr(a, b - a);
      const size_t c1 = it.find(':'), c2 = it.find(':', c1 + 1);
      const std::string name = it.substr(0, c1);
      const int nr = atoi(it.substr(c1 + 1, c2 - c1 - 1).c_str());
      const uint32_t nsg = uint32_t(atoi(it.substr(c2 + 1).c_str()));
      // check against the default's h
      {
        auto e = g.begin();
        enc(*e, 0, cfgs[0].nr, cfgs[0].nsg, cfgs[0].dnr, cfgs[0].t16, 1);
        e->commit_and_wait();
      }
      std::vector<float> ref(h.as<float>(), h.as<float>() + size_t(R) * (k + 1) * Imax);
      {
        auto e = g.begin();
        enc_gu(*e, 0, name, nr, nsg);
        e->commit_and_wait();
      }
      double md = 0, mr = 0;
      for (size_t i = 0; i < ref.size(); ++i) md = std::max(md, double(std::fabs(h.as<float>()[i] - ref[i]))), mr = std::max(mr, double(std::fabs(ref[i])));
      printf("    %s: max |h - default h| / max |h| = %.2e\n", it.c_str(), md / mr);
      gv.push_back({it, [&, name, nr, nsg](Encoder& e, int r) { enc_gu(e, r, name, nr, nsg); }});
      a = b + 1;
    }
    time_variants(gv, 20, rot, 0, "");
  }
  if (g_variants) {  // the two kernels alone
    std::vector<Variant> parts;
    for (const Cfg& c : cfgs) {
      parts.push_back({"gate/up " + label(c), [&, c](Encoder& e, int r) { enc(e, r, c.nr, c.nsg, c.dnr, c.t16, 1); }});
      parts.push_back({"down " + label(c), [&, c](Encoder& e, int r) { enc(e, r, c.nr, c.nsg, c.dnr, c.t16, 2); }});
    }
    time_variants(parts, 20, rot, 0, "");
  }
}

// ------------------------------------------------------------------------------------------------------------------ v8
void case_v8(const Layer& l, uint32_t R) {
  Gpu& g = Gpu::get();
  const uint32_t D = l.D, I = l.I, k = l.k, E = l.E, tiles = 8 * k + 1;
  const int rot = g_validate ? 1 : 16;
  std::vector<Routing> rts;
  for (int i = 0; i < rot; ++i) rts.push_back(make_routing(R, E, k));
  Buffer x = g.alloc(size_t(8) * D * 4), x0b = g.alloc(size_t(8) * D * 4);
  Buffer table = g.alloc(size_t(D) * 8 * 2), sums = g.alloc(size_t(D) / 64 * 8 * 4);
  Buffer tlo = g.alloc(size_t(D) * 8 * 2), slo = g.alloc(size_t(D) / 64 * 8 * 4);
  Buffer hT = g.alloc(size_t(tiles) * I * 8 * 2), hS = g.alloc(size_t(tiles) * I / 64 * 8 * 4);
  Buffer vt = g.alloc(size_t(tiles) * 4), nt = g.alloc(16), slot = g.alloc(size_t(tiles) * 8 * 4), args = g.alloc(64);
  Buffer out = g.alloc(size_t(8) * k * D * 4), sh = g.alloc(size_t(8) * D * 4), stats = g.alloc(64 * 4),
         sm = g.alloc(320 * 4), sm2 = g.alloc(5 * 72 * 4);
  Buffer idx = g.alloc(size_t(rot) * R * k * 4), wts = g.alloc(size_t(rot) * R * (k + 1) * 4);
  for (int i = 0; i < rot; ++i) {
    memcpy(idx.as<uint32_t>() + size_t(i) * R * k, rts[i].idx.data(), size_t(R) * k * 4);
    memcpy(wts.as<float>() + size_t(i) * R * (k + 1), rts[i].wts.data(), size_t(R) * (k + 1) * 4);
  }
  std::vector<float> x0(size_t(8) * D, 0.0f);
  for (size_t i = 0; i < size_t(R) * D; ++i) x0[i] = urand(-2.0f, 2.0f);
  memcpy(x0b.ptr(), x0.data(), x0.size() * 4);
  const std::string gf = fmt_name(l.g.type), df = fmt_name(l.d.type);
  auto enc = [&](Encoder& e, int r, int nf, int parts = 7) {  // 1 prep + plan, 2 gate/up, 4 down + combine
    const uint32_t cs = 4;
    if (!(parts & 1)) goto gu;
    {
    X8PrepParams xp{D, R, D, 0, 0, 1e-6f, D, 0, 1, 0};
    e.use(pipe("x8_prep2"));
    e.buf(0, x); e.buf(1, x); e.buf(2, table); e.buf(3, sums); e.buf(4, x); e.val(5, xp); e.buf(6, tlo); e.buf(7, slo);
    e.buf(8, tlo);  // the T table: not written (write_t 0)
    e.dispatch(D / 256, 1, 1, 256);
    // every touched expert on one MMA-layout tile (Qwen35::enc_moe8_iq), the routing given (no top-k in the plan)
    Moe8PlanParams pp{R, k, E, I / 64, D / (8 * uint32_t(nf) * cs), 0, 0, 1, 1, 0, 0, 0, 0, 1, 0};
    TopkParams tp{E, k, 1};
    e.use(pipe("moe8_plan"));
    e.buf(0, idx, size_t(r) * R * k * 4); e.buf(1, vt); e.buf(2, nt); e.buf(3, slot); e.buf(4, args);
    e.val(5, pp);
    e.buf(6, stats); e.buf(7, sm); e.buf(8, nt);
    e.buf(9, sm2); e.buf(10, out); e.buf(11, out); e.buf(12, out);  // top-k inputs / outputs: unused (topk 0)
    e.val(13, tp);
    e.dispatch(1, 1, 1, std::max(32u, (E + 31) / 32 * 32));
    }
  gu:
    if (!(parts & 2)) goto dn;
    {
    Moe8IqParams gp{D, I, I * 8, I / 64 * 8, k, l.g.sec()};
    e.use(pipe("moe8i_gu_" + gf));
    e.buf(0, l.g.buf); e.buf(1, l.u.buf);
    e.buf(2, l.sg.w); e.buf(3, l.sg.sb); e.buf(4, l.sg.bb);
    e.buf(5, l.su.w); e.buf(6, l.su.sb); e.buf(7, l.su.bb);
    e.buf(8, table); e.buf(9, sums); e.buf(10, hT); e.buf(11, hS);
    e.val(12, gp);
    e.buf(13, vt); e.buf(14, nt);
    e.dispatch_indirect(args, 0, 512);
    }
  dn:
    if (!(parts & 4)) return;
    Moe8IqParams dp{I, D, I * 8, I / 64 * 8, k, l.d.sec()};
    e.use(pipe("moe8i_down_" + df + "_nf" + std::to_string(nf)));
    e.buf(0, l.d.buf);
    e.buf(1, l.sd.w); e.buf(2, l.sd.sb); e.buf(3, l.sd.bb);
    e.buf(4, hT); e.buf(5, hS); e.buf(6, out); e.buf(7, sh);
    e.val(8, dp);
    e.buf(9, vt); e.buf(10, nt); e.buf(11, slot);
    e.dispatch_indirect(args, 12, 32 * cs);
    Moe8CombineParams cp{D, k};
    e.use(pipe("moe8_combine"));
    e.buf(0, out); e.buf(1, sh); e.buf(2, wts, size_t(r) * R * (k + 1) * 4); e.buf(3, x);
    e.val(4, cp);
    e.dispatch((D / 4 + 63) / 64, R, 1, 64);
  };
  std::vector<int> nfs = {2};
  if (g_variants) nfs = {1, 2};
  const Routing& rt = rts[0];
  for (int nf : nfs) {
    memcpy(x.ptr(), x0.data(), x0.size() * 4);
    auto e = g.begin();
    enc(*e, 0, nf);
    e->commit_and_wait();
    // reference: half inputs (the table), h rounded to half (the h table), fp64 sums
    Err ex;
    std::vector<double> wg, wu, wd;
    for (uint32_t t = 0; t < R; ++t) {
      std::vector<double> acc(D, 0.0), acc2(D, 0.0);
      for (uint32_t s = 0; s <= k; ++s) {
        const double w = rt.wts[t * (k + 1) + s];
        std::vector<double> hh(I), h2(I);
        for (uint32_t n = 0; n < I; ++n) {
          double a = 0, b = 0, a2 = 0, b2 = 0;
          if (s == k) {
            for (uint32_t c = 0; c < D; ++c) {
              const double xv = x0[size_t(t) * D + c], xh = hround(xv);
              a += l.sg.at(n, c) * xh, b += l.su.at(n, c) * xh, a2 += l.sg.at(n, c) * xv, b2 += l.su.at(n, c) * xv;
            }
          } else {
            const uint32_t ex2 = rt.idx[t * k + s];
            l.g.row(ex2, n, wg);
            l.u.row(ex2, n, wu);
            for (uint32_t c = 0; c < D; ++c) {
              const double xv = x0[size_t(t) * D + c], xh = hround(xv);
              a += wg[c] * xh, b += wu[c] * xh, a2 += wg[c] * xv, b2 += wu[c] * xv;
            }
          }
          hh[n] = hround(silu(a) * b);
          h2[n] = silu(a2) * b2;
        }
        for (uint32_t n = 0; n < D; ++n) {
          double a = 0, a2 = 0;
          if (s == k) {
            for (uint32_t c = 0; c < I; ++c) a += l.sd.at(n, c) * hh[c], a2 += l.sd.at(n, c) * h2[c];
          } else {
            l.d.row(rt.idx[t * k + s], n, wd);
            for (uint32_t c = 0; c < I; ++c) a += wd[c] * hh[c], a2 += wd[c] * h2[c];
          }
          acc[n] += w * a;
          acc2[n] += w * a2;
        }
      }
      for (uint32_t n = 0; n < D; ++n) {
        const double base = x0[size_t(t) * D + n];
        ex.add(double(x.as<float>()[size_t(t) * D + n]) - base, acc[n], acc2[n]);
      }
    }
    printf("  v8 R=%u %s/%s nf %d: rel.err %.2e (exact inputs), %.2e vs fp64%s\n", R, gf.c_str(), df.c_str(), nf,
           ex.rel(), ex.rel2(), ex.rel() > 2e-3 ? "  FAILED" : "");
  }
  const double eb = double(l.g.L.bytes + l.u.L.bytes + l.d.L.bytes) / E;
  const double sb = double(l.sg.q.size() + l.su.q.size() + l.sd.q.size()) * (1.0 + 4.0 / 64);
  double distinct = 0;
  for (const auto& r2 : rts) {
    std::vector<uint32_t> u = r2.idx;
    std::sort(u.begin(), u.end());
    distinct += double(std::unique(u.begin(), u.end()) - u.begin());
  }
  distinct /= rts.size();
  std::vector<Variant> vs;
  for (int nf : nfs)
    vs.push_back({"x8_prep2 + plan + moe8i nf " + std::to_string(nf), [&, nf](Encoder& e, int r) { enc(e, r, nf); }});
  printf("    (%.1f distinct experts of %u pairs)\n", distinct, R * k);
  time_variants(vs, 20, rot, distinct * eb + sb, "GB/s (distinct experts read)");
  if (g_variants) {  // the passes alone (on the plan of the last full run: rotation 0's experts)
    std::vector<Variant> parts;
    for (int nf : nfs) {
      parts.push_back({"prep + plan nf " + std::to_string(nf), [&, nf](Encoder& e, int) { enc(e, 0, nf, 1); }});
      parts.push_back({"gate/up nf " + std::to_string(nf), [&, nf](Encoder& e, int) { enc(e, 0, nf, 2); }});
      parts.push_back({"down + combine nf " + std::to_string(nf), [&, nf](Encoder& e, int) { enc(e, 0, nf, 4); }});
    }
    {
      auto e = g.begin();
      enc(*e, 0, nfs[0], 1);
      e->commit_and_wait();
    }
    time_variants(parts, 20, 1, 0, "");
  }
}

// ------------------------------------------------------------------------------------------------------------------ pf
void case_pf(const Layer& l, uint32_t T) {
  Gpu& g = Gpu::get();
  const uint32_t D = l.D, I = l.I, k = l.k, E = l.E, P = T * k, BM = 32;
  const Routing rt = make_routing(T, E, k);
  Buffer X = g.alloc(size_t(T) * D * 2), H = g.alloc(size_t(P) * I * 2), Y = g.alloc(size_t(P) * D * 4);
  Buffer idx = g.alloc(size_t(P) * 4), yrows = g.alloc(size_t(P) * 4), xrows = g.alloc(size_t(P) * 4);
  const uint32_t max_tiles = (P + BM - 1) / BM + E;
  Buffer tiles = g.alloc(size_t(max_tiles) * 3 * 4), nt = g.alloc(16);
  memcpy(idx.ptr(), rt.idx.data(), size_t(P) * 4);
  std::vector<uint16_t> xh(size_t(T) * D);
  for (auto& v : xh) v = f2h(urand(-2.0f, 2.0f));
  memcpy(X.ptr(), xh.data(), xh.size() * 2);
  const std::string gf = fmt_name(l.g.type), df = fmt_name(l.d.type);
  struct Cfg {
    uint32_t gbn, gwn, dbn, dwn;
    const char* gk = "";  // "_k32": 32-deep K steps
    const char* dk = "";
  };
  std::vector<Cfg> cfgs = {{64, 4, 64, 4, "_k32", ""}};  // the model's (Qwen35::enc_moe_prefill_iq)
  if (g_variants || g_validate)
    cfgs = {{64, 4, 128, 4}, {64, 4, 64, 4}, {64, 4, 64, 4, "_k32", ""}, {64, 4, 64, 4, "_k32", "_k32"},
            {64, 4, 128, 4, "_k32", "_k32"}, {32, 2, 128, 4}};
  auto enc = [&](Encoder& e, const Cfg& c, bool bucket) {
    if (bucket) {
      Bucket2Params bp{P, k, E, BM, 0};
      e.use(pipe("moe_bucket3"));
      e.buf(0, idx); e.buf(1, yrows); e.buf(2, xrows); e.buf(3, tiles); e.buf(4, nt);
      e.val(5, bp);
      e.dispatch(1, 1, 1, 1024);
    }
    QrmIqParams gp{I, D, D, I, l.g.sec()};
    e.use(pipe("qrmi_gu_" + gf + "_m32_n" + std::to_string(c.gbn) + c.gk));
    e.buf(0, X); e.buf(1, l.g.buf); e.buf(2, l.u.buf); e.buf(3, H);
    e.val(4, gp);
    e.buf(5, tiles); e.buf(6, nt); e.buf(7, xrows);
    e.dispatch(I / c.gbn, max_tiles, 1, 32 * c.gwn);
    QrmIqParams dp{D, I, I, D, l.d.sec()};
    e.use(pipe("qrmi_dn_" + df + "_m32_n" + std::to_string(c.dbn) + c.dk));
    e.buf(0, H); e.buf(1, l.d.buf); e.buf(2, Y);
    e.val(3, dp);
    e.buf(4, tiles); e.buf(5, nt); e.buf(6, yrows);
    e.dispatch(D / c.dbn, max_tiles, 1, 32 * c.dwn);
  };
  // check sampled pairs
  for (const Cfg& c : cfgs) {
    auto e = g.begin();
    enc(*e, c, true);
    e->commit_and_wait();
    const uint32_t* yr = yrows.as<uint32_t>();
    std::vector<uint32_t> pos_of(P);
    for (uint32_t m = 0; m < P; ++m) pos_of[yr[m]] = m;
    Err eh, ey;
    std::vector<double> wg, wu, wd;
    const int samples = g_validate ? 8 : 12;
    for (int sidx = 0; sidx < samples; ++sidx) {
      const uint32_t pr = uint32_t(rng() % P), t = pr / k, ex2 = rt.idx[pr], m = pos_of[pr];
      std::vector<double> hh(I);
      const uint16_t* Hg = H.as<uint16_t>() + size_t(m) * I;
      for (uint32_t n = 0; n < I; ++n) {
        l.g.row(ex2, n, wg);
        l.u.row(ex2, n, wu);
        double a = 0, b = 0;
        for (uint32_t c2 = 0; c2 < D; ++c2) {
          const double xv = h2f(xh[size_t(t) * D + c2]);
          a += wg[c2] * xv, b += wu[c2] * xv;
        }
        hh[n] = silu(a) * b;
        eh.add(h2f(Hg[n]), hround(hh[n]), hh[n]);
      }
      for (uint32_t n = 0; n < D; ++n) {
        l.d.row(ex2, n, wd);
        double a = 0, a2 = 0;
        for (uint32_t c2 = 0; c2 < I; ++c2) a += wd[c2] * h2f(Hg[c2]), a2 += wd[c2] * hh[c2];
        ey.add(Y.as<float>()[size_t(pr) * D + n], a, a2);
      }
    }
    if (&c == &cfgs[0]) {  // gate/up as fp32 SwiGLU rows (Flint's qrmi_gu_*_f32) on the same tiles
      Buffer H32 = g.alloc(size_t(P) * I * 4);
      QrmIqParams gp{I, D, D, I, l.g.sec()};
      auto e2 = g.begin();
      e2->use(pipe("qrmi_gu_" + gf + "_m32_n64_k32_f32"));
      e2->buf(0, X); e2->buf(1, l.g.buf); e2->buf(2, l.u.buf); e2->buf(3, H32);
      e2->val(4, gp);
      e2->buf(5, tiles); e2->buf(6, nt); e2->buf(7, xrows);
      e2->dispatch(I / 64, max_tiles, 1, 128);
      e2->commit_and_wait();
      Err e32;
      for (uint32_t m = 0; m < P; ++m)
        for (uint32_t n = 0; n < I; ++n) {
          const double hv = h2f(H.as<uint16_t>()[size_t(m) * I + n]), fv = H32.as<float>()[size_t(m) * I + n];
          e32.add(fv, hv, hv);  // against the half path's rows: the same sums, one rounding apart
        }
      printf("  pf T=%u %s qrmi_gu f32: max |H32 - H| / max |H| %.2e%s\n", T, gf.c_str(), e32.rel(),
             e32.rel() > 1e-3 ? "  FAILED" : "");
    }
    printf("  pf T=%u %s/%s gu n%u%s/%u sg, dn n%u%s/%u sg: H rel.err %.2e (vs fp64 %.2e), Y rel.err %.2e (vs fp64 %.2e)%s\n",
           T, gf.c_str(), df.c_str(), c.gbn, c.gk, c.gwn, c.dbn, c.dk, c.dwn, eh.rel(), eh.rel2(), ey.rel(), ey.rel2(),
           eh.rel() > 2e-3 || ey.rel() > 1e-5 ? "  FAILED" : "");
  }
  const double flop = 2.0 * P * (2.0 * I * D + double(D) * I);
  std::vector<Variant> vs;
  for (const Cfg& c : cfgs)
    vs.push_back({"bucket3 + qrmi gu n" + std::to_string(c.gbn) + c.gk + " dn n" + std::to_string(c.dbn) + c.dk,
                  [&, c](Encoder& e, int) { enc(e, c, true); }});
  time_variants(vs, std::max(1u, 8192u / T), 1, flop / 1e3, "TFLOP/s");
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir, fmt = "iq2xxs:iq2s";
  int layer = 0, topk = 8, shared = 0;
  std::vector<std::string> cases;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--tensors" && i + 1 < argc) dir = argv[++i];
    else if (a == "--layer" && i + 1 < argc) layer = atoi(argv[++i]);
    else if (a == "--fmt" && i + 1 < argc) fmt = argv[++i];
    else if (a == "--topk" && i + 1 < argc) topk = atoi(argv[++i]);
    else if (a == "--shared" && i + 1 < argc) shared = atoi(argv[++i]);
    else if (a == "--shared-bits" && i + 1 < argc) g_shared_bits = atoi(argv[++i]);
    else if (a == "--validate") g_validate = true;
    else if (a == "--variants") g_variants = true;
    else if (a == "--trials" && i + 1 < argc) g_trials = atoi(argv[++i]);
    else if (a == "--src" && i + 1 < argc) {
      std::ifstream f(argv[++i]);
      if (!f) fatal("cannot read %s", argv[i]);
      Gpu::get().compile(std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()));
    }
    else cases.push_back(a);
  }
  auto want = [&](const std::string& c) {
    if (cases.empty()) return true;
    for (const auto& s : cases)
      if (c.find(s) != std::string::npos) return true;
    return false;
  };
  Gpu& g = Gpu::get();
  printf("%s, %d GPU cores\n", g.name().c_str(), g.core_count());
  Layer l;
  if (g_validate) {
    const size_t c = fmt.find(':');
    l = synth_layer(512, 512, 32, std::min(topk, 4), fmt_type(fmt.substr(0, c)), fmt_type(fmt.substr(c + 1)),
                    shared ? uint32_t(shared) : 512u);
  } else if (!dir.empty()) {
    l = real_layer(dir, layer);
    l.k = uint32_t(topk);
  } else {
    const size_t c = fmt.find(':');
    l = synth_layer(2048, 512, 256, uint32_t(topk), fmt_type(fmt.substr(0, c)), fmt_type(fmt.substr(c + 1)),
                    shared ? uint32_t(shared) : 512u);
  }
  printf("layer: D %u, I %u, shared %u, %u experts, top-%u, gate/up %s, down %s%s\n", l.D, l.I, l.SI, l.E, l.k,
         fmt_name(l.g.type).c_str(), fmt_name(l.d.type).c_str(), dir.empty() ? " (synthetic)" : " (real tensors)");
  for (uint32_t R : {1u, 2u, 4u})
    if (want("rows R=" + std::to_string(R))) case_rows(l, R);
  for (uint32_t R : {2u, 4u, 8u})
    if (want("v8 R=" + std::to_string(R)) && l.SI == l.I) case_v8(l, R);
  for (uint32_t T : {64u, 512u, 2048u, 4096u})
    if (want("pf T=" + std::to_string(T)) && (!g_validate || T <= 512)) case_pf(l, T);
  return 0;
}
