// Nemotron-H prefill GEMMs and attention at Lightning's shapes (a 2048-row chunk): the simdgroup-MMA kernels
// (nh_gemm_*, nh_x4_moe_*, nh_attn_pf2) against the matrix-unit ones (src/kernels/tensor: nh_gemm_t_*, nh_attn_t_*)
// on synthetic weights: GPU time per call (5 back-to-back calls per command buffer, median of 5; attention 2, of 3),
// TFLOP/s, and the largest difference of the outputs relative to the largest output. Case "attn_pf": ATT_POS0=<cached
// positions before the rows> (0). Case "ssm": the Mamba-2 scan of M rows, nh_ssm against nh_ssm_conv + nh_ssd, both
// against fp64 (SSM_POS0 rows before them, SSM_A: A in [-SSM_A, -SSM_A / 16] (8), SSM_ZS: in_proj rows in +-SSM_ZS).
//   nh_pf_bench [case-substring ...]   M=<rows> (2048) PFT_SRC=<src/kernels> (the tensor kernels from that folder's
//                                      common.h + tensor/*.metal instead of the built-in library) ACC=1 (the dense
//                                      cases' rms error against fp64 sums, both kernels)
#include "metal/gpu.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace ni {
extern const char* const kNhKernelSource;
extern const char* const kTensorKernelSource;
}  // namespace ni
using namespace ni;

namespace {

uint64_t g_rng = 0x9E3779B97F4A7C15ull;
uint64_t rnd() {  // splitmix64
  uint64_t z = (g_rng += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
float urand() { return float(rnd() & 0xFFFFFF) / 16777216.0f; }
uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return uint16_t(u >> 16);
}

struct GemmP {  // NhGemmP
  uint32_t M, N, K, x_ld, y_ld;
  float gs;
  uint32_t epi, q_estride, s_estride, k;
};

enum Fmt { FP4 = 1, BF16 = 4, FP8 = 5 };

// codes [E][N][...] and scales of a synthetic matrix stack
struct Mat {
  Fmt f;
  uint32_t N, K, E;
  Buffer q, s, gse;
  size_t qe, se;  // bytes per expert
};

Mat make_mat(Fmt f, uint32_t N, uint32_t K, uint32_t E) {
  Gpu& g = Gpu::get();
  Mat m{f, N, K, E, {}, {}, {}, 0, 0};
  m.qe = size_t(N) * (f == FP4 ? K / 2 : f == FP8 ? K : 2 * K);
  m.se = f == FP4 ? size_t(N) * (K / 16) : 0;
  m.q = g.alloc(m.qe * E + 16);
  m.s = g.alloc(m.se * E + 16);
  m.gse = g.alloc(size_t(E) * 4 + 16);
  uint8_t* q = (uint8_t*)m.q.ptr();
  if (f == BF16) {
    uint16_t* w = (uint16_t*)q;
    for (size_t i = 0; i < m.qe * E / 2; ++i) w[i] = f2bf((urand() - 0.5f) * 0.1f);
  } else if (f == FP8) {
    // E4M3 codes with exponents 1-12 (values ~2^-6 .. 2^5, x 2^-8 in the decode, gs restores 2^8)
    for (size_t i = 0; i < m.qe * E; ++i) q[i] = uint8_t((rnd() & 0x80) | ((1 + rnd() % 12) << 3) | (rnd() & 7));
  } else {
    for (size_t i = 0; i < m.qe * E; ++i) q[i] = uint8_t(rnd());
    uint8_t* s = (uint8_t*)m.s.ptr();
    for (size_t i = 0; i < m.se * E; ++i) s[i] = uint8_t(0x28 + rnd() % 24);  // ~2^-2 .. 2^1
    float* gse = (float*)m.gse.ptr();
    for (uint32_t e = 0; e < E; ++e) gse[e] = 0.5f + urand();
  }
  return m;
}

double e4m3(uint8_t x) {  // E4M3 (bias 7); 0x7F -> 0 as nh_ue4m3
  if ((x & 0x7F) == 0x7F) return 0.0;
  const int e = (x >> 3) & 0xF, m = x & 7;
  const double v = e == 0 ? std::ldexp(double(m), -9) : std::ldexp(1.0 + double(m) / 8.0, e - 7);
  return x & 0x80 ? -v : v;
}
const double kE2M1[16] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6};

// row n of expert 0 as doubles (the kernels' decode, without gs)
std::vector<double> decode_row(const Mat& m, uint32_t n) {
  std::vector<double> w(m.K);
  const uint8_t* q = (const uint8_t*)m.q.ptr();
  if (m.f == BF16) {
    const uint16_t* r = (const uint16_t*)q + size_t(n) * m.K;
    for (uint32_t k = 0; k < m.K; ++k) {
      const uint32_t u = uint32_t(r[k]) << 16;
      float f;
      memcpy(&f, &u, 4);
      w[k] = f;
    }
  } else if (m.f == FP8) {  // word j of a 16-byte group: bytes (0, 1, 2, 3) = elements (2j, 2j + 1, 2j + 8, 2j + 9)
    const uint8_t* r = q + size_t(n) * m.K;
    for (uint32_t g = 0; g < m.K; g += 16)
      for (int j = 0; j < 4; ++j) {
        const uint8_t* b = r + g + 4 * j;
        w[g + 2 * j] = e4m3(b[0]) / 256;
        w[g + 2 * j + 1] = e4m3(b[1]) / 256;
        w[g + 2 * j + 8] = e4m3(b[2]) / 256;
        w[g + 2 * j + 9] = e4m3(b[3]) / 256;
      }
  } else {  // L0: byte j of a block = elements j (low nibble), j + 8 (high)
    const uint8_t* r = q + size_t(n) * (m.K / 2);
    const uint8_t* s = (const uint8_t*)m.s.ptr() + size_t(n) * (m.K / 16);
    for (uint32_t b = 0; b < m.K / 16; ++b)
      for (int j = 0; j < 8; ++j) {
        const uint8_t c = r[b * 8 + j];
        w[b * 16 + j] = kE2M1[c & 15] * e4m3(s[b]);
        w[b * 16 + j + 8] = kE2M1[c >> 4] * e4m3(s[b]);
      }
  }
  return w;
}

const char* fmt_name(Fmt f) { return f == FP4 ? "fp4" : f == FP8 ? "fp8" : "bf16w"; }

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  log_level() = 0;
  Gpu& g = Gpu::get();
  g.compile_extra("Nemotron-H", kNhKernelSource);
  if (const char* src = getenv("PFT_SRC")) {
    // common.h and tensor/*.metal of the kernel folder <src> (as cmake/embed_kernels_tensor.cmake joins them)
    auto rd = [](const std::string& p) {
      std::ifstream in(p);
      if (!in) fatal("cannot read %s", p.c_str());
      return std::string((std::istreambuf_iterator<char>(in)), {});
    };
    std::vector<std::string> files;
    for (const auto& f : std::filesystem::directory_iterator(std::string(src) + "/tensor"))
      if (f.path().extension() == ".metal") files.push_back(f.path().string());
    std::sort(files.begin(), files.end());
    static std::string all = rd(std::string(src) + "/common.h");
    for (const std::string& f : files) {
      std::string b = rd(f);
      const size_t inc = b.find("#include \"common.h\"");
      if (inc != std::string::npos) b.erase(inc, strlen("#include \"common.h\""));
      all += "\n" + b;
    }
    g.compile_extra("tensor", all.c_str(), true);
  } else {
    g.compile_extra("tensor", kTensorKernelSource, true);
  }
  if (getenv("TGMEM")) {  // TGMEM=1: each tensor kernel's static threadgroup memory and thread limit
    for (const std::string& n : g.function_names())
      if (n.rfind("nh_gemm_t", 0) == 0 || n.rfind("nh_attn_t", 0) == 0) {
        Pipeline pl = g.pipeline(n, false);
        printf("%-32s tg mem %6u  max threads %u\n", n.c_str(), pl.tg_mem, pl.max_threads);
      }
    return 0;
  }
  const uint32_t M = getenv("M") ? uint32_t(atoi(getenv("M"))) : 2048;
  std::vector<std::string> filt(argv + 1, argv + argc);
  auto want = [&](const std::string& n) {
    if (filt.empty()) return true;
    for (auto& f : filt)
      if (n.find(f) != std::string::npos) return true;
    return false;
  };

  // X: half rows, random in [-0.25, 0.25] (the relu^2 of the up projection stays in half range)
  const uint32_t KX = 4096;
  const uint32_t P = M * 6;
  Buffer X = g.alloc(size_t(P) * KX * 2);
  {
    __fp16* x = (__fp16*)X.ptr();
    for (size_t i = 0; i < size_t(P) * KX; ++i) x[i] = __fp16((urand() * 2 - 1) * 0.25f);
  }
  Buffer Y0 = g.alloc(size_t(P) * 10304 * 4), Y1 = g.alloc(size_t(P) * 10304 * 4);
  Buffer tiles = g.alloc(size_t(P / 16 + 256) * 16), ntiles = g.alloc(16), sorted = g.alloc(size_t(P) * 4);

  struct Var {
    std::string name;
    uint32_t bm, bn;  // rows / columns per threadgroup
    uint32_t threads;
  };
  auto run = [&](const Var& v, const Mat& w, uint32_t rows, uint32_t epi, const Buffer& y, uint32_t ntile, int calls) {
    const GemmP p{rows, w.N, w.K, w.K, w.N, w.f == FP8 ? 256.0f : 1.0f, epi, uint32_t(w.qe), uint32_t(w.se), 6};
    Pipeline pl = g.pipeline(v.name, false);
    if (!pl.pso) return -1.0;
    auto e = g.begin();
    for (int c = 0; c < calls; ++c) {
      e->use(pl);
      e->buf(0, X);
      e->buf(1, w.q);
      e->buf(2, w.s);
      e->buf(3, y);
      e->val(4, p);
      e->buf(5, tiles);
      e->buf(6, ntiles);
      e->buf(7, sorted);
      e->buf(8, w.gse);  // (the IQ2 table slot: unused)
      e->buf(9, w.gse);
      e->dispatch((w.N + v.bn - 1) / v.bn, ntile ? ntile : (rows + v.bm - 1) / v.bm, 1, v.threads);
    }
    e->commit_and_wait();
    return e->gpu_ms() / calls;
  };
  auto compare = [&](uint32_t rows, uint32_t N, bool half_out) {
    double mx = 0, md = 0;
    for (size_t i = 0; i < size_t(rows) * N; ++i) {
      const double a = half_out ? double(((__fp16*)Y0.ptr())[i]) : double(((float*)Y0.ptr())[i]);
      const double b = half_out ? double(((__fp16*)Y1.ptr())[i]) : double(((float*)Y1.ptr())[i]);
      mx = std::max(mx, std::fabs(a));
      md = std::isfinite(b) ? std::max(md, std::fabs(a - b)) : INFINITY;
    }
    if (!(mx > 0) || !std::isfinite(mx)) fatal("reference output max %g", mx);
    return md / mx;
  };

  // dense: Lightning's Mamba in / out (FP8), shared expert up / down (NVFP4), attention q / k / o (BF16)
  struct Dense {
    const char* name;
    Fmt f;
    uint32_t N, K;
  };
  const Dense dense[] = {{"in_proj", FP8, 10304, 2688}, {"out_proj", FP8, 2688, 4096}, {"shared_up", FP4, 3712, 2688},
                         {"shared_dn", FP4, 2688, 3712}, {"attn_q", BF16, 4096, 2688}, {"attn_k", BF16, 256, 2688},
                         {"attn_o", BF16, 2688, 4096}};
  // (name, rows, columns per threadgroup)
  const char* new_variants[][3] = {{"nh_gemm_t_%s_m64", "64", "64"}};
  for (const Dense& d : dense) {
    if (!want(d.name)) continue;
    Mat w = make_mat(d.f, d.N, d.K, 1);
    const double flop = 2.0 * M * d.N * d.K;
    const Var old{std::string("nh_gemm_") + fmt_name(d.f) + "_m64", 64, 64, 128};
    std::vector<double> to;
    for (int r = 0; r < 5; ++r) to.push_back(run(old, w, M, 0, Y0, 0, 5));
    const double t0 = median(to);
    printf("%-10s %-24s %8.3f ms %6.2f TFLOP/s\n", d.name, old.name.c_str(), t0, flop / t0 / 1e9);
    for (auto& nv : new_variants) {
      char nm[64];
      snprintf(nm, sizeof nm, nv[0], fmt_name(d.f));
      const Var v{nm, uint32_t(atoi(nv[1])), uint32_t(atoi(nv[2])), 128};
      if (!want(v.name) && !filt.empty() && !want(d.name)) continue;
      if (run(v, w, M, 0, Y1, 0, 1) < 0) continue;
      std::vector<double> tn;
      for (int r = 0; r < 5; ++r) tn.push_back(run(v, w, M, 0, Y1, 0, 5));
      const double t1 = median(tn);
      memset(Y1.ptr(), 0, size_t(M) * d.N * 4);
      run(v, w, M, 0, Y1, 0, 1);
      printf("%-10s %-24s %8.3f ms %6.2f TFLOP/s  x%.2f  rel diff %.2e", d.name, v.name.c_str(), t1, flop / t1 / 1e9,
             t0 / t1, compare(M, d.N, false));
      if (getenv("ACC")) {
        // rms error of both outputs against fp64 sums, over 64 rows x all outputs, relative to the outputs' rms
        double e0 = 0, e1 = 0, rr = 0;
        const float gs = d.f == FP8 ? 256.0f : 1.0f;
        for (uint32_t n = 0; n < d.N; ++n) {
          const std::vector<double> wr = decode_row(w, n);
          for (uint32_t t = 0; t < M; t += M / 64) {
            const __fp16* x = (const __fp16*)X.ptr() + size_t(t) * d.K;
            double ref = 0;
            for (uint32_t k = 0; k < d.K; ++k) ref += wr[k] * double(x[k]);
            ref *= gs;
            const double a = ((float*)Y0.ptr())[size_t(t) * d.N + n], b = ((float*)Y1.ptr())[size_t(t) * d.N + n];
            e0 += (a - ref) * (a - ref);
            e1 += (b - ref) * (b - ref);
            rr += ref * ref;
          }
        }
        printf("  rms err vs fp64: old %.3e new %.3e", std::sqrt(e0 / rr), std::sqrt(e1 / rr));
      }
      printf("\n");
    }
  }

  // prefill attention: T rows after pos0 cached positions (ATT_POS0, default 0), 32 query heads over 2 KV heads, head
  // dim 128; nh_attn_pf2_q1_h4_k32 against nh_attn_t_*
  if (want("attn_pf")) {
    const uint32_t T = M, pos0 = getenv("ATT_POS0") ? uint32_t(atoi(getenv("ATT_POS0"))) : 0, Hq = 32, Hkv = 2, D = 128;
    const uint32_t cap = pos0 + T + 64;
    Buffer q = g.alloc(size_t(T) * Hq * D * 4), kc = g.alloc(size_t(Hkv) * cap * D * 2),
           vc = g.alloc(size_t(Hkv) * cap * D * 2);
    for (size_t i = 0; i < size_t(T) * Hq * D; ++i) ((float*)q.ptr())[i] = (urand() * 2 - 1) * 2.0f;
    for (size_t i = 0; i < size_t(Hkv) * cap * D; ++i) {
      ((__fp16*)kc.ptr())[i] = __fp16(urand() * 2 - 1);
      ((__fp16*)vc.ptr())[i] = __fp16(urand() * 2 - 1);
    }
    // rows past pos0 + T: NaN (no kernel may read them)
    for (uint32_t h = 0; h < Hkv; ++h)
      for (size_t i = size_t(h) * cap * D + size_t(pos0 + T) * D; i < size_t(h + 1) * cap * D; ++i)
        ((uint16_t*)kc.ptr())[i] = ((uint16_t*)vc.ptr())[i] = 0x7E00;
    struct AP {
      uint32_t T, pos0, cap, Hq, Hkv;
      float scale;
      uint32_t qb0;
    } ap{T, pos0, cap, Hq, Hkv, 1.0f / std::sqrt(float(D)), 0};
    const double flop = 4.0 * D * Hq * (double(T) * pos0 + double(T) * (T + 1) / 2);
    auto arun = [&](const char* name, uint32_t rows_tg, uint32_t gy, uint32_t threads, const Buffer& y, int calls) {
      Pipeline pl = g.pipeline(name, false);
      if (!pl.pso) return -1.0;
      auto e = g.begin();
      for (int c = 0; c < calls; ++c) {
        e->use(pl);
        e->buf(0, q);
        e->buf(1, kc);
        e->buf(2, vc);
        e->buf(3, y);
        e->bytes(4, &ap, sizeof ap);
        e->dispatch((T + rows_tg - 1) / rows_tg, gy, 1, threads);
      }
      e->commit_and_wait();
      return e->gpu_ms() / calls;
    };
    struct AV {
      const char* name;
      uint32_t rows_tg, gy, threads;
    };
    const AV av[] = {{"nh_attn_pf2_q1_h4_k32", 8, Hq / 4, 128},
                     {"nh_attn_t_s4", 4, Hkv, 128}};
    double t0 = 0;
    for (const AV& a : av) {
      // (nh_attn_pf2 uses 34 KB of threadgroup memory, which the debug layer refuses)
      if (getenv("MTL_DEBUG_LAYER") && !strncmp(a.name, "nh_attn_pf2", 11)) continue;
      Buffer& y = t0 == 0 ? Y0 : Y1;
      if (arun(a.name, a.rows_tg, a.gy, a.threads, y, 1) < 0) continue;
      std::vector<double> tn;
      for (int r = 0; r < 3; ++r) tn.push_back(arun(a.name, a.rows_tg, a.gy, a.threads, y, 2));
      const double t1 = median(tn);
      memset(y.ptr(), 0, size_t(T) * Hq * D * 2);
      arun(a.name, a.rows_tg, a.gy, a.threads, y, 1);
      if (t0 == 0) {
        t0 = t1;
        printf("attn_pf T %u pos0 %u %-22s %8.3f ms %6.2f TFLOP/s\n", T, pos0, a.name, t1, flop / t1 / 1e9);
      } else {
        printf("attn_pf T %u pos0 %u %-22s %8.3f ms %6.2f TFLOP/s  x%.2f  rel diff %.2e\n", T, pos0, a.name, t1,
               flop / t1 / 1e9, t0 / t1, compare(T, Hq * D, true));
      }
    }
  }

  // Mamba-2 scan of a pass of M rows (SSM_POS0 positions before it, default 0; random rows and state) at Lightning's
  // sizes: nh_ssm against nh_ssm_conv + nh_ssd_m + nh_ssd_scan, both against fp64 (heads 0 and 37: y and the state)
  if (want("ssm")) {
    const uint32_t NH = 64, HD = 64, DS = 128, NG = 8, DI = NH * HD, CH = DI + 2 * NG * DS, ZS = DI + CH + NH, T = M;
    const uint32_t pos0 = getenv("SSM_POS0") ? uint32_t(atoi(getenv("SSM_POS0"))) : 0, nch = (T + 31) / 32;
    auto fill = [&](Buffer& b, size_t n, float lo, float hi) {
      b = g.alloc(n * 4 + 16);
      for (size_t i = 0; i < n; ++i) ((float*)b.ptr())[i] = lo + (hi - lo) * urand();
    };
    Buffer zx, ring, st0, cw, cb, A, D, dtb;
    const float zs = getenv("SSM_ZS") ? float(atof(getenv("SSM_ZS"))) : 1.0f,
                al = getenv("SSM_A") ? float(atof(getenv("SSM_A"))) : 8.0f;
    fill(zx, size_t(T) * ZS, -zs, zs);
    fill(ring, size_t(4) * CH, pos0 ? -1.0f : 0.0f, pos0 ? 1.0f : 0.0f);
    fill(st0, size_t(NH) * HD * DS, pos0 ? -0.5f : 0.0f, pos0 ? 0.5f : 0.0f);
    fill(cw, size_t(CH) * 4, -0.5f, 0.5f);
    fill(cb, CH, -0.2f, 0.2f);
    fill(A, NH, -al, -al / 16);
    fill(D, NH, 0.5f, 1.5f);
    fill(dtb, NH, -3.0f, 0.5f);
    Buffer st1 = g.alloc(st0.size), st2 = g.alloc(st0.size), y1 = g.alloc(size_t(T) * DI * 4), y2 = g.alloc(size_t(T) * DI * 4);
    Buffer live = g.alloc(16), cv = g.alloc((size_t(T) * (CH + 2 * NH)) * 4 + 16), sb = g.alloc(size_t(NH) * nch * (32 * 32 + 64) * 4);
    *(uint32_t*)live.ptr() = 0xffffffffu;
    const struct {
      uint32_t T, pos0, write, zx_stride, NH, HD, DS, NG;
    } sp{T, pos0, 1, ZS, NH, HD, DS, NG};
    const uint32_t zero = 0, nchv[1] = {nch};
    auto run_old = [&]() {
      memcpy(st1.ptr(), st0.ptr(), st0.size);
      auto e = g.begin();
      e->use(g.pipeline("nh_ssm"));
      e->buf(0, zx), e->buf(1, ring), e->buf(2, st1), e->buf(3, y1), e->buf(4, cw), e->buf(5, cb), e->buf(6, A);
      e->buf(7, D), e->buf(8, dtb), e->val(9, sp), e->buf(10, live), e->buf(11, zx), e->val(12, zero);
      e->dispatch(NH, 1, 1, 4 * HD);
      e->commit_and_wait();
      return e->gpu_ms();
    };
    auto run_new = [&]() {
      memcpy(st2.ptr(), st0.ptr(), st0.size);
      auto e = g.begin();
      e->use(g.pipeline("nh_ssm_conv"));
      e->buf(0, zx), e->buf(1, ring), e->buf(3, cv), e->buf(4, cw), e->buf(5, cb), e->buf(6, A), e->buf(8, dtb);
      e->val(9, sp);
      e->dispatch(T, (CH + NH + 255) / 256, 1, 256);
      e->use(g.pipeline("nh_ssd_m"));
      e->buf(6, A), e->val(9, sp), e->buf(11, cv), e->buf(12, sb), e->bytes(13, nchv, 4);
      e->dispatch(NG * nch, 1, 1, 128);
      e->use(g.pipeline("nh_ssd_scan"));
      e->buf(0, zx), e->buf(2, st2), e->buf(3, y2), e->buf(7, D), e->val(9, sp), e->buf(11, cv), e->buf(12, sb);
      e->bytes(13, nchv, 4);
      e->dispatch(NH, 1, 1, 256);
      e->commit_and_wait();
      return e->gpu_ms();
    };
    double to = 1e9, tn = 1e9;
    for (int r = 0; r < 3; ++r) to = std::min(to, run_old()), tn = std::min(tn, run_new());
    // fp64 reference for two heads
    const float* z = (const float*)zx.ptr();
    auto raw = [&](int t, uint32_t c) {
      return t >= 0 ? double(z[size_t(t) * ZS + DI + c]) : double(((float*)ring.ptr())[((pos0 + uint32_t(t + 4)) & 3u) * CH + c]);
    };
    auto conv = [&](int t, uint32_t c) {
      const float* w = (const float*)cw.ptr() + c * 4;
      const double v = ((float*)cb.ptr())[c] + w[0] * raw(t - 3, c) + w[1] * raw(t - 2, c) + w[2] * raw(t - 1, c) + w[3] * raw(t, c);
      return v / (1 + std::exp(-v));
    };
    double ey[2] = {0, 0}, es[2] = {0, 0}, ry = 0, rs = 0;
    for (uint32_t h : {0u, 37u}) {
      const uint32_t gi = h / (NH / NG);
      std::vector<double> S(size_t(HD) * DS);
      for (size_t i = 0; i < S.size(); ++i) S[i] = ((float*)st0.ptr())[size_t(h) * HD * DS + i];
      const double Ah = ((float*)A.ptr())[h], Dh = ((float*)D.ptr())[h], db = ((float*)dtb.ptr())[h];
      for (uint32_t t = 0; t < T; ++t) {
        const double u = z[size_t(t) * ZS + DI + CH + h] + db, dt = u > 20 ? u : std::log1p(std::exp(u)), dA = std::exp(dt * Ah);
        std::vector<double> Bv(DS), Cv(DS);
        for (uint32_t n = 0; n < DS; ++n) Bv[n] = conv(int(t), DI + gi * DS + n), Cv[n] = conv(int(t), DI + NG * DS + gi * DS + n);
        for (uint32_t pp = 0; pp < HD; ++pp) {
          const double x = conv(int(t), h * HD + pp);
          double acc = 0;
          for (uint32_t n = 0; n < DS; ++n) {
            double& sv = S[size_t(pp) * DS + n];
            sv = sv * dA + dt * x * Bv[n];
            acc += sv * Cv[n];
          }
          const double zz = z[size_t(t) * ZS + h * HD + pp], yv = (acc + Dh * x) * zz / (1 + std::exp(-zz));
          const size_t o = size_t(t) * DI + h * HD + pp;
          ey[0] += std::pow(((float*)y1.ptr())[o] - yv, 2), ey[1] += std::pow(((float*)y2.ptr())[o] - yv, 2), ry += yv * yv;
        }
      }
      for (size_t i = 0; i < S.size(); ++i) {
        const size_t o = size_t(h) * HD * DS + i;
        es[0] += std::pow(((float*)st1.ptr())[o] - S[i], 2), es[1] += std::pow(((float*)st2.ptr())[o] - S[i], 2), rs += S[i] * S[i];
      }
    }
    printf("ssm T %u pos0 %u: nh_ssm %.3f ms, conv + nh_ssd %.3f ms | rms err vs fp64: y %.3e / %.3e, state %.3e / %.3e\n", T,
           pos0, to, tn, std::sqrt(ey[0] / ry), std::sqrt(ey[1] / ry), std::sqrt(es[0] / rs), std::sqrt(es[1] / rs));
  }

  // routed experts: 128 NVFP4 experts, top-6 per row (uniformly random), up 1856 x 2688 (relu^2 -> half), down
  // 2688 x 1856 (fp32 rows in pair order)
  if (want("moe")) {
    const uint32_t E = 128, I = 1856, H = 2688;
    std::vector<int> eid(P);
    for (uint32_t t = 0; t < M; ++t) {
      int used[6];
      for (int j = 0; j < 6; ++j) {
        int e;
        bool dup;
        do {
          e = int(rnd() % E);
          dup = false;
          for (int i = 0; i < j; ++i) dup |= used[i] == e;
        } while (dup);
        used[j] = e;
        eid[size_t(t) * 6 + j] = e;
      }
    }
    std::vector<uint32_t> cnt(E, 0), off(E, 0);
    for (int e : eid) ++cnt[size_t(e)];
    for (uint32_t e = 1; e < E; ++e) off[e] = off[e - 1] + cnt[e - 1];
    {
      std::vector<uint32_t> pos(off);
      uint32_t* so = (uint32_t*)sorted.ptr();
      for (uint32_t i = 0; i < P; ++i) so[pos[size_t(eid[i])]++] = i;
    }
    auto set_tiles = [&](uint32_t bm) {
      uint32_t* tl = (uint32_t*)tiles.ptr();
      uint32_t n = 0;
      for (uint32_t e = 0; e < E; ++e)
        for (uint32_t j = 0; j * bm < cnt[e]; ++j, ++n) {
          tl[n * 4] = e;
          tl[n * 4 + 1] = off[e] + j * bm;
          tl[n * 4 + 2] = std::min(bm, cnt[e] - j * bm);
          tl[n * 4 + 3] = 0;
        }
      *(uint32_t*)ntiles.ptr() = n;
      return n;
    };
    Mat up = make_mat(FP4, I, H, E), dn = make_mat(FP4, H, I, E);
    const double flop = 2.0 * P * I * H;
    struct MoeV {
      const char* name;
      bool up;
      uint32_t bm, bn;
    };
    const MoeV mv[] = {{"nh_x4_moe_up_m32", true, 32, 64},         {"nh_gemm_fp4_moe_up_m32", true, 32, 64},
                       {"nh_x4_moe_up_m16", true, 16, 64},         {"nh_gemm_t_fp4_moe_up_m16", true, 16, 64},
                       {"nh_gemm_t_fp4_moe_up_m32", true, 32, 64}, {"nh_gemm_t_fp4_moe_up_m64", true, 64, 64},
                       {"nh_gemm_fp4_moe_dn_m32", false, 32, 64},  {"nh_gemm_fp4_moe_dn_m16", false, 16, 64},
                       {"nh_gemm_t_fp4_moe_dn_m16", false, 16, 64}, {"nh_gemm_t_fp4_moe_dn_m32", false, 32, 64},
                       {"nh_gemm_t_fp4_moe_dn_m64", false, 64, 64}, {"nh_gemm_t_fp4_moe_dn_m64n128", false, 64, 128}};
    double base[2] = {0, 0};
    for (const MoeV& m : mv) {
      const uint32_t nt = set_tiles(m.bm);
      const Mat& w = m.up ? up : dn;
      const Var v{m.name, m.bm, m.bn, 128};
      // up: X = token rows (P / 6 of them, K = H); down: X = the pairs' half rows (K = I)
      const uint32_t rows = P;
      Buffer& y = base[m.up ? 0 : 1] == 0 ? Y0 : Y1;
      if (run(v, w, rows, m.up ? 2 : 0, y, nt, 1) < 0) continue;
      std::vector<double> tn;
      for (int r = 0; r < 5; ++r) tn.push_back(run(v, w, rows, m.up ? 2 : 0, y, nt, 5));
      const double t1 = median(tn);
      memset(y.ptr(), 0, size_t(P) * w.N * 4);
      run(v, w, rows, m.up ? 2 : 0, y, nt, 1);
      double& b = base[m.up ? 0 : 1];
      if (b == 0) {
        b = t1;
        printf("moe_%s %-26s %8.3f ms %6.2f TFLOP/s (%u tiles)\n", m.up ? "up" : "dn", m.name, t1, flop / t1 / 1e9, nt);
      } else {
        printf("moe_%s %-26s %8.3f ms %6.2f TFLOP/s (%u tiles)  x%.2f  rel diff %.2e\n", m.up ? "up" : "dn", m.name, t1,
               flop / t1 / 1e9, nt, b / t1, compare(P, w.N, m.up));
      }
    }
  }
  return 0;
}
