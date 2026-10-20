// Nemotron-H kernels on synthetic weights of the real shapes: GPU time per call (10 back-to-back calls per command
// buffer, median of 5) and effective bandwidth or TFLOP/s, with a CPU reference check of each kernel's output.
//   nh_bench [case-substring ...]   (default: every case)
// Cases: one-row matvecs per format ("mv"), routed experts for 1 and 8 token rows ("moe"), the 8-row MMA products of
// the verify pass ("v8": NB row blocks of 8 per threadgroup), the prefill GEMMs ("gemm").
// Variants that measured slower were removed.
#include "metal/gpu.h"
#include "model/dspark.h"
#include "model/nemotron_h.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace ni {
extern const char* const kNhKernelSource;
}
using namespace ni;

namespace {

std::mt19937 rng(42);

float ue4m3(uint8_t x) {
  if (x == 0 || x == 0x7F) return 0.0f;
  const int e = (x >> 3) & 0xF, m = x & 7;
  const float raw = e == 0 ? std::ldexp(float(m), -9) : std::ldexp(1.0f + float(m) / 8.0f, e - 7);
  return raw * 0.5f;
}
const int kfp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// random weights in the repacked layout + their fp32 values
struct Mat {
  NhMat m;
  std::vector<float> w;  // [N][K]
};

Mat make(NhFmt f, uint32_t N, uint32_t K, const std::vector<uint64_t>& grid) {
  Gpu& g = Gpu::get();
  Mat r;
  r.m.fmt = f;
  r.m.N = N;
  r.m.K = K;
  nh_row_bytes(f, K, r.m.q_row, r.m.s_row);
  r.m.q = g.alloc(r.m.q_row * N, "q", false);
  r.m.s = r.m.s_row ? g.alloc(r.m.s_row * N, "s", false) : r.m.q;
  r.w.assign(size_t(N) * K, 0.0f);
  uint8_t* q = r.m.q.as<uint8_t>();
  uint8_t* s = r.m.s.as<uint8_t>();
  std::uniform_int_distribution<int> byte(0, 255);
  for (uint32_t n = 0; n < N; ++n) {
    uint8_t* qr = q + size_t(n) * r.m.q_row;
    uint8_t* sr = s + size_t(n) * r.m.s_row;
    float* wr = r.w.data() + size_t(n) * K;
    if (f == NhFmt::Q8) {
      for (uint32_t b = 0; b < K / 32; ++b) {
        const __fp16 d = __fp16(0.001f + 0.002f * float(byte(rng)) / 255.0f);
        memcpy(sr + b * 2, &d, 2);
        for (int i = 0; i < 32; ++i) {
          const int8_t v = int8_t(byte(rng) - 128);
          qr[b * 32 + i] = uint8_t(v);
          wr[b * 32 + i] = float(d) * float(v);
        }
      }
    } else if (f == NhFmt::FP4) {
      for (uint32_t b = 0; b < K / 16; ++b) {
        const uint8_t sc = uint8_t(byte(rng) % 0x60 + 8);
        sr[b] = sc;
        for (int j = 0; j < 8; ++j) {
          const uint8_t c = uint8_t(byte(rng));
          qr[b * 8 + j] = c;
          wr[b * 16 + j] = kfp4[c & 15] * ue4m3(sc);
          wr[b * 16 + j + 8] = kfp4[c >> 4] * ue4m3(sc);
        }
      }
    } else if (f == NhFmt::IQ2) {
      for (uint32_t b = 0; b < K / 256; ++b) {
        const __fp16 d = __fp16(0.0005f + 0.001f * float(byte(rng)) / 255.0f);
        memcpy(sr + b * 2, &d, 2);
        for (int ib = 0; ib < 8; ++ib) {
          uint32_t a0 = 0, a1 = 0;
          for (int l = 0; l < 4; ++l) a0 |= uint32_t(byte(rng)) << (8 * l);
          for (int l = 0; l < 4; ++l) a1 |= uint32_t(byte(rng) & 127) << (7 * l);
          a1 |= uint32_t(byte(rng) & 15) << 28;
          memcpy(qr + b * 64 + ib * 8, &a0, 4);
          memcpy(qr + b * 64 + ib * 8 + 4, &a1, 4);
          const float db = float(d) * (0.5f + float(a1 >> 28)) * 0.25f;
          for (int l = 0; l < 4; ++l) {
            const uint64_t gp = grid[(a0 >> (8 * l)) & 0xFF];
            uint32_t s7 = (a1 >> (7 * l)) & 127;
            const uint32_t sg = s7 | ((__builtin_popcount(s7) & 1) << 7);
            for (int j = 0; j < 8; ++j) {
              const float m = float((gp >> (8 * j)) & 0xFF) * db;
              wr[b * 256 + ib * 32 + l * 8 + j] = (sg >> j) & 1 ? -m : m;
            }
          }
        }
      }
    } else {
      std::normal_distribution<float> nd(0.0f, 0.02f);
      for (uint32_t k = 0; k < K; ++k) wr[k] = nd(rng);
      memcpy(qr, wr, K * 4);
    }
  }
  return r;
}

std::vector<uint64_t> grid_from_source() {
  // parse nh_iq2xxs_grid out of the embedded kernel source
  std::vector<uint64_t> g;
  const std::string src = kNhKernelSource;
  size_t p = src.find("nh_iq2xxs_grid[256] = {");
  p = src.find('{', p) + 1;
  while (g.size() < 256) {
    p = src.find("0x", p);
    g.push_back(std::stoull(src.substr(p + 2, 16), nullptr, 16));
    p += 18;
  }
  return g;
}

double time_kernel(int reps, const std::function<void(Encoder&)>& enc) {
  Gpu& g = Gpu::get();
  for (int w = 0; w < 3; ++w) {
    auto e = g.begin();
    enc(*e);
    e->commit_and_wait();
  }
  std::vector<double> ts;
  for (int r = 0; r < reps; ++r) {
    auto e = g.begin();
    for (int i = 0; i < 10; ++i) enc(*e);
    e->commit_and_wait();
    ts.push_back(e->gpu_ms() / 10);
  }
  std::sort(ts.begin(), ts.end());
  return ts[ts.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> only;
  for (int i = 1; i < argc; ++i) only.push_back(argv[i]);
  auto want = [&](const std::string& n) {
    if (only.empty()) return true;
    for (auto& o : only)
      if (n.find(o) != std::string::npos) return true;
    return false;
  };
  Gpu& g = Gpu::get();
  g.compile_extra("Nemotron-H", kNhKernelSource);
  const std::vector<uint64_t> grid = grid_from_source();
  Buffer gtab = g.alloc(4096, "iq2_grid_half");  // nh_iq2_fill_tables2: the grid as halves (v8 and GEMM kernels)
  {
    auto e = g.begin();
    e->use(g.pipeline("nh_iq2_fill_tables2"));
    e->buf(0, gtab);
    e->dispatch(1, 1, 1, 256);
    e->commit_and_wait();
  }
  struct Case {
    const char* name;
    NhFmt f;
    uint32_t N, K;
  };
  // ---- one-row matvecs: the generic kernel against the tuned ones ----
  const Case cases[] = {
      {"mv fp4 ssm_in 18560x4096", NhFmt::FP4, 18560, 4096},  {"mv fp4 ssm_out 4096x8192", NhFmt::FP4, 4096, 8192},
      {"mv q8 head 131072x4096", NhFmt::Q8, 131072, 4096},     {"mv q8 attn_q 4096x4096", NhFmt::Q8, 4096, 4096},
      {"mv q8 latent_down 1024x4096", NhFmt::Q8, 1024, 4096},  {"mv q8 latent_up 4096x1024", NhFmt::Q8, 4096, 1024},
      {"mv iq2 shexp_up 5376x4096", NhFmt::IQ2, 5376, 4096},   {"mv iq2 shexp_down 4096x5376", NhFmt::IQ2, 4096, 5376},
      {"mv f32 router 496x4096", NhFmt::F32, 496, 4096},
  };
  for (const Case& c : cases) {
    if (!want(c.name)) continue;
    Mat m = make(c.f, c.N, c.K, grid);
    struct Var {
      std::string kn;
      uint32_t rows_tg;  // output rows per threadgroup of 128 threads
    };
    std::vector<Var> vars;
    const char* fm = c.f == NhFmt::Q8 ? "q8" : c.f == NhFmt::FP4 ? "fp4" : c.f == NhFmt::IQ2 ? "iq2" : "f32";
    vars.push_back({std::string("nh_mv_") + fm + "_r1", 16});
    if (c.f == NhFmt::FP4) {
      vars.push_back({"nh_mv1_fp4", 16});
      vars.push_back({"nh_mv1_fp4_n8", 32});
    }
    if (c.f == NhFmt::IQ2) vars.push_back({"nh_mv1_iq2_c", 16});
    Buffer x = g.alloc(size_t(c.K) * 4, "x"), y = g.alloc(size_t(c.N) * 4, "y");
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (size_t i = 0; i < c.K; ++i) x.as<float>()[i] = nd(rng);
    for (const Var& var : vars) {
      struct {
        uint32_t N, K, xs, ys;
        float gs;
        uint32_t R;
      } p{c.N, c.K, c.K, c.N, 1.0f, 1};
      Pipeline pl = g.pipeline(var.kn);
      const double ms = time_kernel(5, [&](Encoder& e) {
        e.use(pl);
        e.buf(0, m.m.q);
        e.buf(1, m.m.s);
        e.buf(2, x);
        e.buf(3, y);
        e.bytes(4, &p, sizeof p);
        e.dispatch((c.N + var.rows_tg - 1) / var.rows_tg, 1, 1, 128);
      });
      double maxerr = 0, maxref = 0;
      for (uint32_t n = 0; n < c.N; n += std::max<uint32_t>(1, c.N / 64)) {
        double ref = 0;
        for (uint32_t k = 0; k < c.K; ++k) ref += double(m.w[size_t(n) * c.K + k]) * x.as<float>()[k];
        maxerr = std::max(maxerr, std::fabs(ref - y.as<float>()[n]));
        maxref = std::max(maxref, std::fabs(ref));
      }
      const double bytes = double(m.m.q.size) + (m.m.s.mtl != m.m.q.mtl ? double(m.m.s.size) : 0.0);
      printf("%-30s %-16s %8.3f ms  %6.1f GB/s  rel.err %.2e\n", c.name, var.kn.c_str(), ms, bytes / ms / 1e6,
             maxerr / std::max(maxref, 1e-30));
    }
    g.release(x);
    g.release(y);
  }
  // ---- the BF16 router for 2-8 token rows: nh_mv_bf16_r<n> against the M5's nh_mv_bf16_rt (bit for bit) ----
  for (uint32_t N : {512u, 40u}) {
    const uint32_t K = N == 512 ? 4096 : 512;
    const std::string name = "router bf16 " + std::to_string(N) + "x" + std::to_string(K);
    if (!want(name)) continue;
    Buffer w = g.alloc(size_t(N) * K * 2, "w"), x = g.alloc(size_t(8) * K * 4, "x");
    Buffer y0 = g.alloc(size_t(8) * N * 4, "y0"), y1 = g.alloc(size_t(8) * N * 4, "y1");
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (size_t i = 0; i < size_t(N) * K; ++i) {
      const float v = nd(rng) * 0.02f;
      uint32_t b;
      memcpy(&b, &v, 4);
      w.as<uint16_t>()[i] = uint16_t(b >> 16);
    }
    for (size_t i = 0; i < size_t(8) * K; ++i) x.as<float>()[i] = nd(rng);
    for (uint32_t R : {2u, 3u, 5u, 8u}) {
      const uint32_t rk = R <= 2 ? 2 : R <= 4 ? 4 : 8, nr = rk <= 2 ? 4 : rk == 4 ? 2 : 1;
      struct {
        uint32_t N, K, xs, ys;
        float gs;
        uint32_t R;
      } p{N, K, K, N, 1.0f, R};
      Pipeline p0 = g.pipeline("nh_mv_bf16_r" + std::to_string(rk)), p1 = g.pipeline("nh_mv_bf16_rt");
      memset(y0.ptr(), 0, y0.size);
      memset(y1.ptr(), 0, y1.size);
      const double ms0 = time_kernel(5, [&](Encoder& e) {
        e.use(p0);
        e.buf(0, w); e.buf(1, w); e.buf(2, x); e.buf(3, y0); e.bytes(4, &p, sizeof p);
        e.dispatch((N + nr * 4 - 1) / (nr * 4), 1, 1, 128);
      });
      const double ms1 = time_kernel(5, [&](Encoder& e) {
        e.use(p1);
        e.buf(0, w); e.buf(2, x); e.buf(3, y1); e.bytes(4, &p, sizeof p);
        e.dispatch((N + 7) / 8, (R + 1) / 2, 1, 128);
      });
      const bool same = memcmp(y0.ptr(), y1.ptr(), size_t(8) * N * 4) == 0;
      printf("%-30s R=%u nh_mv_bf16_r%u %7.4f ms, nh_mv_bf16_rt %7.4f ms, outputs %s\n", name.c_str(), R, rk, ms0, ms1,
             same ? "bitwise identical" : "DIFFER");
    }
    for (Buffer* b : {&w, &x, &y0, &y1}) g.release(*b);
  }
  // ---- routed experts: 9 of 368 experts (width 2304, latent 1024) per token row, nh_moe2 (one row: decode; 8
  // rows: a verify block, each row taking 4 of its experts from 12 popular ones: ~52 distinct of 72 pairs) ----
  if (want("moe")) {
    const uint32_t E = 368, I = 2304, D = 1024, k = 9;
    Mat up = make(NhFmt::IQ2, I * E, D, grid), dn = make(NhFmt::IQ2, D * E, I, grid);
    Buffer live = g.alloc(16, "live");
    live.as<uint32_t>()[0] = 0xffffffffu;
    for (uint32_t R : {1u, 8u}) {
      Buffer lat = g.alloc(size_t(R) * D * 4, "lat"), h = g.alloc(size_t(R) * k * I * 4, "h"), o = g.alloc(size_t(R) * D * 4, "o");
      Buffer eid = g.alloc(size_t(R) * k * 4, "eid"), ew = g.alloc(size_t(R) * k * 4, "ew"), part = g.alloc(size_t(R) * k * D * 4, "part");
      std::normal_distribution<float> nd(0.0f, 1.0f);
      for (size_t i = 0; i < size_t(R) * D; ++i) lat.as<float>()[i] = nd(rng);
      std::mt19937 er(7);
      for (uint32_t r = 0; r < R; ++r) {
        std::vector<int> got;
        while (R > 1 && got.size() < 4) {
          const int v = int(er() % 12);
          if (std::find(got.begin(), got.end(), v) == got.end()) got.push_back(v);
        }
        while (got.size() < k) {
          const int v = 12 + int(er() % (E - 12));
          if (std::find(got.begin(), got.end(), v) == got.end()) got.push_back(v);
        }
        for (uint32_t j = 0; j < k; ++j) {
          eid.as<int>()[r * k + j] = got[j];
          ew.as<float>()[r * k + j] = 0.1f + 0.01f * float(j);
        }
      }
      struct {
        uint32_t I, D, k, uq, us, dq, ds, hs;
      } mp{I, D, k, uint32_t(up.m.q_row * I), uint32_t(up.m.s_row * I), uint32_t(dn.m.q_row * D), uint32_t(dn.m.s_row * D), I};
      const double bytes_up = double(R * k) * (double(up.m.q_row) + double(up.m.s_row)) * I;
      const double bytes_dn = double(R * k) * (double(dn.m.q_row) + double(dn.m.s_row)) * D;
      Pipeline pu = g.pipeline("nh_moe2_up_iq2"), pd = g.pipeline("nh_moe2_dn_iq2"), ps = g.pipeline("nh_moe2_sum");
      const double ms_up = time_kernel(5, [&](Encoder& e) {
        e.use(pu);
        e.buf(0, up.m.q); e.buf(1, up.m.s); e.buf(2, lat); e.buf(3, h); e.buf(4, eid); e.bytes(5, &mp, sizeof mp);
        e.buf(7, live);
        e.dispatch((I + 15) / 16, R * k, 1, 128);
      });
      const double ms_dn = time_kernel(5, [&](Encoder& e) {
        e.use(pd);
        e.buf(0, dn.m.q); e.buf(1, dn.m.s); e.buf(2, h); e.buf(3, part); e.buf(4, eid); e.buf(5, ew);
        e.bytes(6, &mp, sizeof mp); e.buf(8, live);
        e.dispatch((D + 15) / 16, R * k, 1, 128);
        e.use(ps);
        e.buf(0, part); e.buf(1, o); e.bytes(2, &mp, sizeof mp); e.buf(3, live);
        e.dispatch(R, 1, 1, 256);
      });
      // row 0 against the CPU
      double maxerr = 0, maxref = 0;
      std::vector<double> hh(size_t(k) * I);
      for (uint32_t j = 0; j < k; ++j) {
        const uint32_t ex = uint32_t(eid.as<int>()[j]);
        for (uint32_t n = 0; n < I; ++n) {
          double a = 0;
          for (uint32_t kk = 0; kk < D; ++kk) a += double(up.w[(size_t(ex) * I + n) * D + kk]) * lat.as<float>()[kk];
          a = std::max(a, 0.0);
          hh[size_t(j) * I + n] = a * a;
        }
      }
      for (uint32_t n = 0; n < D; n += 37) {
        double ref = 0;
        for (uint32_t j = 0; j < k; ++j) {
          const uint32_t ex = uint32_t(eid.as<int>()[j]);
          double a = 0;
          for (uint32_t kk = 0; kk < I; ++kk) a += double(dn.w[(size_t(ex) * D + n) * I + kk]) * hh[size_t(j) * I + kk];
          ref += ew.as<float>()[j] * a;
        }
        maxerr = std::max(maxerr, std::fabs(ref - o.as<float>()[n]));
        maxref = std::max(maxref, std::fabs(ref));
      }
      printf("moe R=%u nh_moe2         up %7.3f ms %6.1f GB/s, down + sum %7.3f ms %6.1f GB/s  rel.err %.2e\n", R, ms_up,
             bytes_up / ms_up / 1e6, ms_dn, bytes_dn / ms_dn / 1e6, maxerr / std::max(maxref, 1e-30));
      for (Buffer* b : {&lat, &h, &o, &eid, &ew, &part}) g.release(*b);
    }
  }
  // ---- the DSpark drafter's Q8 products (8 rows): nh_v8_q8_s<NB> against nh_xq8_v8 (repacked codes), each call on
  // another copy of the weights (~300 MB in all, so the calls read DRAM) ----
  const Case dcases[] = {{"dsp q8 qkv 4608x2688", NhFmt::Q8, 4608, 2688},
                         {"dsp q8 o 2688x4096", NhFmt::Q8, 2688, 4096},
                         {"dsp q8 kv 512x2688", NhFmt::Q8, 512, 2688}};
  for (const Case& c : dcases) {
    if (!want(c.name)) continue;
    Mat m = make(c.f, c.N, c.K, grid);
    const uint32_t R = 8;
    const size_t qb = m.m.q.size, sb = m.m.s.size;
    const int copies = int(std::max<size_t>(2, (size_t(300) << 20) / (qb + sb)));
    std::vector<Buffer> cq, cs, rq;
    for (int i = 0; i < copies; ++i) {
      cq.push_back(g.alloc(qb, "cq", false));
      cs.push_back(g.alloc(sb, "cs", false));
      rq.push_back(g.alloc(qb, "rq", false));
      memcpy(cq.back().ptr(), m.m.q.ptr(), qb);
      memcpy(cs.back().ptr(), m.m.s.ptr(), sb);
      memcpy(rq.back().ptr(), m.m.q.ptr(), qb);
      DSpark::repack_xq8(rq.back().as<uint8_t>(), c.N, c.K);
    }
    Buffer x = g.alloc(size_t(R) * c.K * 2, "xh"), y = g.alloc(size_t(R) * c.N * 4, "y");
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (size_t i = 0; i < size_t(R) * c.K; ++i) x.as<__fp16>()[i] = __fp16(nd(rng));
    struct {
      uint32_t N, K, x_ld, y_ld, R;
      float gs;
      uint32_t epi;
    } p{c.N, c.K, c.K, c.N, R, 1.0f, 0};
    for (const char* kn : {"nh_v8_q8_s2", "nh_v8_q8_s4", "nh_xq8_v8"}) {
      const bool xq = std::string(kn) == "nh_xq8_v8";
      const uint32_t nb = std::string(kn) == "nh_v8_q8_s4" ? 4 : 2;
      Pipeline pl = g.pipeline(kn);
      int it = 0;
      const double ms = time_kernel(5, [&](Encoder& e) {
        const int i = it++ % copies;
        e.use(pl);
        e.buf(0, x);
        e.buf(1, xq ? rq[size_t(i)] : cq[size_t(i)]);
        e.buf(2, cs[size_t(i)]);
        e.buf(3, y);
        e.bytes(4, &p, sizeof p);
        e.buf(8, gtab);
        if (xq) e.dispatch((c.N + 15) / 16, 1, 1, 256);
        else e.dispatch((c.N + 8 * nb - 1) / (8 * nb), 1, 1, 128);
      });
      double maxerr = 0, maxref = 0;
      for (uint32_t n = 0; n < c.N; n += std::max<uint32_t>(1, c.N / 64))
        for (uint32_t r = 0; r < R; ++r) {
          double ref = 0;
          for (uint32_t k = 0; k < c.K; ++k)
            ref += double(m.w[size_t(n) * c.K + k]) * double(float(x.as<__fp16>()[size_t(r) * c.K + k]));
          maxerr = std::max(maxerr, std::fabs(ref - y.as<float>()[size_t(r) * c.N + n]));
          maxref = std::max(maxref, std::fabs(ref));
        }
      printf("%-30s %-16s %8.4f ms  %6.1f GB/s  rel.err %.2e\n", c.name, kn, ms, double(qb + sb) / ms / 1e6,
             maxerr / std::max(maxref, 1e-30));
    }
    for (auto* v : {&cq, &cs, &rq})
      for (Buffer& b : *v) g.release(b);
    g.release(x);
    g.release(y);
  }
  // ---- 8-row MMA products (nh_v8_*_s<NB>): Y[t][n] for 8 token rows, X half ----
  const Case vcases[] = {
      {"v8 fp4 ssm_in 18560x4096", NhFmt::FP4, 18560, 4096}, {"v8 fp4 ssm_out 4096x8192", NhFmt::FP4, 4096, 8192},
      {"v8 q8 head 131072x4096", NhFmt::Q8, 131072, 4096},  {"v8 q8 attn_q 4096x4096", NhFmt::Q8, 4096, 4096},
      {"v8 q8 latent_up 4096x1024", NhFmt::Q8, 4096, 1024}, {"v8 q8 latent_down 1024x4096", NhFmt::Q8, 1024, 4096},
      {"v8 iq2 shexp_up 5376x4096", NhFmt::IQ2, 5376, 4096}, {"v8 iq2 shexp_down 4096x5376", NhFmt::IQ2, 4096, 5376},
  };
  for (const Case& c : vcases) {
    if (!want(c.name)) continue;
    Mat m = make(c.f, c.N, c.K, grid);
    const uint32_t R = 8;
    Buffer x = g.alloc(size_t(R) * c.K * 2, "xh"), y = g.alloc(size_t(R) * c.N * 4, "y");
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (size_t i = 0; i < size_t(R) * c.K; ++i) x.as<__fp16>()[i] = __fp16(nd(rng));
    const char* fm = c.f == NhFmt::Q8 ? "q8" : c.f == NhFmt::FP4 ? "fp4" : "iq2";
    for (uint32_t nb : {1u, 2u, 4u, 8u}) {
      if (nb == 8 && c.f != NhFmt::FP4) continue;
      const std::string kn = std::string("nh_v8_") + fm + "_s" + std::to_string(nb);
      struct {
        uint32_t N, K, x_ld, y_ld, R;
        float gs;
        uint32_t epi;
      } p{c.N, c.K, c.K, c.N, R, 1.0f, 0};
      Pipeline pl = g.pipeline(kn);
      const uint32_t rows_tg = 8 * nb;
      const double ms = time_kernel(5, [&](Encoder& e) {
        e.use(pl);
        e.buf(0, x);
        e.buf(1, m.m.q);
        e.buf(2, m.m.s);
        e.buf(3, y);
        e.bytes(4, &p, sizeof p);
        e.buf(8, gtab);
        e.dispatch((c.N + rows_tg - 1) / rows_tg, 1, 1, 128);
      });
      double maxerr = 0, maxref = 0;
      for (uint32_t n = 0; n < c.N; n += std::max<uint32_t>(1, c.N / 64))
        for (uint32_t r = 0; r < R; ++r) {
          double ref = 0;
          for (uint32_t k = 0; k < c.K; ++k)
            ref += double(m.w[size_t(n) * c.K + k]) * double(float(x.as<__fp16>()[size_t(r) * c.K + k]));
          maxerr = std::max(maxerr, std::fabs(ref - y.as<float>()[size_t(r) * c.N + n]));
          maxref = std::max(maxref, std::fabs(ref));
        }
      const double bytes = double(m.m.q.size) + (m.m.s.mtl != m.m.q.mtl ? double(m.m.s.size) : 0.0);
      printf("%-30s %-16s %8.3f ms  %6.1f GB/s  rel.err %.2e\n", c.name, kn.c_str(), ms, bytes / ms / 1e6,
             maxerr / std::max(maxref, 1e-30));
    }
    g.release(x);
    g.release(y);
  }
  // ---- prefill GEMMs: Y[M, N] = X W^T, X half ----
  const Case gcases[] = {
      {"gemm fp4 ssm_in", NhFmt::FP4, 18560, 4096}, {"gemm fp4 ssm_out", NhFmt::FP4, 4096, 8192},
      {"gemm q8 attn_q", NhFmt::Q8, 4096, 4096},    {"gemm iq2 shexp_up", NhFmt::IQ2, 5376, 4096},
      {"gemm iq2 shexp_down", NhFmt::IQ2, 4096, 5376},
  };
  for (const Case& c : gcases) {
    if (!want(c.name)) continue;
    Mat m = make(c.f, c.N, c.K, grid);
    for (uint32_t M : {100u, 2048u}) {
      Buffer x = g.alloc(size_t(M) * c.K * 2, "xh"), y = g.alloc(size_t(M) * c.N * 4, "y");
      std::normal_distribution<float> nd(0.0f, 1.0f);
      for (size_t i = 0; i < size_t(M) * c.K; ++i) x.as<__fp16>()[i] = __fp16(nd(rng));
      const std::string kn = std::string(c.f == NhFmt::Q8 ? "nh_gemm_q8" : c.f == NhFmt::FP4 ? "nh_gemm_fp4" : "nh_gemm_iq2") + "_m64";
      struct {
        uint32_t M, N, K, x_ld, y_ld;
        float gs;
        uint32_t epi, qe, se, k;
      } p{M, c.N, c.K, c.K, c.N, 1.0f, 0, 0, 0, 1};
      Pipeline pl = g.pipeline(kn);
      Buffer dummy = g.alloc(64, "dummy");
      const double ms = time_kernel(3, [&](Encoder& e) {
        e.use(pl);
        e.buf(0, x);
        e.buf(1, m.m.q);
        e.buf(2, m.m.s);
        e.buf(3, y);
        e.bytes(4, &p, sizeof p);
        e.buf(5, dummy);
        e.buf(6, dummy);
        e.buf(7, dummy);
        e.buf(8, gtab);
        e.dispatch((c.N + 63) / 64, (M + 63) / 64, 1, 128);
      });
      double maxerr = 0, maxref = 0;
      for (uint32_t r = 0; r < M; r += std::max(1u, M / 7))
        for (uint32_t n = 0; n < c.N; n += std::max<uint32_t>(1, c.N / 50)) {
          double ref = 0;
          for (uint32_t k = 0; k < c.K; ++k)
            ref += double(float(__fp16(m.w[size_t(n) * c.K + k]))) * double(float(x.as<__fp16>()[size_t(r) * c.K + k]));
          maxerr = std::max(maxerr, std::fabs(ref - y.as<float>()[size_t(r) * c.N + n]));
          maxref = std::max(maxref, std::fabs(ref));
        }
      printf("%-28s M=%-5u %-16s %8.3f ms  %6.2f TFLOP/s  rel.err %.2e\n", c.name, M, kn.c_str(), ms,
             2.0 * M * c.N * c.K / ms / 1e9, maxerr / std::max(maxref, 1e-30));
      g.release(x);
      g.release(y);
    }
  }
  return 0;
}
