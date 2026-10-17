// Nemotron-H decode kernels on synthetic weights of the real shapes: GPU time per call and effective bandwidth,
// with a CPU reference check of each kernel's output.
//   nh_bench [kernel-substring ...]   (default: every case)
// Cases: matvecs per format (FP4 Mamba in/out, Q8 LM head / attention / latent, IQ2 shared expert, F32 router),
// routed experts up/down, the Mamba step.
#include "metal/gpu.h"
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
  struct Case {
    const char* name;
    NhFmt f;
    uint32_t N, K;
  };
  const Case cases[] = {
      {"fp4 ssm_in 18560x4096", NhFmt::FP4, 18560, 4096},  {"fp4 ssm_out 4096x8192", NhFmt::FP4, 4096, 8192},
      {"q8 head 131072x4096", NhFmt::Q8, 131072, 4096},     {"q8 attn_q 4096x4096", NhFmt::Q8, 4096, 4096},
      {"q8 latent_down 1024x4096", NhFmt::Q8, 1024, 4096},  {"q8 latent_up 4096x1024", NhFmt::Q8, 4096, 1024},
      {"iq2 shexp_up 5376x4096", NhFmt::IQ2, 5376, 4096},   {"iq2 shexp_down 4096x5376", NhFmt::IQ2, 4096, 5376},
      {"f32 router 496x4096", NhFmt::F32, 496, 4096},
  };
  for (const Case& c : cases) {
    if (!want(c.name)) continue;
    Mat m = make(c.f, c.N, c.K, grid);
    struct Var {
      std::string kn;
      int R;
      uint32_t nr;
      uint32_t persist = 0;  // persistent: threadgroups of 256 threads, at most this many per GPU core
    };
    std::vector<Var> vars;
    const char* fm = c.f == NhFmt::Q8 ? "q8" : c.f == NhFmt::FP4 ? "fp4" : c.f == NhFmt::IQ2 ? "iq2" : "f32";
    vars.push_back({std::string("nh_mv_") + fm + "_r1", 1, 4});
    if (c.f == NhFmt::FP4) {
      vars.push_back({"nh_mv1_fp4", 1, 4});
      vars.push_back({"nh_mv1_fp4_n8", 1, 8});
      vars.push_back({"nh_mv1_fp4p", 1, 4, 4});
      vars.push_back({"nh_mv1_fp4p", 1, 4, 8});
      vars.push_back({"nh_mv1_fp4p_n2", 1, 2, 8});
      vars.push_back({"nh_mv1_fp4x", 1, 4});
      vars.push_back({"nh_mv1_fp4x_n8", 1, 8});
    }
    if (c.f == NhFmt::IQ2) {
      vars.push_back({"nh_mv1_iq2", 1, 4});
      vars.push_back({"nh_mv1_iq2_c", 1, 4});
      vars.push_back({"nh_mv1_iq2_d", 1, 4});
      vars.push_back({"nh_mv1_iq2p", 1, 4, 4});
      vars.push_back({"nh_mv1_iq2p", 1, 4, 8});
      vars.push_back({"nh_mv1_iq2p_n2", 1, 2, 8});
      vars.push_back({"nh_mv1_iq2_h", 1, 4});
      vars.push_back({"nh_mv1_iq2_f", 1, 4});
      vars.push_back({"nh_mv1_iq2_a", 1, 4});
      vars.push_back({"nh_mv1_iq2_h2", 1, 2});
      vars.push_back({"nh_mv1_iq2_h8", 1, 8});
      vars.push_back({"nh_mv1_iq2_q", 1, 4});
      vars.push_back({"nh_mv1_iq2_q_t", 1, 4});
      vars.push_back({"nh_mv1_iq2_q8r", 1, 8});
      vars.push_back({"nh_mv1_iq2_q8r_t", 1, 8});
    }
    vars.push_back({std::string("nh_mv_") + fm + "_r8", 8, 1});
    for (const Var& var : vars) {
      if (!want(var.kn) && !want(c.name)) continue;
      const int R = var.R;
      Buffer x = g.alloc(size_t(R) * c.K * 4, "x"), y = g.alloc(size_t(R) * c.N * 4, "y");
      std::normal_distribution<float> nd(0.0f, 1.0f);
      for (size_t i = 0; i < size_t(R) * c.K; ++i) x.as<float>()[i] = nd(rng);
      NemotronH* none = nullptr;
      (void)none;
      // a NemotronH instance is not needed: build the dispatch as enc_mv does
      const std::string kn = var.kn;
      const uint32_t nr = var.nr, nsg = 4;
      struct {
        uint32_t N, K, xs, ys;
        float gs;
        uint32_t R;
      } p{c.N, c.K, c.K, c.N, 1.0f, uint32_t(R)};
      Pipeline pl = g.pipeline(kn);
      static Buffer tab, tab2, tabq;
      if (!tab) {
        tab = g.alloc(3072 * 4, "iq2_tab");
        tab2 = g.alloc(18432, "iq2_tab2");
        tabq = g.alloc(512, "iq2_codes");
        auto e = g.begin();
        e->use(g.pipeline("nh_iq2_fill_codes"));
        e->buf(0, tabq);
        e->dispatch(1, 1, 1, 256);
        e->use(g.pipeline("nh_iq2_fill_tables"));
        e->buf(0, tab);
        e->dispatch(1, 1, 1, 256);
        e->use(g.pipeline("nh_iq2_fill_tables2"));
        e->buf(0, tab2);
        e->dispatch(1, 1, 1, 256);
        e->commit_and_wait();
      }
      const bool t2 = kn.rfind("nh_mv1_iq2_", 0) == 0 && kn.size() > 11 && (kn[11] == 'h' || kn[11] == 'f' || kn[11] == 'a');
      const bool tq = kn.rfind("nh_mv1_iq2_q", 0) == 0;
      const double ms = time_kernel(5, [&](Encoder& e) {
        e.use(pl);
        e.buf(0, m.m.q);
        e.buf(1, m.m.s);
        e.buf(2, x);
        e.buf(3, y);
        e.bytes(4, &p, sizeof p);
        e.buf(5, tq ? tabq : t2 ? tab2 : tab);
        if (var.persist) {
          const uint32_t blocks = (c.N + nr * 8 - 1) / (nr * 8);
          e.dispatch(std::min<uint32_t>(blocks, var.persist * uint32_t(g.core_count())), 1, 1, 256);
        } else {
          e.dispatch((c.N + nr * nsg - 1) / (nr * nsg), 1, 1, 32 * nsg);
        }
      });
      // check a few rows
      double maxerr = 0, maxref = 0;
      for (uint32_t n = 0; n < c.N; n += std::max<uint32_t>(1, c.N / 64))
        for (int r = 0; r < R; ++r) {
          double ref = 0;
          for (uint32_t k = 0; k < c.K; ++k) ref += double(m.w[size_t(n) * c.K + k]) * x.as<float>()[size_t(r) * c.K + k];
          maxerr = std::max(maxerr, std::fabs(ref - y.as<float>()[size_t(r) * c.N + n]));
          maxref = std::max(maxref, std::fabs(ref));
        }
      const double bytes = double(m.m.q.size) + (m.m.s.mtl != m.m.q.mtl ? double(m.m.s.size) : 0.0);
      printf("%-28s R=%d %-20s %8.3f ms  %6.1f GB/s  rel.err %.2e\n", c.name, R,
             (kn + (var.persist ? "/" + std::to_string(var.persist) : std::string())).c_str(), ms, bytes / ms / 1e6,
             maxerr / std::max(maxref, 1e-30));
      g.release(x);
      g.release(y);
    }
  }
  // ---- routed experts (decode): 9 of 368 experts, width 2304, latent 1024 ----
  if (want("moe")) {
    const uint32_t E = 368, I = 2304, D = 1024, k = 9;
    Mat up = make(NhFmt::IQ2, I * E, D, grid), dn = make(NhFmt::IQ2, D * E, I, grid);
    for (uint32_t R : {1u, 8u}) {
      Buffer lat = g.alloc(size_t(R) * D * 4, "lat"), h = g.alloc(size_t(R) * k * I * 4, "h"), o = g.alloc(size_t(R) * D * 4, "o");
      Buffer eid = g.alloc(size_t(R) * k * 4, "eid"), ew = g.alloc(size_t(R) * k * 4, "ew");
      std::normal_distribution<float> nd(0.0f, 1.0f);
      for (size_t i = 0; i < size_t(R) * D; ++i) lat.as<float>()[i] = nd(rng);
      for (uint32_t i = 0; i < R * k; ++i) {
        eid.as<int>()[i] = int((i * 37 + 11) % E);
        ew.as<float>()[i] = 0.1f + 0.01f * float(i % k);
      }
      struct {
        uint32_t I, D, k, uq, us, dq, ds, hs;
      } mp{I, D, k, uint32_t(up.m.q_row * I), uint32_t(up.m.s_row * I), uint32_t(dn.m.q_row * D), uint32_t(dn.m.s_row * D), I};
      const double bytes_up = double(R * k) * (double(up.m.q_row) + double(up.m.s_row)) * I;
      const double bytes_dn = double(R * k) * (double(dn.m.q_row) + double(dn.m.s_row)) * D;
      Buffer part = g.alloc(size_t(R) * k * D * 4, "part"), tab2 = g.alloc(18432, "iq2_tab2");
      {
        auto e = g.begin();
        e->use(g.pipeline("nh_iq2_fill_tables2"));
        e->buf(0, tab2);
        e->dispatch(1, 1, 1, 256);
        e->commit_and_wait();
      }
      const char* names[] = {"grid", "persistent", "v2 const", "v2 h", "v2 f", "v2 a", "dedup"};
      for (int v = 0; v < 7; ++v) {
        if (v == 1 || v == 3 || v == 4) continue;
        if (v == 6) {
          // pairs grouped by expert (tiles of <= 8 pairs), each expert row decoded once per 2 tokens
          const uint32_t P = R * k;
          Buffer sorted = g.alloc(P * 4, "sorted"), rank = g.alloc(P * 4, "rank"), tiles = g.alloc((P + E + 2) * 16, "tiles"),
                 ntiles = g.alloc(16, "ntiles"), outp = g.alloc(size_t(P) * D * 4, "outp");
          struct { uint32_t P, E, BM; } bp{P, E, 8};
          {
            auto e = g.begin();
            e->use(g.pipeline("nh_moe_bucket"));
            e->buf(0, eid); e->buf(1, sorted); e->buf(2, tiles); e->buf(3, ntiles); e->buf(4, rank);
            e->bytes(5, &bp, sizeof bp);
            e->dispatch(1, 1, 1, 1024);
            e->commit_and_wait();
          }
          const uint32_t nt = ntiles.as<uint32_t>()[0];
          Pipeline p3u = g.pipeline("nh_moe3_up_iq2"), p3d = g.pipeline("nh_moe3_dn_iq2"), pc = g.pipeline("nh_moe_combine");
          const double ms_up = time_kernel(5, [&](Encoder& e) {
            e.use(p3u);
            e.buf(0, up.m.q); e.buf(1, up.m.s); e.buf(2, lat); e.buf(3, h); e.buf(4, tiles); e.buf(5, ntiles);
            e.buf(6, sorted); e.bytes(7, &mp, sizeof mp);
            e.dispatch((I + 15) / 16, nt * 4, 1, 128);
          });
          Buffer loh = g.alloc(size_t(R) * D * 2, "loh");
          const double ms_dn = time_kernel(5, [&](Encoder& e) {
            e.use(p3d);
            e.buf(0, dn.m.q); e.buf(1, dn.m.s); e.buf(2, h); e.buf(3, outp); e.buf(4, tiles); e.buf(5, ntiles);
            e.buf(6, sorted); e.bytes(7, &mp, sizeof mp);
            e.dispatch((D + 15) / 16, nt * 4, 1, 128);
            struct { uint32_t D, k; } cp{D, k};
            e.use(pc);
            e.buf(0, outp); e.buf(1, ew); e.buf(2, loh); e.bytes(3, &cp, sizeof cp);
            e.dispatch(R, 1, 1, 256);
          });
          // row 0 against the CPU (fp32 h; the combine stores half)
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
            maxerr = std::max(maxerr, std::fabs(ref - float(loh.as<__fp16>()[n])));
            maxref = std::max(maxref, std::fabs(ref));
          }
          printf("moe R=%u %-18s up %7.3f ms %6.1f GB/s, down %7.3f ms %6.1f GB/s  rel.err %.2e (%u tiles)\n", R,
                 names[v], ms_up, bytes_up / ms_up / 1e6, ms_dn, bytes_dn / ms_dn / 1e6, maxerr / std::max(maxref, 1e-30),
                 nt);
          continue;
        }
        const char* suf = v == 3 ? "_h" : v == 4 ? "_f" : v == 5 ? "_a" : "";
        Pipeline pu = g.pipeline(v >= 2 ? std::string("nh_moe2_up_iq2") + suf : v ? "nh_moe1p_up_iq2" : "nh_moe_up_iq2");
        Pipeline pd = g.pipeline(v >= 2 ? std::string("nh_moe2_dn_iq2") + suf : v ? "nh_moe1p_down_iq2" : "nh_moe_down_iq2");
        Pipeline ps = g.pipeline("nh_moe2_sum");
        const uint32_t npairs = R * k;
        const double ms_up = time_kernel(5, [&](Encoder& e) {
          e.use(pu);
          e.buf(0, up.m.q); e.buf(1, up.m.s); e.buf(2, lat); e.buf(3, h); e.buf(4, eid); e.bytes(5, &mp, sizeof mp);
          if (v >= 3) e.buf(6, tab2);
          if (v == 1) {
            e.bytes(6, &npairs, 4);
            e.dispatch(4 * uint32_t(g.core_count()), 1, 1, 256);
          } else {
            e.dispatch((I + 15) / 16, R * k, 1, 128);
          }
        });
        const double ms_dn = time_kernel(5, [&](Encoder& e) {
          e.use(pd);
          e.buf(0, dn.m.q); e.buf(1, dn.m.s); e.buf(2, h); e.buf(3, v >= 2 ? part : o); e.buf(4, eid); e.buf(5, ew);
          e.bytes(6, &mp, sizeof mp);
          if (v >= 3) e.buf(7, tab2);
          if (v == 1) {
            e.bytes(7, &R, 4);
            e.dispatch(4 * uint32_t(g.core_count()), 1, 1, 256);
          } else if (v >= 2) {
            e.dispatch((D + 15) / 16, R * k, 1, 128);
            e.use(ps);
            e.buf(0, part); e.buf(1, o); e.bytes(2, &mp, sizeof mp);
            e.dispatch(R, 1, 1, 256);
          } else {
            e.dispatch((D + 15) / 16, R, 1, 128);
          }
        });
        // check o against the CPU for row 0
        double maxerr = 0, maxref = 0;
        std::vector<double> hh(size_t(k) * I);
        for (uint32_t j = 0; j < k; ++j) {
          const uint32_t e = uint32_t(eid.as<int>()[j]);
          for (uint32_t n = 0; n < I; ++n) {
            double a = 0;
            for (uint32_t kk = 0; kk < D; ++kk) a += double(up.w[(size_t(e) * I + n) * D + kk]) * lat.as<float>()[kk];
            a = std::max(a, 0.0);
            hh[size_t(j) * I + n] = a * a;
          }
        }
        for (uint32_t n = 0; n < D; n += 37) {
          double ref = 0;
          for (uint32_t j = 0; j < k; ++j) {
            const uint32_t e = uint32_t(eid.as<int>()[j]);
            double a = 0;
            for (uint32_t kk = 0; kk < I; ++kk) a += double(dn.w[(size_t(e) * D + n) * I + kk]) * hh[size_t(j) * I + kk];
            ref += ew.as<float>()[j] * a;
          }
          maxerr = std::max(maxerr, std::fabs(ref - o.as<float>()[n]));
          maxref = std::max(maxref, std::fabs(ref));
        }
        printf("moe R=%u %-18s up %7.3f ms %6.1f GB/s, down %7.3f ms %6.1f GB/s  rel.err %.2e\n", R, names[v], ms_up,
               bytes_up / ms_up / 1e6, ms_dn, bytes_dn / ms_dn / 1e6, maxerr / std::max(maxref, 1e-30));
      }
      g.release(part);
      g.release(tab2);
    }
  }
  // ---- 8-row MMA products (nh_v8_*): Y[t][n] for 8 token rows, X half ----
  {
    const Case vcases[] = {
        {"v8 fp4 ssm_in 18560x4096", NhFmt::FP4, 18560, 4096}, {"v8 fp4 ssm_out 4096x8192", NhFmt::FP4, 4096, 8192},
        {"v8 q8 head 131072x4096", NhFmt::Q8, 131072, 4096},  {"v8 q8 attn_q 4096x4096", NhFmt::Q8, 4096, 4096},
        {"v8 q8 latent_up 4096x1024", NhFmt::Q8, 4096, 1024}, {"v8 iq2 shexp_up 5376x4096", NhFmt::IQ2, 5376, 4096},
        {"v8 iq2 shexp_down 4096x5376", NhFmt::IQ2, 4096, 5376},
    };
    for (const Case& c : vcases) {
      if (!want(c.name)) continue;
      Mat m = make(c.f, c.N, c.K, grid);
      const uint32_t R = 8;
      Buffer x = g.alloc(size_t(R) * c.K * 2, "xh"), y = g.alloc(size_t(R) * c.N * 4, "y"), dummy = g.alloc(64, "d");
      Buffer vtab = g.alloc(18432, "iq2_tab2");
      {
        auto e = g.begin();
        e->use(g.pipeline("nh_iq2_fill_tables2"));
        e->buf(0, vtab);
        e->dispatch(1, 1, 1, 256);
        e->commit_and_wait();
      }
      std::normal_distribution<float> nd(0.0f, 1.0f);
      for (size_t i = 0; i < size_t(R) * c.K; ++i) x.as<__fp16>()[i] = __fp16(nd(rng));
      const char* fm = c.f == NhFmt::Q8 ? "q8" : c.f == NhFmt::FP4 ? "fp4" : "iq2";
      for (int vi : {1, 2, -1, -2, -11, -12}) {
        if (vi <= -11 && c.f != NhFmt::FP4) continue;
        const int nb = vi <= -11 ? -vi - 10 : vi < 0 ? -vi : vi;
        const bool split = vi < 0;
        const std::string kn = std::string("nh_v8_") + fm + (vi <= -11 ? "x" : "") + (split ? "_s" : "_b") + std::to_string(nb);
        struct {
          uint32_t N, K, x_ld, y_ld, R;
          float gs;
          uint32_t epi, qe, se, k;
        } p{c.N, c.K, c.K, c.N, R, 1.0f, 0, 0, 0, 1};
        Pipeline pl = g.pipeline(kn);
        const uint32_t nsg = 4, rows_tg = split ? 8 * uint32_t(nb) : 8 * uint32_t(nb) * nsg;
        const double ms = time_kernel(5, [&](Encoder& e) {
          e.use(pl);
          e.buf(0, x);
          e.buf(1, m.m.q);
          e.buf(2, m.m.s);
          e.buf(3, y);
          e.bytes(4, &p, sizeof p);
          e.buf(5, dummy);
          e.buf(6, dummy);
          e.buf(7, dummy);
          e.buf(8, vtab);
          e.dispatch((c.N + rows_tg - 1) / rows_tg, 1, 1, 32 * nsg);
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
        printf("%-28s R=8 %-16s %8.3f ms  %6.1f GB/s  rel.err %.2e\n", c.name, kn.c_str(), ms, bytes / ms / 1e6,
               maxerr / std::max(maxref, 1e-30));
      }
      g.release(x);
      g.release(y);
    }
    // routed experts of 8 token rows: 72 (row, expert) pairs over 40 distinct experts, tiles from nh_moe_bucket
    if (want("v8 moe")) {
      const uint32_t E = 368, I = 2304, D = 1024, k = 9, R = 8, P = R * k;
      Mat up = make(NhFmt::IQ2, I * E, D, grid), dn = make(NhFmt::IQ2, D * E, I, grid);
      Buffer lat = g.alloc(size_t(R) * D * 2, "lat"), h = g.alloc(size_t(P) * I * 2, "h"), o = g.alloc(size_t(P) * D * 4, "o");
      Buffer eid = g.alloc(P * 4, "eid"), sorted = g.alloc(P * 4, "sorted"), rank = g.alloc(P * 4, "rank");
      Buffer tiles = g.alloc((P / 16 + E + 2) * 16, "tiles"), ntiles = g.alloc(16, "ntiles");
      Buffer mtab = g.alloc(18432, "iq2_tab2");
      {
        auto e = g.begin();
        e->use(g.pipeline("nh_iq2_fill_tables2"));
        e->buf(0, mtab);
        e->dispatch(1, 1, 1, 256);
        e->commit_and_wait();
      }
      std::normal_distribution<float> nd(0.0f, 1.0f);
      for (size_t i = 0; i < size_t(R) * D; ++i) lat.as<__fp16>()[i] = __fp16(nd(rng));
      for (uint32_t t = 0; t < R; ++t)
        for (uint32_t j = 0; j < k; ++j) eid.as<int>()[t * k + j] = int((j * 41 + t * (j % 3 == 0 ? 0 : 7) + 5) % E);
      {
        struct { uint32_t P, E, BM; } bp{P, E, 8};
        auto e = g.begin();
        e->use(g.pipeline("nh_moe_bucket"));
        e->buf(0, eid); e->buf(1, sorted); e->buf(2, tiles); e->buf(3, ntiles); e->buf(4, rank);
        e->bytes(5, &bp, sizeof bp);
        e->dispatch(1, 1, 1, 1024);
        e->commit_and_wait();
      }
      const uint32_t nt = ntiles.as<uint32_t>()[0];
      for (int nb : {1, 2}) {
        struct {
          uint32_t N, K, x_ld, y_ld, R;
          float gs;
          uint32_t epi, qe, se, k;
        } pu{I, D, D, I, R, 1.0f, 2, uint32_t(up.m.q_row * I), uint32_t(up.m.s_row * I), k},
            pd{D, I, I, D, R, 1.0f, 0, uint32_t(dn.m.q_row * D), uint32_t(dn.m.s_row * D), k};
        const uint32_t nsg = 4, rows_tg = 8 * uint32_t(nb) * nsg;
        Pipeline plu = g.pipeline("nh_v8_iq2_moe_up_b" + std::to_string(nb));
        Pipeline pld = g.pipeline("nh_v8_iq2_moe_dn_b" + std::to_string(nb));
        const double ms_up = time_kernel(5, [&](Encoder& e) {
          e.use(plu);
          e.buf(0, lat); e.buf(1, up.m.q); e.buf(2, up.m.s); e.buf(3, h); e.bytes(4, &pu, sizeof pu);
          e.buf(5, tiles); e.buf(6, ntiles); e.buf(7, sorted); e.buf(8, mtab);
          e.dispatch((I + rows_tg - 1) / rows_tg, nt, 1, 32 * nsg);
        });
        const double ms_dn = time_kernel(5, [&](Encoder& e) {
          e.use(pld);
          e.buf(0, h); e.buf(1, dn.m.q); e.buf(2, dn.m.s); e.buf(3, o); e.bytes(4, &pd, sizeof pd);
          e.buf(5, tiles); e.buf(6, ntiles); e.buf(7, sorted); e.buf(8, mtab);
          e.dispatch((D + rows_tg - 1) / rows_tg, nt, 1, 32 * nsg);
        });
        // check pair 0's output against the CPU (with h rounded to half as the kernel stores it)
        double maxerr = 0, maxref = 0;
        const uint32_t pair = 0, tok = pair / k, ex = uint32_t(eid.as<int>()[pair]);
        std::vector<double> hh(I);
        for (uint32_t n = 0; n < I; ++n) {
          double a = 0;
          for (uint32_t kk = 0; kk < D; ++kk) a += double(up.w[(size_t(ex) * I + n) * D + kk]) * float(lat.as<__fp16>()[tok * D + kk]);
          a = std::max(a, 0.0);
          hh[n] = float(__fp16(float(a * a)));
        }
        for (uint32_t n = 0; n < D; n += 13) {
          double ref = 0;
          for (uint32_t kk = 0; kk < I; ++kk) ref += double(dn.w[(size_t(ex) * D + n) * I + kk]) * hh[kk];
          maxerr = std::max(maxerr, std::fabs(ref - o.as<float>()[size_t(pair) * D + n]));
          maxref = std::max(maxref, std::fabs(ref));
        }
        const double bu = double(nt) * (up.m.q_row + up.m.s_row) * I, bd = double(nt) * (dn.m.q_row + dn.m.s_row) * D;
        printf("v8 moe R=8 (%u tiles) b%d up %7.3f ms %6.1f GB/s, down %7.3f ms %6.1f GB/s  rel.err %.2e\n", nt, nb, ms_up,
               bu / ms_up / 1e6, ms_dn, bd / ms_dn / 1e6, maxerr / std::max(maxref, 1e-30));
      }
    }
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
      Buffer dummy = g.alloc(64, "dummy"), gtab = g.alloc(18432, "iq2_tab2");
      {
        auto e = g.begin();
        e->use(g.pipeline("nh_iq2_fill_tables2"));
        e->buf(0, gtab);
        e->dispatch(1, 1, 1, 256);
        e->commit_and_wait();
      }
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
