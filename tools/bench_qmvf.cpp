// Decode / small-batch matvec benchmark on one weight shape, kernels interleaved over rounds (A/B in one session).
//   bench_qmvf N K R [rounds] -- spec...
// spec: qmvf_*:<nt>:<rows>:<nsg>   (qmvf.metal; "_norm" in the name folds an RMSNorm; qmvf_gu_* runs gate + up:
//                                   two matrices of N x K, the bytes count both)
//       qmvrc1_b4[_add]              (verify.metal, one threadgroup per token row and 8-row block)
//       qmv2_b4_g64_r2[_norm|_add]   (qmv.metal, R = 1, x staged per threadgroup; also qmv3_b4_r2[_add|_norm])
//       qmvht_t2_r4 | qmvht_t3_r4    (half inner products, x staged as half)
//       mlp_gate_up                  (qmv.metal: norm + gate + up + SwiGLU, R = 1; two matrices)
//       qmvg_multi_f6g64[+norm]      (wfmt.metal: the generic multi-matrix form, one 6-bit matrix, R = 1)
// Weights (4-bit; 8-bit when every spec is a _b8 kernel, 6-bit group 64 when every spec is a _b6 / _f6g64 one) rotate
// over copies (~1 GB) so each dispatch reads DRAM. Prints the median and range of us per matrix (or
// gate + up pair) and GB/s per kernel, and the max relative error of 16 sampled outputs per token row against a
// double-precision CPU reference.
#include "metal/gpu.h"
#include "util/bf16.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace ni;

struct QmvParams { uint32_t N, K, x_stride, y_stride; float eps; };
struct QmvrParams { uint32_t N, K, R, x_ld, y_ld, add; };
struct QmvfParams {
  uint32_t K, R, x_ld, nseg;
  float eps;
  uint32_t N[4], y_ld[4], add[4], tg_begin[5];
  uint32_t write_xn;
};
struct QmvMulti2Params {
  uint32_t K, nseg;
  float eps;
  uint32_t norm, write_xn;
  uint32_t N[4], tg_begin[5], bits[4];
};

struct Mat { Buffer w, s, b; };

static bool starts(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: bench_qmvf N K R [rounds] -- spec...\n");
    return 1;
  }
  const uint32_t N = atoi(argv[1]), K = atoi(argv[2]), R = atoi(argv[3]);
  int rounds = 5, ai = 4;
  if (strcmp(argv[ai], "--") != 0) rounds = atoi(argv[ai++]);
  if (ai >= argc || strcmp(argv[ai], "--") != 0) fatal("expected -- before the kernel specs");
  ++ai;
  struct Spec {
    std::string name;
    uint32_t nt = 1, rows = 4, nsg = 2;
    bool gu = false, norm = false, add = false;
    Pipeline p;
    std::vector<double> ms;
    double err = 0;
  };
  std::vector<Spec> specs;
  for (; ai < argc; ++ai) {
    Spec s;
    std::string a = argv[ai];
    size_t c = a.find(':');
    s.name = a.substr(0, c);
    if (c != std::string::npos) sscanf(a.c_str() + c + 1, "%u:%u:%u", &s.nt, &s.rows, &s.nsg);
    s.gu = starts(s.name, "qmvf_gu") || s.name == "mlp_gate_up";
    s.norm = s.name.find("_norm") != std::string::npos || s.name == "mlp_gate_up";
    if (s.name.size() > 5 && s.name.compare(s.name.size() - 5, 5, "+norm") == 0) {  // qmvg_multi_*: norm by parameter
      s.name.resize(s.name.size() - 5);
      s.norm = true;
    }
    s.add = s.name.find("_add") != std::string::npos;
    s.p = Gpu::get().pipeline(s.name);
    specs.push_back(s);
  }
  Gpu& g = Gpu::get();
  std::mt19937 rng(7);
  // weight width: 8-bit when every spec names a _b8 kernel, 6-bit when every one a _b6 / _f6g64 kernel, else 4-bit
  bool all8 = !specs.empty(), all6 = !specs.empty();
  for (auto& sp : specs) {
    all8 = all8 && sp.name.find("_b8") != std::string::npos;
    all6 = all6 && (sp.name.find("_b6") != std::string::npos || sp.name.find("_f6g64") != std::string::npos);
  }
  const int bits = all8 ? 8 : all6 ? 6 : 4;
  const size_t wbytes = size_t(N) * K * bits / 8, gbytes = size_t(N) * K / 64 * 2;
  const size_t mat_bytes = wbytes + 2 * gbytes;
  int copies = std::clamp(int(1.2e9 / mat_bytes), 2, 400);
  if (copies % 2) ++copies;  // gate / up pairs
  std::vector<Mat> mats;
  std::uniform_real_distribution<float> d(-0.02f, 0.02f);
  for (int c = 0; c < copies; ++c) {
    Mat m{g.alloc(wbytes, nullptr, false), g.alloc(gbytes, nullptr, false), g.alloc(gbytes, nullptr, false)};
    for (size_t i = 0; i < wbytes / 4; ++i) m.w.as<uint32_t>()[i] = rng();
    for (size_t i = 0; i < gbytes / 2; ++i) {
      m.s.as<uint16_t>()[i] = f32_to_bf16(std::fabs(d(rng)) * 0.1f + 1e-4f);
      m.b.as<uint16_t>()[i] = f32_to_bf16(d(rng));
    }
    mats.push_back(m);
  }
  Buffer x = g.alloc(size_t(R) * K * 4), y = g.alloc(size_t(R) * N * 4), y2 = g.alloc(size_t(R) * N * 4);
  Buffer nw = g.alloc(size_t(K) * 2);
  std::normal_distribution<float> nd(0, 1);
  for (size_t i = 0; i < size_t(R) * K; ++i) x.as<float>()[i] = nd(rng) * 3.0f;
  for (uint32_t i = 0; i < K; ++i) nw.as<uint16_t>()[i] = f32_to_bf16(0.5f + 0.5f * std::fabs(nd(rng)));
  const float eps = 1e-6f;

  // reference x (normalized when the kernel folds a norm)
  auto ref_x = [&](bool norm) {
    std::vector<double> xr(size_t(R) * K);
    for (uint32_t t = 0; t < R; ++t) {
      double ss = 0;
      for (uint32_t k = 0; k < K; ++k) ss += double(x.as<float>()[t * K + k]) * x.as<float>()[t * K + k];
      const double inv = norm ? 1.0 / std::sqrt(ss / K + eps) : 1.0;
      for (uint32_t k = 0; k < K; ++k)
        xr[t * K + k] = x.as<float>()[t * K + k] * inv * (norm ? bf16_to_f32(nw.as<uint16_t>()[k]) : 1.0f);
    }
    return xr;
  };
  auto ref_row = [&](const Mat& m, const std::vector<double>& xr, uint32_t t, uint32_t n) {
    double acc = 0;
    const uint32_t mask = (1u << bits) - 1;
    const uint32_t* w = m.w.as<uint32_t>() + size_t(n) * K * bits / 32;
    for (uint32_t k = 0; k < K; ++k) {  // value k at bits [bits * k, bits * k + bits) of the row's stream
      const uint32_t b = bits * k, sh = b % 32;
      uint32_t q = w[b / 32] >> sh;
      if (sh + bits > 32) q |= w[b / 32 + 1] << (32 - sh);
      q &= mask;
      const double wr = double(bf16_to_f32(m.s.as<uint16_t>()[size_t(n) * K / 64 + k / 64])) * q +
                        bf16_to_f32(m.b.as<uint16_t>()[size_t(n) * K / 64 + k / 64]);
      acc += wr * xr[size_t(t) * K + k];
    }
    return acc;
  };

  auto encode = [&](Encoder& e, Spec& s, const Mat& m, const Mat& m2) {
    e.use(s.p);
    if (starts(s.name, "qmvf")) {
      QmvfParams p{};
      p.K = K; p.R = R; p.x_ld = K; p.nseg = 1; p.eps = eps;
      p.N[0] = p.N[1] = N; p.y_ld[0] = p.y_ld[1] = N; p.add[0] = s.add;
      const uint32_t tgs = (N + s.rows * s.nsg - 1) / (s.rows * s.nsg);
      p.tg_begin[0] = 0; p.tg_begin[1] = tgs;
      for (int i = 0; i < 4; ++i) {
        const Mat& mm = (s.gu && i == 1) ? m2 : m;
        e.buf(i * 4 + 0, mm.w); e.buf(i * 4 + 1, mm.s); e.buf(i * 4 + 2, mm.b); e.buf(i * 4 + 3, i == 1 ? y2 : y);
      }
      e.buf(16, x); e.val(17, p); e.buf(18, nw); e.buf(19, y2);
      e.dispatch((R + s.nt - 1) / s.nt, tgs, 1, 32 * s.nsg);
    } else if (starts(s.name, "qmvg_multi")) {  // 8 simdgroups x 2 rows, as Qwen35::enc_qmv_multi
      QmvMulti2Params p{};
      p.K = K; p.nseg = 1; p.eps = eps; p.norm = s.norm; p.N[0] = N; p.bits[0] = uint32_t(bits);
      p.tg_begin[1] = (N + 15) / 16;
      for (int i = 0; i < 4; ++i) {
        e.buf(i * 4 + 0, m.w); e.buf(i * 4 + 1, m.s); e.buf(i * 4 + 2, m.b); e.buf(i * 4 + 3, y);
      }
      e.buf(16, x); e.val(17, p); e.buf(18, nw); e.buf(19, y2);
      e.dispatch((N + 15) / 16, 1, 1, 256);
    } else if (starts(s.name, "qmvrc1")) {
      QmvrParams p{N, K, R, K, N, s.add ? 1u : 0u};
      e.buf(0, m.w); e.buf(1, m.s); e.buf(2, m.b); e.buf(3, x); e.buf(4, y); e.val(5, p);
      e.dispatch(R, (N + 7) / 8, 1, 64);
    } else if (starts(s.name, "qmvht")) {
      QmvrParams p{N, K, R, K, N, 0};
      e.buf(0, m.w); e.buf(1, m.s); e.buf(2, m.b); e.buf(3, x); e.buf(4, y); e.val(5, p);
      e.dispatch((N + 31) / 32, 1, 1, 256);
    } else if (s.name == "mlp_gate_up") {
      QmvParams p{N, K, K, N, eps};
      e.buf(0, m.w); e.buf(1, m.s); e.buf(2, m.b); e.buf(3, m2.w); e.buf(4, m2.s); e.buf(5, m2.b);
      e.buf(6, x); e.buf(7, nw); e.buf(8, y); e.val(9, p);
      e.dispatch((N + 15) / 16, 1, 1, 256);
    } else {  // qmv2_* / qmv3_*: 8 simdgroups x 2 rows
      QmvParams p{N, K, K, N, eps};
      e.buf(0, m.w); e.buf(1, m.s); e.buf(2, m.b); e.buf(3, x); e.buf(4, y); e.val(5, p);
      e.buf(6, y); e.buf(7, nw);
      e.dispatch((N + 15) / 16, 1, 1, 256);
    }
  };

  for (int rd = -1; rd < rounds; ++rd)
    for (auto& s : specs) {
      auto e = g.begin();
      const int step = s.gu ? 2 : 1;
      for (int c = 0; c + step - 1 < copies; c += step) encode(*e, s, mats[c], mats[c + step - 1]);
      e->commit_and_wait();
      if (rd >= 0) s.ms.push_back(e->gpu_ms() / (copies / step));
      if (rd == rounds - 1) {
        // correctness: last dispatch's outputs (y for single matrices / the SwiGLU output)
        if (s.add) continue;  // y accumulated over the copies: not checked
        const Mat& m = mats[copies - step];
        const Mat& m2 = mats[copies - 1];
        const auto xr = ref_x(s.norm);
        const uint32_t TR =
            (starts(s.name, "qmv2") || starts(s.name, "qmv3") || starts(s.name, "qmvg") || s.name == "mlp_gate_up") ? 1 : R;
        double mx = 0;
        for (uint32_t t = 0; t < TR; ++t)
          for (int i = 0; i < 16; ++i) {
            const uint32_t n = uint32_t((uint64_t(i) * 2654435761u) % N);
            double ref = ref_row(m, xr, t, n);
            if (s.gu) {
              const double u = ref_row(m2, xr, t, n);
              ref = ref / (1.0 + std::exp(-ref)) * u;
            }
            const double got = y.as<float>()[size_t(t) * N + n];
            mx = std::max(mx, std::fabs(got - ref) / (std::fabs(ref) + 1e-2));
          }
        s.err = mx;
      }
    }
  for (auto& s : specs) {
    std::vector<double> v = s.ms;
    std::sort(v.begin(), v.end());
    const double med = v[v.size() / 2];
    const double bytes = double(mat_bytes) * (s.gu ? 2 : 1);
    printf("N=%6u K=%5u R=%u  %-28s nt%u r%u sg%-2u %8.2f us (%.2f-%.2f)  %6.1f GB/s  err %.1e\n", N, K, R,
           s.name.c_str(), s.nt, s.rows, s.nsg, med * 1000, v.front() * 1000, v.back() * 1000, bytes / (med * 1e-3) / 1e9,
           s.err);
  }
  return 0;
}
