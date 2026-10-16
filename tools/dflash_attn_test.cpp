// Draft attention (dfl_attn / dfl_attn_mma + dfl_attn_reduce) against a CPU fp64 reference with
// random data at several context lengths, including P > 2047 (wrapped ring, per-row window bound
// j >= P + r - 2047, SPEC §5). Also times both kernels.
//   dflash_attn_test [P ...]   (default 1 85 700 2047 2048 2055 4100 9000)
#include "metal/gpu.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace ni;

int main(int argc, char** argv) {
  std::vector<int> Ps;
  for (int i = 1; i < argc; ++i) Ps.push_back(atoi(argv[i]));
  if (Ps.empty()) Ps = {1, 85, 700, 2047, 2048, 2055, 4100, 9000};
  Gpu& g = Gpu::get();
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  const int R = 8, HQ = 32, HKV = 8, D = 128, W = 2048, S = 32;
  Buffer Q = g.alloc(size_t(HQ) * R * D * 4), Kb = g.alloc(size_t(HKV) * R * D * 4), Vb = g.alloc(size_t(HKV) * R * D * 4);
  Buffer rk = g.alloc(size_t(HKV) * W * D * 2), rv = g.alloc(size_t(HKV) * W * D * 2);
  Buffer po = g.alloc(size_t(S) * HQ * R * D * 4), pml = g.alloc(size_t(S) * HQ * R * 2 * 4);
  Buffer xh = g.alloc(size_t(R) * 4096 * 2), xs = g.alloc(64 * 8 * 4);
  for (size_t i = 0; i < size_t(HQ) * R * D; ++i) Q.as<float>()[i] = nd(rng) * 0.3f;
  for (size_t i = 0; i < size_t(HKV) * R * D; ++i) {
    Kb.as<float>()[i] = nd(rng);
    Vb.as<float>()[i] = nd(rng);
  }
  for (size_t i = 0; i < size_t(HKV) * W * D; ++i) {
    rk.as<uint16_t>()[i] = f32_to_bf16(nd(rng));
    rv.as<uint16_t>()[i] = f32_to_bf16(nd(rng));
  }
  const Pipeline pa = g.pipeline("dfl_attn"), pm = g.pipeline("dfl_attn_mma"), pr = g.pipeline("dfl_attn_reduce");
  const float scale = 1.0f / std::sqrt(128.0f);
  for (int P : Ps) {
    // reference: row r sees context j in [max(0, P + r - 2047), P - 1] (ring slot j mod 2048) + the 8 block rows
    std::vector<double> ref(size_t(R) * HQ * D);
    for (int h = 0; h < HQ; ++h) {
      const int g8 = h / 4;
      for (int r = 0; r < R; ++r) {
        const float* q = Q.as<float>() + (size_t(h) * R + r) * D;
        std::vector<double> sc;
        std::vector<const void*> vrow;
        std::vector<bool> isbf;
        for (int j = std::max(0, P + r - 2047); j < P; ++j) {
          const uint16_t* k = rk.as<uint16_t>() + (size_t(g8) * W + (j & 2047)) * D;
          double s = 0;
          for (int x = 0; x < D; ++x) s += double(q[x]) * bf16_to_f32(k[x]);
          sc.push_back(s * scale);
          vrow.push_back(rv.as<uint16_t>() + (size_t(g8) * W + (j & 2047)) * D);
          isbf.push_back(true);
        }
        for (int b = 0; b < R; ++b) {
          const float* k = Kb.as<float>() + (size_t(g8) * R + b) * D;
          double s = 0;
          for (int x = 0; x < D; ++x) s += double(q[x]) * k[x];
          sc.push_back(s * scale);
          vrow.push_back(Vb.as<float>() + (size_t(g8) * R + b) * D);
          isbf.push_back(false);
        }
        double mx = -1e300, z = 0;
        for (double s : sc) mx = std::max(mx, s);
        std::vector<double> o(D, 0.0);
        for (size_t i = 0; i < sc.size(); ++i) {
          const double w = std::exp(sc[i] - mx);
          z += w;
          for (int x = 0; x < D; ++x)
            o[x] += w * (isbf[i] ? bf16_to_f32(((const uint16_t*)vrow[i])[x]) : ((const float*)vrow[i])[x]);
        }
        for (int x = 0; x < D; ++x) ref[(size_t(r) * HQ + h) * D + x] = o[x] / z;
      }
    }
    const uint32_t lo = P > 2047 ? uint32_t(P - 2047) : 0u, L = uint32_t(P) - lo;
    uint32_t nsplit = std::max<uint32_t>(1, std::min<uint32_t>(S, (L + 63) / 64));
    uint32_t chunk = ((L + nsplit - 1) / nsplit + 15) / 16 * 16;
    if (!chunk) chunk = 16;
    nsplit = std::max<uint32_t>(1, (L + chunk - 1) / chunk);
    for (int variant = 0; variant < 2; ++variant) {
      kp::DflAttn p{uint32_t(P), lo, chunk, nsplit, scale};
      kp::DflAttnRed rp{nsplit};
      auto enc = [&](Encoder& e) {
        e.use(variant ? pm : pa);
        e.buf(0, Q); e.buf(1, rk); e.buf(2, rv); e.buf(3, Kb); e.buf(4, Vb); e.buf(5, po); e.buf(6, pml);
        e.val(7, p);
        e.dispatch(nsplit, 8, 1, 128);
        e.use(pr);
        e.buf(0, po); e.buf(1, pml); e.buf(2, xh); e.buf(3, xs);
        e.val(4, rp);
        e.dispatch(32, 8, 1, 128);
      };
      std::vector<double> ts;
      for (int it = 0; it < 12; ++it) {
        auto e = g.begin();
        enc(*e);
        e->commit_and_wait();
        if (it >= 2) ts.push_back(e->gpu_ms());
      }
      std::sort(ts.begin(), ts.end());
      double num = 0, den = 0, mx = 0;
      for (int r = 0; r < R; ++r)
        for (int h = 0; h < HQ; ++h)
          for (int x = 0; x < D; ++x) {
            const double a = double(xh.as<__fp16>()[r * 4096 + h * 128 + x]), b = ref[(size_t(r) * HQ + h) * D + x];
            num += (a - b) * (a - b);
            den += b * b;
            mx = std::max(mx, std::fabs(a - b));
          }
      // the group sums of the half table
      double serr = 0;
      for (int gi = 0; gi < 64; ++gi)
        for (int r = 0; r < R; ++r) {
          double s = 0;
          for (int x = 0; x < 64; ++x) s += double(xh.as<__fp16>()[r * 4096 + gi * 64 + x]);
          serr = std::max(serr, std::fabs(s - xs.as<float>()[gi * 8 + r]));
        }
      printf("P=%5d %-12s splits %2u x chunk %4u: rel err %.2e max abs %.2e (sums %.1e)  GPU attn+reduce median %.4f ms\n",
             P, variant ? "dfl_attn_mma" : "dfl_attn", nsplit, chunk, std::sqrt(num / den), mx, serr, ts[ts.size() / 2]);
    }
  }
  return 0;
}
