// Draft attention (dfl_attn / dfl_attn_mma + dfl_attn_reduce) against a CPU fp64 reference with
// random data at several context lengths, including P > 2047 (wrapped ring, per-row window bound
// j >= P + r - 2047, SPEC §5). Also times both kernels.
//   dflash_attn_test [P ...]   (default 1 85 700 2047 2048 2055 4100 9000)
// Then dfl_attn_qk (fused q / k norm + RoPE) against dfl_qk_prep + dfl_attn_mma at 8 / 16 / 32 splits, host- and
// GPU-planned. DAT_MMA_ONLY=1 skips the scalar kernel (its threadgroup memory exceeds what the two Metal validation
// layers together allow), DAT_REPS=n times n back-to-back pairs per command buffer, DAT_PARTS=1/2 only the attention
// / the reduce, DAT_SRC=<file> + DAT_KERNEL=<name> (+ DAT_THREADS) a variant appended to the library.
#include "metal/gpu.h"
#include "model/dflash2.h"
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
  if (const char* src = getenv("DAT_SRC")) {  // extra kernel source (variants of dfl_attn_qk, picked by DAT_KERNEL)
    FILE* f = fopen(src, "rb");
    std::string x;
    if (f) {
      char buf[4096];
      size_t n;
      while ((n = fread(buf, 1, sizeof buf, f)) > 0) x.append(buf, n);
      fclose(f);
    }
    g.compile(x);
  }
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  const int R = 8, HQ = 32, HKV = 8, D = 128, W = 2048, S = 32;
  Buffer Q = g.alloc(size_t(HQ) * R * D * 4), Kb = g.alloc(size_t(HKV) * R * D * 4), Vb = g.alloc(size_t(HKV) * R * D * 4);
  Buffer rk = g.alloc(size_t(HKV) * W * D * 2), rv = g.alloc(size_t(HKV) * W * D * 2);
  Buffer po = g.alloc(size_t(S) * HQ * R * D * 4), pml = g.alloc(size_t(S) * HQ * R * 2 * 4);
  Buffer xh = g.alloc(size_t(2 * R) * 4096 * 2), xs = g.alloc(64 * 8 * 4);
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
    for (uint32_t mchunk : {64u, 128u, 256u})
    for (int variant = getenv("DAT_MMA_ONLY") ? 1 : 0; variant < 2; ++variant) {
      uint32_t lo, chunk, nsplit;
      dfl_attn_plan(P, S, mchunk, &lo, &chunk, &nsplit);
      kp::DflAttn p{uint32_t(P), lo, chunk, nsplit, scale};
      kp::DflAttnRed rp{nsplit, uint32_t(P), lo, chunk};
      auto enc = [&](Encoder& e) {
        e.use(variant ? pm : pa);
        e.buf(0, Q); e.buf(1, rk); e.buf(2, rv); e.buf(3, Kb); e.buf(4, Vb); e.buf(5, po); e.buf(6, pml);
        e.val(7, p);
        e.buf(8, xs);  // drafter state: unused with a host-known P
        e.dispatch(nsplit, 8, 1, 128);
        e.use(pr);
        e.buf(0, po); e.buf(1, pml); e.buf(2, xh); e.buf(3, xs);
        e.val(4, rp);
        e.buf(5, xs);
        e.dispatch(32, 8, 1, 128);
      };
      // DAT_REPS=n: n back-to-back attention + reduce pairs per command buffer (the in-pipeline cost per pair)
      static const int reps = getenv("DAT_REPS") ? std::max(1, atoi(getenv("DAT_REPS"))) : 1;
      std::vector<double> ts;
      for (int it = 0; it < 12; ++it) {
        auto e = g.begin();
        for (int k = 0; k < reps; ++k) enc(*e);
        e->commit_and_wait();
        if (it >= 2) ts.push_back(e->gpu_ms() / reps);
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
      printf("P=%5d %-12s min chunk %3u: splits %2u x chunk %4u: rel err %.2e max abs %.2e (sums %.1e)  GPU attn+reduce median %.4f ms\n",
             P, variant ? "dfl_attn_mma" : "dfl_attn", mchunk, nsplit, chunk, std::sqrt(num / den), mx, serr, ts[ts.size() / 2]);
    }
  }
  // dfl_attn_qk (q / k norm + RoPE fused) against dfl_qk_prep + dfl_attn_mma on raw projections, host-planned and
  // GPU-planned (P from the drafter state, adaptive chunk), at 8 / 16 / 32 splits
  {
    Buffer q = g.alloc(size_t(R) * 4096 * 4), kv = g.alloc(size_t(R) * 2048 * 4), qn = g.alloc(128 * 2), kn = g.alloc(128 * 2);
    Buffer inv = g.alloc(64 * 4), cs = g.alloc(8 * 64 * 8), gst = g.alloc(16), xh2 = g.alloc(size_t(2 * R) * 4096 * 2),
           xs2 = g.alloc(64 * 8 * 4);
    for (size_t i = 0; i < size_t(R) * 4096; ++i) q.as<float>()[i] = nd(rng) * 2.0f;
    for (size_t i = 0; i < size_t(R) * 2048; ++i) kv.as<float>()[i] = nd(rng) * 2.0f;
    for (int i = 0; i < 128; ++i) {
      qn.as<uint16_t>()[i] = f32_to_bf16(1.0f + 0.3f * nd(rng));
      kn.as<uint16_t>()[i] = f32_to_bf16(1.0f + 0.3f * nd(rng));
    }
    for (int i = 0; i < 64; ++i) inv.as<float>()[i] = 1.0f / std::pow(1e7f, float(2 * i) / 128.0f);
    for (int P : Ps) {
      if (P + 8 >= (1 << 30)) continue;
      uint32_t* gs = gst.as<uint32_t>();
      gs[0] = uint32_t(P); gs[1] = 0; gs[2] = uint32_t(P);
      auto rope = [&](Encoder& e, uint32_t Pk) {
        e.use(g.pipeline("dfl_rope_table"));
        e.buf(0, inv); e.buf(1, cs);
        e.val(2, Pk);
        e.buf(3, gst);
        e.dispatch(1, 1, 1, 512);
      };
      for (uint32_t splits : {8u, 16u, 32u})
        for (int gpu = 0; gpu < 2; ++gpu) {
          uint32_t lo, chunk, nsplit;
          dfl_attn_plan(P, splits, 0, &lo, &chunk, &nsplit);
          const uint32_t Pk = gpu ? 0xffffffffu : uint32_t(P);
          // reference: dfl_qk_prep + dfl_attn_mma (host plan)
          {
            auto e = g.begin();
            rope(*e, uint32_t(P));
            kp::DflQk qp{uint32_t(P), 1e-6f, 4096, 2048};
            e->use(g.pipeline("dfl_qk_prep"));
            e->buf(0, q); e->buf(1, kv); e->buf(2, qn); e->buf(3, kn); e->buf(4, cs);
            e->buf(5, Q); e->buf(6, Kb); e->buf(7, Vb);
            e->val(8, qp);
            e->dispatch(40, 8, 1, 128);
            kp::DflAttn ap{uint32_t(P), lo, chunk, nsplit, scale};
            e->use(pm);
            e->buf(0, Q); e->buf(1, rk); e->buf(2, rv); e->buf(3, Kb); e->buf(4, Vb); e->buf(5, po); e->buf(6, pml);
            e->val(7, ap);
            e->buf(8, gst);
            e->dispatch(nsplit, 8, 1, 128);
            kp::DflAttnRed rp{nsplit, uint32_t(P), lo, chunk};
            e->use(pr);
            e->buf(0, po); e->buf(1, pml); e->buf(2, xh); e->buf(3, xs);
            e->val(4, rp);
            e->buf(5, gst);
            e->dispatch(32, 8, 1, 128);
            e->commit_and_wait();
          }
          const uint32_t c2 = gpu ? 0u : chunk, n2 = gpu ? splits : nsplit;
          static const int parts = getenv("DAT_PARTS") ? atoi(getenv("DAT_PARTS")) : 3;  // 1 attention, 2 reduce
          auto enc = [&](Encoder& e) {
            kp::DflAttnQk ap{Pk, gpu ? 0u : lo, c2, n2, scale, 1e-6f, 4096, 2048};
            if (parts & 1) {
              e.use(g.pipeline(getenv("DAT_KERNEL") ? getenv("DAT_KERNEL") : "dfl_attn_qk"));
              e.buf(0, q); e.buf(1, kv); e.buf(2, qn); e.buf(3, kn); e.buf(4, cs); e.buf(5, rk); e.buf(6, rv);
              e.buf(7, po); e.buf(8, pml);
              e.val(9, ap);
              e.buf(10, gst);
              e.dispatch(n2, 8, 1, getenv("DAT_THREADS") ? uint32_t(atoi(getenv("DAT_THREADS"))) : 128u);
            }
            if (parts & 2) {
              kp::DflAttnRed rp{n2, Pk, gpu ? 0u : lo, c2};
              e.use(pr);
              e.buf(0, po); e.buf(1, pml); e.buf(2, xh2); e.buf(3, xs2);
              e.val(4, rp);
              e.buf(5, gst);
              e.dispatch(32, 8, 1, 128);
            }
          };
          {
            auto e = g.begin();
            rope(*e, Pk);
            enc(*e);
            e->commit_and_wait();
          }
          double md = 0, ref = 0, sd = 0;
          for (int i = 0; i < R * 4096; ++i) {
            md = std::max(md, std::fabs(double(xh2.as<__fp16>()[i]) - double(xh.as<__fp16>()[i])));
            ref = std::max(ref, std::fabs(double(xh.as<__fp16>()[i])));
          }
          for (int i = 0; i < 64 * 8; ++i) sd = std::max(sd, std::fabs(double(xs2.as<float>()[i]) - xs.as<float>()[i]));
          static const int reps = getenv("DAT_REPS") ? std::max(1, atoi(getenv("DAT_REPS"))) : 1;
          std::vector<double> ts;
          for (int it = 0; it < 7; ++it) {
            auto e = g.begin();
            for (int k = 0; k < reps; ++k) enc(*e);
            e->commit_and_wait();
            if (it >= 2) ts.push_back(e->gpu_ms() / reps);
          }
          std::sort(ts.begin(), ts.end());
          printf("P=%5d dfl_attn_qk %s splits %2u (chunk %3u x %2u): max |o - ref| %.2e (max |ref| %.2e, sums %.1e)  "
                 "GPU attn+reduce median %.4f ms\n", P, gpu ? "gpu " : "host", splits, chunk, nsplit, md, ref, sd,
                 ts[ts.size() / 2]);
        }
    }
  }
  return 0;
}
