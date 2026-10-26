// Draft attention (dfl_attn / dfl_attn_mma + dfl_attn_reduce) against a CPU fp64 reference with
// random data at several context lengths, including P > 2047 (wrapped ring, per-row window bound
// j >= P + r - 2047, SPEC §5). Also times both kernels.
//   dflash_attn_test [P ...]   (default 1 85 700 2047 2048 2055 4100 9000)
// Then dfl_attn_qk (fused q / k norm + RoPE) against dfl_qk_prep + dfl_attn_mma at 8 / 16 / 32 splits, host- and
// GPU-planned. DAT_MMA_ONLY=1 skips the scalar kernel (its threadgroup memory exceeds what the two Metal validation
// layers together allow), DAT_REPS=n times n back-to-back pairs per command buffer, DAT_PARTS=1/2 only the attention
// / the reduce, DAT_SRC=<file> + DAT_KERNEL=<name> (+ DAT_THREADS) a variant appended to the library,
// DAT_PAIR_ONLY=1 only the last check (the pair GEMM, the one kernel here with no threadgroup-memory problem).
// DAT_QK16_ONLY=1 only the 16-row attention check (dfl_attn_qk16 + dfl_attn_reduce with nr = 16): the earlier two
// skip dfl_attn_mma, whose threadgroup memory does not fit under the validation layers. dfl_attn_qk16 compiles to
// more threadgroup memory than MTL_DEBUG_LAYER's 32768-byte assertion as well (51200), so the check runs without the
// layers, like the scalar kernel.
// The last check is the 16-row drafting block's pair GEMM (qmm8t2d / qmm8t2dp) against two per-tile qmm8t dispatches
// on the same T tables, bitwise; it needs the tensor library (Apple10, macOS 26) and is skipped otherwise. main()
// returns 1 if any exactness check fails, 3 when a check selected by DAT_QK16_ONLY / DAT_PAIR_ONLY did not run.
#include "metal/gpu.h"
#include "model/dflash2.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"

namespace ni {
extern const char* const kTensorKernelSource;  // src/kernels/tensor/*: qmm8t2d (the 16-row drafting block's pair GEMM)
}

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
  if (getenv("DAT_PAIR_ONLY")) Ps.clear();  // only the pair-GEMM check, for a validation-layer run of that kernel
  const bool pair_only = getenv("DAT_PAIR_ONLY") != nullptr;
  const bool qk16_only = getenv("DAT_QK16_ONLY") != nullptr;
  Gpu& g = Gpu::get();
  int fails = 0;  // every exactness check below: a script can tell a failing run from a passing one
  bool ran16 = false, ran_pair = false;  // whether each gated check actually ran on this GPU
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
    if (qk16_only) break;
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
      const bool ok = std::sqrt(num / den) <= 2e-3 && mx <= 5e-3 && serr <= 1e-3;  // half-precision output
      if (!ok) ++fails;
      printf("P=%5d %-12s min chunk %3u: splits %2u x chunk %4u: rel err %.2e max abs %.2e (sums %.1e)  GPU attn+reduce "
             "median %.4f ms%s\n", P, variant ? "dfl_attn_mma" : "dfl_attn", mchunk, nsplit, chunk, std::sqrt(num / den),
             mx, serr, ts[ts.size() / 2], ok ? "" : "  FAIL");
    }
  }
  // dfl_attn_qk (q / k norm + RoPE fused) against dfl_qk_prep + dfl_attn_mma on raw projections, host-planned and
  // GPU-planned (P from the drafter state, adaptive chunk), at 8 / 16 / 32 splits
  {
    Buffer q = g.alloc(size_t(R) * 4096 * 4), kv = g.alloc(size_t(R) * 2048 * 4), qn = g.alloc(128 * 2), kn = g.alloc(128 * 2);
    Buffer inv = g.alloc(64 * 4), cs = g.alloc(8 * 64 * 8), gst = g.alloc(16), xh2 = g.alloc(size_t(2 * R) * 4096 * 2),
           xs2 = g.alloc(64 * 8 * 4);
    // 16-row block: the raw q / kv of 16 rows, the reference's normed q / k / v stitched into the kernel's row order,
    // and the reduce's two X8 tiles (values then the permuted T table, 16 x 4096 halves each)
    const int R16 = 16;
    Buffer q16 = g.alloc(size_t(R16) * 4096 * 4), kv16 = g.alloc(size_t(R16) * 2048 * 4), cs16 = g.alloc(16 * 64 * 8),
           cs8b = g.alloc(8 * 64 * 8), qa = g.alloc(size_t(8) * 4096 * 4), kb8 = g.alloc(size_t(8) * 2048 * 4),
           vb8 = g.alloc(size_t(8) * 2048 * 4), Q16 = g.alloc(size_t(R16) * 4096 * 4), K16 = g.alloc(size_t(R16) * 2048 * 4),
           V16 = g.alloc(size_t(R16) * 2048 * 4), po16 = g.alloc(size_t(S) * HQ * R16 * D * 4),
           pml16 = g.alloc(size_t(S) * HQ * R16 * 2 * 4), xh16 = g.alloc(size_t(2 * R16) * 4096 * 2), xs16 = g.alloc(2 * 512 * 4);
    for (size_t i = 0; i < size_t(R16) * 4096; ++i) q16.as<float>()[i] = nd(rng) * 2.0f;
    for (size_t i = 0; i < size_t(R16) * 2048; ++i) kv16.as<float>()[i] = nd(rng) * 2.0f;
    for (size_t i = 0; i < size_t(R) * 4096; ++i) q.as<float>()[i] = nd(rng) * 2.0f;
    for (size_t i = 0; i < size_t(R) * 2048; ++i) kv.as<float>()[i] = nd(rng) * 2.0f;
    for (int i = 0; i < 128; ++i) {
      qn.as<uint16_t>()[i] = f32_to_bf16(1.0f + 0.3f * nd(rng));
      kn.as<uint16_t>()[i] = f32_to_bf16(1.0f + 0.3f * nd(rng));
    }
    for (int i = 0; i < 64; ++i) inv.as<float>()[i] = 1.0f / std::pow(1e7f, float(2 * i) / 128.0f);
    for (int P : Ps) {
      const bool skip8 = qk16_only || P + 8 >= (1 << 30);
      uint32_t* gs = gst.as<uint32_t>();
      gs[0] = uint32_t(P); gs[1] = 0; gs[2] = uint32_t(P);
      auto rope = [&](Encoder& e, uint32_t Pk) {
        e.use(g.pipeline("dfl_rope_table"));
        e.buf(0, inv); e.buf(1, cs);
        e.val(2, Pk);
        e.buf(3, gst);
        e.val(4, 512u);  // rows of the table: 8 rows x 64 dims
        e.dispatch(1, 1, 1, 512);
      };
      if (!skip8)
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
          const bool ok = md <= 5e-3 && sd <= 5e-3;  // the fused path must match prep + mma
          if (!ok) ++fails;
          printf("P=%5d dfl_attn_qk %s splits %2u (chunk %3u x %2u): max |o - ref| %.2e (max |ref| %.2e, sums %.1e)  "
                 "GPU attn+reduce median %.4f ms%s\n", P, gpu ? "gpu " : "host", splits, chunk, nsplit, md, ref, sd,
                 ts[ts.size() / 2], ok ? "" : "  FAIL");
        }
      // The 16-row block (NI_DFLASH_BLOCK=16): dfl_attn_qk16 dispatches both row tiles (tg.z = 0, 1), every row sees
      // all 16 block keys, and dfl_attn_reduce sums with nr = 16 into two X8 tiles. A wrong row-tile index there would
      // not fault, it would only lower acceptance, so both kernels are checked against the same fp64 reference as the
      // 8-row ones above. The reference's normed / roped q, k and v come from two dfl_qk_prep calls (rows 0-7 at P,
      // rows 8-15 at P + 8) stitched into the row order the kernel reads.
      if (g.apple_family() >= 10) {  // dfl_attn_qk16 runs on the matrix units
        auto rope16 = [&](Encoder& e, uint32_t Pk, Buffer& dst, uint32_t rows) {
          e.use(g.pipeline("dfl_rope_table"));
          e.buf(0, inv); e.buf(1, dst);
          e.val(2, Pk);
          e.buf(3, gst);
          e.val(4, 64u * rows);
          e.dispatch(1, 1, 1, 64u * rows);
        };
        for (int half = 0; half < 2; ++half) {  // the reference's q / k / v, 8 rows at a time
          auto e = g.begin();
          rope16(*e, uint32_t(P) + uint32_t(8 * half), half ? cs8b : cs16, 8u);  // rows 0-7 of that half
          kp::DflQk qp{uint32_t(P) + uint32_t(8 * half), 1e-6f, 4096, 2048};
          e->use(g.pipeline("dfl_qk_prep"));
          e->buf(0, q16, size_t(half) * 8 * 4096 * 4);
          e->buf(1, kv16, size_t(half) * 8 * 2048 * 4);
          e->buf(2, qn); e->buf(3, kn); e->buf(4, half ? cs8b : cs16);
          e->buf(5, qa); e->buf(6, kb8); e->buf(7, vb8);
          e->val(8, qp);
          e->dispatch(40, 8, 1, 128);
          e->commit_and_wait();
          for (int h = 0; h < HQ; ++h)
            for (int r = 0; r < 8; ++r)
              for (int d = 0; d < D; ++d)
                Q16.as<float>()[size_t(8 * half + r) * 4096 + h * 128 + d] = qa.as<float>()[(h * 8 + r) * 128 + d];
          for (int gh = 0; gh < HKV; ++gh)
            for (int r = 0; r < 8; ++r)
              for (int d = 0; d < D; ++d) {
                K16.as<float>()[size_t(8 * half + r) * 2048 + gh * 128 + d] = kb8.as<float>()[(gh * 8 + r) * 128 + d];
                V16.as<float>()[size_t(8 * half + r) * 2048 + gh * 128 + d] = vb8.as<float>()[(gh * 8 + r) * 128 + d];
              }
        }
        std::vector<double> ref16(size_t(R16) * HQ * D);
        for (int h = 0; h < HQ; ++h) {
          const int g8 = h / 4;
          for (int r = 0; r < R16; ++r) {
            const float* q = Q16.as<float>() + size_t(r) * 4096 + size_t(h) * D;
            std::vector<double> sc;
            std::vector<const void*> vrow;
            std::vector<bool> isbf;
            for (int j = std::max(0, P + r - 2047); j < P; ++j) {
              const uint16_t* k = rk.as<uint16_t>() + (size_t(g8) * W + (j & 2047)) * D;
              double v = 0;
              for (int x = 0; x < D; ++x) v += double(q[x]) * bf16_to_f32(k[x]);
              sc.push_back(v * scale);
              vrow.push_back(rv.as<uint16_t>() + (size_t(g8) * W + (j & 2047)) * D);
              isbf.push_back(true);
            }
            for (int b = 0; b < R16; ++b) {  // all 16 block keys are visible to every row
              const float* k = K16.as<float>() + size_t(b) * 2048 + size_t(g8) * D;
              double v = 0;
              for (int x = 0; x < D; ++x) v += double(q[x]) * k[x];
              sc.push_back(v * scale);
              vrow.push_back(V16.as<float>() + size_t(b) * 2048 + size_t(g8) * D);
              isbf.push_back(false);
            }
            double mx = -1e300, z = 0;
            for (double v : sc) mx = std::max(mx, v);
            std::vector<double> o(D, 0.0);
            for (size_t i = 0; i < sc.size(); ++i) {
              const double w = std::exp(sc[i] - mx);
              z += w;
              for (int x = 0; x < D; ++x)
                o[x] += w * (isbf[i] ? bf16_to_f32(((const uint16_t*)vrow[i])[x]) : ((const float*)vrow[i])[x]);
            }
            for (int x = 0; x < D; ++x) ref16[(size_t(r) * HQ + h) * D + x] = o[x] / z;
          }
        }
        for (uint32_t splits : {8u, 16u}) {
          uint32_t lo, chunk, nsplit;
          dfl_attn_plan(P, splits, 0, &lo, &chunk, &nsplit);
          auto enc = [&](Encoder& e) {
            kp::DflAttnQk ap{uint32_t(P), lo, chunk, nsplit, scale, 1e-6f, 4096, 2048};
            e.use(g.pipeline("dfl_attn_qk16"));
            e.buf(0, q16); e.buf(1, kv16); e.buf(2, qn); e.buf(3, kn); e.buf(4, cs16);
            e.buf(5, rk); e.buf(6, rv); e.buf(7, po16); e.buf(8, pml16);
            e.val(9, ap);
            e.buf(10, gst);
            e.dispatch(nsplit, 8, 2, 128);  // two row tiles
            kp::DflAttnRed rp{nsplit, uint32_t(P), lo, chunk, uint32_t(R16)};
            e.use(pr);
            e.buf(0, po16); e.buf(1, pml16); e.buf(2, xh16); e.buf(3, xs16);
            e.val(4, rp);
            e.buf(5, gst);
            e.dispatch(32, R16, 1, 128);
          };
          {
            auto e = g.begin();
            rope16(*e, uint32_t(P), cs16, 16u);  // the full 16-row table, for the kernel
            enc(*e);
            e->commit_and_wait();
          }
          double num = 0, den = 0, mx = 0, refmax = 0;
          for (int r = 0; r < R16; ++r)
            for (int h = 0; h < HQ; ++h)
              for (int x = 0; x < D; ++x) {
                const size_t i = size_t((r >> 3) * 16 + (r & 7)) * 4096 + size_t(h) * 128 + x;
                const double a = double(xh16.as<__fp16>()[i]), b = ref16[(size_t(r) * HQ + h) * D + x];
                num += (a - b) * (a - b);
                den += b * b;
                mx = std::max(mx, std::fabs(a - b));
                refmax = std::max(refmax, std::fabs(b));
              }
          const double rel = std::sqrt(num / den);
          const bool ok = rel <= 5e-3 && mx <= 2e-2;  // half-precision output against an fp64 reference
          if (!ok) ++fails;
          ran16 = true;
          printf("P=%5d dfl_attn_qk16 + reduce(nr 16) splits %2u (chunk %3u x %2u): rel err %.2e max abs %.2e "
                 "(max |ref| %.2e)%s\n", P, splits, chunk, nsplit, rel, mx, refmax, ok ? "" : "  FAIL");
        }
      } else {
        printf("dfl_attn_qk16: skipped, needs Apple10\n");
      }
    }
  }
  // A 16-row drafting block is two 8-row tiles (dfl_put2): qmm8t2d computes the pair from one weight load where the
  // per-tile path (qmm8t, dflash2.cpp enc_qmm8 with NI_DFL_PAIR=0) streams the weights once per tile. The pair must be
  // exact against two per-tile dispatches on the same tables: every one of the 16 x N outputs bitwise identical.
  // T tables and group sums come from x8_prep2 (write_t), the layout dfl_put2 writes; both tiling rules the drafter
  // picks are covered: K 2048 (K / 64 = 32: nt 16, no prefetch, ks 8) and K 4096 (K / 64 = 64: nt 32, prefetch, ks 4).
  const bool tensor_ok = [] {
    if (__builtin_available(macOS 26.0, *)) return true;
    return false;
  }();
  if (g.apple_family() >= 10 && tensor_ok) {
    g.compile_extra("tensor", ni::kTensorKernelSource, true);
    if (!g.pipeline("qmm8t2d_b4_nt16", false).pso) {
      printf("qmm8t2d: the tensor library did not build, skipped\n");
    } else {
      ran_pair = true;
      struct Case { uint32_t K, nt, pf, ks; };
      for (const Case c : {Case{2048, 16, 0, 8}, Case{4096, 32, 1, 4}}) {
        const uint32_t K = c.K, N = 1024, R16 = 16, gpr = K / 64;
        Buffer w = g.alloc(size_t(N) * K / 8 * 4), s = g.alloc(size_t(N) * gpr * 2), b = g.alloc(size_t(N) * gpr * 2);
        for (size_t i = 0; i < size_t(N) * K / 8; ++i) w.as<uint32_t>()[i] = rng();
        for (size_t i = 0; i < size_t(N) * gpr; ++i) {
          s.as<uint16_t>()[i] = f32_to_bf16(0.001f + 0.02f * std::fabs(nd(rng)));
          b.as<uint16_t>()[i] = f32_to_bf16(0.01f * nd(rng));
        }
        Buffer x = g.alloc(size_t(R16) * K * 4), xw = g.alloc(256 * 2), xh = g.alloc(size_t(R16) * K * 2),
               tt = g.alloc(size_t(R16) * K * 2), sm = g.alloc(size_t(gpr) * R16 * 4),
               ya = g.alloc(size_t(R16) * N * 4), yb = g.alloc(size_t(R16) * N * 4);
        for (size_t i = 0; i < size_t(R16) * K; ++i) x.as<float>()[i] = nd(rng);
        for (int i = 0; i < 256; ++i) xw.as<uint16_t>()[i] = f32_to_bf16(1.0f);
        {  // the 16 rows' T tables (8 rows per tile, tile t at 8 t rows) and their group sums, as dfl_put2 lays them out
          auto e = g.begin();
          kp::X8Prep pp{K, R16, K, 0, 0, 1e-6f, K, 0, 0, 1};
          e->use(g.pipeline("x8_prep2"));
          e->buf(0, x); e->buf(1, xw); e->buf(2, xh); e->buf(3, sm); e->buf(4, x);
          e->val(5, pp);
          e->buf(6, xh); e->buf(7, sm); e->buf(8, tt);
          e->buf(9, xh);  // (unused unless write_t_lo)
          e->dispatch(K / 256, 2, 1, 256);
          e->commit_and_wait();
        }
        const uint32_t tg = N / c.nt;
        kp::Qmm8 p{};
        p.K = K;
        p.nseg = 1;
        p.ks = c.ks;
        p.N[0] = N;
        p.y_ld[0] = N;
        p.add[0] = 0;
        p.tg_begin[0] = 0;
        p.tg_begin[1] = tg;
        auto bind = [&](Encoder& e, Buffer& y, uint32_t y_row) {
          for (int i = 0; i < 4; ++i) {
            e.buf(i * 4 + 0, w);
            e.buf(i * 4 + 1, s);
            e.buf(i * 4 + 2, b);
            e.buf(i * 4 + 3, y, size_t(y_row) * N * 4);
          }
        };
        {
          auto e = g.begin();
          e->use(g.pipeline(std::string(c.pf ? "qmm8tp_b4_nt" : "qmm8t_b4_nt") + std::to_string(c.nt)));
          p.R = 8;
          for (uint32_t t = 0; t < 2; ++t) {  // per tile: its own T table, sums and 8 output rows
            bind(*e, ya, 8 * t);
            e->buf(16, tt, size_t(t) * 8 * K * 2);
            e->buf(17, sm, size_t(t) * gpr * 8 * 4);
            e->val(18, p);
            e->dispatch(tg, 1, 1, 32 * c.ks);
          }
          e->commit_and_wait();
        }
        {
          auto e = g.begin();
          e->use(g.pipeline(std::string(c.pf ? "qmm8t2dp_b4_nt" : "qmm8t2d_b4_nt") + std::to_string(c.nt)));
          p.R = R16;
          bind(*e, yb, 0);
          e->buf(16, tt, 0);
          e->buf(17, sm, 0);
          e->buf(18, tt, size_t(8) * K * 2);
          e->buf(19, sm, size_t(gpr) * 8 * 4);
          e->val(20, p);
          e->dispatch(tg, 1, 1, 32 * c.ks);
          e->commit_and_wait();
        }
        size_t bad = 0;
        double mx = 0, rms = 0;
        for (size_t i = 0; i < size_t(R16) * N; ++i) {
          const double d = double(ya.as<float>()[i]) - double(yb.as<float>()[i]);
          mx = std::max(mx, std::fabs(d));
          rms += d * d;
          if (ya.as<uint32_t>()[i] != yb.as<uint32_t>()[i]) ++bad;
        }
        double ref = 0;
        for (size_t i = 0; i < size_t(R16) * N; ++i) ref += double(ya.as<float>()[i]) * ya.as<float>()[i];
        printf("qmm8t2d%-4s N=%u K=%u (nt %u, ks %u): %zu of %zu outputs differ from the per-tile qmm8t runs "
               "(max |d| %.2e, rms %.2e of rms %.2e)%s\n", c.pf ? "pf" : "", N, K, c.nt, c.ks, bad,
               size_t(R16) * N, mx, std::sqrt(rms / (R16 * N)), std::sqrt(ref / (R16 * N)),
               bad ? "  NOT EXACT" : "");
        if (bad) ++fails;
      }
    }
  } else {
    printf("qmm8t2d: skipped, needs Apple10 and macOS 26\n");
  }
  if (qk16_only && !ran16) {
    printf("DAT_QK16_ONLY=1, but the 16-row attention check did not run\n");
    return 3;
  }
  if (pair_only && !ran_pair) {
    printf("DAT_PAIR_ONLY=1, but the pair-GEMM check did not run\n");
    return 3;
  }
  return fails ? 1 : 0;
}
