// A verify's routed experts on Lightning's shapes (NVFP4, 128 experts, I 1856, hidden 2688, top-6) on synthetic
// weights: nh_x4_moe_up + nh_x4_moe_dn + nh_moe2_sum2_add per (row, expert) pair, the experts in layout L0 or L2
// (_l2) and at 8 or 4 simdgroups per threadgroup, every variant's h and output compared to the bit with the first
// (L0, 8 simdgroups) and the L0 up products spot-checked against fp64. One call is one layer of one verify (its live
// rows, its pairs); calls rotate over --layers copies of the experts (~5.6 MB per expert and projection) so they
// read DRAM, --calls calls per command buffer (serial, as the engine's verify), median of --reps; the variants
// interleave.
//   nh_moe_bench [--routes file] [--layers n] [--calls n] [--reps n] [--small] [variant ...]
// Routings: synthetic by default (2-8 live rows as the harness prompt's DSpark verifies had them; a row's experts
// repeat earlier rows' with probability 0.38, which gives their ~0.67 distinct experts per pair), or --routes: records
// of uint32 live rows, uint32 R, int32 eid[48] (k = 6).
// --small: two layer copies, four calls: run it under MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 first.
#include "metal/gpu.h"
#include "model/nemotron_h.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace ni {
extern const char* const kNhKernelSource;
}
using namespace ni;

namespace {
uint64_t g_rng = 0x9E3779B97F4A7C15ull;
uint64_t rnd() {
  uint64_t z = (g_rng += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
float nrand() {
  float s = 0;
  for (int i = 0; i < 4; ++i) s += float(rnd() & 0xFFFF) / 65535.0f;
  return (s - 2.0f) * 1.7320508f;
}
double e4m3(uint8_t x) {
  const int ex = (x >> 3) & 0xF, m = x & 7;
  return ex == 0 ? std::ldexp(double(m), -9) : std::ldexp(1.0 + double(m) / 8.0, ex - 7);
}
const double kE2M1[16] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6};
struct Rec {
  uint32_t live, R;
  int32_t eid[48];
};
}  // namespace

int main(int argc, char** argv) {
  int reps = 9, calls = 23, NL = 6;
  bool small = false;
  std::string routes;
  std::vector<std::string> only;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
    else if (a == "--calls" && i + 1 < argc) calls = atoi(argv[++i]);
    else if (a == "--layers" && i + 1 < argc) NL = atoi(argv[++i]);
    else if (a == "--routes" && i + 1 < argc) routes = argv[++i];
    else if (a == "--small") small = true;
    else only.push_back(a);
  }
  if (reps < 1 || calls < 1 || NL < 1 || calls > 4096 || NL > 64)
    fatal("--reps, --calls (<= 4096) and --layers (<= 64) take positive counts");
  if (small) {
    NL = 2;
    calls = 4;
    reps = 1;
  }
  alarm(240);  // a hard wall-clock limit (SIGALRM ends the process)
  constexpr uint32_t E = 128, I = 1856, D = 2688, k = 6, R = 8;
  std::vector<Rec> recs;
  if (!routes.empty()) {
    FILE* f = fopen(routes.c_str(), "rb");
    if (!f) fatal("cannot read %s", routes.c_str());
    Rec r;
    while (fread(&r, sizeof r, 1, f) == 1) recs.push_back(r);
    fclose(f);
    if (recs.empty()) fatal("%s: no routings", routes.c_str());
    for (const Rec& q : recs) {
      if (q.R != R || q.live < 1 || q.live > R) fatal("%s: a routing of %u rows (%u live); this bench takes %u", routes.c_str(), q.R, q.live, R);
      for (uint32_t i = 0; i < q.live * k; ++i)
        if (q.eid[i] < 0 || q.eid[i] >= int(E)) fatal("%s: expert %d out of range", routes.c_str(), q.eid[i]);
    }
  } else {
    const int hist[9] = {0, 0, 230, 529, 575, 506, 368, 414, 575};  // live rows of 3197 recorded layer verifies
    for (int n = 0; n < 4096; ++n) {
      Rec r{};
      int u = int(rnd() % 3197), rows = 2;
      while (u >= hist[rows]) u -= hist[rows++];
      r.live = uint32_t(rows);
      r.R = R;
      std::vector<int> used;
      for (int row = 0; row < rows; ++row) {
        std::vector<int> mine;
        while (mine.size() < k) {
          const int e = !used.empty() && rnd() % 100 < 38 ? used[rnd() % used.size()] : int(rnd() % E);
          if (std::find(mine.begin(), mine.end(), e) == mine.end()) mine.push_back(e);
        }
        for (uint32_t j = 0; j < k; ++j) r.eid[row * k + j] = mine[j];
        used.insert(used.end(), mine.begin(), mine.end());
      }
      for (uint32_t i = uint32_t(rows) * k; i < R * k; ++i) r.eid[i] = int(rnd() % E);
      recs.push_back(r);
    }
  }
  Gpu& g = Gpu::get();
  g.compile_extra("Nemotron-H", kNhKernelSource);
  // the experts: ModelOpt bytes repacked to L0 (q0) and to L2 (q2), block scales, tensor scales
  const size_t qb = size_t(I) * D / 2, sb = size_t(I) * D / 16;
  struct Layer {
    Buffer uq0, uq2, us, dq0, dq2, ds, gu, gd;
  };
  std::vector<Layer> ly(static_cast<size_t>(NL));
  for (Layer& l : ly) {
    l.uq0 = g.alloc(qb * E, "upq0", false);
    l.uq2 = g.alloc(qb * E, "upq2", false);
    l.dq0 = g.alloc(qb * E, "dnq0", false);
    l.dq2 = g.alloc(qb * E, "dnq2", false);
    l.us = g.alloc(sb * E, "ups", false);
    l.ds = g.alloc(sb * E, "dns", false);
    l.gu = g.alloc(E * 4, "gsu");
    l.gd = g.alloc(E * 4, "gsd");
    for (uint32_t e = 0; e < E; ++e) {
      l.gu.as<float>()[e] = 0.004f * (1.0f + 0.5f * float(e % 7));
      l.gd.as<float>()[e] = 0.003f * (1.0f + 0.3f * float(e % 5));
    }
    for (Buffer* b : {&l.us, &l.ds})
      for (size_t i = 0; i < b->size; ++i) b->as<uint8_t>()[i] = uint8_t(0x28 + rnd() % 0x28);
    std::vector<uint64_t> seeds(16);
    for (auto& s : seeds) s = rnd() | 1;
    std::vector<std::thread> th;
    for (uint32_t t = 0; t < 16; ++t)
      th.emplace_back([&, t] {
        std::vector<uint8_t> raw(qb);
        uint64_t x = seeds[t];
        for (uint32_t e = t; e < E; e += 16)
          for (int dn = 0; dn < 2; ++dn) {
            for (size_t i = 0; i < qb; i += 8) {
              x ^= x << 13, x ^= x >> 7, x ^= x << 17;
              memcpy(raw.data() + i, &x, 8);
            }
            const uint32_t N = dn ? D : I, K = dn ? I : D;
            nh_repack_rows(NhFmt::FP4, K, raw.data(), N, (dn ? l.dq0 : l.uq0).as<uint8_t>() + e * qb, nullptr, false);
            nh_repack_rows(NhFmt::FP4, K, raw.data(), N, (dn ? l.dq2 : l.uq2).as<uint8_t>() + e * qb, nullptr, true);
          }
      });
    for (auto& t : th) t.join();
  }
  // per call: routing, inputs, outputs
  struct Call {
    Buffer eid, ew, live, lat, h, part, x;
    uint32_t live_rows, pairs, distinct;
  };
  std::vector<Call> cs(static_cast<size_t>(std::max(calls, 1)));
  const int ncall = int(cs.size());
  double sum_pairs = 0, sum_dist = 0;
  for (int c = 0; c < ncall; ++c) {
    Call& C = cs[size_t(c)];
    const Rec& r = recs[size_t(c) % recs.size()];
    C.eid = g.alloc(R * k * 4, "eid");
    C.ew = g.alloc(R * k * 4, "ew");
    C.live = g.alloc(16, "live");
    C.lat = g.alloc(size_t(R) * D * 4, "lat");
    C.h = g.alloc(size_t(R) * k * I * 4, "h");
    C.part = g.alloc(size_t(R) * k * D * 4, "part");
    C.x = g.alloc(size_t(R) * D * 4, "x");
    memcpy(C.eid.ptr(), r.eid, R * k * 4);
    for (uint32_t i = 0; i < R * k; ++i) C.ew.as<float>()[i] = 0.05f + float(rnd() % 100) / 400.0f;
    C.live.as<uint32_t>()[0] = r.live;
    for (uint32_t i = 0; i < R * D; ++i) C.lat.as<float>()[i] = nrand() * (i % 97 == 0 ? 8.0f : 1.0f);
    C.live_rows = std::min(r.live, R);
    C.pairs = C.live_rows * k;
    std::vector<int> u(r.eid, r.eid + C.pairs);
    std::sort(u.begin(), u.end());
    C.distinct = uint32_t(std::unique(u.begin(), u.end()) - u.begin());
    sum_pairs += C.pairs;
    sum_dist += C.distinct;
  }
  printf("%d calls: %.2f pairs, %.2f distinct experts per call (%s); %d layer copies (%.1f GB per layout)\n", ncall,
         sum_pairs / ncall, sum_dist / ncall, routes.empty() ? "synthetic routings" : routes.c_str(), NL,
         double(NL) * E * (qb + sb) * 2 / 1e9);
  const struct {
    uint32_t I, D, k, uqs, uss, dqs, dss, hs;
  } mp{I, D, k, uint32_t(qb), uint32_t(sb), uint32_t(qb), uint32_t(sb), I};
  struct Var {
    std::string name;
    bool l2;
    uint32_t thr;
  };
  const std::vector<Var> all = {{"l0_8sg", false, 256}, {"l0_4sg", false, 128}, {"l2_8sg", true, 256}, {"l2_4sg", true, 128}};
  std::vector<const Var*> vs;
  for (const Var& v : all)
    if (only.empty() || std::find(only.begin(), only.end(), v.name) != only.end() || &v == &all[0]) vs.push_back(&v);
  // up only (stage 0) or up + down + sum (stage 1)
  auto enc = [&](Encoder& e, int c, const Var& v, int stage) {
    Call& C = cs[size_t(c)];
    const Layer& l = ly[size_t(c % NL)];
    const uint32_t rows_tg = v.thr / 8;
    e.use(NemotronH::pipeline(v.l2 ? "nh_x4_moe_up_l2" : "nh_x4_moe_up"));
    e.buf(0, v.l2 ? l.uq2 : l.uq0);
    e.buf(1, l.us);
    e.buf(2, C.lat);
    e.buf(3, C.h);
    e.buf(4, C.eid);
    e.bytes(5, &mp, sizeof mp);
    e.buf(7, C.live);
    e.buf(9, l.gu);
    e.dispatch((I + rows_tg - 1) / rows_tg, R * k, 1, v.thr);
    if (stage == 0) return;
    e.use(NemotronH::pipeline(v.l2 ? "nh_x4_moe_dn_l2" : "nh_x4_moe_dn"));
    e.buf(0, v.l2 ? l.dq2 : l.dq0);
    e.buf(1, l.ds);
    e.buf(2, C.h);
    e.buf(3, C.part);
    e.buf(4, C.eid);
    e.buf(5, C.ew);
    e.bytes(6, &mp, sizeof mp);
    e.buf(8, C.live);
    e.buf(9, l.gd);
    e.dispatch((D + rows_tg - 1) / rows_tg, R * k, 1, v.thr);
    e.use(NemotronH::pipeline("nh_moe2_sum2_add"));
    e.buf(0, C.part);
    e.buf(1, C.x);
    e.bytes(2, &mp, sizeof mp);
    e.buf(3, C.live);
    e.dispatch((D + 63) / 64, R, 1, 64);
  };
  auto run_all = [&](const Var& v) {
    for (Call& C : cs) {
      memset(C.x.ptr(), 0, C.x.size);
      memset(C.h.ptr(), 0xff, C.h.size);
    }
    auto e = g.begin();
    for (int c = 0; c < ncall; ++c) enc(*e, c, v, 1);
    e->commit_and_wait();
  };
  // the reference (the first variant), spot-checked against fp64 of the dequantized L0 weights
  std::vector<std::vector<float>> ref_h(cs.size()), ref_x(cs.size());
  run_all(*vs[0]);
  double worst = 0;
  for (int c = 0; c < ncall; ++c) {
    const Call& C = cs[size_t(c)];
    ref_h[size_t(c)].assign(C.h.as<float>(), C.h.as<float>() + size_t(C.pairs) * I);
    ref_x[size_t(c)].assign(C.x.as<float>(), C.x.as<float>() + size_t(C.live_rows) * D);
    const Layer& l = ly[size_t(c % NL)];
    for (uint32_t pr = 0; pr < C.pairs && c < 4; pr += 5)
      for (uint32_t n = 0; n < I; n += 97) {
        const int e = C.eid.as<int>()[pr];
        const uint8_t* qr = l.uq0.as<uint8_t>() + size_t(e) * qb + size_t(n) * (D / 2);
        const uint8_t* sr = l.us.as<uint8_t>() + size_t(e) * sb + size_t(n) * (D / 16);
        const float* xr = C.lat.as<float>() + (pr / k) * D;
        double a = 0, mag = 0;
        for (uint32_t kk = 0; kk < D; ++kk) {  // L0: byte i of a block = elements i (low nibble), i + 8 (high)
          const uint8_t byte = qr[(kk / 16) * 8 + (kk % 8)];
          const double w = kE2M1[(kk % 16) < 8 ? (byte & 15) : (byte >> 4)] * e4m3(sr[kk / 16]) * l.gu.as<float>()[e];
          a += w * xr[kk];
          mag += std::fabs(w * xr[kk]);
        }
        const double ref = a > 0 ? a * a : 0;
        worst = std::max(worst, std::fabs(ref_h[size_t(c)][size_t(pr) * I + n] - ref) / std::max(mag * mag, 1e-30));
      }
  }
  printf("%s up vs fp64: max |err| / (sum |w x|)^2 %.2e\n", vs[0]->name.c_str(), worst);
  bool differs = false;
  for (size_t vi = 1; vi < vs.size(); ++vi) {
    run_all(*vs[vi]);
    size_t bh = 0, bx = 0;
    for (int c = 0; c < ncall; ++c) {
      const Call& C = cs[size_t(c)];
      bh += memcmp(C.h.ptr(), ref_h[size_t(c)].data(), ref_h[size_t(c)].size() * 4) != 0;
      bx += memcmp(C.x.ptr(), ref_x[size_t(c)].data(), ref_x[size_t(c)].size() * 4) != 0;
    }
    printf("%-7s vs %s: h %s, out %s (%d calls)\n", vs[vi]->name.c_str(), vs[0]->name.c_str(),
           bh ? "DIFFERS" : "bitwise the same", bx ? "DIFFERS" : "bitwise the same", ncall);
    differs |= bh || bx;
  }
  const double eb = double(qb + sb);  // one expert's projection
  for (int st = 0; st < 2; ++st) {
    std::vector<std::vector<double>> t(vs.size());
    for (int rd = 0; rd < reps + 3; ++rd)
      for (size_t vi = 0; vi < vs.size(); ++vi) {
        auto e = g.begin();
        for (int c = 0; c < ncall; ++c) enc(*e, c, *vs[vi], st);
        e->commit_and_wait();
        if (rd >= 3) t[vi].push_back(e->gpu_ms() * 1000.0 / ncall);
      }
    for (size_t vi = 0; vi < vs.size(); ++vi) {
      std::sort(t[vi].begin(), t[vi].end());
      const double us = t[vi][t[vi].size() / 2];
      printf("%-7s %-15s %7.1f us per layer (min %6.1f)  %4.0f GB/s of distinct experts\n", vs[vi]->name.c_str(),
             st ? "up + down + sum" : "up", us, t[vi][0], sum_dist / ncall * eb * (st ? 2 : 1) / us / 1e3);
    }
  }
  return differs ? 1 : 0;  // a variant whose outputs differ fails the run
}
