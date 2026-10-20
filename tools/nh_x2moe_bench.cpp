// Nemotron 3.5 Bolt's routed IQ2_XXS experts in a verify (nh_x2_moeb_up8 / nh_x2_moeb_dn, nemotron/nh_x_iq2.metal,
// in threadgroups of 1-8 simdgroups) on a measured layer's shape: 368 experts of width 2304 on the latent 1024, 5 token rows x
// 9 experts = 45 pairs over 29 distinct experts (34 tiles of <= 2 pairs: the harness decode's averages). Random
// weights, NC copies of the expert stacks (> 1 GB in all, so the calls read DRAM), each call another copy and another
// draw of the experts; the variants' outputs against the baseline's, bit for bit. GPU time per call (10 calls per
// command buffer, median of --reps), GB/s of the distinct experts' bytes.
//   nh_x2moe_bench [--reps n] [--small] [--same]   (--small: tiny shapes, one copy, for runs with MTL_SHADER_VALIDATION=1
//   / MTL_DEBUG_LAYER=1; --same: one copy, the weights in the cache: the compute side)
#include "metal/gpu.h"
#include "util/log.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace ni {
extern const char* const kNhKernelSource;
}
using namespace ni;

namespace {
struct MoeP {
  uint32_t I, D, k, up_q_stride, up_s_stride, dn_q_stride, dn_s_stride, h_stride;
};
struct Tiles {
  std::vector<uint32_t> sorted;
  std::vector<uint32_t> tiles;  // uint4 each
  uint32_t ntiles = 0;
};
// pairs (rows x k) -> experts; tiles of <= 2 pairs by expert, ascending
Tiles bucket(const std::vector<int>& eid) {
  Tiles t;
  std::vector<std::vector<uint32_t>> by(1024);
  for (size_t i = 0; i < eid.size(); ++i) by[size_t(eid[i])].push_back(uint32_t(i));
  for (uint32_t e = 0; e < 1024; ++e) {
    for (size_t j = 0; j < by[e].size(); j += 2) {
      const uint32_t off = uint32_t(t.sorted.size()) + uint32_t(j);
      t.tiles.insert(t.tiles.end(), {e, off, uint32_t(std::min<size_t>(2, by[e].size() - j)), 0u});
      ++t.ntiles;
    }
    t.sorted.insert(t.sorted.end(), by[e].begin(), by[e].end());
  }
  return t;
}
}  // namespace

int main(int argc, char** argv) {
  int reps = 15;
  bool small = false, same = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--reps" && i + 1 < argc) reps = std::max(1, atoi(argv[++i]));
    else if (a == "--small") small = true;
    else if (a == "--same") same = true;
  }
  Gpu& g = Gpu::get();
  g.compile_extra("Nemotron-H", kNhKernelSource);
  const uint32_t E = small ? 16 : 368, I = small ? 512 : 2304, D = small ? 256 : 1024, k = 9, R = 5, P = R * k;
  const uint32_t maxI = I;
  const size_t up_q = size_t(I) * (D / 4), up_s = size_t(I) * (D / 256) * 2;  // per expert
  const size_t dn_q = size_t(D) * (I / 4), dn_s = size_t(D) * (I / 256) * 2;
  const size_t per_copy = E * (up_q + up_s + dn_q + dn_s);
  const int NC = small || same ? 1 : std::max(2, int(1.2e9 / double(per_copy)));
  std::mt19937_64 rng(5);
  auto fill = [&](Buffer& b, bool scale) {
    if (!scale) {
      for (size_t i = 0; i < b.size / 8; ++i) b.as<uint64_t>()[i] = rng();
    } else {
      std::uniform_real_distribution<float> u(0.0005f, 0.002f);
      for (size_t i = 0; i < b.size / 2; ++i) b.as<__fp16>()[i] = __fp16(u(rng));
    }
  };
  std::vector<Buffer> uq, us, dq, ds;
  for (int c = 0; c < NC; ++c) {
    uq.push_back(g.alloc(E * up_q, "uq", false));
    us.push_back(g.alloc(E * up_s, "us", false));
    dq.push_back(g.alloc(E * dn_q, "dq", false));
    ds.push_back(g.alloc(E * dn_s, "ds", false));
    for (Buffer* b : {&uq.back(), &dq.back()}) {
      if (c == 0) fill(*b, false);
      else memcpy(b->ptr(), (b == &uq.back() ? uq[0] : dq[0]).ptr(), b->size);
    }
    for (Buffer* b : {&us.back(), &ds.back()}) {
      if (c == 0) fill(*b, true);
      else memcpy(b->ptr(), (b == &us.back() ? us[0] : ds[0]).ptr(), b->size);
    }
  }
  // the draws: per call, 29 distinct experts over 45 pairs (each row's 9 experts distinct)
  const int ndraw = 16;
  std::vector<Buffer> tiles_b, sorted_b, args_b;
  std::vector<uint32_t> ntiles;
  for (int d = 0; d < ndraw; ++d) {
    std::vector<int> pool;
    std::vector<int> all(E);
    for (uint32_t e = 0; e < E; ++e) all[e] = int(e);
    std::shuffle(all.begin(), all.end(), rng);
    pool.assign(all.begin(), all.begin() + std::min<uint32_t>(29, E));
    std::vector<int> eid;
    for (uint32_t r = 0; r < R; ++r) {
      std::vector<int> row;
      // rows share experts: the first 9 of the pool for row 0, then mostly new ones
      while (row.size() < k) {
        const int v = pool[size_t(rng() % pool.size())];
        if (std::find(row.begin(), row.end(), v) == row.end()) row.push_back(v);
      }
      eid.insert(eid.end(), row.begin(), row.end());
    }
    Tiles t = bucket(eid);
    Buffer tb = g.alloc(t.tiles.size() * 4), sb = g.alloc(t.sorted.size() * 4), ab = g.alloc(64);
    memcpy(tb.ptr(), t.tiles.data(), t.tiles.size() * 4);
    memcpy(sb.ptr(), t.sorted.data(), t.sorted.size() * 4);
    tiles_b.push_back(tb);
    sorted_b.push_back(sb);
    args_b.push_back(ab);
    ntiles.push_back(t.ntiles);
  }
  double nt = 0;
  for (uint32_t v : ntiles) nt += v;
  Buffer lat = g.alloc(size_t(R) * D * 4), h0 = g.alloc(size_t(P) * maxI * 4), h1 = g.alloc(size_t(P) * maxI * 4);
  Buffer p0 = g.alloc(size_t(P) * D * 4), p1 = g.alloc(size_t(P) * D * 4), ew = g.alloc(size_t(P) * 4);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (size_t i = 0; i < size_t(R) * D; ++i) lat.as<float>()[i] = nd(rng);
  for (uint32_t i = 0; i < P; ++i) ew.as<float>()[i] = 0.05f + 0.01f * float(i % 9);
  const MoeP mp{I, D, k, uint32_t(up_q), uint32_t(up_s), uint32_t(dn_q), uint32_t(dn_s), maxI};
  // a variant: kernel, rows per threadgroup, threads, threadgroup memory
  struct Var {
    std::string label, kern;
    uint32_t rows, thr;
    size_t tgmem;
  };

  // (variants that measured slower and were removed: 16 / 32 up rows per simdgroup on the same activation registers
  // 47.5 / 50.3 us; the down tile's h rows staged in threadgroup memory, 1-4 batches of 4 rows per simdgroup, 53-84)
  const std::vector<Var> ups = {{"up8", "nh_x2_moeb_up8", 8, 32, 0},
                                {"up8 x2sg", "nh_x2_moeb_up8", 16, 64, 0},
                                {"up8 x4sg", "nh_x2_moeb_up8", 32, 128, 0}};
  const std::vector<Var> dns = {{"dn", "nh_x2_moeb_dn", 4, 32, 0},
                                {"dn x2sg", "nh_x2_moeb_dn", 8, 64, 0},
                                {"dn x4sg", "nh_x2_moeb_dn", 16, 128, 0},
                                {"dn x8sg", "nh_x2_moeb_dn", 32, 256, 0}};
  auto set_args = [&](int d, uint32_t ux, uint32_t dx) {
    uint32_t* a = args_b[size_t(d)].as<uint32_t>();
    a[0] = ux; a[1] = ntiles[size_t(d)]; a[2] = 1; a[3] = 0;
    a[4] = dx; a[5] = ntiles[size_t(d)]; a[6] = 1; a[7] = 0;
  };
  auto run_up = [&](const Var& v, Buffer& h, int start, int calls) {
    Pipeline pl = g.pipeline(v.kern);
    auto e = g.begin();
    for (int i = 0; i < calls; ++i) {
      const int c = (start + i) % NC, d = (start + i) % ndraw;
      e->use(pl);
      e->buf(0, uq[size_t(c)]); e->buf(1, us[size_t(c)]); e->buf(2, lat); e->buf(3, h);
      e->buf(4, tiles_b[size_t(d)]); e->buf(5, sorted_b[size_t(d)]); e->val(6, mp);
      e->dispatch_indirect(args_b[size_t(d)], 0, v.thr, 1, 1);
    }
    e->commit_and_wait();
    return e->gpu_ms();
  };
  auto run_dn = [&](const Var& v, const Buffer& h, Buffer& part, int start, int calls) {
    Pipeline pl = g.pipeline(v.kern);
    auto e = g.begin();
    for (int i = 0; i < calls; ++i) {
      const int c = (start + i) % NC, d = (start + i) % ndraw;
      e->use(pl);
      if (v.tgmem) e->tgmem(0, v.tgmem);
      e->buf(0, dq[size_t(c)]); e->buf(1, ds[size_t(c)]); e->buf(2, h); e->buf(3, part);
      e->buf(4, tiles_b[size_t(d)]); e->buf(5, sorted_b[size_t(d)]); e->buf(6, ew); e->val(7, mp);
      e->dispatch_indirect(args_b[size_t(d)], 16, v.thr, 1, 1);
    }
    e->commit_and_wait();
    return e->gpu_ms();
  };
  printf("E %u, I %u, D %u: %d copies of %.0f MB, %d draws, %.1f tiles, %.0f KB of distinct experts per projection\n", E,
         I, D, NC, per_copy / 1e6, ndraw, nt / ndraw, 29.0 * (up_q + up_s) / 1e3);
  const double bytes = std::min<uint32_t>(29, E) * double(up_q + up_s);
  // checks: each variant's outputs on draw 0, copy 0, against the baseline's
  for (const Var& v : ups) {
    for (int d = 0; d < ndraw; ++d) set_args(d, (I + v.rows - 1) / v.rows, 0);
    memset((v.kern == ups[0].kern && v.rows == ups[0].rows ? h0 : h1).ptr(), 0, h0.size);
    run_up(v, v.kern == ups[0].kern && v.rows == ups[0].rows ? h0 : h1, 0, 1);
    if (&v != &ups[0]) printf("  %-12s outputs %s\n", v.label.c_str(), memcmp(h0.ptr(), h1.ptr(), h0.size) ? "DIFFER" : "bitwise identical");
  }
  for (const Var& v : dns) {
    for (int d = 0; d < ndraw; ++d) set_args(d, 0, (D + v.rows - 1) / v.rows);
    Buffer& o = &v == &dns[0] ? p0 : p1;
    memset(o.ptr(), 0, o.size);
    run_dn(v, h0, o, 0, 1);
    if (&v != &dns[0]) printf("  %-12s outputs %s\n", v.label.c_str(), memcmp(p0.ptr(), p1.ptr(), p0.size) ? "DIFFER" : "bitwise identical");
  }
  if (small) return 0;
  std::vector<std::vector<double>> tu(ups.size()), td(dns.size());
  for (int rd = 0; rd < reps + 2; ++rd) {
    for (size_t i = 0; i < ups.size(); ++i) {
      for (int d = 0; d < ndraw; ++d) set_args(d, (I + ups[i].rows - 1) / ups[i].rows, 0);
      const double ms = run_up(ups[i], h1, int(rng() % 1000), 10);
      if (rd >= 2) tu[i].push_back(ms / 10);
    }
    for (size_t i = 0; i < dns.size(); ++i) {
      for (int d = 0; d < ndraw; ++d) set_args(d, 0, (D + dns[i].rows - 1) / dns[i].rows);
      const double ms = run_dn(dns[i], h0, p1, int(rng() % 1000), 10);
      if (rd >= 2) td[i].push_back(ms / 10);
    }
  }
  auto show = [&](const std::vector<Var>& vs, std::vector<std::vector<double>>& ts) {
    for (size_t i = 0; i < vs.size(); ++i) {
      std::sort(ts[i].begin(), ts[i].end());
      const double ms = ts[i][ts[i].size() / 2];
      printf("  %-12s %7.1f us  %6.1f GB/s (min %.1f)\n", vs[i].label.c_str(), ms * 1000, bytes / ms / 1e6, ts[i][0] * 1000);
    }
  };
  show(ups, tu);
  show(dns, td);
  return 0;
}
