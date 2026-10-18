// IQ2_XXS kernels for Nemotron 3.5 Bolt (src/kernels/nemotron/nh_x_iq2.metal) against the current ones, on synthetic
// weights of the real shapes: GPU time per call, effective bandwidth, and a CPU reference check (fp64 dequant) of
// every kernel.
//   nh_x2_bench [options] [case ...]       cases (substrings): "mv up", "mv dn" (shared expert, one row),
//       "moe1 I=<w>" (routed experts, one row), "moe8 I=<w>", "moe32 I=<w>" (routed, 8 / 32 rows), "sh8", "sh32"
//       (shared expert, 8 / 32 rows); I = 1536 2048 2304 2560; default: every case
//   --validate      small, odd shapes; each kernel once, one dispatch (or one routed layer) per command buffer: run it
//                   under MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 to catch out-of-bounds access
//   --hot           one copy of the weights (default: copies / expert sets in rotation, so calls read DRAM)
//   --warm <ms>     GPU warm-up before each timing (default 150)
//   --stats k1,k2 <dir>   GPR count, code size, stack of kernels from their binary archives (no GPU work)
//   --src <file>    append a kernel source at run time (if it defines nh_x2_ kernels, the embedded nh_x_iq2.metal
//                   section is dropped); extra variants: --mv name:rows_tg:threads[,...] (nh_mv1_iq2_c signature),
//                   --sh name:kind:rows_tg:threads[,...] (kind v8 / gemm / x2), --pscheme up:rows:thr:dn:rows:thr
//                   (pair kernels), --bscheme bucket:up:rows:dn:rows:BM (bucketed kernels)
// Timing: kernels round-robin, one command buffer of up to 20 calls (<= ~40 ms) each per round, median per call.
#include "metal/gpu.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace ni {
extern const char* const kNhKernelSource;
}
using namespace ni;

namespace {

std::mt19937 rng(42);
std::vector<uint64_t> g_grid;
bool g_hot = false, g_validate = false;
double g_warm_ms = 150.0;
Buffer g_gtab;  // the grid as halves (nh_iq2_fill_tables2): buffer 8 of the v8 / GEMM kernels

std::vector<uint64_t> grid_from_source(const std::string& src) {
  std::vector<uint64_t> g;
  size_t p = src.find("nh_iq2xxs_grid[256] = {");
  p = src.find('{', p) + 1;
  while (g.size() < 256) {
    p = src.find("0x", p);
    g.push_back(std::stoull(src.substr(p + 2, 16), nullptr, 16));
    p += 18;
  }
  return g;
}

// E stacked IQ2_XXS matrices of N x K in the repacked layout (codes [E][N][K/4], half d [E][N][K/256]); K % 256 == 0
struct Iq2 {
  uint32_t N = 0, K = 0, E = 0;
  Buffer q, s;
  size_t qrow() const { return K / 4; }
  size_t srow() const { return K / 256 * 2; }
  size_t qexp() const { return qrow() * N; }
  size_t sexp() const { return srow() * N; }
  double bytes_per_exp() const { return double(qexp() + sexp()); }
  // fp64 dequant of row n of expert e
  void row(uint32_t e, uint32_t n, std::vector<double>& w) const {
    w.resize(K);
    const uint8_t* qr = q.as<uint8_t>() + e * qexp() + size_t(n) * qrow();
    const __fp16* sr = (const __fp16*)(s.as<uint8_t>() + e * sexp() + size_t(n) * srow());
    for (uint32_t j = 0; j < K / 32; ++j) {
      uint32_t a0, a1;
      memcpy(&a0, qr + j * 8, 4);
      memcpy(&a1, qr + j * 8 + 4, 4);
      const double db = double(float(sr[j / 8])) * (0.5 + double(a1 >> 28)) * 0.25;
      for (int l = 0; l < 4; ++l) {
        const uint64_t gp = g_grid[(a0 >> (8 * l)) & 0xFF];
        const uint32_t s7 = (a1 >> (7 * l)) & 127;
        const uint32_t sg = s7 | ((__builtin_popcount(s7) & 1) << 7);
        for (int i = 0; i < 8; ++i) {
          const double m = double((gp >> (8 * i)) & 0xFF) * db;
          w[j * 32 + l * 8 + i] = (sg >> i) & 1 ? -m : m;
        }
      }
    }
  }
  void release() {
    Gpu::get().release(q);
    Gpu::get().release(s);
  }
};

Iq2 make_iq2(uint32_t N, uint32_t K, uint32_t E) {
  if (K % 256 || N == 0) fatal("IQ2_XXS matrix %u x %u: K must be a multiple of 256", N, K);
  Gpu& g = Gpu::get();
  Iq2 m;
  m.N = N;
  m.K = K;
  m.E = E;
  m.q = g.alloc(m.qexp() * E, "q", false);
  m.s = g.alloc(m.sexp() * E, "s", false);
  // random codes: grid indices, sign fields and sub-block scales uniform; d in [0.0005, 0.0015]
  uint32_t* q = m.q.as<uint32_t>();
  const size_t nq = m.qexp() * E / 4;
  std::uniform_int_distribution<uint32_t> u32;
  for (size_t i = 0; i < nq; ++i) q[i] = u32(rng);
  __fp16* s = m.s.as<__fp16>();
  std::uniform_real_distribution<float> ud(0.0005f, 0.0015f);
  for (size_t i = 0; i < m.sexp() * E / 2; ++i) s[i] = __fp16(ud(rng));
  return m;
}

using Enc = std::function<void(Encoder&, int)>;

double g_t0 = Encoder::host_ms();
bool g_progress = false;  // --progress (default in --validate): timestamped steps on stderr
void progress(const char* what, const std::string& name) {
  if (g_progress) fprintf(stderr, "[%8.1f s] %s %s\n", (Encoder::host_ms() - g_t0) / 1000.0, what, name.c_str());
}
Pipeline pipe(const std::string& name, bool required = true) {
  progress("pipeline", name);
  Pipeline p = Gpu::get().pipeline(name, required);
  progress("  ok", name);
  return p;
}

void run_once(const Enc& enc, int call = 0) {
  progress("run", "");
  auto e = Gpu::get().begin();
  enc(*e, call);
  e->commit_and_wait();
  progress("  done", "");
}

// Several kernels timed round-robin after a warm-up, so clock or load drift hits all alike: median ms per call of
// each. Command buffers hold up to 20 calls and about 40 ms of work (sized from a probe call).
std::vector<double> time_rr(const std::vector<Enc>& encs, int rounds = 15) {
  Gpu& g = Gpu::get();
  std::vector<double> med(encs.size(), 0.0);
  if (g_validate || encs.empty()) return med;
  std::vector<int> calls(encs.size());
  int call = 0;
  for (size_t v = 0; v < encs.size(); ++v) {
    auto e = g.begin();
    encs[v](*e, call++);
    e->commit_and_wait();
    calls[v] = std::max(1, std::min(20, int(40.0 / std::max(e->gpu_ms(), 1e-3))));
  }
  double warm = 0;
  for (int w = 0; w < 2000 && warm < g_warm_ms; ++w) {
    const size_t v = size_t(w) % encs.size();
    auto e = g.begin();
    for (int i = 0; i < calls[v]; ++i) encs[v](*e, call++);
    e->commit_and_wait();
    warm += e->gpu_ms();
  }
  std::vector<std::vector<double>> ts(encs.size());
  for (int r = 0; r < rounds; ++r)
    for (size_t v = 0; v < encs.size(); ++v) {
      auto e = g.begin();
      for (int i = 0; i < calls[v]; ++i) encs[v](*e, call++);
      e->commit_and_wait();
      ts[v].push_back(e->gpu_ms() / calls[v]);
    }
  for (size_t v = 0; v < encs.size(); ++v) {
    std::sort(ts[v].begin(), ts[v].end());
    med[v] = ts[v][ts[v].size() / 2];
  }
  return med;
}

double rel(double err, double ref) { return err / std::max(ref, 1e-30); }

// a result line: name, check, then the timing fields once measured
struct Line {
  std::string head;
  double err;
};
void print_lines(const std::vector<Line>& ls, const std::vector<double>& ms, const std::function<std::string(size_t, double)>& f) {
  for (size_t i = 0; i < ls.size(); ++i) {
    // nh_x2 kernels must match fp64 at fp32 rounding (the GEMM baselines round weights to fp16: ~2e-4)
    const bool bad = ls[i].head.find("nh_x2") != std::string::npos && ls[i].err >= 1e-5;
    if (g_validate) printf("%s  rel.err %.2e%s\n", ls[i].head.c_str(), ls[i].err, bad ? "  FAILED" : "");
    else printf("%s %s  rel.err %.2e%s\n", ls[i].head.c_str(), f(i, ms[i]).c_str(), ls[i].err, bad ? "  FAILED" : "");
  }
}

struct Var {
  std::string name;
  uint32_t rows_tg = 16, threads = 128;
};
std::vector<Var> parse_vars(const std::string& s) {
  std::vector<Var> v;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    Var x;
    const size_t a = item.find(':'), b = item.find(':', a + 1);
    x.name = item.substr(0, a);
    if (a != std::string::npos) x.rows_tg = uint32_t(std::stoul(item.substr(a + 1, b - a - 1)));
    if (b != std::string::npos) x.threads = uint32_t(std::stoul(item.substr(b + 1)));
    v.push_back(x);
  }
  return v;
}
int epi_of(const std::string& n) {
  return n.find("relu2") != std::string::npos ? 2 : n.find("_add") != std::string::npos ? 1 : 0;
}

// ---- one-row matvec: y = epi(W x), the nh_mv1_iq2_c signature ----
struct MvP {
  uint32_t N, K, x_stride, y_stride;
  float gs;
  uint32_t R;
};
void bench_mv(const std::string& label, uint32_t N, uint32_t K, const std::vector<Var>& vars) {
  Gpu& g = Gpu::get();
  const uint32_t C = g_hot || g_validate ? 1 : 6;  // copies in rotation
  Iq2 m = make_iq2(N, K, C);
  Buffer x = g.alloc(size_t(K) * 4, "x"), y = g.alloc(size_t(N) * 4, "y");
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (size_t i = 0; i < K; ++i) x.as<float>()[i] = nd(rng);
  std::vector<uint32_t> rows;
  for (uint32_t n = 0; n < N; n += std::max<uint32_t>(1, N / 97)) rows.push_back(n);
  rows.push_back(N - 1);
  std::vector<double> ref(rows.size()), w;
  for (size_t i = 0; i < rows.size(); ++i) {
    m.row(0, rows[i], w);
    double a = 0;
    for (uint32_t k = 0; k < K; ++k) a += w[k] * x.as<float>()[k];
    ref[i] = a;
  }
  std::vector<Enc> encs;
  std::vector<Line> lines;
  for (const Var& v : vars) {
    Pipeline pl = pipe(v.name, false);
    if (!pl.pso) continue;
    const int epi = epi_of(v.name);
    const MvP p{N, K, K, N, 1.0f, 1};
    Enc enc = [&m, &x, &y, pl, p, v, C](Encoder& e, int call) {
      const uint32_t c = uint32_t(call) % C;
      e.use(pl);
      e.buf(0, m.q, c * m.qexp());
      e.buf(1, m.s, c * m.sexp());
      e.buf(2, x);
      e.buf(3, y);
      e.bytes(4, &p, sizeof p);
      e.dispatch((p.N + v.rows_tg - 1) / v.rows_tg, 1, 1, v.threads);
    };
    for (uint32_t n = 0; n < N; ++n) y.as<float>()[n] = 0.25f;  // the add epilogue adds to it
    run_once(enc);
    double maxerr = 0, maxref = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
      double r = ref[i];
      if (epi == 1) r += 0.25;
      if (epi == 2) r = r > 0 ? r * r : 0.0;
      maxerr = std::max(maxerr, std::fabs(r - y.as<float>()[rows[i]]));
      maxref = std::max(maxref, std::fabs(r));
    }
    char buf[160];
    snprintf(buf, sizeof buf, "%-28s %-22s", label.c_str(), v.name.c_str());
    lines.push_back({buf, rel(maxerr, maxref)});
    encs.push_back(enc);
  }
  const std::vector<double> ms = time_rr(encs);
  const double bytes = m.bytes_per_exp();
  print_lines(lines, ms, [&](size_t, double t) {
    char b[64];
    snprintf(b, sizeof b, "%8.4f ms %6.1f GB/s", t, bytes / t / 1e6);
    return std::string(b);
  });
  g.release(x);
  g.release(y);
  m.release();
}

// ---- routed experts for R token rows. Each row takes 4 of its k experts from 12 popular ones and the rest at random
// (nh_bench's mix: ~50 distinct experts for 8 rows); one row takes k at random. A scheme encodes the whole routed
// layer (up, down, the row sums into out) for an expert set. GB/s counts the bytes of the distinct experts ----
struct MoeP {
  uint32_t I, D, k, up_q_stride, up_s_stride, dn_q_stride, dn_s_stride, h_stride;
};
struct MoeCtx {
  uint32_t I, D, k, R, E, NS, live;
  Iq2 up, dn;
  Buffer lat, h, part, out, eid, ew, livebuf;
  MoeP mp;
};
struct MoeScheme {
  std::string name;
  std::function<void(Encoder&, int set)> enc;   // the whole routed layer (checked, timed)
  std::function<void(Encoder&, int set)> up, dn;  // its up / down projection alone (timed; pair schemes)
};
std::vector<std::function<MoeScheme(MoeCtx&)>> g_extra_schemes;

std::vector<int> moe_set(uint32_t R, uint32_t k, uint32_t E, std::mt19937& er) {
  std::vector<int> v;
  // popular experts only with several rows; the rest must hold the row's other distinct experts
  const uint32_t pop = R > 1 ? std::min(12u, E / 2) : 0u;
  if (E - pop < k - (R > 1 ? std::min(4u, k) : 0u) || (R > 1 && pop < std::min(4u, k)))
    fatal("moe_set: %u experts cannot give %u distinct per row", E, k);
  for (uint32_t r = 0; r < R; ++r) {
    std::vector<int> got;
    while (R > 1 && got.size() < std::min(4u, k)) {
      const int x = int(er() % pop);
      if (std::find(got.begin(), got.end(), x) == got.end()) got.push_back(x);
    }
    while (got.size() < k) {
      const int x = int(pop + er() % (E - pop));
      if (std::find(got.begin(), got.end(), x) == got.end()) got.push_back(x);
    }
    v.insert(v.end(), got.begin(), got.end());
  }
  return v;
}

// the pair scheme: up kernel (nh_moe2_up_iq2 signature) | down kernel (nh_moe2_dn_iq2 signature) | nh_moe2_sum
MoeScheme pair_scheme(MoeCtx& c, const std::string& up, uint32_t up_rows, uint32_t up_thr, const std::string& dn,
                      uint32_t dn_rows, uint32_t dn_thr) {
  Pipeline pu = pipe(up), pd = pipe(dn), ps = pipe("nh_moe2_sum");
  MoeCtx* cp = &c;
  auto eu = [=](Encoder& e, int set) {
    const MoeCtx& c = *cp;
    const uint32_t P = c.R * c.k;
    e.use(pu);
    e.buf(0, c.up.q); e.buf(1, c.up.s); e.buf(2, c.lat); e.buf(3, c.h);
    e.buf(4, c.eid, size_t(set) * P * 4); e.bytes(5, &c.mp, sizeof c.mp); e.buf(7, c.livebuf);
    e.dispatch((c.I + up_rows - 1) / up_rows, P, 1, up_thr);
  };
  auto ed = [=](Encoder& e, int set) {
    const MoeCtx& c = *cp;
    const uint32_t P = c.R * c.k;
    e.use(pd);
    e.buf(0, c.dn.q); e.buf(1, c.dn.s); e.buf(2, c.h); e.buf(3, c.part);
    e.buf(4, c.eid, size_t(set) * P * 4); e.buf(5, c.ew); e.bytes(6, &c.mp, sizeof c.mp); e.buf(8, c.livebuf);
    e.dispatch((c.D + dn_rows - 1) / dn_rows, P, 1, dn_thr);
  };
  auto all = [=](Encoder& e, int set) {
    const MoeCtx& c = *cp;
    eu(e, set);
    ed(e, set);
    e.use(ps);
    e.buf(0, c.part); e.buf(1, c.out); e.bytes(2, &c.mp, sizeof c.mp); e.buf(3, c.livebuf);
    e.dispatch(c.R, 1, 1, 256);
  };
  return {up + " | " + dn, all, eu, ed};
}

// the bucketed scheme: bucket (pairs by expert into tiles of <= BM, indirect dispatch arguments) | up over the tiles |
// down over the tiles | nh_moe2_sum. Kernel signatures as nh_x2_bucket / nh_x2_moeb_up / nh_x2_moeb_dn.
std::vector<Buffer> g_keep;  // scheme scratch buffers
MoeScheme bucket_scheme(MoeCtx& c, const std::string& bucket, const std::string& up, uint32_t up_rows, const std::string& dn,
                        uint32_t dn_rows, uint32_t BM) {
  Gpu& g = Gpu::get();
  Pipeline pb = pipe(bucket), pu = pipe(up), pd = pipe(dn), ps = pipe("nh_moe2_sum");
  const uint32_t P = c.R * c.k;
  Buffer sorted = g.alloc(size_t(P) * 4, "sorted"), tiles = g.alloc(size_t(P) * 16, "tiles"), args = g.alloc(64, "args"),
         rank = g.alloc(size_t(P) * 4, "rank");
  g_keep.insert(g_keep.end(), {sorted, tiles, args, rank});
  MoeCtx* cp = &c;
  char nm[200];
  snprintf(nm, sizeof nm, "%s %s | %s", bucket.c_str(), up.c_str(), dn.c_str());
  return {nm, [=](Encoder& e, int set) {
            const MoeCtx& c = *cp;
            struct {
              uint32_t P, E, BM, k, up_x, dn_x;
            } bp{P, c.E, BM, c.k, (c.I + up_rows - 1) / up_rows, (c.D + dn_rows - 1) / dn_rows};
            e.use(pb);
            e.buf(0, c.eid, size_t(set) * P * 4); e.buf(1, sorted); e.buf(2, tiles); e.buf(3, args); e.buf(4, rank);
            e.bytes(5, &bp, sizeof bp); e.buf(6, c.livebuf);
            e.dispatch(1, 1, 1, 1024);
            e.use(pu);
            e.buf(0, c.up.q); e.buf(1, c.up.s); e.buf(2, c.lat); e.buf(3, c.h); e.buf(4, tiles); e.buf(5, sorted);
            e.bytes(6, &c.mp, sizeof c.mp);
            e.dispatch_indirect(args, 0, 32);
            e.use(pd);
            e.buf(0, c.dn.q); e.buf(1, c.dn.s); e.buf(2, c.h); e.buf(3, c.part); e.buf(4, tiles); e.buf(5, sorted);
            e.buf(6, c.ew); e.bytes(7, &c.mp, sizeof c.mp);
            e.dispatch_indirect(args, 16, 32);
            e.use(ps);
            e.buf(0, c.part); e.buf(1, c.out); e.bytes(2, &c.mp, sizeof c.mp); e.buf(3, c.livebuf);
            e.dispatch(c.R, 1, 1, 256);
          }};
}

void bench_moe(uint32_t I, uint32_t D, uint32_t R, uint32_t E, uint32_t live, const std::vector<std::function<MoeScheme(MoeCtx&)>>& mk) {
  Gpu& g = Gpu::get();
  MoeCtx c;
  c.I = I; c.D = D; c.k = 9; c.R = R; c.E = E; c.NS = g_hot || g_validate ? 1 : 6; c.live = live;
  c.up = make_iq2(I, D, E);
  c.dn = make_iq2(D, I, E);
  const uint32_t P = R * c.k;
  c.lat = g.alloc(size_t(R) * D * 4, "lat");
  c.h = g.alloc(size_t(P) * I * 4, "h");
  c.part = g.alloc(size_t(P) * D * 4, "part");
  c.out = g.alloc(size_t(R) * D * 4, "out");
  c.eid = g.alloc(size_t(c.NS) * P * 4, "eid");
  c.ew = g.alloc(size_t(P) * 4, "ew");
  c.livebuf = g.alloc(16, "live");
  c.livebuf.as<uint32_t>()[0] = live;
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (size_t i = 0; i < size_t(R) * D; ++i) c.lat.as<float>()[i] = nd(rng) * 0.5f;
  for (uint32_t i = 0; i < P; ++i) c.ew.as<float>()[i] = 0.1f + 0.01f * float(i % c.k);
  std::mt19937 er(7);
  double distinct = 0;
  std::vector<int> set0;
  for (uint32_t t = 0; t < c.NS; ++t) {
    std::vector<int> s = moe_set(R, c.k, E, er);
    memcpy(c.eid.as<int>() + size_t(t) * P, s.data(), size_t(P) * 4);
    if (t == 0) set0 = s;
    std::sort(s.begin(), s.end());
    distinct += double(std::unique(s.begin(), s.end()) - s.begin());
  }
  distinct /= c.NS;
  c.mp = MoeP{I, D, c.k, uint32_t(c.up.qexp()), uint32_t(c.up.sexp()), uint32_t(c.dn.qexp()), uint32_t(c.dn.sexp()), I};
  // reference rows of set 0: the first and the last live row (fp64 through relu^2)
  const uint32_t nlive = std::min(R, live);
  std::vector<uint32_t> rows_chk = {0u};
  if (nlive > 1) rows_chk.push_back(nlive - 1);
  std::vector<std::vector<double>> ref;
  std::vector<double> w, hh(I);
  for (uint32_t r : rows_chk) {
    std::vector<double> o(D, 0.0);
    for (uint32_t j = 0; j < c.k; ++j) {
      const uint32_t e = uint32_t(set0[r * c.k + j]);
      for (uint32_t n = 0; n < I; ++n) {
        c.up.row(e, n, w);
        double a = 0;
        for (uint32_t kk = 0; kk < D; ++kk) a += w[kk] * c.lat.as<float>()[size_t(r) * D + kk];
        a = std::max(a, 0.0);
        hh[n] = a * a;
      }
      for (uint32_t n = 0; n < D; ++n) {
        c.dn.row(e, n, w);
        double a = 0;
        for (uint32_t kk = 0; kk < I; ++kk) a += w[kk] * hh[kk];
        o[n] += a * c.ew.as<float>()[r * c.k + j];
      }
    }
    ref.push_back(o);
  }
  char label[64];
  snprintf(label, sizeof label, "moe%u I=%u", R, I);
  std::vector<Enc> encs;
  std::vector<Line> lines;
  std::vector<int> kinds;  // 0 layer, 1 up, 2 down
  for (auto& f : mk) {
    MoeScheme sc = f(c);
    memset(c.out.ptr(), 0, c.out.size);
    run_once([&](Encoder& e, int) { sc.enc(e, 0); });
    double maxerr = 0, maxref = 0;
    for (size_t i = 0; i < rows_chk.size(); ++i)
      for (uint32_t n = 0; n < D; ++n) {
        maxerr = std::max(maxerr, std::fabs(ref[i][n] - c.out.as<float>()[size_t(rows_chk[i]) * D + n]));
        maxref = std::max(maxref, std::fabs(ref[i][n]));
      }
    char buf[200];
    snprintf(buf, sizeof buf, "%-12s %-44s", label, sc.name.c_str());
    lines.push_back({buf, rel(maxerr, maxref)});
    auto fe = sc.enc;
    const int NS = int(c.NS);
    encs.push_back([fe, NS](Encoder& e, int call) { fe(e, call % NS); });
    kinds.push_back(0);
    if (sc.up && !g_validate) {  // the projections alone (after the layer, so h holds valid inputs)
      auto fu = sc.up, fd = sc.dn;
      encs.push_back([fu, NS](Encoder& e, int call) { fu(e, call % NS); });
      encs.push_back([fd, NS](Encoder& e, int call) { fd(e, call % NS); });
      kinds.push_back(1);
      kinds.push_back(2);
      lines.push_back({std::string(label) + std::string(13 - std::min<size_t>(12, strlen(label)), ' ') + "  [up]   " + sc.name, 0.0});
      lines.push_back({std::string(label) + std::string(13 - std::min<size_t>(12, strlen(label)), ' ') + "  [down] " + sc.name, 0.0});
      lines[lines.size() - 2].head.resize(std::max<size_t>(lines[lines.size() - 2].head.size(), 57), ' ');
      lines.back().head.resize(std::max<size_t>(lines.back().head.size(), 57), ' ');
      lines[lines.size() - 2].err = lines[lines.size() - 3].err;
      lines.back().err = lines[lines.size() - 3].err;
    }
  }
  const std::vector<double> ms = time_rr(encs, 11);
  if (!g_validate) printf("%-12s %.1f distinct experts of %u pairs (%u rows x %u)\n", label, distinct, P, R, c.k);
  const double bd = distinct * (c.up.bytes_per_exp() + c.dn.bytes_per_exp()), bp = double(P) * (c.up.bytes_per_exp() + c.dn.bytes_per_exp());
  print_lines(lines, ms, [&](size_t i, double t) {
    const double f = kinds[i] == 0 ? 1.0 : kinds[i] == 1 ? c.up.bytes_per_exp() / (c.up.bytes_per_exp() + c.dn.bytes_per_exp())
                                                         : c.dn.bytes_per_exp() / (c.up.bytes_per_exp() + c.dn.bytes_per_exp());
    char b[96];
    snprintf(b, sizeof b, "%8.4f ms %6.1f GB/s distinct %6.1f of pairs", t, f * bd / t / 1e6, f * bp / t / 1e6);
    return std::string(b);
  });
  for (Buffer* b : {&c.lat, &c.h, &c.part, &c.out, &c.eid, &c.ew, &c.livebuf}) g.release(*b);
  c.up.release();
  c.dn.release();
  for (Buffer& b : g_keep) g.release(b);
  g_keep.clear();
}

// ---- shared expert for M token rows: Y[t][n] = epi(W[n, :] . X[t, :]), X half [M][K]. kinds: "v8" (nh_v8_iq2_s<NB>:
// rows_tg = 8 NB, 128 threads), "gemm" (nh_gemm_iq2_m<BM>: 64 columns x BM rows per threadgroup), "x2" (NhV8P
// signature, rows_tg / threads as given). Checked with epi 0 (fp32 store), timed with the engine's epilogue ----
struct SVar {
  std::string name, kind;
  uint32_t rows_tg = 32, threads = 128;
};
std::vector<SVar> parse_svars(const std::string& s) {
  std::vector<SVar> v;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    std::vector<std::string> f;
    std::stringstream is(item);
    std::string t;
    while (std::getline(is, t, ':')) f.push_back(t);
    SVar x;
    x.name = f[0];
    x.kind = f.size() > 1 ? f[1] : "x2";
    if (f.size() > 2) x.rows_tg = uint32_t(std::stoul(f[2]));
    if (f.size() > 3) x.threads = uint32_t(std::stoul(f[3]));
    v.push_back(x);
  }
  return v;
}
void bench_shared(const std::string& label, uint32_t N, uint32_t K, uint32_t M, uint32_t epi, const std::vector<SVar>& vars) {
  Gpu& g = Gpu::get();
  const uint32_t C = g_hot || g_validate ? 1 : 6;
  Iq2 m = make_iq2(N, K, C);
  Buffer x = g.alloc(size_t(M) * K * 2, "xh"), y = g.alloc(size_t(M) * N * 4, "y"), dummy = g.alloc(64, "dummy");
  Buffer xp = g.alloc(size_t(K) * 16, "xp");  // kind x2p: X permuted for the B fragments (8 token rows)
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (size_t i = 0; i < size_t(M) * K; ++i) x.as<__fp16>()[i] = __fp16(nd(rng));
  std::vector<uint32_t> rows;
  for (uint32_t n = 0; n < N; n += std::max<uint32_t>(1, N / 61)) rows.push_back(n);
  rows.push_back(N - 1);
  std::vector<double> ref(rows.size() * M), w;
  for (size_t i = 0; i < rows.size(); ++i) {
    m.row(0, rows[i], w);
    for (uint32_t t = 0; t < M; ++t) {
      double a = 0;
      for (uint32_t k = 0; k < K; ++k) a += w[k] * double(float(x.as<__fp16>()[size_t(t) * K + k]));
      ref[i * M + t] = a;
    }
  }
  std::vector<Enc> encs;
  std::vector<Line> lines;
  for (const SVar& v : vars) {
    Pipeline pl = pipe(v.name, false);
    if (!pl.pso) continue;
    Pipeline perm;
    if (v.kind == "x2p") perm = pipe(v.name.find("nh_x2") == 0 ? "nh_x2_xperm" : "zz_vrows_xperm");
    auto mk = [&, pl, v, perm](uint32_t ep) -> Enc {
      return [&, pl, v, ep, perm](Encoder& e, int call) {
        const uint32_t c = uint32_t(call) % C;
        e.use(pl);
        e.buf(0, x);
        e.buf(1, m.q, c * m.qexp());
        e.buf(2, m.s, c * m.sexp());
        e.buf(3, y);
        e.buf(8, g_gtab);
        if (v.kind == "x2p") {  // permute X (zz_vrows_xperm / nh_x2_xperm), then the kernel reads it
          struct { uint32_t K, x_ld, R; } pp{K, K, M};
          e.use(perm);
          e.buf(0, x); e.buf(1, xp); e.bytes(2, &pp, sizeof pp);
          e.dispatch((K / 8 * 4 + 63) / 64, 1, 1, 64);
          e.use(pl);
          e.buf(0, xp);
        }
        if (v.kind == "gemm") {
          struct { uint32_t M, N, K, x_ld, y_ld; float gs; uint32_t epi, qe, se, k; } gp{M, N, K, K, N, 1.0f, ep, 0, 0, 1};
          e.bytes(4, &gp, sizeof gp);
          e.buf(5, dummy); e.buf(6, dummy); e.buf(7, dummy);
          e.dispatch((N + 63) / 64, (M + v.rows_tg - 1) / v.rows_tg, 1, 128);
        } else {
          struct { uint32_t N, K, x_ld, y_ld, R; float gs; uint32_t epi; } vp{N, K, K, N, M, 1.0f, ep};
          e.bytes(4, &vp, sizeof vp);
          e.dispatch((N + v.rows_tg - 1) / v.rows_tg, 1, 1, v.threads);
        }
      };
    };
    memset(y.ptr(), 0, y.size);
    run_once(mk(0));
    double maxerr = 0, maxref = 0;
    for (size_t i = 0; i < rows.size(); ++i)
      for (uint32_t t = 0; t < M; ++t) {
        maxerr = std::max(maxerr, std::fabs(ref[i * M + t] - y.as<float>()[size_t(t) * N + rows[i]]));
        maxref = std::max(maxref, std::fabs(ref[i * M + t]));
      }
    char buf[160];
    snprintf(buf, sizeof buf, "%-28s %-22s", label.c_str(), v.name.c_str());
    lines.push_back({buf, rel(maxerr, maxref)});
    encs.push_back(mk(epi));
  }
  const std::vector<double> ms = time_rr(encs);
  const double bytes = m.bytes_per_exp();
  print_lines(lines, ms, [&](size_t, double t) {
    char b[96];
    snprintf(b, sizeof b, "%8.4f ms %6.1f GB/s %5.2f TFLOP/s", t, bytes / t / 1e6, 2.0 * M * N * K / t / 1e9);
    return std::string(b);
  });
  for (Buffer* b : {&x, &y, &dummy, &xp}) g.release(*b);
  m.release();
}


// ---- --stats: code size, stack (spills) and GPR count of kernels from their compiled binary archive (no GPU work;
// the metadata parsing is ni_compile.cpp's) ----
using Bytes = std::vector<uint8_t>;
uint32_t be32(const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }
uint32_t le32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
uint64_t le64(const uint8_t* p) { uint64_t v; memcpy(&v, p, 8); return v; }
bool macho_section(const uint8_t* m, size_t n, const char* seg, const char* sect, const uint8_t** out, size_t* len) {
  if (n < 32) return false;
  uint32_t ncmds = le32(m + 16);
  size_t off = 32;
  for (uint32_t i = 0; i < ncmds && off + 8 <= n; ++i) {
    uint32_t cmd = le32(m + off), csz = le32(m + off + 4);
    if (cmd == 0x19 && off + 72 <= n) {
      uint32_t nsect = le32(m + off + 64);
      for (uint32_t s = 0; s < nsect; ++s) {
        const uint8_t* sd = m + off + 72 + 80 * s;
        if (sd + 80 > m + n) return false;
        if (strncmp((const char*)sd, sect, 16) == 0 && strncmp((const char*)sd + 16, seg, 16) == 0) {
          uint64_t size = le64(sd + 40);
          uint32_t soff = le32(sd + 48);
          if (soff + size > n) return false;
          *out = m + soff;
          *len = size;
          return true;
        }
      }
    }
    off += csz;
  }
  return false;
}
void kernel_stats(const Bytes& b, long* text, long* stack, long* gprs) {
  *text = *stack = *gprs = -1;
  if (b.size() < 8) return;
  uint32_t nfat = be32(b.data() + 4);
  for (uint32_t i = 0; i < nfat && 8 + 20 * (i + 1) <= b.size(); ++i) {
    const uint8_t* e = b.data() + 8 + 20 * i;
    if (be32(e) != 0x01000013) continue;
    uint32_t off = be32(e + 8), size = be32(e + 12);
    if (off + size > b.size()) return;
    const uint8_t *inner, *txt, *md;
    size_t ninner, ntxt, nmd;
    if (!macho_section(b.data() + off, size, "__TEXT", "__compute", &inner, &ninner)) return;
    if (macho_section(inner, ninner, "__TEXT", "__text", &txt, &ntxt)) *text = long(ntxt);
    if (!macho_section(inner, ninner, "__GPU_METADATA", "__compute", &md, &nmd)) return;
    std::vector<uint32_t> w(nmd / 4);
    for (size_t k = 0; k < w.size(); ++k) w[k] = le32(md + 4 * k);
    for (size_t k = 4; k + 2 < w.size(); ++k) {
      uint32_t f = w[k];
      if (f >> 24 != 1 || (((f >> 16) & 0xff) != 1 && ((f >> 16) & 0xff) != 3) || f == 0x01000000 || (f & 0xff) > 3)
        continue;
      uint32_t c = w[k + 1];
      bool seen = c > 0 && (w[k - 1] == c || w[k - 2] == c || w[k - 3] == c || w[k - 4] == c);
      if (seen && w[k + 2] >= 4 && w[k + 2] <= 256) { *stack = c; *gprs = w[k + 2]; return; }
      if (c <= 256) { *stack = 0; *gprs = c; return; }
    }
    return;
  }
}
void print_stats(const std::vector<std::string>& names, const std::string& dir) {
  for (const std::string& n : names) {
    const std::string path = dir + "/" + n + ".bin";
    if (!Gpu::get().archive(n, path)) {
      printf("%-28s archive failed\n", n.c_str());
      continue;
    }
    std::ifstream f(path, std::ios::binary);
    Bytes b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    long text, stack, gprs;
    kernel_stats(b, &text, &stack, &gprs);
    printf("%-28s gprs %3ld  code %6ld B  stack %ld B\n", n.c_str(), gprs, text, stack);
  }
}

// Dev variants must belong to the harness they are timed in (a kernel with another signature reads its parameters from
// the wrong buffers: a bucketed MoE kernel run as a matvec looped ~1e6 times per lane over unmapped addresses and hung
// the GPU). Matvecs: names with "mv" or "rows"; several-row shared: "v8", "v16", "v32", "gemm", "vrows";
// routed schemes: "moe", "up", "dn", "bucket".
void check_family(const std::string& name, const std::vector<std::string>& family, const char* what) {
  for (const std::string& f : family)
    if (name.find(f) != std::string::npos) return;
  fatal("%s is not a %s kernel (refusing to dispatch it with that signature)", name.c_str(), what);
}

std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> f;
  std::stringstream ss(s);
  std::string it;
  while (std::getline(ss, it, c)) f.push_back(it);
  return f;
}

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  std::vector<std::string> only;
  std::string extra_src, mv_extra, sh_extra, stats_dir;
  std::vector<std::string> stats_names;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--src" && i + 1 < argc) extra_src = argv[++i];
    else if (a == "--hot") g_hot = true;
    else if (a == "--validate") g_validate = g_progress = true;
    else if (a == "--progress") g_progress = true;
    else if (a == "--stats" && i + 2 < argc) {
      stats_names = split(argv[++i], ',');
      stats_dir = argv[++i];
    }
    else if (a == "--warm" && i + 1 < argc) g_warm_ms = atof(argv[++i]);
    else if (a == "--mv" && i + 1 < argc) mv_extra = argv[++i];
    else if (a == "--sh" && i + 1 < argc) sh_extra = argv[++i];
    else if (a == "--pscheme" && i + 1 < argc) {
      const auto f = split(argv[++i], ':');
      if (f.size() != 6) fatal("--pscheme up:rows:threads:dn:rows:threads");
      check_family(f[0], {"moe", "up"}, "routed up (pair)");
      check_family(f[3], {"moe", "dn"}, "routed down (pair)");
      g_extra_schemes.push_back([f](MoeCtx& c) {
        return pair_scheme(c, f[0], uint32_t(std::stoul(f[1])), uint32_t(std::stoul(f[2])), f[3], uint32_t(std::stoul(f[4])),
                           uint32_t(std::stoul(f[5])));
      });
    } else if (a == "--bscheme" && i + 1 < argc) {
      const auto f = split(argv[++i], ':');
      if (f.size() != 6) fatal("--bscheme bucket:up:up_rows:dn:dn_rows:BM");
      check_family(f[0], {"bucket"}, "bucket");
      check_family(f[1], {"up"}, "routed up (tiles)");
      check_family(f[3], {"dn"}, "routed down (tiles)");
      g_extra_schemes.push_back([f](MoeCtx& c) {
        return bucket_scheme(c, f[0], f[1], uint32_t(std::stoul(f[2])), f[3], uint32_t(std::stoul(f[4])), uint32_t(std::stoul(f[5])));
      });
    } else only.push_back(a);
  }
  auto want = [&](const std::string& n) {
    if (only.empty()) return true;
    for (auto& o : only)
      if (n.find(o) != std::string::npos) return true;
    return false;
  };
  std::string src = kNhKernelSource;
  if (!extra_src.empty()) {
    std::ifstream f(extra_src);
    if (!f) fatal("cannot read %s", extra_src.c_str());
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string extra = ss.str();
    const std::string mark = "// ---- nemotron/nh_x_iq2.metal ----";
    const size_t a = src.find(mark);
    if (a != std::string::npos && extra.find("void nh_x2_") != std::string::npos) {
      const size_t b = src.find("// ---- nemotron/", a + mark.size());
      src.erase(a, b == std::string::npos ? std::string::npos : b - a);
    }
    src += "\n" + extra;
  }
  g_grid = grid_from_source(src);
  Gpu& g = Gpu::get();
  g.compile_extra("Nemotron-H x2", src.c_str());
  if (!stats_names.empty()) {  // no GPU work
    print_stats(stats_names, stats_dir);
    return 0;
  }
  g_gtab = g.alloc(4096, "iq2_grid_half");
  {
    auto e = g.begin();
    e->use(pipe("nh_iq2_fill_tables2"));
    e->buf(0, g_gtab);
    e->dispatch(1, 1, 1, 256);
    e->commit_and_wait();
  }

  // one row
  std::vector<Var> mv_up = {{"nh_mv1_iq2_c_relu2", 16, 128}, {"nh_x2_mv4_relu2", 4, 32}};
  std::vector<Var> mv_dn = {{"nh_mv1_iq2_c_add", 16, 128}, {"nh_x2_mv3_add", 3, 32}, {"nh_x2_mv4_add", 4, 32}};
  for (const Var& v : parse_vars(mv_extra)) {
    check_family(v.name, {"mv", "rows"}, "one-row matvec");
    (epi_of(v.name) == 1 ? mv_dn : mv_up).push_back(v);
  }
  if (g_validate) {  // odd sizes, K tails of 16 and 8 sub-blocks
    if (want("val mv")) bench_mv("val mv 50x1536 relu2", 50, 1536, mv_up);
    if (want("val mv")) bench_mv("val mv 37x2304 add", 37, 2304, mv_dn);
  } else {
    if (want("mv up")) bench_mv("mv up 5376x4096 relu2", 5376, 4096, mv_up);
    if (want("mv dn")) bench_mv("mv dn 4096x5376 add", 4096, 5376, mv_dn);
  }
  // routed experts
  auto schemes = [&](uint32_t R) {
    std::vector<std::function<MoeScheme(MoeCtx&)>> v;
    v.push_back([](MoeCtx& c) { return pair_scheme(c, "nh_moe2_up_iq2", 16, 128, "nh_moe2_dn_iq2", 16, 128); });
    v.push_back([](MoeCtx& c) { return pair_scheme(c, "nh_x2_moe_up4", 4, 32, "nh_x2_moe_dn4", 4, 32); });
    if (R == 1 || g_validate) v.push_back([](MoeCtx& c) { return pair_scheme(c, "nh_x2_moe_up4", 4, 32, "nh_x2_moe_dn3", 3, 32); });
    v.push_back([](MoeCtx& c) { return pair_scheme(c, "nh_x2_moe_up8", 8, 32, "nh_x2_moe_dn4", 4, 32); });
    if (R > 1) v.push_back([](MoeCtx& c) { return bucket_scheme(c, "nh_x2_bucket", "nh_x2_moeb_up", 4, "nh_x2_moeb_dn", 4, 2); });
    if (R > 1) v.push_back([](MoeCtx& c) { return bucket_scheme(c, "nh_x2_bucket", "nh_x2_moeb_up8", 8, "nh_x2_moeb_dn", 4, 2); });
    for (auto& f : g_extra_schemes) v.push_back(f);
    return v;
  };
  if (g_validate) {  // K a multiple of 256 (the IQ2_XXS block): latent 512 / width 768 (all-tail K: 16 and 24
                     // sub-blocks), 16 experts (many shared), 5 rows of which 4 live; latent 1024 / width 1280 (tail 8)
    if (want("val moe")) bench_moe(768, 512, 1, 16, 0xffffffffu, schemes(1));
    if (want("val moe")) bench_moe(768, 512, 5, 16, 4, schemes(5));
    if (want("val moe")) bench_moe(1280, 1024, 7, 24, 0xffffffffu, schemes(7));
  } else {
    for (uint32_t R : {1u, 8u, 32u})
      for (uint32_t I : {1536u, 2048u, 2304u, 2560u}) {
        const std::string nm = "moe" + std::to_string(R) + " I=" + std::to_string(I);
        if (!only.empty() && !want(nm)) continue;
        bench_moe(I, 1024, R, 368, 0xffffffffu, schemes(R));
      }
  }
  // shared expert, several rows
  for (uint32_t M : {8u, 32u}) {
    std::vector<SVar> sv;
    if (M <= 8) sv.push_back({"nh_v8_iq2_s4", "v8", 32, 128});
    sv.push_back({M <= 16 ? "nh_gemm_iq2_m16" : "nh_gemm_iq2_m32", "gemm", M <= 16 ? 16u : 32u, 128});
    if (M <= 8) sv.push_back({"nh_x2_v8", "x2", 8, 32});
    if (M <= 16) sv.push_back({"nh_x2_v16", "x2", 8, 32});
    sv.push_back({"nh_x2_v32", "x2", 8, 32});
    for (const SVar& v : parse_svars(sh_extra)) {
      check_family(v.name, {"v8", "v16", "v32", "gemm", "vrows"}, "several-row shared");
      sv.push_back(v);
    }
    if (g_validate) {
      if (!want("val sh")) continue;
      // partial token tiles (5, 13, 27 rows), 44 output rows (a partial row block), K 256 and 768
      for (uint32_t m : M == 8 ? std::vector<uint32_t>{5u, 8u} : std::vector<uint32_t>{13u, 27u}) {
        std::vector<SVar> sv2;
        for (const SVar& v : sv)
          if (!(v.name == "nh_x2_v16" && m > 16) && !(v.name == "nh_x2_v8" && m > 8)) sv2.push_back(v);
        bench_shared("val sh" + std::to_string(m) + " 44x256", 44, 256, m, 0, sv2);
        bench_shared("val sh" + std::to_string(m) + " 44x768", 44, 768, m, 0, sv2);
      }
      continue;
    }
    const std::string a = "sh" + std::to_string(M) + " up", b = "sh" + std::to_string(M) + " dn";
    if (want(a)) bench_shared(a + " 5376x4096 relu2", 5376, 4096, M, 2, sv);
    if (want(b)) bench_shared(b + " 4096x5376 add", 4096, 5376, M, 1, sv);
  }
  g.release(g_gtab);
  return 0;
}
