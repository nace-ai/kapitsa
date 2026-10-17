// IQ2_XXS kernels for Nemotron 3.5 Bolt (src/kernels/nemotron/nh_x_iq2.metal) on synthetic weights of the real
// shapes: GPU time per call and effective bandwidth, with a CPU reference check (fp64 dequant) of every kernel.
//   nh_x2_bench [options] [case-substring ...]          (default: every case)
//   --src <file.metal>   append a kernel source at run time (an embedded nh_x_iq2.metal section is left out)
//   --hot                one copy of the weights (default: several copies / expert sets in rotation, so repeated
//                        calls read DRAM as the model does)
//   --mv  name:rows_tg:threads[,...]   extra one-row matvec variants to time (signature of nh_mv1_iq2_c)
// Cases: "mv up" / "mv dn" (shared expert, one row), "moe1 W" (routed experts, one row), "moe8 W" / "moe32 W" (verify
// rows), "sh8" / "sh32" (shared expert, 8 / 32 rows).
#include "metal/gpu.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <functional>
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
bool g_hot = false;
double g_warm_ms = 150.0;  // GPU warm-up before each round-robin timing (--warm ms)
Buffer g_tab_u64, g_tab_h8;  // the grid as ulong[256] / as 8 halves per point (experiments: buffers 5 / 6)

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

// E stacked IQ2_XXS matrices of N x K in the repacked layout (codes [E][N][K/4], half d [E][N][K/256])
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
};

Iq2 make_iq2(uint32_t N, uint32_t K, uint32_t E) {
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

double time_kernel(int reps, int calls, const std::function<void(Encoder&, int)>& enc) {
  Gpu& g = Gpu::get();
  int call = 0;
  double warm = 0;
  for (int w = 0; w < 200 && (w < 3 || warm < 30.0); ++w) {  // warm-up: >= 30 ms of GPU work (clocks up)
    auto e = g.begin();
    for (int i = 0; i < calls; ++i) enc(*e, call++);
    e->commit_and_wait();
    warm += e->gpu_ms();
  }
  std::vector<double> ts;
  for (int r = 0; r < reps; ++r) {
    auto e = g.begin();
    for (int i = 0; i < calls; ++i) enc(*e, call++);
    e->commit_and_wait();
    ts.push_back(e->gpu_ms() / calls);
  }
  std::sort(ts.begin(), ts.end());
  return ts[ts.size() / 2];
}

// Several kernels timed round-robin (one command buffer of `calls` calls each per round, after a warm-up), so clock or
// load drift hits all alike: median ms per call of each
std::vector<double> time_rr(const std::vector<std::function<void(Encoder&, int)>>& encs, int rounds = 15, int calls = 20) {
  Gpu& g = Gpu::get();
  int call = 0;
  double warm = 0;
  for (int w = 0; w < 2000 && warm < g_warm_ms; ++w) {
    auto e = g.begin();
    for (int i = 0; i < calls; ++i) encs[size_t(w) % encs.size()](*e, call++);
    e->commit_and_wait();
    warm += e->gpu_ms();
  }
  std::vector<std::vector<double>> ts(encs.size());
  for (int r = 0; r < rounds; ++r)
    for (size_t v = 0; v < encs.size(); ++v) {
      auto e = g.begin();
      for (int i = 0; i < calls; ++i) encs[v](*e, call++);
      e->commit_and_wait();
      ts[v].push_back(e->gpu_ms() / calls);
    }
  std::vector<double> med;
  for (auto& t : ts) {
    std::sort(t.begin(), t.end());
    med.push_back(t[t.size() / 2]);
  }
  return med;
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

int epi_of(const std::string& n) { return n.find("relu2") != std::string::npos ? 2 : n.find("_add") != std::string::npos ? 1 : 0; }


// ---- register use (from ni_compile.cpp): compiled code size, stack (spills) and GPR count of a kernel ----
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
// " gprs N [stack S]" for a kernel (env X2_STATS_DIR: where the binary archives go), or ""
std::string kstats(const std::string& name) {
  const char* dir = getenv("X2_STATS_DIR");
  if (!dir) return "";
  const std::string path = std::string(dir) + "/" + name + ".bin";
  if (!Gpu::get().archive(name, path)) return " (archive failed)";
  std::ifstream f(path, std::ios::binary);
  Bytes b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  long text, stack, gprs;
  kernel_stats(b, &text, &stack, &gprs);
  char buf[96];
  snprintf(buf, sizeof buf, "  gprs %ld code %ld%s", gprs, text, stack > 0 ? (" STACK " + std::to_string(stack)).c_str() : "");
  return buf;
}

double rel(double err, double ref) { return err / std::max(ref, 1e-30); }

struct MvP {
  uint32_t N, K, x_stride, y_stride;
  float gs;
  uint32_t R;
};

// ---- one-row matvec: y = epi(W x) ----
void bench_mv(const char* label, uint32_t N, uint32_t K, const std::vector<Var>& vars) {
  Gpu& g = Gpu::get();
  const uint32_t C = g_hot ? 1 : 6;  // copies in rotation (6 x 5.7 MB)
  Iq2 m = make_iq2(N, K, C);
  Buffer x = g.alloc(size_t(K) * 4, "x"), y = g.alloc(size_t(N) * 4, "y");
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (size_t i = 0; i < K; ++i) x.as<float>()[i] = nd(rng);
  // reference rows (copy 0)
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
  std::vector<std::function<void(Encoder&, int)>> encs;
  std::vector<std::string> lines;
  for (const Var& v : vars) {
    Pipeline pl = g.pipeline(v.name, false);
    if (!pl.pso) {
      printf("%-26s %-28s (missing)\n", label, v.name.c_str());
      continue;
    }
    const int epi = epi_of(v.name);
    const MvP p{N, K, K, N, 1.0f, 1};
    auto enc = [&m, &x, &y, pl, p, v, C](Encoder& e, int call) {
      const uint32_t c = uint32_t(call) % C;
      e.use(pl);
      e.buf(0, m.q, c * m.qexp());
      e.buf(1, m.s, c * m.sexp());
      e.buf(2, x);
      e.buf(3, y);
      e.bytes(4, &p, sizeof p);
      e.buf(5, g_tab_u64);
      e.buf(6, g_tab_h8);
      e.dispatch((p.N + v.rows_tg - 1) / v.rows_tg, 1, 1, v.threads);
    };
    // check: one call on copy 0, y preset to 0.25 (the add epilogue adds to it)
    for (uint32_t n = 0; n < N; ++n) y.as<float>()[n] = 0.25f;
    {
      auto e = g.begin();
      enc(*e, 0);
      e->commit_and_wait();
    }
    double maxerr = 0, maxref = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
      double r = ref[i];
      if (epi == 1) r += 0.25;
      if (epi == 2) r = r > 0 ? r * r : 0.0;
      maxerr = std::max(maxerr, std::fabs(r - y.as<float>()[rows[i]]));
      maxref = std::max(maxref, std::fabs(r));
    }
    char buf[256];
    snprintf(buf, sizeof buf, "%-26s %-28s %%8.4f ms  %%6.1f GB/s  rel.err %.2e%s", label, v.name.c_str(),
             rel(maxerr, maxref), kstats(v.name).c_str());
    lines.push_back(buf);
    encs.push_back(enc);
  }
  const std::vector<double> ms = time_rr(encs);
  for (size_t i = 0; i < encs.size(); ++i) {
    printf(lines[i].c_str(), ms[i], m.bytes_per_exp() / ms[i] / 1e6);
    printf("\n");
  }
  g.release(x);
  g.release(y);
  g.release(m.q);
  g.release(m.s);
}


struct MoeP {
  uint32_t I, D, k, up_q_stride, up_s_stride, dn_q_stride, dn_s_stride, h_stride;
};

// ---- routed experts, one token (decode): k = 9 experts of E, latent D = 1024, width I; the nh_moe2 signatures:
// up h[pair] = relu(W_up[e] . lat)^2, down part[pair] = ew * W_dn[e] . h[pair] (both per pair) ----
void bench_moe1(uint32_t I, const std::vector<Var>& ups, const std::vector<Var>& dns) {
  Gpu& g = Gpu::get();
  const uint32_t D = 1024, k = 9, E = 48, NS = 16;  // NS expert sets in rotation
  Iq2 up = make_iq2(I, D, E), dn = make_iq2(D, I, E);
  Buffer lat = g.alloc(size_t(D) * 4, "lat"), h = g.alloc(size_t(k) * I * 4, "h"), part = g.alloc(size_t(k) * D * 4, "part");
  Buffer eid = g.alloc(size_t(NS) * k * 4, "eid"), ew = g.alloc(size_t(k) * 4, "ew"), live = g.alloc(16, "live");
  live.as<uint32_t>()[0] = 0xffffffffu;
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (uint32_t i = 0; i < D; ++i) lat.as<float>()[i] = nd(rng) * 0.5f;
  for (uint32_t j = 0; j < k; ++j) ew.as<float>()[j] = 0.1f + 0.01f * float(j);
  std::mt19937 er(7);
  for (uint32_t t = 0; t < NS; ++t) {
    std::vector<int> got;
    while (got.size() < k) {
      const int v = int(er() % E);
      if (std::find(got.begin(), got.end(), v) == got.end()) got.push_back(v);
    }
    for (uint32_t j = 0; j < k; ++j) eid.as<int>()[t * k + j] = got[j];
  }
  const MoeP mp{I, D, k, uint32_t(up.qexp()), uint32_t(up.sexp()), uint32_t(dn.qexp()), uint32_t(dn.sexp()), I};
  // references of set 0: h (fp64) and, for the down check, part from the GPU's h
  std::vector<double> href(size_t(k) * I), w;
  for (uint32_t j = 0; j < k; ++j) {
    const uint32_t e = uint32_t(eid.as<int>()[j]);
    for (uint32_t n = 0; n < I; ++n) {
      up.row(e, n, w);
      double a = 0;
      for (uint32_t kk = 0; kk < D; ++kk) a += w[kk] * lat.as<float>()[kk];
      a = std::max(a, 0.0);
      href[size_t(j) * I + n] = a * a;
    }
  }
  const double bytes = double(k) * up.bytes_per_exp();
  char label[64];
  snprintf(label, sizeof label, "moe1 I=%u up", I);
  std::vector<std::function<void(Encoder&, int)>> encs;
  std::vector<std::string> lines;
  for (const Var& v : ups) {
    Pipeline pl = g.pipeline(v.name, false);
    if (!pl.pso) { printf("%-26s %-28s (missing)\n", label, v.name.c_str()); continue; }
    auto enc = [&, pl, v](Encoder& e, int call) {
      e.use(pl);
      e.buf(0, up.q); e.buf(1, up.s); e.buf(2, lat); e.buf(3, h);
      e.buf(4, eid, size_t(uint32_t(call) % NS) * k * 4);
      e.bytes(5, &mp, sizeof mp); e.buf(7, live);
      e.dispatch((I + v.rows_tg - 1) / v.rows_tg, k, 1, v.threads);
    };
    memset(h.ptr(), 0, h.size);
    { auto e = g.begin(); enc(*e, 0); e->commit_and_wait(); }
    double maxerr = 0, maxref = 0;
    for (size_t i = 0; i < href.size(); ++i) {
      maxerr = std::max(maxerr, std::fabs(href[i] - h.as<float>()[i]));
      maxref = std::max(maxref, std::fabs(href[i]));
    }
    char buf[256];
    snprintf(buf, sizeof buf, "%-26s %-28s %%8.4f ms  %%6.1f GB/s  rel.err %.2e%s", label, v.name.c_str(), rel(maxerr, maxref),
             kstats(v.name).c_str());
    lines.push_back(buf);
    encs.push_back(enc);
  }
  {
    const std::vector<double> ms = time_rr(encs);
    for (size_t i = 0; i < encs.size(); ++i) { printf(lines[i].c_str(), ms[i], bytes / ms[i] / 1e6); printf("\n"); }
  }
  // down: from a fixed h (the last up kernel's output of set 0, recomputed below)
  {
    auto e = g.begin();
    encs.empty() ? void() : encs[0](*e, 0);
    e->commit_and_wait();
  }
  std::vector<double> pref(size_t(k) * D);
  for (uint32_t j = 0; j < k; ++j) {
    const uint32_t e = uint32_t(eid.as<int>()[j]);
    for (uint32_t n = 0; n < D; ++n) {
      dn.row(e, n, w);
      double a = 0;
      for (uint32_t kk = 0; kk < I; ++kk) a += w[kk] * h.as<float>()[size_t(j) * I + kk];
      pref[size_t(j) * D + n] = a * ew.as<float>()[j];
    }
  }
  snprintf(label, sizeof label, "moe1 I=%u down", I);
  encs.clear();
  lines.clear();
  for (const Var& v : dns) {
    Pipeline pl = g.pipeline(v.name, false);
    if (!pl.pso) { printf("%-26s %-28s (missing)\n", label, v.name.c_str()); continue; }
    auto enc = [&, pl, v](Encoder& e, int call) {
      e.use(pl);
      e.buf(0, dn.q); e.buf(1, dn.s); e.buf(2, h); e.buf(3, part);
      e.buf(4, eid, size_t(uint32_t(call) % NS) * k * 4);
      e.buf(5, ew); e.bytes(6, &mp, sizeof mp); e.buf(8, live);
      e.dispatch((D + v.rows_tg - 1) / v.rows_tg, k, 1, v.threads);
    };
    memset(part.ptr(), 0, part.size);
    { auto e = g.begin(); enc(*e, 0); e->commit_and_wait(); }
    double maxerr = 0, maxref = 0;
    for (size_t i = 0; i < pref.size(); ++i) {
      maxerr = std::max(maxerr, std::fabs(pref[i] - part.as<float>()[i]));
      maxref = std::max(maxref, std::fabs(pref[i]));
    }
    char buf[256];
    snprintf(buf, sizeof buf, "%-26s %-28s %%8.4f ms  %%6.1f GB/s  rel.err %.2e%s", label, v.name.c_str(), rel(maxerr, maxref),
             kstats(v.name).c_str());
    lines.push_back(buf);
    encs.push_back(enc);
  }
  {
    const std::vector<double> ms = time_rr(encs);
    for (size_t i = 0; i < encs.size(); ++i) { printf(lines[i].c_str(), ms[i], bytes / ms[i] / 1e6); printf("\n"); }
  }
  for (Buffer* b : {&lat, &h, &part, &eid, &ew, &live, &up.q, &up.s, &dn.q, &dn.s}) g.release(*b);
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> only;
  std::string extra_src, mv_extra, up_extra, dn_extra;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--src" && i + 1 < argc) extra_src = argv[++i];
    else if (a == "--hot") g_hot = true;
    else if (a == "--warm" && i + 1 < argc) g_warm_ms = atof(argv[++i]);
    else if (a == "--mv" && i + 1 < argc) mv_extra = argv[++i];
    else if (a == "--up" && i + 1 < argc) up_extra = argv[++i];
    else if (a == "--dn" && i + 1 < argc) dn_extra = argv[++i];
    else only.push_back(a);
  }
  auto want = [&](const std::string& n) {
    if (only.empty()) return true;
    for (auto& o : only)
      if (n.find(o) != std::string::npos) return true;
    return false;
  };
  std::string src = kNhKernelSource;
  if (!extra_src.empty()) {
    // drop the embedded nh_x_iq2.metal section, append the file
    const std::string mark = "// ---- nemotron/nh_x_iq2.metal ----";
    const size_t a = src.find(mark);
    if (a != std::string::npos) {
      const size_t b = src.find("// ---- nemotron/", a + mark.size());
      src.erase(a, b == std::string::npos ? std::string::npos : b - a);
    }
    std::ifstream f(extra_src);
    if (!f) fatal("cannot read %s", extra_src.c_str());
    std::stringstream ss;
    ss << f.rdbuf();
    src += "\n" + ss.str();
  }
  g_grid = grid_from_source(src);
  Gpu& g = Gpu::get();
  g.compile_extra("Nemotron-H x2", src.c_str());
  g_tab_u64 = g.alloc(256 * 8, "grid_u64");
  g_tab_h8 = g.alloc(256 * 16, "grid_h8");
  for (int i = 0; i < 256; ++i) {
    g_tab_u64.as<uint64_t>()[i] = g_grid[size_t(i)];
    for (int j = 0; j < 8; ++j) g_tab_h8.as<__fp16>()[i * 8 + j] = __fp16(float((g_grid[size_t(i)] >> (8 * j)) & 0xFF));
  }

  std::vector<Var> mv_up = {{"nh_mv1_iq2_c_relu2", 16, 128}}, mv_dn = {{"nh_mv1_iq2_c_add", 16, 128}};
  for (const Var& v : parse_vars(mv_extra)) (epi_of(v.name) == 1 ? mv_dn : mv_up).push_back(v);
  if (want("mv up")) bench_mv("mv up 5376x4096 relu2", 5376, 4096, mv_up);
  if (want("mv dn")) bench_mv("mv dn 4096x5376 add", 4096, 5376, mv_dn);
  std::vector<Var> m_up = {{"nh_moe2_up_iq2", 16, 128}}, m_dn = {{"nh_moe2_dn_iq2", 16, 128}};
  for (const Var& v : parse_vars(up_extra)) m_up.push_back(v);
  for (const Var& v : parse_vars(dn_extra)) m_dn.push_back(v);
  for (uint32_t I : {1536u, 2048u, 2304u, 2560u})
    if (want("moe1 I=" + std::to_string(I)) || want("moe1 all")) bench_moe1(I, m_up, m_dn);
  return 0;
}
