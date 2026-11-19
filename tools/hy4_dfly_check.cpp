// Checks Hy4-flash's DFly2 drafter (src/model/hy4_dfly.h) against an fp64 CPU reference written from the spec
// (vLLM d2d72635's hy_v4 dfly.py) with the attention expanded (k_nope and v per key through
// kv_b_proj, softmax with the sink), not absorbed as the GPU does. The reference mirrors the stored precisions: the target
// hidden held in bf16, latent rows rounded to bf16 then stored half (hy4::LatentCache), RoPE's cos/sin rounded to bf16.
// The head is teacher-forced on the GPU's drafted ids, so a near-tie flip does not cascade.
//   check [--a N]   synthetic drafter (D 256, 16 heads, 2 layers, vocab 1024, 8-bit embedding and head) at a = 0, 37,
//                   301 (or N): block rows written to the caches, z, the 5 steps' logits and ids
//   pack DIR [--a N] the real drafter of a converted pack (model-dfly.safetensors, the target's embedding and head
//                   from its shards, config.json), random history rows; the same comparisons (default a = 300)
//   time DIR        the real drafter's GPU time a cycle at a = 64, 512, 2042 (random history)
//   --config PATH   the pack's config.json when DIR has none
// Takes no lock: run it under lockf -k -t 600 /tmp/kapitsa-gpu.lock build/hy4_dfly_check ...
#include "metal/gpu.h"
#include "model/dflash2.h"
#include "model/hy4_dfly.h"
#include "util/bf16.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace ni;

namespace {

int g_fail = 0;
#define CHECK(c, ...)                             \
  do {                                            \
    if (!(c)) {                                   \
      std::fprintf(stderr, "FAIL: " __VA_ARGS__); \
      std::fprintf(stderr, "\n");                 \
      ++g_fail;                                   \
    }                                             \
  } while (0)

void parallel_for(size_t n, const std::function<void(size_t)>& f) {
  const unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
  std::atomic<size_t> next{0};
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; ++t)
    th.emplace_back([&] {
      for (size_t i; (i = next.fetch_add(1)) < n;) f(i);
    });
  for (auto& t : th) t.join();
}

uint16_t ld16(const uint8_t* p) {
  uint16_t v;
  std::memcpy(&v, p, 2);
  return v;
}
double bf(uint16_t v) { return bf16_to_f32(v); }
double rbf(double x) { return bf16_to_f32(f32_to_bf16(float(x))); }
double rhalf(double x) { return double(_Float16(float(x))); }

// ---- a safetensors file mapped read-only: name -> bf16 data and shape ----
struct Mapped {
  struct T {
    const uint8_t* p = nullptr;
    std::string dtype;
    std::vector<int64_t> shape;
  };
  std::map<std::string, T> t;
  void* base = nullptr;
  size_t len = 0;
  explicit Mapped(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) fatal("cannot open %s", path.c_str());
    struct stat st;
    fstat(fd, &st);
    len = size_t(st.st_size);
    base = mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) fatal("mmap %s failed", path.c_str());
    const uint8_t* b = static_cast<const uint8_t*>(base);
    uint64_t n;
    std::memcpy(&n, b, 8);
    const nlohmann::json j = nlohmann::json::parse(std::string(reinterpret_cast<const char*>(b + 8), size_t(n)));
    for (auto it = j.begin(); it != j.end(); ++it) {
      if (it.key() == "__metadata__") continue;
      T x;
      x.dtype = it.value()["dtype"];
      x.shape = it.value()["shape"].get<std::vector<int64_t>>();
      x.p = b + 8 + n + it.value()["data_offsets"][0].get<uint64_t>();
      t[it.key()] = x;
    }
  }
  ~Mapped() { munmap(base, len); }
  const T& get(const std::string& n) const {
    auto it = t.find("model.dfly_layers." + n);
    if (it == t.end()) fatal("reference: %s missing", n.c_str());
    return it->second;
  }
};

// a bf16 matrix [N, K] (rows r0..)
struct Mat {
  const uint8_t* p = nullptr;
  uint32_t N = 0, K = 0;
  double at(uint32_t n, uint32_t k) const { return bf(ld16(p + (size_t(n) * K + k) * 2)); }
};
Mat mat(const Mapped& m, const std::string& n, uint32_t r0 = 0, uint32_t rows = 0) {
  const Mapped::T& t = m.get(n);
  Mat M;
  M.K = uint32_t(t.shape[1]);
  M.N = rows ? rows : uint32_t(t.shape[0]) - r0;
  M.p = t.p + size_t(r0) * M.K * 2;
  return M;
}
std::vector<double> vec(const Mapped& m, const std::string& n) {
  const Mapped::T& t = m.get(n);
  int64_t ne = 1;
  for (int64_t s : t.shape) ne *= s;
  std::vector<double> v(static_cast<size_t>(ne));
  for (int64_t i = 0; i < ne; ++i) v[size_t(i)] = bf(ld16(t.p + size_t(i) * 2));
  return v;
}

using Vec = std::vector<double>;
Vec linear(const Mat& W, const Vec& x, size_t x0 = 0) {
  Vec y(W.N);
  auto row = [&](size_t n) {
    double s = 0;
    const uint8_t* r = W.p + n * W.K * 2;
    for (uint32_t k = 0; k < W.K; ++k) s += bf(ld16(r + size_t(k) * 2)) * x[x0 + k];
    y[n] = s;
  };
  if (size_t(W.N) * W.K > (1u << 20)) parallel_for(W.N, row);
  else
    for (size_t n = 0; n < W.N; ++n) row(n);
  return y;
}
Vec rms(const Vec& x, const Vec& w, double eps, size_t x0 = 0, size_t n = 0) {
  if (!n) n = w.size();
  double ss = 0;
  for (size_t i = 0; i < n; ++i) ss += x[x0 + i] * x[x0 + i];
  const double inv = 1.0 / std::sqrt(ss / double(n) + eps);
  Vec y(n);
  for (size_t i = 0; i < n; ++i) y[i] = w[i] * (x[x0 + i] * inv);
  return y;
}
double silu(double x) { return x / (1.0 + std::exp(-x)); }

// the target's RoPE on 64 dims as the GPU computes angles (fp32), cos / sin rounded to bf16 when rope_bf16
void rope(double* x, int pos, float theta, bool rbf16) {
  for (int i = 0; i < 32; ++i) {
    const float inv = 1.0f / std::pow(theta, float(2 * i) / 64.0f);
    const float ang = float(pos) * inv;
    double c = std::cos(ang), s = std::sin(ang);
    if (rbf16) c = rbf(c), s = rbf(s);
    const double x0 = x[2 * i], x1 = x[2 * i + 1];
    x[2 * i] = x0 * c - x1 * s;
    x[2 * i + 1] = x1 * c + x0 * s;
  }
}

// row r of an affine-quantized QLinear, dequantized as the GPU does (scale * code + bias)
Vec deq(const QLinear& L, uint32_t r) {
  const uint32_t pw = 32 / uint32_t(L.bits), mask = (1u << L.bits) - 1u;
  const uint32_t* w = L.w->buf.as<uint32_t>() + size_t(r) * (L.K / pw);
  const uint16_t *s = L.s->buf.as<uint16_t>() + size_t(r) * (L.K / 64), *b = L.b->buf.as<uint16_t>() + size_t(r) * (L.K / 64);
  Vec y(L.K);
  for (uint32_t k = 0; k < L.K; ++k)
    y[k] = double(float(bf16_to_f32(s[k / 64])) * float((w[k / pw] >> (L.bits * (k % pw))) & mask) + bf16_to_f32(b[k / 64]));
  return y;
}

struct RefOut {
  std::vector<Vec> blk;  // per layer: [6][576] as stored (bf16, then half)
  std::vector<Vec> z;    // [6][D]
  std::vector<Vec> logits;  // [5][V]
};

// The spec's forward. hist[i]: layer i's cache rows [a][576] as stored; gpu_ids: teacher forcing for the head.
RefOut reference(const Mapped& m, const Hy4DflyConfig& c, const QLinear& emb, const QLinear& head, const Vec& h,
                 uint32_t t0, int a, const std::vector<Vec>& hist, const std::vector<uint32_t>& gpu_ids) {
  const uint32_t D = c.D, H = c.H;
  const double eps = c.eps;
  const int B = 6;
  RefOut o;
  std::vector<Vec> X(B);
  for (int j = 0; j < B; ++j) X[size_t(j)] = deq(emb, j == 0 ? t0 : c.mask_token);
  {
    Vec hb(D);
    for (uint32_t i = 0; i < D; ++i) hb[i] = rbf(h[i]);
    const Vec e0 = rms(X[0], vec(m, "mtp_fusion.enorm.weight"), eps), h0 = rms(hb, vec(m, "mtp_fusion.hnorm.weight"), eps);
    Vec cat(e0);
    cat.insert(cat.end(), h0.begin(), h0.end());
    const Vec f = linear(mat(m, "mtp_fusion.eh_proj.weight"), cat), g = vec(m, "mtp_fusion.gate");
    for (uint32_t i = 0; i < D; ++i) X[0][i] += g[i] * f[i];
  }
  for (size_t li = 0; li < c.kv_layers.size(); ++li) {
    const std::string p = std::to_string(li) + ".";
    const Vec in_ln = vec(m, p + "input_layernorm.weight"), qan = vec(m, p + "self_attn.q_a_layernorm.weight"),
              kvn = vec(m, p + "self_attn.kv_a_layernorm.weight"), sinks = vec(m, p + "self_attn.learnable_sink_param"),
              ppn = vec(m, p + "self_attn.post_proj_norm.weight"), pn = vec(m, p + "post_attention_layernorm.weight");
    const Mat qa = mat(m, p + "self_attn.q_a_proj.weight"), qb = mat(m, p + "self_attn.q_b_proj.weight"),
              kvl = mat(m, p + "self_attn.kv_a_proj_with_mqa.weight", 0, 512),
              kvr = mat(m, p + "self_attn.kv_a_proj_with_mqa.weight", 512, 64),
              kvb = mat(m, p + "self_attn.kv_b_proj.weight"), gate = mat(m, p + "self_attn.linear_gate.weight"),
              op = mat(m, p + "self_attn.o_proj.weight"), mg = mat(m, p + "mlp.gate_proj.weight"),
              mu = mat(m, p + "mlp.up_proj.weight"), md = mat(m, p + "mlp.down_proj.weight");
    std::vector<Vec> Q(B), G(B), U(B);
    Vec blk(size_t(B) * 576);
    for (int j = 0; j < B; ++j) {
      U[size_t(j)] = rms(X[size_t(j)], in_ln, eps);
      Q[size_t(j)] = linear(qb, rms(linear(qa, U[size_t(j)]), qan, eps));
      G[size_t(j)] = linear(gate, U[size_t(j)]);
      const Vec lat = rms(linear(kvl, U[size_t(j)]), kvn, eps);
      Vec kr = linear(kvr, U[size_t(j)]);
      rope(kr.data(), a + j, c.theta, c.rope_bf16);
      for (int i = 0; i < 512; ++i) blk[size_t(j) * 576 + i] = rhalf(rbf(lat[size_t(i)]));
      for (int i = 0; i < 64; ++i) blk[size_t(j) * 576 + 512 + i] = rhalf(rbf(kr[size_t(i)]));
      for (uint32_t hh = 0; hh < H; ++hh) rope(Q[size_t(j)].data() + hh * 256 + 192, a + j, c.theta, c.rope_bf16);
    }
    o.blk.push_back(blk);
    // per key and head: k_nope (192) and v (256) through this layer's kv_b
    const int nk = a + B;
    auto key = [&](int pk) -> const double* { return pk < a ? hist[li].data() + size_t(pk) * 576 : blk.data() + size_t(pk - a) * 576; };
    std::vector<Vec> KV(static_cast<size_t>(nk));
    parallel_for(size_t(nk), [&](size_t pk) {
      const double* cp = key(int(pk));
      Vec r(size_t(H) * 448);
      for (uint32_t n = 0; n < H * 448; ++n) {
        double s = 0;
        const uint8_t* wr = kvb.p + size_t(n) * 512 * 2;
        for (int k = 0; k < 512; ++k) s += bf(ld16(wr + size_t(k) * 2)) * cp[k];
        r[n] = s;
      }
      KV[pk] = std::move(r);
    });
    for (int j = 0; j < B; ++j) {
      Vec att(size_t(H) * 256);
      for (uint32_t hh = 0; hh < H; ++hh) {
        const double* q = Q[size_t(j)].data() + hh * 256;
        const int last = a + j;
        Vec s(size_t(last + 1));
        double mx = sinks[hh];
        for (int pk = 0; pk <= last; ++pk) {
          const double* kn = KV[size_t(pk)].data() + hh * 448;
          const double* kp = key(pk) + 512;
          double d = 0;
          for (int i = 0; i < 192; ++i) d += q[i] * kn[i];
          for (int i = 0; i < 64; ++i) d += q[192 + i] * kp[i];
          s[size_t(pk)] = d / 16.0;
          mx = std::max(mx, s[size_t(pk)]);
        }
        double den = std::exp(sinks[hh] - mx);
        for (int pk = 0; pk <= last; ++pk) den += (s[size_t(pk)] = std::exp(s[size_t(pk)] - mx));
        for (int i = 0; i < 256; ++i) {
          double acc = 0;
          for (int pk = 0; pk <= last; ++pk) acc += s[size_t(pk)] * KV[size_t(pk)][hh * 448 + 192 + i];
          att[hh * 256 + i] = acc / den * (1.0 / (1.0 + std::exp(-G[size_t(j)][hh * 256 + i])));
        }
      }
      const Vec y = rms(linear(op, att), ppn, eps);
      for (uint32_t i = 0; i < D; ++i) X[size_t(j)][i] += y[i];
      const Vec u2 = rms(X[size_t(j)], pn, eps);
      const Vec gg = linear(mg, u2), uu = linear(mu, u2);
      Vec hm(gg.size());
      for (size_t i = 0; i < hm.size(); ++i) hm[i] = silu(gg[i]) * uu[i];
      const Vec dd = linear(md, hm);
      for (uint32_t i = 0; i < D; ++i) X[size_t(j)][i] += dd[i];
    }
  }
  const Vec fn = vec(m, "final_norm.weight");
  for (int j = 0; j < B; ++j) o.z.push_back(rms(X[size_t(j)], fn, eps));
  const Vec hcn = vec(m, "hidden_correction.hidden_norm.weight"), hce = vec(m, "hidden_correction.embed_norm.weight");
  const Mat hg = mat(m, "hidden_correction.gate_proj.weight"), hu = mat(m, "hidden_correction.up_proj.weight"),
            hd = mat(m, "hidden_correction.down_proj.weight");
  for (int k = 1; k <= 5; ++k) {
    const uint32_t prev = k == 1 ? t0 : gpu_ids[size_t(k - 2)];
    Vec g = rms(o.z[size_t(k)], hcn, eps);
    const Vec en = rms(deq(emb, prev), hce, eps);
    g.insert(g.end(), en.begin(), en.end());
    const Vec gg = linear(hg, g), uu = linear(hu, g);
    Vec hm(gg.size());
    for (size_t i = 0; i < hm.size(); ++i) hm[i] = silu(gg[i]) * uu[i];
    const Vec dd = linear(hd, hm);
    Vec zc(o.z[size_t(k)]);
    for (uint32_t i = 0; i < D; ++i) zc[i] += dd[i];
    Vec lg(head.N);
    parallel_for(head.N / 64, [&](size_t blk) {
      for (size_t n = blk * 64; n < blk * 64 + 64; ++n) {
        const Vec w = deq(head, uint32_t(n));
        double s = 0;
        for (uint32_t i = 0; i < D; ++i) s += w[i] * zc[i];
        lg[n] = s;
      }
    });
    o.logits.push_back(lg);
  }
  return o;
}

// ---- inputs ----
Buffer abuf(size_t bytes, const char* label) {
  Buffer b = Gpu::get().alloc(std::max<size_t>(bytes, 16), label, false);
  std::memset(b.ptr(), 0, std::max<size_t>(bytes, 16));
  return b;
}

struct Target {  // the embedding and LM head (owned tensors)
  std::deque<Tensor> own;
  QLinear emb, head;
  std::unique_ptr<WeightStore> ws;
};

QLinear quantized(Target& T, const std::vector<uint16_t>& rows, uint32_t N, uint32_t K, int bits) {
  std::vector<const uint16_t*> rp(N);
  for (uint32_t i = 0; i < N; ++i) rp[i] = rows.data() + size_t(i) * K;
  const uint32_t wpr = K * uint32_t(bits) / 32, gpr = K / 64;
  Buffer w = abuf(size_t(N) * wpr * 4, "q_w"), s = abuf(size_t(N) * gpr * 2, "q_s"), b = abuf(size_t(N) * gpr * 2, "q_b");
  dfl_quantize_cpu(rp, K, bits, false, w.as<uint32_t>(), s.as<uint16_t>(), b.as<uint16_t>());
  QLinear L;
  T.own.push_back(Tensor{w, DType::U32, {int64_t(N), int64_t(wpr)}});
  L.w = &T.own.back();
  T.own.push_back(Tensor{s, DType::BF16, {int64_t(N), int64_t(gpr)}});
  L.s = &T.own.back();
  T.own.push_back(Tensor{b, DType::BF16, {int64_t(N), int64_t(gpr)}});
  L.b = &T.own.back();
  L.N = N, L.K = K, L.bits = bits, L.gs = 64;
  return L;
}

void write_safetensors(const std::string& path, const std::vector<std::pair<std::string, std::vector<int64_t>>>& meta,
                       const std::vector<std::vector<uint16_t>>& data) {
  nlohmann::json h = nlohmann::json::object();
  size_t off = 0;
  for (size_t i = 0; i < meta.size(); ++i) {
    h[meta[i].first] = {{"dtype", "BF16"}, {"shape", meta[i].second}, {"data_offsets", {off, off + data[i].size() * 2}}};
    off += data[i].size() * 2;
  }
  h["__metadata__"] = {{"format", "mlx"}};
  std::string s = h.dump();
  while ((8 + s.size()) % 2) s += ' ';  // MLX headers can leave odd offsets: the loader must not care
  s += ' ';                             // ... make it odd on purpose
  std::ofstream f(path, std::ios::binary);
  const uint64_t n = s.size();
  f.write(reinterpret_cast<const char*>(&n), 8);
  f.write(s.data(), std::streamsize(s.size()));
  for (const auto& d : data) f.write(reinterpret_cast<const char*>(d.data()), std::streamsize(d.size() * 2));
  if (!f) fatal("writing %s failed", path.c_str());
}

// A random drafter at small sizes: D, H heads, q_lora QA, MLP I, nl layers
std::string make_synthetic(Hy4DflyConfig& c, Target& T, uint32_t V, uint32_t QA, uint32_t I, std::mt19937& rng) {
  char tmpl[] = "/tmp/hy4dfly.XXXXXX";
  if (!mkdtemp(tmpl)) fatal("mkdtemp failed");
  const std::string dir = tmpl;
  const uint32_t D = c.D, H = c.H;
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<std::pair<std::string, std::vector<int64_t>>> meta;
  std::vector<std::vector<uint16_t>> data;
  auto add = [&](const std::string& n, std::vector<int64_t> shape, float sd, float mean = 0.0f) {
    size_t ne = 1;
    for (int64_t s : shape) ne *= size_t(s);
    std::vector<uint16_t> v(ne);
    for (auto& x : v) x = f32_to_bf16(mean + sd * nd(rng));
    meta.push_back({"model.dfly_layers." + n, shape});
    data.push_back(std::move(v));
  };
  auto lin = [&](const std::string& n, uint32_t N, uint32_t K, float gain = 1.0f) {
    add(n, {int64_t(N), int64_t(K)}, gain / std::sqrt(float(K)));
  };
  auto nrm = [&](const std::string& n, uint32_t N) { add(n, {int64_t(N)}, 0.1f, 1.0f); };
  for (size_t i = 0; i < c.kv_layers.size(); ++i) {
    const std::string p = std::to_string(i) + ".";
    nrm(p + "input_layernorm.weight", D);
    nrm(p + "post_attention_layernorm.weight", D);
    lin(p + "self_attn.q_a_proj.weight", QA, D);
    nrm(p + "self_attn.q_a_layernorm.weight", QA);
    lin(p + "self_attn.q_b_proj.weight", H * 256, QA, 3.0f);  // scores of a few units
    lin(p + "self_attn.kv_a_proj_with_mqa.weight", 576, D);
    nrm(p + "self_attn.kv_a_layernorm.weight", 512);
    lin(p + "self_attn.kv_b_proj.weight", H * 448, 512);
    add(p + "self_attn.learnable_sink_param", {int64_t(H)}, 1.0f);
    lin(p + "self_attn.linear_gate.weight", H * 256, D);
    lin(p + "self_attn.o_proj.weight", D, H * 256);
    nrm(p + "self_attn.post_proj_norm.weight", D);
    lin(p + "mlp.gate_proj.weight", I, D);
    lin(p + "mlp.up_proj.weight", I, D);
    lin(p + "mlp.down_proj.weight", D, I);
  }
  nrm("mtp_fusion.enorm.weight", D);
  nrm("mtp_fusion.hnorm.weight", D);
  lin("mtp_fusion.eh_proj.weight", D, 2 * D);
  add("mtp_fusion.gate", {int64_t(D)}, 0.5f);
  nrm("final_norm.weight", D);
  nrm("hidden_correction.hidden_norm.weight", D);
  nrm("hidden_correction.embed_norm.weight", D);
  lin("hidden_correction.gate_proj.weight", D, 2 * D);
  lin("hidden_correction.up_proj.weight", D, 2 * D);
  lin("hidden_correction.down_proj.weight", D, D);
  add("confidence_head.proj.weight", {1, int64_t(D)}, 0.1f);
  add("confidence_head.proj.bias", {1}, 0.1f);
  write_safetensors(dir + "/model-dfly.safetensors", meta, data);
  // the target's tables: embedding rows of unit-ish size, a head with logits of a few units
  std::vector<uint16_t> e(size_t(V) * D), h(size_t(V) * D);
  for (auto& x : e) x = f32_to_bf16(nd(rng));
  for (auto& x : h) x = f32_to_bf16(2.0f * nd(rng) / std::sqrt(float(D)));
  T.emb = quantized(T, e, V, D, 8);
  T.head = quantized(T, h, V, D, 8);
  return dir;
}

void load_target(const std::string& pack, Target& T, uint32_t D) {
  T.ws = std::make_unique<WeightStore>();
  T.ws->open_dir(
      pack,
      [](const std::string& n) { return n.rfind("model.embed_tokens.", 0) == 0 || n.rfind("lm_head.", 0) == 0; },
      [](const std::string& f) {
        return f.rfind("model-experts", 0) != 0 && f.rfind("model-engram", 0) != 0 && f != "model-vit.safetensors" &&
               f != "model-dfly.safetensors";
      });
  auto q = [&](const std::string& p) {
    QLinear L;
    L.w = &T.ws->get(p + ".weight");
    L.s = &T.ws->get(p + ".scales");
    L.b = &T.ws->get(p + ".biases");
    L.N = uint32_t(L.w->shape[0]);
    L.K = D;
    L.bits = int(L.w->shape[1] * 32 / D);
    L.gs = int(D / L.s->shape[1]);
    return L;
  };
  T.emb = q("model.embed_tokens");
  T.head = q("lm_head");
}

// random history rows (bf16 values stored half) in each layer's cache, rows < a; returns them as doubles
std::vector<Vec> fill_history(std::vector<hy4::LatentCache>& caches, int a, std::mt19937& rng) {
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<Vec> hist;
  for (auto& c : caches) {
    Vec h(size_t(a) * 576);
    _Float16* rows = c.rows.as<_Float16>();
    for (size_t i = 0; i < h.size(); ++i) {
      h[i] = rhalf(rbf(nd(rng)));
      rows[i] = _Float16(float(h[i]));
    }
    // rows a .. a + 5 hold junk the draft must overwrite
    for (size_t i = size_t(a) * 576; i < size_t(a + 6) * 576; ++i) rows[i] = _Float16(7.0f);
    hist.push_back(std::move(h));
  }
  return hist;
}

struct Err {
  double abs = 0, scale = 0;
  double rel() const { return scale > 0 ? abs / scale : (abs > 0 ? INFINITY : 0); }
  void add(double gpu, double ref) {
    if (!std::isfinite(gpu)) abs = INFINITY;
    abs = std::max(abs, std::fabs(gpu - ref));
    scale = std::max(scale, std::fabs(ref));
  }
};

void run_compare(const std::string& dir, const Hy4DflyConfig& cfg, Target& T, int a, std::mt19937& rng) {
  Gpu& g = Gpu::get();
  Hy4Dfly d(dir, cfg, T.emb, T.head);
  g.make_resident(d.buffers());
  std::vector<hy4::LatentCache> caches;
  for (size_t i = 0; i < cfg.kv_layers.size(); ++i) caches.push_back(hy4::alloc_latent(uint32_t(a + 6), "test_latent"));
  const std::vector<Vec> hist = fill_history(caches, a, rng);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  Hy4Dfly::In in;
  in.h = abuf(size_t(cfg.D) * 4, "h");
  Vec h(cfg.D);
  for (uint32_t i = 0; i < cfg.D; ++i) in.h.as<float>()[i] = float(h[i] = 1.5 * nd(rng));
  in.t0 = abuf(4, "t0");
  const uint32_t t0 = uint32_t(rng() % d.vocab());
  in.t0.as<uint32_t>()[0] = t0;
  in.a = a;
  for (auto& c : caches) in.caches.push_back(&c);
  Buffer out = abuf(5 * 4, "out");
  auto e = g.begin();
  d.draft(*e, in, out);
  e->commit_and_wait();
  std::vector<uint32_t> ids(out.as<uint32_t>(), out.as<uint32_t>() + 5);
  const RefOut r = reference(Mapped(dir + "/model-dfly.safetensors"), cfg, T.emb, T.head, h, t0, a, hist, ids);
  Err eb, ez, el;
  for (size_t li = 0; li < caches.size(); ++li) {
    const _Float16* rows = caches[li].rows.as<_Float16>() + size_t(a) * 576;
    for (size_t i = 0; i < 6 * 576; ++i) eb.add(double(rows[i]), r.blk[li][i]);
  }
  for (int j = 0; j < 6; ++j)
    for (uint32_t i = 0; i < cfg.D; ++i) ez.add(d.z().as<float>()[size_t(j) * cfg.D + i], r.z[size_t(j)][i]);
  int bad_ids = 0, ties = 0;
  for (int k = 0; k < 5; ++k) {
    const float* lg = d.logits().as<float>() + size_t(k) * d.vocab();
    uint32_t best = 0;
    for (uint32_t n = 0; n < d.vocab(); ++n) {
      el.add(lg[n], r.logits[size_t(k)][n]);
      if (r.logits[size_t(k)][n] > r.logits[size_t(k)][best]) best = n;
    }
    if (ids[size_t(k)] != best) {
      const double gap = r.logits[size_t(k)][best] - r.logits[size_t(k)][ids[size_t(k)]];
      if (gap < 1e-2 * el.scale) ++ties;
      else ++bad_ids;
    }
    // the GPU's own argmax of its logits
    uint32_t gb = 0;
    for (uint32_t n = 0; n < d.vocab(); ++n)
      if (lg[n] > lg[gb]) gb = n;
    CHECK(gb == ids[size_t(k)], "a = %d step %d: argmax kernel %u, logits' argmax %u", a, k + 1, ids[size_t(k)], gb);
  }
  std::printf("a = %4d: block rows rel %.2e, z rel %.2e (max |z| %.2f), logits rel %.2e (max %.2f); ids %u %u %u %u %u, "
              "%d differ from the reference's argmax (%d near ties)\n",
              a, eb.rel(), ez.rel(), ez.scale, el.rel(), el.scale, ids[0], ids[1], ids[2], ids[3], ids[4], bad_ids + ties,
              ties);
  CHECK(eb.rel() < 1e-2, "a = %d: block rows rel %.3e", a, eb.rel());
  CHECK(ez.rel() < 1e-2, "a = %d: z rel %.3e", a, ez.rel());
  CHECK(el.rel() < 1e-2, "a = %d: logits rel %.3e", a, el.rel());
  CHECK(bad_ids == 0, "a = %d: %d drafted ids are not the reference's argmax", a, bad_ids);
  for (auto& c : caches) hy4::free_latent(c);
}

void run_time(const std::string& pack, const Hy4DflyConfig& cfg, Target& T) {
  Gpu& g = Gpu::get();
  Hy4Dfly d(pack, cfg, T.emb, T.head);
  g.make_resident(d.buffers());
  std::vector<Buffer> tb = {T.emb.w->buf, T.emb.s->buf, T.emb.b->buf, T.head.w->buf, T.head.s->buf, T.head.b->buf};
  g.make_resident(tb);
  std::mt19937 rng(7);
  for (int a : {64, 512, 2042}) {
    std::vector<hy4::LatentCache> caches;
    for (size_t i = 0; i < cfg.kv_layers.size(); ++i) caches.push_back(hy4::alloc_latent(uint32_t(a + 6), "latent"));
    fill_history(caches, a, rng);
    for (auto& c : caches) g.make_resident({c.rows});
    Hy4Dfly::In in;
    in.h = abuf(size_t(cfg.D) * 4, "h");
    for (uint32_t i = 0; i < cfg.D; ++i) in.h.as<float>()[i] = float(rng() % 1000) / 500.0f - 1.0f;
    in.t0 = abuf(4, "t0");
    in.t0.as<uint32_t>()[0] = 1000;
    in.a = a;
    for (auto& c : caches) in.caches.push_back(&c);
    Buffer out = abuf(20, "out");
    std::vector<double> ms;
    for (int it = 0; it < 22; ++it) {
      auto e = g.begin();
      d.draft(*e, in, out);
      e->commit_and_wait();
      if (it >= 2) ms.push_back(e->gpu_ms());
    }
    std::sort(ms.begin(), ms.end());
    std::printf("time a = %4d: median %.3f ms, min %.3f ms a draft cycle over %zu runs (ids %u %u %u %u %u)\n", a,
                ms[ms.size() / 2], ms[0], ms.size(), out.as<uint32_t>()[0], out.as<uint32_t>()[1], out.as<uint32_t>()[2],
                out.as<uint32_t>()[3], out.as<uint32_t>()[4]);
    for (auto& c : caches) hy4::free_latent(c);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "check";
  std::string pack, cfg_path;
  int a_arg = -1;
  for (int i = 2; i < argc; ++i) {
    const std::string s = argv[i];
    if (s == "--a" && i + 1 < argc) a_arg = std::atoi(argv[++i]);
    else if (s == "--config" && i + 1 < argc) cfg_path = argv[++i];
    else if (pack.empty() && s[0] != '-') pack = s;
    else {
      std::fprintf(stderr, "hy4_dfly_check: unknown argument %s (see the top of tools/hy4_dfly_check.cpp)\n", s.c_str());
      return 2;
    }
  }
  Gpu& g = Gpu::get();
  std::printf("%s, %d GPU cores\n", g.name().c_str(), g.core_count());
  std::mt19937 rng(1234);
  if (mode == "check") {
    Hy4DflyConfig c;
    c.kv_layers = {25, 29};
    c.mask_token = 1000;
    c.D = 256;
    c.H = 16;
    Target T;
    const std::string dir = make_synthetic(c, T, 1024, 128, 512, rng);
    const std::vector<int> as = a_arg >= 0 ? std::vector<int>{a_arg} : std::vector<int>{0, 37, 301};
    for (int a : as) run_compare(dir, c, T, a, rng);
    unlink((dir + "/model-dfly.safetensors").c_str());
    rmdir(dir.c_str());
  } else if (mode == "pack" || mode == "time") {
    if (pack.empty()) fatal("%s needs the pack folder", mode.c_str());
    const Hy4DflyConfig c = Hy4DflyConfig::from_json(cfg_path.empty() ? pack + "/config.json" : cfg_path);
    Target T;
    load_target(pack, T, c.D);
    if (mode == "pack") run_compare(pack, c, T, a_arg >= 0 ? a_arg : 300, rng);
    else run_time(pack, c, T);
  } else {
    std::fprintf(stderr, "hy4_dfly_check: mode check, pack or time\n");
    return 2;
  }
  std::printf("%s\n", g_fail ? "FAILED" : "all checks passed");
  return g_fail ? 1 : 0;
}
