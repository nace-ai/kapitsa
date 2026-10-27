// The EXL3 routed-expert kernels (src/kernels/exl3moe.metal, src/model/moe_exl3.h) against the CPU reference:
//  1. with Sushi's fixtures (its src/exl3/fixtures: 128 x 128 matrices made by its converter, each with `inner`, the
//     weights exllamav3 decodes): exl3::decode_tile of every tile of the MCG ones equal to `inner`, bit for bit;
//  2. random experts (D 512, I 256, 16 experts, top-4; random trellis words, suh and svh at the magnitudes real packs
//     have) at rates n 32, 40, 48, 64 and windows 16 and 15: exl3::enc_rows at 1, 3 and 8 rows and exl3::enc_prefill
//     on moe_bucket3's tiles at 37 and 300 tokens, against exl3::project and the SwiGLU in fp32;
//  3. with a pack folder: the same on layer 0's experts of a Sushi pack (Flash-Next shapes, its own window).
// Error: max |got - ref| / max |ref| over each output; the bar is 3e-3 (the f16 roundings of transformed values reach
// about 1.1e-3). A negative control corrupts one tile of every expert's gate (the last row and column of tiles) and
// must fail the bar.
//   exl3_check [--fixtures DIR] [--pack DIR]       (needs a GPU; run with MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1)
#include "loader/safetensors.h"
#include "metal/gpu.h"
#include "model/moe_exl3.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace ni;
using json = nlohmann::json;

namespace {

int g_fail = 0;

float h2f(uint16_t h) {
  _Float16 v;
  memcpy(&v, &h, 2);
  return float(v);
}

uint16_t f2h(float f) {
  const _Float16 v = _Float16(f);
  uint16_t h;
  memcpy(&h, &v, 2);
  return h;
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

Buffer upload(const void* p, size_t bytes) {
  Buffer b = Gpu::get().alloc(bytes + 16);
  memcpy(b.ptr(), p, bytes);
  return b;
}

// ---- 1. Sushi's fixtures ----
void fixtures(const std::string& dir) {
  for (const char* f : {"exl3_k2p5_mcg_linear", "exl3_k3_mcg_linear", "exl3_k2p5_mcg_w12_linear"}) {
    const std::string path = dir + "/" + f + ".safetensors";
    std::ifstream in(path, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (raw.size() < 8) {
      printf("FAILED: %s missing\n", path.c_str());
      ++g_fail;
      continue;
    }
    uint64_t n = 0;
    memcpy(&n, raw.data(), 8);
    const json h = json::parse(std::string(raw.data() + 8, n));
    auto at = [&](const char* name) { return reinterpret_cast<const uint16_t*>(raw.data() + 8 + n + h[name]["data_offsets"][0].get<size_t>()); };
    const uint32_t hw = uint32_t(h["trellis"]["shape"][2].get<int>());
    const uint32_t window = uint32_t(std::stoi(h["__metadata__"].value("window", std::string("16"))));
    const uint16_t *tr = at("trellis"), *inner = at("inner");
    int bad = 0;
    float w[256];
    for (uint32_t tk = 0; tk < 8; ++tk)
      for (uint32_t tn = 0; tn < 8; ++tn) {
        exl3::decode_tile(tr + (tk * 8 + tn) * hw, hw, window, w);
        for (uint32_t r = 0; r < 16; ++r)
          for (uint32_t c = 0; c < 16; ++c) bad += f2h(w[r * 16 + c]) != inner[(tk * 16 + r) * 128 + tn * 16 + c];
      }
    printf("%s %s (n %u, window %u): %d of 16384 weights differ from exllamav3's\n", bad ? "FAILED" : "ok", f, hw,
           window, bad);
    g_fail += bad != 0;
  }
}

// ---- 2 / 3. the kernels ----
struct Bank {  // a projection's experts on the host
  std::vector<uint16_t> t, suh, svh;
  uint32_t in = 0, out = 0, n = 0;
};

struct Model {
  exl3::Shape sh;
  Bank g, u, d;
};

Bank random_bank(std::mt19937& rng, uint32_t E, uint32_t in, uint32_t out, uint32_t n) {
  Bank b{{}, {}, {}, in, out, n};
  b.t.resize(size_t(E) * in / 16 * out / 16 * n);
  for (auto& v : b.t) v = uint16_t(rng());
  std::uniform_real_distribution<float> su(0.05f, 0.4f), sv(0.1f, 2.0f);
  for (size_t i = 0; i < size_t(E) * in; ++i) b.suh.push_back(f2h((rng() & 1 ? -1.0f : 1.0f) * su(rng)));
  for (size_t i = 0; i < size_t(E) * out; ++i) b.svh.push_back(f2h(sv(rng)));
  return b;
}

exl3::Proj device(const Bank& b) {
  return exl3::Proj{upload(b.t.data(), b.t.size() * 2), upload(b.suh.data(), b.suh.size() * 2),
                    upload(b.svh.data(), b.svh.size() * 2), b.n};
}

void project(const Bank& b, uint32_t e, uint32_t window, const float* x, float* y) {
  exl3::project(b.t.data() + size_t(e) * b.in / 16 * b.out / 16 * b.n, b.suh.data() + size_t(e) * b.in,
                b.svh.data() + size_t(e) * b.out, b.in, b.out, b.n, window, x, y);
}

// expert e's output for input x (D): down(silu(gate x) * up x)
std::vector<float> expert(const Model& m, uint32_t e, const float* x) {
  const uint32_t D = m.sh.D, I = m.sh.I;
  std::vector<float> g(I), u(I), h(I), y(D);
  project(m.g, e, m.sh.window, x, g.data());
  project(m.u, e, m.sh.window, x, u.data());
  for (uint32_t i = 0; i < I; ++i) h[i] = silu(g[i]) * u[i];
  project(m.d, e, m.sh.window, h.data(), y.data());
  return y;
}

float rel_err(const float* got, const std::vector<double>& ref) {
  double num = 0, den = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    num = std::max(num, std::fabs(double(got[i]) - ref[i]));
    den = std::max(den, std::fabs(ref[i]));
    if (!std::isfinite(got[i])) return INFINITY;
  }
  return float(num / std::max(den, 1e-30));
}

constexpr float kBar = 3e-3f;

void report(const std::string& what, float err) {
  const bool ok = err < kBar;
  printf("%s %s: max error %.2e of the largest output\n", ok ? "ok" : "FAILED", what.c_str(), err);
  g_fail += !ok;
}

float check_rows(const Model& m, const exl3::Experts& X, std::mt19937& rng, int R, const std::string& tag,
                 bool report_it = true) {
  const exl3::Shape& sh = m.sh;
  const uint32_t D = sh.D, I = sh.I, k = sh.k, S = uint32_t(R) * k;
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> x(size_t(R) * D), ys(size_t(R) * D), y0(size_t(R) * D), wts(size_t(R) * (k + 1));
  std::vector<uint32_t> idx(S);
  for (auto& v : x) v = nd(rng);
  for (auto& v : ys) v = nd(rng);
  for (auto& v : y0) v = nd(rng);
  for (auto& v : wts) v = std::uniform_real_distribution<float>(0.0f, 0.3f)(rng);
  for (int r = 0; r < R; ++r) {
    std::vector<uint32_t> ids(sh.E);
    for (uint32_t i = 0; i < sh.E; ++i) ids[i] = i;
    std::shuffle(ids.begin(), ids.end(), rng);
    for (uint32_t j = 0; j < k; ++j) idx[r * k + j] = ids[j];
  }
  Gpu& g = Gpu::get();
  Buffer bx = upload(x.data(), x.size() * 4), bi = upload(idx.data(), idx.size() * 4),
         bw = upload(wts.data(), wts.size() * 4), bys = upload(ys.data(), ys.size() * 4),
         by = upload(y0.data(), y0.size() * 4);
  Buffer hg = g.alloc(size_t(S) * I * 4), hu = g.alloc(size_t(S) * I * 4), md = g.alloc(size_t(S) * I * 2),
         yd = g.alloc(size_t(S) * D * 4);
  auto e = g.begin();
  exl3::enc_rows(*e, X, sh, R, bx, bi, bw, bys, hg, hu, md, yd, by);
  e->commit_and_wait();
  // the routed sum alone (y less what it started at and the shared term), so the error is measured on what the
  // experts contribute rather than diluted by y0 and ys
  std::vector<double> ref(size_t(R) * D, 0.0);
  std::vector<float> got(size_t(R) * D);
  for (int r = 0; r < R; ++r) {
    for (uint32_t c = 0; c < D; ++c)
      got[r * D + c] = float(double(by.as<float>()[r * D + c]) - y0[r * D + c] - double(wts[r * (k + 1) + k]) * ys[r * D + c]);
    for (uint32_t j = 0; j < k; ++j) {
      const std::vector<float> o = expert(m, idx[r * k + j], &x[size_t(r) * D]);
      for (uint32_t c = 0; c < D; ++c) ref[r * D + c] += double(wts[r * (k + 1) + j]) * o[c];
    }
  }
  const float err = rel_err(got.data(), ref);
  if (report_it) report(tag + " rows R=" + std::to_string(R), err);
  return err;
}

void check_prefill(const Model& m, const exl3::Experts& X, std::mt19937& rng, int T, const std::string& tag) {
  const exl3::Shape& sh = m.sh;
  const uint32_t D = sh.D, I = sh.I, k = sh.k, P = uint32_t(T) * k;
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<uint16_t> xh(size_t(T) * D);
  std::vector<float> xf(size_t(T) * D);
  for (size_t i = 0; i < xh.size(); ++i) xh[i] = f2h(nd(rng)), xf[i] = h2f(xh[i]);
  std::vector<uint32_t> idx(P);
  // a skewed routing: a few experts take most rows (several 32-row tiles), others one or none
  for (int t = 0; t < T; ++t) {
    std::vector<uint32_t> ids(sh.E);
    for (uint32_t i = 0; i < sh.E; ++i) ids[i] = i;
    std::shuffle(ids.begin() + (rng() % 3 == 0 ? 0 : k / 2), ids.end(), rng);
    for (uint32_t j = 0; j < k; ++j) idx[t * k + j] = ids[j];
  }
  Gpu& g = Gpu::get();
  const uint32_t bm = exl3::kTileRows, max_tiles = (P + bm - 1) / bm + sh.E;
  Buffer bx = upload(xh.data(), xh.size() * 2), bi = upload(idx.data(), idx.size() * 4);
  Buffer yrows = g.alloc(P * 4), xrows = g.alloc(P * 4), tiles = g.alloc((P + sh.E + 1) * 12), ntiles = g.alloc(16);
  Buffer hg = g.alloc(size_t(P) * I * 4), hu = g.alloc(size_t(P) * I * 4), md = g.alloc(size_t(P) * I * 2),
         yd = g.alloc(size_t(P) * D * 4);
  auto e = g.begin();
  const struct {
    uint32_t P, k, E, BM, S;
  } bp{P, k, sh.E, bm, 0};
  e->use(g.pipeline("moe_bucket3"));
  e->buf(0, bi); e->buf(1, yrows); e->buf(2, xrows); e->buf(3, tiles); e->buf(4, ntiles);
  e->val(5, bp);
  e->dispatch(1, 1, 1, 1024);
  exl3::enc_prefill(*e, X, sh, P, max_tiles, bx, tiles, ntiles, xrows, yrows, bi, hg, hu, md, yd);
  e->commit_and_wait();
  float worst = 0.0f;
  for (uint32_t q = 0; q < P; ++q) {
    const std::vector<float> o = expert(m, idx[q], &xf[size_t(q / k) * D]);
    worst = std::max(worst, rel_err(yd.as<float>() + size_t(q) * D, std::vector<double>(o.begin(), o.end())));
  }
  report(tag + " prefill T=" + std::to_string(T) + " (" + std::to_string(ntiles.as<uint32_t>()[0]) + " tiles)", worst);
}

void check_model(const Model& m, std::mt19937& rng, const std::string& tag, const std::vector<int>& rows,
                 const std::vector<int>& toks) {
  std::string why;
  const exl3::Experts X{device(m.g), device(m.u), device(m.d)};
  if (!exl3::shapes_ok(m.sh, X, &why)) {
    printf("FAILED %s: %s\n", tag.c_str(), why.c_str());
    ++g_fail;
    return;
  }
  for (int R : rows) check_rows(m, X, rng, R, tag);
  for (int T : toks) check_prefill(m, X, rng, T, tag);
  // the negative control: the device copy of one tile of every expert's gate flipped (the last tile row and column,
  // where tail bugs would sit), the host reference unchanged
  const uint32_t ot = m.g.out / 16, kt = m.g.in / 16, per = kt * ot * m.g.n;
  for (uint32_t e = 0; e < m.sh.E; ++e) {
    uint16_t* t = X.g.t.as<uint16_t>() + size_t(e) * per + size_t((kt - 1) * ot + ot - 1) * m.g.n;
    for (uint32_t i = 0; i < m.g.n; ++i) t[i] ^= 0x5A5Au;
  }
  const float err = check_rows(m, X, rng, 1, tag, false);
  const bool caught = err >= kBar;
  printf("%s %s: one corrupted tile %s (max error %.2e)\n", caught ? "ok" : "FAILED", tag.c_str(),
         caught ? "fails the bar" : "passes the bar: the check cannot see a wrong tile", err);
  g_fail += !caught;
}

// layer 0's experts of a Sushi pack (its config.json's window), the first 16 experts
void pack(const std::string& dir, std::mt19937& rng) {
  const json cfg = json::parse(std::ifstream(dir + "/config.json"));
  const json t = cfg.contains("text_config") ? cfg["text_config"] : cfg;
  const uint32_t window = cfg["expert_quant"].value("window", 16u), k = uint32_t(t["num_experts_per_tok"].get<int>());
  WeightStore ws;
  const std::string p = "language_model.model.layers.0.mlp.switch_mlp.";
  ws.open_dir(dir, [&](const std::string& n) { return n.rfind(p, 0) == 0; },
              [](const std::string& f) { return f.find("-L00-") != std::string::npos; });
  Model m;
  const uint32_t E = 16;
  auto bank = [&](const char* proj) {
    const Tensor& tr = ws.get(p + proj + ".trellis");
    const Tensor &su = ws.get(p + proj + ".suh"), &sv = ws.get(p + proj + ".svh");
    Bank b;
    b.in = uint32_t(tr.shape[1] * 16), b.out = uint32_t(tr.shape[2] * 16), b.n = uint32_t(tr.shape[3]);
    const size_t per = size_t(b.in) / 16 * b.out / 16 * b.n;
    b.t.assign(tr.buf.as<uint16_t>(), tr.buf.as<uint16_t>() + E * per);
    b.suh.assign(su.buf.as<uint16_t>(), su.buf.as<uint16_t>() + E * b.in);
    b.svh.assign(sv.buf.as<uint16_t>(), sv.buf.as<uint16_t>() + E * b.out);
    return b;
  };
  m.g = bank("gate_proj"), m.u = bank("up_proj"), m.d = bank("down_proj");
  m.sh = exl3::Shape{m.g.in, m.g.out, std::min(k, 4u), E, window};
  check_model(m, rng, "pack layer 0 (n " + std::to_string(m.g.n) + ", window " + std::to_string(window) + ")", {1, 4},
              {40});
}

}  // namespace

int main(int argc, char** argv) {
  std::string fix, pk;
  for (int i = 1; i + 1 < argc; ++i) {
    if (!strcmp(argv[i], "--fixtures")) fix = argv[++i];
    else if (!strcmp(argv[i], "--pack")) pk = argv[++i];
  }
  if (!fix.empty()) fixtures(fix);
  std::mt19937 rng(7);
  for (uint32_t n : {32u, 40u, 48u, 64u})
    for (uint32_t window : {16u, 15u}) {
      if (n != 32 && window != 16) continue;
      Model m;
      m.sh = exl3::Shape{512, 256, 4, 16, window};
      m.g = random_bank(rng, 16, 512, 256, n);
      m.u = random_bank(rng, 16, 512, 256, n);
      m.d = random_bank(rng, 16, 256, 512, n);
      check_model(m, rng, "random n " + std::to_string(n) + " window " + std::to_string(window), {1, 3, 8}, {37, 300});
    }
  if (!pk.empty()) pack(pk, rng);
  printf(g_fail ? "%d FAILED\n" : "all passed\n", g_fail);
  return g_fail ? 1 : 0;
}
