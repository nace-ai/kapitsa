// Check the lossless Q4/coefficient image and its GPU kernel against FP64 and
// the original kernel. Invalid output, stale inputs and boundary writes fail
// the process. GPU modes hold the project lock; check-cpu-codec uses no GPU.
#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/kparams.h"
#include "model/affine_coeff_fixed.h"
#include "util/json_file.h"
#include <fstream>
#include "util/bf16.h"
#include "util/log.h"

#include <algorithm>
#include <chrono>
#include <bit>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ni;
constexpr size_t kGuard = 16;
constexpr float kCanary = 12345.625f;
constexpr double kTolerance = 1e-5;
static const char* kBaseline = "qmm8snb_b4_nf2";
static size_t candidate_dispatches = 0, original_dispatches = 0;
static bool real_fixture_check = false;
static size_t real_fixture_index = 0;

struct Arena {
  Gpu& g;
  std::vector<Buffer> owned;
  explicit Arena(Gpu& gpu) : g(gpu) {}
  ~Arena() { for (auto& b : owned) g.release(b); }
  Buffer alloc(size_t bytes, const char* label, bool tracked = true) {
    Buffer b = g.alloc(std::max(bytes, size_t(16)), label, tracked);
    owned.push_back(b);
    return b;
  }
};
struct Output {
  Buffer full, data;
  uint32_t N, ld, R;
  bool add;
  Output(Arena& a, uint32_t n, uint32_t r, bool residual) : N(n), ld(n + 8), R(r), add(residual) {
    full = a.alloc((size_t(8) * ld + 2 * kGuard) * 4, "qmm48h_checked_output");
    data = full.view(kGuard * 4, size_t(8) * ld * 4);
    reset();
  }
  void reset() {
    std::fill_n(full.as<float>(), full.size / 4, kCanary);
    if (add) for (uint32_t t = 0; t < R; ++t) std::fill_n(data.as<float>() + size_t(t) * ld, N, 0.25f);
  }
  void guards() const {
    for (size_t i = 0; i < kGuard; ++i)
      if (full.as<float>()[i] != kCanary || full.as<float>()[kGuard + size_t(8) * ld + i] != kCanary)
        fatal("QMM48H output boundary overwritten");
    for (uint32_t t = 0; t < 8; ++t)
      for (uint32_t n = t < R ? N : 0; n < ld; ++n)
        if (data.as<float>()[size_t(t) * ld + n] != kCanary) fatal("QMM48H inactive row or stride padding overwritten");
  }
};
struct CoeffImage {
  Gpu& gpu;
  Buffer buffer, headers;
  size_t compressed = 0, blocks = 0;
  explicit CoeffImage(Gpu& g) : gpu(g) {}
  ~CoeffImage() { if (buffer) gpu.release(buffer); if (headers) gpu.release(headers); }
};
struct Matrix { Buffer w, s, b; uint32_t N, K; mutable std::shared_ptr<CoeffImage> coeff; mutable bool fixture_loaded = false; };
static double pack_ms = 0;
static size_t pack_bytes = 0, pack_blocks = 0, compressed_blocks = 0;
static void prepare_coeff(Gpu& g, const Matrix& m) {
  if (m.coeff) return;  // object-owned static weights; explicit reset after replacing their coefficients
  const auto begin = std::chrono::steady_clock::now();
  const auto host = pack_affine_packed_coeff(m.w.as<uint32_t>(), m.s.as<uint16_t>(), m.b.as<uint16_t>(), m.N, m.K / 64);
  auto image = std::make_shared<CoeffImage>(g);
  image->buffer = g.alloc(host.words.size() * 4, "qmm_coefficient_delta", false);
  memcpy(image->buffer.ptr(), host.words.data(), host.words.size() * 4);
  image->headers = g.alloc(host.headers.size() * 4, "qmm_packed_headers", false);
  memcpy(image->headers.ptr(), host.headers.data(), host.headers.size() * 4);
  image->blocks = host.blocks; image->compressed = host.compressed_blocks;
  pack_bytes += image->buffer.size + image->headers.size; pack_blocks += image->blocks; compressed_blocks += image->compressed;
  m.coeff = std::move(image);
  pack_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
}
// Additional, pre-frozen actual-coefficient diagnostic. The original random W/X
// and all numerical checks remain unchanged; both timed arms share these bytes.
static void fixture_coefficients(const Matrix& m, size_t copy) {
  const char* path = getenv("NI_QMM_COEFF_FIXTURES");
  if (!path) return;
  auto manifest = json_file(path);
  if (!manifest || manifest->at("version") != 1) fatal("invalid coefficient fixture manifest");
  const auto key = std::to_string(m.N) + "x" + std::to_string(m.K);
  const auto& shapes = manifest->at("shapes");
  if (!shapes.contains(key)) {
    if (key != "98304x2560") fatal("missing real coefficient shape %s", key.c_str());
    if (!copy) printf("COEFFICIENT_FIXTURE shape=%s random_control=1\n", key.c_str());
    return;
  }
  const auto& entries = shapes.at(key);
  const auto& entry = entries.at(copy % entries.size());
  std::ifstream input(manifest->at("source").get<std::string>(), std::ios::binary);
  if (!input) fatal("cannot open frozen coefficient source");
  input.seekg(0, std::ios::end);
  if (uint64_t(input.tellg()) != manifest->at("source_size").get<uint64_t>()) fatal("source size changed");
  size_t row = 0;
  for (const auto& seg : entry) {
    const size_t n = seg.at("N").get<size_t>(), bytes = n * (m.K / 64) * 2;
    if (seg.at("K") != m.K || row + n > m.N || seg.at("coefficient_bytes") != bytes)
      fatal("bad coefficient segment dimensions");
    for (bool bias : {false, true}) {
      input.seekg(seg.at(bias ? "b_offset" : "s_offset").get<std::streamoff>());
      input.read(reinterpret_cast<char*>((bias ? m.b : m.s).as<uint16_t>() + row * (m.K / 64)), bytes);
      if (!input) fatal("short coefficient fixture read");
    }
    row += n;
  }
  if (row != m.N) fatal("coefficient fixture row count mismatch");
  m.coeff.reset();
  if (!copy) printf("COEFFICIENT_FIXTURE shape=%s entries=%zu random_control=0\n", key.c_str(), entries.size());
}
static Matrix matrix(Arena& a, uint32_t N, uint32_t K, std::mt19937& rng, bool exact) {
  Matrix m{a.alloc(size_t(N) * K / 2, "qmm48h_w", false), a.alloc(size_t(N) * K / 64 * 2, "qmm48h_s", false),
           a.alloc(size_t(N) * K / 64 * 2, "qmm48h_b", false), N, K};
  for (size_t i = 0; i < size_t(N) * K / 8; ++i) m.w.as<uint32_t>()[i] = rng();
  for (uint32_t n = 0; n < N; ++n) {
    m.w.as<uint32_t>()[size_t(n) * K / 8] = 0x76543210u;
    m.w.as<uint32_t>()[size_t(n) * K / 8 + 1] = 0xfedcba98u;
  }
  std::normal_distribution<float> nd(0, 0.02f);
  for (size_t i = 0; i < size_t(N) * K / 64; ++i) {
    m.s.as<uint16_t>()[i] = f32_to_bf16(exact ? 0.0625f : 0.001f + float(rng() % 200) * 0.0001f);
    m.b.as<uint16_t>()[i] = f32_to_bf16(exact ? -0.5f : nd(rng));
  }
  return m;
}
struct Input {
  Buffer x, norm, table, sums;
  uint32_t K, R, ld;
  bool normalize;
  Input(Arena& a, uint32_t k, uint32_t r, uint32_t seed, int pattern = 0, bool norm_on = false)
      : K(k), R(r), ld(k + 8), normalize(norm_on) {
    x = a.alloc(size_t(8) * ld * 4, "qmm48h_input");
    norm = a.alloc(size_t(K) * 2, "qmm48h_norm");
    table = a.alloc(size_t(K) * 8 * 2, "qmm48h_x8_table");
    sums = a.alloc(size_t(K) / 64 * 8 * 4, "qmm48h_x8_sums");
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0, 1);
    std::fill_n(x.as<float>(), size_t(8) * ld, kCanary);
    for (uint32_t t = 0; t < R; ++t) for (uint32_t i = 0; i < K; ++i) {
      float v = pattern == 1 ? float(int(rng() % 17) - 8) : nd(rng);
      if (pattern == 2) v = 0;
      if (pattern == 3) v = (i & 1 ? -1.0f : 1.0f) * (i % 7 == 0 ? 1024.0f : 0x1p-20f);
      if (pattern == 4) v = (i & 1 ? -1.0f : 1.0f) * (i % 7 == 0 ? 65504.0f : 0x1p-24f);
      x.as<float>()[size_t(t) * ld + i] = v;
    }
    for (uint32_t i = 0; i < K; ++i) norm.as<uint16_t>()[i] = f32_to_bf16(0.75f + float(i % 5) * 0.125f);
  }
  std::vector<float> rounded() const {
    std::vector<float> out(size_t(8) * K);
    for (uint32_t g = 0; g < K / 64; ++g) for (uint32_t lane = 0; lane < 32; ++lane) {
      uint32_t q = lane >> 2, fm = (q & 4) | ((lane >> 1) & 3), fn = ((q & 2) << 1) | ((lane & 1) << 1);
      for (uint32_t j = 0; j < 16; ++j) {
        uint32_t f = j / 2, k = 16 * (fm / 2) + 8 * (f / 4) + f % 4 + 4 * (fm % 2);
        out[size_t(fn + j % 2) * K + g * 64 + k] = float(table.as<__fp16>()[(g * 32 + lane) * 16 + j]);
      }
    }
    return out;
  }
};
static void prep(Gpu& g, Encoder& e, const Input& x) {
  const bool v2 = x.K % 256 == 0;
  kp::X8Prep p{x.K, x.R, x.ld, x.normalize ? 1u : 0u, 0, 1e-6f, x.K, 0, 1, 0};
  e.use(g.pipeline(v2 ? "x8_prep2" : "x8_prep"));
  e.buf(0, x.x); e.buf(1, x.norm); e.buf(2, x.table); e.buf(3, x.sums); e.buf(4, x.x); e.val(5, p);
  e.buf(6, x.table); e.buf(7, x.sums); e.buf(8, x.table);
  e.dispatch(v2 ? x.K / 256 : 8, 1, 1, 256);
}
struct Plan { uint32_t cs, ks; };
static Plan default_plan(Gpu& g, uint32_t N, uint32_t K) {
  const uint32_t G = K / 64;
  uint32_t ks = G >= 16 ? 4 : G >= 8 ? 2 : 1;
  if (ks == 4 && G >= 32 && ((N + 15) / 16) * 4 < uint32_t(64 * g.core_count())) ks = 8;
  if (G >= 64) ks = 8;  // M4's default V8Knobs::kplan=2
  return {8 / ks, ks};
}
static void compute(Gpu& g, Encoder& e, const std::string& kernel, const Input& x,
                    const std::vector<Matrix>& m, const std::vector<Output>& y, Plan pl) {
  if (m.empty() || m.size() > 4 || pl.cs < 1 || pl.ks < 1 || pl.cs * pl.ks > 8) fatal("invalid QMM48H dispatch plan");
  if (real_fixture_check) for (const auto& matrix : m) {
    if (!matrix.fixture_loaded) {
      fixture_coefficients(matrix, real_fixture_index++);
      matrix.fixture_loaded = true;
    }
  }
  kp::Qmm8 p{};
  p.K = x.K; p.R = x.R; p.nseg = uint32_t(m.size()); p.ks = pl.ks;
  // The production dispatcher only pairs coefficients when every row has an even group count.
  // With an odd count, the next row's scale/bias pairs would not be four-byte aligned.
  const uint32_t groups = x.K / 64;
  const bool paired_slices = !(groups & 1u) && !(((groups + pl.ks - 1) / pl.ks) & 1u);
  const std::string selected = kernel == "qmm8pc_b4_nf1" && !paired_slices ? kBaseline : kernel;
  const uint32_t selected_nf = selected.ends_with("_nf1") ? 1u : selected.ends_with("_nf4") ? 4u : 2u;
  const std::string actual = (x.K / 64) % 2 && selected.find("snb") != std::string::npos
      ? "qmm8sn_b4_nf" + std::to_string(selected_nf) : selected;
  const uint32_t nf = actual.ends_with("_nf1") ? 1u : actual.ends_with("_nf4") ? 4u : 2u;
  const uint32_t tile_n = 8 * nf;
  if (actual == "qmm8pc_b4_nf1") ++candidate_dispatches;
  else ++original_dispatches;
  e.use(g.pipeline(actual));
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t j = std::min(i, p.nseg - 1);
    if (i < p.nseg) {
      p.N[i] = m[i].N; p.y_ld[i] = y[i].ld; p.add[i] = y[i].add ? 1u : 0u;
      p.tg_begin[i + 1] = p.tg_begin[i] + (m[i].N + tile_n * pl.cs - 1) / (tile_n * pl.cs);
    }
    e.buf(i * 4, m[j].w); e.buf(i * 4 + 1, m[j].s); e.buf(i * 4 + 2, m[j].b); e.buf(i * 4 + 3, y[j].data);
    if (actual == "qmm8pc_b4_nf1") {
      prepare_coeff(g, m[j]); e.buf(i * 4, m[j].coeff->buffer); e.buf(21 + i, m[j].coeff->headers);
    }
  }
  e.buf(16, x.table); e.buf(17, x.sums); e.val(18, p);
  e.buf(19, x.table); e.buf(20, x.sums);
  e.dispatch(p.tg_begin[p.nseg], 1, 1, 32 * pl.cs * pl.ks);
}
static double reference(const Matrix& m, const float* x, uint32_t n, bool add) {
  double out = add ? 0.25 : 0;
  for (uint32_t k = 0; k < m.K; ++k) {
    uint32_t q = (m.w.as<uint32_t>()[size_t(n) * m.K / 8 + k / 8] >> (4 * (k % 8))) & 15;
    size_t i = size_t(n) * (m.K / 64) + k / 64;
    out += (double(bf16_to_f32(m.s.as<uint16_t>()[i])) * q + bf16_to_f32(m.b.as<uint16_t>()[i])) * x[k];
  }
  return out;
}

struct Checks { size_t cases = 0, outputs = 0, bitwise_outputs = 0; double cpu_error = 0, gpu_error = 0; };
static void check_case(Gpu& g, Checks& count, uint32_t N, uint32_t K, uint32_t R, uint32_t ks,
                       uint32_t segments, uint32_t seed, int pattern, bool norm, const std::string& kernel) {
  Arena arena(g);
  Input x(arena, K, R, seed, pattern, norm);
  std::mt19937 rng(seed);
  std::vector<Matrix> matrices;
  std::vector<Output> base, candidate;
  const bool exact = pattern == 1 && !norm;
  for (uint32_t i = 0; i < segments; ++i) {
    matrices.push_back(matrix(arena, N + 3 * i, K, rng, exact));
    base.emplace_back(arena, N + 3 * i, R, (seed + i) % 2 != 0);
    candidate.emplace_back(arena, N + 3 * i, R, (seed + i) % 2 != 0);
  }
  const Plan pl{std::min(2u, 8u / ks), ks};
  for (int replay = 0; replay < 2; ++replay) {
    if (replay) {
      for (uint32_t t = 0; t < R; ++t) for (uint32_t k = 0; k < K; ++k)
        x.x.as<float>()[size_t(t) * x.ld + k] = -x.x.as<float>()[size_t(t) * x.ld + k] + 0.125f;
      for (auto& out : base) out.reset();
      for (auto& out : candidate) out.reset();
    }
    auto e = g.begin();
    prep(g, *e, x);
    compute(g, *e, kBaseline, x, matrices, base, pl);
    compute(g, *e, kernel, x, matrices, candidate, pl);
    e->commit_and_wait();
    const auto hx = x.rounded();
    if (!x.normalize) for (uint32_t t = 0; t < 8; ++t) for (uint32_t k = 0; k < K; ++k) {
      const float want = t < R ? float(__fp16(x.x.as<float>()[size_t(t) * x.ld + k])) : 0.0f;
      if (hx[size_t(t) * K + k] != want) fatal("input half table stale or misindexed");
    }
    for (size_t i = 0; i < matrices.size(); ++i) {
      const Matrix& m = matrices[i];
      base[i].guards(); candidate[i].guards();
      double sq = 0;
      for (uint32_t t = 0; t < R; ++t) for (uint32_t n = 0; n < m.N; ++n) {
        float v = base[i].data.as<float>()[size_t(t) * base[i].ld + n];
        if (!std::isfinite(v)) fatal("baseline produced non-finite output");
        sq += double(v) * v;
      }
      const double rms = std::sqrt(sq / (size_t(R) * m.N));
      for (uint32_t t = 0; t < R; ++t) for (uint32_t n = 0; n < m.N; ++n) {
        float ref = base[i].data.as<float>()[size_t(t) * base[i].ld + n];
        float got = candidate[i].data.as<float>()[size_t(t) * candidate[i].ld + n];
        double err = std::fabs(double(got) - ref) / (std::fabs(ref) + rms + 1e-12);
        if (!std::isfinite(got) || err > kTolerance)
          fatal("GPU check K=%u N=%u R=%u ks=%u t=%u n=%u error=%.9g got=%.9g ref=%.9g", K, m.N, R, ks, t, n, err, got, ref);
        count.gpu_error = std::max(count.gpu_error, err);
        count.bitwise_outputs += std::bit_cast<uint32_t>(got) == std::bit_cast<uint32_t>(ref);
        ++count.outputs;
      }
      std::vector<uint32_t> channels;
      if (m.N <= 1024) { channels.resize(m.N); std::iota(channels.begin(), channels.end(), 0u); }
      else for (uint32_t j = 0; j < 1024; ++j) channels.push_back(uint32_t(uint64_t(j) * (m.N - 1) / 1023));
      std::vector<double> refs;
      sq = 0;
      for (uint32_t t = 0; t < R; ++t) for (uint32_t n : channels) {
        double v = reference(m, hx.data() + size_t(t) * K, n, candidate[i].add);
        refs.push_back(v); sq += v * v;
      }
      const double rrms = std::sqrt(sq / refs.size());
      size_t ri = 0;
      for (uint32_t t = 0; t < R; ++t) for (uint32_t n : channels) {
        double want = refs[ri++];
        for (const Output* out : {&base[i], &candidate[i]}) {
          float got = out->data.as<float>()[size_t(t) * out->ld + n];
          double err = std::fabs(got - want) / (std::fabs(want) + rrms + 1e-12);
          if (err > kTolerance || (exact && double(got) != want))
            fatal("FP64 check K=%u N=%u R=%u ks=%u t=%u n=%u error=%.9g exact=%d", K, m.N, R, ks, t, n, err, exact);
          count.cpu_error = std::max(count.cpu_error, err);
        }
      }
    }
    ++count.cases;
  }
}
static void checks(Gpu& g, const std::string& kernel, bool shapes, bool segments = false) {
  Checks c;
  if (segments) {
    for (uint32_t K : {192u, 256u, 512u}) for (uint32_t N : {17u, 65u})
      for (uint32_t R : {1u, 2u, 7u, 8u}) for (uint32_t ks : {1u, 2u, 4u, 8u})
        for (uint32_t nseg : {2u, 3u})
          check_case(g, c, N, K, R, ks, nseg, uint32_t(c.cases + 7), int((c.cases / 2) % 5), false, kernel);
  } else if (shapes) {
    for (auto [N, K] : std::vector<std::pair<uint32_t, uint32_t>>{{18432, 2560}, {12352, 2560}, {10240, 2560},
                                                                  {2560, 4096}, {2560, 9216}, {248320, 2560}})
      for (uint32_t seed : {7u, 19u, 42u}) {
        check_case(g, c, N, K, 8, default_plan(g, N, K).ks, 1, seed, 0, false, kernel);
        printf("check_shape N=%u K=%u seed=%u passed=1\n", N, K, seed); fflush(stdout);
      }
  } else {
    for (uint32_t R : {1u, 2u, 7u, 8u}) for (uint32_t K : {64u, 128u, 192u, 256u, 320u})
      for (uint32_t N : {1u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 33u}) for (uint32_t ks : {1u, 2u, 4u, 8u}) {
        uint32_t seed = uint32_t(c.cases + 7);
        int pattern = int((c.cases / 2) % 5);
        check_case(g, c, N, K, R, ks, c.cases % 3 == 0 ? 4 : 1, seed, pattern, c.cases % 9 == 0 && pattern != 1, kernel);
      }
  }
  printf("AKO_CHECK={\"passed\":true,\"cases\":%zu,\"outputs\":%zu,\"bitwise_outputs\":%zu,\"max_cpu_error\":%.9g,\"max_gpu_error\":%.9g}\n",
         c.cases, c.outputs, c.bitwise_outputs, c.cpu_error, c.gpu_error);
}

static void scale_checks(Gpu& g, const std::string& kernel) {
  constexpr uint16_t scales[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f,
      0x0080, 0x8080, 0x3c00, 0xbc00, 0x7300, 0x737f, 0x7380, 0xf380,
      0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1, 0xffc1};
  size_t cases = 0, values = 0, nans = 0, infinities = 0;
  for (uint32_t K : {64u, 128u, 192u, 256u, 320u})
    for (uint32_t R : {1u, 2u, 7u, 8u})
      for (uint32_t ks : {1u, 2u, 4u, 8u}) {
        constexpr uint32_t N = 41;
        Arena arena(g);
        Input x(arena, K, R, 71, int(cases % 5));
        std::mt19937 rng(97);
        std::vector<Matrix> m{matrix(arena, N, K, rng, false)};
        for (size_t i = 0; i < size_t(N) * K / 64; ++i) {
          m[0].s.as<uint16_t>()[i] = scales[i % std::size(scales)];
          m[0].b.as<uint16_t>()[i] = 0;
        }
        std::vector<Output> a, b;
        a.emplace_back(arena, N, R, cases % 2); b.emplace_back(arena, N, R, cases % 2);
        const Plan pl{std::min(2u, 8u / ks), ks};
        auto e = g.begin(); prep(g, *e, x);
        compute(g, *e, kBaseline, x, m, a, pl);
        compute(g, *e, kernel, x, m, b, pl);
        e->commit_and_wait(); a[0].guards(); b[0].guards();
        for (uint32_t t = 0; t < R; ++t) for (uint32_t n = 0; n < N; ++n) {
          const float want = a[0].data.as<float>()[size_t(t) * a[0].ld + n];
          const float got = b[0].data.as<float>()[size_t(t) * b[0].ld + n];
          if (std::isnan(want)) {
            if (!std::isnan(got)) fatal("raw scale NaN mismatch K=%u R=%u ks=%u n=%u", K, R, ks, n);
            ++nans;
          } else {
            if (std::bit_cast<uint32_t>(got) != std::bit_cast<uint32_t>(want))
              fatal("raw scale bitwise mismatch K=%u R=%u ks=%u n=%u want=%08x got=%08x",
                    K, R, ks, n, std::bit_cast<uint32_t>(want), std::bit_cast<uint32_t>(got));
            infinities += std::isinf(want);
          }
          ++values;
        }
        ++cases;
      }
  printf("AKO_CHECK={\"passed\":true,\"cases\":%zu,\"values\":%zu,\"nan_class\":%zu,\"infinity_bitwise\":%zu}\n",
         cases, values, nans, infinities);
}


static uint32_t restored_pair(const AffineCoeffFixed& packed, uint32_t n, uint32_t g, uint32_t G) {
  const size_t pos = ((size_t(n)/16)*((G+1)/2)+g/2)*17;
  const uint32_t header = packed.words.at(pos);
  uint32_t sv, bv;
  if (header & 0x80000000u) {
    const size_t raw = pos+(header&0x7fffffffu);
    if (raw%32 || raw < packed.blocks*17 || raw+32 > packed.words.size()) fatal("raw coefficient span/alignment invalid");
    sv = (packed.words.at(raw+(n%16)*2)>>(16*(g%2)))&65535;
    bv = (packed.words.at(raw+(n%16)*2+1)>>(16*(g%2)))&65535;
  } else {
    const uint32_t c = (packed.words.at(pos+1+n%16)>>(16*(g%2)))&65535;
    const uint32_t sm = (header&65535)+(c&255), bm = sm+kAffineCoeffFixedBiasBase+((c>>8)&127);
    sv = sm|(c&32768); bv = bm|((c^32768)&32768);
  }
  return sv|(bv<<16);
}

static void cpu_fixture_checks() {
  const char* path = getenv("NI_QMM_COEFF_FIXTURES");
  if (!path) fatal("CPU fixture check requires the frozen manifest");
  auto manifest = json_file(path);
  if (!manifest) fatal("invalid fixture manifest");
  size_t pairs = 0, compressed = 0, blocks = 0, bytes = 0, matrices = 0;
  for (const auto& entry : manifest->at("shapes").items()) {
    const auto& list = entry.value();
    for (size_t j = 0; j < list.size(); ++j) {
      uint32_t N = 0, K = list.at(j).at(0).at("K").get<uint32_t>();
      for (const auto& seg : list.at(j)) N += seg.at("N").get<uint32_t>();
      const uint32_t G = K/64;
      std::vector<uint16_t> s(size_t(N)*G), b(s.size());
      // CPU-only Buffer views: no MTLBuffer or Gpu object is created in this mode.
      Matrix m{{},Buffer{nullptr,reinterpret_cast<uint8_t*>(s.data()),0,s.size()*2},
                   Buffer{nullptr,reinterpret_cast<uint8_t*>(b.data()),0,b.size()*2},N,K};
      fixture_coefficients(m,j);
      const auto packed = pack_affine_coeff_fixed(s.data(),b.data(),N,G);
      for (uint32_t n = 0; n < N; ++n) for (uint32_t g = 0; g < G; ++g) {
        const size_t i = size_t(n)*G+g;
        if (restored_pair(packed,n,g,G) != (uint32_t(s[i]) | (uint32_t(b[i])<<16))) fatal("full real coefficient roundtrip mismatch");
        ++pairs;
      }
      compressed += packed.compressed_blocks; blocks += packed.blocks; bytes += packed.words.size()*4; ++matrices;
    }
  }
  printf("CPU_FIXTURE_CHECK={\"passed\":true,\"pairs\":%zu,\"compressed_blocks\":%zu,\"blocks\":%zu,\"bytes\":%zu,\"matrices\":%zu}\n",
         pairs,compressed,blocks,bytes,matrices);
}

static void cpu_codec_checks() {
  size_t values = 0;
  auto check = [&](const std::vector<uint16_t>& s, const std::vector<uint16_t>& b, uint32_t N, uint32_t G) {
    const auto packed = pack_affine_coeff_fixed(s.data(),b.data(),N,G);
    for (uint32_t n = 0; n < N; ++n) for (uint32_t g = 0; g < G; ++g) {
      const size_t i = size_t(n)*G+g;
      const uint32_t got = restored_pair(packed,n,g,G);
      if (got != (uint32_t(s[i]) | (uint32_t(b[i])<<16))) fatal("CPU codec bit mismatch");
      ++values;
    }

    // Independent inverse of the new execution layout, including raw slabs and tails.
    std::vector<uint32_t> w(size_t(N)*G*8);
    for (size_t i = 0; i < w.size(); ++i) w[i] = uint32_t(i*2654435761u + 0xfedcba98u);
    const auto image = pack_affine_packed_coeff(w.data(),s.data(),b.data(),N,G);
    for (uint32_t n = 0; n < N; ++n) for (uint32_t g = 0; g < G; ++g) {
      const size_t block = (size_t(n)/16)*((G+1)/2)+g/2;
      const uint32_t h = image.headers.at(block), sh = 16*(g%2);
      uint32_t sv, bv;
      if (h & 0x80000000u) {
        const size_t p = h & 0x7fffffffu;
        if (p%32 || p<image.blocks*272 || p+32>image.words.size()) fatal("packed raw span invalid");
        sv = (image.words.at(p+(n%16)*2)>>sh)&65535;
        bv = (image.words.at(p+(n%16)*2+1)>>sh)&65535;
      } else {
        const uint32_t c = (image.words.at(block*272+256+n%16)>>sh)&65535;
        const uint32_t sm = (h&65535)+(c&255), bm = sm+355+((c>>8)&127);
        sv = sm|(c&32768); bv = bm|((c^32768)&32768);
      }
      const size_t i = size_t(n)*G+g;
      if (sv != s[i] || bv != b[i]) fatal("packed coefficient inverse mismatch");
      for (uint32_t k = 0; k < 8; ++k) {
        const uint32_t fm=n%8, fn=k&6u;
        const uint32_t lane=((fm>>2)<<4)|((fn>>2)<<3)|((fm&3)<<1)|((fn>>1)&1u);
        const size_t p=block*272+((g%2)*2+(n%16)/8)*64+lane*2+(k%2);
        if (image.words.at(p) != w[i*8+k]) fatal("packed weight inverse mismatch");
      }
    }
    return packed.compressed_blocks;
  };
  std::vector<uint16_t> s(65536), b(65536);
  for (uint32_t mode = 0; mode < 4; ++mode) {
    for (uint32_t i = 0; i < 65536; ++i) {
      s[i] = uint16_t(i);
      b[i] = mode == 0 ? uint16_t(i^32768) : mode == 1 ? uint16_t((i+383)^32768)
             : mode == 2 ? uint16_t(~i) : uint16_t(i*127+37);
    }
    check(s,b,65536,1);
  }
  for (uint32_t ds : {255u,256u}) for (uint32_t dd : {127u,128u}) {
    s.assign(32,0x3000); b.assign(32,0xb100);
    s[31] += ds; b[31] = uint16_t(0x3100+ds+dd)|0x8000;
    const size_t compressed = check(s,b,16,2);
    if (compressed != 0) fatal("original D=256 boundary must use raw encoding");
  }
  // Retain the original D=256 cases above; add fast-path and fixed-window edges.
  for (uint32_t ds : {255u,256u}) for (uint32_t dd : {127u,128u}) {
    s.assign(32,0x3000); b.assign(32,uint16_t(0x3000+kAffineCoeffFixedBiasBase)|0x8000);
    s[31] += ds;
    b[31] = uint16_t(0x3000+kAffineCoeffFixedBiasBase+ds+dd)|0x8000;
    const size_t compressed = check(s,b,16,2);
    if (compressed != size_t(ds == 255 && dd == 127)) fatal("fixed codec range boundary mismatch");
  }
  for (uint32_t delta : {354u,355u,482u,483u}) {
    s.assign(32,0x3000); b.assign(32,uint16_t(0x3000+delta)|0x8000);
    const size_t compressed = check(s,b,16,2);
    if (compressed != size_t(delta >= 355 && delta <= 482)) fatal("fixed bias window boundary mismatch");
  }
  for (uint32_t N : {1u,15u,17u,31u,33u}) for (uint32_t G : {1u,3u,5u,16u}) {
    s.assign(size_t(N)*G,0x3000); b.assign(s.size(),0xb17f);
    check(s,b,N,G);
    b.back() ^= 0x8000;  // retain a mixed or wholly raw tail block
    check(s,b,N,G);
  }
  printf("CPU_CODEC_CHECK={\"passed\":true,\"coefficient_pairs\":%zu}\n",values);
}

static void codec_checks(Gpu& g, const std::string& kernel) {
  size_t cases = 0, bits = 0, outputs = 0;
  // First replay the unchanged D=256 fixtures as raw cases, then cover compression.
  for (uint32_t delta_base : {256u,355u})
  for (uint32_t G : {1u,2u,3u,4u,5u,16u}) for (uint32_t N : {1u,17u,33u})
    for (uint32_t R : {1u,2u,7u,8u}) for (uint32_t ks : {1u,2u,4u,8u}) {
      Arena arena(g);
      Input x(arena, G*64, R, 127, 0);
      std::mt19937 rng(211);
      std::vector<Matrix> matrices;
      std::vector<Output> base, candidate;
      std::vector<Buffer> decoded;
      for (uint32_t seg = 0; seg < (N == 17 ? 3u : 1u); ++seg) {
        matrices.push_back(matrix(arena, N+seg*3, G*64, rng, false));
        Matrix& m = matrices.back();
        for (uint32_t n = 0; n < m.N; ++n) for (uint32_t j = 0; j < G; ++j) {
          const size_t i = size_t(n)*G+j, block = (n/16)*((G+1)/2)+j/2;
          const uint16_t sm = uint16_t(0x3b00 + ((n*13+j*7)%256));
          const uint16_t bm = uint16_t(sm + delta_base + ((n*11+j*3)%128));
          const uint16_t sign = (i%2) ? 0x8000 : 0;
          m.s.as<uint16_t>()[i] = sm | sign;
          m.b.as<uint16_t>()[i] = bm | (sign^0x8000);
          if (block%3 == 2 && n%16 == 0 && j%2 == 0) m.b.as<uint16_t>()[i] ^= 0x8000;
        }
        base.emplace_back(arena, m.N, R, seg%2);
        candidate.emplace_back(arena, m.N, R, seg%2);
        decoded.push_back(arena.alloc((size_t(m.N)*G+2*kGuard)*4,"coeff_bits_check"));
      }
      for (int replay = 0; replay < 2; ++replay) {
        for (size_t seg = 0; seg < matrices.size(); ++seg) {
          Matrix& m = matrices[seg];
          if (replay) {
            for (size_t i = 0; i < size_t(m.N)*G; ++i) { m.s.as<uint16_t>()[i] ^= 1; m.b.as<uint16_t>()[i] ^= 1; }
            m.coeff.reset();  // explicit invalidation when immutable parameter storage is replaced
          }
          prepare_coeff(g,m);
          base[seg].reset(); candidate[seg].reset();
          std::fill_n(decoded[seg].as<uint32_t>(), decoded[seg].size/4, 0x7b7b7b7bu);
        }
        auto e = g.begin();
        prep(g,*e,x);
        for (size_t seg = 0; seg < matrices.size(); ++seg) {
          const Matrix& m = matrices[seg];
          const uint32_t dims[2] = {m.N,G};
          e->use(g.pipeline("v8pc_check")); e->buf(0,m.s); e->buf(1,m.b); e->buf(2,m.coeff->buffer);
          e->buf(3,decoded[seg].view(kGuard*4,size_t(m.N)*G*4)); e->val(4,dims); e->buf(5,m.coeff->headers);
          e->dispatch((m.N*G+127)/128,1,1,128);
        }
        const Plan plan{std::min(2u,8u/ks),ks};
        compute(g,*e,kBaseline,x,matrices,base,plan);
        compute(g,*e,kernel,x,matrices,candidate,plan);
        e->commit_and_wait();
        const auto hx = x.rounded();
        for (size_t seg = 0; seg < matrices.size(); ++seg) {
          const Matrix& m = matrices[seg];
          base[seg].guards(); candidate[seg].guards();
          for (size_t j = 0; j < kGuard; ++j)
            if (decoded[seg].as<uint32_t>()[j] != 0x7b7b7b7bu || decoded[seg].as<uint32_t>()[kGuard+size_t(m.N)*G+j] != 0x7b7b7b7bu)
              fatal("coefficient probe buffer boundary overwritten");
          for (size_t i = 0; i < size_t(m.N)*G; ++i) {
            const uint32_t want = m.s.as<uint16_t>()[i] | (uint32_t(m.b.as<uint16_t>()[i])<<16);
            if (decoded[seg].as<uint32_t>()[kGuard+i] != want) fatal("coefficient bit pattern mismatch");
            ++bits;
          }
          std::vector<double> refs;
          double sq = 0;
          for (uint32_t t = 0; t < R; ++t) for (uint32_t n = 0; n < m.N; ++n) {
            const double v = reference(m,hx.data()+size_t(t)*m.K,n,candidate[seg].add);
            refs.push_back(v); sq += v*v;
          }
          const double rms = std::sqrt(sq/refs.size());
          size_t i = 0;
          for (uint32_t t = 0; t < R; ++t) for (uint32_t n = 0; n < m.N; ++n,++i) {
            const float a = base[seg].data.as<float>()[size_t(t)*base[seg].ld+n];
            const float b = candidate[seg].data.as<float>()[size_t(t)*candidate[seg].ld+n];
            if (std::bit_cast<uint32_t>(a) != std::bit_cast<uint32_t>(b)) fatal("compressed QMM bitwise mismatch");
            if (!std::isfinite(b) || std::abs(double(b)-refs[i])/(std::abs(refs[i])+rms+1e-12)>kTolerance)
              fatal("compressed QMM FP64 mismatch");
            ++outputs;
          }
        }
        ++cases;
      }
    }
  if (!compressed_blocks || compressed_blocks == pack_blocks) fatal("codec check did not cover both encodings");
  printf("AKO_CHECK={\"passed\":true,\"cases\":%zu,\"bit_pairs\":%zu,\"outputs\":%zu,\"compressed_blocks\":%zu,\"blocks\":%zu}\n",
         cases,bits,outputs,compressed_blocks,pack_blocks);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: qmm_packed_coeff_check check-cpu-codec|check-cpu-fixtures|check-codec|check-scales|check-small|check-segments|check-shapes|check-real-coeff [kernel]\n");
    return 2;
  }
  if (std::string(argv[1]) == "check-cpu-codec") { cpu_codec_checks(); return 0; }
  bench::GpuLock lock;
  if (!bench::GpuLock::held_by_caller() && !lock.acquire(nullptr, 0)) fatal("GPU lock busy: %s", bench::GpuLock::path().c_str());
  if (std::string(argv[1]) == "check-cpu-fixtures") { cpu_fixture_checks(); return 0; }
  Gpu& g = Gpu::get();
  g.compile();
  const std::string mode = argv[1];
  const std::string kernel = argc > 2 ? argv[2] : "qmm8pc_b4_nf1";
  printf("device=%s cores=%d kernel=%s thermal=%s\n", g.name().c_str(), g.core_count(), kernel.c_str(), Gpu::thermal_state());
  if (kernel != kBaseline && kernel != "qmm8pc_b4_nf1")
    fatal("unsupported kernel: %s", kernel.c_str());
  if (mode == "check-real-coeff") {
    if (!getenv("NI_QMM_COEFF_FIXTURES")) fatal("real coefficient check requires the frozen manifest");
    real_fixture_check = true;
    checks(g, kernel, true);
  }
  else if (mode == "check-codec") codec_checks(g, kernel);
  else if (mode == "check-scales") scale_checks(g, kernel);
  else if (mode == "check-small" || mode == "check-shapes" || mode == "check-segments")
    checks(g, kernel, mode == "check-shapes", mode == "check-segments");
  else { fprintf(stderr, "unknown mode or missing arguments\n"); return 2; }
  printf("QMM_CD_DISPATCH={\"candidate\":%zu,\"original\":%zu}\n",
         candidate_dispatches, original_dispatches);
  return 0;
}
