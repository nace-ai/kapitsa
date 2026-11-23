// Tiny source-reader/kernel integration check; --cpu never locks or initializes Metal.
#include "bench/bench.h"
#include "gguf/quant.h"
#include "metal/gpu.h"
#include "model/inkling2_gguf.h"
#include "model/inkling2_kernels.h"
#include "../tests/fixtures/ink2_mixed_gguf.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <limits>
#include <variant>

namespace {
using namespace ink2_mixed_fixture;
constexpr uint32_t T = 3, Top = 2;
constexpr size_t Guard = 64, GpuCap = 512 * 1024;
constexpr double Tolerance = 2e-4;
constexpr std::array<uint32_t, T * Top> ids = {0, 1, 1, 0, 0, 1};
constexpr std::array<float, T * Top> weights = {.2f, .7f, .4f, .3f, .6f, .15f};
struct GgufP { uint32_t T, H, I, top, type, row_bytes; };
struct NativeP { uint32_t T, H, I, top; };
static_assert(sizeof(GgufP) == 24 && sizeof(NativeP) == 16);
enum class Outer { Correct, Unit, Swapped };

struct Projection {
  const ni::ink2::ExpertSource& source;
  std::vector<uint8_t> raw;
  std::vector<float> decoded;
  Projection(const ni::gguf::File& file, const ni::ink2::ExpertSource& s, bool gu) : source(s) {
    const auto& w = *s.weight;
    const size_t K = w.ne[0], R = w.ne[1];
    const auto& bits = gu ? gu_outer : down_outer;
    if (s.native_nvfp4()) {
      check(s.outer.size() == E, "outer shape");
      for (size_t e = 0; e < E; ++e)
        check(std::bit_cast<uint32_t>(s.outer[e]) == bits[e], "outer bits/order");
    } else check(s.outer.empty(), "block projection has native outer scales");
    raw.resize(w.bytes);
    decoded.resize(E * R * K);
    file.read(w, 0, raw.data(), raw.size());
    check(std::memcmp(raw.data(), raw.data() + R * w.row_bytes(), w.row_bytes()) != 0, "identical experts");
    if (gu) check(std::memcmp(raw.data(), raw.data() + I * w.row_bytes(), w.row_bytes()) != 0, "identical gate/up");
    for (size_t r = 0; r < E * R; ++r)
      ni::gguf::dequantize(w.type, raw.data() + r * w.row_bytes(), decoded.data() + r * K, K);
    for (float value : decoded) check(std::isfinite(value), "nonfinite fixture weight");
  }
  double dot(size_t row, const float* x) const {
    const size_t K = source.weight->ne[0];
    double sum = 0;
    for (size_t k = 0; k < K; ++k) sum += double(decoded[row * K + k]) * x[k];
    return sum;
  }
  double outer(size_t e, Outer mode) const {
    if (!source.native_nvfp4() || mode == Outer::Unit) return 1;
    return source.outer[mode == Outer::Swapped ? E - 1 - e : e];
  }
  size_t gpu_bytes() const {
    const size_t values = decoded.size();
    return source.native_nvfp4() ? values / 2 + values / 16 + E * sizeof(float) + 6 * Guard
                                : raw.size() + 2 * Guard;
  }
};

std::vector<float> reference_act(const Projection& gu, const std::vector<float>& x,
                                 double& max_pre, Outer mode = Outer::Correct) {
  std::vector<float> act(T * Top * I);
  max_pre = 0;
  for (size_t t = 0; t < T; ++t) for (size_t s = 0; s < Top; ++s) for (size_t j = 0; j < I; ++j) {
    const size_t e = ids[t * Top + s], row = e * 2 * I + j;
    const double g = gu.dot(row, x.data() + t * H) * gu.outer(e, mode);
    const double u = gu.dot(row + I, x.data() + t * H) * gu.outer(e, mode);
    max_pre = std::max({max_pre, std::abs(g), std::abs(u)});
    act[(t * Top + s) * I + j] = float((4 * std::tanh(g / 4)) / (1 + std::exp(-g)) * (25 * std::tanh(u / 25)));
  }
  return act;
}
std::vector<float> reference_out(const Projection& down, const std::vector<float>& act,
                                 Outer mode = Outer::Correct) {
  std::vector<float> out(T * H);
  for (size_t t = 0; t < T; ++t) for (size_t h = 0; h < H; ++h) {
    double total = 0;
    for (size_t s = 0; s < Top; ++s) {
      const size_t e = ids[t * Top + s];
      total += weights[t * Top + s] * down.outer(e, mode) * down.dot(e * H + h, act.data() + (t * Top + s) * I);
    }
    out[t * H + h] = float(total);
  }
  return out;
}
double separation(const std::vector<float>& a, const std::vector<float>& b) {
  double delta = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    check(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite CPU reference");
    delta = std::max(delta, std::abs(double(a[i]) - b[i]) / (1 + std::abs(double(a[i]))));
  }
  return delta;
}

struct Buffers {
  ni::Gpu& gpu;
  std::vector<ni::Buffer> owned;
  size_t bytes = 0;
  ni::Buffer alloc(size_t n) {
    check(bytes + n + 2 * Guard <= GpuCap, "guarded GPU buffer cap");
    auto b = gpu.alloc(n + 2 * Guard, "ink2_mixed_check");
    owned.push_back(b);
    bytes += b.size;
    std::memset(b.ptr(), 0xa5, b.size);
    return b.view(Guard, n);
  }
  ni::Buffer copy(const void* src, size_t n) {
    auto b = alloc(n);
    std::memcpy(b.ptr(), src, n);
    return b;
  }
  void check_guards() const {
    for (const auto& b : owned) for (size_t i = 0; i < Guard; ++i)
      check(b.as<uint8_t>()[i] == 0xa5 && b.as<uint8_t>()[b.size - Guard + i] == 0xa5, "GPU buffer canary");
  }
  ~Buffers() { for (auto& b : owned) gpu.release(b); }
};
struct NativeBuffers { ni::Buffer q, scales, outer; };
using Packed = std::variant<ni::Buffer, NativeBuffers>;
Packed upload(Buffers& buffers, const Projection& p, bool gu) {
  if (!p.source.native_nvfp4()) return buffers.copy(p.raw.data(), p.raw.size());
  const auto& w = *p.source.weight;
  const size_t K = w.ne[0], R = w.ne[1];
  NativeBuffers n{buffers.alloc(E * R * K / 2), buffers.alloc(E * R * K / 16),
                  buffers.copy(p.source.outer.data(), E * sizeof(float))};
  for (size_t r = 0; r < E * R; ++r) {
    const size_t dst = (r / R) * R + (gu ? ni::ink2::native_gu_row(r % R, I) : r % R);
    ni::ink2::unpack_nvfp4_row(p.raw.data() + r * w.row_bytes(), K,
                              n.q.as<uint8_t>() + dst * K / 2, n.scales.as<uint8_t>() + dst * K / 16);
  }
  const auto& bits = gu ? gu_outer : down_outer;
  check(std::memcmp(n.outer.ptr(), bits.data(), sizeof(bits)) == 0, "uploaded outer bits/order");
  return n;
}
size_t compare(const ni::Buffer& got, const std::vector<float>& ref, const char* label) {
  double max_abs = 0, max_scaled = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double value = got.as<float>()[i], error = std::abs(value - ref[i]);
    if (!std::isfinite(value) || error > Tolerance * (1 + std::abs(double(ref[i])))) {
      std::fprintf(stderr, "%s index=%zu got=%.9g ref=%.9g\n", label, i, value, ref[i]);
      throw std::runtime_error("GPU/reference mismatch");
    }
    max_abs = std::max(max_abs, error);
    max_scaled = std::max(max_scaled, error / (1 + std::abs(double(ref[i]))));
  }
  std::printf("%s: %zu outputs, max_abs=%.9g max_scaled=%.9g\n", label, ref.size(), max_abs, max_scaled);
  return ref.size();
}

size_t check_pair(ni::Gpu* gpu, uint32_t gu_type, uint32_t down_type) {
  Fixture fixture(gu_type, down_type, Data::Bounded);
  ni::gguf::File file(fixture.path);
  const auto sources = ni::ink2::read_expert_sources(file, "blk.9.", H, I, E);
  check(sources.gu.native_nvfp4() == (gu_type == ni::gguf::kNVFP4), "GU representation");
  check(sources.down.native_nvfp4() == (down_type == ni::gguf::kNVFP4), "down representation");
  Projection gu(file, sources.gu, true), down(file, sources.down, false);
  check(gu.raw.size() + down.raw.size() + (gu.decoded.size() + down.decoded.size()) * sizeof(float) + 65536 < 8 * 1024 * 1024,
        "CPU numeric array cap");
  std::vector<float> x(T * H);
  for (size_t t = 0; t < T; ++t) for (size_t h = 0; h < H; ++h)
    x[t * H + h] = float(int((h * 17 + t * 13 + h / 7) % 33) - 16) / 128;
  double max_pre = 0;
  auto ref_act = reference_act(gu, x, max_pre);
  if (max_pre > 2) {
    const float factor = std::ldexp(1.0f, -int(std::ceil(std::log2(max_pre / 2))));
    for (auto& value : x) value *= factor;
    ref_act = reference_act(gu, x, max_pre);
  }
  check(std::isfinite(max_pre) && max_pre <= 2 && max_pre > .001, "zero or saturated GU fixture");
  const auto ref_out = reference_out(down, ref_act);
  const std::vector<float> zero_act(ref_act.size()), zero_out(ref_out.size());
  check(separation(ref_act, zero_act) > 10 * Tolerance && separation(ref_out, zero_out) > 10 * Tolerance,
        "zero fixture signal");
  for (Outer bad : {Outer::Unit, Outer::Swapped}) {
    double ignored = 0;
    if (sources.gu.native_nvfp4())
      check(separation(ref_act, reference_act(gu, x, ignored, bad)) > 10 * Tolerance, "GU outer mutation is invisible");
    if (sources.down.native_nvfp4())
      check(separation(ref_out, reference_out(down, ref_act, bad)) > 10 * Tolerance, "down outer mutation is invisible");
  }
  const size_t common = (T * H + T * Top * I + T * H) * sizeof(float) + sizeof(ids) + sizeof(weights) + 10 * Guard;
  const size_t planned = common + gu.gpu_bytes() + down.gpu_bytes();
  check(planned <= GpuCap, "planned GPU buffer cap");
  std::printf("preflight GU=%u down=%u max_pre=%.6g guarded_gpu_bytes=%zu\n", gu_type, down_type, max_pre, planned);
  if (!gpu) return 0;

  Buffers b{*gpu, {}, 0};
  const auto gx = b.copy(x.data(), x.size() * sizeof(float)), gi = b.copy(ids.data(), sizeof(ids)),
             gw = b.copy(weights.data(), sizeof(weights));
  const auto act = b.alloc(ref_act.size() * sizeof(float)), out = b.alloc(ref_out.size() * sizeof(float));
  std::fill_n(act.as<float>(), ref_act.size(), std::numeric_limits<float>::quiet_NaN());
  std::fill_n(out.as<float>(), ref_out.size(), std::numeric_limits<float>::quiet_NaN());
  const auto gu_gpu = upload(b, gu, true), down_gpu = upload(b, down, false);
  check(b.bytes == planned, "GPU allocation ledger mismatch");
  const NativeP np{T, H, I, Top};
  {
    auto e = gpu->begin();
    if (const auto* n = std::get_if<NativeBuffers>(&gu_gpu)) {
      e->use(ni::ink2::pipeline("ink2_moe_nvfp4_w13_situ"));
      e->buf(0, gx); e->buf(1, gi); e->buf(2, n->q); e->buf(3, n->scales); e->buf(4, n->outer);
      e->buf(5, act); e->val(6, np); e->dispatch((I + 15) / 16, Top, T, 128);
    } else {
      const GgufP p{T, H, I, Top, gu_type, uint32_t(sources.gu.weight->row_bytes())};
      e->use(ni::ink2::pipeline("ink2_gguf_gu"));
      e->buf(0, gx); e->buf(1, gi); e->buf(2, std::get<ni::Buffer>(gu_gpu)); e->buf(3, act); e->val(4, p);
      e->dispatch((I + 3) / 4, Top, T, 128);
    }
    e->commit_and_wait();
  }
  b.check_guards();
  size_t checked = compare(act, ref_act, "GU/SiTU");
  const std::vector<float> gpu_act(act.as<float>(), act.as<float>() + ref_act.size());
  {
    auto e = gpu->begin();
    if (const auto* n = std::get_if<NativeBuffers>(&down_gpu)) {
      e->use(ni::ink2::pipeline("ink2_moe_nvfp4_w2_acc"));
      e->buf(0, act); e->buf(1, gi); e->buf(2, gw); e->buf(3, n->q); e->buf(4, n->scales);
      e->buf(5, n->outer); e->buf(6, out); e->val(7, np); e->dispatch((H + 15) / 16, T, 1, 128);
    } else {
      const GgufP p{T, H, I, Top, down_type, uint32_t(sources.down.weight->row_bytes())};
      e->use(ni::ink2::pipeline("ink2_gguf_down"));
      e->buf(0, act); e->buf(1, gi); e->buf(2, gw); e->buf(3, std::get<ni::Buffer>(down_gpu));
      e->buf(4, out); e->val(5, p); e->dispatch((H + 3) / 4, T, 1, 128);
    }
    e->commit_and_wait();
  }
  b.check_guards();
  checked += compare(out, reference_out(down, gpu_act), "down/readback-act");
  checked += compare(out, ref_out, "down/end-to-end");
  return checked;
}
size_t run(ni::Gpu* gpu) {
  size_t checked = 0;
  for (const auto& pair : {std::array<uint32_t, 2>{ni::gguf::kIQ3_XXS, ni::gguf::kNVFP4},
                           {ni::gguf::kNVFP4, ni::gguf::kIQ3_XXS},
                           {ni::gguf::kIQ3_XXS, ni::gguf::kIQ3_XXS},
                           {ni::gguf::kNVFP4, ni::gguf::kNVFP4}})
    checked += check_pair(gpu, pair[0], pair[1]);
  return checked;
}
}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--cpu") != 0)) {
    std::fprintf(stderr, "usage: ink2_mixed_gguf_check [--cpu]\n");
    return 1;
  }
  try {
    std::printf("mixed GGUF checker source=%s; no model constructor or production-model dispatch coverage\n", KAPITSA_GIT);
    if (argc == 2) {
      run(nullptr);
      std::puts("mixed GGUF CPU preflight: four source/oracle cases passed; GPU not accessed");
      return 0;
    }
    ni::bench::GpuLock lock;
    if (!ni::bench::GpuLock::held_by_caller() && !lock.acquire(nullptr, 0)) {
      std::fprintf(stderr, "GPU lock unavailable; checker did not wait or initialize Metal.\n");
      return 2;
    }
    auto& gpu = ni::Gpu::get();
    const size_t checked = run(&gpu);
    std::printf("mixed GGUF GPU: %zu comparisons passed on %s\n", checked, gpu.name().c_str());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "mixed GGUF checker: %s\n", error.what());
    return 1;
  }
}
