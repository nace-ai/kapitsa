// Input segmentation for the dense ANE FFN: default geometry checks need no model or GPU.
//   ane_segment_check          CPU: supported widths and the explicit 4096 opt-in
//   ane_segment_check --gpu    also compare segmented/full activation packing and weight staging,
//                              including nonzero offsets, padded rows, tails and allocation guards
// The GPU mode takes BENCH_GPU_LOCK (default /tmp/kapitsa-gpu.lock), unless the caller holds it.
// Run with MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1; these kernels fit the layer's doubled cap.
#include "ane/ffn_split.h"
#include "bench/bench.h"
#include "metal/gpu.h"
#include "util/bf16.h"
#include "util/log.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace ni;

namespace {
size_t checked = 0;
int failures = 0, cases = 0;

struct Guard {
  Buffer allocation, data;
  size_t n;
  explicit Guard(size_t bytes) : n(bytes) {
    allocation = Gpu::get().alloc(bytes + 512, "segment guard");
    memset(allocation.ptr(), 0x6a, allocation.size);
    data = allocation.view(256, bytes);
  }
  ~Guard() { Gpu::get().release(allocation); }
  void check() const {
    const auto* p = allocation.as<unsigned char>();
    for (size_t i = 0; i < 256; ++i)
      if (p[i] != 0x6a || p[256 + n + i] != 0x6a) ++failures;
    checked += 512;
  }
};

void equal(const void* a, const void* b, size_t n, const char* why) {
  ++cases;
  checked += n;
  if (memcmp(a, b, n)) {
    ++failures;
    fprintf(stderr, "FAIL %s, %zu bytes\n", why, n);
  }
}

struct WP { uint32_t row0, col0, width, wpr, gpr, stride, sstride; };

void check_pack(uint32_t D, uint32_t T) {
  auto& g = Gpu::get();
  const uint32_t segment = D == 4096 ? 2048 : 2560;
  const uint32_t R = std::max(32u, (T + 31) / 32 * 32), stride = R + 64;
  Guard input(size_t(R) * D * 2), signs(512 * 4), rotated(size_t(R) * D * 2), scale(R * 2), full(size_t(D) * stride);
  std::mt19937 rng(1234 + T + D);
  for (size_t i = 0; i < size_t(R) * D; ++i)
    input.data.as<_Float16>()[i] = _Float16((int(rng() % 2001) - 1000) / 137.f);
  for (int i = 0; i < 512; ++i) signs.data.as<float>()[i] = i % 3 ? 1.f : -1.f;
  struct RotP { uint32_t T, D; };
  struct PackP { uint32_t T, D, ch0, stride; };
  auto e = g.begin();
  e->use(g.pipeline("ane_rotate"));
  e->buf(0, input.data);
  e->buf(1, signs.data);
  e->buf(2, rotated.data);
  e->buf(3, scale.data);
  e->val(4, RotP{T, D});
  e->dispatch(R, 1, 1, 32 * ane::FfnSplit::rotate_simdgroups(D));
  e->use(g.pipeline("ane_pack"));
  e->buf(0, rotated.data);
  e->buf(1, full.data);
  e->val(2, PackP{T, D, 0, stride});
  e->dispatch(R / 32, D / 32, 1, 32, 8);
  e->commit_and_wait();
  for (uint32_t start = 0; start < D; start += segment) {
    Guard part(size_t(segment) * stride);
    auto p = g.begin();
    p->use(g.pipeline("ane_pack"));
    p->buf(0, rotated.data);
    p->buf(1, part.data);
    p->val(2, PackP{T, D, start, stride});
    p->dispatch(R / 32, segment / 32, 1, 32, 8);
    p->commit_and_wait();
    equal(part.data.ptr(), full.data.as<unsigned char>() + size_t(start) * stride, part.n, "packed channel segment");
    for (uint32_t c = 0; c < segment; ++c) {
      for (uint32_t t = T; t < R; ++t)
        if (part.data.as<signed char>()[size_t(c) * stride + t] != 0) ++failures;
      for (uint32_t t = R; t < stride; ++t)
        if (part.data.as<unsigned char>()[size_t(c) * stride + t] != 0x6a) ++failures;
    }
    part.check();
  }
  input.check();
  signs.check();
  rotated.check();
  scale.check();
  full.check();
}

void check_weights(uint32_t K, uint32_t N, uint32_t row0) {
  auto& g = Gpu::get();
  const uint32_t segment = K == 4096 ? 2048 : 2560;
  const uint32_t source = N + row0, wpr = K / 8, gpr = K / 64, stride = K + 64, sstride = 32;
  Guard w(size_t(source) * wpr * 4), s(size_t(source) * gpr * 2), b(size_t(source) * gpr * 2), signs(512 * 4), rs(N * 2);
  Guard full(size_t(N) * stride), scout(size_t(N) * sstride * 2);
  std::mt19937 rng(111 + K + N + row0);
  for (size_t i = 0; i < size_t(source) * wpr; ++i) w.data.as<uint32_t>()[i] = rng();
  for (size_t i = 0; i < size_t(source) * gpr; ++i) {
    s.data.as<uint16_t>()[i] = f32_to_bf16((rng() % 500 + 1) / 10000.f);
    b.data.as<uint16_t>()[i] = f32_to_bf16(-float(rng() % 300) / 1000.f);
  }
  for (int i = 0; i < 512; ++i) signs.data.as<float>()[i] = i % 3 ? 1.f : -1.f;
  auto e = g.begin();
  e->use(g.pipeline("ane_row_scale_b128"));
  e->buf(0, w.data);
  e->buf(1, s.data);
  e->buf(2, b.data);
  e->buf(3, rs.data);
  e->buf(4, signs.data);
  e->val(5, WP{row0, 0, K, wpr, gpr, 0, 0});
  e->dispatch(N / 8, 1, 1, 256);
  e->use(g.pipeline("ane_stage_b128"));
  e->buf(0, w.data);
  e->buf(1, s.data);
  e->buf(2, b.data);
  e->buf(3, rs.data);
  e->buf(4, full.data);
  e->buf(5, scout.data);
  e->buf(6, signs.data);
  e->val(7, WP{row0, 0, K, wpr, gpr, stride, sstride});
  e->dispatch(N / 8, K / 128, 1, 256);
  e->commit_and_wait();
  for (uint32_t start = 0; start < K; start += segment) {
    const uint32_t ps = segment + 64;
    Guard part(size_t(N) * ps), sp(size_t(N) * sstride * 2);
    auto p = g.begin();
    p->use(g.pipeline("ane_stage_b128"));
    p->buf(0, w.data);
    p->buf(1, s.data);
    p->buf(2, b.data);
    p->buf(3, rs.data);
    p->buf(4, part.data);
    p->buf(5, sp.data);
    p->buf(6, signs.data);
    p->val(7, WP{row0, start, segment, wpr, gpr, ps, sstride});
    p->dispatch(N / 8, segment / 128, 1, 256);
    p->commit_and_wait();
    for (uint32_t r = 0; r < N; ++r) {
      equal(part.data.as<unsigned char>() + size_t(r) * ps,
            full.data.as<unsigned char>() + size_t(r) * stride + start, segment, "staged weight segment");
      for (uint32_t c = segment; c < ps; ++c)
        if (part.data.as<unsigned char>()[size_t(r) * ps + c] != 0x6a) ++failures;
    }
    equal(sp.data.ptr(), scout.data.ptr(), sp.n, "staged row scales and padding");
    part.check();
    sp.check();
  }
  w.check();
  s.check();
  b.check();
  signs.check();
  rs.check();
  full.check();
  scout.check();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 2 || (argc == 2 && strcmp(argv[1], "--gpu"))) {
    fprintf(stderr, "usage: ane_segment_check [--gpu]\n");
    return 2;
  }
  const char* prior = getenv("NI_ANE_H4096");
  const bool had_prior = prior != nullptr;
  const std::string saved = prior ? prior : "";
  auto expect = [](bool ok, const char* what) {
    ++cases;
    if (!ok) {
      ++failures;
      fprintf(stderr, "FAIL %s\n", what);
    }
  };
  unsetenv("NI_ANE_H4096");
  expect(ane::FfnSplit::input_segment(4096) == 0 && ane::FfnSplit::rotate_simdgroups(4096) == 0,
         "4096 is excluded by default");
  expect(ane::split_bytes(4096, 12288, 24, 32, 2) == 0, "unsupported width has no split budget");
  for (const char* v : {"0", "2", "true", ""}) {
    setenv("NI_ANE_H4096", v, 1);
    expect(ane::FfnSplit::input_segment(4096) == 0, "4096 requires the exact opt-in value 1");
  }
  setenv("NI_ANE_H4096", "1", 1);
  expect(ane::FfnSplit::input_segment(4096) == 2048 && ane::FfnSplit::rotate_simdgroups(4096) == 8,
         "4096 uses two input segments and eight simdgroups");
  for (uint32_t h : {0u, 128u, 3840u})
    expect(ane::FfnSplit::input_segment(h) == 0, "unsupported width remains excluded");
  for (uint32_t h : {2560u, 5120u})
    expect(ane::FfnSplit::input_segment(h) == 2560, "existing input segmentation is unchanged");
  expect(ane::FfnSplit::rotate_simdgroups(2560) == 5 && ane::FfnSplit::rotate_simdgroups(5120) == 8,
         "existing rotation geometry is unchanged");
  expect(ane::FfnSplit::rotate_simdgroups(7680) == 0, "rotation rejects more than eight blocks per simdgroup");
  printf("CPU input selection: %d checks, %d failures\n", cases, failures);
  bench::GpuLock lock;
  if (argc == 2 && !failures) {
    if (!bench::GpuLock::held_by_caller()) {
      if (!lock.acquire([](const std::string& who) {
            fprintf(stderr, "waiting for the GPU lock %s%s\n", bench::GpuLock::path().c_str(),
                    who.empty() ? "" : (", held by " + who).c_str());
          }))
        fatal("cannot take the GPU lock: %s", strerror(errno));
    }
    Gpu::get().compile();
    const int before = cases;
    for (uint32_t d : {2560u, 4096u, 5120u})
      for (uint32_t n : {0u, 1u, 31u, 32u, 33u, 127u}) check_pack(d, n);
    for (uint32_t d : {2560u, 4096u, 5120u})
      for (uint32_t n : {8u, 16u})
        for (uint32_t offset : {0u, 8u}) check_weights(d, n, offset);
    printf("GPU input segments: %d comparisons, %zu data/guard bytes checked, %d failures\n",
           cases - before, checked, failures);
  }
  if (had_prior) setenv("NI_ANE_H4096", saved.c_str(), 1);
  else unsetenv("NI_ANE_H4096");
  return failures ? 1 : 0;
}
