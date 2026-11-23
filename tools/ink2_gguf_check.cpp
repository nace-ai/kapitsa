// Synthetic GPU check, no model. Takes the shared GPU lock before initializing Metal.
#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/inkling2_kernels.h"
#include "kernels/inkling2/gguf_blocks.h"
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <random>
#include <vector>

struct MmP { uint32_t M, N, K, type, rb, add; };
struct MoeP { uint32_t T, H, I, top, type, rb; };
static int run_checks(ni::Gpu& gpu, std::vector<ni::Buffer>& owned) {
  std::mt19937 rng(754);
  size_t checked = 0;
  auto buffer = [&](size_t bytes) {
    auto b = gpu.alloc(bytes, "ink2_gguf_check");
    owned.push_back(b);
    return b;
  };
  auto data = [&](unsigned type, unsigned K, unsigned rows) {
    const unsigned bv = ink2_gguf::block_values(type), bb = ink2_gguf::block_bytes(type);
    auto b = buffer(size_t(K / bv) * bb * rows);
    for (size_t i = 0; i < b.size; ++i) b.as<uint8_t>()[i] = uint8_t(rng());
    for (size_t off = 0; off < b.size; off += bb) {
      const size_t s = off + (type == 14 ? 208 : 0);
      b.as<uint8_t>()[s] = 0; b.as<uint8_t>()[s + 1] = 0x14;
      if (type == 12 || type == 13) { b.as<uint8_t>()[s + 2] = 0; b.as<uint8_t>()[s + 3] = 0x10; }
    }
    return b;
  };
  auto check = [&](float got, double ref, const char* name, unsigned type, size_t index) {
    if (!std::isfinite(got) || std::abs(double(got) - ref) > 2e-4 * (1 + std::abs(ref))) {
      std::fprintf(stderr, "%s type=%u index=%zu got=%.9g ref=%.9g\n", name, type, index, got, ref);
      return false;
    }
    ++checked;
    return true;
  };
  for (unsigned type : {6u, 8u, 12u, 13u, 14u, 20u, 23u}) {
    const unsigned K = ink2_gguf::block_values(type) == 32 ? 1152 : 256, M = 3, N = 7;
    const unsigned rb = K / ink2_gguf::block_values(type) * ink2_gguf::block_bytes(type);
    auto w = data(type, K, N), x = buffer(M * K * 4), y = buffer(M * N * 4);
    for (unsigned k = 0; k < M * K; ++k) x.as<float>()[k] = float(int(rng() % 101) - 50) / 100;
    for (unsigned k = 0; k < M * N; ++k) y.as<float>()[k] = 0.25f;
    const MmP p{M, N, K, type, rb, 1};
    { auto e = gpu.begin(); e->use(ni::ink2::pipeline("ink2_gguf_mm"));
      e->buf(0, x); e->buf(1, w); e->buf(2, y); e->val(3, p); e->dispatch((N + 3) / 4, M, 1, 128); e->commit_and_wait(); }
    for (unsigned m = 0; m < M; ++m) for (unsigned n = 0; n < N; ++n) {
      double ref = .25;
      for (unsigned k = 0; k < K; ++k) ref += double(x.as<float>()[m * K + k]) * ink2_gguf::value(w.as<uint8_t>() + n * rb, type, k);
      if (!check(y.as<float>()[m * N + n], ref, "mm/add", type, m * N + n)) return 1;
    }
    const unsigned H = 256, I = K, T = 3, top = 2, experts = 3;
    const unsigned grb = H / ink2_gguf::block_values(type) * ink2_gguf::block_bytes(type);
    auto gu = data(type, H, experts * 2 * I), dn = data(type, I, experts * H);
    auto input = buffer(T * H * 4), ids = buffer(T * top * 4), wt = buffer(T * top * 4);
    auto act = buffer(T * top * I * 4), out = buffer(T * H * 4);
    for (unsigned k = 0; k < T * H; ++k) input.as<float>()[k] = float(int(rng() % 101) - 50) / 100;
    for (unsigned k = 0; k < T * top; ++k) { ids.as<uint32_t>()[k] = (k + 1) % experts; wt.as<float>()[k] = .1f * float(k + 1); }
    const MoeP gp{T, H, I, top, type, grb}, dp{T, H, I, top, type, rb};
    { auto e = gpu.begin(); e->use(ni::ink2::pipeline("ink2_gguf_gu"));
      e->buf(0, input); e->buf(1, ids); e->buf(2, gu); e->buf(3, act); e->val(4, gp);
      e->dispatch((I + 3) / 4, top, T, 128); e->commit_and_wait(); }
    std::vector<float> ref_act(T * top * I);
    for (unsigned t = 0; t < T; ++t) for (unsigned s = 0; s < top; ++s) for (unsigned j = 0; j < I; ++j) {
      const unsigned id = ids.as<uint32_t>()[t * top + s];
      const uint8_t* gate = gu.as<uint8_t>() + (size_t(id) * 2 * I + j) * grb;
      double g = 0, u = 0;
      for (unsigned k = 0; k < H; ++k) { const float v = input.as<float>()[t * H + k];
        g += double(v) * ink2_gguf::value(gate, type, k); u += double(v) * ink2_gguf::value(gate + size_t(I) * grb, type, k); }
      const size_t idx = (size_t(t) * top + s) * I + j;
      ref_act[idx] = float(4 * std::tanh(g / 4) / (1 + std::exp(-g)) * 25 * std::tanh(u / 25));
      if (!check(act.as<float>()[idx], ref_act[idx], "gate-up/SiTU", type, idx)) return 1;
    }
    { auto e = gpu.begin(); e->use(ni::ink2::pipeline("ink2_gguf_down"));
      e->buf(0, act); e->buf(1, ids); e->buf(2, wt); e->buf(3, dn); e->buf(4, out); e->val(5, dp);
      e->dispatch((H + 3) / 4, T, 1, 128); e->commit_and_wait(); }
    for (unsigned t = 0; t < T; ++t) for (unsigned h = 0; h < H; ++h) {
      double total = 0;
      for (unsigned s = 0; s < top; ++s) { double sum = 0;
        const auto* row = dn.as<uint8_t>() + (size_t(ids.as<uint32_t>()[t * top + s]) * H + h) * rb;
        for (unsigned k = 0; k < I; ++k) sum += double(act.as<float>()[(t * top + s) * I + k]) * ink2_gguf::value(row, type, k);
        total += wt.as<float>()[t * top + s] * sum;
      }
      if (!check(out.as<float>()[t * H + h], total, "down/combine", type, t * H + h)) return 1;
    }
  }
  {
    const uint32_t H = 2304, V = 7, T = 3;
    auto q = buffer(V * H), sc = buffer(V * H / 32 * 2), norm = buffer(H * 4);
    auto tokens = buffer(5 * 4), out = buffer(T * H * 4);
    for (unsigned i = 0; i < V * H; ++i) q.as<int8_t>()[i] = int8_t(rng() % 255 - 127);
    for (unsigned i = 0; i < V * H / 32; ++i) sc.as<_Float16>()[i] = _Float16(.003f + float(i % 7) * .0001f);
    const int ids[] = {0, 0, 2, 6, 1};
    std::memcpy(tokens.ptr(), ids, sizeof(ids));
    for (uint32_t f32 : {0u, 1u}) {
      for (unsigned i = 0; i < H; ++i) {
        const float n = .75f + float(i % 9) / 16;
        if (f32) norm.as<float>()[i] = n;
        else norm.as<uint16_t>()[i] = uint16_t(std::bit_cast<uint32_t>(n) >> 16);
      }
      struct { uint32_t T, H, offset, V, f32; float eps; } p{T, H, 2, V, f32, 1e-6f};
      { auto e = gpu.begin(); e->use(ni::ink2::pipeline("ink2_gguf_embed"));
        e->buf(0, tokens); e->buf(1, q); e->buf(2, sc); e->buf(3, norm); e->buf(4, out); e->val(5, p);
        e->dispatch(T, 1, 1, 512); e->commit_and_wait(); }
      for (unsigned t = 0; t < T; ++t) {
        double ss = 0;
        for (unsigned c = 0; c < H; ++c) {
          const size_t index = size_t(ids[t + 2]) * H + c;
          const float v = float(q.as<int8_t>()[index]) * float(sc.as<_Float16>()[index / 32]); ss += double(v) * v;
        }
        const double inv = 1 / std::sqrt(ss / H + p.eps);
        for (unsigned c = 0; c < H; ++c) {
          const size_t index = size_t(ids[t + 2]) * H + c;
          const double ref = float(q.as<int8_t>()[index]) * float(sc.as<_Float16>()[index / 32]) * inv * (.75f + float(c % 9) / 16);
          if (!check(out.as<float>()[size_t(t) * H + c], ref, "embedding/norm", f32, size_t(t) * H + c)) return 1;
        }
      }
    }
  }
  std::printf("ink2 GGUF GPU: %zu outputs match CPU reference\n", checked);
  return 0;
}

int main() {
  ni::bench::GpuLock lock;
  if (!ni::bench::GpuLock::held_by_caller() && !lock.acquire(nullptr, 0)) {
    std::fprintf(stderr, "The GPU is busy; retry ink2_gguf_check after the current holder finishes.\n");
    return 2;
  }
  auto& gpu = ni::Gpu::get();
  std::vector<ni::Buffer> owned;
  const int rc = run_checks(gpu, owned);
  for (auto& b : owned) gpu.release(b);
  return rc;
}
