// The two Flash-Next prefill fusions against the dispatches they replace, bitwise, on small shapes (meant to run
// under the Metal validation layer too: MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1):
//   NI_HC_FUSE     qmt6_mix_m64_n64  vs  qmt6_m64_n64 (the up GEMM into fp32 m [T, S*D]) + hc_mix_h
//   NI_HC_PRECOMB  hc_pre_comb_h     vs  moe_combine (into a zeroed yb) + hc_pre_h, the shared rows ys separate
//                                        or the shared fold's tail of y
// Binds, params, grids and threads are src/model/qwen4_exp.cpp's enc_gr_prefill and qwen35.cpp's enc_qmm_large
// (qmt6 branch) and prefill MoE combine. Every buffer has its exact size, so an out-of-bounds access is the
// validation layer's to report; outputs start as one 0xff pattern, so an output still holding it was not written.
// Takes the bench GPU lock before its first dispatch (NI_LOCK_HELD=1 when the caller already holds it), the way
// tools/event_wait_probe.cpp does, so a bare run cannot sit beside someone's measurement.
//   hc_fold_check          (exit status 1 if any shape fails)
#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace ni;
namespace ni { extern const char* const kTensorKernelSource; }

namespace {

constexpr uint32_t S = 4;  // hc_pre / hc_pre_comb pack the streams in a float4
struct HcParams {          // qwen4.metal
  uint32_t R, D, S, has_y, has_inj;
  float eps;
};
struct MixParams {  // qmt6.metal's Qmt6MixParams
  uint32_t M, N, K, x_ld, D, S, u_ld;
};

// One case's buffers (released with it) filled from a deterministic generator
struct Arena {
  uint64_t seed;
  std::vector<Buffer> bufs;
  ~Arena() {
    for (Buffer& b : bufs) Gpu::get().release(b);
  }
  uint32_t u32() {
    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
    return uint32_t(seed >> 32);
  }
  float uni(float lo, float hi) { return lo + (hi - lo) * float(u32() >> 8) * (1.0f / 16777216.0f); }
  Buffer filled(size_t bytes, uint8_t v) {
    bufs.push_back(Gpu::get().alloc(bytes));
    memset(bufs.back().ptr(), v, bytes);
    return bufs.back();
  }
  Buffer copy(const Buffer& a) {
    Buffer b = filled(a.size, 0);
    memcpy(b.ptr(), a.ptr(), a.size);
    return b;
  }
  Buffer f32(size_t n, float lo, float hi) {
    Buffer b = filled(n * 4, 0);
    for (size_t i = 0; i < n; ++i) b.as<float>()[i] = uni(lo, hi);
    return b;
  }
  Buffer f16(size_t n, float lo, float hi) {
    Buffer b = filled(n * 2, 0);
    for (size_t i = 0; i < n; ++i) b.as<_Float16>()[i] = _Float16(uni(lo, hi));
    return b;
  }
  Buffer bf16(size_t n, float lo, float hi) {
    Buffer b = filled(n * 2, 0);
    for (size_t i = 0; i < n; ++i) b.as<uint16_t>()[i] = f32_to_bf16(uni(lo, hi));
    return b;
  }
  Buffer codes6(size_t words) {  // packed 6-bit codes: any bit pattern is a valid run of codes
    Buffer b = filled(words * 4, 0);
    for (size_t i = 0; i < words; ++i) b.as<uint32_t>()[i] = u32();
    return b;
  }
};

// "" when a and b (n values of T) are the same bits and b has no unwritten (all-0xff) value, else what differs
template <class T>
std::string diff(const char* what, const Buffer& a, const Buffer& b, size_t n) {
  T ones;
  memset(&ones, 0xff, sizeof(T));
  size_t unwritten = 0, ndiff = 0;
  double mx = 0;
  for (size_t i = 0; i < n; ++i) {
    unwritten += !memcmp(&b.as<T>()[i], &ones, sizeof(T));
    if (!memcmp(&a.as<T>()[i], &b.as<T>()[i], sizeof(T))) continue;
    ++ndiff;
    const double d = std::fabs(double(a.as<T>()[i]) - double(b.as<T>()[i]));
    mx = std::isnan(d) || d > mx ? d : mx;
  }
  if (!ndiff && !unwritten) return "";
  char s[160];
  snprintf(s, sizeof(s), " %s: %zu/%zu differ, max abs diff %g, %zu unwritten;", what, ndiff, n, mx, unwritten);
  return s;
}

bool report(const char* shape, const std::string& why) {
  printf("%-52s %s%s\n", shape, why.empty() ? "PASS" : "FAIL", why.c_str());
  return why.empty();
}

// NI_HC_FUSE: u [T, D] half from hc_act_h's half rows h [T, rank], the 6-bit up weights [S*D, rank] and Xn [T, S*D]
bool check_mix(uint32_t T, uint32_t D, uint32_t rank) {
  Gpu& g = Gpu::get();
  Arena a{T * 1000003ull + D * 7919ull + rank};
  const uint32_t N = S * D;
  const Buffer h = a.f16(size_t(T) * rank, -0.3f, 1.0f), w = a.codes6(size_t(N) * rank * 3 / 16),
               sc = a.bf16(size_t(N) * rank / 64, -0.02f, 0.02f), bi = a.bf16(size_t(N) * rank / 64, -0.1f, 0.1f),
               xn = a.f16(size_t(T) * N, -2.0f, 2.0f);
  const Buffer m = a.filled(size_t(T) * N * 4, 0xff), u_ref = a.filled(size_t(T) * D * 2, 0xff),
               u_fused = a.filled(size_t(T) * D * 2, 0xff);
  auto e = g.begin();
  e->use(g.pipeline("qmt6_m64_n64"));
  const kp::Qmm qp{T, N, rank, rank, N, 0, 64, 0, 0};
  e->buf(0, h); e->buf(1, w); e->buf(2, sc); e->buf(3, bi); e->buf(4, m);
  e->val(5, qp);
  e->dispatch((N + 63) / 64, (T + 63) / 64, 1, 128);
  e->use(g.pipeline("hc_mix_h"));
  const HcParams hp{T, D, S, 1, 1, 1e-6f};
  e->buf(0, m); e->buf(1, xn); e->buf(2, u_ref);
  e->val(3, hp); e->val(4, D);
  e->dispatch((D + 255) / 256, T, 1, 256);
  e->use(g.pipeline("qmt6_mix_m64_n64"));
  const MixParams mp{T, N, rank, rank, D, S, D};
  e->buf(0, h); e->buf(1, w); e->buf(2, sc); e->buf(3, bi); e->buf(4, xn); e->buf(5, u_fused);
  e->val(6, mp);
  e->dispatch(D / 64, (T + 63) / 64, 1, 128);
  e->commit_and_wait();
  char shape[96];
  snprintf(shape, sizeof(shape), "qmt6_mix_m64_n64 T=%u D=%u rank=%u", T, D, rank);
  return report(shape, diff<_Float16>("u", u_ref, u_fused, size_t(T) * D));
}

// NI_HC_PRECOMB: the gated residual's first part on X [T, S*D] with the previous MoE's combine sum over the k
// routed rows per token of y [T*k (+ T shared rows under the fold), D]
std::string check_precomb_case(uint32_t T, uint32_t D, uint32_t k, bool fold, bool has_inj) {
  Gpu& g = Gpu::get();
  Arena a{T * 1000003ull + D * 7919ull + k * 131ull + fold * 2 + has_inj};
  const size_t SD = size_t(S) * D;
  const Buffer x = a.f32(T * SD, -1.0f, 1.0f), inj_prev = a.f32(T * 4, 0.0f, 2.0f), norm = a.f32(SD, 0.5f, 1.5f),
               winj = a.bf16(S * SD, -0.05f, 0.05f), y = a.f32(size_t(T) * (k + fold) * D, -1.0f, 1.0f),
               ys = fold ? y.view(size_t(T) * k * D * 4, size_t(T) * D * 4) : a.f32(size_t(T) * D, -1.0f, 1.0f),
               wts = a.f32(size_t(T) * (k + 1), 0.0f, 0.5f);
  const Buffer x_ref = a.copy(x), x_fused = a.copy(x), yb = a.filled(size_t(T) * D * 4, 0);
  const Buffer xn_ref = a.filled(T * SD * 2, 0xff), xn_fused = a.filled(T * SD * 2, 0xff);
  const Buffer inj_ref = a.filled(T * 16, 0xff), inj_fused = a.filled(T * 16, 0xff);
  const Buffer& wi = has_inj ? winj : norm;  // enc_gr_prefill binds the norm when the residual has no injection
  const HcParams p{T, D, S, 1, has_inj, 1e-6f};
  const kp::Combine cp{D, k};
  auto e = g.begin();
  e->use(g.pipeline("moe_combine"));
  e->buf(0, y); e->buf(1, ys); e->buf(2, wts); e->buf(3, yb);
  e->val(4, cp);
  e->dispatch((D / 4 + 63) / 64, T, 1, 64);
  e->use(g.pipeline("hc_pre_h"));
  e->buf(0, x_ref); e->buf(1, yb); e->buf(2, inj_prev); e->buf(3, norm); e->buf(4, wi); e->buf(5, xn_ref);
  e->buf(6, inj_ref);
  e->val(7, p);
  e->dispatch(T, 1, 1, 1024);
  e->use(g.pipeline("hc_pre_comb_h"));
  e->buf(0, x_fused); e->buf(1, inj_prev); e->buf(2, norm); e->buf(3, wi); e->buf(4, xn_fused); e->buf(5, inj_fused);
  e->val(6, p);
  e->buf(7, y); e->buf(8, ys); e->buf(9, wts);
  e->val(10, cp);
  e->dispatch(T, 1, 1, 1024);
  e->commit_and_wait();
  std::string why = diff<float>("X", x_ref, x_fused, T * SD) + diff<_Float16>("Xn", xn_ref, xn_fused, T * SD);
  if (has_inj) why += diff<float>("inj", inj_ref, inj_fused, size_t(T) * 4);
  else if (memcmp(inj_ref.ptr(), inj_fused.ptr(), T * 16)) why += " inj written without has_inj;";
  // the invariant the fused kernel leans on: the pair it replaces leaves yb zero, and it never touches yb itself
  for (size_t i = 0; i < size_t(T) * D; ++i)
    if (yb.as<uint32_t>()[i]) {
      why += " yb not zero after hc_pre_h;";
      break;
    }
  return why;
}

bool check_precomb(uint32_t T, uint32_t D, bool fold) {
  std::string why;
  for (uint32_t k = 1; k <= 8; ++k)
    for (bool has_inj : {false, true})
      if (std::string w = check_precomb_case(T, D, k, fold, has_inj); !w.empty())
        why += " [k=" + std::to_string(k) + " inj=" + std::to_string(has_inj) + "]" + w;
  char shape[96];
  snprintf(shape, sizeof(shape), "hc_pre_comb_h T=%u D=%u ys=%s k=1..8 inj=0,1", T, D, fold ? "fold" : "separate");
  return report(shape, why);
}

}  // namespace

int main() {
  bench::GpuLock lock;  // a bare run must not land beside a measurement: BENCH_GPU_LOCK, else
                        // /tmp/kapitsa-gpu.lock (NI_LOCK_HELD=1 when the caller holds it), as event_wait_probe
  if (!bench::GpuLock::held_by_caller() && !lock.acquire()) fatal("cannot acquire GPU lock");
  Gpu& g = Gpu::get();
  g.compile_extra("tensor", kTensorKernelSource, true);  // qmt6_*: the tensor_ops library
  printf("%s\n", g.name().c_str());
  // T < 64 never fuses (enc_gr_prefill's T >= 64, as the qmt6 branch's M >= 64). T <= NI_SHORTPF_MAX (default 128)
  // takes the short-prefill detour instead, so 64..127 fuse only under NI_SHORTPF=0 or a lower NI_SHORTPF_MAX;
  // 129 and 256 are a partial and a full last row tile the default settings reach.
  for (uint32_t T : {1u, 63u}) printf("qmt6_mix_m64_n64 T=%u: SKIP (never fused: T < 64)\n", T);
  int fails = 0;
  for (uint32_t T : {64u, 65u, 100u, 127u, 129u, 256u})
    for (uint32_t D : {64u, 128u})
      for (uint32_t rank : {64u, 320u}) fails += !check_mix(T, D, rank);
  for (uint32_t T : {64u, 65u, 100u, 127u})
    for (uint32_t D : {64u, 3968u})
      for (bool fold : {false, true}) fails += !check_precomb(T, D, fold);
  printf("%d failing shape(s)\n", fails);
  return fails ? 1 : 0;
}
