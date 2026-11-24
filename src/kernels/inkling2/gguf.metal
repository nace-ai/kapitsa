#include "gguf_blocks.h"

struct Ink2GgufMmP { uint M, N, K, type, row_bytes, add; };
struct Ink2GgufMoeP { uint T, H, I, top, type, row_bytes; };
struct Ink2GgufEmbedP { uint T, H, tok_off, V, norm_f32; float eps; uint type, row_bytes; };

[[kernel]] void ink2_gguf_mm(const device float* x [[buffer(0)]], const device uchar* w [[buffer(1)]],
                            device float* y [[buffer(2)]], constant Ink2GgufMmP& p [[buffer(3)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg.x * 4 + sg, m = tg.y;
  if (row >= p.N || m >= p.M) return;
  const device uchar* wr = w + size_t(row) * p.row_bytes;
  float acc = 0.0f;
  for (uint k = lane; k < p.K; k += 32) acc += x[size_t(m) * p.K + k] * ink2_gguf::value(wr, p.type, k);
  acc = simd_sum(acc);
  if (lane == 0) {
    const size_t out = size_t(m) * p.N + row;
    y[out] = acc + (p.add ? y[out] : 0.0f);
  }
}

[[kernel]] void ink2_gguf_gu(const device float* x [[buffer(0)]], const device uint* ids [[buffer(1)]],
                            const device uchar* w [[buffer(2)]], device float* act [[buffer(3)]],
                            constant Ink2GgufMoeP& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint j = tg.x * 4 + sg, slot = tg.y, t = tg.z;
  if (j >= p.I || slot >= p.top || t >= p.T) return;
  const uint expert = ids[size_t(t) * p.top + slot];
  const device uchar* gate = w + (size_t(expert) * (2 * p.I) + j) * p.row_bytes;
  const device uchar* up = gate + size_t(p.I) * p.row_bytes;
  float g = 0.0f, u = 0.0f;
  for (uint k = lane; k < p.H; k += 32) {
    const float v = x[size_t(t) * p.H + k];
    g += v * ink2_gguf::value(gate, p.type, k);
    u += v * ink2_gguf::value(up, p.type, k);
  }
  g = simd_sum(g); u = simd_sum(u);
  if (lane == 0)
    act[(size_t(t) * p.top + slot) * p.I + j] =
        (4.0f * metal::precise::tanh(g * 0.25f)) * (1.0f / (1.0f + metal::precise::exp(-g))) *
        (25.0f * metal::precise::tanh(u * 0.04f));
}

[[kernel]] void ink2_gguf_down(const device float* act [[buffer(0)]], const device uint* ids [[buffer(1)]],
                              const device float* weights [[buffer(2)]], const device uchar* w [[buffer(3)]],
                              device float* out [[buffer(4)]], constant Ink2GgufMoeP& p [[buffer(5)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg.x * 4 + sg, t = tg.y;
  if (row >= p.H || t >= p.T) return;
  float total = 0.0f;
  for (uint slot = 0; slot < p.top; ++slot) {
    const uint expert = ids[size_t(t) * p.top + slot];
    const device uchar* wr = w + (size_t(expert) * p.H + row) * p.row_bytes;
    const device float* a = act + (size_t(t) * p.top + slot) * p.I;
    float acc = 0.0f;
    for (uint k = lane; k < p.I; k += 32) acc += a[k] * ink2_gguf::value(wr, p.type, k);
    total += weights[size_t(t) * p.top + slot] * simd_sum(acc);
  }
  if (lane == 0) out[size_t(t) * p.H + row] = total;
}

[[kernel]] void ink2_gguf_embed(const device int* tokens [[buffer(0)]], const device char* q [[buffer(1)]],
                               const device half* scales [[buffer(2)]], const device uchar* norm [[buffer(3)]],
                               device float* out [[buffer(4)]], constant Ink2GgufEmbedP& p [[buffer(5)]],
                               uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                               uint nt [[threads_per_threadgroup]]) {
  threadgroup float scratch[512];
  const int id = tokens[p.tok_off + t];
  const uint token = id >= 0 && uint(id) < p.V ? uint(id) : 0u;
  float sum = 0.0f;
  for (uint c = tid; c < p.H; c += nt) {
    const float v = p.row_bytes
        ? ink2_gguf::value((const device uchar*)q + size_t(token) * p.row_bytes, p.type, c)
        : float(q[size_t(token) * p.H + c]) * float(scales[size_t(token) * (p.H / 32) + c / 32]);
    out[size_t(t) * p.H + c] = v;
    sum += v * v;
  }
  scratch[tid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = nt / 2; stride; stride /= 2) {
    if (tid < stride) scratch[tid] += scratch[tid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float inv = metal::precise::rsqrt(scratch[0] / float(p.H) + p.eps);
  for (uint c = tid; c < p.H; c += nt) {
    const float n = p.norm_f32 ? ((const device float*)norm)[c] : float(((const device bfloat*)norm)[c]);
    out[size_t(t) * p.H + c] *= inv * n;
  }
}
