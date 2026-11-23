#include "common.h"

struct Ink2DraftVecP { uint n, mode; };
[[kernel]] void ink2_draft_vec(device float* x [[buffer(0)]], const device float* y [[buffer(1)]],
                              device float* residual [[buffer(2)]], constant Ink2DraftVecP& p [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  if (p.mode == 0) x[i] = float(bfloat(x[i]));
  else if (p.mode == 1) x[i] += y[i];
  else if (p.mode == 2) { x[i] = float(bfloat(x[i])) + y[i]; residual[i] = float(bfloat(x[i])); }
  else if (p.mode == 3) x[i] = 0;
  else x[i] = float(bfloat((x[i] / (1.0f + exp(-x[i]))) * y[i]));
}

struct Ink2DraftMixP { uint rows, first, count, H, weight_row; };
[[kernel]] void ink2_draft_mix(const device float* taps [[buffer(0)]], const device float* blend [[buffer(1)]],
                              device float* mixed [[buffer(2)]], constant Ink2DraftMixP& p [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
  if (i >= p.count * p.H) return;
  const uint row = i / p.H, c = i % p.H;
  float sum = 0;
  for (uint t = 0; t < 4; ++t)
    sum += float(bfloat(taps[(size_t(t) * p.rows + p.first + row) * p.H + c])) * blend[p.weight_row * 4 + t];
  mixed[i] = sum;
}

struct Ink2DraftInitP { uint H, token; float eps; };
[[kernel]] void ink2_draft_init(const device bfloat* embed [[buffer(0)]], const device bfloat* gain [[buffer(1)]],
                               const device bfloat* mask [[buffer(2)]], device float* hidden [[buffer(3)]],
                               constant Ink2DraftInitP& p [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
                               uint row [[threadgroup_position_in_grid]]) {
  threadgroup float scratch[32];
  const device bfloat* src = row == 0 ? embed + size_t(p.token) * p.H : mask;
  float ss = 0;
  if (row == 0) for (uint c = tid; c < p.H; c += 256) ss += float(src[c]) * float(src[c]);
  const float inv = row == 0 ? precise::rsqrt(tg_sum(ss, scratch, tid, 256) / float(p.H) + p.eps) : 1.0f;
  for (uint c = tid; c < p.H; c += 256)
    hidden[size_t(row) * p.H + c] = float(bfloat(float(src[c]) * inv * (row == 0 ? float(gain[c]) : 1.0f)));
}

struct Ink2DraftRopeP { uint rows, heads, D, pos; float eps, theta; };
[[kernel]] void ink2_draft_rope(const device float* x [[buffer(0)]], const device bfloat* gain [[buffer(1)]],
                               device float* out [[buffer(2)]], constant Ink2DraftRopeP& p [[buffer(3)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const size_t base = (size_t(tg.y) * p.heads + tg.x) * p.D;
  float ss = 0;
  for (uint d = lane; d < p.D; d += 32) ss += x[base + d] * x[base + d];
  const float inv = precise::rsqrt(simd_sum(ss) / float(p.D) + p.eps);
  for (uint d = lane; d < p.D / 2; d += 32) {
    const float a = float(bfloat(x[base + d] * inv * float(gain[d])));
    const float b = float(bfloat(x[base + d + p.D / 2] * inv * float(gain[d + p.D / 2])));
    const float angle = float(p.pos + tg.y) * pow(p.theta, -2.0f * float(d) / float(p.D));
    const float cs = cos(angle), sn = sin(angle);
    out[base + d] = float(bfloat(a * cs - b * sn));
    out[base + d + p.D / 2] = float(bfloat(a * sn + b * cs));
  }
}

struct Ink2DraftStoreP { uint rows, width, pos, cap; };
[[kernel]] void ink2_draft_store(const device float* src [[buffer(0)]], device bfloat* dst [[buffer(1)]],
                                constant Ink2DraftStoreP& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < p.rows * p.width) dst[size_t((p.pos + i / p.width) % p.cap) * p.width + i % p.width] = bfloat(src[i]);
}

struct Ink2DraftAttnP { uint pos, cap, window, Hq, Hkv, D, rows; };
[[kernel]] void ink2_draft_attn(const device float* q [[buffer(0)]], const device float* bk [[buffer(1)]],
                               const device float* bv [[buffer(2)]], const device bfloat* ck [[buffer(3)]],
                               const device bfloat* cv [[buffer(4)]], device float* out [[buffer(5)]],
                               constant Ink2DraftAttnP& p [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, row = tg.y, kvhead = head / (p.Hq / p.Hkv);
  const uint start = p.window && p.pos > p.window ? p.pos - p.window : 0;
  const uint ctx = p.pos - start;
  const size_t qb = (size_t(row) * p.Hq + head) * p.D;
  float4 query, acc(0.0f);
  for (uint j = 0; j < 4; ++j) query[j] = q[qb + lane + j * 32];
  float maxv = -INFINITY, denom = 0;
  for (uint t = 0; t < ctx + p.rows; ++t) {
    const bool context = t < ctx;
    const uint slot = context ? (start + t) % p.cap : t - ctx;
    const size_t b = (size_t(slot) * p.Hkv + kvhead) * p.D;
    float4 key, value;
    for (uint j = 0; j < 4; ++j) {
      key[j] = context ? float(ck[b + lane + j * 32]) : bk[b + lane + j * 32];
      value[j] = context ? float(cv[b + lane + j * 32]) : bv[b + lane + j * 32];
    }
    const float score = simd_sum(dot(query, key)) * rsqrt(float(p.D));
    const float next = max(maxv, score), old = exp(maxv - next), prob = exp(score - next);
    acc = acc * old + value * prob;
    denom = denom * old + prob;
    maxv = next;
  }
  for (uint j = 0; j < 4; ++j) out[qb + lane + j * 32] = float(bfloat(acc[j] / denom));
}

struct Ink2DraftFoldP { uint rows, cols, first, vocab; };
[[kernel]] void ink2_draft_fold(const device float* src [[buffer(0)]], device bfloat* dst [[buffer(1)]],
                               constant Ink2DraftFoldP& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < p.rows * p.cols) {
    const uint r = i / p.cols, c = i % p.cols;
    dst[size_t(p.first + c) * p.rows + r] = bfloat(src[size_t(r) * p.cols + c]);
  }
}

struct Ink2DraftRnnP { uint H, rank, step, vocab; };
[[kernel]] void ink2_draft_condition(const device float* hidden [[buffer(0)]], const device float* last [[buffer(1)]],
                                    device float* input [[buffer(2)]], constant Ink2DraftRnnP& p [[buffer(3)]],
                                    uint i [[thread_position_in_grid]]) {
  if (i < 7 * 2 * p.H) {
    const uint r = i / (2 * p.H), c = i % (2 * p.H);
    input[i] = c < p.H ? hidden[size_t(r + 1) * p.H + c] : last[c - p.H];
  }
}
[[kernel]] void ink2_draft_rnn_input(const device float* state [[buffer(0)]], const device bfloat* embed [[buffer(1)]],
                                    const device int* picks [[buffer(2)]], device float* input [[buffer(3)]],
                                    constant Ink2DraftRnnP& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i < 2 * p.rank) {
    const int id = picks[p.step];
    input[i] = i < p.rank ? state[i] : float(embed[size_t(uint(id) < p.vocab ? id : 0) * p.rank + i - p.rank]);
  }
}
[[kernel]] void ink2_draft_rnn_update(const device float* recurrent [[buffer(0)]], const device float* condition [[buffer(1)]],
                                     const device bfloat* bias [[buffer(2)]], device float* state [[buffer(3)]],
                                     device float* signal [[buffer(4)]], constant Ink2DraftRnnP& p [[buffer(5)]],
                                     uint i [[thread_position_in_grid]]) {
  if (i >= p.rank) return;
  float v[3];
  for (uint k = 0; k < 3; ++k) {
    const uint c = k * p.rank + i;
    const float conditioned = condition[size_t(p.step) * 3 * p.rank + c] + float(bias[c]);
    v[k] = float(bfloat(recurrent[c] + conditioned));
  }
  const float gate = float(bfloat(1.0f / (1.0f + exp(-v[0]))));
  const float candidate = float(bfloat(tanh(v[1])));
  state[i] = float(bfloat(float(bfloat(gate * state[i])) +
                          float(bfloat(float(bfloat(1.0f - gate)) * candidate))));
  signal[i] = float(bfloat(tanh(v[2])));
}
[[kernel]] void ink2_draft_scores(const device float* base [[buffer(0)]], const device bfloat* folded [[buffer(1)]],
                                 const device float* signal [[buffer(2)]], device float* scores [[buffer(3)]],
                                 constant Ink2DraftRnnP& p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint token = tg * 4 + sg;
  if (token >= p.vocab) return;
  const device bfloat* w = folded + size_t(token) * (p.rank + 1);
  float sum = 0;
  for (uint c = lane; c < p.rank; c += 32) sum += signal[c] * float(w[c]);
  sum = simd_sum(sum);
  if (lane == 0) {
    const float bias = float(bfloat(sum + float(w[p.rank])));
    scores[token] = float(bfloat(base[size_t(p.step) * p.vocab + token] + bias));
  }
}
