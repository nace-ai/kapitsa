#include "model/ink2_draft.h"
#include "model/ink2_draft_plan.h"
#include "model/inkling2.h"
#include "model/inkling2_kernels.h"
#include "model/kparams.h"
#include "model/sampler.h"
#include "util/bf16.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace ni {
namespace {
using namespace ink2draft;
struct VecP { uint32_t n, mode; };
struct MixP { uint32_t rows, first, count, hidden, weight_row; };
struct InitP { uint32_t hidden, token; float eps; };
struct RopeP { uint32_t rows, heads, dim, pos; float eps, theta; };
struct StoreP { uint32_t rows, width, pos, cap; };
struct AttnP { uint32_t pos, cap, window, hq, hkv, dim, rows; };
struct FoldP { uint32_t rows, cols, first, vocab; };
struct RnnP { uint32_t hidden, rank, step, vocab; };
struct MmP { uint32_t M, N, K, x_ld, y_ld, w_f32, add; };
struct NormP { uint32_t rows, hidden, has_weight, w_f32; float eps; };

float scalar(const Tensor& t, size_t i) {
  if (t.dtype == DType::BF16) return bf16_to_f32(t.buf.as<uint16_t>()[i]);
  if (t.dtype == DType::F32) return t.buf.as<float>()[i];
  throw std::runtime_error("ink2 draft: expected BF16/F32 weights");
}
void check(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(std::string("ink2 draft: ") + what);
}

}

struct Ink2DraftSeq::Impl {
  struct Layer {
    Buffer keys, values;
    uint32_t cap = 0;
  };
  std::array<Layer, 3> layers;
  uint32_t pos = 0;
  Buffer last;
  std::vector<Buffer> retired;

  void release_retired() {
    for (auto& b : retired) Gpu::get().release(b);
    retired.clear();
  }
  ~Impl() {
    release_retired();
    for (auto& layer : layers) {
      Gpu::get().release(layer.keys);
      Gpu::get().release(layer.values);
    }
    Gpu::get().release(last);
  }
};

struct Ink2Draft::Impl {
  WeightStore weights;
  const Tensor& embedding;
  const Tensor& embed_norm;
  const Ink2Mat& unembed;
  std::vector<Buffer> owned;
  std::unordered_map<std::string, Pipeline> pipelines;
  struct Layer {
    const Tensor *mix_norm, *kc, *vc, *knc, *q, *k, *v, *qn, *kn, *o, *in_norm, *post_norm, *gate, *up, *down;
    uint32_t window = 0;
  };
  std::array<Layer, 3> layers;
  SharedLayout layout{1};
  uint32_t vocab = 0;
  float eps = 1e-6f, theta = 1e6f;
  Buffer blend, mixed, hidden, residual, summed, normed, q, k, v, rotated, attended, projection;
  Buffer gate, up, logits, scores, folded, condition_input, condition, recurrent_input, recurrent, state, signal;
  Buffer picks, cand_v, cand_i, probability;
  Tensor rnn_condition, rnn_recurrent;
  const Tensor *output_norm = nullptr, *mask = nullptr, *markov_embed = nullptr, *rnn_bias = nullptr, *markov_norm = nullptr;

  Buffer alloc(size_t bytes, const char* name) {
    Buffer b = Gpu::get().alloc(bytes, name);
    owned.push_back(b);
    return b;
  }
  Buffer scratch(Scratch id) { return alloc(layout[id], "ink2_draft_scratch"); }
  Pipeline pipe(const std::string& name) {
    auto it = pipelines.find(name);
    if (it != pipelines.end()) return it->second;
    auto p = name.starts_with("ink2_") ? ink2::pipeline(name) : Gpu::get().pipeline(name);
    pipelines.emplace(name, p);
    return p;
  }
  const Tensor* tensor(const std::string& name, std::vector<int64_t> shape) {
    const auto& t = weights.get(name);
    check(t.shape == shape && t.dtype == DType::BF16, name.c_str());
    return &t;
  }
  void vec(Encoder& e, Buffer x, Buffer y, Buffer res, uint32_t n, uint32_t mode) {
    e.use(pipe("ink2_draft_vec")); e.buf(0, x); e.buf(1, y); e.buf(2, res);
    e.val(3, VecP{n, mode}); e.dispatch((n + 255) / 256, 1, 1, 256);
  }
  void round(Encoder& e, Buffer x, uint32_t n) { vec(e, x, x, x, n, 0); }
  void copy(Encoder& e, Buffer src, Buffer dst, size_t bytes) {
    e.use(pipe("ink2_copy_bytes")); e.buf(0, src); e.buf(1, dst); e.val(2, uint64_t(bytes));
    e.dispatch(uint32_t((bytes + 4095) / 4096), 1, 1, 256);
  }
  void mm(Encoder& e, const Tensor& w, Buffer x, Buffer y, uint32_t rows, bool rounded = true, bool full_float = false) {
    const uint32_t n = uint32_t(w.shape[0]), kk = uint32_t(w.shape[1]);
    check(kk % 4 == 0, "unaligned matrix inner dimension");
    MmP p{rows, n, kk, kk, n, w.dtype == DType::F32 ? 1u : 0u, 0};
    const bool tiled = rows > 32 && w.dtype == DType::BF16 && !full_float;
    e.use(pipe(tiled ? "ink2_bf16_mm" : "ink2_bf16_mv"));
    e.buf(0, x); e.buf(1, w.buf); e.buf(2, y); e.val(3, p);
    e.dispatch((n + (tiled ? 63 : 3)) / (tiled ? 64 : 4), (rows + (tiled ? 31 : 7)) / (tiled ? 32 : 8), 1, 128);
    if (rounded) round(e, y, rows * n);
  }
  void head(Encoder& e, Buffer x, Buffer y, uint32_t rows, uint32_t first, uint32_t count, bool full_float = false) {
    if (!unembed.is_q8()) {
      check(unembed.raw != nullptr, "unembedding has no weights");
      Tensor w = *unembed.raw;
      w.buf = w.buf.view(size_t(first) * H * 2, size_t(count) * H * 2);
      w.shape = {count, H};
      mm(e, w, x, y, rows, false, full_float);
      return;
    }
    MmP p{rows, count, H, H, count, 0, 0};
    const bool tiled = rows > 32 && !full_float;
    e.use(pipe(tiled ? "ink2_q8_mm" : "ink2_q8_mv"));
    e.buf(0, x); e.buf(1, unembed.q.view(size_t(first) * H, size_t(count) * H));
    e.buf(2, unembed.s.view(size_t(first) * (H / 32) * 2, size_t(count) * (H / 32) * 2));
    e.buf(3, y); e.val(4, p);
    e.dispatch((count + (tiled ? 63 : 3)) / (tiled ? 64 : 4), (rows + (tiled ? 31 : 7)) / (tiled ? 32 : 8), 1, 128);
  }
  void norm(Encoder& e, Buffer src, const Tensor& w, Buffer dst, uint32_t rows, uint32_t dim) {
    e.use(pipe("ink2_rmsnorm")); e.buf(0, src); e.buf(1, w.buf); e.buf(2, dst);
    e.val(3, NormP{rows, dim, 1, w.dtype == DType::F32 ? 1u : 0u, eps});
    e.dispatch(rows, 1, 1, 512);
    round(e, dst, rows * dim);
  }
  void rope(Encoder& e, Buffer src, const Tensor& gain, Buffer dst, uint32_t rows, uint32_t heads, uint32_t start) {
    e.use(pipe("ink2_draft_rope")); e.buf(0, src); e.buf(1, gain.buf); e.buf(2, dst);
    e.val(3, RopeP{rows, heads, D, start, eps, theta}); e.dispatch(heads, rows, 1, 32);
  }
  void store(Encoder& e, Buffer src, Buffer dst, uint32_t rows, uint32_t start, uint32_t cap) {
    e.use(pipe("ink2_draft_store")); e.buf(0, src); e.buf(1, dst);
    e.val(2, StoreP{rows, HKV * D, start, cap}); e.dispatch((rows * HKV * D + 255) / 256, 1, 1, 256);
  }
  void reserve(Encoder& e, Ink2DraftSeq::Impl& d, uint32_t need) {
    for (uint32_t i = 0; i < layers.size(); ++i) {
      auto& layer = d.layers[i];
      if (layer.cap >= need || layers[i].window) continue;
      const auto growth = ink2draft::growth(layer.cap, need, Gpu::get().max_buffer_length());
      const uint32_t cap = growth.cap;
      d.retired.reserve(d.retired.size() + 2);
      Buffer keys = Gpu::get().alloc(growth.one_buffer, "ink2_draft_global_k");
      Buffer values = Gpu::get().alloc(growth.one_buffer, "ink2_draft_global_v");
      if (d.pos) {
        copy(e, layer.keys, keys, size_t(d.pos) * HKV * D * 2);
        copy(e, layer.values, values, size_t(d.pos) * HKV * D * 2);
      }
      // Encoders use unretained references; propose releases these after its fence.
      if (layer.keys) d.retired.push_back(layer.keys);
      if (layer.values) d.retired.push_back(layer.values);
      layer.keys = keys; layer.values = values; layer.cap = cap;
    }
  }
  void fold_head() {
    const Tensor* w = tensor("markov_head.hidden_out.weight", {H, R});
    const Tensor* b = tensor("markov_head.hidden_out.bias", {H});
    Buffer transposed = scratch(Scratch::fold_transposed);
    for (uint32_t r = 0; r <= R; ++r)
      for (uint32_t h = 0; h < H; ++h)
        transposed.as<float>()[size_t(r) * H + h] = r == R ? scalar(*b, h) : scalar(*w, size_t(h) * R + r);
    Buffer tile = scratch(Scratch::fold_tile);
    folded = scratch(Scratch::folded);
    auto e = Gpu::get().begin();
    for (uint32_t first = 0; first < vocab; first += 4096) {
      const uint32_t n = std::min(4096u, vocab - first);
      head(*e, transposed, tile, R + 1, first, n, true);
      e->use(pipe("ink2_draft_fold")); e->buf(0, tile); e->buf(1, folded);
      e->val(2, FoldP{R + 1, n, first, vocab}); e->dispatch(((R + 1) * n + 255) / 256, 1, 1, 256);
      e->split();
    }
    e->commit_and_wait();
  }

  Impl(const std::string& path, const Tensor& emb, const Tensor& gain, const Ink2Mat& out)
      : embedding(emb), embed_norm(gain), unembed(out) {
    const auto plan = read_plan(path);
    layout = plan.layout; vocab = plan.vocab; eps = plan.eps; theta = plan.theta;
    const size_t limit = Gpu::get().max_buffer_length();
    fits(plan.weights.bytes, limit);
    layout.check_limit(limit);
    fits(last_bytes, limit);
    for (size_t i = 0; i < 3; ++i) fits(kv_bytes(window(i)), limit);
    check(vocab > 0 && embedding.dtype == DType::BF16 && embedding.shape.size() == 2 &&
          embedding.shape[1] == H && embedding.shape[0] == plan.padded && embed_norm.dtype == DType::BF16 &&
          embed_norm.numel() == H, "target embedding shape/dtype");
    check(unembed.is_q8() ? (unembed.K == H && unembed.N >= vocab) :
          (unembed.raw && unembed.raw->dtype == DType::BF16 && unembed.raw->shape.size() == 2 && unembed.raw->shape[1] == H &&
           unembed.raw->shape[0] >= vocab), "target unembedding shape/dtype");
    weights.open_dir(path, nullptr, [](const std::string& name) { return name == "mtp.safetensors"; });
    const Tensor* fuse = tensor("fuse_weight", {4, 4});
    blend = scratch(Scratch::blend);
    for (uint32_t r = 0; r < 4; ++r) {
      float mx = -std::numeric_limits<float>::infinity(), denom = 0;
      for (uint32_t t = 0; t < 4; ++t) mx = std::max(mx, scalar(*fuse, r * 4 + t));
      for (uint32_t t = 0; t < 4; ++t) denom += std::exp(scalar(*fuse, r * 4 + t) - mx);
      for (uint32_t t = 0; t < 4; ++t)
        blend.as<float>()[r * 4 + t] = bf16_to_f32(f32_to_bf16(std::exp(scalar(*fuse, r * 4 + t) - mx) / denom));
    }
    output_norm = tensor("norm.weight", {H}); mask = tensor("mask_embed", {H});
    for (uint32_t i = 0; i < layers.size(); ++i) {
      auto& l = layers[i];
      const std::string s = "layers." + std::to_string(i) + ".";
      l.mix_norm = tensor(s + "dflare_norm.weight", {H});
      l.kc = tensor(s + "self_attn.k_proj_ctx.weight", {HKV * D, H});
      l.vc = tensor(s + "self_attn.v_proj_ctx.weight", {HKV * D, H});
      l.knc = tensor(s + "self_attn.k_norm_ctx.weight", {D});
      l.q = tensor(s + "self_attn.q_proj.weight", {HQ * D, H});
      l.k = tensor(s + "self_attn.k_proj.weight", {HKV * D, H});
      l.v = tensor(s + "self_attn.v_proj.weight", {HKV * D, H});
      l.qn = tensor(s + "self_attn.q_norm.weight", {D}); l.kn = tensor(s + "self_attn.k_norm.weight", {D});
      l.o = tensor(s + "self_attn.o_proj.weight", {H, HQ * D});
      l.in_norm = tensor(s + "input_layernorm.weight", {H});
      l.post_norm = tensor(s + "post_attention_layernorm.weight", {H});
      l.gate = tensor(s + "mlp.gate_proj.weight", {I, H}); l.up = tensor(s + "mlp.up_proj.weight", {I, H});
      l.down = tensor(s + "mlp.down_proj.weight", {H, I});
      l.window = window(i);
    }
    markov_embed = tensor("markov_head.markov_w1.weight", {embedding.shape[0], R});
    markov_norm = tensor("markov_head.dflare_norm.weight", {H});
    rnn_bias = tensor("markov_head.rnn_proj.bias", {3 * R});
    const Tensor* rw = tensor("markov_head.rnn_proj.weight", {3 * R, 2 * R + 2 * H});
    rnn_condition = Tensor{scratch(Scratch::rnn_condition), DType::F32, {3 * R, 2 * H}};
    rnn_recurrent = Tensor{scratch(Scratch::rnn_recurrent), DType::F32, {3 * R, 2 * R}};
    for (uint32_t r = 0; r < 3 * R; ++r) {
      for (uint32_t c0 = 0; c0 < 2 * R; ++c0)
        rnn_recurrent.buf.as<float>()[size_t(r) * 2 * R + c0] = scalar(*rw, size_t(r) * (2 * R + 2 * H) + c0);
      for (uint32_t c0 = 0; c0 < 2 * H; ++c0)
        rnn_condition.buf.as<float>()[size_t(r) * 2 * H + c0] = scalar(*rw, size_t(r) * (2 * R + 2 * H) + 2 * R + c0);
    }
    mixed = scratch(Scratch::mixed);
    hidden = scratch(Scratch::hidden); residual = scratch(Scratch::residual); summed = scratch(Scratch::summed); normed = scratch(Scratch::normed);
    q = scratch(Scratch::q); k = scratch(Scratch::k); v = scratch(Scratch::v);
    rotated = scratch(Scratch::rotated); attended = scratch(Scratch::attended); projection = scratch(Scratch::projection);
    gate = scratch(Scratch::gate); up = scratch(Scratch::up); logits = scratch(Scratch::logits); scores = scratch(Scratch::scores);
    condition_input = scratch(Scratch::condition_input); condition = scratch(Scratch::condition); recurrent_input = scratch(Scratch::recurrent_input);
    recurrent = scratch(Scratch::recurrent); state = scratch(Scratch::state); signal = scratch(Scratch::signal);
    picks = scratch(Scratch::picks);
    cand_v = scratch(Scratch::cand_v); cand_i = scratch(Scratch::cand_i); probability = scratch(Scratch::probability);
    fold_head();
  }
  ~Impl() { for (auto& b : owned) Gpu::get().release(b); }

  void observe(Encoder& e, Ink2DraftSeq::Impl& d, const Buffer& captures, int rows, int start) {
    check(rows > 0 && start >= 0 && uint32_t(start) == d.pos, "observe must append committed rows");
    check(captures && captures.size >= size_t(4) * size_t(rows) * H * 4, "capture buffer too small");
    reserve(e, d, append_end(d.pos, rows));
    for (uint32_t first = 0; first < uint32_t(rows); first += CHUNK) {
      const uint32_t count = std::min(CHUNK, uint32_t(rows) - first);
      for (uint32_t layer = 0; layer < layers.size(); ++layer) {
        auto& l = layers[layer];
        auto& context = d.layers[layer];
        e.use(pipe("ink2_draft_mix")); e.buf(0, captures); e.buf(1, blend); e.buf(2, mixed);
        e.val(3, MixP{uint32_t(rows), first, count, H, layer}); e.dispatch((count * H + 255) / 256, 1, 1, 256);
        norm(e, mixed, *l.mix_norm, mixed, count, H);
        mm(e, *l.kc, mixed, k, count); mm(e, *l.vc, mixed, v, count);
        rope(e, k, *l.knc, rotated, count, HKV, d.pos + first);
        store(e, rotated, context.keys, count, d.pos + first, context.cap); store(e, v, context.values, count, d.pos + first, context.cap);
      }
    }
    e.use(pipe("ink2_draft_mix")); e.buf(0, captures); e.buf(1, blend); e.buf(2, d.last);
    e.val(3, MixP{uint32_t(rows), uint32_t(rows - 1), 1, H, 3}); e.dispatch((H + 255) / 256, 1, 1, 256);
    norm(e, d.last, *markov_norm, d.last, 1, H);
    d.pos += uint32_t(rows);
  }

  std::vector<int> propose(Ink2DraftSeq::Impl& d, int anchor, int count) {
    check(d.pos > 0 && anchor >= 0 && uint32_t(anchor) < vocab && count >= 1 && count <= 7, "invalid proposal request");
    picks.as<int>()[0] = anchor;
    auto enc = Gpu::get().begin(); auto& e = *enc;
    e.use(pipe("ink2_draft_init")); e.buf(0, embedding.buf); e.buf(1, embed_norm.buf); e.buf(2, mask->buf); e.buf(3, hidden);
    e.val(4, InitP{H, uint32_t(anchor), eps}); e.dispatch(WIDTH, 1, 1, 256);
    vec(e, residual, residual, residual, WIDTH * H, 3);
    for (uint32_t layer = 0; layer < layers.size(); ++layer) {
      auto& l = layers[layer];
      auto& context = d.layers[layer];
      copy(e, hidden, summed, WIDTH * H * 4); vec(e, summed, residual, residual, WIDTH * H, 1);
      norm(e, summed, *l.in_norm, normed, WIDTH, H);
      mm(e, *l.q, normed, q, WIDTH); mm(e, *l.k, normed, k, WIDTH); mm(e, *l.v, normed, v, WIDTH);
      rope(e, q, *l.qn, q, WIDTH, HQ, d.pos); rope(e, k, *l.kn, rotated, WIDTH, HKV, d.pos);
      e.use(pipe("ink2_draft_attn")); e.buf(0, q); e.buf(1, rotated); e.buf(2, v);
      e.buf(3, context.keys); e.buf(4, context.values); e.buf(5, attended);
      e.val(6, AttnP{d.pos, context.cap, l.window, HQ, HKV, D, WIDTH}); e.dispatch(HQ, WIDTH, 1, 32);
      mm(e, *l.o, attended, projection, WIDTH); vec(e, summed, projection, residual, WIDTH * H, 2);
      norm(e, summed, *l.post_norm, normed, WIDTH, H);
      mm(e, *l.gate, normed, gate, WIDTH); mm(e, *l.up, normed, up, WIDTH);
      vec(e, gate, up, gate, WIDTH * I, 4); mm(e, *l.down, gate, hidden, WIDTH);
    }
    copy(e, hidden, summed, WIDTH * H * 4); vec(e, summed, residual, residual, WIDTH * H, 1);
    norm(e, summed, *output_norm, hidden, WIDTH, H);
    head(e, hidden.view(H * 4, 7 * H * 4), logits, 7, 0, vocab); round(e, logits, 7 * vocab);
    RnnP rp{H, R, 0, vocab};
    e.use(pipe("ink2_draft_condition")); e.buf(0, hidden); e.buf(1, d.last); e.buf(2, condition_input); e.val(3, rp);
    e.dispatch((7 * 2 * H + 255) / 256, 1, 1, 256);
    mm(e, rnn_condition, condition_input, condition, 7, false);
    vec(e, state, state, state, R, 3);
    for (int step = 0; step < count; ++step) {
      rp.step = uint32_t(step);
      e.use(pipe("ink2_draft_rnn_input")); e.buf(0, state); e.buf(1, markov_embed->buf); e.buf(2, picks); e.buf(3, recurrent_input);
      e.val(4, rp); e.dispatch((2 * R + 255) / 256, 1, 1, 256);
      mm(e, rnn_recurrent, recurrent_input, recurrent, 1, false);
      e.use(pipe("ink2_draft_rnn_update")); e.buf(0, recurrent); e.buf(1, condition); e.buf(2, rnn_bias->buf);
      e.buf(3, state); e.buf(4, signal); e.val(5, rp); e.dispatch((R + 255) / 256, 1, 1, 256);
      e.use(pipe("ink2_draft_scores")); e.buf(0, logits); e.buf(1, folded); e.buf(2, signal); e.buf(3, scores); e.val(4, rp);
      e.dispatch((vocab + 3) / 4, 1, 1, 128);
      const uint32_t lists = topk_lists(vocab, 1);
      e.use(pipe("topk_stage1")); e.buf(0, scores); e.buf(1, cand_v); e.buf(2, cand_i);
      e.val(3, kp::TopkStage{vocab, 1, topk_chunk(1)}); e.dispatch(lists, 1, 1, topk_threads(1));
      e.use(pipe("sample_stage2")); e.buf(0, cand_v); e.buf(1, cand_i); e.buf(2, picks); e.buf(3, probability);
      e.val(4, kp::Sample{lists, 1, 0, 1, 0, 0, uint32_t(step + 1)}); e.dispatch(1, 1, 1, 1024);
    }
    e.commit_and_wait();
    d.release_retired();
    return std::vector<int>(picks.as<int>() + 1, picks.as<int>() + count + 1);
  }
};

Ink2DraftSeq::Ink2DraftSeq() : impl_(std::make_unique<Impl>()) {}
Ink2DraftSeq::~Ink2DraftSeq() = default;
Ink2DraftSeq::Ink2DraftSeq(Ink2DraftSeq&&) noexcept = default;
Ink2DraftSeq& Ink2DraftSeq::operator=(Ink2DraftSeq&&) noexcept = default;
int Ink2DraftSeq::position() const { return impl_ ? int(impl_->pos) : 0; }
size_t Ink2DraftSeq::bytes() const {
  if (!impl_) return 0;
  size_t total = impl_->last.size;
  for (const auto& layer : impl_->layers) total += layer.keys.size + layer.values.size;
  for (const auto& b : impl_->retired) total += b.size;
  return total;
}

Ink2Draft::Ink2Draft(const std::string& path, const Tensor& embedding, const Tensor& norm, const Ink2Mat& unembed)
    : impl_(std::make_unique<Impl>(path, embedding, norm, unembed)) {}
Ink2Draft::~Ink2Draft() = default;

size_t Ink2Draft::observe_allocation_bytes(const Ink2DraftSeq& seq, int rows) const {
  check(bool(seq.impl_), "moved-from sequence");
  const uint32_t need = append_end(seq.impl_->pos, rows);
  const size_t limit = Gpu::get().max_buffer_length();
  size_t bytes = 0;
  for (size_t i = 0; i < seq.impl_->layers.size(); ++i)
    if (!impl_->layers[i].window) bytes = add(bytes, growth(seq.impl_->layers[i].cap, need, limit).allocation_bytes);
  return bytes;
}
std::unique_ptr<Ink2DraftSeq> Ink2Draft::new_seq() {
  auto seq = std::unique_ptr<Ink2DraftSeq>(new Ink2DraftSeq);
  auto& d = *seq->impl_;
  d.last = Gpu::get().alloc(last_bytes, "ink2_draft_last");
  for (uint32_t i = 0; i < d.layers.size(); ++i) {
    auto& layer = d.layers[i];
    layer.cap = impl_->layers[i].window;
    if (!layer.cap) continue;
    layer.keys = Gpu::get().alloc(kv_bytes(layer.cap), "ink2_draft_local_k");
    layer.values = Gpu::get().alloc(kv_bytes(layer.cap), "ink2_draft_local_v");
  }
  return seq;
}
void Ink2Draft::observe(Encoder& e, Ink2DraftSeq& seq, const Buffer& captures, int rows, int pos0) {
  impl_->observe(e, *seq.impl_, captures, rows, pos0);
}
std::vector<int> Ink2Draft::propose(Ink2DraftSeq& seq, int anchor, int count) {
  return impl_->propose(*seq.impl_, anchor, count);
}
size_t Ink2Draft::bytes() const {
  size_t total = impl_->weights.total_bytes();
  for (const auto& b : impl_->owned) total += b.size;
  return total;
}
}  // namespace ni
