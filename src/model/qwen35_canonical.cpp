#include "model/qwen35.h"
#include "ane/ffn_split.h"
#include "ane/input_profile.h"
#include "model/canonical_affine.h"
#include "model/kparams.h"
#include "util/log.h"
#include "util/canonical_plan_memo.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace ni {
namespace {
bool skip_tensor(const std::string& n) {
  return n.find("vision_tower") != std::string::npos || n.find("visual.") != std::string::npos || Qwen35::is_mtp(n);
}
void conflicts() {
  for (const char* n : {"NI_DEQUANT", "NI_REQUANT", "NI_QMT_I8"}) {
    const char* v = getenv(n);
    if (v && *v && (strcmp(v, "0") != 0 || !strcmp(n, "NI_DEQUANT") || !strcmp(n, "NI_REQUANT")))
      fatal("NI_KFLASH_CANONICAL_AFFINE: incompatible with %s", n);
  }
}
void require_gpu_only(const Qwen35Config& c) {
  if (ane::input_profile_enabled(ane::FfnSplit::input_segment(uint32_t(c.hidden))))
    fatal("NI_KFLASH_CANONICAL_AFFINE: the current ANE policy is enabled; explicitly set NI_ANE=0 for this GPU prototype");
}
struct Source {
  std::string stem;
  const SafetensorsIndex::Entry *w, *s, *b;
  uint32_t N, K;
  bool deferred;
};
std::pair<uint32_t, uint32_t> expected(const Qwen35Config& c, CanonicalKey key) {
  const uint32_t H = uint32_t(c.hidden), I = uint32_t(c.dense_I), V = uint32_t(c.lin_hv * c.lin_dv);
  switch (key.role) {
    case CanonicalRole::Gate: case CanonicalRole::Up: return {I, H};
    case CanonicalRole::Down: return {H, I};
    case CanonicalRole::GdnQkv: return {uint32_t(2 * c.lin_hk * c.lin_dk) + V, H};
    case CanonicalRole::GdnZ: return {V, H};
    case CanonicalRole::GdnA: case CanonicalRole::GdnB: return {uint32_t(c.lin_hv), H};
    case CanonicalRole::GdnOut: return {H, V};
    case CanonicalRole::Q: return {uint32_t(2 * c.hq * c.head_dim), H};
    case CanonicalRole::K: case CanonicalRole::V: return {uint32_t(c.hkv * c.head_dim), H};
    case CanonicalRole::O: return {H, uint32_t(c.hq * c.head_dim)};
  }
  fatal("canonical affine: unknown role");
}
std::vector<Source> sources(const SafetensorsIndex& index, const Qwen35Config& c) {
  std::vector<Source> out;
  size_t parts = 0;
  auto add = [&](const std::string& stem, uint32_t N, uint32_t K, bool deferred) {
    const auto *w = index.find(stem + ".weight"), *s = index.find(stem + ".scales"), *b = index.find(stem + ".biases");
    if (!N || !K || K % 128 || !w || !s || !b || w->dtype != DType::U32 || s->dtype != DType::BF16 || b->dtype != DType::BF16 ||
        w->shape != std::vector<int64_t>{N, K / 8} || s->shape != std::vector<int64_t>{N, K / 64} || b->shape != s->shape ||
        w->len != size_t(N) * K / 2 || s->len != size_t(N) * (K / 64) * 2 || b->len != s->len)
      fatal("canonical affine: %s must be the original Q4/group64 U32 + BF16 S/B shape", stem.c_str());
    out.push_back({stem, w, s, b, N, K, deferred});
  };
  for (const auto& [name, entry] : index.all()) {
    (void)entry;
    const auto key = canonical_key(name);
    if (!key) continue;
    ++parts;
    if (key.part != 0) continue;
    const bool gdn = key.role >= CanonicalRole::GdnQkv && key.role <= CanonicalRole::GdnOut;
    const bool attn = key.role >= CanonicalRole::Q;
    if ((gdn && !c.is_linear[key.layer]) || (attn && c.is_linear[key.layer]))
      fatal("canonical affine: projection disagrees with layer type: %s", name.c_str());
    const auto [N, K] = expected(c, key);
    add(name.substr(0, name.size() - 7), N, K, true);
  }
  if (out.size() != 248 || parts != 744) fatal("canonical affine: expected 248 layer triplets, got %zu/%zu", out.size(), parts);
  std::sort(out.begin(), out.end(), [](const Source& a, const Source& b) { return a.stem < b.stem; });
  bool head = false;
  for (const char* stem : {"language_model.lm_head", "lm_head", "language_model.model.embed_tokens",
                           "model.language_model.embed_tokens", "model.embed_tokens"})
    if (index.find(std::string(stem) + ".weight")) { add(stem, uint32_t(c.vocab), uint32_t(c.hidden), false); head = true; break; }
  if (!head) fatal("canonical affine: missing retained output head");
  return out;
}
}  // namespace

bool Qwen35::canonical_affine_probe(const Qwen35Config& c) {
  const char* v = getenv("NI_KFLASH_CANONICAL_AFFINE");
  return v && *v && atoi(v) != 0 && !c.moe && c.hidden == 4096 && c.dense_I == 12288 &&
         c.layers == 32 && c.group_size == 64;
}

bool Qwen35::canonical_xp_probe(const Qwen35Config& c) {
  const char* value = getenv("NI_KFLASH_CANONICAL_XP");
  const bool on = value && *value && atoi(value) != 0;
  if (on && !canonical_affine_probe(c))
    fatal("NI_KFLASH_CANONICAL_XP requires the eligible NI_KFLASH_CANONICAL_AFFINE prototype");
  return on;
}

void Qwen35::validate_canonical_affine(const std::string& dir) const {
  conflicts();
  require_gpu_only(cfg_);
  if (v8_packed_coeff_on(cfg_)) fatal("canonical affine: disable legacy NI_V8_PACKED_COEFF_NF1");
  SafetensorsIndex index(dir, skip_tensor);
  for (const auto& x : sources(index, cfg_)) {
    int bits = 0, gs = 0;
    std::string mode;
    const auto pos = x.stem.find(".layers.");
    const std::string canonical = pos != std::string::npos ? "model.layers." + x.stem.substr(pos + 8)
        : x.stem.ends_with("embed_tokens") ? "model.embed_tokens" : "lm_head";
    quant_of(x.stem, canonical, bits, gs, mode);
    if (bits != 4 || gs != 64 || mode != "affine") fatal("canonical affine: selected tensor quantization is not Q4/group64");
  }
}

Qwen35::CanonicalPlan Qwen35::canonical_affine_plan(const std::string& dir, const Qwen35Config& c) {
  conflicts();
  require_gpu_only(c);
  // XP is a per-call reservation, not part of the three immutable packed counts.
  const size_t xp_bytes = canonical_xp_probe(c) ? kCanonicalXpBytes : 0;
  std::string key = "canonical-affine-v1:sb355:w272:page16384:scratch67108864";
  for (int value : {c.hidden, c.layers, c.vocab, c.dense_I, c.group_size, c.lin_hk, c.lin_hv, c.lin_dk,
                     c.lin_dv, c.hq, c.hkv, c.head_dim}) key += ":" + std::to_string(value);
  for (bool linear : c.is_linear) key += linear ? ":L" : ":A";
  const auto counts = canonical_plan_memo::evaluate(dir, key, [&] {
    const double start = now_ms();
    SafetensorsIndex index(dir, skip_tensor);
    canonical_plan_memo::Counts result;
    size_t read_bytes = 0;
    double read_ms = 0, count_ms = 0;
    for (const auto& x : sources(index, c)) {
      if (x.deferred) for (const auto* t : {x.w, x.s, x.b}) result.deferred_bytes += canonical_align(t->len, 256);
      std::vector<uint16_t> s(x.s->len / 2), b(x.b->len / 2);
      const double tr = now_ms();
      index.read(*x.s, 0, s.data(), x.s->len); index.read(*x.b, 0, b.data(), x.b->len);
      read_ms += now_ms() - tr; read_bytes += x.s->len + x.b->len;
      const double tc = now_ms();
      const auto coefficients = pack_affine_coeff_fixed(s.data(), b.data(), x.N, x.K / 64);
      const auto size = canonical_image_size(coefficients.blocks, coefficients.compressed_blocks);
      count_ms += now_ms() - tc;
      result.image_bytes += size.payload + size.headers;
      result.image_pages += size.pages;
    }
    logi("KFlash: canonical affine exact scan: deferred %.6f MiB, packed %.6f MiB (pages %.6f); "
         "read %.3f MiB S/B in %.1f ms, count %.1f ms, total %.1f ms; no W read/GPU work",
         double(result.deferred_bytes) / (1 << 20), double(result.image_bytes) / (1 << 20),
         double(result.image_pages) / (1 << 20), double(read_bytes) / (1 << 20),
         read_ms, count_ms, now_ms() - start);
    return result;
  });
  CanonicalPlan plan{counts.deferred_bytes, counts.image_pages, counts.image_bytes, xp_bytes};
  // This guard runs after BOTH cache hits and misses; the memo never contains XP or a full MemPlan.
  if (plan.image_pages + kCanonicalScratchBytes + plan.xp_bytes > plan.deferred_bytes + kCanonicalExtraLimit)
    fatal("canonical affine: exact image/scratch/XP plan exceeds the 512 MiB net budget");
  logi("KFlash: canonical affine plan total (after memo hit/miss): scratch 64 MiB + XP %.0f MiB, "
       "net pages %.6f MiB; 512 MiB budget checked",
       double(plan.xp_bytes) / (1 << 20),
       double(int64_t(plan.image_pages + kCanonicalScratchBytes + plan.xp_bytes) - int64_t(plan.deferred_bytes)) / (1 << 20));
  return plan;
}

Qwen35::CanonicalStorage::~CanonicalStorage() {
  if (buffers.empty()) return;
  Gpu& g = Gpu::get();
  g.drain();
  for (auto& b : buffers) g.release(b);
}

void Qwen35::load_canonical_affine() {
  Gpu& g = Gpu::get();
  const double start = now_ms();
  const size_t gpu_before = g.allocated_bytes();
  std::unordered_map<const Tensor*, std::string> names;
  size_t deferred = 0, deferred_tensors = 0;
  for (const auto& [name, t] : ws_.all()) {
    names[&t] = name;
    if (canonical_key(name)) {
      if (!ws_.deferred(name) || t.buf) fatal("canonical affine: layer tensor was not deferred: %s", name.c_str());
      deferred += canonical_align(t.byte_size(), 256); ++deferred_tensors;
    }
  }
  if (deferred_tensors != 744) fatal("canonical affine: deferred tensor count is %zu, expected 744", deferred_tensors);
  size_t bytes = 0, pages = 0, temporary_max = 0;
  double read_ms = 0, count_ms = 0, pack_ms = 0;
  auto pack = [&](const QLinear& q, bool missing) {
    if (!q || !q.q4() || q.N == 0 || q.K % 128) fatal("canonical affine: unsupported projection");
    std::vector<uint32_t> w;
    std::vector<uint16_t> s, b;
    const uint32_t* wp = nullptr;
    const uint16_t *sp = nullptr, *bp = nullptr;
    const double tr = now_ms();
    if (missing) {
      w.resize(size_t(q.N) * q.K / 8); s.resize(size_t(q.N) * (q.K / 64)); b.resize(s.size());
      ws_.read_deferred(names.at(q.w), w.data()); ws_.read_deferred(names.at(q.s), s.data());
      ws_.read_deferred(names.at(q.b), b.data());
      wp = w.data(); sp = s.data(); bp = b.data();
    } else {
      if (!q.w->buf || !q.s->buf || !q.b->buf) fatal("canonical affine: retained head has no original buffers");
      wp = q.w->buf.as<uint32_t>(); sp = q.s->buf.as<uint16_t>(); bp = q.b->buf.as<uint16_t>();
    }
    read_ms += now_ms() - tr;
    const double tc = now_ms();
    const auto coefficients = pack_affine_coeff_fixed(sp, bp, q.N, q.K / 64);
    const auto size = canonical_image_size(coefficients.blocks, coefficients.compressed_blocks);
    count_ms += now_ms() - tc;
    if (pages + size.pages + kCanonicalScratchBytes + (canonical_xp_ ? kCanonicalXpBytes : 0) > deferred + kCanonicalExtraLimit)
      fatal("canonical affine: actual packed images exceed the net 512 MiB budget");
    Buffer payload = g.alloc(size.payload, "canonical_affine_payload", false);
    canonical_storage_.buffers.push_back(payload); owned_bufs_.push_back(payload);
    Buffer headers = g.alloc(size.headers, "canonical_affine_headers", false);
    canonical_storage_.buffers.push_back(headers); owned_bufs_.push_back(headers);
    const double tp = now_ms();
    canonical_pack_into(payload.as<uint32_t>(), headers.as<uint32_t>(), wp, coefficients, q.N, q.K / 64);
    pack_ms += now_ms() - tp;
    PackedCoeffBuffers image{payload, headers, 0u, 1u};
    v8_packed_coeff_.emplace(&q, image);
    if (missing) {
      canonical_by_weight_.emplace(q.w, image);
      canonical_owners_.emplace(q.w, &q);
    }
    pages += size.pages; bytes += size.payload + size.headers;
    temporary_max = std::max(temporary_max, w.size() * 4 + (s.size() + b.size()) * 2 + coefficients.words.size() * 4);
  };
  for (const auto& l : layers_) {
    for (const QLinear* q : {&l.mg, &l.mu, &l.md}) pack(*q, true);
    if (l.linear) for (const QLinear* q : {&l.qkv, &l.z, &l.a, &l.b, &l.out}) pack(*q, true);
    else for (const QLinear* q : {&l.q, &l.k, &l.v, &l.o}) pack(*q, true);
  }
  pack(lm_head_, false);
  if (canonical_by_weight_.size() != 248 || v8_packed_coeff_.size() != 249)
    fatal("canonical affine: incomplete packed image set");
  canonical_scratch_ = g.alloc(kCanonicalScratchBytes, "canonical_affine_materialize", true);
  canonical_storage_.buffers.push_back(canonical_scratch_); owned_bufs_.push_back(canonical_scratch_);
  if (canonical_xp_) {
    // Reserve before allocation so constructor-failure cleanup owns the new buffer immediately.
    canonical_storage_.buffers.reserve(canonical_storage_.buffers.size() + 1);
    owned_bufs_.reserve(owned_bufs_.size() + 1);
    canonical_xp_scratch_ = g.alloc(kCanonicalXpBytes, "canonical_prefill_xp", true);
    canonical_storage_.buffers.push_back(canonical_xp_scratch_);
    owned_bufs_.push_back(canonical_xp_scratch_);
    logi("KFlash: canonical XP experimental target prefill: one fixed 48 MiB slot");
  }
  const size_t gpu_after = g.allocated_bytes();
  if (gpu_after < gpu_before || gpu_after - gpu_before > deferred + kCanonicalExtraLimit)
    fatal("canonical affine: observed Metal allocation delta exceeds net budget");
  logi("KFlash: canonical affine 248 deferred layers + retained head, 249 images %.6f MiB/pages %.6f, "
       "64 MiB tracked scratch (+ XP %.0f MiB); original layer W/S/B %.6f MiB absent; net pages %.6f MiB; "
       "serial load %.1f ms (read %.1f, codec %.1f, pack %.1f), temporary logical max %.3f MiB",
       double(bytes) / (1 << 20), double(pages) / (1 << 20), canonical_xp_ ? 48.0 : 0.0, double(deferred) / (1 << 20),
       double(int64_t(pages + kCanonicalScratchBytes + (canonical_xp_ ? kCanonicalXpBytes : 0)) - int64_t(deferred)) / (1 << 20),
       now_ms() - start, read_ms, count_ms, pack_ms, double(temporary_max) / (1 << 20));
}

const QLinear& Qwen35::CanonicalViews::get(const QLinear& q) const {
  for (size_t i = 0; i < originals.size(); ++i) if (originals[i] == q.w) return linears[i];
  return q;
}

Qwen35::CanonicalViews Qwen35::materialize_affine(Encoder& e, const std::vector<const QLinear*>& inputs) {
  CanonicalViews out;
  if (!canonical_affine_) return out;
  std::vector<const QLinear*> unique;
  for (const QLinear* q : inputs) {
    if (!q || !*q) fatal("canonical affine: empty consumer projection");
    if (!q->s || !q->b) fatal("canonical affine: consumer lacks affine coefficient metadata");
    if (q->w->buf && q->s->buf && q->b->buf) continue;
    if (!canonical_by_weight_.count(q->w)) fatal("canonical affine: missing materialization map for empty buffers");
    const QLinear& owner = *canonical_owners_.at(q->w);
    if (q->s != owner.s || q->b != owner.b || q->N != owner.N || q->K != owner.K)
      fatal("canonical affine: altered deferred triplet metadata is not supported");
    if (q->w->shape != std::vector<int64_t>{q->N, q->K / 8})
      fatal("canonical affine: sliced deferred weight views are not supported");
    if (std::none_of(unique.begin(), unique.end(), [&](const QLinear* x) { return x->w == q->w; })) unique.push_back(q);
  }
  if (unique.empty()) return out;
  std::vector<CanonicalSlice> slices;
  size_t end = 0;
  for (const QLinear* q : unique) { slices.push_back(canonical_slice(end, q->N, q->K)); end = slices.back().end; }
  out.tensors.reserve(unique.size() * 3); out.linears.reserve(unique.size()); out.originals.reserve(unique.size());
  e.barrier();  // old scratch readers -> new writers, including a grouped encoder's independent dispatches
  for (size_t i = 0; i < unique.size(); ++i) {
    const QLinear& q = *unique[i]; const auto& slice = slices[i]; const auto& image = canonical_by_weight_.at(q.w);
    const size_t first = out.tensors.size(), wb = size_t(q.N) * q.K / 2, sb = size_t(q.N) * (q.K / 64) * 2;
    for (const Tensor* t : {q.w, q.s, q.b}) out.tensors.push_back(*t);
    out.tensors[first].buf = canonical_scratch_.view(slice.w, wb);
    out.tensors[first + 1].buf = canonical_scratch_.view(slice.s, sb);
    out.tensors[first + 2].buf = canonical_scratch_.view(slice.b, sb);
    QLinear copy = q; copy.w = &out.tensors[first]; copy.s = &out.tensors[first + 1]; copy.b = &out.tensors[first + 2];
    out.originals.push_back(q.w); out.linears.push_back(copy);
    e.use(Gpu::get().pipeline("qmm8pc_unpack_affine"));
    e.buf(0, image.payload); e.buf(1, image.headers);
    e.buf(2, copy.w->buf); e.buf(3, copy.s->buf); e.buf(4, copy.b->buf);
    const uint32_t nk[2] = {q.N, q.K}; e.bytes(5, nk, sizeof nk);
    const uint64_t groups = uint64_t(q.N) * (q.K / 64);
    if (groups > uint64_t(UINT32_MAX)) fatal("canonical affine: unpack thread index overflow");
    e.dispatch(uint32_t((groups + 255) / 256), 1, 1, 256);
  }
  e.barrier();  // every disjoint restored triplet is ready before any original consumer pipeline/bindings
  return out;
}
namespace {
bool xp_span(const Buffer& b, size_t need, size_t alignment) {
  return b && b.base && b.offset % alignment == 0 && need <= b.size &&
         b.offset <= std::numeric_limits<size_t>::max() - b.size;
}
bool xp_overlap(const Buffer& a, const Buffer& b) {
  return a.mtl == b.mtl && a.offset < b.offset + b.size && b.offset < a.offset + a.size;
}
}  // namespace

bool Qwen35::canonical_xp_eligible(const QLinear& q, const QLinear* up, const Buffer& x,
                                   const Buffer& y, int M) const {
  if (!canonical_xp_ || !canonical_affine_ || !canonical_xp_prefill_context_ || M <= 128 || q.N <= 64 ||
      v8_nf2_target_context_ || v8_nf2_prefill_chain_ || v8w_R_ || ane_chunk_) return false;
  if (up && (up->N != q.N || up->K != q.K || up->N <= 64)) return false;
  size_t xp_bytes = 0;
  if (!canonical_xp_size(M, q.K, xp_bytes) ||
      !xp_span(canonical_xp_scratch_, kCanonicalXpBytes, 16) ||
      !xp_span(canonical_scratch_, kCanonicalScratchBytes, 16)) return false;
  if (uint64_t(M) * q.N > UINT32_MAX) return false;  // shader output row indices are uint
  const uint64_t input = uint64_t(M) * q.K * 2, output = uint64_t(M) * q.N * (up ? 2 : 4);
  if (input > SIZE_MAX || output > SIZE_MAX || !xp_span(x, size_t(input), 16) ||
      !xp_span(y, size_t(output), up ? 2 : 4)) return false;
  if (xp_overlap(x, canonical_scratch_) || xp_overlap(x, canonical_xp_scratch_) ||
      xp_overlap(y, canonical_scratch_) || xp_overlap(y, canonical_xp_scratch_) || xp_overlap(x, y)) return false;
  const uint64_t nx = (uint64_t(q.N) + (up ? 63 : 127)) / (up ? 64 : 128), my = (uint64_t(M) + 31) / 32;
  if (nx * my > UINT32_MAX) return false;
  size_t end = 0;
  for (const QLinear* l : {&q, up}) {
    if (!l) continue;
    if (!*l || !l->s || !l->b || !l->q4() || l->q20 || l->exl3 || l->mtp || l->K % 64 ||
        l->w->buf || l->s->buf || l->b->buf) return false;
    const auto owner_it = canonical_owners_.find(l->w);
    const auto image_it = canonical_by_weight_.find(l->w);
    if (owner_it == canonical_owners_.end() || image_it == canonical_by_weight_.end()) return false;
    const QLinear& owner = *owner_it->second;
    if (l->s != owner.s || l->b != owner.b || l->N != owner.N || l->K != owner.K || !l->same_format(owner)) return false;
    if (l->w->dtype != DType::U32 || l->s->dtype != DType::BF16 || l->b->dtype != DType::BF16 ||
        l->w->shape != std::vector<int64_t>{l->N, l->K / 8} ||
        l->s->shape != std::vector<int64_t>{l->N, l->K / 64} || l->b->shape != l->s->shape) return false;
    const auto& image = image_it->second;
    const uint64_t blocks = ((uint64_t(l->N) + 15) / 16) * ((l->K / 64 + 1) / 2);
    if (image.ks != 0 || image.rg != 1 || uint64_t(l->N) * (l->K / 64) > UINT32_MAX ||
        blocks > SIZE_MAX / (272 * 4) || !xp_span(image.payload, size_t(blocks) * 272 * 4, 16) ||
        !xp_span(image.headers, size_t(blocks) * 4, 4)) return false;
    CanonicalXpSlice slice;
    if (!canonical_xp_slice(end, l->N, l->K, slice)) return false;
    end = slice.end;
  }
  return true;
}

bool Qwen35::enc_canonical_xp(Encoder& e, const QLinear& q, const QLinear* up, const Buffer& x,
                              const Buffer& y, int M, bool add) {
  if (!canonical_xp_eligible(q, up, x, y, M)) return false;
  Gpu& g = Gpu::get();
  const char* name = up ? "qmr_xpsb_gu_k32_w14_m32_n64" : "qmr_xpsb_k32_w14_m32_n128";
  const Pipeline pack = g.pipeline("qmr_xpsb_pack_x"), unpack = g.pipeline("qmr_xpsb_unpack"), gemm = g.pipeline(name);
  for (const auto& p : {pack, unpack, gemm})
    if (!p.pso || p.simd_width != 32 || p.tg_mem > g.max_threadgroup_memory()) return false;
  if (pack.max_threads < 256 || unpack.max_threads < 256 || gemm.max_threads < 128) return false;
  struct View { const QLinear* q; const PackedCoeffBuffers* image; Buffer w, sb; } views[2];
  const uint32_t count = up ? 2 : 1;
  size_t end = 0, xp_bytes = 0;
  if (!canonical_xp_size(M, q.K, xp_bytes)) return false;
  for (uint32_t i = 0; i < count; ++i) {
    const QLinear* l = i ? up : &q;
    CanonicalXpSlice sl;
    if (!canonical_xp_slice(end, l->N, l->K, sl)) return false;
    end = sl.end;
    views[i] = {l, &canonical_by_weight_.at(l->w), canonical_scratch_.view(sl.w, size_t(l->N) * l->K / 2),
                canonical_scratch_.view(sl.sb, size_t(l->N) * (l->K / 64) * 4)};
  }
  const Buffer xp = canonical_xp_scratch_.view(0, xp_bytes);
  // Every rejection above precedes the first mutation. Never recurse/fall back after this boundary.
  // No group_begin/end here: it is not nestable, and the surrounding encoder keeps its original schedule.
  e.barrier();
  const uint32_t xshape[3] = {uint32_t(M), q.K, q.K};
  e.use(pack); e.buf(0, x); e.buf(1, xp); e.bytes(2, xshape, sizeof xshape);
  e.dispatch(uint32_t((xp_bytes / 4 + 255) / 256), 1, 1, 256);
  for (uint32_t i = 0; i < count; ++i) {
    const auto& v = views[i]; const uint32_t shape[2] = {v.q->N, v.q->K};
    e.use(unpack); e.buf(0, v.image->payload); e.buf(1, v.image->headers); e.buf(2, v.w); e.buf(3, v.sb);
    e.bytes(4, shape, sizeof shape);
    e.dispatch(uint32_t((uint64_t(v.q->N) * (v.q->K / 64) + 255) / 256), 1, 1, 256);
  }
  e.barrier();
  kp::Qmm p{uint32_t(M), q.N, q.K, q.K, q.N, add ? 1u : 0u, 64, 0, 0};
  e.use(gemm); e.buf(0, xp); e.buf(1, views[0].w); e.buf(2, views[0].sb);
  e.buf(3, views[0].sb); e.buf(4, y); e.val(5, p);  // unused bias slot gets a valid buffer
  if (up) { e.buf(6, views[1].w); e.buf(7, views[1].sb); e.buf(8, views[1].sb); }
  e.dispatch((q.N + (up ? 63 : 127)) / (up ? 64 : 128), (uint32_t(M) + 31) / 32, 1, 128);
  e.barrier();  // consumer -> next reuse, even when the caller is in a grouped encoder
  return true;
}
}  // namespace ni
