#include "common.h"

// ---- On-GPU sampling: two-stage top-k, then temperature / top-p sampling ----
// The sampled token id is written straight into the token buffer so the next step's embedding
// lookup reads it without a CPU round trip.

struct TopkStageParams {
  uint V;      // vocab
  uint K;      // candidates kept per threadgroup (<= 64)
  uint chunk;  // logits per threadgroup (threads * 8)
};

// Callers supply distinct candidate IDs, or UINT_MAX/-inf for exhausted lanes, after filtering NaNs.
// Select the winning value's bits too: equal signed zeros must keep the lower ID's sign.
NI_INLINE void sample_simd_argmax(thread float& v, thread uint& i) {
  const float mv = simd_max(v);
  const uint mi = simd_min(v == mv ? i : 0xffffffffu);
  v = as_type<float>(simd_or(v == mv && i == mi ? as_type<uint>(v) : 0u));
  i = mi;
}

NI_INLINE void tg_argmax(thread float& v, thread uint& i, threadgroup float* sv, threadgroup uint* si, uint tid,
                         uint ntg) {
  uint lane = tid & 31, sg = tid >> 5, nsg = (ntg + 31) >> 5;
  sample_simd_argmax(v, i);
  if (lane == 0) { sv[sg] = v; si[sg] = i; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    v = lane < nsg ? sv[lane] : -INFINITY;
    i = lane < nsg ? si[lane] : 0xffffffff;
    sample_simd_argmax(v, i);
    if (lane == 0) { sv[0] = v; si[0] = i; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  v = sv[0];
  i = si[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// grid: ceil(V / chunk) threadgroups of 256 threads; out: [ntg, K] (value, index), descending.
[[kernel]] void topk_stage1(const device float* logits [[buffer(0)]], device float* cv [[buffer(1)]],
                            device uint* ci [[buffer(2)]], constant TopkStageParams& p [[buffer(3)]],
                            uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                            uint ntg [[threads_per_threadgroup]]) {
  threadgroup float sv[32];
  threadgroup uint si[32];
  float v[8];
  uint base = tg * p.chunk + tid * 8;
  for (int j = 0; j < 8; ++j) v[j] = base + j < p.V ? logits[base + j] : -INFINITY;
  for (uint r = 0; r < p.K; ++r) {
    float bv = -INFINITY;
    uint bi = 0xffffffff;
    for (int j = 0; j < 8; ++j)
      if (v[j] > bv) { bv = v[j]; bi = base + j; }
    tg_argmax(bv, bi, sv, si, tid, ntg);
    if (bi >= base && bi < base + 8) v[bi - base] = -INFINITY;
    if (tid == 0) {
      cv[tg * p.K + r] = bv;
      ci[tg * p.K + r] = bi;
    }
  }
}

struct SampleParams {
  uint n;        // candidates from stage 1
  uint K;        // top-k
  float temp;    // 0 = greedy
  float top_p;
  uint seed;
  uint step;
  uint out_index; // tokens[out_index] = sampled id
};

NI_INLINE uint pcg_hash(uint v) {
  uint state = v * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

// One threadgroup of 1024 threads. The RNG step of the draw: p.step, plus dstep[p.out_index] for sample_stage2_dstep
// (a KFlash tree verify: row r draws at position P + depth(r), so the draw is the plain sampler's for that position).
NI_INLINE void sample_stage2_impl(const device float* cv, const device uint* ci, device int* tokens,
                                  device float* out_probs, constant SampleParams& p, uint step, uint tid, uint ntg,
                                  threadgroup float* sv, threadgroup uint* si, threadgroup float* topv,
                                  threadgroup uint* topi) {
  uint K = max(1u, min(p.K, 64u));
  if (p.temp <= 0.0f) K = 1;
  if (p.temp > 0.0f && p.K >= 8 && p.K <= 64 && p.n % p.K == 0 && p.n / p.K <= 128) {
    // Stage 1 already sorted each list. A lane holds four list heads and advances
    // the winning list, avoiding a threadgroup reduction at every rank.
    if (tid >= 32) return;
    const uint lists = p.n / p.K;
    float v[4];
    uint id[4], pos[4];
    for (uint j = 0; j < 4; ++j) {
      const uint list = tid + j * 32;
      pos[j] = 0;
      v[j] = list < lists ? cv[list * p.K] : -INFINITY;
      id[j] = list < lists ? ci[list * p.K] : 0xffffffffu;
    }
    uint first_id = 0xffffffffu;
    for (uint rank = 0; rank < K; ++rank) {
      float bv = -INFINITY;
      uint bi = 0xffffffffu;
      int bj = -1;
      for (uint j = 0; j < 4; ++j)
        if (v[j] > bv || (v[j] == bv && id[j] < bi)) { bv = v[j]; bi = id[j]; bj = int(j); }
      float wv = bv;
      uint wi = bi;
      sample_simd_argmax(wv, wi);
      // The original selector retains IDs when it removes values. Once every
      // candidate value is exhausted, its -infinity padding repeats the lowest ID.
      first_id = min(first_id, wi);
      if (wv == -INFINITY) wi = first_id;
      if (tid == 0) { topv[rank] = wv; topi[rank] = wi; }
      if (bj >= 0 && bi == wi && bv == wv) {
        const uint list = tid + uint(bj) * 32;
        const bool has_next = ++pos[bj] < p.K;
        v[bj] = has_next ? cv[list * p.K + pos[bj]] : -INFINITY;
        id[bj] = has_next ? ci[list * p.K + pos[bj]] : 0xffffffffu;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  } else {
    constexpr int PER = 4;
    float v[PER];
    uint id[PER];
    for (int j = 0; j < PER; ++j) {
      uint k = tid * PER + j;
      v[j] = k < p.n ? cv[k] : -INFINITY;
      id[j] = k < p.n ? ci[k] : 0xffffffff;
    }
    for (uint r = 0; r < K; ++r) {
      float bv = -INFINITY;
      uint bi = 0xffffffff;
      int bj = -1;
      for (int j = 0; j < PER; ++j)
        if (v[j] > bv || (v[j] == bv && id[j] < bi)) { bv = v[j]; bi = id[j]; bj = j; }
      float wv = bv;
      uint wi = bi;
      tg_argmax(wv, wi, sv, si, tid, ntg);
      if (bj >= 0 && bi == wi && bv == wv) v[bj] = -INFINITY;
      if (tid == 0) { topv[r] = wv; topi[r] = wi; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    uint chosen = topi[0];
    if (p.temp > 0.0f && K > 1) {
      // top-p is taken on the untempered distribution (llama.cpp's and mlx-lm's order: top-k,
      // top-p, then temperature), then the kept candidates are sampled at temperature p.temp.
      float probs[64];
      float tot = 0.0f;
      for (uint r = 0; r < K; ++r) {
        probs[r] = metal::precise::exp(topv[r] - topv[0]);
        tot += probs[r];
      }
      // top-p over the sorted candidates (always keep the first)
      uint keep = K;
      float cum = 0.0f;
      for (uint r = 0; r < K; ++r) {
        cum += probs[r] / tot;
        if (cum >= p.top_p) { keep = r + 1; break; }
      }
      float kt = 0.0f;
      for (uint r = 0; r < keep; ++r) {
        probs[r] = metal::precise::exp((topv[r] - topv[0]) / p.temp);
        kt += probs[r];
      }
      float u = float(pcg_hash(p.seed ^ pcg_hash(step)) & 0xffffff) / 16777216.0f * kt;
      chosen = topi[keep - 1];
      float acc = 0.0f;
      for (uint r = 0; r < keep; ++r) {
        acc += probs[r];
        if (u < acc) { chosen = topi[r]; break; }
      }
    }
    tokens[p.out_index] = int(chosen);
    out_probs[0] = topv[0];
  }
}

[[kernel]] void sample_stage2(const device float* cv [[buffer(0)]], const device uint* ci [[buffer(1)]],
                              device int* tokens [[buffer(2)]], device float* out_probs [[buffer(3)]],
                              constant SampleParams& p [[buffer(4)]], uint tid [[thread_position_in_threadgroup]],
                              uint ntg [[threads_per_threadgroup]]) {
  threadgroup float sv[32];
  threadgroup uint si[32];
  threadgroup float topv[64];
  threadgroup uint topi[64];
  sample_stage2_impl(cv, ci, tokens, out_probs, p, p.step, tid, ntg, sv, si, topv, topi);
}

[[kernel]] void sample_stage2_dstep(const device float* cv [[buffer(0)]], const device uint* ci [[buffer(1)]],
                                    device int* tokens [[buffer(2)]], device float* out_probs [[buffer(3)]],
                                    constant SampleParams& p [[buffer(4)]], const device int* dstep [[buffer(5)]],
                                    uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float sv[32];
  threadgroup uint si[32];
  threadgroup float topv[64];
  threadgroup uint topi[64];
  sample_stage2_impl(cv, ci, tokens, out_probs, p, p.step + uint(dstep[p.out_index]), tid, ntg, sv, si, topv, topi);
}

// ---- Full-vocabulary sampling (kapitsa#1286) ----
// A sampled request whose top-k the two passes above cannot hold: top_k <= 0 (no truncation) or above the 32
// candidates sample_stage2 keeps per row (sampler.h: kTopkMax). The same rules over the whole row: the K best logits
// (value desc, id asc; K = 0: every token), the top-p prefix of that order on the untempered probabilities, then
// exp((l - max) / T) over the kept tokens. The cuts are found by bisecting on the logits' order keys (32 passes over
// the row for top-k, 32 for top-p), ties at a cut taken by lowest id as the sorted lists take them. The draw walks the
// kept tokens in id order: the same law as sample_stage2's walk in value order, not the same token for a given uniform.
// One threadgroup of 1024 threads; simdgroup s owns ids [s * seg, (s + 1) * seg), its lanes 32 apart.
struct SampleFullParams {
  uint V;
  uint K;          // 0 = no truncation
  float temp;      // > 0
  float top_p;
  uint seed;
  uint step;
  uint out_index;  // tokens[out_index] = sampled id
};

constant constexpr uint kFullAll = 0xffffffffu;

NI_INLINE float full_logit(const device float* lg, uint i) {
  const float v = lg[i];
  return isnan(v) ? -INFINITY : v;
}
// order-preserving key of a (non-NaN) logit: a > b <=> key(a) > key(b)
NI_INLINE uint full_key(float v) {
  const uint b = as_type<uint>(v);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
NI_INLINE float full_key_logit(uint k) { return as_type<float>((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k); }

// The kept set: keys above `key`, and of the tokens at `key` the first n by id.
struct FullCut {
  uint key;
  uint n;
};
struct FullRow {
  const device float* lg;
  uint lo, hi, lane, sg, nsg, tid, ntg;
};

NI_INLINE uint full_usum(uint v, threadgroup uint* s, thread const FullRow& r) {
  v = simd_sum(v);
  if (r.lane == 0) s[r.sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint t = r.lane < r.nsg ? s[r.lane] : 0u;
  t = simd_sum(t);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return t;
}

// tokens with key >= t
NI_INLINE uint full_count(thread const FullRow& r, uint t, threadgroup uint* s) {
  uint c = 0;
  for (uint i = r.lo + r.lane; i < r.hi; i += 32) c += full_key(full_logit(r.lg, i)) >= t ? 1u : 0u;
  return full_usum(c, s, r);
}

// the rank (by id) of this simdgroup's first token at key k among all tokens at k
NI_INLINE uint full_tie_base(thread const FullRow& r, uint k, threadgroup uint* s) {
  uint c = 0;
  for (uint i = r.lo + r.lane; i < r.hi; i += 32) c += full_key(full_logit(r.lg, i)) == k ? 1u : 0u;
  c = simd_sum(c);
  if (r.lane == 0) s[r.sg] = c;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint b = 0;
  for (uint j = 0; j < r.sg; ++j) b += s[j];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return b;
}

// untempered mass exp(l - m) of the kept tokens with key >= t
NI_INLINE float full_mass(thread const FullRow& r, FullCut c, uint base, uint t, float m, threadgroup float* sf) {
  float acc = 0.0f;
  uint run = base;
  for (uint b0 = r.lo; b0 < r.hi; b0 += 32) {  // (uniform over the simdgroup: its prefix sums rank the ties)
    const uint i = b0 + r.lane;
    const bool in = i < r.hi;
    const float v = in ? full_logit(r.lg, i) : -INFINITY;
    const uint k = full_key(v);
    const uint tie = in && k == c.key ? 1u : 0u;
    const uint rank = run + simd_prefix_exclusive_sum(tie);
    run += simd_sum(tie);
    if (in && k >= t && (k > c.key || (tie && rank < c.n))) acc += metal::precise::exp(v - m);
  }
  return tg_sum(acc, sf, r.tid, r.ntg);
}

// The list path's select: the largest key t with at least L tokens at or above it, 4 bits a pass (15 thresholds
// counted at once: 8 passes), and the count of tokens above t.
constant constexpr uint kFullList = 1024;
NI_INLINE FullCut full_select(thread const FullRow& r, uint L, threadgroup uint* s16) {
  uint t = 0, above = 0;
  for (int shift = 28; shift >= 0; shift -= 4) {
    uint c[15];
    for (uint j = 0; j < 15; ++j) c[j] = 0;
    for (uint i = r.lo + r.lane; i < r.hi; i += 32) {
      const uint k = full_key(full_logit(r.lg, i));
      if ((k >> uint(shift) >> 4) != (t >> uint(shift) >> 4) && shift < 28) continue;  // (another prefix)
      const uint d = (k >> uint(shift)) & 15u;
      for (uint j = 0; j < 15; ++j) c[j] += d >= j + 1 ? 1u : 0u;
    }
    for (uint j = 0; j < 15; ++j) {
      const uint v = simd_sum(c[j]);
      if (r.lane == 0) s16[r.sg * 16 + j] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // counts within the prefix; the tokens above the prefix (above) add to each
    uint pick = 0, nxt = above;
    for (uint j = 15; j >= 1; --j) {
      uint n = above;
      for (uint g = 0; g < r.nsg; ++g) n += s16[g * 16 + j - 1];
      if (n >= L) {
        pick = j;
        break;
      }
      nxt = n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    t |= pick << uint(shift);
    above = nxt;
  }
  return FullCut{t, L - above};
}

// sort (key desc, id asc) of kFullList entries, all threads
NI_INLINE void full_sort(threadgroup uint* lk, threadgroup uint* li, uint tid, uint ntg) {
  for (uint k = 2; k <= kFullList; k <<= 1)
    for (uint j = k >> 1; j > 0; j >>= 1) {
      for (uint i = tid; i < kFullList; i += ntg) {
        const uint x = i ^ j;
        if (x > i) {
          const bool before = lk[i] > lk[x] || (lk[i] == lk[x] && li[i] < li[x]);  // i ahead of x in the order
          if (((i & k) == 0) != before) {
            const uint a = lk[i], b = li[i];
            lk[i] = lk[x], li[i] = li[x];
            lk[x] = a, li[x] = b;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[kernel]] void sample_full(const device float* logits [[buffer(0)]], device int* tokens [[buffer(1)]],
                            device float* out_probs [[buffer(2)]], constant SampleFullParams& p [[buffer(3)]],
                            uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float sf[32];
  threadgroup uint su[32];
  const uint nsg = (ntg + 31) >> 5, sg = tid >> 5;
  const uint seg = ((p.V + nsg - 1) / nsg + 31) / 32 * 32;
  const uint lo = min(p.V, sg * seg);
  const FullRow r{logits, lo, min(p.V, lo + seg), tid & 31, sg, nsg, tid, ntg};
  float mx = -INFINITY;
  for (uint i = r.lo + r.lane; i < r.hi; i += 32) mx = max(mx, full_logit(logits, i));
  const float m = tg_max(mx, sf, tid, ntg);
  if (!(m > -INFINITY)) {  // every token masked
    if (tid == 0) tokens[p.out_index] = 0, out_probs[0] = m;
    return;
  }
  // The list path: when top-k keeps at most kFullList tokens, or top-p truncates an untruncated row, the best
  // L = min(K, kFullList) tokens are gathered and sorted (value desc, id asc), and sample_stage2's rules run over that
  // list as they run over its 32: the same law, and the same draw for a given uniform. Without top-k the top-p prefix
  // must end inside the list (the untempered mass over the whole row); else, and for a top-k above kFullList, the
  // bisection below runs.
  const uint Kc = p.K > 0 && p.K < p.V ? p.K : p.V;
  if (Kc <= kFullList || (Kc == p.V && p.top_p < 1.0f)) {
    threadgroup uint lk[kFullList], li[kFullList], s16[32 * 16];
    threadgroup atomic_uint nlist;
    threadgroup uint fallback;
    const uint L = min(Kc, kFullList);
    const FullCut lc = full_select(r, L, s16);
    const uint lbase = full_tie_base(r, lc.key, su);
    for (uint i = tid; i < kFullList; i += ntg) lk[i] = 0u, li[i] = 0xffffffffu;
    if (tid == 0) atomic_store_explicit(&nlist, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot_row = 0.0f;  // (Kc > kFullList: the untempered mass of the whole row)
    {
      uint run = lbase;
      for (uint b0 = r.lo; b0 < r.hi; b0 += 32) {
        const uint i = b0 + r.lane;
        const bool in = i < r.hi;
        const float v = in ? full_logit(logits, i) : -INFINITY;
        const uint k = full_key(v);
        const uint tie = in && k == lc.key ? 1u : 0u;
        const uint rank = run + simd_prefix_exclusive_sum(tie);
        run += simd_sum(tie);
        if (in && (k > lc.key || (tie && rank < lc.n))) {
          const uint at = atomic_fetch_add_explicit(&nlist, 1u, memory_order_relaxed);
          if (at < kFullList) lk[at] = k, li[at] = i;
        }
        if (in && Kc > kFullList) tot_row += metal::precise::exp(v - m);
      }
    }
    if (Kc > kFullList) tot_row = tg_sum(tot_row, sf, tid, ntg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    full_sort(lk, li, tid, ntg);
    if (tid == 0) {
      float tot = tot_row;
      if (Kc <= kFullList) {
        tot = 0.0f;
        for (uint q = 0; q < L; ++q) tot += metal::precise::exp(full_key_logit(lk[q]) - m);
      }
      uint keep = L;
      bool reached = Kc <= kFullList;
      float cum = 0.0f;
      for (uint q = 0; q < L; ++q) {
        cum += metal::precise::exp(full_key_logit(lk[q]) - m) / tot;
        if (cum >= p.top_p) {
          keep = q + 1;
          reached = true;
          break;
        }
      }
      fallback = reached ? 0u : 1u;
      if (reached) {
        float kt = 0.0f;
        uint last = li[0];
        for (uint q = 0; q < keep; ++q) {
          const float w = metal::precise::exp((full_key_logit(lk[q]) - m) / p.temp);
          kt += w;
          if (w > 0.0f) last = li[q];
        }
        const float u = float(pcg_hash(p.seed ^ pcg_hash(p.step)) & 0xffffff) / 16777216.0f * kt;
        uint chosen = last;  // (u past the last interval: a rounding gap; the last token with weight)
        float a = 0.0f;
        for (uint q = 0; q < keep; ++q) {
          a += metal::precise::exp((full_key_logit(lk[q]) - m) / p.temp);
          if (u < a) {
            chosen = li[q];
            break;
          }
        }
        tokens[p.out_index] = int(chosen);
        out_probs[0] = m;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (fallback == 0u) return;
  }
  // top-k: the largest key t with at least K tokens at or above it; of the tokens at t, the first K - (those above)
  FullCut cut{0u, kFullAll};
  if (p.K > 0 && p.K < p.V) {
    uint t = 0;
    for (int bit = 31; bit >= 0; --bit) {
      const uint c = t | (1u << uint(bit));
      if (full_count(r, c, su) >= p.K) t = c;
    }
    const uint above = t == 0xffffffffu ? 0u : full_count(r, t + 1, su);
    cut = FullCut{t, p.K - above};
  }
  uint base = full_tie_base(r, cut.key, su);
  // top-p: the shortest prefix of the kept tokens (value desc, id asc) holding top_p of their untempered mass
  if (p.top_p < 1.0f) {
    const float tot = full_mass(r, cut, base, 0u, m, sf);
    const float target = max(p.top_p * tot, 1.17549435e-38f);  // (top_p <= 0: the best token alone, as sample_stage2)
    uint t = 0;
    for (int bit = 31; bit >= 0; --bit) {
      const uint c = t | (1u << uint(bit));
      if (full_mass(r, cut, base, c, m, sf) >= target) t = c;
    }
    const float above = t == 0xffffffffu ? 0.0f : full_mass(r, cut, base, t + 1, m, sf);
    const float w = metal::precise::exp(full_key_logit(t) - m);  // each token at t
    uint n = w > 0.0f ? uint(clamp(ceil((target - above) / w), 1.0f, float(p.V))) : kFullAll;
    if (t == cut.key) n = min(n, cut.n);
    cut = FullCut{t, n};
    base = full_tie_base(r, cut.key, su);
  }
  // the draw: tempered weights of the kept tokens, walked in id order (simdgroup by simdgroup, then lane by lane)
  float acc = 0.0f;
  {
    uint run = base;
    for (uint b0 = r.lo; b0 < r.hi; b0 += 32) {
      const uint i = b0 + r.lane;
      const bool in = i < r.hi;
      const float v = in ? full_logit(logits, i) : -INFINITY;
      const uint k = full_key(v);
      const uint tie = in && k == cut.key ? 1u : 0u;
      const uint rank = run + simd_prefix_exclusive_sum(tie);
      run += simd_sum(tie);
      const bool kept = in && (k > cut.key || (tie && rank < cut.n));
      acc += simd_sum(kept ? metal::precise::exp((v - m) / p.temp) : 0.0f);
    }
  }
  if (r.lane == 0) sf[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float before = 0.0f, kt = 0.0f;
  for (uint j = 0; j < nsg; ++j) {
    if (j == sg) before = kt;
    kt += sf[j];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float u = float(pcg_hash(p.seed ^ pcg_hash(p.step)) & 0xffffff) / 16777216.0f * kt;
  uint pick = 0xffffffffu, last = 0;
  {
    uint run = base;
    float a = before;
    for (uint b0 = r.lo; b0 < r.hi; b0 += 32) {
      const uint i = b0 + r.lane;
      const bool in = i < r.hi;
      const float v = in ? full_logit(logits, i) : -INFINITY;
      const uint k = full_key(v);
      const uint tie = in && k == cut.key ? 1u : 0u;
      const uint rank = run + simd_prefix_exclusive_sum(tie);
      run += simd_sum(tie);
      const bool kept = in && (k > cut.key || (tie && rank < cut.n));
      const float x = kept ? metal::precise::exp((v - m) / p.temp) : 0.0f;
      const float ex = a + simd_prefix_exclusive_sum(x);
      if (x > 0.0f) {
        if (u >= ex && u < ex + x) pick = min(pick, i);
        last = i;
      }
      a += simd_sum(x);
    }
  }
  pick = simd_min(pick);
  last = simd_max(last);
  if (r.lane == 0) su[sg] = pick;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  pick = r.lane < nsg ? su[r.lane] : 0xffffffffu;
  pick = simd_min(pick);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (r.lane == 0) su[sg] = last;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  last = r.lane < nsg ? su[r.lane] : 0u;
  last = simd_max(last);
  if (tid == 0) {
    tokens[p.out_index] = int(pick != 0xffffffffu ? pick : last);  // (no interval held u: a rounding gap at the end)
    out_probs[0] = m;
  }
}

// ---- Structured output: token masks on logits rows ----
// Before a row's top-k stage, the logits of the tokens its mask leaves out (bit t of the row's words clear) become -inf,
// so the sampler and every speculative acceptance after it read the masked distribution: greedy picks the best allowed
// token, sampling draws from the allowed ones renormalized. A word of all ones (every token allowed) writes nothing.
// grid: (ceil(W / 256), R) threadgroups x 256 threads, one 32-token word per thread.
struct LogitMaskParams {
  uint V;      // logits per row that are tokens
  uint W;      // mask words per row (ceil(V / 32))
  uint ld;     // floats between two logits rows
  uint mrow0;  // the mask row of logits row 0
};
[[kernel]] void logit_mask_rows(device float* logits [[buffer(0)]], const device uint* mask [[buffer(1)]],
                                constant LogitMaskParams& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint w = gid.x, r = gid.y;
  if (w >= p.W) return;
  const uint bits = mask[(p.mrow0 + r) * p.W + w];
  if (bits == 0xffffffffu) return;
  device float* row = logits + ulong(r) * p.ld + w * 32;
  const uint n = min(32u, p.V - w * 32);
  for (uint b = 0; b < n; ++b)
    if (!((bits >> b) & 1u)) row[b] = -INFINITY;
}
