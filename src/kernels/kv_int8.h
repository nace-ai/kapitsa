// int8 KV cache rows for the Qwen3.5/3.6/3.8 gated full attention (head_dim 256, attn.metal /
// mm.metal / verify8.metal / verify.metal / tensor/qattn_t.metal kernels).
//
// One (kv head, position) row is 272 bytes:
//   [0, 256)   256 int8 values, symmetric quantization
//   [256, 260) float scale = max|x| / 127 (1.0 when the row is all zero)
//   [260, 272) padding
// 272 % 16 == 0, so a row is exactly 17 uint4 words: the 16-byte-word copies
// (kv_regrow / kv_copy_prefix / kv_tree_compact) stay row-exact with the engine
// passing KVQ_ROW/16 words per position (qwen35.cpp).
//
// Dequant: half = scale * int8; the error is bounded by scale/2 = max|x| / 254
// (<= 0.4% of the row max). Same scheme as the Flint kernels (src/kernels/flint).
// The engine keeps the same layout: Qwen35::kv_bytes_per_pos / kv_per_pos /
// grow_kv / enc_kv_regrow / copy_prefix / disk_rows all use (head_dim + 16) / 16 * 16
// per (head, position), i.e. KVQ_ROW for head_dim 256.


constant constexpr uint KVQ_HD = 256;   // head_dim covered by this layout
constant constexpr uint KVQ_ROW = 272;  // bytes per (head, position)

// The row's float scale.
inline float kvq_scale(const device half* base, uint row_off) {
  return *(const device float*)((const device char*)base + row_off + KVQ_HD);
}

// 4 dims (4 int8 bytes) at (base + row_off + c4*4) -> half4.
inline half4 kvq_load4(const device half* base, uint row_off, uint c4) {
  const device char* row = (const device char*)base + row_off;
  const float sc = *(const device float*)(row + KVQ_HD);
  const char4 q = *(const device char4*)(row + c4 * 4);
  return half4(sc * float(q.x), sc * float(q.y), sc * float(q.z), sc * float(q.w));
}

// 8 dims (8 int8 bytes) at (base + row_off + c8*8) -> uint4 (8 halves, as the fp16
// kernels read them: one uint2 of int8 data expands to 4 half2).
inline uint4 kvq_load8(const device half* base, uint row_off, uint c8) {
  const device char* row = (const device char*)base + row_off;
  const float sc = *(const device float*)(row + KVQ_HD);
  const uint2 raw = *(const device uint2*)(row + c8 * 8);
  const char4 a = as_type<char4>(raw.x), b = as_type<char4>(raw.y);
  uint4 o;
  o.x = as_type<uint>(half2(sc * float(a.x), sc * float(a.y)));
  o.y = as_type<uint>(half2(sc * float(a.z), sc * float(a.w)));
  o.z = as_type<uint>(half2(sc * float(b.x), sc * float(b.y)));
  o.w = as_type<uint>(half2(sc * float(b.z), sc * float(b.w)));
  return o;
}

// One dim (base + row_off + dim) -> half.
inline half kvq_load1(const device half* base, uint row_off, uint dim) {
  const device char* row = (const device char*)base + row_off;
  const float sc = *(const device float*)(row + KVQ_HD);
  return half(sc * float(row[dim]));
}

// Store one value of a row: the threadgroup is one row (D threads, each holding one
// value); scratch holds the simdgroup maxima (D/32 floats needed). Quantizes the row
// against its max-abs scale and writes the 256 int8 values plus the row's float scale.
inline void kvq_store1(device half* base, uint row_off, threadgroup float* scratch, uint tid, float x) {
  device char* row = (device char*)base + row_off;
  const float mx = tg_max(fabs(x), scratch, tid, KVQ_HD);
  const float sc = mx > 0.0f ? mx / 127.0f : 1.0f;
  row[tid] = (char)rint(x / sc);
  if (tid == 0) *(device float*)(row + KVQ_HD) = sc;
}
