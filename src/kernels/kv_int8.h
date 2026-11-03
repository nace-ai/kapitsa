// The KV cache row layout of the Qwen3.5/3.6/3.8 gated full attention (head_dim 256; attn.metal, mm.metal,
// verify.metal, verify8.metal, tensor/qattn_t.metal), in the two layouts the engine can allocate:
//
//   fp16 (the default, NI_KV_I8 unset): 512 bytes, 256 halves a (kv head, position) row, exactly the rows every build
//   before this one wrote. The accessors below are then the plain half4 / uint4 / half reads they replaced.
//
//   int8 (NI_KV_I8=1): 272 bytes a row
//     [0, 256)   256 int8 codes, symmetric quantization
//     [256, 260) float scale = max|x| / 127 (1.0 when the row is all zero)
//     [260, 272) padding
//   272 % 16 == 0, so a row is exactly 17 uint4 words: the 16-byte-word copies (kv_regrow / kv_copy_prefix /
//   kv_tree_compact) stay row-exact with the engine passing KVQ_ROW/16 words per position (qwen35.cpp).
//   Dequant: half = scale * code; the error is bounded by scale/2 = max|x| / 254 (<= 0.4% of the row max).
//
// The kernels address a row by its byte offset and call kvq_load4 / kvq_load8 / kvq_load1 / kvq_store1, so a call site
// is written once for both layouts: only KVQ_ROW (the stride) and these four functions differ, and the choice is the
// library's compile (metal/gpu.mm passes NI_KV_I8 when the environment asks for it). The engine's row size follows the
// same switch: Qwen35::kv_row_bytes / kv_bytes_per_pos / grow_kv / enc_kv_regrow / copy_prefix / disk_rows, Flint's
// kv_row and Qwen4Exp's arow return 512 (fp16) or 272 (int8) for head_dim 256.

constant constexpr uint KVQ_HD = 256;  // head_dim covered by this layout

#ifdef NI_KV_I8
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

#else  // fp16: the layout every build before this one wrote

constant constexpr uint KVQ_ROW = 512;  // bytes per (head, position): 256 halves

// 4 dims (4 halves) at (base + row_off + c4*4) -> half4.
inline half4 kvq_load4(const device half* base, uint row_off, uint c4) {
  return *(const device half4*)((const device char*)base + row_off + c4 * 8);
}

// 8 dims (8 halves) at (base + row_off + c8*8) -> uint4, the 16 bytes the fp16 kernels read.
inline uint4 kvq_load8(const device half* base, uint row_off, uint c8) {
  return *(const device uint4*)((const device char*)base + row_off + c8 * 16);
}

// One dim (base + row_off + dim) -> half.
inline half kvq_load1(const device half* base, uint row_off, uint dim) {
  return *(const device half*)((const device char*)base + row_off + dim * 2);
}

// Store one value of a row. scratch is the int8 layout's threadgroup maxima: unused here.
inline void kvq_store1(device half* base, uint row_off, threadgroup float* scratch, uint tid, float x) {
  (void)scratch;
  *(device half*)((device char*)base + row_off + tid * 2) = half(x);
}

#endif
