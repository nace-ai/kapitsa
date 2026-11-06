[[kernel]] void dfl_attn_reduce_original(const device float* part_o [[buffer(0)]], const device float* part_ml [[buffer(1)]],
                                device half* xh [[buffer(2)]], device float* xs [[buffer(3)]],
                                constant DflAttnRedParams& p [[buffer(4)]], const device uint* gst [[buffer(5)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint d [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[4];
  const uint h = tg.x, r = tg.y;
  uint P = p.P, lo = p.lo, chunk = p.chunk, nsplit = p.nsplit;
  dfl_attn_geom(gst, P, lo, chunk, nsplit);
  float mx = -INFINITY;
  for (uint s = 0; s < nsplit; ++s) mx = max(mx, part_ml[((s * 32 + h) * p.nr + r) * 2]);
  float num = 0.0f, den = 0.0f;
  for (uint s = 0; s < nsplit; ++s) {
    const size_t base = (size_t(s) * 32 + h) * p.nr + r;
    const float m = part_ml[base * 2];
    if (m == -INFINITY) continue;
    const float w = exp(m - mx);
    num += w * part_o[base * 128 + d];
    den += w * part_ml[base * 2 + 1];
  }
  const half o = half(num / den);
  const uint rt = r >> 3;  // X8 tile of the row (16-row blocks: each tile has its own half + T table and sums)
  xh += size_t(rt) * 16 * 4096;
  xs += rt * 512;
  const uint r8 = r & 7u;
  xh[r8 * 4096 + h * 128 + d] = o;
  xh[(8 + r8) * 4096 + h * 128 + (d & 64u) + dfl_tslot(d & 63u)] = o;  // T table
  const float sm = simd_sum(float(o));
  if (lane == 0) red[sg] = sm;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (d == 0) {
    xs[(2 * h) * 8 + r8] = red[0] + red[1];
    xs[(2 * h + 1) * 8 + r8] = red[2] + red[3];
  }
}

