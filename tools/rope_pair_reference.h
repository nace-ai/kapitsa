// Baseline kernels from 15f8a57f (unchanged from 4256a511).
static constexpr const char* reference_source = R"metal(
[[kernel]] void attn_prep_reference(const device float* qg [[buffer(0)]], const device float* kr [[buffer(1)]],
                          const device float* vr [[buffer(2)]], const device bfloat* qn_w [[buffer(3)]],
                          const device bfloat* kn_w [[buffer(4)]], device float* q_out [[buffer(5)]],
                          device half* kc [[buffer(6)]], device half* vc [[buffer(7)]],
                          constant AttnPrepParams& p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint2 tid2 [[thread_position_in_threadgroup]]) {
  const uint tid = tid2.x;
  threadgroup float scratch[32];
  threadgroup float row[512];
  const uint h = tg.x, t = tg.y, D = p.D;
  const uint pos = p.pos + t;
  if (h >= p.Hq + p.Hkv) {  // value head: plain copy into the cache
    uint hv = h - p.Hq - p.Hkv;
    vc[(hv * p.max_seq + pos) * D + tid] = half(vr[(t * p.Hkv + hv) * D + tid]);
    return;
  }
  const bool is_q = h < p.Hq;
  float x = is_q ? qg[(t * p.Hq + h) * 2 * D + tid] : kr[(t * p.Hkv + (h - p.Hq)) * D + tid];
  float ss = tg_sum(x * x, scratch, tid, D);
  x = x * metal::precise::rsqrt(ss / float(D) + p.eps) * float(is_q ? qn_w[tid] : kn_w[tid]);
  row[tid] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint half_rot = p.rot / 2;
  if (tid < p.rot) {
    uint i = tid % half_rot;
    float inv_freq = metal::precise::pow(p.theta, -2.0f * float(i) / float(p.rot));
    float ang = float(pos) * inv_freq;
    float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
    if (tid < half_rot) x = row[tid] * cs - row[tid + half_rot] * sn;
    else x = row[tid] * cs + row[tid - half_rot] * sn;
  }
  if (is_q) q_out[(t * p.Hq + h) * D + tid] = x;
  else kc[((h - p.Hq) * p.max_seq + pos) * D + tid] = half(x);
}
[[kernel]] void attn_prep_m_reference(const device float* qg [[buffer(0)]], const device float* kr [[buffer(1)]],
                            const device float* vr [[buffer(2)]], const device bfloat* qn_w [[buffer(3)]],
                            const device bfloat* kn_w [[buffer(4)]], device float* q_out [[buffer(5)]],
                            device half* kc [[buffer(6)]], device half* vc [[buffer(7)]],
                            constant AttnPrepParams& p [[buffer(8)]], const device int* mpos [[buffer(9)]],
                            constant MropeParams& mr [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],
                            uint2 tid2 [[thread_position_in_threadgroup]]) {
  const uint tid = tid2.x;
  threadgroup float scratch[32];
  threadgroup float row[512];
  const uint h = tg.x, t = tg.y, D = p.D;
  const uint pos = p.pos + t;  // cache row
  if (h >= p.Hq + p.Hkv) {     // value head: plain copy into the cache
    uint hv = h - p.Hq - p.Hkv;
    vc[(hv * p.max_seq + pos) * D + tid] = half(vr[(t * p.Hkv + hv) * D + tid]);
    return;
  }
  const bool is_q = h < p.Hq;
  float x = is_q ? qg[(t * p.Hq + h) * 2 * D + tid] : kr[(t * p.Hkv + (h - p.Hq)) * D + tid];
  float ss = tg_sum(x * x, scratch, tid, D);
  x = x * metal::precise::rsqrt(ss / float(D) + p.eps) * float(is_q ? qn_w[tid] : kn_w[tid]);
  row[tid] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint half_rot = p.rot / 2;
  if (tid < p.rot) {
    uint i = tid % half_rot;
    int rp;
    if (pos < mr.mn) {
      const uint a = (i % 3 == 1 && i < mr.sec1) ? 1u : (i % 3 == 2 && i < mr.sec2) ? 2u : 0u;
      rp = mpos[pos * 3 + a];
    } else {
      rp = int(pos) + mr.delta;
    }
    float inv_freq = metal::precise::pow(p.theta, -2.0f * float(i) / float(p.rot));
    float ang = float(rp) * inv_freq;
    float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
    if (tid < half_rot) x = row[tid] * cs - row[tid + half_rot] * sn;
    else x = row[tid] * cs + row[tid - half_rot] * sn;
  }
  if (is_q) q_out[(t * p.Hq + h) * D + tid] = x;
  else kc[((h - p.Hq) * p.max_seq + pos) * D + tid] = half(x);
}
)metal";
