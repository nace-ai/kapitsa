// Hy4-flash speculative decoding (src/model/hy4_flash.cpp): the DFly2 drafter's target hidden for the next cycle is the
// verify's normed final row of the last accepted token, row acc[0] of the verify's rows.
[[kernel]] void hy4_pick_row(const device float* src [[buffer(0)]], const device int* acc [[buffer(1)]],
                             device float* dst [[buffer(2)]], constant uint& D [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
  if (i < D) dst[i] = src[size_t(acc[0]) * D + i];
}

// Accept the drafts tokens[P+1 .. P+K] while each equals its row's own sample; the first that differs is replaced by
// that sample (all K accepted: the last row's sample follows them). acc[0] = the drafts accepted.
struct Hy4AcceptP {
  uint P, K;
};
[[kernel]] void hy4_accept(const device int* samples [[buffer(0)]], device int* tokens [[buffer(1)]],
                           device int* acc [[buffer(2)]], constant Hy4AcceptP& p [[buffer(3)]]) {
  uint a = 0;
  while (a < p.K && samples[a] == tokens[p.P + 1 + a]) ++a;
  tokens[p.P + 1 + a] = samples[a];
  acc[0] = int(a);
}
