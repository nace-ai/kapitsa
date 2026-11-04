// Research (#350, from #912 §6.3): an online, row-sparse update of the DFlash2 selector's two codebooks from each
// cycle's verified tokens. Not a default; ni_run only (NI_SEL_ONLINE=<lr>, NI_SEL_STATE=<file> carries it across runs).
//
// The lattice edge is E[p][j][k] = sum_r A[pred_j][r] s_{p+1}[r] B[C_p[k]][r] (dfl_lattice). After a cycle the verify
// fixed the emitted tokens y_0 .. y_a; at each position p <= min(a, 6) whose true token is among the 16 candidates,
// one SGD step on the cross-entropy of softmax(S / tau_p) at y, S = cand_v + E, tau_p = 1.3 (1 + 0.1 p):
//   g_k = (q_k - [k = y]) / tau_p,  B[C_p[k]] -= lr g_k (A[pred] * s),  A[pred] -= lr sum_k g_k (B[C_p[k]] * s).
// fp32 master rows on the host; the GPU's bf16 codebook rows get the rounded master (shared memory, written while the
// GPU is idle between cycles: ni_run runs it in mode 2, after the verify is waited on and before the next draft). Same arithmetic as kapitsa-online-drafter-notes/f3/replay_edge.py.
#pragma once
#include "util/bf16.h"

#include <cmath>
#include <cstdio>
#include <unordered_map>
#include <vector>

namespace ni {

struct OnlineSelector {
  static constexpr int R = 256;
  uint16_t* cb[2] = {nullptr, nullptr};  // 0 = predecessor codebook A, 1 = successor codebook B; [V][256] bf16
  float lr = 0.0f;
  std::unordered_map<int, std::vector<float>> m[2];  // touched rows: fp32 master
  long steps = 0, skipped = 0;

  std::vector<float>& row(int w, int t) {
    auto it = m[w].find(t);
    if (it != m[w].end()) return it->second;
    std::vector<float> v(R);
    const uint16_t* src = cb[w] + size_t(t) * R;
    for (int r = 0; r < R; ++r) v[r] = bf16_to_f32(src[r]);
    return m[w].emplace(t, std::move(v)).first->second;
  }
  void store(int w, int t) {
    const std::vector<float>& v = m[w][t];
    uint16_t* dst = cb[w] + size_t(t) * R;
    for (int r = 0; r < R; ++r) dst[r] = f32_to_bf16(v[r]);
  }
  // anchor: the token at P; ci/cv: [7][16] candidates; sel: [8][256] selector hidden (row p+1 = position p);
  // y: the a+1 emitted tokens of the cycle
  void learn(int anchor, const int* ci, const float* cv, const float* sel, const int* y, int a) {
    for (int p = 0; p <= std::min(a, 6); ++p) {
      const int pred = p == 0 ? anchor : y[p - 1], tgt = y[p];
      int w = -1;
      for (int k = 0; k < 16; ++k)
        if (ci[p * 16 + k] == tgt) { w = k; break; }
      if (w < 0) { ++skipped; continue; }
      const float tau = 1.3f * (1.0f + 0.1f * p);
      const float* s = sel + (p + 1) * R;
      std::vector<float> as(R), ga(R, 0.0f);
      { const std::vector<float>& A = row(0, pred); for (int r = 0; r < R; ++r) as[r] = A[r] * s[r]; }
      double S[16], mx = -1e30;
      std::vector<float>* Bk[16];
      for (int k = 0; k < 16; ++k) {
        Bk[k] = &row(1, ci[p * 16 + k]);
        double e = 0;
        for (int r = 0; r < R; ++r) e += double((*Bk[k])[r]) * as[r];
        S[k] = (double(cv[p * 16 + k]) + e) / tau;
        mx = std::max(mx, S[k]);
      }
      double z = 0, g[16];
      for (int k = 0; k < 16; ++k) z += std::exp(S[k] - mx);
      for (int k = 0; k < 16; ++k) g[k] = (std::exp(S[k] - mx) / z - (k == w ? 1.0 : 0.0)) / tau;
      // both gradients from the rows before this step (replay_edge.py stacks Bk and reads Ap first)
      for (int k = 0; k < 16; ++k)
        for (int r = 0; r < R; ++r) ga[r] += float(g[k]) * (*Bk[k])[r] * s[r];
      // a token can appear once per position's candidates, so the 16 B rows are distinct
      for (int k = 0; k < 16; ++k) {
        std::vector<float>& B = *Bk[k];
        for (int r = 0; r < R; ++r) B[r] -= lr * float(g[k]) * as[r];
        store(1, ci[p * 16 + k]);
      }
      std::vector<float>& A = row(0, pred);
      for (int r = 0; r < R; ++r) A[r] -= lr * ga[r];
      store(0, pred);
      ++steps;
    }
  }
  // state file: int32 n, then n x {int32 which, int32 token, float[256] master}
  bool load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int32_t n = 0;
    if (fread(&n, 4, 1, f) != 1) { fclose(f); return false; }
    for (int i = 0; i < n; ++i) {
      int32_t h[2];
      std::vector<float> v(R);
      if (fread(h, 4, 2, f) != 2 || fread(v.data(), 4, R, f) != size_t(R)) break;
      m[h[0]][h[1]] = std::move(v);
      store(h[0], h[1]);
    }
    fclose(f);
    return true;
  }
  void save(const char* path) const {
    FILE* f = fopen(path, "wb");
    if (!f) return;
    int32_t n = int32_t(m[0].size() + m[1].size());
    fwrite(&n, 4, 1, f);
    for (int w = 0; w < 2; ++w)
      for (const auto& [t, v] : m[w]) {
        const int32_t h[2] = {w, t};
        fwrite(h, 4, 2, f);
        fwrite(v.data(), 4, R, f);
      }
    fclose(f);
  }
};

}  // namespace ni
