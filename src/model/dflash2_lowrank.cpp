// DFlash2's low-rank shortlist head (DFlash2::build_lowrank): the factors of a rank-r approximation of the 4-bit
// draft head, from the head's weights alone (Accelerate's BLAS and LAPACK; no calibration data).
//
// The drafter's logits are z = x W^T (W: the head rows, [N][K]). A rank-r map x -> (x A) B^T with B = W P shortlists
// the rows whose exact logits are then computed (dfl_lr_rescore). The approximation is the best rank-r one for inputs
// x with second moment C (z's expected squared error): with C = R^T R (Cholesky), Q the top-r eigenvectors of
// R G R^T (G = W^T W), A = R^-1 Q and P = R^T Q. The drafter's final hidden states lean toward the rows of the tokens
// it predicts, so C is taken from the head itself: sum_t w_t w_t^T / (t + 100) over the rows in id order (a Zipf
// weight on the BPE merge order, a frequency proxy), plus a ridge of 5% of its mean diagonal. Offline on the 4B's
// drafter (7 prompts, 2,000 drafting rows): the true top-16's probability mass inside the shortlist 0.999 at r = 768,
// the same as from the drafter's own hidden states (0.999) and far above a plain SVD of W (0.98).
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <utility>

#include "util/bf16.h"
#include "util/log.h"

namespace ni {

// W: 4-bit group-64 rows (MLX affine: nibble j of word i is weight 8 i + j; w = scale q + bias). A: [r][K] (row j =
// column j of A), P: [r][K] (row j = column j of P). False if a factorization fails.
bool dfl_lowrank_factors(const uint32_t* w, const uint16_t* sc, const uint16_t* bi, uint32_t N, uint32_t K, uint32_t r,
                         std::vector<float>& A, std::vector<float>& P) {
  const uint32_t gpr = K / 64, wpr = K / 8;
  std::vector<double> Gd(size_t(K) * K), C(size_t(K) * K);
  {  // The row tiles and fp32 moments are no longer needed after promotion.
    std::vector<float> G(size_t(K) * K, 0.0f), Cz(size_t(K) * K, 0.0f);
    constexpr uint32_t kChunk = 4096;
    std::vector<float> rows(size_t(kChunk) * K), rows_w(size_t(kChunk) * K);
    for (uint32_t r0 = 0; r0 < N; r0 += kChunk) {
      const uint32_t n = std::min(kChunk, N - r0);
      for (uint32_t i = 0; i < n; ++i) {
        const uint32_t t = r0 + i;
        float* dst = rows.data() + size_t(i) * K;
        for (uint32_t g = 0; g < gpr; ++g) {
          const float s = bf16_to_f32(sc[size_t(t) * gpr + g]), b = bf16_to_f32(bi[size_t(t) * gpr + g]);
          for (uint32_t u = 0; u < 8; ++u) {
            const uint32_t q = w[size_t(t) * wpr + g * 8 + u];
            for (uint32_t j = 0; j < 8; ++j) dst[g * 64 + u * 8 + j] = s * float((q >> (4 * j)) & 15u) + b;
          }
        }
        const float sw = std::sqrt(1.0f / (float(t) + 100.0f));
        for (uint32_t k = 0; k < K; ++k) rows_w[size_t(i) * K + k] = dst[k] * sw;
      }
      cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans, int(K), int(n), 1.0f, rows.data(), int(K), 1.0f, G.data(),
                  int(K));
      cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans, int(K), int(n), 1.0f, rows_w.data(), int(K), 1.0f, Cz.data(),
                  int(K));
    }
    double wsum = 0;
    for (uint32_t t = 0; t < N; ++t) wsum += 1.0 / (double(t) + 100.0);
    // full symmetric double matrices (row-major upper = column-major lower; symmetric either way once mirrored)
    for (uint32_t i = 0; i < K; ++i)
      for (uint32_t j = i; j < K; ++j) {
        Gd[size_t(i) * K + j] = Gd[size_t(j) * K + i] = G[size_t(i) * K + j];
        C[size_t(i) * K + j] = C[size_t(j) * K + i] = Cz[size_t(i) * K + j] / wsum;
      }
  }
  double tr = 0;
  for (uint32_t i = 0; i < K; ++i) tr += C[size_t(i) * K + i];
  for (uint32_t i = 0; i < K; ++i) C[size_t(i) * K + i] += tr / K * 0.05;
  // column-major from here: C = R^T R with R upper
  const __LAPACK_int n = K;
  __LAPACK_int info = 0;
  dpotrf_("U", &n, C.data(), &n, &info);
  if (info != 0) {
    logi("DFlash2 low-rank head: Cholesky failed (%ld)", long(info));
    return false;
  }
  for (uint32_t j = 0; j < K; ++j)  // zero the strict lower triangle (column-major: rows > column)
    for (uint32_t i = j + 1; i < K; ++i) C[size_t(j) * K + i] = 0.0;
  const std::vector<double>& R = C;
  // M = R G R^T
  std::vector<double> M = std::move(Gd);
  cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, int(K), int(K), 1.0, R.data(), int(K),
              M.data(), int(K));
  cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, int(K), int(K), 1.0, R.data(), int(K),
              M.data(), int(K));
  // its top r eigenvectors (Z: K x r, column-major)
  const __LAPACK_int il = K - r + 1, iu = K;
  const double vl = 0, vu = 0, abstol = 0;
  __LAPACK_int m = 0;
  std::vector<double> ev(K), Z(size_t(K) * r);
  std::vector<__LAPACK_int> isuppz(2 * size_t(r));
  __LAPACK_int lwork = -1, liwork = -1, iwq = 0;
  double wq = 0;
  dsyevr_("V", "I", "U", &n, M.data(), &n, &vl, &vu, &il, &iu, &abstol, &m, ev.data(), Z.data(), &n, isuppz.data(),
          &wq, &lwork, &iwq, &liwork, &info);
  lwork = __LAPACK_int(wq);
  liwork = iwq;
  std::vector<double> work(size_t(std::max<__LAPACK_int>(lwork, 1)));
  std::vector<__LAPACK_int> iwork(size_t(std::max<__LAPACK_int>(liwork, 1)));
  dsyevr_("V", "I", "U", &n, M.data(), &n, &vl, &vu, &il, &iu, &abstol, &m, ev.data(), Z.data(), &n, isuppz.data(),
          work.data(), &lwork, iwork.data(), &liwork, &info);
  if (info != 0 || m != __LAPACK_int(r)) {
    logi("DFlash2 low-rank head: eigensolver failed (%ld, %ld of %u vectors)", long(info), long(m), r);
    return false;
  }
  // A = R^-1 Z, P = R^T Z (column j of each = row j of the outputs)
  std::vector<double> Ad = Z, Pd = std::move(Z);
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, int(K), int(r), 1.0, R.data(), int(K),
              Ad.data(), int(K));
  cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, int(K), int(r), 1.0, R.data(), int(K),
              Pd.data(), int(K));
  A.assign(Ad.begin(), Ad.end());
  P.assign(Pd.begin(), Pd.end());
  return true;
}

// B rows (the shortlist logits' row order: head row order[j]) = W[order[j]] P^T, as bf16 [N][r]
void dfl_lowrank_rows(const uint32_t* w, const uint16_t* sc, const uint16_t* bi, uint32_t K, uint32_t r,
                      const std::vector<float>& P, const std::vector<uint32_t>& order, std::vector<uint16_t>& out) {
  const uint32_t gpr = K / 64, wpr = K / 8, N = uint32_t(order.size());
  out.resize(size_t(N) * r);
  constexpr uint32_t kChunk = 4096;
  std::vector<float> rows(size_t(kChunk) * K), y(size_t(kChunk) * r);
  for (uint32_t r0 = 0; r0 < N; r0 += kChunk) {
    const uint32_t n = std::min(kChunk, N - r0);
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t t = order[r0 + i];
      float* dst = rows.data() + size_t(i) * K;
      for (uint32_t g = 0; g < gpr; ++g) {
        const float s = bf16_to_f32(sc[size_t(t) * gpr + g]), b = bf16_to_f32(bi[size_t(t) * gpr + g]);
        for (uint32_t u = 0; u < 8; ++u) {
          const uint32_t q = w[size_t(t) * wpr + g * 8 + u];
          for (uint32_t j = 0; j < 8; ++j) dst[g * 64 + u * 8 + j] = s * float((q >> (4 * j)) & 15u) + b;
        }
      }
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(n), int(r), int(K), 1.0f, rows.data(), int(K), P.data(),
                int(K), 0.0f, y.data(), int(r));
    for (size_t i = 0; i < size_t(n) * r; ++i) out[size_t(r0) * r + i] = f32_to_bf16(y[i]);
  }
}

}  // namespace ni
