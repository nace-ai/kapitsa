#include "ane/default_share.h"
#include "engine/device_profile.h"

#include <algorithm>
#include <cmath>

namespace ni::ane {

uint32_t default_units(const std::string& chip, int cores, uint32_t total, bool wave, bool gpu_mma) {
  // Measured on the M4 Pro (20-core GPU, 16-core ANE, 27B): the wavefront hides the ANE's
  // FFN of one row half under the GPU's mixer of the other, and the GPU stays the bottleneck up to the
  // whole FFN (34 of 34 units: 7.20 s per 2048-row chunk vs 7.40 at 33, 8.40 at 31); the plain split
  // balances at 27 of 34. Per layer and chunk, the ANE takes ~1.94 ms per unit (fixed for the chip family)
  // and the GPU ~100 ms of mixer plus 6.2 ms per unit it keeps, both scaling with 20 / cores. So the GPU
  // stays the bottleneck at the whole FFN up to ~30 cores (MacBook Airs: 8-10), and a bigger GPU (M4 Max:
  // 40 cores) balances at u = (100 + 6.2 x 34) x 20 / cores / (1.94 + 6.2 x 20 / cores) ~ 31 of 34. A
  // larger ANE share also keeps a fanless Air cooler: the ANE draws a fraction of the GPU's power for the
  // same work. M1 and M2 have shares of their own (below); the M3 is left to a calibration
  // (ane_calib --wave --write).
  if (!total) return 0;
  const double c = std::max(1, cores), k = 20.0 / c;
  if (const int family = ane_pre_m3_share(chip)) {
    // M1 and M2: the same balance with their own costs, measured with ane_calib on the 27B (34 units), macOS 26.7.2,
    // per layer and 2048-row chunk. M1 Ultra (64-core GPU): the plain split's ANE 4.65 ms per unit, the GPU's part
    // 2.89 ms per unit it keeps, the GPU alone 111.5 ms a layer (76.7 of it the FFN, so ~35 ms of mixer); wavefront
    // sweep best at 20 of 34 (5.51 s per chunk against 7.14 s GPU alone; 17: 5.92, 24: 5.70, 31: 6.23). M2 Pro
    // (19-core GPU): ANE 5.58 ms per unit, GPU part 8.33, GPU alone 365.7 ms a layer (261.6 the FFN); wavefront best
    // at 30 of 34 (10.8 s against 23.4 s GPU alone; 28: 11.5, 31: 11.2, 32: 11.7, 24: 13.5). In 20-core terms the
    // GPU's part and mixer are M1 9.23 and 112 ms, M2 7.91 and 99 ms. The balance u = k (mixer + g x 34) / (a + g k)
    // with the plain split's a lands under both wavefront optima (M1 Ultra 18 against 20, M2 Pro 28 against 30): the
    // wavefront hides part of the ANE's time. So the wavefront's a is the one that puts each measured chip at its
    // optimum: M1 3.76 ms, M2 4.58 ms.
    // A single-die M1 has one Neural Engine where the Ultra has two dies. On a 32-core M1 Max (kapitsa#1194, a
    // contributor's MacBook Pro, macOS 27.2, the 27B at ~4K tokens) the GPU alone read 129-134 tok/s, and the
    // shares around this formula's 28 of 34 read 223 (28), 225-236 (26) and 220 (24): the same costs hold there.
    // Neither ANE compiles more than 32 units (16,384 channels): 33 and 34 fail ANECCompile on both, 32 compiles.
    // A noncausal pass (Drex Max's encoder, the plain split) stays on the GPU here: it was not output-checked on M1
    // or M2.
    if (!wave) return 0;
    const bool m1 = family == 1;
    const double g = (m1 ? 9.23 : 7.91) * k, mixer = (m1 ? 112.0 : 99.0) * k, a = m1 ? 3.76 : 4.58;
    const double frac = std::min(1.0, (mixer + g * 34) / (a + g) / 34);
    const uint32_t u = std::max<uint32_t>(1, std::min<uint32_t>(total, uint32_t(std::lround(frac * total))));
    return std::min<uint32_t>(u, 32);
  }
  if (!ane_builtin_share(chip)) return 0;
  if (!wave && !ane_plain_share(chip)) return 0;  // M6 and later: the plain split is not output-checked there yet
  if (gpu_mma) {
    // gpu_mma: the GPU's share runs on the M5's matrix units (Qwen35's NI_QMT, 4-bit group 64), ~3.5x cheaper per
    // channel than its simdgroup GEMMs. Measured on the M5 Pro (20 cores, 27B, ane_calib, 2048 rows): the ANE
    // ~1.68 ms per unit and the GPU 1.52 ms per unit it keeps; the GPU-alone chunk is ~4.8 s, ~75 ms per layer, ~50
    // of them the FFN, so ~25 ms of mixer. Wavefront balance u = (25 + 1.52 x 34) k / (1.68 + 1.52 k): 24 of 34 at
    // 20 cores (ane_calib --wave: 22-28 within noise at 2.99-3.30 s per chunk, 34 4.0 s); plain split
    // u = 1.52 k U / (1.68 + 1.52 k): 16 of 34 (measured best 16).
    // The 128-row tile (NI_QMT128) runs the GPU's share faster: on the M5 Max (bench_qmm, the 27B's shapes) gate/up
    // 1.195x and down 1.112x, so a unit costs (2/3) / 1.195 + (1/3) / 1.112 = 0.86 of what it did, 1.52 -> 1.30 ms, and
    // the mixer's projections ~1.15x, 25 -> 22 ms. The wavefront balance moves from 16 to 14 of 34 units at 40 cores:
    // M5 Max, 27B 4K prefill (ni_run, two passes): 16 units 3,632 ms, 14 3,199, 12 3,218, 8 3,480, 20 4,459, the GPU
    // alone 4,043 (bench/results/2026-11-04-m5-max-ane-share-qmt128). The plain split (a noncausal pass) keeps its
    // measured 1.52, not re-measured with the new tile.
    const double gw = 1.30 * k, gp = 1.52 * k, a = 1.68;
    const double frac = std::min(1.0, wave ? (22.0 * k + gw * 34) / (a + gw) / 34 : gp / (a + gp));
    return std::max<uint32_t>(1, std::min<uint32_t>(total, uint32_t(std::lround(frac * total))));
  }
  const double u34 = (100.0 + 6.2 * 34) * k / (1.94 + 6.2 * k);  // on the 27B's 34 units
  // plain split: ANE and GPU part concurrent per layer, balanced at u = g U / (a + g), matched to the 27 of 34
  // measured at 20 cores
  const double plain = 27.0 / 34 * (6.2 * k / (1.94 + 6.2 * k)) / (6.2 / (1.94 + 6.2));
  const double frac = std::min(1.0, wave ? u34 / 34 : plain);
  return std::max<uint32_t>(1, std::min<uint32_t>(total, uint32_t(std::lround(frac * total))));
}

uint32_t default_min_rows(const std::string& chip, int cores, uint32_t units, uint32_t total) {
  // A chunk under twice the smallest function (2 x 512 rows) runs the plain split, at the share the wavefront was
  // sized for. On the M1 and M2 that share is well past the plain split's balance, so the ANE's part is the whole
  // layer's time there. ane_calib, the 27B at 512 rows: M1 Ultra (64 cores) the ANE 1.71 ms per unit, the GPU alone
  // 19.9 ms a layer (63.7 in 20-core terms); M2 Pro (19 cores) 1.56 and 66.3 (69.8). Serving the 27B, GPU alone ->
  // split, two runs each: M1 Ultra at 20 of 34 units, prompts of 620, 710 and 812 tokens 2.39, 2.71, 3.02 s -> 3.22,
  // 3.48, 3.63 s, of 913 and 1010 tokens 3.37, 3.71 -> 3.15, 3.39 s; M2 Pro at 30 of 34, 812 tokens 9.6 -> 8.0 s.
  // So a chunk under 1024 rows splits only where the ANE's part of the plain split beats the GPU alone (the M1
  // Ultra gives up its 913-1023-row window, ~7%, to keep the smaller ones off a slower split).
  const int family = ane_pre_m3_share(chip);
  if (!family || !total) return 512;
  const double k = 20.0 / std::max(1, cores), share = double(units) / total;
  const double ane = (family == 1 ? 1.71 : 1.56) * 34 * share, gpu = (family == 1 ? 63.7 : 69.8) * k;
  return ane < gpu ? 512 : 1024;
}

}  // namespace ni::ane
