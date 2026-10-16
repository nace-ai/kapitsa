// ane_calib: calibrates the ANE share of the dense prefill FFN split (NI_ANE=1) on this Mac.
//   ane_calib <model_dir> [--units 21,23,25,27,29] [--l0 8] [--layers 8] [--reps 2] [--write]
//   ane_calib <model_dir> --wave [--units 29,31,33,34] [--reps 2] [--write]
// --wave (the default NI_ANE schedule) times whole 2048-row prefill chunks (random tokens, all 64
// layers) per share and keeps the fastest, since there the ANE hides under the GPU's mixer and the
// best share depends on the mixer's time too. Without --wave (the plain split, NI_ANE_WAVE=0):
// For each share it builds the split (one program with a 2048- and a 512-row function; a compile
// each unless the ANE daemon holds it), then times split layers on the real weights (random
// post-norm input): the FFN on the GPU alone, the split's GPU part alone, the ANE alone and both
// together. The share with the fastest split layer at 2048 rows wins; the smallest chunk that
// splits is 512 rows if the split beats the GPU alone there by >= 2%, else 1024. --write appends
// the result to ~/.cache/nace-infer-models/ane-cache/calibration.txt, keyed by chip, macOS build
// and model, which NI_ANE=1 then uses (NI_ANE_UNITS still overrides).
#include "ane/ffn_split.h"
#include "metal/gpu.h"
#include "model/qwen35.h"
#include "util/log.h"

#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

using namespace ni;

static std::vector<uint32_t> parse(const std::string& v) {
  std::vector<uint32_t> r;
  for (size_t p = 0; p < v.size();) {
    size_t q = v.find(',', p);
    r.push_back(uint32_t(std::stoul(v.substr(p, q - p))));
    p = q == std::string::npos ? v.size() : q + 1;
  }
  return r;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: ane_calib <model_dir> [--units a,b,..] [--l0 L] [--layers N] [--reps R] [--write]\n");
    return 1;
  }
  std::string dir = argv[1];
  std::vector<uint32_t> units = {21, 23, 25, 27, 29};
  int l0 = 8, n = 8, reps = 2;
  bool write = false, nostage = false, wave = false;
  for (int i = 2; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) fatal("%s needs a value", k.c_str());
      return argv[++i];
    };
    if (k == "--units") units = parse(next());
    else if (k == "--l0") l0 = std::stoi(next());
    else if (k == "--layers") n = std::stoi(next());
    else if (k == "--reps") reps = std::stoi(next());
    else if (k == "--write") write = true;
    else if (k == "--nostage") nostage = true;
    else if (k == "--wave") wave = true;
    else fatal("unknown option %s", k.c_str());
  }
  setenv("NI_ANE_QUEUE", "1", 1);  // the split needs a deep command queue (before the Gpu exists)
  unsetenv("NI_ANE");
  setenv("NI_NO_MTP", "1", 0);
  Qwen35 m(dir, 2048);
  const std::string chip = Gpu::get().name();
  if (wave) {
    bool given = false;
    for (int i = 2; i < argc; ++i) given |= !strcmp(argv[i], "--units");
    if (!given) units = {29, 31, 33, 34};
    printf("ane_calib --wave: %s, macOS %s, 2048-row prefill chunks, %d reps (median)\n", chip.c_str(),
           ane::os_build().c_str(), reps);
    const double gpu = m.ane_bench_chunk(2048, reps, false);
    printf("GPU alone: %.0f ms per chunk (%.1f tok/s)\n", gpu, 2048 / (gpu / 1000));
    uint32_t best = 0;
    double best_ms = 1e30;
    for (uint32_t u : units) {
      const double t0 = now_ms();
      if (!m.ane_rebuild(u, 512, {1024, 2048}, -1, 2)) continue;
      const double build = now_ms() - t0;
      double ms = 0;
      try {
        ms = m.ane_bench_chunk(2048, reps, true);
      } catch (const std::exception& e) {
        printf("%2u units: %s\n", u, e.what());
        continue;
      }
      printf("%2u units: %.0f ms per chunk (%.1f tok/s, %.2fx the GPU alone); build %.1f s\n", u, ms,
             2048 / (ms / 1000), gpu / ms, build / 1000);
      fflush(stdout);
      if (ms < best_ms) best_ms = ms, best = u;
    }
    if (!best) {
      printf("no share could be built\n");
      return 1;
    }
    printf("best: %u units, %.0f ms per 2048-row chunk vs %.0f GPU alone (%.2fx)\n", best, best_ms, gpu, gpu / best_ms);
    if (write) {
      ane::Calibration c;
      c.units = best;
      c.min_rows = 512;
      char d[256], date[32];
      const time_t now = time(nullptr);
      strftime(date, sizeof date, "%Y-%m-%dT%H:%M", localtime(&now));
      snprintf(d, sizeof d, "wave chunk2048=%.0fms gpu=%.0fms date=%s", best_ms, gpu, date);
      c.detail = d;
      const std::string key = ane::machine_key(chip, ane::model_key(dir)) + "|wave";
      if (ane::write_calibration(key, c)) printf("written: %s/calibration.txt (%s)\n", ane::cache_dir().c_str(), key.c_str());
    }
    return 0;
  }
  printf("ane_calib: %s, macOS %s, layers %d..%d, %d reps (median), staging %s\n", chip.c_str(), ane::os_build().c_str(),
         l0, l0 + n - 1, reps, nostage ? "off" : "on");
  printf("%5s %5s | %9s %9s %9s %9s %9s | %s\n", "units", "rows", "gpu_only", "gpu_part", "ane", "split", "ane_in",
         "split vs gpu_only");
  struct Row {
    uint32_t u;
    AneLayerTimes t2048, t512;
  };
  std::vector<Row> rows;
  for (uint32_t u : units) {
    const double t0 = now_ms();
    if (!m.ane_rebuild(u, 512, {512, 2048}, -1, 1)) continue;
    const double build = now_ms() - t0;
    Row r{u, m.ane_bench(l0, n, 2048, reps, !nostage), m.ane_bench(l0, n, 512, reps, !nostage)};
    for (int k = 0; k < 2; ++k) {
      const AneLayerTimes& t = k ? r.t512 : r.t2048;
      printf("%5u %5d | %9.2f %9.2f %9.2f %9.2f %9.2f | %.2fx%s\n", u, k ? 512 : 2048, t.gpu_only, t.gpu_part, t.ane,
             t.split, t.ane_in_split, t.gpu_only / t.split, k ? "" : "");
    }
    printf("      (build %.1f s: %s)\n", build / 1000, m.ane_split()->startup_note().c_str());
    fflush(stdout);
    rows.push_back(r);
  }
  if (rows.empty()) {
    printf("no share could be built\n");
    return 1;
  }
  const Row* best = &rows[0];
  for (const Row& r : rows)
    if (r.t2048.split < best->t2048.split) best = &r;
  // fit: A per unit and the GPU part per GPU unit, from the measured shares
  double a_unit = 0, g_unit = 0;
  for (const Row& r : rows) {
    a_unit += r.t2048.ane / r.u;
    g_unit += r.u < 34 ? r.t2048.gpu_part / (34 - r.u) : 0;
  }
  a_unit /= rows.size();
  g_unit /= rows.size();
  const uint32_t min_rows = best->t512.split <= 0.98 * best->t512.gpu_only ? 512 : 1024;
  printf("best: %u units (split layer %.2f ms vs %.2f GPU alone at 2048 rows, %.2fx); 512 rows: %.2f vs %.2f -> "
         "chunks >= %u rows; fit: ANE %.3f ms/unit, GPU part %.3f ms/GPU unit (2048 rows)\n",
         best->u, best->t2048.split, best->t2048.gpu_only, best->t2048.gpu_only / best->t2048.split, best->t512.split,
         best->t512.gpu_only, min_rows, a_unit, g_unit);
  if (write) {
    ane::Calibration c;
    c.units = best->u;
    c.min_rows = min_rows;
    char d[256], date[32];
    const time_t now = time(nullptr);
    strftime(date, sizeof date, "%Y-%m-%dT%H:%M", localtime(&now));
    snprintf(d, sizeof d, "split2048=%.2fms gpu2048=%.2fms ane=%.2fms gpu_part=%.2fms date=%s", best->t2048.split,
             best->t2048.gpu_only, best->t2048.ane, best->t2048.gpu_part, date);
    c.detail = d;
    const std::string key = ane::machine_key(chip, ane::model_key(dir));
    if (ane::write_calibration(key, c)) printf("written: %s/calibration.txt (%s)\n", ane::cache_dir().c_str(), key.c_str());
    else printf("could not write the calibration\n");
  }
  return 0;
}
