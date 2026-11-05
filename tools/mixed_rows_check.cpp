// Teacher-forced trunk logits and MTP hidden rows for R=1..8 at a fixed real-model prefix.
// Run old/new builds on the same ids, then compare trunk.f32 with logit_compare.py --vocab V
// and mtp.f32 byte for byte. Takes the GPU lock; no sampling decides the tested prefixes.
#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/qwen35.h"
#include "util/log.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>

using namespace ni;
int main(int argc, char** argv) {
  if (argc != 5) fatal("usage: mixed_rows_check MODEL IDS PREFIX OUTPUT_DIR");
  std::ifstream in(argv[2]);
  std::string text((std::istreambuf_iterator<char>(in)), {}), item;
  std::stringstream ss(text);
  std::vector<int> ids;
  while (std::getline(ss, item, ',')) ids.push_back(std::stoi(item));
  const int P = atoi(argv[3]);
  if (P < 1 || ids.size() < size_t(P + 9)) fatal("need PREFIX + 9 input tokens");
  std::filesystem::create_directories(argv[4]);
  const auto path = std::filesystem::path(argv[4]);
  if (std::filesystem::exists(path / "trunk.f32") || std::filesystem::exists(path / "mtp.f32"))
    fatal("output files already exist");
  std::ofstream trunk(path / "trunk.f32", std::ios::binary), mtp(path / "mtp.f32", std::ios::binary);
  if (!trunk || !mtp) fatal("cannot create output files");
  bench::GpuLock lock;
  if (!bench::GpuLock::held_by_caller() && !lock.acquire()) fatal("cannot acquire GPU lock");
  auto& g = Gpu::get();
  Qwen35 m(argv[1], 2048);
  if (!m.has_mtp()) fatal("model has no MTP head");
  auto seq = m.new_seq(P + 32);
  Seq& s = *seq;
  std::copy_n(ids.begin(), P + 9, s.tokens());
  SamplingParams sp;
  for (int pos = 0; pos < P; pos += m.prefill_chunk()) {
    const int rows = std::min(m.prefill_chunk(), P - pos);
    auto e = g.begin();
    m.encode_prefill(*e, s, pos, rows, sp, 0, false);
    // Teacher forcing supplies token P too, so every prompt hidden row can enter MTP history.
    m.encode_mtp_prefill(*e, s, pos, rows, 0);
    e->commit_and_wait();
    if (!m.ane_finish_chunk()) fatal("ANE prefill failed");
  }
  auto checkpoint = m.new_checkpoint();
  { auto e = g.begin(); m.save(*e, s, checkpoint, P); e->commit_and_wait(); }
  if (Encoder::profiling()) Encoder::profile_report(1);
  const int V = m.config().vocab, D = m.config().hidden;
  for (int R = 1; R <= 8; ++R) {
    std::copy_n(ids.begin(), P + 9, s.tokens());
    auto e = g.begin();
    m.restore(*e, s, checkpoint);
    m.encode_verify(*e, s, P, R, sp, 0);
    e->commit_and_wait();
    trunk.write(reinterpret_cast<const char*>(m.logits()), size_t(R) * V * sizeof(float));
    std::copy_n(ids.begin(), P + 9, s.tokens());
    e = g.begin();
    m.encode_mtp(*e, s, P, R, P + 1, m.hidden_rows(), 0, -1);
    e->commit_and_wait();
    mtp.write(reinterpret_cast<const char*>(m.mtp_post().ptr()), size_t(R) * D * sizeof(float));
    printf("P=%d R=%d vocab=%d hidden=%d trunk_rows=%d mtp_rows=%d\n", P, R, V, D, R, R);
    fflush(stdout);
  }
  if (Encoder::profiling()) Encoder::profile_report(1);
  if (!trunk || !mtp) fatal("output write failed");
}
