// Teacher-forced one/two-row verify and MTP comparisons on real prompts. Takes the GPU lock.
// norm_x8_model_check MODEL IDS PREFIX OUT [PASSES=24] [R=1]; old/new runs consume identical token prefixes.
// A: NI_NORM_X8=0; B: NI_NORM_X8=1. OUT must be new. Compare trunk.f32 with
// tools/logit_compare.py --vocab V --floor FLOOR; mtp.f32 contains post-norm hidden rows, not logits.
#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/qwen35.h"
#include "util/log.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string_view>
#include <vector>

using namespace ni;
namespace {
int integer(std::string_view text, bool positive) {
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) text.remove_prefix(1);
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) text.remove_suffix(1);
  int value = -1;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value < (positive ? 1 : 0))
    fatal("invalid integer: %.*s", int(text.size()), text.data());
  return value;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && !strcmp(argv[1], "--help")) {
    puts("norm_x8_model_check MODEL IDS PREFIX OUT [PASSES=24] [R=1]");
    return 0;
  }
  if (argc < 5 || argc > 7) fatal("usage: norm_x8_model_check MODEL IDS PREFIX OUT [PASSES=24] [R=1]");
  std::ifstream input(argv[2]);
  if (!input) fatal("cannot read token ids: %s", argv[2]);
  std::string text((std::istreambuf_iterator<char>(input)), {}), word;
  std::stringstream parse(text);
  std::vector<int> ids;
  while (std::getline(parse, word, ',')) ids.push_back(integer(word, false));
  const int prefix = integer(argv[3], true), rows = argc >= 6 ? integer(argv[5], true) : 24;
  const int R = argc == 7 ? integer(argv[6], true) : 1;
  const size_t count = size_t(prefix) + size_t(rows) * R + 1;
  if ((R != 1 && R != 2) || count > size_t(std::numeric_limits<int>::max()) - 32 || ids.size() < count)
    fatal("unsupported dimensions or not enough tokens");
  const std::filesystem::path out(argv[4]);
  std::error_code ec;
  if (!std::filesystem::create_directories(out, ec))
    fatal("cannot create output directory: %s", ec ? ec.message().c_str() : "already exists");
  std::ofstream trunk(out / "trunk.f32", std::ios::binary), mtp(out / "mtp.f32", std::ios::binary);
  if (!trunk || !mtp) fatal("cannot open outputs");
  bench::GpuLock lock;
  if (!bench::GpuLock::held_by_caller() && !lock.acquire()) fatal("GPU lock failed");
  auto& gpu = Gpu::get();
  gpu.compile();
  Qwen35 model(argv[1], 2048);
  if (!model.has_mtp()) fatal("model needs MTP");
  for (size_t i = 0; i < count; ++i)
    if (ids[i] >= model.config().vocab) fatal("token id %d at index %zu is outside the vocabulary", ids[i], i);
  auto seq = model.new_seq(int(count) + 32);
  std::copy_n(ids.begin(), count, seq->tokens());
  SamplingParams sp;
  for (int pos = 0; pos < prefix; pos += model.prefill_chunk()) {
    auto e = gpu.begin();
    const int count = std::min(model.prefill_chunk(), prefix - pos);
    model.encode_prefill(*e, *seq, pos, count, sp, 0, false);
    model.encode_mtp_prefill(*e, *seq, pos, count, 0);
    e->commit_and_wait();
    if (!model.ane_finish_chunk()) fatal("ANE prefill failed");
  }
  for (int i = 0; i < rows; ++i) {
    const int pos = prefix + i * R;
    // Restore teacher-forced inputs before each pass; the kernels may write a sampled next token.
    std::copy_n(ids.begin(), count, seq->tokens());
    auto e = gpu.begin();
    model.encode_verify(*e, *seq, pos, R, sp, 0);
    e->commit_and_wait();
    model.commit_rows(*seq, R);
    trunk.write(reinterpret_cast<const char*>(model.logits()), size_t(R) * model.config().vocab * 4);
    std::copy_n(ids.begin(), count, seq->tokens());
    e = gpu.begin();
    model.encode_mtp(*e, *seq, pos, R, pos + 1, model.hidden_rows(), 0, -1);
    e->commit_and_wait();
    mtp.write(reinterpret_cast<const char*>(model.mtp_post().ptr()), size_t(R) * model.config().hidden * 4);
    printf("prefix=%d pass=%d R=%d vocab=%d hidden=%d\n", prefix, i, R, model.config().vocab, model.config().hidden);
    fflush(stdout);
  }
  if (Encoder::profiling()) Encoder::profile_report(rows);
  trunk.close(); mtp.close();
  if (!trunk || !mtp) fatal("write failed");
  std::ofstream metadata(out / "run.json");
  metadata << nlohmann::json{{"model", argv[1]}, {"ids", argv[2]}, {"prefix", prefix}, {"passes", rows},
                            {"R", R}, {"records", rows * R}, {"vocab", model.config().vocab},
                            {"hidden", model.config().hidden}, {"commit", KAPITSA_GIT}, {"gpu", gpu.name()},
                            {"norm_x8", getenv("NI_NORM_X8") ? getenv("NI_NORM_X8") : "default"}}.dump(2) << '\n';
  metadata.close();
  if (!metadata) fatal("metadata write failed");
}
