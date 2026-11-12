// Exercise Flash-Next's actual index-cache growth and compare every retained byte.
// Usage: index_growth_check MODEL OLD_CAP NEW_CAP
// Takes the GPU lock before loading the model. Capacity arguments must be positive and increasing.
#include "bench/bench.h"
#include "model/qwen4_exp.h"
#include "util/log.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <string_view>
#include <thread>
#include <vector>

using namespace ni;

int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: index_growth_check MODEL OLD_CAP NEW_CAP\n");
    return 2;
  }
  auto capacity = [](std::string_view text) {
    int n = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), n);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size() ? n : 0;
  };
  const int old_cap = capacity(argv[2]), new_cap = capacity(argv[3]);
  if (old_cap < 1 || new_cap <= old_cap) return 2;
  bench::GpuLock lock;
  if (!bench::GpuLock::held_by_caller() && !lock.acquire()) return 2;
  Qwen4Exp model(argv[1], 32);
  auto s = model.new_seq(new_cap, old_cap);
  auto buffers = [&]() {
    std::vector<Buffer*> out;
    for (auto* v : {&s->ixk, &s->ixb})
      for (auto& b : *v)
        if (b) out.push_back(&b);
    for (auto* b : {&s->mtp_ixk, &s->mtp_ixb})
      if (*b) out.push_back(b);
    return out;
  };
  auto bs = buffers();
  if (bs.empty()) fatal("model has no index caches");
  std::vector<size_t> sizes;
  for (size_t j = 0; j < bs.size(); ++j) {
    sizes.push_back(bs[j]->size);
    auto* p = bs[j]->as<unsigned char>();
    for (size_t i = 0; i < sizes.back(); ++i) p[i] = static_cast<unsigned char>((i * 37 + j * 19) % 251);
  }
  auto& gpu = Gpu::get();
  const size_t before = gpu.allocated_bytes();
  size_t peak = before;
  std::jthread sampler([&](std::stop_token stop) {
    while (!stop.stop_requested()) {
      peak = std::max(peak, gpu.allocated_bytes());
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });
  const auto start = std::chrono::steady_clock::now();
  if (!model.reserve(*s, new_cap)) fatal("growth refused");
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  sampler.request_stop();
  sampler.join();
  const size_t after = gpu.allocated_bytes();
  peak = std::max(peak, after);
  printf("sampled_allocation_before=%zu after=%zu peak=%zu interval_us=100\n", before, after, peak);
  size_t checked = 0, errors = 0;
  for (size_t j = 0; j < bs.size(); ++j) {
    if (bs[j]->size < sizes[j]) fatal("cache shrank");
    const auto* p = bs[j]->as<unsigned char>();
    for (size_t i = 0; i < sizes[j]; ++i) errors += p[i] != static_cast<unsigned char>((i * 37 + j * 19) % 251);
    checked += sizes[j];
  }
  printf("buffers=%zu retained_bytes=%zu mismatches=%zu growth_ms=%.3f capacity=%d\n", bs.size(), checked, errors, ms, s->cap);
  model.drop_kv(*s);
  return errors ? 1 : 0;
}
