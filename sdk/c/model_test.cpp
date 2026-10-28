// Opt-in real-model test; never registered with CTest. Requires an authorized, idle GPU.
#include "kapitsa.h"
#include <nlohmann/json.hpp>
#include <atomic>
#include <barrier>
#include <chrono>
#include <filesystem>
#include <future>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

using Json = nlohmann::ordered_json;
using Engine = std::unique_ptr<kapitsa_engine, decltype(&kapitsa_engine_destroy)>;
using Request = std::unique_ptr<kapitsa_request, decltype(&kapitsa_request_destroy)>;
using String = std::unique_ptr<char, decltype(&kapitsa_string_free)>;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
struct TempLock {
  std::string dir;
  TempLock() {
    char path[] = "/tmp/kapitsa-sdk-model-XXXXXX";
    CHECK(mkdtemp(path)); dir = path;
    setenv("BENCH_GPU_LOCK", (dir + "/gpu.lock").c_str(), 1);
    unsetenv("NI_LOCK_HELD"); unsetenv("NACE_GPU_LOCK_HELD");
  }
  ~TempLock() { std::filesystem::remove_all(dir); }
  bool available() const {
    int fd = open((dir + "/gpu.lock").c_str(), O_RDONLY);
    CHECK(fd >= 0);
    const bool ok = flock(fd, LOCK_EX | LOCK_NB) == 0;
    close(fd); return ok;
  }
};
struct Stream {
  std::atomic<int> deltas{0};
  std::atomic<bool> valid{true};
  bool cancel = false;
};
int delta(const char* data, size_t length, void* user) noexcept {
  auto& s = *static_cast<Stream*>(user);
  try {
    auto j = Json::parse(data, data + length);
    if (!j.contains("content") || !j.contains("reasoning") || !j.contains("tool_calls")) s.valid = false;
    ++s.deltas;
    return s.valid && !s.cancel;
  } catch (...) { s.valid = false; return 0; }  // never unwind through the C callback
}
Request request(kapitsa_engine* engine, const std::string& prompt, int tokens = 16) {
  Json j{{"messages", Json::array({{{"role", "user"}, {"content", prompt}}})},
         {"max_tokens", tokens}, {"temperature", 0}, {"template_kwargs", {{"enable_thinking", false}}}};
  kapitsa_request* raw = nullptr; char* error = nullptr;
  const auto status = kapitsa_request_create(engine, j.dump().c_str(), &raw, &error);
  String message(error, kapitsa_string_free);
  Request result(raw, kapitsa_request_destroy);
  CHECK(status == KAPITSA_OK && result && !message);
  return result;
}
Json generate(kapitsa_request* r, Stream& stream, kapitsa_status expected = KAPITSA_OK) {
  char* output = nullptr; char* error = nullptr;
  const auto status = kapitsa_generate(r, delta, &stream, &output, &error);
  String result(output, kapitsa_string_free), message(error, kapitsa_string_free);
  if (message) std::cerr << message.get() << '\n';
  CHECK(status == expected && stream.valid);
  if (!result) return nullptr;
  return Json::parse(result.get());
}
void failed_loads(const TempLock& lock, const char* model) {
  const auto dir = std::filesystem::path(lock.dir) / "broken-model";
  std::filesystem::create_directory(dir);
  std::filesystem::copy_file(std::filesystem::path(model) / "config.json", dir / "config.json");
  Json options{{"model_dir", dir.string()}, {"slots", 1}, {"context", 64},
               {"prefill_chunk", 64}, {"spec_depth", 0}, {"dflash", "none"}, {"mem_budget_gb", 32}};
  auto check_error = [&](const char* expected) {
    kapitsa_engine* raw = nullptr; char* error = nullptr;
    const auto status = kapitsa_engine_create(options.dump().c_str(), &raw, &error);
    String message(error, kapitsa_string_free);
    if (message) std::cout << "expected load error: " << message.get() << '\n';
    CHECK(status == KAPITSA_ERROR && !raw && message);
    CHECK(std::string(message.get()).find(expected) != std::string::npos);
    CHECK(lock.available());
  };
  check_error(".safetensors");
  std::ofstream(dir / "model.safetensors", std::ios::binary) << "broken";
  check_error("incomplete");
  // A valid small shard is allocated and read, then the incomplete weight set is refused.
  // Repeating it exercises teardown of an incomplete model before a real load below.
  Json config; std::ifstream(model + std::string("/config.json")) >> config;
  const int hidden = config.contains("text_config") ? int(config["text_config"]["hidden_size"]) : int(config["hidden_size"]);
  const auto header = Json{{"model.norm.weight", {{"dtype", "BF16"}, {"shape", {hidden}},
                                                {"data_offsets", {0, hidden * 2}}}}}.dump();
  {
    std::ofstream f(dir / "model.safetensors", std::ios::binary | std::ios::trunc);
    uint64_t length = header.size(); f.write(reinterpret_cast<char*>(&length), 8);
    f << header << std::string(size_t(hidden) * 2, '\0');
  }
  for (int i = 0; i < 3; ++i) check_error("unrecognized weight naming");
  options["model_dir"] = model; options["mem_budget_gb"] = 0.000001;
  check_error("memory");
  std::cout << "PASS missing/corrupt/incomplete models and memory refusal return ERROR; retry unlocks\n";
}
int main(int argc, char** argv) {
  if (argc != 2) { std::cerr << "usage: sdk_model_test MODEL_DIR\n"; return 2; }
  alarm(300);  // deadlock fails the process, without destroying handles still in use
  try {
    TempLock lock;
    failed_loads(lock, argv[1]);
    Json options{{"model_dir", argv[1]}, {"slots", 2}, {"context", 4096},
                 {"prefill_chunk", 512}, {"spec_depth", 0}, {"dflash", "none"}};
    kapitsa_engine* raw = nullptr; char* error = nullptr;
    auto status = kapitsa_engine_create(options.dump().c_str(), &raw, &error);
    String message(error, kapitsa_string_free);
    if (message) std::cerr << message.get() << '\n';
    Engine engine(raw, kapitsa_engine_destroy);
    CHECK(status == KAPITSA_OK && engine && !lock.available());
    // Enough rows to exercise ANE split prefill when available, then prefix reuse.
    std::string prompt;
    for (int i = 0; i < 300; ++i) prompt += "The sky is blue. ";
    prompt += "Reply with a short greeting.";
    Json previous;
    for (int i = 0; i < 3; ++i) {
      auto r = request(engine.get(), prompt);
      Stream s; s.cancel = i == 2;
      auto result = generate(r.get(), s, s.cancel ? KAPITSA_CANCELLED : KAPITSA_OK);
      CHECK(s.deltas > 0 && result["completion_tokens"] > 0);
      if (i == 1) CHECK(result["content"] == previous["content"] && result["cached_tokens"] > 1000);
      if (i == 2) CHECK(result["finish_reason"] == "cancelled");
      previous = result;
      std::cout << Json{{"case", "prefill-reuse-cancel"}, {"round", i}, {"result", result}} << std::endl;
    }
    {
      auto r = request(engine.get(), "Hello");
      Stream s;
      kapitsa_request_cancel(r.get());
      CHECK(generate(r.get(), s, KAPITSA_CANCELLED).is_null() && s.deltas == 0);
      CHECK(generate(r.get(), s, KAPITSA_BUSY).is_null());
    }
    std::cout << "PASS pre-cancel and one-shot request\n";
    for (int round = 0; round < 3; ++round) {
      auto a = request(engine.get(), "List many facts about " + std::to_string(round + 7), 128);
      auto b = request(engine.get(), "List many facts about " + std::to_string(round + 17), 64);
      if (round == 2) {
        engine.reset();
        CHECK(!lock.available());  // requests retain the model and its lifetime-held lock
        options["gpu_lock_timeout_ms"] = 2000;
        const auto start = std::chrono::steady_clock::now();
        status = kapitsa_engine_create(options.dump().c_str(), &raw, &error);
        String busy(error, kapitsa_string_free);
        CHECK(status == KAPITSA_BUSY && !raw && busy);
        CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds(1));
      }
      Stream sa, sb;
      std::barrier start(3);
      auto fa = std::async(std::launch::async, [&] { start.arrive_and_wait(); return generate(a.get(), sa, KAPITSA_CANCELLED); });
      auto fb = std::async(std::launch::async, [&] { start.arrive_and_wait(); return generate(b.get(), sb); });
      auto cancel = std::async(std::launch::async, [&] {
        start.arrive_and_wait();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        while ((!sa.deltas || !sb.deltas) && std::chrono::steady_clock::now() < deadline)
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        kapitsa_request_cancel(a.get());
        CHECK(sa.deltas > 0 && sb.deltas > 0);
      });
      cancel.get();
      auto ra = fa.get(), rb = fb.get();
      CHECK(ra["finish_reason"] == "cancelled" && ra["completion_tokens"] > 0);
      CHECK((rb["finish_reason"] == "stop" || rb["finish_reason"] == "length") && rb["completion_tokens"] > 0);
      std::cout << Json{{"case", "two-requests-third-thread-cancel"}, {"round", round}, {"a", ra}, {"b", rb}} << std::endl;
    }
    CHECK(lock.available());
    options["slots"] = 1; options["context"] = 512;
    status = kapitsa_engine_create(options.dump().c_str(), &raw, &error);
    String again_error(error, kapitsa_string_free);
    Engine again(raw, kapitsa_engine_destroy);
    CHECK(status == KAPITSA_OK && again && !again_error);
    { auto r = request(again.get(), "Hello"); Stream s; CHECK(generate(r.get(), s)["completion_tokens"] > 0); }
    again.reset();
    CHECK(lock.available());
    std::cout << "PASS destroy, reload and generate in the same process\n";
    std::cout << "PASS retained requests and lock release\nPASS all C ABI model checks\n";
  } catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
  alarm(0);
}
