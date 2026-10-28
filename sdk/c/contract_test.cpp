#include "kapitsa.h"
#include "options.h"
#include <iostream>
#include <sys/file.h>
#include <unistd.h>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <future>
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
template<class F> void invalid(F f) {
  bool threw = false;
  try { f(); } catch (const std::exception&) { threw = true; }
  CHECK(threw);
}
int main() {
  CHECK(kapitsa_abi_version() == 1);
  for (auto bytes : {std::string("\xe4\xbd"), std::string("\xa0")})
    CHECK(ni::ojson::parse(kapitsa_sdk::dump(ni::ojson{{"content", bytes}}))["content"] == "\xef\xbf\xbd");
  kapitsa_engine* engine = nullptr;
  char* error = nullptr;
  CHECK(kapitsa_engine_create("{", &engine, &error) == KAPITSA_INVALID_ARGUMENT);
  CHECK(!engine && error); kapitsa_string_free(error);
  CHECK(kapitsa_engine_create("{}", &engine, &error) == KAPITSA_INVALID_ARGUMENT);
  CHECK(!engine && error); kapitsa_string_free(error);
  CHECK(kapitsa_engine_create("{\"model_dir\":\"/missing-kapitsa-sdk-model\"}", &engine, &error) == KAPITSA_INVALID_ARGUMENT);
  CHECK(!engine && error); kapitsa_string_free(error);
  char path[] = "/tmp/kapitsa-sdk-lock-XXXXXX";
  int fd = mkstemp(path);
  CHECK(fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) == 0);
  setenv("BENCH_GPU_LOCK", path, 1);
  unsetenv("NI_LOCK_HELD"); unsetenv("NACE_GPU_LOCK_HELD");
  std::string dir = std::string(path) + "-model";
  std::filesystem::create_directory(dir);
  std::ofstream(dir + "/config.json") << "{}";
  ni::ojson config{{"model_dir", dir}};
  using Clock = std::chrono::steady_clock;
  auto create = [&](kapitsa_status expected, const char* message) {
    const auto start = Clock::now();
    CHECK(kapitsa_engine_create(config.dump().c_str(), &engine, &error) == expected);
    CHECK(!engine && error && std::strstr(error, message)); kapitsa_string_free(error);
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
  };
  alarm(10);  // A regression to blocking acquire must fail, not hang CI.
  CHECK(create(KAPITSA_BUSY, "GPU lock") < 1000);
  config["gpu_lock_timeout_ms"] = 0;
  CHECK(create(KAPITSA_BUSY, "GPU lock") < 1000);
  config["gpu_lock_timeout_ms"] = 150;
  const auto waited = create(KAPITSA_BUSY, "GPU lock");
  CHECK(waited >= 140 && waited < 1500);
  for (auto value : {"-1", "1.5", "true", "null", "\"150\"", "2147483648", "18446744073709551615"}) {
    config["gpu_lock_timeout_ms"] = ni::ojson::parse(value);
    create(KAPITSA_INVALID_ARGUMENT, "gpu_lock_timeout_ms");
  }
  config["gpu_lock_timeout_ms"] = 2000;
  // {} is rejected by Engine::unsupported_model before any model load or GPU initialization.
  // Seeing its error proves the timed wait acquired the lock and reached the loader.
  auto release = std::async(std::launch::async, [&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(60)); flock(fd, LOCK_UN);
  });
  CHECK(create(KAPITSA_ERROR, "model_type") >= 50);
  release.get();
  CHECK(flock(fd, LOCK_EX | LOCK_NB) == 0);  // failed construction released its lock/state
  config["gpu_lock_timeout_ms"] = 0;
  create(KAPITSA_BUSY, "GPU lock");
  for (auto marker : {"NI_LOCK_HELD", "NACE_GPU_LOCK_HELD"}) {
    setenv(marker, "1", 1);
    create(KAPITSA_ERROR, "model_type");  // caller-held lock must not deadlock on itself
    unsetenv(marker);
  }
  flock(fd, LOCK_UN);
  // A wrong field type in model data is a load error, not a malformed SDK option.
  std::ofstream(dir + "/config.json") << R"({"model_type":42})";
  create(KAPITSA_ERROR, "type");
  CHECK(flock(fd, LOCK_EX | LOCK_NB) == 0);
  const auto bad_path = std::string(path) + "/not-a-directory/lock";
  setenv("BENCH_GPU_LOCK", bad_path.c_str(), 1);
  config["gpu_lock_timeout_ms"] = 2000;
  CHECK(create(KAPITSA_ERROR, "GPU lock") < 1000);  // I/O error is not contention
  alarm(0);
  close(fd); unlink(path); unsetenv("BENCH_GPU_LOCK");
  std::filesystem::remove_all(dir);
  char* result = nullptr;
  CHECK(kapitsa_generate(nullptr, nullptr, nullptr, &result, &error) == KAPITSA_INVALID_ARGUMENT);
  CHECK(!result && error); kapitsa_string_free(error);
  // Exercise the full integer range without converting an unsigned value to signed first.
  CHECK(kapitsa_sdk::integer(ni::ojson{{"depth", -1}}, "depth", 0, -1, 32) == -1);
  CHECK(kapitsa_sdk::integer(ni::ojson{{"depth", 32}}, "depth", 0, -1, 32) == 32);
  for (const char* value : {"-2", "33", "9223372036854775808", "18446744073709551615"}) {
    const auto json = ni::ojson::parse(std::string("{\"depth\":") + value + "}");
    invalid([&] { kapitsa_sdk::integer(json, "depth", 0, -1, 32); });
  }
  ni::ChatRequest defaults;
  auto good = ni::ojson::parse(R"({"messages":[{"role":"user","content":"hello"}],"temperature":0,"seed":0})");
  auto r = kapitsa_sdk::request(good, defaults);
  CHECK(r.temperature == 0 && r.seed == 0 && r.max_tokens == defaults.max_tokens);
  good["seed"] = UINT32_MAX;
  CHECK(kapitsa_sdk::request(good, defaults).seed == UINT32_MAX);
  for (auto bad : {R"({"messages":[]})", R"({"messages":[{"role":"user"}],"max_tokens":-1})",
      R"({"messages":[{"role":"user"}],"seed":4294967296})",
      R"({"messages":[{"role":"user"}],"temperature":true})",
      R"({"messages":[{"role":"user"}],"typo":1})"})
    invalid([&] { kapitsa_sdk::request(ni::ojson::parse(bad), defaults); });
  kapitsa_request_cancel(nullptr); kapitsa_request_destroy(nullptr);
  kapitsa_engine_destroy(nullptr); kapitsa_string_free(nullptr);
  std::cout << "SDK CPU contract checks passed (no model or GPU)\n";
}
