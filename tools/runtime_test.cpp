// Control-plane checks and an engine-free subprocess fixture. No Metal linkage.
#include "server/runtime.h"
#include <httplib/httplib.h>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>

using namespace ni;
namespace fs = std::filesystem;
using runtime::json;

static void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
  std::cout << "PASS " << message << "\n";
}

int main(int argc, char** argv) {
  try {
    if (argc == 4 && std::string(argv[1]) == "--fixture") {
      std::atomic<bool> stopped = false;
      runtime::Instance instance(argv[2], argv[3], argv[3], "http://127.0.0.1:8000", "fake", "test",
                                 [&] { stopped = true; });
      instance.update({{"state", "ready"}, {"context", 32768}, {"slots", 4}});
      auto live = std::make_shared<LiveDecode>();
      instance.begin_request("fixture", live);
      live->phase = 2;
      live->prompt = 100;
      while (!stopped) {
        ++live->tokens;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      return 0;
    }
    char tmp[] = "/tmp/kapitsa-runtime-XXXXXX";
    const char* made = mkdtemp(tmp);
    if (!made) throw std::runtime_error("mkdtemp failed");
    struct Cleanup {
      std::string directory;
      ~Cleanup() { std::error_code ec; fs::remove_all(directory, ec); }
    } cleanup{made};
    const std::string dir = made;
    std::atomic<bool> stop_a = false, stop_b = false;
    json old_record;
    {
      runtime::Instance a(dir, "code", "Code", "http://127.0.0.1:8000", "27b", "test", [&] { stop_a = true; });
      runtime::Instance b(dir, "chat", "Chat", "http://127.0.0.1:8001", "4b", "test", [&] { stop_b = true; });
      old_record = a.record();
      check(runtime::discover(dir).size() == 2, "two independent instances are discoverable");
      check(a.record().at("control_url") != b.record().at("control_url"), "each instance owns its control endpoint");
      struct stat st {};
      stat((dir + "/code.json").c_str(), &st);
      check((st.st_mode & 0777) == 0600, "registry credentials are private");
      check(runtime::call(a.record(), "GET", "/v1/status").at("state") == "starting",
            "control is available before engine loading");
      bool refused = false;
      try {
        runtime::Instance duplicate(dir, "code", "Code", "", "", "", [] {});
      } catch (const std::exception&) { refused = true; }
      check(refused && runtime::call(a.record(), "GET", "/v1/status").at("instance_id") == "code",
            "duplicate start is refused without removing the first record");
      check(fs::exists(dir + "/code.json"), "failed duplicate leaves the owner record intact");
      httplib::Client unauthorized(a.record().at("control_url").get<std::string>());
      const auto denied = unauthorized.Get("/v1/status");
      check(denied && denied->status == 401, "control requires the private credential");
      json wrong_run = a.record();
      wrong_run["run_id"] = runtime::new_id();
      refused = false;
      try { runtime::call(wrong_run, "POST", "/v1/stop"); }
      catch (const std::exception&) { refused = true; }
      check(refused && !stop_a, "stale execution cannot stop the current instance");
      auto live = std::make_shared<LiveDecode>();
      const std::string id = a.begin_request("pi", live);
      live->phase = 2;
      live->tokens = 163;
      live->cycles = 53;
      live->prompt = 4100;
      live->cached = 100;
      live->spec = 2;
      a.update({{"state", "ready"}, {"context", 90000}, {"slots", 4}});
      a.details([] { return json{{"memory_used_bytes", 21500000000ull}}; });
      const json first = runtime::call(a.record(), "GET", "/v1/events?after=0");
      check(first.at("requests").size() == 1 && first["requests"][0]["completion_tokens"] == 163,
            "native progress is observable without a terminal");
      check(first["requests"][0]["phase"] == "writing" && first["requests"][0]["spec"] == 2,
            "phase and speculative decoding remain structured");
      check(first.at("memory_used_bytes") == 21500000000ull, "runtime details augment the state");
      live->tokens = 170;
      check(runtime::call(a.record(), "GET", "/v1/status")["requests"][0]["completion_tokens"] == 170,
            "observer disconnect does not cancel a request");
      ChatResult result;
      result.finish_reason = "stop";
      result.prompt_tokens = 4100;
      result.cached_tokens = 100;
      result.completion_tokens = 170;
      result.decode_ms = 4225;
      a.finish_request(id, result);
      const auto finished = runtime::call(a.record(), "GET", "/v1/events?after=" + first.at("cursor").dump());
      check(finished.at("requests").empty() && finished["last_request"]["completion_tokens"] == 170,
            "finished requests leave an exact usage summary");
      check(finished["events"].size() == 1 && finished["events"][0]["type"] == "request_finished",
            "event replay starts after the saved cursor");
      check(runtime::call(a.record(), "GET", "/v1/events?after=" + finished.at("cursor").dump())["events"].empty(),
            "repeated polling does not replay consumed events");
      for (int n = 0; n < 300; ++n) a.event("log", {{"message", std::to_string(n)}});
      const auto gap = runtime::call(a.record(), "GET", "/v1/events?after=0");
      check(gap.at("reset") == true && gap.at("events").size() == 256, "bounded history reports a replay gap");
      {
        std::ofstream malformed(dir + "/broken.json");
        malformed << "broken";
      }
      fs::create_symlink(dir + "/code.json", dir + "/link.json");
      check(runtime::discover(dir).size() == 2, "malformed and symlink records do not hide healthy instances");
      const auto stopping = runtime::call(b.record(), "POST", "/v1/stop");
      check(stopping.at("state") == "stopping", "stop acknowledges the selected instance");
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (!stop_b && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      check(stop_b && !stop_a, "stopping one instance leaves the other untouched");
    }
    check(!fs::exists(dir + "/code.json") && !fs::exists(dir + "/chat.json"), "normal shutdown removes owned records");
    {
      runtime::Instance next(dir, "code", "Code", "", "", "test", [] {});
      check(next.record().at("instance_id") == old_record.at("instance_id") &&
            next.record().at("run_id") != old_record.at("run_id"), "restart preserves instance identity and changes execution");
    }
    check(!runtime::valid_id("../escape") && !runtime::valid_id("") && !runtime::valid_id(std::string(65, 'x')),
          "instance identity cannot escape the registry");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << "\n";
    return 1;
  }
}
