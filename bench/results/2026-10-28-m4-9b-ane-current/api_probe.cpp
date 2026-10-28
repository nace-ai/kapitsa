// An HTTP client using the frozen baseline's benchmark protocol, not its model implementation.
#include "bench/bench.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

using namespace ni::bench;
namespace fs = std::filesystem;

static double epoch() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

static void save(const fs::path& path, const ojson& value) {
  if (fs::exists(path)) throw std::runtime_error("refusing to overwrite " + path.string());
  std::ofstream f(path);
  f << value.dump(2) << '\n';
  if (!f) throw std::runtime_error("could not save " + path.string());
}

int main(int argc, char** argv) {
  try {
    if (argc < 5) throw std::runtime_error("api_probe warmup|request|memory PORT OUT_DIR NONCE [CONTEXT_CHARS_TARGET_TOKENS OUTPUT_TOKENS]");
    const std::string mode = argv[1], nonce = argv[4];
    Settings s;
    s.port = std::atoi(argv[2]);
    s.model = "bench-model";
    const fs::path out = argv[3];
    fs::create_directories(out);
    if (mode == "warmup") {
      Nonces n;
      n.fixed = true;
      const ojson w = warmup(s, n);
      save(out / "warmup.json", w);
      if (!w.value("ok", false)) throw std::runtime_error("native warmup failed");
      return 0;
    }
    if (mode == "memory") {
      const Footprint f = footprint(process_tree({std::atoi(nonce.c_str())}));
      save(out / "memory.json", {{"epoch", epoch()}, {"phys_bytes", f.phys}, {"peak_bytes", f.peak}});
      return 0;
    }
    if ((mode != "request" && mode != "request-file") || argc != 7)
      throw std::runtime_error("request needs context size or messages file, and output count");
    const int context = std::atoi(argv[5]);
    s.decode_tokens = std::atoi(argv[6]);
    ojson msgs;
    if (mode == "request-file") {
      std::ifstream f(argv[5]);
      if (!f) throw std::runtime_error("cannot read messages file");
      f >> msgs;
      if (!msgs.is_array()) throw std::runtime_error("messages must be an array");
    } else {
      msgs = context ? prefill_messages(build_context(context), nonce, kDecodeUser) : decode_messages(nonce);
    }
    const ojson body = body_of(s, msgs, s.decode_tokens, true);
    save(out / "request.json", body);
    const double begin = epoch();
    const Stream r = stream_chat(s, body);
    const double end = epoch();
    const int ct = r.usage.value("completion_tokens", 0);
    const Rate rates = decode_rates(r.ev_t, r.ev_c, ct, s.window_head, s.window_tail);
    const auto distinct = distinct4(r.content);
    const auto cached = cached_of(r.usage, r.timings);
    const auto leaks = thinking_leak(r);
    ojson invalid = ojson::array();
    if (ct != s.decode_tokens) invalid.push_back("completion count mismatch");
    if (r.content.empty()) invalid.push_back("empty visible text");
    if (r.finish != "length") invalid.push_back("finish_reason is not length");
    if (!r.reasoning.empty() || !leaks.empty()) invalid.push_back("unexpected reasoning");
    if (!r.tool_calls.empty()) invalid.push_back("unexpected tool call");
    if (!rates.mid || !rates.e2e) invalid.push_back("missing streamed rate");
    if (distinct && *distinct < s.rep_threshold) invalid.push_back("repetition loop");
    ojson result = {{"request_sha256", sha256_hex(body.dump())}, {"begin_epoch", begin}, {"end_epoch", end},
                    {"content", r.content}, {"reasoning", r.reasoning}, {"content_sha256", sha256_hex(r.content)},
                    {"reasoning_sha256", sha256_hex(r.reasoning)}, {"finish", r.finish}, {"usage", r.usage},
                    {"timings", r.timings}, {"ttft_s", r.ttft}, {"gen_time_s", r.gen_time}, {"total_s", r.total},
                    {"event_times_s", r.ev_t}, {"event_chars", r.ev_c}, {"chunks", r.n_chunks},
                    {"completion_tokens", ct}, {"prompt_tokens", r.usage.value("prompt_tokens", 0)},
                    {"cached_tokens", cached ? ojson(*cached) : ojson(nullptr)},
                    {"native_mid_tps", rates.mid ? ojson(*rates.mid) : ojson(nullptr)},
                    {"stream_generation_tps", rates.e2e ? ojson(*rates.e2e) : ojson(nullptr)},
                    {"rate_details", rates.info}, {"distinct4", distinct ? ojson(*distinct) : ojson(nullptr)},
                    {"invalid", invalid}};
    save(out / "result.json", result);
    result.erase("content");
    result.erase("reasoning");
    result.erase("event_times_s");
    result.erase("event_chars");
    std::puts(result.dump().c_str());
    return invalid.empty() ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "api_probe: %s\n", e.what());
    return 1;
  }
}
