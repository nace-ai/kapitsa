// CPU-only import transaction checks with synthetic GGUF weights and injected I/O failures.
#include "gguf/import.h"
#include <nlohmann/json.hpp>
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {
std::string root, mode;
bool armed = false;
int writes = 0, injected = 0, sentinel = -1;
std::set<int> outputs;
bool output_fd(int fd) {
  if (!armed) return false;
  char path[4096]{};
  if (fcntl(fd, F_GETPATH, path) != 0) return false;
  const std::string p(path);
  const bool match = p.rfind(root + "/", 0) == 0 && p.find(".importing/") != std::string::npos &&
                     p.ends_with(".safetensors");
  if (match) outputs.insert(fd);
  return match;
}
void check(bool ok, const char* message) {
  if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
template<class T> void put(std::ostream& f, T v) { f.write(reinterpret_cast<const char*>(&v), sizeof v); }
void str(std::ostream& f, const std::string& s) { put<uint64_t>(f, s.size()); f << s; }
using J = nlohmann::json;
void value(std::ostream& f, const J& v, bool type = true) {
  if (v.is_string()) { if (type) put<uint32_t>(f, 8); str(f, v.get<std::string>()); }
  else if (v.is_array()) {
    if (type) put<uint32_t>(f, 9);
    put<uint32_t>(f, v.empty() || v[0].is_string() ? 8 : 5);
    put<uint64_t>(f, v.size());
    for (const auto& x : v) value(f, x, false);
  } else { if (type) put<uint32_t>(f, 5); put<int32_t>(f, v.get<int32_t>()); }
}
void fixture(const std::string& path, bool mtp) {
  J m = {{"general.architecture", "qwen35"}, {"qwen35.embedding_length", 64}, {"qwen35.block_count", 1},
         {"qwen35.attention.head_count", 1}, {"qwen35.attention.head_count_kv", 1},
         {"qwen35.attention.key_length", 256}, {"qwen35.ssm.conv_kernel", 4}, {"qwen35.ssm.state_size", 128},
         {"qwen35.ssm.group_count", 1}, {"qwen35.ssm.time_step_rank", 1}, {"qwen35.ssm.inner_size", 128},
         {"qwen35.full_attention_interval", 1}, {"qwen35.feed_forward_length", 64},
         {"tokenizer.ggml.model", "gpt2"}, {"tokenizer.ggml.pre", "qwen35"},
         {"tokenizer.ggml.tokens", {"a", "b", "<|endoftext|>"}}, {"tokenizer.ggml.token_type", {1, 1, 3}},
         {"tokenizer.ggml.merges", J::array()}, {"tokenizer.ggml.eos_token_id", 2},
         {"tokenizer.chat_template", "{{ messages[0]['content'] }}"}};
  std::map<std::string, std::vector<uint64_t>> tensors = {
    {"token_embd.weight", {64, 3}}, {"output_norm.weight", {64}},
    {"blk.0.attn_norm.weight", {64}}, {"blk.0.post_attention_norm.weight", {64}},
    {"blk.0.attn_q.weight", {64, 512}}, {"blk.0.attn_k.weight", {64, 256}},
    {"blk.0.attn_v.weight", {64, 256}}, {"blk.0.attn_output.weight", {256, 64}},
    {"blk.0.attn_q_norm.weight", {256}}, {"blk.0.attn_k_norm.weight", {256}},
    {"blk.0.ffn_gate.weight", {64, 64}}, {"blk.0.ffn_up.weight", {64, 64}},
    {"blk.0.ffn_down.weight", {64, 64}}};
  if (mtp) {
    m["qwen35.block_count"] = 2;
    m["qwen35.nextn_predict_layers"] = 1;
    const auto trunk = tensors;
    for (const auto& [name, dims] : trunk)
      if (name.starts_with("blk.0.")) tensors["blk.1." + name.substr(6)] = dims;
    tensors["blk.1.nextn.enorm.weight"] = {64};
    tensors["blk.1.nextn.hnorm.weight"] = {64};
    tensors["blk.1.nextn.eh_proj.weight"] = {128, 64};
  }
  std::ofstream f(path, std::ios::binary);
  f.write("GGUF", 4); put<uint32_t>(f, 3); put<uint64_t>(f, tensors.size()); put<uint64_t>(f, m.size());
  for (auto it = m.begin(); it != m.end(); ++it) { str(f, it.key()); value(f, it.value()); }
  uint64_t offset = 0;
  for (const auto& [name, dims] : tensors) {
    str(f, name); put<uint32_t>(f, dims.size());
    uint64_t count = 1;
    for (auto d : dims) { put<uint64_t>(f, d); count *= d; }
    put<uint32_t>(f, 0); put<uint64_t>(f, offset); offset += count * sizeof(float);
  }
  while (uint64_t(f.tellp()) % 32) f.put('\0');
  for (uint64_t i = 0; i < offset / 4; ++i) put<float>(f, float(int(i % 23) - 11) / 16);
  check(bool(f), "write synthetic GGUF");
}
std::map<std::string, std::string> contents(const std::string& dir) {
  std::map<std::string, std::string> result;
  for (const auto& e : std::filesystem::recursive_directory_iterator(dir)) {
    if (!e.is_regular_file()) continue;
    std::ifstream f(e.path(), std::ios::binary);
    result[e.path().lexically_relative(dir).string()] =
        std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  }
  return result;
}
}

// Interposition is confined to this test executable and its own output files.
extern "C" ssize_t pwrite(int fd, const void* data, size_t count, off_t offset) {
  static auto real = reinterpret_cast<ssize_t(*)(int, const void*, size_t, off_t)>(dlsym(RTLD_NEXT, "pwrite"));
  if (output_fd(fd)) {
    ++writes;
    if (!injected && ((mode == "header" && offset == 0) || (mode == "data" && offset > 8) ||
                      (mode == "second-header" && outputs.size() == 2 && offset == 0))) {
      ++injected; errno = EIO; return -1;
    }
    if (mode == "cancel" && writes == 2) { ++injected; raise(SIGINT); }
  }
  return real(fd, data, count, offset);
}
extern "C" int ftruncate(int fd, off_t length) {
  static auto real = reinterpret_cast<int(*)(int, off_t)>(dlsym(RTLD_NEXT, "ftruncate"));
  if (output_fd(fd) && (mode == "truncate" || (mode == "second-truncate" && outputs.size() == 2))) {
    ++injected; errno = EIO; return -1;
  }
  return real(fd, length);
}
extern "C" int close(int fd) {
  static auto real = reinterpret_cast<int(*)(int)>(dlsym(RTLD_NEXT, "close"));
  const bool fail = armed && mode == "close" && output_fd(fd) && !injected;
  const int result = real(fd);
  if (fail) {
    ++injected;
    sentinel = open("/dev/null", O_RDONLY);  // a newly reused descriptor must not be closed again during unwinding
    check(sentinel == fd, "reuse closed descriptor for close-error control");
    errno = EIO;
    return -1;
  }
  return result;
}

int main() {
  std::string temp = (std::filesystem::temp_directory_path() / "gguf-output-test-XXXXXX").string();
  const char* made = mkdtemp(temp.data()); check(made, "create test directory");
  root = std::filesystem::canonical(made).string();
  setenv("KAPITSA_HOME", root.c_str(), 1);
  const std::string input = root + "/input.gguf";
  ni::gguf::ImportOptions o;
  o.file = input; o.name = "fixture"; o.threads = 1; o.force = true; o.progress = false; o.drafter = false;
  bool ok = true;
  for (const bool mtp : {false, true}) {
    fixture(input, mtp);
    std::map<std::string, std::string> original;
    for (const char* which : {"none", "truncate", "header", "second-truncate", "second-header", "data", "cancel", "close", "none"}) {
      if (!mtp && std::string(which).starts_with("second-")) continue;
      mode = which; writes = injected = 0; outputs.clear(); sentinel = -1;
      bool threw = false, diagnostic = true;
      armed = true;
      try { ni::gguf::import_model(o); }
      catch (const std::exception& e) {
        threw = true;
        const std::string message(e.what());
        diagnostic = mode == "cancel" ? message == "Stopped; nothing was kept. The same command starts the conversion again."
                                      : message.starts_with("Can't write ") && message.ends_with(std::string(strerror(EIO)) + ".");
      }
      armed = false;
      const bool expected = mode != "none";
      const bool sentinel_ok = sentinel < 0 || fcntl(sentinel, F_GETFD) >= 0;
      int leaked = 0;
      for (int fd : outputs) if (fd != sentinel && fcntl(fd, F_GETFD) >= 0) ++leaked;
      const bool removed = !std::filesystem::exists(root + "/models/.fixture.importing");
      const auto current = contents(root + "/models/fixture");
      if (original.empty()) original = current;
      const bool exact = !current.empty() && current == original;
      const bool pass = threw == expected && injected == int(expected) && leaked == 0 && removed && sentinel_ok && exact && diagnostic;
      std::printf("mtp=%d case=%s injected=%d threw=%d leaked=%d temporary_removed=%d reused_fd_open=%d files_equal=%d diagnostic=%d pass=%d\n",
                  mtp, which, injected, threw, leaked, removed, sentinel_ok, exact, diagnostic, pass);
      ok &= pass;
      mode.clear();
      for (int fd : outputs) if (fd != sentinel && fcntl(fd, F_GETFD) >= 0) close(fd);
      if (sentinel >= 0) close(sentinel);
    }
  }
  std::filesystem::remove_all(root);
  return ok ? 0 : 1;
}
