// Split-file metadata is checked after every part has opened; rejection must release all of them.
#pragma once
#include "gguf/gguf.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace gguf_split_cleanup_test {

inline void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

inline int descriptors() {
  int n = 0;
  for (int fd = 0; fd < 1024; ++fd) n += fcntl(fd, F_GETFD) != -1;
  return n;
}

inline void run() {
  char pattern[] = "/tmp/kapitsa-gguf-split-XXXXXX";
  require(mkdtemp(pattern) != nullptr, "create split fixture directory");
  const std::filesystem::path dir(pattern);
  struct Cleanup {
    std::filesystem::path dir;
    ~Cleanup() { std::error_code ec; std::filesystem::remove_all(dir, ec); }
  } cleanup{dir};
  const auto a = dir / "model-00001-of-00002.gguf", b = dir / "model-00002-of-00002.gguf";
  auto write = [&](const auto& path, const std::string& name, int parts, int tensors) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    auto put = [&](auto value) { out.write(reinterpret_cast<const char*>(&value), sizeof value); };
    auto str = [&](const std::string& s) { put(uint64_t(s.size())); out << s; };
    out << "GGUF"; put(uint32_t(3)); put(uint64_t(1));
    put(uint64_t((parts >= 0) + (tensors >= 0)));
    if (parts >= 0) { str("split.count"); put(uint32_t(2)); put(uint16_t(parts)); }
    if (tensors >= 0) { str("split.tensors.count"); put(uint32_t(11)); put(int64_t(tensors)); }
    str(name); put(uint32_t(1)); put(uint64_t(8)); put(uint32_t(ni::gguf::kI8)); put(uint64_t(0));
    const size_t pos = size_t(out.tellp());
    out << std::string((32 - pos % 32) % 32, '\0') << "ABCDEFGH";
    require(bool(out), "write split fixture");
  };
  bool clean = true;
  auto reject = [&](const auto& path, const char* message, bool entry_name = true) {
    const int before = descriptors();
    bool rejected = false;
    try { ni::gguf::File file(path); }
    catch (const ni::gguf::Error& e) {
      rejected = std::string(e.what()).find(message) != std::string::npos &&
                 (!entry_name || std::string(e.what()).find(path.string()) != std::string::npos);
    }
    require(rejected, "split rejection keeps its diagnostic and entry filename");
    clean = (descriptors() == before) && clean;
  };
  for (int kind = 0; kind < 2; ++kind) {
    write(a, "x", kind == 0 ? 1 : 2, kind == 1 ? 3 : 2);
    write(b, "y", 2, 2);
    const int before = descriptors();
    for (int repeat = 0; repeat < 8; ++repeat)
      for (const auto& entry : {a, b})
        reject(entry, kind == 0 ? "differs from its file name" : "header announces 3");
    std::cout << "split " << (kind == 0 ? "part" : "tensor")
              << " count: 16 expected rejections, descriptor delta " << descriptors() - before << '\n';
  }
  // Existing early failures must still release the other part too.
  write(a, "x", 2, 2);
  std::ofstream(b, std::ios::binary | std::ios::trunc) << "broken";
  reject(a, "not a GGUF", false);
  write(b, "x", 2, 2);
  reject(a, "lists tensor x twice");
  std::filesystem::remove(b);
  reject(a, "Can't open");
  // A corrected file opens immediately after rejection, from either part, with exact tensor reads.
  for (const bool metadata : {false, true}) {
    write(a, "x", metadata ? 2 : -1, metadata ? 2 : -1);
    write(b, "y", metadata ? 2 : -1, metadata ? 2 : -1);
    for (const auto& entry : {a, b}) {
      const int before = descriptors();
      {
        ni::gguf::File file(entry);
        require(file.parts().size() == 2 && file.tensors().size() == 2, "valid split parts and tensors");
        for (const auto& name : {"x", "y"}) {
          const auto* t = file.find(name);
          require(t && t->part == (name[0] == 'y'), "tensor's owning part");
          char output[10]; memset(output, '?', sizeof output);
          file.read(*t, 0, output + 1, 8);
          require(std::string(output, sizeof output) == "?ABCDEFGH?", "complete tensor bytes and guards");
        }
      }
      clean = (descriptors() == before) && clean;
    }
  }
  const auto single = dir / "single.gguf";
  write(single, "z", 7, 99);  // Split metadata has never constrained an unsplit filename.
  const int before = descriptors();
  { ni::gguf::File file(single); require(file.find("z") != nullptr, "unsplit file remains readable"); }
  clean = (descriptors() == before) && clean;
  require(clean, "GGUF constructor must close every descriptor after rejection or destruction");
  std::cout << "PASS split metadata rejection, early errors, corrected reads and descriptor cleanup\n";
}

}  // namespace gguf_split_cleanup_test
