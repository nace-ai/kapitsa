// CPU by default. --gpu checks real Metal failure/retry and allocation ownership,
// under the GPU lock; never run that mode automatically in CTest. --mtp-gpu checks sidecar tensor admission
// through the real loader using tiny Metal buffers, without dispatching a kernel.
#include "util/load_threads.h"
#include "loader/safetensors.h"
#include "loader/load_stop.h"
#include "gguf/gguf.h"
#include "gguf_split_cleanup_test.h"
#include "model/expert_store.h"
#include "model/qwen35.h"
#include "bench/bench.h"
#include <nlohmann/json.hpp>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <fcntl.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <memory>
#include <limits>

#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
using namespace ni;
template<class F> void fails(F f, const std::string& text) {
  bool caught = false;
  try { f(); }
  catch (const FatalError& e) { caught = std::string(e.what()).find(text) != std::string::npos; }
  CHECK(caught);
}
static int descriptors() {
  int n = 0;
  for (int fd = 0; fd < 1024; ++fd) n += fcntl(fd, F_GETFD) != -1;
  return n;
}
static void child_fails(const char* self, const char* mode, const char* path = "") {
  int pipefd[2]; CHECK(pipe(pipefd) == 0);
  const pid_t pid = fork(); CHECK(pid >= 0);
  if (!pid) {
    close(pipefd[0]); dup2(pipefd[1], STDERR_FILENO); close(pipefd[1]);
    unsetenv("NI_ABORT"); execl(self, self, mode, path, nullptr); _exit(127);
  }
  close(pipefd[1]);
  std::string text; char buf[512]; ssize_t n;
  while ((n = read(pipefd[0], buf, sizeof buf)) > 0) text.append(buf, size_t(n));
  close(pipefd[0]); int status = 0; CHECK(waitpid(pid, &status, 0) == pid);
  CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 1);
  CHECK(text.find("[kapitsa] fatal: ") != std::string::npos);
}
// Pages of `path` in the page cache
static size_t cached_pages(const std::string& path) {
  const int fd = open(path.c_str(), O_RDONLY); CHECK(fd >= 0);
  struct stat st {}; CHECK(fstat(fd, &st) == 0);
  void* m = mmap(nullptr, size_t(st.st_size), PROT_READ, MAP_SHARED, fd, 0); CHECK(m != MAP_FAILED);
  const size_t page = size_t(getpagesize()), n = (size_t(st.st_size) + page - 1) / page;
  std::vector<char> v(n); CHECK(mincore(m, size_t(st.st_size), v.data()) == 0);
  munmap(m, size_t(st.st_size)); close(fd);
  size_t r = 0; for (char c : v) r += c & 1;
  return r;
}
// The loader's reads leave the file out of the page cache (#939: F_NOCACHE does nothing for a read that is not
// page-aligned, and safetensors tensors almost never are) and read every byte back exact, in each case:
// a tiny tensor, one of several 32 MB pieces, a page-aligned one read into a destination that is not 16 KB aligned,
// and a last tensor whose final page runs past the end of the file
static void uncached_reads(const std::filesystem::path& dir) {
  constexpr size_t kPage = size_t(16) << 10;
  const auto path = (dir / "c.safetensors").string();
  const std::vector<std::string> names = {"tiny", "big", "pad", "aligned", "gap", "tail"};
  std::vector<size_t> lens = {3, (size_t(64) << 20) + 777, kPage, 2 * kPage, 7, (size_t(5) << 20) + 1};
  std::string hdr;
  std::vector<size_t> at;  // each tensor's file offset
  for (;;) {  // grow "pad" until "aligned" starts on a page; the header's own length moves with it
    hdr = "{";
    at.clear();
    size_t off = 0;
    for (size_t i = 0; i < names.size(); ++i) {
      hdr += (i ? ",\"" : "\"") + names[i] + "\":{\"dtype\":\"U8\",\"shape\":[" + std::to_string(lens[i]) +
             "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + lens[i]) + "]}";
      at.push_back(off);
      off += lens[i];
    }
    hdr += "}";
    hdr.resize(hdr.size() + (13 - hdr.size() % 8) % 8, ' ');  // 8 + header is 5 mod 8: "tiny" and "big" misaligned
    for (auto& a : at) a += 8 + hdr.size();
    if (at[3] % kPage == 0) break;
    lens[2] += kPage - at[3] % kPage;
  }
  CHECK(at[0] % kPage && at[1] % kPage && at[5] % kPage);
  std::vector<uint8_t> file(at[5] + lens[5]);
  CHECK(file.size() % kPage);  // "tail"'s last page runs past the end of the file
  const uint64_t hlen = hdr.size(); memcpy(file.data(), &hlen, 8); memcpy(file.data() + 8, hdr.data(), hdr.size());
  for (size_t i = 8 + hdr.size(); i < file.size(); ++i) file[i] = uint8_t(i * 2654435761u >> 13);
  {  // written past the cache, so whatever is cached afterwards came from the reads
    const int fd = open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644); CHECK(fd >= 0);
    fcntl(fd, F_NOCACHE, 1);
    CHECK(write(fd, file.data(), file.size()) == ssize_t(file.size())); fsync(fd); close(fd);
  }
  const size_t before = cached_pages(path);
  WeightStore ws;
  ws.open_dir(dir.string(), {}, [](const std::string& f) { return f == "c.safetensors"; },
              [](const std::string&) { return true; });
  uint8_t* buf = nullptr;
  CHECK(posix_memalign(reinterpret_cast<void**>(&buf), kPage, lens[1] + 2 * kPage) == 0);
  std::unique_ptr<uint8_t, decltype(&free)> own(buf, free);
  // destinations 16 KB aligned, 256 B aligned (a tensor's place in a GPU buffer) and 1 B off
  for (const size_t shift : {size_t(0), size_t(256), size_t(1)})
    for (size_t i = 0; i < names.size(); ++i) {
      memset(buf, 0xAB, lens[i] + shift + 1);
      ws.read_deferred(names[i], buf + shift);
      CHECK(!memcmp(buf + shift, file.data() + at[i], lens[i]));
      CHECK((!shift || buf[shift - 1] == 0xAB) && buf[shift + lens[i]] == 0xAB);  // nothing written around it
    }
  const size_t after = cached_pages(path), pages = file.size() / size_t(getpagesize());
  std::cout << "page cache: " << before << " -> " << after << " of " << pages << " pages\n";
  CHECK(after <= before + pages / 10);
}
// The destination for a deferred read is sized from shape/dtype, never from data_offsets.
static void tensor_entries(const std::filesystem::path& parent) {
  const auto dir = parent / "tensor-entries";
  std::filesystem::create_directory(dir);
  const auto path = dir / "model.safetensors";
  using J = nlohmann::json;
  auto write = [&](const J& entry, const std::string& data = "") {
    const std::string hdr = J{{"weight", entry}}.dump();
    const uint64_t n = hdr.size();
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&n), 8); f << hdr << data;
  };
  const auto deferred = [](const std::string&) { return true; };
  FatalRecovery recover;
  auto reject = [&](const J& entry, const std::string& why) {
    write(entry, "ABCDEFGH");
    fails([&] { SafetensorsIndex index(dir); }, why);
    fails([&] { WeightStore ws; ws.open_dir(dir); }, why);
    fails([&] { WeightStore ws; ws.open_dir(dir, {}, {}, deferred); }, why);
    // The recoverable diagnostic must identify the bad file too.
    fails([&] { SafetensorsIndex index(dir); }, path.string());
  };
  J entry = {{"dtype", "BF16"}, {"shape", {1}}, {"data_offsets", {0, 8}}};
  reject(entry, "shape/dtype");  // an eight-byte read would overwrite a two-byte destination
  entry["shape"] = {8};
  reject(entry, "shape/dtype");  // the other direction leaves part of the destination unread
  for (const J& dim : {J(-1), J(1.5), J("1"), J(nullptr), J(true), J(uint64_t(1) << 63)}) {
    entry["shape"] = J::array({dim});
    reject(entry, "bad tensor shape");
  }
  entry["shape"] = {uint64_t(1) << 62, 4};
  reject(entry, "tensor shape overflows");
  entry["shape"] = {uint64_t(1) << 62};
  entry["dtype"] = "F32";
  reject(entry, "shape/dtype");  // element count fits, byte count does not
  entry["shape"] = {0};
  entry["data_offsets"] = {UINT64_MAX, UINT64_MAX};
  reject(entry, "tensor offset overflows");

  // Header-only files still serve the memory planner, but cannot be admitted for deferred reads.
  entry = {{"dtype", "BF16"}, {"shape", {4}}, {"data_offsets", {0, 8}}};
  write(entry);
  { SafetensorsIndex index(dir); CHECK(index.get("weight").numel() == 4 && index.get("weight").len == 8); }
  fails([&] { WeightStore ws; ws.open_dir(dir); }, "past its end");
  fails([&] { WeightStore ws; ws.open_dir(dir, {}, {}, deferred); }, "past its end");
  { WeightStore ws; ws.open_dir(dir, [](const std::string&) { return false; }); }

  // Scalar and empty tensors, all supported element widths, an unaligned range and a read after rejection.
  for (const auto& [dtype, width] : std::vector<std::pair<std::string, size_t>>{
           {"U8", 1}, {"I8", 1}, {"F8_E4M3", 1}, {"U16", 2}, {"BF16", 2}, {"F16", 2},
           {"U32", 4}, {"I32", 4}, {"F32", 4}, {"I64", 8}}) {
    for (const J& shape : {J::array(), J::array({2, 3}), J::array({2, 0, 3})}) {
      const size_t count = shape.empty() ? 1 : (shape[1] == 0 ? 0 : 6), bytes = count * width;
      write({{"dtype", dtype}, {"shape", shape}, {"data_offsets", {1, 1 + bytes}}},
            "!" + std::string(bytes, 'x'));
      SafetensorsIndex index(dir);
      CHECK(size_t(index.get("weight").numel()) == count && index.get("weight").len == bytes);
      WeightStore ws; ws.open_dir(dir, {}, {}, deferred);
      CHECK(size_t(ws.get("weight").numel()) == count);
      std::vector<char> out(bytes + 2, '?');
      ws.read_deferred("weight", out.data() + 1);
      CHECK(out.front() == '?' && out.back() == '?');
      CHECK(std::string(out.data() + 1, bytes) == std::string(bytes, 'x'));
      std::fill(out.begin(), out.end(), '?');
      index.read(index.get("weight"), 0, out.data() + 1, bytes);
      CHECK(out.front() == '?' && out.back() == '?');
      CHECK(std::string(out.data() + 1, bytes) == std::string(bytes, 'x'));
    }
  }
  std::filesystem::remove_all(dir);
  std::cout << "PASS tensor shape/range validation, header-only plans, deferred read guards\n";
}
// Unknown has width zero: admitting its nonempty range lets a deferred read overrun a shape-sized destination.
static void unsupported_dtypes(const std::filesystem::path& parent) {
  const auto dir = parent / "unsupported-dtypes";
  std::filesystem::create_directory(dir);
  const auto path = dir / "model.safetensors";
  using J = nlohmann::json;
  const auto skip = [](const std::string& n) { return n == "unused.weight"; };
  const auto keep = [&](const std::string& n) { return !skip(n); };
  const auto defer = [](const std::string&) { return true; };
  FatalRecovery recover;
  for (const std::string dtype : {"BOOL", "I16", "U64", "F64", "F8_E5M2", "BF61"}) {
    for (const size_t bytes : {size_t(0), size_t(8)}) {
      const J bad = {{"dtype", dtype}, {"shape", {bytes ? 1 : 0}}, {"data_offsets", {1, 1 + bytes}}};
      const J good = {{"dtype", "U8"}, {"shape", {1}}, {"data_offsets", {0, 1}}};
      const std::string hdr = J{{"unused.weight", bad}, {"kept.weight", good}}.dump();
      const uint64_t len = hdr.size();
      auto write = [&](bool payload) {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(&len), 8); f << hdr;
        if (payload) f << "K" << std::string(bytes, 'x');
      };
      write(true);
      const std::string why = path.string() + ": tensor unused.weight has unsupported dtype " + dtype;
      fails([&] { SafetensorsIndex idx(dir); }, why);
      fails([&] { WeightStore ws; ws.open_dir(dir); }, why);
      fails([&] { WeightStore ws; ws.open_dir(dir, {}, {}, defer); }, why);
      { WeightStore ws; ws.open_dir(dir, [](const std::string&) { return false; }); CHECK(ws.all().empty()); }
      {
        SafetensorsIndex idx(dir, skip);
        CHECK(idx.all().size() == 1 && idx.get("kept.weight").dtype == DType::U8);
        char out[3] = {'?', '?', '?'};
        idx.read(idx.get("kept.weight"), 0, out + 1, 1);
        CHECK(std::string(out, 3) == "?K?");
        WeightStore ws; ws.open_dir(dir, keep, {}, defer);
        CHECK(ws.all().size() == 1 && ws.get("kept.weight").dtype == DType::U8);
        out[1] = '?'; ws.read_deferred("kept.weight", out + 1);
        CHECK(std::string(out, 3) == "?K?");
      }
      write(false);  // a selected known dtype remains usable in a header-only plan
      SafetensorsIndex idx(dir, skip);
      CHECK(idx.get("kept.weight").numel() == 1 && idx.get("kept.weight").len == 1);
    }
  }
  std::filesystem::remove_all(dir);
  std::cout << "PASS unsupported dtypes rejected, tensor filters and header-only plans preserved\n";
}
// Tensor ranges are checked before import or reads, including sums that wrap uint64_t.
static void gguf_ranges(const std::filesystem::path& parent) {
  const auto dir = parent / "gguf-ranges";
  std::filesystem::create_directory(dir);
  const auto path = dir / "single.gguf";
  auto write = [&](const std::filesystem::path& dst, uint64_t rel, const std::string& data,
                   const std::string& name = "x") {
    std::ofstream f(dst, std::ios::binary);
    auto put = [&](auto x) { f.write(reinterpret_cast<const char*>(&x), sizeof x); };
    f.write("GGUF", 4); put(uint32_t(3)); put(uint64_t(1)); put(uint64_t(0));
    put(uint64_t(name.size())); f << name; put(uint32_t(1)); put(uint64_t(1));
    put(uint32_t(gguf::kF64)); put(rel);
    const size_t at = size_t(f.tellp());
    CHECK(at <= 64); f << std::string(64 - at, '\0') << data;
    CHECK(f.good());
  };
  auto rejected = [&](auto operation, const std::string& text) {
    bool caught = false;
    try { operation(); }
    catch (const gguf::Error& e) { caught = std::string(e.what()).find(text) != std::string::npos; }
    CHECK(caught);
  };
  const int before = descriptors();
  for (const uint64_t rel : {uint64_t(0), uint64_t(8), UINT64_MAX,
                             UINT64_MAX - 7 - 64}) {
    write(path, rel, rel == 8 ? "ABCDEFGH" : "");
    for (int i = 0; i < 8; ++i) {
      rejected([&] { gguf::File f(path); }, "tensor x lies beyond the end");
      rejected([&] { gguf::File f(path); }, path.string());
    }
  }
  CHECK(descriptors() == before);
  write(path, 0, "ABCDEFGH");
  {
    gguf::File f(path);
    const auto* t = f.find("x"); CHECK(t && t->offset == 64 && t->bytes == 8);
    char out[10]; memset(out, '?', sizeof out);
    f.read(*t, 0, out + 1, 8); CHECK(std::string(out, sizeof out) == "?ABCDEFGH?");
    memset(out, '?', sizeof out);
    f.read(*t, 3, out + 1, 3); CHECK(std::string(out, 5) == "?DEF?");
    f.read(*t, 8, nullptr, 0);  // an empty read at the exact end remains valid
    for (const auto& [off, n] : std::vector<std::pair<uint64_t, size_t>>{
             {8, 1}, {9, 0}, {UINT64_MAX, 1}, {UINT64_MAX - 7, 8}, {0, SIZE_MAX}}) {
      memset(out, '?', sizeof out);
      rejected([&] { f.read(*t, off, out + 1, n); }, "internal: read past the end of tensor x");
      CHECK(std::string(out, sizeof out) == std::string(sizeof out, '?'));
    }
    f.read(*t, 0, out + 1, 8); CHECK(std::string(out, sizeof out) == "?ABCDEFGH?");
  }
  CHECK(descriptors() == before);
  const auto a = dir / "split-00001-of-00002.gguf", b = dir / "split-00002-of-00002.gguf";
  write(a, 0, "ABCDEFGH"); write(b, UINT64_MAX - 7 - 64, "", "y");
  rejected([&] { gguf::File f(a); }, "tensor y lies beyond the end");
  CHECK(descriptors() == before);
  write(b, 0, "abcdefgh", "y");
  {
    gguf::File f(a); const auto* t = f.find("y"); CHECK(t && t->part == 1);
    char out[10]; memset(out, '?', sizeof out);
    f.read(*t, 0, out + 1, 8); CHECK(std::string(out, sizeof out) == "?abcdefgh?");
  }
  CHECK(descriptors() == before);
  std::filesystem::remove_all(dir);
  std::cout << "PASS GGUF file/read bounds, wrapped offsets, split-part FD cleanup and valid reads\n";
}

// Relative reads must stay in their tensor even when off + n wraps size_t.
static void safetensors_read_bounds() {
  char pattern[] = "/tmp/kapitsa-read-bounds-XXXXXX";
  CHECK(mkdtemp(pattern));
  const std::filesystem::path dir(pattern), path = dir / "model.safetensors";
  struct Cleanup {
    std::filesystem::path dir;
    ~Cleanup() { std::error_code ec; std::filesystem::remove_all(dir, ec); }
  } cleanup{dir};
  using J = nlohmann::json;
  const std::string payload = "ABCDEFGH";
  const std::string hdr = J{{"x", {{"dtype", "U8"}, {"shape", {8}}, {"data_offsets", {0, 8}}}},
                            {"empty", {{"dtype", "U8"}, {"shape", {0}}, {"data_offsets", {8, 8}}}}}.dump();
  auto write = [&] {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    const uint64_t n = hdr.size();
    f.write(reinterpret_cast<const char*>(&n), 8); f << hdr << payload;
    CHECK(bool(f));
  };
  write();
  const int before = descriptors();
  FatalRecovery recover;
  {
    SafetensorsIndex index(dir);
    const auto& t = index.get("x");
    const auto& empty = index.get("empty");
    CHECK(t.len == payload.size() && empty.len == 0);
    size_t valid = 0;
    for (size_t off = 0; off <= t.len; ++off)
      for (size_t n = 0; n <= t.len - off; ++n) {
        std::vector<char> out(n + 2, '?');
        index.read(t, off, out.data() + 1, n);
        CHECK(out.front() == '?' && out.back() == '?');
        CHECK(std::string(out.data() + 1, n) == payload.substr(off, n));
        ++valid;
      }
    index.read(t, t.len, nullptr, 0);
    index.read(empty, 0, nullptr, 0);
    std::cout << "PASS " << valid << " guarded full/partial/end reads and empty zero reads\n";
    // Enough space for the whole fixture even if a regression admits an invalid range.
    const size_t guard_size = 8 + hdr.size() + payload.size() + 2;
    auto reject = [&](const SafetensorsIndex::Entry& entry, size_t off, size_t n) {
      std::vector<char> out(guard_size, '?');
      bool caught = false;
      try { index.read(entry, off, out.data() + 1, n); }
      catch (const FatalError& e) { caught = std::string(e.what()).find("read past a tensor's end") != std::string::npos; }
      if (!caught)
        std::cout << "Unexpected read acceptance: off=" << off << " n=" << n
                  << " first byte=" << unsigned(static_cast<unsigned char>(out[1]))
                  << " outer guards=" << (out.front() == '?' && out.back() == '?') << '\n';
      CHECK(caught);
      CHECK(std::string(out.data(), out.size()) == std::string(out.size(), '?'));
    };
    const size_t max = std::numeric_limits<size_t>::max();
    for (const auto& [off, n] : std::vector<std::pair<size_t, size_t>>{
           {max, 1}, {max - 3, 8}, {1, max}, {max, max}, {t.len, 1}, {t.len + 1, 0},
           {0, t.len + 1}, {t.len - 1, 2}, {0, max}})
      reject(t, off, n);
    reject(empty, 0, 1); reject(empty, 1, 0); reject(empty, max, 1);
    std::cout << "PASS 12 invalid ranges rejected with unchanged destinations\n";
    // A physical short read still uses the existing recoverable error and may fill its valid prefix.
    std::filesystem::resize_file(path, t.off + 3);
    char short_out[10]; memset(short_out, '?', sizeof short_out);
    fails([&] { index.read(t, 0, short_out + 1, t.len); }, "reading");
    CHECK(std::string(short_out, sizeof short_out) == "?ABC??????");
    memset(short_out, '?', sizeof short_out);
    index.read(t, 0, short_out + 1, 3);
    CHECK(std::string(short_out, sizeof short_out) == "?ABC??????");
    write();
    index.read(t, 0, short_out + 1, t.len);
    CHECK(std::string(short_out, sizeof short_out) == "?ABCDEFGH?");
    std::cout << "PASS short-read error, retained prefix and retry on the same index\n";
    std::atomic<size_t> completed{0};
    LoadThreads readers;
    for (size_t worker = 0; worker < 4; ++worker)
      readers.emplace_back([&, worker] {
        for (size_t i = 0; i < 64; ++i) {
          const size_t off = (i + worker) % t.len;
          char out[3] = {'?', '?', '?'};
          index.read(t, off, out + 1, 1);
          CHECK(out[0] == '?' && out[1] == payload[off] && out[2] == '?');
          memset(out, '?', sizeof out);
          fails([&] { index.read(t, max, out + 1, 1); }, "read past a tensor's end");
          CHECK(std::string(out, sizeof out) == "???");
          ++completed;
        }
      });
    readers.join();
    CHECK(completed == 256);
    std::cout << "PASS 256 concurrent valid/rejected read pairs\n";
  }
  CHECK(descriptors() == before);
  std::cout << "PASS Safetensors relative-read bounds and descriptor cleanup\n";
}

static void cpu(const char* self) {
  CHECK(!FatalRecovery::enabled());
  {
    FatalRecovery recover;
    const std::string long_message(4096, 'x');
    fails([&] { fatal("%s %d", long_message.c_str(), 42); }, long_message + " 42");
    std::atomic<int> finished{0};
    fails([&] {
      LoadThreads jobs;
      jobs.emplace_back([] { fatal("worker read failed"); });
      jobs.emplace_back([&] { std::this_thread::sleep_for(std::chrono::milliseconds(10)); ++finished; });
      jobs.join();
    }, "worker read failed");
    CHECK(finished == 1);
    try {
      LoadThreads jobs;
      jobs.emplace_back([&] { ++finished; });
      throw std::bad_alloc();  // unwind while an already-started job still owns its references
    } catch (const std::bad_alloc&) {}
    CHECK(finished == 2);
  }
  CHECK(!FatalRecovery::enabled());
  child_fails(self, "--fatal");
  child_fails(self, "--unscoped-thread");
  char pattern[] = "/tmp/kapitsa-load-errors-XXXXXX";
  CHECK(mkdtemp(pattern)); const std::filesystem::path dir(pattern);
  // One header read successfully, then one broken shard: both descriptors must close.
  { std::ofstream f(dir / "a.safetensors", std::ios::binary); uint64_t n = 2; f.write(reinterpret_cast<char*>(&n), 8); f << "{}"; }
  std::ofstream(dir / "b.safetensors", std::ios::binary) << "broken";
  const int before = descriptors();
  {
    FatalRecovery recover;
    for (int i = 0; i < 32; ++i) {
      fails([&] { SafetensorsIndex index(dir); }, "incomplete");
      fails([&] { WeightStore weights; weights.open_dir(dir); }, "incomplete");
      const auto lay = q20::slot_layout(128, 128);
      q20::LayerSrc layer;
      for (int j = 0; j < 6; ++j) layer.p[j] = {0, 0, 0, j % 2 ? lay.ud - lay.gd : lay.uc};
      q20::ExpertCache::Memory mem; mem.per_chunk = 1;
      fails([&] {
        q20::ExpertCache cache({(dir / "a.safetensors").string(), (dir / "missing").string()},
                              {layer}, 1, lay, 1, mem, {}, {});
      }, "cannot open");
    }
  }
  CHECK(descriptors() == before);
  child_fails(self, "--bad-file", pattern);
  safetensors_read_bounds();
  tensor_entries(dir);
  unsupported_dtypes(dir);
  gguf_ranges(dir);
  gguf_split_cleanup_test::run();
  uncached_reads(dir);
  std::filesystem::remove_all(dir);
  std::cout << "PASS scoped errors, thread joining/isolation, loader FD cleanup, CLI exit 1\n";
}
// NoLoad avoids unrelated trunk weights; load_mtp still opens the actual sidecar and reads its linears.
struct MtpLoadProbe : Qwen35 {
  MtpLoadProbe() : Qwen35("", 1, NoLoad{}) {}
  void load(const std::string& dir) { load_mtp(dir); }
  QLinear linear(const WeightStore& w) { return own_qlin(w, "mtp.fc"); }
};
static void mtp_gpu() {
  bench::GpuLock lock;
  CHECK(bench::GpuLock::held_by_caller() || lock.acquire());
  FatalRecovery recover;
  char pattern[] = "/tmp/kapitsa-mtp-shapes-XXXXXX";
  CHECK(mkdtemp(pattern));
  const std::filesystem::path dir(pattern);
  struct Cleanup {
    std::filesystem::path dir;
    ~Cleanup() { std::error_code ec; std::filesystem::remove_all(dir, ec); }
  } cleanup{dir};
  using J = nlohmann::json;
  auto write = [&](J w, J sc, J bi, const std::string& wd = "U32", const std::string& sd = "BF16",
                   const std::string& bd = "BF16") {
    J hdr;
    std::string data;
    auto add = [&](const std::string& name, const std::string& dtype, const J& shape) {
      size_t count = 1;
      for (const auto& d : shape) count *= d.get<size_t>();
      const size_t bytes = count * (dtype == "U32" || dtype == "F32" ? 4 : 2);
      hdr[name] = {{"dtype", dtype}, {"shape", shape}, {"data_offsets", {data.size(), data.size() + bytes}}};
      const size_t start = data.size();
      data += std::string(bytes, '\0');
      if (name == "mtp.fc.weight") std::fill(data.begin() + start, data.end(), char(0xA5));
      else if (dtype == "BF16" || dtype == "F16") {
        const uint16_t one = dtype == "BF16" ? 0x3f80 : 0x3c00;
        for (size_t i = start; i < data.size(); i += 2) memcpy(data.data() + i, &one, 2);
      } else if (dtype == "F32") {
        const float one = 1.0f;
        for (size_t i = start; i < data.size(); i += 4) memcpy(data.data() + i, &one, 4);
      }
    };
    for (const char* n : {"mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight"})
      add(n, "BF16", J::array({1}));
    add("mtp.fc.weight", wd, w); add("mtp.fc.scales", sd, sc); add("mtp.fc.biases", bd, bi);
    const std::string h = hdr.dump();
    const uint64_t len = h.size();
    std::ofstream f(dir / "mtp.safetensors", std::ios::binary);
    f.write(reinterpret_cast<const char*>(&len), 8); f << h << data;
  };
  auto reject = [&](const J& w, const J& sc, const J& bi, const std::string& wd = "U32",
                    const std::string& sd = "BF16", const std::string& bd = "BF16") {
    write(w, sc, bi, wd, sd, bd);
    Gpu::AllocationScope allocations;
    MtpLoadProbe probe;
    fails([&] { probe.load(dir); }, "mtp.fc: invalid affine tensor shapes or dtypes");
  };
  reject(J::array(), {1, 1}, {1, 1});
  reject({8}, {1, 1}, {1, 1});
  reject({1, 1, 1, 8}, {1, 1, 1, 1}, {1, 1, 1, 1});
  reject({2, 8}, J::array(), {2, 1});
  reject({2, 8}, {2, 1}, J::array());
  reject({2, 8}, {1, 1}, {2, 1});
  reject({2, 8}, {2, 1}, {2, 2});
  reject({2, 3, 8}, {1, 3, 1}, {1, 3, 1});
  reject({0, 8}, {0, 1}, {0, 1});
  reject({0, 8}, {0, uint64_t(1) << 58}, {0, uint64_t(1) << 58});
  reject({2, 8}, {2, 1}, {2, 1}, "F16");
  reject({2, 8}, {2, 1}, {2, 1}, "U32", "U16");
  reject({2, 8}, {2, 1}, {2, 1}, "U32", "BF16", "U16");
  size_t accepted = 0;
  for (int bits : {3, 4, 5, 6, 8})
    for (bool stacked : {false, true})
      for (const char* dtype : {"BF16", "F16", "F32"}) {
        J w = stacked ? J{2, 3, 2 * bits} : J{3, 2 * bits};
        J sc = stacked ? J{2, 3, 1} : J{3, 1};
        write(w, sc, sc, "U32", dtype, dtype);
        Gpu::AllocationScope allocations;
        WeightStore weights;
        weights.open_dir(dir);
        MtpLoadProbe probe;
        const QLinear q = probe.linear(weights);
        CHECK(q.N == 3 && q.K == 64 && q.gs == 64 && q.bits == bits);
        CHECK(q.w->shape == weights.get("mtp.fc.weight").shape && q.w->dtype == DType::U32);
        CHECK(q.s->shape == weights.get("mtp.fc.scales").shape && q.s->dtype == DType::BF16);
        CHECK(q.b->shape == q.s->shape && q.b->dtype == DType::BF16);
        for (int64_t i = 0; i < q.s->numel(); ++i)
          CHECK(q.s->buf.as<uint16_t>()[i] == 0x3f80 && q.b->buf.as<uint16_t>()[i] == 0x3f80);
        const auto& original = weights.get("mtp.fc.weight");
        CHECK(!memcmp(q.w->buf.ptr(), original.buf.ptr(), original.buf.size));
        ++accepted;
      }
  std::cout << "PASS MTP sidecar admission: 13 malformed cases rejected; " << accepted
            << " dense/stacked format and parameter-dtype combinations accepted (Metal buffers, no dispatch)\n";
}
static void gpu() {
  bench::GpuLock lock;
  CHECK(bench::GpuLock::held_by_caller() || lock.acquire());
  FatalRecovery recover;
  auto& g = Gpu::get();
  fails([&] { g.alloc(g.max_buffer_length() + 1); }, "maximum buffer length");
  fails([&] { g.compile_extra("load-error-test", "this is not Metal"); }, "Metal compile failed");
  g.compile_extra("load-error-test", R"(#include <metal_stdlib>
    kernel void load_error_write(device int* out [[buffer(0)]]) { out[0] = 17; })");
  fails([&] { g.pipeline("load_error_missing_kernel"); }, "not found");
  auto pipeline = g.pipeline("load_error_write");
  auto sentinel = g.alloc(16384);
  // The validation layer lazily allocates its own dispatch instrumentation.
  { auto e = g.begin(); e->use(pipeline); e->buf(0, sentinel); e->dispatch(1, 1, 1, 1); e->commit_and_wait(); }
  sentinel.as<int>()[0] = 123;
  const size_t before = g.allocated_bytes();
  bool freed = false, freed_early = false;
  {
    Gpu::AllocationScope outer;
    auto keep = g.alloc(16384);
    keep.as<int>()[0] = 456;
    fails([&] {
      Gpu::AllocationScope failed_load;
      auto early = g.alloc(16384); g.release(early);
      void* memory = nullptr; CHECK(posix_memalign(&memory, size_t(getpagesize()), 16384) == 0);
      auto owner = std::shared_ptr<void>(memory, [&](void* p) {
        freed = static_cast<int*>(p)[0] == 17; std::free(p);
      });
      auto b = g.wrap(memory, 16384, "load error backing", true, std::move(owner));
      struct ReleaseOnUnwind {
        Gpu& g; Buffer b; bool& freed; bool& early;
        ~ReleaseOnUnwind() { g.release(b); early = freed; }
      } release{g, b, freed, freed_early};
      auto e = g.begin(); e->use(pipeline); e->buf(0, b); e->dispatch(1, 1, 1, 1); e->commit();
      fatal("failure after allocation and GPU submission");
    }, "after allocation");
    CHECK(freed && !freed_early && keep.as<int>()[0] == 456);
  }
  CHECK(sentinel.as<int>()[0] == 123);
  const size_t after = g.allocated_bytes();
  std::cout << "Metal allocated bytes before/after rollback: " << before << '/' << after << '\n';
  CHECK(after == before);
  g.release(sentinel);
  std::cout << "PASS real Metal compile/pipeline retry and buffer rollback after GPU work\n";
}
// A stop asked during the load (src/loader/load_stop.h), in a child per case: argv = --load-stop DIR CASE.
// "ask": the action ends the child from a reader, once, before every piece is read; "ask-index" and "ask-deferred"
// the same through SafetensorsIndex::read and read_deferred(); "no-ask" (action set) and "no-action" (asked) read all.
[[noreturn]] static void load_stop_child(const std::string& dir, const std::string& c) {
  if (c != "no-action") set_load_stop_action([] { (void)!write(STDOUT_FILENO, "S", 1); _exit(42); });
  if (c != "no-ask") ask_load_stop();
  if (c == "ask-index") {
    SafetensorsIndex idx(dir);
    std::vector<uint8_t> b(4096);
    for (const auto& [name, e] : idx.all()) idx.read(e, 0, b.data(), e.len);
  } else if (c == "ask-deferred") {
    WeightStore ws;
    ws.open_dir(dir, {}, {}, [](const std::string&) { return true; });
    std::vector<uint8_t> b(4096);
    for (const auto& [name, t] : ws.all()) ws.read_deferred(name, b.data());
  } else {
    WeightStore ws;
    ws.open_dir(dir);
    CHECK(ws.all().size() == 64);
  }
  (void)!write(STDOUT_FILENO, "R", 1);
  _exit(0);
}
static void load_stop(const char* self) {
  const auto dir = std::filesystem::temp_directory_path() / ("load_stop_test." + std::to_string(getpid()));
  std::filesystem::create_directories(dir);
  for (const char* f : {"a.safetensors", "b.safetensors"}) {  // 2 files, 32 tensors of 4 KB each: 64 reads, 8 readers
    std::string hdr = "{";
    for (int i = 0; i < 32; ++i)
      hdr += (i ? ",\"" : "\"") + std::string(f, 1) + std::to_string(i) + "\":{\"dtype\":\"U8\",\"shape\":[4096],"
             "\"data_offsets\":[" + std::to_string(i * 4096) + "," + std::to_string((i + 1) * 4096) + "]}";
    hdr += "}";
    std::vector<uint8_t> file(8 + hdr.size() + 32 * 4096, 7);
    const uint64_t hlen = hdr.size(); memcpy(file.data(), &hlen, 8); memcpy(file.data() + 8, hdr.data(), hdr.size());
    std::ofstream(dir / f, std::ios::binary).write(reinterpret_cast<const char*>(file.data()), std::streamsize(file.size()));
  }
  for (const std::string c : {"ask", "ask-index", "ask-deferred", "no-ask", "no-action"}) {
    int out[2]; CHECK(pipe(out) == 0);
    const pid_t pid = fork(); CHECK(pid >= 0);
    if (!pid) {
      close(out[0]); dup2(out[1], STDOUT_FILENO); close(out[1]);
      execl(self, self, "--load-stop", dir.c_str(), c.c_str(), nullptr); _exit(127);
    }
    close(out[1]);
    std::string text; char buf[64]; ssize_t n;
    while ((n = read(out[0], buf, sizeof buf)) > 0) text.append(buf, size_t(n));
    close(out[0]); int status = 0; CHECK(waitpid(pid, &status, 0) == pid);
    const bool stops = c.rfind("ask", 0) == 0;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != (stops ? 42 : 0) || text != (stops ? "S" : "R"))
      throw std::runtime_error("load stop, " + c + ": status " + std::to_string(status) + ", output '" + text + "'");
  }
  std::filesystem::remove_all(dir);
  std::cout << "PASS a stop asked during the load ends it from one reader; none asked or no action reads it all\n";
}
int main(int argc, char** argv) {
  if (argc > 3 && std::string(argv[1]) == "--load-stop") load_stop_child(argv[2], argv[3]);
  if (argc > 1 && std::string(argv[1]) == "--fatal") fatal("outside SDK recovery");
  if (argc > 1 && std::string(argv[1]) == "--unscoped-thread") {
    FatalRecovery recover;
    std::thread t([] { fatal("unrelated thread remains fatal"); }); t.join();
  }
  if (argc > 2 && std::string(argv[1]) == "--bad-file") { WeightStore ws; ws.open_dir(argv[2]); }
  try {
    if (argc > 1 && std::string(argv[1]) == "--gguf-split-cleanup") gguf_split_cleanup_test::run();
    else if (argc > 1 && std::string(argv[1]) == "--safetensors-read-bounds") safetensors_read_bounds();
    else if (argc > 1 && std::string(argv[1]) == "--mtp-gpu") mtp_gpu();
    else if (argc > 1 && std::string(argv[1]) == "--gpu") gpu();
    else {
      cpu(argv[0]);
      load_stop(argv[0]);
    }
  } catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}
