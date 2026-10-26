#include "util/json_file.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <list>
#include <mutex>

namespace ni {

namespace {

struct Entry {
  std::string path;
  dev_t dev;
  ino_t ino;
  off_t size;
  timespec mtime;
  std::shared_ptr<const nlohmann::json> value;
};

// a model's config.json and its drafter's are what a plan reads; a few more cover the catalog's variants in turn
constexpr size_t kKeep = 8;

}  // namespace

std::shared_ptr<const nlohmann::json> json_file(const std::string& path) {
  static const auto none = std::make_shared<const nlohmann::json>();
  static std::mutex mu;
  static std::list<Entry> cache;  // most recently used first

  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return none;
  struct stat st;
  if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
    close(fd);
    return none;
  }
  auto same = [&](const Entry& e) {
    return e.path == path && e.dev == st.st_dev && e.ino == st.st_ino && e.size == st.st_size &&
           e.mtime.tv_sec == st.st_mtimespec.tv_sec && e.mtime.tv_nsec == st.st_mtimespec.tv_nsec;
  };
  {
    std::lock_guard<std::mutex> lk(mu);
    for (auto it = cache.begin(); it != cache.end(); ++it)
      if (same(*it)) {
        cache.splice(cache.begin(), cache, it);
        close(fd);
        return cache.front().value;
      }
  }
  std::string text(size_t(st.st_size), '\0');
  size_t got = 0;
  while (got < text.size()) {
    const ssize_t n = read(fd, text.data() + got, text.size() - got);
    if (n <= 0) break;
    got += size_t(n);
  }
  close(fd);
  if (got != text.size()) return none;
  nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
  if (j.is_discarded()) return none;
  auto value = std::make_shared<const nlohmann::json>(std::move(j));
  std::lock_guard<std::mutex> lk(mu);
  cache.remove_if([&](const Entry& e) { return e.path == path; });
  cache.push_front(Entry{path, st.st_dev, st.st_ino, st.st_size, st.st_mtimespec, value});
  if (cache.size() > kKeep) cache.pop_back();
  return value;
}

}  // namespace ni
