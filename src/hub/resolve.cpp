// What --model names (catalog names with an optional variant, Hugging Face repos, folders) and where models live.
#include "hub/hub.h"

#include <pwd.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <strings.h>

namespace ni::hub {

static std::string lower(std::string s) {
  for (auto& ch : s) ch = char(std::tolower(static_cast<unsigned char>(ch)));
  return s;
}

const Variant* CatalogModel::variant(const std::string& v) const {
  const std::string want = v.empty() ? default_variant : lower(v);
  for (const auto& x : variants)
    if (x.name == want) return &x;
  return nullptr;
}

const CatalogModel* find_model(const std::string& name) {
  const std::string n = lower(name);
  for (const auto& m : catalog()) {
    if (n == m.name) return &m;
    for (const auto& a : m.aliases)
      if (n == a) return &m;
  }
  return nullptr;
}

const Variant* find_variant_by_repo(const std::string& repo, const std::string& revision, const CatalogModel** model) {
  const std::string r = lower(repo);
  for (const auto& m : catalog())
    for (const auto& v : m.variants)
      if (lower(v.repo) == r && (revision.empty() || revision == v.revision)) {
        if (model) *model = &m;
        return &v;
      }
  return nullptr;
}

std::string short_name(const CatalogModel& m) {
  std::string s = m.name;
  for (const auto& a : m.aliases)
    if (a.size() < s.size()) s = a;
  return s;
}

std::string full_name(const CatalogModel& m, const Variant& v) { return m.name + ":" + v.name; }

std::string short_name(const CatalogModel& m, const Variant& v) {
  return v.name == m.default_variant ? short_name(m) : short_name(m) + ":" + v.name;
}

static std::string user_home() {
  if (const char* h = getenv("HOME"); h && *h) return h;
  if (const passwd* pw = getpwuid(getuid()); pw && pw->pw_dir) return pw->pw_dir;
  return "/tmp";
}

static std::string expand(const std::string& p) {
  if (p == "~") return user_home();
  if (p.rfind("~/", 0) == 0) return user_home() + p.substr(1);
  return p;
}

std::string home_dir() {
  if (const char* h = getenv("KAPITSA_HOME"); h && *h) {
    std::string d = expand(h);
    while (d.size() > 1 && d.back() == '/') d.pop_back();
    return d;
  }
  return user_home() + "/.cache/kapitsa";
}

std::string models_dir() { return home_dir() + "/models"; }

std::string tilde(const std::string& path) {
  const std::string h = user_home();
  if (path.size() >= h.size() && path.compare(0, h.size(), h) == 0 && (path.size() == h.size() || path[h.size()] == '/'))
    return "~" + path.substr(h.size());
  return path;
}

static bool is_dir(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// "org/name" or "org/name@revision" (also https://huggingface.co/org/name): false if `s` is not that shape.
static bool parse_repo(std::string s, std::string& repo, std::string& rev) {
  for (const char* p : {"https://huggingface.co/", "http://huggingface.co/", "huggingface.co/", "hf.co/"})
    if (s.rfind(p, 0) == 0) s = s.substr(strlen(p));
  const size_t at = s.find('@');
  std::string r = at == std::string::npos ? s : s.substr(0, at);
  rev = at == std::string::npos ? "" : s.substr(at + 1);
  while (!r.empty() && r.back() == '/') r.pop_back();
  const size_t slash = r.find('/');
  if (slash == std::string::npos || slash == 0 || slash + 1 >= r.size() || r.find('/', slash + 1) != std::string::npos)
    return false;
  auto ok = [](char ch) { return std::isalnum(static_cast<unsigned char>(ch)) || ch == '-' || ch == '_' || ch == '.'; };
  for (size_t i = 0; i < r.size(); ++i)
    if (i != slash && !ok(r[i])) return false;
  if (r[0] == '.' || r[slash + 1] == '.') return false;
  if (at != std::string::npos) {
    if (rev.empty()) return false;
    for (char ch : rev)
      if (!ok(ch) && ch != '/') return false;
  }
  repo = r;
  return true;
}

static std::string choices() {
  std::string s;
  for (const auto& m : catalog()) {
    if (!s.empty()) s += ", ";
    s += m.name + " (" + short_name(m) + ")";
  }
  return s;
}

static std::string variants_of(const CatalogModel& m) {
  std::string s;
  for (const auto& v : m.variants) s += (s.empty() ? "" : ", ") + v.name + (v.name == m.default_variant ? " (the default)" : "");
  return s;
}

static ModelRef catalog_ref(const CatalogModel& m, const Variant& v) {
  ModelRef r;
  r.kind = ModelRef::kCatalog;
  r.model = &m;
  r.variant = &v;
  r.repo = v.repo;
  r.revision = v.revision;
  r.name = full_name(m, v);
  r.id = v.name == m.default_variant ? m.name : full_name(m, v);
  r.dir = models_dir() + "/" + m.name + "-" + v.name;
  return r;
}

ModelRef resolve(const std::string& arg0) {
  std::string arg = arg0;
  while (!arg.empty() && std::isspace(static_cast<unsigned char>(arg.back()))) arg.pop_back();
  while (!arg.empty() && std::isspace(static_cast<unsigned char>(arg.front()))) arg.erase(arg.begin());
  if (arg.empty()) throw Error("No model given: one of " + choices() + ", a Hugging Face repo (org/name) or a folder.");
  ModelRef r;
  auto local = [&](const std::string& path) {
    r.kind = ModelRef::kLocal;
    r.dir = path;
    while (r.dir.size() > 1 && r.dir.back() == '/') r.dir.pop_back();
    r.name = r.dir;
    r.id = r.dir.substr(r.dir.find_last_of('/') + 1);
    return r;
  };
  const bool path_like = arg[0] == '/' || arg[0] == '~' || arg.rfind("./", 0) == 0 || arg.rfind("../", 0) == 0 ||
                         arg == "." || arg == "..";
  if (path_like) {
    const std::string p = expand(arg);
    if (!is_dir(p) && p.size() > 5 && strcasecmp(p.c_str() + p.size() - 5, ".gguf") == 0 && access(p.c_str(), R_OK) == 0)
      throw Error(arg + " is a GGUF file: `kapitsa import " + arg + "` converts it into a model folder (kapitsa serve "
                  "--model " + arg + " converts it on first use).");
    if (!is_dir(p)) throw Error("No folder at " + arg + ". A model is a folder, a name (" + choices() +
                                ") or a Hugging Face repo (org/name).");
    return local(p);
  }
  // <name>[:<variant>]
  const size_t colon = arg.rfind(':');
  const std::string base = colon == std::string::npos ? arg : arg.substr(0, colon);
  if (const CatalogModel* m = find_model(base)) {
    const std::string want = colon == std::string::npos ? "" : arg.substr(colon + 1);
    const Variant* v = m->variant(want);
    if (!v) throw Error(m->name + " has no variant \"" + want + "\". Its variants: " + variants_of(*m) + ".");
    ModelRef c = catalog_ref(*m, *v);
    c.variant_given = !want.empty();
    return c;
  }
  if (arg.find('/') != std::string::npos && is_dir(arg)) return local(arg);
  std::string repo, rev;
  if (parse_repo(arg, repo, rev)) {
    const CatalogModel* m = nullptr;
    if (const Variant* v = find_variant_by_repo(repo, rev, &m)) {  // with its companions
      ModelRef c = catalog_ref(*m, *v);
      c.variant_given = true;
      return c;
    }
    r.kind = ModelRef::kRepo;
    r.repo = repo;
    r.revision = rev;
    r.name = rev.empty() ? repo : repo + "@" + rev;
    r.id = repo;
    std::string folder = repo.substr(0, repo.find('/')) + "--" + repo.substr(repo.find('/') + 1);
    if (!rev.empty()) {
      std::string rv = rev;
      std::replace(rv.begin(), rv.end(), '/', '-');
      folder += "@" + rv;
    }
    r.dir = models_dir() + "/" + folder;
    return r;
  }
  // a folder of the models dir by its name (kapitsa import's conversions)
  if (arg.find('/') == std::string::npos && arg[0] != '.' && is_dir(models_dir() + "/" + arg)) return local(models_dir() + "/" + arg);
  if (is_dir(arg)) return local(arg);
  throw Error("Unknown model \"" + arg + "\". Choose " + choices() +
              ", a Hugging Face repo of a Qwen3.5-family MLX model (org/name), or a model folder.");
}

}  // namespace ni::hub
