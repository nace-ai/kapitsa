// The built-in models, where models live, and what --model names (see hub.h).
#include "hub/hub.h"

#include <pwd.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>

namespace ni::hub {

// Repos and revisions are the Nace Agent app's (packages/nace-runtime/src/catalog-data.ts), so the app and the CLI
// fetch the same bytes. Best first: best_fit() takes the first one that fits the Mac.
//
// Memory (GB, 1e9 bytes), measured on the M4 Pro with `kapitsa doctor --json --ctx 32768 --slots 2` (opt/air:
// weights + MTP head + 2048-row prefill buffers + two slots' recurrent state + the Neural Engine split of the dense
// models) plus the DFlash2 drafter's resident size at load (the app's phys_footprint deltas: 35B +0.58, 27B +1.54).
// The KV cache then grows with each conversation: 35B 22,528 bytes per token, 27B 69,632, 4B 36,864.
const std::vector<CatalogModel>& catalog() {
  static const std::vector<CatalogModel> c = {
      {"qwen3.6-35b-a3b",
       {"qwen3.6-35b", "35b", "35b-a3b"},
       "Qwen3.6-35B-A3B",
       "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",  // mlx-community/Qwen3.6-35B-A3B-4bit + its MTP head
       "95460516fbfe34c48f577f67c9ed191f5d61fbaf",
       {{"dflash2", "incoai/Qwen3.6-35B-A3B-DFlash2", "51ef7b6923ad6c14cb1bb41c37a9041446496ab6",
         {"config.json", "model.safetensors", "LICENSE"}, "dflash2", "", 0}},
       22.07,
       21.21 + 0.58,
       22528,
       "MoE, 3B active per token: fast replies and fast reading"},
      {"qwen3.8-27b",
       {"27b"},
       "Qwen3.8-27B",
       "lmstudio-community/Qwen3.8-27B-MLX-4bit",
       "6067b15cf581666a4aecf6af3afaba4bb5efc20c",
       {// the pack has no MTP head; MTPLX's 27B pack ships Qwen's (and mtplx_runtime.json: the head reads the
        // post-norm hidden state)
        {"mtp", "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed", "1d5087d2062c02b279180a53e4016cf9cd7a3d7e",
         {"mtp.safetensors", "mtplx_runtime.json"}, "", "", 0},
        {"dflash2", "incoai/Qwen3.8-27B-DFlash2", "015e795645c74b1a0eeef3b570031fb62e769bc5",
         {"config.json", "model.safetensors"}, "dflash2", "", 0}},
       20.17,
       18.08 + 1.54,
       69632,
       "dense: the strongest coder of the three, slower to write"},
      {"qwen3.5-4b",
       {"4b"},
       "Qwen3.5-4B",
       "lmstudio-community/Qwen3.5-4B-MLX-4bit",
       "c43ee1d65576a5d98de1e8405cac93c371a655c1",
       {// the MLX pack drops the MTP head: built from the official checkpoint's 15 mtp.* tensors (241 MB by range
        // requests, not the 9.3 GB checkpoint), byte for byte what tools/mtp_sidecar.py and the app make
        {"mtp-sidecar", "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", {}, "",
         "4f491dd27975520ec9c9bfa970ca215cc7f12a0328761d3656ab7b9f388e472a", 241200816}},
       3.30,
       3.66,
       36864,
       "small and quick: for 16-24 GB Macs and simple tasks"},
  };
  return c;
}

static std::string lower(std::string s) {
  for (auto& ch : s) ch = char(std::tolower(static_cast<unsigned char>(ch)));
  return s;
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

const CatalogModel* find_model_by_repo(const std::string& repo) {
  const std::string r = lower(repo);
  for (const auto& m : catalog())
    if (lower(m.repo) == r) return &m;
  return nullptr;
}

double comfortable_gb(const CatalogModel& m) { return m.fixed_gb + 16384.0 * m.kv_bytes_per_token / 1e9; }

const CatalogModel& best_fit(size_t budget_bytes) {
  for (const auto& m : catalog())
    if (comfortable_gb(m) * 1e9 <= double(budget_bytes)) return m;
  return catalog().back();
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

std::string short_name(const CatalogModel& m) {
  std::string s = m.name;
  for (const auto& a : m.aliases)
    if (a.size() < s.size()) s = a;
  return s;
}

static std::string choices() {
  std::string s;
  for (const auto& m : catalog()) {
    if (!s.empty()) s += ", ";
    s += m.name + " (" + short_name(m) + ")";
  }
  return s;
}

ModelRef resolve(const std::string& arg0) {
  std::string arg = arg0;
  while (!arg.empty() && std::isspace(static_cast<unsigned char>(arg.back()))) arg.pop_back();
  while (!arg.empty() && std::isspace(static_cast<unsigned char>(arg.front()))) arg.erase(arg.begin());
  if (arg.empty()) throw Error("--model needs a value: one of " + choices() + ", a Hugging Face repo (org/name) or a folder.");
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
    if (!is_dir(p)) throw Error("No folder at " + arg + ". --model takes a model folder, a name (" + choices() +
                                ") or a Hugging Face repo (org/name).");
    return local(p);
  }
  if (const CatalogModel* m = find_model(arg)) {
    r.kind = ModelRef::kCatalog;
    r.model = m;
    r.repo = m->repo;
    r.revision = m->revision;
    r.name = r.id = m->name;
    r.dir = models_dir() + "/" + m->name;
    return r;
  }
  if (arg.find('/') != std::string::npos && is_dir(arg)) return local(arg);
  std::string repo, rev;
  if (parse_repo(arg, repo, rev)) {
    const CatalogModel* m = find_model_by_repo(repo);
    if (m && (rev.empty() || rev == m->revision)) {  // the catalog's pinned files, with their companions
      r.kind = ModelRef::kCatalog;
      r.model = m;
      r.repo = m->repo;
      r.revision = m->revision;
      r.name = r.id = m->name;
      r.dir = models_dir() + "/" + m->name;
      return r;
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
  if (is_dir(arg)) return local(arg);
  throw Error("Unknown model \"" + arg + "\". Choose " + choices() +
              ", a Hugging Face repo of a Qwen3.5-family MLX model (org/name), or a model folder.");
}

}  // namespace ni::hub
