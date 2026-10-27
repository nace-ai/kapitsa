#include "splash/discover.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <unistd.h>

namespace fs = std::filesystem;

namespace ni::splash {
namespace {

bool is_file(const fs::path& p) {
  std::error_code ec;
  return fs::is_regular_file(p, ec);
}

bool is_executable_file(const fs::path& p) { return is_file(p) && access(p.c_str(), X_OK) == 0; }

fs::path models_dir(const std::string& home) { return fs::path(home) / "Library/Application Support/Splash/models/.resolved"; }

struct Model {
  std::string name, family;
};

std::vector<Model> models(const std::string& home) {
  std::vector<Model> out;
  std::error_code ec;
  for (fs::directory_iterator it(models_dir(home), ec), end; !ec && it != end; it.increment(ec)) {
    const fs::path& d = it->path();
    if (!is_file(d / "tokenizer/tokenizer.json")) continue;
    Model m{d.filename().string(), "?"};
    std::ifstream f(d / "model.json");
    auto j = nlohmann::json::parse(f, nullptr, false);
    if (j.is_object() && j.contains("family") && j["family"].is_string()) m.family = j["family"].get<std::string>();
    out.push_back(m);
  }
  std::sort(out.begin(), out.end(), [](const Model& a, const Model& b) { return a.name < b.name; });
  return out;
}

}  // namespace

Found find_binary(const std::string& home, const std::string& splash_home, const std::string& path_env,
                  const std::vector<std::string>& system_roots) {
  if (!splash_home.empty()) {  // named by the user: that install or an error, never another one
    fs::path bin = fs::path(splash_home) / "engine/splash";
    if (is_executable_file(bin)) return {bin.string(), ""};
    return {"", "SPLASH_HOME is " + splash_home + " but " + bin.string() + " is not an executable file.\nFix or unset SPLASH_HOME, or give --splash-binary <path to engine/splash>."};
  }
  std::vector<fs::path> roots;
  std::stringstream ss(path_env);
  for (std::string dir; std::getline(ss, dir, ':');) {
    if (dir.empty()) continue;
    std::error_code ec;
    fs::path exe = fs::canonical(fs::path(dir) / "splash", ec);  // through a symlink to where the install is
    if (ec) continue;
    roots.push_back(exe.parent_path().parent_path());  // <root>/engine/splash or <root>/bin/splash
  }
  roots.push_back(fs::path(home) / "Applications/Splash");
  for (const std::string& r : system_roots) roots.push_back(r);
  roots.push_back(fs::path(home) / "Splash");
  std::string tried;
  for (const fs::path& r : roots) {
    fs::path bin = r / "engine/splash";
    if (is_executable_file(bin)) return {bin.string(), ""};
    tried += "\n  " + bin.string();
  }
  return {"", "no Splash engine binary found; looked for:" + tried +
                  "\nInstall Splash, set SPLASH_HOME to its folder, or give --splash-binary <path to engine/splash>."};
}

std::vector<std::string> list_models(const std::string& home) {
  std::vector<std::string> out;
  for (const Model& m : models(home)) out.push_back(m.family + "  " + m.name);
  return out;
}

Found find_model(const std::string& home, const std::string& want) {
  std::vector<Model> all = models(home);
  const std::string dir = models_dir(home).string();
  if (all.empty())
    return {"", "no Splash model found: " + dir + " has no folder with a tokenizer. Run a model in Splash once, or give --splash-model <folder>."};
  if (want.empty()) {
    if (all.size() == 1) return {dir + "/" + all[0].name, ""};
    std::string e = "Splash has " + std::to_string(all.size()) + " resolved models in " + dir + "; say which with --splash-model <folder or its first 6+ characters>:";
    for (const Model& m : all) e += "\n  " + m.family + "  " + m.name;
    return {"", e};
  }
  if (want.size() < 6) return {"", "--splash-model " + want + ": a folder name prefix needs at least 6 characters"};
  std::vector<const Model*> hit;
  for (const Model& m : all)
    if (m.name.compare(0, want.size(), want) == 0) hit.push_back(&m);
  if (hit.size() == 1) return {dir + "/" + hit[0]->name, ""};
  if (hit.empty()) return {"", "--splash-model " + want + ": no resolved model in " + dir + " starts with it"};
  return {"", "--splash-model " + want + " matches " + std::to_string(hit.size()) + " models; give more characters"};
}

}  // namespace ni::splash
