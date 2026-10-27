// Where Splash's engine binary and resolved model folders are on this Mac, so `kapitsa serve --engine splash` can run
// without --splash-binary and --splash-model (docs/splash-backend.md). Pure functions of the paths given: no process
// state, so the test builds a throwaway Splash tree.
#pragma once

#include <string>
#include <vector>

namespace ni::splash {

// A path found, or why not. `error` names what is missing and where it was looked for, and the flag that replaces it.
struct Found {
  std::string path, error;
  bool ok() const { return error.empty(); }
};

// The engine binary (<root>/engine/splash), looked for in: $SPLASH_HOME (`splash_home`), the directory a `splash` on PATH
// resolves into (<root>/bin/.. or the binary itself), then <home>/Applications/Splash, the `system_roots` (default
// /Applications/Splash) and <home>/Splash. A test passes {} so the Mac it runs on does not decide the result.
Found find_binary(const std::string& home, const std::string& splash_home, const std::string& path_env,
                  const std::vector<std::string>& system_roots = {"/Applications/Splash"});

// A resolved model folder under <home>/Library/Application Support/Splash/models/.resolved/ (the ones with a tokenizer).
// `want` empty: the only one there is; with several the error lists each (family, folder name) and asks for one.
// `want` a folder name or a prefix of one (at least 6 characters, unique): that one.
Found find_model(const std::string& home, const std::string& want);

// The same listing, as "family  folder" lines (the order the error uses).
std::vector<std::string> list_models(const std::string& home);

}  // namespace ni::splash
