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
  std::string note;  // set when the pick was not asked for: which one it is and how to change it (print it)
  bool ok() const { return error.empty(); }
};

// The engine binary (<root>/engine/splash), looked for in: $SPLASH_HOME (`splash_home`), the directory a `splash` on PATH
// resolves into (<root>/bin/.. or the binary itself), then <home>/Applications/Splash, the `system_roots` (default
// /Applications/Splash) and <home>/Splash. A test passes {} so the Mac it runs on does not decide the result.
// `last` (what the last `serve --engine splash` ran, read_last) is tried after all of those, so a Splash installed anywhere
// needs --splash-binary once.
Found find_binary(const std::string& home, const std::string& splash_home, const std::string& path_env,
                  const std::vector<std::string>& system_roots = {"/Applications/Splash"}, const std::string& last = "");

// A resolved model folder under <home>/Library/Application Support/Splash/models/.resolved/ (the ones with a tokenizer).
// `want` empty: the only one there is; with several the error lists each (family, folder name) and asks for one.
// `want` a folder name or a prefix of one (at least 6 characters, unique): that one.
// `want` else a family or source name, any case, as a part of it ("flash-next", "27b", "Q2_0"): the one model it names; when
// that is several folders of one family and source (Splash keeps each re-resolution), the newest; else an error that lists them.
// `want` empty with several there: `last` (the folder the last run used) when it is one of them.
Found find_model(const std::string& home, const std::string& want, const std::string& last = "");

// The same listing, as "family  folder" lines (the order the error uses).
std::vector<std::string> list_models(const std::string& home);

// What the last `kapitsa serve --engine splash` ran: the engine binary and the model folder (empty when unknown), kept in a
// small JSON file (<kapitsa home>/splash.json) so the next run needs neither flag. Reading a missing or bad file gives empty
// fields; writing one that is already so, or that fails, does nothing.
struct Last {
  std::string binary, model;
};
Last read_last(const std::string& file);
void write_last(const std::string& file, const Last& last);

// What `kapitsa serve --engine splash --check` prints: the engine binary and the model it would run (given, or found as above),
// the resolved models there are, and for whatever is missing the error find_binary / find_model gives, with the command that
// fixes it. `ok` when both are found. Pure: nothing is started.
struct Check {
  std::string text;
  bool ok = false;
};
// The rule for a --splash-binary given by the user, used by serve and --check alike: a file that can be run, else the error.
Found explicit_binary(const std::string& path);

Check check(const std::string& home, const std::string& splash_home, const std::string& path_env, const std::string& binary,
            const std::string& model, const Last& last,
            const std::vector<std::string>& system_roots = {"/Applications/Splash"});

}  // namespace ni::splash
