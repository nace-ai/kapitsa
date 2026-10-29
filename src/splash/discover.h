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

// KFlash's Splash engine: Splash with Kapitsa's patches (packaging/splash/README.md), which packaging/splash/build.py builds
// into <kapitsa home>/engines/splash-kflash-<tag>-<id>/ (engine/splash, kernels/ksx_h1.metal, kflash.json). The patches are
// off unless the child's environment turns them on (SPLASH_KSX_H1=3, SPLASH_KSX_KERNELS=<build>/kernels).
struct KFlash {
  std::string binary, kernels;  // empty when there is no complete build
  std::string tag;              // the Splash release its patches are on (kflash.json's "splash_tag")
};
// The most recently built complete one there (by its kflash.json's time).
KFlash find_kflash(const std::string& kapitsa_home);

// The resolved models KFlash runs by default: the ones an interleaved A/B against stock Splash measured faster on this
// kind of Mac (docs/splash-backend.md lists them with the run). Matched on model.json's "model" (the source), exactly.
bool kflash_by_default(const std::string& model_folder);

// The release of the Splash engine at `binary` (<release>/engine/splash): release.json's "version" next to it, else "".
std::string release_version(const std::string& binary);

// --splash-kflash: auto runs KFlash for kflash_by_default's models when a build is there, no --splash-binary was given
// and the stock engine it would replace is the release KFlash is built on (the A/B measured it only against that one:
// against a newer Splash it may be slower, so auto keeps the newer stock engine and says so); on runs it for any model (an error when there is no build, or with --splash-binary); off never.
enum class KFlashMode { automatic, on, off };
struct EnginePick {
  std::string binary;            // what to start: the KFlash build's engine, else `binary` as given or found
  std::vector<std::string> env;  // "NAME=value" for the child: empty for stock Splash
  std::string note, error;       // note: which engine and why (print it); error: why `on` cannot be honoured
};
EnginePick pick_engine(KFlashMode mode, const std::string& binary, bool binary_given, const KFlash& kflash,
                       const std::string& model_folder);

}  // namespace ni::splash
