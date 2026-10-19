// Models by name: the built-in catalog, the local model store (~/.cache/kapitsa/models; KAPITSA_HOME moves the
// base), Hugging Face downloads (parallel ranges, resumable, size- and checksum-verified, reusing the Hugging Face
// cache and folders the user names), the companions that make a model faster (MTP head, DFlash2 drafter), and
// the MTP sidecar builder (a port of tools/mtp_sidecar.py with byte-identical output).
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ni::hub {

// A failure to report to the user as is (offline, not found, out of disk ...): what() is one or two sentences.
struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};
// Ctrl+C during a download: the parts are kept, the same command continues it.
struct Stopped : Error {
  using Error::Error;
};

// Files fetched with a model to make it faster. The repos and revisions match the Nace Agent app's model catalog, so
// the app and the CLI download the same bytes.
struct Companion {
  std::string kind;                // "mtp": an MTP head's files; "dflash2": a DFlash2 drafter ("dflash": a DFlash v1
                                   // drafter, in the same dflash2/ folder); "lilicorr": a LiLiCorr drafter (Nemotron
                                   // 3.5 Bolt, Super-VL); "dspark": a DSpark drafter (Nemotron 3.5 Lightning);
                                   // "mtp-sidecar": mtp.safetensors built from an official checkpoint's mtp.* tensors
  std::string repo, revision;
  std::vector<std::string> files;  // the files taken from the repo ("mtp-sidecar": none)
  std::string subdir;              // where they go in the model's folder ("" = beside the weights)
  std::string sha256;              // "mtp-sidecar": sha256 of the result (a pinned revision always builds the same file)
  uint64_t size = 0;               // "mtp-sidecar": its size in bytes
  // a draft model (DFlash2, DFlash v1, LiLiCorr, DSpark), not an MTP head's files
  bool drafter() const { return kind == "dflash2" || kind == "dflash" || kind == "lilicorr" || kind == "dspark"; }
};

// One way to run a model: its weights at one precision, with the companions that speed them up. `--model
// <model>:<variant>` picks one, `--model <model>` the model's default.
struct Variant {
  std::string name;                   // "4bit", "8bit", "6bit", "bf16" (lower case): what follows the ':'
  std::string quant;                  // what the weights are: "MLX affine 4-bit, group 64"
  std::string label;                  // how the home screen names it ("": from the name, 4bit -> 4-bit)
  std::string repo, revision;         // the weights (every file but .gitattributes)
  std::vector<Companion> companions;
  double download_gb = 0;             // weights + companions
};

// The models of the catalog (src/hub/catalog.cpp): Qwen from small to large, then Nemotron from small to large.
struct CatalogModel {
  std::string name;                  // "qwen3.6-35b-a3b"; with the variant, the folder (qwen3.6-35b-a3b-4bit)
  std::vector<std::string> aliases;  // "qwen3.6-35b", "35b"
  std::string title;                 // "Qwen3.6-35B-A3B"
  std::string about;                 // one line for --help
  std::string brief;                 // a few words for the models table of `kapitsa` in a terminal ("": about)
  std::string default_variant;       // the variant `--model <name>` takes where it fits (the API names it <name>)
  // On a Mac the default does not fit (the engine's memory plan of its headers: one --ctx conversation with its
  // drafter), `--model <name>` takes the first of these that fits, else the last: smaller variants, best first.
  std::vector<std::string> fallbacks;
  std::vector<Variant> variants;
  // The variant named `v` (case-insensitive; "" = the default), nullptr if the model has none of that name.
  const Variant* variant(const std::string& v = "") const;
};

const std::vector<CatalogModel>& catalog();
const CatalogModel* find_model(const std::string& name_or_alias);  // case-insensitive; nullptr if unknown
// The catalog variant whose weights are `repo` (case-insensitive) at `revision` ("" = any), or nullptr.
const Variant* find_variant_by_repo(const std::string& repo, const std::string& revision, const CatalogModel** model);
std::string short_name(const CatalogModel& m);                     // its shortest alias: "35b"
// "qwen3.5-4b:4bit", and "4b" / "4b:8bit" (a variant other than the default is always named)
std::string full_name(const CatalogModel& m, const Variant& v);
std::string short_name(const CatalogModel& m, const Variant& v);


std::string home_dir();     // $KAPITSA_HOME, else ~/.cache/kapitsa
std::string models_dir();   // home_dir() + "/models"
std::string tilde(const std::string& path);  // $HOME/x -> ~/x, for messages

// What --model names.
struct ModelRef {
  enum Kind { kCatalog, kRepo, kLocal } kind = kLocal;
  const CatalogModel* model = nullptr;  // kCatalog
  const Variant* variant = nullptr;     // kCatalog
  std::string repo, revision;           // kCatalog, kRepo ("" revision: the repo's main branch)
  std::string name;                     // for messages and `list`: "qwen3.5-4b:4bit", "org/name[@rev]" or the folder
  std::string id;                       // the API's model id: "qwen3.5-4b" (default variant), "qwen3.5-4b:8bit", ...
  std::string dir;                      // the model's folder
  bool variant_given = false;           // kCatalog: named by <name>:<variant> or by its repo (not the default)
};
// A catalog name or alias with an optional variant ("35b", "qwen3.6-35b", "27b:8bit" ...), a Hugging Face repo id
// "org/name[@revision]", or a local folder. Throws Error with the choices when `arg` is none of them.
ModelRef resolve(const std::string& arg);

// A download that does not fit the disk: what it needs and what is free beyond the 1 GB ensure() keeps spare
struct SpaceShort {
  std::string model;  // "qwen3.6-35b-a3b:4bit"
  std::string dir;    // its folder (not one to delete to make room)
  uint64_t need = 0;  // bytes left to download
  uint64_t free = 0;  // bytes free beyond the spare
};

struct PullOptions {
  std::vector<std::string> from;  // folders to take files from that match the pinned files' checksums
  bool progress = true;           // progress line on stderr
  // Asked when the download does not fit the disk (in a terminal, ui::make_room offers to delete other downloaded
  // models); true when it freed space, and the check runs again
  std::function<bool(const SpaceShort&)> make_room;
};

// Makes sure everything the model needs is in ref.dir and returns it. A finished download (its manifest, written
// last) is reused without any network access. Otherwise: list the repos, take what is already there (finished
// files, and in a folder Kapitsa did not download only those that match the pinned checksums; the Hugging Face cache;
// `from` folders; checksums verified, APFS clones so no extra disk), check the free disk space, download the rest in
// parallel ranges (resumable: run again after an interruption), build the MTP
// sidecar if the model has one, verify sizes and sha256, and write the manifest. Throws Error.
std::string ensure(const ModelRef& ref, const PullOptions& opt);
// Every file the manifest lists is there (a finished download).
bool is_complete(const std::string& dir);
// Finished, and (catalog models) at the revisions this Kapitsa pins: ensure() needs no network.
bool is_current(const ModelRef& ref);

// What a memory plan reads for a model, without its weights: config.json and each safetensors file's header (the
// 8-byte length and the JSON, no tensor data), the companions' too (the MTP head's, the drafter's in dflash2/). For a
// downloaded model, a local folder: the model's own folder. For a catalog model not downloaded yet: fetched once into
// ~/.cache/kapitsa/plans/<name> (a few hundred KB; pinned revisions, so it never changes). "" when that fails
// (offline) or for a repo outside the catalog.
std::string plan_dir(const ModelRef& ref);
// While set, plan_dir() fetches nothing: a plan not already on this Mac is "" (and counted). For answers that must
// be instant and offline, like the help screen's best model for this Mac: exact when offline_misses() stays 0.
void set_offline(bool on);
int offline_misses();

// A model in the store, for `kapitsa list`.
struct Installed {
  std::string name, variant, dir, repo, commit;  // name: "qwen3.5-4b:4bit" (catalog), "org/name" or the folder
  bool complete = false;
  uint64_t bytes = 0;                // on disk (finished files + downloaded parts)
  uint64_t total = 0;                // when complete or known
  std::vector<std::string> extras;   // "MTP head", "DFlash2 drafter"
};
std::vector<Installed> installed();
// What deleting `dir` gives back to the disk: each file's data that no APFS clone shares (ATTR_CMNEXT_PRIVATESIZE),
// else its allocated size. A model cloned from another copy (Nace Agent's, a --from folder, the Hugging Face cache)
// frees little or nothing.
uint64_t freeable_bytes(const std::string& dir);

// tools/mtp_sidecar.py: mtp.safetensors from an official Qwen3.5 checkpoint's 15 bf16 mtp.* tensors (a Hugging
// Face repo, fetched by range requests at `revision`, or a local folder of HF safetensors). Norm weights get + 1
// (MLX convention), the header records base_hidden_variant. `source` is what the header names (the repo id).
void build_mtp_sidecar(const std::string& src, const std::string& revision, const std::string& out,
                       const std::string& hidden = "post_norm");

// Hex sha256 of a file (CommonCrypto).
std::string sha256_file(const std::string& path);
// "3.30 GB", "241 MB", "12 KB"
std::string fmt_bytes(uint64_t bytes);

}  // namespace ni::hub
