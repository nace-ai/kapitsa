// Models by name: the built-in catalog, the local model store (~/.cache/kapitsa/models; KAPITSA_HOME moves the
// base), Hugging Face downloads (parallel ranges, resumable, size- and checksum-verified, reusing the Hugging Face
// cache and folders the user names), the companions that make a model faster (MTP head, DFlash2 drafter), and
// the MTP sidecar builder (a port of tools/mtp_sidecar.py with byte-identical output).
#pragma once
#include <cstddef>
#include <cstdint>
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

// Files fetched with a model to make it faster. The repos and revisions are the Nace Agent app's catalog
// (packages/nace-runtime/src/catalog-data.ts), so the app and the CLI download the same bytes.
struct Companion {
  std::string kind;                // "mtp": an MTP head's files; "dflash2": a DFlash2 drafter;
                                   // "mtp-sidecar": mtp.safetensors built from an official checkpoint's mtp.* tensors
  std::string repo, revision;
  std::vector<std::string> files;  // the files taken from the repo ("mtp-sidecar": none)
  std::string subdir;              // where they go in the model's folder ("" = beside the weights)
  std::string sha256;              // "mtp-sidecar": sha256 of the result (a pinned revision always builds the same file)
  uint64_t size = 0;               // "mtp-sidecar": its size in bytes
};

struct CatalogModel {
  std::string name;                  // canonical name, the folder and the API's model id: "qwen3.6-35b-a3b"
  std::vector<std::string> aliases;  // "qwen3.6-35b", "35b"
  std::string title;                 // "Qwen3.6-35B-A3B"
  std::string repo, revision;        // the weights (every file but .gitattributes)
  std::vector<Companion> companions;
  double download_gb = 0;            // weights + companions
  double fixed_gb = 0;               // GPU memory at load with its companions (2 slots, see catalog.cpp)
  double kv_bytes_per_token = 0;     // KV cache per token of conversation
  std::string about;                 // one line for `kapitsa list`
};

const std::vector<CatalogModel>& catalog();
const CatalogModel* find_model(const std::string& name_or_alias);  // case-insensitive; nullptr if unknown
const CatalogModel* find_model_by_repo(const std::string& repo);   // the catalog entry whose weights are `repo`
std::string short_name(const CatalogModel& m);                     // its shortest alias: "35b"

// GPU memory a model needs to start with room for a 16k-token conversation.
double comfortable_gb(const CatalogModel& m);
// The best catalog model for a GPU memory budget (bytes): the first of the catalog that fits comfortably, else the
// smallest. The catalog is ordered best first.
const CatalogModel& best_fit(size_t budget_bytes);

std::string home_dir();     // $KAPITSA_HOME, else ~/.cache/kapitsa
std::string models_dir();   // home_dir() + "/models"
std::string tilde(const std::string& path);  // $HOME/x -> ~/x, for messages

// What --model names.
struct ModelRef {
  enum Kind { kCatalog, kRepo, kLocal } kind = kLocal;
  const CatalogModel* model = nullptr;  // kCatalog
  std::string repo, revision;           // kCatalog, kRepo ("" revision: the repo's main branch)
  std::string name;                     // for messages: the catalog name, "org/name[@rev]" or the folder
  std::string id;                       // the API's model id
  std::string dir;                      // the model's folder
};
// A catalog name or alias ("qwen3.6-35b", "35b" ...), a Hugging Face repo id "org/name[@revision]", or a local
// folder. Throws Error with the choices when `arg` is none of them.
ModelRef resolve(const std::string& arg);

struct PullOptions {
  std::vector<std::string> from;  // folders to take files from that match the pinned files' checksums
  bool progress = true;           // progress line on stderr
};

// Makes sure everything the model needs is in ref.dir and returns it. A finished download (its manifest, written
// last) is reused without any network access. Otherwise: list the repos, take what is already there (finished
// files, the Hugging Face cache, `from` folders; checksums verified, APFS clones so no extra disk), check the free
// disk space, download the rest in parallel ranges (resumable: run again after an interruption), build the MTP
// sidecar if the model has one, verify sizes and sha256, and write the manifest. Throws Error.
std::string ensure(const ModelRef& ref, const PullOptions& opt);
// Every file the manifest lists is there (a finished download).
bool is_complete(const std::string& dir);
// Finished, and (catalog models) at the revisions this Kapitsa pins: ensure() needs no network.
bool is_current(const ModelRef& ref);

// A model in the store, for `kapitsa list`.
struct Installed {
  std::string name, dir, repo, commit;
  bool complete = false;
  uint64_t bytes = 0;                // on disk (finished files + downloaded parts)
  uint64_t total = 0;                // when complete or known
  std::vector<std::string> extras;   // "MTP head", "DFlash2 drafter"
};
std::vector<Installed> installed();

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
