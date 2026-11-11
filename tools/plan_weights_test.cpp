// A plan's model is on this Mac only when its weights are in it (issue #1187): planning writes config.json and the
// safetensors headers, no weights, so `kapitsa doctor` called a downloaded catalog model "planned from its headers".
// hub::is_plan() and hub::plan_has_model() are what decides the label, and the doctor line is checked end to end on plan
// folders this test makes from a downloaded model's own config.json and file names (its shards are sparse files, so no
// weights are written). Needs no GPU and loads no model.
//   plan_weights_test [--keep]
#include "hub/hub.h"

#include <fcntl.h>
#include <mach-o/dyld.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace hub = ni::hub;

extern char** environ;

static int failures = 0;
static bool keep = false;

static void check(bool c, const std::string& what) {
  if (!c) {
    ++failures;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

static void write_file(const std::string& path, const std::string& text) {
  std::ofstream(path, std::ios::binary) << text;
}

static std::string read_text(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// a file of `size` bytes that is only allocated where it is written: a header, or a whole shard, without the weights
static void sparse_file(const std::string& path, uint64_t size) {
  std::ofstream(path, std::ios::binary).close();
  check(truncate(path.c_str(), off_t(size)) == 0, "make a file of " + std::to_string(size) + " B at " + path);
}

// A plan as planning leaves one holds header-sized shards, not the tiny sparse ones here: the model is not in it until a
// shard is bigger than its header (kHeaderBytes, 1 MB in fetch.cpp). `sparse_file` grows a shard past that where a test
// wants the weights, and this shrinks the rest back to the header of a real file, a few hundred KB.
static void shrink_to_header(const std::string& path) { check(truncate(path.c_str(), off_t(256 << 10)) == 0, "shrink " + path); }

// the shard files a safetensors index names, in order
static std::vector<std::string> index_shards(const std::string& index) {
  std::vector<std::string> out;
  size_t at = 0;
  while ((at = index.find("\"model", at)) != std::string::npos) {
    const size_t start = at + 1, end = index.find_first_of("\\\"", start);
    if (end == std::string::npos) break;
    std::string f = index.substr(start, end - start);
    if (f.size() > 5 && f.compare(f.size() - 5, 5, ".json") != 0 && f.find(".safetensors") != std::string::npos &&
        std::find(out.begin(), out.end(), f) == out.end())
      out.push_back(f);
    at = end;
  }
  return out;
}

// `kapitsa doctor <flag> <arg>`, its stdout (the report comes either way and the exit is not what this checks)
static std::string run_doctor(const std::string& program, const std::string& flag, const std::string& arg,
                              const std::string& home, const std::string& log) {
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_addopen(&fa, 1, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
  std::string a0 = program, d = "doctor", f = flag, v = arg;
  char* argv[] = {a0.data(), d.data(), f.data(), v.data(), nullptr};
  std::string home_env = "KAPITSA_HOME=" + home;
  std::vector<char*> envp;
  for (char** e = environ; *e; ++e)
    if (std::string(*e).rfind("KAPITSA_HOME=", 0) != 0) envp.push_back(*e);
  envp.push_back(home_env.data());
  envp.push_back(nullptr);
  pid_t pid = 0;
  const int rc = posix_spawn(&pid, program.c_str(), &fa, nullptr, argv, envp.data());
  posix_spawn_file_actions_destroy(&fa);
  if (rc != 0) return "";
  int status = 0;
  waitpid(pid, &status, 0);
  return read_text(log);
}

// Whether the report calls this plan not downloaded, in either shape doctor's label line gives it
static bool says_planned(const std::string& report, const std::string& name) {
  return report.find("  " + name + ", not downloaded: planned from its headers") != std::string::npos ||
         report.find("/" + name + ": not downloaded: planned from its headers") != std::string::npos;
}

int main(int argc, char** argv) {
  keep = argc > 1 && std::string(argv[1]) == "--keep";
  // the CLI beside this test's own binary (build/plan_weights_test -> build/kapitsa)
  char exe[4096] = {0};
  uint32_t size = sizeof exe;
  _NSGetExecutablePath(exe, &size);
  const std::string kapitsa_bin = (fs::path(fs::weakly_canonical(exe)).parent_path() / "kapitsa").string();
  // doctor's catalog section is planned offline by its own visit (doctor.cpp), so a run here makes no network request
  // for a variant this Mac lacks; --models-dir makes the report the folders below, not this Mac's models.

  // a model a real install already holds: its own config.json and shard names, as `kapitsa doctor` reads them. Read
  // before KAPITSA_HOME points this test at its own folder, and only test fixtures come out of it (sizes, not weights).
  std::string config, index, model;
  for (const auto& m : hub::catalog()) {
    if (!config.empty()) break;
    for (const auto& v : m.variants) {
      const std::string d = hub::models_dir() + "/" + m.name + "-" + v.name;
      if (fs::exists(d + "/config.json") && fs::exists(d + "/model.safetensors.index.json")) {
        config = read_text(d + "/config.json"), index = read_text(d + "/model.safetensors.index.json");
        model = m.name + "-" + v.name;
        break;
      }
    }
  }
  if (config.empty() || index.empty() || index_shards(index).empty()) {
    std::printf("SKIP: no downloaded catalog model to make a plan of (its config.json and index are the fixture)\n");
    return 0;
  }
  const std::string root = (fs::temp_directory_path() / ("k1187-plans-" + std::to_string(::getpid()))).string();
  fs::remove_all(root);
  const std::string home = root + "/home", plans = home + "/plans";
  fs::create_directories(plans);
  setenv("KAPITSA_HOME", home.c_str(), 1);
  const std::vector<std::string> shards = index_shards(index);
  std::printf("the fixture is %s: %zu shard(s)\n", model.c_str(), shards.size());

  // a plan as plan_dir leaves one: config.json, the index, each shard as the header it was planned from
  auto plan = [&](const std::string& name) {
    const std::string d = plans + "/" + name;
    fs::create_directories(d);
    write_file(d + "/config.json", config);
    write_file(d + "/model.safetensors.index.json", index);
    write_file(d + "/.complete", "planned");
    for (const std::string& s : shards) {
      write_file(d + "/" + s, "");
      shrink_to_header(d + "/" + s);
    }
    return d;
  };

  // planned from its headers: what plan_dir leaves for a model never pulled
  const std::string bare = plan("kw-planned");
  check(hub::is_plan(bare), "a folder under plans/ is a plan");
  check(!hub::plan_has_model(bare), "a plan holding headers, not weights, has no model");

  // downloaded into the plan: the same shards, holding the weights
  const std::string full = plan("kw-weight");
  for (const std::string& s : shards) sparse_file(full + "/" + s, 3 << 20);
  check(hub::plan_has_model(full), "a plan holding its weights has the model");

  // a model split into shards ("model-00001-of-00002.safetensors") is in its plan when its first shard is, and the
  // index beside them lists the shards without being one
  const std::string split = plans + "/kw-split";
  fs::create_directories(split);
  write_file(split + "/config.json", config);
  write_file(split + "/model.safetensors.index.json",
             "{\"weight_map\": {\"a\": \"model-00001-of-00002.safetensors\", \"b\": \"model-00002-of-00002.safetensors\"}}");
  write_file(split + "/.complete", "planned");
  write_file(split + "/model-00001-of-00002.safetensors", "");
  shrink_to_header(split + "/model-00001-of-00002.safetensors");
  check(!hub::plan_has_model(split), "a split model whose first shard is a header has no model");
  sparse_file(split + "/model-00001-of-00002.safetensors", 3 << 20);
  check(hub::plan_has_model(split), "a split model whose first shard holds the weights has the model");

  // not a plan: the store's own folder, and anything else
  check(!hub::is_plan(home + "/models/" + model), "the store is not a plan");
  check(!hub::is_plan(home), "the home is not a plan");
  check(!hub::plan_has_model(home + "/models/" + model), "an empty store folder has no model");

  // the catalogue's plan of a model, and the store's copy of it under the same name: the copy with the model stands
  // for the name, whatever order the folders come in, and it is not called not downloaded
  const std::string store_copy = home + "/models/" + model;
  fs::create_directories(store_copy);
  write_file(store_copy + "/config.json", config);
  write_file(store_copy + "/model.safetensors.index.json", index);
  write_file(store_copy + "/model.safetensors", "");
  sparse_file(store_copy + "/model.safetensors", 3 << 20);

  // the doctor line, end to end
  if (!fs::exists(kapitsa_bin)) {
    std::printf("SKIP: the doctor line (%s is not built)\n", kapitsa_bin.c_str());
  } else {
    const std::string log = home + "/doctor.txt";
    const std::string report = run_doctor(kapitsa_bin, "--models-dir", plans, home, log);
    check(!report.empty(), "doctor printed a report (" + log + ")");
    if (keep && !report.empty()) std::printf("%s\n", report.c_str());
    // the plan the split model names is listed: a plan that is only planned keeps a name of its own in the report
    check(report.find("kw-split") != std::string::npos, "doctor lists a plan of a model split into shards");
    check(says_planned(report, "kw-planned"), "doctor: a plan with only headers says planned from its headers");
    // a plan holding its own weights is a copy of a model on this Mac: it is listed, and not called not downloaded
    check(report.find("kw-weight") != std::string::npos, "doctor: a plan with its weights is listed at all");
    check(!says_planned(report, "kw-weight"), "doctor: a plan with its weights is not called not downloaded");
    // the store's copy of a model the catalogue plans from its headers: one row, the store's, and it is not called
    // not downloaded, whatever order the folders are listed in
    int rows = 0;
    for (size_t at = report.find("  " + model); at != std::string::npos; at = report.find("  " + model, at + 1)) ++rows;
    check(rows == 1, "doctor: the store's copy and its plan are one row");
    check(report.find("  " + model + " (") != std::string::npos,
          "doctor: the store's copy of the model is listed as downloaded");
  }

  if (keep) std::printf("kept %s\n", root.c_str());
  else fs::remove_all(root);
  if (failures == 0) std::printf("PASS: a plan has its model only when its weights are in it\n");
  return failures == 0 ? 0 : 1;
}
