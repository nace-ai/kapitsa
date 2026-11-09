// `kapitsa import` attaches the catalog's drafter to a GGUF import of a known shape (#364, #967).
//   drafter_attach_test        CPU, offline: which shape gets which drafter, and a folder that has one (or an unknown
//                              shape) is left alone
//   drafter_attach_test --net  also fetches the Qwen3.5-4B drafter (1.3 GB) into a temp folder, checks the files and
//                              the manifest, and that a second call fetches nothing
#include "hub/hub.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

using namespace ni;

static int g_fail = 0;
static void check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_fail;
}
static void write(const std::string& p, const std::string& s) { std::ofstream(p) << s; }
static bool exists(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0;
}

int main(int argc, char** argv) {
  const bool net = argc > 1 && !strcmp(argv[1], "--net");
  // Offline mode must never download: a point of the hub's fetch at a dead local address makes any fetch fail at once
  // (a missed case in this file once started a 3.8 GB download). --net leaves the network alone.
  if (!net) {
    setenv("HF_ENDPOINT", "http://127.0.0.1:9", 1);
    setenv("HF_HUB_OFFLINE", "1", 1);
  }
  auto repo = [](int h, int l) -> std::string {
    const hub::Companion* c = hub::drafter_for_shape(h, l);
    return c ? c->repo : "";
  };
  check(repo(5120, 64) == "nace-ai/Qwen3.8-27B-DFlash2-Kapitsa", "27B shape (5120, 64 layers): the 27B DFlash2 drafter");
  check(repo(2048, 40) == "incoai/Qwen3.6-35B-A3B-DFlash2", "35B-A3B shape (2048, 40): the 35B DFlash2 drafter");
  check(repo(2560, 32) == "z-lab/Qwen3.5-4B-DFlash", "4B shape (2560, 32): the 4B DFlash drafter");
  check(repo(4096, 32) == "z-lab/Qwen3.5-9B-DFlash", "9B shape (4096, 32): the 9B DFlash drafter");
  check(repo(5120, 62).empty() && repo(1024, 24).empty(), "an unknown shape gets none");

  char tmpl[] = "/tmp/drafter_attach_XXXXXX";
  const std::string root = mkdtemp(tmpl);
  auto folder = [&](const std::string& name, int h, int l) {
    const std::string d = root + "/" + name;
    mkdir(d.c_str(), 0755);
    write(d + "/config.json", "{\"hidden_size\": " + std::to_string(h) + ", \"num_hidden_layers\": " + std::to_string(l) + "}");
    return d;
  };
  const std::string unknown = folder("unknown", 1024, 24);
  check(hub::fetch_drafter(unknown, false).empty() && !exists(unknown + "/dflash2"), "an unknown shape: nothing fetched");
  const std::string has = folder("has", 5120, 64);
  mkdir((has + "/dflash2").c_str(), 0755);
  write(has + "/dflash2/model.safetensors", "x");  // its weights: an installed drafter (a folder without them is resumed)
  check(hub::fetch_drafter(has, false).empty() && !exists(has + "/dflash2/config.json"), "a folder with dflash2/ and weights: left alone");
  // a stopped download leaves config.json in dflash2/: that is not an installed drafter, the next fetch resumes it
  const std::string partial = folder("partial", 5120, 64);
  mkdir((partial + "/dflash2").c_str(), 0755);
  write(partial + "/dflash2/config.json", "{}");
  check(!hub::drafter_installed(partial), "dflash2/ with config.json alone: not an installed drafter (resumes)");
  write(partial + "/dflash2/model.safetensors", "x");
  check(hub::drafter_installed(partial), "dflash2/ with its weights: installed");
  // --no-drafter is remembered in the manifest; a plain import clears it
  const std::string declined = folder("declined", 5120, 64);
  mkdir((declined + "/.kapitsa").c_str(), 0755);
  write(declined + "/.kapitsa/manifest.json", "{\"files\": [], \"companions\": [], \"total_bytes\": 7}");
  check(!hub::drafter_declined(declined), "a fresh manifest: the drafter is not declined");
  hub::set_drafter_declined(declined, true);
  check(hub::drafter_declined(declined), "--no-drafter is remembered");
  hub::set_drafter_declined(declined, false);
  check(!hub::drafter_declined(declined), "a plain import clears it");
  const std::string nocfg = root + "/nocfg";
  mkdir(nocfg.c_str(), 0755);
  check(hub::fetch_drafter(nocfg, false).empty(), "a folder with no config.json: nothing");

  if (net) {
    const std::string d = folder("four", 2560, 32);
    mkdir((d + "/.kapitsa").c_str(), 0755);
    write(d + "/.kapitsa/manifest.json", "{\"files\": [], \"companions\": [], \"total_bytes\": 100}");
    const std::string got = hub::fetch_drafter(d, true);
    check(got == "Qwen3.5-4B-DFlash", "4B shape: fetched " + got);
    struct stat st;
    check(stat((d + "/dflash2/model.safetensors").c_str(), &st) == 0 && st.st_size > 1000000000, "dflash2/model.safetensors is there");
    check(exists(d + "/dflash2/config.json") && !exists(d + "/.kapitsa/parts/dflash2/model.safetensors.part"), "config.json there, no part left");
    check(hub::fetch_drafter(d, false).empty(), "a second call fetches nothing");
    auto manifest = [&] { std::ifstream f(d + "/.kapitsa/manifest.json"); return std::string((std::istreambuf_iterator<char>(f)), {}); };
    auto count = [](const std::string& h, const std::string& n) { size_t c = 0; for (size_t p = 0; (p = h.find(n, p)) != std::string::npos; p += n.size()) ++c; return c; };
    check(count(manifest(), "\"companions\"") == 1 && count(manifest(), "Qwen3.5-4B-DFlash") == 1, "the manifest lists the drafter once as a companion");
    check(hub::is_complete(d), "the drafter is not in \"files\": is_complete() ignores it");
    // deleting dflash2/ (disk cleanup) and fetching again: still one companion entry, bytes not counted twice
    std::system(("rm -rf '" + d + "/dflash2'").c_str());
    check(hub::is_complete(d), "a folder without its drafter is still a complete import");
    check(hub::fetch_drafter(d, false) == "Qwen3.5-4B-DFlash", "fetched again after the folder was deleted");
    check(count(manifest(), "Qwen3.5-4B-DFlash") == 1, "still one companion entry");
    const std::string tb = manifest().substr(manifest().find("\"total_bytes\":"));
    check(atoll(tb.c_str() + tb.find(':') + 1) > 1000000000 && atoll(tb.c_str() + tb.find(':') + 1) < 2000000000, "total_bytes counts the drafter once (100 + about 1.3 GB, not twice)");
  }
  std::string rm = "rm -rf " + root;
  if (root.rfind("/tmp/drafter_attach_", 0) == 0) std::system(rm.c_str());
  std::printf("%s\n", g_fail ? "FAILED" : "all ok");
  return g_fail ? 1 : 0;
}
