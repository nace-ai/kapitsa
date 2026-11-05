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
  auto repo = [](int h, int l) -> std::string {
    const hub::Companion* c = hub::drafter_for_shape(h, l);
    return c ? c->repo : "";
  };
  check(repo(5120, 64) == "incoai/Qwen3.8-27B-DFlash2", "27B shape (5120, 64 layers): the 27B DFlash2 drafter");
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
  check(hub::fetch_drafter(has, false).empty() && !exists(has + "/dflash2/model.safetensors"), "a folder with dflash2/: left alone");
  const std::string nocfg = root + "/nocfg";
  mkdir(nocfg.c_str(), 0755);
  check(hub::fetch_drafter(nocfg, false).empty(), "a folder with no config.json: nothing");

  if (net) {
    const std::string d = folder("four", 2560, 32);
    const std::string got = hub::fetch_drafter(d, true);
    check(got == "Qwen3.5-4B-DFlash", "4B shape: fetched " + got);
    struct stat st;
    check(stat((d + "/dflash2/model.safetensors").c_str(), &st) == 0 && st.st_size > 1000000000, "dflash2/model.safetensors is there");
    check(exists(d + "/dflash2/config.json") && !exists(d + "/.kapitsa/parts/dflash2/model.safetensors.part"), "config.json there, no part left");
    check(hub::fetch_drafter(d, false).empty(), "a second call fetches nothing");
  }
  std::string rm = "rm -rf " + root;
  if (root.rfind("/tmp/drafter_attach_", 0) == 0) std::system(rm.c_str());
  std::printf("%s\n", g_fail ? "FAILED" : "all ok");
  return g_fail ? 1 : 0;
}
