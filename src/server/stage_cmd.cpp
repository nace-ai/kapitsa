// kapitsa stage: this Mac holds a run of a model's layers for another Mac's `kapitsa serve --split` (src/cluster/pipe.h):
// Qwen3.8-Flash-Next split across Macs by layers, so that Macs whose memory each holds only part of it run it whole.
#include "cluster/pipe.h"
#include "engine/doctor.h"
#include "engine/engine.h"
#include "hub/hub.h"
#include "metal/gpu.h"
#include "model/qwen4_exp.h"
#include "server/cli.h"
#include "util/log.h"
#include "util/term.h"

#include <nlohmann/json.hpp>

#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>

namespace ni {

namespace {

const char* kHelp =
    "kapitsa stage --model M [--layers A-B] [--port P] [--host H] [--mem-budget GB] [--prefill-chunk N]\n\n"
    "Holds a run of Qwen3.8-Flash-Next's layers for another Mac on the network, the head, which holds the first ones\n"
    "and takes the requests:\n"
    "  this Mac:  kapitsa stage --model flash-next\n"
    "  the head:  kapitsa serve --model flash-next --split <this Mac>.local\n"
    "The head asks each stage for its memory and speed and gives each the layers that fit it with the fastest pass;\n"
    "the stage loads them then. Each pass of the model runs the head's layers, then this Mac's, and comes back: one\n"
    "hand-off a Mac whatever the rows (a speculative verify of several tokens included). Every Mac needs the same model\n"
    "files. The stage serves one head at a time and waits for the next one when it leaves. More stages: the head\n"
    "lists them (--split a.local,b.local).\n\n"
    "  --model M          the model (a catalog name, a Hugging Face repo or a folder), downloaded if needed\n"
    "  --layers A-B       hold layers A to B (1-based, inclusive; \"25-\": to the last) whatever the head would place:\n"
    "                     every stage of a split is started with them then, each from where the one before it ends\n"
    "  --port P           the port the head connects to (default 9330)\n"
    "  --host H           the address to listen on (default 0.0.0.0: every network of this Mac)\n"
    "  --mem-budget GB    the GPU memory Kapitsa may use here (default: Metal's working set less a margin)\n"
    "  --prefill-chunk N  the most rows a pass may bring (default 2112: the head's 2048-row chunks)\n";

std::string mac_name() {
  char b[256] = "";
  if (gethostname(b, sizeof b) != 0 || !*b) return "this Mac";
  std::string n = b;
  if (n.size() > 6 && n.compare(n.size() - 6, 6, ".local") == 0) n.resize(n.size() - 6);
  return n;
}

}  // namespace

int run_stage(int argc, char** argv, int first) {
  std::string model, layers, host = "0.0.0.0";
  int port = cluster::kStagePort, chunk = prefill_chunk_room(2048);
  double budget_gb = 0;
  const char* help = "kapitsa stage --help";
  for (int i = first; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) usage_error(a + " needs a value", help);
      return argv[++i];
    };
    if (a == "--help" || a == "-h") {
      fputs(kHelp, stdout);
      return 0;
    } else if (a == "--model" || a == "-m") model = next();
    else if (a == "--layers") layers = next();
    else if (a == "--port") port = to_int(a, next(), help);
    else if (a == "--host") host = next();
    else if (a == "--mem-budget") budget_gb = atof(next().c_str());
    else if (a == "--prefill-chunk") chunk = to_int(a, next(), help);
    else usage_error("unknown option " + a, help);
  }
  if (model.empty()) usage_error("--model is needed", help);
  if (port < 1 || port > 65535) usage_error("--port must be 1-65535", help);
  // the model, downloaded first if needed
  std::string dir;
  try {
    const hub::ModelRef ref = hub::resolve(expand_home(model));
    hub::PullOptions po;
    dir = hub::ensure(ref, po);
  } catch (const hub::Stopped&) {
    return 130;
  } catch (const hub::Error& e) {
    return fail(e.what());
  }
  if (!Qwen4Exp::is_model(dir))
    return fail(dir + " is not Qwen3.8-Flash-Next: only that model splits across Macs so far.");
  cluster::StageOptions o;
  o.model_dir = dir;
  o.host = host;
  o.port = port;
  o.name = mac_name();
  o.mem = double(budget_gb > 0 ? size_t(budget_gb * 1e9) : default_mem_budget());
  o.gbs = bandwidth_gbs(Gpu::get().name(), Gpu::get().core_count());
  o.chunk = chunk;
  if (!layers.empty()) {
    const int L = Qwen35::read_config(dir).layers;
    const size_t dash = layers.find('-');
    const int a = atoi(layers.c_str());
    const int b = dash != std::string::npos && dash + 1 < layers.size() ? atoi(layers.c_str() + dash + 1) : L;
    // the head keeps the layers up to the n-gram block's (it reads the tokens): Flash-Next's first two
    const nlohmann::json root = nlohmann::json::parse(std::ifstream(dir + "/config.json"), nullptr, false);
    const nlohmann::json t = root.is_object() && root.contains("text_config") ? root["text_config"] : root;
    const int first = t.is_object() && t.contains("ple_layer_ids") && t["ple_layer_ids"].is_array() &&
                              !t["ple_layer_ids"].empty() ? t["ple_layer_ids"][0].get<int>() + 1 : 2;
    if (a < first || b > L || a > b)
      usage_error("--layers " + layers + ": A-B from " + std::to_string(first) + " to " + std::to_string(L) +
                  " (the head keeps layers 1-" + std::to_string(first - 1) + ", the n-gram block's among them)", help);
    o.lo = a - 1;
    o.hi = b;
  }
  o.status = [](const std::string& s) {
    char t[16];
    const time_t now = time(nullptr);
    strftime(t, sizeof t, "%H:%M:%S", localtime(&now));
    printf("%s  %s\n", t, s.c_str());
    fflush(stdout);
  };
  printf("Stage for %s on %s (%s, %.1f GB of GPU memory%s). On the head Mac: kapitsa serve --model %s --split %s.local%s\n",
         dir.substr(dir.find_last_of('/') + 1).c_str(), o.name.c_str(), Gpu::get().name().c_str(), o.mem / 1e9,
         o.gbs > 0 ? (", " + std::to_string(int(o.gbs)) + " GB/s").c_str() : "", model.c_str(), o.name.c_str(),
         port == cluster::kStagePort ? "" : (":" + std::to_string(port)).c_str());
  fflush(stdout);
  cluster::run_stage(o);
}

}  // namespace ni
