#include "loader/safetensors.h"
#include "util/log.h"
#include <array>
#include <cstdio>
#include <fstream>
#include <filesystem>
namespace ni {
Gpu& Gpu::get() { throw std::runtime_error("unexpected GPU access"); }
Buffer Gpu::alloc(size_t, const char*, bool) { throw std::runtime_error("unexpected allocation"); }
void Gpu::release(Buffer&) { throw std::runtime_error("unexpected release"); }
}
int main() {
  char tmp[] = "/tmp/kapitsa-dtype-probe-XXXXXX";
  if (!mkdtemp(tmp)) return 2;
  const std::string h = R"({"w":{"dtype":"BOOL","shape":[1],"data_offsets":[0,8]}})";
  { std::ofstream f(std::string(tmp)+"/model.safetensors",std::ios::binary);
    uint64_t n=h.size(); f.write((const char*)&n,8); f << h << "ABCDEFGH"; }
  ni::FatalRecovery scope;
  try {
    ni::WeightStore w; w.open_dir(tmp,{}, {}, [](const std::string&){return true;});
    std::array<char,8> guarded; guarded.fill('?');
    w.read_deferred("w",guarded.data());
    std::printf("shape/dtype destination bytes=%zu, declared tensor bytes=8; guard after destination='%.*s' (expected eight untouched guard bytes)\n", size_t(w.get("w").numel())*ni::dtype_size(w.get("w").dtype),8,guarded.data());
  } catch(const ni::FatalError& e) { std::printf("rejected before read: %s\n",e.what()); }
  std::filesystem::remove_all(tmp);
}
