#include "server/service.h"
#include "server/runtime.h"
#include <cmath>
#include <set>
#include <stdexcept>

namespace ni::server {
namespace {
int integer(const ojson& j, const char* key, int fallback, int lo, int hi) {
  if (!j.contains(key)) return fallback;
  if (!j[key].is_number_integer() || (j[key].is_number_unsigned() && j[key].get<uint64_t>() > uint64_t(hi)))
    throw std::invalid_argument(std::string(key) + " must be a whole number in range");
  const auto n = j[key].get<int64_t>();
  if (n < lo || n > hi) throw std::invalid_argument(std::string(key) + " out of range");
  return int(n);
}
}  // namespace

Options native_options(const ojson& j) {
  const std::set<std::string> fields{"protocol", "instance_id", "name", "model", "port", "context", "slots",
    "mem_budget_gb", "spec", "dflash", "disk_cache", "prefill_chunk"};
  if (!j.is_object()) throw std::invalid_argument("runtime configuration must be an object");
  for (auto it = j.begin(); it != j.end(); ++it)
    if (!fields.count(it.key())) throw std::invalid_argument("unknown runtime field: " + it.key());
  if (integer(j, "protocol", 1, 1, 1) != runtime::kProtocol) throw std::invalid_argument("unsupported runtime protocol");
  Options o;
  o.headless = o.managed = o.yes = o.priv = o.local = true;
  o.port = integer(j, "port", 8000, 1, 65535);
  o.instance_id = j.value("instance_id", "serve-" + std::to_string(o.port));
  if (!runtime::valid_id(o.instance_id)) throw std::invalid_argument("invalid instance_id");
  o.name = j.value("name", o.instance_id);
  if (o.name.empty() || o.name.size() > 128) throw std::invalid_argument("name must be 1-128 bytes");
  o.model = j.at("model").get<std::string>();
  if (o.model.empty() || o.model.size() > 4096) throw std::invalid_argument("model is required");
  o.context_req = integer(j, "context", 0, 1, 1048576);
  o.engine.slots = integer(j, "slots", 0, 1, 4);
  o.engine.spec_depth = integer(j, "spec", -1, -1, 32);
  o.engine.prefill_chunk = integer(j, "prefill_chunk", 0, 1, 8192);
  o.engine.dflash = j.value("dflash", "");
  o.engine.disk_cache = j.value("disk_cache", "");
  if (!o.engine.disk_cache.empty()) {
    uint64_t bytes = 0;
    if (!pcache::parse_size(o.engine.disk_cache, &bytes)) throw std::invalid_argument("invalid disk_cache size");
  }
  if (j.contains("mem_budget_gb")) {
    if (!j["mem_budget_gb"].is_number()) throw std::invalid_argument("mem_budget_gb must be a number");
    o.engine.mem_budget_gb = j["mem_budget_gb"].get<double>();
    if (!std::isfinite(o.engine.mem_budget_gb) || o.engine.mem_budget_gb <= 0 || o.engine.mem_budget_gb > 4096)
      throw std::invalid_argument("mem_budget_gb out of range");
  }
  return o;
}
}  // namespace ni::server
