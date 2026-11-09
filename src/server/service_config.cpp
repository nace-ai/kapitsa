// The one validator of `kapitsa serve`'s settings (service.h), and the command line's flags as its JSON.
#include "server/service.h"

#include "engine/prefix_cache.h"
#include "model/dory.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace ni::server {
using nlohmann::json;

namespace {
enum class Kind { text, flag, whole, gb, list, comma_list };
struct Field {
  const char* key;
  Kind kind;
};
// Every serve flag, as its JSON field (`--prefill-chunk` is prefill_chunk), in the order of `kapitsa serve --help`'s
// groups; docs/serve-config.schema.json lists the same names
constexpr Field kFields[] = {
    {"model", Kind::text},          {"model_id", Kind::text},        {"engine", Kind::text},
    {"splash_binary", Kind::text},  {"splash_model", Kind::text},    {"splash_arg", Kind::list},
    {"splash_persistent_cache", Kind::flag}, {"splash_conversation_salt", Kind::flag},
    {"splash_kflash", Kind::text},  {"check", Kind::flag},           {"host", Kind::text},
    {"port", Kind::whole},          {"profile", Kind::text},         {"private", Kind::flag},
    {"local", Kind::flag},          {"peer", Kind::list},            {"context", Kind::text},
    {"slots", Kind::whole},         {"spec", Kind::whole},           {"dflash", Kind::text},
    {"prefill_chunk", Kind::whole}, {"api_key", Kind::text},         {"allow_switch", Kind::flag},
    {"idle_unload", Kind::text},    {"mem_budget", Kind::gb},        {"image_max_pixels", Kind::whole},
    {"timing", Kind::text},         {"disk_cache", Kind::text},      {"loops", Kind::whole},
    {"loop_kv", Kind::text},        {"split", Kind::comma_list},     {"verbose", Kind::flag},
    {"yes", Kind::flag},
};

const Field* field(const std::string& key) {
  for (const Field& f : kFields)
    if (key == f.key) return &f;
  return nullptr;
}

std::string flag_name(const std::string& key) {
  std::string s = "--" + key;
  for (char& c : s)
    if (c == '_') c = '-';
  return s;
}

[[noreturn]] void refuse(const std::string& why) { throw std::invalid_argument(why); }

// A whole JSON number, as JSON Schema counts one: 8000 and 8000.0 both are, 8000.5 is not
bool integral(const json& v) {
  if (v.is_number_integer()) return true;
  if (!v.is_number_float()) return false;
  const double d = v.get<double>();
  return std::isfinite(d) && d == std::floor(d) && std::fabs(d) < 9.2e18;
}

long long integral_value(const json& v) {
  if (v.is_number_float()) return (long long)v.get<double>();
  if (v.is_number_unsigned() && v.get<unsigned long long>() > (unsigned long long)LLONG_MAX) return LLONG_MAX;
  return v.get<long long>();
}

// The JSON's fields, read with the names errors give them
struct Reader {
  const json& j;
  Source source;
  std::string name(const char* key) const { return source == Source::flags ? flag_name(key) : key; }
  bool has(const char* key) const { return j.contains(key); }
  std::string text(const char* key, std::string fallback = "") const {
    if (!has(key)) return fallback;
    if (!j[key].is_string()) refuse(name(key) + " must be a string");
    return j[key].get<std::string>();
  }
  bool flag(const char* key) const {
    if (!has(key)) return false;
    if (!j[key].is_boolean()) refuse(name(key) + " must be true or false");
    return j[key].get<bool>();
  }
  // a whole number from 0 (what the flags' to_int takes) up to `hi`
  long long whole(const char* key, long long fallback, long long hi = INT_MAX) const {
    if (!has(key)) return fallback;
    const json& v = j[key];
    if (!integral(v) || integral_value(v) < 0 || integral_value(v) > hi)
      refuse(name(key) + " needs a whole number from 0 to " + std::to_string(hi) + ", not " + v.dump());
    return integral_value(v);
  }
  std::vector<std::string> strings(const char* key) const {
    if (!has(key)) return {};
    const json& v = j[key];
    if (!v.is_array()) refuse(name(key) + " must be a list of strings");
    std::vector<std::string> out;
    for (const json& s : v) {
      if (!s.is_string()) refuse(name(key) + " must be a list of strings");
      out.push_back(s.get<std::string>());
    }
    return out;
  }
};

// disk_cache's forms in JSON (the schema's pattern): auto, off, no, none, 0, or a number with its unit (16G, 512MiB,
// 1.5T, 4096B): a bare number of bytes ("16") is refused, as the flag's hex, exponents and spaces around it are
bool json_size(const std::string& s) {
  static const std::regex form("^(auto|off|no|none|0|[0-9]*[.]?[0-9]+ ?([kmgt](i?b)?|b))$", std::regex::icase);
  return std::regex_match(s, form);
}

bool loopback(const std::string& host) { return host == "127.0.0.1" || host == "localhost" || host == "::1"; }
}  // namespace

const std::vector<std::string>& config_fields() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v;
    for (const Field& f : kFields) v.push_back(f.key);
    return v;
  }();
  return names;
}

Options parse_options(const json& j, Source source) {
  if (!j.is_object()) refuse("the configuration must be a JSON object");
  const Reader r{j, source};
  for (auto it = j.begin(); it != j.end(); ++it)
    if (!field(it.key()))
      refuse(source == Source::flags ? "unknown option " + flag_name(it.key()) : "unknown field \"" + it.key() + "\"");
  Options o;
  EngineOptions& e = o.engine;
  o.model = r.text("model");
  e.model_id = r.text("model_id");
  if (const std::string v = r.text("engine", "native"); v == "splash" || v == "native") o.engine_splash = v == "splash";
  else refuse(r.name("engine") + " takes native or splash, not \"" + v + "\"");
  e.splash_binary = r.text("splash_binary");
  e.splash_model = r.text("splash_model");
  e.splash_args = r.strings("splash_arg");
  e.splash_persistent_cache = r.flag("splash_persistent_cache");
  e.splash_conversation_salt = r.flag("splash_conversation_salt");
  o.splash_kflash = r.text("splash_kflash", "auto");
  o.check_only = r.flag("check");
  o.host = r.text("host");
  o.port = int(r.whole("port", 8000));
  if (o.port < 1 || o.port > 65535) refuse(r.name("port") + " must be 1-65535");
  o.profile = r.text("profile");
  if (r.has("profile") && o.profile != "fast" && o.profile != "precise")
    refuse(r.name("profile") + " takes fast or precise, not \"" + o.profile + "\"");
  o.priv = r.flag("private");
  o.local = r.flag("local");
  o.peers = r.strings("peer");
  if (r.has("context")) {  // tokens, k or m of them, or max; JSON may give the tokens as a number
    const json& v = j["context"];
    const std::string s = v.is_string() ? v.get<std::string>()
                          : integral(v) && integral_value(v) >= 0 ? std::to_string(integral_value(v))
                                                                   : v.dump();
    std::string why;
    // in JSON a token count is a number: "131072" is refused, "128k" and "max" are strings
    const bool digits = !s.empty() && s.find_first_not_of("0123456789") == std::string::npos;
    if (v.is_string() && digits && source == Source::json)
      refuse(r.name("context") + " takes the token count as a number (131072), not the string \"" + s +
             "\"; k and m forms (\"128k\", \"1m\") and \"max\" are strings");
    o.context = (integral(v) && integral_value(v) >= 0) || v.is_string() ? parse_context(s, &why) : 0;
    if (!o.context)
      refuse(r.name("context") + " takes tokens (131072), k or m of them (128k, 1m) or max, not \"" + s + "\"");
  }
  e.slots = int(r.whole("slots", e.slots));
  e.spec_depth = int(r.whole("spec", e.spec_depth));
  e.dflash = r.text("dflash");
  e.prefill_chunk = int(r.whole("prefill_chunk", e.prefill_chunk));
  o.api_key = r.text("api_key");
  o.allow_switch = r.flag("allow_switch");
  if (r.has("idle_unload")) {  // auto, 0 or off, or minutes (0.5 to a week)
    const json& v = j["idle_unload"];
    const std::string s = v.is_string() ? v.get<std::string>() : v.dump();
    double m = -2;
    if (s == "auto") m = -1;
    else if (s == "0" || s == "off") m = 0;
    else if (v.is_number() && v.get<double>() == 0) m = 0;
    else if (v.is_number() || (v.is_string() && source == Source::flags)) {  // JSON: minutes are a number
      char* end = nullptr;
      const double x = strtod(s.c_str(), &end);
      if (end != s.c_str() && !*end && x >= 0.5 && x <= 10080) m = x;  // (NaN is refused too)
    }
    if (m == -2)
      refuse(r.name("idle_unload") + " takes auto, 0 (off) or the minutes the server idles first (0.5 or more), not \"" +
             s + "\"");
    o.idle_unload_min = m;
  }
  if (r.has("mem_budget")) {
    const json& v = j["mem_budget"];
    if (!v.is_number() || !(v.get<double>() > 0))
      refuse(r.name("mem_budget") + " needs a size in GB, not " + v.dump());
    e.mem_budget_gb = v.get<double>();
  }
  e.image_max_pixels = long(r.whole("image_max_pixels", e.image_max_pixels));
  o.timing = r.text("timing", "both");
  if (o.timing != "ttft" && o.timing != "prefill" && o.timing != "both")
    refuse(r.name("timing") + " takes ttft, prefill or both, not \"" + o.timing + "\"");
  e.disk_cache = r.text("disk_cache");
  if (uint64_t bytes = 0; r.has("disk_cache") && (!pcache::parse_size(e.disk_cache, &bytes) ||
                                                  (source == Source::json && !json_size(e.disk_cache))))
    refuse(r.name("disk_cache") + " takes auto, 0 (off) or a size" + (source == Source::json ? " with its unit" : "") +
           " such as 16G, not \"" + e.disk_cache + "\"");
  if (r.has("loops")) {
    e.loops = int(r.whole("loops", 0));
    if (e.loops < 1 || e.loops > Dory::kMaxLoops) refuse(r.name("loops") + ": 1 to 4");
  }
  e.loop_kv = r.text("loop_kv");
  if (LoopKv kv; r.has("loop_kv") && !parse_loop_kv(e.loop_kv, &kv)) refuse(r.name("loop_kv") + ": last-loop or per-loop");
  o.split = r.strings("split");
  if (r.has("split") && (o.split.empty() || std::find(o.split.begin(), o.split.end(), "") != o.split.end()))
    refuse(r.name("split") + " takes the stages' host:port, comma-separated");
  o.verbose = r.flag("verbose");
  o.yes = r.flag("yes");

  // combinations
  if (o.check_only && !o.engine_splash) refuse(r.name("check") + " needs " + r.name("engine") + " splash");
  if (o.engine_splash) {  // Splash's engine serves its model folder, here and nowhere else
    if (!o.model.empty() || !o.split.empty() || o.allow_switch)
      refuse(r.name("engine") + " splash serves the " + r.name("splash_model") + " folder: " + r.name("model") + ", " +
             r.name("split") + " and " + r.name("allow_switch") + " do not apply");
    if (o.splash_kflash != "auto" && o.splash_kflash != "on" && o.splash_kflash != "off")
      refuse(r.name("splash_kflash") + " must be auto, on or off");
  } else if (!e.splash_binary.empty() || !e.splash_model.empty() || !e.splash_args.empty() ||
             e.splash_persistent_cache || e.splash_conversation_salt || o.splash_kflash != "auto") {
    refuse(r.name("splash_binary") + ", " + r.name("splash_model") + ", " + r.name("splash_arg") + ", " +
           r.name("splash_persistent_cache") + ", " + r.name("splash_conversation_salt") + " and " +
           r.name("splash_kflash") + " need " + r.name("engine") + " splash");
  }
  // --allow-switch lets any client load models: without --api-key only on this Mac (an empty host then becomes
  // 127.0.0.1: run_service)
  if (o.allow_switch && o.api_key.empty() && !o.host.empty() && !loopback(o.host))
    refuse(r.name("allow_switch") + " on a network address needs " + r.name("api_key") +
           " (any client could load models)");
  return o;
}

std::optional<json> flags_to_json(const std::vector<std::string>& args, bool positional_model) {
  json j = json::object();
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a == "--help" || a == "-h") return std::nullopt;
    if (i == 0 && positional_model && !a.empty() && a[0] != '-') {  // kapitsa serve 35b
      j["model"] = a;
      continue;
    }
    const std::string key = a == "-m" ? "model" : a == "--ctx" ? "context" : a.rfind("--", 0) == 0 ? [&] {
      std::string k = a.substr(2);
      for (char& c : k)
        if (c == '-') c = '_';
      return k;
    }() : "";
    // (a flag is spelled with '-' only: --prefill_chunk is not one)
    const Field* f = a.find('_') == std::string::npos ? field(key) : nullptr;
    if (!f) refuse("unknown option " + a);
    if (f->kind == Kind::flag) {
      j[key] = true;
      continue;
    }
    if (i + 1 >= args.size()) refuse(a + " needs a value");
    const std::string& v = args[++i];
    switch (f->kind) {
      case Kind::whole: {  // as to_int: 0 to INT32_MAX
        char* end = nullptr;
        errno = 0;
        const long x = strtol(v.c_str(), &end, 10);
        if (v.empty() || *end || errno || x < 0 || x > INT32_MAX) refuse(a + " needs a whole number, not \"" + v + "\"");
        j[key] = x;
        break;
      }
      case Kind::gb: {  // a size in GB above 0
        char* end = nullptr;
        const double x = strtod(v.c_str(), &end);
        if (v.empty() || *end || !(x > 0)) refuse(a + " needs a size in GB, not \"" + v + "\"");
        j[key] = x;
        break;
      }
      case Kind::list:
        j[key].push_back(v);
        break;
      case Kind::comma_list: {  // each flag adds its comma-separated parts: --split a:1 --split b:2 is two stages
        json& parts = j[key];
        if (!parts.is_array()) parts = json::array();
        for (size_t at = 0; at <= v.size();) {
          const size_t comma = std::min(v.find(',', at), v.size());
          if (comma > at) parts.push_back(v.substr(at, comma - at));
          at = comma + 1;
        }
        break;
      }
      default:
        j[key] = v;
    }
  }
  return j;
}

json read_config(const std::string& from) {
  std::ostringstream text;
  if (from == "-") {
    text << std::cin.rdbuf();
    keep_stdin(text.str());
  } else {
    std::ifstream f(from);
    if (!f) throw std::invalid_argument("can't read " + from);
    text << f.rdbuf();
  }
  json j = json::parse(text.str(), nullptr, false);
  if (j.is_discarded()) throw std::invalid_argument((from == "-" ? std::string("stdin") : from) + " is not JSON");
  return j;
}

void keep_stdin(const std::string& text) {
  std::string path = (getenv("TMPDIR") && *getenv("TMPDIR") ? std::string(getenv("TMPDIR")) : std::string("/tmp")) +
                     "/kapitsa-config-XXXXXX";
  const int fd = mkstemp(path.data());  // 0600
  if (fd < 0) throw std::invalid_argument("can't keep stdin's JSON for a restart: " + std::string(strerror(errno)));
  unlink(path.c_str());
  bool ok = true;
  for (size_t at = 0; ok && at < text.size();) {
    const ssize_t n = write(fd, text.data() + at, text.size() - at);
    if (n > 0) at += size_t(n);
    else ok = n < 0 && errno == EINTR;
  }
  ok = ok && lseek(fd, 0, SEEK_SET) == 0 && dup2(fd, STDIN_FILENO) == STDIN_FILENO;
  const int err = errno;
  close(fd);
  if (!ok) throw std::invalid_argument("can't keep stdin's JSON for a restart: " + std::string(strerror(err)));
}

}  // namespace ni::server
