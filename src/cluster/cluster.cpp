// Kapitsa on several Macs: what a server tells the others, and what is decided from it (cluster.h).
#include "cluster/cluster.h"

#include "util/term.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <tuple>

namespace ni::cluster {

using ojson = nlohmann::ordered_json;

namespace {

// Numbers another Mac sends past this are taken as missing (or, shown, as this): no RAM, speed or round trip comes near
constexpr double kMaxNum = 1e7;

std::string fmt0(double v) {
  char b[32];
  snprintf(b, sizeof b, "%.0f", std::isfinite(v) && v > 0 ? std::min(v, kMaxNum) : 0.0);
  return b;
}

std::string lower(std::string s) {
  for (auto& c : s) c = char(tolower(uint8_t(c)));
  return s;
}

// `s` cut to at most `n` bytes, not inside a UTF-8 character
std::string cut(std::string s, size_t n) {
  if (s.size() <= n) return s;
  s.resize(n);
  size_t i = s.size();  // where the last character starts, and how many of its continuation bytes are left
  int left = 0;
  while (i > 0 && left < 3 && (uint8_t(s[i - 1]) & 0xC0) == 0x80) --i, ++left;
  if (i > 0) {
    const uint8_t lead = uint8_t(s[i - 1]);
    if (left < (lead >= 0xF0 ? 3 : lead >= 0xE0 ? 2 : lead >= 0xC0 ? 1 : 0)) s.resize(i - 1);  // cut through it
  }
  return s;
}

// `v` as a number from 0 to kMaxNum, else `def`
double to_num(const std::string& v, double def) {
  char* end = nullptr;
  const double x = strtod(v.c_str(), &end);
  return v.empty() || *end || !std::isfinite(x) || x < 0 || x > kMaxNum ? def : x;
}

// How precise a variant is, for picking among a model's variants: its bits (bf16 16, "4bit" 4, "q2_0" 2, FP8 8,
// NVFP4 4, the GGUF small-memory variants "16gb" about 2), 0 when the name does not say
int precision_rank(const std::string& v) {
  if (v == "bf16" || v == "fp16" || v == "f16") return 16;
  auto ends = [&](const char* x) { const size_t n = strlen(x); return v.size() > n && v.compare(v.size() - n, n, x) == 0; };
  if (ends("bit")) return atoi(v.c_str());
  if (v.size() > 1 && v[0] == 'q' && isdigit(uint8_t(v[1]))) return atoi(v.c_str() + 1);
  if (v.find("fp8") != std::string::npos) return 8;
  if (v.find("fp4") != std::string::npos) return 4;
  if (ends("gb")) return 2;
  return 0;
}

std::string chip_short(const std::string& chip) {
  return chip.rfind("Apple ", 0) == 0 ? chip.substr(6) : chip;
}

}  // namespace

std::string encode_txt(const MacInfo& m) {
  std::string out;
  auto put = [&](const std::string& k, const std::string& v) {
    const std::string kv = cut(k + "=" + v, 255);  // a TXT string holds 255 bytes
    out += char(uint8_t(kv.size()));
    out += kv;
  };
  put("proto", std::to_string(m.proto));
  put("id", m.id);
  put("model", m.model);
  put("whole", m.whole ? "1" : "0");
  put("ram", fmt0(m.ram_gb));
  put("tps", fmt0(m.tok_s));
  return out;
}

MacInfo decode_txt(const void* txt, size_t len) {
  MacInfo m;
  m.proto = 0;  // a record without one is not a Kapitsa server's
  const auto* p = static_cast<const uint8_t*>(txt);
  for (size_t i = 0; p && i < len;) {
    const size_t n = p[i++];
    if (n > len - i) break;
    const std::string kv(reinterpret_cast<const char*>(p + i), n);
    i += n;
    const size_t eq = kv.find('=');
    if (eq == std::string::npos) continue;
    const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
    if (k == "proto") m.proto = int(to_num(v, 0));
    else if (k == "id") m.id = v;
    else if (k == "model") m.model = v;
    else if (k == "whole") m.whole = v == "1";
    else if (k == "ram") m.ram_gb = to_num(v, 0);
    else if (k == "tps") m.tok_s = to_num(v, 0);
  }
  return m;
}

static const char* const kKeys[] = {"proto", "id",      "name", "chip",      "ram_gb", "budget_gb", "free_gb", "model",
                                    "whole", "tok_s",   "version", "port",   "host",   "routes_to", "rtt_ms",  "link"};

ojson to_json(const MacInfo& m) {
  ojson j{{"proto", m.proto},         {"id", m.id},          {"name", m.name},     {"chip", m.chip},
          {"ram_gb", m.ram_gb},       {"budget_gb", m.budget_gb}, {"free_gb", m.free_gb}, {"model", m.model},
          {"whole", m.whole},         {"tok_s", m.tok_s},    {"version", m.version}, {"port", m.port},
          {"host", m.host},           {"routes_to", m.routes_to}, {"rtt_ms", m.rtt_ms}, {"link", m.link}};
  if (m.extra.is_object())
    for (const auto& [k, v] : m.extra.items())
      if (!j.contains(k)) j[k] = v;
  return j;
}

MacInfo from_json(const ojson& j) {
  MacInfo m;
  m.proto = 0;
  if (!j.is_object()) return m;
  auto str = [&](const char* k, std::string& out) {
    if (j.contains(k) && j[k].is_string()) out = j[k].get<std::string>();
  };
  // numbers from -1 (a round trip not answered) to kMaxNum, integers within +-2^30; others stay at their defaults
  auto num = [&](const char* k, double& out) {
    if (!j.contains(k) || !j[k].is_number()) return;
    const double x = j[k].get<double>();
    if (std::isfinite(x) && x >= -1 && x <= kMaxNum) out = x;
  };
  auto integer = [&](const char* k, int& out) {
    if (!j.contains(k)) return;
    const auto& v = j[k];
    if (v.is_number_unsigned() ? v.get<uint64_t>() < (uint64_t(1) << 30)
                               : v.is_number_integer() && v.get<int64_t>() > -(int64_t(1) << 30) &&
                                     v.get<int64_t>() < (int64_t(1) << 30))
      out = int(v.get<int64_t>());
  };
  integer("proto", m.proto);
  str("id", m.id);
  str("name", m.name);
  str("chip", m.chip);
  num("ram_gb", m.ram_gb);
  num("budget_gb", m.budget_gb);
  num("free_gb", m.free_gb);
  str("model", m.model);
  if (j.contains("whole") && j["whole"].is_boolean()) m.whole = j["whole"].get<bool>();
  num("tok_s", m.tok_s);
  str("version", m.version);
  integer("port", m.port);
  str("host", m.host);
  str("routes_to", m.routes_to);
  num("rtt_ms", m.rtt_ms);
  str("link", m.link);
  for (const auto& [k, v] : j.items())
    if (std::find_if(std::begin(kKeys), std::end(kKeys), [&](const char* x) { return k == x; }) == std::end(kKeys))
      m.extra[k] = v;
  return m;
}

std::string model_base(const std::string& model) {
  const std::string l = lower(model);
  return l.substr(0, l.find(':'));
}

std::string model_variant(const std::string& model) {
  const std::string l = lower(model);
  const size_t c = l.find(':');
  return c == std::string::npos ? "" : l.substr(c + 1);
}

std::optional<MacInfo> pick_server(const std::vector<MacInfo>& peers, const std::string& model, bool runs_whole_here) {
  if (runs_whole_here || model.empty()) return std::nullopt;
  const std::string base = model_base(model), variant = model_variant(model);
  // smaller is better: another variant, a less precise one, the round trip (unknown last), the speed
  auto rank = [&](const MacInfo& p) {
    const std::string v = model_variant(p.model);
    const bool same = v == variant;
    return std::make_tuple(same ? 0 : 1, same ? 0 : -precision_rank(v), p.rtt_ms > 0 ? p.rtt_ms : 1e9, -p.tok_s);
  };
  const MacInfo* best = nullptr;
  for (const auto& p : peers) {
    if (p.proto != kProto || !p.whole || !p.routes_to.empty() || p.rtt_ms < 0 || p.host.empty() || p.port <= 0) continue;
    if (model_base(p.model) != base) continue;
    if (!best || rank(p) < rank(*best)) best = &p;
  }
  if (!best) return std::nullopt;
  return *best;
}

std::string describe(const MacInfo& m) {
  std::string s = !m.name.empty() ? m.name : !m.host.empty() ? m.host : "a Mac";
  std::string what = chip_short(m.chip);
  if (m.ram_gb > 0) what += (what.empty() ? "" : ", ") + fmt0(m.ram_gb) + " GB";
  return what.empty() ? s : s + " (" + what + ")";
}

std::string link_text(const MacInfo& m) {
  if (m.rtt_ms < 0) return "not answering";
  if (m.this_mac()) return "this Mac";
  std::string s = m.link.empty() ? "network" : m.link;
  if (m.rtt_ms > 0) s += m.rtt_ms < 1 ? " <1 ms" : " " + fmt0(m.rtt_ms) + " ms";
  return s;
}

std::string how_it_runs(const MacInfo& m) {
  if (!m.routes_to.empty()) return "proxy";
  if (m.model.empty()) return "";
  return m.whole ? "in memory" : "streams from SSD";
}

std::vector<std::string> format_table(const MacInfo& here, const std::vector<MacInfo>& servers, int width) {
  using namespace term;
  struct Row {
    bool mine = false;
    std::string mac, chip, model, runs, tps, link;
  };
  std::vector<MacInfo> order;
  for (const auto& s : servers)
    if (s.this_mac()) order.push_back(s);
  std::sort(order.begin(), order.end(), [](const MacInfo& a, const MacInfo& b) { return a.port < b.port; });
  const size_t mine = order.size();
  for (const auto& s : servers)
    if (!s.this_mac()) order.push_back(s);
  std::sort(order.begin() + long(mine), order.end(), [](const MacInfo& a, const MacInfo& b) {
    return std::make_tuple(lower(a.name), a.host, a.port) < std::make_tuple(lower(b.name), b.host, b.port);
  });
  std::vector<Row> rows;
  auto chip_ram = [](const MacInfo& m) {
    std::string c = chip_short(m.chip);
    if (m.ram_gb > 0) c += (c.empty() ? "" : std::string(" ") + kDot + " ") + fmt0(m.ram_gb) + " GB";
    return c;
  };
  if (mine == 0) {  // no server here: this Mac all the same, first
    Row r;
    r.mine = true;
    r.mac = here.name.empty() ? "this Mac" : here.name;
    r.chip = chip_ram(here);
    r.model = dim("not serving");
    r.link = "this Mac";
    rows.push_back(r);
  }
  for (const auto& m : order) {
    Row r;
    r.mine = m.this_mac();
    r.mac = !m.name.empty() ? m.name : m.host;
    r.chip = chip_ram(m);
    r.model = m.model.empty() ? dim("idle") : m.model;
    r.runs = how_it_runs(m);
    if (!m.routes_to.empty()) r.runs = "proxy " + std::string(kArrow) + " " + m.routes_to;
    else if (r.runs == "streams from SSD") r.runs = warn(r.runs);
    r.tps = m.tok_s > 0 ? fmt0(m.tok_s) : dim("–");
    r.link = m.rtt_ms < 0 ? warn(link_text(m)) : link_text(m);
    rows.push_back(r);
  }
  // column widths: the headers' and the cells', then narrower where the line would not fit
  int wm = 3, wc = 10, wo = 5, wr = 4, wt = 5, wl = 4;  // "Mac", "Chip · RAM", "Model", "Runs", "tok/s", "Link"
  for (const auto& r : rows) {
    wm = std::max(wm, term::width(r.mac)), wc = std::max(wc, term::width(r.chip)), wo = std::max(wo, term::width(r.model));
    wr = std::max(wr, term::width(r.runs)), wt = std::max(wt, term::width(r.tps)), wl = std::max(wl, term::width(r.link));
  }
  bool chip = true;
  // "  ● " and the columns, 3 spaces apart
  auto total = [&] { return 4 + wm + 3 + (chip ? wc + 3 : 0) + wo + 3 + wr + 3 + wt + 3 + wl; };
  // what gives way first: a proxy's Mac in the Runs column, a long Mac name, the chip, then the model; the room the chip
  // leaves goes back to the names
  const int full_m = wm, full_r = wr;
  while (total() > width && wr > 16) --wr;
  while (total() > width && wm > 14) --wm;
  if (total() > width) chip = false;
  while (total() > width && wm > 10) --wm;
  while (total() > width && wr > 9) --wr;
  while (total() > width && wo > 14) --wo;
  while (total() < width && wm < full_m) ++wm;
  while (total() < width && wr < full_r) ++wr;
  std::vector<std::string> out;
  auto line = [&](const std::string& mark, const std::string& mac, const std::string& c, const std::string& model,
                  const std::string& runs, const std::string& tps, const std::string& link) {
    std::string l = "  " + mark + " " + left(clip(mac, wm), wm) + "   ";
    if (chip) l += left(clip(c, wc), wc) + "   ";
    l += left(clip(model, wo), wo) + "   " + left(clip(runs, wr), wr) + "   " + right(tps, wt) + "   " + link;
    while (!l.empty() && l.back() == ' ') l.pop_back();
    return clip(l, width);  // a terminal narrower still: the end of the line goes
  };
  out.push_back(dim(line(" ", "Mac", "Chip · RAM", "Model", "Runs", "tok/s", "Link")));
  for (const auto& r : rows)
    out.push_back(line(r.mine ? accent(kOn) : " ", r.mine ? bold(r.mac) : r.mac, r.chip, r.model, r.runs, r.tps, r.link));
  return out;
}

}  // namespace ni::cluster
