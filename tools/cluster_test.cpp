// Kapitsa on several Macs (src/cluster/), CPU only, no network: the Bonjour TXT record and GET /cluster's JSON, each
// encoded and read back; where a Mac that would not hold a model whole sends its requests (pick_server); and the table
// of `kapitsa cluster`, laid out from made-up servers.
//   cluster_test          the checks above; exit status 1 if any fails
//   cluster_test --live   also on this Mac's network: a server advertised over Bonjour (a stand-in HTTP server on a
//                         free port) must be found, read and timed by a Browser within a few seconds
#include "cluster/cluster.h"
#include "cluster/discovery.h"
#include "util/log.h"
#include "util/term.h"

#include <httplib/httplib.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

using namespace ni;
using namespace ni::cluster;
using ojson = nlohmann::ordered_json;

namespace {

int fails = 0, passes = 0;

void check(bool ok, const std::string& name, const std::string& detail = "") {
  printf("%s %s%s\n", ok ? "PASS" : "FAIL", name.c_str(), ok || detail.empty() ? "" : ("\n     " + detail).c_str());
  ok ? ++passes : ++fails;
}

// `s` without its escape sequences (the table is styled when the test runs in a terminal)
std::string plain(const std::string& s) {
  std::string out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] != '\033') {
      out += s[i];
      continue;
    }
    if (i + 1 < s.size() && s[i + 1] == '[') {
      i += 2;
      while (i < s.size() && !(s[i] >= 0x40 && s[i] <= 0x7E)) ++i;
    } else {
      ++i;
    }
  }
  return out;
}

MacInfo server(const std::string& name, const std::string& model, bool whole, double rtt, double tps,
               const std::string& link = "Wi-Fi") {
  MacInfo m;
  m.id = name + "-id";
  m.name = name;
  m.chip = "Apple M3 Ultra";
  m.ram_gb = 512;
  m.model = model;
  m.whole = whole;
  m.rtt_ms = rtt;
  m.tok_s = tps;
  m.host = "192.168.1.20";
  m.port = 8000;
  m.link = link;
  return m;
}

std::string picked(const std::vector<MacInfo>& peers, const std::string& model, bool whole_here = false) {
  const auto p = pick_server(peers, model, whole_here);
  return p ? p->name : "(none)";
}

// Servers named by address (--peer, KAPITSA_PEERS, <home>/peers), for networks that block Bonjour between Macs
void peer_checks() {
  char tmpl[] = "/tmp/cluster_test.XXXXXX";
  const std::string home = mkdtemp(tmpl);
  setenv("KAPITSA_HOME", home.c_str(), 1);
  setenv("KAPITSA_PEERS", " studio.local:8001, 10.0.0.5 ,studio.local:8001", 1);
  FILE* f = fopen((home + "/peers").c_str(), "w");
  fputs("# office Macs\n10.0.0.7:8000   # the M3\n\n10.0.0.5\n", f);
  fclose(f);
  auto all = [] {
    std::string s;
    for (const auto& p : named_peers()) s += p + ";";
    return s;
  };
  check(all() == "studio.local:8001;10.0.0.5;10.0.0.7:8000;", "peers: the variable's and the file's, trimmed, once each",
        all());
  add_peer("10.0.0.9:8002");
  check(all() == "studio.local:8001;10.0.0.5;10.0.0.9:8002;10.0.0.7:8000;", "peers: --peer adds to the variable", all());
  unsetenv("KAPITSA_PEERS");
  remove((home + "/peers").c_str());
  rmdir(home.c_str());
  unsetenv("KAPITSA_HOME");
}

void txt_checks() {
  MacInfo m;
  m.id = "0123456789abcdef";
  m.model = "qwen3.8-flash-next:4bit";
  m.whole = true;
  m.ram_gb = 128;
  m.tok_s = 58;
  const std::string t = encode_txt(m);
  check(t.size() > 8 && uint8_t(t[0]) == 7 && t.compare(1, 7, "proto=1") == 0, "txt: length-prefixed key=value strings",
        std::to_string(uint8_t(t[0])));
  const MacInfo d = decode_txt(t.data(), t.size());
  check(d.proto == kProto && d.id == m.id && d.model == m.model && d.whole && d.ram_gb == 128 && d.tok_s == 58,
        "txt: proto, id, model, whole, ram, tps read back", to_json(d).dump());
  check(d.name.empty() && d.port == 0 && d.chip.empty(), "txt: what it does not carry stays empty");
  m.tok_s = 58.6, m.ram_gb = 47.9;
  const MacInfo r = decode_txt(encode_txt(m).data(), encode_txt(m).size());
  check(r.tok_s == 59 && r.ram_gb == 48, "txt: numbers as whole numbers", to_json(r).dump());
  // a model name past a TXT string's 255 bytes is cut, not inside a character
  m.model = std::string(247, 'a') + "é" + "xyz";
  const std::string lt = encode_txt(m);
  const MacInfo l = decode_txt(lt.data(), lt.size());
  check(l.model == std::string(247, 'a') + "é" && l.whole, "txt: a long value cut at 255 bytes, on a character boundary",
        std::to_string(l.model.size()));
  m.model = "a=b";
  const std::string et = encode_txt(m);
  check(decode_txt(et.data(), et.size()).model == "a=b", "txt: '=' in a value");
  // malformed: a length past the end keeps what came before; no proto is not a Kapitsa record
  std::string bad = std::string(1, char(7)) + "proto=1" + std::string(1, char(9)) + "id=12345";  // 9 bytes claimed, 8 left
  const MacInfo b = decode_txt(bad.data(), bad.size());
  check(b.proto == 1 && b.id.empty(), "txt: a string past the end is dropped, the ones before it kept");
  const std::string none = std::string(1, char(5)) + "id=ab";
  check(decode_txt(none.data(), none.size()).proto == 0 && decode_txt(nullptr, 0).proto == 0,
        "txt: no proto (or nothing) is not a Kapitsa server's");
}

void json_checks() {
  MacInfo m = server("Studio", "qwen3.8-flash-next:4bit", true, 4.25, 58.5);
  m.version = "0.2.10";
  m.budget_gb = 491.5;
  m.free_gb = 380.25;
  m.routes_to = "";
  m.since = 1792000000.5;
  m.stage_port = 9330;
  m.stage_of = "0123456789abcdef";
  m.extra["layers"] = ojson::array({0, 20});
  const MacInfo r = from_json(ojson::parse(to_json(m).dump()));
  check(r.proto == m.proto && r.id == m.id && r.name == m.name && r.chip == m.chip && r.ram_gb == m.ram_gb &&
            r.budget_gb == m.budget_gb && r.free_gb == m.free_gb && r.model == m.model && r.whole == m.whole &&
            r.tok_s == m.tok_s && r.version == m.version && r.port == m.port && r.host == m.host &&
            r.routes_to == m.routes_to && r.since == m.since && r.stage_port == m.stage_port &&
            r.stage_of == m.stage_of && r.rtt_ms == m.rtt_ms && r.link == m.link,
        "json: every field read back", to_json(r).dump());
  check(r.extra == m.extra && to_json(r).contains("layers"), "json: fields this protocol does not name kept",
        r.extra.dump());
  const MacInfo bad = from_json(ojson::parse(R"({"proto":1,"port":"80","whole":"yes","ram_gb":"lots","name":7})"));
  check(bad.proto == 1 && bad.port == 0 && !bad.whole && bad.ram_gb == 0 && bad.name.empty(),
        "json: mistyped fields stay at their defaults");
  check(from_json(ojson::parse("[1,2]")).proto == 0 && from_json(ojson::parse("{}")).proto == 0,
        "json: not an object, or no proto: not a Kapitsa server's");
  // out of range: as missing (another Mac's numbers are not trusted to fit an int, or a line)
  const MacInfo huge = from_json(ojson::parse(
      R"({"proto":1,"port":18446744073709551615,"ram_gb":1e308,"tok_s":-5,"rtt_ms":1e30,"budget_gb":9223372036854775808})"));
  check(huge.proto == 1 && huge.port == 0 && huge.ram_gb == 0 && huge.tok_s == 0 && huge.rtt_ms == 0 &&
            huge.budget_gb == 0,
        "json: numbers out of range stay at their defaults", to_json(huge).dump());
  const std::string big = std::string(1, char(10)) + "proto=1e30" + std::string(1, char(10)) + "ram=1e308" +
                          std::string(1, char(6)) + "tps=-3";
  const MacInfo bt = decode_txt(big.data(), big.size());
  check(bt.proto == 0 && bt.ram_gb == 0 && bt.tok_s == 0, "txt: numbers out of range taken as missing",
        to_json(bt).dump());
  MacInfo shown = server("Big", "m", true, 1e308, 1e300);
  check(link_text(shown).size() < 40 && plain(format_table(shown, {shown}, 200)[2]).find("10000000") != std::string::npos,
        "a number out of range is shown cut to 10000000, not as a wrong one", link_text(shown));
  const auto keys = to_json(MacInfo{});
  std::string names;
  for (const auto& [k, v] : keys.items()) names += k + " ";
  check(names == "proto id name chip ram_gb budget_gb free_gb model whole tok_s version port host routes_to since "
                 "stage_port stage_of rtt_ms link ",
        "json: the keys, in order", names);
}

// another Mac's GET /cluster body, which nothing authenticates: deep, large or not JSON is no answer, and no crash
void parse_checks() {
  MacInfo m = server("Studio", "qwen3.8-flash-next:4bit", true, 4, 58);
  const auto ok = parse_cluster(to_json(m).dump());
  check(ok && ok->name == "Studio" && ok->whole, "parse_cluster: a server's body");
  std::string deep = R"({"proto":1,"x":)";
  for (int i = 0; i < 20000; ++i) deep += "[";
  for (int i = 0; i < 20000; ++i) deep += "]";
  deep += "}";
  check(!parse_cluster(deep), "parse_cluster: nested 20,000 deep: no answer (and no crash)");
  std::string nine = R"({"proto":1,"x":[[[[[[[[1]]]]]]]]})", eight = R"({"proto":1,"x":[[[[[[[1]]]]]]]})";
  check(!parse_cluster(nine) && parse_cluster(eight), "parse_cluster: at most 8 levels");
  const std::string big = R"({"proto":1,"name":")" + std::string(70000, 'a') + R"("})";
  check(!parse_cluster(big), "parse_cluster: past 64 KiB: no answer");
  check(!parse_cluster("not json") && !parse_cluster("[1,2]") && !parse_cluster(""), "parse_cluster: not an object");
}

void pick_checks() {
  const std::string fn = "qwen3.8-flash-next:4bit";
  std::vector<MacInfo> p = {server("Far", fn, true, 9, 70), server("Near", fn, true, 2, 50)};
  check(picked(p, fn, true) == "(none)", "pick: nowhere when this Mac holds the model whole");
  check(picked(p, "") == "(none)", "pick: nowhere for no model");
  check(picked(p, fn) == "Near", "pick: the lowest round trip first");
  p[0].rtt_ms = 2;
  check(picked(p, fn) == "Far", "pick: then the fastest");
  check(picked(p, "Qwen3.8-Flash-Next:4BIT") == "Far", "pick: names compared without case");
  // the same variant first, else the most precise
  std::vector<MacInfo> v = {server("Four", "qwen3.8-flash-next:4bit", true, 1, 90),
                            server("Two", "qwen3.8-flash-next:q2_0", true, 30, 20)};
  check(picked(v, "qwen3.8-flash-next:q2_0") == "Two", "pick: the same variant before a closer one");
  std::vector<MacInfo> w = {server("Six", "qwen3.6-35b-a3b:6bit", true, 1, 90), server("Bf", "qwen3.6-35b-a3b:bf16", true, 20, 40),
                            server("Gguf", "qwen3.6-35b-a3b:16gb", true, 1, 99)};
  check(picked(w, "qwen3.6-35b-a3b:4bit") == "Bf", "pick: else the most precise variant (bf16 over 6-bit over 16gb)");
  check(picked(w, "qwen3.6-35b-a3b") == "Bf", "pick: a name without a variant takes any");
  // never: streaming, a proxy, not answering, another protocol, another model, nowhere to connect
  std::vector<MacInfo> no = {server("Streams", fn, false, 1, 99), server("Proxy", fn, true, 1, 99),
                             server("Silent", fn, true, -1, 99), server("Future", fn, true, 1, 99),
                             server("Other", "qwen3.8-27b:4bit", true, 1, 99), server("Nowhere", fn, true, 1, 99),
                             server("NoPort", fn, true, 1, 99)};
  no[1].routes_to = "Studio";
  no[3].proto = 2;
  no[5].host = "";
  no[6].port = 0;
  check(picked(no, fn) == "(none)", "pick: never a server that streams, sends requests on, is silent, speaks another "
                                    "protocol, runs another model or has no address");
  no.push_back(server("Unmeasured", fn, true, 0, 99));
  no.push_back(server("Measured", fn, true, 40, 1));
  check(picked(no, fn) == "Measured", "pick: an unmeasured round trip after a measured one");
  check(model_base("Qwen3.8-Flash-Next:4bit") == "qwen3.8-flash-next" && model_variant("Qwen3.8-Flash-Next:4bit") == "4bit" &&
            model_base("org/name") == "org/name" && model_variant("org/name").empty(),
        "model_base, model_variant");
}

// Two Macs that would each stream a model pair: the later started is the earlier one's stage
void pair_checks() {
  const std::string q2 = "qwen3.8-flash-next:q2_0";
  auto alone = [&](const std::string& name, double since, double rtt) {
    MacInfo m = server(name, q2, false, rtt, 20);
    m.since = since;
    return m;
  };
  auto with = [](const std::vector<MacInfo>& peers, const std::string& model) {
    const auto h = pair_with(peers, model);
    return h ? h->name : std::string("(none)");
  };
  const std::vector<MacInfo> two = {alone("Late", 200, 2), alone("Early", 100, 9)};
  check(with(two, q2) == "Early", "pair: with the server that started first");
  check(with(two, "Qwen3.8-Flash-Next:Q2_0") == "Early", "pair: names compared without case");
  check(with(two, "qwen3.8-flash-next:4bit") == "(none)", "pair: only with the same variant (a split reads the same files)");
  std::vector<MacInfo> no = {alone("Whole", 1, 1), alone("Proxy", 2, 1), alone("Stage", 3, 1), alone("Far", 4, 40),
                             alone("Silent", 5, -1), alone("Future", 6, 1)};
  no[0].whole = true;
  no[1].routes_to = "Studio";
  no[2].stage_port = 9330, no[2].stage_of = "Early-id";
  no[5].proto = 2;
  check(with(no, q2) == "(none)", "pair: never with a server that holds the model whole, sends its requests on, is a "
                                  "stage, is past 15 ms, is silent or speaks another protocol");
  // a head's stages: those paired with it, nearest first
  MacInfo a = alone("A", 1, 3), b = alone("B", 2, 1), c = alone("C", 3, 2);
  a.stage_port = 9331, a.stage_of = "Head-id";
  b.stage_port = 9332, b.stage_of = "Other-id";
  c.stage_port = 9333, c.stage_of = "Head-id";
  const auto st = stages_for({a, b, c}, q2, "Head-id");
  check(st.size() == 2 && st[0].name == "C" && st[1].name == "A", "pair: a head's stages, nearest first, not another's");
  check(stages_for({a, c}, "qwen3.8-flash-next:4bit", "Head-id").empty() && stages_for({a, c}, q2, "").empty(),
        "pair: no stages for another variant or no head");
  c.rtt_ms = 30;
  check(stages_for({a, c}, q2, "Head-id").size() == 1, "pair: not a stage past 15 ms");
  // of two that run it alone, the later gives way (the same second: the larger id)
  MacInfo me = alone("Me", 300.7, 0.2), first = alone("First", 100, 3), same = alone("Same", 300.2, 3);
  check(gives_way(me, {first}) && !gives_way(first, {me}), "pair: the later server gives way, the earlier stays");
  same.id = "aaa", me.id = "zzz";
  check(gives_way(me, {same}) && !gives_way(same, {me}), "pair: started the same second: the larger id gives way");
  check(!gives_way(me, {me}), "pair: not to itself");
  first.rtt_ms = 30;
  check(!gives_way(me, {first}), "pair: not over a long round trip");
}

void table_checks() {
  MacInfo here;
  here.name = "Yerbol's MacBook Pro";
  here.chip = "Apple M4 Pro";
  here.ram_gb = 48;
  here.link = "this Mac";
  MacInfo mine = server("Yerbol's MacBook Pro", "qwen3.5-4b:4bit", true, 0.2, 84, "this Mac");
  mine.chip = "Apple M4 Pro", mine.ram_gb = 48, mine.host = "127.0.0.1";
  MacInfo studio = server("Studio", "qwen3.8-flash-next:4bit", true, 4.2, 58);
  MacInfo mini = server("Mac mini", "qwen3.8-flash-next:4bit", false, 0.6, 0, "Ethernet");
  mini.chip = "Apple M4", mini.ram_gb = 16, mini.routes_to = "Studio";
  MacInfo air = server("Air", "qwen3.8-flash-next:q2_0", false, -1, 17);
  air.chip = "Apple M2", air.ram_gb = 24;
  MacInfo stage = server("Pro", "qwen3.8-flash-next:q2_0", false, 2, 0);
  stage.stage_port = 9330, stage.routes_to = "Studio";
  check(describe(studio) == "Studio (M3 Ultra, 512 GB)" && link_text(studio) == "Wi-Fi 4 ms" &&
            link_text(mini) == "Ethernet <1 ms" && link_text(mine) == "this Mac" && link_text(air) == "not answering",
        "describe, link_text", describe(studio) + " | " + link_text(studio) + " | " + link_text(mini));
  check(how_it_runs(studio) == "in memory" && how_it_runs(air) == "streams from SSD" && how_it_runs(mini) == "proxy" &&
            how_it_runs(stage) == "stage" && how_it_runs(MacInfo{}).empty(),
        "how_it_runs");
  check(plain(format_table(here, {stage}, 120)[2]).find("stage › Studio") != std::string::npos,
        "table: a stage, and the Mac it holds layers for", plain(format_table(here, {stage}, 120)[2]));
  const std::vector<std::string> t = format_table(here, {studio, air, mine, mini}, 120);
  std::string all;
  for (const auto& l : t) all += plain(l) + "\n";
  bool ok = t.size() == 5;
  const std::vector<std::string> want = {"Mac", "Yerbol's MacBook Pro", "Air", "Mac mini", "Studio"};
  for (size_t i = 0; ok && i < want.size(); ++i) ok = plain(t[i]).find(want[i]) != std::string::npos;
  check(ok, "table: a header, this Mac's server first, then the others by name", all);
  const std::string h = plain(t[0]), s = plain(t[4]);
  check(h.find("Chip · RAM") != std::string::npos && h.find("Model") != std::string::npos &&
            h.find("Runs") != std::string::npos && h.find("tok/s") != std::string::npos && h.find("Link") != std::string::npos,
        "table: the header names every column", h);
  check(s.find("M3 Ultra · 512 GB") != std::string::npos && s.find("qwen3.8-flash-next:4bit") != std::string::npos &&
            s.find("in memory") != std::string::npos && s.find(" 58 ") != std::string::npos &&
            s.find("Wi-Fi 4 ms") != std::string::npos,
        "table: a server's row", s);
  check(plain(t[3]).find("proxy › Studio") != std::string::npos && plain(t[3]).find("Ethernet <1 ms") != std::string::npos &&
            plain(t[2]).find("streams from SSD") != std::string::npos && plain(t[2]).find("not answering") != std::string::npos &&
            plain(t[1]).find("this Mac") != std::string::npos,
        "table: a proxy, a Mac that streams and does not answer, this Mac", all);
  // the columns line up: each column starts where its header does
  auto col = [](const std::string& line, const std::string& what) {
    const size_t at = line.find(what);
    return at == std::string::npos ? -1 : term::width(line.substr(0, at));
  };
  check(col(h, "Model") == col(s, "qwen3.8") && col(h, "Runs") == col(s, "in memory") &&
            col(h, "Link") == col(s, "Wi-Fi") && col(h, "Model") == col(plain(t[1]), "qwen3.5-4b"),
        "table: columns aligned", all);
  // no server here: this Mac all the same, first, not serving
  const auto u = format_table(here, {studio}, 120);
  check(u.size() == 3 && plain(u[1]).find("Yerbol's MacBook Pro") != std::string::npos &&
            plain(u[1]).find("not serving") != std::string::npos && plain(u[1]).find("M4 Pro · 48 GB") != std::string::npos,
        "table: this Mac without a server still comes first", plain(u[1]));
  // narrow terminals: nothing wider than asked, the chip column given up first
  for (int w : {100, 80, 64}) {
    const auto n = format_table(here, {studio, air, mine, mini}, w);
    int widest = 0;
    for (const auto& l : n) widest = std::max(widest, term::width(l));
    check(widest <= w, "table: at most " + std::to_string(w) + " columns", std::to_string(widest));
  }
  check(plain(format_table(here, {studio}, 64)[0]).find("Chip") == std::string::npos,
        "table: no chip column where it does not fit");
}

// On this Mac's network: a stand-in server advertised, then found by a browser
void live_checks() {
  httplib::Server srv;
  MacInfo me = this_mac();
  me.id = new_id();
  me.model = "cluster-test:4bit";
  me.whole = true;
  me.tok_s = 42;
  srv.Get("/cluster", [&](const httplib::Request&, httplib::Response& res) { res.set_content(to_json(me).dump(), "application/json"); });
  srv.Get("/health", [](const httplib::Request&, httplib::Response& res) { res.set_content(R"({"status":"ok"})", "application/json"); });
  const int port = srv.bind_to_any_port("0.0.0.0");
  me.port = port;
  std::thread th([&] { srv.listen_after_bind(); });
  check(port > 0 && !me.name.empty() && me.ram_gb > 0 && me.id.size() == 16, "live: this Mac and a stand-in server",
        to_json(me).dump());
  {
    Advertiser ad(port, [&] { return me; });
    Browser b;
    b.wait(3);
    const auto peers = b.peers();
    const MacInfo* found = nullptr;
    for (const auto& p : peers)
      if (p.id == me.id) found = &p;
    check(found && found->this_mac() && found->rtt_ms > 0 && found->model == me.model && found->port == port &&
              found->host == "127.0.0.1",
          "live: advertised, browsed, read and timed (link \"this Mac\")",
          "error: " + b.error() + " / " + ad.error() + "; found " + std::to_string(peers.size()) + ": " +
              (found ? to_json(*found).dump() : std::string("none")));
    const auto route = pick_server(peers, "cluster-test:4bit", false);
    check(route && route->id == me.id, "live: pick_server routes to it");
    printf("     lan address %s; link to it: %s\n", lan_address().c_str(), lan_address().empty() ? "-" : link_to(lan_address()).c_str());
  }
  // a peer that drips its body, one nested deeper than allowed, and one whose /health never ends: shown as not
  // answering within ~8 s, while a healthy server is read again all the same (its speed changed meanwhile)
  httplib::Server drip, deep, flood;
  MacInfo slow = me, nest = me, endless = me;
  slow.id = new_id(), nest.id = new_id(), endless.id = new_id();
  drip.Get("/cluster", [&](const httplib::Request&, httplib::Response& res) {
    res.set_chunked_content_provider("application/json", [&](size_t off, httplib::DataSink& sink) {
      std::this_thread::sleep_for(std::chrono::milliseconds(off == 0 ? 0 : 700));
      return sink.write(off == 0 ? "{" : " ", 1);
    });
  });
  deep.Get("/cluster", [&](const httplib::Request&, httplib::Response& res) {
    std::string b = R"({"proto":1,"x":)" + std::string(20000, '[') + std::string(20000, ']') + "}";
    res.set_content(b, "application/json");
  });
  flood.Get("/cluster", [&](const httplib::Request&, httplib::Response& res) {
    res.set_content(to_json(endless).dump(), "application/json");
  });
  flood.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
    res.set_chunked_content_provider("application/json", [&](size_t, httplib::DataSink& sink) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      return sink.write(std::string(1024, ' ').data(), 1024);
    });
  });
  for (httplib::Server* s : {&drip, &deep})
    s->Get("/health", [](const httplib::Request&, httplib::Response& res) { res.set_content("{}", "application/json"); });
  slow.port = drip.bind_to_any_port("0.0.0.0");
  nest.port = deep.bind_to_any_port("0.0.0.0");
  endless.port = flood.bind_to_any_port("0.0.0.0");
  std::thread td([&] { drip.listen_after_bind(); }), te([&] { deep.listen_after_bind(); }),
      tf([&] { flood.listen_after_bind(); });
  {
    Advertiser a1(port, [&] { return me; }), a2(slow.port, [&] { return slow; }), a3(nest.port, [&] { return nest; }),
        a4(endless.port, [&] { return endless; });
    Browser b;
    b.wait(3);
    me.tok_s = 77;  // read again within kReadEvery plus a read
    const double t0 = now_ms();
    bool fresh = false, slow_out = false, nest_out = false, flood_out = false;
    while (now_ms() - t0 < 12000 && !(fresh && slow_out && nest_out && flood_out)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
      for (const auto& p : b.peers()) {
        if (p.port == port && p.tok_s == 77) fresh = true;
        if (p.port == slow.port && p.rtt_ms < 0) slow_out = true;
        if (p.port == nest.port && p.rtt_ms < 0) nest_out = true;
        if (p.port == endless.port && p.rtt_ms < 0) flood_out = true;
      }
    }
    check(fresh, "live: a healthy server read again beside one that drips its body");
    check(slow_out && nest_out && flood_out,
          "live: a dripping, a too-deep and an endless-/health server shown as not answering");
  }
  srv.stop();
  drip.stop();
  deep.stop();
  flood.stop();
  th.join();
  td.join();
  te.join();
  tf.join();
}

}  // namespace

int main(int argc, char** argv) {
  const bool live = argc > 1 && !strcmp(argv[1], "--live");
  peer_checks();
  txt_checks();
  json_checks();
  parse_checks();
  pick_checks();
  pair_checks();
  table_checks();
  if (live) live_checks();
  printf("%d passed, %d failed\n", passes, fails);
  return fails ? 1 : 0;
}
