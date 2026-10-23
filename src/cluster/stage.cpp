// A stage of a split model (src/cluster/pipe.h): `kapitsa stage`'s server.
#include "cluster/pipe.h"
#include "cluster/wire.h"
#include "model/qwen4_exp.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <memory>
#include <unordered_map>

namespace ni::cluster {

using json = nlohmann::json;
using namespace wire;

namespace {

void free_seq(Qwen4Exp& m, Seq& s) {
  m.drop_kv(s);
  Gpu& g = Gpu::get();
  for (auto* v : {&s.conv, &s.rec, &s.pend_qkv, &s.pend_k, &s.pend_g, &s.pend_d})
    for (Buffer& b : *v) g.release(b);
  for (Buffer* b : {&s.tok, &s.pconv, &s.rope_pos, &s.pend_path}) g.release(*b);
}

void free_ckpt(Checkpoint& c) {
  Gpu& g = Gpu::get();
  for (auto* v : {&c.conv, &c.rec})
    for (Buffer& b : *v) g.release(b);
  g.release(c.pconv);
}

// The share of the model a stage holds: the layers it was started with, else the last ones a head gave it
struct Share {
  std::unique_ptr<Qwen4Exp> m;
  int lo = -1, hi = -1;
};

// Makes `sh` hold layers [lo, hi) of the model. A share that changes leaves the old one's buffers allocated (Qwen35
// does not release them, as with `kapitsa serve --allow-switch`): a stage given other layers again and again is better
// restarted. False with *why when the share does not fit this Mac.
bool load_share(Share& sh, const StageOptions& o, int lo, int hi, std::string* why) {
  if (sh.m && sh.lo == lo && sh.hi == hi) return true;
  Qwen4Exp::set_split({lo, hi, nullptr});
  const MemPlan mp = Qwen4Exp::plan(o.model_dir, o.chunk);
  const double keep = 2.0 * double(mp.slot_state) + double(mp.kv_per_pos) * 65536;  // two 32K-token conversations
  // (a share whose experts stream keeps half a GB beside its store: Qwen4Exp::init_store)
  const double need = double(mp.weights + mp.activations) + keep + (mp.streamed ? 0.5e9 : 0);
  if (need > o.mem) {
    char b[256];
    snprintf(b, sizeof b, "%s: layers %d-%d need %.1f GB here with two 32K-token conversations, and it allows %.1f GB",
             o.name.c_str(), lo + 1, hi, need / 1e9, o.mem / 1e9);
    *why = b;
    return false;
  }
  if (o.status)
    o.status("loading layers " + std::to_string(lo + 1) + "-" + std::to_string(hi) + " (" +
             std::to_string(int(mp.weights / 1e8) / 10.0).substr(0, 4) + " GB)");
  sh.m.reset();
  sh.m = std::make_unique<Qwen4Exp>(o.model_dir, o.chunk, size_t(std::max(1.0, o.mem - keep)));
  sh.m->set_mem_budget(size_t(o.mem));
  sh.lo = lo;
  sh.hi = hi;
  return true;
}

// One head's session: its handshake (a query, answered and done; or the ring's, with the layers to hold), then its
// messages until it leaves
void serve_head(Share& sh, const std::string& digest, int up, const std::string& peer, const StageOptions& o) {
  Header h;
  std::string body;
  if (!readable(up, 10000) || !read_all(up, &h, sizeof h) || h.magic != kMagic || h.op != kHello ||
      !read_payload(up, h, &body, 1 << 20))
    return;
  json hello = json::parse(body, nullptr, false);
  if (!hello.is_object()) return;
  const Qwen35Config c = Qwen35::read_config(o.model_dir);
  const json root = json::parse(std::ifstream(o.model_dir + "/config.json"), nullptr, false);
  const json tc = root.is_object() && root.contains("text_config") ? root["text_config"] : root;
  const size_t hc = size_t(tc.is_object() ? tc.value("hc_count", 4) : 4);
  const size_t xb = hc * c.hidden * 4, yb = size_t(c.hidden) * 4, ib = hc * 4;
  const std::string head = hello.value("head", peer);
  std::string err;
  if (h.flags & kFail) err = "-";  // an earlier stage refused: passed on as it is
  else if (hello.value("proto", 0) != kPipeProto) err = o.name + ": another version of Kapitsa (update both Macs)";
  else if (hello.value("digest", std::string()) != digest)
    err = o.name + ": its model files differ from the head's (the same model and variant on both Macs)";
  else if (hello.value("row_bytes", std::vector<size_t>{}) != std::vector<size_t>{xb, yb, ib})
    err = o.name + ": the rows' shapes differ";
  if (hello.value("query", false)) {  // what this stage is, on this connection, then done
    json info = {{"name", o.name}, {"mem", o.mem}, {"gbs", o.gbs}, {"rows", o.chunk}, {"fixed", o.lo >= 0}};
    if (o.lo >= 0) info["lo"] = o.lo, info["hi"] = o.hi;
    if (!err.empty()) h.flags |= kFail, info["error"] = err;
    const std::string b = info.dump();
    write_msg(up, h, b.data(), b.size());
    return;
  }
  const std::vector<std::string> chain = hello.value("chain", std::vector<std::string>{});
  const std::string ret = hello.value("ret", std::string());
  const bool last = chain.empty();
  // the layers this stage holds for this head: its own (--layers), else the first of the head's list
  json assign = hello.value("assign", json::array());
  if (err.empty()) {
    int lo = o.lo, hi = o.hi;
    if (lo < 0 && assign.is_array() && !assign.empty()) lo = assign[0][0], hi = assign[0][1];
    if (lo < 0) err = o.name + ": the head gave it no layers";
    else if (o.lo >= 0 && assign.is_array() && !assign.empty() && (assign[0][0] != o.lo || assign[0][1] != o.hi))
      err = o.name + ": started with layers " + std::to_string(o.lo + 1) + "-" + std::to_string(o.hi);
    else load_share(sh, o, lo, hi, &err);
  }
  // where this stage's messages go: the next stage, else the head (its return port, or back the way they came)
  int down = up;
  std::string cerr;
  if (!last) down = connect_to(chain[0], 5000, &cerr);
  else if (!ret.empty()) down = connect_to(ret, 5000, &cerr);
  if (down < 0) {
    logi("stage: %s", cerr.c_str());
    return;
  }
  if (err.empty()) {
    hello["stages"].push_back({{"name", o.name}, {"lo", sh.lo}, {"hi", sh.hi}, {"rows", sh.m->prefill_chunk()}});
    hello["chain"] = std::vector<std::string>(chain.begin() + (last ? 0 : 1), chain.end());
    if (assign.is_array() && !assign.empty()) assign.erase(assign.begin());
    hello["assign"] = assign;
  } else if (err != "-") {
    h.flags |= kFail;
    hello["error"] = err;
  }
  body = hello.dump();
  write_msg(down, h, body.data(), body.size());
  if (!err.empty()) {
    if (o.status) o.status(err == "-" ? "a head's split was refused before this Mac" : err);
    if (down != up) close(down);
    return;
  }
  Qwen4Exp& m = *sh.m;
  if (o.status)
    o.status("serving layers " + std::to_string(m.layer_lo() + 1) + "-" + std::to_string(m.layer_hi()) + " for " + head);

  Gpu& g = Gpu::get();
  std::unordered_map<int, std::unique_ptr<Seq>> seqs;
  std::unordered_map<int, Checkpoint> ckpts;
  long passes = 0, rows = 0;
  double gpu_ms = 0, t_report = now_ms();
  std::string payload;
  for (;;) {
    if (!read_all(up, &h, sizeof h) || h.magic != kMagic) break;
    auto seq = [&]() -> Seq* {
      auto it = seqs.find(h.seq);
      return it == seqs.end() ? nullptr : it->second.get();
    };
    if (is_pass(h.op)) {
      const size_t R = size_t(h.rows);
      Seq* s = seq();
      if (!s || R < 1 || R > size_t(m.prefill_chunk()) || h.bytes != R * (xb + yb + ib)) break;
      if (!read_all(up, m.residual().ptr(), R * xb) || !read_all(up, m.block_out().ptr(), R * yb) ||
          !read_all(up, m.injection().ptr(), R * ib))
        break;
      Remote::Pass p;
      p.kind = Remote::Kind(h.op - 1);
      p.seq = h.seq;
      p.pos = h.pos;
      p.rows = h.rows;
      p.n_pend = h.n_pend;
      p.snap = h.a;
      p.snap_row = h.b;
      auto ck = ckpts.find(h.a);
      auto e = g.begin();
      m.encode_stage(*e, *s, p, h.a >= 0 && ck != ckpts.end() ? &ck->second : nullptr);
      e->commit_and_wait();
      gpu_ms += e->gpu_ms();
      ++passes;
      rows += long(R);
      // on to the next stage, or back to the head (a prefill's streams alone, its injection applied)
      const bool x_only = last && h.op == kPrefill;
      h.bytes = R * (xb + (x_only ? 0 : yb + ib));
      iovec iov[4] = {{&h, sizeof h},
                      {m.residual().ptr(), R * xb},
                      {m.block_out().ptr(), R * yb},
                      {m.injection().ptr(), R * ib}};
      if (!write_all(down, iov, x_only ? 2 : 4)) break;
    } else {
      if (!read_payload(up, h, &payload)) break;
      Seq* s = seq();
      switch (h.op) {
        case kNewSeq:
          seqs[h.seq] = m.new_seq(h.a, 0);
          break;
        case kGrow:
          if (!s || !m.reserve(*s, h.a)) h.flags |= kFail;
          else h.b = std::min(h.b, s->cap);
          break;
        case kDropKv:
          if (s) m.drop_kv(*s);
          break;
        case kReset:
          if (s) m.reset(*s);
          break;
        case kNewCkpt:
          ckpts[h.a] = m.new_checkpoint();
          break;
        case kSave:
        case kRestore:
        case kFlush: {
          auto ck = ckpts.find(h.a);
          if (!s || (h.op != kFlush && ck == ckpts.end())) break;
          auto e = g.begin();
          if (h.op == kSave) m.save(*e, *s, ck->second, h.pos);
          else if (h.op == kRestore) m.restore(*e, *s, ck->second);
          else {
            s->n_pend = h.n_pend;
            m.encode_flush(*e, *s);
          }
          e->commit_and_wait();
          break;
        }
        case kRope:
          if (!s) break;
          s->rope_n = h.a;
          s->rope_delta = h.b;
          if (h.a > 0) {
            const size_t need = size_t(h.a) * 12;
            if (payload.size() != need) break;
            if (!s->rope_pos || s->rope_pos.size < need) {
              g.release(s->rope_pos);
              s->rope_pos = g.alloc((need + 49151) / 49152 * 49152, "rope_positions");
            }
            memcpy(s->rope_pos.ptr(), payload.data(), need);
          }
          break;
        default:  // kPing
          break;
      }
      // on with it (a rope table to the stages after this one; the head needs none of it back)
      if (last) payload.clear();
      if (!write_msg(down, h, payload.data(), payload.size())) break;
    }
    if (now_ms() - t_report > 60000 && passes) {
      logi("stage: %ld passes (%ld rows) in the last minute, %.1f ms of GPU a pass", passes, rows, gpu_ms / passes);
      passes = rows = 0;
      gpu_ms = 0;
      t_report = now_ms();
    }
  }
  for (auto& [id, s] : seqs) free_seq(m, *s);
  for (auto& [id, c] : ckpts) free_ckpt(c);
  if (down != up) close(down);
  if (o.status) o.status(head + " left; waiting for the next head");
}

}  // namespace

void run_stage(const StageOptions& o) {
  std::string why;
  Share sh;
  if (o.lo >= 0 && !load_share(sh, o, o.lo, o.hi, &why)) fatal("kapitsa stage: %s", why.c_str());
  const int lfd = listen_on(o.host, o.port, &why);
  if (lfd < 0) fatal("kapitsa stage: %s", why.c_str());
  const std::string digest = model_digest(o.model_dir);
  if (o.status)
    o.status("listening on port " + std::to_string(o.port) +
             (o.lo >= 0 ? "" : "; the head that connects says which layers to load"));
  for (;;) {
    sockaddr_storage a{};
    socklen_t len = sizeof a;
    const int up = accept(lfd, reinterpret_cast<sockaddr*>(&a), &len);
    if (up < 0) continue;
    tune(up);
    serve_head(sh, digest, up, peer_addr(a), o);
    close(up);
  }
}

}  // namespace ni::cluster
