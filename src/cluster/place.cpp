#include "cluster/place.h"
#include "model/qwen4_exp.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <limits>

namespace ni::cluster {

Placement place(const std::vector<LayerCost>& layers, const LayerCost& head, int min_head,
                const std::vector<Member>& members) {
  Placement out;
  const int L = int(layers.size()), n = int(members.size());
  if (n < 1 || L < n || min_head > L - (n - 1)) return out;
  double slowest = 0;
  for (const Member& m : members)
    if (m.gbs > 0) slowest = slowest > 0 ? std::min(slowest, m.gbs) : m.gbs;
  if (slowest <= 0) slowest = 100;  // (no member's bandwidth known: any common value orders the runs the same way)
  // prefix sums of the layers' bytes
  std::vector<double> w(size_t(L) + 1, 0), r(size_t(L) + 1, 0);
  for (int i = 0; i < L; ++i) {
    w[size_t(i) + 1] = w[size_t(i)] + layers[size_t(i)].weights;
    r[size_t(i) + 1] = r[size_t(i)] + layers[size_t(i)].decode;
  }
  // best[i][k]: the shortest pass of members 0 .. i-1 holding layers [0, k), and of those the one whose least free
  // memory (slack) is largest, so that equal passes leave every Mac room for its caches; from[i][k]: where member
  // i-1's run starts
  constexpr double kNo = std::numeric_limits<double>::infinity();
  std::vector<std::vector<double>> best(size_t(n) + 1, std::vector<double>(size_t(L) + 1, kNo));
  std::vector<std::vector<double>> slack(size_t(n) + 1, std::vector<double>(size_t(L) + 1, kNo));
  std::vector<std::vector<int>> from(size_t(n) + 1, std::vector<int>(size_t(L) + 1, -1));
  best[0][0] = 0;
  for (int i = 0; i < n; ++i) {
    const Member& m = members[size_t(i)];
    const double gbs = m.gbs > 0 ? m.gbs : slowest;
    const double own_w = i == 0 ? head.weights : 0, own_r = i == 0 ? head.decode : 0;
    for (int a = 0; a < L; ++a) {
      if (best[size_t(i)][size_t(a)] == kNo) continue;
      // every later member keeps at least one layer; the head at least min_head
      for (int b = a + std::max(1, i == 0 ? min_head : 1); b <= L - (n - 1 - i); ++b) {
        const double held = own_w + w[size_t(b)] - w[size_t(a)];
        if (held > m.mem) break;  // (more layers only need more)
        const double t = best[size_t(i)][size_t(a)] + (own_r + r[size_t(b)] - r[size_t(a)]) / (gbs * 1e6);
        const double sl = std::min(slack[size_t(i)][size_t(a)], m.mem - held);
        double& bt = best[size_t(i) + 1][size_t(b)];
        double& bs = slack[size_t(i) + 1][size_t(b)];
        const double tol = 1e-9 * std::max(t, bt == kNo ? 0.0 : bt);
        if (t < bt - tol || (t <= bt + tol && sl > bs)) {
          bt = t;
          bs = sl;
          from[size_t(i) + 1][size_t(b)] = a;
        }
      }
    }
  }
  if (best[size_t(n)][size_t(L)] == kNo) return out;
  out.fits = true;
  out.ms = best[size_t(n)][size_t(L)];
  out.lo.assign(size_t(n) + 1, L);
  for (int i = n, k = L; i > 0; --i) {
    k = from[size_t(i)][size_t(k)];
    out.lo[size_t(i) - 1] = k;
  }
  for (int i = 0; i < n; ++i)
    out.held.push_back((i == 0 ? head.weights : 0) + w[size_t(out.lo[size_t(i) + 1])] - w[size_t(out.lo[size_t(i)])]);
  return out;
}

double whole_ms(const std::vector<LayerCost>& layers, const LayerCost& head, double gbs) {
  double r = head.decode;
  for (const LayerCost& l : layers) r += l.decode;
  return gbs > 0 ? r / (gbs * 1e6) : 0;
}

Placement plan_split(const std::string& dir, std::vector<Member> members, int chunk, bool may_stream) {
  Qwen4Exp::LayerBytes hb;
  const std::vector<Qwen4Exp::LayerBytes> lb = Qwen4Exp::layer_bytes(dir, &hb);
  const MemPlan mp = Qwen4Exp::plan(dir, chunk);  // the whole model's (no split set yet)
  // what a stage checks before it loads its share (stage.cpp's load_share): two slots' states on the linear-attention
  // layers, two 32K-token conversations' caches on the full-attention ones (the MTP head's with the head); the prefill
  // buffers, and half a GB beside a streamed pack's store
  int lin = 0, att = 0;
  for (const auto& l : lb) (l.linear ? lin : att) += 1;
  const int mtp = mp.mtp > 0 ? 1 : 0;
  const double kv = double(mp.kv_per_pos) * 65536 / std::max(att + mtp, 1);  // a full-attention layer's caches
  const double state = 2.0 * double(mp.slot_state) / std::max(lin, 1);       // a linear-attention layer's states
  std::vector<LayerCost> layers;
  for (const auto& l : lb) layers.push_back({l.weights + (l.linear ? state : kv), l.decode});
  const LayerCost head{hb.weights + mtp * kv, hb.decode};
  for (Member& m : members) m.mem -= double(mp.activations) + (mp.streamed ? 0.5e9 : 0);
  // the n-gram block's layer and the ones before it stay with the head (it reads the tokens)
  int min_head = 1;
  {
    std::ifstream f(dir + "/config.json");
    const nlohmann::json root = nlohmann::json::parse(f, nullptr, false);
    const nlohmann::json t = root.is_object() && root.contains("text_config") ? root["text_config"] : root;
    if (t.is_object() && t.contains("ple_layer_ids") && t["ple_layer_ids"].is_array() && !t["ple_layer_ids"].empty())
      min_head = std::max(1, t["ple_layer_ids"][0].get<int>());  // (one-indexed: the block before layer id - 1)
  }
  Placement p = place(layers, head, min_head, members);
  if (p.fits || !may_stream || !mp.streamed) return p;
  // the Macs do not hold the experts between them: each keeps its dense weights and a store of 3 layers' experts in
  // memory, and streams the rest of its experts from its SSD
  for (size_t i = 0; i < layers.size(); ++i) layers[i].weights -= lb[i].experts;
  const double store = 3 * Qwen4Exp::store_layer_bytes(dir);
  for (Member& m : members) m.mem -= store;
  p = place(layers, head, min_head, members);
  p.streams = p.fits;
  return p;
}

}  // namespace ni::cluster
